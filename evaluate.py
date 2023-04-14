import os
import numpy as np
import time
from collections import OrderedDict
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import get_dataset
import argparser
from utils.shared import args
from utils.sketch_utils import *
from third_party.hog import HOGLayerMoreComplicated


class MergedModelWrapper(nn.Module):
    def __init__(self, model1, model2):
        super(MergedModelWrapper, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        return self.model1(x), self.model2(x)

    def get_representation(self, x, rep_type=None):
        z1 = self.model1.get_representation(x, rep_type)
        z2 = self.model2.get_representation(x, rep_type)

        return torch.cat([z1, z2], dim=-1)


class RepLabelPair(Dataset):
    def __init__(self, model, img_loader, label_fn_dict, rep_type='LBS+'):
        device = args.device

        self.zs = list()
        self.labels = {task: list() for task in label_fn_dict.keys()}

        for img, label in tqdm(img_loader):
            for task, label_fn in label_fn_dict.items():
                label_ = label_fn(label).to(device)
                self.labels[task].append(label_)

            if isinstance(img, list):               # for paired dataset (img_q, ...)
                img = img[0]

            with torch.no_grad():
                z = model.get_representation(img.to(device), rep_type=rep_type)

            self.zs.append(z)

        self.zs = torch.cat(self.zs, dim=0)
        for task, label_list in self.labels.items():
            self.labels[task] = torch.cat(label_list, dim=0)

    def __getitem__(self, index):
        label_dict = {task: label[index] for task, label in self.labels.items()}
        return self.zs[index], label_dict

    def __len__(self):
        return self.zs.shape[0]
    
    def dim_len(self):
        return self.zs.shape[1]


class LinearProbe():
    def __init__(self, data, tasks, model):
        self.model = model
        self.device = args.device
        self.data = data
        self.tasks = tasks

        self.task_bin = OrderedDict()
        self.label_fn_dict = OrderedDict()

        task_dict = self.set_task_dict(data)

        for task in tasks:
            assert task in task_dict 

            label_fn, num = task_dict[task]

            self.task_bin[task] = num
            self.label_fn_dict[task] = label_fn

        _, _, train_set_, test_set_, _, _ = get_dataset(data, args.data_root, eval_only=True)
        train_loader_ = DataLoader(train_set_, shuffle=False, pin_memory=True, 
                                        batch_size=args.eval_batch_size, drop_last=False, num_workers=8)
        test_loader_ = DataLoader(test_set_, shuffle=False, pin_memory=True,
                                       batch_size=args.eval_batch_size, drop_last=False, num_workers=8)

        self.train_set = RepLabelPair(model, train_loader_, self.label_fn_dict, args.rep_type)
        self.test_set = RepLabelPair(model, test_loader_, self.label_fn_dict, args.rep_type)

        self.train_loader = DataLoader(self.train_set, shuffle=True, pin_memory=False, batch_size=args.eval_batch_size, drop_last=True)
        self.test_loader = DataLoader(self.test_set, shuffle=False, pin_memory=False, batch_size=args.eval_batch_size, drop_last=True)

        self.critic = torch.nn.CrossEntropyLoss()

        self.momentum = args.eval_momentum
        self.weight_decay = args.eval_weight_decay
        self.lr_decay_rate = args.eval_lr_decay_rate
        self.cosine = args.eval_cosine
        self.warm = args.eval_warm
        self.epochs = args.eval_epochs
        self.lr_list = np.array(args.eval_lr_cand)

        iterations = args.eval_lr_decay_epochs.split(',')
        self.lr_decay_epochs = list([])
        for it in iterations:
            self.lr_decay_epochs.append(int(it))

        self.two_layer = False

    def set_task_dict(self, data):
        if data =='stl10':
            task_dict = {
                'class': (lambda l: l, 10),
            }
        elif data.startswith('geoclidean_elements'):
            task_dict = {
                'class': (lambda l: l, 17),
            }
        elif data.startswith('geoclidean_constraints'):
            task_dict = {
                'class': (lambda l: l, 20),
            }
        elif data.startswith('mnist'):
            task_dict = {
                'class': (lambda l: l[0], 10),
            }
        elif data.startswith('transmnist'):     # TODO
            task_dict = {
                'class': (lambda l: l, data.count('_'))
            }
        elif data.startswith('clevr'):
            pos_to_idx = {
                'rightmost': lambda l: (l[1] - (l[1][:, :, 6] == 0).unsqueeze(-1)*100)[:, :, 4].argmax(dim=1),
                'leftmost': lambda l: (l[1] + (l[1][:, :, 6] == 0).unsqueeze(-1)*100)[:, :, 4].argmin(dim=1),
                'topmost': lambda l: (l[1] + (l[1][:, :, 6] == 0).unsqueeze(-1)*100)[:, :, 5].argmin(dim=1),
                'bottommost': lambda l: (l[1] - (l[1][:, :, 6] == 0).unsqueeze(-1)*100)[:, :, 5].argmax(dim=1),
            }

            task_dict = OrderedDict()
            for position in ['rightmost', 'leftmost', 'topmost', 'bottommost']:
                task_dict[f'{position}_color'] = (lambda l: l[0][torch.arange(l[0].shape[0]), pos_to_idx[position](l), 0], 8)
                task_dict[f'{position}_size'] = (lambda l: l[0][torch.arange(l[0].shape[0]), pos_to_idx[position](l), 1], 2)
                task_dict[f'{position}_shape'] = (lambda l: l[0][torch.arange(l[0].shape[0]), pos_to_idx[position](l), 2], 3)
                task_dict[f'{position}_material'] = (lambda l: l[0][torch.arange(l[0].shape[0]), pos_to_idx[position](l), 3], 2)
                # task_dict[f'{position}_pos'] = (lambda l: l[1][torch.arange(l[1].shape[0]), pos_to_idx[position](l)][:, [4, 5]], 2)

            def shift_right_object(l, shift):
                right_idx = pos_to_idx['rightmost'](l)
                batch_range = torch.arange(l[1].shape[0])

                shifted_l = l[1].clone()
                shifted_l[batch_range, right_idx, 4] -= shift
                shifted_right_idx = pos_to_idx['rightmost']((l[0], shifted_l))

                return l[0][batch_range, shifted_right_idx, 0]
            
            task_dict['rightmost_shift'] = (lambda l: shift_right_object(l, 0.3), 8)

            thrid_rightmost = lambda l: (l[1] - (l[1][:, :, 6] == 0).unsqueeze(-1)*100)[:, :, 4].topk(k=3, dim=1)[1][:, -1]
            task_dict['rightmost_third'] = (lambda l: l[0][torch.arange(l[0].shape[0]), thrid_rightmost(l), 0], 8)

        return task_dict
        
    def warmup_learning_rate(self, warm_epochs, warmup_from, warmup_to, epoch, batch_id, total_batches, optimizer):
        if self.warm and epoch <= warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (warm_epochs * total_batches)
            lr = warmup_from + p * (warmup_to - warmup_from)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def adjust_learning_rate(self, lr, optimizer, epoch):
        if self.cosine:
            eta_min = lr * (self.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / self.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
            if steps > 0:
                lr = lr * (self.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def eval_task(self):
        print(f'evaluating {self.data}, {self.tasks}')
        tic = time.time()
        max_acc = {task: 0 for task in self.tasks}

        for lr in self.lr_list:
            z_size = self.train_set.dim_len()
            total_num = sum(self.task_bin.values())
            if self.two_layer:
                print('using 2 layer classifier')
                linear = nn.Sequential(
                    nn.Linear(z_size, z_size),
                    nn.ReLU(),
                    nn.Linear(z_size, total_num),
                ).to(self.device)
                
            else:
                linear = nn.Linear(z_size, total_num).to(self.device)

            optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

            warmup_from = 0.0
            warm_epochs = self.epochs//20
            if self.cosine:
                eta_min = lr * (self.lr_decay_rate ** 3)
                warmup_to = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * warm_epochs / self.epochs)) / 2
            else:
                warmup_to = lr

            for e in range(1, self.epochs+1):
                self.adjust_learning_rate(lr, optimizer, e)
                ### train probe
                linear.train()

                for idx, (z, label) in enumerate(self.train_loader):
                    self.warmup_learning_rate(warm_epochs, warmup_from, warmup_to, e, idx, len(self.train_loader), optimizer)
                    z = linear(z.detach()).split(tuple(self.task_bin.values()), dim=1)
                    
                    loss = 0
                    for idx, task in enumerate(self.tasks):
                        pred = z[idx]
                        loss += self.critic(pred, label[task])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                log = f'lr {lr}, epoch {e}, train loss: {loss.item():.3f} | '

                ### eval
                linear.eval()
                acc = OrderedDict({task: 0 for task in self.tasks})
                with torch.no_grad():
                    for z, label in self.test_loader:
                        z = linear(z).split(tuple(self.task_bin.values()), dim=1)
                        for idx, task in enumerate(self.tasks):
                            pred = z[idx]
                            acc[task] += (pred.argmax(dim=1) == label[task]).sum().item()

                    for task in self.tasks:
                        acc[task] = acc[task] / len(self.test_set) * 100
                        max_acc[task] = max(acc[task], max_acc[task])
                        log += f'{task}: {acc[task]:.2f}%, '

                print(log, end='\r')

        result = max_acc
        result_log = f'\nevaluation results for {self.data}\n'
        for task, acc in max_acc.items():
            result_log += f'{task}: {acc:.3f}%, '
        print(f'{result_log}\nelapsed time: {time.time()-tic:.2f}s')
        return result


def eval_sketch(model, tasks):
    for task in tasks:
        if task == 'retrieval':
            _, _, _, test_set, _, _ = get_dataset(args.dataset, args.data_root, eval_only=True)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
            results.update(eval_shoe(model, test_loader))
            del tasks[task]
            continue

    probe = LinearProbe(args.dataset, tasks, model)
    results = probe.eval_task()
    return results

def eval_shoe(model, test_loader):
        image_feature_all = []
        image_name = []
        sketch_feature_all = []
        sketch_name = []
        start_time = time.time()

        for (img, mask, _, _), _, labels in test_loader:
            positive_feature = model.get_representation(img.to(args.device), rep_type=args.rep_type)
            sketch_feature = model.get_representation(mask.to(args.device), rep_type=args.rep_type)
            
            sketch_feature_all.append(sketch_feature)
            sketch_name.append(labels)
            image_feature_all.append(positive_feature)
            image_name.append(labels)
            # for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            #     if positive_name not in Image_Name:
            #         Image_Name.append(sanpled_batch['positive_path'][i_num])
            #         Image_Feature_ALL.append(positive_feature[i_num])

        sketch_name = torch.cat(sketch_name, dim=0)
        sketch_feature_all = torch.cat(sketch_feature_all, dim=0)
        image_name = torch.cat(image_name, dim=0)
        image_feature_all = torch.cat(image_feature_all, dim=0)

        rank = torch.zeros(len(sketch_name))

        for num, sketch_feature in enumerate(sketch_feature_all):
            position_query = num

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), image_feature_all)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  image_feature_all[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        print('Time:{}'.format(time.time() - start_time))
        print(f'{top1}, {top10}')
        return {'retrieval_top1': top1, 'retrieval_top10': top10}


def load_model_with_args(eval_args):
    save_path = None

    if eval_args.baseline == 'baseline':
        assert eval_args.dataset != ''
        model = load_baseline(eval_args)

    elif eval_args.baseline == 'btcvae':
        model = load_btcvae(eval_args)

    elif eval_args.baseline == 'geossl':
        model = load_geossl(eval_args)

    elif eval_args.baseline == 'ltd':
        assert eval_args.dataset != ''
        model = load_ltd(eval_args)

    elif eval_args.baseline == 'paint':
        assert eval_args.dataset != ''
        model = load_painter(eval_args)
        save_path = os.path.join(eval_args.path, f'result_{eval_args.dataset}.txt')

    elif eval_args.baseline == 'hog':
        assert eval_args.dataset != ''
        mean_in = False
        if 'geoclidean' in eval_args.dataset or 'mnist' in eval_args.dataset:
            mean_in = True
        model = HOGLayerMoreComplicated(mean_in=mean_in)
        save_path = os.path.join(eval_args.path, f'result_{eval_args.dataset}.txt')

    elif eval_args.baseline == 'hog_cnn':
        assert eval_args.dataset != ''
        mean_in = False
        if 'geoclidean' in eval_args.dataset or 'mnist' in eval_args.dataset:
            mean_in = True
        hog_model = HOGLayerMoreComplicated(mean_in=mean_in)
        cnn_model = load_baseline(eval_args)
        model = MergedModelWrapper(hog_model, cnn_model)
        save_path = os.path.join('logs/hog', f'result_cnn_{eval_args.dataset}.txt')


    elif eval_args.baseline == 'clip':
        assert eval_args.dataset != ''
        model = load_clip(eval_args)
        save_path = os.path.join(eval_args.path, f'result_{eval_args.dataset}.txt')

    else:
        model = load_model(eval_args)
        if eval_args.rep_type == 'as_train':
            eval_args.rep_type = args.rep_type
    if eval_args.dataset == '':
        eval_args.dataset = args.dataset
    return save_path, model

def set_tasks_from_dataset(eval_args):
    if args.dataset.startswith('clevr'):
        if args.dataset == 'clevr':
            tasks = ['rightmost_color', 'leftmost_color', 'bottommost_color']
            tasks += ['rightmost_size', 'rightmost_shape', 'rightmost_material']
        else:
            cond = args.dataset.split('_')[1]
            tasks = [f'{pos}_{cond}' for pos in ['rightmost', 'leftmost', 'topmost', 'bottommost']]
            tasks += ['rightmost_size', 'rightmost_shape', 'rightmost_material']
        tasks += ['rightmost_shift', 'rightmost_third']
    elif args.dataset.startswith('mnist'):
        tasks = ['class']
    elif args.dataset == 'shoe':
        tasks = ['retrieval']
    else:
        tasks = ['class']
    return tasks


if __name__ == "__main__":
    parser = ArgumentParser()

    # model dataset
    parser.add_argument('path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--rep_type', type=str, default='as_train')

    # other setting
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_to', type=str, default='')
    
    # baseline
    parser.add_argument('--baseline', type=str, choices=['ours', 'baseline', 'btcvae', 'geossl', 'ltd', 'paint', 'hog', 'clip', 'hog_cnn'], default='ours')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--method', type=str, default='supcon', choices=['supcon', 'simclr', 'moco', 'ce', 'btcvae', 'geossl'])

    eval_args = parser.parse_args()

    eval_args.device = 'cpu' if eval_args.no_cuda else 'cuda:0'
    save_path = os.path.join(os.path.dirname(eval_args.path), f'result_{eval_args.save_to}.txt')

    save_path_, model = load_model_with_args(eval_args)
    if save_path_ is not None:
        save_path = save_path_

    model.eval()
    model.to(eval_args.device)
    
    if eval_args.baseline != 'ours':
        update_args(argparser.parse_arguments())
    update_args(vars(eval_args))

    tasks = set_tasks_from_dataset(eval_args)
    
    result = eval_sketch(model, tasks)

    with open(save_path, 'a') as f:
        f.write(f'{eval_args.path}({eval_args.rep_type})- ')
        for k, v in result.items():
            f.write(f'{k}: {v:.3f} | ')
        f.write('\n')
