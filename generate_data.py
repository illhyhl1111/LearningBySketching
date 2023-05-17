import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from third_party.clipasso.models.painter_params import Painter, PainterOptimizer
from third_party.clipasso.models.loss import Loss
from third_party.clipasso import sketch_utils as utils
from argparser import parse_arguments
import argparse

import numpy as np
from PIL import Image

import os
import pickle
import time
import random
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--img_paths', type=str, nargs='+', help='image file-paths (with wildcards) to process.')
parser.add_argument('--key_steps', type=int, nargs='+',
                    default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--chunk', type=int, nargs=2, help='--chunk (num_chunks) (chunk_index)')

parser.add_argument('--width', type=float, default=1.5, help='foreground-stroke width')
parser.add_argument('--width_bg', type=float, default=8.0, help='background-stroke width')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1.0)

parser.add_argument('--no_tqdm', action='store_true')
parser.add_argument('--no_track_time', action='store_true')
parser.add_argument('--visualize', action='store_true')

# CLIPasso arguments
parser.add_argument("--target", type=str)
parser.add_argument("--path_svg", type=str, default="none")
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=500)
parser.add_argument("--num_stages", type=int, default=1)
parser.add_argument("--color_vars_threshold", type=float, default=0.0)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--control_points_per_seg", type=int, default=4)
parser.add_argument("--attention_init", type=int, default=1)
parser.add_argument("--saliency_model", type=str, default="clip")
parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
parser.add_argument("--xdog_intersec", type=int, default=0)
parser.add_argument("--mask_object_attention", type=int, default=0)
parser.add_argument("--softmax_temp", type=float, default=0.3)
parser.add_argument("--percep_loss", type=str, default="none")
parser.add_argument("--train_with_clip", type=int, default=0)
parser.add_argument("--clip_weight", type=float, default=0)
parser.add_argument("--start_clip", type=int, default=0)
parser.add_argument("--num_aug_clip", type=int, default=4)
parser.add_argument("--include_target_in_aug", type=int, default=0)
parser.add_argument("--augment_both", type=int, default=1)
parser.add_argument("--augemntations", type=str, default="affine")
parser.add_argument("--noise_thresh", type=float, default=0.5)
parser.add_argument("--force_sparse", type=float, default=1)
parser.add_argument("--clip_conv_loss", type=float, default=1)
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
parser.add_argument("--clip_model_name", type=str, default="RN101")
parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
parser.add_argument("--clip_text_guide", type=float, default=0)
parser.add_argument("--text_target", type=str, default="none")


class ImageDataset(Dataset):
    def __init__(self, path_formats, transform=None):
        self.path_formats = path_formats
        self.transform = transform

        self.image_paths = []
        for path_format in self.path_formats:
            self.image_paths += glob(path_format, recursive='**' in path_format)
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return index, image


class DataGenerator(nn.Module):
    def __init__(self, args):
        super(DataGenerator, self).__init__()
        
        self.args = args
        
        # Initialize the renderers.
        renderers = []
        renderers.append(Painter(
            args,
            args.num_strokes, args.num_segments,
            imsize=args.image_scale,
            device=args.device,
        ))
        for _ in range(args.batch_size - 1):
            renderers.append(Painter(
                args,
                args.num_strokes, args.num_segments,
                imsize=args.image_scale,
                device=args.device,
                clip_model=renderers[0].clip_model,
                clip_preprocess=renderers[0].clip_preprocess,
                dino_model=renderers[0].dino_model
            ))

        self.renderers = nn.ModuleList(renderers)
        self.criterion = Loss(args)
        self.u2net = utils.get_u2net(args)

    def save_sample_visualization(self, sample_name, sample_image, sample_paths):
        fig, axs = plt.subplots(1, len(sample_paths)+1, figsize=(3+3*len(sample_paths), 3))

        axs[0].set_title('image')
        axs[0].imshow(sample_image)

        t = np.linspace(0, 1, 10)
        for i, (step, step_paths) in enumerate(sample_paths.items()):
            curves = cubic_bezier(step_paths['pos'], t)
            axs[i+1].set_title(f'step {step}')
            for curve, color, width in zip(curves, step_paths['color'], step_paths['radius']):
                axs[i+1].plot(*curve.T[::-1], c=color)
            axs[i+1].set_ylim(1,-1)
            axs[i+1].tick_params(axis='both', which='major', labelsize=5)

        vis_filename = os.path.join(self.args.output_dir, f'vis/{sample_name}.jpg')
        fig.savefig(vis_filename)
        plt.close()

    def _generate(self, image, mask, num_iter, num_strokes, width, attn_colors, path_dicts=None, gradual_colors=True, use_tqdm=False):
        curr_batch_size = image.size(0)
        if path_dicts is None:
            path_dicts = [None] * curr_batch_size
        renderers = self.renderers[:curr_batch_size]

        for renderer, curr_image, curr_mask, path_dict in zip(renderers, image, mask, path_dicts):
            renderer.set_random_noise(0)
            renderer.init_image(
                target_im=curr_image.unsqueeze(0),
                mask=curr_mask.unsqueeze(0),
                stage=0,
                randomize_colors=False,
                attn_colors=attn_colors,
                attn_colors_stroke_sigma=5.0,
                path_dict=path_dict,
                new_num_strokes=num_strokes,
                new_width=width
            )

        if num_iter == 0:
            for renderer in renderers:
                for key_step in self.args.key_steps:
                    renderer.log_shapes(str(key_step))
            path_dicts = [renderer.path_dict_np(radius=width) for renderer in renderers]
            if gradual_colors:
                for sample_paths in path_dicts:
                    ts = np.linspace(0, 1, len(sample_paths))
                    for t, step_paths in zip(ts, sample_paths.values()):
                        step_paths['color'] *= t
            return path_dicts

        optimizer = PainterOptimizer(self.args, renderers)
        optimizer.init_optimizers()

        steps = range(num_iter)
        if use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            for renderer in renderers:
                renderer.set_random_noise(step)

            optimizer.zero_grad_()
            sketches = torch.cat([renderer.get_image().to(self.args.device) for renderer in renderers], dim=0)
            loss = sum(self.criterion(sketches, image.detach(), step, points_optim=optimizer).values()).mean()
            loss.backward()
            optimizer.step_(optimize_points=True, optimize_colors=False)

            if (step+1) in self.args.key_steps:
                for renderer in renderers:
                    renderer.log_shapes()
                    renderer.log_shapes(str(step+1))

        return [renderer.path_dict_np(radius=width) for renderer in renderers]

    def generate_for_batch(self, index, image, use_tqdm=False):
        sample_names = [f'{idx}_{self.args.seed}' for idx in index.tolist()]

        foreground, background, mask, _ = utils.get_mask_u2net_batch(self.args, image, net=self.u2net, return_background=True)
        with torch.no_grad():
            mask_areas = mask.view(mask.size(0), -1).mean(dim=1).tolist()
            mask_areas = dict(zip(sample_names, mask_areas))
        
        num_strokes_fg = self.args.num_strokes - self.args.num_background
        num_strokes_bg = self.args.num_background
        stroke_width_fg = self.args.width
        stroke_width_bg = self.args.width_bg

        path_dicts = self._generate(foreground, mask, self.args.num_iter, num_strokes_fg, stroke_width_fg, False, use_tqdm=use_tqdm)
        
        if not self.args.enable_color:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas

        color_dicts = self._generate(foreground, mask, 0, None, stroke_width_fg, True, path_dicts=path_dicts, use_tqdm=use_tqdm)
        for paths, colors in zip(path_dicts, color_dicts):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['color'] = colors[step]['color']

        if num_strokes_bg <= 0:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas
        
        path_dicts_bg = self._generate(background, 1 - mask, 0, num_strokes_bg, stroke_width_bg, True, use_tqdm=use_tqdm)
        for paths, paths_bg in zip(path_dicts, path_dicts_bg):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['pos'] = np.concatenate([paths[step]['pos'], paths_bg[step]['pos']], axis=0)
                paths[step]['color'] = np.concatenate([paths[step]['color'], paths_bg[step]['color']], axis=0)
                if 'radius' in paths[step]:
                    paths[step]['radius'] = np.concatenate([paths[step]['radius'], paths_bg[step]['radius']], axis=0)

        path_dicts = dict(zip(sample_names, path_dicts))
        return path_dicts, mask_areas
    
    def generate_for_dataset(self, dataloader, use_tqdm=False, track_time=False):
        path_dicts = {}
        mask_areas = {}

        if track_time:
            start_time = time.time()

        min_index = next(iter(dataloader.dataset.indices))
        max_index = next(iter(reversed(dataloader.dataset.indices)))

        for batch_index, (index, image) in enumerate(dataloader):
            if track_time:
                print(f'generating samples for {index.min().item()}..{index.max().item()} of {min_index}..{max_index}:')
            image = image.to(self.args.device)
            
            batch_path_dicts, batch_mask_areas = self.generate_for_batch(index, image, use_tqdm=use_tqdm)
            path_dicts.update(batch_path_dicts)
            mask_areas.update(batch_mask_areas)

            if self.args.visualize:
                for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
                    sample_image = image[i].detach().cpu().permute(1, 2, 0).numpy()
                    self.save_sample_visualization(sample_name, sample_image, sample_paths)

            if track_time:
                time_passed = time.time() - start_time
                # TODO

        if track_time:
            print(f'took {time_passed:.02f}s to generate {len(dataloader.dataset)} samples.')

        return path_dicts, mask_areas


def cubic_bezier(p, t):
    p = p.reshape(-1, 4, 1, 2)
    t = t.reshape(1, -1, 1)
    return ((1-t)**3)*p[:,0] + 3*((1-t)**2)*t*p[:,1] + 3*(1-t)*(t**2)*p[:,2] + (t**3)*p[:,3]


def get_dataset(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(args.img_paths, transform=transform)
    if args.chunk is not None:
        num_chunks, chunk_index = args.chunk
        chunk_size = int(np.ceil(len(dataset) / num_chunks))
        chunk_start = chunk_size * chunk_index
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        dataset = Subset(dataset, range(chunk_start, chunk_end))

    return dataset


def main(args=None):
    if args is None:
        args = parse_arguments()
        args.update(vars(parser.parse_args()))
    args.num_iter = max(args.key_steps)
    args.image_scale = args.image_size
    args.color_lr = 0.01

    # TODO: debug
    # args.use_gpu = False
    # args.no_cuda = True
    # args.device = 'cpu'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)

    data_generator = DataGenerator(args).to(args.device)
    path_dicts, mask_areas = data_generator.generate_for_dataset(dataloader, use_tqdm=not args.no_tqdm, track_time=not args.no_track_time)

    with open(os.path.join(args.output_dir, f'data_{args.seed}_{args.chunk[1]}.pkl'), 'wb') as file:
        pickle.dump(path_dicts, file)
    with open(os.path.join(args.output_dir, f'maskareas_{args.seed}_{args.chunk[1]}.pkl'), 'wb') as file:
        pickle.dump(mask_areas, file)

if __name__ == '__main__':
    main()