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
from glob import glob

import time
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()

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
parser.add_argument('--color_lr', type=float, default=0.01)

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
            self.image_paths += glob(path_format)
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

    def _generate(self, image, mask, num_iter, num_strokes, width, attn_colors, path_dicts=None, use_tqdm=False):
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
            return [renderer.path_dict_np(radius=width) for renderer in renderers]

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

    def generate_for_batch(self, image, use_tqdm=False):
        foreground, background, mask, _ = utils.get_mask_u2net_batch(self.args, image, net=self.u2net, return_background=True)
        with torch.no_grad():
            mask_areas = mask.view(mask.size(0), -1).mean(dim=1).cpu().numpy().tolist()
        
        num_strokes_fg = self.args.num_strokes - self.args.num_background
        num_strokes_bg = self.args.num_background
        stroke_width_fg = self.args.width
        stroke_width_bg = self.args.width_bg

        path_dicts_fg = self._generate(foreground, mask, self.args.num_iter, num_strokes_fg, stroke_width_fg, False, use_tqdm=use_tqdm)
        path_dicts_fg = self._generate(foreground, mask, 0, 0, stroke_width_fg, True, path_dicts=path_dicts_fg, use_tqdm=use_tqdm)
        path_dicts_bg = self._generate(background, 1 - mask, 0, num_strokes_bg, stroke_width_bg, True, use_tqdm=use_tqdm)

    def generate_for_dataset(self, dataloader, use_tqdm=False, track_time=False):
        if track_time:
            start_time = time.time()

        for batch_index, (index, image) in enumerate(dataloader):
            print(f'Generating samples for {index.min().item()}..{index.max().item()} out of {len(dataloader.dataset)}:')
            image = image.to(self.args.device)
            self.generate_for_batch(image, use_tqdm=use_tqdm)

            if track_time:
                time_passed = time.time() - start_time
                # TODO

        if track_time:
            print(f'Took {time_passed:.02f}s to generate {len(dataloader.dataset)} samples.')


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

    # TODO: debug
    args.use_gpu = False
    args.no_cuda = True
    args.device = 'cpu'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    data_generator = DataGenerator(args)
    data_generator.generate_for_dataset(dataloader, use_tqdm=True, track_time=True)


if __name__ == '__main__':
    main()