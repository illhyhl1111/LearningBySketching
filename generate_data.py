from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from third_party.clipasso.models.painter_params import Painter
from argparser import parse_arguments
import argparse

import numpy as np
from PIL import Image
from glob import glob

from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--img_paths', type=str, nargs='+')
parser.add_argument('--key_steps', type=int, nargs='+',
                    default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=10)

# CLIPasso arguments
parser.add_argument("--target", help="target image path")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--path_svg", type=str, default="none")
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=500)
parser.add_argument("--num_stages", type=int, default=1)
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--color_lr", type=float, default=0.01)
parser.add_argument("--color_vars_threshold", type=float, default=0.0)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--width", type=float, default=1.5)
parser.add_argument("--control_points_per_seg", type=int, default=4)
parser.add_argument("--attention_init", type=int, default=1)
parser.add_argument("--saliency_model", type=str, default="clip")
parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
parser.add_argument("--xdog_intersec", type=int, default=0)
parser.add_argument("--mask_object_attention", type=int, default=0)
parser.add_argument("--softmax_temp", type=float, default=0.3)
parser.add_argument("--augemntations", type=str, default="affine")
parser.add_argument("--noise_thresh", type=float, default=0.5)
parser.add_argument("--force_sparse", type=float, default=1)
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
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image


def initialize_renderers(args):
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

    return renderers


def main(args=None):
    if args is None:
        args = parse_arguments()
        args.update(vars(parser.parse_args()))
    args.image_scale = args.image_size
    
    np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(args.img_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # TODO: debug
    print(len(dataset))
    
    renderers = initialize_renderers(args)


if __name__ == '__main__':
    main()