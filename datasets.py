import os
import numpy as np
import random
from PIL import Image
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle

from utils.sketch_utils import *


class SketchDataset(Dataset):
    def __init__(
        self, sketch_root, data_root, image_size, data_type="stl10", split="train+unlabeled", augment=True, label_fn=lambda x: x
    ):
        """Initialize SketchDataset with the original image dataseet and pre-generated ground truth sketch

        Args:
            sketch_root (path): Root directory of the pre-generated ground truth sketch data
            data_root (path): Root directory of the original image dataset
            image_size (int): Resolution of the image to return
            data_type (str, optional): The name of the image dataset, should be one of ['stl10', 'clevr', 'shoe']. Defaults to "stl10".
            split (str, optional): Split of the image dataset. Defaults to "train+unlabeled".
            augment (bool, optional): Use random augmentation. Defaults to True.
            label_fn (functional, optional): Functions that map a label from the original dataset to a target label. Defaults to identity function.

        Raises:
            NotImplementedError: Rasises if data_type not in ['stl10', 'clevr', 'shoe'].
        """
        if data_type in ["stl10", "clevr", "shoe"]:
            sketch_dir = os.path.join(sketch_root, f"path_{data_type}.pkl")
        else:
            raise NotImplementedError
        
        with open(sketch_dir, "rb") as f:
            paths_dict_ = pickle.load(f)
        self.paths_dict = {}
        self.data_type = data_type
        self.augment = augment

        for key, val in paths_dict_.items():        # key: [idx]_[seed]
            # discard if it does not contain information about both the initial stroke and L intermediate strokes.
            if len(val) != 9:
                continue

            # change
            data_idx, seed = key.split("_")
            data_idx = int(data_idx)
            seed = int(seed)
            if data_idx in self.paths_dict:
                self.paths_dict[data_idx][seed] = val
            else:
                self.paths_dict[data_idx] = {seed: val}

        # maps the index of Dataset to data_idx (index of the ground truth dataset)
        self.idx_to_key = sorted(self.paths_dict.keys())        

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if data_type == "stl10":
            img_dataset_ = datasets.STL10(data_root, transform=transform, download=True, split=split)
            self.size = len(self.idx_to_key)
            self.img_dataset = lambda idx: img_dataset_[idx][0]
            self.label_dataset = lambda idx: label_fn(img_dataset_[idx][1])

            mask_path = os.path.join(data_root, "stl10_binary", f"STL10_{split}_mask_{image_size}.pkl")

        elif data_type == "clevr":
            img_dataset_ = CLEVRDataset(data_root, image_size, "train")
            self.size = len(self.idx_to_key)
            self.img_dataset = lambda idx: img_dataset_[idx][0]
            self.label_dataset = lambda idx: label_fn(img_dataset_[idx][1])

            mask_path = os.path.join(data_root, "clevr", "images", f"CLEVR_{split}_mask_{image_size}.pkl")

        elif data_type == "shoe":
            with open(os.path.join(data_root, "ShoeV2", "ShoeV2_Coordinate"), "rb") as fp:
                coordinate = pickle.load(fp)

            idx_to_path = os.listdir(os.path.join(data_root, "ShoeV2", "photo"))
            idx_to_path.sort()

            img_dataset_ = datasets.ImageFolder(os.path.join(data_root, "ShoeV2"), transform=transform)
            self.img_dataset = lambda idx: img_dataset_[idx][0]
            self.label_dataset = lambda idx: idx                    # use the index of the dataset as the label

            assert split in ["train", "test"]
            path_split = [idx.split("/")[-1].split("_")[0] for idx in coordinate if split in idx]
            split_idx_to_path = list(dict.fromkeys(path_split))
            split_idx_to_idx = [idx_to_path.index(path + ".png") for path in split_idx_to_path]
            self.size = len(split_idx_to_idx)
            self.idx_to_key = split_idx_to_idx

            mask_path = os.path.join(data_root, "ShoeV2", f"SHOE_mask_{image_size}.pkl")

        else:       # something wrong
            raise NotImplementedError

        ### load image mask
        if os.path.exists(mask_path):
            with open(mask_path, "rb") as f:
                self.masked = pickle.load(f)

        #### For ease of implementation, we left the paired sketch of the shoe image (included in the dataset) as the masked image. 
        elif data_type == "shoe":
            print("loading QMUL-ShoeV2 sketch")
            self.save_shoe_sketch(mask_path, coordinate, idx_to_path)

        else:
            print("data masking")
            self.save_masked_img(data_root, image_size, data_type, split, mask_path)


    def save_shoe_sketch(self, mask_path, coordinate, idx_to_path):
        from third_party.rasterize import rasterize_Sketch

        self.masked = list()

        transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

        for idx in range(len(idx_to_path)):
            shoe_idx = idx_to_path[idx].split(".")[0]
            vector_xs = [val for key, val in coordinate.items() if shoe_idx in key]
            sketch_imgs = []
            for vector_x in vector_xs:
                sketch_img = rasterize_Sketch(vector_x)
                sketch_img = Image.fromarray(sketch_img).convert("RGB")
                sketch_imgs.append(transform(sketch_img))

            sketch_imgs = 1 - torch.stack(sketch_imgs, dim=0)
            self.masked.append(sketch_imgs)

        with open(mask_path, "wb") as f:
            pickle.dump(self.masked, f, protocol=pickle.HIGHEST_PROTOCOL)


    def save_masked_img(self, data_root, image_size, data_type, split, mask_path):
        self.masked = list()

        if data_type == "stl10":
            transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            mask_img_dataset = datasets.STL10(data_root, transform=transform, download=True, split=split)
        elif data_type == "clevr":
            mask_img_dataset = CLEVRDataset(data_root, 224, "train")
        else:       # something wrong
            raise NotImplementedError

        images = []
        for key in tqdm(self.idx_to_key):
            images.append(mask_img_dataset[key][0].cuda())
            if len(images) > 100:
                images = torch.stack(images, dim=0)
                mask = mask_image(images)
                self.masked.append(resize(mask, image_size, InterpolationMode.NEAREST).cpu())
                images = []

        if len(images) > 0:
            images = torch.stack(images, dim=0)
            mask = mask_image(images)
            self.masked.append(resize(mask, image_size, InterpolationMode.NEAREST).cpu())

        self.masked = torch.cat(self.masked, dim=0)

        with open(mask_path, "wb") as f:
            pickle.dump(self.masked, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):
        sketch_idx = self.idx_to_key[index]
        pos_list = []
        color_list = []

        val = self.paths_dict[sketch_idx][0]
        for idx in sorted(map(int, val.keys())):
            pos = torch.tensor(val[f"{idx}"]["pos"]).view(-1, 4, 2)
            color = torch.tensor(val[f"{idx}"]["color"])
            pos_list.append(pos)
            color_list.append(color)

        pos, color = torch.stack(pos_list, dim=0), torch.stack(color_list, dim=0)

        img = self.img_dataset(sketch_idx)
        label = self.label_dataset(sketch_idx)

        if self.data_type == "shoe":            # randomly return one of the sketch of the corresponding shoe image
            mask = random.choice(self.masked[sketch_idx])
        else:
            mask = self.masked[index]

        if self.augment:
            if self.data_type == "clevr":
                img_q, mask_q, pos_q, color_q = random_aug(img, mask, pos, color, min_crop_frac=0.9, flip_p=0, jitter_weak=True)
                img_k, mask_k, pos_k, color_k = random_aug(img, mask, pos, color, min_crop_frac=0.9, flip_p=0, jitter_weak=True)
            elif self.data_type == "shoe":
                img_q, mask_q, pos_q, color_q = random_aug(img, mask, pos, color, jitter_p=0)
                img_k, mask_k, pos_k, color_k = random_aug(img, mask, pos, color, jitter_p=0)
            else:
                img_q, mask_q, pos_q, color_q = random_aug(img, mask, pos, color)
                img_k, mask_k, pos_k, color_k = random_aug(img, mask, pos, color)

        else:
            img_q, mask_q, pos_q, color_q = img, mask, pos, color
            img_k, mask_k, pos_k, color_k = img, mask, pos, color
        return (img_q, mask_q, pos_q, color_q), (img_k, mask_k, pos_k, color_k), label


    def __len__(self):
        return self.size


class RotatedMNISTData(Dataset):
    def __init__(self, root, train, angles, additional_transform=None, sample_per_domain=100):
        self.angles = angles
        self.num_angles = len(angles)
        self.sample_per_domain = sample_per_domain

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0)),
                transforms.ToTensor(),
                lambda x: 1 - x,
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    lambda x: 1 - x,
            ])
        if additional_transform is not None:
            self.transform = additional_transform(self.transform)

        self.dataset = datasets.MNIST(root=root, download=True, train=train)

    def __getitem__(self, index):
        mnist_idx = index % self.sample_per_domain
        image, class_label = self.dataset[mnist_idx]

        angle_idx = index // self.sample_per_domain
        theta = self.angles[angle_idx]
        angle_label = torch.div(theta, 15, rounding_mode="floor").long()
        image = transforms.functional.rotate(image, theta, transforms.functional.InterpolationMode.BILINEAR)
        image = self.transform(image)

        return image, (class_label, angle_label)

    def __len__(self):
        return self.sample_per_domain * self.num_angles


class TransformedMNISTData(Dataset):
    def __init__(self, root, train, transform_options=[], samples_per_class=100):
        assert len(transform_options) > 0

        self.transform_options = transform_options
        self.samples_per_class = samples_per_class

        post_transform = [
            transforms.Resize((32, 32)),
        ]

        if train:
            post_transform += [
                transforms.RandomResizedCrop(size=32, scale=(0.7, 1.0)),
            ]

        post_transform += [transforms.ToTensor(), lambda x: 1 - x]

        self.post_transform = transforms.Compose(post_transform)

        self.dataset = datasets.MNIST(root=root, train=train, download=True)

    def __len__(self):
        return len(self.transform_options) * self.samples_per_class

    def __getitem__(self, index):
        image_index = index % self.samples_per_class
        transform_index = index // self.samples_per_class

        image, _ = self.dataset[image_index]
        transform = self.transform_options[transform_index]
        transformed_image = transform(image)

        image = self.post_transform(image)
        transformed_image = self.post_transform(transformed_image)

        return (image, transformed_image), transform_index


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class CLEVRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size,
        split="train",
        data_size=-1,
        cont_label_transform=None,
        sort_key=None,
        attributes=None,
        attribute_classes=None,
        max_object_count=None,
    ):
        assert split in ["train", "val"]

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.img_dataset = []
        split_to_size = {
            "train": 70000,
            "val": 10000,
        }
        self.width, self.height = 480, 320

        if data_size == -1:
            self.size = split_to_size[split]
        else:
            self.size = data_size

        pickle_path = os.path.join(root_dir, "clevr", "images", f"CLEVR_{split}_{image_size}.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                self.img_dataset = pickle.load(f)[: self.size]

        else:
            for idx in tqdm(range(self.size)):
                img = Image.open(
                    os.path.join(root_dir, "clevr", "images", split, f"CLEVR_{split}_{idx:06}.png")
                )
                img = img.convert("RGB")
                self.img_dataset.append(transform(img))

            with open(pickle_path, "wb") as f:
                pickle.dump(self.img_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        if attributes is None:
            attributes = ["color", "size", "rotation", "shape", "3d_coords", "material", "pixel_coords"]
        if attribute_classes is None:
            attribute_classes = {
                "color": ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"],
                "size": ["large", "small"],
                "shape": ["cube", "cylinder", "sphere"],
                "material": ["metal", "rubber"],
            }
        if max_object_count is None:
            max_object_count = 10

        self.split = split
        self.cont_label_transform = cont_label_transform
        self.sort_key = sort_key
        self.scene_labels = json.load(
            open(os.path.join(root_dir, f"clevr/scenes/CLEVR_{split}_scenes.json"))
        )["scenes"]
        self.attributes = attributes.copy()
        self.attribute_class_indices = {
            attribute: dict(zip(classes, range(len(classes)))) for attribute, classes in attribute_classes.items()
        }
        self.max_object_count = max_object_count

    def force_length(self, x, length):
        """force the # of the object label into length"""
        if x.size(0) > length:
            return x[:length]
        return torch.cat([x, torch.zeros_like(x[:1]).expand(length - x.size(0), *([-1] * (len(x.size()) - 1)))], dim=0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        scene_labels = self.scene_labels[index]
        image = self.img_dataset[index]

        disc_labels = []
        cont_labels = []
        for obj in scene_labels["objects"]:
            obj_disc_labels = []
            obj_cont_labels = []

            for attribute, value in obj.items():
                if attribute in self.attribute_class_indices:
                    obj_disc_labels.append(self.attribute_class_indices[attribute][value])
                elif type(value) == list:
                    obj_cont_labels += value
                else:
                    obj_cont_labels.append(value)

            disc_labels.append(obj_disc_labels)
            cont_labels.append(obj_cont_labels)
        disc_labels = np.array(disc_labels)
        cont_labels = np.array(cont_labels)

        if self.sort_key is not None:
            sort_keys = [self.sort_key(disc, cont) for disc, cont in zip(disc_labels, cont_labels)]
            indices = np.argsort(sort_keys)
            disc_labels = disc_labels[indices]
            cont_labels = cont_labels[indices]
        disc_labels = torch.from_numpy(disc_labels).to(torch.long)
        cont_labels = torch.from_numpy(cont_labels).to(torch.float)

        # ad-hoc: index dependent
        cont_labels[:, 0] = (cont_labels[:, 0] * (np.pi / 180) + np.pi) % (2 * np.pi) - np.pi
        cont_labels[:, -3] = (cont_labels[:, -3] / self.width) * 2 - 1
        cont_labels[:, -2] = (cont_labels[:, -2] / self.height) * 2 - 1
        cont_labels[:, -1] = (cont_labels[:, -1] / self.width) * 2 - 1  # height?

        if self.cont_label_transform is not None:
            cont_labels = self.cont_label_transform(cont_labels)

        disc_labels = self.force_length(disc_labels, self.max_object_count)
        cont_labels = self.force_length(cont_labels, self.max_object_count)

        return image, (disc_labels, cont_labels)  # ([color, size, shape, material], [rotation(1), 3d_coords(3), pixel_coords(3)])


class Geoclidean(Dataset):
    def __init__(self, root, data_type="constraints", split="train", data_per_class=500, transform=None):
        assert data_type in ["constraints", "elements"]
        assert split in ["train", "close", "far", "positive"]

        self.data_type = data_type
        self.split = split
        self.data_num = data_per_class
        self.transform = transform

        if data_type == "constraints":
            self.size = data_per_class * 20
        else:
            self.size = data_per_class * 17

        split_dir = "train" if split == "train" else "test"
        self.data_dir = os.path.join(root, "geoclidean-transfer", data_type, split_dir)
        self.idx_to_class = sorted(os.listdir(self.data_dir))

    def __getitem__(self, index):
        class_idx = index // self.data_num
        data_idx = index % self.data_num + 1

        image_dir = os.path.join(self.data_dir, self.idx_to_class[class_idx])
        if self.split == "train":
            file_name = f"in_{data_idx}_fin.png"
            image_dir = os.path.join(image_dir, file_name)
        elif self.split == "positive":
            file_name = f"in_{data_idx}_fin.png"
            image_dir = os.path.join(image_dir, self.split, file_name)
        else:
            file_name = f"out_{self.split}_{data_idx}_fin.png"
            image_dir = os.path.join(image_dir, self.split, file_name)

        img = Image.open(image_dir)
        img = img.convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, class_idx

    def __len__(self):
        return self.size


def get_dataset(data_type, data_root, sketch_root="gt_sketches", eval_only=False):
    """Get train, validation, test split of the dataset.
    Since the training dataset used to train the model is very small, 
    the training dataset used for linear probing is set aside as the original image dataset.

    Args:
        data_type (string): The name of the original dataset, should be one of 'stl10', 'shoe' or starts with 'mnist', 'transmnist', 'geoclidean' or 'clevr'.
                            formats of types starting with 'mnist': mnist_[train domain separated with comma]_[test domain]. ex) mnist_30,45_0,90
                            formats of types starting with 'transmnist': transmnist_[eye | rot{degree} | hflip | scale{factor}]*. ex) transmnist_rot90_hflip
                            formats of types starting with 'geoclidean': geoclidean_[type]_[split]. ex) geoclidean_elements_positive
                            formats of types starting with 'clevr': clevr_[task]?. ex) clevr_color
        data_root (path): Root directory of the original image dataset
        sketch_root (path, optional): Path of pre-generated ground truth sketch data. Defaults to "gt_sketches".
        eval_only (bool, optional): Returns splits for evaluation only. Defaults to False.

    Raises:
        AssertionError: Undefined arguments for data_type
        NotImplementedError: Unspecified data_type

    Returns:
        [type]: [description]
    """

    train_dataset, val_dataset, eval_train_dataset, eval_test_dataset = None, None, None, None

    if data_type == "stl10":
        image_shape = (3, 128, 128)
        class_num = 10

        if eval_only:
            train_dataset, val_dataset = None, None
        else:
            label_fn = lambda l: l

            train_dataset = SketchDataset(
                sketch_root,
                data_root,
                image_shape[1],
                label_fn=label_fn,
            )
            train_dataset, val_dataset = split_dataset(train_dataset, 0.1, use_stratify=False)

        transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
        ])
        eval_train_dataset = datasets.STL10(data_root, transform=transform, download=True, split="train")
        eval_test_dataset = datasets.STL10(data_root, transform=transform, download=True, split="test")

    elif data_type.startswith("clevr"):  # clevr_color
        image_shape = (3, 128, 128)
        class_num = -1

        if eval_only:
            train_dataset, val_dataset = None, None
        else:
            if len(data_type.split("_")) == 1:
                task = None
                label_fn = lambda l: l

            else:
                task = data_type.split("_")[1]
                task_dict = {
                    "rightmost_color": lambda l: l[0][l[1][l[1][:, 6] != 0][:, 4].argmax(dim=0), 0],
                    "rightmost_size": lambda l: l[0][l[1][l[1][:, 6] != 0][:, 4].argmax(dim=0), 1],
                    "rightmost_shape": lambda l: l[0][l[1][l[1][:, 6] != 0][:, 4].argmax(dim=0), 2],
                    "rightmost_material": lambda l: l[0][l[1][l[1][:, 6] != 0][:, 4].argmax(dim=0), 3],
                }
                assert f"rightmost_{task}" in task_dict, f"task {task} undefined for clevr"

                label_fn = lambda l: task_dict[f"rightmost_{task}"](l)

                class_num_dict = {
                    'color': 8,
                    'size': 2, 
                    'shape': 3, 
                    'material': 2, 
                }
                class_num = class_num_dict[task]

            train_dataset = SketchDataset(
                sketch_root,
                data_root,
                image_shape[1],
                data_type="clevr",
                label_fn=label_fn,
            )
            train_dataset, val_dataset = split_dataset(train_dataset, 0.2, use_stratify=False)

        eval_train_dataset = CLEVRDataset(data_root, image_shape[1], "train", data_size=10000)
        eval_test_dataset = CLEVRDataset(data_root, image_shape[1], "val", data_size=10000)

    elif data_type == "shoe":
        image_shape = (3, 128, 128)
        class_num = -1

        if eval_only:
            train_dataset = None
        else:
            train_dataset = SketchDataset(
                sketch_root,
                data_root,
                image_shape[1],
                data_type="shoe",
                augment=False,
                split="train",
            )
        val_dataset = SketchDataset(
            sketch_root,
            data_root,
            image_shape[1],
            data_type="shoe",
            augment=False,
            split="test",
        )

        eval_train_dataset = None
        eval_test_dataset = val_dataset

    elif data_type.startswith("transmnist"):      # only for test
        image_shape = (1, 32, 32)
        class_num = 1
        transform_args = data_type.split("_")[1:]

        debug_texts = []

        transform_options = []
        for transform_arg in transform_args:
            debug_texts.append([])

            transform_option = []
            for elem_arg in transform_arg.split(","):
                if elem_arg == "eye":
                    debug_texts[-1].append("do nothing.")
                    transform_option.append(lambda x: x)

                elif elem_arg.startswith("rot"):
                    angle = round(float(elem_arg.replace("rot", "")))
                    debug_texts[-1].append(f"rotate {angle} degrees.")
                    transform_option.append(lambda x, angle=angle: transforms.functional.rotate(x, angle))
                    class_num *= 2

                elif elem_arg == "hflip":
                    debug_texts[-1].append("flip horizontally.")
                    transform_option.append(lambda x: transforms.functional.hflip(x))
                    class_num *= 2

                elif elem_arg.startswith("scale"):
                    scale = float(elem_arg.replace("scale", ""))
                    debug_texts[-1].append(f"scale {scale * 100}%.")
                    transform_option.append(
                        lambda x, scale=scale: transforms.functional.affine(
                            x, angle=0, translate=(0, 0), scale=scale, shear=0, interpolation=InterpolationMode.BILINEAR
                        )
                    )
                else:
                    raise AssertionError(f'invalid arg {elem_arg} in transmnist')
            transform_options.append(transforms.Compose(transform_option))

        for index, lines in enumerate(debug_texts):
            print(f"option {index}:")
            for line in lines:
                print(f"  {line}")

        eval_train_dataset = TransformedMNISTData(data_root, True, transform_options=transform_options, samples_per_class=2500)
        eval_test_dataset = TransformedMNISTData(data_root, False, transform_options=transform_options, samples_per_class=10000)
    
    elif data_type.startswith("mnist"):  # rotated mnist, ex: mnist_30,45_0,90
        image_shape = (1, 32, 32)
        class_num = 10

        try:
            train_degrees, test_degrees = data_type.split("_")[1:]
            train_degrees = list(map(int, train_degrees.split(",")))
            test_degrees = list(map(int, test_degrees.split(",")))

            print(f"train_deg: {train_degrees}, test_deg: {test_degrees}")
        except Exception:
            raise AssertionError(f'invalid rotation degrees {train_degrees}_{test_degrees} in mnist')

        train_dataset = RotatedMNISTData(
            root=data_root, train=True, angles=train_degrees, additional_transform=TwoCropTransform, sample_per_domain=2500
        )
        train_dataset, val_dataset = split_dataset(train_dataset)
        eval_train_dataset = train_dataset
        eval_test_dataset = RotatedMNISTData(root=data_root, train=False, angles=test_degrees, sample_per_domain=10000)

    elif data_type.startswith("geoclidean"):  # geoclidean_elements_close
        image_shape = (1, 64, 64)

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            lambda x: 1 - x,
            transforms.RandomAffine(
                degrees=90, translate=(0.1, 0.1), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR
            ),
            lambda x: 1 - x,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        assert len(data_type.split("_")) == 3, f'requires a format such as geoclidean_[type]_[split] for geoclidean'
        _, data_type, test_split = data_type.split("_")

        if data_type == 'elements':
            class_num = 17
        else:
            class_num = 20

        train_dataset_ = Geoclidean(
            root=data_root, data_type=data_type, split="train", transform=TwoCropTransform(train_transform), data_per_class=500
        )
        train_dataset, val_dataset = split_dataset(train_dataset_, 0.98, use_stratify=True)        # 10 for training, 490 for validation
        eval_train_dataset = train_dataset_
        eval_test_dataset = Geoclidean(root=data_root, data_type=data_type, split=test_split, transform=test_transform)

    else:
        raise NotImplementedError

    return train_dataset, val_dataset, eval_train_dataset, eval_test_dataset, image_shape, class_num
