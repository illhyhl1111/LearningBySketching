import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import get_dataset
from models.LBS import SketchModel
from models.clip_loss import Loss as CLIPLoss
from loss import LBS_loss_fn, l1_loss_fn
from file_writer import FileWriter
from evaluate import eval_sketch

import argparser
from utils.sketch_utils import *
from utils.shared import args
from utils.shared import stroke_config as config
from utils.shared import update_args, update_config

clip_loss_fn = None

def unpack_dataloader(datas, train_with_gt=True, use_mask=False):
    assert (len(datas) == 3 and train_with_gt) or (len(datas) == 2 and not train_with_gt)

    if train_with_gt:
        imgs_q, imgs_k, labels = datas
        img, mask, pos, color = imgs_q
        img_k, mask_k, pos_k, color_k = imgs_k

        if not use_mask:
            mask_ = torch.ones_like(mask)
        else:
            mask_ = mask

        return {
            "img": img.to(device, non_blocking=True),
            "fore": (img * mask_ + (1 - mask_)).to(device, non_blocking=True),
            "back": (img * (1 - mask_) + mask_).to(device, non_blocking=True),
            "pos": pos.to(device, non_blocking=True),                                           # [bs, 9, nL, 8]
            "color": color.to(device, non_blocking=True),                                       # [bs, 9, nL, 3]
            "mask": mask.to(device, non_blocking=True),

            "img_k": img_k.to(device, non_blocking=True),
            "pos_k": pos_k.to(device, non_blocking=True),                                       # [bs, 9, nL, 8]
            "color_k": color_k.to(device, non_blocking=True),                                   # [bs, 9, nL, 3]
            "mask_k": mask_k.to(device, non_blocking=True),

            "label": [label.to(device, non_blocking=True) for label in labels],
        }

    (img, img_k), label = datas
    if isinstance(label, list):
        label, _ = label  # rotated mnist: label, angle

    return {
        "img": img.to(device, non_blocking=True),
        "img_k": img_k.to(device, non_blocking=True),
        "label": label.to(device, non_blocking=True),
    }


def train(model, optimizer, scheduler, loaders, logger, train_with_gt=True):
    """Train a model with gt

    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        scheduler ([type]): [description]
        loaders ([type]): [description]
        logger ([type]): [description]
    """
    train_loader, val_loader, _, eval_test_loader = loaders

    image_grids = {}

    global clip_loss_fn
    clip_loss_fn = CLIPLoss()

    if train_with_gt:
        if args.dataset == "shoe":
            image_grids["recon"] = ImageGrid(num_img=10, nrow=4)
        else:
            image_grids["recon"] = ImageGrid(num_img=10, nrow=6)
        image_grids["sequential"] = ImageGrid(num_img=config.n_lines + 1, nrow=5)
    else:
        image_grids["recon"] = ImageGrid(num_img=8, nrow=2)
        image_grids["sequential"] = ImageGrid(num_img=config.n_lines + 1, nrow=5)

    best_loss = 1e8

    # for in range [1 ~ args.epoch], while epoch 0 is only for visualizing the initialized model.
    for epoch in range(args.start_epoch, args.epochs + 1):
        progress = epoch / args.epochs
        model.set_progress(progress)

        ### train
        if epoch != args.start_epoch:
            model.train()
            train_epoch(model, optimizer, scheduler, train_loader, logger, train_with_gt, epoch)

        ### validation
        logger.log_dirname(f"Epoch {epoch}")
        model.eval()

        if epoch % args.validate_every == 0:
            val_loss, imgs, sketches = validation(model, optimizer, val_loader, logger, train_with_gt, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                state_dict = model.state_dict()
                torch.save(state_dict, logger.basepath + "/model_best.pt")

            torch.save(state_dict, logger.basepath + "/model.pt")
            torch.save({
                "epoch": epoch,
                "optim": optimizer.state_dict(),
            }, logger.basepath + "/optim.pt")

            ### tensorboard log
            if train_with_gt:
                image_grid = image_grids.get("recon")
                plot_results_gt(model, logger, image_grid, imgs, sketches, epoch)

            else:
                image_grid = image_grids.get("recon")
                plot_results_l1(model, logger, image_grids, imgs, sketches, epoch)

            image_grid = image_grids.get("sequential")
            plot_sequential(model, logger, eval_test_loader, image_grid, epoch)

        ### evaluate with current model & save
        if epoch != 0 and epoch % args.evaluate_every == 0:
            if not args.no_eval:
                eval_task(model, logger, epoch)

            state_dict = model.state_dict()
            torch.save(state_dict, logger.basepath + f"/model_{epoch}.pt")


def eval_task(model, logger, epoch):
    if args.dataset.startswith("clevr"):
        tasks = ["rightmost_color", "rightmost_size", "rightmost_shape", "rightmost_material"]
    elif args.dataset == "shoe":
        tasks = ["retrieval"]
    else:
        tasks = ["class"]
    sketches = eval_sketch(model, tasks)

    for task, acc in sketches.items():
        logger.scalar_summary(f"eval_{task}", acc, epoch)


def plot_sequential(model, logger, eval_test_loader, image_grid, epoch):
    process = model.eval_grid(eval_test_loader.dataset)
    process_grid = image_grid.update(process)
    logger.figure_summary("validation_process", process_grid, epoch)


def plot_results_l1(model, logger, image_grid, inputs, sketches, epoch):
    img = inputs["img"]
    sketch = sketches["sketch"]

    stacked_results = torch.stack([img[:8], sketch[:8]], dim=1).flatten(0, 1)

    img_grid = image_grid.update(stacked_results)
    logger.figure_summary("reconstruction", img_grid, epoch)


def plot_results_gt(model, logger, image_grid, inputs, sketches, epoch):
    gt_pos = inputs['pos']
    gt_color = inputs['color']
    pos_final = gt_pos[:, -1]
    color_final = gt_color[:, -1]

                        ##### pad background strokes with dummy stroke
    num_gt_foreground = pos_final.shape[1]
    num_dummy = config.n_lines - num_gt_foreground
    padded_stroke = {
        "position": torch.cat([pos_final, torch.ones_like(pos_final)[:, :num_dummy]], dim=1).to(args.device),
        "color": torch.cat([color_final, torch.ones_like(color_final)[:, :num_dummy]], dim=1).to(args.device),
    }
    gt_sketch = model.rasterize_stroke(padded_stroke, 'color')

    if args.dataset == "shoe":
        stacked_results = torch.stack([
                        inputs["img"], inputs["mask"], gt_sketch, sketches["sketch_color"],
                        ], dim=1).flatten(0, 1)
    else:
        stacked_results = torch.stack([
                        inputs["img"], inputs["back"], gt_sketch,
                        sketches["sketch_color"], sketches["sketch_black"], sketches["sketch_background"],
                        ], dim=1).flatten(0, 1)

    img_grid = image_grid.update(stacked_results)
    logger.figure_summary("reconstruction", img_grid, epoch)   


def train_epoch(model, optimizer, scheduler, train_loader, logger, train_with_gt, epoch):
    loss_dict = {}
    for idx, datas in enumerate(train_loader):
        steps = epoch * len(train_loader) + idx

        inputs = unpack_dataloader(datas, train_with_gt)

        if train_with_gt:
            _, loss = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=True)
        else:
            _, loss = l1_loss_fn(model, optimizer, inputs, train_model=True)
        loss_dict.update(loss)

        loss_dict["lr"] = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if idx % args.print_every == 0:
            logger.log(
                f"[Epoch {epoch:3d} iter {idx:4d}] \
                    [{loss_dict['loss_total'].item():.3f}, {loss_dict['accuracy'].item():.2f}]"
            )

            for name, values in loss_dict.items():
                logger.scalar_summary(name, values, steps)


def validation(model, optimizer, val_loader, logger, train_with_gt, epoch):
    val_loss = 0
    val_acc = 0
    val_count = 0

    for idx, datas in enumerate(val_loader):
        if idx == 20:  # validate for 20 steps
            break

        inputs = unpack_dataloader(datas, device, train_with_gt)
        batch_size = inputs["img"].shape[0]

        with torch.no_grad():
            if train_with_gt:
                sketches, val_losses = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=False)
            else:
                sketches, val_losses = l1_loss_fn(model, optimizer, inputs, train_model=False)

            val_loss += val_losses["loss_total"]
            val_acc += val_losses["accuracy"] * batch_size
            val_count += batch_size

    val_loss /= idx
    val_acc /= val_count

    logger.scalar_summary("val_loss", val_loss, epoch)
    logger.scalar_summary("val_acc", val_acc, epoch)
    logger.log(f"[Epoch {epoch:3d}] Val: [{val_loss.item():.3f}, {val_acc.item():.2f}]")

    return val_loss, inputs, sketches


def main():
    args = argparser.parse_arguments()

    train_set, val_set, eval_train_set, eval_test_set, image_shape, class_num = get_dataset(args.dataset, data_root=args.data_root)

    args.image_size = image_shape[1]
    args.image_num_channel = image_shape[0]
    args.class_num = class_num
    stroke_config = argparser.get_stroke_config(args)

    update_args(args)
    update_config(stroke_config)

    global device
    device = args.device

    train_loader = DataLoader(train_set, shuffle=True, num_workers=16, pin_memory=True, batch_size=args.batch)
    val_loader = DataLoader(val_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=args.batch)
    eval_train_loader = DataLoader(eval_train_set, shuffle=False, num_workers=8, pin_memory=True, batch_size=256)
    eval_test_loader = DataLoader(eval_test_set, shuffle=False, num_workers=8, pin_memory=True, batch_size=256)

    model = SketchModel()
    model = model.to(device)

    t_0 = args.epochs * len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=args.lr, T_0=t_0, T_mult=1, T_up=t_0 // 20, gamma=0.5)

    if args.load_path is not None:
        checkpoint = torch.load(os.path.join(args.load_path, "enc.pt"))
        model.load_state_dict(checkpoint)
        checkpoint_opt = torch.load(os.path.join(args.load_path, "optim.pt"))["optim"]
        optimizer.load_state_dict(checkpoint_opt)
        args.start_epoch = torch.load(os.path.join(args.load_path, "optim.pt"))["epoch"]
        scheduler.step(args.start_epoch * len(train_loader))

    xp_time = time.strftime("%m%d-%H%M%S")
    logger = FileWriter(
        xpid=args.xpid,
        tag=f"{args.comment}_seed{args.seed}",
        xp_args=args.__dict__,
        rootdir="logs",
        timestamp=xp_time,
        use_tensorboard=(not args.disable_tensorboard),
        resume=False,
    )
    args.logdir = logger.basepath

    logger.log(model)
    logger.log(f"# Params: {count_parameters(model)}")
    args.starting_step = 0

    if args.dataset.startswith("mnist") or args.dataset.startswith("geoclidean"):
        train_with_gt = False
    else:
        train_with_gt = True

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=(train_loader, val_loader, eval_train_loader, eval_test_loader),
        logger=logger,
        train_with_gt=train_with_gt
    )


if __name__ == "__main__":
    main()
