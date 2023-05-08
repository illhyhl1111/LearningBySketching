import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.sketch_utils import *
from utils.shared import args
from utils.shared import stroke_config as config


criterion_l1 = torch.nn.L1Loss().cuda()
criterion_ce = torch.nn.CrossEntropyLoss().cuda()
criterion_pixel_ = torch.nn.L1Loss(reduction="none").cuda()
criterion_triplet = torch.nn.TripletMarginLoss(margin=0.5)


def criterion_pixel(inputs, targets):
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    loss = criterion_pixel_(inputs, targets)
    loss *= (targets - targets.mean(dim=1, keepdim=True)).abs()
    return loss.mean()


def simclr_loss(q, k, queue, temperature=0.1):
    k = F.normalize(k, dim=1)
    q = F.normalize(q, dim=1)
    queue = F.normalize(queue, dim=1)

    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue.t()])

    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

    loss = criterion_ce(logits, labels)
    return loss


def supcon_loss(q, k, labels, k_lables, temperature=0.1):
    k = F.normalize(k, dim=1)
    q = F.normalize(q, dim=1)

    device = q.device
    mask = torch.eq(labels.unsqueeze(1), k_lables.unsqueeze(1).T).float().to(device)

    # compute logits
    logits = torch.div(torch.matmul(q, k.T), temperature)
    exp_logits = torch.exp(logits)
    logits_denominator = exp_logits.sum(dim=1)
    logits_numerator = (exp_logits * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # compute log_prob
    log_prob = torch.log(logits_numerator / logits_denominator)

    return -log_prob.mean()


def position_loss_fn(raw_position, control_position=None, lbd_1=1.0, lbd_2=1.0, lbd_3=1.0):
    # raw_pos: [batch, nlines, nsegments+1, 2]
    if control_position is None:
        control_position = raw_position

    # force start,end point inside [-1, 1], control point inside [-1,2. 1.2]
    boundary_penalty = torch.relu(torch.norm(raw_position, p=float("inf"), dim=-1) - 1).mean()
    boundary_penalty += torch.relu(torch.norm(control_position, p=float("inf"), dim=-1) - 1.2).mean()

    # force different lines to locate in different position
    line_adjacent_penalty = 0
    nlines = raw_position.shape[1]
    for l in range(1, nlines):
        line_pose_diff = torch.norm(raw_position[:, :l] - raw_position[:, l : l + 1], p=2, dim=-1).sum(2).mean(1)
        line_adjacent_penalty += torch.relu(-line_pose_diff + 2).pow(2).mean()

    # force x[0] < x[-1]
    alignment_penalty = torch.relu(raw_position[:, :, 0, 1] - raw_position[:, :, -1, 1]).mean()

    return  lbd_1 * boundary_penalty + lbd_2 * line_adjacent_penalty + lbd_3 * alignment_penalty


def hungarian_match(position_g, radius_g, color_g, valid_g, position_s, radius_s, color_s):
        bs, nL = position_s.shape[:2]
        cur_valid_gt_size = 0

        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(position_g.shape[0]):
                is_valid_gt = valid_g[i]
                cost_matrix_l1 = torch.cdist(position_s[i], position_g[i, is_valid_gt], p=1)        # [nL, nvalid]

                if config.enable_c:
                    cost_matrix_l1_color = torch.cdist(color_s[i], color_g[i, is_valid_gt], p=1)    # [nL, nvalid]
                else:
                    cost_matrix_l1_color = 0

                if config.enable_r:
                    cost_matrix_l1_rad = torch.cdist(radius_s[i], radius_g[i, is_valid_gt], p=1)    # [nL, nvalid]
                else:
                    cost_matrix_l1_rad = 0

                cost_sum = cost_matrix_l1 + cost_matrix_l1_color + cost_matrix_l1_rad               # [nL, nvalid]
                r, c = linear_sum_assignment(cost_sum.cpu())                                        # [npair], [npair]
                r_idx.append(torch.tensor(r + nL * i).cuda())
                c_idx.append(torch.tensor(c + cur_valid_gt_size).cuda())
                cur_valid_gt_size += is_valid_gt.int().sum().item()

            r_idx = torch.cat(r_idx, dim=0)                                                         # [Npair]
            c_idx = torch.cat(c_idx, dim=0)                                                         # [Npair]
            paired_gt_decision = torch.zeros(bs * nL).cuda()                                        # [bs * nL]
            paired_gt_decision[r_idx] = 1.0

        return r_idx, c_idx


def hungarian_loss(stroke, gt):

    position_g = gt["position"]                         # [bs, nL, npos]
    radius_g = gt["radius"] / 5                         # [bs, nL, 1]
    color_g = gt["color"]                               # [bs, nL, 3]
    valid_g = position_g.abs().mean(dim=2) < 0.95

    n_back = config.n_back
    if n_back == 0:
        n_back = -config.n_lines

    position_s = stroke["position"][:, :-n_back]        # [bs, nL, npos]
    radius_s = stroke["radius"][:, :-n_back] / 5        # [bs, nL, 1]
    color_s = stroke["color"][:, :-n_back]              # [bs, nL, 3]

    assert position_g.shape == position_s.shape
    r_idx, c_idx = hungarian_match(position_g, radius_g, color_g, valid_g, position_s, radius_s, color_s)

    paired_gt_param = position_g[valid_g][c_idx, :]                                             # [Npair, npos]
    paired_pred_param = position_s.flatten(end_dim=1)[r_idx, :]                                 # [Npair, npos]

    loss_gt_pos = criterion_l1(paired_pred_param, paired_gt_param.detach())

    if config.enable_c:
        stroke_color = color_s.flatten(end_dim=1)                                               # [bs*nL, ncolor]
        paired_gt_color = color_g[valid_g][c_idx, :]                                            # [Npair, ncolor]
        paired_stroke_color = stroke_color[r_idx, :]                                            # [Npair, ncolor]

        loss_gt_color = criterion_l1(paired_stroke_color, paired_gt_color.detach())
    else:
        loss_gt_color = 0

    if config.enable_r:
        pred_radius = radius_s.flatten(end_dim=1)                                               # [bs*nL, nrad]
        paired_gt_rad = radius_g[valid_g][c_idx, :]                                             # [Npair, nrad]
        paired_stroke_rad = pred_radius[r_idx, :]                                               # [Npair, nrad]

        loss_gt_rad = criterion_l1(paired_stroke_rad, paired_gt_rad.detach())
    else:
        loss_gt_rad = 0

    return loss_gt_pos, loss_gt_color, loss_gt_rad


def update_model(model, opt, labels, k, loss):
    model.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    opt.step()

    if args.lbd_e > 0 and args.embed_loss in ["simclr", "supcon"]:
        model.update_key_encoder()
        if args.embed_loss == "simclr":
            model.update_queue(k)
        else:
            model.update_queue(k, labels)
            

def guide_loss_fn(inputs, lbs_output):
    img_background = inputs.get("back", None)
    stroke = lbs_output['stroke']
    intermediate = lbs_output['intermediate']
    sketch_black = lbs_output['sketch_background']

    loss_gt_pos = torch.zeros(1).to(args.device)
    loss_gt_color = torch.zeros(1).to(args.device)

    n_back = config.n_back
    if n_back == 0:
        n_back = -config.n_lines

    gt = {
        "position": inputs["pos"][:, -1],
        "color": inputs["color"][:, -1],
        "radius": torch.ones(*inputs["pos"][:, -1].shape[:-1], 1).to(args.device) * 2,
    }

    ### use L1 loss for background strokes
    if n_back > 0:
        loss_gt_back = criterion_l1(sketch_black, img_background)
    else:
        loss_gt_back = torch.zeros(1).to(args.device)

    if args.hungarian:
        loss_gt_pos, loss_gt_color, _ = hungarian_loss(stroke, gt)

    else:
        loss_gt_pos = criterion_l1(stroke["position"][:, :-n_back], gt["position"])
        if config.enable_c:
            loss_gt_color = criterion_l1(stroke["color"][:, :-n_back], gt["color"])

    ### progressive optimization process
    if args.prev_weight > 0:
        for layer_idx in range(1, args.n_layers):
            pos = inputs["pos"][:, layer_idx]
            color = inputs["color"][:, layer_idx]
            stroke = intermediate[layer_idx]

            gt = {
                "position": pos, 
                "color": color, 
                "radius": torch.ones(*pos.shape[:-1], 1).to(pos.device) * 2
            }

            if args.hungarian:
                loss_gt_pos_, loss_gt_color_, _ = hungarian_loss(stroke, gt)
                loss_gt_pos += loss_gt_pos_ * args.prev_weight
                loss_gt_color += loss_gt_color_ * args.prev_weight
            else:
                loss_gt_pos += criterion_l1(stroke["position"][:, :-n_back], pos) * args.prev_weight

                if config.enable_c:
                   loss_gt_color += criterion_l1(stroke["color"][:, :-n_back], color) * args.prev_weight

    return loss_gt_pos, loss_gt_color, loss_gt_back


def embed_loss_fn(model, inputs, labels, q, k):
    accuracy = torch.zeros(1).to(args.device)

    if args.embed_loss == "none":
        loss_embed = torch.zeros(1).to(args.device)

    elif args.embed_loss == "ce":
        loss_embed = criterion_ce(q, labels)
        accuracy = (q.argmax(dim=1) == labels).sum() / q.shape[0] * 100

    elif args.embed_loss == "triplet":
        rep = model.get_representation(inputs["img"], rep_type='h')
        neg = torch.cat([rep[1:], rep[:1]], dim=0)
        pos = model.get_representation(inputs["mask"], rep_type='h')
        # rep = F.normalize(q, dim=1)
        # neg = torch.cat([rep[1:], rep[:1]], dim=0)
        # pos = model.get_projection(inputs["mask"])
        # pos = F.normalize(pos, dim=1)
        loss_embed = criterion_triplet(rep, pos, neg)

    elif args.embed_loss == "simclr":
        loss_embed = simclr_loss(q, k.detach(), model.get_queue(), temperature=args.temperature)

    elif args.embed_loss == "supcon":
        loss_embed = supcon_loss(q, model.get_queue(), labels, model.get_queue_l(), temperature=args.temperature)
    
    else:
        raise NotImplementedError
                                        
    return loss_embed, accuracy


def LBS_loss_fn(model, opt, clip_loss_fn, inputs, train_model=True):
    img = inputs["img"]
    img_foreground = inputs["fore"]
    img_k = inputs["img_k"]
    labels = inputs["label"]

    lbs_output = model(img)
    lbs_output['intermediate'] = model.get_intermediate_strokes()
    q = lbs_output['projection']
    sketch_color = lbs_output['sketch_color']
    sketch_black = lbs_output['sketch_black']
    sketch_background = lbs_output['sketch_background']


    if args.embed_loss in ['simclr', 'supcon']:
        k = model.get_key_value(img_k)
    else:
        k = None

    ##### L_{embed} #####
    if args.lbd_e > 0:
        loss_embed, accuracy = embed_loss_fn(model, inputs, labels, q, k)
        loss_embed *= args.lbd_e
    else:
        loss_embed = torch.zeros(1).to(args.device)
        accuracy = torch.zeros(1).to(args.device)

    ##### L_{guide} #####
    loss_gt_pos, loss_gt_color, loss_gt_back = guide_loss_fn(inputs, lbs_output)
    loss_gt_pos *= args.lbd_g
    loss_gt_color *= args.lbd_g
    loss_gt_back *= args.lbd_g
    
    ##### L_{percept} #####
    if args.lbd_p != 0:
        clip_loss_dict = clip_loss_fn(sketch_black, img_foreground, None, None, 1, None)
        loss_percept = sum(list(clip_loss_dict.values())) * args.lbd_p
    else:
        loss_percept = torch.zeros(1).to(args.device)

    loss_gt = loss_gt_pos + loss_gt_color + loss_gt_back
    loss_LBS = loss_gt + loss_percept + loss_embed

    if train_model:
        update_model(model, opt, labels, k, loss_LBS)

    losses = {
        f"loss_embed_{args.embed_loss}": loss_embed,
        "loss_gt_pos": loss_gt_pos,
        "loss_gt_color": loss_gt_color,
        "loss_gt_back": loss_gt_back,
        "loss_percept": loss_percept,
        "loss_total": loss_LBS,
        "accuracy": accuracy,
    }

    return {
        "masked_images": img,
        "sketch_color": sketch_color,
        "sketch_black": sketch_black,
        "sketch_background": sketch_background,
    }, losses


def l1_loss_fn(model, opt, inputs, train_model=True):
    img = inputs['img']
    img_k = inputs['img_k']
    labels = inputs['label']

    lbs_output = model(img, sketch_type='black')
    stroke = lbs_output['stroke']
    sketch = lbs_output['sketch_black']
    q = lbs_output['projection']

    if args.embed_loss in ["simclr", "supcon"]:
        k = model.get_key_value(img_k)
    else:
        k = None

    loss_penalty = position_loss_fn(stroke["raw_position"], stroke["control_position"], 1, 10, 1) * args.lbd_g

    accuracy = torch.zeros(1).to(args.device)

    ##### L_{embed} #####
    if args.lbd_e > 0:
        loss_embed, accuracy = embed_loss_fn(model, inputs, labels, q, k)
        loss_embed *= args.lbd_e
    else:
        loss_embed = torch.zeros(1).to(args.device)

    #### L_1 ####
    if args.lbd_p != 0:
        loss_l1 = criterion_pixel(sketch, img) * args.lbd_p
    else:
        loss_l1 = torch.zeros(1).to(args.device)

    loss_LBS = loss_l1 + loss_embed + loss_penalty
    losses = {
        "loss_penalty": loss_penalty,
        "loss_l1": loss_l1,
        f"loss_embed_{args.embed_loss}": loss_embed,
        "loss_total": loss_LBS,
        "accuracy": accuracy,
    }

    if train_model:
        update_model(model, opt, labels, k, loss_LBS)

    return {
        "masked_images": img,
        "sketch": sketch,
    }, losses
