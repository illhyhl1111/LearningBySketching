import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, CenterCrop, Normalize, InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import utils.sketch_utils as sketch_utils
from models.renderer import Renderer
from models.resnet import resnet18
import CLIP_.clip as clip

from utils.shared import args
from utils.shared import stroke_config as config

class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)


CLIP_encoder = None
VGG_model = None


def linear_(in_dim, out_dim, bn=True):
    if bn:
        return [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        ]
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
    ]

def conv_(in_channels, out_channels, kernel_size, stride, padding, bn=True):
    if bn:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    ]
    

class StrokeEmbeddingNet(nn.Module):
    """Constructs a Stroke Embedding Network."""
    def __init__(self, n_hidden, n_embedding, d_project, output_num=-1):
        super(StrokeEmbeddingNet, self).__init__()
        n_lines = config.n_lines                                             # number of real lines
        rep_len = config.n_params

        self.output_num = output_num
        self.stroke_to_rep = nn.Sequential(
            Lambda(lambda x: x.flatten(0, 1)),                                 # [bs, nlines, stroke_len] -> [bs*nlines, stroke_len]
            *linear_(rep_len, n_hidden),
            *linear_(n_hidden, n_hidden),
            Lambda(lambda x: x.view(-1, n_lines, n_hidden)),                   # [bs*nlines, n_hidden] -> [bs, nlines, n_hidden]
            Lambda(lambda x: x.mean(dim=1)),

            *linear_(n_hidden, n_embedding),
        )

        if output_num == -1:
            self.projection_layer = nn.Sequential(
                *linear_(n_embedding, n_embedding),
                nn.Linear(n_embedding, d_project),
                nn.BatchNorm1d(d_project, affine=False),
            )
        else:
            self.projection_layer = nn.Linear(n_embedding, int(output_num))                

    def forward(self, z):
        embed_z = self.embed(z)
        return self.projection_layer(embed_z)

    def embed(self, z):
        return self.stroke_to_rep(z)


class StrokeGenerator(nn.Module):
    def __init__(self, patch_num, n_hidden, n_layers):
        super(StrokeGenerator, self).__init__()

        self.n_lines = config.n_lines                                               # number of real lines
        self.n_lines_decode = config.n_lines + int(config.connected)                # number of lines to be decoded
        self.n_position = config.n_pos
        self.n_decode_color = config.n_color if config.enable_c else 0
        self.n_decode_rad = config.n_rad if config.enable_r else 0

        self.row_embed = nn.Parameter(torch.rand(patch_num[0], n_hidden // 2))
        self.col_embed = nn.Parameter(torch.rand(patch_num[1], n_hidden // 2))

        decoder_layers = torch.nn.TransformerDecoderLayer(n_hidden, 8, batch_first=True, dim_feedforward=n_hidden*2,
                                                          activation='gelu', norm_first=True, dropout=0.0)
        self.decoder_norm = nn.LayerNorm(n_hidden, eps=1e-5)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, n_layers, self.decoder_norm)

        self.intermediate_strokes = dict()
        self.init_stroke = nn.Parameter(torch.randn(self.n_lines_decode, n_hidden) * 0.02)

        # number of decoded parameters for a single stroke
        n_decode_params = self.n_position + self.n_decode_color + self.n_decode_rad
        if config.connected:
            n_decode_params += 2

        self.decode_stroke = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden // 4, n_decode_params),
        )

        self.init_hook()

    def init_hook(self):
        for idx, layer in enumerate(self.transformer_decoder.layers):
            layer.register_forward_hook(self.save_outputs_hook(idx+1))
            self.intermediate_strokes[idx+1] = torch.empty(0)

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            if self.decoder_norm is not None:
                self.intermediate_strokes[layer_id] = self.decoder_norm(output)
            else:
                self.intermediate_strokes[layer_id] = output
        return fn

    def decode_to_params(self, hidden):
        bs = hidden.shape[0]
        hidden = self.decode_stroke(hidden)
        device = hidden.device

        p_cont, p_col, p_rad = torch.split(hidden, [self.n_position, self.n_decode_color, self.n_decode_rad], dim=2)

        if config.connected:
            p_cont = torch.cat([p_cont[:, :-1, -2:], p_cont[:, 1:]], dim=-1)                                    # [bs, nlines, ncont+2]
            p_col = p_col[:, 1:]
            p_rad = p_rad[:, 1:]


        raw_position = p_cont.view(bs, self.n_lines, -1, 2)
        if config.line_style == 'bezier':
            coordpairs = config.coordpairs

            stacked_cont = [raw_position[:, :, coordpairs[0, 0]]]
            stacked_cont += [raw_position[:, :, coordpairs[i, -1]] for i in range(coordpairs.shape[0])]
            control_position = torch.stack(stacked_cont, dim=-2)                                                 # [batch, nlines, nsegments+1, 2]
        else:
            control_position = raw_position

        if not config.enable_c:
            p_col = torch.zeros(bs, self.n_lines, config.n_color).to(device)
        else:
            p_col = torch.sigmoid(p_col)

        if not config.enable_r:
            p_rad = torch.ones(bs, self.n_lines, config.n_rad).to(device)
            n_foreground = self.n_lines - config.n_back
            p_rad[:, n_foreground:] *= 5            # thick background stroke
        else:
            p_rad = torch.sigmoid(p_rad) * 4 + 1

        return {
            'position': p_cont,
            'radius': p_rad,
            'color': p_col,
            "raw_position": raw_position, 
            "control_position": control_position, 
        }

    def forward(self, cnn_feature):
        bs = cnn_feature.shape[0]
        device = cnn_feature.device
        h, w = cnn_feature.shape[-2:]

        cnn_feature_permuted = cnn_feature.flatten(2).permute(0, 2, 1)          # NChw -> N(h*w)C

        pos_embed = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)

        init_stroke = self.init_stroke.repeat(bs, 1, 1).to(device)
        
        # Set the last embedding to 1 for background strokes
        init_background = torch.zeros_like(init_stroke)
        n_foreground = self.n_lines_decode - config.n_back
        init_background[:, n_foreground:, 0] = 1
        init_stroke += init_background

        hidden = self.transformer_decoder(init_stroke, cnn_feature_permuted + pos_embed)

        strokes = self.decode_to_params(hidden)

        return strokes

    def get_intermediate_strokes(self):
        return {k: self.decode_to_params(v) for k, v in self.intermediate_strokes.items()}

class LBS(nn.Module):
    def __init__(self):
        super(LBS, self).__init__()
        
        n_hidden = args.n_hidden
        n_embedding = args.n_embedding
        d_project = args.d_project
        n_layers = args.n_layers
        
        self.rep_type = args.rep_type
        image_size = args.image_size
        self.train_encoder = args.train_encoder
        self.use_mask = not args.no_mask
        
        if args.dataset.startswith('mnist') or args.dataset.startswith('geoclidean'):
            use_l1 = True
            self.train_encoder = True
            self.use_mask = False
        else:
            use_l1 = False

        self.normalize = Compose([
            Resize(image_size, interpolation=BICUBIC),
            CenterCrop(image_size),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if self.train_encoder:
            if use_l1:
                self.cnn_encoder = nn.Sequential(
                    *conv_(config.n_color, n_hidden//8, 4, 2, 1),
                    *conv_(n_hidden//8, n_hidden//4, 4, 2, 1),
                    *conv_(n_hidden//4, n_hidden//2, 4, 2, 1),
                    *conv_(n_hidden//2, n_hidden, 3, 1, 1),
                )
            else:
                self.cnn_encoder = resnet18(in_channel=config.n_color)

            H, W = image_size, image_size
            patch_num = H//8, W//8

        else:
            # use pretrained clip encoder
            global CLIP_encoder
            CLIP_encoder, _ = clip.load("ViT-B/32", args.device, jit=False)
            CLIP_encoder = CLIP_encoder.visual
            CLIP_encoder.eval()

            self.normalize = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

            patch_num = 7, 7

        self.stroke_generator = StrokeGenerator(patch_num, n_hidden, n_layers)

         # number of parameters for a single parameterized strokes
        self.num_stroke_params = config.n_params

        if args.embed_loss == 'ce':
            output_num = args.class_num
        else:
            output_num = -1

        self.embedding_net = StrokeEmbeddingNet(n_hidden, n_embedding, d_project, output_num)

        self.reset_parameters()

    def reset_parameters(self, model=None, init_type='normal', init_gain=0.05):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)

        if model is None:
            model = self
        print('initialize network with %s' % init_type)
        model.apply(init_func)

    def vectorize_stroke(self, stroke, flatten=True):
        rep = torch.cat([stroke['position'].flatten(2), stroke['color'], stroke['radius']], dim=-1)
        assert self.num_stroke_params == rep.shape[-1]

        if flatten:
            rep = rep.flatten(1)
        return rep
    
    def get_representation(self, rep_type, z_e, z_p, z_h):
        if rep_type == 'LBS+':
            rep_type = 'eph'
        elif rep_type == 'LBS':
            rep_type = 'ep'

        allowed = ['e', 'p', 'h']
        assert len(rep_type) > 0, 'length of "rep_type" should not be empty string'
        assert all(c in allowed for c in rep_type), \
            f'unsupported "rep_type" {rep_type}, should be one of LBS+, LBS, or a string consisting of only e, p, and h'
        
        rep_list = []
        if 'e' in rep_type:
            rep_list.append(z_e)
        if 'p' in rep_type:
            rep_list.append(z_p)
        if 'h' in rep_type:
            rep_list.append(z_h)

        return torch.cat(rep_list, dim=1)

    def forward(self, image, masked):
        image = self.normalize(image)
        if self.use_mask:
            masked = self.normalize(masked)

        if self.train_encoder:
            cnn_feature = self.cnn_encoder(image)
            if self.use_mask:
                masked_feature = self.cnn_encoder(masked)
        else:
            global CLIP_encoder
            def forward_clip(image):
                dtype = CLIP_encoder.conv1.weight.dtype

                x = image.type(dtype)
                x = CLIP_encoder.conv1(x)                               # [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)               # [*, width, grid ** 2]
                x = x.permute(0, 2, 1)                                  # [*, grid ** 2, width]
                x = torch.cat([CLIP_encoder.class_embedding.to(x.dtype) \
                               + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                               , x], dim=1)                             # [*, grid ** 2 + 1, width]
                x = x + CLIP_encoder.positional_embedding.to(x.dtype)
                x = CLIP_encoder.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = CLIP_encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = CLIP_encoder.ln_controlt(x[:, 1:, :])
                        
                x = x.permute(0, 2, 1)
                return x
            
            with torch.no_grad():
                x = forward_clip(image)
                cnn_feature = x.view(*x.shape[:2], 7, 7)
                if self.use_mask:
                    masked_feature = forward_clip(masked)
        
        if self.use_mask:
            cnn_feature = torch.cat([cnn_feature, masked_feature], dim=1)

        strokes = self.stroke_generator(cnn_feature)
        stroke_vector = self.vectorize_stroke(strokes, flatten=False)
        projection = self.embedding_net(stroke_vector)
        stroke_embedding = self.embedding_net.embed(stroke_vector)
        cnn_feature = F.adaptive_avg_pool2d(cnn_feature, (1, 1)).squeeze(-1).squeeze(-1)

        z = self.get_representation(self.rep_type, cnn_feature, stroke_vector.flatten(1), stroke_embedding)

        return {
            'z_e': cnn_feature, 
            'z_p': stroke_vector.flatten(1),
            'z_h': stroke_embedding,
            'stroke': strokes,
            'projection': projection,
            'z': z
        }


class MoCoLBS(LBS):
    class Queue:
        def __init__(self, K, init_q):
            self._queue = init_q
            self.K = K

        @property
        def q(self):
            assert self._queue.shape[0] == self.K
            return self._queue

        def update(self, k):
            k = k.detach()
            self._queue = torch.cat((k, self._queue), dim=0)[:self.K]

        def size(self):
            return self._queue.shape[0]


    def __init__(self) -> None:
        super(MoCoLBS, self).__init__()

        model_state_dict = self.state_dict()

        self.key_encoder = LBS()
        self.key_encoder.load_state_dict(model_state_dict)
        sketch_utils.set_grad(self.key_encoder, False)

        ### init queue
        inital_queue = torch.randn(args.max_queue, args.d_project).to(args.device)
        self.queue = self.Queue(args.max_queue, inital_queue)

        if args.embed_loss == 'supcon':
            class_num = args.class_num
            inital_queue_l = torch.randint(class_num, (args.max_queue, )).to(args.device)
            self.queue_l = self.Queue(args.max_queue, inital_queue_l)
        else:
            self.queue_l = None

        self.momentum = args.momentum

    def parameters(self, recurse: bool = True):
        return filter(lambda p: p.requires_grad, super(MoCoLBS, self).parameters(recurse))
        
    def update_queue(self, inputs, labels=None):
        self.queue.update(inputs)
        if labels is not None and self.queue_l is not None:
            self.queue_l.update(labels)

    def update_key_encoder(self):
        for q_params, k_params in zip(self.cnn_encoder.parameters(), self.key_encoder.parameters()):
            k_params.data.copy_(self.momentum*k_params + q_params*(1.0-self.momentum))

    def get_queue(self):
        return self.queue.q

    def get_queue_l(self):
        assert self.queue_l is not None
        return self.queue_l.q
    
    def get_key_value(self, image, masked):
        with torch.no_grad():
            k = self.key_encoder(image, masked)['projection']
        return k


class SketchModel(nn.Module):
    def __init__(self):
        super(SketchModel, self).__init__()

        if args.embed_loss in ['simclr', 'supcon']:
            self.lbs_model = MoCoLBS()
        else:
            self.lbs_model = LBS()

        self.renderer = Renderer(args.image_size, min(64, args.image_size))

        self.use_mask = self.lbs_model.use_mask

    def get_masked_img(self, image, mask=None):
        if self.use_mask:
            if mask is None:
                mask = sketch_utils.mask_image(image)
            foreground = image * mask + (1 - mask)
            background = image * (1 - mask) + mask
        else:
            foreground = image
            background = None
        
        return foreground, background

    def forward(self, image, mask=None, sketch_type=['color', 'black', 'background']):
        foreground, background = self.get_masked_img(image, mask)
            
        lbs_output = self.lbs_model(foreground, background)
        sketch = self.renderer(lbs_output['stroke'], sketch_type)

        for idx, types in enumerate(sketch_type):
            lbs_output[f'sketch_{types}'] = sketch[idx]

        return lbs_output
    
    def get_intermediate_strokes(self):
        return self.lbs_model.stroke_generator.get_intermediate_strokes()

    def get_key_value(self, image, mask=None):
        foreground, background = self.get_masked_img(image, mask)

        if isinstance(self.lbs_model, MoCoLBS):
            return self.lbs_model.get_key_value(foreground, background)
        
        return None

    def sketch_image(self, image, sketch_type='color'):
        foreground, background = self.get_masked_img(image, None, None)

        stroke = self.lbs_model(foreground, background)['stroke']
        return self.rasterize_stroke(stroke, sketch_type)

    def set_progress(self, progress):
        self.renderer.set_sigma2(progress)

    def parameters(self):
        return self.lbs_model.parameters()
    
    def update_queue(self, inputs, labels=None):
        assert isinstance(self.lbs_model, MoCoLBS)

        self.lbs_model.queue.update(inputs)
        if labels is not None and self.lbs_model.queue_l is not None:
            self.lbs_model.queue_l.update(labels)

    def update_key_encoder(self):
        assert isinstance(self.lbs_model, MoCoLBS)

        for q_params, k_params in zip(self.lbs_model.cnn_encoder.parameters(), self.lbs_model.key_encoder.parameters()):
            k_params.data.copy_(self.lbs_model.momentum*k_params + q_params*(1.0-self.lbs_model.momentum))

    def get_queue(self):
        assert isinstance(self.lbs_model, MoCoLBS)
        return self.lbs_model.queue.q

    def get_queue_l(self):
        assert isinstance(self.lbs_model, MoCoLBS)
        assert self.lbs_model.queue_l is not None
        return self.lbs_model.queue_l.q

    def get_representation(self, image, mask=None, rep_type='LBS+'):
        if rep_type == 'full':
            rep_type = ''

        foreground, background = self.get_masked_img(image, mask)
        lbs_output = self.lbs_model(foreground, background)
        z_e = lbs_output['z_e']
        z_p = lbs_output['z_p']
        z_h = lbs_output['z_h']
        return self.lbs_model.get_representation(rep_type, z_e, z_p, z_h)
    
    def rasterize_stroke(self, stroke, sketch_type='color'):
        return self.renderer(stroke, sketch_type)

    def eval_grid(self, dataset):
        eval_num = 5
        grid = []
        device = args.device

        with torch.no_grad():
            for i in range(eval_num):
                if isinstance(dataset[i][0], tuple) or isinstance(dataset[i][0], list):
                    image = dataset[i][0][0].to(device).unsqueeze(0)
                elif isinstance(dataset[i], tuple) or isinstance(dataset[i], list):
                    image = dataset[i][0].to(device).unsqueeze(0)
                else:
                    image = dataset[i].to(device).unsqueeze(0)
                foreground, background = self.get_masked_img(image, None)

                grid.append(foreground)

                with torch.no_grad():
                    stroke = self.lbs_model(foreground, background)['stroke']
                p, s, c, b = self.renderer.reshape_params(stroke)

                for j in range(p.shape[1]):
                    grid.append(self.renderer.rendering_reshaped_params((p[:, :j+1], s[:, :j+1], c[:, :j+1], b)))

        grid = torch.cat(grid, dim=0)
        grid = grid.view(eval_num, -1, *grid.shape[1:]).transpose(0, 1).flatten(0, 1)
        return grid
