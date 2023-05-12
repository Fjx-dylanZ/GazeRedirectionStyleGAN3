import math
import pickle

import numpy as np
import torch
import torch.nn as nn

from models.inversion_encoder.networks_arcface import get_blocks, bottleneck_IR_SE


class GradualStyleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x) # [b,512,H,W]->[b,512,1,1]
        # (H,W) in [(16,16),(32,32),(64,64)]
        x = x.view(-1, self.out_c) # [b,512,1,1]-> [b,512]
        x = self.linear(x)
        x = nn.LeakyReLU()(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, spatial, **styleblock):
        super(TransformerBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.num_encoder_layers = styleblock['num_encoder_layers']
        num_pools = int(np.log2(spatial))-4
        modules = []
#        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
#                    nn.LeakyReLU()]
        for i in range(num_pools-1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.positional_encoding = PositionalEncoding(out_c)
        self.transformer_encoder = nn.Transformer(num_encoder_layers=self.num_encoder_layers).encoder
#(out_c, out_c)

    def forward(self, x):
        x = self.convs(x) # [b,512,H,W]->[b,512,16,16]
        # (H,W) in [(16,16),(32,32),(64,64)]
        x = x.view(x.shape[0], -1, self.out_c) # [b,256,512]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)[:,0,:]
        return x


class GradualStyleEncoder(torch.nn.Module):
    def __init__(self, **styleblock):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(50) # num_layers=50
        unit_module = bottleneck_IR_SE # 'ir_se' bottleneck

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        ) # [b,3,256,256]->[b,3,64,64]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList() # feat->latent

        # TODO: 
        # need some other method for handling w[0]
            # train w[0] separately ?
        # coarse_ind, middle_ind tuning
        self.style_count = 16
        self.coarse_ind = 3
        self.middle_ind = 7

        if 'arch' in styleblock:
            for i in range(self.style_count):
                if i < self.coarse_ind:
                    style = TransformerBlock(512, 512, 16, **styleblock)
                elif i < self.middle_ind:
                    style = TransformerBlock(512, 512, 32, **styleblock)
                else:
                    style = TransformerBlock(512, 512, 64, **styleblock)
                self.styles.append(style)
        else:
            for i in range(self.style_count):
                if i < self.coarse_ind:
                    style = GradualStyleBlock(512, 512, 16)
                elif i < self.middle_ind:
                    style = GradualStyleBlock(512, 512, 32)
                else:
                    style = GradualStyleBlock(512, 512, 64)
                self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x) # [b,3,256,256]->[b,64,256,256]

        latents = []
        modulelist = list(self.body._modules.values())

        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x # [b,128,64,64]
            elif i == 20:
                c2 = x # [b,256,32,32]
            elif i == 23:
                c3 = x # [b,512,16,16]

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2)) # [b,512,32,32]
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1)) # [b,512,64,64]
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
#       Adding Transformer structure in this results in gradient exploding
#        if self.neck is not None:
#            out = self.neck(out)
        return out


class Encoder(torch.nn.Module):
    """stylegan3 encoder implementation
    based on pixel2sylte2pixel GradualStyleEncoder

    (b, 3, 256, 256) -> (b, 16, 512)

    stylegan3 generator synthesis
    (b, 16, 512) -> (b, 3, 1024, 1024)
    """
    def __init__(
        self,
        pretrained=None,
        w_avg=None,
        **kwargs, 
    ):
        super(Encoder, self).__init__()
        self.encoder = GradualStyleEncoder(**kwargs) # 50, irse
        self.resume_step = 0
        self.w_avg = w_avg

        # load weight
        if pretrained is not None:
            with open(pretrained, 'rb') as f:
                dic = pickle.load(f)
            weights = dic['E']
            weights_ = dict()
            for layer in weights:
                if 'module.encoder' in layer:
                    weights_['.'.join(layer.split('.')[2:])] = weights[layer]
            self.resume_step = dic['step']
            self.encoder.load_state_dict(weights_, strict=True)
            del weights
        else:
            irse50 = torch.load("pretrained/model_ir_se50.pth", map_location='cpu')
            weights = {k:v for k,v in irse50.items() if "input_layer" not in k}
            self.encoder.load_state_dict(weights, strict=False)

    def forward(self, img):
        if self.w_avg is None:
            return self.encoder(img)
        else: # train delta_w, from w_avg
            delta_w = self.encoder(img)
            w = delta_w + self.w_avg.repeat(delta_w.shape[0],1,1)
            return w
