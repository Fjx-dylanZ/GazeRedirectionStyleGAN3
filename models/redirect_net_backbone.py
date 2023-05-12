import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dnnlib
import numpy as np
import legacy
from utils.attention import QKVAttention, CrossAttention, InvertedCrossAttention
from utils.dynamic_linear import DynamicLinear

class StyleGAN3FeatureExtractor(nn.Module):
    def __init__(self,
                 path=None):
        super().__init__()
        with dnnlib.util.open_url(path) as f:
            self.Generator = legacy.load_network_pkl(f)['G_ema'].eval()
            self.Generator.requires_grad_(False) # api: G.synthesis(batched_latents)

    def forward(self, tensor_in, layer_out_list=None):
        ws = tensor_in.unbind(dim=1)
        x = self.Generator.synthesis.input(ws[0])
        feature_maps = []
        for i, (name, w) in enumerate(zip(self.Generator.synthesis.layer_names, ws[1:])):
            x = getattr(self.Generator.synthesis, name)(x, w)
            if layer_out_list is not None and (i in layer_out_list or name in layer_out_list):
                feature_maps.append(x)
        if self.Generator.synthesis.output_scale != 1:
            x = x * self.Generator.synthesis.output_scale
        
        if layer_out_list is not None:
            return x, feature_maps
        return x
    
    def synthesis(self, x):
        return self.Generator.synthesis(x)


class UNet_Up(nn.Module):
    def __init__(self, in_channels, out_channels, embed_gaze_dim, embed_dim,
                 num_heads=8, entry_block=False, dropout=0.1,
                 attn_q_dim=None
                 ):
        # entry_block: if True, then it is the first block of the UNet (512, 4, 4) -> (512, 16, 16)
        #              otherwise, just upsample by 2
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.entry_block = entry_block
        if attn_q_dim is None:
            self.guided_attention = CrossAttention(in_channels, in_channels, in_channels, num_heads)
        else:
            self.guided_attention = CrossAttention(attn_q_dim, in_channels, in_channels, num_heads)
        #self.guided_attention = InvertedCrossAttention(embed_gaze_dim, embed_dim, num_heads)
        #self.self_attention = QKVAttention(in_channels, in_channels, num_heads)
        self.attention_norm = nn.BatchNorm2d(in_channels)
        self.norm_out = nn.BatchNorm2d(in_channels)
        self.attention_mlp = nn.Sequential(
            DynamicLinear(bias=True, same_dim=True),
            nn.GELU(),
            nn.Dropout(dropout),
            DynamicLinear(bias=True, same_dim=True),
            nn.Dropout(dropout)
        )

        #self.upsample = nn.Upsample(scale_factor=2 if not entry_block else 4, mode="nearest", align_corners=False)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2 if not entry_block else 4, stride=2 if not entry_block else 4, padding=0)
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1) # potentially change to DoubleConvj
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True) if not entry_block else nn.Identity()
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, gaze_head_query, stacked_features=None,
                 upsample_gaze_head=False,
                 concat_not_add=False,
                 stylegan3_features=None
                 ):
        # x: (batch_size, in_channels, W, H)

        # Attention
        if concat_not_add:
            if stylegan3_features is not None and stacked_features is not None:
                #print(f'stacked_features shape: {stacked_features.shape}, stylegan3_features shape: {stylegan3_features.shape}, x shape: {x.shape}')
                x_query = torch.cat([x, stylegan3_features, stacked_features], dim=1)
            elif stylegan3_features is not None:
                x_query = torch.cat([x, stylegan3_features], dim=1)
            elif stacked_features is not None:
                x_query = torch.cat([x, stacked_features], dim=1)
            else:
                x_query = x
            #print(f"x_query shape: {x_query.shape}, gaze_head_query shape: {gaze_head_query.shape}")
            x_query = self.guided_attention(x_query, gaze_head_query)
        else:
            x_query = self.guided_attention(x, gaze_head_query)
        x_query = self.attention_norm(self.attention_mlp(x_query))
        #print(f"x_query shape: {x_query.shape}")
        #print(f"stacked_features shape: {stacked_features.shape}")
        if stacked_features is not None and not concat_not_add:
            x_query = x_query + stacked_features
        
        #x_query = self.self_attention(x_query)
        x_query = x_query + x
        x_query = self.norm_out(x_query)
        x_query = self.relu(x_query)

        # Upsample
        x_query = self.upsample(x_query)
        x_query = self.conv(x_query)
        #x_query = self.norm(x_query)

        if upsample_gaze_head:
            return x_query, gaze_head_query
        else:
            return x_query
        

class UNet_Down(nn.Module):
    def __init__(self, in_channels, out_channels, embed_gaze_dim, embed_dim, num_heads=8, exit_block=False, dropout=0.1, attn_q_dim=None, teacher_forcing=False):
        # exit_block: if True, then it is the last block of the UNet (512, 16, 16) -> (512, 4, 4)
        #             otherwise, just downsample by 2
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exit_block = exit_block
        if attn_q_dim is None:
            self.guided_attention = CrossAttention(in_channels, in_channels, in_channels, num_heads)
        else:
            self.guided_attention = CrossAttention(attn_q_dim, in_channels, in_channels, num_heads)
            self.input_correction = nn.Conv2d(in_channels, attn_q_dim, kernel_size=1, padding=0, stride=1)
        #self.guided_attention = InvertedCrossAttention(embed_gaze_dim, embed_dim, num_heads)
        #self.self_attention = QKVAttention(in_channels, in_channels, num_heads)
        self.attention_norm = nn.BatchNorm2d(in_channels)
        self.norm_out = nn.BatchNorm2d(in_channels)
        self.attention_mlp = nn.Sequential(
            DynamicLinear(bias=True, same_dim=True),
            nn.GELU(),
            nn.Dropout(dropout),
            DynamicLinear(bias=True, same_dim=True),
            nn.Dropout(dropout)
        )
        #self.downsample = nn.MaxPool2d(kernel_size=2 if not exit_block else 4, stride=2 if not exit_block else 4)
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=2 if not exit_block else 4, stride=2 if not exit_block else 4, padding=0)
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True) if not exit_block else nn.Identity()
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.teacher_forcing = teacher_forcing

    def forward(self, x, gaze_head_query, stacked_features=None, downsample_gaze_head=False,
                concat_not_add=False,
                stylegan3_features=None,
                 skip_connection=None):
        # x: (batch_size, in_channels, W, H)

        # Attention
        if concat_not_add:
            if stylegan3_features is not None and stacked_features is not None:
                #print(f'stacked_features shape: {stacked_features.shape}, stylegan3_features shape: {stylegan3_features.shape}, x shape: {x.shape}')
                x_query = torch.cat([x, stylegan3_features, stacked_features], dim=1)
            elif stylegan3_features is not None:
                x_query = torch.cat([x, stylegan3_features], dim=1)
            elif stacked_features is not None:
                x_query = torch.cat([x, stacked_features], dim=1)
            else:
                x_query = x
            #print(f"x_query shape: {x_query.shape}, gaze_head_query shape: {gaze_head_query.shape}")
            x_query = self.guided_attention(x_query, gaze_head_query)
        else:
            
            if self.teacher_forcing:
                #print(f"prior x shape: {x.shape}")
                x_fixed = self.input_correction(x) # fix annoying input dim mismatch when using teacher forcing
                #print(f"x shape: {x.shape}, gaze_head_query shape: {gaze_head_query.shape}")
                x_query = self.guided_attention(x_fixed, gaze_head_query)
            else:
                x_query = self.guided_attention(x, gaze_head_query)
        x_query = self.attention_norm(self.attention_mlp(x_query))

        #print(f"x_query shape: {x_query.shape}")
        #x_query = self.self_attention(x_query)
        x_query = self.norm_out(x_query + x)
        x_query = self.relu(x_query)

        # Downsample
        x_query = self.downsample(x_query)
        x_query = self.conv(x_query)
        #x_query = self.norm(x_query)

        # Skip connection
        if skip_connection is not None:
            x_query = x_query + skip_connection
        
        if downsample_gaze_head:
            return x_query, gaze_head_query
        else:
            return x_query




class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, is_upsample=False, is_cross_attention=False):
        super().__init__()
        self.attention = QKVAttention(in_channels, out_channels, num_heads)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.is_upsample = is_upsample
        if is_upsample:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.cross_attention = CrossAttention(out_channels, out_channels, out_channels, num_heads)

    def forward(self, x1, x2=None, x_query=None):
        if self.is_upsample:
            x1 = self.upsample(x1)
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x = self.conv(x1)
        if self.is_cross_attention and x_query is not None:
            out = x + self.cross_attention(x_query, x)
        else:
            out = x + self.attention(x)
        return out

class GazeHeadMapping(nn.Module):
    def __init__(self, out_channels, in_channels=1):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        assert x.shape[-1] == 4
        x = self.mapping(x.unsqueeze(-1)).permute(0, 2, 1)
        return x.view(x.shape[0], x.shape[1], 2, 2)



# class Encoder(nn.modules):
#     def __init__(self,
#             generator="pretrained/stylegan3-r-ffhqu-256x256.pkl",
#             inversion_encoder=inversion_encoder.Encoder,
#             inversion_encoder_checkpoint="pretrained/inversion_encoder_snapshot_100000.pkl",
#             gaze_estimator=gaze_estimation_net.VGG_Gaze_Estimator,
#             gaze_estimator_checkpoint="pretrained/at_step_0069999_vggFreezeFalse.pth.tar",
#             ):
#         super(Encoder, self).__init__()
#         # -------------------- Backbone Network --------------------
#         # Initialize Gaze / Head Pose Estimator
#         self.GazeEstimator = gaze_estimator()
#         self.GazeEstimator.load_state_dict(torch.load(gaze_estimator_checkpoint), strict=True)
#         self.GazeEstimator.eval()
#         self.GazeEstimator.requires_grad_(False)
#         # Initialize Inversion Encoder
#         self.InversionEncoder = inversion_encoder(pretrained=inversion_encoder_checkpoint, w_avg=None)
#         self.InversionEncoder.eval()
#         self.InversionEncoder.requires_grad_(False)
#         # Initialize StyleGAN3 Generator
#         with dnnlib.util.open_url(generator) as f:
#             self.Generator = legacy.load_network_pkl(f)['G_ema'].eval()
#         self.Generator.requires_grad_(False) # api: G.synthesis(batched_latents)
#         # -------------------- Redirect Network --------------------
#         self.gaze_mapper = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.Tanh()
#         )
#         self.head_pose_mapper = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.Tanh()
#         ) # later to be concatenated with input latent vector (16, 512) -> (18, 512)
#         self.encoder = nn.Sequential(
#             UNetBlock(512, 64),
#             nn.MaxPool2d(2),
#             UNetBlock(64, 128),
#             nn.MaxPool2d(2),
#             UNetBlock(128, 256),
#             nn.MaxPool2d(2),
#             UNetBlock(256, 512)
#         )

#     def forward(self, x):
        

