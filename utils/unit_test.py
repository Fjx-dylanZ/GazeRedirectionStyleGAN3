import torch
import torch.nn as nn
import torch.nn.functional as F

import attention
from models.redirect_net_backbone import UNet_Up
def main():
    # assume feature map size is (512, 8, 8)
    att = attention.QKVAttention(512, 512, 8)
    x = torch.randn(1, 512, 8, 8)
    #print(x.view(1, 512, -1).permute(1, 0, 2).shape)
    out = att(x)
    print(out.shape)
    assert out.shape == (1, 512, 8, 8)

    # assume key/val feature map size is (512, 4, 1)
    # assume query feature map size is (512, 8, 8)
    att = attention.CrossAttention(in_channels_q=512, in_channels_kv=512, out_channels=512, num_heads=8)
    x_query = torch.randn(1, 512, 8, 8) # use vgg16 feature map
    x_key_value = torch.randn(1, 512, 1, 1) # gaze head feature map
    out = att(x_query, x_key_value)
    print(out.shape)

    att = attention.InvertedCrossAttention(in_channels_q=4, in_channels_kv=64, num_heads=4)
    x_query = torch.randn(1, 512, 4, 1) # gaze head feature map
    x_key_value = torch.randn(1, 512, 8, 8) # use vgg16 feature map
    out = att(x_query, x_key_value)
    print(out.shape)

    # UNet_Up
    up = UNet_Up(512, 512, num_heads=8, entry_block=True)
    gaze_query = torch.randn(1, 2)
    head_pose_query = torch.randn(1, 2)
    x = torch.randn(1, 16, 512)
    x = x.permute(0, 2, 1).view(1, 512, 4, 4)
    features = torch.randn(1, 512, 4, 4)
    out = up(x, gaze_query, head_pose_query, stacked_features=features, upsample_gaze_head=False)
    print(out.shape)


if __name__ == "__main__":
    main()