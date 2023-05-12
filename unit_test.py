import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.attention as attention
from models.redirect_net_backbone import UNet_Up, UNet_Down, GazeHeadMapping
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
    up = UNet_Up(512, 512, num_heads=8, entry_block=True, embed_dim=None, embed_gaze_dim=None)
    gaze_query = torch.randn(4, 2)
    head_pose_query = torch.randn(4, 2)
    gh_mapper = GazeHeadMapping(512)
    gh_512 = gh_mapper(torch.cat((gaze_query, head_pose_query), dim=-1))
    x = torch.randn(4, 16, 512)
    x = x.permute(0, 2, 1).view(4, 512, 4, 4)
    features = torch.randn(4, 512, 4, 4)
    out = up(x, gaze_head_query=gh_512, stacked_features=features, upsample_gaze_head=False)
    print(out.shape)

    #UNet_Up with entry block=False
    up = UNet_Up(256, 128, num_heads=8, entry_block=False, embed_dim=None, embed_gaze_dim=None)
    gaze_query = torch.randn(4, 2)
    head_pose_query = torch.randn(4, 2)
    gh_mapper = GazeHeadMapping(256)
    gh_256 = gh_mapper(torch.cat((gaze_query, head_pose_query), dim=-1))
    x = torch.randn(4, 256, 32, 32)
    features = torch.randn(4, 256, 32, 32)
    out = up(x, gaze_head_query=gh_256, stacked_features=features)
    print(out.shape)

    # UNet_Down
    down = UNet_Down(512, 512, num_heads=8, exit_block=True, embed_dim=None, embed_gaze_dim=None)
    gaze_query = torch.randn(4, 2)
    head_pose_query = torch.randn(4, 2)
    gh_mapper = GazeHeadMapping(512)
    gh_512 = gh_mapper(torch.cat((gaze_query, head_pose_query), dim=-1))
    skip_connection = torch.randn(4, 512, 4, 4)
    x = torch.randn(4, 512, 16, 16)
    features = torch.randn(4, 512, 16, 16)
    out = down(x, gaze_head_query=gh_512, stacked_features=features, skip_connection=skip_connection)
    print(out.shape)

    # UNet_Down with exit block=False
    down = UNet_Down(128, 256, num_heads=8, exit_block=False, embed_dim=None, embed_gaze_dim=None).cuda()
    gaze_query = torch.randn(4, 2)
    head_pose_query = torch.randn(4, 2)
    gh_mapper = GazeHeadMapping(128)
    gh_128 = gh_mapper(torch.cat((gaze_query, head_pose_query), dim=-1))
    gh_128 = gh_128.cuda()
    skip_connection = torch.randn(4, 256, 32, 32).cuda()
    x = torch.randn(4, 128, 64, 64).cuda()
    features = torch.randn(4, 128, 64, 64).cuda()
    out = down(x, gaze_head_query=gh_128, stacked_features=features, skip_connection=skip_connection)
    #print(sum(p.numel() for p in down.parameters() if p.requires_grad))
    print(out.shape)

    # RedirectNet
    from models.redirect_net import RedirectNet
    redirect_net = RedirectNet().cuda()
    x_a = torch.randn(4, 3, 128, 128).cuda()
    x_b = torch.randn(4, 3, 128, 128).cuda()
    out = redirect_net(x_a, x_b)
    print(type(out))

    # RedirectNetLoss
    from models.redirect_net import RedirectNetLoss
    redirect_net_loss = RedirectNetLoss().cuda()
    output_dict = {
            'latent_pred': torch.randn(4, 16, 512).cuda(),
            'latent_target': torch.randn(4, 16, 512).cuda(),
            'gaze_pred': torch.rand(4, 2).cuda(),
            'head_pose_pred': torch.rand(4, 2).cuda(),
            'gaze_target': torch.rand(4, 2).cuda(),
            'head_pose_target': torch.rand(4, 2).cuda(),
            'image_pred': torch.randn(4, 3, 256, 256).cuda(),
            'image_target': torch.randn(4, 3, 256, 256).cuda(),
            'id_reference': torch.randn(4, 3, 256, 256).cuda()
        }
    loss, loss_dict = redirect_net_loss(output_dict)
    print(loss)
    print(loss_dict)

    # StyleGAN3 Feature Extractor
    from models.redirect_net_backbone import StyleGAN3FeatureExtractor
    stylegan3_feature_extractor = StyleGAN3FeatureExtractor(path="pretrained/stylegan3-r-ffhqu-256x256.pkl")
    stylegan3_feature_extractor.cuda()
    x = torch.randn(4, 16, 512).cuda()
    layer_out = ['L4_52_1024', 'L9_148_362', 'L12_276_128']
    out, feature_maps = stylegan3_feature_extractor(x, layer_out_list=layer_out)
    print(out.shape)
    print([feature_map.shape for feature_map in feature_maps])

        




if __name__ == "__main__":
    main()