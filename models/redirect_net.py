import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.gaze_estimation_net as gaze_estimation_net
import models.inversion_encoder.networks_encoder as inversion_encoder
from models.redirect_net_backbone import GazeHeadMapping, UNet_Up, UNet_Down, StyleGAN3FeatureExtractor
from utils.losses import GazeHeadLoss
from torchvision import transforms
import dnnlib
import legacy
from lpips import LPIPS
from utils.loss_encoder import IDLoss

class RedirectNet(nn.Module):
    def __init__(self,
                generator="pretrained/stylegan3-r-ffhqu-256x256.pkl",
                inversion_encoder=inversion_encoder.Encoder,
                inversion_encoder_checkpoint="pretrained/inversion_encoder_snapshot_100000.pkl",
                gaze_estimator=gaze_estimation_net.VGG_Gaze_Estimator,
                gaze_estimator_checkpoint="pretrained/at_step_0069999_vggFreezeFalse.pth.tar",
                use_stylegan3_feature=True
                ):
        super(RedirectNet, self).__init__()


        # -------------------- Backbone Network --------------------
        # Initialize Gaze / Head Pose Estimator
        self.GazeEstimator = gaze_estimator()
        self.GazeEstimator.load_state_dict(torch.load(gaze_estimator_checkpoint), strict=True)
        self.GazeEstimator.eval()
        self.GazeEstimator.requires_grad_(False)
        # Initialize Inversion Encoder
        self.InversionEncoder = inversion_encoder(pretrained=inversion_encoder_checkpoint, w_avg=None)
        self.InversionEncoder.eval()
        self.InversionEncoder.requires_grad_(False)
        # Initialize StyleGAN3 Generator
        # with dnnlib.util.open_url(generator) as f:
        #     self.Generator = legacy.load_network_pkl(f)['G_ema'].eval()
        # self.Generator.requires_grad_(False) # api: G.synthesis(batched_latents)
        self.Generator = StyleGAN3FeatureExtractor(path=generator)
        # -------------------- Redirect Network --------------------
        self.gaze_head_mapping_512 = GazeHeadMapping(512)
        self.gaze_head_mapping_256 = GazeHeadMapping(256)
        self.gaze_head_mapping_128 = GazeHeadMapping(128)
        #self.gaze_head_mapping_64 = GazeHeadMapping(64)

        ## stylegan3 feature map conv
        self.gan_conv_512x4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=5, stride=3, padding=0),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.gan_conv_512x16 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, stride=3, padding=0),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.gan_conv_256x32 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=3, padding=0),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.gan_conv_128x64 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, stride=3, padding=0),
            nn.AdaptiveAvgPool2d((64, 64))
        )


        # init 512 gaze head mapping with channel-wise mean, std
        self.channel_mean = torch.load("utils/mean_std.pt")["channel_mean"]
        self.channel_std = torch.load("utils/mean_std.pt")["channel_std"]
        self.mean_latent = torch.load("utils/mean_std.pt")["mean_latent"]
        self.mean_latent.requires_grad_(False).cuda()
        self.k = nn.Parameter(torch.ones(512), requires_grad=True).cuda()

        self.latent_entry = UNet_Up(512, 512,
                                    entry_block=True,
                                    embed_gaze_dim=4,
                                    embed_dim=16,
                                    attn_q_dim=512+512+512 # TODO: avoid hard coded
                                    ) 
        self.up1 = UNet_Up(512, 256,
                            embed_gaze_dim=4,
                            embed_dim=16*16,
                            attn_q_dim=512+512+256
                            )
        self.up2 = UNet_Up(256, 128, embed_gaze_dim=4, embed_dim=32*32,
                           attn_q_dim=256+256+128
                           )
        #self.up3 = UNet_Up(128, 64) #
        #self.up4 = UNet_Up(64, 32) #
        # self.bottle_neck = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        #self.down1 = UNet_Down(32, 64)
        #self.down2 = UNet_Down(64, 128)
        self.down1 = UNet_Down(128, 256, embed_dim=64*64, embed_gaze_dim=4,
                                 attn_q_dim=128+128+64, teacher_forcing=True)
        self.down2 = UNet_Down(256, 512, embed_gaze_dim=4, embed_dim=32*32,
                                    attn_q_dim=256+256+128, teacher_forcing=True)
        self.latent_exit = UNet_Down(512, 512, exit_block=True, embed_gaze_dim=4, embed_dim=16*16,
                                        attn_q_dim=512+512+256, teacher_forcing=True)
        self.use_stylegan3_feature = use_stylegan3_feature

    def forward(self, x_a, x_b, teacher_forcing=0.5):
        '''
        x_a: source image (batch_size, 3, 128, 128)
        g_a: source gaze vector (batch_size, 2)
        h_a: source head pose vector (batch_size, 2)
        x_b: target image (batch_size, 3, 128, 128)
        g_b: target gaze vector (batch_size, 2)
        h_b: target head pose vector (batch_size, 2)
        '''
        # Resize to 256x256
        t_256 = transforms.Resize((256, 256), antialias=True)
        x_a = t_256(x_a)
        x_b = t_256(x_b)
        #print(f"x_a: {x_a.shape}")
        # get latent code from source image and target image
        w_a = self.InversionEncoder(x_a) # (batch_size, 16, 512)
        w_b = self.InversionEncoder(x_b) # target

        # get gaze and head pose vector from source image and target image
        x_a_hat = self.Generator.synthesis(w_a) # (batch_size, 3, 256, 256)
        x_b_hat = self.Generator.synthesis(w_b)

        # transform to 128x128 for gaze estimation
        t_128 = transforms.Resize((128, 128), antialias=True)
        g_a, h_a, stacked_features_a = self.GazeEstimator(t_128(x_a_hat), feature_out_layers=[15, 22 ,30])
        #print([f.shape for f in stacked_features_a])
        g_b, h_b, _ = self.GazeEstimator(x_b_hat, feature_out_layers=None)

        #feature_a_64x128 = stacked_features_a[0]
        #feature_a_128x64 = stacked_features_a[1]
        feature_a_256x32 = stacked_features_a[0]
        feature_a_512x16 = stacked_features_a[1]
        feature_a_512x4 = stacked_features_a[2]
        #print(f"feature_a_512x4: {feature_a_512x4.shape}")
        
        # stylegan3 feature extract
        if self.use_stylegan3_feature:
            layer_out = ['L4_52_1024', 'L8_148_512', 'L10_276_256']
            _, (feature_gan_512x4, feature_gan_512x16, feature_gan_256x32) = self.Generator(w_a, layer_out_list=layer_out)
            feature_gan_512x4, feature_gan_512x16, feature_gan_256x32 = feature_gan_512x4.to(torch.float32), feature_gan_512x16.to(torch.float32), feature_gan_256x32.to(torch.float32)
            feature_gan_512x4 = self.gan_conv_512x4(feature_gan_512x4)
            feature_gan_512x16 = self.gan_conv_512x16(feature_gan_512x16)
            feature_gan_256x32 = self.gan_conv_256x32(feature_gan_256x32)
            concat_not_add = True
        else:
            feature_gan_512x4 = None
            feature_gan_512x16 = None
            feature_gan_256x32 = None
            concat_not_add = False # assumed when not using stylegan3 features

        # remap gaze and head pose vector
        gh_a = torch.cat([g_a, h_a], dim=-1)
        gh_b = torch.cat([g_b, h_b], dim=-1)
        gh_a_512 = self.gaze_head_mapping_512(gh_a) # (batch_size, 512, 2, 2)
        gh_a_256 = self.gaze_head_mapping_256(gh_a)
        #gh_a_128 = self.gaze_head_mapping_128(gh_a)
        #gh_a_64 = self.gaze_head_mapping_64(gh_a)

        gh_b_512 = self.gaze_head_mapping_512(gh_b)
        gh_b_256 = self.gaze_head_mapping_256(gh_b)
        gh_b_128 = self.gaze_head_mapping_128(gh_b)
        #gh_b_64 = self.gaze_head_mapping_64(gh_b)

        # U-Net Upsampling
        
        latent_16x16 = self.latent_entry(w_a.permute(0, 2, 1).view(w_a.size()[0], 512, 4, 4), gaze_head_query=gh_a_512,
                                         stacked_features=feature_a_512x4,
                                         concat_not_add=concat_not_add,
                                         stylegan3_features=feature_gan_512x4)

                                         
        latent_32x32 = self.up1(latent_16x16, gaze_head_query=gh_a_512,
                                        stacked_features=feature_a_512x16,
                                        concat_not_add=concat_not_add,
                                        stylegan3_features=feature_gan_512x16)
        latent_64x64 = self.up2(latent_32x32, gaze_head_query=gh_a_256,
                                        stacked_features=feature_a_256x32,
                                        concat_not_add=concat_not_add,
                                        stylegan3_features=feature_gan_256x32)
        # latent_128x128 = self.up3(latent_64x64, gaze_head_query=gh_a_128,
        #                                 stacked_features=feature_a_128x64)
        # latent_256x256 = self.up4(latent_128x128, gaze_query=gh_a_64,
        #                                 stacked_features=feature_a_64x128)
        #latent_64x64 = self.bottle_neck(latent_64x64)        
        # U-Net Downsampling with skip connections
        # latent_out = self.down1(latent_256x256, gaze_head_query=gh_b_64,
        #                                 stacked_features=None, skip_connection=latent_128x128)
        # latent_out = self.down2(latent_out, gaze_head_query=gh_b_128,
        #                                 stacked_features=None, skip_connection=latent_64x64)
        if teacher_forcing > np.random.random():

            _, _, stacked_features_b = self.GazeEstimator(t_128(x_b_hat), feature_out_layers=[8, 15, 22])
            feature_b_128x64 = stacked_features_b[0]
            feature_b_256x32 = stacked_features_b[1]
            feature_b_512x16 = stacked_features_b[2]

            
            # stylegan3 feature extract
            if self.use_stylegan3_feature:
                layer_out = ['L4_52_1024', 'L8_148_512', 'L10_276_256']
                _, (_, feature_gan_512x16_b, feature_gan_256x32_b) = self.Generator(w_b, layer_out_list=layer_out)
                feature_gan_512x16_b, feature_gan_256x32_b = feature_gan_512x16_b.to(torch.float32), feature_gan_256x32_b.to(torch.float32)
                #print(f"shape of feature_gan_512x16_b: {feature_gan_512x16_b.shape}")
                #print(f"shape of feature_gan_256x32_b: {feature_gan_256x32_b.shape}")
                #feature_gan_512x4 = self.gan_conv_512x4(feature_gan_512x4)
                feature_gan_512x16_b = self.gan_conv_512x16(feature_gan_512x16_b)
                feature_gan_256x32_b_cp = feature_gan_256x32_b.clone()
                feature_gan_256x32_b = self.gan_conv_256x32(feature_gan_256x32_b)
                feature_gan_128x64_b = self.gan_conv_128x64(feature_gan_256x32_b_cp)
                concat_not_add = True
            else:
                #feature_gan_512x4 = None
                feature_gan_512x16_b = None
                feature_gan_256x32_b= None
                feature_gan_128x64_b = None
                concat_not_add = False # assumed when not using stylegan3 features

        else:
            feature_b_128x64 = None
            feature_b_256x32 = None
            feature_b_512x16 = None
            concat_not_add = False
            feature_gan_512x16_b = None
            feature_gan_256x32_b = None
            feature_gan_128x64_b = None

        latent_out = self.down1(latent_64x64, gaze_head_query=gh_b_128,
                                        stacked_features=feature_b_128x64, skip_connection=latent_32x32,
                                        stylegan3_features=feature_gan_128x64_b, concat_not_add=concat_not_add)
        #print(f"latent_out shape: {latent_out.shape}")
        latent_out = self.down2(latent_out, gaze_head_query=gh_b_256,
                                        stacked_features=feature_b_256x32, skip_connection=latent_16x16,
                                        stylegan3_features=feature_gan_256x32_b, concat_not_add=concat_not_add)
        latent_out = self.latent_exit(latent_out, gaze_head_query=gh_b_512,
                                        stacked_features=feature_b_512x16, skip_connection=w_a.permute(0, 2, 1).view(w_a.size()[0], 512, 4, 4),
                                        stylegan3_features=feature_gan_512x16_b, concat_not_add=concat_not_add)
        latent_out = latent_out.view(latent_out.size()[0], 512, 16).permute(0, 2, 1) # (B, 16, 512)

        latent_out = self.mean_latent - self.k * (latent_out - self.mean_latent)

        #latent_out = latent_out + w_a
        # estimate gaze and head pose vector from latent code
        x_b_pred = self.Generator.synthesis(latent_out)

        g_b_pred, h_b_pred, _ = self.GazeEstimator(t_128(x_b_pred), feature_out_layers=None)
        return {
            'latent_pred': latent_out, # (B, 16, 512)
            'latent_target': w_b, # (B, 16, 512)
            'gaze_pred': g_b_pred, # (B, 2)
            'head_pose_pred': h_b_pred, # (B, 2)
            'gaze_target': g_b, # (B, 2)
            'head_pose_target': h_b, # (B, 2)
            'image_pred': x_b_pred, # (B, 3, 256, 256)
            'image_target': x_b_hat, # (B, 3, 256, 256) encoded, not original
            'id_reference': x_a_hat, # (B, 3, 256, 256)
            'latent_reference': w_a # (B, 3, 256, 256)
        }
    

class RedirectNetLoss(nn.Module):
    def __init__(self, 
                 lambda_gaze=1, 
                 lambda_head_pose=1, 
                 lambda_direct_latent=1, 
                 lambda_image_reconstruction=1, 
                 lambda_perceptual=1,
                 lambda_identity=1):
        super().__init__()
        self.lambda_gaze = lambda_gaze
        self.lambda_head_pose = lambda_head_pose
        self.lambda_direct_latent = lambda_direct_latent
        self.lambda_image_reconstruction = lambda_image_reconstruction
        self.lambda_perceptual = lambda_perceptual
        self.lambda_identity = lambda_identity

        self.gaze_loss = GazeHeadLoss()
        self.head_pose_loss = GazeHeadLoss()
        self.direct_latent_loss = nn.SmoothL1Loss()
        self.image_reconstruction_loss = nn.SmoothL1Loss()
        self.perceptual_loss = LPIPS(net='alex', verbose=False)
        self.identity_loss = IDLoss()

    def forward(self, output_dict):
        gaze_loss = self.gaze_loss(output_dict['gaze_pred'], output_dict['gaze_target'])
        head_pose_loss = self.head_pose_loss(output_dict['head_pose_pred'], output_dict['head_pose_target'])
        direct_latent_loss = self.direct_latent_loss(output_dict['latent_pred'], output_dict['latent_target'])
        image_reconstruction_loss = self.image_reconstruction_loss(output_dict['image_pred'], output_dict['image_target'])
        perceptual_loss = self.perceptual_loss(output_dict['image_pred'], output_dict['image_target']).squeeze().mean()
        identity_loss, id_improve = self.identity_loss(output_dict['image_pred'], output_dict['image_target'], output_dict['id_reference'])

        loss = self.lambda_gaze * gaze_loss + \
               self.lambda_head_pose * head_pose_loss + \
               self.lambda_direct_latent * direct_latent_loss + \
               self.lambda_image_reconstruction * image_reconstruction_loss + \
               self.lambda_perceptual * perceptual_loss + \
               self.lambda_identity * identity_loss

        return loss, {
            'gaze_loss': gaze_loss * self.lambda_gaze,
            'head_pose_loss': head_pose_loss * self.lambda_head_pose,
            'direct_latent_loss': direct_latent_loss * self.lambda_direct_latent,
            'image_reconstruction_loss': image_reconstruction_loss * self.lambda_image_reconstruction,
            'perceptual_loss': perceptual_loss * self.lambda_perceptual,
            'identity_loss': identity_loss * self.lambda_identity,
            'id_improve': id_improve
        }
