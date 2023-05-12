import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# import VGG model
from torchvision import models

class VGG_Gaze_Estimator(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG_Gaze_Estimator, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # discard FC layers
        self.vgg16 = self.vgg16.features

        self.FC1 = nn.Linear(512, 64, bias=True)
        self.FC2 = nn.Linear(64, 64, bias=True)
        self.FC3 = nn.Linear(64, 4, bias=True)

        self.leakly_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        # initialize weights
        nn.init.kaiming_normal_(self.FC1.weight.data)
        nn.init.constant_(self.FC1.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC2.weight.data)
        nn.init.constant_(self.FC2.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC3.weight.data)
        nn.init.constant_(self.FC3.bias.data, val=0)

    def forward(self, x, feature_out_layers: list = None):
        features = []
        for i, layer in enumerate(self.vgg16):
            x = layer(x)
            if feature_out_layers is None: continue
            if i in feature_out_layers:
                features.append(x)
        #print(x.shape)
        x = x.mean(-1).mean(-1) # global average pooling
        #print(x.shape, "after global average pooling")
        x = self.leakly_relu(self.FC1(x))
        x = self.leakly_relu(self.FC2(x))
        x = self.tanh(self.FC3(x))
        x = torch.pi * x * 0.5
        gaze_estimate = x[:, :2]
        head_estimate = x[:, 2:]
        return gaze_estimate, head_estimate, features \
            if feature_out_layers is not None \
            else None



