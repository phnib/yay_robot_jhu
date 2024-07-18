# This file contains the model for the EfficientViT AutoEncoder adjusted for further use

import os
import sys

import torch
import torch.nn as nn

path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

from instructor.submodules.gsvit_submodule.gsvit.EfficientViT.classification.model.build import EfficientViT_M5

def process_inputs(images):    
    """
    Flip color channels, e.g., from RGB to BGR
    
    Args:
        images (torch.Tensor): Input images
    
    Returns:
        images (torch.Tensor): Images with flipped color channels
    """
    
    tmp = images[:, 0, :, :].clone()
    images[:, 0, :, :] = images[:, 2, :, :]
    images[:, 2, :, :] = tmp
    return images


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Decoder(nn.Module):
    def __init__(self, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size

        # Initial representation
        self.fc = nn.Linear(384*4*4, 7 * 7 * 1024)
        self.bn1d = nn.BatchNorm1d(7 * 7 * 1024)
        self.gelu = nn.GELU()

        # Decoder layers
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn2 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()
        #self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn3 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        #self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn4 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

        # Residual blocks with SE attention
        self.res2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            SEAttention(64),
            nn.ReLU()
        )

        # was 256
        self.res1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            SEAttention(512),
            nn.ReLU()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x.reshape(self.batch_size, 384*4*4))
        x = self.bn1d(x)
        x = self.gelu(x)
        x = x.view(-1, 1024, 7, 7)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.res1(x) + x
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.res2(x) + x
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


class EfficientViTAutoEncoder(nn.Module):
    def __init__(self, batch_size=16):
        super(EfficientViTAutoEncoder, self).__init__()
        self.decoder = Decoder(batch_size)
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # remove the classification head
        self.evit = torch.nn.Sequential(*list(self.evit.children())[:-1])

    def forward(self, x):
        x = process_inputs(x)
        out = self.evit(x)
        decoded = self.decoder.forward(out)
        decoded = process_inputs(decoded)
        return decoded
