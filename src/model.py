import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
PyTorch model definition for the colorization model.
Based on ResNet-34 encoder and a U-Net-like decoder.
"""

class ColorizationModel(nn.Module):
    def __init__(self, num_classes=313):
        super(ColorizationModel, self).__init__()
        
        # Encoder: ResNet-34
        # Using pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify first layer to accept 1 channel (L) instead of 3 (RGB)
        # Averaging the weights of the original.
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # 64 channels
        self.layer2 = resnet.layer2 # 128 channels
        self.layer3 = resnet.layer3 # 256 channels
        self.layer4 = resnet.layer4 # 512 channels
        
        # Decoder
        # Skip connections from layers 1, 2, 3
        
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1), # +256 from layer3 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1), # +128 from layer2 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), # +64 from layer1 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        """
        Final layers to get to original resolution and num_classes
        After up1, we are at H/2, W/2 (because ResNet starts with stride 2 conv + stride 2 maxpool = /4)
        ResNet structure:
        Input: H, W
        conv1+pool: H/4, W/4
        layer1: H/4, W/4 (64)
        layer2: H/8, W/8 (128)
        layer3: H/16, W/16 (256)
        layer4: H/32, W/32 (512)
        
        Decoder:
        up4 (from layer4): H/16, W/16 (256)
        concat layer3 (256): 512 -> up3 -> H/8, W/8 (128)
        concat layer2 (128): 256 -> up2 -> H/4, W/4 (64)
        concat layer1 (64): 128 -> up1 -> H/2, W/2 (64)
        
        One more upsample to get to H, W
        Skip connection from conv1 (which has 64 channels)
        Input to up0 is 64 (from up1) + 64 (from conv1) = 128
        Refinement Block (extra smoothing/sharpening before final classification)
        Refinement Block (Dilated Convolution for Context)
        Input comes from u1 (64 from up1 + 64 from c1 skip) = 128 channels
        """
        self.refine = nn.Sequential(
            # Layer 1: Dilated Conv (dilation=4) for expanded receptive field (Context Expansion Strategy)
            nn.Conv2d(128, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 2: Standard Conv to smooth features (Output 64 channels to match final_conv)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # On-the-fly normalization
        # x comes in as (B, 1, H, W) in range 0..100.
        # Reduce range to [-1, 1] for ResNet.
        x = (x - 0.5) / 0.5
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)        # (B, 64, H/2, W/2) -> Save for skip
        x_pool = self.maxpool(c1) # H/4
        
        l1 = self.layer1(x_pool) # H/4
        l2 = self.layer2(l1)     # H/8
        l3 = self.layer3(l2)     # H/16
        l4 = self.layer4(l3)     # H/32
        
        # Decoder
        u4 = self.up4(l4)        # H/16
        u4 = torch.cat([u4, l3], dim=1)
        
        u3 = self.up3(u4)        # H/8
        u3 = torch.cat([u3, l2], dim=1)
        
        u2 = self.up2(u3)        # H/4
        u2 = torch.cat([u2, l1], dim=1)
        
        u1 = self.up1(u2)        # H/2
        
        # Skip connection from c1
        u1 = torch.cat([u1, c1], dim=1) 
        
        # Stops at 112x112 (u1)

        # Refinement
        u0_refined = self.refine(u1)
        
        out = self.final_conv(u0_refined) 
        return out
