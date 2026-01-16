import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
PyTorch model definition for the colorization model.
Based on ResNet-34 encoder and a U-Net-like decoder.
"""


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # 1x1 Convs to reduce dimensions for efficiency
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Learnable scale parameter, initialized to 0 to start as Identity
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        # Query: B x (C//8) x N where N=H*W
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # Key: B x (C//8) x N
        proj_key = self.key_conv(x).view(B, -1, H * W)
        
        # Energy: B x N x N (Relationship between every pixel to every pixel)
        energy = torch.bmm(proj_query, proj_key) # Matrix mult
        attention = self.softmax(energy) # Attention map
        
        # Value: B x C x N
        proj_value = self.value_conv(x).view(B, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

class ColorizationModel(nn.Module):
    def __init__(self, num_classes=313):
        super(ColorizationModel, self).__init__()
        
        # Encoder: ResNet-34
        # Using pretrained weights
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify first layer to accept 1 channel (L) instead of 3 (RGB)
        
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
        
        # Self-Attention at bottleneck (512 channels)
        self.attention = SelfAttention(512)
        
        # Decoder
        # Skip connections from layers 1, 2, 3
        
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1, bias=False), # +256 from layer3 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1, bias=False), # +128 from layer2 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1, bias=False), # +64 from layer1 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        """
        Decoder architecture:
        up4: H/32 -> H/16 (concat with layer3)
        up3: H/16 -> H/8  (concat with layer2)
        up2: H/8  -> H/4  (concat with layer1)
        up1: H/4  -> H/2  (concat with c1)
        """
        self.refine = nn.Sequential(
            # Layer 1: Dilated Conv (dilation=4) for expanded receptive field
            nn.Conv2d(128, 64, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 2: Standard Conv to smooth features (Output 64 channels to match final_conv)
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # On-the-fly normalization
        # x comes in as (B, 1, H, W) in range 0..1.
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
        
        # Apply Self-Attention
        l4 = self.attention(l4)
        
        
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
