# models/convolutions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Standard double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for lightweight models"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightDoubleConv(nn.Module):
    """Double convolution block using depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class GroupedLightweightDoubleConv(nn.Module):
    """Double convolution block using group convolutions and depthwise separable convs"""
    def __init__(self, in_channels, out_channels, mid_channels=None, groups=4):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # First depthwise + grouped pointwise conv
        self.conv1_depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, 
            groups=in_channels, bias=False
        )
        self.conv1_pointwise = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, 
            groups=min(groups, in_channels), bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Second depthwise + grouped pointwise conv
        self.conv2_depthwise = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, 
            groups=mid_channels, bias=False
        )
        self.conv2_pointwise = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, 
            groups=min(groups, mid_channels), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x