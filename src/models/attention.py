# models/attention.py
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        
        # Apply attention
        return x * attention


class DoubleConvWithSE(nn.Module):
    """Double convolution block with Squeeze-and-Excitation"""
    def __init__(self, in_channels, out_channels, mid_channels=None, reduction=16):
        super().__init__()
        from .convolutions import DoubleConv
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)
        return x


class LightweightDoubleConvWithSE(nn.Module):
    """Lightweight double convolution block with Squeeze-and-Excitation"""
    def __init__(self, in_channels, out_channels, mid_channels=None, reduction=16):
        super().__init__()
        from .convolutions import LightweightDoubleConv
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv = LightweightDoubleConv(in_channels, out_channels, mid_channels)
        self.se = SEBlock(out_channels, reduction=reduction)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x