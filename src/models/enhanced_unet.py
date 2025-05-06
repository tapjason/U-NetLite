# models/enhanced_unet.py
import torch
import torch.nn as nn
from .unet import UNet, Down, Up, OutConv
from .convolutions import DoubleConv, LightweightDoubleConv, DepthwiseSeparableConv, GroupedLightweightDoubleConv
from .attention import DoubleConvWithSE, LightweightDoubleConvWithSE, SpatialAttention

class EnhancedUNet(nn.Module):
    """Enhanced U-Net with options for SE blocks and lightweight convolutions"""
    def __init__(self, n_channels, n_classes, use_se=False, use_lightweight=False, 
                 se_reduction=16, bilinear=False):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        if use_se and use_lightweight:
            ConvBlock = lambda in_ch, out_ch, mid_ch=None: LightweightDoubleConvWithSE(
                in_ch, out_ch, mid_ch, reduction=se_reduction)
        elif use_lightweight:
            ConvBlock = LightweightDoubleConv
        else:
            ConvBlock = DoubleConv
        
        # Encoder path
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = Down(64, 128, conv_block=ConvBlock)
        self.down2 = Down(128, 256, conv_block=ConvBlock)
        self.down3 = Down(256, 512, conv_block=ConvBlock)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, conv_block=ConvBlock)
        
        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear, conv_block=ConvBlock)
        self.up2 = Up(512, 256 // factor, bilinear, conv_block=ConvBlock)
        self.up3 = Up(256, 128 // factor, bilinear, conv_block=ConvBlock)
        self.up4 = Up(128, 64, bilinear, conv_block=ConvBlock)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SpatialAttentionUNet(EnhancedUNet):
    """U-Net with spatial attention"""
    def __init__(self, n_channels, n_classes, use_se=True, use_lightweight=True, se_reduction=16, bilinear=False):
        super(SpatialAttentionUNet, self).__init__(
            n_channels, n_classes, use_se, use_lightweight, se_reduction, bilinear
        )
        # Add spatial attention blocks
        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        self.spatial_attention2 = SpatialAttention(kernel_size=7)
        self.spatial_attention3 = SpatialAttention(kernel_size=7)
        self.spatial_attention4 = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x1 = self.spatial_attention1(x1)
        
        x2 = self.down1(x1)
        x2 = self.spatial_attention2(x2)
        
        x3 = self.down2(x2)
        x3 = self.spatial_attention3(x3)
        
        x4 = self.down3(x3)
        x4 = self.spatial_attention4(x4)
        
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UltraLightUNet(nn.Module):
    """Ultra-lightweight U-Net using grouped convolutions"""
    def __init__(self, n_channels, n_classes, base_channels=16, groups=4, bilinear=False):
        super(UltraLightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder with reduced channels and grouped convolutions
        self.inc = GroupedLightweightDoubleConv(n_channels, base_channels, groups=1)
        self.down1 = Down(base_channels, base_channels*2, 
                          conv_block=lambda in_ch, out_ch, mid_ch=None: 
                          GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        self.down2 = Down(base_channels*2, base_channels*4, 
                          conv_block=lambda in_ch, out_ch, mid_ch=None: 
                          GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        self.down3 = Down(base_channels*4, base_channels*8, 
                          conv_block=lambda in_ch, out_ch, mid_ch=None: 
                          GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels*8, base_channels*16//factor, 
                          conv_block=lambda in_ch, out_ch, mid_ch=None: 
                          GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        
        # Decoder
        self.up1 = Up(base_channels*16, base_channels*8//factor, bilinear, 
                      conv_block=lambda in_ch, out_ch, mid_ch=None: 
                      GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        self.up2 = Up(base_channels*8, base_channels*4//factor, bilinear, 
                      conv_block=lambda in_ch, out_ch, mid_ch=None: 
                      GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        self.up3 = Up(base_channels*4, base_channels*2//factor, bilinear, 
                      conv_block=lambda in_ch, out_ch, mid_ch=None: 
                      GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        self.up4 = Up(base_channels*2, base_channels, bilinear, 
                      conv_block=lambda in_ch, out_ch, mid_ch=None: 
                      GroupedLightweightDoubleConv(in_ch, out_ch, mid_ch, groups=groups))
        
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits