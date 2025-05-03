# unet_3d.py
class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        super().__init__()
        self.encoder1 = self.conv_block_3d(n_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self.conv_block_3d(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self.conv_block_3d(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bottleneck = self.conv_block_3d(128, 256)
        
        self.upconv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.decoder3 = self.conv_block_3d(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.decoder2 = self.conv_block_3d(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.decoder1 = self.conv_block_3d(64, 32)
        
        self.out = nn.Conv3d(32, n_classes, 1)
    
    def conv_block_3d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.decoder3(torch.cat([self.upconv3(b), e3], 1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], 1))
        
        return self.out(d1)

# enhanced_unet_3d.py
class EnhancedUNet3D(nn.Module):
    # Similar to UNet3D but with:
    # 1. Squeeze-and-excitation blocks
    # 2. Depthwise separable convolutions
    # 3. Lightweight layers
    pass