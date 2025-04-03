import torch
import torch.nn as nn
import torchsummary
import numpy as np
import torch.nn.functional as F

class BigUNet3D(nn.Module):
    def __init__(self):
        super(BigUNet3D, self).__init__()

        # TODO: stacked (paired) input
        
        # Encoder (축소된 채널 수)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(2)
        # Bottleneck (중간 병목 레이어)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64+32, 32, kernel_size=3, padding=1),  # Skip connection 적용
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(16+8, 8, kernel_size=3, padding=1),  # Skip connection 적용
            nn.ReLU(),
            nn.Conv3d(8, 3, kernel_size=3, padding=1)  # 최종 출력 (3채널)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)  # Output: (batch, 16, H, W, D)
        x = self.pool1(enc1)     # Downsample

        enc2 = self.encoder2(x)  # Output: (batch, 64, H/2, W/2, D/2)
        x = self.pool2(enc2)     # Downsample

        # Bottleneck
        x = self.bottleneck(x)   # Output: (batch, 64, H/4, W/4, D/4)

        # Decoding path
        x = self.upconv2(x)      # Upsample to (batch, 32, H/2, W/2, D/2)
        
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.decoder2(x)

        x = self.upconv1(x)      # Upsample to (batch, 8, H, W, D)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.decoder1(x)

        return x  # Output shape: (batch_size, 3, H, W, D)
    
class UNet_2input(nn.Module):
    def __init__(self):
        super(UNet_2input, self).__init__()
        
        # Encoder (축소된 채널 수)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(2)
        # Bottleneck (중간 병목 레이어)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64+32, 32, kernel_size=3, padding=1),  # Skip connection 적용
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(16+8, 8, kernel_size=3, padding=1),  # Skip connection 적용
            nn.ReLU(),
            nn.Conv3d(8, 3, kernel_size=3, padding=1)  # 최종 출력 (3채널)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)  # Output: (batch, 16, H, W, D)
        x = self.pool1(enc1)     # Downsample

        enc2 = self.encoder2(x)  # Output: (batch, 64, H/2, W/2, D/2)
        x = self.pool2(enc2)     # Downsample

        # Bottleneck
        x = self.bottleneck(x)   # Output: (batch, 64, H/4, W/4, D/4)

        # Decoding path
        x = self.upconv2(x)      # Upsample to (batch, 32, H/2, W/2, D/2)
        
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.decoder2(x)

        x = self.upconv1(x)      # Upsample to (batch, 8, H, W, D)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.decoder1(x)

        return x  # Output shape: (batch_size, 3, H, W, D)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class VoxelMorphUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(VoxelMorphUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 32)
        self.enc4 = ConvBlock(32, 32)
        self.enc5 = ConvBlock(32, 32)
        
        # Decoder
        self.dec4 = ConvBlock(32 + 32, 32)
        self.dec3 = ConvBlock(32 + 32, 32)
        self.dec2 = ConvBlock(32 + 32, 32)
        self.dec1 = ConvBlock(16 + 32, 16)
        self.dec0 = ConvBlock(16, 16)
        self.dec_final = ConvBlock(16, 16)
        
        self.flow = nn.Conv3d(16, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x = torch.cat([moving, fixed], dim=1)
        
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        e5 = self.enc5(F.max_pool3d(e4, 2))
        # e5 = self.enc5(e4)
        
        # Decoding path
        d4 = F.interpolate(e5, scale_factor=2, mode='trilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = F.interpolate(d4, scale_factor=2, mode='trilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        d0 = self.dec0(d1)
        d_final = self.dec_final(d0)
        
        flow = self.flow(d_final)
        return flow
    
class R2Net_UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super(R2Net_UNet, self).__init__()
        
        # Encoder (논문 기반 채널 수 적용)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder (Up-sampling & Skip Connection 적용)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.Conv3d(16+16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
			nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
		)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.dec1(x4)
        x6 = self.dec2(torch.cat([x3, x5], dim=1))
        x7 = self.dec3(torch.cat([x2, x6], dim=1))
        x8 = self.dec4(torch.cat([x1, x7], dim=1))
        return x8

if __name__ == "__main__":
    model = R2Net_UNet().cuda()

    torchsummary.summary(model, (2, int(192/4), int(224/4), int(192/4)))