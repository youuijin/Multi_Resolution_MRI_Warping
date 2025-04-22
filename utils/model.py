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
    
    def forward(self, x, return_feature=False):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.dec1(x4)
        x6 = self.dec2(torch.cat([x3, x5], dim=1))
        x7 = self.dec3(torch.cat([x2, x6], dim=1))
        x8 = self.dec4(torch.cat([x1, x7], dim=1))
        if return_feature:
            return x8, x2
        return x8

# MrRegNet
class MrReg(nn.Module):
    def __init__(self, in_channels=2, num_levels=3, int_steps=1):
        super().__init__()
        self.encoder = Encoder(in_channels, num_levels)
        self.decoder = Decoder(num_levels, int_steps)

    def forward(self, x):
        # x = torch.cat([source, target], dim=1)
        x, features = self.encoder(x)
        disps, disps_res = self.decoder(x, features)
        # print(disps)
        return disps[0][0]
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_levels):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_levels):
            block = ResidualBlock(current_channels, 32)
            self.blocks.append(block)
            if i < num_levels - 1:
                self.downsamples.append(nn.Conv3d(32, 32, kernel_size=1, stride=2))
            current_channels = 32

    def forward(self, x):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        # return x, features
        return x


class Decoder(nn.Module):
    def __init__(self, num_levels, int_steps):
        super().__init__()
        self.num_levels = num_levels
        self.int_steps = int_steps
        self.deconvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.flow_predictors = nn.ModuleList()
        self.rescaler = ResizeTransform(0.5, ndims=3)

        for i in range(num_levels):
            # self.deconvs.append(nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
            # self.convs.append(nn.Conv3d(64, 32, kernel_size=3, padding=1))
            # self.flow_predictors.append(nn.Conv3d(32, 3, kernel_size=3, padding=1))
            if i > 0:
                self.deconvs.append(nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
                self.convs.append(nn.Conv3d(64, 32, kernel_size=3, padding=1))
            else:
                self.deconvs.append(None)
                self.convs.append(nn.Conv3d(32, 32, kernel_size=3, padding=1))
            self.flow_predictors.append(nn.Conv3d(32, 3, kernel_size=3, padding=1))

    def forward(self, x, enc_feats):
        disps = []
        disps_res = []
        prev_disp = None

        for i in range(self.num_levels):
            if i > 0:
                x = self.deconvs[i](x)
                x = F.leaky_relu(x, 0.2)
                x = torch.cat([x, enc_feats[-i - 1]], dim=1)
            x = F.leaky_relu(self.convs[i](x), 0.2)

            flow = self.flow_predictors[i](x)
            if prev_disp is not None:
                prev_disp = self.rescaler(prev_disp) + flow
            else:
                prev_disp = flow

            flow_pos = prev_disp
            flow_neg = -flow_pos

            if self.int_steps > 0:
                vecint = VecInt(flow_pos.shape[2:], self.int_steps)
                flow_pos = vecint(flow_pos)
                flow_neg = vecint(flow_neg)

            disps.append([flow_pos, flow_neg])
            disps_res.append([flow, -flow])

        return disps, disps_res

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)


    def forward(self, src, flow):

        # new locations
        grid = self.grid.to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        wrapped = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode="border")

        return wrapped

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        if type(vel_resize) in [list, tuple]:
            self.factor = tuple([1.0/x for x in vel_resize])
        else:
            self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if type(self.factor) is tuple:
                x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
                for i in range(x.shape[1]):
                    x[:,i,...] = x[:,i,...] * self.factor[i]
        elif self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
    
class FusionWeightNet(nn.Module):
    def __init__(self, in_channels=109, hidden_channels=16, out_channels=3):
        super(FusionWeightNet, self).__init__()

        self.compress = nn.Conv3d(in_channels, 32, kernel_size=1) # feature compression

        self.encoder = nn.Sequential(
            nn.Conv3d(32, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.output_conv = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)  # Output: (B, 3, H, W, D)

    def preprocessing(self, img, phis, features):
        phi1, phi2, phi3 = phis
        phi1 = F.interpolate(phi1, size=phi3.shape[2:], mode='trilinear', align_corners=False) # upsample
        phi2 = F.interpolate(phi2, size=phi3.shape[2:], mode='trilinear', align_corners=False) # upsample

        edge_map = self.compute_edge_map(img)

        mag1, mag2, mag3 = self.compute_magnitude(phi1), self.compute_magnitude(phi2), self.compute_magnitude(phi3)

        feat1, feat2, feat3 = features
        feat1 = F.interpolate(feat1, size=phi3.shape[2:], mode='trilinear', align_corners=False) # upsample
        feat2 = F.interpolate(feat2, size=phi3.shape[2:], mode='trilinear', align_corners=False) # upsample
        feat3 = F.interpolate(feat3, size=phi3.shape[2:], mode='trilinear', align_corners=False) # upsample

        fusion_input = torch.cat([phi1, phi2, phi3, mag1, mag2, mag3, edge_map, feat1, feat2, feat3], dim=1)
        # fusion_input = torch.cat([phi1, phi2, phi3, edge_map, feat1, feat2, feat3]) # TODO: not using magnitude
        return fusion_input

    def compute_edge_map(self, image):
        """Compute simple edge map via gradient magnitude"""
        gx = torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :])
        gy = torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :])
        gz = torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1])
        edge = torch.zeros_like(image)
        edge[:, :, 1:, :, :] += gx
        edge[:, :, :, 1:, :] += gy
        edge[:, :, :, :, 1:] += gz
        # normalize to [0, 1]
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-5)
        return edge
    
    def compute_magnitude(self, phi):
        """Compute magnitude of displacement field φ"""
        return torch.norm(phi, dim=1, keepdim=True)

    def forward(self, img, phis, features):
        """
        x: (B, in_channels=7, H, W, D)
        Returns:
            weights: (B, 3, H, W, D), softmax along channel dimension
        """
        x = self.preprocessing(img, phis, features)
        x = self.compress(x) # adding feature compression
        feat = self.encoder(x)
        weights = self.output_conv(feat)
        weights = F.softmax(weights, dim=1)
        return weights

if __name__ == "__main__":
    model = MrReg().cuda()

    torchsummary.summary(model, (2, int(192), int(224), int(192)))