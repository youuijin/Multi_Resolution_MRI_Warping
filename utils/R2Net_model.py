import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import numpy as np
    
class U_Net_Hyprid(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, out_layers=1):
        super(U_Net_Hyprid, self).__init__()

        assert out_channels in [3, 6]
        assert out_layers in [1, 2, 3, 4]

        self.out_layers = out_layers
        self.out_channels = out_channels
        
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
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Decoder (Up-sampling & Skip Connection 적용)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.Conv3d(16+16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
		)

        self.flows = nn.ModuleList([nn.Identity() for _ in range(4)])
        for i, res in enumerate([32, 32, 32, 8]):
            if 4 - self.out_layers <= i:
                self.flows[i] = nn.Conv3d(res, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # decoding
        if self.out_channels == 6:
            means = [] # only residual
            stds = [] # model output = log \sigma^2 -> exp(0.5*log \sigma^2)
            x5 = self.dec1(x4)
            concated = self.flows[0](x5)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:])) # output = log variance
            up_x5 = self.upsample1(x5)
            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            concated = self.flows[1](x6)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))
            up_x6 = self.upsample2(x6)
            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            concated = self.flows[2](x7)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))
            up_x7 = self.upsample3(x7)
            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            concated = self.flows[3](x8)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))

            means, stds = means[-self.out_layers:], stds[-self.out_layers:]

            tot_means = self.combine_residuals(means)
            tot_stds = self.combine_residuals(stds)
            
            return tot_means, tot_stds, means, stds
        
        elif self.out_channels == 3:
            disp = []
            x5 = self.dec1(x4)
            concated = self.flows[0](x5)
            disp.append(concated)
            up_x5 = self.upsample1(x5)
            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            concated = self.flows[1](x6)
            disp.append(concated)
            up_x6 = self.upsample2(x6)
            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            concated = self.flows[2](x7)
            disp.append(concated)
            up_x7 = self.upsample3(x7)
            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            concated = self.flows[3](x8)
            disp.append(concated)

            disp = disp[-self.out_layers:]

            tot_disp = self.combine_residuals(disp)
            
            return tot_disp, disp
    
    def combine_residuals(self, flows):
        tot_flows = [flows[0]]
        for f in flows[1:]:
            prev = F.interpolate(tot_flows[-1], size=f.shape[2:], mode='trilinear')
            tot_flows.append(prev + f)
        return tot_flows
    
class Correction(nn.Module):
    def __init__(self, out_layers):
        super(Correction, self).__init__()
        self.out_layers = out_layers

    def compute_p(self, std, gamma=3.0):
        p = torch.exp(-gamma * std)
        return p
    
    def gaussian_kernel3d(self, kernel_size=3, sigma=1.0, device='cuda'):
        """Create 3D Gaussian kernel."""
        coords = torch.arange(kernel_size, device=device) - kernel_size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), -1)
        kernel = torch.exp(-((grid**2).sum(-1)) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    
    def smooth_phi(self, phi, sigma=1.0):
        """Apply Gaussian smoothing to deformation field phi."""
        B, C, H, W, D = phi.shape
        kernel = self.gaussian_kernel3d(kernel_size=3, sigma=sigma, device=phi.device)
        smoothed = F.conv3d(phi, kernel.expand(C, 1, -1, -1, -1), padding=1, groups=C)
        return smoothed
    
    def mean_smooth(self, phi):
        """
        Apply simple mean smoothing over 3x3x3 neighborhood.

        Args:
            phi: (B, 3, H, W, D) deformation field

        Returns:
            smoothed_phi: (B, 3, H, W, D)
        """
        kernel = torch.ones((3, 1, 3, 3, 3), device=phi.device) / 26.0
        kernel[:, :, 1, 1, 1] = 0.0 # center zero 
        smoothed_phi = F.conv3d(phi, kernel, padding=1, groups=phi.shape[1])
        return smoothed_phi

    def forward(self, deform, std, save_path=None):
        '''
        deform_list : accumulated deform list
        std_list : accumulated std list
        '''
        p = self.compute_p(std, gamma=3.)
        p = p.clamp(0.0, 1.0)

        smoothed_phi = self.mean_smooth(deform) # smoothing function 선택
        if save_path is not None:
            np.save(save_path, smoothed_phi.cpu().numpy())

        p_broadcast = p.expand_as(deform)
        
        corrected = p_broadcast * deform + (1-p_broadcast) * smoothed_phi

        return corrected

if __name__ == "__main__":
    model = U_Net_Hyprid(out_layers=1).cuda()

    # torchsummary.summary(model, (2, int(192), int(224), int(192)))
    summary(model, input_size=(1, 2, 192, 224, 192), depth=4)