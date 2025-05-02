import torch
import torch.nn.functional as F
import numpy as np
import random
import numpy as np
import nibabel as nib
import torch.random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_deformation(displace_field):
    D, H, W = displace_field.shape[2:]
    norm_disp = torch.zeros_like(displace_field)
    norm_disp[:, 0] = displace_field[:, 0] / (W / 2)
    norm_disp[:, 1] = displace_field[:, 1] / (H / 2)
    norm_disp[:, 2] = displace_field[:, 2] / (D / 2)
    return norm_disp

def apply_deformation_using_disp(img, displace_field):
    """
    img: torch.Tensor of shape (1, 1, D, H, W) - single channel 3D image
    displace_field: torch.Tensor of shape (1, 3, D, H, W) - displace field (dx, dy, dz)
    x : W, y : H, z : D
    """
    B, _, D, H, W = img.shape
    
    # Generate normalized grid
    d = torch.linspace(-1, 1, D)
    h = torch.linspace(-1, 1, H)
    w = torch.linspace(-1, 1, W)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=0)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1).to(img.device)

    # Apply deformation
    normalized_disp = normalize_deformation(displace_field)
    deformed_grid = grid + normalized_disp

    # Reshape grid to match F.grid_sample requirements
    deformed_grid = deformed_grid.permute(0, 2, 3, 4, 1)

    # Perform deformation
    deformed_img = F.grid_sample(img, deformed_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return deformed_img

def apply_displace_segment(seg, displace_field):
    """
    seg: torch.Tensor of shape (1, 1, D, H, W) - single channel 3D image
    displace_field: torch.Tensor of shape (1, D, H, W, 3) - displace field (dx, dy, dz)
    """
    D, H, W = seg.shape[2:]
    
    # Generate normalized grid
    d = torch.linspace(-1, 1, D)
    h = torch.linspace(-1, 1, H)
    w = torch.linspace(-1, 1, W)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), axis=-1).unsqueeze(0).to(seg.device)

    # Apply deformation
    displace_field = displace_field.permute(0, 2, 3, 4, 1)  # Rearrange to (N, D, H, W, 3)
    deformed_grid = grid + displace_field

    # Reshape grid to match F.grid_sample requirements
    deformed_grid = deformed_grid.view(seg.shape[0], D, H, W, 3)

    # Perform deformation
    deformed_seg = F.grid_sample(seg, deformed_grid, mode='nearest', padding_mode='border', align_corners=True)

    return deformed_seg

def denormalize_image(normalized_img, img_min, img_max):
    # print(img_min, img_max)
    return normalized_img * (img_max.item() - img_min.item()) + img_min.item()

def save_deformed_image(deformed_tensor, output_path, img_min, img_max, affine):
    """
    변형된 PyTorch Tensor 이미지를 .nii.gz 형식으로 저장하는 함수.

    Parameters:
    - deformed_tensor: 변형된 3D 텐서 (torch.Tensor) (shape: 1, 1, D, H, W)
    - original_nifti_path: 원본 NIfTI 파일 경로 (메타데이터 유지 목적)
    - output_path: 저장할 NIfTI 파일 경로
    """
    # 1. 텐서를 NumPy 배열로 변환 (CPU로 이동 및 차원 축소)
    deformed_img = deformed_tensor.squeeze().cpu().detach().numpy()
    deformed_img = denormalize_image(deformed_img, img_min, img_max)

    # 4. 새로운 NIfTI 객체 생성 및 저장
    deformed_nifti = nib.Nifti1Image(deformed_img, affine=affine)
    nib.save(deformed_nifti, output_path)
    
def mm_norm(x):
    B = x.shape[0]
    out = []
    for i in range(B):
        xi = x[i]
        out.append((xi - xi.min()) / (xi.max() - xi.min() + 1e-8))
    return torch.stack(out)

def resample_pytorch_5d(batch_tensor, current_spacing, target_spacing):
    """
    주어진 target spacing에 맞게 5D (B, C, D, H, W) 텐서를 resampling하는 함수.

    :param batch_tensor: (B, C, D, H, W) 형태의 PyTorch Tensor
    :param current_spacing: 현재 voxel 크기 (depth_spacing, height_spacing, width_spacing)
    :param target_spacing: 목표 voxel 크기 (depth_spacing, height_spacing, width_spacing)
    :return: Resampled PyTorch Tensor
    """
    # 기존 voxel 크기와 목표 voxel 크기에 따른 scaling factor 계산
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    
    # 현재 텐서의 크기
    B, C, D, H, W = batch_tensor.shape  # B: 배치 크기, C: 채널

    # 목표 크기 계산
    new_D, new_H, new_W = int(D * scale_factors[0]), int(H * scale_factors[1]), int(W * scale_factors[2])

    # Interpolation 적용
    resampled_tensor = F.interpolate(
        batch_tensor,  # (B, C, D, H, W) 형태 유지
        size=(new_D, new_H, new_W),
        mode='trilinear',
        align_corners=True
    )

    return resampled_tensor


def upsample_deformation(deformation_field, target_shape, current_shape):
    """
    Deformation Field를 현재 resolution으로 upsampling.
    :param deformation_field: (B, 3, D, H, W) 형태의 deformation field
    :param target_shape: (new_D, new_H, new_W) 형태의 목표 해상도
    :param current_shape: (D, H, W) 현재 해상도
    :return: Upsampled deformation field
    """
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]
    
    # Trilinear interpolation 적용하여 deformation field 크기 변경
    upsampled_deformation = F.interpolate(
        deformation_field, size=target_shape, mode='trilinear', align_corners=True
    )

    # 변위 벡터 크기 조정 (보간 후 변위 크기도 scaling factor 반영 필요)
    # for i in range(3):  # x, y, z 각각 적용
    #     upsampled_deformation[:, i, :, :, :] *= scale_factors[i]

    return upsampled_deformation

def compose_displace(prev_displace, new_displace):
    """
    Additive Method
    이전 resolution의 displace을 새로운 resolution에서 적용 (warping).
    :param prev_displace: 이전 resolution의 displace field (B, 3, D, H, W)
    :param new_displace: 현재 resolution에서 학습한 displace field (B, 3, D, H, W)
    :return: Composed displace field (B, 3, D, H, W)
    """
    if prev_displace is None:
        return new_displace

    D, H, W = new_displace.shape[2:]

    # 이전 displace field를 현재 resolution으로 upsample
    prev_displace_upsampled = upsample_deformation(prev_displace, (D, H, W), prev_displace.shape[2:])

    # Composition 수행 (Additive)
    composed_displace = prev_displace_upsampled + new_displace
    return composed_displace

def add_identity_to_deformation(deformation_field):
    D, H, W, _ = deformation_field.shape
    identity = np.stack(np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'), axis=-1)
    return deformation_field + identity

# --- 2. Jacobian Determinant 계산 ---
def compute_jacobian_determinant(displacement_field):
    """
    변형장의 Jacobian Determinant를 계산하여 반환
    displacement_field: (X, Y, Z, 3) 형태의 변형장 (displacement field를 의미)
    """
    if displacement_field.shape[-1] !=3:
        displacement_field = np.transpose(displacement_field, (1, 2, 3, 0)) # (X, Y, Z, 3)

    H, W, D, _ = displacement_field.shape
    
    ## denormalize
    def_voxel = displacement_field.copy()
    def_voxel[..., 0] *= (W - 1) / 2
    def_voxel[..., 1] *= (H - 1) / 2
    def_voxel[..., 2] *= (D - 1) / 2

    deformation_field = add_identity_to_deformation(def_voxel)

    dx = np.gradient(deformation_field[..., 0], axis=0)  # dφ_x/dx
    dy = np.gradient(deformation_field[..., 0], axis=1)  # dφ_x/dy
    dz = np.gradient(deformation_field[..., 0], axis=2)  # dφ_x/dz

    ex = np.gradient(deformation_field[..., 1], axis=0)  # dφ_y/dx
    ey = np.gradient(deformation_field[..., 1], axis=1)  # dφ_y/dy
    ez = np.gradient(deformation_field[..., 1], axis=2)  # dφ_y/dz

    fx = np.gradient(deformation_field[..., 2], axis=0)  # dφ_z/dx
    fy = np.gradient(deformation_field[..., 2], axis=1)  # dφ_z/dy
    fz = np.gradient(deformation_field[..., 2], axis=2)  # dφ_z/dz

    # Jacobian 행렬 구성
    jacobian = np.zeros(deformation_field.shape[:-1] + (3, 3))
    jacobian[..., 0, 0] = dx
    jacobian[..., 0, 1] = dy
    jacobian[..., 0, 2] = dz

    jacobian[..., 1, 0] = ex
    jacobian[..., 1, 1] = ey
    jacobian[..., 1, 2] = ez

    jacobian[..., 2, 0] = fx
    jacobian[..., 2, 1] = fy
    jacobian[..., 2, 2] = fz

    # # Determinant 계산
    jacobian_det = np.linalg.det(jacobian)
    return jacobian_det

def calculate_negative_rate(displacement):
    jacobian_det = compute_jacobian_determinant(displacement.detach().cpu().numpy().squeeze(0))
    negative_mask = jacobian_det <= 0.
    neg_num = negative_mask.sum().item()
    tot_num = np.prod(jacobian_det.shape).item()
    neg_rate = neg_num/tot_num

    return neg_rate



if __name__ in '__main__':
    m = 0
    # for _ in range(10000):
    #     t = torch.randn((4, 3, 8, 8, 8))
    #     p = torch.randn((4, 3, 8, 8, 8))
    #     m += normalized_cross_correlation(t, p)
    # print(m/10000)
    imgsz = 128
    t = torch.randn((1, 3, 192, 224, 192))
    p = torch.randn((1, 3, 192, 224, 192))
    #  normalized_cross_correlation(t, p)
    # print(normalized_cross_correlation(t, p))