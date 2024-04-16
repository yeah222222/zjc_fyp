import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds
import open3d as o3d

def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)

def save_pc_as_ply(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu()).astype(np.float64) 
    pcd = o3d.geometry.PointCloud()
    print(type(points))
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu()).astype(np.float64) 
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file
    
def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    # print('distance_transform',distance_transform.shape)
    return distance_transform


def default(x, d):
    return d if x is None else x


def get_num_points(x: Pointclouds, /):
    return x.points_padded().shape[1]


def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas
