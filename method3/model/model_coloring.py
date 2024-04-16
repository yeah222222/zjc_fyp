import sys

# 指定正确的 `model` 包路径

from atlasnet.trainer_model import EncoderDecoder
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from pytorch3d.renderer import PerspectiveCameras
from .point_cloud_transformer_model import PointCloudTransformerModel
from .projection_model import PointCloudProjectionModel
import sys,os
import numpy as np
import open3d as o3d
from .model_utils import save_pc_as_ply
import pickle
# sys.path.append('/mnt/d/junch_data/AtlasNet/AtlasNet_MY/')
# print(sys.path)
# print("Current working directory:", os.getcwd())
# from model.trainer_model import EncoderDecoder
# def get_pytorch3d_camera(img_RT):


# # 假设你已经加载了你的旋转矩阵R和平移向量T
# # 例如：R_blender, T_blender = ...
#     # img_RT = np.load(rt_path)
#     # img_RT = torch.tensor(img_RT)
#     R_blender, T_blender = img_RT[:,:3, :3], img_RT[:,:, -1]
#     print('R_blender',R_blender.shape)
#     print('T_blender',T_blender.shape)
#     # 确保R_blender和T_blender是浮点数类型的张量
#     R_blender = R_blender.to(dtype=torch.float32)
#     T_blender = T_blender.to(dtype=torch.float32)

#     # 创建一个翻转Z轴的张量
#     flip_z = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)

#     # R_blender.to('cuda:0')
#     # flip_z.to('cuda:0')
#     # 旋转矩阵和平移向量的转换
#     R_pytorch3d = torch.mm(R_blender.to('cuda:0'), flip_z.to('cuda:0'))
#     T_pytorch3d = torch.mm(T_blender.unsqueeze(0), flip_z).squeeze(0)

#     # 为了创建一个PerspectiveCameras实例，我们需要确保R和T有一个批次维度
#     # 如果它们不是批次维度，你可以使用unsqueeze来添加一个维度
#     R_pytorch3d = R_pytorch3d.unsqueeze(0)  # 形状应该是(1, 3, 3)
#     T_pytorch3d = T_pytorch3d.unsqueeze(0)  # 形状应该是(1, 3)

#     # 创建PyTorch3D的相机
#     cameras = PerspectiveCameras(device="cpu", R=R_pytorch3d, T=T_pytorch3d)

#     # 现在cameras应该是正确配置的
#     return cameras
import torch
from pytorch3d.renderer import PerspectiveCameras

# def get_pytorch3d_camera(batch_img_RT):
#     # 确保batch_img_RT是浮点数类型的张量
#     batch_img_RT = torch.tensor(batch_img_RT, dtype=torch.float32)

#     # 提取旋转矩阵R和平移向量T
#     R_blender = batch_img_RT[:, :3, :3]  # 取每个样本的前3x3部分作为旋转矩阵
#     T_blender = batch_img_RT[:, :3, 3]   # 取每个样本的最后一列作为平移向量

#     # 创建一个翻转Z轴的张量
#     flip_z = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)

#     # 如果batch_img_RT在GPU上，确保flip_z也在GPU上
#     if batch_img_RT.is_cuda:
#         flip_z = flip_z.cuda()

#     # 计算PyTorch3D使用的旋转矩阵，对于批量操作，使用torch.matmul
#     R_pytorch3d = torch.matmul(R_blender, flip_z)

#     # 计算PyTorch3D使用的平移向量，对于批量操作，使用torch.einsum来做批量的矩阵乘法和加法
#     T_pytorch3d = -torch.einsum('bij,bj->bi', R_blender.transpose(1, 2), T_blender)

#     # 创建PyTorch3D的相机，注意R和T已经包含了批次维度
#     cameras = PerspectiveCameras(device=batch_img_RT.device, R=R_pytorch3d, T=T_pytorch3d)

#     return cameras
import torch
from pytorch3d.renderer import PerspectiveCameras

# def get_pytorch3d_camera(batch_img_RT,K):
#     # 确保batch_img_RT是浮点数类型的张量
#     batch_img_RT = batch_img_RT.float()

#     # 提取旋转矩阵R和平移向量T
#     R_blender = batch_img_RT[:, :3, :3]  # 取每个样本的前3x3部分作为旋转矩阵
#     T_blender = batch_img_RT[:, :3, 3]   # 取每个样本的最后一列作为平移向量

#     # 创建一个翻转Z轴的张量并放置到正确的设备上
#     flip_z = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=batch_img_RT.device)

#     # 计算PyTorch3D使用的旋转矩阵，对于批量操作，使用torch.matmul
#     R_pytorch3d = torch.matmul(R_blender, flip_z)

#     # 计算PyTorch3D使用的平移向量，对于批量操作，使用torch.einsum来做批量的矩阵乘法和加法
#     T_pytorch3d = -torch.einsum('bij,bj->bi', R_blender.transpose(1, 2), T_blender)

#     # 创建PyTorch3D的相机，注意R和T已经包含了批次维度
#     cameras = PerspectiveCameras(device=batch_img_RT.device, R=R_pytorch3d, T=T_pytorch3d,K=K)

#     return cameras


def find_nearest_point(arr1,arr2): #arr1: gt arr2: output
    # print(arr1.shape)
    b,n,_ = arr1.shape
    distances = torch.cdist(arr1, arr2)  # [b, n, n]

    min_indices = torch.argmin(distances, dim=2)  # [b, n]

    batch_indices = torch.arange(b).view(-1, 1).expand(-1, n)
        
    return batch_indices, min_indices

class PointCloudColoringModel(PointCloudProjectionModel):
    
    def __init__(
        self,
        opt,
        point_cloud_model: str,
        point_cloud_model_layers: int,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)
        self.opt = opt
 
        # Checks
        if self.predict_shape or not self.predict_color:
            raise NotImplementedError('Must predict color, not shape, for coloring')
        

        # Create point cloud model for processing point cloud
        self.point_cloud_model = PointCloudTransformerModel(
            num_layers=point_cloud_model_layers,
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )


    def _forward(
        self, 
        gen_points,
        points,
        # camera: Optional[CamerasBase],
        RT,
        K,
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_point_cloud: bool = False,
        noise_std: float = 0.0,
    ):
        # print('points', points.shape)
        if image_rgb.shape[1] == 4:
            # 如果图像有4个通道，只保留前3个通道 (RGB)
            image_rgb = image_rgb[:, :3, :, :]
        #print('image_rgb',image_rgb.shape)

       # print('output points',output_points.shape)
        pc = Pointclouds(points=gen_points) 
        
        # Normalize colors and convert to tensor
        x = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
       # print('pc', x.shape)
        #x_points, x_colors = x[:, :, :3], x[:, :, 3:]
        x_points = x
        batch_indices, min_indices = find_nearest_point(gen_points,points[:,:,:3]) #b,n,3
        x_colors =  points[batch_indices,min_indices][:,:,3:]

        x_colors = (x_colors - self.colors_mean) / self.colors_std
        # save_pc_as_ply(save_path='ee111.ply', points=gen_points[0], colors=x_colors[0])
        # save_pc_as_ply(save_path='ee222.ply', points=points[0][:,:3], colors=points[0][:,3:])
        # return
        #print('x_cplors---------',x_colors.shape)

        # Add noise to points. TODO: Add to config.
        x_input = x_points + torch.randn_like(x_points) * noise_std

        # Conditioning
        camera = PerspectiveCameras(R=RT[:,:3,:3],T=RT[:,:3,3],K=K,device='cuda:0')
        x_input = self.get_input_with_conditioning(x_input, camera=camera, 
            image_rgb=image_rgb, mask=mask)

        # Forward
        pred_colors = self.point_cloud_model(x_input)
        #print('pred colros', pred_colors.shape)
        # During inference, we return the point cloud with the predicted colors
        if return_point_cloud:
            
            gen_points, predict_colors  = self.tensor_to_point_cloud(
                torch.cat((x_points, pred_colors), dim=2), denormalize=True, unscale=True)
            return gen_points, predict_colors, points

        # During training, we have ground truth colors and return the loss
        loss = F.mse_loss(pred_colors, x_colors)
        return loss

    def forward(self, batch: dict, **kwargs):
        """A wrapper around the forward method"""
        if not isinstance(batch, dict):  # fixes a bug with multiprocessing where batch becomes a dict
            #batch = FrameData(**batch)  # it really makes no sense, I do not understand it
            raise ValueError("The input batch must be a dictionary.")
        return self._forward(
            gen_points = batch.get('gen_points'),
            points = batch.get('sequence_point_cloud'),
            RT = batch.get('RT_matrix'),
            K = batch.get('K_matrix'),
            image_rgb = batch.get('image_rgb'),
            mask = batch.get('fg_probability'),
            **kwargs,
        )