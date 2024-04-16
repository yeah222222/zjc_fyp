import sys,os
sys.path.append('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc')
from dataset.trainer_dataset import TrainerDataset
from pytorch3d.loss import chamfer_distance
from easydict import EasyDict
from model.model_coloring import PointCloudColoringModel
import torch.optim as optim
import logging
import dataset.my_utils as my_utils
import torch
import math
from copy import deepcopy
import open3d as o3d
import numpy as np

def save_pc_as_ply(save_path, points, colors=None, normals=None):
    assert save_path[-3:] == 'ply', 'not .ply file'
    if type(points) == torch.Tensor:
        points = np.asarray(points.detach().cpu())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if type(colors) == torch.Tensor:
            colors = np.asarray(colors.detach().cpu())
        assert colors.min()>=0 and colors.max()<=1
        pcd.colors = o3d.utility.Vector3dVector(colors) # should be within the range of [0, 1]
    if normals is not None:
        if type(normals) == torch.Tensor:
            normals = np.asarray(normals.detach().cpu())
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True) # should be saved as .ply file

def main():
#opts
    opt = EasyDict()
    opt.demo = False
    opt.train = False
    opt.device = 'cuda:0'
    opt.shapenet13 = True
    #batchsize
    opt.batch_size = 24
    opt.batch_size_test = 24
    #epoch
    opt.nepoch = 2001
    opt.start_epoch = 0
    #learning rate
    opt.lrate = 0.001
    opt.lr_decay_1 = 420
    opt.lr_decay_2 = 440
    opt.lr_decay_3 = 445
    #data handling
    opt.SVR = True
    opt.train_only_encoder = True
    opt.sample = True
    opt.class_choice = []
    opt.normalization = 'UnitBall'
    opt.number_points = 2500
    opt.number_points_eval = 2500
    opt.random_rotation =  True
    opt.data_augmentation_axis_rotation = True
    opt.data_augmentation_random_flips = True
    opt.random_translation = True
    opt.anisotropic_scaling = True
    #Network
    opt.num_layers = 2
    opt.hidden_neurons = 512
    opt.loop_per_epoch = 1
    opt.nb_primitives = 25
    opt.template_type = "SQUARE" #choices = ["SQUARE", "SPHERE"]
    opt.bottleneck_size = 1024
    opt.activation = 'relu' # choices = ['relu', 'sigmoid', 'softplus', 'logsigmoid', 'softsign', 'tanh']
    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    opt.dim_template = dim_template_dict[opt.template_type]
    opt.reload_continue_training_model = None
    opt.reload_decoder_path = None
    opt.training_media_path = '/mnt/d/junch_data/AtlasNet/AtlasNet_MY/demo'
#load network model coloring

    RELOAD_TRAINING_MODEL = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/ckpt/modelcoloring_pymesh_RTK/400-model.pth'

    my_utils.green_print('Reload continue training model')
    network = PointCloudColoringModel(opt=opt, point_cloud_model='pvcnn', point_cloud_model_layers=1, point_cloud_model_embed_dim=64).to(opt.device)
    network.load_state_dict(torch.load(RELOAD_TRAINING_MODEL, map_location='cuda:0'))

      
#load dataset
    trainer_dataset =  TrainerDataset(opt=opt)    
    datasets_dict = trainer_dataset.build_dataset()
    
#Accelerator config
    dataloader_train = datasets_dict.dataloader_train
    dataloader_val = datasets_dict.dataloader_test

#LOG

                

    # test epoch
 
    iterator_train = dataloader_val.__iter__()
    save_dir = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/visaul400cityu'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        network.eval()
        for dic_test in iterator_train:
            dic_test = {k: v.to(opt.device) for k, v in dic_test.items() if isinstance(v, torch.Tensor)}
            gen_points, predict_colors, gt = network(dic_test)
            for i in range(gen_points.shape[0]):
                sp_pred = os.path.join(save_dir,str(i)+'_pred.ply')
                save_pc_as_ply(save_path=sp_pred, points=gen_points[i], colors=predict_colors[i])
                sp_gt = os.path.join(save_dir,str(i)+'_gt.ply')
                save_pc_as_ply(save_path=sp_gt, points=gt[i][:,:3], colors=gt[i][:,3:])

    
        
if __name__ == '__main__':
    main()
    