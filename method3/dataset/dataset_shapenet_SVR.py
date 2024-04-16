import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os,sys
from PIL import Image
sys.path.append ('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/dataset/')
import dataset.my_utils as my_utils
import pickle
from os.path import join, dirname, exists
from easydict import EasyDict
import json
from termcolor import colored
import dataset.pointcloud_processor as pointcloud_processor
from copy import deepcopy
import open3d as o3d

import torch
from pytorch3d.renderer import PerspectiveCameras
import numpy as np

def read_point_cloud(file_path):
    with open(file_path, 'r') as file:
        ply_data = file.readlines()
    
    # Find where the header ends (indicated by "end_header")
    header_end_index = 0
    for i, line in enumerate(ply_data):
        if "end_header" in line:
            header_end_index = i + 1
            break
    
    # Process the vertex data after the header
    vertex_data = []
    for line in ply_data[header_end_index:]:
        if line.strip() == "":  # Skip empty lines if any
            continue
        parts = line.split()
        if len(parts) == 6:  # x, y, z, red, green, blue
            # Convert the vertex data to float for coordinates and int for colors
            vertex = [float(parts[0]), float(parts[1]), float(parts[2]),
                      int(parts[3]), int(parts[4]), int(parts[5])]
            vertex_data.append(vertex)
    
    # Convert the list to a NumPy array
    vertex_array = np.array(vertex_data)
    
    return vertex_array

class ShapeNet(data.Dataset):
    """
    Shapenet Dataloader
    Uses Shapenet V1
    Make sure to respect shapenet Licence.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt, train=True):
        self.opt = opt
        self.num_sample = opt.number_points if train else 2500
        self.train = train
        self.num_image_per_object = 12
        self.init_normalization()
        self.init_singleview()
        self.mask_data_dir = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/data_mask'
        # if not opt.demo:
        my_utils.red_print('Create Shapenet Dataset...')
        # Select classes
        if opt.shapenet13:
            opt.class_choice = ["car"] 
        # Create Cache path
        self.path_dataset = join(dirname(__file__), 'data_car_SVR_VISIBLE_pymesh_3dcamera', 'cache')#data_car_SVR_VISIBLE_pymesh_3dcamera
        self.RT_dataset = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/RT_matrix'
        self.K_dataset = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/K_matrix'
        if not exists(self.path_dataset):
            os.mkdir(self.path_dataset)
        self.path_dataset = join(self.path_dataset,
                                    self.opt.normalization + str(train) + "_".join(self.opt.class_choice))
        with open('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/angle_dict.pickle', 'rb') as handle:
            self.angle_dict = pickle.load(handle)
        if exists(self.path_dataset + "info.pkl"):
            # Reload dataset
            my_utils.red_print(f"Reload dataset : {self.path_dataset}")
            with open(self.path_dataset + "info.pkl", "rb") as fp:
                self.data_metadata = pickle.load(fp)

            self.data_points = torch.load(self.path_dataset + "_points.pth")
            self.gen_points = torch.load(self.path_dataset + "_genpoints.pth")
            self.RT = torch.load(self.path_dataset + "_RT.pth")
            self.K = torch.load(self.path_dataset + "_K.pth")
        else:
            self.data_metadata = []
            dataset_path = '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc'
            all_image_path = os.listdir(join(dataset_path, 'data'))
            all_ply_path = os.listdir(join(dataset_path, 'data_plyfile_junyun'))    #data_sameas_diffusion_ply
            list_pointcloud = sorted(all_ply_path)
            if self.train:
                list_pointcloud = list_pointcloud[:int(len(list_pointcloud) * 0.8)]
                
            else:
                list_pointcloud = list_pointcloud[int(len(list_pointcloud) * 0.8):]

            self.data_points = []
            self.gen_points = []
            self.RT = []
            self.K = []
            for ply_path in list_pointcloud:
                ply_points = self._getitem(pointcloud_path = join(dataset_path, 'data_plyfile_junyun', ply_path))
                img_dir = join(dataset_path, 'data', ply_path.split('.ply')[0].split('_withcolor')[0]+'_')#data
                # print('img_dir',img_dir)
                images_ply = os.listdir(img_dir)
                for im in images_ply:
                    if im.endswith('.png'):#data_ply_color data data_atlasnet_pymesh_poissondisk
                        self.data_metadata.append({'pointcloud_path': join(dataset_path, 'data_plyfile_junyun', ply_path), 'img_path': join(img_dir, im)})
                        #self.data_metadata.append({'img_path': join(img_dir, im)})
                        self.data_points.append(ply_points)
                        # gen_points = np.loadtxt(join(dataset_path,'data_xyz_16x_fps_visible',ply_path.split('.ply')[0].split('_withcolor')[0]+'_',im.split('.png')[0]+'.xyz'))
                        # self.gen_points.append(torch.from_numpy(gen_points).float())
                        gen_p = self._getitem(pointcloud_path=join(dataset_path, 'data_atlasnet_pymesh_poissondisk',ply_path.split('.ply')[0].split('_withcolor')[0]+'_' ,im.split('.png')[0].split('_')[0]+'.ply'))
                        self.gen_points.append(gen_p[:,:3])
                        self.RT.append(torch.tensor(np.loadtxt(os.path.join(self.RT_dataset,str(self.angle_dict[im.split('.png')[0].split('_')[0]])+'.txt')).astype(np.float32)))
                        self.K.append(torch.tensor(np.loadtxt(os.path.join(self.K_dataset,str(self.angle_dict[im.split('.png')[0].split('_')[0]])+'.txt')).astype(np.float32)))
            with open(self.path_dataset + "info.pkl", "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)

            torch.save(self.data_points, self.path_dataset + "_points.pth")
            torch.save(self.gen_points, self.path_dataset + "_genpoints.pth")
            torch.save(self.RT, self.path_dataset + "_RT.pth")
            torch.save(self.K, self.path_dataset + "_K.pth")
            my_utils.yellow_print('save points and data success')

    def init_normalization(self):
        # if not self.opt.demo:
        my_utils.red_print("Dataset normalization : " + self.opt.normalization)

        if self.opt.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def init_singleview(self):
        ## Define Image Transforms
        self.transforms = transforms.Compose([

            transforms.ToTensor(),
        ])

    #     # RandomResizedCrop or RandomCrop
    #     self.dataAugmentation = transforms.Compose([
    #         transforms.RandomCrop(127),
    #         transforms.RandomHorizontalFlip(),
    #     ])

    #     self.validating = transforms.Compose([
    #         transforms.CenterCrop(127),
    #     ])

    def _getitem(self, pointcloud_path):
        
        # pointcloud_path, image_path, pointcloud, category = self.datapath[index]
        # my_utils.yellow_print(pointcloud_path)
        # TODO: change the npy file into ply file to handle (has solved)
        # points = np.load(pointcloud_path)
        # points = torch.from_numpy(points).float()

        # pcd = o3d.io.read_point_cloud(pointcloud_path)
        # points = np.asarray(pcd.points)
        # points = torch.from_numpy(points).float()
        # points[:, :3] = self.normalization_function(points[:, :3])
        points = read_point_cloud(pointcloud_path)
        points = torch.from_numpy(points).float()
        points[:, :3] = self.normalization_function(points[:, :3])
        points[:, 3:] = points[:, 3:]/255.
       
        return points

    def __getitem__(self, index):
        
       
        return_dict = deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        genpoints = self.gen_points[index]
        genpoints = genpoints.clone()
        RT_matrix = self.RT[index]
        RT_matrix = RT_matrix.clone()
        K_matrix = self.K[index]
        K_matrix = K_matrix.clone()
        # print('pointssssssssss',points.shape)
        # return
        if self.opt.sample:
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]
        return_dict['sequence_point_cloud'] = points[:, :].contiguous()
        return_dict['gen_points'] = genpoints[:, :].contiguous()
        # return_dict['camera'] = torch.tensor(np.load(self.data_metadata[index]['img_path'].split('.png')[0]+'.npy'))
        return_dict['RT_matrix'] = RT_matrix
        return_dict['K_matrix'] = K_matrix
        # print(return_dict['camera'].shape)
        # return 
        # Image processing
        if self.opt.SVR:
            if self.train:
                N = np.random.randint(1, self.num_image_per_object)
                im = Image.open(return_dict['img_path'])
                #im = self.dataAugmentation(im)  # random crop
                # 应用数据增强
          

                # # 确保转换后的图像是PIL图像
                # if isinstance(im, torch.Tensor):
                #     im = transforms.ToPILImage()(im)

                # # 保存图像为JPG格式
                # im.save(join('/mnt/d/junch_data/AtlasNet/AtlasNet_MY/augumentation/cache2',return_dict['image_path'].split('/')[-2]+"_transform"+return_dict['image_path'].split('/')[-1]))
            else:
                im = Image.open(return_dict['img_path'])
               # im = self.validating(im)  # center crop
            im = self.transforms(im)  # scale
            
            return_dict['image_rgb'] = im
            mask_image = Image.open(join(self.mask_data_dir,return_dict['img_path'].split('/')[-2],return_dict['img_path'].split('/')[-1].split('.png')[0].split('_')[0]+'_mask.png')).convert('L') 

            # 将灰度图像二值化，生成二进制mask，阈值设定为127
            mask_np = np.array(mask_image) > 127

            # 将二进制mask转换为PyTorch张量
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32))
            # print('mask tensor ++++++++++++++++++', mask_tensor.shape)
            # return
            
            return_dict['fg_probability'] = mask_tensor
            final_dict = deepcopy(return_dict)
            del final_dict['img_path']
            del final_dict['pointcloud_path']
            # print(final_dict.keys())
        return final_dict

    def __len__(self):
        return len(self.data_metadata)

    @staticmethod
    def int2str(N):

        return "0" + str(N)
   
    def load(self, path):
        ext = path.split('.')[-1]
        if ext == "npy" or ext == "ply" or ext == "obj":
            return self.load_point_input(path)
        else:
            return self.load_image(path)
        
    def load_image(self, path):
        im = Image.open(path)
        # im = self.transforms(im)
        # if isinstance(im, torch.Tensor):
        #     im = transforms.ToPILImage()(im)

        #         # 保存图像为JPG格式
        # #im.save(join('/mnt/d/junch_data/AtlasNet/AtlasNet_MY/augumentation/cache2',return_dict['image_path'].split('/')[-2]+"_transform"+return_dict['image_path'].split('/')[-1]))
        # im.save("/mnt/d/junch_data/AtlasNet/AtlasNet_MY/dataset/2_notanytransform.jpg")
        # return
        im = self.transforms(im)
        im = im[:3, :, :]

        return_dict = {
            'image': im.unsqueeze_(0),
            'operation': None,
            'path': path,
        }
        my_utils.red_print('I am the load_image function')
        return return_dict
    
    def load_point_input(self, path):
        ext = path.split('.')[-1]
        if ext == "npy":
            points = np.load(path)
        elif ext == "ply" or ext == "obj":
            import pymesh
            mesh_load = pymesh.load_mesh(path)
            points = mesh_load.vertices


        else:
            print("invalid file extension")

        points = torch.from_numpy(points).float()
        operation = pointcloud_processor.Normalization(points, keep_track=True)
        if self.opt.normalization == "UnitBall":
            operation.normalize_unitL2ball()
        elif self.opt.normalization == "BoundingBox":
            operation.normalize_bounding_box()
        else:
            pass
        return_dict = {
            'points': points,
            'operation': operation,
            'path': path,
        }
        return return_dict
    # def load(self, path):
    #     ext = path.split('.')[-1]
    #     if ext == "npy" or ext == "ply" or ext == "obj":
    #         return self.load_point_input(path)
    #     else:
    #         return self.load_image(path)

    # def load_point_input(self, path):
    #     ext = path.split('.')[-1]
    #     if ext == "npy":
    #         points = np.load(path)
    #     elif ext == "ply" or ext == "obj":
    #         import pymesh
    #         points = pymesh.load_mesh(path).vertices
    #     else:
    #         print("invalid file extension")

    #     points = torch.from_numpy(points).float()
    #     operation = pointcloud_processor.Normalization(points, keep_track=True)
    #     if self.opt.normalization == "UnitBall":
    #         operation.normalize_unitL2ball()
    #     elif self.opt.normalization == "BoundingBox":
    #         operation.normalize_bounding_box()
    #     else:
    #         pass
    #     return_dict = {
    #         'points': points,
    #         'operation': operation,
    #         'path': path,
    #     }
    #     return return_dict


    # def load_image(self, path):
    #     im = Image.open(path)
    #     im = self.validating(im)
    #     im = self.transforms(im)
    #     im = im[:3, :, :]
    #     return_dict = {
    #         'image': im.unsqueeze_(0),
    #         'operation': None,
    #         'path': path,
    #     }
    #     return return_dict


if __name__ == '__main__':
    print('Testing Shapenet dataset')
    opt = {"normalization": "UnitBall", "class_choice": ["plane"], "SVR": False, "sample": True, "number_points": 2500,
           "shapenet13": True, "device": 'cuda:0'}
    d = ShapeNet(EasyDict(opt), train=True)
    print(d[1])
    a = len(d)
