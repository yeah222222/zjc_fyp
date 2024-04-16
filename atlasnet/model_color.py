import torch.nn.functional as F
import torch.nn as nn
import torch
from .trainer_model import EncoderDecoder
import dataset.my_utils as my_utils
import numpy as np
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from . import resnet

def write_ply_with_colors(vertices, colors,save_path,  text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    vertices: array of point cloud vertices of size (N, 3)
    colors: array of colors corresponding to vertices of size (N, 3), in normalized [0, 1] range
    """
    # Ensure that vertices and colors have the same number of points
    print(type(colors))
    print(type(colors[0]))

    assert vertices.shape[0] == colors.shape[0], "Vertices and colors must have the same number of points"

    # Normalize color values to [0, 255] range
    colors = (colors *255).astype(np.uint8)

    # Combine vertices and colors into one array
    vertex_colored = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    for i in range(vertices.shape[0]):
        vertex_colored[i] = tuple(vertices[i]) + tuple(colors[i])

    # Define the PlyElement for vertices with colors and create the PlyData instance
    ply_element = PlyElement.describe(vertex_colored, 'vertex', comments=['vertices with color'])
    ply_data = PlyData([ply_element], text=text)

    # Write the data to a .ply file
    ply_data.write(save_path)
    
def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")

def find_nearest_point(arr1,arr2): #arr1: gt arr2: output

    b,n,_ = arr1.shape
    distances = torch.cdist(arr1, arr2)  # [b, n, n]

    min_indices = torch.argmin(distances, dim=2)  # [b, n]

    batch_indices = torch.arange(b).view(-1, 1).expand(-1, n)
        
    return batch_indices, min_indices

class OutputColorModel(nn.Module):
    def __init__(self, opt) :
        super(OutputColorModel, self).__init__()
        self.opt = opt 
        self.bottleneck_size = opt.bottleneck_size
        self.reload_SVR_for_trainingcolor = '/mnt/d/junch_data/AtlasNet/AtlasNet_MY/save_ckpt_3dimension_25square_inputimage_outputpc_onlyindex/1830-model.pth'#/mnt/d/junch_data/AtlasNet/AtlasNet_MY/save_ckpt_3dimension_25square_inputimage/1455-model.pth
        self.color = True
        
        self.encoder_image = resnet.resnet18(pretrained=False, num_classes=self.opt.bottleneck_size)
        
        self.network_SVR = EncoderDecoder(self.opt)
        if self.opt.reload_continue_training_model is None:    
            if self.color and self.reload_SVR_for_trainingcolor is not None:
                self.network_SVR.load_state_dict(torch.load(self.reload_SVR_for_trainingcolor, map_location='cuda:0'))
        my_utils.yellow_print("Preparing for color training")
        
                
        for param in self.network_SVR.parameters():
            param.requires_grad = False

        self.conv_list = nn.ModuleList([torch.nn.Conv1d(2051, 2096 ,1),
                                        torch.nn.Conv1d(2096, 1024, 1),
                                        torch.nn.Conv1d(1024, 512, 1),
                                        torch.nn.Conv1d(512, 256, 1),
                                        torch.nn.Conv1d(256, 128, 1),
                                        torch.nn.Conv1d(128, 64, 1)])
        
        self.last_conv = torch.nn.Conv1d(64, 3, 1)
        
        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(i) for i in [2096,1024,512,256,128,64]])
        self.activation = get_activation(self.opt.activation)
        
    def forward(self,image, points): #image and gt points with xyzrgb
        output_points = self.network_SVR(image)
        output_points = output_points.transpose(2,3).contiguous()
        output_points = output_points.view(output_points.shape[0], -1, 3).contiguous()#b,n,3
        output_points = output_points.transpose(1,2).contiguous()#b,3,n
        my_utils.magenta_print(output_points.shape)
        
        self.latent_vec = self.network_SVR.encoder(image).unsqueeze(2) #b,1024,1
        self.latent_vec = self.latent_vec.repeat(1,1,output_points.shape[2]) #b,1024,n
       # my_utils.magenta_print(self.latent_vec.shape)
        self.image_encode = self.encoder_image(image).unsqueeze(2) #b,1024,1
        self.image_encode = self.image_encode.repeat(1,1,output_points.shape[2])#b,1024,n
        
        input_points = torch.cat((output_points, self.latent_vec, self.image_encode), dim=1) #b,2051, n 
        my_utils.magenta_print(input_points.shape)
        
        for i in range(6):
            input_points = self.activation(self.bn_list[i](self.conv_list[i](input_points)))
        input_points = self.last_conv(input_points) #b,3,n
        
        input_points = input_points.transpose(1,2).contiguous()#b,n,3
        batch_indices, min_indices = find_nearest_point(points[:,:,:3],output_points.transpose(1,2).contiguous()) #b,n,3 

        gt_colors = points[batch_indices,min_indices][:,:,3:]
        # mse_loss = F.mse_loss(input=input_points, target=gt_colors, reduction='mean')
        return  input_points, gt_colors
    
    
    def generate_ply(self, image): #image and gt points with xyzrgb
        
        output_points = self.network_SVR(image, train=False)
        output_points = output_points.transpose(2,3).contiguous()
        output_points = output_points.view(output_points.shape[0], -1, 3).contiguous()#b,n,3
        output_points = output_points.transpose(1,2).contiguous()#b,3,n
        
        
       
        #my_utils.magenta_print(output_points.shape)
        
        self.latent_vec = self.network_SVR.encoder(image).unsqueeze(2) #b,1024,1
        self.latent_vec = self.latent_vec.repeat(1,1,output_points.shape[2]) #b,1024,n
       # my_utils.magenta_print(self.latent_vec.shape)
        self.image_encode = self.encoder_image(image).unsqueeze(2) #b,1024,1
        self.image_encode = self.image_encode.repeat(1,1,output_points.shape[2])#b,1024,n
        
        colors = torch.cat((output_points, self.latent_vec, self.image_encode), dim=1) #b,2051, n         
      
       # my_utils.magenta_print(input_points.shape)
        print('colors',colors.device)
        for i in range(6):
            colors = self.activation(self.bn_list[i](self.conv_list[i](colors)))
        colors = self.last_conv(colors) #b,3,n
        colors = colors.transpose(1,2).contiguous()#b,n,3
        
        output_points = output_points.transpose(1,2)
        my_utils.magenta_print(output_points[0].shape)
        my_utils.magenta_print(colors[0].shape)
        return  output_points, colors
        