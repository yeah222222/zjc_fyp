import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from .model_blocks import Mapping2Dto3D, Identity
from .template import get_template
import dataset.my_utils as my_utils

class Atlasnet(nn.Module):

    def __init__(self, opt):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(Atlasnet, self).__init__()
        self.opt = opt
        self.device = opt.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = opt.number_points // opt.nb_primitives #2500/25=100
        self.nb_pts_in_primitive_eval = opt.number_points_eval // opt.nb_primitives

        # if opt.remove_all_batchNorms:
        #     torch.nn.BatchNorm1d = Identity
        #     print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]#25个templates

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        # Sample points in the patches
        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)), 
                latent_vector.device) for i in range(self.opt.nb_primitives)]
            #print("input_points" , input_points[0].shape) #input_points torch.Size([25, 1, 2, 100])
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).unsqueeze(1) for i in
                                   range(0, self.opt.nb_primitives)], dim=1) #latent vector torch.Size([32, 1024, 1])
                                                                             #output_points torch.Size([32, 25, 3, 100])
        #print("output_points",output_points.shape)
        # Return the deformed pointcloud
        return output_points.contiguous()  # batch, nb_prim, 3, num_point

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device)
                        for i in range(self.opt.nb_primitives)]
        print("1 input points", input_points[0].shape)
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]
        print("2 input points", input_points[0].shape)
        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.opt.nb_primitives)]

 
        output_meshes = [pymesh.form_mesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().detach().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.opt.nb_primitives)]

        # Deform return the deformed pointcloud
        mesh = pymesh.merge_meshes(output_meshes)

        return mesh
    
    def generate_ply(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import pymesh
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device)
                        for i in range(self.opt.nb_primitives)]
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = []
        for i in range(self.opt.nb_primitives):
            output_points.append(self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze().transpose(1, 0).contiguous().cpu().detach().numpy())
        print(output_points)
        return output_points
