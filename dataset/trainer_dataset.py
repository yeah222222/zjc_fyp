import torch
import os,sys
sys.path.append('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc')
import dataset.dataset_shapenet_SVR as dataset_shapenet

from easydict import EasyDict
import dataset.my_utils as my_utils

class TrainerDataset(object):
    def __init__(self, opt):  
        self.opt = opt  
        super(TrainerDataset, self).__init__()
        
        
    def build_dataset(self):
        """
        Create dataset
        Author : Thibault Groueix 01.11.2019
        """

        self.datasets = EasyDict()
        # Create Datasets
        self.datasets.dataset_train = dataset_shapenet.ShapeNet(self.opt, train=True)
        self.datasets.dataset_test = dataset_shapenet.ShapeNet(self.opt, train=False)
        
        if not self.opt.demo:
        # Create dataloaders
            self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                            batch_size=self.opt.batch_size,
                                                                            shuffle=True)
                                                                            #num_workers=int(self.opt.workers))
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=True) 
                                                                        #num_workers=int(self.opt.workers)) 

            self.datasets.len_dataset = len(self.datasets.dataset_train)
            self.datasets.len_dataset_test = len(self.datasets.dataset_test)
        
        return self.datasets
