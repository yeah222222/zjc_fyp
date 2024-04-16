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
from transformers import get_scheduler
from accelerate import Accelerator
sys.path.append('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc')
from model.optim import AdamOptimizerConfig,CosineSchedulerConfig,RunConfig,OptimizerConfig,compute_grad_norm
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
     
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
    if RELOAD_TRAINING_MODEL is None:    
        print('jinru xiexie')
        network = PointCloudColoringModel(opt=opt, point_cloud_model='pvcnn', point_cloud_model_layers=1, point_cloud_model_embed_dim=64)
        network =  network.to(opt.device)
    else:
        my_utils.green_print('Reload continue training model')
        network = PointCloudColoringModel(opt=opt, point_cloud_model='pvcnn', point_cloud_model_layers=1, point_cloud_model_embed_dim=64).to(opt.device)
        network.load_state_dict(torch.load(RELOAD_TRAINING_MODEL, map_location='cuda:0'))
#Accelerator initialize

    accelerator = Accelerator(mixed_precision=RunConfig().mixed_precision, cpu=RunConfig().cpu, 
    gradient_accumulation_steps=OptimizerConfig().gradient_accumulation_steps)
    
#load optimizer
    optimizer_config = AdamOptimizerConfig()

    no_decay = ["bias", "LayerNorm.weight"]
    parameters = [
        {
            "params": [p for n, p in network.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in network.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # 根据配置创建优化器
    if optimizer_config.name == 'AdamW':
        optimizer = optim.AdamW(parameters,
                                lr=optimizer_config.lr,
                                **optimizer_config.kwargs)
        
#load scheduler
    scheduler = get_scheduler(optimizer=optimizer, **CosineSchedulerConfig().kwargs)     
      
#load dataset
    trainer_dataset =  TrainerDataset(opt=opt)    
    datasets_dict = trainer_dataset.build_dataset()
    
#Accelerator config
    dataloader_train = datasets_dict.dataloader_train
    dataloader_val = datasets_dict.dataloader_test
    network, optimizer, scheduler, dataloader_train, dataloader_val = accelerator.prepare(
            network, optimizer, scheduler, dataloader_train, dataloader_val)
#LOG
    logger = get_logger('/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/log/model_coloring_pymesh5.log',verbosity=1, name="my_logger") 
#Training    
    iteration = 0
#Training no accelerator
    # for epoch in range(0, opt.nepoch):
        
    #     logger.info("-----The {} Epoch-----".format(epoch))
    #     network.train()
    #     iterator = dataloader_train.__iter__()
    #     for dic in iterator:  
        
    #         # # data augmentation

    #         # if datasets_dict.data_augmenter is not None and not opt.SVR:
    #         #     datasets_dict.data_augmenter(dic['image'])
    #         dic = {k: v.to(opt.device) for k, v in dic.items() if isinstance(v, torch.Tensor)}
    #         optimizer.zero_grad()
    #         #increment iteration
    #         iteration += 1
    #         #network input
    #         loss = network(dic)
            
    #         logger.info("Iteration: {} Loss: {}".format(iteration, loss))
            
    #         # loss.backward()
    #         accelerator.backward(loss)
    #         optimizer.step()
    #         scheduler.step()
    #         logger.info("Learning rate: %f" % (opt.lrate))
    for epoch in range(401, opt.nepoch):
        logger.info("-----The {} Epoch-----".format(epoch))
        network.train()
        iterator = dataloader_train.__iter__()
        for i, dic in enumerate(iterator):
            # 需要检查是否达到了训练批次的限制
            if (RunConfig.limit_train_batches is not None) and (i >= RunConfig.limit_train_batches):
                break

            dic = {k: v.to(opt.device) for k, v in dic.items() if isinstance(v, torch.Tensor)}
            network.train()

            # Gradient accumulation
            with accelerator.accumulate(network):
                loss = network(dic)
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if OptimizerConfig().clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(network.parameters(), OptimizerConfig().clip_grad_norm)
                    grad_norm_clipped = compute_grad_norm(network.parameters())

                if (i + 1) % accelerator.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    iteration += 1
                    
                
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                else:
                    logger.info("Iteration: {} Loss: {}".format(iteration, loss))
                    logger.info("grad_norm_clipped: {}".format(grad_norm_clipped))
                

        # test epoch
        if epoch % 15== 0:
            iterator_test = dataloader_val.__iter__()
            with torch.no_grad():
                network.eval()
                for dic_test in iterator_test:
                    dic_test = {k: v.to(opt.device) for k, v in dic_test.items() if isinstance(v, torch.Tensor)}
                    loss = network(dic_test)
                    logger.info("Validation loss in the validation set is {}".format(loss))
        
        #save network
        if epoch % 5== 0:
            logger.info("saving net...")
            if accelerator.is_main_process:
                accelerator.save(network.state_dict(), '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/ckpt/modelcoloring_pymesh_RTK/'+ str(epoch) + '-model.pth')
                accelerator.save(optimizer.state_dict(), '/mnt/d/junch_data/projection_conditioned_point_cloud_diffusion/Projection_zjc/ckpt/modelcoloring_pymesh_RTK/'+ str(epoch) + '-optim.pth')
                logger.info("network saved")
    
    accelerator.wait_for_everyone()
    

    
        
if __name__ == '__main__':
    main()
    