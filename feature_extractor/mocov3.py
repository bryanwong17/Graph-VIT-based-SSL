import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import grad_scaler, autocast_mode
import torch.nn.functional as F

import os
import math
import shutil
from datetime import datetime
from functools import partial

import moco.builder
import moco.loader
import moco.optimizer

import vits

import numpy as np

torch.manual_seed(0)


# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp

#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False
    
def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        # copy and put config.yaml to the model_checkpoints_folder
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class MocoV3(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device() # set to cuda if available
        self.writer = SummaryWriter() # use tensorboard
        self.dataset = dataset

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def adjust_learning_rate(self, optimizer, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        # warmup_epochs: 10
        if epoch < self.config['warmup_epochs']:
            lr = self.config['learning_rate'] * epoch / self.config['warmup_epochs'] 
        else:
            lr = self.config['learning_rate'] * 0.5 * (1. + math.cos(math.pi * (epoch - self.config['warmup_epochs']) / (self.config['epochs'] - self.config['warmup_epochs'])))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        

    def adjust_moco_momentum(self, epoch):
        """Adjust moco momentum based on current epoch"""
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.config['epochs'])) * (1. - self.config['moco_m'])
        return m

    def train(self):

        start_time = datetime.now()
        print("Start Time: {start_time}")

        print("train")
        # get train and valid loader from function inside of dataset
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__["vit_base"], stop_grad_conv1 = self.config['stop_grad_conv1']),
            self.config['moco_dim'], self.config['moco_mlp_dim'], self.config['moco_t'])
        
        # model = moco.builder.MoCo_ResNet(
        #     partial(torchvision.models.__dict__["resnet50"],  zero_init_residual=True),
        #     self.config['moco_dim'], self.config['moco_mlp_dim'], self.config['moco_t'])
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model, device_ids=eval(self.config['gpu_ids']))

        # check if there is any pretrained weights
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)
            

        # self.config['learning_rate'] = self.config['learning_rate'] * self.config['batch_size'] / 256
        optimizer = torch.optim.AdamW(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))

        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(model, optimizer,
        #                                     opt_level='O2',
        #                                     keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join('runs/mocov3', self.writer.log_dir)

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        # initialize at first to infinity
        best_valid_loss = np.inf
        scaler = grad_scaler.GradScaler()

        moco_m = self.config['moco_m']
        iter_per_epochs = len(train_loader)
        # epochs:20_
        for epoch_counter in range(self.config['epochs']):
            for batch_idx, (xis, xjs) in enumerate(train_loader):
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                self.adjust_learning_rate(optimizer, epoch_counter + batch_idx / iter_per_epochs)
                if self.config['moco_m_cos']:
                    moco_m = self.adjust_moco_momentum(epoch_counter + batch_idx / iter_per_epochs)

                loss = model(xis, xjs, moco_m)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("[%d/%d] step: %d train_loss: %.3f" % (epoch_counter, self.config['epochs'], n_iter, loss))
                
                # if apex_support and self.config['fp16_precision']:
                #     print("AMP IS WORKING")
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, moco_m)
                print("[%d/%d] val_loss: %.3f" % (epoch_counter, self.config['epochs'], valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss

                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved at {}'.format(os.path.join(model_checkpoints_folder, 'model.pth' )))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        print(f"Training Execution time: {datetime.now() - start_time}")
        
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs/mocov3/runs', self.config['fine_tune_from'])
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, moco_m):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # MocoV3
                loss = model(xis, xjs, moco_m)
                valid_loss += loss.detach() #loss.item()
                counter += 1
            valid_loss /= counter
        # model.train()
        return valid_loss
