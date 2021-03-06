import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import grad_scaler, autocast_mode
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.nt_xent_try import NTXentLoss_try
import os
import shutil
import sys
from datetime import datetime


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


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device() # set to cuda if available
        self.writer = SummaryWriter() # use tensorboard
        self.dataset = dataset
        # loss from NTXentLoss
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss']) # batch_size: 256, # temperature: 0.5, use_cosine_similarity: True

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):
        
        # get the representations and representations after projections (latent vectors)
        ris, zis = model(xis)  # [N,C]
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # calculate contrastive learning loss
        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        start_time = datetime.now()
        print("Start Time: {start_time}")

        print("train")
        # get train and valid loader from function inside of dataset
        train_loader, valid_loader = self.dataset.get_data_loaders()
        # use resnet18 as base_model and output_dim = 512
        model = ResNetSimCLR(**self.config["model"])# .to(self.device)
        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model, device_ids=eval(self.config['gpu_ids']))
        # no pretrained model (training from scratch)
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)
            

        # learning rate: 1e-5, weight decay: 10e-6
        optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=eval(self.config['weight_decay']))

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
#                            cp                                    last_epoch=-1)
        # epochs: 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)
        
        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(model, optimizer,
        #                                       opt_level='O2',
        #                                       keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join('runs/simclr', self.writer.log_dir)

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        # initialize at first to infinity
        best_valid_loss = np.inf
        # scaler = grad_scaler.GradScaler()

        model.zero_grad()
        # epochs:20
        for epoch_counter in range(self.config['epochs']):
            for batch_idx, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    # self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("[%d/%d] step: %d train_loss: %.3f" % (epoch_counter, self.config['epochs'], n_iter, loss))
                
                # if apex_support and self.config['fp16_precision']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print("[%d/%d] val_loss: %.3f" % (epoch_counter, self.config['epochs'], valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss

                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved at {}'.format(os.path.join(model_checkpoints_folder, 'model.pth' )))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        print(f"Training Execution time: {datetime.now() - start_time}")
        
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs/simclr/runs', self.config['fine_tune_from'])
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.detach() #loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
