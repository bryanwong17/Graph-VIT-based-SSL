import torch
import pytorch_lightning as pl

import os
from datetime import datetime
from functools import partial

import moco.builder_lightning
import moco.loader
import moco.optimizer

import vits

import data_aug.dataset_wrapper_lightning as wrapper_lightning

torch.manual_seed(0)

def _load_pre_trained_weights(model, config):
        try:
            checkpoints_folder = os.path.join('./runs/mocov3/runs', config['fine_tune_from'])
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

class MocoV3(object):

    def __init__(self, config):
        self.config = config
        
    def train(self):

        start_time = datetime.now()
        print("Start Time: {start_time}")
        print("train")

        data_module = wrapper_lightning.MocoV3DataModule(self.config['batch_size'], **self.config['dataset'])
        train_loader, val_loader = data_module.get_data_loaders()

        model = moco.builder_lightning.MoCo_ViT(
            partial(vits.__dict__["vit_base"], stop_grad_conv1 = self.config['stop_grad_conv1']), self.config, train_loader, val_loader)

        # model = moco.builder_lightning.MoCo_ResNet(
        #     partial(torchvision.models.__dict__["resnet50"],  zero_init_residual=True), self.config, train_loader, val_loader)

        # check if there is any pretrained weights
        # model = _load_pre_trained_weights(model, self.config)
        trainer = pl.Trainer(precision=16, max_epochs=10, deterministic=True, gpus=1)
        trainer.fit(model, train_loader, val_loader)

        print(f"Training Execution time: {datetime.now() - start_time}")
        