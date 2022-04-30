

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np
import shutil
import os
import pytorch_lightning as pl

import data_aug.dataset_wrapper_lightning as wrapper_lightning

class MoCo(pl.LightningModule):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def _save_config_file(self, model_checkpoints_folder):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            # copy and put config.yaml to the model_checkpoints_folder
            shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

    def __init__(self, base_encoder, config, train_loader, val_loader):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()
        
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.dim = config['moco_dim']
        self.mlp_dim = config['moco_mlp_dim']
        self.T = config['moco_t']

        self.moco_m = config['moco_m']
        self.moco_m_cos = config['moco_m_cos']

        self.writer = SummaryWriter()
        self.model_checkpoints_folder = os.path.join('runs/mocov3', self.writer.log_dir)

        self._save_config_file(self.model_checkpoints_folder)

        # metrics
        self.train_loss_epoch = []
        self.val_loss_epoch = []

        # training
        self.best_val_loss = np.inf
        self.n_iter = 0

        # build encoders
        self.base_encoder = base_encoder(num_classes=self.mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=self.mlp_dim)


        self._build_projector_and_predictor_mlps(self.dim, self.mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim # layer = 0 -> input_dim, else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim # last layer -> output_dim, else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False)) # without learnable parameters
                                                               # random patch projection -> stop gradient right after patch projection 
        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) # matrix multiplication(q, k.t())
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda() 
        return nn.CrossEntropyLoss()(logits / self.T, labels) * (2 * self.T) # tau = T = temperature

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features (encoder backbone + remove fc layer -> projection mlp + predictor)
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder -> f_k = m * f_k + (1-m) * f_q

            # compute momentum features as targets (momentum encoder backbone + remove fc layer -> projection mlp)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        # total loss
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    
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
    
    def training_step(self, batch, batch_idx):
        xis, xjs = batch
        iter_per_epochs = len(batch)
        self.adjust_learning_rate(self.optimizer, self.current_epoch + batch_idx / iter_per_epochs)
        # loss for each step
        loss = self.forward(xis, xjs, self.moco_m)

        if self.moco_m_cos:
            self.moco_m = self.adjust_moco_momentum(self.current_epoch + batch_idx / iter_per_epochs)

        if(self.n_iter % self.config['log_every_n_steps']):
            print("[%d]/[%d] step: %d train_loss: %.3f" %(self.current_epoch, self.config['epochs'], self.n_iter, loss))
        
        self.n_iter += 1
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, step_results):
        step_losses = [x['loss'] for x in step_results]
        epoch_loss = torch.tensor(step_losses).mean().item()
        self.train_loss_epoch.append(epoch_loss)

        # Print results (take last one to get recent epoch)
        train_loss = self.train_loss_epoch[-1]
        val_loss = self.val_loss_epoch[-1]
        print(f"{self.current_epoch + 1} - Train loss: {train_loss:.3f} Val loss: {val_loss:.3f}")
    
    def validation_step(self, batch, batch_idx):
        xis, xjs = batch
        loss = self.forward(xis, xjs, self.moco_m)
        self.log('val_loss', loss)
    
    def validation_epoch_end(self, step_results):
        step_losses = [x['val_loss'] for x in step_results]
        epoch_loss = torch.tensor(step_losses).mean().item()
        self.val_loss_epoch.append(epoch_loss)

        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            torch.save(self.state_dict(), os.path.join(self.model_checkpoints_folder, 'model.pth'))
            print('saved at {}'.format(os.path.join(self.model_checkpoints_folder, 'model.pth' )))

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learning_rate'], weight_decay=eval(self.config['weight_decay']))
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    

class MoCo_ResNet(MoCo):
    # polymorphism
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fully connected (fc) layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim) # num_layers: 2, input: hidden_dim, output: dim
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False) # last_bn = False


class MoCo_ViT(MoCo):
    # polymorphism
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fully connected (fc) layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
