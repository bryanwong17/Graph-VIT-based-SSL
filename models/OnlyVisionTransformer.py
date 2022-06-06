import torch
import torch.nn as nn


from .ViT import *
from .ccn import CCNBlock

from torch_geometric.nn import dense_mincut_pool
from torch.nn import Linear


class Classifier(nn.Module):
    def __init__(self, n_class):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        # input_dim:512, output_dim: 64, use torch.nn.BatchNorm1d(output_dim), add self (y += x), F.normalize
        self.conv1 = CCNBlock(2048,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0)       # 512->64
        #self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                          # 64->100
        self.pool1 =  nn.MaxPool1d(10, stride=5) 
        # self.relu1 = nn.ReLU()

    #As for what is mask, maybe refer to this paper https://yangliang.github.io/pdf/ijcai19_mask.pdf
    def forward(self,node_feat,labels,is_print=False, graphcam_flag=False):
        X=node_feat
        X = self.conv1(X)
        X = self.pool1(X.permute(0,2,1)).permute(0,2,1)

        #s = self.pool1(X) # linear transformation after convolution
        #s = self.relu1(s)
    
        
        #X = X.unsqueeze(0) if X.dim() == 2 else X
        #s = s.unsqueeze(0) if s.dim() == 2 else s


        #s = torch.softmax(s, dim=-1)

        #X = torch.matmul(s.transpose(1, 2), X)


        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)
        
        #Transformer layer
        out = self.transformer(X)

        # loss
        loss = self.criterion(out, labels)
        loss = loss 
        # pred
        pred = out.data.max(1)[1]

        return pred,labels,loss
