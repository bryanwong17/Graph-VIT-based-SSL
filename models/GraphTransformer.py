import torch
import torch.nn as nn


from .ViT import *
from .gcn import GCNBlock

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
        self.conv1 = GCNBlock(2048,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                          # 100-> 20
        # self.relu1 = nn.ReLU()

    def forward(self,node_feat,labels,adj,mask,is_print=False, graphcam_flag=False):
        X=node_feat
        X=mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)
        # s = self.relu1(s)
    
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # loss
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        # pred
        pred = out.data.max(1)[1]

        return pred,labels,loss
