import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=2,threshold=float('inf'))

# GCN basic operation (operates message propagation and aggregation in the graph)
class CCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0,add_self=0, normalize_embedding=0,
            dropout=0.0,relu=0, bias=True):
        super(CCNBlock,self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu=relu
        self.bn=bn
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x):
        #y = torch.matmul(adj, x)
        y=0
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu=='relu':
            y=torch.nn.functional.relu(y)
            print('hahah')
        elif self.relu=='lrelu':
            y=torch.nn.functional.leaky_relu(y,0.1)
        return y
