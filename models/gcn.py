import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=2,threshold=float('inf'))

# GCN basic operation (operates message propagation and aggregation in the graph)
class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bn=0,add_self=0, normalize_embedding=0,
            dropout=0.0,relu=0, bias=True):
        super(GCNBlock,self).__init__()
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

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index),y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index=[]
            ssum=0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]]=y[i,0:index[i]]
            bn_tensor_bf=self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i,0:index[i]]=bn_tensor_bf[start_index[i]:start_index[i+1]]
            y=bn_tensor_af
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu=='relu':
            y=torch.nn.functional.relu(y)
            print('hahah')
        elif self.relu=='lrelu':
            y=torch.nn.functional.leaky_relu(y,0.1)
        return y

#experimental function, untested
class masked_batchnorm(nn.Module):
    def __init__(self,feat_dim,epsilon=1e-10):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(feat_dim))
        self.beta=nn.Parameter(torch.zeros(feat_dim))
        self.eps=epsilon

    def forward(self,x,mask):
        '''
        x: node feat, [batch,node_num,feat_dim]
        mask: [batch,node_num]
        '''
        mask1 = mask.unsqueeze(2)
        mask_sum = mask.sum()
        mean = x.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        temp = (x - mean)**2
        temp = temp*mask1
        var = temp.sum(dim=(0,1),keepdim=True)/(self.eps+mask_sum)
        rstd = torch.rsqrt(var+self.eps)
        x=(x-mean)*rstd
        return ((x*self.alpha) + self.beta)*mask1