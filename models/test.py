import torch
import torch.nn as nn


#from .ViT import *
#from .ccn import CCNBlock

from torch_geometric.nn import dense_mincut_pool
from torch.nn import Linear

x = torch.rand(5, 128, 32)
pool = nn.MaxPool1d(2, 2)
print((pool(x.permute(0,2,1)).permute(0,2,1)).shape)  # shape (5, 128, 32) -> (5, 64, 32)