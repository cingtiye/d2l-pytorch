import sys
sys.path.append('../')
import d2l
import torch
from torch import nn
# atten = d2l.DotProductAttention(dropout=0.)
# keys = torch.ones((2,10,2),dtype=torch.float)
# values = torch.arange((40), dtype=torch.float).view(1,10,4).repeat(2,1,1)
# atten(torch.ones((2,1,2),dtype=torch.float), keys, values, torch.FloatTensor([2, 6]))
#
# atten = d2l.MLPAttention(ipt_dim=2,units = 8, dropout=0)
# atten(torch.ones((2,1,2), dtype = torch.float), keys, values, torch.FloatTensor([2, 6]))

layernorm = nn.LayerNorm(normalized_shape=2, elementwise_affine=True)
batchnorm = nn.BatchNorm1d(num_features=2, affine=True)
X = torch.FloatTensor([[1,2], [3,4]])
print('layer norm:', layernorm(X))
print('batch norm:', batchnorm(X))
# torch.nn.DataParallel(model)

d2l.load_data_fashion_mnist(batch_size=32, resize=None, root='../data/')