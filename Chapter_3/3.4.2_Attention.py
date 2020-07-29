import torch
import torch.nn as nn
import torch.nn.functional as F

'''
  a: the attended items/vectors. Size: batch x m x dim 
  x: the items/vectors attending to a. Size: batch x n x dim
'''
def attention(a, x):
    # use inner product to compute attention scores. Size: batch x n x m
    scores = x.bmm(a.transpose(1, 2))
    # softmax over the last dimension
    alpha = F.softmax(scores, dim=-1)
    # attention vector. Size: batch x n x dim
    attended = alpha.bmm(a) 
    return attended

batch = 10
m = 20
n = 30
dim = 15
a = torch.randn(batch, m, dim)
x = torch.randn(batch, n, dim)
res = attention(a, x)
print(res.shape) # torch.Size([10, 30, 15])

