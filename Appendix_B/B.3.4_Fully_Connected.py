import torch
import torch.nn as nn
# a network consisting of four fully-connected layers
# input layer size is 30, two hidden layers have size 50 and 70, and the output layer size is 1
linear1 = nn.Linear(30, 50)
linear2 = nn.Linear(50, 70)
linear3 = nn.Linear(70, 1)
# one batch consisting of 10 input data samples, each a 30D tensor
x = torch.randn(10, 30)
# 10 outputs, each a 1D tensor
res = linear3(linear2(linear1(x)))   
print(res.shape) # torch.Size([10, 1])