import torch
import torch.nn as nn

# a CNN with 1 input channel, 3 output channels and filters of size 5 x 5
conv = nn.Conv2d(1, 3, 5)
# one batch consisting of 10 input data samples, each a single-channel tensor of size 32 x 32
x = torch.randn(10, 1, 32, 32)  
# the output tensor has a size of 10 x 3 x 28 x 28: 10 data samples, each a 3-channel tensor of size 28 x 28 (28=32-5+1)
y = conv(x)
print(y.shape) # torch.Size([10, 3, 28, 28]