import torch
import torch.nn as nn

layer = nn.Dropout(0.1)   # Dropout layer with a dropout probability of 0.1
input = torch.randn(5, 2)
print(input)
output = layer(input)     # the output tensor has size 5 x 2，with each element being 0 with a probability of 10%
print(output)