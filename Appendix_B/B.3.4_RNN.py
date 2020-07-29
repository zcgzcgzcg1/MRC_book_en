import torch
import torch.nn as nn

# 2-layer GRU with an input dimension of 10, and hidden state dimension of 20. The second dimension of the input tensor is the batch.
rnn = nn.GRU(10, 20, num_layers=2)    
# one batch contains 3 sequences of length 5. Each element is represented by a 10D embedding.
x = torch.randn(5, 3, 10) 
# initial 20D hidden state for 3 sequences, 2 layers
h0 = torch.randn(2, 3, 20)
# output contains all RNN states in the last layer, of size 5 x 3 x 20
# hn contains the last element’s hidden state, of size 2 x 3 x 20
output, hn = rnn(x, h0) 

print(output.shape) # torch.Size([5, 3, 20])
print(hn.shape) # torch.Size([2, 3, 20])