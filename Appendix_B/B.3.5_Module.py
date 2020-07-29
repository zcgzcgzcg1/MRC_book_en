import torch
import torch.nn as nn

# the custom network class inherits from nn.Module
class FirstNet(nn.Module):  
    # the constructor function
    # input_dim: dimension of input tensor
    # rnn_dim: input dimension of RNN
    # state_dim: dimension of RNN’s hidden state
    def __init__(self, input_dim, rnn_dim, state_dim):
        # call the parent class’s constructor function
        super(FirstNet, self).__init__()
        # fully connected layer with input dimension of input_dim and output dimension of rnn_dim
        self.linear = nn.Linear(input_dim, rnn_dim)
        # dropout layer with a dropout probability of 0.3
        self.dropout = nn.Dropout(0.3)
        # single-layer unidirectional GRU with input dimension of rnn_dim and hidden state dimension of state_dim
        self.rnn = nn.GRU(rnn_dim, state_dim, batch_first=True)  
    
    # forward computation function
    # Input:
    # x: the input tensor. Size: batch x seq_len x input_dim
    # Output:
    # result tensor. Size: batch x 1 x state_dim
    def forward(self, x):
        # dropout the outputs from the fully connected layer. Size: batchxseq_lenxrnn_dim
        rnn_input = self.dropout(self.linear(x))
        # the last hidden state of GRU. Size: 1xbatchxstate_dim
        _, hn = self.rnn(rnn_input) 
        # swap dim0 and dim1. Size: batch x 1 x state_dim
        return hn.transpose(0, 1)

net = FirstNet(10, 20, 15) # instantiate the class
# first dimension is batch
# 3 sequences of five elements, each with a 10D embedding
x = torch.randn(3, 5, 10)    
res = net(x)              # Size: 3 x 1 x 15
print(res.shape)          # torch.Size([3, 1, 15])