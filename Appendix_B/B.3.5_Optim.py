import torch
import torch.nn as nn
import torch.optim as optim   # optimization package of PyTorch

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

net = FirstNet(10, 20, 15)
net.train()              # set FirstNet to training mode (e.g. dropout is used)
#net.cuda()              # put the network in GPU (if GPU is available)

# randomly initialize the training data
# 30 input sequences, each with 5 elements represented by a 10D tensor
x = torch.randn(30, 5, 10)  
y = torch.randn(30, 1)       # 30 ground-truth values
# stochastic gradient descent (SGD) optimizer, with a learning rate of 0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)  
for batch_id in range(10):
    # get the current data batch (batch_size=3)
    x_now = x[batch_id * 3: (batch_id + 1) * 3]
    y_now = y[batch_id * 3 : (batch_id + 1) * 3]
    res = net(x_now)                      # the result is of size 3 x 1 x 15
    y_hat, _ = torch.max(res, dim=2)      # use max-pooling to get the prediction y_hat，of size 3 x 1
    # mean square error (MSE) loss function
    loss = torch.sum(((y_now - y_hat) ** 2.0)) / 3  
    print('loss =', loss)
    optimizer.zero_grad()                   # clears gradients from previous batches
    loss.backward()                         # automatic backpropagation
    optimizer.step()                        # the optimizer updates the parameters in the opposite direction of derivatives

net.eval()         # set FirstNet to evaluation mode (e.g. dropout is not applied in this mode)
y_pred = net(x)    # get output in evaluation mode