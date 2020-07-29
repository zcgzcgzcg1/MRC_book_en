import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_Maxpool(nn.Module):
    # word_dim: dimension of word vectors
    # window_size: length of CNN window
    # out_channels: the number of output channels in CNN
    def __init__(self, word_dim, window_size, out_channels):
        super(CNN_Maxpool, self).__init__()
        # 1 input channel, out_channels output channels, and a filter with size window_size x word_dim
        self.cnn = nn.Conv2d(1, out_channels, (window_size, word_dim)) 
        
    # Input:
    # x is the text tensor. Size: batch x seq_len x word_dim
    # Output: 
    # res is the output text vectors, each of which has out_channels dimensions
    def forward(self, x):
        # Convert input to single channel. Size: batchx1xseq_lenxword_dim
        x_unsqueeze = x.unsqueeze(1) 
        # Go through CNN. Size: batch x out_channels x new_seq_len x 1
        x_cnn = self.cnn(x_unsqueeze) 
        # Remove the last dimension, size: batch x out_channels x new_seq_len
        x_cnn_result = x_cnn.squeeze(3) 
        # max-pooling, which computes the maximum value in the last dimension. Size: batch x out_channels        
        res, _ = x_cnn_result.max(2)  
        return res

class NLUNet(nn.Module):
    # word_dim: dimension of word vectors
    # window_size: window length of CNN
    # out_channels: number of output channels
    # K: number of categories
    def __init__(self, word_dim, window_size, out_channels, K):
        super(NLUNet, self).__init__()
        # CNN and max-pooling
        self.cnn_maxpool = CNN_Maxpool(word_dim, window_size, out_channels)  
        # fully connected output layer
        self.linear = nn.Linear(out_channels, K)     
    # Input:
    # x：input tensor. Size: batch x seq_len x word_dim
    # Output: 
    # class_score: predicted score for each class. Size: batch x K
    def forward(self, x):
        # text vector. Size: batch x out_channels
        doc_embed = self.cnn_maxpool(x)  
        # classification scores. Size: batch x K
        class_score = self.linear(doc_embed)     
        return class_score

K = 3     # 3 categories
net = NLUNet(10, 3, 15, K)
# 30 sequences, each with 5 words. Each word vector is 10D.
x = torch.randn(30, 5, 10, requires_grad=True)   
# 30 ground-truth category labels, which are integers from 0 to K-1
y = torch.LongTensor(30).random_(0, K) 
optimizer = optim.SGD(net.parameters(), lr=1)  
# size of res: batch x K
res = net(x)
# cross entropy loss function in PyTorch, which includes softmax operation
loss_func = nn.CrossEntropyLoss() 
loss = loss_func(res, y)
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
res = net(x)
loss = loss_func(res, y)
print('loss2 =', loss) # loss2 should be less than loss1