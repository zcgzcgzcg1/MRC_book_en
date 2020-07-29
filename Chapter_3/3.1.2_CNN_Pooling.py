import torch
import torch.nn as nn
import torch.nn.functional as F

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


batch = 10
seq_len = 20
word_dim = 50
window_size = 3
out_channels = 100
x = torch.randn(batch, seq_len, word_dim)
cnn_maxpool = CNN_Maxpool(word_dim, window_size, out_channels)
res = cnn_maxpool(x)
print(res.shape) # torch.Size([10, 100])