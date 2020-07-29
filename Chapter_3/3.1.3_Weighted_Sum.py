import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSum(nn.Module):
    # word_dim: dimension of input word vectors
    def __init__(self, word_dim):
        super(WeightedSum, self).__init__()
        self.b = nn.Linear(word_dim, 1) # the parameter vector

    # Input:
    # x is the input tensor. Size: batch x seq_len x word_dim
    # Output:
    # res is the text tensor. Size: batch x word_dim
    def forward(self, x):
        # score by inner product. Size: batch x seq_len x 1
        scores = self.b(x)
        # softmax operation. Size: batch x seq_len x 1
        weights = F.softmax(scores, dim = 1) 
        # The weighted sum is computed by matrix multiplication. Size: batch x word_dim x 1
        res = torch.bmm(x.transpose(1, 2), weights)  
        # Delete the last dimension. Size: batch x word_dim
        res = res.squeeze(2)  
        return res

batch = 10
seq_len = 20
word_dim = 50
x = torch.randn(batch, seq_len, word_dim)
weighted_sum = WeightedSum(word_dim)
res = weighted_sum(x)
print(res.shape) # torch.Size([10, 50])