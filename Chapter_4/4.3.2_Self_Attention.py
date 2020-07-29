import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    # dim is the dimension of input vectors
    # hidden_dim is the (lower) projection dimension in attention
    def __init__(self, dim, hidden_dim):
        super(SelfAttention, self).__init__()
        # parameter matrix W
        self.W = nn.Linear(dim, hidden_dim)

    # Input:
    # x is the group of vectors for self-attention. Size: batch x n x dim
    # Output:
    # attended is the attention vectors. Size: batch x n x dim
    def forward(self, x):
        # project x to a lower dimensional space. Size: batch x n x hidden_dim
        hidden = self.W(x)
        # compute attention scores. Size: batch x n x n
        scores = hidden.bmm(hidden.transpose(1, 2))
        # softmax over the last dimension
        alpha = F.softmax(scores, dim=-1)
        # attention vectors. Size: batch x n x dim
        attended = alpha.bmm(x) 
        return attended

batch = 10
n = 15
dim = 40
hidden_dim = 20
x = torch.randn(batch, n, dim)
self_attention = SelfAttention(dim, hidden_dim)
res = self_attention(x)
print(res.shape) # torch.Size([10, 15, 40])