import torch
import torch.nn as nn

class BiRNN(nn.Module):
    # word_dim: dimension of word vectors
    # hidden_size: dimension of RNN hidden states
    def __init__(self, word_dim, hidden_size):
        super(BiRNN, self).__init__()
        # Bi-directional GRU, where the first dimension is the batch size
        self.gru = nn.GRU(word_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True) 

    # Input: 
    # Tensor x. Size: batch x seq_len x word_dim
    # Output: 
    # The text tensor. Size: batch x (2 x hidden_size)
    def forward(self, x):
        batch = x.shape[0]
        # output is the hidden states in the last RNN layer for each word. Size: batch x seq_len x (2 x hidden_size)
        # last_hidden is the RNN’s hidden state for the last word. Size: 2 x batch x hidden_size
        output, last_hidden = self.gru(x)
        return last_hidden.transpose(0,1).contiguous().view(batch, -1)  

batch = 10
seq_len = 20
word_dim = 50
hidden_size = 100
x = torch.randn(batch, seq_len, word_dim)
birnn = BiRNN(word_dim, hidden_size)
res = birnn(x)
print(res.shape) # torch.Size([10, 200])
