import torch
import torch.nn as nn

class Contextual_Embedding(nn.Module):
    # word_dim is the dimension of word vectors
    # state_dim is the dimension of hidden states in RNN
    # rnn_layer is the number of RNN layers
    def __init__(self, word_dim, state_dim, rnn_layer):
        super(Contextual_Embedding, self).__init__()
        # multi-layer bidirectional GRU with input dimension word_dim and hidden state dimension state_dim
        self.rnn = nn.GRU(word_dim, state_dim, num_layers=rnn_layer, bidirectional=True, batch_first=True)  
        
    # Input:
    # x: the input word vectors. Size: batch x seq_len x word_dim. batch is number of text samples, seq_len is the maximum number of words in one text sample, word_dim is the dimension of word vectors.
    # Output:
    # res: the contextual embeddings for all words. Size: batch x seq_len x out_dim
    def forward(self, x):
        # contextual embedding. Size: batch x seq_len x out_dim, where out_dim  = 2 x state_dim.
        res, _ = self.rnn(x) 
        return res

batch = 10
seq_len = 20
word_dim = 50
state_dim = 100
rnn_layer = 2
x = torch.randn(batch, seq_len, word_dim)
context_embed = Contextual_Embedding(word_dim, state_dim, rnn_layer)
res = context_embed(x)
print(res.shape) # torch.Size([10, 20, 200])