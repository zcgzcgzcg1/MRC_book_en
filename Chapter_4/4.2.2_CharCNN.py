import torch
import torch.nn as nn
import torch.nn.functional as F

class Char_CNN_Maxpool(nn.Module):
    # char_num: size of character dictionary
    # char_dim: length of character vector
    # window_size: window size of CNN
    # out_channels: number of output channels in CNN
    def __init__(self, char_num, char_dim, window_size, out_channels):
        super(Char_CNN_Maxpool, self).__init__()
        # character embedding matrix, with char_num vectors of a dimension of char_dim
        self.char_embed = nn.Embedding(char_num, char_dim)
        # 1 input channel, out_channels output channels, the filter size is window_size x char_dim
        self.cnn = nn.Conv2d(1, out_channels, (window_size, char_dim)) 
        
    # Input:
    # char_ids: character ids (0~char_num-1). Size: batch x seq_len x word_len. batch is the number of input text samples, seq_len is the maximum number of words in one sample, word_len is the maximum length of word. 
    # Output:
    # res: character embedding of all words. Size: batch x seq_len x out_channels
    def forward(self, char_ids):
        # obtain character embeddings from character ids. Size: batch x seq_len x word_len x char_dim
        x = self.char_embed(char_ids)
        # merge the first two dimensions to make a single input channel. Size: (batch x seq_len) x 1 x word_len x char_dim
        x_unsqueeze = x.view(-1, x.shape[2], x.shape[3]).unsqueeze(1) 
        # go through CNN layer. Size: (batch x seq_len) x out_channels x new_seq_len x 1
        x_cnn = self.cnn(x_unsqueeze) 
        # remove the last dimension. Size: (batch x seq_len) x out_channels x new_seq_len
        x_cnn_result = x_cnn.squeeze(3) 
        # max-pooling takes the maximum value over the last dimension. Size:(batch x seq_len) x out_channels
        res, _ = x_cnn_result.max(2)  
        return res.view(x.shape[0], x.shape[1], -1)

batch = 10
seq_len = 20
word_len = 12
char_num = 26
char_dim = 10
window_size = 3
out_channels = 8
char_cnn = Char_CNN_Maxpool(char_num, char_dim, window_size, out_channels)
char_ids = torch.LongTensor(batch, seq_len, word_len).random_(0, char_num - 1)
res = char_cnn(char_ids)
print(res.shape) # torch.Size([10, 20, 8])
