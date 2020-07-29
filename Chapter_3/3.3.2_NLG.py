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
        # Convert input to single channel. Size: batch x 1 x seq_lenxword_dim
        x_unsqueeze = x.unsqueeze(1) 
        # Go through CNN. Size: batch x out_channels x new_seq_len x 1
        x_cnn = self.cnn(x_unsqueeze) 
        # Remove the last dimension, size: batch x out_channels x new_seq_len
        x_cnn_result = x_cnn.squeeze(3) 
        # max-pooling, which computes the maximum value in the last dimension. Size: batch x out_channels        
        res, _ = x_cnn_result.max(2)  
        return res


class NLGNet(nn.Module):
    # word_dim: length of the word vector
    # window_size: length of the CNN window
    # rnn_dim: dimension of RNN’s hidden state
    # vocab_size: size of the vocabulary
    def __init__(self, word_dim, window_size, rnn_dim, vocab_size):
        super(NLGNet, self).__init__()
        # the word embedding matrix which returns word vector given word id
        self.embed = nn.Embedding(vocab_size, word_dim)  
        # CNN and max-pooling
        self.cnn_maxpool = CNN_Maxpool(word_dim, window_size, rnn_dim)
        # single-layer one-way GRU
        self.rnn = nn.GRU(word_dim, rnn_dim, batch_first=True) 
        # fully connected output layer, producing scores for each word
        self.linear = nn.Linear(rnn_dim, vocab_size)

    # Input:
    # x_id: word ids of input text. Size: batch x x_seq_len
    # y_id: word ids of ground-truth output text. Size: batch x y_seq_len
    # Output:
    # word_scores: score for each word at each position. Size: batch x y_seq_len x vocab_size
    def forward(self, x_id, y_id):
        # Obtain word vectors for input text. Size: batch x x_seq_len x word_dim
        x = self.embed(x_id) 
        # Obtain word vectors for ground-truth output text. Size: batch x y_seq_len x word_dim
        y = self.embed(y_id) 
        # text vector. Size: batch x cnn_channels
        doc_embed = self.cnn_maxpool(x)
        # use text vector as the initial RNN state. Size: 1 x batch x y_seq_len x rnn_dim
        h0 = doc_embed.unsqueeze(0)
        # RNN hidden states at each position. Size: batch x y_seq_len x rnn_dim
        rnn_output, _ = self.rnn(y, h0)
        # score for each word at each position. Size: batch x y_seq_len x vocab_size
        word_scores = self.linear(rnn_output)   
        return word_scores

vocab_size = 100                        # 100 words
net = NLGNet(10, 3, 15, vocab_size)     # set up the network
# 30 instances of input text, each containing 10 words
x_id = torch.LongTensor(30, 10).random_(0, vocab_size) 
# 30 instances of ground-truth output text, each containing 8 words
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=1) 
# score for each vocabulary word at each position. Size: 30 x 8 x vocab_size
word_scores = net(x_id, y_id)
# cross entropy loss function in PyTorch, which includes softmax operation
loss_func = nn.CrossEntropyLoss()
# convert word_scores into a 2D array and y_id into a 1D array to calculate the loss
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step() 
word_scores = net(x_id, y_id)
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss2 =', loss) # loss2 should be smaller than loss1
