import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
  a: the attended items/vectors. Size: batch x m x dim 
  x: the items/vectors attending to a. Size: batch x n x dim
'''
def attention(a, x):
    # use inner product to compute attention scores. Size: batch x n x m
    scores = x.bmm(a.transpose(1, 2))
    # softmax over the last dimension
    alpha = F.softmax(scores, dim=-1)
    # attention vector. Size: batch x n x dim
    attended = alpha.bmm(a) 
    return attended

class Seq2SeqOutputLayer(nn.Module):
    # word_dim is the dimension of question vector and article word vectors from the interaction layer
    # embed is the word embedding module from the encoding layer, i.e. nn.Embedding(vocab_size, word_dim)
    # vocab_size is the size of the vocabulary
    def __init__(self, embed, word_dim, vocab_size):
        super(Seq2SeqOutputLayer, self).__init__()
        # reuse the word embedder from the encoding layer
        self.embed = embed
        self.vocab_size = vocab_size
        # encoder RNN of a single-layer one-way GRU
        self.encoder_rnn = nn.GRU(word_dim, word_dim, batch_first=True)
        # decoder RNN cell (GRU)
        self.decoder_rnncell = nn.GRUCell(word_dim, word_dim)
        # convert the concatenation of hidden state and attention vector into a vector of dimension word_dim
        self.combine_state_attn = nn.Linear(word_dim + word_dim, word_dim)
        # fully connected layer for generating word scores
        self.linear = nn.Linear(word_dim, vocab_size, bias=False)
        # share parameters between the fully connected layer and word embedder
        self.linear.weight = embed.weight
        
    # Input:
    # x: article word vectors from the interaction layer. Size: batch x x_seq_len x word_dim
    # q: question vector from the interaction layer. Size: batch x word_dim
    # y_id: word ids of the ground-truth answers. Size: batch x y_seq_len
    # Output:
    # scores: predicted scores of each answer word. Size: batch x y_seq_len x vocab_size
    def forward(self, x, q, y_id):
        # embed the ground-truth answer words. Size: batch x y_seq_len x word_dim
        y = self.embed(y_id) 
        # the encoder RNN takes question vector q as the initial state
        # enc_states are the hidden states of each article word. Size: batch x x_seq_len x word_dim
        # enc_last_state is the last hidden state. Size: 1 x batch x word_dim
        enc_states, enc_last_state = self.encoder_rnn(x, q.unsqueeze(0))
        # enc_last_state is used as the initial state of the decoder RNN. Size: batch x word_dim
        prev_dec_state = enc_last_state.squeeze(0)
        # scores contains the scores assigned to answer words at each position
        scores = torch.zeros(y_id.shape[0], y_id.shape[1], self.vocab_size)
        for t in range(0, y_id.shape[1]):
            # feed the previous hidden state and the word vector of the t-th word in the ground-truth answer into the decoder RNN. Size: batch x word_dim
            new_state = self.decoder_rnncell(y[:,t,:].squeeze(1), prev_dec_state)     
            # use the attention function in Section 3.4 to obtain the attention vector. Size: batch x word_dim
            context = attention(enc_states, new_state.unsqueeze(1)).squeeze(1)
            # convert the concatenation of hidden state and attention vector into word_dim-dimension. Size: batch x word_dim
            new_state = self.combine_state_attn(torch.cat((new_state, context), dim=1))
            # predict the scores for all dictionary words at the t-th position
            scores[:, t, :] = self.linear(new_state)
            # pass the new hidden state to the next GRU cell
            prev_dec_state = new_state
        return scores

# 100 words
vocab_size = 100
# word vectors are 20D
word_dim = 20
embed = nn.Embedding(vocab_size, word_dim)
# word ids of 30 ground-truth answers, each with 8 words
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
# the encoding and interaction layers are omitted here, replaced by randomized tensors
# the interaction layer outputs:
# 1) article word vector x. Size: 30 x x_seq_len x word_dim
# 2) question vector q. Size: 30 x word_dim
x = torch.randn(30, 10, word_dim)
q = torch.randn(30, word_dim)
# set up the network
net = Seq2SeqOutputLayer(embed, word_dim, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=0.1) 
# get the scores for words at each position. Size: 30 x y_seq_len x vocab_size
word_scores = net(x, q, y_id)
# nn.CrossEntropyLoss contains both softmax and cross entropy computation
loss_func = nn.CrossEntropyLoss()
# convert word_scores into a 2D tensor and y_id into a 1D tensor to calculate the loss
# word_scores contain scores for the 2nd, 3rd, 4th... word, so it is offset by 1 position before comparing with y_id
loss = loss_func(word_scores[:,:-1,:].contiguous().view(-1, vocab_size), y_id[:,1:].contiguous().view(-1))
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
word_scores = net(x, q, y_id)
loss = loss_func(word_scores[:,:-1,:].contiguous().view(-1, vocab_size), y_id[:,1:].contiguous().view(-1))
print('loss2 =', loss) # loss2 should be less than loss1