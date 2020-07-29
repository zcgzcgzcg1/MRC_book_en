# install the allennlp package
# pip install allennlp / pip3 install allennlp

# Python code (version 3.6 required)
import torch
from torch import nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.nn.util import remove_sentence_boundaries
# url to the pre-trained model
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# the ELMo class
elmo_bilm = ElmoEmbedder(options_file, weight_file).elmo_bilm
elmo_bilm.cuda()
sentences = [['Today', 'is', 'sunny', '.'], ['Hello', '!']]
# obtain character ids for each word. Size: batch_size × max_sentence_len × word_len
character_ids = batch_to_ids(sentences).cuda()
# ELMo's output
bilm_output = elmo_bilm(character_ids)
# ELMo embeddings for each layer
layer_activations = bilm_output['activations']
# indicate whether there is a word at each position
mask_with_bos_eos = bilm_output['mask']
# remove the special sentence start and end symbols added by ELMo
without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos) for layer in layer_activations]
# three layers of 1024D ELMo embeddings. Size: 3 × batch_size × max_sentence_len × 1024
all_layers = torch.cat([ele[0].unsqueeze(0) for ele in without_bos_eos], dim=0)
# paraemeters for weighted sum
s = nn.Parameter(torch.Tensor([1., 1., 1.]), requires_grad=True).cuda()
# normalize the weights 
s = F.softmax(s, dim=0)
# the multiplier γ
gamma = nn.Parameter(torch.Tensor(1, 1), requires_grad=True).cuda()
# ELMo embedding. Size: batch_size × max_sentence_len × 1024
res = (all_layers[0]*s[0]+ all_layers[1]*s[1]+ all_layers[2]*s[2]) * gamma
print(res.shape)
