# download source code of CoVe from GitHub
# git clone https://github.com/salesforce/cove.git
# cd cove
# install the required packages
# pip install -r requirements.txt
# install CoVe
# python setup.py develop

# Python code
import torch
from torchtext.vocab import GloVe
from cove import MTLSTM
# GloVe embeddings. Size: 2.1M × 300
glove = GloVe(name='840B', dim=300, cache='.embeddings')
# two input sentences with word ids
inputs = torch.LongTensor([[10, 2, 3, 0], [7, 8, 10, 3]])
# length of the sentences
lengths = torch.LongTensor([3, 4])
# MTLSTM is the class for CoVe
cove = MTLSTM(n_vocab=glove.vectors.shape[0], vectors=glove.vectors, model_cache='.embeddings')
# contextual embeddings from Cove for each word in each sentence. Size: 2 × 4 × 600
outputs = cove(inputs, lengths)
print(outputs.shape) # torch.Size([2, 4, 600])
