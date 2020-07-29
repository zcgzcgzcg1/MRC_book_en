# install the transformer package which includes BERT
# pip install pytorch-transformers

# Python code
import torch
from pytorch_transformers import *
# use the case insensitive BERT-base model
config = BertConfig.from_pretrained('bert-base-uncased')
# the tokenization API used by BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# load the BERT variant specifically for extractive-answer question answering, which contains the output layer
model = BertForQuestionAnswering(config)
# preprocess the training data
# obtain the tokenized word id. Size: batch_size (1) × seq_length (4)
input_ids = torch.tensor(tokenizer.encode("This is an example")).unsqueeze(0) 
# record the start and end positions of the ground-truth answers in the article. Size: batch_size
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
# get the output from the model
outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
# get the cross entropy loss function, and the predicted scores for each article token being the start and end position of the answer. Size: batch_size (1) x seq_length
loss, start_scores, end_scores = outputs
print('Loss =', loss)
print('Start scores:', start_scores)
print('End scores:', end_scores)
