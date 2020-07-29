import torch
import numpy as np
import torch.nn.functional as F

# Input:
# the article has m words
# prob_s of size m is the probability that the answer starts from each article word
# prob_e of size m is the probability that the answer ends at each article word
# L is the maximum number of words in the predicted answer
# Output:
# best_start, best_end indicate the start and end positions with the highest probability
def get_best_interval(prob_s, prob_e, L):
    # obtain an m x m matrix prob, where prob[i, j] = prob_s[i] x prob_e[j]
    prob = torch.ger(prob_s, prob_e) 
    # limit prob to contain only valid start-end position pairs, i.e. prob[i, j]=0 if i>j or j-i+1>L
    prob.triu_().tril_(L - 1) 
    # convert into a numpy array
    prob = prob.numpy()
    # the predicted answer starts from the best_start-th word and ends at  the best_end-th word, with the highest probability
    best_start, best_end = np.unravel_index(np.argmax(prob), prob.shape)
    return best_start, best_end

sent_len = 20
L = 5
prob_s = F.softmax(torch.randn(sent_len), dim=0)
prob_e = F.softmax(torch.randn(sent_len), dim=0)
best_start, best_end = get_best_interval(prob_s, prob_e, L)
print(best_start, best_end)