'''
Suppose the text A has been tokenized, and special symbols <s> and </s> are added. Each sentence is represented by a list of words. Vocab is the list of all words, and K is the parameter for Laplace Smoothing. 
Example:
  Input:
  A=[['<s>', 'today', 'is', 'monday', '</s>'], 
     ['<s>', 'we', 'go', 'fishing', 'today', '</s>'],
     ['<s>', 'we', 'have', 'meetings', 'today', '</s>']]
  vocab=['<s>', '</s>', 'today', 'we', 'is', 'go', 'have', 'fishing', 'meetings', 'monday']
  Call bigram(A, vocab, 1)
  Output:
  P(<s>|<s>)=0.077
  P(</s>|<s>)=0.077
  P(today|<s>)=0.154
'''

def bigram(A, vocab, K):
    cnt = {word: 0 for word in vocab} 
    cnt2 = {word: {word2: 0 for word2 in vocab} for word in vocab} 
    # cnt[word] is the number of occurrences of word in the text, and cnt2[word][word2] is the number of occurrences of word word2 in the text.
    for sent in A:
        for i, word in enumerate(sent):
            cnt[word] += 1
            if i + 1 < len(sent):                
                cnt2[word][sent[i + 1]] += 1
    for word in cnt2:
        for word2 in cnt2[word]: 
            # Laplace Smoothing
            prob = (cnt2[word][word2]+K) / (cnt[word] +K * len(vocab) + 0.0) 
            print((cnt2[word][word2]+K), cnt[word] +K * len(vocab))
            print('P({0}|{1})={2}'.format(word2, word, prob))

A=[['<s>', 'today', 'is', 'monday', '</s>'], 
   ['<s>', 'we', 'go', 'fishing', 'today', '</s>'],
   ['<s>', 'we', 'have', 'meetings', 'today', '</s>']]
vocab=['<s>', '</s>', 'today', 'we', 'is', 'go', 'have', 'fishing', 'meetings', 'monday']

bigram(A, vocab, 1)
