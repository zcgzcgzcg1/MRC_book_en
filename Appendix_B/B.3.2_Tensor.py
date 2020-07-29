import torch
# a real-value tensor of size 2 x 3 initialized from an array
a = torch.FloatTensor([[1.2, 3.4, 5], [3, 6, 7.4]])
print(a)
# a real-value tensor of size 5 x 6 where each element is drawn from the normal distribution N(0,1)
b = torch.randn(5, 6) 
print(b)
# change a single element in the tensor a
a[0, 2] = 4.0 
print(a)
