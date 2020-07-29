import torch
a = torch.ones(1)        # 1D tensor
a = a.cuda()             # a is place into the GPU
a.requires_grad          # False
a.requires_grad = True   # the gradient of a needs to be computed
b = torch.ones(1)
x = 3 * a + b            # x is the result
x.requires_grad          # True，as the gradient of a needs to be computed
x.backward()             # compute the derivatives of all parameters
a.grad                   # tensor([ 3.])，i.e. the derivative is 3 
