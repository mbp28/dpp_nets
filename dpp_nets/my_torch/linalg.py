import torch
from torch.autograd import Function

class my_svd(Function):
    
    def forward(self, matrix):
        u, s, v = torch.svd(matrix)
        self.save_for_backward(matrix, u, s, v)
        return u, s, v
        
    def backward(self, grad_u, grad_s, grad_v):
        # gradients with respect to u and v are currently ignored, implement later. 
        matrix, u, s, v = self.saved_tensors
        from_s = u.mm(grad_s.diag()).mm(v.t())
        return from_s