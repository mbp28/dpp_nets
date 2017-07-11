import torch
from torch.autograd import Function
import numpy as np

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

class my_full_svd(Function):
    
    def forward(self, matrix):
        assert matrix.size(0) <= matrix.size(1) # m <= n
    	# note that matrix = u * x * v.t() 
        u, s, v = torch.svd(matrix)
        self.save_for_backward(matrix, u, s, v)
        return u, s, v
        
    def backward(self, grad_u, grad_s, grad_v):
        # gradients with respect v are currently ignored, implement later. 
        matrix, u, s, v = self.saved_tensors
        from_s = u.mm(grad_s.diag()).mm(v.t())

        # for singular vectors - this is for u 
        s_2 = (s**2).expand(u.size())
        F = 1 / (s_2 - s_2.t())
        F[F.abs() == np.inf] = 0
        med = (u.t() * F).mm(grad_u) + grad_u.mm(u * F.t())
        from_u = u.mm(med).mm(torch.diag(s)).mm(v.t())

        full_grad = from_u + from_s
        return full_grad

class toy_svd(Function):
    
    def forward(self, matrix):
        # note that matrix = u * x * v.t() 
        u, s, v = torch.svd(matrix)
        self.save_for_backward(matrix, s)
        return u, s
        
    def backward(self, grad_u, grad_s):
        # gradients with respect v are currently ignored, implement later. 
        matrix, s = self.saved_tensors
        u, s, v = torch.svd(matrix)

        from_s = u.mm(grad_s.diag()).mm(v.t())

        # for singular vectors - this is for u 
        s_2 = (s**2).expand(u.size())
        F = 1 / (s_2 - s_2.t())
        F[F.abs() == np.inf] = 0
        med = (u.t() * F).mm(grad_u) + grad_u.mm(u * F.t())
        from_u = u.mm(med).mm(torch.diag(s)).mm(v.t())

        full_grad = from_u + from_s

        return full_grad

