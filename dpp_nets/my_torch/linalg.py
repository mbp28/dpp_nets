import torch
from torch.autograd import Function
import numpy as np

from torch.autograd import Function

class custom_decomp(Function):
    """
    This is a "strange" but useful decomposition. 
    It takes in a matrix A of size m x n. 
    It will return the eigenvectors (vecs) 
    and eigenvalues (vals) of the matrix 
    L = A.mm(A.t()), but without actually
    computing L. 
    
    Any square or rectangular shapes are allowed
    for A. :-)
    """
    def forward(self, mat):
        vecs, vals, _ = torch.svd(mat)
        vals.pow_(2)
        self.save_for_backward(mat, vals, vecs)
        
        return vals, vecs
    
    def backward(self, grad_vals, grad_vecs):
        # unpack
        mat, vals, vecs = self.saved_tensors 
        N = vals.size(0)
        
        # auxillary
        I = grad_vecs.new(N,N).copy_(torch.eye(N))
        F = vals.expand(N,N) - vals.expand(N,N).t()
        F.add_(I).reciprocal_().sub_(I)

        # gradient
        grad_mat = grad_vals.diag() + F * (vecs.t().mm(grad_vecs)) # intermediate variable 
        grad_mat = vecs.mm(grad_mat.mm(vecs.t())) # this is the gradient wrt L
        grad_mat = grad_mat.mm(mat) + grad_mat.t().mm(mat) # this is the grad wrt A
        
        return grad_mat

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


class custom_svd(Function):
    
    def forward(self, matrix):
        assert matrix.size(0) <= matrix.size(1)  # m <= n
        # note that matrix = u * x * v.t()
        u, s, v = torch.svd(matrix)
        self.save_for_backward(matrix, u, s, v)
        return u, s, v

    def backward(self, grad_u, grad_s, grad_v):
        # gradients with respect v are currently ignored, implement later. 
        matrix, u, s, v = self.saved_tensors
        from_s = u.mm(grad_s.diag()).mm(v.t())

        # for singular vectors - this is for u 
        s_2 = (s ** 2).expand(u.size())
        F = 1 / (s_2 - s_2.t())
        F[F.abs() == np.inf] = 0
        med = (u.t() * F).mm(grad_u) + grad_u.mm(u * F.t())
        from_u = u.mm(med).mm(torch.diag(s)).mm(v.t())

        full_grad = from_u + from_s
        return full_grad

class custom_eig(Function):
    
    def forward(self, matrix):
        assert matrix.size(0) == matrix.size(1)
        e, v = torch.eig(matrix, eigenvectors=True)
        e = e[:,0]
        self.save_for_backward(e, v)
        return e, v

    def backward(self, grad_e, grad_v):
        e, v = self.saved_tensors
        dim = v.size(0)
        E = e.expand(dim, dim) - e.expand(dim, dim).t()
        I = E.new(dim, dim).copy_(torch.eye(dim))
        F = (1 / (E + I)) - I 
        M = grad_e.diag() + F * (v.t().mm(grad_v))
        grad_matrix = v.mm(M).mm(v.t())
        return grad_matrix
