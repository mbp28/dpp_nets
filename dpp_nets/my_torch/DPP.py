import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

import dpp_nets.dpp as dpp

class DPP(Function):
    """
    Uses Numpy Functions to sample from the DPP implicitly 
    defined through embd, returns score as a gradient in the
    backward computation (needs to be complemented by hooks
    for REINFORCE or control variate training)

    Arguments:
    Depending on whether you're training Double or Float, provide
    dtype = torch.FloatTensor
    dtype = torch.DoubleTensor
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def forward(self, embd):

    	# Perform SVD to get eigenvalue decomposition of L
        u1, s, u2 = torch.svd(embd)
        e = torch.pow(s,2).numpy()
        v = u1.numpy()

        # Sample subset from the DPP
        subset = torch.from_numpy(dpp.sample_dpp(e, v, one_hot=True))
        subset = subset.type(self.dtype)

        # Save tensors for backward (gradient computation)
        self.save_for_backward(embd, subset)

        return subset
        
    def backward(self, grad_output):
        embd, subset = self.saved_tensors
        embd, subset = embd.numpy(), subset.numpy()

        score = torch.from_numpy(dpp.score_dpp(embd, subset))
        score = score.type(self.dtype)

        return score

class DPPLayer(nn.Module):
    """
    Uses Numpy Functions to sample from the DPP implicitly 
    defined through embd, returns score as a gradient in the
    backward computation (needs to be complemented by hooks
    for REINFORCE or control variate training)

    Arguments:
    Depending on whether you're training Double or Float, provide
    dtype = torch.FloatTensor
    dtype = torch.DoubleTensor
    """
    def __init__(self, dtype):
        super(DPPLayer, self).__init__()
        self.dtype = dtype

    def forward(self, embd):
        return DPP(self.dtype)(embd)

class DPP2(Function):
    """
    Uses Numpy Functions to sample from the DPP implicitly 
    defined through embd, returns score as a gradient in the
    backward computation (needs to be complemented by hooks
    for REINFORCE or control variate training)

    Arguments:
    Depending on whether you're training Double or Float, provide
    dtype = torch.FloatTensor
    dtype = torch.DoubleTensor
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def forward(self, vals, vecs):
        """
        Given L = E * E.t() and E = u * s * v.t()
        vals are the eigenvalues of L, respectively s**2
        vecs are the eigenvectors of L, respectively u
        """

        # Transform to numpy
        e = vals.numpy()
        v = vecs.numpy()

        # Sample subset from the DPP
        subset = torch.from_numpy(dpp.sample_dpp(e, v, one_hot=True))
        subset = subset.type(self.dtype)

        # Save tensors for backward (gradient computation)
        self.save_for_backward(vals, vecs, subset)

        return subset
        
    def backward(self, grad_output):
        vals, vecs, subset = self.saved_tensors
        n_selected = int(subset.sum())

        # from full matrix
        from_full = 1 / vals

        # from subset
        P = torch.diag(subset)
        P = P[subset.expand_as(P).t().byte()].view(n_selected, -1)

        submat = P.mm(vecs).mm(vals.diag()).mm(vecs.t()).mm(P.t())
        submat_inv = submat.inverse()
        med = P.t().mm(submat_inv).mm(P).mm(vecs)

        grad_vals = self.dtype(vecs.t().mm(med).diag() + from_full)
        grad_vecs = self.dtype(2 * med.mm(vals.diag()))

        return grad_vals, grad_vecs



