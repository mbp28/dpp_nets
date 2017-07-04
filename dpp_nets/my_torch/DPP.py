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