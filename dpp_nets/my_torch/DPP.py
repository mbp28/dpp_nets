import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import StochasticFunction
from torch.autograd import Variable

import dpp_nets.dpp as dpp
from dpp_nets.my_torch.utilities import omit_slice
from dpp_nets.my_torch.utilities import orthogonalize

class DPP_Numpy(Function):
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

class DPPLayer_Numpy(nn.Module):
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
        super(DPPLayer_Numpy, self).__init__()
        self.dtype = dtype

    def forward(self, embd):
        return DPP_Numpy(self.dtype)(embd)


class DPP(StochasticFunction):
    
    def forward(self, vals, vecs):

        # Sometimes orthogonalization fails (i.e. deletes vectors)
        # In that case just retry!
        self.dtype = vals.type()

        while True:
            try:
                # Set-up
                n = vecs.size(0)
                n_vals = vals.size(0)

                # Sample a set size
                index = (vals / (vals + 1)).bernoulli().byte()
                k = torch.sum(index)

                # Check for empty set
                if not k:
                    subset = vals.new().resize_(n).copy_(torch.zeros(n))
                    self.save_for_backward(vals, vecs, subset) 
                    return subset
                
                # Check for full set
                if k == n:
                    subset =  vals.new().resize_(n).copy_(torch.ones(n))
                    self.save_for_backward(vals, vecs, subset) 
                    return subset

                # Sample a subset
                V = vecs[index.expand_as(vecs)].view(n, -1)
                subset = vals.new().resize_(n).copy_(torch.zeros(n))
                
                while subset.sum() < k:

                    # Sample an item
                    probs = V.pow(2).sum(1).t()
                    item = probs.multinomial(1)[0,0]
                    subset[item] = 1
                    
                    # CHeck if we got k items now
                    if subset.sum() == k:
                        break

                    # Choose eigenvector to eliminate
                    j = V[item, ].abs().sign().unsqueeze(1).t().multinomial(1)[0,0]
                    Vj = V[:, j]
                    
                    # Update vector basis
                    V = omit_slice(V,1,j)
                    V.sub_(Vj.ger(V[item, :] / Vj[item]))

                    # Orthogonalize vector basis
                    V, _ = torch.qr(V)
                    
                self.save_for_backward(vals, vecs, subset) 

                return subset
            except RuntimeError:
                print("RuntimeError")
                continue
                
            break
        
    def backward(self, reward):
        #TODO: Need to check this!
        # Checked it! Looks good.

        # Set-up
        if False:
            vals, vecs, subset = self.saved_tensors#
            dtype = self.dtype
            n = vecs.size(0)
            n_vals = vals.size(0)
            subset_sum = subset.long().sum()

            # auxillary
            matrix = vecs.mm(vals.diag()).mm(vecs.t())
            P = torch.eye(n).masked_select(subset.expand(n,n).t().byte()).view(subset_sum, -1).type(dtype)
            submatrix = P.mm(matrix).mm(P.t())
            subinv = torch.inverse(submatrix)
            Pvecs = P.mm(vecs)
        
            # gradiens
            grad_vals = 1 / vals
            grad_vals += Pvecs.t().mm(subinv).mm(Pvecs).diag()
            grad_vecs = P.t().mm(subinv).mm(Pvecs).mm(vals.diag())

            grad_vals.mul_(reward)
            grad_vecs.mul_(reward)

            return grad_vals, grad_vecs

        vals, vecs, subset = self.saved_tensors#
        dtype = self.dtype
        n = vecs.size(0)
        n_vals = vals.size(0)
        subset_sum = subset.long().sum()

        grad_vals = 1 / vals
        grad_vecs = torch.zeros(n, n_vals).type(dtype)

        if subset_sum:
            # auxillary
            matrix = vecs.mm(vals.diag()).mm(vecs.t())
            P = torch.eye(n).masked_select(subset.expand(n,n).t().byte()).view(subset_sum, -1).type(dtype)
            submatrix = P.mm(matrix).mm(P.t())
            subinv = torch.inverse(submatrix)
            Pvecs = P.mm(vecs)

            grad_vals += Pvecs.t().mm(subinv).mm(Pvecs).diag()
            grad_vecs += P.t().mm(subinv).mm(Pvecs).mm(vals.diag())    

        grad_vals.mul_(reward)
        grad_vecs.mul_(reward)

        return grad_vals, grad_vecs

