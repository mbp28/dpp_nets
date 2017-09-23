import torch
from torch.autograd import Variable
from torch.autograd import Function


def omit_slice(tensor, dim, omit):
        
    # Make dimension to index the 0th dimension
    tensor.transpose_(dim, 0)
        
    # Gather sizes
    l = tensor.size(0)
    sizes = list(tensor.size())[1:]
    sizes.reverse()

    # Create index        
    index = tensor.new().resize_(l-1).copy_(torch.LongTensor([*range(omit), *range(omit + 1, l)])).long()
    # index = torch.LongTensor([*range(omit), *range(omit + 1, l)])
    index = index.expand(*sizes, index.size(0)).permute(*range(len(sizes),-1,-1))
    
    # Apply filter
    sliced = tensor.gather(0,index)
    sliced.transpose_(dim, 0)
    
    return sliced

def orthogonalize(vecs):
    """
    THIS IS SUPER_SLOW, NEED TO UPDATE THIS!! 
    MAKE IT QUICKER!!

    """
    for index, col in enumerate(vecs.t()):
        for b in range(index):
            vecs[:,index] = col - col.dot(vecs[:,b]) * vecs[:,b]
        vecs[:,index] = vecs[:,index] / torch.norm(vecs[:,index])
    return vecs

def pad_with_zeros(tensor, dim, length):

    if tensor.size(dim) == length:
        return tensor
    
    else:
        n_pads = length - tensor.size(dim)
        zeros_size = list(tensor.size()) 
        zeros_size[dim] = n_pads
        zeros = Variable(torch.zeros(*zeros_size)).type(tensor.data.type())
        cat = torch.cat([tensor, zeros], dim=dim)
        return cat

def pad_tensor(tensor, dim, fill, length):

    if tensor.size(dim) == length:
        return tensor
    
    else:
        n_pads = length - tensor.size(dim)
        fill_size = list(tensor.size()) 
        fill_size[dim] = n_pads
        fill = fill * torch.ones(*fill_size).type(tensor.type())
        cat = torch.cat([tensor, fill], dim=dim)
        return cat


def compute_baseline(losses):
	"""
	Computes the individual baselines from a list of losses
	according to adapted VIMCO.
	Arguments:
	- losses: list
	Output:
	- baselines: list
	"""
	n = len(losses)
	loss_sum = sum(losses)

	baseline = [(n/(n-1)) * loss - (loss_sum / (n-1)) for loss in losses]

	return baseline

def save_grads(model_dict, t): 
    """
    A closure to save the gradient wrt to input of a nn module.
    Arguments:
    - model_dict: defaultdict(list)
    - t: dictionary key (usually training iteration)
    """
    def hook(module, grad_input, grad_output):
        model_dict[t].append(grad_input[0].data)

    return hook

def reinforce_grad(loss):
    """
    A closure to modify the gradient of a nn module. 
    Use to implement REINFORCE gradient. Gradients will
    be multiplied by loss.
    Arguments: 
    - loss: Gradients are multiplied by loss, should be a scalar
    """
    def hook(module, grad_input, grad_output):
        new_grad = grad_input * loss
        return new_grad
        
    return hook

import torch
import numpy as np

def compute_alpha(reinforce_grads, logprob_grads, el_mean=False): 
    """
    reinforce_grads is expected to be a list of Tensors that hold the REINFORCE gradients
    logprob_grads is expected to be a list of Tensors that hold the SCORES
        
    """
    assert len(reinforce_grads) == len(logprob_grads)
    
    scores = torch.stack(logprob_grads)
    reinforce_grads = torch.stack(reinforce_grads)
    min_loss = torch.min(reinforce_grads / scores)
    scores = scores - torch.mean(scores, dim=0).expand_as(scores)
    reinforce_grads = reinforce_grads - torch.mean(reinforce_grads, dim=0).expand_as(reinforce_grads)


    assert reinforce_grads.size() == scores.size()
    
    score_var = torch.var(scores, dim=0)
    cov_reinforce_score = torch.mean(reinforce_grads * scores, dim=0)

    cov_reinforce_score[cov_reinforce_score == np.inf] = np.nan
    score_var[score_var == 0] = np.nan

    alpha = (cov_reinforce_score / score_var).squeeze(0)
    alpha[alpha != alpha] = 0
    alpha = torch.DoubleTensor(alpha)
            
    if el_mean:
        alpha = torch.DoubleTensor([alpha.mean(dim=1)]).expand_as(alpha)
    
    return alpha
