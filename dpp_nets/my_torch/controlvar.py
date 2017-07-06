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


    assert reinforce_grads.shape == scores.shape
    
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
