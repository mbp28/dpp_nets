import torch

def compute_alpha(reinforce_grads, logprob_grads, mean=False, noflip=False, clip=False): 
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
    alpha = (cov_reinforce_score / score_var).squeeze(0)
    alpha[alpha != alpha] = 0
            
    if mean:
        alpha = torch.DoubleTensor([alpha.mean()]).expand_as(alpha)
    
    if noflip:
        alpha.clamp_(max=min_loss)
        
    if clip: 
        alpha.clamp_(min=-min_loss)
    
    return alpha
