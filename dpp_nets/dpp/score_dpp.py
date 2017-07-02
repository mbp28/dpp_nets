import numpy as np

def score_dpp(embd, subset):
    """
    Given am embd, such that L = embd * embd.T and a subset sampled by the DPP, returns 
    the gradient of the log-probability (score) with respect to each element in embd.

    Arguments:
    embd: An embedding for the L-ensemble of the DPP, such that L = embd * embd.T
    subset: The subset sampled by the DPP.
    
    Outputs:
    The gradient of the log-probability (score) with respect to each element in embd
    
    """

    # Gradient from sampled submatrix 
    subset = subset.flatten().astype(bool)
    subembd = embd[subset]
    submat = subembd.dot(subembd.T)
    submatinv = np.linalg.inv(submat)
    subgrad = np.zeros(embd.shape)
    subgrad[subset] = 2 * submatinv.dot(subembd) 
    
    # Gradient from whole L matrix
    K = embd.T.dot(embd)
    I_k = np.eye(embd.shape[1])
    I = np.eye(embd.shape[0])
    inv = np.linalg.inv(I_k + K)
    B = I - embd.dot(inv).dot(embd.T)
    grad = 2 * B.dot(embd) 

    # Compbine both gradients
    full_gradient = subgrad - grad
    
    return full_gradient