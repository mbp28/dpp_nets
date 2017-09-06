import numpy as np
from scipy.linalg import orth
from itertools import chain, combinations

def sample_dpp(vals, vecs, k=0, one_hot=False):
    """
    This function expects 
    
    Arguments: 
    vals: NumPy 1D Array of Eigenvalues of Kernel Matrix
    vecs: Numpy 2D Array of Eigenvectors of Kernel Matrix

    """
    # Extract real part
    vals = np.real(vals)
    vecs = np.real(vecs)

    n = vecs.shape[0] # number of items in ground set
    n_vals = vals.shape[0]
    
    # k-DPP
    if k:
        index = sample_k(vals, k) # sample_k, need to return index

    # Sample set size
    else:
        index = (np.random.rand(n_vals) < (vals / (vals + 1)))
        k = np.sum(index)
    
    # Check for empty set
    if not k:
        return np.zeros(n) if one_hot else np.empty(0)
    
    # Check for full set
    if k == n:
        return np.ones(n) if one_hot else np.arange(k, dtype=float) 
    
    # Choose eigenvectors
    V = vecs[:, index]

    # Sample a set of k items 
    items = list()

    for i in range(k):
        p = np.sum(V**2, axis=1)
        p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
        item = (np.random.rand() <= p).argmax()
        items.append(item)
        
        # Delete one eigenvector not orthogonal to e_item and find new basis
        j = (np.abs(V[item, :]) > 0).argmax() 
        Vj = V[:, j]
        V = orth(V - (np.outer(Vj,(V[item, :] / Vj[item])))) 
    
    items.sort()
    sample = np.array(items, dtype=float)    

    if one_hot:
        sample = np.zeros(n)
        sample[items] = np.ones(k)
    
    return sample 

def computeMAP(L):

    # initialization
    n = L.shape[0]
    no_choice = list(range(n))
    choice = []
    best_p = 0

    while True:

        candidates = [choice + [j] for j in no_choice]
        submats = [L[np.ix_(cand, cand)] for cand in candidates]
        probs = [np.linalg.det(submat) - best_p for submat in submats]

        if all(p <= 0 for p in probs):
            return choice
        else:
            which = np.argmax(np.array(probs))
            choice = candidates[which]
            which_elem = choice[-1]
            no_choice.remove(which_elem)
            best_p += probs[which]


def exactMAP(L):

    n = L.shape[0]
    
    # Generate powerset
    s = list(range(n))
    powerset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    
    # Compute Probabilities 
    probs = np.array([np.linalg.det(L[np.ix_(choice, choice)]) for choice in powerset])
    which = np.argmax(probs)
    MAP = powerset[which], probs[which]
    
    return MAP
    