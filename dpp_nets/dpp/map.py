from itertools import chain, combinations
import numpy as np

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