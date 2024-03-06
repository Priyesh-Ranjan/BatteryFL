from scipy.stats import entropy
import numpy as np

def Entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    probs = counts/np.sum(counts)
    print(value)
    print(counts)
    ent = entropy(probs, base = len(value))
    print(ent)
    return ent