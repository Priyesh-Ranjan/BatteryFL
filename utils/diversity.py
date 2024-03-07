from scipy.stats import entropy
import numpy as np

def Entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    probs = counts/np.sum(counts)
    ent = entropy(probs, base = len(value))
    return ent