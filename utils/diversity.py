from scipy.stats import entropy
import numpy as np

def Entropy(labels, threshold=np.inf):
    value, counts = np.unique(labels, return_counts=True)
    # saturate the counts to the threshold
    counts = np.minimum(counts, threshold)
    probs = counts/np.sum(counts)
    ent = entropy(probs)
    return ent