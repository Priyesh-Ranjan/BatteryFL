from scipy.stats import entropy
import numpy as np

def Entropy(labels, threshold=np.inf):
    # saturate the counts to the threshold
    counts = np.minimum(labels, threshold)
    probs = counts/np.sum(counts)
    ent = entropy(probs)
    return ent