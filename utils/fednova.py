import torch
import torch.nn as nn
import numpy as np

def fednova(input, data_ratio, norm_vector):

    x = input.squeeze(0)
    w = data_ratio*np.sum(norm_vector*data_ratio)/norm_vector
    w = w/ w.sum()
    print("Weights while aggregation are:",w)
    out = torch.sum(x* w, dim=1, keepdim=True)
    return out


class Net(nn.Module):
    def __init__(self, data_ratio, normalized_vector_vals):
        super(Net, self).__init__()
        
        self.ratio = data_ratio
        self.norm_vectors = normalized_vector_vals

    def forward(self, input):
        out = fednova(input, self.ratio, self.norm_vectors)
        return out