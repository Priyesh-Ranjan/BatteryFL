import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

def Loss(model, dataset, indices, bsz, device, optimizer, criterion):
    model.to(device)
    model.eval()
    I_vals = []
    loader = DataLoader(dataset, shuffle=False, batch_size=bsz)        
    for batch_index, (data, target) in enumerate(loader):
            data = data.to(device)
            #target = torch.tensor(np.array([target])).to(self.device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            print(np.shape(loss.item()))
            I_vals.extend(loss.item()/np.sum(loss.item()))
    model.cpu()        
    return I_vals


def TracIn(model, dataset, indices, bsz, device, optimizer, criterion):
    print("Not done yet")
    return np.zeros(len(indices))