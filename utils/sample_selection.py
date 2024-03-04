import torch
import torch.nn as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

def Loss(model, dataset, indices, bsz, device, optimizer, criterion):
    model.to(device)
    model.eval()
    I_vals = []
    criterion = F.CrossEntropyLoss(reduction='none')
    loader = DataLoader(dataset, shuffle=False, batch_size=bsz)        
    for batch_index, (data, target) in enumerate(loader):
            data = data.to(device)
            #target = torch.tensor(np.array([target])).to(self.device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_each = criterion(output, target)
            loss_all =  torch.mean(loss_each)
            loss_all.backward()
            #print(np.shape(loss_each.detach()))
            I_vals.extend(loss_each.detach().cpu().numpy()/np.sum(loss_each.detach().cpu().numpy()))
    model.cpu()        
    return I_vals


def TracIn(model, dataset, indices, bsz, device, optimizer, criterion):
    print("Not done yet")
    return np.zeros(len(indices))