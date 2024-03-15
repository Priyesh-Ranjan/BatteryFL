import torch
import torch.nn as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

def Loss(model, dataset, device, optimizer, criterion):
    model.to(device)
    model.eval()
    loss_all = []
    criterion = F.CrossEntropyLoss(reduction='none')
    loader = DataLoader(dataset, shuffle=False, batch_size=dataset.__len__())
    for batch_index, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
    
    # Forward pass to get model output
        output = model(data)
    
    # Calculate loss for each sample in the batch
        loss_each = criterion(output, target)
    
    # Detach losses and move to CPU
        loss_each_np = loss_each.detach().cpu().numpy()
        loss_all.extend(loss_each_np)

    # Calculate influence values outside the loop for efficiency
    total_loss = np.sum(loss_all)
    if total_loss == 0:
        I_vals = np.full(len(loss_all), 1/len(loss_all))
    else:
        I_vals = np.array([loss / total_loss for loss in loss_all])
    #assert not np.isnan(np.sum(I_vals)).any() , "Influence values contain NaNs"

    # Optionally, move model back to CPU if not using it on GPU afterwards
    # model.cpu()
    return I_vals, np.mean(np.array(loss_all))


def TracIn(model, dataset, device, optimizer, criterion):
    model.to(device)
    model.eval()
    tracin = []
    loss_all = []
    criterion = F.CrossEntropyLoss(reduction='none')
    loader = DataLoader(dataset, shuffle=False, batch_size=1)        
    for batch_index, (data, target) in enumerate(loader):
            data = data.to(device); target = target.to(device)
            data.requires_grad_(True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            data_grad = data.grad.data
            tracin_score = torch.dot(data_grad.view(-1), data_grad.view(-1))
            tracin.extend(tracin_score)
    total_trac = np.sum(tracin)
    #model.cpu()      
    if total_trac == 0:
        I_vals = np.full(len(tracin), 1/len(tracin))
    else:
        I_vals = np.array([t / total_trac for t in tracin])
    #assert not np.isnan(np.sum(I_vals)).any() , "Influence values contain NaNs"
    return I_vals, np.mean(np.array(loss_all))