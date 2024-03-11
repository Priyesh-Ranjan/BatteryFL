from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import TensorDataset, DataLoader, Subset
from collections import Counter
from utils.sample_selection import Loss, TracIn
from utils.diversity import Entropy

class Client():
    def __init__(self, cid, battery, model, dataLoader, optimizer, criterion=F.nll_loss, 
                 reputation_method = 'loss', device='cpu', batch_size = 64,
                 upload_battery=3, download_battery=3, collection_battery=0.002, training_battery=0.002, collection_size=100, collection_prob = 0.95,
                 alpha=0.5, beta=0.5, gamma=0.5, mu=0.5, training_size = 200, entropy_threshold = 0.4, collection_budget = 10, training_budget = 10):
        self.cid = cid
        self.prob = collection_prob
        self.battery = battery
        self.model = model
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.device = device
        self.log_interval = len(dataLoader) - 1
        self.init_stateChange()
        self.originalState = deepcopy(model.state_dict())
        self.isTrained = False
        self.inner_epochs = 1
        self.criterion = criterion
        self.indices = [(0,0)]
        self.top_slice = 0
        self.bottom_slice = 0
        self.dataset = []
        self.losses = []
        self.reputation = np.zeros(len(dataLoader.dataset))
        self.reputation_method = reputation_method
        self.diversity_method = "Entropy"
        #self.convergence_method = "train"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.batch_size = batch_size
        self.upload = upload_battery
        self.download = download_battery
        self.collection = collection_battery
        self.training = training_battery
        self.size = collection_size
        self.threshold = entropy_threshold
        self.collection_budget = collection_budget
        self.training_budget = training_budget
        self.training_size = training_size
        self.num_classes = 10

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()
        self.battery -= self.download
        
    def participation(self):
        if len(self.losses) : loss_val = self.losses[-1]
        else : loss_val = -1
        return loss_val, self.battery, self.battery - (self.collection_budget + self.training_budget)
        
    def perform_task(self):
        if self.battery > 0:
            print('-----------------------------Client {}-----------------------------'.format(self.cid))
            self.battery -= (self.training_budget+self.collection_budget)
            #if self.check_diversity() :
            #    if self.check_convergence() :   
            #        self.collect_data()
            #        self.train()
            #    else :
            #        self.train()
            #else :
            #        self.collect_data()
            #        self.train()
            self.collect_data()
            self.train()
            if self.isTrained == True : self.update()   
            else : print("No training done, nothing to upload")
            self.battery += self.training_budget+self.collection_budget
        else :
            print("Client",self.cid,"is out of battery!")
    
    def check_convergence(self):                                                            # Returns 1 if model has converged, 0 otherwise
        if not(len(self.losses)) :
            #print("No training done yet. Model assumed non-convergent on data\n")
            return 0
        elif len(self.losses) == 1:
            return 0
        elif np.median(self.losses) - self.losses[-1] <= 0.1*self.losses[-1] :                  # This is what I have put as placeholder here. We need to change this
            print("Loss on existing data converged.")
            return 1
        else :
            return 0
    """if self.convergence_method == "loss" :
            if not(len(self.test_losses)) :
                print("No training done yet. Model assumed non-convergent on data\n")
                return 0
            else:
                #subset = Subset(self.dataLoader.dataset, range(self.top_slice))
                subset = Subset(self.dataLoader.dataset, list(indices))
                _, loss_sum = Loss(self.model, subset, self.batch_size, self.device, self.optimizer, self.criterion)
                print("Loss on all collected data", loss_sum)
                if loss_sum <= self.test_losses[-1] :
                    print("Loss on existing data converged. Need to collect more\n")
                    return 1
                else : 
                    print("Loss not converged yet so training again\n")
                    return 0"""
                    
    def check_diversity(self):                                                               # Returns 1 if data is diverse, 0 otherwise
        if self.top_slice == 0 :
            #print("No data yet")
            return 0
        else :
            labels = self.dataLoader.dataset.get_labels(range(self.bottom_slice, self.top_slice))
            if len(set(labels)) != self.num_classes:    # Checking atleast one sample should be from each class
                return 0
            elif self.diversity_method == "Entropy":
                entropy_value = Entropy(labels)
                if entropy_value >= self.threshold :
                    print("Data Quality good...\n")
                    return 1
                else : 
                    #print("Collected data has poor quality. Collecting more...\n")
                    return 0
    
    def update_reputation(self, indices) :
        shuffled = np.random.permutation(indices)
        if self.reputation_method == 'loss' :
            I_vals, _ = Loss(self.model, self.dataset, self.device, self.optimizer, self.criterion)
        elif self.reputation_method == 'tracin' :
            I_vals, _ = TracIn(self.model, self.dataset, self.device, self.optimizer, self.criterion)
            
        for idx, val in enumerate(shuffled) :    
            self.reputation[val] = self.alpha*self.reputation[val] + (1 - self.alpha)*I_vals[idx]
        print("Reputation Updated\n")    

    def select_older_data(self):
        y = self.dataLoader.dataset.get_labels(list(range(self.bottom_slice,self.top_slice)))
        counts = Counter(y)
        num_classes = len(counts)
        #old_dataset = Subset(self.dataLoader.dataset, old_samples)
        comp = self.dataLoader.dataset.get_labels(list(range(self.bottom_slice)))
        indices = []
        for c,num in counts.items() :
            idx = np.asarray(comp==c).nonzero()[0]
            #print(idx)
            #idx = old_dataset.targets == c
            r = self.reputation[idx]
            req = max(0, -(self.top_slice//-num_classes) - num)
            if len(idx) :
                samples = np.random.choice(idx, req, p = np.exp(r/self.gamma)/np.sum(np.exp(r/self.gamma)))
                indices.extend(samples)
        print("From the previous collection", len(indices), "samples are selected for training")    
        return indices
        
    def select_data(self, total_quantity):   
        #new_dataset = Subset(self.dataLoader.dataset, new_indices)
        if total_quantity > self.top_slice :
            print("Do not have",total_quantity,"samples! Training on whatever is present.")
            old_indices = []
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.top_slice)))
            training_indices = np.array(old_indices+list(range(self.bottom_slice,self.top_slice)))
            #while total_quantity > self.top_slice: self.collect_data()
        elif total_quantity > self.top_slice - self.bottom_slice:
            old_indices = self.select_older_data()
            training_indices = np.array(old_indices+list(range(self.bottom_slice,self.top_slice)))
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.top_slice))+old_indices)
        else : 
            old_indices = []
            training_indices = np.array(old_indices+list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
        print("In total, Client will train on", len(self.dataset), "samples")
        print("Updating Reputation using the",self.reputation_method,"method")
        self.update_reputation(training_indices)
        return training_indices

    def train(self):
        inner_epochs = int(int(self.training_budget/self.training)/self.training_size)
        if self.check_convergence() :
            self.collect_data()
        print("Starting training")    
        flag = 0
        for e in range(inner_epochs):
            print("Round",e+1,"\n")
            training_indices = self.select_data(self.training_size)
            loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)
            self.model.to(self.device)
            self.model.train()
            ind = 0; loss_sum = 0
            for batch_index, (data, target) in enumerate(loader):
                if self.training_budget < (self.batch_size)*self.training or self.check_convergence():
                    flag = 1
                    break
                else : 
                    data = data.to(self.device)
                    target = target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += loss.sum().detach().numpy()
                    ind = batch_index
                    self.training_budget -= (self.batch_size)*self.training
            if loss_sum : self.losses.append(loss_sum/(self.batch_size*ind))
            self.isTrained = True
            self.model.cpu()  ## avoid occupying gpu when idle
            print("Client trained on",(ind)*self.batch_size,"samples.")
            print(" Used around",(ind*self.batch_size)*self.training,"battery")
            print("Average loss on the data = ", self.losses[-1],"\n")
            total_indices = np.array(range(self.top_slice))
            for i in np.delete(total_indices,np.argwhere(np.isin(total_indices, training_indices))) :
                if self.reputation[i] < self.mu :
                    self.reputation[i] = self.mu*self.reputation[i] + (1 - self.beta)*self.reputation[i]
            if flag : break
        
        
    def collect_data(self):
        while not(self.check_diversity()) and (self.collection_budget>=self.size*self.collection):
                print("Collecting data...(Check diversity function is too slow so it takes a lot of time)")
                start = self.top_slice
                for num in range(self.size):
                    if self.check_diversity():
                        break
                    elif np.random.random() <= self.prob: 
                        self.top_slice += 1
                        if self.top_slice >= len(self.dataLoader.dataset):
                            self.top_slice = len(self.dataLoader.dataset)
                            print("All the data that could have been collected is collected!")
                            break
                self.collection_budget -= self.collection*(self.top_slice - self.bottom_slice)
                print("Samples collected per class:",Counter(self.dataLoader.dataset.get_labels(range(start,self.top_slice))))
                print("Client collected",str(self.top_slice-start),"samples. Total samples = ",self.top_slice)
        self.bottom_slice = start        
        self.indices.append((self.bottom_slice, self.top_slice))    
        
    def report_battery(self) :
        return self.battery
            
    def test(self, testDataLoader):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        c = 0
        f1 = 0
        conf = np.zeros([10,10])
        with torch.no_grad():
            for data, target in testDataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
                conf += confusion_matrix(target.cpu(),pred.cpu(), labels = [i for i in range(10)])
                f1 += f1_score(target.cpu(), pred.cpu(), average = 'weighted')*count
                c+=count


        test_loss /= len(testDataLoader.dataset)
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        # Uncomment to print the test scores of each client
        print('-----------------------------Client {}-----------------------------'.format(self.cid))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.3f}'.format(test_loss, correct, count,
                                                                                              accuracy, f1/c))
        #print("client {} ## Test F1 Score: {:.3f}".format(self.cid,f1/c))
        print("Testing Confusion Matrix:")
        print(conf.astype(int))
        print("\n")
        return test_loss, accuracy, f1/c, conf.astype(int)


    def train_checking(self):
        self.model.to(self.device)
        self.model.eval()
        train_loss = 0
        correct = 0
        count = 0
        c = 0
        f1 = 0
        conf = np.zeros([10,10])
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.dataLoader):
                if batch_index*64 <= self.indices[-1][1]:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    train_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    count += pred.shape[0]
                    conf += confusion_matrix(target.cpu(),pred.cpu(), labels = [i for i in range(10)])
                    f1 += f1_score(target.cpu(), pred.cpu(), average = 'weighted')*count
                    c+=count
                else : break    


        train_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        
        # Uncomment to print the test scores of each client
        print('-----------------------------Client {}-----------------------------'.format(self.cid))
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%, F1 Score: {:.3f})'.format(train_loss, correct, count, accuracy, f1/c))
        #print("client {} ## Train F1 Score: {:.3f}".format(self.cid,f1/c))
        print("Training Confusion Matrix:")
        print(conf.astype(int))
        print("\n")
        return train_loss, accuracy, f1/c, conf.astype(int)


    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
        self.isTrained = False
        print("Client",self.cid,"sending model to the server \n")
        print("\n")
        self.battery -= self.upload

    #         self.test(self.dataLoader)
    def getDelta(self):
        return self.stateChange
