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

class Client():
    def __init__(self, cid, battery, model, dataLoader, optimizer, criterion=F.nll_loss, 
                 method = 'loss', device='cpu', inner_epochs=1, batch_size = 64,
                 upload_battery=3, download_battery=3, collection_battery=2, training_battery=0.002, collection_size=1000, prob = 0.95):
        self.cid = cid
        self.prob = prob
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
        #self.losses = []
        self.reputation = np.zeros(len(dataLoader.dataset))
        self.method = method
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 0.5
        self.batch_size = batch_size
        self.upload = upload_battery
        self.download = download_battery
        self.collection = collection_battery
        self.training = training_battery
        self.size = collection_size

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
        
    #def perform_task(self):
    #    if not(self.check_diversity) or budget:
    #        self.collect_data()
    #    else : 
    #        self.select_data(quantity)
    #        self.train(quantity)
                    
    def check_diversity(self):
        if self.dataset == [] :
            return 0
        else : return 1
    
    def update_reputation(self, indices) :
        shuffled = np.random.shuffle(indices)
        if self.method == 'loss' :
            I_vals = Loss(self.model, self.dataset, shuffled, self.batch_size, self.device, self.optimizer, self.criterion)
        elif self.method == 'tracin' :
            I_vals = TracIn(self.model, self.dataset, shuffled, self.batch_size)
            
        for idx, val in enumerate(shuffled) :    
            self.reputation[val] = self.alpha*self.reputation[val] + (1 - self.alpha)*I_vals[idx]
        print("Reputation Updated")    

    def select_older_data(self, new_indices, old_quantity):
        y = self.dataLoader.dataset.get_labels(new_indices)
        counts = dict(Counter(y))  
        num_classes = len(counts)
        old_indices = list(range(self.bottom_slice))
        #old_dataset = Subset(self.dataLoader.dataset, old_samples)
        comp = self.dataLoader.dataset.get_labels(old_indices)
        indices = []
        for c,num in counts.items() :
            idx = np.asarray(comp==c).nonzero()
            #idx = old_dataset.targets == c
            r = self.reputation[idx]
            req = max(0, old_quantity/num_classes - num)
            samples = np.random.choice(idx, req, p = np.exp(r*self.gamma)/np.sum(np.exp(r*self.gamma)))
            indices.extend(samples)
        print("From the previous collection ", len(indices), " samples are selected for training")    
        return indices
        
    def select_data(self, total_quantity):   
        #new_dataset = Subset(self.dataLoader.dataset, new_indices)
        if total_quantity > self.top_slice :
            print("Do not have that many samples! Collecting more data")
            while total_quantity > self.top_slice: self.collect_data()
        if total_quantity > self.top_slice - self.bottom_slice:
            old_indices = self.select_older_data(list(range(self.bottom_slice,self.top_slice)), 
                                                 total_quantity - len(list(range(self.bottom_slice,self.top_slice))))
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.top_slice))+old_indices)
        else : 
            old_indices = []
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
        print("In total, client",self.cid,"will train on", len(self.dataset), "samples")
        print("Updating Reputation using the",self.method,"method")
        self.update_reputation(np.array(old_indices+list(range(self.bottom_slice,self.top_slice))))

    def train(self):
        total_quantity = 1500
        self.select_data(total_quantity)
        self.model.to(self.device)
        self.model.train()
        print("Starting training")
        #for e in range(self.inner_epochs):
        ind = 0; loss_sum = 0
        loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)        
        for batch_index, (data, target) in enumerate(loader):
            if (batch_index)*self.batch_size < total_quantity :
                data = data.to(self.device)
                #target = torch.tensor(np.array([target])).to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.sum()
                ind = batch_index
            else : break
        self.loss = loss_sum/len(DataLoader.dataset)
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle
        print("Client trained on",ind*self.batch_size,"samples. Used around",(ind*self.batch_size)*self.training,"battery")
        print("Average loss on the data = ", self.loss)
        self.battery -= (ind*self.batch_size)*self.training
        
        
    def collect_data(self):
        start = self.top_slice; end = self.top_slice
        for num in range(self.size):
            if np.random.random() <= self.prob: end += 1
        self.indices.append((start, end))    
        self.top_slice = end; self.bottom_slice = start
        self.battery -= self.collection
        print("Client collected",str(end-start),"samples. Total samples = ",self.top_slice)
        
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

    #         self.test(self.dataLoader)
    def getDelta(self):
        print("Client",self.cid,"sending model to the server")
        self.battery -= self.upload
        return self.stateChange
