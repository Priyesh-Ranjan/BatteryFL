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
        self.prob = collection_prob                                            # probability of collection operation succeeding
        self.battery = battery                                                 # amount of battery client has
        self.model = model                                                     # client model
        self.dataLoader = dataLoader                                           # total data the client can ever access, client collects chunks of this every time it does a collect operation
        self.optimizer = optimizer                                             # optimizer for the nn model
        self.device = device                                                   # cuda or cpu                    
        self.init_stateChange()                                                # delta_0
        self.originalState = deepcopy(model.state_dict())                      # W_0
        self.isTrained = False                                                 # Is the model trained after the last upload or not
        self.criterion = criterion                                             # Loss function
        self.top_slice = 0                                                     # The index of the last data sample collected
        self.bottom_slice = 0                                                  # The index of the first sample collected this round, it resets to self.top_slice after uploading the model at the end
        self.dataset = []                                                      # dataset for training
        self.losses = []                                                       # list for all the losses
        self.reputation = np.zeros(len(dataLoader.dataset))                    # reputation for all the samples
        self.reputation_method = reputation_method                             # method for updating reputation loss or tracin
        self.diversity_method = "Entropy"                                      # method for checking data diversity
        self.alpha = alpha                                                     # Alpha value for the reputation update
        self.beta = beta                                                       # Beta value for the reputation update
        self.gamma = gamma                                                     # Gamma value for the reputation update
        self.mu = mu                                                           # Mu value for the reputation update
        self.batch_size = batch_size                                           # training batch size
        self.upload = upload_battery
        self.download = download_battery
        self.collection = collection_battery
        self.training = training_battery
        self.size = collection_size
        self.threshold = entropy_threshold                                     # threshold for the entropy method
        self.collection_budget = collection_budget                             # Total collection budget, exhausts as more collection happens
        self.initial_collection_budget = collection_budget                     # Collection budget in the beginning, remains same over the entire round
        self.training_budget = training_budget                                 # Total training budget, exhausts as more training happens
        self.initial_training_budget = training_budget                         # Training budget in the beginning, remains same over the entire round
        self.training_size = training_size                                     # Size of training that will happen in a round, pre-defined based on dataset
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
        self.battery -= self.download                                          # Download battery decreased upon downloading data
        
    def participation(self):                                                                              # Client provide info to the server
        if len(self.losses) : loss_val = self.losses[-1]                                                  # If previous loss exists then that is returned
        else : loss_val = 100000                                                                          # otherwise loss is assumed infinite
        return loss_val, self.battery, self.battery - (self.collection_budget + self.training_budget)     # return L_t; b_t and b_(t+1)
        
    def perform_task(self):                                                                               # Deciding what to do
        if self.battery >= self.upload:                                                                    # If you can't even upload then no point of training
            print('-----------------------------Client {}-----------------------------'.format(self.cid))
            self.training_budget = self.initial_training_budget                                           # Training budget allocated every round
            self.collection_budget = self.initial_collection_budget                                       # Collection budget allocated every round
            print("Clients training budget is", self.training_budget)
            print("Clients collection budget is", self.collection_budget)
            if self.battery >= self.initial_training_budget + self.initial_collection_budget :            # If can do both collection and training then do both
                self.battery -= (self.training_budget+self.collection_budget)                             # Battery is decreased by that amount
                self.collect_data()                                                                       # Collect data
                self.train()                                                                              # Train
                if self.isTrained == True : self.update()                                                 # If training happened then upload
                else : print("No training done, nothing to upload")
                self.battery += self.training_budget+self.collection_budget                               # Leftover battery is added back to the total battery
            elif self.battery >= self.initial_training_budget + self.upload:                              # If can only train/collect data + some leftover
                self.collection_budget = self.battery - self.training_budget - self.upload                # leftover becomes collection_budget
                self.train()                                                                              # then train
                self.update()                                                                             # update
                self.battery = 0                                                                          # no battery left anymore
            elif self.battery <= self.training_budget + self.upload and self.battery > self.upload:       # else if some leftover battery > upload battery left
                self.training_budget = self.battery - self.upload                                         # leftover is training
                self.train()                                                                              # train
                self.update()                                                                             # update
                self.battery = 0
        else :
            print("Client",self.cid,"is out of battery!")
    
    def check_convergence(self):                                                                # Returns 1 if model has converged, 0 otherwise
        if not(len(self.losses)) :
            #print("No training done yet. Model assumed non-convergent on data\n")
            return 0
        elif len(self.losses) == 1:                                                             # If there is only one round of training
            return 0
        elif np.median(self.losses) - self.losses[-1] <= 0.1*self.losses[-1] :                  # This is what I have put as placeholder here. We need to change this
            print("Loss on existing data converged.")
            return 1
        else :
            return 0
    
    def check_diversity(self, bottom_index, top_index):                                         # Need some better metric I think
        if bottom_index - top_index == 0:
            return 0                                                                            # If no data
        else:
            labels = self.dataLoader.dataset.get_labels(range(bottom_index, top_index))         # get labels from the data method
            if len(set(labels)) != self.num_classes:                                            # if you don't have all the classes yet then diversity = 0
                return 0
            elif self.diversity_method == "Entropy":
                entropy_value = Entropy(labels)                                                 # checking entropy
                if entropy_value >= self.threshold:                                             # If it is higher than threshold then diversity passed
                    print("Data quality is good.")
                    return 1
                else:
                    return 0
    
    def update_reputation(self, indices) :                                                                   # Updating reputation of the passed samples
        shuffled = np.random.permutation(indices)
        if self.reputation_method == 'loss' :
            I_vals, _ = Loss(self.model, self.dataset, self.device, self.optimizer, self.criterion)
        elif self.reputation_method == 'tracin' :
            I_vals, _ = TracIn(self.model, self.dataset, self.device, self.optimizer, self.criterion)
            
        for idx, val in enumerate(shuffled) :    
            self.reputation[val] = self.alpha*self.reputation[val] + (1 - self.alpha)*I_vals[idx]            # Updating with alpha
        print("Reputation Updated\n")    

    def select_older_data(self):                                                                             # Selecting older data
        if self.bottom_slice != self.top_slice :
            """This if else is here because in some cases, data is not collected and only training happens
            in that case there is an error here. I am just making it so that the current round of data is not
            selected as there is no data in the current round. We only select data from the previous rounds"""
            y = self.dataLoader.dataset.get_labels(list(range(self.bottom_slice,self.top_slice)))                    # Getting all the labels for this round
            counts = Counter(y)                                                                                      # label counts for each class
            num_classes = len(counts)                                                                                # num_classes in the current data
            comp = self.dataLoader.dataset.get_labels(list(range(self.bottom_slice)))                                # comp is the labels for all the previous rounds
            indices = []
            for c,num in counts.items() :                                                                            # for every label
                idx = np.asarray(comp==c).nonzero()[0]                                                               # indexes of comp where comp == the current class
                r = self.reputation[idx]                                                                             # reputation of those samples
                req = max(0, -(self.top_slice//-num_classes) - num)                                                  # required samples (looks weird because of ceiling fn)
                if len(idx) :                                                                                        # If those number of samples exist
                    samples = np.random.choice(idx, req, p = np.exp(r/self.gamma)/np.sum(np.exp(r/self.gamma)))      # then pick with the probability
                    indices.extend(samples)
        else :
            """This if else is here because in some cases, data is not collected and only training happens
            in that case there is an error here. I am just making it so that the current round of data is not
            selected as there is no data in the current round. We only select data from the previous rounds"""
            comp = self.dataLoader.dataset.get_labels(list(range(self.bottom_slice)))
            indices = []
            counts = Counter(comp)                                                                                   # here only taking past samples   
            num_classes = len(counts)
            for c,num in counts.items() :
                idx = np.asarray(comp==c).nonzero()[0]
                r = self.reputation[idx]
                req = max(0, -(self.bottom_slice//-num_classes))                                                     # here the required is just the whole thing
                if len(idx) :
                    samples = np.random.choice(idx, req, p = np.exp(r/self.gamma)/np.sum(np.exp(r/self.gamma)))      # same process
                    indices.extend(samples)           
        print("From the previous collection", len(indices), "samples are selected for training")    
        return indices
        
    def select_data(self, total_quantity):   
        if total_quantity > self.top_slice :
            """If the size of training that needs to happen is more than all the data that you have then train on whatever you have"""
            print("Do not have",total_quantity,"samples! Training on whatever is present.")
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.top_slice)))
            training_indices = np.array(list(range(self.top_slice)))
        elif total_quantity > self.top_slice - self.bottom_slice:
            """If the required quantity is more than the amount of data collected, then need to chose the older samples.
            Also runs when no data is collected so all data is from older samples"""
            old_indices = self.select_older_data()
            training_indices = np.array(old_indices+list(range(self.bottom_slice,self.top_slice)))
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.top_slice))+old_indices)
        else : 
            """If you collected more samples than were needed, then we do not need any more samples so only the current samples"""
            training_indices = np.array(list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
            self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
        print("In total, Client will train on", len(self.dataset), "samples")
        print("Updating Reputation using the",self.reputation_method,"method")
        self.update_reputation(training_indices)                                                                              # updating reputation
        return training_indices

    def train(self):
        inner_epochs = int(int(self.training_budget/self.training)/self.training_size)                                # How many rounds you can train on the given budget
        if self.check_convergence() :                                                                                 # Checking if converged
            self.collect_data('train')                                                                                # Then need more data
        print("Starting training \n")    
        print("")
        flag = 0
        for e in range(inner_epochs):                                                                                 # training for the given rounds
            print("Round",e+1,"\n")
            training_indices = self.select_data(self.training_size)                                                   # training indices obtained from the budget
            loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)
            self.model.to(self.device)
            self.model.train()
            ind = 0; loss_sum = 0
            for batch_index, (data, target) in enumerate(loader):
                if self.training_budget < (self.batch_size)*self.training or self.check_convergence():                # if budget is exhausted or model is converged
                    flag = 1                                                                                          # need to break out of everything
                    break
                else : 
                    data = data.to(self.device)                                                                       # From here......
                    target = target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += loss.sum().detach().numpy()                                                           # ......to here is just training
                    ind = batch_index
                    self.training_budget -= (self.batch_size)*self.training                                           # budget is decreased by the amount training is done
            if loss_sum : self.losses.append(loss_sum/(self.batch_size*ind))                                          # average of loss being appended
            self.isTrained = True
            self.model.cpu()  ## avoid occupying gpu when idle
            print("Client trained on",(ind)*self.batch_size,"samples.")
            print(" Used around",(ind*self.batch_size)*self.training,"battery")
            print("Average loss on the data = ", self.losses[-1],"                           (It is infinite if the data trained on is less than batch_size)\n")
            total_indices = np.array(range(self.top_slice))                                                           # all the samples that we have right now
            for i in np.delete(total_indices,np.argwhere(np.isin(total_indices, training_indices))) :                 # ones whose reputation wasn't updated yet
                if self.reputation[i] < self.mu :
                    self.reputation[i] = self.mu*self.reputation[i] + (1 - self.beta)*self.reputation[i]              # pardoning
            if flag : break                                                                                           # if budget exceeded or converged, break
        
    def collect_data(self, location = 'main'):                                                                        # collects data
        if location == 'main' : loc = 0                                                                               # if called from perform_task function check diversity of total data
        elif location == 'train' : loc = self.bottom_slice                                                            # if called from train function check diversity of data collected this round
        while not(self.check_diversity(loc, self.top_slice)) and (self.collection_budget>=self.size*self.collection): # if diversity is matched or budget will go before this collection operation
                print("Collecting data...(Check diversity function is too slow so it takes a lot of time)")
                start = self.top_slice
                for num in range(self.size):                                                                          # collecting every sample
                    if self.check_diversity(loc, self.top_slice):                                                     # checking diversity after every sample
                        break
                    elif np.random.random() <= self.prob:                                                             # if random number between (0,1) generates > collection probability
                        self.top_slice += 1                                                                           # add one sample by increasing the top slice
                        if self.top_slice >= len(self.dataLoader.dataset):
                            self.top_slice = len(self.dataLoader.dataset)                                             # reached the end of the entire data available
                            print("All the data that could have been collected is collected!")
                            break
                self.collection_budget -= self.collection*(self.top_slice - start)                                    # subtracting the battery spent from the budget 
                print("Samples collected per class:",Counter(self.dataLoader.dataset.get_labels(range(self.bottom_slice,self.top_slice))))
                print("Client collected",str(self.top_slice-start),"samples. Total samples this round = ",self.top_slice-self.bottom_slice,".Overall samples =",self.top_slice,"\n")
        if (self.collection_budget < self.size*self.collection) : print("Ran out of collection battery quota")        # if ran out of collection budget     
        
    def report_battery(self) :
        return self.battery
            
    def test(self, testDataLoader):                                            # For testing the model performance, Will NOT happen irl, just for simulation
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


    """def train_checking(self):
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
        return train_loss, accuracy, f1/c, conf.astype(int)"""


    def update(self):                                                                    # Updating the model parameters so that server can get them
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
        self.isTrained = False
        print("Client",self.cid,"sending model to the server \n")
        print("\n")
        self.bottom_slice = self.top_slice                                               # Bottom slice is updated for the next round of collection
        self.battery -= self.upload                                                      # battery reduces by upload battery amount

    #         self.test(self.dataLoader)
    def getDelta(self):
        return self.stateChange
