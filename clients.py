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
                 reputation_method = 'loss', device='cpu', batch_size = 64, momentum = 0.5,
                 collection_size=100, collection_prob = 0.95,
                 alpha=0.5, beta=0.5, gamma=0.5, mu=0.5, training_size = 200, entropy_threshold = 0.7, round_budget = 10, num_classes = 10):
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
        self.upload = 0
        self.download = 0
        self.collection = 0
        self.training = (np.random.uniform(1.5,2)**2)*np.random.uniform(5,15)/10**6
        self.threshold = entropy_threshold                                     # threshold for the entropy method
        self.round_budget = round_budget*self.battery/100                      # Round budget in the beginning, remains same over the entire round
        self.training_size = training_size                                     # Size of training that will happen in a round, pre-defined based on dataset
        self.num_classes = num_classes
        self.label_distribution = [0 for _ in range(self.num_classes)]
        self.collection_budget = 0
        self.training_budget = 0
        self.local_normalizing_vec = 0
        self.data_size = 0
        self.momentum = momentum

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states
        
    def init_collection_battery(self, scaling_factor):
        self.collection = scaling_factor*self.training
    
    def init_update_battery(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size = param_size+buffer_size    
        self.upload = size*np.float128(3.78*1e-17)
        self.download = self.upload/30

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()
        self.battery -= self.download                                          # Download battery decreased upon downloading data
        
    def participation(self):                                                                              # Client provide info to the server
        # If previous loss exists then that is returned otherwise loss is assumed infinite
        if len(self.losses) : 
            loss_val = self.losses[-1] 
        else : 
            loss_val = 1e10
        prev_expenditure =  self.collection_budget + self.training_budget 
        if prev_expenditure == 0:
            future_battery = self.battery - self.round_budget
        else:
            future_battery = self.battery - prev_expenditure - self.upload -self.download
        return loss_val, self.battery, future_battery                                   # return L_t; b_t and b_(t+1)
        
    def perform_task(self):                                                                               # Deciding what to do
        initial_battery = self.battery
        if self.battery >= self.upload:                                                                    # If you can't even upload then no point of training
            print('-----------------------------Client {}-----------------------------'.format(self.cid))
            #resetting the budget
            self.training_budget = 0                                        
            self.collection_budget = 0                                      
            if self.battery >= self.training:                                                              # If can at least train on one sample
                self.collect_data()                                                                        # Collect data
                self.train()                                                                               # Train
                self.update()                                                                              # If training happened then upload
        else :
            print("Client",self.cid,"is out of battery!")
        print("Clients training budget is", self.training_budget)
        print("Clients collection budget is", self.collection_budget)
        print("Client total energy spent is", initial_battery - self.battery)
        if self.collection_budget + self.training_budget + self.upload > self.round_budget:
            assert False, "Client {} has exceeded the round budget".format(self.cid)
    
    def check_convergence(self):                                                                # Returns 1 if model has converged, 0 otherwise
        #TODO: [AA] the class frequency should not be computed from scratch every time, it can be easily cached ad updated
        if not(len(self.losses)) :
            #print("No training done yet. Model assumed non-convergent on data\n")
            return False
        elif len(self.losses) == 1:                                                             # If there is only one round of training
            return False
        else:
            # check if the loss has converged in the last 4 rounds
            return abs(self.losses[-1] - self.losses[-2])/np.median(self.losses[-4:-1]) < 0.01        # If the loss has converged then return True
    
    def check_diversity(self, bottom_index, top_index):                                         # Need some better metric I think
        if bottom_index - top_index == 0:
            return False                                                                            # If no data
        else:
            if 0 in self.label_distribution != self.num_classes:                                            # if you don't have all the classes yet then diversity = 0
                return False
            elif self.diversity_method == "Entropy":
                entropy_value = Entropy(self.label_distribution, threshold=self.training_size/self.num_classes)  # checking entropy
                return entropy_value >= self.threshold                                             # If it is higher than threshold then diversity passed
    
    def update_reputation(self, indices) :                                                                   # Updating reputation of the passed samples
        #assert not np.isnan(self.reputation).any() , "Reputation has nan values before update"
        shuffled = np.random.permutation(indices)
        if self.reputation_method == 'loss' :
            I_vals, _ = Loss(self.model, self.dataset, self.device, self.optimizer, self.criterion)
        elif self.reputation_method == 'tracin' :
            I_vals, _ = TracIn(self.model, self.dataset, self.device, self.optimizer, self.criterion)
            
        self.reputation[shuffled] *= self.alpha 
        self.reputation[shuffled] += (1 - self.alpha)*I_vals
        print("Reputation Updated\n")    
        #assert not np.isnan(self.reputation).any() , "Reputation has nan values after update"

    def select_older_data(self):                                                                             # Selecting older data
        required_samples = max(0, self.training_size - (self.top_slice - self.bottom_slice))                 # required samples
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
                req = max(0, (required_samples//num_classes) - num)                                                  # required samples (looks weird because of ceiling fn)
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
                #no nan in r
                #assert not np.isnan(r).any() , "Reputation has nan values"
                req = max(0, (required_samples//num_classes))                                                     # here the required is just the whole thing
                if len(idx) :
                    #replace nan with 0
                    p = np.exp(r/self.gamma)/np.sum(np.exp(r/self.gamma))
                    p[np.isnan(p)] = min(p[~np.isnan(p)])
                    samples = np.random.choice(idx, req, p = p)
                    indices.extend(samples)           
        print("From the previous collection", len(indices), "samples are selected for training")    
        return indices
        
    def select_data(self, total_quantity):   
        if total_quantity > self.top_slice :
            """If the size of training that needs to happen is more than all the data that you have then train on whatever you have"""
            print("Do not have",total_quantity,"samples! Training on whatever is present.")
            #self.dataset = Subset(self.dataLoader.dataset, list(range(self.top_slice)))
            training_indices = np.array(list(range(self.top_slice)))
        elif total_quantity > self.top_slice - self.bottom_slice:
            """If the required quantity is more than the amount of data collected, then need to chose the older samples.
            Also runs when no data is collected so all data is from older samples"""
            old_indices = self.select_older_data()
            training_indices = np.array(old_indices+list(range(self.bottom_slice,self.top_slice)))
            #self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.top_slice))+old_indices)
        else : 
            """If you collected more samples than were needed, then we do not need any more samples so only the current samples"""
            training_indices = np.array(list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
            #self.dataset = Subset(self.dataLoader.dataset, list(range(self.bottom_slice,self.bottom_slice+total_quantity)))
        print("In total, Client", self.cid, "will train on", len(training_indices), "samples")
        print("Updating Reputation using the",self.reputation_method,"method")
        return training_indices

    def train(self):
        inner_epochs = int((self.round_budget-self.collection_budget - self.upload)/(self.training_size*self.training))            # How many rounds you can train on the given budget
        print("Starting training")    
        print(f"{inner_epochs} rounds of training can be done \n")
        print("")
        flag = 0
        # if no data: skip
        if self.top_slice == 0:
            print("No data to train on")
            return
        local_counter = 0
        self.local_normalizing_vec = 0
        for e in range(inner_epochs):                                                                                 # training for the given rounds
            print("Round",e+1,"\n")
            #TODO [AA] : Where are the training indices used? The dataloader does not consider them!!
            # The subset function selects those samples from the dataset and creates a new one that can be used by the dataloader
            training_indices = self.select_data(self.training_size)                                                   # training indices obtained from the budget
            self.dataset = Subset(self.dataLoader.dataset, list(training_indices))
            self.update_reputation(training_indices)                                                                              # updating reputation
            loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, drop_last=False, num_workers=4)                # dataloader for the training
            self.model.to(self.device)
            self.model.train()
            ind = 0; 
            for batch_index, (data, target) in enumerate(loader):
                # if budget exceeded
                if self.training_budget + (self.batch_size)*self.training +self.upload +self.collection_budget> self.round_budget: 
                    flag = 1                                                                                          
                    break
                if self.check_convergence():                # model is converged
                    #[AA] if the new data has been used for training at least once, then we can break out of the loop
                    #otherwise, we need to continue training on the new data
                    flag = 1                                                                                          # need to break out of everything
                data = data.to(self.device)                                                                       # From here......
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                loss_sum = loss.sum().cpu().detach().numpy()                                                           # ......to here is just training
                ind += len(data)
                self.training_budget += (self.batch_size)*self.training                                           # budget is decreased by the amount training is done
                self.losses.append(loss_sum/(self.batch_size))                                                    # average of loss being appended
                # For FedNova
                local_counter = local_counter * self.momentum + 1
                self.local_normalizing_vec += local_counter
                del data, target, output, loss
                torch.cuda.empty_cache()        
                self.bottom_slice = self.top_slice                                               # Bottom slice is updated for the next round of collection

            self.isTrained = True
            self.model.cpu()  ## avoid occupying gpu when idle
            #TODO [AA] : this sometimes reports less samples
            # I fixed it
            print("Client trained on",ind,"samples.")
            print(" Used around",self.training_budget,"battery")
            print("Average loss on the data = ", self.losses[-1],"                           (It is infinite if the data trained on is less than batch_size)\n")
            total_indices = np.array(range(self.top_slice))                                                           # all the samples that we have right now
            self.data_size = len(training_indices)

            #select all the values that are not in the training indices
            #assert not np.isnan(self.reputation).any() , "Reputation has nan values before pardoning"
            unused_samples = np.array([i for i in total_indices if i not in training_indices], dtype=np.int64)                         # ones whose reputation wasn't updated yet
            low_reputation_unused = unused_samples[np.where(self.reputation[unused_samples] < self.mu)]               
            self.reputation[low_reputation_unused] *= (1-self.beta)                                                     # pardoning
            self.reputation[low_reputation_unused] += self.mu*self.beta
            #assert not np.isnan(self.reputation).any() , "Reputation has nan values after pardoning"

            if flag : # if budget exceeded or converged, break
                print("Early stopping training")
                break
        self.dataset = None
        
    def collect_data(self):                                                                         # collects data
        loc = self.bottom_slice                             
        print("Collecting data...")
        start = self.top_slice
        mean_reputation = np.mean(self.reputation[:self.top_slice]) if self.top_slice > 0 else 0
        while True:     
            # if diversity is matched or budget will be insufficient to train
            if self.top_slice - loc >= self.training_size:
                break
            if ( not self.check_convergence() and self.check_diversity(0, self.top_slice)) and self.top_slice > self.training_size: 
                # PR: Shouldn't this be training size - loc instead of just training_size?
                    #print all the conditions
                    print(f"Collection stopped: Top slice = {self.top_slice},  collection budget = {self.collection_budget}, convergence = {self.check_convergence()}, diversity = {self.check_diversity(0, self.top_slice)}")
                    break
            if int((self.round_budget-self.collection_budget-self.collection - self.upload)/(self.training_size*self.training)) == 0:
                break
            self.collection_budget += self.collection                                                         # subtracting the battery spent from the budget +
            if np.random.random() <= self.prob:                                                               # if random number between (0,1) generates > collection probability
                if self.top_slice >= len(self.dataLoader.dataset):
                    self.top_slice = len(self.dataLoader.dataset)                                             # reached the end of the entire data available
                    print("All the data that could have been collected is collected!")
                    break

                self.top_slice += 1                                                                           # add one sample by increasing the top slice
                #increase the count of the label
                self.label_distribution[self.dataLoader.dataset.get_labels([self.top_slice-1])[0]] += 1
        #print("Samples collected per class:",Counter(self.dataLoader.dataset.get_labels(range(self.bottom_slice,self.top_slice))))

        #assert not np.isnan(self.reputation).any() , "Reputation has nan values before new element"
        self.reputation[start:self.top_slice] = mean_reputation
        #assert not np.isnan(self.reputation).any() , "Reputation has nan values after new element"
        print("Dataset distribution:",Counter(self.dataLoader.dataset.get_labels(range(0,self.top_slice))))
        print("Client collected",str(self.top_slice-start),"samples. Total samples this round = ",self.top_slice-self.bottom_slice,".Overall samples =",self.top_slice,"\n")
        #if (self.collection_budget < self.size*self.collection) : print("Ran out of collection battery quota")        # if ran out of collection budget     
        
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
        conf = np.zeros([self.num_classes,self.num_classes])
        with torch.no_grad():
            for data, target in self.dataLoader:
                target = target.to(self.device)
                output = self.model(data.to(self.device))
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
                conf += confusion_matrix(target.cpu(),pred.cpu(), labels = [i for i in range(self.num_classes)])
                f1 += f1_score(target.cpu(), pred.cpu(), average = 'weighted')*count
                c+=count
                del data, target, output
                torch.cuda.empty_cache()        

        test_loss /= len(testDataLoader.dataset)
        accuracy = 100. * correct / count
        #self.model.cpu()  ## avoid occupying gpu when idle
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
        #assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
        self.isTrained = False
        print("Client",self.cid,"sending model to the server \n")
        print("\n")
        self.battery -= (self.upload + self.training_budget + self.collection_budget)      # battery reduces by upload battery amount

    #         self.test(self.dataLoader)
    def getDelta(self):
        return self.stateChange
