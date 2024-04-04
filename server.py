from __future__ import print_function

from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import numpy as np
import utils.client_selection as client_selection

from utils import utils
import time

class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu', client_selection='ours', num_classes = 10):
        self.clients = []                                                      # list of clients
        self.model = model
        self.dataLoader = dataLoader                                           # for server testing
        self.device = device                                                   # for server testing
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0                                                          # number of epochs
        self.AR = self.FedAvg                                                  # Aggregation algorithm
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.client_selection = client_selection
        self.selected_clients = []
        self.num_classes = num_classes
        self.mu = 0
        self.var = 0

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)
        
    def get_selected_clients(self):
        clients = [client.cid for client in self.selected_clients]
        return clients

    def test(self):
        print("-----------------------------Server-----------------------------")
        print("[Server] Start testing \n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        c = 0
        f1 = 0
        conf = np.zeros([self.num_classes,self.num_classes])
        with torch.no_grad():
            data, target = [], []
            for batch_data, batch_target in self.dataLoader:
                data.append(batch_data)
                target.append(batch_target)
            data = torch.cat(data, dim=0).to(self.device)
            target = torch.cat(target, dim=0).to(self.device)
            output = self.model(data)
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
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.4f}\n'.format(test_loss, correct, count, accuracy, f1/c))
        #print("F1 Score: {:.4f} \n".format(f1/c))    
        print("Confusion Matrix :")
        print(conf.astype(int),"\n") 
        return test_loss, accuracy, f1/c, conf.astype(int)

    def do(self):                                                              # One round of communication, client training is asynchronous
        if self.client_selection == 'ours' :
            self.selected_clients = client_selection.Our_Algorithm(self.clients)
        elif self.client_selection == 'AGE' :
            self.selected_clients = client_selection.genetic(self.clients, algorithm="age2")
        elif self.client_selection == 'NSGA' :    
            self.selected_clients = client_selection.genetic(self.clients, algorithm="nsga2")
        elif self.client_selection == "EAFL" :
            self.selected_clients = client_selection.eafl(self.clients)
        else :
            print("Selection Algorithm not recognized. Choosing all the clients")
            self.selected_clients = self.clients
        self.selected_clients = [c for c in self.selected_clients if c.battery > (c.upload + c.download)]           # Selecting clients with battery
        print("Clients selected this round are:",[c.cid for c in self.selected_clients])
        if self.selected_clients == []:
            return False
        
        mu_1, var_1 = self.get_batteries()
        
        for c in self.selected_clients:
            c.setModelParameter(self.model.state_dict())                       # distribute server model to clients
            c.perform_task()                                                   # selected clients will perform the FSM

        if self.isSaveChanges:
            self.saveChanges(self.selected_clients)
        print("----------------------------------------------------------------")
        print("\n")
        tic = time.perf_counter()
        Delta = self.AR(self.selected_clients)                                      # Getting model weights from the clients
        toc = time.perf_counter()
        print("\n")
        print(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")
        
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1
        
        mu_2, var_2 = self.get_batteries()
        
        self.mu_function(mu_1,mu_2,var_1,var_2)
        self.var_function(mu_1,mu_2,var_1,var_2)
        
        return True

    def get_batteries(self) :
        batteries = []
        for c in self.selected_clients:
            batteries.append(c.report_battery())
        batteries = np.array(batteries)
        return np.mean(batteries), np.var(batteries)
    
    def mu_function(self, mu_1, mu_2, var_1, var_2) :
        self.mu = (len(self.selected_clients)*(mu_1+mu_2)**2 + 2*(var_1+var_2))/(2*(mu_1**2+mu_2**2)+4)
        
    def var_function(self, mu_1, mu_2, var_1, var_2) :
        var_sum = var_1 + var_2
        mu_sum = mu_1 + mu_2
        mu_sum_sq = mu_1**2 + mu_2**2
        
        part_1 = (2/(len(self.selected_clients)*(var_sum + mu_sum**2)))**2/(len(self.selected_clients)/2*(mu_sum_sq + 2))**2
        
        part_2 = ((len(self.selected_clients)/2*(var_sum + mu_sum**2))**2 *len(self.selected_clients)*(2 + mu_sum_sq))/(len(self.selected_clients)/2*(mu_sum_sq + 2))**4

        self.var = part_1 + part_2

    def saveChanges(self, clients):                                            # To save the model as PCA

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        
        n = len([c for c in clients])
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        input = torch.stack(vecs, 1).unsqueeze(0)
        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        print(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False
        saveOriginal = True
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            print(f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
#             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            print(f'[Server] Update vectors have been saved to {savepath}')

    ## Aggregation functions ##

    def set_AR(self, ar):                                                      # Right now only fedavg and fedmedian
        #TODO: [AA] We need to use FedNova!
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'median':
            self.AR = self.FedMedian
        elif ar == 'fednova':
            self.AR = self.FedNova
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out
    
    def FedNova(self, clients):
        from utils.fednova import Net
        norm_vector = []
        data_ratio = []
        for c in clients:
            norm_vector.append(c.local_normalizing_vec)
            data_ratio.append(c.data_size)
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net(data_ratio = np.array(data_ratio)/np.sum(np.array(data_ratio)), normalized_vector_vals = np.array(norm_vector)).cpu()(arr.cpu()))
        return out
    
        ## Helper functions, act as adaptor from aggregation function to the federated learning system##

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        result = func(torch.stack(vecs, 1).unsqueeze(0))
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta(self.upload) for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]

        resultDelta = func(deltas)

        Delta.update(resultDelta)
        return Delta
