from __future__ import print_function

from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import numpy as np

from utils import utils
import time

class Server():
    def __init__(self, model, dataLoader, upload_battery, download_battery, collection_battery, training_battery, collection_size, criterion=F.nll_loss, device='cpu'):
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
        self.path_to_aggNet = ""
        self.upload = upload_battery
        self.download = download_battery                                       # Disregard
        self.collection = collection_battery                                   # Disregard
        self.training = training_battery                                       # Disregard
        self.collection_size = collection_size                                 # Disregard

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    def test(self):
        print("[Server] Start testing \n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        c = 0
        f1 = 0
        conf = np.zeros([10,10])
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
            conf += confusion_matrix(target.cpu(),pred.cpu(), labels = [i for i in range(10)])
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

    def f1(self, battery_current, battery_future, S):                          # Gives the f1 value from the battery levels
        if len(S) == 0 :
            not_S = [i for i in range(len(self.clients))]
        else :     
            not_S = [i for i in range(len(battery_current)) if i not in S]

        if len(not_S) != 0 :
            sum_battery_current_not_S_squared = sum(battery_current[i] ** 2 for i in not_S)
            sum_battery_current_not_S = sum(battery_current[i] for i in not_S)
        else :
            sum_battery_current_not_S_squared = 0 ; sum_battery_current_not_S = 0
        
        if len(S) != 0 :
            sum_battery_future_S_squared = sum(battery_future[i] ** 2 for i in S)
            sum_battery_future_S = sum(battery_future[i] for i in S)
        else :
            sum_battery_future_S_squared = 0
            sum_battery_future_S = 0
    
        numerator = (sum_battery_current_not_S + sum_battery_future_S) ** 2
        denominator = sum_battery_current_not_S_squared + sum_battery_future_S_squared
        fitness_value = numerator / denominator / len(battery_current)
    
        return fitness_value
    
    def f2(self, loss, S):                                                     # Gives the f2 values from the loss levels
        loss_val = np.array(loss)
        idx = np.array(S)
        if len(S) == 0:
            return 0
    
        #non_negative_losses_idx = loss_val[idx >= 0]
        #non_negative_losses = loss_val[loss_val >= 0]
    
        #if not non_negative_losses.any():
        #    return 0
    
        return np.sum(loss_val[S]) / np.sum(loss_val)

    def Select_Clients(self):                                                  # Select client function (Need to fix some errors)
        num_clients = len(self.clients)
        loss_val = []; battery1 = []; battery2 = []
        for c in self.clients :
            loss, battery_current, battery_future = c.participation()
            battery2.append(battery_future); battery1.append(battery_current); loss_val.append(loss)
        loss_val = [v if v < 1e10 else np.mean(loss_val) for v in loss_val]
        S = []
        while len(S) < len(self.clients):
            current_score = min(self.f2(loss_val, S), self.f1(battery1, battery2, S))
            F1 = [self.f1(battery1, battery2, list(set(S + [c]))) for c in range(num_clients)]
            F2 = [self.f2(loss_val, list(set(S + [c]))) for c in range(num_clients)]
            F = np.minimum(F1, F2)
            l = [i for i in range(num_clients) if i not in S and self.clients[i].battery > 0]
            if l==[] or max(F[l]) <= current_score:
                break
            N = []
            for c in range(num_clients):
                if c in S: 
                    continue
                ND = True
                for c_prime in range(num_clients):
                    if c_prime in S + [c]: 
                        continue
                    if (F1[c_prime] >= F1[c]) and (F2[c_prime] >= F2[c]) and (F1[c_prime] > F1[c] or F2[c_prime] > F2[c]):
                        ND = False
                        break
                if ND:
                    N.append(c)
            max_index = np.argmax(F[N])
            S.append(N[max_index])

        selected_clients = [self.clients[c] for c in S]   
        print("Clients selected this round are:",S)
        return selected_clients

    def do(self):                                                              # One round of communication, client training is asynchronous
        selected_clients = self.Select_Clients()                              # Trying to debug this still
        #selected_clients = [c for c in self.clients if c.battery > 0]           # Selecting clients with battery
        if selected_clients == []:
            return False
        for c in selected_clients:
            c.setModelParameter(self.model.state_dict())                       # distribute server model to clients
            c.perform_task()                                                   # selected clients will perform the FSM

        if self.isSaveChanges:
            self.saveChanges(selected_clients)

        tic = time.perf_counter()
        Delta = self.AR(selected_clients)                                      # Getting model weights from the clients
        toc = time.perf_counter()
        print("\n")
        print("-----------------------------Server-----------------------------")
        print(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")
        
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1
        return True

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
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
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
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
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
