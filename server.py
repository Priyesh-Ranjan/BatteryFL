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
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu', client_selection='ours'):
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

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

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

    def do(self):                                                              # One round of communication, client training is asynchronous
        if self.client_selection == 'ours' :
            selected_clients = client_selection.Our_Algorithm(self.clients)
        elif self.client_selection == 'AGE' :
            selected_clients = client_selection.genetic(self.clients, algorithm="age2")
        elif self.client_selection == 'NSGA' :    
            selected_clients = client_selection.genetic(self.clients, algorithm="nsga2")
        elif self.client_selection == "EAFL" :
            selected_clients = client_selection.eafl(self.clients)
        else :
            print("Selection Algorithm not recognized. Choosing all the clients")
            selected_clients = self.clients
        selected_clients = [c for c in selected_clients if c.battery > (c.upload + c.download)]           # Selecting clients with battery
        print("Clients selected this round are:",[c.cid for c in selected_clients])
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
        print("----------------------------------------------------------------")
        print("\n")
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
