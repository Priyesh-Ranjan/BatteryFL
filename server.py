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
            for batch_data, batch_target in self.dataLoader:
                batch_target = batch_target.to(self.device)
                output = self.model(batch_data.to(self.device))
                test_loss += self.criterion(output, batch_target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(batch_target.view_as(pred)).sum().item()
                count += pred.shape[0]
                conf += confusion_matrix(batch_target.cpu(),pred.cpu(), labels = [i for i in range(self.num_classes)])
                f1 += f1_score(batch_target.cpu(), pred.cpu(), average = 'weighted')*count
                c+=count
                del batch_data, batch_target, output
                torch.cuda.empty_cache()        
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

        

        participations = [c.participation() for c in self.clients]
        loss_val = np.array([p[0] if p[2]>0 else 0 for p in participations])
        loss_val = np.array([v if v < 1e9 or np.all(v>=1e9) else np.mean(loss_val[loss_val < 1e9]) for v in loss_val])
        battery1 = np.array([max(0,p[1]) for p in participations])
        battery2 = np.array([max(0,p[2]) for p in participations])

        f1 = client_selection.f1(battery1, battery2, [c.cid for c in self.selected_clients])
        f2 = client_selection.f2(loss_val, [c.cid for c in self.selected_clients])
        
        mu_l, s_l = np.mean(loss_val), np.std(loss_val)+1e-9
        mu_b1, s_b1 = np.mean(battery1), np.std(battery1)+1e-9
        mu_b2, s_b2 = np.mean(battery2), np.std(battery2)+1e-9

        mu1, s1 = self.expected_loss(mu_l, s_l)
        mu2, s2 = self.expected_battery(mu_b1, s_b1, mu_b2, s_b2)
        #		P\left( \bigcap_{i=i}^2 \frac{f'_i(S)-f'_i(s)}{\sigma_i} \leq 0\right) \geq \prod_{i=1}^2 \left(1-\frac{\sigma_i^2}{\sigma_i^2+(f'_i(s)-\mu_0)^2}\right)
        self.prob_loss    = (1-s1**2/(s1**2 + (mu1 - f1)**2)) 
        self.prob_battery = (1-s2**2/(s2**2 + (mu2 - f2)**2))
        self.prob_dominated = self.prob_loss*self.prob_battery

        print("Clients selected this round are:",[c.cid for c in self.selected_clients])
        if self.selected_clients == []:
            return False
        
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
        
        
        return True

    def get_batteries(self) :
        batteries = []
        for c in self.selected_clients:
            batteries.append(c.report_battery())
        batteries = np.array(batteries)
        return np.mean(batteries), np.var(batteries)
    
    def expected_loss(self, mu, s) :
        # \sigma_2$ of $\frac{1}{8}+\frac{\sigma_l^4 + \mu_l^2 }{4\mu_l^2|\mathcal{C}|}
        s2 = 1/8 + (s**4 + mu**2)/(4*mu**2*len(self.clients))
        return 0.5, s2

    def expected_battery(self, mu1, s1, mu2, s2) :
        #E(X) = \frac{|\mathcal{c}|}{2}(\sigma_{b1}^2 + \sigma_{b2}^2) + \left(\frac{|\mathcal{c}|}{2}(\mu_{b1} + \mu_{b2})\right)^2 
        #E(Y) = \frac{|\mathcal{C}|}{2}(\mu_{b1}^2 + \mu_{b2}^2 +2)
        E_x = len(self.clients)/2*(s1**2 + s2**2) + (len(self.clients)/2*(mu1 + mu2))**2
        E_y = len(self.clients)/2*(mu1**2 + mu2**2 + 2)
        
        #var(X) <= \left(\frac{2}{|\mathcal{C}|((\sigma_{b1}^2 + \sigma_{b2}^2) + (\mu_{b1} + \mu_{b2})^2}\right)^2
        #var(Y) = |\mathcal{C}|(2+\mu_{b1}^2 + \mu_{b2}^2)
        V_x = (2/(len(self.clients)*((s1**2 + s2**2) + (mu1 + mu2)**2)))**2
        V_y = len(self.clients)*(2 + mu1**2 + mu2**2)

        # mu1 =  \frac{\mathbb{E}(X)}{\mathbb{E}(Y)} + \frac{\mathbb{E}(X) }{\mathbb{E}(Y)^3}\text{var}(Y)
        mu1 = E_x/E_y + E_x/(E_y**3)*V_y
        # s1 = 	\frac{\text{var}(X)}{\mathbb{E}(Y)^2} + \frac{\mathbb{E}(X)^2 }{\mathbb{E}(Y)^4}\text{var}(Y)
        s1 = V_x/(E_y**2) + E_x**2/(E_y**4)*V_y
        return mu1, s1

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
