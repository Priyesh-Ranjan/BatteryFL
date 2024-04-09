import numpy as np
import pymoo.core.problem
import pymoo.algorithms.moo.nsga2
import pymoo.algorithms.moo.age2
import pymoo.operators.crossover
import pymoo.operators.mutation.bitflip
import pymoo.optimize
import pymoo.operators.sampling.rnd

def f1(battery_current, battery_future, S):                          # Gives the f1 value from the battery levels
        if len(S) == 0 :
            not_S = [i for i in range(len(battery_current))]
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
    
def f2(loss_val, S):                                                     # Gives the f2 values from the loss levels
        idx = np.array(S)
        if len(S) == 0:
            return 0
        return np.sum(loss_val[S]) / np.sum(loss_val)

def dominates(i, j, objectives):                                     # Check if the first solution dominates the second
        for o in objectives:
            if not o[i] >= o[j]:
                return False
        for o in objectives:
            if o[i] > o[j]:
                return True
        return False

def Our_Algorithm(clients):                                                  # Select client function (Need to fix some errors)
        num_clients = len(clients)
        participations = [c.participation() for c in clients]
        loss_val = np.array([p[0] for p in participations])
        loss_val = np.array([v if v < 1e10 else np.max(loss_val[loss_val < 1e10]) for v in loss_val])
        battery1 = np.array([p[1] for p in participations])
        battery2 = np.array([p[2] for p in participations])
        S = []
        while len(S) < len(clients):
            current_score = min(f2(loss_val, S), f1(battery1, battery2, S))
            F1 = [f1(battery1, battery2, list(set(S + [c]))) for c in range(num_clients)]
            F2 = [f2(loss_val, list(set(S + [c]))) for c in range(num_clients)]
            F = np.minimum(F1, F2)
            l = [i for i in range(num_clients) if i not in S and clients[i].battery > (clients[i].upload + clients[i].download)]
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
                    if dominates(c_prime, c, [F1, F2]):
                        ND = False
                        break
                if ND:
                    N.append(c)
            max_index = np.argmax(F[N])
            S.append(N[max_index])

        selected_clients = [clients[c] for c in S]   
        return selected_clients

class BatteryLossOptimization(pymoo.core.problem.Problem):
    def __init__(self, clients):
        super().__init__(n_var=len(clients),
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([0]*len(clients)),  # lower bounds
                         xu=np.array([1]*len(clients)))  # upper bounds
        self.clients = clients

    def _evaluate(self, x, out, *args, **kwargs):
        out_F = []
        for row in x:
            S = [i for i in range(len(row)) if row[i]]
            participations = [c.participation() for c in self.clients]
            loss_val = np.array([p[0] for p in participations])
            loss_val = np.array([v if v < 1e10 else np.mean(loss_val[loss_val < 1e10]) for v in loss_val])
            battery1 = np.array([p[1] for p in participations])
            battery2 = np.array([p[2] for p in participations])
            out_F.append([-f1(battery1, battery2, S), -f2(loss_val, S)])
        out["F"] = np.array(out_F)
         

def genetic(clients, algorithm="nsga2", generations=20, population_size=100, mutation_rate=0.1):
    #each element in the population is encoded as a binary vector of length len(clients)
    options = {
         "pop_size": population_size,
            "sampling": pymoo.operators.sampling.rnd.BinaryRandomSampling(),
            "crossover": pymoo.operators.crossover.sbx.SBX(),
            "mutation": pymoo.operators.mutation.bitflip.BitflipMutation(mutation_rate),
            "eliminate_duplicates": True
    }

    if algorithm == "nsga2":
         alg = pymoo.algorithms.moo.nsga2.NSGA2(**options)
    elif algorithm == "age2":
         alg = pymoo.algorithms.moo.age2.AGEMOEA2(**options)

                                 
    res = pymoo.optimize.minimize(BatteryLossOptimization(clients), 
                                    alg, 
                                    termination=('n_gen', generations),
                                    seed=1,
                                    verbose=False)

    F = np.maximum(res.F[:,0], res.F[:,1])
    best_idx = np.argmax(F)
    best_solution = res.X[best_idx]
    return [clients[i] for i in range(len(clients)) if best_solution[i]]

def eafl(clients, f=0.25, selected=5):
    participations = [c.participation() for c in clients]
    loss_val = np.array([p[0] for p in participations])
    loss_val = np.array([v if v < 1e10 else np.mean(loss_val[loss_val < 1e10]) for v in loss_val])
    battery1 = np.array([p[1] for p in participations])
    
    reward = (1-f)*(battery1) + f*loss_val

    selected_clients = [clients[i] for i in np.argsort(reward)[:selected]]
    return selected_clients