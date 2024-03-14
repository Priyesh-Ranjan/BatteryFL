import numpy as np
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
    
def f2(loss, S):                                                     # Gives the f2 values from the loss levels
        loss_val = np.array(loss)
        idx = np.array(S)
        if len(S) == 0:
            return 0
    
        #non_negative_losses_idx = loss_val[idx >= 0]
        #non_negative_losses = loss_val[loss_val >= 0]
    
        #if not non_negative_losses.any():
        #    return 0
    
        return np.sum(loss_val[S]) / np.sum(loss_val)

def Our_Algorithm(clients):                                                  # Select client function (Need to fix some errors)
        num_clients = len(clients)
        loss_val = []; battery1 = []; battery2 = []
        for c in clients :
            loss, battery_current, battery_future = c.participation()
            battery2.append(battery_future); battery1.append(battery_current); loss_val.append(loss)
        loss_val = [v if v < 1e10 else np.mean(loss_val) for v in loss_val]
        S = []
        while len(S) < len(clients):
            current_score = min(f2(loss_val, S), f1(battery1, battery2, S))
            F1 = [f1(battery1, battery2, list(set(S + [c]))) for c in range(num_clients)]
            F2 = [f2(loss_val, list(set(S + [c]))) for c in range(num_clients)]
            F = np.minimum(F1, F2)
            l = [i for i in range(num_clients) if i not in S and clients[i].battery > 0]
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

        selected_clients = [clients[c] for c in S]   
        print("Clients selected this round are:",S)
        return selected_clients