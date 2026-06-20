'''
    Algorithm 2
'''
import numpy as np
import math
import time

def ignore_nan(arr):
    return max(filter(lambda x: not (math.isinf(x) or math.isnan(x)), arr))

def cal_leakage1(eps, alpha, beta, delta, t, r):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)
    leakage = np.log(1 + delta*(t-r)*(Eps-1))
    # print("leakage1 ", leakage)
    return leakage

def cal_leakage2(eps, alpha, beta, delta, t, r):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)
    leakage = np.log((-1 + Alpha*Beta*Eps + Alpha*(1- r*delta) + Alpha*Eps*(delta*r-1)  + delta*Eps*(r-t) + delta*(t-r))/(- delta*r + Eps*(- 1 + delta*r + Beta*(1 + delta*(r-t)))  + Beta*(-1 + Alpha + delta*(t-r))))
    # print("leakage2 ", leakage)
    return leakage

def cal_leakage3(eps, alpha, beta, delta, t, r):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)

    leakage = np.log((Alpha*Eps)/(-1 + Alpha + Eps * (1 + delta * (r - t)) + delta * (t - r)))
    # print("leakage3 ", leakage)
    return leakage

def delta_constraint(alpha, eps, delta, t, r):
    Alpha = np.exp(alpha)
    Eps = np.exp(eps)

    expression = (Alpha - 1)/((Eps - 1) * (t - r))
    # print( " expression ", expression, " delta ", delta)
    return delta >= min(expression, 1/(t-1))

def g_prime(alpha, beta, eps, delta, t, r):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)
    flag = 0
    if delta_constraint(alpha, eps, delta, t, r):
        g_prime = 0
        flag = 0
    else:
        g_prime_1 = (-1 + Beta + delta * r + Beta * delta * r)/((Alpha * Beta -1)*(t - r))
        g_prime_2 = (1- delta * (t - r))/((t - r) * Alpha)

        g_prime = min(g_prime_1, g_prime_2)
        if g_prime_1 <= g_prime_2:
            flag = 1
        else:
            flag = 2
    if g_prime <= 0:
        g_prime = 0
        flag = 0
    # print(" g_prime ", g_prime)
    return g_prime, flag

def r_constraints(alpha, beta, eps, delta, g_prime, t, r):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)

    r_1 = (1 - Beta + Beta * delta * t - g_prime * t + Alpha * Beta * g_prime * t)/((Alpha * Beta - 1)*g_prime + delta * (Beta + 1))
    r_2 = (t * Alpha * g_prime + t * delta - 1)/(Alpha * g_prime + delta)
    r_max = np.ceil(max(r_1, r_2, 1))
    r_max = min(r_max, t-1)
    # print("r_1 ", r_1, " r_2 ", r_2, "r_max ", r_max, " t ", t)
    return r_max == r

def get_alpha_beta_new(G, G_prime, delta_initial = 0):
    delta = 0
    for i in range(len(G)):
        if G_prime[i] == 0:
            if delta < G[i]:
                delta = G[i]
    G = G - delta
    ratio = G/G_prime
    filter_ratio_array = []
    highest_ratio = 0
    p1 = 0
    for index_i, i in enumerate(ratio):
        if not(math.isinf(i) or math.isnan(i)):
            filter_ratio_array.append(i)
            if highest_ratio < i:
                highest_ratio = i
                p1 = G[index_i]
    ratio = np.array(filter_ratio_array)
    # print(ratio_array)

    if len(ratio) == 0:
        alpha = 0.000001
    else:
        ratio = np.sort(ratio)
        alpha = np.log(ratio[-1]) if ratio[-1] > 0 else 0.000001
    return alpha, delta+delta_initial, p1

def computing_optimal_t(G, G_prime, alpha, beta, delta_initial, delta):
    effective_t = 0
    original_t = len(G)
    
    for i in range(original_t):
        if ((G[i]+delta_initial) > (np.exp(alpha)*(G_prime[i]-delta_initial)+delta)) or ((G[i]+delta_initial) > (np.exp(alpha)*(G_prime[i]-delta_initial))): # or (G_prime[i]+delta) > np.exp(beta)*(G[i]-delta):
            effective_t += 1
    return effective_t

def Algo2_privacy_leakage_final(eps, CMF = [], sparse_approx =  False, delta_ = 0, default_beta = 0, is_calibrated = False, optimal_APL=0, eps_0=0.1, for_distribution_bias = False, DB_value = False, leakage_only = False, DB_parameters = (0, 0, 0, 0)):
    if not(leakage_only):
        num_rows, num_columns = np.shape(CMF)
        max_val = 0
        alpha_list = []
        beta_list = []
        p1_dict = {}
        delta_list = [delta_]
        t = np.shape(CMF)[1]

        for i in range(num_rows):
            G = CMF[i,:]
            for j in range(num_rows):
                if i == j:
                    continue
                G_prime = CMF[j,:]
                G_1 = []
                G_prime_1 = []

                if sparse_approx:
                    for k in range(num_columns):
                        if  G[k] >= 0.001 or G_prime[k] >= 0.001: # if  G[k] != 0 or G_prime[k] != 0:
                            G_1.append(G[k])
                            G_prime_1.append(G_prime[k])
                    CMF_new = np.array([G_1, G_prime_1])
                    t = np.shape(CMF_new)[1] 
                    if t < 2:
                        continue
                else:
                    t = np.shape(CMF)[1]
                    G_1 = G
                    G_prime_1 = G_prime

                alpha, delta_1, p1 = get_alpha_beta_new(np.array(G_1), np.array(G_prime_1), delta_)
                if for_distribution_bias:
                    str_alpha = str(alpha)
                    if str_alpha in p1_dict.keys():
                        if p1_dict[str_alpha] < p1:
                            p1_dict[str_alpha] = p1
                    else:
                        p1_dict[str_alpha] = p1
                delta = delta_1
                delta_list.append(delta)
                if default_beta == 0:
                    beta, delta_2, _ = get_alpha_beta_new(np.array(G_prime_1), np.array(G_1), delta_)
                else:
                    beta = default_beta
                if alpha < beta: # or delta_1 < delta_2:
                    continue
                
                alpha_list.append(alpha)
                beta_list.append(beta)
        try:      
            alpha = max(alpha_list)
            beta = max(beta_list)
            delta = max(delta_list)
        except:
            return eps, 0, 0.1
        
        if is_calibrated:
            beta = np.log((1+np.exp(alpha)*(np.exp(eps_0)-1)-np.exp(eps_0)*np.exp(optimal_APL))/(np.exp(alpha+eps_0)+np.exp(optimal_APL)-(np.exp(alpha)+np.exp(eps_0))*np.exp(optimal_APL)))
            if beta < 0:
                beta = 0
            # print("theoretical_beta ", beta, " initial beta ", beta, " alpha ", alpha, " optimal_APL ", optimal_APL)

        # if DB_value:
        #     return alpha, beta, delta
    else:
        max_val = 0
        alpha, beta, delta, t = DB_parameters

    start_time = time.time()

    leakage = "Error"
    if delta > 0:
        print("Delta active")
        t_list = [2]
        for i in range(num_rows):
            G = CMF[i,:]
            for j in range(num_rows):
                if i == j:
                    continue
                G_prime = CMF[j,:]
                t_list.append(computing_optimal_t(G, G_prime, alpha, beta, delta_initial=delta_, delta=delta))
        t = max(t_list)
        print("original t ", np.shape(CMF)[0], np.shape(CMF)[1], " new t ", t)
    if DB_value:
            return alpha, beta, delta, t
    if delta == 0:
        leakage = np.log((-1 + np.exp(alpha)+np.exp(alpha+eps)*(-1 + np.exp(beta)))/(np.exp(beta)*(-1 + np.exp(alpha))+np.exp(eps)*(-1 + np.exp(beta))))
    else:
        for r in range(1, t):
            g, flag = g_prime(alpha=alpha, beta=beta, eps=eps, delta=delta, t=t, r=r)

            if r_constraints(alpha=alpha, beta=beta, eps=eps, delta=delta, g_prime=g, t=t, r=r):
                if flag == 0:
                    leakage = cal_leakage1(alpha=alpha, beta=beta, eps=eps, delta=delta, t=t, r=r)
                elif flag == 1:
                    leakage = cal_leakage2(alpha=alpha, beta=beta, eps=eps, delta=delta, t=t, r=r)
                else:
                    leakage = cal_leakage3(alpha=alpha, beta=beta, eps=eps, delta=delta, t=t, r=r)
                # print("r val ", r, " delta ", delta, "leakage ", leakage)
                break
        if delta >= 2/t:
            leakage = eps
    if leakage > max_val:
        max_val = leakage
    # print("Dependency Budget leakage new", max_val, "delta ", delta)
    end_time = time.time()
    if for_distribution_bias:
        return max_val, end_time-start_time, alpha, p1_dict[str(alpha)]
    return max_val, end_time-start_time, beta


def uncalibrate_CPL(eps, CMF, sparse_approx =  False, delta_ = 0, default_beta = 0, target_val = 0):
    beta = 0
    max_val,t, beta = Algo2_privacy_leakage_final(eps, CMF, sparse_approx =  sparse_approx, delta_ = delta_, default_beta = beta)
    init_max = max_val
    initial_beta = beta
    while target_val - max_val < 0.01:
        max_val,t, beta = Algo2_privacy_leakage_final(eps, CMF, sparse_approx =  sparse_approx, delta_ = delta_, default_beta = beta)
        beta -= min(0.2,0.01*max(abs(target_val - max_val)*100, 1))
        if beta <= 0:
            beta = 0.01
            print("Hit minus!", " target_val ", target_val, " max_val ", max_val, 'initial max', init_max, "initial_beta ", initial_beta)
            return max_val, t, beta
    # print("target_val ", target_val, "Final max_val", max_val, "beta", beta)
    beta += 0.2 # min(0.1,0.005*max(abs(target_val - max_val)*1, 1))
    return max_val, t, beta
