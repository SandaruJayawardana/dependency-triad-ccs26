'''
    Algorithm HCC_2
'''
import numpy as np
import time 

def compute_H(G, G_prime, eps):
    Q = G/(G_prime + 0.000000000000001)
    sorted_indices = np.argsort(Q)[::-1]
    A = 0
    B = 0

    for index_i, i in enumerate(sorted_indices):
        if Q[i] > (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1)):
            A += G[i]
            B += G_prime[i]
        else:
            break
    return (1+A*(np.exp(eps)-1))/(1+B*(np.exp(eps)-1))

def HCC_2(eps, CMF):
    num_rows, _ = np.shape(CMF)
    max_val = 0
    start_time = time.time()
    if num_rows == 1:
        return eps, 0
    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            G_prime = CMF[j,:]
            L = compute_H(G, G_prime, eps)
            if max_val < L:
                max_val = L
    leakage = np.log(max_val)
    end_time = time.time()
    leakage = min(leakage, eps)
    leakage = max(0, leakage)
    return leakage, end_time - start_time
 