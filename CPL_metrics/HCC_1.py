'''
    Algorithm 1
'''

import numpy as np

def random_response_mechanism(alphabet_size, eps, delta = 0):
    vector = (np.exp(eps))*np.ones(alphabet_size)-1 + alphabet_size * delta
    return (np.diag(vector) + 1 - delta)/(alphabet_size - 1 + np.exp(eps))

def compute_leakage(C, G, G_prime):
    return np.sum(C*G)/np.sum(C*G_prime)

def HCC_1(Q, CMF, eps, is_GRR = False):
    # print("Algo 1 optimal")
    num_rows, num_columns = np.shape(CMF)

    if is_GRR:
        Q = random_response_mechanism(num_columns, eps)
    
    num_rows_Q, num_columns_Q = np.shape(Q)
    max_val = 0
    
    for k in range(num_columns_Q):
        C = Q[:,k]
        for i in range(num_rows):
            G = CMF[i,:]
            for j in range(num_rows):
                if i == j:
                    continue
                G_prime = CMF[j,:]
                
                L = compute_leakage(C, G, G_prime)
                # print("C", C, "G ", G, " G_prime ", G_prime, " L ", np.log(L))
                if max_val < L:
                    max_val = L
    leakage = np.log(max_val)
    leakage = max(0, leakage)
    leakage = min(eps, leakage)
    return leakage
 