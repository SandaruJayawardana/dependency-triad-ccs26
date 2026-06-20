'''
    Estimate CPL with calibrating beta.
'''

import numpy as np
import math

from src.CPL_metrics.HCC_2 import HCC_2 

def ignore_nan(arr):
    return max(filter(lambda x: not (math.isinf(x) or math.isnan(x)), arr))

def cal_CPL(eps, alpha, beta, delta):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)

    B = (Beta - 1)*(1 - delta) / (Alpha * Beta - 1)

    if (-1 + delta + Alpha - delta * Eps) < 0:
        B = 0

    A = Alpha*B + delta

    leakage = np.log((1 + A * (Eps - 1)) / (1 + B * (Eps - 1)))

    return leakage

def get_alpha(G, G_prime):
    ratio = G/G_prime
    filter_ratios = []

    for i in ratio:
        if not(math.isinf(i) or math.isnan(i)):
            filter_ratios.append(i)

    return np.log(max(filter_ratios))

def delta_compute(CMF, alpha):
    delta_list = []
    m, t = np.shape(CMF)
    Alpha = np.exp(alpha)

    for i in range(m):
        G = CMF[i,:]
        for j in range(m):
            if i == j:
                continue
            G_prime = CMF[j,:]
            delta_inner = 0
            for k in range(t):
                g = G[k]
                g_prime = G_prime[k]
                delta_inner += max(0, g - max(0, Alpha * g_prime))
            delta_list.append(delta_inner)

    delta = min(1, max(delta_list))
    return delta

def calibrate_CPL(eps, CMF = [], eps_0=0.1, CPL_star = 0):
    CPL_star, _ = HCC_2(eps_0, CMF)
    num_rows, _ = np.shape(CMF)
    alpha_list = []

    for i in range(num_rows):
        G = CMF[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            G_prime = CMF[j,:]
            alpha = get_alpha(np.array(G), np.array(G_prime))
            alpha_list.append(alpha)

    alpha = max(alpha_list)
        
    beta = max(0, np.log((1+np.exp(alpha)*(np.exp(eps_0)-1)-np.exp(eps_0)*np.exp(CPL_star))/(np.exp(alpha+eps_0)+np.exp(CPL_star)-(np.exp(alpha)+np.exp(eps_0))*np.exp(CPL_star))))    
    delta = delta_compute(CMF = CMF, alpha = alpha)
    CPL = cal_CPL(eps = eps, alpha = alpha, beta = beta, delta = delta)
    
    return CPL
