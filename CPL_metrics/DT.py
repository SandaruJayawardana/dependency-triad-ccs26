
"""
Dependency Triad (DT) Metric Implementation
===========================================

This module implements the Dependency Triad (DT) metric for quantifying
correlation-induced privacy leakage between dependent attributes under Local
Differential Privacy (LDP). The DT metric summarizes the dependency between
a target attribute X_k and a dependent attribute \hat{X} using three parameters:
(alpha, beta, delta). These parameters can then be used to estimate the
correlation-induced privacy leakage (CPL) under a given privacy budget epsilon.

The main function is `DT()`, which supports two modes:

1. Parameter estimation:
   Computes the DT parameters (alpha, beta, delta) from a conditional
   probability matrix (CMF) and an optional distributional uncertainty matrix
   (Delta).

2. Leakage estimation:
   Computes the CPL value from previously estimated DT parameters.

Main Functions
--------------
DT(eps, CMF, Delta=0, leakage_only=False, DT_parameters=(0, 0, 0),
   get_parameters=False)
    Main interface for computing DT parameters or the corresponding CPL.

cal_(CMF, Delta, tol=1e-12)
    Computes the DT parameters (alpha, beta, delta) from the conditional
    probability matrix and uncertainty bounds.

CPL(eps, alpha, beta, delta)
    Computes the estimated CPL using the DT parameters and privacy budget eps.

Inputs
------
CMF : numpy.ndarray
    Conditional probability matrix representing p(\hat{x} | x), where each row
    corresponds to a value of X_k and each column corresponds to a value of
    \hat{X}.

Delta : numpy.ndarray or int, optional
    Distributional uncertainty matrix associated with CMF. If Delta=0, no
    uncertainty is assumed.

eps : float
    Local differential privacy budget.

Outputs
-------
Depending on the selected mode, the module returns either:
    - the DT parameters (alpha, beta, delta), or
    - the estimated CPL value.

Author
------
Sandaru Jayawardana

"""

import numpy as np
import math

from scipy.optimize import linprog

def ignore_nan(arr):
    return max(filter(lambda x: not (math.isinf(x) or math.isnan(x)), arr))

def CPL(eps, alpha, beta, delta):
    Alpha = np.exp(alpha)
    Beta = np.exp(beta)
    Eps = np.exp(eps)

    B = (Beta - 1)*(1 - delta) / (Alpha * Beta - 1)

    if (-1 + delta + Alpha - delta * Eps) < 0:
        B = 0

    A = Alpha*B + delta
    leakage = np.log((1 + A * (Eps - 1)) / (1 + B * (Eps - 1)))
    
    return leakage

def cal_(CMF, Delta, tol = 1e-12):
    alpha = 0
    beta = 0
    delta = 0
    phi = 0

    num_rows, b = np.shape(CMF)

    for i in range(num_rows):
        G = CMF[i,:]
        L = G - Delta[i,:]
        U = G + Delta[i,:]
        for j in range(num_rows):
            if i == j:
                continue
            delta_temp = 0
            G_prime = CMF[j,:]
            L_prime = G_prime - Delta[j,:]
            U_prime = G_prime + Delta[j,:]

            M_plus = np.maximum(0, U - L_prime)
            M_minus = np.maximum(0, U_prime - L)
            denom = M_plus + M_minus
            theta = np.zeros(b)
            theta[denom > tol] = M_plus[denom > tol] / denom[denom > tol]

            for k in range(b):
                g = G[k]
                g_prime = G_prime[k]
                d = Delta[i, k]
                d_prime = Delta[j, k]
                alpha = max(alpha, np.log(min(1, g + d)/(g_prime - d_prime)))

                if g_prime - d_prime <= 0:
                    delta_temp += g + d

            delta = min(1, max(delta, delta_temp))
            n = 3 * b
            c = np.zeros(n)
            c[2*b:3*b] = -1          # maximize sum z = minimize -sum z

            # sum G = 1, sum Gp = 1
            Aeq = np.zeros((2, n))
            Aeq[0, 0:b] = 1
            Aeq[1, b:2*b] = 1
            beq = np.array([1, 1])

            # z_i <= theta_i (G_i - Gp_i + M_minus_i)
            Aub = np.zeros((b, n))
            bub = theta * M_minus

            for ii in range(b):
                Aub[ii, ii] = -theta[ii]          # -theta_i G_i
                Aub[ii, b+ii] = theta[ii]         # +theta_i Gp_i
                Aub[ii, 2*b+ii] = 1              # +z_i

            bounds = (
                [(L[ij], U[ij]) for ij in range(b)] +
                [(L_prime[ik], U_prime[ik]) for ik in range(b)] +
                [(0, None) for _ in range(b)]
            )

            res = linprog(c, A_ub=Aub, b_ub=bub,
                        A_eq=Aeq, b_eq=beq,
                        bounds=bounds, method="highs-ipm") 

            if not res.success:
                raise RuntimeError(res.message)
            
            x = res.x
            z = x[2*b:]
            phi = max(phi, sum(z))

    if np.isposinf(alpha):
        print("phi ", phi)
        beta = np.log((-1 + delta) / (-1 + phi - 1e-12))
    else:
        beta = np.log((1 + phi + np.exp(alpha) * (-1 + delta) - 2 * delta) / (1 + (-1 + phi) * np.exp(alpha) - delta))
    
    if np.isnan(beta):
        beta = alpha
    return alpha, beta, delta

def DT(eps, CMF = 0, Delta = 0, leakage_only = False, DT_parameters = (0,0,0), get_parameters=False):
    if leakage_only:
        alpha, beta, delta = DT_parameters
        return CPL(alpha=alpha, beta=beta, delta=delta, eps=eps)
    
    if isinstance(Delta, int):
        Delta = np.zeros((np.shape(CMF)))
    
    
    print(np.shape(Delta))
    alpha, beta, delta = cal_(CMF, Delta)

    if get_parameters:
        return alpha, beta, delta
    return CPL(alpha=alpha, beta=beta, delta=delta, eps=eps)
