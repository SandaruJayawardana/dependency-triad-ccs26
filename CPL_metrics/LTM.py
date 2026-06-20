"""
LTM Algorithm Implementation
============================

This module implements the LTM algorithm from the paper
"Quantifying Differential Privacy under Temporal Correlations."

The code is reproduced from the original implementation and is included here
for comparison with the proposed method.

Reference
---------
Quantifying Differential Privacy under Temporal Correlations

Author
------
Original authors of the LTM paper.
Reproduced/adapted by Sandaru Jayawardana for experimental evaluation.
"""


import numpy as np

def precompute(CMF):
    CMF = np.copy(CMF)
    # print(CMF)
    row_count, column_count = np.shape(CMF)

    qM = []
    dM = []
    aM = []

    for i in range(row_count):
        for j in range(row_count):
            if i == j:
                continue
            q = np.copy(CMF[i,:])
            d = np.copy(CMF[j,:])
            # print(CMF[i,:], q, CMF[j,:], d)
            for k in range(column_count):
                q_k = q[k]
                d_k = d[k]

                if q_k <= d_k:
                    q[k] = d[k] = 0
            
            if np.count_nonzero(q) == 0 and np.count_nonzero(d) == 0:
                qArr = q
                dArr = d
                aArr = q
            else:
                sorted_indices = np.argsort(q/d)[::-1]

                qArr = []
                dArr = []
                aArr = []
                sum_q = 0
                sum_d = 0
                for _, l in enumerate(sorted_indices):
                    sum_q += q[l]
                    sum_d += d[l]
                    qArr.append(sum_q)
                    dArr.append(sum_d)
                    aArr.append((q[l] - d[l])/(qArr[-1] * d[l] - dArr[-1] * q[l]))

            qM.append(qArr)
            dM.append(dArr)
            aM.append(aArr)
    
    return np.array(qM), np.array(dM), np.array(aM)

def binary_search(aArr, a):
    # print("aArr ", aArr, "a", a)
    for i in range(len(aArr)):

        k = len(aArr)-1-i

        if aArr[k] > a:
            return k
        
    return 0

def binary_search_new(aArr, a):
    # print("aArr", aArr, "a", a)
    l = len(aArr)
    if l == 1:
        return 0
    if l == 0:
        return 0
    if aArr[l//2] == a:
        return l//2
    elif aArr[l//2] > a or np.isnan(aArr[l//2]):
        return l//2 + binary_search_new(aArr[l//2:], a)
    else:
        return binary_search_new(aArr[:l//2], a)
    


def LTM(eps, qM, dM, aM):
    # print("a, qM, dM, aM", a, qM, dM, aM)
    L = 0
    q = 0
    d = 0
    for j in range(np.shape(qM)[0]):
        qArr = qM[j]
        dArr = dM[j]
        aArr = aM[j]
        k = binary_search(aArr=aArr, a=eps)
        
        PL = np.log((qArr[k] * (np.exp(eps)-1) + 1)/(dArr[k] * (np.exp(eps)-1) + 1))

        if PL > L:
            L = PL
            q = qArr[k]
            d = dArr[k]
    # print("L, q, d ", L, q, d)
    return L, q, d
