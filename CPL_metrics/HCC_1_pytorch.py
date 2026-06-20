'''
    Algorithm HCC-1
    Paper - The Hidden Cost of Correlation: Rethinking Privacy Leakage in Local Differential Privacy.
'''

import torch

def random_response_mechanism_torch(alphabet_size, eps, delta=0.0, device=None, dtype=torch.float32):
    """
    PyTorch GRR / random response matrix.
    Q[y, x] = Pr[Y = y | X = x]
    """

    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=dtype, device=device)

    E = torch.exp(eps)

    denom = alphabet_size - 1 + E

    diag_val = (E + (alphabet_size - 1) * delta) / denom
    off_val = (1.0 - delta) / denom

    Q = torch.full(
        (alphabet_size, alphabet_size),
        off_val,
        dtype=dtype,
        device=device
    )

    Q.fill_diagonal_(diag_val)

    return Q

def hcc1_loss_hard(Q, CMF, eps, is_GRR=False, tiny=1e-12):
    """
    Differentiable PyTorch version of HCC_1.

    Q:   tensor of shape [domain_size, domain_size]
    CMF: tensor of shape [num_rows, domain_size]
    eps: scalar float or tensor
    """

    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=CMF.dtype, device=CMF.device)

    num_rows, num_columns = CMF.shape

    if num_rows == 1:
        return eps * 0.0

    if is_GRR:
        Q = random_response_mechanism_torch(
            num_columns,
            eps,
            device=CMF.device,
            dtype=CMF.dtype
        )

    # scores[i, k] = sum_x Q[x, k] * CMF[i, x]
    scores = CMF @ Q

    # ratio[i, j, k] = scores[i, k] / scores[j, k]
    numerator = scores[:, None, :]
    denominator = scores[None, :, :]

    log_ratio = torch.log(numerator + tiny) - torch.log(denominator + tiny)

    # remove i == j cases
    mask = ~torch.eye(num_rows, dtype=torch.bool, device=CMF.device)
    log_ratio = log_ratio[mask]

    leakage = torch.max(log_ratio)

    # For training, lower clamp is okay.
    leakage = torch.clamp_min(leakage, 0.0)

    # Optional upper clamp, but be careful: it can kill gradients when leakage > eps.
    # leakage = torch.minimum(leakage, eps)

    return leakage
