import torch

def LTM_loss(a, qArr, dArr, aArr, tiny=1e-12):
    """
    PyTorch version of cal_PL.

    a: scalar tensor
    qM: tensor of shape [num_rows, num_grid]
    dM: tensor of shape [num_rows, num_grid]
    aM: tensor of shape [num_rows, num_grid]

    Assumption: each aM[j] is sorted in ascending order.
    """

    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=qArr.dtype, device=qArr.device)


    PL_values = []
    q_values = []
    d_values = []

    exp_a_minus_1 = torch.exp(a) - 1.0

    k = torch.searchsorted(aArr, a, right=True) - 1
    k = torch.clamp(k, min=0, max=aArr.shape[0] - 1)

    q = qArr[k]
    d = dArr[k]

    PL = torch.log(
            (q * exp_a_minus_1 + 1.0 + tiny) /
            (d * exp_a_minus_1 + 1.0 + tiny)
        )

    PL_values.append(PL)
    q_values.append(q)
    d_values.append(d)

    PL_values = torch.stack(PL_values)

    max_idx = torch.argmax(PL_values)

    L = PL_values[max_idx]
    # q_selected = torch.stack(q_values)[max_idx]
    # d_selected = torch.stack(d_values)[max_idx]

    L = torch.clamp_min(L, 0.0)

    return L
