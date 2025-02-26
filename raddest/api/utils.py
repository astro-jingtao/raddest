import numpy as np
from raddest.reg.constructor import parse_piecewise_params

def to_each_piece_params(n_piece, params):
    sig = params[-1]
    b0, x0, k_lst, dx_lst = parse_piecewise_params(n_piece, params[:-1])
    k_lst = list(k_lst)
    b_lst = [b0]
    x_lst = [x0]
    for i in range(1, n_piece):
        this_b = (k_lst[i - 1] - k_lst[i]) * x_lst[i - 1] + b_lst[i - 1]
        b_lst.append(this_b)
        if i < n_piece - 1:
            this_x = x_lst[i - 1] + dx_lst[i - 1]
            x_lst.append(this_x)
    k_lst = np.asarray(k_lst)
    b_lst = np.asarray(b_lst)
    x_lst = np.asarray(x_lst)
    sig = float(sig)
    return k_lst, b_lst, x_lst, sig