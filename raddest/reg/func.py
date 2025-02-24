import numpy as np
from scipy.special import expit as _sigmoid


def sigmoid(x, x0, beta):
    return _sigmoid(beta * (x - x0))


def get_saturated_pars(k, b, x0, x1, c0, c1):
    n0 = k / (k * x0 + b - c0)
    n1 = k / (k * x1 + b - c1)
    m0 = k / (n0 * np.exp(n0 * x0))
    m1 = k / (n1 * np.exp(n1 * x1))
    return n0, n1, m0, m1


def saturated_linear(x,
                     k,
                     b,
                     x0,
                     x1,
                     c0,
                     c1,
                     n0=None,
                     n1=None,
                     m0=None,
                     m1=None):
    if n0 is None or n1 is None or m0 is None or m1 is None:
        n0, n1, m0, m1 = get_saturated_pars(k, b, x0, x1, c0, c1)
    y = np.zeros_like(x)
    y[x < x0] = m0 * np.exp(n0 * x[x < x0]) + c0
    y[x > x1] = m1 * np.exp(n1 * x[x > x1]) + c1
    y[(x >= x0) & (x <= x1)] = k * x[(x >= x0) & (x <= x1)] + b
    return y


def saturated_linear_smooth(x, k, b, x0, x1, c0, c1, width=0.01):

    n0, n1, m0, m1 = get_saturated_pars(k, b, x0, x1, c0, c1)
    w_low = sigmoid(x, x0, -1 / width)
    y_low = m0 * np.exp(n0 * x) + c0
    w_high = sigmoid(x, x1, 1 / width)
    y_high = m1 * np.exp(n1 * x) + c1
    y_linear = k * x + b
    y_smooth = w_low * y_low + w_high * y_high + (1 - w_low -
                                                  w_high) * y_linear
    return y_smooth


def generate_piecewise_linear(n):

    def piecewise_linear(x, b0, x0, k_lst, dx_lst, width=0.01):

        y_lst = np.empty((n, *x.shape))
        w_lst = np.empty((n, *x.shape))
        y_lst[0] = k_lst[0] * x + b0
        w_lst[0] = sigmoid(x, x0, -1 / width)
        last_left_w = w_lst[0]
        this_node = x0
        this_b = b0
        for i in range(1, n):
            this_b = k_lst[i - 1] * this_node + this_b - k_lst[i] * this_node
            y_lst[i] = k_lst[i] * x + this_b
            if i == n - 1:
                w_lst[i] = 1 - last_left_w
            else:
                this_node += dx_lst[i - 1]
                this_left_w = sigmoid(x, this_node, -1 / width)
                w_lst[i] = (1 - last_left_w) * this_left_w
                last_left_w = this_left_w

        y = np.sum(w_lst * y_lst, axis=0)

        return y

    return piecewise_linear
