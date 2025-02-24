import numpy as np
from scipy.special import expit as _sigmoid

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

    def sigmoid(x, x0, beta):
        return _sigmoid(beta * (x - x0))

    n0, n1, m0, m1 = get_saturated_pars(k, b, x0, x1, c0, c1)
    w_low = sigmoid(x, x0, -1 / width)
    y_low = m0 * np.exp(n0 * x) + c0
    w_high = sigmoid(x, x1, 1 / width)
    y_high = m1 * np.exp(n1 * x) + c1
    y_linear = k * x + b
    y_smooth = w_low * y_low + w_high * y_high + (1 - w_low -
                                                  w_high) * y_linear
    return y_smooth
