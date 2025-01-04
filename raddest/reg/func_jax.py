from jax import jit
from jax.nn import sigmoid as _sigmoid
import jax.numpy as jnp


@jit
def saturated_linear_smooth_jax(x, k, b, x0, x1, c0, c1, width=0.01):

    def get_pars(k, b, x0, x1, c0, c1):
        n0 = k / (k * x0 + b - c0)
        n1 = k / (k * x1 + b - c1)
        m0 = k / (n0 * jnp.exp(n0 * x0))
        m1 = k / (n1 * jnp.exp(n1 * x1))
        return n0, n1, m0, m1

    def sigmoid(x, x0, beta):
        return _sigmoid(beta * (x - x0))

    n0, n1, m0, m1 = get_pars(k, b, x0, x1, c0, c1)
    w_low = sigmoid(x, x0, -1 / width)
    y_low = m0 * jnp.exp(n0 * x) + c0
    w_high = sigmoid(x, x1, 1 / width)
    y_high = m1 * jnp.exp(n1 * x) + c1
    y_linear = k * x + b
    y_smooth = w_low * y_low + w_high * y_high + (1 - w_low -
                                                  w_high) * y_linear
    return y_smooth
