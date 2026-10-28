from jax import jit
from jax.nn import sigmoid as _sigmoid
import jax.numpy as jnp


def sigmoid(x, x0, beta):
    return _sigmoid(beta * (x - x0))


@jit
def saturated_linear_smooth_jax(x, k, b, x0, x1, c0, c1, width=0.01):

    def get_pars(k, b, x0, x1, c0, c1):
        n0 = k / (k * x0 + b - c0)
        n1 = k / (k * x1 + b - c1)
        m0 = k / (n0 * jnp.exp(n0 * x0))
        m1 = k / (n1 * jnp.exp(n1 * x1))
        return n0, n1, m0, m1

    n0, n1, m0, m1 = get_pars(k, b, x0, x1, c0, c1)
    w_low = sigmoid(x, x0, -1 / width)
    y_low = m0 * jnp.exp(n0 * x) + c0
    w_high = sigmoid(x, x1, 1 / width)
    y_high = m1 * jnp.exp(n1 * x) + c1
    y_linear = k * x + b
    y_smooth = w_low * y_low + w_high * y_high + (1 - w_low -
                                                  w_high) * y_linear
    return y_smooth


@jit
def loglinear_scatter_jax(x, params, eps):
    """Jax twin of ``raddest.reg.ks.loglinear_scatter`` with the intrinsic
    scatter injected as a fixed unit-normal draw ``eps`` (reparameterization).

    ``params = (k, b, sig)``:
    ``y = 10^(k * log10(x) + b + sig * eps)``.
    """
    return jnp.power(10, params[0] * jnp.log10(jnp.abs(x)) + params[1] +
                     params[2] * eps)


def generate_piecewise_linear_jax(n):

    @jit
    def piecewise_linear(x, b0, x0, k_lst, dx_lst, width=0.01):

        y_lst = jnp.empty((n, *x.shape))
        w_lst = jnp.empty((n, *x.shape))
        y_lst = y_lst.at[0].set(k_lst[0] * x + b0)
        w_lst = w_lst.at[0].set(sigmoid(x, x0, -1 / width))
        last_left_w = w_lst[0]
        this_node = x0
        this_b = b0
        for i in range(1, n):
            this_b = k_lst[i - 1] * this_node + this_b - k_lst[i] * this_node
            y_lst = y_lst.at[i].set(k_lst[i] * x + this_b)
            if i == n - 1:
                w_lst = w_lst.at[i].set(1 - last_left_w)
            else:
                this_node += dx_lst[i - 1]
                this_left_w = sigmoid(x, this_node, -1 / width)
                w_lst = w_lst.at[i].set((1 - last_left_w) * this_left_w)
                last_left_w = this_left_w

        y = (w_lst * y_lst).sum(axis=0)

        # return y, w_lst, y_lst
        return y

    return piecewise_linear
