"""Jax twins of ``raddest.reg.func``: jit-compiled, differentiable versions of
the saturated and piecewise-linear regression functions."""

from __future__ import annotations

from typing import Callable

import jax
from jax import jit
from jax.nn import sigmoid as _sigmoid
import jax.numpy as jnp
from numpy.typing import ArrayLike


def sigmoid(x: ArrayLike, x0: float, beta: float) -> jax.Array:
    """Jax logistic sigmoid ``1 / (1 + exp(-beta * (x - x0)))``."""
    return _sigmoid(beta * (x - x0))


@jit
def saturated_linear_smooth_jax(
    x: ArrayLike,
    *,
    k: jax.Array,
    b: jax.Array,
    x0: jax.Array,
    x1: jax.Array,
    c0: jax.Array,
    c1: jax.Array,
    width: jax.Array = 0.01,
) -> jax.Array:
    """Jax twin of ``raddest.reg.func.saturated_linear_smooth``: sigmoid-blended
    exponential tails around a linear core, all parameters keyword-only."""

    def get_pars(
        *,
        k: jax.Array,
        b: jax.Array,
        x0: jax.Array,
        x1: jax.Array,
        c0: jax.Array,
        c1: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        n0 = k / (k * x0 + b - c0)
        n1 = k / (k * x1 + b - c1)
        m0 = k / (n0 * jnp.exp(n0 * x0))
        m1 = k / (n1 * jnp.exp(n1 * x1))
        return n0, n1, m0, m1

    n0, n1, m0, m1 = get_pars(k=k, b=b, x0=x0, x1=x1, c0=c0, c1=c1)
    w_low = sigmoid(x, x0, -1 / width)
    y_low = m0 * jnp.exp(n0 * x) + c0
    w_high = sigmoid(x, x1, 1 / width)
    y_high = m1 * jnp.exp(n1 * x) + c1
    y_linear = k * x + b
    y_smooth = w_low * y_low + w_high * y_high + (1 - w_low - w_high) * y_linear
    return y_smooth


@jit
def loglinear_scatter_jax(
    x: ArrayLike,
    params: jax.Array,
    eps: ArrayLike,
) -> jax.Array:
    """Jax twin of ``raddest.reg.mock.loglinear_scatter`` with the intrinsic
    scatter injected as a fixed unit-normal draw ``eps`` (reparameterization).

    ``params = (k, b, sig)``:
    ``y = 10^(k * log10(x) + b + sig * eps)``.
    """
    return jnp.power(10, params[0] * jnp.log10(jnp.abs(x)) + params[1] + params[2] * eps)


def generate_piecewise_linear_jax(n: int) -> Callable:
    """Jax twin of ``raddest.reg.func.generate_piecewise_linear``; the returned
    callable is ``piecewise_linear(x, *, b0, x0, k_lst, dx_lst, width=0.01)``."""

    @jit
    def piecewise_linear(
        x: ArrayLike,
        *,
        b0: jax.Array,
        x0: jax.Array,
        k_lst: jax.Array,
        dx_lst: jax.Array,
        width: jax.Array = 0.01,
    ) -> jax.Array:

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
