"""Grid integration of the (obs, err) space of both axes, with optional
variable transforms, producing integration points and weights."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import numpy as np
from numpy.typing import ArrayLike

from ..tools import (
    concat,
    get_ext_w_2d,
    get_ext_w_2d_grid,
    get_ext_w_2d_inter_grid,
    get_ext_w_2d_qgrid,
    get_ext_w_2d_qmc,
    get_p_flow,
)
from ..transfer import Transfer

if TYPE_CHECKING:
    from pzflow import Flow

_SIMPLE_INT_METHODS: dict[str, Callable] = {
    'ghq': get_ext_w_2d,
    'qmc': get_ext_w_2d_qmc,
    'grid': get_ext_w_2d_grid,
    'qgrid': get_ext_w_2d_qgrid,
    'inter_grid': get_ext_w_2d_inter_grid,
}

# Hybrid methods: ghq for well-behaved points, the mapped integrator for
# points whose errors are large relative to the data spread.
_HYBRID_INT_METHODS: dict[str, Callable] = {
    'ghq+qgrid': get_ext_w_2d_qgrid,
    'ghq+grid': get_ext_w_2d_grid,
}


def _resolve_int_method(
    int_method: str,
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
) -> str:
    """Resolve ``'auto'`` to ``'ghq'`` or ``'qgrid'``; pass other methods through.

    The ratio compares the typical error (twice the median) with the data
    spread (84th minus 16th percentile); errors wider than half the spread
    need the quantile-grid integrator instead of sparse Gauss-Hermite nodes.
    """
    if int_method != 'auto':
        return int_method
    x_err_ratio = 2 * jnp.median(x_err) / (jnp.percentile(x_obs, 84) -
                                           jnp.percentile(x_obs, 16))
    y_err_ratio = 2 * jnp.median(y_err) / (jnp.percentile(y_obs, 84) -
                                           jnp.percentile(y_obs, 16))
    if (x_err_ratio > 0.5) or (y_err_ratio > 0.5):
        return 'qgrid'
    return 'ghq'


def _get_ext_w_hybrid(
    second_fn: Callable,
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    *,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-point hybrid integration: ghq nodes where errors are small,
    ``second_fn`` nodes where an error ratio exceeds 0.5, ghq block first."""
    x_err_ratio = 2 * x_err / (jnp.percentile(x_obs, 84) -
                               jnp.percentile(x_obs, 16))
    y_err_ratio = 2 * y_err / (jnp.percentile(y_obs, 84) -
                               jnp.percentile(y_obs, 16))
    to_use_second = (x_err_ratio > 0.5) | (y_err_ratio > 0.5)

    x_ext_ghq, y_ext_ghq, x_err_ext_ghq, y_err_ext_ghq, w_ghq = get_ext_w_2d(
        x_obs[~to_use_second], y_obs[~to_use_second], x_err[~to_use_second],
        y_err[~to_use_second], n)
    w_ghq = jnp.repeat(w_ghq[None, :], (~to_use_second).sum(), axis=0)

    x_ext_second, y_ext_second, x_err_ext_second, y_err_ext_second, w_second \
        = second_fn(x_obs[to_use_second], y_obs[to_use_second],
                    x_err[to_use_second], y_err[to_use_second], n)

    return (concat(x_ext_ghq, x_ext_second, axis=0),
            concat(y_ext_ghq, y_ext_second, axis=0),
            concat(x_err_ext_ghq, x_err_ext_second, axis=0),
            concat(y_err_ext_ghq, y_err_ext_second, axis=0),
            concat(w_ghq, w_second, axis=0))


def _get_ext_w(
    int_method: str,
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    *,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Dispatch the 2D integration method; raise ``ValueError`` when unknown."""
    fn = _SIMPLE_INT_METHODS.get(int_method)
    if fn is not None:
        return fn(x_obs, y_obs, x_err, y_err, n)
    fn = _HYBRID_INT_METHODS.get(int_method)
    if fn is not None:
        return _get_ext_w_hybrid(fn, x_obs, y_obs, x_err, y_err, n=n)
    raise ValueError(f"Invalid integration method: {int_method}")


def _apply_transfer(
    transfer: Transfer | None,
    params: ArrayLike | None,
    ext: ArrayLike,
    err_ext: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Apply ``transfer.trans`` when a transfer is given; pass through otherwise."""
    if transfer is None:
        return ext, err_ext
    return transfer.trans(ext, err_ext, params=params)


def _flow_p(
    flow: Flow,
    ux: ArrayLike,
    vx: ArrayLike,
    shape: tuple[int, int],
    batch_size: int | None,
) -> jax.Array:
    """Flow density ``p(u | v)`` on the flattened grid, reshaped to ``shape``."""
    p = get_p_flow(flow, ux.flatten(), vx.flatten(), batch_size=batch_size)
    return p.reshape(shape)


def get_xyw_trans(x_obs: ArrayLike,
                  y_obs: ArrayLike,
                  x_err: ArrayLike,
                  y_err: ArrayLike,
                  *,
                  flowx: Flow,
                  flowy: Flow,
                  x_range: tuple[float, float],
                  x_err_range: tuple[float, float],
                  y_range: tuple[float, float],
                  y_err_range: tuple[float, float],
                  int_method: str = 'ghq',
                  ignore_y_err_c_y: bool = False,
                  flow_batch_size: int | None = None,
                  kde_batch_size: int | None = None,
                  get_w_add: Callable[
                      [ArrayLike, ArrayLike, ArrayLike, ArrayLike], ArrayLike
                  ] | None = None,
                  x_transfer: Transfer | None = None,
                  x_transfer_params: ArrayLike | None = None,
                  y_transfer: Transfer | None = None,
                  y_transfer_params: ArrayLike | None = None,
                  n: int = 21) -> tuple[jax.Array, jax.Array, jax.Array]:
    '''Grid points and weights for the 2D (obs, err) integration of both axes.

    Here x_range, x_err_range, y_range, y_err_range are for original varibles, not transformed ones.

    get_w_add: callable, (x_ext, y_ext, x_err_ext, y_err_ext) -> w_add, additional weight for each point.
    '''

    # x, x_err | y, y_err <-> u, v

    int_method = _resolve_int_method(int_method, x_obs, y_obs, x_err, y_err)
    _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = _get_ext_w(
        int_method, x_obs, y_obs, x_err, y_err, n=n)

    ux_ext, vx_ext = _apply_transfer(x_transfer, x_transfer_params,
                                     _x_ext, _x_err_ext)
    uy_ext, vy_ext = _apply_transfer(y_transfer, y_transfer_params,
                                     _y_ext, _y_err_ext)

    x_min, x_max = x_range
    x_err_min, x_err_max = x_err_range
    y_min, y_max = y_range
    y_err_min, y_err_max = y_err_range

    is_in_range_x = (_x_ext > x_min) & (_x_ext < x_max) & (
        _x_err_ext > x_err_min) & (_x_err_ext < x_err_max)
    is_in_range_y = (_y_ext > y_min) & (_y_ext < y_max) & (
        _y_err_ext > y_err_min) & (_y_err_ext < y_err_max)
    is_in_range = is_in_range_x & is_in_range_y

    s1, s2 = ux_ext.shape
    # px_xerr : w1
    px_xerr = _flow_p(flowx, ux_ext, vx_ext, (s1, s2), flow_batch_size)
    if x_transfer is not None:
        px_xerr = px_xerr * x_transfer.jac(_x_ext,
                                           _x_err_ext,
                                           params=x_transfer_params)
    px_xerr = jnp.where(is_in_range, px_xerr, 0)

    # wy: w2

    samples = flowy.sample(20000, seed=0)
    uy_sample = samples['x_obs'].to_numpy()
    vy_sample = samples['x_err'].to_numpy()
    if y_transfer is None:
        y_sample = uy_sample
    else:
        y_sample, _ = y_transfer.inv(uy_sample,
                                     vy_sample,
                                     params=y_transfer_params)

    if ignore_y_err_c_y:
        wy = jnp.ones_like(uy_ext)
    else:

        kernel_y = gaussian_kde(y_sample)

        if kde_batch_size is None:
            py = kernel_y.pdf(_y_ext.flatten()).reshape(s1, s2)
        else:
            py = kde_pdf_batch(kernel_y,
                               _y_ext.flatten(),
                               batch_size=kde_batch_size).reshape(s1, s2)

        py_yerr = _flow_p(flowy, uy_ext, vy_ext, (s1, s2), flow_batch_size)
        if y_transfer is not None:
            py_yerr = py_yerr * y_transfer.jac(
                _y_ext, _y_err_ext, params=y_transfer_params)

        eps = 1e-10
        wy = jnp.where(is_in_range, py_yerr / (py + eps), 0)
        wy = jnp.nan_to_num(wy, nan=0)

    if get_w_add is None:
        w_add = 1
    else:
        w_add = get_w_add(_x_ext, _y_ext, _x_err_ext, _y_err_ext)

    if len(_w.shape) == 1:
        _w = _w.reshape(1, -1)

    w = _w * px_xerr * wy * w_add
    w = jnp.where(is_in_range, w, 0)

    return jnp.array(_x_ext), jnp.array(_y_ext), jnp.array(w)


def get_w_add_lognorm(
    x_ext: ArrayLike,
    y_ext: ArrayLike,
    x_err_ext: ArrayLike,
    y_err_ext: ArrayLike,
) -> jax.Array:
    '''Additional weight for log-normal y: 1 / (y ln 10), the log10 Jacobian.'''
    return 1 / y_ext / np.log(10)


def kde_pdf_batch(
    kernel: Callable[[ArrayLike], jax.Array],
    x: ArrayLike,
    batch_size: int = 100000,
) -> jax.Array:
    '''Evaluate a kde kernel on ``x`` in batches, keeping memory bounded.'''
    n = x.shape[0]
    pdf = jnp.zeros(n)

    # Calculate the number of batches
    num_batches = int(jnp.ceil(n / batch_size))  # This ensures we round up

    for i in range(num_batches):
        # print(f"Calculating batch {i+1}/{num_batches}")
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, n)
        pdf_batch = kernel(x[batch_start:batch_end])
        pdf = pdf.at[batch_start:batch_end].set(pdf_batch)

    return pdf
