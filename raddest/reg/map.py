from typing import Union

import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import numpy as np

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


def get_xyw_trans(x_obs,
                  y_obs,
                  x_err,
                  y_err,
                  flowx,
                  flowy,
                  x_range,
                  x_err_range,
                  y_range,
                  y_err_range,
                  int_method='ghq',
                  ignore_y_err_c_y=False,
                  flow_batch_size=None,
                  kde_batch_size=None,
                  get_w_add=None,
                  x_transfer: Union[Transfer, None] = None,
                  x_transfer_params=None,
                  y_transfer: Union[Transfer, None] = None,
                  y_transfer_params=None,
                  n=21):
    '''
    Here x_range, x_err_range, y_range, y_err_range are for original varibles, not transformed ones.

    get_w_add: callable, (x_ext, y_ext, x_err_ext, y_err_ext) -> w_add, additional weight for each point.
    '''

    # x, x_err | y, y_err <-> u, v

    if int_method == 'auto':
        x_err_ratio = 2 * jnp.median(x_err) / (jnp.percentile(x_obs, 84) -
                                               jnp.percentile(x_obs, 16))
        y_err_ratio = 2 * jnp.median(y_err) / (jnp.percentile(y_obs, 84) -
                                               jnp.percentile(y_obs, 16))
        if (x_err_ratio > 0.5) or (y_err_ratio > 0.5):
            int_method = 'qgrid'
            # print(int_method)
        else:
            int_method = 'ghq'
            # print(int_method)

    if int_method == 'ghq':
        _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d(
            x_obs, y_obs, x_err, y_err, n)
    elif int_method == 'qmc':
        _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d_qmc(
            x_obs, y_obs, x_err, y_err, n)
    elif int_method == 'grid':
        _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d_grid(
            x_obs, y_obs, x_err, y_err, n)
    elif int_method == 'qgrid':
        _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d_qgrid(
            x_obs, y_obs, x_err, y_err, n)
    elif int_method == 'inter_grid':
        _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d_inter_grid(
            x_obs, y_obs, x_err, y_err, n)
    elif int_method == 'ghq+qgrid':
        x_err_ratio = 2 * x_err / (jnp.percentile(x_obs, 84) -
                                   jnp.percentile(x_obs, 16))
        y_err_ratio = 2 * y_err / (jnp.percentile(y_obs, 84) -
                                   jnp.percentile(y_obs, 16))
        to_use_qgrid = (x_err_ratio > 0.5) | (y_err_ratio > 0.5)

        # print(to_use_qgrid.sum())

        _x_ext_ghq, _y_ext_ghq, _x_err_ext_ghq, _y_err_ext_ghq, _w_ghq = get_ext_w_2d(
            x_obs[~to_use_qgrid], y_obs[~to_use_qgrid], x_err[~to_use_qgrid],
            y_err[~to_use_qgrid], n)
        _w_ghq = jnp.repeat(_w_ghq[None, :], (~to_use_qgrid).sum(), axis=0)

        _x_ext_qgrid, _y_ext_qgrid, _x_err_ext_qgrid, _y_err_ext_qgrid, _w_qgrid = get_ext_w_2d_qgrid(
            x_obs[to_use_qgrid], y_obs[to_use_qgrid], x_err[to_use_qgrid],
            y_err[to_use_qgrid], n)

        _x_ext = concat(_x_ext_ghq, _x_ext_qgrid, axis=0)
        _y_ext = concat(_y_ext_ghq, _y_ext_qgrid, axis=0)
        _x_err_ext = concat(_x_err_ext_ghq, _x_err_ext_qgrid, axis=0)
        _y_err_ext = concat(_y_err_ext_ghq, _y_err_ext_qgrid, axis=0)
        _w = concat(_w_ghq, _w_qgrid, axis=0)
    elif int_method == 'ghq+grid':
        x_err_ratio = 2 * x_err / (jnp.percentile(x_obs, 84) -
                                   jnp.percentile(x_obs, 16))
        y_err_ratio = 2 * y_err / (jnp.percentile(y_obs, 84) -
                                   jnp.percentile(y_obs, 16))
        to_use_grid = (x_err_ratio > 0.5) | (y_err_ratio > 0.5)

        # print(to_use_grid.sum())

        _x_ext_ghq, _y_ext_ghq, _x_err_ext_ghq, _y_err_ext_ghq, _w_ghq = get_ext_w_2d(
            x_obs[~to_use_grid], y_obs[~to_use_grid], x_err[~to_use_grid],
            y_err[~to_use_grid], n)
        _w_ghq = jnp.repeat(_w_ghq[None, :], (~to_use_grid).sum(), axis=0)

        _x_ext_grid, _y_ext_grid, _x_err_ext_grid, _y_err_ext_grid, _w_grid = get_ext_w_2d_grid(
            x_obs[to_use_grid], y_obs[to_use_grid], x_err[to_use_grid],
            y_err[to_use_grid], n)

        _x_ext = concat(_x_ext_ghq, _x_ext_grid, axis=0)
        _y_ext = concat(_y_ext_ghq, _y_ext_grid, axis=0)
        _x_err_ext = concat(_x_err_ext_ghq, _x_err_ext_grid, axis=0)
        _y_err_ext = concat(_y_err_ext_ghq, _y_err_ext_grid, axis=0)
        _w = concat(_w_ghq, _w_grid, axis=0)
    else:
        raise ValueError(f"Invalid integration method: {int_method}")

    if x_transfer is None:
        ux_ext = _x_ext
        vx_ext = _x_err_ext
    else:
        ux_ext, vx_ext = x_transfer.trans(_x_ext,
                                          _x_err_ext,
                                          params=x_transfer_params)

    if y_transfer is None:
        uy_ext = _y_ext
        vy_ext = _y_err_ext
    else:
        uy_ext, vy_ext = y_transfer.trans(_y_ext,
                                          _y_err_ext,
                                          params=y_transfer_params)

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
    if x_transfer is None:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx,
                       ux_ext.flatten(),
                       vx_ext.flatten(),
                       batch_size=flow_batch_size).reshape(s1, s2), 0)
    else:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx,
                       ux_ext.flatten(),
                       vx_ext.flatten(),
                       batch_size=flow_batch_size).reshape(s1, s2) *
            x_transfer.jac(_x_ext, _x_err_ext, params=x_transfer_params), 0)

    # wy: w2

    samples = flowy.sample(20000, seed=0)
    uy_sample = samples['x_obs'].to_numpy()
    vy_sample = samples['x_err'].to_numpy()
    if y_transfer is None:
        y_sample, y_err_sample = uy_sample, vy_sample
    else:
        y_sample, y_err_sample = y_transfer.inv(uy_sample,
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

        # hist based
        # py_bin, edges = jnp.histogram(y_sample,
        #                       bins=50,
        #                       density=True,
        #                       range=(y_min, y_max))
        # center = (edges[:-1] + edges[1:]) / 2
        # py = jnp.interp(_y_ext.flatten(), center, py_bin).reshape(s1, s2)

        if y_transfer is None:
            py_yerr = get_p_flow(flowy,
                                 uy_ext.flatten(),
                                 vy_ext.flatten(),
                                 batch_size=flow_batch_size).reshape(s1, s2)
        else:
            py_yerr = get_p_flow(
                flowy,
                uy_ext.flatten(),
                vy_ext.flatten(),
                batch_size=flow_batch_size).reshape(s1, s2) * y_transfer.jac(
                    _y_ext, _y_err_ext, params=y_transfer_params)

        eps = 1e-10
        wy = jnp.where(is_in_range, py_yerr / (py + eps), 0)
        wy = jnp.nan_to_num(wy, nan=0)

    if get_w_add is None:
        w_add = 1
    else:
        w_add = get_w_add(_x_ext, _y_ext, _x_err_ext, _y_err_ext)
        # w_add = 1 / _y_ext / np.log(10)

    if len(_w.shape) == 1:
        _w = _w.reshape(1, -1)

    w = _w * px_xerr * wy * w_add
    w = jnp.where(is_in_range, w, 0)

    return jnp.array(_x_ext), jnp.array(_y_ext), jnp.array(w)


def get_w_add_lognorm(x_ext, y_ext, x_err_ext, y_err_ext):
    return 1 / y_ext / np.log(10)


def kde_pdf_batch(kernel, x, batch_size=100000):
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
