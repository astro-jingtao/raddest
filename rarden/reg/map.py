from typing import Union

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import gaussian_kde

from ..tools import get_ext_w_2d, get_p_flow
from ..transfer import Transfer, neglogplus


def get_xyw_trans_lognorm(x_obs,
                          y_obs,
                          x_err,
                          y_err,
                          flowx,
                          flowy,
                          x_range,
                          x_err_range,
                          y_range,
                          y_err_range,
                          x_transfer: Union[Transfer, None] = None,
                          x_transfer_params=None,
                          y_transfer: Union[Transfer, None] = None,
                          y_transfer_params=None,
                          n=21):
    '''
    Here x_range, x_err_range, y_range, y_err_range are for original varibles, not transformed ones.
    '''

    # x, x_err | y, y_err <-> u, v

    _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d(
        x_obs, y_obs, x_err, y_err, n)

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
            get_p_flow(flowx, ux_ext.flatten(),
                       vx_ext.flatten()).reshape(s1, s2), 0)
    else:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx, ux_ext.flatten(), vx_ext.flatten()).reshape(
                s1, s2) *
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
    kernel_y = gaussian_kde(y_sample)
    py = kernel_y.pdf(_y_ext.flatten()).reshape(s1, s2)

    if y_transfer is None:
        py_yerr = get_p_flow(flowy, uy_ext.flatten(),
                             vy_ext.flatten()).reshape(s1, s2)
    else:
        py_yerr = get_p_flow(
            flowy, uy_ext.flatten(), vy_ext.flatten()).reshape(
                s1, s2) * y_transfer.jac(
                    _y_ext, _y_err_ext, params=y_transfer_params)

    eps = 1e-10
    wy = jnp.where(is_in_range, py_yerr / (py + eps), 0)
    wy = jnp.nan_to_num(wy, nan=0)

    w = _w.reshape(1, -1) * px_xerr / _y_ext / np.log(10) * wy
    w = jnp.where(is_in_range, w, 0)

    return jnp.array(_x_ext), jnp.array(_y_ext), jnp.array(w)


def get_xyw_trans_neglogplus_lognorm(x_obs,
                                     y_obs,
                                     x_err,
                                     y_err,
                                     flowx,
                                     flowy,
                                     x_range,
                                     x_err_range,
                                     y_range,
                                     y_err_range,
                                     x_transfer_params,
                                     y_transfer_params,
                                     n=21):
    return get_xyw_trans_lognorm(x_obs,
                                 y_obs,
                                 x_err,
                                 y_err,
                                 flowx,
                                 flowy,
                                 x_range,
                                 x_err_range,
                                 y_range,
                                 y_err_range,
                                 x_transfer=neglogplus,
                                 x_transfer_params=x_transfer_params,
                                 y_transfer=neglogplus,
                                 y_transfer_params=y_transfer_params,
                                 n=n)
