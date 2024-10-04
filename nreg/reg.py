from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import grad, jacrev, jit
from jax.scipy.stats import gaussian_kde
from scipy.spatial import KDTree

from .tools import get_ext_w_2d, get_p_flow
from .x_xerr import gen_xobs_trans
from .transfer import Transfer, neglogplus


def construct_trainer(log_likelihood, step):

    def trainer(params,
                x_ext,
                y_ext,
                w,
                lr=1e-3,
                n_step=10000,
                atol=1e-3,
                rtol=1e-3,
                verbose=False):
        params_old = params

        for i in range(n_step):
            params = step(params, x_ext, y_ext, w, lr)

            if (i % 1000 == 0):
                if verbose:
                    print(i, *params, log_likelihood(params, x_ext, y_ext, w))
                if (jnp.allclose(jnp.array(params),
                                 jnp.array(params_old),
                                 atol=atol,
                                 rtol=rtol)) and (i > 0):
                    if verbose:
                        print(f'converged, so stop with {i} iterations')
                    break
                else:
                    params_old = params
        return params

    return trainer


def construct_log_likelihood(likelihood):

    @jit
    def log_likelihood(params, x_ext, y_ext, w):
        return jnp.mean(jnp.log(likelihood(params, x_ext, y_ext, w) + 1e-10))

    d_log_likelihood = jit(grad(log_likelihood, argnums=(0, )))
    dd_log_likelihood = jit(
        jacrev(grad(log_likelihood, argnums=(0, )), argnums=(0, )))

    return log_likelihood, d_log_likelihood, dd_log_likelihood


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


def loglinear_x2y(x_gen,
                  x_obs_gen,
                  y_gen_kdt,
                  y_err_gen_kdt,
                  k,
                  b,
                  sig,
                  return_more=False):
    tree = KDTree(y_gen_kdt.reshape(-1, 1))

    log10_y_gen = k * np.log10(x_gen) + b + np.random.normal(scale=sig,
                                                             size=x_gen.shape)

    # avoid inf
    y_gen = np.clip(np.power(10, log10_y_gen), 0, y_gen_kdt.max())

    _, y_err_match_index = tree.query(y_gen.reshape(-1, 1))
    y_err_gen = y_err_gen_kdt[y_err_match_index]
    y_obs_gen = y_gen + np.random.normal(scale=y_err_gen)

    if return_more:
        return x_obs_gen, y_obs_gen, y_err_gen, y_gen
    else:
        return x_obs_gen, y_obs_gen


def gen_xy_obs_trans_prepare(flowx,
                             flowy,
                             x_range,
                             x_err_range,
                             y_range,
                             y_err_range,
                             n=1000,
                             x_transfer: Union[Transfer, None] = None,
                             x_transfer_params=None,
                             y_transfer: Union[Transfer, None] = None,
                             y_transfer_params=None,
                             return_numpy=False):
    x_min, x_max = x_range
    x_err_min, x_err_max = x_err_range

    x_gen, x_err_gen, x_obs_gen = gen_xobs_trans(
        flowx,
        n,
        x_min,
        x_max,
        x_err_min,
        x_err_max,
        transfer=x_transfer,
        transfer_params=x_transfer_params)

    y_min, y_max = y_range
    y_err_min, y_err_max = y_err_range

    y_gen_kdt, y_err_gen_kdt, _ = gen_xobs_trans(
        flowy,
        n,
        y_min,
        y_max,
        y_err_min,
        y_err_max,
        transfer=y_transfer,
        transfer_params=y_transfer_params)

    if return_numpy:
        x_gen = np.array(x_gen)
        x_obs_gen = np.array(x_obs_gen)
        y_gen_kdt = np.array(y_gen_kdt)
        y_err_gen_kdt = np.array(y_err_gen_kdt)

    return x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt


def gen_xy_obs_trans_loglinear(flowx,
                               flowy,
                               x_range,
                               x_err_range,
                               y_range,
                               y_err_range,
                               k,
                               b,
                               sig,
                               n=1000,
                               x_transfer: Union[Transfer, None] = None,
                               x_transfer_params=None,
                               y_transfer: Union[Transfer, None] = None,
                               y_transfer_params=None,
                               return_numpy=False,
                               return_more=False):

    x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt = gen_xy_obs_trans_prepare(
        flowx,
        flowy,
        x_range,
        x_err_range,
        y_range,
        y_err_range,
        n=n,
        x_transfer=x_transfer,
        x_transfer_params=x_transfer_params,
        y_transfer=y_transfer,
        y_transfer_params=y_transfer_params,
        return_numpy=return_numpy)

    if not return_more:
        return loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt, k, b,
                             sig)

    x_obs_gen, y_obs_gen, y_err_gen, y_gen = loglinear_x2y(x_gen,
                                                           x_obs_gen,
                                                           y_gen_kdt,
                                                           y_err_gen_kdt,
                                                           k,
                                                           b,
                                                           sig,
                                                           return_more=True)
    return x_obs_gen, y_obs_gen, x_gen, y_gen, x_err_gen, y_err_gen


def gen_xy_obs_trans_neglogplus_loglinear(flowx,
                                          flowy,
                                          x_range,
                                          x_err_range,
                                          y_range,
                                          y_err_range,
                                          k,
                                          b,
                                          sig,
                                          x_transfer_params,
                                          y_transfer_params,
                                          n=1000,
                                          return_numpy=False):
    return gen_xy_obs_trans_loglinear(flowx,
                                      flowy,
                                      x_range,
                                      x_err_range,
                                      y_range,
                                      y_err_range,
                                      k,
                                      b,
                                      sig,
                                      n=n,
                                      x_transfer=neglogplus,
                                      x_transfer_params=x_transfer_params,
                                      y_transfer=neglogplus,
                                      y_transfer_params=y_transfer_params,
                                      return_numpy=return_numpy)
