import jax.numpy as jnp
import numpy as np
from jax import grad, jacrev, jit
from jax.scipy.stats import gaussian_kde
from scipy.spatial import KDTree

from .GaussHermiteQuadrature import GaussHermiteQuadrature
from .tools import (get_ext_w_2d, get_p_flow, get_x_err_ext, neg_log_plus,
                    trans_neglogplus, trans_neglogplus_dx,
                    trans_neglogplus_inv, trans_neglogplus_jac)
from .x_xerr import gen_xobs_trans


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
                          x_trans_func=None,
                          x_trans_func_jac=None,
                          x_trans_params=None,
                          y_trans_func=None,
                          y_trans_func_jac=None,
                          y_trans_func_inv=None,
                          y_trans_params=None,
                          n=21):
    '''
    Here x_range, x_err_range, y_range, y_err_range are for original varibles, not transformed ones.
    '''

    # x, x_err | y, y_err <-> u, v

    _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d(
        x_obs, y_obs, x_err, y_err, n)

    if x_trans_func is None:
        ux_ext = _x_ext
        vx_ext = _x_err_ext
    else:
        ux_ext, vx_ext = x_trans_func(_x_ext,
                                      _x_err_ext,
                                      params=x_trans_params)

    if y_trans_func is None:
        uy_ext = _y_ext
        vy_ext = _y_err_ext
    else:
        uy_ext, vy_ext = y_trans_func(_y_ext,
                                      _y_err_ext,
                                      params=y_trans_params)

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
    if x_trans_func is None:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx, ux_ext.flatten(),
                       vx_ext.flatten()).reshape(s1, s2), 0)
    else:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx, ux_ext.flatten(), vx_ext.flatten()).reshape(
                s1, s2) *
            x_trans_func_jac(_x_ext, _x_err_ext, params=x_trans_params), 0)

    # wy: w2

    samples = flowy.sample(20000, seed=0)
    uy_sample = samples['x_obs'].to_numpy()
    vy_sample = samples['x_err'].to_numpy()
    y_sample, y_err_sample = y_trans_func_inv(uy_sample,
                                              vy_sample,
                                              params=y_trans_params)
    kernel_y = gaussian_kde(y_sample)
    py = kernel_y.pdf(_y_ext.flatten()).reshape(s1, s2)

    if y_trans_func is None:
        py_yerr = get_p_flow(flowy, uy_ext.flatten(),
                             vy_ext.flatten()).reshape(s1, s2)
    else:
        py_yerr = get_p_flow(flowy,
                             uy_ext.flatten(), vy_ext.flatten()).reshape(
                                 s1, s2) * y_trans_func_jac(
                                     _y_ext, _y_err_ext, params=y_trans_params)

    eps = 1e-10
    wy = jnp.where(is_in_range, py_yerr / (py + eps), 0)
    wy = jnp.nan_to_num(wy, nan=0)

    w = _w.reshape(1, -1) * px_xerr / _y_ext / np.log(10) * wy
    w = jnp.where(is_in_range, w, 0)

    return jnp.array(_x_ext), jnp.array(_y_ext), jnp.array(w)


def get_xyw_trans_lognorm_old(x_obs,
                              y_obs,
                              x_err,
                              y_err,
                              flowx,
                              flowy,
                              x_range,
                              x_err_range,
                              y_range,
                              y_err_range,
                              x_trans_func=None,
                              x_trans_func_jac=None,
                              x_trans_params=None,
                              y_trans_func=None,
                              y_trans_func_dx=None,
                              y_trans_params=None,
                              n=21):
    '''
    Here x_range, x_err_range, y_range, y_err_range are for original varibles, not transformed ones.
    '''

    # x, x_err | y, y_err <-> u, v

    _x_ext, _y_ext, _x_err_ext, _y_err_ext, _w = get_ext_w_2d(
        x_obs, y_obs, x_err, y_err, n)

    if x_trans_func is None:
        x_ext = _x_ext
        x_err_ext = _x_err_ext
    else:
        x_ext, x_err_ext = x_trans_func(_x_ext,
                                        _x_err_ext,
                                        params=x_trans_params)

    if y_trans_func is None:
        y_ext = _y_ext
        y_err_ext = _y_err_ext
    else:
        y_ext, y_err_ext = y_trans_func(_y_ext,
                                        _y_err_ext,
                                        params=y_trans_params)

    x_min, x_max = x_range
    x_err_min, x_err_max = x_err_range
    y_min, y_max = y_range
    y_err_min, y_err_max = y_err_range

    is_in_range_x = (_x_ext > x_min) & (_x_ext < x_max) & (
        _x_err_ext > x_err_min) & (_x_err_ext < x_err_max)
    is_in_range_y = (_y_ext > y_min) & (_y_ext < y_max) & (
        _y_err_ext > y_err_min) & (_y_err_ext < y_err_max)
    is_in_range = is_in_range_x & is_in_range_y

    s1, s2 = x_ext.shape
    # px_xerr : w1
    if x_trans_func is None:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx, x_ext.flatten(),
                       x_err_ext.flatten()).reshape(s1, s2), 0)
    else:
        px_xerr = jnp.where(
            is_in_range,
            get_p_flow(flowx, x_ext.flatten(), x_err_ext.flatten()).reshape(
                s1, s2) *
            x_trans_func_jac(_x_ext, _x_err_ext, params=x_trans_params), 0)

    # wy: w2
    # TODO: just use p(y, y_err)/p(y)
    y_sample = flowy.sample(10000)['x_obs'].to_numpy()
    kernel_y = gaussian_kde(y_sample)

    py = kernel_y.pdf(y_ext.flatten()).reshape(s1, s2)
    py_yerr = get_p_flow(flowy, y_ext.flatten(),
                         y_err_ext.flatten()).reshape(s1, s2)

    eps = 1e-10
    if y_trans_func is None:
        wy = jnp.where(is_in_range, py_yerr / (py + eps), 0)
    else:
        # p(y_err|y) = p(v|u) * |d/dy|
        wy = jnp.where(
            is_in_range, py_yerr / (py + eps) *
            y_trans_func_dx(_y_ext, _y_err_ext, params=y_trans_params), 0)
        # wy = jnp.where(
        #     is_in_range, py_yerr / (py + eps) *
        #     y_trans_func_dx(_y_err_ext, _y_ext, params=y_trans_params[::-1]), 0)
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
                                     x_trans_params,
                                     y_trans_params,
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
                                 x_trans_func=trans_neglogplus,
                                 x_trans_func_jac=trans_neglogplus_jac,
                                 x_trans_params=x_trans_params,
                                 y_trans_func=trans_neglogplus,
                                 y_trans_func_dx=trans_neglogplus_dx,
                                 y_trans_params=y_trans_params,
                                 n=n)


def loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt, k, b, sig):
    tree = KDTree(y_gen_kdt.reshape(-1, 1))

    log10_y_gen = k * np.log10(x_gen) + b + np.random.normal(scale=sig,
                                                             size=x_gen.shape)

    # avoid inf
    y_gen = np.clip(np.power(10, log10_y_gen), 0, y_gen_kdt.max())

    _, y_err_match_index = tree.query(y_gen.reshape(-1, 1))
    y_err_gen = y_err_gen_kdt[y_err_match_index]
    y_obs_gen = y_gen + np.random.normal(scale=y_err_gen)

    return x_obs_gen, y_obs_gen


def gen_xy_obs_trans_prepare(flowx,
                             flowy,
                             x_range,
                             x_err_range,
                             y_range,
                             y_err_range,
                             n=1000,
                             x_trans_func_inv=None,
                             x_trans_params=None,
                             y_trans_func_inv=None,
                             y_trans_params=None,
                             return_numpy=False):
    x_min, x_max = x_range
    x_err_min, x_err_max = x_err_range

    x_gen, _, x_obs_gen = gen_xobs_trans(flowx,
                                         n,
                                         x_min,
                                         x_max,
                                         x_err_min,
                                         x_err_max,
                                         trans_func_inv=x_trans_func_inv,
                                         trans_params=x_trans_params)

    y_min, y_max = y_range
    y_err_min, y_err_max = y_err_range

    y_gen_kdt, y_err_gen_kdt, _ = gen_xobs_trans(
        flowy,
        n,
        y_min,
        y_max,
        y_err_min,
        y_err_max,
        trans_func_inv=y_trans_func_inv,
        trans_params=y_trans_params)

    if return_numpy:
        x_gen = np.array(x_gen)
        x_obs_gen = np.array(x_obs_gen)
        y_gen_kdt = np.array(y_gen_kdt)
        y_err_gen_kdt = np.array(y_err_gen_kdt)

    return x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt


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
                               x_trans_func_inv=None,
                               x_trans_params=None,
                               y_trans_func_inv=None,
                               y_trans_params=None,
                               return_numpy=False):

    x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt = gen_xy_obs_trans_prepare(
        flowx,
        flowy,
        x_range,
        x_err_range,
        y_range,
        y_err_range,
        n=n,
        x_trans_func_inv=x_trans_func_inv,
        x_trans_params=x_trans_params,
        y_trans_func_inv=y_trans_func_inv,
        y_trans_params=y_trans_params,
        return_numpy=return_numpy)

    return loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt, k, b, sig)


def gen_xy_obs_trans_neglogplus_loglinear(flowx,
                                          flowy,
                                          x_range,
                                          x_err_range,
                                          y_range,
                                          y_err_range,
                                          k,
                                          b,
                                          sig,
                                          x_trans_params,
                                          y_trans_params,
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
                                      x_trans_func_inv=trans_neglogplus_inv,
                                      x_trans_params=x_trans_params,
                                      y_trans_func_inv=trans_neglogplus_inv,
                                      y_trans_params=y_trans_params,
                                      return_numpy=return_numpy)


# —————————————————————— old code ——————————————————————

# def get_xyw_lognorm(x_obs,
#                     y_obs,
#                     x_err,
#                     y_err,
#                     flowx,
#                     flowy,
#                     x_median,
#                     x_err_median,
#                     y_median,
#                     y_err_median,
#                     x_trans=False,
#                     y_trans=False,
#                     n=21,
#                     x_shift=0,
#                     y_shift=0):
#     """
#     w = w_ghq * p(x, x_err) * p(y_err|y) / y / ln(10)
#     x = x_nf + x_shift
#     y = y_nf + y_shift
#     """
#     ghq = GaussHermiteQuadrature(dim=2, nodes=n)
#     x = ghq.X[:, 0].reshape(1, -1)
#     y = ghq.X[:, 1].reshape(1, -1)

#     _x_ext = x_obs.reshape(-1, 1) + np.sqrt(2) * x_err.reshape(-1, 1) * x
#     _y_ext = y_obs.reshape(-1, 1) + np.sqrt(2) * y_err.reshape(-1, 1) * y

#     _x_err_ext = get_x_err_ext(x_err, n * n)
#     _y_err_ext = get_x_err_ext(y_err, n * n)

#     if x_trans:
#         x_ext = neg_log_plus((_x_ext - x_shift) / x_median)
#         x_err_ext = neg_log_plus(_x_err_ext / x_err_median)
#     else:
#         x_ext = _x_ext
#         x_err_ext = _x_err_ext

#     if y_trans:
#         y_ext = neg_log_plus((_y_ext - y_shift) / y_median)
#         y_err_ext = neg_log_plus(_y_err_ext / y_err_median)
#     else:
#         y_ext = _y_ext
#         y_err_ext = _y_err_ext

#     s1, s2 = x_ext.shape
#     if x_trans:
#         px_xerr = jnp.where(
#             (_x_ext > 0) & (_y_ext > 0),
#             get_p_flow(flowx, x_ext.flatten(), x_err_ext.flatten()).reshape(
#                 s1, s2) / (_x_ext + x_median) / (_x_err_ext + x_err_median), 0)
#     else:
#         px_xerr = jnp.where((_x_ext > 0) & (_y_ext > 0),
#                             get_p_flow(flowx, x_ext.flatten(),
#                                        x_err_ext.flatten()).reshape(s1, s2), 0)

#     y_sample = flowy.sample(10000)['x_obs'].to_numpy()
#     kernel_y = gaussian_kde(y_sample)

#     py_yerr = jnp.where((_x_ext > 0) & (_y_ext > 0),
#                         get_p_flow(flowy, y_ext.flatten(),
#                                    y_err_ext.flatten()).reshape(s1, s2), 0)
#     py = kernel_y.pdf(y_ext.flatten()).reshape(s1, s2)

#     eps = 1e-10
#     if y_trans:
#         wy = jnp.where((_x_ext > 0) & (_y_ext > 0),
#                        py_yerr / (py + eps) / (_y_err_ext + y_err_median), 0)
#     else:
#         wy = jnp.where((_x_ext > 0) & (_y_ext > 0), py_yerr / (py + eps), 0)
#     # replace nan with 0
#     wy = jnp.nan_to_num(wy, nan=0)

#     # 应该有个 1/pi 的系数，但对结果没有影响
#     w = ghq.W.reshape(1, -1) * px_xerr / _y_ext / np.log(10) * wy

#     w = jnp.where((_x_ext < x_obs.max()) & (_y_ext < y_obs.max()), w, 0)

#     return jnp.array(_x_ext), jnp.array(_y_ext), jnp.array(w)

# def gen_xy_obs(flowx,
#                x_median,
#                x_err_median,
#                x_max,
#                xerr_max,
#                flowy,
#                y_median,
#                y_err_median,
#                y_max,
#                yerr_max,
#                k,
#                b,
#                sig,
#                n=1000,
#                x_trans=False,
#                y_trans=False,
#                x_shift=0,
#                y_shift=0):

#     if x_trans:
#         x_gen, x_err_gen, x_obs_gen = gen_xobs_trans(flowx,
#                                                      n,
#                                                      x_median,
#                                                      x_err_median,
#                                                      x_min=0,
#                                                      x_max=x_max,
#                                                      xerr_min=0,
#                                                      xerr_max=xerr_max,
#                                                      x_shift=x_shift)
#     else:
#         x_gen, x_err_gen, x_obs_gen = gen_xobs(flowx, n)

#     if y_trans:
#         y_gen_kdt, y_err_gen_kdt, _ = gen_xobs_trans(flowy,
#                                                      n,
#                                                      y_median,
#                                                      y_err_median,
#                                                      x_min=0,
#                                                      x_max=y_max,
#                                                      xerr_min=0,
#                                                      xerr_max=yerr_max,
#                                                      x_shift=y_shift)
#     else:
#         y_gen_kdt, y_err_gen_kdt, _ = gen_xobs(flowy, n)

#     tree = KDTree(y_gen_kdt.reshape(-1, 1))

#     log10_y_gen = k * np.log10(x_gen) + b + np.random.normal(scale=sig, size=n)
#     y_gen = np.power(10, log10_y_gen)

#     _, y_err_match_index = tree.query(y_gen.reshape(-1, 1))
#     y_err_gen = y_err_gen_kdt[y_err_match_index]
#     y_obs_gen = y_gen + np.random.normal(scale=y_err_gen)

#     return x_obs_gen, y_obs_gen
