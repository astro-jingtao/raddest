from typing import Union

import numpy as np
from ndtest.ksnd import ks2d2s

from ..transfer import Transfer, NegLogPlus
from ..x_xerr import gen_xobs_trans
from .constructor import construct_generator_cond


def gen_xy_obs_prepare(flowx,
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


def loglinear_scatter(x, params):
    k, b, sig = params
    return np.power(
        10,
        k * np.log10(x) + b + np.random.normal(scale=sig, size=x.shape))

loglinear_generator_cond = construct_generator_cond(loglinear_scatter)

def linear_scatter(x, params):
    k, b, sig = params
    return k * x + b + np.random.normal(scale=sig, size=x.shape)

linear_generator_cond = construct_generator_cond(linear_scatter)


def loglinear_x2y(x, x_obs, y_kdt, y_err_kdt, k, b, sig, return_more=False):

    y_pred, y_err_pred, y_obs_pred = loglinear_generator_cond(
        x, y_kdt, y_err_kdt, (k, b, sig))

    if return_more:
        return x_obs, y_obs_pred, y_err_pred, y_pred
    else:
        return x_obs, y_obs_pred


def gen_xy_obs_loglinear(flowx,
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

    x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt = gen_xy_obs_prepare(
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


def gen_xy_obs_loglinear_neglogplus(flowx,
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
    return gen_xy_obs_loglinear(flowx,
                                flowy,
                                x_range,
                                x_err_range,
                                y_range,
                                y_err_range,
                                k,
                                b,
                                sig,
                                n=n,
                                x_transfer=NegLogPlus(),
                                x_transfer_params=x_transfer_params,
                                y_transfer=NegLogPlus(),
                                y_transfer_params=y_transfer_params,
                                return_numpy=return_numpy)


def get_ks_reg_loglinear(x_gen,
                         x_obs_gen,
                         y_kdt,
                         y_err_kdt,
                         params,
                         x_obs,
                         y_obs,
                         nboot=None,
                         n_jobs=1):

    _, _, y_obs_gen = loglinear_generator_cond(x_gen, y_kdt, y_err_kdt, params)

    return ks2d2s(x_obs_gen,
                  y_obs_gen,
                  x_obs,
                  y_obs,
                  nboot=nboot,
                  n_jobs=n_jobs)


def compare_loglinear(k,
                      b,
                      sig,
                      x_gen,
                      x_obs_gen,
                      y_gen_kdt,
                      y_err_gen_kdt,
                      x_obs,
                      y_obs,
                      nboot=None,
                      n_jobs=1):
    return get_ks_reg_loglinear(x_gen,
                                x_obs_gen,
                                y_gen_kdt,
                                y_err_gen_kdt, (k, b, sig),
                                x_obs,
                                y_obs,
                                nboot=nboot,
                                n_jobs=n_jobs)


def parser_anylinear(res, iargs):
    res_arr = np.zeros((len(res), 5))
    for i, (r, a) in enumerate(zip(res, iargs)):
        res_arr[i, :] = (*a, r[0], r[1])
    return res_arr


def print_mean_std_loglinear(_mean, _std):
    # print res with std
    print('----')
    print(f'k: {_mean[0]:.3f} +/- {_std[0]:.3f}')
    print(f'b: {_mean[1]:.3f} +/- {_std[1]:.3f}')
    print(f'sig: {_mean[2]:.3f} +/- {_std[2]:.3f}')
