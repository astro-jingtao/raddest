from typing import Union

import numpy as np
from asa.Bcorner import quantile
from ndtest.ksnd import ks2d2s

from ..grid_search import get_weighted_res
from ..transfer import Transfer, neglogplus
from ..x_xerr import gen_xobs_trans
from .constructor import construct_generator_cond


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


def loglinear(x, params):
    k, b, sig = params
    return np.power(
        10,
        k * np.log10(x) + b + np.random.normal(scale=sig, size=x.shape))


loglinear_generator_cond = construct_generator_cond(loglinear)


def loglinear_x2y(x, x_obs, y_kdt, y_err_kdt, k, b, sig, return_more=False):

    y_pred, y_err_pred, y_obs_pred = loglinear_generator_cond(
        x, y_kdt, y_err_kdt, (k, b, sig))

    if return_more:
        return x_obs, y_obs_pred, y_err_pred, y_pred
    else:
        return x_obs, y_obs_pred


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


def get_ks_reg_loglinear(x_gen,
                         x_obs_gen,
                         y_gen_kdt,
                         y_err_gen_kdt,
                         params,
                         x_obs,
                         y_obs,
                         nboot=None,
                         n_jobs=1):
    k = params[0]
    b = params[1]
    sig = params[2]

    x_obs_gen, y_obs_gen = loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt,
                                         y_err_gen_kdt, k, b, sig)

    return ks2d2s(x_obs_gen,
                  y_obs_gen,
                  x_obs,
                  y_obs,
                  nboot=nboot,
                  n_jobs=n_jobs)


def parser_loglinear(res, iargs):
    res_arr = np.zeros((len(res), 5))
    for i, (r, a) in enumerate(zip(res, iargs)):
        res_arr[i, :] = (*a, r[0], r[1])
    return res_arr


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


def gen_new_grid_mean_std(res_arr, n=10):
    res_mean, res_std = get_weighted_res(res_arr[:, :3].T, res_arr[:, 4])
    dk = max(0.2, 2 * res_std[0])
    k_list = np.linspace(res_mean[0] - dk, res_mean[0] + dk, n)
    db = max(0.2, 2 * res_std[1])
    b_list = np.linspace(res_mean[1] - db, res_mean[1] + db, n)
    dsig = max(0.2, 2 * res_std[2])
    sig_list = np.linspace(max(0, res_mean[2] - dsig), res_mean[2] + dsig, n)
    return (k_list, b_list, sig_list)


def gen_new_grid_quantile(res_arr, n=10, _min_range=0.1):
    q_list = []
    for i in range(3):
        q_16, q_50, q_84 = quantile(res_arr[:, i], [0.16, 0.5, 0.84],
                                    weights=res_arr[:, 4])
        q_list.append((q_16, q_50, q_84))
    q_list = np.array(q_list)

    k_med = q_list[0, 1]
    k_low = max(k_med - q_list[0, 0], _min_range)
    k_up = max(q_list[0, 2] - k_med, _min_range)
    k_list = np.linspace(k_med - 3 * k_low, k_med + 3 * k_up, n)

    b_med = q_list[1, 1]
    b_low = max(b_med - q_list[1, 0], _min_range)
    b_up = max(q_list[1, 2] - b_med, _min_range)
    b_list = np.linspace(b_med - 3 * b_low, b_med + 3 * b_up, n)

    sig_med = q_list[2, 1]
    sig_low = max(sig_med - q_list[2, 0], _min_range)
    sig_up = max(q_list[2, 2] - sig_med, _min_range)
    sig_list = np.linspace(max(0, sig_med - 3 * sig_low), sig_med + 3 * sig_up,
                           n)

    return (k_list, b_list, sig_list)


def print_mean_std_loglinear(arg0, arg1):
    # print res with std
    print('----')
    print(f'k: {arg0[0]:.3f} +/- {arg1[0]:.3f}')
    print(f'b: {arg0[1]:.3f} +/- {arg1[1]:.3f}')
    print(f'sig: {arg0[2]:.3f} +/- {arg1[2]:.3f}')
