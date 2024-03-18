import numpy as np
from asa.Bcorner import quantile
from ndtest.ksnd import ks2d2s

from .grid_search import get_weighted_res
from .reg import loglinear_x2y


def get_ks_reg_loglinear(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt, params,
                         x_obs, y_obs):
    k = params[0]
    b = params[1]
    sig = params[2]

    x_obs_gen, y_obs_gen = loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt,
                                         y_err_gen_kdt, k, b, sig)

    return ks2d2s(
        x_obs_gen,
        y_obs_gen,
        x_obs,
        y_obs,
    )


def parser(res, iargs):
    res_arr = np.zeros((len(res), 5))
    for i, (r, a) in enumerate(zip(res, iargs)):
        res_arr[i, :] = (*a, r[0], r[1])
    return res_arr


def compare_loglinear(k, b, sig, x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt,
                      x_obs, y_obs):
    return get_ks_reg_loglinear(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt,
                                (k, b, sig), x_obs, y_obs)


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


def print_mean_std(arg0, arg1):
    # print res with std
    print('----')
    print(f'k: {arg0[0]:.3f} +/- {arg1[0]:.3f}')
    print(f'b: {arg0[1]:.3f} +/- {arg1[1]:.3f}')
    print(f'sig: {arg0[2]:.3f} +/- {arg1[2]:.3f}')
