import itertools
import numpy as np
from joblib import Parallel, delayed
from statsmodels.stats.weightstats import DescrStatsW

def get_weighted_res(xs, w, std_min=0.01):
    _mean = []
    _std = []
    for x in xs:
        w_x = DescrStatsW(x, weights=w)
        _mean.append(w_x.mean)
        _std.append(max(w_x.std, std_min))
    return np.array(_mean), np.array(_std)


def grid_search(func, parmas, parser, num_process=10):
    iargs = list(itertools.product(*parmas))
    res = Parallel(n_jobs=num_process)(delayed(func)(*args) for args in iargs)
    return parser(res, iargs)

def zoom_in_grid(func, params, w_index, parser, gen_new_grid, print_mean_std, num_process=10, max_i=10, rtol=1e-3, atol=1e-3, verbose=False):
    n_par = len(params)
    res_arr = grid_search(func, params, parser, num_process=num_process)
    res_mean, res_std = get_weighted_res(res_arr[:, :n_par].T, res_arr[:, w_index])
    
    if verbose:
        print_mean_std(res_mean, res_std)

    for _ in range(max_i):

        # params = gen_new_grid(res_mean, res_std)
        params = gen_new_grid(res_arr)
        res_arr = grid_search(func, params, parser, num_process=num_process)
        res_mean_new, res_std_new = get_weighted_res(res_arr[:, :n_par].T, res_arr[:, w_index])
        if verbose:
            print_mean_std(res_mean_new, res_std_new)

        # 收敛判断
        if np.allclose((res_mean_new - res_mean)[:3] / res_mean[:3],
                       0,
                       rtol=rtol,
                       atol=atol):
            if verbose:
                print('converge')
            return res_arr
        res_mean = res_mean_new
        res_std = res_std_new
    
    if verbose:
        print('max iteration reached')
    
    return res_arr