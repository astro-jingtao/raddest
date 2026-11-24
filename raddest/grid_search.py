"""Grid-search machinery: weighted result summaries, parallel or batched
grid evaluation, zoom-in iteration, and new-grid generators."""

from __future__ import annotations

import itertools
from typing import Callable

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import ArrayLike
from statsmodels.stats.weightstats import DescrStatsW


def get_weighted_res(
    xs: ArrayLike,
    w: ArrayLike,
    std_min: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted per-parameter mean and std (std floored at std_min)."""
    _mean = []
    _std = []
    for x in xs:
        w_x = DescrStatsW(x, weights=w)
        _mean.append(w_x.mean)
        _std.append(max(w_x.std, std_min))
    return np.array(_mean), np.array(_std)


def grid_search(
    func: Callable,
    parmas: list,
    parser: Callable,
    num_process: int = 10,
    batch_func: Callable | None = None,
) -> np.ndarray:
    """Evaluate ``func`` on the full cartesian product of ``parmas``.

    ``batch_func``, when given, replaces the per-point ``func`` calls: it
    receives the whole grid as one ``(G, n_par)`` array and returns an
    ``(G, n_out)`` array, one output row per grid point. This is the hook a
    jax backend uses to evaluate the grid inside one process (no joblib
    workers).
    """
    iargs = list(itertools.product(*parmas))
    if batch_func is None:
        res = Parallel(n_jobs=num_process)(delayed(func)(*args) for args in iargs)
    else:
        params_grid = np.asarray(iargs, dtype=float)
        res = list(map(tuple, batch_func(params_grid)))
    return parser(res, iargs)


def zoom_in_grid(
    func: Callable,
    params: list,
    w_index: int,
    parser: Callable,
    gen_new_grid: Callable,
    *,
    print_mean_std: Callable,
    num_process: int = 10,
    max_i: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = False,
    batch_func: Callable | None = None,
) -> np.ndarray:
    """Iterate grid_search with grids regenerated from the weighted fit,
    stopping when the first three weighted means converge (rtol/atol) or
    after max_i rounds."""
    n_par = len(params)
    res_arr = grid_search(func, params, parser, num_process=num_process, batch_func=batch_func)
    res_mean, res_std = get_weighted_res(res_arr[:, :n_par].T, res_arr[:, w_index])

    if verbose:
        print_mean_std(res_mean, res_std)

    for _ in range(max_i):
        params = gen_new_grid(res_arr, n_par=n_par, w_index=w_index)
        res_arr = grid_search(func, params, parser, num_process=num_process, batch_func=batch_func)
        res_mean_new, res_std_new = get_weighted_res(res_arr[:, :n_par].T, res_arr[:, w_index])
        if verbose:
            print_mean_std(res_mean_new, res_std_new)

        # 收敛判断
        if np.allclose((res_mean_new - res_mean)[:3] / res_mean[:3], 0, rtol=rtol, atol=atol):
            if verbose:
                print("converge")
            return res_arr
        res_mean = res_mean_new
        res_std = res_std_new

    if verbose:
        print("max iteration reached")

    return res_arr


def gen_grid(
    cen: float,
    d_low: float,
    d_up: float,
    n: int,
    _range: tuple[float, float],
) -> np.ndarray:
    """n points from cen-d_low to cen+d_up, clipped to _range."""
    return np.linspace(max(_range[0], cen - d_low), min(_range[1], cen + d_up), n)


def gen_new_grid_mean_std(
    res_arr: np.ndarray,
    *,
    n_par: int,
    w_index: int,
    n: int | float | list = 10,
    n_sigma: int | float | list = 2,
    d_par_min: float | list = 0.2,
    range_par: list | None = None,
) -> list:
    """New grid per parameter: weighted mean +/- n_sigma*std (min d_par_min),
    clipped to range_par; scalars broadcast to per-parameter lists."""
    res_mean, res_std = get_weighted_res(res_arr[:, :n_par].T, res_arr[:, w_index])

    # if n, n_sigma, d_par_min are integers/floats, we convert them to lists of integers
    if isinstance(n, (int, float)):
        n = [n] * n_par
    if isinstance(n_sigma, (int, float)):
        n_sigma = [n_sigma] * n_par
    if isinstance(d_par_min, (int, float)):
        d_par_min = [d_par_min] * n_par

    if range_par is None:
        range_par = [[-np.inf, np.inf] for _ in range(n_par)]

    new_list = []

    for i in range(n_par):
        d_this_par = max(d_par_min[i], n_sigma[i] * res_std[i])
        new_list.append(gen_grid(res_mean[i], d_this_par, d_this_par, n[i], range_par[i]))

    return new_list


def gen_new_grid_quantile(
    res_arr: np.ndarray,
    *,
    n_par: int,
    w_index: int,
    n: int | float | list = 10,
    q_low: float | list = 0.16,
    q_up: float | list = 0.84,
    d_par_min: float | list = 0.2,
    range_par: list | None = None,
) -> list:
    """New grid per parameter around the weighted median, spanning the
    [q_low, q_up] weighted quantiles (min half-width d_par_min), clipped to
    range_par; scalars broadcast to per-parameter lists."""
    # if n, n_sigma, d_par_min are integers/floats, we convert them to lists of integers
    if isinstance(n, (int, float)):
        n = [n] * n_par
    if isinstance(q_low, (int, float)):
        q_low = [q_low] * n_par
    if isinstance(q_up, (int, float)):
        q_up = [q_up] * n_par
    if isinstance(d_par_min, (int, float)):
        d_par_min = [d_par_min] * n_par

    if range_par is None:
        range_par = [[-np.inf, np.inf] for _ in range(n_par)]

    new_list = []

    for i in range(n_par):
        q_low_value, med, q_up_value = quantile(
            res_arr[:, i], [q_low[i], 0.5, q_up[i]], weights=res_arr[:, w_index]
        )
        d_low = max(d_par_min[i], med - q_low_value)
        d_up = max(d_par_min[i], q_up_value - med)

        new_list.append(gen_grid(med, d_low, d_up, n[i], range_par[i]))

    return new_list


def flag_bad(x: ArrayLike) -> np.ndarray:
    """
    It returns True if the input is NaN or Inf, and False otherwise

    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates that the corresponding element
    of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)


def quantile(
    x: ArrayLike,
    q: ArrayLike,
    weights: ArrayLike | None = None,
    N_min: int = 2,
) -> list | np.ndarray:
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample.

    N_min : int
        The minimum number of samples required in an output bin. If there are
        not enough samples, the bin will be returned as `nan`.

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.asarray(x)

    x_bad = flag_bad(x)
    x = x[~x_bad]

    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if x.size < N_min:
        return np.nan + np.zeros(q.shape)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    # There are many methods to calculate percentile
    # https://numpy.org/doc/stable/reference/generated/numpy.percentile.htm
    # TODO: The better way is to use the native weights support of np.percentil in numpy 2.0.0
    # But now we use the following method
    if weights is None:
        return np.percentile(x, list(100.0 * q))
    weights = np.atleast_1d(weights).astype(float)
    if len(x) != len(weights):
        raise ValueError("Dimension mismatch: len(weights) != len(x)")
    idx = np.argsort(x)
    sw = weights[idx]
    cdf = np.cumsum(sw)[:-1]
    cdf /= cdf[-1]
    cdf = np.append(0, cdf)
    return np.interp(q, cdf, x[idx]).tolist()
