"""KS-test regression ingredients: sample generation from flows for both axes,
log/linear scatter generators, and KS comparators over the (k, b, sig) family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from ndtest.ksnd import ks2d2s
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t as student_t

from ..transfer import NegLogPlus, Transfer
from ..x_xerr import gen_xobs_trans
from .constructor import construct_generator_cond

if TYPE_CHECKING:
    from pzflow import Flow


def gen_xy_obs_prepare(flowx: Flow,  # noqa: PLR0913 — one parameter per physical input: two flows x four range bounds x two transfers plus sampling size and output dtype
                       flowy: Flow,
                       *,
                       x_range: tuple[float, float],
                       x_err_range: tuple[float, float],
                       y_range: tuple[float, float],
                       y_err_range: tuple[float, float],
                       n: int = 1000,
                       x_transfer: Transfer | None = None,
                       x_transfer_params: ArrayLike | None = None,
                       y_transfer: Transfer | None = None,
                       y_transfer_params: ArrayLike | None = None,
                       return_numpy: bool = False,
                       ) -> tuple[jax.Array | np.ndarray, jax.Array | np.ndarray,
                                  jax.Array | np.ndarray, jax.Array | np.ndarray,
                                  jax.Array | np.ndarray]:
    """Draw the x sample and the y KDTree tables from the two flows.

    Returns (x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt), as numpy
    arrays when return_numpy is set.
    """
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


def loglinear_scatter(x: ArrayLike, params: ArrayLike) -> np.ndarray:
    '''Draw y ~ 10^(k log10 x + b + N(0, sig)) for params = (k, b, sig).'''
    k, b, sig = params
    return np.power(
        10,
        k * np.log10(x) + b + np.random.normal(scale=sig, size=x.shape))


loglinear_generator_cond = construct_generator_cond(loglinear_scatter)


def linear_scatter(x: ArrayLike, params: ArrayLike) -> np.ndarray:
    '''Draw y ~ k x + b + N(0, sig) for params = (k, b, sig).'''
    k, b, sig = params
    return k * x + b + np.random.normal(scale=sig, size=x.shape)


linear_generator_cond = construct_generator_cond(linear_scatter)


def loglinear_scatter_student_t(
    x: ArrayLike,
    params: ArrayLike,
) -> np.ndarray:
    '''Draw y ~ 10^(k log10 x + b + sig * t_df) for params = (k, b, df, sig).'''
    k, b, df, sig = params
    return np.power(
        10,
        k * np.log10(x) + b + student_t.rvs(df, scale=sig, size=x.shape))


loglinear_generator_cond_student_t = construct_generator_cond(
    loglinear_scatter_student_t)


def loglinear_x2y(
    x: ArrayLike,
    x_obs: ArrayLike,
    y_kdt: ArrayLike,
    y_err_kdt: ArrayLike,
    *,
    k: float,
    b: float,
    sig: float,
    return_more: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    """Push x through the loglinear conditional generator.

    Returns (x_obs, y_obs_pred), or with return_more also the matched errors
    and noise-free prediction: (x_obs, y_obs_pred, y_err_pred, y_pred).
    """
    y_pred, y_err_pred, y_obs_pred = loglinear_generator_cond(
        x, y_kdt, y_err_kdt, (k, b, sig))

    if return_more:
        return x_obs, y_obs_pred, y_err_pred, y_pred
    return x_obs, y_obs_pred


def gen_xy_obs_loglinear(flowx: Flow,  # noqa: PLR0913 — gen_xy_obs_prepare's twelve inputs plus the (k, b, sig) model triple and the return_more switch
                         flowy: Flow,
                         *,
                         x_range: tuple[float, float],
                         x_err_range: tuple[float, float],
                         y_range: tuple[float, float],
                         y_err_range: tuple[float, float],
                         k: float,
                         b: float,
                         sig: float,
                         n: int = 1000,
                         x_transfer: Transfer | None = None,
                         x_transfer_params: ArrayLike | None = None,
                         y_transfer: Transfer | None = None,
                         y_transfer_params: ArrayLike | None = None,
                         return_numpy: bool = False,
                         return_more: bool = False,
                         ) -> tuple[np.ndarray, np.ndarray] | tuple[
                             np.ndarray, np.ndarray, jax.Array | np.ndarray,
                             jax.Array | np.ndarray, jax.Array | np.ndarray,
                             jax.Array | np.ndarray]:
    """Full mock observation draw: both axes from the flows, then the y side
    through the loglinear conditional generator with (k, b, sig)."""

    x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt = gen_xy_obs_prepare(
        flowx,
        flowy,
        x_range=x_range,
        x_err_range=x_err_range,
        y_range=y_range,
        y_err_range=y_err_range,
        n=n,
        x_transfer=x_transfer,
        x_transfer_params=x_transfer_params,
        y_transfer=y_transfer,
        y_transfer_params=y_transfer_params,
        return_numpy=return_numpy)

    if not return_more:
        return loglinear_x2y(x_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt,
                             k=k, b=b, sig=sig)

    x_obs_gen, y_obs_gen, y_err_gen, y_gen = loglinear_x2y(x_gen,
                                                           x_obs_gen,
                                                           y_gen_kdt,
                                                           y_err_gen_kdt,
                                                           k=k,
                                                           b=b,
                                                           sig=sig,
                                                           return_more=True)
    return x_obs_gen, y_obs_gen, x_gen, y_gen, x_err_gen, y_err_gen


# PLR0913 (13 args) waived here: gen_xy_obs_loglinear with the two transfers
# fixed to NegLogPlus, so it carries their params instead.
def gen_xy_obs_loglinear_neglogplus(  # noqa: PLR0913
    flowx: Flow,
    flowy: Flow,
    *,
    x_range: tuple[float, float],
    x_err_range: tuple[float, float],
    y_range: tuple[float, float],
    y_err_range: tuple[float, float],
    k: float,
    b: float,
    sig: float,
    x_transfer_params: ArrayLike,
    y_transfer_params: ArrayLike,
    n: int = 1000,
    return_numpy: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, jax.Array | np.ndarray, jax.Array | np.ndarray,
    jax.Array | np.ndarray, jax.Array | np.ndarray]:
    """gen_xy_obs_loglinear with both axis transforms fixed to NegLogPlus."""
    return gen_xy_obs_loglinear(flowx,
                                flowy,
                                x_range=x_range,
                                x_err_range=x_err_range,
                                y_range=y_range,
                                y_err_range=y_err_range,
                                k=k,
                                b=b,
                                sig=sig,
                                n=n,
                                x_transfer=NegLogPlus(),
                                x_transfer_params=x_transfer_params,
                                y_transfer=NegLogPlus(),
                                y_transfer_params=y_transfer_params,
                                return_numpy=return_numpy)


# PLR0913 (9 args) waived here: the four generated-sample arrays plus the
# model triple and the two observed reference arrays are the comparison's
# complete data bundle.
def get_ks_reg_loglinear(  # noqa: PLR0913
    x_gen: ArrayLike,
    x_obs_gen: ArrayLike,
    y_kdt: ArrayLike,
    y_err_kdt: ArrayLike,
    params: ArrayLike,
    *,
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    nboot: int | None = None,
    n_jobs: int = 1,
) -> tuple[float, float]:
    """2D two-sample KS of the loglinear mock against the observed sample."""

    _, _, y_obs_gen = loglinear_generator_cond(x_gen, y_kdt, y_err_kdt, params)

    return ks2d2s(x_obs_gen,
                  y_obs_gen,
                  x_obs,
                  y_obs,
                  nboot=nboot,
                  n_jobs=n_jobs)


def compare_loglinear(k: float,  # noqa: PLR0913 — the (k, b, sig) model triple plus the six-sample KS comparison bundle and its two bootstrap controls
                      b: float,
                      sig: float,
                      *,
                      x_gen: ArrayLike,
                      x_obs_gen: ArrayLike,
                      y_gen_kdt: ArrayLike,
                      y_err_gen_kdt: ArrayLike,
                      x_obs: ArrayLike,
                      y_obs: ArrayLike,
                      nboot: int | None = None,
                      n_jobs: int = 1,) -> tuple[float, float]:
    """get_ks_reg_loglinear with the model triple expanded into arguments."""
    return get_ks_reg_loglinear(x_gen,
                                x_obs_gen,
                                y_gen_kdt,
                                y_err_gen_kdt, (k, b, sig),
                                x_obs=x_obs,
                                y_obs=y_obs,
                                nboot=nboot,
                                n_jobs=n_jobs)


def parser_anylinear(
    res: list[tuple],
    iargs: list[tuple],
) -> np.ndarray:
    """Stack grid results as (n_points, 5): params columns then (stat, p)."""
    res_arr = np.zeros((len(res), 5))
    for i, (r, a) in enumerate(zip(res, iargs)):
        res_arr[i, :] = (*a, r[0], r[1])
    return res_arr


def print_mean_std_loglinear(
    _mean: ArrayLike,
    _std: ArrayLike,
) -> None:
    """Print the (k, b, sig) mean +/- std summary lines."""
    # print res with std
    print('----')
    print(f'k: {_mean[0]:.3f} +/- {_std[0]:.3f}')
    print(f'b: {_mean[1]:.3f} +/- {_std[1]:.3f}')
    print(f'sig: {_mean[2]:.3f} +/- {_std[2]:.3f}')
