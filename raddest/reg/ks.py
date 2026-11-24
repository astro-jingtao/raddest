"""KS comparators over the (k, b, sig) family: score a loglinear mock sample
against the observed 2D sample with the two-sample KS test."""

from __future__ import annotations

from ndtest.ksnd import ks2d2s
from numpy.typing import ArrayLike

from raddest.reg.scatter import loglinear_generator_cond


def get_ks_reg_loglinear(
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

    return ks2d2s(x_obs_gen, y_obs_gen, x_obs, y_obs, nboot=nboot, n_jobs=n_jobs)


def compare_loglinear(
    k: float,
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
    n_jobs: int = 1,
) -> tuple[float, float]:
    """get_ks_reg_loglinear with the model triple expanded into arguments."""
    return get_ks_reg_loglinear(
        x_gen,
        x_obs_gen,
        y_gen_kdt,
        y_err_gen_kdt,
        (k, b, sig),
        x_obs=x_obs,
        y_obs=y_obs,
        nboot=nboot,
        n_jobs=n_jobs,
    )
