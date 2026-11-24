"""Shrinkage estimators for (x_obs, x_err) pairs under Gaussian or Poisson
error models, plus the recalibration and noise tools they build on."""

from __future__ import annotations

import jax
from jax import jit
import jax.numpy as jnp
from ndtest.ksnd import ks2d2s
import numpy as np
from numpy.typing import ArrayLike
import statsmodels.api as sm


@jit
def shrink_gg(x_obs: ArrayLike, x_err: ArrayLike) -> jax.Array:
    """Tweedie shrinkage when p(x_obs) is a normal distribution.

    gg for Gaussian error - Gaussian distribution: returns
    ``B * x_obs + (1 - B) * mu`` with ``B = (var - err^2) / var``.
    """
    mu = jnp.mean(x_obs)
    sigma2 = jnp.var(x_obs)
    B = (sigma2 - jnp.square(x_err)) / sigma2
    return B * x_obs + (1 - B) * mu


def shrink_possion(
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-error shrinkage via the SNR solution (no closed form).

    Model: x_err = k * x**1/2 + b + d, d ~ N(0, sigma^2); x_obs = x + e,
    e ~ N(0, x_err^2). Steals the population estimate from the SNR method,
    then regresses sqrt(x_pred) on x_err for the predicted scale.
    """
    k, b, sigma, mu_x, S_x = solve_SNR(x_obs, x_err)

    x_pred_pop = mu_x
    w_pop = 1 / S_x**2

    x_pred_obs = x_obs
    w_obs = 1 / x_err**2

    x_pred_p = (w_pop * x_pred_pop + w_obs * x_pred_obs) / (w_pop + w_obs)

    b, k = regress_b_k_poisson(x_pred_p, x_err)

    x_pred_err = ((x_err - b) / k) ** 2
    # std_pred_err = 2 * np.abs((x_err - b) / k) * sigma / k
    # w_err = 1 / std_pred_err**2

    # x_pred = (w_pop * x_pred_pop + w_obs * x_pred_obs +
    #           w_err * x_pred_err) / (w_pop + w_obs + w_err)

    x_err_pred = err_add_noise(np.asarray(x_err), sigma)
    # x_err_pred = x_err

    return x_pred_err, x_err_pred


def regress_b_k_poisson(x_pred: ArrayLike, x_err: ArrayLike) -> np.ndarray:
    """OLS of x_err on const + sqrt(clip(x_pred, 0)); returns [b, k]."""
    x_pred = np.sqrt(np.clip(np.asarray(x_pred), 0, None))
    x_err = np.asarray(x_err)

    X = sm.add_constant(x_pred)
    model = sm.OLS(x_err, X)
    results = model.fit()
    return results.params


def _solve_SNR(
    mu_err: jax.Array,
    mu_obs: jax.Array,
    S_err: jax.Array,
    S_obs: jax.Array,
    C_obs_err: jax.Array,
) -> tuple[
    jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float
]:
    """Solve the SNR system from summary statistics.

    Model: x_err = k * x + b + d, d ~ N(0, sigma^2); x_obs = x + e,
    e ~ N(0, x_err^2). Returns (k, b, sigma, mu_x, S_x); the fallback 0.01
    replaces non-negative-square-root failures. See notes/guess.ipynb.
    """
    k = -C_obs_err / (S_err**2 - S_obs**2 + mu_err**2)
    b = (C_obs_err * mu_obs + S_err**2 * mu_err - S_obs**2 * mu_err + mu_err**3) / (
        S_err**2 - S_obs**2 + mu_err**2
    )
    sigma_sq = (C_obs_err**2 + S_err**4 - S_err**2 * S_obs**2 + S_err**2 * mu_err**2) / (
        S_err**2 - S_obs**2 + mu_err**2
    )
    sigma = sigma_sq**0.5 if sigma_sq >= 0 else 0.01
    mu_x = mu_obs
    S_x_sq = -(S_err**2) + S_obs**2 - mu_err**2
    S_x = S_x_sq**0.5 if S_x_sq >= 0 else 0.01

    return k, b, sigma, mu_x, S_x


def solve_SNR(
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[
    jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float, jax.Array | float
]:
    """Estimate (k, b, sigma, mu_x, S_x) of the SNR error model from samples."""
    mu_err = jnp.mean(x_err)
    mu_obs = jnp.mean(x_obs)
    S_err = jnp.std(x_err)
    S_obs = jnp.std(x_obs)
    C_obs_err = jnp.cov(x_obs, x_err)[0, 1]
    return _solve_SNR(mu_err, mu_obs, S_err, S_obs, C_obs_err)


def shrink_SNR(
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[jax.Array, np.ndarray]:
    """Inverse-variance shrinkage using the SNR error-model solution.

    Model: x_err = k * x + b + d, d ~ N(0, sigma^2); x_obs = x + e,
    e ~ N(0, x_err^2). Weights the population mean, the observation, and the
    error-corrected prediction by their inverse variances.
    """
    k, b, sigma, mu_x, S_x = solve_SNR(x_obs, x_err)

    x_pred_pop = mu_x
    w_pop = 1 / S_x**2

    x_pred_obs = x_obs
    w_obs = 1 / x_err**2

    x_pred_err = (x_err - b) / k
    w_err = 1 / (sigma / k) ** 2

    x_pred = (w_pop * x_pred_pop + w_obs * x_pred_obs + w_err * x_pred_err) / (
        w_pop + w_obs + w_err
    )

    x_err_pred = err_add_noise(np.asarray(x_err), sigma)

    return x_pred, x_err_pred


def recalibrate_x(x_sh: ArrayLike, x_obs: ArrayLike) -> np.ndarray:
    """OLS-recalibrate the shrunk values x_sh against the raw x_obs."""
    x_sh = np.asarray(x_sh)
    x_obs = np.asarray(x_obs)

    X = sm.add_constant(x_sh)
    model = sm.OLS(x_obs, X)
    results = model.fit()
    return results.predict(X)


def _shrink_auto(
    x_obs: ArrayLike,
    x_err: ArrayLike,
    *,
    to_recalibrate: bool | None,
    return_best_info: bool,
    verbose: bool,
) -> (
    tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]
    | tuple[jax.Array | np.ndarray, jax.Array | np.ndarray, str]
):
    """Try every (method, recalibrate) pair and keep the best ks2d2s p-value."""
    if to_recalibrate is not None:
        print("Warning: is_recalibrate is ignored when method is set to auto.")

    x_sh_lst = []
    p_value_lst = []
    info_lst = []
    for m in ["gg", "poisson", "SNR"]:
        for recal in [True, False]:
            x_sh, x_err_sh = shrink_data(x_obs, x_err, method=m, to_recalibrate=recal)
            x_sh_lst.append((x_sh, x_err_sh))
            x_obs_mock = x_sh + np.random.normal(0, x_err_sh)
            p_value_lst.append(ks2d2s(x_obs, x_err, x_obs_mock, x_err_sh)[1])
            info_lst.append(f"{m}, recalibrate={recal}")

    best_idx = np.argmax(p_value_lst)
    if verbose:
        for i, info in enumerate(info_lst):
            print(f"{info}: p-value={p_value_lst[i]}")

        print(f"Best method: {info_lst[best_idx]}: p-value={p_value_lst[best_idx]}")

    if return_best_info:
        return *x_sh_lst[best_idx], info_lst[best_idx]

    return x_sh_lst[best_idx]


def _apply_shrink_method(
    x_obs: np.ndarray,
    x_err: np.ndarray,
    method: str,
) -> tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]:
    """Dispatch one shrinkage method over numpy inputs."""
    if method == "gg":
        x_shrink = shrink_gg(x_obs, x_err)
        x_err_shrink = x_err
    elif method == "poisson":
        x_shrink, x_err_shrink = shrink_possion(x_obs, x_err)
    elif method == "SNR":
        x_shrink, x_err_shrink = shrink_SNR(x_obs, x_err)
    elif method == "none":
        x_shrink = x_obs
        x_err_shrink = x_err
    else:
        raise ValueError("Invalid shrinkage method")
    return x_shrink, x_err_shrink


def shrink_data(
    x_obs: ArrayLike,
    x_err: ArrayLike,
    *,
    method: str = "gg",
    to_recalibrate: bool | None = None,
    return_best_info: bool = False,
    verbose: bool = False,
) -> (
    tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]
    | tuple[jax.Array | np.ndarray, jax.Array | np.ndarray, str]
):
    """Shrink (x_obs, x_err) toward the population, preserving input type.

    method: 'gg', 'poisson', 'SNR', 'none', or 'auto' (try all methods x
    recalibration, keep the best ks2d2s p-value against mock noisy data;
    to_recalibrate is ignored for 'auto', pass it per-call instead).
    to_recalibrate: OLS-recalibrate the shrunk values against x_obs.
    return_best_info: with method='auto', also return the winning
    "method, recalibrate" description string.
    jax array inputs come back as jax arrays, numpy as numpy.
    """
    if method == "auto":
        return _shrink_auto(
            x_obs,
            x_err,
            to_recalibrate=to_recalibrate,
            return_best_info=return_best_info,
            verbose=verbose,
        )

    is_x_jax_arr = isinstance(x_obs, jnp.ndarray)
    is_x_err_jax_arr = isinstance(x_err, jnp.ndarray)

    if is_x_jax_arr:
        x_obs = np.array(x_obs)
    if is_x_err_jax_arr:
        x_err = np.array(x_err)

    x_shrink, x_err_shrink = _apply_shrink_method(x_obs, x_err, method)

    # None and False are equivalent
    if to_recalibrate:
        x_shrink = recalibrate_x(x_shrink, x_obs)

    if is_x_jax_arr:
        x_shrink = jnp.array(x_shrink)
    if is_x_err_jax_arr:
        x_err_shrink = jnp.array(x_err_shrink)

    return x_shrink, x_err_shrink


def err_add_noise(
    x_err: np.ndarray,
    sigma: float,
    maxiter: int = 100,
) -> np.ndarray:
    """Add N(0, sigma) noise to x_err, resampling negatives until non-negative.

    Gives up after maxiter rounds, zeroing any remaining negative values with
    a printed warning.
    """
    x_err_noisy = x_err + np.random.normal(0, sigma, size=x_err.size)
    iteration = 0

    while np.any(x_err_noisy < 0) and iteration < maxiter:
        mask = x_err_noisy < 0
        x_err_noisy[mask] = x_err[mask] + np.random.normal(0, sigma, size=np.sum(mask))
        iteration += 1

    if iteration == maxiter:
        print("Max iterations reached before achieving non-negative results.")
        # Optionally reset negative values to a default value (e.g., 0)
        x_err_noisy[x_err_noisy < 0] = 0  # Or other strategies

    return x_err_noisy
