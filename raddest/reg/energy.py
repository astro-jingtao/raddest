"""Energy-distance based distribution comparison for regression fitting.

The energy distance between two samples ``X`` and ``Y`` is
``E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` (with the Euclidean metric).
It is zero if and only if the two samples come from the same distribution and
grows with their separation, which makes it usable both as a two-sample test
statistic and, being differentiable almost everywhere, as a fitting loss.

Two flavors are provided for the two fitting optimizers:

- :func:`energy_test` — a permutation test (numpy, jax-free) returning the test
  statistic and a p-value; used inside grid-search worker processes, where a
  jax import would allocate a GPU context per worker.
- :func:`energy_distance_jax` — the differentiable jax energy distance used as
  an autodiff loss, plus :func:`construct_energy_loss`, which wires a
  differentiable conditional generator into that loss.

Both flavors are parity-tested against the ``pted`` package (see
``test/test_energy.py``).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


def _jax_cdist_sq(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pairwise squared Euclidean distances between rows of ``x`` and ``y``.

    Uses the squared-norm identity
    ``||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i . y_j``
    to avoid materializing the ``(nx, ny, d)`` difference tensor.
    """
    x_sq = jnp.sum(x**2, axis=-1)
    y_sq = jnp.sum(y**2, axis=-1)
    return x_sq[:, None] + y_sq[None, :] - 2.0 * (x @ y.T)


def _jax_cdist(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pairwise Euclidean distances between rows of ``x`` and rows of ``y``."""
    return jnp.sqrt(jnp.maximum(_jax_cdist_sq(x, y), 0.0))


def energy_distance_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Differentiable energy distance between two samples of 2D points.

    Parameters
    ----------
    x : jax.Array
        First sample, shape ``(N, D)``.
    y : jax.Array
        Second sample, shape ``(M, D)``; ``D`` must match ``x``.

    Returns
    -------
    jax.Array
        Scalar energy distance ``2 E||x - y|| - E||x - x'|| - E||y - y'||``,
        kept as a jax array.

    Notes
    -----
    This is the V-statistic form (within-sample means include the zero
    self-distances on the diagonal), matching the pted reference statistic
    exactly. Its gradient with respect to the sample values is NOT finite
    (the derivative of ``sqrt`` at the zero diagonal), so for gradient-based
    fitting use the U-statistic loss built by :func:`construct_energy_loss`
    instead of differentiating this function.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"x and y must have the same feature shape, not {x.shape} and {y.shape}"
        )
    d_xx = _jax_cdist(x, x)
    d_yy = _jax_cdist(y, y)
    d_xy = _jax_cdist(x, y)
    return 2.0 * jnp.mean(d_xy) - jnp.mean(d_xx) - jnp.mean(d_yy)


_SQRT_FLOOR = 1e-12


def _energy_distance_u_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Gradient-safe U-statistic energy distance (off-diagonal within-blocks).

    Same formula as :func:`energy_distance_jax`, except the within-sample
    means run over off-diagonal pairs only, every ``sqrt`` gets the
    ``_SQRT_FLOOR`` floor, and the constant ``x``-``x`` block is detached —
    so all gradients with respect to ``y`` are finite. The value differs from
    the V-statistic by O(1/N).
    """
    n = x.shape[0]
    m = y.shape[0]
    d_xx = jnp.sqrt(jnp.maximum(_jax_cdist_sq(x, x), _SQRT_FLOOR)) * (
        1.0 - jnp.eye(n, dtype=x.dtype))
    e_xx = jax.lax.stop_gradient(jnp.sum(d_xx) / (n * (n - 1)))

    d_yy = jnp.sqrt(jnp.maximum(_jax_cdist_sq(y, y), _SQRT_FLOOR)) * (
        1.0 - jnp.eye(m, dtype=y.dtype))
    e_yy = jnp.sum(d_yy) / (m * (m - 1))

    d_xy = jnp.sqrt(jnp.maximum(_jax_cdist_sq(x, y), _SQRT_FLOOR))
    e_xy = jnp.mean(d_xy)

    return 2.0 * e_xy - e_xx - e_yy


def energy_distance_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Numpy energy distance between two samples (worker-safe reference form).

    Parameters
    ----------
    x : np.ndarray
        First sample, shape ``(N, D)``.
    y : np.ndarray
        Second sample, shape ``(M, D)``; ``D`` must match ``x``.

    Returns
    -------
    float
        Scalar energy distance, the same statistic :func:`energy_distance_jax`
        computes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"x and y must have the same feature shape, not {x.shape} and {y.shape}"
        )
    z = np.concatenate((x, y), axis=0)
    d = cdist(z, z)
    nx = len(x)
    ny = len(y)
    e_xx = d[:nx, :nx].sum() / nx**2
    e_yy = d[nx:, nx:].sum() / ny**2
    e_xy = d[:nx, nx:].sum() / (nx * ny)
    return float(2 * e_xy - e_xx - e_yy)


def energy_test(
    x: ArrayLike,
    y: ArrayLike,
    permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Two-sample permutation test on the energy distance (numpy, jax-free).

    The pooled distance matrix is computed once and each permutation only
    re-indexes it, following the same algorithm as ``pted.pted`` with
    ``two_tailed=False``: the p-value is the right-tail fraction
    ``(1 + #{permuted stats >= observed stat}) / (1 + permutations)``.

    Parameters
    ----------
    x : array_like
        First sample, shape ``(N, D)``.
    y : array_like
        Second sample, shape ``(M, D)``; ``D`` must match ``x``.
    permutations : int, optional
        Number of permutations. The p-value resolution is ``1 / (1 +
        permutations)``. Default is 200.
    rng : numpy.random.Generator, optional
        Generator used for the permutations. A fresh default generator is used
        when None. Default is None.

    Returns
    -------
    test_stat : float
        Energy distance between ``x`` and ``y``.
    p_value : float
        One-tailed permutation p-value; small when the samples differ.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(
            f"x and y must be at least 2D (N, D), not {x.shape} and {y.shape}"
        )
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"x and y must have the same feature shape, not {x.shape} and {y.shape}"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Input contains NaN or Inf")

    if rng is None:
        rng = np.random.default_rng()

    nx = len(x)
    z = np.concatenate((x, y), axis=0)
    d = cdist(z, z)

    def stat_from_matrix(d_matrix: np.ndarray) -> float:
        e_xx = d_matrix[:nx, :nx].sum() / nx**2
        e_yy = d_matrix[nx:, nx:].sum() / (z.shape[0] - nx) ** 2
        e_xy = d_matrix[:nx, nx:].sum() / (nx * (z.shape[0] - nx))
        return 2 * e_xy - e_xx - e_yy

    test_stat = stat_from_matrix(d)
    n_all = z.shape[0]
    count = 0
    for _ in range(permutations):
        idx = rng.permutation(n_all)
        if stat_from_matrix(d[idx][:, idx]) >= test_stat:
            count += 1

    return float(test_stat), float((1 + count) / (1 + permutations))


def construct_energy_compare(
    generator_cond: Callable,
    x_gen: np.ndarray,
    x_obs_gen: np.ndarray,
    y_kdt: np.ndarray,
    y_err_kdt: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    permutations: int = 200,
) -> Callable[..., tuple[float, float]]:
    """Build a grid-search compare function based on the energy test.

    Mirrors :func:`raddest.reg.constructor.construct_ks_compare`: for each
    parameter point the conditional generator produces synthetic observations,
    which are compared to the observed sample as 2D points ``(x, y)`` via
    :func:`energy_test`. The closure references only numpy objects, so it is
    safe to run inside joblib worker processes (no jax import).

    Parameters
    ----------
    generator_cond : callable
        Conditional generator built by ``construct_generator_cond``;
        ``generator_cond(x_gen, y_kdt, y_err_kdt, params) ->
        (y_pred, y_err_pred, y_obs_pred)``.
    x_gen : np.ndarray
        Latent x draws used as generator input.
    x_obs_gen : np.ndarray
        Observed-side x values paired with the generated y values.
    y_kdt : np.ndarray
        y sample of the yflow, used for error matching in the generator.
    y_err_kdt : np.ndarray
        y error values paired with ``y_kdt``.
    x_obs : np.ndarray
        Observed x values.
    y_obs : np.ndarray
        Observed y values.
    permutations : int, optional
        Number of permutations per grid point. Default is 200.

    Returns
    -------
    callable
        ``energy_compare(*params) -> (test_stat, p_value)``.
    """

    obs = np.stack([np.asarray(x_obs, dtype=float), np.asarray(y_obs, dtype=float)],
                   axis=1)
    x_obs_gen = np.asarray(x_obs_gen, dtype=float)

    def energy_compare(*params: float) -> tuple[float, float]:
        _, _, y_obs_gen = generator_cond(x_gen, y_kdt, y_err_kdt, params)
        gen = np.stack([x_obs_gen, np.asarray(y_obs_gen, dtype=float)], axis=1)
        return energy_test(gen, obs, permutations=permutations)

    return energy_compare


def match_y_err_jax(
    y_pred: jax.Array, y_kdt: jax.Array, y_err_kdt: jax.Array, batch_size: int = 512
) -> jax.Array:
    """Match each predicted y with the error of its nearest ``y_kdt`` sample.

    Mirrors the KDTree nearest-neighbor error matching of
    ``construct_generator_cond``. The matching indices are piecewise-constant
    in ``y_pred``, so their dependence on the parameters is cut off with
    ``stop_gradient``: the matched errors track the current prediction while
    the loss gradient treats them as constants.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted y values, shape ``(N,)``.
    y_kdt : jax.Array
        y sample of the yflow, shape ``(M,)``.
    y_err_kdt : jax.Array
        y error values paired with ``y_kdt``, shape ``(M,)``.
    batch_size : int, optional
        Row batch for the pairwise comparison, bounding the ``(batch, M)``
        intermediate. Default is 512.

    Returns
    -------
    jax.Array
        Matched y errors, shape ``(N,)``.
    """

    def match_one(yi: jax.Array) -> jax.Array:
        idx = jnp.argmin(jnp.abs(y_kdt - yi))
        return jax.lax.stop_gradient(y_err_kdt[idx])

    n = y_pred.shape[0]
    n_batch = (n + batch_size - 1) // batch_size
    pad = n_batch * batch_size - n
    if pad:
        y_pred = jnp.pad(y_pred, (0, pad), constant_values=y_kdt[0])
    matched = jax.vmap(jax.vmap(match_one))(y_pred.reshape(n_batch, batch_size))
    return matched.reshape(-1)[:n]


def construct_energy_loss(
    func_x2y_jax: Callable,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_gen: np.ndarray,
    x_obs_gen: np.ndarray,
    y_kdt: np.ndarray,
    y_err_kdt: np.ndarray,
    y_min: float | None = None,
    y_max: float | None = None,
    seed: int | np.random.Generator | None = 0,
) -> Callable[..., jax.Array]:
    """Build a differentiable energy-distance loss for gradient-based fitting.

    A synthetic sample is generated differentiably from the parameters as
    ``y_obs_gen = clip(func_x2y_jax(x_gen, params, eps_scatter)) +
    eps_obs * y_err_matched`` with the unit-normal draws ``eps_scatter`` and
    ``eps_obs`` fixed at construction (reparameterization), exactly mirroring
    ``construct_generator_cond`` composed with a scatter function whose noise
    is injected from outside.

    The loss is the U-statistic energy distance between the 2D points
    ``(x_obs, y_obs)`` and ``(x_obs_gen, y_obs_gen)``: within-sample means run
    over off-diagonal pairs and every ``sqrt`` is floored, so gradients with
    respect to ``params`` are finite (differentiating the V-statistic
    :func:`energy_distance_jax` hits the zero self-distances on the diagonal
    of the distance matrix and yields NaN gradients).

    Parameters
    ----------
    func_x2y_jax : callable
        Differentiable scatter function, ``func_x2y_jax(x, params, eps) ->
        y_pred`` where ``eps`` is a fixed ``(N,)`` unit-normal draw carried in
        the scatter the same way the model's intrinsic scatter parameter is.
    x_obs : np.ndarray
        Observed x values.
    y_obs : np.ndarray
        Observed y values.
    x_gen : np.ndarray
        Latent x draws used as generator input.
    x_obs_gen : np.ndarray
        Observed-side x values paired with the generated y values.
    y_kdt : np.ndarray
        y sample of the yflow, used for error matching.
    y_err_kdt : np.ndarray
        y error values paired with ``y_kdt``.
    y_min : float or None, optional
        Clip lower bound for the predicted y; the minimum of ``y_kdt`` when
        None. Default is None.
    y_max : float or None, optional
        Clip upper bound for the predicted y; the maximum of ``y_kdt`` when
        None. Default is None.
    seed : int, numpy.random.Generator or None, optional
        Seed or generator for the two fixed unit-normal draws; 0 when the
        default. Default is 0.

    Returns
    -------
    callable
        ``loss(params) -> scalar jax.Array``, differentiable in ``params``.
    """
    if y_min is None:
        y_min = float(np.min(y_kdt))
    if y_max is None:
        y_max = float(np.max(y_kdt))

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    eps_scatter = jnp.asarray(rng.standard_normal(len(x_gen)), dtype=jnp.float32)
    eps_obs = jnp.asarray(rng.standard_normal(len(x_gen)), dtype=jnp.float32)

    obs_2d = jnp.stack(
        [jnp.asarray(x_obs, dtype=jnp.float32), jnp.asarray(y_obs, dtype=jnp.float32)],
        axis=1,
    )
    x_gen_j = jnp.asarray(x_gen, dtype=jnp.float32)
    x_obs_gen_j = jnp.asarray(x_obs_gen, dtype=jnp.float32)
    y_kdt_j = jnp.asarray(y_kdt, dtype=jnp.float32)
    y_err_kdt_j = jnp.asarray(y_err_kdt, dtype=jnp.float32)
    y_fill = jnp.asarray(np.mean(y_kdt), dtype=jnp.float32)

    def loss(params: jax.Array) -> jax.Array:
        y_pred = func_x2y_jax(x_gen_j, params, eps_scatter)
        y_pred = jnp.clip(y_pred, y_min, y_max)
        y_pred = jnp.nan_to_num(y_pred, nan=y_fill)
        y_err_matched = match_y_err_jax(y_pred, y_kdt_j, y_err_kdt_j)
        y_obs_gen = y_pred + eps_obs * y_err_matched
        gen_2d = jnp.stack([x_obs_gen_j, y_obs_gen], axis=1)
        return _energy_distance_u_jax(obs_2d, gen_2d)

    return loss
