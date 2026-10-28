"""Energy-distance based distribution comparison for regression fitting.

The energy distance between two samples ``X`` and ``Y`` is
``E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` (with the Euclidean metric).
It is zero if and only if the two samples come from the same distribution and
grows with their separation, which makes it usable both as a two-sample test
statistic and, being differentiable almost everywhere, as a fitting loss.

Three flavors are provided for the fitting optimizers:

- :func:`energy_test` — a permutation test (numpy, jax-free) returning the test
  statistic and a p-value; used inside grid-search worker processes, where a
  jax import would allocate a GPU context per worker. With
  ``chunk_size``/``chunk_iter`` each distance is estimated on sub-samples
  (pted's chunked mode).
- :func:`energy_distance_jax` — the differentiable jax energy distance used as
  an autodiff loss, plus :func:`construct_energy_loss`, which wires a
  differentiable conditional generator into that loss.
- :func:`energy_test_jax` — the permutation test vectorized in jax (mask
  quadratic form), plus :func:`construct_energy_compare_jax`, which evaluates
  a whole parameter grid in one jax process — the GPU-parallel alternative to
  the multiprocess numpy path.

The flavors are parity-tested against the ``pted`` package (see
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


def _energy_test_full(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Full-sample permutation loop on the precomputed pooled distance matrix.

    Returns the observed statistic and the number of permuted statistics at
    least as large.
    """
    nx = len(x)
    n_all = z.shape[0]
    d = cdist(z, z)

    def stat_from_matrix(d_matrix: np.ndarray) -> float:
        e_xx = d_matrix[:nx, :nx].sum() / nx**2
        e_yy = d_matrix[nx:, nx:].sum() / (n_all - nx) ** 2
        e_xy = d_matrix[:nx, nx:].sum() / (nx * (n_all - nx))
        return 2 * e_xy - e_xx - e_yy

    test_stat = stat_from_matrix(d)
    count = 0
    for _ in range(permutations):
        idx = rng.permutation(n_all)
        if stat_from_matrix(d[idx][:, idx]) >= test_stat:
            count += 1
    return test_stat, count


def _energy_test_chunked(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    *,
    chunk_size: int,
    chunk_iter: int,
) -> tuple[float, int]:
    """Chunked permutation loop: every statistic is estimated on sub-samples.

    Follows ``pted.utils.pted_chunk_numpy``: each estimate averages
    ``chunk_iter`` energy distances of random ``chunk_size`` sub-samples, and
    each permutation shuffles the pooled sample before re-estimating.
    """
    nx = len(x)
    n_all = z.shape[0]

    def chunk_estimate(a: np.ndarray, b: np.ndarray) -> float:
        est = 0.0
        for _ in range(chunk_iter):
            idx_a = rng.choice(len(a), size=min(len(a), chunk_size), replace=False)
            idx_b = rng.choice(len(b), size=min(len(b), chunk_size), replace=False)
            est += energy_distance_numpy(a[idx_a], b[idx_b])
        return est / chunk_iter

    test_stat = chunk_estimate(x, y)
    count = 0
    for _ in range(permutations):
        zp = z[rng.permutation(n_all)]
        if chunk_estimate(zp[:nx], zp[nx:]) >= test_stat:
            count += 1
    return test_stat, count


def energy_test(
    x: ArrayLike,
    y: ArrayLike,
    permutations: int = 200,
    rng: np.random.Generator | None = None,
    *,
    chunk_size: int | None = None,
    chunk_iter: int | None = None,
) -> tuple[float, float]:
    """Two-sample permutation test on the energy distance (numpy, jax-free).

    The pooled distance matrix is computed once and each permutation only
    re-indexes it, following the same algorithm as ``pted.pted`` with
    ``two_tailed=False``: the p-value is the right-tail fraction
    ``(1 + #{permuted stats >= observed stat}) / (1 + permutations)``.

    With ``chunk_size``/``chunk_iter`` given, each energy distance (observed
    and permuted) is instead *estimated* on random sub-samples of
    ``chunk_size`` points per sample, averaged over ``chunk_iter`` draws —
    the chunked mode of ``pted`` (``pted_chunk_numpy``), useful for large
    samples: the per-estimate cost drops from O(n^2) to
    O(chunk_size^2 * chunk_iter). The permutation test stays exact; only its
    sensitivity to the difference between the samples decreases.

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
    chunk_size : int or None, optional
        Number of points per sub-sample in chunked estimation; must be given
        together with ``chunk_iter``, and left as None for the full-sample
        computation. Default is None.
    chunk_iter : int or None, optional
        Number of chunked estimates averaged per energy distance. Default is
        None.

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
    if (chunk_size is None) != (chunk_iter is None):
        raise ValueError(
            "chunk_size and chunk_iter must be given together for chunked "
            "estimation, or both left as None")

    if rng is None:
        rng = np.random.default_rng()

    z = np.concatenate((x, y), axis=0)

    if chunk_size is None:
        test_stat, count = _energy_test_full(x, y, z, permutations, rng)
    else:
        test_stat, count = _energy_test_chunked(x,
                                                y,
                                                z,
                                                permutations,
                                                rng,
                                                chunk_size=chunk_size,
                                                chunk_iter=chunk_iter)

    return float(test_stat), float((1 + count) / (1 + permutations))


def _permutation_masks(n: int,
                       nx: int,
                       permutations: int,
                       seed: int) -> jax.Array:
    """Membership masks of ``permutations`` pooled-sample permutations.

    Row ``p`` is 1 where permutation ``p`` assigns the pooled point to the
    first sample (size ``nx``) and 0 elsewhere, shape ``(permutations, n)``.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), permutations)
    perms = jax.vmap(lambda k: jax.random.permutation(k, n))(keys)
    return (perms < nx).astype(jnp.float32)


def _masked_energy_stats(d: jax.Array, nx: int, masks: jax.Array) -> jax.Array:
    """Energy distance of the observed split plus every permuted split.

    Each statistic is computed from the pooled distance matrix ``d`` via the
    mask quadratic form ``S_ab = sum_ij a_i d_ij b_j`` (one matmul for all
    permutations at once), never by re-indexing the matrix.
    """
    n = d.shape[0]
    ny = n - nx
    m0 = jnp.concatenate([jnp.ones(nx, dtype=d.dtype),
                          jnp.zeros(ny, dtype=d.dtype)])
    eye_masks = jnp.concatenate([m0[None], masks], axis=0)

    left = eye_masks @ d  # (1 + P, n)
    s_xx = jnp.sum(left * eye_masks, axis=1)
    s_xy = jnp.sum(left * (1.0 - eye_masks), axis=1)
    s_yy = jnp.sum(d) - s_xx - 2 * s_xy
    stats = 2 * s_xy / (nx * ny) - s_xx / nx**2 - s_yy / ny**2
    return stats  # row 0: observed split; rows 1..: permuted splits


def energy_test_jax(
    x: ArrayLike,
    y: ArrayLike,
    *,
    permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Vectorized two-sample permutation test on the energy distance (jax).

    Same statistic, permutation scheme, and p-value definition as
    :func:`energy_test`, computed on the jax default device with the
    permutations batched through a mask quadratic form instead of a python
    loop over matrix re-indexing. Used by the jax backend of the energy
    grid-search regressor; standalone use gives a fast single-pair test.

    Parameters
    ----------
    x : array_like
        First sample, shape ``(N, D)``.
    y : array_like
        Second sample, shape ``(M, D)``; ``D`` must match ``x``.
    permutations : int, optional
        Number of permutations. Default is 200.
    seed : int, optional
        Seed of the permutation draws. Default is 0.

    Returns
    -------
    test_stat : float
        Energy distance between ``x`` and ``y``.
    p_value : float
        One-tailed permutation p-value; small when the samples differ.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError(
            f"x and y must be at least 2D (N, D), not {x.shape} and {y.shape}")
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"x and y must have the same feature shape, not {x.shape} and {y.shape}")

    @jax.jit
    def run() -> jax.Array:
        z = jnp.concatenate([x, y], axis=0)
        d = _jax_cdist(z, z)
        masks = _permutation_masks(z.shape[0], x.shape[0], permutations, seed)
        stats = _masked_energy_stats(d, x.shape[0], masks)
        count = jnp.sum(stats[1:] >= stats[0])
        p_value = (1 + count) / (1 + permutations)
        return jnp.stack([stats[0], p_value])

    test_stat, p_value = run()
    return float(test_stat), float(p_value)


def construct_energy_compare(
    generator_cond: Callable,
    x_gen: np.ndarray,
    x_obs_gen: np.ndarray,
    y_kdt: np.ndarray,
    y_err_kdt: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    permutations: int = 200,
    *,
    chunk_size: int | None = None,
    chunk_iter: int | None = None,
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
    chunk_size : int or None, optional
        Chunked-estimation sub-sample size per :func:`energy_test`;
        full-sample computation when None. Default is None.
    chunk_iter : int or None, optional
        Chunked-estimation iterations per :func:`energy_test`. Default is
        None.

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
        return energy_test(gen,
                           obs,
                           permutations=permutations,
                           chunk_size=chunk_size,
                           chunk_iter=chunk_iter)

    return energy_compare


def construct_energy_compare_jax(
    func_x2y_jax: Callable,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_gen: np.ndarray,
    x_obs_gen: np.ndarray,
    y_kdt: np.ndarray,
    y_err_kdt: np.ndarray,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    permutations: int = 200,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a jax batch compare function for grid-search energy fitting.

    The batch counterpart of :func:`construct_energy_compare`: instead of one
    joblib worker per grid point (numpy, fresh noise per point), the whole
    grid is evaluated inside one jax process — synthetic samples are
    generated differentiably-style from fixed reparameterized noise draws
    (exactly as in :func:`construct_energy_loss`), and the permutation test
    of each grid point runs through the vectorized mask form of
    :func:`energy_test_jax`. Grid points share one set of permutation draws
    (common random numbers), which reduces the noise of comparing p-values
    across the grid.

    Running in a single process is what avoids the per-worker GPU-context
    OOM of the multiprocess numpy backend; the parallelism comes from the
    batched ``(P, n) @ (n, n)`` statistic matmuls on the device.

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
    permutations : int, optional
        Number of permutations per grid point. Default is 200.
    seed : int, optional
        Seed of the fixed noise draws and of the shared permutation masks.
        Default is 0.

    Returns
    -------
    callable
        ``energy_compare_batch(params_grid) -> (G, 2)`` array of
        ``(test_stat, p_value)`` rows, one per grid point.
    """
    if y_min is None:
        y_min = float(np.min(y_kdt))
    if y_max is None:
        y_max = float(np.max(y_kdt))

    rng = np.random.default_rng(seed)
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

    masks = _permutation_masks(obs_2d.shape[0] + len(x_gen), len(x_gen),
                               permutations, seed)

    def _one_grid_point(params: jax.Array) -> jax.Array:
        y_pred = func_x2y_jax(x_gen_j, params, eps_scatter)
        y_pred = jnp.clip(y_pred, y_min, y_max)
        y_pred = jnp.nan_to_num(y_pred, nan=y_fill)
        y_err_matched = match_y_err_jax(y_pred, y_kdt_j, y_err_kdt_j)
        y_obs_gen = y_pred + eps_obs * y_err_matched
        gen_2d = jnp.stack([x_obs_gen_j, y_obs_gen], axis=1)
        z = jnp.concatenate([gen_2d, obs_2d], axis=0)
        d = _jax_cdist(z, z)
        stats = _masked_energy_stats(d, len(x_gen), masks)
        count = jnp.sum(stats[1:] >= stats[0])
        p_value = (1 + count) / (1 + permutations)
        return jnp.stack([stats[0], p_value])

    # one jitted call evaluates the whole grid: lax.map walks the grid points
    # sequentially on the device, so peak memory stays at one point's
    # distance matrix and permutation intermediates.
    run_grid = jax.jit(
        lambda params_grid: jax.lax.map(_one_grid_point, params_grid))

    def energy_compare_batch(params_grid: np.ndarray) -> np.ndarray:
        params_grid = jnp.asarray(params_grid, dtype=jnp.float32)
        if params_grid.ndim != 2:
            raise ValueError(
                f"params_grid must be 2D (G, n_par), not {params_grid.shape}")
        res = run_grid(params_grid)
        return np.asarray(res)

    return energy_compare_batch


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
