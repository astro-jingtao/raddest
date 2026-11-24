"""Integration-grid and weight helpers shared by the regressors: 1D and 2D
node/weight generators (Gauss-Hermite, QMC, grids), batching utilities, and
transform-aware data plumbing."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike
from scipy.stats.qmc import MultivariateNormalQMC

from raddest.gauss_hermite_quadrature import GaussHermiteQuadrature
from raddest.transfer import Transfer

if TYPE_CHECKING:
    from pzflow import Flow


def get_wy_1d(n: int) -> tuple[jax.Array, jax.Array]:
    """Gauss-Hermite nodes and weights as (1, n) row arrays."""
    x, w = hermgauss(n)
    x = jnp.array(x.reshape(1, -1))
    w = jnp.array(w.reshape(1, -1))
    return x, w


def get_musig_1d(
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Reshape (x_obs, x_err) into (N, 1) column arrays."""
    mu = jnp.array(x_obs.reshape(-1, 1))
    sig = jnp.array(x_err.reshape(-1, 1))
    return mu, sig


def get_x_err_ext(x_err: ArrayLike, n: int) -> jax.Array:
    """Repeat x_err as (N, n) columns, one row per observation."""
    return jnp.repeat(x_err.reshape(-1, 1), n, axis=1)


def get_ext_w(
    n: int,
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """1D Gauss-Hermite nodes around each observation with quadrature weights."""
    # https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    x, w = get_wy_1d(n)
    mu, sig = get_musig_1d(x_obs, x_err)
    x_ext = jnp.sqrt(2) * sig * x + mu
    x_err_ext = jnp.repeat(x_err.reshape(-1, 1), n, axis=1)
    return x_ext, x_err_ext, w / jnp.sqrt(jnp.pi)


def round_to_2power(x: float) -> int:
    """Round x up to the next power of 2."""
    return int(2 ** np.ceil(np.log2(x)))


def is_2power(x: float) -> bool:
    """True when x is an exact power of 2."""
    return np.log2(x) % 1 == 0


def get_ext_w_qmc(
    n: int,
    x_obs: ArrayLike,
    x_err: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quasi-Monte-Carlo standard-normal draws as nodes with uniform weights;
    n is rounded up to a power of 2 (with a printed notice)."""

    if not is_2power(n):
        print(f"n={n} is not a power of 2, rounding to {round_to_2power(n)}")
        n = round_to_2power(n)

    x = (
        MultivariateNormalQMC(
            mean=[
                0,
            ],
            cov=[
                [
                    1,
                ]
            ],
        )
        .random(n)
        .T
    )
    mu, sig = get_musig_1d(x_obs, x_err)
    x_ext = jnp.array(x * sig + mu)
    x_err_ext = jnp.repeat(x_err.reshape(-1, 1), n, axis=1)
    w = jnp.ones_like(x_ext) / n
    return x_ext, x_err_ext, w


def get_ext_w_inter_grid(
    n: int,
    x_obs: ArrayLike,
    x_err: ArrayLike,
    k: int = 5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unit-interval grid stretched to each observation's k-sigma interval,
    weighted by the Gaussian pdf times cell area."""
    x_ext_edge = jnp.linspace(0, 1, n + 1)
    x_ext, area = edge_to_ext(x_obs.size, x_ext_edge)
    _x_min = jnp.clip(x_obs - k * x_err, a_min=x_obs.min(), a_max=None)[:, None]
    _x_max = jnp.clip(x_obs + k * x_err, a_min=None, a_max=x_obs.max())[:, None]
    x_ext = x_ext * (_x_max - _x_min) + _x_min
    area = area * (_x_max - _x_min)
    x_err_ext = get_x_err_ext(x_err, n)

    w = norm.pdf(x_ext, loc=x_obs[:, None], scale=x_err[:, None]) * area

    return x_ext, x_err_ext, w


def edge_to_ext(
    N: int,
    x_ext_edge: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Cell centers repeated per observation and the cell areas."""
    x_ext = (x_ext_edge[1:] + x_ext_edge[:-1]) / 2
    x_ext = jnp.repeat(x_ext[None, :], N, axis=0)
    area = x_ext_edge[1:] - x_ext_edge[:-1]
    return x_ext, area


def ext_w_2d_empty_wrapper(func: Callable) -> Callable:
    """Make a 2D ext-w function tolerate empty inputs by returning five empty
    arrays."""

    @wraps(func)
    def wrapper(
        *args: ArrayLike, **kwargs: ArrayLike
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        if len(args[0]) == 0:
            return jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([])
        return func(*args, **kwargs)

    return wrapper


@ext_w_2d_empty_wrapper
def get_ext_w_2d(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """2D Gauss-Hermite product nodes around each (x, y) observation."""
    ghq = GaussHermiteQuadrature(dim=2, nodes=n)
    x = ghq.X[:, 0].reshape(1, -1)
    y = ghq.X[:, 1].reshape(1, -1)

    x_ext = x_obs.reshape(-1, 1) + np.sqrt(2) * x_err.reshape(-1, 1) * x
    y_ext = y_obs.reshape(-1, 1) + np.sqrt(2) * y_err.reshape(-1, 1) * y

    x_err_ext = get_x_err_ext(x_err, n * n)
    y_err_ext = get_x_err_ext(y_err, n * n)

    return x_ext, y_ext, x_err_ext, y_err_ext, ghq.W


@ext_w_2d_empty_wrapper
def get_ext_w_2d_qmc(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """2D quasi-Monte-Carlo nodes with uniform weights; n rounds up to a
    power of 2 (with a printed notice)."""

    if not is_2power(n):
        print(f"n={n} is not a power of 2, rounding to {round_to_2power(n)}")
        n = round_to_2power(n)

    X = (
        MultivariateNormalQMC(
            mean=[
                0,
                0,
            ],
            cov=[
                [
                    1,
                    0,
                ],
                [
                    0,
                    1,
                ],
            ],
        )
        .random(n)
        .T
    )

    x_ext = X[0] * x_err.reshape(-1, 1) + x_obs.reshape(-1, 1)
    y_ext = X[1] * y_err.reshape(-1, 1) + y_obs.reshape(-1, 1)
    x_err_ext = get_x_err_ext(x_err, n)
    y_err_ext = get_x_err_ext(y_err, n)
    w = jnp.ones(n) / n
    return x_ext, y_ext, x_err_ext, y_err_ext, w


@ext_w_2d_empty_wrapper
def get_ext_w_2d_inter_grid(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
    *,
    k: int = 5,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Unit-interval 2D grid stretched to each observation's k-sigma box,
    weighted by the product Gaussian pdf times cell area."""
    x_ext_edge = jnp.linspace(0, 1, n + 1)
    y_ext_edge = jnp.linspace(0, 1, n + 1)
    x_ext, y_ext, area = edge_to_ext_2d(x_obs.size, x_ext_edge, y_ext_edge)

    _x_min = jnp.clip(x_obs - k * x_err, a_min=x_obs.min(), a_max=None)[:, None]
    _x_max = jnp.clip(x_obs + k * x_err, a_min=None, a_max=x_obs.max())[:, None]
    _y_min = jnp.clip(y_obs - k * y_err, a_min=y_obs.min(), a_max=None)[:, None]
    _y_max = jnp.clip(y_obs + k * y_err, a_min=None, a_max=y_obs.max())[:, None]

    # scale
    x_ext = x_ext * (_x_max - _x_min) + _x_min
    y_ext = y_ext * (_y_max - _y_min) + _y_min
    area = area * (_x_max - _x_min) * (_y_max - _y_min)

    x_err_ext = get_x_err_ext(x_err, n * n)
    y_err_ext = get_x_err_ext(y_err, n * n)

    w = (
        norm.pdf(x_ext, loc=x_obs[:, None], scale=x_err[:, None])
        * norm.pdf(y_ext, loc=y_obs[:, None], scale=y_err[:, None])
        * area
    )

    return x_ext, y_ext, x_err_ext, y_err_ext, w


@ext_w_2d_empty_wrapper
def get_ext_w_2d_grid(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Uniform-edge grid over the observed ranges."""
    x_ext_edge = jnp.linspace(x_obs.min(), x_obs.max(), n + 1)
    y_ext_edge = jnp.linspace(y_obs.min(), y_obs.max(), n + 1)
    return get_ext_w_2d_from_edge(
        x_obs, y_obs, x_err, y_err, n, x_ext_edge=x_ext_edge, y_ext_edge=y_ext_edge
    )


@ext_w_2d_empty_wrapper
def get_ext_w_2d_qgrid(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Quantile-edge grid over the observed samples."""
    x_ext_edge = jnp.quantile(x_obs, jnp.linspace(0, 1, n + 1))
    y_ext_edge = jnp.quantile(y_obs, jnp.linspace(0, 1, n + 1))
    return get_ext_w_2d_from_edge(
        x_obs, y_obs, x_err, y_err, n, x_ext_edge=x_ext_edge, y_ext_edge=y_ext_edge
    )


def get_ext_w_2d_from_edge(
    x_obs: ArrayLike,
    y_obs: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
    n: int,
    *,
    x_ext_edge: ArrayLike,
    y_ext_edge: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Grid cells given explicit edges, weighted by the product Gaussian pdf
    times cell area."""
    x_ext, y_ext, area = edge_to_ext_2d(x_obs.size, x_ext_edge, y_ext_edge)

    w = (
        norm.pdf(x_ext, loc=x_obs[:, None], scale=x_err[:, None])
        * norm.pdf(y_ext, loc=y_obs[:, None], scale=y_err[:, None])
        * area
    )

    x_err_ext = get_x_err_ext(x_err, n * n)
    y_err_ext = get_x_err_ext(y_err, n * n)
    return x_ext, y_ext, x_err_ext, y_err_ext, w


def edge_to_ext_2d(
    N: int,
    x_ext_edge: ArrayLike,
    y_ext_edge: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """2D cell centers (flattened, repeated per observation) and cell areas."""
    x_ext_cen = (x_ext_edge[:-1] + x_ext_edge[1:]) / 2
    y_ext_cen = (y_ext_edge[:-1] + y_ext_edge[1:]) / 2
    x_ext, y_ext = grid_and_flat(x_ext_cen, y_ext_cen)
    x_ext = jnp.repeat(x_ext[None, :], N, axis=0)
    y_ext = jnp.repeat(y_ext[None, :], N, axis=0)

    x_ext_width = x_ext_edge[1:] - x_ext_edge[:-1]
    y_ext_width = y_ext_edge[1:] - y_ext_edge[:-1]
    x_ext_width, y_ext_width = grid_and_flat(x_ext_width, y_ext_width)
    area = x_ext_width * y_ext_width
    return x_ext, y_ext, area


def get_p_flow(
    flow: Flow,
    x: ArrayLike,
    x_err: ArrayLike,
    batch_size: int | None = None,
) -> jax.Array:
    """Calculate probabilities in batches to avoid OOM errors.

    Parameters
    ----------
    flow : pzflow flow model
        The flow model whose log-probability is evaluated.
    x : array_like
        Array of x values.
    x_err : array_like
        Array of x error values.
    batch_size : int or None, optional
        Number of samples to process in each batch. If None, process all at
        once. Default is None.

    Returns
    -------
    array
        Array of probabilities.
    """

    if batch_size is None:
        # If n_batch is None, process the entire array
        concat_input = jnp.c_[x, x_err]  # Use jnp.c_ for concatenation
        log_p = flow._log_prob(
            flow._params, concat_input, jnp.zeros(shape=(concat_input.shape[0], 1))
        )
        return jnp.exp(log_p)

    # If n_batch is specified, process in chunks
    n_samples = x.shape[0]
    probabilities = []

    for i in range(0, n_samples, batch_size):
        end_index = min(i + batch_size, n_samples)
        x_batch = x[i:end_index]
        x_err_batch = x_err[i:end_index]

        concat_input = jnp.c_[x_batch, x_err_batch]  # Use jnp.c_ for concatenation
        log_p_batch = flow._log_prob(
            flow._params, concat_input, jnp.zeros(shape=(concat_input.shape[0], 1))
        )
        probabilities.append(jnp.exp(log_p_batch))

    return jnp.concatenate(probabilities)


# ----


def grid_and_flat(
    x_1d: ArrayLike,
    y_1d: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Meshgrid of two 1D arrays, flattened."""
    x_grid, y_grid = jnp.meshgrid(x_1d, y_1d)
    return x_grid.flatten(), y_grid.flatten()


def concat(*args: ArrayLike, axis: int = 0) -> jax.Array:
    """Concatenate skipping empty arrays."""
    to_concat = [arr for arr in args if arr.size > 0]
    return jnp.concatenate(to_concat, axis=axis)


def transform_data(
    x: ArrayLike,
    x_err: ArrayLike,
    transfer: Transfer | None,
    transfer_params: ArrayLike | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply transfer.trans when a transfer is given; pass through otherwise."""

    if transfer is None:
        return x, x_err

    u, v = transfer.trans(x, x_err, transfer_params)

    return u, v
