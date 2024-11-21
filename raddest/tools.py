import jax.numpy as jnp
import numpy as np
from jax import jit
from numpy.polynomial.hermite import hermgauss
from scipy.stats.qmc import MultivariateNormalQMC

from .GaussHermiteQuadrature import GaussHermiteQuadrature


@jit
def neg_log_plus(x):
    return jnp.where(x > 0, jnp.log(x + 1), x)


@jit
def neg_log_plus_inv(x):
    return jnp.where(x > 0, jnp.exp(x) - 1, x)


def get_wy_1d(n):
    x, w = hermgauss(n)
    x = jnp.array(x.reshape(1, -1))
    w = jnp.array(w.reshape(1, -1))
    return x, w


def get_musig_1d(x_obs, x_err):
    mu = jnp.array(x_obs.reshape(-1, 1))
    sig = jnp.array(x_err.reshape(-1, 1))
    return mu, sig


def get_x_err_ext(x_err, n):
    return jnp.repeat(x_err.reshape(-1, 1), n, axis=1)


def get_ext_w(n, x_obs, x_err):
    x, w = get_wy_1d(n)
    mu, sig = get_musig_1d(x_obs, x_err)
    x_ext = jnp.sqrt(2) * sig * x + mu
    x_err_ext = jnp.repeat(x_err.reshape(-1, 1), n, axis=1)
    # w 应该有个 1/sqrt(pi) 的系数，但对结果没有影响
    return x_ext, x_err_ext, w


def round_to_2power(x):
    return int(2**np.ceil(np.log2(x)))


def is_2power(x):
    return np.log2(x) % 1 == 0


def get_ext_w_qmc(n, x_obs, x_err):

    if not is_2power(n):
        print(f"n={n} is not a power of 2, rounding to {round_to_2power(n)}")
        n = round_to_2power(n)

    x = MultivariateNormalQMC(mean=[
        0,
    ], cov=[[
        1,
    ]]).random(n).T
    mu, sig = get_musig_1d(x_obs, x_err)
    x_ext = jnp.array(x * sig + mu)
    x_err_ext = jnp.repeat(x_err.reshape(-1, 1), n, axis=1)
    w = jnp.ones_like(x_ext)
    return x_ext, x_err_ext, w


def get_ext_w_2d(x_obs, y_obs, x_err, y_err, n):
    ghq = GaussHermiteQuadrature(dim=2, nodes=n)
    x = ghq.X[:, 0].reshape(1, -1)
    y = ghq.X[:, 1].reshape(1, -1)

    x_ext = x_obs.reshape(-1, 1) + np.sqrt(2) * x_err.reshape(-1, 1) * x
    y_ext = y_obs.reshape(-1, 1) + np.sqrt(2) * y_err.reshape(-1, 1) * y

    x_err_ext = get_x_err_ext(x_err, n * n)
    y_err_ext = get_x_err_ext(y_err, n * n)

    return x_ext, y_ext, x_err_ext, y_err_ext, ghq.W


def get_ext_w_2d_qmc(x_obs, y_obs, x_err, y_err, n):

    if not is_2power(n):
        print(f"n={n} is not a power of 2, rounding to {round_to_2power(n)}")
        n = round_to_2power(n)

    X = MultivariateNormalQMC(mean=[
        0,
        0,
    ], cov=[[
        1,
        0,
    ], [
        0,
        1,
    ]]).random(n).T

    x_ext = X[0] * x_err.reshape(-1, 1) + x_obs.reshape(-1, 1)
    y_ext = X[1] * y_err.reshape(-1, 1) + y_obs.reshape(-1, 1)
    x_err_ext = get_x_err_ext(x_err, n)
    y_err_ext = get_x_err_ext(y_err, n)
    w = jnp.ones(n)
    return x_ext, y_ext, x_err_ext, y_err_ext, w


def get_p_flow(flow, x, x_err, batch_size=None):
    """
    Calculate probabilities in batches to avoid OOM errors.

    Parameters:
    - flow: The flow model.
    - x: Array of x values.
    - x_err: Array of x error values.
    - batch_size: Number of samples to process in each batch. If None, process all at once.

    Returns:
    - Array of probabilities.
    """

    if batch_size is None:
        # If n_batch is None, process the entire array
        concat_input = jnp.c_[x, x_err]  # Use jnp.c_ for concatenation
        log_p = flow._log_prob(flow._params, concat_input,
                               jnp.zeros(shape=(concat_input.shape[0], 1)))
        return jnp.exp(log_p)

    # If n_batch is specified, process in chunks
    n_samples = x.shape[0]
    probabilities = []

    for i in range(0, n_samples, batch_size):
        end_index = min(i + batch_size, n_samples)
        x_batch = x[i:end_index]
        x_err_batch = x_err[i:end_index]

        concat_input = jnp.c_[x_batch,
                              x_err_batch]  # Use jnp.c_ for concatenation
        log_p_batch = flow._log_prob(
            flow._params, concat_input,
            jnp.zeros(shape=(concat_input.shape[0], 1)))
        probabilities.append(jnp.exp(log_p_batch))

    return jnp.concatenate(probabilities)


# ----
@jit
def shrink_gg(x_obs, x_err):
    # Tweedie's formula when p(x_obs) is considered to be a normal distribution
    mu = jnp.mean(x_obs)
    sigma2 = jnp.var(x_obs)
    B = (sigma2 - jnp.square(x_err)) / sigma2
    return B * x_obs + (1 - B) * mu
