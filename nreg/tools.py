import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.stats import norm
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


def get_p_flow(flow, x, x_err):
    log_p = flow._log_prob(flow._params, jnp.c_[x, x_err],
                           jnp.zeros(shape=(x.shape[0], 1)))
    return jnp.exp(log_p)


# -----------


@jit
def trans_neglogplus(x, x_err, params):
    x_median, x_err_median = params
    x_trans = neg_log_plus(x / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


@jit
def trans_neglogplus_inv(x_trans, x_err_trans, params):
    x_median, x_err_median = params
    x = neg_log_plus_inv(x_trans) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


@jit
def trans_neglogplus_dx(x, x_err, params):
    x_median, x_err_median = params
    return jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)


@jit
def trans_neglogplus_jac(x, x_err, params):
    x_median, x_err_median = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


# -----------

# u = cos * log(x) - sin * log(x_err)
# v = sin * log(x) + cos * log(x_err)


@jit
def trans_xlog_rotation(x, x_err, params):
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    log_x = jnp.log(x)
    log_x_err = jnp.log(x_err)
    u = cos * log_x - sin * log_x_err
    v = sin * log_x + cos * log_x_err
    return u, v


@jit
def trans_xlog_rotation_inv(u, v, params):
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    x = jnp.exp(u * cos + v * sin)
    x_err = jnp.exp(-u * sin + v * cos)
    return x, x_err


@jit
def trans_xlog_rotation_dx():
    ...


@jit
def trans_xlog_rotation_jac(x, x_err, params):
    return 1 / (x * x_err)


# xshift_neglogplus
# log((x + shift)/x_median + 1)


@jit
def trans_xshift_neglogplus(x, x_err, params):
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus((x + x_shift) / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


@jit
def trans_xshift_neglogplus_inv(x_trans, x_err_trans, params):
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans) * x_median - x_shift
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


@jit
def trans_xshift_neglogplus_dx(x, x_err, params):
    x_median, x_err_median, x_shift = params
    return jnp.where((x + x_shift) / x_median > 0,
                     1 / (x + x_shift + x_median), 1 / x_median)


@jit
def trans_xshift_neglogplus_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
    k_x = jnp.where((x + x_shift) / x_median > 0, 1 / (x + x_shift + x_median),
                    1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


# -----------

# neglogplus_xshift
# log(x/x_median + 1) + shift


@jit
def trans_neglogplus_xshift(x, x_err, params):
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus(x / x_median) + x_shift
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


@jit
def trans_neglogplus_xshift_inv(x_trans, x_err_trans, params):
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans - x_shift) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


@jit
def trans_neglogplus_xshift_dx(x, x_err, params):
    x_median, x_err_median, x_shift = params
    return jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)


@jit
def trans_neglogplus_xshift_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


# ----
@jit
def shrink_gg(x_obs, x_err):
    # Tweedie's formula when p(x_obs) is considered to be a normal distribution
    mu = jnp.mean(x_obs)
    sigma2 = jnp.var(x_obs)
    B = (sigma2 - jnp.square(x_err)) / sigma2
    return B * x_obs + (1 - B) * mu
