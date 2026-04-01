import jax.numpy as jnp
from jax import jit
import numpy as np
import statsmodels.api as sm
from ndtest.ksnd import ks2d2s


@jit
def shrink_gg(x_obs, x_err):
    # Tweedie's formula when p(x_obs) is considered to be a normal distribution
    # gg for Gaussian error - Gaussian distribution
    mu = jnp.mean(x_obs)
    sigma2 = jnp.var(x_obs)
    B = (sigma2 - jnp.square(x_err)) / sigma2
    return B * x_obs + (1 - B) * mu


def shrink_possion(x_obs, x_err):
    # x_err = k * x**1/2 + b + d, d ~ N(0, sigma^2)
    # x_obs = x + e, e ~ N(0, x_err^2)
    # no analytical solution for poisson case
    # let's steal some results from SNR method
    k, b, sigma, mu_x, S_x = solve_SNR(x_obs, x_err)
    # print(mu_x, S_x)

    x_pred_pop = mu_x
    w_pop = 1 / S_x**2

    x_pred_obs = x_obs
    w_obs = 1 / x_err**2

    x_pred_p = (w_pop * x_pred_pop + w_obs * x_pred_obs) / (w_pop + w_obs)

    b, k = regress_b_k_poisson(x_pred_p, x_err)
    # print(f'k={k}, b={b}, sigma={sigma}, mu_x={mu_x}, S_x={S_x}')

    x_pred_err = ((x_err - b) / k)**2
    # std_pred_err = 2 * np.abs((x_err - b) / k) * sigma / k
    # w_err = 1 / std_pred_err**2

    # x_pred = (w_pop * x_pred_pop + w_obs * x_pred_obs +
    #           w_err * x_pred_err) / (w_pop + w_obs + w_err)

    x_err_pred = err_add_noise(x_err, sigma)
    # x_err_pred = x_err

    return x_pred_err, x_err_pred


def regress_b_k_poisson(x_pred, x_err):
    x_pred = np.sqrt(np.clip(np.asarray(x_pred), 0, None))
    x_err = np.asarray(x_err)

    # print(np.clip(np.asarray(x_pred), 0, None))

    X = sm.add_constant(x_pred)
    model = sm.OLS(x_err, X)
    results = model.fit()
    return results.params


def _solve_SNR(mu_err, mu_obs, S_err, S_obs, C_obs_err):
    # x_err = k * x + b + d, d ~ N(0, sigma^2)
    # x_obs = x + e, e ~ N(0, x_err^2)
    # see notes/guess.ipynb for derivation
    k = -C_obs_err / (S_err**2 - S_obs**2 + mu_err**2)
    b = (C_obs_err * mu_obs + S_err**2 * mu_err - S_obs**2 * mu_err +
         mu_err**3) / (S_err**2 - S_obs**2 + mu_err**2)
    sigma_sq = ((C_obs_err**2 + S_err**4 - S_err**2 * S_obs**2 +
                 S_err**2 * mu_err**2) / (S_err**2 - S_obs**2 + mu_err**2))
    sigma = sigma_sq**0.5 if sigma_sq >= 0 else 0.01
    mu_x = mu_obs
    S_x_sq = (-S_err**2 + S_obs**2 - mu_err**2)
    S_x = S_x_sq**0.5 if S_x_sq >= 0 else 0.01

    return k, b, sigma, mu_x, S_x


def solve_SNR(x_obs, x_err):
    mu_err = jnp.mean(x_err)
    mu_obs = jnp.mean(x_obs)
    S_err = jnp.std(x_err)
    S_obs = jnp.std(x_obs)
    C_obs_err = jnp.cov(x_obs, x_err)[0, 1]
    return _solve_SNR(mu_err, mu_obs, S_err, S_obs, C_obs_err)


def shrink_SNR(x_obs, x_err):
    # x_err = k * x + b + d, d ~ N(0, sigma^2)
    # x_obs = x + e, e ~ N(0, x_err^2)

    k, b, sigma, mu_x, S_x = solve_SNR(x_obs, x_err)

    x_pred_pop = mu_x
    w_pop = 1 / S_x**2

    x_pred_obs = x_obs
    w_obs = 1 / x_err**2

    x_pred_err = (x_err - b) / k
    w_err = 1 / (sigma / k)**2

    x_pred = (w_pop * x_pred_pop + w_obs * x_pred_obs +
              w_err * x_pred_err) / (w_pop + w_obs + w_err)

    x_err_pred = err_add_noise(x_err, sigma)

    return x_pred, x_err_pred


def recalibrate_x(x_sh, x_obs):
    x_sh = np.asarray(x_sh)
    x_obs = np.asarray(x_obs)

    X = sm.add_constant(x_sh)
    model = sm.OLS(x_obs, X)
    results = model.fit()
    return results.predict(X)


def shrink_data(x_obs,
                x_err,
                method='gg',
                to_recalibrate=None,
                return_best_info=False,
                verbose=False):

    if method == 'auto':

        if to_recalibrate is not None:
            print(
                'Warning: is_recalibrate is ignored when method is set to auto.'
            )

        x_sh_lst = []
        p_value_lst = []
        info_lst = []
        for m in ['gg', 'poisson', 'SNR']:
            for recal in [True, False]:
                x_sh, x_err_sh = shrink_data(x_obs, x_err, m, recal)
                x_sh_lst.append((x_sh, x_err_sh))
                x_obs_mock = x_sh + np.random.normal(0, x_err_sh)
                p_value_lst.append(
                    ks2d2s(x_obs, x_err, x_obs_mock, x_err_sh)[1])
                info_lst.append(f'{m}, recalibrate={recal}')

        best_idx = np.argmax(p_value_lst)
        if verbose:
            for i, info in enumerate(info_lst):
                print(f'{info}: p-value={p_value_lst[i]}')

            print(
                f'Best method: {info_lst[best_idx]}: p-value={p_value_lst[best_idx]}'
            )

        if return_best_info:
            return *x_sh_lst[best_idx], info_lst[best_idx]

        return x_sh_lst[best_idx]

    is_x_jax_arr = isinstance(x_obs, jnp.ndarray)
    is_x_err_jax_arr = isinstance(x_err, jnp.ndarray)

    if is_x_jax_arr:
        x_obs = np.array(x_obs)
    if is_x_err_jax_arr:
        x_err = np.array(x_err)

    if method == 'gg':
        x_shrink = shrink_gg(x_obs, x_err)
        x_err_shrink = x_err
    elif method == 'poisson':
        x_shrink, x_err_shrink = shrink_possion(x_obs, x_err)
    elif method == 'SNR':
        x_shrink, x_err_shrink = shrink_SNR(x_obs, x_err)
    elif method == 'none':
        x_shrink = x_obs
        x_err_shrink = x_err
    else:
        raise ValueError('Invalid shrinkage method')

    # None and False are equivalent
    if to_recalibrate:
        x_shrink = recalibrate_x(x_shrink, x_obs)


    if is_x_jax_arr:
        x_shrink = jnp.array(x_shrink)
    if is_x_err_jax_arr:
        x_err_shrink = jnp.array(x_err_shrink)

    return x_shrink, x_err_shrink


def err_add_noise(x_err, sigma, maxiter=100):
    x_err_noisy = x_err + np.random.normal(0, sigma, size=x_err.size)
    iteration = 0

    while np.any(x_err_noisy < 0) and iteration < maxiter:
        mask = x_err_noisy < 0
        x_err_noisy[mask] = x_err[mask] + np.random.normal(
            0, sigma, size=np.sum(mask))
        iteration += 1

    if iteration == maxiter:
        print("Max iterations reached before achieving non-negative results.")
        # Optionally reset negative values to a default value (e.g., 0)
        x_err_noisy[x_err_noisy < 0] = 0  # Or other strategies

    return x_err_noisy
