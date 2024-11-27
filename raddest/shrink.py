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


@jit
def shrink_possion(x_obs, x_err):
    k = x_err**2 / x_obs
    k_median = jnp.median(k)
    x_pred_median = x_err**2 / k_median
    return x_pred_median


@jit
def shrink_SNR(x_obs, x_err):
    k = x_err / x_obs
    k_median = jnp.median(k)
    x_pred_median = x_err / k_median
    return x_pred_median


def recalibrate(x_pred, x_obs):
    x_pred = np.asarray(x_pred)
    x_obs = np.asarray(x_obs)
    
    X = sm.add_constant(x_pred)
    model = sm.OLS(x_obs, X)
    results = model.fit()
    return results.predict(X)


def shrink_data(x_obs,
                x_err,
                method='auto',
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
                x_sh = shrink_data(x_obs, x_err, m, recal)
                x_sh_lst.append(x_sh)
                x_obs_mock = x_sh + np.random.normal(0, x_err)
                p_value_lst.append(ks2d2s(x_obs, x_err, x_obs_mock, x_err)[1])
                info_lst.append(f'{m}, recalibrate={recal}')

        best_idx = np.argmax(p_value_lst)
        if verbose:
            for i, info in enumerate(info_lst):
                print(f'{info}: p-value={p_value_lst[i]}')

            print(
                f'Best method: {info_lst[best_idx]}: p-value={p_value_lst[best_idx]}'
            )

        if return_best_info:
            return x_sh_lst[best_idx], info_lst[best_idx]

        return x_sh_lst[best_idx]

    if method == 'gg':
        x_shrink = shrink_gg(x_obs, x_err)
    elif method == 'poisson':
        x_shrink = shrink_possion(x_obs, x_err)
    elif method == 'SNR':
        x_shrink = shrink_SNR(x_obs, x_err)
    elif method == 'none':
        x_shrink = x_obs
    else:
        raise ValueError('Invalid shrinkage method')

    # None and False are equivalent
    if to_recalibrate:
        x_shrink = recalibrate(x_shrink, x_obs)

    return x_shrink
