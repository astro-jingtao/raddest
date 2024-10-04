# TODO: quantile transform

from typing import Union

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import grad, jit
from jax.scipy.special import logsumexp
from pzflow import Flow

from .tools import (get_ext_w, get_ext_w_qmc, shrink_gg)
from .transfer import Transfer, neglogplus, log_rotation, xshift_neglogplus, neglogplus_xshift


def nf_fit_frame(u_ext,
                 v_ext,
                 w,
                 params,
                 optimizer,
                 update_params,
                 log_likelihood,
                 n_train=1000,
                 check_per_step=100,
                 verbose=False,
                 early_stop=True,
                 max_stuck=10,
                 stuck_frac=0.01):

    opt_state = optimizer.init(params)

    loss_best = jnp.inf
    params_best = params.copy()
    stuck_num = 0
    loss = []
    for i in range(n_train):
        params, opt_state = update_params(params, opt_state, u_ext, v_ext, w)
        if i % check_per_step == 0:
            _loss = log_likelihood(params, u_ext, v_ext, w)
            if verbose:
                print(f"step {i}: loss {_loss}")
            if early_stop:
                if _loss >= loss_best:
                    stuck_num += 1
                    if stuck_num > max_stuck:
                        break
                elif jnp.abs(_loss - loss_best) < jnp.abs(stuck_frac *
                                                          loss_best):
                    stuck_num += 1
                    loss_best = _loss
                    params_best = params.copy()
                    if stuck_num > max_stuck:
                        break
                else:
                    stuck_num = 0
                    loss_best = _loss
                    params_best = params.copy()

            loss.append(_loss)
    return params_best, loss


def find_uv_bounds(_bijector_info):
    if _bijector_info[0] == 'Chain':
        for tup in _bijector_info[1]:
            result = find_uv_bounds(tup)
            if result is not None:
                return result
        return None  # If do not find bounds in chain，return None
    elif _bijector_info[0] == 'ShiftBounds':
        return (_bijector_info[1][0][0], _bijector_info[1][1][0],
                _bijector_info[1][0][1], _bijector_info[1][1][1])
    else:
        return None  # If not 'Chain' or 'ShiftBounds'，return None


def construct_nf_fit_func(optimizer, flow):
    '''
    u_ext and v_ext are transformed value
    x_min, x_max, x_err_min, x_err_max are for original value
    the jacobian of transformation is included in the weight, w
    '''

    uv_bounds = find_uv_bounds(flow._bijector_info)

    if uv_bounds is None:
        raise ValueError("Cannot find bounds in bijector")

    u_min, u_max, v_min, v_max = uv_bounds

    @jit
    def gaussian_nf(params, u_ext, v_ext, w):
        s1, s2 = u_ext.shape
        _nf_logpdf = nf_logpdf(params, u_ext.flatten(), v_ext.flatten())
        cond = w > 0
        return logsumexp(jnp.log(jnp.where(cond, w, 1e-10)) +
                         jnp.where(cond, _nf_logpdf.reshape(s1, s2), -10),
                         axis=1)

    @jit
    def nf_logpdf(params, u, v):
        # we need inner where
        # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
        cond = (u > u_min) & (u < u_max) & (v > v_min) & (v < v_max)
        return jnp.where(
            cond,
            _nf_logpdf(params, jnp.clip(u, u_min, u_max),
                       jnp.clip(v, v_min, v_max)), -10)

    @jit
    def _nf_logpdf(params, u, v):
        return flow._log_prob(params, jnp.c_[u, v],
                              jnp.zeros(shape=(u.shape[0], 1)))

    @jit
    def log_likelihood(params, u_ext, v_ext, w):
        return -jnp.mean(gaussian_nf(params, u_ext, v_ext, w))

    d_log_likelihood = jit(grad(log_likelihood, argnums=(0, )))

    @jit
    def update_params(params, opt_state, u_ext, v_ext, w):
        d_params = d_log_likelihood(params, u_ext, v_ext, w)[0]
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_params, log_likelihood


def construct_x_xerr_w(x_obs,
                       x_err,
                       n_node=21,
                       qmc=False,
                       transfer: Union[Transfer, None] = None,
                       transfer_params=None):
    '''
    Here x_obs and x_err are the original value
    x_min, x_max, x_err_min, x_err_max are for original value
    '''

    if qmc:
        _x_ext, _x_err_ext, w = get_ext_w_qmc(n_node, x_obs, x_err)
    else:
        _x_ext, _x_err_ext, w = get_ext_w(n_node, x_obs, x_err)

    if transfer is None:
        u_ext = _x_ext
        v_ext = _x_err_ext
    else:
        u_ext, v_ext = transfer.trans(_x_ext,
                                      _x_err_ext,
                                      params=transfer_params)
        w = w * transfer.jac(_x_ext, _x_err_ext, params=transfer_params)
        w = jnp.nan_to_num(w, nan=0)
        u_ext = jnp.nan_to_num(u_ext, nan=0)
        v_ext = jnp.nan_to_num(v_ext, nan=0)

    return u_ext, v_ext, w


# TODO: Tweedie's formula
def flow_params_init(x_obs,
                     x_err,
                     x_min=0,
                     x_max=jnp.inf,
                     x_err_min=0,
                     x_err_max=jnp.inf,
                     flow=None,
                     transfer: Union[Transfer, None] = None,
                     transfer_params=None,
                     n_pre=100,
                     verbose=False):

    if flow is None:
        flow = Flow(["x_obs", "x_err"])
        x_obs_shrink = shrink_gg(x_obs, x_err)

        is_in_range = (x_obs_shrink >= x_min) & (x_obs_shrink <= x_max) & (
            x_err >= x_err_min) & (x_err <= x_err_max)

        if transfer is None:
            data_pd = pd.DataFrame({
                'x_obs': x_obs_shrink[is_in_range],
                'x_err': x_err[is_in_range]
            })
        else:
            u, v = transfer.trans(x_obs_shrink[is_in_range],
                                  x_err[is_in_range],
                                  params=transfer_params)
            data_pd = pd.DataFrame({'x_obs': u, 'x_err': v})

        _ = flow.train(data_pd,
                       convolve_errs=False,
                       epochs=n_pre,
                       progress_bar=verbose)

    params = [flow._params[0], flow._params[1]]

    return flow, params


# TODO: give trans_params
def nf_fit_trans_neglogplus(x_obs, x_err, **kwargs):

    x_median = np.median(x_obs)
    x_err_median = np.median(x_err)

    transfer_params = [x_median, x_err_median]

    return nf_fit_general(x_obs,
                          x_err,
                          transfer=neglogplus,
                          transfer_params=transfer_params,
                          **kwargs)


def nf_fit_trans_log_rotation(x_obs, x_err, *, theta, **kwargs):

    transfer_params = theta

    return nf_fit_general(x_obs,
                          x_err,
                          transfer=log_rotation,
                          transfer_params=transfer_params,
                          **kwargs)


def nf_fit_general(x_obs,
                   x_err,
                   *,
                   optimizer,
                   n_pre=100,
                   n_train=500,
                   n_node=21,
                   qmc=False,
                   verbose=False,
                   flow=None,
                   x_min=0,
                   x_max=jnp.inf,
                   x_err_min=0,
                   x_err_max=jnp.inf,
                   check_per_step=10,
                   early_stop=True,
                   max_stuck=5,
                   stuck_frac=0.05,
                   transfer: Union[Transfer, None] = None,
                   transfer_params=None):

    u_ext, v_ext, w = construct_x_xerr_w(x_obs,
                                         x_err,
                                         n_node=n_node,
                                         qmc=qmc,
                                         transfer=transfer,
                                         transfer_params=transfer_params)

    flow, params = flow_params_init(x_obs,
                                    x_err,
                                    x_min=x_min,
                                    x_max=x_max,
                                    x_err_min=x_err_min,
                                    x_err_max=x_err_max,
                                    flow=flow,
                                    transfer=transfer,
                                    transfer_params=transfer_params,
                                    n_pre=n_pre,
                                    verbose=verbose)

    update_params, log_likelihood = construct_nf_fit_func(optimizer, flow)

    params, loss = nf_fit_frame(u_ext,
                                v_ext,
                                w,
                                params,
                                optimizer,
                                update_params,
                                log_likelihood,
                                n_train=n_train,
                                check_per_step=check_per_step,
                                verbose=verbose,
                                early_stop=early_stop,
                                max_stuck=max_stuck,
                                stuck_frac=stuck_frac)

    flow._params = params

    return flow, loss, transfer_params


def gen_xobs_trans(flow,
                   N,
                   x_min,
                   x_max,
                   x_err_min,
                   x_err_max,
                   transfer: Union[Transfer, None] = None,
                   transfer_params=None,
                   return_numpy=False):
    '''
    Here x_min, x_max, xerr_min, xerr_max are for original varible (x, x_err)
    '''

    samples = flow.sample(2 * N, seed=0)
    u_all = samples['x_obs'].to_numpy()
    v_all = samples['x_err'].to_numpy()

    if transfer is None:
        x, x_err = u_all, v_all
    else:
        x, x_err = transfer.inv(u_all, v_all, transfer_params)

    # We can have this because this is used to fit theta
    # The normalization constant do not affect results
    is_in_range = (x > x_min) & (x < x_max) & (x_err > x_err_min) & (
        x_err < x_err_max)

    x = x[is_in_range][:N]
    x_err = x_err[is_in_range][:N]

    x_obs = x + np.random.normal(0, x_err)

    if return_numpy:
        x = np.array(x)
        x_err = np.array(x_err)
        x_obs = np.array(x_obs)

    return x, x_err, x_obs


def gen_xobs_trans_log_rotation(flow,
                                N,
                                x_min,
                                x_max,
                                x_err_min,
                                x_err_max,
                                transfer_params,
                                return_numpy=False):
    '''
    return:
        x_gen, x_err_gen, x_obs_gen
    '''
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          transfer=log_rotation,
                          transfer_params=transfer_params,
                          return_numpy=return_numpy)


def gen_xobs_trans_neglogplus(flow,
                              N,
                              x_min,
                              x_max,
                              x_err_min,
                              x_err_max,
                              transfer_params,
                              return_numpy=False):
    '''
    return:
        x_gen, x_err_gen, x_obs_gen
    '''
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          transfer=neglogplus,
                          transfer_params=transfer_params,
                          return_numpy=return_numpy)


def gen_xobs_trans_neglogplus_xshift(flow,
                                     N,
                                     x_min,
                                     x_max,
                                     x_err_min,
                                     x_err_max,
                                     transfer_params,
                                     return_numpy=False):
    # neglogplus_xshift
    # log(x/x_median + 1) + shift
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          transfer=neglogplus_xshift,
                          transfer_params=transfer_params,
                          return_numpy=return_numpy)


def gen_xobs_trans_xshift_neglogplus(flow,
                                     N,
                                     x_min,
                                     x_max,
                                     x_err_min,
                                     x_err_max,
                                     transfer_params,
                                     return_numpy=False):
    # xshift_neglogplus
    # log((x + shift)/x_median + 1)
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          transfer=xshift_neglogplus,
                          transfer_params=transfer_params,
                          return_numpy=return_numpy)
