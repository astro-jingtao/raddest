# TODO: quantile transform

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import jit, grad
import pandas as pd
import optax
from pzflow import Flow

from .tools import get_ext_w, neg_log_plus, trans_neglogplus, trans_neglogplus_inv, trans_neglogplus_jac, trans_neglogplus_xshift_inv, trans_xshift_neglogplus_inv


def nf_fit_frame(x_ext,
                 x_err_ext,
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
    stuck_num = 0
    loss = []
    for i in range(n_train):
        params, opt_state = update_params(params, opt_state, x_ext, x_err_ext,
                                          w)
        if i % check_per_step == 0:
            _loss = log_likelihood(params, x_ext, x_err_ext, w)
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
                    if stuck_num > max_stuck:
                        break
                else:
                    stuck_num = 0
                    loss_best = _loss
            loss.append(_loss)
    return params, loss


def construct_nf_fit_func(optimizer, flow, x_min, x_max, x_err_min, x_err_max):
    '''
    x_ext and x_err_ext are transformed value
    x_min, x_max, x_err_min, x_err_max are also for transformed value
    the jacobian of transformation is included in the weight, w
    '''

    @jit
    def gaussian_nf(params, x_ext, x_err_ext, w):
        s1, s2 = x_ext.shape
        _nf_logpdf = nf_logpdf(params, x_ext.flatten(), x_err_ext.flatten())
        return logsumexp(jnp.log(w + 1e-10) + _nf_logpdf.reshape(s1, s2),
                         axis=1)

    @jit
    def nf_logpdf(params, x, x_err):
        return jnp.where((x > x_min) & (x < x_max) & (x_err > x_err_min) &
                         (x_err < x_err_max),
                         _nf_logpdf(params, jnp.clip(x, x_min, x_max),
                                    jnp.clip(x_err, x_err_min,
                                             x_err_max)), -10)

    @jit
    def _nf_logpdf(params, x, x_err):
        return flow._log_prob(params, jnp.c_[x, x_err],
                              jnp.zeros(shape=(x.shape[0], 1)))

    @jit
    def log_likelihood(params, x_ext, x_err_ext, w):
        # to avoid nan
        return -jnp.mean(gaussian_nf(params, x_ext, x_err_ext, w))

    d_log_likelihood = jit(grad(log_likelihood, argnums=(0, )))

    @jit
    def update_params(params, opt_state, x_ext, x_err_ext, w):
        d_params = d_log_likelihood(params, x_ext, x_err_ext, w)[0]
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_params, log_likelihood


def construct_x_xerr_w(x_obs,
                       x_err,
                       x_min,
                       x_max,
                       x_err_min,
                       x_err_max,
                       n_node=21,
                       trans_func=None,
                       trans_func_jac=None,
                       trans_params=None):
    '''
    Here x_obs and x_err are the original value
    x_min, x_max, x_err_min, x_err_max are for transformed value
    '''

    _x_ext, _x_err_ext, w = get_ext_w(n_node, x_obs, x_err)

    if trans_func is None:
        x_ext = _x_ext
        x_err_ext = _x_err_ext
    else:
        x_ext, x_err_ext = trans_func(_x_ext, _x_err_ext, params=trans_params)

    w = jnp.where((x_ext > x_min) & (x_ext < x_max) & (x_err_ext > x_err_min) &
                  (x_err_ext < x_err_max),
                  w * trans_func_jac(_x_ext, _x_err_ext, params=trans_params),
                  0)

    # w = jnp.where(
    #     (_x_ext > 0) & (_x_err_ext > 0),
    #     w * trans_func_jac(_x_ext, _x_err_ext, params=trans_params), 0)

    return x_ext, x_err_ext, w


# TODO: Tweedie's formula
def flow_params_init(x_obs,
                     x_err,
                     flow=None,
                     trans_func=None,
                     trans_params=None,
                     n_pre=100,
                     verbose=False):

    if flow is None:
        flow = Flow(["x_obs", "x_err"])
        if trans_func is None:
            data_pd = pd.DataFrame({'x_obs': x_obs, 'x_err': x_err})
        else:
            x_obs_trans, x_err_trans = trans_func(x_obs,
                                                  x_err,
                                                  params=trans_params)
            data_pd = pd.DataFrame({
                'x_obs': x_obs_trans,
                'x_err': x_err_trans
            })

        _ = flow.train(data_pd,
                       convolve_errs=False,
                       epochs=n_pre,
                       progress_bar=verbose)

    params = [flow._params[0], flow._params[1]]

    return flow, params


# TODO: give trans_params
def nf_fit_trans_neglogplus(x_obs,
                            x_err,
                            *,
                            optimizer,
                            n_pre=100,
                            n_train=500,
                            n_node=21,
                            verbose=False,
                            flow=None,
                            x_min=0,
                            x_max=15,
                            x_err_min=0,
                            x_err_max=5,
                            check_per_step=10,
                            early_stop=True,
                            max_stuck=5,
                            stuck_frac=0.05):

    x_median = np.median(x_obs)
    x_err_median = np.median(x_err)

    trans_params = [x_median, x_err_median]

    x_min_trans, x_err_min_trans = trans_neglogplus(x_min,
                                                    x_err_min,
                                                    params=trans_params)
    x_max_trans, x_err_max_trans = trans_neglogplus(x_max,
                                                    x_err_max,
                                                    params=trans_params)

    x_ext, x_err_ext, w = construct_x_xerr_w(
        x_obs,
        x_err,
        x_min_trans,
        x_max_trans,
        x_err_min_trans,
        x_err_max_trans,
        n_node=n_node,
        trans_func=trans_neglogplus,
        trans_func_jac=trans_neglogplus_jac,
        trans_params=trans_params)

    flow, params = flow_params_init(x_obs,
                                    x_err,
                                    flow=flow,
                                    trans_func=trans_neglogplus,
                                    trans_params=trans_params,
                                    n_pre=n_pre,
                                    verbose=verbose)

    update_params, log_likelihood = construct_nf_fit_func(
        optimizer, flow, x_min_trans, x_max_trans, x_err_min_trans,
        x_err_max_trans)

    params, loss = nf_fit_frame(x_ext,
                                x_err_ext,
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

    return flow, loss, trans_params


def gen_xobs_trans(flow,
                   N,
                   x_min,
                   x_max,
                   x_err_min,
                   x_err_max,
                   trans_func_inv=None,
                   trans_params=None,
                   return_numpy=False):
    '''
    Here x_min, x_max, xerr_min, xerr_max are for original varible (x, x_err)
    '''

    samples = flow.sample(2 * N, seed=0)
    x_all = samples['x_obs'].to_numpy()
    x_err_all = samples['x_err'].to_numpy()

    if trans_func_inv is None:
        x, x_err = x_all, x_err_all
    else:
        x, x_err = trans_func_inv(x_all, x_err_all, trans_params)

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


def gen_xobs_trans_neglogplus(flow,
                              N,
                              x_min,
                              x_max,
                              x_err_min,
                              x_err_max,
                              trans_params,
                              return_numpy=False):
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          trans_func_inv=trans_neglogplus_inv,
                          trans_params=trans_params,
                          return_numpy=return_numpy)


def gen_xobs_trans_neglogplus_shift(flow,
                                    N,
                                    x_min,
                                    x_max,
                                    x_err_min,
                                    x_err_max,
                                    trans_params,
                                    return_numpy=False):
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          trans_func_inv=trans_neglogplus_xshift_inv,
                          trans_params=trans_params,
                          return_numpy=return_numpy)


def gen_xobs_trans_shift_neglogplus(flow,
                                    N,
                                    x_min,
                                    x_max,
                                    x_err_min,
                                    x_err_max,
                                    trans_params,
                                    return_numpy=False):
    return gen_xobs_trans(flow,
                          N,
                          x_min,
                          x_max,
                          x_err_min,
                          x_err_max,
                          trans_func_inv=trans_xshift_neglogplus_inv,
                          trans_params=trans_params,
                          return_numpy=return_numpy)
