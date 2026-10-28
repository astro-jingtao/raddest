"""Variable transforms for the (x, x_err) plane: the Transfer wrapper plus
the neglogplus / log-rotation / xshift families and their composition."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
from jax import jit
import jax.numpy as jnp
import joblib


class Transfer:
    """A paired variable transform: forward, inverse, jacobian, and an
    optional data-driven parameter estimator, each wrapped with jit."""

    def __init__(self,
                 trans: Callable,
                 inv: Callable,
                 jac: Callable,
                 n_params: int,
                 *,
                 params_estimator: Callable | None = None,
                 name: str | None = None) -> None:
        '''Bind the four callables; params_estimator and name are optional.'''
        self.dumpable_core = {
            'trans': trans,
            'inv': inv,
            'jac': jac,
            'n_params': n_params,
            'params_estimator': params_estimator
        }
        self.trans = jit(trans)
        self.inv = jit(inv)
        self.jac = jit(jac)
        self.n_params = n_params
        if params_estimator is None:
            self.params_estimator = None
        else:
            self.params_estimator = jit(params_estimator)

        if name is None:
            self.name = 'Unnamed'
        else:
            self.name = name

    @classmethod
    def load(cls, path: str) -> Transfer:
        '''Rebuild a saved Transfer from a joblib dump of its core.'''
        return cls(**joblib.load(path))

    def save(self, path: str) -> None:
        '''Joblib-dump the un-jitted core (trans/inv/jac/estimator/n_params).'''
        joblib.dump(self.dumpable_core, path)


def NegLogPlus() -> Transfer:
    '''neglogplus on both axes; params = (x_median, x_err_median).'''
    return Transfer(_trans_neglogplus,
                    _trans_neglogplus_inv,
                    _trans_neglogplus_jac,
                    2,
                    params_estimator=_trans_neglogplus_estimator,
                    name='NegLogPlus')


def LogRotation() -> Transfer:
    '''Rotate (log x, log x_err) by theta; params = (theta,).'''
    return Transfer(_trans_log_rotation,
                    _trans_log_rotation_inv,
                    _trans_log_rotation_jac,
                    1,
                    name='LogRotation')


def XShift() -> Transfer:
    '''Shift x by a constant; params = (x_shift,).'''
    return Transfer(_trans_xshift,
                    _trans_xshift_inv,
                    _trans_xshift_jac,
                    1,
                    name='XShift')


def XShiftNegLogPlus() -> Transfer:
    '''log((x + shift)/x_median + 1); params = (x_median, x_err_median, shift).'''
    return Transfer(_trans_xshift_neglogplus,
                    _trans_xshift_neglogplus_inv,
                    _trans_xshift_neglogplus_jac,
                    3,
                    name='XShiftNegLogPlus')


def NegLogPlusXShift() -> Transfer:
    '''log(x/x_median + 1) + shift; params = (x_median, x_err_median, shift).'''
    return Transfer(_trans_neglogplus_xshift,
                    _trans_neglogplus_xshift_inv,
                    _trans_neglogplus_xshift_jac,
                    3,
                    name='NegLogPlusXShift')


# -- neglogplus --


def _trans_neglogplus(x: jax.Array, x_err: jax.Array,
                      params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median = params
    x_trans = neg_log_plus(x / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_neglogplus = jit(_trans_neglogplus)


def _trans_neglogplus_inv(x_trans: jax.Array, x_err_trans: jax.Array,
                          params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median = params
    x = neg_log_plus_inv(x_trans) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_neglogplus_inv = jit(_trans_neglogplus_inv)


def _trans_neglogplus_jac(x: jax.Array, x_err: jax.Array,
                          params: jax.Array) -> jax.Array:
    x_median, x_err_median = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


trans_neglogplus_jac = jit(_trans_neglogplus_jac)


def _trans_neglogplus_estimator(
        x: jax.Array,
        x_err: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median = jnp.median(x)
    x_err_median = jnp.median(x_err)
    return x_median, x_err_median


trans_neglogplus_estimator = jit(_trans_neglogplus_estimator)

# -- log_rotation --

# u = cos * log(x) - sin * log(x_err)
# v = sin * log(x) + cos * log(x_err)


def _trans_log_rotation(x: jax.Array, x_err: jax.Array,
                        params: jax.Array) -> tuple[jax.Array, jax.Array]:
    theta, = jnp.atleast_1d(jnp.asarray(params))
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    log_x = jnp.log(x)
    log_x_err = jnp.log(x_err)
    u = cos * log_x - sin * log_x_err
    v = sin * log_x + cos * log_x_err
    return u, v


trans_log_rotation = jit(_trans_log_rotation)


def _trans_log_rotation_inv(u: jax.Array, v: jax.Array,
                            params: jax.Array) -> tuple[jax.Array, jax.Array]:
    theta, = jnp.atleast_1d(jnp.asarray(params))
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    x = jnp.exp(u * cos + v * sin)
    x_err = jnp.exp(-u * sin + v * cos)
    return x, x_err


trans_log_rotation_inv = jit(_trans_log_rotation_inv)


def _trans_log_rotation_jac(x: jax.Array, x_err: jax.Array,
                            params: jax.Array) -> jax.Array:
    return 1 / (x * x_err)


trans_log_rotation_jac = jit(_trans_log_rotation_jac)

# -- xshift --
# u = x + shift
# v = x_err


def _trans_xshift(x: jax.Array, x_err: jax.Array,
                  params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_shift, = jnp.atleast_1d(jnp.asarray(params))
    u = x + x_shift
    v = x_err
    return u, v


trans_xshift = jit(_trans_xshift)


def _trans_xshift_inv(u: jax.Array, v: jax.Array,
                      params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_shift, = jnp.atleast_1d(jnp.asarray(params))
    x = u - x_shift
    x_err = v
    return x, x_err


trans_xshift_inv = jit(_trans_xshift_inv)


def _trans_xshift_jac(x: jax.Array, x_err: jax.Array,
                      params: jax.Array) -> jax.Array:
    return 1


trans_xshift_jac = jit(_trans_xshift_jac)

# -- xshift_neglogplus --

# log((x + shift)/x_median + 1)


def _trans_xshift_neglogplus(
        x: jax.Array, x_err: jax.Array,
        params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus((x + x_shift) / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_xshift_neglogplus = jit(_trans_xshift_neglogplus)


def _trans_xshift_neglogplus_inv(
        x_trans: jax.Array, x_err_trans: jax.Array,
        params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans) * x_median - x_shift
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_xshift_neglogplus_inv = jit(_trans_xshift_neglogplus_inv)


def _trans_xshift_neglogplus_jac(
        x: jax.Array, x_err: jax.Array, params: jax.Array) -> jax.Array:
    x_median, x_err_median, x_shift = params
    k_x = jnp.where((x + x_shift) / x_median > 0, 1 / (x + x_shift + x_median),
                    1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


trans_xshift_neglogplus_jac = jit(_trans_xshift_neglogplus_jac)

# -- neglogplus_xshift --

# log(x/x_median + 1) + shift


def _trans_neglogplus_xshift(
        x: jax.Array, x_err: jax.Array,
        params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus(x / x_median) + x_shift
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_neglogplus_xshift = jit(_trans_neglogplus_xshift)


def _trans_neglogplus_xshift_inv(
        x_trans: jax.Array, x_err_trans: jax.Array,
        params: jax.Array) -> tuple[jax.Array, jax.Array]:
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans - x_shift) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_neglogplus_xshift_inv = jit(_trans_neglogplus_xshift_inv)


def _trans_neglogplus_xshift_jac(
        x: jax.Array, x_err: jax.Array, params: jax.Array) -> jax.Array:
    x_median, x_err_median, _ = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


trans_neglogplus_xshift_jac = jit(_trans_neglogplus_xshift_jac)

# -- cdf --
# u = cdf(x), v = cdf(x_err)

# @jit
# def trans_cdf(x, x_err, params):
#     n_node_x = params[-2]
#     n_node_x_err = params[-1]
#     coef_x = params[:n_node_x * 3].reshape(3, -1)
#     node_x = params[n_node_x * 3:n_node_x * 4]
#     coef_x_err = params[n_node_x * 4:n_node_x * 4 + n_node_x_err * 3].reshape(
#         3, -1)
#     node_x_err = params[n_node_x * 4 + n_node_x_err * 3:n_node_x * 4 +
#                         n_node_x_err * 4]

#     PPoly(coef_x, node_x)(x)

#     x_err_trans = neg_log_plus(x_err / x_err_median)
#     return x_trans, x_err_trans


def _trans_combined(x: jax.Array,
                    x_err: jax.Array,
                    params: jax.Array,
                    *,
                    trans_1: Callable,
                    trans_2: Callable,
                    n_1: int) -> tuple[jax.Array, jax.Array]:
    '''Apply trans_1 then trans_2, splitting params at n_1.'''
    u1, v1 = trans_1(x, x_err, params[:n_1])
    u2, v2 = trans_2(u1, v1, params[n_1:])
    return u2, v2


def _inv_combined(u2: jax.Array,
                  v2: jax.Array,
                  params: jax.Array,
                  *,
                  inv_1: Callable,
                  inv_2: Callable,
                  n_1: int) -> tuple[jax.Array, jax.Array]:
    '''Invert trans_2 then trans_1, splitting params at n_1.'''
    u1, v1 = inv_2(u2, v2, params[n_1:])
    x, x_err = inv_1(u1, v1, params[:n_1])
    return x, x_err


def _jac_combined(x: jax.Array,
                  x_err: jax.Array,
                  params: jax.Array,
                  *,
                  jac_1: Callable,
                  jac_2: Callable,
                  trans_1: Callable,
                  n_1: int) -> jax.Array:
    '''Product of the two jacobians, evaluating jac_2 after trans_1.'''
    jac1 = jac_1(x, x_err, params[:n_1])
    u1, v1 = trans_1(x, x_err, params[:n_1])
    jac2 = jac_2(u1, v1, params[n_1:])
    return jac1 * jac2


def _params_estimator_combined(
        x: jax.Array,
        x_err: jax.Array,
        *,
        est_1: Callable,
        est_2: Callable,
        trans_1: Callable) -> jax.Array:
    '''Estimate params of transfer 1, then of transfer 2 on the transformed
    data; returns the concatenated parameter vector.'''
    p1 = jnp.asarray(est_1(x, x_err))
    u1, v1 = trans_1(x, x_err, p1)
    p2 = jnp.asarray(est_2(u1, v1))
    return jnp.concatenate([p1, p2])


def combine_transfers_binary(
    transfer1: Transfer | None,
    transfer2: Transfer | None,
) -> Transfer | None:
    '''Compose two (optional) transfers into one Transfer.'''

    if transfer1 is None:
        return transfer2
    if transfer2 is None:
        return transfer1

    n_1 = transfer1.n_params
    n_2 = transfer2.n_params
    n_tot = n_1 + n_2

    _trans = partial(_trans_combined,
                     trans_1=transfer1.dumpable_core['trans'],
                     trans_2=transfer2.dumpable_core['trans'],
                     n_1=n_1)

    _inv = partial(_inv_combined,
                   inv_1=transfer1.dumpable_core['inv'],
                   inv_2=transfer2.dumpable_core['inv'],
                   n_1=n_1)

    _jac = partial(_jac_combined,
                   jac_1=transfer1.dumpable_core['jac'],
                   jac_2=transfer2.dumpable_core['jac'],
                   trans_1=transfer1.dumpable_core['trans'],
                   n_1=n_1)

    if transfer1.dumpable_core['params_estimator'] and transfer2.dumpable_core[
            'params_estimator']:

        _params_estimator = partial(
            _params_estimator_combined,
            est_1=transfer1.dumpable_core['params_estimator'],
            est_2=transfer2.dumpable_core['params_estimator'],
            trans_1=transfer1.dumpable_core['trans'])
    else:
        _params_estimator = None

    name = f'{transfer1.name} + {transfer2.name}'

    return Transfer(_trans,
                    _inv,
                    _jac,
                    n_tot,
                    params_estimator=_params_estimator,
                    name=name)


def combine_transfers(*args: Transfer | None) -> Transfer | None:
    '''Left-fold combine_transfers_binary over the given transfers.'''
    if len(args) == 1:
        return args[0]
    return combine_transfers_binary(args[0], combine_transfers(*args[1:]))

# functions


@jit
def neg_log_plus(x: jax.Array) -> jax.Array:
    '''log(x + 1) for x > 0, identity otherwise (smooth at 0).'''
    return jnp.where(x > 0, jnp.log(x + 1), x)


@jit
def neg_log_plus_inv(x: jax.Array) -> jax.Array:
    '''exp(x) - 1 for x > 0, identity otherwise (inverse of neg_log_plus).'''
    return jnp.where(x > 0, jnp.exp(x) - 1, x)
