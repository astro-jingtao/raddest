from functools import partial

import jax.numpy as jnp
import joblib
from jax import jit


class Transfer:

    def __init__(self,
                 trans,
                 inv,
                 jac,
                 n_params,
                 params_estimator=None,
                 name=None):
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
    def load(cls, path):
        return cls(**joblib.load(path))

    def save(self, path):
        joblib.dump(self.dumpable_core, path)


def NegLogPlus():
    return Transfer(_trans_neglogplus,
                    _trans_neglogplus_inv,
                    _trans_neglogplus_jac,
                    2,
                    params_estimator=_trans_neglogplus_estimator,
                    name='NegLogPlus')


def LogRotation():
    return Transfer(_trans_log_rotation,
                    _trans_log_rotation_inv,
                    _trans_log_rotation_jac,
                    1,
                    name='LogRotation')


def XShift():
    return Transfer(_trans_xshift,
                    _trans_xshift_inv,
                    _trans_xshift_jac,
                    1,
                    name='XShift')


def XShiftNegLogPlus():
    return Transfer(_trans_xshift_neglogplus,
                    _trans_xshift_neglogplus_inv,
                    _trans_xshift_neglogplus_jac,
                    3,
                    name='XShiftNegLogPlus')


def NegLogPlusXShift():
    return Transfer(_trans_neglogplus_xshift,
                    _trans_neglogplus_xshift_inv,
                    _trans_neglogplus_xshift_jac,
                    3,
                    name='NegLogPlusXShift')


# -- neglogplus --


def _trans_neglogplus(x, x_err, params):
    x_median, x_err_median = params
    x_trans = neg_log_plus(x / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_neglogplus = jit(_trans_neglogplus)


def _trans_neglogplus_inv(x_trans, x_err_trans, params):
    x_median, x_err_median = params
    x = neg_log_plus_inv(x_trans) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_neglogplus_inv = jit(_trans_neglogplus_inv)


def _trans_neglogplus_jac(x, x_err, params):
    x_median, x_err_median = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


trans_neglogplus_jac = jit(_trans_neglogplus_jac)


def _trans_neglogplus_estimator(x, x_err):
    x_median = jnp.median(x)
    x_err_median = jnp.median(x_err)
    return x_median, x_err_median


trans_neglogplus_estimator = jit(_trans_neglogplus_estimator)

# -- log_rotation --

# u = cos * log(x) - sin * log(x_err)
# v = sin * log(x) + cos * log(x_err)


def _trans_log_rotation(x, x_err, params):
    theta, = jnp.atleast_1d(jnp.asarray(params))
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    log_x = jnp.log(x)
    log_x_err = jnp.log(x_err)
    u = cos * log_x - sin * log_x_err
    v = sin * log_x + cos * log_x_err
    return u, v


trans_log_rotation = jit(_trans_log_rotation)


def _trans_log_rotation_inv(u, v, params):
    theta, = jnp.atleast_1d(jnp.asarray(params))
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    x = jnp.exp(u * cos + v * sin)
    x_err = jnp.exp(-u * sin + v * cos)
    return x, x_err


trans_log_rotation_inv = jit(_trans_log_rotation_inv)


def _trans_log_rotation_jac(x, x_err, params):
    return 1 / (x * x_err)


trans_log_rotation_jac = jit(_trans_log_rotation_jac)

# -- xshift --
# u = x + shift
# v = x_err


def _trans_xshift(x, x_err, params):
    x_shift, = jnp.atleast_1d(jnp.asarray(params))
    u = x + x_shift
    v = x_err
    return u, v


trans_xshift = jit(_trans_xshift)


def _trans_xshift_inv(u, v, params):
    x_shift, = jnp.atleast_1d(jnp.asarray(params))
    x = u - x_shift
    x_err = v
    return x, x_err


trans_xshift_inv = jit(_trans_xshift_inv)


def _trans_xshift_jac(x, x_err, params):
    return 1


trans_xshift_jac = jit(_trans_xshift_jac)

# -- xshift_neglogplus --

# log((x + shift)/x_median + 1)


def _trans_xshift_neglogplus(x, x_err, params):
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus((x + x_shift) / x_median)
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_xshift_neglogplus = jit(_trans_xshift_neglogplus)


def _trans_xshift_neglogplus_inv(x_trans, x_err_trans, params):
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans) * x_median - x_shift
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_xshift_neglogplus_inv = jit(_trans_xshift_neglogplus_inv)


def _trans_xshift_neglogplus_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
    k_x = jnp.where((x + x_shift) / x_median > 0, 1 / (x + x_shift + x_median),
                    1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


trans_xshift_neglogplus_jac = jit(_trans_xshift_neglogplus_jac)

# -- neglogplus_xshift --

# log(x/x_median + 1) + shift


def _trans_neglogplus_xshift(x, x_err, params):
    x_median, x_err_median, x_shift = params
    x_trans = neg_log_plus(x / x_median) + x_shift
    x_err_trans = neg_log_plus(x_err / x_err_median)
    return x_trans, x_err_trans


trans_neglogplus_xshift = jit(_trans_neglogplus_xshift)


def _trans_neglogplus_xshift_inv(x_trans, x_err_trans, params):
    x_median, x_err_median, x_shift = params
    x = neg_log_plus_inv(x_trans - x_shift) * x_median
    x_err = neg_log_plus_inv(x_err_trans) * x_err_median
    return x, x_err


trans_neglogplus_xshift_inv = jit(_trans_neglogplus_xshift_inv)


def _trans_neglogplus_xshift_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
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


def _trans_combined(x, x_err, params, trans_1, trans_2, n_1):
    u1, v1 = trans_1(x, x_err, params[:n_1])
    u2, v2 = trans_2(u1, v1, params[n_1:])
    return u2, v2


def _inv_combined(u2, v2, params, inv_1, inv_2, n_1):
    u1, v1 = inv_2(u2, v2, params[n_1:])
    x, x_err = inv_1(u1, v1, params[:n_1])
    return x, x_err


def _jac_combined(x, x_err, params, jac_1, jac_2, trans_1, n_1):
    jac1 = jac_1(x, x_err, params[:n_1])
    u1, v1 = trans_1(x, x_err, params[:n_1])
    jac2 = jac_2(u1, v1, params[n_1:])
    return jac1 * jac2


def _params_estimator_combined(x, x_err, est_1, est_2, trans_1):
    p1 = est_1(x, x_err)
    u1, v1 = trans_1(x, x_err, p1)
    p2 = est_2(u1, v1)
    return jnp.concatenate([p1, p2])


def combine_transfers_binary(transfer1, transfer2):

    if transfer1 is None:
        return transfer2
    elif transfer2 is None:
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


def combine_transfers(*args):
    if len(args) == 1:
        return args[0]
    else:
        return combine_transfers_binary(args[0], combine_transfers(*args[1:]))

# functions

@jit
def neg_log_plus(x):
    return jnp.where(x > 0, jnp.log(x + 1), x)


@jit
def neg_log_plus_inv(x):
    return jnp.where(x > 0, jnp.exp(x) - 1, x)