import jax.numpy as jnp
import joblib
from jax import jit

from .tools import neg_log_plus, neg_log_plus_inv


class Transfer:

    def __init__(self, trans, inv, jac, n_params, params_estimator=None):
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
        self.params_estimator = jit(params_estimator)

    @classmethod
    def load(cls, path):
        return cls(**joblib.load(path))

    def save(self, path):
        joblib.dump(self.dumpable_core, path)


def NegLogPlus():
    return Transfer(_trans_neglogplus, _trans_neglogplus_inv,
                    _trans_neglogplus_jac, 2, _trans_neglogplus_estimator)


def LogRotation():
    return Transfer(_trans_log_rotation, _trans_log_rotation_inv,
                    _trans_log_rotation_jac, 1)


def XShiftNegLogPlus():
    return Transfer(_trans_xshift_neglogplus, _trans_xshift_neglogplus_inv,
                    _trans_xshift_neglogplus_jac, 3)


def NegLogPlusXShift():
    return Transfer(_trans_neglogplus_xshift, _trans_neglogplus_xshift_inv,
                    _trans_neglogplus_xshift_jac, 3)


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
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    log_x = jnp.log(x)
    log_x_err = jnp.log(x_err)
    u = cos * log_x - sin * log_x_err
    v = sin * log_x + cos * log_x_err
    return u, v


trans_log_rotation = jit(_trans_log_rotation)


def _trans_log_rotation_inv(u, v, params):
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    x = jnp.exp(u * cos + v * sin)
    x_err = jnp.exp(-u * sin + v * cos)
    return x, x_err


trans_log_rotation_inv = jit(_trans_log_rotation_inv)


def _trans_log_rotation_jac(x, x_err, params):
    return 1 / (x * x_err)


trans_log_rotation_jac = jit(_trans_log_rotation_jac)

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

all_trans_func = [
    'neglogplus', 'log_rotation', 'xshift_neglogplus', 'neglogplus_xshift'
]

NAME_FUNC_MAPPER = {}
for trans_name in all_trans_func:
    NAME_FUNC_MAPPER[f'trans_{trans_name}'] = globals()[f'trans_{trans_name}']
    NAME_FUNC_MAPPER[f'trans_{trans_name}_inv'] = globals(
    )[f'trans_{trans_name}_inv']
    NAME_FUNC_MAPPER[f'trans_{trans_name}_jac'] = globals(
    )[f'trans_{trans_name}_jac']

    if f'trans_{trans_name}_estimator' in globals():
        NAME_FUNC_MAPPER[f'trans_{trans_name}_estimator'] = globals(
        )[f'trans_{trans_name}_estimator']
