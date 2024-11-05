import joblib

import jax.numpy as jnp
from jax import jit
from .tools import neg_log_plus, neg_log_plus_inv


class Transfer:

    def __init__(self, trans, inv, jac, n_params, params_estimator=None):
        self.trans = trans
        self.inv = inv
        self.jac = jac
        self.n_params = n_params
        self.params_estimator = params_estimator

    @classmethod
    def load(cls, path, name_func_mapper=None):
        if name_func_mapper is None:
            name_func_mapper = NAME_FUNC_MAPPER
        load_dict = joblib.load(path)
        cls_par = {
            k: NAME_FUNC_MAPPER[load_dict[k]]
            for k in ['trans', 'inv', 'jac', 'params_estimator']
        }
        cls_par['n_params'] = load_dict['n_params']
        return cls(**cls_par)

    def save(self, path):
        save_dict = {
            'trans':
            self.trans.__name__,
            'inv':
            self.inv.__name__,
            'jac':
            self.jac.__name__,
            'n_params':
            self.n_params,
            'params_estimator':
            self.params_estimator.__name__
            if self.params_estimator is not None else None
        }
        joblib.dump(save_dict, path)


# -- neglogplus --


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
def trans_neglogplus_jac(x, x_err, params):
    x_median, x_err_median = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


@jit
def trans_neglogplus_estimator(x, x_err):
    x_median = jnp.median(x)
    x_err_median = jnp.median(x_err)
    return x_median, x_err_median


neglogplus = Transfer(trans_neglogplus,
                      trans_neglogplus_inv,
                      trans_neglogplus_jac,
                      2,
                      params_estimator=trans_neglogplus_estimator)

# -- log_rotation --

# u = cos * log(x) - sin * log(x_err)
# v = sin * log(x) + cos * log(x_err)


@jit
def trans_log_rotation(x, x_err, params):
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    log_x = jnp.log(x)
    log_x_err = jnp.log(x_err)
    u = cos * log_x - sin * log_x_err
    v = sin * log_x + cos * log_x_err
    return u, v


@jit
def trans_log_rotation_inv(u, v, params):
    theta = params
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    x = jnp.exp(u * cos + v * sin)
    x_err = jnp.exp(-u * sin + v * cos)
    return x, x_err


@jit
def trans_log_rotation_jac(x, x_err, params):
    return 1 / (x * x_err)


log_rotation = Transfer(trans_log_rotation, trans_log_rotation_inv,
                        trans_log_rotation_jac, 1)

# -- xshift_neglogplus --

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
def trans_xshift_neglogplus_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
    k_x = jnp.where((x + x_shift) / x_median > 0, 1 / (x + x_shift + x_median),
                    1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


xshift_neglogplus = Transfer(trans_xshift_neglogplus,
                             trans_xshift_neglogplus_inv,
                             trans_xshift_neglogplus_jac, 3)

# -- neglogplus_xshift --

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
def trans_neglogplus_xshift_jac(x, x_err, params):
    x_median, x_err_median, x_shift = params
    k_x = jnp.where(x / x_median > 0, 1 / (x + x_median), 1 / x_median)
    k_x_err = jnp.where(x_err / x_err_median > 0, 1 / (x_err + x_err_median),
                        1 / x_err_median)
    return k_x * k_x_err


neglogplus_xshift = Transfer(trans_neglogplus_xshift,
                             trans_neglogplus_xshift_inv,
                             trans_neglogplus_xshift_jac, 3)

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
