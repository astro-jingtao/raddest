import os
from functools import partial
from typing import Union

import jax.numpy as jnp
import joblib
import numpy as np
import optax
from jax import jit
from jax.scipy.stats import norm
from ndtest.ksnd import ks2d2s
from pzflow import Flow

from ..grid_search import (gen_new_grid_mean_std, gen_new_grid_quantile,
                           get_weighted_res, grid_search, zoom_in_grid)
from ..reg.constructor import (construct_generator_cond, construct_ks_compare,
                               construct_likelihood, construct_linear_step,
                               construct_log_likelihood, construct_trainer)
from ..reg.ks import (gen_xy_obs_prepare, loglinear_scatter, parser_loglinear,
                      print_mean_std_loglinear)
from ..reg.map import get_xyw_trans_lognorm
from ..transfer import Transfer
from ..x_xerr import (construct_nf_fit_func, construct_x_xerr_w,
                      gen_xobs_trans, nf_fit_general)


class FlowVI:

    def __init__(self,
                 flow: Union[Flow, None] = None,
                 transfer: Union[Transfer, None] = None,
                 transfer_params=None,
                 x_min=-jnp.inf,
                 x_max=jnp.inf,
                 x_err_min=-jnp.inf,
                 x_err_max=jnp.inf):
        self.flow: Union[Flow, None] = flow
        self.transfer: Union[Transfer, None] = transfer

        # sourcery skip: merge-else-if-into-elif
        if self.transfer is None:  # transfer not used
            if transfer_params is not None:
                print(
                    "Warning: transfer_params provided, but transfer is None, the transfer_params will be ignored."
                )
        else:  # transfer used
            if transfer_params is None:  # transfer_params not provided
                if self.transfer.params_estimator is None:
                    raise ValueError(
                        "The transfer function does not have a parameter estimator, please provide transfer_params."
                    )
            elif get_params_len(
                    transfer_params
            ) != self.transfer.n_params:  # transfer_params has wrong length
                raise ValueError(
                    "transfer_params must have the same length as the number of parameters in the transfer function."
                )

        self.transfer_params = transfer_params

        self.x_min = x_min
        self.x_max = x_max
        self.x_err_min = x_err_min
        self.x_err_max = x_err_max

        self.fitted = False
        self.loss = None

    @classmethod
    def load(cls, path):
        att = joblib.load(os.path.join(path, 'attribute'))
        flow = Flow(file=os.path.join(path, 'flow'))
        if att['transfer']:
            transfer = Transfer.load(os.path.join(path, 'transfer'))
            att['transfer'] = transfer
        else:
            transfer = None

        is_fitted = att.pop('fitted')
        new_cls = cls(flow=flow, **att)
        new_cls.fitted = is_fitted

        return new_cls

    def fit(self,
            x_obs,
            x_err,
            optimizer=optax.adam(1e-3),
            n_pre=100,
            n_train=500,
            n_node=21,
            qmc=False,
            verbose=False,
            x_min=None,
            x_max=None,
            x_err_min=None,
            x_err_max=None,
            check_per_step=10,
            early_stop=True,
            max_stuck=5,
            stuck_frac=0.05,
            re_estimate_transer_params=False):

        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if x_err_min is None:
            x_err_min = self.x_err_min * 0.8
        if x_err_max is None:
            x_err_max = self.x_err_max * 1.2

        # sourcery skip: swap-nested-ifs, merge-nested-ifs
        if self.transfer is not None:  # transfer used
            if re_estimate_transer_params or (self.transfer_params is None):
                if self.transfer.params_estimator is None:
                    raise ValueError(
                        "The transfer function does not have a parameter estimator."
                    )
                else:
                    self.transfer_params = self.transfer.params_estimator(
                        x_obs, x_err)

        self.flow, self.loss, _ = nf_fit_general(
            x_obs,
            x_err,
            optimizer=optimizer,
            flow=self.flow,
            transfer=self.transfer,
            transfer_params=self.transfer_params,
            n_pre=n_pre,
            n_train=n_train,
            n_node=n_node,
            qmc=qmc,
            verbose=verbose,
            x_min=x_min,
            x_max=x_max,
            x_err_min=x_err_min,
            x_err_max=x_err_max,
            check_per_step=check_per_step,
            early_stop=early_stop,
            max_stuck=max_stuck,
            stuck_frac=stuck_frac)

        self.fitted = True

    def check_fit(self):
        if not self.fitted:
            raise ValueError("Model not fitted yet.")

    def sample(self,
               N,
               x_min=None,
               x_max=None,
               x_err_min=None,
               x_err_max=None,
               return_numpy=False):

        self.check_fit()

        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if x_err_min is None:
            x_err_min = self.x_err_min * 0.8
        if x_err_max is None:
            x_err_max = self.x_err_max * 1.2

        return gen_xobs_trans(self.flow,
                              N,
                              x_min,
                              x_max,
                              x_err_min,
                              x_err_max,
                              transfer=self.transfer,
                              transfer_params=self.transfer_params,
                              return_numpy=return_numpy)

    def log_likelihood(self, x_obs, x_err, n_node=21, qmc=False):

        self.check_fit()

        u_ext, v_ext, w = construct_x_xerr_w(
            x_obs,
            x_err,
            transfer=self.transfer,
            transfer_params=self.transfer_params,
            n_node=n_node,
            qmc=qmc)

        # to have a separate function for log_likelihood
        _, log_likelihood = construct_nf_fit_func(optax.adam(1e-3), self.flow)

        return log_likelihood([self.flow._params[0], self.flow._params[1]],
                              u_ext, v_ext, w)

    def ks_test(self,
                x_obs,
                x_err,
                N,
                x_min=None,
                x_max=None,
                x_err_min=None,
                x_err_max=None):

        x_gen, x_err_gen, x_obs_gen = self.sample(N,
                                                  x_min=x_min,
                                                  x_max=x_max,
                                                  x_err_min=x_err_min,
                                                  x_err_max=x_err_max,
                                                  return_numpy=True)

        return ks2d2s(x_obs, x_err, x_obs_gen, x_err_gen)

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            if not os.path.isdir(folder):
                raise ValueError(
                    f"{folder} already exists, but is not a folder.")
            if len(os.listdir(folder)) > 0:
                raise ValueError(
                    f"Folder {folder} already exists, and is not empty.")

        # save all the attributes of the object (except the flow)
        save_dict = {
            'transfer_params': self.transfer_params,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'x_err_min': self.x_err_min,
            'x_err_max': self.x_err_max,
            'fitted': self.fitted,
        }

        if self.transfer is None:
            save_dict['transfer'] = None
        else:
            save_dict['transfer'] = True
            self.transfer.save(os.path.join(folder, 'transfer'))

        joblib.dump(save_dict, os.path.join(folder, 'attribute'))

        # save the flow separately
        self.flow.save(os.path.join(folder, 'flow'))


class RegressorBase:

    def __init__(self, xflow, yflow, func_x2y=None):
        self.xflow = xflow
        self.yflow = yflow

        self.set_generator_cond(func_x2y)

    def _set_range(self,
                   x_range,
                   x_err_range,
                   y_range,
                   y_err_range,
                   return_numpy=False):

        if x_range is None:
            x_range = [self.xflow.x_min, self.xflow.x_max]
        if x_err_range is None:
            x_err_range = [self.xflow.x_err_min, self.xflow.x_err_max]
        if y_range is None:
            y_range = [self.yflow.x_min, self.yflow.x_max]
        if y_err_range is None:
            y_err_range = [self.yflow.x_err_min, self.yflow.x_err_max]

        if return_numpy:
            x_range = np.array(x_range)
            x_err_range = np.array(x_err_range)
            y_range = np.array(y_range)
            y_err_range = np.array(y_err_range)

        return x_range, x_err_range, y_range, y_err_range

    def set_generator_cond(self, func_x2y):
        if func_x2y is not None:
            self.generator_cond = construct_generator_cond(func_x2y)
        else:
            self.generator_cond = None

    def sample(self,
               params,
               N=1000,
               x_range=None,
               x_err_range=None,
               y_range=None,
               y_err_range=None):

        if self.generator_cond is None:
            raise ValueError(
                "No conditional generator is not set. Please set it use `self.set_generator_cond()` first."
            )

        x_range, x_err_range, y_range, y_err_range = self._set_range(
            x_range, x_err_range, y_range, y_err_range)

        x_gen, x_err_gen, x_obs_gen, y_gen_kdt, y_err_gen_kdt = gen_xy_obs_prepare(
            self.xflow.flow,
            self.yflow.flow,
            x_range,
            x_err_range,
            y_range,
            y_err_range,
            x_transfer=self.xflow.transfer,
            x_transfer_params=self.xflow.transfer_params,
            y_transfer=self.yflow.transfer,
            y_transfer_params=self.yflow.transfer_params,
            n=N,
            return_numpy=True)

        y_gen, y_err_gen, y_obs_gen = self.generator_cond(x_gen,
                                                          y_gen_kdt,
                                                          y_err_gen_kdt,
                                                          params,
                                                          y_min=y_range[0],
                                                          y_max=y_range[-1])

        return x_obs_gen, y_obs_gen, x_err_gen, y_err_gen, x_gen, y_gen


class RegressorML(RegressorBase):

    def __init__(self, xflow, yflow, p_cond, step_constructor, func_x2y=None):

        super().__init__(xflow, yflow, func_x2y=func_x2y)

        self.likelihhod = construct_likelihood(p_cond)
        self.log_likelihood, self.d_log_likelihood, self.dd_log_likelihood = construct_log_likelihood(
            self.likelihhod)
        self.step = step_constructor(self.d_log_likelihood)
        self.trainer = construct_trainer(self.log_likelihood, self.step)

    def fit(self,
            x_obs,
            x_err,
            y_obs,
            y_err,
            params_0,
            n=21,
            x_range=None,
            x_err_range=None,
            y_range=None,
            y_err_range=None,
            lr=1e-3,
            n_step=10000,
            atol=1e-3,
            rtol=1e-3,
            verbose=False):

        x_range, x_err_range, y_range, y_err_range = self._set_range(
            x_range, x_err_range, y_range, y_err_range)

        # TODO: use the get_xyw_trans for any correlation, not just lognorm
        x_ext, y_ext, w = get_xyw_trans_lognorm(
            x_obs,
            y_obs,
            x_err,
            y_err,
            self.xflow.flow,
            self.yflow.flow,
            x_range,
            x_err_range,
            y_range,
            y_err_range,
            x_transfer=self.xflow.transfer,
            x_transfer_params=self.xflow.transfer_params,
            y_transfer=self.yflow.transfer,
            y_transfer_params=self.yflow.transfer_params,
            n=n)

        return self.trainer(params_0,
                            x_ext,
                            y_ext,
                            w,
                            lr=lr,
                            n_step=n_step,
                            atol=atol,
                            rtol=rtol,
                            verbose=verbose)


class RegressorLogLinearML(RegressorML):

    def __init__(self, xflow, yflow):

        @jit
        def loglinear_jax(x, params):
            return params[0] * jnp.log10(jnp.abs(x)) + params[1]

        @jit
        def lognorm(x, y, params):
            '''
            params: (...), sig

            return: 
            if (x < 0) or (y < 0) -> 0
            else -> gaussian(log10(y)|loc(x, params[:-1]), params[:-1])

            notes:
            This is p(log10(y)|loc, scale), while p(y|loc, scale) = p(log10(y)|loc, scale)/y/ln(10)
            Addtional cofficients should be absorbed in weight
            '''
            return jnp.where((x < 0) | (y < 0), 0,
                             norm.pdf(jnp.log10(jnp.abs(y)),
                                      loc=loglinear_jax(x, params[:-1]),
                                      scale=params[-1]))

        super().__init__(xflow,
                         yflow,
                         lognorm,
                         construct_linear_step,
                         func_x2y=loglinear_scatter)


class RegressorKS(RegressorBase):

    def __init__(self,
                 xflow,
                 yflow,
                 func_x2y,
                 *,
                 parser,
                 print_mean_std,
                 gen_new_grid='mean_std'):

        if func_x2y is None:
            raise ValueError("No function to convert x to y is provided.")

        super().__init__(xflow, yflow, func_x2y=func_x2y)

        self.parser = parser
        self.print_mean_std = print_mean_std

        if isinstance(gen_new_grid, str):
            self.gen_new_grid_name = gen_new_grid
            if gen_new_grid == 'mean_std':
                self.gen_new_grid = gen_new_grid_mean_std
            elif gen_new_grid == 'quantile':
                self.gen_new_grid = gen_new_grid_quantile
            else:
                raise ValueError(f"Unknown gen_new_grid: {gen_new_grid}")
        elif callable(gen_new_grid):
            self.gen_new_grid_name = 'custom'
            self.gen_new_grid = gen_new_grid

    def _set_gen_new_grid_kwargs(self, gen_new_grid_kwargs):
        if gen_new_grid_kwargs is None:
            gen_new_grid_kwargs = {}
        return gen_new_grid_kwargs

    def _set_gen_new_grid_refine_kwargs(self, gen_new_grid_refine_kwargs):
        if gen_new_grid_refine_kwargs is None:
            gen_new_grid_refine_kwargs = {}

        if 'n' not in gen_new_grid_refine_kwargs:
            gen_new_grid_refine_kwargs['n'] = 20

        return gen_new_grid_refine_kwargs

    def fit(self,
            x_obs,
            y_obs,
            *,
            params_list,
            N=1000,
            x_range=None,
            x_err_range=None,
            y_range=None,
            y_err_range=None,
            num_process=10,
            max_i=10,
            rtol=1e-3,
            atol=1e-3,
            verbose=False,
            refine=True,
            gen_new_grid_kwargs=None,
            gen_new_grid_refine_kwargs=None):

        gen_new_grid_kwargs = self._set_gen_new_grid_kwargs(
            gen_new_grid_kwargs)

        gen_new_grid_refine_kwargs = self._set_gen_new_grid_refine_kwargs(
            gen_new_grid_refine_kwargs)

        x_obs = np.asarray(x_obs)
        y_obs = np.asarray(y_obs)

        x_range, x_err_range, y_range, y_err_range = self._set_range(
            x_range, x_err_range, y_range, y_err_range, return_numpy=True)

        x_gen, _, x_obs_gen, y_kdt, y_err_kdt = gen_xy_obs_prepare(
            self.xflow.flow,
            self.yflow.flow,
            x_range,
            x_err_range,
            y_range,
            y_err_range,
            x_transfer=self.xflow.transfer,
            x_transfer_params=self.xflow.transfer_params,
            y_transfer=self.yflow.transfer,
            y_transfer_params=self.yflow.transfer_params,
            n=N,
            return_numpy=True)

        # Use construct_ks_compare but not define ks_compare here, or each process in zoom_in_grid will try to import jax thus lead to OOM.
        compare = construct_ks_compare(self.generator_cond, x_gen, x_obs_gen,
                                       y_kdt, y_err_kdt, x_obs, y_obs)

        w_index = len(params_list) + 1

        this_gen_new_grid = partial(self.gen_new_grid, **gen_new_grid_kwargs)

        res_arr = zoom_in_grid(compare,
                               params_list,
                               w_index,
                               self.parser,
                               this_gen_new_grid,
                               self.print_mean_std,
                               num_process=num_process,
                               max_i=max_i,
                               rtol=rtol,
                               atol=atol,
                               verbose=verbose)

        if refine:
            n_par = len(params_list)
            params_list_refine = self.gen_new_grid(
                res_arr,
                n_par=n_par,
                w_index=w_index,
                **gen_new_grid_refine_kwargs)
            res_arr = grid_search(compare,
                                  params_list_refine,
                                  self.parser,
                                  num_process=num_process)

        params_mean, params_std = get_weighted_res(res_arr[:, :n_par].T,
                                                   res_arr[:, w_index],
                                                   std_min=0)

        return params_mean, params_std, res_arr


class RegressorLogLinearKS(RegressorKS):

    def __init__(self, xflow, yflow, gen_new_grid='mean_std'):

        super().__init__(xflow,
                         yflow,
                         func_x2y=loglinear_scatter,
                         parser=parser_loglinear,
                         print_mean_std=print_mean_std_loglinear,
                         gen_new_grid=gen_new_grid)

    def _set_gen_new_grid_kwargs(self, gen_new_grid_kwargs):
        gen_new_grid_kwargs = super()._set_gen_new_grid_kwargs(
            gen_new_grid_kwargs)

        if 'range_par' not in gen_new_grid_kwargs:
            gen_new_grid_kwargs['range_par'] = [[-np.inf, np.inf],
                                                [-np.inf, np.inf], [0, np.inf]]

        return gen_new_grid_kwargs

    def _set_gen_new_grid_refine_kwargs(self, gen_new_grid_refine_kwargs):
        gen_new_grid_refine_kwargs = super()._set_gen_new_grid_refine_kwargs(
            gen_new_grid_refine_kwargs)

        if 'range_par' not in gen_new_grid_refine_kwargs:
            gen_new_grid_refine_kwargs['range_par'] = [[-np.inf, np.inf],
                                                       [-np.inf, np.inf],
                                                       [0, np.inf]]

        return gen_new_grid_refine_kwargs


def get_params_len(params):
    return 1 if isinstance(params, (int, float)) else len(params)
