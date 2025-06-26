from functools import partial
from typing import Tuple

import emcee
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.stats import norm
from jax.scipy.stats import t as student_t
from scipy.stats import t as student_t_scipy
from ndtest.ksnd import ks2d2s
from scipy.stats import qmc

from raddest.api.utils import \
    to_each_piece_params as _to_each_piece_params
from raddest.api.utils import \
    to_each_piece_params_studentT as _to_each_piece_params_studentT
from raddest.grid_search import (gen_new_grid_mean_std, gen_new_grid_quantile,
                                 get_weighted_res, grid_search, zoom_in_grid)
from raddest.reg.constructor import (
    construct_generator_cond, construct_ks_compare, construct_likelihood,
    construct_linear_step, construct_linear_student_t_step,
    construct_log_likelihood, construct_saturated_linear_step,
    construct_trainer)
from raddest.reg.constructor import \
    parse_piecewise_params as _parse_piecewise_params
from raddest.reg.constructor import piecewise_linear_step_factory
from raddest.reg.func import generate_piecewise_linear, saturated_linear_smooth
from raddest.reg.func_jax import (generate_piecewise_linear_jax,
                                  saturated_linear_smooth_jax)
from raddest.reg.ks import (gen_xy_obs_prepare, linear_scatter,
                            loglinear_scatter, loglinear_scatter_student_t,
                            parser_anylinear, print_mean_std_loglinear)
from raddest.reg.map import get_w_add_lognorm, get_xyw_trans


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

    def _check_generator_cond(self):
        if self.generator_cond is None:
            raise ValueError(
                "No conditional generator is not set. Please set it use `self.set_generator_cond()` first."
            )

    def sample(self,
               params,
               N=10000,
               x_range=None,
               x_err_range=None,
               y_range=None,
               y_err_range=None):
        '''
        Generate N data points.
        
        Parameters
        ----------
        params : dict
            Modle parameters used for the data generations.
        N : int, optional
            Number of points to generate. Default is 10000.
        x_range : tuple of (float, float), optional
            Range (min, max) for x values. Default is None.
            If None, x_range is set to [self.xflow.x_min, self.xflow.x_max].
        x_err_range : tuple of (float, float), optional
            Range (min, max) for x errors. Default is None.
            If None, x_err_range is set to [self.xflow.x_err_min, self.xflow.x_err_max].
        y_range : tuple of (float, float), optional
            Range (min, max) for y values. Default is None.
            If None, y_range is set to [self.yflow.x_min, self.yflow.x_max].
        y_err_range : tuple of (float, float), optional
            Range (min, max) for y errors. Default is None.
            If None, y_err_range is set to [self.yflow.x_err_min, self.yflow.x_err_max].

        Returns
        --------
        x_obs_gen : array_like
            Generated x values.
        y_obs_gen : array_like
            Generated y values.
        x_err_gen : array_like
            Generated x errors.
        y_err_gen : array_like
            Generated y errors.
        x_gen : array_like
            Generated x values.
        y_gen : array_like
            Generated y values.
        '''

        self._check_generator_cond()

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

    def ks_test(self,
                params,
                x_obs,
                y_obs,
                N=10000,
                M=None,
                K=1,
                x_range=None,
                x_err_range=None,
                y_range=None,
                y_err_range=None):
        """
        Perform 2D-KS test with K times of iterations. N data points are generated, and M observed data points are selected for each iteration.

        Parameters
        ----------
        params : dict
            Parameters used for the data generator.
        x_obs : array_like
            Observed x values.
        y_obs : array_like
            Observed y values.
        N : int, optional
            Number of points to generate for the test. Default is 10000.
        M : int, optional
            Number of data points to select for the test. If None, M is set to the number of x_obs. Default is None.
        K : int, optional
            Number of iterations for the test. Default is 1.
        x_range : tuple of (float, float), optional
            Range (min, max) for generating x values. Default is None.
        x_err_range : tuple of (float, float), optional
            Range (min, max) for generating x errors. Default is None.
        y_range : tuple of (float, float), optional
            Range (min, max) for generating y values. Default is None.
        y_err_range : tuple of (float, float), optional
            Range (min, max) for generating y errors. Default is None.

        Returns
        --------
        D or D_lst : float or list of float
            Single D value if K=1, otherwise list of D values.
        p or p_lst : float or list of float
            Single p value if K=1, otherwise list of p values.
        Notes
        ------
        If M is larger than the number of data points, M is set to the number of data points, and a warning is issued.
        """

        self._check_generator_cond()

        N_data = x_obs.size

        if M is None:
            M = N_data

        if M > N_data:
            print(
                "Warning: M is larger than the number of data points, M is set to the number of data points."
            )
            M = N_data

        # TODO: parallelize this loop

        D_lst, p_lst = [], []

        for _ in range(K):

            x_obs_gen, y_obs_gen, _, _, _, _ = self.sample(
                params,
                N=N,
                x_range=x_range,
                x_err_range=x_err_range,
                y_range=y_range,
                y_err_range=y_err_range)

            selected = np.random.choice(N_data, M, replace=False)
            D, p = ks2d2s(x_obs[selected], y_obs[selected], x_obs_gen,
                          y_obs_gen)

            D_lst.append(D)
            p_lst.append(p)

        return (D_lst[0], p_lst[0]) if K == 1 else (D_lst, p_lst)


# TODO: cache x_ext, y_ext, w ?
class RegressorML(RegressorBase):

    def __init__(self,
                 xflow,
                 yflow,
                 *,
                 p_cond,
                 step_constructor,
                 get_w_add=None,
                 func_x2y=None,
                 log_prior=None):
        '''
        log_prior: callable, log_prior(params) -> log(P(params))
        '''

        super().__init__(xflow, yflow, func_x2y=func_x2y)

        self._likelihood_ext = construct_likelihood(p_cond)
        self._log_likelihood_ext, self._d_log_likelihood_ext, self._dd_log_likelihood_ext = construct_log_likelihood(
            self._likelihood_ext)
        self._v_log_likelihood_ext = vmap(self._log_likelihood_ext,
                                          in_axes=(0, None, None, None))

        self.step = step_constructor(self._d_log_likelihood_ext)
        self.get_w_add = get_w_add
        self.trainer = construct_trainer(self._log_likelihood_ext, self.step)
        self.log_prior = log_prior if log_prior is not None else lambda params: 0.0

        @jit
        def _log_posterior_ext(params, x_ext, y_ext, w, n_sample):
            # we use the mean of the log-likelihood as the loss
            # here we multiply the number of samples to get the truu log-likelihood
            return self._log_likelihood_ext(
                params, x_ext, y_ext, w) * n_sample + self.log_prior(params)

        self._log_posterior_ext = _log_posterior_ext
        self._v_log_posterior_ext = vmap(_log_posterior_ext,
                                         in_axes=(0, None, None, None, None))

    def get_ext_data(self,
                     x_obs,
                     x_err,
                     y_obs,
                     y_err,
                     n=21,
                     x_range=None,
                     x_err_range=None,
                     y_range=None,
                     y_err_range=None,
                     flow_batch_size=None,
                     kde_batch_size=None,
                     ignore_y_err_c_y=False,
                     int_method='ghq'):

        x_range, x_err_range, y_range, y_err_range = self._set_range(
            x_range, x_err_range, y_range, y_err_range)

        x_ext, y_ext, w = get_xyw_trans(
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
            int_method=int_method,
            ignore_y_err_c_y=ignore_y_err_c_y,
            flow_batch_size=flow_batch_size,
            kde_batch_size=kde_batch_size,
            get_w_add=self.get_w_add,
            x_transfer=self.xflow.transfer,
            x_transfer_params=self.xflow.transfer_params,
            y_transfer=self.yflow.transfer,
            y_transfer_params=self.yflow.transfer_params,
            n=n)

        return x_ext, y_ext, w

    def get_cov(self, x_obs, x_err, y_obs, y_err, params, **kwargs):

        # make sure params is a jnp array
        # it will affect the shape of the hessian
        params = jnp.asarray(params)

        x_ext, y_ext, w = self.get_ext_data(x_obs, x_err, y_obs, y_err,
                                            **kwargs)

        n_sample = x_obs.size
        hessian = self._dd_log_likelihood_ext(params, x_ext, y_ext, w)
        # we use the mean of the log-likelihood as the loss
        # here we divide the number of samples to get the true covariance
        return -jnp.linalg.inv(hessian[0][0]) / n_sample

    def log_likelihood(self, x_obs, x_err, y_obs, y_err, params, **kwargs):

        x_ext, y_ext, w = self.get_ext_data(x_obs, x_err, y_obs, y_err,
                                            **kwargs)

        n_sample = x_obs.size
        # we use the mean of the log-likelihood as the loss
        # here we multiply the number of samples to get the true log-likelihood

        return self._v_log_likelihood_ext(self._get_v_params(params), x_ext,
                                          y_ext, w) * n_sample

    def log_posterior(self, x_obs, x_err, y_obs, y_err, params, **kwargs):

        x_ext, y_ext, w = self.get_ext_data(x_obs, x_err, y_obs, y_err,
                                            **kwargs)
        n_sample = x_obs.size
        return self._v_log_posterior_ext(self._get_v_params(params), x_ext,
                                         y_ext, w, n_sample)

    def fit(self,
            x_obs,
            x_err,
            y_obs,
            y_err,
            params_0,
            lr=1e-3,
            n_step=1_000_000,
            atol=1e-3,
            rtol=1e-3,
            verbose=False,
            **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        params_ml: jnp.ndarray
        params_std: jnp.ndarray
        cov: jnp.ndarray

        x_ext, y_ext, w = self.get_ext_data(x_obs, x_err, y_obs, y_err,
                                            **kwargs)

        params_ml = jnp.asarray(
            self.trainer(params_0,
                         x_ext,
                         y_ext,
                         w,
                         lr=lr,
                         n_step=n_step,
                         atol=atol,
                         rtol=rtol,
                         verbose=verbose))

        cov = self.get_cov(x_obs, x_err, y_obs, y_err, params_ml, **kwargs)

        params_std = jnp.sqrt(jnp.diag(cov))

        return params_ml, params_std, cov

    def get_posterior_qmc(self,
                          x_obs,
                          x_err,
                          y_obs,
                          y_err,
                          l_bounds,
                          u_bounds,
                          m=15,
                          **kwargs):

        if len(l_bounds) != len(u_bounds):
            raise ValueError(
                "l_bounds and u_bounds should have the same length.")

        d = len(l_bounds)
        sampler = qmc.Sobol(d=d, scramble=True)

        params_grid = qmc.scale(sampler.random_base2(m=m), l_bounds, u_bounds)

        log_posterior = self.log_posterior(x_obs, x_err, y_obs, y_err,
                                           params_grid, **kwargs)

        weight = jnp.exp(log_posterior - jnp.nanmax(log_posterior))

        return params_grid, weight

    def get_posterior_mcmc(self,
                           x_obs,
                           x_err,
                           y_obs,
                           y_err,
                           *,
                           ndim,
                           initial_state,
                           nsteps=10000,
                           sampler_kwargs=None,
                           run_mcmc_kwargs=None,
                           sampler=None,
                           **kwargs) -> emcee.EnsembleSampler:

        if sampler_kwargs is None:
            sampler_kwargs = {}

        if run_mcmc_kwargs is None:
            run_mcmc_kwargs = {}

        x_ext, y_ext, w = self.get_ext_data(x_obs, x_err, y_obs, y_err,
                                            **kwargs)

        def log_prob_target(params):
            n_sample = x_obs.size
            log_p = self._v_log_posterior_ext(self._get_v_params(params),
                                              x_ext, y_ext, w, n_sample)
            # nan to -inf
            return jnp.where(jnp.isnan(log_p), -jnp.inf, log_p)

        initial_state = jnp.asarray(initial_state)

        if sampler is None:
            nwalkers = initial_state.shape[0]
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            log_prob_target,
                                            vectorize=True,
                                            **sampler_kwargs)

        sampler.run_mcmc(initial_state, nsteps, **run_mcmc_kwargs)

        return sampler

    def _get_v_params(self, params):
        params = jnp.asarray(params)
        if params.ndim == 1:
            params = params[None, :]
        return params


class LogLinearML(RegressorML):

    def __init__(self, xflow, yflow, log_prior=None):

        @jit
        def loglinear_jax(x, params):
            return params[0] * jnp.log10(jnp.abs(x)) + params[1]

        @jit
        def lognorm(x, y, params):
            '''
            params: (...), sig

            return: 
            if (x < 0) or (y < 0) -> 0
            else -> gaussian(log10(y)|loc(x, params[:-1]), params[-1])

            notes:
            This is p(log10(y)|loc, scale), while p(y|loc, scale) = p(log10(y)|loc, scale)/y/ln(10)
            Addtional cofficients should be absorbed in weight
            '''
            return jnp.where((x < 0) | (y < 0), 0,
                             norm.pdf(jnp.log10(jnp.abs(y)),
                                      loc=loglinear_jax(x, params[:-1]),
                                      scale=params[-1]))

        if log_prior is None:

            def log_prior(params):
                return jnp.where(params[-1] <= 0, -jnp.inf, 0)

        super().__init__(xflow,
                         yflow,
                         p_cond=lognorm,
                         step_constructor=construct_linear_step,
                         get_w_add=get_w_add_lognorm,
                         func_x2y=loglinear_scatter,
                         log_prior=log_prior)


class SaturatedLogLinearML(RegressorML):

    def __init__(self, xflow, yflow, width=0.1, log_prior=None):

        def saturated_loglinear_scatter(x, params):
            params_loglinear = params[:-1]
            sig = params[-1]
            log10_y_pred = saturated_linear_smooth(np.log10(x),
                                                   *params_loglinear,
                                                   width=width)
            return np.power(
                10, log10_y_pred + np.random.normal(scale=sig, size=x.shape))

        @jit
        def saturated_loglinear_jax(x, params):
            '''
            params: (k, b, x0, x1, c0, c1)
            '''
            log10_y_pred = saturated_linear_smooth_jax(jnp.log10(jnp.abs(x)),
                                                       *params,
                                                       width=width)
            return log10_y_pred

        @jit
        def lognorm(x, y, params):
            return jnp.where(
                (x < 0) | (y < 0), 0,
                norm.pdf(jnp.log10(jnp.abs(y)),
                         loc=saturated_loglinear_jax(x, params[:-1]),
                         scale=params[-1]))

        if log_prior is None:

            def log_prior(params):
                k, b, x0, x1, c0, c1, sig = params
                is_bad = (sig <= 0) | (x0 > x1) | ((c1 - c0) * k < 0)
                return jnp.where(is_bad, -jnp.inf, 0)

        super().__init__(xflow,
                         yflow,
                         p_cond=lognorm,
                         step_constructor=construct_saturated_linear_step,
                         get_w_add=get_w_add_lognorm,
                         func_x2y=saturated_loglinear_scatter,
                         log_prior=log_prior)


class PiecewiseLogLinearML(RegressorML):

    def __init__(self,
                 xflow,
                 yflow,
                 n_piece,
                 width=0.05,
                 A_eq=None,
                 b_eq=None,
                 log_prior=None):

        self.n_piece = n_piece
        piecewise_linear = generate_piecewise_linear(n_piece)
        piecewise_linear_jax = generate_piecewise_linear_jax(n_piece)

        def piecewise_loglinear_scatter(x, params):

            sig = params[-1]
            log10_y_pred = piecewise_linear(np.log10(x),
                                            *self.parse_piecewise_params(
                                                params[:-1]),
                                            width=width)
            return np.power(
                10, log10_y_pred + np.random.normal(scale=sig, size=x.shape))

        @jit
        def piecewise_loglinear_jax(x, params):
            '''
            params: (b0, x0) + k_lst + dx_lst
            '''
            log10_y_pred = piecewise_linear_jax(
                jnp.log10(jnp.abs(x)),
                *self.parse_piecewise_params(params),
                width=width)
            return log10_y_pred

        @jit
        def lognorm(x, y, params):
            # print(len(params))
            return jnp.where(
                (x < 0) | (y < 0), 0,
                norm.pdf(jnp.log10(jnp.abs(y)),
                         loc=piecewise_loglinear_jax(x, params[:-1]),
                         scale=params[-1]))

        if log_prior is None:

            def log_prior(params):
                sig = params[-1]
                _, _, _, dx_lst = self.parse_piecewise_params(params[:-1])
                is_bad = jnp.any(dx_lst <= 0) | (sig <= 0)
                return jnp.where(is_bad, -jnp.inf, 0)

        super().__init__(xflow,
                         yflow,
                         p_cond=lognorm,
                         step_constructor=piecewise_linear_step_factory(
                             n_piece, A_eq=A_eq, b_eq=b_eq),
                         get_w_add=get_w_add_lognorm,
                         func_x2y=piecewise_loglinear_scatter,
                         log_prior=log_prior)

    def parse_piecewise_params(self, params):
        return _parse_piecewise_params(self.n_piece, params)

    def to_each_piece_params(self, params):
        return _to_each_piece_params(self.n_piece, params)


# TODO: to implement
class PiecewiseLogLinearStudentTML(RegressorML):

    def __init__(self,
                 xflow,
                 yflow,
                 n_piece,
                 width=0.05,
                 A_eq=None,
                 b_eq=None,
                 log_prior=None):

        self.n_piece = n_piece
        piecewise_linear = generate_piecewise_linear(n_piece)
        piecewise_linear_jax = generate_piecewise_linear_jax(n_piece)

        def piecewise_loglinear_scatter(x, params):

            df, sig = params[-2:]
            log10_y_pred = piecewise_linear(np.log10(x),
                                            *self.parse_piecewise_params(
                                                params[:-2]),
                                            width=width)
            return np.power(
                10, log10_y_pred +
                student_t_scipy.rvs(df, scale=sig, size=x.shape))

        @jit
        def piecewise_loglinear_jax(x, params):
            '''
            params: (b0, x0) + k_lst + dx_lst
            '''
            log10_y_pred = piecewise_linear_jax(
                jnp.log10(jnp.abs(x)),
                *self.parse_piecewise_params(params),
                width=width)
            return log10_y_pred

        @jit
        def logstudent_t(x, y, params):
            # print(len(params))
            return jnp.where(
                (x < 0) | (y < 0), 0,
                student_t.pdf(jnp.log10(jnp.abs(y)),
                              params[-2],
                              loc=piecewise_loglinear_jax(x, params[:-2]),
                              scale=params[-1]))

        if log_prior is None:

            def log_prior(params):
                df, sig = params[-2:]
                _, _, _, dx_lst = self.parse_piecewise_params(params[:-2])
                is_bad = jnp.any(dx_lst <= 0) | (sig <= 0) | (df <= 0)
                return jnp.where(is_bad, -jnp.inf, 0)

        super().__init__(
            xflow,
            yflow,
            p_cond=logstudent_t,
            step_constructor=piecewise_linear_studentT_step_factory(n_piece,
                                                                    A_eq=A_eq,
                                                                    b_eq=b_eq),
            get_w_add=get_w_add_lognorm,
            func_x2y=piecewise_loglinear_scatter,
            log_prior=log_prior)

    def parse_piecewise_params(self, params):
        return _parse_piecewise_params(self.n_piece, params)

    def to_each_piece_params(self, params):
        return _to_each_piece_params_studentT(self.n_piece, params)


class LogLinearStudentTML(RegressorML):
    # https://arxiv.org/abs/2411.02380

    def __init__(self, xflow, yflow, log_prior=None):

        @jit
        def loglinear_jax(x, params):
            return params[0] * jnp.log10(jnp.abs(x)) + params[1]

        @jit
        def logstudent_t(x, y, params):
            '''
            params: (...), sig

            return: 
            if (x < 0) or (y < 0) -> 0
            else -> student_t(log10(y)|loc(x, params[:-1]), df=params[-2], sigma=params[-1])

            notes:
            This is p(log10(y)|loc, scale), while p(y|loc, scale) = p(log10(y)|loc, scale)/y/ln(10)
            Addtional cofficients should be absorbed in weight
            '''
            return jnp.where((x < 0) | (y < 0), 0,
                             student_t.pdf(jnp.log10(jnp.abs(y)),
                                           params[-2],
                                           loc=loglinear_jax(x, params[:-2]),
                                           scale=params[-1]))

        if log_prior is None:

            def log_prior(params):
                return jnp.where((params[-1] <= 0) | (params[-2] <= 0),
                                 -jnp.inf, 0)

        super().__init__(
            xflow,
            yflow,
            p_cond=logstudent_t,
            step_constructor=construct_linear_student_t_step,
            get_w_add=get_w_add_lognorm,  # same as lognorm
            func_x2y=loglinear_scatter_student_t,
            log_prior=log_prior)


class LinearML(RegressorML):

    def __init__(self, xflow, yflow):

        @jit
        def linear_jax(x, params):
            return params[0] * x + params[1]

        @jit
        def linearnorm(x, y, params):
            '''
            params: (...), sig

            return: 
            gaussian(y|loc(x, params[:-1]), params[-1])

            notes:
            No addtinonal cofficients needed
            '''
            return norm.pdf(y,
                            loc=linear_jax(x, params[:-1]),
                            scale=params[-1])

        super().__init__(xflow,
                         yflow,
                         p_cond=linearnorm,
                         step_constructor=construct_linear_step,
                         get_w_add=None,
                         func_x2y=linear_scatter)


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


class LogLinearKS(RegressorKS):

    def __init__(self, xflow, yflow, gen_new_grid='mean_std'):

        super().__init__(xflow,
                         yflow,
                         func_x2y=loglinear_scatter,
                         parser=parser_anylinear,
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


class LinearKS(RegressorKS):

    def __init__(self, xflow, yflow, gen_new_grid='mean_std'):

        super().__init__(xflow,
                         yflow,
                         func_x2y=linear_scatter,
                         parser=parser_anylinear,
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
