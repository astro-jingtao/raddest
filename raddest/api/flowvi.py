import os
import shutil
from typing import Union

import jax.numpy as jnp
import numpy as np
import joblib
import optax
from ndtest.ksnd import ks2d2s
from pzflow import Flow

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
        if att['flow']:
            flow = Flow(file=os.path.join(path, 'flow'))
            att['flow'] = flow
        else:
            flow = None

        if att['transfer']:
            transfer = Transfer.load(os.path.join(path, 'transfer'))
            att['transfer'] = transfer
        else:
            transfer = None

        is_fitted = att.pop('fitted')
        loss = att.pop(
            'loss') if 'loss' in att else None  # for backward compatibility
        new_cls = cls(**att)
        new_cls.fitted = is_fitted
        new_cls.loss = loss

        return new_cls

    def fit(self,
            x_obs,
            x_err,
            optimizer=optax.adam(1e-3),
            n_pre=100,
            n_train=500,
            n_node=21,
            int_method='ghq',
            x_min=None,
            x_max=None,
            x_err_min=None,
            x_err_max=None,
            guess_method='gg',
            guess_recalibrate=None,
            guess_verbose=False,
            derive_uv_bounds=False,
            skip_fitting_if_init_good=False,
            init_good_theshold=0.05,
            init_check_N=10000,
            check_per_step=10,
            early_stop=True,
            max_stuck=5,
            stuck_frac=0.05,
            re_estimate_transer_params=False,
            verbose=False):

        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if x_err_min is None:
            x_err_min = self.x_err_min
        if x_err_max is None:
            x_err_max = self.x_err_max

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
            int_method=int_method,
            verbose=verbose,
            x_min=x_min,
            x_max=x_max,
            x_err_min=x_err_min,
            x_err_max=x_err_max,
            check_per_step=check_per_step,
            early_stop=early_stop,
            max_stuck=max_stuck,
            stuck_frac=stuck_frac,
            guess_method=guess_method,
            guess_recalibrate=guess_recalibrate,
            guess_verbose=guess_verbose,
            derive_uv_bounds=derive_uv_bounds,
            skip_fitting_if_init_good=skip_fitting_if_init_good,
            init_good_theshold=init_good_theshold,
            init_check_N=init_check_N)

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
        '''
        
        return:
            x_gen: generated x values
            x_err_gen: generated x error values
            x_obs_gen: generated x observation values
        '''

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

        x_obs = jnp.asarray(x_obs)
        x_err = jnp.asarray(x_err)

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
                N=10000,
                M=None,
                K=1,
                x_min=None,
                x_max=None,
                x_err_min=None,
                x_err_max=None):
        """
        Perform 2D-KS test.

        Parameters
        -----------
        x_obs : array_like
            Observed x values.
        x_err : array_like
            Observed x errors.
        N : int, optional
            Number of points to generate for the test. Default is 10000.
        M : int, optional
            Number of data points to select for the test. If None, M is set to the number of x_obs. Default is None.
        K : int, optional
            Number of iterations for the test. Default is 1.
        x_min : float, optional
            Minimum x value for the generated data. Default is None.
        x_max : float, optional
            Maximum x value for the generated data. Default is None.
        x_err_min : float, optional
            Minimum x error value for the generated data. Default is None.
        x_err_max : float, optional
            Maximum x error value for the generated data. Default is None.

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

        N_data = x_obs.size

        if M is None:
            M = N_data

        if M > N_data:
            print(
                "Warning: M is larger than the number of data points, M is set to the number of data points."
            )
            M = N_data

        D_lst, p_lst = [], []

        for _ in range(K):

            _, x_err_gen, x_obs_gen = self.sample(N,
                                                  x_min=x_min,
                                                  x_max=x_max,
                                                  x_err_min=x_err_min,
                                                  x_err_max=x_err_max,
                                                  return_numpy=True)

            selected = np.random.choice(N_data, M, replace=False)
            D, p = ks2d2s(x_obs[selected], x_err[selected], x_obs_gen,
                          x_err_gen)

            D_lst.append(D)
            p_lst.append(p)

        return (D_lst[0], p_lst[0]) if K == 1 else (D_lst, p_lst)

    def save(self, folder, force=False):

        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            if not os.path.isdir(folder):
                if not force:
                    raise RuntimeError(
                        f"{folder} already exists, but is not a folder.")
                os.remove(folder)
                os.makedirs(folder)
            if len(os.listdir(folder)) > 0:
                if not force:
                    raise RuntimeError(
                        f"Folder {folder} already exists, and is not empty.")

                shutil.rmtree(folder)
                os.makedirs(folder)
        # save all the attributes of the object (except the flow)
        save_dict = {
            'transfer_params': self.transfer_params,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'x_err_min': self.x_err_min,
            'x_err_max': self.x_err_max,
            'fitted': self.fitted,
            'loss': self.loss
        }

        if self.transfer is None:
            save_dict['transfer'] = None
        else:
            save_dict['transfer'] = True
            self.transfer.save(os.path.join(folder, 'transfer'))

        # save the flow separately
        if self.flow is None:
            save_dict['flow'] = None
        else:
            save_dict['flow'] = True
            self.flow.save(os.path.join(folder, 'flow'))

        joblib.dump(save_dict, os.path.join(folder, 'attribute'))


def get_params_len(params):
    return 1 if isinstance(params, (int, float)) else len(params)
