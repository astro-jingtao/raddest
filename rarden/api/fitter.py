from typing import Union
import os

import joblib
from pzflow import Flow
import jax.numpy as jnp
import optax

from ..transfer import Transfer
from ..x_xerr import nf_fit_general, gen_xobs_trans, construct_x_xerr_w, construct_nf_fit_func


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
            stuck_frac=0.05):

        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if x_err_min is None:
            x_err_min = self.x_err_min * 0.8
        if x_err_max is None:
            x_err_max = self.x_err_max * 1.2

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


class Regressor:
    def __init__(self):
        pass

