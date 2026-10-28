import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.transfer import NegLogPlus

TRUE_PARAMS = {'k': 2.0, 'b': 0.5, 'sig': 0.1}


def make_obs_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 3, size=n)
    x_err = rng.uniform(0.05, 0.15, size=n)
    x_obs = rng.normal(x, x_err)

    y = 10**(TRUE_PARAMS['k'] * np.log10(x) + TRUE_PARAMS['b'] +
             rng.normal(scale=TRUE_PARAMS['sig'], size=n))
    y_err = 0.05 * y
    y_obs = rng.normal(y, y_err)

    return x_obs, x_err, y_obs, y_err


def fit_tiny_flow(x_obs, x_err):
    flow_vi = FlowVI(x_min=float(x_obs.min()) * 0.5,
                     x_max=float(x_obs.max()) * 1.1,
                     x_err_min=0,
                     x_err_max=float(x_err.max()) * 1.2,
                     transfer=NegLogPlus(),
                     transfer_params=[float(np.median(x_obs)),
                                      float(np.median(x_err))])
    flow_vi.fit(x_obs,
                x_err,
                n_pre=10,
                n_train=20,
                n_node=5,
                verbose=False,
                x_min=float(x_obs.min()) * 0.5,
                x_max=float(x_obs.max()) * 1.1,
                x_err_min=0,
                x_err_max=float(x_err.max()) * 1.2,
                early_stop=False)
    return flow_vi


@pytest.fixture(scope='session')
def obs_data():
    return make_obs_data()


@pytest.fixture(scope='session')
def tiny_flows(obs_data):
    x_obs, x_err, y_obs, y_err = obs_data
    return fit_tiny_flow(x_obs, x_err), fit_tiny_flow(y_obs, y_err)


@pytest.fixture
def obs_data_jax(obs_data):
    x_obs, x_err, y_obs, y_err = obs_data
    return (jnp.asarray(x_obs), jnp.asarray(x_err), jnp.asarray(y_obs),
            jnp.asarray(y_err))
