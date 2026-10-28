import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.fitter import LogLinearEnergyGrad, LogLinearEnergyPT, RegressorEnergyGrad
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
                n_pre=20,
                n_train=50,
                n_node=7,
                verbose=False,
                x_min=float(x_obs.min()) * 0.5,
                x_max=float(x_obs.max()) * 1.1,
                x_err_min=0,
                x_err_max=float(x_err.max()) * 1.2,
                early_stop=False)
    return flow_vi


@pytest.fixture(scope='module')
def fitted_regressors():
    np.random.seed(0)
    x_obs, x_err, y_obs, y_err = make_obs_data()

    xflow = fit_tiny_flow(x_obs, x_err)
    yflow = fit_tiny_flow(y_obs, y_err)

    regressor_pt = LogLinearEnergyPT(xflow, yflow, permutations=50)
    regressor_grad = LogLinearEnergyGrad(xflow, yflow)
    return regressor_pt, regressor_grad, x_obs, x_err, y_obs, y_err


class TestLogLinearEnergyPT:

    def test_fit_recovers_params(self, fitted_regressors):
        regressor_pt, _, x_obs, _, y_obs, _ = fitted_regressors

        params_list = [
            np.linspace(1.0, 3.0, 4),
            np.linspace(-0.5, 1.5, 4),
            np.linspace(0.05, 0.2, 3),
        ]

        params_mean, params_std, res_arr = regressor_pt.fit(x_obs,
                                                            y_obs,
                                                            params_list=params_list,
                                                            N=300,
                                                            num_process=2,
                                                            max_i=1,
                                                            refine=False)

        assert res_arr.shape[1] == 5
        assert np.all(params_std >= 0)
        assert abs(params_mean[0] - TRUE_PARAMS['k']) < 0.6
        assert abs(params_mean[1] - TRUE_PARAMS['b']) < 0.6
        assert abs(params_mean[2] - TRUE_PARAMS['sig']) < 0.15

    def test_energy_test_prefers_true_params(self, fitted_regressors):
        regressor_pt, _, x_obs, _, y_obs, _ = fitted_regressors
        np.random.seed(1)

        _, p_good = regressor_pt.energy_test(
            (TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']),
            x_obs,
            y_obs,
            N=300,
            M=300,
            permutations=100)
        _, p_bad = regressor_pt.energy_test((4.0, 2.0, 0.4),
                                            x_obs,
                                            y_obs,
                                            N=300,
                                            M=300,
                                            permutations=100)

        assert p_good > p_bad


class TestLogLinearEnergyGrad:

    def test_fit_decreases_loss_and_moves_toward_truth(self, fitted_regressors):
        _, regressor_grad, x_obs, _, y_obs, _ = fitted_regressors

        params_0 = jnp.asarray([1.5, 0.0, 0.2], dtype=jnp.float32)
        params_hat, loss_history = regressor_grad.fit(x_obs,
                                                      y_obs,
                                                      params_0,
                                                      N=300,
                                                      lr=5e-2,
                                                      n_step=2000,
                                                      check_per_step=100,
                                                      seed=0)

        assert loss_history[-1] < loss_history[0]

        d0 = np.linalg.norm(np.asarray(params_0) -
                            [TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']])
        d1 = np.linalg.norm(
            params_hat -
            [TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']])
        assert d1 < d0


class TestRegressorEnergyGradValidation:

    def test_missing_jax_func_raises(self):
        with pytest.raises(TypeError):
            RegressorEnergyGrad(None, None, func_x2y=lambda x, p: x)
