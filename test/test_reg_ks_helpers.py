import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.reg.ks import (
    compare_loglinear,
    gen_xy_obs_loglinear,
    gen_xy_obs_loglinear_neglogplus,
    get_ks_reg_loglinear,
    linear_scatter,
    loglinear_scatter,
    loglinear_scatter_student_t,
    loglinear_x2y,
)
from raddest.transfer import NegLogPlus


class TestScatterFunctions:

    def test_loglinear_scatter(self):
        np.random.seed(0)
        x = np.array([1.0, 10.0, 100.0])
        y = loglinear_scatter(x, (2.0, 0.5, 0.0))
        np.testing.assert_allclose(y, 10**(2 * np.log10(x) + 0.5), rtol=1e-6)

    def test_linear_scatter(self):
        np.random.seed(0)
        x = np.array([1.0, 2.0])
        y = linear_scatter(x, (2.0, 1.0, 0.0))
        np.testing.assert_allclose(y, 2 * x + 1, rtol=1e-6)

    def test_loglinear_scatter_student_t(self):
        np.random.seed(0)
        x = np.array([1.0, 10.0])
        y = loglinear_scatter_student_t(x, (2.0, 0.5, 100.0, 0.0))
        np.testing.assert_allclose(y, 10**(2 * np.log10(x) + 0.5), rtol=1e-3)


class TestLogLinearX2Y:

    def _inputs(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(1, 10, size=50)
        x_obs = x + rng.normal(scale=0.05, size=50)
        y_kdt = np.linspace(2, 100, 200)
        y_err_kdt = np.linspace(0.1, 1.0, 200)
        return x, x_obs, y_kdt, y_err_kdt

    def test_default_and_more(self):
        x, x_obs, y_kdt, y_err_kdt = self._inputs()
        np.random.seed(0)
        out = loglinear_x2y(x, x_obs, y_kdt, y_err_kdt, 2.0, 0.5, 0.1)
        assert len(out) == 2
        np.random.seed(1)
        out_more = loglinear_x2y(x,
                                 x_obs,
                                 y_kdt,
                                 y_err_kdt,
                                 2.0,
                                 0.5,
                                 0.1,
                                 return_more=True)
        assert len(out_more) == 4


class TestGenXYObsLoglinear:

    def _args(self):
        rng = np.random.default_rng(2)
        x_obs = rng.uniform(1, 3, size=100)
        x_err = rng.uniform(0.05, 0.15, size=100)
        y = 10**(2 * np.log10(x_obs) + 0.5 + rng.normal(scale=0.1, size=100))
        y_err = 0.05 * y

        def tiny_flow(v_obs, v_err):
            flow = FlowVI(x_min=float(v_obs.min()) * 0.5,
                          x_max=float(v_obs.max()) * 1.1,
                          x_err_min=0,
                          x_err_max=float(v_err.max()) * 1.2,
                          transfer=NegLogPlus(),
                          transfer_params=[
                              float(np.median(v_obs)),
                              float(np.median(v_err))
                          ])
            flow.fit(v_obs,
                     v_err,
                     n_pre=10,
                     n_train=20,
                     n_node=5,
                     early_stop=False,
                     x_min=float(v_obs.min()) * 0.5,
                     x_max=float(v_obs.max()) * 1.1,
                     x_err_min=0,
                     x_err_max=float(v_err.max()) * 1.2)
            return flow

        return tiny_flow(x_obs, x_err), tiny_flow(y, y_err)

    def _ranges(self, xflow, yflow):
        # mirror RegressorBase._set_range: the flows' own fitted bounds
        return ([xflow.x_min, xflow.x_max], [xflow.x_err_min, xflow.x_err_max],
                [yflow.x_min, yflow.x_max], [yflow.x_err_min, yflow.x_err_max])

    def test_gen_and_more(self):
        xflow, yflow = self._args()
        ranges = self._ranges(xflow, yflow)
        tkw = {
            'x_transfer': xflow.transfer,
            'x_transfer_params': xflow.transfer_params,
            'y_transfer': yflow.transfer,
            'y_transfer_params': yflow.transfer_params,
        }
        np.random.seed(0)
        out = gen_xy_obs_loglinear(xflow.flow, yflow.flow, *ranges,
                                   2.0, 0.5, 0.1,
                                   n=50,
                                   return_numpy=True,
                                   **tkw)
        assert len(out) == 2
        for o in out:
            assert np.asarray(o).shape == (50, )

        np.random.seed(1)
        out_more = gen_xy_obs_loglinear(xflow.flow, yflow.flow, *ranges,
                                        2.0, 0.5, 0.1,
                                        n=50,
                                        return_numpy=True,
                                        return_more=True,
                                        **tkw)
        assert len(out_more) == 6

    def test_neglogplus_variant(self):
        xflow, yflow = self._args()
        ranges = self._ranges(xflow, yflow)
        np.random.seed(2)
        x_obs_gen, y_obs_gen = gen_xy_obs_loglinear_neglogplus(
            xflow.flow, yflow.flow, *ranges, 2.0, 0.5, 0.1,
            xflow.transfer_params, yflow.transfer_params,
            n=50,
            return_numpy=True)
        assert np.asarray(x_obs_gen).shape == (50, )


class TestGetKSRegLoglinear:

    def test_runs_with_nboot(self):
        rng = np.random.default_rng(3)
        x_gen = rng.uniform(1, 10, size=60)
        y_kdt = np.linspace(2, 100, 200)
        y_err_kdt = np.full(200, 0.5)
        x_obs = rng.uniform(1, 10, size=60)
        y_obs = 10**(2 * np.log10(x_obs) + 0.5 +
                     rng.normal(scale=0.1, size=60))

        np.random.seed(0)
        D, p = get_ks_reg_loglinear(x_gen,
                                    x_gen,
                                    y_kdt,
                                    y_err_kdt, (2.0, 0.5, 0.1),
                                    x_obs,
                                    y_obs,
                                    nboot=20,
                                    n_jobs=1)
        assert 0 <= p <= 1


class TestCompareLoglinear:

    def test_runs(self):
        rng = np.random.default_rng(11)
        x_gen = rng.uniform(1, 10, size=50)
        y_kdt = np.linspace(2, 100, 200)
        y_err_kdt = np.full(200, 0.5)
        x_obs = rng.uniform(1, 10, size=50)
        y_obs = 10**(2 * np.log10(x_obs) + 0.5 +
                     rng.normal(scale=0.1, size=50))

        np.random.seed(0)
        D, p = compare_loglinear(2.0,
                                 0.5,
                                 0.1,
                                 x_gen,
                                 x_gen,
                                 y_kdt,
                                 y_err_kdt,
                                 x_obs,
                                 y_obs)
        assert 0 <= p <= 1
