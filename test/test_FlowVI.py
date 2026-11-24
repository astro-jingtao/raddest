import os
import shutil

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.transfer import NegLogPlus

pytestmark = pytest.mark.slow


def fit_tiny_flowvi():
    x = np.random.uniform(1, 3, size=200)
    x_err = np.random.uniform(0.1, 0.5, size=200) + x * 0.1
    x_obs = np.random.normal(x, x_err)

    flow_vi = FlowVI(
        x_min=0,
        x_max=float(x_obs.max()),
        x_err_min=0,
        x_err_max=float(x_err.max()) * 1.2,
        transfer=NegLogPlus(),
        transfer_params=[np.median(x_obs), np.median(x_err)],
    )
    flow_vi.fit(
        x_obs,
        x_err,
        n_pre=10,
        n_train=20,
        n_node=5,
        verbose=False,
        x_min=0,
        x_max=float(x_obs.max()),
        x_err_min=0,
        x_err_max=float(x_err.max()) * 1.2,
        early_stop=False,
    )
    return flow_vi, jnp.asarray(x_obs), jnp.asarray(x_err)


class TestFlowVILogLikelihood:
    """Regression tests: log_likelihood must pass a valid int_method down.

    FlowVI.log_likelihood used to forward a ``qmc`` keyword that
    ``construct_x_xerr_w`` does not accept (TypeError).
    """

    def test_default(self):
        flow_vi, x_obs, x_err = fit_tiny_flowvi()
        res = flow_vi.log_likelihood(x_obs, x_err)
        assert jnp.isfinite(res)

    def test_qmc_true(self):
        flow_vi, x_obs, x_err = fit_tiny_flowvi()
        res = flow_vi.log_likelihood(x_obs, x_err, qmc=True)
        assert jnp.isfinite(res)


class TestFlowVISaveLoad:
    def test_base_case(self):

        flowvi = FlowVI(transfer=NegLogPlus(), x_min=0, x_max=10, x_err_min=0, x_err_max=10)

        flowvi.save("test_flowvi")
        assert os.path.exists("test_flowvi")

        flowvi_loaded = FlowVI.load("test_flowvi")

        assert flowvi_loaded.x_min == flowvi.x_min
        assert flowvi_loaded.x_max == flowvi.x_max
        assert flowvi_loaded.x_err_min == flowvi.x_err_min
        assert flowvi_loaded.x_err_max == flowvi.x_err_max

        shutil.rmtree("test_flowvi")

    def test_no_flow_case(self):

        flowvi = FlowVI(transfer=NegLogPlus(), x_min=0, x_max=10, x_err_min=0, x_err_max=10)

        flowvi.save("test_flowvi")
        assert os.path.exists("test_flowvi")

        flowvi_loaded = FlowVI.load("test_flowvi")
        assert flowvi_loaded.flow is None

        shutil.rmtree("test_flowvi")

    def test_force_save(self):

        flowvi = FlowVI(transfer=NegLogPlus(), x_min=0, x_max=10, x_err_min=0, x_err_max=10)

        os.makedirs("test_flowvi", exist_ok=True)
        with open("test_flowvi/test_flowvi", "w", encoding="utf-8") as f:
            f.write("test")

        with pytest.raises(RuntimeError):
            flowvi.save("test_flowvi")

        flowvi.save("test_flowvi", force=True)
        FlowVI.load("test_flowvi")
        shutil.rmtree("test_flowvi")

        with open("test_flowvi", "w", encoding="utf-8") as f:
            f.write("test")

        with pytest.raises(RuntimeError):
            flowvi.save("test_flowvi")

        flowvi.save("test_flowvi", force=True)
        FlowVI.load("test_flowvi")
        shutil.rmtree("test_flowvi")
