# pylint:disable=wrong-import-position


import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.api.post_processer import post_shift
from raddest.transfer import NegLogPlus

pytestmark = pytest.mark.slow

avoid_heavy = False


@pytest.mark.skipif(avoid_heavy, reason="avoid heavy tests")
class TestPostShift:
    def test_post_shift(self):

        x = np.random.uniform(1, 3, size=10000)
        x_err = np.random.uniform(0.1, 0.5, size=10000) + x * 0.6
        x_obs = np.random.normal(x, x_err)

        x_err = jnp.asarray(x_err)
        x_obs = jnp.asarray(x_obs)
        x_obs_shift = x_obs + 0.5

        flow_vi = FlowVI(
            x_min=0,
            x_max=x_obs.max(),
            x_err_min=0,
            x_err_max=x_err.max() * 1.2,
            transfer=NegLogPlus(),
            transfer_params=[np.median(x_obs), np.median(x_err)],
        )

        flow_vi.fit(
            x_obs,
            x_err,
            n_pre=100,
            n_train=1000,
            n_node=21,
            verbose=False,
            x_min=0,
            x_max=x_obs.max(),
            x_err_min=0,
            x_err_max=x_err.max() * 1.2,
            check_per_step=10,
            early_stop=True,
            max_stuck=5,
            stuck_frac=0.01,
        )

        _, p_old = flow_vi.ks_test(x_obs_shift, x_err, N=10000)

        flow_vi_new = post_shift(x_obs_shift, x_err, flow_vi, shift_method="median")

        _, p_new = flow_vi_new.ks_test(x_obs_shift, x_err, N=10000)

        assert p_new > p_old

        assert jnp.isclose(
            flow_vi_new.log_likelihood(1, 1),
            flow_vi.log_likelihood(1 + flow_vi_new.transfer_params[0], 1),
        )
