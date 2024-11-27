# pylint: disable=wrong-import-position
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pzflow import Flow
from jax.scipy.stats import norm

from raddest.tools import get_ext_w_2d_grid, get_ext_w_2d_qgrid, get_p_flow

avoid_heavy = False


@pytest.mark.skipif(avoid_heavy, reason="avoid heavy tests")
class TestGetPFlow:

    def test_in_batch_mode(self):
        x_obs = np.random.normal(size=2048)
        x_err = np.random.normal(size=2048)
        flow = Flow(["x_obs", "x_err"])
        flow.train(pd.DataFrame({"x_obs": x_obs, "x_err": x_err}))

        p_all = get_p_flow(flow, x_obs, x_err, batch_size=None)
        p_batch = get_p_flow(flow, x_obs, x_err, batch_size=256)
        assert p_all.shape == p_batch.shape
        assert np.allclose(p_all, p_batch, equal_nan=True, rtol=1e-4)


class TestExtW2D:

    def test_grid(self):
        x_obs = jnp.array([0.2, 0.4])
        x_err = jnp.array([0.1, 0.2])
        y_obs = jnp.array([0.3, 0.9])
        y_err = jnp.array([0.15, 0.45])

        x_ext, y_ext, x_err_ext, y_err_ext, w = get_ext_w_2d_grid(
            x_obs, y_obs, x_err, y_err, 2)

        assert jnp.allclose(x_ext,
                            jnp.array([[0.25, 0.35] * 2, [0.25, 0.35] * 2]))
        assert jnp.allclose(
            y_ext,
            jnp.array([[0.45, 0.45] + [0.75, 0.75],
                       [0.45, 0.45] + [0.75, 0.75]]))

        assert jnp.allclose(x_err_ext, jnp.array([[0.1] * 4, [0.2] * 4]))
        assert jnp.allclose(y_err_ext, jnp.array([[0.15] * 4, [0.45] * 4]))

        area = 0.2 / 2 * 0.6 / 2
        for i in range(2):
            for j in range(4):
                assert jnp.isclose(
                    w[i, j],
                    area *
                    norm.pdf(x_ext[i, j], loc=x_obs[i], scale=x_err[i]) *
                    norm.pdf(y_ext[i, j], loc=y_obs[i], scale=y_err[i]))

    def test_qgrid(self):
        x_obs = jnp.linspace(0, 1, 1000)
        y_obs = jnp.linspace(0, 1, 1000)
        x_err = jnp.array(np.random.uniform(size=1000))
        y_err = jnp.array(np.random.uniform(size=1000))

        x_ext_q, y_ext_q, x_err_ext_q, y_err_ext_q, w_q = get_ext_w_2d_qgrid(
            x_obs, y_obs, x_err, y_err, 10)
        x_ext, y_ext, x_err_ext, y_err_ext, w = get_ext_w_2d_grid(
            x_obs, y_obs, x_err, y_err, 10)

        assert jnp.allclose(x_ext_q, x_ext)
        assert jnp.allclose(y_ext_q, y_ext)
        assert jnp.allclose(x_err_ext_q, x_err_ext)
        assert jnp.allclose(y_err_ext_q, y_err_ext)
        assert jnp.allclose(w_q, w)
