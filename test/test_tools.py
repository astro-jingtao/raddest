# pylint: disable=wrong-import-position
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from numpy.polynomial.hermite import hermgauss
import pandas as pd
import pytest
from pzflow import Flow

from raddest.tools import (
    concat,
    get_ext_w,
    get_ext_w_2d,
    get_ext_w_2d_grid,
    get_ext_w_2d_inter_grid,
    get_ext_w_2d_qgrid,
    get_ext_w_2d_qmc,
    get_ext_w_inter_grid,
    get_ext_w_qmc,
    get_p_flow,
    is_2power,
    round_to_2power,
    transform_data,
)
from raddest.transfer import NegLogPlus

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
        # float32 batching reorders reductions; tolerance stays above that floor
        assert np.allclose(p_all, p_batch, equal_nan=True, rtol=1e-3, atol=1e-7)


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


class TestExtW1D:

    def test_get_ext_w_matches_formula(self):
        x_obs = jnp.array([0.0, 1.0])
        x_err = jnp.array([0.1, 0.2])
        x_ext, _x_err_ext, w = get_ext_w(5, x_obs, x_err)
        assert x_ext.shape == (2, 5)
        # first quadrature point: sqrt(2) * sig * x_0 + mu
        nodes, weights = hermgauss(5)
        assert float(x_ext[0, 0]) == pytest.approx(
            float(jnp.sqrt(2) * 0.1 * nodes[0]), abs=1e-6)
        assert float(w[0, 0]) == pytest.approx(weights[0] / np.sqrt(np.pi),
                                               rel=1e-5)

    def test_qmc_rounds_to_2power(self, capsys):
        x_obs = jnp.array([0.0, 1.0])
        x_err = jnp.array([0.1, 0.2])
        x_ext, _x_err_ext, w = get_ext_w_qmc(5, x_obs, x_err)
        assert 'rounding to' in capsys.readouterr().out
        assert x_ext.shape == (2, 8)
        # each row of weights sums to 1
        assert float(jnp.sum(w)) == pytest.approx(2.0, rel=1e-4)

    def test_inter_grid(self):
        x_obs = jnp.array([0.0, 1.0])
        x_err = jnp.array([0.1, 0.2])
        x_ext, x_err_ext, _w = get_ext_w_inter_grid(6, x_obs, x_err)
        assert x_ext.shape == (2, 6)
        assert x_err_ext.shape == (2, 6)


class TestExtW2DVariants:

    DATA = {
        'x_obs': jnp.array([0.2, 0.4]),
        'y_obs': jnp.array([0.3, 0.9]),
        'x_err': jnp.array([0.1, 0.2]),
        'y_err': jnp.array([0.15, 0.45]),
    }

    def test_ghq_2d(self):
        d = self.DATA
        x_ext, _y_ext, _x_err_ext, _y_err_ext, w = get_ext_w_2d(
            d['x_obs'], d['y_obs'], d['x_err'], d['y_err'], 3)
        assert x_ext.shape == (2, 9)
        assert w.shape == (9, )

    def test_qmc_2d_rounds(self, capsys):
        d = self.DATA
        x_ext, _y_ext, _, _, w = get_ext_w_2d_qmc(d['x_obs'], d['y_obs'],
                                                 d['x_err'], d['y_err'], 5)
        assert 'rounding to' in capsys.readouterr().out
        assert x_ext.shape == (2, 8)
        assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-4)

    def test_inter_grid_2d(self):
        d = self.DATA
        x_ext, _y_ext, _, _, _w = get_ext_w_2d_inter_grid(d['x_obs'], d['y_obs'],
                                                        d['x_err'], d['y_err'],
                                                        3)
        assert x_ext.shape == (2, 9)

    def test_empty_input_short_circuits(self):
        # the wrapper reads args[0], so the arrays must be passed positionally
        for func in (get_ext_w_2d, get_ext_w_2d_qmc, get_ext_w_2d_inter_grid,
                     get_ext_w_2d_grid, get_ext_w_2d_qgrid):
            out = func(jnp.array([]), jnp.array([]), jnp.array([]),
                       jnp.array([]), 3)
            assert len(out) == 5
            assert all(o.size == 0 for o in out)


class TestSmallHelpers:

    def test_round_to_2power(self):
        assert round_to_2power(5) == 8
        assert round_to_2power(8) == 8

    def test_is_2power(self):
        assert is_2power(8)
        assert not is_2power(5)

    def test_concat_skips_empty(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([])
        c = jnp.array([3.0])
        np.testing.assert_allclose(np.asarray(concat(a, b, c)), [1, 2, 3])

    def test_transform_data_applies_transfer(self):
        x = jnp.array([1.0, 2.0])
        x_err = jnp.array([0.1, 0.1])
        u, v = transform_data(x, x_err, NegLogPlus(), [1.5, 0.1])
        assert u.shape == x.shape
        assert v.shape == x_err.shape
