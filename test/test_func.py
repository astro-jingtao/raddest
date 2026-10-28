import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np
import pytest

from raddest.GaussHermiteQuadrature import GaussHermiteQuadrature
from raddest.interpolator import MyPchipInterpolator
from raddest.reg.func import (
    generate_piecewise_linear,
    get_saturated_pars,
    saturated_linear,
    saturated_linear_smooth,
    sigmoid,
)
from raddest.reg.func_jax import (
    generate_piecewise_linear_jax,
    saturated_linear_smooth_jax,
    sigmoid as sigmoid_jax,
)


class TestSigmoid:

    def test_jax_matches_numpy(self):
        x = np.linspace(-5, 5, 21)
        np.testing.assert_allclose(sigmoid_jax(x, 0.0, 2.0),
                                   sigmoid(x, 0.0, 2.0),
                                   rtol=1e-6)


class TestSaturatedLinear:

    def test_pars_and_continuity(self):
        k, b, x0, x1, c0, c1 = 1.0, 0.0, -1.0, 1.0, -2.0, 2.0
        n0, n1, m0, m1 = get_saturated_pars(k, b, x0, x1, c0, c1)

        x = np.linspace(-5, 5, 201)
        y = saturated_linear(x, k, b, x0, x1, c0, c1)
        assert np.all(np.isfinite(y))

        # the three branches join continuously at x0 and x1
        assert y[np.argmin(np.abs(x - x0))] == pytest.approx(
            k * x0 + b, abs=1e-6)
        assert y[np.argmin(np.abs(x - x1))] == pytest.approx(
            k * x1 + b, abs=1e-6)

        # explicit-parameters call matches the derived one
        y2 = saturated_linear(x, k, b, x0, x1, c0, c1, n0, n1, m0, m1)
        np.testing.assert_allclose(y, y2)

    def test_smooth_matches_piecewise_far_from_transition(self):
        k, b, x0, x1, c0, c1 = 1.0, 0.0, -1.0, 1.0, -2.0, 2.0
        y_smooth = saturated_linear_smooth(np.array([0.0]), k, b, x0, x1, c0,
                                           c1, width=0.01)
        assert float(y_smooth[0]) == pytest.approx(k * 0 + b, rel=1e-3)

    def test_jax_matches_numpy(self):
        k, b, x0, x1, c0, c1 = 1.0, 0.0, -1.0, 1.0, -2.0, 2.0
        x = np.linspace(-5, 5, 201)
        y_np = saturated_linear_smooth(x, k, b, x0, x1, c0, c1, width=0.05)
        y_jax = saturated_linear_smooth_jax(x, k, b, x0, x1, c0, c1, width=0.05)
        np.testing.assert_allclose(y_jax, y_np, rtol=1e-4, atol=1e-6)


class TestPiecewiseLinear:

    @pytest.mark.parametrize('n_piece', [1, 2, 3])
    def test_jax_matches_numpy(self, n_piece):
        rng = np.random.default_rng(3 + n_piece)
        k_lst = rng.uniform(0.5, 2.0, size=n_piece)
        dx_lst = rng.uniform(0.2, 0.5, size=max(n_piece - 1, 0))
        b0, x0 = 0.0, -1.0

        x = np.linspace(0.5, 4.0, 101)
        f_np = generate_piecewise_linear(n_piece)
        f_jax = generate_piecewise_linear_jax(n_piece)

        y_np = f_np(x, b0, x0, k_lst, dx_lst, width=0.05)
        y_jax = f_jax(x, b0, x0, k_lst, dx_lst, width=0.05)
        np.testing.assert_allclose(y_jax, y_np, rtol=1e-4, atol=1e-6)

    def test_two_piece_tracks_each_line_far_from_knots(self):
        b0, x0 = 0.0, 1.0
        k_lst = np.array([1.0, 3.0])
        dx_lst = np.array([1.0])
        f = generate_piecewise_linear(2)

        y_left = float(f(np.array([0.2]), b0, x0, k_lst, dx_lst, width=0.01)[0])
        y_right = float(f(np.array([2.8]), b0, x0, k_lst, dx_lst, width=0.01)[0])

        b1 = k_lst[0] * x0 + b0 - k_lst[1] * x0
        assert y_left == pytest.approx(k_lst[0] * 0.2 + b0, rel=1e-2)
        assert y_right == pytest.approx(k_lst[1] * 2.8 + b1, rel=1e-2)


class TestGaussHermiteQuadrature:

    def test_1d(self):
        ghq = GaussHermiteQuadrature(dim=1, nodes=10)
        assert ghq.X.shape == (10, )
        assert ghq.W.shape == (10, )
        assert np.sum(ghq.W) == pytest.approx(np.sqrt(np.pi), rel=1e-10)

    def test_2d(self):
        ghq = GaussHermiteQuadrature(dim=2, nodes=5)
        assert ghq.X.shape == (25, 2)
        assert ghq.W.shape == (25, )
        assert np.sum(ghq.W) == pytest.approx(np.pi, rel=1e-10)


class TestMyPchipInterpolator:

    def test_constructor_is_a_broken_stub(self):
        # __init__ calls get_inverse_coefficients_breakpoints, which does not
        # exist anywhere: the class cannot be instantiated as shipped.
        with pytest.raises(AttributeError):
            MyPchipInterpolator(coefficients=None, breakpoints=None)

    def test_from_params_is_a_broken_stub(self):
        # from_params unpacks a literal Ellipsis: it cannot run as shipped.
        with pytest.raises(TypeError):
            MyPchipInterpolator.from_params([1, 2, 3])

    def test_from_data_also_hits_broken_constructor(self):
        x = np.linspace(0, 10, 11)
        y = np.exp(0.3 * x)
        with pytest.raises(AttributeError):
            MyPchipInterpolator.from_data(x, y)
