import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
from pted import pted as pted_fn
import pytest
from scipy.spatial import KDTree

from raddest.reg.constructor import construct_generator_cond
from raddest.reg.energy import (
    _energy_distance_u_jax,
    construct_energy_compare,
    construct_energy_loss,
    energy_distance_jax,
    energy_distance_numpy,
    energy_test,
    match_y_err_jax,
)

# Tolerances calibrated above the measured float32 noise floor of the
# squared-norm identity vs direct cdist (worst observed ~5e-5 relative).
RTOL_JAX = 1e-3
RTOL_NUMPY = 1e-12


def make_samples(seed=42, n_x=200, n_y=150, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_x, 2)) + shift
    y = rng.normal(size=(n_y, 2))
    return x, y


class TestEnergyDistanceParity:
    """Differential parity against the pted reference implementation."""

    def test_numpy_matches_pted_statistic(self):
        x, y = make_samples(shift=0.3)
        t, _, _ = pted_fn(x, y, permutations=10, return_all=True)
        assert energy_distance_numpy(x, y) == pytest.approx(t, rel=RTOL_NUMPY)

    def test_jax_matches_numpy(self):
        x, y = make_samples(shift=0.3)
        e_jax = float(
            energy_distance_jax(jnp.asarray(x, dtype=jnp.float32),
                                jnp.asarray(y, dtype=jnp.float32)))
        assert e_jax == pytest.approx(energy_distance_numpy(x, y),
                                      rel=RTOL_JAX,
                                      abs=1e-4)

    def test_jax_matches_pted_same_distribution(self):
        x, y = make_samples()
        t, _, _ = pted_fn(x, y, permutations=10, return_all=True)
        e_jax = float(
            energy_distance_jax(jnp.asarray(x, dtype=jnp.float32),
                                jnp.asarray(y, dtype=jnp.float32)))
        assert e_jax == pytest.approx(t, rel=RTOL_JAX, abs=1e-4)

    def test_zero_for_identical_samples(self):
        x = np.random.default_rng(1).normal(size=(50, 2))
        e = float(energy_distance_jax(jnp.asarray(x, dtype=jnp.float32),
                                      jnp.asarray(x, dtype=jnp.float32)))
        assert e == pytest.approx(0.0, abs=1e-4)

    def test_u_statistic_close_to_v_statistic(self):
        x, y = make_samples(shift=0.3)
        e_u = float(
            _energy_distance_u_jax(jnp.asarray(x, dtype=jnp.float32),
                                   jnp.asarray(y, dtype=jnp.float32)))
        assert e_u == pytest.approx(energy_distance_numpy(x, y),
                                    rel=5e-2,
                                    abs=5e-2)

    def test_u_statistic_is_differentiable(self):
        x = jnp.asarray(np.random.default_rng(2).normal(size=(30, 2)),
                        dtype=jnp.float32)
        y = 0.5 * x + jnp.asarray(
            np.random.default_rng(3).normal(size=(30, 2)), dtype=jnp.float32)

        def shift_y(theta):
            return _energy_distance_u_jax(x, y + theta)

        grad = jax.grad(shift_y)(jnp.asarray(0.3, dtype=jnp.float32))
        assert jnp.isfinite(grad)
        assert float(jnp.abs(grad)) > 0


class TestEnergyDistanceErrors:

    def test_feature_shape_mismatch(self):
        with pytest.raises(ValueError, match='feature shape'):
            energy_distance_jax(jnp.zeros((10, 2)), jnp.zeros((10, 3)))
        with pytest.raises(ValueError, match='feature shape'):
            energy_distance_numpy(np.zeros((10, 2)), np.zeros((10, 3)))


class TestEnergyTest:

    def test_same_distribution_gives_large_p(self):
        x, y = make_samples(n_x=300, n_y=300)
        _, p = energy_test(x, y, permutations=200, rng=np.random.default_rng(0))
        assert p > 0.05

    def test_shifted_distribution_gives_small_p(self):
        x, y = make_samples(n_x=300, n_y=300, shift=1.0)
        _, p = energy_test(x, y, permutations=200, rng=np.random.default_rng(0))
        assert p < 0.05

    def test_statistic_increases_with_shift(self):
        rng = np.random.default_rng(7)
        y = rng.normal(size=(200, 2))
        e_small = energy_test(y + 0.1, y + 0.5, permutations=10, rng=rng)[0]
        e_large = energy_test(y + 0.1, y + 2.0, permutations=10, rng=rng)[0]
        assert e_large > e_small

    def test_p_value_resolution(self):
        x, y = make_samples(shift=5.0, n_x=100, n_y=100)
        _, p = energy_test(x, y, permutations=99, rng=np.random.default_rng(0))
        assert p == pytest.approx(1.0 / 100)

    def test_input_validation(self):
        with pytest.raises(ValueError, match='at least 2D'):
            energy_test(np.zeros(10), np.zeros((10, 2)))
        with pytest.raises(ValueError, match='feature shape'):
            energy_test(np.zeros((10, 2)), np.zeros((10, 3)))
        with pytest.raises(ValueError, match='NaN or Inf'):
            energy_test(np.full((10, 2), np.nan), np.zeros((10, 2)))


class TestMatchYErrJax:

    def test_matches_kdtree(self):
        rng = np.random.default_rng(11)
        y_pred = rng.uniform(-2, 2, size=100)
        y_kdt = rng.uniform(-2, 2, size=400)
        y_err_kdt = rng.uniform(0.1, 1.0, size=400)

        _, idx = KDTree(y_kdt.reshape(-1, 1)).query(y_pred.reshape(-1, 1))
        expected = y_err_kdt[idx]

        got = np.asarray(
            match_y_err_jax(jnp.asarray(y_pred, dtype=jnp.float32),
                            jnp.asarray(y_kdt, dtype=jnp.float32),
                            jnp.asarray(y_err_kdt, dtype=jnp.float32)))
        np.testing.assert_allclose(got, expected, rtol=1e-5)


class TestConstructEnergyCompare:

    def _setup(self):
        rng = np.random.default_rng(5)
        x_gen = rng.uniform(1, 10, size=200)
        y_kdt = np.linspace(5, 50, 400)
        y_err_kdt = np.full(400, 0.5)
        x_obs = rng.uniform(1, 10, size=200)
        y_obs = 3 * x_obs + 1 + rng.normal(scale=0.5, size=200)
        return x_gen, y_kdt, y_err_kdt, x_obs, y_obs

    def test_good_fit_smaller_statistic(self):
        x_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._setup()
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x +
                                            params[1])
        compare = construct_energy_compare(gen_cond,
                                            x_gen,
                                            x_gen,
                                            y_kdt,
                                            y_err_kdt,
                                            x_obs,
                                            y_obs,
                                            permutations=50)
        np.random.seed(0)
        stat_good, p_good = compare(3.0, 1.0)
        stat_bad, p_bad = compare(5.0, 1.0)
        assert stat_good < stat_bad
        assert p_good > p_bad

    def test_return_types(self):
        x_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._setup()
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x +
                                            params[1])
        compare = construct_energy_compare(gen_cond,
                                            x_gen,
                                            x_gen,
                                            y_kdt,
                                            y_err_kdt,
                                            x_obs,
                                            y_obs,
                                            permutations=20)
        np.random.seed(1)
        stat, p = compare(3.0, 1.0)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0 < p <= 1


class TestConstructEnergyLoss:

    def test_loss_smaller_for_true_params_and_differentiable(self):
        rng = np.random.default_rng(13)
        n = 200
        x_gen = rng.uniform(1, 10, size=n)
        x_obs_gen = x_gen + rng.normal(scale=0.1, size=n)
        y_kdt = np.linspace(5, 50, 400)
        y_err_kdt = np.full(400, 0.5)
        x_obs = rng.uniform(1, 10, size=n)
        y_obs = 3 * x_obs + 1 + rng.normal(scale=0.6, size=n)

        def linear_jax(x, params, eps):
            return params[0] * x + params[1] + params[2] * eps

        loss = construct_energy_loss(linear_jax,
                                     x_obs,
                                     y_obs,
                                     x_gen,
                                     x_obs_gen,
                                     y_kdt,
                                     y_err_kdt,
                                     seed=0)
        loss_good = loss(jnp.asarray([3.0, 1.0, 0.6], dtype=jnp.float32))
        loss_bad = loss(jnp.asarray([5.0, 1.0, 0.6], dtype=jnp.float32))
        assert float(loss_good) < float(loss_bad)

        grads = jax.grad(loss)(jnp.asarray([3.0, 1.0, 0.6], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(grads))
