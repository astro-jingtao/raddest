import itertools

import jax
import jax.numpy as jnp
import numpy as np
from pted import pted as pted_fn
from pted.utils import pted_chunk_numpy
import pytest
from scipy.spatial import KDTree

from raddest.reg import energy
from raddest.reg.constructor import construct_generator_cond
from raddest.reg.energy import (
    _energy_distance_u_jax,
    construct_energy_compare,
    construct_energy_compare_jax,
    construct_energy_loss,
    energy_distance_jax,
    energy_distance_numpy,
    energy_test,
    energy_test_jax,
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
            energy_distance_jax(
                jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
            )
        )
        assert e_jax == pytest.approx(energy_distance_numpy(x, y), rel=RTOL_JAX, abs=1e-4)

    def test_jax_matches_pted_same_distribution(self):
        x, y = make_samples()
        t, _, _ = pted_fn(x, y, permutations=10, return_all=True)
        e_jax = float(
            energy_distance_jax(
                jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
            )
        )
        assert e_jax == pytest.approx(t, rel=RTOL_JAX, abs=1e-4)

    def test_zero_for_identical_samples(self):
        x = np.random.default_rng(1).normal(size=(50, 2))
        e = float(
            energy_distance_jax(
                jnp.asarray(x, dtype=jnp.float32), jnp.asarray(x, dtype=jnp.float32)
            )
        )
        assert e == pytest.approx(0.0, abs=1e-4)

    def test_u_statistic_close_to_v_statistic(self):
        x, y = make_samples(shift=0.3)
        e_u = float(
            _energy_distance_u_jax(
                jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32)
            )
        )
        assert e_u == pytest.approx(energy_distance_numpy(x, y), rel=5e-2, abs=5e-2)

    def test_u_statistic_is_differentiable(self):
        x = jnp.asarray(np.random.default_rng(2).normal(size=(30, 2)), dtype=jnp.float32)
        y = 0.5 * x + jnp.asarray(np.random.default_rng(3).normal(size=(30, 2)), dtype=jnp.float32)

        def shift_y(theta):
            return _energy_distance_u_jax(x, y + theta)

        grad = jax.grad(shift_y)(jnp.asarray(0.3, dtype=jnp.float32))
        assert jnp.isfinite(grad)
        assert float(jnp.abs(grad)) > 0


class TestEnergyDistanceErrors:
    def test_feature_shape_mismatch(self):
        with pytest.raises(ValueError, match="feature shape"):
            energy_distance_jax(jnp.zeros((10, 2)), jnp.zeros((10, 3)))
        with pytest.raises(ValueError, match="feature shape"):
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
        with pytest.raises(ValueError, match="at least 2D"):
            energy_test(np.zeros(10), np.zeros((10, 2)))
        with pytest.raises(ValueError, match="feature shape"):
            energy_test(np.zeros((10, 2)), np.zeros((10, 3)))
        with pytest.raises(ValueError, match="NaN or Inf"):
            energy_test(np.full((10, 2), np.nan), np.zeros((10, 2)))


class TestEnergyTestChunked:
    """Chunked estimation mode (pted's chunk_size/chunk_iter)."""

    # Tolerances calibrated above the measured Monte-Carlo noise floor of the
    # chunk estimator (c=100, I=10 on n=200/150: bias ~+0.02, std ~0.016,
    # max observed |chunk - full| = 0.046).
    ABS_CHUNK_VS_FULL = 0.09
    ABS_CHUNK_VS_PTED = 0.1

    def test_chunk_covering_sample_reproduces_full_statistic(self):
        # With chunk_size >= max(N, M) and chunk_iter=1 every draw uses the
        # whole sample, so the estimate is exactly the full-sample statistic.
        x, y = make_samples(shift=0.3)
        stat_full, _ = energy_test(x, y, permutations=5, rng=np.random.default_rng(3))
        stat_cover, _ = energy_test(
            x, y, permutations=5, rng=np.random.default_rng(3), chunk_size=500, chunk_iter=1
        )
        # not bitwise: the pooled-matrix sum and the per-chunk sums add the
        # same distances in a different order
        assert stat_cover == pytest.approx(stat_full, rel=RTOL_NUMPY)

    def test_chunk_stat_close_to_full_statistic(self):
        x, y = make_samples(shift=0.3)
        stat_full, _ = energy_test(x, y, permutations=5, rng=np.random.default_rng(3))
        stat_chunk, _ = energy_test(
            x, y, permutations=5, rng=np.random.default_rng(3), chunk_size=100, chunk_iter=10
        )
        assert stat_chunk == pytest.approx(stat_full, abs=self.ABS_CHUNK_VS_FULL)

    def test_chunk_matches_pted_chunk_algorithm(self):
        x, y = make_samples(shift=0.3)
        stat_ours, _ = energy_test(
            x, y, permutations=5, rng=np.random.default_rng(3), chunk_size=100, chunk_iter=10
        )
        stat_pted, _ = pted_chunk_numpy(x, y, permutations=5, chunk_size=100, chunk_iter=10)
        # Both are Monte-Carlo estimates of the same expected chunk statistic
        # (bias included); the tolerance covers the two independent noise.
        assert stat_ours == pytest.approx(stat_pted, abs=self.ABS_CHUNK_VS_PTED)

    def test_chunk_retains_discriminating_power(self):
        x_same, y_same = make_samples(n_x=300, n_y=300)
        _, p_same = energy_test(
            x_same,
            y_same,
            permutations=100,
            rng=np.random.default_rng(0),
            chunk_size=100,
            chunk_iter=10,
        )
        assert p_same > 0.05
        x_shift, y_shift = make_samples(n_x=300, n_y=300, shift=1.0)
        _, p_shift = energy_test(
            x_shift,
            y_shift,
            permutations=100,
            rng=np.random.default_rng(0),
            chunk_size=100,
            chunk_iter=10,
        )
        assert p_shift < 0.05

    def test_chunk_deterministic_given_rng(self):
        x, y = make_samples(shift=0.3)
        res1 = energy_test(
            x, y, permutations=10, rng=np.random.default_rng(11), chunk_size=50, chunk_iter=5
        )
        res2 = energy_test(
            x, y, permutations=10, rng=np.random.default_rng(11), chunk_size=50, chunk_iter=5
        )
        assert res1 == res2

    def test_chunk_args_must_be_given_together(self):
        x, y = make_samples(shift=0.3)
        with pytest.raises(ValueError, match="together"):
            energy_test(x, y, permutations=5, rng=np.random.default_rng(0), chunk_size=50)
        with pytest.raises(ValueError, match="together"):
            energy_test(x, y, permutations=5, rng=np.random.default_rng(0), chunk_iter=5)


class TestEnergyTestJax:
    """The vectorized jax permutation test."""

    def test_statistic_matches_numpy(self):
        x, y = make_samples(shift=0.3)
        stat_np, _ = energy_test(x, y, permutations=5, rng=np.random.default_rng(0))
        stat_jax, _ = energy_test_jax(x, y, permutations=5)
        assert stat_jax == pytest.approx(stat_np, rel=RTOL_JAX, abs=1e-4)
        assert stat_jax == pytest.approx(energy_distance_numpy(x, y), rel=RTOL_JAX, abs=1e-4)

    def test_same_distribution_gives_large_p(self):
        x, y = make_samples(n_x=300, n_y=300)
        _, p = energy_test_jax(x, y, permutations=200)
        assert p > 0.05

    def test_shifted_distribution_gives_smallest_p(self):
        x, y = make_samples(n_x=300, n_y=300, shift=1.0)
        _, p = energy_test_jax(x, y, permutations=200)
        assert p == pytest.approx(1.0 / 201)

    def test_seed_reproducible_and_statistic_seed_independent(self):
        x, y = make_samples(shift=0.3)
        res_0 = energy_test_jax(x, y, permutations=50, seed=0)
        res_0_again = energy_test_jax(x, y, permutations=50, seed=0)
        res_1 = energy_test_jax(x, y, permutations=50, seed=1)
        assert res_0 == res_0_again
        assert res_0[0] == pytest.approx(res_1[0], rel=RTOL_JAX, abs=1e-4)

    def test_input_validation(self):
        with pytest.raises(ValueError, match="at least 2D"):
            energy_test_jax(np.zeros(10), np.zeros((10, 2)))
        with pytest.raises(ValueError, match="feature shape"):
            energy_test_jax(np.zeros((10, 2)), np.zeros((10, 3)))


class TestJaxNumpyBackendParity:
    """Same data through the numpy and jax permutation engines must agree.

    The observed statistic is deterministic and must match to float32
    accuracy; the p-values are Monte-Carlo estimates over independent
    permutation streams, so they agree only up to each stream's binomial
    noise (std <= 0.5 / sqrt(permutations)).
    """

    def test_identical_samples_zero_statistic_and_unit_p(self):
        x, _ = make_samples(n_x=120, n_y=120)
        stat_np, p_np = energy_test(x, x, permutations=50, rng=np.random.default_rng(0))
        stat_jax, p_jax = energy_test_jax(x, x, permutations=50)
        assert stat_np == pytest.approx(0.0, abs=1e-12)
        assert stat_jax == pytest.approx(0.0, abs=1e-4)
        assert p_np == pytest.approx(1.0)
        assert p_jax == pytest.approx(1.0)

    def test_shifted_samples_minimal_p_on_both(self):
        x, y = make_samples(n_x=300, n_y=300, shift=1.0)
        stat_np, p_np = energy_test(x, y, permutations=200, rng=np.random.default_rng(0))
        stat_jax, p_jax = energy_test_jax(x, y, permutations=200)
        assert p_np == pytest.approx(1.0 / 201)
        assert p_jax == pytest.approx(1.0 / 201)
        assert stat_jax == pytest.approx(stat_np, rel=RTOL_JAX, abs=1e-4)

    def test_same_distribution_pvalues_within_monte_carlo_noise(self):
        # Each engine estimates the same true p with binomial std
        # sqrt(p(1-p)/B) <= 0.5/sqrt(200) ~ 0.035, so |dp| > 0.25 is a
        # >5-sigma event for two independent streams.
        x, y = make_samples(n_x=300, n_y=300)
        _, p_np = energy_test(x, y, permutations=200, rng=np.random.default_rng(0))
        _, p_jax = energy_test_jax(x, y, permutations=200, seed=1)
        assert abs(p_np - p_jax) < 0.25

    @pytest.mark.parametrize(("n_x", "n_y"), [(37, 53), (53, 37)])
    def test_statistic_matches_numpy_unequal_sizes(self, n_x, n_y):
        # asymmetric splits exercise the mask quadratic form at nx != n/2
        x, y = make_samples(n_x=n_x, n_y=n_y, shift=0.3)
        stat_np, _ = energy_test(x, y, permutations=20, rng=np.random.default_rng(0))
        stat_jax, _ = energy_test_jax(x, y, permutations=20)
        assert stat_jax == pytest.approx(stat_np, rel=RTOL_JAX, abs=1e-4)


def make_energy_grid_inputs(
    n=200, seed=13
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_gen = rng.uniform(1, 10, size=n)
    x_obs_gen = x_gen + rng.normal(scale=0.1, size=n)
    y_kdt = np.linspace(5, 50, 400)
    y_err_kdt = np.full(400, 0.5)
    x_obs = rng.uniform(1, 10, size=n)
    y_obs = 3 * x_obs + 1 + rng.normal(scale=0.6, size=n)
    return x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs


class TestGridBackendParity:
    """The numpy grid compare and the jax grid compare on identical inputs.

    The numpy backend draws fresh generator noise per call; the jax backend
    fixes one noise realization at construction and shares it across the
    grid. Their statistics therefore agree only up to the noise-realization
    spread, which this test measures on the numpy side itself and checks
    the jax statistic falls inside (calibrated: z = -0.7 and -0.3 for the
    good and bad parameter points).
    """

    N_NOISE_DRAWS = 12
    Z_TOL = 5.0

    def _compares(self, permutations: int) -> tuple:
        x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs = make_energy_grid_inputs()
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x + params[1])
        compare_np = construct_energy_compare(
            gen_cond,
            x_gen,
            x_obs_gen,
            y_kdt,
            y_err_kdt,
            x_obs=x_obs,
            y_obs=y_obs,
            permutations=permutations,
        )
        compare_jax = construct_energy_compare_jax(
            lambda x, params, eps: params[0] * x + params[1],
            x_obs,
            y_obs,
            x_gen,
            x_obs_gen,
            y_kdt=y_kdt,
            y_err_kdt=y_err_kdt,
            permutations=permutations,
            seed=0,
        )
        return compare_np, compare_jax

    @pytest.mark.parametrize("params", [(3.0, 1.0), (5.0, 1.0)])
    def test_jax_statistic_within_numpy_noise_band(self, params):
        compare_np, compare_jax = self._compares(permutations=200)
        stats_np = []
        for seed in range(self.N_NOISE_DRAWS):
            np.random.seed(seed)
            stats_np.append(compare_np(*params)[0])
        stats_np = np.asarray(stats_np)
        mu, sd = stats_np.mean(), stats_np.std()
        assert sd > 0  # the noise band is measurable, not degenerate
        stat_jax = float(compare_jax(np.asarray([params]))[0, 0])
        assert stat_jax == pytest.approx(mu, abs=self.Z_TOL * sd)


class TestConstructEnergyCompareJax:
    """The batch grid compare closure (jax grid backend)."""

    @staticmethod
    def _make_inputs(
        n=200, seed=13
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_gen = rng.uniform(1, 10, size=n)
        x_obs_gen = x_gen + rng.normal(scale=0.1, size=n)
        y_kdt = np.linspace(5, 50, 400)
        y_err_kdt = np.full(400, 0.5)
        x_obs = rng.uniform(1, 10, size=n)
        y_obs = 3 * x_obs + 1 + rng.normal(scale=0.6, size=n)
        return x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs

    @staticmethod
    def linear_jax(x, params, eps) -> jax.Array:
        return params[0] * x + params[1] + params[2] * eps

    def test_batch_rows_match_single_point_generation(self):
        # replicate the closure's fixed-noise generation for one grid point
        # and check the batch result against energy_test_jax on that sample
        x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._make_inputs()
        seed = 0
        compare = construct_energy_compare_jax(
            self.linear_jax,
            x_obs,
            y_obs,
            x_gen,
            x_obs_gen,
            y_kdt=y_kdt,
            y_err_kdt=y_err_kdt,
            permutations=50,
            seed=seed,
        )
        params = np.array([[3.0, 1.0, 0.6], [5.0, 1.0, 0.6]])
        res = compare(params)
        assert res.shape == (2, 2)

        rng = np.random.default_rng(seed)
        eps_scatter = rng.standard_normal(len(x_gen))
        eps_obs = rng.standard_normal(len(x_gen))
        for i, p in enumerate(params):
            y_pred = p[0] * x_gen + p[1] + p[2] * eps_scatter
            y_pred = np.clip(y_pred, y_kdt.min(), y_kdt.max())
            y_err_matched = y_err_kdt[np.argmin(np.abs(y_kdt[:, None] - y_pred[None, :]), axis=0)]
            y_obs_gen = y_pred + eps_obs * y_err_matched
            gen = np.stack([x_obs_gen, y_obs_gen], axis=1)
            obs = np.stack([x_obs, y_obs], axis=1)
            stat, pval = energy_test_jax(gen, obs, permutations=50, seed=seed)
            assert res[i, 0] == pytest.approx(stat, rel=RTOL_JAX, abs=1e-4)
            assert res[i, 1] == pytest.approx(pval, rel=1e-6)

    def test_good_params_smaller_statistic_and_larger_p(self):
        x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._make_inputs()
        compare = construct_energy_compare_jax(
            self.linear_jax,
            x_obs,
            y_obs,
            x_gen,
            x_obs_gen,
            y_kdt=y_kdt,
            y_err_kdt=y_err_kdt,
            permutations=50,
        )
        params = np.array([[3.0, 1.0, 0.6], [5.0, 1.0, 0.6]])
        stat_good, p_good = compare(params[[0]])[0]
        stat_bad, p_bad = compare(params[[1]])[0]
        assert stat_good < stat_bad
        assert p_good > p_bad

    def test_grid_of_points_runs(self):
        x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._make_inputs()
        compare = construct_energy_compare_jax(
            self.linear_jax,
            x_obs,
            y_obs,
            x_gen,
            x_obs_gen,
            y_kdt=y_kdt,
            y_err_kdt=y_err_kdt,
            permutations=20,
        )
        grid = np.array(list(itertools.product([2.8, 3.0, 3.2], [0.5, 1.0], [0.3, 0.6])))
        res = compare(grid)
        assert res.shape == (12, 2)
        assert np.all((res[:, 1] > 0) & (res[:, 1] <= 1))

    def test_invalid_grid_shape_raises(self):
        x_gen, x_obs_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._make_inputs()
        compare = construct_energy_compare_jax(
            self.linear_jax,
            x_obs,
            y_obs,
            x_gen,
            x_obs_gen,
            y_kdt=y_kdt,
            y_err_kdt=y_err_kdt,
            permutations=20,
        )
        with pytest.raises(ValueError, match="must be 2D"):
            compare(np.array([3.0, 1.0, 0.6]))


class TestMatchYErrJax:
    def test_matches_kdtree(self):
        rng = np.random.default_rng(11)
        y_pred = rng.uniform(-2, 2, size=100)
        y_kdt = rng.uniform(-2, 2, size=400)
        y_err_kdt = rng.uniform(0.1, 1.0, size=400)

        _, idx = KDTree(y_kdt.reshape(-1, 1)).query(y_pred.reshape(-1, 1))
        expected = y_err_kdt[idx]

        got = np.asarray(
            match_y_err_jax(
                jnp.asarray(y_pred, dtype=jnp.float32),
                jnp.asarray(y_kdt, dtype=jnp.float32),
                jnp.asarray(y_err_kdt, dtype=jnp.float32),
            )
        )
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
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x + params[1])
        compare = construct_energy_compare(
            gen_cond, x_gen, x_gen, y_kdt, y_err_kdt, x_obs=x_obs, y_obs=y_obs, permutations=50
        )
        np.random.seed(0)
        stat_good, p_good = compare(3.0, 1.0)
        stat_bad, p_bad = compare(5.0, 1.0)
        assert stat_good < stat_bad
        assert p_good > p_bad

    def test_return_types(self):
        x_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._setup()
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x + params[1])
        compare = construct_energy_compare(
            gen_cond, x_gen, x_gen, y_kdt, y_err_kdt, x_obs=x_obs, y_obs=y_obs, permutations=20
        )
        np.random.seed(1)
        stat, p = compare(3.0, 1.0)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0 < p <= 1

    def test_chunk_kwargs_reach_the_test(self, monkeypatch):
        x_gen, y_kdt, y_err_kdt, x_obs, y_obs = self._setup()
        gen_cond = construct_generator_cond(lambda x, params: params[0] * x + params[1])
        compare_chunk = construct_energy_compare(
            gen_cond,
            x_gen,
            x_gen,
            y_kdt,
            y_err_kdt,
            x_obs=x_obs,
            y_obs=y_obs,
            permutations=20,
            chunk_size=100,
            chunk_iter=10,
        )
        stat, p = compare_chunk(3.0, 1.0)
        assert isinstance(stat, float)
        assert 0 < p <= 1
        # the generator draws a fresh synthetic sample per call, so plumbing
        # is proven by spying on energy_test rather than by value comparison
        calls = {}
        real_energy_test = energy.energy_test

        def spy(x, y, **kwargs):
            calls.update(kwargs)
            return real_energy_test(x, y, **kwargs)

        monkeypatch.setattr(energy, "energy_test", spy)
        compare_chunk(3.0, 1.0)
        assert calls == {"permutations": 20, "chunk_size": 100, "chunk_iter": 10}


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

        loss = construct_energy_loss(
            linear_jax, x_obs, y_obs, x_gen, x_obs_gen, y_kdt=y_kdt, y_err_kdt=y_err_kdt, seed=0
        )
        loss_good = loss(jnp.asarray([3.0, 1.0, 0.6], dtype=jnp.float32))
        loss_bad = loss(jnp.asarray([5.0, 1.0, 0.6], dtype=jnp.float32))
        assert float(loss_good) < float(loss_bad)

        grads = jax.grad(loss)(jnp.asarray([3.0, 1.0, 0.6], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(grads))
