# pylint: disable=wrong-import-position
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.shrink import err_add_noise, shrink_data, shrink_possion

seed = 42
np.random.seed(seed)


class TestShrinkPossion:

    def test_handle_negative(self):

        x = np.random.uniform(0, 0.2, size=10_000)
        x_err = np.sqrt(x) + np.random.uniform(0.4, 0.6, size=10_000)
        x_obs = x + np.random.normal(0, x_err)

        # should not raise error
        _, _ = shrink_possion(x_obs, x_err)

    def test_handle_err_noise_negative(self):

        x = np.random.uniform(2, 3, size=10_000)
        x_err = 0.1 * np.sqrt(x) + np.random.uniform(0.1, 5, size=10_000)
        x_obs = x + np.random.normal(0, x_err)

        _, x_err_shrink = shrink_possion(x_obs, x_err)

        assert np.all(x_err_shrink >= 0)


class TestShrinkDataMethods:

    @pytest.fixture
    def gauss_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 1, size=500)
        x_err = rng.uniform(0.1, 0.5, size=500)
        x_obs = rng.normal(x, x_err)
        return x_obs, x_err

    def test_gg(self, gauss_data):
        x_obs, x_err = gauss_data
        x_sh, _x_err_sh = shrink_data(x_obs, x_err, method='gg')
        assert x_sh.shape == x_obs.shape
        # shrinkage pulls extreme values toward the mean
        assert np.abs(x_sh - x_obs.mean()).max() < np.abs(
            x_obs - x_obs.mean()).max()

    def test_poisson_and_snr(self, gauss_data):
        x_obs, x_err = gauss_data
        for method in ['poisson', 'SNR']:
            x_sh, x_err_sh = shrink_data(x_obs, x_err, method=method)
            assert np.isfinite(x_sh).all()
            assert (x_err_sh >= 0).all()

    def test_none_passthrough(self, gauss_data):
        x_obs, x_err = gauss_data
        x_sh, x_err_sh = shrink_data(x_obs, x_err, method='none')
        np.testing.assert_array_equal(x_sh, x_obs)
        np.testing.assert_array_equal(x_err_sh, x_err)

    def test_invalid_method_raises(self, gauss_data):
        x_obs, x_err = gauss_data
        with pytest.raises(ValueError, match='Invalid shrinkage method'):
            shrink_data(x_obs, x_err, method='bogus')

    def test_recalibrate(self, gauss_data):
        x_obs, x_err = gauss_data
        x_sh, _ = shrink_data(x_obs, x_err, method='gg', to_recalibrate=True)
        assert x_sh.shape == x_obs.shape

    def test_jax_array_roundtrip(self, gauss_data):
        x_obs, x_err = gauss_data
        x_sh, x_err_sh = shrink_data(jnp.asarray(x_obs), jnp.asarray(x_err),
                                     method='gg')
        assert isinstance(x_sh, jnp.ndarray)
        assert isinstance(x_err_sh, jnp.ndarray)

    def test_auto_verbose_and_best_info(self, gauss_data, capsys):
        x_obs, x_err = gauss_data
        np.random.seed(0)
        out = shrink_data(x_obs,
                          x_err,
                          method='auto',
                          to_recalibrate=True,
                          return_best_info=True,
                          verbose=True)
        assert len(out) == 3
        x_sh, _x_err_sh, info = out
        assert 'recalibrate' in info
        assert 'Warning' in capsys.readouterr().out
        assert np.isfinite(x_sh).all()

    def test_auto_plain(self, gauss_data):
        x_obs, x_err = gauss_data
        np.random.seed(1)
        x_sh, _x_err_sh = shrink_data(x_obs, x_err, method='auto')
        assert np.isfinite(x_sh).all()


class TestErrAddNoise:

    def test_maxiter_path_clamps_to_zero(self, capsys):
        np.random.seed(0)
        # huge sigma relative to the errors keeps drawing negatives
        out = err_add_noise(np.full(50, 0.001), sigma=100.0, maxiter=2)
        assert (out >= 0).all()
        assert 'Max iterations reached' in capsys.readouterr().out
