

import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
import pytest

from raddest.reg.constructor import (
    construct_generator_cond,
    construct_ks_compare,
    construct_likelihood,
    construct_linear_step,
    construct_linear_student_t_step,
    construct_log_likelihood,
    construct_lognorm_likelihood,
    construct_saturated_linear_step,
    construct_trainer,
    flat_piecewise_params,
    parse_piecewise_params,
    piecewise_linear_step_factory,
)
from raddest.reg.func_jax import saturated_linear_smooth_jax as func_jax_saturated


def make_1d_data(n=50, n_ext=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 3, size=n)
    y = 2 * x + 1 + rng.normal(scale=0.1, size=n)
    # the likelihood consumes quadrature-extended 2D arrays: (N, n_ext)
    x_ext = np.repeat(x[:, None], n_ext, axis=1)
    y_ext = y[:, None] + rng.normal(scale=0.05, size=(n, n_ext))
    w = np.full((n, n_ext), 1.0 / n_ext)
    return x_ext, y_ext, w


def linearnorm(x, y, params):
    return norm.pdf(y, loc=params[0] * x + params[1], scale=params[2])


class TestLikelihood:

    def test_likelihood_and_grads(self):
        x, y, w = make_1d_data()
        likelihood = construct_likelihood(linearnorm)
        log_lik, d_log_lik, dd_log_lik = construct_log_likelihood(likelihood)

        params = jnp.asarray([2.0, 1.0, 0.1])
        v = log_lik(params, x, y, w)
        assert jnp.isfinite(v)

        g = d_log_lik(params, x, y, w)[0]
        assert g.shape == (3, )
        assert jnp.all(jnp.isfinite(g))

        h = dd_log_lik(params, x, y, w)
        assert h[0][0].shape == (3, 3)
        assert jnp.all(jnp.isfinite(h[0][0]))

    def test_lognorm_likelihood_factory(self):
        def func(x, params):
            return func_jax_saturated(x, *params)

        likelihood = construct_lognorm_likelihood(func)
        x, y, w = make_1d_data()
        v = likelihood(jnp.asarray([1.0, 0.0, -1.0, 1.0, -2.0, 2.0, 0.1]), x,
                       y, w)
        assert jnp.all(jnp.isfinite(v))


class TestTrainer:

    def test_converges_and_stops(self, capsys):
        x, y, w = make_1d_data()
        likelihood = construct_likelihood(linearnorm)
        log_lik, d_log_lik, _ = construct_log_likelihood(likelihood)
        step = construct_linear_step(d_log_lik)
        trainer = construct_trainer(log_lik, step)

        params = trainer((jnp.asarray(1.9), jnp.asarray(0.9), jnp.asarray(0.2)),
                         x,
                         y,
                         w,
                         lr=1e-2,
                         n_step=5000,
                         verbose=True)
        assert float(params[0]) == pytest.approx(2.0, abs=0.1)
        assert '1000 ' in capsys.readouterr().out

    def test_zero_gradient_converges(self, capsys):
        x, y, w = make_1d_data()

        def d_log_lik_zero(_params, _x, _y, _w):
            return (jnp.asarray([0.0, 0.0, 0.0]), )

        likelihood = construct_likelihood(linearnorm)
        log_lik, _, _ = construct_log_likelihood(likelihood)
        step = construct_linear_step(d_log_lik_zero)
        trainer = construct_trainer(log_lik, step)

        params = trainer((jnp.asarray(1.9), jnp.asarray(0.9), jnp.asarray(0.2)),
                         x,
                         y,
                         w,
                         lr=1e-2,
                         n_step=100000,
                         verbose=True)
        assert float(params[0]) == pytest.approx(1.9, rel=1e-5)
        assert 'converged' in capsys.readouterr().out

    def test_max_iterations(self, capsys):
        x, y, w = make_1d_data()
        likelihood = construct_likelihood(linearnorm)
        log_lik, d_log_lik, _ = construct_log_likelihood(likelihood)
        step = construct_linear_step(d_log_lik)
        trainer = construct_trainer(log_lik, step)

        params = trainer((jnp.asarray(1.9), jnp.asarray(0.9), jnp.asarray(0.2)),
                         x,
                         y,
                         w,
                         lr=1e-9,
                         n_step=3000)
        assert len(params) == 3
        out = capsys.readouterr().out
        assert '0 1.9' in out or out == ''


class TestSteps:

    def test_linear_step_clips(self):
        def d_log_lik(_params, _x, _y, _w):
            return (jnp.asarray([1e6, -1e6, 1e6]), )

        step = construct_linear_step(d_log_lik)
        params = step((jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0)),
                      None, None, None, lr=1.0)
        k, b, sig = params
        assert float(k) == 1e2  # clipped up
        assert float(b) == -1e2  # clipped down
        assert float(sig) == 1e2

    def test_linear_student_t_step_clips(self):
        def d_log_lik(_params, _x, _y, _w):
            return (jnp.asarray([1e6, -1e6, 1e6, -1e6]), )

        step = construct_linear_student_t_step(d_log_lik)
        _k, _b, df, sig = step(
            (jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0),
             jnp.asarray(1.0)), None, None, None, lr=1.0)
        assert float(df) == pytest.approx(1e2)
        assert float(sig) == pytest.approx(1e-2, rel=1e-5)

    def test_saturated_step_handles_nan_grads(self):
        def d_log_lik(_params, _x, _y, _w):
            return (jnp.asarray([np.nan] * 7), )

        step = construct_saturated_linear_step(d_log_lik)
        params_in = tuple(jnp.asarray(v) for v in (1.0, 0.0, -1.0, 1.0, -2.0,
                                                  2.0, 0.1))
        params_out = step(params_in, None, None, None, lr=1e-3)
        for p_in, p_out in zip(params_in, params_out):
            assert float(p_out) == pytest.approx(float(p_in), rel=1e-5)

    def test_piecewise_step_with_linear_constraint(self):
        n_piece = 2

        def d_log_lik(_params, _x, _y, _w):
            return (jnp.asarray([0.0] * (2 + n_piece + n_piece - 1 + 1)), )

        # constrain k1 + k2 = 1 (A_eq @ params[:-1] block for slopes)
        A_eq = jnp.asarray([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        b_eq = jnp.asarray([1.0])
        constructor = piecewise_linear_step_factory(n_piece, A_eq=A_eq, b_eq=b_eq)
        step = constructor(d_log_lik)

        params = jnp.asarray([0.0, 1.0, 0.4, 0.6, 0.5, 0.1])
        params_new = step(params, None, None, None, lr=0.0)
        np.testing.assert_allclose(np.asarray(params_new),
                                   np.asarray(params),
                                   rtol=1e-5)

    def test_piecewise_step_b_eq_requires_A_eq(self):
        with pytest.raises(ValueError, match='b_eq should be None'):
            piecewise_linear_step_factory(2, A_eq=None, b_eq=[1.0])


class TestPiecewiseParse:

    def test_parse_and_flat_round_trip(self):
        params = (0.0, 1.0, 2.0, 3.0, 0.5, 0.7)
        b0, x0, k_lst, dx_lst = parse_piecewise_params(2, params)
        assert (b0, x0) == (0.0, 1.0)
        assert list(k_lst) == [2.0, 3.0]
        assert list(dx_lst) == [0.5, 0.7]
        assert flat_piecewise_params(b0, x0, k_lst, dx_lst) == params


class TestGeneratorCond:

    def test_nan_clip_and_match(self):
        func_x2y = lambda x, params: params[0] * x + params[1]  # noqa: E731
        generator_cond = construct_generator_cond(func_x2y)

        x = np.array([1.0, np.nan, 3.0])
        y_kdt = np.linspace(0, 10, 50)
        y_err_kdt = np.linspace(0.1, 0.5, 50)

        y_pred, _y_err_pred, y_obs_pred = generator_cond(x, y_kdt, y_err_kdt,
                                                        (2.0, 1.0),
                                                        y_min=0.0,
                                                        y_max=10.0)
        # nan input becomes the mean of y_kdt
        assert np.isfinite(y_pred).all()
        assert np.isfinite(y_obs_pred).all()
        assert y_pred.max() <= 10.0

    def test_ks_compare(self):
        func_x2y = lambda x, params: params[0] * x + params[1]  # noqa: E731
        generator_cond = construct_generator_cond(func_x2y)
        rng = np.random.default_rng(1)
        x_gen = rng.uniform(1, 3, size=100)
        y_kdt = np.linspace(2, 10, 200)
        y_err_kdt = np.full(200, 0.1)
        x_obs = rng.uniform(1, 3, size=100)
        y_obs = 2 * x_obs + 1 + rng.normal(scale=0.2, size=100)

        compare = construct_ks_compare(generator_cond, x_gen, x_gen, y_kdt,
                                       y_err_kdt, x_obs, y_obs)
        np.random.seed(0)
        D_good, _p_good = compare(2.0, 1.0)
        D_bad, _p_bad = compare(5.0, 1.0)
        assert D_good < D_bad


class TestPiecewiseStepBEqDefault:

    def test_a_eq_without_b_eq_zeros(self):
        n_piece = 2

        def d_log_lik(_params, _x, _y, _w):
            return (jnp.asarray([0.0] * 6), )

        # A_eq without b_eq defaults b_eq to zeros
        A_eq = jnp.asarray([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        constructor = piecewise_linear_step_factory(n_piece, A_eq=A_eq)
        step = constructor(d_log_lik)
        params = jnp.asarray([0.0, 1.0, 0.4, 0.6, 0.5, 0.1])
        params_new = step(params, None, None, None, lr=0.0)
        # b_eq defaults to zeros, so the projection enforces k1 + k2 = 0
        np.testing.assert_allclose(float(params_new[2] + params_new[3]),
                                   0.0,
                                   rtol=1e-5,
                                   atol=1e-6)
        assert float(params_new[0]) == pytest.approx(0.0, abs=1e-6)
