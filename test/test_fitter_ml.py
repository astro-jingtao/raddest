

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.fitter import (
    LinearKS,
    LinearML,
    LogLinearEnergyGrad,
    LogLinearKS,
    LogLinearML,
    LogLinearStudentTML,
    PiecewiseLogLinearML,
    PiecewiseLogLinearStudentTML,
    RegressorBase,
    RegressorKS,
    SaturatedLogLinearML,
)

TRUE_PARAMS = {'k': 2.0, 'b': 0.5, 'sig': 0.1}


@pytest.fixture(scope='module')
def ml_regressor(tiny_flows):
    xflow, yflow = tiny_flows
    return LogLinearML(xflow, yflow)


@pytest.mark.slow
class TestLogLinearMLFit:

    def test_fit_recovers_params(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data

        params_ml, params_std, cov = ml_regressor.fit(x_obs,
                                                      x_err,
                                                      y_obs,
                                                      y_err,
                                                      params_0=jnp.asarray(
                                                          [1.5, 0.0, 0.2]),
                                                      n_step=3000)

        assert params_ml.shape == (3, )
        assert params_std.shape == (3, )
        assert cov.shape == (3, 3)
        assert abs(float(params_ml[0]) - TRUE_PARAMS['k']) < 0.5
        assert np.all(np.asarray(params_std) >= 0)

    def test_log_likelihood_and_posterior(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data

        ll = ml_regressor.log_likelihood(x_obs, x_err, y_obs, y_err,
                                         jnp.asarray([2.0, 0.5, 0.1]))
        assert ll.shape == (1, )
        assert jnp.all(jnp.isfinite(ll))

        lp = ml_regressor.log_posterior(x_obs, x_err, y_obs, y_err,
                                        jnp.asarray([2.0, 0.5, 0.1]))
        assert lp.shape == (1, )

        # invalid scatter is excluded by the prior
        lp_bad = ml_regressor.log_posterior(x_obs, x_err, y_obs, y_err,
                                            jnp.asarray([2.0, 0.5, -0.1]))
        assert float(lp_bad[0]) == -jnp.inf

    def test_get_cov(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data
        cov = ml_regressor.get_cov(x_obs,
                                   x_err,
                                   y_obs,
                                   y_err,
                                   params=jnp.asarray([2.0, 0.5, 0.1]))
        assert cov.shape == (3, 3)

    def test_get_ext_data_int_methods(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data
        for int_method in ['ghq', 'qmc', 'inter_grid']:
            x_ext, y_ext, w = ml_regressor.get_ext_data(x_obs,
                                                        x_err,
                                                        y_obs,
                                                        y_err,
                                                        n=5,
                                                        int_method=int_method)
            assert x_ext.shape == y_ext.shape == w.shape

    def test_sample_and_tests(self, ml_regressor, obs_data):
        x_obs, _, y_obs, _ = obs_data
        np.random.seed(2)

        out = ml_regressor.sample((TRUE_PARAMS['k'], TRUE_PARAMS['b'],
                                   TRUE_PARAMS['sig']),
                                  N=300)
        assert len(out) == 6
        assert all(np.asarray(o).shape == (300, ) for o in out)

        _D, p = ml_regressor.ks_test((TRUE_PARAMS['k'], TRUE_PARAMS['b'],
                                     TRUE_PARAMS['sig']),
                                    x_obs,
                                    y_obs,
                                    N=300,
                                    M=100,
                                    K=1)
        assert 0 <= p <= 1

        D_lst, p_lst = ml_regressor.ks_test(
            (TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']),
            x_obs,
            y_obs,
            N=300,
            M=100,
            K=2)
        assert len(D_lst) == len(p_lst) == 2

        # default M (= number of observed points)
        _D, _p = ml_regressor.ks_test(
            (TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']),
            x_obs,
            y_obs,
            N=200)
        _E, _p = ml_regressor.energy_test(
            (TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']),
            x_obs,
            y_obs,
            N=200,
            permutations=50)

        E, p = ml_regressor.energy_test((TRUE_PARAMS['k'], TRUE_PARAMS['b'],
                                         TRUE_PARAMS['sig']),
                                        x_obs,
                                        y_obs,
                                        N=200,
                                        M=100,
                                        K=1,
                                        permutations=50)
        assert E >= 0
        assert 0 < p <= 1

        E_lst, p_lst = ml_regressor.energy_test(
            (TRUE_PARAMS['k'], TRUE_PARAMS['b'], TRUE_PARAMS['sig']),
            x_obs,
            y_obs,
            N=200,
            M=100,
            K=2,
            permutations=50)
        assert len(E_lst) == len(p_lst) == 2

    def test_ks_test_m_too_large_warns(self, ml_regressor, obs_data, capsys):
        x_obs, _, y_obs, _ = obs_data
        np.random.seed(3)
        _D, _p = ml_regressor.ks_test((2.0, 0.5, 0.1),
                                      x_obs,
                                      y_obs,
                                      N=200,
                                      M=10**6)
        assert 'Warning' in capsys.readouterr().out

    def test_get_posterior_qmc(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data
        params_grid, weight = ml_regressor.get_posterior_qmc(
            x_obs,
            x_err,
            y_obs,
            y_err, [1.5, 0.0, 0.05], [2.5, 1.0, 0.2],
            m=4)
        assert params_grid.shape == (16, 3)
        assert weight.shape == (16, )
        assert bool(jnp.all(weight >= 0))

    def test_get_posterior_mcmc(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data
        nwalkers = 6
        initial_state = jnp.asarray(
            np.random.default_rng(0).normal([2.0, 0.5, 0.1], 0.05,
                                            size=(nwalkers, 3)))
        sampler = ml_regressor.get_posterior_mcmc(x_obs,
                                                  x_err,
                                                  y_obs,
                                                  y_err,
                                                  ndim=3,
                                                  initial_state=initial_state,
                                                  nsteps=20)
        assert sampler.chain.shape == (nwalkers, 20, 3)


@pytest.mark.slow
class TestOtherMLRegressors:

    def test_linear_ml_moves_and_stays_finite(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data
        regressor = LinearML(xflow, yflow)
        params_ml, _std, _cov = regressor.fit(x_obs,
                                              x_err,
                                              y_obs,
                                              y_err,
                                              params_0=jnp.asarray(
                                                  [1.5, 0.5, 0.2]),
                                              n_step=3000)
        assert jnp.all(jnp.isfinite(params_ml))
        assert float(jnp.max(jnp.abs(params_ml - jnp.asarray([1.5, 0.5, 0.2])))) > 0

    def test_saturated_log_linear_ml(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data
        regressor = SaturatedLogLinearML(xflow, yflow, width=0.1)
        params_ml, _std, _cov = regressor.fit(x_obs,
                                              x_err,
                                              y_obs,
                                              y_err,
                                              params_0=jnp.asarray(
                                                  [2.0, 0.0, -0.5, 0.5, -2.0,
                                                   2.0, 0.2]),
                                              n_step=1500)
        assert jnp.all(jnp.isfinite(params_ml))

    def test_piecewise_log_linear_ml(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data
        regressor = PiecewiseLogLinearML(xflow, yflow, n_piece=2, width=0.05)
        params_ml, _std, _cov = regressor.fit(x_obs,
                                              x_err,
                                              y_obs,
                                              y_err,
                                              params_0=jnp.asarray(
                                                  [0.5, 0.0, 2.0, 2.0, 0.5,
                                                   0.2]),
                                              n_step=1500)
        assert jnp.all(jnp.isfinite(params_ml))
        assert float(params_ml[-1]) > 0  # scatter stays positive

        k_lst, _b_lst, _x_lst, _sig = regressor.to_each_piece_params(
            np.asarray(params_ml))
        assert len(k_lst) == 2

    def test_piecewise_log_linear_ml_with_constraint(self, tiny_flows,
                                                     obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data
        A_eq = jnp.asarray([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        b_eq = jnp.asarray([4.0])
        regressor = PiecewiseLogLinearML(xflow,
                                         yflow,
                                         n_piece=2,
                                         width=0.05,
                                         A_eq=A_eq,
                                         b_eq=b_eq)
        params_ml, _std, _cov = regressor.fit(x_obs,
                                              x_err,
                                              y_obs,
                                              y_err,
                                              params_0=jnp.asarray(
                                                  [0.5, 0.0, 2.0, 2.0, 0.5,
                                                   0.2]),
                                              n_step=200)
        np.testing.assert_allclose(
            np.asarray(params_ml)[2] + np.asarray(params_ml)[3], 4.0, rtol=1e-4)

    def test_loglinear_student_t_ml(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data
        regressor = LogLinearStudentTML(xflow, yflow)
        params_ml, _std, _cov = regressor.fit(x_obs,
                                              x_err,
                                              y_obs,
                                              y_err,
                                              params_0=jnp.asarray(
                                                  [1.5, 0.0, 10.0, 0.2]),
                                              n_step=1500)
        assert jnp.all(jnp.isfinite(params_ml))

    def test_piecewise_student_t_ml_is_broken(self, tiny_flows):
        # The class references a step factory that does not exist (see
        # TODO.md); instantiating it raises NameError as shipped.
        xflow, yflow = tiny_flows
        with pytest.raises(NameError):
            PiecewiseLogLinearStudentTML(xflow, yflow, n_piece=2)


@pytest.mark.slow
class TestLogLinearKSFit:

    def test_fit_with_refine(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, _, y_obs, _ = obs_data
        regressor = LogLinearKS(xflow, yflow)

        params_mean, _params_std, res_arr = regressor.fit(
            x_obs,
            y_obs,
            params_list=[
                np.linspace(1.5, 2.5, 3),
                np.linspace(0.0, 1.0, 3),
                np.linspace(0.05, 0.15, 2),
            ],
            N=200,
            num_process=2,
            max_i=1,
            verbose=True,
            gen_new_grid_kwargs={'n': 3},
            gen_new_grid_refine_kwargs={'n': 3})

        assert res_arr.shape[1] == 5
        assert abs(params_mean[0] - TRUE_PARAMS['k']) < 0.8

    def test_unknown_gen_new_grid_rejected(self, tiny_flows):
        xflow, yflow = tiny_flows
        with pytest.raises(ValueError, match='Unknown gen_new_grid'):
            RegressorKS(xflow, yflow, lambda x, p: x, parser=lambda r,
                        i: r, print_mean_std=lambda a, b: None,
                        gen_new_grid='bogus')


class TestRegressorBaseBranches:

    def test_no_generator_cond_raises(self, tiny_flows):
        xflow, yflow = tiny_flows
        base = RegressorBase(xflow, yflow)
        with pytest.raises(ValueError, match='No conditional generator'):
            base.sample((1.0, 0.0, 0.1), N=10)

    def test_energy_test_m_too_large_warns(self, ml_regressor, obs_data,
                                           capsys):
        x_obs, _, y_obs, _ = obs_data
        np.random.seed(4)
        _E, _p = ml_regressor.energy_test((2.0, 0.5, 0.1),
                                          x_obs,
                                          y_obs,
                                          N=100,
                                          M=10**6,
                                          permutations=10)
        assert 'Warning' in capsys.readouterr().out

    def test_qmc_bounds_length_mismatch(self, ml_regressor, obs_data):
        x_obs, x_err, y_obs, y_err = obs_data
        with pytest.raises(ValueError, match='same length'):
            ml_regressor.get_posterior_qmc(x_obs, x_err, y_obs, y_err,
                                           [1.5, 2.5], [0.0, 1.0, 2.0])


class TestRegressorKSInit:

    def test_func_none_raises(self, tiny_flows):
        xflow, yflow = tiny_flows
        with pytest.raises(ValueError, match='No function'):
            RegressorKS(xflow, yflow, None, parser=lambda r, i: r,
                        print_mean_std=lambda a, b: None)

    def test_quantile_gen_new_grid(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        regressor = RegressorKS(xflow,
                                yflow,
                                lambda x, params: params[0] * x + params[1],
                                parser=lambda res, iargs: np.zeros(
                                    (len(res), 5)),
                                print_mean_std=lambda a, b: None,
                                gen_new_grid='quantile')
        assert regressor.gen_new_grid_name == 'quantile'

        _x_obs, _, _y_obs, _ = obs_data
        np.random.seed(0)
        res = regressor.gen_new_grid(
            np.column_stack([np.linspace(1, 3, 9), np.linspace(0, 1, 9),
                             np.linspace(0.05, 0.2, 9),
                             np.linspace(0, 1, 9), np.linspace(0, 1, 9)]),
            n_par=3,
            w_index=4)
        assert len(res) == 3

    def test_custom_gen_new_grid(self, tiny_flows):
        def custom_grid(res_arr, *, n_par, w_index, **kwargs):
            return [np.linspace(0, 1, 3) for _ in range(n_par)]

        xflow, yflow = tiny_flows
        regressor = RegressorKS(xflow,
                                yflow,
                                lambda x, params: params[0] * x + params[1],
                                parser=lambda res, iargs: np.zeros(
                                    (len(res), 5)),
                                print_mean_std=lambda a, b: None,
                                gen_new_grid=custom_grid)
        assert regressor.gen_new_grid_name == 'custom'


class TestScatterAndPriorBodies:

    def test_sample_calls_scatters(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        _x_obs, _, _y_obs, _ = obs_data
        np.random.seed(5)
        out = SaturatedLogLinearML(xflow, yflow, width=0.1).sample(
            (2.0, 0.0, -0.5, 0.5, -2.0, 2.0, 0.1), N=50)
        assert np.asarray(out[1]).shape == (50, )

        np.random.seed(6)
        out = PiecewiseLogLinearML(xflow, yflow, n_piece=2).sample(
            (0.5, 0.0, 2.0, 2.0, 0.5, 0.1), N=50)
        assert np.asarray(out[1]).shape == (50, )

        np.random.seed(7)
        out = LogLinearStudentTML(xflow, yflow).sample((2.0, 0.5, 10.0, 0.1),
                                                       N=50)
        assert np.asarray(out[1]).shape == (50, )

        np.random.seed(8)
        out = LinearML(xflow, yflow).sample((2.0, 1.0, 0.1), N=50)
        assert np.asarray(out[1]).shape == (50, )

    def test_log_priors_evaluate(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        x_obs, x_err, y_obs, y_err = obs_data

        lp = SaturatedLogLinearML(xflow, yflow,
                                  width=0.1).log_posterior(x_obs, x_err, y_obs,
                                                           y_err,
                                                           jnp.asarray([
                                                               2.0, 0.0,
                                                               -0.5, 0.5,
                                                               -2.0, 2.0, 0.1
                                                           ]))
        assert jnp.all(jnp.isfinite(lp))

        lp = PiecewiseLogLinearML(xflow, yflow,
                                  n_piece=2).log_posterior(x_obs, x_err, y_obs,
                                                           y_err,
                                                           jnp.asarray([
                                                               0.5, 0.0,
                                                               2.0, 2.0,
                                                               0.5, 0.1
                                                           ]))
        assert jnp.all(jnp.isfinite(lp))

        lp = LogLinearStudentTML(xflow, yflow).log_posterior(
            x_obs, x_err, y_obs, y_err, jnp.asarray([2.0, 0.5, 10.0, 0.1]))
        assert jnp.all(jnp.isfinite(lp))

        lp = LinearML(xflow, yflow).log_posterior(x_obs, x_err, y_obs, y_err,
                                                  jnp.asarray([2.0, 1.0,
                                                               0.1]))
        assert jnp.all(jnp.isfinite(lp))

        # SaturatedLogLinearML prior flags a bad configuration (x0 > x1)
        lp_bad = SaturatedLogLinearML(xflow, yflow, width=0.1).log_posterior(
            x_obs, x_err, y_obs, y_err,
            jnp.asarray([2.0, 0.0, 0.5, -0.5, -2.0, 2.0, 0.1]))
        assert float(lp_bad[0]) == -jnp.inf


@pytest.mark.slow
class TestLinearKS:

    def test_init_and_ranges(self, tiny_flows, obs_data):
        xflow, yflow = tiny_flows
        regressor = LinearKS(xflow, yflow)
        kwargs = regressor._set_gen_new_grid_kwargs(None)
        assert len(kwargs['range_par']) == 3
        refine_kwargs = regressor._set_gen_new_grid_refine_kwargs(None)
        assert len(refine_kwargs['range_par']) == 3
        assert refine_kwargs['n'] == 20


@pytest.mark.slow
class TestEnergyGradVerbose:

    def test_verbose_and_converged(self, tiny_flows, obs_data, capsys):
        xflow, yflow = tiny_flows
        x_obs, _, y_obs, _ = obs_data
        regressor = LogLinearEnergyGrad(xflow, yflow)

        # lr = 0 keeps params fixed, so the convergence check fires
        params_hat, loss_history = regressor.fit(
            x_obs,
            y_obs,
            jnp.asarray([1.5, 0.0, 0.2]),
            N=100,
            lr=0.0,
            n_step=1000,
            check_per_step=100,
            verbose=True)
        assert 'converged' in capsys.readouterr().out
        assert params_hat.shape == (3, )
        assert len(loss_history) >= 2
