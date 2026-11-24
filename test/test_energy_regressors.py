import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.fitter import (
    LinearKS,
    LogLinearEnergyGrad,
    LogLinearEnergyPT,
    LogLinearKS,
    RegressorEnergyGrad,
    RegressorEnergyPT,
    RegressorKS,
)
from raddest.api.flowvi import FlowVI
from raddest.transfer import NegLogPlus

TRUE_PARAMS = {"k": 2.0, "b": 0.5, "sig": 0.1}


def make_obs_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 3, size=n)
    x_err = rng.uniform(0.05, 0.15, size=n)
    x_obs = rng.normal(x, x_err)

    y = 10 ** (
        TRUE_PARAMS["k"] * np.log10(x)
        + TRUE_PARAMS["b"]
        + rng.normal(scale=TRUE_PARAMS["sig"], size=n)
    )
    y_err = 0.05 * y
    y_obs = rng.normal(y, y_err)

    return x_obs, x_err, y_obs, y_err


def fit_tiny_flow(x_obs, x_err):
    flow_vi = FlowVI(
        x_min=float(x_obs.min()) * 0.5,
        x_max=float(x_obs.max()) * 1.1,
        x_err_min=0,
        x_err_max=float(x_err.max()) * 1.2,
        transfer=NegLogPlus(),
        transfer_params=[float(np.median(x_obs)), float(np.median(x_err))],
    )
    flow_vi.fit(
        x_obs,
        x_err,
        n_pre=20,
        n_train=50,
        n_node=7,
        verbose=False,
        x_min=float(x_obs.min()) * 0.5,
        x_max=float(x_obs.max()) * 1.1,
        x_err_min=0,
        x_err_max=float(x_err.max()) * 1.2,
        early_stop=False,
    )
    return flow_vi


@pytest.fixture(scope="module")
def fitted_regressors():
    np.random.seed(0)
    x_obs, x_err, y_obs, y_err = make_obs_data()

    xflow = fit_tiny_flow(x_obs, x_err)
    yflow = fit_tiny_flow(y_obs, y_err)

    regressor_pt = LogLinearEnergyPT(xflow, yflow, permutations=50)
    regressor_pt_jax = LogLinearEnergyPT(xflow, yflow, permutations=50, backend="jax")
    regressor_grad = LogLinearEnergyGrad(xflow, yflow)
    return regressor_pt, regressor_pt_jax, regressor_grad, x_obs, x_err, y_obs, y_err


@pytest.mark.slow
class TestLogLinearEnergyPT:
    def test_fit_recovers_params(self, fitted_regressors):
        regressor_pt, _, _, x_obs, _, y_obs, _ = fitted_regressors

        params_list = [
            np.linspace(1.0, 3.0, 4),
            np.linspace(-0.5, 1.5, 4),
            np.linspace(0.05, 0.2, 3),
        ]

        params_mean, params_std, res_arr = regressor_pt.fit(
            x_obs, y_obs, params_list=params_list, N=300, num_process=2, max_i=1, refine=False
        )

        assert res_arr.shape[1] == 5
        assert np.all(params_std >= 0)
        assert abs(params_mean[0] - TRUE_PARAMS["k"]) < 0.6
        assert abs(params_mean[1] - TRUE_PARAMS["b"]) < 0.6
        assert abs(params_mean[2] - TRUE_PARAMS["sig"]) < 0.15

    def test_energy_test_prefers_true_params(self, fitted_regressors):
        regressor_pt, _, _, x_obs, _, y_obs, _ = fitted_regressors
        np.random.seed(1)

        _, p_good = regressor_pt.energy_test(
            (TRUE_PARAMS["k"], TRUE_PARAMS["b"], TRUE_PARAMS["sig"]),
            x_obs,
            y_obs,
            N=300,
            M=300,
            permutations=100,
        )
        _, p_bad = regressor_pt.energy_test(
            (4.0, 2.0, 0.4), x_obs, y_obs, N=300, M=300, permutations=100
        )

        assert p_good > p_bad


@pytest.mark.slow
class TestLogLinearEnergyPTJaxBackend:
    def test_fit_recovers_params(self, fitted_regressors):
        _, regressor_pt_jax, _, x_obs, _, y_obs, _ = fitted_regressors

        params_list = [
            np.linspace(1.0, 3.0, 4),
            np.linspace(-0.5, 1.5, 4),
            np.linspace(0.05, 0.2, 3),
        ]

        params_mean, params_std, res_arr = regressor_pt_jax.fit(
            x_obs, y_obs, params_list=params_list, N=300, num_process=1, max_i=1, refine=False
        )

        assert res_arr.shape[1] == 5
        assert np.all(params_std >= 0)
        assert abs(params_mean[0] - TRUE_PARAMS["k"]) < 0.6
        assert abs(params_mean[1] - TRUE_PARAMS["b"]) < 0.6
        assert abs(params_mean[2] - TRUE_PARAMS["sig"]) < 0.15

    def test_invalid_backend_and_chunk_combination(self, fitted_regressors):
        regressor_pt, *_ = fitted_regressors
        xflow, yflow = regressor_pt.xflow, regressor_pt.yflow
        with pytest.raises(ValueError, match="must be 'numpy' or 'jax'"):
            LogLinearEnergyPT(xflow, yflow, backend="gpu")
        with pytest.raises(ValueError, match="numpy backend"):
            LogLinearEnergyPT(xflow, yflow, chunk_size=50, chunk_iter=5, backend="jax")


class TestEnergyPTHierarchy:
    """Model and fitting strategy are separate inheritance axes."""

    def test_loglinear_energy_pt_is_not_a_loglinear_ks(self):
        # no flows needed: the class hierarchy is inspectable without fitting
        regressor_pt = LogLinearEnergyPT(None, None)
        assert isinstance(regressor_pt, RegressorEnergyPT)
        assert isinstance(regressor_pt, RegressorKS)
        assert not isinstance(regressor_pt, LogLinearKS)

    def test_sigma_grid_defaults_shared_with_ks_models(self):
        pt = LogLinearEnergyPT(None, None)
        ks = LogLinearKS(None, None)
        linear = LinearKS(None, None)
        for reg in (pt, ks, linear):
            assert reg._set_gen_new_grid_kwargs({})["range_par"][2] == [0, np.inf]
            assert reg._set_gen_new_grid_refine_kwargs({})["range_par"][2] == [0, np.inf]

    def test_jax_backend_requires_func_x2y_jax(self):
        class NoJaxPT(RegressorEnergyPT):
            def __init__(self) -> None:
                RegressorEnergyPT.__init__(
                    self,
                    None,
                    None,
                    lambda x, p: x,
                    parser=lambda res, iargs: res,
                    print_mean_std=lambda m, s: None,
                    backend="jax",
                )

        with pytest.raises(ValueError, match="func_x2y_jax"):
            NoJaxPT().fit(None, None, params_list=[[1.0]], N=10)


@pytest.mark.slow
class TestLogLinearEnergyGrad:
    def test_fit_decreases_loss_and_moves_toward_truth(self, fitted_regressors):
        _, _, regressor_grad, x_obs, _, y_obs, _ = fitted_regressors

        params_0 = jnp.asarray([1.5, 0.0, 0.2], dtype=jnp.float32)
        params_hat, loss_history = regressor_grad.fit(
            x_obs, y_obs, params_0, N=300, lr=5e-2, n_step=2000, check_per_step=100, seed=0
        )

        assert loss_history[-1] < loss_history[0]

        d0 = np.linalg.norm(
            np.asarray(params_0) - [TRUE_PARAMS["k"], TRUE_PARAMS["b"], TRUE_PARAMS["sig"]]
        )
        d1 = np.linalg.norm(params_hat - [TRUE_PARAMS["k"], TRUE_PARAMS["b"], TRUE_PARAMS["sig"]])
        assert d1 < d0


class TestRegressorEnergyGradValidation:
    def test_missing_jax_func_raises(self):
        with pytest.raises(TypeError):
            RegressorEnergyGrad(None, None, func_x2y=lambda x, p: x)


@pytest.mark.slow
class TestBackendFitParity:
    """The numpy and jax grid backends produce the same fit on shared inputs.

    Both fits run on the same flows and observed data; the backends draw
    their synthetic-sample noise and permutation streams independently (the
    jax backend shares one fixed noise realization across the whole grid),
    so parity holds to the fit-to-fit Monte-Carlo noise, not bitwise.
    Calibrated on this fixture's data: stat-column correlation 0.999
    (numpy-vs-numpy rerun of the same fit: 0.998), mean-statistic relative
    difference 1.2% (numpy rerun: 0.1%), and params_mean/params_std
    agreement within [0.012, 0.009, 0.004] (numpy rerun: [0.005, 0.002,
    0.001]).
    """

    def test_jax_backend_fit_matches_numpy_backend_fit(self, fitted_regressors):
        regressor_pt, regressor_pt_jax, _, x_obs, _, y_obs, _ = fitted_regressors
        params_list = [
            np.linspace(1.6, 2.4, 2),
            np.linspace(0.1, 0.9, 2),
            np.linspace(0.05, 0.15, 2),
        ]

        np.random.seed(7)
        mean_np, std_np, res_np = regressor_pt.fit(
            x_obs, y_obs, params_list=params_list, N=300, num_process=2, max_i=1, refine=False
        )
        mean_jax, std_jax, res_jax = regressor_pt_jax.fit(
            x_obs, y_obs, params_list=params_list, N=300, num_process=1, max_i=1, refine=False
        )

        # same grid points in the same order
        assert res_np.shape == res_jax.shape
        np.testing.assert_allclose(res_np[:, :3], res_jax[:, :3])

        stat_np, stat_jax = res_np[:, 3], res_jax[:, 3]
        assert np.corrcoef(stat_np, stat_jax)[0, 1] > 0.995
        assert abs(stat_jax.mean() - stat_np.mean()) / stat_np.mean() < 0.05
        assert np.all((res_np[:, 4] > 0) & (res_np[:, 4] <= 1))
        assert np.all((res_jax[:, 4] > 0) & (res_jax[:, 4] <= 1))

        assert np.all(np.abs(mean_jax - mean_np) < [0.05, 0.05, 0.02])
        assert np.all(np.abs(std_jax - std_np) < [0.05, 0.05, 0.02])
