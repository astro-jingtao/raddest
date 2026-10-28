

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from raddest.api.flowvi import FlowVI
from raddest.transfer import NegLogPlus, XShiftNegLogPlus
from raddest.x_xerr import (
    construct_x_xerr_w,
    gen_xobs_trans,
    gen_xobs_trans_log_rotation,
    gen_xobs_trans_neglogplus,
    gen_xobs_trans_neglogplus_xshift,
    gen_xobs_trans_xshift_neglogplus,
    nf_fit_trans_log_rotation,
    nf_fit_trans_neglogplus,
)

RNG = np.random.default_rng(0)


def make_data(n=300):
    x = RNG.uniform(1, 3, size=n)
    x_err = RNG.uniform(0.05, 0.15, size=n)
    x_obs = RNG.normal(x, x_err)
    return x_obs, x_err


class TestNfFitVariants:

    def test_neglogplus_roundtrip(self):
        x_obs, x_err = make_data()
        flow, _loss, params = nf_fit_trans_neglogplus(x_obs,
                                               x_err,
                                               optimizer=optax.adam(1e-2),
                                               n_pre=5,
                                               n_train=5,
                                               n_node=5,
                                               verbose=False,
                                               early_stop=False)
        _x_gen, _x_err_gen, x_obs_gen = gen_xobs_trans_neglogplus(
            flow, 100, 0.5, 3.5, 0.0, 0.5, params)
        assert np.asarray(x_obs_gen).shape == (100, )

    def test_log_rotation_roundtrip(self):
        x_obs, x_err = make_data()
        flow, _loss, params = nf_fit_trans_log_rotation(x_obs,
                                                 x_err,
                                                 theta=0.1,
                                                 optimizer=optax.adam(1e-2),
                                                 n_pre=5,
                                                 n_train=5,
                                                 n_node=5,
                                                 verbose=False,
                                                 early_stop=False)
        _x_gen, _x_err_gen, x_obs_gen = gen_xobs_trans_log_rotation(
            flow, 100, 0.5, 3.5, 0.0, 0.5, params)
        assert np.asarray(x_obs_gen).shape == (100, )


class TestGenXobsTransWrappers:

    def test_no_transfer(self):
        x_obs, x_err = make_data()
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False)
        _x_gen, _x_err_gen, x_obs_gen = gen_xobs_trans(flow_vi.flow,
                                                     50,
                                                     0.5,
                                                     3.5,
                                                     0.0,
                                                     0.5,
                                                     return_numpy=True)
        assert np.asarray(x_obs_gen).shape == (50, )

    def test_xshift_variants_with_zero_shift(self):
        x_obs, x_err = make_data()
        flow, _loss, params = nf_fit_trans_neglogplus(x_obs,
                                               x_err,
                                               optimizer=optax.adam(1e-2),
                                               n_pre=5,
                                               n_train=5,
                                               n_node=5,
                                               early_stop=False)
        # shift = 0 reduces both xshift variants to plain NegLogPlus
        p3 = [float(params[0]), float(params[1]), 0.0]
        out1 = gen_xobs_trans_neglogplus_xshift(flow, 50, 0.5, 3.5, 0.0, 0.5,
                                                p3)
        out2 = gen_xobs_trans_xshift_neglogplus(flow, 50, 0.5, 3.5, 0.0, 0.5,
                                                p3)
        for out in (out1, out2):
            assert np.asarray(out[2]).shape == (50, )


class TestFlowVIFitOptions:

    def test_skip_fitting_if_init_good(self):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False,
                    skip_fitting_if_init_good=True,
                    init_good_threshold=-1.0,
                    verbose=True)
        assert flow_vi.fitted
        assert flow_vi.loss == []

    def test_skip_fitting_check_runs_but_continues(self, capsys):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False,
                    skip_fitting_if_init_good=True,
                    init_good_threshold=0.99,
                    verbose=True)
        assert flow_vi.fitted
        assert 'KS test p-value' in capsys.readouterr().out

    def test_early_stop_stuck(self):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=50,
                    n_node=5,
                    check_per_step=1,
                    early_stop=True,
                    max_stuck=1,
                    stuck_frac=1e-9,
                    verbose=False)
        assert flow_vi.fitted

    def test_re_estimate_transfer_params(self):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5,
                         transfer=NegLogPlus(),
                         transfer_params=None)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False,
                    re_estimate_transer_params=True)
        assert len(flow_vi.transfer_params) == 2

    def test_re_estimate_without_estimator_raises(self):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5,
                         transfer=XShiftNegLogPlus(),
                         transfer_params=[1.0, 0.1, 0.5])
        with pytest.raises(ValueError, match='parameter estimator'):
            flow_vi.fit(x_obs,
                        x_err,
                        n_pre=0,
                        n_train=0,
                        n_node=3,
                        re_estimate_transer_params=True)

    def test_derive_uv_bounds(self):
        x_obs, x_err = make_data(200)
        flow_vi = FlowVI(x_min=0,
                         x_max=4,
                         x_err_min=0,
                         x_err_max=0.5)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False,
                    derive_uv_bounds=True)
        assert flow_vi.fitted


from pzflow import Flow
from pzflow.bijectors import Chain, RollingSplineCoupling

from raddest.x_xerr import construct_nf_fit_func, nf_fit_general


class TestFindUVBounds:

    def test_chain_without_bounds_raises(self):
        flow = Flow(["x_obs", "x_err"], bijector=Chain(RollingSplineCoupling(2)))
        with pytest.raises(ValueError, match='Cannot find bounds'):
            construct_nf_fit_func(optax.adam(1e-3), flow)

    def test_flow_without_bounds_raises_at_fit(self):

        x_obs, x_err = make_data(100)
        flow = Flow(["x_obs", "x_err"], bijector=Chain(RollingSplineCoupling(2)))
        with pytest.raises(ValueError, match='does not have bounds'):
            nf_fit_general(x_obs,
                           x_err,
                           optimizer=optax.adam(1e-2),
                           flow=flow,
                           n_pre=0,
                           n_train=1,
                           n_node=3,
                           early_stop=False)


class TestConstructXXerrWMethods:

    def test_inter_grid(self):
        x_ext, _x_err_ext, _w = construct_x_xerr_w(jnp.asarray([1.0, 2.0]),
                                                 jnp.asarray([0.1, 0.2]),
                                                 n_node=6,
                                                 int_method='inter_grid')
        assert x_ext.shape == (2, 6)
