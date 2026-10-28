

import jax.numpy as jnp
import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.reg.map import get_w_add_lognorm, get_xyw_trans, kde_pdf_batch
from raddest.tools import transform_data
from raddest.transfer import NegLogPlus

RANGES = ([0.5, 3.5], [0.0, 0.4], [1.0, 100.0], [0.0, 10.0])


def transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err):
    return {
        'x_transfer': xflow.transfer,
        'x_transfer_params': xflow.transfer_params,
        'y_transfer': yflow.transfer,
        'y_transfer_params': yflow.transfer_params,
    }


class FakeKernel:

    def __call__(self, x):
        return jnp.exp(-jnp.abs(x))


class TestGetXYWTrans:

    def test_int_methods_all_shapes(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        tkw = transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err)
        for method in ['ghq', 'qmc', 'inter_grid', 'grid', 'qgrid']:
            x_ext, y_ext, w = get_xyw_trans(x_obs,
                                            y_obs,
                                            x_err,
                                            y_err,
                                            xflow.flow,
                                            yflow.flow,
                                            *RANGES,
                                            int_method=method,
                                            n=5,
                                            **tkw)
            assert x_ext.shape == y_ext.shape == w.shape
            assert float(jnp.nan_to_num(w).sum()) > 0

    def test_int_method_auto_selects(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows

        # large errors -> qgrid branch
        big_err = np.full_like(x_err, 5.0)
        tkw = transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err)
        x_ext, _y_ext, w = get_xyw_trans(x_obs,
                                        y_obs,
                                        big_err,
                                        y_err,
                                        xflow.flow,
                                        yflow.flow,
                                        [0.5, 3.5], [0.0, 20.0], [1.0, 100.0],
                                        [0.0, 10.0],
                                        int_method='auto',
                                        n=4,
                                        **tkw)
        assert x_ext.shape == w.shape

        # small errors -> ghq branch
        x_ext, _y_ext, w = get_xyw_trans(x_obs,
                                        y_obs,
                                        x_err,
                                        y_err,
                                        xflow.flow,
                                        yflow.flow,
                                        *RANGES,
                                        int_method='auto',
                                        n=4,
                                        **tkw)
        assert x_ext.shape == w.shape

    def test_invalid_int_method_raises(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        with pytest.raises(ValueError, match='Invalid integration method'):
            get_xyw_trans(x_obs,
                          y_obs,
                          x_err,
                          y_err,
                          xflow.flow,
                          yflow.flow,
                          *RANGES,
                          int_method='bogus',
                          n=4)

    def test_with_transfer(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        transfer = NegLogPlus()
        transfer_params = [float(np.median(x_obs)), float(np.median(x_err))]
        y_transfer = NegLogPlus()
        y_transfer_params = [float(np.median(y_obs)), float(np.median(y_err))]

        x_ext, _y_ext, w = get_xyw_trans(
            x_obs,
            y_obs,
            x_err,
            y_err,
            xflow.flow,
            yflow.flow,
            *RANGES,
            int_method='ghq',
            x_transfer=transfer,
            x_transfer_params=transfer_params,
            y_transfer=y_transfer,
            y_transfer_params=y_transfer_params,
            n=5)
        assert x_ext.shape == w.shape

    def test_ignore_y_err_and_w_add(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        tkw = transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err)
        x_ext, _y_ext, w = get_xyw_trans(x_obs,
                                        y_obs,
                                        x_err,
                                        y_err,
                                        xflow.flow,
                                        yflow.flow,
                                        *RANGES,
                                        int_method='ghq',
                                        ignore_y_err_c_y=True,
                                        get_w_add=get_w_add_lognorm,
                                        n=5,
                                        **tkw)
        assert x_ext.shape == w.shape
        assert float(jnp.nan_to_num(w).sum()) >= 0

    def test_batch_sizes(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        tkw = transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err)
        x_ext, _y_ext, w = get_xyw_trans(x_obs,
                                        y_obs,
                                        x_err,
                                        y_err,
                                        xflow.flow,
                                        yflow.flow,
                                        *RANGES,
                                        int_method='qmc',
                                        flow_batch_size=16,
                                        kde_batch_size=16,
                                        n=8,
                                        **tkw)
        assert x_ext.shape == w.shape


class TestKdePdfBatch:

    def test_batching_matches_single_call(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=250)
        pdf_one = kde_pdf_batch(FakeKernel(), x, batch_size=1000)
        pdf_batched = kde_pdf_batch(FakeKernel(), x, batch_size=100)
        np.testing.assert_allclose(pdf_one, pdf_batched, rtol=1e-6)


def test_get_w_add_lognorm():
    y_ext = jnp.array([1.0, 10.0])
    w = get_w_add_lognorm(jnp.array([5.0, 5.0]), y_ext,
                          jnp.array([0.1, 0.1]), jnp.array([0.1, 0.1]))
    np.testing.assert_allclose(w, 1 / np.array([1.0, 10.0]) / np.log(10))


def test_transform_data_none_passthrough():
    x = jnp.array([1.0, 2.0])
    x_err = jnp.array([0.1, 0.1])
    u, v = transform_data(x, x_err, None, None)
    assert u is x
    assert v is x_err


class TestGetXYWTransMixed:

    def test_mixed_methods(self, obs_data, tiny_flows):
        x_obs, x_err, y_obs, y_err = obs_data
        xflow, yflow = tiny_flows
        tkw = transfer_kwargs(xflow, yflow, x_obs, x_err, y_obs, y_err)
        # mixed-error data: half the points have big errors
        x_err_mixed = np.where(np.arange(len(x_err)) % 2 == 0, x_err,
                               x_err * 50)
        for method in ['ghq+qgrid', 'ghq+grid']:
            x_ext, _y_ext, w = get_xyw_trans(x_obs,
                                            y_obs,
                                            x_err_mixed,
                                            y_err,
                                            xflow.flow,
                                            yflow.flow,
                                            *RANGES,
                                            int_method=method,
                                            n=4,
                                            **tkw)
            assert x_ext.shape[0] == len(x_obs)
            assert float(jnp.nan_to_num(w).sum()) > 0

    def test_no_transfer_branches(self, obs_data, tiny_flows):
        # a flow trained without a transfer exercises the transfer-None
        # branches for x, while y keeps its transfer
        rng = np.random.default_rng(9)
        x = rng.uniform(1, 3, size=100)
        x_err = rng.uniform(0.05, 0.15, size=100)
        x_obs_plain = rng.normal(x, x_err)
        x_err_plain = x_err
        plain_flow = FlowVI(x_min=float(x_obs_plain.min()) * 0.5,
                            x_max=float(x_obs_plain.max()) * 1.1,
                            x_err_min=0,
                            x_err_max=float(x_err_plain.max()) * 1.2)
        plain_flow.fit(x_obs_plain,
                       x_err_plain,
                       n_pre=5,
                       n_train=10,
                       n_node=5,
                       early_stop=False,
                       x_min=float(x_obs_plain.min()) * 0.5,
                       x_max=float(x_obs_plain.max()) * 1.1,
                       x_err_min=0,
                       x_err_max=float(x_err.max()) * 1.2)

        _x_obs, x_err, y_obs, y_err = obs_data
        _, yflow = tiny_flows
        # paired arrays must have equal length
        n = len(x_obs_plain)
        x_ext, _y_ext, w = get_xyw_trans(
            x_obs_plain,
            y_obs[:n],
            x_err_plain,
            y_err[:n],
            plain_flow.flow,
            yflow.flow,
            *RANGES,
            int_method='ghq',
            y_transfer=yflow.transfer,
            y_transfer_params=yflow.transfer_params,
            n=5)
        assert x_ext.shape == w.shape
        assert float(jnp.nan_to_num(w).sum()) > 0


class TestGetXYWTransNoYTransfer:

    def test_both_transfers_none(self, obs_data):
        rng = np.random.default_rng(10)
        n = 100
        x = rng.uniform(1, 3, size=n)
        x_err = rng.uniform(0.05, 0.15, size=n)
        x_obs_plain = rng.normal(x, x_err)
        y = 10**(2 * np.log10(x) + 0.5 + rng.normal(scale=0.1, size=n))
        y_err = 0.05 * y
        y_obs_plain = rng.normal(y, y_err)

        def fit_plain(v_obs, v_err):
            flow = FlowVI(x_min=float(v_obs.min()) * 0.5,
                          x_max=float(v_obs.max()) * 1.1,
                          x_err_min=0,
                          x_err_max=float(v_err.max()) * 1.2)
            flow.fit(v_obs,
                     v_err,
                     n_pre=5,
                     n_train=10,
                     n_node=5,
                     early_stop=False,
                     x_min=float(v_obs.min()) * 0.5,
                     x_max=float(v_obs.max()) * 1.1,
                     x_err_min=0,
                     x_err_max=float(v_err.max()) * 1.2)
            return flow

        xflow_plain = fit_plain(x_obs_plain, x_err)
        yflow_plain = fit_plain(y_obs_plain, y_err)

        x_ext, _y_ext, w = get_xyw_trans(x_obs_plain,
                                        y_obs_plain,
                                        x_err,
                                        y_err,
                                        xflow_plain.flow,
                                        yflow_plain.flow,
                                        *RANGES,
                                        int_method='ghq',
                                        n=5)
        assert x_ext.shape == w.shape
        assert float(jnp.nan_to_num(w).sum()) > 0
