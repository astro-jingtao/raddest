

import shutil

import numpy as np
import pytest

from raddest.api.flowvi import FlowVI
from raddest.api.post_processer import get_shift
from raddest.api.utils import to_each_piece_params, to_each_piece_params_studentT
from raddest.transfer import NegLogPlus, XShiftNegLogPlus
from raddest.x_xerr import construct_x_xerr_w


class TestToEachPieceParams:

    def test_two_pieces(self):
        # (b0, x0, k1, k2, dx1, sig)
        params = [0.0, 1.0, 2.0, 3.0, 0.5, 0.1]
        k_lst, b_lst, x_lst, sig = to_each_piece_params(2, params)
        assert list(k_lst) == [2.0, 3.0]
        # second intercept: (k1 - k2) * x0 + b0 = -1
        assert list(b_lst) == [0.0, -1.0]
        assert list(x_lst) == [1.0]
        assert sig == 0.1

    def test_student_t_variant(self):
        # (b0, x0, k1, k2, dx1, df, sig)
        params = [0.0, 1.0, 2.0, 3.0, 0.5, 7.0, 0.1]
        k_lst, _b_lst, _x_lst, sig, df = to_each_piece_params_studentT(2, params)
        assert list(k_lst) == [2.0, 3.0]
        assert df == 7.0
        assert sig == 0.1


class TestGetShift:

    DATA = (np.array([1.0, 2.0, 2.0, 3.0, 10.0]), np.array([2.0, 3.0, 3.0, 4.0,
                                                            11.0]))

    def test_median_and_mean(self):
        x_obs, x_gen = self.DATA
        assert get_shift(x_obs, x_gen, 'median') == pytest.approx(1.0)
        assert get_shift(x_obs, x_gen, 'mean') == pytest.approx(
            float(np.mean(x_gen) - np.mean(x_obs)))

    def test_histogram(self):
        x_obs, x_gen = self.DATA
        shift = get_shift(x_obs, x_gen, 'histogram')
        assert np.isfinite(shift)


@pytest.mark.slow
class TestFlowVIBranches:

    def test_transfer_params_ignored_warning(self, capsys):
        _flow_vi = FlowVI(x_min=0,
                          x_max=10,
                          x_err_min=0,
                          x_err_max=10,
                          transfer=None,
                          transfer_params=[1, 1])
        assert 'ignored' in capsys.readouterr().out

    def test_transfer_without_params_raises(self):
        # a transfer without an estimator and without explicit params raises
        with pytest.raises(ValueError, match='provide transfer_params'):
            FlowVI(x_min=0,
                   x_max=10,
                   x_err_min=0,
                   x_err_max=10,
                   transfer=XShiftNegLogPlus(),
                   transfer_params=None)

    def test_transfer_params_wrong_length_raises(self):
        with pytest.raises(ValueError, match='same length'):
            FlowVI(x_min=0,
                   x_max=10,
                   x_err_min=0,
                   x_err_max=10,
                   transfer=NegLogPlus(),
                   transfer_params=[1, 2, 3])

    def test_check_fit_raises(self):
        flow_vi = FlowVI(x_min=0,
                         x_max=10,
                         x_err_min=0,
                         x_err_max=10)
        with pytest.raises(ValueError, match='not fitted'):
            flow_vi.check_fit()
        del flow_vi

    def test_save_without_transfer(self, obs_data):
        x_obs, x_err, _, _ = obs_data
        flow_vi = FlowVI(x_min=0,
                         x_max=float(x_obs.max()),
                         x_err_min=0,
                         x_err_max=float(x_err.max()) * 1.2)
        flow_vi.fit(x_obs,
                    x_err,
                    n_pre=5,
                    n_train=5,
                    n_node=5,
                    early_stop=False)
        flow_vi.save('test_flowvi_notransfer')
        loaded = FlowVI.load('test_flowvi_notransfer')
        assert loaded.transfer is None
        assert loaded.flow is not None
        shutil.rmtree('test_flowvi_notransfer')

    def test_ks_test_m_warning(self, tiny_flows, obs_data, capsys):
        xflow, _ = tiny_flows
        x_obs, _, _, _ = obs_data
        np.random.seed(0)
        _D, _p = xflow.ks_test(x_obs, x_obs, N=100, M=10**6)
        assert 'Warning' in capsys.readouterr().out

    def test_sample_with_explicit_ranges(self, tiny_flows):
        xflow, _ = tiny_flows
        out = xflow.sample(100,
                           x_min=xflow.x_min,
                           x_max=xflow.x_max,
                           x_err_min=xflow.x_err_min,
                           x_err_max=xflow.x_err_max)
        x_gen, x_err_gen, x_obs_gen = out
        assert np.asarray(x_obs_gen).shape == (100, )
        del x_gen, x_err_gen


import jax.numpy as jnp


class TestXXErrExtras:

    def test_construct_x_xerr_w_invalid_method(self):
        with pytest.raises(ValueError, match='Invalid integration method'):
            construct_x_xerr_w(jnp.array([1.0, 2.0]), jnp.array([0.1, 0.1]),
                               int_method='bogus')
