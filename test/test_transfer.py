import os

import jax.numpy as jnp
import numpy as np

from raddest.transfer import (
    LogRotation,
    NegLogPlus,
    NegLogPlusXShift,
    Transfer,
    XShift,
    XShiftNegLogPlus,
    combine_transfers,
)


class TestNegLogPlus:
    def test_consistency(self):
        neglogplus = NegLogPlus()
        x, x_err = 1, 1
        params = [1, 1]
        x_loop, x_err_loop = neglogplus.inv(*neglogplus.trans(x, x_err, params), params)
        assert x == x_loop
        assert x_err == x_err_loop

    def test_save_load(self):
        neglogplus = NegLogPlus()
        neglogplus.save("test_neglogplus.pkl")
        neglogplus_loaded = Transfer.load("test_neglogplus.pkl")

        x, x_err = 1, 1
        params = [1, 1]
        u_ori, v_ori = neglogplus.trans(x, x_err, params)
        u_load, v_load = neglogplus_loaded.trans(x, x_err, params)
        assert u_ori == u_load
        assert v_ori == v_load

        os.remove("test_neglogplus.pkl")


class TestCombinedTransfer:
    def test_consistency(self):
        t1 = NegLogPlus()
        t2 = LogRotation()
        t3 = NegLogPlus()

        combined = combine_transfers(t1, t2, t3)

        x, x_err = 1, 1
        params = [1, 1, 1, 1, 1]
        u1, v1 = t1.trans(x, x_err, params[:2])
        u2, v2 = t2.trans(u1, v1, params[2:3])
        u3, v3 = t3.trans(u2, v2, params[3:])
        u_comb, v_comb = combined.trans(x, x_err, params)

        assert u_comb == u3
        assert v_comb == v3

        u2_inv, v2_inv = t3.inv(u3, v3, params[3:])
        u1_inv, v1_inv = t2.inv(u2_inv, v2_inv, params[2:3])
        x_loop, x_err_loop = t1.inv(u1_inv, v1_inv, params[:2])
        x_loop_comb, x_err_loop_comb = combined.inv(u_comb, v_comb, params)

        assert np.isclose(x, x_loop_comb)
        assert np.isclose(x_err, x_err_loop_comb)

        assert np.isclose(x_loop, x_loop_comb)
        assert np.isclose(x_err_loop, x_err_loop_comb)

        jac_1 = t1.jac(x, x_err, params[:2])
        jac_2 = t2.jac(u1, v1, params[2:3])
        jac_3 = t3.jac(u2, v2, params[3:])
        jac_comb = combined.jac(x, x_err, params)

        assert np.isclose(jac_comb, jac_1 * jac_2 * jac_3)

    def test_save_load(self):
        t1 = NegLogPlus()
        t2 = LogRotation()
        t3 = NegLogPlus()

        combined = combine_transfers(t1, t2, t3)

        combined.save("test_combined.pkl")
        combined_loaded = Transfer.load("test_combined.pkl")

        x, x_err = 1, 1
        params = [1, 1, 1, 1, 1]
        u_ori, v_ori = combined.trans(x, x_err, params)
        u_load, v_load = combined_loaded.trans(x, x_err, params)
        assert u_ori == u_load
        assert v_ori == v_load

        os.remove("test_combined.pkl")


class TestShiftTransfers:
    def test_xshift_roundtrip(self):
        t = XShift()
        params = [0.5]
        x, x_err = 2.0, 0.1
        u, v = t.trans(x, x_err, params)
        x_back, x_err_back = t.inv(u, v, params)
        assert np.isclose(float(x_back), x)
        assert np.isclose(float(x_err_back), x_err)
        assert float(t.jac(x, x_err, params)) == 1.0

    def test_xshift_neglogplus_roundtrip(self):
        t = XShiftNegLogPlus()
        params = [1.0, 0.1, 0.5]
        x, x_err = 2.0, 0.3
        u, v = t.trans(x, x_err, params)
        x_back, x_err_back = t.inv(u, v, params)
        assert np.isclose(float(x_back), x, rtol=1e-5)
        assert np.isclose(float(x_err_back), x_err, rtol=1e-5)
        jac = t.jac(x, x_err, params)
        assert np.isfinite(float(jac))

        # nonpositive inputs take the constant branch of the jacobian
        jac_neg = t.jac(-5.0, -0.5, params)
        assert np.isfinite(float(jac_neg))

    def test_neglogplus_xshift_roundtrip(self):
        t = NegLogPlusXShift()
        params = [1.0, 0.1, 0.5]
        x, x_err = 2.0, 0.3
        u, v = t.trans(x, x_err, params)
        x_back, x_err_back = t.inv(u, v, params)
        assert np.isclose(float(x_back), x, rtol=1e-5)
        assert np.isclose(float(x_err_back), x_err, rtol=1e-5)
        assert np.isfinite(float(t.jac(x, x_err, params)))
        assert np.isfinite(float(t.jac(-5.0, -0.5, params)))

    def test_params_estimators(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1, 3, size=50)
        x_err = rng.uniform(0.1, 0.3, size=50)

        p = NegLogPlus().params_estimator(jnp.asarray(x), jnp.asarray(x_err))
        assert len(p) == 2
        assert np.isclose(float(p[0]), float(np.median(x)))

    def test_shift_transfers_have_no_estimator(self):
        assert XShiftNegLogPlus().params_estimator is None
        assert NegLogPlusXShift().params_estimator is None


class TestCombineTransfersEdge:
    def test_none_cases(self):
        t = NegLogPlus()
        assert combine_transfers(t, None) is t
        assert combine_transfers(None, t) is t
        assert combine_transfers(None) is None

    def test_combined_estimator(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(1, 3, size=50)
        x_err = rng.uniform(0.1, 0.3, size=50)

        t = combine_transfers(NegLogPlus(), NegLogPlus())
        assert t.params_estimator is not None
        p = t.params_estimator(jnp.asarray(x), jnp.asarray(x_err))
        assert len(p) == 4

    def test_combined_without_estimator(self):
        def _id(x, x_err, params):
            return x, x_err

        def _jac(x, x_err, params):
            return 1.0

        t1 = Transfer(_id, _id, _jac, 0, name="NoEst1")
        t2 = Transfer(_id, _id, _jac, 0, name="NoEst2")
        combined = combine_transfers(t1, t2)
        assert combined.params_estimator is None
