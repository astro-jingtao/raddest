import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np
from raddest.transfer import (LogRotation, NegLogPlus, Transfer,
                              combine_transfers)


class TestNegLogPlus:

    def test_consistency(self):
        neglogplus = NegLogPlus()
        x, x_err = 1, 1
        params = [1, 1]
        x_loop, x_err_loop = neglogplus.inv(
            *neglogplus.trans(x, x_err, params), params)
        assert x == x_loop
        assert x_err == x_err_loop

    def test_save_load(self):
        neglogplus = NegLogPlus()
        neglogplus.save('test_neglogplus.pkl')
        neglogplus_loaded = Transfer.load('test_neglogplus.pkl')

        x, x_err = 1, 1
        params = [1, 1]
        u_ori, v_ori = neglogplus.trans(x, x_err, params)
        u_load, v_load = neglogplus_loaded.trans(x, x_err, params)
        assert u_ori == u_load
        assert v_ori == v_load

        os.remove('test_neglogplus.pkl')


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

        combined.save('test_combined.pkl')
        combined_loaded = Transfer.load('test_combined.pkl')

        x, x_err = 1, 1
        params = [1, 1, 1, 1, 1]
        u_ori, v_ori = combined.trans(x, x_err, params)
        u_load, v_load = combined_loaded.trans(x, x_err, params)
        assert u_ori == u_load
        assert v_ori == v_load

        os.remove('test_combined.pkl')
