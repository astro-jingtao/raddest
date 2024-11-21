import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np
import pandas as pd
from pzflow import Flow

from raddest.tools import get_p_flow


class TestGetPFlow:

    def test_in_batch_mode(self):
        x_obs = np.random.normal(size=2048)
        x_err = np.random.normal(size=2048)
        flow = Flow(["x_obs", "x_err"])
        flow.train(pd.DataFrame({"x_obs": x_obs, "x_err": x_err}))

        p_all = get_p_flow(flow, x_obs, x_err, batch_size=None)
        p_batch = get_p_flow(flow, x_obs, x_err, batch_size=256)
        assert p_all.shape == p_batch.shape
        assert np.allclose(p_all, p_batch, equal_nan=True, rtol=1e-4)
