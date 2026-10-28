# pylint: disable=wrong-import-position
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np

from raddest.shrink import shrink_possion

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
