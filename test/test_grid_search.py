import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

import numpy as np
import pytest

from raddest.grid_search import (
    flag_bad,
    gen_grid,
    gen_new_grid_mean_std,
    gen_new_grid_quantile,
    get_weighted_res,
    grid_search,
    quantile,
    zoom_in_grid,
)


def quad_compare(k, b, c):
    # (stat, p) pair minimized near (k, b, c) = (2, 1, 0.5)
    r = (k - 2)**2 + (b - 1)**2 + (c - 0.5)**2
    return r, np.exp(-10 * r)


def parser(res, iargs):
    res_arr = np.zeros((len(res), 5))
    for i, (r, a) in enumerate(zip(res, iargs)):
        res_arr[i, :] = (*a, r[0], r[1])
    return res_arr


class TestGridSearch:

    def test_grid_search_runs(self):
        params = [np.linspace(1, 3, 3), np.linspace(0, 2, 3),
                  np.linspace(0, 1, 2)]
        res_arr = grid_search(quad_compare, params, parser, num_process=2)
        assert res_arr.shape == (18, 5)
        # the best point is the closest grid point to (2, 1, 0.5)
        best = res_arr[np.argmax(res_arr[:, 4])]
        assert best[0] == pytest.approx(2.0)
        assert best[1] == pytest.approx(1.0)

    def test_zoom_in_grid_verbose_and_converge(self, capsys):
        params = [np.linspace(1, 3, 3), np.linspace(0, 2, 3),
                  np.linspace(0, 1, 2)]
        res_arr = zoom_in_grid(quad_compare,
                               params,
                               4,
                               parser,
                               gen_new_grid_mean_std,
                               lambda m, s: print('mean', m, 'std', s),
                               num_process=2,
                               max_i=3,
                               verbose=True)
        assert res_arr.shape[1] == 5
        assert 'mean' in capsys.readouterr().out

    def test_zoom_in_grid_max_iterations(self, capsys):
        params = [np.linspace(1, 3, 2), np.linspace(0, 2, 2),
                  np.linspace(0, 1, 2)]
        def drifting_grid(res_arr, *, n_par, w_index, **kwargs):
            # each call shifts the grid, so the weighted means keep moving
            # and the convergence check never passes
            drifting_grid.i = getattr(drifting_grid, 'i', 0) + 1
            off = 0.3 * drifting_grid.i
            return [np.linspace(1 + off, 3 + off, 3) for _ in range(n_par)]

        res_arr = zoom_in_grid(quad_compare,
                               params,
                               4,
                               parser,
                               drifting_grid,
                               lambda m, s: None,
                               num_process=2,
                               max_i=2,
                               verbose=True)
        assert res_arr.shape[1] == 5
        assert 'max iteration reached' in capsys.readouterr().out


class TestGenNewGrid:

    def test_mean_std_scalars_and_lists(self):
        res_arr = np.zeros((4, 6))
        res_arr[:, :3] = np.array([[1.9, 0.9, 0.4], [2.1, 1.1, 0.6],
                                   [1.8, 0.8, 0.3], [2.2, 1.2, 0.7]])
        res_arr[:, 5] = 1.0

        grid_s = gen_new_grid_mean_std(res_arr, n_par=3, w_index=5, n=5)
        grid_l = gen_new_grid_mean_std(res_arr,
                                       n_par=3,
                                       w_index=5,
                                       n=[5, 5, 5],
                                       n_sigma=[2, 2, 2],
                                       d_par_min=[0.2, 0.2, 0.2],
                                       range_par=[[-5, 5], [-5, 5], [-5, 5]])
        for gs, gl in zip(grid_s, grid_l):
            np.testing.assert_allclose(gs, gl)

    def test_quantile_scalars_and_lists(self):
        res_arr = np.zeros((4, 6))
        res_arr[:, :3] = np.array([[1.9, 0.9, 0.4], [2.1, 1.1, 0.6],
                                   [1.8, 0.8, 0.3], [2.2, 1.2, 0.7]])
        res_arr[:, 5] = 1.0

        grid_s = gen_new_grid_quantile(res_arr, n_par=3, w_index=5, n=5)
        grid_l = gen_new_grid_quantile(res_arr,
                                       n_par=3,
                                       w_index=5,
                                       n=[5, 5, 5],
                                       q_low=[0.16, 0.16, 0.16],
                                       q_up=[0.84, 0.84, 0.84],
                                       d_par_min=[0.2, 0.2, 0.2],
                                       range_par=[[-5, 5], [-5, 5], [-5, 5]])
        for gs, gl in zip(grid_s, grid_l):
            np.testing.assert_allclose(gs, gl)

    def test_gen_grid_clips_to_range(self):
        grid = gen_grid(0.0, 10.0, 10.0, 5, [-1.0, 1.0])
        assert grid.min() >= -1.0
        assert grid.max() <= 1.0


class TestWeightedRes:

    def test_mean_std(self):
        xs = np.array([[1.0, 3.0], [2.0, 4.0]])
        w = np.array([1.0, 3.0])
        mean, std = get_weighted_res(xs, w)
        assert mean[0] == pytest.approx(2.5)
        assert mean[1] == pytest.approx(3.5)
        assert (std >= 0).all()

    def test_std_floor(self):
        xs = np.array([[1.0, 1.0], [1.0, 1.0]])
        w = np.array([1.0, 1.0])
        mean, std = get_weighted_res(xs, w, std_min=0.5)
        assert std[0] == pytest.approx(0.5)


class TestQuantile:

    def test_unweighted_matches_numpy(self):
        x = np.linspace(0, 1, 101)
        res = quantile(x, [0.25, 0.5, 0.75])
        np.testing.assert_allclose(res, [0.25, 0.5, 0.75], atol=0.01)

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        res = quantile(x, [0.5], weights=w)
        # the weighted cdf normalizes by cumsum[:-1][-1], giving 2.5 here
        assert res[0] == pytest.approx(2.5, abs=0.01)

    def test_too_few_samples_returns_nan(self):
        res = quantile(np.array([1.0]), [0.5], N_min=2)
        assert np.isnan(res[0])

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError, match='between 0 and 1'):
            quantile(np.arange(10.0), [1.5])

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='Dimension mismatch'):
            quantile(np.arange(10.0), [0.5], weights=np.ones(3))


def test_flag_bad():
    x = np.array([0.0, np.nan, np.inf, 1.0])
    np.testing.assert_array_equal(flag_bad(x), [False, True, True, False])
