import numpy as np
import pytest

from raddest.api.utils import to_each_piece_params
from raddest.reg.constructor import flat_piecewise_params, parse_piecewise_params


class TestPiecewiseParams:

    @pytest.mark.parametrize(("params", "flatted_params"), [
        ((0, 1, (2, 2), ()), (0, 1, 2, 2)),
        ((1, 2, (3, 4, -2), (5, )), (1, 2, 3, 4, -2, 5)),
    ])
    def test_flat_piecewise_params(self, params, flatted_params):
        assert flat_piecewise_params(*params) == flatted_params

    @pytest.mark.parametrize(("n_piece", "params", "parsed_params"), [
        (2, (0, 1, 2, 2), (0, 1, (2, 2), ())),
        (3, (1, 2, 3, 4, -2, 5), (1, 2, (3, 4, -2), (5, ))),
    ])
    def test_parse_piecewise_params(self, n_piece, params, parsed_params):
        assert parse_piecewise_params(n_piece, params) == parsed_params

    @pytest.mark.parametrize(("n_piece", "params", "each_piece_params"), [
        (2, (0, 1, 2, 2, 3), ((2, 2), (0, 0), (1, ), 3)),
        (3, (1, 2, 3, 4, -2, 5, 6), ((3, 4, -2), (1, -1, 41), (2, 7), 6)),
    ])
    def test_to_each_piece_params(self, n_piece, params, each_piece_params):
        for p, p_tar in zip(to_each_piece_params(n_piece, params),
                            each_piece_params):
            assert np.allclose(p, p_tar)
