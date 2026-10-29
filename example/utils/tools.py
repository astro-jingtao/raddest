"""Array hygiene helpers shared by the example linear-model fits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def flag_bad(x: ArrayLike) -> np.ndarray:
    """
    It returns True if the input is NaN or Inf, and False otherwise

    :param x: the input data
    :return: A boolean array of the same shape as x, where True indicates that the corresponding element
    of x is NaN or +/-inf.
    """
    return np.isnan(x) | np.isinf(x)


def remove_bad(xs: list) -> list:
    """Drop rows flagged bad in any array of the list (bad = NaN/Inf; 2D
    arrays flag per row)."""
    if xs[0].ndim == 1:
        n_sample = len(xs[0])
    else:
        n_sample = xs[0].shape[0]
    bad = np.zeros(n_sample, dtype=bool)
    for x in xs:
        if x.ndim == 1:
            bad |= flag_bad(x)
        else:
            bad |= flag_bad(x).any(axis=1)
    return [x[~bad] for x in xs]


def all_asarray(xs: list) -> list:
    """np.asarray every entry of the list."""
    return [np.asarray(x) for x in xs]
