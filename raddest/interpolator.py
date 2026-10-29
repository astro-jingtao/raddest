"""PCHIP monotone-cubic interpolation wrapper (inverse support incomplete:
see TODO.md — MyPchipInterpolator cannot currently be instantiated)."""

from __future__ import annotations

from interpax import PchipInterpolator as _PchipInterpolator, PPoly as _PPoly
import numpy as np
from numpy.typing import ArrayLike


class MyPchipInterpolator:
    """PCHIP interpolator with a (not yet implemented) inverse polynomial.

    TODO: instantiating currently fails — get_inverse_coefficients_breakpoints
    does not exist (see TODO.md).
    """

    def __init__(self,
                 coefficients: ArrayLike | None = None,
                 breakpoints: ArrayLike | None = None) -> None:
        """Store the PPoly pieces and derive the inverse (unimplemented)."""
        self.coefficients = coefficients
        self.breakpoints = breakpoints
        self.inv_coefficients, self.inv_breakpoints = self.get_inverse_coefficients_breakpoints(
            coefficients, breakpoints)

    @classmethod
    def from_data(cls, x: ArrayLike, y: ArrayLike) -> MyPchipInterpolator:
        """Build from (x, y) samples via interpax's PchipInterpolator."""
        ppoly = _PchipInterpolator(x, y)
        return cls(coefficients=ppoly.c, breakpoints=ppoly.x)

    @classmethod
    def from_params(cls, params: ArrayLike) -> MyPchipInterpolator:
        """Build from a flat parameter vector (unimplemented)."""
        _coefficients, _breakpoints = ...
        return cls(coefficients=_coefficients, breakpoints=_breakpoints)

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the PPoly forward transform at x."""
        return _PPoly(c=self.coefficients, x=self.breakpoints)(x)

    def inverse_transform(self, y: ArrayLike) -> np.ndarray:
        """Evaluate the (unimplemented) inverse transform at y."""
        return _PPoly(c=self.inv_coefficients, x=self.inv_breakpoints)(y)

    def diff(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the derivative of the PPoly at x."""
        return _PPoly(c=self.coefficients, x=self.breakpoints).derivative()(x)
