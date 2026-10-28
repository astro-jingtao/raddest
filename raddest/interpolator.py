from interpax import PchipInterpolator as _PchipInterpolator, PPoly as _PPoly


class MyPchipInterpolator:

    def __init__(self, coefficients=None, breakpoints=None):
        self.coefficients = coefficients
        self.breakpoints = breakpoints
        self.inv_coefficients, self.inv_breakpoints = self.get_inverse_coefficients_breakpoints(
            coefficients, breakpoints)

    @classmethod
    def from_data(cls, x, y):
        ppoly = _PchipInterpolator(x, y)
        return cls(coefficients=ppoly.c, breakpoints=ppoly.x)

    @classmethod
    def from_params(cls, params):
        _coefficients, _breakpoints = ...
        return cls(coefficients=_coefficients, breakpoints=_breakpoints)

    def transform(self, x):
        return _PPoly(c=self.coefficients, x=self.breakpoints)(x)

    def inverse_transform(self, y):
        return _PPoly(c=self.inv_coefficients, x=self.inv_breakpoints)(y)

    def diff(self, x):
        return _PPoly(c=self.coefficients, x=self.breakpoints).derivative()(x)
