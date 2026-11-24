"""Classic linear fits (OLS/WLS/ODR) used as references in the example
notebooks."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.odr import ODR, Data, Model
import statsmodels.api as sm

from utils.tools import all_asarray, remove_bad


def get_linear_model(k: float, b: float) -> Callable:
    """Closure computing y = k x + b."""

    def func(x: ArrayLike) -> np.ndarray:
        return k * x + b

    return func


def preprocess(xs: list) -> list:
    """
    allasarray + remove_bad
    """
    return remove_bad(all_asarray(xs))


def xy2logxy(
    x: ArrayLike,
    y: ArrayLike,
    x_err: ArrayLike,
    y_err: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform (x, y, x_err, y_err) to log10 space with error propagation."""
    logx = np.log10(x)
    logy = np.log10(y)
    logx_err = x_err / (x * np.log(10))
    logy_err = y_err / (y * np.log(10))
    return logx, logy, logx_err, logy_err


def get_OLS(x: ArrayLike, y: ArrayLike, return_res: bool = False) -> Any:
    """Ordinary least squares y = kx + b; returns the statsmodels results when
    return_res, else a dict with k/b (value, err), residual std, and func."""
    # y = kx + b
    x, y = preprocess([x, y])

    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()

    if return_res:
        return results

    b, k = results.params
    b_err, k_err = results.HC1_se

    func = get_linear_model(k, b)

    std = np.std(func(x) - y)

    return {"k": (k, k_err), "b": (b, b_err), "std": std, "func": func}


def get_WLS(x: ArrayLike, y: ArrayLike, y_err: ArrayLike, return_res: bool = False) -> Any:
    """Weighted least squares y = kx + b with 1/y_err^2 weights; returns as
    get_OLS does."""
    # y = kx + b
    x, y, y_err = preprocess([x, y, y_err])

    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=1.0 / np.square(y_err))
    results = model.fit()

    if return_res:
        return results

    b, k = results.params
    b_err, k_err = results.HC1_se

    func = get_linear_model(k, b)

    std = np.std(func(x) - y)

    return {"k": (k, k_err), "b": (b, b_err), "std": std, "func": func}


def get_ODR(
    x: ArrayLike,
    y: ArrayLike,
    x_err: ArrayLike | None = None,
    y_err: ArrayLike | None = None,
    return_res: bool = False,
) -> Any:
    """Orthogonal distance regression y = kx + b with errors on both axes;
    returns the ODR output when return_res, else a dict with k/b (value,
    err), per-axis residual stds, and both direction functions."""

    # https://docs.scipy.org/doc/scipy/reference/odr.html

    def f(B: ArrayLike, x: ArrayLike) -> np.ndarray:
        return B[0] * x + B[1]

    if x_err is None:
        x_err = np.ones_like(x)
    if y_err is None:
        y_err = np.ones_like(y)

    x, y, x_err, y_err = preprocess([x, y, x_err, y_err])

    linear = Model(f)
    mydata = Data(x, y, wd=1.0 / np.square(x_err), we=1.0 / np.square(y_err))
    myodr = ODR(mydata, linear, beta0=[1.0, 0.0])
    results = myodr.run()

    if return_res:
        return results

    k, b = results.beta
    k_err, b_err = results.sd_beta

    func_xy = get_linear_model(k, b)
    func_yx = get_linear_model(1 / k, -b / k)

    x_std = np.std(func_yx(y) - x)
    y_std = np.std(func_xy(x) - y)

    d_std = np.std((func_xy(x) - y) / np.sqrt(k**2 + 1))

    return {
        "k": (k, k_err),
        "b": (b, b_err),
        "x_std": x_std,
        "y_std": y_std,
        "d_std": d_std,
        "func_xy": func_xy,
        "func_yx": func_yx,
    }
