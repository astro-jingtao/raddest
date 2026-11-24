# Method Guide — worked recipes per fitting family

Read when writing actual fitting code. Each section is a complete, minimal
call sequence; the classes come from `raddest.api.fitter` and
`raddest.api.flowvi`, the transfer from `raddest.transfer`.

## 1. Flow fitting preamble (every family)

```python
import numpy as np
from raddest.api.flowvi import FlowVI
from raddest.transfer import NegLogPlus


def fit_flow(v_obs, v_err):
    flow = FlowVI(
        x_min=float(v_obs.min()) * 0.5,
        x_max=float(v_obs.max()) * 1.1,
        x_err_min=0,
        x_err_max=float(v_err.max()) * 1.2,
        transfer=NegLogPlus(),
        transfer_params=[float(np.median(v_obs)), float(np.median(v_err))],
    )
    flow.fit(
        v_obs,
        v_err,
        n_pre=100,
        n_train=1000,
        n_node=21,
        x_min=float(v_obs.min()) * 0.5,
        x_max=float(v_obs.max()) * 1.1,
        x_err_min=0,
        x_err_max=float(v_err.max()) * 1.2,
    )
    return flow


xflow = fit_flow(x_obs, x_err)
yflow = fit_flow(y_obs, y_err)
```

`NegLogPlus` maps a strictly positive variable to an unbounded one; use it
whenever the quantity is positive (fluxes, masses, radii). Omit `transfer`
for variables that can be negative. Raise `n_train` until the flow's own
KS check (`flow.ks_test(v_obs, v_err, 10000)`) no longer rejects.

## 2. ML likelihood family

```python
from raddest.api.fitter import LogLinearML

regressor = LogLinearML(xflow, yflow)
params_ml, params_std, cov = regressor.fit(
    x_obs,
    x_err,
    y_obs,
    y_err,
    params_0=np.array([1.5, 0.0, 0.2]),  # (k, b, sig)
)
```

`fit` runs gradient ascent on the exact likelihood (Gauss-Hermite
integration over the error convolution) and returns the ML parameters,
their std from the inverse Hessian, and the covariance. Start `sig` (last
element) positive; bad starts mostly manifest as slow convergence — check
`verbose=True` output.

Piecewise variant: `PiecewiseLogLinearML(xflow, yflow, n_piece)` with
parameter layout `(b0, x0, k_1..k_n, dx_1..dx_{n-1}, sig)` — intercept and
first breakpoint, per-piece slopes, breakpoint spacings (positive), and
scatter.

## 3. KS grid family

```python
import numpy as np
from raddest.api.fitter import LogLinearKS

regressor = LogLinearKS(xflow, yflow)
params_mean, params_std, res_arr = regressor.fit(
    x_obs,
    y_obs,
    params_list=[
        np.linspace(1.0, 3.0, 10),  # k
        np.linspace(-0.5, 1.5, 10),  # b
        np.linspace(0.05, 0.2, 5),  # sig
    ],
    N=1000,
    num_process=10,
)
```

`res_arr` rows are `(k, b, sig, D, p)` — parameter values plus the KS
statistic and p-value of that grid point. The zoom-in loop re-centers the
grid on the p-weighted parameter mean; the returned `params_mean` /
`params_std` summarize the final grid. Set `refine=False` to skip the final
dense pass.

## 4. Energy grid family

```python
from raddest.api.fitter import LogLinearEnergyPT

regressor = LogLinearEnergyPT(xflow, yflow, permutations=200)
params_mean, params_std, res_arr = regressor.fit(
    x_obs,
    y_obs,
    params_list=[
        np.linspace(1.0, 3.0, 10),
        np.linspace(-0.5, 1.5, 10),
        np.linspace(0.05, 0.2, 5),
    ],
    N=1000,
    num_process=10,
)
```

Same grid machinery as the KS family; each grid point's weight is the
one-tailed permutation p-value of the two-sample energy distance between
generated and observed 2D points. `res_arr` rows are `(k, b, sig, E, p)`.
The p-value resolution is `1 / (1 + permutations)`; each point costs
O((N + M)^2 × permutations), so lower `N` or `permutations` first when the
grid is large.

## 5. Energy gradient family

```python
import jax.numpy as jnp
from raddest.api.fitter import LogLinearEnergyGrad

regressor = LogLinearEnergyGrad(xflow, yflow)
params_hat, loss_history = regressor.fit(
    x_obs,
    y_obs,
    jnp.asarray([1.5, 0.0, 0.2], dtype=jnp.float32),  # (k, b, sig)
    N=1000,
    lr=5e-2,
)
```

The synthetic sample is generated differentiably from the parameters with
fixed noise draws (seed via `seed=`), and adam minimizes the energy
distance to the observed sample. `loss_history` holds the loss at every
`check_per_step` steps — a flat tail means convergence, a constant high
loss means a bad start or a too-small `N`. There is no Hessian covariance
here; propagate uncertainty by bootstrap if needed.

## 6. Evaluation

Both tests compare a fresh synthetic sample generated at given parameters
against the observed data:

```python
D, p = regressor.ks_test((k_hat, b_hat, sig_hat), x_obs, y_obs, N=10000)
E, p = regressor.energy_test((k_hat, b_hat, sig_hat), x_obs, y_obs, N=10000, permutations=200)
```

`energy_test` also accepts `M` (subsample size of the observed data) and
`K` (repeat count, returning lists). Prefer comparing p-values between
candidate parameter sets over reading one absolute p-value as verdict.

## 7. Ranges

All regressor calls accept `x_range`, `x_err_range`, `y_range`,
`y_err_range`; when None they default to the flows' fitted ranges. Pass
explicit ranges when fitting on a subset region (for example excluding
outliers below a flux cut) — otherwise the generator samples the full
flow range and the comparison mixes regions.
