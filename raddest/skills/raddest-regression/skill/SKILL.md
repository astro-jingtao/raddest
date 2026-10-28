---
name: raddest-regression
description: Regress y on x with realistic observational errors, non-Gaussian distributions, and nonlinear relations using the raddest package (JAX-based). Use when fitting a relation between two measured quantities whose values AND errors both scatter, when errors convolve through a nonlinearity, or when asked to use raddest, FlowVI, or any raddest regressor class (LogLinearML, LinearML, PiecewiseLogLinearML, SaturatedLogLinearML, LogLinearStudentTML, LogLinearKS, LinearKS, LogLinearEnergyPT, LogLinearEnergyGrad), or to pick between likelihood, KS-grid, and energy-distance fitting for such data.
---

# raddest Regression

## Role

You fit a y-versus-x relation to astronomical-style data where each point carries an observational error on both axes and the intrinsic distribution matters. The raddest package models each variable's joint (value, error) distribution with a normalizing flow (FlowVI), then fits a conditional relation on top. Your job: structure the data, fit the flows, choose the fitting method deliberately, fit, and evaluate.

## Hard Rules

1. **Verify the import first.** Run `import raddest.api.fitter` in the target project's interpreter before any advice executes. If it fails, stop and ask the user which environment has raddest installed — never guess an environment.
2. **Two flows before any regressor.** Every regressor needs an xflow and a yflow, each a FlowVI fitted to that variable's (value, error) pairs. An unfitted flow passed to a regressor is a defect.
3. **Never instantiate `PiecewiseLogLinearStudentTML`.** It is documented as not implemented and raises NameError as shipped.
4. **Never fit a method because it is first in a list.** Pick per the decision table below; the families differ in what they assume and what they cost.

## Data Requirements

Four equal-length 1D arrays per dataset: `x_obs`, `x_err`, `y_obs`, `y_err` (observed values and their per-point observational errors). The underlying model assumes each observed value is the true value plus Gaussian noise of the given error, with the true values drawn from the flow-modeled intrinsic distribution.

## Workflow

1. Verify the environment (Hard Rule 1).
2. Collect the four arrays; confirm equal lengths and non-negative errors.
3. Fit the two flows — one FlowVI per variable (recipe in `references/method-guide.md` §1).
4. Choose the fitting family with the decision table.
5. Fit; for likelihood fits also obtain the covariance (`fit` returns it); for grid families read the weighted mean and std it returns.
6. Evaluate: run `ks_test` or `energy_test` on the fitted model at the fitted parameters; a good fit gives a non-small p-value, and comparing p-values between candidate parameter sets is more informative than one absolute number.
7. Report parameters with their standard deviations, the method used, and the evaluation p-value.

## Decision Table — choosing the fitting family

| Family | Classes | Criterion optimized | Use when | Cost |
| ------ | ------- | ------------------- | -------- | ---- |
| ML likelihood | `LogLinearML`, `LinearML`, `SaturatedLogLinearML`, `PiecewiseLogLinearML`, `LogLinearStudentTML` | exact likelihood (autodiff) | a parametric p(y\|x) form fits the science; fastest and gives a covariance | low |
| KS grid | `LogLinearKS`, `LinearKS` | 2D two-sample KS via zoom-in grid | no usable likelihood form; distribution-distance criterion wanted; grid-friendly (≤4) parameter count | high (grid × generation) |
| Energy grid | `LogLinearEnergyPT` | energy-test p-value via same zoom-in grid | same situations as KS grid, but an exact two-sample test statistic (permutation energy distance) is preferred over KS | high, plus O((N+M)^2 × permutations) per grid point |
| Energy gradient | `LogLinearEnergyGrad` | energy distance itself, minimized by adam (differentiable generation) | distribution-distance criterion wanted AND a smooth parametric generator exists; no grid | medium |

Parameter conventions: log-linear families fit y = 10^(k·log10(x) + b + sig·ε); linear families y = k·x + b + sig·ε; `sig` is the intrinsic scatter in log10-space (or linear space) and must stay positive.

Full call recipes, including the FlowVI preamble, ranges handling, and per-family fit calls: read `references/method-guide.md` when actually writing the fitting code (steps 3–5).

## Out of Scope

- Building custom normalizing flows beyond `FlowVI`, and custom `Transfer` functions beyond the shipped ones — the package's own docs own that.
- MCMC posterior sampling mechanics (`get_posterior_mcmc`, `get_posterior_qmc`) — use only when the user explicitly asks; their emcee/Sobol interfaces are self-describing.
- Denoising or shrinkage entry points (`raddest.shrink`, `raddest.tools`) — separate concerns, not part of the regression loop.

## Supporting Files

- `references/method-guide.md` — worked recipes: flow fitting preamble, one section per family with exact fit calls, evaluation calls, and the parameter vector layouts. Read during steps 3–5; skip when only advising on method choice.
