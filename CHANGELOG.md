# Change Log

## [unreleased]

### Added

- Example notebook `example/fit_loglinear_energy_pt_jax.ipynb`: `LogLinearEnergyPT(backend='jax')` at GPU scale — 200 permutations per grid point, a 2000-point synthetic sample at every point, full zoom-in plus refinement (whole fit ~13 s on one GPU), with goodness-of-fit and model-vs-data checks.
- All seven example notebooks are verified by full end-to-end execution and the failures repaired: the piecewise and saturation notebooks call the keyword-only model functions in their keyword forms (`piecewise_linear`, `saturated_linear`, `saturated_linear_smooth(_jax)`; star-unpack call sites rewritten), the removed `PiecewiseLogLinearML(is_fixed=...)` idiom is replaced by its `A_eq`/`b_eq` hyperplane equivalent, `fit_x_xerr.ipynb` imports `FlowVI` from `raddest.api.flowvi`, the saturation notebook's MCMC walker init caps the Hessian-based sigma band (its `(x0, x1)` are weakly identified and produce NaN std entries), and every notebook's CUDA device cell uses `os.environ.setdefault` so an externally selected GPU wins over the old hardcoded index.
- Cross-backend parity tests for the jax and numpy engines: same-data statistic/p-value agreement of `energy_test` vs `energy_test_jax` (identical samples, fully shifted samples, same-distribution p-values within Monte-Carlo noise, asymmetric sample splits), the numpy grid compare vs the jax grid compare on identical inputs (the jax statistic must fall inside the numpy noise-realization band), and a `LogLinearEnergyPT` fit through both backends on shared flows and data (grid points identical, stat columns correlated > 0.995, `params_mean`/`params_std` agreeing within calibrated tolerances).

### Changed

- `RegressorKS` is split into `RegressorGrid` — the statistic-agnostic zoom-in grid machinery, whose `compare_constructor` now defaults to None with a clear `fit` error when no statistic is bound — and a thin KS-specific `RegressorKS(RegressorGrid)` layer binding `construct_ks_compare`. `LogLinearKS`/`LinearKS` inherit the KS layer (behavior unchanged), and `RegressorEnergyPT` inherits `RegressorGrid` directly instead of riding the KS name. The `fit` docstring's statistic references are neutralized accordingly.
- `raddest.api.post_processer` is renamed to `raddest.api.post_processor` (the module filename was misspelled); the test file renames with it.
- Lint no longer selects PLR0913 (too-many-arguments): estimator and training interfaces in this repo legitimately carry one slot per physical knob, so the rule only produced blanket waivers; the 27 `# noqa: PLR0913` comments (and their reason blocks) are removed and `ruff.toml` drops the rule and its `max-args` setting.
- All relative imports are converted to absolute (`from raddest.x import ...`, `from utils.tools import ...` in the example helpers, 13 sites) and TID252 (ban relative imports) is selected in `ruff.toml` to keep the repo absolute.
- The whole repository is formatted with `ruff format` (double quotes, 100 columns): package, tests, example helpers, and the Python code blocks embedded in `README.md` and the shipped `raddest-regression` skill's method guide; `ruff format --check` is clean.
- `example/utils` (notebook helpers) gains type annotations, docstrings, and package/module docstrings; `setup.py` a module docstring.
- Package misc: module docstrings for `raddest`, `raddest.GaussHermiteQuadrature`, `raddest.interpolator`; type annotations and docstrings in `raddest.interpolator` (its known dead reference stays recorded in TODO.md) and the jit closures of `raddest.x_xerr`.
- `raddest.api.flowvi`: `FlowVI.__init__` takes `transfer_params` and the bounds keyword-only, `fit` takes everything after the data pair keyword-only, `sample` takes the ranges keyword-only, and `ks_test` takes everything after the data pair keyword-only. `raddest.api.utils` and `raddest.api.post_processer` gain type annotations and docstrings; all four modules gain module docstrings. The example notebooks are updated to the keyword forms (27 call sites).
- `raddest.api.fitter` public methods take their control options keyword-only: `RegressorBase.sample` (ranges), `ks_test`/`energy_test` (everything after the observed pair), `RegressorML.get_ext_data` (everything after the data quartet), `RegressorML.fit` (`params_0` and optimizer options), `get_posterior_qmc` (`l_bounds`/`u_bounds`/`m`), `PiecewiseLogLinearML.__init__` and `PiecewiseLogLinearStudentTML.__init__` (everything after `n_piece`), and `LogLinearEnergyPT.__init__` (everything after the two flows). Type annotations, class/method docstrings, and a module docstring throughout. The single known F821 (`piecewise_linear_studentT_step_factory`) stays recorded in TODO.md for the next round.
- `raddest.grid_search`: `zoom_in_grid` takes `print_mean_std` (and the loop options, as before) keyword-only, and `gen_new_grid_quantile` mirrors `gen_new_grid_mean_std` with all options keyword-only; type annotations and docstrings throughout, module docstring.
- `raddest.tools`: `get_ext_w_2d_from_edge` takes the two edge arrays keyword-only and `get_ext_w_2d_inter_grid` takes `k` keyword-only; type annotations and docstrings throughout, module docstring.
- `raddest.transfer`: `Transfer.__init__` takes `params_estimator` and `name` keyword-only, and the `_combined` composition helpers take their component callables keyword-only (no external call sites affected). Type annotations and docstrings throughout the module.
- `raddest.x_xerr` functions take only their leading pair positionally and everything else keyword-only: `nf_fit_frame` (data triple + params + optimizer positional, callables and convergence controls keyword-only), `construct_x_xerr_w` / `flow_params_init` (`x_obs`, `x_err` positional), `get_uv_min_max` (bounds positional), and `gen_xobs_trans(_log_rotation/_neglogplus/_neglogplus_xshift/_xshift_neglogplus)` (`flow`, `N` positional; range bounds and transform params keyword-only). Type annotations, docstrings, module docstring.
- `raddest.reg.energy` compare/loss constructors take `y_kdt`/`y_err_kdt` (and for `construct_energy_compare` also `x_obs`/`y_obs`/`permutations`) as keyword-only, matching the `construct_ks_compare` protocol: `construct_energy_compare`, `construct_energy_compare_jax`, `construct_energy_loss`.
- `get_xyw_trans` now takes only the four data arrays (`x_obs`, `y_obs`, `x_err`, `y_err`) positionally; `flowx`, `flowy`, the four range tuples and all remaining options are keyword-only. The integration dispatch (`auto`/`ghq`/`qmc`/`grid`/`qgrid`/`inter_grid`/hybrids) is factored into helpers inside `raddest.reg.map`, and the module gains type annotations and docstrings.
- `shrink_data` takes only (`x_obs`, `x_err`) positionally; `method` and the remaining options are keyword-only. The method dispatch and the `auto` model-selection loop are factored into helpers inside `raddest.shrink`, and the module gains type annotations and docstrings.
- `raddest.reg.ks` sample-generation and comparison functions take only the leading flow pair (or sample-array group) positionally; the four range tuples, model parameters, and options are keyword-only: `gen_xy_obs_prepare`, `loglinear_x2y`, `gen_xy_obs_loglinear(_neglogplus)`, `get_ks_reg_loglinear`, `compare_loglinear`. Type annotations and docstrings throughout.
- Model-parameter functions in `raddest.reg.func` / `raddest.reg.func_jax` now take the input array positionally and every model parameter keyword-only: `saturated_linear`, `saturated_linear_smooth` (+ jax twins), `get_saturated_pars`, and the closures returned by `generate_piecewise_linear(_jax)`. In `raddest.reg.constructor`, the `trainer` loop's optimizer options (`lr`, `n_step`, `atol`, `rtol`, `verbose`), `generator_cond`'s `y_min`/`y_max`, and `construct_ks_compare`'s `x_obs`/`y_obs` become keyword-only; the `compare_constructor` subclass protocol documented in `raddest.api.fitter` is updated to match. All three modules gain type annotations and docstrings.

## 0.10.0 - 2026-10-29

### Added

- JAX grid backend for energy-test fitting: `LogLinearEnergyPT(backend='jax')` evaluates the whole parameter grid in a single jax process with vectorized permutations, instead of one worker process per grid point; the numpy backend remains the default. `raddest.reg.energy.energy_test_jax` exposes the vectorized test on its own.
- Chunked estimation for the energy test (`chunk_size`/`chunk_iter` in `energy_test`, `construct_energy_compare` and `LogLinearEnergyPT`): each energy distance is estimated on random sub-samples, mirroring `pted`'s chunked mode, for large samples.
- `RegressorEnergyPT` grid-search strategy class; `LogLinearEnergyPT` now composes it (no longer a `LogLinearKS` subclass; constructor unchanged).
- Energy-distance fitting methods: grid search on energy-test p-values (`LogLinearEnergyPT`) and gradient descent on the energy distance (`LogLinearEnergyGrad`), with `energy_test` evaluation on `RegressorBase`.
- A Claude Agent Skill (`raddest-regression`) shipped as package data; `python -m raddest.skills` prints the deployment guide.
- Example notebook `example/fit_loglinear_energy.ipynb` covering both energy optimizers and both grid backends.
- GitHub Actions CI running the fast test subset (`pytest -m "not slow"`) on ubuntu/macos/windows; heavy end-to-end fitting tests carry the `slow` marker.
- `--cuda-device` pytest option selecting the test GPU (default `0`, comma lists allowed).

### Changed

- `get_shift` accepts `'histogram'` (the previous `'histgram'` spelling is removed) and raises `ValueError` for unknown methods instead of returning `None`.
- `FlowVI.fit` builds its default `optax.adam(1e-3)` optimizer per call (previously a single optimizer created at import time was silently reused).
- Notebooks are normalized by nbstripout on `git add`: code and outputs are kept, metadata and execution counts are stripped.
- Deal with jax array in `shrink_data`
- Fix import error in example notebook

## 0.9.0 - 2025-06-26

### Added

- `construct_x_xerr_w` support new integration method `inter_grid`
- `get_posterior_mcmc` for `RegressorML`
- `SaturatedLogLinearML`
- `PiecewiseLogLinearML`
  - Allow linear equation constrain
  - Allow Student's T distribution
- Remove 'Regressor' in the name of each pre-defined ML and KS based regressors.

### Changed

- `construct_x_xerr_w` use `int_method` argument to replace previous `qmc`

## 0.8.0 - 2024-11-27

### Added

- `get_xyw_trans` support new integration method
  - `grid`, `qgrid`, `auto`, `ghq+qgrid`, `inter_grid`
- `flow_params_init` support
  - guess x by different method (supported by `shrink.shrink_data`)
  - allow `derive_uv_bounds` 
- `README`:
  - bib Citation
  - Acknowledge to multivariate Gauss Hermite quadrature Implementation

### Changed
- `get_xyw_trans` and `RegressorML.get_ext_data` argument `use_qmc`(bool) replaced by `int_method`(string) 
- Split `FlowVI` to `api/flowvi.py`, but can also be imported from `fitter.py`
- `construct_x_xerr_w` set the weight of points out of given range to 0, as 2d case.
- `Transfer` add `name`



## 0.7.0 - 2024-11-21

### Added

- Now `FlowVI.ks_test` support for test `K` times and use `M` observed data points in each test.
- Add `RegressorBase.ks_test`.
- Add `RegressorLogLinearStudentTML` that use Student's t distribution to model intrinsic scatter.
- `get_xyw_trans` support 
  - `use_qmc`: use QMC to calculate integral.
  - `ignore_y_err_c_y`: if True, calculate `P(x_obs, y_obs, x_err|theta, y_err)` but not `P(x_obs, y_obs, x_err, y_err|theta)`.

## Changed

- `RegressorBase.sample` default `N=1000` to `N=10000`.
- `get_p_flow` support calculate in batch to avoid OOM.

## 0.6.0 - 2024-11-12

### Added
- The scratch of quantile transformation
- Save loss when save `FlowVI`
- Add `get_cov` for `RegressorML` to measure the covariance matrix by using the Hessian as a Gaussian approximation
- Add `get_posterior_qmc` for `RegressorML` to generate the posterior by quasi monte carlo method

### Changed

- `RegressorML.fit` return best-fitted parameters, corresponding uncertainties, and covariance matrix.

## 0.5.0 - 2024-11-07

### Added
- Support force save for `FlowVI`
- Make `Transfer` can be combined
- `post_shift` method for `FlowVI`

### Changed

- Refactor `raddest.transfer`, e.g. `log_rotation` → `LogRotation`
- Set `N=10000` as default for `raddest.api.fitter.ks_test` 

### Fixed

- fix the bug when try to save `FlowVI` with `flow=None` 

## 0.4.1 - 2024-11-06

### Changed

- Rename the package to `raddest`

## 0.4.0 - 2024-11-05

### Changed

- Rename the package to `garden`

## 0.3.0 - 2024-11-01

### Added

- `RegressorLinearML`, `RegressorLinearKS`, and corresponding example.
- Add `utils` in example to make it more clean.
- Add `self.log_likelihood`, `self.log_posterior` methods for `RegressorML`.
- Use `corner` in example to showcase the posterior.


## 0.2.0 - 2024-10-31

### Added

- Recommend to use `python==3.9`, the development version of `rarden`.
- Add gitee mirror for `ndtest`.
- Require `python>=3.9`, due to the requirement of `optax` 
  - https://github.com/google-deepmind/optax/issues/711
- install `matplotlib` and `ipykernel` when create environment in README. 
  - These two packages are required for running the examples. If `matplotlib` is installed at the end (by conda), it may lead to the conflict of the version of `numpy` and `numpy-base`.
- Acknowledge in README
- Add `pyproject.toml`
- Support apply KDE in batch in `RegressorML.fit()` and `get_xyw_trans` to avoid OOM.

### Changed

- Link the installation of `ndtest` to its own README.
- Add `get_xyw_trans` in `map.py` for general correlation. 

## 0.1.0 - 2024-10-28

### Added

- The First Working Version