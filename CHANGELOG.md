# Change Log

## [unreleased]

### Added

- `construct_x_xerr_w` support new integration method `inter_grid`
- `get_posterior_mcmc` for `RegressorML`
- `RegressorSaturatedLogLinearML`

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