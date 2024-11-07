# Change Log

## [unreleased]

### Added
- The scratch of quantile transformation

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