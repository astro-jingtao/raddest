# Change Log

## [Unreleased]

### Added

- Recommend to use `python==3.9`, the development version of `rarden`.
- Add gitee mirror for `ndtest`.
- Require `python>=3.9`, due to the requirement of `optax` 
  - https://github.com/google-deepmind/optax/issues/711
- install `matplotlib` and `ipykernel` when create environment in README. 
  - These two packages are required for running the examples. If `matplotlib` is installed at the end (by conda), it may lead to the conflict of the version of `numpy` and `numpy-base`.
- Acknowledge in README

### Changed

- Link the installation of `ndtest` to its own README.

## 0.0.1 - 2024-10-28

### Added

- The First Working Version