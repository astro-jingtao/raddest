# raddest

reGression for Astronomical data with Realistic Distributions, Errors and Non-linearity.

## Changelog

- Always maintain `CHANGELOG.md` in sync with the changes being made: every commit that adds, changes, fixes, or removes user-facing behavior (API, CLI, examples, packaging) adds its entry to the `[unreleased]` section in the same commit — never batch changelog updates for later.
- Release flow: versions are cut by tagging (`v<version>` annotated tags), not by version-bump commits alone. A release commit moves the accumulated `[unreleased]` entries into a `## <version> - <date>` section, bumps `version` in `setup.py`, and the tag points at that commit.

## Development

- Environment: conda env `raddest` (`conda run -n raddest ...`); `ruff` and `pytest` live there, not on PATH.
- Test suite: `conda run -n raddest python -m pytest -q` from the repo root (full run ~10 min). Tests select the GPU with `--cuda-device=<id>` (default `0`, comma list allowed) — pick a free one with `nvidia-smi`.
- Notebooks: `*.ipynb` are filtered by nbstripout on `git add` (keep code + outputs, strip metadata/execution counts). One-time setup after cloning: `nbstripout --install` and `git config filter.nbstripout.clean "$(which python) -m nbstripout --keep-output"`.
