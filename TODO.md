# TODO

Deferred `ruff` findings (config: `ruff.toml`, introduced with the ruff setup). Auto-fixable findings were fixed in the code at introduction time; the findings below are left unfixed on purpose and should be worked through over time. Regenerate the live list with `ruff check . --output-format concise`. Current total: 1425 findings (the counts below drift as code changes; the live list is authoritative).

The Bug-level and small-refactor findings from the original triage were fixed on 2026-10-28 (B008 shared optimizer state, unused imports, RUF005/RUF059/RET503/SIM102/PERF401/PT006/UP035/UP006/D417 items, plus the `'histgram'`→`'histogram'` spelling fix in `get_shift`).

## Bugs (fix first)

- `raddest/api/fitter.py` — `F821`: `PiecewiseLogLinearStudentTML.__init__` references `piecewise_linear_studentT_step_factory`, which does not exist anywhere in the codebase; instantiating the class raises `NameError`. The class is marked `# TODO: to implement` — either implement the student-T piecewise step factory or remove the class until it exists. Its method bodies are excluded from coverage for exactly this reason.
- `raddest/interpolator.py` — `MyPchipInterpolator.__init__` calls `get_inverse_coefficients_breakpoints`, which does not exist anywhere; the class cannot be instantiated, so `transform`/`inverse_transform`/`diff`/`from_params` are unreachable. Either implement the inverse-coefficients construction (nontrivial for a PPoly) or drop the module; its bodies are excluded from coverage for this reason.

## Structural campaigns (bulk work, schedule explicitly)

- Missing type annotations (`ANN001`/`ANN201`/`ANN202`/`ANN204`/`ANN205`/`ANN206`): ~1171 findings across the package — a full annotation pass.
- Missing docstrings (`D100`-`D104`, `D107`, `D102` methods): ~170 findings.
- Wide signatures (`PLR0913`/`PLR0917`): ~78 findings across `reg/ks.py`, `x_xerr.py`, `api/fitter.py`, `api/flowvi.py`, `grid_search.py`, `reg/map.py`, `reg/func*.py`, `shrink.py`, `tools.py`, `transfer.py`, `reg/energy.py` — most call sites are internal constructors; converting trailing parameters to keyword-only is an API-visible change and should be done module by module.
- Complex functions (`C901`, `PLR0912`, `PLR0915`): `reg/map.py:get_xyw_trans` (28 branches) and `shrink.py:shrink_data` (17 branches) — decompose when next touched; plus one `PLR0915` long function.
