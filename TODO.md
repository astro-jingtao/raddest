# TODO

Deferred `ruff` findings (config: `ruff.toml`, introduced with the ruff setup). Auto-fixable findings were fixed in the code at introduction time; the findings below are left unfixed on purpose and should be worked through over time. Regenerate the live list with `ruff check . --output-format concise`.

## Bugs (fix first)

- `raddest/api/fitter.py` — `F821`: `PiecewiseLogLinearStudentTML.__init__` references `piecewise_linear_studentT_step_factory`, which does not exist anywhere in the codebase; instantiating the class raises `NameError`. The class is marked `# TODO: to implement` — either implement the student-T piecewise step factory or remove the class until it exists.
- `raddest/api/flowvi.py` — `B008`: `FlowVI.fit` evaluates `optax.adam(1e-3)` in the default argument, once at import time; a stale shared optimizer state is reused across calls. Move the call into the function body or a module-level singleton.
- `raddest/tools.py` — `F401`: `jax.jit` imported but unused.
- `test/test_hyperplane_projection.py` — `F401`: `pytest` imported but unused.

## Small refactors (one finding = one small change)

- `raddest/api/fitter.py` — `UP035`/`UP006`: `typing.Tuple` deprecated; use `tuple` (requires `from __future__ import annotations` or dropping py3.8-era style; py39 floor keeps `Tuple` working, so this is cosmetic).
- `raddest/api/fitter.py:555` — `RUF059`: unpacked variable `b` never used in `SaturatedLogLinearML` default `log_prior`.
- `raddest/reg/map.py:176` — `RUF059`: unpacked variable `y_err_sample` never used.
- `raddest/transfer.py:260` — `RUF059`: unpacked variable `x_shift` never used.
- `raddest/api/post_processer.py:19` — `RUF005`: list concatenation → iterable unpacking.
- `raddest/api/post_processer.py:30` — `RET503`: missing explicit `return` at end of function returning non-`None`.
- `raddest/api/utils.py:28` — `RUF005`: concatenation → `[*params[:-2], params[-1]]`.
- `raddest/api/flowvi.py:189` — `SIM102`: collapsible nested `if`.
- `raddest/tools.py:267` — `PERF401`: manual loop → list comprehension.
- `raddest/reg/constructor.py:43` — `D417`: `construct_likelihood` docstring missing `p_cond` description.
- `raddest/tools.py:215` — `D417`: `get_p_flow` docstring missing `batch_size`, `flow`, `x`, `x_err` descriptions.
- `test/test_piecewise.py` — `PT006` (3x): pass a tuple as the first `parametrize` argument.
- `test/test_tools.py:49-50` — `RUF005` (2x): concatenation → single list literal.
- 12 findings are marked fixable only via `--unsafe-fixes` (RUF005/RUF059/PT006 class); review individually before applying — never batch-apply unsafe fixes.

## Structural campaigns (bulk work, schedule explicitly)

- Missing type annotations (`ANN001`/`ANN201`/`ANN202`/`ANN204`/`ANN206`): ~1100 findings across the package — a full annotation pass.
- Missing docstrings (`D100`-`D107`, `D102` methods): ~170 findings.
- Wide signatures (`PLR0913`/`PLR0917`, >8 args / >5 positional): ~70 findings across `reg/ks.py`, `x_xerr.py`, `api/fitter.py`, `api/flowvi.py`, `grid_search.py`, `reg/map.py`, `reg/func*.py`, `shrink.py`, `tools.py`, `transfer.py` — most call sites are internal constructors; converting trailing parameters to keyword-only is an API-visible change and should be done module by module.
- Complex functions (`C901`, `PLR0912`, `PLR0915`): `reg/map.py:get_xyw_trans` (complexity 19, 28 branches, 96 statements) and `shrink.py:shrink_data` (complexity 17, 17 branches) — decompose when next touched.
