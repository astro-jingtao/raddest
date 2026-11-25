# TODO

Deferred `ruff` findings (config: `ruff.toml`, introduced with the ruff setup). The 2026-10-29 lint campaign resolved all 1425 findings from the original triage except the F821 dead-code pair below (user decision: defer to a next round — implement or remove then; do not silence with noqa). Regenerate the live list with `ruff check . --output-format concise`. Current total: 1 finding (the ruff-visible half of the pair below).

## Bugs (fix first)

- `raddest/api/fitter.py` — `F821` (the repo's last ruff finding): `PiecewiseLogLinearStudentTML.__init__` references `piecewise_linear_studentT_step_factory`, which does not exist anywhere in the codebase; instantiating the class raises `NameError`. The class is marked `# TODO: to implement` — either implement the student-T piecewise step factory or remove the class until it exists. Its method bodies are excluded from coverage for exactly this reason.
- `raddest/interpolator.py` — `MyPchipInterpolator.__init__` calls `get_inverse_coefficients_breakpoints`, which does not exist anywhere; the class cannot be instantiated, so `transform`/`inverse_transform`/`diff`/`from_params` are unreachable (ruff-invisible because the call goes through `self.`). Either implement the inverse-coefficients construction (nontrivial for a PPoly) or drop the module; its bodies are excluded from coverage for this reason.

## Structural campaigns (bulk work, schedule explicitly)

- (none — the annotation, docstring, and wide-signature campaigns were completed on 2026-10-29; PLR0917 was resolved by keyword-only conversions, and PLR0913 was deselected outright on 2026-11-24 with its 27 waivers removed.)

## Naming decisions (pending)

Names that no longer match what the thing is; none is executed yet — each needs a rename decision (including hard-rename vs deprecated-alias policy) before the next release.

- `RegressorKS` (`raddest/api/fitter.py`): despite the name it is the general distribution-distance grid-search regressor — the comparison statistic is decided by `compare_constructor` (its own docstring says subclasses replace it to fit against a different distribution distance; `RegressorEnergyPT` already subclasses it). Candidate: `RegressorGrid`, naming the shared zoom-in-grid strategy alongside `RegressorML` / `RegressorEnergyPT` / `RegressorEnergyGrad`. Blast radius: `fitter.py` (definition, bases of `LogLinearKS` / `LinearKS` / `RegressorEnergyPT`, two docstrings), `test/test_fitter_ml.py` (4 uses + `TestRegressorKSInit`), `test/test_energy_regressors.py` (import + one isinstance); not referenced in README, notebooks, or the shipped skill.
- ~~`raddest/api/post_processer.py`: misspelling of `post_processor`~~ — done 2026-11-25: renamed to `raddest.api.post_processor`, test file renamed with it.
- ~~`raddest/reg/ks.py`: module name binds to KS but hosts statistic-agnostic machinery~~ — done 2026-11-25: the scatter/generation machinery moved to `raddest.reg.mock` and the parser/print helpers to `raddest.grid_search`; `ks.py` keeps only the KS comparators.
- Checked and found accurate: `ks_test` / `energy_test` methods (each is the named statistic), `RegressorEnergyPT` / `LogLinearEnergyPT` (permutation test), `RegressorEnergyGrad`, the `RegressorML` family (maximum likelihood), `_PositiveSigmaGridMixin`, `FlowVI`, the `Transfer` subclasses. `GaussHermiteQuadrature.py` uses a CamelCase filename (PEP 8 wants lowercase) — cosmetic; fold into any rename round.
