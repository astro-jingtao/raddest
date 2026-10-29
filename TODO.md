# TODO

Deferred `ruff` findings (config: `ruff.toml`, introduced with the ruff setup). The 2026-10-29 lint campaign resolved all 1425 findings from the original triage except the F821 dead-code pair below (user decision: defer to a next round — implement or remove then; do not silence with noqa). Regenerate the live list with `ruff check . --output-format concise`. Current total: 1 finding (the ruff-visible half of the pair below).

## Bugs (fix first)

- `raddest/api/fitter.py` — `F821` (the repo's last ruff finding): `PiecewiseLogLinearStudentTML.__init__` references `piecewise_linear_studentT_step_factory`, which does not exist anywhere in the codebase; instantiating the class raises `NameError`. The class is marked `# TODO: to implement` — either implement the student-T piecewise step factory or remove the class until it exists. Its method bodies are excluded from coverage for exactly this reason.
- `raddest/interpolator.py` — `MyPchipInterpolator.__init__` calls `get_inverse_coefficients_breakpoints`, which does not exist anywhere; the class cannot be instantiated, so `transform`/`inverse_transform`/`diff`/`from_params` are unreachable (ruff-invisible because the call goes through `self.`). Either implement the inverse-coefficients construction (nontrivial for a PPoly) or drop the module; its bodies are excluded from coverage for this reason.

## Structural campaigns (bulk work, schedule explicitly)

- (none — the annotation, docstring, and wide-signature campaigns were completed on 2026-10-29; PLR0913 sites carry per-site `# noqa` waivers with reasons, PLR0917 was resolved by keyword-only conversions.)
