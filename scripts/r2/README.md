# R2 Parity Spec

This directory defines the R2 command-parity contract for `souc`.

- `parity-spec.toml`: declarative parity and cultural-fidelity cases.
- `parity_spec_lint.py`: static schema validator used by gates/CI.
- `parity_spec_exec.py`: runnable executor for `[[case]]` contracts.
- `golden/`: committed deterministic command-output fixtures for contract review.

Run locally:

```bash
python3 scripts/r2/parity_spec_lint.py
python3 scripts/r2/parity_spec_exec.py
python3 scripts/r2/parity_spec_exec.py --self-test
```

The matrix is intentionally fast-running and focused on parity smoke:
- Runtime smoke: `run` (2 fixtures), `check` (1 fixture).
- Top-level contracts: `souc --help`, `souc --version`.
- Core command help: `compile`, `build`, `check`, `run`, `repl`, `bench`,
  `fmt`, `lint`, `analyze`, `test`, `info`, `diagnostics`.
- Major surface help from `souc --help`: `target`, `sysroot`, `clean`,
  `watch`, `serve`, plus `diagnostics check`.
- Informational command: `souc info` with stable substring assertions.

## Golden Contract Policy

Use `scripts/r2/golden/` for deterministic outputs that should be reviewed
as committed fixtures (for example command help text, version text, and
stable smoke stdout/stderr/exit behavior).

Do not enforce exact golden text for volatile output that naturally changes
between builds (for example `souc info` commit/build-date fields). Keep
those as substring contracts in `parity-spec.toml`.

`scripts/r2/parity_spec_exec.py` always writes fresh run artifacts under
`artifacts/r2/`; update committed goldens intentionally when behavior
changes by design.
