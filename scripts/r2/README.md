# R2 Parity Spec

This directory defines the R2 command-parity contract for `souc`.

- `parity-spec.toml`: declarative parity and cultural-fidelity cases.
- `parity_spec_lint.py`: static schema validator used by gates/CI.

Run locally:

```bash
python3 scripts/r2/parity_spec_lint.py
```

The spec is intentionally minimal and can be expanded case-by-case as
R2 parity hardening proceeds.
