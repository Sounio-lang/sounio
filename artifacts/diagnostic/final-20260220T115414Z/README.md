# Diagnostic Artifact Bundle (Portable)

This bundle captures release-readiness evidence for the 2026-02-20 strict parity run.

Included:
- `TRACEABILITY_MATRIX.md`
- `logs/` (all command outputs used for acceptance)
- `cargo-subprocess/` metadata

Excluded from git portability:
- `cargo-target/`
- `cargo-home/`
- `npm-cache/`

Those cache directories were generated during isolated diagnostics and moved outside the repo to avoid committing machine-local cache payloads.
