# special::erf Run-Proof — Plan
> Default Madaros (self-contained). No source edits (disjoint). Spec: docs/superpowers/specs/2026-07-14-special-erf-vertical-design.md.
1. Run-proof `tests/stdlib/special/test_erf_stdlib.sio` (erf/erfc/ncdf/nquantile known values + odd/complement/round-trip). `ERF_STDLIB_OK`.
2. Example `examples/special/erf_report.sio` + gate `scripts/special_erf_gate.sh` → `SPECIAL_ERF_GATE_OK`.
3. math-review; governance sync; PR to main (rebase-on-conflict).
