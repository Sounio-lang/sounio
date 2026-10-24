# Known gaps in IEEE 754 special values

**Closed (KL-2 / #2389, 2026-09-11).** PF-aware f64 compares and `print_f64` /
`println` of `inf`/`nan` are fixed on both engines. Witnesses live in
`tests/run-pass/ieee_f64_{nan_compare,print_inf,print_nan}.sio`, pinned by
`scripts/ci/madaros_ieee_f64_special_values_gate.sh`.

This directory has no open numerics gaps at present.
