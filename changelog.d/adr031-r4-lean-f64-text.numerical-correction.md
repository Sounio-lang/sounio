`lean_single` prints a finite `f64` as its exact decimal with six fraction
digits at every magnitude, the digits of C++23 `printf("%.6f")` (ADR-031
R4), as Madaros does. It used to switch to scientific notation at
|x| >= 1e15 (`1.000000e19`) and for values that round to zero
(`1.000000e-9`); `1.0e19` now prints `10000000000000000000.000000` and
`1.0e-9` prints `0.000000`. Magnitudes at or above 2^63 are expanded
exactly in base-1e9 limbs. Release class: `numerical-correction`.

Pinned by `tests/run-pass/print_f64_fixed_decimal_lean_single.sio`, whose
expected lines are C++ `printf("%.6f")` output for the same doubles.
