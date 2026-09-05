# Known gaps in IEEE 754 special values — reproductions

Each file states its EXPECTED outcome in its first line. They are kept out of
`tests/run-pass/` and `tests/compile-fail/` because the defect *is* that they
run. Measured 2026-09-02 on both engines while fixing the `ulp()` precondition
of `examples/chemistry/rep_stagnation.sio`; the ratchet gate is
`scripts/ci/language_gap_ratchet_gate.sh`.

| file | engine | expected | what it shows |
|---|---|---|---|
| `nan_compare_is_not_ieee.sio` | both | runs, prints `1 0 1` | `==`, `!=`, `<` treat NaN as ordered; IEEE says `0 1 0` |
| `print_inf_never_returns.sio` | lean_single | hangs | `println(inf)` does not terminate |
| `print_inf_never_returns.sio` | Madaros | prints `9223372036854775808.000000` | `println(inf)` emits the 2^63 integer-indefinite pattern |
| `print_nan_is_garbage.sio` | lean_single | prints non-numeric bytes | `println(nan)` reads past its buffer |
| `print_nan_is_garbage.sio` | Madaros | prints `-9223372036854775808.000000` | `println(nan)` emits −2^63 |

Consequence for numerical code: a NaN cannot be detected by comparison
(`x != x` is false), and a non-finite value cannot be printed for diagnosis.
Bound every loop whose exit condition is a floating-point comparison, and
check ranges with `x >= lo && x < hi` after the loop rather than testing for
NaN before it — that is what `ulp()` in `rep_stagnation.sio` now does.
