Fixed: an integer literal that fits no 64-bit integer type is refused at parse time (E276, ADR-015 row 1) instead of silently wrapping, on both engines. `99999999999999999999` used to compile to `7766279631452241919` and `0x1FFFFFFFFFFFFFFFF` to `-1`, both with exit code 0.

The refusal is width-based, not sign-based, so it leaves every program that was already correct alone: `-9223372036854775808` (i64 MIN written as a prefix minus over a value that fits — 27 sites in stdlib and the f128/f256 oracle tables) and the `u64` band `(2^63-1, 2^64-1]` (51 bit patterns in the codegen emitters, `0xBF31A9BDE18F45C7`, plus annotated `u64` literals) stay accepted. Choosing within that band needs the declared type, which is the checker's half of row 1 and is tracked separately.

Two `0 - 9223372036854775808` sites in `lean_single.sio` — binary minus, so genuinely refused — moved to the `0x8000000000000000` bit-pattern spelling the codegen emitters already use; same value, and the seed can now compile its own source.

Pins: `tests/compile-fail/int_literal_too_wide_{madaros,lean_single}.sio`, `tests/compile-fail/int_hex_over_64bit_{madaros,lean_single}.sio` (twin files per engine, the `indep_dsep_forged_literal_*` convention), and `tests/run-pass/int_literal_min_spelling_still_accepted.sio` so the preserved spelling cannot regress.
