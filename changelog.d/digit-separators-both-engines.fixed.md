Fixed (ADR-015 row 8, both engines): `_` is a digit separator only between two digits, the Madaros rule, now also in `lean_single`. `1_000_000`, `0xFF_FF`, `1_000.000_5` and `1_0e1_0` work on both. `lean_single` used to read `1_000_000` as `1` followed by a name `_000_000` (E200) and refused `1_000.000_5`.

`1__0` and `10_` are refused with a dedicated diagnostic (E284, "digit separator `_` must sit between two digits") on both engines, instead of failing later as a number plus a stray name. A `_` followed by a letter after a number is still a unit suffix (`500_mg`). Probes `int_sep`, `int_sep_double`, `int_sep_trail`, `flt_sep` now agree.

Details: `lean_single` re-scans float literals exactly (correct rounding) and that re-scan skips the separators, so `1_000.000_5` is converted like `1000.0005`. A separated decimal integer is checked against 2^64-1 with an overflow flag instead of by digit count, so the exactly-2^63 case with a binary minus (which the unseparated path refuses) is not distinguished for separated literals.

Pins: `digit_separators_{madaros,lean_single}` (run-pass), `digit_separator_double_*` and `digit_separator_trailing_*` (compile-fail).
