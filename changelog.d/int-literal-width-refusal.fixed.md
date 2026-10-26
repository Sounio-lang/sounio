Fixed: an integer literal that fits no 64-bit integer type is refused at parse time (E276, ADR-015 row 1) instead of silently wrapping. `99999999999999999999` used to compile to `7766279631452241919` and `0x1FFFFFFFFFFFFFFFF` to `-1`, both with exit code 0.

The refusal is width-based, not sign-based, so it leaves every program that was already correct alone: `-9223372036854775808` (i64 MIN written as a prefix minus over a value that fits) and the `u64` band `(2^63-1, 2^64-1]` — 51 bit patterns in the codegen emitters (`0xBF31A9BDE18F45C7`) and 6 annotated `u64` literals in the corpus — stay accepted. Choosing within that band needs the declared type, which is the checker's half of row 1 and is tracked separately.

Refused in `main.sio`-reachable Madaros parses only: the shared parser is not used by the lean_single seed, and that half needs a seed refresh (own change, own PR).
