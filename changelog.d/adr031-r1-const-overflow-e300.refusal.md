An `i64` constant expression whose value does not fit `i64`
(`9223372036854775807 + 1`, `MIN - 1`, `MIN * 2`, `MIN / -1`) is refused at
compile time with E300 (ADR-031 R1), instead of being folded with wrapping
arithmetic or left to the run-time overflow trap. Pinned by
`tests/compile-fail/i64_const_expr_overflow.sio`.
