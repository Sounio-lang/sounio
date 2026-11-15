Madaros: a non-literal `i64` argument outside 32 bits that reaches an `i32` or
`u32` parameter now panics at the parameter boundary (ADR-031 R1, the same
fail-closed shape as the SIO-588 arithmetic checks). Before this change,
`fn id(x: i32) -> i32 { x }` returned `9999999999`, a 64-bit value held in
an `i32`. The `int_arg_narrow_nonliteral` case that expected the old masked
value now passes an in-range value. The out-of-range call is pinned as
`i32_arg_narrow` in section 4 of `scripts/ci/madaros_int_overflow_gate.sh`.
