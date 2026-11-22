Signed integer overflow fails closed in the Madaros native backend (SIO-207).

`i64` arithmetic used to wrap silently (`9223372036854775807 + 1` printed
`-9223372036854775808`, exit 0), and an `i32` variable happily stored `2^31`
— a value outside its own type. Both are silent wrong values (SIO-8 class).

Runtime signed overflow of `+`, `-`, `*` and unary negation now traps
(exit 1, the builtin-panic shape) via a per-function `jo` guard in
`native::codegen_x86_linux`. Intentional two's-complement wrap remains
available as the explicit `wrapping_add`/`wrapping_sub`/`wrapping_mul`
builtins (bit-identical to the old operators); the stdlib sites that wrap
by design (SplitMix64/xoshiro state evolution, Fibonacci hashing, sqrt
bit-hack seeds) are migrated to them. A compile-time-known constant bound
to a narrow integer annotation (`i8`/`u8`/`i32`/`u32`) that does not fit
is refused at compile time with E300.

Known limits, by design of phase 1: `OpAdd`/`OpSub`/`OpMul` are
signedness-agnostic in the IR, so the trap fires on the signed-overflow
bit pattern (exact unsigned semantics and dynamic narrow-width
enforcement await IR signedness variants); the aarch64 preview backend
does not trap yet. Pinned by `scripts/ci/madaros_int_overflow_gate.sh`
and `tests/compile-fail/i32_literal_out_of_range.sio`.
