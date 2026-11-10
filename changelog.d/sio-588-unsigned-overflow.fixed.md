Unsigned integer overflow is now exact on the Madaros native backend (SIO-588).

Unsigned operands (`u8`/`u16`/`u32`/`u64`/`usize`, or a bare literal paired
with one) lower to the new `OpUAdd`/`OpUSub`/`OpUMul` IR variants, which
emit the same bit-producing `add`/`sub`/`imul` with no overflow guard:
`u64::MAX + 1` wraps to `0`. Dynamic `i32` results sign-extend the low 32
bits and dynamic `u32` results mask them, so a runtime `a + 1` on an `i32`
wraps to `i32::MIN` and no 64-bit value leaks through a 32-bit-typed
operation.

Requires SIO-207 (card 207, PR #3435) for the signed half of the story:
the `i64` trap, the `wrapping_*` builtins, and the `E300` refusal of
out-of-range narrow constants. On pre-207 trees the gate runs the SIO-588
sections only (4-5) and reports the 207 sections (1-3, 6) as loud skips.

Pinned by `scripts/ci/madaros_int_overflow_gate.sh` sections 4-5 and
`tests/run-pass/u64_wrap_arith.sio`, `tests/run-pass/i32_dynamic_wrap.sio`.
