Unsigned and 32-bit integer overflow now fail on the Madaros native backend
(SIO-588, ADR-031 R1). Release class: `semantics`.

Unsigned operands (`u8`/`u16`/`u32`/`u64`/`usize`, or a bare literal paired
with one) lower to the new `OpUAdd`/`OpUSub`/`OpUMul` IR variants. These
are guarded on the unsigned condition: `add`/`sub` are followed by `jc`, and
multiplication uses `mul` (CF set iff the high half is nonzero) followed by
`jc`. `u64::MAX + 1`, `0u64 - 1` and `u64::MAX * 2` panic; a result that
fits `u64` but not `i64` (`2^63`) does not.

Dynamic `i32` and `u32` results of `+`, `-`, `*` and unary `-` are
range-checked against 32 bits and panic when they do not fit, instead of
being sign-extended or masked. In-range results are unchanged.

Programs that relied on modular arithmetic write it with `wrapping_add`,
`wrapping_sub` or `wrapping_mul` (SIO-207).

Pinned by `tests/run-pass/u64_unsigned_arith_in_range.sio` and
`tests/run-pass/i32_u32_dynamic_in_range.sio` (no false trap at the bounds)
and by section 4 of `scripts/ci/madaros_int_overflow_gate.sh` (eight
overflow probes, each must fail closed).
