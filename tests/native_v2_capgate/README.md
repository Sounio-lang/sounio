# native-v2 source->ELF capability gate
Run: tests/native_v2_capgate/run.sh <modular-souc.elf>
Baseline 2026-06-04 (feat/native-v2-source-bridge): 17/17 (FULL).
2026-06-05: 20/20 (+wide-int i128: mul, div).
2026-06-05: 24/24 (+wide-int i128: mod, compare ==/!=/</<=/>/>=).
2026-06-05: 26/26 (+wide-int i128: funnel >> via x86 shrd; 26 exercises the cross-limb merge).
PASS: literals, arith +-*/% + precedence, comparisons, if/else, let, var+mutation,
while, for-in ranges, 1-4 arg calls, recursion, structs, arrays, bool, inline floats.

Wide ints (i128, Zig-style N-limb of u64) — 18..24:
  Real >64-bit arithmetic source->ELF. 18: ((2^32*2^32)>>64) as i64 == 1; 19: the mul
  discriminator (3*2^32 * 5*2^32)>>64 == 15; 20: division (15*2^64)/3 >>64 == 5;
  21: (15*2^64)%7 == 2; 22: 2^64 < 2^65 == 1; 23: 15*2^64 != 16*2^64 == 1;
  24: 2^65 > 2^64 == 1 (each input-dependent and high-limb-deciding => real N-limb
  arithmetic, not fixed witnesses; a low-limb-only impl would give 0). Width detected
  syntactically from cast_type (the by-value Checker SRET-crashes the bridge, so no
  checker sidecar on this path).
  Supported: inline-CAST form (X as i128) for + - * / %, >> by ANY literal bit count
  (limb-aligned uses a limb-copy; otherwise a funnel shift via x86 shrd — test 25:
  (2^40*2^60)>>96 == 16; test 26: (2^64+2^63)>>63 == 3 exercises the cross-limb merge),
  and unsigned compare ==/!=/</<=/>/>= between two equal-width
  wide operands. Division/remainder require a SINGLE-LIMB divisor (a `X as i128` cast,
  high limbs 0).
  HONEST REJECTS (no ELF, never a silent wrong answer — via IrWideReject): a multi-limb
  divisor, a NON-LITERAL shift, a mixed-width compare, and any other wide op (shl, bitwise).
  These reach the lowerer's reject sentinel, which has no codegen arm, so the backend
  returns false and emits no binary.
  NOT yet: i128 vars (need lowerer ident-type tracking) [Increment C]; full N/N division
  (Knuth Algorithm D) + multi-limb mod [Increment D].
  KNOWN BOUNDARY (pre-existing): mixed-width + - * (one operand narrow) reads a garbage
  high limb; keep both operands two-wide casts. Compare already rejects this case.
