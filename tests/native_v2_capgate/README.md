# native-v2 source->ELF capability gate
Run: tests/native_v2_capgate/run.sh <modular-souc.elf>
Baseline 2026-06-04 (feat/native-v2-source-bridge): 17/17 (FULL).
2026-06-05: 20/20 (+wide-int i128: mul, div).
PASS: literals, arith +-*/% + precedence, comparisons, if/else, let, var+mutation,
while, for-in ranges, 1-4 arg calls, recursion, structs, arrays, bool, inline floats.

Wide ints (i128, Zig-style N-limb of u64) — 18,19,20:
  Real >64-bit arithmetic source->ELF. 18: ((2^32*2^32)>>64) as i64 == 1; 19: the mul
  discriminator (3*2^32 * 5*2^32)>>64 == 15; 20: division (15*2^64)/3 >>64 == 5
  (input-dependent high limbs => real N-limb arithmetic, not fixed witnesses). Width
  detected syntactically from cast_type (the by-value Checker SRET-crashes the bridge,
  so no checker sidecar on this path).
  Supported: inline-CAST form (X as i128) for + - * /, and limb-aligned >> (multiple of
  64). Division requires a SINGLE-LIMB divisor (high limbs 0).
  NOT yet: i128 vars (need lowerer ident-type tracking), non-aligned shift, full N/N
  division (Knuth D), mod/compare (these fall through to a clean backend reject, never a
  silent wrong answer).
