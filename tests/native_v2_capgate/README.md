# native-v2 source->ELF capability gate
Run: tests/native_v2_capgate/run.sh <modular-souc.elf>
Baseline 2026-06-04 (feat/native-v2-source-bridge): 16/17 (all but bare f64-var casts).
PASS: literals, arith +-*/% + precedence, comparisons, if/else, let, var+mutation,
while, for-in ranges, 1-4 arg calls, recursion, structs, arrays, bool, inline floats.
GAP: bare f64-VAR casts / two-f64-var ops (needs variable-f64-type tracking).
