# native-v2 source->ELF capability gate
Run: tests/native_v2_capgate/run.sh <modular-souc.elf>
Baseline 2026-06-04 (feat/native-v2-source-bridge): 17/17 (FULL).
PASS: literals, arith +-*/% + precedence, comparisons, if/else, let, var+mutation,
while, for-in ranges, 1-4 arg calls, recursion, structs, arrays, bool, inline floats.
