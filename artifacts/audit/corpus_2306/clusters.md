# Corpus failure signature clusters — issue #2306

scanned=1824 failing_now=133 new_vs_baseline=9 baseline_size=271

| count | signature | examples |
|---|---:|---|
| 8 | `lowering-error record: total=N hard=N` | budget64_test.sio, compress_huffman_fixed.sio, generic_struct_instantiate.sio, generic_struct_nested.sio |
| 1 | `error[E221] in run-pass/math_atan_quadrant_reduction::main at N..N: this math function is bound for typechecking but the` | math_atan_quadrant_reduction.sio |
