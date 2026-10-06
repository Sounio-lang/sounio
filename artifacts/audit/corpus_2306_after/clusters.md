# Corpus failure signature clusters — issue #2306

scanned=1824 failing_now=128 new_vs_baseline=4 baseline_size=271

| count | signature | examples |
|---|---:|---|
| 3 | `lowering-error record: total=N hard=N` | compress_huffman_fixed.sio, println_string_array_field_element.sio, smt_qflia_basic.sio |
| 1 | `error[E221] in run-pass/math_atan_quadrant_reduction::main at N..N: this math function is bound for typechecking but the` | math_atan_quadrant_reduction.sio |
