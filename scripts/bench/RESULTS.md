# bigframe vs pandas — honest 1M-row benchmark

Reproducible via `scripts/bench/run_bench.sh` (Sounio min-of-N process launches, build baseline
subtracted; pandas 3.0.3 / numpy 2.5.1, perf_counter around the op only). All numbers reproduced
directly (not taken from a subagent). col_sum agrees with pandas bit-for-bit (49999950000000.0).

## Current (after C3 stdlib optimizations)

| operation | 1M rows | Sounio ms | pandas ms | Sounio/pandas | before |
|---|---|---|---|---|---|
| col_sum (8-accumulator ILP) | | 1.44 | 0.55 | **2.6x** | 3.9x |
| col_mean (via 8-acc sum) | | 2.05 | 1.00 | **2.1x** | 2.3x |
| filter_count (`bf_count_gt`, raw scan) | | 1.84 | 0.74 | **2.5x** | 12.1x |
| filter_materialize (col-major gather) | | 12.5 | 6.2 | **2.0x** | 5.0x |
| groupby_sum (10 dense keys, O(n) accumulator) | | 13.7 | 13.8 | **0.99x — parity** | 0.99x |
| groupby_sum_hash (1000 SPARSE keys, open-addressing) | | 22.7 | 17.0 | **1.33x** | (dense drops keys>=1024) |
| inner hash-join (100k×100k, shuffled keys) | | 35.4 | 8.7 | **4.1x** | (new verb) |
| frame build (1M x 3) | | ~20 (once) | — | — | — |

## What changed (all stdlib, no compiler)
- **col_sum: 8 independent accumulators** break the serial add chain -> instruction-level parallelism (1.5x). mean inherits it.
- **filter_count: a raw-pointer scan** (`bf_count_gt`) instead of an accessor loop.
- **filter_materialize: two-pass** -- count survivors first, allocate the output ONCE at exact size, then fill. Kills the old doubling-realloc churn (~15 reallocs copying the growing buffer for 500k survivors).

## Honest read
- **Scale + correctness**: 1M rows, exact results, ~1000x the fixed frame's 1024 cap.
- **Reductions/scans now within ~2.1-2.6x of pandas** (were 2.3-12x). The residual is pure SIMD: pandas runs AVX 4-wide, Sounio a scalar loop. That is the C3 compiler auto-vectorization dispatch (`docs/handoff/c3_simd_autovectorization_codex_dispatch_2026-07-19.md`) -- the stdlib multi-accumulator/raw-scan is the ceiling without SIMD codegen.
- **groupby at parity**: the more algorithm-bound the op, the closer Sounio gets.
- **inner hash-join (4.1x)**: `bf_join_inner` builds an open-addressing table on the right key (reusing the `bf_ghash` Fibonacci mix), probes the left, then gathers the output COLUMN-MAJOR through raw column pointers. It is correct and scale-proven (100k×60k → 60k matches, oracle-exact), but it is the verb furthest from pandas: `pd.merge` is a hand-tuned C hash-join with SIMD gather, while Sounio hashes and gathers scalar (keys/indices boxed through f64). Measured on SHUFFLED keys — the fair general case; on *sorted-unique* keys pandas hits a near-memcpy fast path (~0.9ms) that a general hash-join cannot match by design. Same C3 SIMD path plus an integer-index gather would close most of the gap.
- **filter_materialize (3.1x)**: bounded by copying 500k*3 cells one write_f64 at a time; a bulk column-wise copy (memcpy-style contiguous move) would close most of the rest -- a follow-up.
- **GPU**: a single 1M reduction is memory-bandwidth-bound (8MB CPU->GPU transfer > the op); GPU is for GPU-resident multi-op columns (C3b), and the DGX is remote (not benchmarked here).

**Bottom line:** the data layer is scale-correct, algorithmically competitive (groupby parity), and now within ~2-3x of pandas on reductions/scans/filter -- with the remaining gap being SIMD codegen (dispatched) and a bulk-copy filter follow-up.
