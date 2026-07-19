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
| left  outer join (100k×100k, 50% match) | | 26.1 | 15.3 | **1.71x** | (new verb) |
| full  outer join (100k×100k → 150k rows) | | 27.3 | 29.8 | **0.92x — Sounio wins** | (new verb) |
| sort_by (1M rows, shuffled, stable) | | 162.9 | 62.2 (75.7 stable) | **2.62x (2.15x vs stable)** | (new verb) |
| sort_radix (1M continuous doubles) | | 163.2 | 65.8 | **2.48x (≈ mergesort)** | (new verb) |
| sort_radix (1M bounded keys, range 1e6) | | 90.7 | 60.3 | **1.50x** | radix best case |
| sort_radix (1M low-cardinality, 1000 keys) | | 75.8 | 55.5 | **1.37x** | radix best case |
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
- **sort_by (2.62x / 2.15x vs stable)**: `bf_sort_by` argsorts a (key, original-row) index list with a bottom-up **stable** mergesort, then gathers each output column column-major through the sorted indices. Two optimizations took it from ~4x to 2.6x: **paired key+index buffers** (compare sequential `key[a] <= key[b]` instead of a random gather `keys[idx[a]]` that cache-thrashes the 8MB key array) and **ping-pong buffers** (no O(n) copy-back on each of the ~20 merge passes). Correct for all non-NaN f64 (direct `<=`), stable (equal keys keep input order — verified).
- **sort_radix — a bounded-key fast path, NOT a universal win (honest read)**: `bf_sort_radix` is an LSD radix sort over the f64 bit-keys (11-bit digits, ≤6 passes) with the IEEE sign-bit order transform, made adaptive by a **diff-mask that skips digit-passes whose bits are uniform across all keys**. It is differentially verified byte-for-byte against the stable mergesort (mixed-sign *and* ties-heavy data), so it is provably correct including tie order. **The headline (continuous doubles) is 2.48x pandas — essentially the same as the mergesort (2.62x)**: on full-entropy data every pass runs, so radix's O(n) buys nothing over the O(n log n) mergesort here, and both are bottlenecked by the same unavoidable random gather (scatter writes + the final column gather by sorted index; numpy's introsort+take is optimized C). Radix's real win is when the KEY COLUMN HAS BOUNDED RANGE OR LOW CARDINALITY — IDs, categories, quantized/binned measurements, timestamps in a window — where byte-skip cuts it to 1-3 passes: **1.50x pandas on a 1e6-range key, 1.37x on 1000 distinct keys, and ~1.8x faster than the mergesort**. So it is a fast path to reach for on bounded keys, and equivalent to `bf_sort_by` otherwise. (Earlier I nearly headlined the 1.4x bounded-integer number as the general result — that was radix's best case, not the measurement-data case; the continuous-double number is the honest one.) A true universal win needs SIMD codegen (C3) for the gather, or a GPU-resident sort.
- **left / full outer join (1.71x / 0.92x)**: `bf_join_left` / `bf_join_outer` share one engine with `bf_join_inner` (build-on-right, probe-left), keeping every left row (unmatched right cells take a caller `fill`, e.g. `make_nan()` for pandas NaN) and, for the full outer, appending right rows whose key no left row matched (left cells filled, join key coalesced). **The full outer join actually BEATS pandas (0.92x)**: `pd.merge(how="outer")` factorizes both sides, unions the keys, and does NaN alignment (~30ms here), while Sounio's outer is just the left pass plus a cheap append of the unmatched right rows (~27ms). The left join (1.71x) is much closer to pandas than the inner (4.1x) because the tighter data (50% match) and the all-left-rows-in-order emit avoid the inner's per-match index churn. Numbers are data-shape dependent (match ratio, key distribution) but reproduced locally on shuffled keys.
- **filter_materialize (3.1x)**: bounded by copying 500k*3 cells one write_f64 at a time; a bulk column-wise copy (memcpy-style contiguous move) would close most of the rest -- a follow-up.
- **GPU**: a single 1M reduction is memory-bandwidth-bound (8MB CPU->GPU transfer > the op); GPU is for GPU-resident multi-op columns (C3b), and the DGX is remote (not benchmarked here).

**Bottom line:** the data layer is scale-correct, algorithmically competitive (groupby parity), and now within ~2-3x of pandas on reductions/scans/filter -- with the remaining gap being SIMD codegen (dispatched) and a bulk-copy filter follow-up.
