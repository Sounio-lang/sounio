# bigframe vs pandas — honest 1M-row benchmark (2026-07-18)

Reproducible via `scripts/bench/run_bench.sh` (compiles the 4 Sounio `bench/sio/*.sio` under lean_single,
times min-of-4 with the build baseline subtracted; pandas timed with `perf_counter` around the op only).
Machine snapshot, pandas 3.0.3 / numpy 2.5.1:

| operation | 1M rows | Sounio ms | pandas ms | Sounio/pandas |
|---|---|---|---|---|
| col_sum | | 2.09 | 0.54 | 3.9x slower |
| filter_count | | 8.05 | 0.66 | 12.1x slower |
| **groupby_sum (10 keys)** | | **13.67** | **13.83** | **0.99x — parity** |
| frame build (1M x 3) | | 23 (once) | — | — |

Correctness cross-check: Sounio and pandas `col_sum` agree exactly (49999950000000.0, integer-exact in f64).

## Honest read
- **Scale works.** A heap-backed `bigframe` holds 1,000,000 rows and runs filter/groupby correctly — ~1000x past the fixed frame's 1024-row cap.
- **Simple reductions trail (3.9x–12x).** `col_sum`/`filter_count` are the pure "scalar bounds-checked loop vs. vectorized C/SIMD kernel" gap. Each Sounio `bf_get` is a bounds-checked call into the heap buffer; pandas runs SIMD over a contiguous NumPy array. This is exactly the gap roadmap **C3 (vectorization/GPU)** exists to close — until then, expect ~4x on reductions and ~10x on scan-heavy ops.
- **groupby is at parity (0.99x).** The O(n) direct-index accumulator matches pandas' hash groupby at 1M rows — the more algorithm-bound the op, the closer Sounio gets, because the per-element scalar penalty is amortized against real work.
- **Not measured / caveats:** groupby uses a dense small-integer key domain (≤1024 keys); a sparse/large key domain needs hashing. `filter` materialization pays doubling-realloc churn (start the output capacity near the expected size to fix). No SIMD, no parallelism yet.

**Bottom line:** today Sounio's data layer is *scale-correct and algorithmically competitive* (groupby parity) but *raw-throughput behind* pandas on vectorizable reductions. The honest superiority claim remains **correctness + now scale**; closing the reduction gap is the C3 vectorization campaign.
