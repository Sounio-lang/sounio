<!-- docs:meta
topic_id: repo.docs.handoff.c3-simd-autovectorization-codex-dispatch-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.c3-simd-autovectorization-codex-dispatch-2026-07-19
-->

# Dispatch to CODEX-2 — C3: SIMD auto-vectorization of reduction loops (close the pandas gap)

**Date:** 2026-07-19
**Owner:** CODEX-2 (native codegen / lean_single backend)
**Author:** data-science lane (measured on the shipped bigframe benchmark)
**Status:** measured, quantified — the remaining reduction gap vs pandas is vectorization

---

## TL;DR

After the stdlib-level fix (below), `bf_col_sum` over 1,000,000 f64 is **1.44 ms** vs pandas/numpy
**0.55 ms** — a **2.6x** gap. That residual gap is **pure SIMD vectorization**: numpy's sum runs AVX
(4-wide f64) over a contiguous buffer; Sounio emits a scalar loop. Closing it needs the **native
backend to auto-vectorize simple reduction loops** (or expose SIMD intrinsics). No stdlib trick can
recover it — the loop is already an optimal raw-pointer scalar loop.

## What was measured (reproducible: `scripts/bench/run_bench.sh`)

Reduction of 1,000,000 f64 (Sounio min-of-6, build baseline subtracted; numpy/pandas 3.0.3 perf_counter):

| variant | ms | note |
|---|---|---|
| `bf_get(r,c)` per element | 2.09 | bounds-checked accessor |
| raw `read_f64(p,i)`, 1 accumulator | 2.09 | **identical** — the accessor is inlined; call overhead is NOT the bottleneck |
| raw, **8 accumulators (SHIPPED)** | 1.44 | ILP: independent accumulators break the serial add chain (~1.5x) |
| numpy `.sum()` (AVX SIMD) | 0.24–0.55 | vectorized C kernel |

**Correction to an earlier analysis:** the reduction gap was hypothesized to be per-element
bounds-checked `bf_get` calls. That is FALSE — the raw-pointer loop and the `bf_get` loop are
byte-for-byte the same time (2.09 ms), so the accessor is already inlined. The gap is entirely the
scalar-vs-SIMD loop.

## What the stdlib already did (shipped in this PR)

`bf_col_sum` now uses **8 independent accumulators** over the raw column buffer (`read_f64`), summed
pairwise at the end. This is instruction-level parallelism — the CPU issues multiple `addsd` per cycle
because the accumulators are independent — and it is ~1.5x over the single-accumulator loop, correct to
the bit (`49999950000000.0`). This is the ceiling of what stdlib can do without SIMD.

## The ask

Make the native/lean_single backend **auto-vectorize counted loops of the form
`acc[k] += load(ptr + i*8)`** (reductions, elementwise map, filter-compare) into packed SIMD
(`addpd`/`mulpd`/`cmppd` + a horizontal reduce), OR expose SIMD builtins (`f64x4_load`, `f64x4_add`,
`f64x4_hsum`) that stdlib can target directly. Either closes the remaining ~2.6x on reductions and the
~10x on scan/compare ops (`filter`) toward numpy/pandas kernel speed.

- Simplest first target: recognize a `while i < n { acc = acc + read_f64(p, i); i = i + 1 }` loop and
  emit a 4-wide `addpd` body + scalar tail. That single pattern covers `bf_col_sum`/`bf_col_mean` and,
  generalized, the filter compare.
- Alternatively (broader): a `@simd` loop annotation or `f64x4` primitive type.

## Acceptance
- `bf_col_sum` over 1M f64 drops from ~1.44 ms toward ~0.3–0.6 ms (within ~1.5x of numpy).
- Correctness unchanged (bit-exact for integer-valued data; documented FP-associativity note for
  reordered sums, same as the multi-accumulator version already is).

## Scope / non-goals
- **GPU is a separate, later path:** a single 1M-element reduction is memory-bandwidth-bound; a
  CPU→GPU transfer of 8 MB costs more than the sum. GPU wins only for GPU-resident columns with many
  ops — that is a C3b campaign on top of the existing `self-hosted/gpu/` PTX backend, not this one.
- This dispatch is CPU SIMD only: the highest-leverage, most broadly-applicable perf win.

## Pointers
- Shipped optimization + benchmark: `stdlib/data/bigframe.sio` (`bf_col_sum`), `scripts/bench/RESULTS.md`,
  `scripts/bench/run_bench.sh`. Roadmap: `docs/vision/sounio_dataframe_overall_superiority_roadmap_2026-07-18.md` (C3).
