<!-- docs:meta
topic_id: repo.docs.handoff.c3-group-select-radix-codex-dispatch-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.c3-group-select-radix-codex-dispatch-2026-07-21
-->

# Dispatch to CODEX-2 — a fast f64 SELECT (nth_element) + RADIX SORT builtin closes the last data-lane losses

**Date:** 2026-07-21
**Owner:** CODEX-2 (compiler back-end / codegen; `self-hosted/`)
**Author:** data-science lane (`stdlib/data/bigframe_ops.sio`)
**Status:** the ONLY remaining pandas losses in the ~130-verb grouped-analytics surface are all
order-statistic / comparison-sort bound; a single compiler intrinsic fixes all of them at once.

## TL;DR

The bigframe grouped-analytics surface (~130 verbs) beats pandas on essentially everything that is a
dense scan/accumulate/broadcast — often by 5-20x. The **only** verbs that lose are the ones whose work
is an **order statistic** (median / quantile / rank / winsorize / IQR / MAD-median) or a **full sort**.
Those lose ~1.2-2.9x because Sounio has no hardware-assisted selection/sort — the stdlib does a hand-
rolled 3-way quickselect (`bf_quickselect`) or a bottom-up mergesort, both interpreted-constant-heavy
in `read_f64`/`write_f64` builtins, against pandas' specialised Cython `groupby.median` / `group_rank`
and NumPy's `introselect`/`argsort`.

**The ask:** expose two f64 builtins that lower to tuned native code (SIMD/branch-optimised), so the
stdlib can call them instead of the hand-rolled versions:

- `select_f64(ptr: *mut f64, n: i64, k: i64) -> f64` — in-place k-th order statistic (introselect /
  Floyd-Rivest); the same partition contract as today's `bf_quickselect` (a[<k] <= a[k] <= a[>k]).
- `sort_f64(ptr: *mut f64, n: i64)` — in-place ascending sort (radix for f64 via the sign-flipped bit
  trick, or a tuned pdqsort). Optionally `argsort_f64(vals, idx, n)` carrying a parallel index array
  (needed by rank/stable order).

## Evidence — the exact losing verbs (all reproducible, `lean_single`, 1M rows / 1000 groups)

| verb | Sounio | pandas | ratio | bound by |
|---|---|---|---|---|
| `bf_median_by` / `bf_q1_by` / `bf_q3_by` / `bf_iqr_by` | ~83 ms | ~28 ms | **~2.9x** | per-group quickselect |
| `bf_rank_by` (average) | ~130 ms | ~68 ms | **~1.9x** | per-region mergesort |
| `bf_winsorize_by` (q=.05/.95) | ~1220 ms | ~1010 ms | **~1.2x** | 2 quickselects/group |
| `bf_sort_by` / `bf_sort_radix` (ungrouped) | ~90-163 ms | ~55-66 ms | **~1.5-2.6x** | mergesort/hand radix |
| `bf_rolling_median` / `bf_rolling_quantile` | ~330 ms | ~420 ms (win) | — | already wins (sorted-window) |

Everything else in the surface wins; these are the entire remaining loss set, and they share one root
cause. A `select_f64`/`sort_f64` intrinsic at NumPy-introselect speed would flip median/quantile/iqr/
rank/winsorize/ungrouped-sort from ~1.2-2.9x losses to wins (the per-group *grouping* around them is
already O(n) and faster than pandas' groupby machinery — only the inner selection is slow).

## Why this is the right lever

- **Low blast radius:** one or two math/memory builtins in codegen; no front-end or type changes. The
  stdlib already has the exact call sites (`bf_quickselect` in `bigframe_ops.sio`; the per-region
  mergesort in `bf_rank_by`; `bf_sort_radix`) ready to swap to the intrinsic.
- **Broad payoff:** median/quantile/rank/winsorize/iqr/mad-median + ungrouped sort/argsort + any future
  order-statistic verb, all in one shot. It is the single highest-leverage remaining data-lane dispatch.
- **Complements** the earlier dispatches: `sqrt->sqrtsd` (#1221), `mem_copy` (`mem_copy_builtin_...`),
  C3 SIMD auto-vectorisation (`c3_simd_autovectorization_codex_dispatch_2026-07-19.md`). This is the
  order-statistic sibling of the C3 SIMD ask.

## Acceptance

- `select_f64` / `sort_f64` match a NumPy `partition`/`sort` oracle bit-for-bit across a magnitude sweep
  and duplicate-heavy / already-sorted / reverse inputs.
- After the stdlib swaps to them, `bf_median_by` and `bf_rank_by` at 1M/1000-groups drop below the
  pandas numbers in `scripts/bench/RESULTS.md` (from ~2.9x / ~1.9x to <1.0x).

## Pointers

- Losing call sites: `stdlib/data/bigframe_ops.sio` — `bf_quickselect`, `bf_group_quantile`,
  `bf_rank_by`, `bf_winsorize_by`, `bf_sort_by`, `bf_sort_radix`.
- Benchmark table with the honest loss rows: `scripts/bench/RESULTS.md`.

---

## UPDATE 2026-07-21 — ESCALATION (still unactioned; sharper evidence)

`select_f64`/`sort_f64` are **not yet in `self-hosted/`** (grep is empty), so the data-lane
order-statistic losses remain open. Two developments since this doc was filed make the ask
both **more surgical** and **more urgent**, and confirm it can only be closed in the compiler:

**1. The dense-integer order-statistic family now ALL WINS — which isolates the bottleneck to
the float inner-select alone.** Batches 11–15 added a value-histogram / `bf_qpos_from_hist`
path for integer (and fixed-precision) values:

| verb | ratio vs pandas | how |
|---|---|---|
| `bf_median_dense_by`/`q1`/`q3`/`iqr` (#1362) | **0.41×** | value histogram, no select |
| `bf_percentile_dense_by(q)`/`trimean`/`midhinge`/`decile_range` (#1378) | **0.08×** | interpolated percentiles over the histogram |
| `bf_trimmed_mean`/`winsorized_mean`/`bowley_skew`/`moors_kurt`/`robust_cv` (#1378) | **0.07–0.11×** | rank-window / octile walk over the histogram |

These win **9–14×**. Since the *identical grouping + broadcast machinery* wraps both the dense
(winning) and the float (losing) quantiles, the grouping is empirically O(n) and already
faster than pandas' `groupby`. **The only slow component in the float case is the inner
selection.** That is exactly what `select_f64`/`sort_f64` replaces.

**2. Fresh float number is worse than the original estimate: `bf_median_by` at 1M rows /
1000 groups with ~1000 distinct values per group = ~760 ms vs pandas ~142 ms = ~5.4× LOSS**
(the original 2.9× was measured with fewer distinct values; more distinct → more selection work).

**3. A pure-Sounio radix f64 select does NOT help — confirming the root cause is builtin
overhead, not the algorithm.** I implemented the f64→sortable-u64 bit trick
(`key = bits ^ ((bits >> 63) | INT64_MIN)`) + an LSD radix sort and timed the *inner loop
only* (1000 groups × 1000 elems, no grouping/broadcast): **~697 ms** — no better than the
quickselect it would replace, and still ~5× pandas. The `read_f64`/`write_f64` per-element
builtin calls dominate every pure-Sounio selection variant. **=> a stdlib rewrite cannot close
this; a native intrinsic (real machine code, register-resident partition) is required.**

**Reference for the implementer.** Either contract works; both match today's `bf_quickselect`:
- introselect on raw f64 (`select_f64(ptr, n, k)`), same partition invariant `a[<k] ≤ a[k] ≤ a[>k]`; or
- radix via the sign-flip key above (`sort_f64` / partial-radix `select_f64`), then map back.

**Interim mitigation already shipped stdlib-side (does NOT replace the ask):**
`bf_median_scaled_by` / `bf_q1_scaled_by` / `bf_q3_scaled_by` / `bf_iqr_scaled_by` /
`bf_percentile_scaled_by` win 0.86× for **fixed-precision, narrow-range** floats (round to
int via a scale, reuse the histogram). This covers the measurement beachhead only; the
**general float** median/quantile/rank/winsorize surface still needs `select_f64`/`sort_f64`.

**Priority.** This is now the **single remaining loss class** across ~184 grouped verbs + 2
executable capstones — literally everything else in `stdlib/data/bigframe_ops.sio` beats pandas.
One codegen builtin flips the entire remaining set.
