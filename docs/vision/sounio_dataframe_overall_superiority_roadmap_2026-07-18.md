<!-- docs:meta
topic_id: repo.docs.vision.sounio-dataframe-overall-superiority-roadmap-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.vision.sounio-dataframe-overall-superiority-roadmap-2026-07-18
-->

# Roadmap: Sounio DataFrame — the plan to be *overall* superior to pandas

**Date:** 2026-07-18
**Status:** strategic roadmap / call to arms — compiler-dependent campaigns are dispatchable to CODEX-2
**Author:** data-science lane

---

## 0. The thesis (and why it is not naïve)

**A new dataframe can overtake pandas outright.** This is not aspiration — it already happened: **Polars**, started from zero, passed pandas on performance in ~4 years and is taking real mindshare. So "you can't beat an incumbent dataframe" is *empirically false*.

Sounio holds two cards that **neither pandas nor Polars** holds:

1. **Correctness-native.** Uncertainty (GUM), dimensional units, exact & certified arithmetic, machine-checked invariants (Lean), provenance — all first-class, shipped, run-proofed (22 modules on `main`). No mainstream dataframe has *any* of this.
2. **A GPU-native compilation path.** Sounio already emits PTX and has run Sounio→CUBIN on Blackwell (`self-hosted/gpu/` PTX emitters, `stdlib/gpu/` kernels, the DGX Spark bring-up). pandas is CPU-only; cuDF is GPU but has no correctness layer.

**The uncontested quadrant is `fast × correct × GPU-native`.** Nobody is there. That is the target — not out-pandas-ing pandas on pandas' turf, but building the dataframe nobody has built.

The "structural gaps" I once called walls are **engineering campaigns** with known techniques. Below is how each falls.

---

## 1. The five campaigns

### C1 — Reliability: kill the silent-miscompile wall  *(CODEX-2, foundational, blocks everything)*
Today `souc` can emit **wrong values with a clean exit** past a shape-sensitive op-count (why `data::bigrat` is oracle-gated). This is the single non-negotiable: a dataframe whose engine may silently miscompute cannot be trusted in production, in any niche. **This is a bug, not a destiny.** Fix it and "it compiled" becomes a correctness signal again.
- Deliverable: root-cause the capacity/aliasing wall in codegen; make the bignat/bigrat selftests pass at 10× current op-count without oracle babysitting.
- Until then: every exact/big path stays oracle-gated (already in place).

### C2 — Scale: heap-backed columnar store  *(CODEX-2 heap + data lane)*
The `64×1024` cap (65 536 f64 slots, by-value struct) is the hard scale wall. Break it with a **columnar, heap-backed** store — one contiguous buffer per column, grown on demand — instead of a fixed by-value matrix.
- Prereq: real heap allocation from the compiler (JIT malloc is currently broken — `KNOWN_LIMITATIONS`). This is the gating dependency; dispatch it.
- Target: 64 → thousands of columns, 1024 → millions of rows.
- Adopt the **Apache Arrow** memory layout as the column format (see C4).

### C3 — Performance: vectorized + GPU kernels  *(data lane + existing GPU backend)*
Once columns are contiguous (C2), reductions/filters/joins become vectorizable. Sounio **already has the GPU backend** — point it at dataframe kernels.
- CPU: SIMD-friendly loops over Arrow buffers; the loop form already sidesteps the codegen wall (proven with `bigrat_col_sum`).
- GPU: reuse `self-hosted/gpu/` PTX emission for `groupby`/`join`/`elementwise` over large columns → a path pandas structurally does not have.

### C4 — Ecosystem: Arrow interop as the bridge  *(data lane)*
Don't rebuild the PyData universe — **plug into it**. Adopt Arrow as the on-disk/in-memory format and you inherit zero-copy interop with pandas, Polars, DuckDB, Parquet, and the whole Arrow ecosystem. Add FFI for the rest. This is exactly how Polars bootstrapped ecosystem reach without 20 years of network effects.
- Deliverable: read/write Arrow IPC + Parquet; a thin FFI boundary.

### C5 — The correctness moat: deepen and lift into the type system  *(data lane + CODEX-2)*
The 22 correctness modules are the differentiator. Two moves widen the moat:
- **Lift units + uncertainty into `Knowledge<f64 in unit>` at compile time** (already dispatched, #1113) — the machinery mostly exists in `check_binary_units`. This makes correctness a *type-level guarantee*, not a library call.
- Integrate the rigor spectrum (statistical / worst-case / certified / exact) as first-class column dtypes, so a column *is* uncertain-or-exact-or-interval and every operation propagates correctly by construction.

---

## 2. Sequencing

1. **C1 (reliability)** — gates trust; nothing production-grade without it. Start now.
2. **C2 (heap columnar)** — gates scale + performance; needs compiler heap. Dispatch in parallel with C1.
3. **C4 (Arrow)** can begin against the current store and pays off immediately for interop.
4. **C3 (vectorized/GPU)** lands on top of C2.
5. **C5 (type-level correctness)** runs continuously; the library layer is already shipping.

Milestone ladder:
- **M1** — reliability fix + heap allocation available → `souc` trustworthy, columns growable.
- **M2** — Arrow read/write + a million-row columnar frame → scale parity for the niche and beyond.
- **M3** — GPU groupby/join on large frames → performance pandas cannot match on CPU.
- **M4** — `Knowledge<T>`-typed correctness columns → the moat becomes a language guarantee.

At **M3+M4**, the claim is no longer "better than pandas for a niche" — it is *fast, correct, and GPU-native in one tool*, which is strictly a superset of what pandas offers.

---

## 3. Honest risks (ambition ≠ denial)

- **Everything gates on C1.** If the compiler's silent-miscompile wall isn't fixed, the ceiling stays low. It is the highest-leverage item, and it is CODEX-2's.
- **Heap is a real prerequisite** for C2/C3; without it, scale stays capped. Also CODEX-2's.
- **Ecosystem parity is years, not months** — but Arrow interop (C4) buys most of the value early without rebuilding it.
- The correctness moat is **already real and shipping**; it is the safest bet and should keep advancing regardless.

None of these are laws of physics. They are targets. Polars proved the incumbent is beatable; Sounio's GPU + correctness cards are a stronger hand than Polars started with. The plan is to play it.

---

## 4. Immediate next actions
- Dispatch **C1** (reliability root-cause) and **C2 heap** to CODEX-2 as concrete compiler work orders.
- Begin **C4** (Arrow read/write) on the data lane against the current store.
- Keep shipping **C5** correctness modules and push the `Knowledge<T>` binding (#1113).

The scoreboard is not "match pandas." It is "build the dataframe that is right *and* fast *and* GPU-native — the one nobody has built." That is winnable, and the first move is fixing a compiler bug, not accepting a wall.
