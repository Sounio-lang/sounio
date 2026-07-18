<!-- docs:meta
topic_id: repo.docs.handoff.c2-heap-columnar-dataframe-codex-dispatch-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.c2-heap-columnar-dataframe-codex-dispatch-2026-07-18
-->

## CODEX-2 Work Order: C2 — Heap-Backed Columnar DataFrame Store

**Date:** 2026-07-18
**Campaign:** C2 (roadmap: DataFrame scale-out, "overall superior to pandas")
**Status:** OPEN — dispatch ready, not started
**Depends on:** C1 (DataFrame reliability lane) must be green first; also blocked on default-engine regression noted below
**Related:** C4 (SIO1 / `arrow_bridge`, already shipped) — this campaign should target Arrow-compatible layout so C2 and C4 interoperate zero-copy

---

### TL;DR

The Sounio `DataFrame` (`stdlib/data/dataframe.sio`) is a 100%-by-value, fixed-size struct: `data: [f64; 65536]`, `col_names: [i64; 64]` — a hard **64 columns × 1024 rows** ceiling baked into the type itself, copied wholesale on every operation (`df_filter_gt`, `df_select_cols`, `df_sort_by`, etc.). This cannot scale to millions of rows no matter how the algorithms are tuned — the cap is structural, not algorithmic.

Real heap allocation **already works** in Sounio — genuine libc `calloc`/`realloc`/`free` via `stdlib/mem/box.sio`, empirically verified up to 400MB in a single allocation and 5M-element read/write loops for both i64 and f64 — **but only under the `lean_single` compiler engine**. The default engine (`native_v2`, plain `./bin/souc`) cannot even compile the repo's own existing "passing" heap test (`tests/run-pass/heap_vec_growth.sio` fails with `E137: use of undeclared variable`), and has no f64 heap read/write builtins at all (`READ_F64`/`WRITE_F64` exist only in `lean_single.sio`'s backend emitters, not in `native_compile_driver.sio`).

So C2 is two work items bundled: **(1) fix or route around the default-engine regression that blocks heap I/O**, and **(2) build a new heap-backed columnar f64 store from scratch** — no growable f64 Vec/column type exists anywhere in stdlib today (`heap_map.sio` is the only real precedent, and it's i64-only).

---

### Why It Matters

C2 gates the "scale" and "performance" legs of the roadmap goal to be overall superior to pandas — a DataFrame capped at 1024 rows is a toy, not a data-processing library. This campaign is explicitly listed as depending on **C1** (DataFrame correctness/reliability): don't build a bigger house on a foundation that isn't verified sound. It also intersects **C4** (SIO1 / `arrow_bridge`, already shipped) — if the new heap-backed column format is Arrow-compatible from day one, C2's storage becomes zero-copy interoperable with whatever already consumes `arrow_bridge`, instead of requiring a second conversion layer later.

---

### What Exists Today

**DataFrame (`stdlib/data/dataframe.sio`):**
- `data: [f64; 65536]` (flat, row-major or col-major fixed array — 64 cols × 1024 rows worth of cells)
- `col_names: [i64; 64]`
- Every operation (`df_new`, `df_filter_gt`, `df_select_cols`, `df_sort_by`, …) constructs and returns a **new by-value struct** — no pointers, no heap, anywhere in the file.
- `NativeF64Vec` (`stdlib/collections/native_vec.sio`) is a similarly fixed `[f64; 65536]`-capacity slab, explicitly commented as a legacy workaround from when "JIT malloc is broken" — that comment is now **stale**; the workaround is obsoletable.

**Heap allocation — real and working:**
- `stdlib/mem/box.sio` — genuine `extern "C"` `calloc`/`realloc`/`free` (not mmap, despite stale comments). Documented as active in `docs/compiler/KNOWN_LIMITATIONS.md:227`: "`stdlib/mem/` - heap_alloc/heap_free (malloc/free stubs), arena bump allocator, box/rc/arc wrappers — all active."
- `stdlib/collections/heap_map.sio` — a real growable open-addressing i64→i64 hashmap using `heap_alloc`/`heap_realloc`/`heap_free` with doubling growth. **This is the template to imitate**, not a DataFrame precedent — it doesn't hold f64s or columns.
- `stdlib/mem/arena.sio` — bump allocator on top of raw malloc.
- `stdlib/display/window.sio`, `stdlib/render/renderer3d.sio` — heap-allocate large pixel/depth buffers already, proving the allocator handles real workloads.

**Empirically verified (compiled + run, not just `check`, under `SOUNIO_SOUC_ENGINE=lean_single`):**
| Test | Result |
|---|---|
| 5,000,000 i64 (40MB): alloc → write loop → read loop → sum | correct (12,499,997,500,000), 33ms |
| 5,000,000 f64 (40MB): alloc → write loop → read loop → sum | correct (18,749,996,250,000.0), 62ms |
| Single calloc of 400MB (50,000,000 f64 slots) | succeeded instantly, no compiler-imposed cap found |

Conclusion: **the ceiling on buffer size is just process/OS memory, not anything Sounio-imposed.** The `65536`-cell DataFrame cap is 100% a compile-time choice in the struct definition, not a platform limit.

---

### The Ask

1. **Diagnose and fix the default-engine (`native_v2`) heap I/O regression**, OR formally accept `lean_single`-only for this feature and document it (consistent with the known "512-vreg wall on native_v2 → fall back to `lean_single`" pattern from other lanes). Specifically:
   - `read_i64`/`write_i64`/`read_f64`/`write_f64` fail to compile at all under plain `./bin/souc` — even the repo's existing `tests/run-pass/heap_vec_growth.sio`, nominally a passing test, does not compile by default (`error[E137]: use of undeclared variable`). This is a **regression worth flagging and fixing/triaging separately** from the DataFrame work, since it silently breaks a test that claims to pass.
   - No `V2_BUILTIN_READ_F64`/`WRITE_F64` exist in `native_compile_driver.sio` at all (only the I64 pair) — f64 heap I/O is currently `lean_single`-exclusive by construction, not just by bug.
   - If native_v2 is not fixed in this campaign, the DataFrame's heap-backed path must be gated to compile only under `lean_single`, with the default-engine build either falling back to the existing fixed-slab DataFrame or failing loudly (not silently miscompiling).

2. **Design and implement `HeapF64Vec`** — a growable, contiguous, heap-backed f64 vector — using the `heap_map.sio` growth pattern (`heap_alloc` → `heap_realloc` doubling → `heap_free`) as the template, built on `read_f64`/`write_f64` (or bit-reinterpret through `read_i64`/`write_i64` if targeting native_v2 — note **no `f64_to_bits`/`from_bits` intrinsic currently exists in stdlib**; only a stale comment claims the JIT "can provide" one, so this would need to be written too if the bit-reinterpret path is chosen).

3. **Rewrite `DataFrame` as a columnar heap store**: one `HeapF64Vec`-style buffer per column (`n_rows`, `cap`, `n_cols`, array of column pointers/handles) instead of the flat `[f64; 65536]` + `[i64; 64]` fixed arrays. All row/cell access goes through `read_f64`/`write_f64` instead of static-array indexing. Growth via `heap_realloc`, doubling, same discipline as `heap_map.sio`.

4. **Adopt Apache Arrow's in-memory columnar layout** as the target format for the new column buffers (contiguous typed buffers + validity bitmap per column, standard Arrow physical layout) — not because Arrow's spec is required for correctness, but so the buffers are directly zero-copy-compatible with whatever `arrow_bridge` (C4, already shipped) already produces/consumes. This avoids building a second, incompatible in-memory format that then needs its own conversion layer to interop with C4's output.

---

### Proposed Design

```
struct HeapF64Column {
    ptr: *mut u8       // heap_alloc'd buffer, Arrow-layout f64 array (+ optional validity bitmap)
    len: i64            // rows currently populated
    cap: i64             // rows currently allocated
}

struct HeapDataFrame {
    columns: *mut HeapF64Column   // heap-allocated array of column handles
    col_names: *mut i64            // heap-allocated, growable (or interned-string table)
    n_cols: i64
    n_rows: i64
}
```

- `df_new()` — `heap_alloc` a small initial column-handle array; each column starts with a modest initial capacity (e.g. 1024 rows) and doubles via `heap_realloc` on overflow, exactly mirroring `heap_map_maybe_grow` in `heap_map.sio`.
- `df_push_row(df, values)` — appends one f64 per column, growing any column that hits capacity.
- `df_filter_gt`, `df_select_cols`, `df_sort_by`, etc. — rewritten to operate through `read_f64`/`write_f64` (or Arrow-buffer equivalents) instead of static-array copies; where possible, produce new heap-backed columns rather than copying the whole by-value struct.
- `df_free(df)` — `heap_free` every column buffer + the handle arrays (mirror `box.sio`'s ownership discipline — no leaks, no double-frees; consider an `arc`/`rc` wrapper from `stdlib/mem/` if columns need to be shared/sliced without copying).
- Engine gating: the file (or a `lean_single`-only variant of it) should either `#[cfg]`-style gate on engine, or ship as a second module (`dataframe_heap.sio`) alongside the existing fixed-slab `dataframe.sio`, so C1's already-verified fixed-slab path keeps working under native_v2 while the heap path is proven under `lean_single`.

---

### Acceptance Criteria

- [ ] A 1,000,000-row, multi-column `HeapDataFrame` **builds** (via repeated `df_push_row` or bulk load) without hitting any fixed-size cap.
- [ ] A reduction (`sum`, `filter_gt` + count, or equivalent) over the 1M-row frame **produces a correct result**, cross-checked against a Python/independent oracle (same discipline as the exact-algebra and CDCL lanes — souc has known silent-miscompile risk, never trust `check:OK` alone; compile + run + compare).
- [ ] `HeapF64Vec`/`HeapF64Column` growth is exercised past its initial capacity at least twice (i.e., the test forces at least 2 `heap_realloc` doublings) and still reads back correct values at both old and new offsets.
- [ ] Memory is actually freed (`heap_free` on every allocated column) — verify no leak via repeated alloc/free cycles at scale (e.g., build+free a 1M-row frame 10× in a loop, confirm stable process RSS).
- [ ] The existing fixed-slab `DataFrame` (C1's surface) is **not broken** by this change — either left untouched as a separate type, or the heap-backed replacement passes whatever C1 gate/regression tests already exist for `dataframe.sio`.
- [ ] Column buffer layout is documented against Arrow's spec (buffer + validity bitmap) closely enough that a follow-up can wire it into `arrow_bridge` (C4) without a data copy — doesn't need to be wired in *this* campaign, but the layout choice should not preclude it.
- [ ] Whatever engine this ships under (native_v2 fix vs. `lean_single`-only) is explicitly stated in the PR/commit and in `docs/compiler/KNOWN_LIMITATIONS.md` if it stays `lean_single`-only.

---

### Risks

- **Gated on C1.** If DataFrame correctness/reliability work in C1 changes the struct shape or the public API contract, this heap rewrite could need rebasing. Confirm C1 status before starting the rewrite in earnest.
- **Default-engine heap I/O is currently broken**, not just missing — this is a *regression* (an existing repo test that's nominally green doesn't actually compile). If left unfixed, C2's entire deliverable is `lean_single`-only, which narrows how/where the new DataFrame can be used (consistent with, but adding to, the existing "some lanes are lean_single-only" pattern — see `madaros-codegen` skill).
- **Silent-miscompile wall.** Per repo-wide experience (exact-algebra lane, CDCL lane): souc has a track record of silently miscompiling rather than erroring. Every acceptance-criterion check must be an actual compile+run+compare against an independent oracle, never a `check:OK` alone.
- **No f64 bit-reinterpret intrinsic** exists yet if the native_v2/bit-hack path is chosen instead of fixing native_v2's f64 builtins directly — that's extra unplanned scope hiding behind option 1's "OR" branch.
- **Ownership/lifetime bugs are new territory for DataFrame** — the current by-value struct has no aliasing/double-free surface at all; the heap rewrite introduces exactly that class of bug for the first time. Recommend defensive bounds/index checks matching the CDCL lane's discipline (bound-checked buffer writes, explicit free-once).

---

### Pointers

- `stdlib/data/dataframe.sio` — current fixed-slab DataFrame (to be extended/replaced)
- `stdlib/mem/box.sio` — real heap_alloc/heap_realloc/heap_free (calloc/realloc/free wrappers)
- `stdlib/mem/arena.sio` — bump allocator alternative
- `stdlib/collections/heap_map.sio` — **the growth-pattern template to imitate** (heap_alloc → heap_realloc doubling → heap_free, i64-keyed)
- `stdlib/collections/native_vec.sio` — legacy fixed-slab `NativeF64Vec`; comment claiming "JIT malloc is broken" is now stale, should be updated/removed once this campaign lands
- `docs/compiler/KNOWN_LIMITATIONS.md:227` — documents heap_alloc/heap_free as active
- `self-hosted/compiler/native_compile_driver.sio` — default-engine builtin table; has `READ_I64`/`WRITE_I64` only, no F64 variants
- `self-hosted/compiler/lean_single.sio` — has `emit_read_f64`/`emit_write_f64` in the a32/a64 backends; the only place f64 heap I/O currently works
- `tests/run-pass/heap_vec_growth.sio` — existing "passing" test that actually fails to compile under the default engine (the regression to triage)
- `tests/run-pass/heap_alloc_basic.sio` — basic heap alloc smoke test
- C4 / `arrow_bridge` — target interop surface for Arrow-compatible column layout (already shipped; check its buffer format before finalizing `HeapF64Column`'s layout)
- `madaros-codegen` skill — background on the `lean_single` fallback pattern and known native_v2 fragility, relevant to work item 1

**Scratch reproductions (not in repo, for reference — re-derive, don't assume they persist):** heap scale tests for i64/f64 at 5M elements and a 400MB single-calloc test were run and passed only under `SOUNIO_SOUC_ENGINE=lean_single`; all failed to even compile under the default engine. Re-run equivalent smoke tests as part of this campaign's own verification rather than relying on the prior scratch files.
