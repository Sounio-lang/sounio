<!-- docs:meta
topic_id: repo.docs.handoff.mem-copy-builtin-codex-dispatch-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.mem-copy-builtin-codex-dispatch-2026-07-19
-->

# Dispatch to CODEX-2 — expose a bulk-copy builtin (mem_copy_f64 / memcpy) for the lean_single/native backend

**Date:** 2026-07-19
**Owner:** CODEX-2 (compiler builtins / lean_single + native codegen)
**Author:** data-science lane (measured on the bigframe filter)
**Status:** measured, quantified, low blast radius — a builtin wire-up, not new machinery

---

## TL;DR

Bulk memory copies in stdlib (contiguous DataFrame slices, buffer growth, binary serialization) are
currently a scalar `read_f64`/`write_f64` element loop that runs at **~3.25 GB/s** — ~6x below memory
bandwidth. libc `memcpy` (SIMD / `rep movsb`) does the same move at **~20 GB/s**. Expose a **bulk-copy
builtin** (`mem_copy_f64(dst: *mut f64, src: *mut f64, n: i64)`, or a generic byte `mem_copy(dst, src,
bytes)`) that lowers to `memcpy`, and every bulk move in the data layer speeds up ~6x. The machinery
already exists in the compiler; this is wiring, not invention.

## Measurement (reproducible)

Copying a contiguous 12 MB block (500,000 rows x 3 f64 columns) via the current `read_f64`/`write_f64`
loop: **3.69 ms/copy (~3.25 GB/s)**. At memory bandwidth (~20 GB/s) that is ~0.6 ms. In the shipped
`bf_filter_gt` contiguous fast-path, the copy is ~3.7 ms of a 9.1 ms filter; a `memcpy` builtin would
drop the filter toward ~6 ms, closing the pandas gap on that op from ~2.1x toward ~1.4x. The same win
applies to every bulk move (below).

## The machinery already exists (cite)

- **The builtin pattern to copy:** `read_f64`/`write_f64` are already backend builtins emitted directly
  by the lean_single codegen — `self-hosted/compiler/lean_single.sio:8855` (`emit_read_f64`),
  `:8870` (`emit_write_f64`), dispatched by name at `:13133` (`src_match(ns, ne-ns, "read_f64")`).
  A `mem_copy` builtin slots in exactly the same way: recognize the name, emit a call to libc `memcpy`
  (or an inlined `rep movsb`).
- **memcpy is already used in other backends:** `self-hosted/llvm/builder.sio:267` (`llvm_build_memcpy`
  -> `LLVMBuildMemCpy`, `ffi.sio:254`), and the GPU runtime `self-hosted/gpu/runtime/cuda.sio:472`
  (`cuda_memcpy_h2d`/`d2h`). So the concept and the FFI are present; only the lean_single/native
  user-facing builtin is missing.
- Heap works only under lean_single today (see the C2 dispatch), so the builtin must live in the
  lean_single/native builtin set alongside `read_f64`/`write_f64`.

## The ask

Add a builtin, callable from Sounio under lean_single/native:

```
mem_copy_f64(dst: *mut f64, src: *mut f64, n: i64)      // copy n f64 (n*8 bytes)
// or generic:
mem_copy(dst: *mut u8, src: *mut u8, bytes: i64)         // lowers to memcpy(dst, src, bytes)
```

lowering to libc `memcpy` (non-overlapping; add `mem_move`/`memmove` if overlap support is wanted).
Mirror it in the LLVM backend via the existing `llvm_build_memcpy`. No new dependency — `memcpy` is
libc, already linked (the heap uses libc `calloc`/`realloc`/`free`).

## Where stdlib uses it immediately

- `stdlib/data/bigframe_ops.sio` `bf_filter_gt` contiguous fast-path — one `mem_copy_f64` per column
  instead of a per-element loop.
- `stdlib/data/bigframe.sio` `bf_push_row` / `bf_reserve` growth — bulk-copy on realloc paths.
- `stdlib/data/arrow_bridge.sio` SIO1 serialization — bulk-copy each column buffer into the output.
- Any future columnar copy / concat / take.

## Acceptance

- `mem_copy_f64` over 1.5M f64 sustains >= 10 GB/s (vs ~3.25 GB/s for the current loop).
- `bf_filter_gt` on 500k contiguous survivors drops from ~9.1 ms toward ~6 ms.
- Bit-identical output (it is a byte copy).

## Scope / non-goals

- Non-overlapping `memcpy` semantics are enough for the columnar use (src and dst are distinct heap
  buffers). Overlap-safe `memmove` is optional.
- This is orthogonal to the C3 SIMD auto-vectorization dispatch
  (`c3_simd_autovectorization_codex_dispatch_2026-07-19.md`): that one speeds up *reductions/scans*
  (compute), this one speeds up *bulk moves* (copy). Both are needed to reach kernel-speed.

## Pointers
- Measurement + usage: `stdlib/data/bigframe_ops.sio` (`bf_filter_gt` contiguous path),
  `scripts/bench/RESULTS.md`. Builtin template: `self-hosted/compiler/lean_single.sio:8855-8886`.
  Existing memcpy: `self-hosted/llvm/builder.sio:267`, `self-hosted/gpu/runtime/cuda.sio:472`.
