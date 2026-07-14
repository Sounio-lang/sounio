<!-- docs:meta
topic_id: repo.docs.audit.data-io-lane-compiler-blocker-2026-07-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.data-io-lane-compiler-blocker-2026-07-13
-->

# Data & Science I/O lane — blocked by compiler-owned file-I/O builtins

**Date:** 2026-07-13
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine)
**Context:** Implementing the approved Data & Science I/O spec
(`docs/superpowers/specs/2026-07-13-data-science-io-design.md`) under the constraint "no compiler
changes" (compiler owned by CODEX-2). Empirical probing during Phase 0 uncovered that the lane's
foundational primitives are broken at the compiler/runtime level.

This is a **forensic dispatch for CODEX-2**, per CLAUDE.md §8 (do not patch `self-hosted/` ad hoc).

## Summary

The "Data & Science I/O" lane is, mechanically, *read a file's bytes → transform → write bytes out*.
On current Madaros, **both file primitives are broken** and both are compiler-owned builtins:

| Primitive | Status | Evidence |
|---|---|---|
| `read_file(path)` | **SIGSEGV (exit 139)** | probe below; matches known bug in `docs/audit/MADAROS_BUILTIN_EMISSION_2026-06-24.md:41` |
| `write_file(path, &buf, n)` | **writes correct byte-count from WRONG address** (garbage on disk) | probe below; every caller shape fails identically |
| `print(...)` → stdout | **works** | assembled byte content prints correctly |
| `file_size(path)` | works | returns 12 for a 12-byte file |

Because file read/write is broken and the compiler is off-limits, the lane cannot deliver its core
("read/write real-world data files") without a CODEX-2 fix. Only **stdout output works today**.

## Reproductions (all under `export SOUNIO_STDLIB_PATH=$(pwd)/stdlib`)

### read_file → segfault
```sounio
fn main() -> i32 with IO, Mut, Panic, Div {
    let path = "/tmp/w_out.csv"           // a real 12-byte file
    let sz = file_size(path)              // prints 12  (OK)
    let raw = read_file(path)             // <-- SIGSEGV here
    let s = str_from_bytes(raw, sz)
    print("readlen="); print_int(str_len(s)); print("\n")
    return 0
}
```
`./bin/souc compile r.sio -o r.elf && ./r.elf` → `size=12` then **exit 139**.

### write_file → wrong buffer address (garbage bytes)
```sounio
fn main() -> i32 with IO, Mut, Panic, Div {
    var buf: [i8; 64] = [0; 64]
    let body = "x,y\n1,2\n3,4\n"           // 12 bytes
    let n = str_len(body)
    var i: i64 = 0
    while i < n { buf[i as usize] = str_char_at(body, i) as i8; i = i + 1 }
    let rc = write_file(path, &buf, n)    // rc = 12 (correct count) ...
    print("wrote="); print_int(rc); print("\n")
    return 0
}
```
Result: `wrote=12`, and the on-disk file is exactly 12 bytes but contains **garbage**, not `x,y\n1,2\n3,4\n`.
Call-shape matrix (all wrong):
- `write_file(path, buf, n)`  → rc = **-14 (EFAULT)**, no file
- `write_file(path, &buf, n)` → rc = n, **garbage bytes**
- `write_file(path, &b.data, n)` with `struct Buf { data: [u8; 64] }` → rc = n, **garbage bytes**
- `write_file(path, buf, 3u)` → **compile error**

Conclusion: the builtin receives the length correctly but the buffer base pointer is mis-lowered.
Caller-side shapes cannot fix it.

## Secondary findings (older compiler surface removed)

Many `.sio` files were written against a more permissive earlier compiler and **no longer compile** on
Madaros v0.80.0:

- `stdlib/data/frame.sio` — fails on both engines (Madaros `E004`/`E019`; lean_single `typecheck: failed`).
  Uses builtin `String` (struct) + dynamic-array `.push()`/`.len()` + `++`, none of which the current
  compiler accepts. **This was the spec's planned hub; it is dead code.**
- `examples/cognitive_ossm/export_results.sio` — the reference file-I/O example — fails
  (`method calls are not supported for this type`, `visibility preflight failed`).

Working idiom on current Madaros (verified green + runnable):
- **`string` primitive + `str_*` free functions** (`str_len`, `str_char_at`, `str_from_bytes`,
  `str_slice`) — NOT builtin `String`.
- **Fixed-capacity slabs** for growable data: `NativeF64Vec`/`NativeI64Vec` (cap 65536,
  `stdlib/collections/native_vec.sio`), or large fixed arrays (`[f64; 100000]` checks OK). No heap
  (JIT `malloc` broken — `KNOWN_LIMITATIONS.md:68`).
- `.method()` works only on **structs with `impl`**, not on bare array types.
- `print(...)` to stdout works for arbitrarily-assembled byte content.

## Impact on the spec

- **Readers (targets 1 + 3: CSV, Parquet, netCDF, HDF5):** blocked — require `read_file`.
- **File writers (target 4 incl. artifact-to-disk):** blocked — require a working `write_file`.
- **stdout writers (CSV/JSON/table to stdout, redirected with `>`):** **feasible today** — need only
  `print`, `string`, `str_*`, and fixed-capacity in-memory columns.
- The 65536-element (no-heap) ceiling also makes the spec's "full chunked+filtered HDF5 → DataFrame"
  goal incoherent regardless of I/O (real scientific HDF5 is millions of elements).

## Recommended asks for CODEX-2 (blockers, in priority order)

1. **Fix `write_file` buffer-pointer ABI** — length is passed correctly; base pointer is not. Unblocks
   all file output.
2. **Fix `read_file` SIGSEGV** — unblocks every reader (the bulk of the lane).
3. (Lower) A heap-backed growable buffer, or confirmation that ~65536-element fixed slabs are the
   intended ceiling, so the DataFrame hub can be sized honestly.

Until (1)/(2) land, the buildable slice is: an in-memory fixed-capacity DataFrame (string-primitive
idiom) + **writers that emit to stdout**. That is genuinely usable (Unix redirection) and needs no
broken builtin.
