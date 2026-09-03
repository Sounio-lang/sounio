<!-- docs:meta
topic_id: repo.docs.audit.data-io-trilha-b-builtin-bufptr-dispatch-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.data-io-trilha-b-builtin-bufptr-dispatch-2026-07-14
-->

# Data & Science I/O — Trilha B dispatch: `&local_array` mis-lowered into builtin calls

> ⚠️ **CORRECTION (2026-07-17) — this dispatch's Defect 1 is MISDIAGNOSED. Do not fix as written.**
> Re-verified on current `main` (Madaros v0.80.0), tree clean, no `self-hosted/` changes kept:
> - **`str_from_bytes` is NOT broken.** The canonical call shape `str_from_bytes(buf, n)` works
>   (used by 54 `self-hosted/` files incl. `main.sio`; baseline prints `hi`). The repro below used
>   `str_from_bytes(&buf, n)` — the **wrong shape**: `&buf` passes the slot address, `buf` passes the
>   array handle the builtin expects. A codegen "fix" that adds a deref for `&buf` **breaks the
>   canonical `buf` shape** (verified: it made `str_from_bytes(buf,n)` SIGSEGV) and would break
>   Madaros self-compilation. Reverted.
> - **`read_file` is `read_file(path: string) -> string`** (per LSP + all 54 callers), NOT the
>   `(path:*u8, buf:*u8, max_len)->i64` shape the codegen comment claims, and NOT a `[i8;N]` byte
>   array (the repro's shape). `read_file(path)` and `file_size(path)` on a `string` **succeed**
>   (Madaros reads every source file this way).
> - **The one REAL defect — ROOT-CAUSED and FIXED 2026-07-17** (commit on branch
>   `fix/native-builtin-local-array-marshalling`). The native-v2 `emit_builtin_read_file`
>   (`self-hosted/native/codegen_x86_linux.sio`) implemented a **different 3-arg** function
>   `read_file(path,buf,max_len)->i64(bytes_read)` and returned an **integer**, while the contract is
>   1-arg `read_file(path:string)->string` returning a **pointer to a freshly-mmap'd buffer**. So 1-arg
>   callers dereferenced a non-pointer → SIGSEGV on `raw[i]`/`str_len`/`str_char_at`. It worked
>   in-compiler only because Madaros's own read_file is emitted by a DIFFERENT emitter (the lean_single
>   seed's 16 MiB cached-mmap read_file / the bootstrap chain), not by this native-v2 path — so the fix
>   is native-v2-user-programs-only and cannot touch the bootstrap/self-host. **Fix:** rewrote
>   `emit_builtin_read_file` to mirror `bootstrap_v0.sio:15028-15087` byte-for-byte (mmap→open→read→
>   close→return buffer ptr). **Verified:** `read_file`+`str_len`+`str_char_at` byte-exact over a
>   multi-line CSV; `str_from_bytes(buf,n)` intact; 0/40 run-pass compile regressions.
>
> **✅ READER LANDMINE FIXED (2026-07-20)** on `fix/madaros-string-index-native-v2`: `raw[i]` /
> `s[i]` on packed strings (`read_file` / `str_from_bytes` / string lit) was lowered as a GC-handle
> array load (resolve + `[base+idx*8]`) → SIGSEGV. Lower now routes string bases to
> `IrIndexGet` `label_id=3` (raw byte load). Gate:
> `scripts/native_string_index_packed_gate.sh` (needs current-source Madaros). `str_char_at` /
> `str_len` remain valid alternatives.
>
> **Remaining siblings (NOT fixed; separate items):** (1) ~~`write_file`~~ — **FIXED 2026-07-19** on
> `fix/madaros-write-file-handle-abi`: `emit_builtin_write_file` now resolves the native-v2 GC handle
> and unpacks 8-byte boxed slots into a packed buffer (mirrors `str_from_bytes`). Canonical shape is
> `write_file(path, buf, n)` handle-by-value. Gate: `scripts/native_write_file_handle_abi_gate.sh`
> (needs current-source Madaros). (1b) ~~`&buf` residual~~ — **FIXED 2026-07-20** on
> `fix/madaros-d2-ref-buf-builtin`: native-v2 `OpRef` is LEA of the handle slot, so
> `write_file(path, &buf, n)` / `str_from_bytes(&buf, n)` used to pass a pointer into
> `resolve_handle` → SEGV/garbage. Lower now auto-unwraps `&`/`&!` and ref-typed slots at the
> **call site** for those two builtins so both shapes share the proven handle unpack path.
> Do **not** add an unconditional deref inside the builtin body (that previously broke the
> canonical handle shape + self-host). Gate: `scripts/native_d2_ref_buf_builtin_gate.sh`.
> (2) The `self-hosted/native/codegen.sio` old-backend copy of `emit_builtin_read_file`
> (reachable via wide/render drivers, not the default path). (3) ~~The `raw[i]`
> string-index-operator crash~~ — **FIXED 2026-07-20** (see above). (4) `file_size`/
> `read_file` fail on some ABSOLUTE paths but work on relative ones (path-length or cwd resolution —
> unconfirmed).
>
> Net: the "`&local_array` into builtin" root cause below is not the bug. `str_from_bytes`/`file_size`
> work; `read_file` and `write_file` are FIXED (native-v2 user path; shipping prebuilt lags until
> rebuild). Everything below is retained as the original (incorrect) dispatch record.

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine)
**Owner:** CODEX-2 (`self-hosted/` — native codegen, builtin argument marshalling)
**Status:** forensic dispatch (per CLAUDE.md §8 — do not patch `self-hosted/` ad hoc)

## Why this dispatch

Trilha A of the Data & Science I/O vertical shipped a byte-exact CSV writer whose sink
is **stdout only** (commit `49192c173`, gate `scripts/data_io_csv_gate.sh`). It is stdout-only
because the file sink and the buffer→string primitive are broken at the compiler. This dispatch
localises the **single root** that unblocks both, plus a distinct native-emission defect that
blocks module reuse. Fixing Defect 1 lets Sounio write real data files; fixing Defect 2 lets a
program compute (`epistemic::gum`) and format (`data::csv`) in one binary.

The prior blocker note (`docs/audit/DATA_IO_LANE_COMPILER_BLOCKER_2026-07-13.md`) framed this as
"file-I/O builtins broken". That framing is too narrow: the defect is not about files. It is
about **passing the address of a local fixed array to any builtin**.

---

## Defect 1 (PRIMARY) — `&buf` passed to a builtin receives a wrong base pointer

`read_file`, `write_file`, and `str_from_bytes` all fail identically, and all three take a
reference to a local buffer. This is one bug, not three.

| Builtin | Call shape | Symptom |
|---|---|---|
| `str_from_bytes(&buf, n)` | `&[i8; N]` | **SIGSEGV (139)** |
| `read_file(path)` | returns buffer | **SIGSEGV (139)** |
| `write_file(path, &buf, n)` | `&[i8; N]` | writes `n` bytes of **garbage** (wrong source address) |

### Minimal, non-destructive repro (no disk writes)

```sounio
fn main() -> i32 with IO, Mut, Panic, Div {
    var buf: [i8; 16] = [0; 16]
    buf[0 as usize] = 104 as i8   // 'h'
    buf[1 as usize] = 105 as i8   // 'i'
    print(str_from_bytes(&buf, 2))   // <-- SIGSEGV
    print("\n")
    return 0
}
```
`./bin/souc compile r.sio -o r.elf && ./r.elf` → **exit 139**, no output.

### Localisation — the bug is in builtin argument marshalling, NOT `&array` lowering

A **user-defined** function receives the identical `&[i8; 16]` argument **correctly**:

```sounio
fn peek(p: &[i8; 16]) -> i64 {
    return (*p)[0] as i64
}
fn main() -> i32 with IO, Mut, Panic, Div {
    var buf: [i8; 16] = [0; 16]
    buf[0 as usize] = 104 as i8
    print_int(peek(&buf))    // prints 104 — CORRECT
    print("\n")
    return 0
}
```
Output: `104`. So general `&[i8; N]` argument lowering is sound. The wrong base pointer is
introduced **only on the builtin-call path**. The user-function call path is the working
reference to diff against.

### Proposed fix locus

Native codegen for builtin calls (the argument-marshalling that lowers a `&local_array` operand
to a pointer register/stack slot). The user-function call path lowers the same operand correctly;
align the builtin path to it. Suspected: builtin args take the array's *value*/first-element slot
rather than its address, or an extra load is emitted. Compare the two emitted call sequences for
`peek(&buf)` vs `str_from_bytes(&buf, 2)`.

### Acceptance gate (proposed)

Byte-exact round-trip, added to `scripts/data_io_csv_gate.sh` once green:
1. `str_from_bytes(&buf, n)` reproduces an assembled buffer (`"hi"` → `hi`).
2. `write_file(path, &buf, n)` then `read_file(path)` returns identical bytes for a 12-byte CSV.

---

## Defect 2 (SECONDARY) — two-dependency-module programs fail native emission

A program importing **two** modules type-checks and lowers, then fails native emission:

```
Merged IR: 135 functions
Error: Failed to write native binary ... rc=12
error: multimodule native thin-link compilation failed
```

Repro: a `main()` that does `use data::csv::*` **and** `use epistemic::gum::*` and calls one
function from each. One import alone (either module) compiles and runs. This is distinct from the
E137 / print_f64 / visibility-preflight defects already dispatched in
`docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md` — those fail at *type-check*;
this one passes type-check and lowering and fails at the **native thin-link** step (rc=12).

**Impact / workaround:** blocks reusing `data::csv` from a program that also imports the science
module. Trilha A's `examples/epistemic/gum_to_csv.sio` works around it by importing only
`epistemic::gum` and inlining formatting byte-identical to `csv_field_f64`; it collapses to a
single call once this is fixed.

### Proposed fix locus

The multimodule native driver / thin-link path (`module_native_driver`, the
`multimodule native thin-link` emit). rc=12 surfaces after `imported_compile: lower_done` with a
merged 2-dependency IR.

---

## Priority

Defect 1 first — it is the higher-leverage single root (file sink + buffer→string) and has a
minimal, non-destructive repro and a clean working reference (user-fn path) to diff against.
Defect 2 unblocks reuse but has a documented inline workaround.

## AI disclosure

Repros and localisation by AI agent (Claude) under human direction, on Madaros v0.80.0. All repros
are re-runnable with `export SOUNIO_STDLIB_PATH=$(pwd)/stdlib` from the repo root. No `self-hosted/`
sources were modified. GAIDeT-ICMJE 2025.
