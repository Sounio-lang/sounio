<!-- docs:meta
topic_id: repo.docs.audit.module-const-string-read-file-zero-bytes-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.module-const-string-read-file-zero-bytes-dispatch-2026-08-16
-->

# Module-level `const string` bindings carry a broken runtime representation — dispatch

**Date:** 2026-08-16
**Engine:** lean_single, source-built fixed point (`make build` gen3, md5 `37c1cf8a43ab74143994ec77b9a45e5e`; identical to the refreshed `bin/souc-lean-single-x86_64`)
**Parent:** `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` §"Secondary bugs", item #5, originally framed as "`read_file()` with a module-level `const string` argument returns 0 bytes". This dispatch **widens the defect**: it is not a `read_file` problem — the const string's runtime value itself is corrupt.
**Owner:** unassigned
**Status:** OPEN — dispatched, reproduced, three distinct symptoms discriminated; root cause **not** localised. No `self-hosted/` change made here.

## Why this dispatch

A module-level `const S: string = "…"` binding compiles and type-checks, but the value the program actually receives is garbage. Downstream code cannot tell: `read_file(S)` fails "cleanly" (0 bytes), and this dispatch shows the same binding **prints wrong content** and **crashes `str_len`**. The LEMON pipeline lost days to this (every "bridge CSV length = 0 bytes" was this defect, not the fork-timing hypothesis its author was chasing at the time — the parent dispatch item #5 already concedes this).

## Defect and reproduction

### Symptom 1 — `read_file(const)` returns 0 bytes (the original report)

`/tmp/sounio_f5_probe.txt` pre-created with 16 bytes (`0123456789ABCDEF`):

```sounio
const P: string = "/tmp/sounio_f5_probe.txt"
fn read_via_param(p: string) -> i64 with IO, Div { str_len(read_file(p)) as i64 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let via_const  = str_len(read_file(P)) as i64
    let lit = "/tmp/sounio_f5_probe.txt"
    let via_let    = str_len(read_file(lit)) as i64
    let via_inline = str_len(read_file("/tmp/sounio_f5_probe.txt")) as i64
    let via_param  = read_via_param("/tmp/sounio_f5_probe.txt")
    … print all four …
}
```

Verbatim output (probe `/tmp/ffi_probe/bug5_constpath.sio`):

```
const path bytes:  0
let path bytes:    16
inline lit bytes:  16
fn param bytes:    16
```

The binding kind is the only variable: `const` at module scope → 0; `let` local, inline literal, and `string` parameter → 16.

### Symptom 2 (new) — `str_len` of the same const SIGSEGVs

```sounio
const S: string = "ab"
fn main() -> i32 with IO, Mut, Panic, Div {
    let a = str_len(S) as i64   // ← process dies here
    …
}
```

rc=139 (SIGSEGV), no output. Probe `/tmp/ffi_probe/bug5_str_isol.sio`.

### Symptom 3 (new) — `print` of the same const emits wrong content

```sounio
const S: string = "ab"
fn main() … { print(S) }
```

Prints `0` (not `ab`), rc=0. Probe `/tmp/ffi_probe/bug5_str_isol2.sio`.

### Control — `const i64` is unaffected

`const K: i64 = 42` reads `42` both in `main` and via a helper fn (probe `/tmp/ffi_probe/bug5x_consti64.sio`). The defect is specific to the **string** representation of module-level consts.

## Ruled out

- **`read_file`**: symptoms 2 and 3 contain no `read_file` at all.
- **Fork/write visibility** (parent item #4): no FFI, no child, no writes — the file predates the process.
- **The literal-length defect** (`LEAN_SINGLE_SYSTEM_CMD_LENGTH_SIGSEGV_DISPATCH_2026-08-16.md`): the consts here are 2 and 27 bytes, far below the 127-byte boundary; and inline literals of the same content work.
- **`const` evaluation order / helper-fn visibility**: `read_k()` (helper reading a const) works for i64; the string variant fails in `main` itself, so it is not a cross-fn visibility problem.

## Root-cause locus (hypothesis, not isolated)

The three symptoms together say the const's runtime representation (pointer/length pair, or however lean_single materialises a `string` global) is initialised with garbage: `print` dereferences something printable-but-wrong ("0"), `str_len` dereferences far enough to fault, `read_file` fails early and returns 0. Where module-level `const string` initialisation is emitted in `lean_single.sio` has **not** been located. Relation to the documented **D6** defect (`docs/compiler/KNOWN_LIMITATIONS.md:100` — module-level `const` read from a non-`main` fn miscompiles, i64, **Madaros** engine) is an open question, not an identification: different engine, different type, and this defect manifests in `main` itself. The discriminating experiments that would settle it (const `i64` under lean_single — done, clean; const `string` under Madaros from `main` — not run here, Madaros multi-decl parsing blocks the file shape) leave it unresolved.

## Proposed fix locus

Deferred to a future dispatch-gated change once the const-string initialisation site is identified. The fix must be verified against all three symptoms (read_file bytes, str_len, print content) plus the i64 control.

## Acceptance gate (proposed)

Engine-forced test: a module-level `const S: string = "ab"` asserted equal to `"ab"` (str_len == 2 and both chars), a `const P` path read_file asserting real bytes, and a `const K: i64 == 42` control, all in one program.

## Impact if unaddressed

Every module-scope `const string` (paths, names, format templates, CSV headers) is a silent wrong-value or crash under lean_single. The standing workaround — inline the literal as a local `let` — is what `examples/cayley_dickson_lemon_g2_ffi.sio` now does.

## AI disclosure

Repros, symptom discrimination, and the i64 control by AI agent (Claude) under human direction, 2026-08-16, on lean_single gen3 (md5 `37c1cf8a…`). Probes regenerable verbatim from §Defect. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
