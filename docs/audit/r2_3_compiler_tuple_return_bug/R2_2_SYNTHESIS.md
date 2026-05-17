<!-- docs:meta
topic_id: repo.docs.audit.r2-3-compiler-tuple-return-bug.r2-2-synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-3-compiler-tuple-return-bug.r2-2-synthesis
-->

# R.2.2 — Cause B Compiler Bug — Synthesis & Halt

**Date:** 2026-05-16
**Status:** ROOT CAUSE NOT FIXED IN THIS SESSION. Workaround documented. Park-Miller (R.2.1 Phase F) shipped at `111a2ea5` already bypasses the buggy path; PBPK28 work unblocked.

## Headline

The Cause B compiler bug — first identified as "PCG state aliasing in stdlib `distributions.sio`" — is not in stdlib code. It is in the **souc compiler's lowering of tuple-return `(Struct, scalar)` access** when the struct return contains five or more fields (4 i64 + 1 f64 in the canonical repro). The bug corrupts the f64 storage on first `println` of `tuple.1`; subsequent reads of `tuple.1` return 0.0 (or a different garbage value depending on stack layout).

R.2.1 §7 reserved compiler-internal bugs for a separate dispatch. The operator's "Fix B then continue" override presumed a tractable surface (line 6943 `UFN_CALL_STRUCT_RESULT` slot allocation). Diagnosis has narrowed past that hypothesis. The bug is **not** simple slot reuse in the caller; static analysis of `nested_tuple.elf` shows no writes to the corrupted byte between the first `println(r1.1)` and the second read. A real fix likely touches tuple-return / scalar-extract lowering or interacts with the function-call ABI in a way that needs gdb watchpoints or compiler-internal instrumentation to localize.

**Recommendation:** open a separate R.2.3 compiler dispatch. Do NOT patch from this session; lean_single fixed-point bootstrap risk exceeds the value of an unproven fix.

## Reproducer

Canonical: `/tmp/sounio_pure_1/r2_2_b/repro/nested_tuple.sio`. Variants `nt3`–`nt13` for narrowing.

Minimal failing pattern:
```sounio
struct S4 { a: i64, b: i64, c: i64, d: i64 }
fn step_inner(s: S4) -> (S4, i64) with Mut, Div, Panic { ... 4 mults + sum ... }
fn step_outer(s: S4) -> (S4, f64) with Mut, Div, Panic {
    let inner = step_inner(s)
    (inner.0, (inner.1 as f64) / 1000000000.0)
}
fn main() with IO, Mut, Div, Panic {
    var rng = make(20260516)         // big-magnitude seed required
    let r1 = step_outer(rng)
    println(r1.1)                    // prints 1362.073482 ✓
    println(r1.1)                    // prints 0.000000 ✗
    println(r1.1)                    // 0.000000 ✗
}
```

## What we learned

| Probe | Setup | Outcome | Inference |
|---|---|---|---|
| `nt4`, `nt5`, `nt6` | seed=10, simple arithmetic | All `r1.1` reads correct | Bug requires big-magnitude arithmetic, multi-field sum, divide-by-1e9 |
| `nt8` | `let f1 = r1.1` BEFORE first println | `f1` stable; later `r1.1` reads = 0 | Scalar copy preserves value; subsequent field reads fail |
| `nt9` | `println(r1.1)` then `rng = r1.0` then `println(r1.1)` | First = 1362.07; second = 0 | Corruption happens at first println, persists |
| `nt10` | Two consecutive `println(r1.1)`, no other code | First = 1362.07; second = 0 | First `println(r1.1)` itself corrupts |
| `nt11` | Three consecutive `println(r1.1)` | 1362.07 / 0 / 0 | First call corrupts, stays corrupted |
| `nt12` | `println(r1.1)` then `let f1 = r1.1` then `println(f1)` | 1362.07 / 0 / 0 | Even bind-after captures 0 — storage already gone |
| `nt13` | `let _unused = r1.1` then `println(r1.1)` | 1362.07 (single line) | Bare read does NOT corrupt; only the first `println(r1.1)` does |

## What is NOT the bug

- Not "slot reuse in `UFN_CALL_STRUCT_RESULT` handler" (line 6943, `native_compile_driver.sio`). Both SRET buffers in main are at distinct non-overlapping offsets (-0x90 for r1, -0x100 for r2). Static asm grep across `nested_tuple.elf` and `nt9.elf`, `nt10.elf` shows exactly **one** write to the saved-pointer slot `-0x98(%rbp_main)` (at the post-call save) and **zero** writes to the f64 storage slot `-0x70(%rbp_main)` between the first println and subsequent reads.
- Not `dst_pcg64_*` source code. The repro contains no PCG code; it triggers on plain `(S4, f64)` tuple-return.
- Not nested-tuple-return per se. The bug fires on a single `step_outer` whose body returns a 5-field tuple where field 0..3 = S4 fields and field 4 = f64 derived from a chained tuple call.

## Workaround (validated)

Bind scalar fields of a struct-tuple result to locals immediately after the call, before any other use:

```sounio
let r1 = step_outer(rng)
let f1 = r1.1               // <-- snapshot BEFORE any println / call
// downstream code uses f1, not r1.1
```

`park_miller.sio` (already committed `111a2ea5`) uses single-field `ParkMiller { state: f64 }` return and never triggers the 5-field path. PBPK28 work proceeds via `pm_*` API.

## Out of scope for R.2.2

- G3 compiler patch — DEFERRED. Needs:
  1. gdb watchpoint on `-0x70(%rbp_main)` in the running `nt10.elf` to identify the actual instruction that writes 0.0.
  2. Audit of tuple `.1` field-extract lowering in `native_compile_driver.sio` — look for places where the f64 scalar field of `(Struct, f64)` is treated specially (cvtsi2sd path, push-pop temporaries).
  3. Verify `lean_single_fixed_point_gate.sh` PASS after any patch — non-negotiable bootstrap safety.
- G4 stdlib PCG64 restoration — DEFERRED until G3 lands.

## Bootstrap status

UNCHANGED. No compiler edit attempted. `bin/souc` md5 unmodified. `lean_single` fixed-point intact. Park-Miller submodule was the only stdlib change (commit `111a2ea5`, additive, no compiler interaction).

## Recommendation to operator

1. Accept R.2.2 closure with park_miller workaround.
2. Open R.2.3 compiler dispatch with this synthesis + `nt10.sio` as the canonical repro. Allocate gdb-capable session.
3. PBPK28 D.7 work proceeds NOW using `stdlib/random/park_miller.sio`. No further blocker for thesis-bound output.
