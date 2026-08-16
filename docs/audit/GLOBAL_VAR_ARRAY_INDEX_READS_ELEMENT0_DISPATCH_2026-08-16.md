<!-- docs:meta
topic_id: repo.docs.audit.global-var-array-index-reads-element0-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.global-var-array-index-reads-element0-dispatch-2026-08-16
-->

# Module-level `var` array reads with a runtime index always return element 0 — dispatch

**Date:** 2026-08-16
**Engine:** lean_single, source-built fixed point (`make build` gen3, md5 `37c1cf8a43ab74143994ec77b9a45e5e`; identical to the refreshed `bin/souc-lean-single-x86_64`)
**Parent:** `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` §"Secondary bugs", item #6; also re-observed as the fourth bug in the SedenTree-16 commit (`e76a09778f`: "a module-level array read with a runtime loop index always returned element 0"). Repro detail first recorded in the trailing comment block of `examples/cayley_dickson_lemon_g2_ffi.sio` at `e1109c4773`.
**Owner:** unassigned
**Status:** OPEN — dispatched, reproduced with a clean discriminator; root cause **not** localised. No `self-hosted/` change made here.

## Why this dispatch

A module-level `var arr: [T; N]` indexed with a runtime-computed index silently reads **element 0 for every index**, purely on the read side — writes to the same shape of global read back correctly through the same indexing form. Any program that precomputes a table at init (constants, generator tables, lookup data) at module scope and consumes it in a loop gets element 0 every time, with no error. Two independent programs (the LEMON G2 generators, the SedenTree-16 basis tables) hit this independently.

## Defect and reproduction

```sounio
var g: [i64; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

fn read_at(a: &[i64; 14], idx: usize) -> i64 with Div { a[idx] }

fn main() -> i32 with IO, Mut, Panic, Div {
    var hits_global: i64 = 0
    var hits_local: i64 = 0
    var hits_param: i64 = 0
    var l: [i64; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    var i: i64 = 0
    while i < 14 {
        if g[i as usize] == i + 1 { hits_global = hits_global + 1 }
        if l[i as usize] == i + 1 { hits_local = hits_local + 1 }
        if read_at(&l, i as usize) == i + 1 { hits_param = hits_param + 1 }
        i = i + 1
    }
    // print the three counters (hits_global / hits_local / hits_param)
    …
}
```

Verbatim output (probe `/tmp/ffi_probe/bug6_globalarr.sio`):

```
global(runtime idx) hits: 1/14
local(runtime idx)  hits: 14/14
local via &[i64;14] hits: 14/14
```

`1/14` = only index 0 matched: `g[i]` yielded element 0's value (1) for every `i`. The identical loop shape over a local array, and over the same local array passed as `&[i64;14]`, are perfect.

## Ruled out

- **The indexing expression itself** (`arr[i as usize]`): correct on locals and on `&[T;N]` parameters in the same program, same loop.
- **"Large globals collide"** (the documented `BASIS_COUNT` scalar explanation, `examples/cayley_dickson_g2_derivation_basis.sio` history): reproduced with a single `[i64;14]` global in an otherwise-trivial file — no other globals, no size pressure.
- **The literal/length defect** (`LEAN_SINGLE_SYSTEM_CMD_LENGTH_SIGSEGV_DISPATCH_2026-08-16.md`): no string literal in the probe; all literals are 1-digit.
- **Read-after-write timing** (parent dispatch item #4): no FFI, no fork, no files in this probe.

## Root-cause locus (hypothesis, not isolated)

Unknown. The signature — global array **reads** with a *runtime* index return element 0, while **writes** to a same-shape global with a runtime index land correctly — points at the emission of the global-array *load* address: the dynamic index appears not to be added to the global's base on the read path (addressing-mode emission for `GLOBAL[i]` lvalue-vs-rvalue), i.e. the rvalue path loads from the base while the lvalue path computes base+index. Where exactly in `lean_single.sio`'s global-array addressing that divergence lives has **not** been located; the write path working is the strongest available clue. Confidence: low-to-medium; this paragraph is a hypothesis for the next dispatch, not a claim.

## Proposed fix locus

Deferred to a future dispatch-gated change once the load-address emission site is identified. Constraint to record now: the fix must preserve the write path's behaviour (it is correct today) and the `&[T;N]` parameter path.

## Acceptance gate (proposed)

Engine-forced test (as `tests/run-pass/ffi_system_exec.sio` does): a module-level `var` table of distinct values, read at runtime indexes 0..N−1, asserting `N/N` matches, plus a write-then-read-back arm to guard the working write path.

## Impact if unaddressed

Any module-scope table consumed with computed indexes under lean_single returns element 0. Both known victims worked around it by moving tables into `main()` locals and threading `&[T;N]` parameters — workable but it defeats the purpose of module-scope constants and is silent when forgotten.

## AI disclosure

Repro and discrimination by AI agent (Claude) under human direction, 2026-08-16, on lean_single gen3 (md5 `37c1cf8a…`). Probe regenerable verbatim from §Defect. No `self-hosted/` sources were modified. GAIDeT-ICMJE 2025.
