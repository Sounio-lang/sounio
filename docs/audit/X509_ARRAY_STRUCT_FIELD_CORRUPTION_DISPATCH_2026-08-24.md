<!-- docs:meta
topic_id: repo.docs.audit.x509-array-struct-field-corruption-dispatch-2026-08-24
authority: repo_only
audience: users
last_validated: 2026-08-24
validated_by: controller (tls-on-madaros branch, X.509 sub-project)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.x509-array-struct-field-corruption-dispatch-2026-08-24
-->

# Forensic dispatch — a struct field named `value` was silently clobbered by its sibling, because a synthetic `Knowledge` layout shadows user field names

> **Title and framing corrected 2026-08-24.** This dispatch was filed as a size
> threshold on array-of-struct writes. It is not one. The trigger is the
> identifier `value`; size, array length and index form are all irrelevant.
> See "Root cause" below. Fix: PR #2126.

**Filed:** 2026-08-24 · **Status:** ROOT-CAUSED, fix in PR #2126 · **Protocol:** CLAUDE.md §8.

Branch: `tls-on-madaros`. Discovered while building the X.509 semantic layer
(`docs/superpowers/plans/2026-08-24-madaros-x509-plan.md`, Tasks 5-6). Blocks
Task 6 (`stdlib/x509/cert.sio`'s `x509_parse_extensions`/
`x509_parse_general_names`) and, transitively, Task 7 (outer `Certificate`
assembly). Full background and the complete "9 techniques tried and
rejected" trail: `.superpowers/sdd/2026-08-24-madaros-x509-plan/task-6-report.md`.
Already-catalogued sibling findings this escalates: Findings 20, 22, 23 in
`docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md`. This dispatch
records Finding 24 from that doc as a standalone, actionable bug report.

## Root cause (2026-08-24) — the synthetic `Knowledge` layout shadows user field names

`ir_register_knowledge_layout` (`self-hosted/ir/lower.sio`) preseeds a synthetic
`Knowledge` struct layout — `value@0`, `variance@1`, `confidence@2` — as **entry
0** of the struct layout table, before any user struct is registered.
`field_idx_from_name_simple` resolves a field *name* to a slot by scanning the
whole table in order and returning the first hit **in any struct**. That path
runs whenever the base is not a typed local, and `arr[i].f` reaches it because
array locals record a length but no element type.

So `extensions[0].value` resolved to Knowledge's slot 0 — **the same slot as
`oid`**. The second store overwrote the first, and both reads went to slot 0 as
well. That is why `value` read back correct, why `value[511]` looked fine, and
why only `oid` appeared corrupt: every byte in the measured output follows from
one wrong slot index. `field_idx_from_name`'s cross-table fallback carries the
identical shadow.

**The trigger is the identifier, not the shape.** Confirmed two ways,
independently:

- renaming **only** the field `value` to `val` in the failing program makes it
  correct — `85 29 170 187 204`;
- renaming a working program's field to `value` breaks it.

Everything else was measured and ruled out: total size, array length,
per-element size, dynamic vs literal index, stack aliasing of the source
buffers, interleaved scalar fields, and the number of writes into the source
buffers. It also fails on the `normal` compile path, so it is **not**
native-v2-specific.

This is why the nine source-level workarounds in Task 6's report all failed —
none of them changed a name. It is also why the scale table below reads the way
it does: `SctEntryL` is clean and `ExtL` is broken because of what their fields
are called, not how big they are.

**Scope.** Any Sounio program with a struct field named `value`, `variance` or
`confidence` accessed through a non-identifier base. Nothing about DER parsing,
byte arrays or array-of-struct is required; the field type is irrelevant.

**Fix.** PR #2126 consults the synthetic layout last: a user layout always wins,
and `Knowledge` answers only if nothing else does, preserving genuine
`Knowledge<T>` access through an untyped base.

## The defect (as originally filed — the symptom is accurate, the size framing is not)

Writing two or more `[u8;N]` fields of the **same array-of-struct element**
silently corrupts one field with bytes belonging to the other, once the
struct and/or array cross some size threshold. The corruption is
deterministic and directional: **whichever field is written chronologically
later in program order ends up holding its own correct bytes; the
earlier-written field's slot ends up holding a copy of the later field's
bytes instead of its own.** Scalar fields (`i32`, `bool`) in the same struct
are unaffected at the sizes tested here (Finding 22's own repro at
`GeneralName` scale did see scalar corruption too — see "Related evidence"
below — so scalars are not universally safe, just unaffected in this
specific repro).

## Repro — decisive, minimal, reproduces at `ExtensionEntry`'s real scale

```sio
//@ run-pass
struct ExtL {
    oid: [u8; 20],
    oid_len: i32,
    critical: bool,
    value: [u8; 512],
    value_len: i32,
}
fn zero_ext() -> ExtL {
    ExtL { oid: [0; 20], oid_len: 0, critical: false, value: [0; 512], value_len: 0 }
}

fn main() with IO {
    var extensions: [ExtL; 32] = [
        zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(),
        zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(),
        zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(),
        zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(), zero_ext(),
    ]
    var count: i32 = 0

    var oid_buf: [u8; 20] = [0; 20]
    oid_buf[0] = 0x55
    oid_buf[1] = 0x1D
    var val_buf: [u8; 512] = [0; 512]
    val_buf[0] = 0xAA
    val_buf[1] = 0xBB
    val_buf[511] = 0xCC

    // Field-by-field, DIRECTLY into the array element -- no intermediate
    // local var, no struct literal, no whole-struct copy anywhere. This is
    // the exact pattern this branch's Finding 22 prescribes as the fix for
    // the (narrower) whole-struct-copy corruption it documents.
    extensions[count as usize].oid = oid_buf
    extensions[count as usize].oid_len = 2
    extensions[count as usize].critical = true
    extensions[count as usize].value = val_buf
    extensions[count as usize].value_len = 512
    count = count + 1

    println(extensions[0].oid[0] as i64)     // expect 85 (0x55) -- GOT 170 (0xAA, val_buf's byte)
    println(extensions[0].oid[1] as i64)     // expect 29 (0x1D) -- GOT 187 (0xBB, val_buf's byte)
    println(extensions[0].oid_len as i64)    // expect 2 -- correct
    println(extensions[0].critical as i64)   // expect 1 -- correct
    println(extensions[0].value[0] as i64)   // expect 170 (0xAA) -- correct
    println(extensions[0].value[1] as i64)   // expect 187 (0xBB) -- correct
    println(extensions[0].value[511] as i64) // expect 204 (0xCC) -- correct
    println(extensions[0].value_len as i64)  // expect 512 -- correct
}
```

Run under default Madaros:

```bash
export SOUNIO_STDLIB_PATH=<repo>/stdlib
./bin/souc run repro.sio
```

Actual output: `170 187 2 1 170 187 204 512`. Expected: `85 29 2 1 170 187 204
512`. `oid` (written first) silently receives `value`'s (written second)
first two bytes; every other field is correct.

## What rules out the naive explanations

- **Not "whole-struct-copy into an array slot"** (the mechanism Finding 22
  already documents) — every write above is field-by-field, directly into
  the array element, with no struct literal and no intermediate local
  variable anywhere in the reachable path.
- **Not total program size / merged function count.** Task 6's implementer
  first characterized this as correlating with total merged IR function
  count (observed failing in the 79-102 function range, passing at 4-63).
  The controller independently tested this specific claim and it does not
  hold: padding a program to **110** merged functions with 105 trivial
  filler functions, wrapped around an already-confirmed-safe write (see
  "Related evidence" below), stayed correct. Adding 25 sequential 3-tuple
  destructures (75 extra live locals) inside the very function performing
  the array write, again around an already-safe write, also stayed
  correct. Total function count and a single function's local-variable
  count are not, in isolation, the trigger.
- **Not the array size alone, nor the struct size alone** — see the
  scale table below; the threshold tracks total struct-instance size
  (roughly array-length × per-element-byte-footprint), not either factor
  independently, but this has not been proven, only observed.

## Related evidence — scale table (all field-by-field-into-array-element writes, no whole-struct copies)

| Struct | Array len | `[u8;N]` fields | Approx bytes/entry | Total array bytes | Result |
|---|---:|---|---:|---:|---|
| `SctEntryL` (mirrors `SctEntry`) | 8 | `[u8;32]` + `[u8;128]` | ~188 | ~1.5 KB | **Correct** (also true of the OLD whole-struct-literal pattern at this scale — see Finding 22) |
| `RdnEntryL` (mirrors `RdnEntry`) | 16 | `[u8;20]` + `[u8;128]` | ~156 | ~2.5 KB | Broken with the OLD whole-struct pattern (Finding 22); not independently re-tested field-by-field at this exact scale, but `RdnEntry`'s real usage (`x509_parse_name`, Task 5, shipped) uses field-by-field and passes its own test — real fixture is tiny (2 entries), so this is a weak data point, not a clean confirmation either way |
| `GeneralNameL` (mirrors `GeneralName`) | 32 | `[u8;253]` + `[u8;20]` (+ nested `X509Name`) | ~281+ | ~9 KB+ | Broken with the OLD whole-struct pattern (Finding 22); Task 6's real usage with the field-by-field fix still fails at runtime (implementer's report) |
| `ExtL` (mirrors `ExtensionEntry`) | 32 | `[u8;20]` + `[u8;512]` | ~537 | ~17 KB | **Broken**, confirmed above, WITH the field-by-field fix applied |

## Follow-up measurement 2026-08-24 — it is NOT a size threshold

The dispatch's framing ("once the struct/array crosses a size threshold") does
not survive a direct test. The repro above was run unchanged, plus two variants
that change exactly one dimension each:

| variant | array len | `value` field | output | verdict |
|---|---|---|---|---|
| repro as filed | 32 | `[u8;512]` | `170 187 2 1 170 187 204 512` | corrupt |
| **8 elements** | **8** | `[u8;512]` | `170 187 2 1 170 187 204 512` | **corrupt** |
| **small field** | 32 | **`[u8;64]`** | `170 187 2 1 170 187 204 64` | **corrupt** |

Dropping the array from 32 to 8 does not cure it. Dropping the field from 512 to
64 bytes does not cure it. `oid` receives `0xAA 0xBB` -- `value`'s bytes -- in all
three. So the defect is the `array[i].field = buffer` pattern with `[u8;N]`
itself, independent of scale.

Both compilers agree: built from source at `main` AND at this branch's
`e6b3cc8d98`, all three variants produce byte-identical wrong output. The branch
differs from main in `self-hosted/native/` (71 commits), so this is not a
main-vs-branch artifact.

Consequence for the scale table below: the `SctEntryL` row marked **Correct** at
8 entries is not evidence of a lower bound. Its rows mix two different write
patterns (whole-struct vs field-by-field), and the field-by-field pattern fails
at 8 entries too. The table measures the patterns, not a threshold.

### One cause ruled out

`struct_deep_copy_instr_headroom` (`lower.sio`) degrades a deep copy to the
#1475 FLAT copy -- which shares the handle -- when the projected instruction cost
would not fit under `IR_MAX_INSTRS`. That is a mechanism which would produce
exactly this symptom (two fields aliasing one buffer) and would look
size-dependent. It is **not** firing here: `SOUNIO_LOWER_LIVE_TRACE=1` emits no
deep-copy trace for any of the three variants.

Method note: measured, not inferred. Each variant was compiled with
`--native-v2-compile` and executed; the numbers above are program output.

## Impact if unaddressed

Blocks `stdlib/x509/cert.sio`'s `x509_parse_extensions` and
`x509_parse_general_names` (Task 6), and by extension Task 7 (outer
`Certificate` assembly, which must build both arrays at real scale plus
more). No source-level workaround was found across 9 distinct techniques
(see Task 6's report for the full list: field reordering, split helper
functions, two-pass writes, exclusive-ref-per-field mutator functions, a
flat-parallel-array redesign, and raw-pointer byte writes to probe field
layout directly — the last one segfaulted). This is likely to affect any
future Sounio program building a moderately large `(array of struct)` where
the struct carries two or more sizeable `[u8;N]`/byte-array-shaped fields —
not specific to X.509 or DER parsing.

## Superseded: the original investigation suggestions

The three starting points filed here — compare codegen offsets at passing vs
failing scale, look for a shared byte-store helper with Findings 3/20, and
binary-search intermediate array lengths for a KB-scale threshold — all point
away from the cause. They are kept only as a record: each presumes the defect
scales with bytes, and it does not. A binary search across array lengths would
have returned "corrupt at every length", which is what happened when it was
finally run.

## Regression gate (once a fix lands)

Landed with PR #2126 as `tests/madaros/source_to_elf/knowledge_field_shadow_exit0.sio`,
registered in that gate's manifest — the gate compiles with Madaros and is run
by CI (`ci.yml:765`). It was proven to discriminate: red on the unpatched tree,
green on the patched one. A `tests/run-pass/` fixture is **not** sufficient on
its own, because CI runs that suite under `souc-stage2`, which never carried the
defect.

Re-running the repro above confirms `170 187 2 1 170 187 204 512` becomes
`85 29 2 1 170 187 204 512`. Then
resume `docs/superpowers/plans/2026-08-24-madaros-x509-plan.md` Task 6 from
its current WIP state (uncommitted in the `tls-on-madaros` worktree:
`stdlib/x509/cert.sio` modified, `stdlib/x509/ext_build.sio` new,
`tests/run-pass/x509_parse_extensions.sio` and
`tests/run-pass/x509_parse_san.sio` new, all currently failing at runtime
for exactly this reason) — the DER-walking logic in that WIP is already
independently confirmed correct and should not need to change, only the
array-of-struct write step.

## AI disclosure

Investigated and written by an AI coding assistant (Claude, subagent-driven
development) as controller for the X.509 sub-project on this branch. Root
cause is NOT characterized — this dispatch documents symptom, decisive
repro, and ruled-out explanations only; it does not propose a compiler fix.
