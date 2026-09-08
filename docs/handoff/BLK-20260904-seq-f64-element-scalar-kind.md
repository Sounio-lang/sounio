<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260904-seq-f64-element-scalar-kind
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260904-seq-f64-element-scalar-kind
-->

# Blocker: BLK-20260904-seq-f64-element-scalar-kind

```text
Blocker-ID: BLK-20260904-seq-f64-element-scalar-kind
Status: closed
Severity: B1
Class: compiler-semantics
Owner: lang-limits-20260903
Lane: lang-limits-20260903
Worktree: /workspace/.wt/lang-limits
Branch: fix/language-limitations
Files-Owned: self-hosted/ir/lower.sio
Repro: tests/run-pass/seq_f64_element_scalar_kind.sio
Observed: reading a Seq<f64> element produced 2^63 instead of the value, and
  println refused the same read with "unresolved scalar kind".
Expected: a Seq<f64> element is a float in every spelling.
Acceptance-Gate: scripts/ci/madaros_seq_f64_scalar_kind_gate.sh
Evidence-Level: E4
Evidence: scripts/ci/madaros_seq_f64_scalar_kind_gate.sh ->
  MADAROS_SEQ_F64_SCALAR_KIND_GATE_OK, run 2026-09-06 against two engines built
  from source: 39d72a37a9 (md5 54eccf3d), the merge of PR #2413, and main
  6b2b7ff9b0 (md5 593ab4c9), two merges later. Remote: GitHub CI run
  34030342462 on main 6b2b7ff9b0, job "Madaros Current-Source f64 Lowering",
  step "Seq<f64> element reads classify as float" -> success.
Fallback-Path: bind the read to a typed local before use.
LLM-Offload: not-required
Residual: `&Seq<T>` PARAMETER is NOT fixed -- separate path, wrong value in one
  shape and SIGSEGV in another, on any element type. Moved to
  docs/compiler/KNOWN_LIMITATIONS.md and ratcheted; see Closure.
Next-Action: none for this blocker. The parameter path is tracked separately.
```

## Observed

Measured on the committed `bin/madaros-linux-x86_64`, Madaros v0.80.0. A
wrong-code path, not a rejection: the programs type-checked clean.

| shape | before | after |
|---|---|---|
| `println(w.get(0))` | refused, "unresolved scalar kind" | `0.250000` |
| accumulating loop, `i64` counter | `9203105838531608576.0` | `1.000000` |
| accumulating loop, `usize` counter | `9203105838531608576.0` | `1.000000` |
| `w.get(0) + w.get(1)` unrolled | `9223372036854775808.0` (2^63) | `1.000000` |
| `Seq<f64>` struct field via `&Self` | `9203105838531608576.0` | `1.000000` |
| **`&Seq<f64>` parameter** | wrong value / SIGSEGV | **unchanged** |

`Seq<i64>` was correct throughout the same shapes, which is what identified this
as float classification rather than anything about Seq.

## Root cause

Seq access is an intrinsic: `seq.get(i)` lowers to `__sounio_seq_get`, which has
no fn symbol. Two routines classify an expression's type, and both resolve a
method call by looking up its mangled name:

- `expr_result_scalar_kind_ref` — consulted by `println` dispatch
- `expr_result_is_float_ref` — consulted by arithmetic

The lookup misses for an intrinsic, and both fall through to their integer
default. The `a[i]` spelling has an answer in both (`ExprIndex` arms); the
`seq.get(i)` spelling had one in neither. `.len()` was already special-cased as
an intrinsic a few lines above, in exactly the place the `.get()` case was
missing.

Underneath that, a `Seq<f64>` local was never marked as holding float elements:
`lower_type_expr_is_array_of_f64` only recognised `TypeExprKind::TypeArray`, and
the Seq branch of let/var lowering recorded the element LAYOUT but not its KIND.

## Fix

`self-hosted/ir/lower.sio`, three pieces:

1. `lower_type_expr_is_array_of_f64` recognises `Seq<f64>` — a Seq of floats is
   an array of float elements, spelled as a named type with a type argument.
2. The Seq branch of let/var lowering marks the local float-element, as the
   array branch always did.
3. Both classifiers answer the `.get()` case, beside the `.len()` intrinsic.

## Method note

Three fix attempts and three rebuilds preceded this, all aimed at marking the
RESULT of `seq_get` at its emission sites. All three were wrong, and the
symptom said so from the start: `println` refusing with "unresolved scalar kind"
is missing information, not corrupted information. A trace build settled it in
one cycle by showing that the marking branches did fire and were simply never
consumed. The trace should have been the second step, not the fifth.

The three speculative `ir_mark_float_reg` emissions were removed once the
classifiers were fixed, and all six verifications still pass without them.

## Residual — NOT claimed closed

- **`&Seq<f64>` as a parameter.** Distinct from the local and struct-field
  paths. One shape returns a wrong value, another segfaults. Not investigated.
- Seq element types other than f64/f32 and i64 are untested here.

## Closure (2026-09-06)

Closed as fixed, not waived. Landed in PR #2413, merged to `main` as
`0f46f7c3dc`.

What closes it: `scripts/ci/madaros_seq_f64_scalar_kind_gate.sh`, the acceptance
gate named in the record above, returns `MADAROS_SEQ_F64_SCALAR_KIND_GATE_OK`.
Re-run on 2026-09-06 against a Madaros built from the merged commit
`39d72a37a9`, engine md5 `54eccf3d`, with `SOUNIO_MADAROS_BIN` pinned, and again
against one built from `main` at `6b2b7ff9b0`, engine md5 `593ab4c9`, two merges
later. Green on both. The second is the one that matters for a closure: it says
the fix survives on the branch it landed on, not only on the commit that made
it.

The gate runs two fixtures, not one: the minimal
`tests/run-pass/seq_f64_element_scalar_kind.sio`, and the stdlib consumer
`tests/stdlib/graph/test_sinkhorn_e2e.sio`, which carries `Seq<f64>` measures and
matrices read through `&ProbMeasure` / `&CostMatrix` receivers. The second is
there because the first alone would pass on a fix that only handled bare locals.
Both are wired into `.github/workflows/ci.yml` (job "Madaros Current-Source f64
Lowering").

### Remote confirmation (E4)

GitHub CI run 34030342462, on `main` at `6b2b7ff9b0`, job "Madaros
Current-Source f64 Lowering": the gate step reports success on a runner-built
compiler, with no local environment involved. That is what lifts this record
from E3 (gate-bound) to E4 (remote-confirmed).

The same job fails at a later, unrelated step — "Run changed Madaros tests with
the current-source compiler", an intermittent exit 137 (killed) while rechecking
a fixed list of 20 known-failure tests. It fails identically on `main` and has
failed at other commits, so it is not attributable to this work and is not
claimed fixed here.

### The residual is live, and did not close with this record

`&Seq<T>` as a **parameter** is still broken. Closing this blocker without
moving that finding would have left an active wrong-code defect recorded only
inside a document marked closed — which is the failure mode this whole round of
work was about.

Re-measured 2026-09-06 on the `39d72a37a9` engine, `Seq<i64>` holding
`10, 20, 30`:

| shape | by reference | by value |
|---|---|---|
| `s.len().unwrap(...)` | `4201410`, same every run | `3` |
| `s.get(0)` | a stack address, different every run | `10` |
| accumulating loop over both | SIGSEGV (rc 139) | `60` |

Both by-reference results are addresses rather than data: one static, one moving
under ASLR across five runs. The callee is reading one indirection level off —
it receives the address of the handle instead of the handle. `souc check`
accepts all of it.

On the `main` engine `593ab4c9` only the witness was re-run, not the full table;
it reports the gap present.

This also **narrows** the original residual: it is not specific to `f64`. It
reproduces on `Seq<i64>`, so it is not a float-classification defect at all and
never belonged to this blocker's root cause.

It now lives in three places that stay open:

- `docs/compiler/KNOWN_LIMITATIONS.md` — the measured limitation and the working
  rule (take `Seq<T>` by value).
- `tests/known-gaps/language/seq_ref_param_loses_handle.sio` — the repro, which
  asserts the defect.
- `scripts/ci/language_gap_ratchet_gate.sh` — a ratchet line that fails on
  purpose when the parameter path starts working, which is the signal to move
  the repro into `tests/run-pass/` and write the migration note.

To re-open this blocker: `bash scripts/ci/madaros_seq_f64_scalar_kind_gate.sh`.
A non-zero exit or a missing sentinel re-opens it. The parameter residual is a
different failure and does not re-open this record.
