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
Status: open
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
Evidence-Level: E3
Fallback-Path: bind the read to a typed local before use.
LLM-Offload: not-required
Residual: `&Seq<f64>` PARAMETER is NOT fixed -- separate path, wrong value in
  one shape and SIGSEGV in another.
Next-Action: root-cause the parameter path.
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
