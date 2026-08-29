<!-- docs:meta
topic_id: repo.docs.audit.zd-family-call-path-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zd-family-call-path-2026-08-20
-->

---
title: The eight ZD types cannot be called, and the one path in verifies nothing
status: measured
date: 2026-08-20
last_validated: 2026-08-20
engines: Madaros v0.80.0 (default), lean_single
---

# The eight ZD types cannot be called, and the one path in verifies nothing

`Forgettable`, `ExactlyPrivate`, `Editable`, `CapabilityGated`, `Composable`,
`Audited`, `Revivable`, `Interpretable` — diagnostics E200–E207, the family the
Lean development proves the G3 theorem about. This measures what a program can
actually do with them.

## A function taking one cannot be called

    fn release(x: ExactlyPrivate<f64>) -> f64 with ZD { 0.0 }
    fn caller() -> f64 with ZD { release(1.0) }

| engine | result |
|---|---|
| Madaros v0.80.0 | `error[E009]: argument type does not match parameter` / `expected ExactlyPrivate` / `found f64` |
| lean_single | accepted, ELF emitted |

**All eight behave identically** under Madaros — E009 on any call with a bare
value. Producing one from a body of the inner type
(`fn make() -> ExactlyPrivate<f64> { 1.0 }`) gives **E008**.

Remove the caller and the same file checks clean. This is why no test in the tree
calls one, and why both witnesses the ExactlyPrivate forensic shipped
(`docs/audit/exactly_private_ta/witness_ep_one.sio`,
`docs/audit/exactly_private_lean/witness_ceremony_with_zd.sio`) declare a
parameter and never call it. It is not an oversight in those files. **The call
does not typecheck.**

At the time of measurement the corpus contained **zero** `run-pass` tests
exercising any of the eight.

## There is exactly one way in, and it checks the target only

    fn caller() -> f64 with ZD { release(1.0 as ExactlyPrivate<f64>) }   // check: OK

The cast works. Its controls are what matter:

| cast | Madaros |
|---|---|
| `1.0 as NaoExisteZorble` | **`error[E009]`** — the target type *is* verified |
| `"uma string" as ExactlyPrivate<f64>` | **`check: OK`** |

So the cast confirms the target type exists and verifies **nothing about the
source**. A string enters an `ExactlyPrivate<f64>` without a diagnostic.

## There is no ceremony either: E201 was never reached

This document first said the guarantee amounts to *declare `with ZD`, then cast
anything in*. That is still too generous, and the correction is recorded rather
than folded in silently, because it runs in the direction of my own earlier
claim.

`lower_exactly_private_type` is where `E201` — *"ExactlyPrivate&lt;T&gt; requires ZD
effect"* — is emitted. **Six positions were written without any `with ZD` at all,
and every one checks clean:**

| position | Madaros |
|---|---|
| `fn r(x: ExactlyPrivate<f64>) -> f64 { 0.0 }` | `check: OK` |
| `let x: ExactlyPrivate<f64> = 1.0 as ExactlyPrivate<f64>` | `check: OK` |
| the same, inside a function with a live caller | `check: OK` |
| `fn make() -> ExactlyPrivate<f64> { ... }` (return position) | `check: OK` |
| `struct Holder { p: ExactlyPrivate<f64> }` | `check: OK` |
| `docs/audit/exactly_private_lean/witness_nozd.sio` — the witness written to demonstrate the refusal | `check: OK` |

The cause is structural, and it is why a patch to the obvious function changes
nothing. There are **two** lowering spines. Parameter types go through
`checker_lower_type_expr_mut` (`check/check.sio:2329`), whose `match` handles
`TypeNamed`, `TypeUnit`, `TypeNever`, `TypeInfer`, `TypeSelfUpper`,
`TypeKnowledge`, `TypeModel`, `TypePolicy`, `TypeDecisionPolicy` and
`TypeDeferralPolicy` — and **not one** of the eight ZD kinds. They fall to the
`_ =>` default. `lower_exactly_private_type`, with its E201 and its inner-type
lowering, sits on the *other* spine.

**Verified by positive control, not inferred.** `zd_locus_is_wellformed` was
forced to return `false` unconditionally and the compiler rebuilt: a locus that
must then be refused was still accepted. The function is not reached.

The claim is bounded: six positions were measured and none reached it. That is
not a proof of unreachability, but it is every position a program can put the
type in that I could construct.

## What the guarantee currently amounts to

Write `with ZD`, then cast any value at all into the wrapper — including one of a
type the wrapper does not name. `lower_exactly_private_type`
(`check/check.sio`) confirms the effect was declared, emits E201 if it was not,
and returns the inner type.

`docs/internal/concepts/type-interrogation.md` calls this failure type 3,
*ceremony instead of proof*. Both that description and this document's first
version of it are too generous. **There is no ceremony.** The compiler does not
ask the programmer to assert anything: the effect requirement is unreached, the
wrapper accepts a string through a cast, and a program may write any of the eight
names in any position with no consequence of any kind.

## Scope of the claim

This measures the **call path**, not the algebra. The Lean development
(`formal/lean4/`) proves what it proves; nothing here touches that. What is
measured is that the type the theorem names, in the language, today, can be
inhabited by a string.

## Reproduce

    export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
    # E009 on the call
    printf 'fn r(x: ExactlyPrivate<f64>) -> f64 with ZD { 0.0 }\nfn c() -> f64 with ZD { r(1.0) }\nfn main() -> i32 with IO { 0 }\n' > /tmp/a.sio
    ./bin/souc check /tmp/a.sio
    # accepted through the cast, from a string
    printf 'fn r(x: ExactlyPrivate<f64>) -> f64 with ZD { 0.0 }\nfn c() -> f64 with ZD { r("s" as ExactlyPrivate<f64>) }\nfn main() -> i32 with IO { 0 }\n' > /tmp/b.sio
    ./bin/souc check /tmp/b.sio
