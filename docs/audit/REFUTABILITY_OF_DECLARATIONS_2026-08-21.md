<!-- docs:meta
topic_id: repo.docs.audit.refutability-of-declarations-2026-08-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.refutability-of-declarations-2026-08-21
-->

# Which declarations can Sounio refute?

**Date:** 2026-08-21. **Engines:** Madaros (`bin/souc`, default) and
`SOUNIO_SOUC_ENGINE=lean_single`. Every row below was probed with a wrong
declaration *and* a correct one — a probe that only passes measures nothing.

## The question

Sounio lets you declare many things. A declaration is **refutable** when the
compiler holds evidence that can contradict it — not merely when it is parsed
and stored. Founder ruling, 2026-08-21: a written effect declaration becomes a
claim the body can refute. This document asks the same question of every other
declaration in the language, because the answer decides what that ruling is
part of.

## The table

| declaration | refutable? | probe |
|---|---|---|
| type | **yes** | ordinary type checking |
| refinement alias | **yes**, after the fix in this branch | `let p: Pos = 0 - 5` |
| units | **yes**, after the fix in this branch | `let d: mg = 1.0; let v: mL = 1.0; d + v` |
| effect | **no** — declaring *more* than the body performs is free | `with Div` on a body that only uses `%` |
| integer width | **no** — write `i512` for a value that fits in `i64`, nothing objects | — |
| **uncertainty** | **no** | `measure(500.0, uncertainty: 0.0 - 2.5)` → **`rc=0`** |
| **provenance** | **no** | an unrecognised kind becomes `DERIVED` in silence |

## What was measured, per row

**Uncertainty.** A *negative* uncertainty is physically impossible. Both the
correct and the impossible form are accepted:

```
measure(500.0, uncertainty: 2.5)        rc=0
measure(500.0, uncertainty: 0.0 - 2.5)  rc=0
```

The positive control passes, so the probe is valid. Nothing in the compiler
contradicts a declared uncertainty.

**Provenance.** `provenance_from_ast` in `self-hosted/check/epistemic.sio` maps
five AST kinds to five constants and ends:

```sounio
provenance_new(PROVENANCE_KIND_DERIVED)
```

Anything unrecognised — and `None` — becomes `DERIVED`. An unknown provenance is
not refused; it is **relabelled**, with no diagnostic. Writing a wrong
provenance and writing none are the same program.

**Units and refinements** were dead under the default engine until the fix in
this branch, and the failure was worse than an absence: the correct program and
the incorrect one received the *same* diagnostic.

| | correct program | incorrect program |
|---|---|---|
| `lean_single` | no error | `incompatible unit dimensions` / `refinement type violation` |
| Madaros, before | `error[E001]` | `error[E001]` |
| Madaros, after | no error | `incompatible unit dimensions` / `unit mismatch` |

Cause: two resolvers — `units.find` and `refinement_table_find` — existed only
on the by-value type spine. Annotations traverse the `*mut` spine, which had
neither. See the commit for the full account.

CI runs `lean_single`, so none of this was visible from any check that exists.

## The pattern

**In a language whose thesis is knowledge under uncertainty, the epistemic
declarations are exactly the ones nothing contradicts.** You may declare that
your measurement has negative uncertainty and the compiler will agree. You may
invent a provenance and it will be quietly rewritten to `DERIVED`.

The two that *are* refutable — units and refinements — were refutable only on
the engine CI runs, not on the engine the user gets. That is not an absent
feature; it is a silent regression in the one part of this table that was
already built.

## Why this is one agenda and not seven

Every row is the same move in a different lane:

- an **effect** declaration is a claim the *body* can refute
- a **width** declaration is a claim the *proved bound* can refute
- a **precision** declaration is a claim the *uncertainty* can refute
- a **unit** declaration is a claim the *dimensions* can refute

The author asserts; the compiler holds the evidence. No language does this
across the board — Koka and OCaml 5 infer effects and make the annotation
optional; Rust and Java demand it and never infer; Zig gives you arbitrary
widths and never asks whether you need them; LiquidHaskell and F* prove bounds
and never choose anything from the proof.

## Three-valued, not binary

`egate` in `stdlib/eisa/core.sio` already returns **ok / marginal / fail**
rather than a boolean, and the refutation verdict should inherit that, because
the two errors are not the same error:

- **under-declaring is unsoundness** — the program does what the signature
  denies. Hard error.
- **over-declaring is imprecision** — the program promises more care than it
  exercises. Not always a lie: it is sometimes deliberate defence, or forward
  compatibility.

A language that treats every over-declaration as an error forces everyone to
re-tighten signatures on every refactor, which is hostile — and is probably why
nobody has shipped this. The way out is that over-declaration is not an error
but a **measurement the language hands back**: *you declared `Div, Mod, Panic`;
the body produces `Mod`. Intended?*

That would give Sounio something no compiler has: it tells you where you are
being more careful than you need to be, and leaves the choice with you.

## Owed

- The uncertainty and provenance rows are open. Neither has a gate.
- Whether over-declaration should warn, measure, or refuse is a founder
  decision and is not taken here.
