<!-- docs:meta
topic_id: repo.docs.internal.concepts.verified-lowering
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.verified-lowering
-->

# Verified Lowering

Concept-ID: `SOUNIO-VERIFIED-LOWERING`

Status: **Hypothesis** — architectural ruling by the founder on 2026-08-19.
Nothing implements it. The distance between the ruling and the tree is stated
below in full, because a ruling recorded without its cost reads as a plan.

## Founder Intent

> ENIR becomes the only path. `ir/` and the e-graph are an optional
> accelerator, never mandatory, and every rewrite rule must carry translation
> validation before it may be used.

Verification is not a mode for special code. It is the floor. Optimisation is
what has to earn its way in.

## Why the source-level work does not survive without this

`SOUNIO-EPISTEMIC-ERASURE` fixes what a *program* may silently discard. Below
it sits a compiler, and every guarantee evaporates if the IR does not carry
what the source promised.

Measured 2026-08-19 on `origin/main`: the production IR has **no epistemic
type**. It has `IrEpistemicSection`, `ir_lift_knowledge`, and in `ir/lower.sio`
a family of emitters — `emit_variance_add`, `emit_variance_independent_product`,
`emit_variance_independent_div`, `emit_variance_scale`,
`bind_variance_for_field`, `alloc_variance_slot_for_local`.

**Uncertainty propagation is implemented as machine operations, not as a
property of a value.** Variance lives in a slot somebody must bind to the right
register, and the GUM rule is a sequence of instructions somebody must emit in
the right order.

Two consequences follow, and both are already visible:

- `emit_variance_independent_product` — *independent*. The independence
  assumption that `SOUNIO-PROVENANCE` says must be justified is the **name of
  a function**, with nothing checking it holds.
- The FO variance defect stops being mysterious. If variance is a hand-bound
  slot, **forgetting to bind it is the natural failure mode**. "≥3 arguments or
  an imported helper" is not a strange condition — it is where
  `bind_variance_for_base_reg` stopped being called. It is a manual-bookkeeping
  defect, not an arithmetic one.

## Why ENIR and not a repair of `ir/`

`enir/` is not an ordinary IR. It ships `verify.sio`, `canonical.sio`,
`hash.sio` and `interpreter.sio`, and the ENIR MIR is described in-tree as
translation-validated: each transformation can be shown equivalent to the one
before it.

`ir/` carries an e-graph with **1000+ rewrite rules and no translation
validation**. For ordinary code that is good engineering — rewrite hard, test
hard. For epistemic content it is 1000+ opportunities to change a variance
silently, with the result still compiling and still printing a number.

The two workloads have different **trust requirements**, and the founder's
ruling is that the verified one is the baseline rather than the exception.

## The cost, stated plainly

| | `ir/` (live) | `enir/` (ruled to become the path) |
|---|---:|---:|
| size | 3405 KB, 46 files | 400 KB, 14 files |
| production importers | 48 | **0** |
| instruction ceiling | `IR_MAX_INSTRS` 16384 | `EMIR_MAX_INSTRS` **128** |
| compiles `main.sio` (10,705 declared fns) | yes | no |

`enir/` **typechecks clean today** — 14/14 files `rc=0` under Madaros on Slurm,
with a validated positive control (`docs/audit/ENIR_MIR_DISCONNECT_COST_2026-08-19.md`,
PR #1951 family). It is disconnected, not rotten. It has CI: the E1–E3 gate set
runs when `self-hosted/enir` or `tools/eisa` change, and ≥12 scripts build
`enir/driver.sio` and drive it. What it has never had is a production importer.

**The compiler compiles itself.** If ENIR is the only path, Madaros descends
through ENIR, and today ENIR cannot compile Madaros — not marginally, but by
orders of magnitude.

## The incremental shape

The transition is not a rewrite. **Rules are validated one at a time, and an
unvalidated rule is simply not used.** The maturity ladder applies to rewrite
rules exactly as it applies to types: a rule is `Garden` until someone
validates it, and `Claim-ready` when a correct transformation is proven
equivalent **and** an incorrect one is refused.

That makes the path long but **monotone**, and measurable from day one: how
many of the 1000+ rules are validated.

## Required Invariants

- An unvalidated rewrite may not run on epistemic content. Not "should not" —
  the pipeline must be unable to select it.
- Uncertainty is carried, not bound. Any representation where forgetting a
  binding produces a compiling program with a wrong number reproduces the FO
  defect under a new name.
- Independence is a claim, not a function name. An emitter that assumes
  independent sources must be reachable only where provenance establishes it.
- Optimisation never becomes mandatory. If a path cannot be taken without the
  accelerator, the accelerator is the pipeline and this concept is void.

## Claims Forbidden

- Do not describe ENIR as the compilation path, the default, or in transition
  to either. It has zero production importers.
- Do not present the e-graph as validated in whole or in part. No rule carries
  translation validation today.
- Do not cite `enir/` typechecking clean as evidence of readiness. Typechecking
  is not lowering, and a 128-instruction ceiling is not a compiler.
- Do not attach a schedule to this. It is a ruling about direction, recorded on
  the day it was made.

## Related

- `SOUNIO-EPISTEMIC-ERASURE` — what this must preserve; the source-level half
- `SOUNIO-PROVENANCE` — supplies the independence that
  `emit_variance_independent_product` assumes
- `MATURITY_LADDER.md` — the ladder, here applied to rewrite rules
