<!-- docs:meta
topic_id: repo.docs.audit.layer-crossings-map-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.layer-crossings-map-2026-08-19
-->

# Layer crossings — what proves it, what declares it, what asks

## Why this exists

Sounio was built by one person. That is not context, it is the **mechanism**.

When one author builds three layers in sequence — a Lean theorem, a Sounio
library, a compiler type gate — the earlier layer is alive in memory while the
later one is written. The type *feels* connected to the theorem because the
author knows the theorem is there. There is no handoff, so there is no moment at
which anyone asks *"and what does this connect to?"*, and the relation is never
written down.

Every disconnection measured on 2026-08-19 has that shape. The layers are
individually correct. What is missing is the **crossing**, and the crossing lived
in one head.

This map is the crossings, written down. It is seeded from measurements made on
2026-08-19 and is **incomplete by construction** — only the author can say which
crossings were intended and never made.

## The map

| property | proved by | declared by | **asked by** |
|---|---|---|---|
| `ExactlyPrivate<T>` exactness | `formal/lean4/SounioSurgicalInterventions.lean` §2, `unlearning_kernel_exact`, `native_decide`, no `sorry` | `check.sio` `lower_exactly_private_type`, E201 | **nothing** — the guard checks `with ZD` and returns `inner_ty` |
| `Forgettable<T>` annihilation | *unmeasured* — `ForgettableTypeInfo` occurs once in the tree, its own declaration | `check.sio` `lower_forgettable_type`, E200 | **nothing** |
| `Editable<T>` (G5) kernel confinement | *unmeasured* | E202 | **nothing** |
| `CapabilityGated<T>` (G7) | *unmeasured* | E203 | **nothing** |
| `Composable<T>` (G8) | *unmeasured* | E204 | **nothing** |
| `Audited<T>` (G9) Lean obligation | *unmeasured* — the type's own promise is that it emits one | E205 | **nothing** |
| `Revivable<T>` (G10) | *unmeasured* | E206 | **nothing** |
| `Interpretable<T>` (AMI) 168 classes | `zd_classes_168` in the same Lean file | E207 | **nothing** |
| `Knowledge<T>.epsilon` bound | GUM coverage factor; `gum_k95` computes it | `parser/ast.sio` `EpsilonBound { op, value }` | **nothing** |
| `Knowledge<T>.provenance` | — | `AstProvenanceKind`, six cases | parser reaches **three**; unknown components dropped without diagnostic |
| unit dimension | `stdlib/units/lib.sio` works as a value-level library | `unit mg;` registers **dimensionless** | nothing distinguishes a registered unit from an invented name |
| `with Div` totality | `check/refinement.sio` `pred_implies` + path narrowing both work | `with Div` on 46,219 signatures | **nothing** — `/` generates no obligation, and `a / d` type-checks with no `Div` at all |
| effect discharge by handler | — | `handle` parses and type-checks | **nothing** — `ExprHandle` occurs 0 times in `ir/`, `native/`, `enir/` |

## The one crossing that was made

`self-hosted/ir/egraph.sio` — 3,526 lines, 286 functions, equality saturation —
is imported by `self-hosted/compiler/main.sio` and `ir/opt_cleanup.sio`.

It is the only large piece examined that is both designed and connected, and the
reason is structural rather than virtuous: **the crossing was the point of the
piece**. An optimiser that nothing calls optimises nothing, so it could not look
finished while disconnected. Every other property on this map looks finished
without its crossing — which is exactly why the crossing was skippable.

## How to use this

A new property added to a type gets a row **before** the type gate is written,
with the third column filled or explicitly marked owed. `SOUNIO-TYPE-INTERROGATION`
is the rule; this is its ledger.

Rows marked *unmeasured* are not claims that no proof exists. They are claims
that **nobody has looked**, which is a different and cheaper thing to fix.

## Claims forbidden

- That the un-crossed layers are wrong. Each is correct in itself; what is
  missing is the relation.
- That this map is complete. Only the author knows which crossings were intended.
