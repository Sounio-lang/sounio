<!-- docs:meta
topic_id: repo.docs.internal.concepts.type-interrogation
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.type-interrogation
-->

# Type Interrogation

Concept-ID: `SOUNIO-TYPE-INTERROGATION`

Status: **Hypothesis** — the rule is stated here for the first time; five
independent instances of its violation were measured on 2026-08-19, and none is
fixed.

## Founder Intent

> A property declared in a type must be interrogated somewhere in the pipeline,
> or the type is decorative.

Sounio's types carry claims other languages do not attempt: an error bound, a
validity condition, a provenance, a privacy guarantee, a totality obligation. A
type that carries a claim nobody ever asks about is worse than a type that
carries none — it converts an unmade guarantee into an apparent one, and the
reader cannot tell the difference from the source.

## The rule

For every property a type constructor carries, **at least one point in the
pipeline must ask whether it holds**, and the asking must be reachable from
ordinary source.

### The naming clause

> **A type declares which proposition it interrogates, and that proposition must
> be decidable.**

Founder ruling, 2026-08-19. Without it the rule is unsatisfiable and therefore
disciplines nobody.

The measured case that forced it: `ExactlyPrivate<T>` names a *privacy*
guarantee, and part of that guarantee is **not decidable at all**. That the
payload `T` is *"the contribution of subject U"* is a **semantic** fact about
provenance, not an algebraic one
(`docs/audit/EXACTLY_PRIVATE_LEAN_BRIDGE_DISPATCH_2026-08-19.md`). A rule
demanding that the compiler establish it demands the impossible, and a rule that
cannot be satisfied is either ignored or met by pretending.

So the obligation is not *prove the whole promise*. It is: **say what you check,
and check what you say.**

`ExactlyPrivate<T, A>` under this clause does not promise *"this is U's
contribution"*. It promises **"this value lies in the annihilator kernel of
`A`"** — which is decidable over the finite model, and is exactly what
`unlearning_kernel_exact` and `every_primitive_has_4_annihilators` already prove.

**What this makes enforceable.** A type is in default not for failing to prove
everything, but for **promising what it did not name**. Measured against that,
today's `ExactlyPrivate<T>` fails: the name promises exact privacy and the
interrogation is `with ZD`.

**Where the rest of the promise goes.** The value↔subject link does not vanish —
it **moves**, to provenance, which is a different field with its own
interrogation. And provenance today has three unwritable kinds and a collision
between *absent* and *`derived`*
(`docs/audit/PROVENANCE_LAYER_STAIRCASE_2026-08-19.md`). So the two defects
measured on 2026-08-19 are the **two halves of one guarantee**, and neither is
complete without the other.

Three distinct failures are all violations, and they are not interchangeable:

1. **Declared, never asked.** The property is a member of the type and no stage
   reads it.
2. **Asked at declaration, dropped after.** A guard fires where the type is
   written, and lowering discards the property, so nothing downstream can ask
   again.
3. **Ceremony instead of proof.** The compiler requires the *programmer* to
   assert something (declare an effect, name a bound) and never verifies the
   thing the assertion is about.

The third is the most dangerous, because it produces a diagnostic. A refusal is
read as enforcement.

## The five measured instances

`origin/main`, 2026-08-19. Each was measured independently, several by different
agents with different instruments.

| type property | what is asked | what is not |
|---|---|---|
| `Knowledge<T>.epsilon` | nothing — no site consults it | whether the computed variance satisfies the bound |
| `ExactlyPrivate<T>` | that the function declares `with ZD` (`check.sio` `lower_exactly_private_type`, by effect id 18 — structurally, not by name) | that the contribution is **algebraically zero**. After lowering the type *is* `inner_ty`: `ExactlyPrivate<f64>` becomes `f64` and nothing downstream knows |
| `Forgettable<T>` | nothing — `ForgettableTypeInfo` occurs once in the tree, its own declaration | that the value was consumed by a ZD operation before escaping |
| unit types (`mg`) | nothing that distinguishes them — `let d: qzx = 500.0` for an invented name gives the **same diagnostic** as a registered unit | dimension: `(500.0 as mg) as m` (mass → length) is accepted, and `500.0 as flurb` compiles |
| `with Div` / `with Panic` | nothing — no `has_effect_id` decision consults either; and `fn raw(a, d) { a / d }` type-checks on both engines **without** declaring `Div` | that division is total. `pred_implies` and path narrowing both work in `check/refinement.sio`; `/` never asks them |

`ExactlyPrivate<T>` is the clearest case of failure 3. The `with ZD` requirement
is real, structural, and has a compile-fail test with a golden stderr
(`error[E201]`). It obliges the programmer to *say* they are in zero-divisor
territory. It does not verify that any annihilation occurred. A reader who sees
E201 fire concludes the privacy guarantee is enforced. It is the declaration that
is enforced.

## Why this is one rule and not five repairs

Each instance has an obvious local fix, and five local fixes leave the rule
unstated, so the sixth property added to a type repeats the pattern. Measured
support for that reading: `SOUNIO-SPEC` frame, *"the recurring shape"* — six
subsystems designed, built, and never connected, with the design and the partial
implementation committed **in the same commit**. Nothing rotted. The interrogation
was never written.

Stating the rule turns each instance from an archaeological discovery into a
conformance obligation, which is the difference between a language that has these
properties and a language that **keeps** them.

## Claims Forbidden

What this concept does not claim:

- Not that a property must be interrogated at every stage. Once, reachably, is
  the requirement.
- Not that ceremony is worthless. `with ZD` is a real constraint and catches real
  mistakes; it is simply not the guarantee the type's name promises.
- Not that the five are equally severe. `ExactlyPrivate<T>` is a security claim
  and is therefore the one where an apparent guarantee costs most.

## Related

- `SOUNIO-EPISTEMIC-ERASURE` — separating a value from its uncertainty is an act
- `SOUNIO-EFFECT-DECLARATION` — `with X` requires `X` to exist
- `SOUNIO-S-G-R` — silent, growing, reachable ⇒ a gate is owed
- `SOUNIO-GATING-ENGINE` — a green must name the engine that produced it
- `SOUNIO-SPEC-06` §6.1.1, `SOUNIO-SPEC-08` §8.2 — the effect and epistemic
  measurements this concept generalises
