<!-- docs:meta
topic_id: repo.docs.planning.action-plan-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.planning.action-plan-2026-08-20
-->

---
title: Action plan — turning the 2026-08-20 measurements into work
status: active
date: 2026-08-20
last_validated: 2026-08-20
---

# Action plan — turning the 2026-08-20 measurements into work

One day of measurement produced fourteen pull requests and six founder-facing
questions. This converts it into work, and it exists because of a specific
criticism made during that day: *"more audits before that is measuring the same
hole."*

**Every item below names the gate it moves.** An item whose progress cannot be
read off a number is not in this plan — narrative progress is how the previous
debts became invisible in the first place.

## 0. The rule this plan is built on

Sounio's ordinary machinery is sound and its ambitious machinery is loose. That is
not an impression; it is the shape of eight measurements:

- the effect **row** discipline propagates and refuses (`E035`) for all 30 names,
  generically, and was mistaken for absent because the check names no effect
- `TypeNamed` lowers every ordinary struct on **both** spines and refuses a
  mismatch with `E009`
- `Panic`, `Mut`, `Div` need no special case because the generic row consumes them

What breaks is always the **special claim**: a type carrying an extra promise, an
effect with a payload, a builtin literal. So the plan attacks joints, not parts.

## 1. Decided — implement

| ruling | where | state |
|---|---|---|
| **ε is the type-level error bound; belief is the value layer with its own name** (§8.2.6) | `S08` #2037 | first step **DONE** — `Epistemic(N)` now enforced, `E215`, agreeing with lean_single at N=400/401/950/999 (#2048) |
| **A function type carries the effects of the function** (§6.0) | `S06` | ratchet frozen at **216** bare parameter-position function types. Not yet shrinking. |
| **`Panic` must have a function** | measured | already true — `E035` propagates. Closed. |

**Owed against ruling 1:** the five ε-divergent `compile-fail` tests are written in
the wrong layer (§8.2.6). Rewriting them into the value layer is what drives
`epsilon_engine_parity.frozen` from **5** toward **0**.

## 2. Cause fixes, in order, each with its gate

The order is not preference. Each item removes the reason the next one cannot be
measured.

**C1 — an unknown effect name must be refused.** `with Zorblex` checks clean on
both engines *and does not propagate*; the generic row is strong and simply does
not apply to names outside the list. Cause: `collect_effects_from_list` drops
`eff_id < 0` silently. *Gate: `effect_name_closed_list` (2,845) must shrink.*
**In flight — grok-cli2.**

**C2 — the ninth effect must not vanish.** Same guard, other half:
`eff_id >= 0 && count < 8`. Widening `[i64; 8]` is a separate conversation; the
defect is the silent drop. *Gate: same as C1.* **Bundled with C1.**

**C3 — bring the ambitious kinds onto the `_mut` spine.** Twenty kinds — the whole
ZD, proof, causal, privacy, aleatoric and session surface — fall to a `_ =>` that
counts an error and prints nothing. This is why `E201` is unreachable and why a
correct locus check sits on the wrong spine. *Gate: `silent_type_spine` (19) must
shrink.* **In flight — cursor-3, starting with the eight ZD kinds.**

**C4 — the ZD family must be inhabitable, or say it is not.** No `run-pass` test in
the tree exercises any of the eight; a call gives `E009`, a return `E008`, and the
only way in is a cast that accepts `"a string" as ExactlyPrivate<f64>`. C3 is a
prerequisite. *Gate: a new `run-pass` witness that a ZD wrapper can be constructed
and refused for a real reason.*

**C5 — shrink the function-type effect debt.** 216 parameter-position function
types declare no effects against a ruling that says they carry them.
*Gate: `fn_type_effect_ratchet` (216) must shrink.* Not started.

## 3. Decisions owed — measured, unblocked, waiting

| # | question | measured basis |
|---|---|---|
| 1 | **`i256`: real, alias, or Reserved?** | `i256` is `i64`; the Lorenz certificate reaches `868,167,572 × 2^63`; `fn i256_*` occurs 0× in stdlib. `f128`'s `E218` is the honest template. |
| 2 | **Does a declared integer width mean anything?** | none does: `i8` gives 200 for 100+100. |
| 3 | **Is the integer tower closed or open?** | `i7`, `i999999`, `u4096` all typecheck; `i0` does not. |
| 4 | **Canonical epistemic value shape** | 18 shapes → **10 classes** modulo spelling; 26 of 45 sites are one idea; the residue is binary: scalar confidence (26) or Beta `(α,β)` (10). |
| 5 | **`int`/`uint`, `own`/`handle`, `u16`** | documented and dead — decision table in #2049. |
| 6 | **Does `Reserved` belong on both engines?** | `E218` is Madaros-only; lean_single accepts `f128`. |
| 7 | **Effect rows: abstract over effects?** | nothing in live code requires row polymorphism; `[i64; 8]` is a bounded set. Decides §6's subtyping direction. |
| 8 | **Does `Epistemic(950)` satisfy `Epistemic(400)`?** | §8, not §6 — belief is the value layer. |

## 4. Owed measurement — not decisions

**M1 — recompute the Lorenz obligations at full width.** The arithmetic is unsound
(§12.2.6). One obligation, step-5 remainder, has been recomputed and **survives**
(`source_lte_ok = 1`). One obligation is not the certificate. Owed: the same across
every covered obligation, then across those marked `NOT EXECUTABLE`.

**M2 — the `-1` payload sentinel collides.** `Epistemic(-1)` is indistinguishable
from no payload. Harmless today because negative confidence is absurd; a defect the
moment a consumer reads `payload >= 0`.

**M3 — census fragility.** `stdlib_type_census.py` reads a fixed 14,000-char
window; the function is 6,470 today and undercounts silently if it grows.

## 5. Spec, section by section

Written today: **§3** (#2042), **§6.1.0** (#2047), **§8.2.4-bis/ter/5/6**, **§12**
(#2041). Frame rows updated to match.

Next, in the order the measurements make possible:

- **§1 Lexical / §2 Grammar** — five of seven `CLAUDE.md` rows measured as style,
  four bang-macros accepted with three SIGSEGV, `assert!` inert. All lexical facts
  with an executable gate already built (#2044).
- **§4 Type system rules** — 53% of stdlib types never appear in a parameter; the
  two-spine split. Blocked on nothing; C3 will change the numbers.
- **§5 Effects vocabulary** — 30 names over 29 ids; `Confidence` aliases
  `Epistemic`. §6.1.0 already carries the derivation.

## 6. What is not in this plan, and why

Not the 3,156 stdlib types. **53% never reaching a parameter is a fact about how
the library is written, not a defect list** — a value constructed and read but
never passed is legitimate. It becomes a question only where a type carries a claim.

Not widening `[i64; 8]`. The measured defect is the silent drop, and a wider table
with the same silence is the same defect at a higher number.

Not `stdlib/systems` corrections. The arithmetic is unsound and one verdict has
been recomputed and holds; the rest are unaudited (M1). Patching before M1 would be
changing an artefact whose behaviour is not yet known.

## Claims Forbidden

- Not that any Lorenz certificate conclusion is wrong. One was recomputed and
  survives; the others are unaudited.
- Not that the 21 effects with no special-case logic are unconsumed. They are
  consumed by the generic row; the column that said otherwise was mine and is
  corrected.
- Not that a stdlib type lacking a compiler kind is a defect. Ordinary types are
  lowered generically by design.
- Not that this plan is complete. Every item names its gate; anything that cannot
  name one is not yet understood well enough to be planned.
