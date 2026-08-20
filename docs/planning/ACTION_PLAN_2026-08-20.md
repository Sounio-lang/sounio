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

## 2-bis. RULED 2026-08-20 — four decisions, and the hard option was taken four times

The founder ruled on four of the eight. **Every one chose capability over
convenience.** They are recorded together because they are not independent: they
compose into a single direction — *Sounio's declarations become true.* Each of the
four is today a name that promises and does not deliver.

### R1 — `i256` is implemented for real, in limbs

Not `Reserved`, not an alias. **And it is much cheaper than "a new arithmetic
library", because most of it already exists unused** — measured after the ruling,
and it corrects a claim of mine in §12 and `KNOWN_LIMITATIONS`.

I wrote that *"`fn i256_*` occurs zero times, so there is no limb implementation
underneath."* True about the **name**, false about the **fact**. There is limb
machinery in two places:

**In the compiler, and it is already width-generic:**

    // self-hosted/ir/numeric_payload.sio
    pub fn ir_wide_numeric_required_limb_count(format_id: i64) -> i64 with Div {
        let descriptor = binary_format_descriptor_for_id(format_id)
        descriptor.storage_bits / 64          // <- the limb count DERIVES from the width
    }

    pub struct IrWideNumericPayloadEntry { format_id, limb_start, limb_count }
    pub struct IrWideNumericPayloadPool   { entries: [...; 256], payload_count }

`self-hosted/check/numeric_format.sio` (54 lines) already carries descriptors for
`binary128` (`storage_bits: 128`) and `binary256` (`storage_bits: 256`).

**In the Lorenz certificate itself**, hand-rolled under a different naming
convention: `lorenz_dec4_add_limb`, `lorenz_dec4_mul_small_limb`,
`lorenz_dec4_div3_exact_limb`, `lorenz_i256_beta_z_limb`. So the certificate is not
naive about width — parts of it already carry limbs by hand, which is what makes
the raw typed-`i256` product at `step5.sio:2310` the exception rather than the rule.

**What is actually missing is the connection, not the representation.**
`wide_numeric` occurs **0 times** in `self-hosted/ir/lower.sio` and **0 times** in
`self-hosted/native/codegen_x86_linux.sio`. The pool is built and unused — the same
shape as every other debt measured today, and this time it works in our favour.

**So R1 becomes:** two descriptors (`storage_bits: 256`, `storage_bits: 512`), the
arithmetic ops, and the lowering/codegen wiring the pool never got.

**And `i512` costs one descriptor line beyond `i256`**, because `storage_bits / 64`
does the rest. The founder's instinct that *"building i512 would already be good"*
is measured: at the representation layer it is nearly free, and building both at
once is what proves the width-generic path is genuinely generic rather than an
`i256` special case wearing a loop.

**Acceptance:** the peak the Lorenz certificate actually reaches —
`868,167,572 × 2^63` at `lorenz_i256_cert_step5.sio:2310` — computes exactly under
`i256`, an arbitrary-precision oracle agrees, **and the same expression under
`i512` agrees with it**. Two widths passing the same oracle is the evidence that
the path is generic. *Gate: `run-pass` witnesses at both widths reproducing the
measured peak.*

### R2 — every declared integer width wraps at its declared width

`i8` gives `200` for `100 + 100` where `-56` is due; `u8` gives `400`; `i32` gives
`4000000000`. All of them become true.

**The risk is named, not hidden:** code that today relies on the silent `i64`
behind a narrow annotation **changes behaviour**. That is not a reason to decline —
it is the reason to land it behind a witness suite per width before anything else
depends on it.

**Acceptance:** a `run-pass` per width whose expected output is the wrapped value.
Interacts with R1: `i256` is the same question at the other end of the tower, so
one representation strategy should serve both.

### R3 — the epistemic value carries a Beta posterior, not a scalar confidence

    struct Epistemic { value: f64, uncertainty: f64, conf_alpha: f64, conf_beta: f64 }

The minority form today — **10 of 45 sites** against 26 for the scalar — chosen
because it knows what the scalar cannot: `0.5` from two observations is
`Beta(1,1)`, `0.5` from two thousand is `Beta(1000,1000)`, and a point cannot tell
them apart.

**This is the only one of the four that changes a number rather than a check.**
GUM propagation stops composing variances and starts composing *evidence*. It
touches the dissertation's own surface. Sequencing follows from that: the
propagation rule is specified and tested against an oracle **before** any site is
migrated.

**Acceptance:** the 26 scalar sites migrate without changing a published result, or
each changed result is derived and explained. *Gate: `S08` conformance.*

### R4 — effects become a real row, with effect variables

    fn map<A, B, e>(xs: [A; 8], f: fn(A) -> B with e) -> [B; 8] with e

The largest type-system change on the list, taken where **nothing in live code
requires it** — the only `with e` in the tree is in `effect.sio.old`.

**That is a deliberate inversion and it is worth saying plainly.** Every debt
measured on 2026-08-20 has the shape *built, and not connected*: the capability
arrived before the consumer and the consumer never came. R4 chooses to build a
capability before its consumer **again** — knowingly. The mitigation is the one
this repository already has: a ratchet from day one, so the gap between the
capability and its first live use is a number that must shrink rather than a fact
nobody is watching.

`current_effects: [i64; 8]` is a fixed width, and a fixed width is the negation of
a row. So R4 subsumes **C2**: the ninth-effect drop stops being a hole to plug and
becomes a representation to replace.

**Acceptance:** a live `.sio` in `tests/run-pass/` that abstracts over an effect
and could not be written today. *Gate: a new ratchet counting effect-polymorphic
signatures, starting at 0 and required to grow — the first ratchet in this tree
that ratchets upward.*

## 2-ter. What the four rulings imply for order

They interact, so the order is not the order they were asked in.

1. **R2 before R1.** Both are representation questions about the same tower —
   narrow widths at one end, `i256` at the other. One strategy should serve both,
   and getting `i8` right is the cheap rehearsal for getting `i256` right.
2. **R3's propagation rule before R3's migration.** The Beta posterior changes a
   *number*, not a check. Specify and test the composition against an
   arbitrary-precision oracle first; migrate the 26 sites second. Migrating first
   would change published results before the rule that changed them is written down.
3. **R1 unblocks M1.** Recomputing the Lorenz obligations at full width is exactly
   what a real `i256` does natively. Do not build a one-off recomputation harness
   that R1 will make redundant.
4. **R4 subsumes C2**, and should not start before C1 lands. C1 makes an unknown
   effect name refused; R4 changes what an effect row *is*. Doing them at once
   means neither has a stable base to be tested against.
5. **C3 is independent of all four** and is already in flight. It is the only cause
   fix that needs nothing from these rulings.

## 3. Decisions owed — measured, unblocked, waiting

| # | question | measured basis |
|---|---|---|
| ~~1~~ | ~~`i256`~~ | **RULED — R1, implement in limbs.** |
| ~~2~~ | ~~integer widths~~ | **RULED — R2, wrap at the declared width.** |
| 3 | **Is the integer tower closed or open?** | `i7`, `i999999`, `u4096` all typecheck; `i0` does not. |
| ~~4~~ | ~~canonical epistemic shape~~ | **RULED — R3, Beta `(α, β)`.** |
| 5 | **`int`/`uint`, `own`/`handle`, `u16`** | documented and dead — decision table in #2049. |
| 6 | **Does `Reserved` belong on both engines?** | `E218` is Madaros-only; lean_single accepts `f128`. |
| ~~7~~ | ~~abstract over effects?~~ | **RULED — R4, a real row with effect variables.** §6's subtyping direction follows from it and is still owed. |
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
