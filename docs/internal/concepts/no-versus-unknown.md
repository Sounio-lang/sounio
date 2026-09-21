<!-- docs:meta
topic_id: repo.docs.internal.concepts.no-versus-unknown
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.no-versus-unknown
-->

# No Versus Unknown

Concept-ID: `SOUNIO-NO-VERSUS-UNKNOWN`

Status: **Hypothesis** — a principle underneath the rest of this corpus, named
on 2026-08-19 after it was measured three separate times in one day. Nothing
enforces it.

## Founder Intent

> A system must be able to distinguish **"no"** from **"I do not know"**.

When ignorance produces the same output as knowledge, every reader downstream
is reasoning about a verdict that was never reached. The result is not merely
wrong — it is **indistinguishable from right**, which is the only kind of wrong
that survives review.

## The three measurements

All three were found on the same day, in three unrelated subsystems, before the
common shape was noticed.

**1. Effects — `with NoSuchEffectX` compiles.**
`effect_name_to_id` returns `-1` for an unknown name and both collection paths
drop it: `rc=0`, `check: OK`. The compiler is not accepting the effect; it does
not know the word. But acceptance and ignorance emit the same result, so 2,800
`with Mod` declarations across 360 files have said nothing for years while their
authors believed they had declared something.
→ `SOUNIO-EFFECT-DECLARATION`

**2. Uncertainty — `variance_of` returns `0.000000`.**
A value that passed through `.value` has no variance to report, and a value
whose variance is genuinely zero reports the same digits. *There is no
uncertainty here* and *I was never told the uncertainty* are printed
identically, and the second is a fabrication wearing the first's face.
→ `SOUNIO-EPISTEMIC-ERASURE`

**3. Handlers — the multi-shot refusal is a ghost.**
The compile-time fast path appears to reject a multi-shot handler, which would
be the correct routing behaviour. Measured
(`docs/audit/EFFECTS_JUNCTION_ROUTING_2026-08-19.md`): it rejects because the
name is unknown to it. *"The machinery is not discriminating multi-shot;
name-ignorance is."* A test asserting that refusal would pass, and would be
measuring nothing.
→ `SOUNIO-VERIFIED-LOWERING`, the junction

## Why this sits underneath the others

Several concepts in this registry are instances of it rather than neighbours of
it. `SOUNIO-EFFECT-DECLARATION` requires declaration so that an unknown name
becomes a **refusal** instead of a silence. `SOUNIO-EPISTEMIC-ERASURE` marks a
projected value so that absent uncertainty stops looking like zero uncertainty.
`SOUNIO-SIGNAL-DIRECTION` is the same collision one layer up: improvement and
breakage share a colour. `MATURITY_LADDER`'s `Reserved` exists precisely so that
*not implemented* and *deliberately refused* stop reading alike.

The pattern was visible in each and named in none.

## The asymmetry that makes it dangerous

A false "no" is loud. Something that should have worked does not, and somebody
investigates within the hour.

A "no" that is really "I do not know" is **silent, and it accumulates**. Nothing
fails. The declaration is written, the number is printed, the test passes. The
cost is paid later by whoever trusts the output — and by then the chain from
answer back to ignorance has been lost.

This is why the corpus keeps arriving at refusal rather than default: a refusal
is a decision the system can be held to, and a default is a decision nobody made.

## Required Invariants

- Ignorance and negation must be **separately observable**. Where a single
  channel carries both, the channel is defective regardless of how correct each
  individual answer is.
- A test that passes must be shown to pass **for its stated reason**. The
  multi-shot ghost passes; it measures nothing. This is what the negative
  control in the two-programme test exists to catch, and it is why an
  unmeasured refusal is not evidence.
- An unknown input is refused by name, never absorbed. `-1`, `0.000000` and
  silence are absorption; a named diagnostic is refusal.
- Where the distinction cannot yet be made, say so in the record. A concept that
  cannot separate the two states must state that it cannot — which is itself an
  instance of the principle applied to documentation.

## Claims Forbidden

- Do not read this as requiring three-valued logic everywhere. It requires that
  the two states be distinguishable **where a reader will act on the
  difference**; a boolean is fine where ignorance cannot arise.
- Do not treat the three measurements as the full set. They are what one day
  surfaced, and the shape was noticed only after the third.
- Do not present this as enforced. No gate distinguishes ignorance from negation
  anywhere in the tree; each of the three remains open in its own concept.
- Do not use this to argue that `-1` or `0.0` are always wrong. The defect is
  the collision, not the sentinel: a sentinel that no correct value can occupy
  is fine, and the failures above are exactly the cases where a correct value
  can.

## Related

- `SOUNIO-EFFECT-DECLARATION`, `SOUNIO-EPISTEMIC-ERASURE`,
  `SOUNIO-SIGNAL-DIRECTION`, `MATURITY_LADDER` — instances
- `SOUNIO-EFFORT-LOCATION` — why the fix is a refusal rather than a note
