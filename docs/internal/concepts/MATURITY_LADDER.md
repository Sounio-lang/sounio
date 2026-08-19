<!-- docs:meta
topic_id: repo.docs.internal.concepts.maturity-ladder
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.maturity-ladder
-->

# Maturity Ladder — five states, and how to measure one

The evidence progression in `FOUNDER_INTENT.md` and `docs/internal/garden/README.md`
is `Garden -> Hypothesis -> Executable -> Claim-ready`. This document adds the
rule that makes the ladder measurable, and the fifth state the archaeology of
2026-08-19 found by failing.

## The ladder is monotone

    Claim-ready  =>  Executable  =>  Hypothesis  =>  Garden

Each rung **requires** every rung beneath it. A kind cannot be `Claim-ready`
without being `Executable`. If no program constructs it and passes, the
highest reachable position is `Hypothesis`, however many refusals the
compiler emits.

This was not stated in the first archaeology protocol. A lane applied the
four definitions independently, correctly, and reached a position the
definitions permitted and the ladder does not: `TyF128` marked `Claim-ready`
because Madaros refuses every use of it. The defect was in the protocol.

## The fifth state: Reserved

    Reserved   the name is taken, the system REFUSES every use with a NAMED
               diagnostic, and no use passes. Fail-closed by decision,
               semantics unimplemented.

`Reserved` is **not** `Hypothesis`. In `Hypothesis` the system is passive: it
does not know, and says nothing. In `Reserved` the system knows, acts, and
says why it refuses.

`Reserved` is **not** `Claim-ready`. Refusing everything is not
discriminating. A kind that rejects the correct program and the incorrect one
with the same diagnostic is not typing anything.

`Reserved` is honest, and worth more than `Hypothesis`. It is declared
waiting, not omission. It sits beside the ladder rather than on it: a
`Reserved` kind has not failed to reach `Executable`, it has been held short
of it on purpose.

## Reserved splits in two

Founder ruling, 2026-08-19. `Reserved` as first stated confuses two things
that do not owe the same, and reading them alike is what lets a promise sit
unpaid indefinitely without anyone deciding that it should.

    reserved-owed   the name is taken AND SOMEONE OWES. Carries three
                    mandatory fields:
                      Reserved-Owner       who signs
                      Reserved-Since       ISO date
                      Reserved-Blocked-On  the TECHNICAL CONDITION missing,
                                           never a deadline

    reserved-taken  the name is taken so that nothing else may claim it.
                    OWES NOTHING; staying this way forever is correct.
                    Requires only a non-empty Reserved-Reason.

**A promise that never lands misleads; a marker that stays is honest.** Before
the split they read identically, so a promised `f128` and a blocked keyword
were indistinguishable.

`Reserved-Blocked-On` is a **condition, not a date**. `TyF128` is blocked on
`f128` emission in the x86-64 backend — without it the type exists and does not
descend. A reader then knows whether this is an afternoon or a year, and nobody
has to guess.

**The date is visible and does not expire.** Same rule the founder set for
`Evidence-Does-Not-Count`: nothing expires, nothing goes red with age, and the
gate **prints the active reservations with owner and age in days even when it
passes**, oldest first. Age derives from `Reserved-Since`, never from the git
log — otherwise editing the file for an unrelated reason erases the memory of
when the reservation was made, and visibility dies by maintenance.

**Both halves still owe a refusal.** A reserved name whose use is not refused
is not reserved; it is merely unimplemented, which is a different state. The
negative evidence requirement applies to `reserved-owed` and `reserved-taken`
alike.

**Malformed is redder than absent.** A `reserved-owed` missing Owner, Since or
Blocked-On fails, and the diagnostic names which of the three is missing. If
declaring badly were cheaper than declaring well, everyone would declare badly.

### What is not yet reserved, and should be

`i256`, `i512`, `u256` and `u512` are named in the founder's specification and
exist in **no enum at all** — not even reserved. Writing `i512` today produces a
generic unknown-type error, indistinguishable from a typo. Reserving them would
record the intent **inside the compiler** rather than only in a document, and
give them a diagnostic of their own. `i512` was named as the seed of the
Cayley-Dickson tower (`SOUNIO-PRECISION-PRESERVATION`).

Measured instance: `TyF128` and `TyF256` under Madaros. `E218` on bind, on
arithmetic, and on signature-only use.

## The test that decides a position

Write **two** programs: one that must pass and one that must fail.

| correct program | wrong program | position |
|---|---|---|
| passes | passes | label — nothing is being typed |
| **fails** | **fails** | **Reserved** |
| passes | fails | **Claim-ready** — the only case that is |
| passes | passes | `Executable`, not `Claim-ready` |

**Without both programs there is no position above `Hypothesis`.** One
program never decides. A single accepting witness cannot distinguish a type
from a label; a single refusal cannot distinguish a type from a wall.

## Why this belongs in governance, not in an audit

`FOUNDER_INTENT.md` asks what information was made invisible so the system
could look simpler. A four-state ladder was simpler than the language, and
the price was a kind recorded at the wrong rung. The state existed and had
even been observed — `f128/f256` had already been called "seed reservations"
in the known-failure census hours earlier — but had no standing, so the
observation could not be carried.

When a measurement produces an impossible position, suspect the ruler before
the measurement. The impossible position is a site.
