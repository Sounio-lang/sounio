<!-- docs:meta
topic_id: repo.docs.audit.zd-exactness-floating-point-boundary-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zd-exactness-floating-point-boundary-2026-08-19
-->

# The exactness boundary: formal at compile time, not numerical at run time

## Why this exists

An independent second opinion (Qwen 3.8 Max via `empryo`, run **blind** — the
parallel receipt `UNLEARNING_QUADRANT_2026-08-19.md` was neither shown nor
consulted) was asked one thing above all others: **is there a published reason
this may not work?**

There is, and it is direct. It does not refute the claim. It **bounds** it, and
the bound must be stated in the claim rather than discovered by a reviewer.

## The published objection

Jack Kowalski, *"Zero Divisors in Sedenion Algebra under Floating-Point
Arithmetic"* (entropment.com; **essay/preprint, not peer-reviewed**).

- Over exact ℝ, zero divisors of a Cayley–Dickson algebra are **stable algebraic
  points**: `a·b = 0` holds exactly, because the exact equalities hold across
  coupled components.
- Under IEEE-754, each sedenion → octonion → quaternion multiplication step
  introduces **independent rounding per operation**. The exact equalities
  degenerate into inequalities `|component| < ε`.
- Consequently zero divisors stop being fixed points and become numerical
  **orbits** that drift toward zero or are repelled from it depending on
  rounding history. Repeated multiplications **do not converge to exact zero**
  even where the real-algebra product is zero. The author frames the non-zero
  residue as an invariant of the projection history, not a defect.

Its non-peer-reviewed status does not weaken it much: the mechanism is
elementary and follows from IEEE-754 alone.

## What it does to the Sounio claim

`formal/lean4/SounioSurgicalInterventions.lean` proves exact annihilation by
`native_decide` over a **finite, exact model** — integers/rationals, or
axiomatic reals. That is sound *as a statement about the model*.

It does not transfer to `f64` execution. There, *"exactly zero in the
zero-divisor core"* becomes `≈ ε`.

> **The boundary of the claim is: formal exactness verified at compile time;
> numerical exactness not guaranteed at run time.**

**Sounio's own measurement reached the same place first.**
`ZD_ANNIHILATE_BUILTIN_DISPATCH_2026-08-19.md` (#2017) already recorded, as an
obligation fragment that escapes the checker: *"runtime `[f64;16]` weights equal
a stated kernel combination after float ops — floating-point; at best residual
bounds, not exact Lean equality."* The internal forensic and the external
literature agree, arrived at independently.

## What must therefore change in how the claim is stated

- **Not**: "the contribution is algebraically zero".
- **But**: "the contribution is algebraically zero *in the verified model*, and
  the compiled program is verified to perform the annihilating operation; the
  residual under IEEE-754 is bounded, not zero."

Under `SOUNIO-TYPE-INTERROGATION`'s naming clause this is not a retreat — it is
the clause working. The type names the proposition it interrogates, and the
proposition is the one that is decidable.

## A second bound, from the same source

The receipt also records a **retrain-equivalence impossibility for local
unlearning** (preprint, Oct 2025): retrain equivalence is unattainable for local
methods. This bounds the *no-retraining* qualifier and should be read together
with the exact-unlearning-for-restricted-models finding in
`UNLEARNING_QUADRANT_2026-08-19.md`.

## Convergence between the two independent measurements

Both were run blind of each other and agree that the three-qualifier conjunction
is unoccupied. The second opinion adds the two bounds above, which the first did
not surface because it was not asked to prioritise disconfirming evidence.

## Claims forbidden

- Stating exact annihilation as a **runtime** property of compiled Sounio.
- Reading "FREE" as "proven nonexistent". The consulted agent's own caveat: the
  sweep was arXiv plus indexed web, conference proceedings were not exhaustively
  covered, and the backend was rate-limited during part of the run.
