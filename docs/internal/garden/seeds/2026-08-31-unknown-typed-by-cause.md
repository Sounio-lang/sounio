<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-31-unknown-typed-by-cause
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-31-unknown-typed-by-cause
-->

# The unknown, typed by cause — NaN's payload bits, finally used

> **Status**: Garden seed | **Last validated**: 2026-08-31 | **Source**: kimi-cli2 conversation, third seed of the arc (after Pharos 2026-08-29 and the convergence detector 2026-08-30)

## Butterfly

> o desconhecido tipado por causa — a fraude epistêmica mais comum da ciência é ausência de evidência virando evidência de ausência

The phrase survived from the butterfly list: missing data collapses into
zero, `null`, or a silent imputation — and the *reason* it is missing, which
is itself decision-relevant information, is erased at the moment of collapse.
The founder's erasure list already names the fraud ("absence of evidence
reported as evidence of absence"); this butterfly is its type-system form.

## Core Idea

**The cause of not-knowing is information, and it should be typed.** A missing
value is not a smaller kind of value; it is a different kind of object whose
*admissible operations depend on the cause*:

- `Unmeasured` — nobody looked; fillable by a measurement plan.
- `Withheld` — someone has it and will not share; fillable by negotiation or
  legal process, not by computation.
- `Redacted` — existed, deliberately removed; the removal is a fact about the
  world that must survive.
- `Lost` — existed, destroyed; unfillable, but its past existence matters.
- `Expired` — was known; knowledge decayed past its validity (converges with
  the claims-that-decay butterfly and the `ValidUntil` annotation that parses
  but is never enforced — see the observer surface survey).
- `NeverExisted` — unmeasurable *in principle*: the counterfactual. "What
  would this patient's outcome have been without treatment" is not unmeasured;
  it is unmeasurable. Causal inference is the discipline of reconstructing
  `NeverExisted` under declared assumptions — a different epistemic act from
  measuring, and the type system should say so.

The rule that gives the butterfly teeth: **a computation over an
`Unknown<cause>` cannot produce a `Measured` result without an explicit,
cause-appropriate resolution construct** — and zero-filling is not a
resolution. This is NaN done right: NaN propagates but carries no cause and
offers no way to refuse.

Two historical anchors make this less speculative than it sounds:

1. **Rubin's missing-data taxonomy (1976)** — MCAR/MAR/MNAR — is the mature
   statistical version of this butterfly: the missingness *mechanism*
   determines what inference is legitimate. It lives in textbooks and
   reviewers' checklists, never in the type of the value.
2. **IEEE 754 reserved the channel and nobody used it.** A NaN carries 51
   payload bits (22 for 32-bit) — space in the hardware encoding explicitly
   reserved for *why this is not a number*. Decades of software write 0 or
   the quiet-NaN default into them. The hardware already believes the cause
   matters; the ecosystem never showed up.

The dual with the observer butterfly is exact: the observer is the provenance
of *knowledge*; this is the provenance of *ignorance*. A complete epistemic
type system tracks both what you know-with-who and what you
don't-know-with-why.

## Connections

- [`2026-07-11-the-zero-of-encounter.md`](2026-07-11-the-zero-of-encounter.md) —
  the existing seed separating kinds of zero (absence, cancellation,
  annihilation, resolution, rounding); this butterfly extends it from kinds
  of zero to kinds of *absence*.
- `artifacts/audit/observer_surface_survey_20260831.md` — the dual survey:
  where *producer* identity lives and dies in the stack. The dead
  `source_id` socket found there (`check/epistemic.sio:225-245`) is the same
  shape of unwired organ this butterfly would need for causes.
- [`FOUNDER_INTENT.md`](../../../FOUNDER_INTENT.md) — the erasure list, and
  the clinical domain where the cause decides the correct action (a missing
  creatinine that is `Unmeasured` says *order the test*; one that is
  `Withheld` says something else entirely).
- External prior art to confront: Rubin (1976) MCAR/MAR/MNAR; SQL's NULL
  (Codd wanted two nulls — missing/applicable vs missing/inapplicable; the
  standard collapsed them and bought forty years of three-valued-logic pain);
  IEEE 754 NaN payloads; option types (`None` carries no cause).

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Captured: the phrase, the six-cause taxonomy, the resolution rule, the two historical anchors, the observer duality. |
| `Hypothesis` | A cause-carrying `Unknown` type with cause-appropriate resolution constructs is expressible in the existing Knowledge machinery; causes must propagate through computation like noise-sets do in the NS calculus. |
| `Executable` | None. No probe, no witness, no syntax exists. |
| `Claim-ready` | No. |

## What This Is Not

- Not a claim that typing missingness makes MNAR tractable — Rubin's point is
  that some mechanisms defeat inference without extra assumptions. The type
  can force honesty; it cannot force solvability.
- Not a claim that IEEE NaN payloads were "meant" for this; they are a
  reserved channel, and this is one honest use.
- Not a critique of `Option`/`Maybe` in general — they answer "is it there?",
  a different question from "why not?".
- Not a design document; the taxonomy is a sketch, not a lattice.

## Next Executable Bridge

The **laundering witness**: a program that sums a measured value with an
`Unknown(Unmeasured)` and attempts to produce a `Measured` result via an
implicit default — expected compile-fail. The dual probe: the same program
with an explicit `resolve_by_plan` construct — expected run-pass. Two small
programs that fix the entire semantics of the idea before any design
discussion. (Requires a from-source compiler build; was blocked by the
fleet build-lock deadlock, which has since cleared.)

## Session Record

Planted 2026-08-31 (UTC), third seed of one continuous kimi-cli2 conversation.
Same discipline as its siblings: question fixed before any executable, no
code, no expected values, no novelty claim over Rubin, Codd, or IEEE 754.
