<!-- docs:meta
topic_id: repo.docs.internal.concepts.signal-direction
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.signal-direction
-->

# Signal Direction

Concept-ID: `SOUNIO-SIGNAL-DIRECTION`

Status: **Hypothesis** — the failure it describes was measured today; nothing
distinguishes the two directions yet.

## Founder Intent

> A progress signal must not be indistinguishable from a regression signal.

The repository already knows how to say *"a claim about this engine has become
false"*. It has exactly one way to say it: **red**. And it uses the same red
whether the claim became false because the compiler got worse, or because it
got **better**.

## The measurement that produced this

`main` was red for **nine hours** on 2026-08-19, on
`Madaros Current-Source f64 Lowering`. Twelve cancelled runs, seven failures,
zero successes. Four PRs were held behind it. A lane was dispatched to bisect
two candidate commits, and a no-revert rule was broadcast to seven lanes to
protect one of them.

The job log said:

```
Fail: 0
XPAS  gum_fo_import_boundary.sio (known failure now passes)
Unexpected passes (stale known-failure): 1
XPAS_FATAL: 1 known-failure tag(s) passed on this engine
Process completed with exit code 1
```

**Zero failures.** Nothing had broken. A witness marked `//@ known-failure`
had **passed** — and `gum_fo_import_boundary` is first-order variance across an
import boundary, exactly what `7be969ed05` (#1939, *"imported 1-2 arg helpers
keep first-order variance"*) had repaired.

main was red because **someone fixed a bug and nobody removed the label**.

The XPASS gate (#1910) behaved exactly as designed: a known-failure that passes
is a stale claim, and a stale claim is a defect. The mechanism was armed the
previous night precisely so the repository would announce the day the FO blocker
fell. **It announced. And the announcement was read as damage** — by the same
person who armed it.

## Why this is a language concern and not a CI preference

`SOUNIO-NO-IMPLICIT-DEGRADATION` says nothing may be lost in silence. This is
its mirror: **nothing may be gained in a way that reads as loss.** A system that
reports improvement and breakage identically forces every reader to re-derive
which one happened, and the cost of getting it wrong is asymmetric — a
regression read as progress ships a defect; progress read as a regression spends
nine hours hunting a culprit that does not exist and risks reverting the fix.

That second failure is the one that occurred, and it nearly took the repair with
it. The no-revert rule broadcast that day exists because the fix was one command
away from being undone as collateral.

## Required Invariants

- A stale claim caused by improvement and a genuine regression are **different
  verdicts**, whatever their exit code. Where a runner offers only pass/fail, the
  distinction must be in the first line of output, not buried.
- The direction must be derivable without reading source. `XPAS_FATAL` names the
  mechanism but not the direction; a reader learns the run improved only by
  opening the test and the commit that touched it.
- Blocking is orthogonal to direction. A progress signal **may** legitimately
  block a merge — a stale claim is still wrong — but it must not be *reported*
  as breakage.
- The pair matters more than either half. A gate that only announces regression
  makes progress invisible; the repository then has no record of the day a
  blocker fell, and cannot tell whether it is getting better.

## Claims Forbidden

- Do not read this as a criticism of the XPASS gate. It did exactly what it was
  designed to do; the defect is that its output does not carry direction.
- Do not treat "make it green" as the fix. The stale label is a real defect and
  must be removed with evidence, not silenced.
- Do not conclude that FO is repaired. The witness passing may mean the fix is
  real **or** that the witness stopped measuring what it claimed to. That
  distinction is under measurement and is not settled here.
- Do not implement direction by guessing. A rule inferring "improvement" from
  `Fail: 0` plus an XPASS would be a heuristic, and a heuristic that misreports
  direction is worse than no direction at all.

## Related

- `SOUNIO-NO-IMPLICIT-DEGRADATION` — the same principle, opposite sign
- `MATURITY_LADDER` — a stale known-failure is a rung claim that reality passed
