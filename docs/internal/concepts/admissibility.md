<!-- docs:meta
topic_id: repo.docs.internal.concepts.admissibility
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.admissibility
-->

# Admissibility

Concept-ID: `SOUNIO-ADMISSIBILITY`

Status: **Hypothesis** — the coupling ruled by the founder on 2026-08-19 is not
implemented. The *types* below are live and tested; what does not exist is the
link between them and `SOUNIO-EPISTEMIC-ERASURE`.

## Founder Intent

Sounio's epistemic work has two halves, and only one of them had a
specification.

**Degradation** — how knowledge is lost — was specified on 2026-08-19:
`SOUNIO-EPISTEMIC-ERASURE`, `SOUNIO-NO-IMPLICIT-DEGRADATION`,
`SOUNIO-PROVENANCE`, `SOUNIO-JUSTIFICATION`. That is the **input** side.

**Admissibility** — what one is permitted to *do* with the knowledge that
remains — is the **output** side. It was implemented first and specified never.

Founder ruling: **`Admissible<T>` requires non-degraded input.** An
unjustified degradation anywhere in the chain makes the decision inadmissible.
Not a warning, not a record on the side: the decision does not typecheck.

## Measured state (2026-08-19, `origin/main`)

Ten decision- and causality-typed kinds are live in the checker
(`self-hosted/check/types.sio:55-65`, constructors at 1064–1530):

| kind | stated role | test/example files |
|---|---|---:|
| `TyContest` | contest carrier | **122** |
| `TyDeferred` | `Deferred<T>` — explicit action deferral certificate | **52** |
| `TyRobust` | robustness carrier | 28 |
| `TyAdmissible` | `Admissible<T>` — decision-admissibility proof carrier | 16 |
| `TyIntervention` | `Intervention<T>` — causal `do(X=x)` | 13 |
| `TyCounterfactual` | `Counterfactual<T>` — causal `Y(x)` | 12 |
| `TyValidated` | `Validated<T>` — compile-time validated | 11 |
| `TyRollbackCertificate` | rollback certificate | 4 |
| `TyDecisionPolicy` | decision policy | 1 |
| `TyDeferralPolicy` | deferral policy | — |

**These are not `Garden`.** They are implemented and exercised.

What is missing is not implementation but **a place where they are defined**.
No concept document is dedicated to any of them; they are mentioned across
others — `contest` in 10 documents, `intervention` in 5, `admissib` in 3. That
is dispersion, which is the state from which documents contradict one another
without anyone noticing.

## The joint

An `Admissible<T>` asserts a decision is admissible. **Admissible given what?**
If the value behind it passed through a `.value` and lost its uncertainty, the
admissibility was computed over a number that is no longer knowledge. Today
nothing connects the two: `Admissible` cannot see that its input was degraded,
and the erasure machinery does not know a decision is downstream.

The ruling closes that: an unjustified degradation in the chain makes the
decision inadmissible. `attest(v, uncertainty:, because:)` remains the way
through, and its floor is a discharged proof (`SOUNIO-JUSTIFICATION`) — so a
chain that legitimately strips uncertainty must say so with a theorem rather
than by silence.

**Deciding is the fifth sink.** `SOUNIO-EPISTEMIC-ERASURE` names four
boundaries where a marked value is refused: reading uncertainty, leaving a
public stdlib function, comparison against a test tolerance, and printing or
writing out. All four concern **reporting** a number. This one concerns
**acting** on it, and it is the one that reaches the world.

## The row that was in the wrong table

`SOUNIO-NO-IMPLICIT-DEGRADATION` lists *evaluation outside the validated
domain* as a degradation with no act. **It belongs here instead.** A PBPK model
run at a dose outside its calibration has lost no information — the information
is intact. What fails is that the decision is not admissible at that point.
Degradation is about the information; admissibility is about the domain and the
policy. Filing it as degradation is what left it without an act for so long.

## Required Invariants

- Admissibility is computed over knowledge, not over numbers. A decision whose
  support was degraded without justification does not typecheck.
- The escape is `attest` with provenance, and its floor is a proof. There is no
  weaker way past this boundary, because a decision is where a wrong number
  stops being a printed digit and becomes an action.
- Admissibility and degradation are **not** the same axis and neither subsumes
  the other. Both must be green: the information survived, **and** the decision
  fits the domain.
- A concept scattered across ten documents has no definition. These kinds get
  one place, and the ten mentions defer to it.

## Claims Forbidden

- Do not describe the coupling as implemented. `Admissible<T>` performs no
  check on the epistemic history of its input today.
- Do not read the test-file counts as evidence that the kinds are
  `Claim-ready`. Counting files that mention a name measures reach, not the
  two-program test.
- Do not treat this document as a specification of what each of the ten kinds
  *means*. It specifies the **joint**; the individual definitions do not exist
  yet and their absence is the finding.
- Do not move *evaluation outside the validated domain* out of the degradation
  table until an act exists for it here. Removing it from one table without
  landing it in the other loses it entirely.

## Related

- `SOUNIO-EPISTEMIC-ERASURE` — the input half; deciding becomes its fifth sink
- `SOUNIO-JUSTIFICATION` — supplies the only way past the boundary
- `SOUNIO-NO-IMPLICIT-DEGRADATION` — holds the misfiled row
