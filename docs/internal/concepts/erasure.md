<!-- docs:meta
topic_id: repo.docs.internal.concepts.erasure
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.erasure
-->

# Epistemic Erasure

Concept-ID: `SOUNIO-EPISTEMIC-ERASURE`

Status: **Hypothesis** — designed by the founder in session on 2026-08-19,
recorded here before implementation. No compiler surface implements it, and
no fixture pair exists. Under the monotone maturity ladder
(`MATURITY_LADDER.md`) this may not be described as Executable or
Claim-ready until a correct program passes and a wrong program is refused.

## Founder Intent

Sounio's epistemic effect vocabulary points in one direction. `Observe`,
`Learn`, `Witness`, `Prob` and `Audit` all name ways knowledge is **acquired**.
Not one of them names the inverse.

The founder's ruling, stated in session:

> The compiler is not failing. The program was allowed to say it.

The operation that destroys epistemic content is today legal, unmarked, and
invisible. It is spelled `.value`.

```sounio
let k = measure(2.0, uncertainty: 2.0)
let r = combinar(k.value, x, y)     // uncertainty gone, silently
```

`Knowledge` is a plain struct — `value`, `variance`, `confidence` — and is not
linear. Dropping one requires nothing. `.value` yields an `f64`, and an `f64`
has no uncertainty to lose, so every layer below behaves correctly while the
claim quietly stops being a claim.

## The erasure this prevents

From `FOUNDER_INTENT.md`: *"a small or unresolved effect reported as zero"*.
This is that erasure at its source, in the core epistemic type.

It is important that the FO variance defect
(`docs/audit/FO_VARIANCE_ACROSS_FN_INDEPENDENT_VERIFY_2026-08-18.md`, live on
46 of 51 dissertation surfaces) is a **symptom, not the disease**. Repairing
the codegen so variance survives a call does not make the line above safe: the
projection still discards the uncertainty, and `0.000000` remains the honest
answer to a question the program had already thrown away.

## Design

**Projection taints.** `k.value` does not yield a plain `f64`. It yields a
value that remembers its uncertainty was discarded, and arithmetic over it
stays marked. This is information-flow typing applied to uncertainty, and it
is the only formulation that refuses the FO class **at compile time** rather
than when someone runs the right witness.

**The annotation burden falls on the claim, not the plumbing.** The programmer
never writes the marked type; the checker infers it and stays silent.
Functions that sum, iterate, sort or index declare nothing and keep saying
`f64`. Only functions that **assert** something epistemic declare that they
require clean input. Taint systems die on migration when every intermediate
must be annotated; here the boundary set is small and is itself the
specification of what counts as a claim in Sounio.

**Declassification is an act, not a coercion.** Returning knowledge to a bare
number requires `attest(v, uncertainty: u, provenance: <provenance>)` — an
assertion someone signs, never a cast. Every site where uncertainty was
restored by hand is therefore visible.

## The four sinks

A marked value is refused at exactly the four boundaries where a number stops
being plumbing and becomes an assertion. The founder closed all four:

1. **Reading uncertainty** — `variance_of`, `gum_k95`, confidence, coverage
   interval. Asking the uncertainty of a value whose uncertainty was erased
   returns `0.000000` today; it must refuse. This is the minimum: without it
   the FO class can still print a zero that looks like a perfect measurement.
2. **Leaving a public stdlib function** — the boundary where knowledge passes
   from whoever computed it to whoever will use it without knowing how.
3. **Compared against a test tolerance** — a test that passes by comparing a
   value which already lost its uncertainty has validated the arithmetic and
   pretended to validate the science. This is the sink that stops a green gate
   from lying.
4. **Printed or written out** — `println` of a value presented as a
   measurement, or a write to a results file. After this the number leaves the
   program and no reader can tell the uncertainty was erased.

## Required Invariants

- Projection out of an epistemic type is never silent. It either propagates a
  mark or is refused; it does not yield an indistinguishable bare number.
- A marked value is refused at all four sinks. Closing three is not a weaker
  version of this concept; it is a different concept, because the open sink is
  where the claim escapes.
- Restoring uncertainty requires provenance. A function that attaches
  uncertainty to a bare number without a source is a back door and must be
  named as one.
- The marked type is **inferred, never required in plumbing signatures**. A
  design that forces intermediate functions to annotate has failed its own
  adoption test regardless of soundness.
- Repairing variance propagation does not close this concept. The two are
  independent: FO is about whether the value survives; erasure is about
  whether discarding it may be silent.

## Claims Forbidden

- Do not describe this as implemented, partially implemented, or as a
  compiler behaviour. Nothing in `self-hosted/` reads this document.
- Do not present the FO variance work as closing this concept, or this concept
  as closing FO.
- Do not quote the sink list as enforced anywhere. It is a design decision of
  record, and its enforcement does not yet exist.

## Related

- `SOUNIO-EXACTNESS` — the sibling erasure: a decided fact narrowed to a
  measurement. Both concern a qualitative property lost without a mark.
- `MATURITY_LADDER.md` — why this document is Hypothesis and what would move it.
