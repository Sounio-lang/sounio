<!-- docs:meta
topic_id: repo.docs.research.paper-a-priorart-gate-signoff-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-priorart-gate-signoff-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — prior-art gate sign-off (2026-08-25)

Closes the CANDIDATE status that synthesis §24/§25 attached to the narrow novelty claim,
subject to the residual stated at the end. Format follows the house prior-art memo
(`prior_art_k3_signed_switching_2026-08-04.md`).

## The claim under test

> *Sounio carries the source-identity of a value's uncertainty as a noise-symbol set in
> its type, and makes the independence assumption of uncertainty arithmetic a checked
> precondition: an independence-assuming add/merge over operands whose source-sets are
> non-disjoint (or unknown) is a compile-time type error unless a proved-disjoint
> certificate holds.*

The novelty asserted is **not** any of: noise-symbol tracking, correlation-aware
propagation, the Blackwell/anti-garbling frame, or uncertainty-in-the-type. It is their
**intersection**: compile-time *rejection* of an unsound-independence operator, driven by a
source-set carried *in the type* of an uncertainty-typed language.

## Gates run

**Gate 1 — affine arithmetic / static analysis.** Comba & Stolfi (1993) and Goubault &
Putot's Fluctuat (VMCAI 2011; arXiv:0807.2961; arXiv:1002.2236) track noise-symbol
correlations. **Result: kills the bare "we track source identity" claim; does not reach the
intersection** — Fluctuat is an external analyzer producing an enclosure, rejects nothing,
and puts nothing in a type.

**Gate 2 — QIF / Blackwell.** McIver–Morgan–Smith (POST 2014); Alvim et al. (SQIF, 2020)
identify program refinement with the Blackwell garbling order and use DPI-monotonicity as
soundness. **Result: kills the bare "we discovered the frame" claim; does not reach the
intersection** — QIF orders confidentiality channels and measures leakage, not variance
propagation, and is not a static typing rule at arithmetic operators.

**Gate 3 — adversarial deep pass (2026-08-25).** Two targeted searches for anything
occupying the intersection directly:
- *"type system reject correlated uncertainty propagation compile time covariance soundness"*
  — all hits are physical-domain covariance-propagation frameworks (aerospace TPA, GPS/Kalman,
  orbit determination, quantum noise); **none is a PL type system.**
- *"data processing inequality typing rule refinement type uncertainty variance monotonicity"*
  — DPI is well-documented information theory; refinement types exist; **no result ties DPI /
  the Blackwell order to a typing rule for uncertainty propagation.** The refinement-type hits
  concern monotonicity of the *typing relation*, unrelated to information monotonicity.

**Neighbours checked and distinguished (from §9):** Uncertain⟨T⟩ (ASPLOS 2014),
Measurements.jl, GUM tooling, Ferson p-boxes — carry uncertainty, do not track source
identity, do not reject. NumFuzz (2024), Bean (2025), type-based rounding-error analysis
(2025) — numeric-property-in-the-type, but the invariant is roundoff, not
correlation-soundness. IFC/taint types — the machinery, not the reading.

## Verdict

The narrow claim **survives all three gates** in the intersection form. No located work puts
a noise-symbol source-set in a type and rejects an independence-assuming operator at
compile time. The claim may be asserted in the paper as stated, with the mandatory
attributions to affine arithmetic (source identity) and QIF/Blackwell (the soundness frame)
carried up front (§1, §9) — the contribution is explicitly the intersection, not either
component.

## Residual (stated, not hidden)

- **Patent / grey literature not exhaustively searched.** Two USPTO hits surfaced in Gate 3
  concern declarative-data-scripting type checking, unrelated. A full patent clearance is out
  of scope for a research-novelty gate and is not claimed here.
- **"To our knowledge" retained.** Per house discipline the paper keeps the hedge; the gate
  raises confidence from CANDIDATE to *assertable-with-attribution*, not to
  exhaustively-certified.
- **The non-associative lift (Paper B) is a separate claim** with its own, larger open
  question (the octonion associator as the Blackwell obstruction) and is not covered by this
  sign-off.
