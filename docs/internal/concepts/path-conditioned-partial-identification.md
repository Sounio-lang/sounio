<!-- docs:meta
topic_id: repo.docs.internal.concepts.path-conditioned-partial-identification
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.path-conditioned-partial-identification
-->

# Path-Conditioned Partial Identification

Concept-ID: `SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION`

Status: executable bounded synthetic model.

Canonical surface:
`stdlib/epistemic/proof_carrying_path_conditioned_identification.sio`

The generated `docs:meta` validation date is the repository-wide documentation
governance baseline. It is not a D8 scientific sign-off, evidence date, or
authenticity claim; the executable acceptance surface below carries that role.

## Meaning

D8 represents latent-state identification as a path-conditioned set problem.
It deliberately refuses five common collapses:

1. equal scalar projections do not make latent states identical;
2. histories `AB` and `BA` do not become interchangeable because they contain
   the same action labels;
3. an exact identified set is not a confidence region, predictive set, value
   interval, p-box, credal set, or posterior;
4. association is not intervention or counterfactual evidence;
5. a synthetic suffering proxy is not human suffering or clinical authority.

The executable fixture contains three nominal states with the same scalar
projection. Under one frozen model and assumption set, the `AB` history admits
states `{A, B}` and the `BA` history admits `{B, C}`. A declared synthetic
history-invariant compatibility constraint `{A, C}` refines them to `{A}` and
`{C}` respectively. The two refined sets are therefore nonempty and exactly
separated inside this finite fixture.

## Exactness Boundary

`ExactABIdentifiedSetReceipt` and `ExactBAIdentifiedSetReceipt` mean exact
enumeration of all compatible members of the declared three-state family under
model `12200` and assumption set `12210`. They do not claim:

- a statistically sharp identified set in an unbounded population;
- estimator consistency, confidence coverage, calibration, or prediction;
- empirical adequacy of the model or assumptions;
- recovery of a patient's functional or psychiatric state.

The finite point receipts mean only that a refined finite set is a singleton.
They carry the source set, model, assumptions, ordered history, observation,
and synthetic provenance. They explicitly decline global functional-state,
causal, and clinical claims.

## Approximation Boundary

Forgetting exactness creates nominal `SoundOuterABIdentifiedSetReceipt` and
`SoundOuterBAIdentifiedSetReceipt` values. The masks happen to remain `{A, B}`
and `{B, C}` in the fixture, but the exact-enumeration warrant is not retained.

The outer-only view admits nine nonempty pairs of exact completions: four
overlap and five are disjoint. Consequently, overlap of those outers is
insufficient to decide whether their hidden exact completions overlap.
`OuterOnlyOverlapUndecidedReceipt` records precisely that deliberately
information-forgetting view. It does not erase the separate exact common-member
witness `{B}` already present in the full D8 evidence graph.

The parallel ontology classifies the available epistemic warrant. Its sibling
classes express nominal non-substitution in Sounio; they are not asserted as
OWL-disjoint classes and do not transport runtime kernel receipts.
The gate proves those rejections for the current source only. D8 does not claim
a sealed module theorem preventing a future owner from explicitly adding a
coercion or subtype relation.

## Provenance And Missingness

Every refining observation used by the positive fixture carries declared
synthetic policy-decision and observation provenance. The root and two children
are public, unsealed fixtures. The missingness and conflict children are
alternative branches from the same root, not one observed chronology.

`NotMeasuredReceipt` and `ObservedFalseReceipt` are nominally distinct.
Missingness under the declared policy preserves the previous identified set and
produces an abstention. A conflicting observation that eliminates every member
produces `ModelEvidenceConflictReceipt`; it does not select a nearest state or
silently repair the model.

Exact identity is carried by explicit IDs, masks, histories, model fields,
assumptions, and validators. Base-31 fingerprints and all printed checksums are
diagnostic only. They are not general injective encodings, cryptographic
commitments, authenticity evidence, or authority tokens.

## D7 Boundary

D8 consumes one D7-shaped local decision only to demonstrate refusal of reuse
at a different fixture occurrence. D7 supplies no `AB` or `BA` semantics, set
membership, exactness warrant, statistical theorem, source/IR rewrite
permission, compiler capability, or native `Contest` evidence.

The D8 construction performs no rebracketing and changes no compiler path.

## Causal And Clinical Boundary

The fixture can record that path identity is associated with different
compatible sets. It establishes neither an intervention contrast nor a
counterfactual. The history-invariant compatibility constraint is a synthetic
model restriction, not two jointly observed outcomes and not cross-world
evidence.

D8 proves no diagnosis, prognosis, treatment response, consent state, suffering
state, prescribing rule, or clinical action. Its purpose is to make accidental
promotion into those categories fail by type.

## Runtime And Ontology Boundary

The reusable kernel and imported API witness are current-source check evidence.
Runtime evidence comes from a standalone scalar mirror plus an independent
finite oracle. Imported multimodule execution remains outside the claim under
`BLK-20260718-D6-MULTIMODULE-RUNTIME`.

The ontology is a parallel nominal surface. `ontology_transport=0` remains part
of the executable receipt. Advisory science-boundary `UNKNOWN` output is not
counted as an authoritative scientific verdict.

## Acceptance Surface

The D8 gate must:

- select canonical Madaros and reject a requested legacy engine;
- typecheck the kernel, ontology, and imported witness;
- execute the native scalar mirror and the independent finite oracle;
- reproduce the frozen sets, refinements, provenance, missingness, conflict,
  separation, and approximation counts exactly;
- reject every nominal substitution across path, state, approximation,
  statistical, uncertainty, causal, contextual, suffering, and clinical
  categories;
- verify the concept registry and binding manifest;
- recursively keep D7 through D0 green.

## Pending Interface

`statistical-coverage-and-empirical-state-binding` remains pending. It would
require a declared sampling process, estimator and coverage semantics, external
data and custody, empirical model criticism, and sealed validation. None may be
inferred from the D8 synthetic receipts.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
