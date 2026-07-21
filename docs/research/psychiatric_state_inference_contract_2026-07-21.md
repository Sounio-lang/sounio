<!-- docs:meta
topic_id: repo.docs.research.psychiatric-state-inference-contract-2026-07-21
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.psychiatric-state-inference-contract-2026-07-21
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Psychiatric State Is Not Occupancy

Status: historical, source-backed research snapshot and implementation boundary

Date: 2026-07-21

Claim boundary: this is a design and evidence contract. It does not diagnose a
person, recommend a medication or dose, establish a biological mechanism, or
claim that any psychiatric system is octonionic. It specifies what a future
Sounio model would need to preserve before it could support a narrower research
claim.

## The Research Pressure

The useful idea is not that a medication has a mysterious or arbitrary effect.
It is that several distinct facts are routinely collapsed into one convenient
number:

```text
measured receptor occupancy
!= model-derived receptor binding
!= pathway-specific functional response
!= an individual's evolving functional state
!= an identified causal effect
!= a treatment recommendation
```

The same collapse appears in computational psychiatry when a cross-sectional
symptom graph, a time-series association, a hidden-state embedding, and an
interventional conclusion are presented as if they were interchangeable. They
are not. Sounio's possible contribution is to make the transitions explicit,
typed, and separately auditable.

## What the Literature Licenses

### Occupancy Is a Measurement Layer, Not a Functional-State Theorem

Yokoi et al. measured dose-dependent D2/D3 occupancy after aripiprazole in a
human PET study, including high striatal occupancy without the expected EPS
pattern in that sample. That is evidence about a PET occupancy relationship; it
does not make occupancy a complete functional-state variable.

- Yokoi et al. (2002), [D2/D3 PET occupancy study](https://pubmed.ncbi.nlm.nih.gov/12093598/).
- Burris et al. (2002), [high-affinity partial agonism at human D2 receptors](https://pubmed.ncbi.nlm.nih.gov/12065741/).
- Natesan et al. (2006), [occupancy and functional antagonism dissociation in animal models](https://pubmed.ncbi.nlm.nih.gov/16319908/).
- Gründer et al. (2022), [molecular imaging of dopamine partial agonists](https://pmc.ncbi.nlm.nih.gov/articles/PMC9020768/).

These sources motivate a modeling distinction among occupancy, intrinsic
efficacy, endogenous tone, pathway coupling, receptor regulation, time scale,
and measurement protocol. They do not license a patient-level inference from a
synthetic proxy or a declaration that a particular experience was caused by a
particular receptor state.

### Dynamic Networks Need Causal Discipline

The network approach is valuable because it treats symptoms and context as
potentially interacting over time. But the field's own reviews emphasize that
between-person and within-person networks are not interchangeable, that formal
causal theories remain scarce, and that cross-sectional edges do not identify
interventions. Recent longitudinal work in psychosis also found substantial
individual variability in network structure.

- Robinaugh et al. (2020), [network-psychopathology agenda](https://pmc.ncbi.nlm.nih.gov/articles/PMC7334828/).
- Forbes et al. (2019), [promise versus reality critique](https://pmc.ncbi.nlm.nih.gov/articles/PMC6732676/).
- Isvoranu et al. (2024), [mental-health network-analysis overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564129/).
- [Longitudinal psychosis network heterogeneity](https://pmc.ncbi.nlm.nih.gov/articles/PMC12229628/).

Therefore a Sounio graph or state-space fit must remain a model receipt until
its causal assumptions, intervention semantics, and validation regime are made
explicit. Association is useful evidence; it is not a hidden `do` operator.

### Dynamic Regimes Need Their Assumptions Attached

Dynamic treatment-regime (DTR) research gives the right formal shape for the
question "what action at this time, given this history?" It also makes clear why
an evolving history alone is insufficient: randomized sequential designs or
stated observational assumptions such as consistency, positivity, and no
unmeasured time-varying confounding are needed for causal interpretation.

- Chakraborty and Moodie (2013), [dynamic treatment regimes review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4231831/).
- Loh and Jorgensen (2025), [DTRs from longitudinal observational data](https://pubmed.ncbi.nlm.nih.gov/40048215/).
- [Time-varying optimal DTR tutorial](https://pmc.ncbi.nlm.nih.gov/articles/PMC11637529/).

This is a guardrail, not an obstacle. It tells Sounio that a proposed policy
must carry the history representation, target outcome, intervention definition,
assumption set, estimand, and validation receipt. A policy proposal that lacks
one of these is a simulation result, not a treatment rule.

### Control Theory Is a Probe, Not an Authorization

Control-theoretic work in psychological networks provides a useful language for
state transitions and intervention costs. It also explicitly warns that an
estimated network alone does not establish the most effective intervention.

- Henry, Robinaugh, and Fried (2022), [control of psychological networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC9205512/).
- Stiso et al. (2021), [methodological considerations for brain-network controllability](https://pmc.ncbi.nlm.nih.gov/articles/PMC7734595/).

Sounio may represent a control objective and its assumptions, but it must not
compile a controllability score into a clinical instruction.

## Proposed Semantic Decomposition

The following is a prospective taxonomy, not new Sounio syntax and not an
implemented standard-library API.

| Layer | What it may represent | What it must not silently become | Minimum receipt |
| --- | --- | --- | --- |
| `ObservationReceipt` | A measured PET, assay, EMA, scale, or sensor datum with protocol and uncertainty. | A latent state or causal effect. | instrument, protocol, time, uncertainty, provenance. |
| `ModelProjection` | A value computed from observations under named PK/PD or state-space assumptions. | A new observation. | model identity, parameters, numerical path, input receipts. |
| `FunctionalContextReceipt` | Inputs required to construct a bounded functional surrogate: occupancy, efficacy, endogenous tone, pathway/regulation context, and time. | A clinical outcome or a full mental state. | all declared context axes and missingness. |
| `FunctionalStateSurrogate` | The result of an explicitly named, bounded functional model. | A diagnosis, symptom report, or treatment effect. | context receipt, transform version, domain and uncertainty. |
| `TrajectoryReceipt` | Ordered, time-stamped and parenthesized evolution under declared updates. | A bag of endpoints or an order-free summary. | event order, grouping, time basis, state transitions. |
| `CausalEffectEstimate` | An identified interventional estimand under an SCM, target-trial, or stated DTR assumptions. | An observational association or a policy. | graph or trial emulation, estimand, identification assumptions, sensitivity analysis. |
| `DecisionProposal` | A research policy candidate with objective, harms, constraints, and abstention conditions. | An executable clinical directive. | utility/harms, eligibility, uncertainty threshold, external validation status. |
| `ClinicalValidationReceipt` | Evidence from an appropriate validation and governance process. | A compiler pass or model fit. | cohort/design, outcomes, calibration, oversight, limits. |

The word `surrogate` matters. A future executable model can be useful without
pretending that its internal scalar or vector is a person's complete state.

## Collision Matrix for the First Executable Bridge

The first bridge should remain synthetic and should prove separations, not
clinical predictions. Each row requires matching scalar observations but a
non-substitutable typed receipt.

| Collision | Hold fixed | Vary | Required result |
| --- | --- | --- | --- |
| Efficacy collision | Occupancy and endogenous tone | Intrinsic efficacy | Two functional-surrogate receipts differ. |
| Tone collision | Occupancy, ligand and intrinsic efficacy | Endogenous tone | Two functional-surrogate receipts differ. |
| Pathway collision | Occupancy and total scalar activation proxy | G-protein / beta-arrestin / regulation vector | Scalar collision cannot erase distinct pathway receipts. |
| Temporal collision | Initial and final projected scalars | Ordered events or parenthesization | Trajectory receipts remain distinct. |
| Causal-status collision | Same observational fit | Identification assumptions or intervention definition | An association cannot substitute for an identified causal effect. |

The existing bounded ontology witness already proves the smallest category
boundary: `ReceptorOccupancyObservation` is rejected where `FunctionalState` is
required. It must remain a negative control, not be overstated as a functional
pharmacology implementation.

## Evidence Ladder

```text
Garden intuition
  -> named synthetic surrogate and collision witnesses
  -> current-source compiler checks and native execution without fallback
  -> model calibration against an explicitly scoped experimental dataset
  -> preregistered longitudinal/site-aware evaluation and null controls
  -> causal-identification and sensitivity evidence for a specified estimand
  -> independently governed clinical validation
```

No arrow in this ladder is automatic. A compiler witness is evidence that the
representation was preserved on a named path; it is not evidence for the
biological model. A good predictive fit is not evidence that an intervention
would work. A valid causal estimand is not a prescription.

## Current Sounio Position

| Surface | Current evidence | Boundary |
| --- | --- | --- |
| `SOUNIO-ORDERED-PATH-PROVENANCE` | Bounded synthetic source-to-IR witness keeps order, grouping, observation, and functional-state categories distinct. | Not a clinical, pharmacological, or causal prediction. |
| `stdlib/darwin_pbpk/pd/d2_occupancy.sio` | A drug-modeling occupancy projection with named parameters and validation surfaces. | It is a model projection, not a complete functional-state model. |
| `stdlib/cybernetic/psychiatry.sio` | Exploratory cybernetic constructs for state, observation, and feedback. | It is not a validated clinical model or a causal treatment engine. |
| `SOUNIO-NONASSOCIATIVE-ORDER` | Order and grouping can be an explicit semantic object. | It does not prove that every psychiatric process is non-associative. |
| DTR / causal layer | Research direction only for this psychiatric bridge. | No causal-policy or clinical-decision claim is presently supported. |

The next technical dependency is compiler closure for imported native programs:
the Issue #901 complete-layout-catalog repair must pass the exact 255-custom
plus `Knowledge` / 256-custom plus `Knowledge` boundary and then D6/D11/D12 on
the same current-source compiler without fallback. That repair is an enabling
condition for larger synthetic witnesses, not evidence for this research
contract itself.

## Falsifiers and Failure Conditions

This proposal should be demoted or revised if any of the following occurs:

- an explicit functional model cannot produce a reproducible collision where
  occupancy is held fixed but declared context changes the surrogate;
- pathway-vector or trajectory distinctions add no measurable information in a
  preregistered assay beyond matched simpler controls;
- a proposed receipt cannot be independently reconstructed from its stated
  inputs and assumptions;
- a causal-effect type can be manufactured from observational association
  without explicit identification assumptions;
- a decision proposal can cross into a clinical directive without an
  abstention/validation boundary.

Negative results here are informative: they identify which distinctions do not
earn the cost of a first-class representation in a particular model.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: psychiatric-state-inference-research-v0
Owner: codex-root
Concept-IDs: SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-PHYSICAL-OBSERVATION; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: do not collapse observation, model projection, functional surrogate, trajectory, causal effect, policy proposal, and clinical validation into one value or claim
Transformation: none; this document maps a literature-backed research boundary onto existing Sounio concepts
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the cited literature and existing bounded witnesses justify a research requirement to keep the listed layers distinguishable
Claims-Forbidden: clinical efficacy, diagnostic utility, biological mechanism, individual treatment recommendation, causal effect from association, and general language support for the prospective taxonomy
Assumptions: cited literature is used as a design baseline; every future implementation supplies its own model, data, identification, and validation evidence
Write-Set: docs/research/psychiatric_state_inference_contract_2026-07-21.md
Read-Set: docs/internal/concepts/ordered-path-provenance.md; stdlib/darwin_pbpk/pd/d2_occupancy.sio; stdlib/cybernetic/psychiatry.sio; tests/compile-fail/ordered_path_occupancy_cannot_replace_state.sio
Positive-Witness: existing bounded compile-fail category rejection and ordered-path provenance gate; future synthetic collision matrix is not yet implemented
Negative-Witness: an occupancy observation fails where a functional state is required; no causal or clinical assertion is accepted from this document
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Integration-Target: internal research planning only
Authoritative-Only-If: the document remains aligned with the named current concepts and its forbidden claims remain intact
```

## Next Executable Bridge

After the imported-native catalog repair is accepted, add one isolated,
non-clinical source fixture and gate. It should construct the first four
collision rows above, expose the resulting receipts only through typed
continuations, and prove all of the following:

```text
same occupancy != same functional-surrogate receipt
same scalar functional proxy != same pathway receipt
same endpoints != same ordered trajectory receipt
observation != functional surrogate != causal effect != decision proposal
```

The fixture must use synthetic constants, no patient-level input, no dosing
recommendation, no hidden fallback, and an explicit error when a required
context axis is missing. Only after that gate is current-source native-green is
it sensible to discuss a larger package-level pharmacology model.
