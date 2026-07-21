# Typed State Inference: Implementation Roadmap

**Status:** internal implementation plan. It does not create a clinical
library, medical device, causal estimator, or authorization surface.

This roadmap turns the psychiatric research contracts into an implementable
Sounio direction without confusing compile-time category checks with scientific
validation. It is the bridge from formal distinctions to small, import-bearing
programs that can be falsified.

## 1. Language Scope

```text
observation             != state projection
state projection        != causal effect
causal effect           != unique decision
decision ambiguity      != data-acquisition authority
dyadic association      != causal interference
any research receipt    != clinical authorization
```

Sounio is not asked to decide what is true in psychiatry. It is asked to make
selected category substitutions impossible or visible in research code.

## 2. Four-Layer Library Shape

The implementation keeps generic evidence mechanics distinct from psychiatric
vocabulary.

| Layer | Future module family | Role | Forbidden promotion |
|---|---|---|---|
| Generic evidence core | `stdlib/research/evidence_*` | scope, provenance, assumptions, defeaters, abstention, evidence references. | domain truth or action authority. |
| Ordered systems core | `stdlib/research/ordered_*` | time-bounded observations, events, state projections, aggregation boundaries, model contests. | physical, psychological, or relational mechanism. |
| Scientific adapters | `stdlib/research/psychiatric_*` | functional-path, measurement, decision, dyadic, and interference receipt vocabularies. | validated psychiatry or patient-level result. |
| External authority boundary | no constructible library result | independently governed validation, consent, accountability, and authorization. | code-generated clinical authority. |

The first two layers are candidates for cross-domain reuse. The third is an
explicit research adapter. The fourth is deliberately outside the compiler.

## 3. Minimal Generic Core

The initial library should be nominal and small. It should not start by adding
grammar, effects, IR fields, or a hidden clinical ontology.

```text
EvidenceReferenceReceipt
    identity of an observation, analysis, test, or external record plus scope

AssumptionReceipt
    named assumption, domain, expiry/review condition, and sensitivity route

DefeaterReceipt
    counter-evidence, unresolved precondition, or domain mismatch

ClaimScopeReceipt
    allowed claim, target context, excluded claims, and invalidation condition

AbstentionReceipt
    missing prerequisite, reason, source provenance, and research review route

OrderedHistoryReceipt
    role/event/observation sequence with time references and, where grouping is
    used, an explicit reference to a separately justified aggregation boundary

ModelContestReceipt
    candidate representations, discriminating feature, evaluation plan, and
    result or discrepancy route
```

Ordinary nominal records plus constructors are sufficient for the first
API-boundary tests. They are not an unforgeable capability system. If nominal
construction is later demonstrated insufficient, module opacity or capability
design becomes a separate language question; it cannot be assumed for free.

## 4. Psychiatric Research Adapters

Adapters add domain-specific receipts while depending on the generic core rather
than reimplementing its provenance and abstention logic.

| Adapter | Required generic inputs | Result it may form | Result it may not form |
|---|---|---|---|
| Functional-path context | ordered history, observation scope, assay/model scope | `FunctionalPathStateReceipt` or abstention. | receptor mechanism or treatment result. |
| Temporal/measurement context | time origin, observation process, measurement model | comparable projection or measurement abstention. | identified effect. |
| Counterfactual context | question, assumptions, target context, selective-risk scope | bounded research candidate or abstention. | clinical authority. |
| Model adequacy context | model specification, synthetic regime, inference procedure, discrepancy plan | scoped recovery/calibration/adequacy receipt. | mechanism truth. |
| Nonregular decision context | contrast, uncertainty, regularity, competing-outcome basis | candidate set, unique-action abstention, information-value question. | treatment or acquisition order. |
| Dyadic context | membership, role, paired observation, interaction coding, interference scope | association or scoped spillover estimate/abstention. | relationship intervention authority. |

The table is a dependency graph, not a workflow for a clinician. Each adapter
can return an abstention that retains its missing prerequisite rather than
defaulting to a scalar, a zero, or a confident label.

### 4.1 Canonical Composition Rules

The generic core supplies shared *shape*, not a hidden supertype or an
implicit conversion lattice. A future adapter must name an explicit
scope-preserving constructor whenever it composes receipts from another
contract. Names that look nearby in prose are deliberately not aliases:

| Terms | Canonical distinction | Forbidden convenience conversion |
|---|---|---|
| `CausalEffectEstimate` / `SourceContextEffectEstimate` / `TargetContextEffectEstimate` | The first is the counterfactual contract's general identified-effect term. The latter two make the source or target context explicit. | No effect name may be silently widened, narrowed, or transported without a receipt that carries the corresponding context and transport assumptions. |
| `ResearchDecisionCandidate` / `DecisionCandidateSetReceipt` | The former is a bounded counterfactual research comparison with harms, eligibility, and selective-prediction inputs. The latter preserves unresolved multiplicity or preference ambiguity in a nonregular decision problem. | A candidate set is not a research decision candidate, unique action, or treatment instruction. |
| `ClinicalEvaluationReceipt` / `ClinicalValidationReceipt` / `ClinicalAuthorizationReceipt` | Evaluation names independently governed study/workflow evidence; validation names the broader empirical-and-governance evidence class; authorization names a role-bound permission to act. All remain external. | Neither evaluation nor validation, alone or together with compiler receipts, may construct authorization. |
| `AbstentionReceipt` / specialized abstentions | The generic receipt records the common refusal payload. A specialized receipt records its distinct failed prerequisite, such as noninvariance, nonregular uniqueness, acquisition, dyadic identification, model adequacy, or parenthesization. | A specialized abstention may not be dropped, by name alone, into an unrelated API or erased into a harmless null/zero result. |
| `OrderedHistoryReceipt` / `AggregationBoundaryReceipt` | A history records ordered evidence. A boundary explains why a particular grouping denotes a distinct intervention, reset, time scale, or aggregation rule. | An ordered sequence, even one that references a boundary, may not certify bracket sensitivity. |

For the initial nominal implementation, the safe representation is explicit
composition: a specialized record carries or references an
`AbstentionReceipt`, a context-scoped effect carries its scope receipt, and an
adapter constructor checks every named input. There is no ambient inheritance,
string-name matching, or automatic receipt coercion. A later trait, opaque
module, or capability design would need its own threat model and acceptance
tests before it changes this rule.

## 5. Staged Delivery

### Stage 0: Design Evidence

The research contracts describe receipt vocabularies, forbidden arrows,
discriminating synthetic cases, semantic boundaries, and explicit non-goals.
Existing direct single-module psychiatric controls demonstrate selected nominal
API boundaries only.

**Exit criterion:** documentation consistency passes and every planned receipt
has a stated scope plus a collision that would falsify a substitution.

### Stage 1: Compiler Trust Before New Imports

**Dependency:** #901 must produce one source-fresh Madaros ELF that compiles
and executes imported D11 and D12 witnesses with `rc=0`, with no fallback.

Same-file tests cannot establish that a receipt distinction survives module
import, merge, lowering, native emission, and runtime. No new psychiatric
imported library becomes accepted evidence before this gate is real.

### Stage 2: Generic Import-Bearing Core

Create only the generic evidence, abstention, and ordered-history library plus
a focused synthetic suite. Keep fixtures tiny: one imported definition module,
one positive constructor, and paired compile-fail callers for each forbidden
substitution.

```text
ObservationReceipt                 cannot satisfy CausalEffectEstimate
AssumptionReceipt                  cannot satisfy EvidenceReferenceReceipt
AbstentionReceipt                  cannot satisfy ResearchDecisionCandidate
OrderedHistoryReceipt              cannot satisfy AggregationBoundaryReceipt
EvidenceReferenceReceipt           cannot satisfy ClinicalAuthorizationReceipt
DecisionCandidateSetReceipt        cannot satisfy ResearchDecisionCandidate
ClinicalValidationReceipt          cannot satisfy ClinicalAuthorizationReceipt
```

All values are synthetic tokens. There is no patient data, identity, medication,
diagnosis, clinical action, or performance claim.

### Stage 3: One Adapter At A Time

Implement adapters in this order because each has a smaller prerequisite set
than the next:

1. temporal/measurement receipt separation;
2. functional-path versus scalar-proxy separation;
3. nonregular decision and information-acquisition separation;
4. dyadic association versus interference separation;
5. counterfactual/transport/authority composition.

Each adapter gets one import-bearing positive and paired negatives. A passing
test proves only the named API distinction, not empirical or clinical status.

### Stage 4: Synthetic Model Contests

After receipt boundaries are stable, add deterministic synthetic contest
harnesses. The aim is to reject a representation that loses a declared feature.

```text
argmax model                  fails a near-zero/nonregular discriminator
individual-only model         fails a declared dyadic-history discriminator
scalar activation proxy       fails a functional-path discriminator
ordered associative model     defeats a nonassociative model when bracketing
                              adds no discriminating feature
```

Here Sounio's special value becomes concrete: code must name the evidence and
representation boundary it would otherwise erase.

### Stage 5: Empirical And Authority Work Stay External

Empirical datasets, protocol approval, consent, data linkage, causal study
design, real-population calibration, clinical evaluation, and any authority to
act are separately governed. No Stage 1-4 compiler success changes that status.

## 6. Cross-Domain Reuse Without Ontology Leakage

The generic core is reusable because its meanings are methodological:
observation, scope, assumption, defeater, ordered history, model contest, and
abstention. Each domain retains its own adapter and authority boundary.

| Domain | Legitimate adapter distinction | Forbidden import |
|---|---|---|
| Chemical process research | measured concentration versus mechanism or process-control conclusion. | psychiatric state labels or clinical authority vocabulary. |
| Architecture and built environment | inspection observation versus compliance or occupancy-safety conclusion. | treatment, diagnosis, or interpersonal causal labels. |
| Legal research workflow | document provenance versus a scoped research question or legal conclusion. | legal authority from a generic receipt. |
| Machine behavior research | telemetry observation versus substrate stress, sentience, or moral-status claim. | biological or clinical causality. |

The language provides reusable discipline, not a universal ontology.

## 7. Promotion Gates

| Promotion | Required evidence | Refuse promotion when |
|---|---|---|
| Direct fixture to imported fixture | source-fresh #901 imported runtime acceptance. | only same-file/default-wrapper evidence exists. |
| Nominal boundary to stronger capability claim | explicit attack test, then separately designed opaque/capability solution. | no demonstrated threat model exists. |
| Synthetic contest to empirical research | preregistered protocol, measurement/observation plan, and suitable governance. | synthetic success is described as real-world validity. |
| Research result to action authority | external validation and accountable authorization outside Sounio. | a score, receipt, binary, or documentation is offered as authority. |

## 8. Semantic-Lane Declaration

```text
Semantic-Lane-ID: psychiatric-typed-state-inference-roadmap-v0
Owner: Codex psychiatric state-inference research lane
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-ORDERED-PATH-PROVENANCE; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: Sounio can make evidence and category boundaries explicit without claiming that types establish science, causality, clinical validity, or authority
Transformation: compose existing research contracts into an implementation order with import-bearing acceptance prerequisites and stage-specific stop conditions
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a generic evidence/abstention/ordered-history core and staged scientific adapters are the proposed first implementation shape after #901 acceptance
Claims-Forbidden: new grammar requirement; unforgeable capability claim from nominal records; empirical or clinical result from synthetic fixtures; cross-domain ontology equivalence; clinical or legal authority from code
Assumptions: current contracts remain aligned with their sources and the source-fresh #901 gate can eventually establish imported runtime integrity for selected fixtures
Write-Set: docs/research/psychiatric_typed_state_inference_implementation_roadmap_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_ACCEPTANCE_REPORT.md
Read-Set: FOUNDER_INTENT.md; AGENTS.md; existing psychiatric research contracts; docs/internal/concepts/{science-research-boundary,epistemic-numeric-value,ordered-path-provenance,nonassociative-order}.md
Positive-Witness: future source-fresh imported generic evidence-core fixture with a positive constructor and category-specific compile failures
Negative-Witness: attempt to promote a direct fixture, synthetic contest, nominal record, or research receipt into a broader empirical or authority claim
Acceptance-Gate: bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; source-fresh #901 D11/D12 imported compile-and-execute gate before Stage 2
Integration-Target: internal research planning, then staged #901-gated library work
Authoritative-Only-If: contracts, compiler provenance, fixtures, and claimed evidence levels remain aligned
```

```text
Semantic-Outcome: a staged route exists from evidence-bound research documentation to a reusable typed library without eliding external empirical and authority boundaries
Concept-Status-Before: psychiatric contracts specified individual boundaries but no unified library layering or delivery sequence
Concept-Status-After: generic core, domain adapters, external authority, test ladders, and promotion gates are explicit
Distinctions-Added: generic method != domain ontology; direct fixture != imported proof; nominal boundary != unforgeable capability; synthetic contest != empirical validation
Distinctions-Preserved: research model != empirical result; causal estimand != authority; compiler success != clinical validation
Distinctions-Erased: none
Evidence-Run: repository inventory; git diff --check; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no fallback compiler or direct-only fixture is evidence for imported runtime or clinical authority
Legacy-Kept: existing psychiatric research contracts and workflow fixtures remain unchanged
Conflicting-Lanes: #901 compiler and generated governance metadata remain separately owned
Next-Semantic-Interface: source-fresh generic evidence-core import fixture after #901 acceptance
```
