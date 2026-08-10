<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-endogenous-observability-d4-2026-07-15
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-endogenous-observability-d4-2026-07-15
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Endogenous Observability D4

Status: frozen bounded synthetic specification, 2026-07-15.

## Thesis

D4 tests a programming-language proposition:

> When state or policy affects whether, when, and how a measurement exists,
> the production history of the observable belongs to the evidence. Absence
> may update a declared observation-process contest, but it cannot become a
> target value, an identified missingness mechanism, or clinical authority.

The key object is not a richer nullable scalar. It is a typed custody chain:

```text
measurement intent
-> observation-policy decision
-> delivery event
-> response opportunity
-> response timing
-> observed value | typed window nonresponse
```

This is an exact finite software fixture, not a model of a patient. In D4,
**proof-carrying** means that every bounded contest transition retains enough
data for a consumer to replay the prediction mask, provenance recurrence, and
declared burden at runtime. It claims no Lean theorem, compiler metatheorem,
cryptographic authenticity, sealed constructors, physical measurement, or
clinical validation.

## Semantic lane declaration

```text
Semantic-Lane-ID: psychiatric-regimes-d4-proof-carrying-endogenous-observability
Owner: Codex implementation under founder direction
Concept-IDs: SOUNIO-ENDOGENOUS-OBSERVABILITY; SOUNIO-REFLEXIVE-INQUIRY; SOUNIO-PROOF-CARRYING-INFERENCE
Intent-Preserved: value absence, opportunity absence, policy withholding, delivery failure, delayed response, provenance, ambiguity, and claim level cannot collapse silently
Transformation: D3 passive missing evidence becomes a typed observation-process contest
Types-Changed: new stdlib and ontology types only
Effects-Changed: none in compiler IR; observation-process receipts are library-level executable artifacts
IR-Changed: none
Claims-Introduced: one bounded fixture proves a custody partition, an observational-equivalence collision, and within-family retry discrimination
Claims-Forbidden: MAR/MNAR identification; global recoverability; biological mechanism; real-person randomization; consent; physical observation; suffering; diagnosis; prognosis; treatment; clinical authority; historical priority
Assumptions: frozen four-hypothesis family; exact integer/Boolean traces; deterministic synthetic retry predictions; exact provenance; no stochastic noise; no real person
Write-Set: D4 kernel, ontology, witnesses, negative fixtures, oracle, gate, concept contract, registry, this specification, offload log
Read-Set: D0-D3 epistemic and ontology surfaces; missingness identifiability; informative observation times; EMA response timing; selective labels; active measurement
Positive-Witness: full custody maps mask 15 to mask 6; synthetic retry response maps 6 to 2; synthetic retry nonresponse maps 6 to 4
Negative-Witness: missing cannot become value; equivalence cannot become recoverability; ambiguity cannot become mechanism; policy-erased and disconnected evidence abstain
Acceptance-Gate: scripts/ci/proof_carrying_endogenous_observability_gate.sh
Integration-Target: research/psychiatric-regime-contest-20260712
Authoritative-Only-If: canonical Madaros checks reusable surfaces, native and ontology witnesses execute, independent exhaustive oracle agrees, negatives reject, and D3-D0 regressions remain green
```

The ontology portion of this acceptance boundary is category-level and
parallel. D4's executable kernel returns ordinary receipt structs, while the
ontology module and focused fixtures independently encode corresponding nominal
non-subsumptions. No kernel-produced D4 value is currently carried into IR as
an ontology-typed result, and the gate does not imply such transport.

The bounded base-31 provenance fingerprint is an audit convenience. It is not
collision-free identity, cryptographic authentication, or a substitute for the
exact predecessor and provenance fields replayed by each transition.

## Literature compass

D4 combines established results whose conjunction exposes a useful PL gap. It
does not claim priority over any field in the table.

| Literature | Established result | D4's narrower PL question |
|---|---|---|
| Informative observation times | Longitudinal visit timing can depend on the outcome process; joint models explicitly represent the longitudinal and observation-time processes. <https://pubmed.ncbi.nlm.nih.gov/18759841/> | Can a language prevent an observed value from losing the visit policy and opportunity chain that made it observable? |
| Informative visiting in EHR data | Simulations show bias when an informative clinical visiting process is ignored, and distinguish visiting-at-random assumptions from joint modeling. <https://pmc.ncbi.nlm.nih.gov/articles/PMC6919310/> | Can policy and delivery remain typed provenance rather than invisible covariates? |
| EMA response latency | In 339 participants prompted five times daily for seven days, higher activity at the prompt predicted delayed response; activity fell before response. The study did not find significant systematic bias in reported activity, but did find more random error. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11739121/> | Can prompt time, window close, and arrival time remain distinct without rewriting a late value into the prompt state? |
| Momentary nonresponse predictors | In an EMA study of binge-eating disorder, lower positive affect, lower hunger, later signals, later study days, and a previously missed signal predicted nonresponse. <https://pubmed.ncbi.nlm.nih.gov/33905971/> | Can nonresponse update only models that explicitly predict the response process, without becoming a symptom score? |
| Observational equivalence of missingness models | Molenberghs and colleagues prove that every MNAR model has a MAR counterpart with equal fit to observed data, although their predictions for missing values can differ. <https://doi.org/10.1111/j.1467-9868.2007.00640.x> | Can the program return an equivalence class and demand extra assumptions or evidence rather than declaring a mechanism from fit? |
| Graphical missing-data theory | Recoverability and testability depend on the missingness graph and target query; some claims are recoverable, some testable, and others neither. <https://proceedings.mlr.press/v33/mohan14.html> <https://proceedings.mlr.press/v72/mohan18a.html> | Can recoverability be a separate receipt whose prerequisites cannot be replaced by an observed-trace match? |
| Adaptive EMA | JITA-EMA simulations adapt item choice and stopping to context and uncertainty, reducing administered items in the evaluated scenarios; effects in deployed studies remain an empirical question. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10450096/> | Can an adaptive selection policy remain part of evidence custody so the chosen moments are not treated as policy-free samples? |
| Controllable missingness and active observing | Work on controllable missingness and active observing treats whether to acquire features or observations as a decision with cost. <https://arxiv.org/abs/2204.03872> <https://proceedings.neurips.cc/paper_files/paper/2023/file/9050e8d5b5de08d16e65dc79ad5c0146-Paper-Conference.pdf> | Can a retry be typed as a declared information-gathering action with burden, provenance, and authority limits? |
| Selective labels | In decision systems, outcomes are observed only for cases selected by an earlier decision, creating labels whose availability depends on policy. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915/> | Can “not observed because not selected” be made impossible to pass as “selected and negative”? |
| Dyadic informative missingness | Shared-parameter models have been used when missingness in dyadic longitudinal data may depend on both partners' outcomes. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5568500/> | Can future relational models carry whose state affected whose observation opportunity without collapsing the dyad? |

The compass points to a distinction that ordinary `Option<Value>` cannot
express:

```text
same stored token
!= same route to absence
!= same response opportunity
!= same likelihood contribution
!= same recoverability assumptions
!= same ethical permission to acquire more evidence
```

## The identifiability boundary

The strongest literature constraint is the observational-equivalence result.
D4 therefore **does not identify MAR or MNAR** from its original trace. The
labels “declared target-independent” and “declared target-dependent” are names
of deterministic rival programs in a frozen family. They are not statistical
classification receipts and do not assert ignorability.

An equal observed-data fit cannot select a missingness mechanism by itself.
D4 makes that failure productive: `ObservedTraceEquivalenceReceipt` records
that the original traces agree, hidden fixture targets differ, and predictions
under an additional retry differ. Its fields explicitly deny empirical
identification, missingness-taxonomy identification, and global recoverability.

## Frozen four-mode fixture

The custody vector is:

```text
(considered, scheduled, delivered, opportunity,
 responded-in-window, delayed-at-window-close,
 value-present, value)
```

The four synthetic modes predict:

| mode | hidden fixture target | exact custody vector |
|---|---:|---|
| delivery failure | 2 | `(1,1,0,0,0,0,0,-1)` |
| declared target-independent nonresponse | 2 | `(1,1,1,1,0,0,0,-1)` |
| declared target-dependent suppression | 8 | `(1,1,1,1,0,0,0,-1)` |
| policy withholding | 8 | `(1,0,0,0,0,0,0,-1)` |

Every row coarsens to legacy value `-1`. With only that token, the version
space stays:

```text
mask 15 = {delivery failure, independent, dependent, policy withholding}
```

With the complete observed custody vector `(1,1,1,1,0,0,0,-1)`, exact replay
produces:

```text
mask 15 -> mask 6 = {independent, dependent}
```

This establishes that scheduling, delivery, and response opportunity occurred
inside the fixture. It does not reveal the hidden target and does not identify
the response mechanism. D4 must emit
`EndogenousObservabilityAmbiguityReceipt`.

## Synthetic retry

The extra evidence action is a frozen, synthetic exogenous retry at tick 3. Its
assignment is outside the state-dependent policy in this fixture; no random
sampling or stochastic outcome is claimed:

```text
independent rival -> response in retry window, value 2
dependent rival   -> no response in retry window, no value
```

After the ambiguous custody trace:

```text
retry response(value=2): mask 6 -> mask 2, hypothesis 311
retry nonresponse:       mask 6 -> mask 4, hypothesis 312
```

The custody evidence costs five declared fixture units and the retry costs
four. Provenance identifiers `7101, 7102` yield the exact recurrence:

```text
7101 * 31 + 7102 = 227233
```

The resulting `DeclaredResponseMechanismIdentificationReceipt` says only that
one rival remains inside the frozen family. It explicitly denies observation
of the original hidden target, global truth, MAR/MNAR classification,
biological mechanism, and clinical authority. The retry assignment is not a
real-person experiment, consent artifact, or treatment authorization.

## Temporal custody

“No response within the window” is not “no response ever.” A separate control
closes the original zero-length synthetic window at tick 2 and records a value
arriving at tick 3:

```text
prompt tick = 2
arrival tick = 3
elapsed ticks = 1
aligned observed tick = 3
```

`DelayedResponseReceipt` and `ContemporaneousObservedValueReceipt` are
different types. Alignment preserves tick 3 and records that no retroactive
assignment to tick 2 occurred. A substantive model could later introduce an
explicit latent-time relationship, but lateness alone cannot silently rewrite
history.

## Absence is not a value

A window nonresponse can be evidence about a declared response process. It
still **cannot be coerced into a numeric target value**. The same applies to a
delivery failure, a policy-withheld prompt, and a coarsened missing token.

D4 enforces these distinct failures:

```text
not scheduled != scheduled but not delivered
not delivered != delivered without response
no response in window != no eventual response
late value != contemporaneous prompt value
observational equivalence != recoverability
within-family identification != biological mechanism
declared burden != suffering
```

Policy-erased custody and a retry whose provenance does not link to the prior
trace both abstain with the survivor mask unchanged. This matters because the
observation policy determines which states were eligible to become data.

## Exact verification

The independent Python oracle does not import the Sounio implementation. It:

- enumerates all `2^7 * 3 = 384` custody representations formed by seven
  Boolean fields and values `{-1,2,8}`;
- proves that only the three declared custody cells match any mode;
- verifies the partition masks `{1,6,8}`;
- checks both retry branches, burden, and provenance arithmetic;
- enumerates all `4! = 24` hypothesis-ID relabelings;
- verifies delayed alignment and both abstention paths.

Hypothesis identifiers never enter prediction functions. Relabeling can change
the selected opaque ID, but cannot change the observation partition.

## Falsification and demotion

D4 fails if any of the following occurs:

- the legacy missing token removes a hypothesis;
- a delivery failure or policy withholding becomes participant nonresponse;
- window nonresponse becomes a numeric target value;
- the equal original traces select either central mechanism;
- hypothesis identifiers affect predictions or partitions;
- policy-erased custody updates the contest;
- a disconnected retry updates the contest;
- a delayed response is aligned to the prompt tick without an explicit model;
- retry response or nonresponse selects a mode outside the frozen table;
- within-family selection becomes MAR/MNAR, recoverability, biology, suffering,
  consent, or clinical action;
- the independent exhaustive oracle disagrees;
- D3, D2, D1, or D0 regresses.

Even a green gate establishes no empirical psychiatric response mechanism,
causal missingness process, calibrated symptom model, safety, utility, or
ethical justification for active measurement. It also establishes no runtime
kernel-to-ontology result identity.

## New terrain exposed

D4 freezes policy so its epistemic role can be isolated. The next frontier is
**policy-state feedback**: an adaptive policy changes what becomes observable;
the resulting evidence changes the inferred state; that inference changes the
next policy. In medicine and psychiatry, such a loop also carries burden,
equity, consent, and escalation constraints.

A future D5 should therefore not merely optimize information gain. It should
contest policies under explicit observation budgets and typed ethical
constraints, retain positivity/coverage failures, and return abstention when
the evidence needed to compare policies was never eligible for observation.
That is where active sensing, causal recoverability, dyadic context, and the
founder's suffering-minimization objective can meet without turning an elegant
inference engine into an unauthorized clinical actor.
