<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-path-conditioned-identification-d8-2026-07-18
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-path-conditioned-identification-d8-2026-07-18
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# D8: Proof-Carrying Path-Conditioned Partial Identification

Date: 2026-07-18
Evidence level: executable bounded synthetic model
Concept-ID: `SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION`

The generated `docs:meta` validation date is the repository-wide documentation
governance baseline, not a D8 scientific sign-off or evidence date. The dated
fixture, executable gate, and review log are the D8 evidence surfaces.

## Question

Can a programming language preserve the fact that latent-state claims depend
on ordered history, model assumptions, observation policy, approximation
warrant, and provenance, while refusing to turn those claims into statistics,
causality, patient truth, or clinical authority?

D8 answers this question for one finite synthetic family. It implements exact
set enumeration, path-specific nominal types, provenance-bearing refinement,
information-forgetting outer controls, missingness abstention, explicit
model-evidence conflict, and typed authority refusals.

## Semantic Commitments

D8 uses these terms narrowly:

- **compatible state**: a state retained by the declared fixture predicate;
- **exact identified set**: every compatible state in the frozen finite family
  has been enumerated under one model and assumption set;
- **sound outer set**: a nominal information-forgetting view guaranteed to
  contain the corresponding hidden exact set;
- **witnessed inner set**: members for which the fixture carries an explicit
  compatibility witness;
- **finite point identification**: a refined set has exactly one member;
- **separation**: two exact refined sets are nonempty and disjoint;
- **conflict**: an observation eliminates every member of the current set.

These are executable fixture meanings, not imported statistical or clinical
theorems.

## Frozen State Family

| State | State ID | Scalar projection | Initial `AB` | Initial `BA` | Synthetic constraint |
|---|---:|---:|:---:|:---:|:---:|
| A | 12001 | 877 | yes | no | yes |
| B | 12002 | 877 | yes | yes | no |
| C | 12003 | 877 | no | yes | yes |

All three nominal states collide under scalar projection `877`:

```text
projection(A) = projection(B) = projection(C) = 877
```

The exact direct fields are authoritative. The state-family fingerprint
`11917026` and projection-collision checksum `369428683` are audit diagnostics.
Because IDs exceed the base used in the recurrence, the recurrence is not a
general injective representation.

## Ordered Histories And Initial Sets

The paths carry distinct nominal receipts:

```text
AB actions = (14001, 14002), fingerprint = 448033
BA actions = (14002, 14001), fingerprint = 448063
```

For model `12200`, protocol `8800`, assumption set `12210`, and declared
individual context, the exhaustive finite compatibility predicates yield:

```text
I_AB = {A, B} = mask 3
I_BA = {B, C} = mask 6
I_AB intersect I_BA = {B} = mask 2
```

Receipt `{B}` witnesses initial compatibility. The initial separation request
is therefore refused with reason mask `3` and checksum `2650573642`.

The model checksum `11728748010` binds the fixture's explicit model,
assumption, subject, window, instrument, policy, family, and history fields for
diagnostic replay. It is not a cryptographic seal.

## Synthetic Provenance

The root observation and its alternative missingness and conflict branches
carry distinct synthetic policy decisions:

| Branch | Provenance ID | Source | Tick | Policy decision | Predecessor | Fingerprint |
|---|---:|---:|---:|---:|---:|---:|
| root constraint | 12351 | 16001 | 1 | 16101 | 0 | 477185883 |
| missing branch | 12352 | 16002 | 2 | 16102 | 12351 | 477229017 |
| conflict branch | 12353 | 16003 | 3 | 16103 | 12351 | 477259800 |

The corresponding policy-decision checksums are `111567302`, `111574037`, and
`111580772`. Construction is public and unsealed; every receipt states that a
real policy decision and real-world custody have not been established. The two
children are alternative fixture branches, not consecutive events in one
patient chronology.

## Exact Refinement And Separation

The synthetic history-invariant constraint has compatible mask `{A, C} = 5`.
It is applied as the same declared model restriction to both histories:

```text
R_AB = I_AB intersect {A, C} = {A} = mask 1
R_BA = I_BA intersect {A, C} = {C} = mask 4
R_AB intersect R_BA = empty
```

The subset receipts have checksums `15515` and `18492`. The finite singleton
receipts have checksums `384337994` and `384369739`. They retain the source set,
refined set, model, assumptions, history, observation, and provenance, and
explicitly state `global_identification_claimed=false`.

Exact separation checksum `12401023` retains both histories, model,
assumptions, observation, and provenance. Separation is a relation between
model-compatible sets. It is not an intervention effect, a counterfactual, a
causal mechanism, or evidence that a real person changed state.

## Exact, Inner, And Outer Warrants

D8 intentionally forgets exactness to construct nominal sound outers. Their
masks remain `3` and `6`, but their types no longer authorize exact-enumeration
claims. Under those two outer bounds there are nine nonempty pairs of possible
exact completions:

```text
completion pairs = 9
overlapping exact completions = 4
disjoint exact completions = 5
```

Thus the outer-only view is undecided. Its checksum is `12090976311463`.
This undecidedness is local to the counterfactual information-forgetting view;
the complete D8 graph separately carries the exact initial common-member
witness `{B}`. Saying that the complete graph has unknown overlap would be
false.

Exhaustive enumeration over the three-state universe gives:

| Oracle domain | Count |
|---|---:|
| all left/right subset pairs | 64 |
| nonempty `(left, right, evidence)` triples | 343 |
| triples satisfying the direct refinement identity | 343 |
| post-refinement disjoint triples | 174 |
| initially overlapping then post-refinement disjoint | 90 |
| triples with both refinements nonempty | 205 |
| both nonempty and post-refinement disjoint | 36 |
| sound exact/outer tuples | 361 |
| disjoint-outer tuples | 24 |
| disjoint-outer soundness violations | 0 |
| overlapping outers with disjoint exact sets | 120 |
| overlapping outers with overlapping exact sets | 217 |

The exhaustive identity is:

```text
(left intersect evidence) intersect (right intersect evidence)
    = (left intersect right) intersect evidence
```

It is a finite set identity over masks, not a statement that histories compose
associatively or that clinical trajectories can be reordered.

## Missingness And Conflict

`NotMeasuredReceipt` is not `ObservedFalseReceipt`. When the outcome is missing
under the declared synthetic policy, D8 retains the current `AB` set (`mask 3`)
and emits abstention checksum `12373740403`; it does not refine from absence.

The alternative conflict branch presents a synthetic `C`-only observation to
the refined `AB` set `{A}`. The result is the empty set. D8 emits
`ModelEvidenceConflictReceipt` checksum `24113281431937` with
`nearest_state_selected=false`. Empty compatibility is not permission to pick
the closest state, hide a violated assumption, or force a clinical label.

## D7 Relationship

D8 uses D7 only as a negative boundary. A public D7 local equality decision for
fixture occurrence `11001` is presented at D8 occurrence `17001`; the mismatch
produces refusal checksum `2175470118`, reason mask `3`.

D7 supplies no path-conditioned set semantics. D8 performs no rebracketing,
source rewrite, IR rewrite, compiler capability issuance, or native `Contest`
construction. The two concepts remain independently registered.

## Literature Compass

The design is adjacent to several established bodies of work:

- Partial-identification theory treats the compatible object as a set rather
  than forcing an unsupported point. Random-set methods provide one rigorous
  route to identified regions, but they do not validate D8's fixture.
  [Beresteanu, Molchanov, and Molinari](https://doi.org/10.1016/j.jeconom.2011.06.003)
- Estimation and confidence regions for an identified set are separate from the
  set itself. D8 has no sampling distribution, estimator, confidence level, or
  coverage theorem. [Chernozhukov, Hong, and Tamer](https://www.jstor.org/stable/4502015)
- Confidence intervals under partial identification require their own coverage
  semantics. A D8 exact finite set is therefore not a confidence region.
  [Imbens and Manski](https://doi.org/10.1111/j.1468-0262.2004.00555.x)
- Imprecise probabilities model uncertainty through objects such as lower
  previsions or sets of probabilities. This motivates keeping credal and p-box
  categories separate from state compatibility sets.
  [Walley](https://doi.org/10.1016/S0888-613X(00)00031-1)
- Robust sequential decision results show that rectangularity and information
  structure matter. D8 borrows only the discipline of retaining structure; it
  proves no robust-control theorem.
  [Iyengar](https://doi.org/10.1287/moor.1040.0129)
- Selective-label and performative-prediction work shows why observation policy
  and action-dependent data generation cannot be treated as passive sampling.
  D8 models only declared synthetic policy provenance.
  [Lakkaraju et al.](https://doi.org/10.1145/3097983.3098066),
  [Perdomo et al.](https://proceedings.mlr.press/v119/perdomo20a.html)
- Positivity failures can invalidate inverse-probability analyses. D8 does not
  estimate propensities or repair positivity; missingness causes abstention.
  [Cole and Hernan](https://pmc.ncbi.nlm.nih.gov/articles/PMC2732954/)
- Computational psychiatry requires explicit generative assumptions and model
  and parameter recovery checks. That literature motivates D8's refusal to
  infer patient truth from a synthetic fixture.
  [Hess et al.](https://doi.org/10.5334/cpsy.116)
- Dynamical approaches to psychiatry motivate history-sensitive models, but do
  not establish this model's empirical validity or authorize treatment.
  [Dynamical systems in psychiatry](https://pubmed.ncbi.nlm.nih.gov/38568618/)
- Proof-carrying code and provenance ontologies motivate checkable evidence and
  lineage. D8 has neither a sealed trusted proof checker nor PROV-O runtime
  transport. [Necula](https://doi.org/10.1145/263699.263712),
  [PROV-O](https://www.w3.org/TR/prov-o/)

The plausible novelty is the integration of nominal ordered histories, finite
partial-identification warrants, provenance-bearing refinement, and typed
authority refusals in one executable language surface. No reviewed source
establishes that Sounio is the first language to do this. A systematic novelty
review remains required before any priority claim.

Likewise, the current negative ontology witnesses prove current-source nominal
non-substitution. They are neither OWL disjointness axioms nor a sealed theorem
against a future explicit coercion.

## Exact Supported Claim

For one frozen three-state synthetic model, Sounio can:

- preserve three nominal states despite equal scalar projections;
- retain distinct `AB` and `BA` histories and exact compatible sets;
- refine those sets under one provenance-bearing synthetic constraint;
- certify two nonempty singleton refinements and their exact separation;
- distinguish exact, sound-outer, witnessed-inner, and heuristic warrants;
- abstain under declared missingness and report empty-set conflict;
- reject substitutions into statistical, uncertainty, causal, dyadic,
  suffering, clinical, compiler, and Contest categories.

The native scalar mirror and independent oracle agree on every frozen value.
The imported reusable module is check-only evidence.

## Claims Not Supported

D8 does not establish:

- empirical validity, ecological validity, or external validity;
- a population identified set or statistical sharpness;
- estimator consistency, uncertainty calibration, or confidence coverage;
- a likelihood, posterior, p-box, credal set, predictive distribution, or risk
  score;
- causal identification, intervention effects, or counterfactual truth;
- a recovered functional, psychiatric, diagnostic, or suffering state;
- an individualized or dyadic treatment recommendation;
- real policy provenance, consent, custody, or tamper evidence;
- collision resistance, authenticity, or unforgeability of checksums;
- runtime kernel-to-ontology transport;
- native Contest/IR integration or compiler rewrite authority;
- a general algorithm for infinite or continuous state spaces;
- novelty or priority over the literature.

## Falsifiers

The bounded claim fails if:

- any scalar projection differs from `877` or a scalar substitutes for a state;
- either ordered history substitutes for the other;
- the exact initial masks differ from `3` and `6`, or their intersection from
  `2`;
- refinement differs from masks `1` and `4` or loses provenance;
- either singleton is promoted to global state truth;
- exact, outer, inner, heuristic, confidence, predictive, p-box, credal, or
  posterior categories typecheck as interchangeable;
- outer-only overlap is reported as real exact overlap;
- missingness refines the set, or conflict selects a nearest state;
- association or exact separation substitutes for intervention,
  counterfactual, human suffering, or clinical action;
- the D7 decision is reused across occurrences or promoted to D8 semantics;
- the oracle and native mirror disagree on a frozen receipt or enumeration;
- D8 changes the compiler, performs rebracketing, creates Contest IR, or claims
  ontology transport.

## Validation Contract

Acceptance requires canonical Madaros, kernel and imported checks, native and
ontology execution, the independent exhaustive oracle, the complete negative
matrix, exact registry and binding rows, documentation and semantic-registry
checks, recursive D7-D0 gates, and independent LLM math review.

The advisory science-boundary verdict is currently `UNKNOWN`; it is reported,
not promoted to authoritative evidence.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: psychiatric-regime-D8-path-conditioned-partial-identification
Owner: Codex scientific protocol lane; compiler capability owner remains codex-2
Concept-IDs: SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION, SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL
Intent-Preserved: history, assumptions, provenance, approximation warrant, and authority remain observable
Transformation: add finite path-conditioned identified sets, refinements, abstentions, conflicts, and typed refusals
Types-Changed: new stdlib and parallel ontology types only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: exact bounded D8 supported claim above
Claims-Forbidden: patient truth, statistical coverage, causal identification, clinical authority, compiler authority, native Contest bridge
Assumptions: exact frozen integer fixture, three-state closed family, declared synthetic provenance, current D7 boundary
Write-Set: D8 stdlib, ontology, concept/spec, tests, oracle, gate, bindings, governance metadata, offload log
Read-Set: D7-D0 kernels/tests/gates, semantic registry, ontology validation scripts
Positive-Witness: imported check-only API witness, native scalar mirror, exhaustive finite oracle, parallel ontology witness
Negative-Witness: path/state/approximation/statistical/uncertainty/causal/context/suffering/clinical/compiler/Contest boundaries
Acceptance-Gate: scripts/ci/proof_carrying_path_conditioned_identification_gate.sh
Pending-Interface: statistical-coverage-and-empirical-state-binding
```

## Integration Receipt

```text
Semantic-Outcome: bounded path-conditioned partial identification with explicit abstention and conflict
Distinctions-Added: scalar != state; AB != BA; exact != outer != inner != heuristic; missing != false; association != intervention
Distinctions-Preserved: model != patient; identified set != statistical set; proxy != suffering; public receipt != authority
Distinctions-Erased: none
Runtime-Path: standalone scalar native mirror plus independent exhaustive oracle
Imported-Path: check-only; multimodule runtime blocker remains explicit
Ontology-Path: parallel nominal evidence; runtime transport false
Compiler-Path: unchanged; rebracketing false; rewrites 0
Contest-Path: unchanged; Contest/TyContest/IrContest receipts 0
Legacy-Kept: D7-D0, all compiler paths, and all existing ontology routes
```
