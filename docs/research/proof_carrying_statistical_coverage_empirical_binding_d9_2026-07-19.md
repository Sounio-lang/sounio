<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-statistical-coverage-empirical-binding-d9-2026-07-19
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-statistical-coverage-empirical-binding-d9-2026-07-19
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# D9: Proof-Carrying Statistical Coverage And Empirical Binding

Date: 2026-07-19

Evidence level: executable finite synthetic receipt model plus a hash-bound
external candidate-data software-integration refusal replay

Concept-ID: `SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING`

In this lane, **proof-carrying** is a project-local term for typed values that
carry exact finite witness receipts consumed by later checks. It is not a claim
of Necula-style proof-carrying code, a proof term accepted by Lean or Coq, or a
machine-checked theorem about general statistical coverage. The executable
claim below is deliberately bounded by frozen programs, fixtures, compiler
identity, and negative type-checking evidence.

The documentation-governance validation date above is not a scientific
validation date. The dated protocol, immutable hashes, executable gate, and
review log are the evidence surfaces for this lane.

The generated `historical` label means this research narrative is preserved for
lineage; it does not mean that prose alone is current executable evidence. The
canonical present contract is
[Proof-Carrying Statistical Coverage And Empirical Binding](../internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md).
Every result below is commit-bound and must be re-established by the linked gate
after any source or compiler change.

## Executable Evidence Map

- [canonical Sounio kernel](../../stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio)
- [standalone Sounio runtime witness](../../tests/run-pass/clinical_statistical_coverage_empirical_binding_native_witness.sio)
- [imported check-only API witness](../../tests/run-pass/clinical_statistical_coverage_empirical_binding_witness.sio)
- [independent exhaustive and external oracle](../../scripts/research/proof_carrying_statistical_coverage_empirical_binding_oracle.py)
- [acceptance and tamper gate](../../scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh)
- [external fixture manifest and boundaries](../../tests/fixtures/psychiatric_d9/README.md)
- [parallel ontology](../../stdlib/ontology/statistical_coverage_empirical_binding.sio)
- [compile-fail evidence directory](../../tests/compile-fail/)

The narrative is not a substitute for those artifacts. The Python oracle and
the standalone Sounio program independently enumerate the finite synthetic
family; the gate compares their exact receipts and checks every negative
program.

The reviewable artifact is the commit tree containing those relative links,
not this Markdown file in isolation. A detached copy of this narrative is not
sufficient to reproduce or falsify the executable claim.

## Research Question

Can a programming language make it difficult to confuse:

1. an exact identified set under a model;
2. a confidence procedure and one realized confidence region;
3. a predictive set for a future response;
4. a declared applicability context;
5. an externally validated empirical binding;
6. a patient state;
7. clinical action authority?

D9 answers a bounded version. It makes those categories nominally distinct,
attaches exact finite arithmetic to their declared targets, and requires
abstention when calibration, positivity, compatibility, custody, or validation
obligations fail.

## Literature-Derived Design Constraints

### Identified set versus confidence region

Imbens and Manski distinguish confidence statements concerning a partially
identified parameter from the identified interval itself. Chernozhukov, Hong,
and Tamer likewise formulate inference for set-identified parameters with a
separate criterion and confidence construction. D9 therefore refuses to treat
the D8 exact compatible set as a confidence region.

- [Imbens and Manski, 2004](https://doi.org/10.1111/j.1468-0262.2004.00555.x)
- [Chernozhukov, Hong, and Tamer, 2007](https://doi.org/10.1111/j.1468-0262.2007.00794.x)

### Marginal versus conditional and selected coverage

Distribution-free marginal predictive coverage does not imply unrestricted
pointwise conditional coverage. Recent work makes conditional guarantees
relative to declared groups or classes, while selection-aware inference changes
the validity target after a selection rule has acted. D9 consequently gives
marginal, subgroup, and selection-conditioned receipts different types.

- [Barber et al., 2021](https://arxiv.org/abs/1903.04684)
- [Gibbs, Cherian, and Candes, 2025](https://doi.org/10.1093/jrsssb/qkaf008)
- [Jin and Ren, 2025](https://doi.org/10.1093/jrsssb/qkaf016)

### Population, positivity, and transport

Transport from a source to a target population requires assumptions connecting
the two domains. Positivity and overlap determine where a target contrast is
supported; changing or trimming the population changes the target. D9 carries
population and support identities and refuses to equate sampling support with
treatment positivity, exchangeability, or a causal effect.

- [Pearl and Bareinboim, 2014](https://doi.org/10.1214/14-STS486)
- [Dahabreh et al., 2019](https://doi.org/10.1111/biom.13009)
- [Crump et al., 2009](https://doi.org/10.1093/biomet/asn055)

### Calibration, provenance, and abstention

Forecast calibration is a joint frequency property of forecasts and outcomes;
it is not metrological calibration of an instrument. PROV-O provides a model
for derivation and responsibility but does not make provenance identical to
trust, integrity, authenticity, or validity. Selective classification treats
abstention as its own risk-coverage decision, not as a negative class label.

- [Gneiting, Balabdaoui, and Raftery, 2007](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- [W3C PROV-O](https://www.w3.org/TR/prov-o/)
- [El-Yaniv and Wiener, 2010](https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf)

### Prediction versus action

Prediction does not identify treatment effect or clinical utility. Decision
curve analysis itself needs explicit consequences and threshold preferences.
D9 therefore makes clinical authority unreachable from its statistical and
external-candidate receipts.

- [Vickers and Elkin, 2006](https://doi.org/10.1177/0272989X06295361)

These sources motivate the semantic distinctions. They do not validate the D9
fixtures, prove the Sounio implementation correct, or establish novelty. Every
row below is an illustrative separation map, not a claim that D9 implements or
discharges a cited theorem's hypotheses.

| Literature constraint | D9 executable surface | Claim deliberately absent |
|---|---|---|
| identified set is not its confidence procedure | `ExactABIdentifiedSetReceipt` versus `D9ConfidenceRegionReceipt` compile-fail wall | population inference theorem |
| marginal is not unrestricted conditional coverage | marginal, rare-group, and selection-conditioned nominal receipts | individual conditional coverage |
| transport needs source/target assumptions | target population, time window, and instrument-population compatibility receipts | external transportability |
| positivity determines supported target | sampling-support pass/failure receipts | treatment overlap or exchangeability |
| forecast and metrological calibration differ | separate finite-coverage and instrument-calibration receipts | validated measurement instrument |
| provenance is not trust | lineage, mismatch, integrity, authenticity, and custody walls | source authenticity or consent |
| abstention is not a label | selection and binding abstention receipts | negative prediction or escalation |
| prediction is not utility or treatment effect | predictive-set and causal-ambiguity receipts | clinical benefit or action authority |

The table maps separations, not cited theorem hypotheses one-for-one. D9 does
not claim to formalize or prove any cited paper's theorem.

## Synthetic Coverage Contest

The D8 target is the exactly enumerated set `{A, B}`, mask `3`, under frozen
model `12200` and assumption set `12210`. D9 treats it only as a target.

The set-valued procedure is:

| Outcome | Returned mask | Contains A | Contains B | Contains whole `{A,B}` |
|---:|---:|:---:|:---:|:---:|
| 0 | 3 | yes | yes | yes |
| 1 | 1 | yes | no | no |
| 2 | 2 | no | yes | no |
| 3 | 7 | yes | yes | yes |

Mask `7` is a strict superset containing `{A,B}` plus a third synthetic member.
It exists to exercise the fact that whole-target coverage permits a strict
superset. The later UCI binary-label replay uses only masks `1`, `2`, and `3`;
it is a separate software-integration path, and this synthetic enumeration is
not evidence that the UCI procedure has any statistical validity.

Two exact mass functions share denominator `12`:

| Design | Weights | Whole-set coverage | Min memberwise coverage |
|---|---|---:|---:|
| A | `5,1,1,5` | `10/12` | `11/12` |
| B | `1,5,5,1` | `2/12` | `7/12` |

At outcome zero both realized regions are numerically mask `3`. The collision
does not erase their sampling designs or their distinct coverage receipts.

The threshold `3/4` is an arbitrary frozen fixture requirement, not a
statistically or clinically derived optimum. It is evaluated by exact cross
products:

```text
A: 10 * 4 = 40 >= 36 = 3 * 12
B:  2 * 4 =  8 <  36 = 3 * 12
```

Its sensitivity is transparent: any exact threshold greater than `1/6` and at
most `5/6` separates these two whole-set coverage fractions; thresholds at or
below `1/6` pass both, and thresholds above `5/6` fail both. D9 freezes `3/4`
only as one executable separator in that interval.

The diagnostic permille representations preserve quotient and remainder:

```text
10 * 1000 = 833 * 12 + 4
 2 * 1000 = 166 * 12 + 8
```

## Exhaustive Finite Enumeration

For every nonnegative integer vector `(w0,w1,w2,w3)` summing to `12`, the
independent oracle calculates support and whole-set coverage. It obtains:

| Quantity | Count |
|---|---:|
| all designs | 455 |
| positive on all four outcomes | 165 |
| at least one absent outcome | 290 |
| support size 1 | 4 |
| support size 2 | 66 |
| support size 3 | 220 |
| support size 4 | 165 |
| positive designs adequate at `3/4` | 25 |

Positive-design whole-set numerator histogram:

```text
2:9, 3:16, 4:21, 5:24, 6:25,
7:24, 8:21, 9:16, 10:9
```

The standalone Sounio witness independently repeats this enumeration at
runtime. This is exact evidence for one finite artificial mass family only.
It is illustrative only and makes no scaling, power, efficiency, or
general-purpose inference claim.
The integer quotient/remainder identities are executable assertions in both the
[kernel](../../stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio)
and [native witness](../../tests/run-pass/clinical_statistical_coverage_empirical_binding_native_witness.sio).
They are not presented as Lean theorems or as a general formalization of
Euclidean division.

## Selection Control

The second frozen family has ten occasions:

```text
common group       9/9 covered
rare group         0/1 covered
marginal           9/10 covered
selected rare      0/1 covered
```

The control does not estimate a general selective-coverage theorem. It is a
counterexample to an accidental promotion from marginal to selected coverage.
The selection failure produces abstention and cannot become a negative
prediction or escalation receipt.

## Public Candidate Dataset

D9 includes the official UCI
[Drug Consumption (Quantified)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
dataset, DOI [10.24432/C5TC7S](https://doi.org/10.24432/C5TC7S), under the
repository's recorded CC BY 4.0 declaration.

The official dataset page describes 1,885 respondents, 12 attributes, 18 drug
outputs, and no missing values. D9 stores the raw supplied data file, not an
enriched patient record.

Frozen byte receipts:

```text
official archive SHA-256
0fb006913b8ecde52560dd04e4b8d10c75aad938d337a9e7e11fcab7dd1f6993

raw data SHA-256
90b8cf500b07ad455baf9fe1dc519998c75a1df6d87f6bd7069176f0826ea8c1

evaluation protocol SHA-256
b443191335bda3eb0eaa3bd8fee47a30cebc16080ad9e0862d85d9734fee4a1e
```

The host gate rejects any data or protocol byte change before analysis. Hash
matching is an evidence surface only for reproducibility and local tamper
detection: it proves equality to the committed bytes and nothing about the
world that produced them. It carries no provenance, consent, deidentification,
collection-condition, authenticity, measurement-validity, or custody
guarantee, and cannot enter the empirical-binding constructor.

## Locally Declared Evaluation Protocol

The JSON protocol was locally written and hashed before calculating the
full-dataset result in this execution workflow. This is a self-attested order,
not independently verifiable temporal priority. It freezes:

- eligible rows: `Semeron == CL0`;
- partition: respondent ID modulo 5;
- development remainders: `0,1,2`;
- calibration remainder: `3`;
- evaluation remainder: `4`;
- observed response: Benzos column, recent `CL4/CL5/CL6`, otherwise
  nonrecent `CL0/CL1/CL2/CL3`;
- fixed score: `Nscore + Impulsive + SS`;
- predictive mask `1` at score `<= -1`, mask `2` at score `>= 1`, and mask `3`
  otherwise;
- exact adequacy gate: coverage at least `3/4`;
- required support bands: low, middle, and high;
- expected final authority result: abstain from empirical binding.

The score is not fitted and is not presented as a validated clinical model. ID
partitioning is deterministic, not randomization. The 1,125-row development
partition is deliberately unused: it fits no coefficient, selects no threshold,
and contributes to no reported coverage receipt. It is retained as a frozen
quarantine partition for possible future work, not as evidence of model
development. The `3/4` gate is an arbitrary software-fixture requirement.

The before-analysis ordering is a local D9 workflow record. It has no
independent timestamp, public preregistration, or pre-analysis Git commit and
must not be represented as externally certified temporal priority. The
SHA-256 gate detects later byte changes relative to the committed protocol; it
cannot detect outcome inspection, protocol selection, or editing that may have
occurred before those bytes were committed. The original protocol bytes remain
unchanged after the calculation so this limitation stays auditable rather than
being patched into the frozen JSON.

## External Software-Integration Refusal Replay

The following counts are a deliberately failed software-integration replay on
public health-adjacent data. They are not a clinical evaluation, a serious
predictive-model evaluation, or scientific evidence about benzodiazepine use.
They must not be cited as predictive performance for clonazepam response,
psychiatric state, or any clinical task. Their evidentiary role is only to show
that the implementation reproduces a frozen calculation, detects failure of
its arbitrary held-out gate, and forces abstention.

| Accounting item | Count |
|---|---:|
| raw rows | 1,885 |
| eligible rows | 1,877 |
| Semeron-excluded rows | 8 |
| development | 1,125 |
| calibration | 377 |
| evaluation | 375 |

Calibration result:

```text
covered             286/377
permille diagnostic 758 remainder 234
recent outcomes     58
prediction masks    1:122, 2:109, 3:146
declared gate        adequate
```

Held-out evaluation result:

```text
covered             265/375
permille diagnostic 706 remainder 250
recent outcomes     58
prediction masks    1:124, 2:136, 3:115
declared gate        insufficient
```

All three score bands appear in calibration and evaluation. Support-band
compatibility therefore passes, while the held-out coverage gate fails.

The final refusal does not depend on selecting `3/4`. Changing the threshold
could clear reason bit `2`, but metrological calibration, verified collection
window, external custody, and sealed validation remain absent. Their exact
reason mask is `4 + 32 + 64 + 128 = 228`, so the external path must still
abstain even without the held-out-coverage failure.

The exact external abstention mask is `230`:

```text
2   held-out evaluation/model criticism failed
4   metrological instrument calibration unavailable
32  collection window unverified
64  external custody unsealed
128 sealed validation unavailable
```

No empirical binding, patient state, causal effect, or clinical authority
receipt is issued.

## Interpretation Boundary

The UCI outcome is a self-reported benzodiazepine recency category. It does not
measure whether clonazepam caused sleepiness, activation, symptom relief,
adverse effects, tolerance, withdrawal, or any other individualized response.
It is not a diagnostic interview or a prescribing dataset.

The analysis must therefore not be described as a clonazepam model, a mental
state model, a patient-specific prediction, or evidence for changing treatment.
Its role is narrower: it demonstrates that Sounio can replay a public table and
a locally declared protocol, expose a failed held-out software gate, and refuse
to overpromote the result.

## Provenance Collision

The synthetic provenance control carries the same table fingerprint and
realized mask under two different dataset, source, instrument, population,
window, design, and transformation identities. D9 emits a mismatch receipt.

This keeps four questions separate:

1. **lineage**: what derivation is declared;
2. **integrity**: whether bytes changed;
3. **authenticity**: whether the purported source is genuine;
4. **measurement validity**: whether the instrument measures the intended
   construct for the target use.

None is inferred merely from another.

## Binding Truth Table

For the three bounded fixture gates, calibration, sampling positivity, and
instrument-population compatibility, D9 enumerates all eight Boolean
combinations. Exactly one all-pass combination can produce a
`D9DeclaredContextBindingReceipt`; the other seven produce abstention.

Here, `declared context binding` means only that the synthetic fixture's
declared identities and three bounded gate receipts match. It is deliberately
not an empirical-validation or authority token. A downstream program that
ignores these nominal distinctions or assigns clinical meaning to an arbitrary
value lies outside the current type-safety claim.

Even the all-pass declared fixture binding carries:

```text
external_empirical_binding = false
patient_state = false
clinical_action_authority = false
```

The external candidate path cannot reach even that synthetic all-pass state,
because its held-out gate fails and external obligations remain absent.

## Sealed Authority Categories

Reserved authority structs are module-private and have no positive
constructors. Public consumer functions make wrong substitutions observable to
the type checker. The gate verifies 48 exact expected/found rejections and
three `E176` private-constructor rejections.

For example, passing `ExactABIdentifiedSetReceipt` to the D9 confidence-region
consumer is rejected with:

```text
expected D9ConfidenceRegionReceipt
found ExactABIdentifiedSetReceipt
```

Attempting a direct `D9EmpiricalBindingReceipt` struct literal is rejected with
`E176`, `struct constructor is private in its defining module`. The full use
sites are part of the linked compile-fail matrix and acceptance gate.

This guarantee applies only to the exact checked module and selected compiler
revision recorded by a gate run. It does not claim that a future compiler or
owner could never add an explicit constructor or coercion. Such a change would
alter the concept contract and fail the D9 gate until reviewed; no version-
independent unforgeability claim is made.

## Runtime And Compiler Scope

The reusable kernel and imported witness are check-only because imported
multimodule runtime remains outside the evidence claim under
`BLK-20260718-D6-MULTIMODULE-RUNTIME`. The standalone Sounio witness and Python
oracle provide independent runtime evidence.

The current loader preserves the new D8-to-D9 nominal wall with the established
wildcard import but not with a selective single-type import. D9 fixes the
wildcard source shape in its gate. No compiler, resolver, IR, `Contest`, or
legacy ontology implementation is modified.

Recursive D8-D0 validation on the current-main Madaros binary is separately
blocked by `BLK-20260719-D9-D4-CURRENT-MAIN-AST-CLOSURE`: the byte-identical D4
source passes with the D8 head binary and fails AST-closure parsing with the
current-main binary. This is a retarget/integration blocker owned by codex-2,
not evidence against the D9 coverage model and not permission to use a fallback.

## Exact Supported Claim

D9 demonstrates, for bounded frozen fixtures, that Sounio can:

- retain an identified-set target without calling it a confidence region;
- attach exact whole-set coverage to a procedure and design;
- distinguish whole-set, memberwise, marginal, subgroup, selected, and
  predictive scopes;
- preserve fractions without silent permille truncation;
- retain population, time, instrument, calibration, support, and provenance
  identities;
- bind and replay committed public candidate-data and locally declared protocol
  bytes by hash;
- record a failed held-out software-fixture check;
- mandate abstention when calibration, positivity, compatibility, custody, or
  validation obligations fail;
- reject promotion into empirical binding, patient state, causal treatment
  effect, and clinical action authority.

## Claims Not Supported

D9 does not establish:

- a statistically efficient or consistent estimator;
- predictive-model validation, clinical performance, or scientific utility of
  the UCI score;
- externally established preregistration, temporal priority, or protection
  against pre-commit outcome-dependent protocol selection;
- asymptotic, Bayesian, conformal, or distribution-free validity;
- unrestricted conditional or individual coverage;
- external transportability or treatment positivity;
- a causal effect, comparator, counterfactual, benefit-harm balance, or utility;
- metrological validation of any questionnaire or instrument;
- original consent, deidentification audit, authenticity, or custody;
- a recovered psychiatric, functional, diagnostic, prognostic, or suffering
  state;
- an individualized recommendation, escalation, prescription, or clinical act;
- novelty or priority over the literature.

## Falsifiers

The bounded D9 claim fails if:

- the D8 exact set substitutes for a D9 confidence region;
- equal realized masks erase the different design coverage;
- whole-set and memberwise coverage substitute for one another;
- exact threshold decisions depend on truncated permille;
- the synthetic enumeration differs from 455, 165, 290, or 25;
- marginal coverage substitutes for selected coverage;
- lineage substitutes for integrity, authenticity, validity, or custody;
- statistical and metrological calibration substitute for each other;
- sampling positivity substitutes for a causal treatment effect;
- any failed gate creates a binding instead of abstention;
- the dataset or protocol can be altered without SHA-256 refusal;
- the external counts or exact coverage receipts drift;
- a declared binding or abstention becomes empirical binding, patient state, or
  clinical authority;
- a reserved authority struct can be constructed outside its module;
- the lane changes the compiler, resolver, D0-D8 semantics, or legacy ontology
  path.

## Validation Contract

Acceptance requires canonical Madaros; exact kernel and witness checks; native
and ontology execution; exhaustive synthetic and external oracle replay;
dataset and protocol tamper negatives; all 51 compile-fail programs; exact
concept registry and binding rows; both ontology-validation paths; recursive
D8-D0 gates; and independent math and clinical-authority review.
