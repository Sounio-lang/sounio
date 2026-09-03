<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-deployment-validity-revocable-authority-d10-2026-07-19
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-deployment-validity-revocable-authority-d10-2026-07-19
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

> Metadata note: `last_validated` is generated from the repository governance
> baseline. The scientific evidence and review date for this D10 fixture is
> 2026-07-19.


# D10: Proof-Carrying Deployment Validity And Revocable Authority

Date: 2026-07-19

Evidence level: bounded exact synthetic typestate, independent finite oracle,
negative type-checking evidence, and immutable local spending/revocation
transition traces.

Concept-ID: `SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY`

## Research Question

Can a language preserve the difference between statistical evidence and a
revocable deployment warrant, while making site, version, time, monitoring,
change control, safe deferral, and institutional authority explicit?

D10 answers only for frozen fixtures. Its maximum positive artifact is a
site-scoped synthetic canary lease. Production and clinical authorities have
private types and no producers.

The positive stage types implement classic nominal typestate. Any caller can
invoke the frozen fixture producers and pipe their private return values by
inference, so the lane proves stage non-substitution, not caller authenticity
or evidence origin.

## Literature-Derived Constraints

Fixed-horizon inference does not automatically survive continuous monitoring
or optional stopping. Confidence sequences and safe anytime-valid inference
motivate distinct fixed-time, time-uniform, e-value, and e-process receipts.

- [Howard et al., 2021](https://doi.org/10.1214/20-AOS1991)
- [Grunwald, de Heide, and Koolen, 2024](https://doi.org/10.1093/jrsssb/qkae011)
- [Ramdas et al., 2023](https://arxiv.org/abs/2210.01948)
- [Anytime-Valid Conformal Risk Control, 2026](https://arxiv.org/abs/2602.04364)

Healthcare AI guidance treats validation and monitoring as lifecycle- and
context-bound. Representative data, independent testing, fit-for-purpose
reference standards, human-AI team performance, local workflow, real-world
monitoring, change control, and safe operation are distinct obligations.

- [IMDRF N88, 2025](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles)
- [FDA PCCP final guidance, 2025](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence)
- [FDA AI-enabled device lifecycle draft, 2025](https://www.fda.gov/media/184856/download)
- [FUTURE-AI, 2025](https://doi.org/10.1136/bmj-2024-081554)
- [NHS DCB0129](https://digital.nhs.uk/data-and-information/information-standards/governance/latest-activity/standards-and-collections/dcb0129-clinical-risk-management-its-application-in-the-manufacture-of-health-it-systems/)
- [NHS DCB0160](https://digital.nhs.uk/binaries/content/assets/website-assets/data-and-information/information-standards/standards-and-collections/dcb0160/0160252018spec.pdf)

The FDA lifecycle document cited above remains draft and is not for
implementation. The PCCP guidance is final as of 18 August 2025. These sources
motivate receipt dimensions; D10 does not claim regulatory compliance.

Typestate, parameterised computations, graded resources, capabilities, and
revocation are prior art. D10's possible research direction is their
domain-specific composition around statistical warrant, not any of those
mechanisms individually.

- [Strom and Yemini, 1986](https://doi.org/10.1109/TSE.1986.6312929)
- [Atkey, 2009](https://doi.org/10.1017/S095679680900728X)
- [Gordon, 2021](https://doi.org/10.1145/3450272)
- [Orchard, Liepelt, and Eades, 2019](https://doi.org/10.1145/3341714)
- [AURA, 2008](https://doi.org/10.1145/1411204.1411212)
- [Typed attenuated capabilities, 2017](https://doi.org/10.4230/LIPIcs.ECOOP.2017.20)
- [Necula PCC, 1997](https://doi.org/10.1145/263699.263712)

## Exact Contests

### C1: discrimination is not calibration

The A/B score vectors have identical ordering and threshold decisions but exact
Brier scores `1/40` and `29/160`.

### C2: fixed horizon is not time uniform

Two separate looks each cover `12/16`. Stopping at the first miss reduces the
reported coverage to `9/16`. A different bounded procedure has simultaneous
two-look coverage `12/16` and therefore retains `12/16` at the frozen stopping
rule. No general confidence-sequence theorem is claimed.

### C3: a fixed-time e-value is not an e-process

The two-step fair-tree process has expectation one at time zero and each fixed
time. Its frozen rule stops after first-step `H` and otherwise continues,
giving stopped values `[2,2,0,0]` on `[HH,HT,TH,TT]` and expectation one. A
matching terminal number without process history cannot replace the process
receipt.

### C4: manufacturer evidence is not local deployment safety

The same research artifact is presented to two synthetic sites. Only site A
has the declared workflow, operator, override, stop, and local safety receipts.
Site B must remain quarantined.

### C5: abstention is not safe deferral

Both sites produce the same model abstention. Site A has capacity `2`,
acknowledgements `2`, and the response-time requirement met. Site B has
capacity `1`, acknowledgements `0`, and no completed handoff.

The site-A token is issued only after distinct handoff, capacity, response-SLA,
and unresolved-case-policy receipts agree on the site and destination. The
public observation derives those fields from the resulting private token.

### C6: matching metrics do not authorize a change

Two model updates preserve the same frozen discrimination and decision counts.
Only one follows the declared modification description, protocol, acceptance
criteria, and impact assessment. The other is quarantined.

The declared impact-assessment identifier must equal the assessment receipt
identifier directly. A separate conformance receipt binds all four declared
obligations before the private modification stage can be produced.

### C7: drift categories and absence claims remain distinct

The score-vector collision keeps AUROC fixed while calibration changes.
`CalibrationDrift`, `InputDistributionShift`, and `PerformanceDrift` are
siblings. `NoDetectedShift` is an observation under a detector, not `NoShift`.

### C8: spending and revocation are immutable local traces

An exact budget of `100` units is spent as `40 + 60`. A further `10` is refused.
The private spent token preserves nonces `30811` and `30812`, and the lease
preserves the second nonce plus ledger epoch one. A repeated nonce is recorded
as reuse. A lease valid at epoch one is marked revoked at epoch two. These
calculations are an immutable local transition trace; they neither create a
shared transactional ledger nor statically consume aliases or invalidate a
copied old facet.

## Falsification And Demotion

The bounded claim fails if any wrong-stage or authority-promotion program
type-checks, if any private authority literal can be constructed, if the native
and Python exact receipts disagree, if a stale/unsupported change reaches the
fixture lease, or if recursive D9-D0 evidence regresses.

Even with a green gate, the concept must be demoted if documentation describes
the lease as real deployment permission, spending as affine, revocation as
live, or the finite sequential fixture as a general statistical theorem.
