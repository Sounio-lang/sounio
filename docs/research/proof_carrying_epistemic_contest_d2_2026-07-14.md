<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-epistemic-contest-d2-2026-07-14
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-epistemic-contest-d2-2026-07-14
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Epistemic Contest D2

Status: frozen bounded synthetic specification, 2026-07-14.

## Thesis

D2 tests a programming-language proposition, not a psychiatric mechanism:

> An inference should be represented as a compiler-checkable transition from
> an explicit hypothesis version space through admissible evidence to a
> bounded conclusion or abstention, while preserving the conclusions that the
> evidence does not authorize.

The new object is not merely a prediction and not merely a provenance record.
It is a typed chain containing the declared family, common probe, exact
prediction table, observation status, provenance, survivor set, eliminated
set, burden spent, and conclusion boundary.

Each transition retains the exact current and predecessor provenance IDs,
source, and tick. The state also carries a bounded base-31 fingerprint for
quick audit comparison; that fingerprint is explicitly non-authoritative and
is not claimed collision-free. With at most eight evidence links and IDs at
most `10^6`, its exact maximum is `28,429,701,248,000,000`, within signed
64-bit range.

Madaros v0.80.0 does not yet enforce cross-module field visibility as a
constructor-sealing boundary. D2 therefore revalidates every hypothesis,
family, probe, provenance edge, state invariant, transition delta, and exact
prediction replay at each consuming function. This makes forged inconsistent
receipts fail closed, but it is not cryptographic authenticity and does not
establish unforgeability against hostile code. Compiler-level sealed receipt
construction remains a separate language feature.

## Semantic lane declaration

```text
Semantic-Lane-ID: psychiatric-regimes-d2-proof-carrying-inference
Owner: Codex implementation under founder direction
Concept-IDs: SOUNIO-PROOF-CARRYING-INFERENCE; SOUNIO-DYADIC-NONREDUCTION; SOUNIO-RELATIONAL-ASSOCIATOR
Intent-Preserved: missing, order, provenance, ambiguity, and claim level cannot disappear silently
Transformation: finite model contest becomes a typed evidence-state transition system
Types-Changed: new stdlib and ontology types only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: one bounded synthetic contest can update, abstain, identify within-family, refute-family, and select a bounded minimax probe
Claims-Forbidden: global truth; causal mechanism; physical intervention; suffering; diagnosis; prognosis; treatment; clinical authority; historical priority
Assumptions: frozen four-model family; exact deterministic predictions; three probes; declared integer burden; complete enumeration only inside those bounds
Write-Set: D2 kernel, ontology, witnesses, negative fixtures, oracle, gate, concept contract, registry, this specification, offload log
Read-Set: D0/D1 surfaces; psychiatric three-model acquisition; longitudinal belief revision; ontology checker examples
Positive-Witness: H3 identified by admissible A=600 then C=650; full family refuted by admissible B=700
Negative-Witness: missing/unaudited evidence cannot update; identification cannot become truth/causality/action; burden cannot become suffering
Acceptance-Gate: scripts/ci/proof_carrying_model_contest_gate.sh
Integration-Target: research/psychiatric-regime-contest-20260712
Authoritative-Only-If: canonical Madaros typechecks reusable surfaces, native witness and ontology execute, Python oracle agrees, negatives reject, D1 regresses green
```

The ontology portion of that acceptance boundary is category-level and
parallel. The executable kernel returns ordinary D2 receipt structs, while the
ontology module and focused fixtures independently encode the corresponding
nominal non-subsumptions. D2 does not yet carry a kernel-produced value into IR
as an ontology-typed result; a future result-identity bridge must provide its
own source-to-IR witness.

## Literature baseline

The observation/intervention boundary is non-negotiable. Pearl's causal
hierarchy and do-calculus distinguish observational conditioning from
interventional quantities; predictive disagreement alone cannot cross that
layer: <https://ftp.cs.ucla.edu/pub/stat_ser/bareinboim-etal-ch27-acm-2021.pdf>.

Predictive state representations define state using action-conditional
predictions of future observations. This supports treating a vector of probe
predictions as a state rival rather than insisting on an inaccessible latent
ontology: <https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf>.

Sequential controlled sensing chooses experiments from the current information
state. D2 uses a finite exact minimax analogue, not the asymptotic stochastic
optimality results of that literature:
<https://arxiv.org/abs/1205.0858> and <https://arxiv.org/abs/1203.4626>.

Mitchell's version-space framing treats learning as search through hypotheses
consistent with examples. D2 retains the entire surviving set instead of
silently selecting one member: <https://doi.org/10.1016/0004-3702(82)90040-6>.

Rubin showed that ignoring the missingness process requires explicit
conditions. Consequently D2 makes a missing observation a different type and
does not treat it as disagreement:
<https://academic.oup.com/biomet/article-abstract/63/3/581/270932>.

Provenance semirings show that derivational information can be propagated
algebraically rather than stored as disposable metadata. D2 implements only a
small ordered provenance ledger, not a general semiring:
<https://www.cs.ucdavis.edu/~green/papers/pods07.pdf>.
Language-integrated provenance in Links further demonstrates that provenance
metadata can be distinguished through types:
<https://arxiv.org/abs/1607.04104>.

Reject-option classification establishes that abstention can be an optimal
decision rather than an error. D2 does not reproduce Chow's probabilistic
theorem, but adopts the narrower programming-language principle that an
ambiguous state has a first-class abstention result:
<https://doi.org/10.1109/TIT.1970.1054406>.

Epistemic programming itself is not claimed as new. EPLAS and later
program-epistemic logics are direct prior art, while SEPIO and newer evidence
ontologies model claims and their supporting evidence. D2's evidence is only
that Sounio can make the following finite inference boundaries executable:
<https://ojs.aaai.org/index.php/AAAI/article/view/25769>,
<https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.

Clinical value-of-information work motivates sequential research designs, but
the D2 burden units are synthetic and cannot be read as patient utility, harm,
or suffering: <https://pubmed.ncbi.nlm.nih.gov/20040743/>.

## Difference from existing Sounio surfaces

`psychiatric_three_model_acquisition_witness.sio` selects a discriminating
probe from three dynamics but does not consume observations into a typed
version space. `clinical_longitudinal_belief_revision_witness.sio` preserves
exact support, provisional elimination, reconsideration, and model refinement,
but does not require admissible observation provenance or distinguish
singleton identification from whole-family failure.

D0 establishes bounded non-reduction. D1 establishes grouping sensitivity and
an expanded-state rival. D2 composes their lesson into an inference lifecycle:

```text
declared alternatives
  -> admissible evidence or typed abstention
  -> exact survivor transition
  -> ambiguity | within-family identification | family refutation
  -> explicit refusal of truth, causality, suffering, and action
```

## Frozen family

Four hypotheses share the same current observable projection. Their internal
predictive modes and grouping codes differ. Predictions are exact thousandths:

| hypothesis | grouping | A | B | C |
|---|---:|---:|---:|---:|
| H0 | left | 400 | 450 | 500 |
| H1 | left | 400 | 550 | 500 |
| H2 | right | 600 | 450 | 500 |
| H3 | right | 600 | 450 | 650 |

Declared probe burdens are `A=2`, `B=3`, and `C=5`. They are ordinal fixture
units only.

No single probe identifies all four hypotheses. Root A produces blocks
`{H0,H1}` and `{H2,H3}`. Probe B separates the first block; probe C separates
the second. The adaptive worst-case burden is therefore:

```text
2 + max(3, 5) = 7
```

Roots B and C cannot complete identification within depth two. The complete
preset word A,B,C has burden `2+3+5=10`. Exhaustion is only over the declared
three-probe, depth-two adaptive policy space.

Within that declared finite space the A-root policy is the unique complete
depth-two policy and therefore bounded-minimax optimal. No optimality is
claimed outside the frozen probes, costs, horizon, or deterministic table.

## Evidence transitions

An admissible observation requires all of:

```text
value present
schema matches probe
protocol matches contest
provenance chain complete
predecessor equals the last accepted provenance identifier
source identifier present
ordered tick after prior evidence
same input for all surviving hypotheses
```

The positive path is:

```text
initial survivors = {H0,H1,H2,H3}
A=600            -> {H2,H3}, burden=2
C=650            -> {H3},    burden=7
```

The conclusion is `FiniteFamilyIdentificationReceipt(H3)`. It means only that
H3 is the sole member of this declared family consistent with both admissible
observations.

Four adversarial paths are mandatory:

1. Missing A leaves all four hypotheses alive and emits abstention.
2. A value without complete provenance cannot enter the update function and
   emits abstention through a separate runtime path.
3. A locally well-formed provenance link whose predecessor is not the last
   accepted link emits abstention and leaves the version space unchanged.
4. Admissible B=700 disagrees with every declared prediction. The result is
   `DeclaredFamilyRefutationReceipt`, not nearest-model selection.

## Falsification and demotion

D2 fails if any of the following occurs:

- a missing, unaudited, or provenance-disconnected observation eliminates a hypothesis;
- hypothesis identifiers, rather than predictions, influence survival;
- relabeling hypotheses changes partitions or policy burden;
- an observation is compared under different probes across hypotheses;
- the adaptive burden or partition differs from the independent oracle;
- empty survival selects a model instead of refuting the family;
- a transition certifies without replaying its source bits, probe, value,
  exact table, counts, burden, and provenance delta;
- singleton survival becomes global truth, causality, suffering, or action;
- the D1 gate regresses.

Even a green D2 does not establish calibrated probabilities, stochastic
optimality, empirical adequacy, a real mental state, causal identification, or
clinical usefulness. It also does not establish runtime kernel-to-ontology
transport. Those require new types, data, assumptions, interfaces, and gates.
