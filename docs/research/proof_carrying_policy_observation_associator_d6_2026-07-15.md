<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-policy-observation-associator-d6-2026-07-15
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-policy-observation-associator-d6-2026-07-15
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Policy-Observation Associator D6

Status: executable finite synthetic specification
Date: 2026-07-15
Concept-ID: `SOUNIO-POLICY-OBSERVATION-ASSOCIATOR`

## Research Question

For one declared finite policy-observation operator, can formal grouping change
the synthetic epistemic result when operand identity, operand order, operand
type, and the binary composition rule are all held fixed?

D6 answers yes for one declared finite partial operator. It consumes the D5
policy, coverage, burden, and provenance categories, adds an explicit
evidence-commit boundary, and exhibits a composable triple for which:

```text
((a * b) * c) != (a * (b * c))
```

This is a counterexample for a partial operation on one composable triple. It
is not a theorem about every observation system, ordinary function
composition, monadic bind, a total magma, a patient, or psychiatric treatment.

D6 is an explicit existence construction. The custody priority was deliberately
chosen to make scope operational and, on this witness, to violate associativity.
It is published as part of the operator, not inferred from data or presented as
an emergent law of observation. D5 motivates keeping policy, commitment,
burden, and provenance distinct, but a different declared operator could be
associative. The programming-language contribution is the typed, replayable,
claim-bounded representation of that construction, not the bare fact that some
hand-defined partial operation can be non-associative.

## Executable Surfaces

This specification accompanies, rather than substitutes for, the executable
artifacts:

- kernel: `stdlib/epistemic/proof_carrying_policy_observation_associator.sio`;
- imported API witness:
  `tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio`;
- native witness:
  `tests/run-pass/clinical_proof_carrying_policy_observation_associator_native_witness.sio`;
- independent oracle:
  `scripts/research/proof_carrying_policy_observation_associator_oracle.py`;
- recursive gate:
  `scripts/ci/proof_carrying_policy_observation_associator_gate.sh`.

These are Sounio and Python artifacts, not Lean modules. “Proof-carrying” here
means source-level bounded receipt validators, a separately executed scalar
native witness, an independent enumerator, and compiler-enforced category
boundaries, not a proof-assistant soundness theorem. On the current Madaros
multimodule path the imported reusable kernel is typechecked conformance
evidence; direct imported execution fails at thin-link and is not runtime
evidence.

The ontology evidence is a parallel nominal boundary. The executable D6
kernel returns ordinary receipt structs, while
`stdlib/ontology/policy_observation_associator.sio` and its focused fixtures
independently encode corresponding nominal non-subsumptions. No kernel-produced
D6 value is currently carried into IR as an ontology-typed result, and the gate
does not imply such transport.

The ontology run-pass witness redeclares a focused local nominal fixture and
prints a conformance sentinel. It does not import a kernel-produced receipt or
execute an ontology query, so its output is category-level compiler evidence,
not runtime ontology-result evidence.

`SOUNIO-REBRACKETING-AUTHORITY` remains the separate compiler-level authority
concept. This scientific counterexample demonstrates that one declared
operation is grouping-sensitive; its receipt cannot authorize, prohibit, or
certify an optimizer rewrite without the compiler's own transformation
certificate.

## Frozen Operands

All operands have type `PolicyObservationCarrierReceipt`. Their order is fixed.

| Operand | ID | Position | Meaning | Origin mask |
| --- | ---: | ---: | --- | ---: |
| `a` | 9101 | 1 | D5 policy context plus committed anchor evidence | 1 |
| `b` | 9102 | 2 | evidence-commit boundary | 2 |
| `c` | 9103 | 3 | pending D5 synthetic target probe | 4 |

The policy atom retains the exact D5 anchor state: target-family mask `3`,
burden `3`, and provenance `8101`. It also retains adaptive decision `8102`,
which considered but withheld the target probe, and coverage gap `8301`.

The pending-probe atom retains the D5 synthetic exogenous assignment `8600`,
cost `4`, target value `8`, and prospective provenance `8103 -> 8101`. Pending
does not mean observed. The commit boundary is a synthetic evaluation boundary,
not a laboratory, biological, legal, or clinical event.

## One Partial Operator

`compose_policy_observation_carriers` is the only semantic carrier operator in
the witness. Its domain consists of adjacent, disjoint fragments of the frozen
ordered program. A composability receipt replays:

```text
left.last_position + 1 = right.first_position
left.origin_mask & right.origin_mask = 0
same family, protocol, operator, and budget
```

The API also carries `pair_id`, `result_id`, and a composability receipt. Those
are audit/custody parameters: the resolution, evidence, burden, flat-trace, and
tree recurrences do not branch on their numeric values. They do not select a
different semantic operator for either parenthesization.

Both inner pairs `a*b` and `b*c` are in the domain. Both outer pairs
`(a*b)*c` and `a*(b*c)` are also in the domain. Thus the inequality is not
produced by making one parenthesization undefined.

For a pending probe on the right, the resolution rule is:

```text
boundary in left and policy in left     -> WITHHELD
boundary in left and no policy in left  -> COMMITTED
already COMMITTED in either input       -> COMMITTED
already WITHHELD in either input        -> WITHHELD
otherwise                               -> PENDING
```

The first rule that applies is used in the displayed order. In particular,
already committed evidence wins before an outer policy is considered. This is
a declared custody rule of the frozen D6 operator. Committed evidence is
monotone only inside the declared fixture; D6 does not infer a universal law of
evidence, memory, records, or persons.

The grouping-tree fingerprint is audit output, not an input to the resolution
rule. Semantic divergence is independently witnessed by probe status, survivor
mask, burden, evidence count, and provenance fingerprint.

## Exact Counterexample

### Left grouping

`a*b` combines the policy and boundary before the probe is present. When `c`
arrives, both policy and boundary occur in the left carrier, so the probe is
withheld:

```text
grouping                  = ((a*b)*c)
probe status              = WITHHELD (2)
observed target           = absent
survivor mask             = 3
burden                    = 3
evidence count            = 1
evidence fingerprint      = 8101
grouping tree fingerprint = 9037326
```

### Right grouping

`b*c` resolves the pending probe while no policy is in that inner left carrier,
so it commits target `8` with provenance `8103`. Composing the outer policy
adds the anchor evidence but cannot retroactively erase the probe:

```text
grouping                  = (a*(b*c))
probe status              = COMMITTED (3)
observed target           = 8
survivor mask             = 2
burden                    = 7
evidence count            = 2
evidence fingerprint      = 8101 * 31 + 8103 = 259234
grouping tree fingerprint = 573396
```

The result difference is a diagnostic bitset, not a metric or algebraic
associator:

```text
status differs         -> 1
survivor mask differs  -> 2
burden differs         -> 4
evidence count differs -> 8
total                  -> 15
```

Because the carrier has no declared additive group, D6 proves direct
inequality. It does not subtract the two states or claim a scalar algebraic
associator.

## Associative Control and Expanded-State Rival

The ordered leaves are checksummed with associative base-101 concatenation:

```text
flat(x ++ y) = flat(x) * 101^(leaves(y)) + flat(y)
```

Both parenthesizations therefore produce the same flat receipt:

```text
flat((a ++ b) ++ c) = flat(a ++ (b ++ c)) = 93767706
```

The carriers and final witness separately retain the three ordered operand IDs;
the checksum is not used to infer their identity. The kernel and independent
oracle also encode all six numeric label orders as a checksum-only control.
Their values are `93767706`, `93767806`, `93777806`, `93778006`, `93788006`,
and `93788106`, so there are zero collisions in this finite set. These are not
six admissible operator compositions and do not prove semantic relabel
invariance. Base 101 is not claimed to be generally injective for arbitrary IDs
or lengths. All six bounded values are checked against signed-64-bit limits.

The state-expansion rival then retains the binary grouping tree. Its audit
recurrence is `tree(left) * 31 + tree(right)`, producing `9037326` and `573396`.
The flat payloads are equal, but the projected receipts retain different
carrier IDs and therefore are not identical receipts. Once grouping is part of
the payload, the two expanded payloads are distinct. An explicit two-case
frozen-tree replay recovers `WITHHELD` from `9037326` and `COMMITTED` from
`573396`. D6 therefore does not claim irreducible memory or a contradiction
between identical complete states. The projection that erases grouping is
exactly where outcome factorability is lost.

The tree fingerprint itself is deliberately non-associative and is not used as
evidence for the semantic counterexample. Status, mask, burden, and evidence
would still differ if tree fingerprints were omitted.

Neither the base-31 tree/evidence recurrences nor the base-101 flat recurrence
is claimed to be a generally collision-free identity or authentication scheme.
The six explicit flat values establish non-collision only for the frozen
three-label-order family; exact ordered operand and provenance fields remain
the authoritative replay inputs.

## Exhaustive Oracle

The independent oracle generates every full binary tree over the frozen
ordered leaves. There are exactly two trees and four unique binary applications:

```text
a*b
(a*b)*c
b*c
a*(b*c)
```

It replays the operator rather than importing Sounio outputs. It verifies both
semantic results, both fingerprints, the difference bitset, the associative
flat control, all six label-order checksum bounds, five invalid-pair controls,
and every reachable composition whose input already contains committed probe
evidence. There is one such outer application and zero erasures.

This is exhaustive only for the three-atom frozen program. It is not exhaustive
over arbitrary policies, observations, handler calculi, or clinical histories.

## Typed Non-Equivalences

Madaros must reject all of the following category substitutions:

- non-associativity witness -> causal policy-observation mechanism;
- non-associativity witness -> empirical psychiatric order effect;
- non-associativity witness -> clinical action;
- partial operator -> total associative magma;
- withheld target -> committed synthetic observation;
- committed observation -> statistical positivity;
- non-associativity witness -> consent or suffering;
- flat trace -> grouping-retained state;
- flat trace -> complete-state equivalence;
- policy withholding -> real participant nonresponse;
- committed-evidence monotonicity -> causality.

The ontology independently keeps carriers, observation resolutions,
composition laws, state representations, evidence boundaries, and empirical or
clinical interpretations as sibling categories. Its negative gates repeat the
most important non-subsumptions.

## Literature Compass

The literature supplies a compass, not empirical validation of D6:

- Moggi, *Notions of Computation and Monads* (1991), develops monadic calculi
  as a basis for reasoning about computational equivalence across effects:
  https://doi.org/10.1016/0890-5401(91)90052-4
- Plotkin and Pretnar, *Handling Algebraic Effects* (2013), give algebraic
  effects and handlers a model-theoretic treatment and connect free models to
  computational monads:
  https://lmcs.episciences.org/705
- Wu, Schrijvers, and Hinze, *Effect Handlers in Scope* (2014), explicitly
  study scoped constructs and show that reordering handlers can give different
  semantics:
  https://doi.org/10.1145/2633357.2633358
- Green, Karvounarakis, and Tannen, *Provenance Semirings* (2007), show how
  algebraic annotations can retain how database results depend on source data:
  https://doi.org/10.1145/1265530.1265535
- Lakkaraju et al., *The Selective Labels Problem* (2017), show how observed
  outcomes can depend on prior decisions rather than form a random sample:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915/
- Zhan et al., *Policy Learning with Adaptively Collected Data* (2022), analyze
  statistical dependence induced by adaptive collection:
  https://arxiv.org/abs/2105.02344

D6 does not implement Moggi's monadic bind, the Plotkin-Pretnar calculus, Wu et
al.'s handler language, or provenance semirings. Those works sharpen the rival
models and caution against confusing a scoped domain operator with ordinary
associative computational composition. The selective-label and adaptive-data
papers motivate why observation policy belongs in the state; they do not
establish D6's synthetic transition table or a psychiatric effect.

More precisely, Moggi is the associative computational-composition rival that
D6 refuses to target; Plotkin and Pretnar motivate separating effect operations
from their interpretations; Wu et al. supply the direct technical warning that
scope and handler order can change semantics; Green et al. motivate retaining
derivation information rather than treating an output as provenance-free; and
the adaptive-data papers motivate treating observation policy as part of the
data-generating state. None entails D6's priority table.

## Relation to Existing Sounio Associators

D1 already exhibits a nonzero exact associator for a weighted rational
mediation rule. The effectful reset-grouping fixture separately demonstrates
scoped token placement in a synthetic dynamical state. D6 does not duplicate or
broaden either claim. Its new surface is the direct binding of D5 policy,
coverage, burden, and provenance receipts to one partial composition law, plus
a monotone committed-evidence boundary and corresponding compile-time category
barriers.

The D2-D6 kernels use ordinary library receipt structs. They do not instantiate
native `Contest<T, ...>` syntax, construct `TyContest` or `IrContest`, or bind
the compiler-owned contest index. Likewise, a D6 receipt is scientific evidence
about one library operator, not a compiler rebracketing capability.

## Hard Boundaries

- D6 does not establish a universal observation algebra.
- D6 does not establish an empirical psychiatric order effect.
- D6 does not establish a causal feedback mechanism.
- D6 does not establish statistical positivity, overlap, or policy value.
- Synthetic burden is not tolerability, harm, suffering, or preference.
- Withholding by a fixture policy is not participant refusal or nonresponse.
- The pending and committed probes are not real-person measurements.
- The evidence-commit boundary is not consent, ethics approval, or a legal act.
- No receipt authorizes diagnosis, monitoring, prognosis, or treatment.

## Acceptance Gate

`scripts/ci/proof_carrying_policy_observation_associator_gate.sh` must:

1. resolve canonical `bin/souc` to Madaros without engine fallback;
2. typecheck the kernel, ontology, and imported API witness;
3. execute the exact native witness and match every output receipt;
4. execute the independent exhaustive oracle;
5. observe every clinical and ontology category error as compiler rejection;
6. verify the Concept-ID and literature/claim boundaries;
7. recursively execute D5, which recursively covers D4 through D0.
