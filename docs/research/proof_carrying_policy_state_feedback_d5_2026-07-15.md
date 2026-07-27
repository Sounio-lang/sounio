<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-policy-state-feedback-d5-2026-07-15
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-policy-state-feedback-d5-2026-07-15
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Policy-State Feedback D5

Status: executable finite synthetic specification
Date: 2026-07-15
Concept-ID: `SOUNIO-POLICY-STATE-FEEDBACK`

## Research Question

What must a scientific language preserve when an evidence summary governs the
next observation policy, and that policy can remove the very observation that
would challenge the summary?

D5 makes one narrow answer executable: it gives distinct types to an observed
anchor, an unobserved target, a state summary, policy withholding, a coverage
gap, target ambiguity, a budget refusal, and within-family identification. The
compiler is then asked to reject every coercion between those categories.

This is a synthetic epistemic kernel. It is not a patient model, a psychiatric
theory, a causal discovery result, an OPE estimator, or a clinical pathway.

## Frozen Construction

The declared family contains exactly two fixture hypotheses:

| Hypothesis | ID | tick-1 anchor | hidden tick-2 target |
| --- | ---: | ---: | ---: |
| low target | 510 | 2 | 2 |
| high target | 511 | 2 | 8 |

The family begins at survivor mask `3` (`01 | 10`). The observed anchor is `2`,
costs 3 declared units, and carries provenance `8101`. It is predicted by both
hypotheses, so the survivor mask remains `3`.

The state-summary function emits code `2` from the observed anchor only. The
summary explicitly says that no hidden target was included and that the target
is not identified. The frozen adaptive exploit policy maps summary code `2` to:

```text
candidate considered = true
eligible              = false
scheduled             = false
withheld by policy    = true
```

The same summary produces the same policy decision in both worlds. With no new
observation, repeating that composition twice yields the same summary and the
same withholding decision. D5 calls this an `absorbing_within_frozen_fixture`
witness. It does not call it a global equilibrium or causal mechanism.

## The Collision

The low and high worlds have identical admissible logged traces:

```text
anchor=2 -> summary=2 -> policy=withhold -> no tick-2 value
```

Yet their hidden tick-2 targets are `2` and `8`. Therefore:

- both target hypotheses survive (`mask=3`);
- target ambiguity is present;
- the discriminating cell has zero eligible and zero observed opportunities;
- support for comparing the logging policy with a probing policy is absent;
- statistical positivity and global overlap are not established;
- neither target-policy value nor counterfactual policy ordering is identified.

These are related facts, but D5 refuses to collapse them. Ambiguity describes
the surviving target family. Coverage describes the observation opportunities
created by the logging policy. Policy-value nonidentification describes a
counterfactual comparison that the log cannot support.

## Exogenous Coverage Witness

A synthetic assignment outside the adaptive policy authorizes one tick-2
coverage probe. The assignment is exogenous only inside the frozen fixture; it
does not claim real-person randomization, consent, intervention, or clinical
authority.

The declared budget is exact:

```text
capacity       = 7
anchor cost    = 3
remaining      = 4
coverage probe = 4
final spent    = 7
```

The probe has provenance `8103`, linked to anchor provenance `8101` while
retaining the overridden policy decision `8102`. The evidence fingerprint is:

```text
8101 * 31 + 8103 = 259234
```

Probe value `2` yields mask `1` and selects hypothesis `510`. Probe value `8`
yields mask `2` and selects hypothesis `511`. This is identification only
inside the declared family, where the map from target value to mode is
explicitly injective on the declared support `{2, 8}`. One covered synthetic cell does not establish
statistical positivity, global overlap, empirical model validity, or the value
of either policy.

A second four-unit probe finds zero units remaining. Its result is a
`BudgetExceededObservationActionReceipt` with `action_executed=false`. A budget
refusal is not consent withdrawal, subjective suffering, participant
nonresponse, or a missing target value.

## Executable Claims

1. `C1`: Equal anchor observations preserve both hypotheses.
2. `C2`: Equal summaries induce equal withholding decisions in the two worlds.
3. `C3`: Two bounded iterations replay the same summary-policy trace.
4. `C4`: The logged trace cannot distinguish hidden targets `2` and `8`.
5. `C5`: Zero eligible and observed discriminating opportunities produce a
   coverage-gap receipt, not a positivity receipt.
6. `C6`: One exogenous probe maps values `2` and `8` to masks `1` and `2`.
7. `C7`: Costs `3 + 4` exactly exhaust cap `7`; another probe is refused.
8. `C8`: Evidence provenance is order-sensitive and reaches fingerprint
   `259234` without silent truncation.
9. `C9`: The partition is invariant under both permutations of hypothesis IDs.
10. `C10`: Policy-erased and disconnected evidence preserve mask `3` and
    abstain.

The independent oracle enumerates all `2^4 * 3 = 48` Boolean/value custody
tuples for a prospective probe. Exactly two are admissible in this fixture:
fully considered, exogenously authorized, scheduled, present values `2` or `8`.
It also checks both identifier permutations and the maximum two-link
fingerprint `32,000,000` for signed-64-bit safety.

Each private contest transition retains the complete source recurrence and the
evidence payload: family, protocol, budget, masks, counts, burden, tick, value,
provenance link, policy decision, and result state. Consumers re-run that
transition validator before reading the result or emitting within-family
identification. The exogenous assignment is also linked to the exact coverage
gap and logging-policy decision it overrides.

## Typed Non-Equivalences

The negative suite requires Madaros to reject all of the following:

- state summary -> observed target value;
- policy withholding -> observed target value;
- observed tick-1 anchor -> observed tick-2 target;
- policy coverage gap -> statistical positivity;
- policy-comparison nonidentification -> off-policy value identification;
- synthetic budget -> consent, ethics approval, or suffering;
- budget-exceeded action -> admissible coverage probe;
- bounded feedback witness -> causal feedback mechanism;
- within-family target identification -> clinical policy action;
- synthetic assignment -> real-person randomization;
- disconnected probe -> admissible evidence;
- policy-erased trace -> bounded feedback witness.

The ontology independently represents observed values, policy artifacts,
coverage artifacts, inference artifacts, and budget artifacts as sibling
categories. Its negative gates reject withholding as a value, a coverage gap as
positivity, and a budget as consent.

This ontology evidence is a parallel nominal boundary. D5's executable kernel
returns ordinary receipt structs, while the ontology module and focused
fixtures independently encode corresponding nominal non-subsumptions. No
kernel-produced D5 value is currently carried into IR as an ontology-typed
result, and the gate does not imply such transport.

The bounded base-31 provenance fingerprint is an audit convenience. It is not
collision-free identity, cryptographic authentication, or a substitute for the
exact predecessor, policy-decision, and provenance fields replayed by each
transition.

## Literature Compass

The literature motivates the distinctions; it does not validate the fixture:

- Lakkaraju et al., *The Selective Labels Problem* (2017), show that observed
  outcomes can be consequences of prior decisions rather than a random sample:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915/
- Zhan et al., *Policy Learning with Adaptively Collected Data* (2022), note
  that adaptive collection creates dependence and can leave treatments without
  enough observations for some individual types:
  https://arxiv.org/abs/2105.02344
- Wang, Agarwal, and Dudik, *Optimal and Adaptive Off-policy Evaluation in
  Contextual Bandits* (ICML 2017), formalize evaluation of a target policy from
  data collected by another policy and quantify the difficulty of the agnostic
  setting: https://proceedings.mlr.press/v70/wang17a.html
- Kato et al., *A Practical Guide of Off-Policy Evaluation for Bandit Problems*
  (2020), emphasize that theoretical OPE guarantees require conditions on both
  the target and data-generating policies:
  https://arxiv.org/abs/2010.12470
- Perdomo et al., *Performative Prediction* (ICML 2020), study the different
  case where predictions used for decisions can influence their target:
  https://proceedings.mlr.press/v119/perdomo20a.html
- Schneider et al., *Just-in-time adaptive ecological momentary assessment*
  (2024), motivate adaptive item selection as a response to the tension between
  participant burden and momentary classification precision:
  https://link.springer.com/article/10.3758/s13428-023-02083-8

D5 is narrower than each connection. Its policy controls observability, not the
latent target itself; its budget is declared synthetic accounting, not an
empirical burden model; and its single-cell coverage witness is not an OPE
identifiability theorem.

## Non-Associativity Boundary

The fixture exposes order-sensitive composition: summarizing the anchor and
then withholding leaves mask `3`, whereas appending the admissible exogenous
observation yields mask `1` or `2`. That is the computational terrain in which
a typed non-associative observation algebra may be developed. D5 does not yet claim a formal associativity counterexample because the compared compositions
contain different admissible actions. A future step must define one common
binary composition operator and then exhibit `((a * b) * c) != (a * (b * c))`
under identical operands and typing rules.

## Hard Boundaries

- D5 does not establish statistical positivity or global overlap.
- D5 does not estimate a policy value, treatment effect, symptom trajectory,
  latent state, or causal mechanism.
- The bounded absorbing trace is fixture-specific, not a universal statement
  about adaptive assessment.
- The family is deliberately non-exhaustive in reality.
- Numeric burden is not empirical burden, tolerability, harm, or suffering.
- A budget refusal is not consent withdrawal.
- The synthetic exogenous assignment is not a clinical experiment.
- No receipt authorizes diagnosis, prognosis, monitoring, or treatment.

## Acceptance Gate

`scripts/ci/proof_carrying_policy_state_feedback_gate.sh` must:

1. resolve canonical `bin/souc` to Madaros with no fallback;
2. typecheck the kernel, ontology, and imported API witness;
3. execute the native witness and exact output receipts;
4. execute the independent exhaustive oracle;
5. observe all clinical and ontology category errors as compiler rejections;
6. verify the executable Concept-ID;
7. recursively run D4, which recursively covers D3 through D0.
