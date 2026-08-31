<!-- docs:meta
topic_id: repo.docs.internal.concepts.policy-state-feedback
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.policy-state-feedback
-->

# Proof-Carrying Policy-State Feedback


Status: **executable**

Concept-ID: `SOUNIO-POLICY-STATE-FEEDBACK`

## Founder Intent

Evidence is not collected outside the system it describes. A state summary can
govern the next observation policy; that policy can then determine which
evidence remains possible. The resulting absence must retain its policy and
budget provenance instead of being interpreted as a low value, participant
nonresponse, or confirmation of the current summary.

## Executable Core

The D5 kernel freezes two synthetic hypotheses. Both predict anchor value `2`
at tick 1, while their hidden tick-2 targets are `2` and `8`. The common anchor
produces summary code `2`; a deterministic exploit policy withholds the only
discriminating tick-2 probe. The observed and policy traces are therefore equal
under both hypotheses, and both remain live.

The kernel represents that state with separate ambiguity, coverage-gap, and
policy-comparison-nonidentification receipts. One provenance-linked synthetic
exogenous probe can identify the target inside the declared two-hypothesis
family while exactly exhausting a declared burden budget. A further probe is a
typed budget refusal, not another observation.

## Ontology Binding

`stdlib/ontology/policy_state_feedback.sio` distinguishes observed values,
policy summaries and withholding, coverage artifacts, inference artifacts,
budgets, consent, and clinical authority.

This is currently a parallel nominal boundary. The ontology module and its
negative witnesses independently re-express the kernel's distinctions, but a
runtime D5 receipt is not yet transported as an ontology-typed result. A
result-identity bridge requires separate source-to-IR evidence.

## Required Invariants

- An observed anchor is not an observed target at another tick.
- A state summary is not a value observation.
- Policy withholding is not a value or participant nonresponse.
- Target ambiguity and lack of policy coverage are distinct receipts.
- A local coverage probe does not establish statistical positivity or global
  overlap.
- Within-family target identification does not identify a policy value.
- A budget refusal cannot enter the evidence contest.
- Declared burden accounting is not consent, ethics approval, suffering, harm,
  or clinical risk.
- Policy-erased and provenance-disconnected artifacts must abstain.
- Relabeling hypothesis identifiers cannot alter the prediction partition.
- The base-31 provenance fingerprint is a bounded audit convenience, not a
  collision-free identity or authentication proof.

## Claims Forbidden

- The bounded repeated trace is a real causal feedback mechanism or a global
  equilibrium.
- The exploit policy is clinically appropriate, empirically learned, or
  representative of an actual decision system.
- The two-hypothesis family is exhaustive in reality.
- A synthetic exogenous probe is real-person randomization or intervention.
- Target identification in the fixture licenses off-policy evaluation,
  diagnosis, prognosis, monitoring, or treatment.
- A numeric burden cap is evidence of consent, tolerability, distress, or
  suffering.
- A green gate establishes psychiatric validity or priority over adaptive
  sampling, selective-label, performative-prediction, or OPE research.

## Current Surfaces

- `stdlib/epistemic/proof_carrying_policy_state_feedback.sio`
- `stdlib/ontology/policy_state_feedback.sio`
- `scripts/research/proof_carrying_policy_state_feedback_oracle.py`
- `scripts/ci/proof_carrying_policy_state_feedback_gate.sh`
