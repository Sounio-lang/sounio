<!-- docs:meta
topic_id: repo.docs.internal.concepts.endogenous-observability
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.endogenous-observability
-->

# Proof-Carrying Endogenous Observability


Status: **executable**

Concept-ID: `SOUNIO-ENDOGENOUS-OBSERVABILITY`

## Founder Intent

An observation is not merely a value slot. Whether a measurement was
considered, scheduled, delivered, answerable, answered in time, answered late,
or withheld by policy belongs to the epistemic state. Collapsing those events
to one missing token can erase evidence and manufacture false certainty.

## Executable Core

The D4 kernel represents four frozen synthetic hypotheses:

- scheduled but not delivered;
- delivered with declared target-independent window nonresponse;
- delivered with declared target-dependent response suppression;
- withheld by the observation policy.

All four coarsen to the same legacy missing token. A complete custody trace
separates delivery failure and policy withholding, but the two response
hypotheses remain observationally equivalent. D4 emits an ambiguity receipt at
that boundary. A provenance-linked synthetic retry may identify one hypothesis
only within the declared family.

## Ontology Binding

`stdlib/ontology/endogenous_observability.sio` distinguishes policy
withholding, delivery failure, participant nonresponse, delayed response,
observed value, observation-process ambiguity, biological mechanism, consent,
and clinical authority.

This is currently a parallel nominal boundary. The ontology module and its
negative witnesses independently re-express the kernel's distinctions, but a
runtime D4 receipt is not yet transported as an ontology-typed result. A
result-identity bridge requires separate source-to-IR evidence.

## Required Invariants

- Measurement intent, policy, delivery, opportunity, response, and value are
  distinct artifacts.
- No response opportunity is not participant nonresponse.
- Window nonresponse means no response within the declared window, not no
  eventual response.
- A delayed value is aligned to its arrival tick unless an explicit temporal
  model authorizes another alignment.
- A legacy missing token cannot update the observation contest.
- Policy-erased or disconnected evidence must abstain.
- The base-31 provenance fingerprint is a bounded audit convenience, not a
  collision-free identity or authentication proof.
- Observational equivalence is not empirical mechanism identification or
  global recoverability.
- A retry can discriminate only inside the frozen family.
- Declared burden is not suffering, harm, or clinical risk.

## Claims Forbidden

- The deterministic modes establish MCAR, MAR, MNAR, ignorability, or a
  biological response mechanism.
- The four-hypothesis family is exhaustive in reality.
- A synthetic exogenous retry is not real-person randomization, consent, or an
  intervention.
- Nonresponse is a symptom, latent target value, or zero.
- A delayed response is a contemporaneous value at the original prompt.
- A within-family selection authorizes diagnosis, prognosis, monitoring, or
  treatment.
- A green gate establishes empirical psychiatric validity or historical
  priority over missing-data, informative-visiting, or active-sensing work.

## Current Surfaces

- `stdlib/epistemic/proof_carrying_endogenous_observability.sio`
- `stdlib/ontology/endogenous_observability.sio`
- `scripts/research/proof_carrying_endogenous_observability_oracle.py`
- `scripts/ci/proof_carrying_endogenous_observability_gate.sh`
