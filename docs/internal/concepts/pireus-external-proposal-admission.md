<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-external-proposal-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-external-proposal-admission
-->

# Pireus external proposal admission

Concept-ID: `SOUNIO-PIREUS-EXTERNAL-PROPOSAL-ADMISSION`
Semantic-Lane-ID: `continuity-20260906`
Owner: `codex-pireus`
Status: executable admission; live Inkling acceptance and materialization pending.

The founder-authorized boundary accepts LLM suggestions as untrusted data.
The Sounio executable `tools/pireus/continuity/admission.sio` parses and admits
the construction and reconstructs its labeled Cayley-Dickson tensor.
Python transports and retains bytes, invokes the native executable, and observes
its receipts. No Python result grants semantic or scientific authority.

The context file is a trusted, frozen input supplied by the research run;
a proposal cannot supply or override its capabilities. Context provenance must
be retained by `cycle.py prepare --evidence`. The CLI accepts a prepared context or invokes research_context.sio. That
producer queries the actual TripleStore/SPARQL machinery and explicitly
adds research-local declared primitive facts, retaining missing, unknown and
contradictory controls. A synthetic context in tests is not observed hardware.

The first grammar varies odd affine lane permutations, direct or shuffle loads,
AoS or SoA layout, and unroll factors 1/2/4/8/16. Admission is distinct from
materialization: Sounio reconstructs and identifies these plans and emits bounded PTX.
GPU observations are compared by material_parity.sio; performance remains a
separate measurement and decision stage.

The numeric contract is dimension 16, f64, no FMA, output k with ascending
right operand j and left i = k XOR j. This is a distinct ordered experiment
from the older ontology operation DAG's ascending-i reduction. Equal integer
structure tensors do not erase this floating-point distinction.

Tensor encoding `cd16-abk-offset1-v1` traverses (a,b,k) in ascending order and
maps coefficients -1, 0, +1 to one byte 0, 1, 2 respectively. Sounio reconstructs
all 4096 coefficients and hashes those bytes. No tensor or expected hash is an
accepted proposal field. Plan identity includes lane map, load, layout and
unroll, so equal tensors do not collapse distinct material plans.

Unknown, contradictory, missing and incompatible capabilities refuse dependent
admission. Duplicate fields, trailing content, non-integer numeric syntax,
snapshot drift, precision/order changes and authority/result injection refuse.
The receipt includes context/proposal hashes and explicitly leaves
`fp_parity=UNMEASURED`, `claim_ready=false`, and V13/V14 formal obligations OPEN.

Positive witness: two distinct lane/layout/schedule constructions with the same
reconstructed tensor, and eight deterministic plans through the persisted CLI.
Negative witness: adversarial proposal bytes, changed dependencies and engine.
Acceptance: `test_admission.py <rebuilt-admission.elf>` and
`test_cycle.py <rebuilt-admission.elf>`; evidence is committed under
`tools/pireus/continuity/validation`.

Material parity now has positive and poisoned-sign negative evidence on both
Sparks. The finite-bits-nan-class-v1 contract compares exact non-NaN bits and
NaN class without claiming payload preservation or general FP proof.
benchmark_decision.sio requires four paired comparisons (both controls on both
nodes), median gain >= 5% and positive lower bootstrap bounds.
Pending acceptance: a real Inkling batch; integrated live performance trial;
new-operator grammar and classification; GRPO corpus. Historical V0–V14 producers
and their authority boundaries remain distinct and retained.
