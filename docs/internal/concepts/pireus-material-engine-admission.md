<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-material-engine-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-material-engine-admission
-->

# Pireus Material Engine Admission

Concept-ID: `SOUNIO-PIREUS-MATERIAL-ENGINE-ADMISSION`

Status: `executable`

Canonical surface:
`stdlib/hardware/pireus/material_engine_admission.sio`

## Meaning

Material engine admission is the Sounio-owned predicate that consumes a sealed
target-local material receipt and adds only the machine and execution engine
that the receipt actually witnessed.

It preserves these distinctions:

```text
canonical target != transport locator
transport locator != observed machine
observed machine != observed execution engine
observed engine != engine blueprint
ISA != execution interface
material parity != semantic authority
engine admitted != cost observed
selector covered != whole operation covered
```

The current frozen overlay admits the CPU engine witnessed on
`Mac17,7 / Apple M5 Max` and the GPU engine witnessed on
`spark-3c59 / NVIDIA GB10` at `192.168.3.24`. It does not admit an Apple GPU,
a DGX CPU, or any machine or engine for `192.168.3.48`.

## Exact Evidence Boundary

An admission requires:

1. the committed Garden and all frozen parent files to match by SHA-256;
2. the target-specific material receipt and evidence pair to match;
3. the receipt, evidence, hardware identity, material operation, target,
   locator, machine, engine kind, ISA, and interface to form one exact record;
4. the producer to remain `C++ / MATERIAL_PARITY` and the semantic authority
   to remain Sounio;
5. material execution to be observed and blueprint-only evidence to be false;
6. selector coverage to be true while whole-operation coverage remains false;
7. cost value, comparison, review promotion, parity, and claim readiness to
   remain closed.

The ontology is then read back by direct triple iteration. Each
admission-defining single-valued `(subject, predicate)` pair in the fixed v1
checklist must have its expected cardinality: exactly one expected IRI and no
other object for the same pair when present, or zero edges for that pair when
absence is part of the coordinate. Newly minted resource IDs are pairwise
distinct and absent from the hash-frozen base. For multi-valued cluster and
eligibility pairs, direct readback requires each newly named member, the frozen
base fixes prior members, the evaluator locks the total cardinality, and the
store digest locks the complete append order. The entire append-ordered store
is committed to a separate SHA-256 digest, including literal values through
their exact `f64_to_bits` representation.

## Unresolved DGX Locator

`demetrios@192.168.3.48` is represented as a transport locator plus a
value-free identity request. It has no `LOCATOR_RESOLVES_MACHINE` edge and no
machine or engine ID. A prior route failure cannot be promoted to absence,
identity, or equivalence with `.24`.

`PIREUS_CLUSTER_DARWIN` is the founder-named heterogeneous fleet cluster, not
the Darwin operating-system target. Membership in that cluster does not merge
the Apple, DGX, or Xeon target identities.

## Cost Boundary

The admitted Apple CPU and DGX `.24` GPU engines receive request-eligibility
rows for dependency latency and reciprocal throughput. These are permissions
to issue a complete target-local measurement request. They are not measured
values, summaries, comparisons, rankings, speedups, or lowering preferences.

## Frozen Semantics

The exact counts, record order, negative witnesses, store digest, result digest,
and authority boundaries are fixed in
`docs/research/pireus_material_engine_admission_semantics.md`.

The dedicated gate is:

```bash
bash scripts/ci/pireus_material_engine_admission.sh
```

## Claims Forbidden

- Apple GPU material identity;
- DGX CPU material identity;
- any `.48` machine or engine identity;
- whole-operation coverage;
- a cost, performance, equivalence, or lowering result;
- parity or claim-ready promotion;
- external LLM confirmation;
- Python or Rust authority or oracle use.

## Semantic Outcome

Semantic-Outcome: additive receipt-bound material identity overlay

Concept-Status-Before: `garden`

Concept-Status-After: `executable`

Distinctions-Added: locator versus machine; receipt-bound engine versus
blueprint; engine admission versus request eligibility versus observed cost

Distinctions-Preserved: target, machine, engine, kind, ISA, interface,
evidence role, operation scope, semantic authority, and claim stage

Distinctions-Erased: none

Evidence-Run: first Sounio authority stream, dedicated Sounio test, live parent
hashes, exact triple readback, ordered ontology digest, and negative mutations

Fallback-Path: none; the frozen executable uses explicit `lean_single` routing

Legacy-Kept: the base execution-engine ontology and target-cost ledger remain
unchanged and are consumed as frozen parents

Conflicting-Lanes: none observed at lane start

Next-Semantic-Interface: target-local realization of one request-eligible cost
coordinate; separate material identity attempt for `.48`
