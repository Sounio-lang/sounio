<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-28-pireus-material-engine-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-28-pireus-material-engine-admission
-->

# Garden Seed: Pireus Material Engine Admission

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: founder

Concept-ID: `SOUNIO-PIREUS-MATERIAL-ENGINE-ADMISSION`

Semantic-Lane-ID: `pireus-material-engine-admission-20260828`

## Butterfly

> "Mac e `sounio-language-macbook` na tailnet. Apple e DGX sao alvos
> canonicos tambem."

Pireus already separates canonical target, machine, execution engine, ISA,
interface, and target blueprint. It also has two sealed target-local material
receipts: one AArch64 `TBL` realization on the Apple M5 Max and one CUDA
`SHFL.BFLY` realization on the NVIDIA GB10 at `192.168.3.24`.

The missing bridge is admission. A material receipt must be able to add the
machine and engine that it actually witnessed without turning a network
locator into hardware identity, a canonical target into a concrete machine,
or one observed machine into all machines with a similar name.

The Pireus form is:

```text
MaterialEngineAdmission<
    canonical_target,
    transport_locator,
    observed_machine,
    observed_engine,
    engine_kind,
    isa,
    interface,
    hardware_receipt,
    material_receipt,
    evidence
>
```

Every axis is part of the claim. An absent axis remains absent.

## Question

How can Pireus consume already sealed material-parity receipts as evidence for
exactly the engines that executed, while refusing to infer Apple GPU identity,
DGX CPU identity, or a second DGX machine from a sibling locator?

## Existing Frozen Parents

The first executable child must consume, not revise, these artifacts:

| Parent | Role | SHA-256 |
| --- | --- | --- |
| execution-engine source | machine, engine, ISA, interface, and blueprint distinction | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| execution-engine semantics | frozen Darwin multi-engine meaning | `c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233` |
| execution-engine receipt | frozen Sounio authority record | `9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd` |
| target-cost source | target-local request and observation contract | `7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc` |
| target-cost semantics | frozen value and comparability boundary | `0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199` |
| target-cost receipt | frozen Sounio authority record | `b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca` |
| target-cost evidence | flat frozen request record | `06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b` |
| Apple A64 material receipt | sealed target-local `TBL` execution and identity | `c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840` |
| Apple A64 material evidence | flat Apple parity record | `2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e` |
| DGX PTX material receipt | sealed target-local `SHFL.BFLY` execution and identity | `3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385` |
| DGX PTX material evidence | flat DGX parity record | `2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b` |

The child must live-check executable Sounio parents where the current compiler
path permits it and hash-check every documentary parent. A mismatch is a typed
refusal, never a fallback.

## Identity Layers

The admission predicate preserves:

```text
canonical target != transport locator
transport locator != observed machine
observed machine != observed execution engine
observed execution engine != engine blueprint
ISA != execution interface
material parity != semantic authority
```

A locator says where a request may be transported. It does not say what will
answer, whether the endpoint is reachable, or which engine will execute.

## Apple Material Boundary

The sealed Apple receipt binds:

```text
login locator: demetrios@sounio-language-macbook
tailnet identity: sounio-language-macbook
hostname: Sounio-Language-MacBook
canonical target: apple_silicon
machine model: Mac17,7
observed CPU: Apple M5 Max
architecture and ISA: arm64 / AArch64
material operation: A64 TBL
hardware SHA-256: 49702cf6d0b079bf52bf26f98f377266e41d4ce232fea99eb80c30d6554dbc28
```

This admits the observed Apple CPU engine only. It does not admit an Apple GPU
engine, a Metal execution engine, a GPU ISA, or whole-operation coverage.
`Metal` remains an interface term, not an ISA or a witnessed engine.

## DGX Material Boundary

The sealed DGX receipt binds:

```text
login locator: demetrios@192.168.3.24
transport address: 192.168.3.24
hostname: spark-3c59
canonical target: dgx
host architecture: aarch64
observed GPU: NVIDIA GB10
GPU ISA: NVIDIA SM121
execution interface: CUDA
material operation: PTX SHFL.BFLY / SASS SHFL.BFLY
hardware SHA-256: 8b048f0a20ac0967af5622606935aa4ea4e6caf0baef6a3dcd9b7ff58f2a66d4
```

This admits the observed DGX GPU engine only. It does not admit a DGX CPU
engine, turn `CUDA` into an ISA, or prove complete-operation coverage.

The locator `demetrios@192.168.3.48` remains a value-free material identity
request. Prior `No route to host` observations do not identify a machine or
engine behind it. Shared `DGX`, `GB10`, subnet, owner, or canonical-target
labels may not transplant the `.24` identity onto `.48`.

## Admission Predicate

An engine may enter the observed-engine overlay only when all of these hold:

1. the Garden and frozen parents match by hash;
2. the material receipt is sealed and has `Parity-Receipt-Valid=true`;
3. the producer role is `MATERIAL_PARITY` and Sounio remains the declared
   semantic authority;
4. canonical target, locator, machine, engine kind, ISA, interface, hardware
   hash, operation subject, and evidence hash agree with the exact receipt;
5. the receipt names an execution on that engine rather than only a blueprint,
   static corpus, compiler listing, or unreachable endpoint;
6. the receipt is not being reused for a sibling engine, machine, or locator;
7. the requested ontology assertions do not exceed the receipt's material
   boundary.

Admission is monotone over the frozen base ontology. Existing Darwin machines
and engines remain unchanged. Rejection adds no identity fact.

## Cost Interface

Material engine admission makes a target/engine coordinate eligible to receive
a value-free target-cost request. It does not produce a cost value and does not
make a request comparable with another target.

```text
engine admitted != cost observed
selector executed != whole operation covered
same quantity and unit != comparable experiment
```

The first child therefore emits no material cost observation, summary,
comparison, speedup, ranking, or lowering preference.

## First Sounio Executable

The first child should be:

```text
stdlib/hardware/pireus/material_engine_admission.sio
examples/pireus_material_engine_admission.sio
tests/stdlib/hardware/test_pireus_material_engine_admission.sio
```

It must:

1. bind the eleven frozen parent artifacts above and this committed Garden;
2. construct an additive observed-machine and observed-engine overlay;
3. encode the exact Apple CPU and DGX `.24` GPU receipt coordinates;
4. retain `.48` as an unresolved locator and value-free identity request;
5. keep Apple GPU and DGX CPU as blueprints only;
6. distinguish AArch64 and SM121 ISAs from Metal and CUDA interfaces;
7. expose explicit admission and rejection records;
8. make admitted engines eligible only for value-free target-cost requests;
9. emit no cost value, comparison, speedup, ranking, or claim promotion;
10. define in Sounio a canonical record order and integer serialization, then
    commit every emitted record and negative witness to a deterministic digest;
11. keep `PARITY_OPEN=false` and `CLAIM_READY=false` for the child overlay.

No expected record count, result digest, target digest, negative count, or
request count is defined here. Those values may first appear only in the
Sounio executable stream after this Garden is committed.

## Negative Surface

The child must include mutation witnesses for at least:

1. missing Garden binding;
2. missing or mismatched frozen parent;
3. material receipt/evidence hash drift;
4. canonical target transplanted across receipts;
5. locator promoted to machine identity;
6. blueprint promoted without material execution;
7. Apple CPU receipt promoted to Apple GPU admission;
8. Metal promoted to an ISA;
9. Apple receipt transplanted to DGX;
10. DGX `.24` receipt transplanted to `.48`;
11. shared GB10 or DGX label used as sibling identity;
12. DGX GPU receipt promoted to DGX CPU admission;
13. CUDA promoted to an ISA;
14. DGX receipt transplanted to Apple;
15. selector receipt promoted to whole-operation coverage;
16. material parity promoted to semantic authority;
17. static vendor or compiler evidence promoted to material execution;
18. engine admission promoted to a cost observation;
19. engine admission promoted to cross-target comparability;
20. external review promoted to authority or confirmation;
21. premature parity or claim readiness;
22. Python or Rust producer/oracle request reaching interpreter launch.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This Garden must be committed before an expected child result is written. No
Lean, Koka, C++, Haskell, benchmark, remote capture, or new target execution
may run for the child before the Sounio result and semantics are frozen by
hash. The historical Apple and DGX receipts are immutable parents, not new
child executions.

## Enforcement

The existing Loom language-authority guardian remains the only stage and role
enforcer. The child gate must:

- authorize Sounio execution before launch;
- fail closed on missing policy, error, timeout, hash drift, or parent drift;
- log every `ALLOW` or `DENY` decision and reason;
- verify source, semantics, Garden, parent, toolchain, hardware, command, and
  result hashes;
- run a live parent-file tamper negative;
- request a Python producer deliberately and prove `E110` with zero
  interpreter launches;
- contain no Python or Rust guardian or semantic oracle.

Shell may transport, hash, and compare frozen records. It may not invent an
expected identity, admission, cost, or Sounio result.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Exact material receipt boundary and identity distinctions fixed here. |
| `Hypothesis` | Receipt-bound admission blocks the enumerated target, sibling, and engine laundering paths. |
| `Executable` | Pending the first post-Garden Sounio child. |
| `Claim-ready` | No. No new cost value, parity claim, or comparison exists. |

## What This Is Not

- It is not a new remote hardware capture.
- It is not evidence that `192.168.3.48` names a DGX machine.
- It is not Apple GPU or DGX CPU material identity.
- It is not whole-operation material coverage.
- It is not a benchmark, latency, throughput, energy, or cost result.
- It is not cross-target comparability or instruction equivalence.
- It is not a new guardian, parity implementation, or external confirmation.

## Next Material Bridge

After the child stream and semantics are frozen, Pireus may issue the first
cost request whose target/engine coordinate is now materially admitted. A
separate post-freeze runner may also try the `.48` identity request. Failure to
reach it must preserve the unresolved state rather than infer absence or copy
the `.24` identity.

## Semantic Lane Declaration

Concept-IDs: `SOUNIO-PIREUS-MATERIAL-ENGINE-ADMISSION`;
`SOUNIO-PIREUS-EXECUTION-ENGINE`;
`SOUNIO-PIREUS-TARGET-COST-OBSERVATION`;
`SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`

Intent-Preserved: material receipts admit only the exact machine and engine
they witnessed; unknown sibling locators remain unknown

Transformation: sealed Apple and DGX material parity receipts to a Sounio-owned
observed-machine and observed-engine overlay plus an unresolved `.48` request

Types-Changed: none; additive ontology vocabulary and record schema

Effects-Changed: ontology construction and query use `Mut` and `Epistemic`;
receipt hashing and executable output additionally use `IO`, `Alloc`, `Panic`,
and `Div`

IR-Changed: none

Claims-Introduced: exact Apple CPU and DGX `.24` GPU material engine admission,
only if the Sounio gate passes

Claims-Forbidden: Apple GPU admission; DGX CPU admission; `.48` machine or
engine identity; cost value; whole-operation coverage; cross-target
comparability; parity or claim promotion

Assumptions: frozen parent hashes identify the already sealed receipts and
evidence; no child remote execution is needed to consume them

Write-Set: Garden, first Sounio module/executable/test, concept contract,
semantics, receipt, evidence, registry, governance metadata, CI gate, and
review log

Read-Set: the eleven frozen parents, Loom language-authority policy, canonical
compiler resolver, and ontology/query modules

Positive-Witness: post-Garden Sounio stream with exact admitted coordinates and
an unresolved `.48` identity request

Negative-Witness: parent drift, identity transplant, sibling laundering,
authority promotion, cost promotion, and forbidden Python/Rust pre-launch
requests all fail closed

Acceptance-Gate: `scripts/ci/pireus_material_engine_admission.sh`

Integration-Target: Pireus execution-engine ontology and target-cost request
ledger

Authoritative-Only-If: the committed Garden precedes the Sounio executable,
all parent hashes and live checks pass, the child stream is deterministic, the
semantics is frozen, Loom authorizes each action, and `CLAIM_READY=false`

Open-Questions: identity and engine behind `.48`; whether and how to admit
Apple GPU and DGX CPU through separate material receipts; first target-local
cost request; later cross-target comparability proof
