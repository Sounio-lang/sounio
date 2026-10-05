<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-execution-engine-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-execution-engine-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Execution Engine Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Recorded-At-UTC: `2026-08-27`

Completed-Stage: `SEMANTICS_FROZEN`

Next-Stage: `PARITY_OPEN`

Next-Stage-Status: `BLOCKED_PENDING_CONCEPT_REGISTRATION_AND_LOOM_ACCEPTANCE`

## Authority Binding

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

Semantic-Lane-ID: `pireus-execution-engine-20260827`

Garden-Seed:
`docs/internal/garden/seeds/2026-08-27-pireus-darwin-multi-engine.md`

Garden-Seed-SHA256:
`ce0efb78d40cb5867121375bef7cd91f08ac0a39326e2bd2ee37db98382b49a6`

Garden-Commit:
`b5283350b7dee4903a8ceb95aa7a4e3b5568dffc`

Concept-Contract:
`docs/internal/concepts/pireus-execution-engine-ontology.md`

Concept-Contract-SHA256:
`53f5847d1b72d1b3e2a21b67b7f9024d071ab90aaed6e515f658e4ee27fa4aaa`

Ontology-Source:
`stdlib/hardware/pireus/execution_engine.sio`

Ontology-Source-SHA256:
`8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e`

Executable-Source:
`examples/pireus_execution_engine_query.sio`

Executable-Source-SHA256:
`fe5d0e92b43929661a88250c108bf5e507a46c0687bdb896335c613d33121fd8`

Frozen-Semantics:
`docs/research/pireus_execution_engine_semantics.md`

Frozen-Semantics-SHA256:
`c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233`

Canonical-Output-SHA256:
`740b1e7a5854373690a4c5e720aea9a11019af618290b6412ab5fad0fc81d808`

## Material Inputs

The CPU identities and topology are inherited from the frozen Pireus v0.1
receipt. Fresh `nvidia-smi` reads produced:

```text
r740:  NVIDIA RTX A5000, sm_86, driver 595.71.05
r740:  Quadro RTX 8000, sm_75, driver 595.71.05
5860:  NVIDIA RTX 4000 Ada Generation, sm_89, driver 595.71.05
r770:  NVIDIA L4, sm_89, driver 595.71.05
```

The Garden seed retains the exact UUID-bearing rows. These are vendor-driver
observations, not expected-result producers.

## Toolchain

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
execution_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
ontology_query_kernel=stdlib/ontology/query.sio
ontology_query_kernel_sha256=e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb
```

## Execution Hardware

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

## Commands

```bash
/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_execution_engine_query.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_execution_engine_query.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_execution_engine_query.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_execution_engine_query.sio
```

The two complete Sounio streams were byte-identical.

## Sounio-Produced Result

```text
SOUNIO_AUTHORITY schema=pireus-execution-engine.v0 role=SEMANTIC_AUTHORITY
PIREUS_ENGINE_STORE triples=290

PIREUS_DARWIN machines=5
 engines=9
 cpu=5
 gpu=4
 multi_engine_machines=3

PIREUS_MACHINE_ENGINES r740=3
 dl380=1
 n5860=2
 t560=1
 r770=2

PIREUS_ENGINE_ISA x86_64=5
 sm75=1
 sm86=1
 sm89=2
 sm121_observed=0
 distinct_observed=4

PIREUS_ENGINE_INTERFACE cuda_observed=4
 metal_observed=0
 driver_595_71_05=4

PIREUS_BLUEPRINTS darwin=2
 apple=2
 dgx=2
 apple_metal=1
 dgx_cuda=1

PIREUS_TARGET_ENGINES darwin_observed=9
 apple_observed=0
 dgx_observed=0

PIREUS_ENGINE_NEGATIVE metal_as_isa=0
 gpu_x86=0

PIREUS_ENGINE_SUMMARY failures=0
```

Line breaks around integers are emitted by the selected Sounio runtime's
`print_int`; no formatter rewrote the hashed stream.

## Validation Classification

| Check | Result | Classification |
| --- | --- | --- |
| rebuilt ontology check | unanimous, `rebuilt_direct` | current-source checker accepted |
| `lean_single` check | exit 0 | Sounio compiler path accepted |
| `lean_single` run, twice | exit 0, identical | semantic-authority result |
| default Madaros run | rejected during imported visibility preflight | compiler/runtime divergence |
| Metal through ISA predicate | zero | negative witness |
| GPU with x86-64 ISA | zero | negative witness |
| Apple/DGX observed engines | zero | declaration/observation boundary |

The default run failure produced no replacement result and did not weaken the
explicit compiler-path receipt.

## Prohibited-Oracles Gate

No Python, Rust, Node, Ruby, `awk`, or `bc` was used to construct the ontology,
expected counts, queries, or frozen stream. Shell and `kubectl` transported
material observations and commands; Sounio produced the semantic result.

No parity language or external LLM review was invoked. External LLM offload
reviews invoked: none; this is an internal ontology and material-profile
artifact without mathematical, clinical, or external-facing claims.

## Evidence Boundary

The receipt establishes the named topology and distinctions only. It does not
establish current device availability beyond the recorded read, instruction
support, transfer cost, scheduling, coherence, lowering, or performance.

`PARITY_OPEN` requires proposed Concept-ID
`SOUNIO-PIREUS-EXECUTION-ENGINE` to be registered and this frozen Sounio receipt
to be accepted by the active Loom owner.

## Loom Admission (Append-Only)

Recorded UTC: `2026-08-27T05:28:16Z`

The frozen receipt was submitted to Loom frame `9020` with:

```text
stage=2 SOUNIO_EXECUTABLE
action=3 FREEZE_SEMANTICS
language=1 Sounio
role=1 SEMANTIC_AUTHORITY
policy_state=1 available
semantic_write=1
expected_result_write=1
parity_receipt_valid=0
review_promoted=0
exception_and_waiver_fields=0
guardian_fields=0
parent_semantics_sha256=absent
waiver_sha256=absent
```

The complete 82-field frame bound:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `fe5d0e92b43929661a88250c108bf5e507a46c0687bdb896335c613d33121fd8` |
| Frozen semantics | `c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `9279bca3593ec138933589505e7a944e85400638daf11d3b02905ebec80baecb` |
| Sounio result | `740b1e7a5854373690a4c5e720aea9a11019af618290b6412ab5fad0fc81d808` |

The toolchain record was this exact UTF-8 text with one final `LF`:

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
```

The command record was this exact UTF-8 text with one final `LF`:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_execution_engine_query.sio
```

The hardware record is identical to the seven-line record printed above. The
operational runtime remained the fixture-matched realization of the frozen
Sounio Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

A deliberate Python-oracle frame with the same receipt bindings was denied
before an interpreter or requested effect ran:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

Both decision hashes include the final `LF`. This admission accepts
`SEMANTICS_FROZEN`; it does not register the Concept-ID, open parity, or promote
a claim.
