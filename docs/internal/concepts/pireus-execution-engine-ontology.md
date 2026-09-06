<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-execution-engine-ontology
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-execution-engine-ontology
-->

# Pireus Execution Engine Ontology

Proposed Concept-ID: `SOUNIO-PIREUS-EXECUTION-ENGINE`

Status: `SEMANTICS_FROZEN_PENDING_REGISTRY_AND_LOOM_ACCEPTANCE`

Semantic-Lane-ID: `pireus-execution-engine-20260827`

## Intent

Represent machines as owners of execution engines so Pireus can describe a
heterogeneous cluster without attaching one scalar architecture to a machine
or pretending a Xeon CPU implements a GPU ISA.

```text
Cluster -> Machine -> ExecutionEngine -> ISA
                                      -> ExecutionInterface
                                      -> DriverProfile
```

## Authority

The semantic authority is the Sounio pair:

```text
stdlib/hardware/pireus/execution_engine.sio
examples/pireus_execution_engine_query.sio
```

The live CPU and GPU rows are material inputs. Sounio defines their ontology
projection, queries, expected counts, distinctions, and negative controls.

## Core Distinctions

- A `Cluster` contains `Machine` individuals.
- A machine owns zero or more `ExecutionEngine` individuals.
- CPU and GPU are `EngineKind` values.
- x86-64, AArch64, and NVIDIA `sm_*` values are `EngineISA` values.
- CUDA and Metal are `ExecutionInterface` values, not ISAs.
- A live engine is an `ObservedExecutionEngine`.
- A target can declare an `EngineBlueprint` without creating an observed
  machine or engine.
- Kernel observation and driver observation retain distinct evidence roles.

## Frozen Darwin Projection

All five observed CPU engines are Xeon/x86-64. Four NVIDIA GPU engines add
three material GPU ISAs:

| Machine | CPU engine | GPU engines |
| --- | --- | --- |
| r740 | Xeon/x86-64 | RTX A5000/sm_86; Quadro RTX 8000/sm_75 |
| dl380 | Xeon/x86-64 | none observed |
| 5860 | Xeon/x86-64 | RTX 4000 Ada/sm_89 |
| t560 | Xeon/x86-64 | none observed |
| r770 | Xeon/x86-64 | L4/sm_89 |

The observed cluster therefore has nine engines and four distinct ISA
identities: x86-64, `sm_75`, `sm_86`, and `sm_89`.

All four GPU observations bind driver profile `595.71.05` and interface CUDA.
This records driver output; it is not an instruction-execution witness.

## Canonical Blueprints

Each canonical target declares exactly two engine blueprints:

```text
Darwin Xeon:   CPU/x86-64; GPU/CUDA
Apple Silicon: CPU/AArch64; GPU/Metal
DGX Spark:     CPU/AArch64; GPU/sm_121/CUDA
```

The Darwin GPU blueprint does not require every Darwin machine to contain a
GPU. The Apple GPU blueprint names Metal as an interface and makes no Apple GPU
ISA claim. The DGX `sm_121` value is a declared repository route, not a live
machine observation.

## Required Negatives

The executable must preserve:

- zero Apple observed engines;
- zero DGX observed engines;
- zero observed `sm_121` engines;
- zero engines with Metal stored as an ISA;
- zero GPU engines with x86-64 stored as their ISA.

## Evidence Boundary

This concept establishes topology, identity, evidence role, and queryable
separation. It does not establish device availability at a later time,
instruction support, scheduling, data movement, coherence, cost, lowering, or
performance.

No Python, Rust, Node, Ruby, `awk`, or `bc` participates in the semantic or
expected-result path. `PARITY_OPEN` remains closed until registry and Loom
acceptance.
