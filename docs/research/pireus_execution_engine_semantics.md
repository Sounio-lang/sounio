<!-- docs:meta
topic_id: repo.docs.research.pireus-execution-engine-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-execution-engine-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Execution Engine Semantics v0

Date: `2026-08-27`

Stage: `SEMANTICS_FROZEN`

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

## Mandatory Order

The Garden seed was committed as `b5283350b7dee4903a8ceb95aa7a4e3b5568dffc`
before the Sounio model or witness existed.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

Parity and claim promotion are closed.

## Semantic Model

The frozen v0.1 target/profile store is extended, not modified. The extension
adds classes for cluster, execution engine, observed execution engine, engine
kind, engine ISA, execution interface, driver profile, engine blueprint, and
multi-engine machine.

An observed CPU engine has:

```text
MachineHasEngine
type ExecutionEngine
type ObservedExecutionEngine
EngineKind CPU
EngineISA x86-64
evidence ObservedKernel
```

An observed NVIDIA GPU engine additionally carries CUDA interface and driver
profile, with `ObservedDriver` evidence. A blueprint uses
`DeclaredBlueprint` evidence and is never typed as an observed engine.

## Frozen Result

The extension adds 126 triples to the frozen v0.1 store, producing 290 total.

| Query | Result |
| --- | ---: |
| Darwin machines | 5 |
| observed CPU engines | 5 |
| observed GPU engines | 4 |
| multi-engine machines | 3 |
| r740 engines | 3 |
| dl380 engines | 1 |
| 5860 engines | 2 |
| t560 engines | 1 |
| r770 engines | 2 |
| x86-64 engines | 5 |
| `sm_75` engines | 1 |
| `sm_86` engines | 1 |
| `sm_89` engines | 2 |
| distinct observed ISA identities | 4 |
| CUDA engines | 4 |
| driver `595.71.05` engines | 4 |
| blueprints per canonical target | 2 |
| Apple Metal blueprint | 1 |
| DGX CUDA blueprint | 1 |

Apple and DGX return zero observed engines. `sm_121` returns zero observed
engines. Metal queried through the ISA predicate returns zero, and GPU plus
x86-64 returns zero.

## Interpretation

Darwin is heterogeneous at the cluster and machine-engine levels. Its five CPU
engines remain uniformly Xeon/x86-64. The additional material ISAs belong to
four separate GPU engines. No engine is inferred to implement another engine's
ISA.

CUDA and Metal are execution interfaces. This model deliberately refuses the
common shortcut of placing them in an ISA field.

## Validation Boundary

The rebuilt/current-source ontology wrapper check was unanimous with
`provenance=rebuilt_direct`. The explicit `lean_single` Sounio path typechecked
and executed the witness twice with byte-identical output.

The current default Madaros `run` path later rejected imported ontology
internals during its visibility preflight. That execution-path divergence
created no alternative result and is retained as compiler evidence rather than
hidden as a semantic failure.

## Non-Claims

The semantics do not establish live Apple or DGX access, Apple GPU ISA,
instruction execution, CUDA/Metal equivalence, data-transfer cost, scheduling,
latency, throughput, lowering correctness, or Cayley-Dickson acceleration.
