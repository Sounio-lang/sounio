<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-material-ontology
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-material-ontology
-->

# Pireus: The Material Port Of Sounio

> **Status**: Garden seed with first executable bridge | **Last validated**: 2026-08-27 | **Source**: founder conversation during Cayley-Dickson XOR lowering

## Butterfly

> se nos ingerissemos toda a ontologia de codigos dos processadores x86,
> AArch e Mac silicon, tornaria nossa vida infinitamente mais facil, pois ja
> teriamos as ferramentas prontas

The name is **Pireus**, after the port of Peiraias in Athens.

## Core Idea

Pireus is the material ontology of Sounio: a common semantic port where
processor architectures arrive without losing their differences. It should
allow Sounio to ask which instruction capabilities can realize a frozen
semantic DAG, under explicit lane, source, masking, precision, feature, ABI,
and evidence constraints.

Pireus does not need a second ontology engine. Sounio already has classes,
properties, individuals, triple storage, query, closure, reasoning, caching,
mapping, and federation under `stdlib/ontology`. Pireus is a hardware-domain
ontology built on that substrate.

## Foundational Distinctions

```text
instruction semantics != microarchitecture cost
architecture          != platform ABI
declared capability   != measured capability
vendor statement      != Sounio-produced expected result
legal realization     != optimal schedule
unknown               != unsupported
```

## Evidence Labels

| Layer | Status |
| --- | --- |
| `Garden` | Named by the founder; ontology substrate identified in the repository. |
| `Hypothesis` | A material ontology can turn frozen semantic DAGs into constrained capability queries shared across architectures. |
| `Executable` | `examples/pireus_vector_capability_query.sio` executes with positive and negative synthetic capability witnesses. |
| `Claim-ready` | No. No vendor corpus, real opcode, encoding, schedule, or cost has been accepted. |

## Authority Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Raw vendor material may later enter as content-addressed evidence. Sounio must
produce the first executable representation and canonical expected results.
LLVM, C++, external LLMs, and processor measurements may compare or review only
after a frozen Sounio artifact opens parity.

## What This Is Not

- Not a new ontology framework parallel to `stdlib/ontology`.
- Not a claim that the complete x86 or AArch64 instruction sets are ingested.
- Not a claim that Apple publishes every microarchitectural detail.
- Not a backend rewrite.
- Not an instruction-cost table.
- Not permission to promote LLVM metadata or an external parser into semantic
  authority.

## First Executable Bridge

Use `ontology::query` to express two synthetic permutation capabilities and
query the frozen Cayley-Dickson routing requirement:

```text
operation     = single-source lane permutation
element bits  = 64
lanes         = 8
sources       = 1
evidence role = synthetic Sounio witness
```

The positive witness must find exactly the matching synthetic capability. The
negative witness must reject a two-source capability. This validates Pireus's
ontological language and query path without asserting a real processor fact.

## Connection

- `examples/cayley_dickson_xor_simd_dag.sio` supplies the first frozen client
  requirement: width-eight routing is single-source per destination chunk and
  has eight within-chunk lane patterns.
- `stdlib/ontology/query.sio` supplies the first executable storage and query
  substrate.
- Proposed Concept-ID: `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`.
