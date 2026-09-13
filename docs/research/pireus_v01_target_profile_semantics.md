<!-- docs:meta
topic_id: repo.docs.research.pireus-v01-target-profile-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-v01-target-profile-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus v0.1 Target And Material Profile Semantics

Semantic-Lane-ID: `pireus-v01-target-profile-20260827`

Owner: `founder`

Concept-IDs: proposed `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`,
`SOUNIO-SCIENCE-RESEARCH-BOUNDARY`, `SOUNIO-NONASSOCIATIVE-ORDER`

Stage: `SEMANTICS_FROZEN`

## Authority

`examples/pireus_target_profile_query.sio` is the sole semantic producer and
expected-result producer for Pireus v0.1. It extends the frozen Pireus v0 store
through `stdlib/hardware/pireus/target_profile.sio` and continues to use
Sounio's existing `stdlib/ontology/query.sio` engine.

Live shell commands supplied material observations only. They did not compute
query expectations, classify capability forms, or decide semantic results.
Python, Rust, disposable-language oracles, parity languages, vendor databases,
and external LLMs did not participate in the acceptance execution.

## Target Semantics

Pireus v0.1 contains exactly three canonical targets:

```text
DarwinXeon    architecture=x86_64
AppleSilicon architecture=AArch64
DGXSpark     architecture=CUDA_SM
```

Every target carries `DeclaredTarget` evidence. This means it belongs to the
founder-authorized Pireus target universe. It does not mean a machine was
reached, executed, or measured.

## Material Observation Semantics

Five `Machine` individuals are linked to `DarwinXeon`, one for each live Slurm
worker. Each links to an x86-64 `ProcessorProfile` and carries
`ObservedKernel` evidence:

| Machine | Profile | Logical CPUs | Family/model/stepping |
| --- | --- | ---: | --- |
| `r740` | Xeon Gold 6148 | 80 | 6/85/4 |
| `dl380` | Xeon Gold 6262V | 96 | 6/85/7 |
| `5860` | Xeon W3-2423 | 12 | 6/143/8 |
| `t560` | Xeon Gold 6526Y | 64 | 6/207/2 |
| `r770` | Xeon 6730P | 128 | 6/173/1 |

Profiles also record socket, core, and thread topology. `ReportedFeature`
triples retain selected flags read from `/proc/cpuinfo`. Their evidence is
presence in the kernel report, not successful instruction execution.

The frozen Sounio query counts are:

```text
AVX512F       5
AVX512_VNNI  4
AVX512_VBMI2 3
AMX_TILE      3
```

No Apple Silicon or DGX machine individual exists in v0.1. Their observed
machine counts are therefore zero while their canonical-target membership is
one. This is deliberate negative evidence against promotion by declaration.

## Operand Role Semantics

Pireus v0 used `source_count` to separate two synthetic permutation forms. v0.1
does not reinterpret or mutate that frozen vocabulary. It adds precise fields:

```text
data source count
selector source count
mask source count
destination access
operand roles
```

The positive form has one payload data source, one selector, zero mask sources,
and a write-only destination. The control form has two payload data sources,
one selector, zero mask sources, and a read-write destination. Both remain
synthetic Sounio witnesses and denote no vendor instruction.

The positive query must return exactly the first form and reject the control.
A query with zero selector sources must return no form.

## Frozen Result

The Sounio-produced store contains 164 triples and produces:

```text
canonical targets = 3
observed Darwin Xeon machines = 5
observed Apple Silicon machines = 0
observed DGX Spark machines = 0
reported feature counts = 5/4/3/3
one-data plus selector form matches = 1
two-data plus selector control matches = 1
zero-selector negative matches = 0
failures = 0
```

## Evidence Boundary

This establishes a Sounio-native distinction among intended targets, observed
machines, processor profiles, reported features, and operand roles. It does not
establish instruction encodings, execution availability, measured cost, or an
optimal Cayley-Dickson lowering.

Existing Apple and DGX operational scripts name canonical targets but currently
contain Python receipt helpers. They are excluded from the v0.1 acceptance path
and cannot satisfy the founder's no-Python authority contract without later
migration. Pireus does not create a competing guardian; Loom remains the
canonical language-authority enforcement lane.

## Transition Boundary

The v0.1 stage is `SEMANTICS_FROZEN`. `PARITY_OPEN` remains closed until the
proposed Concept-ID is registered and the active Loom owner accepts the frozen
receipt. Vendor instruction ingestion is a later semantic artifact and may not
retroactively change these expected results.
