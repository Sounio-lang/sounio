<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-darwin-multi-engine
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-darwin-multi-engine
-->

# Pireus: One Machine, Several Engines

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder

## Butterfly

> "O cluster Darwin reune varias arquiteturas. Todos os CPUs sao Xeon. Apple e
> DGX sao alvos canonicos tambem."

The apparent tension is the model. Every observed Darwin CPU is Xeon/x86-64,
yet a Darwin machine may also carry one or more NVIDIA execution engines with a
different machine ISA. Architecture belongs to an engine, not directly to the
machine or cluster.

```text
Cluster -> Machine -> ExecutionEngine -> ISA
                                      -> ExecutionInterface
                                      -> MaterialProfile
```

## Fresh Material Observation

At `2026-08-27`, the Slurm worker pods exposed these exact `nvidia-smi` rows:

```text
r740-proxmox
0, NVIDIA RTX A5000, GPU-5618b771-c8f2-16a6-b94e-0ba3537c47db, 8.6, 595.71.05
1, Quadro RTX 8000, GPU-1a84782e-5b22-def2-2575-0abf3f971016, 7.5, 595.71.05

5860-proxmox
0, NVIDIA RTX 4000 Ada Generation, GPU-3ea9adc0-d67b-54e3-78dc-65dc80bbf70c, 8.9, 595.71.05

r770-proxmox
0, NVIDIA L4, GPU-b3ffdfa8-b165-5562-f443-f40daaeef893, 8.9, 595.71.05
```

These are kernel/driver observations, not Sounio expected results. The five CPU
profiles already frozen in Pireus v0.1 remain:

```text
r740   Intel Xeon Gold 6148
dl380  Intel Xeon Gold 6262V
5860   Intel Xeon W3-2423
t560   Intel Xeon Gold 6526Y
r770   Intel Xeon 6730P
```

The resulting observed topology hypothesis is:

| Machine | CPU engines | GPU engines | Material engine ISAs |
| --- | ---: | ---: | --- |
| r740 | 1 | 2 | x86-64, NVIDIA SM 8.6, NVIDIA SM 7.5 |
| dl380 | 1 | 0 | x86-64 |
| 5860 | 1 | 1 | x86-64, NVIDIA SM 8.9 |
| t560 | 1 | 0 | x86-64 |
| r770 | 1 | 1 | x86-64, NVIDIA SM 8.9 |

This makes four observed architecture identities across the Darwin cluster:
x86-64, `sm_75`, `sm_86`, and `sm_89`. It does not make any Darwin CPU
multi-ISA.

## Canonical Target Blueprints

Targets should declare engine blueprints without pretending those blueprints
are live observations:

```text
Darwin Xeon:     x86-64 CPU; optional NVIDIA GPU through CUDA
Apple Silicon:   AArch64 CPU; Apple GPU through Metal
DGX Spark:       AArch64 CPU; NVIDIA GPU through CUDA, repository route sm_121
```

Metal and CUDA are execution interfaces, not ISAs. An Apple GPU ISA is not
named here. A declared `sm_121` route is not a fresh DGX machine observation.

## Proposed Distinctions

The first executable model should keep these concepts separate:

- `Cluster` versus `Machine`;
- `Machine` versus `ExecutionEngine`;
- CPU versus GPU engine kind;
- ISA versus execution interface;
- target blueprint versus observed engine;
- driver-reported compute capability versus executed instruction witness;
- engine identity versus material profile;
- one machine with several engines versus one engine with several ISAs.

## Questions For Sounio

The Sounio witness should be able to ask:

1. How many CPU and GPU engines are observed in Darwin?
2. Which machines have both kinds, and how many engines does r740 carry?
3. Which distinct material ISAs occur in the cluster?
4. Which compute-capability and driver profiles were observed?
5. Which two engine blueprints belong to each canonical target?
6. Can Apple and DGX remain canonical while returning zero observed machines
   and zero observed engines?
7. Can a negative query prove that Metal was not mislabeled as an ISA?

## Failure Boundary

The executable must fail if it collapses machine architecture into one scalar,
attaches a GPU ISA to a Xeon CPU engine, promotes a blueprint to an observation,
or treats CUDA/Metal as an ISA.

No latency, throughput, availability, lowering, or Cayley-Dickson performance
claim is authorized by this Garden record.

## Mandatory Order

```text
this Garden seed
-> Sounio ExecutionEngine ontology and witness
-> frozen semantics and result hashes
-> Loom acceptance
-> parity consumers
-> material execution measurements
```
