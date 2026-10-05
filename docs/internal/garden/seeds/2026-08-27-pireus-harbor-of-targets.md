<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-harbor-of-targets
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-harbor-of-targets
-->

# Pireus: A Harbor of Targets

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder

## Butterfly

> "Todos sao Xeon. Apple e DGX sao alvos canonicos tambem."

Pireus is not a claim that one cluster already contains every architecture. It
is the harbor in which several material targets can be named without being
mistaken for one another or for evidence that has not yet arrived.

## Core Idea

The live Darwin compute fleet provides a controlled first experiment: every
observed worker is x86-64 and every observed processor is an Intel Xeon, while
the exact models, topology, and reported feature sets differ. Pireus can hold
the ISA family constant and make material variation queryable.

Apple Silicon and DGX are canonical Pireus targets too. A canonical target is
part of the intended observation and lowering universe; it is not automatically
an observed machine and does not inherit material facts from another target.

The ontology therefore needs three independent axes:

```text
canonical target != observed machine != material evidence
architecture family != processor profile != operation capability
declared support != executed witness != measured cost
```

## Observed Darwin Shore

The following identities and topologies were read from the five live Slurm
worker pods on 2026-08-27. They are Garden inputs for the first executable, not
yet frozen Pireus claims.

| Node | Processor string | Sockets | Cores/socket | Threads/core | Reported vector landmarks |
| --- | --- | ---: | ---: | ---: | --- |
| `r740-proxmox` | Intel Xeon Gold 6148 @ 2.40GHz | 2 | 20 | 2 | AVX2, AVX-512F/DQ/CD/BW/VL |
| `dl380-proxmox` | Intel Xeon Gold 6262V @ 1.90GHz | 2 | 24 | 2 | prior set plus AVX-512 VNNI |
| `5860-proxmox` | Intel Xeon W3-2423 | 1 | 6 | 2 | VBMI/VBMI2, BF16, FP16, AMX |
| `t560-proxmox` | Intel Xeon Gold 6526Y | 2 | 16 | 2 | VBMI/VBMI2, BF16, FP16, AMX |
| `r770-proxmox` | Intel Xeon 6730P | 2 | 32 | 2 | VBMI/VBMI2, BF16, FP16, AMX |

The table records kernel-reported presence only. It says nothing yet about
instruction availability under an OS save-state policy, encoding correctness,
latency, throughput, frequency behavior, compiler selection, or optimality.

## Canonical Targets

- Darwin Xeon fleet: live material profiles are presently observable.
- Apple Silicon: canonical AArch64/macOS target; current repository routing
  names `aarch64-macos`, but no fresh machine profile is asserted here.
- DGX Spark: canonical CUDA target; current repository routing names `sm_121`,
  but no fresh remote runtime observation is asserted here.

## Connections

- `docs/internal/garden/seeds/2026-08-27-pireus-material-ontology.md`
- `docs/internal/concepts/pireus-material-ontology.md`
- `stdlib/hardware/pireus/model.sio`
- `examples/pireus_vector_capability_query.sio`
- `scripts/apple/apple_native_v2_ssh_gate.sh`
- `scripts/dev/dgx_spark_public_gpu_gate.sh`

The Apple and DGX scripts are evidence that these targets are already named in
repository operations. Their current implementation is not an authority source
for Pireus v0.1 and is not invoked by its acceptance path.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Canonical target universe and live Darwin observations captured. |
| `Hypothesis` | Separating target, observation, and evidence will prevent capability promotion by implication. |
| `Executable` | Pending a new Sounio witness; Pireus v0 remains frozen and unchanged. |
| `Claim-ready` | No. No cost, lowering, vendor-corpus, Apple runtime, or DGX runtime claim exists. |

## What This Is Not

- It is not a claim that Darwin is heterogeneous in ISA.
- It is not a claim that Apple Silicon or DGX was reached in this session.
- It is not a benchmark or a processor-generation performance ordering.
- It is not proof that a reported CPU flag is usable by a generated program.
- It is not permission for C++, a vendor database, shell output, or an external
  LLM to define Sounio semantics or expected query results.
- It is not a second guardian; Loom remains the language-authority enforcement
  lane.

## Next Executable Bridge

Create a Sounio-only Pireus v0.1 witness that represents canonical targets,
observed machines, processor profiles, evidence states, operand roles, and
access modes. It must positively find the five observed Xeon profiles, retain
Apple and DGX as canonical but unobserved targets, reject promotion from
declared to observed, and distinguish a selector source from a second payload
data source.
