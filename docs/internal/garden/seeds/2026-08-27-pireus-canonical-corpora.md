<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-canonical-corpora
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-canonical-corpora
-->

# Pireus: Ships, Engines, And Charts

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder

## Butterfly

> "Se nos ingerissemos toda a ontologia de codigos dos processadores x86,
> AArch e Mac silicon, ja teriamos as ferramentas prontas."

The charts are the architecture corpora. The ships are machines. Their engines
are the actual execution units. Pireus is the harbor that keeps these objects
related without mistaking one for another.

## Core Idea

An ontology of instruction forms can make lowering a typed query instead of a
backend guess. But a system target is not one architecture:

```text
Target -> Machine -> ExecutionEngine -> ISA -> InstructionForm
                                      -> MaterialProfile -> Measurement
```

A Darwin node can combine a Xeon CPU with an NVIDIA GPU. Apple Silicon combines
Arm CPU cores with an Apple GPU. DGX Spark combines CPU and NVIDIA GPU engines.
The frozen Pireus v0.1 one-architecture target link is therefore a first
witness, not the final cardinality. It must remain frozen while a later Garden-
first artifact introduces `ExecutionEngine` explicitly.

## Four Evidence Layers

Pireus should ingest four layers without flattening them:

1. **Normative architecture**: instruction forms, encodings, operands, feature
   requirements, and specified behavior.
2. **Platform profile**: which engines and architectural features a particular
   machine reports as present and enabled.
3. **Material measurement**: latency, throughput, frequency behavior, and other
   costs measured under a named toolchain and workload.
4. **Compiler relation**: which Sounio operation may lower to which instruction
   form under a proven semantic and material precondition.

```text
normative form != reported availability != measured cost != selected lowering
```

## Canonical Corpus Map

### Intel x86-64

Primary corpus: [Intel XED](https://github.com/intelxed/xed), pinned for the
first inventory to release `v2026.08.23`, commit
`0bcb6237345c5066726dcc08b3d87928df3b5b26`. The repository and datafiles carry
Apache License 2.0 terms.

Relevant structured surfaces include:

- `datafiles/files.cfg`, the manifest connecting decoder, encoder, chip,
  register, width, map, CPUID, and instruction inputs;
- `xed-isa.txt` plus extension-specific `*.xed.txt` instruction records;
- `xed-fields.txt`, `xed-regs.txt`, `xed-operand-types.txt`, and
  `xed-operand-width.txt`;
- `cpuid.xed.txt`, `xed-chips.txt`, and opcode-map descriptions.

XED's internal record syntax is structured but not promised as a stable public
schema. Every import must therefore pin release, commit, file hash, parser
version, and accepted grammar. XED's Python JSON exporter is not an admissible
authority path under the founder contract. The first importer must parse the
pinned source records in Sounio; a later C++ XED API consumer may provide only
`MATERIAL_PARITY` after the Sounio result is frozen.

The first inspected slice already shows why operand roles are load-bearing:

- `VPERMPD` has an `IMM8` selector form and a vector-index selector form;
- its destination is syntactically write-only in the inspected 512-bit forms;
- `VPERMI2PD` and `VPERMT2PD` expose read-write destination forms and multiple
  read operands;
- syntactic operand position alone does not decide whether an operand is a
  selector, payload source, merge source, or mask.

No semantic role assignment is frozen by this Garden inventory.

### Arm AArch64

Primary normative corpus: the Arm A-profile A64 Instruction Set Architecture,
currently published as document `DDI0602` with versioned HTML and downloadable
XML instruction descriptions. Arm states that the XML/HTML instruction
descriptions and the Arm Architecture Reference Manual originate from the same
source.

Architecture-feature relationships belong to AARCHMRS, the Arm Architecture
Machine Readable Specification. Its Features package contains `Features.json`,
a JSON schema, guidance, feature values, and Boolean constraints. The official
Arm introduction documents package `AARCHMRS_A_profile-2024-12.tar.gz`; the
exact current archive and its terms must be acquired and pinned before import.

The first comparative permutation family to inspect is A64 `TBL`/`TBX`, but no
equivalence with an x86 form or a Cayley-Dickson lowering is asserted here.

Arm archive terms are not assumed to equal Apache-2.0. Pireus must retain the
download's README and terms, and must not redistribute the corpus until those
terms are reviewed.

### Apple Silicon

Apple Silicon is not a replacement ISA corpus for AArch64. Its CPU normative
instruction layer comes from Arm; Apple supplies platform and material-profile
evidence.

Official Apple guidance identifies these observation surfaces:

- `sysctl hw` and `sysctlbyname("hw.optional.arm.FEAT_*", ...)` for reported
  CPU features;
- performance-level CPU counts, cache sizes, cache-line size, and VM page size;
- the Apple Silicon CPU Optimization Guide for material guidance;
- Metal feature-set tables and runtime feature queries for the Apple GPU.

Absence of a `sysctl` key must remain different from a proven unsupported Arm
feature. Apple CPU and Apple GPU must become separate `ExecutionEngine`
individuals under the same machine.

### NVIDIA And DGX

Primary normative virtual-ISA corpus: NVIDIA's current PTX ISA documentation.
The live documentation reports PTX ISA 9.3; the repository's DGX route is
presently configured for CUDA 13.0 and `sm_121`. A future run must pin the
remote `ptxas` version, PTX version, target architecture, and generated binary
hash rather than merge those values into one label.

PTX is a virtual ISA translated to a target GPU instruction set. Therefore:

```text
PTX instruction != SASS instruction != DGX machine capability
```

No official NVIDIA machine-readable instruction database was identified in
this inventory. The official HTML/PDF is normative for PTX; `ptxas` and
`nvdisasm` outputs can later provide material observations for a pinned GPU.
Third-party reverse-engineered SASS databases may be research references, but
cannot become semantic authority or silently fill undocumented vendor facts.

DGX must also be represented as a multi-engine system, not as a synonym for
`CUDA_SM`.

## Ingestion Contract

Each corpus enters through its own mandatory sequence:

```text
Garden source/version/license inventory
-> Sounio parser and expected result
-> frozen source, corpus, grammar, semantics, and output hashes
-> Loom acceptance
-> parity consumers
-> material measurements
-> bounded lowering claim
```

Vendor data is evidence input, never an expected-result producer. The Sounio
executable decides how records become Pireus concepts and must reject records
outside its frozen grammar instead of guessing or silently dropping fields.

## Repository Shape

Pireus remains in the Sounio monorepo while the ontology and importer contract
are changing together. Raw vendor snapshots should stay out of the main tree
until licensing and size are understood. The first receipts can record pinned
upstream URLs, commits, versions, and hashes.

An eventual `sounio-lang/pireus` repository becomes appropriate when corpus
snapshots need independent release cadence, license notices, and large-data
history. It must consume the already-frozen Sounio semantic contract rather
than establish a second implementation.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Multi-engine model and canonical corpus map captured. |
| `Hypothesis` | A typed relation among form, engine, profile, measurement, and lowering can replace backend guessing. |
| `Executable` | Pireus v0/v0.1 only; no vendor importer or multi-engine model executed. |
| `Claim-ready` | No. Corpus completeness, semantic equivalence, costs, and lowering optimality remain open. |

## What This Is Not

- It is not permission to ingest all corpora in one unreviewable step.
- It is not a claim that Darwin CPUs are multi-ISA; all five observed CPUs are
  Xeon/x86-64.
- It is not a claim that a machine report, vendor database, or decoder library
  defines Sounio semantics.
- It is not a claim that XED, Arm XML, Apple `sysctl`, and PTX carry equivalent
  kinds of evidence.
- It is not SASS documentation and does not elevate reverse engineering to
  vendor fact.
- It is not another guardian. Loom owns authority enforcement.

## Next Executable Bridge

After Loom accepts the frozen Pireus v0.1 receipt, implement a grammar-bounded
Sounio importer for only the pinned XED records needed by `VPERMPD`,
`VPERMI2PD`, and `VPERMT2PD`. The witness must preserve raw operand access,
width, mask, selector representation, ISA set, and source provenance before it
assigns any Pireus semantic operand role. Unknown fields and grammar drift must
fail closed.
