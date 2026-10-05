<!-- docs:meta
topic_id: repo.docs.architecture.compiler-maturity-blueprint
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.compiler-maturity-blueprint
-->

# Compiler Maturity Program Blueprint

## Summary

This note names the current architectural shift without treating it as a public
release line.

`Compiler Maturity Program` is the internal label for the work required to move
Sounio from a small, blob-oriented self-hosted path toward a bundle-first,
evidence-producing compiler that can support larger scientific and epistemic
surfaces.

This is not release branding, not semantic-version signaling, and not a claim
that the large-surface rebuilt path is already stable.

## Why The Repo Has Outgrown The Small-Compiler Model

The current self-hosted work is no longer about isolated import lookup or a
single source buffer.

Recent M4 through M9 work established a different reality:

- module closure can now be made explicit and inspectable
- bundle-resolved import authority can be restored for the validated probe set
- larger surfaces fail for attributable internal-capacity reasons rather than
  opaque parse noise
- wrapper provenance remains operationally stronger than direct-driver semantic
  authority on ontology-sized surfaces

That combination means the next compiler problems are about module execution,
capacity truth, and evidence-bearing failure classification, not only parser or
ontology features in isolation.

## Current State

### M8

M8 restored bundle-resolved import authority on the validated probe set:

- `m4_namespaced_direct_probe` returned `111`
- `m4_namespaced_transitive_probe` returned `44`
- `m4_flat_direct_probe` returned `305`
- `m4_flat_transitive_probe` returned `316`

The important architectural outcome is that execution-time path guessing is no
longer the main blocker for those probes. Resolution can be treated as a
bundle-owned decision for that validated subset.

### M9

M9 established the current large-surface frontier on the generated self-host.

For `self-hosted/ci/m4_large_surface_probe.sio`:

- at `2 MiB`, closure truncates with `rc=2`, `module_count=46`,
  `total_requested_bytes=2810871`, and `first_truncated_index=28`
- at `4 MiB`, `8 MiB`, and `16 MiB`, byte truncation is gone, but execution
  still fails with `rc=139`, `ND_COUNT=262143`, `ovf_nd=1`, and `ovf_pool=1`

For `self-hosted/ci/ontology_witness_program_probe.sio`:

- at `2 MiB`, closure truncates with `rc=2`, `module_count=48`,
  `total_requested_bytes=2834721`, and `first_truncated_index=30`
- at `4 MiB`, `8 MiB`, and `16 MiB`, byte truncation is gone, but execution
  still fails with `rc=139`, `ND_COUNT=262143`, `ovf_nd=1`, and `ovf_pool=1`

The first attributed large-surface parse failures currently land in
`self-hosted/compiler/module_loader.sio`.

### Authority Reading

- wrapper provenance is still stronger than direct-driver semantic authority
- direct-driver semantic closure is not reopened on ontology-sized surfaces
- `boot4.elf` has not been promoted here as a new stable baseline

## Core Architecture Lanes

### Evidence Core

Evidence Core owns artifacts that make compiler truth inspectable and
versionable:

- `ClosureBundle`
- `ResolutionTable`
- `ExecutionTruth`
- `TruthWitness`
- `VerdictProvenance`

The practical goal is not only to emit a binary, but also to say what program
was seen, what path was taken, and what truth is actually authorized.

### Module Core

Module Core owns resolved closure construction and truthful bundle execution.

Its working rules are:

- resolve imports once during closure construction
- keep module identity explicit across load, lex, parse, and later phases
- let execution consume resolved bundle entries instead of recomputing paths

### Semantic Core

Semantic Core owns typed meaning after module and execution truth exist.

It includes:

- ontology inference and kernel behavior
- boundary-engine policy
- validation-layer policy
- scalar summary surfaces used by rebuilt-driver code

This lane remains constrained by the current truth frontier documented in
`truth-frontier.md` and `semantic-contracts.md`.

### Scientific Core

Scientific Core treats scientific computing as compiler architecture, not only
as stdlib inventory.

It includes:

- units and scientific effects
- numerical kernels
- GPU and SIMD lowering surfaces
- hypercomplex algebra in the compiler

### Epistemic Model Substrate

Epistemic Model Substrate is the internal name for evidence-bearing data that a
future epistemic tooling layer could consume.

It is not a claim about a shipped model. It is the recognition that future
tooling should learn from:

- closure structure
- provenance-bearing verdicts
- truth-layer distinctions
- explicit `OK`, `REJECT`, `UNKNOWN`, and `INTERNAL_ERROR` outcomes

## Program Sequence

The current order of work is:

1. `M9` large-surface bundle execution
2. `T6` direct-driver truth restoration
3. `S3` semantic closure
4. `H1` hypercomplex algebra in the compiler

This sequence intentionally keeps semantic expansion behind module and
execution truth.

## Non-Goals

This note does not claim any of the following:

- semantic closure is complete
- ontology-sized direct-driver truth is stable
- large-surface self-host execution is already healthy above `4 MiB`
- `boot4.elf` has been promoted as a trusted new baseline
- `Compiler Maturity Program` is a public release or semver line

## Related Docs

- [truth-frontier.md](./truth-frontier.md)
- [truth-layers.md](./truth-layers.md)
- [module-closure-truth.md](./module-closure-truth.md)
- [scientific-core.md](./scientific-core.md)
- [semantic-contracts.md](./semantic-contracts.md)
- [decisions/README.md](../decisions/README.md)
