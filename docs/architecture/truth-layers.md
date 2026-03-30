<!-- docs:meta
topic_id: repo.docs.architecture.truth-layers
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.truth-layers
-->

# Truth Layers

## Summary

This note defines the current internal truth taxonomy for rebuilt/self-hosted
work. It exists to prevent one layer of success from being mistaken for another.

In the current compiler maturity work, the important distinction is not only
whether a build returned `0`, but which layer of truth was actually established.

## Outcome Vocabulary

The following outcomes are intentionally distinct:

- `OK`
- `REJECT`
- `UNKNOWN`
- `INTERNAL_ERROR`

`UNKNOWN` is not a soft form of `OK`, and `INTERNAL_ERROR` is not a semantic
verdict.

## Layer Definitions

### Closure Truth

Closure Truth answers:

- which modules were requested
- which modules were resolved
- what order they entered the closure
- whether unresolved symbols remain

Closure Truth is green when the compiler can say what world it attempted to
load.

### Capacity Truth

Capacity Truth answers:

- whether byte capacity truncated the closure
- which internal tables saturated first
- whether failure happened by explicit cap, not by silent corruption

Capacity Truth is green when failure modes are attributable to a specific
resource frontier.

### Execution Truth

Execution Truth answers:

- which modules were actually lexed
- which modules were parsed
- whether module-local execution surfaces are attributed correctly

Execution Truth is stronger than Closure Truth. A module can appear in a
closure witness and still fail execution truth.

### Verdict Truth

Verdict Truth answers:

- whether the compiler is authorized to say `OK` or `REJECT`
- whether disagreement should collapse to `UNKNOWN`
- whether a scalar surface is reporting an actual semantic verdict or only a
  driver-local summary

### Provenance Truth

Provenance Truth answers:

- which path produced the verdict
- whether the result came from rebuilt direct truth, wrapper mediation, or a
  mixed path

Wrapper provenance can be operationally useful even when rebuilt direct truth
is not yet authoritative.

### Semantic Truth

Semantic Truth answers:

- whether the relevant checker/kernel/boundary logic produced the intended
  meaning for the target program

Semantic Truth is the strongest layer in this stack. It depends on the layers
above it instead of replacing them.

## Hard Rules

- Closure Truth does not imply Execution Truth.
- Execution Truth does not imply Semantic Truth.
- Direct-driver truth stays blocked until Execution Truth is restored on the
  target surface.
- Wrapper provenance may still be the most honest operational authority even
  when rebuilt direct paths exist.

## Current Anchors

### Tiny Frontier

The tiny rebuilt ontology frontier is documented in
[truth-frontier.md](./truth-frontier.md).

That note shows a narrow area where rebuilt direct and fallback compile agree
on good fixtures, while bad fixtures still collapse to wrapper-level `unknown`.

### Large-Surface M9 Matrix

Recent M9 validation on the generated self-host establishes the current
large-surface truth shape:

| Probe | Cap | Closure Truth | Capacity Truth | Execution Truth |
| --- | --- | --- | --- | --- |
| `m4_large_surface_probe` | `2 MiB` | green | byte truncation at module `28` | blocked |
| `m4_large_surface_probe` | `4/8/16 MiB` | green | node/pool saturation, not bytes | red |
| `ontology_witness_program_probe` | `2 MiB` | green | byte truncation at module `30` | blocked |
| `ontology_witness_program_probe` | `4/8/16 MiB` | green | node/pool saturation, not bytes | red |

More concretely:

- `m4_large_surface_probe` reaches `module_count=46` and
  `total_requested_bytes=2810871`
- `ontology_witness_program_probe` reaches `module_count=48` and
  `total_requested_bytes=2834721`
- at `4/8/16 MiB`, both surfaces still hit `ND_COUNT=262143`, `ovf_nd=1`, and
  `ovf_pool=1`
- the first attributed large-surface parse failures currently land in
  `self-hosted/compiler/module_loader.sio`

This is the current example of Closure Truth green plus Execution Truth red.

## Current Reading

The present maturity picture is:

- Closure Truth: substantially improved by bundle witnesses
- Capacity Truth: substantially improved by explicit truncation and saturation
  reporting
- Execution Truth: restored on validated import probes, not yet restored on the
  large ontology-sized surface
- Verdict Truth: still mixed on ontology semantics
- Provenance Truth: wrapper path remains stronger than rebuilt direct semantic
  authority
- Semantic Truth: not reopened broadly

## Related Docs

- [compiler-maturity-blueprint.md](./compiler-maturity-blueprint.md)
- [truth-frontier.md](./truth-frontier.md)
- [module-closure-truth.md](./module-closure-truth.md)
- [semantic-contracts.md](./semantic-contracts.md)
