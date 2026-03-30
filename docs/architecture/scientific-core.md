<!-- docs:meta
topic_id: repo.docs.architecture.scientific-core
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.scientific-core
-->

# Scientific Core

## Summary

This note defines why scientific computing belongs in compiler architecture for
Sounio, not only in stdlib inventory.

The repo and public docs already frame scientific support as a system-level
story involving epistemic typing, scientific runtime lanes, dedicated compiler
surfaces, and hypercomplex work. This note makes that architecture explicit for
internal planning.

## Why Scientific Core Belongs In Compiler Architecture

Scientific computing in Sounio is not only a collection of domain packages.

It interacts with:

- typed evidence and provenance expectations
- module and closure execution on larger surfaces
- runtime and lowering decisions
- IR-visible scientific types and operations

Treating Scientific Core as compiler architecture keeps these responsibilities
aligned with the rest of the maturity program instead of scattering them across
stdlib-only narratives.

## What Belongs Here

Scientific Core includes:

- units and dimensional semantics where compiler/runtime behavior matters
- scientific effects and domain-relevant execution contracts
- numerical kernels that rely on compiler/runtime guarantees
- GPU and SIMD lowering surfaces for scientific workloads
- hypercomplex algebra in the compiler

Hypercomplex work belongs here because the repo already treats it as more than
an ordinary library topic. The current public scientific docs also position
hypercomplex work as part of the broader compiler/science map.

## What Does Not Belong Here

Scientific Core does not include:

- pure domain examples that have no compiler or runtime consequence
- roadmap-only claims with no source-tree or fixture anchor
- marketing claims that outrun checked artifacts or passing gates

When a scientific area has only source-tree inventory and no validated path, it
should be described as inventory or direction, not as proven compiler behavior.

## Relation To Other Cores

### Evidence Core

Evidence Core provides the witnesses and provenance needed to describe what the
compiler actually executed for a scientific surface.

### Module Core

Module Core provides truthful resolved closure and execution for scientific
surfaces that are large enough to stress the self-hosted compiler.

### Semantic Core

Semantic Core governs typed meaning, rejection behavior, and boundary policy
where scientific constructs interact with checker logic.

Scientific Core depends on these other cores instead of replacing them.

## Hypercomplex Workstream Expectations

The current hypercomplex lane should be treated as a formal compiler workstream
with, at minimum:

- a type and literal model
- layout and ABI expectations
- lowering targets such as SIMD or GPU where relevant
- witness and provenance implications for scientific execution

This keeps hypercomplex work aligned with the same evidence-bearing discipline
required elsewhere in the compiler.

## Current Anchors

The strongest current anchors for this note are:

- public scientific overview:
  `website/src/content/docs/en/compiler/scientific-features.mdx`
- implementation-facing source-tree lanes:
  `self-hosted/hypercomplex/`, `self-hosted/gpu/`, `self-hosted/tensor/`,
  `self-hosted/distributed/`
- passing science-oriented test surfaces:
  `tests/stdlib/fmri/`, `tests/stdlib/darwin_pbpk/`, and the broader hyper
  execution lanes described in the public scientific docs

## Current Reading

Scientific Core should be treated as an explicit architecture lane in the
Compiler Maturity Program, but not yet as proof that every scientific surface
is equally mature.

The responsible reading is:

- the compiler already has real scientific ambition and real scientific proof
  points
- larger scientific or ontology-adjacent surfaces still depend on module and
  execution truth maturing first

## Related Docs

- [compiler-maturity-blueprint.md](./compiler-maturity-blueprint.md)
- [truth-layers.md](./truth-layers.md)
- [module-closure-truth.md](./module-closure-truth.md)
