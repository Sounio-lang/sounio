<!-- docs:meta
topic_id: repo.docs.research.hypercomplex-compiler-track
authority: repo_only
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.hypercomplex-compiler-track
-->

# Hypercomplex Algebra in the Compiler

This document opens the Wave 2 hypercomplex lane as an internal research track.
It is not a public capability claim, not a merge-blocking program, and not a
statement that the checked public artifacts currently expose quaternion,
octonion, or sedenion compiler features.

## Repo-grounded evidence

Current relevant sources in the repository:

- `self-hosted/hypercomplex/quat_simd.sio`
- `self-hosted/hypercomplex/octonion.sio`
- `self-hosted/hypercomplex/test_quat_simd.sio`
- `self-hosted/hypercomplex/test_octonion.sio`
- `self-hosted/hlir/ir.sio`
- `examples/gpu_hypercomplex.sio`
- `docs/research/self_hosted_hypercomplex.md`

Current facts:

- `self-hosted/hlir/ir.sio` already names `HlirTypeQuat`,
  `HlirTypeOctonion`, and `HlirTypeSedenion`
- `hlir_type_is_hypercomplex` already treats those kinds as a coherent family
- `self-hosted/hypercomplex/*` contains executable quaternion/octonion/sedenion
  math and tests
- `examples/gpu_hypercomplex.sio` is a CPU-runnable baseline example, not a
  GPU capability proof
- `docs/research/self_hosted_hypercomplex.md` is historical lineage, not a
  current production contract

## Touchpoint inventory

### Typing and type representation

Repo evidence:

- `self-hosted/hlir/ir.sio` declares hypercomplex HLIR kinds
- `self-hosted/hypercomplex/quat_simd.sio` and `octonion.sio` define concrete
  value-level representations and algebra

Classification:

- `prototype-safe`
  - HLIR type inventory and representation audits
  - type-size/layout review for quat/octonion/sedenion carriers
- `production-deferred`
  - public type-checker claims for source-language hypercomplex types
  - stable ABI guarantees for hypercomplex argument passing

### Symbolic and lowering touchpoints

Repo evidence:

- `self-hosted/hlir/ir.sio` says HLIR is consumed by LLVM and GPU codegen
- `examples/gpu_hypercomplex.sio` shows a scalar baseline worth preserving as a
  reference oracle

Classification:

- `prototype-safe`
  - internal lowering experiments that map known hypercomplex ops onto existing
    scalar/vector forms behind non-public fixtures
  - reference-oracle comparisons against `examples/gpu_hypercomplex.sio`
- `production-deferred`
  - checked-artifact lowering claims for hypercomplex kernels
  - public GPU/runtime support claims for hypercomplex lowering

### IR normalization and optimizer identities

Repo evidence:

- quaternion multiplication exists in `quat_simd.sio`
- octonion and sedenion multiplication semantics exist in `octonion.sio`

Classification:

- `research-only`
  - algebra-aware rewrite systems
  - canonicalization strategies for octonion/sedenion expressions
  - optimizer identities beyond trivial scalar decomposition
- `production-deferred`
  - any optimizer pass that assumes associativity or multiplicative inverses

### Algebraic danger zones

These are the boundaries that must block naive optimizer or lowering work:

- octonions are non-associative
- sedenions have zero divisors
- inverse/normalization rules are numerically unstable near small norms
- reordering reductions can change results materially
- backend vectorization cannot assume scalar algebra laws that do not hold for
  the chosen structure

Classification:

- `research-only`
  - formalizing safe rewrite subsets under non-associativity
  - zero-divisor-aware optimization boundaries
- `production-deferred`
  - any pass that would silently reassociate, fuse, or factor expressions
    across unsafe algebraic boundaries

## Required classification set

### `research-only`

- optimizer algebra for octonion and sedenion expressions
- non-associative normalization theory
- zero-divisor-aware symbolic rewrites
- benchmark-driven decisions about GPU-native hypercomplex kernels

### `prototype-safe`

- HLIR/type inventory audits
- internal comparison harnesses against scalar/reference implementations
- non-public lowering experiments that preserve existing behavior
- fixture and oracle work that strengthens evidence without expanding claims

### `production-deferred`

- public type-system support claims
- checked-artifact hypercomplex ABI promises
- optimizer rewrites that depend on unsafe algebraic assumptions
- public GPU capability promotions for hypercomplex kernels

## Wave 2 operating rule

Hypercomplex work may advance as research and prototype inventory in parallel
with GPU Wave 2, but it must not:

- become a merge-blocking lane in this wave
- inflate public capability claims
- weaken selfhost fixed-point, parity, or provenance discipline
- outrun the evidence carried by tests, artifacts, and governed manifests
