<!-- docs:meta
topic_id: repo.docs.research.pireus-dgx-ptx-shfl-lowering-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-dgx-ptx-shfl-lowering-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus DGX PTX SHFL XOR Lowering

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

```text
GARDEN commit=e5275df61f60
SOUNIO_EXECUTABLE=stdlib/hardware/pireus/dgx_ptx_shfl_lowering.sio
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The DGX Garden existed before the child source. Sounio reconstructed the
complete corpus, parsed the selected section, and emitted the control and
coverage result before this document recorded it.

## Frozen Parents

```text
lowering_source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
lowering_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
lowering_receipt_sha256=daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
ptx_source_sha256=ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba
ptx_semantics_sha256=1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697
ptx_receipt_sha256=e68f6edacfa85c48cd3cb51ab4929975a187174b0b1ab980a2c0f0868f5f38fa
```

The frozen `prmt` parent supplies corpus and virtual-ISA lineage only. The
child does not promote its within-value byte permutation into a lane operation.

## Pinned PTX Corpus

```text
release=CUDA 13.2.0
ptx_isa=9.2
html_bytes=3428895
html_chunks=4
html_sha256=fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457
selected_chunk=part-001.part
selected_chunk_bytes=1000000
selected_chunk_sha256=3e080ba7e8e556e29aed0c69ef818de39cb48b67ee6b442bd018dcb6ffa9bd8d
selected_section_id=data-movement-and-conversion-instructions-shfl-sync
```

Sounio first invokes the complete frozen PTX importer, then structurally
locates one `shfl.sync` section in the pinned chunk. Only exact syntax and
semantics lines inside their selected blocks define executable rules.

## Derived PTX Semantics

The selected section yields:

```text
instruction=shfl.sync.mode.b32
modes=.up,.down,.bfly,.idx
payload_bits=32
membermask_bits=32
lane_bits=5
bval_bits=5
cval_bits=5
segmask_bits=5
introduced_ptx=6.0
target_sm_min=30
```

For `.bfly`, the parsed rule is:

```text
j = lane XOR bval
pval = (j <= maxLane)
if !pval then j = lane
d = SourceA[j]
```

The source also states that inactive or nonmember source lanes are undefined,
and that an executing thread outside `membermask` is undefined. Those are
admission conditions, not behavior to optimize away.

## XOR Coordinate Bridge

Sounio admits logical lanes `[0,15]` inside the 32-lane warp and derives:

```text
membermask_low=65535
segmask=0
cval=15
packed_c=15
bval=d
lane=i
j=i XOR d
d in [0,15]
i in [0,15]
```

With `segmask=0`, `maxLane=15`. Therefore all 256 computed sources remain in
the admitted active/member subset, and the own-lane fallback is never taken.

```text
displacements=16
logical_cells=256
matched_cells=256
mismatched_cells=0
in_range_cells=256
own_lane_fallback_cells=0
active_source_cells=256
member_source_cells=256
max_source_lane=15
```

One `f64` payload is decomposed into two `b32` components. For every output
lane, displacement, component, and bit, Sounio compares the symbolic input
coordinate
`(lane XOR displacement) * 64 + component * 32 + bit` against the coordinate
selected by the parsed `.bfly` source lane. This couples payload preservation
to the lane mapping instead of accepting an identity manufactured from the
component and bit counters alone:

```text
f64_components=2
payload_component_cells=512
symbolic_payload_bits=16384
matched_payload_bits=16384
logical_index_coverage=true
symbolic_payload_address_preservation=true
xor_permute_candidate=true
abstract_shfl_sync_instructions=32
identity_shfl_sync_instructions=2
nontrivial_shfl_sync_instructions=30
```

The raw abstract count is 16 displacements times two `b32` components and
deliberately includes the two identity-component cells for `d=0`. The separate
non-trivial count excludes them. Neither count is a SASS instruction count, an
emitted sequence, a minimum, or a performance estimate.

Candidate digest, as eight emitted unsigned 32-bit limbs:

```text
1996463492:2773712531:2232409634:959326894:3198512505:1781073490:2005131219:1992148529
```

## Negative Surface

All 25 Sounio witnesses pass. They refuse missing Garden, frozen-parent drift,
corpus drift, section-shape drift, prose-as-semantics, `prmt` laundering,
foreign controls, logical aliasing, invalid participation, inactive or
out-of-bound sources, symbolic payload-address loss, selector-cell drift,
whole-operation
promotion, foreign operation import, exact-tree promotion, PTX-to-SASS
promotion, canonical-to-observed promotion, cost claims, non-Sounio authority,
early parity, early claim readiness, PTX-parent drift, semantic incompleteness,
and loss of canonical-target status.

## Authority Boundary

```text
dgx_canonical=true
dgx_observed=false
ptx_sass_equivalence_count=0
dgx_material_receipt_count=0
compiler_emission_count=0
cost_record_count=0
frozen_operation_nodes=5
resolved_selector_nodes=1
unresolved_other_nodes=4
exact_ascending_i_tree_reduction=false
PARITY_OPEN=false
CLAIM_READY=false
```

PTX virtual-ISA acceptance is not generated SASS, DGX capability, or executed
DGX behavior. This freeze establishes only a local `.bfly` candidate for the
frozen `XOR_PERMUTE` node.

## Compiler Path

The receipt-bearing execution uses the explicit `lean_single` bootstrap seed.
The canonical Madaros path remains blocked on its current cross-module field
resolution diagnostics. The bootstrap result is recorded as Sounio authority,
not as a retrospective fallback supplied by another language. No compiler
emission or DGX observation is inferred from it.
