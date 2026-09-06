<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-metal-table-cells-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-metal-table-cells-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple Metal Table-Cell Projection Receipt

Date: 2026-08-27

## Authority State

```text
concept=SOUNIO-PIREUS-APPLE-METAL-TABLE-CELL-PROJECTION
producer_language=Sounio
producer_role=SEMANTIC_AUTHORITY
stage=SEMANTICS_FROZEN
parity_open=false
claim_ready=false
capability_facts_created=0
apple_silicon_observed=false
dgx_observed=false
```

This receipt freezes a geometry and text-ownership projection. It does not
interpret any cell as an Apple hardware capability and does not promote Apple
Silicon or DGX declarations to observations.

## Ordered Provenance

The result did not exist at Garden admission.

```text
garden_commit=5654776bb0b55b8d0169cf195ecc46ec17a4b154
garden_seed_sha256=c7d1c96a0c90a35736a45c3ad648e12c57015ed0728c41c9328fd1c5c7ef93be
first_sounio_executable_commit=bd982e6747
first_sounio_result_sha256=6679906deee291b2590ef39a8645e42a1fe650ee4b6baefd1728ee61eed659be
semantics_freeze_commit=5a823e3520aeafac0d19a3342705ce7b0ea088e8
```

Commit `bd982e6747` first produced the result in Sounio without treating it as
an expected value. Commit `5a823e3520` then added that prior result as the
fail-closed expectation. This preserves:

```text
GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN
```

No parity language was executed.

## Frozen Hashes

```text
sounio_source=stdlib/document/pdf_table.sio
sounio_source_sha256=719863ef62e58d3b6e34965747e4e7518e704ae0694aad86da40f46b9058255c
frozen_semantics=examples/pireus_apple_metal_table_cells.sio
frozen_semantics_sha256=b7ab9d79bc3ceb6f2c0ae5b46772b637092a903036c51c7a0d9b148dda523b7b
negative_witness=examples/pireus_apple_metal_table_cells_negatives.sio
negative_witness_sha256=706a711f3d9916e6e9bf34c2d09ceff461a8a4347273aa2d2c78a2ac62dcd33f
parent_text_projection_sha256=aebdea5034dc20201edf555bbb257e2971eb86127be602e630be1135564f93f8
cell_projection_sha256=6679906deee291b2590ef39a8645e42a1fe650ee4b6baefd1728ee61eed659be
cell_projection_serialized_bytes=223436
```

## Frozen Semantics

The projection:

1. Uses the pinned Sounio PDF loader and requires its exact inventory.
2. Decodes all 18 content streams with the Sounio inflater.
3. Interprets PDF `q`, `Q`, and `cm` graphics state in fixed-point units.
4. Projects `m`, `l`, `h`, and `re` path geometry through the active CTM.
5. Rounds path numbers beyond six fractional digits to the nearest
   millionth, with halves away from zero, and counts every quantized value.
6. Distinguishes stroke, fill, fill-stroke, and discarded clipping paths.
7. Admits a fill as a rectangle only when its four projected edges contain
   exactly two horizontal and two vertical non-degenerate segments.
8. Canonicalizes exact duplicate rectangles by page and bounds.
9. Assigns every non-empty text object to the unique smallest containing
   canonical rectangle. An exact smallest-area tie remains unassigned.
10. Preserves empty rectangles and the parent text object's exact scalar
    start/count ranges in the cell digest.

The digest binds the parent text digest, canonical cell order and bounds,
cell object/scalar counts, and every text-object-to-cell assignment.

## Toolchain

```text
entrypoint=./bin/souc
entrypoint_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
engine=lean_single
engine_role=bootstrap_seed
engine_path=bin/souc-lean-single-x86_64
engine_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
raw_elf_invoked=false
python_invoked=false
rust_invoked=false
```

The canonical wrapper was always used. `SOUNIO_SOUC_ENGINE=lean_single`
selects the preserved bootstrap engine through that wrapper.

## Hardware

```text
architecture=x86_64
vendor=GenuineIntel
model=INTEL(R) XEON(R) GOLD 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

## Commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_apple_metal_table_cells.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_apple_metal_table_cells.sio \
  /workspace/.tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_apple_metal_table_cells_negatives.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_apple_metal_table_cells_negatives.sio
```

The positive witness was repeated after freezing and returned the same result.

## Sounio Result

```text
pages=18
decoded_bytes=1095453
operators=77986
segments=17872
horizontal_segments=9722
vertical_segments=8150
oblique_segments=0
quantized_numbers=30
fill_paths=1904
filled_rectangles=1904
nonrectangular_fills=0
duplicate_rectangles=12
canonical_cells=1892
nonempty_cells=1870
empty_cells=22
text_objects=3193
empty_text_objects=3
assigned_objects=3190
unassigned_objects=0
nested_resolutions=3117
exact_tie_objects=0
containment_matches=8788
cell_projection_serialized_bytes=223436
cell_projection_sha256=6679906DEEE291B2590EF39A8645E42A1FE650EE4B6BAEFD1728EE61EED659BE
expected=true
```

## Negative Gates

```text
missing_corpus=DENY
invalid_geometry=DENY
page_mismatch=DENY
cell_capacity_exceeded=DENY
```

The upstream Garden transaction also denied premature execution, premature
parity/claim promotion, and a deliberate Python-oracle attempt before this
child existed. This witness itself executes no host-language oracle.

## Compiler Divergence

The default Madaros check currently fails on the imported large-PDF surface
with repeated `E012 this type has no field named` diagnostics, including the
already frozen parent modules. The same wrapper with the documented
`lean_single` bootstrap engine checks and executes both witnesses. Therefore:

```text
default_path=FAIL
bootstrap_wrapper_path=PASS
fallback_claim=false
```

This is a compiler-path divergence, not evidence that the frozen Sounio result
was produced by another language. It must remain visible until Madaros reaches
parity on this source surface.

## Closed Boundary

This receipt authorizes neither parity nor a hardware claim:

```text
PARITY_OPEN=false
CLAIM_READY=false
```
