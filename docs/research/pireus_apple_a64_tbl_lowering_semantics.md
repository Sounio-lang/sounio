<!-- docs:meta
topic_id: repo.docs.research.pireus-apple-a64-tbl-lowering-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-apple-a64-tbl-lowering-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus Apple A64 TBL XOR Lowering

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

```text
GARDEN commit=e5275df61f60
SOUNIO_EXECUTABLE=stdlib/hardware/pireus/apple_a64_tbl_lowering.sio
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The Garden was committed before the child source. The first Sounio execution
parsed the vendor XML and emitted every count and verdict below before this
document recorded them.

## Frozen Parents

```text
lowering_source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
lowering_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
lowering_receipt_sha256=daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
aarchmrs_source_sha256=ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b
aarchmrs_semantics_sha256=ed66cc2e2fe27ce06842c1ef2091e2f482b8bcb2d4b84e4e649361ca957b7b14
aarchmrs_receipt_sha256=cd64c91c330c9a81e554408a10de4bccbdf9984395ec049c48dc99148aa11934
```

The child also verifies the exact Apple Garden hash and the complete AARCHMRS
corpus binding. It does not amend either parent.

## Pinned Vendor Inputs

```text
release=Arm A-profile A64 ISA XML 2025-12
archive_sha256=845ed227a6692ddb6b602da2ecbbac776620195a9c001ec576ced3a9a53dc26b
tbl_advsimd.xml bytes=14897 sha256=48ef32ed67b9824ba39eb58518faec196472c3a574cf1bbe1f3a494811a6cbbe
tbx_advsimd.xml bytes=14962 sha256=fa21f8c0784ec327ca9089552d22b55e0eb4b9dd6e0a2eeb078eeed0e203ca79
notice.xml bytes=5212 sha256=7f6e2780187dc8eb12b53d97eb435be19597b1af256a84fb44d4b5bd41846747
```

Sounio verifies all three files. It structurally parses the selected TBL XML;
the adjacent TBX and notice files close transport and license lineage only.

## Derived A64 Semantics

The structural parser found exactly one selected root, four encodings, one
decode block, and one execute block. It derived:

```text
Q.width=1
len.width=2
table_registers=4
table_bits=512
table_bytes=64
output_bits=128
output_bytes=16
f64_per_table=8
f64_per_output=2
```

The accepted bounded ASL rules define `datasize`, `elements`, `regs`, the TBL
discriminator, table construction, zero default, byte index, table bound,
byte lookup, and destination write. These rules came from the selected
operation blocks, not from an arbitrary prose substring.

## XOR Coordinate Bridge

For displacement `d` and logical index `i`, both in `[0,15]`, Sounio derives:

```text
expected_source = i XOR d
source_group = expected_source / 8
source_element = expected_source % 8
byte_control = source_element * 8 + payload_byte
payload_byte in [0,7]
```

Each 512-bit table contains eight `f64` values. Each 128-bit TBL result carries
two `f64` values, so one displacement requires eight abstract TBL applications.
The source group is constant across the two output elements in each result.
It selects which logical block of eight `f64` values supplies the four A64
table registers for that abstract application; the group is not smuggled into
the six-bit TBL byte control. Material register loads and compiler emission
remain outside this candidate.

The complete finite result is:

```text
displacements=16
logical_cells=256
matched_cells=256
mismatched_cells=0
in_domain_source_cells=256
out_of_domain_source_cells=0
bijective_displacements=16
dimension_matches_bits=true
byte_control_cells=2048
matched_byte_controls=2048
max_control=63
out_of_range_controls=0
abstract_tbl_applications=128
```

Logical-index equality is separate from payload equality. Sounio symbolically
tracks every bit position of each 64-bit input payload through the eight byte
controls:

```text
symbolic_payload_bits=16384
matched_payload_bits=16384
logical_index_coverage=true
symbolic_payload_address_preservation=true
xor_permute_candidate=true
```

For every payload bit, Sounio records both absolute coordinates:

```text
expected_bit = expected_source * 64 + payload_byte * 8 + bit
reconstructed_bit = decoded_source * 64 + decoded_byte * 8 + bit
```

Both coordinates enter the candidate digest. The coverage predicate also
requires `dimension == 1 << bits`, rejects every source outside `[0,15]`, and
for each of the 16 displacements counts exactly one visit to each of the 16
source indices. Only then can byte-control and symbolic payload-address
preservation contribute to the candidate verdict. `out_of_range_controls=0`
is a finite closure of the derived six-bit control domain; it is not an
independent hardware observation.

Candidate digest, as eight emitted unsigned 32-bit limbs:

```text
472477255:3903797350:1348128039:308036239:1349218781:3920188038:1317640714:1357523880
```

## Negative Surface

All 15 Sounio witnesses pass. They independently refuse parent drift, missing
decode, missing execute, wrong `Q` width, wrong `len` width, out-of-range byte
control, aliased logical source, swapped payload byte, whole-operation
promotion, exact-tree promotion, architecture-to-observation promotion, early
parity, early claim readiness, loss of canonical-target status, and promotion
of a non-Sounio producer to semantic authority.

The same authority predicate is live in the success path. It consumes the
verified parent closure, decoded and executed rule sets, derived maximum
control, full logical-cell and byte-control counts, unresolved-operation
boundary, observation and receipt state, canonical-target declaration,
Sounio authority, and the closed parity/claim flags. The negative witnesses
mutate that predicate one field at a time; they are not a second acceptance
mechanism.

## Authority Boundary

```text
apple_silicon_canonical=true
apple_silicon_observed=false
apple_material_receipt_count=0
compiler_emission_count=0
cost_record_count=0
unresolved_other_nodes=4
exact_ascending_i_tree_reduction=false
PARITY_OPEN=false
CLAIM_READY=false
```

This freeze establishes a local A64 TBL candidate for the frozen
`XOR_PERMUTE` node. It does not establish Apple Silicon availability, compiler
emission, an instruction count, a cost, a speedup, or the complete five-node
Cayley-Dickson operation.

## Compiler Path

The receipt-bearing execution uses the explicit `lean_single` bootstrap seed.
The canonical `bin/souc` default selected Madaros v0.80.0 and failed closed
while checking the imported frozen surface, producing a large cascade of
nameless `E012` field diagnostics and an `E008` return diagnostic. That is a
compiler-path blocker, not an alternate semantic result. No frozen parent or
Madaros source was changed in this lane.
