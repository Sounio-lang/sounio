<!-- docs:meta
topic_id: repo.docs.research.pireus-intel-vpermpd-selector-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-intel-vpermpd-selector-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus Intel VPERMPD Selector Semantics

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

The Garden record was committed first:

```text
path=docs/internal/garden/seeds/2026-08-27-pireus-intel-vpermpd-selector-semantics.md
sha256=6eec7cf4cab6e716a5205aef8e9bc59a73e0f718c9300e986362b37b229e5a5c
commit=19d2c05578
```

The first Sounio executable immediately followed it:

```text
commit=0258185d1cfc40a1a71f766059654f3f1eb0e294
```

The expected selector rule, normalized counts, and material-match result were
emitted by that Sounio artifact before this prose was written.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

`PARITY_OPEN` and `CLAIM_READY` remain false.

## Exact Inputs

### Intel SDM

```text
publisher=Intel Corporation
manual=Intel 64 and IA-32 Architectures Software Developer's Manual
manual_version=092
volume=2C
pdf_bytes=3298744
pdf_sha256=939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
pdf_envelope=%PDF-1.6
```

### Intel XED

```text
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
path=datafiles/avx512f/avx512-foundation-isa.xed.txt
bytes=458470
sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
```

### Frozen XOR Material Parent

```text
source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
```

Sounio reloads the parent and XED input at execution time. A stored Markdown
receipt is a lineage identifier, not a substitute for the live parent match.

## Bounded PDF Projection

The Sounio reader verifies the exact file length and computes SHA-256 over the
loaded bytes before it accepts content. It validates the PDF 1.6 envelope,
EOF marker, and indirect-object `startxref` target, then scans the bounded
stream inventory.

```text
stream_markers=2034
flate_streams=2032
decoded_flate_streams=2031
flate_decode_failures=0
oversized_flate_outputs=1
oversized_output_stream_offset=3292436
oversized_output_class=/XRef
oversized_output_content_candidate=false
```

One Flate stream exceeds the bounded decoded-output buffer. Sounio classifies
that exact stream as an xref stream and rejects it as a content candidate. The
semantic gate therefore does not pretend that all 2032 Flate streams were
inflated, and it does not silently ignore an unclassified content stream.

The decoded PDF text projection yields unique normalized anchors:

```text
description_page=5-522
description_stream_offset=1678050
description_page_matches=1
vector_form_matches=1
description_selector_rule_matches=1

operation_page=5-524
operation_stream_offset=1684617
operation_page_matches=1
operation_vector_form_matches=1
operation_selector_cells=8
```

These anchors provide page and stream lineage for the normalized fields. The
document does not reproduce the vendor section verbatim.

## Normalized Vector-Control Semantics

The selected form is normalized as:

```text
form=EVEX_512_VECTOR_CONTROL
vector_bits=512
element_bits=64
index_element_bits=64
selector_bits=3
selector_mask=7
source_lanes=8
```

For destination lane `k in [0,7]`, let `control[k]` be the corresponding
64-bit control element. The selected source lane is:

```text
selected_lane(k) = control[k] & 7
```

This rule selects one of eight lanes from one 512-bit source vector. Masking,
merge/zero behavior, memory-fault behavior, and other forms are outside the
selector question frozen here.

## Match Against The Frozen XOR Layout

The parent layout gives, for displacement `d`, output chunk `c`, and output
lane `l`:

```text
bits         = 4
dimension    = 16
chunk_count  = 2
chunk_lanes  = 8
d            in [0,15]
c            in [0,1]
l            in [0,7]
i            = 8*c + l
j            = i XOR d
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

For a fixed `(d,c)`, `source_chunk` does not depend on `l`, while each
`source_lane` lies in `[0,7]`. XOR by the fixed three-bit value `d & 7` is a
bijection, so those eight values are exactly a permutation of `[0,7]`.
Therefore one vector-control form application can express each eight-lane
group by setting its eight control elements to the parent `source_lane`
values.

Sounio checks the complete finite table:

```text
xor_selector_cells=256
xor_selector_matches=256
xor_selector_failures=0
one_source_groups=32
abstract_form_applications=32
vector_control_complete=true
```

The result authorizes this narrow instruction-form match:

```text
one_source_form_sufficient=true
two_source_form_required=false
instruction_match_authorized=true
```

The scope is the frozen `XOR_PERMUTE` selector layout only. It does not state
that the compiler emits the form, that the target executes it, or that the
full Cayley-Dickson multiplication uses 32 instructions.

## Immediate Form Is Separate

The immediate-control form does not share the complete vector-control result.
The Sounio executable tests all eight low displacement patterns:

```text
patterns_tested=8
supported=4
refused=4
complete=false
```

The four refused patterns cross the immediate form's 256-bit selection halves.
For `d in [0,3]`, lane bit 2 is preserved and the same two-bit permutation is
applied to both halves. For `d in [4,7]`, lane bit 2 is flipped, which the
immediate form cannot express. No immediate selector is therefore accepted as
a complete realization of the frozen layout.

## Negative Surface

All 18 in-process Sounio witnesses pass. They cover loaded-byte digest
mutation; wrong vector, data-element, index-element, selector, and lane widths;
an out-of-domain vector selector; incomplete and out-of-domain immediate
patterns; missing PDF semantics, missing parent validity, and wrong group
count; and attempted promotion of material observation, lowering, cost,
parity, or claim readiness.

The external Loom boundary separately classifies producer language and stage.
It is not replaced by the in-process witnesses.

## Compiler-Path Boundary

The receipt-bearing authority command uses `./bin/souc` with
`SOUNIO_SOUC_ENGINE=lean_single`. The default Madaros v0.80 checker currently
fails while checking the 12-module imported surface with pre-existing
cross-module diagnostics including `E008`, `E012`, and `E035`. This failure is
recorded rather than treated as a fallback result or as evidence against the
successful explicit authority path.

No Madaros repair is part of this semantic lane.

## Canonical Targets

| Target | Canonical | Selector result | Materially observed |
| --- | --- | --- | --- |
| Darwin Xeon | true | Intel vendor form matched to abstract layout | false |
| Apple Silicon | true | unresolved in this lane | false |
| DGX | true | unresolved in this lane | false |

The PDF establishes vendor-defined semantics. It does not establish AVX-512
availability, compiler emission, or execution on any Darwin Xeon.

## Closed Claims

This freeze does not establish:

- any emitted instruction or instruction count;
- any hardware capability or execution observation;
- latency, throughput, cost, register pressure, or speedup;
- a lowering for twist, multiplication, fixed ascending-`i` horizontal
  reduction, or output placement;
- the earlier roughly 112-instruction estimate;
- Apple Silicon, DGX, or cross-ISA parity;
- Walsh-Hadamard diagonalization or subquadratic twisted convolution;
- a theorem about the Fano plane or the seven-negative-sign regularity;
- Lean 4, Koka, C++, or Haskell parity.

The next legal transition after the Loom freeze is `PARITY_OPEN`; this document
does not request it.
