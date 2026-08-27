<!-- docs:meta
topic_id: repo.docs.research.pireus-xor-material-matching-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-xor-material-matching-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus XOR Material Matching

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Chain

The Garden record is:

```text
path=docs/internal/garden/seeds/2026-08-27-pireus-xor-material-matching.md
sha256=c07d58a40a457b8fa3ce524eb625fa17ffdbdf04e521a584cc20dba4eb4c13f1
```

The selector executable existed before this document and before the frozen
matcher. Sounio produced the 256 rows, counts, target states, negative
witnesses, and digests. The matcher was then added from that stream. A
post-matcher execution was byte-identical.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

`PARITY_OPEN` and `CLAIM_READY` remain false.

## Frozen Parent

The live Pireus operation parent passes its complete frozen matcher and its
result digest equals:

```text
operation_semantics_sha256=40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
operation_receipt_sha256=9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330
operation_result_digest=84edf6bae148754ebd0e8722368e2eb06095cd929779c36def4f3bb5000013a3
```

This child realizes only the parent's `XOR_PERMUTE` node. It neither changes
nor rederives the remaining operation graph.

## Selector Semantics

The frozen shape is:

```text
bits=4
dimension=16
element_bits=64
chunk_lanes=8
chunk_count=2
cell_count=256
group_count=32
```

For every `d in [0,15]`, `c in [0,1]`, and `l in [0,7]`, the canonical row is:

```text
index        = 16*d + 8*c + l
i            = 8*c + l
j            = i XOR d
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

Because the high one-bit and low three-bit fields are disjoint,

```text
j = 8*source_chunk + source_lane.
```

The Sounio executable enumerates all 256 rows and records 256 successful
reconstructions. For each of the 32 fixed `(d,c)` groups, `source_chunk` is
independent of `l`; all 32 groups pass. Thus the abstract layout needs one
source chunk per output chunk and an eight-lane selector.

The enumeration checks the executable implementation. The bit-field identity
above explains why it holds. Neither statement names a hardware instruction.

## XED Evidence

The exact vendor input is:

```text
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
path=datafiles/avx512f/avx512-foundation-isa.xed.txt
bytes=458470
sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
```

The child binds three lineage objects:

```text
frozen_content=5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612
current_historical_envelope=d96d6d57ba1e296930caec5f4f0aff8e2898b3b1d5df6bfaacb96a19333266f7
current_receipt=2dfc243381acb8d365112b3b4075ccabf944de6ff081b4626f9a4f693f136af6
```

Sounio reloads the vendor bytes, checks the vendor digest, parses them through
the frozen importer, rebuilds the ontology, and queries:

```text
total_forms=8
vpermpd_forms=4
vpermi2pd_forms=2
vpermt2pd_forms=2
```

Those live counts equal the values recorded in the frozen parent receipt. The
field `frozen_receipt_reference_present=true` means the reference and expected
record are present; it does not claim the Markdown receipt was decoded by the
Sounio module.

The following remain false:

```text
selector_behavior_derived_from_form_presence
selector_semantics_receipt_present
instruction_match_authorized
immediate_sufficiency_authorized
two_source_necessity_authorized
```

XED form grammar is therefore ontology evidence, not an instruction-behavior
theorem. In particular, this freeze does not answer whether every XOR selector
is realizable by `VPERMPD`, whether an immediate form suffices, or whether a
two-source form is needed.

## Canonical Targets

| Target | Canonical | Candidate plan | Observed | Material receipt |
| --- | --- | --- | --- | --- |
| Darwin Xeon | true | two abstract f64x8 chunks | false | false |
| Apple Silicon | true | unresolved | false | false |
| DGX | true | unresolved | false | false |

Darwin machines are Xeon in the frozen target profile. This result contains no
emitted-code or hardware-execution receipt for any target.

## Negative Surface

All 21 Sounio negatives pass. They cover absent and mismatched operation
parents and receipts; wrong logical, dimensional, element, and chunk widths;
out-of-range displacement, chunk, and lane; corrupted source-chunk and
source-lane selectors; unsupported two-source, immediate, behavior,
emitted-use, target-observation, and cost promotions; parity or claim before
their receipts; and prohibited semantic-producer roles.

Claim-policy mutations run through the same gate used by the accepted result.
The local producer-role witness does not enforce process execution. Loom is
the external pre-action guardian and remains required.

## Frozen Digests

| Object | SHA-256 |
| --- | --- |
| selector cells | `fd515df6be12f316cc771cf043aa0220bae7556362074a12eefe7f5a8f718928` |
| parent and XED evidence | `824e0d324a62a817f2d116f51e6b909534add1a3bae46a093dc09139d1230774` |
| target declarations | `3e2e2f881cc866c4a99b6b95ea933153df35e0472b42f51f9a0c5a2f0cb975b5` |
| negative witnesses | `95f189ec40d69f9b2dcc3263557e6076e92324200e5cc7ccb7c525fca19e82bb` |

The final authority stream contains 2,675 lines and 35,870 bytes. Its SHA-256
is `009f7fed37b8d20909c2a4915d9231a3416eaba41998b4290e3787274e85537f`.

## Closed Claims

The frozen semantics do not establish:

- instruction selector behavior or instruction equivalence;
- `VPERMPD`, `VPERMI2PD`, or `VPERMT2PD` sufficiency or necessity;
- an immediate selector encoding;
- an emitted instruction, instruction count, latency, throughput, or speedup;
- a lowering for twist, multiply, fixed ascending-`i` reduce, or output;
- a material observation on Darwin Xeon, Apple Silicon, or DGX;
- Walsh-Hadamard diagonalization or subquadratic complexity;
- a Fano-plane explanation;
- Lean 4, Koka, C++, or Haskell parity.

The next admissible research step is a separately admitted selector-semantics
receipt for an ISA family. It is not yet `PARITY_OPEN`.
