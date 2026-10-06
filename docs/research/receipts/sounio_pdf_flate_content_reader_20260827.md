<!-- docs:meta
topic_id: repo.docs.research.receipts.sounio-pdf-flate-content-reader-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.sounio-pdf-flate-content-reader-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio PDF Flate Content Reader Receipt

**Date:** 2026-08-27
**Concept-ID:** `SOUNIO-PDF-FLATE-CONTENT-READER`
**Language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`
**Requested stage:** `SEMANTICS_FROZEN`

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The Garden artifact existed and received Loom `GARDEN` and `COMMIT` ALLOW
decisions before implementation began. Its SHA-256 is
`80bc230e24b4817305249514c71ea7ccbf1f37b6457a585523427106aa4dc42c`.
The canonical docs-registry owner has an active exclusive claim over the
governance metadata, so the Garden, concept, semantics, and receipt were handed
to that owner instead of bypassing the pre-commit hook. After the local Garden,
freeze, and commit actions were all admitted by Loom, the two Sounio source
files were committed alone as
`00c2fbe79c6376c56b0fd7fea5fc154f6440424c`; the governance bundle remains
uncommitted in this lane until its canonical Garden acceptance is identified.
`PARITY_OPEN=false` and `CLAIM_READY=false`.

## Authority Source

| File | SHA-256 |
| --- | --- |
| `docs/internal/garden/seeds/2026-08-27-sounio-pdf-flate-content-reader.md` | `80bc230e24b4817305249514c71ea7ccbf1f37b6457a585523427106aa4dc42c` |
| `stdlib/document/pdf_flatedecode.sio` | `9361c69164c41421b13ad3b6763f3998edbae51076ac66e07c326716992de61a` |
| `examples/sounio_pdf_flate_content_reader.sio` | `0ca59bed14b4484a46955329cb3acbf410b73ab72cab4d9eacec62da19776708` |
| `docs/internal/concepts/sounio-pdf-flate-content-reader.md` | `1b53d90d5627a3872f6dd7ad011c91b92581f21fa1b2657d8352c909c2b0a3f0` |
| `docs/research/sounio_pdf_flate_content_reader_semantics.md` | `95de48c5b3372be22047c97e468eaa76fc1079c9983487d277de1195d02445f5` |

The executable example hash is bound to Loom as Sounio source. The paired
semantics hash is bound as frozen semantics. The implementation module and
Garden hashes are separately recorded so neither substrate can drift.

## Vendor Corpus

```text
url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
observed_last_modified=Tue, 09 Jun 2026 00:00:35 GMT
document_date=May 21, 2026
content_type=application/pdf
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
local_path=/tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf
```

The live URL is mutable. The payload is not committed. Any later vendor bytes
must begin at a new Garden pin rather than update this result in place.

## Toolchain

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
inflater_module=stdlib/compress/inflate.sio
inflater_module_sha256=2d788a553d3b0255c67789e1c8c2b2fc997762a76212db9613eaa1d8a5f762e3
```

The exact base toolchain record bound to Loom hashes to
`2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e`.

## Execution Hardware

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

The exact hardware record hashes to
`fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0`.

## Commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/sounio_pdf_flate_content_reader.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/sounio_pdf_flate_content_reader.sio \
  /tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf

/tmp/pireus-v01-ontology-validation-souc check \
  examples/sounio_pdf_flate_content_reader.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/sounio_pdf_flate_content_reader.sio -- \
  /tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf
```

The exact direct-run command record, with final `LF`, hashes to
`3158b1528289dfdcc6bc509bd5631403dbdf73664b8d635e16717eba6ed48e52`.
Two direct runs were byte-identical. The rebuilt check returned `verdict=ok`,
`provenance=rebuilt_direct`, and unanimous driver/fallback evidence. The
rebuilt program-output suffix was byte-identical to the direct authority
stream. No raw ELF, host PDF parser, or host decompressor was invoked.

## Sounio-Produced Inventory

```text
PDF_CORPUS bytes=3041713 error=0 digest_match=1
PDF_ENVELOPE version_1_3=1 eof=1 startxref=2924864
PDF_XREF in_use=5833 declared_size=5834 root=5525 root_generation=0
PDF_PAGE_TREE root=5524 nodes=4 declared_pages=18 pages=18
PDF_FLATE direct_streams=33 selected_contents=18 redundant_filter_entries=18 omitted_stream_endobj=18 nonselected=15 oversized_nonselected=5 largest_length=446335
PDF_CONTENT_TOTAL compressed_bytes=198152 decoded_bytes=1095453 blocks=18 stored=0 fixed=0 dynamic=18 literals=22723 copies=80561 max_distance=32505
PDF_CONTENT_AGGREGATE length_prefixed_sha256=22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526
PDF_EXPECTED inventory=1
PDF_BOUNDARY content_bytes_only=1 text_operators=0 font_mapping=0 apple_feature_semantics=0 hardware_observations=0 lowering_claims=0
PDF_SUMMARY failures=0
```

The ordered per-content records are:

| Index | Page object | Content object | Compressed | Decoded | SHA-256 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 4 | 362 | 714 | `88f3f4d9fd40c9c3b9e66f9fa0076d92b599a6ef43862c0601ad7f29d09cb5c9` |
| 1 | 10 | 12 | 4,941 | 22,427 | `d95dff06469b94ed9e02a8c53aefd81a32ce408a46c54f1b435530650a285b63` |
| 2 | 18 | 20 | 11,821 | 65,628 | `394aecfb1ad0e0ea3ed96d65cf929d750b1e4d7bc33b8f6e4b62e1704fbd8315` |
| 3 | 26 | 28 | 11,266 | 66,554 | `b0d8b0066614c2bf6c6e8b08880ced2a3e5cad471c7945d8ae3f9fe03c4df382` |
| 4 | 29 | 31 | 12,700 | 70,511 | `18dc45b35292064597d5f40bc9fe0be1ef9bfb74b959026900236ba13f3e864b` |
| 5 | 45 | 47 | 13,640 | 66,988 | `dcb38624004aac82968f8ba8c3b3b37569b461e3f488f42eb7a4935d9efcca43` |
| 6 | 97 | 99 | 16,473 | 98,111 | `7bd074c337cb9b4a71bcd3667c9770583c7cbfe470c848e25f7126e204bc4b80` |
| 7 | 103 | 105 | 15,422 | 93,168 | `4ed23e2d13cbc56e3dcda2e26a9b1a32438654b61dcc747017893dfbaeb4e824` |
| 8 | 106 | 109 | 15,398 | 93,084 | `c8ec6338e6616f1dc9cee1620bb5a8a8dfa2f0a73cbe0a8929cdb9900cee5077` |
| 9 | 110 | 112 | 8,466 | 36,043 | `6351f68bb42875f7437bfe43e0bfbbf960ff77d94b9ac676912b7b36545b077b` |
| 10 | 126 | 128 | 11,784 | 64,197 | `8c4bc258a77d5c7943815fbcc1e19310952694315c66d639dd203d2cf37ed560` |
| 11 | 130 | 132 | 11,371 | 69,278 | `23136083b5c9fa4ea7f6d8362df31696f2c56a845986c820c57026eecb315791` |
| 12 | 133 | 135 | 9,558 | 56,271 | `b01c734a0be1322c7a16f5c717ed70f21f790d54e0275cf5603f012963030026` |
| 13 | 136 | 138 | 12,769 | 75,827 | `67a71bf976d7e9ec7cb7b4971d6ca8ad57798a92adf4687c0453899db4881bcc` |
| 14 | 139 | 141 | 14,241 | 72,566 | `e41098fcd347000f552cc7b7ac72e15966a4c8133f6b67e2a0fc14e55685a196` |
| 15 | 174 | 176 | 10,114 | 46,413 | `eb3702feac9fd4d2b82622473a771b96816c032113400749895960833e05f296` |
| 16 | 183 | 186 | 14,111 | 82,927 | `ff5167857cfff128c69cd6ed436a6d9dc975689231cbcea398da902cc056a3b4` |
| 17 | 227 | 229 | 3,715 | 14,746 | `cf4b55f42a7d82561f8ba88b76637b8c9693f82e17473536affa61bd621fdbaa` |

The exact authority stream, including bootstrap integer-printer line breaks,
hashes to:

```text
41947f41db6ba3fb1f380a5f6008282144aa19f25384c31842bc7c9f46232232
```

## Discovered Corpus Shape

Two initially strict checks refused the pinned file and were retained as
explicit profile facts:

1. all 18 selected dictionaries repeat the identical direct
   `/Filter /FlateDecode` entry;
2. all 18 selected streams omit `endobj` after `endstream`.

The first shape accepts only identical repeated filter values. The second
accepts omission only when the following non-whitespace offset is an exact
in-use object offset in the parsed xref. Divergence remains fail-closed.

The runtime's three-argument large-file `read_file` form is not shared by the
`lean_single` route. The accepted implementation therefore uses the common
one-argument dynamic byte buffer and performs an explicit Sounio copy into a
bounded global word buffer. Binary NUL bytes never participate in string-length
semantics, and the complete Sounio SHA-256 guards short reads.

## Negative Evidence

All eight negative results are produced in Sounio and equal `1`:

- complete-file digest sabotage;
- PDF header sabotage;
- duplicate insertion through the xref marking function used by the parser;
- repeated visit through the page-cycle function used by the traversal;
- declared/actual page-count mismatch;
- two filter entries with a conflicting second value;
- selected compressed length above inflater capacity;
- bad Adler-32 mapped through the PDF inflater-error namespace.

A deliberate Python-oracle Loom frame with identical receipt bindings was
denied before interpreter execution, exit code 110.

## Evidence Boundary

This receipt establishes page-tree resolution and byte-exact decompression of
18 selected content streams. It does not establish PDF text interpretation,
font resource resolution, ToUnicode mapping, Apple feature rows, processor or
GPU-family thresholds, observed hardware, instruction equivalence, material
cost, lowering correctness, or performance.

Lean, Koka, C++, and Haskell parity were not opened. External LLM review was
not invoked: this internal byte/document substrate changes no mathematical or
clinical claim and is not an external-facing publication.

## Loom Admission

The local frozen-semantics request used frame `9020`:

```text
stage=2 SOUNIO_EXECUTABLE
action=3 FREEZE_SEMANTICS
language=1 Sounio
role=1 SEMANTIC_AUTHORITY
policy_state=1 available
semantic_write=1
expected_result_write=1
parity_receipt_valid=0
review_promoted=0
exception_and_waiver_fields=0
guardian_fields=0
parent_semantics_sha256=absent
waiver_sha256=absent
```

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `0ca59bed14b4484a46955329cb3acbf410b73ab72cab4d9eacec62da19776708` |
| Frozen semantics | `95de48c5b3372be22047c97e468eaa76fc1079c9983487d277de1195d02445f5` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `3158b1528289dfdcc6bc509bd5631403dbdf73664b8d635e16717eba6ed48e52` |
| Sounio result | `41947f41db6ba3fb1f380a5f6008282144aa19f25384c31842bc7c9f46232232` |

The local runtime remained the fixture-matched realization of frozen Loom
semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

The deliberate Python-oracle frame returned:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

The `action=10 COMMIT` frame at stage `SEMANTICS_FROZEN`, using the same
complete receipt bindings, also returned:

```text
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

The local decision admits the semantic bundle. Canonical Concept-ID and docs
registry acceptance remain with the active Loom owner; this lane does not
bypass that ownership or promote itself to canonical authority.
