<!-- docs:meta
topic_id: repo.docs.research.receipts.sounio-pdf-text-projection-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.sounio-pdf-text-projection-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio PDF Text Projection Receipt

**Date:** 2026-08-27
**Concept-ID:** `SOUNIO-PDF-TEXT-PROJECTION`
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

The source-lane Garden artifact existed and received local Loom `GARDEN` and
`COMMIT` ALLOW decisions before the first implementation. Its pre-governance
SHA-256 is
`7177d16b71a391811731df3a8cef2b7cd3656b31830e624194267c3e6c90077e`.
The canonical authority owner materialized that Garden first at commit
`a897f96ee446d47496d562e930ea076be0076d7c`; governance-only normalization
produced SHA-256
`6aa909e3de1849ee7b08b43353c94927712d2f71be9a0fdcd1bb66019739c385`.
`PARITY_OPEN=false` and `CLAIM_READY=false`.

## Authority Source

| File | SHA-256 |
| --- | --- |
| `stdlib/crypto/sha256.sio` | `dae5dd7e0334bf0e5c65ab07fa732c2e167b2e9c28acaace96bae188bc29d0d5` |
| `stdlib/compress/inflate.sio` | `2d788a553d3b0255c67789e1c8c2b2fc997762a76212db9613eaa1d8a5f762e3` |
| `stdlib/document/pdf_flatedecode.sio` | `baae6d9fc2f9a36ad607fb676c9c646c5b27a27d5ead68b21fa2ed33e2ab14e3` |
| `stdlib/document/pdf_text.sio` | `486db062799a89cd216cc3c2375da9e579d3d5b8a9543c27cd191c6413a45278` |
| `examples/sounio_pdf_text_projection.sio` | `fd70acacc4a1afd0a257e223943e1935a88263bdcb0de633c0f09c408bc4061b` |
| `examples/sounio_pdf_text_projection_negatives.sio` | `e090b441fa6f7fda788c2fa7054f1d3ac3a4f43555f182cef9a0c393f07f241c` |
| Garden seed, source lane | `7177d16b71a391811731df3a8cef2b7cd3656b31830e624194267c3e6c90077e` |
| Garden seed, canonical governance projection | `6aa909e3de1849ee7b08b43353c94927712d2f71be9a0fdcd1bb66019739c385` |
| Concept contract | `65d5df27f79cacd332554b9681dba6eee44d7f12adef7c80459f7c950940fafe` |
| Semantics | `ae495077aece5855e21810590f937c58bea743b1655d76e5c52acc24b1c6e17e` |

After local Loom freeze and commit admission, the five-file Sounio source slice
was committed alone in the source lane as
`601d150f8858a41af1852074b07d968fa4c07ef5`. The canonical authority owner
accepted the byte-identical source slice as
`781109c2db1cf7276346aaa51716bfa658f3556e`, whose direct parent is the
canonical Garden commit above.

## Corpus Receipts

```text
apple_pdf_url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
apple_pdf_bytes=3041713
apple_pdf_sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
apple_roman_url=https://www.unicode.org/Public/MAPPINGS/VENDORS/APPLE/ROMAN.TXT
apple_roman_sha256=18e571645be895e9553ed5c842ea8f65f9c5d3c9ccb43e66e0c33a132ed0d721
content_aggregate_sha256=22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526
```

The live URLs are mutable. New bytes require a new Garden pin.

## Toolchain

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
toolchain_record_sha256=2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e
```

## Hardware

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
hardware_record_sha256=fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0
```

## Commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/sounio_pdf_text_projection.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/sounio_pdf_text_projection.sio \
  /tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/sounio_pdf_text_projection_negatives.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/sounio_pdf_text_projection_negatives.sio
/tmp/pireus-v01-ontology-validation-souc check \
  examples/sounio_pdf_text_projection.sio
/tmp/pireus-v01-ontology-validation-souc run \
  examples/sounio_pdf_text_projection.sio -- \
  /tmp/pireus-apple-metal-20260521/Metal-Feature-Set-Tables.pdf
```

Both direct corpus runs were byte-identical. Both rebuilt checks reported
`verdict=ok`, `provenance=rebuilt_direct`, and unanimous driver/fallback
resolution. The rebuilt corpus run's 737-line program-output suffix was
byte-identical to the direct authority stream. No raw ELF, host PDF parser,
host decompressor, Python, or Rust was invoked.

## Sounio Result

```text
PDF_TEXT_RESOURCES error=0 pages=18 direct=0 indirect=18 inherited=0 depth=0 bindings=99 fonts=10 tounicode=3 without_tounicode=7 encoding=10 descendants=0
PDF_TEXT_OPERATORS error=0 pages=18 bytes=1095453 comments=0 names=16451 numbers=125383 literals=25062 hex=0 array_opens=2397 array_closes=2397 dict_opens=5319 dict_closes=5319 operators=77986 distinct=27 BT=3193 Tm=3190 Tf=3236 TJ=2397 Tj=839 Tc=1148 ET=3193
PDF_TEXT_GLYPHS error=0 pages=18 objects=3193 fonts=3236 matrices=3190 spacing=1148 strings=25032 arrays=2397 adjustments=21822 codes=35978 non_ascii=78 stack=6
PDF_TEXT_UNICODE error=0 fonts=10 cmaps=3 codespaces=3 bfchar=0 bfrange=4 entries=4 glyphs=35978 macroman=35904 cmap=74 non_ascii=152 private_use=2
PDF_TEXT_PROJECTION error=0 pages=18 objects=3193 nonempty=3190 empty=3 matrixless=3 multiple_matrix=0 shows=3236 scalars=35978 graphics_depth=2
PDF_TEXT_LINES error=0 pages=18 objects=3190 lines=730 tolerance=5000000 max_spread=5000000 min_gap=6000000 ambiguous=0
PDF_TEXT_DIGEST error=0 bytes=891830 sha256=AEBDEA5034DC20201EDF555BBB257E2971EB86127BE602E630BE1135564F93F8 expected=1
```

The complete stream contains all 730 projected lines and hashes externally,
as receipt material, to:

```text
5766e0719113274ba55da184cc7879e2e1a52a74def87998ec12488c230b0fea
```

The semantic digest above was produced and checked inside Sounio. The external
stream hash did not define the expected result.

## Negative Evidence

The direct and rebuilt negative witnesses both report:

```text
PDF_TEXT_NEGATIVES cases=7
failures=0
```

The cases cover propagated parser error, deterministic page/y/x ordering,
inclusive five-point grouping, six-point separation, page boundary, ignored
empty object, repeated-run digest equality, scalar sabotage changing the
digest, and inconsistent object-to-line mapping refusal. Some cases exercise
more than one invariant.

## Authority Boundary

This receipt establishes document text scalars, their font provenance,
fixed-point object origins, deterministic line membership, and the complete
projection digest. It does not establish table cells, Apple-family semantics,
feature inheritance, hardware observations, instruction equivalence,
lowering, material cost, or performance.

Lean, Koka, C++, and Haskell parity were not opened. External LLM review was
not invoked: this is an internal document substrate, not a mathematical or
clinical claim and not an external-facing publication.

## Loom Admission

The local freeze frame used stage `SOUNIO_EXECUTABLE`, action
`FREEZE_SEMANTICS`, language Sounio, role `SEMANTIC_AUTHORITY`, available
policy, both authority-write flags, and complete receipt bindings:

| Binding | SHA-256 |
| --- | --- |
| Sounio executable | `fd70acacc4a1afd0a257e223943e1935a88263bdcb0de633c0f09c408bc4061b` |
| Frozen semantics | `ae495077aece5855e21810590f937c58bea743b1655d76e5c52acc24b1c6e17e` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `19243e93542eced071d677d71609b6a11ce6b00781e7cff6cfddfe3dc1cf2159` |
| Sounio result | `5766e0719113274ba55da184cc7879e2e1a52a74def87998ec12488c230b0fea` |

The Sounio Loom runtime remained the fixture-matched language authority:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
freeze_decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
commit_decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

A deliberate Python parity-oracle pre-execution frame bound the same frozen
source, semantics, toolchain, hardware, and command. It was denied before any
interpreter execution:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
```

Local Loom has admitted the proposed freeze and source commit. Canonical
`SEMANTICS_FROZEN` registration still requires the authority owner's
transaction; `PARITY_OPEN=false` and `CLAIM_READY=false` remain unchanged.
