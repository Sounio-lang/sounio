<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-aarchmrs-tbl-import-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-aarchmrs-tbl-import-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus AARCHMRS TBL/TBX Import Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Recorded-At-UTC: `2026-08-27`

Completed-Stage: `SEMANTICS_FROZEN`

Next-Stage: `PARITY_OPEN`

Next-Stage-Status: `BLOCKED_PENDING_CONCEPT_REGISTRATION_AND_LOOM_ACCEPTANCE`

## Authority Binding

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

Semantic-Lane-ID: `pireus-aarchmrs-tbl-import-20260827`

Garden-Seed:
`docs/internal/garden/seeds/2026-08-27-pireus-aarchmrs-open-corpus.md`

Garden-Seed-SHA256:
`293711308ae794f9846b3c9986c0f6b57373c50083c8a331c02a36ad63f8f385`

Garden-Commit:
`19e3019da30d40640b8697d0c99b0032a322ce55`

Concept-Contract:
`docs/internal/concepts/pireus-aarchmrs-tbl-import.md`

Concept-Contract-SHA256:
`5ac1a6dae809511d0579464630813b4bca8bb067ab43346587fda223ff4f9202`

Importer-Source:
`stdlib/hardware/pireus/aarchmrs_import.sio`

Importer-Source-SHA256:
`ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b`

Executable-Source:
`examples/pireus_aarchmrs_tbl_import.sio`

Executable-Source-SHA256:
`739c87486cb0187b4bb5392b3e2931a2fbd14355213115b3c4854218a8dada21`

Frozen-Semantics:
`docs/research/pireus_aarchmrs_tbl_import_semantics.md`

Frozen-Semantics-SHA256:
`de4408f7a09d6e224b93ce0b5decdacdae26437b458e9a16b381fd3aa473daa0`

Canonical-Output-SHA256:
`80e659203cbd9601cdb3d5eccfa6614f8b189afa125f075206a16dd777565db8`

## Vendor Corpus

Official package URL:

```text
https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-OS-Machine-Readable-Data/AARCHMRS_BSD/AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12.tar.gz
```

Pinned evidence:

```text
archive_bytes=5371270
archive_sha256=4dc5da62a5c856d7b1086b895075f54807f821ea21a333049cb0f40f9479cecc
instructions_bytes=115312065
instructions_sha256=bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b
architecture=vFATAp1-A
build=518
ref=2025-12_rel
schema=2.7.4
license=BSD-3-Clause
```

The archive, README, notice, and three top-level JSON files were retained in
`/tmp/pireus-aarchmrs-2025-12` for this run and were not added to Git.

## Byte Transport

The native bootstrap `read_file` limit required a mechanical split below 1
MiB:

```bash
split -d -a 3 --additional-suffix=.part -b 1000000 \
  /tmp/pireus-aarchmrs-2025-12/Instructions.json \
  /tmp/pireus-aarchmrs-2025-12/chunks/part-
```

This produced 116 chunks: 115 at 1,000,000 bytes and one at 312,065 bytes.
Concatenation outside Sounio was used only as a transport-integrity observation
and matched the pinned file digest. The Sounio executable independently read
the chunks in order, reconstructed the total byte count, computed the complete
SHA-256, parsed JSON, selected forms, materialized the ontology, and produced
the result.

## Toolchain

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
execution_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
ontology_query_kernel=stdlib/ontology/query.sio
ontology_query_kernel_sha256=e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb
sha256_kernel=stdlib/crypto/sha256.sio
sha256_kernel_sha256=6c5c6895f2d3b094ea114ee3ba894c535cb12e7822e7c902fbd52771aac7537a
```

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

## Commands

```bash
/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_aarchmrs_tbl_import.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_aarchmrs_tbl_import.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_aarchmrs_tbl_import.sio \
  /tmp/pireus-aarchmrs-2025-12/chunks/part-
```

The complete Sounio authority stream was executed repeatedly and remained
byte-identical.

## Sounio-Produced Result

```text
SOUNIO_AUTHORITY schema=pireus-aarchmrs-tbl.v0 role=SEMANTIC_AUTHORITY
PIREUS_ARM_CORPUS archive=AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12.tar.gz file=Instructions.json bytes=115312065
 chunks=116
 error=0
 sha256=bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b digest_match=1

PIREUS_ARM_META architecture=vFATAp1-A build=518 ref=2025-12_rel schema=2.7.4
PIREUS_ARM_JSON objects=615423
 strings=2994616
 max_depth=24
 instruction_objects=6569

PIREUS_ARM_FORMS total=11
 tbl=6
 tbx=5

PIREUS_ARM_OPERATION tbl_advsimd=4
 tbx_advsimd=4
 tbl_z_zz=2
 tbx_z_zz=1

PIREUS_ARM_VECTOR fixed_128=8
 scalable=3

PIREUS_ARM_TABLE_REGISTERS one=4
 two=3
 three=2
 four=2

PIREUS_ARM_ONTOLOGY triples=259
 forms=11

PIREUS_ARM_NEGATIVE duplicate_field=1
 selected_shape=1
 malformed_json=1
 digest=1

PIREUS_ARM_SEMANTIC_ROLE_ASSIGNMENTS count=0
PIREUS_ARM_SUMMARY failures=0

```

`print_int` in the selected Sounio runtime terminates its integer output with an
`LF`; the leading-space continuation lines above are therefore part of the
hashed stream.

## Validation Classification

| Check | Result | Classification |
| --- | --- | --- |
| rebuilt ontology check | unanimous, `rebuilt_direct` | current-source checker accepted |
| `lean_single` check | exit 0, no diagnostics | Sounio compiler path accepted |
| complete corpus run | exit 0 | semantic-authority result |
| repeated complete stream | identical | deterministic result |
| complete SHA and length | match | ordered byte reconstruction accepted |
| duplicate selected field | rejected | fail-closed negative |
| unknown selected shape | rejected | fail-closed negative |
| malformed JSON | rejected | fail-closed negative |
| empty-stream digest | rejected | fail-closed negative |

## Prohibited-Oracles Gate

No Python, Rust, Node, Ruby, `awk`, or `bc` was used to parse the JSON, select
forms, compute expected counts, materialize the ontology, or produce the
canonical result. Shell, `curl`, `tar`, `split`, `cat`, and `sha256sum`
transported or inspected material bytes only. Their observations did not define
the Sounio projection or expected result.

No parity language or external LLM review was invoked. External LLM offload
reviews invoked: none; this internal importer receipt adds no mathematical,
clinical, or external-facing claim.

## Evidence Boundary

The receipt establishes the exact corpus, complete-stream parser, selected
record shapes, structural counts, and ontology query agreement only. It does
not establish instruction equivalence, Apple/DGX availability, operand semantic
roles, material cost, backend selection, or Cayley-Dickson lowering.

`PARITY_OPEN` requires registration of proposed Concept-ID
`SOUNIO-PIREUS-AARCHMRS-IMPORT` and executable acceptance by the active Loom
owner.

## Loom Admission (Append-Only)

Recorded UTC: `2026-08-27T05:52:11Z`

The frozen receipt was submitted to Loom frame `9020` with:

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

The complete 82-field frame bound:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `739c87486cb0187b4bb5392b3e2931a2fbd14355213115b3c4854218a8dada21` |
| Frozen semantics | `de4408f7a09d6e224b93ce0b5decdacdae26437b458e9a16b381fd3aa473daa0` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `055e5a497a9c57dd86a8d499ce7ed295669ae386fa3992788ee3276ec195ad9d` |
| Sounio result | `80e659203cbd9601cdb3d5eccfa6614f8b189afa125f075206a16dd777565db8` |

The toolchain record was this exact UTF-8 text with one final `LF`:

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
```

The command record was this exact UTF-8 text with one final `LF`:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_aarchmrs_tbl_import.sio /tmp/pireus-aarchmrs-2025-12/chunks/part-
```

The hardware record is identical to the seven-line record printed above. The
operational runtime remained the fixture-matched realization of the frozen
Sounio Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

A deliberate Python-oracle frame with the same receipt bindings was denied
before an interpreter or requested effect ran:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

Both decision hashes include the final `LF`. This admission accepts
`SEMANTICS_FROZEN`; it does not register the Concept-ID, open parity, or promote
a claim.
