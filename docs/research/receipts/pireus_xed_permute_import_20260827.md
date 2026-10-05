<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xed-permute-import-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xed-permute-import-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XED Permutation Import Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Recorded-At-UTC: `2026-08-27`

Completed-Stage: `SEMANTICS_FROZEN`

Next-Stage: `PARITY_OPEN`

Next-Stage-Status: `BLOCKED_PENDING_CONCEPT_REGISTRATION_AND_LOOM_ACCEPTANCE`

## Authority Binding

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

Semantic-Lane-ID: `pireus-xed-permute-import-20260827`

Garden-Seed:
`docs/internal/garden/seeds/2026-08-27-pireus-canonical-corpora.md`

Garden-Seed-SHA256:
`3c7e398d057d8d5eb5a0353ad48a9d0cae0a35af813574ee6742d8b5da30f5c1`

Concept-Contract:
`docs/internal/concepts/pireus-xed-permute-import.md`

Concept-Contract-SHA256:
`daaf378bb3da39eea269324eb948d576dcfc1ddcf435aaa9272507f48870594f`

Importer-Source:
`stdlib/hardware/pireus/xed_import.sio`

Importer-Source-SHA256:
`c65d63a490038d874f9d1ae34458ff44793049eb7ec01bee01981df7974cbeb9`

Executable-Source:
`examples/pireus_xed_permute_import.sio`

Executable-Source-SHA256:
`7831f26752a67c40ef1ea228d0a86167e9fe47abe8db2a156ca7f9779c70c491`

Frozen-Semantics:
`docs/research/pireus_xed_permute_import_semantics.md`

Frozen-Semantics-SHA256:
`5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612`

Canonical-Output-SHA256:
`1dcb0fa54123aa590fdd0c51c0c3d6e810f53e95b5ea25bbe4be0e639aeef460`

## Vendor Input

```text
upstream=https://github.com/intelxed/xed
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
path=datafiles/avx512f/avx512-foundation-isa.xed.txt
bytes=458470
sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
license=Apache-2.0
```

The file was read from the pinned read-only checkout at
`/tmp/pireus-xed-v2026.08.23-20260827`. It was not copied into the repository.

## Toolchain

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
selected_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
sha256_module=stdlib/crypto/sha256.sio
sha256_module_sha256=6c5c6895f2d3b094ea114ee3ba894c535cb12e7822e7c902fbd52771aac7537a
ontology_query_kernel=stdlib/ontology/query.sio
ontology_query_kernel_sha256=e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb
```

The pure-Sounio SHA-256 suite passed the empty string, `abc`, streaming `abc`,
and HMAC test vectors before the corpus result was frozen.

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
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_xed_permute_import.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_xed_permute_import.sio \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_xed_permute_import.sio \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  tests/stdlib/crypto/test_sha256.sio
```

The two complete authority streams were byte-identical and each hashed to the
canonical output digest above.

## Sounio-Produced Result

```text
SOUNIO_AUTHORITY schema=pireus-xed-permute.v0 role=SEMANTIC_AUTHORITY
PIREUS_XED_CORPUS release=v2026.08.23 commit=0bcb6237345c5066726dcc08b3d87928df3b5b26 file=avx512-foundation-isa.xed.txt bytes=458470
 sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038 digest_match=1

PIREUS_XED_FORMS total=8
 vpermi2pd=2
 vpermpd=4
 vpermt2pd=2

PIREUS_XED_ACCESS write=4
 read_write=4
 mask_read=8

PIREUS_XED_SELECTOR raw_unassigned=4
 uimm8=2
 register_index=2
 semantic_role_assignments=0
PIREUS_XED_STORAGE register=4
 memory=4

PIREUS_XED_OPERANDS total=32
 read=24
 dest_zf64=8
 register_read_zf64=10
 memory_read_f64=4
 immediate_read=2

PIREUS_XED_ONTOLOGY triples=246
 forms=8
 family_counts=2
,4
,2

PIREUS_XED_NEGATIVE hash=1
 unknown_field=1
 missing_field=1

PIREUS_XED_SUMMARY failures=0
```

The line breaks around integer fields are emitted by the selected Sounio
runtime's `print_int`; no formatter rewrote the hashed stream.

## Validation Classification

| Check | Result | Classification |
| --- | --- | --- |
| `lean_single` typecheck | exit 0 | Sounio compiler path accepted |
| `lean_single` execution, twice | exit 0, identical | semantic-authority result |
| pure-Sounio SHA-256 vectors | 4 passed, 0 failed | digest implementation witness |
| rebuilt ontology checker | parsed, witness 0 | rebuilt/current-source checker accepted |
| default Madaros fallback | rejected | imported-struct-field compiler divergence |
| combined ontology wrapper | `mixed/unknown` | not promoted to unanimous pass |
| hash mutation | refused | negative witness |
| unknown selected field | refused | negative witness |
| missing selected field | refused | negative witness |

The Madaros disagreement produced no alternative expected result. The
authoritative stream came only from the Sounio executable on the explicitly
identified Sounio compiler path.

## Prohibited-Oracles Gate

The importer, executable, check, run, output comparison, and freeze path invoked
no Python, Rust, Node, Ruby, `awk`, or `bc`. Shell transported commands and
calculated receipt hashes; it did not classify vendor records or produce
expected counts.

The deliberate Python-oracle pre-execution denial already recorded for Pireus
v0.1 remains the language-authority control. No prohibited interpreter was
launched here.

## Evidence Boundary

This receipt establishes exactly eight pinned vendor records and their raw
Pireus projection. It does not establish semantic operand roles for
`VPERMI2PD` or `VPERMT2PD`, executable support on a Darwin node, instruction
behavior, encodings, cost, lowering correctness, or a Cayley-Dickson speedup.

Lean, Koka, C++, Haskell, and external LLM parity/review were not invoked.
External LLM offload reviews invoked: none; this internal importer receipt adds
no mathematical, clinical, or external-facing claim.

`PARITY_OPEN` requires registration of proposed Concept-ID
`SOUNIO-PIREUS-XED-PERMUTE-IMPORT` and executable acceptance by the active Loom
owner.

## Loom Admission (Append-Only)

Recorded UTC: `2026-08-27T05:15:05Z`

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

The complete frame bound these receipt fields:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `7831f26752a67c40ef1ea228d0a86167e9fe47abe8db2a156ca7f9779c70c491` |
| Frozen semantics | `5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `ebd5727e9294b8cc656965c5007c6df65c1deea1357bed3827b72e2f27788abb` |
| Sounio result | `1dcb0fa54123aa590fdd0c51c0c3d6e810f53e95b5ea25bbe4be0e639aeef460` |

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
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_xed_permute_import.sio /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
```

The hardware record is identical to the seven-line record printed above and
hashes to the value bound by the frame.

The operational runtime remained the fixture-matched realization of the frozen
Sounio Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
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
`SEMANTICS_FROZEN`; it does not register the Concept-ID or open parity.
