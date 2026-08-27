<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-a64-tbl-material-parity-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-a64-tbl-material-parity-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple A64 TBL Material Parity Receipt

Receipt-Schema: `sounio-material-parity-receipt.v1`

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`

Producer-Language: `C++`

Producer-Role: `MATERIAL_PARITY`

Semantic-Authority-Language: `Sounio`

Stage: `PARITY_OPEN`

Parity-Receipt-Valid: `true`

Claim-Ready: `false`

## Boundary

This receipt compares one A64 `TBL` realization on an Apple M5 Max with the
expected coordinate map and candidate digest born earlier in Sounio. C++ did
not create or revise the selector, payload-address rules, digest, unresolved
operation boundary, or expected result.

The exact frozen parent is:

```text
sounio_source=stdlib/hardware/pireus/apple_a64_tbl_lowering.sio
sounio_source_sha256=79c2e859ffe81f3add1ebb36608a5995672c10a5c1645ec4500a03fcd9bcd031
frozen_semantics=docs/research/pireus_apple_a64_tbl_lowering_semantics.md
frozen_semantics_sha256=377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6
frozen_sounio_result_sha256=d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81
```

The parity implementation is:

```text
tools/pireus/material_sha256.hpp sha256=ae54c8f455d5ef057f182212aacd466bdf5e014898872706e80e51f6b16e7782
tools/pireus/apple_a64_tbl_material_parity.cpp sha256=7bb640be1093b3add99961c43d7c53da276cceacef70b7ee9e2002c961c5d66e
scripts/ci/pireus_canonical_target_material_parity.sh sha256=d66b6d7b4b28216d0babfc5f8ad90d58cd38f870a6fcc4e3bfc70033707239c3
scripts/ci/pireus_apple_a64_tbl_material_parity.sh sha256=6c579875152120ecaf81094f792852028f28bc228a16d2aca70db0e1d5e469a9
```

## Material Result

The probe fills the two 64-bit outputs of each 128-bit vector with one
four-register `vqtbl4q_u8` operation. It derives the 16 byte controls from the
frozen XOR coordinates, observes which unique payload arrived, and recomputes
the frozen Sounio SHA-256 transcript from those observed sources.

```text
displacements=16
logical_cells=256
matched_cells=256
mismatched_cells=0
in_domain_source_cells=256
bijective_displacements=16
byte_control_cells=2048
matched_byte_controls=2048
max_control=63
out_of_range_controls=0
abstract_tbl_applications=128
symbolic_payload_bits=16384
matched_payload_bits=16384
same_source_group_per_output=true
candidate_digest=472477255:3903797350:1348128039:308036239:1349218781:3920188038:1317640714:1357523880
mutation_mismatching_cells=1
result=PASS
result_lines=30
result_bytes=1012
result_sha256=bd64bc56037a64a93c0136fa29a6ff1a294e8b84be549a5a4abfeaaf81a2e700
```

The emitted assembly has one static `tbl.16b` site inside the material loop:

```text
tbl.16b v2, { v3, v4, v5, v6 }, v2
```

That site is dynamically reached for the 128 abstract applications. This is a
fact about this source and Apple clang build, not a minimum instruction count,
latency model, or generic backend claim.

## Hardware And Toolchain

```text
canonical_target=apple_silicon
tailnet_identity=sounio-language-macbook
hostname=Sounio-Language-MacBook
os=Darwin 27.0.0
architecture=arm64
model=Mac17,7
cpu=Apple M5 Max
target=J714c
hardware_sha256=49702cf6d0b079bf52bf26f98f377266e41d4ce232fea99eb80c30d6554dbc28

compiler=Apple clang version 21.0.0 (clang-2100.3.27.1)
compiler_target=arm64-apple-darwin27.0.0
xcode=27.0
xcode_build=27A5228h
toolchain_sha256=2e20f3f44c17d6fc4c1e58b26c38cf3af1ea2df887778d0aa723e6ddfe4b72e1
command_sha256=e290c25bf7cf3d5c47d3c255d11ab89d8a5eba775a63685fbe0b981cfd76bff5
```

Emitted artifacts reproduced across the manual run and two full gate runs:

```text
binary_sha256=299a41090348d47b518929af6dab8137ada9032a046def3e8052a5d721c6fd71
assembly_sha256=f3c3216faab1809b20e419c4ed345eaf658e892a5f1e6d4545fc6d095d699f76
objdump_sha256=0a05d0ba5562260c062c6338c4b9818df9cc711d0bcbc4f93ef536c77617fab8
reproduced_builds=3
```

The durable command is:

```bash
scripts/ci/pireus_apple_a64_tbl_material_parity.sh
```

## Loom Decisions

No C++ compilation or target write occurred before `PARITY_EXECUTE` returned
`next_stage=PARITY_OPEN`.

```text
write_frame_sha256=d2df3c3c21add1efb21fbf5510365e9fbc00798288bc3423de0bedc307923586
parity_open_frame_sha256=e15e5827c7ab13c86b6f0623cc309e4520fa9b3b6f4511dfb8ebde56b213e400
parity_open_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
receipt_seal_frame_sha256=6e0c0d9317ab24c6cada927df7b0f5fb622974b72af39cf8e29fea96191e34f5
receipt_seal_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
commit_frame_sha256=6a81a9e41e3469f6de2bbf43a001e015ffbb56ee80da6a96571415b97953d8c6
commit_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW
next_stage=PARITY_OPEN
```

The shared deliberate Python request was refused with `code=110` before any
interpreter launch:

```text
python_frame_sha256=7b89a9e7d1d8399dc13d60bbfa97d8c9e8da7d437f7b1f546e64cfb18aa90004
python_decision_sha256=42a2eba7ea7889f7526d1e452196003debe55eb388c5854f6d70cc69bdcf8ea4
interpreter_launch_count=0
```

The T560 key stayed mounted at its source host and was used only for SSH
authentication. It was neither printed nor copied.

## Closed Claims

This receipt does not resolve the four non-selector operation nodes, prove an
exact reduction tree, establish cross-ISA equivalence, measure performance, or
promote a cost model. External LLM output remains review-only. Rust was not
used. `CLAIM_READY` remains false.

Raw compact evidence is in
`docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt`.
