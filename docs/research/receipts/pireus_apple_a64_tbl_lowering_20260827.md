<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-a64-tbl-lowering-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-a64-tbl-lowering-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple A64 TBL XOR Lowering Receipt

Date: 2026-08-27
Schema: `sounio-semantic-authority-receipt.v1`
Producer language: `Sounio`
Language role: `SEMANTIC_AUTHORITY`
Completed stage: `SEMANTICS_FROZEN`
`PARITY_OPEN=false`
`CLAIM_READY=false`

## Transition

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

The expected result was born in the Sounio executable identified below. Arm
XML, the bootstrap compiler, shell transport, the material machines, and review
models did not define or confirm it.

## Sounio Source

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `stdlib/hardware/pireus/apple_a64_tbl_lowering.sio` | 49135 | `79c2e859ffe81f3add1ebb36608a5995672c10a5c1645ec4500a03fcd9bcd031` |
| `examples/pireus_apple_a64_tbl_lowering.sio` | 7588 | `0fa666dd3c07e3d261b11a49e08c1cba3e1822f2bfbcff2bc73a71d610711c5b` |
| `docs/research/pireus_apple_a64_tbl_lowering_semantics.md` | 7184 | `377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6` |

The source manifest is SHA-256 over the two standard `sha256sum` records, in
the table order above:

```text
source_manifest_sha256=03c0a315a579e568b14876dc06a116565599598f4f6cf7ac4a1bb6221a3d1e09
semantics_sha256=377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6
```

## Parent Closure

| Artifact | SHA-256 |
| --- | --- |
| Apple Garden | `85c03626b396563ee69460dacad6faa9a7cb8719ad661aca26692e0e099df5a0` |
| lowering source | `7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb` |
| lowering semantics | `9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970` |
| lowering receipt | `daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346` |
| AARCHMRS source | `ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b` |
| AARCHMRS semantics | `ed66cc2e2fe27ce06842c1ef2091e2f482b8bcb2d4b84e4e649361ca957b7b14` |
| AARCHMRS receipt | `cd64c91c330c9a81e554408a10de4bccbdf9984395ec049c48dc99148aa11934` |
| `tbl_advsimd.xml` | `48ef32ed67b9824ba39eb58518faec196472c3a574cf1bbe1f3a494811a6cbbe` |
| `tbx_advsimd.xml` | `fa21f8c0784ec327ca9089552d22b55e0eb4b9dd6e0a2eeb078eeed0e203ca79` |
| `notice.xml` | `7f6e2780187dc8eb12b53d97eb435be19597b1af256a84fb44d4b5bd41846747` |

```text
parent_record_sha256=a479ff676865104174ed6f34972724680f5024db7093ac5e4a0a64d9afb16f6f
```

The record is SHA-256 over the ordered `sha256sum` records for those ten
artifacts. The XML files are pinned inputs, not semantic producers.

## Toolchain And Hardware

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
toolchain_record_sha256=2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e

os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
hardware_record_sha256=b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0
```

This x86_64 machine transported and ran the Sounio authority executable. It is
not an Apple material observation.

## Command And Result

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_apple_a64_tbl_lowering.sio \
  docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-apple-silicon.md \
  stdlib/hardware/pireus/xor_lowering_legality.sio \
  docs/research/pireus_xor_lowering_legality_semantics.md \
  docs/research/receipts/pireus_xor_lowering_legality_20260827.md \
  stdlib/hardware/pireus/aarchmrs_import.sio \
  docs/research/pireus_aarchmrs_tbl_import_semantics.md \
  docs/research/receipts/pireus_aarchmrs_tbl_import_20260827.md \
  /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/tbl_advsimd.xml \
  /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/tbx_advsimd.xml \
  /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/notice.xml
```

```text
command_sha256=2bab9bfea153113dccf161dff5b7f8ef80a146bd62bb0d60a25611d8432a756a
result_sha256=d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81
byte_identical_repetitions=2
```

The Sounio stream reports:

```text
table_registers=4 table_bytes=64 output_bits=128
logical_cells=256 matched_cells=256
in_domain_sources=256 out_of_domain_sources=0
bijective_displacements=16 dimension_matches_bits=1
byte_controls=2048 matched_byte_controls=2048
payload_bits=16384 matched_payload_bits=16384
max_control=63 out_of_range=0
abstract_tbl_applications=128
xor_permute_candidate=1
unresolved_other_nodes=4 exact_tree_refused=1
negative_witnesses=15/15
canonical=1 observed=0 material_receipts=0 compiler_emissions=0 costs=0
parity_open=0 claim_ready=0 error=0 failures=0
candidate_digest=472477255:3903797350:1348128039:308036239:1349218781:3920188038:1317640714:1357523880
```

The abstract application count includes no claim about emitted instructions,
minimum cost, throughput, or Apple hardware.

## Loom Admission

```text
precheck_frame_sha256=854ccf67c9a276ca4aa11b2996325855622b3730e4c8444f11175e67c57f1bae
prerun_frame_sha256=a457d7859b3336701d538e07d876ef15d97fedcafd3daf02f2fe22f85262176f
freeze_frame_sha256=cec8d7c4b299b3e88d911b699fc09bcb3d1483b1d7c13e651c1de2ecb098d1cd
freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
commit_frame_sha256=cd355c5a2b524b69b3fbdd4d55dcc3d1e083684808f0194c54fde6e7a5de2992
commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
```

The freeze used `schema=9020`, `stage=2`, `action=3`, `language=1`, `role=1`,
`policy_state=1`, `semantic_write=1`, and `expected_result_write=1`. Parity,
review-promotion, waiver, and transitional-guardian fields were zero.

## Review-Only Findings

```text
providers=xai/grok-4.5,zai/GLM-5.2
role=REVIEW_ONLY
raw_first=/tmp/llm-offload-IRDMx1/
raw_followup=/tmp/llm-offload-n0J35c/
raw_final=/tmp/llm-offload-Ig7UrK/
```

The first review found that the original candidate did not explicitly require
power-of-two dimension closure, in-domain XOR sources, or per-displacement
bijection. Sounio was changed to check all three. A later local audit, not a
model, found that the authority predicate still protected only the negative
surface and that payload bits were counted without absolute symbolic source
addresses. The final Sounio source makes the same 15-input authority predicate
decide the live result and hashes both expected and reconstructed bit
coordinates. The final review found no wrong XOR, size, control-bound, or
negative-case arithmetic; it requested only tighter wording around derived
control closure and abstract source-group selection, which the paired semantics
now supplies. No provider created or confirmed any count, digest, result, or
authority decision.

## Negative Enforcement

The paired Garden deliberately presented a Python expected-result producer to
the Loom guardian. It was denied before interpreter launch:

```text
python_oracle_frame_sha256=59b5ccd7f6d649d3f40fcb9b0b85a14c76c8d25400795d2f326281bee30386a6
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language
interpreter_launch_count=0
```

Rust was not used. No disposable-language oracle supplied expected values.

## Canonical Compiler Diagnostic

The public wrapper resolved to Madaros v0.80.0 and failed closed before a
result stream:

```text
canonical_toolchain_sha256=bfcc2599684a6c7fd72ec1c84b66c287f29a1f7ae55396cb889feb59e97d3bd3
canonical_command_sha256=6420e5c0c85dc35b1e80f4af347ee955d9e1ca4218c597572a51bb38fad25a04
canonical_action2_frame_sha256=f3b91268a463a859a37903754bfd5b0401a045d197e4c5e0def2e4c3e848ffe5
canonical_diagnostic_sha256=3d3330283faa4b49b80bcea694504b71a4db5ac6ced1916cfcec347f3def0d58
exit_code=1 verdict=1 E012_count=629 E008_count=1
```

The dominant nameless `E012` cascade crosses frozen parents and prevents a
source-local diagnosis. It is a canonical compiler blocker, not evidence
against the accepted `lean_single` Sounio authority stream.

## Claim Boundary

This receipt establishes a structurally derived A64 `TBL` candidate for the
frozen 16-lane `XOR_PERMUTE` node. It establishes no Apple Silicon observation,
compiler emission, material parity, whole-operation lowering, exact reduction
tree, instruction minimum, cost, performance, or cross-ISA parity.

Unresolved blocker contract:

```text
Blocker-ID=BLOCKER-PIREUS-CANONICAL-MADAROS-FIELD-RESOLUTION-20260827
severity=P1
class=build/bootstrap-path
evidence=executable
owner=Madaros compiler lane
acceptance_gate=./bin/souc check examples/pireus_apple_a64_tbl_lowering.sio exits 0 without nameless field-resolution cascade
next_action=isolate the earliest valid named E012 in the canonical resolver and repair field/type propagation without changing frozen Pireus semantics
```

Legacy parent artifacts were intentionally retained unchanged.
