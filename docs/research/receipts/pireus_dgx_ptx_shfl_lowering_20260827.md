<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-dgx-ptx-shfl-lowering-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-dgx-ptx-shfl-lowering-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus DGX PTX SHFL XOR Lowering Receipt

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

The expected result was born in the Sounio executable identified below. PTX
HTML, the bootstrap compiler, shell transport, DGX machines, and review models
did not define or confirm it.

## Sounio Source

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `stdlib/hardware/pireus/dgx_ptx_shfl_lowering.sio` | 41674 | `4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404` |
| `examples/pireus_dgx_ptx_shfl_lowering.sio` | 8061 | `976866431a13fd7ea833ecd3f6fa81983573a389e32bb2e1e4a779d99ac73dd8` |
| `docs/research/pireus_dgx_ptx_shfl_lowering_semantics.md` | 6285 | `a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336` |

```text
source_manifest_sha256=a5e3d1c25f0c3745ad8d0e78f96ab0c2bd5ea0cd68a71ba11453ea06f9c1d733
semantics_sha256=a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336
```

The source manifest is SHA-256 over the two standard `sha256sum` records, in
the table order above.

## Parent Closure

| Artifact | SHA-256 |
| --- | --- |
| DGX Garden | `c084d7a6ebe728931371b60af0c41d3ca1dad7198fc8bacdaf8ce9c491f884a2` |
| lowering source | `7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb` |
| lowering semantics | `9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970` |
| lowering receipt | `daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346` |
| PTX source | `ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba` |
| PTX semantics | `1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697` |
| PTX receipt | `e68f6edacfa85c48cd3cb51ab4929975a187174b0b1ab980a2c0f0868f5f38fa` |
| `part-000.part` | `6590d9e3ba60e55e3f0d2cb7f1d83cd3d5735abb7526517709533cfa3093ee91` |
| `part-001.part` | `3e080ba7e8e556e29aed0c69ef818de39cb48b67ee6b442bd018dcb6ffa9bd8d` |
| `part-002.part` | `9b120aa8ca72eabc4db120a19486292c5b4715f8f29a31c9b284e7651195ae91` |
| `part-003.part` | `0ae51edf20e03d37e77f826350b3d63e726fa2e55cebc6a60e00ce00e733292d` |

```text
ptx_corpus_bytes=3428895
parent_record_sha256=1de516eb2e819cac86385f86123e26db8e31e059404f59d29e786d2073a3bc71
```

The record is SHA-256 over the ordered `sha256sum` records for those eleven
artifacts. The PTX corpus is a pinned input, not a semantic producer.

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
not a DGX material observation.

## Command And Result

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_dgx_ptx_shfl_lowering.sio \
  docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-dgx.md \
  stdlib/hardware/pireus/xor_lowering_legality.sio \
  docs/research/pireus_xor_lowering_legality_semantics.md \
  docs/research/receipts/pireus_xor_lowering_legality_20260827.md \
  stdlib/hardware/pireus/ptx_import.sio \
  docs/research/pireus_ptx_prmt_import_semantics.md \
  docs/research/receipts/pireus_ptx_prmt_import_20260827.md \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part-
```

```text
command_sha256=3ef15a2030f83773ee948598b37727743fb73b639b97e556388090e8defb40b0
result_sha256=495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e
byte_identical_repetitions=2
```

The Sounio stream reports:

```text
warp_lane_bits=5 warp_lanes=32 admitted_lanes=16
membermask_bits=32 membermask_low=65535
cval=15 segmask=0 packed_c=15
f64_components=2
logical_cells=256 matched_cells=256
in_range_cells=256 own_lane_fallback_cells=0
active_source_cells=256 member_source_cells=256 max_source_lane=15
payload_component_cells=512 symbolic_payload_bits=16384 matched_payload_bits=16384
symbolic_payload_address_preservation=1
abstract_shfl_sync_instructions=32
identity_shfl_sync_instructions=2
nontrivial_shfl_sync_instructions=30
xor_permute_candidate=1
unresolved_other_nodes=4 exact_tree_refused=1
negative_witnesses=25/25
canonical=1 observed=0 ptx_sass_equivalences=0 material_receipts=0
compiler_emissions=0 costs=0 parity_open=0 claim_ready=0
error=0 failures=0
candidate_digest=1996463492:2773712531:2232409634:959326894:3198512505:1781073490:2005131219:1992148529
```

The 32/2/30 values are raw structural counts, including and separating `d=0`.
They are not SASS counts, minima, costs, or performance estimates.

## Loom Admission

```text
precheck_frame_sha256=87b58f8b1dff429844a4913d6f8b32e7db12fbe0eb2497106aa72f7df6a425e1
prerun_frame_sha256=f6bd989409f73686d6b9909fe72802c5476163f1cd5076b020e8c8fc68968ad8
freeze_frame_sha256=3aa426e97e146f180d2fe0a5961153e5a7323bb893b94388a66c23bfa4ea2ab0
freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
commit_frame_sha256=286561341ac2f8a5229a51ea196567ddc6e002589803dd5adb5724b968939dc6
commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
```

The freeze used `schema=9020`, `stage=2`, `action=3`, `language=1`, `role=1`,
`policy_state=1`, `semantic_write=1`, and `expected_result_write=1`. Parity,
review-promotion, waiver, and transitional-guardian fields were zero.

## Review-Only Findings

```text
providers=xai/grok-4.5,zai/GLM-5.2
role=REVIEW_ONLY
raw_first=/tmp/llm-offload-8fB9Xx/
raw_followup=/tmp/llm-offload-NQGYfo/
```

The first review found that payload preservation compared a constructed bit
index to itself and that the negative matrix omitted three gate inputs. Sounio
was changed to compare the parsed `.bfly` source-lane address to the expected
XOR source address, complete all 25 gate witnesses, and make that gate decide
the live result. The follow-up found no algebraic error and tightened the claim
to symbolic payload-address preservation. No provider created or confirmed any
count, digest, result, or authority decision.

## Negative Enforcement

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
canonical_command_sha256=3cf716f9176c5d359147c759256be4e569c688558acc8dabfbf449967e3c894e
canonical_action2_frame_sha256=2f1cc9aba44b04c4dedff7c11a8d1571b1e77591e91be69415b48ee2390701c0
canonical_diagnostic_sha256=704243070dd1e3f0b2269e964461203feb7080529a6548ebbd5cda84ecd68c7d
exit_code=1 verdict=1 E012_count=726 E008_count=1
```

The dominant nameless `E012` cascade crosses frozen parents and prevents a
source-local diagnosis. It is a canonical compiler blocker, not evidence
against the accepted `lean_single` Sounio authority stream.

## Claim Boundary

This receipt establishes a structurally derived PTX `shfl.sync.bfly.b32`
candidate for the frozen 16-lane `XOR_PERMUTE` node. It establishes no DGX
observation, SASS equivalence, compiler emission, material parity,
whole-operation lowering, exact reduction tree, instruction minimum, cost,
performance, or cross-ISA parity.

Unresolved blocker contract:

```text
Blocker-ID=BLOCKER-PIREUS-CANONICAL-MADAROS-FIELD-RESOLUTION-20260827
severity=P1
class=build/bootstrap-path
evidence=executable
owner=Madaros compiler lane
acceptance_gate=./bin/souc check examples/pireus_dgx_ptx_shfl_lowering.sio exits 0 without nameless field-resolution cascade
next_action=isolate the earliest valid named E012 in the canonical resolver and repair field/type propagation without changing frozen Pireus semantics
```

Legacy parent artifacts were intentionally retained unchanged.
