<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-dgx-ptx-shfl-material-parity-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-dgx-ptx-shfl-material-parity-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus DGX PTX SHFL Material Parity Receipt

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

This receipt compares one CUDA C++ realization on an NVIDIA GB10 with the
expected PTX `.bfly` coordinate map and candidate digest born earlier in
Sounio. C++ did not create or revise the member mask, source-lane rules,
payload-address transcript, digest, unresolved operation boundary, or expected
result.

The exact frozen parent is:

```text
sounio_source=stdlib/hardware/pireus/dgx_ptx_shfl_lowering.sio
sounio_source_sha256=4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404
frozen_semantics=docs/research/pireus_dgx_ptx_shfl_lowering_semantics.md
frozen_semantics_sha256=a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336
frozen_sounio_result_sha256=495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e
```

The parity implementation is:

```text
tools/pireus/material_sha256.hpp sha256=ae54c8f455d5ef057f182212aacd466bdf5e014898872706e80e51f6b16e7782
tools/pireus/dgx_ptx_shfl_material_parity.cu sha256=6820fa05ff91cb89012bb0a7651896e196d5ff379be8f90afdc0b8ae08a8688a
scripts/ci/pireus_canonical_target_material_parity.sh sha256=d66b6d7b4b28216d0babfc5f8ad90d58cd38f870a6fcc4e3bfc70033707239c3
scripts/ci/pireus_dgx_ptx_shfl_material_parity.sh sha256=72b3868bb039db269b18dfa64a3b1aa6fc50cd5861e2decbbb72233d9e7e0e0e
```

## Material Result

One warp with the lower 16 lanes active executes two
`__shfl_xor_sync(0x0000ffff, component, d, 16)` operations for every frozen
displacement. The host comparator identifies the material source from a unique
64-bit payload and recomputes the Sounio SHA-256 transcript from those observed
sources.

```text
gpu_name=NVIDIA GB10
compute_capability=12.1
sounio_abstract_membermask_low=65535
sounio_abstract_segmask=0
sounio_abstract_cval=15
sounio_abstract_packed_c=15
emitted_ptx_c=4127
emitted_ptx_c_hex=0x101f
displacements=16
logical_cells=256
matched_cells=256
mismatched_cells=0
in_range_cells=256
inferred_own_lane_fallback_cells=0
active_source_cells=256
member_source_cells=256
max_source_lane=15
sounio_abstract_f64_components=2
material_b32_components=2
payload_component_cells=512
symbolic_payload_bits=16384
matched_payload_bits=16384
abstract_shfl_sync_instructions=32
identity_shfl_sync_instructions=2
nontrivial_shfl_sync_instructions=30
candidate_digest=1996463492:2773712531:2232409634:959326894:3198512505:1781073490:2005131219:1992148529
mutation_mismatching_cells=1
result=PASS
result_lines=39
result_bytes=1285
result_sha256=1e776e655761bd9e59322ac64e736629bd35586a958d503ea814ddddaa865f3c
```

CUDA 13.0 unrolled the frozen displacement loop. The PTX has 32 static
`shfl.sync.bfly.b32` sites and the GB10 SASS has 32 corresponding
`SHFL.BFLY` sites. This is a compiler-and-command observation, not a minimum,
latency claim, throughput claim, or generalized cost.

Every emitted PTX site uses `c=4127` (`0x101f`), while the frozen Sounio
candidate records the abstract tuple `segmask=0`, `cval=15`, `packed_c=15`.
The finite lower-16-lane coordinate and payload result agrees, but the operand
encoding does not. This receipt therefore establishes material coordinate
parity, not PTX operand-encoding equality.

## Hardware And Toolchain

```text
canonical_target=dgx
transport_address=192.168.3.24
hostname=spark-3c59
os=Ubuntu 24.04.4 LTS
kernel=Linux 6.17.0-1021-nvidia
architecture=aarch64
gpu=NVIDIA GB10
driver=580.159.03
compute_capability=12.1
hardware_sha256=8b048f0a20ac0967af5622606935aa4ea4e6caf0baef6a3dcd9b7ff58f2a66d4

nvcc=CUDA 13.0 V13.0.88 build 36424714
ptxas=CUDA 13.0 V13.0.88 build 36424714
cuobjdump=CUDA 13.0 V13.0.85 build 36400806
nvdisasm=CUDA 13.0 V13.0.85 build 36400806
host_cxx=g++ 13.3.0
gpu_code=sm_121
toolchain_sha256=10ab88d927b1285a5ccc6e717c5beb721d76f6e074c4d0c1e9d2f36072c57cf5
command_sha256=cf1796b78caaedc7866e26ba2885cd2fdc3224bdd948e40f8b72fea092223a72
```

The PTX and SASS artifacts reproduced across the pre-review and corrected
builds. Two builds of the corrected source still produced distinct linked
binaries:

```text
ptx_sha256=480c3de12dd2e77b5c29e4f0b889e282fa8c4e0dd1147a151a039b2749db2d2f
sass_sha256=5f34ba10b94797219b128522f1edd34bdbe2f915b67b2c75412c293d81a299f4
binary_build_1_sha256=1913297de0167efc2e9da3a981cb8df53559a65c4380882781614d400edba8c8
binary_build_2_sha256=81d94ebcf31cab7af321cc3789bc065a0ed7e21e98f597d6815108bc4e51ea50
binary_reproducible=false
ptx_reproducible=true
sass_reproducible=true
result_reproducible=true
```

The CUDA-linked executable packaging is therefore not a reproducible artifact
under this command. The stable parity roots are the frozen source, command,
PTX, SASS, and result hashes; the receipt does not conceal the binary drift.

The durable command is:

```bash
scripts/ci/pireus_dgx_ptx_shfl_material_parity.sh
```

## Loom Decisions

No CUDA C++ compilation or target write occurred before `PARITY_EXECUTE`
returned `next_stage=PARITY_OPEN`.

```text
write_frame_sha256=9b8cd05852c4ac571b142e47f948b85aab0e19b356f49da00ef3a1176eb74d73
parity_open_frame_sha256=c9c496f3d3327b0c61080c9e1315e41885fb5e28c129abeae8b8440d41aa1ad9
parity_open_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
receipt_seal_frame_sha256=630a66ebae11ca1c46a900a32d526e389626916c7f727d97845cc466e2d595a9
receipt_seal_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
commit_frame_sha256=44cb1a9c5476b211aec02418bb52b08fb5057e956ca9ab5bb831266df487597c
commit_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW
next_stage=PARITY_OPEN
```

The deliberate Python request was refused with `code=110` before any
interpreter launch:

```text
python_frame_sha256=7b89a9e7d1d8399dc13d60bbfa97d8c9e8da7d437f7b1f546e64cfb18aa90004
python_decision_sha256=42a2eba7ea7889f7526d1e452196003debe55eb388c5854f6d70cc69bdcf8ea4
interpreter_launch_count=0
```

The T560 key stayed mounted at its source host and was used only for SSH
authentication. It was neither printed nor copied.

## Second Address

`192.168.3.48` returned `No route to host` from both the T560 namespace and
`spark-3c59`. It is not covered by this receipt and remains materially
unobserved. This does not weaken the GB10 receipt for `192.168.3.24`; it prevents
one observed DGX from being laundered into two.

## Closed Claims

This receipt does not resolve the three non-selector operation nodes, prove an
exact reduction tree, establish cross-ISA equivalence, measure performance, or
promote a cost model. External LLM output remains review-only. Rust was not
used. `CLAIM_READY` remains false.

Raw compact evidence is in
`docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt`.
