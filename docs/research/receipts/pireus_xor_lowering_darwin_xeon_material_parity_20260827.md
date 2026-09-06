<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-lowering-darwin-xeon-material-parity-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-lowering-darwin-xeon-material-parity-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XOR Lowering Darwin Xeon Material Parity Receipt

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

This receipt records one material realization of the frozen Sounio lowering on
one Darwin Xeon node. C++ consumed the masks, selectors, input generator, output
bits, and ascending reduction order created earlier by Sounio. It did not create
or revise semantic values.

The bound parent is:

```text
frozen_sounio_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
sounio_freeze_commit=43adc7f9e7c9
cpp_source_sha256=c5d1ab99da8d7567387772f1b98baf4a162618b82378876853a57ff0362b6cf8
runner_sha256=d6fde54a113edc76291ab6f6b94168e943e13ce0577d4d785a4ba09a34b89d3b
```

## Material Result

The AVX-512 kernel produced the 16 frozen output bit patterns exactly. All 256
vector products matched the corresponding frozen-order scalar terms. Flipping
one sign bit changed one output lane; changing one selector cell changed two.

```text
partner_cells=256
partner_failures=0
negative_cells=120
positive_cells=136
vector_term_matching_cells=256
frozen_scalar_matching_lanes=16
vector_matching_lanes=16
vector_mismatching_lanes=0
sign_mutation_mismatching_lanes=1
selector_mutation_mismatching_lanes=2
ascending_i=true
reassociated=false
result=PASS
result_sha256=fe851cccb1487d3977c491426cd89e1445e3c234fbce8c5444972a441b8876e4
```

This is bit parity for the frozen finite input and exact command below. It is
not a generic real-number equivalence claim.

## XOR Permute Answer

For `f64`, GCC emitted `vpermpd`. The low three displacement bits select one of
eight in-half XOR controls, while bit 3 selects the other eight-lane source
half. The compiled kernel therefore executes two one-source `vpermpd`
instructions per displacement. It contains no `vpermi2pd` or `vpermt2pd`.

There are three static `vpermpd` sites because the first half has two mutually
exclusive source paths and the second half is common. This answers the earlier
material question for this compiler and Xeon; it does not establish a universal
backend rule.

The strict reducer compiled to a straight 16-instruction `addsd` chain in
ascending address order. Masked `vpxorq` realizes the coefficient sign and two
`vmulpd` instructions realize the chunk products.

## Toolchain And Hardware

```text
cluster=Darwin
pod=sounio-workspace-control-0
kubernetes_node=t560-proxmox
architecture=x86_64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
logical_cpus=64
sockets=2
numa_nodes=4
avx512f=true
avx512dq=true
rounding_mode=FE_TONEAREST
mxcsr=0x1f80
flush_to_zero=false
denormals_are_zero=false
hardware_record_sha256=93e3daf8bd024bf2664d379b4ccbc0305ce5823285148cc71058aabdc3c0f550

compiler=/usr/bin/g++
compiler_version=g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
compiler_sha256=1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769
toolchain_record_sha256=1d1e239e199ce5e7416e3d5c66892121ee7bfd1436d1cb2f5f77a486aff85b72
```

The runner compiles with:

```text
-std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off
-fno-tree-vectorize -fno-tree-slp-vectorize -Wall -Wextra -Werror
```

The exact durable command is:

```bash
scripts/ci/pireus_xor_lowering_darwin_xeon_material_parity.sh
```

## Emitted Artifacts

```text
result_lines=52
result_bytes=1436
result_sha256=fe851cccb1487d3977c491426cd89e1445e3c234fbce8c5444972a441b8876e4
binary_sha256=c88cd9ba43e106c1721ab99ea501c1c797935ed77e46f64aedab333f963e399f
assembly_sha256=f889237596f421351f351253672c24013c8ff575bea9da435d93edbfbcdfa1e6
objdump_sha256=26d2f07613acaa91e5aa89a1d41208ae97ebea7283634ef810bd7179b6371251
reproducible_builds=2
all_four_artifact_hashes_reproduced=true
```

The compact, committed evidence record contains all 16 output bits and the
relevant emitted-instruction excerpt:
`docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt`.

## Loom Decisions

No C++ compilation or execution occurred before the parity-open decision.

```text
write_frame_sha256=ff2f8d7ad064b68b39dc49f95db8867d57afc1292f53b2518924e680c59bfa38
parity_open_frame_sha256=9dec5cc55616b35c7b33c35e17ff1b1d0085fb7beecccf3a2f60eb5817b1c666
parity_open_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
receipt_seal_frame_sha256=486e45f0ed5362c6b0ed730f6bc22425fc40e1fb137f13eafa3686a59854e627
receipt_seal_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
commit_frame_sha256=02489983a9109bc4e3b99a1b4c2ca2d915d93cd98114459a89ded6f721d2e2ff
commit_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW
decision_code=0
next_stage=PARITY_OPEN
```

The receipt is sealed but has not been consumed by a later Sounio promotion.
Consequently the frozen plan remains unchanged, material authorization is not
retrospectively written into it, and `CLAIM_READY` remains false.

## Negative Enforcement

A deliberate Python oracle request with complete frozen bindings was refused
before interpreter launch:

```text
frame_sha256=653275a52c611c7eaec66e474d324fc23ca4a66ac4b6e42084b87a7d3c54b130
decision_sha256=42a2eba7ea7889f7526d1e452196003debe55eb388c5854f6d70cc69bdcf8ea4
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY
code=110
reason=forbidden-language
next_stage=PARITY_OPEN
interpreter_launch_count=0
```

Rust was not used.

## Closed Claims

This receipt contains no wall-clock benchmark, generic instruction-cost claim,
numerical-refinement claim, compiler-wide lowering claim, Apple Silicon
observation, DGX observation, cross-ISA parity, subquadratic algorithm, WHT
rewrite, or Fano theorem. Apple Silicon and DGX remain canonical and materially
unobserved. External LLM review is review-only and cannot confirm this result.
