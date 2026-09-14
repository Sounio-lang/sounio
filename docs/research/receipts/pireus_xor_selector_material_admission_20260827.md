<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-selector-material-admission-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-selector-material-admission-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XOR Selector Material Admission Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-SELECTOR-MATERIAL-ADMISSION`

Semantic-Lane-ID: `pireus-xor-selector-material-admission-20260827`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

This receipt records the first Sounio authority overlay that admits three
already sealed target-local material receipts against the frozen five-node XOR
lowering parent.

```text
material_files=6
material_file_matches=6
admitted_receipts=3
admitted_nodes=7
unresolved_nodes=8
refused_nodes=0
selector_targets=3
whole_operation_targets=1
incomplete_targets=2
unnamed_node_claims=1
encoding_differences=1
negative_witnesses=22/22
cost_records=0
generic_instruction_cost=false
cross_isa_equivalence=false
transform_authorized=false
parity_open=false
claim_ready=false
```

The Sounio source, matcher, executable stream, paired frozen semantics, and
flat evidence file are the audit surfaces. This page records their outcome; it
does not independently prove the material observations.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden | `b53115358687f2d660d3bc5596f07a37aa4929fb` | fixed admission schema without expected records |
| Sounio executable | `fdd444afc5ba0e7529bfee532640dc0a665bfc3f` | emitted records and digests before frozen prose |
| semantics freeze | enclosing Git commit | binds semantics, receipt, evidence, gate, and registry |

The Garden commit is an ancestor of the first Sounio executable commit.

## Sounio Sources

| Artifact | SHA-256 |
| --- | --- |
| Garden | `68b2844934cc1e7544794dd5fdb35d56387a58ad2a536075d81a6378feda34fe` |
| module | `b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66` |
| authority executable | `dd7ecca3f8401ec7228033e5829ea8481ede8d5b8c3b3a5e514ab76a4caddb2e` |
| dedicated test | `66e64a9b023bb09e611f83210415040e4e1ad5d580cfa1abe4554db7fcb50b8e` |
| source manifest | `b3d65a5d278ceb034862af8315a861d556b6dcc0ad01b095196517e3e333b6a1` |
| frozen semantics | `17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9` |

The source manifest is SHA-256 over the ordered `sha256sum` records for the
module, executable, and test. The Loom source field uses the module hash
because the module owns the record schema and exact matcher.

## Frozen Parent Closure

The ordered parent manifest contains the Garden root and nine required
lowering/material parents:

```text
garden_artifacts=1
lowering_parent_artifacts=3
material_receipt_evidence_artifacts=6
parent_manifest_sha256=23eeef8d222c99674bc3a3f92ea5cb46772fc5d7a58ed74af36469a9f32ef712
lowering_live_match=true
material_file_matches=6
```

The exact ten artifact hashes are listed in the paired semantics. Sounio
re-evaluates the lowering parent and reads and hashes all six material files.

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

This Linux Xeon host transported the Sounio authority executable. It is not a
new Darwin, Apple, or DGX material observation.

## Command And Stream

The exact authority command record was:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_xor_selector_material_admission.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt | tee /tmp/pireus-xor-selector-material-admission.authority.txt
```

```text
command_sha256=bd6dc15675b133ac133114fffbf70e9d6eb3f851bf5c7864b296726bf5cdee97
lines=201
bytes=3473
result_sha256=e8a0c579b064a63837058f6dd2c2d578ea22062444e5c18f769309afa838f176
error=0
failures=0
```

Three authority executions were byte-identical. The dedicated Sounio test
emitted `PIREUS_XOR_SELECTOR_MATERIAL_ADMISSION_EXECUTABLE_OK` with SHA-256
`ed8719a3b76b982213811c753f2b5fa397029755a56f2a163ea0e69a33b47e2e`.

## Canonical Records

| Target | Binding | Admitted | Unresolved | Refused | Whole | Encoding | Static sites | Secondary sites |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Darwin Xeon | 0 | 5 | 0 | 0 | true | uncomparable | 3 | 0 |
| Apple Silicon | 0 | 1 | 4 | 0 | false | uncomparable | 1 | 0 |
| DGX | 0 | 1 | 4 | 0 | false | different | 32 | 32 |

```text
darwin_digest_words=3500460042:535259591:2326513808:3224911087:1734937489:1992726449:2327971184:3455846257
apple_digest_words=4128037373:2968517972:275166301:843980919:2897963936:177801273:2520874503:391658731
dgx_digest_words=3400682017:537641554:1229578524:2856974856:2840878714:2939595095:1922878528:3271869735
admission_digest_words=1981472606:3869995793:2634373272:1299673842:1000193586:119066938:1531096938:209245254
```

The Apple parent identifies `demetrios@sounio-language-macbook` on the
tailnet, host `Sounio-Language-MacBook`, model `Mac17,7`, Apple M5 Max. This
receipt binds that parent and does not claim a second Apple execution.

The DGX unnamed resolved count remains one, while nodes 1 through 4 remain
unresolved. No IP address or node identity is inferred in this lane.

## Review-Only Offload

xAI/Grok 4.5 and Z.AI/GLM-5.2 acted as hostile reviewers only. They found and
caused repairs to live-file binding, aggregate digest completeness, claim
boundary dataflow, and per-record digest completeness.

```text
role=REVIEW_ONLY
raw_first=/tmp/llm-offload-QBvVp9/
raw_final=/tmp/llm-offload-W1gcR6/
raw_closure=/tmp/llm-offload-E0468v/
raw_semantics=/tmp/llm-offload-0cQRu8/
```

No model executed Sounio or created or confirmed records, counts, digests,
expected values, material observations, Loom decisions, or authority.

## Freeze Decision

For this admission frame, the Loom field named `parent_semantics_sha256`
carries the ordered parent-manifest digest defined above. It is an alias at the
frame boundary, not a claim that the material receipt files are semantic
producers.

```text
source_sha256=b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66
semantics_sha256=17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9
parent_semantics_sha256=23eeef8d222c99674bc3a3f92ea5cb46772fc5d7a58ed74af36469a9f32ef712
toolchain_sha256=2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e
hardware_sha256=b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0
command_sha256=bd6dc15675b133ac133114fffbf70e9d6eb3f851bf5c7864b296726bf5cdee97
result_sha256=e8a0c579b064a63837058f6dd2c2d578ea22062444e5c18f769309afa838f176
frame_sha256=6c5f66fab86d98939fb8fae746e97dd3c7f0cec779b9c32f64aa0991b1070eef
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
```

## Negative Enforcement

All twenty-two in-process Sounio mutation witnesses pass. A live-file negative
changes a copy of the Darwin receipt in `/tmp`; Sounio reports five matches,
`binding_error=1`, refuses the Darwin receipt, returns `ERR_RECEIPT=2`, and
exits 1.

```text
tampered_command_sha256=2cf3d783d002e3034fbaeea0887c17e09f88d040d1dba6f6206dede430cd857b
tampered_frame_sha256=4b22f4248e944af127bb1b7d47d04b59e2bd3602e923d6aa0944dd57ea183b3f
```

The external Loom guardian separately refused a Python parity producer before
interpreter launch:

```text
python_frame_sha256=e2fe5456c6284f019eba4b5a34ab4662746f6dcb1da397895f78b80bf7413ea6
python_decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
interpreter_launch_count=0
```

Rust was not used.

## Closed Claims

This receipt opens no Lean, Koka, C++, Haskell, or LLM authority for the
admission overlay. The admitted C++ material observations retain their
`MATERIAL_PARITY` role and cannot become semantic authority.

No new target execution, compiler emission, instruction minimum, cost,
performance result, cross-ISA equivalence, complete Apple or DGX lowering,
operand-encoding equality, numerical reassociation, Walsh-Hadamard transform,
or subquadratic algorithm is established.

`PARITY_OPEN=false` and `CLAIM_READY=false`.
