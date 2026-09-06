<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-lowering-legality-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-lowering-legality-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XOR Lowering Legality Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`

Semantic-Lane-ID: `pireus-xor-lowering-legality-20260827`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

This receipt records the Sounio authority stream for a complete semantic
schedule over the five frozen operation nodes. The stream reports 32 derived
coefficient masks, 256 accepted partner identities, and 16 bit-matched outputs
without reassociation. The executable source, byte-exact matcher, and paired
semantics document are the audit surfaces; this receipt is not an independent
symbolic proof.

The checked cell identity is `partner[d,i] = i XOR d`, and each sign bit is
derived from `algebra::cayley_dickson::cd_sigma(i, i XOR d, 4)`.

```text
semantic_authorized_nodes=5
material_candidate_nodes=1
material_authorized_nodes=0
unresolved_material_nodes=5
sign_groups=32
negative_cells=120
positive_cells=136
partner_table_valid=true
bit_exact_lanes=16
tree_reduction_exact=false
```

The sole material candidate is the frozen Intel vector-control form for
`XOR_PERMUTE`. It is not a compiler emission or complete lowering.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden | `3eff32209f18da9b10d679c75230e4cbc7a2ca7b` | admitted the legality question without expected values |
| Sounio executable | `2f23d18a2725b0b1fdc750439a1580421d177b6a` | emitted the result before frozen prose existed |
| semantics freeze | enclosing Git commit | binds source, semantics, receipt, and registry |

## Sounio Sources

| Artifact | SHA-256 |
| --- | --- |
| Garden | `88e3ca09aa3771691d75fd65e32feab228f491f479363738e9a1bbd460b84d5e` |
| module | `7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb` |
| authority executable | `f4cd9a96b508b24aaf3a00c92c042162da3f59e28c336fb8d248f2cf984b9d14` |
| dedicated test | `7afb3626399fdaedaae41ff5be3e1be63c65a11684ca25cce8e59052b8c59b5f` |
| source manifest | `c839f3fac9271cb0169e859474a6fe8d6d3565342c2828960d7f164f63946a77` |
| concatenated source bundle | `1be5d0bac30706057eb24d7ac4b548737117d2f186862673a0c14cdc824406aa` |

The Loom `source_sha256` field uses the module hash because the module owns the
semantic matcher. The manifest and bundle retain the complete executable and
test lineage.

## Frozen Parent Receipt

```text
parent_bundle_sha256=012126771572d6634f2255c606d6a5315953e1874a68e0cfc2b2c2ae0f01b0aa
parent_triplets=3
parent_hashes=9
operation_live_match=true
material_live_match=true
intel_live_match=true
```

The nine individual hashes are listed in the paired semantics document and
encoded in the Sounio module.

## Toolchain, Hardware, And Command

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
selected_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
toolchain_record_sha256=4dbb7639978fe5f09ce5b1488522a552d9925ded37c7978927a442607e9a6050
kernel=7.0.2-5-pve
architecture=x86_64
logical_cpus=64
sockets=2
numa_nodes=4
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
hardware_record_sha256=c903bd1adbfa515b08553c4a11967c33374f055b8f757aeed3ae2c4316338a0d
```

The exact authority command was:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_xor_lowering_legality.sio \
  /tmp/intel-sdm-vol-2c-326018-092.pdf \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
```

```text
command_sha256=ddc0d1b9e53a1be6db7dabea7e7987d6a00f5a38c8931ac8b39938cafcce94bd
```

## Authority Stream

```text
lines=330
bytes=5354
sha256=9fe67671a7fa3e5141136f2c385d42ea99474fcb7d50fc886bb76687d28c59ac
error=0
failures=0
```

The dedicated Sounio test emitted
`PIREUS_XOR_LOWERING_EXECUTABLE_OK`; its one-line SHA-256 is
`dd523634dd757b51c4e9644d5336146e3ae6f0d5d88c8bd0c92a36e23351d104`.

## Default Compiler Diagnostic

The unforced command `./bin/souc check
examples/pireus_xor_lowering_legality.sio` selected Madaros v0.80 and exited 1
while checking the imported parent surface. It emitted the advisory
science-boundary verdict `UNKNOWN` and diagnostics in these classes:

```text
E008=1
E011=34
E012=479
E013=26
E035=9
E137=8
log_lines=3879
log_bytes=102358
log_sha256=1d3b15096b3d3986f69bc5f37a483a623a2ed6324924b5233ded8247ad68c108
```

This is a default-path compiler/checker blocker. It is not a second expected
result, a fallback authority result, or evidence against the successful
explicit `lean_single` authority execution. No Madaros source changed here.

## Frozen Result Digests

```text
plan_sha256=7fadb763fd506fb2e6473ae31e17a0a32e20e110ba48e314489dcb028b9ac2b2
signs_sha256=2cd91f8e407ab465fdaa985bb59bb99216501b39b85300c1186d22bf807535bd
execution_sha256=6440e6e81515deb30f38a68e16282051b30a1dd08cc3fb3511b406fd495da1e0
targets_sha256=45ce281bdaf5ab1d78b63261e5bdb028818dfb31ee92a031e65e131c5bb7fc95
witnesses_sha256=c7acb2e78da96b509ea351856e33b9e6d71887cff817f1f0ce2b4df295d5c8fd
```

## Review-Only Offload

xAI/Grok 4.5 reviewed the Garden and executable. It found the missing local
partner-table validation, an implicit chunk-divisibility assumption, and weak
negative probes. The final source validates every partner identity, enforces
divisibility and capacity, corrupts a real sign mask, injects an invalid Intel
form into `MULTIPLY`, separates bit-order from numerical-contract checks, and
names its result `bit_exact_to_frozen_parent`.

```text
provider=xai/grok-4.5
outcome=REVIEW_COMPLETE
role=REVIEW_ONLY
raw_garden=/tmp/llm-offload-oWpMYg/
raw_first=/tmp/llm-offload-kTaKx1/
raw_followup=/tmp/llm-offload-N2ytnu/
raw_semantics=/tmp/llm-offload-lD1itY/
raw_receipt=/tmp/llm-offload-TN8qy0/
```

No model created or confirmed masks, counts, output bits, digests, or
authority values.

The semantics review independently recounted all mask bits and requested an
explicit `sigma` source; the final text names the exact hash-pinned Sounio
`cd_sigma` function. The receipt review required the result paragraph to state
that this page records executable outcomes rather than proving them; the audit
surfaces are now named explicitly.

## Freeze Decision

The complete Loom frame bound source, reviewed semantics, parent closure,
toolchain, hardware, command, and the Sounio authority stream:

```text
source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
parent_semantics_sha256=012126771572d6634f2255c606d6a5315953e1874a68e0cfc2b2c2ae0f01b0aa
toolchain_sha256=4dbb7639978fe5f09ce5b1488522a552d9925ded37c7978927a442607e9a6050
hardware_sha256=c903bd1adbfa515b08553c4a11967c33374f055b8f757aeed3ae2c4316338a0d
command_sha256=ddc0d1b9e53a1be6db7dabea7e7987d6a00f5a38c8931ac8b39938cafcce94bd
result_sha256=9fe67671a7fa3e5141136f2c385d42ea99474fcb7d50fc886bb76687d28c59ac
frame_sha256=663fcdb02649ad33587f3eba3439c390d73a2d5528b5cfe795278d3b7946a499
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
```

## Negative Enforcement

All 20 in-process Sounio mutation witnesses pass. With the complete frozen
bindings, the external Loom guardian separately refused a would-be Python
parity producer before interpreter launch:

```text
frame_sha256=604e072467362321d8dfe8f998f75d17a16715fc4c1be833f9a71964042dab22
decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
interpreter_launch_count=0
```

Rust was not used.

## Canonical Targets And Closed Claims

Darwin Xeon, Apple Silicon, and DGX are canonical and materially unobserved.
This receipt authorizes no compiler emission, instruction count, hardware run,
cost, performance claim, Apple or DGX lowering, cross-ISA parity, subquadratic
algorithm, Walsh-Hadamard rewrite, or Fano theorem.
