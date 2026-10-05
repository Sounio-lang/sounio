<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-convolution-operation-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-convolution-operation-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XorConvolution Operation Sounio Authority Receipt

> **Date**: 2026-08-27 | **Stage**: `SEMANTICS_FROZEN`
>
> **Language**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Scope

This receipt binds the first material-neutral Pireus operation DAG for the
frozen bits=4 `XorConvolution` semantics. It establishes exact parent bindings,
the five-node operation path, capability requirements, fixed reduction order,
canonical target declarations, Sounio output bits, negative promotion gates,
and six Sounio-produced digests.

It establishes no target lowering, processor observation, instruction count,
performance result, transform theorem, cross-language parity, or claim-ready
promotion.

## Ordered Commit Chain

| Stage | Commit | Content |
| --- | --- | --- |
| `GARDEN` | `9a28b9478a` | operation-DAG boundary with no executable or expected result |
| `GARDEN` addendum | `37aa784c93` | frozen graph-parent receipt binding after bootstrap collision diagnosis |
| `SOUNIO_EXECUTABLE` | `5329b82de9` | first Sounio operation, executable output, validators, and dedicated test |
| `SEMANTICS_FROZEN` | `2f696bf6b1` | post-result matcher, concept contract, and hash-pinned semantics |

The executable commit is descended from both Garden commits. The byte-exact
matcher and all expected lane bits appear only in the later freeze commit.

## Frozen Artifacts

| Artifact | SHA-256 |
| --- | --- |
| primary Garden seed | `d87fd24678611877846e8b36c5ca2c70a8fcf4033f40032ef3dee603b2bc6d88` |
| parent-link Garden addendum | `02c72dd3aa71debbdf36a8125c744a16d6bf99d472ea5a9f6208f73757a4cd5e` |
| first executable module | `b80b624910593ce027504ae5d076191f50f3468fdf9c09140bffde240e2e5ee3` |
| first executable entrypoint | `e5447fff49e492ecfd6c07cdf42b7ea18d1edc0553771c1ac230c4e9afaee16a` |
| frozen module | `bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8` |
| frozen entrypoint | `e5447fff49e492ecfd6c07cdf42b7ea18d1edc0553771c1ac230c4e9afaee16a` |
| dedicated test source | `774bc99cc8af305b89dba376e60d6bc27c66343dcb846310295c390ed5197b95` |
| concept contract | `f9f2daf9991cf6f55941ce4f76eee62229e86dd7df6bf7177a8da20ee531d752` |
| frozen semantics | `40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1` |

The first executable source record combines its module and entrypoint paths
and hashes. Its SHA-256 is
`72ec1be2503bc5949304b3dd41695da1f7a9280b7c05f673f46e445abf54fb9d`.
The corresponding frozen source record SHA-256 is
`69e3ea7f30913b692f5fad571f4964abdadf420579a2d04c84dc159335b801bd`.

## Frozen Parent Receipt

The live XorConvolution parent semantics are pinned to:

```text
da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502
```

The graph-identity parent is bound through the complete frozen Sounio receipt:

```text
identity_module=caedf51babd450db0af50f9755e677786cc8b563ad923f3598153759859f9985
semantics=8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb
authority_stream=5b3efa606d86805aa222ced72a37ed87e7b3dab66b21e58e0547163aa19c83dd
registry=9b56f6f0306d949e2266776ee34f05f3ba1dec4239e0bba9411b3aed9c2b27ce
dependency=4dd37bf1cdd774e4ab840e5444d7b18b8a1d0990063901b8a85743a7ac2abbcc
lifted_graph=0bcf3ef8b9598cb4363864d9ba75d9b050a22df501b80a09eda7290b3e331765
occurrence=57218fbb4a6d640e4651dea0d14a17a54559a2f559e45e3186a46df7d8a05950
collision=3a72cc5158aa0e841b4b13de2a924d1bca516778b651ae3f1fe9be80d26925bb
provenance=1e962677cfb1846a5e5b9dd70c13c25cae5f92ad905f6ad795a8912b4e352f20
```

The SHA-256 of the exact parent record used by the authority frames is
`a66508502c9bd8bbb4299b375db20967f7f1efa6303b4ca34bb26dc109579558`.

The receipt path is a bootstrap accommodation. Importing both frozen parent
modules into one `lean_single` bundle causes flattened private-helper name
collisions. The available Madaros prebuilt reported version v0.80 and emitted
broad import/struct diagnostics. Neither failure was used to redefine or
invalidate a frozen parent.

## Toolchain Receipt

The semantic-authority run used the public wrapper, never a raw ELF:

```text
engine=lean_single
wrapper=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
```

The SHA-256 of that exact record is
`0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556`.

This is the explicit `lean_single` authority path. The default Madaros path,
rebuilt ontology-validation wrapper, and fallback compiler paths did not
produce or confirm this result.

## Hardware Receipt

```text
kernel=Linux 7.0.2-5-pve
architecture=x86_64
logical_cpus=64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
sockets=2
cores_per_socket=16
```

The SHA-256 of that exact record is
`c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d`.

This machine has AVX-512 capability, but this receipt contains no observation
that the operation or its compiler output used AVX-512.

## Command Receipt

The authority command was:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_xor_convolution_operation.sio
```

Its newline-terminated command record has SHA-256
`18edcffbb7bd6b13d17d4205eed884faff23b39fb7bafc0218f965bce066a65b`.

The dedicated semantic test command was:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_xor_convolution_operation.sio
```

Its command record has SHA-256
`b9d71bdb7757289166f598e1cb87ef6ae17da539743213c8dec8a0459afe3d8b`.
It ran only after a Loom `CI` pre-action frame returned ALLOW.

## Sounio Result

The final authority stream is 204 lines and 5,043 bytes. Two pre-freeze and two
post-matcher executions were byte-identical. Its SHA-256 is:

```text
9fef54b41f4089ba25ccbbbcaea50ab1ec16f981fed41e163de0ac1877acf39b
```

Sounio produced:

```text
bits=4
dimension=16
valid=1
error=0
xor_parent_frozen=1
graph_parent_frozen=1
xor_parent_match=1
graph_parent_receipt_valid=1

overlay_graph_id=7
overlay_owner_id=9
nodes=5
edges=4
nonassociative_barriers=3
output_matches_parent=1

zero_free=1
normalized=1
displacement_only=0
rank_one_separable=0
left_square=1
group_two_cocycle=0
associator_defects=1848
wht_rewrite_authorized=0

material_observations=0
lowerings=0
cost_records=0
material_receipts=0
negative_tests=20/20
failures=0
parity_open=0
claim_ready=0
```

The path is `XOR_PERMUTE -> TWIST_APPLY -> MULTIPLY ->
HORIZONTAL_REDUCE -> OUTPUT_LANE`. Its barrier mask is
`[false,true,false,true,true]`, so the three barriers are `TWIST_APPLY`,
`HORIZONTAL_REDUCE`, and `OUTPUT_LANE`.

The algebraic fields are finite values consumed from the hash-pinned Sounio
parent. The operation matcher binds them; this receipt does not independently
prove them. `associator_defects` counts the parent's complete ordered-triple
domain `[0,15]^3` under the standard twist associator equation:

```text
sigma(i,j) * sigma(i XOR j,k)
  != sigma(j,k) * sigma(i,j XOR k).
```

The exact 16-lane result-bit vector is frozen in the semantic document and the
post-result Sounio matcher.

## Operation Digests

| Object | SHA-256 |
| --- | --- |
| parent bindings | `12686c8de11d8eb6fe422085ee39b983045c81ff26e6967813aa1543e5b586e5` |
| operation graph | `ce636b3bbaea074e2033b6656120256f53b75d00b0958d57269a63d836a90bfa` |
| capabilities | `b8f34a86e3fd7e22225c755528e160f4f5eabf1b8d0c57e473a0455064db8ebe` |
| targets | `38f74af02731ec3c2ac6ac1ce57659cde2351c157726dcb19aa8d6163d7282c6` |
| result | `84edf6bae148754ebd0e8722368e2eb06095cd929779c36def4f3bb5000013a3` |
| negative witness | `d726dd98af6b52258f91709d8df3f050eee6b4944e19e67743707ef1d2a08a23` |

These are Sounio-produced digests over explicit canonical fields. No external
model or disposable-language oracle computed a comparison value.

## Canonical Target Declarations

| Target | Canonical | Observed | Material receipts |
| --- | --- | --- | ---: |
| Darwin Xeon | true | false | 0 |
| Apple Silicon | true | false | 0 |
| DGX | true | false | 0 |

All observed Darwin CPUs in the admitted parent are Xeon. Apple Silicon and
DGX are canonical targets, but canonical declaration is not observation.

## Loom Admission

The operational guardian was the installed native realization of frozen Sounio
Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
freeze_manifest_sha256=5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
```

The final `SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN` frame bound:

| Field | SHA-256 |
| --- | --- |
| frozen source record | `69e3ea7f30913b692f5fad571f4964abdadf420579a2d04c84dc159335b801bd` |
| frozen semantics | `40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1` |
| parent record | `a66508502c9bd8bbb4299b375db20967f7f1efa6303b4ca34bb26dc109579558` |
| toolchain record | `0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556` |
| hardware record | `c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d` |
| command record | `18edcffbb7bd6b13d17d4205eed884faff23b39fb7bafc0218f965bce066a65b` |
| Sounio result stream | `9fef54b41f4089ba25ccbbbcaea50ab1ec16f981fed41e163de0ac1877acf39b` |

The frame used `schema=9020`, `stage=2`, `action=3`, `language=1`, `role=1`,
`policy_state=1`, `semantic_write=1`, `expected_result_write=1`, and zero for
parity, review-promotion, waiver, and guardian-bootstrap fields. Loom returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

Separate pre-action frames allowed Sounio execution, CI, documentation gates,
review-only offload, executable commit, and freeze commit. No parity transition
was requested.

## Deliberate Python Refusal

Before Sounio execution, the same binding was presented as `language=7
Python`, `role=7 PROHIBITED`. Loom refused before any interpreter launch:

```text
exit_code=110
SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=GARDEN
decision_sha256=30b2f51d293ccfe005319d25d6c5179cd6fc015ae7d0e3ec5a11b162f9c7def7
```

No Python or Rust program created, checked, or confirmed a semantic value.
Node executed only the repository's deterministic documentation-metadata
generator after semantic values existed. It was not an oracle and did not make
an authority decision.

## Review-Only Offload

xAI/Grok 4.5 performed focused `math-review` passes only after the first Sounio
result existed. It caught vacuous negative validators, an insufficient WHT
promotion condition, and wording that mixed mathematical legality with receipt
authorization. Those issues were corrected and the Sounio source was
re-admitted and re-executed after each semantic change.

The semantic-document review then required the finite parent-binding boundary
and exact associator-defect domain to be explicit. The final review found no
wrong claim. Z.AI returned empty artifacts twice, while Qwen, DeepSeek, and
Groq fallbacks failed externally. The review is therefore single-provider
degraded and cannot confirm authority.

No model created or confirmed result bits, counts, digests, or expected values.
The append-only record is in `.claude/llm_offload_log.md`.

## Validation

Passed on the explicit `lean_single` authority path:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_xor_convolution_operation.sio
byte comparison: two pre-freeze and two post-matcher streams identical
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_xor_convolution_operation.sio
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
git diff --check
```

The dedicated test printed `PIREUS_XOR_OPERATION_OK` and exited zero. All
execution and CI commands above were preceded by an ALLOW decision from Loom.

## Legacy And Remaining Boundaries

The frozen XorConvolution and graph-identity modules were intentionally kept
unchanged. The operation consumes their contracts; it does not replace them.

Still closed:

- Lean 4 formal parity;
- Koka effect parity;
- C++ material parity;
- optional Haskell denotational baseline;
- target lowering and material measurement on Darwin Xeon, Apple Silicon, and DGX;
- x86, AArch64, Apple Metal, PTX, or SASS instruction selection;
- cost, throughput, latency, and speedup claims;
- Fano-plane interpretation of the seven-negative rows;
- Walsh-Hadamard diagonalization or any subquadratic algorithm;
- native no-Python hook migration beyond the installed Loom policy runtime.

`PARITY_OPEN=false` and `CLAIM_READY=false`.
