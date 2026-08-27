<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-lowering-darwin-xeon-fleet-parity-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-lowering-darwin-xeon-fleet-parity-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XOR Lowering Darwin Xeon Fleet Parity Receipt

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

This receipt extends the one-node Darwin material result to all five Xeon nodes
currently present in the Darwin cluster. It consumes the exact C++ binary and
the exact expected bits previously derived from frozen Sounio semantics. It
does not create or revise any semantic value.

```text
frozen_sounio_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
sounio_freeze_commit=43adc7f9e7c9
darwin_xeon_material_parity_commit=1a6934a49610
cpp_source_sha256=c5d1ab99da8d7567387772f1b98baf4a162618b82378876853a57ff0362b6cf8
binary_sha256=c88cd9ba43e106c1721ab99ea501c1c797935ed77e46f64aedab333f963e399f
```

Each node had its own hardware record and exact transport command. Loom received
one `PARITY_EXECUTE` request per node and returned `ALLOW` before the binary was
copied into or executed inside that worker pod.

## Fleet Result

| Node | Xeon | Model/step | CPUs | Sockets/NUMA | Result |
| --- | --- | --- | ---: | ---: | --- |
| `5860-proxmox` | w3-2423 | 143/8 | 12 | 1/1 | exact |
| `dl380-proxmox` | Gold 6262V | 85/7 | 96 | 2/2 | exact |
| `r740-proxmox` | Gold 6148 | 85/4 | 80 | 2/2 | exact |
| `r770-proxmox` | 6730P | 173/1 | 128 | 2/2 | exact |
| `t560-proxmox` | Gold 6526Y | 207/2 | 64 | 2/4 | exact |

All five nodes expose AVX-512F and AVX-512DQ. On every node, the same binary
produced the same 52-line, 1436-byte output:

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

The complete 55-line fleet summary was reproduced twice byte for byte:

```text
fleet_summary_lines=55
fleet_summary_bytes=3243
fleet_summary_sha256=ea9e2e1f4be7926c76262876960bb673455ef9786a391286676f8b4c17539e19
reproduced_fleet_runs=2
```

This records five node-scoped executions of one exact binary on one frozen
finite input returning the same output digest. It does not prove equivalence
between the five CPU models, and it is not a timing result or generic numerical
equivalence claim.

## Fleet Bindings

```text
fleet_runner_sha256=b285eaed595c6d52a0e272aee67d75f288088d79155bda8fdb73acee30852a78
fleet_hardware_sha256=a165131d9bb9754946c1062f06ad52c1322e8043598d648f05c3cc63cf0c4635
fleet_command_sha256=bdf3b556fa267c75ce68794c90645a9c5925061c456ffbea9f6d8685f28a256b
toolchain_record_sha256=1d1e239e199ce5e7416e3d5c66892121ee7bfd1436d1cb2f5f77a486aff85b72
```

The exact durable command is:

```bash
scripts/ci/pireus_xor_lowering_darwin_xeon_fleet_parity.sh
```

The runner fails closed on source, runner, binary, hardware, command, Guardian,
decision, or result drift. Shell utilities only transport, classify, and hash
records; they do not produce semantic or expected values.

## Loom Decisions

```text
write_frame_sha256=17456e186f832c4183fd676b08023666e4ca3744a65c563d0dff732dcb837cb3
node_5860_frame_sha256=c365d59de972abc4fdf29de4c3b46043035c668ae130172e1c7081b3629ab034
node_dl380_frame_sha256=413914912a0c551f7c29bd94c5dfd5bf65913b670faff305c0251e16848834ce
node_r740_frame_sha256=762ab04efc12d5ed5ecd160d02e1287f1639133b6bb56a87d7f463fd4c8b2643
node_r770_frame_sha256=501effb1fe58a2ef037de07313f8a325fe1e9f7c7488ef53034fa8543f897221
node_t560_frame_sha256=612ce583159b7f4741fb7c03eb61ec83fabda6978d9014e3f8a8c75fd3fced92
receipt_seal_frame_sha256=c1fd648a083c70ea4542a2ce8a13f5574b2ad7bc6d0a263c78443c295afc03ac
commit_frame_sha256=6d53d5ed5d55ece0f4583a0e1dbf636e9acdfd4ebb56e4ee00b996f219edf9cc
allow_decision_sha256=d0d918e742c2c3791f353fd63340af9222a2ce4bd91f257dd1d0d8e66681ae5e
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW
decision_code=0
next_stage=PARITY_OPEN
```

The receipt is sealed but has not been consumed by a Sounio claim promotion.
`CLAIM_READY` therefore remains false.

## Negative Enforcement

A deliberate request to execute the aggregate parity action as a Python oracle
was refused before interpreter launch:

```text
python_oracle_frame_sha256=c77edb1ae6d1dd457f61d0f00937dc42d63a5f8a851323c9ac7e21d0d3d2b2d3
python_oracle_decision_sha256=42a2eba7ea7889f7526d1e452196003debe55eb388c5854f6d70cc69bdcf8ea4
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY
code=110
reason=forbidden-language
next_stage=PARITY_OPEN
interpreter_launch_count=0
```

## Canonical Targets

Apple Silicon and DGX remain canonical targets. Neither was observed in this
receipt. The configured Apple hostname did not resolve; the DGX address was
reachable but refused the available SSH credentials. Those are access facts,
not parity failures and not permission to manufacture expected results.

The kernel output contains the inherited line `material_nodes_realized=5`.
That line predates this fleet run and is not treated as evidence. The fleet
count comes only from the five independently bound records above.

## Closed Claims

This receipt contains no performance measurement, generic instruction-cost
claim, compiler-wide lowering claim, Apple Silicon observation, DGX observation,
cross-ISA parity, subquadratic algorithm, WHT rewrite, or Fano theorem. External
LLM review remains `REVIEW_ONLY` and cannot confirm the result. Python and Rust
were not used as producers, oracles, or Guardians.

The machine-readable evidence is
`docs/research/evidence/pireus_xor_lowering_darwin_xeon_fleet_parity_20260827.txt`.
