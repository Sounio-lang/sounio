<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-graph-identity-composition-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-graph-identity-composition-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Graph Identity Composition Receipt

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-GRAPH-IDENTITY-COMPOSITION`

Semantic-Lane-ID: `pireus-graph-identity-20260827`

Status: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Authority Binding

```text
producing_language=Sounio
language_role=SEMANTIC_AUTHORITY
semantic_authority_language=Sounio
semantic_authority_role=SEMANTIC_AUTHORITY
```

No parity language, external model, or disposable script produced the registry,
lifted identities, collision census, query counts, negatives, or expected
result.

## Mandatory Order

| Stage | Commit | Evidence |
| --- | --- | --- |
| `GARDEN` | `9bb946a4ed64` | byte-exact founder seed, no executable |
| `SOUNIO_EXECUTABLE` | `122a3cc591` | first Sounio result, no expected values in source |
| `SEMANTICS_FROZEN` | `7c69298111` | Sounio frozen-result predicate and hash-pinned semantics |

The transition order is therefore repository-visible rather than asserted
retrospectively.

## Source Binding

| Artifact | SHA-256 |
| --- | --- |
| Garden seed | `31611e5bfb92b0043f19302d6cf7865d106ea732658bc88378509393dc5fe300` |
| concept contract | `b0909137ba1b0fb48f0924cbd1f04c7e44cfcab3c799840109954b1e22bdf8d2` |
| identity module | `caedf51babd450db0af50f9755e677786cc8b563ad923f3598153759859f9985` |
| executable witness | `6708c1b970b6d0961a1c338252dcd55d9499e16249bafb42373fdc6df7651898` |
| frozen semantics | `8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb` |

The seven admitted producer sources are:

| Producer | SHA-256 |
| --- | --- |
| Pireus model | `ee4589ab4dad2a47a136629dcab6e93aa2f215cf114a8e2f7b3f24a89d39ed9d` |
| target profile | `d41726a8a7eba62132e3763cf6a71938de746ec9d58ce8a20caa40709546d6a4` |
| execution engine | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| XED importer | `c65d63a490038d874f9d1ae34458ff44793049eb7ec01bee01981df7974cbeb9` |
| AARCHMRS importer | `ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b` |
| PTX importer | `ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba` |
| Apple Metal importer | `b43f48c723283d65c3e1df1824f6284303a71967e20deab2c9fe8c7b72f97587` |

## Material Inputs

The four vendor corpora were already pinned by their local Sounio authority
receipts. This composition revalidated their complete digests in Sounio:

| Corpus | SHA-256 | Digest match |
| --- | --- | ---: |
| Intel XED AVX-512 foundation slice | `e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038` | 1 |
| Arm AARCHMRS `Instructions.json` | `bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b` | 1 |
| NVIDIA PTX 9.2 HTML | `fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457` | 1 |
| Apple `mtlgpufamily.json` | `f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea` | 1 |

The corpora remain outside Git. Chunk splitting is byte transport only;
Sounio reconstructs and hashes the complete AARCHMRS and PTX inputs.

## Toolchain

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
```

The exact record above, including final `LF`, hashes to
`2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e`.

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

The exact record above, including final `LF`, hashes to
`fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0`.

## Commands

```bash
./bin/souc check examples/pireus_graph_identity_composition.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_graph_identity_composition.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_graph_identity_composition.sio \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt \
  /tmp/pireus-aarchmrs-2025-12/chunks/part- \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part- \
  /tmp/pireus-apple-metal-20260521/mtlgpufamily.json

/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_graph_identity_composition.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/pireus_graph_identity_composition.sio -- \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt \
  /tmp/pireus-aarchmrs-2025-12/chunks/part- \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part- \
  /tmp/pireus-apple-metal-20260521/mtlgpufamily.json
```

The exact direct-run command as one line with final `LF` hashes to
`8756c380f55dfb37a60beaba08fcdd9605715a1ceb6b7ba2f6864cf3c6889237`.
No raw ELF was invoked.

## Compiler Resolution

The default public wrapper resolved to Madaros v0.80.0. Its advisory science
boundary was `UNKNOWN`, and its check failed before execution with:

```text
run_check_mode: AST closure incomplete nodes=11
 unresolved=0
 saturated=false
```

This is classified as a current Madaros closure-parser limitation on the large
multi-module witness. It produced no result and no fallback semantics.

The explicit `lean_single` path through the same public wrapper typechecked and
executed successfully. The rebuilt/current-source ontology wrapper check
returned:

```text
verdict=ok
provenance=rebuilt_direct
resolution=unanimous
```

Its run completed successfully and its `SOUNIO_AUTHORITY` suffix was identical
to the direct authority stream.

## Sounio-Produced Result

```text
producers=7
namespace_owners=8
term_references=805
canonical_lifted_triples=650
source_occurrences=1621
inherited_occurrences=971
canonical_triples_with_multiple_occurrences=290
collision_keys=24
collision_owner_pairs=24
derived_7033xx_collision_keys=6
canonical_targets=3
target_source_occurrences=18
machine_profile_typed_join=5
apple_blueprint_links=1
forward_reverse_equal=1
parent_closures_verified=1
negative_witnesses=26/26
failures=0
```

The complete collision-key set is:

```text
703000 703001 703002 703003 703004
703100 703101 703102 703103 703104
703200 703210 703211 703220 703221 703230 703231 703240
703300 703301 703302 703303 703304 703310
```

Every row is IRI sort `1` and maps owner `4` and owner `6` to distinct lifted
identities. For example:

```text
local=703000 owner_a=4 lifted_a=8703000 owner_b=6 lifted_b=12703000
local=703300 owner_a=4 lifted_a=8703300 owner_b=6 lifted_b=12703300
```

The Sounio digests are:

```text
registry=9b56f6f0306d949e2266776ee34f05f3ba1dec4239e0bba9411b3aed9c2b27ce
dependency=4dd37bf1cdd774e4ab840e5444d7b18b8a1d0990063901b8a85743a7ac2abbcc
lifted_graph=0bcf3ef8b9598cb4363864d9ba75d9b050a22df501b80a09eda7290b3e331765
occurrence=57218fbb4a6d640e4651dea0d14a17a54559a2f559e45e3186a46df7d8a05950
collision=3a72cc5158aa0e841b4b13de2a924d1bca516778b651ae3f1fe9be80d26925bb
provenance=1e962677cfb1846a5e5b9dd70c13c25cae5f92ad905f6ad795a8912b4e352f20
```

Two pre-freeze runs and the post-freeze run were byte-identical. The exact
stream has 263 lines, 4,813 bytes, and SHA-256
`5b3efa606d86805aa222ced72a37ed87e7b3dab66b21e58e0547163aa19c83dd`.

## Negative Surface

Sounio rejected or distinguished all 26 cases required by the frozen
contract, including producer/namespace duplication, namespace overlap,
parent-graph defects, closure mutation, unregistered terms, invalid shared or
foreign claims, owner rebinding, incompatible/double lifted identities,
IRI/literal swaps, typed-join sort reuse, missing literal owners, changed value
bits, overflow, capacity, inherited query inflation, occurrence loss,
declaration-only collision census, provenance loss, and local-query promotion.

Reverse graph arrival produced all six identical digests. Parent processing
remained topological and deterministic.

## Loom Admission

The operational realization was the shared native runtime for the frozen
Sounio Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
```

The `GARDEN -> SOUNIO_EXECUTABLE` frame bound the executable source,
toolchain, hardware, and command records and returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE
decision_sha256=2d490815fb2e56b74303a46cda871f5e28eefa053f5d5dbc0e701cdf97fab266
```

The later frame used:

```text
schema=9020
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

Its six required receipt bindings were:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `6708c1b970b6d0961a1c338252dcd55d9499e16249bafb42373fdc6df7651898` |
| frozen semantics | `8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb` |
| toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| command record | `8756c380f55dfb37a60beaba08fcdd9605715a1ceb6b7ba2f6864cf3c6889237` |
| Sounio result | `5b3efa606d86805aa222ced72a37ed87e7b3dab66b21e58e0547163aa19c83dd` |

Loom returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

The pre-commit frame at `SEMANTICS_FROZEN` with the same complete bindings was
also allowed.

## Deliberate Python Refusal

The same freeze bindings were presented with `language=7 Python` and
`role=7 PROHIBITED`. The guardian refused the frame before any interpreter or
requested effect could run:

```text
exit_code=110
SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

The earlier pre-executable Python frame was likewise denied and kept the next
stage at `GARDEN`.

## Evidence Boundary

This receipt establishes only collision-free global identity, exact parent
closure, canonical graph composition, provenance retention, literal shape, and
typed composed joins.

It emits and claims:

```text
instruction_equivalences=0
processor_observations=0
capability_inheritance=0
lowering_choices=0
performance_claims=0
parity_open=0
claim_ready=0
```

Apple Silicon and DGX are canonical targets, not observations. All recorded
Darwin CPU profiles remain Xeon. The frozen importers and generic ontology
query kernel were intentionally kept unchanged.

## Reporting Contract

Files changed by the executable and freeze phases:

```text
stdlib/hardware/pireus/graph_identity.sio
examples/pireus_graph_identity_composition.sio
docs/internal/concepts/pireus-graph-identity-composition.md
docs/research/pireus_graph_identity_composition_semantics.md
docs/internal/concepts/registry.tsv
docs/governance/frozen-repo-docs.v1.json
docs/governance/topic-registry.v1.json
docs/governance/DOCS_AUTHORITY_MATRIX.md
docs/governance/DOCS_ACCEPTANCE_REPORT.md
```

Validation paths used:

```text
default_path=Madaros-check-failed-before-execution
rebuilt_wrapper=check-and-run-pass-provenance-rebuilt_direct
authority_path=public-wrapper-with-explicit-lean_single-check-and-run-pass
fallback_semantics=none
legacy_paths_kept=yes
```

Remaining blocker: the default Madaros closure parser does not yet accept this
large imported witness. That compiler limitation does not weaken or broaden
the frozen Sounio result. Formal, effect, material, and optional denotational
parity remain unopened future work.

LLM-offload reviews invoked: none. This is an internal ontology composition
artifact without mathematical, clinical, or external-facing claims.
