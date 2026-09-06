<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-xor-material-matching-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-xor-material-matching-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XOR Material Matching Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-MATERIAL-MATCHING`

Semantic-Lane-ID: `pireus-xor-material-20260827`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

The frozen selector plan projects the bits=4 `f64x16` XOR partner permutation
onto two abstract `f64x8` chunks:

```text
i            = 8*c + l
j            = i XOR d
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

Sounio enumerated 256 cells, reconstructed all 256 partner indices, and
accepted all 32 fixed `(d,c)` one-source groups. All 21 negative witnesses
passed. The result contains zero material observations, zero instruction
matches, zero lowerings, and zero cost records.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden admission | `562c8cb06f` | admitted the selector-plan question |
| XED lineage correction | `9e8e9884dc` | separated frozen content from the historical docs envelope |
| first executable source | `eedc47c8fe` | created the Sounio evaluator before expected values existed |
| operation hash correction | `860fb5cea3` | fixed a mis-grouped parent SHA limb; prior stream discarded |
| claim-gate tightening | `2dba5b3022` | routed promotion negatives through the accepted-result gate |
| live XED derivation | `d9e99cc620` | reloaded and queried the digest-pinned vendor corpus in Sounio |
| provenance narrowing | `927800c60d` | removed overbroad evidence names |
| semantics freeze | `fcbf9bc00a` | added matcher, semantics, registry and frozen docs entries |

Every semantic source change invalidated the preceding pre-freeze stream. The
stream recorded below is the final one. Adding the matcher did not change it:
the post-matcher execution was byte-identical.

## Frozen Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Garden plus XED-lineage addendum | `c07d58a40a457b8fa3ce524eb625fa17ffdbdf04e521a584cc20dba4eb4c13f1` |
| combined Sounio source record | `387f726e9cb90f123f275846f7841992f6dfc76c25638a1fb101d84de1c8c4ec` |
| Sounio module | `eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575` |
| authority executable | `370729af65a388986a09dcf0055839ecb737c44534b6e58c24f16a027095cc12` |
| Sounio test | `46c0e02ad339d28c5c7c79b0bcb13aafbd9125a13b376b5d2968ce87ddc47760` |
| concept contract | `a9a76ae70a67cb599f9fd517bdf4f5b3aa7125c6a93baec8cc833b33c403f1a2` |
| frozen semantics | `b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554` |

The source record is SHA-256 over the module, executable, and test bytes in
that order.

## Frozen Parent

```text
operation_semantics_sha256=40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
operation_receipt_sha256=9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330
operation_result_digest=84edf6bae148754ebd0e8722368e2eb06095cd929779c36def4f3bb5000013a3
```

The child live-imported and matched the complete frozen operation result and
separately compared its live result digest with the value above.

## XED Input And Lineage

```text
upstream=https://github.com/intelxed/xed
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
path=datafiles/avx512f/avx512-foundation-isa.xed.txt
bytes=458470
sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
```

The execution read the pinned checkout path:

```text
/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
```

The Sounio importer checked the vendor digest, parsed the corpus, rebuilt its
ontology, and queried:

```text
total_forms=8
vpermpd_forms=4
vpermi2pd_forms=2
vpermt2pd_forms=2
```

The bound lineage hashes are:

```text
xed_frozen_content=5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612
xed_current_historical_envelope=d96d6d57ba1e296930caec5f4f0aff8e2898b3b1d5df6bfaacb96a19333266f7
xed_current_receipt=2dfc243381acb8d365112b3b4075ccabf944de6ff081b4626f9a4f693f136af6
```

These facts establish form presence and raw grammar only. They do not
establish selector behavior or instruction equivalence.

## Toolchain, Hardware, And Command

```text
toolchain_record_sha256=0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556
hardware_record_sha256=c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d
public_wrapper=./bin/souc
selected_engine=lean_single
control_cpu=Intel Xeon Gold 6526Y
```

The exact authority command was:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_xor_material_matching.sio \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt \
  > /tmp/pireus-xor-material-authority.v0.txt
```

Command-record SHA-256:

```text
ea26d11c329471903aa7c68f94e84b8047b151de1c3004152c3da19151ae823e
```

This used the public wrapper with the explicit bootstrap seed. It did not use
the default Madaros path or a fallback result.

## Authority Stream

```text
lines=2675
bytes=35870
sha256=009f7fed37b8d20909c2a4915d9231a3416eaba41998b4290e3787274e85537f
valid=1
failures=0
```

The four Sounio digests are:

| Object | SHA-256 |
| --- | --- |
| selector cells | `fd515df6be12f316cc771cf043aa0220bae7556362074a12eefe7f5a8f718928` |
| parent and XED evidence | `824e0d324a62a817f2d116f51e6b909534add1a3bae46a093dc09139d1230774` |
| target declarations | `3e2e2f881cc866c4a99b6b95ea933153df35e0472b42f51f9a0c5a2f0cb975b5` |
| negative witnesses | `95f189ec40d69f9b2dcc3263557e6076e92324200e5cc7ccb7c525fca19e82bb` |

## Canonical Targets

| Target | Canonical | Candidate plan | Observed | Material receipt |
| --- | --- | --- | --- | --- |
| Darwin Xeon | true | two abstract f64x8 chunks | false | false |
| Apple Silicon | true | unresolved | false | false |
| DGX | true | unresolved | false | false |

Canonical declaration was not promoted to observation.

## Loom Decisions

The Sounio language-authority runtime admitted Sounio execution, tests,
documentation gates, review-only offload, the exact freeze frame, and the
freeze commit. The final transition bound the source, semantics, parent,
toolchain, hardware, command, and result hashes above and returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
```

A deliberate would-be Python producer was framed before any interpreter
launch. Loom returned:

```text
exit_code=110
SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
interpreter_launch_count=0
```

The present Python executable was not launched. Rust was not used.

## Validation

| Gate | Result | Classification |
| --- | --- | --- |
| authority executable | exit 0 | Sounio semantic-authority stream |
| post-matcher executable | byte-identical | frozen matcher did not create the result |
| dedicated Sounio test with pinned XED input | `PIREUS_XOR_MATERIAL_EXECUTABLE_OK` | accepted |
| docs registry and selftest | pass | governance accepted |
| docs consistency | pass | governance accepted |
| semantic status | concept executable, no runtime alerts | observational scanner |
| Python producer attempt | E110 before launch | prohibited language refused |

The test without the external pinned XED path reports that the input is
required and exits without claiming validation. The receipt-bearing test used
the exact path shown above.

## Review-Only Offload

xAI/Grok 4.5 performed three executable-review passes and one receipt-review
pass after Sounio produced the result. It found no wrong XOR packing identity.
It required several useful
tightenings: claim negatives now share the accepted-result gate; XED counts
are derived from digest-matched vendor bytes; the live operation digest is
compared explicitly; and provenance names no longer claim that Markdown was
decoded or that the local role classifier replaces Loom.

The receipt review found no contradictory or overbroad mathematical claim.
The review remained single-provider degraded. No external model created or
confirmed selector rows, counts, target states, digests, or expected values.

## Closed Claims And Next Gate

This receipt does not authorize:

- selector behavior for any XED form;
- an instruction match for `VPERMPD`, `VPERMI2PD`, or `VPERMT2PD`;
- immediate-form sufficiency or two-source necessity;
- emitted AVX-512 on Darwin;
- any Apple Silicon or DGX observation;
- lowering of twist, multiply, fixed-order reduction, or output;
- an instruction count, cost, latency, throughput, or speedup;
- a Walsh-Hadamard or subquadratic claim;
- Lean 4, Koka, C++, or Haskell parity.

The next legal step is a new Garden for a pinned selector-semantics receipt.
`PARITY_OPEN=false` and `CLAIM_READY=false` remain frozen.
