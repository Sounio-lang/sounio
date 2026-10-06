# XorConvolution Sounio Authority Receipt

> **Date**: 2026-08-27 | **Stage**: `SEMANTICS_FROZEN`
>
> **Language**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Scope

This receipt binds the first executable semantics for
`SOUNIO-XOR-CONVOLUTION-COCYCLE`. It establishes the direct-pair and
displacement-reduction indexing forms, the finite `bits=4` Cayley-Dickson twist
classification, exact result bits, negative witnesses, and Sounio-produced
digests.

It does not open parity, select a material lowering, or promote a performance
claim.

## Ordered Commit Chain

| Stage | Commit | Content |
| --- | --- | --- |
| `GARDEN` | `59b3772dc2` | founder-authorized boundary with no expected result |
| `SOUNIO_EXECUTABLE` | `aada53b483` | first Sounio operation, classifier, witness, and tests |
| `SEMANTICS_FROZEN` | `59805b8712` | post-result matcher and frozen semantic documents |

The first executable commit is descended from the Garden commit. The matcher
containing expected values appears only in the later freeze commit.

## Frozen Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Garden seed | `246651b5804f8f24ddc6a8292d898db5483a7495f355c53fb1c4b50b7fb62e80` |
| first executable module | `f74582098c661bae44d62f46707aff440fc0fc13bca9510f6c62ab2b507bf817` |
| first executable entrypoint | `a843df000761a59c4a7dbf0d8d9e487094200275ad6a75fde32df633fa2ab0d4` |
| frozen module | `5454dac832feff899a05a64d4573be9be5192b08a4c28f869074773bef59151a` |
| frozen entrypoint | `7ecff7fae2f8c40bb4e7109bd08e315adc5981d6fb5d4fe04cbf9822a9cd126f` |
| concept contract | `0f083bafb17683ce3291eee10ed486c1e890dbe3c7c7a37db0439fc3e674ac5f` |
| frozen semantics | `da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502` |
| dedicated test source | `1160a4804eb63c966032aa6ade363121efb231cd859d33f6f40c6f47a00575c2` |

The untracked founder research note in the promoted workspace was observed at
SHA-256
`533b8aa9e407f16848e2e554da45d1111b90a18927884bda4be03da3c3461bbe`.
It motivated the Garden and did not supply an expected executable result.

## Toolchain Receipt

The semantic-authority run used the public wrapper, never a raw ELF:

```text
engine=lean_single
wrapper=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
```

The SHA-256 of that exact five-line record is
`0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556`.

This is the explicit `lean_single` authority path. The default Madaros path,
rebuilt current-source ontology wrapper, and fallback path were not used to
create the result.

## Hardware Receipt

```text
kernel=Linux 7.0.2-5-pve
architecture=x86_64
logical_cpus=64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
sockets=2
cores_per_socket=16
```

The SHA-256 of that exact six-line record is
`c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d`.

This machine has AVX-512 capability, but this receipt contains no observation
that the emitted program used AVX-512.

## Command Receipt

The authority command was:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/xor_convolution_cocycle.sio
```

Its newline-terminated command record has SHA-256
`a86d8723efc1c897d336f9b3712dc800e45cbc0189e3d37c08c327a0bcde3ac0`.

The dedicated semantic test command was:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/algebra/test_xor_convolution.sio
```

It ran only after a Loom `CI` pre-action frame at `SEMANTICS_FROZEN` returned
ALLOW.

## Sounio Result

The authority stream is 164 lines and 3443 bytes. Two pre-freeze executions
and one post-freeze execution were byte-identical. Its SHA-256 is:

```text
99fec6de74e2f19c6ce53a480fffd2e861dc5caf6d0a31655c92c4ca48bde5aa
```

The post-freeze matcher therefore did not change the result stream.

Sounio produced:

```text
bits=4
dimension=16
mismatch_count=0
first_mismatch=-1
max_abs_difference_bits=0

plus=136
minus=120
zero=0
other=0
zero_free=1
normalized=1
displacement_only=0
rank_one_separable=0
left_square=1
group_two_cocycle=0
associator_defects=1848

negative_signs_by_displacement=
15 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7

negative_tests=20/20
failures=0
parity_open=0
claim_ready=0
```

Here `associator_defects` counts all ordered triples `(i,j,k)` in
`[0,15]^3` for which

```text
sigma(i,j) * sigma(i XOR j,k)
  != sigma(j,k) * sigma(i,j XOR k).
```

The left-square enumeration covers every `i` in `[1,15]` and every `j` in
`[0,15]`. The unit index `i=0` is deliberately excluded; normalization handles
that row, and no full-alternativity claim is inferred.

The exact result-bit vector is frozen in the semantic document and in the
post-result Sounio matcher.

## First Counterexamples

Sounio produced the first failures in its fixed enumeration order:

```text
displacement-only:
  i=1 j=1 d=0 lhs=-1 rhs=+1

rank-one separable:
  i=1 j=1 d=0 lhs=-1 rhs=+1

standard group 2-cocycle:
  i=1 j=2 k=4 d=7 lhs=+1 rhs=-1
```

The last witness establishes only that this selected `bits=4` sign table is
not a group 2-cocycle; it classifies neither other bit widths nor cohomologous
tables. The repository's left-square identity still holds for every
enumerated nonzero `i` and every `j`. These predicates are distinct.

## Sounio Digests

| Object | SHA-256 |
| --- | --- |
| contract | `486ef520df7669f360be3c531ea5fad28e0e70ea793e6d53ef6b9d34c1090856` |
| inputs | `4d4152b488cb59c4e451af1d3255d077bd5c708c88aabf46fc334a47522c1039` |
| direct result | `2aad13bb99d7f04fcc1116036ccdd2b47abee50aba7c4d8eb2801c40e0c07b6c` |
| horizontal result | `2aad13bb99d7f04fcc1116036ccdd2b47abee50aba7c4d8eb2801c40e0c07b6c` |
| twist table | `1c09a640d55cc98cfb9c51a5144dd28d1a0c5dd4ff9e8fc4a7e9d7b189cdc014` |
| properties | `f24a6f2f6c8d0c3c77440f54d6c2e683ba413c9e56d81a7f0aa08f1d84edf928` |
| witnesses | `42155cb1da7c3564bda5a83fc98d02040200095ee54902bbe5c87f96e9e3e18e` |

The equality of direct and horizontal result digests is Sounio-produced
evidence, not an external comparison.

## Loom Admission

The operational guardian was the installed native realization of frozen Sounio
Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
```

The `GARDEN -> SOUNIO_EXECUTABLE` pre-action frame bound the first executable
source, toolchain, hardware, and command records. Loom returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE
decision_sha256=2d490815fb2e56b74303a46cda871f5e28eefa053f5d5dbc0e701cdf97fab266
```

The freeze frame used:

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
| frozen Sounio entrypoint | `7ecff7fae2f8c40bb4e7109bd08e315adc5981d6fb5d4fe04cbf9822a9cd126f` |
| frozen semantics | `da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502` |
| toolchain record | `0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556` |
| hardware record | `c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d` |
| command record | `a86d8723efc1c897d336f9b3712dc800e45cbc0189e3d37c08c327a0bcde3ac0` |
| Sounio result stream | `99fec6de74e2f19c6ce53a480fffd2e861dc5caf6d0a31655c92c4ca48bde5aa` |

Loom returned:

```text
SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

Separate pre-action frames also allowed the executable commit, freeze commit,
and dedicated Sounio CI test. No parity transition was requested.

## Deliberate Python Refusal

Before the first Sounio execution, the same source, toolchain, hardware, and
command bindings were presented as `language=7 Python`, `role=7 PROHIBITED`.
The guardian refused the frame before any Python interpreter could run:

```text
exit_code=110
SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=GARDEN
decision_sha256=30b2f51d293ccfe005319d25d6c5179cd6fc015ae7d0e3ec5a11b162f9c7def7
```

No Python or Rust program created, checked, or confirmed the semantic result.
Node was used only by the existing deterministic documentation-metadata
generator after the semantic documents existed; it computed no mathematical
value or authority decision.

## Review-Only Offload

xAI/Grok 4.5 performed three `math-review` passes after the Sounio executable
had produced its result:

- the Sounio module and predicate definitions;
- the frozen Garden wording;
- the frozen semantic document.

The reviews found no false identity. Their quantifier and defect-domain
precision suggestions were carried into the semantic document. They created no
expected result and cannot confirm authority. The append-only review record is
in `.claude/llm_offload_log.md`.

## Validation

Passed:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/xor_convolution_cocycle.sio
cmp pre-freeze-run-a pre-freeze-run-b
cmp pre-freeze-run-a post-freeze-run
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/algebra/test_xor_convolution.sio
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
bash scripts/dev/sounio_semantic_status.sh
git diff --check
```

The dedicated test printed `XOR_CONVOLUTION_OK` and exited zero. The semantic
status scanner reports the concept as executable and reports no runtime alert.

The broader `scripts/ci/semantic_coordination_gate.sh` remains red on the same
pre-existing missing docs-governance registration for
`docs/internal/concepts/falsification-carrying-development.contract`; the
identical failure occurs on the clean parent worktree and is baseline noise for
this lane.

## Legacy And Remaining Boundaries

`stdlib/algebra/cayley_dickson.sio` was intentionally retained unchanged and is
the canonical source of `cd_sigma`. The new module is an operation contract and
classifier over that function, not a replacement Cayley-Dickson algebra.

Still closed:

- Pireus operation-node graph and global graph-identity occurrence;
- Lean 4, Koka, C++, and optional Haskell parity;
- default Madaros validation and current-source material parity;
- x86, AArch64, Apple Silicon, Metal, PTX, or SASS lowerings;
- emitted-instruction and performance measurements;
- Fano interpretation of the uniform nonzero displacement counts;
- any subquadratic algorithm for the nonseparable, nonassociative twist;
- native no-Python hook migration outside the installed Loom policy runtime.

`PARITY_OPEN=false` and `CLAIM_READY=false`.
