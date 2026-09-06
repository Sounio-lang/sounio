<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-xor-lowering-legality
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-xor-lowering-legality
-->

# Pireus XOR Lowering Legality

Concept-ID: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-xor-lowering-legality-20260827`

## Intent

Define the complete bits=4 XorConvolution lowering-legality boundary before
any target implementation is allowed to measure or claim a full lowering.

The semantic producer is Sounio:

```text
stdlib/hardware/pireus/xor_lowering_legality.sio
examples/pireus_xor_lowering_legality.sio
tests/stdlib/hardware/test_pireus_xor_lowering_legality.sio
```

## Causal Parents

The executable binds and live-validates three frozen Pireus parents:

| Parent | Source SHA-256 | Semantics SHA-256 | Receipt SHA-256 |
| --- | --- | --- | --- |
| operation DAG | `bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8` | `40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1` | `9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330` |
| XOR material layout | `eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575` | `b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554` | `cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df` |
| Intel selector semantics | `4f7e007aa432564b873c941e239c353aeed3b11883844b73ec7e31ace4811b20` | `ba25ceb18685ed656ecaf1c577eb95a698ca214fe2a939fce5a8ffd6d106b243` | `fddad1442d0b21201bccf57fce380a2d57a94bb55bee9924636d06473128218f` |

No one parent owns a complete lowering. The child refuses missing, mismatched,
or non-live parent bindings.

## Frozen Semantic Schedule

| Node | Semantic form | Groups | Exact order | Material form |
| --- | --- | ---: | --- | --- |
| `XOR_PERMUTE` | indexed chunk map | 32 | preserved | Intel vector-control candidate |
| `TWIST_APPLY` | coefficient sign map | 32 | preserved | unresolved |
| `MULTIPLY` | ordered `f64` product | 32 | preserved | unresolved |
| `HORIZONTAL_REDUCE` | ascending-`i` fold | 16 | preserved | unresolved |
| `OUTPUT_LANE` | output-lane store | 16 | preserved | unresolved |

All five semantic forms are authorized by the frozen DAG. Zero material forms
are authorized. The Intel form remains a local candidate for `XOR_PERMUTE`,
not a compiler emission or whole-operation lowering.

## Sign Plan

Sounio derives 32 eight-lane sign groups from
`sigma(i, i XOR d)`, where `sigma` is the frozen Sounio
`algebra::cayley_dickson::cd_sigma(i,j,4)` function. The complete 256-cell
table contains 120 negative and 136 positive coefficients, with no zero or
other coefficient.

The masks are frozen as semantic coefficient masks. They do not select an ISA
sign-bit instruction, arithmetic negation instruction, or target encoding.

## Numeric Boundary

The executable validates every material partner identity:

```text
partner[d,i] = i XOR d
```

It then evaluates each output lane in ascending `i` order using the derived
sign mask and obtains bit identity with the frozen Sounio parent in all 16
lanes. This is `bit_exact_to_frozen_parent`, not equality over real numbers and
not permission to reassociate IEEE-754 addition.

A tree reduction is refused as an exact lowering. A future reassociated plan
requires a separately admitted numerical or refinement contract.

## Canonical Targets

| Target | Canonical | Materially observed | Result in this lane |
| --- | --- | --- | --- |
| Darwin Xeon | true | false | Intel selector candidate only |
| Apple Silicon | true | false | unresolved |
| DGX | true | false | unresolved |

All Darwin CPUs in the frozen target profile are Xeon. Canonical declaration
is not observation.

## Closed Claims

This freeze does not establish:

- compiler emission or an instruction sequence;
- an instruction count, including the earlier estimate near 112;
- AVX-512 execution on Darwin Xeon;
- a material sign, multiply, reduction, or output form;
- latency, throughput, scheduling, register pressure, cost, or speedup;
- Apple Silicon or DGX lowering semantics;
- cross-ISA equivalence;
- Walsh-Hadamard diagonalization or subquadratic twisted convolution;
- a Fano-plane explanation of sign regularity;
- Lean 4, Koka, C++, or Haskell parity.

The external Loom guardian remains the stage and producer-language authority.
`PARITY_OPEN=false` and `CLAIM_READY=false`.
