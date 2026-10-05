<!-- docs:meta
topic_id: repo.docs.research.pireus-xor-lowering-legality-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-xor-lowering-legality-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus XOR Lowering Legality

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

```text
GARDEN commit=3eff32209f18da9b10d679c75230e4cbc7a2ca7b
SOUNIO_EXECUTABLE commit=2f23d18a2725b0b1fdc750439a1580421d177b6a
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The Garden existed before the child source. The first Sounio stream emitted
the plan, sign masks, counts, and digests before any expectation or prose
recorded them.

## Frozen Parent Closure

```text
operation_source_sha256=bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8
operation_semantics_sha256=40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
operation_receipt_sha256=9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330
material_source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
material_semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
material_receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
intel_source_sha256=4f7e007aa432564b873c941e239c353aeed3b11883844b73ec7e31ace4811b20
intel_semantics_sha256=ba25ceb18685ed656ecaf1c577eb95a698ca214fe2a939fce5a8ffd6d106b243
intel_receipt_sha256=fddad1442d0b21201bccf57fce380a2d57a94bb55bee9924636d06473128218f
parent_bundle_sha256=012126771572d6634f2255c606d6a5315953e1874a68e0cfc2b2c2ae0f01b0aa
```

The child runs each parent evaluator and exact frozen matcher. Markdown hashes
are lineage bindings, not substitutes for live Sounio validation.

## Complete Five-Node Plan

| Index | Node kind | Capability | Semantic form | Material form | Groups | Barrier | Refusal |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 0 | `XOR_PERMUTE` | lane XOR permute | indexed chunk map | Intel vector control | 32 | false | target observation missing |
| 1 | `TWIST_APPLY` | per-term sign | coefficient sign map | unresolved | 32 | true | material semantics missing |
| 2 | `MULTIPLY` | float multiply | ordered `f64` product | unresolved | 32 | false | material semantics missing |
| 3 | `HORIZONTAL_REDUCE` | fixed-order reduce | ascending-`i` fold | unresolved | 16 | true | material semantics missing |
| 4 | `OUTPUT_LANE` | output materialize | output-lane store | unresolved | 16 | true | material semantics missing |

```text
semantic_authorized_nodes=5
material_candidate_nodes=1
material_authorized_nodes=0
unresolved_material_nodes=5
reassociation_required_nodes=0
exact_tree_reduction_refused=true
```

The plan preserves every frozen barrier. The Intel candidate covers only the
selector node and remains unauthorized as a target lowering.

## Derived Sign Masks

Each row lists `(displacement, chunk0_mask, chunk1_mask)` in decimal. Bit `l`
is one exactly when `sigma(8*c+l, (8*c+l) XOR d) = -1`.

Here `sigma(i,j)` is exactly the Sounio function
`algebra::cayley_dickson::cd_sigma(i,j,4)` consumed by the hash-pinned
operation parent. The child re-evaluates that function for every cell and
does not replace it with a Fano, separable, or displacement-only rule.

```text
0   254 255
1   104 150
2   194 60
3   164 90
4   14  240
5   84  170
6   152 102
7   50  204
8   254 0
9   148 149
10  56  57
11  82  83
12  224 225
13  138 139
14  38  39
15  76  77
```

```text
groups=32
complete_cells=256
negative_cells=120
positive_cells=136
other_cells=0
```

These are semantic coefficient masks. No target sign mechanism is selected.

## Bit-Exact Ordered Witness

For every `d in [0,15]` and `i in [0,15]`, Sounio first validates:

```text
partner_indices[d*16+i] = i XOR d
```

It then evaluates:

```text
sum = 0
for i in ascending [0,15]:
  coefficient = sign_mask(d, i)
  sum = sum + coefficient * a[i] * b[i XOR d]
output[d] = sum
```

The multiplication association and the addition order match the frozen parent.

```text
partner_table_valid=true
matching_lanes=16
mismatching_lanes=0
first_mismatch=-1
ascending_i=true
reassociated=false
bit_exact_to_frozen_parent=true
```

This finite witness is not a proof over all rounding environments and does not
authorize a SIMD tree reduction.

## Frozen Digests

```text
plan_sha256=7fadb763fd506fb2e6473ae31e17a0a32e20e110ba48e314489dcb028b9ac2b2
signs_sha256=2cd91f8e407ab465fdaa985bb59bb99216501b39b85300c1186d22bf807535bd
execution_sha256=6440e6e81515deb30f38a68e16282051b30a1dd08cc3fb3511b406fd495da1e0
targets_sha256=45ce281bdaf5ab1d78b63261e5bdb028818dfb31ee92a031e65e131c5bb7fc95
witnesses_sha256=c7acb2e78da96b509ea351856e33b9e6d71887cff817f1f0ce2b4df295d5c8fd
```

All 20 in-process negative witnesses pass. They mutate parent bindings, node
shape/order/capability/barriers, sign masks, multiply form, accumulation order,
whole-operation promotion, observation, emission, cost, parity, and claim
readiness. Producer-language refusal remains the external Loom responsibility.

## Canonical Targets And Closed Claims

Darwin Xeon, Apple Silicon, and DGX are canonical and unobserved. Only Darwin
has the local Intel selector candidate. The result contains:

```text
material_observations=0
compiler_emissions=0
cost_records=0
numerical_refinement_receipts=0
parity_open=false
claim_ready=false
```

No instruction count, target execution, performance result, cross-ISA parity,
subquadratic algorithm, Walsh-Hadamard rewrite, or Fano theorem is established.
