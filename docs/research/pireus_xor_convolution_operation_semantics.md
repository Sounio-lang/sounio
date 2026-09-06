<!-- docs:meta
topic_id: repo.docs.research.pireus-xor-convolution-operation-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-xor-convolution-operation-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus XorConvolution Operation DAG

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Chain

The byte-exact Garden parents are:

| Garden record | SHA-256 |
| --- | --- |
| `docs/internal/garden/seeds/2026-08-27-pireus-xor-convolution-operation.md` | `d87fd24678611877846e8b36c5ca2c70a8fcf4033f40032ef3dee603b2bc6d88` |
| `docs/internal/garden/seeds/2026-08-27-pireus-xor-operation-parent-link.md` | `02c72dd3aa71debbdf36a8125c744a16d6bf99d472ea5a9f6208f73757a4cd5e` |

Commit `5329b82de9` contains the first executable Sounio operation graph. It was
committed without this document and without the frozen-result predicate. The
values below were emitted before the matcher existed. The matcher and this
semantic document were added only after two byte-identical Sounio executions.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

Parity and claim promotion remain closed.

## Frozen Parents

The live XorConvolution parent is pinned by semantics SHA-256:

```text
da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502
```

The graph-identity parent is bound through its frozen Sounio receipt:

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

The receipt path is a bootstrap accommodation, not a second graph-identity
implementation. Importing both frozen modules into one `lean_single` bundle
collides on flattened private helper names. The available Madaros prebuilt is
v0.80 and is not evidence against the frozen parent.

## Exact Operation Graph

The admitted path is:

```text
1 XOR_PERMUTE        capability=LANE_XOR_PERMUTE   fixed_order=false barrier=false
2 TWIST_APPLY        capability=PER_TERM_SIGN      fixed_order=false barrier=true
3 MULTIPLY           capability=FLOAT_MULTIPLY     fixed_order=false barrier=false
4 HORIZONTAL_REDUCE  capability=FIXED_ORDER_REDUCE fixed_order=true  barrier=true
5 OUTPUT_LANE        capability=OUTPUT_MATERIALIZE fixed_order=true  barrier=true
```

The four edges are `1->2`, `2->3`, `3->4`, and `4->5`. The overlay graph ID is
`7`, the overlay owner ID is `9`, and the nonassociative barrier count is `3`.

The operation preserves the parent contract:

```text
bits=4
dimension=16
accumulation_order=ascending i for each output d
output_matches_parent=true
```

## Exact Output

The exact IEEE-754 `f64` bits for lanes `0..15` are:

```text
 4613118981945187609
-4611510553506841004
-4614727410383534212
-4608966312158910915
 4604258003457569030
-4604813642372634225
 4612475610569848966
 4616423571282154268
-4600193066131565794
 4615458514219146306
-4606699890268513426
-4603058993167165201
-4605968786432901330
-4614201015621893512
-4608907823852061948
-4604608933298662840
```

The parent classification is retained exactly:

```text
zero_free=true
normalized=true
displacement_only=false
rank_one_separable=false
left_square=true
group_two_cocycle=false
associator_defect_count=1848
wht_rewrite_authorized=false
```

These are finite Sounio-produced fields consumed from the hash-pinned parent,
not identities independently proved by this operation document. The parent
defines `associator_defect_count` as the cardinality of

```text
{(i,j,k) in [0,15]^3 :
  sigma(i,j) * sigma(i XOR j,k)
    != sigma(j,k) * sigma(i,j XOR k)}.
```

`zero_free`, normalization, displacement dependence, rank-one separability,
the left-square law, and the standard group 2-cocycle law use the complete
finite domains recorded in the frozen parent semantics. This operation binds
and checks that result; it does not substitute a new derivation.

`wht_rewrite_authorized=false` is an epistemic gate. It records that no
transform-identity receipt exists and makes no impossibility or complexity
claim.

## Canonical Targets And Material Boundary

The target declarations are:

| Target | Canonical | Observed | Evidence role |
| --- | --- | --- | --- |
| Darwin Xeon | true | false | declared canonical |
| Apple Silicon | true | false | declared canonical |
| DGX | true | false | declared canonical |

The result contains zero material observations, zero lowerings, zero cost
records, and zero material receipts.

## Negative Surface

All 20 deliberate negatives pass. They cover absent and mismatched parents,
invalid bit width, corrupted topology, missing capabilities, changed
accumulation order, false displacement/separability/cocycle/associativity/WHT
promotions, missing canonical targets, observation and material claims without
receipts, parity before freeze, and claim promotion before parity.

The graph masks, bit/dimension relation, accumulation order, parity transition,
and claim transition are exercised through validators and mutated inputs. They
are not tautological booleans.

## Frozen Digests

All digests were computed inside Sounio over explicit canonical fields:

| Object | SHA-256 |
| --- | --- |
| parent bindings | `12686c8de11d8eb6fe422085ee39b983045c81ff26e6967813aa1543e5b586e5` |
| operation graph | `ce636b3bbaea074e2033b6656120256f53b75d00b0958d57269a63d836a90bfa` |
| capabilities | `b8f34a86e3fd7e22225c755528e160f4f5eabf1b8d0c57e473a0455064db8ebe` |
| targets | `38f74af02731ec3c2ac6ac1ce57659cde2351c157726dcb19aa8d6163d7282c6` |
| result | `84edf6bae148754ebd0e8722368e2eb06095cd929779c36def4f3bb5000013a3` |
| negative witness | `d726dd98af6b52258f91709d8df3f050eee6b4944e19e67743707ef1d2a08a23` |

The pre-freeze authority stream is 5,043 bytes over 204 lines and has SHA-256
`9fef54b41f4089ba25ccbbbcaea50ab1ec16f981fed41e163de0ac1877acf39b`.

## Review-Only Boundary

xAI reviewed the gate structure after Sounio produced the result. It found and
helped tighten vacuous negative surfaces, the missing transform-identity
receipt requirement, and the distinction between mathematical legality and
epistemic authorization. Z.AI returned empty artifacts twice and three fallback
providers failed externally, so the review is single-provider degraded.

No model created or confirmed output bits, counts, digests, or expected
results. Sounio remains semantic authority.

## Non-Claims

The frozen semantics do not claim:

- an ISA instruction mapping or coverage result;
- `vpermps`, `vpermi2ps`, NEON, SVE, SME, Metal, PTX, or SASS suitability;
- an instruction count, cost, throughput, latency, or speedup;
- a material observation on Darwin Xeon, Apple Silicon, or DGX;
- a Fano-plane explanation of the seven-negative displacement rows;
- Walsh-Hadamard diagonalization or a subquadratic algorithm;
- parity in Lean 4, Koka, C++, or Haskell.

The next legal stage remains `SEMANTICS_FROZEN`. `PARITY_OPEN=false` and
`CLAIM_READY=false` until separately admitted by Loom.
