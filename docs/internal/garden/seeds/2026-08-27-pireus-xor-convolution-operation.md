<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-convolution-operation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-convolution-operation
-->

# Pireus: Give The Twisted Reduction A Harbor Graph

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Sounio now has two separately frozen semantic parents:

1. a collision-safe Pireus graph identity and composition contract; and
2. a nonassociative, XOR-graded convolution contract whose displacement form
   performs one horizontal reduction for each output lane.

The algebra is executable, but Pireus cannot yet name its internal operation
graph. The target ontologies therefore have no semantic object against which a
future lowering may prove that a lane permutation, sign application,
multiplication, and horizontal reduction realize the same computation.

This Garden opens exactly that missing boundary:

```text
frozen XorConvolution semantics ----+
                                    |
frozen Pireus graph identity -------+-> Pireus operation DAG
                                    |
canonical target declaration -------+
```

The graph is semantic structure, not emitted code. It creates no instruction
selection, cost, performance, or hardware-support result.

## Frozen Parents

The first Sounio executable must bind the byte-exact frozen semantic documents,
not an unversioned concept name:

```text
XorConvolution semantics:
  docs/research/xor_convolution_cocycle_semantics.md
  sha256=da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502

Pireus graph-identity semantics:
  docs/research/pireus_graph_identity_composition_semantics.md
  sha256=8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb
```

It must also consume the Sounio parent APIs rather than reproduce their
algorithms as local expected data:

```text
stdlib/algebra/xor_convolution.sio
stdlib/hardware/pireus/graph_identity.sio
```

A missing parent, a zero parent hash, or a one-bit parent-hash change is an
admission failure. A later receipt may replace either parent only through a new
Garden descendant; semantic drift must not look like ordinary recompilation.

## Operation DAG

For `bits=4`, the admitted graph has five ordered semantic nodes:

```text
XOR_PERMUTE
  -> TWIST_APPLY
  -> MULTIPLY
  -> HORIZONTAL_REDUCE
  -> OUTPUT_LANE
```

The nodes mean:

- `XOR_PERMUTE`: for fixed displacement `d`, select `b[i XOR d]` in ascending
  `i` order;
- `TWIST_APPLY`: apply the selected frozen sign
  `sigma(i, i XOR d)` without assuming separability;
- `MULTIPLY`: form the signed product with `a[i]`;
- `HORIZONTAL_REDUCE`: accumulate all terms for the same `d` in ascending
  `i` order;
- `OUTPUT_LANE`: expose that reduction as `r[d]`.

The graph must preserve this edge order. In particular, it may not move the
twist across the permutation by pretending that the sign is a function of the
output displacement alone, and it may not replace the graph with a
Walsh-Hadamard transform by pretending the twist is rank-one separable.

The first executable must derive its output bits through the frozen
`XorConvolution` API and bind those bits into its operation-graph digest. It may
not copy the frozen result vector into this Garden or introduce a second
implementation of Cayley-Dickson multiplication.

## Algebra Classification

The parent has already established, for the selected `bits=4` table:

```text
zero_free=true
normalized=true
displacement_only=false
rank_one_separable=false
left_square=true
group_two_cocycle=false
```

The Pireus node must carry that classification without widening it. The safe
description remains:

```text
a nonassociative XOR-graded algebra over (Z/2Z)^4,
with a normalized {-1,+1}-valued twist and explicit associator defect
```

The founder's surface spelling `XorConvolution(bits, cocycle)` may remain as a
name, but Pireus must not promote that spelling into a standard group
2-cocycle claim. The associator defect is semantic evidence that affects legal
graph rewrites.

## Capability Boundary

The graph records abstract capability requirements only:

```text
lane XOR permutation
per-term sign application
floating multiplication
fixed-order horizontal reduction
output-lane materialization
```

Those requirements do not identify an instruction. A material realization
must later provide a receipt that binds a target, corpus, instruction form,
operand roles, lane scope, toolchain, hardware, command, and observed result.

Pireus may declare three canonical target families for later matching:

```text
Darwin Xeon
Apple Silicon
DGX
```

All Darwin processors admitted by this project are Xeon. Apple Silicon and DGX
are equally canonical targets. Canonical declaration means only that the target
is in scope; it is not evidence that any target implements the operation, that
two targets are equivalent, or that a lowering is fast.

## First Sounio Result

The first executable must be born in Sounio and produce, without a prewritten
expected result:

- the admitted parent-hash bindings;
- the ordered node and edge records;
- the abstract capability requirements;
- the three canonical target declarations with observation state kept absent;
- the inherited nonassociative classification;
- the operation output bits obtained from the frozen parent API;
- deterministic graph, dependency, capability, result, and witness digests;
- deterministic positive and negative witnesses;
- `parity_open=false` and `claim_ready=false`.

Its result stream must be reproducible byte-for-byte before any expected-value
matcher is written. Only a later `SEMANTICS_FROZEN` commit may contain the
expected counts, words, or digests produced by that execution.

## Required Negative Witnesses

The Sounio authority executable must deliberately reject at least:

1. an absent XorConvolution parent hash;
2. an absent Pireus graph-identity parent hash;
3. a mismatched XorConvolution parent hash;
4. a mismatched Pireus graph-identity parent hash;
5. a bit width other than the admitted `bits=4` projection;
6. a reordered, missing, duplicated, or cyclic operation edge;
7. a missing abstract capability requirement;
8. a changed ascending-`i` accumulation order;
9. a displacement-only sign promotion;
10. a rank-one separability promotion;
11. a standard group 2-cocycle promotion;
12. an associativity or unrestricted reassociation promotion;
13. a direct Walsh-Hadamard replacement claim;
14. a missing canonical Darwin Xeon declaration;
15. a missing canonical Apple Silicon declaration;
16. a missing canonical DGX declaration;
17. a target-support observation without a material receipt;
18. an instruction, lowering, or cost claim without a material receipt;
19. `PARITY_OPEN` before the frozen Sounio artifact exists; and
20. `CLAIM_READY` before parity and material evidence exist.

These are semantic negatives. They are not synthetic performance benchmarks.

## Authority Order

The mandatory order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Lean 4, Koka, C++, and optional Haskell may run
only after the Sounio source and semantics are frozen and identified by hash.
They may prove, compare, or measure; they may not create the operation graph or
its expected result retrospectively. External LLMs are `REVIEW_ONLY`.

Python and Rust are forbidden as semantic or material oracles. Substituting a
disposable Node, Ruby, shell, `awk`, or `bc` computation does not satisfy the
Sounio-first contract.

Before the first Sounio execution, Loom must deny a deliberate Python oracle
frame before interpreter start and allow the exact Sounio source, toolchain,
hardware, and command frame. Loom must fail closed on missing policy, error, or
timeout and record every `ALLOW` or `DENY` reason.

## Claim Boundary

After the first freeze, the lane may claim only:

- Sounio has a deterministic Pireus operation DAG for the frozen `bits=4`
  twisted XOR reduction;
- the DAG preserves the frozen accumulation order and nonassociative
  classification;
- Darwin Xeon, Apple Silicon, and DGX are declared canonical matching targets;
- no material realization has yet been observed.

It may not claim:

- AVX-512, AVX2, NEON, SVE, SME, Metal, PTX, SASS, or any named instruction
  realizes a node;
- a permutation is one-source or two-source on any target;
- the graph has a measured or estimated instruction cost;
- Walsh-Hadamard or another subquadratic algorithm is valid;
- cross-target equivalence, compiler quality, or speedup;
- formal, effect, material, or denotational parity;
- readiness for an external performance claim.

## Exit Gate

This Garden reaches `SOUNIO_EXECUTABLE` only when:

1. the exact Garden hash is recorded;
2. Loom admits the exact Sounio executable frame;
3. the executable imports both frozen parent APIs;
4. all graph objects, results, witnesses, and digests are produced by Sounio;
5. repeated authority runs are byte-identical;
6. all required negative witnesses pass;
7. documentation and semantic registries pass;
8. no parity language has executed; and
9. no material or performance claim has been promoted.

This seed establishes only `GARDEN` for
`SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION`.
