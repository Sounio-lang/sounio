# Pireus Cubic Operator Forge v4 Contract

Status: `SEMANTICS_FROZEN`

Concept-ID: `SOUNIO-PIREUS-CUBIC-OPERATOR-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Contract

Pireus v4 is an executable operator generator. It does not merely enumerate
existing instruction names or select one inherited operator. Starting from the
frozen v3 Operator Genome, it emits a complete, content-addressed population of
48 signed XOR operators from a declared mixed-cubic mutation grammar.

The authority order was observed:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This contract stops at `SEMANTICS_FROZEN`. No Lean, Koka, C++, Haskell, target
lowering, schedule search, or material measurement is admitted by this freeze.

## Chronology

The Git history is part of the receipt:

```text
Garden                     ff7cf2b50ce11da0e086198885f33c597badae9c
matcher-free executable    35e33f2b43bd4e607e90490d57955f70a7823f4b
first authority evidence   04a6672356a6ef4d982af636c82b18bae03fdce9
frozen Sounio matcher       7d0c1896bc9552e3733f62b3f4d42dcbd988dd78
```

The matcher-free source hash is
`fb25f6e4f4e78bed37c6e9400c76e1eb355d7f15d1fa513598778365aad08b29`.
It contains no v4 result matcher, child diagnostic vector, or frozen digest.

Its first transcript was committed before the matcher and remains addressable
at SHA-256
`3435ea095019996cd5e3c3bf55810ae033ff07940e7080e728e5acb936f438eb`
with 1730 lines and 23597 bytes. The frozen replay differs only by the appended
`frozen_match=1` and `frozen_mismatch_code=0` fields. Its SHA-256 is
`d27915015cabda1d11211968e0bde5655757599d8dc3313fbfc0506877e49694`
with 1732 lines and 23637 bytes.

The native Loom language-authority Guardian returned `ALLOW` before the first
execution and before the frozen replay. Both used `./bin/souc` with
`SOUNIO_SOUC_ENGINE=lean_single`; no raw ELF was invoked.

## Frozen parent

The direct semantic parent is Operator Genome v3:

```text
parent source       92765416ad8854376a779ef452f89497e2df77f225bf5a4eb5f74f4cd9004a6d
parent semantics    99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926
parent freeze       0b4486ae3c7d0034ffb82208f19330b710ed7d7e92115e93a6f411b354dd03f6
parent transcript   3e79844d3dbd9034e0d8706bf0c3055cba9a7dda0fcfb2daae959e9dbf0c1905
parent parity-open  b2100377695575e024e333a4519687a0ff727989198f7ee0213d0f78c36bc7eb
```

The v4 executable re-evaluates that parent and requires its frozen v3 matcher,
semantic digest, receipt scope, and artifact lineage to agree. A v3
`PARITY_OPEN` receipt does not transfer parity status to v4.

## Frozen grammar

Let `x_r` be bit `r` of the left index and `y_s` bit `s` of the right
index, with all arithmetic in `F2`. The grammar contains:

```text
L2R1(r,s,t) = x_r * x_s * y_t,  r < s, t free
L1R2(r,s,t) = x_r * y_s * y_t,  s < t, r free
```

Each family has `C(4,2)*4 = 24` descriptors. Canonical IDs 0 through 23 are
the nested `(r,s,t)` enumeration of `L2R1`; IDs 24 through 47 are the nested
`(r,s,t)` enumeration of `L1R2`. No descriptor is ranked or selected.

For parent Boolean sign phase `b_parent`, mutation `m`, and indices `i,j`:

```text
b_child(i,j) = b_parent(i,j) XOR m(i,j)
sigma_child(i,j) = sigma_parent(i,j) * (-1)^m(i,j)
```

Every mutation flips exactly 32 of the 256 ordered sign cells. It changes no
partner, destination, ordinal, reduction order, or scalar operation.

## Bounded novelty certificate

For a Boolean two-cochain `m`, the executable uses the 2-cocycle failure

```text
delta_m(i,j,k) = m(i,j)
                 XOR m(i XOR j,k)
                 XOR m(j,k)
                 XOR m(i,j XOR k)
```

Every bilinear phase `i^T B j` has zero failure. For every cubic descriptor,
Sounio constructs the unit-vector witness `(e_r,e_s,e_t)` and obtains failure
one. It then checks the witness against every one of the 65536 packed 4x4
bilinear matrices. The frozen execution produced:

```text
mutation witnesses          48
witness failures             0
bilinear witness checks      3145728 = 48 * 65536
bilinear witness failures    0
pairwise child checks        1128 = C(48,2)
pairwise child collisions    0
```

Therefore every emitted child lies outside the exact, fixed-coordinate
65536-member bilinear phase grammar, and all 48 child sign tables are distinct.
This is `relative_bilinear_grammar_novelty=true`.

It is not a proof of inequivalence under `GL(4,2)`, gauge action, algebra
isomorphism, isotopy, arbitrary program transformation, or any broader search
space. It is not a historical, priority, scientific, algorithmic, or material
novelty claim.

## Frozen population

Sounio generated and checked:

```text
children                       48
sign cells                  12288
parent-delta checks         12288
semantic-cell checks        12288
fixture checks                768
microprogram groups          1536
unresolved target obligations 1920
selected child                  -1
ranking present              false
```

All corresponding failure counts are zero. Each child retains the v3 XOR
partner map and strict ascending-source reduction spine. Each child has 32
eight-lane group masks and 40 unresolved material obligations.

The following four 48-entry vectors are full-child execution goldens, in child
ID order:

```text
negative cells =
96,96,96,92,96,96,96,92,100,100,100,104,88,96,96,88,
100,104,108,104,100,108,104,104,96,96,96,100,96,96,96,100,
96,104,104,104,100,96,96,104,104,104,104,104,108,108,112,112

negative squares =
5,5,5,3,5,5,5,3,3,3,3,3,5,7,7,5,
3,5,5,5,3,5,5,5,5,5,3,5,3,3,5,5,
3,7,5,5,5,5,3,7,5,5,3,3,3,5,5,5

commutator defects =
90,90,90,90,90,90,90,90,98,90,90,98,90,98,98,90,
90,90,90,90,90,90,90,90,90,90,98,90,90,90,90,90,
90,98,90,90,90,90,90,98,90,90,90,90,98,90,90,90

associator defects =
1848,1848,1848,1752,1848,1848,1848,1752,
1752,1704,1704,1656,1848,1848,1848,1752,
1704,1752,1704,1656,1704,1704,1752,1656,
1848,1848,1752,1848,1704,1704,1848,1848,
1704,1848,1752,1704,1848,1848,1704,1848,
1704,1752,1752,1752,1656,1752,1656,1656
```

They are diagnostics counted over the complete child sign tables, not derived
invariants of the cubic mutation alone, and are not fitness values. The ordered
`f64` fixture is smoke-only. It repeats
the same strict fold in two Sounio paths and is not an arbitrary-input floating
point theorem.

## Content identities

Every child digest binds its descriptor, complete 256-cell sign table, 32 group
masks, 16 fixture words, structural diagnostics, and unresolved target count.
Every child-lineage digest additionally binds all five parent artifact hashes.
The population digest binds all 48 child and child-lineage digests in canonical
order.

The frozen Sounio digests are:

```text
lineage    e61ec07a1befb28afc388e3be5292ba36591488ea17592796d37e80d3c1ab24f
grammar    1beaa0baf3f113590ff8c5d4795d7634ba9e5adc5422fda68b90b992d99ea56e
population 0bf700289944bc47f19f6bf7f1b04fe956a7f6f7224fb3a0e6bc284f4bb705e1
receipt    60ec8cec43177ea9441e4dab4dbecc49b974f49722f80a7f5d30cf11d689c6fc
forge      e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff
```

The forge digest is the v4 semantic identity for future parity receipts.

## Material envelopes

Every child carries an unresolved envelope for each canonical target:

```text
701200 Darwin Xeon
701201 Apple Silicon
701202 DGX Spark
711001 dual AMD Alveo U250
```

There are ten obligations per child and target, hence `48*4*10=1920` total.
Observations, discharges, material receipts, lowerings, cost records, and
performance records are all zero. The dual U250 target binds two cards but no
partition, kernel, bitstream, or measured result.

## Negative certificates

Sounio executed 35 deliberate request refusals. They cover forbidden Python
and Rust authority, C++ and LLM authority promotion, broken parent binding,
grammar mutation, child selection or ranking, target/material overstatement,
partner/destination/ordinal/reduction changes, fixture or diagnostic promotion,
prefreeze parity, broader novelty claims, review promotion, and invalid waiver.

The native Guardian separately governs process launch. A Sounio-level refusal
does not substitute for pre-execution denial of a forbidden oracle.

## Claim boundary

The freeze establishes one bounded result: Sounio generated 48 distinct
content-addressed operator children outside the declared fixed-coordinate
bilinear phase grammar and preserved the inherited XOR execution skeleton.

The frozen receipt keeps all of these false:

```text
declared_gl4_gauge_inequivalence
relative_algebraic_novelty
algebra_isomorphism_complete
algorithmic_novelty
material_novelty
scientific_novelty
global_novelty
historical_novelty
priority_claim
external_prior_art_complete
formal_parity_open
effect_parity_open
material_parity_open
target_lowering_admitted
target_cost_admitted
target_performance_admitted
claim_ready
```

External LLM reviews checked formulas, array indexing, matcher structure, and
scope only. They did not execute Sounio, supply or confirm a golden, establish
broader novelty, rank a child, or act as semantic authority.

## Next admissible transition

Only a Guardian-authorized receipt bound to forge SHA-256
`e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff`
may open parity. Opening parity authorizes Lean, Koka, C++, Haskell, and target
work to compare, prove, or measure. It does not select a child, discharge any
material obligation, establish broader novelty, or make v4 claim-ready.
