# Pireus Operator Genome v3 Contract

Status: `SEMANTICS_FROZEN`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-GENOME`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Contract

Pireus Operator Genome v3 turns the frozen bilinear Operator Genesis winner
into one executable, target-neutral semantic normal form. The genome owns the
operator. Xeon, Apple Silicon, DGX, and dual AMD Alveo U250 may later compete
to materialize it, but no target may alter its partner map, sign, destination,
ordinal, scalar contract, or reduction barriers.

The authority order was observed:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This contract stops at `SEMANTICS_FROZEN`. No parity implementation ran before
this freeze, and no claim-ready promotion is authorized.

## Chronology

The Git history is part of the receipt:

```text
Garden                     038261eff749e62e78502f9f7525393cc4651c56
matcher-free executable    37c7a1192c81e1d9bc1be22615787899b88cf9d0
first authority evidence   52a60403ca56c3827a6f35b179a3a08254fe84f1
frozen Sounio matcher       a794bfa2325dc6c5c545c95a7247dc1e3b1db4d4
```

The first executable commit contains no expected v3 mask, negative count,
fixture output, semantic digest, or frozen matcher. The first transcript is
preserved in the parent of the matcher commit with SHA-256
`8158ef30dbc31f2bd5aaae477643b3f93ae16cbfa4e8391d8ddd3b06f1a2cfe6`.

The native Loom language-authority Guardian returned `ALLOW` before both the
first execution and the freeze replay. Every execution used `./bin/souc` with
`SOUNIO_SOUC_ENGINE=lean_single`; no raw ELF was invoked.

## Frozen lineage

The direct semantic parent is Operator Genesis bilinear v2:

```text
parent source       31f5fe668c100f0aa27b4c4405c022c127e5445a743d5029e2d913da8dfd8a44
parent semantics    bb5560806ea7a84a0cc5f88ec5d4adbea4004ec6b2560af6e4d8de31b3a88d3b
parent transcript   d8fa8bac03d9b09f970f6bd328f9b295165c1e56823c799a46771886123cacd0
parent selected     a264defd7a6af854ccfa1cc1a7239c505bfc2bc0ea8dee93b17dd09952d96443
parent contract     0cb51e12e17be8500be1de679c9ce95d67b8dbffb0750be511833cb76d8548e8
parent freeze       38f4d5c0a46029283bc21fd901a60e1f7f08332b48317fd40548abf91fe2e6aa
```

The inherited selected identity is:

```text
bits=4
dimension=16
class_id=26
quadratic_code=198
packed_matrix=1128
matrix_rows=(8,6,4,0)
```

The quadratic-code encoding is also inherited exactly. Bits 0 through 3 are
`B_00,B_11,B_22,B_33`; bits 4 through 9 are
`B_01 XOR B_10`, `B_02 XOR B_20`, `B_03 XOR B_30`,
`B_12 XOR B_21`, `B_13 XOR B_31`, and `B_23 XOR B_32`. For rows
`(8,6,4,0)`, bits 1, 2, 6, and 7 are set, hence
`2 + 4 + 64 + 128 = 198`. `class_id=26` is the hash-bound class label emitted
by the frozen parent census; v3 does not replace that census with a symbolic
class-number formula.

The live `cd_sigma` source is bound at
`e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed`.
All 256 calls returned either `-1` or `+1`.

The recursive Cayley-Dickson convention lives in that exact Sounio source and
in the frozen v2 parent. v3 checks its extensional range and uses its returned
signs; this contract does not claim a new independent cocycle proof.

## Frozen genotype

For `i,j in F2^4`, with coordinate `r` stored in integer bit `r` and matrix
entry `(r,s)` stored in packed bit `4*r+s`, the exact law is:

```text
b_B(i,j)     = i^T B j mod 2
sigma_B(i,j) = cd_sigma(i,j,4) * (-1)^b_B(i,j)
r[d]         = sum_(i=0..15) sigma_B(i,i XOR d)
               * a[i] * b[i XOR d]
```

The Sounio execution derived 104 negative cells among the 256 ordered pairs.
The independent recount also produced 104 with zero discrepancy.

The value 104 is a hash-bound exhaustive Sounio execution golden. It is not a
symbolic closed form, an algebra classification, or a novelty metric.

## Frozen microprogram

The displacement-major normal form contains exactly 256 cells:

```text
cell(d,i) = (destination=d,
             lhs=i,
             rhs=i XOR d,
             sign=sign_bit(i,i XOR d),
             ordinal=i)
```

Every fixed-`d` partner map is a permutation. Sounio executed 256 permutation
checks, 256 coverage checks, and found zero failures.

The 32 eight-lane negative masks, in `(d,chunk)` order, are:

```text
2,169,148,192,242,166,148,192,
206,154,148,192,148,192,62,106,
168,252,194,105,162,9,200,99,
138,33,224,75,128,43,234,65
```

Groups 0 through 15 are same-chunk and groups 16 through 31 are cross-chunk.
Equivalently, both chunks are local for `d<8` and both cross for `d>=8`.

These masks are Sounio execution goldens. Their freeze does not claim that
eight lanes are the physical width of every target.

## Ordered fixture

Both Sounio paths use the same ascending-`i` addition spine. The direct path
derives the formula per term; the microprogram path reads the generated cells.
Extensional checks already bind all 256 partner/sign/destination/ordinal fields.

On the exactly representable integer-valued fixture, both paths produced these
16 `f64_to_bits` words:

```text
4644284339167690752,4644565814144401408,
-4587127326709383168,4647099088934797312,
4631248529308778496,4646747245213908992,
4642824187726004224,-4579369172663795712,
-4575595648757268480,-4579545094524239872,
4646219479632576512,4642542712749293568,
4647063904562708480,-4588112489127870464,
4643651020470091776,4639376119261298688
```

This is a bit-exact fixture replay, not a universal theorem about floating-point
equivalence under arbitrary inputs or execution modes.

## Frozen digests

All digests are produced inside Sounio using the serialization fixed by the
Garden:

```text
lineage      58134500690005767cce11108e90d764529b15da72a30809c5f493bd4e3f4f8b
genotype     aeab0cad2b569e7686734ef8a20c18aaa10bf4ef64c453d4bb22fd430f8ecfd0
microprogram dee9deb1923745b81098e245c2eb4c050a1fd05c3077e5b3256e381758918027
groups       441a9314e9bb3bbe5cf4cbc9fc3821e927cd7b6d3bf6e4e35ff53f7b61ff48d9
evaluation   b2820bf3eaeb6f7ccbae210e755902ac9ec49fa00600097fe13a30743f8039f2
targets      2328713a70bc401781b2f76bedcb8e5763cce5acaabcc8afaf0e8396607bd5b2
receipt      f8427b5f1e1742bdd6d6bc0e0ef2547972247f62372210ee8825526f9b8010eb
genome       99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926
```

The genome digest is the v3 semantic identity for later parity receipts.

## Materialization envelopes

The frozen genome emits four canonical target envelopes:

```text
701200 Darwin Xeon
701201 Apple Silicon
701202 DGX Spark
711001 dual AMD Alveo U250
```

Each envelope has ten obligations:

1. partner coverage;
2. sign coverage;
3. no extra cell;
4. destination preservation;
5. ordered reduction;
6. scalar contract;
7. memory contract;
8. lineage binding;
9. replay binding;
10. claim boundary.

All forty obligations are unresolved. Observations, discharges, material
receipts, lowerings, and cost records are all zero. The U250 envelope binds two
required cards and two declared engine slots but does not select an output or
data partition between them.

Target completeness means that every envelope field and unresolved obligation
is present. It is not a claim that any target can execute the genome.

## Negative certificates

Sounio executed 25 deliberate refusals and admitted all 25 as refusals:

- Python oracle;
- Rust oracle;
- C++ semantic authority;
- LLM semantic authority;
- missing parent;
- unfrozen parent;
- matrix override;
- parity before freeze;
- missing target;
- observation without receipt;
- changed partner;
- changed sign;
- dropped cell;
- duplicated cell;
- reassociated reduction;
- wrong U250 card count;
- algorithmic promotion;
- material promotion;
- scientific promotion;
- global promotion;
- historical promotion;
- priority promotion;
- claim-ready promotion;
- review promotion;
- invalid waiver.

The outer Guardian separately authorizes stage transitions and will fail closed
on missing policy, timeout, forbidden-language launch requests, authority-role
promotion, and wrong-stage parity requests.

## Claim boundary

The freeze establishes executable genome derivation relative to the frozen v2
parent. `parent_receipt_relative_semantic_novelty=true` records a checked parent
receipt fact; v3 does not independently prove or widen novelty.

`complete` on genotype and microprogram means complete enumeration and wiring
under this normal form. It establishes no composition-algebra law, norm
multiplicativity, alternativity, associativity, algebra isomorphism, transform
identity, or subquadratic algorithm.

The frozen receipt keeps all of these false:

```text
formal_parity_open
effect_parity_open
material_parity_open
target_lowering_admitted
target_cost_admitted
target_performance_admitted
algorithmic_novelty
material_novelty
scientific_novelty
global_novelty
historical_novelty
priority_claim
claim_ready
```

External LLM reviews checked formulas and matcher consistency only. They did
not execute Sounio, supply or confirm a golden, establish novelty, or act as
semantic authority.

## Next admissible transition

Only a Guardian-authorized receipt bound to genome SHA-256
`99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926`
may open parity. Opening parity authorizes Lean, Koka, C++, Haskell, and target
schedule searches to compare, prove, or measure. It does not discharge any of
the forty materialization obligations and does not make the genome claim-ready.
