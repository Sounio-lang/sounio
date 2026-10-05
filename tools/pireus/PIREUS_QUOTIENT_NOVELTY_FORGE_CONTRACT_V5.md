# Pireus Quotient Novelty Forge v5 Contract

Requested status: `SEMANTICS_FROZEN`

Effective only with a native Guardian `ALLOW` receipt for the frozen replay.

Concept-ID: `SOUNIO-PIREUS-QUOTIENT-NOVELTY-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Contract

Pireus v5 turns operator novelty into a typed, replayable quotient certificate.
It starts from the frozen 48-member v4 population, discovers the exact subgroup
of the declared `GL(4,2) x C2` action that preserves the frozen parent modulo a
basis-sign gauge, and partitions the population under three nested equivalence
relations.

The semantic authority order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This contract stops at `SEMANTICS_FROZEN`. It opens no Lean, Koka, C++,
Haskell, target-lowering, cost, performance, historical, or priority result.

## Chronology

Git history is part of the receipt:

```text
Garden                         4de9a4679f
matcher-free executable        5b0ace3892
preserved failed attempt       d69835f65f
compact action repair          c118ebf998
first successful observation   5768d08797
frozen Sounio matcher           cbd64ff6ed
```

The matcher-free source hash was
`799565db9a23ad99e300226ba500f06e7cd801a1acecd9ce63f21e5e8da936c4`.
It contained no stabilizer count, partition, representative, witness, digest,
or frozen matcher.

The first Guardian-authorized run of that source exited 139 with no stdout.
Its empty evidence and failure receipt remain committed. It was not promoted to
a semantic result. A representation-only repair replaced a dense 131072-entry
membership mask plus three 40320-entry arrays with one canonical packed-action
array. The complete matrix/action universe was unchanged. Static compiler BSS
fell from 6130408 to 1048984 bytes.

The repaired matcher-free source hash was
`0a2490e0d21f8b9c4004cd6d6fe1caf5c03a2d239bda8d029a427327c42bd0ac`.
Its second Guardian-authorized run succeeded through `./bin/souc` with explicit
`SOUNIO_SOUC_ENGINE=lean_single`; no raw ELF was invoked. The first successful
transcript is SHA-256
`7dfc4fa51b689d9295cf37252ffac0b54a108086a6c88dd9ba8c374769ac644f`,
2916 lines and 39979 bytes. It was committed before the frozen matcher.

## Frozen parent

The direct semantic parent is Cubic Operator Forge v4:

```text
parent source       2c295c48bcd2de0f43a42787dcc612f78c7d40d528641e4fec890858d881c974
parent semantics    e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff
parent freeze       1da425c1ff53273825a71b46850e0cd9e7d4cd5b77aa79eb65ef269aadd5a87b
```

The v5 executable also binds the v4 transcript and parity-open receipt hashes,
re-evaluates the parent through its frozen Sounio matcher, and reconstructs one
parent sign table from all 48 children. It checks `48*256=12288` reconstructed
cells and the same number of sign-range cells with zero failures. Parent parity
does not transfer parity status to v5.

## Gauge quotient

Let `V=F2^4`. A basis-sign gauge is a function `q:V -> F2` with `q(0)=0`.
It acts on a Boolean sign table `f:V x V -> F2` by the coboundary

```text
dq(i,j) = q(i) XOR q(j) XOR q(i XOR j).
```

Linear functions form the four-dimensional kernel of `d`, so the effective
gauge image has dimension `15-4=11` and order 2048. Sounio fixes the eleven
tree cells `(e_r,v)` with the highest set bit of `v` equal to `r`, proves the
11-by-11 pivot system has full rank, and executes all 2048 canonical gauge
round trips over all 256 cells. Every failure counter is zero.

## Parent action group

A row-packed matrix encoding is a nonnegative integer `0 <= M < 65536`.
Sounio classifies all encodings and observes exactly 20160 invertible matrices.
With optional operand exchange `s in C2`, the declared action universe has
40320 elements and direct-product law

```text
(M1,s1) * (M2,s2) = (M1 compose M2, s1 XOR s2).
```

An action is admitted exactly when its displacement on the frozen parent is a
gauge coboundary:

```text
GaugeAut(f0) = {(M,s) | normalize((M,s).f0 XOR f0).remainder = 0}.
```

Packed actions use the lossless code `2*M+s`. Sounio certifies encode/decode,
identity, two-sided inverse, every ordered-pair closure, matrix composition on
all 16 vectors, parent replay on all 256 cells, and canonical gauge pullback on
all eleven basis gauges and 256 cells.

The first successful Sounio observation produced:

```text
matrix encodings                  65536
invertible matrices               20160
actions considered                40320
admitted actions                     12
admitted without swap                 6
admitted with swap                    6
inverse checks                       12
closure checks                      144
composition vector checks          2304
gauge-equivariance checks         33792
all corresponding failures            0
```

This is the projected action group on sign tables modulo gauge. The canonical
gauge attached to an action is an equality witness. v5 does not claim a lifted
group law on triples `(M,s,q)` or faithfulness beyond the declared finite
action and gauge domains.

## Frozen quotient atlas

For the 48 ordered v4 children, Sounio freezes three nested profiles:

```text
Q0: exact 256-cell sign-table equality
Q1: equality modulo the 2048-element basis-sign gauge image
Q2: equality modulo the complete parent-relative GaugeAut(f0) action
```

Every profile is checked for reflexivity, symmetry, transitivity, exact
partition coverage, canonical minimum-ID serialization, and full 256-cell
member-to-representative witness replay. Q0 refines Q1 and Q1 refines Q2.

The frozen partition summary is:

```text
profile   classes   minimum size   maximum size   size sum
Q0             48              1              1         48
Q1             48              1              1         48
Q2             14              2              4         48
```

The atlas digest binds all 256-cell raw and normalized child tables, 6912
relation cells, class counts, representatives, member masks, member counts,
144 child class IDs, 144 representatives, all 144 `(matrix,swap,gauge)`
witnesses, class digests, profile digests, and every certificate counter.

No child is selected. A minimum child ID is only the canonical serialization
of one equivalence class, never a rank, score, winner, or material choice.

## Typed novelty

The positive v5 result is exactly:

```text
Novelty<
  Population = frozen v4 48-child population,
  Equivalence = parent-relative GL(4,2) x C2 modulo basis-sign gauge,
  Parent = frozen v4 parent sign table,
  Witness = replayable (matrix,swap,gauge),
  Stage = SOUNIO_EXECUTABLE
>
```

Within this declared finite universe, the 48 candidates form 14 Q2 classes.
That is a parent-relative operator-novelty atlas. It is not a proof of global
linear/swap/gauge classification over all sign tables, nonlinear permutation
classification, isotopy, algebra isomorphism, algorithmic novelty, material
novelty, scientific novelty, historical novelty, or priority.

## Content identities

The frozen Sounio digests are:

```text
lineage    8c409f3bcd23be504b22cf04b2fb5a9e5f708e774ae902deff5ce9e7f30fe974
normalizer 9b347a79d09d8327681f48191cd99b065eb32cbc74797d59637736348ff73f92
actions    3aa6883df8f2446a4ef11352a9f6e17f80ad67a0ee2589d8e04072f6551f7e9a
atlas      302919a1809d6106d7faf421de94524eff7d79b9adb843abf0d787cc8e550e83
receipt    b4fecd5dbbc2888ad6694a165d118f0821fa86317ed804ce6c8679ff63ed56d1
forge      9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21
```

The forge digest is the v5 semantic identity for later parity receipts. The
frozen matcher recomputes lineage, normalizer, full atlas, receipt, and forge
digests and binds the observed action digest to the exhaustive action
certificate. SHA-256 words are serialized as eight unsigned 32-bit limbs held
in Sounio `i64` values.

## Material envelopes

The inherited population retains 1920 unresolved obligations across the four
canonical target families:

```text
701200 Darwin Xeon
701201 Apple Silicon
701202 DGX Spark
711001 dual AMD Alveo U250
```

Target observations, discharges, material receipts, lowerings, cost records,
and performance records remain zero. No target is preferred and no hardware
result changes semantic equivalence.

## Negative certificates

Sounio executes 31 deliberate refusals. They include Python and Rust oracles,
C++ or LLM semantic authority, broken parent binding, result injection,
selection/ranking, material claims without receipts, global quotient or
isomorphism promotion, pre-freeze parity, claim-ready promotion, review
promotion, and invalid waiver.

The native Guardian separately controls launch. A Sounio-level refusal is not
substituted for pre-execution denial of a forbidden process.

## Claim boundary

The freeze establishes a complete quotient atlas only for the frozen 48-child
population under the declared parent-relative finite equivalence. All of these
remain false:

```text
global_linear_swap_gauge_quotient_complete
nonlinear_permutation_quotient_complete
isotopy_quotient_complete
algebra_isomorphism_complete
relative_algebraic_novelty
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

External LLM reviews checked the finite algebra, packing, quotient arithmetic,
matcher structure, and scope only. They did not execute Sounio, provide or
confirm an expected value, select an operator, establish broad novelty, or
serve as semantic authority.
