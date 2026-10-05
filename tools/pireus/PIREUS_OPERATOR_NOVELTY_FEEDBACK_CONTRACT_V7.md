# Pireus Operator Novelty Feedback Contract v7

Status: `SOUNIO_EXECUTABLE`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-NOVELTY-FEEDBACK`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Authority chronology

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The Garden is committed independently. The v7 module, example, and structural
test are matcher-free. They contain no expected membership, class, action,
gauge, residual, outcome, transcript, or digest. The first executable result
may be recorded only after this matcher-free source is committed.

Authority roles are fixed:

```text
Sounio   SEMANTIC_AUTHORITY
Lean 4   FORMAL_PARITY
Koka     EFFECT_PARITY
C++      MATERIAL_PARITY
Haskell  OPTIONAL_DENOTATIONAL_BASELINE
LLM      REVIEW_ONLY
```

Python and Rust are forbidden as generators, oracles, validators, freeze
producers, and parity legs. No disposable language may replace them as a
semantic oracle.

## Frozen inputs

v7 reconciles two frozen Sounio lineages:

```text
Pireus Quotient Novelty Forge v5
-> Pireus Operator-Lowering Forge v6

Pireus XOR Convolution Operation
-> Pireus XOR Lowering Legality
```

The module binds the v5 quotient source, semantics, and freeze hashes; the v6
lowering source, semantics, and freeze hashes; and the XOR-lowering source,
semantics, and receipt hashes. It also executes the live v5 and v6 frozen
matchers. The first CD16 challenge is reconstructed from the live Sounio
`cd_sigma` function. The gate must independently hash the XOR-lowering files
because the semantic executable does not read repository files as an oracle.

The frozen parents remain unchanged. v7 does not append CD16 to the old
48-child population and does not alter any of the old 14 classes.

## Generic challenge

The public evaluator accepts a `PireusOnfChallenge` carrying an explicit
256-cell bit table plus Sounio producer and role. The first challenge builder
defines:

```text
C(i,j) = 0  if cd_sigma(i,j,4) = +1
C(i,j) = 1  if cd_sigma(i,j,4) = -1
```

Any other coefficient is rejected. The builder constructs the table directly
in `(i,j)` order and independently reconstructs it in `(d,i)` order with
`j=i XOR d`. It checks unique coverage, destination recovery, sign equality,
and all 256 cells.

CD16 is the first challenge, not a hard boundary. Any later challenge must
first acquire its own Sounio semantic identity and frozen lineage.

## Exact relation

Let:

```text
V = F2^(Z2^4 x Z2^4)
U = GL(4,2) x C2
B = { delta(q) | q(0)=0 }
delta(q)(i,j) = q(i) XOR q(j) XOR q(i XOR j)
```

The normalized gauge slice fixes `q` to zero on `0,1,2,4,8` and packs the
remaining vectors in this order:

```text
3,5,6,7,9,10,11,12,13,14,15
```

Subtracting the unique linear form that agrees on the standard basis maps
every `q(0)=0` to exactly one normalized gauge with the same coboundary. Thus
the 11-bit slice maps bijectively onto `B`.

The right action is:

```text
(T . (M,0))(i,j) = T(Mi,Mj)
(T . (M,1))(i,j) = T(Mj,Mi)
```

The executable scans all 16-bit matrix encodings, admits the invertible ones,
tests both swap choices, and defines:

```text
G_parent = { g in U | g(parent) XOR parent is in B }
```

It certifies identity, inverse, closure, parent replay, and equality with all
2304 public v5 Q2 relation bits. This full relation replay is what binds the
reimplemented private mechanics to the frozen public quotient.

## Normal form

For any difference table `D`, the normalizer fixes:

```text
q(0)=q(1)=q(2)=q(4)=q(8)=0
edge(v)=highest set basis bit of v
parent(v)=v XOR edge(v)
q(v)=D(parent(v),edge(v)) XOR q(parent(v))
R(i,j)=D(i,j) XOR delta(q)(i,j)
```

The executable checks all 11 pivot gauges, every gauge word, every cellwise
round trip, uniqueness, and equality with the public v5 certificate counts.

## Bridge-or-seed result

For each frozen class representative `P_k` and each `g` in `G_parent`, Sounio
computes:

```text
D(k,g) = C XOR (P_k . g)
(R(k,g),q(k,g)) = normalize(D(k,g))
```

Every pair witness replays all 256 cells:

```text
C = (P_k . g) XOR delta(q(k,g)) XOR R(k,g)
```

The canonical witness is the least finite tuple:

```text
(
  residual_nonzero_cells,
  residual_words_lexicographic,
  class_id,
  representative_child,
  matrix_code,
  swap,
  gauge_word
)
```

Exactly one outcome is legal:

```text
ExistingClassBridge  if one or more residuals are zero
OperatorSeed         if every residual is nonzero
```

For a bridge, all zero hits are counted and the least zero tuple is emitted.
For a seed, every nonmembership check must be nonzero, the complete finite
separation count must close, and the chosen residual table must reconstruct the
challenge. The seed is relative to the frozen v5/v6 quotient only.

## First transcript surface

The matcher-free example prints:

- all lineage identities and live checks;
- all 256 challenge cells and packed words;
- the complete normalizer certificate;
- every admitted parent action and gauge;
- the 48x48 v5 relation replay census;
- every challenge/class/action pair witness;
- the canonical bridge or seed and all 256 residual cells;
- the receipt, negative witnesses, and seven digests;
- `claim_ready=false`.

No target observation, material receipt, cost, performance record, parity
result, or LLM answer enters the decision.

## Post-first freeze matcher

Only after the matcher-free source and its first Sounio transcript were
committed, the Sounio module gained an exact mismatch classifier. It binds:

- all nine parent/challenge lineage identities;
- the 256-cell challenge table and independent address census;
- the complete normalizer and admitted parent-action certificates;
- the 14 representatives and all 168 class/action separations;
- the canonical outcome tuple, 96-bit residual, and 256-cell replay;
- all request refusals, receipt flags, and seven result digests.

The matcher returns a nonzero category code for any drift. Its success freezes
the already-produced finite result; it does not mutate the result, open parity,
admit target evidence, or promote relative separation to a broad novelty claim.

## Negative contract

The executable refuses or detects:

- Python, Rust, C++, or an LLM as semantic authority;
- missing policy or lineage;
- unfrozen or unbound parents;
- an altered finite universe;
- any expected result or write from a parity language;
- review promotion;
- material, cost, or performance promotion;
- broad or historical novelty promotion;
- parity before freeze or claim promotion;
- an invalid founder waiver;
- a short challenge or non-bit cell;
- bridge/seed overlap;
- a zero seed, nonzero bridge, or corrupted replay.

The native Guardian separately denies forbidden processes before launch and
records every decision. Sounio request negatives do not substitute for that
pre-execution control.

## Claim boundary

Even a passing first execution proves neither semantics freeze nor parity. It
may state only that Sounio produced one exact bridge-or-seed result relative to
the declared frozen finite quotient.

It does not establish wider linear or nonlinear equivalence, isotopy, algebra
isomorphism, algorithmic novelty, material support, performance, historical
novelty, scientific novelty, priority, or `CLAIM_READY`.
