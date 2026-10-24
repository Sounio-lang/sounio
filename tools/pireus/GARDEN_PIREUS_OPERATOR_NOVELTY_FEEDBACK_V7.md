# Garden: Pireus Operator Novelty Feedback v7

Status: `GARDEN`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-NOVELTY-FEEDBACK`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Pireus v3-v6 already forms a closed semantic loop from operator grammar to
finite children, parent-relative quotient classes, typed lowering programs,
and machine residuals. The next step is to let an operator arriving from
outside that closed population push back on the population itself.

v7 proposes an operator novelty feedback kernel:

```text
external operator challenge
        |
        v
frozen quotient orbit scan
        |
        +---- zero remainder ----> ExistingClassBridge
        |
        +---- nonzero remainder -> gauge-normalized OperatorSeed
                                      |
                                      v
                              next operator generation
```

This is not a synonym for classification. When the challenge lies outside the
declared quotient, Pireus must synthesize the exact semantic remainder that
makes it different, preserve that remainder as an executable operator, and
emit a reconstruction witness. Novelty becomes a generated object rather than
a label attached after a benchmark. In v7, `irreducible` is never used in a
ring-theoretic sense; the precise object is a nonzero residual after the
declared basis-sign gauge normalization.

The first bounded challenge is the independently frozen CD16 twisted XOR
operator already carried by the Sounio XOR-lowering lineage. It must be tested
against the independently frozen v6 operator atlas. The result is not declared
in this Garden.

## Authority order

The only admissible order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. The first executable representation, orbit
scan, bridge decision, residual, seed, reconstruction witness, result, and
digest must be born in Sounio. Lean 4 is `FORMAL_PARITY`, Koka is
`EFFECT_PARITY`, C++ is `MATERIAL_PARITY`, and Haskell is an optional
denotational baseline only after the Sounio source and semantics are frozen by
hash.

External LLMs are review-only. They may identify a defect but cannot produce,
confirm, select, or promote a semantic result. Python and Rust are forbidden
as generators, oracles, validators, freeze producers, or parity legs. Node,
Ruby, shell, awk, `bc`, or another disposable language may not replace them as
a semantic oracle.

The first v7 executable must contain no expected membership decision, bridge
class, representative, action, gauge, residual weight, seed identity, result
digest, output transcript, or frozen matcher. Sounio must produce every first
result before those values can be frozen.

## Frozen lineages to reconcile

v7 joins two existing Sounio lineages without rewriting either one.

The quotient lineage is:

```text
v3 Operator Genome
  -> v4 Cubic Operator Forge
  -> v5 Quotient Novelty Forge
  -> v6 Operator-Lowering Forge
```

Its immediate frozen v6 identities currently include:

```text
module_sha256=
178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0

semantics_sha256=
bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1

freeze_sha256=
973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181
```

The external challenge lineage is the independently frozen Sounio CD16 XOR
lowering legality surface:

```text
xor_lowering_source_sha256=
7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb

xor_lowering_semantics_sha256=
9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970

xor_lowering_receipt_sha256=
daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
```

These hashes identify ancestry; they do not predeclare the v7 result. The first
executable must also replay the live Sounio parent matchers. A hash-bound but
semantically failing parent is not admissible.

v7 may consume the v5 raw tables and v6 class identities only through their
public frozen results. It may not edit either frozen parent, add the challenge
retrospectively to the old 48-child population, or reinterpret the old 14
classes after seeing the challenge.

## The challenge operator

For the first experiment, the external challenge is the complete sign-bit
table produced by the live Sounio `cd_sigma` function:

```text
C(i, j) = sign_bit(cd_sigma(i, j, 4))

sign_bit(+1) = 0
sign_bit(-1) = 1
```

for all `(i, j)` in `Z2^4 x Z2^4`. Any other `cd_sigma` value is an executable
failure rather than an implicit sign convention. v7 does not infer a cocycle,
Cayley-Dickson, or other algebraic law from the function name. It consumes the
live Sounio table as a frozen finite object.

The existing destination schedule enumerates the same address space by
`d = i XOR j`, then `j = i XOR d`. The executable must check this cellwise
bijection rather than assume it from prose. All 256 cells must be recovered and
checked. A sampled table, output-vector match, benchmark checksum, or
instruction trace cannot substitute for the complete operator table.

The challenge is independent of material lowering. Existing AVX-512, Apple,
DGX, or future U250 receipts may be evaluated for applicability only after v7
has produced and frozen a semantic relation. A material implementation cannot
define the challenge or decide its class.

The eventual v7 API must accept an explicit 256-cell challenge with frozen
lineage, not hard-code CD16 as the only possible operator. CD16 is the first
executable challenge, not the boundary of the mechanism.

## The declared relation

The comparison relation is a complete executable restatement of the
parent-relative v5 `Q2` profile inherited by v6. Let:

```text
V = F2^(Z2^4 x Z2^4)
U = GL(4, 2) x C2
B = { delta(q) | q : Z2^4 -> F2 and q(0) = 0 }
Q_norm = { q | q(0) = q(e_0) = ... = q(e_3) = 0 }
N = { delta(q) | q in Q_norm } = B
G_parent = Stab_U([parent] in V / B)
```

Here `q` is an arbitrary bit-valued function on the remaining 11 vectors of
`Z2^4`, and:

```text
delta(q)(i, j) = q(i) XOR q(j) XOR q(i XOR j)
```

The equality `N = B` is part of the executable contract, not an appeal to
notation. For every `q` with `q(0)=0`, there is a unique linear form `l_q` whose
values agree with `q` on the four standard basis vectors. Then:

```text
q_norm = q XOR l_q
q_norm in Q_norm
delta(q_norm) = delta(q), because delta(l_q) = 0
```

Conversely, if `q` is in `Q_norm` and `delta(q)=0`, then `q` is linear and
vanishes on a basis, so `q=0`. Thus `delta` is injective on `Q_norm`, `N` has
exactly `2^11` tables, and `Q_norm` provides a unique name for every element of
`B`.

For `g = (M, s)`, v7 fixes the same right-action convention used by the frozen
v5 table transformer:

```text
(C . (M, 0))(i, j) = C(M i, M j)
(C . (M, 1))(i, j) = C(M j, M i)
(M, s) * (N, t) = (M N, s XOR t)
(M, s)^-1 = (M^-1, s)
```

Matrix pullback sends a coboundary to a coboundary, and swap leaves
`delta(q)(i,j)` unchanged because the displayed expression is symmetric in
`i` and `j`. Therefore `B=N` is `U`-invariant, the right action descends to
`V/B`, and `G_parent` is a subgroup: the stabilizer of the parent class. This
definition, including the right-action composition law and subgroup closure,
must be cellwise executable and covered by inverse, composition, closure, and
replay checks. `G_parent` is a computed subset of `U`, not another name for the
full product.

The 11-bit gauge word is also fixed before execution. Let:

```text
(v_0, ..., v_10) = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
gauge_word(q) = sum_r q(v_r) * 2^r
```

These are exactly the nonzero, non-basis vectors in ascending four-bit integer
order. This is the explicit bijection from `Q_norm` to integers `[0,2047]`;
it provides canonical gauge names without changing the table relation.

For a difference table `D`, v7 fixes the v5 normalizer complement by this
recurrence in ascending integer order:

```text
q(0) = q(1) = q(2) = q(4) = q(8) = 0
edge(v) = the highest set basis bit of v
parent(v) = v XOR edge(v)
q(v) = D(parent(v), edge(v)) XOR q(parent(v))
R(i, j) = D(i, j) XOR delta(q)(i, j)
```

The recurrence is evaluated for the 11 non-basis vectors. It pins the 11 pivot
functionals and therefore the concrete complement used to name `R`; pivot,
roundtrip, uniqueness, and full 256-cell replay remain executable obligations.

The finite universe is fixed before execution:

```text
16-bit matrix encodings scanned = 65536
invertible matrices in GL(4, 2) = 20160
swap choices in C2 = 2
actions considered in U = 40320
basis-sign gauge words = |B| = |N| = 2^11 = 2048
```

These are universe cardinalities, not expected membership or result values.
The admitted `G_parent` cardinality is produced only by the first Sounio
execution.

`C` is related to a frozen representative `P_k` only if there exists an action
`g` in `G_parent` and a basis-sign gauge `q` in `N` such that every cell
satisfies:

```text
C(i, j) = g(P_k)(i, j) XOR delta(q)(i, j)
```

The stabilizer is therefore determined as data by the complete fixed universe,
the explicit action on `V/B`, the explicit gauge normalizer, and the frozen
parent table. v7 cannot guess the action set from the old class witnesses,
assume that every witness action appears, or widen the relation to all linear
maps, nonlinear permutations, isotopies, algebra isomorphisms, or numerical
coincidence.

The basis-sign normalizer may be reimplemented in v7 because the v5 helpers are
private, but its pivots, roundtrip, and uniqueness checks must be reconstructed
from the explicit definition above and the public frozen tables. This is a new
Sounio executable profile bound to the old definition, not a mutation of the
frozen parent source. Any divergence from the frozen v5 certificate counts is
a lineage failure, not permission to redefine `Q2`.

## Residual synthesis

For every frozen class representative `P_k` and every admitted action `a`, v7
must compute:

```text
D(k, a) = C XOR action(a, P_k)
(R(k, a), q(k, a)) = normalize(D(k, a))
```

The normalizer must provide an exact cellwise replay:

```text
C = action(a, P_k) XOR delta(q(k, a)) XOR R(k, a)
```

Two legal semantic outcomes exist.

### ExistingClassBridge

If an orbit candidate has `R(k, a) = 0`, v7 emits:

```text
ExistingClassBridge<
  ChallengeIdentity,
  FrozenClassIdentity,
  RepresentativeIdentity,
  ParentAction,
  Gauge,
  CellwiseReplay
>
```

The bridge proves only membership in the declared parent-relative finite
quotient. It does not prove algebra isomorphism under a wider relation,
algorithmic novelty, material novelty, historical novelty, or priority.

### OperatorSeed

If the exhaustive orbit scan has no zero remainder, v7 emits:

```text
OperatorSeed<
  ChallengeIdentity,
  FrozenQuotientIdentity,
  BaseClassIdentity,
  ParentAction,
  Gauge,
  GaugeNormalizedNonzeroResidual,
  CellwiseReplay,
  ExhaustiveSeparationCertificate
>
```

The seed is the gauge-normalized nonzero residual `R`, represented as a
complete operator table and not as prose. Applying it to its frozen base
witness must reconstruct `C` exactly. The seed may then enter a later Sounio
generation grammar as an operator-producing delta.

The canonical seed witness is selected only to make identity deterministic.
Candidates are ordered by this predeclared tuple:

```text
(
  residual_nonzero_cells,
  residual_words_lexicographic,
  frozen_class_id,
  representative_child_id,
  action_matrix_code,
  action_swap_bit,
  gauge_word
)
```

This is not target ranking, performance selection, or a claim that the chosen
residual is minimal under an undeclared metric. It is canonical naming inside
the declared finite scan. If one or more zero residuals exist, the bridge uses
the same order with `residual_nonzero_cells=0` and zero residual words; all zero
hits are counted, and the least tuple supplies the canonical bridge witness.

## Why this is operator generation

The generated object is not the original challenge. It is the semantic delta
that the current Pireus ontology cannot quotient away:

```text
known operator orbit + generated gauge-normalized residual = challenged operator
```

That residual can be:

- replayed as an operator by itself;
- composed with a frozen representative;
- used as a mutation seed for the next finite grammar;
- lowered independently to expose missing machine primitives;
- compared with future challenges by its own frozen identity.

This creates a disciplined novelty feedback cycle:

```text
operator generation
-> quotient
-> lowering
-> external or machine challenge
-> gauge-normalized nonzero residual
-> next operator generation
```

Pireus therefore becomes capable of expanding its operator vocabulary in
response to exact semantic pressure while keeping the old vocabulary frozen.

## Required first executable certificates

The matcher-free Sounio executable must emit at least:

- both frozen lineage bindings and live-parent checks;
- all 256 challenge cells and packed table words;
- challenge reconstruction checks from the CD16 definition;
- complete matrix-encoding and `GL(4,2) x C2` scan counts;
- the exact admitted parent actions and their gauges;
- parent stabilizer replay checks;
- basis-sign normalizer pivot, roundtrip, and uniqueness checks;
- every frozen class representative consumed from the public atlas;
- one normalized residual witness per representative/action pair;
- exhaustive relation checks across the declared orbit;
- exactly one typed outcome: `ExistingClassBridge` or `OperatorSeed`;
- all zero-residual hits plus the canonical bridge witness when a bridge exists;
- complete cellwise replay of the chosen outcome;
- an exhaustive separation certificate when the seed outcome is produced;
- distinct lineage, challenge, orbit, residual, outcome, and forge digests;
- negative witnesses and explicit failure codes;
- `claim_ready=false`.

The first transcript must preserve the complete result. A later compact view
may be derived only from the frozen full transcript.

## Negative witnesses

The first executable must fail closed for at least:

- Python as semantic authority or oracle;
- Rust as semantic authority or oracle;
- C++ as semantic authority;
- an external LLM promoted above review-only;
- missing, unfrozen, mismatched, or live-failing parent lineage;
- a challenge with fewer or more than 256 cells;
- a challenge cell outside `{0, 1}`;
- a predeclared result, class, action, gauge, residual, digest, or transcript;
- an incomplete matrix universe;
- a guessed or injected parent stabilizer;
- an altered parent representative table;
- a bridge without zero residual and cellwise replay;
- a seed without exhaustive separation;
- a zero residual promoted as a new seed;
- a nonzero residual promoted as an existing bridge;
- a reconstructed challenge that differs in one cell;
- material evidence used to decide semantic membership;
- target cost or performance used to choose the semantic seed;
- parity before freeze;
- claim readiness before all required parity and evidence stages;
- a waiver not issued by the founder, scoped, purpose-bound, and live.

Guardian enforcement must happen before execution and fail closed for missing
policy, timeout, or policy error. Decisions must record `ALLOW` or `DENY` and a
reason. Receipts must retain source, frozen-semantics, producer language, role,
toolchain, hardware, command, and result identities. Parity receipts and LLM
reviews cannot be promoted to semantic authority.

## Claims explicitly forbidden

Even a successful frozen v7 result does not establish:

- novelty outside the declared parent-relative v5/v6 quotient;
- novelty under all `GL(4,2) x C2` actions independent of the parent;
- nonlinear, isotopy, or algebra-isomorphism novelty;
- mathematical uniqueness beyond the declared canonicalization;
- sub-quadratic acceleration;
- a new machine instruction or emitted lowering;
- performance, cost, or energy improvement;
- material support on Xeon, Apple Silicon, DGX, or U250;
- historical novelty, scientific novelty, publication priority, or patent
  priority;
- `CLAIM_READY`.

The first admissible positive statement is deliberately narrow:

```text
Sounio computed a bridge or a gauge-normalized nonzero operator seed for one explicit
challenge relative to the frozen Pireus v5/v6 quotient, with complete replay.
```

That sentence becomes available only if the corresponding executable and
freeze gates pass.

## Falsification

The v7 hypothesis fails for this profile if any of the following occurs:

- the two frozen Sounio lineages cannot be replayed together without changing
  either parent;
- the declared stabilizer cannot be recovered from the fixed v7 universe,
  action, normalizer, and public frozen parent table;
- normalizer replay or uniqueness fails;
- the orbit scan is incomplete;
- both outcome kinds are admitted, or neither is admitted;
- the emitted bridge or seed cannot reconstruct all 256 challenge cells;
- a claimed seed lacks a complete non-membership certificate;
- an injected expected result can alter the first semantic outcome;
- semantic membership depends on a material target, external language, or LLM.

A failed experiment remains evidence about the missing semantic interface. It
must not be repaired by weakening the quotient or substituting a material
result.

## Semantic lane declaration

```text
Semantic-Lane-ID: pireus-operator-genome-v3-20260829
Owner: codex
Concept-IDs: SOUNIO-PIREUS-OPERATOR-NOVELTY-FEEDBACK (proposed)
Intent-Preserved: Pireus generates executable operator novelty without letting a target or external language define semantics retrospectively.
Transformation: Reconcile an explicit external operator with a frozen parent-relative quotient and synthesize an exact gauge-normalized nonzero residual when no bridge exists.
Types-Changed: New challenge, bridge, residual, seed, separation-certificate, outcome, and receipt types in a new module only.
Effects-Changed: New executable may use Mut, Panic, Div, and Alloc; semantic classification has no target or material effect.
IR-Changed: none
Claims-Introduced: Garden hypothesis that Pireus can generate a reconstructible operator seed relative to a frozen finite quotient.
Claims-Forbidden: Wider algebraic, algorithmic, material, historical, scientific, patent, or priority novelty; target support; performance; claim readiness.
Assumptions: Both named Sounio parent lineages remain hash-bound and live; the v5 Q2 definition is reproducible from public frozen facts.
Write-Set: New v7 Garden, Sounio module, example, test, contracts, gates, receipts, and later parity artifacts; no frozen-parent edits.
Read-Set: Frozen Pireus v5/v6 modules and receipts; frozen Sounio XOR-operation and XOR-lowering lineage; Guardian policy/runtime.
Positive-Witness: Exactly one Sounio-produced ExistingClassBridge or OperatorSeed with 256-cell replay and all required certificates.
Negative-Witness: Injected authority, lineage, universe, stabilizer, table, result, witness, material-promotion, parity-order, and claim-order violations are denied.
Acceptance-Gate: scripts/ci/pireus_operator_novelty_feedback.sh
Integration-Target: current Pireus lineage after isolated Garden, executable, freeze, and parity commits.
Authoritative-Only-If: Sounio first result is matcher-free, complete, hash-bound, frozen before parity, and Guardian-enforced fail closed.
```

## Garden boundary

This document defines a hypothesis, an executable contract, and falsifiers. It
contains no executable semantic result. The next legal step is a separate
commit containing the first matcher-free Sounio implementation.
