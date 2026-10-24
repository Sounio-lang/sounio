# Garden: Pireus Operator Genome v3

Status: `GARDEN`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-GENOME`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Operator Genesis bilinear v2 discovered a bounded relative-semantic novelty,
selected the exact operator represented by packed matrix `B=1128`, and stopped
before lowering. That stop was correct, but discovery alone is not the Pireus
we want. Pireus must turn a generated operator into an executable semantic
object that can demand implementations from radically different materials
without letting any material redefine the operator.

The v3 question is fixed before an executable, a sign mask, a microprogram
digest, a target plan, or an expected result exists:

> Can Sounio derive a complete, canonical, target-neutral OperatorGenome from
> the frozen v2 winner, execute that genome directly, and emit a sealed set of
> materialization obligations for Xeon, Apple Silicon, DGX, and dual AMD Alveo
> U250, while preserving exact ordered semantics and keeping every material
> claim closed?

The intended answer is not fixed by this Garden. Sounio must answer first.

## Authority order

The only admissible order remains:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The first Sounio executable contains no expected sign masks, per-displacement
counts, target-envelope digest, evaluation vector, result vector, or genome
digest. Lean, Koka, C++, Haskell, Xeon, Apple Silicon, DGX, U250, and external
LLMs may compare, prove, or measure only after a hash-frozen Sounio result
exists.

Python and Rust are forbidden as generators, oracles, validators, freeze
producers, or parity legs. A shell, Node, Ruby, awk, bc, or another disposable
language may not be substituted as a semantic oracle. Cryptographic hashing,
file comparison, process control, and fail-closed policy enforcement remain
non-semantic tooling.

## Frozen parent identity

The semantic parent is the already frozen Sounio v2 result:

```text
source_sha256     = 31f5fe668c100f0aa27b4c4405c022c127e5445a743d5029e2d913da8dfd8a44
semantics_sha256  = bb5560806ea7a84a0cc5f88ec5d4adbea4004ec6b2560af6e4d8de31b3a88d3b
transcript_sha256 = d8fa8bac03d9b09f970f6bd328f9b295165c1e56823c799a46771886123cacd0
selected_sha256   = a264defd7a6af854ccfa1cc1a7239c505bfc2bc0ea8dee93b17dd09952d96443
contract_sha256   = 0cb51e12e17be8500be1de679c9ce95d67b8dbffb0750be511833cb76d8548e8
freeze_sha256     = 38f4d5c0a46029283bc21fd901a60e1f7f08332b48317fd40548abf91fe2e6aa
cd_sigma_sha256   = e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed
```

The selected semantic identity is:

```text
bits                  = 4
dimension             = 16
base_twist            = Cayley-Dickson-16
class_id              = 26
quadratic_code        = 198
packed_bilinear_matrix= 1128
class_quadratic_size  = 28
class_raw_size        = 1792
square_negatives      = 5
commutator_defects    = 90
associator_defects    = 1848
```

v3 may derive from this identity. It may not rerun selection with a new
objective, choose another representative, widen the corpus, or promote the
bounded v2 novelty receipt.

The frozen material-neutral operation and legality parents remain informative
constraints, not authority over the new sign table:

```text
xor_operation_source_sha256 = bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8
xor_legality_source_sha256  = 7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
```

They establish the five semantic nodes, ascending-`i` reduction, and
nonassociative barriers. They do not already describe `B=1128`; v3 must derive
that operator rather than pretending the old Cayley-Dickson masks are enough.

## OperatorGenome

An `OperatorGenome` is the canonical executable normal form of one generated
operator. It is neither a source-language AST nor a target instruction list.
It binds four layers:

1. `Lineage`: exact parent hashes and the selected v2 semantic identity.
2. `Genotype`: the base twist and packed bilinear phase that define every sign.
3. `MicroProgram`: every partner, sign, dependency, and ordered reduction.
4. `MaterializationEnvelope`: target-independent obligations plus four target
   projections that may be solved only after freeze.

Two genomes are byte-identical only when all four layers serialize identically.
A material schedule, cost, instruction name, driver version, or machine address
is not part of genome identity.

`Canonical` in v3 means canonical under the lineage, coordinate convention,
normal form, and serialization fixed by this contract. It is not a theorem that
no other algebraic presentation can denote an isomorphic operator.

## Genotype law

Let `V=F2^4`, with indices serialized as integers in `[0,15]`. The parent
operator is fixed by:

```text
b_B(i,j)     = i^T B j                         in F2
sigma_B(i,j) = cd_sigma(i,j,4) * (-1)^b_B(i,j)
r[d]         = sum_(i=0..15) sigma_B(i,i XOR d)
               * a[i] * b[i XOR d]
```

`B=1128` uses the same low-nibble-first row encoding frozen by v2. Sounio must
derive `b_B` from the packed matrix and must call the live Sounio
`cd_sigma(i,j,4)` base twist. A copied 256-cell golden table is not an
admissible first executable.

The coordinate and matrix bit order are explicit. Coordinate `r` of an integer
index `x` is

```text
x_r = (x >> r) AND 1, for r in [0,3].
```

Matrix row `r` is nibble `r` from the least-significant end and column `s` is
bit `s` of that nibble:

```text
B_rs = (1128 >> (4*r+s)) AND 1
b_B(i,j) = XOR_(r=0..3,s=0..3) (i_r AND B_rs AND j_s).
```

Thus the packed rows of `1128 = 0x0468` are `(8,6,4,0)`. This statement fixes
representation only; the executable still derives all cells rather than
copying a table.

The inherited base function is exactly
`stdlib/algebra/cayley_dickson.sio::cd_sigma` at the source hash above, with
the convention `e_i * e_j = cd_sigma(i,j,bits) * e_(i XOR j)`. On the v3
domain Sounio must check extensionally that every returned value is either
`-1` or `+1` before deriving a sign bit. A zero or any other value invalidates
the genome.

The genome stores the derived sign bit rather than a floating sign value:

```text
sign_bit(i,j) = (cd_sigma(i,j,4) < 0) XOR b_B(i,j).
```

This makes sign application an exact Boolean obligation. Numeric multiplication
still follows the selected scalar representation at execution time.

## Displacement-normal microprogram

The canonical microprogram is displacement-major. For every displacement
`d in [0,15]` and source index `i in [0,15]`, Sounio derives one cell:

```text
destination = d
lhs_index   = i
rhs_index   = i XOR d
sign_bit    = sign_bit(i, i XOR d)
ordinal     = i
```

There are exactly 256 cells. The cell order is `(d,i)` ascending, with `d` as
the major coordinate. This is a serialization rule, not an expected semantic
digest.

Each displacement is additionally split into two fixed eight-lane groups:

```text
group(d,chunk,lane):
  i         = 8*chunk + lane
  rhs_index = i XOR d
  sign_bit  = sign_bit(i,rhs_index)

d     in [0,15]
chunk in [0,1]
lane  in [0,7]
```

There are exactly 32 groups and 256 lane cells. This eight-lane projection is
the common comparison surface for the existing Xeon selector ontology. It is
not a claim that every target has eight physical lanes.

For `d<8`, `rhs_index` remains in the source chunk. For `d>=8`, it crosses to
the opposite chunk. Sounio must derive and record this fact per group. A target
may use a different physical decomposition, but its material proof must map
back to the same 256 canonical cells.

Each group records:

- displacement and chunk;
- eight RHS indices in lane order;
- an eight-bit negative mask;
- source-chunk and cross-chunk classification;
- first and last global ordinals;
- dependency barrier identifiers.

No expected mask or negative count is fixed here.

## Executable meaning

The genome must be directly executable in Sounio. Given two 16-element input
vectors, the reference evaluator executes each output lane in exact ascending
`i` order:

```text
acc_0 = signed_product(0)
acc_1 = acc_0 + signed_product(1)
...
acc_15 = acc_14 + signed_product(15)
r[d] = acc_15
```

The reference path must be checked against an independently structured direct
evaluation of the same genotype law inside Sounio. The two paths may share
`cd_sigma` and the packed-matrix law, but they may not share a precomputed sign
table or a target lowering. Extensional cell checks establish equality of all
partner, sign, destination, and ordinal fields. The numeric fixture checks one
exact execution of that already cell-equal plan; it is not a universal
floating-point equivalence theorem.

The first executable chooses deterministic integer-valued `f64` inputs whose
products and reductions are exactly representable. This makes equality a
bit-exact semantic check rather than a tolerance chosen after observing a
result. The input vectors and output vector are outputs of the first Sounio
execution, not Garden goldens.

The executable also verifies every one of the 256 partner cells and every
microprogram ordinal. An output-vector match alone is insufficient because
different erroneous cell plans can collide on one fixture.

## Semantic DAG and barriers

Every cell instantiates the frozen five-node shape:

```text
XOR_PERMUTE
-> TWIST_APPLY
-> MULTIPLY
-> HORIZONTAL_REDUCE
-> OUTPUT_LANE
```

`TWIST_APPLY`, `HORIZONTAL_REDUCE`, and `OUTPUT_LANE` remain semantic barriers.
For strict execution, a target must preserve the exact ascending-`i` reduction
tree. Reassociation, FMA contraction, approximate arithmetic, stochastic
rounding, and transform substitution are separate modes that require separate
Sounio contracts and receipts. None is authorized by v3.

No Walsh-Hadamard or subquadratic rewrite is authorized. The genome preserves
enough structure for a later transform search to ask that question exactly;
it does not answer it.

## MaterializationEnvelope

The envelope is a set of proof obligations, not a lowering. Every target
projection must eventually discharge:

1. `PartnerCoverage`: every canonical `(d,i)` cell maps to exactly one target
   product path.
2. `SignCoverage`: every derived sign bit is applied exactly once before its
   ordered addition.
3. `NoExtraCell`: no target product contributes outside the 256-cell genome.
4. `DestinationPreservation`: all cells for displacement `d` contribute only
   to output `d`.
5. `OrderedReduction`: strict mode preserves ascending `i` and all barriers.
6. `ScalarContract`: input, product, accumulator, and output representations
   are explicit.
7. `MemoryContract`: movement does not alias, drop, duplicate, or reorder a
   semantic cell.
8. `LineageBinding`: the material receipt binds this frozen genome hash.
9. `ReplayBinding`: command, toolchain, hardware, source hash, result, and
   policy decision are recorded.
10. `ClaimBoundary`: implementation success does not promote algorithmic,
    material, scientific, historical, global, or priority novelty.

A target solver may fuse nodes only when its proof reconstructs this envelope.
An instruction catalog match is evidence of a candidate implementation, not a
semantic proof.

## Four canonical target projections

The first executable declares exactly four target families:

```text
DARWIN_XEON         = 701200
APPLE_SILICON       = 701201
DGX_SPARK           = 701202
AMD_ALVEO_U250_DUAL = 711001
```

Declaration creates unresolved obligations only. It is not observation,
capability, lowering, cost, performance, or parity evidence.

### Xeon

The projection exposes the 32 eight-lane groups, including same-chunk and
cross-chunk RHS selection, exact sign masks, and the ordered 16-term reduction.
Later material search may ask whether one-source permutes, two-source permutes,
mask operations, loads, or another legal schedule realize each obligation.
No AVX width or instruction mnemonic is selected here.

### Apple Silicon

The projection exposes the same canonical cells and leaves CPU, GPU, Metal,
NEON, SVE-family assumptions, lane width, and instruction choice unresolved.
Apple family declarations and observed engines may constrain later search but
may not alter the genome.

### DGX

The projection exposes the same canonical cells and leaves CPU/GPU placement,
warp organization, shared-memory staging, shuffle form, PTX/SASS identity, and
instruction choice unresolved. A PTX operation name is not material evidence
of an emitted SASS path.

### Dual AMD Alveo U250

The projection binds the already declared dual-card target family, two engine
slots, XCU250 fabric identity, and four DDR banks per card as target lineage.
It leaves kernel existence, XRT execution, data partition, inter-card movement,
pipeline initiation interval, resource use, timing closure, and correctness
unresolved.

The two cards do not imply that the 16 outputs are split eight-and-eight. A
later Sounio-authored material plan must choose and bind any partition. The
second card may remain an unresolved slot without invalidating the genome.

## Genome-to-phenotype search boundary

After `SEMANTICS_FROZEN`, Pireus may open independent phenotype searches:

```text
OperatorGenome
-> obligation hypergraph
-> target capability candidates
-> legal schedule candidates
-> material execution candidates
-> measured receipts
-> admitted phenotype
```

This is where the ingested processor ontologies become generative rather than
documentary. They may enumerate ways to satisfy the genome, reject impossible
schedules, and rank admitted schedules by explicit objectives. They may not
invent expected semantic results or silently modify the operator.

Each phenotype is a projection of one genome. Many phenotypes may coexist;
there remains one semantic authority.

Future operator search may use admitted phenotype costs as a new objective,
but that creates a new Garden and a new operator-generation generation. It may
not retrospectively change v2 selection or v3 identity.

## Canonical serialization and digests

The first executable must produce independent Sounio SHA-256 digests for:

- frozen-parent lineage;
- genotype identity;
- 256-cell displacement microprogram;
- 32-group eight-lane projection;
- strict reference evaluation fixture;
- four target envelopes;
- bounded receipt;
- complete genome.

Integer fields serialize as signed 64-bit big-endian values. Booleans serialize
as `0` or `1`. Arrays serialize in the orders fixed above. Text labels do not
enter semantic digests; stable numeric identities do.

No digest literal appears in the first executable. Digests become expected
values only after a committed matcher-free executable produces the first
transcript and that transcript is audited.

## Bounded receipt

At `SOUNIO_EXECUTABLE`, the receipt may establish only that:

- the v2 parent is admitted and exactly bound;
- the selected genotype is reproduced from the parent result;
- all 256 canonical cells are derived;
- all 32 comparison groups are complete;
- the microprogram and direct Sounio evaluators agree;
- four unresolved target envelopes are emitted;
- hardware has not modified semantic identity.

`Complete` in this list means every declared cell and wiring obligation is
present and internally consistent. It does not establish a composition-algebra
law, norm multiplicativity, alternativity, associativity, or algebra
isomorphism. Any later observed mask, negative count, fixture output, or digest
is a hash-bound Sounio execution golden, not a symbolic closed form.

The receipt must keep all of these false:

```text
semantics_frozen
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

Relative semantic novelty remains inherited from v2 as a parent fact. v3 does
not rediscover or widen it. The new bounded statement, if Sounio establishes
it, is **executable genome derivation**: the frozen generated operator has a
complete material-neutral microprogram and explicit phenotype obligations.

## Negative certificates

The first executable must deliberately refuse at least:

- Python as semantic producer;
- Rust as semantic producer;
- C++ promoted from material parity to semantic authority;
- external LLM promoted from review-only to authority;
- missing or mismatched v2 parent identity;
- a selected matrix other than the live parent result;
- pre-freeze parity opening;
- target observation promoted from declaration alone;
- a target plan that changes a partner index or sign bit;
- a target plan that duplicates or drops a cell;
- an exact-mode plan that reassociates reduction;
- a U250 declaration with any card count other than two;
- a material, algorithmic, scientific, historical, global, priority, or
  claim-ready promotion;
- an invalid founder waiver.

Waivers remain founder-only, scoped, purpose-bound, and expiring. A waiver may
authorize an action; it may not make a false semantic statement true.

## First executable protocol

The chronology is part of the evidence:

1. Commit this Garden before the executable exists.
2. Obtain review-only mathematical review; reviewers receive no expected
   result and may not execute or confirm the Sounio output.
3. Implement the Sounio module, example, and structural test without expected
   masks, results, or digests.
4. Commit the matcher-free executable before its first semantic run.
5. Route launch through the native Guardian and invoke `./bin/souc`, never a
   raw ELF.
6. Record the first Sounio transcript and independently audit internal
   arithmetic and scope.
7. Only then add frozen matchers, contract, manifest, replay gate, and receipts.
8. Open Lean, Koka, C++, Haskell, or hardware parity only from the frozen v3
   hash.

The first successful material target does not become authority over the other
three. Every parity receipt names its language role, toolchain, hardware,
command, source hash, frozen semantics hash, and result.

## What this Garden does not claim

It does not claim that `B=1128` is a new algebra, the best operator, useful for
a scientific workload, faster than Cayley-Dickson, subquadratic, patentable,
historically novel, or first. It does not claim that any canonical target can
execute it efficiently, or at all.

It fixes a stronger engineering and language thesis:

> Pireus treats a generated operator as a first-class executable genome. The
> genome owns meaning; processor ontologies compete to materialize it.

Whether the executable satisfies that thesis is left to Sounio.
