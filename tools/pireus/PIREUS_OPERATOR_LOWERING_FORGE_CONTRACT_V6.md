# Pireus Operator-Lowering Forge Contract v6

Status: `SEMANTICS_FROZEN`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-LOWERING-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Authority chronology

The v6 semantic order is immutable:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The chronology that made the exact matcher admissible is:

```text
92232819d5  Garden committed
66dd1e871a  matcher-free Sounio executable committed
6d303318bf  first Guardian-authorized Sounio transcript committed
```

The matcher values in
`stdlib/hardware/pireus/operator_lowering_forge.sio` were added only after all
three commits existed. The first transcript is therefore the origin of the v6
expected result. The matcher did not create the result retrospectively.

The authority roles remain:

```text
Sounio   SEMANTIC_AUTHORITY
Lean 4   FORMAL_PARITY
Koka     EFFECT_PARITY
C++      MATERIAL_PARITY
Haskell  OPTIONAL_DENOTATIONAL_BASELINE
LLM      REVIEW_ONLY
```

Python and Rust are forbidden as generators, oracles, validators, freeze
producers, or parity legs. No disposable language may replace them as a
semantic oracle.

## Frozen parent

The immediate semantic parent is the exact v5 Pireus Quotient Novelty Forge.
The v6 executable replays the parent and requires the frozen v5 matcher before
constructing any lowering cell.

The inherited operator quotient is v5 Q2 only:

```text
operator classes = 14
representatives   = 0,1,2,3,8,9,11,12,13,15,16,17,18,19
selected child    = -1
ranking present   = false
```

v6 does not widen Q2, recompute a different operator population, or use target
behavior to select a parent child.

## Frozen central object

The semantic schema is:

```text
EmbodiedOperator<
  OperatorIdentity,
  OperatorEquivalence,
  TargetEnvelope,
  LoweringProgram,
  ProgramEquivalence,
  RefinementObligations,
  MaterialReceipts,
  Stage
>
```

The first v6 executable realizes a finite array of these cells. The material
receipt coordinate remains empty. A generated cell is not an admitted
lowering.

## Frozen graph grammar

The target-neutral node vocabulary is exactly:

```text
1 SourceLoad
2 XorPartnerMap
3 SignMap
4 OrderedProduct
5 OrderedFold<ascending_i>
6 DestinationStore
7 LanePack
8 LaneUnpack
9 TargetPrimitive
```

The finite generation axes are:

```text
operator classes  14, inherited from v5 Q2
target envelopes  4
pack widths       1,2,4,8,16
route kinds       DIRECT, COMPOSED
serializations    CANONICAL, ALTERNATE
order             ASCENDING_I only
```

The grammar executes two shape laws:

```text
dimension = 2^bits = 16
cells     = dimension^2 = 256
```

`TREE` is not an admitted order. It appears only as a deliberate sabotage of
the program-equivalence key. v6 does not claim to enumerate all lowering
graphs, all schedules, all vector widths, or all orders.

## Exact first atlas

Sounio produced:

```text
candidate cells                   1120
generation attempts               1120
capacity overflows                   0
declared program classes           560
declared machine-envelope classes    4
```

The count is the executed finite product:

```text
14 * 4 * 5 * 2 * 2 = 1120
```

The program quotient ignores only serialization in this grammar. Every
program class therefore has one canonical and one alternate serialization:

```text
1120 / 2 = 560
```

The executable performed:

```text
program witness checks          1120  failures 0
cross-class separation checks 626080  failures 0
machine-envelope checks         1120  failures 0
serialization collapse checks    560  failures 0
tree-order sabotage checks       1120  failures 0
```

`626080` counts only pairs assigned different program classes. It is the full
unordered 1120-cell pair census minus the 560 canonical/alternate pairs:

```text
(1120 * 1119 / 2) - 560 = 626080
```

This is an exact quotient of the declared finite v6 grammar. It is not a
global program-equivalence theorem.

## Semantic reconstruction scope

For each of the 14 inherited representatives, the executable traverses all
256 `(source,destination)` cells and checks the XOR destination identity:

```text
partner = source XOR destination
source XOR partner = destination
```

The executed counts are:

```text
parent class-relation checks       28  failures 0
partner-index checks             3584  failures 0
source-order construction checks 3584  failures 0
destination identity checks      3584  failures 0
sign-word range checks             112  failures 0
sign-word round trips              112  failures 0
```

The eight sign words for every representative are unpacked cell by cell and
repacked, then compared with the hash-bound parent table. This is a
serialization round trip against the frozen parent. It is deliberately marked
`semantic_decomposition_definitional=true`. It is not an independent proof of
the parent sign table, a numerical equivalence proof, or a discharge of target
denotation, precision, ordering, effect, or material obligations.

## Target envelopes

The first executable contains four canonical semantic envelopes:

```text
726101  XEON
726102  APPLE_SILICON
726103  DGX
726104  U250
```

Their physical endpoint requirements are frozen as:

```text
XEON           1
APPLE_SILICON  1
DGX            2
U250           2
total          6
```

All observation and material-receipt fields are zero in the semantic
executable. This does not deny the already committed v5 material observations.
It prevents later material evidence from being smuggled into the first v6
semantic result. The v6 material-machine quotient is `NOT_COMPUTED`.

The target primitive labels are candidates only:

```text
1 XEON_SELECTOR
2 APPLE_TABLE
3 DGX_SUBGROUP_XOR
4 U250_FABRIC
```

They do not assert instruction denotation, compiler emission, endpoint
availability, cost, performance, or complete lowering.

## Typed residual result

Every generated cell has three discharged structural obligations and nine
unresolved target/material obligations:

```text
cells                       1120
discharged obligations      3360 = 1120 * 3
unresolved obligations     10080 = 1120 * 9
refuted obligations            0
```

The residual taxonomy produced:

```text
LoweringSeed   560
PrimitiveSeed  420
FabricSeed     140
OperatorSeed     0
total         1120
```

The `LoweringSeed` count is the composed route for every operator, target, pack
width, and serialization. The `PrimitiveSeed` count is the direct route for
the three fixed-architecture envelopes. The `FabricSeed` count is the direct
U250 route. The current grammar emits no target-derived operator semantics, so
`OperatorSeed=0` is a frozen first result rather than a hidden success.

Every residual remains nonzero. Therefore:

```text
admitted lowerings  0
selected candidate -1
ranking present     false
```

No cost or performance model pruned the semantic population.

## Three quotient boundary

The frozen scopes are:

```text
Q_operator  inherited exact v5 Q2
Q_program   exact only for the declared finite v6 grammar
Q_machine   exact only for declared target-envelope identity
```

The material-machine quotient is not computed. The product atlas therefore
types lowering, primitive, and embodiment novelty coordinates without claiming
that any coordinate is historically new, materially realized, useful, or
optimal.

## Negative witnesses

The Sounio executable passes 21/21 negative witnesses. They include:

- Python and Rust authority attempts;
- C++ and LLM semantic-authority attempts;
- missing policy and missing/unfrozen/unbound parent;
- wrong grammar dimensions;
- expected-result injection;
- selection and ranking before freeze;
- material and parity activity before freeze;
- claim and review promotion;
- invalid founder waiver;
- failure to collapse two serializations of the same graph;
- collapse of ascending and tree order;
- endpoint erasure;
- U250 fabric-residual fallback.

Before the first process dispatch, the native Guardian independently denied:

```text
Python             E110
Rust               E110
policy missing     E101
policy timeout     E102
policy error       E103
C++ semantic write E113
C++ expected write E114
review promotion   E119
parity pre-freeze  E112
```

The Python and Rust processes were not launched.

## Frozen digests

The complete candidate arrays, quotient assignments, witnesses, residual
masks, seed kinds, and obligation masks are absorbed into `atlas_digest`.

```text
lineage 1362486852:1199191553:3136252067:3927939706:743362006:2548388592:367445655:294740523
grammar 1741576844:3511803475:4114603717:1484560359:983396196:1362631209:2090336790:2466145555
targets 1178030270:2553538819:3882705700:3211727400:3843654613:275317601:1148804265:1162831727
semantic 284432857:519474153:1234600921:2759907709:1151458261:3046043008:3035575678:2657662517
atlas 1550165084:3733282509:3323663250:871519586:3948427729:1163695456:1069774959:2729693357
receipt 1431375457:687484805:1164180164:195578933:3568044397:441466530:1518228999:2908285955
forge 2539393129:4020369131:3147403558:2306440881:94983304:453189920:2257839762:3786373918
```

The committed first transcript is:

```text
tools/pireus/evidence/operator_lowering_forge_v6.txt
sha256 f7dd2398e3c0568f11e1cca5d2712fbe67169771bc2ade53171215f60197e689
```

## Parity boundary

After the freeze receipt is sealed, parity languages may compare only these
frozen semantics and scopes.

Lean may prove the finite quotient and obligation invariants. Koka may compare
the authority/effect transitions. C++ may observe target material facts. None
may add a candidate, change an expected value, discharge an obligation by
performance, create an `OperatorSeed`, or select a lowering.

## Explicit non-claims

v6 does not claim:

- a novel mathematical operator beyond the frozen v5 population;
- a historically novel lowering, instruction, or fabric;
- a complete x86, Apple, PTX, Metal, XRT, or U250 denotation;
- a complete material-machine quotient;
- a compiler-emitted lowering on any target;
- a cost or performance result;
- optimality;
- sub-quadratic twisted convolution;
- parity completion;
- claim readiness or priority.

The exact frozen result is narrower and more useful: Pireus can now generate a
typed, quotient-aware atlas of operator-lowering possibilities and turn every
missing material span into a replayable seed without inventing a successful
lowering.
