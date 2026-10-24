# Garden: Pireus Operator-Lowering Forge v6

Status: `GARDEN`

Concept-ID: `SOUNIO-PIREUS-OPERATOR-LOWERING-FORGE`

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

Founder direction preserved:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

Pireus v3-v5 established a Sounio-first path from a finite operator grammar to
generated children and then to an exact, parent-relative quotient. The material
parity leg of v5 established a second boundary: the canonical target classes are
real, but target observation alone neither lowers an operator nor proves that a
lowering preserves it.

v6 joins those two bodies of work without collapsing them. Pireus becomes a
bidirectional operator-machine discovery system:

```text
operator semantics -> candidate programs -> existing machines
        ^                    |
        |                    v
new operator seed <- typed residual <- missing machine capability
```

The compiler direction asks how a frozen operator can inhabit a machine. The
inverse direction asks what semantic primitive, instruction, or fabric would
make an otherwise blocked lowering possible. A failed lowering is therefore
not only an error. When its semantic remainder is exact and replayable, it is a
seed for machine novelty.

This is stronger than an instruction database, a peephole selector, or an
autotuner. It is also more disciplined than unrestricted program synthesis:
every generated object has a frozen semantic ancestor, an explicit quotient,
and an evidence stage.

## Authority order

The only admissible order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The first executable representation, generation grammar, candidate identities,
semantic residuals, expected-result schema, and first results must be born in
Sounio. Lean 4 may prove formal parity, Koka may establish effect parity, C++
may establish material parity, and Haskell may provide an optional denotational
baseline only after the Sounio source and semantics are frozen by hash.

External LLMs are review-only. They may find a defect but may not produce,
confirm, select, or promote a semantic result. Python and Rust are forbidden as
generators, oracles, validators, freeze producers, or parity legs. Node, Ruby,
shell, awk, `bc`, or another disposable language may not replace them as a
semantic oracle.

The first v6 executable must contain no expected candidate count, quotient
count, admitted lowering, residual identity, invented primitive, winner,
ranking, speedup, cost, target result, or frozen result matcher. Sounio must
produce every first result.

## Frozen ancestry

v6 does not reopen or reinterpret the v5 quotient. Its immediate semantic
parent is the frozen Pireus Quotient Novelty Forge v5 chain:

```text
v3 Operator Genome
  -> v4 Cubic Operator Forge
  -> v5 Quotient Novelty Forge
  -> v6 Operator-Lowering Forge
```

The v5 population, parent-relative equivalence, representatives, witnesses,
and non-selection result remain immutable parent facts. v6 may reference a
parent member or quotient class only through its frozen identity. It may not
recreate the population from prose, reorder it, silently widen the v5
equivalence, or select a v5 child by observing a target.

The current twisted XOR problem remains the first bounded laboratory:

```text
r[d] = sum_i sigma(i, i XOR d) * a[i] * b[i XOR d]
```

with 16 inputs, 16 destinations, fixed ascending-`i` fold semantics, and the
frozen sign table. v6 does not infer or separately certify a cocycle law from
that name or table. The destination-major horizontal form is part of the
semantic ancestry. A vertical accumulation, reassociated tree, approximate
reduction, changed NaN behavior, changed signed-zero behavior, or changed
exception behavior is a different semantic proposal until a declared relation
proves otherwise.

## The central object

v6 introduces an embodied operator cell:

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

This notation is the semantic schema for v6. It does not claim that Sounio
already has dependent type syntax with this spelling.

An embodied cell is not admitted merely because its program executes. It must
carry four identities that cannot be substituted for one another:

- semantic identity: which frozen operator is being implemented;
- program identity: which typed lowering graph was generated;
- target identity: which declared capability envelope the graph inhabits;
- evidence identity: which receipts discharge which obligations.

The same operator may have many non-equivalent lowerings. The same lowering
skeleton may have several material embodiments. Two programs that print the
same finite test vector are not thereby program-equivalent. A fast target
receipt cannot repair a missing refinement witness.

## Novelty is a product, not a scalar

v6 refuses a single Boolean called `novel`. It constructs a novelty coordinate:

```text
NoveltyCoordinate = (
  operator_class,
  lowering_class,
  primitive_class,
  embodiment_class,
  evidence_stage
)
```

The coordinates mean:

- operator novelty: a semantic operator is outside the declared operator
  quotient of its frozen parent;
- lowering novelty: a semantics-preserving program is outside the declared
  program quotient for the same operator and target envelope;
- primitive novelty: a typed semantic residual requires a primitive outside
  the ingested capability ontology;
- embodiment novelty: a primitive or lowering has a new material realization;
- evidence stage: Garden, executable, frozen, parity-open, or claim-ready.

These are distinct coordinates, not a claim of statistical or algebraic
independence. One coordinate may constrain another. A known operator can receive
a novel lowering. A novel operator can lower through known instructions. A
novel primitive can implement a known operator. A first U250 embodiment need
not imply a new mathematical operator. None of these internal distinctions
establishes historical priority or public novelty.

## Three quotients

Every candidate is compared under three explicitly declared equivalences:

```text
Q_operator : semantic operators / frozen operator equivalence
Q_program  : typed lowering graphs / declared program equivalence
Q_machine  : material embodiments / declared machine equivalence
```

`Q_operator` begins with the frozen v5 relation and may be widened only by a
new Sounio-authority profile. `Q_program` may include alpha-renaming, dead
administrative nodes, target-neutral serialization, and node commutation only
when the graph certifies that the nodes share no data, effect, or order
dependency. It must preserve ordered reductions and observable effects.
`Q_machine` may abstract serial numbers or endpoint addresses only when the
target contract declares them irrelevant.

The Forge studies cells in the product space:

```text
Q_operator x Q_program x Q_machine
```

That product is the actual discovery atlas. It prevents a compiler optimization
from being mislabeled as algebraic novelty and prevents a new algebraic child
from being mislabeled as a material advance.

## A typed lowering language

Pireus must not synthesize opaque strings of opcodes. The first Sounio
executable defines a finite typed graph language whose nodes denote semantic
transformations before they name machine instructions.

The initial graph vocabulary must distinguish at least:

```text
SourceLoad
XorPartnerMap<d>
CocycleSignMap<d>
OrderedProduct
OrderedFold<ascending_i>
DestinationStore<d>
LanePack
LaneUnpack
TargetPrimitive
UnresolvedResidual
```

Target-neutral nodes may be refined by target-specific inhabitants only after
their domains, ranges, lane topology, bit behavior, memory behavior, and effects
are explicit. An instruction mnemonic is evidence about a possible inhabitant,
not its denotation.

The grammar must preserve the exact distinction between:

- permutation and table lookup;
- numeric negation and sign-bit transformation;
- scalar order and reassociated reduction;
- one-source and two-source selection;
- architectural instruction semantics and compiler emission;
- compiler emission and observed silicon behavior;
- fixed hardware and reconfigurable fabric.

This is where the unresolved `vpermps` versus `vpermi2ps` question belongs: not
as prose in a cost estimate, but as a typed arity and source-domain obligation.
The Forge must be able to emit `UNRESOLVED_SOURCE_ARITY` without guessing.

## Generation, not selection

The first v6 generation pass is exhaustive within a precommitted finite grammar.
It creates candidates; it does not choose a winner.

For each frozen operator identity and canonical target envelope, Sounio must:

1. decompose the operator into the typed semantic graph;
2. enumerate legal graph rewrites within the frozen grammar;
3. map each rewritten node to zero or more ontology-backed inhabitants;
4. emit explicit obligations for every proposed refinement;
5. compute semantic and program quotient witnesses;
6. retain nonzero residuals instead of hiding them behind fallback;
7. serialize the complete atlas without ranking or selection.

Target pressure may propose a new operator mutation, but it cannot define that
operator retrospectively. Such a proposal must return to the semantic side of
the cycle as a new `OperatorSeed`, receive an independent Sounio denotation and
lineage, and only then re-enter lowering generation.

No cost model may prune the first complete generation. Cost becomes admissible
only after semantic candidates and their unresolved obligations are frozen.

## Semantic residuals become invention seeds

When a target has no inhabitant for a semantic node, the Forge computes a typed
residual:

```text
Residual<
  RequiredDenotation,
  InputShape,
  OutputShape,
  Effects,
  Precision,
  Ordering,
  TargetEnvelope
>
```

A residual is not an instruction claim. It is the exact semantic delta between
the generated program and the target ontology.

From a nonzero residual, Sounio may derive four distinct Garden seeds:

```text
LoweringSeed     new composition of existing primitives
PrimitiveSeed    proposed instruction denotation
FabricSeed       proposed reconfigurable data path
OperatorSeed     proposed semantic operator with frozen lineage
```

This is the inverse-compiler half of Pireus. The system does not merely ask
what the machine can do. It can state, with a replayable witness, what machine
capability is missing.

A `PrimitiveSeed` must specify denotation before encoding. A `FabricSeed` must
specify stream topology, state, latency contract, precision, ordering, and host
boundary before an xclbin exists. An `OperatorSeed` must specify semantics
before a benchmark exists. The four seed kinds cannot be promoted into one
another.

## Canonical target envelopes

v6 keeps all canonical target families in the discovery atlas:

- Darwin x86_64 machines are Xeon targets;
- Linux Xeon is a canonical server target;
- Apple Silicon is a canonical CPU/GPU target family;
- both DGX Spark endpoints are canonical GPU targets;
- both AMD Alveo U250 cards are canonical reconfigurable targets.

Endpoint observations and material receipts remain separate from target-family
semantics. One observed DGX cannot discharge the second endpoint. One observed
U250 cannot discharge the second card. An unavailable endpoint remains a typed
hole, not a silently reduced target count.

Each target envelope must declare:

```text
architecture + execution domain + lane topology + precision
+ memory spaces + effects + ordering + capability facts + evidence stage
```

The Forge may also emit a counterfactual target envelope for a `PrimitiveSeed`.
Counterfactual targets are useful design objects, but they have zero material
evidence and cannot be counted as canonical hardware observations.

## Why U250 changes the programme

Xeon, Apple Silicon, and DGX constrain Pireus to instructions already chosen by
other architectures. The U250 path allows Pireus to test the inverse direction:
materialize a residual as a data path whose primitive semantics were generated
from the operator rather than inherited from an ISA.

For the twisted XOR laboratory, candidate fabric questions include:

- whether XOR partner addressing should be wiring, RAM indexing, or streaming;
- whether cocycle signs should be stored, generated, or fused into arithmetic;
- whether all destinations share a permutation network;
- how exact ascending-`i` order is preserved;
- which precision and exception model is materialized;
- whether a discovered primitive remains useful across several operator classes.

These are questions, not current answers. v6 creates the typed surface on which
an answer can later become a Sounio semantic artifact, a frozen fabric contract,
and only then a U250 material receipt.

## Ontology role

The x86, AArch64, Apple, PTX, CUDA, Metal, PCIe, XRT, and FPGA ontologies supply
capability facts and provenance. They do not become semantic authority merely
because they are vendor-authored.

Every imported fact used by generation must carry:

```text
source identity
source hash
extractor identity
extractor hash
architectural scope
target scope
semantic completeness state
provenance stage
```

Missing, contradictory, partial, or version-mismatched ontology facts must fail
closed or remain explicit holes. A mnemonic match cannot satisfy a denotation
obligation. A toolchain report cannot satisfy a silicon observation. A silicon
observation cannot satisfy an operator proof.

## Refinement obligations

Every generated lowering carries an obligation ledger. The ledger must keep at
least these classes separate:

- semantic decomposition completeness;
- node-local denotation preservation;
- graph composition preservation;
- precision and exceptional-value preservation;
- reduction-order preservation;
- memory and effect preservation;
- target capability provenance;
- compiler emission evidence;
- material execution evidence;
- cost observation;
- performance observation;
- quotient witness completeness.

An obligation may be `DISCHARGED`, `REFUTED`, or `UNRESOLVED`. It may not be
implicitly true. A refuted candidate remains in the atlas as diagnostic
evidence unless its serialization is malformed.

Lean parity may discharge formal obligations only after freeze. Koka parity may
compare the effect ledger only after freeze. C++ may observe the material target
only after freeze. None may add a missing Sounio candidate, expected result, or
semantic relation.

## First executable boundary

The first Sounio executable is intentionally claim-closed. It must produce:

- frozen parent bindings;
- the complete precommitted graph grammar;
- generated operator-lowering cells;
- the three quotient partitions and their witnesses;
- an obligation ledger per cell;
- typed residuals and seed kinds;
- canonical target envelopes with unresolved endpoints retained;
- nonzero digests over grammar, atlas, quotients, obligations, and residuals;
- adversarial negative witnesses;
- zero rankings, zero selected candidates, zero cost records, zero performance
  records, zero claim promotions.

The first executable must not execute target binaries, query remote endpoints,
read later parity artifacts, or use material behavior to change candidate
semantics. The native Guardian must authorize before execution and must fail
closed on missing policy, error, or timeout.

Only after the first transcript is committed may the exact Sounio result be
encoded into a frozen matcher and semantics receipt. Only after that freeze may
parity open.

## Required adversarial witnesses

The first executable and gate must deliberately reject or preserve as unresolved
at least these counterexamples:

1. a target mnemonic is present but its denotation is absent;
2. a one-source selector is laundered into a two-source selector;
3. an ascending fold is replaced by a tree reduction;
4. numeric negation is substituted for sign-bit transformation without a
   declared exceptional-value relation;
5. a material benchmark is offered as a semantic proof;
6. a target observation is offered without its source and hardware hashes;
7. a parity receipt attempts to create a new expected result;
8. a review-only LLM receipt attempts semantic promotion;
9. a Python or Rust process is offered as an oracle;
10. a cost model prunes a candidate before semantic freeze;
11. a v5 child is selected retrospectively from target performance;
12. the second DGX endpoint is silently merged with the first;
13. the second U250 card is silently erased;
14. a counterfactual primitive is counted as existing hardware;
15. two serializations of the same graph fail to collapse under the declared
    program quotient;
16. two graphs with different observable order collapse under that quotient;
17. a residual is discarded by fallback instead of becoming a typed hole;
18. a material language attempts to write operator semantics;
19. a parity or review receipt attempts to become semantic authority;
20. policy absence, Guardian error, or Guardian timeout attempts execution.

Each negative must be exercised before the positive execution path. Detection
after target execution is insufficient.

## Falsifiers

The v6 architecture is demoted or refuted if any of the following occurs:

- the generated atlas depends on target results that did not exist at freeze;
- an operator identity changes when only its target envelope changes;
- a claimed lowering cannot produce a replayable refinement ledger;
- the program quotient collapses observably different ordered reductions;
- a residual cannot be reconstructed from the blocked graph and target facts;
- the same frozen inputs produce different candidate identities;
- missing ontology or target evidence silently becomes false or zero;
- a material receipt changes a Sounio semantic result;
- any non-Sounio language creates the first expected result;
- any candidate is ranked or selected before the claim boundary permits it.

A zero-candidate or all-residual result would not refute the architecture. It
would be a valid first measurement of the declared grammar. A failure to derive
useful primitive or fabric seeds would narrow the inverse-compiler thesis but
must not be hidden by widening the grammar after seeing results.

## Explicit non-claims

At Garden stage, v6 does not claim:

- that any generated operator is historically new;
- that any generated lowering is globally optimal;
- that the v5 population contains a useful operator;
- that an instruction seed is implementable in fixed silicon;
- that an FPGA seed fits, routes, meets timing, or outperforms a CPU/GPU;
- that `vpermps`, `vpermi2ps`, TBL/TBX, `shfl.sync`, Metal, or any U250
  primitive is a complete lowering;
- that semantic equivalence follows from finite numeric agreement;
- that the three declared quotients cover all meaningful equivalences;
- that material parity is performance parity;
- that internal novelty coordinates establish publication priority.

## The larger Pireus

If v6 survives, Pireus becomes a harbor in the technical sense: semantic
objects arrive with lineage, machines arrive with capabilities and provenance,
and the system constructs typed routes between them. Where no route exists, it
does not invent a success. It returns the exact missing span as a design object.

That permits a longer programme:

```text
operator generation
-> equivalence discovery
-> lowering generation
-> residual extraction
-> instruction generation
-> FPGA fabric generation
-> material parity
-> cost and performance measurement
-> bounded claim
```

Pireus would then be able to discover not only operators, and not only ways to
compile them, but also the machine vocabulary that their faithful execution
requires. v6 plants that capability as a falsifiable architecture. It does not
pretend that the architecture has already been executed.
