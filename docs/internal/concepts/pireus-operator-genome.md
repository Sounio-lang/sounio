<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-operator-genome
authority: repo_only
audience: users
last_validated: 2026-08-29
validated_by: codex
source_of_truth: stdlib/hardware/pireus/operator_genome.sio
-->

# Pireus Operator Genome

Concept-ID: `SOUNIO-PIREUS-OPERATOR-GENOME`

Status: `PARITY_OPEN` (Lean/Koka/C++ parity permitted, not executed)

Semantic-Lane-ID: `pireus-operator-genome-v3-20260829`

## Intent

Turn the frozen winner of Pireus Operator Genesis bilinear v2 into a canonical
executable normal form that owns semantic identity across heterogeneous
materializations.

The semantic authority pair is:

```text
stdlib/hardware/pireus/operator_genome.sio
examples/pireus_operator_genome.sio
```

The structural test is:

```text
tests/stdlib/hardware/test_pireus_operator_genome.sio
```

The matcher-free executable was committed before its first authorized Sounio
run. The first transcript was then preserved before the frozen matcher existed.
The current source, contract, freeze, transcript, and parity-opening receipt are:

```text
stdlib/hardware/pireus/operator_genome.sio
tools/pireus/PIREUS_OPERATOR_GENOME_CONTRACT_V3.md
tools/pireus/operator_genome.freeze.v3
tools/pireus/evidence/operator_genome_v3.txt
tools/pireus/operator_genome.parity-open.v3
scripts/ci/pireus_operator_genome.sh
```

## Frozen Parent

The direct parent is the frozen Sounio bilinear v2 result at matrix `1128`,
quadratic code `198`, and declared class `26`. v3 executes the live parent,
requires its frozen matcher, binds its source, semantics, transcript, selected,
contract, and freeze hashes, and preserves its bounded receipt.

The existing XOR operation and lowering-legality modules constrain the
five-node operation shape, ascending-`i` reduction, and nonassociative barriers.
They do not contain the `B=1128` sign table and are not silently redefined.

## Semantic Boundary

The genome derives:

```text
Lineage
-> Genotype
-> 256-cell displacement microprogram
-> 32 eight-lane comparison groups
-> strict reference evaluation
-> four materialization envelopes
```

For every displacement `d` and source `i`, the canonical cell is:

```text
destination = d
lhs         = i
rhs         = i XOR d
sign        = cd_sigma(i, i XOR d, 4)
              * (-1)^(i^T B (i XOR d))
ordinal     = i
```

The two Sounio evaluators share the genotype law but not a precomputed sign
table: one evaluates the formula directly and one executes the derived
microprogram. Extensional checks establish partner, sign, destination, and
ordinal identity for all 256 cells. Both evaluators then preserve the exact
ascending-`i` addition spine and compare bit-exact `f64` results on one exactly
representable integer fixture. The fixture is a smoke check, not a proof over
all floating-point inputs.

`Canonical` is scoped to the fixed lineage, coordinate convention, normal
form, and serialization. It is not a complete algebra-isomorphism theorem.

## Target Boundary

The executable declares unresolved envelopes for:

- Darwin Xeon (`701200`);
- Apple Silicon (`701201`);
- DGX Spark (`701202`);
- dual AMD Alveo U250 (`711001`).

Each envelope contains ten proof obligations: partner coverage, sign coverage,
no extra cell, destination preservation, ordered reduction, scalar contract,
memory contract, lineage binding, replay binding, and claim boundary.

All forty obligations remain unresolved in the first executable. Target
declaration is not observation. The two U250 cards are target lineage and do
not imply an eight-output-per-card partition.

## Preserved Constraints

- Sounio remains semantic authority.
- Hardware cannot change a partner, sign, destination, ordinal, or barrier.
- Strict mode forbids reassociation, contraction, transform substitution, and
  approximate arithmetic.
- No WHT or subquadratic transform is authorized.
- No instruction, lowering, cost, performance, or material parity is admitted.
- Python and Rust are forbidden semantic oracles.
- C++ remains material parity only.
- External LLMs remain review-only.

## Claim Boundary

The first executable may establish only executable genome derivation relative
to the already frozen v2 parent. It inherits v2's bounded relative semantic
novelty as lineage; it does not rediscover or widen it.

`complete` on the genotype and microprogram means complete enumeration and
wiring under this normal form. It proves no composition-algebra law, norm
multiplicativity, alternativity, associativity, or algebra isomorphism.

The observed negative count, 32 masks, fixture output bits, and digests become
hash-bound Sounio execution goldens at freeze. They are not symbolic closed
forms and no external reviewer independently derives or confirms them.

It establishes no algebraic, algorithmic, material, scientific, global,
historical, priority, or claim-ready novelty. The Sounio result is frozen at
genome SHA-256
`99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926`.
The staged receipt opens parity, while all parity implementations, material
observations, forty target obligations, and `CLAIM_READY` remain unexecuted or
unresolved.

## Pending Interface

The next interfaces are formal parity, effect parity, target schedule search,
material parity, cost, and performance. Future operator generations may use
admitted material costs without retrospectively changing this frozen genome.
