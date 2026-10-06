# Pireus Operator Genesis

Status: `SEMANTICS_FROZEN`

Concept-ID: proposed `SOUNIO-PIREUS-OPERATOR-GENESIS`

The registry row remains pending while `docs/internal/concepts/registry.tsv` is
owned by another live lane. This draft contract and its canonical Sounio file
reserve the concept without writing through that ownership.

## Semantic lane

Semantic-Lane-ID: `pireus-operator-genesis-20260828`

Owner: `founder`

Concept-IDs: proposed `SOUNIO-PIREUS-OPERATOR-GENESIS`;
`SOUNIO-XOR-CONVOLUTION-COCYCLE`;
`SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION`;
`SOUNIO-SCIENCE-RESEARCH-BOUNDARY`.

Intent-Preserved: Pireus must generate and attack operator candidates without
letting a parity language, hardware target, or external reviewer create the
operator meaning or expected result retrospectively.

Transformation: add a bounded Sounio-owned operator grammar, exact finite
canonicalization, exhaustive comparison against a hashable internal corpus,
and scope-bearing novelty receipts.

Types-Changed: none. New types are `OperatorSpace`, `OperatorGrammar`,
`EquivalenceGroup`, `GeneratedCandidate`, `InequivalenceWitness`,
`OperatorFingerprint`, `GenesisSearch`, and `NoveltyReceipt`.

Effects-Changed: none.

IR-Changed: none.

Claims-Introduced: after the executable gate passes, Sounio can state bounded
semantic novelty and that a selected candidate is inequivalent to every member
of the declared corpus under every action of the declared finite group. The
scope is exactly the frozen corpus and group; `relative_algebraic_novelty`
remains false because the action universe is not a complete algebraic
classification.

Claims-Forbidden: historical or global novelty; priority; completeness under
`GL(4,2)`, sign gauges, isotopy, or arbitrary basis changes; algorithmic
novelty; sub-quadratic complexity; material novelty; speedup; scientific
novelty; formal parity; effect parity; hardware parity; `CLAIM_READY`.

Assumptions: the imported `cd_sigma` implementation is the live Sounio
authority for the base twist; the 16-candidate grammar and 48-action group are
finite boundaries rather than completeness claims; SHA-256 binds serialized
inputs but does not prove mathematical correctness.

Write-Set: `tools/pireus/GARDEN_PIREUS_OPERATOR_GENESIS_V0.md`, this contract,
`stdlib/hardware/pireus/operator_genesis.sio`,
`examples/pireus_operator_genesis.sio`,
`tests/stdlib/hardware/test_pireus_operator_genesis.sio`, and later freeze,
evidence, and gate files named by the Garden.

Read-Set: `stdlib/algebra/cayley_dickson.sio`,
`stdlib/algebra/xor_convolution.sio`, the Pireus XOR semantic and lowering
contracts, the canonical Loom language-authority Guardian, and repository
governance.

Positive-Witness: the Sounio executable enumerates all 16 grammar members,
canonicalizes each across 48 actions, performs 144 corpus comparisons per
candidate, selects the maximum minimum-distance member, and emits nonzero
digests without imported expected results.

Negative-Witness: mutations refuse missing policy/grammar/corpus/group,
incomplete search counts, Python/Rust oracles, C++/LLM semantic authority,
parity before freeze, broad novelty promotions, fingerprint-only proof,
incomplete exhaustion, zero-distance novelty, and malformed waivers. The final
gate must also submit an actual Python-oracle frame to the canonical Sounio
Guardian and observe pre-execution refusal.

Acceptance-Gate: `scripts/ci/pireus_operator_genesis.sh`. It pins the Garden,
first executable, frozen source, base twist, semantics bundle, toolchain,
hardware, command, result, transcript, Guardian frames, and exact negative
decisions; replays Sounio byte-for-byte; and refuses a tampered transcript.

Integration-Target: `stdlib/hardware/pireus` and later ontology-derived lowering
campaigns for canonical Xeon, DGX, Apple Silicon, and U250 material voters.

Authoritative-Only-If: the first candidate and expected result are emitted by
Sounio before any parity implementation exists; the freeze binds the exact
Sounio source, imported base twist, Garden, contract, grammar, group, corpus,
transcript, and receipt; and the Guardian admits that lineage.

## v0 executable boundary

For `m` from 0 through 15, the grammar defines

```text
phase_m(i,j) = (-1)^parity(m AND i AND j)
sigma_m(i,j) = cd_sigma(i,j,4) * phase_m(i,j)
```

The action group is `S4 x C2`: the coordinate-permutation group of order 24 on
the four XOR bits, composed with optional left/right operand exchange. For a
coordinate permutation `pi`, the executable action is exactly

```text
T_(pi,0)(i,j) = T(pi(i),pi(j))
T_(pi,1)(i,j) = T(pi(j),pi(i)).
```

Because `pi` is linear over the XOR coordinates, it relabels the destination
consistently. Using `pi` rather than `pi^-1` does not change the enumerated
orbit because all 24 permutations are present. The canonical table is the
lexicographic minimum of the 48 transformed 256-cell sign tables, with
`-1 < +1`; the lowest action ID breaks equal-table ties.

The internal corpus contains untwisted XOR, the live Cayley-Dickson-16 table,
and the diagonal bicharacter `(-1)^parity(i AND j)`. A candidate's score is the
minimum Hamming distance from its canonical table to every transformed corpus
table. The exhaustive search maximizes this score, breaking ties by seeded
generation order.

## Frozen result

The ordered phase ancestry is:

```text
GARDEN             aaef53eb0a6f15a6d0041f347cc107ed69310de4
SOUNIO_EXECUTABLE  d034de5927eee7e4382c39926c5d5ab79a347a79
SEMANTICS_FROZEN   999efd27bc6def4cf0756f870568302070659363
FREEZE_GATE        0de10c556f419b01b71597e4f3cdc2193836e3f9
STATIC_CHECK_FIX   4ce3307d544d5191a68c2372dfc266ced526a70a
```

The first executable commit contains no exact matcher. The later freeze admits
the Sounio-produced winner. `STATIC_CHECK_FIX` removes only terminal commas from
fixed-size multiline array literals after `souc check` exposed its strict
arity parser; both executable consumers pass `check`, and the authority output
remains byte-identical to the original freeze:

```text
phase_mask=13
score=100
canonical_action=19
nearest_corpus=CAYLEY_DICKSON_16
nearest_corpus_action=17
minimum_hamming_distance=100
witness=(i=1,j=4,candidate=-1,corpus=+1)
positive=144
negative=112
commutator_defects=210
associator_defects=1848
displacement_negative_counts=7 repeated 16 times
```

These values are execution goldens for the hash-bound v0 Sounio program. They
are not universal identities derived from Cayley-Dickson axioms.

The accepted interpretation is:

```text
relative_semantic_novelty=true
declared_action_inequivalence=true
relative_algebraic_novelty=false
algebra_isomorphism_complete=false
global_novelty=false
scientific_novelty=false
parity_open=false
claim_ready=false
```

Lean, Koka, C++, Haskell, target lowering, and material execution remain closed
until a separately Guardian-admitted parity action names this exact semantics
hash.

## Integration receipt

Semantic-Outcome: complete bounded Sounio search with a frozen winner and
declared-action inequivalence witness.

Concept-Status-Before: Garden.

Concept-Status-After: `SEMANTICS_FROZEN`, proposed registry row pending.

Distinctions-Added: semantic novelty versus declared-action inequivalence
versus complete algebraic classification versus historical/scientific novelty.

Distinctions-Preserved: Sounio semantic authority; nonassociative structure;
finite equivalence scope; review-only LLMs; material voters separated from
semantic truth.

Distinctions-Erased: none.

Evidence-Run: exact Sounio example and test; 16 candidates; 48 actions; 144
comparisons per candidate; 24/24 in-module refusals; native Guardian `E110`
Python refusal before process launch; `E101` missing policy; `E102` timeout;
`E119` LLM promotion; `E113` C++ authority; transcript tamper refusal.

Fallback-Path: explicit `SOUNIO_SOUC_ENGINE=lean_single` bootstrap fallback via
`./bin/souc`; no raw ELF and no default Madaros result used as authority.

Legacy-Kept: all existing Pireus XOR operation, lowering, target, and material
paths.

Conflicting-Lanes: none on code; `docs/internal/concepts/registry.tsv` remains
owned by the live Loom lane.

Next-Semantic-Interface: expand the equivalence universe with sign gauges and
selected `GL(4,2)` transformations, then open formal/effect/material parity on
the unchanged frozen v0 candidate.
