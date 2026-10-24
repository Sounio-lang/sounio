# Garden: Pireus Operator Genesis v0

Status: `GARDEN`

First phrase preserved from the founder:

> Pireus deve ser capaz de alem de tudo, gerar novelty de operadores.

This Garden turns Pireus from a machine-description and lowering system into
an executable operator-discovery system. Pireus must be able to generate a
bounded operator space, quotient it by an explicit equivalence group, attack
each surviving candidate with counterexamples, and emit a novelty receipt whose
scope is no broader than the search that actually ran.

## Authority order

The only admissible order is:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio owns the first executable grammar, corpus, equivalence group, search
objective, candidate selection, witnesses, and expected result. Lean, Koka,
C++, Haskell, hardware probes, and external LLMs may act only after the exact
Sounio source and semantics are frozen.

## Operator Genesis loop

```text
OperatorSpace
  -> GeneratedCandidate
  -> canonicalize under EquivalenceGroup
  -> compare with FrozenCorpus
  -> produce InequivalenceWitness or equivalence refutation
  -> fingerprint
  -> NoveltyReceipt
  -> target lowering and material voters
  -> retain, demote, or refute
  -> ontology update
```

No later phase may replace the candidate selected by the frozen Sounio search.
A parity implementation may expose an error in the Sounio semantics, but a
corrected search then requires a new Sounio freeze and a new receipt lineage.

## First campaign: TwistedXor16Genesis

The v0 search space is deliberately finite and exhaustible. Every candidate
has basis indices in `Z2^4`, destination `i XOR j`, coefficients in `{ -1, 1 }`,
and ascending source-index accumulation. For mask `m` in `[0, 15]`, Sounio
generates

```text
phase_m(i, j) = (-1)^parity(m AND i AND j)
sigma_m(i, j) = cd_sigma(i, j, 4) * phase_m(i, j)

r_m[d] = sum_i sigma_m(i, i XOR d) * a[i] * b[i XOR d]
```

This is a grammar of 16 sign-twisted XOR bilinear operators. It is not a claim
that the family is complete, that its members are historically new, or that
the phase is a gauge transformation. Those questions remain testable rather
than assumed.

## Frozen equivalence group

v0 admits exactly these transformations:

1. all 24 simultaneous permutations of the four bit coordinates of both
   input indices and the XOR destination;
2. optional exchange of the left and right operands.

The resulting finite group has 48 actions. A canonical form is the
lexicographically least row-major 256-cell sign table under those actions,
with `-1` ordered before `+1`. This total order makes the canonical table
unique even when the operator has a nontrivial stabilizer. If several actions
produce that same table, the canonical action is the lowest numeric action ID.

v0 does not quotient by arbitrary `GL(4, 2)` basis changes, sign gauges,
independent input/output permutations, scaling, or isotopy. A candidate that
survives v0 may cease to be novel relative to a larger equivalence group. The
receipt must name this limitation.

## Frozen internal corpus

The first corpus contains three executable operator tables:

- untwisted XOR: every coefficient is `+1`;
- Cayley-Dickson-16: `cd_sigma(i, j, 4)`;
- diagonal bicharacter: `(-1)^parity(i AND j)`.

The corpus is a regression and search boundary, not a literature database. Its
hash must bind the identities, order, dimensions, and all 768 sign cells.
`cd_sigma` is not redefined in this document: its authoritative recursive
definition is the Sounio implementation in
`stdlib/algebra/cayley_dickson.sio`. The executable imports that definition,
and the freeze binds its source hash so an implementation change cannot retain
the old corpus identity silently.

## Search objective

For every generated candidate, Sounio exhaustively computes the minimum
Hamming distance between its canonical table and every transformed member of
the frozen corpus. The candidate score is the minimum of those distances.

v0 selects the candidate with maximum score. Ties are broken by the seeded
generation order and then by candidate mask. The seed changes enumeration
order only; it cannot add or remove a member of the exhaustive 16-candidate
space.

An admitted relative-novelty result requires:

- all 16 candidates evaluated;
- all 48 group actions evaluated for each canonical form;
- all `3 * 48` candidate-to-corpus comparisons completed for the winner;
- no exact equivalence found;
- a positive minimum Hamming distance to every corpus member;
- a concrete differing cell for the nearest transformed corpus member;
- nonzero hashes for grammar, corpus, group, candidate, canonical form,
  fingerprint, witness, and receipt.

## Fingerprint v0

The selected candidate records at least:

- positive and negative coefficient counts;
- ordered commutator-defect count over all 256 pairs `(i, j)`, where a defect
  is `sigma(i, j) != sigma(j, i)`;
- associator-identity defect count over all 4096 triples `(i, j, k)`, where a
  defect is
  `sigma(i,j) * sigma(i XOR j,k) != sigma(j,k) * sigma(i,j XOR k)`;
- negative count for every XOR displacement;
- nearest corpus member and distance;
- canonical transformation;
- exact canonical-table digest.

These are discriminators and counterexample surfaces. They are not a complete
isomorphism invariant.

## Novelty vocabulary

v0 distinguishes:

- `SEMANTIC_NOVELTY`: a new typed operator meaning relative to a frozen corpus;
- `ALGEBRAIC_NOVELTY`: inequivalence under an equivalence universe separately
  accepted as adequate for the algebraic claim;
- `ALGORITHMIC_NOVELTY`: a new executable evaluation strategy;
- `MATERIAL_NOVELTY`: a lowering or cost frontier on a named machine;
- `SCIENTIFIC_NOVELTY`: an externally defensible contribution after prior-art
  search and appropriate evidence.

The first executable may establish only bounded semantic novelty and exact
inequivalence under its frozen corpus and 48-action group. Because v0 omits
`GL(4,2)`, sign gauges, and isotopy, its `relative_algebraic_novelty` field
remains false. Algorithmic, material, scientific, historical, priority, and
global novelty remain false or open.

## NoveltyReceipt v0

The receipt must bind:

- Sounio source hash and semantics-freeze hash;
- grammar, corpus, equivalence-group, and search-space hashes;
- generator seed and exhaustive candidate count;
- selected candidate identity and canonical form;
- invariant fingerprint;
- nearest known corpus operators;
- exhaustive comparison counts;
- inequivalence witness and falsifiers;
- producer language and language role;
- formal, effect, denotational, and material parity status;
- exact novelty scope and all forbidden promotions;
- toolchain, hardware, command, and result when executed materially.

`NoveltyReceipt` is a scoped scientific object. It must never serialize a bare
`novel=true` without the corpus hash, equivalence-group hash, and scope.

## Canonical material voters

After `SEMANTICS_FROZEN`, the same selected operator may be lowered and measured
on these canonical Pireus targets:

- Xeon CPU targets, including AVX2 and AVX-512 capabilities actually observed;
- NVIDIA DGX GPU targets, using only observed CUDA/PTX/SASS resources;
- Apple Silicon CPU/GPU targets, using only observed AArch64/Metal resources;
- AMD Alveo U250 FPGA targets, using observed XRT, shell, memory, DSP, BRAM, and
  fabric resources.

Targets vote on material cost and realizability. They do not vote on semantic
truth and cannot alter the frozen candidate.

## Required refusals

The executable and its gate must fail closed for:

- missing corpus, group, grammar, policy, source hash, or semantics hash;
- incomplete candidate, group-action, or corpus comparison counts;
- a zero-distance candidate promoted as corpus-relative novelty;
- a single fingerprint promoted as proof of equivalence or inequivalence;
- global, historical, priority, algorithmic, material, or scientific novelty
  promoted from the v0 receipt;
- parity execution before a Sounio freeze;
- C++, Lean, Koka, Haskell, or a hardware target promoted to semantic producer;
- an external LLM review promoted to confirmation;
- Python or Rust proposed or executed as an oracle;
- a waiver not issued by the founder or lacking scope, purpose, and expiry.

The negative gate must deliberately submit a Python-oracle execution frame to
the canonical Sounio language-authority Guardian and observe refusal before any
Python payload executes.

## Falsifiers

This Garden is demoted or replaced if any of these occurs:

- two transformations assigned the same group identity act differently;
- canonicalization is not invariant across all 48 actions;
- an admitted winner is exactly equivalent to a frozen corpus member;
- replaying the same seed and frozen inputs selects a different candidate;
- a hash does not change when its owned semantic surface is changed;
- a parity implementation is able to define or amend the expected result;
- the gate can execute a forbidden oracle before refusal.

## Exit from Garden

`SOUNIO_EXECUTABLE` opens only when Sounio can enumerate the complete space,
select one candidate, emit its scoped receipt, and pass positive and negative
tests without imported expected numerical results.

`SEMANTICS_FROZEN` opens only after the first Sounio result exists and the
source, semantics, grammar, corpus, group, selected candidate, and receipt are
all hash-bound.

`PARITY_OPEN` opens only after the canonical Guardian accepts that freeze.

`CLAIM_READY` remains closed in v0.
