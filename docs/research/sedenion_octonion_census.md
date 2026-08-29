<!-- docs:meta
topic_id: repo.docs.research.sedenion-octonion-census
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-octonion-census
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The basis-aligned octonion census in the sedenion (a verified fingerprint)

**One line.** Of the **15** basis-aligned 3-dim `F₂` index-subspaces of the sedenion, **8** are
zero-divisor-free (genuine octonions / division algebras) and **7** are quasi-octonions carrying zero
divisors (matching Cawagas 2004); and **exactly one** is *Clifford-pure* (all internal ambient left-mult
pairs anticommute) — the base octonion `{e₁..e₇}`. Through the base quaternion `⟨e₁,e₂,e₃⟩` the three
copies give ambient-`L` non-anticommuting counts **{0, 6, 12}**. This is an executed algebraic
fingerprint; it is **consistent with** (illustrates), and does **not** prove, Erratum E1.

## Three distinct partitions (do not conflate)

A basis-aligned octonion candidate is a 3-dim `F₂` subspace of the sedenion index space `F₂⁴` (7 nonzero
vectors, XOR-closed); there are 15. Three *different* properties partition them:

| Property | Definition | Count |
|---|---|---|
| **zero-divisor-free** | the 8-dim subalgebra contains no annihilating pair (division algebra ⟹ genuine octonion) | **8** |
| quasi-octonion | contains a zero divisor | **7** (Cawagas, Discuss. Math. 2004) |
| **Clifford-pure** | all `C(7,2)=21` internal ambient `{L_i,L_j}=0` in the 16-dim `Cℓ(8)` (`sedenion_clifford8.md`) | **exactly 1** = `{e₁..e₇}` |

These are genuinely different: `8 ZD-free ⊋ 1 Clifford-pure`. A subspace can be a genuine octonion (e.g.
`{1,2,3,8,9,10,11} = ⟨e₁,e₂,e₃⟩ ⊕ ⟨e₁,e₂,e₃⟩·e₈`, the quaternion doubled by `e₈`) yet fail Clifford
purity because its ambient `L`-operators do not all anticommute. Through the base quaternion `{1,2,3}`,
the three copies (third generator `c ∈ {4,8,12}`) give ambient-`L` non-anti counts `{0, 6, 12}` — only
`c=4` (`{1..7}`) is pure; the non-purity of the other two sits entirely on the doubling seam.

## Interpretation — honest boundary

This fingerprint **illustrates** Erratum E1 (`docs/papers/sedenion-fano-geometry.md`): the ambient
embedding already singles out one basis-aligned octonion as Clifford-pure, so the three octonion copies
of Gresnigt's family-`S₃` construction cannot all be Clifford-pure-and-basis-aligned. But it is **not a
proof** that the family `S₃` is non-monomial:

- The non-monomiality of the *family* `S₃` rests on **Brown's theorem** (`Aut(𝕊)=G₂×S₃`), which E1
  cites — not on this census.
- `PSL(2,7)` — the monomial-168 itself — **contains `S₃` subgroups**, so a basis-index census cannot by
  itself separate the family `S₃` (Brown's disjoint direct factor) from an `S₃` sitting inside the
  monomial-168. The census is consistent with E1; it does not discriminate the two.

Reported as an executed algebraic fingerprint, not a group-theoretic conclusion.

## Firewall note (a discrepancy caught and resolved)

Two natural early tests gave misleading answers, and were discarded: (a) the *internal* subalgebra
product always anticommutes (imaginary units anticommute internally), so it cannot distinguish the
copies; (b) a naive associator alternativity test `A(i,i,j)=A(i,j,j)=0` passes for **all** 15 subspaces
(basis units are alternative in the doubled-index sense; sedenion non-associativity lives on distinct
triples). The load-bearing discriminators are the **ambient** left-mult anticommutation `{L_i,L_j}` (for
Clifford purity) and the **zero-divisor-containment** test (for the octonion/quasi-octonion split) — the
latter added after an adversarial review flagged that "15 octonions" was an unverified label conflicting
with Cawagas's 7 quasi-octonions.

## Certification (3 legs)
- **souc**: `tests/run-pass/sedenion_octonion_census.sio` → `OCTCENSUS OK` (bin/souc AND stage2 agree).
- **Python oracle**: `scripts/research/sedenion_octonion_census_oracle.py`; gate
  `scripts/ci/sedenion_octonion_census_gate.sh`.
- **Lean `native_decide`**: `formal/lean4/SounioSedenionOctonionCensus.lean` — `nsub_15`, `zdfree_8`,
  `pure_1`, `base_octonion_pure`, `quaternion_triple_0_6_12` (independent enumeration; the `NSUB=15`,
  `ZDFREE=8`, `PURE=1` claims are 3-leg).

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_octonion_census.sio
python3 scripts/research/sedenion_octonion_census_oracle.py
bash scripts/ci/sedenion_octonion_census_gate.sh
(cd formal/lean4 && lake build SounioSedenionOctonionCensus)
```
