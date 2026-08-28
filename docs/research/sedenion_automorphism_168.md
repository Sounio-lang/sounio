<!-- docs:meta
topic_id: repo.docs.research.sedenion-automorphism-168
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-automorphism-168
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The 168 IS a group: the sedenion monomial-automorphism subgroup ≅ PSL(2,7), fixing e₈ (executed)

> **Correction (2026-07-07, Erratum E1 of `docs/papers/sedenion-fano-geometry.md`).** This document
> originally called the order-168 group "`Aut(𝕊)`". That is imprecise: 168 is the **signed-permutation
> (monomial) automorphism subgroup** that fixes e₈ — a *finite subgroup* of the full automorphism group,
> which by Brown's theorem is `Aut(𝕊) ≅ Aut(𝕆) × S₃ = G₂ × S₃` (a 14-dim continuous `G₂` times the
> triality `S₃`). The monomial-168 sits inside the `G₂` factor; the fermion-generation structure lives in
> the *disjoint* `S₃` (triality) factor, so it cannot be realised as a signed permutation of basis units.
> Read every "`Aut(𝕊)`" below as "the monomial-automorphism subgroup". The executed facts (order 168,
> `≅ PSL(2,7)`, fixes e₈, Fano collineations) are unchanged and correct.

**One line.** The `168 = |PSL(2,7)|` that pervades the sedenion geometry (the ZD census, the non-Fano
count, the 7 fibers, the 42 quartets, the `1848 = 11·168` associator side) **is a genuine group**: the
sedenion **signed-permutation (monomial) automorphism subgroup** (fixing e₈), of order **168**, sitting
inside `GL(4,2) ≅ A₈` (order 20160) at index 120 — a finite subgroup of `Aut(𝕊) ≅ G₂ × S₃`, not the full
group (see the Correction above). And it **fixes e₈** — the octonion→sedenion doubling seam is the
*unique fixed point* of this group. Consequently **`1848 = 11·168` is NOT a group order**: the 168 is a
group, but the 11 is combinatorial (the e₈-grade factor), not a group index.

## Definition

A linear map `M ∈ GL(d,2)` acts on the imaginary indices `1..2^d−1` as `F₂^d` vectors (so it preserves
`⊕`). `M` is a **signed automorphism** iff there are signs `ε: index → ±1` with `φ(e_i) = ε(i) e_{Mi}`
an algebra automorphism — equivalently iff the sign ratio `σ(Mi,Mj)·σ(i,j)` is a coboundary
`δε(i,j) = ε(i)ε(j)ε(i⊕j)`. This is a decidable `F₂` linear-consistency check.

## Result

| Tower | signed automorphisms | ambient |
|---|---|---|
| octonions `{1..7}` | **168** = `\|GL(3,2)\|` | all of `GL(3,2)` |
| sedenions `{1..15}` | **168** | index 120 in `\|GL(4,2)\| = 20160 = \|A₈\|` |

- **The sedenion group fixes e₈**: the orbit of index 8 is the singleton `{8}`. The orbit partition of
  `{1..15}` is `{1..7} ∪ {8} ∪ {9..15}` — the octonionic units, the lone doubling generator e₈, and the
  doubled units. Every automorphism restricts to an octonion (Fano) automorphism and acts on the doubled
  copy accordingly; e₈ is untouched.
- **So the e₈ throughline is group-theoretic**: e₈ bounds the zero-divisor set (`sedenion_e8_boundary.md`)
  and carries the extra `168` on the associator side (`sedenion_associator_1848.md`) *because* it is the
  unique fixed point of `Aut(𝕊)`. The group is why 168 recurs — it acts on the 84 ZD vertices, the 168
  pairs, the 7 fibers, the 42 quartets, and the 1848 associator triples.
- **`1848 = 11·168` is not a group order.** `Aut(𝕊)` has order 168; the 11 is the combinatorial
  grade-decomposition factor (`11 = 10 + 1`, the e₈ grade carrying the extra copy). This settles, in the
  honest negative, the open group-interpretation question of `#668`.

## Honest scope + a caught compiler defect

The order-168 claim is standard for octonions (`Aut ≅ GL(3,2) = PSL(2,7)`); the sedenion count and the
e₈-fixed-point are computed here. **Cross-verification caught a real compiler defect**: the committed
`bin/souc` **miscompiles** the `d=4` `GL(4,2)` sweep (returns 17882 / 432, varying with the code) while
the fresh **stage2** souc, an independent Python oracle (rigorous highest-bit-pivot Gaussian), and Lean
`native_decide` all agree on **168**. A separate lesson: an order-*dependent* F₂ reduction can give a
plausible-but-wrong count — the certified computation uses rigorous highest-bit pivoting.

## Certification (3 legs; no `bin/souc` gate — it miscompiles this)

- **Executed in Sounio (stage2):** `tests/run-pass/sedenion_automorphism_168.sio` → `AUT OK`
  (`OCT_AUTOS 168 / SED_AUTOS 168 / SED_FIX_E8 168`), run by the Full Test Suite (stage2 souc). ~0.35 s.
- **Python oracle:** `scripts/research/sedenion_automorphism_168_oracle.py` → 168/168/168, orbit of e₈ = `{8}`.
- **Lean `native_decide`:** `formal/lean4/SounioSedenionAutomorphism.lean` → `oct_168`, `sed_168`,
  `sed_fix_e8_168`. Non-`default_target` (three GL(4,2) sweeps, ~1 min): `lake build SounioSedenionAutomorphism`.

## Reproduce

```bash
SOUNIO_TEST_SOUC_BIN=/path/to/souc-stage2  # bin/souc MISCOMPILES this; use a fresh stage2
python3 scripts/research/sedenion_automorphism_168_oracle.py
(cd formal/lean4 && lake build SounioSedenionAutomorphism)
```
