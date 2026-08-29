<!-- docs:meta
topic_id: repo.docs.research.sedenion-fano-fibers
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-fano-fibers
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The 7 fibers ARE the Fano plane; Aut(𝕊) is its collineation group (executed)

**One line.** The 7 sedenion zero-divisor fibers, labeled `L = lo⊕hi ∈ {9..15}`, **are the 7 points of
the Fano plane `PG(2,2)`** (via `L ∧ 7 ∈ F₂³∖{0}`), and the sedenion automorphism group `Aut(𝕊)`
(order 168, `sedenion_automorphism_168.md`) acts on them **faithfully, transitively, permuting the 7
Fano lines** — i.e. as the full Fano collineation group `PGL(3,2) = PSL(2,7)`. This is the geometric
answer to *why 168*: **168 = the Fano collineations, the 7 fibers = the Fano points, e₈ = the fixed
point** (the doubling direction, outside the plane).

## Why the fibers are Fano points

Every sedenion automorphism fixes `e₈`, so it maps a fiber label `L = 8 ⊕ t` (with `t = L ∧ 7 ∈ 1..7`)
to `M(8 ⊕ t) = 8 ⊕ M(t)` — a linear action on the lower three bits `t ∈ F₂³∖{0}`. The seven nonzero
vectors of `F₂³` are exactly the seven points of `PG(2,2)`, and their `{a, b, a⊕b}` triples are the
seven lines. So the fiber labels carry the Fano-point structure, and the group acts by collineations.

## Result (verified)

| Fact | Value |
|---|---|
| signed automorphisms | 168 |
| distinct permutations of the 7 fibers | **168** (faithful) |
| orbit of one fiber | **7** (transitive) |
| the 7 Fano lines `{a,b,a⊕b}` | **permuted** by every automorphism |

So the action on the 7 fibers realizes all `168 = |PGL(3,2)| = |PSL(2,7)|` Fano collineations. Combined
with the earlier bricks, the entire zero-divisor geometry is the **Fano plane** decorated by the
octonion→sedenion doubling: 7 fibers = points (each a `K_{6,6}−3K_{2,2}` graph), 42 quartets = `2·K₇`
edges, `168 = 84 + 84` (dagger), all under the Fano collineation group, with `e₈` its fixed point.

## Certification

- **Lean `native_decide`:** `formal/lean4/SounioSedenionFano.lean` (imports `SounioSedenionAutomorphism`)
  → `fibers_faithful` (168), `fibers_transitive` (7), `fano_lines_preserved`. Non-`default_target`
  (~1 min): `lake build SounioSedenionFano`.
- **Python oracle:** `scripts/research/sedenion_fano_fibers_oracle.py` → `AUTOS 168 / FIBER_PERMS 168 /
  ORBIT1 7 / FANO_LINES_OK True`.
- **Sounio engine:** the underlying automorphism sweep is `tests/run-pass/sedenion_automorphism_168.sio`
  (stage2; `sedenion_automorphism_168.md`). Two independent checkers (Lean, Python) verify the Fano action.

## Reproduce

```bash
python3 scripts/research/sedenion_fano_fibers_oracle.py
(cd formal/lean4 && lake build SounioSedenionFano)
```
