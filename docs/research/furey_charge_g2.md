<!-- docs:meta
topic_id: repo.docs.research.furey-charge-g2
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.furey-charge-g2
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The Furey charge operator + a state-level test: the G₂ automorphism is not a family symmetry

**One line.** Built the Furey `Cℓ(6)` one-generation ladder (Witt relations exact over ℤ[i]) and its
charge operator `Q`, then showed the `G₂` automorphism `φ` (which permutes Gresnigt's three octonions,
`sedenion_gresnigt_octonions.md`) **does not commute with `Q`** — so `φ` does **not** act as a
charge-preserving (family) symmetry on the fermion states. This settles, at the **state level**, that
`φ` is not a bridge from the zero-divisor monomial-168 to fermion generations. **Erratum E1 stands.**

## The construction

Furey's `ℂ⊗𝕆 → Cℓ(6)` (one Standard-Model generation): left-multiplications `L_a` on the octonion
(8-dim), ladder operators `α_i = ½(−L_{a_i} + i L_{b_i})` for pairs `(1,2),(3,4),(5,6)`. Scaling by 2
gives Gaussian-integer matrices `M_i = 2α_i = −L_{a_i} + i L_{b_i}`. Certified exactly:

1. **Witt relations** `{M_i, M_i†} = M_i M_i† + M_i† M_i = 4·I` — the `M_i` are a genuine `Cℓ(6)` ladder.
2. **Charge** `Q = ⅓ Σ_i α_i† α_i`; scaled, `D = 12·Q = Σ_i M_i† M_i` is a Gaussian-integer matrix.
3. One-generation **charge multiplicities** (Fock over 3 modes): `3Q ∈ {0, 1×3, 2×3, 3}`, i.e. electric
   charges `{0, ±⅓, ±⅔, ±1}` (with conjugates) — the SM generation content.

## The decisive test (state level)

A family symmetry must **preserve charge** — it maps generations at *equal* charge (`e→μ→τ`, all charge
`−1`). The `G₂` automorphism `φ` acts on the octonion as the permutation `P_φ: e_j ↦ e_{g(j)}`,
`g=(1 2 3)(5 6 7)`. Because `(P D)[r][c] = D[g⁻¹(r)][c]` and `(D P)[r][c] = D[r][g(c)]`, a single index
comparison certifies:

> **`[P_φ, D] ≠ 0`** — `φ` does **not** commute with the charge operator.

`φ` conjugates the Furey charge to a *different* operator (it rotates the distinguished direction `e₇`
into a ladder direction). Therefore, although `φ` permutes the three octonion subalgebras, it does **not**
act as a charge-preserving symmetry on the fermion states, so it is **not** the family `S₃`.

## Honest boundary (scope)

- This rules out **this specific `φ`** (a `G₂`, color-side element). It does **not** prove that *no*
  monomial-168 element realizes the family symmetry — that would require exhibiting the charge-preserving
  generation-permuting map and checking it, or ruling out all of them.
- The genuine family `S₃` (Brown factor, `Aut(𝕊)=G₂×S₃`, non-monomial/rotational) needs the explicit
  generators from Gresnigt's §5, not yet in hand — **OPEN**.
- A real positive bridge still requires building the three-generation ideals `T_i` and showing Brown's
  `S₃` acts on those states as a monomial-168 element. Not done.

Combined with `φ ∈ G₂` (`sedenion_gresnigt_octonions.md`), this is a second, state-level confirmation
that the octonion-permutation `φ` is not the family symmetry. **Erratum E1** (monomial-168 ⊂ G₂; family =
disjoint Brown S₃) is untouched.

## Certification (3 legs)
- **souc**: `tests/run-pass/furey_charge_g2.sio` → `FUREYCHARGE OK` (bin/souc AND stage2 agree).
- **Python oracle**: `scripts/research/furey_charge_g2_oracle.py`; gate `scripts/ci/furey_charge_g2_gate.sh`.
- **Lean `native_decide`**: `formal/lean4/SounioFureyChargeG2.lean` — `witt_relations`,
  `phi_does_not_preserve_charge`, `charge_multiplicities`.

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/furey_charge_g2.sio
python3 scripts/research/furey_charge_g2_oracle.py
bash scripts/ci/furey_charge_g2_gate.sh
(cd formal/lean4 && lake build SounioFureyChargeG2)
```
