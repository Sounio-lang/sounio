<!-- docs:meta
topic_id: repo.docs.research.madore-spindle-q3q11-2026-05-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madore-spindle-q3q11-2026-05-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madore's Moser spindle over ℚ(√3,√11) — parametric multiquadratic framework (2026-05-30/31)

Mathlib-free Lean 4. Four files, all `lake build`-green, no `sorry`/`sorryAx`.

## What is proved

### `formal/lean4/SounioMultiquadParam.lean` — parametric framework over `primes : List Nat`

- `sqrtR : Nat → Real` (generic Newton root) + `sqrtR_sq` (`(√p)² = p`, `p ≥ 1`).
- `radS`/`evalS` over `2^|S|` masks; `HasRadicals`/`IndepMultiquad`; base case `indep_base`.
- `sqrt_new_not_in_Q` — generic ℚ-irrationality core (any non-square `m` ⇒ √m ∉ ℚ).
- `Np_ne_zero` — generic degree-2 **domain core**: `a² − p·b² ≠ 0` for non-square `p`.
- `indep_3_11` (degree 4, spindle basis `{1,√3,√11,√33}`) and `indep_3_5_11` (degree 8) —
  proved by bridging to the existing `indep8`. The latter re-grounds the **χ ≥ 5 over ℚ(√3,√5,√11)**
  line as the `S = [3,5,11]` instance (`SounioDeGreyChi5Param`).

### `formal/lean4/SounioMoserSpindleQ311.lean` — the headline (Madore's χ ≥ 4 line)

The Moser spindle is realised with **exact** coordinates in ℚ(√3,√11) (integer 4-tuples
`a + b√3 + c√11 + d√33`, scaled ×12), and:

- `edges_unit` — each of the **11 unit edges** has squared distance exactly the rational `144`
  (all √3/√11/√33 components cancel), by `decide` on integer arithmetic. Axioms `{propext}`.
- `spindle_not_3_colourable` — **not 3-colourable**: the 11 edge-disequalities over `Fin 3` are
  refuted by `omega` (no `native_decide`). `spindle_4_colourable` gives the witness ⇒ **χ = 4**.
- `q311_plane_needs_4_colours` — via the generic `UnitDistanceChromatic` reduction:
  **χ(ℚ(√3,√11)²) ≥ 4**, the base case of Madore (arXiv:1509.07023). Axioms `{propext, Quot.sound}`.

### `formal/lean4/SounioMoserSpindleQ311Real.lean` — **χ(ℝ²) ≥ 4 discharged**

- `phi311 : Qf → Real` via `alpha311 = alpha4(·,·,0,0) + alpha4(·,·,0,0)·√11`.
- `E_mul311` / `phi311_mul` — ring homomorphism on the spindle field.
- `dist2Real_emb` — squared distance commutes with embedding.
- `chi_R2_ge_4` — `¬ Nonempty (PlaneColouring (Real × Real) unitReal144 3)` where
  `unitReal144` means `dist² = 144` at the native ×12 integer scale.
- **Reading:** rescaling the plane by `1/12` turns spindle edges into Euclidean unit edges
  (`dist² = 1`); the chromatic lower bound is unchanged. A formal `dist² = 1` corollary is
  optional follow-up (needs scalar-multiplication lemmas on the quotient reals).

### `formal/lean4/MadoreSpindleVitrine.lean` — single-entry showcase

Build-time manifest tying together `indep_3_11`, `spindle_not_3_colourable`,
`q311_plane_needs_4_colours`, and `chi_R2_ge_4`. `lake build MadoreSpindleVitrine`.

## Positioning vs de Grey (χ ≥ 5)

| Line | Field | Degree | Vertices | Certificate |
|------|-------|--------|----------|-------------|
| **Madore** (this work) | ℚ(√3,√11) | 4 | 7 (Moser spindle) | `omega` + `decide` |
| **de Grey** (prior) | ℚ(√3,√5,√11) | 8 | 529 (G₅₂₉) | souc_sat LRAT |

Madore is the **minimal** multiquadratic base case; de Grey is the **sharpened** degree-8 line.

## Open obligation (honest scope)

`MultiquadIndepTheorem : ∀ S, HasRadicals S → IndepMultiquad S` is stated as an explicit
obligation — *not* an axiom, *not* `sorry`. Discharged instances:

| Support | Theorem | Method |
|---------|---------|--------|
| `[]` | `indep_base` | `qR_inj` |
| `[p]` (non-square) | `indep_singleton` | `indep_1_sqrt` / `no_qR_eq_sqrtR` |
| `[3,11]` | `indep_3_11` | bridge to `indep8` |
| `[3,5,11]` | `indep_3_5_11` | bridge to `indep8` |

**§10 inductive packaging:** `MultiquadIndepTheorem_of` closes the general theorem once these
three cons-step obligations are proved for all `S'`:

1. `EvalSConsSplitObligation` — mask split `evalS (p::S') c = A + √p·B`
2. `SqrtNewNotInTower` — `√p ∉ K_{S'}` (squarefree peel + `no_rat_sqrt`; base case `sqrt_new_not_in_tower_nil`)
3. `TowerHasInverses` — recursive norm/conjugate inverse (base case `tower_has_inverses_nil`; lift of `Q35_inv`)

The unrestricted `∀S` proof remains open at these three points; no `sorry` in the wrapper.

## Review

`bin/llm-offload -t math-review -p xai` (Grok 4.1) on the Lean files: **proved statements [OK]**,
open obligation correctly identified. See `.claude/llm_offload_log.md` (2026-05-30/31 entries).
