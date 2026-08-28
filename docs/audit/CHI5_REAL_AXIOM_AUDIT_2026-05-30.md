<!-- docs:meta
topic_id: repo.docs.audit.chi5-real-axiom-audit-2026-05-30
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.chi5-real-axiom-audit-2026-05-30
-->

# χ(ℝ²) ≥ 5 over a Mathlib-free ℝ — axiom audit

**Date:** 2026-05-30
**Artefact:** `formal/lean4/SounioDeGreyChi5Real.lean`
**Toolchain:** Lean 4 v4.30.0, Mathlib-free, self-hosted proof stack
**Build:** `lake build SounioDeGreyChi5Real` → exit 0

## 1. Statement

```lean
namespace DeGrey529.RealPlane

theorem chi_R2_ge_5_unconditional :
    ¬ Nonempty (UnitDistanceChromatic.PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      (DeGrey529.TransferWf.rootedTransfer SounioSqrt.RealCauchyField.rootedFieldReal).unit 4) :=
  SounioSqrt.RealCauchyField.chi_R2_ge_5 DeGrey529.Closed.not_VColourable
```

The theorem asserts, with **no remaining hypotheses**, that the unit-distance graph on
the plane `ℝ × ℝ` admits **no** proper 4-colouring — i.e. the chromatic number of the
plane satisfies χ(ℝ²) ≥ 5.

Here `ℝ := SounioSqrt.RealCauchyField.Real = Quotient realSetoid` is the **Mathlib-free**
real field, constructed from scratch as the quotient of Cauchy `Rat` sequences modulo the
`RealEq` (null-difference) equivalence, then assembled into a `SounioSqrt.RootedField`
(ordered field + the four prime square roots √3, √5, √7, √11; no total `sqrt`, no
completeness axiom). The unit-distance relation is
`(DeGrey529.TransferWf.rootedTransfer rootedFieldReal).unit`.

The proof term is simply `chi_R2_ge_5 not_VColourable`: it feeds the discharged SAT leg
(`DeGrey529.Closed.not_VColourable`) into the conditional transfer result
(`SounioSqrt.RealCauchyField.chi_R2_ge_5`), discharging its sole hypothesis.

## 2. Full `#print axioms` output (verbatim)

```
'DeGrey529.RealPlane.chi_R2_ge_5_unconditional' depends on axioms: [propext,
 Classical.choice,
 Quot.sound,
 g529_check._native.native_decide.ax_1_1,
 g529_not_colourable._native.native_decide.ax_1_1,
 g529_not_colourable._native.native_decide.ax_1_2,
 g529_not_colourable._native.native_decide.ax_1_3,
 g529_not_colourable._native.native_decide.ax_1_4,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_10,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_11,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_12,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_13,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_14,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_15,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_16,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_17,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_18,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_3,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_4,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_5,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_6,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_7,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_8,
 MultiquadRing.perm_range_xor._native.native_decide.ax_1_9,
 Closed.edges_eq._native.native_decide.ax_1_1,
 Concrete.geom_all_edges_unitFP._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.cauchy_bounded._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.cauchy_bounded._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.decay._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.decay_modulus._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.err_step._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.error_contract._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.error_contract._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.exists_pow_le._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.hquarter._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.hquarter._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.hquarter._native.native_decide.ax_1_3,
 SounioSqrt.RealCauchyField.hquarter._native.native_decide.ax_1_4,
 SounioSqrt.RealCauchyField.hquarter._native.native_decide.ax_1_5,
 SounioSqrt.RealCauchyField.inv_le_one._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.le_add_one._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.mono_eq_gen._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.mono_step._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.mono_step._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.mono_step_bound._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.mono_step_bound._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.newton_cauchy._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.newton_cauchy._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.newton_cauchy._native.native_decide.ax_1_3,
 SounioSqrt.RealCauchyField.newton_cauchy._native.native_decide.ax_1_4,
 SounioSqrt.RealCauchyField.newton_inv._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.newton_inv._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.newton_inv._native.native_decide.ax_1_3,
 SounioSqrt.RealCauchyField.one_le_primeRat._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.per_step._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.per_step._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.per_step._native.native_decide.ax_1_3,
 SounioSqrt.RealCauchyField.q4_bound._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.q4_nonneg._native.native_decide.ax_1_3,
 SounioSqrt.RealCauchyField.q4_nonneg._native.native_decide.ax_1_4,
 SounioSqrt.RealCauchyField.q4_step_le._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.quarter_four._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.quarter_four._native.native_decide.ax_1_2,
 SounioSqrt.RealCauchyField.rat_K_pos._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.rat_add_halves._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.rat_add_thirds._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.rat_half_pos._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.rat_third_pos._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.rootR_nonneg._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.zero_ne_oneR._native.native_decide.ax_1_1,
 SounioSqrt.RealCauchyField.zero_ne_oneR._native.native_decide.ax_1_2,
 DeGrey529.TransferWf.allX_ne._native.native_decide.ax_1_1✝,
 DeGrey529.TransferWf.allY_ne._native.native_decide.ax_1_1✝,
 SounioSqrt.RealCauchyField.rat_prod_bound._native.native_decide.ax_1_1✝]
```

## 3. Trust-base classification

The axiom list partitions into exactly two classes (plus a confirmed absence).

### 3.1 Kernel-standard axioms

- `propext` — propositional extensionality.
- `Classical.choice` — the axiom of choice. It enters here via the multiplicative-inverse
  construction's use of classical decidability (`Classical.propDecidable` / `Classical.em`
  on `RealEq a zeroR` in `invReal`), not from any analytic completeness assumption.
- `Quot.sound` — quotient soundness (used pervasively: `ℝ` itself is a `Quotient`, as is
  the QF multiquadratic kernel).

These three are the standard Lean 4 foundational axioms. A development that uses only these
is "kernel-only" in the usual sense.

### 3.2 `native_decide` reduction axioms

Every remaining entry matches the `*._native.native_decide.ax_*` pattern — the
`Lean.ofReduceBool` family generated when a closed decidable proposition is discharged by
`native_decide`. These mean the result trusts the **Lean compiler's compiled evaluation**
of the corresponding closed checks, namely:

- **SAT-certificate re-checking** (`g529_check.*`, `g529_not_colourable.*`,
  `Closed.edges_eq.*`): souc_sat's CDCL LRAT certificate for G₅₂₉ re-verified by Lean
  core's LRAT checker, plus the edge-list equality.
- **Geometry / unit-distance leg** (`Concrete.geom_all_edges_unitFP.*`,
  `DeGrey529.TransferWf.allX_ne.*`, `allY_ne.*`): every G₅₂₉ edge is a unit pair over the
  exact symbolic field-plane, with nonzero embedding denominators.
- **Multiquadratic kernel** (`MultiquadRing.perm_range_xor.*`): the XOR-reindex/permutation
  facts underpinning the QF → F ring homomorphism.
- **Analytic ℝ toolkit** (the `SounioSqrt.RealCauchyField.*` family — `cauchy_bounded`,
  `decay`, `newton_cauchy`, `rat_*`, `q4_*`, `rootR_nonneg`, `zero_ne_oneR`, etc.): the
  closed `Rat`-scalar arithmetic facts used by the Cauchy/Newton convergence proofs.

**Plainly stated:** this is a *compiler-trust* assumption (the compiled `decide` kernel
correctly evaluated these closed boolean checks). It is **not** kernel-only proof checking.
This is the sanctioned footprint for this artefact: `native_decide` reduction axioms are
acceptable here; they are the standard cost of reflective certificate checking.

### 3.3 Confirmed absence: no `sorryAx`

The axiom list contains **no `sorryAx`** — there is zero `sorry` in the proof or in any of
its transitive dependencies. (The only textual "sorry" emitted by the build is inside this
artefact's own `#eval` summary string, "no sorry".)

## 4. Honest scope note

The **mathematical** theorem χ(ℝ²) ≥ 5 is due to Aubrey de Grey (2018), is peer-reviewed,
and has been independently SAT-verified by multiple groups. It is **not** a new
mathematical result, and no novelty of the mathematics is claimed here.

The **artefact** contribution is the *fully Mathlib-free, self-hosted formalisation* of that
bound, resting on a **from-scratch constructive ℝ** (the Cauchy-quotient `RootedField`,
with its own ordered-field axioms, multiplicative inverse, ε-eventual order, and Newton
prime roots). This closes the QF ↪ ℝ gap that an earlier review flagged as `[OVERREACH]`:
the bound is now stated and proved over the genuine real plane ℝ × ℝ, not merely over the
symbolic field-plane QF × QF. The trust base is the three standard Lean axioms plus the
`native_decide` compiler-evaluation axioms enumerated above — and nothing else.

**Prior art (do not overclaim):** a Lean 4 formalisation of χ(ℝ²) ≥ 5 already exists
(`vasnesterov/HadwigerNelson`), built **on Mathlib + LeanSAT + an external SAT solver**.
A "first formalisation of χ(ℝ²) ≥ 5" claim would therefore be false. The defensible framing
is *"a dependency-minimal, Mathlib-free, self-hosted formalisation over a from-scratch
constructive ℝ"*. See `docs/research/chi5-mathlib-free-novelty-2026-05-30.md` for the
literature review and the pre-claim checklist.
