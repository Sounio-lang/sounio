/-!
# Sounio.Scheduler.ClarkFoldDeterminism

Pure Lean 4 stdlib.  No `Mathlib` imports.  No `sorry`.  No `Classical`
(axiom footprint matches `CriticalPathLowerBound` / `PortMatchingSound`).

Companion proof obligation for the Clark 1961 pairwise moment-matched
max-of-two-normals fold in `cgplan_clark_fold`
(`self-hosted/native/codegen_plan.sio`, lines 909–944).

The fold computes a moment-matched normal approximation to `max(X1, X2)`
for independent `X_i ~ N(m_i, v_i)` in fixed-point integer arithmetic:

      a   = √(v1 + v2)
      α   = (m1 − m2) / a
      E[max]   = m1·Φ(α) + m2·Φ(−α) + a·φ(α)
      E[max²]  = (m1²+v1)·Φ(α) + (m2²+v2)·Φ(−α) + (m1+m2)·a·φ(α)
      Var      = E[max²] − E[max]²

**Determinism (equimodal swap invariance).**  When the two operands are
equimodal — i.e. `m1 = m2 = m` — the fold output is invariant under
swapping the variance arguments:

      clarkFoldRaw cfg m v1 m v2 = clarkFoldRaw cfg m v2 m v1

This is the structural guarantee `cgplan_finalize_makespan_clark` needs
when iteratively folding a tied top-tier of finish-times in any order:
the result depends only on the multiset of variances, not on the order
in which they are presented to the fold.

The proof rests on two operational invariants of the Sounio CDF / PDF
table helpers:

  * `cgplan_phi_x1000`  satisfies `Φ(−α) = 1000 − Φ(α)`  — by
    construction (the `negate` branch).  Forces `Φ(0) = 500`.
  * `cgplan_phi_pdf_x1000` satisfies `φ(−α) = φ(α)` — by construction
    (the absolute-value branch).  Forces `φ(0)` well-defined.

Under those two structural symmetries, the equimodal `(m₁=m₂)` case
collapses `Φ(α)` and `Φ(−α)` to a common value of `500`, and the only
`v`-dependent fragment of the formula is symmetric in `(v1, v2)` by
`Int.add_comm`.

Integer overflow & 64-bit truncation are not modelled here (Sounio uses
`i64`; this file works over `Int`).  The bridge — that the i64
arithmetic in `cgplan_clark_fold` faithfully implements the `Int`
semantics under the operating ranges produced by Phase D — is a separate
operational claim (paper-side, alongside the Hoare-classical split).
-/

namespace Sounio.Scheduler.ClarkFold

/-- Abstract configuration of the Sounio Clark fold: integer-valued CDF
    and PDF helpers (output × 1000), plus an integer square root.  The
    two `_symm` / `_even` fields capture the operational structure of
    `cgplan_phi_x1000` / `cgplan_phi_pdf_x1000`. -/
structure ClarkConfig where
  phi : Int → Int
  phiPdf : Int → Int
  isqrt : Int → Int
  /-- CDF reflection: `Φ(−α) = 1000 − Φ(α)`.  Forces `Φ(0) = 500`. -/
  phi_symm : ∀ x : Int, phi (-x) = 1000 - phi x
  /-- PDF parity: `φ(−α) = φ(α)`. -/
  pdf_even : ∀ x : Int, phiPdf (-x) = phiPdf x

/-- Degenerate output: when `sum_v ≤ 0` or `a_x1000 ≤ 0`, the fold falls
    back to `(max m1 m2, 0)` — no normal-approximation work. -/
def degenerate (m1 m2 : Int) : Int × Int :=
  (if m1 ≥ m2 then m1 else m2, 0)

/-- The variance-output computation of the Clark fold.  Factored out so
    that the `e_max2_x1000` symmetrisation step in `equimodal_swap_v` is
    a one-line `Int.add_comm` rewrite rather than a deep `congr` chase. -/
def clarkFoldVariance
    (m1 v1 m2 v2 : Int)
    (a_x1000 phi_a pdf_a e_max_x1000 : Int) : Int :=
  let phi_na := 1000 - phi_a
  let e_max2_x1000 :=
        (m1 * m1 + v1) * phi_a
      + (m2 * m2 + v2) * phi_na
      + ((m1 + m2) * a_x1000 * pdf_a) / 1000
  let e_max2_x1m   := e_max2_x1000 * 1000
  let e_max_sq_x1m := e_max_x1000 * e_max_x1000
  let v_out_x1m :=
        if e_max2_x1m - e_max_sq_x1m < 0 then 0
        else e_max2_x1m - e_max_sq_x1m
  v_out_x1m / 1000000

/-- The mean-output computation of the Clark fold. -/
def clarkFoldMean
    (m1 m2 a_x1000 phi_a pdf_a : Int) : Int :=
  let phi_na := 1000 - phi_a
  let e_max_x1000 := m1 * phi_a + m2 * phi_na + (a_x1000 * pdf_a) / 1000
  e_max_x1000 / 1000

/-- The main (non-degenerate) Clark-fold computation, given a positive
    `a_x1000`.  Wraps `clarkFoldMean` and `clarkFoldVariance` into the
    pair the fold returns. -/
def mainCompute
    (cfg : ClarkConfig) (m1 v1 m2 v2 a_x1000 : Int) : Int × Int :=
  let alpha_x1000 := ((m1 - m2) * 1000000) / a_x1000
  let phi_a := cfg.phi alpha_x1000
  let pdf_a := cfg.phiPdf alpha_x1000
  let e_max_x1000 :=
        m1 * phi_a + m2 * (1000 - phi_a) + (a_x1000 * pdf_a) / 1000
  (clarkFoldMean m1 m2 a_x1000 phi_a pdf_a,
   clarkFoldVariance m1 v1 m2 v2 a_x1000 phi_a pdf_a e_max_x1000)

/-- The single-cycle Clark moment-matched fold of two independent normal
    finish-time distributions.  Step-for-step mirror of `cgplan_clark_fold`
    working over `Int` (no i64 overflow modelled). -/
def clarkFoldRaw (cfg : ClarkConfig) (m1 v1 m2 v2 : Int) : Int × Int :=
  let sum_v := v1 + v2
  if sum_v ≤ 0 then degenerate m1 m2
  else
    let a_x1000 := cfg.isqrt (sum_v * 1000000)
    if a_x1000 ≤ 0 then degenerate m1 m2
    else mainCompute cfg m1 v1 m2 v2 a_x1000

/-- The CDF anchor: `Φ(0) = 500`.  Follows from `phi_symm` at `x = 0`
    combined with `−0 = 0` in `Int`. -/
private theorem phi_at_zero (cfg : ClarkConfig) : cfg.phi 0 = 500 := by
  have h : cfg.phi (-(0 : Int)) = 1000 - cfg.phi 0 := cfg.phi_symm 0
  rw [Int.neg_zero] at h
  omega

/-- The degenerate branch collapses to `(m, 0)` when both operand means
    are equal. -/
private theorem degenerate_equimodal (m : Int) : degenerate m m = (m, 0) := by
  unfold degenerate
  rw [if_pos (Int.le_refl m)]

/-- The variance computation is symmetric in `(v1, v2)` when both
    operands have the same mean and `phi_a = 500` (so `phi_na = 1000 −
    500 = 500` as well).  Direct from `Int.add_comm`. -/
private theorem clarkFoldVariance_equimodal_swap
    (m v1 v2 a_x1000 pdf_a e_max_x1000 : Int) :
    clarkFoldVariance m v1 m v2 a_x1000 500 pdf_a e_max_x1000
    = clarkFoldVariance m v2 m v1 a_x1000 500 pdf_a e_max_x1000 := by
  unfold clarkFoldVariance
  -- After zeta-reduction of `phi_na := 1000 - 500`, the only `v`-dependent
  -- substring of `e_max2_x1000` is `(m*m + v_i) * 500 + (m*m + v_j) * 500`,
  -- symmetric in `(v1, v2)` by `Int.add_comm`.  Show the whole
  -- `e_max2_x1000` inner sum is symmetric, then rewrite.
  have h_e_max2 :
      (m * m + v1) * 500 + (m * m + v2) * (1000 - 500)
        + ((m + m) * a_x1000 * pdf_a) / 1000
    = (m * m + v2) * 500 + (m * m + v1) * (1000 - 500)
        + ((m + m) * a_x1000 * pdf_a) / 1000 := by
    show (m * m + v1) * 500 + (m * m + v2) * 500
           + ((m + m) * a_x1000 * pdf_a) / 1000
       = (m * m + v2) * 500 + (m * m + v1) * 500
           + ((m + m) * a_x1000 * pdf_a) / 1000
    congr 1
    exact Int.add_comm _ _
  simp only [h_e_max2]

/-- The non-degenerate computation is symmetric under `(v1, v2)` swap
    when the operand means are equal: `alpha = (m − m) · 10⁶ / a = 0`,
    so `phi_a = phi 0 = 500` and the variance helper handles the rest. -/
private theorem mainCompute_equimodal_swap_v
    (cfg : ClarkConfig) (m v1 v2 a_x1000 : Int) :
    mainCompute cfg m v1 m v2 a_x1000
    = mainCompute cfg m v2 m v1 a_x1000 := by
  unfold mainCompute
  have hzero : (m - m) * 1000000 / a_x1000 = (0 : Int) := by
    rw [Int.sub_self, Int.zero_mul, Int.zero_ediv]
  have hphi0 := phi_at_zero cfg
  -- Reduce `alpha_x1000 := (m - m) * 10⁶ / a_x1000` to `0`, then zeta-reduce
  -- so `cfg.phi alpha_x1000` becomes `cfg.phi 0`, then specialise to `500`.
  simp only [hzero, hphi0]
  refine Prod.mk.injEq _ _ _ _ |>.mpr ⟨rfl, ?_⟩
  exact clarkFoldVariance_equimodal_swap m v1 v2 a_x1000 (cfg.phiPdf 0) _

/-- **Theorem (ClarkFoldDeterminism, equimodal).**  When both operands
    share a mean, the Clark fold is invariant under swapping their
    variance arguments.  This is the structural guarantee that
    `cgplan_finalize_makespan_clark` needs when collapsing a tied
    top-tier of finish-times into one. -/
theorem clarkFold_equimodal_swap_v
    (cfg : ClarkConfig) (m v1 v2 : Int) :
    clarkFoldRaw cfg m v1 m v2 = clarkFoldRaw cfg m v2 m v1 := by
  unfold clarkFoldRaw
  -- Align `sum_v` on both sides.
  rw [show v2 + v1 = v1 + v2 from Int.add_comm v2 v1]
  -- First `if`: `(v1 + v2) ≤ 0`.
  if hsv : v1 + v2 ≤ 0 then
    simp only [if_pos hsv, degenerate_equimodal]
  else
    simp only [if_neg hsv]
    -- Second `if`: `a_x1000 ≤ 0` (a_x1000 = isqrt((v1+v2)·10⁶)).
    if ha : cfg.isqrt ((v1 + v2) * 1000000) ≤ 0 then
      simp only [if_pos ha, degenerate_equimodal]
    else
      simp only [if_neg ha]
      exact mainCompute_equimodal_swap_v cfg m v1 v2 _

end Sounio.Scheduler.ClarkFold
