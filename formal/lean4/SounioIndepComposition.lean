-- SounioIndepComposition.lean
-- Machine-checked proofs for the composition-independence obligation of
-- PR #1758 ("Independência na composição: quadratura passa a exigir prova
-- de d-separação").
--
-- Lake-buildable module: cd formal/lean4 && lake build SounioIndepComposition
--
-- Mathlib-free, following the discipline of SounioMeasConf.lean: core Lean 4
-- only, integer-scaled quantities, no `Real.sqrt`, no `nlinarith`.
--
-- HOW SQRT IS AVOIDED. Every claim below is stated on VARIANCES rather than
-- on standard uncertainties. Since x ↦ √x is monotone on the non-negatives,
--     u_a ≤ u_b   ↔   u_a² ≤ u_b²
-- so a comparison of variances is exactly a comparison of uncertainties, and
-- the square root never has to be constructed. The JCGM combination laws are
-- polynomial in this form, which is what makes them provable by `omega` in
-- core Lean: with v₁, v₂ and the covariance c as atoms, every statement in
-- §1–§3 is LINEAR.
--
-- WHAT IS PROVED
--   §1  quadrature IS the independence law (JCGM 100:2008 eq. 10 is eq. 13 at c = 0)
--   §2  with positive covariance quadrature is strictly TIGHTER than the truth
--       — unsound in the direction that matters (SEMANTICS.md Invariant 2)
--   §3  the additive bound is never wrong, only wide; tight exactly at ρ = +1
--   §4  the N-step accumulation law: composing N fully-correlated contributions
--       in quadrature understates the uncertainty by exactly √N
--   §5  d-separation: the collider row, which is what separates d-separation
--       from a reachability check (Berkson 1946)
--
-- §4 is the bridge to measurement. benchmarks/chemistry/RESULTS.md §5 reports
-- a band that scales as √(T·dt) and an underestimation that grows as √(T/dt).
-- N = T/dt is the number of steps, and the per-step parameter term carries the
-- SAME rate parameter at every step — correlation +1, not 0. Theorem
-- `quadrature_understates_correlated_sum` says the ratio of the true variance
-- to the quadrature variance over N such steps is exactly N, i.e. a factor √N
-- on the uncertainty. That is the measured law, derived rather than fitted.

namespace Sounio.IndepComposition

-- ─────────────────────────────────────────────────────────────────────────────
-- §0  Domain
-- ─────────────────────────────────────────────────────────────────────────────

-- Variances and covariances are integer-scaled (fixed point): a variance is
-- non-negative, a covariance may have either sign. `Int` is used directly and
-- NOT behind an abbreviation -- `omega` does not reduce through an `abbrev`,
-- so `abbrev Int := Int` silently costs every proof in this file.

/-- JCGM 100:2008 eq. (13), the GENERAL law of combination, in variance form:
    u_c² = u₁² + u₂² + 2·cov.  `cov` is `ρ·u₁·u₂`. -/
def varGeneral (v₁ v₂ cov : Int) : Int := v₁ + v₂ + 2 * cov

/-- JCGM 100:2008 eq. (10), the law for INDEPENDENT inputs: u_c² = u₁² + u₂².
    This is what `graded_compose` applied unconditionally before #1758. -/
def varQuadrature (v₁ v₂ : Int) : Int := v₁ + v₂

/-- The conservative additive bound, in variance form: (u₁ + u₂)².
    Here `p` is the product u₁·u₂, kept as an atom so no square root is needed. -/
def varAdditive (v₁ v₂ p : Int) : Int := v₁ + v₂ + 2 * p

-- ─────────────────────────────────────────────────────────────────────────────
-- §1  Quadrature is exactly the independence law
-- ─────────────────────────────────────────────────────────────────────────────

/-- Quadrature agrees with the general law precisely when the covariance
    vanishes. Quadrature is not an approximation that is usually fine — it is
    the ρ = 0 case, and nothing else. -/
theorem quadrature_iff_zero_covariance (v₁ v₂ cov : Int) :
    varQuadrature v₁ v₂ = varGeneral v₁ v₂ cov ↔ cov = 0 := by
  unfold varQuadrature varGeneral
  constructor <;> intro h <;> omega

/-- Restated in the direction the compiler needs: given independence, the tight
    bound is sound. This is the ONLY hypothesis under which it is. -/
theorem quadrature_sound_of_independent (v₁ v₂ cov : Int) (h : cov = 0) :
    varQuadrature v₁ v₂ = varGeneral v₁ v₂ cov := by
  unfold varQuadrature varGeneral
  omega

-- ─────────────────────────────────────────────────────────────────────────────
-- §2  With positive covariance, quadrature lies
-- ─────────────────────────────────────────────────────────────────────────────

/-- Positively correlated inputs: quadrature reports a STRICTLY SMALLER
    variance than the truth. A bound tighter than the truth is unsound — the
    case `SEMANTICS.md` Invariant 2 calls a lie.

    #1758's motivating remark is that this is the common case, not the exotic
    one: physiological series are positively correlated through time of day. -/
theorem quadrature_understates_of_positive_covariance
    (v₁ v₂ cov : Int) (h : 0 < cov) :
    varQuadrature v₁ v₂ < varGeneral v₁ v₂ cov := by
  unfold varQuadrature varGeneral
  omega

/-- The converse direction, which is why the obligation cannot be waived by
    inspection: quadrature is sound (an upper bound) exactly when the
    covariance is non-positive. Assuming independence without proof buys the
    tight bound on an unchecked premise. -/
theorem quadrature_sound_iff_nonpositive_covariance (v₁ v₂ cov : Int) :
    varGeneral v₁ v₂ cov ≤ varQuadrature v₁ v₂ ↔ cov ≤ 0 := by
  unfold varQuadrature varGeneral
  constructor <;> intro h <;> omega

-- ─────────────────────────────────────────────────────────────────────────────
-- §3  The additive bound is never wrong, only wide
-- ─────────────────────────────────────────────────────────────────────────────

/-- With `p = u₁·u₂`, the Cauchy–Schwarz bound `cov ≤ p` (i.e. ρ ≤ 1) makes the
    additive bound sound for EVERY admissible correlation. This is why #1758
    inverts the default: width is the safe direction. -/
theorem additive_sound (v₁ v₂ cov p : Int) (hcs : cov ≤ p) :
    varGeneral v₁ v₂ cov ≤ varAdditive v₁ v₂ p := by
  unfold varGeneral varAdditive
  omega

/-- The additive bound is TIGHT at ρ = +1, so it is not merely safe but the
    least sound upper bound available without an independence proof. -/
theorem additive_tight_at_unit_correlation (v₁ v₂ p : Int) :
    varGeneral v₁ v₂ p = varAdditive v₁ v₂ p := by
  unfold varGeneral varAdditive
  omega

/-- Quadrature is below the additive bound whenever u₁·u₂ is non-negative, so
    the two defaults are genuinely ordered: swapping the default from
    quadrature to additive can only widen a reported band, never narrow it. -/
theorem quadrature_below_additive (v₁ v₂ p : Int) (hp : 0 ≤ p) :
    varQuadrature v₁ v₂ ≤ varAdditive v₁ v₂ p := by
  unfold varQuadrature varAdditive
  omega

-- ─────────────────────────────────────────────────────────────────────────────
-- §4  The N-step accumulation law — the bridge to measurement
-- ─────────────────────────────────────────────────────────────────────────────

/-- Variance of N contributions of equal uncertainty `u`, combined in
    quadrature as though independent: N·u². -/
def varQuadN (n : Nat) (u : Int) : Int := (n : Int) * (u * u)

/-- Variance of the SAME N contributions when they are fully correlated
    (ρ = +1), which is the case when every step carries the same parameter:
    the uncertainties add, so the variance is (N·u)². -/
def varCorrN (n : Nat) (u : Int) : Int := ((n : Int) * u) * ((n : Int) * u)

/-- **The underestimation law.** For N fully-correlated contributions, the true
    variance is exactly N times the quadrature variance — so the true
    UNCERTAINTY is √N times the quadrature uncertainty.

    With N = T/dt this is the √(T/dt) factor measured in
    `benchmarks/chemistry/RESULTS.md` §5.3, and the √dt step-size dependence of
    §5.1 is the same statement read along dt. A per-step quadrature source is
    not a modelling choice that happens to be conservative; it understates by a
    factor that grows without bound as the step shrinks. -/
theorem quadrature_understates_correlated_sum (n : Nat) (u : Int) :
    varCorrN n u = (n : Int) * varQuadN n u := by
  unfold varCorrN varQuadN
  simp [Int.mul_assoc, Int.mul_left_comm, Int.mul_comm]

/-- The degenerate reading that makes the law easy to miss: at N = 1 quadrature
    and the correlated sum agree exactly. A single composition is no evidence
    that repeated composition is sound. -/
theorem accumulation_agrees_at_one_step (u : Int) :
    varCorrN 1 u = varQuadN 1 u := by
  unfold varCorrN varQuadN
  simp

-- ─────────────────────────────────────────────────────────────────────────────
-- §5  d-separation: the collider row
-- ─────────────────────────────────────────────────────────────────────────────

/-- The three ways two edges can meet at a middle node on an undirected path,
    in Pearl's classification. -/
inductive Junction where
  /-- `a → m → c` (or its mirror): a chain. -/
  | chain
  /-- `a ← m → c`: a fork, i.e. a common cause. -/
  | fork
  /-- `a → m ← c`: a collider, i.e. a common effect. -/
  | collider
deriving DecidableEq, Repr

/-- Whether a path through this junction is ACTIVE (unblocked), as a function
    of whether the middle node is in the conditioning set.

    The collider row is inverted with respect to the other two, and that
    inversion is the whole content of d-separation: conditioning normally
    BLOCKS a path, but at a collider it OPENS one (Berkson 1946). A checker
    that assumed conditioning is monotonically helpful — a reachability check
    with a blocklist — would get exactly this row wrong. -/
def active : Junction → Bool → Bool
  | .chain,    conditioned => !conditioned
  | .fork,     conditioned => !conditioned
  | .collider, conditioned =>  conditioned

/-- Chain and fork: conditioning on the middle node blocks the path. -/
theorem chain_blocked_by_conditioning : active .chain true = false := by rfl
theorem fork_blocked_by_conditioning  : active .fork  true = false := by rfl

/-- Collider, marginally: the path is already blocked, WITHOUT conditioning. -/
theorem collider_blocked_marginally : active .collider false = false := by rfl

/-- **Collider, conditioned: the path OPENS.** This is the discriminating case
    of #1758's table against Pearl, and the one a reachability check fails. -/
theorem collider_opened_by_conditioning : active .collider true = true := by rfl

/-- The inversion, stated as the asymmetry itself: at every junction the
    collider is active exactly when the other two are not. -/
theorem collider_inverts_the_others (b : Bool) :
    active .collider b = !(active .chain b) := by
  cases b <;> rfl

/-- Conditioning is therefore NOT monotone in the blocking direction: there is
    a junction where adding to the conditioning set turns a blocked path
    active. Any implementation whose search only ever removes edges when the
    conditioning set grows is unsound for that reason. -/
theorem conditioning_not_monotone :
    ∃ j : Junction, active j false = false ∧ active j true = true :=
  ⟨.collider, rfl, rfl⟩

end Sounio.IndepComposition
