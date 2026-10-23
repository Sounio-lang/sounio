-- formal/lean4/SounioSurfaceKinetics.lean
/-!
# Sounio.SurfaceKinetics — Surface-Microkinetics Structural Invariants

Independent Lean 4 formal check for `stdlib/chemistry/surface.sio`. This
file does **not** transcribe or import any Sounio source. It re-derives
the underlying algebra from scratch and proves the three structural
claims that module rests on, as pure identities over `ℚ`.

It sits alongside the independent C++23 oracle
(`stdlib/chemistry/oracles/surface_oracle.cpp`, Newton on the algebraic
steady state) and follows the Mathlib-free `formal/lean4/` convention
established by `SounioCatalysisKinetics.lean` — plain `lean_lib`, zero
`require`, Lean 4 core plus `grind`. See that file's header for why this
directory rather than `formal/`.

## What is proved, and why these three

`surface.sio` makes three claims that are not numerical and therefore
cannot be settled by agreeing with an oracle. They are algebraic, and so
they are proved:

1. **Site conservation is decided by the column sums, and only by them.**
   `sites_conserved()` tests that every step's stoichiometric
   coefficients sum to zero. §1 proves that this test is exactly right:
   the total site count is unchanged *for every possible rate* if and
   only if those sums vanish. Not "sufficient" — an iff. A mechanism
   that passes the test cannot leak sites at any rate, and one that
   fails it leaks at some rate.

2. **The dissociative isotherm's square root is forced.** §2 and §3
   prove that the equilibrium coverage of a molecular adsorbate is
   `Kp/(1+Kp)` while that of a dissociating diatomic is
   `√(Kp)/(1+√(Kp))` — stated without square roots, by parameterising on
   `s` and proving the equilibrium holds exactly when `s² = Kp`. This is
   the hydrogen case: `H₂ + 2* ⇌ 2H*`.

3. **The closed form IS the mechanism's limit.** §4 proves that a
   Langmuir–Hinshelwood mechanism whose adsorption steps are in
   quasi-equilibrium has rate exactly
   `k·KA·pA·KB·pB / (1 + KA·pA + KB·pB)²` — the closed form implemented
   independently in `chemistry::catalysis::rate_law_eval` kind 5 and in
   `chemistry::surface::lh_dual_site_rate`. The numerical agreement
   between those two and the integrated mechanism is checked in
   `tests/stdlib/chemistry/test_surface_stdlib.sio`; this is why they
   must agree.

All four theorems are over `ℚ` (core Lean 4's exact rationals), a
faithful sub-ordered-field of `ℝ`; every identity here transports to `ℝ`
verbatim. There is no analysis content — no limits, no continuity, no
ODE existence — so nothing is lost by working over `ℚ`.

Zero `sorry`.

Every docstring below was rewritten after an adversarial math-review
(xai/grok-4.6, 2026-09-09) found four overreaching claims and one
redundant hypothesis in the first version, and a second reviewer
(mistral-large) passed all eight theorems that existed then. The ninth,
`dissociative_coverage`, was added BECAUSE of the review -- an earlier
docstring asserted it while the file proved only its two halves. What the review changed is
recorded at each site rather than silently corrected. The proofs
themselves were sound throughout, which is precisely why the prose around
them needed checking separately.
-/

namespace Sounio.SurfaceKinetics

-- ================================================================
-- §1. Site conservation is exactly the vanishing of the column sums.
-- ================================================================

/-- Net change in the total site count contributed by one elementary
step running at rate `r`, for a mechanism with three adsorbates and the
free site as a fourth species. -/
def totalSiteChange (a b c v r : Rat) : Rat := a * r + b * r + c * r + v * r

/-- **Site conservation, and it is an iff.** A single elementary step
leaves the total site count unchanged *at every rate* precisely when its
stoichiometric coefficients — the free site's included — sum to zero.

The forward direction is what `chemistry::surface::sites_conserved`
relies on: a mechanism passing that test cannot create or destroy sites
however fast its steps run. The reverse direction is what makes the test
*complete* rather than merely sound: a step whose coefficients do not
sum to zero leaks at some rate, so nothing is being let through. A
checker with only the forward direction could pass a leaky mechanism;
this one cannot. -/
theorem sites_conserved_iff (a b c v : Rat) :
    (∀ r : Rat, totalSiteChange a b c v r = 0) ↔ a + b + c + v = 0 := by
  unfold totalSiteChange
  constructor
  · intro h
    have h1 := h 1
    grind
  · intro h r
    grind

-- ================================================================
-- §2. Langmuir isotherm, non-dissociative:  A(g) + * ⇌ A*.
-- ================================================================

/-- **Langmuir isotherm.** At adsorption/desorption equilibrium
`kf·p·v = kr·θ`, with the site balance `θ + v = 1`, the coverage
satisfies `θ·(kr + kf·p) = kf·p`.

This undivided identity holds unconditionally — no positivity, no bound
on `p`, no ODE, no nonvanishing hypothesis. **Reading it as
`θ = Kp/(1+Kp)` with `K = kf/kr` requires `kr ≠ 0`, which this theorem
does not supply**: the degenerate `kf = kr = 0` satisfies both hypotheses
for *every* `θ` with `θ + v = 1`, so the identity is vacuously true there
while the quotient form is meaningless. The divided reading is
`langmuir_isotherm_div`, which states its own hypothesis.

(Flagged by an adversarial math-review, xai/grok-4.6, 2026-09-09: the
original docstring offered the quotient form as an equivalent of the
undivided one.) -/
theorem langmuir_isotherm (kf kr p theta v : Rat)
    (heq : kf * p * v = kr * theta) (hsum : theta + v = 1) :
    theta * (kr + kf * p) = kf * p := by
  have hv : v = 1 - theta := by grind
  subst hv
  grind

/-- The same statement in the familiar divided form, which needs the
denominator to be nonzero — as it always is for physical `kr > 0`,
`kf, p ≥ 0`. -/
theorem langmuir_isotherm_div (kf kr p theta v : Rat)
    (heq : kf * p * v = kr * theta) (hsum : theta + v = 1)
    (hD : kr + kf * p ≠ 0) :
    theta = kf * p / (kr + kf * p) := by
  have h := langmuir_isotherm kf kr p theta v heq hsum
  grind

-- ================================================================
-- §3. Langmuir isotherm, dissociative:  A₂(g) + 2* ⇌ 2A*.
--     The hydrogen case.
-- ================================================================

/-- **Dissociative isotherm, square root and all, with no square roots
in sight.** Write the state as a ratio: `theta = s * v`, where `s` is
covered sites per free site. Then the dissociative equilibrium
`kf·p·v² = kr·theta²` holds **exactly when** `kf·p = kr·s²`, i.e. when
`s = √(Kp)`.

The square in `s` comes from the two sites the diatomic occupies. It is
the structural fact behind the half-order reaction order measured in
`examples/chemistry/h2_surface_reaction_order.sio`.

Stated multiplicatively, with no division anywhere, so it needs only
`v ≠ 0` — a surface with no free sites at all, where the ratio `s` is
not defined.

**Three things this does NOT say, all three flagged by an adversarial
math-review (xai/grok-4.6, 2026-09-09) against a docstring that did say
them.** `kr·s² = kf·p` is *not* `s = √(Kp)` over `ℚ`: `kr` may vanish,
`Kp` need not be a square in `ℚ` at all, and both `s` and `-s` solve it
(the physical branch is `s ≥ 0`, the covered-to-free ratio, which nothing
here imposes). Nor does this theorem alone give a coverage — that needs
the site balance too, and is `dissociative_coverage` below, which the
earlier docstring claimed without proving. And that coverage, `s/(1+s)`,
grows as `√p` only while `s ≪ 1`; at saturation it flattens toward 1 and
does not grow as `√p` at all. -/
theorem dissociative_isotherm_iff (kf kr p s v theta : Rat)
    (hv : v ≠ 0) (hratio : theta = s * v) :
    kf * p * v ^ 2 = kr * theta ^ 2 ↔ kf * p = kr * s ^ 2 := by
  subst hratio
  have hv2 : v ^ 2 ≠ 0 := by grind
  constructor
  · intro h
    have hfac : (kf * p - kr * s ^ 2) * v ^ 2 = 0 := by grind
    grind
  · intro h
    grind

/-- The site balance, in the same ratio coordinates: `theta + v = 1`
with `theta = s * v` forces `v * (1 + s) = 1`, which is the
`theta = s/(1+s)`, `v = 1/(1+s)` of the usual statement without needing
`1 + s` to be invertible in the statement itself. -/
theorem ratio_site_balance (s v theta : Rat)
    (hratio : theta = s * v) (hsum : theta + v = 1) :
    v * (1 + s) = 1 := by
  subst hratio
  grind

/-- **The coverage itself**, assembled from the two pieces above. With
`θ = s·v` and the site balance `θ + v = 1`, the coverage satisfies
`θ·(1+s) = s` — that is, `θ = s/(1+s)`, in undivided form. Together with
`dissociative_isotherm_iff` (`kr·s² = kf·p`) this is the whole content of
`θ = √(Kp)/(1+√(Kp))`, with the square root replaced by the
covered-to-free ratio it names.

Written out because an earlier docstring on `dissociative_isotherm_iff`
asserted this statement while the file proved only its two halves
separately. Assembling them is one line; claiming the assembly without
writing it is the kind of gap a review exists to find. -/
theorem dissociative_coverage (s v theta : Rat)
    (hratio : theta = s * v) (hsum : theta + v = 1) :
    theta * (1 + s) = s := by
  have hv := ratio_site_balance s v theta hratio hsum
  subst hratio
  grind

/-- The two isotherms are genuinely different functions of pressure, and
this exhibits a witness rather than asserting it: at `K = 4`, `p = 1`,
the molecular form `Kp/(1+Kp)` gives `4/5` while the dissociative form
`s/(1+s)` at `s = √(Kp) = 2` gives `2/3`. A model that assumes the wrong
one is not slightly off; it is answering a different question. -/
theorem isotherms_differ :
    (4 : Rat) / 5 ≠ (2 : Rat) / 3 := by
  grind

-- ================================================================
-- §4. The dual-site Langmuir–Hinshelwood closed form IS the
--     quasi-equilibrium limit of the elementary mechanism.
-- ================================================================

/-- **The reduction, machine-checked.** Take an elementary
Langmuir–Hinshelwood mechanism: `A` and `B` adsorb reversibly and are in
quasi-equilibrium, so `θA = KA·pA·v` and `θB = KB·pB·v`; the surface
reaction is slow, so the site balance is `θA + θB + v = 1`. Then the
surface-reaction rate `k·θA·θB` is exactly

  `k · KA·pA · KB·pB / (1 + KA·pA + KB·pB)²`

which is what `chemistry::catalysis::rate_law_eval` kind 5 and
`chemistry::surface::lh_dual_site_rate` each compute, written
independently.

**What this is, and what it is not.** It is the exact algebraic
consequence of imposing quasi-equilibrium and the site balance — nothing
is approximated, and the two independently written closed forms are
therefore the same object rather than two unrelated models. It is **not**
a limit theorem: no limit is taken, no residual is bounded, and no ODE
trajectory is shown to approach this state. Whether a real mechanism sits
near the quasi-equilibrium point is a numerical question, measured rather
than proved, by `test_lh_reduces_to_closed_form` — which reports the gap
that opens when a surface reaction is fast enough to drain the
adsorbates.

(The distinction was forced by an adversarial math-review, xai/grok-4.6,
2026-09-09, against a docstring that called this a limit.) -/
theorem lh_quasi_equilibrium (k KA pA KB pB thetaA thetaB v : Rat)
    (hA : thetaA = KA * pA * v)
    (hB : thetaB = KB * pB * v)
    (hsum : thetaA + thetaB + v = 1) :
    k * thetaA * thetaB * (1 + KA * pA + KB * pB) ^ 2
      = k * (KA * pA) * (KB * pB) := by
  subst hA
  subst hB
  -- The site balance forces v * D = 1, where D is the Langmuir denominator.
  have hv : v * (1 + KA * pA + KB * pB) = 1 := by grind
  -- Squaring it is the only step that is not immediate: the left-hand side
  -- carries v^2 * D^2, which is exactly (v * D)^2 = 1.
  have hv2 : (v * (1 + KA * pA + KB * pB)) * (v * (1 + KA * pA + KB * pB)) = 1 := by
    grind
  grind

/-- The same reduction in the divided form the two Sounio
implementations actually compute, available once the denominator is
known nonzero -- as it is for any physical `KA·pA, KB·pB ≥ 0`. This is
literally the body of `chemistry::catalysis::rate_law_eval` kind 5 and
of `chemistry::surface::lh_dual_site_rate`. -/
theorem lh_quasi_equilibrium_div (k KA pA KB pB thetaA thetaB v : Rat)
    (hA : thetaA = KA * pA * v)
    (hB : thetaB = KB * pB * v)
    (hsum : thetaA + thetaB + v = 1) :
    k * thetaA * thetaB
      = k * (KA * pA) * (KB * pB) / (1 + KA * pA + KB * pB) ^ 2 := by
  have h := lh_quasi_equilibrium k KA pA KB pB thetaA thetaB v hA hB hsum
  -- The denominator cannot vanish and need not be assumed away: the site
  -- balance already forces v * D = 1, and 0 = 1 is false in Rat. An earlier
  -- version carried it as a hypothesis; the same review that caught the
  -- overreaches above pointed out it was derivable.
  have hv : v * (1 + KA * pA + KB * pB) = 1 := by grind
  have hD : (1 + KA * pA + KB * pB) ≠ 0 := by grind
  have hD2 : (1 + KA * pA + KB * pB) ^ 2 ≠ 0 := by grind
  grind

end Sounio.SurfaceKinetics

-- Audit trail: these print the axiom dependencies of each theorem. A `sorry`
-- anywhere in a proof shows up here as `sorryAx`; the expected output names
-- only Lean's own three (propext, Classical.choice, Quot.sound), and for the
-- purely computational ones, none at all.
#print axioms Sounio.SurfaceKinetics.sites_conserved_iff
#print axioms Sounio.SurfaceKinetics.langmuir_isotherm
#print axioms Sounio.SurfaceKinetics.langmuir_isotherm_div
#print axioms Sounio.SurfaceKinetics.dissociative_isotherm_iff
#print axioms Sounio.SurfaceKinetics.ratio_site_balance
#print axioms Sounio.SurfaceKinetics.dissociative_coverage
#print axioms Sounio.SurfaceKinetics.isotherms_differ
#print axioms Sounio.SurfaceKinetics.lh_quasi_equilibrium
#print axioms Sounio.SurfaceKinetics.lh_quasi_equilibrium_div
