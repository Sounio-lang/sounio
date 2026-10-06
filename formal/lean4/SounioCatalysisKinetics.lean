-- formal/lean4/SounioCatalysisKinetics.lean
/-!
# Sounio.CatalysisKinetics — Enzyme-Kinetics Algebraic Invariants

Independent Lean4 formal check for the Sounio chemistry stdlib's
catalysis-mechanism suite. This file does **not** transcribe or import
any Sounio source — it re-derives the underlying mathematics of two
standard enzyme-kinetics formulas from scratch and proves that they
hold as pure algebraic identities. It exists as a cross-language oracle
alongside independent C++23 / Koka / F* / F# checks of the same claims.

## Dependency note (why this lives in `formal/lean4/`, not `formal/`)

`formal/lakefile.lean` is a Mathlib-dependent package, but Mathlib is
**not prebuilt** in this environment (`.lake/` there is ~9.6M, no
resolved `packages`, and `lake exe cache get` has no wired-up `cache`
executable). A from-source Mathlib build was judged too slow to be
worth forcing for two goals that are pure `CommRing`/`Field` algebra
with no analysis content.

`formal/lean4/` is this repository's *existing*, independently
established convention for Mathlib-free Lean verification (see e.g.
`SounioFrechetRat.lean`'s "Rat lift" stage, `SounioSqrtField.lean`,
etc.) — a plain `lake_lib` with zero `require`, built entirely against
Lean 4 core (`Init.Data.Rat`) plus the core `grind` tactic (which as of
this toolchain, `leanprover/lean4:v4.33.0`, has its own ring/field
E-matching engine and needs no Mathlib). This file follows that same
convention rather than inventing a third dependency path.

Per the "real (or rational) values" latitude in the task: both
theorems are proved over `ℚ` (`Rat`), core Lean 4's built-in exact
rational type, which is a faithful sub-ordered-field of `ℝ` — any
identity proved here transports to `ℝ` verbatim once a `ℝ` (e.g. via
Mathlib, or an in-tree Cauchy-sequence construction as sketched by
`SounioRealCauchy.lean`) is in scope. No real-analysis content (limits,
continuity, `Real.sqrt`, ...) is needed for either claim below, so nothing
is lost by working over `ℚ`.

Zero `sorry`, zero `axiom`, zero `require`.
-/

namespace Sounio.CatalysisKinetics

-- ================================================================
-- §1. Enzyme conservation for E + S ⇌ ES → E + P.
-- ================================================================

/-- Forward binding rate `kf · [E] · [S]`. -/
def fwdRate (kf E S : Rat) : Rat := kf * E * S

/-- Reverse (unbinding) rate `kr · [ES]`. -/
def revRate (kr ES : Rat) : Rat := kr * ES

/-- Catalytic turnover rate `kcat · [ES]`. -/
def catRate (kcat ES : Rat) : Rat := kcat * ES

/-- `d[E]/dt` under the mass-action mechanism `E + S ⇌ ES → E + P`. -/
def dE_dt (kf kr kcat E S ES : Rat) : Rat :=
  -(fwdRate kf E S) + revRate kr ES + catRate kcat ES

/-- `d[ES]/dt` under the same mechanism. -/
def dES_dt (kf kr kcat E S ES : Rat) : Rat :=
  fwdRate kf E S - revRate kr ES - catRate kcat ES

/-- **Enzyme conservation.** For the classic mass-action mechanism
`E + S ⇌ ES → E + P` with

  `fwd = kf·E·S`,  `rev = kr·ES`,  `cat = kcat·ES`,
  `dE/dt  = -fwd + rev + cat`,
  `dES/dt =  fwd - rev - cat`,

total free+bound enzyme is conserved: `dE/dt + dES/dt = 0`, identically,
for *all* rate constants and concentrations — no positivity, no ODE
existence/uniqueness, just that the two right-hand sides cancel term by
term. This is the algebraic backbone of "total enzyme `[E]+[ES]` is a
conserved quantity" for this mechanism. -/
theorem enzyme_conservation (kf kr kcat E S ES : Rat) :
    dE_dt kf kr kcat E S ES + dES_dt kf kr kcat E S ES = 0 := by
  unfold dE_dt dES_dt fwdRate revRate catRate
  grind

-- ================================================================
-- §2. Hill equation half-saturation identity.
-- ================================================================

/-- The Hill equation `Hill(S) = Vmax · Sⁿ / (Kmⁿ + Sⁿ)`, Hill
coefficient `n : ℕ`. -/
def Hill (Vmax Km S : Rat) (n : Nat) : Rat :=
  Vmax * S ^ n / (Km ^ n + S ^ n)

/-- **Hill half-saturation identity.** For any Hill coefficient `n`
and any `Km > 0`, evaluating the Hill equation at its own
half-saturation constant gives exactly half of `Vmax`:
`Hill(Km) = Vmax / 2`.

Substituting `S := Km` collapses `Sⁿ` to `Kmⁿ`, so the goal reduces to
`Vmax·Kmⁿ / (2·Kmⁿ) = Vmax/2`, which needs precisely one fact —
`Kmⁿ ≠ 0` — to justify cancelling the shared factor `Kmⁿ`. That fact is
supplied by `Rat.pow_pos` (positivity of a positive base raised to any
natural power), which holds for `n = 0` too (`Km⁰ = 1 ≠ 0`), so the
theorem genuinely needs no lower bound on `n` — an earlier draft carried
an unused `0 < n` hypothesis the proof never used; removed after an
xai/grok-4.5 math-review flagged the doc/theorem mismatch it caused.
Everything downstream is closed by `grind`'s built-in field-cancellation
reasoning. -/
theorem hill_half_saturation (Vmax Km : Rat) (n : Nat)
    (hKm : 0 < Km) :
    Hill Vmax Km Km n = Vmax / 2 := by
  unfold Hill
  have hp : Km ^ n ≠ 0 := Rat.ne_of_gt (Rat.pow_pos hKm)
  -- Abstract `Kmⁿ` to a single nonzero rational `p`; the remaining
  -- goal `Vmax * p / (p + p) = Vmax / 2` is pure field cancellation.
  generalize Km ^ n = p at hp ⊢
  grind

end Sounio.CatalysisKinetics
