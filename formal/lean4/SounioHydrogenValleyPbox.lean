-- formal/lean4/SounioHydrogenValleyPbox.lean
/-!
# Sounio Hydrogen VALLEY-CHAIN p-box — corner-exactness of the composed coupling, machine-checked

Machine-checks, in core Lean 4 over exact rationals (no Mathlib, no
`sorry`), the composition theorem behind
`demos/hydrogen/valley_chain_epistemic.sio` (branch
`demos/hydrogen-valley-chain`, PR #1587): the claim that "monotone
chains keep the composed corner p-box exact".

The demo couples three epistemic stages into one effective capacity
factor on the TRIERES delivered-cost chain:

```
CF_eff = f_s · f_c · CF
f_s = 1 − (L30/100)·(τ/30)      subsurface availability (L30 = 30-yr H2
                                 loss in PERCENT, τ = residence years)
f_c = R                          compressor reliability as availability
                                 derate (identity — trivially monotone)
```

* **§1 The coupling maps** (`fsOf`, `cfEff`) — the rational spine.

* **§2 Generic interval corner-exactness.** For a three-factor product
  of nonnegative factors, each varying in an interval, the box extrema
  are attained at the lo/hi corners (`prod3_corner_bounds`); and `f_s`
  is antitone in L30 for τ ≥ 0 (`fsOf_antitone`). No independence
  assumption anywhere: this is order theory, not probability.

* **§3 The composition theorem** (`valley_corner_exact`). If the
  dispensed-cost response `D` is antitone in `CF_eff` (a *premise*,
  stated explicitly — see the boundary note below), then for EVERY
  point of the composed input box,
  `D(best corner) ≤ D(point) ≤ D(worst corner)` with
  best = (L30 lo, R hi, CF hi) and worst = (L30 hi, R lo, CF lo).
  The composed corner p-box is therefore EXACT — attained at corners of
  the input p-boxes, not a grid-scan approximation. The probability
  reading ("the p-box on P(D < 6) transfers through endpoint
  evaluation") is the already-mechanized
  `SounioHydrogenPbox.monotone_event_equiv` / `pbox_transfer`.

* **§4 The receipt** (`native_decide` over `Rat`, the demo's constants):
  the `f_s` corners reproduce the demo's printed values
  (0.999123062 / 0.999847309 at τ = 1), the composed `CF_eff` corners
  are computed exactly, the worst corner is strictly positive, and the
  corner ordering is sane.

## Premises (stated explicitly, per the mission)

1. `0 ≤ τ`, interval orderings `lo ≤ hi`, nonnegativity of the R and CF
   lower bounds — all trivially true in the demo.
2. **SIDE CONDITION (quantified):** `0 ≤ f_s` at the *worst* loss corner,
   i.e. `L30_hi·τ ≤ 3000`. The product corner bounds need the whole
   `f_s` interval nonnegative; §4 proves the premise holds up to
   τ = 1140 years and first fails at τ = 1141 — the demo's τ = 1 yr
   sits four orders of magnitude inside the validity domain, so the
   demo's label survives formalization with an explicit margin.
3. **PREMISE (not mechanized):** `D` antitone in `CF_eff`. Analytically
   `∂D/∂CF_eff = −a·e_spec/(8760·CF_eff²) < 0` for the economic chain
   (only the production term depends on CF). Mechanizing it requires
   ordered-field reciprocal lemmas beyond core `Rat`; consistent with
   `SounioHydrogenPbox.monotone_event_equiv`, which likewise takes
   monotonicity as hypotheses. The demo's 1/CF_eff domain additionally
   needs the worst-corner `CF_eff > 0` — §4 discharges that for the
   demo's intervals (`0.007198… > 0`; if the reliability p-box's lower
   endpoint ever touched 0 the cost model itself, not just the bound,
   would break).

No sorry. No Mathlib.
-/

namespace Sounio.HydrogenValleyPbox

-- ================================================================
-- S1. The coupling maps (rational spine of the demo)
-- ================================================================

/-- Subsurface availability factor: `f_s = 1 − (L30/100)·(τ/30)`, with
`L30` the 30-yr H2 loss in percent and `τ` the storage residence time
in years. ILLUSTRATIVE coupling in the demo (labeled swap slot). -/
def fsOf (tau l30 : Rat) : Rat := 1 - l30 * tau / 3000

/-- Effective capacity factor: `CF_eff = f_s · f_c · CF`. -/
def cfEff (fs fc cf : Rat) : Rat := fs * fc * cf

-- ================================================================
-- S2. Generic interval corner-exactness for monotone maps
-- ================================================================

/-- Scaling by a nonnegative constant preserves order. -/
theorem scale_nonneg_mono (c lo x : Rat) (hc : 0 ≤ c) (h : lo ≤ x) :
    c * lo ≤ c * x := by
  have h1 : 0 ≤ c * (x - lo) :=
    Rat.mul_nonneg hc ((Rat.le_iff_sub_nonneg lo x).1 h)
  rw [Rat.le_iff_sub_nonneg]
  grind

/-- `f_s` is antitone in L30 whenever τ ≥ 0: more loss, less delivered. -/
theorem fsOf_antitone (tau l l' : Rat) (htau : 0 ≤ tau) (h : l ≤ l') :
    fsOf tau l' ≤ fsOf tau l := by
  have hs : 0 ≤ (l' - l) * tau := Rat.mul_nonneg ((Rat.le_iff_sub_nonneg l l').1 h) htau
  rw [Rat.le_iff_sub_nonneg]
  unfold fsOf
  grind

/-- Corners of `f_s` on the L30 interval. -/
theorem fsOf_corners (tau l lLo lHi : Rat) (htau : 0 ≤ tau)
    (hlb : lLo ≤ l) (hlt : l ≤ lHi) :
    fsOf tau lHi ≤ fsOf tau l ∧ fsOf tau l ≤ fsOf tau lLo :=
  ⟨fsOf_antitone tau l lHi htau hlt, fsOf_antitone tau lLo l htau hlb⟩

/-- Three-factor product of nonnegative interval factors: the box
extrema are attained at the (lo, lo, lo) and (hi, hi, hi) corners.
The nonnegativity of every *lower* bound is the load-bearing premise —
if a factor's interval straddles 0, the single-corner assignment breaks
(this is exactly side condition 2 of the header). -/
theorem prod3_corner_bounds (aLo aHi bLo bHi cLo cHi a b c : Rat)
    (haLo : 0 ≤ aLo) (hab : aLo ≤ a) (hat : a ≤ aHi)
    (hbLo : 0 ≤ bLo) (hbb : bLo ≤ b) (hbt : b ≤ bHi)
    (hcLo : 0 ≤ cLo) (hcb : cLo ≤ c) (hct : c ≤ cHi) :
    aLo * bLo * cLo ≤ a * b * c ∧ a * b * c ≤ aHi * bHi * cHi := by
  have ha : 0 ≤ a := Rat.le_trans haLo hab
  have hb : 0 ≤ b := Rat.le_trans hbLo hbb
  have hc : 0 ≤ c := Rat.le_trans hcLo hcb
  have haHi : 0 ≤ aHi := Rat.le_trans ha hat
  have hcHi : 0 ≤ cHi := Rat.le_trans hc hct
  -- two-factor bounds
  have hlo2 : aLo * bLo ≤ a * b := by
    have s1 : aLo * bLo ≤ aLo * b := scale_nonneg_mono aLo bLo b haLo hbb
    have s2 : aLo * b ≤ a * b := by
      rw [Rat.mul_comm aLo b, Rat.mul_comm a b]
      exact scale_nonneg_mono b aLo a hb hab
    exact Rat.le_trans s1 s2
  have hup2 : a * b ≤ aHi * bHi := by
    have s1 : a * b ≤ aHi * b := by
      rw [Rat.mul_comm a b, Rat.mul_comm aHi b]
      exact scale_nonneg_mono b a aHi hb hat
    have s2 : aHi * b ≤ aHi * bHi := scale_nonneg_mono aHi b bHi haHi hbt
    exact Rat.le_trans s1 s2
  -- nonnegativity of the partial products
  have hnn_ab : 0 ≤ a * b := Rat.mul_nonneg ha hb
  have hnn_lo2 : 0 ≤ aLo * bLo := Rat.mul_nonneg haLo hbLo
  -- extend by the third factor
  constructor
  · have s1 : aLo * bLo * cLo ≤ aLo * bLo * c := scale_nonneg_mono (aLo * bLo) cLo c hnn_lo2 hcb
    have s2 : aLo * bLo * c ≤ a * b * c := by
      rw [Rat.mul_comm (aLo * bLo) c, Rat.mul_comm (a * b) c]
      exact scale_nonneg_mono c (aLo * bLo) (a * b) hc hlo2
    exact Rat.le_trans s1 s2
  · have s1 : a * b * c ≤ a * b * cHi := scale_nonneg_mono (a * b) c cHi hnn_ab hct
    have s2 : a * b * cHi ≤ aHi * bHi * cHi := by
      rw [Rat.mul_comm (a * b) cHi, Rat.mul_comm (aHi * bHi) cHi]
      exact scale_nonneg_mono cHi (a * b) (aHi * bHi) hcHi hup2
    exact Rat.le_trans s1 s2

-- ================================================================
-- S3. The composition theorem: the composed corner p-box is EXACT
-- ================================================================

/-- Valley-chain corner-exactness. Under the explicit premises

* `τ ≥ 0`, interval orderings, nonnegative R/CF lower bounds
  (the orderings are stated for premise completeness; they are
  implied by the membership bounds below, hence `_`-named),
* the side condition `0 ≤ f_s` at the worst loss corner
  (`L30_hi·τ ≤ 3000`, quantified in §4),
* `D` antitone in `CF_eff` (premise 3 of the header),

every point `(L, R, C)` of the composed input box satisfies

`D(best) ≤ D(L, R, C) ≤ D(worst)`

with best = `(L30 lo, R hi, CF hi)` (maximal `CF_eff`) and
worst = `(L30 hi, R lo, CF lo)` (minimal `CF_eff`): the antitone
response flips the `CF_eff` order. The composed corner p-box is
therefore exact — its extrema are attained at input-p-box corners. -/
theorem valley_corner_exact (D : Rat → Rat)
    (tau lLo lHi rLo rHi cfLo cfHi L R C : Rat)
    (htau : 0 ≤ tau) (_hll : lLo ≤ lHi)
    (hrLo : 0 ≤ rLo) (_hrr : rLo ≤ rHi)
    (hcfLo : 0 ≤ cfLo) (_hcfc : cfLo ≤ cfHi)
    (hfsnn : 0 ≤ fsOf tau lHi)
    (hD : ∀ x y : Rat, x ≤ y → D y ≤ D x)
    (hlb : lLo ≤ L) (hlt : L ≤ lHi)
    (hrb : rLo ≤ R) (hrt : R ≤ rHi)
    (hcb : cfLo ≤ C) (hct : C ≤ cfHi) :
    D (cfEff (fsOf tau lLo) rHi cfHi) ≤ D (cfEff (fsOf tau L) R C) ∧
    D (cfEff (fsOf tau L) R C) ≤ D (cfEff (fsOf tau lHi) rLo cfLo) := by
  obtain ⟨hfs1, hfs2⟩ := fsOf_corners tau L lLo lHi htau hlb hlt
  have hb := prod3_corner_bounds (fsOf tau lHi) (fsOf tau lLo) rLo rHi cfLo cfHi
      (fsOf tau L) R C hfsnn hfs1 hfs2 hrLo hrb hrt hcfLo hcb hct
  unfold cfEff
  exact ⟨hD _ _ hb.2, hD _ _ hb.1⟩

-- ================================================================
-- S4. The receipt: the demo's constants, computed exactly
-- ================================================================

/-- L30 p-box lower endpoint (demo print: 0.458073). -/
def L30LO : Rat := 458073 / 1000000
/-- L30 p-box upper endpoint (demo print: 2.630814). -/
def L30HI : Rat := 2630814 / 1000000
/-- Compressor-reliability p-box lower endpoint (demo print: 0.0131). -/
def RLO : Rat := 131 / 10000
/-- Compressor-reliability p-box upper endpoint (demo print: 0.9989). -/
def RHI : Rat := 9989 / 10000
/-- Capacity-factor p-box lower endpoint (demo print: 0.55). -/
def CFLO : Rat := 55 / 100
/-- Capacity-factor p-box upper endpoint (demo print: 0.80). -/
def CFHI : Rat := 80 / 100
/-- Storage residence time used in the demo: τ = 1 yr. -/
def TAU : Rat := 1

/-- The `f_s` worst corner reproduces the demo's printed 0.999123062. -/
theorem fs_worst_corner : fsOf TAU L30HI = 999123062 / 1000000000 := by
  native_decide

/-- The `f_s` best corner reproduces the demo's printed 0.999847309. -/
theorem fs_best_corner : fsOf TAU L30LO = 999847309 / 1000000000 := by
  native_decide

/-- Composed `CF_eff` at the best corner (demo print: 0.79899798156808). -/
theorem cfEff_best_corner :
    cfEff (fsOf TAU L30LO) RHI CFHI = 79899798156808 / 100000000000000 := by
  native_decide

/-- Composed `CF_eff` at the worst corner (demo print: 0.00719868166171). -/
theorem cfEff_worst_corner :
    cfEff (fsOf TAU L30HI) RLO CFLO = 719868166171 / 100000000000000 := by
  native_decide

/-- The worst corner is strictly positive: the demo's 1/CF_eff domain is
valid on the whole composed box (premise 3's side condition). -/
theorem cfEff_worst_pos : 0 < cfEff (fsOf TAU L30HI) RLO CFLO := by
  native_decide

/-- Corner sanity: worst `CF_eff` ≤ best `CF_eff` at the demo's endpoints. -/
theorem cfEff_corners_ordered :
    cfEff (fsOf TAU L30HI) RLO CFLO ≤ cfEff (fsOf TAU L30LO) RHI CFHI := by
  native_decide

/-- Side condition 2 holds with a wide margin: `f_s ≥ 0` at the worst
loss corner for every τ ≤ 1140 years. -/
theorem tau_margin_holds : 0 ≤ fsOf 1140 L30HI := by
  native_decide

/-- …and first fails at τ = 1141 years: the margin is quantified, not
assumed. The demo's τ = 1 yr sits four orders of magnitude inside. -/
theorem tau_margin_fails : fsOf 1141 L30HI < 0 := by
  native_decide

end Sounio.HydrogenValleyPbox
