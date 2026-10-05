/-
  SounioFoCssSurfaceParity — algebraic FO Css surface independence (residual §5.4).

  Closes the *mathematical* half of residual 4 from
  `docs/dissertation/handoff/fo_pk_method_science_package.md`:

    Import ↔ method agreement is not only gate-backed numerical parity;
    the pure algebraic models of the four oral-Css FO surfaces are
    definitionally equal, and the default-seed first-order freezes are
    exact rationals matching R1–R4.

  Does **not** claim Madaros IR / compiler surface commutativity for
  arbitrary programs (that remains executable evidence under
  `scripts/ci/fo_pk_import_method_driver_gate.sh`).

  Surfaces modelled (mirror `stdlib/epistemic/fo.sio` + R1/R4 drivers):
    Import        fo_css = fo_infusion_rate / fo_clearance
    Site          (F·Dose/τ) / (CL0 · e^η)
    Method        Pk.css — same pure formula
    Call-result   make_pk(...).css — same pure formula after construction

  Default seeds (η = 0 ⇒ e^η = 1):
    F = 4/5, Dose = 500, τ = 12, CL0 = 5, V0 = 50,
    σ_F = 1/20, σ_Dose = 10, σ_CL0 = 3/10, σ_η = 1/10, σ_V0 = 2.

  FO freezes (exact ℚ, matching annex / Z.AI re-derivation):
    Css = 20/3
    Var(Css) = 191/240
    Var(CL)  = 17/50
    Var(rate)= 689/144
    Var(E)   = 1575 + 1250·ρ   (ρ ∈ {0, 1/2, 1} as 0, 1/2, 1)
    kel shared-η cancel: kel = 1/10, Var(kel) = 13/250000

  Mathlib-free. Zero sorry. Structural surface equalities by `rfl`;
  seed freezes by `native_decide` on Bool certificates (regression-anchor
  convention, cf. SounioCDCoreLaw). Companion executable certificate:
  `scripts/research/fo_css_surface_parity_cert.py`.
-/
namespace SounioFoCssSurfaceParity

-- ── §1. Pure algebraic surfaces (Rat model of oral Css) ───────────────────

/-- Oral infusion rate R = F·Dose/τ. Mirrors `fo_infusion_rate`. -/
def infusionRate (F Dose tau : Rat) : Rat := F * Dose / tau

/-- Clearance CL = CL0 · eEta (eEta stands for e^η; FO evaluates at means with eEta = 1). -/
def clearance (CL0 eEta : Rat) : Rat := CL0 * eEta

/-- Import surface: nested pure helpers (stdlib `fo_css`). -/
def cssImport (F Dose tau CL0 eEta : Rat) : Rat :=
  infusionRate F Dose tau / clearance CL0 eEta

/-- Site / method / call-result surface: same formula, no helper names. -/
def cssSite (F Dose tau CL0 eEta : Rat) : Rat :=
  (F * Dose / tau) / (CL0 * eEta)

/-- Method surface alias (dissertation `Pk.css`). -/
def cssMethod (F Dose tau CL0 eEta : Rat) : Rat :=
  cssSite F Dose tau CL0 eEta

/-- Call-result surface alias (`make_pk(...).css`). -/
def cssCallResult (F Dose tau CL0 eEta : Rat) : Rat :=
  cssMethod F Dose tau CL0 eEta

/-- Import ≡ site definitionally (helper peel is pure renaming). -/
theorem css_import_eq_site (F Dose tau CL0 eEta : Rat) :
    cssImport F Dose tau CL0 eEta = cssSite F Dose tau CL0 eEta := rfl

/-- Method ≡ site. -/
theorem css_method_eq_site (F Dose tau CL0 eEta : Rat) :
    cssMethod F Dose tau CL0 eEta = cssSite F Dose tau CL0 eEta := rfl

/-- Call-result ≡ method. -/
theorem css_call_result_eq_method (F Dose tau CL0 eEta : Rat) :
    cssCallResult F Dose tau CL0 eEta = cssMethod F Dose tau CL0 eEta := rfl

/-- Transitivity: all four surfaces agree. -/
theorem css_surfaces_agree (F Dose tau CL0 eEta : Rat) :
    cssImport F Dose tau CL0 eEta = cssSite F Dose tau CL0 eEta ∧
    cssMethod F Dose tau CL0 eEta = cssSite F Dose tau CL0 eEta ∧
    cssCallResult F Dose tau CL0 eEta = cssSite F Dose tau CL0 eEta := by
  exact ⟨rfl, rfl, rfl⟩

-- ── §2. Default seeds ─────────────────────────────────────────────────────

def F0 : Rat := 4 / 5
def Dose0 : Rat := 500
def tau0 : Rat := 12
def CL0 : Rat := 5
def V0 : Rat := 50
def eEta0 : Rat := 1   -- e^0

def sigF : Rat := 1 / 20
def sigDose : Rat := 10
def sigCL0 : Rat := 3 / 10
def sigEta : Rat := 1 / 10
def sigV0 : Rat := 2

-- ── §3. Point evaluation freezes ──────────────────────────────────────────

/-- Css point at default seeds = 20/3 (= 6.666…). -/
def css_point_ok : Bool :=
  cssImport F0 Dose0 tau0 CL0 eEta0 == (20 : Rat) / 3

theorem css_point_freeze : css_point_ok = true := by native_decide

/-- Site surface freezes to the same point. -/
def css_site_point_ok : Bool :=
  cssSite F0 Dose0 tau0 CL0 eEta0 == (20 : Rat) / 3

theorem css_site_point_freeze : css_site_point_ok = true := by native_decide

/-- Rate = F·Dose/τ = 100/3. -/
def rate_point_ok : Bool :=
  infusionRate F0 Dose0 tau0 == (100 : Rat) / 3

theorem rate_point_freeze : rate_point_ok = true := by native_decide

/-- CL = CL0 · 1 = 5. -/
def cl_point_ok : Bool :=
  clearance CL0 eEta0 == 5

theorem cl_point_freeze : cl_point_ok = true := by native_decide

-- ── §4. First-order variance (independent inputs, at means) ───────────────
--
-- For y = f(x), FO: Var ≈ Σ_i (∂f/∂x_i)² σ_i².
-- At means with eEta = 1:
--   ∂Css/∂F    = Dose/(τ·CL0)     = 500/60 = 25/3
--   ∂Css/∂Dose = F/(τ·CL0)        = (4/5)/60 = 1/75
--   ∂Css/∂CL0  = −F·Dose/(τ·CL0²) = −400/300 = −4/3
--   ∂Css/∂η    = −Css             = −20/3
--   (∂Css/∂τ ignored when σ_τ = 0)

def sensCssF : Rat := Dose0 / (tau0 * CL0)
def sensCssDose : Rat := F0 / (tau0 * CL0)
def sensCssCL0 : Rat := - (F0 * Dose0) / (tau0 * CL0 * CL0)
def sensCssEta : Rat := - cssImport F0 Dose0 tau0 CL0 eEta0

def foVarCss : Rat :=
  sensCssF * sensCssF * sigF * sigF +
  sensCssDose * sensCssDose * sigDose * sigDose +
  sensCssCL0 * sensCssCL0 * sigCL0 * sigCL0 +
  sensCssEta * sensCssEta * sigEta * sigEta

/-- Var(Css) = 191/240 (= 0.795833…). -/
def var_css_ok : Bool := foVarCss == (191 : Rat) / 240

theorem var_css_freeze : var_css_ok = true := by native_decide

-- Clearance CL = CL0 · eEta; at eEta = 1:
--   ∂CL/∂CL0 = 1, ∂CL/∂η = CL0 = 5
--   Var = 1²·(3/10)² + 5²·(1/10)² = 9/100 + 25/100 = 34/100 = 17/50

def foVarCL : Rat :=
  (1 : Rat) * (1 : Rat) * sigCL0 * sigCL0 +
  CL0 * CL0 * sigEta * sigEta

def var_cl_ok : Bool := foVarCL == (17 : Rat) / 50

theorem var_cl_freeze : var_cl_ok = true := by native_decide

-- Rate R = F·Dose/τ; τ fixed:
--   ∂R/∂F = Dose/τ = 125/3, ∂R/∂Dose = F/τ = 1/15
--   Var = (125/3)²·(1/20)² + (1/15)²·10² = 689/144

def sensRateF : Rat := Dose0 / tau0
def sensRateDose : Rat := F0 / tau0

def foVarRate : Rat :=
  sensRateF * sensRateF * sigF * sigF +
  sensRateDose * sensRateDose * sigDose * sigDose

def var_rate_ok : Bool := foVarRate == (689 : Rat) / 144

theorem var_rate_freeze : var_rate_ok = true := by native_decide

-- ── §5. Exposure E = CL0·V0·eEtaCl·eEtaV with Corr(η_cl, η_v) = ρ ────────
--
-- At means eEta = 1:
--   ∂E/∂CL0 = V0 = 50, ∂E/∂V0 = CL0 = 5
--   ∂E/∂η_cl = ∂E/∂η_v = 250
--   Var = 50²·(0.3)² + 5²·2² + 250²·(0.1)² + 250²·(0.1)²
--       + 2·250·250·(0.1)·(0.1)·ρ
--       = 1575 + 1250·ρ

def foVarExposure (rho : Rat) : Rat :=
  let sCL := V0
  let sV  := CL0
  let sE1 : Rat := CL0 * V0
  let sE2 : Rat := CL0 * V0
  sCL * sCL * sigCL0 * sigCL0 +
  sV * sV * sigV0 * sigV0 +
  sE1 * sE1 * sigEta * sigEta +
  sE2 * sE2 * sigEta * sigEta +
  (2 : Rat) * sE1 * sE2 * sigEta * sigEta * rho

def var_E_rho0_ok : Bool := foVarExposure 0 == 1575
def var_E_rho_half_ok : Bool := foVarExposure (1 / 2) == 2200
def var_E_rho1_ok : Bool := foVarExposure 1 == 2825

theorem var_E_rho0_freeze : var_E_rho0_ok = true := by native_decide
theorem var_E_rho_half_freeze : var_E_rho_half_ok = true := by native_decide
theorem var_E_rho1_freeze : var_E_rho1_ok = true := by native_decide

/-- Linear law certificate: Var(E,ρ) = 1575 + 1250·ρ at the three table rows. -/
def var_E_law_ok : Bool :=
  foVarExposure 0 == 1575 + 1250 * (0 : Rat) &&
  foVarExposure (1 / 2) == 1575 + 1250 * (1 / 2 : Rat) &&
  foVarExposure 1 == 1575 + 1250 * (1 : Rat)

theorem var_E_law_freeze : var_E_law_ok = true := by native_decide

-- ── §6. Shared-η kel cancellation ─────────────────────────────────────────
--
-- kel = (CL0 · e) / (V0 · e). At e = 1 this is CL0/V0 = 1/10.
-- With shared η the latent cancels in the point value; FO residual is from
-- CL0, V0 only:
--   ∂kel/∂CL0 = 1/V0 = 1/50, ∂kel/∂V0 = −CL0/V0² = −1/500
--   Var = (1/50)²·(3/10)² + (1/500)²·2² = 13/250000 = 5.2e-5

def kelShared (CL0_ V0_ eEta : Rat) : Rat := (CL0_ * eEta) / (V0_ * eEta)
def kelPeeled (CL0_ V0_ : Rat) : Rat := CL0_ / V0_

def kel_point_ok : Bool :=
  kelShared CL0 V0 eEta0 == (1 : Rat) / 10 &&
  kelPeeled CL0 V0 == (1 : Rat) / 10

theorem kel_point_freeze : kel_point_ok = true := by native_decide

/-- At unit eEta the shared form equals the peeled form (computational cancel).
    Not `rfl`: Rat reduction of `(CL0·1)/(V0·1)` is not definitionally `CL0/V0`. -/
theorem kel_shared_eq_peeled_at_unit :
    kelShared CL0 V0 1 = kelPeeled CL0 V0 := by native_decide

def sensKelCL0 : Rat := 1 / V0
def sensKelV0 : Rat := - CL0 / (V0 * V0)

def foVarKel : Rat :=
  sensKelCL0 * sensKelCL0 * sigCL0 * sigCL0 +
  sensKelV0 * sensKelV0 * sigV0 * sigV0

def var_kel_ok : Bool := foVarKel == (13 : Rat) / 250000

theorem var_kel_freeze : var_kel_ok = true := by native_decide

-- ── §7. τ-scaling of FO variance ──────────────────────────────────────────
--
-- Css ∝ 1/τ ⇒ Var(Css) ∝ 1/τ² when only τ changes among deterministic slots.
-- Relative to τ = 12:
--   τ = 8  → scale (12/8)² = 9/4 = 2.25 → Var = 191/240 · 9/4 = 1719/960 = 573/320
--   Wait: 191/240 * 9/4 = 1719/960. Simplify: 1719÷3=573, 960÷3=320 → 573/320 = 1.790625.
--   τ = 24 → scale 1/4 → Var = 191/960 = 0.1989583…

def varCssAtTau (tau : Rat) : Rat :=
  -- recompute FO var with scaled sensitivities (all sens ∝ 1/tau)
  let sF := Dose0 / (tau * CL0)
  let sD := F0 / (tau * CL0)
  let sC := - (F0 * Dose0) / (tau * CL0 * CL0)
  let sE := - (F0 * Dose0) / (tau * CL0)   -- −Css
  sF*sF*sigF*sigF + sD*sD*sigDose*sigDose + sC*sC*sigCL0*sigCL0 + sE*sE*sigEta*sigEta

def var_tau8_ok : Bool := varCssAtTau 8 == (573 : Rat) / 320
def var_tau12_ok : Bool := varCssAtTau 12 == (191 : Rat) / 240
def var_tau24_ok : Bool := varCssAtTau 24 == (191 : Rat) / 960

theorem var_tau8_freeze : var_tau8_ok = true := by native_decide
theorem var_tau12_freeze : var_tau12_ok = true := by native_decide
theorem var_tau24_freeze : var_tau24_ok = true := by native_decide

/-- Scale factors relative to τ=12: 9/4 and 1/4. -/
def tau_scale_ok : Bool :=
  varCssAtTau 8 == foVarCss * (9 : Rat) / 4 &&
  varCssAtTau 24 == foVarCss * (1 : Rat) / 4

theorem tau_scale_freeze : tau_scale_ok = true := by native_decide

-- ── §8. Bundle: residual-4 algebraic closeout ─────────────────────────────

/-- All Bool certificates green. -/
def residual4_algebraic_ok : Bool :=
  css_point_ok &&
  css_site_point_ok &&
  rate_point_ok &&
  cl_point_ok &&
  var_css_ok &&
  var_cl_ok &&
  var_rate_ok &&
  var_E_rho0_ok &&
  var_E_rho_half_ok &&
  var_E_rho1_ok &&
  var_E_law_ok &&
  kel_point_ok &&
  var_kel_ok &&
  var_tau8_ok &&
  var_tau12_ok &&
  var_tau24_ok &&
  tau_scale_ok

theorem residual4_algebraic_closeout : residual4_algebraic_ok = true := by native_decide

end SounioFoCssSurfaceParity
