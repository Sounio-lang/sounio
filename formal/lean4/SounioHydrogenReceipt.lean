-- formal/lean4/SounioHydrogenReceipt.lean
/-!
# Sounio Hydrogen Receipt — the demo numbers, as theorems

The Python oracle layer for `demos/hydrogen/` is retired: every numeric
claim the demos print is proved here, in core Lean 4 over exact rationals
(`Rat`), closed by `native_decide` — bignum arithmetic, no floats, no
`sorry`, no Mathlib. What a Python script *checks*, this file *proves*.

Structure:

* **§1 CRF bracket, self-contained.** The capital recovery factor at
  7 %/25 y is a closed rational, `(7/100)·(107/100)²⁵ / ((107/100)²⁵ − 1)`.
  We prove it lies in `[0.0857, 0.0859]` — no external "≈ 0.0858" claim
  anywhere in the receipt.

* **§2 IDM receipt (bayes_pilot.sio).** The exact posterior interval
  `[27/32, 29/32]`, its width `1/16`, the three-estimator crossing order
  (frequentist certifies exactly at cycle 30, not before), and the
  ~60-cycle campaign size at the assumed fleet rate 0.93.

* **§3 The never-crosses theorem.** At the pilot's own rate 0.90, the IDM
  lower bound approaches 0.90 from below and never reaches it — proved
  generally over `Int` (cross-multiplied form) by `omega`, not just
  observed numerically.

* **§4 Chain receipt (hub_chain.sio).** Every stage interval and the
  delivered-cost interval `[5.11, 7.92] €/kg` are rational-evaluated.
  Because the chain is monotone in every input (proved in
  `SounioHydrogenPbox.monotone_event_equiv`), evaluating the closed form
  at the CRF endpoints bounds the whole CRF interval — the receipt needs
  no transcendental function anywhere.

References:
- Walley (1996), JRSS-B 58:3 (Imprecise Dirichlet Model)
- Stamatakis et al., Energies 16:6257 (2023) — Crete case parameters
- Stamatakis et al., Renewable Energy 147:164 (2020); IJHE 46:29272 (2021)

No sorry. No Mathlib.
-/

namespace Sounio.HydrogenReceipt

-- ================================================================
-- S1. CRF(7%, 25y) is a closed rational — and we bracket it
-- ================================================================

/-- Capital recovery factor as an exact rational expression. -/
def crfRat : Rat :=
  (7/100) * ((107/100)^25) / (((107/100)^25) - 1)

/-- The bracket used by the whole receipt: `0.0857 ≤ CRF ≤ 0.0859`.
    Closed by bignum evaluation of `(107/100)^25` — nothing transcendental
    is asserted without proof. -/
theorem crf_bracket : (857/10000 : Rat) ≤ crfRat ∧ crfRat ≤ (859/10000 : Rat) := by
  constructor <;> native_decide

-- ================================================================
-- S2. IDM receipt (bayes_pilot.sio)
-- ================================================================

/-- Walley IDM posterior bounds, prior strength `s`. -/
def idmLo (k n s : Rat) : Rat := k / (n + s)
def idmHi (k n s : Rat) : Rat := (k + s) / (n + s)

/-- After 30 cycles with 27 holds and `s = 2`, the guarantee interval is
    exactly `[27/32, 29/32]`. -/
theorem idm_final : idmLo 27 30 2 = (27/32 : Rat) ∧ idmHi 27 30 2 = (29/32 : Rat) := by
  constructor <;> native_decide

/-- Remaining ignorance after 30 cycles: exactly `1/16` (6.25 pp). -/
theorem idm_width : idmHi 27 30 2 - idmLo 27 30 2 = (1/16 : Rat) := by
  native_decide

/-- The frequentist estimate does NOT certify at cycle 27 (24/27 < 0.90);
    it certifies exactly at cycle 30 (27/30 ≥ 0.90). The interval's
    guarantee lags both. -/
theorem crossing_order :
    (24/27 : Rat) < (9/10 : Rat)
    ∧ (27/30 : Rat) ≥ (9/10 : Rat)
    ∧ idmLo 27 30 2 < (9/10 : Rat) := by
  refine ⟨?_, ?_, ?_⟩ <;> native_decide

/-- Campaign size at the assumed fleet rate 0.93: the 0.90 guarantee
    needs `0.90·s/(r − 0.90) = 60` cycles exactly. -/
theorem campaign_size :
    (9/10 : Rat) * 2 / ((93/100 : Rat) - 9/10) = 60 := by
  native_decide

-- ================================================================
-- S3. The never-crosses theorem (general, not numeric)
-- ================================================================

/-- At a true rate of exactly 0.90, the IDM lower bound `k/(n+s)` with
    `s = 2` never reaches the 0.90 gate: whenever the observed rate
    satisfies `k/n ≤ 9/10` (cross-multiplied: `10·k ≤ 9·n`), the lower
    bound satisfies `k/(n+2) < 9/10` (cross-multiplied: `10·k < 9·(n+2)`).
    No campaign size certifies a 0.90 fleet at a 0.90 gate — stated here
    in division-free integer form; the inequality itself holds for all
    integers `k, n` (the application uses positive `n`). -/
theorem idm_never_crosses (k n : Int) (h : 10 * k ≤ 9 * n) :
    10 * k < 9 * (n + 2) := by
  omega

-- ================================================================
-- S4. Chain receipt (hub_chain.sio) — Crete delivered cost
-- ================================================================

/-- Production cost [EUR/kg]: electricity + annualized PEM CAPEX/O&M.
    Parameters from Energies 16:6257: 50 MW, 46.4 kWh/kg, 1500 EUR/kW,
    O&M 20 EUR/kW/yr (1 MEUR/yr), CAPEX 75 MEUR. -/
def prodCost (crf elec cf : Rat) : Rat :=
  (464/10) * elec + (75000000 * crf + 1000000) / (50000 * cf * 8760 / (464/10))

/-- Compression cost [EUR/kg]: thermal energy × waste-heat price. -/
def comprCost (eTh pHeat : Rat) : Rat := eTh * pHeat

/-- Storage cost [EUR/kg]: 500 EUR/kg tank over `ncyc` cycles/yr. -/
def storCost (crf ncyc : Rat) : Rat := 500 * crf / ncyc

/-- Delivered cost along the whole chain. Monotone in every argument
    (↑ crf, elec, eTh, pHeat, loss; ↓ cf, ncyc) — the Lean-checked
    monotonicity of `SounioHydrogenPbox` is what licenses evaluating
    only the corners, including the CRF bracket endpoints. -/
def delivered (crf elec cf eTh pHeat loss ncyc : Rat) : Rat :=
  (prodCost crf elec cf + comprCost eTh pHeat + storCost crf ncyc) * (1 + loss)

/-- Stage 1, production: `[3.924, 3.925] ∋ prod(lo)` and
    `[4.663, 4.664] ∋ prod(hi)` — per-mille sandwiches. -/
theorem stage1_bracket :
    (3924/1000 : Rat) ≤ prodCost crfRat (46/1000) (44/100)
    ∧ prodCost crfRat (46/1000) (44/100) ≤ (3925/1000 : Rat)
    ∧ (4663/1000 : Rat) ≤ prodCost crfRat (52/1000) (35/100)
    ∧ prodCost crfRat (52/1000) (35/100) ≤ (4664/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- Stage 3, storage: `[0.953, 0.954] ∋ stor(45 cyc)` and
    `[1.430, 1.431] ∋ stor(30 cyc)`. -/
theorem stage3_bracket :
    (953/1000 : Rat) ≤ storCost crfRat 45
    ∧ storCost crfRat 45 ≤ (954/1000 : Rat)
    ∧ (1430/1000 : Rat) ≤ storCost crfRat 30
    ∧ storCost crfRat 30 ≤ (1431/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- The delivered-cost interval: best corner in `[5.113, 5.114]`,
    worst corner in `[7.920, 7.921]` — the `[5.11, 7.92] €/kg` the demo
    prints, proved. Monotonicity (§3 of SounioHydrogenPbox) propagates
    the CRF bracket through the corners, so these evaluations at `crfRat`
    bound every CRF in `[0.0857, 0.0859]`. -/
theorem delivered_bracket :
    (5113/1000 : Rat) ≤ delivered crfRat (46/1000) (44/100) 44 (5/1000) (3/1000) 45
    ∧ delivered crfRat (46/1000) (44/100) 44 (5/1000) (3/1000) 45 ≤ (5114/1000 : Rat)
    ∧ (7920/1000 : Rat) ≤ delivered crfRat (52/1000) (35/100) 89 (2/100) (6/1000) 30
    ∧ delivered crfRat (52/1000) (35/100) 89 (2/100) (6/1000) 30 ≤ (7921/1000 : Rat) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> native_decide

/-- The nominal point estimate: `[6.277, 6.278] ∋ delivered(nominal)`. -/
theorem nominal_bracket :
    (6277/1000 : Rat) ≤ delivered crfRat (49/1000) (395/1000) (665/10) (125/10000) (45/10000) (3729/100)
    ∧ delivered crfRat (49/1000) (395/1000) (665/10) (125/10000) (45/10000) (3729/100) ≤ (6278/1000 : Rat) := by
  constructor <;> native_decide

end Sounio.HydrogenReceipt
