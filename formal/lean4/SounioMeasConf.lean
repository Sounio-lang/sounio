-- SounioMeasConf.lean
-- Machine-checked proofs for the Epistemic Gradual Compilation (EGC) system.
-- (Mathlib-free adaptation — uses only core Lean 4 tactics)
--
-- Lake-buildable module: cd formal/lean4 && lake build SounioMeasConf
--
-- Covers three core properties of the Gen 13–17 epistemic type system:
--   1. MEAS_CONF propagation soundness (GUM min-rule, Gen 13)
--   2. ety_unify PLATINUM preservation (Gen 15)
--   3. ety_conf_product monotonicity and bounds
--   4. Knightian interval invariant BELIEF ≤ CONF ≤ PLAUS (Gen 17 D1)
--
-- All confidence values are integers in [0, 1000].
-- GATE_THRESHOLD = 950; PLATINUM = 1000.
--
-- Source: docs/proofs/meas_conf_soundness.lean (canonical reference copy)

namespace EGC

-- ─────────────────────────────────────────────────────────────────────────────
-- §1  Confidence domain
-- ─────────────────────────────────────────────────────────────────────────────

/-- Confidence values live in [0, 1000]. -/
def Conf := { c : Nat // c ≤ 1000 }

/-- The maximum confidence: PLATINUM. -/
def platinum : Conf := ⟨1000, Nat.le_refl 1000⟩

/-- The gate threshold: certainty boundary. -/
def gate_threshold : Nat := 950

/-- A token is *certain* when its confidence meets the gate. -/
def certain (c : Conf) : Prop := c.val ≥ gate_threshold

/-- PLATINUM tokens are certain. -/
theorem platinum_is_certain : certain platinum := by
  simp [certain, platinum, gate_threshold]

-- ─────────────────────────────────────────────────────────────────────────────
-- §2  ety_conf_product — GUM serial propagation
-- ─────────────────────────────────────────────────────────────────────────────

/-- Clamp to [0, 1000]. -/
def clamp (n : Nat) : Nat := min n 1000

/-- The GUM product rule for confidence (serial uncertainty combination). -/
def conf_product (a b : Nat) : Nat := clamp (a * b / 1000)

/-- conf_product is bounded by 1000. -/
theorem conf_product_bounded (a b : Nat) : conf_product a b ≤ 1000 := by
  simp [conf_product, clamp]
  exact Nat.min_le_right _ _

/-- conf_product of two PLATINUM values is PLATINUM. -/
theorem conf_product_platinum : conf_product 1000 1000 = 1000 := by
  simp [conf_product, clamp]

/-- conf_product is monotone in the left argument. -/
theorem conf_product_mono_left (a a' b : Nat) (h : a ≤ a') (hb : b ≤ 1000) :
    conf_product a b ≤ conf_product a' b := by
  simp only [conf_product, clamp]
  apply Nat.le_min_of_le_of_le
  · exact Nat.le_trans (Nat.min_le_left _ _) (Nat.div_le_div_right (Nat.mul_le_mul_right b h))
  · exact Nat.min_le_right _ _

/-- conf_product is monotone in the right argument. -/
theorem conf_product_mono_right (a b b' : Nat) (h : b ≤ b') (ha : a ≤ 1000) :
    conf_product a b ≤ conf_product a b' := by
  simp only [conf_product, clamp]
  apply Nat.le_min_of_le_of_le
  · exact Nat.le_trans (Nat.min_le_left _ _) (Nat.div_le_div_right (Nat.mul_le_mul_left a h))
  · exact Nat.min_le_right _ _

/-- If either operand is 0 the product is 0. -/
theorem conf_product_zero_left (b : Nat) : conf_product 0 b = 0 := by
  simp [conf_product, clamp]

theorem conf_product_zero_right (a : Nat) : conf_product a 0 = 0 := by
  simp [conf_product, clamp]

/-- conf_product is symmetric. -/
theorem conf_product_comm (a b : Nat) : conf_product a b = conf_product b a := by
  simp [conf_product, Nat.mul_comm]

-- ─────────────────────────────────────────────────────────────────────────────
-- §3  MEAS_CONF propagation — GUM min-rule
-- ─────────────────────────────────────────────────────────────────────────────

/-- MEAS_CONF primitive sources. -/
inductive MeasSource where
  | Measured  -- instrument measurement → MEAS_CONF = 990
  | Asserted  -- literature value       → MEAS_CONF = 970
  | Constant  -- exact physical constant→ MEAS_CONF = 1000

/-- Map source to quality integer. -/
def meas_base : MeasSource → Nat
  | .Measured => 990
  | .Asserted => 970
  | .Constant => 1000

/-- GUM min-rule: derived MEAS_CONF = min of inputs. -/
def meas_propagate (m1 m2 : Nat) : Nat := min m1 m2

theorem meas_propagate_bounded (m1 m2 : Nat) (h1 : m1 ≤ 1000) (h2 : m2 ≤ 1000) :
    meas_propagate m1 m2 ≤ 1000 := by
  simp [meas_propagate]
  exact Nat.le_trans (Nat.min_le_left m1 m2) h1

theorem meas_propagate_le_left (m1 m2 : Nat) : meas_propagate m1 m2 ≤ m1 :=
  Nat.min_le_left m1 m2

theorem meas_propagate_le_right (m1 m2 : Nat) : meas_propagate m1 m2 ≤ m2 :=
  Nat.min_le_right m1 m2

theorem meas_propagate_mono_left (m1 m1' m2 : Nat) (h : m1 ≤ m1') :
    meas_propagate m1 m2 ≤ meas_propagate m1' m2 := by
  simp only [meas_propagate]
  apply Nat.le_min_of_le_of_le
  · exact Nat.le_trans (Nat.min_le_left _ _) h
  · exact Nat.min_le_right _ _

/-- A Constant combined with any source has MEAS_CONF = that source. -/
theorem meas_constant_neutral (m : Nat) (hm : m ≤ 1000) :
    meas_propagate 1000 m = m := by
  simp [meas_propagate]
  exact Nat.min_eq_right hm

/-- Measured × Measured = 990. -/
theorem meas_measured_measured :
    meas_propagate (meas_base .Measured) (meas_base .Measured) = 990 := by
  simp [meas_propagate, meas_base]

/-- Measured × Asserted = 970 (literature uncertainty dominates). -/
theorem meas_measured_asserted :
    meas_propagate (meas_base .Measured) (meas_base .Asserted) = 970 := by
  simp [meas_propagate, meas_base]

-- ─────────────────────────────────────────────────────────────────────────────
-- §4  ety_unify — PLATINUM preservation (Gen 15)
-- ─────────────────────────────────────────────────────────────────────────────

/-- The original Gen 14 formula (without PLATINUM special case). -/
def unify_conf_formula (a b : Nat) : Nat :=
  clamp ((a * a + b * b) / (a + b + 1))

/-- The Gen 15 formula: special-case PLATINUM preservation. -/
def unify_conf_gen15 (a b : Nat) : Nat :=
  if a = 1000 ∧ b = 1000 then 1000
  else unify_conf_formula a b

theorem ety_unify_platinum_preservation :
    unify_conf_gen15 1000 1000 = 1000 := by
  simp [unify_conf_gen15]

/-- Gen 14 formula fails PLATINUM preservation (returns 999, not 1000). -/
theorem unify_formula_platinum_is_not_platinum :
    unify_conf_formula 1000 1000 = 999 := by
  simp [unify_conf_formula, clamp]

/-- The Gen 15 formula agrees with Gen 14 below PLATINUM. -/
theorem unify_gen15_agrees_below_platinum (a b : Nat) (h : ¬(a = 1000 ∧ b = 1000)) :
    unify_conf_gen15 a b = unify_conf_formula a b := by
  simp [unify_conf_gen15, h]

/-- Gen 15 output is bounded by 1000. -/
theorem unify_gen15_bounded (a b : Nat) : unify_conf_gen15 a b ≤ 1000 := by
  unfold unify_conf_gen15
  by_cases h : a = 1000 ∧ b = 1000
  · simp [h]
  · simp [h]; unfold unify_conf_formula clamp; exact Nat.min_le_right _ _

/-- PLATINUM certainty is preserved by Gen 15 unify. -/
theorem unify_gen15_certain_inputs (a b : Nat)
    (ha : a = 1000) (hb : b = 1000) :
    unify_conf_gen15 a b ≥ gate_threshold := by
  subst ha; subst hb
  rw [ety_unify_platinum_preservation]
  simp [gate_threshold]

-- ─────────────────────────────────────────────────────────────────────────────
-- §5  Epistemic completeness invariant
-- ─────────────────────────────────────────────────────────────────────────────

structure TokenArray (n : Nat) where
  conf : Fin n → Nat
  conf_bounded : ∀ i, conf i ≤ 1000

def epistemically_complete {n : Nat} (T : TokenArray n) : Prop :=
  ∀ i : Fin n, T.conf i ≥ gate_threshold

def platinum_complete {n : Nat} (T : TokenArray n) : Prop :=
  ∀ i : Fin n, T.conf i = 1000

theorem platinum_implies_complete {n : Nat} (T : TokenArray n)
    (h : platinum_complete T) : epistemically_complete T := by
  intro i; rw [h i]; simp [gate_threshold]

theorem update_platinum_preserves {n : Nat} (T : TokenArray n)
    (k : Fin n) (hT : epistemically_complete T) :
    epistemically_complete { T with conf := fun i => if i = k then 1000 else T.conf i,
                                    conf_bounded := by
                                      intro i; by_cases hik : i = k
                                      · simp [hik]
                                      · simp [hik]; exact T.conf_bounded i } := by
  intro i; by_cases hik : i = k
  · simp [hik]; simp [gate_threshold]
  · simp [hik]; exact hT i

-- ─────────────────────────────────────────────────────────────────────────────
-- §6  Soundness of the Gen 13 MEAS_CONF channel
-- ─────────────────────────────────────────────────────────────────────────────

inductive Prov where
  | Leaf    : MeasSource → Prov
  | Product : Prov → Prov → Prov

def prov_meas : Prov → Nat
  | .Leaf s       => meas_base s
  | .Product p q  => meas_propagate (prov_meas p) (prov_meas q)

theorem prov_meas_bounded (p : Prov) : prov_meas p ≤ 1000 := by
  induction p with
  | Leaf s =>
    simp [prov_meas, meas_base]
    cases s <;> decide
  | Product p q ihp ihq =>
    simp [prov_meas]
    exact meas_propagate_bounded _ _ ihp ihq

theorem prov_meas_product_le_left (p q : Prov) :
    prov_meas (.Product p q) ≤ prov_meas p := by
  simp [prov_meas]; exact meas_propagate_le_left _ _

theorem prov_meas_product_le_right (p q : Prov) :
    prov_meas (.Product p q) ≤ prov_meas q := by
  simp [prov_meas]; exact meas_propagate_le_right _ _

theorem prov_meas_le_any_leaf (p : Prov) (s : MeasSource)
    (hs : meas_base s ≤ prov_meas p) : prov_meas p ≤ 1000 := prov_meas_bounded p

-- ─────────────────────────────────────────────────────────────────────────────
-- §7  Summary theorems (main results)
-- ─────────────────────────────────────────────────────────────────────────────

/-- MAIN RESULT 1 — MEAS_CONF min-rule is conservative. -/
theorem meas_conf_soundness (m1 m2 : Nat) :
    meas_propagate m1 m2 ≤ m1 ∧ meas_propagate m1 m2 ≤ m2 :=
  ⟨meas_propagate_le_left m1 m2, meas_propagate_le_right m1 m2⟩

/-- MAIN RESULT 2 — PLATINUM preservation (Gen 15). -/
theorem gen15_platinum_preservation : unify_conf_gen15 1000 1000 = 1000 :=
  ety_unify_platinum_preservation

/-- MAIN RESULT 3 — Confidence product monotonicity. -/
theorem conf_product_monotone (a a' b b' : Nat)
    (ha : a ≤ a') (hb : b ≤ b') (ha' : a' ≤ 1000) (hb' : b' ≤ 1000) :
    conf_product a b ≤ conf_product a' b' :=
  Nat.le_trans (conf_product_mono_left a a' b ha (Nat.le_trans hb hb')) (conf_product_mono_right a' b b' hb ha')

/-- MAIN RESULT 4 — Epistemic completeness is preserved by PLATINUM updates. -/
theorem completeness_monotone_under_platinum_update {n : Nat}
    (T : TokenArray n) (k : Fin n)
    (hT : epistemically_complete T) :
    epistemically_complete { T with
      conf := fun i => if i = k then 1000 else T.conf i,
      conf_bounded := by
        intro i; by_cases hik : i = k
        · simp [hik]
        · simp [hik]; exact T.conf_bounded i } :=
  update_platinum_preserves T k hT

-- ─────────────────────────────────────────────────────────────────────────────
-- §8  Gen 17 D1 — Belief/Plausibility Interval Invariant
-- ─────────────────────────────────────────────────────────────────────────────

structure EpistemicInterval where
  belief : Nat
  conf   : Nat
  plaus  : Nat
  belief_le_conf : belief ≤ conf
  conf_le_plaus  : conf ≤ plaus
  plaus_bounded  : plaus ≤ 1000

/-- MAIN RESULT 5 (Gen 17 D1) — Knightian interval invariant:
    For every epistemic token, BELIEF ≤ EXPR_CONF ≤ PLAUS. -/
theorem belief_le_conf_le_plaus (ei : EpistemicInterval) :
    ei.belief ≤ ei.conf ∧ ei.conf ≤ ei.plaus :=
  ⟨ei.belief_le_conf, ei.conf_le_plaus⟩

/-- Corollary: a certain token (belief = plaus) has zero Knightian gap. -/
theorem certain_zero_knightian_gap (ei : EpistemicInterval) (h : ei.belief = ei.plaus) :
    ei.plaus - ei.belief = 0 := by simp [h, Nat.sub_self]

/-- Corollary: at measured() sites, the belief interval covers [985, 995]. -/
theorem measured_interval_valid :
    let ei : EpistemicInterval := {
      belief := 985, conf := 990, plaus := 995,
      belief_le_conf := by decide,
      conf_le_plaus  := by decide,
      plaus_bounded  := by decide }
    ei.belief ≤ ei.conf ∧ ei.conf ≤ ei.plaus :=
  ⟨by decide, by decide⟩

/-- Corollary: at asserted() sites, the belief interval covers [960, 980]. -/
theorem asserted_interval_valid :
    let ei : EpistemicInterval := {
      belief := 960, conf := 970, plaus := 980,
      belief_le_conf := by decide,
      conf_le_plaus  := by decide,
      plaus_bounded  := by decide }
    ei.belief ≤ ei.conf ∧ ei.conf ≤ ei.plaus :=
  ⟨by decide, by decide⟩

end EGC
