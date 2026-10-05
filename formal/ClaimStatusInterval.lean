/-!
# Sounio.ClaimStatusInterval — interval (second-order) confidences for claims

Formal companion to
`artifacts/ontology-frontiers/epistemic-claim-status/interval_claims.sio`
(frontier: `epistemic-claim-status`; see the `FRONTIER.md` there). It extends
`formal/OntologyClaimStatus.lean`, which models confidence as a single
per-mille scalar, closing the documented gap *"second-order uncertainty not
modelled"* with **interval confidences**.

## Setting

A claim now carries an interval `IConf = [lo, hi]` of per-mille confidences
(`Nat`, 0–1000) instead of a point value: `lo` is the most pessimistic
justified confidence, `hi` the most optimistic. Validity means the interval
is non-empty and in range: `lo ≤ hi ≤ 1000`.

The two propagation rules of the scalar model lift pointwise to the bounds:

1. **Interval weakest link (derivation).** `wl a b = [min a.lo b.lo,
   min a.hi b.hi]`. Proved: validity preservation (`wl_valid`), soundness
   (`wl_sound` — the derived interval contains the pointwise weakest link of
   any points inside the premise intervals), and threshold preservation for
   the `lo` side along chains of arbitrary length (`chainIConf_lo_ge`).

2. **Interval Dempster-Shafer fusion (independent sources).** Each bound is
   fused with the scaled numerator of the scalar model,
   `dsNum a b = 1000·1000 - (1000-a)·(1000-b)`, and rescaled by division:
   `ds a b = [dsNum a.lo b.lo / 1000, dsNum a.hi b.hi / 1000]`. Proved:
   validity preservation (`ds_valid`), soundness (`ds_sound` — the fused
   interval contains the pointwise fusion of any interior points, from the
   monotonicity of `dsNum` in each argument, `dsNum_mono_left` /
   `dsNum_mono_right`, themselves proved from `Nat.mul_le_mul`), threshold
   preservation on the `lo` side (`ds_lo_ge`), and the lift property
   `fused.lo ≥ max (a.lo) (b.lo)` (`ds_lo_ge_max`).

Concrete instances reproduce the numbers of the `.sio` prototype and are
checked by `native_decide`/`decide`.

Self-contained. No Mathlib. Zero sorry. No new axioms.
-/

namespace Sounio.ClaimStatusInterval

-- ---------------------------------------------------------------------------
-- §0. Interval confidences
-- ---------------------------------------------------------------------------

/-- Per-mille interval confidence: `[lo, hi]` with `lo` the pessimistic and
    `hi` the optimistic justified confidence. -/
structure IConf where
  lo : Nat
  hi : Nat
  deriving DecidableEq, Repr

/-- Validity: the interval is non-empty and lies in the per-mille range. -/
def IConf.valid (c : IConf) : Prop := c.lo ≤ c.hi ∧ c.hi ≤ 1000

/-- A point lies inside an interval. -/
def IConf.contains (c : IConf) (x : Nat) : Prop := c.lo ≤ x ∧ x ≤ c.hi

-- ---------------------------------------------------------------------------
-- §1. Interval weakest-link derivation
-- ---------------------------------------------------------------------------

/-- Interval weakest link: pointwise minimum of the bounds. -/
def wl (a b : IConf) : IConf := ⟨min a.lo b.lo, min a.hi b.hi⟩

/-- **Validity preservation**: the weakest link of valid intervals is valid. -/
theorem wl_valid {a b : IConf} (ha : a.valid) (hb : b.valid) : (wl a b).valid := by
  obtain ⟨ha1, ha2⟩ := ha
  obtain ⟨hb1, hb2⟩ := hb
  constructor
  · show min a.lo b.lo ≤ min a.hi b.hi
    omega
  · show min a.hi b.hi ≤ 1000
    omega

/-- **Soundness**: the interval weakest link contains the pointwise weakest
    link of any points inside the premise intervals. -/
theorem wl_sound {a b : IConf} {x y : Nat}
    (hx : a.contains x) (hy : b.contains y) :
    (wl a b).contains (min x y) := by
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hy1, hy2⟩ := hy
  constructor
  · show min a.lo b.lo ≤ min x y
    omega
  · show min x y ≤ min a.hi b.hi
    omega

/-- Threshold preservation for a single derivation step (`lo` side). -/
theorem wl_lo_ge {t : Nat} {a b : IConf} (ha : t ≤ a.lo) (hb : t ≤ b.lo) :
    t ≤ (wl a b).lo := by
  show t ≤ min a.lo b.lo
  omega

-- ---------------------------------------------------------------------------
-- §2. Derivation chains of arbitrary length (interval version)
-- ---------------------------------------------------------------------------

/-- Confidence of an interval derivation chain: fold the premises through
    `wl`, starting from the first premise's interval. -/
def chainIConf (acc : IConf) (l : List IConf) : IConf := l.foldl wl acc

/-- **Threshold preservation**: if the first premise's `lo` and every chain
    premise's `lo` meet threshold `t`, so does the derived interval's `lo`. -/
theorem chainIConf_lo_ge {t : Nat} (acc : IConf) (l : List IConf)
    (hacc : t ≤ acc.lo) (hl : ∀ x ∈ l, t ≤ x.lo) :
    t ≤ (chainIConf acc l).lo := by
  induction l generalizing acc with
  | nil => exact hacc
  | cons y ys ih =>
      have h1 : chainIConf acc (y :: ys) = chainIConf (wl acc y) ys := rfl
      rw [h1]
      apply ih
      · have hy : t ≤ y.lo := hl y (List.mem_cons.mpr (Or.inl rfl))
        show t ≤ min acc.lo y.lo
        omega
      · intro x hx
        exact hl x (List.mem_cons.mpr (Or.inr hx))

-- ---------------------------------------------------------------------------
-- §3. Interval Dempster-Shafer fusion
-- ---------------------------------------------------------------------------

/-- Scaled Dempster-Shafer numerator for per-mille confidences (as in
    `OntologyClaimStatus`): `dsNum a b = 1000·ds(a,b)` where
    `ds(a,b) = 1-(1-a/1000)(1-b/1000)`. -/
def dsNum (a b : Nat) : Nat := 1000 * 1000 - (1000 - a) * (1000 - b)

theorem dsNum_comm (a b : Nat) : dsNum a b = dsNum b a := by
  unfold dsNum
  rw [Nat.mul_comm (1000 - a) (1000 - b)]

/-- The scaled numerator never exceeds `1000·1000`. -/
theorem dsNum_le (a b : Nat) : dsNum a b ≤ 1000 * 1000 :=
  Nat.sub_le _ _

/-- **Monotonicity in the left argument** (from `Nat.mul_le_mul`). Truncated
    subtraction keeps this true even above 1000. -/
theorem dsNum_mono_left {a a' : Nat} (h : a ≤ a') (b : Nat) :
    dsNum a b ≤ dsNum a' b := by
  have h1 : 1000 - a' ≤ 1000 - a := Nat.sub_le_sub_left h 1000
  have h2 : (1000 - a') * (1000 - b) ≤ (1000 - a) * (1000 - b) :=
    Nat.mul_le_mul h1 (Nat.le_refl (1000 - b))
  show 1000 * 1000 - (1000 - a) * (1000 - b) ≤
       1000 * 1000 - (1000 - a') * (1000 - b)
  exact Nat.sub_le_sub_left h2 (1000 * 1000)

/-- **Monotonicity in the right argument** (from `Nat.mul_le_mul`). -/
theorem dsNum_mono_right (a : Nat) {b b' : Nat} (h : b ≤ b') :
    dsNum a b ≤ dsNum a b' := by
  have h1 : 1000 - b' ≤ 1000 - b := Nat.sub_le_sub_left h 1000
  have h2 : (1000 - a) * (1000 - b') ≤ (1000 - a) * (1000 - b) :=
    Nat.mul_le_mul (Nat.le_refl (1000 - a)) h1
  show 1000 * 1000 - (1000 - a) * (1000 - b) ≤
       1000 * 1000 - (1000 - a) * (1000 - b')
  exact Nat.sub_le_sub_left h2 (1000 * 1000)

/-- Joint monotonicity: `dsNum` grows when both arguments grow. -/
theorem dsNum_mono {a a' b b' : Nat} (ha : a ≤ a') (hb : b ≤ b') :
    dsNum a b ≤ dsNum a' b' :=
  Nat.le_trans (dsNum_mono_left ha b) (dsNum_mono_right a' hb)

/-- Interval DS fusion: fuse the bounds with the scaled numerator and rescale
    by exact per-mille division. -/
def ds (a b : IConf) : IConf :=
  ⟨dsNum a.lo b.lo / 1000, dsNum a.hi b.hi / 1000⟩

/-- **Validity preservation**: the DS fusion of valid intervals is valid. -/
theorem ds_valid {a b : IConf} (ha : a.valid) (hb : b.valid) : (ds a b).valid := by
  obtain ⟨ha1, _⟩ := ha
  obtain ⟨hb1, _⟩ := hb
  constructor
  · show dsNum a.lo b.lo / 1000 ≤ dsNum a.hi b.hi / 1000
    exact Nat.div_le_div_right (dsNum_mono ha1 hb1)
  · show dsNum a.hi b.hi / 1000 ≤ 1000
    have h : dsNum a.hi b.hi / 1000 ≤ (1000 * 1000) / 1000 :=
      Nat.div_le_div_right (dsNum_le a.hi b.hi)
    omega

/-- **Soundness**: the fused interval contains the pointwise fusion of any
    points inside the source intervals. -/
theorem ds_sound {a b : IConf} {x y : Nat}
    (hx : a.contains x) (hy : b.contains y) :
    (ds a b).contains (dsNum x y / 1000) := by
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hy1, hy2⟩ := hy
  constructor
  · show dsNum a.lo b.lo / 1000 ≤ dsNum x y / 1000
    exact Nat.div_le_div_right (dsNum_mono hx1 hy1)
  · show dsNum x y / 1000 ≤ dsNum a.hi b.hi / 1000
    exact Nat.div_le_div_right (dsNum_mono hx2 hy2)

/-- The scaled fusion never drops below the left source (scaled). -/
theorem dsNum_ge_left {a b : Nat} (ha : a ≤ 1000) :
    1000 * a ≤ dsNum a b := by
  have h1 : (1000 - a) * (1000 - b) ≤ (1000 - a) * 1000 :=
    Nat.mul_le_mul (Nat.le_refl (1000 - a)) (Nat.sub_le 1000 b)
  have h2 : (1000 - a) * 1000 = 1000 * 1000 - a * 1000 :=
    Nat.sub_mul 1000 a 1000
  have h3 : 1000 * 1000 - ((1000 - a) * 1000) = a * 1000 := by
    rw [h2]
    have ha2 : a * 1000 ≤ 1000 * 1000 :=
      Nat.mul_le_mul ha (Nat.le_refl 1000)
    exact Nat.sub_sub_self ha2
  calc 1000 * a = a * 1000 := Nat.mul_comm 1000 a
    _ = 1000 * 1000 - ((1000 - a) * 1000) := h3.symm
    _ ≤ 1000 * 1000 - (1000 - a) * (1000 - b) :=
        Nat.sub_le_sub_left h1 (1000 * 1000)
    _ = dsNum a b := rfl

/-- The scaled fusion never drops below the right source (scaled). -/
theorem dsNum_ge_right {a b : Nat} (hb : b ≤ 1000) :
    1000 * b ≤ dsNum a b := by
  rw [dsNum_comm]
  exact dsNum_ge_left hb

/-- **Lift property (`lo` side)**: the fused `lo` never drops below the left
    source's `lo`. -/
theorem ds_lo_ge_left {a b : IConf} (ha : a.lo ≤ 1000) :
    a.lo ≤ (ds a b).lo := by
  have h1 : 1000 * a.lo ≤ dsNum a.lo b.lo := dsNum_ge_left ha
  show a.lo ≤ dsNum a.lo b.lo / 1000
  omega

/-- **Lift property (`lo` side)**: the fused `lo` never drops below the right
    source's `lo`. -/
theorem ds_lo_ge_right {a b : IConf} (hb : b.lo ≤ 1000) :
    b.lo ≤ (ds a b).lo := by
  have h1 : 1000 * b.lo ≤ dsNum a.lo b.lo := dsNum_ge_right hb
  show b.lo ≤ dsNum a.lo b.lo / 1000
  omega

/-- The fused `lo` never drops below the best source `lo`. -/
theorem ds_lo_ge_max {a b : IConf} (ha : a.lo ≤ 1000) (hb : b.lo ≤ 1000) :
    max a.lo b.lo ≤ (ds a b).lo := by
  cases Nat.le_total a.lo b.lo with
  | inl hab =>
      rw [Nat.max_eq_right hab]
      exact ds_lo_ge_right hb
  | inr hba =>
      rw [Nat.max_eq_left hba]
      exact ds_lo_ge_left ha

/-- **Threshold preservation (fusion)**: if both source `lo`s meet threshold
    `t`, so does the fused interval's `lo`. -/
theorem ds_lo_ge {t : Nat} {a b : IConf}
    (ha : t ≤ a.lo) (_hb : t ≤ b.lo) (ha1000 : a.lo ≤ 1000) :
    t ≤ (ds a b).lo :=
  Nat.le_trans ha (ds_lo_ge_left ha1000)

-- ---------------------------------------------------------------------------
-- §4. Concrete instances (the `.sio` prototype's numbers)
-- ---------------------------------------------------------------------------

/-- Interval weakest link of [0.90, 0.98] and [0.85, 0.95] is [0.85, 0.95]. -/
theorem ex_wl : wl ⟨900, 980⟩ ⟨850, 950⟩ = ⟨850, 950⟩ := by native_decide

/-- Chain [0.90,0.98] → [0.85,0.95] → [0.88,0.96] keeps lo at 0.85. -/
theorem ex_chain :
    chainIConf ⟨900, 980⟩ [⟨850, 950⟩, ⟨880, 960⟩] = ⟨850, 950⟩ := by
  native_decide

/-- The chain's lo clears the 0.85 high-confidence threshold. -/
theorem ex_chain_threshold :
    850 ≤ (chainIConf ⟨900, 980⟩ [⟨850, 950⟩, ⟨880, 960⟩]).lo := by
  native_decide

/-- DS fusion of assay [0.55, 0.65] and docking [0.50, 0.60]:
    lo = dsNum 550 500 / 1000 = 775000/1000 = 0.775,
    hi = dsNum 650 600 / 1000 = 860000/1000 = 0.860. -/
theorem ex_ds : ds ⟨550, 650⟩ ⟨500, 600⟩ = ⟨775, 860⟩ := by native_decide

/-- The fused interval contains the pointwise fusion of the midpoints:
    dsNum 600 550 / 1000 = 820 ∈ [775, 860]. -/
theorem ex_ds_contains_midpoint :
    (ds ⟨550, 650⟩ ⟨500, 600⟩).contains (dsNum 600 550 / 1000) := by
  unfold IConf.contains
  native_decide

/-- The fused lo (0.775) clears the best source lo (0.55) — and even the
    0.70 threshold that no single source lo meets. -/
theorem ex_ds_lifts :
    max 550 500 ≤ (ds ⟨550, 650⟩ ⟨500, 600⟩).lo ∧
    700 ≤ (ds ⟨550, 650⟩ ⟨500, 600⟩).lo := by
  native_decide

/-- Validity of all concrete intervals involved. -/
theorem ex_valid :
    IConf.valid ⟨900, 980⟩ ∧ IConf.valid ⟨850, 950⟩ ∧
    IConf.valid ⟨880, 960⟩ ∧ IConf.valid ⟨550, 650⟩ ∧
    IConf.valid ⟨500, 600⟩ ∧ IConf.valid (ds ⟨550, 650⟩ ⟨500, 600⟩) := by
  unfold IConf.valid
  native_decide

end Sounio.ClaimStatusInterval
