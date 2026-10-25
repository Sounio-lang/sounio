import SounioIntervalExtended

/-!
ADR-019 followup: exact four-corner endpoint products, then reciprocal/division.
The additive model remains unchanged. Infinity is not a finite member. The
endpoint product handles 0*infinity as a set-bound case before arithmetic.
No IEEE instruction, decimal conversion or signed-zero encoding is modeled.
-/
namespace Sounio.IntervalExtended.MulDiv
open Std Sounio.IntervalExtended

/-- Total endpoint order; the rational order is the canonical Std order. -/
def ELe : ExtendedRat → ExtendedRat → Prop
  | .negInf,_ => True
  | _,.posInf => True
  | .finite a,.finite b => a ≤ b
  | _,_ => False

instance (a b : ExtendedRat) : Decidable (ELe a b) := by
  cases a <;> cases b <;> unfold ELe <;> infer_instance

def minE (a b : ExtendedRat) := if ELe a b then a else b

def maxE (a b : ExtendedRat) := if ELe a b then b else a

theorem le_refl (a : ExtendedRat) : ELe a a := by
  cases a with
  | finite q => exact Rat.le_refl
  | _ => trivial

theorem le_trans (a b c : ExtendedRat) (hab : ELe a b) (hbc : ELe b c) : ELe a c := by
  cases a <;> cases b <;> cases c <;> simp_all [ELe] <;> exact Rat.le_trans hab hbc

theorem le_total (a b : ExtendedRat) : ELe a b ∨ ELe b a := by
  cases a <;> cases b <;> simp_all [ELe] <;> grind

theorem min_le_left (a b : ExtendedRat) : ELe (minE a b) a := by
  unfold minE;split
  · exact le_refl _
  · rename_i h;exact (le_total a b).resolve_left h

theorem min_le_right (a b : ExtendedRat) : ELe (minE a b) b := by
  unfold minE;split
  · assumption
  · exact le_refl _

theorem le_max_left (a b : ExtendedRat) : ELe a (maxE a b) := by
  unfold maxE;split
  · assumption
  · exact le_refl _

theorem le_max_right (a b : ExtendedRat) : ELe b (maxE a b) := by
  unfold maxE;split
  · exact le_refl _
  · rename_i h;exact (le_total a b).resolve_left h

theorem le_min (a b c : ExtendedRat) (hab : ELe a b) (hac : ELe a c) : ELe a (minE b c) := by
  unfold minE;split <;> assumption

theorem max_le (a b c : ExtendedRat) (hac : ELe a c) (hbc : ELe b c) : ELe (maxE a b) c := by
  unfold maxE;split <;> assumption

theorem min_mono (a b c d : ExtendedRat) (hac : ELe a c) (hbd : ELe b d) :
    ELe (minE a b) (minE c d) :=
  le_min _ _ _ (le_trans _ _ _ (min_le_left _ _) hac)
    (le_trans _ _ _ (min_le_right _ _) hbd)

theorem max_mono (a b c d : ExtendedRat) (hac : ELe a c) (hbd : ELe b d) :
    ELe (maxE a b) (maxE c d) :=
  max_le _ _ _ (le_trans _ _ _ hac (le_max_left _ _))
    (le_trans _ _ _ hbd (le_max_right _ _))

/-- Extended endpoint product. Zero is handled before infinite sign selection. -/
def mulE : ExtendedRat → ExtendedRat → ExtendedRat
  | .finite x,.finite y => .finite (x*y)
  | .finite x,.negInf => if x=0 then .finite 0 else if 0<x then .negInf else .posInf
  | .negInf,.finite x => if x=0 then .finite 0 else if 0<x then .negInf else .posInf
  | .finite x,.posInf => if x=0 then .finite 0 else if 0<x then .posInf else .negInf
  | .posInf,.finite x => if x=0 then .finite 0 else if 0<x then .posInf else .negInf
  | .negInf,.negInf => .posInf
  | .posInf,.posInf => .posInf
  | _,_ => .negInf

theorem mulE_comm (a b : ExtendedRat) : mulE a b = mulE b a := by
  cases a <;> cases b <;> simp [mulE,Rat.mul_comm]

theorem mul_mono_nonneg (a x y : ExtendedRat) (ha : ELe (.finite 0) a)
    (hxy : ELe x y) : ELe (mulE a x) (mulE a y) := by
  cases a with
  | negInf => simp [ELe] at ha
  | finite a =>
    have hf : ∀ p q : Rat, p ≤ q → a*p ≤ a*q :=
      fun _ _ h => Rat.mul_le_mul_of_nonneg_left h ha
    cases x <;> cases y <;> simp_all [ELe,mulE] <;> grind [ELe]
  | posInf => cases x <;> cases y <;> simp_all [ELe,mulE] <;> grind [ELe]

theorem mul_mono_nonpos (a x y : ExtendedRat) (ha : ELe a (.finite 0))
    (hxy : ELe x y) : ELe (mulE a y) (mulE a x) := by
  cases a with
  | posInf => simp [ELe] at ha
  | finite a =>
    have hf : ∀ p q : Rat, p ≤ q → a*q ≤ a*p :=
      fun p q h => Sounio.IntervalFinite.mul_le_of_nonpos a p q ha h
    cases x <;> cases y <;> simp_all [ELe,mulE] <;> grind [ELe]
  | negInf => cases x <;> cases y <;> simp_all [ELe,mulE] <;> grind [ELe]

theorem scale_bounds (a c d y : ExtendedRat) (hcy : ELe c y) (hyd : ELe y d) :
    ELe (minE (mulE a c) (mulE a d)) (mulE a y) ∧
    ELe (mulE a y) (maxE (mulE a c) (mulE a d)) := by
  by_cases ha : ELe (.finite 0) a
  · exact ⟨le_trans _ _ _ (min_le_left _ _) (mul_mono_nonneg a c y ha hcy),
      le_trans _ _ _ (mul_mono_nonneg a y d ha hyd) (le_max_right _ _)⟩
  · have hn := (le_total (.finite 0) a).resolve_left ha
    exact ⟨le_trans _ _ _ (min_le_right _ _) (mul_mono_nonpos a y d hn hyd),
      le_trans _ _ _ (mul_mono_nonpos a c y hn hcy) (le_max_left _ _)⟩

def low4 (a b c d : ExtendedRat) := minE (minE a b) (minE c d)
def high4 (a b c d : ExtendedRat) := maxE (maxE a b) (maxE c d)

theorem product_corners (a b c d x y : ExtendedRat)
    (hax : ELe a x) (hxb : ELe x b) (hcy : ELe c y) (hyd : ELe y d) :
    ELe (low4 (mulE a c) (mulE a d) (mulE b c) (mulE b d)) (mulE x y) ∧
    ELe (mulE x y) (high4 (mulE a c) (mulE a d) (mulE b c) (mulE b d)) := by
  have ha := scale_bounds a c d y hcy hyd
  have hb := scale_bounds b c d y hcy hyd
  have h := scale_bounds y a b x hax hxb
  have hx : ELe (minE (mulE a y) (mulE b y)) (mulE x y) ∧
      ELe (mulE x y) (maxE (mulE a y) (mulE b y)) := by
    simpa only [mulE_comm] using h
  exact ⟨le_trans _ _ _ (min_mono _ _ _ _ ha.1 hb.1) hx.1,
    le_trans _ _ _ hx.2 (max_mono _ _ _ _ ha.2 hb.2)⟩

/-- Infinite singletons and reversed/invalid endpoint pairs have no members. -/
def ofEndpoints : ExtendedRat → ExtendedRat → Interval
  | .negInf,.posInf => entire
  | .negInf,.finite b => make none (some b)
  | .finite a,.posInf => make (some a) none
  | .finite a,.finite b => make (some a) (some b)
  | _,_ => .empty

theorem ofEndpoints_contains (a b : ExtendedRat) (x : Rat)
    (ha : ELe a (.finite x)) (hb : ELe (.finite x) b) : Contains (ofEndpoints a b) x := by
  cases a with
  | posInf => exact False.elim ha
  | negInf =>
    cases b with
    | negInf => exact False.elim hb
    | finite b => exact make_contains none (some b) x ⟨True.intro,hb⟩
    | posInf => exact entire_contains x
  | finite a =>
    cases b with
    | negInf => exact False.elim hb
    | finite b => exact make_contains (some a) (some b) x ⟨ha,hb⟩
    | posInf => exact make_contains (some a) none x ⟨ha,True.intro⟩

theorem member_lower (i : Interval) (x : Rat) (hx : Contains i x) : ELe (lowerEndpoint i) (.finite x) := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp => cases a <;> exact hx.1

theorem member_upper (i : Interval) (x : Rat) (hx : Contains i x) : ELe (.finite x) (upperEndpoint i) := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp => cases b <;> exact hx.2

def mul (i j : Interval) : Interval :=
  match i,j with
  | .empty,_ => .empty
  | _,.empty => .empty
  | _,_ =>
    ofEndpoints
      (low4 (mulE (lowerEndpoint i) (lowerEndpoint j)) (mulE (lowerEndpoint i) (upperEndpoint j))
        (mulE (upperEndpoint i) (lowerEndpoint j)) (mulE (upperEndpoint i) (upperEndpoint j)))
      (high4 (mulE (lowerEndpoint i) (lowerEndpoint j)) (mulE (lowerEndpoint i) (upperEndpoint j))
        (mulE (upperEndpoint i) (lowerEndpoint j)) (mulE (upperEndpoint i) (upperEndpoint j)))

theorem mul_contains (i j : Interval) (x y : Rat) (hx : Contains i x) (hy : Contains j y) :
    Contains (mul i j) (x*y) := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp =>
    cases j with
    | empty => exact False.elim hy
    | span c d hq =>
      exact ofEndpoints_contains _ _ _
        (product_corners _ _ _ _ _ _ (member_lower _ _ hx) (member_upper _ _ hx)
          (member_lower _ _ hy) (member_upper _ _ hy)).1
        (product_corners _ _ _ _ _ _ (member_lower _ _ hx) (member_upper _ _ hx)
          (member_lower _ _ hy) (member_upper _ _ hy)).2

/-- Finite pole/sign branches. No reciprocal is formed for the zero-point set. -/
def reciprocalFinite (a b : Rat) : Interval :=
  if a=0 then
    if b=0 then .empty else make (some b⁻¹) none
  else if b=0 then make none (some a⁻¹)
  else if 0<a ∨ b<0 then make (some b⁻¹) (some a⁻¹)
  else entire

theorem reciprocalFinite_contains (a b y : Rat) (hab : a ≤ b)
    (hy : a ≤ y ∧ y ≤ b) (hne : y ≠ 0) : Contains (reciprocalFinite a b) y⁻¹ := by
  by_cases ha : a=0
  · subst a
    by_cases hb : b=0
    · subst b;have hz : y=0 := Rat.le_antisymm hy.2 hy.1;exact False.elim (hne hz)
    · have hypos : 0<y := by grind
      have hbpos : 0<b := by grind
      have lower := Sounio.IntervalFinite.inverse_order y b hne (Rat.ne_of_gt hbpos)
        hy.2 (Rat.mul_pos hypos hbpos)
      simpa [reciprocalFinite,hb] using make_contains (some b⁻¹) none y⁻¹ ⟨lower,True.intro⟩
  · by_cases hb : b=0
    · subst b
      have hyneg : y<0 := by grind
      have haneg : a<0 := by grind
      have upper := Sounio.IntervalFinite.inverse_order a y (Rat.ne_of_lt haneg) hne
        hy.1 (Sounio.IntervalFinite.negative_product_pos a y haneg hyneg)
      simpa [reciprocalFinite,ha] using make_contains none (some a⁻¹) y⁻¹ ⟨True.intro,upper⟩
    · by_cases sep : 0<a ∨ b<0
      · have bounds := Sounio.IntervalFinite.reciprocal_bounds ⟨a,b,hab⟩ y hy sep
        simpa [reciprocalFinite,ha,hb,sep] using make_contains (some b⁻¹) (some a⁻¹) y⁻¹ bounds
      · simpa [reciprocalFinite,ha,hb,sep] using entire_contains y⁻¹

theorem inverse_nonpos (y : Rat) (hy : y<0) : y⁻¹ ≤ 0 := by
  have h : ¬0<y⁻¹ := by intro hp;have := Rat.inv_pos.mp hp;grind
  grind

/-- Inversion maps infinite bounds to the finite closure endpoint zero. A pole
    touching an endpoint creates a half-line; a strict crossing takes its hull. -/
def reciprocal : Interval → Interval
  | .empty => .empty
  | .span (some a) (some b) _ => reciprocalFinite a b
  | .span none (some b) _ =>
    if b<0 then make (some b⁻¹) (some 0)
    else if b=0 then make none (some 0)
    else entire
  | .span (some a) none _ =>
    if 0<a then make (some 0) (some a⁻¹)
    else if a=0 then make (some 0) none
    else entire
  | .span none none _ => entire

theorem reciprocal_contains (i : Interval) (y : Rat) (hy : Contains i y)
    (hne : y ≠ 0) : Contains (reciprocal i) y⁻¹ := by
  cases i with
  | empty => exact False.elim hy
  | span lo hi hp =>
    cases lo with
    | none =>
      cases hi with
      | none => exact entire_contains _
      | some b =>
        by_cases hb : b<0
        · have hyneg : y<0 := by have := hy.2;change y≤b at this;grind
          have hupper := inverse_nonpos y hyneg
          have hlower := Sounio.IntervalFinite.inverse_order y b hne (Rat.ne_of_lt hb)
            hy.2 (Sounio.IntervalFinite.negative_product_pos y b hyneg hb)
          simpa [reciprocal,hb] using make_contains (some b⁻¹) (some 0) y⁻¹ ⟨hlower,hupper⟩
        · by_cases hz : b=0
          · subst b
            have hyneg : y<0 := by have := hy.2;change y≤0 at this;grind
            simpa [reciprocal] using make_contains none (some 0) y⁻¹ ⟨True.intro,inverse_nonpos y hyneg⟩
          · simpa [reciprocal,hb,hz] using entire_contains y⁻¹
    | some a =>
      cases hi with
      | some b => exact reciprocalFinite_contains a b y (hp a b rfl rfl) hy hne
      | none =>
        by_cases ha : 0<a
        · have hypos : 0<y := by have := hy.1;change a≤y at this;grind
          have hlower := Rat.le_of_lt (Rat.inv_pos.mpr hypos)
          have hupper := Sounio.IntervalFinite.inverse_order a y (Rat.ne_of_gt ha) hne
            hy.1 (Rat.mul_pos ha hypos)
          simpa [reciprocal,ha] using make_contains (some 0) (some a⁻¹) y⁻¹ ⟨hlower,hupper⟩
        · by_cases hz : a=0
          · subst a
            have hypos : 0<y := by have := hy.1;change 0≤y at this;grind
            simpa [reciprocal] using make_contains (some 0) none y⁻¹ ⟨Rat.le_of_lt (Rat.inv_pos.mpr hypos),True.intro⟩
          · simpa [reciprocal,ha,hz] using entire_contains y⁻¹

def div (i j : Interval) : Interval := mul i (reciprocal j)

theorem div_contains (i j : Interval) (x y : Rat) (hx : Contains i x) (hy : Contains j y)
    (hne : y ≠ 0) : Contains (div i j) (x/y) :=
  mul_contains i (reciprocal j) x y⁻¹ hx (reciprocal_contains j y hy hne)

def containsZero : Interval → Bool
  | .empty => false
  | .span lo hi _ =>
    (match lo with | none => true | some q => decide (q≤0)) &&
    (match hi with | none => true | some q => decide (0≤q))

theorem containsZero_correct (i : Interval) : containsZero i = true ↔ Contains i 0 := by
  cases i with
  | empty => simp [containsZero,Contains]
  | span lo hi hp => cases lo <;> cases hi <;> simp [containsZero,Contains,Lower,Upper]

def mulDecorated (a b : Decorated) : Decorated :=
  let i := mul a.interval b.interval
  ⟨i,cap (meet a.decoration b.decoration) i⟩

def divDecorated (a b : Decorated) : Decorated :=
  let i := div a.interval b.interval
  let d := cap (meet a.decoration b.decoration) i
  ⟨i,if containsZero b.interval then meet d .trv else d⟩

theorem mul_decoration_left (a b : Decorated) :
    rank (mulDecorated a b).decoration ≤ rank a.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_left _ _)

theorem mul_decoration_right (a b : Decorated) :
    rank (mulDecorated a b).decoration ≤ rank b.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_right _ _)

theorem div_zero_containing_decoration (a b : Decorated) (hz : Contains b.interval 0) :
    rank (divDecorated a b).decoration ≤ rank .trv := by
  have h := (containsZero_correct b.interval).mpr hz
  simp only [divDecorated,h,ite_true]
  exact meet_rank_right _ _

def zeroPoint : Interval := make (some 0) (some 0)

theorem zero_point_reciprocal : reciprocal zeroPoint = .empty := rfl

theorem zero_denominator_empty (i : Interval) : div i zeroPoint = .empty := by
  cases i <;> rfl

theorem zero_times_entire : lowerEndpoint (mul zeroPoint entire) = .finite 0 ∧
    upperEndpoint (mul zeroPoint entire) = .finite 0 := by decide +kernel

theorem zero_times_entire_tight (x : Rat) : Contains (mul zeroPoint entire) x ↔ x=0 := by
  change (0≤x ∧ x≤0) ↔ x=0
  constructor
  · intro h;exact Rat.le_antisymm h.2 h.1
  · intro h;subst x;exact ⟨Rat.le_refl,Rat.le_refl⟩

theorem positive_unbounded_product :
    lowerEndpoint (mul (positiveRay 1) (positiveRay 2)) = .finite 2 ∧
    upperEndpoint (mul (positiveRay 1) (positiveRay 2)) = .posInf := by decide +kernel

theorem negative_unbounded_product :
    lowerEndpoint (mul (negativeRay (-1)) (negativeRay (-2))) = .finite 2 ∧
    upperEndpoint (mul (negativeRay (-1)) (negativeRay (-2))) = .posInf := by decide +kernel

theorem negative_infinite_singleton_empty : ofEndpoints .negInf .negInf = .empty := rfl

theorem positive_infinite_singleton_empty : ofEndpoints .posInf .posInf = .empty := rfl

theorem positive_ray_reciprocal :
    lowerEndpoint (reciprocal (positiveRay 2)) = .finite 0 ∧
    upperEndpoint (reciprocal (positiveRay 2)) = .finite (1/2) := by decide +kernel

theorem negative_ray_reciprocal :
    lowerEndpoint (reciprocal (negativeRay (-2))) = .finite (-1/2) ∧
    upperEndpoint (reciprocal (negativeRay (-2))) = .finite 0 := by decide +kernel

theorem crossing_reciprocal_hull :
    lowerEndpoint (reciprocal (make (some (-2)) (some 3))) = .negInf ∧
    upperEndpoint (reciprocal (make (some (-2)) (some 3))) = .posInf := by decide +kernel

theorem touching_positive_reciprocal :
    lowerEndpoint (reciprocal (make (some 0) (some 2))) = .finite (1/2) ∧
    upperEndpoint (reciprocal (make (some 0) (some 2))) = .posInf := by decide +kernel

theorem touching_negative_reciprocal :
    lowerEndpoint (reciprocal (make (some (-2)) (some 0))) = .negInf ∧
    upperEndpoint (reciprocal (make (some (-2)) (some 0))) = .finite (-1/2) := by decide +kernel

/-- The canonical finite engine is preserved, not copied into this module. -/
theorem minE_finite (a b : Rat) : minE (.finite a) (.finite b) = .finite (min a b) := by
  simp only [minE,ELe,Rat.min_def]
  split <;> simp_all

theorem maxE_finite (a b : Rat) : maxE (.finite a) (.finite b) = .finite (max a b) := by
  simp only [maxE,ELe,Rat.max_def]
  split <;> simp_all

theorem make_ofFinite (i : Sounio.IntervalFinite.Interval) :
    make (some i.lo) (some i.hi) = ofFinite i := by simp [make,i.ordered,ofFinite]

theorem mul_ofFinite (i j : Sounio.IntervalFinite.Interval) :
    mul (ofFinite i) (ofFinite j) = ofFinite (Sounio.IntervalFinite.mul i j) := by
  simp only [mul,ofFinite,lowerEndpoint,upperEndpoint,mulE,low4,high4,minE_finite,maxE_finite,ofEndpoints]
  exact make_ofFinite (Sounio.IntervalFinite.mul i j)

theorem reciprocal_ofFinite (j : Sounio.IntervalFinite.Interval)
    (sep : Sounio.IntervalFinite.SeparatedZero j) :
    reciprocal (ofFinite j) = ofFinite (Sounio.IntervalFinite.reciprocal j sep) := by
  have hlo : j.lo ≠ 0 := by
    rcases sep with hp | hn <;> have := j.ordered <;> grind
  have hhi : j.hi ≠ 0 := by
    rcases sep with hp | hn <;> have := j.ordered <;> grind
  have hs : 0<j.lo ∨ j.hi<0 := sep
  have horder : j.hi⁻¹ ≤ j.lo⁻¹ := (Sounio.IntervalFinite.reciprocal j sep).ordered
  simp [reciprocal,ofFinite,reciprocalFinite,hlo,hhi,hs,make,horder,Sounio.IntervalFinite.reciprocal]

theorem div_ofFinite (i j : Sounio.IntervalFinite.Interval)
    (sep : Sounio.IntervalFinite.SeparatedZero j) :
    div (ofFinite i) (ofFinite j) = ofFinite (Sounio.IntervalFinite.div i j sep) := by
  unfold div Sounio.IntervalFinite.div
  rw [reciprocal_ofFinite j sep]
  exact mul_ofFinite i (Sounio.IntervalFinite.reciprocal j sep)

/-- Conditional outward hulls reuse the same imported primitive premises.
    These are not a theorem about the shape of emitted hardware opcodes. -/
def mulRounded (r : Sounio.IntervalFinite.Rounding) (i j : Interval) := outward r (mul i j)
def divRounded (r : Sounio.IntervalFinite.Rounding) (i j : Interval) := outward r (div i j)

theorem mulRounded_contains (r : Sounio.IntervalFinite.Rounding) (i j : Interval)
    (x y : Rat) (hx : Contains i x) (hy : Contains j y) :
    Contains (mulRounded r i j) (x*y) := outward_contains r _ _ (mul_contains i j x y hx hy)

theorem divRounded_contains (r : Sounio.IntervalFinite.Rounding) (i j : Interval)
    (x y : Rat) (hx : Contains i x) (hy : Contains j y) (hne : y ≠ 0) :
    Contains (divRounded r i j) (x/y) := outward_contains r _ _ (div_contains i j x y hx hy hne)

theorem div_decoration_left (a b : Decorated) :
    rank (divDecorated a b).decoration ≤ rank a.decoration := by
  have base := Nat.le_trans (cap_rank (meet a.decoration b.decoration) (div a.interval b.interval))
    (meet_rank_left a.decoration b.decoration)
  by_cases h : containsZero b.interval = true
  · have hn := Nat.le_trans (meet_rank_left (cap (meet a.decoration b.decoration) (div a.interval b.interval)) .trv) base
    simpa [divDecorated,h] using hn
  · simpa [divDecorated,h] using base

theorem div_decoration_right (a b : Decorated) :
    rank (divDecorated a b).decoration ≤ rank b.decoration := by
  have base := Nat.le_trans (cap_rank (meet a.decoration b.decoration) (div a.interval b.interval))
    (meet_rank_right a.decoration b.decoration)
  by_cases h : containsZero b.interval = true
  · have hn := Nat.le_trans (meet_rank_left (cap (meet a.decoration b.decoration) (div a.interval b.interval)) .trv) base
    simpa [divDecorated,h] using hn
  · simpa [divDecorated,h] using base

/-- For non-ill input decorations the zero-point division policy is exactly trv.
    An ill input remains invalid; it is not silently upgraded to a valid set. -/
theorem zero_denominator_decoration (a : Decorated) (db : Decoration)
    (ha : a.decoration ≠ .ill) (hb : db ≠ .ill) :
    (divDecorated a ⟨zeroPoint,db⟩).decoration = .trv := by
  rcases a with ⟨i,da⟩
  have hz : containsZero zeroPoint = true := rfl
  change (if containsZero zeroPoint then meet (cap (meet da db) (div i zeroPoint)) .trv
    else cap (meet da db) (div i zeroPoint)) = .trv
  rw [zero_denominator_empty i,hz]
  change meet (meet (meet da db) .trv) .trv = .trv
  cases da <;> cases db <;> simp_all [meet,rank]

theorem zero_crossing_division_tight (x : Rat) :
    Contains (div zeroPoint (make (some (-2)) (some 3))) x ↔ x=0 := by
  exact zero_times_entire_tight x

end Sounio.IntervalExtended.MulDiv
