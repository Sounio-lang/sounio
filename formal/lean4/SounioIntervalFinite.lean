import Std

/-!
Finite, nonempty exact-rational enclosure increment for ADR-019/SIO-178.
No IEEE bit semantics, infinities, empty encoding, decorations or hardware
rounding claim is made. Rounding carries primitive scalar inequalities as
explicit parameters; their binding to actual RD/RU instructions is NOT ESTABLISHED.
-/
namespace Sounio.IntervalFinite
open Std

/-- Only proper finite rational intervals inhabit this first proof frame. -/
structure Interval where
  lo : Rat
  hi : Rat
  ordered : lo ≤ hi

def Contains (i : Interval) (x : Rat) : Prop := i.lo ≤ x ∧ x ≤ i.hi
def low4 (a b c d : Rat) : Rat := min (min a b) (min c d)
def high4 (a b c d : Rat) : Rat := max (max a b) (max c d)

/-- Reversing a scalar inequality needs the scalar's nonpositive sign. -/
theorem mul_le_of_nonpos (a b c : Rat) (ha : a ≤ 0) (hbc : b ≤ c) :
    a*c ≤ a*b := by
  have hn : 0 ≤ -a := by grind
  have hm := Rat.mul_le_mul_of_nonneg_left hbc hn
  grind

/-- A signed scalar maps a finite segment inside its two endpoint products. -/
theorem left_max (a b : Rat) : a ≤ max a b := by grind
theorem right_max (a b : Rat) : b ≤ max a b := by grind

theorem scale_bounds (a c d y : Rat) (hcy : c ≤ y) (hyd : y ≤ d) :
    min (a*c) (a*d) ≤ a*y ∧ a*y ≤ max (a*c) (a*d) := by
  by_cases ha : 0 ≤ a
  · have h1 := Rat.mul_le_mul_of_nonneg_left hcy ha
    have h2 := Rat.mul_le_mul_of_nonneg_left hyd ha
    exact ⟨Rat.le_trans min_le_left h1, Rat.le_trans h2 (right_max _ _)⟩
  · have hn : a ≤ 0 := by grind
    have h1 := mul_le_of_nonpos a y d hn hyd
    have h2 := mul_le_of_nonpos a c y hn hcy
    exact ⟨Rat.le_trans min_le_right h1, Rat.le_trans h2 (left_max _ _)⟩

/-- All signs, including intervals straddling zero: four actual corners suffice. -/
theorem product_corners (a b c d x y : Rat)
    (hax : a ≤ x) (hxb : x ≤ b) (hcy : c ≤ y) (hyd : y ≤ d) :
    low4 (a*c) (a*d) (b*c) (b*d) ≤ x*y ∧
    x*y ≤ high4 (a*c) (a*d) (b*c) (b*d) := by
  have ha := scale_bounds a c d y hcy hyd
  have hb := scale_bounds b c d y hcy hyd
  by_cases hy : 0 ≤ y
  · have h1 := Rat.mul_le_mul_of_nonneg_right hax hy
    have h2 := Rat.mul_le_mul_of_nonneg_right hxb hy
    exact ⟨Rat.le_trans (Rat.le_trans min_le_left ha.1) h1,
      Rat.le_trans h2 (Rat.le_trans hb.2 (right_max _ _))⟩
  · have hn : y ≤ 0 := by grind
    have h1 : b*y ≤ x*y := by
      simpa only [Rat.mul_comm] using mul_le_of_nonpos y x b hn hxb
    have h2 : x*y ≤ a*y := by
      simpa only [Rat.mul_comm] using mul_le_of_nonpos y a x hn hax
    exact ⟨Rat.le_trans (Rat.le_trans min_le_right hb.1) h1,
      Rat.le_trans h2 (Rat.le_trans ha.2 (left_max _ _))⟩

def add (i j : Interval) : Interval :=
  ⟨i.lo+j.lo, i.hi+j.hi, by have := i.ordered; have := j.ordered; grind⟩
def sub (i j : Interval) : Interval :=
  ⟨i.lo-j.hi, i.hi-j.lo, by have := i.ordered; have := j.ordered; grind⟩
def mul (i j : Interval) : Interval :=
  let h := product_corners i.lo i.hi j.lo j.hi i.lo j.lo
    (Rat.le_refl) i.ordered (Rat.le_refl) j.ordered
  ⟨low4 (i.lo*j.lo) (i.lo*j.hi) (i.hi*j.lo) (i.hi*j.hi),
   high4 (i.lo*j.lo) (i.lo*j.hi) (i.hi*j.lo) (i.hi*j.hi),
   Rat.le_trans h.1 h.2⟩

theorem add_contains (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (add i j) (x+y) := by
  rcases hx with ⟨hx1,hx2⟩; rcases hy with ⟨hy1,hy2⟩
  constructor <;> simp only [add] at * <;> grind

theorem sub_contains (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (sub i j) (x-y) := by
  rcases hx with ⟨hx1,hx2⟩; rcases hy with ⟨hy1,hy2⟩
  constructor <;> simp only [sub] at * <;> grind

theorem mul_contains (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (mul i j) (x*y) :=
  product_corners i.lo i.hi j.lo j.hi x y hx.1 hx.2 hy.1 hy.2

/-- A real division domain condition, not an assumed enclosure conclusion. -/
def SeparatedZero (j : Interval) : Prop := 0 < j.lo ∨ j.hi < 0

theorem denominator_member_ne_zero (j : Interval) (y : Rat)
    (hy : Contains j y) (sep : SeparatedZero j) : y ≠ 0 := by
  rcases sep with hp | hn <;> grind [Contains]

theorem negative_product_pos (a b : Rat) (ha : a < 0) (hb : b < 0) :
    0 < a*b := by
  have h1 : 0 < -a := by grind
  have h2 : 0 < -b := by grind
  have h3 := Rat.mul_pos h1 h2
  grind

/-- Reciprocal reverses order when the product is positive and neither pole is zero. -/
theorem inverse_order (a b : Rat) (ha : a ≠ 0) (hb : b ≠ 0)
    (hab : a ≤ b) (hp : 0 < a*b) : b⁻¹ ≤ a⁻¹ := by
  apply Rat.le_of_mul_le_mul_right (c := a*b) (hc := hp)
  calc
    b⁻¹*(a*b) = a := by grind
    _ ≤ b := hab
    _ = a⁻¹*(a*b) := by grind

theorem reciprocal_bounds (j : Interval) (y : Rat)
    (hy : Contains j y) (sep : SeparatedZero j) :
    j.hi⁻¹ ≤ y⁻¹ ∧ y⁻¹ ≤ j.lo⁻¹ := by
  rcases sep with hp | hn
  · have hypos : 0 < y := by grind [Contains]
    have hhipos : 0 < j.hi := by grind [Contains]
    exact ⟨inverse_order y j.hi (Rat.ne_of_gt hypos) (Rat.ne_of_gt hhipos)
        hy.2 (Rat.mul_pos hypos hhipos),
      inverse_order j.lo y (Rat.ne_of_gt hp) (Rat.ne_of_gt hypos)
        hy.1 (Rat.mul_pos hp hypos)⟩
  · have hyneg : y < 0 := by grind [Contains]
    have hloneg : j.lo < 0 := by grind [Contains]
    exact ⟨inverse_order y j.hi (Rat.ne_of_lt hyneg) (Rat.ne_of_lt hn)
        hy.2 (negative_product_pos y j.hi hyneg hn),
      inverse_order j.lo y (Rat.ne_of_lt hloneg) (Rat.ne_of_lt hyneg)
        hy.1 (negative_product_pos j.lo y hloneg hyneg)⟩

def reciprocal (j : Interval) (sep : SeparatedZero j) : Interval :=
  ⟨j.hi⁻¹, j.lo⁻¹,
   (reciprocal_bounds j j.lo ⟨Rat.le_refl,j.ordered⟩ sep).1⟩

/-- Four quotient corners, denominator endpoints in reverse order through reciprocal. -/
def div (i j : Interval) (sep : SeparatedZero j) : Interval := mul i (reciprocal j sep)

theorem div_low_corners (i j : Interval) (sep : SeparatedZero j) :
    (div i j sep).lo =
      low4 (i.lo/j.hi) (i.lo/j.lo) (i.hi/j.hi) (i.hi/j.lo) := rfl

theorem div_high_corners (i j : Interval) (sep : SeparatedZero j) :
    (div i j sep).hi =
      high4 (i.lo/j.hi) (i.lo/j.lo) (i.hi/j.hi) (i.hi/j.lo) := rfl

theorem div_contains (i j : Interval) (sep : SeparatedZero j) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (div i j sep) (x/y) := by
  exact mul_contains i (reciprocal j sep) x y⁻¹ hx (reciprocal_bounds j y hy sep)

/-- Abstract finite-carrier maps. Fields are explicit primitive premises, NOT
    established IEEE/hardware facts. No global rounding axiom is declared. -/
structure Rounding where
  rd : Rat → Rat
  ru : Rat → Rat
  rd_le : ∀ q : Rat, rd q ≤ q
  le_ru : ∀ q : Rat, q ≤ ru q

def outward (r : Rounding) (i : Interval) : Interval :=
  ⟨r.rd i.lo, r.ru i.hi, Rat.le_trans (r.rd_le _) (Rat.le_trans i.ordered (r.le_ru _))⟩

theorem outward_contains (r : Rounding) (i : Interval) (x : Rat)
    (hx : Contains i x) : Contains (outward r i) x :=
  ⟨Rat.le_trans (r.rd_le _) hx.1, Rat.le_trans hx.2 (r.le_ru _)⟩

/-- Componentwise min/max order uses no monotonicity premise about rounding maps. -/
theorem min_mono (a b c d : Rat) (ha : a ≤ c) (hb : b ≤ d) :
    min a b ≤ min c d := by grind
theorem max_mono (a b c d : Rat) (ha : a ≤ c) (hb : b ≤ d) :
    max a b ≤ max c d := by grind

/-- Each scalar corner is rounded separately, matching the primitive operation
    shape while leaving its actual IEEE/hardware binding unestablished. -/
def roundedCorners (r : Rounding) (a b c d : Rat) : Interval :=
  ⟨low4 (r.rd a) (r.rd b) (r.rd c) (r.rd d),
   high4 (r.ru a) (r.ru b) (r.ru c) (r.ru d),
   Rat.le_trans (Rat.le_trans min_le_left min_le_left)
    (Rat.le_trans (r.rd_le a)
     (Rat.le_trans (r.le_ru a)
      (Rat.le_trans (left_max _ _) (left_max _ _))))⟩

theorem roundedCorners_contains (r : Rounding) (a b c d x : Rat)
    (hx : low4 a b c d ≤ x ∧ x ≤ high4 a b c d) :
    Contains (roundedCorners r a b c d) x := by
  have h1 := min_mono (r.rd a) (r.rd b) a b (r.rd_le a) (r.rd_le b)
  have h2 := min_mono (r.rd c) (r.rd d) c d (r.rd_le c) (r.rd_le d)
  have hlo := min_mono (min (r.rd a) (r.rd b)) (min (r.rd c) (r.rd d))
    (min a b) (min c d) h1 h2
  have h3 := max_mono a b (r.ru a) (r.ru b) (r.le_ru a) (r.le_ru b)
  have h4 := max_mono c d (r.ru c) (r.ru d) (r.le_ru c) (r.le_ru d)
  have hhi := max_mono (max a b) (max c d)
    (max (r.ru a) (r.ru b)) (max (r.ru c) (r.ru d)) h3 h4
  exact ⟨Rat.le_trans hlo hx.1,Rat.le_trans hx.2 hhi⟩

def addRounded (r : Rounding) (i j : Interval) := outward r (add i j)
def subRounded (r : Rounding) (i j : Interval) := outward r (sub i j)
def mulRounded (r : Rounding) (i j : Interval) :=
  roundedCorners r (i.lo*j.lo) (i.lo*j.hi) (i.hi*j.lo) (i.hi*j.hi)
def divRounded (r : Rounding) (i j : Interval) (sep : SeparatedZero j) :=
  mulRounded r i (reciprocal j sep)

theorem addRounded_contains (r : Rounding) (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (addRounded r i j) (x+y) :=
  outward_contains r _ _ (add_contains i j x y hx hy)
theorem subRounded_contains (r : Rounding) (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (subRounded r i j) (x-y) :=
  outward_contains r _ _ (sub_contains i j x y hx hy)
theorem mulRounded_contains (r : Rounding) (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (mulRounded r i j) (x*y) :=
  roundedCorners_contains r _ _ _ _ _ (mul_contains i j x y hx hy)
theorem divRounded_contains (r : Rounding) (i j : Interval) (sep : SeparatedZero j)
    (x y : Rat) (hx : Contains i x) (hy : Contains j y) :
    Contains (divRounded r i j sep) (x/y) :=
  mulRounded_contains r i (reciprocal j sep) x y⁻¹ hx (reciprocal_bounds j y hy sep)

theorem mulRounded_low_corners (r : Rounding) (i j : Interval) :
    (mulRounded r i j).lo =
      low4 (r.rd (i.lo*j.lo)) (r.rd (i.lo*j.hi))
        (r.rd (i.hi*j.lo)) (r.rd (i.hi*j.hi)) := rfl

theorem divRounded_low_corners (r : Rounding) (i j : Interval) (sep : SeparatedZero j) :
    (divRounded r i j sep).lo =
      low4 (r.rd (i.lo/j.hi)) (r.rd (i.lo/j.lo))
        (r.rd (i.hi/j.hi)) (r.rd (i.hi/j.lo)) := rfl

/-- An exact mathematical model witnesses consistency of the primitive premises.
    It is not an implementation of an IEEE directed-rounding instruction. -/
def exactRounding : Rounding := ⟨id,id,fun _ => Rat.le_refl,fun _ => Rat.le_refl⟩

def signedNumerator : Interval := ⟨-3,1,by decide⟩
def signedMultiplier : Interval := ⟨-4,2,by decide⟩
def negativeDenominator : Interval := ⟨-4,-2,by decide⟩
def positiveDenominator : Interval := ⟨2,4,by decide⟩
def crossingDenominator : Interval := ⟨-1,1,by decide⟩
def touchingDenominator : Interval := ⟨0,2,by decide⟩

theorem negative_separated : SeparatedZero negativeDenominator := by
  unfold SeparatedZero negativeDenominator
  decide +kernel
theorem positive_separated : SeparatedZero positiveDenominator := by
  unfold SeparatedZero positiveDenominator
  decide +kernel

/-- Closed rational checks are reduced by the kernel, not native_decide. -/
theorem signed_product_endpoints :
    (mul signedNumerator signedMultiplier).lo = -6 ∧
    (mul signedNumerator signedMultiplier).hi = 12 := by
  unfold mul signedNumerator signedMultiplier low4 high4
  decide +kernel
theorem negative_division_endpoints :
    (div signedNumerator negativeDenominator negative_separated).lo = -1/2 ∧
    (div signedNumerator negativeDenominator negative_separated).hi = 3/2 := by
  unfold div mul reciprocal signedNumerator negativeDenominator low4 high4
  decide +kernel
theorem positive_division_endpoints :
    (div signedNumerator positiveDenominator positive_separated).lo = -3/2 ∧
    (div signedNumerator positiveDenominator positive_separated).hi = 1/2 := by
  unfold div mul reciprocal signedNumerator positiveDenominator low4 high4
  decide +kernel
theorem crossing_is_outside_frame : ¬ SeparatedZero crossingDenominator := by
  unfold SeparatedZero crossingDenominator
  decide +kernel
theorem touching_is_outside_frame : ¬ SeparatedZero touchingDenominator := by
  unfold SeparatedZero touchingDenominator
  decide +kernel

end Sounio.IntervalFinite
