import SounioIntervalFinite

/-!
ADR-019 exact rational set model, extended by infinite endpoints.
This increment proves addition/subtraction for finite, one-sided unbounded,
entire and empty intervals. General multiplication/division and IEEE bindings
remain outside this increment; finite arithmetic is the imported canonical model.
An infinite endpoint is not a real singleton member. No floating-point encoding,
signed-zero representation or hardware rounding guarantee follows from this file.
-/
namespace Sounio.IntervalExtended
open Std

inductive ExtendedRat where
  | negInf
  | finite (q : Rat)
  | posInf
  deriving Repr, DecidableEq

/-- none is -infinity on a lower bound, +infinity on an upper bound. -/
def Lower (b : Option Rat) (x : Rat) : Prop :=
  match b with | none => True | some q => q ≤ x

def Upper (b : Option Rat) (x : Rat) : Prop :=
  match b with | none => True | some q => x ≤ q

def Proper (lo hi : Option Rat) : Prop :=
  ∀ a b : Rat, lo = some a → hi = some b → a ≤ b

/-- Only proper spans can be constructed. Empty is a distinct set value. -/
inductive Interval where
  | empty
  | span (lo hi : Option Rat) (proper : Proper lo hi)

/-- The interval denotes finite real/rational members, never infinite members. -/
def Contains (i : Interval) (x : Rat) : Prop :=
  match i with | .empty => False | .span lo hi _ => Lower lo x ∧ Upper hi x

def ContainsExtended (i : Interval) (x : ExtendedRat) : Prop :=
  match x with | .finite q => Contains i q | _ => False

def lowerEndpoint : Interval → ExtendedRat
  | .empty => .posInf
  | .span none _ _ => .negInf
  | .span (some q) _ _ => .finite q

def upperEndpoint : Interval → ExtendedRat
  | .empty => .negInf
  | .span _ none _ => .posInf
  | .span _ (some q) _ => .finite q

/-- Constructor for computed bounds. Reversed finite bounds denote empty. -/
def make (lo hi : Option Rat) : Interval :=
  match lo, hi with
  | some a, some b =>
    if h : a ≤ b then .span (some a) (some b) (by
      intro x y hx hy; cases hx; cases hy; exact h)
    else .empty
  | none, some b => .span none (some b) (by intro x y hx; cases hx)
  | some a, none => .span (some a) none (by intro x y _ hy; cases hy)
  | none, none => .span none none (by intro x y hx; cases hx)

def entire : Interval := make none none

def ofFinite (i : Sounio.IntervalFinite.Interval) : Interval :=
  .span (some i.lo) (some i.hi) (by
    intro x y hx hy; cases hx; cases hy; exact i.ordered)

def boundAdd (a b : Option Rat) : Option Rat :=
  match a,b with | some x,some y => some (x+y) | _,_ => none

def boundNeg (a : Option Rat) : Option Rat := a.map (fun x => -x)

def add : Interval → Interval → Interval
  | .empty,_ => .empty
  | _,.empty => .empty
  | .span a b _,.span c d _ => make (boundAdd a c) (boundAdd b d)

def neg : Interval → Interval
  | .empty => .empty
  | .span a b _ => make (boundNeg b) (boundNeg a)

def sub (i j : Interval) : Interval := add i (neg j)

theorem make_contains (lo hi : Option Rat) (x : Rat)
    (hx : Lower lo x ∧ Upper hi x) : Contains (make lo hi) x := by
  cases lo with
  | none => cases hi <;> simpa [make, Contains, Lower, Upper] using hx
  | some a =>
    cases hi with
    | none => simpa [make, Contains, Lower, Upper] using hx
    | some b =>
      have hab : a ≤ b := Rat.le_trans hx.1 hx.2
      simpa [make, hab, Contains] using hx

theorem lower_add (a b : Option Rat) (x y : Rat)
    (hx : Lower a x) (hy : Lower b y) : Lower (boundAdd a b) (x+y) := by
  cases a <;> cases b <;> simp_all [Lower, boundAdd] <;> grind

theorem upper_add (a b : Option Rat) (x y : Rat)
    (hx : Upper a x) (hy : Upper b y) : Upper (boundAdd a b) (x+y) := by
  cases a <;> cases b <;> simp_all [Upper, boundAdd] <;> grind

theorem lower_neg (b : Option Rat) (x : Rat)
    (hx : Upper b x) : Lower (boundNeg b) (-x) := by
  cases b <;> simp_all [boundNeg, Lower, Upper] <;> grind

theorem upper_neg (a : Option Rat) (x : Rat)
    (hx : Lower a x) : Upper (boundNeg a) (-x) := by
  cases a <;> simp_all [boundNeg, Lower, Upper] <;> grind

theorem add_contains (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (add i j) (x+y) := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp =>
    cases j with
    | empty => exact False.elim hy
    | span c d hq =>
      exact make_contains _ _ _ ⟨lower_add a c x y hx.1 hy.1,
        upper_add b d x y hx.2 hy.2⟩

theorem neg_contains (i : Interval) (x : Rat)
    (hx : Contains i x) : Contains (neg i) (-x) := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp => exact make_contains _ _ _ ⟨lower_neg b x hx.2,upper_neg a x hx.1⟩

theorem sub_contains (i j : Interval) (x y : Rat)
    (hx : Contains i x) (hy : Contains j y) : Contains (sub i j) (x-y) := by
  have h := add_contains i (neg j) x (-y) hx (neg_contains j y hy)
  simpa [sub, Rat.sub_eq_add_neg] using h

theorem span_has_member (a b : Option Rat) (hp : Proper a b) :
    ∃ x : Rat, Contains (.span a b hp) x := by
  cases a with
  | none =>
    cases b with
    | none => exact ⟨0, trivial, trivial⟩
    | some q => exact ⟨q, trivial, Rat.le_refl⟩
  | some p =>
    cases b with
    | none => exact ⟨p, Rat.le_refl, trivial⟩
    | some q => exact ⟨p, Rat.le_refl, hp p q rfl rfl⟩

theorem empty_iff_no_member (i : Interval) : i = .empty ↔ ¬ ∃ x, Contains i x := by
  cases i with
  | empty => simp [Contains]
  | span a b hp =>
    constructor
    · intro h; cases h
    · intro h; exact False.elim (h (span_has_member a b hp))

theorem nonempty_add (i j : Interval) (hi : i ≠ .empty) (hj : j ≠ .empty) :
    add i j ≠ .empty := by
  have hi' : ∃ x, Contains i x := by
    cases i with
    | empty => exact False.elim (hi rfl)
    | span a b hp => exact span_has_member a b hp
  have hj' : ∃ y, Contains j y := by
    cases j with
    | empty => exact False.elim (hj rfl)
    | span a b hp => exact span_has_member a b hp
  obtain ⟨x,hx⟩ := hi'; obtain ⟨y,hy⟩ := hj'
  intro hempty
  have hm := add_contains i j x y hx hy
  rw [hempty] at hm
  exact hm

theorem neg_empty : neg .empty = .empty := rfl

theorem add_empty_left (i : Interval) : add .empty i = .empty := rfl

theorem add_empty_right (i : Interval) : add i .empty = .empty := by cases i <;> rfl

theorem sub_empty_left (i : Interval) : sub .empty i = .empty := rfl

theorem sub_empty_right (i : Interval) : sub i .empty = .empty := add_empty_right i

theorem entire_contains (x : Rat) : Contains entire x := ⟨trivial,trivial⟩

theorem empty_has_no_member (x : Rat) : ¬ Contains .empty x := by exact id

theorem negative_infinity_not_member (i : Interval) : ¬ ContainsExtended i .negInf := by exact id

theorem positive_infinity_not_member (i : Interval) : ¬ ContainsExtended i .posInf := by exact id

theorem empty_endpoints : lowerEndpoint .empty = .posInf ∧
    upperEndpoint .empty = .negInf := ⟨rfl,rfl⟩

theorem entire_endpoints : lowerEndpoint entire = .negInf ∧
    upperEndpoint entire = .posInf := ⟨rfl,rfl⟩

theorem ofFinite_contains (i : Sounio.IntervalFinite.Interval) (x : Rat) :
    Contains (ofFinite i) x ↔ Sounio.IntervalFinite.Contains i x := Iff.rfl

/-- The same exact finite engine is the concordance reference for bounded inputs. -/
theorem finite_add_concordance (i j : Sounio.IntervalFinite.Interval) (x : Rat) :
    Contains (add (ofFinite i) (ofFinite j)) x ↔
    Sounio.IntervalFinite.Contains (Sounio.IntervalFinite.add i j) x := by
  have h : i.lo+j.lo ≤ i.hi+j.hi := by have := i.ordered; have := j.ordered; grind
  simp [add,ofFinite,boundAdd,make,h,Contains,Lower,Upper,
    Sounio.IntervalFinite.Contains,Sounio.IntervalFinite.add]

/-- Uses only the imported finite model's explicit primitive premises. The
    connection of these maps to IEEE directed instructions is NOT ESTABLISHED. -/
def outward (r : Sounio.IntervalFinite.Rounding) : Interval → Interval
  | .empty => .empty
  | .span a b _ => make (a.map r.rd) (b.map r.ru)

theorem outward_contains (r : Sounio.IntervalFinite.Rounding) (i : Interval) (x : Rat)
    (hx : Contains i x) : Contains (outward r i) x := by
  cases i with
  | empty => exact False.elim hx
  | span a b hp =>
    apply make_contains
    constructor
    · cases a with
      | none => trivial
      | some q => exact Rat.le_trans (r.rd_le q) hx.1
    · cases b with
      | none => trivial
      | some q => exact Rat.le_trans hx.2 (r.le_ru q)

def addRounded (r : Sounio.IntervalFinite.Rounding) (i j : Interval) := outward r (add i j)
def subRounded (r : Sounio.IntervalFinite.Rounding) (i j : Interval) := outward r (sub i j)

theorem addRounded_contains (r : Sounio.IntervalFinite.Rounding) (i j : Interval)
    (x y : Rat) (hx : Contains i x) (hy : Contains j y) :
    Contains (addRounded r i j) (x+y) :=
  outward_contains r _ _ (add_contains i j x y hx hy)

theorem subRounded_contains (r : Sounio.IntervalFinite.Rounding) (i j : Interval)
    (x y : Rat) (hx : Contains i x) (hy : Contains j y) :
    Contains (subRounded r i j) (x-y) :=
  outward_contains r _ _ (sub_contains i j x y hx hy)

/-- Decoration order: weaker information has the smaller rank. -/
inductive Decoration where
  | ill | trv | def | dac | com
  deriving Repr, DecidableEq

def rank : Decoration → Nat
  | .ill => 0 | .trv => 1 | .def => 2 | .dac => 3 | .com => 4

def meet (a b : Decoration) : Decoration := if rank a ≤ rank b then a else b

def bounded : Interval → Bool
  | .span (some _) (some _) _ => true
  | _ => false

/-- Empty has at most trv; an unbounded nonempty span has at most dac. -/
def cap (d : Decoration) : Interval → Decoration
  | .empty => meet d .trv
  | i@(.span _ _ _) => if bounded i then d else meet d .dac

structure Decorated where
  interval : Interval
  decoration : Decoration

def addDecorated (a b : Decorated) : Decorated :=
  let i := add a.interval b.interval
  ⟨i,cap (meet a.decoration b.decoration) i⟩

def subDecorated (a b : Decorated) : Decorated :=
  let i := sub a.interval b.interval
  ⟨i,cap (meet a.decoration b.decoration) i⟩

theorem meet_rank_left (a b : Decoration) : rank (meet a b) ≤ rank a := by
  unfold meet; split <;> simp_all <;> omega

theorem meet_rank_right (a b : Decoration) : rank (meet a b) ≤ rank b := by
  unfold meet; split <;> simp_all

theorem cap_rank (d : Decoration) (i : Interval) : rank (cap d i) ≤ rank d := by
  cases i with
  | empty => exact meet_rank_left _ _
  | span lo hi hp =>
    by_cases h : bounded (.span lo hi hp) = true
    · simp [cap,h]
    · simpa [cap,h] using meet_rank_left d .dac

theorem cap_empty_rank (d : Decoration) : rank (cap d .empty) ≤ rank .trv := meet_rank_right _ _

theorem add_decoration_left (a b : Decorated) :
    rank (addDecorated a b).decoration ≤ rank a.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_left _ _)

theorem add_decoration_right (a b : Decorated) :
    rank (addDecorated a b).decoration ≤ rank b.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_right _ _)

theorem sub_decoration_left (a b : Decorated) :
    rank (subDecorated a b).decoration ≤ rank a.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_left _ _)

theorem sub_decoration_right (a b : Decorated) :
    rank (subDecorated a b).decoration ≤ rank b.decoration :=
  Nat.le_trans (cap_rank _ _) (meet_rank_right _ _)

def positiveRay (q : Rat) : Interval := make (some q) none

def negativeRay (q : Rat) : Interval := make none (some q)

theorem positive_ray_sum : lowerEndpoint (add (positiveRay 2) (positiveRay 3)) = .finite 5 ∧
    upperEndpoint (add (positiveRay 2) (positiveRay 3)) = .posInf := by
  decide +kernel

theorem negative_ray_sum : lowerEndpoint (add (negativeRay (-2)) (negativeRay (-3))) = .negInf ∧
    upperEndpoint (add (negativeRay (-2)) (negativeRay (-3))) = .finite (-5) := by
  decide +kernel

theorem opposite_rays_sum : lowerEndpoint (add (positiveRay 2) (negativeRay 3)) = .negInf ∧
    upperEndpoint (add (positiveRay 2) (negativeRay 3)) = .posInf := by
  decide +kernel

theorem same_side_ray_difference : lowerEndpoint (sub (positiveRay 2) (negativeRay 3)) = .finite (-1) ∧
    upperEndpoint (sub (positiveRay 2) (negativeRay 3)) = .posInf := by
  decide +kernel

theorem reversed_finite_empty : make (some 2) (some 1) = .empty := by
  simp [make]; decide +kernel

end Sounio.IntervalExtended
