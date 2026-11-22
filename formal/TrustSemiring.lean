/-!
# Trust semiring for provenance in Sounio (no Mathlib)

A provenance annotation records, for each trust level, the fewest inference
steps of any derivation reaching *at least* that level. `none` = no such
derivation (+∞). This is the Pareto frontier of (level, steps) over the
three-level chain signed > inferred > assumed, and it is the product of three
tropical semirings (min, +) on `Option Nat`.

Also recorded: the lexicographic "pick the best derivation" design is NOT a
semiring (distributivity fails); witness at the end.
-/

namespace Trust

/-- Tropical value: `some n` = n steps, `none` = +∞ (no derivation). -/
abbrev T := Option Nat

/-- ⊕ : best alternative (fewest steps). -/
def tmin : T → T → T
  | none, b => b
  | a, none => a
  | some a, some b => some (Nat.min a b)

/-- ⊗ : joint use (steps add; +∞ absorbs). -/
def tadd : T → T → T
  | some a, some b => some (a + b)
  | _, _ => none

theorem tmin_comm (a b : T) : tmin a b = tmin b a := by
  cases a <;> cases b <;> simp [tmin, Nat.min_comm]

theorem tmin_assoc (a b c : T) : tmin (tmin a b) c = tmin a (tmin b c) := by
  cases a <;> cases b <;> cases c <;> simp [tmin, Nat.min_assoc]

theorem tmin_zero (a : T) : tmin none a = a := by cases a <;> rfl

theorem tmin_idem (a : T) : tmin a a = a := by
  cases a <;> simp [tmin]

theorem tadd_comm (a b : T) : tadd a b = tadd b a := by
  cases a <;> cases b <;> simp [tadd, Nat.add_comm]

theorem tadd_assoc (a b c : T) : tadd (tadd a b) c = tadd a (tadd b c) := by
  cases a <;> cases b <;> cases c <;> simp [tadd, Nat.add_assoc]

theorem tadd_one (a : T) : tadd (some 0) a = a := by
  cases a <;> simp [tadd]

theorem tadd_annihilate (a : T) : tadd none a = none := by cases a <;> rfl

theorem tadd_distrib (a b c : T) :
    tadd a (tmin b c) = tmin (tadd a b) (tadd a c) := by
  cases a <;> cases b <;> cases c <;> simp [tadd, tmin] <;> omega

/-- Absorption (0-stability): the empty derivation dominates everything.
    This is the property under which Datalog fixpoints over the semiring converge. -/
theorem tmin_absorb (a : T) : tmin (some 0) a = some 0 := by
  cases a <;> simp [tmin]

/-- Steps order: `tle x y` iff x needs at most as many steps as y. -/
def tle : T → T → Prop
  | _, none => True
  | none, some _ => False
  | some a, some b => a ≤ b

theorem tmin_mono {a b c d : T} (h1 : tle a b) (h2 : tle c d) :
    tle (tmin a c) (tmin b d) := by
  cases a <;> cases b <;> cases c <;> cases d <;> simp_all [tle, tmin, Nat.min_def] <;>
    (repeat' split) <;> omega

theorem tadd_mono {a b c d : T} (h1 : tle a b) (h2 : tle c d) :
    tle (tadd a c) (tadd b d) := by
  cases a <;> cases b <;> cases c <;> cases d <;> simp_all [tle, tadd] <;> omega

/-- Annotation: fewest steps for a derivation of level ≥ signed / ≥ inferred / ≥ assumed. -/
structure Ann where
  s : T
  i : T
  a : T
deriving DecidableEq

/-- Well-formed: a derivation at a higher level is also at every lower level. -/
def Ann.wf (x : Ann) : Prop := tle x.i x.s ∧ tle x.a x.i

def plus (x y : Ann) : Ann := ⟨tmin x.s y.s, tmin x.i y.i, tmin x.a y.a⟩
def times (x y : Ann) : Ann := ⟨tadd x.s y.s, tadd x.i y.i, tadd x.a y.a⟩
def zero : Ann := ⟨none, none, none⟩
/-- A signed fact used directly: 0 steps at every level. -/
def one : Ann := ⟨some 0, some 0, some 0⟩

theorem plus_comm (x y : Ann) : plus x y = plus y x := by
  simp [plus, tmin_comm]
theorem plus_assoc (x y z : Ann) : plus (plus x y) z = plus x (plus y z) := by
  simp [plus, tmin_assoc]
theorem plus_zero (x : Ann) : plus zero x = x := by
  simp [plus, zero, tmin_zero]
theorem plus_idem (x : Ann) : plus x x = x := by
  simp [plus, tmin_idem]
theorem times_comm (x y : Ann) : times x y = times y x := by
  simp [times, tadd_comm]
theorem times_assoc (x y z : Ann) : times (times x y) z = times x (times y z) := by
  simp [times, tadd_assoc]
theorem times_one (x : Ann) : times one x = x := by
  simp [times, one, tadd_one]
theorem times_zero (x : Ann) : times zero x = zero := by
  simp [times, zero, tadd_annihilate]
theorem distrib (x y z : Ann) : times x (plus y z) = plus (times x y) (times x z) := by
  simp [times, plus, tadd_distrib]
theorem absorb (x : Ann) : plus one x = one := by
  simp [plus, one, tmin_absorb]

theorem plus_wf {x y : Ann} (hx : x.wf) (hy : y.wf) : (plus x y).wf :=
  ⟨tmin_mono hx.1 hy.1, tmin_mono hx.2 hy.2⟩
theorem times_wf {x y : Ann} (hx : x.wf) (hy : y.wf) : (times x y).wf :=
  ⟨tadd_mono hx.1 hy.1, tadd_mono hx.2 hy.2⟩

/-! ## The rejected design: lexicographic "best derivation" -/

/-- Levels as 2 > 1 > 0 (signed > inferred > assumed). -/
def lexPlus (x y : Nat × Nat) : Nat × Nat :=
  if x.1 > y.1 then x else if y.1 > x.1 then y
  else if x.2 ≤ y.2 then x else y
def lexTimes (x y : Nat × Nat) : Nat × Nat := (Nat.min x.1 y.1, x.2 + y.2)

/-- Counterexample: b = (signed, 100), c = (inferred, 0), a = (assumed, 0). -/
theorem lex_not_distrib :
    lexTimes (0, 0) (lexPlus (2, 100) (1, 0))
      ≠ lexPlus (lexTimes (0, 0) (2, 100)) (lexTimes (0, 0) (1, 0)) := by
  decide

end Trust
