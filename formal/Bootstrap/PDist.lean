/-
PDist: finite distributions whose weights are rational intervals (decisions 40
and 41 of the root-A stage-1 design).

The physical layer of the x86 model (silicon faults) is a probability monad:
each step returns a distribution over next states. The weight of an outcome is
not a number but an interval [lo, hi] of rationals, because what we know of the
silicon is itself uncertain: a per-operation fault rate of about 10⁻⁷ is a
calibration with error bars, not an exact value. Sounio's epistemic DNA applied
to the physics.

Core Lean only: `Rat` from the standard library, no Mathlib, no floats.

  - `PD α`: a finite list of outcomes with interval weights.
  - `PD.ok`: every weight is a nonnegative interval, and the total mass, as an
    interval, contains 1.
  - `pure`, `bind`; `pure_ok`, `bind_ok`: the monad keeps `ok`.
  - `flip`: a single-event-upset channel on one bit, with fault probability in
    an interval; `flip_ok`.
  - `faultFree`: the zero-fault branch; `faultFree_bind` and `faultFree_pure`
    connect the monad to the deterministic semantics already proved
    (X86Min, the Sporos reference semantics).
-/
namespace PDist

/-- A rational interval. -/
structure Iv where
  lo : Rat
  hi : Rat
deriving DecidableEq, Repr

/-- A weight: nonnegative, lower end below upper end. -/
def Iv.ok (w : Iv) : Prop := 0 ≤ w.lo ∧ w.lo ≤ w.hi

def Iv.one : Iv := ⟨1, 1⟩

/-- Product of two nonnegative intervals. -/
def Iv.mul (a b : Iv) : Iv := ⟨a.lo * b.lo, a.hi * b.hi⟩

/-- 1 - p, for p inside [0, 1]. -/
def Iv.compl (p : Iv) : Iv := ⟨1 - p.hi, 1 - p.lo⟩

theorem Iv.one_ok : Iv.one.ok := ⟨by decide, by decide⟩

theorem Iv.mul_ok {a b : Iv} (ha : a.ok) (hb : b.ok) : (a.mul b).ok := by
  refine ⟨Rat.mul_nonneg ha.1 hb.1, ?_⟩
  simp only [Iv.mul]
  have h1 : a.lo * b.lo ≤ a.hi * b.lo := Rat.mul_le_mul_of_nonneg_right ha.2 hb.1
  have h2 : a.hi * b.lo ≤ a.hi * b.hi :=
    Rat.mul_le_mul_of_nonneg_left hb.2 (Rat.le_trans ha.1 ha.2)
  exact Rat.le_trans h1 h2

theorem add_le_add' {a b c d : Rat} (h1 : a ≤ b) (h2 : c ≤ d) : a + c ≤ b + d :=
  Rat.le_trans (Rat.add_le_add_right.2 h1) (Rat.add_le_add_left.2 h2)

/-- A finite distribution with interval weights. -/
abbrev PD (α : Type) := List (α × Iv)

def PD.lo {α : Type} (d : PD α) : Rat := (d.map fun x => x.2.lo).foldr (· + ·) 0
def PD.hi {α : Type} (d : PD α) : Rat := (d.map fun x => x.2.hi).foldr (· + ·) 0

/-- Every weight is an interval of nonnegatives, and the mass, as an interval,
contains 1: the true probabilities can sum to 1. -/
def PD.ok {α : Type} (d : PD α) : Prop := (∀ x ∈ d, x.2.ok) ∧ d.lo ≤ 1 ∧ 1 ≤ d.hi

def PD.pure {α : Type} (a : α) : PD α := [(a, Iv.one)]

/-- Scale every weight of `d` by `w`. -/
def PD.scale {α : Type} (w : Iv) (d : PD α) : PD α := d.map fun x => (x.1, w.mul x.2)

def PD.bind {α β : Type} (d : PD α) (f : α → PD β) : PD β :=
  d.flatMap fun x => PD.scale x.2 (f x.1)

theorem PD.pure_ok {α : Type} (a : α) : (PD.pure a).ok := by
  refine ⟨?_, ?_, ?_⟩
  · intro x hx; simp only [PD.pure, List.mem_cons, List.not_mem_nil, or_false] at hx; subst hx; exact Iv.one_ok
  · show (1 : Rat) + 0 ≤ 1; decide +kernel
  · show (1 : Rat) ≤ 1 + 0; decide +kernel

/-! ### Sums -/

def sumR (l : List Rat) : Rat := l.foldr (· + ·) 0

theorem sumR_append (a b : List Rat) : sumR (a ++ b) = sumR a + sumR b := by
  induction a with
  | nil => simp [sumR, Rat.zero_add]
  | cons x r ih =>
    show x + sumR (r ++ b) = (x + sumR r) + sumR b
    rw [ih, Rat.add_assoc]

theorem sumR_mul_left (c : Rat) (l : List Rat) : sumR (l.map (c * ·)) = c * sumR l := by
  induction l with
  | nil => show (0 : Rat) = c * 0; rw [Rat.mul_zero]
  | cons x r ih =>
    show c * x + sumR (r.map (c * ·)) = c * (x + sumR r)
    rw [ih, Rat.mul_add]

theorem sumR_le {f g : α → Rat} (l : List α) (h : ∀ x ∈ l, f x ≤ g x) :
    sumR (l.map f) ≤ sumR (l.map g) := by
  induction l with
  | nil => exact Rat.le_refl
  | cons x r ih =>
    show f x + sumR (r.map f) ≤ g x + sumR (r.map g)
    have h1 := h x (by simp)
    have h2 := ih (fun y hy => h y (by simp [hy]))
    exact add_le_add' h1 h2

theorem PD.lo_eq {α : Type} (d : PD α) : d.lo = sumR (d.map fun x => x.2.lo) := rfl
theorem PD.hi_eq {α : Type} (d : PD α) : d.hi = sumR (d.map fun x => x.2.hi) := rfl

theorem PD.lo_scale {α : Type} (w : Iv) (d : PD α) : (PD.scale w d).lo = w.lo * d.lo := by
  rw [PD.lo_eq, PD.lo_eq, ← sumR_mul_left]
  simp only [PD.scale, List.map_map, Function.comp_def, Iv.mul]

theorem PD.hi_scale {α : Type} (w : Iv) (d : PD α) : (PD.scale w d).hi = w.hi * d.hi := by
  rw [PD.hi_eq, PD.hi_eq, ← sumR_mul_left]
  simp only [PD.scale, List.map_map, Function.comp_def, Iv.mul]

theorem PD.lo_bind {α β : Type} (d : PD α) (f : α → PD β) :
    (d.bind f).lo = sumR (d.map fun x => x.2.lo * (f x.1).lo) := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    simp only [PD.bind, List.flatMap_cons] at ih ⊢
    rw [PD.lo_eq, List.map_append, sumR_append, ← PD.lo_eq, ← PD.lo_eq, ih, PD.lo_scale]
    rfl

theorem PD.hi_bind {α β : Type} (d : PD α) (f : α → PD β) :
    (d.bind f).hi = sumR (d.map fun x => x.2.hi * (f x.1).hi) := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    simp only [PD.bind, List.flatMap_cons] at ih ⊢
    rw [PD.hi_eq, List.map_append, sumR_append, ← PD.hi_eq, ← PD.hi_eq, ih, PD.hi_scale]
    rfl

/-- Binding keeps a distribution well formed: if the true probabilities of `d`
and of every `f a` can sum to 1, so can those of `d.bind f`. -/
theorem PD.bind_ok {α β : Type} {d : PD α} {f : α → PD β} (hd : d.ok) (hf : ∀ a, (f a).ok) :
    (d.bind f).ok := by
  obtain ⟨hw, hlo, hhi⟩ := hd
  refine ⟨?_, ?_, ?_⟩
  · intro y hy
    simp only [PD.bind, List.mem_flatMap, PD.scale, List.mem_map] at hy
    obtain ⟨x, hx, z, hz, rfl⟩ := hy
    exact Iv.mul_ok (hw x hx) ((hf x.1).1 z hz)
  · rw [PD.lo_bind]
    rw [PD.lo_eq] at hlo
    refine Rat.le_trans (sumR_le (g := fun x => x.2.lo) d fun x hx => ?_) hlo
    have := Rat.mul_le_mul_of_nonneg_left (hf x.1).2.1 (hw x hx).1
    rw [Rat.mul_one] at this; exact this
  · rw [PD.hi_bind]
    rw [PD.hi_eq] at hhi
    refine Rat.le_trans hhi (sumR_le (f := fun x => x.2.hi) d fun x hx => ?_)
    have h0 : 0 ≤ x.2.hi := Rat.le_trans (hw x hx).1 (hw x hx).2
    have := Rat.mul_le_mul_of_nonneg_left (hf x.1).2.2 h0
    rw [Rat.mul_one] at this; exact this

/-! ### A single-event upset -/

/-- A fault probability: an interval inside [0, 1]. -/
def Iv.prob (p : Iv) : Prop := 0 ≤ p.lo ∧ p.lo ≤ p.hi ∧ p.hi ≤ 1

/-- The value `x`, with bit `b` inverted with probability in `p`. Each outcome is
tagged: `false` the silicon computed right, `true` it flipped the bit. -/
def flip (p : Iv) (b x : Nat) : PD (Nat × Bool) :=
  [((x, false), p.compl), ((Nat.xor x (2 ^ b), true), p)]

theorem flip_ok {p : Iv} (hp : p.prob) (b x : Nat) : (flip p b x).ok := by
  obtain ⟨h0, h1, h2⟩ := hp
  refine ⟨?_, ?_, ?_⟩
  · intro y hy
    simp only [flip, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl
    · refine ⟨?_, ?_⟩
      · show (0 : Rat) ≤ 1 - p.hi
        exact (Rat.le_iff_sub_nonneg _ _).1 h2
      · show 1 - p.hi ≤ 1 - p.lo
        rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg]
        exact Rat.add_le_add_left.2 (Rat.neg_le_neg h1)
    · exact ⟨h0, h1⟩
  · show (1 - p.hi) + (p.lo + 0) ≤ 1
    rw [Rat.add_zero]
    have : p.lo ≤ p.hi := h1
    calc (1 - p.hi) + p.lo ≤ (1 - p.hi) + p.hi := (Rat.add_le_add_left).2 this
      _ = 1 := by rw [Rat.sub_add_cancel]
  · show 1 ≤ (1 - p.lo) + (p.hi + 0)
    rw [Rat.add_zero]
    calc (1 : Rat) = (1 - p.lo) + p.lo := by rw [Rat.sub_add_cancel]
      _ ≤ (1 - p.lo) + p.hi := (Rat.add_le_add_left).2 h1

/-! ### The zero-fault branch -/

/-- Outcomes tagged with whether a fault occurred on the way. -/
abbrev PDF (α : Type) := PD (α × Bool)

/-- What happens if the silicon makes no fault: the values on fault-free outcomes. -/
def faultFree {α : Type} (d : PDF α) : List α := (d.filter fun x => !x.1.2).map fun x => x.1.1

/-- Sequencing with fault tags: a fault anywhere marks the outcome. -/
def bindF {α β : Type} (d : PDF α) (f : α → PDF β) : PDF β :=
  d.bind fun x => (f x.1).map fun y => ((y.1.1, x.2 || y.1.2), y.2)

def pureF {α : Type} (a : α) : PDF α := PD.pure (a, false)

theorem faultFree_pure {α : Type} (a : α) : faultFree (pureF a) = [a] := rfl

/-- The fault-free branch of a sequence is the sequence of fault-free branches:
the deterministic semantics, read off the probabilistic one. -/
theorem faultFree_bind {α β : Type} (d : PDF α) (f : α → PDF β) :
    faultFree (bindF d f) = (faultFree d).flatMap fun a => faultFree (f a) := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    simp only [bindF, PD.bind, List.flatMap_cons, faultFree, List.filter_append, List.map_append] at ih ⊢
    rw [ih]
    cases t with
    | true =>
      simp [PD.scale, List.filter_map, Function.comp_def]
    | false =>
      simp [PD.scale, List.filter_map, Function.comp_def]

/-! ### Witness -/

/-- One add on 64-bit words, then an upset on bit 3 with probability between
5·10⁻⁸ and 2·10⁻⁷: the mass interval contains 1, and the fault-free branch is
the plain sum. -/
def p7 : Iv := ⟨1 / 20000000, 1 / 5000000⟩

theorem witness :
    (flip p7 3 (5 + 7)).lo ≤ 1 ∧ 1 ≤ (flip p7 3 (5 + 7)).hi ∧ faultFree (flip p7 3 (5 + 7)) = [12] := by
  refine ⟨?_, ?_, rfl⟩ <;> decide +kernel

end PDist
