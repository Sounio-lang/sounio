/-
PDist: finite distributions with exact rational weights, for the physical layer
of the x86 model (decisions 40 and 41 of the root-A stage-1 design, and the
correction to 41 of 2026-11-16).

The silicon's fault rate is not known exactly: a per-operation rate of about
10⁻⁷ is a calibration with error bars. That uncertainty is an interval
[lo, hi] on the rate, not on each weight. Every weight here is an exact
rational, a function of the rate p, and every bound is proved for all p in the
interval. An interval on each weight would widen at every step: after n steps
the mass lies in [(1 − w)ⁿ, (1 + w)ⁿ], which for w = 1.5·10⁻⁷ and n = 10⁹ is
vacuous. With the interval on the parameter, a bound linear in n stays linear.

Core Lean only: `Rat` from the standard library, no Mathlib, no floats.

  - `PD α`: a finite list of outcomes with rational weights; `PD.ok`: the
    weights are nonnegative and sum to exactly 1.
  - `PDF α`: outcomes tagged with whether a fault occurred.
  - `pureF`, `bindF`; `pureF_ok`, `bindF_ok`.
  - `faultFree`: the zero-fault branch; `faultFree_bindF` connects the monad to
    the deterministic semantics.
  - `fm`: the probability that a fault occurred. `fm_bindF_le`: a sequence's
    fault probability is at most the first part's plus the second part's bound.
    This is what makes a run of n steps cost at most n times the per-step rate.
  - `flip`: a single-event upset on one bit, at rate p; `flip_ok`, `fm_flip`.
-/
namespace PDist

/-- An interval of rates. -/
structure Iv where
  lo : Rat
  hi : Rat
deriving DecidableEq, Repr

/-- A rate interval inside [0, 1]. -/
def Iv.prob (i : Iv) : Prop := 0 ≤ i.lo ∧ i.lo ≤ i.hi ∧ i.hi ≤ 1

def Iv.mem (i : Iv) (p : Rat) : Prop := i.lo ≤ p ∧ p ≤ i.hi

/-- A rate: a number in [0, 1]. -/
def Rate (p : Rat) : Prop := 0 ≤ p ∧ p ≤ 1

theorem Iv.rate {i : Iv} (hi : i.prob) {p : Rat} (hp : i.mem p) : Rate p :=
  ⟨Rat.le_trans hi.1 hp.1, Rat.le_trans hp.2 hi.2.2⟩

theorem add_le_add' {a b c d : Rat} (h1 : a ≤ b) (h2 : c ≤ d) : a + c ≤ b + d :=
  Rat.le_trans (Rat.add_le_add_right.2 h1) (Rat.add_le_add_left.2 h2)

/-- A finite distribution with exact weights. -/
abbrev PD (α : Type) := List (α × Rat)

def mass {α : Type} : PD α → Rat
  | [] => 0
  | x :: r => x.2 + mass r

/-- Nonnegative weights summing to exactly 1. -/
def PD.ok {α : Type} (d : PD α) : Prop := (∀ x ∈ d, 0 ≤ x.2) ∧ mass d = 1

/-- Outcomes tagged with whether a fault occurred on the way. -/
abbrev PDF (α : Type) := PD (α × Bool)

def pureF {α : Type} (a : α) : PDF α := [((a, false), 1)]

/-- The outcomes of `e`, each weighted by `w` and marked faulty when `t` is. -/
def tagW {β : Type} (t : Bool) (w : Rat) (e : PDF β) : PDF β :=
  e.map fun y => ((y.1.1, t || y.1.2), w * y.2)

/-- Sequencing with fault tags: a fault anywhere marks the outcome. -/
def bindF {α β : Type} : PDF α → (α → PDF β) → PDF β
  | [], _ => []
  | ((a, t), w) :: r, f => tagW t w (f a) ++ bindF r f

/-- The probability that a fault occurred. -/
def fm {α : Type} : PDF α → Rat
  | [] => 0
  | ((_, t), w) :: r => (if t then w else 0) + fm r

/-- The probability that no fault occurred. -/
def ffm {α : Type} : PDF α → Rat
  | [] => 0
  | ((_, t), w) :: r => (if t then 0 else w) + ffm r

/-- What happens if the silicon makes no fault: the values on fault-free outcomes. -/
def faultFree {α : Type} : PDF α → List α
  | [] => []
  | ((a, t), _) :: r => if t then faultFree r else a :: faultFree r

/-! ### Sums -/

theorem mass_append {α : Type} (a b : PD α) : mass (a ++ b) = mass a + mass b := by
  induction a with
  | nil => simp [mass, Rat.zero_add]
  | cons x r ih => simp only [List.cons_append, mass, ih, Rat.add_assoc]

theorem fm_append {α : Type} (a b : PDF α) : fm (a ++ b) = fm a + fm b := by
  induction a with
  | nil => simp [fm, Rat.zero_add]
  | cons x r ih => obtain ⟨⟨_, _⟩, _⟩ := x; simp only [List.cons_append, fm, ih, Rat.add_assoc]

theorem ffm_append {α : Type} (a b : PDF α) : ffm (a ++ b) = ffm a + ffm b := by
  induction a with
  | nil => simp [ffm, Rat.zero_add]
  | cons x r ih => obtain ⟨⟨_, _⟩, _⟩ := x; simp only [List.cons_append, ffm, ih, Rat.add_assoc]

theorem faultFree_append {α : Type} (a b : PDF α) : faultFree (a ++ b) = faultFree a ++ faultFree b := by
  induction a with
  | nil => rfl
  | cons x r ih =>
    obtain ⟨⟨_, t⟩, _⟩ := x
    cases t <;> simp [faultFree, ih]

theorem mass_split {α : Type} (d : PDF α) : mass d = fm d + ffm d := by
  induction d with
  | nil => show (0 : Rat) = 0 + 0; rw [Rat.add_zero]
  | cons x r ih =>
    obtain ⟨⟨_, t⟩, w⟩ := x
    simp only [mass, fm, ffm, ih]
    cases t
    · simp only [Bool.false_eq_true, if_false]
      rw [Rat.zero_add, ← Rat.add_assoc, Rat.add_comm w, Rat.add_assoc]
    · simp only [if_true]
      rw [Rat.zero_add, Rat.add_assoc]

theorem mass_tagW {β : Type} (t : Bool) (w : Rat) (e : PDF β) : mass (tagW t w e) = w * mass e := by
  induction e with
  | nil => simp [tagW, mass, Rat.mul_zero]
  | cons y r ih =>
    simp only [tagW, List.map_cons, mass] at ih ⊢
    rw [ih, Rat.mul_add]

theorem fm_tagW {β : Type} (t : Bool) (w : Rat) (e : PDF β) :
    fm (tagW t w e) = if t then w * mass e else w * fm e := by
  induction e with
  | nil => cases t <;> simp [tagW, fm, mass, Rat.mul_zero]
  | cons y r ih =>
    obtain ⟨⟨b, u⟩, v⟩ := y
    simp only [tagW, List.map_cons, fm, mass] at ih ⊢
    rw [ih]
    cases t <;> cases u <;> simp [Rat.mul_add, Rat.mul_zero, Rat.zero_add]

theorem ffm_tagW {β : Type} (t : Bool) (w : Rat) (e : PDF β) :
    ffm (tagW t w e) = if t then 0 else w * ffm e := by
  induction e with
  | nil => cases t <;> simp [tagW, ffm, Rat.mul_zero]
  | cons y r ih =>
    obtain ⟨⟨b, u⟩, v⟩ := y
    simp only [tagW, List.map_cons, ffm] at ih ⊢
    rw [ih]
    cases t <;> cases u <;> simp [Rat.mul_add, Rat.mul_zero, Rat.zero_add]

theorem faultFree_tagW {β : Type} (t : Bool) (w : Rat) (e : PDF β) :
    faultFree (tagW t w e) = if t then [] else faultFree e := by
  induction e with
  | nil => cases t <;> rfl
  | cons y r ih =>
    obtain ⟨⟨b, u⟩, v⟩ := y
    simp only [tagW, List.map_cons, faultFree] at ih ⊢
    rw [ih]
    cases t <;> cases u <;> simp

theorem ffm_nonneg {α : Type} {d : PDF α} (h : ∀ x ∈ d, 0 ≤ x.2) : 0 ≤ ffm d := by
  induction d with
  | nil => exact Rat.le_refl
  | cons x r ih =>
    obtain ⟨⟨_, t⟩, w⟩ := x
    have hw : 0 ≤ w := h _ (List.mem_cons_self ..)
    have hr := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [ffm]
    cases t
    · have := add_le_add' hw hr; rw [Rat.add_zero] at this; simpa using this
    · simp only [if_true]; rw [Rat.zero_add]; exact hr

theorem fm_nonneg {α : Type} {d : PDF α} (h : ∀ x ∈ d, 0 ≤ x.2) : 0 ≤ fm d := by
  induction d with
  | nil => exact Rat.le_refl
  | cons x r ih =>
    obtain ⟨⟨_, t⟩, w⟩ := x
    have hw : 0 ≤ w := h _ (List.mem_cons_self ..)
    have hr := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [fm]
    cases t
    · simp only [Bool.false_eq_true, if_false]; rw [Rat.zero_add]; exact hr
    · have := add_le_add' hw hr; rw [Rat.add_zero] at this; simpa using this

/-! ### The monad -/

theorem pureF_ok {α : Type} (a : α) : (pureF a).ok := by
  refine ⟨?_, ?_⟩
  · intro x hx
    simp only [pureF, List.mem_cons, List.not_mem_nil, or_false] at hx
    subst hx; show (0 : Rat) ≤ 1; decide +kernel
  · show (1 : Rat) + 0 = 1; rw [Rat.add_zero]

theorem faultFree_pureF {α : Type} (a : α) : faultFree (pureF a) = [a] := rfl

theorem fm_pureF {α : Type} (a : α) : fm (pureF a) = 0 := by
  show (if false = true then (1 : Rat) else 0) + 0 = 0; simp only [Bool.false_eq_true, if_false]; rw [Rat.add_zero]

theorem bindF_ok {α β : Type} {d : PDF α} {f : α → PDF β} (hd : d.ok) (hf : ∀ a, (f a).ok) :
    (bindF d f).ok := by
  obtain ⟨hw, hm⟩ := hd
  refine ⟨?_, ?_⟩
  · clear hm
    induction d with
    | nil => intro x hx; simp [bindF] at hx
    | cons x r ih =>
      obtain ⟨⟨a, t⟩, w⟩ := x
      have h0 : 0 ≤ w := hw _ (List.mem_cons_self ..)
      intro y hy
      simp only [bindF, List.mem_append] at hy
      rcases hy with hy | hy
      · simp only [tagW, List.mem_map] at hy
        obtain ⟨z, hz, rfl⟩ := hy
        exact Rat.mul_nonneg h0 ((hf a).1 z hz)
      · exact ih (fun y hy => hw y (List.mem_cons_of_mem _ hy)) y hy
  · rw [← hm]
    clear hm hw
    induction d with
    | nil => rfl
    | cons x r ih =>
      obtain ⟨⟨a, t⟩, w⟩ := x
      simp only [bindF, mass, mass_append, mass_tagW, (hf a).2, Rat.mul_one, ih]

/-- The zero-fault branch of a sequence is the sequence of zero-fault branches:
the deterministic semantics, read off the probabilistic one. -/
theorem faultFree_bindF {α β : Type} (d : PDF α) (f : α → PDF β) :
    faultFree (bindF d f) = (faultFree d).flatMap fun a => faultFree (f a) := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    simp only [bindF, faultFree_append, faultFree_tagW, ih]
    cases t <;> simp [faultFree]

/-- If every continuation faults with probability at most `q`, a sequence faults
with probability at most that of its first part, plus `q` times the probability
that the first part did not fault. -/
theorem fm_bindF_le {α β : Type} {d : PDF α} {f : α → PDF β} {q : Rat}
    (hw : ∀ x ∈ d, 0 ≤ x.2) (hf : ∀ a, (f a).ok) (hq : ∀ a, fm (f a) ≤ q) :
    fm (bindF d f) ≤ fm d + q * ffm d := by
  induction d with
  | nil => show (0 : Rat) ≤ 0 + q * 0; rw [Rat.mul_zero, Rat.add_zero]; exact Rat.le_refl
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    have h0 : 0 ≤ w := hw _ (List.mem_cons_self ..)
    have hr := ih (fun y hy => hw y (List.mem_cons_of_mem _ hy))
    simp only [bindF, fm_append, fm_tagW, fm, ffm]
    cases t
    · simp only [Bool.false_eq_true, if_false, Rat.zero_add]
      have h1 : w * fm (f a) ≤ w * q := Rat.mul_le_mul_of_nonneg_left (hq a) h0
      calc w * fm (f a) + fm (bindF r f) ≤ w * q + (fm r + q * ffm r) := add_le_add' h1 hr
        _ = fm r + q * (w + ffm r) := by
          rw [Rat.mul_add, Rat.mul_comm w q, ← Rat.add_assoc, Rat.add_comm (q * w) (fm r), Rat.add_assoc]
    · simp only [if_true, (hf a).2, Rat.mul_one]
      rw [Rat.zero_add, Rat.add_assoc]
      exact Rat.add_le_add_left.2 hr

/-- The fault-free part of a distribution has mass at most 1. -/
theorem ffm_le_one {α : Type} {d : PDF α} (h : d.ok) : ffm d ≤ 1 := by
  have hs := mass_split d
  rw [h.2] at hs
  have hf := fm_nonneg h.1
  calc ffm d = 0 + ffm d := (Rat.zero_add _).symm
    _ ≤ fm d + ffm d := Rat.add_le_add_right.2 hf
    _ = 1 := hs.symm

/-- The simple form: faults compose at most additively. -/
theorem fm_bindF_le' {α β : Type} {d : PDF α} {f : α → PDF β} {q : Rat} (hd : d.ok) (h0 : 0 ≤ q)
    (hf : ∀ a, (f a).ok) (hq : ∀ a, fm (f a) ≤ q) : fm (bindF d f) ≤ fm d + q := by
  refine Rat.le_trans (fm_bindF_le hd.1 hf hq) (Rat.add_le_add_left.2 ?_)
  have := Rat.mul_le_mul_of_nonneg_left (ffm_le_one hd) h0
  rwa [Rat.mul_one] at this

/-! ### A single-event upset -/

/-- The value `x`, with bit `b` inverted at rate `p`. -/
def flip (p : Rat) (b x : Nat) : PDF Nat := [((x, false), 1 - p), ((Nat.xor x (2 ^ b), true), p)]

theorem flip_ok {p : Rat} (hp : Rate p) (b x : Nat) : (flip p b x).ok := by
  refine ⟨?_, ?_⟩
  · intro y hy
    simp only [flip, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl
    · exact (Rat.le_iff_sub_nonneg _ _).1 hp.2
    · exact hp.1
  · show (1 - p) + (p + 0) = 1
    rw [Rat.add_zero, Rat.sub_add_cancel]

theorem fm_flip (p : Rat) (b x : Nat) : fm (flip p b x) = p := by
  simp [flip, fm, Rat.add_zero, Rat.zero_add]

/-! ### Witness -/

/-- A calibration: a rate between 5·10⁻⁸ and 2·10⁻⁷ per operation. -/
def p7 : Iv := ⟨1 / 20000000, 1 / 5000000⟩

theorem p7_prob : p7.prob := by refine ⟨?_, ?_, ?_⟩ <;> decide +kernel

/-- One add on 64-bit words, then an upset on bit 3: for every rate in the
calibration, the outcome is a distribution, it faults with probability at most
2·10⁻⁷, and its fault-free branch is the plain sum. -/
theorem witness (p : Rat) (hp : p7.mem p) :
    (flip p 3 (5 + 7)).ok ∧ fm (flip p 3 (5 + 7)) ≤ 1 / 5000000 ∧ faultFree (flip p 3 (5 + 7)) = [12] :=
  ⟨flip_ok (p7.rate p7_prob hp) _ _, by rw [fm_flip]; exact hp.2, rfl⟩

end PDist
