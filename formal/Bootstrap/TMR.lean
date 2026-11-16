/-
TMR: the vote kernel of the triple-modular-redundancy interpreter (decisions 43
to 49 of the root-A stage-1 design).

Three copies of the state step independently on faulty silicon. Each copy's step
is a distribution `D` in which every fault-free outcome is the correct next
state `g` and a fault occurs with probability at most `q`. The three results
are voted:

  - two equal values win, and the winner is written back to all three copies
    (decision 48: the losing copy is repaired);
  - three different values mean no majority, and the interpreter refuses
    (decision 49: `none`, exit status 3).

`tri_err`: one voted step is wrong (refused, or a wrong state) with probability
at most 3·q². A single fault is always corrected; it takes two faulty copies to
go wrong.

`trun_err`: n voted steps are wrong with probability at most n·3·q², against
n·q for the unhardened run (`X86Int.baseline`). The order in the fault rate
goes from 1 to 2.

What this module does not yet cover, and the next ones must:

  - the vote here compares whole states; decision 48 votes only what a step
    wrote, which leaves latent faults in words that are only read, and the
    bound must then carry their exposure (decision 50);
  - the voter itself runs on faulty silicon; here it is exact;
  - the copies' faults are independent by construction (three draws), which the
    spatial layout and triplicated code of decisions 46 and 47 are meant to make
    true of the real machine.
-/
import Bootstrap.PDist

namespace TMR

open PDist

variable {σ : Type} [DecidableEq σ]

/-- Majority of three; `none` when all three differ. -/
def vote (a b c : σ) : Option σ :=
  if a = b ∨ a = c then some a else if b = c then some b else none

theorem vote_of_two {a b c g : σ} (h : (a = g ∧ b = g) ∨ (a = g ∧ c = g) ∨ (b = g ∧ c = g)) :
    vote a b c = some g := by
  unfold vote
  rcases h with ⟨ha, hb⟩ | ⟨ha, hc⟩ | ⟨hb, hc⟩
  · simp [ha, hb]
  · simp [ha, hc]
  · rw [hb, hc]
    by_cases h : a = g
    · simp [h]
    · simp [h]

/-- `1` when `b`, else `0`. -/
def ι (b : Bool) : Rat := if b then 1 else 0

theorem ι_nonneg (b : Bool) : 0 ≤ ι b := by cases b <;> decide

theorem ι_le_one (b : Bool) : ι b ≤ 1 := by cases b <;> decide

/-! ### Weighted sums over a distribution -/

/-- `Σ w · φ(outcome)`. -/
def sumW {α : Type} (D : PDF α) (φ : α × Bool → Rat) : Rat :=
  match D with
  | [] => 0
  | x :: r => x.2 * φ x.1 + sumW r φ

theorem sumW_le {α : Type} {D : PDF α} (hw : ∀ x ∈ D, 0 ≤ x.2) {φ ψ : α × Bool → Rat}
    (h : ∀ x ∈ D, φ x.1 ≤ ψ x.1) : sumW D φ ≤ sumW D ψ := by
  induction D with
  | nil => exact Rat.le_refl
  | cons x r ih =>
    have h0 := hw x (List.mem_cons_self ..)
    have h1 := Rat.mul_le_mul_of_nonneg_left (h x (List.mem_cons_self ..)) h0
    have h2 := ih (fun y hy => hw y (List.mem_cons_of_mem _ hy)) (fun y hy => h y (List.mem_cons_of_mem _ hy))
    exact add_le_add' h1 h2

/-- An integrand that depends only on the fault tag splits by the clean and the
faulty mass. -/
theorem sumW_tag {α : Type} (D : PDF α) (ψ : Bool → Rat) :
    sumW D (fun x => ψ x.2) = ffm D * ψ false + fm D * ψ true := by
  induction D with
  | nil =>
    show (0 : Rat) = 0 * ψ false + 0 * ψ true
    rw [Rat.zero_mul, Rat.zero_mul, Rat.add_zero]
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    show w * ψ t + sumW r (fun x => ψ x.2) = ffm (((a, t), w) :: r) * ψ false + fm (((a, t), w) :: r) * ψ true
    rw [ih]
    cases t <;> simp only [ffm, fm, if_true, Bool.false_eq_true, if_false] <;> grind

theorem sumW_congr {α : Type} (D : PDF α) {φ ψ : α × Bool → Rat} (h : ∀ x, φ x = ψ x) :
    sumW D φ = sumW D ψ := by
  have : φ = ψ := funext h
  rw [this]

theorem sumW_const {α : Type} (D : PDF α) (c : Rat) : sumW D (fun _ => c) = mass D * c := by
  induction D with
  | nil => show (0 : Rat) = 0 * c; rw [Rat.zero_mul]
  | cons x r ih =>
    show x.2 * c + sumW r (fun _ => c) = (x.2 + mass r) * c
    rw [ih, Rat.add_mul]

theorem mem_faultFree {α : Type} {D : PDF α} {x : (α × Bool) × Rat} (hx : x ∈ D) (ht : x.1.2 = false) :
    x.1.1 ∈ faultFree D := by
  induction D with
  | nil => simp at hx
  | cons y r ih =>
    obtain ⟨⟨b, u⟩, v⟩ := y
    rcases List.mem_cons.1 hx with rfl | hr
    · simp only at ht; subst ht; simp [faultFree]
    · have := ih hr
      cases u <;> simp [faultFree, this]

/-- A clean outcome of a distribution whose clean branch is `[g]` is `g`. -/
theorem clean_val {α : Type} {D : PDF α} {g : α} (h : faultFree D = [g]) {x : (α × Bool) × Rat}
    (hx : x ∈ D) (ht : x.1.2 = false) : x.1.1 = g := by
  have := mem_faultFree hx ht
  rw [h] at this
  simpa using this

/-! ### Sequencing with weights -/

def scaleL {β : Type} (w : Rat) (d : PDF β) : PDF β := d.map fun y => (y.1, w * y.2)

/-- Each outcome of `D` continued by `K`, with weights multiplied. -/
def bindW {α β : Type} (D : PDF α) (K : α × Bool → PDF β) : PDF β :=
  match D with
  | [] => []
  | x :: r => scaleL x.2 (K x.1) ++ bindW r K

theorem fm_scaleL {β : Type} (w : Rat) (d : PDF β) : fm (scaleL w d) = w * fm d := by
  induction d with
  | nil => show (0 : Rat) = w * 0; rw [Rat.mul_zero]
  | cons y r ih =>
    obtain ⟨⟨b, t⟩, v⟩ := y
    simp only [scaleL, List.map_cons, fm] at ih ⊢
    rw [ih]; cases t <;> simp <;> grind

theorem mass_scaleL {β : Type} (w : Rat) (d : PDF β) : mass (scaleL w d) = w * mass d := by
  induction d with
  | nil => show (0 : Rat) = w * 0; rw [Rat.mul_zero]
  | cons y r ih =>
    simp only [scaleL, List.map_cons, mass] at ih ⊢
    rw [ih, Rat.mul_add]

theorem faultFree_scaleL {β : Type} (w : Rat) (d : PDF β) : faultFree (scaleL w d) = faultFree d := by
  induction d with
  | nil => rfl
  | cons y r ih =>
    obtain ⟨⟨b, t⟩, v⟩ := y
    simp only [scaleL, List.map_cons, faultFree] at ih ⊢
    rw [ih]

theorem fm_bindW {α β : Type} (D : PDF α) (K : α × Bool → PDF β) :
    fm (bindW D K) = sumW D (fun a => fm (K a)) := by
  induction D with
  | nil => rfl
  | cons x r ih => simp only [bindW, sumW, fm_append, fm_scaleL, ih]

theorem mass_bindW {α β : Type} (D : PDF α) (K : α × Bool → PDF β) :
    mass (bindW D K) = sumW D (fun a => mass (K a)) := by
  induction D with
  | nil => rfl
  | cons x r ih => simp only [bindW, sumW, mass_append, mass_scaleL, ih]

theorem mem_faultFree_bindW {α β : Type} {D : PDF α} {K : α × Bool → PDF β} {y : β}
    (h : y ∈ faultFree (bindW D K)) : ∃ x ∈ D, y ∈ faultFree (K x.1) := by
  induction D with
  | nil => simp [bindW, faultFree] at h
  | cons x r ih =>
    simp only [bindW, faultFree_append, faultFree_scaleL, List.mem_append] at h
    rcases h with h | h
    · exact ⟨x, List.mem_cons_self .., h⟩
    · obtain ⟨z, hz, hy⟩ := ih h
      exact ⟨z, List.mem_cons_of_mem _ hz, hy⟩

theorem bindW_nonneg {α β : Type} {D : PDF α} {K : α × Bool → PDF β} (hw : ∀ x ∈ D, 0 ≤ x.2)
    (hk : ∀ a, ∀ y ∈ K a, 0 ≤ y.2) : ∀ y ∈ bindW D K, 0 ≤ y.2 := by
  induction D with
  | nil => intro y hy; simp [bindW] at hy
  | cons x r ih =>
    intro y hy
    simp only [bindW, List.mem_append] at hy
    rcases hy with hy | hy
    · simp only [scaleL, List.mem_map] at hy
      obtain ⟨z, hz, rfl⟩ := hy
      exact Rat.mul_nonneg (hw x (List.mem_cons_self ..)) (hk _ z hz)
    · exact ih (fun z hz => hw z (List.mem_cons_of_mem _ hz)) y hy

/-! ### One voted step -/

/-- The outcome of a vote, marked as an error unless it is the correct state. -/
def mark (g : σ) (v : Option σ) : PDF (Option σ) := [((v, decide (v ≠ some g)), 1)]

/-- Three independent copies step through `D`, and their results are voted. -/
def tri (D : PDF σ) (g : σ) : PDF (Option σ) :=
  bindW D fun a => bindW D fun b => bindW D fun c => mark g (vote a.1 b.1 c.1)

theorem tri_ok {D : PDF σ} (hD : D.ok) (g : σ) : (tri D g).ok := by
  refine ⟨?_, ?_⟩
  · refine bindW_nonneg hD.1 fun a => bindW_nonneg hD.1 fun b => bindW_nonneg hD.1 fun c => ?_
    intro y hy
    simp only [mark, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; show (0 : Rat) ≤ 1; decide
  · have h1 : ∀ v : Option σ, mass (mark g v) = 1 := fun v => by
      show (1 : Rat) + 0 = 1; rw [Rat.add_zero]
    simp only [tri, mass_bindW, h1, sumW_const, hD.2, Rat.one_mul]

theorem tri_clean {D : PDF σ} (g : σ) : ∀ y ∈ faultFree (tri D g), y = some g := by
  intro y hy
  obtain ⟨_, _, hy⟩ := mem_faultFree_bindW hy
  obtain ⟨_, _, hy⟩ := mem_faultFree_bindW hy
  obtain ⟨_, _, hy⟩ := mem_faultFree_bindW hy
  simp only [mark, faultFree] at hy
  split at hy
  · simp at hy
  · next h => simp at h hy; rw [hy]; exact h

/-- Three pairwise fault coincidences. -/
def P3 (t1 t2 t3 : Bool) : Rat := ι (t1 && t2) + ι (t1 && t3) + ι (t2 && t3)

theorem P3_nonneg (t1 t2 t3 : Bool) : 0 ≤ P3 t1 t2 t3 := by
  cases t1 <;> cases t2 <;> cases t3 <;> simp [P3, ι] <;> decide +kernel

theorem two_or (t1 t2 t3 : Bool) :
    (t1 = false ∧ t2 = false) ∨ (t1 = false ∧ t3 = false) ∨ (t2 = false ∧ t3 = false) ∨ 1 ≤ P3 t1 t2 t3 := by
  cases t1 <;> cases t2 <;> cases t3 <;> simp [P3, ι] <;> decide +kernel

theorem err_le {D : PDF σ} {g : σ} (hc : faultFree D = [g]) {a b c : (σ × Bool) × Rat}
    (ha : a ∈ D) (hb : b ∈ D) (hcc : c ∈ D) :
    ι (decide (vote a.1.1 b.1.1 c.1.1 ≠ some g)) ≤ P3 a.1.2 b.1.2 c.1.2 := by
  by_cases hv : vote a.1.1 b.1.1 c.1.1 = some g
  · rw [hv]
    have : ι (decide (some g ≠ some g)) = 0 := by simp [ι]
    rw [this]; exact P3_nonneg _ _ _
  · refine Rat.le_trans (ι_le_one _) ?_
    rcases two_or a.1.2 b.1.2 c.1.2 with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ | h
    · exact absurd (vote_of_two (Or.inl ⟨clean_val hc ha h1, clean_val hc hb h2⟩)) hv
    · exact absurd (vote_of_two (Or.inr (Or.inl ⟨clean_val hc ha h1, clean_val hc hcc h2⟩))) hv
    · exact absurd (vote_of_two (Or.inr (Or.inr ⟨clean_val hc hb h1, clean_val hc hcc h2⟩))) hv
    · exact h

theorem le_of_eq' {a b : Rat} (h : a = b) : a ≤ b := h ▸ Rat.le_refl

/-- One voted step is wrong with probability at most `3 · e²`, where `e` is a
copy's fault probability. -/
theorem fm_tri_le {D : PDF σ} (hD : D.ok) {g : σ} (hc : faultFree D = [g]) :
    fm (tri D g) ≤ 3 * (fm D * fm D) := by
  have hs : ffm D + fm D = 1 := by rw [Rat.add_comm, ← mass_split, hD.2]
  generalize he : fm D = e at hs ⊢
  generalize hu : ffm D = u at hs
  have hm : ∀ v : Option σ, fm (mark g v) = ι (decide (v ≠ some g)) := fun v => by
    show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
    rw [Rat.add_zero]; rfl
  simp only [tri, fm_bindW, hm]
  -- bound the integrand by the pairwise coincidences
  have step1 : sumW D (fun a => sumW D (fun b => sumW D (fun c => ι (decide (vote a.1 b.1 c.1 ≠ some g)))))
      ≤ sumW D (fun a => sumW D (fun b => sumW D (fun c => P3 a.2 b.2 c.2))) := by
    refine sumW_le hD.1 fun a ha => sumW_le hD.1 fun b hb => sumW_le hD.1 fun c hcc => ?_
    exact err_le hc ha hb hcc
  refine Rat.le_trans step1 (le_of_eq' ?_)
  -- evaluate the coincidence sum, innermost first
  have L3 : ∀ t1 t2 : Bool, sumW D (fun c => P3 t1 t2 c.2) = ι (t1 && t2) + e * (ι t1 + ι t2) := by
    intro t1 t2
    rw [sumW_tag D (fun t => P3 t1 t2 t), hu, he]
    cases t1 <;> cases t2 <;> simp only [P3, ι, Bool.and_true, Bool.and_false, Bool.true_and,
      Bool.false_and, if_true, if_false, Bool.false_eq_true] <;> grind
  have L2 : ∀ t1 : Bool, sumW D (fun b => sumW D (fun c => P3 t1 b.2 c.2)) = 2 * e * ι t1 + e * e := by
    intro t1
    rw [sumW_congr D (fun b => L3 t1 b.2)]
    rw [sumW_tag D (fun t => ι (t1 && t) + e * (ι t1 + ι t)), hu, he]
    cases t1 <;> simp only [ι, Bool.and_true, Bool.and_false, Bool.true_and,
      Bool.false_and, if_true, if_false, Bool.false_eq_true] <;> grind
  rw [sumW_congr D (fun a => L2 a.2), sumW_tag D (fun t => 2 * e * ι t + e * e), hu, he]
  simp only [ι, if_true, if_false, Bool.false_eq_true]
  grind

theorem sq_le {e q : Rat} (h0 : 0 ≤ e) (h : e ≤ q) : e * e ≤ q * q := by
  have h1 : e * e ≤ e * q := Rat.mul_le_mul_of_nonneg_left h h0
  have h2 : e * q ≤ q * q := Rat.mul_le_mul_of_nonneg_right h (Rat.le_trans h0 h)
  exact Rat.le_trans h1 h2

/-- A copy's step: a distribution whose clean branch is the correct next state
`f s`, faulting with probability at most `q`. -/
structure Channel (σ : Type) where
  f : σ → σ
  step : σ → PDF σ
  q : Rat
  q0 : 0 ≤ q
  ok : ∀ s, (step s).ok
  clean : ∀ s, faultFree (step s) = [f s]
  le : ∀ s, fm (step s) ≤ q

theorem tri_err (C : Channel σ) (s : σ) : fm (tri (C.step s) (C.f s)) ≤ 3 * (C.q * C.q) := by
  have h0 := fm_nonneg (C.ok s).1
  refine Rat.le_trans (fm_tri_le (C.ok s) (C.clean s)) ?_
  have := sq_le h0 (C.le s)
  grind

/-! ### n voted steps -/

/-- `f` applied `n` times. -/
def iter (f : σ → σ) : Nat → σ → σ
  | 0, s => s
  | n + 1, s => iter f n (f s)

/-- The voted run; `none` is a refusal, which stops the run. -/
def trun (C : Channel σ) : Nat → Option σ → PDF (Option σ)
  | 0, o => pureF o
  | n + 1, some s => bindF (tri (C.step s) (C.f s)) (trun C n)
  | _ + 1, none => pureF none

theorem trun_ok (C : Channel σ) : ∀ n o, (trun C n o).ok := by
  intro n
  induction n with
  | zero => intro o; exact pureF_ok _
  | succ n ih =>
    intro o
    cases o with
    | some s => exact bindF_ok (tri_ok (C.ok s) _) ih
    | none => exact pureF_ok _

/-- Without an error, the voted run computes the correct state. -/
theorem trun_clean (C : Channel σ) : ∀ n s, ∀ y ∈ faultFree (trun C n (some s)), y = some (iter C.f n s) := by
  intro n
  induction n with
  | zero => intro s y hy; simpa [trun, faultFree_pureF] using hy
  | succ n ih =>
    intro s y hy
    simp only [trun, faultFree_bindF, List.mem_flatMap] at hy
    obtain ⟨a, ha, hy⟩ := hy
    rw [tri_clean _ a ha] at hy
    rw [ih (C.f s) y hy]; rfl

/-- n voted steps are wrong (a refusal or a wrong state) with probability at
most `n · 3 · q²`. -/
theorem trun_err (C : Channel σ) : ∀ n o, fm (trun C n o) ≤ n * (3 * (C.q * C.q)) := by
  have hq : 0 ≤ 3 * (C.q * C.q) := by
    have : 0 ≤ C.q * C.q := Rat.mul_nonneg C.q0 C.q0
    grind
  intro n
  induction n with
  | zero =>
    intro o; rw [trun, fm_pureF]
    show (0 : Rat) ≤ ((0 : Nat) : Rat) * _
    rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]; exact Rat.le_refl
  | succ n ih =>
    intro o
    have hn : ((n + 1 : Nat) : Rat) * (3 * (C.q * C.q)) = 3 * (C.q * C.q) + n * (3 * (C.q * C.q)) := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    have hnn : 0 ≤ (n : Rat) * (3 * (C.q * C.q)) := Rat.mul_nonneg Rat.natCast_nonneg hq
    rw [hn]
    cases o with
    | some s =>
      exact Rat.le_trans (fm_bindF_le' (tri_ok (C.ok s) _) hnn (trun_ok C n) ih)
        (add_le_add' (tri_err C s) Rat.le_refl)
    | none =>
      show fm (pureF (none : Option σ)) ≤ _
      rw [fm_pureF]; grind

/-! ### Witness -/

/-- A counter whose increment may suffer a bit-0 upset at rate `p`. -/
def counter (p : Rat) (hp : Rate p) : Channel Nat where
  f s := s + 1
  step s := flip p 0 (s + 1)
  q := p
  q0 := hp.1
  ok s := flip_ok hp 0 (s + 1)
  clean _ := rfl
  le s := by rw [fm_flip]; exact Rat.le_refl

/-- At any rate in [5·10⁻⁸, 2·10⁻⁷], a million voted increments go wrong with
probability at most 1.2·10⁻⁷. Unhardened, the same million steps are only bounded
by 0.2. -/
theorem witness (p : Rat) (hp : p7.mem p) (o : Option Nat) :
    fm (trun (counter p (p7.rate p7_prob hp)) 1000000 o) ≤ 3 / 25000000 := by
  refine Rat.le_trans (trun_err _ _ _) ?_
  show ((1000000 : Nat) : Rat) * (3 * (p * p)) ≤ 3 / 25000000
  have h0 : 0 ≤ p := (p7.rate p7_prob hp).1
  have h2 : p * p ≤ (1 / 5000000) * (1 / 5000000) := sq_le h0 hp.2
  have : ((1000000 : Nat) : Rat) * (3 * ((1 / 5000000) * (1 / 5000000))) = 3 / 25000000 := by
    decide +kernel
  rw [← this]
  refine Rat.mul_le_mul_of_nonneg_left ?_ Rat.natCast_nonneg
  grind

end TMR
