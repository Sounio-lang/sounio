/-
Runs cut at a loop head, and a decidable check for refinement.

`runU g im head n s` steps from `s` until it comes back to `head` (after at least
one step), for at most `n` steps. With `g = true` it stops, stuck, at a read of
standard input when the input is used up: one iteration of a loop checked from a
state holding only that iteration's input must not read past it, since the input
that follows (`r` in `StLe r o`) would be read there.

`runU_mono` lifts `step_mono` to these runs. `runU_run` turns one into a run of
`run`. `leB` decides a sufficient condition for `StLe`, so that an iteration can
be checked by evaluation in the kernel.
-/
import Bootstrap.Refine

namespace X86Min

variable {r o : List Nat}

def stepG (g : Bool) (im : Img) (s : St) : Sum St Outcome :=
  if (g && readsEOF im s) = true then .inr (.stuck s.rip 3) else step im s

def runU (g : Bool) (im : Img) (head : Nat) (n : Nat) : St → Sum St Outcome :=
  Nat.rec (motive := fun _ => St → Sum St Outcome) (fun _ => .inr .outOfFuel)
    (fun _ ih s => match stepG g im s with
      | .inl t => if t.rip == head then .inl t else ih t
      | .inr o => .inr o) n

theorem runU_succ (g : Bool) (im : Img) (head n : Nat) (s : St) :
    runU g im head (n + 1) s = match stepG g im s with
      | .inl t => if t.rip == head then .inl t else runU g im head n t
      | .inr o => .inr o := rfl

/-- A read at the end of the input with `rax` undefined is stuck. -/
theorem step_syscall_none (im : Img) (s : St) (len : Nat)
    (hd : im.decodeAt s.rip = some (.syscall, len)) (h0 : s.reg 0 = none) :
    ∃ w, step im s = .inr (.stuck s.rip w) := by
  unfold step; rw [hd]; dsimp only; rw [h0]; exact ⟨2, rfl⟩

theorem stepG_mono (g : Bool) (im : Img) {a b : St} (h : StLe r o a b) (hg : r = [] ∨ g = true) :
    StepLe r o (stepG g im a) (stepG g im b) := by
  unfold stepG
  by_cases ha : (g && readsEOF im a) = true
  · rw [if_pos ha]; exact StepLe.stuck _ _ _
  · rw [if_neg ha]
    have hga : r = [] ∨ readsEOF im a = false := by
      rcases hg with hg | hg
      · exact .inl hg
      · subst hg; simp at ha; exact .inr ha
    by_cases hb : (g && readsEOF im b) = true
    · rw [if_pos hb]
      -- b reads at the end of input, a does not: then a's rax is undefined
      simp only [Bool.and_eq_true, readsEOF] at hb
      obtain ⟨hgt, ⟨hbi, hb0⟩, hbd⟩ := hb
      have hai : a.inp = [] := by
        have := h.inp; simp only [List.isEmpty_iff] at hbi; rw [hbi] at this
        exact (List.append_eq_nil_iff.mp this.symm).1
      have hrip : b.rip = a.rip := h.rip.symm
      rw [hrip] at hbd
      split at hbd
      · rename_i len hdd
        have ha0 : a.reg 0 = none := by
          cases e : a.reg 0 with
          | none => rfl
          | some v =>
            have hv := h.reg 0 v e
            simp only [beq_iff_eq] at hb0; rw [hb0] at hv
            cases hv
            exfalso; apply ha; subst hgt
            simp [readsEOF, hai, e, hdd]
        obtain ⟨w, hw⟩ := step_syscall_none im a len hdd ha0
        rw [hw]; exact StepLe.stuck _ _ _
      · cases hbd
    · rw [if_neg hb]; exact step_mono im h hga

theorem StepLe.fuel (rb : Sum St Outcome) : StepLe r o (.inr .outOfFuel) rb := by
  constructor
  · intro _ e; cases e
  · intro _ _ e; cases e

theorem StepLe.exit' (st : Nat) (y : List Nat) :
    StepLe r o (.inr (.exit st y)) (.inr (.exit st (o.reverse ++ y))) := by
  constructor
  · intro _ e; cases e
  · intro _ _ e; cases e; rfl

theorem runU_mono (g : Bool) (im : Img) (head : Nat) (hg : r = [] ∨ g = true) :
    ∀ (n : Nat) {a b : St}, StLe r o a b → StepLe r o (runU g im head n a) (runU g im head n b) := by
  intro n; induction n with
  | zero => intro a b _; exact StepLe.fuel _
  | succ n ih =>
    intro a b h
    rw [runU_succ, runU_succ]
    have hs := stepG_mono g im h hg
    cases ea : stepG g im a with
    | inl t =>
      obtain ⟨t', et', ht⟩ := hs.1 t ea
      rw [et']; dsimp only
      rw [← ht.rip]
      by_cases hh : (t.rip == head) = true
      · rw [if_pos hh, if_pos hh]; exact StepLe.inl ht
      · rw [if_neg hh, if_neg hh]; exact ih ht
    | inr x =>
      dsimp only
      cases x with
      | exit st y => rw [hs.2 st y ea]; exact StepLe.exit' _ _
      | stuck p w => exact StepLe.stuck _ _ _
      | outOfFuel => exact StepLe.fuel _

theorem stepG_inl {g : Bool} {im : Img} {s t : St} (h : stepG g im s = .inl t) : step im s = .inl t := by
  unfold stepG at h; split at h
  · cases h
  · exact h

theorem stepG_exit {g : Bool} {im : Img} {s : St} {st : Nat} {x : List Nat}
    (h : stepG g im s = .inr (.exit st x)) : step im s = .inr (.exit st x) := by
  unfold stepG at h; split at h
  · cases h
  · exact h

/-- An iteration that comes back to the head is a prefix of the run. -/
theorem runU_run {g : Bool} {im : Img} {head : Nat} :
    ∀ (n : Nat) {s t : St}, runU g im head n s = .inl t → ∃ k, ∀ F, run im (k + F) s = run im F t := by
  intro n; induction n with
  | zero => intro s t e; cases e
  | succ n ih =>
    intro s t e
    rw [runU_succ] at e
    cases es : stepG g im s with
    | inl u =>
      rw [es] at e; dsimp only at e
      have hs := stepG_inl es
      by_cases hh : (u.rip == head) = true
      · rw [if_pos hh] at e; cases e
        exact ⟨1, fun F => by rw [Nat.add_comm, run_succ, hs]⟩
      · rw [if_neg hh] at e
        obtain ⟨k, hk⟩ := ih e
        exact ⟨k + 1, fun F => by rw [Nat.add_right_comm, run_succ, hs]; exact hk F⟩
    | inr x => rw [es] at e; cases e

/-- An iteration that ends the run. -/
theorem runU_exit {g : Bool} {im : Img} {head : Nat} :
    ∀ (n : Nat) {s : St} {st : Nat} {x : List Nat}, runU g im head n s = .inr (.exit st x) →
      ∃ k, run im k s = .exit st x := by
  intro n; induction n with
  | zero => intro s st x e; cases e
  | succ n ih =>
    intro s st x e
    rw [runU_succ] at e
    cases es : stepG g im s with
    | inl u =>
      rw [es] at e; dsimp only at e
      have hs := stepG_inl es
      by_cases hh : (u.rip == head) = true
      · rw [if_pos hh] at e; cases e
      · rw [if_neg hh] at e
        obtain ⟨k, hk⟩ := ih e
        exact ⟨k + 1, by rw [run_succ, hs]; exact hk⟩
    | inr y =>
      rw [es] at e; dsimp only at e; cases e
      exact ⟨1, by rw [run_succ, stepG_exit es]⟩

/-- More fuel does not change an exit. -/
theorem run_fuel {im : Img} : ∀ (F m : Nat) {s : St} {st : Nat} {x : List Nat},
    run im F s = .exit st x → run im (F + m) s = .exit st x := by
  intro F; induction F with
  | zero => intro m s st x e; cases e
  | succ F ih =>
    intro m s st x e
    rw [Nat.add_right_comm, run_succ]
    rw [run_succ] at e
    cases es : step im s with
    | inl t => rw [es] at e; exact ih m e
    | inr y => rw [es] at e; exact e

/-! ### Composing refinements -/

theorem OLe.trans {α : Type} {x y z : Option α} (h1 : OLe x y) (h2 : OLe y z) : OLe x z :=
  fun v hv => h2 v (h1 v hv)

theorem StLe.trans {r1 o1 r2 o2 : List Nat} {a b c : St} (h1 : StLe r1 o1 a b) (h2 : StLe r2 o2 b c) :
    StLe (r1 ++ r2) (o1 ++ o2) a c :=
  ⟨fun i => (h1.reg i).trans (h2.reg i), h1.rlen.trans h2.rlen, h1.rip.trans h2.rip,
   ⟨h1.fl.cf.trans h2.fl.cf, h1.fl.zf.trans h2.fl.zf, h1.fl.sf.trans h2.fl.sf, h1.fl.of.trans h2.fl.of⟩,
   fun x => (h1.byte x).trans (h2.byte x), h1.mlen.trans h2.mlen, h1.lo.trans h2.lo,
   by rw [h2.inp, h1.inp, List.append_assoc], by rw [h2.out, h1.out, List.append_assoc]⟩

/-- Moving the first input bytes of `r` into the abstract state. -/
theorem StLe.shift_inp {x : List Nat} {a b : St} (h : StLe (x ++ r) o a b) :
    StLe r o { a with inp := a.inp ++ x } b :=
  ⟨h.reg, h.rlen, h.rip, h.fl, h.byte, h.mlen, h.lo, by rw [h.inp, List.append_assoc], h.out⟩

theorem StLe.app_self (a : St) : StLe r o a { a with inp := a.inp ++ r, out := a.out ++ o } :=
  ⟨fun _ => OLe.refl _, rfl, rfl, FlLe.refl _, fun _ => OLe.refl _, rfl, rfl, rfl, rfl⟩

/-- Two exits of the same run, with different fuel, are the same exit. -/
theorem run_det {im : Img} {s : St} {F G a b : Nat} {x y : List Nat}
    (h1 : run im F s = .exit a x) (h2 : run im G s = .exit b y) : a = b ∧ x = y := by
  have e1 := run_fuel F G h1
  have e2 := run_fuel G F h2
  rw [Nat.add_comm, e2] at e1
  cases e1; exact ⟨rfl, rfl⟩

/-- `load` does not look at the input. -/
theorem load_inp (f : List Nat) (sp : Nat) (L : List Nat) :
    load f sp L = (load f sp []).map (fun q => (q.1, { q.2 with inp := L })) := by
  unfold load
  simp only [Option.bind_eq_bind, Option.map_bind, apply_ite (Option.map _), Option.map_none, pure,
    Option.map_some, Function.comp_def]

/-! ### Deciding refinement -/

def oleB {α : Type} [BEq α] : Option α → Option α → Bool
  | none, _ => true
  | some x, y => y == some x

def lleB : List (Option Nat) → List (Option Nat) → Bool
  | [], [] => true
  | x :: l, y :: m => oleB x y && lleB l m
  | _, _ => false

def leB (r o : List Nat) (a b : St) : Bool :=
  lleB a.regs b.regs && a.rip == b.rip &&
  oleB a.fl.cf b.fl.cf && oleB a.fl.zf b.fl.zf && oleB a.fl.sf b.fl.sf && oleB a.fl.of b.fl.of &&
  lleB a.mem b.mem && a.lo == b.lo && b.inp == a.inp ++ r && b.out == a.out ++ o

theorem oleB_sound {α : Type} [BEq α] [LawfulBEq α] {x y : Option α} (h : oleB x y = true) : OLe x y := by
  intro v hv; subst hv; simp only [oleB, beq_iff_eq] at h; exact h

theorem lleB_sound : ∀ {l m : List (Option Nat)}, lleB l m = true →
    l.length = m.length ∧ ∀ i : Nat, OLe ((l[i]?).getD none) ((m[i]?).getD none)
  | [], [], _ => ⟨rfl, fun _ => OLe.refl _⟩
  | x :: l, y :: m, h => by
    simp only [lleB, Bool.and_eq_true] at h
    obtain ⟨hl, hi⟩ := lleB_sound h.2
    refine ⟨by simp [hl], fun i => ?_⟩
    cases i with
    | zero => exact oleB_sound h.1
    | succ i => exact hi i
  | [], _ :: _, h => by simp [lleB] at h
  | _ :: _, [], h => by simp [lleB] at h

theorem leB_sound {a b : St} (h : leB r o a b = true) : StLe r o a b := by
  simp only [leB, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨hr, hrip⟩, hcf⟩, hzf⟩, hsf⟩, hof⟩, hm⟩, hlo⟩, hi⟩, ho⟩ := h
  obtain ⟨hrl, hri⟩ := lleB_sound hr
  obtain ⟨hml, hmi⟩ := lleB_sound hm
  refine ⟨fun i => hri i, hrl, hrip, ⟨oleB_sound hcf, oleB_sound hzf, oleB_sound hsf, oleB_sound hof⟩,
    fun x => ?_, hml, hlo, hi, ho⟩
  simp only [St.rdByte, hlo]
  split
  · exact hmi _
  · exact OLe.refl _

end X86Min
