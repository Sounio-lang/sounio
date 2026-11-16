/-
seed-a on every listing. For every input `L` of bytes,

  (∃ F, exec F seedA L = .exit 0 bs) ↔ HexA L none bs     (`seedA_correct`)

and seed-a exits 1 on every listing that is not valid format A
(`seedA_refuses`). `HexA` is the grammar of format A (Bootstrap/Formats.lean).

The proof is an induction on the input over one iteration of the seed's loop.
At the loop head the seed keeps its state in r12 (the accumulator, with the
sentinel 1) and r14 (inside a comment), as its listing says; `absA` is the head
state with only those and rsp defined. For each of the 34 head states and each
of the 256 bytes the kernel runs one iteration of the model from `absA` and
checks that it comes back to the head in the state `stepA` says, having written
the byte `stepA` says, or exits 1 when `stepA` refuses (`iter_all`). At the end
of input it checks the exit status (`eof_all`). `runU_mono` (Bootstrap/Loop.lean)
carries each iteration from the abstract state to every concrete state above
it, with the rest of the input and the output so far, and `decA_cons` says that
`stepA` is one step of the reference decoder `decA`, which `decA_iff` proves
equal to the grammar.

Kernel time, measured 2026-11-15 (Slurm job 22889): 749 s.
-/
import Bootstrap.Loop
import Bootstrap.Formats
import Bootstrap.SeedData
open X86Min X86Min.Data X86Min.Formats

namespace X86Min.SeedA

def imA : Img := Img.mk' seedA 0x400000 271

def sp : Nat := 0x7ffffffde000

/-- The state `load` gives, with no input. -/
def s0 : St :=
  { regs := (List.replicate 16 none).set 4 (some sp), rip := 0x400078, fl := .undef,
    mem := List.replicate winSize none, lo := sp - winSize / 2, inp := [], out := [] }

theorem load_eq : load seedA sp [] = some (imA, s0) := by decide +kernel

/-- `loop:` in seed-a.hex, the read at the top of the loop. -/
def head : Nat := 0x400081

def accOf : Option Nat → Nat
  | none => 1
  | some h => 16 + h

/-- The loop head as the seed's comments describe it: only rsp, r12 (the
accumulator, 1 or 0x10 + pending nibble) and r14 (inside a comment) defined. -/
def absA (com : Bool) (p : Option Nat) : St :=
  { regs := (((List.replicate 16 none).set 4 (some sp)).set 12 (some (accOf p))).set 14
      (some (if com then 1 else 0)),
    rip := head, fl := .undef, mem := List.replicate winSize none, lo := sp - winSize / 2,
    inp := [], out := [] }

/-- One input byte of `decA`: the next comment flag, pending nibble and bytes
written, or `none` for a refusal. -/
def stepA (com : Bool) (p : Option Nat) (c : Nat) : Option (Bool × Option Nat × List Nat) :=
  if com then some (c != 10, p, [])
  else if c = 35 then some (true, p, [])
  else if isSep c then some (false, p, [])
  else match hexVal c, p with
    | some v, none => some (false, some v, [])
    | some v, some h => some (false, none, [h * 16 + v])
    | none, _ => none

/-- One iteration from the loop head on byte `c` does what `stepA` says. -/
def iterCheck (R : Sum St Outcome) (com : Bool) (p : Option Nat) (c : Nat) : Bool :=
  match R with
  | .inl t => match stepA com p c with
    | some (com', p', e) => leB [] e (absA com' p') t
    | none => false
  | .inr (.exit 1 _) => (stepA com p c).isNone
  | _ => false

def iterOK (com : Bool) (p : Option Nat) (c : Nat) : Bool :=
  iterCheck (runU true imA head 80 { absA com p with inp := [c] }) com p c

/-- At the end of input, exit 0 with nothing pending, 1 otherwise. -/
def eofCheck (R : Sum St Outcome) (p : Option Nat) : Bool :=
  match R with
  | .inr (.exit st x) => x == [] && st == (if p.isNone then 0 else 1)
  | _ => false

def eofOK (com : Bool) (p : Option Nat) : Bool := eofCheck (runU false imA head 80 (absA com p)) p

/-- From entry to the loop head. -/
def entryCheck (R : Sum St Outcome) : Bool :=
  match R with
  | .inl t => leB [] [] (absA false none) t
  | _ => false

def entryOK : Bool := entryCheck (runU true imA head 10 s0)

def validP : Option Nat → Prop
  | none => True
  | some h => h < 16

/-! ### The enumeration, checked by the kernel

One declaration per loop-head state, so that the kernel's cache is per state.
-/

theorem iter_false_none : (List.range 256).all (iterOK false none) = true := by decide +kernel
theorem iter_false_0 : (List.range 256).all (iterOK false (some 0)) = true := by decide +kernel
theorem iter_false_1 : (List.range 256).all (iterOK false (some 1)) = true := by decide +kernel
theorem iter_false_2 : (List.range 256).all (iterOK false (some 2)) = true := by decide +kernel
theorem iter_false_3 : (List.range 256).all (iterOK false (some 3)) = true := by decide +kernel
theorem iter_false_4 : (List.range 256).all (iterOK false (some 4)) = true := by decide +kernel
theorem iter_false_5 : (List.range 256).all (iterOK false (some 5)) = true := by decide +kernel
theorem iter_false_6 : (List.range 256).all (iterOK false (some 6)) = true := by decide +kernel
theorem iter_false_7 : (List.range 256).all (iterOK false (some 7)) = true := by decide +kernel
theorem iter_false_8 : (List.range 256).all (iterOK false (some 8)) = true := by decide +kernel
theorem iter_false_9 : (List.range 256).all (iterOK false (some 9)) = true := by decide +kernel
theorem iter_false_10 : (List.range 256).all (iterOK false (some 10)) = true := by decide +kernel
theorem iter_false_11 : (List.range 256).all (iterOK false (some 11)) = true := by decide +kernel
theorem iter_false_12 : (List.range 256).all (iterOK false (some 12)) = true := by decide +kernel
theorem iter_false_13 : (List.range 256).all (iterOK false (some 13)) = true := by decide +kernel
theorem iter_false_14 : (List.range 256).all (iterOK false (some 14)) = true := by decide +kernel
theorem iter_false_15 : (List.range 256).all (iterOK false (some 15)) = true := by decide +kernel
theorem iter_true_none : (List.range 256).all (iterOK true none) = true := by decide +kernel
theorem iter_true_0 : (List.range 256).all (iterOK true (some 0)) = true := by decide +kernel
theorem iter_true_1 : (List.range 256).all (iterOK true (some 1)) = true := by decide +kernel
theorem iter_true_2 : (List.range 256).all (iterOK true (some 2)) = true := by decide +kernel
theorem iter_true_3 : (List.range 256).all (iterOK true (some 3)) = true := by decide +kernel
theorem iter_true_4 : (List.range 256).all (iterOK true (some 4)) = true := by decide +kernel
theorem iter_true_5 : (List.range 256).all (iterOK true (some 5)) = true := by decide +kernel
theorem iter_true_6 : (List.range 256).all (iterOK true (some 6)) = true := by decide +kernel
theorem iter_true_7 : (List.range 256).all (iterOK true (some 7)) = true := by decide +kernel
theorem iter_true_8 : (List.range 256).all (iterOK true (some 8)) = true := by decide +kernel
theorem iter_true_9 : (List.range 256).all (iterOK true (some 9)) = true := by decide +kernel
theorem iter_true_10 : (List.range 256).all (iterOK true (some 10)) = true := by decide +kernel
theorem iter_true_11 : (List.range 256).all (iterOK true (some 11)) = true := by decide +kernel
theorem iter_true_12 : (List.range 256).all (iterOK true (some 12)) = true := by decide +kernel
theorem iter_true_13 : (List.range 256).all (iterOK true (some 13)) = true := by decide +kernel
theorem iter_true_14 : (List.range 256).all (iterOK true (some 14)) = true := by decide +kernel
theorem iter_true_15 : (List.range 256).all (iterOK true (some 15)) = true := by decide +kernel

theorem eof_facts : [false, true].all (fun com =>
    eofOK com none && (List.range 16).all (fun h => eofOK com (some h))) = true := by decide +kernel

theorem iter_row (com : Bool) (p : Option Nat) (hp : validP p) :
    (List.range 256).all (iterOK com p) = true := by
  cases com with
  | false =>
    cases p with
    | none => exact iter_false_none
    | some h =>
      simp only [validP] at hp
      match h, hp with
      | 0, _ => exact iter_false_0
      | 1, _ => exact iter_false_1
      | 2, _ => exact iter_false_2
      | 3, _ => exact iter_false_3
      | 4, _ => exact iter_false_4
      | 5, _ => exact iter_false_5
      | 6, _ => exact iter_false_6
      | 7, _ => exact iter_false_7
      | 8, _ => exact iter_false_8
      | 9, _ => exact iter_false_9
      | 10, _ => exact iter_false_10
      | 11, _ => exact iter_false_11
      | 12, _ => exact iter_false_12
      | 13, _ => exact iter_false_13
      | 14, _ => exact iter_false_14
      | 15, _ => exact iter_false_15
      | _ + 16, hp => omega
  | true =>
    cases p with
    | none => exact iter_true_none
    | some h =>
      simp only [validP] at hp
      match h, hp with
      | 0, _ => exact iter_true_0
      | 1, _ => exact iter_true_1
      | 2, _ => exact iter_true_2
      | 3, _ => exact iter_true_3
      | 4, _ => exact iter_true_4
      | 5, _ => exact iter_true_5
      | 6, _ => exact iter_true_6
      | 7, _ => exact iter_true_7
      | 8, _ => exact iter_true_8
      | 9, _ => exact iter_true_9
      | 10, _ => exact iter_true_10
      | 11, _ => exact iter_true_11
      | 12, _ => exact iter_true_12
      | 13, _ => exact iter_true_13
      | 14, _ => exact iter_true_14
      | 15, _ => exact iter_true_15
      | _ + 16, hp => omega

theorem iter_all (com : Bool) (p : Option Nat) (hp : validP p) (c : Nat) (hc : c < 256) :
    iterOK com p c = true :=
  List.all_eq_true.mp (iter_row com p hp) c (List.mem_range.mpr hc)

theorem eof_all (com : Bool) (p : Option Nat) (hp : validP p) : eofOK com p = true := by
  have h := List.all_eq_true.mp eof_facts com (by cases com <;> simp)
  simp only [Bool.and_eq_true] at h
  cases p with
  | none => exact h.1
  | some v => exact List.all_eq_true.mp h.2 v (List.mem_range.mpr hp)

theorem entry_ok : entryOK = true := by decide +kernel

theorem hexVal_lt {c v : Nat} (h : hexVal c = some v) : v < 16 := by
  unfold hexVal at h; split at h
  · cases h; omega
  · split at h
    · cases h; omega
    · cases h

theorem stepA_valid {com : Bool} {p : Option Nat} {c : Nat} {com' : Bool} {p' : Option Nat} {e : List Nat}
    (hs : stepA com p c = some (com', p', e)) (hp : validP p) : validP p' := by
  unfold stepA at hs
  split at hs
  · cases hs; exact hp
  split at hs
  · cases hs; exact hp
  split at hs
  · cases hs; exact hp
  cases hx : hexVal c with
  | none => rw [hx] at hs; cases hs
  | some v =>
    cases p with
    | none => rw [hx] at hs; cases hs; exact hexVal_lt hx
    | some h => rw [hx] at hs; cases hs; trivial

theorem decA_cons (c : Nat) (l : List Nat) (com : Bool) (p : Option Nat) (acc : List Nat) :
    decA (c :: l) com p acc = match stepA com p c with
      | some (com', p', e) => decA l com' p' (e ++ acc)
      | none => none := by
  cases com
  · simp only [decA, stepA, Bool.false_eq_true, if_false]
    by_cases h35 : c = 35
    · simp [h35]
    · simp only [h35, if_false]
      by_cases hs : isSep c = true
      · simp [hs]
      · simp only [hs, if_false, Bool.false_eq_true]
        split <;> simp_all
  · simp only [decA, stepA, if_true]
    by_cases h10 : c = 10
    · simp [h10]
    · have : (c != 10) = true := by simp [h10]
      simp [h10, this]

theorem loop : ∀ (L : List Nat) (com : Bool) (p : Option Nat) (acc : List Nat) (s : St),
    (∀ c ∈ L, c < 256) → validP p → StLe L acc (absA com p) s →
    match decA L com p acc with
    | some bs => ∃ k, run imA k s = .exit 0 bs
    | none => ∃ k x, run imA k s = .exit 1 x
  | [], com, p, acc, s, _, hp, h => by
    have he : eofCheck (runU false imA head 80 (absA com p)) p = true := eof_all com p hp
    have hm := runU_mono false imA head (.inl rfl) 80 h
    generalize runU false imA head 80 (absA com p) = R at he hm
    cases R with
    | inl t => cases he
    | inr oc =>
      cases oc with
      | exit st x =>
        simp only [eofCheck, Bool.and_eq_true, beq_iff_eq] at he
        obtain ⟨hx, hst⟩ := he
        obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 st x rfl)
        subst hx
        cases p with
        | none => simp at hst; subst hst; simp only [decA]; exact ⟨k, by simpa using hk⟩
        | some v => simp at hst; subst hst; simp only [decA]; exact ⟨k, _, hk⟩
      | stuck _ _ => cases he
      | outOfFuel => cases he
  | c :: l, com, p, acc, s, hL, hp, h => by
    have hc := hL c (by simp)
    have hi : iterCheck (runU true imA head 80 { absA com p with inp := [c] }) com p c = true :=
      iter_all com p hp c hc
    have h' : StLe l acc { absA com p with inp := [c] } s := StLe.shift_inp (x := [c]) h
    have hm := runU_mono true imA head (.inr rfl) 80 h'
    rw [decA_cons]
    generalize runU true imA head 80 { absA com p with inp := [c] } = R at hi hm
    cases R with
    | inl t =>
      simp only [iterCheck] at hi
      cases hs : stepA com p c with
      | none => rw [hs] at hi; cases hi
      | some q =>
        obtain ⟨com', p', e⟩ := q
        rw [hs] at hi; dsimp only at hi ⊢
        obtain ⟨t', ht', htt⟩ := hm.1 t rfl
        have h2 := StLe.trans (leB_sound hi) htt
        obtain ⟨k, hk⟩ := runU_run 80 ht'
        have ih := loop l com' p' (e ++ acc) t' (fun c hc => hL c (by simp [hc])) (stepA_valid hs hp)
          (by simpa using h2)
        cases hd : decA l com' p' (e ++ acc) with
        | some bs => rw [hd] at ih; obtain ⟨k2, hk2⟩ := ih; exact ⟨k + k2, by rw [hk]; exact hk2⟩
        | none => rw [hd] at ih; obtain ⟨k2, x, hk2⟩ := ih; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩
    | inr oc =>
      cases oc with
      | exit st x =>
        obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 _ _ rfl)
        match st, hi, hk with
        | 1, hi, hk =>
          simp only [iterCheck] at hi
          cases hs : stepA com p c with
          | none => exact ⟨k, _, hk⟩
          | some q => rw [hs] at hi; cases hi
        | 0, hi, _ => cases hi
        | _ + 2, hi, _ => cases hi
      | stuck _ _ => cases hi
      | outOfFuel => cases hi

theorem load_A (L : List Nat) : load seedA 0x7ffffffde000 L = some (imA, { s0 with inp := L }) := by
  rw [load_inp, show (0x7ffffffde000 : Nat) = sp from rfl, load_eq]; rfl

/-- What `exec` does on every listing: if the listing is valid format A, exit 0
with the bytes it describes; if not, exit 1. -/
theorem exec_decA (L : List Nat) (hL : ∀ c ∈ L, c < 256) :
    match decA L false none [] with
    | some bs => ∃ F, exec F seedA L = .exit 0 bs
    | none => ∃ F x, exec F seedA L = .exit 1 x := by
  have he : entryCheck (runU true imA head 10 s0) = true := entry_ok
  have h0 : StLe L [] s0 { s0 with inp := L } := StLe.app_self (r := L) (o := []) s0
  have hm := runU_mono true imA head (.inr rfl) 10 h0
  generalize runU true imA head 10 s0 = R at he hm
  cases R with
  | inr _ => cases he
  | inl t =>
    simp only [entryCheck] at he
    obtain ⟨t', ht', htt⟩ := hm.1 t rfl
    obtain ⟨k, hk⟩ := runU_run 10 ht'
    have hloop := loop L false none [] t' hL trivial (by simpa using StLe.trans (leB_sound he) htt)
    simp only [exec_of_load (load_A L)]
    cases hd : decA L false none [] with
    | some bs => rw [hd] at hloop; obtain ⟨k2, hk2⟩ := hloop; exact ⟨k + k2, by rw [hk]; exact hk2⟩
    | none => rw [hd] at hloop; obtain ⟨k2, x, hk2⟩ := hloop; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩

/-- seed-a exits 0 with output `bs` exactly when `bs` is what the listing `L`
describes in format A. -/
theorem seedA_correct (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) :
    (∃ F, exec F seedA L = .exit 0 bs) ↔ HexA L none bs := by
  rw [← decA_iff]
  have h := exec_decA L hL
  simp only [exec_of_load (load_A L)] at h ⊢
  cases hd : decA L false none [] with
  | some bs' =>
    rw [hd] at h
    obtain ⟨F, hF⟩ := h
    constructor
    · rintro ⟨G, hG⟩; rw [(run_det hF hG).2]
    · intro e; cases e; exact ⟨F, hF⟩
  | none =>
    rw [hd] at h
    obtain ⟨F, x, hF⟩ := h
    constructor
    · rintro ⟨G, hG⟩; cases (run_det hF hG).1
    · intro e; cases e

/-- On a listing that is not valid format A, seed-a exits 1. -/
theorem seedA_refuses (L : List Nat) (hL : ∀ c ∈ L, c < 256) (h : ∀ bs, ¬ HexA L none bs) :
    ∃ F x, exec F seedA L = .exit 1 x := by
  have he := exec_decA L hL
  cases hd : decA L false none [] with
  | some bs => exact absurd ((decA_iff L bs).mp hd) (h bs)
  | none => rw [hd] at he; exact he

end X86Min.SeedA
