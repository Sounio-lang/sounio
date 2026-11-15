/-
seed-b on every listing. For every input `L` of bytes,

  (∃ F, exec F seedB L = .exit 0 bs) ↔ OctB L bs     (`seedB_correct`)

and seed-b exits 1 on every listing that is not valid format B
(`seedB_refuses`). `OctB` is the grammar of format B (Bootstrap/Formats.lean).

The proof is an induction on the input over one iteration of the seed's loop.
At the loop head the seed keeps its state in r12 (the accumulator), r13 (digits
read in the group) and r14 (inside a comment), as its listing says; `absB` is
the head state with only those and rsp defined. For each of the 38 head states
the decoder reaches and each of the 256 bytes the kernel runs one iteration of
the model from `absB` and checks that it comes back to the head in the state `stepB` says, having written
the byte `stepB` says, or exits 1 when `stepB` refuses (`iter_all`). At the end
of input it checks the exit status (`eof_all`). `runU_mono` (Bootstrap/Loop.lean)
carries each iteration from the abstract state to every concrete state above
it, with the rest of the input and the output so far, and `decB_cons` says that
`stepB` is one step of the reference decoder `decB`, which `decB_iff` proves
equal to the grammar.

Kernel time, measured 2026-11-15 (Slurm job 22889): 1139 s.
-/
import Bootstrap.Loop
import Bootstrap.Formats
import Bootstrap.SeedData
open X86Min X86Min.Data X86Min.Formats

namespace X86Min.SeedB

def imB : Img := Img.mk' seedB 0x400000 283

def sp : Nat := 0x7ffffffde000

/-- The state `load` gives, with no input. -/
def s0 : St :=
  { regs := (List.replicate 16 none).set 4 (some sp), rip := 0x400078, fl := .undef,
    mem := List.replicate winSize none, lo := sp - winSize / 2, inp := [], out := [] }

theorem load_eq : load seedB sp [] = some (imB, s0) := by decide +kernel

/-- `loop:` in seed-b.oct (offset 0201), the read at the top of the loop. -/
def head : Nat := 0x400081

/-- The loop head as the seed's comments describe it: only rsp, r12 (the
accumulator), r13 (digits read in the group) and r14 (inside a comment) defined. -/
def absB (com : Bool) (n a : Nat) : St :=
  { regs := ((((List.replicate 16 none).set 4 (some sp)).set 12 (some a)).set 13 (some n)).set 14
      (some (if com then 1 else 0)),
    rip := head, fl := .undef, mem := List.replicate winSize none, lo := sp - winSize / 2,
    inp := [], out := [] }

/-- One input byte of `decB`: the next comment flag, digit count, accumulator and
bytes written, or `none` for a refusal. -/
def stepB (com : Bool) (n a c : Nat) : Option (Bool × Nat × Nat × List Nat) :=
  if com then some (c != 10, n, a, [])
  else if (c = 59 || isSep c) = true then
    (if n ≠ 0 then none else if c = 59 then some (true, 0, a, []) else some (false, 0, a, []))
  else match octVal c with
    | none => none
    | some v =>
      if (n = 0 && v > 3) = true then none
      else if n + 1 = 3 then some (false, 0, 0, [a * 8 + v])
      else some (false, n + 1, a * 8 + v, [])

def iterCheck (R : Sum St Outcome) (com : Bool) (n a c : Nat) : Bool :=
  match R with
  | .inl t => match stepB com n a c with
    | some (com', n', a', e) => leB [] e (absB com' n' a') t
    | none => false
  | .inr (.exit 1 _) => (stepB com n a c).isNone
  | _ => false

/-- One iteration from the loop head on byte `c` does what `stepB` says. -/
def iterOK (com : Bool) (n a c : Nat) : Bool :=
  iterCheck (runU true imB head 80 { absB com n a with inp := [c] }) com n a c

def eofCheck (R : Sum St Outcome) (n : Nat) : Bool :=
  match R with
  | .inr (.exit st x) => x == [] && st == (if n = 0 then 0 else 1)
  | _ => false

/-- At the end of input, exit 0 on a group boundary, 1 otherwise. -/
def eofOK (com : Bool) (n a : Nat) : Bool := eofCheck (runU false imB head 80 (absB com n a)) n

def entryCheck (R : Sum St Outcome) : Bool :=
  match R with
  | .inl t => leB [] [] (absB false 0 0) t
  | _ => false

/-- From entry to the loop head. -/
def entryOK : Bool := entryCheck (runU true imB head 10 s0)

/-- The loop-head states `decB` reaches from its start. -/
def validB (com : Bool) (n a : Nat) : Prop :=
  (n = 0 ∧ a = 0) ∨ (com = false ∧ n = 1 ∧ a < 4) ∨ (com = false ∧ n = 2 ∧ a < 32)

/-! ### The enumeration, checked by the kernel

One declaration per loop-head state, so that the kernel's cache is per state.
-/
theorem iter_false_0 : (List.range 256).all (iterOK false 0 0) = true := by decide +kernel
theorem iter_true_0 : (List.range 256).all (iterOK true 0 0) = true := by decide +kernel
theorem iter_1_0 : (List.range 256).all (iterOK false 1 0) = true := by decide +kernel
theorem iter_1_1 : (List.range 256).all (iterOK false 1 1) = true := by decide +kernel
theorem iter_1_2 : (List.range 256).all (iterOK false 1 2) = true := by decide +kernel
theorem iter_1_3 : (List.range 256).all (iterOK false 1 3) = true := by decide +kernel
theorem iter_2_0 : (List.range 256).all (iterOK false 2 0) = true := by decide +kernel
theorem iter_2_1 : (List.range 256).all (iterOK false 2 1) = true := by decide +kernel
theorem iter_2_2 : (List.range 256).all (iterOK false 2 2) = true := by decide +kernel
theorem iter_2_3 : (List.range 256).all (iterOK false 2 3) = true := by decide +kernel
theorem iter_2_4 : (List.range 256).all (iterOK false 2 4) = true := by decide +kernel
theorem iter_2_5 : (List.range 256).all (iterOK false 2 5) = true := by decide +kernel
theorem iter_2_6 : (List.range 256).all (iterOK false 2 6) = true := by decide +kernel
theorem iter_2_7 : (List.range 256).all (iterOK false 2 7) = true := by decide +kernel
theorem iter_2_8 : (List.range 256).all (iterOK false 2 8) = true := by decide +kernel
theorem iter_2_9 : (List.range 256).all (iterOK false 2 9) = true := by decide +kernel
theorem iter_2_10 : (List.range 256).all (iterOK false 2 10) = true := by decide +kernel
theorem iter_2_11 : (List.range 256).all (iterOK false 2 11) = true := by decide +kernel
theorem iter_2_12 : (List.range 256).all (iterOK false 2 12) = true := by decide +kernel
theorem iter_2_13 : (List.range 256).all (iterOK false 2 13) = true := by decide +kernel
theorem iter_2_14 : (List.range 256).all (iterOK false 2 14) = true := by decide +kernel
theorem iter_2_15 : (List.range 256).all (iterOK false 2 15) = true := by decide +kernel
theorem iter_2_16 : (List.range 256).all (iterOK false 2 16) = true := by decide +kernel
theorem iter_2_17 : (List.range 256).all (iterOK false 2 17) = true := by decide +kernel
theorem iter_2_18 : (List.range 256).all (iterOK false 2 18) = true := by decide +kernel
theorem iter_2_19 : (List.range 256).all (iterOK false 2 19) = true := by decide +kernel
theorem iter_2_20 : (List.range 256).all (iterOK false 2 20) = true := by decide +kernel
theorem iter_2_21 : (List.range 256).all (iterOK false 2 21) = true := by decide +kernel
theorem iter_2_22 : (List.range 256).all (iterOK false 2 22) = true := by decide +kernel
theorem iter_2_23 : (List.range 256).all (iterOK false 2 23) = true := by decide +kernel
theorem iter_2_24 : (List.range 256).all (iterOK false 2 24) = true := by decide +kernel
theorem iter_2_25 : (List.range 256).all (iterOK false 2 25) = true := by decide +kernel
theorem iter_2_26 : (List.range 256).all (iterOK false 2 26) = true := by decide +kernel
theorem iter_2_27 : (List.range 256).all (iterOK false 2 27) = true := by decide +kernel
theorem iter_2_28 : (List.range 256).all (iterOK false 2 28) = true := by decide +kernel
theorem iter_2_29 : (List.range 256).all (iterOK false 2 29) = true := by decide +kernel
theorem iter_2_30 : (List.range 256).all (iterOK false 2 30) = true := by decide +kernel
theorem iter_2_31 : (List.range 256).all (iterOK false 2 31) = true := by decide +kernel

theorem eof_facts : (eofOK false 0 0 && eofOK true 0 0 &&
    (List.range 4).all (eofOK false 1) && (List.range 32).all (eofOK false 2)) = true := by decide +kernel

theorem iter_row (com : Bool) (n a : Nat) (hv : validB com n a) :
    (List.range 256).all (iterOK com n a) = true := by
  rcases hv with ⟨hn, ha⟩ | ⟨hc, hn, ha⟩ | ⟨hc, hn, ha⟩
  · subst hn ha; cases com
    · exact iter_false_0
    · exact iter_true_0
  · subst hc hn
    match a, ha with
    | 0, _ => exact iter_1_0
    | 1, _ => exact iter_1_1
    | 2, _ => exact iter_1_2
    | 3, _ => exact iter_1_3
    | _ + 4, ha => omega
  · subst hc hn
    match a, ha with
    | 0, _ => exact iter_2_0
    | 1, _ => exact iter_2_1
    | 2, _ => exact iter_2_2
    | 3, _ => exact iter_2_3
    | 4, _ => exact iter_2_4
    | 5, _ => exact iter_2_5
    | 6, _ => exact iter_2_6
    | 7, _ => exact iter_2_7
    | 8, _ => exact iter_2_8
    | 9, _ => exact iter_2_9
    | 10, _ => exact iter_2_10
    | 11, _ => exact iter_2_11
    | 12, _ => exact iter_2_12
    | 13, _ => exact iter_2_13
    | 14, _ => exact iter_2_14
    | 15, _ => exact iter_2_15
    | 16, _ => exact iter_2_16
    | 17, _ => exact iter_2_17
    | 18, _ => exact iter_2_18
    | 19, _ => exact iter_2_19
    | 20, _ => exact iter_2_20
    | 21, _ => exact iter_2_21
    | 22, _ => exact iter_2_22
    | 23, _ => exact iter_2_23
    | 24, _ => exact iter_2_24
    | 25, _ => exact iter_2_25
    | 26, _ => exact iter_2_26
    | 27, _ => exact iter_2_27
    | 28, _ => exact iter_2_28
    | 29, _ => exact iter_2_29
    | 30, _ => exact iter_2_30
    | 31, _ => exact iter_2_31
    | _ + 32, ha => omega

theorem iter_all (com : Bool) (n a : Nat) (hv : validB com n a) (c : Nat) (hc : c < 256) :
    iterOK com n a c = true :=
  List.all_eq_true.mp (iter_row com n a hv) c (List.mem_range.mpr hc)

theorem eof_all (com : Bool) (n a : Nat) (hv : validB com n a) : eofOK com n a = true := by
  have h := eof_facts
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨⟨h00, h10⟩, h1⟩, h2⟩ := h
  rcases hv with ⟨hn, ha⟩ | ⟨hc, hn, ha⟩ | ⟨hc, hn, ha⟩
  · subst hn ha; cases com
    · exact h00
    · exact h10
  · subst hc hn; exact List.all_eq_true.mp h1 a (List.mem_range.mpr ha)
  · subst hc hn; exact List.all_eq_true.mp h2 a (List.mem_range.mpr ha)

theorem entry_ok : entryOK = true := by decide +kernel

theorem octVal_le {c v : Nat} (h : octVal c = some v) : v ≤ 7 := by
  unfold octVal at h; split at h
  · cases h; omega
  · cases h

theorem stepB_valid {com : Bool} {n a c : Nat} {com' : Bool} {n' a' : Nat} {e : List Nat}
    (hs : stepB com n a c = some (com', n', a', e)) (hv : validB com n a) : validB com' n' a' := by
  unfold stepB at hs
  split at hs
  · cases hs
    rcases hv with ⟨hn, ha⟩ | ⟨hc, _⟩ | ⟨hc, _⟩
    · exact .inl ⟨hn, ha⟩
    all_goals (subst hc; contradiction)
  split at hs
  · split at hs
    · cases hs
    · rename_i hn
      have hn0 : n = 0 := by omega
      have ha0 : a = 0 := by
        rcases hv with ⟨_, ha⟩ | ⟨_, hn1, _⟩ | ⟨_, hn1, _⟩
        · exact ha
        · omega
        · omega
      split at hs <;> (cases hs; exact .inl ⟨rfl, ha0⟩)
  · rename_i hcom _
    have hc : com = false := by cases com <;> simp_all
    cases hx : octVal c with
    | none => rw [hx] at hs; cases hs
    | some v =>
      rw [hx] at hs; dsimp only at hs
      have hv7 := octVal_le hx
      split at hs
      · cases hs
      · rename_i hbig
        split at hs
        · cases hs; exact .inl ⟨rfl, rfl⟩
        · cases hs
          rename_i h3
          rcases hv with ⟨hn, ha⟩ | ⟨_, hn, ha⟩ | ⟨_, hn, ha⟩
          · subst hn ha
            have : v ≤ 3 := by simp at hbig; omega
            exact .inr (.inl ⟨rfl, rfl, by omega⟩)
          · subst hn; exact .inr (.inr ⟨rfl, rfl, by omega⟩)
          · subst hn; omega

theorem decB_cons (c : Nat) (l : List Nat) (com : Bool) (n a : Nat) (acc : List Nat) :
    decB (c :: l) com n a acc = match stepB com n a c with
      | some (com', n', a', e) => decB l com' n' a' (e ++ acc)
      | none => none := by
  cases com
  · simp only [decB, stepB, Bool.false_eq_true, if_false]
    split
    · split
      · rfl
      · split <;> rfl
    · cases octVal c with
      | none => rfl
      | some v =>
        dsimp only
        split
        · rfl
        · split <;> rfl
  · simp only [decB, stepB, if_true]
    by_cases h10 : c = 10
    · simp [h10]
    · have : (c != 10) = true := by simp [h10]
      simp [h10, this]

theorem loop : ∀ (L : List Nat) (com : Bool) (n a : Nat) (acc : List Nat) (s : St),
    (∀ c ∈ L, c < 256) → validB com n a → StLe L acc (absB com n a) s →
    match decB L com n a acc with
    | some bs => ∃ k, run imB k s = .exit 0 bs
    | none => ∃ k x, run imB k s = .exit 1 x
  | [], com, n, a, acc, s, _, hv, h => by
    have he : eofCheck (runU false imB head 80 (absB com n a)) n = true := eof_all com n a hv
    have hm := runU_mono false imB head (.inl rfl) 80 h
    generalize runU false imB head 80 (absB com n a) = R at he hm
    cases R with
    | inl t => cases he
    | inr oc =>
      cases oc with
      | exit st x =>
        simp only [eofCheck, Bool.and_eq_true, beq_iff_eq] at he
        obtain ⟨hx, hst⟩ := he
        obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 st x rfl)
        subst hx
        simp only [decB]
        by_cases hn : n = 0
        · simp only [hn, if_true] at hst ⊢; subst hst; exact ⟨k, by simpa using hk⟩
        · simp only [hn, if_false] at hst ⊢; subst hst; exact ⟨k, _, hk⟩
      | stuck _ _ => cases he
      | outOfFuel => cases he
  | c :: l, com, n, a, acc, s, hL, hv, h => by
    have hc := hL c (by simp)
    have hi : iterCheck (runU true imB head 80 { absB com n a with inp := [c] }) com n a c = true :=
      iter_all com n a hv c hc
    have h' : StLe l acc { absB com n a with inp := [c] } s := StLe.shift_inp (x := [c]) h
    have hm := runU_mono true imB head (.inr rfl) 80 h'
    rw [decB_cons]
    generalize runU true imB head 80 { absB com n a with inp := [c] } = R at hi hm
    cases R with
    | inl t =>
      simp only [iterCheck] at hi
      cases hs : stepB com n a c with
      | none => rw [hs] at hi; cases hi
      | some q =>
        obtain ⟨com', n', a', e⟩ := q
        rw [hs] at hi; dsimp only at hi ⊢
        obtain ⟨t', ht', htt⟩ := hm.1 t rfl
        have h2 := StLe.trans (leB_sound hi) htt
        obtain ⟨k, hk⟩ := runU_run 80 ht'
        have ih := loop l com' n' a' (e ++ acc) t' (fun c hc => hL c (by simp [hc])) (stepB_valid hs hv)
          (by simpa using h2)
        cases hd : decB l com' n' a' (e ++ acc) with
        | some bs => rw [hd] at ih; obtain ⟨k2, hk2⟩ := ih; exact ⟨k + k2, by rw [hk]; exact hk2⟩
        | none => rw [hd] at ih; obtain ⟨k2, x, hk2⟩ := ih; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩
    | inr oc =>
      cases oc with
      | exit st x =>
        obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 _ _ rfl)
        match st, hi, hk with
        | 1, hi, hk =>
          simp only [iterCheck] at hi
          cases hs : stepB com n a c with
          | none => exact ⟨k, _, hk⟩
          | some q => rw [hs] at hi; cases hi
        | 0, hi, _ => cases hi
        | _ + 2, hi, _ => cases hi
      | stuck _ _ => cases hi
      | outOfFuel => cases hi

theorem load_B (L : List Nat) : load seedB 0x7ffffffde000 L = some (imB, { s0 with inp := L }) := by
  rw [load_inp, show (0x7ffffffde000 : Nat) = sp from rfl, load_eq]; rfl

/-- What `exec` does on every listing: if the listing is valid format B, exit 0
with the bytes it describes; if not, exit 1. -/
theorem exec_decB (L : List Nat) (hL : ∀ c ∈ L, c < 256) :
    match decB L false 0 0 [] with
    | some bs => ∃ F, exec F seedB L = .exit 0 bs
    | none => ∃ F x, exec F seedB L = .exit 1 x := by
  have he : entryCheck (runU true imB head 10 s0) = true := entry_ok
  have h0 : StLe L [] s0 { s0 with inp := L } := StLe.app_self (r := L) (o := []) s0
  have hm := runU_mono true imB head (.inr rfl) 10 h0
  generalize runU true imB head 10 s0 = R at he hm
  cases R with
  | inr _ => cases he
  | inl t =>
    simp only [entryCheck] at he
    obtain ⟨t', ht', htt⟩ := hm.1 t rfl
    obtain ⟨k, hk⟩ := runU_run 10 ht'
    have hloop := loop L false 0 0 [] t' hL (.inl ⟨rfl, rfl⟩)
      (by simpa using StLe.trans (leB_sound he) htt)
    simp only [exec_of_load (load_B L)]
    cases hd : decB L false 0 0 [] with
    | some bs => rw [hd] at hloop; obtain ⟨k2, hk2⟩ := hloop; exact ⟨k + k2, by rw [hk]; exact hk2⟩
    | none => rw [hd] at hloop; obtain ⟨k2, x, hk2⟩ := hloop; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩

/-- seed-b exits 0 with output `bs` exactly when `bs` is what the listing `L`
describes in format B. -/
theorem seedB_correct (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) :
    (∃ F, exec F seedB L = .exit 0 bs) ↔ OctB L bs := by
  rw [← decB_iff]
  have h := exec_decB L hL
  simp only [exec_of_load (load_B L)] at h ⊢
  cases hd : decB L false 0 0 [] with
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

/-- On a listing that is not valid format B, seed-b exits 1. -/
theorem seedB_refuses (L : List Nat) (hL : ∀ c ∈ L, c < 256) (h : ∀ bs, ¬ OctB L bs) :
    ∃ F x, exec F seedB L = .exit 1 x := by
  have he := exec_decB L hL
  cases hd : decB L false 0 0 [] with
  | some bs => exact absurd ((decB_iff L bs).mp hd) (h bs)
  | none => rw [hd] at he; exact he

end X86Min.SeedB
