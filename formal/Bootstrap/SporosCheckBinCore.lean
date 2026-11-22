/-
SporosCheckBin: spr1_check, the checker of Sporos bytecode in machine code
(bootstrap/model/spr1_check.hex), against the semantics of X86Int / X86Env.

This first part: the image, the table of its instructions, and the machinery
for running code symbolically (`Reaches`, `Frame`, the step lemmas).
-/
import Bootstrap.SporosCheckBinData
import Bootstrap.X86Dec

namespace SporosCheckBin

open X86Env X86Int

def BASE : Nat := 0x400000

/-- The code as `X86Int.step` sees it: the bytes of spr1_check at 0x400000. -/
def im : Nat → Option (Ins × Nat) := X86Dec.imageOf spr1Bytes BASE

/-! ### The table is what the decoder reads from the bytes -/

/-- Walk the code once: each table entry decodes at its address from the bytes
that remain there. -/
def walkOK : List Nat → Nat → List (Nat × Ins × Nat) → Bool
  | _, _, [] => true
  | rest, a, (a', i, l) :: t =>
    a == a' && X86Dec.decode (fun j => rest[j]?) a == some (i, l) && walkOK (rest.drop l) (a + l) t

theorem walkOK_sound (code : List Nat) : ∀ (tbl : List (Nat × Ins × Nat)) (rest : List Nat) (a : Nat),
    walkOK rest a tbl = true → BASE ≤ a → rest = code.drop (a - BASE) →
    ∀ e ∈ tbl, X86Dec.imageOf code BASE e.1 = some e.2
  | [], _, _, _, _, _, e, he => by simp at he
  | (a', i, l) :: t, rest, a, h, ha, hr, e, he => by
    simp only [walkOK, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, hd⟩, ht⟩ := h
    rcases List.mem_cons.1 he with rfl | he
    · show X86Dec.imageOf code BASE _ = some (i, l)
      unfold X86Dec.imageOf
      rw [if_neg (by omega), ← hd, hr]
      simp only [List.getElem?_drop]
    · refine walkOK_sound code t _ _ ht (by omega) ?_ e he
      rw [hr, List.drop_drop]
      exact congrArg (fun k => code.drop k) (by omega)

set_option maxRecDepth 100000 in
theorem walk_core_ok : walkOK (spr1Bytes.drop (L_tripled - BASE)) L_tripled coreTable = true := by
  decide +kernel

set_option maxRecDepth 100000 in
theorem walk_init_ok : walkOK (spr1Bytes.drop (L_start - BASE)) L_start initTable = true := by
  decide +kernel

theorem table_ok : ∀ e ∈ spr1Table, im e.1 = some e.2 := by
  intro e he
  simp only [spr1Table, List.mem_append] at he
  rcases he with he | he
  · exact walkOK_sound spr1Bytes _ _ _ walk_core_ok (by decide) rfl e he
  · exact walkOK_sound spr1Bytes _ _ _ walk_init_ok (by decide) rfl e he

/-! ### Code at an address -/

def CodeAt (c : List (Nat × Ins × Nat)) : Prop := ∀ e ∈ c, im e.1 = some e.2

theorem CodeAt_nil : CodeAt [] ↔ True := by simp [CodeAt]

theorem CodeAt_cons {e : Nat × Ins × Nat} {c : List (Nat × Ins × Nat)} :
    CodeAt (e :: c) ↔ im e.1 = some e.2 ∧ CodeAt c := by
  simp [CodeAt]

theorem CodeAt_append {c d : List (Nat × Ins × Nat)} : CodeAt (c ++ d) ↔ CodeAt c ∧ CodeAt d := by
  simp only [CodeAt, List.mem_append]
  constructor
  · intro h; exact ⟨fun e he => h e (Or.inl he), fun e he => h e (Or.inr he)⟩
  · rintro ⟨h1, h2⟩ e (he | he)
    · exact h1 e he
    · exact h2 e he

/-- Every entry of `c` is an entry of `t`, both in address order. -/
def subOK : List (Nat × Ins × Nat) → List (Nat × Ins × Nat) → Bool
  | [], _ => true
  | _ :: _, [] => false
  | e :: c, f :: t => if e == f then subOK c t else subOK (e :: c) t

theorem subOK_sound : ∀ (c t : List (Nat × Ins × Nat)), subOK c t = true → ∀ e ∈ c, e ∈ t
  | [], _, _, e, he => by simp at he
  | _ :: _, [], h, _, _ => by simp [subOK] at h
  | e :: c, f :: t, h, x, hx => by
    simp only [subOK] at h
    split at h
    · rename_i hef
      have hef : e = f := by simpa using hef
      rcases List.mem_cons.1 hx with rfl | hx
      · exact List.mem_cons.2 (Or.inl hef)
      · exact List.mem_cons_of_mem _ (subOK_sound c t h x hx)
    · exact List.mem_cons_of_mem _ (subOK_sound (e :: c) t h x hx)

theorem codeAt_of_subOK {c : List (Nat × Ins × Nat)} (h : subOK c spr1Table = true) : CodeAt c :=
  fun e he => table_ok e (subOK_sound c _ h e he)

/-! ### Runs -/

theorem run_exit (c : Nat) : ∀ n, run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

theorem run_stuck : ∀ n, run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

theorem run_add : ∀ (m n : Nat) (o : Out), run im (m + n) o = run im n (run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [run, run_add m n (step im s)]
  | _ + 1, n, .exit c => by simp [run, run_exit]
  | _ + 1, n, .stuck => by simp [run, run_stuck]

/-- From `s`, some number of steps lead to a state satisfying `P`. -/
def Reaches (s : St) (P : St → Prop) : Prop := ∃ n s', run im n (.ok s) = .ok s' ∧ P s'

/-- From `s`, some number of steps lead to the outcome `o`. -/
def Ends (s : St) (P : Out → Prop) : Prop := ∃ n, P (run im n (.ok s))

theorem Reaches.refl {s : St} {P : St → Prop} (h : P s) : Reaches s P := ⟨0, s, rfl, h⟩

theorem Reaches.trans {s : St} {P Q : St → Prop} (h : Reaches s P) (k : ∀ s', P s' → Reaches s' Q) :
    Reaches s Q := by
  obtain ⟨n, s', hn, hp⟩ := h
  obtain ⟨m, s'', hm, hq⟩ := k s' hp
  exact ⟨n + m, s'', by rw [run_add, hn, hm], hq⟩

theorem Reaches.mono {s : St} {P Q : St → Prop} (h : Reaches s P) (k : ∀ s', P s' → Q s') :
    Reaches s Q := h.trans fun s' hp => Reaches.refl (k s' hp)

theorem Reaches.ends {s : St} {P : St → Prop} {Q : Out → Prop} (h : Reaches s P)
    (k : ∀ s', P s' → Ends s' Q) : Ends s Q := by
  obtain ⟨n, s', hn, hp⟩ := h
  obtain ⟨m, hm⟩ := k s' hp
  exact ⟨n + m, by rw [run_add, hn]; exact hm⟩

/-- One step through the instruction at `rip`. -/
theorem reach_step {s : St} {ins : Ins} {len : Nat} (hf : im s.rip = some (ins, len)) {P : St → Prop}
    (h : match exec ins len s with | .ok s' => Reaches s' P | _ => False) : Reaches s P := by
  revert h
  cases he : exec ins len s with
  | ok s' =>
    intro h
    obtain ⟨n, s'', hn, hp⟩ := h
    exact ⟨n + 1, s'', by rw [Nat.add_comm, run_add]; simp [run, step, hf, he, hn], hp⟩
  | exit c => intro h; exact h.elim
  | stuck => intro h; exact h.elim

theorem ends_step {s : St} {ins : Ins} {len : Nat} (hf : im s.rip = some (ins, len)) {P : Out → Prop}
    (h : match exec ins len s with | .ok s' => Ends s' P | o => P o) : Ends s P := by
  revert h
  cases he : exec ins len s with
  | ok s' =>
    intro h
    obtain ⟨n, hn⟩ := h
    exact ⟨n + 1, by rw [Nat.add_comm, run_add]; simpa [run, step, hf, he] using hn⟩
  | exit c => intro h; exact ⟨1, by simpa [run, step, hf, he] using h⟩
  | stuck => intro h; exact ⟨1, by simpa [run, step, hf, he] using h⟩

/-! ### Registers -/

theorem reg_set_eq {s : St} {i : Nat} {v : Option Nat} (h : i < s.regs.length) : (s.set i v).reg i = v := by
  simp [St.set, St.reg, h]

theorem reg_set_ne {s : St} {i j : Nat} {v : Option Nat} (h : i ≠ j) : (s.set i v).reg j = s.reg j := by
  simp [St.set, St.reg, h]

@[simp] theorem set_regs_length {s : St} {i : Nat} {v : Option Nat} :
    (s.set i v).regs.length = s.regs.length := by simp [St.set]

@[simp] theorem set_rip {s : St} {i : Nat} {v : Option Nat} : (s.set i v).rip = s.rip := rfl
@[simp] theorem set_mem {s : St} {i : Nat} {v : Option Nat} : (s.set i v).mem = s.mem := rfl
@[simp] theorem set_k {s : St} {i : Nat} {v : Option Nat} : (s.set i v).k = s.k := rfl
@[simp] theorem set_fl {s : St} {i : Nat} {v : Option Nat} : (s.set i v).fl = s.fl := rfl

/-- What a stretch of code may change: the registers in `S`, the flags and rip. -/
structure Frame (S : List Nat) (s s' : St) : Prop where
  regs : ∀ j, j ∉ S → s'.reg j = s.reg j
  len : s'.regs.length = s.regs.length
  mem : s'.mem = s.mem
  k : s'.k = s.k

theorem Frame.refl (S : List Nat) (s : St) : Frame S s s := ⟨fun _ _ => rfl, rfl, rfl, rfl⟩

theorem Frame.trans {S T : List Nat} {s s' s'' : St} (h : Frame S s s') (h' : Frame T s' s'') :
    Frame (S ++ T) s s'' :=
  ⟨fun j hj => by
      rw [h'.regs j (fun hm => hj (List.mem_append.2 (Or.inr hm))),
        h.regs j (fun hm => hj (List.mem_append.2 (Or.inl hm)))],
    by rw [h'.len, h.len], by rw [h'.mem, h.mem], by rw [h'.k, h.k]⟩

theorem Frame.mono {S T : List Nat} {s s' : St} (h : Frame S s s') (hst : ∀ x ∈ S, x ∈ T) :
    Frame T s s' :=
  ⟨fun j hj => h.regs j (fun hm => hj (hst j hm)), h.len, h.mem, h.k⟩

/-! ### Flags after a comparison of 64-bit words -/

theorem sub_mod_eq_zero {x y : Nat} (hx : x < W) (hy : y < W) : ((x + W - y) % W == 0) = (x == y) := by
  by_cases h : x = y
  · subst h; simp
  · have : (x + W - y) % W ≠ 0 := by
      by_cases hlt : x < y
      · rw [Nat.mod_eq_of_lt (by omega)]; omega
      · rw [show x + W - y = (x - y) + W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
        omega
    rw [beq_eq_false_iff_ne.2 this, beq_eq_false_iff_ne.2 h]

theorem cond_sub {x y : Nat} (hx : x < W) (hy : y < W) :
    X86Int.cond (aluF .sub x y).2 2 = some (decide (x < y)) ∧
    X86Int.cond (aluF .sub x y).2 3 = some (decide (y ≤ x)) ∧
    X86Int.cond (aluF .sub x y).2 4 = some (decide (x = y)) ∧
    X86Int.cond (aluF .sub x y).2 5 = some (decide (x ≠ y)) ∧
    X86Int.cond (aluF .sub x y).2 6 = some (decide (x ≤ y)) ∧
    X86Int.cond (aluF .sub x y).2 7 = some (decide (y < x)) := by
  have hz := sub_mod_eq_zero hx hy
  simp only [X86Int.cond, aluF, hz]
  rcases Nat.lt_trichotomy x y with h | h | h
  · simp [h, Nat.ne_of_lt h, Nat.le_of_lt h, Nat.not_le.2 h, Nat.not_lt.2 (Nat.le_of_lt h)]
  · subst h; simp
  · simp [h, Nat.ne_of_gt h, Nat.le_of_lt h, Nat.not_le.2 h, Nat.not_lt.2 (Nat.le_of_lt h)]

theorem cond_test_e {x : Nat} : X86Int.cond (aluF .and x x).2 4 = some (decide (x = 0)) := by
  by_cases h : x = 0 <;> simp [X86Int.cond, aluF, h]

theorem cond_test_ne {x : Nat} : X86Int.cond (aluF .and x x).2 5 = some (decide (x ≠ 0)) := by
  by_cases h : x = 0 <;> simp [X86Int.cond, aluF, h]

end SporosCheckBin
