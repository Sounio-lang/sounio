/-
SporosCheckArm64BinCore: machine-code image, fixed-width walk verification
(`walkOK`, `WordImg`), symbolic reachability (`Reaches`, `Ends`, `Frame`),
register/memory simplification, condition-code lemmas, and ARM64 step lemmas
for `spr1_check_arm64` over `AArch64Phys` / `AArch64Env`.
-/
import Bootstrap.SporosCheckArm64BinData
import Bootstrap.AArch64Env
import Bootstrap.AArch64Phys

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys

def BASE : Nat := 0x400000

theorem W_eq : AArch64Env.W = 2 ^ 64 := rfl
theorem pow2_64 : pow2 64 = 2 ^ 64 := rfl
theorem pow2_32 : pow2 32 = 2 ^ 32 := rfl

theorem replicate_pad_len (n : Nat) :
    (List.replicate n padWordBytes).flatten.length = 4 * n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [List.replicate_succ, List.flatten_cons, List.length_append, ih]
    have h4 : padWordBytes.length = 4 := rfl
    omega

theorem replicate_pad_getElem : ∀ (n k i : Nat), k < n → i < 4 →
    (List.replicate n padWordBytes).flatten[4 * k + i]? = padWordBytes[i]?
  | 0, _, _, hk, _ => by cases hk
  | _ + 1, 0, i, _, hi => by
    simp only [List.replicate_succ, List.flatten_cons, Nat.mul_zero, Nat.zero_add]
    exact List.getElem?_append_left (by simp [padWordBytes, wordBytes]; omega)
  | n + 1, k + 1, i, hk, hi => by
    simp only [List.replicate_succ, List.flatten_cons]
    have hge : padWordBytes.length ≤ 4 * (k + 1) + i := by simp [padWordBytes, wordBytes]; omega
    rw [List.getElem?_append_right hge]
    have hidx : 4 * (k + 1) + i - padWordBytes.length = 4 * k + i := by
      simp [padWordBytes, wordBytes]; omega
    rw [hidx]
    exact replicate_pad_getElem n k i (by omega) hi

theorem hdrBytes_len : hdrBytes.length = 176 := by decide +kernel
theorem pad0Bytes_len : pad0Bytes.length = 4176 := by unfold pad0Bytes; rw [replicate_pad_len]
theorem corePassBytes_len : corePassBytes.length = 8184 := by decide +kernel
theorem pad1Bytes_len : pad1Bytes.length = 17872 := by unfold pad1Bytes; rw [replicate_pad_len]
theorem tailBytes_len : tailBytes.length = 192 := by decide +kernel
theorem pad2Bytes_len : pad2Bytes.length = 16640 := by unfold pad2Bytes; rw [replicate_pad_len]
theorem initBytes_len : initBytes.length = 396 := by decide +kernel

theorem spr1Arm64Bytes_len : spr1Arm64Bytes.length = 47636 := by
  simp only [spr1Arm64Bytes, List.length_append, hdrBytes_len, pad0Bytes_len,
    corePassBytes_len, pad1Bytes_len, tailBytes_len, pad2Bytes_len, initBytes_len]

/-- The executable code image as `AArch64Phys.step` sees it: `spr1Arm64Bytes` loaded at `0x400000`. -/
def im : Img := imageOf spr1Arm64Bytes BASE

/-- Binary-tree cache backing `im` at depth 14 (`4 * 2^14 = 65536 >= 47636`). -/
def spr1WordImg : WordImg := WordImg.mk' spr1Arm64Bytes BASE 14

theorem im_eq_wordImg : im = spr1WordImg.toImg :=
  (WordImg.toImg_eq_imageOf spr1Arm64Bytes BASE 14 (by rw [spr1Arm64Bytes_len]; decide)).symm

theorem im_fixedWidth : im.FixedWidth :=
  imageOf_fixedWidth spr1Arm64Bytes BASE

/-! ### Fixed-width 32-bit walk verification -/

def walkOK : List Nat → Nat → List (Nat × Ins × Nat) → Bool
  | _, _, [] => true
  | b0 :: b1 :: b2 :: b3 :: rest, a, (a', i, l) :: t =>
    a == a' && l == 4 && a % 4 == 0 &&
    AArch64Phys.decodeWord (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) == some i &&
    walkOK rest (a + 4) t
  | [], _, _ :: _ => false
  | [_], _, _ :: _ => false
  | [_, _], _, _ :: _ => false
  | [_, _, _], _, _ :: _ => false

private theorem le_cons4 (b0 b1 b2 b3 : Nat) (rest : List Nat) :
    AArch64Phys.le (fun j => (b0 :: b1 :: b2 :: b3 :: rest)[j]?) 0 4 =
      some (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) := by
  simp [AArch64Phys.le, bind, Option.bind, pure]

theorem walkOK_sound (code : List Nat) :
    ∀ (tbl : List (Nat × Ins × Nat)) (rest : List Nat) (a : Nat),
      walkOK rest a tbl = true → BASE ≤ a → rest = code.drop (a - BASE) →
      ∀ e ∈ tbl, imageOf code BASE e.1 = some e.2
  | [], _, _, _, _, _, e, he => by simp at he
  | (_, _, _) :: _, [], _, h, _, _, _, _ => by simp [walkOK] at h
  | (_, _, _) :: _, [_], _, h, _, _, _, _ => by simp [walkOK] at h
  | (_, _, _) :: _, [_, _], _, h, _, _, _, _ => by simp [walkOK] at h
  | (_, _, _) :: _, [_, _, _], _, h, _, _, _, _ => by simp [walkOK] at h
  | (a', i, l) :: t, b0 :: b1 :: b2 :: b3 :: rest, a, h, ha, hr, e, he => by
    simp only [walkOK, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨rfl, rfl⟩, ha4⟩, hd⟩, ht⟩ := h
    have hlen_drop : 4 ≤ (code.drop (a - BASE)).length := by
      rw [← hr]; simp
    have hlen_code : a + 4 ≤ BASE + code.length := by
      rw [List.length_drop] at hlen_drop; omega
    rcases List.mem_cons.1 he with rfl | he
    · unfold imageOf AArch64Phys.decode
      have hc : (BASE ≤ a && a + 4 ≤ BASE + code.length && BASE % 4 == 0 && a % 4 == 0) = true := by
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
        exact ⟨⟨⟨ha, hlen_code⟩, by decide⟩, ha4⟩
      rw [if_pos hc]
      have hfun : (fun j => code[a - BASE + j]?) = (fun j => (b0 :: b1 :: b2 :: b3 :: rest)[j]?) := by
        funext j; rw [hr, List.getElem?_drop]
      rw [hfun]
      simp only [bind, Option.bind, pure, le_cons4, hd]
    · refine walkOK_sound code t rest (a + 4) ht (by omega) ?_ e he
      have hdrop : rest = (b0 :: b1 :: b2 :: b3 :: rest).drop 4 := rfl
      rw [hdrop, hr, List.drop_drop]
      exact congrArg (fun k => code.drop k) (by omega)

private theorem drop_append_left_eq {α : Type} {l1 l2 : List α} {n : Nat} (h : l1.length = n) :
    (l1 ++ l2).drop n = l2 := by
  subst h; simp

theorem walkOK_append_right : ∀ (bs rest : List Nat) (a : Nat) (tbl : List (Nat × Ins × Nat)),
    walkOK bs a tbl = true → walkOK (bs ++ rest) a tbl = true
  | _, _, _, [], _ => by simp [walkOK]
  | _ :: _ :: _ :: _ :: bs', rest, a, (_, _, _) :: t, h => by
    simp only [List.cons_append, walkOK, Bool.and_eq_true] at h ⊢
    exact ⟨h.1, walkOK_append_right bs' rest (a + 4) t h.2⟩
  | [], _, _, _ :: _, h => by simp [walkOK] at h
  | [_], _, _, _ :: _, h => by simp [walkOK] at h
  | [_, _], _, _, _ :: _, h => by simp [walkOK] at h
  | [_, _, _], _, _, _ :: _, h => by simp [walkOK] at h

theorem drop_tripled_eq :
    spr1Arm64Bytes.drop (L_tripled - BASE) =
      corePassBytes ++ (pad1Bytes ++ tailBytes ++ pad2Bytes ++ initBytes) := by
  have hsplit : spr1Arm64Bytes =
      (hdrBytes ++ pad0Bytes) ++ (corePassBytes ++ (pad1Bytes ++ tailBytes ++ pad2Bytes ++ initBytes)) := by
    simp [spr1Arm64Bytes, List.append_assoc]
  rw [hsplit]
  exact drop_append_left_eq (by simp only [List.length_append, hdrBytes_len, pad0Bytes_len]; decide)

theorem drop_vote_eq :
    spr1Arm64Bytes.drop (L_vote - BASE) = tailBytes ++ (pad2Bytes ++ initBytes) := by
  have hsplit : spr1Arm64Bytes =
      (hdrBytes ++ pad0Bytes ++ corePassBytes ++ pad1Bytes) ++ (tailBytes ++ (pad2Bytes ++ initBytes)) := by
    simp [spr1Arm64Bytes, List.append_assoc]
  rw [hsplit]
  exact drop_append_left_eq (by
    simp only [List.length_append, hdrBytes_len, pad0Bytes_len, corePassBytes_len, pad1Bytes_len]; decide)

theorem drop_start_eq :
    spr1Arm64Bytes.drop (L_start - BASE) = initBytes := by
  have hsplit : spr1Arm64Bytes =
      (hdrBytes ++ pad0Bytes ++ corePassBytes ++ pad1Bytes ++ tailBytes ++ pad2Bytes) ++ initBytes := by
    simp [spr1Arm64Bytes, List.append_assoc]
  rw [hsplit]
  exact drop_append_left_eq (by
    simp only [List.length_append, hdrBytes_len, pad0Bytes_len, corePassBytes_len,
      pad1Bytes_len, tailBytes_len, pad2Bytes_len]; decide)

theorem walk_corePass_ok : walkOK corePassBytes L_tripled corePassTable = true := by
  decide +kernel

theorem walk_tail_ok : walkOK tailBytes L_vote tailTable = true := by
  decide +kernel

theorem walk_init_ok : walkOK initBytes L_start initTable = true := by
  decide +kernel

theorem table_ok : ∀ e ∈ spr1Arm64Table, im e.1 = some e.2 := by
  intro e he
  simp only [spr1Arm64Table, List.mem_append] at he
  rcases he with (he | he) | he
  · exact walkOK_sound spr1Arm64Bytes _ _ _
      (walkOK_append_right _ _ _ _ walk_corePass_ok) (by decide) drop_tripled_eq.symm e he
  · exact walkOK_sound spr1Arm64Bytes _ _ _
      (walkOK_append_right _ _ _ _ walk_tail_ok) (by decide) drop_vote_eq.symm e he
  · exact walkOK_sound spr1Arm64Bytes _ _ _ walk_init_ok (by decide) drop_start_eq.symm e he

/-! ### Code sub-tables (`CodeAt`, `subOK`) -/

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

theorem codeAt_of_subOK {c : List (Nat × Ins × Nat)} (h : subOK c spr1Arm64Table = true) :
    CodeAt c :=
  fun e he => table_ok e (subOK_sound c _ h e he)

/-! ### Runs (`Reaches`, `Ends`) -/

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

def Reaches (s : St) (P : St → Prop) : Prop := ∃ n s', run im n (.ok s) = .ok s' ∧ P s'
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

theorem reach_step {s : St} {ins : Ins} {len : Nat} (hf : im s.pc = some (ins, len)) {P : St → Prop}
    (h : match exec ins len s with | .ok s' => Reaches s' P | _ => False) : Reaches s P := by
  revert h
  cases he : exec ins len s with
  | ok s' =>
    intro h
    obtain ⟨n, s'', hn, hp⟩ := h
    exact ⟨n + 1, s'', by rw [Nat.add_comm, run_add]; simp [run, step, hf, he, hn], hp⟩
  | exit _ => intro h; exact h.elim
  | stuck => intro h; exact h.elim

theorem ends_step {s : St} {ins : Ins} {len : Nat} (hf : im s.pc = some (ins, len)) {P : Out → Prop}
    (h : match exec ins len s with | .ok s' => Ends s' P | o => P o) : Ends s P := by
  revert h
  cases he : exec ins len s with
  | ok s' =>
    intro h
    obtain ⟨n, hn⟩ := h
    exact ⟨n + 1, by rw [Nat.add_comm, run_add]; simpa [run, step, hf, he] using hn⟩
  | exit c => intro h; exact ⟨1, by simpa [run, step, hf, he] using h⟩
  | stuck => intro h; exact ⟨1, by simpa [run, step, hf, he] using h⟩

theorem reach_at {s : St} {a : Nat} {ins : Ins} {len : Nat} (hf : im a = some (ins, len)) (ha : s.pc = a)
    {P : St → Prop} (h : match exec ins len s with | .ok s' => Reaches s' P | _ => False) :
    Reaches s P := reach_step (by rw [ha]; exact hf) h

theorem reach_x {s : St} {a : Nat} {ins : Ins} {len : Nat} {s1 : St} (f : im a = some (ins, len))
    (ha : s.pc = a) (e : exec ins len s = .ok s1) {P : St → Prop} (h : Reaches s1 P) : Reaches s P :=
  reach_step (by rw [ha]; exact f) (by rw [e]; exact h)

/-! ### Registers, zero/SP views, and `Frame` -/

theorem reg_def (s : St) (j : Nat) : s.reg j = (s.regs[j]?).getD none := rfl

@[simp] theorem setReg_regs (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).regs = s.regs.set i v := rfl
@[simp] theorem setReg_regs_length (s : St) (i : Nat) (v : Option Nat) :
    (s.setReg i v).regs.length = s.regs.length := by simp [St.setReg]
@[simp] theorem setReg_sp (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).sp = s.sp := rfl
@[simp] theorem setReg_pc (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).pc = s.pc := rfl
@[simp] theorem setReg_fl (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).fl = s.fl := rfl
@[simp] theorem setReg_mem (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).mem = s.mem := rfl
@[simp] theorem setReg_k (s : St) (i : Nat) (v : Option Nat) : (s.setReg i v).k = s.k := rfl

theorem reg_setReg_eq {s : St} {i : Nat} {v : Option Nat} (h : i < s.regs.length) :
    (s.setReg i v).reg i = v := by
  simp [St.setReg, St.reg, h]

theorem reg_setReg_ne {s : St} {i j : Nat} {v : Option Nat} (h : i ≠ j) :
    (s.setReg i v).reg j = s.reg j := by
  simp [St.setReg, St.reg, h]

@[simp] theorem wrZR_31 (s : St) (w v : Nat) : s.wrZR w 31 v = s := rfl

theorem wrZR_ne31 {s : St} {w i v : Nat} (hne : i ≠ 31) :
    s.wrZR w i v = s.setReg i (some (v % pow2 w)) := by
  simp [St.wrZR, hne]

@[simp] theorem wrZR_regs_length (s : St) (w i v : Nat) :
    (s.wrZR w i v).regs.length = s.regs.length := by
  unfold St.wrZR; split <;> simp

@[simp] theorem wrZR_sp (s : St) (w i v : Nat) : (s.wrZR w i v).sp = s.sp := by
  unfold St.wrZR; split <;> rfl
@[simp] theorem wrZR_pc (s : St) (w i v : Nat) : (s.wrZR w i v).pc = s.pc := by
  unfold St.wrZR; split <;> rfl
@[simp] theorem wrZR_fl (s : St) (w i v : Nat) : (s.wrZR w i v).fl = s.fl := by
  unfold St.wrZR; split <;> rfl
@[simp] theorem wrZR_mem (s : St) (w i v : Nat) : (s.wrZR w i v).mem = s.mem := by
  unfold St.wrZR; split <;> rfl
@[simp] theorem wrZR_k (s : St) (w i v : Nat) : (s.wrZR w i v).k = s.k := by
  unfold St.wrZR; split <;> rfl

theorem reg_wrZR_eq {s : St} {w i v : Nat} (hi : i < 31) (hl : s.regs.length = 31) :
    (s.wrZR w i v).reg i = some (v % pow2 w) := by
  have hne : i ≠ 31 := by omega
  rw [wrZR_ne31 hne, reg_setReg_eq (by omega)]

theorem reg_wrZR_ne {s : St} {w i j v : Nat} (hne : i ≠ j) :
    (s.wrZR w i v).reg j = s.reg j := by
  unfold St.wrZR; split
  · rfl
  · exact reg_setReg_ne hne

theorem wrSP_ne31 {s : St} {w i v : Nat} (hne : i ≠ 31) :
    s.wrSP w i v = s.setReg i (some (v % pow2 w)) := by
  simp [St.wrSP, hne]

theorem rdZR_31 (s : St) (w : Nat) : s.rdZR w 31 = some 0 := rfl

theorem rdZR_of_reg {s : St} {w i v : Nat} (hne : i ≠ 31) (h : s.reg i = some v) :
    s.rdZR w i = some (v % pow2 w) := by
  simp [St.rdZR, hne, h]

theorem rdSP_of_reg {s : St} {w i v : Nat} (hne : i ≠ 31) (h : s.reg i = some v) :
    s.rdSP w i = some (v % pow2 w) := by
  simp [St.rdSP, hne, h]

@[simp] theorem regs_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).regs = regs := rfl
@[simp] theorem sp_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).sp = sp := rfl
@[simp] theorem pc_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).pc = pc := rfl
@[simp] theorem fl_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).fl = fl := rfl
@[simp] theorem mem_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).mem = mem := rfl
@[simp] theorem k_mk (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags)
    (mem : Mem) (k : Kern) : (St.mk regs sp pc fl mem k).k = k := rfl
@[simp] theorem reg_mk_self (s : St) (pc : Nat) (fl : Flags) (mem : Mem) (k : Kern) (j : Nat) :
    (St.mk s.regs s.sp pc fl mem k).reg j = s.reg j := rfl

/-- Frame invariant: registers outside `S`, register count, `sp`, `mem`, and `k` are preserved. -/
structure Frame (S : List Nat) (s s' : St) : Prop where
  regs : ∀ j, j ∉ S → s'.reg j = s.reg j
  len : s'.regs.length = s.regs.length
  sp : s'.sp = s.sp
  mem : s'.mem = s.mem
  k : s'.k = s.k

theorem Frame.refl (S : List Nat) (s : St) : Frame S s s :=
  ⟨fun _ _ => rfl, rfl, rfl, rfl, rfl⟩

theorem Frame.trans {S T : List Nat} {s s' s'' : St} (h : Frame S s s') (h' : Frame T s' s'') :
    Frame (S ++ T) s s'' :=
  ⟨fun j hj => by
      rw [h'.regs j (fun hm => hj (List.mem_append.2 (Or.inr hm))),
        h.regs j (fun hm => hj (List.mem_append.2 (Or.inl hm)))],
    by rw [h'.len, h.len], by rw [h'.sp, h.sp], by rw [h'.mem, h.mem], by rw [h'.k, h.k]⟩

theorem Frame.mono {S T : List Nat} {s s' : St} (h : Frame S s s') (hst : ∀ x ∈ S, x ∈ T) :
    Frame T s s' :=
  ⟨fun j hj => h.regs j (fun hm => hj (hst j hm)), h.len, h.sp, h.mem, h.k⟩

theorem frame_regs {S : List Nat} {s s' : St} (hr : ∀ j, j ∉ S → s'.reg j = s.reg j)
    (hl : s'.regs.length = s.regs.length) (hsp : s'.sp = s.sp)
    (hm : s'.mem = s.mem) (hk : s'.k = s.k) : Frame S s s' :=
  ⟨hr, hl, hsp, hm, hk⟩

theorem Frame.wrZR {S : List Nat} {s s' : St} (h : Frame S s s') {w i v : Nat} (hi : i ∈ S) :
    Frame S s (s'.wrZR w i v) :=
  ⟨fun j hj => by
    have hne : i ≠ j := fun heq => hj (heq ▸ hi)
    rw [reg_wrZR_ne hne, h.regs j hj],
   by simp [h.len], by simp [h.sp], by simp [h.mem], by simp [h.k]⟩

theorem reach_frame {S : List Nat} {s s1 : St} {P : St → Prop} (h1 : Frame S s s1)
    (h2 : Reaches s1 (fun s2 => Frame S s1 s2 ∧ P s2)) :
    Reaches s1 (fun s2 => Frame S s s2 ∧ P s2) :=
  h2.mono fun _ ⟨hf, hp⟩ => ⟨(h1.trans hf).mono (by simp), hp⟩

theorem Frame.trans_same {S : List Nat} {s s' s'' : St} (h : Frame S s s') (h' : Frame S s' s'') :
    Frame S s s'' :=
  (h.trans h').mono (by simp)

theorem Frame.setReg {S : List Nat} {s s' : St} {i : Nat} {v : Option Nat}
    (h : Frame S s s') (hi : i ∈ S) : Frame S s (s'.setReg i v) :=
  ⟨fun j hj => by
     have hne : i ≠ j := fun heq => hj (heq ▸ hi)
     rw [reg_setReg_ne hne, h.regs j hj],
   by simp [h.len], by simp [h.sp], by simp [h.mem], by simp [h.k]⟩

@[simp] theorem getD_set_ne {α : Type} {l : List (Option α)} {i j : Nat} {v : Option α} (h : i ≠ j) :
    ((l.set i v)[j]?).getD none = (l[j]?).getD none := by
  simp [List.getElem?_set_ne h]

macro "frame_tac" : tactic =>
  `(tactic| (
    refine frame_regs (fun j hj => ?_) (by simp [*]) (by simp [*]) (by simp [*]) (by simp [*])
    repeat (first
      | rw [reg_wrZR_ne (fun h => hj (h ▸ (by decide)))]
      | rw [reg_setReg_ne (fun h => hj (h ▸ (by decide)))]
      | rw [getD_set_ne (fun h => hj (h ▸ (by decide)))])
    exact rfl))

/-! ### ARM64 arithmetic, memory, and condition-code helpers -/

def ea (a disp : Nat) : Nat := (a + disp) % AArch64Env.W

theorem ea_small {a d : Nat} (h : a + d < 2 ^ 64) : ea a d = a + d := Nat.mod_eq_of_lt h

theorem add_small {a d : Nat} (h : a + d < 2 ^ 64) : (addSubOp false 64 a d).1 = a + d :=
  Nat.mod_eq_of_lt h

theorem sub_small {a d : Nat} (ha : a < 2 ^ 64) (h : d ≤ a) : (addSubOp true 64 a d).1 = a - d := by
  unfold addSubOp; simp only [pow2_64, ↓reduceIte]
  rw [show a + 2 ^ 64 - d = (a - d) + 2 ^ 64 by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

private theorem addSub_sub_mod_ne_zero {w x y : Nat} (hx : x < pow2 w) (hy : y < pow2 w) (hne : x ≠ y) :
    (x + pow2 w - y) % pow2 w ≠ 0 := by
  by_cases hlt : x < y
  · rw [Nat.mod_eq_of_lt (by omega)]; omega
  · rw [show x + pow2 w - y = (x - y) + pow2 w by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    omega

theorem cond_sub {w x y : Nat} (hx : x < pow2 w) (hy : y < pow2 w) :
    AArch64Phys.cond (addSubOp true w x y).2 0 = some (decide (x = y)) ∧
    AArch64Phys.cond (addSubOp true w x y).2 1 = some (decide (x ≠ y)) ∧
    AArch64Phys.cond (addSubOp true w x y).2 2 = some (decide (y ≤ x)) ∧
    AArch64Phys.cond (addSubOp true w x y).2 3 = some (decide (x < y)) ∧
    AArch64Phys.cond (addSubOp true w x y).2 8 = some (decide (y < x)) ∧
    AArch64Phys.cond (addSubOp true w x y).2 9 = some (decide (x ≤ y)) := by
  simp only [AArch64Phys.cond, addSubOp, bind, Option.bind, pure]
  rcases Nat.lt_trichotomy x y with h | h | h
  · have hne : x ≠ y := Nat.ne_of_lt h
    have hznz := addSub_sub_mod_ne_zero hx hy hne
    have hle : x ≤ y := Nat.le_of_lt h
    have hge : ¬ y ≤ x := Nat.not_le.2 h
    have hgt : ¬ y < x := Nat.not_lt.2 hle
    simp [h, hne, hznz, hle, hge, hgt]
  · subst h; simp
  · have hne : x ≠ y := Nat.ne_of_gt h
    have hznz := addSub_sub_mod_ne_zero hx hy hne
    have hge : y ≤ x := Nat.le_of_lt h
    have hle : ¬ x ≤ y := Nat.not_le.2 h
    have hlt : ¬ x < y := Nat.not_lt.2 hge
    simp [h, hne, hznz, hge, hle, hlt]

theorem cond_mi_sub0 {x : Nat} (hx : x < 2 ^ 64) :
    AArch64Phys.cond (addSubOp true 64 x 0).2 4 = some (msb 64 x) := by
  have hm : (x + 2 ^ 64 - 0) % 2 ^ 64 = x := by
    rw [Nat.sub_zero, Nat.add_mod_right, Nat.mod_eq_of_lt hx]
  simp only [AArch64Phys.cond, addSubOp, pow2_64, ↓reduceIte, hm, bind, Option.bind, pure]
  rfl

theorem msb64_lt {x : Nat} (h : x < 2 ^ 63) : msb 64 x = false := by
  simp only [msb, pow2]
  have hd : x / 2 ^ (64 - 1) = 0 := Nat.div_eq_of_lt (by simpa using h)
  simp [hd]

theorem msb64_neg_enomem : msb 64 (AArch64Env.neg AArch64Env.ENOMEM) = true := by decide

/-! ### Input buffer (`Buf`) and little-endian memory reads -/

def Buf (m : Mem) (b : Nat) (bs : List Nat) : Prop :=
  ∀ i (hi : i < bs.length), m.rd (b + i) = some (bs[i]'hi)

def u32 (bs : List Nat) (pos : Nat) : Nat := leVal ((bs.drop pos).take 4)

theorem leVal_lt : ∀ (l : List Nat), (∀ x ∈ l, x < 256) → leVal l < 256 ^ l.length
  | [], _ => by simp [leVal]
  | b :: r, h => by
    have hb : b < 256 := h b List.mem_cons_self
    have hr := leVal_lt r fun x hx => h x (List.mem_cons_of_mem _ hx)
    simp only [leVal, List.length_cons]
    have hpow : 256 ^ (r.length + 1) = 256 * 256 ^ r.length := by rw [Nat.pow_succ, Nat.mul_comm]
    rw [hpow]; omega

theorem u32_lt {bs : List Nat} (hbs : ∀ x ∈ bs, x < 256) (pos : Nat) : u32 bs pos < 2 ^ 32 := by
  have hl : ((bs.drop pos).take 4).length ≤ 4 := by
    simp only [List.length_take, List.length_drop]; exact Nat.min_le_left _ _
  have hlt := leVal_lt ((bs.drop pos).take 4) fun x hx =>
    hbs x (List.mem_of_mem_drop (List.mem_of_mem_take hx))
  have hle : 256 ^ ((bs.drop pos).take 4).length ≤ 256 ^ 4 :=
    Nat.pow_le_pow_right (by decide) hl
  exact Nat.lt_of_lt_of_le hlt (Nat.le_trans hle (by decide))

private theorem drop_eq_cons {bs : List Nat} {pos : Nat} (hp : pos < bs.length) :
    bs.drop pos = bs[pos] :: bs.drop (pos + 1) := by
  apply List.ext_getElem
  · simp only [List.length_drop, List.length_cons]; omega
  · intro i hi1 hi2
    cases i with
    | zero => simp [List.getElem_drop]
    | succ k => simp [List.getElem_drop]

private theorem rdN_buf {m : Mem} {b : Nat} {bs : List Nat} (hb : Buf m b bs) :
    ∀ k pos, pos + k ≤ bs.length → m.rdN (b + pos) k = some ((bs.drop pos).take k)
  | 0, pos, _ => by simp [Mem.rdN]
  | k + 1, pos, hle => by
    have hp : pos < bs.length := by omega
    have hr := hb pos hp
    have ih := rdN_buf hb k (pos + 1) (by omega)
    have hr1 : b + pos + 1 = b + (pos + 1) := by omega
    rw [← hr1] at ih
    simp only [Mem.rdN, hr, ih, bind, Option.bind, pure]
    congr 1
    rw [drop_eq_cons hp, List.take_succ_cons]

theorem rdMem_w4_buf {m : Mem} {b : Nat} {bs : List Nat} (hb : Buf m b bs) {pos : Nat}
    (hp : pos + 4 ≤ bs.length) : rdMem m (b + pos) .w4 = some (u32 bs pos) := by
  simp only [rdMem, MemSize.bytes, rdN_buf hb 4 pos hp, Option.map_some, u32]

theorem rdMem_b1_buf {m : Mem} {b : Nat} {bs : List Nat} (hb : Buf m b bs) {pos : Nat}
    (hp : pos < bs.length) : rdMem m (b + pos) .b1 = some bs[pos] := by
  have h1 : pos + 1 ≤ bs.length := by omega
  have hr := rdN_buf hb 1 pos h1
  have htake : (bs.drop pos).take 1 = [bs[pos]] := by
    rw [drop_eq_cons hp]; rfl
  unfold rdMem MemSize.bytes
  rw [hr, htake]
  rfl

/-! ### ARM64 step rewrite lemmas and `xs` tactic -/

section
variable {s : St} {len : Nat}

theorem exec_movz_64 {rd imm16 hw : Nat} :
    exec (.movWide 2 64 rd imm16 hw) len s =
      .ok ({ s with pc := (s.pc + len) % W }.wrZR 64 rd (imm16 * pow2 (hw * 16))) := rfl

theorem exec_movn_64 {rd imm16 hw : Nat} :
    exec (.movWide 0 64 rd imm16 hw) len s =
      .ok ({ s with pc := (s.pc + len) % W }.wrZR 64 rd (pow2 64 - 1 - imm16 * pow2 (hw * 16))) := rfl

theorem xmovk {w rd imm16 hw : Nat} :
    exec (.movWide 3 w rd imm16 hw) len s =
      match s.rdZR w rd with
      | some old =>
        let sh := hw * 16
        .ok ({ s with pc := (s.pc + len) % W }.wrZR w rd
          (old % pow2 sh + imm16 * pow2 sh + old / pow2 (sh + 16) * pow2 (sh + 16)))
      | none => .stuck := by
  cases h : s.rdZR w rd <;> simp [exec, h, lift, bind, Option.bind, pure]

theorem xaddSubImm {isSub setFlags : Bool} {w rd rn imm : Nat} :
    exec (.addSubImm isSub setFlags w rd rn imm) len s =
      match s.rdSP w rn with
      | some a =>
        let (res, f) := addSubOp isSub w a (imm % pow2 w)
        let s1 := { s with pc := (s.pc + len) % W }
        .ok (if setFlags then { s1 with fl := f }.wrZR w rd res else s1.wrSP w rd res)
      | none => .stuck := by
  cases h : s.rdSP w rn <;> cases setFlags <;> simp [exec, h, lift, bind, Option.bind, pure]

theorem xaddSubReg {isSub setFlags : Bool} {w rd rn rm : Nat} {sh : ShiftKind} {amt : Nat} :
    exec (.addSubReg isSub setFlags w rd rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b =>
        let (res, f) := addSubOp isSub w a (shiftVal sh w b amt)
        let s1 := { s with pc := (s.pc + len) % W }
        .ok ((if setFlags then { s1 with fl := f } else s1).wrZR w rd res)
      | _, _ => .stuck := by
  cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xlogicReg {op : LogicOp} {inv : Bool} {w rd rn rm : Nat} {sh : ShiftKind} {amt : Nat} :
    exec (.logicReg op inv w rd rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b =>
        let bSh := shiftVal sh w b amt
        let bOp := if inv then pow2 w - 1 - bSh else bSh
        let res := match op with
          | .and | .ands => Nat.land a bOp
          | .orr => Nat.lor a bOp
          | .eor => Nat.xor a bOp
        let s1 := { s with pc := (s.pc + len) % W }
        if op == .ands then
          let f : Flags := ⟨some (msb w res), some (res == 0), some false, some false⟩
          .ok ({ s1 with fl := f }.wrZR w rd res)
        else .ok (s1.wrZR w rd res)
      | _, _ => .stuck := by
  cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> cases op <;> cases inv <;>
    simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem exec_adr {isPage : Bool} {rd imm : Nat} :
    exec (.adr isPage rd imm) len s =
      let base := if isPage then (s.pc / 4096) * 4096 else s.pc
      .ok ({ s with pc := (s.pc + len) % W }.wrZR 64 rd ((base + imm) % W)) := rfl

theorem xldr {sz : MemSize} {rt rn off : Nat} :
    exec (.ldr sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match rdMem s.mem ((base + off) % W) sz with
          | some v => .ok ({ s with pc := (s.pc + len) % W }.wrZR sz.width rt v)
          | none => .stuck
      | none => .stuck := by
  cases h1 : s.rdSP 64 rn with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some base =>
    by_cases hsp : rn = 31 ∧ base % 16 ≠ 0
    · simp only [exec, h1, lift, bind, Option.bind, pure]; simp [hsp]
    · cases h2 : rdMem s.mem ((base + off) % W) sz <;>
        simp [exec, h1, hsp, h2, lift, bind, Option.bind, pure]

theorem xldur {sz : MemSize} {rt rn off : Nat} :
    exec (.ldur sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match rdMem s.mem ((base + off) % W) sz with
          | some v => .ok ({ s with pc := (s.pc + len) % W }.wrZR sz.width rt v)
          | none => .stuck
      | none => .stuck := by
  cases h1 : s.rdSP 64 rn with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some base =>
    by_cases hsp : rn = 31 ∧ base % 16 ≠ 0
    · simp only [exec, h1, lift, bind, Option.bind, pure]; simp [hsp]
    · cases h2 : rdMem s.mem ((base + off) % W) sz <;>
        simp [exec, h1, hsp, h2, lift, bind, Option.bind, pure]

theorem xstr {sz : MemSize} {rt rn off : Nat} :
    exec (.str sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match s.rdZR sz.width rt with
          | some v =>
            match wrMem s.mem ((base + off) % W) v sz with
            | some m' => .ok { s with pc := (s.pc + len) % W, mem := m' }
            | none => .stuck
          | none => .stuck
      | none => .stuck := by
  cases h1 : s.rdSP 64 rn with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some base =>
    by_cases hsp : rn = 31 ∧ base % 16 ≠ 0
    · simp only [exec, h1, lift, bind, Option.bind, pure]; simp [hsp]
    · cases h2 : s.rdZR sz.width rt with
      | none => simp [exec, h1, hsp, h2, lift, bind, Option.bind, pure]
      | some v =>
        cases h3 : wrMem s.mem ((base + off) % W) v sz <;>
          simp [exec, h1, hsp, h2, h3, lift, bind, Option.bind, pure]

theorem exec_b_false {off : Nat} :
    exec (.b false off) len s = .ok { s with pc := (s.pc + off) % W } := rfl

theorem xbcond {cc off : Nat} :
    exec (.bcond cc off) len s =
      match AArch64Phys.cond s.fl cc with
      | some true => .ok { s with pc := (s.pc + off) % W }
      | some false => .ok { s with pc := (s.pc + len) % W }
      | none => .stuck := by
  cases h : AArch64Phys.cond s.fl cc with
  | none => simp [exec, h, lift, bind, Option.bind, pure]
  | some b => cases b <;> simp [exec, h, lift, bind, Option.bind, pure]

theorem xcbz {nz : Bool} {w rt off : Nat} :
    exec (.cbz nz w rt off) len s =
      match s.rdZR w rt with
      | some v =>
        let taken := if nz then v != 0 else v == 0
        .ok { s with pc := if taken then (s.pc + off) % W else (s.pc + len) % W }
      | none => .stuck := by
  cases h : s.rdZR w rt <;> cases nz <;> simp [exec, h, lift, bind, Option.bind, pure] <;> split <;> rfl

end

syntax "xs " term:max (" using " Lean.Parser.Tactic.simpArg,*)? : tactic
macro_rules
  | `(tactic| xs $f) =>
    `(tactic| (refine reach_at $f (by first | rfl | omega | (dsimp only; omega) | (simp only [W_eq, Nat.reducePow] at *; omega)) ?_;
               simp (config := { decide := true }) only [exec_movz_64, exec_movn_64, xmovk,
                 xaddSubImm, xaddSubReg, xlogicReg, exec_adr, xldr, xldur, xstr,
                 exec_b_false, xbcond, xcbz,
                 rdZR_31, rdZR_of_reg, rdSP_of_reg, wrZR_31, wrZR_ne31, wrSP_ne31,
                 reg_setReg_eq, reg_setReg_ne, setReg_regs, setReg_regs_length,
                 setReg_sp, setReg_pc, setReg_fl, setReg_mem, setReg_k,
                 wrZR_regs_length, wrZR_sp, wrZR_pc, wrZR_fl, wrZR_mem, wrZR_k,
                 regs_mk, sp_mk, pc_mk, fl_mk, mem_mk, k_mk, reg_mk_self,
                 shiftVal, pow2_64, pow2_32, W_eq, Nat.reducePow, Nat.reduceMod,
                 ↓reduceIte, reduceCtorEq, Bool.false_eq_true, Bool.true_eq_false,
                 decide_true, decide_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true]))
  | `(tactic| xs $f using $hs,*) =>
    `(tactic| (refine reach_at $f (by first | rfl | omega | (dsimp only; omega) | (simp only [W_eq, Nat.reducePow] at *; omega)) ?_;
               simp (config := { decide := true }) only [exec_movz_64, exec_movn_64, xmovk,
                 xaddSubImm, xaddSubReg, xlogicReg, exec_adr, xldr, xldur, xstr,
                 exec_b_false, xbcond, xcbz,
                 rdZR_31, rdZR_of_reg, rdSP_of_reg, wrZR_31, wrZR_ne31, wrSP_ne31,
                 reg_setReg_eq, reg_setReg_ne, setReg_regs, setReg_regs_length,
                 setReg_sp, setReg_pc, setReg_fl, setReg_mem, setReg_k,
                 wrZR_regs_length, wrZR_sp, wrZR_pc, wrZR_fl, wrZR_mem, wrZR_k,
                 regs_mk, sp_mk, pc_mk, fl_mk, mem_mk, k_mk, reg_mk_self,
                 shiftVal, pow2_64, pow2_32, W_eq, Nat.reducePow, Nat.reduceMod,
                 ↓reduceIte, reduceCtorEq, Bool.false_eq_true, Bool.true_eq_false,
                 decide_true, decide_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true, $hs,*]))

end SporosCheckArm64Bin
