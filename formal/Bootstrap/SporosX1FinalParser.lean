/-
Stage-1 final proof, parser phase: the parser segment triples,
the bridge to `Sporos.decode`, and the witness accept run to
exit 0.  Split out of `Bootstrap.SporosX1Final` unchanged; the
next phase is `Bootstrap.SporosX1FinalRefuse`.
-/
import Bootstrap.SporosX1FinalLoader

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-! ### Parser segment infrastructure -/

theorem rdN_length {m : Mem} : ∀ {a n : Nat} {bs : List Nat},
    m.rdN a n = some bs → bs.length = n := by
  intro a n
  induction n generalizing a with
  | zero =>
    intro bs h
    have h2 : m.rdN a 0 = some [] := rfl
    rw [h2] at h
    have hbs : bs = [] := (Option.some.inj h).symm
    rw [hbs]; rfl
  | succ n ih =>
    intro bs h
    cases bs with
    | nil =>
      exfalso
      have h1 : (do let b ← m.rd a; let r ← Mem.rdN m (a + 1) n; pure (b :: r)) = some ([] : List Nat) := h
      obtain ⟨b, hb, h2⟩ := bind_some h1
      obtain ⟨r, hr, h3⟩ := bind_some h2
      simp at h3
    | cons b bs =>
      obtain ⟨-, hr⟩ := rdN_cons h
      rw [List.length_cons, ih hr]

theorem rdN_slice {m : Mem} {a n o k : Nat} {bs : List Nat}
    (h : m.rdN a n = some bs) (hok : o + k ≤ n) :
    m.rdN (a + o) k = some ((bs.drop o).take k) := by
  have hlen : bs.length = n := rdN_length h
  have hsplit1 : bs = bs.take o ++ bs.drop o :=
    (List.take_append_drop o bs).symm
  have h1 : m.rdN a ((bs.take o).length + (bs.drop o).length) =
      some (bs.take o ++ bs.drop o) := by
    have hl : (bs.take o).length + (bs.drop o).length = n := by
      rw [← List.length_append, List.take_append_drop]; exact hlen
    rw [hl, ← hsplit1]; exact h
  obtain ⟨-, h2⟩ := rdN_append h1
  have htake : (bs.take o).length = o := by
    rw [List.length_take, hlen, Nat.min_eq_left (by omega)]
  rw [htake] at h2
  have hsplit2 : bs.drop o = (bs.drop o).take k ++ (bs.drop o).drop k :=
    (List.take_append_drop k (bs.drop o)).symm
  have h3 : m.rdN (a + o)
      (((bs.drop o).take k).length + ((bs.drop o).drop k).length) =
      some ((bs.drop o).take k ++ (bs.drop o).drop k) := by
    have hl : ((bs.drop o).take k).length + ((bs.drop o).drop k).length =
        (bs.drop o).length := by
      rw [← List.length_append, List.take_append_drop]
    rw [hl, ← hsplit2]; exact h2
  obtain ⟨h4, -⟩ := rdN_append h3
  have hklen : ((bs.drop o).take k).length = k := by
    rw [List.length_take, List.length_drop, hlen,
      Nat.min_eq_left (by omega)]
  rw [hklen] at h4
  exact h4

theorem leVal_take4_mod32 {xs : List Nat} (h8 : xs.length = 8) :
    leVal xs % 2 ^ 32 = leVal (xs.take 4) % 2 ^ 32 := by
  have hsplit : xs = xs.take 4 ++ xs.drop 4 :=
    (List.take_append_drop 4 xs).symm
  have hlen4 : (xs.take 4).length = 4 := by
    rw [List.length_take, h8]; decide
  have h256 : 256 ^ 4 = 2 ^ 32 := by decide
  have hdecomp := leVal_append (xs.take 4) (xs.drop 4)
  rw [← hsplit, hlen4, h256] at hdecomp
  rw [hdecomp, Nat.add_mul_mod_self_left]

theorem find_append_left : ∀ (m : Mem) (t : Mem) (a i : Nat)
    {j : Nat} {r : Region},
    Mem.find m a i = some (j, r) → Mem.find (m ++ t) a i = some (j, r) := by
  intro m
  induction m with
  | nil =>
    intro t a i j r h
    simp [Mem.find] at h
  | cons r0 rest ih =>
    intro t a i j r h
    have h1 : (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
        else Mem.find rest a (i + 1)) = some (j, r) := h
    by_cases hcond : r0.base ≤ a ∧ a < r0.hi
    · rw [if_pos hcond] at h1
      have hg : Mem.find ((r0 :: rest) ++ t) a i =
          (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
            else Mem.find (rest ++ t) a (i + 1)) := rfl
      rw [hg, if_pos hcond]; exact h1
    · rw [if_neg hcond] at h1
      have hg : Mem.find ((r0 :: rest) ++ t) a i =
          (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
            else Mem.find (rest ++ t) a (i + 1)) := rfl
      rw [hg, if_neg hcond]; exact ih t a (i + 1) h1

theorem rd_append_left {m : Mem} {t : Mem} {a b : Nat}
    (h : m.rd a = some b) : (m ++ t).rd a = some b := by
  have h1 : (do let x ← Mem.find m a 0; x.2.data[a - x.2.base]?) =
      some b := h
  obtain ⟨⟨j, r⟩, hf, hb⟩ := bind_some h1
  have hg : (m ++ t).rd a =
      (do let x ← Mem.find (m ++ t) a 0; x.2.data[a - x.2.base]?) := rfl
  rw [hg, find_append_left m t a 0 hf]
  exact hb

theorem rdN_append_left {m : Mem} {t : Mem} :
    ∀ {a n : Nat} {bs : List Nat},
    m.rdN a n = some bs → (m ++ t).rdN a n = some bs := by
  intro a n
  induction n generalizing a with
  | zero =>
    intro bs h
    have h2 : m.rdN a 0 = some [] := rfl
    rw [h2] at h
    have hbs : bs = [] := (Option.some.inj h).symm
    rw [hbs]; rfl
  | succ n ih =>
    intro bs h
    cases bs with
    | nil =>
      exfalso
      have h1 : (do let b ← m.rd a; let r ← Mem.rdN m (a + 1) n; pure (b :: r)) = some ([] : List Nat) := h
      obtain ⟨b, hb, h2⟩ := bind_some h1
      obtain ⟨r, hr, h3⟩ := bind_some h2
      simp at h3
    | cons b bs =>
      obtain ⟨hb, hr⟩ := rdN_cons h
      exact rdN_cons_some (rd_append_left hb) (ih hr)

theorem syscall_mmap_exact {k : Kern} {m : Mem} {a2 A : Nat}
    (ha2 : a2 ≠ 0) (hhead : k.oracle.head? = some (.at A))
    (hplace : X86Env.place m (some (.at A)) (X86Env.pages a2) =
      some (some A)) :
    X86Env.syscall k m 9 0 a2 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret A (Kern.next k).2
        (m ++ [⟨A, List.replicate (X86Env.pages a2) 0, true⟩])) := by
  have hnext1 : (Kern.next k).1 = some (.at A) := kern_next_of_head hhead
  simp only [X86Env.syscall]
  rw [if_neg (by simp), if_neg ha2]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    simp [hplace]

/-- The `RDC` body at `aRDC`, success case, with the frame the
parser composition needs (the exported `RDCSpec` post is too weak
on preserved registers): only `rax`, `rdx`, `rsp` and `rsi`
change, and memory changes only in the one stack slot of the
inner `call` to `L32`.  The exit is the `ret` instruction at
`aRDC + 0x24`, with `rsp` still decremented by the caller's
`call`. -/
theorem rdc_body_ok {c e sp : Nat} {s : X86Int.St}
    (hrip : s.rip = aRDC) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 1)
    (hr4 : s.reg 4 = some sp) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hframe : ∀ m, wr64 s.mem ((sp + W - 8) % W) (aRDC + 32) = some m →
      m.rdN c 4 = s.mem.rdN c 4) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aRDC + 0x24 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1, wr64 s.mem (sp - 8) (aRDC + 32) = some m1 ∧
        s'.mem = m1) := by
  have hc := stage1_parserCode
  have haRDC : aRDC = base + 0x271 := rfl
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- step 1: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip, haRDC]; exact hc.p271) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x274 := by
    rw [ht1]; show s.rip + 3 = base + 0x274; rw [hrip, haRDC]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- step 2: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p274) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x277 := by
    rw [ht2]; show t1.rip + 3 = base + 0x277; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have ht2fl : X86Int.cond t2.fl 3 = some true := by
    have hcf : t2.fl.cf = some false := by
      rw [ht2]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- step 3: jae 0x27e
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x27e } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p277) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x27e } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x27e := by rw [ht3]
  have ht3r2 : t3.reg 2 = some (e - c) := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).reg 2 = some (e - c)
    rw [reg_with_rip]; exact ht2r2
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  -- step 4: cmp rdx, 4
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p27e) ht3r2
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
    ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x282 := by
    rw [ht4]; show t3.rip + 4 = base + 0x282; rw [ht3rip]
  have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub (e - c) (4 % W)).2.cf = some false
      rw [h4W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show t3.reg i = s.reg i
    exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show t3.mem = s.mem
    exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]; show t3.regs.length = 16; exact ht3len
  -- step 5: jae 0x289
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x289 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p282) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x289 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x289 := by rw [ht5]
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg 6 = some c
    rw [reg_with_rip, ht4fr 6 (by decide)]; exact hr6
  have ht5r4 : t5.reg 4 = some sp := by
    have h := ht4fr 4 (by decide)
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg 4 = some sp
    rw [reg_with_rip]
    rw [h]; exact hr4
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht4fr i hi
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).mem = s.mem
    rw [mem_with_rip]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).regs.length = 16
    rw [len_with_rip]; exact ht4len
  -- step 6: mov rdx, rsi
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 2 (some c) with rip := t5.rip + 3 } :=
    step_mov (by rw [ht5rip]; exact hc.p289) ht5r6
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 2 (some c) with rip := t5.rip + 3 } := ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x28c := by
    rw [ht6]; show t5.rip + 3 = base + 0x28c; rw [ht5rip]
  have ht6r2 : t6.reg 2 = some c := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  have ht6fr : ∀ i, i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht5fr i hi
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  have ht6r4 : t6.reg 4 = some sp := by
    have h := ht6fr 4 (by decide)
    rw [h]; exact hr4
  -- step 7: call L32
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (aRDC + 32) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (aRDC + 32))
  have hm1 : wr64 t6.mem ((sp + W - 8) % W) (aRDC + 32) = some m1 := by
    rw [ht6mem]; exact hm1raw
  have hm1' : wr64 t6.mem ((sp + W - 8) % W) (t6.rip + 5) = some m1 := by
    have h32 : aRDC + 32 = t6.rip + 5 := by rw [ht6rip, haRDC]
    rw [← h32]; exact hm1
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aL32 } := by
    have h := step_call (im := stage1Im) (s := t6) (m := m1)
      (by rw [ht6rip]; exact hc.p28c) ht6r4 hm1'
    simpa using h
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aL32 } :=
    ⟨_, rfl⟩
  have ht7rip : t7.rip = aL32 := by rw [ht7]
  have ht7r2 : t7.reg 2 = some c := by
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg 2 = some c
    rw [reg_set_other (by decide)]; exact ht6r2
  have ht7fr : ∀ i, i ≠ 2 → i ≠ 4 → t7.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht6fr i hi2
  have ht7mem : t7.mem = m1 := by rw [ht7]
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht6len
  have hm1rd : m1.rdN c 4 = s.mem.rdN c 4 := by
    apply hframe
    rw [← ht6mem]; exact hm1
  -- L32 body: one ld32 step to its ret instruction
  obtain ⟨bs, hbs⟩ : ∃ bs, s.mem.rdN c 4 = some bs :=
    Option.isSome_iff_exists.mp hrd
  have hbs7 : t7.mem.rdN c 4 = some bs := by
    rw [ht7mem, hm1rd]; exact hbs
  have hrd32 : rd32 t7.mem (ea c 0) = some (leVal bs) := by
    unfold rd32 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, hbs7]; rfl
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 } :=
    step_ld32 (by rw [ht7rip]; exact stage1_l32.d0) ht7r2 hrd32
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 } := ⟨_, rfl⟩
  have hrun8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have ht8rip : t8.rip = aL32 + 2 := by
    rw [ht8]; show t7.rip + 2 = aL32 + 2; rw [ht7rip]
  have ht8r0v : t8.reg 0 = (s.mem.rdN c 4).map leVal := by
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide), hbs]; rfl
  have ht8mem : t8.mem = t7.mem := by
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).mem =
      t7.mem
    rw [mem_with_rip, mem_set]
  have ht8fr : ∀ i, i ≠ 0 → t8.reg i = t7.reg i := by
    intro i hi
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).reg i =
      t7.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 0 (some (leVal bs))).regs.length = 16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by
    have h7k : t7.k = s.k := by
      rw [ht7]
      show (t6.set 4 (some ((sp + W - 8) % W))).k = s.k
      have h6k : t6.k = s.k := by
        rw [ht6]
        show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).k = s.k
        have h5k : t5.k = s.k := by
          rw [ht5]
          show ({ t4 with rip := base + 0x289 }).k = s.k
          have h4k : t4.k = s.k := by
            rw [ht4]
            show t3.k = s.k
            rw [ht3]
            show ({ t2 with rip := base + 0x27e }).k = s.k
            exact ht2k
          exact h4k
        exact h5k
      exact h6k
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).k = s.k
    exact h7k
  -- step: ret from L32
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have hW : W = 2 ^ 64 := rfl
    have : sp + W - 8 = (sp - 8) + W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht8r4 : t8.reg 4 = some (sp - 8) := by
    have h := ht8fr 4 (by decide)
    rw [h]
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [ht6len]; decide), hsp8mod]
  have hrd64slot : rd64 t8.mem (sp - 8) = some (aRDC + 32) := by
    rw [ht8mem, ht7mem]
    have h := hm1
    rw [ht6mem, hsp8mod] at h
    exact rd64_wr64 h
  have haL32 : aL32 = base + 0x26e := rfl
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 4 (some ((sp - 8 + 8) % W)) with rip := aRDC + 32 } :=
    step_ret (im := stage1Im) (by rw [ht8rip, haL32]; exact hc.p270)
      ht8r4 hrd64slot
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 4 (some ((sp - 8 + 8) % W)) with rip := aRDC + 32 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x291 := by
    rw [ht9]; show aRDC + 32 = base + 0x291; rw [haRDC]
  have hspret : (sp - 8 + 8) % W = sp := by
    have hW : W = 2 ^ 64 := rfl
    rw [show sp - 8 + 8 = sp by omega, Nat.mod_eq_of_lt hspW]
  have ht9r4 : t9.reg 4 = some sp := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 4 = some sp
    rw [reg_set_self (by rw [ht8len]; decide), hspret]
  have ht9r6 : t9.reg 6 = some c := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]
    rw [ht8fr 6 (by decide), ht7fr 6 (by decide) (by decide)]
    exact hr6
  have ht9r0 : t9.reg 0 = (s.mem.rdN c 4).map leVal := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]
    exact ht8r0v
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi4 _
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]
    rw [ht8fr i hi0]
    exact ht7fr i hi2 hi4
  have ht9mem : t9.mem = m1 := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).mem = m1
    rw [mem_set, ht8mem, ht7mem]
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).regs.length = 16
    rw [len_set]; exact ht8len
  -- step: add rsi, 4
  have hadd4 : (aluF .add c (4 % W)).1 = c + 4 := by
    rw [h4W, aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t9.rip + 4 } :=
    step_aluI_add (by rw [ht9rip]; exact hc.p291) ht9r6
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t9.rip + 4 } := ⟨_, rfl⟩
  -- compose the run
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have hrun : run stage1Im 10 (Out.ok s) = Out.ok t10 := by
    rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, hrun8,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
  have hm1wr : wr64 s.mem (sp - 8) (aRDC + 32) = some m1 := by
    have h := hm1
    rwa [ht6mem, hsp8mod] at h
  refine ⟨10, t10, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht10]
    show t9.rip + 4 = aRDC + 0x24
    rw [ht9rip, haRDC]
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]; exact ht9r0
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 6 = some (c + 4)
    rw [reg_set_self (by rw [ht9len]; decide), hadd4]
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ht9r4
  · intro i hi0 hi2 hi4 hi6
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht9fr i hi0 hi2 hi4 hi6
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).regs.length = 16
    rw [len_set]; exact ht9len
  · intro x k hdis
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem.rdN x k =
      s.mem.rdN x k
    rw [mem_set, ht9mem]
    exact rdN_wr64_other hm1wr (by
      rcases hdis with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem.rdN c 4 =
      s.mem.rdN c 4
    rw [mem_set, ht9mem]
    exact hm1rd
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).k = s.k
    have h9k : t9.k = s.k := by
      rw [ht9]
      show (t8.set 4 (some ((sp - 8 + 8) % W))).k = s.k
      exact ht8k
    exact h9k
  · refine ⟨m1, hm1wr, ?_⟩
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem = m1
    rw [mem_set]; exact ht9mem

/-- The `RDC` body at `aRDC`, refusal case: fewer than four
bytes between cursor and end sends the run to `aFail1`, touching
only `rdx`. -/
theorem rdc_body_fail {c e : Nat} {s : X86Int.St}
    (hrip : s.rip = aRDC) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have haRDC : aRDC = base + 0x271 := rfl
  have haFail1 : aFail1 = base + 0x124 := rfl
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- step 1: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip, haRDC]; exact hc.p271) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x274 := by
    rw [ht1]; show s.rip + 3 = base + 0x274; rw [hrip, haRDC]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- step 2: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p274) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x277 := by
    rw [ht2]; show t1.rip + 3 = base + 0x277; rw [ht1rip]
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  by_cases hec : e < c
  · -- first jae not taken: fall to the jmp at 0x279
    have ht2fl : X86Int.cond t2.fl 3 = some false := by
      have hcf : t2.fl.cf = some true := by
        rw [ht2]; exact aluF_sub_cf_true hec
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e3 : X86Int.step stage1Im t2 =
        .ok { t2 with rip := base + 0x279 } := by
      have h := step_jcc (im := stage1Im) (s := t2)
        (by rw [ht2rip]; exact hc.p277) ht2fl
      simpa [ht2rip] using h
    obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
        { t2 with rip := base + 0x279 } := ⟨_, rfl⟩
    have ht3rip : t3.rip = base + 0x279 := by rw [ht3]
    have e4 : X86Int.step stage1Im t3 =
        .ok { t3 with rip := base + 0x124 } :=
      step_jmp (by rw [ht3rip]; exact hc.p279)
    obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
        { t3 with rip := base + 0x124 } := ⟨_, rfl⟩
    have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
      rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
    have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
      rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
      rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
    refine ⟨4, t4, hrun, ?_, ?_, ?_, ?_, ?_⟩
    · rw [ht4]; show base + 0x124 = aFail1; rw [haFail1]
    · intro i hi
      have h3i : t3.reg i = s.reg i := by
        rw [ht3]
        show ({ t2 with rip := base + 0x279 }).reg i = s.reg i
        rw [reg_with_rip]; exact ht2fr i hi
      rw [ht4]
      show ({ t3 with rip := base + 0x124 }).reg i = s.reg i
      rw [reg_with_rip]; exact h3i
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).mem = s.mem
      rw [mem_with_rip]; exact ht2mem
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).regs.length = 16
      rw [len_with_rip]
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).regs.length = 16
      rw [len_with_rip]; exact ht2len
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).k = s.k
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).k = s.k
      exact ht2k
  · -- c ≤ e < c + 4: second jae not taken: jmp at 0x284
    have hcle : c ≤ e := by omega
    have hsub : (aluF .sub e c).1 = e - c := by
      rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have ht2r2 : t2.reg 2 = some (e - c) := by
      rw [ht2]
      show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
      rw [reg_set_self (by rw [ht1len]; decide), hsub]
    have ht2fl : X86Int.cond t2.fl 3 = some true := by
      have hcf : t2.fl.cf = some false := by
        rw [ht2]; exact aluF_sub_cf (by omega)
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e3 : X86Int.step stage1Im t2 =
        .ok { t2 with rip := base + 0x27e } := by
      have h := step_jcc (im := stage1Im) (s := t2)
        (by rw [ht2rip]; exact hc.p277) ht2fl
      simpa using h
    obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
        { t2 with rip := base + 0x27e } := ⟨_, rfl⟩
    have ht3rip : t3.rip = base + 0x27e := by rw [ht3]
    have ht3r2 : t3.reg 2 = some (e - c) := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).reg 2 = some (e - c)
      rw [reg_with_rip]; exact ht2r2
    have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
      intro i hi
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).reg i = s.reg i
      rw [reg_with_rip]; exact ht2fr i hi
    have ht3mem : t3.mem = s.mem := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).mem = s.mem
      rw [mem_with_rip]; exact ht2mem
    have ht3len : t3.regs.length = 16 := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).regs.length = 16
      rw [len_with_rip]; exact ht2len
    have ht3k : t3.k = s.k := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).k = s.k
      exact ht2k
    -- cmp rdx, 4
    have e4 : X86Int.step stage1Im t3 =
        .ok { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
      step_cmpI (by rw [ht3rip]; exact hc.p27e) ht3r2
    obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
        { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
      ⟨_, rfl⟩
    have ht4rip : t4.rip = base + 0x282 := by
      rw [ht4]; show t3.rip + 4 = base + 0x282; rw [ht3rip]
    have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
    have ht4fl : X86Int.cond t4.fl 3 = some false := by
      have hcf : t4.fl.cf = some true := by
        rw [ht4]
        show (aluF .sub (e - c) (4 % W)).2.cf = some true
        rw [h4W]; exact aluF_sub_cf_true (by omega)
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e5 : X86Int.step stage1Im t4 =
        .ok { t4 with rip := base + 0x284 } := by
      have h := step_jcc (im := stage1Im) (s := t4)
        (by rw [ht4rip]; exact hc.p282) ht4fl
      simpa [ht4rip] using h
    obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
        { t4 with rip := base + 0x284 } := ⟨_, rfl⟩
    have ht5rip : t5.rip = base + 0x284 := by rw [ht5]
    have e6 : X86Int.step stage1Im t5 =
        .ok { t5 with rip := base + 0x124 } :=
      step_jmp (by rw [ht5rip]; exact hc.p284)
    obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
        { t5 with rip := base + 0x124 } := ⟨_, rfl⟩
    have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
      rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
    have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
      rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
    have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
      rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
    have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
      rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
    have hrun : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
      rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1,
        show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2,
        show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3,
        show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
    refine ⟨6, t6, hrun, ?_, ?_, ?_, ?_, ?_⟩
    · rw [ht6]; show base + 0x124 = aFail1; rw [haFail1]
    · intro i hi
      have h4i : t4.reg i = s.reg i := by
        rw [ht4]
        show t3.reg i = s.reg i
        exact ht3fr i hi
      have h5i : t5.reg i = s.reg i := by
        rw [ht5]
        show ({ t4 with rip := base + 0x284 }).reg i = s.reg i
        rw [reg_with_rip]; exact h4i
      rw [ht6]
      show ({ t5 with rip := base + 0x124 }).reg i = s.reg i
      rw [reg_with_rip]; exact h5i
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht4]
      show t3.mem = s.mem
      exact ht3mem
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).regs.length = 16
      rw [len_with_rip]
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).regs.length = 16
      rw [len_with_rip]
      rw [ht4]
      show t3.regs.length = 16
      exact ht3len
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).k = s.k
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).k = s.k
      rw [ht4]
      show t3.k = s.k
      exact ht3k

/-- A parser `call RDC` site, success case: from the call
instruction to the instruction after it, with `rax` the `u32` at
the cursor, the cursor advanced by four, `rsp` restored, and
memory changed only in the two stack slots `[sp-16, sp)`. -/
theorem call_rdc_ok {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5)) (hsiteW : site + 5 < W)
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hdisj : c + 4 ≤ sp - 16 ∨ sp ≤ c) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 5 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1 m2, wr64 s.mem (sp - 8) (site + 5) = some m1 ∧
        wr64 m1 (sp - 8 - 8) (aRDC + 32) = some m2 ∧ s'.mem = m2) := by
  have hc := stage1_parserCode
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hsp16mod : (sp + W - 16) % W = sp - 16 := by
    have h : sp + W - 16 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have haddr2 : (sp - 8 + W - 8) % W = sp - 16 := by
    have h : sp - 8 + W - 8 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  -- the call step
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (site + 5) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (site + 5))
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (site + 5) = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    rwa [hsp8mod] at hm1
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    step_call (im := stage1Im) (s := s) (m := m1)
      (by rw [hrip]; exact hsite) hr4 (by rw [hrip]; exact hm1)
  obtain ⟨s1, hs1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    ⟨_, rfl⟩
  have hs1rip : s1.rip = aRDC := by rw [hs1]
  have hs1r6 : s1.reg 6 = some c := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 15 = some e
    rw [reg_set_other (by decide)]; exact hr15
  have hs1r4 : s1.reg 4 = some (sp - 8) := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [hlen16]; decide), hsp8mod]
  have hs1fr : ∀ i, i ≠ 4 → s1.reg i = s.reg i := by
    intro i hi
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have hs1mem : s1.mem = m1 := by rw [hs1]
  have hs1len : s1.regs.length = 16 := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have hs1k : s1.k = s.k := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  have hm1rd : m1.rdN c 4 = s.mem.rdN c 4 := by
    apply rdN_wr64_other hm1wr
    rcases hdisj with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  have hs1rd : (s1.mem.rdN c 4).isSome := by
    rw [hs1mem, hm1rd]; exact hrd
  have hs1room : StackRoom s1 1 := by
    intro sp1 hsp1 i hi v
    have hsp1eq : sp1 = sp - 8 := by
      rw [hs1r4] at hsp1
      exact (Option.some.inj hsp1).symm
    have hi0 : i = 0 := by omega
    subst hsp1eq
    subst hi0
    have hh : (wr64 s.mem (sp - 16) v).isSome := by
      have h2 : (wr64 s.mem ((sp + W - 16) % W) v).isSome :=
        hroom sp hr4 1 (by decide) v
      rwa [hsp16mod] at h2
    have h3 : (wr64 m1 (sp - 16) v).isSome :=
      wr64_some_of_wr64 hm1wr hh
    have h4 : (sp - 8 + W - 8 * (0 + 1)) % W = sp - 16 := haddr2
    rw [hs1mem, h4]
    exact h3
  have hs1frame : ∀ m, wr64 s1.mem ((sp - 8 + W - 8) % W) (aRDC + 32) =
      some m → m.rdN c 4 = s1.mem.rdN c 4 := by
    intro m hm
    rw [haddr2] at hm
    rw [hs1mem] at hm ⊢
    apply rdN_wr64_other hm
    rcases hdisj with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  -- the body
  obtain ⟨nb, s2, hrunb, hs2rip, hs2r0, hs2r6, hs2r4, hs2fr, hs2len,
      hs2memfr, hs2rd, hs2k, hs2memwr⟩ :=
    rdc_body_ok hs1rip hs1r6 hs1r15 hce hcW heW hs1rd hs1room hs1r4
      (by omega) (by omega) hs1len hs1frame
  -- the ret at aRDC + 0x24
  have hrd64 : rd64 s2.mem (sp - 8) = some (site + 5) := by
    have h1 : s2.mem.rdN (sp - 8) 8 = s1.mem.rdN (sp - 8) 8 :=
      hs2memfr (sp - 8) 8 (Or.inr (by omega))
    show (s2.mem.rdN (sp - 8) 8).map leVal = some (site + 5)
    rw [h1, hs1mem]
    have h := rd64_wr64 hm1wr
    rw [Nat.mod_eq_of_lt hsiteW] at h
    exact h
  have e2 : X86Int.step stage1Im s2 =
      .ok { s2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 } :=
    step_ret (im := stage1Im) (by rw [hs2rip]; exact hc.p295) hs2r4 hrd64
  obtain ⟨s3, hs3⟩ : ∃ t : X86Int.St, t =
      { s2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 } :=
    ⟨_, rfl⟩
  have hspret : (sp - 8 + 8) % W = sp := by
    rw [show sp - 8 + 8 = sp by omega, Nat.mod_eq_of_lt hspW]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok s1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hs1, run_zero]
  have r2 : run stage1Im 1 (Out.ok s2) = Out.ok s3 := by
    rw [run_succ_ok stage1Im 0 s2, e2, ← hs3, run_zero]
  have hrun : run stage1Im (1 + nb + 1) (Out.ok s) = Out.ok s3 := by
    rw [show 1 + nb + 1 = 1 + (nb + 1) by omega, run_add stage1Im 1 _,
      r1, run_add stage1Im nb 1, hrunb, r2]
  have hm1fr : ∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x →
      m1.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  refine ⟨1 + nb + 1, s3, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs3]
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]
    rw [hs1mem, hm1rd] at hs2r0
    exact hs2r0
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 6 = some (c + 4)
    rw [reg_set_other (by decide)]; exact hs2r6
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 4 = some sp
    rw [reg_set_self (by rw [hs2len]; decide), hspret]
  · intro i hi0 hi2 hi4 hi6
    rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]
    rw [hs2fr i hi0 hi2 hi4 hi6]
    exact hs1fr i hi4
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).regs.length = 16
    rw [len_set]; exact hs2len
  · intro x k hdis
    rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).mem.rdN x k = s.mem.rdN x k
    rw [mem_set]
    have h1 : s2.mem.rdN x k = s1.mem.rdN x k := by
      apply hs2memfr
      rcases hdis with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    rw [h1, hs1mem]
    apply hm1fr
    rcases hdis with h | h
    · exact Or.inl (by omega)
    · exact Or.inr h
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).mem.rdN c 4 = s.mem.rdN c 4
    rw [mem_set]
    rw [hs1mem, hm1rd] at hs2rd
    exact hs2rd
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).k = s.k
    rw [← hs1k]; exact hs2k
  · obtain ⟨m2', hm2', hs2mem'⟩ := hs2memwr
    refine ⟨m1, m2', hm1wr, ?_, ?_⟩
    · rwa [hs1mem] at hm2'
    · rw [hs3]
      show (s2.set 4 (some ((sp - 8 + 8) % W))).mem = m2'
      rw [mem_set]; exact hs2mem'

/-- A parser `call RDC` site, refusal case: fewer than four
bytes left sends the run to `aFail1` with `rsp` still decremented
by the call, and memory changed only in the return slot. -/
theorem call_rdc_fail {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5))
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hroom : StackRoom s 1)
    (hr4 : s.reg 4 = some sp) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ s'.reg 4 = some (sp - 8) ∧
      (∀ i, i ≠ 2 → i ≠ 4 → s'.reg i = s.reg i) ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem := by
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (site + 5) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (site + 5))
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (site + 5) = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    rwa [hsp8mod] at hm1
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    step_call (im := stage1Im) (s := s) (m := m1)
      (by rw [hrip]; exact hsite) hr4 (by rw [hrip]; exact hm1)
  obtain ⟨s1, hs1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    ⟨_, rfl⟩
  have hs1rip : s1.rip = aRDC := by rw [hs1]
  have hs1r6 : s1.reg 6 = some c := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 15 = some e
    rw [reg_set_other (by decide)]; exact hr15
  have hs1r4 : s1.reg 4 = some (sp - 8) := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [hlen16]; decide), hsp8mod]
  have hs1fr : ∀ i, i ≠ 4 → s1.reg i = s.reg i := by
    intro i hi
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have hs1mem : s1.mem = m1 := by rw [hs1]
  have hs1len : s1.regs.length = 16 := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have hs1k : s1.k = s.k := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  obtain ⟨nb, s2, hrunb, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    rdc_body_fail hs1rip hs1r6 hs1r15 hce hcW heW hs1len
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok s1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hs1, run_zero]
  have hrun : run stage1Im (1 + nb) (Out.ok s) = Out.ok s2 := by
    rw [run_add stage1Im 1 nb, r1, hrunb]
  refine ⟨1 + nb, s2, hrun, hs2rip, ?_, ?_, ?_, hs2len, ?_, ?_⟩
  · have h := hs2fr 4 (by decide)
    rw [h]; exact hs1r4
  · intro i hi2 hi4
    rw [hs2fr i hi2]; exact hs1fr i hi4
  · intro x k hdis
    rw [hs2mem, hs1mem]
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  · rw [← hs1k]; exact hs2k
  · rw [hs2mem, hs1mem]; exact wrN_wshape _ _ _ hm1raw

/-- Parser header, first half: from `loaded` (`base + 0x134`,
`rbx` the file buffer, `r13` its length) to just after the
`SPR1` magic check (`base + 0x15b`), with `r15` the end pointer
and `rsi` the cursor at the buffer start.  Memory and kernel
are untouched. -/
theorem parser_header_a {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hmagic : leVal (bs.take 4) % 2 ^ 32 = 827478099)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x15b ∧ s'.reg 15 = some (B + L) ∧
      s'.reg 6 = some B ∧
      (∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBW : B < W := by omega
  have hbslen : bs.length = 20 := rdN_length h20
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x134: mov r15, rbx
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 15 (some B) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p134) hr3
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 15 (some B) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x137 := by
    rw [ht1]; show s.rip + 3 = base + 0x137; rw [hrip]
  have ht1r15 : t1.reg 15 = some B := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg 15 = some B
    rw [reg_with_rip, reg_set_self (hlen 15 (by decide))]
  have ht1r13 : t1.reg 13 = some L := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg 13 = some L
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr13
  have ht1fr : ∀ i, i ≠ 15 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- 0x137: add r15, r13
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } :=
    step_alu_add (by rw [ht1rip]; exact hc.p137) ht1r15 ht1r13
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x13a := by
    rw [ht2]; show t1.rip + 3 = base + 0x13a; rw [ht1rip]
  have ht2r15 : t2.reg 15 = some (B + L) := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).reg 15 = some (B + L)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt hBL]
  have ht2fr : ∀ i, i ≠ 15 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2r3 : t2.reg 3 = some B := by
    rw [ht2fr 3 (by decide)]; exact hr3
  have ht2r13 : t2.reg 13 = some L := by
    rw [ht2fr 13 (by decide)]; exact hr13
  -- 0x13a: mov rsi, rbx
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 6 (some B) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p13a) ht2r3
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 6 (some B) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x13d := by
    rw [ht3]; show t2.rip + 3 = base + 0x13d; rw [ht2rip]
  have ht3r6 : t3.reg 6 = some B := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r15 : t3.reg 15 = some (B + L) := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 15 =
      some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r15
  have ht3r13 : t3.reg 13 = some L := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 13 = some L
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r13
  have ht3fr : ∀ i, i ≠ 6 → i ≠ 15 → t3.reg i = s.reg i := by
    intro i hi6 hi15
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht2fr i hi15
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- 0x13d: cmp r13, 16
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p13d) ht3r13
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x141 := by
    rw [ht4]; show t3.rip + 4 = base + 0x141; rw [ht3rip]
  have h16W : 16 % W = 16 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub L (16 % W)).2.cf = some false
      rw [h16W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x141: jae 0x148
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x148 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p141) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x148 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x148 := by rw [ht5]
  have ht5r6 : t5.reg 6 = some B := by
    have h4r6 : t4.reg 6 = some B := by
      rw [ht4]; show t3.reg 6 = some B; exact ht3r6
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg 6 = some B
    rw [reg_with_rip]; exact h4r6
  have ht5r15 : t5.reg 15 = some (B + L) := by
    have h4r15 : t4.reg 15 = some (B + L) := by
      rw [ht4]; show t3.reg 15 = some (B + L); exact ht3r15
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg 15 = some (B + L)
    rw [reg_with_rip]; exact h4r15
  have ht5fr : ∀ i, i ≠ 6 → i ≠ 15 → t5.reg i = s.reg i := by
    intro i hi6 hi15
    have h4i : t4.reg i = s.reg i := by
      rw [ht4]; show t3.reg i = s.reg i; exact ht3fr i hi6 hi15
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg i = s.reg i
    rw [reg_with_rip]; exact h4i
  have ht5mem : t5.mem = s.mem := by
    have h4m : t4.mem = s.mem := by
      rw [ht4]; show t3.mem = s.mem; exact ht3mem
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).mem = s.mem
    rw [mem_with_rip]; exact h4m
  have ht5len : t5.regs.length = 16 := by
    have h4l : t4.regs.length = 16 := by
      rw [ht4]; show t3.regs.length = 16; exact ht3len
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).regs.length = 16
    rw [len_with_rip]; exact h4l
  -- 0x148: ld rax, [rsi]
  have hB8 : s.mem.rdN B 8 = some (bs.take 8) := by
    have h := rdN_slice h20 (show 0 + 8 ≤ 20 by decide)
    simpa using h
  have hld : rd64 t5.mem (ea B 0) = some (leVal (bs.take 8)) := by
    unfold rd64 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hBW, ht5mem, hB8]; rfl
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (leVal (bs.take 8))) with rip := t5.rip + 3 } :=
    step_ld (by rw [ht5rip]; exact hc.p148) ht5r6 hld
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (leVal (bs.take 8))) with rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x14b := by
    rw [ht6]; show t5.rip + 3 = base + 0x14b; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (leVal (bs.take 8)) := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 0 = some (leVal (bs.take 8))
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t6.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht5fr i hi6 hi15
  have ht6r6 : t6.reg 6 = some B := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r6
  have ht6r15 : t6.reg 15 = some (B + L) := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 15 = some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r15
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  -- 0x14b: mov32 rax, eax
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 } :=
    step_mov32 (by rw [ht6rip]; exact hc.p14b) ht6r0
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x14d := by
    rw [ht7]; show t6.rip + 2 = base + 0x14d; rw [ht6rip]
  have htake8len : (bs.take 8).length = 8 := by
    rw [List.length_take, hbslen]; decide
  have htk : (bs.take 8).take 4 = bs.take 4 := by
    rw [List.take_take, show min 4 8 = 4 from by decide]
  have ht7r0 : t7.reg 0 = some 827478099 := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 0 = some 827478099
    rw [reg_with_rip, reg_set_self (by rw [ht6len]; decide),
      leVal_take4_mod32 htake8len, htk, hmagic]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t7.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht6fr i hi0 hi6 hi15
  have ht7r6 : t7.reg 6 = some B := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht6r6
  have ht7r15 : t7.reg 15 = some (B + L) := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 15 = some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht6r15
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht6len
  -- 0x14d: cmp rax, "SPR1"
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7 with fl := (aluF .sub 827478099 (827478099 % W)).2, rip := t7.rip + 7 } :=
    step_cmpI (by rw [ht7rip]; exact hc.p14d) ht7r0
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7 with fl := (aluF .sub 827478099 (827478099 % W)).2, rip := t7.rip + 7 } :=
    ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x154 := by
    rw [ht8]; show t7.rip + 7 = base + 0x154; rw [ht7rip]
  have himmW : 827478099 % W = 827478099 :=
    Nat.mod_eq_of_lt (by unfold W; decide)
  have ht8fl : X86Int.cond t8.fl 4 = some true := by
    have hz : (aluF .sub 827478099 (827478099 % W)).1 = 0 := by
      rw [aluF_sub_fst, himmW,
        show 827478099 + W - 827478099 = W by omega, Nat.mod_self]
    have hzf : t8.fl.zf = some true := by
      rw [ht8]
      show (aluF .sub 827478099 (827478099 % W)).2.zf = some true
      simp [aluF_zf, hz]
    have hcc := cond_cc4 hzf
    simpa using hcc
  -- 0x154: je 0x15b
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8 with rip := base + 0x15b } := by
    have h := step_jcc (im := stage1Im) (s := t8)
      (by rw [ht8rip]; exact hc.p154) ht8fl
    simpa using h
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8 with rip := base + 0x15b } := ⟨_, rfl⟩
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have hrun : run stage1Im 9 (Out.ok s) = Out.ok t9 := by
    rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
  have ht9k : t9.k = s.k := by
    have h7k : t7.k = s.k := by
      have h6k : t6.k = s.k := by
        have h5k : t5.k = s.k := by
          have h4k : t4.k = s.k := by
            have h3k : t3.k = s.k := by
              have h2k : t2.k = s.k := by
                rw [ht2]
                show (t1.set 15 (some (aluF .add B L).1)).k = s.k
                rw [ht1]
                show ({ s.set 15 (some B) with rip := s.rip + 3 }).k = s.k
                rfl
              rw [ht3]
              show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).k = s.k
              exact h2k
            rw [ht4]
            show t3.k = s.k
            exact h3k
          rw [ht5]
          show ({ t4 with rip := base + 0x148 }).k = s.k
          exact h4k
        rw [ht6]
        show ({ t5.set 0 (some (leVal (bs.take 8))) with
          rip := t5.rip + 3 }).k = s.k
        exact h5k
      rw [ht7]
      show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 }).k = s.k
      exact h6k
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).k = s.k
    rw [ht8]
    show t7.k = s.k
    exact h7k
  refine ⟨9, t9, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht9k⟩
  · rw [ht9]
  · have h8r15 : t8.reg 15 = some (B + L) := by
      rw [ht8]; show t7.reg 15 = some (B + L); exact ht7r15
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg 15 = some (B + L)
    rw [reg_with_rip]; exact h8r15
  · have h8r6 : t8.reg 6 = some B := by
      rw [ht8]; show t7.reg 6 = some B; exact ht7r6
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg 6 = some B
    rw [reg_with_rip]; exact h8r6
  · intro i hi0 hi6 hi15
    have h8i : t8.reg i = s.reg i := by
      rw [ht8]; show t7.reg i = s.reg i; exact ht7fr i hi0 hi6 hi15
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg i = s.reg i
    rw [reg_with_rip]; exact h8i
  · rw [ht9]
    show ({ t8 with rip := base + 0x15b }).mem = s.mem
    rw [mem_with_rip, ht8]
    show t7.mem = s.mem
    exact ht7mem
  · rw [ht9]
    show ({ t8 with rip := base + 0x15b }).regs.length = 16
    rw [len_with_rip, ht8]
    show t7.regs.length = 16
    exact ht7len

/-- One header field load: `mov rdx, rsi; add rdx, off;
ld rax, [rdx]; mov32 dst, eax`, leaving in `dst` the `u32` at
`B + off` (as computed: the low 32 bits of the 8-byte load). -/
theorem field_load {site B off dst len4 : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hc1 : stage1Im site = some (.mov 2 6, 3))
    (hc2 : stage1Im (site + 3) = some (.aluI .add 2 off, 4))
    (hc3 : stage1Im (site + 7) = some (.ld 0 2 0, 3))
    (hc4 : stage1Im (site + 10) = some (.mov32 dst 0, len4))
    (hrip : s.rip = site) (hr6 : s.reg 6 = some B)
    (h20 : s.mem.rdN B 20 = some bs) (hoff : off + 8 ≤ 20)
    (hoffW : off < W) (hBoff : B + off < W)
    (hdst : dst < 16) (hdst6 : dst ≠ 6)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 10 + len4 ∧
      s'.reg dst = some (leVal ((bs.drop off).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some B ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ dst → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hbslen : bs.length = 20 := rdN_length h20
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- mov rdx, rsi
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some B) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc1) hr6
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some B) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = site + 3 := by
    rw [ht1]; show s.rip + 3 = site + 3; rw [hrip]
  have ht1r2 : t1.reg 2 = some B := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg 2 = some B
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some B := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- add rdx, off
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .add B (off % W)).1) with
        fl := (aluF .add B (off % W)).2, rip := t1.rip + 4 } :=
    step_aluI_add (by rw [ht1rip]; exact hc2) ht1r2
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .add B (off % W)).1) with
        fl := (aluF .add B (off % W)).2, rip := t1.rip + 4 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = site + 7 := by
    rw [ht2]; show t1.rip + 4 = site + 7; rw [ht1rip]
  have hoffm : off % W = off := Nat.mod_eq_of_lt hoffW
  have ht2r2 : t2.reg 2 = some (B + off) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg 2 = some (B + off)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst, hoffm,
      Nat.mod_eq_of_lt hBoff]
  have ht2r6 : t2.reg 6 = some B := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg 6 = some B
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).regs.length = 16
    rw [len_set]; exact ht1len
  -- ld rax, [rdx]
  have hslice : s.mem.rdN (B + off) 8 = some ((bs.drop off).take 8) :=
    rdN_slice h20 hoff
  have hld : rd64 t2.mem (ea (B + off) 0) =
      some (leVal ((bs.drop off).take 8)) := by
    unfold rd64 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hBoff, ht2mem, hslice]; rfl
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 } :=
    step_ld (by rw [ht2rip]; exact hc3) ht2r2 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = site + 10 := by
    rw [ht3]; show t2.rip + 3 = site + 10; rw [ht2rip]
  have ht3r0 : t3.reg 0 = some (leVal ((bs.drop off).take 8)) := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).reg 0 = some (leVal ((bs.drop off).take 8))
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht2fr i hi2
  have ht3r6 : t3.reg 6 = some B := by
    rw [ht3fr 6 (by decide) (by decide)]; exact hr6
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- mov32 dst, eax
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
        rip := t3.rip + len4 } :=
    step_mov32 (by rw [ht3rip]; exact hc4) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
        rip := t3.rip + len4 } := ⟨_, rfl⟩
  have htk8len : ((bs.drop off).take 8).length = 8 := by
    rw [List.length_take, List.length_drop, hbslen,
      Nat.min_eq_left (by omega)]
  have htk : ((bs.drop off).take 8).take 4 = (bs.drop off).take 4 := by
    rw [List.take_take, show min 4 8 = 4 from by decide]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
    rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
  have ht4k : t4.k = s.k := by
    have h3k : t3.k = s.k := by
      have h2k : t2.k = s.k := by
        rw [ht2]
        show (t1.set 2 (some (aluF .add B (off % W)).1)).k = s.k
        rw [ht1]
        show ({ s.set 2 (some B) with rip := s.rip + 3 }).k = s.k
        rfl
      rw [ht3]
      show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 }).k = s.k
      exact h2k
    rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).k = s.k
    exact h3k
  refine ⟨t4, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht4k⟩
  · rw [ht4]; show t3.rip + len4 = site + 10 + len4; rw [ht3rip]
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg dst =
      some (leVal ((bs.drop off).take 4) % 2 ^ 32)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; exact hdst),
      leVal_take4_mod32 htk8len, htk]
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg 6 = some B
    rw [reg_with_rip,
      reg_set_other (show (6 : Nat) ≠ dst from fun h => hdst6 h.symm)]
    exact ht3r6
  · intro i hi0 hi2 hid
    rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hid]; exact ht3fr i hi0 hi2
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len

/-- Parser header, second half: from `base + 0x15b` (cursor at
the buffer start) to the data-skip loop head (`base + 0x18f`),
loading `main` into `r12`, `nwords` into `r10`, `ndata` into
`rcx`, advancing the cursor past the 16-byte header, and
refusing zero `nwords` (the refusal case is separate). -/
theorem parser_header_b {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x15b) (hr6 : s.reg 6 = some B)
    (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hnw : leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 10 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      s'.reg 1 = some (leVal ((bs.drop 12).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some (B + 16) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBoff : ∀ o : Nat, o ≤ 12 → B + o < W := by
    intro o ho; omega
  -- main := u32 at B+4 (site 0x15b, exit 0x168)
  obtain ⟨s1, hrun1, hs1rip, hs1r12, hs1r6, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    field_load (site := base + 0x15b) (off := 4) (dst := 12) (len4 := 3)
      (by exact hc.p15b)
      (by exact hc.p15e) (by exact hc.p162) (by exact hc.p165)
      hrip hr6 h20 (by decide) (by unfold W; decide) (hBoff 4 (by decide))
      (by decide) (by decide) hlen16
  -- nwords := u32 at B+8 (site 0x168, exit 0x175)
  have hs1rip' : s1.rip = base + 0x168 := by
    rw [hs1rip]
  have h20_1 : s1.mem.rdN B 20 = some bs := by rw [hs1mem]; exact h20
  obtain ⟨s2, hrun2, hs2rip, hs2r10, hs2r6, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    field_load (site := base + 0x168) (off := 8) (dst := 10) (len4 := 3)
      (by exact hc.p168) (by exact hc.p16b) (by exact hc.p16f)
      (by exact hc.p172)
      hs1rip' hs1r6 h20_1 (by decide) (by unfold W; decide)
      (hBoff 8 (by decide)) (by decide) (by decide) hs1len
  -- ndata := u32 at B+12 (site 0x175, exit 0x181)
  have hs2rip' : s2.rip = base + 0x175 := by
    rw [hs2rip]
  have h20_2 : s2.mem.rdN B 20 = some bs := by rw [hs2mem]; exact h20_1
  obtain ⟨s3, hrun3, hs3rip, hs3r1, hs3r6, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    field_load (site := base + 0x175) (off := 12) (dst := 1) (len4 := 2)
      (by exact hc.p175) (by exact hc.p178) (by exact hc.p17c)
      (by exact hc.p17f)
      hs2rip' hs2r6 h20_2 (by decide) (by unfold W; decide)
      (hBoff 12 (by decide)) (by decide) (by decide) hs2len
  have hs3rip' : s3.rip = base + 0x181 := by
    rw [hs3rip]
  have hs3r12 : s3.reg 12 =
      some (leVal ((bs.drop 4).take 4) % 2 ^ 32) := by
    have h2 : s2.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) := by
      rw [hs2fr 12 (by decide) (by decide) (by decide)]; exact hs1r12
    rw [hs3fr 12 (by decide) (by decide) (by decide)]; exact h2
  have hs3r10 : s3.reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
    rw [hs3fr 10 (by decide) (by decide) (by decide)]; exact hs2r10
  -- 0x181: add rsi, 16
  have e10 : X86Int.step stage1Im s3 =
      .ok { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } :=
    step_aluI_add (by rw [hs3rip']; exact hc.p181) hs3r6
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x185 := by
    rw [ht1]; show s3.rip + 4 = base + 0x185; rw [hs3rip']
  have h16W : 16 % W = 16 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht1r6 : t1.reg 6 = some (B + 16) := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg 6 = some (B + 16)
    rw [reg_set_self (by rw [hs3len]; decide), aluF_add_fst, h16W,
      Nat.mod_eq_of_lt (by omega)]
  have ht1r10 : t1.reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32)
    rw [reg_set_other (by decide)]; exact hs3r10
  have ht1fr : ∀ i, i ≠ 6 → t1.reg i = s3.reg i := by
    intro i hi
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg i = s3.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).mem = s.mem
    rw [mem_set, hs3mem, hs2mem, hs1mem]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).regs.length = 16
    rw [len_set]; exact hs3len
  -- 0x185: test r10, r10
  have e11 : X86Int.step stage1Im t1 =
      .ok { t1 with
        fl := (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
          (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2,
        rip := t1.rip + 3 } :=
    step_test (by rw [ht1rip]; exact hc.p185) ht1r10 ht1r10
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with
        fl := (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
          (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2,
        rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x188 := by
    rw [ht2]; show t1.rip + 3 = base + 0x188; rw [ht1rip]
  have ht2fl : X86Int.cond t2.fl 5 = some true := by
    have hzf : t2.fl.zf = some false := by
      rw [ht2]
      show (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
        (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hnw]
    have hcc := cond_cc5 hzf
    simpa using hcc
  -- 0x188: jne 0x18f
  have e12 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x18f } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p188) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x18f } := ⟨_, rfl⟩
  -- compose
  have r10 : run stage1Im 1 (Out.ok s3) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s3, e10, ← ht1, run_zero]
  have r11 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e11, ← ht2, run_zero]
  have r12 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e12, ← ht3, run_zero]
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok t3 := by
    rw [show (15 : Nat) = 4 + 11 by omega, run_add stage1Im 4 11, hrun1,
      show (11 : Nat) = 4 + 7 by omega, run_add stage1Im 4 7, hrun2,
      show (7 : Nat) = 4 + 3 by omega, run_add stage1Im 4 3, hrun3,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
  have ht3k : t3.k = s.k := by
    have h2k : t2.k = s.k := by
      have h1k : t1.k = s.k := by
        rw [ht1]
        show (s3.set 6 (some (aluF .add B (16 % W)).1)).k = s.k
        exact hs3k.trans (hs2k.trans hs1k)
      rw [ht2]
      show t1.k = s.k
      exact h1k
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).k = s.k
    exact h2k
  have ht3fr' : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
      t3.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi6 hi10 hi12
    have h2i : t2.reg i = s.reg i := by
      have h1i : t1.reg i = s.reg i := by
        rw [ht1fr i hi6]
        rw [hs3fr i hi0 hi2 hi1]
        rw [hs2fr i hi0 hi2 hi10]
        exact hs1fr i hi0 hi2 hi12
      rw [ht2]
      show t1.reg i = s.reg i
      exact h1i
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).reg i = s.reg i
    rw [reg_with_rip]; exact h2i
  have ht3r : ∀ j : Nat, j = 12 ∨ j = 10 ∨ j = 1 ∨ j = 6 →
      t3.reg j = t1.reg j := by
    intro j _
    have h2j : t2.reg j = t1.reg j := by
      rw [ht2]; show t1.reg j = t1.reg j; rfl
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).reg j = t1.reg j
    rw [reg_with_rip]; exact h2j
  refine ⟨15, t3, hrun, ?_, ?_, ?_, ?_, ?_, ht3fr', ?_, ?_, ht3k⟩
  · rw [ht3]
  · rw [ht3r 12 (Or.inl rfl), ht1fr 12 (by decide)]; exact hs3r12
  · rw [ht3r 10 (Or.inr (Or.inl rfl))]; exact ht1r10
  · rw [ht3r 1 (Or.inr (Or.inr (Or.inl rfl))), ht1fr 1 (by decide)]
    exact hs3r1
  · rw [ht3r 6 (Or.inr (Or.inr (Or.inr rfl)))]; exact ht1r6
  · rw [ht3]
    show ({ t2 with rip := base + 0x18f }).mem = s.mem
    rw [mem_with_rip, ht2]
    show t1.mem = s.mem
    exact ht1mem
  · rw [ht3]
    show ({ t2 with rip := base + 0x18f }).regs.length = 16
    rw [len_with_rip, ht2]
    show t1.regs.length = 16
    exact ht1len

/-- The data segments at cursor `c` in memory: lengths `ds`,
each stored as a `u32` at the cursor followed by that many
bytes, ending at cursor `c'`.  Mirrors `decMany decData`. -/
inductive DataSeq (m : Mem) : Nat → List Nat → Nat → Prop
  | nil (c : Nat) : DataSeq m c [] c
  | cons {c d : Nat} {ds : List Nat} {c' : Nat}
      (hd : rd32 m c = some d) (hrest : DataSeq m (c + 4 + d) ds c') :
      DataSeq m c (d :: ds) c'

theorem DataSeq.le {m : Mem} {c : Nat} {ds : List Nat} {c' : Nat}
    (h : DataSeq m c ds c') : c ≤ c' := by
  induction h with
  | nil c => exact Nat.le_refl c
  | cons hd hrest ih => omega

/-- One data-loop iteration, success case: from the loop head
`base + 0x18f` back to it, consuming one length-prefixed
segment. -/
theorem parser_data_step_ok {c e d n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hnW : n < W) (hce : c + 4 ≤ e)
    (hd : rd32 s.mem c = some d) (hcd : c + 4 + d ≤ e)
    (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 16 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧ s'.reg 1 = some (n - 1) ∧
      s'.reg 6 = some (c + 4 + d) ∧ s'.reg 15 = some e ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hcW : c < W := by omega
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x18f: test rcx, rcx
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p18f) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x192 := by
    rw [ht1]; show s.rip + 3 = base + 0x192; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 4 = some false := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [show n ≠ 0 by omega]
    have hcc := cond_cc4 hzf
    simpa using hcc
  -- 0x192: jz not taken
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x194 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p192) ht1fl
    simpa [ht1rip] using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x194 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x194 := by rw [ht2]
  have ht2r15 : t2.reg 15 = some e := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg 15 = some e
    rw [reg_with_rip, ht1]
    show s.reg 15 = some e
    exact hr15
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg 6 = some c
    rw [reg_with_rip, ht1]
    show s.reg 6 = some c
    exact hr6
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    rfl
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).mem = s.mem
    rw [mem_with_rip, ht1]
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).regs.length = 16
    rw [len_with_rip, ht1]
    show s.regs.length = 16
    exact hlen16
  -- 0x194: mov rdx, r15
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 2 (some e) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p194) ht2r15
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 2 (some e) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x197 := by
    rw [ht3]; show t2.rip + 3 = base + 0x197; rw [ht2rip]
  have ht3r2 : t3.reg 2 = some e := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r6
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht2fr i
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- 0x197: sub rdx, rsi
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t3.rip + 3 } :=
    step_alu_sub (by rw [ht3rip]; exact hc.p197) ht3r2 ht3r6
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x19a := by
    rw [ht4]; show t3.rip + 3 = base + 0x19a; rw [ht3rip]
  have hsub1 : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht4r2 : t4.reg 2 = some (e - c) := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht3len]; decide), hsub1]
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht3len
  -- 0x19a: cmp rdx, 4
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t4.rip + 4 } :=
    step_cmpI (by rw [ht4rip]; exact hc.p19a) ht4r2
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t4.rip + 4 } :=
    ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x19e := by
    rw [ht5]; show t4.rip + 4 = base + 0x19e; rw [ht4rip]
  have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht5fl : X86Int.cond t5.fl 3 = some true := by
    have hcf : t5.fl.cf = some false := by
      rw [ht5]
      show (aluF .sub (e - c) (4 % W)).2.cf = some false
      rw [h4W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x19e: jae 0x1a5
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5 with rip := base + 0x1a5 } := by
    have h := step_jcc (im := stage1Im) (s := t5)
      (by rw [ht5rip]; exact hc.p19e) ht5fl
    simpa using h
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5 with rip := base + 0x1a5 } := ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x1a5 := by rw [ht6]
  have ht6r6 : t6.reg 6 = some c := by
    have h5r6 : t5.reg 6 = some c := by
      rw [ht5]; show t4.reg 6 = some c
      rw [ht4fr 6 (by decide)]; exact hr6
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).reg 6 = some c
    rw [reg_with_rip]; exact h5r6
  have ht6fr : ∀ i, i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi
    have h5i : t5.reg i = s.reg i := by
      rw [ht5]; show t4.reg i = s.reg i; exact ht4fr i hi
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).reg i = s.reg i
    rw [reg_with_rip]; exact h5i
  have ht6mem : t6.mem = s.mem := by
    have h5m : t5.mem = s.mem := by
      rw [ht5]; show t4.mem = s.mem; exact ht4mem
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).mem = s.mem
    rw [mem_with_rip]; exact h5m
  have ht6len : t6.regs.length = 16 := by
    have h5l : t5.regs.length = 16 := by
      rw [ht5]; show t4.regs.length = 16; exact ht4len
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).regs.length = 16
    rw [len_with_rip]; exact h5l
  -- 0x1a5: ld32 rax, [rsi]
  have hld : rd32 t6.mem (ea c 0) = some d := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, ht6mem]; exact hd
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some d) with rip := t6.rip + 2 } :=
    step_ld32 (by rw [ht6rip]; exact hc.p1a5) ht6r6 hld
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some d) with rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x1a7 := by
    rw [ht7]; show t6.rip + 2 = base + 0x1a7; rw [ht6rip]
  have ht7r0 : t7.reg 0 = some d := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).reg 0 = some d
    rw [reg_with_rip, reg_set_self (by rw [ht6len]; decide)]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht6fr i hi2
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7fr 6 (by decide) (by decide)]; exact hr6
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht6len
  -- 0x1a7: mov rax, rax
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 0 (some d) with rip := t7.rip + 3 } :=
    step_mov (by rw [ht7rip]; exact hc.p1a7) ht7r0
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 0 (some d) with rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x1aa := by
    rw [ht8]; show t7.rip + 3 = base + 0x1aa; rw [ht7rip]
  have ht8r0 : t8.reg 0 = some d := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).reg 0 = some d
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide)]
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht7fr i hi0 hi2
  have ht8r6 : t8.reg 6 = some c := by
    rw [ht8fr 6 (by decide) (by decide)]; exact hr6
  have ht8r15 : t8.reg 15 = some e := by
    rw [ht8fr 15 (by decide) (by decide)]; exact hr15
  have ht8mem : t8.mem = s.mem := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht7len
  -- 0x1aa: add rsi, 4
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t8.rip + 4 } :=
    step_aluI_add (by rw [ht8rip]; exact hc.p1aa) ht8r6
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t8.rip + 4 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x1ae := by
    rw [ht9]; show t8.rip + 4 = base + 0x1ae; rw [ht8rip]
  have ht9r6 : t9.reg 6 = some (c + 4) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 6 = some (c + 4)
    rw [reg_set_self (by rw [ht8len]; decide), aluF_add_fst, h4W,
      Nat.mod_eq_of_lt (by omega)]
  have ht9r0 : t9.reg 0 = some d := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9r15 : t9.reg 15 = some e := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 15 = some e
    rw [reg_set_other (by decide)]; exact ht8r15
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = s.mem := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).mem = s.mem
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).regs.length = 16
    rw [len_set]; exact ht8len
  -- 0x1ae: mov rdx, r15
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 2 (some e) with rip := t9.rip + 3 } :=
    step_mov (by rw [ht9rip]; exact hc.p1ae) ht9r15
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 2 (some e) with rip := t9.rip + 3 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x1b1 := by
    rw [ht10]; show t9.rip + 3 = base + 0x1b1; rw [ht9rip]
  have ht10r2 : t10.reg 2 = some e := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht9len]; decide)]
  have ht10r6 : t10.reg 6 = some (c + 4) := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 6 =
      some (c + 4)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht9r6
  have ht10r0 : t10.reg 0 = some d := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 0 = some d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht9r0
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t10.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht9fr i hi0 hi2 hi6
  have ht10mem : t10.mem = s.mem := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht9len
  -- 0x1b1: sub rdx, rsi
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 2 (some (aluF .sub e (c + 4)).1) with
        fl := (aluF .sub e (c + 4)).2, rip := t10.rip + 3 } :=
    step_alu_sub (by rw [ht10rip]; exact hc.p1b1) ht10r2 ht10r6
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 2 (some (aluF .sub e (c + 4)).1) with
        fl := (aluF .sub e (c + 4)).2, rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x1b4 := by
    rw [ht11]; show t10.rip + 3 = base + 0x1b4; rw [ht10rip]
  have hsub2 : (aluF .sub e (c + 4)).1 = e - (c + 4) := by
    rw [aluF_sub_fst, show e + W - (c + 4) = (e - (c + 4)) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht11r2 : t11.reg 2 = some (e - (c + 4)) := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 2 =
      some (e - (c + 4))
    rw [reg_set_self (by rw [ht10len]; decide), hsub2]
  have ht11r0 : t11.reg 0 = some d := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht10r0
  have ht11r6 : t11.reg 6 = some (c + 4) := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 6 = some (c + 4)
    rw [reg_set_other (by decide)]; exact ht10r6
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t11.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht10fr i hi0 hi2 hi6
  have ht11mem : t11.mem = s.mem := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).mem = s.mem
    rw [mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).regs.length = 16
    rw [len_set]; exact ht10len
  -- 0x1b4: sub rdx, rax
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .sub (e - (c + 4)) d).1) with
        fl := (aluF .sub (e - (c + 4)) d).2, rip := t11.rip + 3 } :=
    step_alu_sub (by rw [ht11rip]; exact hc.p1b4) ht11r2 ht11r0
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .sub (e - (c + 4)) d).1) with
        fl := (aluF .sub (e - (c + 4)) d).2, rip := t11.rip + 3 } :=
    ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x1b7 := by
    rw [ht12]; show t11.rip + 3 = base + 0x1b7; rw [ht11rip]
  have ht12fl : X86Int.cond t12.fl 3 = some true := by
    have hcf : t12.fl.cf = some false := by
      rw [ht12]
      show (aluF .sub (e - (c + 4)) d).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t12.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht11fr i hi0 hi2 hi6
  have ht12r6 : t12.reg 6 = some (c + 4) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg 6 =
      some (c + 4)
    rw [reg_set_other (by decide)]; exact ht11r6
  have ht12r0 : t12.reg 0 = some d := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht11r0
  have ht12mem : t12.mem = s.mem := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).mem = s.mem
    rw [mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).regs.length = 16
    rw [len_set]; exact ht11len
  -- 0x1b7: jae 0x1be
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12 with rip := base + 0x1be } := by
    have h := step_jcc (im := stage1Im) (s := t12)
      (by rw [ht12rip]; exact hc.p1b7) ht12fl
    simpa using h
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12 with rip := base + 0x1be } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x1be := by rw [ht13]
  have ht13r6 : t13.reg 6 = some (c + 4) := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg 6 = some (c + 4)
    rw [reg_with_rip]; exact ht12r6
  have ht13r0 : t13.reg 0 = some d := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg 0 = some d
    rw [reg_with_rip]; exact ht12r0
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t13.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg i = s.reg i
    rw [reg_with_rip]; exact ht12fr i hi0 hi2 hi6
  have ht13mem : t13.mem = s.mem := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).mem = s.mem
    rw [mem_with_rip]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).regs.length = 16
    rw [len_with_rip]; exact ht12len
  -- 0x1be: add rsi, rax
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13.set 6 (some (aluF .add (c + 4) d).1) with
        fl := (aluF .add (c + 4) d).2, rip := t13.rip + 3 } :=
    step_alu_add (by rw [ht13rip]; exact hc.p1be) ht13r6 ht13r0
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13.set 6 (some (aluF .add (c + 4) d).1) with
        fl := (aluF .add (c + 4) d).2, rip := t13.rip + 3 } := ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x1c1 := by
    rw [ht14]; show t13.rip + 3 = base + 0x1c1; rw [ht13rip]
  have ht14r6 : t14.reg 6 = some (c + 4 + d) := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg 6 =
      some (c + 4 + d)
    rw [reg_set_self (by rw [ht13len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  have ht14r1 : t14.reg 1 = some n := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg 1 = some n
    rw [reg_set_other (by decide), ht13fr 1 (by decide) (by decide)
      (by decide)]
    exact hr1
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t14.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht13fr i hi0 hi2 hi6
  have ht14mem : t14.mem = s.mem := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).mem = s.mem
    rw [mem_set]; exact ht13mem
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).regs.length = 16
    rw [len_set]; exact ht13len
  -- 0x1c1: sub rcx, 1
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14.set 1 (some (aluF .sub n (1 % W)).1) with
        fl := (aluF .sub n (1 % W)).2, rip := t14.rip + 4 } :=
    step_aluI_sub (by rw [ht14rip]; exact hc.p1c1) ht14r1
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14.set 1 (some (aluF .sub n (1 % W)).1) with
        fl := (aluF .sub n (1 % W)).2, rip := t14.rip + 4 } := ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x1c5 := by
    rw [ht15]; show t14.rip + 4 = base + 0x1c5; rw [ht14rip]
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht15r1 : t15.reg 1 = some (n - 1) := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg 1 = some (n - 1)
    rw [reg_set_self (by rw [ht14len]; decide), aluF_sub_fst, h1W,
      show n + W - 1 = (n - 1) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 →
      t15.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi6
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi1]; exact ht14fr i hi0 hi2 hi6
  have ht15r6 : t15.reg 6 = some (c + 4 + d) := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg 6 =
      some (c + 4 + d)
    rw [reg_set_other (by decide)]; exact ht14r6
  have ht15r15 : t15.reg 15 = some e := by
    rw [ht15fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have ht15mem : t15.mem = s.mem := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).regs.length = 16
    rw [len_set]; exact ht14len
  -- 0x1c5: jmp 0x18f
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15 with rip := base + 0x18f } :=
    step_jmp (by rw [ht15rip]; exact hc.p1c5)
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15 with rip := base + 0x18f } := ⟨_, rfl⟩
  -- compose (16 steps)
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  have hrun : run stage1Im 16 (Out.ok s) = Out.ok t16 := by
    rw [show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r1,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r2,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r3,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r4,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r5,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r6,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r7,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r8,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r9,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r10,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r11,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r12,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r13,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r14,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r15, r16]
  have ht16k : t16.k = s.k := by
    have h15k : t15.k = s.k := by
      have h14k : t14.k = s.k := by
        have h13k : t13.k = s.k := by
          have h12k : t12.k = s.k := by
            have h11k : t11.k = s.k := by
              have h10k : t10.k = s.k := by
                have h9k : t9.k = s.k := by
                  have h8k : t8.k = s.k := by
                    have h7k : t7.k = s.k := by
                      have h6k : t6.k = s.k := by
                        have h5k : t5.k = s.k := by
                          have h4k : t4.k = s.k := by
                            have h3k : t3.k = s.k := by
                              have h2k : t2.k = s.k := by
                                rw [ht2]
                                show ({ t1 with rip := base + 0x194 }).k =
                                  s.k
                                rw [ht1]
                              rw [ht3]
                              show ({ t2.set 2 (some e) with
                                rip := t2.rip + 3 }).k = s.k
                              exact h2k
                            rw [ht4]
                            show (t3.set 2
                              (some (aluF .sub e c).1)).k = s.k
                            exact h3k
                          rw [ht5]
                          show t4.k = s.k
                          exact h4k
                        rw [ht6]
                        show ({ t5 with rip := base + 0x1a5 }).k = s.k
                        exact h5k
                      rw [ht7]
                      show ({ t6.set 0 (some d) with
                        rip := t6.rip + 2 }).k = s.k
                      exact h6k
                    rw [ht8]
                    show ({ t7.set 0 (some d) with
                      rip := t7.rip + 3 }).k = s.k
                    exact h7k
                  rw [ht9]
                  show (t8.set 6 (some (aluF .add c (4 % W)).1)).k = s.k
                  exact h8k
                rw [ht10]
                show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).k =
                  s.k
                exact h9k
              rw [ht11]
              show (t10.set 2 (some (aluF .sub e (c + 4)).1)).k = s.k
              exact h10k
            rw [ht12]
            show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).k = s.k
            exact h11k
          rw [ht13]
          show ({ t12 with rip := base + 0x1be }).k = s.k
          exact h12k
        rw [ht14]
        show (t13.set 6 (some (aluF .add (c + 4) d).1)).k = s.k
        exact h13k
      rw [ht15]
      show (t14.set 1 (some (aluF .sub n (1 % W)).1)).k = s.k
      exact h14k
    rw [ht16]
    show ({ t15 with rip := base + 0x18f }).k = s.k
    exact h15k
  refine ⟨t16, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ht16k⟩
  · rw [ht16]
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 1 = some (n - 1)
    rw [reg_with_rip]; exact ht15r1
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 6 = some (c + 4 + d)
    rw [reg_with_rip]; exact ht15r6
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 15 = some e
    rw [reg_with_rip]; exact ht15r15
  · intro i hi0 hi1 hi2 hi6
    rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg i = s.reg i
    rw [reg_with_rip]; exact ht15fr i hi0 hi1 hi2 hi6
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).mem = s.mem
    rw [mem_with_rip]; exact ht15mem
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).regs.length = 16
    rw [len_with_rip]; exact ht15len

/-- The whole data-skip loop, success case: from the loop head
with `ds.length` segments described by `DataSeq` to
`base + 0x1c7` (`data_done`), cursor at the end of the last
segment. -/
theorem parser_data_loop_ok {e : Nat} :
    ∀ {c : Nat} {ds : List Nat} {c' : Nat} {s : X86Int.St},
      DataSeq s.mem c ds c' → s.rip = base + 0x18f →
      s.reg 1 = some ds.length → s.reg 6 = some c →
      s.reg 15 = some e → c' ≤ e → e < W → ds.length < W →
      s.regs.length = 16 →
      ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
        s'.rip = base + 0x1c7 ∧ s'.reg 1 = some 0 ∧
        s'.reg 6 = some c' ∧ s'.reg 15 = some e ∧
        (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
        s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  intro c ds
  induction ds generalizing c with
  | nil =>
    intro c' s hseq hrip hr1 hr6 hr15 hc'e heW hlenW hlen16
    cases hseq
    have hc := stage1_parserCode
    have hr1' : s.reg 1 = some 0 := hr1
    -- test rcx, rcx (zero)
    have e1 : X86Int.step stage1Im s =
        .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
      step_test (by rw [hrip]; exact hc.p18f) hr1' hr1'
    obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
        { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
    have ht1rip : t1.rip = base + 0x192 := by
      rw [ht1]; show s.rip + 3 = base + 0x192; rw [hrip]
    have ht1fl : X86Int.cond t1.fl 4 = some true := by
      have hzf : t1.fl.zf = some true := by
        rw [ht1]
        show (aluF .and 0 0).2.zf = some true
        simp [aluF_zf, aluF_and_fst_self]
      have hcc := cond_cc4 hzf
      simpa using hcc
    have e2 : X86Int.step stage1Im t1 =
        .ok { t1 with rip := base + 0x1c7 } := by
      have h := step_jcc (im := stage1Im) (s := t1)
        (by rw [ht1rip]; exact hc.p192) ht1fl
      simpa using h
    obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
        { t1 with rip := base + 0x1c7 } := ⟨_, rfl⟩
    have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
      rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
    have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
      rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
    have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
      rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
    have ht2k : t2.k = s.k := by
      rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).k = s.k
      rw [ht1]
    refine ⟨2, t2, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ht2k⟩
    · rw [ht2]
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 1 = some 0
      rw [reg_with_rip, ht1]
      show s.reg 1 = some 0
      exact hr1'
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 6 = some c
      rw [reg_with_rip, ht1]
      show s.reg 6 = some c
      exact hr6
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 15 = some e
      rw [reg_with_rip, ht1]
      show s.reg 15 = some e
      exact hr15
    · intro i _ _ _ _
      rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg i = s.reg i
      rw [reg_with_rip, ht1]
      rfl
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).mem = s.mem
      rw [mem_with_rip, ht1]
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).regs.length = 16
      rw [len_with_rip, ht1]
      show s.regs.length = 16
      exact hlen16
  | cons d ds ih =>
    intro c' s hseq hrip hr1 hr6 hr15 hc'e heW hlenW hlen16
    cases hseq with
    | cons hd hrest =>
      have hle : c + 4 + d ≤ e := Nat.le_trans (DataSeq.le hrest) hc'e
      have hlencons : (d :: ds).length = ds.length + 1 :=
        List.length_cons
      obtain ⟨s1, hrun1, hs1rip, hs1r1, hs1r6, hs1r15, hs1fr, hs1mem,
          hs1len, hs1k⟩ :=
        parser_data_step_ok hrip hr1 hr6 hr15
          (by rw [hlencons]; omega) hlenW
          (by omega) hd hle heW hlen16
      have hs1r1' : s1.reg 1 = some ds.length := by
        rw [hlencons] at hs1r1
        simpa using hs1r1
      have hseq1 : DataSeq s1.mem (c + 4 + d) ds c' := by
        rw [hs1mem]; exact hrest
      obtain ⟨n2, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r15, hs2fr,
          hs2mem, hs2len, hs2k⟩ :=
        ih hseq1 hs1rip hs1r1' hs1r6 hs1r15 hc'e heW
          (by rw [hlencons] at hlenW; omega) hs1len
      have hrun : run stage1Im (16 + n2) (Out.ok s) = Out.ok s2 := by
        rw [run_add stage1Im 16 n2, hrun1, hrun2]
      exact ⟨16 + n2, s2, hrun, hs2rip, hs2r1, hs2r6, hs2r15,
        fun i hi0 hi1 hi2 hi6 =>
          (hs2fr i hi0 hi1 hi2 hi6).trans (hs1fr i hi0 hi1 hi2 hi6),
        hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k⟩

/-- The `nwords * 20` guard, success case: from `data_done`
(`base + 0x1c7`, `r10 = nwords`, cursor `rsi`, end `r15`) to
`base + 0x1ec`, when twenty bytes per word fit in what remains.
Only `rax` and `rdx` change. -/
theorem parser_guard_ok {c e nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1c7) (hr10 : s.reg 10 = some nw)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hnw32 : nw < 2 ^ 32) (hce : c ≤ e) (heW : e < W)
    (hguard : 20 * nw ≤ e - c) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x1ec ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h2 : 2 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h4 : 4 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h8 : 8 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h16 : 16 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h20 : 20 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  -- 0x1c7: mov rax, r10
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some nw) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p1c7) hr10
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some nw) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x1ca := by
    rw [ht1]; show s.rip + 3 = base + 0x1ca; rw [hrip]
  have ht1r0 : t1.reg 0 = some nw := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).reg 0 = some nw
    rw [reg_with_rip, reg_set_self (hlen 0 (by decide))]
  have ht1fr : ∀ i, i ≠ 0 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- 0x1ca: add rax, rax (2*nw)
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 0 (some (aluF .add nw nw).1) with
        fl := (aluF .add nw nw).2, rip := t1.rip + 3 } :=
    step_alu_add (by rw [ht1rip]; exact hc.p1ca) ht1r0 ht1r0
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 0 (some (aluF .add nw nw).1) with
        fl := (aluF .add nw nw).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x1cd := by
    rw [ht2]; show t1.rip + 3 = base + 0x1cd; rw [ht1rip]
  have ht2r0 : t2.reg 0 = some (2 * nw) := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).reg 0 = some (2 * nw)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : nw + nw < W)]
    congr 1; omega
  have ht2fr : ∀ i, i ≠ 0 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).regs.length = 16
    rw [len_set]; exact ht1len
  -- 0x1cd: add rax, rax (4*nw)
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1) with
        fl := (aluF .add (2 * nw) (2 * nw)).2, rip := t2.rip + 3 } :=
    step_alu_add (by rw [ht2rip]; exact hc.p1cd) ht2r0 ht2r0
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1) with
        fl := (aluF .add (2 * nw) (2 * nw)).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x1d0 := by
    rw [ht3]; show t2.rip + 3 = base + 0x1d0; rw [ht2rip]
  have ht3r0 : t3.reg 0 = some (4 * nw) := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).reg 0 =
      some (4 * nw)
    rw [reg_set_self (by rw [ht2len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 2 * nw + 2 * nw < W)]
    congr 1; omega
  have ht3fr : ∀ i, i ≠ 0 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht2len
  -- 0x1d0: mov rdx, rax (4*nw saved)
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 } :=
    step_mov (by rw [ht3rip]; exact hc.p1d0) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x1d3 := by
    rw [ht4]; show t3.rip + 3 = base + 0x1d3; rw [ht3rip]
  have ht4r2 : t4.reg 2 = some (4 * nw) := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg 2 =
      some (4 * nw)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r0 : t4.reg 0 = some (4 * nw) := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg 0 =
      some (4 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht3r0
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht3fr i hi0
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with
      rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  -- 0x1d3: add rax, rax (8*nw)
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1) with
        fl := (aluF .add (4 * nw) (4 * nw)).2, rip := t4.rip + 3 } :=
    step_alu_add (by rw [ht4rip]; exact hc.p1d3) ht4r0 ht4r0
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1) with
        fl := (aluF .add (4 * nw) (4 * nw)).2, rip := t4.rip + 3 } :=
    ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x1d6 := by
    rw [ht5]; show t4.rip + 3 = base + 0x1d6; rw [ht4rip]
  have ht5r0 : t5.reg 0 = some (8 * nw) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg 0 =
      some (8 * nw)
    rw [reg_set_self (by rw [ht4len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 4 * nw + 4 * nw < W)]
    congr 1; omega
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht4fr i hi0 hi2
  have ht5r2 : t5.reg 2 = some (4 * nw) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg 2 =
      some (4 * nw)
    rw [reg_set_other (by decide)]; exact ht4r2
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht4len
  -- 0x1d6: add rax, rax (16*nw)
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1) with
        fl := (aluF .add (8 * nw) (8 * nw)).2, rip := t5.rip + 3 } :=
    step_alu_add (by rw [ht5rip]; exact hc.p1d6) ht5r0 ht5r0
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1) with
        fl := (aluF .add (8 * nw) (8 * nw)).2, rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x1d9 := by
    rw [ht6]; show t5.rip + 3 = base + 0x1d9; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (16 * nw) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg 0 =
      some (16 * nw)
    rw [reg_set_self (by rw [ht5len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 8 * nw + 8 * nw < W)]
    congr 1; omega
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht5fr i hi0 hi2
  have ht6r2 : t6.reg 2 = some (4 * nw) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg 2 =
      some (4 * nw)
    rw [reg_set_other (by decide)]; exact ht5r2
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht5len
  -- 0x1d9: add rax, rdx (20*nw)
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1) with
        fl := (aluF .add (16 * nw) (4 * nw)).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p1d9) ht6r0 ht6r2
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1) with
        fl := (aluF .add (16 * nw) (4 * nw)).2, rip := t6.rip + 3 } :=
    ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x1dc := by
    rw [ht7]; show t6.rip + 3 = base + 0x1dc; rw [ht6rip]
  have ht7r0 : t7.reg 0 = some (20 * nw) := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).reg 0 =
      some (20 * nw)
    rw [reg_set_self (by rw [ht6len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 16 * nw + 4 * nw < W)]
    congr 1; omega
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht6fr i hi0 hi2
  have ht7r15 : t7.reg 15 = some e := by
    rw [ht7fr 15 (by decide) (by decide)]; exact hr15
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7fr 6 (by decide) (by decide)]; exact hr6
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht6len
  -- 0x1dc: mov rdx, r15
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 2 (some e) with rip := t7.rip + 3 } :=
    step_mov (by rw [ht7rip]; exact hc.p1dc) ht7r15
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 2 (some e) with rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x1df := by
    rw [ht8]; show t7.rip + 3 = base + 0x1df; rw [ht7rip]
  have ht8r2 : t8.reg 2 = some e := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide)]
  have ht8r6 : t8.reg 6 = some c := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht7r6
  have ht8r0 : t8.reg 0 = some (20 * nw) := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 0 =
      some (20 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht7r0
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht7fr i hi0 hi2
  have ht8mem : t8.mem = s.mem := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht7len
  -- 0x1df: sub rdx, rsi (remaining)
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t8.rip + 3 } :=
    step_alu_sub (by rw [ht8rip]; exact hc.p1df) ht8r2 ht8r6
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t8.rip + 3 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x1e2 := by
    rw [ht9]; show t8.rip + 3 = base + 0x1e2; rw [ht8rip]
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht9r2 : t9.reg 2 = some (e - c) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht8len]; decide), hsub]
  have ht9r0 : t9.reg 0 = some (20 * nw) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg 0 = some (20 * nw)
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → t9.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = s.mem := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht8len
  -- 0x1e2: cmp rax, rdx
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with fl := (aluF .sub (20 * nw) (e - c)).2, rip := t9.rip + 3 } :=
    step_cmp (by rw [ht9rip]; exact hc.p1e2) ht9r0 ht9r2
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with fl := (aluF .sub (20 * nw) (e - c)).2, rip := t9.rip + 3 } :=
    ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x1e5 := by
    rw [ht10]; show t9.rip + 3 = base + 0x1e5; rw [ht9rip]
  have ht10fl : X86Int.cond t10.fl 6 = some true := by
    by_cases hlt : 20 * nw < e - c
    · have hcf : t10.fl.cf = some true := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.cf = some true
        exact aluF_sub_cf_true hlt
      have hzf : t10.fl.zf =
          some ((aluF .sub (20 * nw) (e - c)).1 == 0) := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.zf =
          some ((aluF .sub (20 * nw) (e - c)).1 == 0)
        exact aluF_zf .sub _ _
      simp [X86Int.cond, hcf, hzf]
    · have heq : 20 * nw = e - c := by omega
      have hzval : (aluF .sub (20 * nw) (e - c)).1 = 0 := by
        rw [aluF_sub_fst,
          show 20 * nw + W - (e - c) = W by omega, Nat.mod_self]
      have hzf : t10.fl.zf = some true := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.zf = some true
        simp [aluF_zf, hzval]
      have hcf : t10.fl.cf = some false := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.cf = some false
        exact aluF_sub_cf hlt
      simp [X86Int.cond, hcf, hzf]
  -- 0x1e5: jbe 0x1ec
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x1ec } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p1e5) ht10fl
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x1ec } := ⟨_, rfl⟩
  -- compose (11 steps)
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
  have ht11k : t11.k = s.k := by
    have h10k : t10.k = s.k := by
      have h9k : t9.k = s.k := by
        have h8k : t8.k = s.k := by
          have h7k : t7.k = s.k := by
            have h6k : t6.k = s.k := by
              have h5k : t5.k = s.k := by
                have h4k : t4.k = s.k := by
                  have h3k : t3.k = s.k := by
                    have h2k : t2.k = s.k := by
                      rw [ht2]
                      show (t1.set 0 (some (aluF .add nw nw).1)).k = s.k
                      rw [ht1]
                      show ({ s.set 0 (some nw) with
                        rip := s.rip + 3 }).k = s.k
                      rfl
                    rw [ht3]
                    show (t2.set 0
                      (some (aluF .add (2 * nw) (2 * nw)).1)).k = s.k
                    exact h2k
                  rw [ht4]
                  show ({ t3.set 2 (some (4 * nw)) with
                    rip := t3.rip + 3 }).k = s.k
                  exact h3k
                rw [ht5]
                show (t4.set 0
                  (some (aluF .add (4 * nw) (4 * nw)).1)).k = s.k
                exact h4k
              rw [ht6]
              show (t5.set 0
                (some (aluF .add (8 * nw) (8 * nw)).1)).k = s.k
              exact h5k
            rw [ht7]
            show (t6.set 0
              (some (aluF .add (16 * nw) (4 * nw)).1)).k = s.k
            exact h6k
          rw [ht8]
          show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).k = s.k
          exact h7k
        rw [ht9]
        show (t8.set 2 (some (aluF .sub e c).1)).k = s.k
        exact h8k
      rw [ht10]
      show t9.k = s.k
      exact h9k
    rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).k = s.k
    exact h10k
  refine ⟨t11, hrun, ?_, ?_, ?_, ?_, ht11k⟩
  · rw [ht11]
  · intro i hi0 hi2
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i; exact ht9fr i hi0 hi2
    rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).reg i = s.reg i
    rw [reg_with_rip]; exact h10i
  · rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).mem = s.mem
    rw [mem_with_rip, ht10]
    show t9.mem = s.mem
    exact ht9mem
  · rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).regs.length = 16
    rw [len_with_rip, ht10]
    show t9.regs.length = 16
    exact ht9len

/-- The words-table `mmap`, success case: from `base + 0x1ec`
(cursor in `rsi`, `nwords` in `r10`) to `base + 0x22d`, saving
the cursor on the stack, asking the oracle for `64 * nwords`
bytes at `T`, and leaving the table base in `r14` with `rsp`
decremented by the saved cursor slot. -/
theorem parser_mmap1_ok {c nw sp T : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hhead : s.k.oracle.head? = some (.at T))
    (hTpage : T % page = 0) (hTpos : 0 < T)
    (hTbound : T + X86Env.pages (64 * nw) ≤ 2 ^ 47)
    (hfree : s.mem.free T (X86Env.pages (64 * nw)) = true)
    (hlen16 : s.regs.length = 16) :
    ∃ (s' : X86Int.St) (m2 : Mem), run stage1Im 20 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x22d ∧ s'.reg 14 = some T ∧
      s'.reg 4 = some (sp - 8) ∧
      s'.mem = m2 ++
        [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] ∧
      (∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x → m2.rdN x k = s.mem.rdN x k) ∧
      rd64 s'.mem (sp - 8) = some (c % W) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → i ≠ 14 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 ∧
      wshape m2 = wshape s.mem := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hsp16mod : (sp + W - 16) % W = sp - 16 := by
    have h : sp + W - 16 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have haddr2 : (sp - 8 + W - 8) % W = sp - 16 := by
    have h : sp - 8 + W - 8 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have h64 : ∀ k : Nat, k ≤ 64 → k * nw < W := by
    intro k hk
    have hW : W = 2 ^ 64 := rfl
    have h1 : k * nw ≤ 64 * nw := Nat.mul_le_mul_right nw hk
    have h2a : nw ≤ 2 ^ 32 - 1 := by omega
    have h2b : 64 * nw ≤ 64 * (2 ^ 32 - 1) :=
      Nat.mul_le_mul_left 64 h2a
    have h2c : 64 * (2 ^ 32 - 1) < 2 ^ 64 := by decide
    omega
  -- stack writes, obtained once
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) c =
      some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) c)
  have hm1 : wr64 s.mem ((sp + W - 8) % W) c = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) c = some m1 := by
    rwa [hsp8mod] at hm1
  have hw1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1raw
  -- 0x1ec: push rsi
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 1 } :=
    step_push (by rw [hrip]; exact hc.p1ec) hr6 hr4 hm1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 1 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x1ed := by
    rw [ht1]; show s.rip + 1 = base + 0x1ed; rw [hrip]
  have ht1r4 : t1.reg 4 = some (sp - 8) := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (hlen 4 (by decide)), hsp8mod]
  have ht1r10 : t1.reg 10 = some nw := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 10 = some nw
    rw [reg_set_other (by decide)]; exact hr10
  have ht1fr : ∀ i, i ≠ 4 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = m1 := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have ht1k : t1.k = s.k := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  -- 0x1ed: mov32 rdx, r10d
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 } :=
    step_mov32 (by rw [ht1rip]; exact hc.p1ed) ht1r10
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x1f0 := by
    rw [ht2]; show t1.rip + 3 = base + 0x1f0; rw [ht1rip]
  have hnwmod : nw % 2 ^ 32 = nw := Nat.mod_eq_of_lt hnw32
  have ht2r2 : t2.reg 2 = some nw := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg 2 =
      some nw
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide), hnwmod]
  have ht2fr : ∀ i, i ≠ 2 → i ≠ 4 → t2.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht1fr i hi4
  have ht2r4 : t2.reg 4 = some (sp - 8) := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht1r4
  have ht2mem : t2.mem = m1 := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with
      rip := t1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  have ht2k : t2.k = s.k := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).k = s.k
    exact ht1k
  -- six doublings of rdx: nw -> 64*nw, collected as one run of 6
  have hdbl : ∀ (u : X86Int.St) (site : Nat) (v : Nat),
      stage1Im site = some (.alu .add 2 2, 3) → u.rip = site →
      u.reg 2 = some v → v + v < W → u.regs.length = 16 →
      ∃ u', run stage1Im 1 (Out.ok u) = Out.ok u' ∧
        u'.rip = site + 3 ∧ u'.reg 2 = some (2 * v) ∧
        (∀ i, i ≠ 2 → u'.reg i = u.reg i) ∧ u'.mem = u.mem ∧
        u'.regs.length = 16 ∧ u'.k = u.k := by
    intro u site v hsite hurip hur2 hvW hulen
    have e : X86Int.step stage1Im u =
        .ok { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } :=
      step_alu_add (by rw [hurip]; exact hsite) hur2 hur2
    obtain ⟨u', hu'⟩ : ∃ t : X86Int.St, t =
        { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } := ⟨_, rfl⟩
    have hrun : run stage1Im 1 (Out.ok u) = Out.ok u' := by
      rw [run_succ_ok stage1Im 0 u, e, ← hu', run_zero]
    refine ⟨u', hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu']; show u.rip + 3 = site + 3; rw [hurip]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).reg 2 = some (2 * v)
      rw [reg_set_self (by rw [hulen]; decide), aluF_add_fst,
        Nat.mod_eq_of_lt hvW, Nat.two_mul]
    · intro i hi
      rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).reg i = u.reg i
      rw [reg_set_other hi]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).mem = u.mem
      rw [mem_set]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).regs.length = 16
      rw [len_set]; exact hulen
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).k = u.k
      rfl
  obtain ⟨t3, hrun3, ht3rip, ht3r2, ht3fr, ht3mem, ht3len, ht3k⟩ :=
    hdbl t2 (base + 0x1f0) nw hc.p1f0 ht2rip ht2r2
      (by have := h64 2 (by decide); omega) ht2len
  obtain ⟨t4, hrun4, ht4rip, ht4r2, ht4fr, ht4mem, ht4len, ht4k⟩ :=
    hdbl t3 (base + 0x1f3) (2 * nw) (by rw [show base + 0x1f0 + 3 = base + 0x1f3 from rfl] at ht3rip; exact hc.p1f3)
      (by rw [ht3rip]) ht3r2
      (by have := h64 4 (by decide); omega) ht3len
  obtain ⟨t5, hrun5, ht5rip, ht5r2, ht5fr, ht5mem, ht5len, ht5k⟩ :=
    hdbl t4 (base + 0x1f6) (2 * (2 * nw)) (by exact hc.p1f6)
      (by rw [ht4rip]) ht4r2
      (by have := h64 8 (by decide); omega) ht4len
  obtain ⟨t6, hrun6, ht6rip, ht6r2, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    hdbl t5 (base + 0x1f9) (2 * (2 * (2 * nw))) (by exact hc.p1f9)
      (by rw [ht5rip]) ht5r2
      (by have := h64 16 (by decide); omega) ht5len
  obtain ⟨t7, hrun7, ht7rip, ht7r2, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    hdbl t6 (base + 0x1fc) (2 * (2 * (2 * (2 * nw)))) (by exact hc.p1fc)
      (by rw [ht6rip]) ht6r2
      (by have := h64 32 (by decide); omega) ht6len
  obtain ⟨t8, hrun8, ht8rip, ht8r2, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    hdbl t7 (base + 0x1ff) (2 * (2 * (2 * (2 * (2 * nw)))))
      (by exact hc.p1ff) (by rw [ht7rip]) ht7r2
      (by have := h64 64 (by decide); omega) ht7len
  have ht8rip' : t8.rip = base + 0x202 := ht8rip
  have ht8r2' : t8.reg 2 = some (64 * nw) := by
    have h : 2 * (2 * (2 * (2 * (2 * (2 * nw))))) = 64 * nw := by omega
    rw [h] at ht8r2; exact ht8r2
  have ht8r4 : t8.reg 4 = some (sp - 8) := by
    rw [ht8fr 4 (by decide), ht7fr 4 (by decide), ht6fr 4 (by decide),
      ht5fr 4 (by decide), ht4fr 4 (by decide), ht3fr 4 (by decide)]
    exact ht2r4
  have ht8frs : ∀ i, i ≠ 2 → i ≠ 4 → t8.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht8fr i hi2, ht7fr i hi2, ht6fr i hi2, ht5fr i hi2, ht4fr i hi2,
      ht3fr i hi2]
    exact ht2fr i hi2 hi4
  have ht8mems : t8.mem = m1 := by
    rw [ht8mem, ht7mem, ht6mem, ht5mem, ht4mem, ht3mem]; exact ht2mem
  have ht8ks : t8.k = s.k := by
    rw [ht8k, ht7k, ht6k, ht5k, ht4k, ht3k]; exact ht2k
  -- 0x202: movI rax, 9
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 0 (some (9 % W)) with rip := t8.rip + 5 } :=
    step_movI (by rw [ht8rip']; exact hc.p202)
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 0 (some (9 % W)) with rip := t8.rip + 5 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x207 := by
    rw [ht9]; show t8.rip + 5 = base + 0x207; rw [ht8rip']
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht9r0 : t9.reg 0 = some 9 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_self (by rw [ht8len]; decide), h9W]
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi4
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht8frs i hi2 hi4
  have ht9r2 : t9.reg 2 = some (64 * nw) := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 2 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht8r2'
  have ht9r4 : t9.reg 4 = some (sp - 8) := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht8r4
  have ht9r7 : ∃ v, t9.reg 7 = some v := by
    obtain ⟨v, hv⟩ := hr7
    exact ⟨v, by rw [ht9fr 7 (by decide) (by decide) (by decide)]; exact hv⟩
  have ht9r9 : ∃ v, t9.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by rw [ht9fr 9 (by decide) (by decide) (by decide)]; exact hv⟩
  have ht9mem : t9.mem = m1 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht8mems
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht8len
  have ht9k : t9.k = s.k := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).k = s.k
    exact ht8ks
  -- 0x207: xor rdi, rdi
  obtain ⟨v7, hv7⟩ := ht9r7
  have hxor : ∀ a : Nat, (aluF .xor a a).1 = 0 := by
    intro a; simp [aluF, Nat.xor_self]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t9.rip + 2 } :=
    step_alu_xor (by rw [ht9rip]; exact hc.p207) hv7 hv7
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t9.rip + 2 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x209 := by
    rw [ht10]; show t9.rip + 2 = base + 0x209; rw [ht9rip]
  have ht10r7 : t10.reg 7 = some 0 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 7 = some 0
    rw [reg_set_self (by rw [ht9len]; decide), hxor]
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 7 → t10.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi7
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg i = s.reg i
    rw [reg_set_other hi7]; exact ht9fr i hi0 hi2 hi4
  have ht10r0 : t10.reg 0 = some 9 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 0 = some 9
    rw [reg_set_other (by decide)]; exact ht9r0
  have ht10r2 : t10.reg 2 = some (64 * nw) := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 2 = some (64 * nw)
    rw [reg_set_other (by decide)]; exact ht9r2
  have ht10r4 : t10.reg 4 = some (sp - 8) := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide)]; exact ht9r4
  have ht10mem : t10.mem = m1 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).mem = m1
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).regs.length = 16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).k = s.k
    exact ht9k
  -- 0x209: mov rsi, rdx
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 } :=
    step_mov (by rw [ht10rip]; exact hc.p209) ht10r2
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x20c := by
    rw [ht11]; show t10.rip + 3 = base + 0x20c; rw [ht10rip]
  have ht11r6 : t11.reg 6 = some (64 * nw) := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 6 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_self (by rw [ht10len]; decide)]
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t11.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht10fr i hi0 hi2 hi4 hi7
  have ht11r0 : t11.reg 0 = some 9 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r0
  have ht11r7 : t11.reg 7 = some 0 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r7
  have ht11r4 : t11.reg 4 = some (sp - 8) := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r4
  have ht11mem : t11.mem = m1 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with
      rip := t10.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht10len
  have ht11k : t11.k = s.k := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).k = s.k
    exact ht10k
  -- 0x20c: movI rdx, 3
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (3 % W)) with rip := t11.rip + 5 } :=
    step_movI (by rw [ht11rip]; exact hc.p20c)
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (3 % W)) with rip := t11.rip + 5 } := ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x211 := by
    rw [ht12]; show t11.rip + 5 = base + 0x211; rw [ht11rip]
  have h3W : 3 % W = 3 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht12r2 : t12.reg 2 = some 3 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 2 =
      some 3
    rw [reg_with_rip, reg_set_self (by rw [ht11len]; decide), h3W]
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t12.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht11fr i hi0 hi2 hi4 hi6 hi7
  have ht12r0 : t12.reg 0 = some 9 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r0
  have ht12r6 : t12.reg 6 = some (64 * nw) := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 6 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r6
  have ht12r7 : t12.reg 7 = some 0 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r7
  have ht12r4 : t12.reg 4 = some (sp - 8) := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r4
  have ht12mem : t12.mem = m1 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with
      rip := t11.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht11len
  have ht12k : t12.k = s.k := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).k = s.k
    exact ht11k
  -- 0x211: movI r10, 34
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12.set 10 (some (34 % W)) with rip := t12.rip + 6 } :=
    step_movI (by rw [ht12rip]; exact hc.p211)
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12.set 10 (some (34 % W)) with rip := t12.rip + 6 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x217 := by
    rw [ht13]; show t12.rip + 6 = base + 0x217; rw [ht12rip]
  have h34W : 34 % W = 34 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht13r10 : t13.reg 10 = some 34 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by rw [ht12len]; decide), h34W]
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t13.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi10]
    exact ht12fr i hi0 hi2 hi4 hi6 hi7
  have ht13r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t13.reg j = t12.reg j := by
    intro j hj
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg j =
      t12.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht13mem : t13.mem = m1 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with
      rip := t12.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht12len
  have ht13k : t13.k = s.k := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).k = s.k
    exact ht12k
  -- 0x217: pushI -1
  have ht13r4 : t13.reg 4 = some (sp - 8) := by
    rw [ht13r 4 (Or.inr (Or.inr (Or.inl rfl)))]; exact ht12r4
  obtain ⟨m2, hm2⟩ : ∃ m, wr64 m1 ((sp - 8 + W - 8) % W)
      (18446744073709551615 % W) = some m := by
    have hh : (wr64 s.mem (sp - 16) (18446744073709551615 % W)).isSome := by
      have h2 : (wr64 s.mem ((sp + W - 16) % W)
          (18446744073709551615 % W)).isSome :=
        hroom sp hr4 1 (by decide) _
      rwa [hsp16mod] at h2
    have h3 := wr64_some_of_wr64 hm1wr hh
    rw [haddr2]
    exact Option.isSome_iff_exists.mp h3
  have hm2wr : wr64 m1 (sp - 16) (18446744073709551615 % W) = some m2 := by
    rwa [haddr2] at hm2
  have hw2 : wshape m2 = wshape s.mem :=
    (wrN_wshape _ _ _ hm2wr).trans hw1
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13.set 4 (some ((sp - 8 + W - 8) % W)) with mem := m2, rip := t13.rip + 2 } := by
    have hm2' : wr64 t13.mem ((sp - 8 + W - 8) % W)
        (18446744073709551615 % W) = some m2 := by
      rw [ht13mem]; exact hm2
    exact step_pushI (by rw [ht13rip]; exact hc.p217) ht13r4 hm2'
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13.set 4 (some ((sp - 8 + W - 8) % W)) with mem := m2, rip := t13.rip + 2 } :=
    ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x219 := by
    rw [ht14]; show t13.rip + 2 = base + 0x219; rw [ht13rip]
  have ht14r4 : t14.reg 4 = some (sp - 16) := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg 4 = some (sp - 16)
    rw [reg_set_self (by rw [ht13len]; decide), haddr2]
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t14.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht13fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht14r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t14.reg j = t13.reg j := by
    intro j hj
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg j = t13.reg j
    rw [reg_set_other (by omega)]
  have ht14mem : t14.mem = m2 := by rw [ht14]
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht13len
  have ht14k : t14.k = s.k := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).k = s.k
    exact ht13k
  -- 0x219: pop r8
  have hImm : 18446744073709551615 % W = W - 1 := by
    have h : (18446744073709551615 : Nat) = W - 1 := by
      unfold W; decide
    rw [h]
    exact Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)
  have hrd64m2 : rd64 m2 (sp - 16) = some (W - 1) := by
    have h := rd64_wr64 hm2wr
    rwa [hImm, Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)] at h
  have e15 : X86Int.step stage1Im t14 =
      .ok { (t14.set 4 (some ((sp - 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := t14.rip + 2 } := by
    have hv : rd64 t14.mem (sp - 16) = some (W - 1) := by
      rw [ht14mem]; exact hrd64m2
    exact step_pop (by rw [ht14rip]; exact hc.p219) ht14r4 hv
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { (t14.set 4 (some ((sp - 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := t14.rip + 2 } := ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x21b := by
    rw [ht15]; show t14.rip + 2 = base + 0x21b; rw [ht14rip]
  have hspret : (sp - 16 + 8) % W = sp - 8 := by
    rw [show sp - 16 + 8 = sp - 8 by omega,
      Nat.mod_eq_of_lt (by omega)]
  have ht15r8 : t15.reg 8 = some (W - 1) := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg 8 = some (W - 1)
    rw [reg_set_self (by
      show 8 < (t14.set 4 (some ((sp - 16 + 8) % W))).regs.length
      rw [len_set, ht14len]; decide)]
  have ht15r4 : t15.reg 4 = some (sp - 8) := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide),
      reg_set_self (by rw [ht14len]; decide), hspret]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 10 → t15.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi10
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg i = s.reg i
    rw [reg_set_other hi8, reg_set_other hi4]
    exact ht14fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht15r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t15.reg j = t14.reg j := by
    intro j hj
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg j = t14.reg j
    rw [reg_set_other (by omega), reg_set_other (by omega)]
  have ht15mem : t15.mem = m2 := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).mem = m2
    rw [mem_set, mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).regs.length = 16
    rw [len_set, len_set]; exact ht14len
  have ht15k : t15.k = s.k := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).k = s.k
    exact ht14k
  -- 0x21b: xor r9, r9
  have ht15r9v : ∃ v, t15.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by
      rw [ht15fr 9 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)]
      exact hv⟩
  obtain ⟨v9, hv9⟩ := ht15r9v
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t15.rip + 3 } :=
    step_alu_xor (by rw [ht15rip]; exact hc.p21b) hv9 hv9
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t15.rip + 3 } := ⟨_, rfl⟩
  have ht16rip : t16.rip = base + 0x21e := by
    rw [ht16]; show t15.rip + 3 = base + 0x21e; rw [ht15rip]
  have ht16r9 : t16.reg 9 = some 0 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg 9 = some 0
    rw [reg_set_self (by rw [ht15len]; decide), hxor]
  have ht16fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 9 → i ≠ 10 → t16.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg i = s.reg i
    rw [reg_set_other hi9]
    exact ht15fr i hi0 hi2 hi4 hi6 hi7 hi8 hi10
  have ht16r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 ∨
      j = 8 ∨ j = 10 → t16.reg j = t15.reg j := by
    intro j hj
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg j = t15.reg j
    rw [reg_set_other (by omega)]
  have ht16mem : t16.mem = m2 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).mem = m2
    rw [mem_set]; exact ht15mem
  have ht16len : t16.regs.length = 16 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).regs.length = 16
    rw [len_set]; exact ht15len
  have ht16k : t16.k = s.k := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).k = s.k
    exact ht15k
  have ht16r0 : t16.reg 0 = some 9 := by
    rw [ht16r 0 (Or.inl rfl), ht15r 0 (Or.inl rfl),
      ht14r 0 (Or.inl rfl), ht13r 0 (Or.inl rfl)]
    exact ht12r0
  have ht16r7 : t16.reg 7 = some 0 := by
    rw [ht16r 7 (by decide), ht15r 7 (by decide), ht14r 7 (by decide),
      ht13r 7 (by decide)]
    exact ht12r7
  have ht16r6 : t16.reg 6 = some (64 * nw) := by
    rw [ht16r 6 (by decide), ht15r 6 (by decide), ht14r 6 (by decide),
      ht13r 6 (by decide)]
    exact ht12r6
  have ht16r2 : t16.reg 2 = some 3 := by
    rw [ht16r 2 (by decide), ht15r 2 (by decide), ht14r 2 (by decide),
      ht13r 2 (by decide)]
    exact ht12r2
  have ht16r10 : t16.reg 10 = some 34 := by
    rw [ht16r 10 (by decide), ht15r 10 (by decide), ht14r 10 (by decide)]
    exact ht13r10
  have ht16r8 : t16.reg 8 = some (W - 1) := by
    rw [ht16r 8 (by decide)]
    exact ht15r8
  -- 0x21e: syscall (mmap)
  have hm2free : m2.free T (X86Env.pages (64 * nw)) = true := by
    rw [free_of_wshape hw2]; exact hfree
  have hplace : X86Env.place m2 (some (.at T)) (X86Env.pages (64 * nw)) =
      some (some T) :=
    place_at hTpage hTpos hTbound hm2free
  have hsc : X86Env.syscall t16.k t16.mem 9 0 (64 * nw) PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret T (Kern.next s.k).2
        (m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩])) := by
    rw [ht16k, ht16mem]
    exact syscall_mmap_exact (by
      have : 0 < nw := Nat.pos_of_ne_zero hnw0; omega) hhead hplace
  have e17 : X86Int.step stage1Im t16 =
      .ok { ((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩], rip := t16.rip + 2 } :=
    step_syscall_ret (by rw [ht16rip]; exact hc.p21e) ht16r0 ht16r7
      ht16r6 ht16r2 ht16r10 ht16r8 ht16r9 hsc
  obtain ⟨t17, ht17⟩ : ∃ t : X86Int.St, t =
      { ((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩], rip := t16.rip + 2 } :=
    ⟨_, rfl⟩
  have ht17rip : t17.rip = base + 0x220 := by
    rw [ht17]; show t16.rip + 2 = base + 0x220; rw [ht16rip]
  have ht17r0 : t17.reg 0 = some T := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg 0 = some T
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht16len]; decide)]
  have ht17fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t17.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht16fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht17r4 : t17.reg 4 = some (sp - 8) := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide),
      ht16r 4 (Or.inr (Or.inr (Or.inl rfl)))]
    exact ht15r4
  have ht17mem : t17.mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
    rw [ht17]
  have ht17len : t17.regs.length = 16 := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht16len
  have ht17k : t17.k = (Kern.next s.k).2 := by
    rw [ht17]
  -- 0x220: test rax, rax
  have e18 : X86Int.step stage1Im t17 =
      .ok { t17 with fl := (aluF .and T T).2, rip := t17.rip + 3 } :=
    step_test (by rw [ht17rip]; exact hc.p220) ht17r0 ht17r0
  obtain ⟨t18, ht18⟩ : ∃ t : X86Int.St, t =
      { t17 with fl := (aluF .and T T).2, rip := t17.rip + 3 } := ⟨_, rfl⟩
  have ht18rip : t18.rip = base + 0x223 := by
    rw [ht18]; show t17.rip + 3 = base + 0x223; rw [ht17rip]
  have hmsb : msb T = false := by
    have hT63 : T < 2 ^ 63 := by
      have hpg : 0 ≤ X86Env.pages (64 * nw) := Nat.zero_le _
      have h47 : (2 : Nat) ^ 47 < 2 ^ 63 := by decide
      omega
    unfold msb
    rw [Nat.div_eq_of_lt hT63]
    rfl
  have ht18fl : X86Int.cond t18.fl 9 = some true := by
    have hsf : t18.fl.sf = some false := by
      rw [ht18]
      show (aluF .and T T).2.sf = some false
      have h1 : (aluF .and T T).2.sf = some (msb (aluF .and T T).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  -- 0x223: jns 0x22a
  have e19 : X86Int.step stage1Im t18 =
      .ok { t18 with rip := base + 0x22a } := by
    have h := step_jcc (im := stage1Im) (s := t18)
      (by rw [ht18rip]; exact hc.p223) ht18fl
    simpa using h
  obtain ⟨t19, ht19⟩ : ∃ t : X86Int.St, t =
      { t18 with rip := base + 0x22a } := ⟨_, rfl⟩
  have ht19rip : t19.rip = base + 0x22a := by rw [ht19]
  have ht19r0 : t19.reg 0 = some T := by
    have h18r0 : t18.reg 0 = some T := by
      rw [ht18]; show t17.reg 0 = some T; exact ht17r0
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg 0 = some T
    rw [reg_with_rip]; exact h18r0
  have ht19fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t19.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h18i : t18.reg i = s.reg i := by
      rw [ht18]; show t17.reg i = s.reg i
      exact ht17fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg i = s.reg i
    rw [reg_with_rip]; exact h18i
  have ht19r4 : t19.reg 4 = some (sp - 8) := by
    have h18r4 : t18.reg 4 = some (sp - 8) := by
      rw [ht18]; show t17.reg 4 = some (sp - 8); exact ht17r4
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg 4 = some (sp - 8)
    rw [reg_with_rip]; exact h18r4
  have ht19mem : t19.mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
    have h18m : t18.mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
      rw [ht18]; show t17.mem = _; exact ht17mem
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
    rw [mem_with_rip]; exact h18m
  have ht19len : t19.regs.length = 16 := by
    have h18l : t18.regs.length = 16 := by
      rw [ht18]; show t17.regs.length = 16; exact ht17len
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).regs.length = 16
    rw [len_with_rip]; exact h18l
  have ht19k : t19.k = (Kern.next s.k).2 := by
    have h18k : t18.k = (Kern.next s.k).2 := by
      rw [ht18]; show t17.k = (Kern.next s.k).2; exact ht17k
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).k = (Kern.next s.k).2
    exact h18k
  -- 0x22a: mov r14, rax
  have e20 : X86Int.step stage1Im t19 =
      .ok { t19.set 14 (some T) with rip := t19.rip + 3 } :=
    step_mov (by rw [ht19rip]; exact hc.p22a) ht19r0
  obtain ⟨t20, ht20⟩ : ∃ t : X86Int.St, t =
      { t19.set 14 (some T) with rip := t19.rip + 3 } := ⟨_, rfl⟩
  -- compose the 20 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  have r17 : run stage1Im 1 (Out.ok t16) = Out.ok t17 := by
    rw [run_succ_ok stage1Im 0 t16, e17, ← ht17, run_zero]
  have r18 : run stage1Im 1 (Out.ok t17) = Out.ok t18 := by
    rw [run_succ_ok stage1Im 0 t17, e18, ← ht18, run_zero]
  have r19 : run stage1Im 1 (Out.ok t18) = Out.ok t19 := by
    rw [run_succ_ok stage1Im 0 t18, e19, ← ht19, run_zero]
  have r20 : run stage1Im 1 (Out.ok t19) = Out.ok t20 := by
    rw [run_succ_ok stage1Im 0 t19, e20, ← ht20, run_zero]
  have hrun : run stage1Im 20 (Out.ok s) = Out.ok t20 := by
    rw [show (20 : Nat) = 1 + 19 by omega, run_add stage1Im 1 19, r1,
      show (19 : Nat) = 1 + 18 by omega, run_add stage1Im 1 18, r2,
      show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, hrun3,
      show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, hrun4,
      show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, hrun5,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, hrun6,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, hrun7,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, hrun8,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r9,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r10,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r11,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r12,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r13,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r14,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r15,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r16,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r17,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r18,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r19, r20]
  have hm2fr : ∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x →
      m2.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    have h1 : m2.rdN x k = m1.rdN x k := by
      apply rdN_wr64_other hm2wr
      rcases hdis with h | h
      · exact Or.inr h
      · exact Or.inl (by omega)
    rw [h1]
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  have hrd64cur : rd64 t20.mem (sp - 8) = some (c % W) := by
    have h20mem : t20.mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
      rw [ht20]
      show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
      rw [mem_with_rip, mem_set]; exact ht19mem
    have hm2cur : rd64 m2 (sp - 8) = some (c % W) := by
      have h1 : rd64 m2 (sp - 8) = rd64 m1 (sp - 8) :=
        rd64_wr64_other hm2wr (Or.inl (by omega))
      rw [h1]; exact rd64_wr64 hm1wr
    obtain ⟨bs8, hbs8, hle⟩ : ∃ bs, m2.rdN (sp - 8) 8 = some bs ∧
        leVal bs = c % W := by
      have h := hm2cur
      unfold rd64 at h
      have hs : (m2.rdN (sp - 8) 8).isSome := by
        cases h8 : m2.rdN (sp - 8) 8 with
        | none => rw [h8] at h; simp at h
        | some bs => simp
      obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hs
      exact ⟨bs, hbs, by rw [hbs] at h; exact Option.some.inj h⟩
    rw [h20mem]
    show ((m2 ++ [(⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
      true⟩ : Region)]).rdN (sp - 8) 8).map leVal = some (c % W)
    rw [rdN_append_left hbs8]
    simp [hle]
  refine ⟨t20, m2, hrun, ?_, ?_, ?_, ?_, hm2fr, hrd64cur, ?_, ?_, ?_,
    hw2⟩
  · rw [ht20]; show t19.rip + 3 = base + 0x22d; rw [ht19rip]
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg 14 =
      some T
    rw [reg_with_rip, reg_set_self (by rw [ht19len]; decide)]
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht19r4
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
    rw [mem_with_rip, mem_set]; exact ht19mem
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11 hi14
    rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi14]
    exact ht19fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht19len
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).k =
      (Kern.next s.k).2
    exact ht19k

/-- The scratch `mmap`, success case: from `base + 0x22d`
(`r13` still the file length, `rsp` decremented by the saved
cursor) to `base + 0x258`, asking the oracle for `L` bytes at
`S` and leaving the scratch base in `r11`. -/
theorem parser_mmap2_ok {L sp1 S : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hhead : s.k.oracle.head? = some (.at S))
    (hSpage : S % page = 0) (hSpos : 0 < S)
    (hSbound : S + X86Env.pages L ≤ 2 ^ 47)
    (hfree : s.mem.free S (X86Env.pages L) = true)
    (hlen16 : s.regs.length = 16) :
    ∃ (s' : X86Int.St) (m1 : Mem),
      run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x258 ∧ s'.reg 11 = some S ∧
      s'.reg 4 = some sp1 ∧
      s'.mem = m1 ++
        [⟨S, List.replicate (X86Env.pages L) 0, true⟩] ∧
      (∀ x k, x + k ≤ sp1 - 8 ∨ sp1 ≤ x →
        m1.rdN x k = s.mem.rdN x k) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 ∧
      wshape m1 = wshape s.mem := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp1 + W - 8) % W = sp1 - 8 := by
    have h : sp1 + W - 8 = (sp1 - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hxor : ∀ a : Nat, (aluF .xor a a).1 = 0 := by
    intro a; simp [aluF, Nat.xor_self]
  -- 0x22d: movI rax, 9
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (9 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.p22d)
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (9 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x232 := by
    rw [ht1]; show s.rip + 5 = base + 0x232; rw [hrip]
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht1r0 : t1.reg 0 = some 9 := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_self (hlen 0 (by decide)), h9W]
  have ht1fr : ∀ i, i ≠ 0 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1r4 : t1.reg 4 = some sp1 := by
    rw [ht1fr 4 (by decide)]; exact hr4
  have ht1r13 : t1.reg 13 = some L := by
    rw [ht1fr 13 (by decide)]; exact hr13
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).k = s.k
    rfl
  -- 0x232: xor rdi, rdi
  obtain ⟨v7, hv7⟩ : ∃ v, t1.reg 7 = some v := by
    obtain ⟨v, hv⟩ := hr7
    exact ⟨v, by rw [ht1fr 7 (by decide)]; exact hv⟩
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t1.rip + 2 } :=
    step_alu_xor (by rw [ht1rip]; exact hc.p232) hv7 hv7
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x234 := by
    rw [ht2]; show t1.rip + 2 = base + 0x234; rw [ht1rip]
  have ht2r7 : t2.reg 7 = some 0 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 7 = some 0
    rw [reg_set_self (by rw [ht1len]; decide), hxor]
  have ht2fr : ∀ i, i ≠ 0 → i ≠ 7 → t2.reg i = s.reg i := by
    intro i hi0 hi7
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg i = s.reg i
    rw [reg_set_other hi7]; exact ht1fr i hi0
  have ht2r0 : t2.reg 0 = some 9 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 0 = some 9
    rw [reg_set_other (by decide)]; exact ht1r0
  have ht2r13 : t2.reg 13 = some L := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 13 = some L
    rw [reg_set_other (by decide)]; exact ht1r13
  have ht2r4 : t2.reg 4 = some sp1 := by
    rw [ht2fr 4 (by decide) (by decide)]; exact hr4
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).k = s.k
    exact ht1k
  -- 0x234: mov rsi, r13
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 6 (some L) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p234) ht2r13
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 6 (some L) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x237 := by
    rw [ht3]; show t2.rip + 3 = base + 0x237; rw [ht2rip]
  have ht3r6 : t3.reg 6 = some L := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 6 = some L
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 7 → t3.reg i = s.reg i := by
    intro i hi0 hi6 hi7
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht2fr i hi0 hi7
  have ht3r0 : t3.reg 0 = some 9 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r0
  have ht3r7 : t3.reg 7 = some 0 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r7
  have ht3r4 : t3.reg 4 = some sp1 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 4 = some sp1
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r4
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  have ht3k : t3.k = s.k := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).k = s.k
    exact ht2k
  -- 0x237: movI rdx, 3
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (3 % W)) with rip := t3.rip + 5 } :=
    step_movI (by rw [ht3rip]; exact hc.p237)
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (3 % W)) with rip := t3.rip + 5 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x23c := by
    rw [ht4]; show t3.rip + 5 = base + 0x23c; rw [ht3rip]
  have h3W : 3 % W = 3 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4r2 : t4.reg 2 = some 3 := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg 2 =
      some 3
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide), h3W]
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 → t4.reg i = s.reg i := by
    intro i hi0 hi2 hi6 hi7
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht3fr i hi0 hi6 hi7
  have ht4r : ∀ j : Nat, j = 0 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t4.reg j = t3.reg j := by
    intro j hj
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg j =
      t3.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).k = s.k
    exact ht3k
  -- 0x23c: movI r10, 34
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 10 (some (34 % W)) with rip := t4.rip + 6 } :=
    step_movI (by rw [ht4rip]; exact hc.p23c)
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 10 (some (34 % W)) with rip := t4.rip + 6 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x242 := by
    rw [ht5]; show t4.rip + 6 = base + 0x242; rw [ht4rip]
  have h34W : 34 % W = 34 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht5r10 : t5.reg 10 = some 34 := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide), h34W]
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t5.reg i = s.reg i := by
    intro i hi0 hi2 hi6 hi7 hi10
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi10]; exact ht4fr i hi0 hi2 hi6 hi7
  have ht5r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t5.reg j = t4.reg j := by
    intro j hj
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg j =
      t4.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht5r4 : t5.reg 4 = some sp1 := by
    rw [ht5r 4 (by decide), ht4r 4 (by decide)]; exact ht3r4
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with
      rip := t4.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).k = s.k
    exact ht4k
  -- 0x242: pushI -1
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp1 + W - 8 * (0 + 1)) % W)
      (18446744073709551615 % W) = some m :=
    Option.isSome_iff_exists.mp (hroom sp1 hr4 0 (by decide) _)
  have hm1 : wr64 s.mem ((sp1 + W - 8) % W) (18446744073709551615 % W) =
      some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp1 - 8) (18446744073709551615 % W) =
      some m1 := by
    rwa [hsp8mod] at hm1
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 4 (some ((sp1 + W - 8) % W)) with mem := m1, rip := t5.rip + 2 } := by
    have hm1' : wr64 t5.mem ((sp1 + W - 8) % W)
        (18446744073709551615 % W) = some m1 := by
      rw [ht5mem]; exact hm1
    exact step_pushI (by rw [ht5rip]; exact hc.p242) ht5r4 hm1'
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 4 (some ((sp1 + W - 8) % W)) with mem := m1, rip := t5.rip + 2 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x244 := by
    rw [ht6]; show t5.rip + 2 = base + 0x244; rw [ht5rip]
  have ht6r4 : t6.reg 4 = some (sp1 - 8) := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg 4 = some (sp1 - 8)
    rw [reg_set_self (by rw [ht5len]; decide), hsp8mod]
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t6.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht5fr i hi0 hi2 hi6 hi7 hi10
  have ht6r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t6.reg j = t5.reg j := by
    intro j hj
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg j = t5.reg j
    rw [reg_set_other (by omega)]
  have ht6mem : t6.mem = m1 := by rw [ht6]
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).k = s.k
    exact ht5k
  -- 0x244: pop r8
  have hImm : 18446744073709551615 % W = W - 1 := by
    have h : (18446744073709551615 : Nat) = W - 1 := by
      unfold W; decide
    rw [h]
    exact Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)
  have hrd64m1 : rd64 m1 (sp1 - 8) = some (W - 1) := by
    have h := rd64_wr64 hm1wr
    rwa [hImm, Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)] at h
  have e7 : X86Int.step stage1Im t6 =
      .ok { (t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8 (some (W - 1)) with
        rip := t6.rip + 2 } := by
    have hv : rd64 t6.mem (sp1 - 8) = some (W - 1) := by
      rw [ht6mem]; exact hrd64m1
    exact step_pop (by rw [ht6rip]; exact hc.p244) ht6r4 hv
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { (t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8 (some (W - 1)) with
        rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x246 := by
    rw [ht7]; show t6.rip + 2 = base + 0x246; rw [ht6rip]
  have hspret : (sp1 - 8 + 8) % W = sp1 := by
    rw [show sp1 - 8 + 8 = sp1 by omega, Nat.mod_eq_of_lt hspW]
  have ht7r8 : t7.reg 8 = some (W - 1) := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg 8 = some (W - 1)
    rw [reg_set_self (by
      show 8 < (t6.set 4 (some ((sp1 - 8 + 8) % W))).regs.length
      rw [len_set, ht6len]; decide)]
  have ht7r4 : t7.reg 4 = some sp1 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg 4 = some sp1
    rw [reg_set_other (by decide),
      reg_set_self (by rw [ht6len]; decide), hspret]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 10 → t7.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi10
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg i = s.reg i
    rw [reg_set_other hi8, reg_set_other hi4]
    exact ht6fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht7r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t7.reg j = t6.reg j := by
    intro j hj
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg j = t6.reg j
    rw [reg_set_other (by omega), reg_set_other (by omega)]
  have ht7mem : t7.mem = m1 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).mem = m1
    rw [mem_set, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).regs.length = 16
    rw [len_set, len_set]; exact ht6len
  have ht7k : t7.k = s.k := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).k = s.k
    exact ht6k
  -- 0x246: xor r9, r9
  obtain ⟨v9, hv9⟩ : ∃ v, t7.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by
      rw [ht7fr 9 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)]
      exact hv⟩
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t7.rip + 3 } :=
    step_alu_xor (by rw [ht7rip]; exact hc.p246) hv9 hv9
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x249 := by
    rw [ht8]; show t7.rip + 3 = base + 0x249; rw [ht7rip]
  have ht8r9 : t8.reg 9 = some 0 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg 9 = some 0
    rw [reg_set_self (by rw [ht7len]; decide), hxor]
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 9 → i ≠ 10 → t8.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg i = s.reg i
    rw [reg_set_other hi9]
    exact ht7fr i hi0 hi2 hi4 hi6 hi7 hi8 hi10
  have ht8r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 ∨
      j = 8 ∨ j = 10 → t8.reg j = t7.reg j := by
    intro j hj
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg j = t7.reg j
    rw [reg_set_other (by omega)]
  have ht8mem : t8.mem = m1 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).mem = m1
    rw [mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).regs.length = 16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).k = s.k
    exact ht7k
  have ht8r0 : t8.reg 0 = some 9 := by
    rw [ht8r 0 (Or.inl rfl), ht7r 0 (Or.inl rfl), ht6r 0 (Or.inl rfl),
      ht5r 0 (Or.inl rfl), ht4r 0 (Or.inl rfl)]
    exact ht3r0
  have ht8r7 : t8.reg 7 = some 0 := by
    rw [ht8r 7 (by decide), ht7r 7 (by decide), ht6r 7 (by decide),
      ht5r 7 (by decide), ht4r 7 (by decide)]
    exact ht3r7
  have ht8r6 : t8.reg 6 = some L := by
    rw [ht8r 6 (by decide), ht7r 6 (by decide), ht6r 6 (by decide),
      ht5r 6 (by decide), ht4r 6 (by decide)]
    exact ht3r6
  have ht8r2 : t8.reg 2 = some 3 := by
    rw [ht8r 2 (by decide), ht7r 2 (by decide), ht6r 2 (by decide),
      ht5r 2 (by decide)]
    exact ht4r2
  have ht8r10 : t8.reg 10 = some 34 := by
    rw [ht8r 10 (by decide), ht7r 10 (by decide), ht6r 10 (by decide)]
    exact ht5r10
  have ht8r8 : t8.reg 8 = some (W - 1) := by
    rw [ht8r 8 (by decide)]
    exact ht7r8
  have ht8r4 : t8.reg 4 = some sp1 := by
    rw [ht8r 4 (by decide)]
    exact ht7r4
  -- 0x249: syscall (mmap)
  have hw1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1raw
  have hm1free : m1.free S (X86Env.pages L) = true := by
    rw [free_of_wshape hw1]; exact hfree
  have hplace : X86Env.place m1 (some (.at S)) (X86Env.pages L) =
      some (some S) :=
    place_at hSpage hSpos hSbound hm1free
  have hsc : X86Env.syscall t8.k t8.mem 9 0 L PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret S (Kern.next s.k).2
        (m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩])) := by
    rw [ht8k, ht8mem]
    exact syscall_mmap_exact hL hhead hplace
  have e9 : X86Int.step stage1Im t8 =
      .ok { ((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩], rip := t8.rip + 2 } :=
    step_syscall_ret (by rw [ht8rip]; exact hc.p249) ht8r0 ht8r7
      ht8r6 ht8r2 ht8r10 ht8r8 ht8r9 hsc
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { ((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩], rip := t8.rip + 2 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x24b := by
    rw [ht9]; show t8.rip + 2 = base + 0x24b; rw [ht8rip]
  have ht9r0 : t9.reg 0 = some S := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg 0 = some S
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht8len]; decide)]
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t9.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht8fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht9r4 : t9.reg 4 = some sp1 := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg 4 = some sp1
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide)]
    exact ht8r4
  have ht9mem : t9.mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
    rw [ht9]
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht8len
  have ht9k : t9.k = (Kern.next s.k).2 := by
    rw [ht9]
  -- 0x24b: test rax, rax
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with fl := (aluF .and S S).2, rip := t9.rip + 3 } :=
    step_test (by rw [ht9rip]; exact hc.p24b) ht9r0 ht9r0
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with fl := (aluF .and S S).2, rip := t9.rip + 3 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x24e := by
    rw [ht10]; show t9.rip + 3 = base + 0x24e; rw [ht9rip]
  have hmsb : msb S = false := by
    have hS63 : S < 2 ^ 63 := by
      have hpg : 0 ≤ X86Env.pages L := Nat.zero_le _
      have h47 : (2 : Nat) ^ 47 < 2 ^ 63 := by decide
      omega
    unfold msb
    rw [Nat.div_eq_of_lt hS63]
    rfl
  have ht10fl : X86Int.cond t10.fl 9 = some true := by
    have hsf : t10.fl.sf = some false := by
      rw [ht10]
      show (aluF .and S S).2.sf = some false
      have h1 : (aluF .and S S).2.sf = some (msb (aluF .and S S).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  -- 0x24e: jns 0x255
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x255 } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p24e) ht10fl
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x255 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x255 := by rw [ht11]
  have ht11r0 : t11.reg 0 = some S := by
    have h10r0 : t10.reg 0 = some S := by
      rw [ht10]; show t9.reg 0 = some S; exact ht9r0
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg 0 = some S
    rw [reg_with_rip]; exact h10r0
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t11.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i
      exact ht9fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg i = s.reg i
    rw [reg_with_rip]; exact h10i
  have ht11r4 : t11.reg 4 = some sp1 := by
    have h10r4 : t10.reg 4 = some sp1 := by
      rw [ht10]; show t9.reg 4 = some sp1; exact ht9r4
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg 4 = some sp1
    rw [reg_with_rip]; exact h10r4
  have ht11mem : t11.mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
    have h10m : t10.mem =
        m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
      rw [ht10]; show t9.mem = _; exact ht9mem
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩]
    rw [mem_with_rip]; exact h10m
  have ht11len : t11.regs.length = 16 := by
    have h10l : t10.regs.length = 16 := by
      rw [ht10]; show t9.regs.length = 16; exact ht9len
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).regs.length = 16
    rw [len_with_rip]; exact h10l
  have ht11k : t11.k = (Kern.next s.k).2 := by
    have h10k : t10.k = (Kern.next s.k).2 := by
      rw [ht10]; show t9.k = (Kern.next s.k).2; exact ht9k
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).k = (Kern.next s.k).2
    exact h10k
  -- 0x255: mov r11, rax
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 11 (some S) with rip := t11.rip + 3 } :=
    step_mov (by rw [ht11rip]; exact hc.p255) ht11r0
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 11 (some S) with rip := t11.rip + 3 } := ⟨_, rfl⟩
  -- compose the 12 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have hrun : run stage1Im 12 (Out.ok s) = Out.ok t12 := by
    rw [show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r1,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r2,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r3,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r4,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r5,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r6,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r7,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r8,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r9,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
  have hm1fr : ∀ x k, x + k ≤ sp1 - 8 ∨ sp1 ≤ x →
      m1.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  refine ⟨t12, m1, hrun, ?_, ?_, ?_, ?_, hm1fr, ?_, ?_, ?_, hw1⟩
  · rw [ht12]; show t11.rip + 3 = base + 0x258; rw [ht11rip]
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg 11 =
      some S
    rw [reg_with_rip, reg_set_self (by rw [ht11len]; decide)]
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg 4 =
      some sp1
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r4
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩]
    rw [mem_with_rip, mem_set]; exact ht11mem
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi11]
    exact ht11fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht11len
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).k =
      (Kern.next s.k).2
    exact ht11k

/-- One `pop rax` step with full frame. -/
theorem pop0_step {t : X86Int.St} {spx w : Nat}
    (hi : stage1Im t.rip = some (.pop 0, 1))
    (hr4 : t.reg 4 = some spx) (hspW : spx + 8 < W)
    (hv : rd64 t.mem spx = some w) (hlen : t.regs.length = 16) :
    ∃ t' : X86Int.St,
      run stage1Im 1 (Out.ok t) = Out.ok t' ∧
      t'.rip = t.rip + 1 ∧ t'.reg 0 = some w ∧
      t'.reg 4 = some (spx + 8) ∧
      (∀ i, i ≠ 0 → i ≠ 4 → t'.reg i = t.reg i) ∧
      t'.mem = t.mem ∧ t'.regs.length = 16 ∧ t'.k = t.k := by
  have e1 : X86Int.step stage1Im t =
      .ok { (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
        rip := t.rip + 1 } :=
    step_pop hi hr4 hv
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
        rip := t.rip + 1 } := ⟨_, rfl⟩
  have hrun : run stage1Im 1 (Out.ok t) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 t, e1, ← ht1, run_zero]
  refine ⟨t1, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg 0 = some w
    rw [reg_with_rip, reg_set_self (by
      show 0 < (t.set 4 (some ((spx + 8) % W))).regs.length
      rw [len_set]; omega)]
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg 4 = some (spx + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0),
      reg_set_self (by omega), Nat.mod_eq_of_lt hspW]
  · intro i hi0 hi4
    rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg i = t.reg i
    rw [reg_with_rip, reg_set_other hi0, reg_set_other hi4]
  · rw [ht1]; rfl
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen
  · rw [ht1]; rfl

/-- Parser setup after the two mmaps: restore the cursor from the
saved slot, point `rdi` at the words table, reload `nwords` into
`r13` (masked to 32 bits), pop six initial-frame words to make
stack room for the checker's calls, and jump to the word loop. -/
theorem parser_setup_ok {sp1 c T B v w1 w2 w3 w4 w5 w6 : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x258) (hr4 : s.reg 4 = some sp1)
    (hr14 : s.reg 14 = some T) (hr3 : s.reg 3 = some B)
    (hspW : sp1 + 56 < W) (hBW : B + 8 < W)
    (hcur : rd64 s.mem sp1 = some c)
    (hnw : rd64 s.mem (B + 8) = some v)
    (hp1 : rd64 s.mem (sp1 + 8) = some w1)
    (hp2 : rd64 s.mem (sp1 + 16) = some w2)
    (hp3 : rd64 s.mem (sp1 + 24) = some w3)
    (hp4 : rd64 s.mem (sp1 + 32) = some w4)
    (hp5 : rd64 s.mem (sp1 + 40) = some w5)
    (hp6 : rd64 s.mem (sp1 + 48) = some w6)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2c9 ∧ s'.reg 6 = some c ∧
      s'.reg 7 = some T ∧ s'.reg 13 = some (v % 2 ^ 32) ∧
      s'.reg 4 = some (sp1 + 56) ∧ s'.reg 0 = some w6 ∧
      s'.mem = s.mem ∧
      (∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x258: pop rsi (restore cursor)
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.p258) hr4 hcur
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x259 := by
    rw [ht1]; show s.rip + 1 = base + 0x259; rw [hrip]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg 6 = some c
    rw [reg_with_rip, reg_set_self (by
      show 6 < (s.set 4 (some ((sp1 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ht1r4 : t1.reg 4 = some (sp1 + 8) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg 4 = some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (hlen 4 (by decide)),
      Nat.mod_eq_of_lt (by omega)]
  have ht1fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t1.reg i = s.reg i := by
    intro i _ hi4 hi6 _ _
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6, reg_set_other hi4]
  have ht1r14 : t1.reg 14 = some T := by
    rw [ht1fr 14 (by decide) (by decide) (by decide) (by decide)
      (by decide)]; exact hr14
  have ht1mem : t1.mem = s.mem := by rw [ht1]; rfl
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen16
  -- 0x259: mov rdi, r14
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 7 (some T) with rip := t1.rip + 3 } :=
    step_mov (by rw [ht1rip]; exact hc.p259) ht1r14
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 7 (some T) with rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x25c := by
    rw [ht2]; show t1.rip + 3 = base + 0x25c; rw [ht1rip]
  have ht2r7 : t2.reg 7 = some T := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 7 = some T
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide)]
  have ht2r3 : t2.reg 3 = some B := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 3 = some B
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 7),
      ht1fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide)]; exact hr3
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    exact ht1r6
  have ht2r4 : t2.reg 4 = some (sp1 + 8) := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ht1r4
  have ht2fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t2.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi7]
    exact ht1fr i hi0 hi4 hi6 hi7 hi13
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x25c: ld r13, [rbx+8] (nwords field, 8-byte read inside header)
  have hld : rd64 t2.mem (ea B 8) = some v := by
    unfold ea
    rw [Nat.mod_eq_of_lt hBW, ht2mem]; exact hnw
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 13 (some v) with rip := t2.rip + 4 } :=
    step_ld (by rw [ht2rip]; exact hc.p25c) ht2r3 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 13 (some v) with rip := t2.rip + 4 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x260 := by
    rw [ht3]; show t2.rip + 4 = base + 0x260; rw [ht2rip]
  have ht3r13 : t3.reg 13 = some v := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 13 = some v
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 13)]
    exact ht2r6
  have ht3r7 : t3.reg 7 = some T := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 7 = some T
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 13)]
    exact ht2r7
  have ht3r4 : t3.reg 4 = some (sp1 + 8) := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact ht2r4
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t3.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi13]
    exact ht2fr i hi0 hi4 hi6 hi7 hi13
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x260: mov32 r13, r13
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 } :=
    step_mov32 (by rw [ht3rip]; exact hc.p260) ht3r13
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x263 := by
    rw [ht4]; show t3.rip + 3 = base + 0x263; rw [ht3rip]
  have ht4r13 : t4.reg 13 = some (v % 2 ^ 32) := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 13 =
      some (v % 2 ^ 32)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r6 : t4.reg 6 = some c := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 6 =
      some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 13)]
    exact ht3r6
  have ht4r7 : t4.reg 7 = some T := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 7 =
      some T
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 13)]
    exact ht3r7
  have ht4r4 : t4.reg 4 = some (sp1 + 8) := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact ht3r4
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t4.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi13]
    exact ht3fr i hi0 hi4 hi6 hi7 hi13
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with
      rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x263..0x268: six pop rax (initial-frame words, never read)
  obtain ⟨t5, r5, ht5rip, -, ht5r4, ht5fr, ht5mem, ht5len, ht5k⟩ :=
    pop0_step (t := t4) (by rw [ht4rip]; exact hc.p263) ht4r4
      (by omega) (by rw [ht4mem]; exact hp1) ht4len
  have ht5rip' : t5.rip = base + 0x264 := by rw [ht5rip, ht4rip]
  obtain ⟨t6, r6, ht6rip, -, ht6r4, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    pop0_step (t := t5) (by rw [ht5rip']; exact hc.p264) ht5r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 = sp1 + 16 := by omega
        rw [haddr, ht5mem, ht4mem]; exact hp2) ht5len
  have ht6rip' : t6.rip = base + 0x265 := by rw [ht6rip, ht5rip']
  obtain ⟨t7, r7, ht7rip, -, ht7r4, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    pop0_step (t := t6) (by rw [ht6rip']; exact hc.p265) ht6r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 = sp1 + 24 := by omega
        rw [haddr, ht6mem, ht5mem, ht4mem]; exact hp3) ht6len
  have ht7rip' : t7.rip = base + 0x266 := by rw [ht7rip, ht6rip']
  obtain ⟨t8, r8, ht8rip, -, ht8r4, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    pop0_step (t := t7) (by rw [ht7rip']; exact hc.p266) ht7r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 = sp1 + 32 := by omega
        rw [haddr, ht7mem, ht6mem, ht5mem, ht4mem]; exact hp4) ht7len
  have ht8rip' : t8.rip = base + 0x267 := by rw [ht8rip, ht7rip']
  obtain ⟨t9, r9, ht9rip, -, ht9r4, ht9fr, ht9mem, ht9len, ht9k⟩ :=
    pop0_step (t := t8) (by rw [ht8rip']; exact hc.p267) ht8r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 + 8 = sp1 + 40 := by omega
        rw [haddr, ht8mem, ht7mem, ht6mem, ht5mem, ht4mem]
        exact hp5) ht8len
  have ht9rip' : t9.rip = base + 0x268 := by rw [ht9rip, ht8rip']
  obtain ⟨t10, r10, ht10rip, ht10r0, ht10r4, ht10fr, ht10mem, ht10len,
      ht10k⟩ :=
    pop0_step (t := t9) (by rw [ht9rip']; exact hc.p268) ht9r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 + 8 + 8 = sp1 + 48 := by omega
        rw [haddr, ht9mem, ht8mem, ht7mem, ht6mem, ht5mem, ht4mem]
        exact hp6) ht9len
  have ht10rip' : t10.rip = base + 0x269 := by rw [ht10rip, ht9rip']
  -- registers 6, 7, 13 survive the six pops; frame composes
  have fr10 : ∀ i, i ≠ 0 → i ≠ 4 → t10.reg i = t4.reg i := by
    intro i hi0 hi4
    rw [ht10fr i hi0 hi4, ht9fr i hi0 hi4, ht8fr i hi0 hi4,
      ht7fr i hi0 hi4, ht6fr i hi0 hi4, ht5fr i hi0 hi4]
  have ht10r6 : t10.reg 6 = some c := by
    rw [fr10 6 (by decide) (by decide)]; exact ht4r6
  have ht10r7 : t10.reg 7 = some T := by
    rw [fr10 7 (by decide) (by decide)]; exact ht4r7
  have ht10r13 : t10.reg 13 = some (v % 2 ^ 32) := by
    rw [fr10 13 (by decide) (by decide)]; exact ht4r13
  have ht10r4' : t10.reg 4 = some (sp1 + 56) := by
    have haddr : sp1 + 8 + 8 + 8 + 8 + 8 + 8 + 8 = sp1 + 56 := by omega
    rw [ht10r4, haddr]
  have ht10mem' : t10.mem = s.mem := by
    rw [ht10mem, ht9mem, ht8mem, ht7mem, ht6mem, ht5mem]; exact ht4mem
  have ht10k' : t10.k = s.k := by
    rw [ht10k, ht9k, ht8k, ht7k, ht6k, ht5k]; exact ht4k
  -- 0x269: jmp word loop
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x2c9 } :=
    step_jmp (by rw [ht10rip']; exact hc.p269)
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x2c9 } := ⟨_, rfl⟩
  -- compose the 11 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
  refine ⟨t11, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht11]
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 6 = some c
    rw [reg_with_rip]; exact ht10r6
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 7 = some T
    rw [reg_with_rip]; exact ht10r7
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 13 = some (v % 2 ^ 32)
    rw [reg_with_rip]; exact ht10r13
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 4 = some (sp1 + 56)
    rw [reg_with_rip]; exact ht10r4'
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 0 = some w6
    rw [reg_with_rip]; exact ht10r0
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).mem = s.mem
    exact ht10mem'
  · intro i hi0 hi4 hi6 hi7 hi13
    rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg i = s.reg i
    rw [reg_with_rip, fr10 i hi0 hi4]
    exact ht4fr i hi0 hi4 hi6 hi7 hi13
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).regs.length = 16
    rw [len_with_rip]; exact ht10len
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).k = s.k
    exact ht10k'

/-- Word-loop entry: copy `nwords` from `r13` into the loop
counter `rcx`. -/
theorem parser_words_init {nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2c9) (hr13 : s.reg 13 = some nw)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some nw ∧
      (∀ i, i ≠ 1 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 1 (some nw) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p2c9) hr13
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 1 (some nw) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + 3 = base + 0x2cc; rw [hrip]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).reg 1 = some nw
    rw [reg_with_rip, reg_set_self (by omega)]
  · intro i hi
    rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  · rw [ht1]; exact rfl

/-- Word-loop head with words remaining: enter the body. -/
theorem parser_words_head_pos {n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2cc) (hr1 : s.reg 1 = some n)
    (hn : n ≠ 0) (hnW : n < W) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2d6 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p2cc) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x2cf := by
    rw [ht1]; show s.rip + 3 = base + 0x2cf; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some true := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [Nat.mod_eq_of_lt hnW, hn]
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x2d6 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p2cf) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x2d6 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
  · intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).reg i = s.reg i
    rw [reg_with_rip]
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).mem = s.mem
    rw [ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).k = s.k
    rw [ht1]

/-- Word-loop head with no words remaining: jump to the trailing
check. -/
theorem parser_words_head_zero {s : X86Int.St}
    (hrip : s.rip = base + 0x2cc) (hr1 : s.reg 1 = some 0) :
    ∃ s' : X86Int.St,
      run stage1Im 3 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x419 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p2cc) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x2cf := by
    rw [ht1]; show s.rip + 3 = base + 0x2cf; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some false := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p2cf) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x2d1 := by
    rw [ht2]; show t1.rip + 2 = base + 0x2d1; rw [ht1rip]
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x419 } :=
    step_jmp (by rw [ht2rip]; exact hc.p2d1)
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x419 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have hrun : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
    rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
  refine ⟨t3, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht3]
  · intro i
    rw [ht3]
    show ({ t2 with rip := base + 0x419 }).reg i = s.reg i
    rw [reg_with_rip, ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).mem = s.mem
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [ht1]
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).regs.length = s.regs.length
    rw [len_with_rip, ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).k = s.k
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).k = s.k
    rw [ht1]

/-- One table-slot store `st [rdi+off], r` with full frame. -/
theorem store_slot {site off slen p v r : Nat} {m : Mem} {s : X86Int.St}
    (hst : stage1Im site = some (.st 7 off r, slen))
    (hrip : s.rip = site) (hr7 : s.reg 7 = some p)
    (hrv : s.reg r = some v) (hpW : p + off < W)
    (hm : wr64 s.mem (p + off) v = some m) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = site + slen ∧ s'.mem = m ∧
      rd64 s'.mem (p + off) = some (v % W) ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, p + off + 8 ≤ x ∨ x + k ≤ p + off →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hea : ea p off = p + off := by
    unfold ea
    rw [Nat.mod_eq_of_lt hpW]
  have hm' : wr64 s.mem (ea p off) v = some m := by rwa [hea]
  have e1 : X86Int.step stage1Im s =
      .ok { s with mem := m, rip := s.rip + slen } :=
    step_st (by rw [hrip]; exact hst) hr7 hrv hm'
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with mem := m, rip := s.rip + slen } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have ht1mem : t1.mem = m := by rw [ht1]
  refine ⟨t1, r1, ?_, ht1mem, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + slen = site + slen; rw [hrip]
  · rw [ht1mem]; exact rd64_wr64 hm
  · intro i
    rw [ht1]
    show ({ s with mem := m, rip := s.rip + slen }).reg i = s.reg i
    exact rfl
  · rw [ht1]
  · rw [ht1]
  · rw [ht1mem]
    exact wrN_wshape _ _ _ hm
  · intro x k hd
    rw [ht1mem]
    exact rdN_wr64_other hm hd

/-- One word field: `call RDC` then store `rax` at `[rdi+off]`. -/
theorem field_store {site off slen c e sp p : Nat} {bs4 : List Nat}
    {s : X86Int.St}
    (hcall : stage1Im site = some (.call aRDC, 5))
    (hst : stage1Im (site + 5) = some (.st 7 off 0, slen))
    (hsiteW : site + 5 < W)
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : s.mem.rdN c 4 = some bs4) (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hdisj : c + 4 ≤ sp - 16 ∨ sp ≤ c)
    (hr7 : s.reg 7 = some p) (hpW : p + off < W)
    (hdisjp : p + off + 8 ≤ c ∨ c + 4 ≤ p + off)
    (hwr : (wr64 s.mem (p + off) (leVal bs4)).isSome) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 5 + slen ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      rd64 s'.mem (p + off) = some (leVal bs4 % W) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      s'.mem.rdN c 4 = some bs4 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + off + 8 ≤ x ∨ x + k ≤ p + off) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r0, hs1r6, hs1r4, hs1fr, hs1len,
      hs1memfr, hs1rd, hs1k, hs1memwr⟩ :=
    call_rdc_ok hcall hsiteW hrip hr6 hr15 hce hcW heW
      (by rw [hrd]; exact Option.isSome_some) hroom hr4 hsp16 hspW
      hlen16 hdisj
  obtain ⟨m1, m2, hm1, hm2, hs1mem⟩ := hs1memwr
  have hs1r0v : s1.reg 0 = some (leVal bs4) := by
    rw [hs1r0, hrd]; rfl
  have hs1r7 : s1.reg 7 = some p := by
    rw [hs1fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hr7
  have hwr1 : (wr64 m1 (p + off) (leVal bs4)).isSome :=
    wr64_some_of_wr64 hm1 hwr
  have hwr2 : (wr64 s1.mem (p + off) (leVal bs4)).isSome := by
    rw [hs1mem]
    exact wr64_some_of_wr64 hm2 hwr1
  obtain ⟨m3, hm3⟩ : ∃ m, wr64 s1.mem (p + off) (leVal bs4) = some m :=
    Option.isSome_iff_exists.mp hwr2
  obtain ⟨s2, hrun2, hs2rip, hs2mem, hs2slot, hs2fr, hs2len, hs2k,
      hs2sh, hs2memfr⟩ :=
    store_slot hst hs1rip hs1r7 hs1r0v hpW hm3
  have hsh : wshape s2.mem = wshape s.mem := by
    rw [hs2sh, hs1mem]
    have h2 : wshape m2 = wshape m1 := wrN_wshape _ _ _ hm2
    have h1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1
    rw [h2, h1]
  have hroom2 : StackRoom s2 2 := by
    apply StackRoom_of_wshape hsh _ hroom
    rw [hs2fr 4]
    rw [hs1r4, hr4]
  have hrun : run stage1Im (n1 + 1) (Out.ok s) = Out.ok s2 := by
    rw [run_add stage1Im n1 1, hrun1, hrun2]
  refine ⟨n1 + 1, s2, hrun, hs2rip, ?_, ?_, ?_, ?_, ?_, ?_, hsh, hroom2,
    ?_, ?_⟩
  · rw [hs2fr 6]; exact hs1r6
  · rw [hs2fr 4]; exact hs1r4
  · exact hs2slot
  · intro i hi0 hi2 hi4 hi6
    rw [hs2fr i]
    exact hs1fr i hi0 hi2 hi4 hi6
  · rw [hs2len]; exact hs1len
  · rw [hs2k]; exact hs1k
  · rw [hs2memfr c 4 hdisjp, hs1rd, hrd]
  · intro x k hd1 hd2
    rw [hs2memfr x k hd2]
    exact hs1memfr x k hd1

/-- The five header fields of one word: five `RDC` reads stored
at entry slots 0, 8, 16, 24, 32, then the cursor (code pointer)
stored at slot 40.  Ends at the code-loop setup. -/
theorem parser_word_fields {c e sp p : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr7 : s.reg 7 = some p)
    (hr4 : s.reg 4 = some sp)
    (h20 : s.mem.rdN c 20 = some bs) (hce : c + 20 ≤ e)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hwr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s.mem (p + off) v).isSome) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x306 ∧ s'.reg 6 = some (c + 20) ∧
      s'.reg 4 = some sp ∧ s'.reg 7 = some p ∧
      rd64 s'.mem p = some (leVal (bs.take 4) % W) ∧
      rd64 s'.mem (p + 8) = some (leVal ((bs.drop 4).take 4) % W) ∧
      rd64 s'.mem (p + 16) = some (leVal ((bs.drop 8).take 4) % W) ∧
      rd64 s'.mem (p + 24) = some (leVal ((bs.drop 12).take 4) % W) ∧
      rd64 s'.mem (p + 32) = some (leVal ((bs.drop 16).take 4) % W) ∧
      rd64 s'.mem (p + 40) = some ((c + 20) % W) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      s'.mem.rdN c 20 = some bs ∧
      (∀ (off v : Nat), off + 8 ≤ 64 →
        (wr64 s'.mem (p + off) v).isSome) ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) → s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hbW : base + 0x400 < W := by decide
  -- field slices of the 20 header bytes
  have hf0 : s.mem.rdN c 4 = some (bs.take 4) := by
    have h := rdN_slice h20 (show 0 + 4 ≤ 20 by decide)
    simpa using h
  -- slot 0
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r6, hs1r4, hs1slot, hs1fr, hs1len,
      hs1k, hs1sh, hs1room, hs1rd, hs1memfr⟩ :=
    field_store (site := base + 0x2d6) (off := 0) (slen := 3)
      hc.p2d6 hc.p2db (Nat.lt_of_le_of_lt (by omega) hbW) hrip hr6 hr15 (by omega) hcW heW
      hf0 hroom hr4 hsp16 hspW hlen16 (by omega) hr7 (by omega)
      (by omega) (by
        have h := hwr 0 (leVal (bs.take 4)) (by decide)
        simpa using h)
  have hs1rip' : s1.rip = base + 0x2de := hs1rip
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have hs1r7 : s1.reg 7 = some p := by
    rw [hs1fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hr7
  have hs1h20 : s1.mem.rdN c 20 = some bs := by
    rw [hs1memfr c 20 (by omega) (by omega)]; exact h20
  have hs1wr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s1.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s.mem s1.mem hs1sh _
      (hwr off v hoff)
  have hf1 : s1.mem.rdN (c + 4) 4 = some ((bs.drop 4).take 4) :=
    rdN_slice hs1h20 (show 4 + 4 ≤ 20 by decide)
  -- slot 8
  obtain ⟨n2, s2, hrun2, hs2rip, hs2r6, hs2r4, hs2slot, hs2fr, hs2len,
      hs2k, hs2sh, hs2room, hs2rd, hs2memfr⟩ :=
    field_store (site := base + 0x2de) (off := 8) (slen := 4)
      hc.p2de hc.p2e3 (Nat.lt_of_le_of_lt (by omega) hbW) hs1rip' hs1r6 hs1r15 (by omega)
      (by omega) heW hf1 hs1room hs1r4 hsp16 hspW hs1len (by omega)
      hs1r7 (by omega) (by omega) (hs1wr 8 _ (by decide))
  have hs2rip' : s2.rip = base + 0x2e7 := hs2rip
  have hs2r15 : s2.reg 15 = some e := by
    rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r15
  have hs2r7 : s2.reg 7 = some p := by
    rw [hs2fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r7
  have hs2h20 : s2.mem.rdN c 20 = some bs := by
    rw [hs2memfr c 20 (by omega) (by omega)]; exact hs1h20
  have hs2wr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s2.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s1.mem s2.mem hs2sh _
      (hs1wr off v hoff)
  have hs2slot0 : rd64 s2.mem p = some (leVal (bs.take 4) % W) := by
    have h : s2.mem.rdN p 8 = s1.mem.rdN p 8 :=
      hs2memfr p 8 (by omega) (by omega)
    unfold rd64
    rw [h]
    have h1 : rd64 s1.mem (p + 0) = some (leVal (bs.take 4) % W) :=
      hs1slot
    unfold rd64 at h1
    simpa using h1
  have hf2 : s2.mem.rdN (c + 8) 4 = some ((bs.drop 8).take 4) :=
    rdN_slice hs2h20 (show 8 + 4 ≤ 20 by decide)
  have hs2r6' : s2.reg 6 = some (c + 8) := by
    have e1 : c + 4 + 4 = c + 8 := by omega
    rwa [e1] at hs2r6
  -- slot 16
  obtain ⟨n3, s3, hrun3, hs3rip, hs3r6, hs3r4, hs3slot, hs3fr, hs3len,
      hs3k, hs3sh, hs3room, hs3rd, hs3memfr⟩ :=
    field_store (site := base + 0x2e7) (off := 16) (slen := 4)
      hc.p2e7 hc.p2ec (Nat.lt_of_le_of_lt (by omega) hbW) hs2rip' hs2r6' hs2r15 (by omega)
      (by omega) heW hf2 hs2room hs2r4 hsp16 hspW hs2len (by omega)
      hs2r7 (by omega) (by omega) (hs2wr 16 _ (by decide))
  have hs3rip' : s3.rip = base + 0x2f0 := hs3rip
  have hs3r15 : s3.reg 15 = some e := by
    rw [hs3fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r15
  have hs3r7 : s3.reg 7 = some p := by
    rw [hs3fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r7
  have hs3h20 : s3.mem.rdN c 20 = some bs := by
    rw [hs3memfr c 20 (by omega) (by omega)]; exact hs2h20
  have hs3wr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s3.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s2.mem s3.mem hs3sh _
      (hs2wr off v hoff)
  have hs3slot0 : rd64 s3.mem p = some (leVal (bs.take 4) % W) := by
    unfold rd64
    rw [hs3memfr p 8 (by omega) (by omega)]
    have h := hs2slot0
    unfold rd64 at h
    exact h
  have hs3slot8 : rd64 s3.mem (p + 8) =
      some (leVal ((bs.drop 4).take 4) % W) := by
    unfold rd64
    rw [hs3memfr (p + 8) 8 (by omega) (by omega)]
    have h := hs2slot
    unfold rd64 at h
    exact h
  have hf3 : s3.mem.rdN (c + 12) 4 = some ((bs.drop 12).take 4) :=
    rdN_slice hs3h20 (show 12 + 4 ≤ 20 by decide)
  have hs3r6' : s3.reg 6 = some (c + 12) := by
    have e1 : c + 8 + 4 = c + 12 := by omega
    rwa [e1] at hs3r6
  -- slot 24
  obtain ⟨n4, s4, hrun4, hs4rip, hs4r6, hs4r4, hs4slot, hs4fr, hs4len,
      hs4k, hs4sh, hs4room, hs4rd, hs4memfr⟩ :=
    field_store (site := base + 0x2f0) (off := 24) (slen := 4)
      hc.p2f0 hc.p2f5 (Nat.lt_of_le_of_lt (by omega) hbW) hs3rip' hs3r6' hs3r15 (by omega)
      (by omega) heW hf3 hs3room hs3r4 hsp16 hspW hs3len (by omega)
      hs3r7 (by omega) (by omega) (hs3wr 24 _ (by decide))
  have hs4rip' : s4.rip = base + 0x2f9 := hs4rip
  have hs4r15 : s4.reg 15 = some e := by
    rw [hs4fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs3r15
  have hs4r7 : s4.reg 7 = some p := by
    rw [hs4fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs3r7
  have hs4h20 : s4.mem.rdN c 20 = some bs := by
    rw [hs4memfr c 20 (by omega) (by omega)]; exact hs3h20
  have hs4wr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s4.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s3.mem s4.mem hs4sh _
      (hs3wr off v hoff)
  have hs4slot0 : rd64 s4.mem p = some (leVal (bs.take 4) % W) := by
    unfold rd64
    rw [hs4memfr p 8 (by omega) (by omega)]
    have h := hs3slot0
    unfold rd64 at h
    exact h
  have hs4slot8 : rd64 s4.mem (p + 8) =
      some (leVal ((bs.drop 4).take 4) % W) := by
    unfold rd64
    rw [hs4memfr (p + 8) 8 (by omega) (by omega)]
    have h := hs3slot8
    unfold rd64 at h
    exact h
  have hs4slot16 : rd64 s4.mem (p + 16) =
      some (leVal ((bs.drop 8).take 4) % W) := by
    unfold rd64
    rw [hs4memfr (p + 16) 8 (by omega) (by omega)]
    have h := hs3slot
    unfold rd64 at h
    exact h
  have hf4 : s4.mem.rdN (c + 16) 4 = some ((bs.drop 16).take 4) :=
    rdN_slice hs4h20 (show 16 + 4 ≤ 20 by decide)
  have hs4r6' : s4.reg 6 = some (c + 16) := by
    have e1 : c + 12 + 4 = c + 16 := by omega
    rwa [e1] at hs4r6
  -- slot 32
  obtain ⟨n5, s5, hrun5, hs5rip, hs5r6, hs5r4, hs5slot, hs5fr, hs5len,
      hs5k, hs5sh, hs5room, hs5rd, hs5memfr⟩ :=
    field_store (site := base + 0x2f9) (off := 32) (slen := 4)
      hc.p2f9 hc.p2fe (Nat.lt_of_le_of_lt (by omega) hbW) hs4rip' hs4r6' hs4r15 (by omega)
      (by omega) heW hf4 hs4room hs4r4 hsp16 hspW hs4len (by omega)
      hs4r7 (by omega) (by omega) (hs4wr 32 _ (by decide))
  have hs5rip' : s5.rip = base + 0x302 := hs5rip
  have hs5r7 : s5.reg 7 = some p := by
    rw [hs5fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs4r7
  have hs5h20 : s5.mem.rdN c 20 = some bs := by
    rw [hs5memfr c 20 (by omega) (by omega)]; exact hs4h20
  have hs5wr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s5.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s4.mem s5.mem hs5sh _
      (hs4wr off v hoff)
  have hs5r6' : s5.reg 6 = some (c + 20) := by
    have e1 : c + 16 + 4 = c + 20 := by omega
    rwa [e1] at hs5r6
  have slot_at : ∀ {off : Nat} {w : Nat},
      rd64 s4.mem (p + off) = some w → off + 8 ≤ 32 →
      rd64 s5.mem (p + off) = some w := by
    intro off w h hoff
    unfold rd64
    rw [hs5memfr (p + off) 8 (by omega) (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  -- slot 40: store the cursor (code pointer)
  obtain ⟨m6, hm6⟩ : ∃ m, wr64 s5.mem (p + 40) (c + 20) = some m :=
    Option.isSome_iff_exists.mp (hs5wr 40 _ (by decide))
  obtain ⟨s6, hrun6, hs6rip, hs6mem, hs6slot, hs6fr, hs6len, hs6k,
      hs6sh, hs6memfr⟩ :=
    store_slot (site := base + 0x302) (off := 40) (slen := 4)
      (r := 6) hc.p302 hs5rip' hs5r7 hs5r6' (by omega) hm6
  have hs6sh' : wshape s6.mem = wshape s.mem := by
    rw [hs6sh, hs5sh, hs4sh, hs3sh, hs2sh, hs1sh]
  have hrun : run stage1Im (n1 + n2 + n3 + n4 + n5 + 1) (Out.ok s) =
      Out.ok s6 := by
    rw [show n1 + n2 + n3 + n4 + n5 + 1 =
        n1 + (n2 + (n3 + (n4 + (n5 + 1)))) by omega,
      run_add stage1Im n1 _, hrun1, run_add stage1Im n2 _, hrun2,
      run_add stage1Im n3 _, hrun3, run_add stage1Im n4 _, hrun4,
      run_add stage1Im n5 1, hrun5, hrun6]
  have slot6 : ∀ {off : Nat} {w : Nat},
      rd64 s5.mem (p + off) = some w → off + 8 ≤ 40 →
      rd64 s6.mem (p + off) = some w := by
    intro off w h hoff
    unfold rd64
    rw [hs6memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have hslot0 : rd64 s6.mem p = some (leVal (bs.take 4) % W) := by
    have h5 : rd64 s5.mem (p + 0) = some (leVal (bs.take 4) % W) :=
      slot_at (off := 0) (by simpa using hs4slot0) (by decide)
    have h6 := slot6 (off := 0) h5 (by decide)
    simpa using h6
  have hroom6 : StackRoom s6 2 := by
    apply StackRoom_of_wshape hs6sh' _ hroom
    rw [hs6fr 4, hs5r4, hr4]
  refine ⟨n1 + n2 + n3 + n4 + n5 + 1, s6, hrun, hs6rip, ?_, ?_, ?_,
    hslot0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hs6sh', hroom6, ?_, ?_, ?_⟩
  · rw [hs6fr 6]; exact hs5r6'
  · rw [hs6fr 4]; exact hs5r4
  · rw [hs6fr 7]; exact hs5r7
  · exact slot6 (slot_at hs4slot8 (by decide)) (by decide)
  · exact slot6 (slot_at hs4slot16 (by decide)) (by decide)
  · have h5 : rd64 s5.mem (p + 24) =
        some (leVal ((bs.drop 12).take 4) % W) :=
      slot_at hs4slot (by decide)
    exact slot6 h5 (by decide)
  · exact slot6 hs5slot (by decide)
  · exact hs6slot
  · intro i hi0 hi2 hi4 hi6
    rw [hs6fr i, hs5fr i hi0 hi2 hi4 hi6, hs4fr i hi0 hi2 hi4 hi6,
      hs3fr i hi0 hi2 hi4 hi6, hs2fr i hi0 hi2 hi4 hi6,
      hs1fr i hi0 hi2 hi4 hi6]
  · rw [hs6len]; exact hs5len
  · rw [hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · rw [hs6memfr c 20 (by omega)]; exact hs5h20
  · intro off v hoff
    exact wrN_some_of_wshape _ s.mem s6.mem hs6sh' _
      (hwr off v hoff)
  · intro x k hd1 hd2
    rw [hs6memfr x k (by omega), hs5memfr x k hd1 (by omega),
      hs4memfr x k hd1 (by omega), hs3memfr x k hd1 (by omega),
      hs2memfr x k hd1 (by omega), hs1memfr x k hd1 (by omega)]

/-- Code-loop bounds check and opcode load: from the body entry
to the length dispatch, with the opcode byte in `rax`. -/
theorem parser_code_bounds {c e op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x315) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 1 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : s.mem.rd c = some op) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 6 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x330 ∧ s'.reg 0 = some op ∧
      s'.reg 6 = some c ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x315: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p315) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x318 := by
    rw [ht1]; show s.rip + 3 = base + 0x318; rw [hrip]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- 0x318: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p318) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x31b := by
    rw [ht2]; show t1.rip + 3 = base + 0x31b; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have ht2fl : X86Int.cond t2.fl 3 = some true := by
    have hcf : t2.fl.cf = some false := by
      rw [ht2]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x31b: jae 0x322
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x322 } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p31b) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x322 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x322 := by rw [ht3]
  have ht3r2 : t3.reg 2 = some (e - c) := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).reg 2 = some (e - c)
    rw [reg_with_rip]; exact ht2r2
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x322: cmp rdx, 1
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub (e - c) (1 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p322) ht3r2
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub (e - c) (1 % W)).2, rip := t3.rip + 4 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x326 := by
    rw [ht4]; show t3.rip + 4 = base + 0x326; rw [ht3rip]
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub (e - c) (1 % W)).2.cf = some false
      rw [h1W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show t3.reg i = s.reg i
    exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show t3.mem = s.mem
    exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]; show t3.regs.length = 16; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x326: jae 0x32d
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x32d } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p326) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x32d } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x32d := by rw [ht5]
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).reg 6 = some c
    rw [reg_with_rip, ht4fr 6 (by decide)]; exact hr6
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).reg i = s.reg i
    rw [reg_with_rip]; exact ht4fr i hi
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).mem = s.mem
    rw [mem_with_rip]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).regs.length = 16
    rw [len_with_rip]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  -- 0x32d: ld8 rax, [rsi]
  have hld : t5.mem.rd (ea c 0) = some op := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, ht5mem]; exact hrd
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some op) with rip := t5.rip + 3 } :=
    step_ld8 (by rw [ht5rip]; exact hc.p32d) ht5r6 hld
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some op) with rip := t5.rip + 3 } := ⟨_, rfl⟩
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have hrun : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
    rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
  refine ⟨t6, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht6]; show t5.rip + 3 = base + 0x330; rw [ht5rip]
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg 0 = some op
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r6
  · intro i hi0 hi2
    rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht5fr i hi2
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  · rw [ht6]; exact ht5k

/-- The length dispatch of the code loop, as a relation between
an opcode byte and its instruction length.  Mirrors the compare
chain at `0x330` and `decInstr` in `Sporos.Bytecode`. -/
def OpLen (op l : Nat) : Prop :=
  (op < 57 ∧ l = 1) ∨ (op = 128 ∧ l = 9) ∨
  ((op = 131 ∨ op = 132 ∨ op = 133 ∨ op = 140) ∧ l = 1) ∨
  ((129 ≤ op ∧ op ≤ 139 ∧ op ≠ 131 ∧ op ≠ 132 ∧ op ≠ 133) ∧ l = 5)

/-- Length dispatch, primitive opcodes: `cmp 57` not above-or-equal
falls to the length-1 assignment. -/
theorem parser_disp_lt57 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hop : op < 57) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 1 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have h57W : 57 % W = 57 := Nat.mod_eq_of_lt (by unfold W; decide)
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x330: cmp rax, 57
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub op (57 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact hc.p330) hr0
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .sub op (57 % W)).2, rip := s.rip + 4 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x334 := by
    rw [ht1]; show s.rip + 4 = base + 0x334; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 3 = some false := by
    have hcf : t1.fl.cf = some true := by
      rw [ht1]
      show (aluF .sub op (57 % W)).2.cf = some true
      rw [h57W]; exact aluF_sub_cf_true hop
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht1fr : ∀ i, t1.reg i = s.reg i := by
    intro i
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  have ht1mem : t1.mem = s.mem := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by rw [ht1]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]
  -- 0x334: jae not taken
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p334) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x336 := by
    rw [ht2]; show t1.rip + 2 = base + 0x336; rw [ht1rip]
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht1fr i
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x336: jmp 0x3ad
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x3ad } :=
    step_jmp (by rw [ht2rip]; exact hc.p336)
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x3ad } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3ad := by rw [ht3]
  have ht3fr : ∀ i, t3.reg i = s.reg i := by
    intro i
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x3ad: mov r10d, 1
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 10 (some (1 % W)) with rip := t3.rip + 6 } :=
    step_movI (by rw [ht3rip]; exact hc.p3ad)
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 10 (some (1 % W)) with rip := t3.rip + 6 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
    rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
  refine ⟨t4, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht4]; show t3.rip + 6 = base + 0x3b3; rw [ht3rip]
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg 10 =
      some 1
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide), h1W]
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg 0 =
      some op
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10),
      ht3fr 0]; exact hr0
  · intro i hi
    rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht3fr i
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with
      rip := t3.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  · rw [ht4]; exact ht3k

/-- A subtraction of distinct in-range values is nonzero. -/
theorem sub_fst_ne_zero {op k : Nat} (hop : op < W) (hk : k < W)
    (hne : op ≠ k) : (aluF .sub op k).1 ≠ 0 := by
  rw [aluF_sub_fst]
  by_cases h : k ≤ op
  · rw [show op + W - k = (op - k) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
    omega
  · rw [Nat.mod_eq_of_lt (by omega)]
    omega

/-- A self-subtraction is zero. -/
theorem sub_fst_self (k : Nat) : (aluF .sub k k).1 = 0 := by
  rw [aluF_sub_fst, Nat.add_sub_cancel_left, Nat.mod_self]

/-- One unconditional jump step with full frame. -/
theorem jmp_step {a t len : Nat} {s : X86Int.St}
    (hdec : stage1Im a = some (.jmp t, len)) (hrip : s.rip = a) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧ s'.rip = t ∧
      (∀ i, s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s = .ok { s with rip := t } :=
    step_jmp (by rw [hrip]; exact hdec)
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u = { s with rip := t } :=
    ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]
  · intro i
    rw [ht1]
    show ({ s with rip := t }).reg i = s.reg i
    rw [reg_with_rip]
  · rw [ht1]
  · rw [ht1]
  · rw [ht1]

/-- One `movI` step with full frame. -/
theorem movI_step {a d imm len : Nat} {s : X86Int.St}
    (hdec : stage1Im a = some (.movI d imm, len)) (hrip : s.rip = a)
    (hd : d < s.regs.length) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = a + len ∧ s'.reg d = some (imm % W) ∧
      (∀ i, i ≠ d → s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s =
      .ok { s.set d (some (imm % W)) with rip := s.rip + len } :=
    step_movI (by rw [hrip]; exact hdec)
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { s.set d (some (imm % W)) with rip := s.rip + len } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + len = a + len; rw [hrip]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).reg d =
      some (imm % W)
    rw [reg_with_rip, reg_set_self hd]
  · intro i hi
    rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).mem =
      s.mem
    rw [mem_with_rip, mem_set]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with
      rip := s.rip + len }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  · rw [ht1]; exact rfl

/-- One `cmpI`+`jcc` pair with a given condition outcome. -/
theorem cmpI_jcc_pair {a t k op clen cc : Nat} {b : Bool}
    {s : X86Int.St}
    (hcmp : stage1Im a = some (.cmpI 0 k, clen))
    (hjcc : stage1Im (a + clen) = some (.jcc cc t, 2))
    (hrip : s.rip = a) (hr0 : s.reg 0 = some op)
    (hfl : X86Int.cond (aluF .sub op (k % W)).2 cc = some b) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = (if b then t else a + clen + 2) ∧
      (∀ i, s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub op (k % W)).2, rip := s.rip + clen } :=
    step_cmpI (by rw [hrip]; exact hcmp) hr0
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { s with fl := (aluF .sub op (k % W)).2, rip := s.rip + clen } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = a + clen := by
    rw [ht1]; show s.rip + clen = a + clen; rw [hrip]
  have ht1fl : X86Int.cond t1.fl cc = some b := by
    rw [ht1]
    show X86Int.cond (aluF .sub op (k % W)).2 cc = some b
    exact hfl
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := if b then t else t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hjcc) ht1fl
    exact h
  obtain ⟨t2, ht2⟩ : ∃ u : X86Int.St, u =
      { t1 with rip := if b then t else t1.rip + 2 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
    show (if b then t else t1.rip + 2) = if b then t else a + clen + 2
    rw [ht1rip]
  · intro i
    rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).reg i =
      s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).regs.length =
      s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).k = s.k
    rw [ht1]

/-- Condition outcome for `cmp op, k` with `op ≠ k` under cc5. -/
theorem fl_jne {op k : Nat} (hop : op < W) (hk : k < W)
    (hne : op ≠ k) :
    X86Int.cond (aluF .sub op (k % W)).2 5 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hzf : (aluF .sub op (k % W)).2.zf = some false := by
    rw [aluF_zf, hkmod]
    have hbeq : ((aluF .sub op k).1 == 0) = false := by
      cases h : ((aluF .sub op k).1 == 0) with
      | false => rfl
      | true =>
        rw [beq_iff_eq] at h
        exact absurd h (sub_fst_ne_zero hop hk hne)
    rw [hbeq]
  have hcc := cond_cc5 hzf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op = k` under cc5. -/
theorem fl_je {op k : Nat} (hk : k < W) (heq : op = k) :
    X86Int.cond (aluF .sub op (k % W)).2 5 = some false := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hzf : (aluF .sub op (k % W)).2.zf = some true := by
    rw [aluF_zf, hkmod, heq]
    have hbeq : ((aluF .sub k k).1 == 0) = true := by
      rw [beq_iff_eq]; exact sub_fst_self k
    rw [hbeq]
  have hcc := cond_cc5 hzf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `k ≤ op` under cc3. -/
theorem fl_jae {op k : Nat} (hk : k < W) (h : ¬ op < k) :
    X86Int.cond (aluF .sub op (k % W)).2 3 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hcf : (aluF .sub op (k % W)).2.cf = some false := by
    rw [hkmod]; exact aluF_sub_cf h
  have hcc := cond_cc3 hcf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op < k` under cc3. -/
theorem fl_jb {op k : Nat} (hk : k < W) (h : op < k) :
    X86Int.cond (aluF .sub op (k % W)).2 3 = some false := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hcf : (aluF .sub op (k % W)).2.cf = some true := by
    rw [hkmod]; exact aluF_sub_cf_true h
  have hcc := cond_cc3 hcf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op ≤ k` under cc6. -/
theorem fl_jbe {op k : Nat} (hop : op < W) (hk : k < W)
    (h : op ≤ k) :
    X86Int.cond (aluF .sub op (k % W)).2 6 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  by_cases hlt : op < k
  · have hcf : (aluF .sub op (k % W)).2.cf = some true := by
      rw [hkmod]; exact aluF_sub_cf_true hlt
    have hzf : (aluF .sub op (k % W)).2.zf =
        some ((aluF .sub op (k % W)).1 == 0) :=
      aluF_zf .sub _ _
    simp [X86Int.cond, hcf, hzf]
  · have heq : op = k := by omega
    have hcf : (aluF .sub op (k % W)).2.cf = some false := by
      rw [hkmod]; exact aluF_sub_cf (by omega)
    have hzf : (aluF .sub op (k % W)).2.zf = some true := by
      rw [aluF_zf, hkmod, heq]
      have hbeq : ((aluF .sub k k).1 == 0) = true := by
        rw [beq_iff_eq]; exact sub_fst_self k
      rw [hbeq]
    simp [X86Int.cond, hcf, hzf]

/-- Length dispatch, literal opcode `0x80`: length 9. -/
theorem parser_disp_128 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (heq : op = 128) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 7 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 9 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_je (by decide) heq)
  have hs2rip' : s2.rip = base + 0x344 := hs2rip
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    jmp_step hc.p344 hs2rip'
  have hs3len16 : s3.regs.length = 16 := by
    rw [hs3len, hs2len, hs1len]; exact hlen16
  obtain ⟨s4, hrun4, hs4rip, hs4r10, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    movI_step hc.p3a5 hs3rip (by rw [hs3len16]; decide)
  have hs4rip' : s4.rip = base + 0x3ab := hs4rip
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    jmp_step hc.p3ab hs4rip'
  have hrun : run stage1Im 7 (Out.ok s) = Out.ok s5 := by
    rw [show (7 : Nat) = 2 + 5 by omega, run_add stage1Im 2 5, hrun1,
      show (5 : Nat) = 2 + 3 by omega, run_add stage1Im 2 3, hrun2,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, hrun3,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun4,
      hrun5]
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  refine ⟨s5, hrun, hs5rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs5fr 10, hs4r10, h9W]
  · rw [hs5fr 0, hs4fr 0 (by decide), hs3fr 0, hs2fr 0, hs1fr 0]
    exact hr0
  · intro i hi
    rw [hs5fr i, hs4fr i hi, hs3fr i, hs2fr i, hs1fr i]
  · rw [hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
  · rw [hs5len, hs4len]; exact hs3len16
  · rw [hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Length dispatch, 5-byte opcodes (`129..139` except the
1-byte specials): the whole compare chain, ending at the
length-5 assignment. -/
theorem parser_disp_5 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 129 ≤ op) (hhi : op ≤ 139)
    (h131 : op ≠ 131) (h132 : op ≠ 132) (h133 : op ≠ 133)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 18 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 5 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hopW : op < W := Nat.lt_of_le_of_lt hhi (by decide)
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_jne hopW (by decide) (by omega))
  have hs2rip' : s2.rip = base + 0x349 := hs2rip
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
      (fl_jne hopW (by decide) h131)
  have hs3rip' : s3.rip = base + 0x357 := hs3rip
  have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
      (fl_jne hopW (by decide) h132)
  have hs4rip' : s4.rip = base + 0x365 := hs4rip
  have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
      (fl_jne hopW (by decide) h133)
  have hs5rip' : s5.rip = base + 0x373 := hs5rip
  have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
  obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
    cmpI_jcc_pair hc.p373 hc.p37a hs5rip' hs5r0
      (fl_jne hopW (by decide) (by omega))
  have hs6rip' : s6.rip = base + 0x381 := hs6rip
  have hs6r0 : s6.reg 0 = some op := by rw [hs6fr 0]; exact hs5r0
  obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
    cmpI_jcc_pair hc.p381 hc.p388 hs6rip' hs6r0
      (fl_jae (by decide) (by omega))
  have hs7rip' : s7.rip = base + 0x38f := hs7rip
  have hs7r0 : s7.reg 0 = some op := by rw [hs7fr 0]; exact hs6r0
  obtain ⟨s8, hrun8, hs8rip, hs8fr, hs8mem, hs8len, hs8k⟩ :=
    cmpI_jcc_pair hc.p38f hc.p396 hs7rip' hs7r0
      (fl_jbe hopW (by decide) hhi)
  have hs8rip' : s8.rip = base + 0x39d := hs8rip
  have hs8len16 : s8.regs.length = 16 := by
    rw [hs8len, hs7len, hs6len, hs5len, hs4len, hs3len, hs2len,
      hs1len]; exact hlen16
  obtain ⟨s9, hrun9, hs9rip, hs9r10, hs9fr, hs9mem, hs9len, hs9k⟩ :=
    movI_step hc.p39d hs8rip' (by rw [hs8len16]; decide)
  have hs9rip' : s9.rip = base + 0x3a3 := hs9rip
  obtain ⟨s10, hrun10, hs10rip, hs10fr, hs10mem, hs10len, hs10k⟩ :=
    jmp_step hc.p3a3 hs9rip'
  have hrun : run stage1Im 18 (Out.ok s) = Out.ok s10 := by
    rw [show (18 : Nat) = 2 + 16 by omega, run_add stage1Im 2 16, hrun1,
      show (16 : Nat) = 2 + 14 by omega, run_add stage1Im 2 14, hrun2,
      show (14 : Nat) = 2 + 12 by omega, run_add stage1Im 2 12, hrun3,
      show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10, hrun4,
      show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8, hrun5,
      show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun6,
      show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun7,
      show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun8,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun9,
      hrun10]
  have h5W : 5 % W = 5 := Nat.mod_eq_of_lt (by unfold W; decide)
  refine ⟨s10, hrun, hs10rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs10fr 10, hs9r10, h5W]
  · rw [hs10fr 0, hs9fr 0 (by decide), hs8fr 0, hs7fr 0, hs6fr 0,
      hs5fr 0, hs4fr 0, hs3fr 0, hs2fr 0, hs1fr 0]
    exact hr0
  · intro i hi
    rw [hs10fr i, hs9fr i hi, hs8fr i, hs7fr i, hs6fr i, hs5fr i,
      hs4fr i, hs3fr i, hs2fr i, hs1fr i]
  · rw [hs10mem, hs9mem, hs8mem, hs7mem, hs6mem, hs5mem, hs4mem,
      hs3mem, hs2mem, hs1mem]
  · rw [hs10len, hs9len]; exact hs8len16
  · rw [hs10k, hs9k, hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Length dispatch, 1-byte special opcodes `131, 132, 133, 140`:
the chain falls out at the matching equality test to the
length-1 assignment. -/
theorem parser_disp_spec {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hspec : op = 131 ∨ op = 132 ∨ op = 133 ∨ op = 140)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 1 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hopW : op < W := by
    have h1 : op ≤ 140 := by omega
    exact Nat.lt_of_le_of_lt h1 (by decide)
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_jne hopW (by decide) (by omega))
  have hs2rip' : s2.rip = base + 0x349 := hs2rip
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  rcases hspec with h131 | h132 | h133 | h140
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_je (by decide) h131)
    have hs3rip' : s3.rip = base + 0x352 := hs3rip
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      jmp_step hc.p352 hs3rip'
    have hs4len16 : s4.regs.length = 16 := by
      rw [hs4len, hs3len, hs2len, hs1len]; exact hlen16
    obtain ⟨s5, hrun5, hs5rip, hs5r10, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      movI_step hc.p3ad hs4rip (by rw [hs4len16]; decide)
    have hrun : run stage1Im 8 (Out.ok s) = Out.ok s5 := by
      rw [show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun1,
        show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun2,
        show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun3,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun4,
        hrun5]
    refine ⟨8, s5, hrun, hs5rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs5r10, h1W]
    · rw [hs5fr 0 (by decide), hs4fr 0, hs3fr 0, hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs5fr i hi, hs4fr i, hs3fr i, hs2fr i, hs1fr i]
    · rw [hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs5len]; exact hs4len16
    · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_je (by decide) h132)
    have hs4rip' : s4.rip = base + 0x360 := hs4rip
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      jmp_step hc.p360 hs4rip'
    have hs5len16 : s5.regs.length = 16 := by
      rw [hs5len, hs4len, hs3len, hs2len, hs1len]; exact hlen16
    obtain ⟨s6, hrun6, hs6rip, hs6r10, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      movI_step hc.p3ad hs5rip (by rw [hs5len16]; decide)
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok s6 := by
      rw [show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8, hrun1,
        show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun2,
        show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun3,
        show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun4,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun5,
        hrun6]
    refine ⟨10, s6, hrun, hs6rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs6r10, h1W]
    · rw [hs6fr 0 (by decide), hs5fr 0, hs4fr 0, hs3fr 0, hs2fr 0,
        hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs6fr i hi, hs5fr i, hs4fr i, hs3fr i, hs2fr i, hs1fr i]
    · rw [hs6mem, hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs6len]; exact hs5len16
    · rw [hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_jne hopW (by decide) (by omega))
    have hs4rip' : s4.rip = base + 0x365 := hs4rip
    have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
        (fl_je (by decide) h133)
    have hs5rip' : s5.rip = base + 0x36e := hs5rip
    obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      jmp_step hc.p36e hs5rip'
    have hs6len16 : s6.regs.length = 16 := by
      rw [hs6len, hs5len, hs4len, hs3len, hs2len, hs1len]
      exact hlen16
    obtain ⟨s7, hrun7, hs7rip, hs7r10, hs7fr, hs7mem, hs7len, hs7k⟩ :=
      movI_step hc.p3ad hs6rip (by rw [hs6len16]; decide)
    have hrun : run stage1Im 12 (Out.ok s) = Out.ok s7 := by
      rw [show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10,
        hrun1, show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8,
        hrun2, show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6,
        hrun3, show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4,
        hrun4, show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2,
        hrun5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        hrun6, hrun7]
    refine ⟨12, s7, hrun, hs7rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs7r10, h1W]
    · rw [hs7fr 0 (by decide), hs6fr 0, hs5fr 0, hs4fr 0, hs3fr 0,
        hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs7fr i hi, hs6fr i, hs5fr i, hs4fr i, hs3fr i, hs2fr i,
        hs1fr i]
    · rw [hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs7len]; exact hs6len16
    · rw [hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_jne hopW (by decide) (by omega))
    have hs4rip' : s4.rip = base + 0x365 := hs4rip
    have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
        (fl_jne hopW (by decide) (by omega))
    have hs5rip' : s5.rip = base + 0x373 := hs5rip
    have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
    obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      cmpI_jcc_pair hc.p373 hc.p37a hs5rip' hs5r0
        (fl_je (by decide) h140)
    have hs6rip' : s6.rip = base + 0x37c := hs6rip
    obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
      jmp_step hc.p37c hs6rip'
    have hs7len16 : s7.regs.length = 16 := by
      rw [hs7len, hs6len, hs5len, hs4len, hs3len, hs2len, hs1len]
      exact hlen16
    obtain ⟨s8, hrun8, hs8rip, hs8r10, hs8fr, hs8mem, hs8len, hs8k⟩ :=
      movI_step hc.p3ad hs7rip (by rw [hs7len16]; decide)
    have hrun : run stage1Im 14 (Out.ok s) = Out.ok s8 := by
      rw [show (14 : Nat) = 2 + 12 by omega, run_add stage1Im 2 12,
        hrun1, show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10,
        hrun2, show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8,
        hrun3, show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6,
        hrun4, show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4,
        hrun5, show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2,
        hrun6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        hrun7, hrun8]
    refine ⟨14, s8, hrun, hs8rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs8r10, h1W]
    · rw [hs8fr 0 (by decide), hs7fr 0, hs6fr 0, hs5fr 0, hs4fr 0,
        hs3fr 0, hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs8fr i hi, hs7fr i, hs6fr i, hs5fr i, hs4fr i, hs3fr i,
        hs2fr i, hs1fr i]
    · rw [hs8mem, hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem,
        hs1mem]
    · rw [hs8len]; exact hs7len16
    · rw [hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- The whole length dispatch, by `OpLen` case. -/
theorem parser_dispatch {op l : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hOp : OpLen op l) (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some l ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  rcases hOp with ⟨h1, hl⟩ | ⟨h1, hl⟩ | ⟨h1, hl⟩ | ⟨h1, hl⟩
  · subst hl
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_lt57 hrip hr0 h1 hlen16
    exact ⟨4, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_128 hrip hr0 h1 hlen16
    exact ⟨7, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨n, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_spec hrip hr0 h1 hlen16
    exact ⟨n, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨hlo, hhi, h131, h132, h133⟩ := h1
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_5 hrip hr0 hlo hhi h131 h132 h133 hlen16
    exact ⟨18, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩

/-- `rdN` outside a single-byte write is unchanged. -/
theorem rdN_wr_other {m m' : Mem} {a v x n : Nat}
    (h : m.wr a v = some m') (hd : a + 1 ≤ x ∨ x + n ≤ a) :
    m'.rdN x n = m.rdN x n := by
  have hN : m.wrN a [v] = some m' := by simp [Mem.wrN, h]
  exact rdN_wrN_other n [v] m a x hN (fun i hi j _ => by
    simp at hi
    omega)

/-- Code-loop tail: length check, store the length byte in the
scratch at `S + (c - B)`, advance the cursor, decrement the
count, and jump back to the loop head. -/
theorem parser_code_tail {c e B S l n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x3b3) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr10 : s.reg 10 = some l)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (hr1 : s.reg 1 = some (n + 1))
    (hce : c + l ≤ e) (hcW : c < W) (heW : e < W)
    (hBc : B ≤ c) (hSW : S + (c - B) < W)
    (hl256 : l < 256) (hnW : n < W)
    (hwr : (s.mem.wr (S + (c - B)) l).isSome)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 6 = some (c + l) ∧
      s'.reg 1 = some n ∧ s'.reg 10 = some l ∧
      s'.mem.rd (S + (c - B)) = some l ∧
      (∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, S + (c - B) + 1 ≤ x ∨ x + k ≤ S + (c - B) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x3b3: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p3b3) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x3b6 := by
    rw [ht1]; show s.rip + 3 = base + 0x3b6; rw [hrip]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t1.reg i = s.reg i := by
    intro i _ hi2 _
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1r10 : t1.reg 10 = some l := by
    rw [ht1fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- 0x3b6: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p3b6) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x3b9 := by
    rw [ht2]; show t1.rip + 3 = base + 0x3b9; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t2.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht1fr i hi1 hi2 hi6
  have ht2r10 : t2.reg 10 = some l := by
    rw [ht2fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x3b9: cmp rdx, r10
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with fl := (aluF .sub (e - c) l).2, rip := t2.rip + 3 } :=
    step_cmp (by rw [ht2rip]; exact hc.p3b9) ht2r2 ht2r10
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with fl := (aluF .sub (e - c) l).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3bc := by
    rw [ht3]; show t2.rip + 3 = base + 0x3bc; rw [ht2rip]
  have ht3fl : X86Int.cond t3.fl 3 = some true := by
    have hcf : t3.fl.cf = some false := by
      rw [ht3]
      show (aluF .sub (e - c) l).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht3fr : ∀ i, t3.reg i = t2.reg i := by
    intro i
    rw [ht3]
    show t2.reg i = t2.reg i
    exact rfl
  have ht3mem : t3.mem = s.mem := by rw [ht3]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by rw [ht3]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x3bc: jae 0x3c3
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x3c3 } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p3bc) ht3fl
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x3c3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x3c3 := by rw [ht4]
  have ht4fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t4.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg i = s.reg i
    rw [reg_with_rip, ht3fr i]; exact ht2fr i hi1 hi2 hi6
  have ht4r6 : t4.reg 6 = some c := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg 6 = some c
    rw [reg_with_rip, ht3fr 6]
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht4r3 : t4.reg 3 = some B := by
    rw [ht4fr 3 (by decide) (by decide) (by decide)]; exact hr3
  have ht4r11 : t4.reg 11 = some S := by
    rw [ht4fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht4r10 : t4.reg 10 = some l := by
    rw [ht4fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).mem = s.mem
    rw [mem_with_rip]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).regs.length = 16
    rw [len_with_rip]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x3c3: mov rdx, rsi
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 2 (some c) with rip := t4.rip + 3 } :=
    step_mov (by rw [ht4rip]; exact hc.p3c3) ht4r6
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 2 (some c) with rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x3c6 := by
    rw [ht5]; show t4.rip + 3 = base + 0x3c6; rw [ht4rip]
  have ht5r2 : t5.reg 2 = some c := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide)]
  have ht5fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t5.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]
    exact ht4fr i hi1 hi2 hi6
  have ht5r3 : t5.reg 3 = some B := by
    rw [ht5fr 3 (by decide) (by decide) (by decide)]; exact hr3
  have ht5r11 : t5.reg 11 = some S := by
    rw [ht5fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht5r10 : t5.reg 10 = some l := by
    rw [ht5fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht4r6
  have ht5r1 : t5.reg 1 = some (n + 1) := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 1 =
      some (n + 1)
    rw [reg_with_rip, reg_set_other (by decide)]
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg 1 = some (n + 1)
    rw [reg_with_rip, ht3fr 1]
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide), ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 1 =
      some (n + 1)
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr1
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  -- 0x3c6: sub rdx, rbx
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 2 (some (aluF .sub c B).1) with
        fl := (aluF .sub c B).2, rip := t5.rip + 3 } :=
    step_alu_sub (by rw [ht5rip]; exact hc.p3c6) ht5r2 ht5r3
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 2 (some (aluF .sub c B).1) with
        fl := (aluF .sub c B).2, rip := t5.rip + 3 } := ⟨_, rfl⟩
  have hsub2 : (aluF .sub c B).1 = c - B := by
    rw [aluF_sub_fst, show c + W - B = (c - B) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht6rip : t6.rip = base + 0x3c9 := by
    rw [ht6]; show t5.rip + 3 = base + 0x3c9; rw [ht5rip]
  have ht6r2 : t6.reg 2 = some (c - B) := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 2 = some (c - B)
    rw [reg_set_self (by rw [ht5len]; decide), hsub2]
  have ht6fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t6.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht5fr i hi1 hi2 hi6
  have ht6r11 : t6.reg 11 = some S := by
    rw [ht6fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht6r6 : t6.reg 6 = some c := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht5r6
  have ht6r1 : t6.reg 1 = some (n + 1) := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht5r1
  have ht6r10 : t6.reg 10 = some l := by
    rw [ht6fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).mem = s.mem
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).regs.length = 16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by rw [ht6]; exact ht5k
  -- 0x3c9: add rdx, r11
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 2 (some (aluF .add (c - B) S).1) with
        fl := (aluF .add (c - B) S).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p3c9) ht6r2 ht6r11
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 2 (some (aluF .add (c - B) S).1) with
        fl := (aluF .add (c - B) S).2, rip := t6.rip + 3 } := ⟨_, rfl⟩
  have hadd : (aluF .add (c - B) S).1 = S + (c - B) := by
    rw [aluF_add_fst, show c - B + S = S + (c - B) by omega,
      Nat.mod_eq_of_lt hSW]
  have ht7rip : t7.rip = base + 0x3cc := by
    rw [ht7]; show t6.rip + 3 = base + 0x3cc; rw [ht6rip]
  have ht7r2 : t7.reg 2 = some (S + (c - B)) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 2 =
      some (S + (c - B))
    rw [reg_set_self (by rw [ht6len]; decide), hadd]
  have ht7fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t7.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht6fr i hi1 hi2 hi6
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht6r6
  have ht7r1 : t7.reg 1 = some (n + 1) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht6r1
  have ht7r10 : t7.reg 10 = some l := by
    rw [ht7fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).mem = s.mem
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).regs.length = 16
    rw [len_set]; exact ht6len
  have ht7k : t7.k = s.k := by rw [ht7]; exact ht6k
  -- 0x3cc: st8 [rdx], r10b
  obtain ⟨m8, hm8⟩ : ∃ m, s.mem.wr (S + (c - B)) l = some m :=
    Option.isSome_iff_exists.mp hwr
  have hm8' : t7.mem.wr (ea (S + (c - B)) 0) (l % 256) = some m8 := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hSW, ht7mem,
      Nat.mod_eq_of_lt hl256]
    exact hm8
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7 with mem := m8, rip := t7.rip + 3 } :=
    step_st8 (by rw [ht7rip]; exact hc.p3cc) ht7r2 ht7r10 hm8'
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7 with mem := m8, rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x3cf := by
    rw [ht8]; show t7.rip + 3 = base + 0x3cf; rw [ht7rip]
  have ht8fr : ∀ i, t8.reg i = t7.reg i := by
    intro i
    rw [ht8]
    show t7.reg i = t7.reg i
    exact rfl
  have ht8mem : t8.mem = m8 := by rw [ht8]
  have ht8len : t8.regs.length = 16 := by rw [ht8]; exact ht7len
  have ht8k : t8.k = s.k := by rw [ht8]; exact ht7k
  -- 0x3cf: add rsi, r10
  have ht8r6 : t8.reg 6 = some c := by rw [ht8fr 6]; exact ht7r6
  have ht8r10 : t8.reg 10 = some l := by rw [ht8fr 10]; exact ht7r10
  have ht8r1 : t8.reg 1 = some (n + 1) := by rw [ht8fr 1]; exact ht7r1
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 6 (some (aluF .add c l).1) with
        fl := (aluF .add c l).2, rip := t8.rip + 3 } :=
    step_alu_add (by rw [ht8rip]; exact hc.p3cf) ht8r6 ht8r10
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 6 (some (aluF .add c l).1) with
        fl := (aluF .add c l).2, rip := t8.rip + 3 } := ⟨_, rfl⟩
  have hadd2 : (aluF .add c l).1 = c + l := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have ht9rip : t9.rip = base + 0x3d2 := by
    rw [ht9]; show t8.rip + 3 = base + 0x3d2; rw [ht8rip]
  have ht9r6 : t9.reg 6 = some (c + l) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg 6 = some (c + l)
    rw [reg_set_self (by rw [ht8len]; decide), hadd2]
  have ht9r1 : t9.reg 1 = some (n + 1) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht8r1
  have ht9fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg i = s.reg i
    rw [reg_set_other hi6, ht8fr i]; exact ht7fr i hi1 hi2 hi6
  have ht9mem : t9.mem = m8 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).mem = m8
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).regs.length = 16
    rw [len_set]; exact ht8len
  have ht9k : t9.k = s.k := by rw [ht9]; exact ht8k
  -- 0x3d2: sub rcx, 1
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1) with
        fl := (aluF .sub (n + 1) (1 % W)).2, rip := t9.rip + 4 } :=
    step_aluI_sub (by rw [ht9rip]; exact hc.p3d2) ht9r1
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1) with
        fl := (aluF .sub (n + 1) (1 % W)).2, rip := t9.rip + 4 } :=
    ⟨_, rfl⟩
  have hsub3 : (aluF .sub (n + 1) (1 % W)).1 = n := by
    rw [h1W, aluF_sub_fst, show n + 1 + W - 1 = n + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt hnW]
  have ht10rip : t10.rip = base + 0x3d6 := by
    rw [ht10]; show t9.rip + 4 = base + 0x3d6; rw [ht9rip]
  have ht10r1 : t10.reg 1 = some n := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg 1 = some n
    rw [reg_set_self (by rw [ht9len]; decide), hsub3]
  have ht10r6 : t10.reg 6 = some (c + l) := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg 6 =
      some (c + l)
    rw [reg_set_other (by decide)]; exact ht9r6
  have ht10fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t10.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi1]; exact ht9fr i hi1 hi2 hi6
  have ht10mem : t10.mem = m8 := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).mem = m8
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).regs.length = 16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by rw [ht10]; exact ht9k
  -- 0x3d6: jmp loop head
  obtain ⟨t11, hrun11, ht11rip, ht11fr, ht11mem, ht11len, ht11k⟩ :=
    jmp_step hc.p3d6 ht10rip
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10,
      hrun11]
  have ht11mem' : t11.mem = m8 := by rw [ht11mem]; exact ht10mem
  refine ⟨t11, hrun, ht11rip, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht11fr 6]; exact ht10r6
  · rw [ht11fr 1]; exact ht10r1
  · rw [ht11fr 10, ht10fr 10 (by decide) (by decide) (by decide)]
    exact hr10
  · rw [ht11mem']
    have h := rd_wr_same hm8
    rwa [Nat.mod_eq_of_lt hl256] at h
  · intro i hi1 hi2 hi6
    rw [ht11fr i]; exact ht10fr i hi1 hi2 hi6
  · rw [ht11len]; exact ht10len
  · rw [ht11k]; exact ht10k
  · rw [ht11mem']; exact wr_wshape hm8
  · intro x k hd
    rw [ht11mem']
    exact rdN_wr_other hm8 hd

/-- Every instruction encoding starts with an opcode byte whose
dispatch length is the encoding length. -/
theorem enc_head (x : Instr) :
    ∃ op rest, x.enc = op :: rest ∧ OpLen op x.enc.length := by
  cases x with
  | prim p =>
    exact ⟨p.code, [], rfl, Or.inl ⟨Prim_code_lt p, rfl⟩⟩
  | lit n =>
    exact ⟨0x80, leBytes n 8, rfl,
      Or.inr (Or.inl ⟨rfl, by simp [Instr.enc, leBytes]⟩)⟩
  | str i =>
    exact ⟨0x81, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | pick n =>
    exact ⟨0x82, leBytes n 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | tor =>
    exact ⟨0x83, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inl rfl, rfl⟩))⟩
  | fromr =>
    exact ⟨0x84, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inl rfl), rfl⟩))⟩
  | rfetch =>
    exact ⟨0x85, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), rfl⟩))⟩
  | locget i =>
    exact ⟨0x86, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | locset i =>
    exact ⟨0x87, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jmp t =>
    exact ⟨0x88, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jz t =>
    exact ⟨0x89, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jnz t =>
    exact ⟨0x8a, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | call w =>
    exact ⟨0x8b, leBytes w 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | ret =>
    exact ⟨0x8c, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inr (Or.inr rfl)), rfl⟩))⟩

/-- Code-loop head with instructions remaining: enter the body. -/
theorem parser_code_head_pos {n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some (n + 1))
    (hnW : n + 1 < W) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x315 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and (n + 1) (n + 1)).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p30b) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and (n + 1) (n + 1)).2, rip := s.rip + 3 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x30e := by
    rw [ht1]; show s.rip + 3 = base + 0x30e; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some true := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and (n + 1) (n + 1)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [Nat.mod_eq_of_lt hnW]
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x315 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p30e) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x315 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
  · intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x315 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).k = s.k
    rw [ht1]

/-- Code-loop head with no instructions remaining: jump to the
word tail. -/
theorem parser_code_head_zero {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some 0) :
    ∃ s' : X86Int.St,
      run stage1Im 3 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3db ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p30b) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x30e := by
    rw [ht1]; show s.rip + 3 = base + 0x30e; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some false := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p30e) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x310 := by
    rw [ht2]; show t1.rip + 2 = base + 0x310; rw [ht1rip]
  obtain ⟨t3, hrun3, ht3rip, ht3fr, ht3mem, ht3len, ht3k⟩ :=
    jmp_step hc.p310 ht2rip
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
    rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, hrun3]
  refine ⟨t3, hrun, ht3rip, ?_, ?_, ?_, ?_⟩
  · intro i
    rw [ht3fr i, ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht3mem, ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht3len, ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht3k, ht2]
    show ({ t1 with rip := t1.rip + 2 }).k = s.k
    rw [ht1]

/-- The length-scratch layout the code loop writes: at
`S + (pos - B)`, the byte length of the instruction at `pos`. -/
def LenLay (m : Mem) (B S c : Nat) : List Instr → Prop
  | [] => True
  | x :: xs => m.rd (S + (c - B)) = some x.enc.length ∧
      LenLay m B S (c + x.enc.length) xs

/-- A single-byte write is a one-byte `wrN`. -/
theorem wr_eq_wrN (m : Mem) (a v : Nat) : m.wr a v = m.wrN a [v] := by
  cases h : m.wr a v <;> simp [Mem.wrN, h]

/-- `rd` transport along an `rdN _ 1` equality. -/
theorem rd_of_rdN1 {m m' : Mem} {a v : Nat}
    (h : m'.rdN a 1 = m.rdN a 1) (hm : m.rd a = some v) :
    m'.rd a = some v := by
  have h1 : m.rdN a 1 = some [v] := by simp [Mem.rdN, hm]
  have h2 : m'.rdN a (0 + 1) = some (v :: []) := by
    show m'.rdN a 1 = some [v]
    rw [h]; exact h1
  exact (rdN_cons h2).1

/-- The instruction count never exceeds the byte count. -/
theorem code_length_le_bytes : ∀ (code : List Instr),
    code.length ≤ codeBytes code
  | [] => by decide
  | x :: xs => by
    have ih := code_length_le_bytes xs
    obtain ⟨op, rest, henc, _⟩ := enc_head x
    have hpos : 0 < x.enc.length := by
      rw [henc]; exact Nat.zero_lt_succ _
    show (x :: xs).length ≤ codeBytes (x :: xs)
    simp only [List.length_cons, codeBytes]
    omega

/-- One iteration of the code loop over the head instruction. -/
theorem parser_code_step {c e B S n : Nat} {x : Instr} {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some (n + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (henc : s.mem.rdN c x.enc.length = some x.enc)
    (hce : c + x.enc.length ≤ e) (hcW : c < W) (heW : e < W)
    (hnW : n + 1 < W) (hBc : B ≤ c) (hSW : S + (c - B) < W)
    (hwr : (s.mem.wr (S + (c - B)) x.enc.length).isSome)
    (hlen16 : s.regs.length = 16) :
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 6 = some (c + x.enc.length) ∧
      s'.reg 1 = some n ∧
      s'.mem.rd (S + (c - B)) = some x.enc.length ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c - B) + 1 ≤ y ∨ y + k ≤ S + (c - B) →
        s'.mem.rdN y k = s.mem.rdN y k) := by
  obtain ⟨op, rest, henc_eq, hOp⟩ := enc_head x
  have hlenx : x.enc.length = rest.length + 1 := by
    rw [henc_eq]; simp
  have hlen256 : x.enc.length < 256 := by
    rcases hOp with ⟨_, hl⟩ | ⟨_, hl⟩ | ⟨_, hl⟩ | ⟨_, hl⟩ <;>
      omega
  have hrd : s.mem.rd c = some op := by
    have h : s.mem.rdN c (rest.length + 1) = some (op :: rest) := by
      rw [← hlenx, ← henc_eq]; exact henc
    exact (rdN_cons h).1
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    parser_code_head_pos hrip hr1 hnW
  have hs1r6 : s1.reg 6 = some c := by rw [hs1fr 6]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by rw [hs1fr 15]; exact hr15
  have hs1len16 : s1.regs.length = 16 := by rw [hs1len]; exact hlen16
  have hs1rd : s1.mem.rd c = some op := by rw [hs1mem]; exact hrd
  obtain ⟨s2, hrun2, hs2rip, hs2r0, hs2r6, hs2fr, hs2mem, hs2len,
      hs2k⟩ :=
    parser_code_bounds hs1rip hs1r6 hs1r15 (by omega) hcW heW hs1rd
      hs1len16
  have hs2len16 : s2.regs.length = 16 := hs2len
  obtain ⟨nd, s3, hrun3, hs3rip, hs3r10, hs3r0, hs3fr, hs3mem, hs3len,
      hs3k⟩ :=
    parser_dispatch hs2rip hs2r0 hOp hs2len16
  have hs3mem' : s3.mem = s.mem := by rw [hs3mem, hs2mem, hs1mem]
  have hs3r6 : s3.reg 6 = some c := by
    rw [hs3fr 6 (by decide)]; exact hs2r6
  have hs3r15 : s3.reg 15 = some e := by
    rw [hs3fr 15 (by decide), hs2fr 15 (by decide) (by decide),
      hs1fr 15]
    exact hr15
  have hs3r3 : s3.reg 3 = some B := by
    rw [hs3fr 3 (by decide), hs2fr 3 (by decide) (by decide), hs1fr 3]
    exact hr3
  have hs3r11 : s3.reg 11 = some S := by
    rw [hs3fr 11 (by decide), hs2fr 11 (by decide) (by decide),
      hs1fr 11]
    exact hr11
  have hs3r1 : s3.reg 1 = some (n + 1) := by
    rw [hs3fr 1 (by decide), hs2fr 1 (by decide) (by decide), hs1fr 1]
    exact hr1
  have hs3wr : (s3.mem.wr (S + (c - B)) x.enc.length).isSome := by
    rw [hs3mem']; exact hwr
  obtain ⟨s4, hrun4, hs4rip, hs4r6, hs4r1, hs4r10, hs4rd, hs4fr,
      hs4len, hs4k, hs4sh, hs4memfr⟩ :=
    parser_code_tail hs3rip hs3r6 hs3r15 hs3r10 hs3r3 hs3r11 hs3r1
      hce hcW heW hBc hSW hlen256 (by omega) hs3wr hs3len
  have hrun : run stage1Im (2 + 6 + nd + 11) (Out.ok s) = Out.ok s4 := by
    rw [show 2 + 6 + nd + 11 = 2 + (6 + (nd + 11)) by omega,
      run_add stage1Im 2 _, hrun1, run_add stage1Im 6 _, hrun2,
      run_add stage1Im nd 11, hrun3, hrun4]
  refine ⟨2 + 6 + nd + 11, s4, hrun, hs4rip, hs4r6, hs4r1, hs4rd, ?_,
    hs4len, ?_, ?_, ?_⟩
  · intro i hi0 hi1 hi2 hi6 hi10
    rw [hs4fr i hi1 hi2 hi6, hs3fr i hi10,
      hs2fr i hi0 hi2, hs1fr i]
  · rw [hs4k, hs3k, hs2k, hs1k]
  · rwa [hs3mem'] at hs4sh
  · intro y k hd
    rw [hs4memfr y k hd, hs3mem']

/-- The whole code loop over one word's instruction list: from
the loop head back to the loop head with count zero, the cursor
past the code bytes, `CodeSeq` preserved and `LenLay` written. -/
theorem parser_code_loop : ∀ (code : List Instr) {c e B S : Nat}
    {s : X86Int.St},
    s.rip = base + 0x30b → s.reg 1 = some code.length →
    s.reg 6 = some c → s.reg 15 = some e →
    s.reg 3 = some B → s.reg 11 = some S →
    CodeSeq s.mem c code →
    c + codeBytes code ≤ e → c < W → e < W →
    B ≤ c → S + (c + codeBytes code - B) < W →
    (S + (c + codeBytes code - B) ≤ c ∨ c + codeBytes code ≤ S + (c - B)) →
    (∀ a v, S + (c - B) ≤ a → a < S + (c + codeBytes code - B) →
      (s.mem.wr a v).isSome) →
    s.regs.length = 16 →
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + codeBytes code) ∧
      CodeSeq s'.mem c code ∧ LenLay s'.mem B S c code ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c + codeBytes code - B) ≤ y ∨
        y + k ≤ S + (c - B) → s'.mem.rdN y k = s.mem.rdN y k) := by
  intro code
  induction code with
  | nil =>
    intro c e B S s hrip hr1 hr6 _ _ _ _ _ hcW _ _ _ _ _ hlen16
    refine ⟨0, s, rfl, hrip, hr1, ?_, trivial, trivial, ?_, hlen16,
      rfl, rfl, ?_⟩
    · show s.reg 6 = some (c + codeBytes [])
      rw [show codeBytes [] = 0 from rfl]
      exact hr6
    · intro i _ _ _ _ _; exact rfl
    · intro y k _; exact rfl
  | cons x xs ih =>
    intro c e B S s hrip hr1 hr6 hr15 hr3 hr11 hseq hce hcW heW hBc
      hSW hdisj hwr hlen16
    obtain ⟨henc, hseqx⟩ := hseq
    have hcb : codeBytes (x :: xs) = x.enc.length + codeBytes xs := rfl
    have hpos : 0 < x.enc.length := by
      obtain ⟨op, rest, heq, _⟩ := enc_head x
      rw [heq]; exact Nat.zero_lt_succ _
    have hlenW : (x :: xs).length < W := by
      have h1 := code_length_le_bytes (x :: xs)
      omega
    have hr1' : s.reg 1 = some (xs.length + 1) := hr1
    have hwr0 : (s.mem.wr (S + (c - B)) x.enc.length).isSome := by
      apply hwr
      · omega
      · have hpos : 0 < x.enc.length := by
          obtain ⟨op, rest, heq, _⟩ := enc_head x
          rw [heq]; exact Nat.zero_lt_succ _
        omega
    have hSW0 : S + (c - B) < W := by omega
    obtain ⟨k1, s1, hrun1, hs1rip, hs1r6, hs1r1, hs1rd, hs1fr, hs1len,
        hs1k, hs1sh, hs1memfr⟩ :=
      parser_code_step hrip hr1' hr6 hr15 hr3 hr11 henc (by omega)
        hcW heW hlenW hBc hSW0 hwr0 hlen16
    have hs1r15 : s1.reg 15 = some e := by
      rw [hs1fr 15 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr15
    have hs1r3 : s1.reg 3 = some B := by
      rw [hs1fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr3
    have hs1r11 : s1.reg 11 = some S := by
      rw [hs1fr 11 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr11
    have hs1seq : CodeSeq s1.mem (c + x.enc.length) xs := by
      apply CodeSeq_of_frame hseqx
      intro y k hy hk
      apply hs1memfr
      omega
    have hs1wr : ∀ a v, S + (c + x.enc.length - B) ≤ a →
        a < S + (c + x.enc.length + codeBytes xs - B) →
        (s1.mem.wr a v).isSome := by
      intro a v ha1 ha2
      have h1 : (s.mem.wrN a [v]).isSome := by
        rw [← wr_eq_wrN]
        apply hwr <;> omega
      have h2 := wrN_some_of_wshape _ s.mem s1.mem hs1sh _ h1
      rwa [← wr_eq_wrN] at h2
    obtain ⟨k2, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2seq, hs2lay, hs2fr,
        hs2len, hs2k, hs2sh, hs2memfr⟩ :=
      ih hs1rip hs1r1 hs1r6 hs1r15 hs1r3 hs1r11 hs1seq (by omega)
        (by omega) heW (by omega) (by omega) (by omega) hs1wr hs1len
    have hrun : run stage1Im (k1 + k2) (Out.ok s) = Out.ok s2 := by
      rw [run_add stage1Im k1 k2, hrun1, hrun2]
    have hseq' : CodeSeq s2.mem c (x :: xs) := by
      constructor
      · have h1 : s2.mem.rdN c x.enc.length =
            s1.mem.rdN c x.enc.length := by
          apply hs2memfr
          omega
        have h2 : s1.mem.rdN c x.enc.length =
            s.mem.rdN c x.enc.length := by
          apply hs1memfr
          omega
        rw [h1, h2]; exact henc
      · exact hs2seq
    have hlay' : LenLay s2.mem B S c (x :: xs) := by
      constructor
      · have h1 : s2.mem.rdN (S + (c - B)) 1 =
            s1.mem.rdN (S + (c - B)) 1 := by
          apply hs2memfr
          omega
        exact rd_of_rdN1 h1 hs1rd
      · exact hs2lay
    refine ⟨k1 + k2, s2, hrun, hs2rip, hs2r1, ?_, hseq', hlay', ?_,
      hs2len, ?_, ?_, ?_⟩
    · show s2.reg 6 = some (c + codeBytes (x :: xs))
      rw [hcb]
      have h := hs2r6
      rwa [show c + x.enc.length + codeBytes xs =
        c + (x.enc.length + codeBytes xs) by omega] at h
    · intro i hi0 hi1 hi2 hi6 hi10
      rw [hs2fr i hi0 hi1 hi2 hi6 hi10]
      exact hs1fr i hi0 hi1 hi2 hi6 hi10
    · rw [hs2k, hs1k]
    · rw [hs2sh, hs1sh]
    · intro y k hd
      have h1 : s2.mem.rdN y k = s1.mem.rdN y k := by
        apply hs2memfr
        omega
      have h2 : s1.mem.rdN y k = s.mem.rdN y k := by
        apply hs1memfr
        omega
      rw [h1, h2]

/-- Code-loop entry: push the outer word counter and load this
word's instruction count from slot 24. -/
theorem parser_code_enter {sp p m nc : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x306) (hr1 : s.reg 1 = some m)
    (hr7 : s.reg 7 = some p) (hr4 : s.reg 4 = some sp)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hpW : p + 24 < W)
    (hslot : rd64 s.mem (p + 24) = some nc)
    (hdisj : p + 24 + 8 ≤ sp - 8 ∨ sp ≤ p + 24)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 1 = some nc ∧
      s'.reg 4 = some (sp - 8) ∧
      rd64 s'.mem (sp - 8) = some (m % W) ∧
      rd64 s'.mem (p + 24) = some nc ∧
      (∀ i, i ≠ 1 → i ≠ 4 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨m1, hm1raw⟩ : ∃ mm, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) m =
      some mm :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) m)
  have hm1 : wr64 s.mem ((sp + W - 8) % W) m = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) m = some m1 := by
    rwa [hsp8mod] at hm1
  -- 0x306: push rcx
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with
        mem := m1, rip := s.rip + 1 } :=
    step_push (by rw [hrip]; exact hc.p306) hr1 hr4 hm1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with
        mem := m1, rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x307 := by
    rw [ht1]; show s.rip + 1 = base + 0x307; rw [hrip]
  have ht1r4 : t1.reg 4 = some (sp - 8) := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (hlen 4 (by decide)), hsp8mod]
  have ht1r7 : t1.reg 7 = some p := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 7 = some p
    rw [reg_set_other (by decide)]; exact hr7
  have ht1fr : ∀ i, i ≠ 4 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = m1 := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have ht1slot : rd64 t1.mem (p + 24) = some nc := by
    rw [ht1mem]
    have h := rd64_wr64_other hm1wr (m := s.mem) (x := p + 24) (by
      rcases hdisj with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
    rw [h]; exact hslot
  -- 0x307: ld rcx, [rdi+24]
  have hld : rd64 t1.mem (ea p 24) = some nc := by
    unfold ea
    rwa [Nat.mod_eq_of_lt hpW]
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 1 (some nc) with rip := t1.rip + 4 } :=
    step_ld (by rw [ht1rip]; exact hc.p307) ht1r7 hld
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 1 (some nc) with rip := t1.rip + 4 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  have ht2mem : t2.mem = m1 := by
    rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht1mem
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]; show t1.rip + 4 = base + 0x30b; rw [ht1rip]
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg 1 = some nc
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide)]
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht1r4
  · rw [ht2mem]; exact rd64_wr64 hm1wr
  · rw [ht2mem]
    have h := rd64_wr64_other hm1wr (m := s.mem) (x := p + 24) (by
      rcases hdisj with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
    rw [h]; exact hslot
  · intro i hi1 hi4
    rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi1]; exact ht1fr i hi4
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  · rw [ht2]; exact ht1k
  · rw [ht2mem]; exact wrN_wshape _ _ _ hm1wr
  · intro x k hd
    rw [ht2mem]
    apply rdN_wr64_other hm1wr
    rcases hd with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)

set_option maxHeartbeats 0 in
/-- Word tail after the code loop: pop the outer counter, store
the ann pointer at slot 48, skip `12 * nann` annotation bytes,
advance the entry pointer and decrement the counter, back to
the word-loop head. -/
theorem parser_word_tail {sp1 p c e r na : Nat} {m48 : Mem}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x3db) (hr4 : s.reg 4 = some sp1)
    (hr6 : s.reg 6 = some c) (hr7 : s.reg 7 = some p)
    (hr15 : s.reg 15 = some e)
    (hpop : rd64 s.mem sp1 = some (r + 1))
    (hspW : sp1 + 8 < W) (hrW : r < W)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hnaW : 12 * na < W) (hce : c + 12 * na ≤ e)
    (hslot32 : rd64 s.mem (p + 32) = some na)
    (hm48 : wr64 s.mem (p + 48) c = some m48)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 18 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 6 = some (c + 12 * na) ∧
      s'.reg 7 = some (p + 64) ∧ s'.reg 1 = some r ∧
      s'.reg 4 = some (sp1 + 8) ∧
      rd64 s'.mem (p + 48) = some (c % W) ∧
      rd64 s'.mem (p + 32) = some na ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, p + 48 + 8 ≤ x ∨ x + k ≤ p + 48 →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have h64W : 64 % W = 64 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x3db: pop rcx
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.p3db) hr4 hpop
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x3dc := by
    rw [ht1]; show s.rip + 1 = base + 0x3dc; rw [hrip]
  have ht1r1 : t1.reg 1 = some (r + 1) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 1 = some (r + 1)
    rw [reg_with_rip, reg_set_self (by
      show 1 < (s.set 4 (some ((sp1 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ht1r4 : t1.reg 4 = some (sp1 + 8) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 4 = some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_self (hlen 4 (by decide)), Nat.mod_eq_of_lt hspW]
  have ht1fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t1.reg i = s.reg i := by
    intro i _ hi1 _ hi4 _ _
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi1, reg_set_other hi4]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 1),
      reg_set_other (by decide : (6 : Nat) ≠ 4)]
    exact hr6
  have ht1r7 : t1.reg 7 = some p := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 7 = some p
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 1),
      reg_set_other (by decide : (7 : Nat) ≠ 4)]
    exact hr7
  have ht1r15 : t1.reg 15 = some e := by
    rw [ht1fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hr15
  have ht1mem : t1.mem = s.mem := by rw [ht1]; rfl
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  -- 0x3dc: st [rdi+48], rsi
  have hm48' : wr64 t1.mem (p + 48) c = some m48 := by
    rw [ht1mem]; exact hm48
  obtain ⟨t2, hrun2, ht2rip, ht2mem, ht2slot, ht2fr, ht2len, ht2k,
      ht2sh, ht2memfr⟩ :=
    store_slot (site := base + 0x3dc) (off := 48) (slen := 4) (r := 6)
      hc.p3dc ht1rip ht1r7 ht1r6 (by omega) hm48'
  have ht2rip' : t2.rip = base + 0x3e0 := ht2rip
  have ht2len16 : t2.regs.length = 16 := by
    rw [ht2len]; exact ht1len
  have ht2slot32 : rd64 t2.mem (p + 32) = some na := by
    rw [ht2mem]
    have h := rd64_wr64_other hm48 (m := s.mem) (x := p + 32)
      (Or.inr (by omega))
    rw [h]; exact hslot32
  -- 0x3e0: ld rax, [rdi+32]
  have ht2r7 : t2.reg 7 = some p := by rw [ht2fr 7]; exact ht1r7
  have hld : rd64 t2.mem (ea p 32) = some na := by
    unfold ea
    rwa [Nat.mod_eq_of_lt (by omega : p + 32 < W)]
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some na) with rip := t2.rip + 4 } :=
    step_ld (by rw [ht2rip']; exact hc.p3e0) ht2r7 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some na) with rip := t2.rip + 4 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3e4 := by
    rw [ht3]; show t2.rip + 4 = base + 0x3e4; rw [ht2rip']
  have ht3r0 : t3.reg 0 = some na := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 0 = some na
    rw [reg_with_rip, reg_set_self (by rw [ht2len16]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t3.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0, ht2fr i]
    exact ht1fr i hi0 hi1 hi2 hi4 hi6 hi7
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 6]
    exact ht1r6
  have ht3r7 : t3.reg 7 = some p := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 7 = some p
    rw [reg_with_rip, reg_set_other (by decide)]
    exact ht2r7
  have ht3r1 : t3.reg 1 = some (r + 1) := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 1 =
      some (r + 1)
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 1]
    exact ht1r1
  have ht3r4 : t3.reg 4 = some (sp1 + 8) := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 4]
    exact ht1r4
  have ht3r15 : t3.reg 15 = some e := by
    rw [ht3fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hr15
  have ht3mem : t3.mem = m48 := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len16
  have ht3k : t3.k = s.k := by
    rw [ht3]
    show t2.k = s.k
    rw [ht2k]; exact ht1k
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  -- 0x3e4: mov rdx, rax
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some na) with rip := t3.rip + 3 } :=
    step_mov (by rw [ht3rip]; exact hc.p3e4) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some na) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x3e7 := by
    rw [ht4]; show t3.rip + 3 = base + 0x3e7; rw [ht3rip]
  have ht4r2 : t4.reg 2 = some na := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg 2 = some na
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r0 : t4.reg 0 = some na := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg 0 = some na
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht3r0
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → t4.reg i = t3.reg i := by
    intro i _ hi2
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg i =
      t3.reg i
    rw [reg_with_rip, reg_set_other hi2]
  have ht4mem : t4.mem = m48 := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  -- 0x3e7: add rax, rax (2*na)
  have hdbl : ∀ v : Nat, 2 * v < W → (aluF .add v v).1 = 2 * v := by
    intro v hv
    rw [aluF_add_fst, show v + v = 2 * v by omega,
      Nat.mod_eq_of_lt hv]
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 0 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t4.rip + 3 } :=
    step_alu_add (by rw [ht4rip]; exact hc.p3e7) ht4r0 ht4r0
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 0 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x3ea := by
    rw [ht5]; show t4.rip + 3 = base + 0x3ea; rw [ht4rip]
  have ht5r0 : t5.reg 0 = some (2 * na) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg 0 = some (2 * na)
    rw [reg_set_self (by rw [ht4len]; decide),
      hdbl na (by omega)]
  have ht5r2 : t5.reg 2 = some na := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg 2 = some na
    rw [reg_set_other (by decide)]; exact ht4r2
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → t5.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg i = t3.reg i
    rw [reg_set_other hi0]; exact ht4fr i hi0 hi2
  have ht5mem : t5.mem = m48 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).mem = m48
    rw [mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).regs.length = 16
    rw [len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  -- 0x3ea: add rax, rax (4*na)
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t5.rip + 3 } :=
    step_alu_add (by rw [ht5rip]; exact hc.p3ea) ht5r0 ht5r0
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x3ed := by
    rw [ht6]; show t5.rip + 3 = base + 0x3ed; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (4 * na) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_self (by rw [ht5len]; decide),
      hdbl (2 * na) (by omega)]
    congr 1
    omega
  have ht6r2 : t6.reg 2 = some na := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg 2 =
      some na
    rw [reg_set_other (by decide)]; exact ht5r2
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → t6.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi0]; exact ht5fr i hi0 hi2
  have ht6mem : t6.mem = m48 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).mem = m48
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by rw [ht6]; exact ht5k
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  -- 0x3ed: add rdx, rdx (2*na)
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 2 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p3ed) ht6r2 ht6r2
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 2 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t6.rip + 3 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x3f0 := by
    rw [ht7]; show t6.rip + 3 = base + 0x3f0; rw [ht6rip]
  have ht7r2 : t7.reg 2 = some (2 * na) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg 2 = some (2 * na)
    rw [reg_set_self (by rw [ht6len]; decide),
      hdbl na (by omega)]
  have ht7r0 : t7.reg 0 = some (4 * na) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg 0 = some (4 * na)
    rw [reg_set_other (by decide)]; exact ht6r0
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg i = t3.reg i
    rw [reg_set_other hi2]; exact ht6fr i hi0 hi2
  have ht7mem : t7.mem = m48 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).mem = m48
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).regs.length = 16
    rw [len_set]; exact ht6len
  have ht7k : t7.k = s.k := by rw [ht7]; exact ht6k
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  -- 0x3f0: add rdx, rdx (4*na)
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 2 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t7.rip + 3 } :=
    step_alu_add (by rw [ht7rip]; exact hc.p3f0) ht7r2 ht7r2
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 2 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t7.rip + 3 } :=
    ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x3f3 := by
    rw [ht8]; show t7.rip + 3 = base + 0x3f3; rw [ht7rip]
  have ht8r2 : t8.reg 2 = some (4 * na) := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg 2 =
      some (4 * na)
    rw [reg_set_self (by rw [ht7len]; decide),
      hdbl (2 * na) (by omega)]
    congr 1
    omega
  have ht8r0 : t8.reg 0 = some (4 * na) := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_other (by decide)]; exact ht7r0
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi2]; exact ht7fr i hi0 hi2
  have ht8mem : t8.mem = m48 := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).mem = m48
    rw [mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by rw [ht8]; exact ht7k
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  -- 0x3f3: add rdx, rdx (8*na)
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 2 (some (aluF .add (4 * na) (4 * na)).1) with
        fl := (aluF .add (4 * na) (4 * na)).2, rip := t8.rip + 3 } :=
    step_alu_add (by rw [ht8rip]; exact hc.p3f3) ht8r2 ht8r2
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 2 (some (aluF .add (4 * na) (4 * na)).1) with
        fl := (aluF .add (4 * na) (4 * na)).2, rip := t8.rip + 3 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x3f6 := by
    rw [ht9]; show t8.rip + 3 = base + 0x3f6; rw [ht8rip]
  have ht9r2 : t9.reg 2 = some (8 * na) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg 2 =
      some (8 * na)
    rw [reg_set_self (by rw [ht8len]; decide),
      hdbl (4 * na) (by omega)]
    congr 1
    omega
  have ht9r0 : t9.reg 0 = some (4 * na) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → t9.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi2]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = m48 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).mem = m48
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht8len
  have ht9k : t9.k = s.k := by rw [ht9]; exact ht8k
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  -- 0x3f6: add rax, rdx (12*na)
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 0 (some (aluF .add (4 * na) (8 * na)).1) with
        fl := (aluF .add (4 * na) (8 * na)).2, rip := t9.rip + 3 } :=
    step_alu_add (by rw [ht9rip]; exact hc.p3f6) ht9r0 ht9r2
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 0 (some (aluF .add (4 * na) (8 * na)).1) with
        fl := (aluF .add (4 * na) (8 * na)).2, rip := t9.rip + 3 } :=
    ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x3f9 := by
    rw [ht10]; show t9.rip + 3 = base + 0x3f9; rw [ht9rip]
  have ht10r0 : t10.reg 0 = some (12 * na) := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).reg 0 =
      some (12 * na)
    rw [reg_set_self (by rw [ht9len]; decide), aluF_add_fst,
      show 4 * na + 8 * na = 12 * na by omega,
      Nat.mod_eq_of_lt hnaW]
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → t10.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi0]; exact ht9fr i hi0 hi2
  have ht10r15 : t10.reg 15 = some e := by
    rw [ht10fr 15 (by decide) (by decide)]; exact ht3r15
  have ht10r6 : t10.reg 6 = some c := by
    rw [ht10fr 6 (by decide) (by decide)]; exact ht3r6
  have ht10mem : t10.mem = m48 := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).mem = m48
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by rw [ht10]; exact ht9k
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  -- 0x3f9: mov rdx, r15
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 2 (some e) with rip := t10.rip + 3 } :=
    step_mov (by rw [ht10rip]; exact hc.p3f9) ht10r15
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 2 (some e) with rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x3fc := by
    rw [ht11]; show t10.rip + 3 = base + 0x3fc; rw [ht10rip]
  have ht11r2 : t11.reg 2 = some e := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht10len]; decide)]
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → t11.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg i =
      t3.reg i
    rw [reg_with_rip, reg_set_other hi2]
    exact ht10fr i hi0 hi2
  have ht11r6 : t11.reg 6 = some c := by
    rw [ht11fr 6 (by decide) (by decide)]; exact ht3r6
  have ht11r0 : t11.reg 0 = some (12 * na) := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg 0 =
      some (12 * na)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r0
  have ht11mem : t11.mem = m48 := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht10len
  have ht11k : t11.k = s.k := by rw [ht11]; exact ht10k
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  -- 0x3fc: sub rdx, rsi
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t11.rip + 3 } :=
    step_alu_sub (by rw [ht11rip]; exact hc.p3fc) ht11r2 ht11r6
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t11.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht12rip : t12.rip = base + 0x3ff := by
    rw [ht12]; show t11.rip + 3 = base + 0x3ff; rw [ht11rip]
  have ht12r2 : t12.reg 2 = some (e - c) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht11len]; decide), hsub]
  have ht12r0 : t12.reg 0 = some (12 * na) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg 0 = some (12 * na)
    rw [reg_set_other (by decide)]; exact ht11r0
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → t12.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg i = t3.reg i
    rw [reg_set_other hi2]; exact ht11fr i hi0 hi2
  have ht12mem : t12.mem = m48 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).mem = m48
    rw [mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht11len
  have ht12k : t12.k = s.k := by rw [ht12]; exact ht11k
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  -- 0x3ff: cmp rdx, rax
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12 with fl := (aluF .sub (e - c) (12 * na)).2, rip := t12.rip + 3 } :=
    step_cmp (by rw [ht12rip]; exact hc.p3ff) ht12r2 ht12r0
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12 with fl := (aluF .sub (e - c) (12 * na)).2, rip := t12.rip + 3 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x402 := by
    rw [ht13]; show t12.rip + 3 = base + 0x402; rw [ht12rip]
  have ht13fl : X86Int.cond t13.fl 3 = some true := by
    have hcf : t13.fl.cf = some false := by
      rw [ht13]
      show (aluF .sub (e - c) (12 * na)).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht13fr : ∀ i, t13.reg i = t12.reg i := by
    intro i
    rw [ht13]
    show t12.reg i = t12.reg i
    exact rfl
  have ht13mem : t13.mem = m48 := by rw [ht13]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by rw [ht13]; exact ht12len
  have ht13k : t13.k = s.k := by rw [ht13]; exact ht12k
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  -- 0x402: jae 0x409
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with rip := base + 0x409 } := by
    have h := step_jcc (im := stage1Im) (s := t13)
      (by rw [ht13rip]; exact hc.p402) ht13fl
    simpa using h
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with rip := base + 0x409 } := ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x409 := by rw [ht14]
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → t14.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).reg i = t3.reg i
    rw [reg_with_rip, ht13fr i]; exact ht12fr i hi0 hi2
  have ht14r6 : t14.reg 6 = some c := by
    rw [ht14fr 6 (by decide) (by decide)]; exact ht3r6
  have ht14r0 : t14.reg 0 = some (12 * na) := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).reg 0 = some (12 * na)
    rw [reg_with_rip, ht13fr 0]; exact ht12r0
  have ht14mem : t14.mem = m48 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).mem = m48
    rw [mem_with_rip]; exact ht13mem
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).regs.length = 16
    rw [len_with_rip]; exact ht13len
  have ht14k : t14.k = s.k := by rw [ht14]; exact ht13k
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  -- 0x409: add rsi, rax
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14.set 6 (some (aluF .add c (12 * na)).1) with
        fl := (aluF .add c (12 * na)).2, rip := t14.rip + 3 } :=
    step_alu_add (by rw [ht14rip]; exact hc.p409) ht14r6 ht14r0
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14.set 6 (some (aluF .add c (12 * na)).1) with
        fl := (aluF .add c (12 * na)).2, rip := t14.rip + 3 } :=
    ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x40c := by
    rw [ht15]; show t14.rip + 3 = base + 0x40c; rw [ht14rip]
  have ht15r6 : t15.reg 6 = some (c + 12 * na) := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_self (by rw [ht14len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t15.reg i = t3.reg i := by
    intro i hi0 hi2 hi6
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).reg i = t3.reg i
    rw [reg_set_other hi6]; exact ht14fr i hi0 hi2
  have ht15r7 : t15.reg 7 = some p := by
    rw [ht15fr 7 (by decide) (by decide) (by decide)]; exact ht3r7
  have ht15r1 : t15.reg 1 = some (r + 1) := by
    rw [ht15fr 1 (by decide) (by decide) (by decide)]; exact ht3r1
  have ht15mem : t15.mem = m48 := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).mem = m48
    rw [mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).regs.length = 16
    rw [len_set]; exact ht14len
  have ht15k : t15.k = s.k := by rw [ht15]; exact ht14k
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  -- 0x40c: add rdi, 64
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15.set 7 (some (aluF .add p (64 % W)).1) with
        fl := (aluF .add p (64 % W)).2, rip := t15.rip + 4 } :=
    step_aluI_add (by rw [ht15rip]; exact hc.p40c) ht15r7
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15.set 7 (some (aluF .add p (64 % W)).1) with
        fl := (aluF .add p (64 % W)).2, rip := t15.rip + 4 } :=
    ⟨_, rfl⟩
  have ht16rip : t16.rip = base + 0x410 := by
    rw [ht16]; show t15.rip + 4 = base + 0x410; rw [ht15rip]
  have ht16r7 : t16.reg 7 = some (p + 64) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 7 =
      some (p + 64)
    rw [reg_set_self (by rw [ht15len]; decide), h64W, aluF_add_fst,
      Nat.mod_eq_of_lt hpW]
  have ht16r6 : t16.reg 6 = some (c + 12 * na) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_other (by decide)]; exact ht15r6
  have ht16r1 : t16.reg 1 = some (r + 1) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 1 =
      some (r + 1)
    rw [reg_set_other (by decide)]; exact ht15r1
  have ht16fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 →
      t16.reg i = t3.reg i := by
    intro i hi0 hi2 hi6 hi7
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg i = t3.reg i
    rw [reg_set_other hi7]; exact ht15fr i hi0 hi2 hi6
  have ht16mem : t16.mem = m48 := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).mem = m48
    rw [mem_set]; exact ht15mem
  have ht16len : t16.regs.length = 16 := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).regs.length = 16
    rw [len_set]; exact ht15len
  have ht16k : t16.k = s.k := by rw [ht16]; exact ht15k
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  -- 0x410: sub rcx, 1
  have e17 : X86Int.step stage1Im t16 =
      .ok { t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1) with
        fl := (aluF .sub (r + 1) (1 % W)).2, rip := t16.rip + 4 } :=
    step_aluI_sub (by rw [ht16rip]; exact hc.p410) ht16r1
  obtain ⟨t17, ht17⟩ : ∃ t : X86Int.St, t =
      { t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1) with
        fl := (aluF .sub (r + 1) (1 % W)).2, rip := t16.rip + 4 } :=
    ⟨_, rfl⟩
  have hsubr : (aluF .sub (r + 1) (1 % W)).1 = r := by
    rw [h1W, aluF_sub_fst, show r + 1 + W - 1 = r + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt hrW]
  have ht17rip : t17.rip = base + 0x414 := by
    rw [ht17]; show t16.rip + 4 = base + 0x414; rw [ht16rip]
  have ht17r1 : t17.reg 1 = some r := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 1 = some r
    rw [reg_set_self (by rw [ht16len]; decide), hsubr]
  have ht17r6 : t17.reg 6 = some (c + 12 * na) := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_other (by decide)]; exact ht16r6
  have ht17r7 : t17.reg 7 = some (p + 64) := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 7 =
      some (p + 64)
    rw [reg_set_other (by decide)]; exact ht16r7
  have ht17fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 7 →
      t17.reg i = t3.reg i := by
    intro i hi0 hi1 hi2 hi6 hi7
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi1]; exact ht16fr i hi0 hi2 hi6 hi7
  have ht17mem : t17.mem = m48 := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).mem = m48
    rw [mem_set]; exact ht16mem
  have ht17len : t17.regs.length = 16 := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).regs.length =
      16
    rw [len_set]; exact ht16len
  have ht17k : t17.k = s.k := by rw [ht17]; exact ht16k
  have r17 : run stage1Im 1 (Out.ok t16) = Out.ok t17 := by
    rw [run_succ_ok stage1Im 0 t16, e17, ← ht17, run_zero]
  -- 0x414: jmp word-loop head
  obtain ⟨t18, hrun18, ht18rip, ht18fr, ht18mem, ht18len, ht18k⟩ :=
    jmp_step hc.p414 ht17rip
  have hrun : run stage1Im 18 (Out.ok s) = Out.ok t18 := by
    rw [show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, r1,
      show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, hrun2,
      show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r3,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r4,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r5,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r6,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r7,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r8,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r9,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r10,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r11,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r12,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r13,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r14,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r15,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r16,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r17,
      hrun18]
  have ht18mem' : t18.mem = m48 := by rw [ht18mem]; exact ht17mem
  refine ⟨t18, hrun, ht18rip, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht18fr 6]; exact ht17r6
  · rw [ht18fr 7]; exact ht17r7
  · rw [ht18fr 1]; exact ht17r1
  · rw [ht18fr 4, ht17fr 4 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact ht3r4
  · rw [ht18mem']
    have h : rd64 m48 (p + 48) = some (c % W) := by
      rw [ht2mem] at ht2slot
      exact ht2slot
    exact h
  · rw [ht18mem']
    have h := rd64_wr64_other hm48 (m := s.mem) (x := p + 32)
      (Or.inr (by omega))
    rw [h]; exact hslot32
  · intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht18fr i, ht17fr i hi0 hi1 hi2 hi6 hi7]
    exact ht3fr i hi0 hi1 hi2 hi4 hi6 hi7
  · rw [ht18len]; exact ht17len
  · rw [ht18k]; exact ht17k
  · rw [ht18mem']; exact wrN_wshape _ _ _ hm48
  · intro x k hd
    rw [ht18mem']
    exact rdN_wr64_other hm48 hd

/-- `rd` equality from an `rdN _ 1` equality. -/
theorem rd_eq_of_rdN1 {m m' : Mem} {a : Nat}
    (h : m'.rdN a 1 = m.rdN a 1) : m'.rd a = m.rd a := by
  cases h1 : m.rd a with
  | none =>
    cases h2 : m'.rd a with
    | none => rfl
    | some v =>
      have e1 : m.rdN a 1 = none := by simp [Mem.rdN, h1]
      have e2 : m'.rdN a 1 = some [v] := by simp [Mem.rdN, h2]
      rw [e2, e1] at h
      exact absurd h (by simp)
  | some v =>
    exact rd_of_rdN1 h h1

/-- `LenLay` transport along point-wise `rd` equality on the
scratch range. -/
theorem LenLay_of_frame : ∀ (code : List Instr) {m m' : Mem}
    {B S c : Nat}, B ≤ c → LenLay m B S c code →
    (∀ a, S + (c - B) ≤ a → a < S + (c + codeBytes code - B) →
      m'.rd a = m.rd a) →
    LenLay m' B S c code := by
  intro code
  induction code with
  | nil => intro m m' B S c _ _ _; trivial
  | cons x xs ih =>
    intro m m' B S c hBc hlay hfr
    obtain ⟨h0, hrest⟩ := hlay
    have hpos : 0 < x.enc.length := by
      obtain ⟨op, rest, heq, _⟩ := enc_head x
      rw [heq]; exact Nat.zero_lt_succ _
    constructor
    · rw [hfr (S + (c - B)) (by omega) (by
        show S + (c - B) <
          S + (c + codeBytes (x :: xs) - B)
        have hcb : codeBytes (x :: xs) =
          x.enc.length + codeBytes xs := rfl
        omega)]
      exact h0
    · apply ih (by omega) hrest
      intro a ha1 ha2
      apply hfr
      · omega
      · have hcb : codeBytes (x :: xs) =
          x.enc.length + codeBytes xs := rfl
        omega

/-- One full word of the parser loop: fields, code loop, ann
skip, entry advance, back to the word-loop head with one fewer
word remaining. -/
theorem parser_word_step {B e S sp p c r : Nat} {wd : Word}
    {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr1 : s.reg 1 = some (r + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr7 : s.reg 7 = some p) (hr4 : s.reg 4 = some sp)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (h20 : s.mem.rdN c 20 = some bs)
    (heq0 : leVal (bs.take 4) % W = wd.ins)
    (heq1 : leVal ((bs.drop 4).take 4) % W = wd.outs)
    (heq2 : leVal ((bs.drop 8).take 4) % W = wd.locals)
    (heq3 : leVal ((bs.drop 12).take 4) % W = wd.code.length)
    (heq4 : leVal ((bs.drop 16).take 4) % W = wd.ann.length)
    (hseq : CodeSeq s.mem (c + 20) wd.code)
    (htot : c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ e)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hr1W : r + 1 < W) (hBc : B ≤ c)
    (hSW : S + (c + 20 + codeBytes wd.code - B) < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hScrB : S + (c + 20 + codeBytes wd.code - B) ≤ c + 20 ∨
      c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ S + (c + 20 - B))
    (hScrE : S + (c + 20 + codeBytes wd.code - B) ≤ p ∨
      p + 64 ≤ S + (c + 20 - B))
    (hScrStk : S + (c + 20 + codeBytes wd.code - B) ≤ sp - 8 ∨
      sp ≤ S + (c + 20 - B))
    (hwrE : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s.mem (p + off) v).isSome)
    (hwrS : ∀ a v, S + (c + 20 - B) ≤ a →
      a < S + (c + 20 + codeBytes wd.code - B) →
      (s.mem.wr a v).isSome) :
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some r ∧
      s'.reg 6 = some (c + 20 + codeBytes wd.code +
        12 * wd.ann.length) ∧
      s'.reg 7 = some (p + 64) ∧ s'.reg 4 = some sp ∧
      rd64 s'.mem p = some wd.ins ∧
      rd64 s'.mem (p + 8) = some wd.outs ∧
      rd64 s'.mem (p + 16) = some wd.locals ∧
      rd64 s'.mem (p + 24) = some wd.code.length ∧
      rd64 s'.mem (p + 32) = some wd.ann.length ∧
      rd64 s'.mem (p + 40) = some ((c + 20) % W) ∧
      rd64 s'.mem (p + 48) =
        some ((c + 20 + codeBytes wd.code) % W) ∧
      CodeSeq s'.mem (c + 20) wd.code ∧
      LenLay s'.mem B S (c + 20) wd.code ∧
      s'.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) =
        s.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) →
        (S + (c + 20 + codeBytes wd.code - B) ≤ x ∨
          x + k ≤ S + (c + 20 - B)) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  -- fields
  obtain ⟨nf, s1, hrunf, hs1rip, hs1r6, hs1r4, hs1r7, hs1s0, hs1s8,
      hs1s16, hs1s24, hs1s32, hs1s40, hs1fr, hs1len, hs1k, hs1sh,
      hs1room, hs1h20, hs1wr, hs1memfr⟩ :=
    parser_word_fields hrip hr6 hr15 hr7 hr4 h20 (by omega) hcW heW
      hpW hroom hsp16 hspW hlen16 hEB hSB hES hwrE
  have hs1seq : CodeSeq s1.mem (c + 20) wd.code := by
    apply CodeSeq_of_frame hseq
    intro y k hy hk
    apply hs1memfr <;> omega
  have hs1ann : s1.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) :=
    hs1memfr _ _ (by omega) (by omega)
  have hs1r1 : s1.reg 1 = some (r + 1) := by
    rw [hs1fr 1 (by decide) (by decide) (by decide) (by decide)]
    exact hr1
  have hs1s24' : rd64 s1.mem (p + 24) = some wd.code.length := by
    rw [hs1s24, heq3]
  have hroom1 : StackRoom s1 1 := by
    intro spx hspx i hi v
    exact hs1room spx hspx i (by omega) v
  -- enter
  obtain ⟨s2, hrun2, hs2rip, hs2r1, hs2r4, hs2pop, hs2s24, hs2fr,
      hs2len, hs2k, hs2sh, hs2memfr⟩ :=
    parser_code_enter hs1rip hs1r1 hs1r7 hs1r4 hroom1 (by omega)
      hspW (by omega) hs1s24' (by omega) hs1len
  have hs2seq : CodeSeq s2.mem (c + 20) wd.code := by
    apply CodeSeq_of_frame hs1seq
    intro y k hy hk
    apply hs2memfr
    omega
  have hs2ann : s2.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) := by
    rw [hs2memfr _ _ (by omega)]; exact hs1ann
  have hs2r6 : s2.reg 6 = some (c + 20) := by
    rw [hs2fr 6 (by decide) (by decide)]; exact hs1r6
  have hs2r15 : s2.reg 15 = some e := by
    rw [hs2fr 15 (by decide) (by decide),
      hs1fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have hs2r3 : s2.reg 3 = some B := by
    rw [hs2fr 3 (by decide) (by decide),
      hs1fr 3 (by decide) (by decide) (by decide) (by decide)]
    exact hr3
  have hs2r11 : s2.reg 11 = some S := by
    rw [hs2fr 11 (by decide) (by decide),
      hs1fr 11 (by decide) (by decide) (by decide) (by decide)]
    exact hr11
  have hs2sh' : wshape s2.mem = wshape s.mem := by rw [hs2sh, hs1sh]
  have hs2wr : ∀ a v, S + (c + 20 - B) ≤ a →
      a < S + (c + 20 + codeBytes wd.code - B) →
      (s2.mem.wr a v).isSome := by
    intro a v ha1 ha2
    have h1 : (s.mem.wrN a [v]).isSome := by
      rw [← wr_eq_wrN]; exact hwrS a v ha1 ha2
    have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
    rwa [← wr_eq_wrN] at h2
  -- code loop
  obtain ⟨k3, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3seq, hs3lay, hs3fr,
      hs3len, hs3k, hs3sh, hs3memfr⟩ :=
    parser_code_loop wd.code hs2rip hs2r1 hs2r6 hs2r15 hs2r3 hs2r11
      hs2seq (by omega) (by omega) heW (by omega) hSW (by omega)
      hs2wr hs2len
  have hs3ann : s3.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) := by
    rw [hs3memfr _ _ (by omega)]; exact hs2ann
  have slot3 : ∀ {off w : Nat}, off + 8 ≤ 48 →
      rd64 s2.mem (p + off) = some w → rd64 s3.mem (p + off) = some w := by
    intro off w hoff h
    unfold rd64
    rw [hs3memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have slot2 : ∀ {off w : Nat}, off + 8 ≤ 64 →
      rd64 s1.mem (p + off) = some w → rd64 s2.mem (p + off) = some w := by
    intro off w _ h
    unfold rd64
    rw [hs2memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have hs3s32 : rd64 s3.mem (p + 32) = some wd.ann.length :=
    slot3 (by decide) (slot2 (by decide) (by rw [hs1s32, heq4]))
  have hs3pop : rd64 s3.mem (sp - 8) = some (r + 1) := by
    unfold rd64
    rw [hs3memfr (sp - 8) 8 (by omega)]
    have h := hs2pop
    unfold rd64 at h
    rwa [Nat.mod_eq_of_lt hr1W] at h
  -- loop exit
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    parser_code_head_zero hs3rip hs3r1
  have hs4r4 : s4.reg 4 = some (sp - 8) := by
    rw [hs4fr 4, hs3fr 4 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs2r4
  have hs4r6 : s4.reg 6 = some (c + 20 + codeBytes wd.code) := by
    rw [hs4fr 6]; exact hs3r6
  have hs4r7 : s4.reg 7 = some p := by
    rw [hs4fr 7, hs3fr 7 (by decide) (by decide) (by decide)
      (by decide) (by decide), hs2fr 7 (by decide) (by decide)]
    exact hs1r7
  have hs4r15 : s4.reg 15 = some e := by
    rw [hs4fr 15, hs3fr 15 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs2r15
  have hs4pop : rd64 s4.mem (sp - 8) = some (r + 1) := by
    rw [hs4mem]; exact hs3pop
  have hs4s32 : rd64 s4.mem (p + 32) = some wd.ann.length := by
    rw [hs4mem]; exact hs3s32
  have hs4sh : wshape s4.mem = wshape s1.mem := by
    rw [hs4mem, hs3sh, hs2sh]
  obtain ⟨m48, hm48⟩ : ∃ m, wr64 s4.mem (p + 48)
      (c + 20 + codeBytes wd.code) = some m := by
    have h1 : (s1.mem.wrN (p + 48)
        (bytes8 (c + 20 + codeBytes wd.code))).isSome :=
      hs1wr 48 _ (by decide)
    have h2 := wrN_some_of_wshape _ s1.mem s4.mem hs4sh _ h1
    exact Option.isSome_iff_exists.mp h2
  -- word tail
  obtain ⟨s5, hrun5, hs5rip, hs5r6, hs5r7, hs5r1, hs5r4, hs5s48,
      hs5s32, hs5fr, hs5len, hs5k, hs5sh, hs5memfr⟩ :=
    parser_word_tail hs4rip hs4r4 hs4r6 hs4r7 hs4r15 hs4pop
      (by omega) (by omega) (by omega) heW hpW (by omega) (by omega)
      hs4s32 hm48 (by rw [hs4len]; exact hs3len)
  have hrun : run stage1Im (nf + 2 + k3 + 3 + 18) (Out.ok s) =
      Out.ok s5 := by
    rw [show nf + 2 + k3 + 3 + 18 = nf + (2 + (k3 + (3 + 18))) by omega,
      run_add stage1Im nf _, hrunf, run_add stage1Im 2 _, hrun2,
      run_add stage1Im k3 _, hrun3, run_add stage1Im 3 18, hrun4, hrun5]
  have slot5 : ∀ {off w : Nat}, off + 8 ≤ 48 →
      rd64 s1.mem (p + off) = some w → rd64 s5.mem (p + off) = some w := by
    intro off w hoff h
    have h3 : rd64 s3.mem (p + off) = some w :=
      slot3 hoff (slot2 (by omega) h)
    have h4 : rd64 s4.mem (p + off) = some w := by rw [hs4mem]; exact h3
    unfold rd64
    rw [hs5memfr (p + off) 8 (by omega)]
    have h' := h4
    unfold rd64 at h'
    exact h'
  have hs5sh' : wshape s5.mem = wshape s.mem := by
    rw [hs5sh, hs4mem, hs3sh, hs2sh, hs1sh]
  have hroom5 : StackRoom s5 2 := by
    apply StackRoom_of_wshape hs5sh' _ hroom
    have h1 : sp - 8 + 8 = sp := by omega
    rw [show s5.reg 4 = some sp by rw [hs5r4, h1], hr4]
  refine ⟨nf + 2 + k3 + 3 + 18, s5, hrun, hs5rip, hs5r1, hs5r6, hs5r7,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, hs5s48, ?_, ?_, ?_, ?_, hs5len, ?_,
    hs5sh', hroom5, ?_⟩
  · have h1 : sp - 8 + 8 = sp := by omega
    rw [hs5r4, h1]
  · have h := slot5 (off := 0) (by decide) (by simpa using hs1s0)
    rwa [heq0] at h
  · have h := slot5 (by decide) hs1s8
    rwa [heq1] at h
  · have h := slot5 (by decide) hs1s16
    rwa [heq2] at h
  · have h := slot5 (by decide) hs1s24
    rwa [heq3] at h
  · exact hs5s32
  · exact slot5 (by decide) hs1s40
  · have h1 : s5.mem.rdN (c + 20) (codeBytes wd.code) =
        s3.mem.rdN (c + 20) (codeBytes wd.code) := by
      rw [hs5memfr _ _ (by omega), hs4mem]
    exact CodeSeq_of_frame hs3seq (fun y k hy hk => by
      have e1 : s5.mem.rdN y k = s4.mem.rdN y k :=
        hs5memfr y k (by omega)
      rw [e1, hs4mem])
  · apply LenLay_of_frame wd.code (by omega) hs3lay
    intro a ha1 ha2
    have e1 : s5.mem.rd a = s4.mem.rd a :=
      rd_eq_of_rdN1 (hs5memfr a 1 (by omega))
    rw [e1, hs4mem]
  · rw [hs5memfr _ _ (by omega), hs4mem]; exact hs3ann
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
    rw [hs5fr i hi0 hi1 hi2 hi4 hi6 hi7, hs4fr i,
      hs3fr i hi0 hi1 hi2 hi6 hi10,
      hs2fr i hi1 hi4]
    exact hs1fr i hi0 hi2 hi4 hi6
  · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
  · intro x k hd1 hd2 hd3
    rw [hs5memfr x k (by omega), hs4mem, hs3memfr x k hd3,
      hs2memfr x k (by omega), hs1memfr x k hd1 hd2]

/-- Buffer bytes consumed by a word list. -/
def wordsBytes : List Word → Nat
  | [] => 0
  | wd :: wds => (20 + codeBytes wd.code + 12 * wd.ann.length) +
      wordsBytes wds

/-- Memory-level annotation layout: entry `i` at `p + 12*i`
holds `(index, d, a)` as three `u32`s. -/
def AnnLayB (m : Mem) (p : Nat) (ann : List (Nat × Height)) : Prop :=
  ∀ i (hi : i < ann.length),
    (m.rdN (p + 12 * i) 4).map leVal = some ann[i].1 ∧
    (m.rdN (p + 12 * i + 4) 4).map leVal = some ann[i].2.d ∧
    (m.rdN (p + 12 * i + 8) 4).map leVal = some ann[i].2.a ∧
    ann[i].1 < W

theorem AnnLayB_of_frame {m m' : Mem} {p : Nat}
    {ann : List (Nat × Height)}
    (h : AnnLayB m p ann)
    (hfr : ∀ x k, p ≤ x → x + k ≤ p + 12 * ann.length →
      m'.rdN x k = m.rdN x k) : AnnLayB m' p ann := by
  intro i hi
  obtain ⟨h1, h2, h3, h4⟩ := h i hi
  refine ⟨?_, ?_, ?_, h4⟩
  · rw [hfr (p + 12 * i) 4 (by omega) (by omega)]; exact h1
  · rw [hfr (p + 12 * i + 4) 4 (by omega) (by omega)]; exact h2
  · rw [hfr (p + 12 * i + 8) 4 (by omega) (by omega)]; exact h3

/-- What the parser needs laid out in the buffer for a word
list: header fields, code encodings and annotation arrays. -/
def WordsLay (m : Mem) (c : Nat) : List Word → Prop
  | [] => True
  | wd :: wds =>
    (∃ bs, m.rdN c 20 = some bs ∧
      leVal (bs.take 4) % W = wd.ins ∧
      leVal ((bs.drop 4).take 4) % W = wd.outs ∧
      leVal ((bs.drop 8).take 4) % W = wd.locals ∧
      leVal ((bs.drop 12).take 4) % W = wd.code.length ∧
      leVal ((bs.drop 16).take 4) % W = wd.ann.length) ∧
    CodeSeq m (c + 20) wd.code ∧
    AnnLayB m (c + 20 + codeBytes wd.code) wd.ann ∧
    WordsLay m (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds

theorem WordsLay_of_frame : ∀ (wds : List Word) {m m' : Mem}
    {c : Nat}, WordsLay m c wds →
    (∀ x k, c ≤ x → x + k ≤ c + wordsBytes wds →
      m'.rdN x k = m.rdN x k) →
    WordsLay m' c wds := by
  intro wds
  induction wds with
  | nil => intro m m' c _ _; trivial
  | cons wd wds ih =>
    intro m m' c h hfr
    obtain ⟨⟨bs, h20, e0, e1, e2, e3, e4⟩, hseq, hann, hrest⟩ := h
    have hcb : wordsBytes (wd :: wds) =
        (20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds := rfl
    refine ⟨⟨bs, ?_, e0, e1, e2, e3, e4⟩, ?_, ?_, ?_⟩
    · rw [hfr c 20 (by omega) (by omega)]; exact h20
    · apply CodeSeq_of_frame hseq
      intro y k hy hk
      apply hfr <;> omega
    · apply AnnLayB_of_frame hann
      intro y k hy hk
      apply hfr <;> omega
    · apply ih hrest
      intro y k hy hk
      apply hfr <;> omega

/-- What the parser leaves in the words table, code buffer,
length scratch and ann arrays for a word list. -/
def TabLay (m : Mem) (B S tbl c : Nat) : List Word → Prop
  | [] => True
  | wd :: wds =>
    rd64 m tbl = some wd.ins ∧
    rd64 m (tbl + 8) = some wd.outs ∧
    rd64 m (tbl + 16) = some wd.locals ∧
    rd64 m (tbl + 24) = some wd.code.length ∧
    rd64 m (tbl + 32) = some wd.ann.length ∧
    rd64 m (tbl + 40) = some ((c + 20) % W) ∧
    rd64 m (tbl + 48) = some ((c + 20 + codeBytes wd.code) % W) ∧
    CodeSeq m (c + 20) wd.code ∧
    LenLay m B S (c + 20) wd.code ∧
    AnnLayB m (c + 20 + codeBytes wd.code) wd.ann ∧
    TabLay m B S (tbl + 64)
      (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds

set_option maxHeartbeats 0 in
/-- The whole parser word loop over a word list: from the
word-loop head back to the word-loop head with count zero and
the full table laid out. -/
theorem parser_words_loop : ∀ (wds : List Word)
    {B e S sp p c : Nat} {s : X86Int.St},
    s.rip = base + 0x2cc → s.reg 1 = some wds.length →
    s.reg 6 = some c → s.reg 7 = some p →
    s.reg 15 = some e → s.reg 4 = some sp →
    s.reg 3 = some B → s.reg 11 = some S →
    WordsLay s.mem c wds →
    B ≤ c → c + wordsBytes wds ≤ e → e < W →
    p + 64 * wds.length < W → S + (c + wordsBytes wds - B) < W →
    wds.length < W →
    StackRoom s 2 → 16 ≤ sp → sp < W → s.regs.length = 16 →
    (p + 64 * wds.length ≤ B ∨ e ≤ p) →
    (p + 64 * wds.length ≤ sp - 16 ∨ sp ≤ p) →
    (p + 64 * wds.length ≤ S + (c - B) ∨
      S + (c + wordsBytes wds - B) ≤ p) →
    (e ≤ sp - 16 ∨ sp ≤ B) →
    (S + (c + wordsBytes wds - B) ≤ c ∨
      c + wordsBytes wds ≤ S + (c - B)) →
    (S + (c + wordsBytes wds - B) ≤ sp - 16 ∨ sp ≤ S + (c - B)) →
    (∀ (off v : Nat), off + 8 ≤ 64 * wds.length →
      (wr64 s.mem (p + off) v).isSome) →
    (∀ a v, S + (c - B) ≤ a → a < S + (c + wordsBytes wds - B) →
      (s.mem.wr a v).isSome) →
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + wordsBytes wds) ∧
      s'.reg 7 = some (p + 64 * wds.length) ∧
      s'.reg 4 = some sp ∧
      TabLay s'.mem B S p c wds ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 * wds.length ≤ x ∨ x + k ≤ p) →
        (S + (c + wordsBytes wds - B) ≤ x ∨ x + k ≤ S + (c - B)) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  intro wds
  induction wds with
  | nil =>
    intro B e S sp p c s hrip hr1 hr6 hr7 _ hr4 _ _ _ _ _ _ _ _ _
      hroom _ _ hlen16 _ _ _ _ _ _ _ _
    refine ⟨0, s, rfl, hrip, hr1, ?_, ?_, hr4, trivial, ?_, hlen16,
      rfl, rfl, hroom, ?_⟩
    · show s.reg 6 = some (c + wordsBytes [])
      exact hr6
    · show s.reg 7 = some (p + 64 * [].length)
      exact hr7
    · intro i _ _ _ _ _ _ _; exact rfl
    · intro x k _ _ _; exact rfl
  | cons wd wds ih =>
    intro B e S sp p c s hrip hr1 hr6 hr7 hr15 hr4 hr3 hr11 hlay hBc
      htot heW hpW hSW hlenW hroom hsp16 hspW hlen16 hTB hTStk hTScr
      hBStk hBScr hScrStk hwrE hwrS
    obtain ⟨⟨bs, h20, e0, e1, e2, e3, e4⟩, hseq, hann, hrest⟩ := hlay
    have hwb : wordsBytes (wd :: wds) =
        (20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds := rfl
    have hlen' : wds.length + 1 = (wd :: wds).length := rfl
    have hcW : c < W := by omega
    -- head: enter the body
    obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
      parser_words_head_pos hrip hr1 (by omega) hlenW
    have hs1len16 : s1.regs.length = 16 := by rw [hs1len]; exact hlen16
    have hs1r : ∀ i v, s.reg i = some v → s1.reg i = some v := by
      intro i v h
      rw [hs1fr i]; exact h
    -- one word step
    obtain ⟨k1, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r7, hs2r4, hs2s0,
        hs2s8, hs2s16, hs2s24, hs2s32, hs2s40, hs2s48, hs2seq, hs2lay,
        hs2ann, hs2fr, hs2len, hs2k, hs2sh, hs2room, hs2memfr⟩ :=
      parser_word_step hs1rip (hs1r 1 _ hr1) (hs1r 6 _ hr6)
        (hs1r 15 _ hr15) (hs1r 7 _ hr7) (hs1r 4 _ hr4) (hs1r 3 _ hr3)
        (hs1r 11 _ hr11) (by rw [hs1mem]; exact h20) e0 e1 e2 e3 e4
        (by rw [hs1mem]; exact hseq) (by omega) hcW heW (by omega)
        (by omega) hBc (by omega)
        (StackRoom_of_mem hs1mem (hs1fr 4) hroom)
        hsp16 hspW hs1len16 (by omega) (by omega) (by omega)
        (by omega) (by omega) (by omega)
        (by
          intro off v hoff
          rw [hs1mem]
          exact hwrE off v (by omega))
        (by
          intro a v ha1 ha2
          rw [hs1mem]
          exact hwrS a v (by omega) (by omega))
    have hs2sh' : wshape s2.mem = wshape s.mem := by
      rw [hs2sh, hs1mem]
    have hs2lay' : WordsLay s2.mem
        (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds := by
      apply WordsLay_of_frame wds hrest
      intro y k hy hk
      rw [hs2memfr y k (by omega) (by omega) (by omega), hs1mem]
    have hs2wrE : ∀ (off v : Nat), off + 8 ≤ 64 * wds.length →
        (wr64 s2.mem (p + 64 + off) v).isSome := by
      intro off v hoff
      have h1 : (s.mem.wrN (p + 64 + off) (bytes8 v)).isSome := by
        have h2 := hwrE (64 + off) v (by omega)
        rwa [show p + (64 + off) = p + 64 + off by omega] at h2
      exact wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
    have hs2wrS : ∀ a v,
        S + (c + 20 + codeBytes wd.code + 12 * wd.ann.length - B) ≤ a →
        a < S + (c + 20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds - B) →
        (s2.mem.wr a v).isSome := by
      intro a v ha1 ha2
      have h1 : (s.mem.wrN a [v]).isSome := by
        rw [← wr_eq_wrN]
        exact hwrS a v (by omega) (by omega)
      have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
      rwa [← wr_eq_wrN] at h2
    have hs2r15 : s2.reg 15 = some e := by
      rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 15]
      exact hr15
    have hs2r3 : s2.reg 3 = some B := by
      rw [hs2fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 3]
      exact hr3
    have hs2r11 : s2.reg 11 = some S := by
      rw [hs2fr 11 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 11]
      exact hr11
    -- induction over the rest
    obtain ⟨k2, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3r7, hs3r4, hs3tab,
        hs3fr, hs3len, hs3k, hs3sh, hs3room, hs3memfr⟩ :=
      ih hs2rip hs2r1 hs2r6 hs2r7 hs2r15 hs2r4 hs2r3 hs2r11 hs2lay'
        (by omega) (by omega) heW (by omega) (by omega) (by omega)
        hs2room hsp16 hspW hs2len (by omega) (by omega) (by omega)
        hBStk (by omega) (by omega) hs2wrE hs2wrS
    have hrun : run stage1Im (2 + k1 + k2) (Out.ok s) = Out.ok s3 := by
      rw [show 2 + k1 + k2 = 2 + (k1 + k2) by omega,
        run_add stage1Im 2 _, hrun1, run_add stage1Im k1 k2, hrun2,
        hrun3]
    have slot3 : ∀ {off w : Nat}, off + 8 ≤ 56 →
        rd64 s2.mem (p + off) = some w →
        rd64 s3.mem (p + off) = some w := by
      intro off w hoff h
      unfold rd64
      rw [hs3memfr (p + off) 8 (by omega) (by omega) (by omega)]
      have h' := h
      unfold rd64 at h'
      exact h'
    have hseq3 : CodeSeq s3.mem (c + 20) wd.code := by
      apply CodeSeq_of_frame hs2seq
      intro y k hy hk
      apply hs3memfr <;> omega
    have hlay3 : LenLay s3.mem B S (c + 20) wd.code := by
      apply LenLay_of_frame wd.code (by omega) hs2lay
      intro a ha1 ha2
      exact rd_eq_of_rdN1 (hs3memfr a 1 (by omega) (by omega)
        (by omega))
    have hann2 : AnnLayB s2.mem (c + 20 + codeBytes wd.code)
        wd.ann := by
      apply AnnLayB_of_frame hann
      intro y k hy hk
      rw [hs2memfr y k (by omega) (by omega) (by omega), hs1mem]
    have hann3 : AnnLayB s3.mem (c + 20 + codeBytes wd.code)
        wd.ann := by
      apply AnnLayB_of_frame hann2
      intro y k hy hk
      apply hs3memfr <;> omega
    have htab : TabLay s3.mem B S p c (wd :: wds) :=
      ⟨slot3 (off := 0) (by decide) (by simpa using hs2s0),
        slot3 (by decide) hs2s8, slot3 (by decide) hs2s16,
        slot3 (by decide) hs2s24, slot3 (by decide) hs2s32,
        slot3 (by decide) hs2s40, slot3 (by decide) hs2s48,
        hseq3, hlay3, hann3, hs3tab⟩
    refine ⟨2 + k1 + k2, s3, hrun, hs3rip, hs3r1, ?_, ?_, hs3r4, htab,
      ?_, hs3len, ?_, ?_, hs3room, ?_⟩
    · show s3.reg 6 = some (c + wordsBytes (wd :: wds))
      rw [hwb]
      have h := hs3r6
      rwa [show c + 20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds =
        c + (20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds) by omega] at h
    · show s3.reg 7 = some (p + 64 * (wd :: wds).length)
      have h := hs3r7
      rwa [show p + 64 + 64 * wds.length =
        p + 64 * (wd :: wds).length by
        rw [← hlen']; omega] at h
    · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
      rw [hs3fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10,
        hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10, hs1fr i]
    · rw [hs3k, hs2k, hs1k]
    · rw [hs3sh, hs2sh, hs1mem]
    · intro x k hd1 hd2 hd3
      rw [hs3memfr x k hd1 (by omega) (by omega),
        hs2memfr x k hd1 (by omega) (by omega), hs1mem]

set_option maxHeartbeats 0 in
/-- Trailing checks: cursor at buffer end, `main` below the
word count, and the main word's `ins` field zero; falls into
the checker entry. -/
theorem parser_trailing_ok {c e mn nw T : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw) (hr14 : s.reg 14 = some T)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : mn < nw) (hTW : T + 64 * mn < W)
    (hslot : rd64 s.mem (T + 64 * mn) = some 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧ s'.reg 0 = some 0 ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hdbl : ∀ v : Nat, 2 * v < W → (aluF .add v v).1 = 2 * v := by
    intro v hv
    rw [aluF_add_fst, show v + v = 2 * v by omega,
      Nat.mod_eq_of_lt hv]
  -- 0x419: cmp rsi, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } :=
    step_cmp (by rw [hrip]; exact hc.p419) hr6 hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x41c := by
    rw [ht1]; show s.rip + 3 = base + 0x41c; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 4 = some true := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .sub c e).2.zf = some true
      rw [aluF_zf, hceq]
      have hbeq : ((aluF .sub e e).1 == 0) = true := by
        rw [beq_iff_eq]; exact sub_fst_self e
      rw [hbeq]
    have hcc := cond_cc4 hzf
    simpa using hcc
  have ht1fr : ∀ i, t1.reg i = s.reg i := by
    intro i
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  have ht1mem : t1.mem = s.mem := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by rw [ht1]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  -- 0x41c: je 0x423
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x423 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p41c) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x423 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x423 := by rw [ht2]
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht1fr i
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).mem = s.mem
    rw [mem_with_rip]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).regs.length = 16
    rw [len_with_rip]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  -- 0x423: cmp r12, r13
  have ht2r12 : t2.reg 12 = some mn := by rw [ht2fr 12]; exact hr12
  have ht2r13 : t2.reg 13 = some nw := by rw [ht2fr 13]; exact hr13
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with fl := (aluF .sub mn nw).2, rip := t2.rip + 3 } :=
    step_cmp (by rw [ht2rip]; exact hc.p423) ht2r12 ht2r13
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with fl := (aluF .sub mn nw).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x426 := by
    rw [ht3]; show t2.rip + 3 = base + 0x426; rw [ht2rip]
  have ht3fl : X86Int.cond t3.fl 2 = some true := by
    have hcf : t3.fl.cf = some true := by
      rw [ht3]
      show (aluF .sub mn nw).2.cf = some true
      exact aluF_sub_cf_true hmn
    have hcc := cond_cc2 hcf
    simpa using hcc
  have ht3fr : ∀ i, t3.reg i = s.reg i := by
    intro i
    rw [ht3]
    show t2.reg i = s.reg i
    exact ht2fr i
  have ht3mem : t3.mem = s.mem := by rw [ht3]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by rw [ht3]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  -- 0x426: jb 0x42d
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x42d } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p426) ht3fl
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x42d } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x42d := by rw [ht4]
  have ht4fr : ∀ i, t4.reg i = s.reg i := by
    intro i
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).reg i = s.reg i
    rw [reg_with_rip]; exact ht3fr i
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).mem = s.mem
    rw [mem_with_rip]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).regs.length = 16
    rw [len_with_rip]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  -- 0x42d: mov rdx, r12; six doublings; add rdx, r14
  have ht4r12 : t4.reg 12 = some mn := by rw [ht4fr 12]; exact hr12
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 2 (some mn) with rip := t4.rip + 3 } :=
    step_mov (by rw [ht4rip]; exact hc.p42d) ht4r12
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 2 (some mn) with rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x430 := by
    rw [ht5]; show t4.rip + 3 = base + 0x430; rw [ht4rip]
  have ht5r2 : t5.reg 2 = some mn := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).reg 2 = some mn
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide)]
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht4fr i
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  -- six doublings of rdx, as one chain of explicit steps
  have dbl_step : ∀ {u : X86Int.St} {a v : Nat},
      stage1Im a = some (.alu .add 2 2, 3) → u.rip = a →
      u.reg 2 = some v → 2 * v < W → u.regs.length = 16 →
      ∃ u', run stage1Im 1 (Out.ok u) = Out.ok u' ∧
        u'.rip = a + 3 ∧ u'.reg 2 = some (2 * v) ∧
        (∀ i, i ≠ 2 → u'.reg i = u.reg i) ∧ u'.mem = u.mem ∧
        u'.regs.length = 16 ∧ u'.k = u.k := by
    intro u a v hdec hurip hur2 hv hulen
    have e : X86Int.step stage1Im u =
        .ok { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } :=
      step_alu_add (by rw [hurip]; exact hdec) hur2 hur2
    obtain ⟨u1, hu1⟩ : ∃ t : X86Int.St, t =
        { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } := ⟨_, rfl⟩
    have rr : run stage1Im 1 (Out.ok u) = Out.ok u1 := by
      rw [run_succ_ok stage1Im 0 u, e, ← hu1, run_zero]
    refine ⟨u1, rr, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu1]; show u.rip + 3 = a + 3; rw [hurip]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).reg 2 = some (2 * v)
      rw [reg_set_self (by rw [hulen]; decide), hdbl v hv]
    · intro i hi
      rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).reg i = u.reg i
      rw [reg_set_other hi]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).mem = u.mem
      rw [mem_set]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).regs.length = 16
      rw [len_set]; exact hulen
    · rw [hu1]; exact rfl
  obtain ⟨t6, r6, ht6rip, ht6r2, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    dbl_step hc.p430 ht5rip ht5r2 (by omega) ht5len
  have ht6rip' : t6.rip = base + 0x433 := ht6rip
  obtain ⟨t7, r7, ht7rip, ht7r2, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    dbl_step hc.p433 ht6rip' ht6r2 (by omega) ht6len
  have ht7rip' : t7.rip = base + 0x436 := ht7rip
  have ht7r2' : t7.reg 2 = some (4 * mn) := by
    have h := ht7r2
    rwa [show 2 * (2 * mn) = 4 * mn by omega] at h
  obtain ⟨t8, r8, ht8rip, ht8r2, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    dbl_step hc.p436 ht7rip' ht7r2' (by omega) ht7len
  have ht8rip' : t8.rip = base + 0x439 := ht8rip
  have ht8r2' : t8.reg 2 = some (8 * mn) := by
    have h := ht8r2
    rwa [show 2 * (4 * mn) = 8 * mn by omega] at h
  obtain ⟨t9, r9, ht9rip, ht9r2, ht9fr, ht9mem, ht9len, ht9k⟩ :=
    dbl_step hc.p439 ht8rip' ht8r2' (by omega) ht8len
  have ht9rip' : t9.rip = base + 0x43c := ht9rip
  have ht9r2' : t9.reg 2 = some (16 * mn) := by
    have h := ht9r2
    rwa [show 2 * (8 * mn) = 16 * mn by omega] at h
  obtain ⟨t10, r10, ht10rip, ht10r2, ht10fr, ht10mem, ht10len,
      ht10k⟩ :=
    dbl_step hc.p43c ht9rip' ht9r2' (by omega) ht9len
  have ht10rip' : t10.rip = base + 0x43f := ht10rip
  have ht10r2' : t10.reg 2 = some (32 * mn) := by
    have h := ht10r2
    rwa [show 2 * (16 * mn) = 32 * mn by omega] at h
  obtain ⟨t11, r11, ht11rip, ht11r2, ht11fr, ht11mem, ht11len,
      ht11k⟩ :=
    dbl_step hc.p43f ht10rip' ht10r2' (by omega) ht10len
  have ht11rip' : t11.rip = base + 0x442 := ht11rip
  have ht11r2' : t11.reg 2 = some (64 * mn) := by
    have h := ht11r2
    rwa [show 2 * (32 * mn) = 64 * mn by omega] at h
  have ht11fr' : ∀ i, i ≠ 2 → t11.reg i = s.reg i := by
    intro i hi
    rw [ht11fr i hi, ht10fr i hi, ht9fr i hi, ht8fr i hi, ht7fr i hi,
      ht6fr i hi]
    exact ht5fr i hi
  have ht11r14 : t11.reg 14 = some T := by
    rw [ht11fr' 14 (by decide)]; exact hr14
  have ht11mem' : t11.mem = s.mem := by
    rw [ht11mem, ht10mem, ht9mem, ht8mem, ht7mem, ht6mem]; exact ht5mem
  have ht11k' : t11.k = s.k := by
    rw [ht11k, ht10k, ht9k, ht8k, ht7k, ht6k]; exact ht5k
  -- 0x442: add rdx, r14
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .add (64 * mn) T).1) with
        fl := (aluF .add (64 * mn) T).2, rip := t11.rip + 3 } :=
    step_alu_add (by rw [ht11rip']; exact hc.p442) ht11r2' ht11r14
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .add (64 * mn) T).1) with
        fl := (aluF .add (64 * mn) T).2, rip := t11.rip + 3 } :=
    ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x445 := by
    rw [ht12]; show t11.rip + 3 = base + 0x445; rw [ht11rip']
  have ht12r2 : t12.reg 2 = some (T + 64 * mn) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).reg 2 =
      some (T + 64 * mn)
    rw [reg_set_self (by rw [ht11len]; decide), aluF_add_fst,
      show 64 * mn + T = T + 64 * mn by omega,
      Nat.mod_eq_of_lt hTW]
  have ht12fr : ∀ i, i ≠ 2 → t12.reg i = s.reg i := by
    intro i hi
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht11fr' i hi
  have ht12mem : t12.mem = s.mem := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).mem = s.mem
    rw [mem_set]; exact ht11mem'
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).regs.length = 16
    rw [len_set]; exact ht11len
  have ht12k : t12.k = s.k := by rw [ht12]; exact ht11k'
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  -- 0x445: ld rax, [rdx]
  have hld : rd64 t12.mem (ea (T + 64 * mn) 0) = some 0 := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hTW, ht12mem]; exact hslot
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12.set 0 (some 0) with rip := t12.rip + 3 } :=
    step_ld (by rw [ht12rip]; exact hc.p445) ht12r2 hld
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12.set 0 (some 0) with rip := t12.rip + 3 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x448 := by
    rw [ht13]; show t12.rip + 3 = base + 0x448; rw [ht12rip]
  have ht13r0 : t13.reg 0 = some 0 := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).reg 0 = some 0
    rw [reg_with_rip, reg_set_self (by rw [ht12len]; decide)]
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → t13.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht12fr i hi2
  have ht13mem : t13.mem = s.mem := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht12len
  have ht13k : t13.k = s.k := by rw [ht13]; exact ht12k
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  -- 0x448: test rax, rax; 0x44b: je checker entry
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with fl := (aluF .and 0 0).2, rip := t13.rip + 3 } :=
    step_test (by rw [ht13rip]; exact hc.p448) ht13r0 ht13r0
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with fl := (aluF .and 0 0).2, rip := t13.rip + 3 } :=
    ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x44b := by
    rw [ht14]; show t13.rip + 3 = base + 0x44b; rw [ht13rip]
  have ht14fl : X86Int.cond t14.fl 4 = some true := by
    have hzf : t14.fl.zf = some true := by
      rw [ht14]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc4 hzf
    simpa using hcc
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14 with rip := base + 0x452 } := by
    have h := step_jcc (im := stage1Im) (s := t14)
      (by rw [ht14rip]; exact hc.p44b) ht14fl
    simpa using h
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14 with rip := base + 0x452 } := ⟨_, rfl⟩
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok t15 := by
    rw [show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r1,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r2,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r3,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r4,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r5,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r6,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r7,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r8,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r9,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r10,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r11,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r12,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r13,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r14, r15]
  refine ⟨t15, hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht15]
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).reg 0 = some 0
    rw [reg_with_rip, ht14]
    show t13.reg 0 = some 0
    exact ht13r0
  · intro i hi0 hi2
    rw [ht15]
    show ({ t14 with rip := base + 0x452 }).reg i = s.reg i
    rw [reg_with_rip, ht14]
    show t13.reg i = s.reg i
    exact ht13fr i hi0 hi2
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).mem = s.mem
    rw [mem_with_rip, ht14]
    show t13.mem = s.mem
    exact ht13mem
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).regs.length = 16
    rw [len_with_rip, ht14]
    show t13.regs.length = 16
    exact ht13len
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).k = s.k
    rw [ht14]
    show t13.k = s.k
    exact ht13k

/-- `leBytes` has the requested length. -/
theorem leBytes_length (n k : Nat) : (leBytes n k).length = k := by
  simp [leBytes]

/-- Head/tail decomposition of `leBytes`. -/
theorem leBytes_succ (n k : Nat) :
    leBytes n (k + 1) = (n % 256) :: leBytes (n / 256) k := by
  apply List.ext_getElem
  · simp [leBytes]
  · intro i hi1 _
    cases i with
    | zero => simp [leBytes]
    | succ j =>
      have hj : j < k := by
        have h := hi1
        simp [leBytes] at h
        omega
      have e1 : (leBytes n (k + 1))[j + 1]'(by
            simpa [leBytes] using hi1) = n / 256 ^ (j + 1) % 256 := by
        simp [leBytes, hj]
      have e2 : ((n % 256) :: leBytes (n / 256) k)[j + 1]'(by
            simp [leBytes, hj]) = n / 256 / 256 ^ j % 256 := by
        simp [leBytes, hj]
      rw [e1, e2, Nat.div_div_eq_div_mul]
      congr 1
      rw [Nat.pow_succ']

/-- A byte-bounded list has `leVal` below the place bound. -/
theorem leVal_bound : ∀ {bs : List Nat}, (∀ b ∈ bs, b < 256) →
    leVal bs < 256 ^ bs.length
  | [], _ => by decide
  | b :: bs, h => by
    have hb : b < 256 := h b (.head _)
    have ih := leVal_bound (fun x hx => h x (.tail _ hx))
    show leVal (b :: bs) < 256 ^ (b :: bs).length
    simp only [leVal, List.length_cons, Nat.pow_succ]
    omega

/-- `leBytes` of a byte-bounded list's `leVal` is the list. -/
theorem leBytes_leVal : ∀ {bs : List Nat}, (∀ b ∈ bs, b < 256) →
    leBytes (leVal bs) bs.length = bs
  | [], _ => by simp [leBytes, leVal]
  | b :: bs, h => by
    have hb : b < 256 := h b (.head _)
    have ih := leBytes_leVal (fun x hx => h x (.tail _ hx))
    show leBytes (leVal (b :: bs)) (b :: bs).length = b :: bs
    simp only [List.length_cons]
    rw [show leVal (b :: bs) = b + 256 * leVal bs from rfl,
      leBytes_succ]
    congr 1
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb]
    · have e : (b + 256 * leVal bs) / 256 = leVal bs := by
        rw [Nat.add_mul_div_left _ _ (by decide : 0 < 256),
          Nat.div_eq_of_lt hb, Nat.zero_add]
      rw [e]; exact ih

/-- The `rdLE` fold is `leVal`, at a shifted range. -/
theorem fold_zip_range'_aux : ∀ (l : List Nat) (s acc : Nat),
    (l.zip (List.range' s l.length)).foldl
      (fun acc x => acc + x.1 * 256 ^ x.2) acc =
      acc + 256 ^ s * leVal l
  | [], _, _ => by
    rw [show leVal [] = 0 from rfl]; simp
  | b :: bs, s, acc => by
    have ih := fold_zip_range'_aux bs (s + 1) (acc + b * 256 ^ s)
    show (((b :: bs).zip
      (List.range' s (b :: bs).length)).foldl _ acc) = _
    simp only [List.length_cons, List.range'_succ, List.zip_cons_cons,
      List.foldl_cons]
    rw [ih]
    have e1 : 256 ^ (s + 1) * leVal bs = 256 ^ s * (256 * leVal bs) := by
      rw [Nat.pow_succ', Nat.mul_assoc, Nat.mul_left_comm (256 ^ s) 256]
    have e2 : leVal (b :: bs) = b + 256 * leVal bs := rfl
    rw [e1, e2, Nat.mul_add, Nat.mul_comm b (256 ^ s)]
    omega

/-- The `rdLE` fold is `leVal`. -/
theorem fold_zip_range (l : List Nat) :
    (l.zip (List.range l.length)).foldl
      (fun acc x => acc + x.1 * 256 ^ x.2) 0 = leVal l := by
  rw [List.range_eq_range']
  have h := fold_zip_range'_aux l 0 0
  simpa using h

/-- What a successful `rdLE` says. -/
theorem rdLE_eq {bs : List Nat} {k v : Nat} {rest : List Nat}
    (h : rdLE bs k = some (v, rest)) :
    v = leVal (bs.take k) ∧ rest = bs.drop k ∧ k ≤ bs.length := by
  unfold rdLE at h
  by_cases hlt : bs.length < k
  · rw [if_pos hlt] at h
    exact absurd h (by simp)
  · rw [if_neg hlt] at h
    obtain ⟨hv, hr⟩ := Prod.mk.inj (Option.some.inj h)
    refine ⟨?_, hr.symm, by omega⟩
    have hlen : (bs.take k).length = k := by
      rw [List.length_take]
      omega
    have h2 := fold_zip_range (bs.take k)
    rw [hlen] at h2
    rw [← h2]
    exact hv.symm

/-- A successful `rdLE` on bounded bytes re-encodes exactly. -/
theorem rdLE_inv {bs : List Nat} {k v : Nat} {rest : List Nat}
    (h : rdLE bs k = some (v, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = leBytes v k ++ rest ∧ v < 256 ^ k ∧ k ≤ bs.length := by
  obtain ⟨hv, hr, hk⟩ := rdLE_eq h
  have hlen : (bs.take k).length = k := by
    rw [List.length_take]; omega
  have hmem : ∀ b ∈ bs.take k, b < 256 := fun b hbmem =>
    hb b (List.mem_of_mem_take hbmem)
  have htake : leBytes v k = bs.take k := by
    have ht := leBytes_leVal hmem
    rw [hlen, ← hv] at ht
    exact ht
  have hbound : v < 256 ^ k := by
    have ht := leVal_bound hmem
    rw [hlen] at ht
    rw [hv]
    exact ht
  refine ⟨?_, hbound, hk⟩
  calc bs = bs.take k ++ bs.drop k := (List.take_append_drop k bs).symm
    _ = leBytes v k ++ rest := by rw [← htake, ← hr]

/-- Immediate bounds forced by the fixed-width encodings. -/
def InstrBound : Instr → Prop
  | .prim _ => True
  | .lit n => n < 256 ^ 8
  | .str i => i < 256 ^ 4
  | .pick n => n < 256 ^ 4
  | .tor => True
  | .fromr => True
  | .rfetch => True
  | .locget i => i < 256 ^ 4
  | .locset i => i < 256 ^ 4
  | .jmp t => t < 256 ^ 4
  | .jz t => t < 256 ^ 4
  | .jnz t => t < 256 ^ 4
  | .call w => w < 256 ^ 4
  | .ret => True

/-- There are 57 primitives. -/
theorem prim_all_length : Prim.all.length = 57 := by decide

/-- The code of the `op`-th primitive is `op`. -/
theorem prim_code_get (op : Nat) (h : op < Prim.all.length) :
    (Prim.all[op]'h).code = op := by
  have h57 : op < 57 := by
    rw [prim_all_length] at h; exact h
  have e : Prim.all[op]? = some (Prim.all[op]'h) :=
    List.getElem?_eq_getElem h
  have chk : (List.range 57).all
      (fun i => (Prim.all[i]?.map Prim.code) == some i) = true := by decide
  have hm := List.all_eq_true.mp chk op (List.mem_range.mpr h57)
  rw [e] at hm
  simp at hm
  exact hm

/-- An opcode that is not a primitive and not a known opcode
decodes to nothing. -/
theorem decInstr_none_of {op : Nat} {r : List Nat}
    (hop : ¬ op < Prim.all.length)
    (h80 : op ≠ 0x80) (h81 : op ≠ 0x81) (h82 : op ≠ 0x82)
    (h83 : op ≠ 0x83) (h84 : op ≠ 0x84) (h85 : op ≠ 0x85)
    (h86 : op ≠ 0x86) (h87 : op ≠ 0x87) (h88 : op ≠ 0x88)
    (h89 : op ≠ 0x89) (h8a : op ≠ 0x8a) (h8b : op ≠ 0x8b)
    (h8c : op ≠ 0x8c) :
    decInstr (op :: r) = none := by
  unfold decInstr
  split
  · simp_all
  · split
    · simp_all <;> omega
    · split <;> simp_all

/-- Inversion of an immediate-carrying decode. -/
theorem imm_inv {k op : Nat} {f : Nat → Instr}
    {r rest : List Nat} {x : Instr}
    (hb : ∀ b ∈ r, b < 256)
    (h : (rdLE r k).map (fun y => (f y.1, y.2)) = some (x, rest))
    (henc : ∀ v, (f v).enc = op :: leBytes v k) :
    ∃ v, x = f v ∧ v < 256 ^ k ∧ op :: r = x.enc ++ rest := by
  rw [Option.map_eq_some_iff] at h
  obtain ⟨⟨v, r2⟩, hr, hfx⟩ := h
  have hx : f v = x := congrArg Prod.fst hfx
  have hr2 : r2 = rest := congrArg Prod.snd hfx
  obtain ⟨hr0, hbnd, -⟩ := rdLE_inv hr hb
  refine ⟨v, hx.symm, hbnd, ?_⟩
  calc op :: r = op :: (leBytes v k ++ r2) := by rw [hr0]
    _ = (op :: leBytes v k) ++ rest := by
      simp only [List.cons_append, hr2]
    _ = x.enc ++ rest := by rw [← henc v, hx]

/-- A decoded instruction re-encodes to exactly the bytes it
consumed, with bounded immediates. -/
theorem decInstr_inv {bs rest : List Nat} {x : Instr}
    (h : decInstr bs = some (x, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = x.enc ++ rest ∧ InstrBound x := by
  cases bs with
  | nil => simp [decInstr] at h
  | cons op rest0 =>
    have hb0 : ∀ b ∈ rest0, b < 256 := fun b hbmem =>
      hb b (.tail _ hbmem)
    by_cases hop : op < Prim.all.length
    · have e : decInstr (op :: rest0) =
          some (Instr.prim (Prim.all[op]'hop), rest0) := by
        unfold decInstr
        split
        · simp_all
        · split
          · simp_all
          · simp_all <;> omega
      rw [e] at h
      obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
      subst hx; subst hr
      refine ⟨?_, trivial⟩
      show op :: rest0 = (Instr.prim (Prim.all[op]'hop)).enc ++ rest0
      show op :: rest0 = [(Prim.all[op]'hop).code] ++ rest0
      simp only [prim_code_get op hop, List.singleton_append]
    · have hcases : op = 0x80 ∨ op = 0x81 ∨ op = 0x82 ∨ op = 0x83 ∨
          op = 0x84 ∨ op = 0x85 ∨ op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨
          op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c ∨
          (op ≠ 0x80 ∧ op ≠ 0x81 ∧ op ≠ 0x82 ∧ op ≠ 0x83 ∧ op ≠ 0x84 ∧
          op ≠ 0x85 ∧ op ≠ 0x86 ∧ op ≠ 0x87 ∧ op ≠ 0x88 ∧ op ≠ 0x89 ∧
          op ≠ 0x8a ∧ op ≠ 0x8b ∧ op ≠ 0x8c) := by omega
      rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl |
        rfl | rfl | rfl | rfl | rfl | rfl |
        ⟨h80, h81, h82, h83, h84, h85, h86, h87, h88, h89, h8a, h8b,
          h8c⟩
      · have e : decInstr (0x80 :: rest0) =
            (rdLE rest0 8).map (fun y => (.lit y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x81 :: rest0) =
            (rdLE rest0 4).map (fun y => (.str y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x82 :: rest0) =
            (rdLE rest0 4).map (fun y => (.pick y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x83 :: rest0) = some (.tor, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x84 :: rest0) = some (.fromr, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x85 :: rest0) = some (.rfetch, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x86 :: rest0) =
            (rdLE rest0 4).map (fun y => (.locget y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x87 :: rest0) =
            (rdLE rest0 4).map (fun y => (.locset y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x88 :: rest0) =
            (rdLE rest0 4).map (fun y => (.jmp y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x89 :: rest0) =
            (rdLE rest0 4).map (fun y => (.jz y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8a :: rest0) =
            (rdLE rest0 4).map (fun y => (.jnz y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8b :: rest0) =
            (rdLE rest0 4).map (fun y => (.call y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8c :: rest0) = some (.ret, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · rw [decInstr_none_of hop h80 h81 h82 h83 h84 h85 h86 h87 h88
          h89 h8a h8b h8c] at h
        simp at h

/-- Inversion of `decMany`: the consumed bytes are the flat
encodings of the items, the count matches, and every item
satisfies the per-item predicate. -/
theorem decMany_inv {α : Type} {f : List Nat → Option (α × List Nat)}
    {e : α → List Nat} {P : α → Prop}
    (hf : ∀ {bs rest} {x : α}, f bs = some (x, rest) →
      (∀ b ∈ bs, b < 256) → bs = e x ++ rest ∧ P x) :
    ∀ {n bs rest} {xs : List α}, decMany f n bs = some (xs, rest) →
      (∀ b ∈ bs, b < 256) →
      bs = (xs.flatMap e) ++ rest ∧ xs.length = n ∧ ∀ x ∈ xs, P x := by
  intro n
  induction n with
  | zero =>
    intro bs rest xs h hb
    rw [show decMany f 0 bs = some ([], bs) from rfl] at h
    obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
    subst hx; subst hr
    exact ⟨rfl, rfl, fun x hx => by simp at hx⟩
  | succ n ih =>
    intro bs rest xs h hb
    have hstep : decMany f (n + 1) bs =
        (f bs).bind (fun p =>
          (decMany f n p.2).bind (fun q => some (p.1 :: q.1, q.2))) := rfl
    rw [hstep, Option.bind_eq_some_iff] at h
    obtain ⟨⟨x, bs1⟩, h1, h2⟩ := h
    rw [Option.bind_eq_some_iff] at h2
    obtain ⟨⟨xs', bs2⟩, h3, h4⟩ := h2
    have h3' : decMany f n bs1 = some (xs', bs2) := h3
    have hx : x :: xs' = xs :=
      congrArg Prod.fst (Option.some.inj h4)
    have hr : bs2 = rest :=
      congrArg Prod.snd (Option.some.inj h4)
    subst hx; subst hr
    obtain ⟨he1, hPx⟩ := hf h1 hb
    have hb1 : ∀ b ∈ bs1, b < 256 := fun b hbmem =>
      hb b (he1.symm ▸ List.mem_append_right _ hbmem)
    obtain ⟨he2, hlen, hP⟩ := ih h3' hb1
    refine ⟨?_, by simp [hlen], ?_⟩
    · calc bs = e x ++ bs1 := he1
        _ = e x ++ ((xs'.flatMap e) ++ bs2) := by rw [he2]
        _ = ((x :: xs').flatMap e) ++ bs2 := by
          rw [List.flatMap_cons, List.append_assoc]
    · intro z hz
      simp only [List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact hPx
      · exact hP z hz

/-- The per-annotation byte bounds. -/
def AnnBound (x : Nat × Height) : Prop :=
  x.1 < 256 ^ 4 ∧ x.2.d < 256 ^ 4 ∧ x.2.a < 256 ^ 4

/-- Inversion of `decAnn`. -/
theorem decAnn_inv {bs rest : List Nat} {x : Nat × Height}
    (h : decAnn bs = some (x, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = (leBytes x.1 4 ++ leBytes x.2.d 4 ++ leBytes x.2.a 4) ++
      rest ∧ AnnBound x := by
  have hstep : decAnn bs =
      (rdLE bs 4).bind (fun p =>
        (rdLE p.2 4).bind (fun q =>
          (rdLE q.2 4).bind
            (fun r => some ((p.1, ⟨q.1, r.1⟩), r.2)))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨i, bs1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨d, bs2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨a, bs3⟩, h3, h4⟩ := h
  have h2' : rdLE bs1 4 = some (d, bs2) := h2
  have h3' : rdLE bs2 4 = some (a, bs3) := h3
  have hx : (i, (⟨d, a⟩ : Height)) = x :=
    congrArg Prod.fst (Option.some.inj h4)
  have hr : bs3 = rest :=
    congrArg Prod.snd (Option.some.inj h4)
  subst hr
  obtain ⟨he1, hbi, -⟩ := rdLE_inv h1 hb
  have hb1 : ∀ b ∈ bs1, b < 256 := fun b hbmem =>
    hb b (he1.symm ▸ List.mem_append_right _ hbmem)
  obtain ⟨he2, hbd, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ bs2, b < 256 := fun b hbmem =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hbmem)
  obtain ⟨he3, hba, -⟩ := rdLE_inv h3' hb2
  subst hx
  refine ⟨?_, hbi, hbd, hba⟩
  calc bs = leBytes i 4 ++ bs1 := he1
    _ = leBytes i 4 ++ (leBytes d 4 ++ bs2) := by rw [he2]
    _ = leBytes i 4 ++ (leBytes d 4 ++ (leBytes a 4 ++ bs3)) := by
      rw [he3]
    _ = (leBytes i 4 ++ leBytes d 4 ++ leBytes a 4) ++ bs3 := by
      simp only [List.append_assoc]

/-- Inversion of `decData`. -/
theorem decData_inv {bs rest d : List Nat}
    (h : decData bs = some (d, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = (leBytes d.length 4 ++ d) ++ rest ∧ d.length < 256 ^ 4 := by
  have hstep : decData bs =
      (rdLE bs 4).bind (fun p =>
        if p.2.length < p.1 then none
        else some (p.2.take p.1, p.2.drop p.1)) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨n, bs1⟩, h1, h⟩ := h
  by_cases hlt : bs1.length < n
  · rw [if_pos hlt] at h
    exact absurd h (by simp)
  · rw [if_neg hlt] at h
    have hd : bs1.take n = d := congrArg Prod.fst (Option.some.inj h)
    have hr : bs1.drop n = rest := congrArg Prod.snd (Option.some.inj h)
    obtain ⟨he1, hbn, -⟩ := rdLE_inv h1 hb
    have hdlen : d.length = n := by
      rw [← hd, List.length_take]
      omega
    refine ⟨?_, by rw [hdlen]; exact hbn⟩
    calc bs = leBytes n 4 ++ bs1 := he1
      _ = leBytes n 4 ++ (bs1.take n ++ bs1.drop n) := by
        rw [List.take_append_drop]
      _ = (leBytes d.length 4 ++ d) ++ rest := by
        rw [hdlen, ← hd, ← hr]
        simp only [List.append_assoc]

/-- The per-word byte bounds. -/
def WordBound (w : Word) : Prop :=
  w.ins < 256 ^ 4 ∧ w.outs < 256 ^ 4 ∧ w.locals < 256 ^ 4 ∧
    w.code.length < 256 ^ 4 ∧ w.ann.length < 256 ^ 4 ∧
    (∀ x ∈ w.code, InstrBound x) ∧ (∀ x ∈ w.ann, AnnBound x)

/-- Inversion of `decWord`. -/
theorem decWord_inv {bs rest : List Nat} {w : Word}
    (h : decWord bs = some (w, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = w.enc ++ rest ∧ WordBound w := by
  have hstep : decWord bs =
      (rdLE bs 4).bind (fun p1 =>
        (rdLE p1.2 4).bind (fun p2 =>
          (rdLE p2.2 4).bind (fun p3 =>
            (rdLE p3.2 4).bind (fun p4 =>
              (rdLE p4.2 4).bind (fun p5 =>
                (decMany decInstr p4.1 p5.2).bind (fun p6 =>
                  (decMany decAnn p5.1 p6.2).bind (fun p7 =>
                    some (⟨p1.1, p2.1, p3.1, p6.1, p7.1⟩,
                      p7.2)))))))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨vi, bs1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨vo, bs2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨vl, bs3⟩, h3, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nc, bs4⟩, h4, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨na, bs5⟩, h5, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨code, bs6⟩, h6, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨ann, bs7⟩, h7, h8⟩ := h
  have h2' : rdLE bs1 4 = some (vo, bs2) := h2
  have h3' : rdLE bs2 4 = some (vl, bs3) := h3
  have h4' : rdLE bs3 4 = some (nc, bs4) := h4
  have h5' : rdLE bs4 4 = some (na, bs5) := h5
  have h6' : decMany decInstr nc bs5 = some (code, bs6) := h6
  have h7' : decMany decAnn na bs6 = some (ann, bs7) := h7
  have hw : (⟨vi, vo, vl, code, ann⟩ : Word) = w :=
    congrArg Prod.fst (Option.some.inj h8)
  have hr : bs7 = rest := congrArg Prod.snd (Option.some.inj h8)
  subst hr; subst hw
  obtain ⟨he1, hbi, -⟩ := rdLE_inv h1 hb
  have hb1 : ∀ b ∈ bs1, b < 256 := fun b hm =>
    hb b (he1.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he2, hbo, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ bs2, b < 256 := fun b hm =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he3, hbl, -⟩ := rdLE_inv h3' hb2
  have hb3 : ∀ b ∈ bs3, b < 256 := fun b hm =>
    hb2 b (he3.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he4, hbnc, -⟩ := rdLE_inv h4' hb3
  have hb4 : ∀ b ∈ bs4, b < 256 := fun b hm =>
    hb3 b (he4.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he5, hbna, -⟩ := rdLE_inv h5' hb4
  have hb5 : ∀ b ∈ bs5, b < 256 := fun b hm =>
    hb4 b (he5.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he6, hclen, hcP⟩ :=
    decMany_inv (f := decInstr) (e := Instr.enc) (P := InstrBound)
      (fun hh hb' => decInstr_inv hh hb') h6' hb5
  have hb6 : ∀ b ∈ bs6, b < 256 := fun b hm =>
    hb5 b (he6.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he7, halen, haP⟩ :=
    decMany_inv (f := decAnn)
      (e := fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4) (P := AnnBound)
      (fun hh hb' => decAnn_inv hh hb') h7' hb6
  have hwenc : (⟨vi, vo, vl, code, ann⟩ : Word).enc =
      leBytes vi 4 ++ leBytes vo 4 ++ leBytes vl 4 ++
        leBytes code.length 4 ++ leBytes ann.length 4 ++
        (code.flatMap Instr.enc) ++
        (ann.flatMap fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
          leBytes x.2.a 4) := rfl
  refine ⟨?_, hbi, hbo, hbl,
    by show code.length < 256 ^ 4; rw [hclen]; exact hbnc,
    by show ann.length < 256 ^ 4; rw [halen]; exact hbna, hcP, haP⟩
  rw [hwenc, he1, he2, he3, he4, he5, he6, he7, hclen, halen]
  simp only [List.append_assoc]

/-- Inversion of `decode`: a decoded file is exactly the encoding
of its program, all header, word, code, annotation and data
fields being byte-bounded as their widths force. -/
theorem decode_inv {f : List Nat} {pr : Prog}
    (h : decode f = some pr) (hb : ∀ b ∈ f, b < 256) :
    f = pr.enc ∧ pr.main < 256 ^ 4 ∧ pr.words.length < 256 ^ 4 ∧
      pr.data.length < 256 ^ 4 ∧ (∀ w ∈ pr.words, WordBound w) ∧
      (∀ d ∈ pr.data, d.length < 256 ^ 4) := by
  have hmagic : ∃ t, f = 0x53 :: 0x50 :: 0x52 :: 0x31 :: t := by
    unfold Sporos.decode at h
    split at h
    · exact ⟨_, rfl⟩
    · simp at h
  obtain ⟨t, rfl⟩ := hmagic
  have hbt : ∀ b ∈ t, b < 256 := fun b hm =>
    hb b (by simp [hm])
  have hstep : Sporos.decode (0x53 :: 0x50 :: 0x52 :: 0x31 :: t) =
      (rdLE t 4).bind (fun p1 =>
        (rdLE p1.2 4).bind (fun p2 =>
          (rdLE p2.2 4).bind (fun p3 =>
            (decMany decData p3.1 p3.2).bind (fun p4 =>
              (decMany decWord p2.1 p4.2).bind (fun p5 =>
                if p5.2.isEmpty = true then
                  some (⟨p5.1, p4.1, p1.1⟩ : Prog)
                else none))))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨mn, t1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nw, t2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nd, t3⟩, h3, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨data, t4⟩, h4, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨words, t5⟩, h5, h6⟩ := h
  have h2' : rdLE t1 4 = some (nw, t2) := h2
  have h3' : rdLE t2 4 = some (nd, t3) := h3
  have h4' : decMany decData nd t3 = some (data, t4) := h4
  have h5' : decMany decWord nw t4 = some (words, t5) := h5
  have h6' : (if t5.isEmpty = true then
      some (⟨words, data, mn⟩ : Prog) else none) = some pr := h6
  have ht5e : t5.isEmpty = true := by
    cases hbe : t5.isEmpty with
    | true => rfl
    | false =>
      rw [if_neg (by simp [hbe])] at h6'
      exact absurd h6' (by simp)
  rw [if_pos ht5e] at h6'
  have hpr : (⟨words, data, mn⟩ : Prog) = pr := Option.some.inj h6'
  have ht5 : t5 = [] := by
    cases t5 with
    | nil => rfl
    | cons a l => simp at ht5e
  obtain ⟨he1, hbmn, -⟩ := rdLE_inv h1 hbt
  have hb1 : ∀ b ∈ t1, b < 256 := fun b hm =>
    hbt b (he1.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he2, hbnw, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ t2, b < 256 := fun b hm =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he3, hbnd, -⟩ := rdLE_inv h3' hb2
  have hb3 : ∀ b ∈ t3, b < 256 := fun b hm =>
    hb2 b (he3.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he4, hdlen, hdP⟩ :=
    decMany_inv (f := decData)
      (e := fun d => leBytes d.length 4 ++ d)
      (P := fun d => d.length < 256 ^ 4)
      (fun hh hb' => decData_inv hh hb') h4' hb3
  have hb4 : ∀ b ∈ t4, b < 256 := fun b hm =>
    hb3 b (he4.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he5, hwlen, hwP⟩ :=
    decMany_inv (f := decWord) (e := Word.enc) (P := WordBound)
      (fun hh hb' => decWord_inv hh hb') h5' hb4
  have henc : (⟨words, data, mn⟩ : Prog).enc =
      [0x53, 0x50, 0x52, 0x31] ++ leBytes mn 4 ++
        leBytes words.length 4 ++ leBytes data.length 4 ++
        (data.flatMap fun d => leBytes d.length 4 ++ d) ++
        words.flatMap Word.enc := rfl
  subst hpr
  refine ⟨?_, hbmn, by rw [hwlen]; exact hbnw,
    by rw [hdlen]; exact hbnd, hwP, hdP⟩
  rw [henc, he1, he2, he3, he4, he5, ht5, hwlen, hdlen]
  simp only [List.append_assoc, List.append_nil, List.cons_append,
    List.nil_append]

/-- The value of a `u32` field below its bound, read at `W`. -/
theorem leVal4_W {v : Nat} (h : v < 256 ^ 4) :
    leVal (leBytes v 4) % W = v := by
  rw [leVal_leBytes, Nat.mod_eq_of_lt h,
    Nat.mod_eq_of_lt (Nat.lt_trans h (by unfold W; decide))]

/-- The value of a `u32` field, read at `2^32`. -/
theorem leVal4_32 (v : Nat) :
    leVal (leBytes v 4) % 2 ^ 32 = v % 2 ^ 32 := by
  rw [leVal_leBytes]
  show v % 2 ^ 32 % 2 ^ 32 = v % 2 ^ 32
  exact Nat.mod_mod v (2 ^ 32)

/-- Head of a `leBytes` field. -/
theorem take4_leBytes (v : Nat) (r : List Nat) :
    (leBytes v 4 ++ r).take 4 = leBytes v 4 := by
  rw [List.take_append_of_le_length (by simp [leBytes_length]),
    List.take_of_length_le (by simp [leBytes_length])]

/-- Tail after a `leBytes` field. -/
theorem drop4_leBytes (v : Nat) (r : List Nat) :
    (leBytes v 4 ++ r).drop 4 = r := by
  rw [show (4 : Nat) = (leBytes v 4).length by simp [leBytes_length]]
  exact List.drop_left

/-- A slice inside the first of two appended lists. -/
theorem seg_slice_append {l r : List Nat} {o k : Nat}
    (h : o + k ≤ l.length) :
    ((l ++ r).drop o).take k = (l.drop o).take k := by
  rw [List.drop_append_of_le_length (by omega),
    List.take_append_of_le_length]
  rw [List.length_drop]
  omega

/-- The `i`-th segment of a flatMap with constant segment
length. -/
theorem flatSeg_take {α : Type} (f : α → List Nat) (k : Nat) :
    ∀ (l : List α) (i : Nat) (hi : i < l.length),
    (∀ x ∈ l, (f x).length = k) →
    ((l.flatMap f).drop (k * i)).take k = f (l[i]'hi) := by
  intro l
  induction l with
  | nil => intro i hi _; simp at hi
  | cons a l ih =>
    intro i hi hlen
    cases i with
    | zero =>
      show (f a ++ l.flatMap f).take k = f a
      rw [List.take_append_of_le_length (by
          have hh := hlen a (.head _); omega),
        List.take_of_length_le (by
          have hh := hlen a (.head _); omega)]
    | succ i =>
      have hi' : i < l.length := by
        have h := hi
        simp only [List.length_cons] at h
        omega
      have step : k * (i + 1) = k + k * i := by
        rw [Nat.left_distrib, Nat.mul_one, Nat.add_comm (k * i) k]
      rw [step, ← List.drop_drop]
      have hda : ((a :: l).flatMap f).drop k = l.flatMap f := by
        show (f a ++ l.flatMap f).drop k = l.flatMap f
        rw [← hlen a (.head _)]
        exact List.drop_left
      rw [hda, List.getElem_cons_succ]
      exact ih i hi' (fun x hx => hlen x (.tail _ hx))

/-- Encoded code length matches `codeBytes`. -/
theorem codeBytes_flatMap : ∀ (code : List Instr),
    (code.flatMap Instr.enc).length = codeBytes code
  | [] => rfl
  | x :: xs => by
    calc ((x :: xs).flatMap Instr.enc).length
        = (x.enc ++ xs.flatMap Instr.enc).length := by
          rw [List.flatMap_cons]
      _ = x.enc.length + codeBytes xs := by
          rw [List.length_append, codeBytes_flatMap xs]
      _ = codeBytes (x :: xs) := rfl

/-- `CodeSeq` from one read of the whole encoding. -/
theorem CodeSeq_of_rdN : ∀ (code : List Instr) {m : Mem} {c : Nat},
    m.rdN c (codeBytes code) = some (code.flatMap Instr.enc) →
    CodeSeq m c code := by
  intro code
  induction code with
  | nil => intro m c _; trivial
  | cons x xs ih =>
    intro m c h
    have hh : m.rdN c (x.enc.length + codeBytes xs) =
        some ((x :: xs).flatMap Instr.enc) := h
    have h1 : m.rdN c x.enc.length = some x.enc := by
      have hok1 : 0 + x.enc.length ≤ x.enc.length + codeBytes xs := by
        omega
      have hs := rdN_slice hh hok1
      simp only [Nat.add_zero, List.drop_zero] at hs
      rw [hs, List.flatMap_cons, List.take_left]
    have h2 : m.rdN (c + x.enc.length) (codeBytes xs) =
        some (xs.flatMap Instr.enc) := by
      have hok2 : x.enc.length + codeBytes xs ≤
          x.enc.length + codeBytes xs := Nat.le_refl _
      have hs := rdN_slice hh hok2
      have e : (xs.flatMap Instr.enc).take (codeBytes xs) =
          xs.flatMap Instr.enc := by
        rw [← codeBytes_flatMap xs]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, List.flatMap_cons, List.drop_left, e]
    exact ⟨h1, ih h2⟩

/-- Encoded annotation length. -/
theorem annflat_length : ∀ (ann : List (Nat × Height)),
    (ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
      leBytes x.2.a 4)).length = 12 * ann.length
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons]
    simp only [List.length_append, leBytes_length,
      List.length_cons, annflat_length l]
    omega

/-- `AnnLayB` from one read of the whole annotation array. -/
theorem AnnLayB_of_rdN {m : Mem} {p : Nat}
    {ann : List (Nat × Height)}
    (h : m.rdN p (12 * ann.length) =
      some (ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4)))
    (hb : ∀ x ∈ ann, AnnBound x) : AnnLayB m p ann := by
  intro i hi
  have hseglen : ∀ x ∈ ann,
      ((fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4) x).length = 12 := fun x _ => by
        simp only [List.length_append, leBytes_length]
  have hseg : m.rdN (p + 12 * i) 12 =
      some (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4) := by
    have hs := rdN_slice h (o := 12 * i) (k := 12) (by
      have : i + 1 ≤ ann.length := hi
      have e : 12 * i + 12 = 12 * (i + 1) := by omega
      omega)
    rw [hs, flatSeg_take _ 12 ann i hi hseglen]
  have hbnd := hb ann[i] (List.getElem_mem hi)
  obtain ⟨hb1, hb2, hb3⟩ := hbnd
  have hW : ann[i].1 < W :=
    Nat.lt_trans hb1 (by unfold W; decide)
  have f0 : m.rdN (p + 12 * i) 4 = some (leBytes ann[i].1 4) := by
    have hs := rdN_slice hseg (o := 0) (k := 4) (by omega)
    simp only [Nat.add_zero, List.drop_zero] at hs
    rw [hs]
    exact congrArg some (take4_leBytes _ _)
  have e1 : (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 4 =
      leBytes ann[i].2.d 4 ++ leBytes ann[i].2.a 4 := by
    rw [List.append_assoc]
    exact drop4_leBytes _ _
  have f1 : m.rdN (p + 12 * i + 4) 4 = some (leBytes ann[i].2.d 4) := by
    have hs := rdN_slice hseg (o := 4) (k := 4) (by omega)
    rw [hs, e1]
    exact congrArg some (take4_leBytes _ _)
  have f2 : m.rdN (p + 12 * i + 8) 4 = some (leBytes ann[i].2.a 4) := by
    have hs := rdN_slice hseg (o := 8) (k := 4) (by omega)
    have e : (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 8 = leBytes ann[i].2.a 4 := by
      show ((leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 4).drop 4 = leBytes ann[i].2.a 4
      rw [e1]
      exact drop4_leBytes _ _
    rw [hs, e]
    exact congrArg some
      (List.take_of_length_le (by simp [leBytes_length]))
  refine ⟨?_, ?_, ?_, hW⟩
  · rw [f0]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb1]
  · rw [f1]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb2]
  · rw [f2]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb3]

/-- Encoded word length. -/
theorem wordEnc_length (wd : Word) :
    wd.enc.length = 20 + codeBytes wd.code + 12 * wd.ann.length := by
  show (leBytes wd.ins 4 ++ leBytes wd.outs 4 ++
    leBytes wd.locals 4 ++ leBytes wd.code.length 4 ++
    leBytes wd.ann.length 4 ++ (wd.code.flatMap Instr.enc) ++
    (wd.ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
      leBytes x.2.a 4))).length =
    20 + codeBytes wd.code + 12 * wd.ann.length
  simp only [List.length_append, leBytes_length, codeBytes_flatMap,
    annflat_length]

/-- Encoded words length matches `wordsBytes`. -/
theorem wordsBytes_flatMap : ∀ (wds : List Word),
    (wds.flatMap Word.enc).length = wordsBytes wds
  | [] => rfl
  | wd :: wds => by
    show ((wd :: wds).flatMap Word.enc).length =
      (20 + codeBytes wd.code + 12 * wd.ann.length) + wordsBytes wds
    rw [List.flatMap_cons, List.length_append, wordEnc_length,
      wordsBytes_flatMap wds]

/-- A word encoding, right-associated. -/
theorem wordEnc_form (wd : Word) : wd.enc =
    leBytes wd.ins 4 ++ (leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
      (leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
        ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4))))))) := by
  unfold Word.enc
  simp only [List.append_assoc]

/-- The five header fields of a word encoding, as the parser
reads them from its first 20 bytes. -/
theorem word_fields (wd : Word) (hb : WordBound wd) :
    leVal ((wd.enc.take 20).take 4) % W = wd.ins ∧
    leVal (((wd.enc.take 20).drop 4).take 4) % W = wd.outs ∧
    leVal (((wd.enc.take 20).drop 8).take 4) % W = wd.locals ∧
    leVal (((wd.enc.take 20).drop 12).take 4) % W = wd.code.length ∧
    leVal (((wd.enc.take 20).drop 16).take 4) % W = wd.ann.length := by
  have hf := wordEnc_form wd
  have d4 : wd.enc.drop 4 =
      leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
        (leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
          ((wd.code.flatMap Instr.enc) ++
            (wd.ann.flatMap (fun x => leBytes x.1 4 ++
              leBytes x.2.d 4 ++ leBytes x.2.a 4)))))) := by
    rw [hf]; exact drop4_leBytes _ _
  have d8 : wd.enc.drop 8 =
      leBytes wd.locals 4 ++ (leBytes wd.code.length 4 ++
        (leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4))))) := by
    have e : wd.enc.drop 8 = (wd.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : wd.enc.drop 12 =
      leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
        ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4)))) := by
    have e : wd.enc.drop 12 = (wd.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have d16 : wd.enc.drop 16 =
      leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
        (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4))) := by
    have e : wd.enc.drop 16 = (wd.enc.drop 12).drop 4 :=
      (List.drop_drop (i := 4) (j := 12)).symm
    rw [e, d12]; exact drop4_leBytes _ _
  have tk : ∀ (o : Nat), o ≤ 16 →
      ((wd.enc.take 20).drop o).take 4 = (wd.enc.drop o).take 4 := by
    intro o ho
    rw [List.drop_take, List.take_take, Nat.min_eq_left (by omega)]
  obtain ⟨hb1, hb2, hb3, hb4, hb5, -, -⟩ := hb
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [List.take_take, Nat.min_eq_left (by decide : 4 ≤ 20), hf,
      take4_leBytes]
    exact leVal4_W hb1
  · rw [tk 4 (by decide), d4, take4_leBytes]
    exact leVal4_W hb2
  · rw [tk 8 (by decide), d8, take4_leBytes]
    exact leVal4_W hb3
  · rw [tk 12 (by decide), d12, take4_leBytes]
    exact leVal4_W hb4
  · rw [tk 16 (by decide), d16, take4_leBytes]
    exact leVal4_W hb5

/-- Past the five header fields of a word encoding sit the code
and annotation encodings. -/
theorem wordEnc_drop20 (wd : Word) :
    wd.enc.drop 20 = (wd.code.flatMap Instr.enc) ++
      (wd.ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4)) := by
  have hf := wordEnc_form wd
  have e : wd.enc.drop 20 = (wd.enc.drop 16).drop 4 :=
    (List.drop_drop (i := 4) (j := 16)).symm
  have d16 : wd.enc.drop 16 =
      leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
        (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4))) := by
    have e4 : wd.enc.drop 16 = (wd.enc.drop 12).drop 4 :=
      (List.drop_drop (i := 4) (j := 12)).symm
    have d12 : wd.enc.drop 12 =
        leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
          ((wd.code.flatMap Instr.enc) ++
            (wd.ann.flatMap (fun x => leBytes x.1 4 ++
              leBytes x.2.d 4 ++ leBytes x.2.a 4)))) := by
      have e3 : wd.enc.drop 12 = (wd.enc.drop 8).drop 4 :=
        (List.drop_drop (i := 4) (j := 8)).symm
      have d8 : wd.enc.drop 8 =
          leBytes wd.locals 4 ++ (leBytes wd.code.length 4 ++
            (leBytes wd.ann.length 4 ++
              ((wd.code.flatMap Instr.enc) ++
                (wd.ann.flatMap (fun x => leBytes x.1 4 ++
                  leBytes x.2.d 4 ++ leBytes x.2.a 4))))) := by
        have e2 : wd.enc.drop 8 = (wd.enc.drop 4).drop 4 :=
          (List.drop_drop (i := 4) (j := 4)).symm
        have d4 : wd.enc.drop 4 =
            leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
              (leBytes wd.code.length 4 ++
                (leBytes wd.ann.length 4 ++
                  ((wd.code.flatMap Instr.enc) ++
                    (wd.ann.flatMap (fun x => leBytes x.1 4 ++
                      leBytes x.2.d 4 ++ leBytes x.2.a 4)))))) := by
          rw [hf]; exact drop4_leBytes _ _
        rw [e2, d4]; exact drop4_leBytes _ _
      rw [e3, d8]; exact drop4_leBytes _ _
    rw [e4, d12]; exact drop4_leBytes _ _
  rw [e, d16]
  exact drop4_leBytes _ _

/-- `WordsLay` from one read of the whole words encoding. -/
theorem WordsLay_of_rdN : ∀ (wds : List Word) {m : Mem} {c : Nat},
    m.rdN c (wordsBytes wds) = some (wds.flatMap Word.enc) →
    (∀ w ∈ wds, WordBound w) → WordsLay m c wds := by
  intro wds
  induction wds with
  | nil => intro m c _ _; trivial
  | cons wd wds ih =>
    intro m c h hb
    have hbwd := hb wd (.head _)
    have hh : m.rdN c
        ((20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds) =
        some (wd.enc ++ wds.flatMap Word.enc) := by
      rwa [List.flatMap_cons] at h
    have h20 : m.rdN c 20 = some (wd.enc.take 20) := by
      have hok : 0 + 20 ≤ (20 + codeBytes wd.code +
          12 * wd.ann.length) + wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      simp only [Nat.add_zero, List.drop_zero] at hs
      rw [hs, List.take_append_of_le_length (by
        rw [wordEnc_length]; omega)]
    have hcode : m.rdN (c + 20) (codeBytes wd.code) =
        some (wd.code.flatMap Instr.enc) := by
      have hok : 20 + codeBytes wd.code ≤ (20 + codeBytes wd.code +
          12 * wd.ann.length) + wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      have e : (wd.code.flatMap Instr.enc).take (codeBytes wd.code) =
          wd.code.flatMap Instr.enc := by
        rw [← codeBytes_flatMap]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, seg_slice_append (l := wd.enc)
        (r := wds.flatMap Word.enc) (o := 20) (k := codeBytes wd.code)
        (by rw [wordEnc_length]; omega), wordEnc_drop20,
        List.take_append_of_le_length
          (by have hcc := codeBytes_flatMap wd.code; omega), e]
    have hann : m.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
        some (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4)) := by
      have hok : (20 + codeBytes wd.code) + 12 * wd.ann.length ≤
          (20 + codeBytes wd.code + 12 * wd.ann.length) +
            wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      have ec : c + (20 + codeBytes wd.code) =
          c + 20 + codeBytes wd.code := by omega
      rw [ec] at hs
      have edrop : (wd.enc ++ wds.flatMap Word.enc).drop
            (20 + codeBytes wd.code) =
          wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4) ++
            wds.flatMap Word.enc := by
        have e1 : (wd.enc ++ wds.flatMap Word.enc).drop
              (20 + codeBytes wd.code) =
            ((wd.enc ++ wds.flatMap Word.enc).drop 20).drop
              (codeBytes wd.code) := (List.drop_drop).symm
        have e2 : (wd.enc ++ wds.flatMap Word.enc).drop 20 =
            (wd.code.flatMap Instr.enc) ++
              ((wd.ann.flatMap (fun x => leBytes x.1 4 ++
                  leBytes x.2.d 4 ++ leBytes x.2.a 4)) ++
                wds.flatMap Word.enc) := by
          rw [List.drop_append_of_le_length (by
            rw [wordEnc_length]; omega), wordEnc_drop20,
            List.append_assoc]
        rw [e1, e2, ← codeBytes_flatMap]
        exact List.drop_left
      have e : (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4)).take
          (12 * wd.ann.length) =
          wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4) := by
        rw [← annflat_length]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, edrop,
        List.take_append_of_le_length
          (by have haa := annflat_length wd.ann; omega), e]
    have hrest : m.rdN
        (c + 20 + codeBytes wd.code + 12 * wd.ann.length)
        (wordsBytes wds) = some (wds.flatMap Word.enc) := by
      have hok : (20 + codeBytes wd.code + 12 * wd.ann.length) +
            wordsBytes wds ≤ (20 + codeBytes wd.code +
              12 * wd.ann.length) + wordsBytes wds := Nat.le_refl _
      have hs := rdN_slice hh hok
      have ec : c + (20 + codeBytes wd.code + 12 * wd.ann.length) =
          c + 20 + codeBytes wd.code + 12 * wd.ann.length := by omega
      rw [ec] at hs
      have e : (wds.flatMap Word.enc).take (wordsBytes wds) =
          wds.flatMap Word.enc := by
        rw [← wordsBytes_flatMap]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, ← wordEnc_length wd, List.drop_left, e]
    obtain ⟨f1, f2, f3, f4, f5⟩ := word_fields wd hbwd
    refine ⟨⟨wd.enc.take 20, h20, f1, f2, f3, f4, f5⟩,
      CodeSeq_of_rdN _ hcode,
      AnnLayB_of_rdN hann (fun x hx => hbwd.2.2.2.2.2.2 x hx),
      ih hrest (fun w hw => hb w (.tail _ hw))⟩

/-- Buffer bytes consumed by a data segment list. -/
def dataBytes : List (List Nat) → Nat
  | [] => 0
  | d :: ds => (4 + d.length) + dataBytes ds

/-- Encoded data length matches `dataBytes`. -/
theorem dataBytes_flat : ∀ (ds : List (List Nat)),
    (ds.flatMap (fun d => leBytes d.length 4 ++ d)).length =
      dataBytes ds
  | [] => rfl
  | d :: ds => by
    show ((d :: ds).flatMap (fun d => leBytes d.length 4 ++
        d)).length = (4 + d.length) + dataBytes ds
    rw [List.flatMap_cons, List.length_append, List.length_append,
      leBytes_length, dataBytes_flat ds]

/-- `DataSeq` from one read of the whole data encoding. -/
theorem DataSeq_of_rdN : ∀ (ds : List (List Nat)) {m : Mem}
    {c : Nat},
    m.rdN c (dataBytes ds) =
      some (ds.flatMap (fun d => leBytes d.length 4 ++ d)) →
    (∀ d ∈ ds, d.length < 256 ^ 4) →
    DataSeq m c (ds.map List.length) (c + dataBytes ds) := by
  intro ds
  induction ds with
  | nil =>
    intro m c _ _
    show DataSeq m c [] (c + 0)
    rw [Nat.add_zero]
    exact DataSeq.nil c
  | cons d ds ih =>
    intro m c h hb
    have hdlen := hb d (.head _)
    have hh : m.rdN c ((4 + d.length) + dataBytes ds) =
        some ((leBytes d.length 4 ++ d) ++
          ds.flatMap (fun d => leBytes d.length 4 ++ d)) := by
      rwa [List.flatMap_cons] at h
    have hd : rd32 m c = some d.length := by
      have hok : 0 + 4 ≤ (4 + d.length) + dataBytes ds := by omega
      have hs := rdN_slice hh hok
      simp only [Nat.add_zero, List.drop_zero] at hs
      show (m.rdN c 4).map leVal = some d.length
      rw [hs, List.take_append_of_le_length (by
          simp only [List.length_append, leBytes_length]; omega),
        take4_leBytes]
      simp [leVal_leBytes, Nat.mod_eq_of_lt hdlen]
    have hrest : m.rdN (c + 4 + d.length) (dataBytes ds) =
        some (ds.flatMap (fun d => leBytes d.length 4 ++ d)) := by
      have hok : (4 + d.length) + dataBytes ds ≤
          (4 + d.length) + dataBytes ds := Nat.le_refl _
      have hs := rdN_slice hh hok
      have hdrop : ((leBytes d.length 4 ++ d) ++
            ds.flatMap (fun d => leBytes d.length 4 ++ d)).drop
          (4 + d.length) =
          ds.flatMap (fun d => leBytes d.length 4 ++ d) := by
        rw [show (4 + d.length) =
          (leBytes d.length 4 ++ d).length by
            simp only [List.length_append, leBytes_length]]
        exact List.drop_left
      have e : (ds.flatMap (fun d => leBytes d.length 4 ++
            d)).take (dataBytes ds) =
          ds.flatMap (fun d => leBytes d.length 4 ++ d) := by
        rw [← dataBytes_flat]
        exact List.take_of_length_le (Nat.le_refl _)
      have ec : c + (4 + d.length) = c + 4 + d.length := by omega
      rw [ec] at hs
      rw [hs, hdrop, e]
    show DataSeq m c (d.length :: ds.map List.length)
      (c + ((4 + d.length) + dataBytes ds))
    rw [show c + ((4 + d.length) + dataBytes ds) =
      (c + 4 + d.length) + dataBytes ds by omega]
    exact DataSeq.cons hd
      (ih hrest (fun x hx => hb x (.tail _ hx)))

/-- A program encoding, right-associated. -/
theorem enc_form (pr : Prog) : pr.enc =
    [0x53, 0x50, 0x52, 0x31] ++ (leBytes pr.main 4 ++
      (leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))))) := by
  unfold Prog.enc
  simp only [List.append_assoc]

/-- Encoded program length. -/
theorem enc_length (pr : Prog) :
    pr.enc.length = 16 + dataBytes pr.data + wordsBytes pr.words := by
  rw [enc_form]
  have hmagiclen : ([0x53, 0x50, 0x52, 0x31] : List Nat).length = 4 :=
    rfl
  simp only [List.length_append, leBytes_length, hmagiclen,
    dataBytes_flat, wordsBytes_flatMap]
  omega

/-- Past the sixteen header bytes of a program encoding sit the
data and words encodings. -/
theorem enc_drop16 (pr : Prog) :
    pr.enc.drop 16 =
      (pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
        (pr.words.flatMap Word.enc) := by
  have hf := enc_form pr
  have d4 : pr.enc.drop 4 =
      leBytes pr.main 4 ++ (leBytes pr.words.length 4 ++
        (leBytes pr.data.length 4 ++
          ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
            (pr.words.flatMap Word.enc)))) := by
    rw [hf, show (4 : Nat) =
      ([0x53, 0x50, 0x52, 0x31] : List Nat).length from rfl]
    exact List.drop_left
  have d8 : pr.enc.drop 8 =
      leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))) := by
    have e : pr.enc.drop 8 = (pr.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : pr.enc.drop 12 =
      leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc)) := by
    have e : pr.enc.drop 12 = (pr.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have e : pr.enc.drop 16 = (pr.enc.drop 12).drop 4 :=
    (List.drop_drop (i := 4) (j := 12)).symm
  rw [e, d12]
  exact drop4_leBytes _ _

/-- The header values of a program encoding, as the parser reads
them from its first 20 bytes, plus the drop-4/8/12 forms the
composition needs for the field loads. -/
theorem enc_header_vals (pr : Prog) :
    leVal ((pr.enc.take 20).take 4) % 2 ^ 32 = 827478099 ∧
    leVal (((pr.enc.take 20).drop 4).take 4) % 2 ^ 32 =
      pr.main % 2 ^ 32 ∧
    leVal (((pr.enc.take 20).drop 8).take 4) % 2 ^ 32 =
      pr.words.length % 2 ^ 32 ∧
    leVal (((pr.enc.take 20).drop 12).take 4) % 2 ^ 32 =
      pr.data.length % 2 ^ 32 := by
  have hf := enc_form pr
  have d4 : pr.enc.drop 4 =
      leBytes pr.main 4 ++ (leBytes pr.words.length 4 ++
        (leBytes pr.data.length 4 ++
          ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
            (pr.words.flatMap Word.enc)))) := by
    rw [hf, show (4 : Nat) =
      ([0x53, 0x50, 0x52, 0x31] : List Nat).length from rfl]
    exact List.drop_left
  have d8 : pr.enc.drop 8 =
      leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))) := by
    have e : pr.enc.drop 8 = (pr.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : pr.enc.drop 12 =
      leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc)) := by
    have e : pr.enc.drop 12 = (pr.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have tk : ∀ (o : Nat), o ≤ 16 →
      ((pr.enc.take 20).drop o).take 4 = (pr.enc.drop o).take 4 := by
    intro o ho
    rw [List.drop_take, List.take_take, Nat.min_eq_left (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [List.take_take, Nat.min_eq_left (by decide : 4 ≤ 20), hf,
      show (4 : Nat) = ([0x53, 0x50, 0x52, 0x31] : List Nat).length
        from rfl, List.take_left]
    decide
  · rw [tk 4 (by decide), d4, take4_leBytes]
    exact leVal4_32 _
  · rw [tk 8 (by decide), d8, take4_leBytes]
    exact leVal4_32 _
  · rw [tk 12 (by decide), d12, take4_leBytes]
    exact leVal4_32 _

/-- The first twenty bytes of a program in memory. -/
theorem enc_head20 {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc)
    (h20 : 20 ≤ pr.enc.length) :
    m.rdN B 20 = some (pr.enc.take 20) := by
  have hok : 0 + 20 ≤ pr.enc.length := h20
  have hs := rdN_slice h hok
  simpa using hs

/-- The data section of a program in memory. -/
theorem enc_data_slice {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc) :
    m.rdN (B + 16) (dataBytes pr.data) =
      some (pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) := by
  have hok : 16 + dataBytes pr.data ≤ pr.enc.length := by
    rw [enc_length]; omega
  have hs := rdN_slice h hok
  have e : (pr.data.flatMap (fun d => leBytes d.length 4 ++
        d)).take (dataBytes pr.data) =
      pr.data.flatMap (fun d => leBytes d.length 4 ++ d) := by
    rw [← dataBytes_flat]
    exact List.take_of_length_le (Nat.le_refl _)
  rw [hs, enc_drop16,
    List.take_append_of_le_length (by
      have hcc := dataBytes_flat pr.data; omega), e]

/-- The words section of a program in memory. -/
theorem enc_words_slice {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc) :
    m.rdN (B + 16 + dataBytes pr.data) (wordsBytes pr.words) =
      some (pr.words.flatMap Word.enc) := by
  have hok : (16 + dataBytes pr.data) + wordsBytes pr.words ≤
      pr.enc.length := by
    rw [enc_length]; omega
  have hs := rdN_slice h hok
  have ec : B + (16 + dataBytes pr.data) =
      B + 16 + dataBytes pr.data := by omega
  rw [ec] at hs
  have edrop : pr.enc.drop (16 + dataBytes pr.data) =
      pr.words.flatMap Word.enc := by
    have e1 : pr.enc.drop (16 + dataBytes pr.data) =
        (pr.enc.drop 16).drop (dataBytes pr.data) :=
      (List.drop_drop (i := dataBytes pr.data) (j := 16)).symm
    rw [e1, enc_drop16, ← dataBytes_flat]
    exact List.drop_left
  have e : (pr.words.flatMap Word.enc).take (wordsBytes pr.words) =
      pr.words.flatMap Word.enc := by
    rw [← wordsBytes_flatMap]
    exact List.take_of_length_le (Nat.le_refl _)
  rw [hs, edrop, e]

/-- From `checkProg`, the main word exists and takes no inputs. -/
theorem checkProg_main {pr : Prog} (h : Sporos.checkProg pr = true) :
    ∃ hm : pr.main < pr.words.length, pr.words[pr.main].ins = 0 := by
  unfold Sporos.checkProg at h
  rw [Bool.and_eq_true] at h
  obtain ⟨-, hm⟩ := h
  cases hwm : pr.words[pr.main]? with
  | none => rw [hwm] at hm; simp at hm
  | some m =>
    rw [hwm] at hm
    have hins : m.ins = 0 := beq_iff_eq.mp hm
    obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.mp hwm
    exact ⟨hlt, by rw [hget]; exact hins⟩

/-- Twenty bytes per word at least. -/
theorem wordsBytes_ge : ∀ (wds : List Word),
    20 * wds.length ≤ wordsBytes wds
  | [] => by simp [wordsBytes]
  | wd :: wds => by
    show 20 * (wd :: wds).length ≤
      (20 + codeBytes wd.code + 12 * wd.ann.length) + wordsBytes wds
    have ih := wordsBytes_ge wds
    simp only [List.length_cons]
    have e : 20 * (wds.length + 1) = 20 + 20 * wds.length := by omega
    omega

/-- The oracle after one consumed choice. -/
theorem next_oracle_tail {k : Kern} {c : Choice} {rest : List Choice}
    (h : k.oracle = c :: rest) : (Kern.next k).2.oracle = rest := by
  cases ho : k.oracle with
  | nil => simp [ho] at h
  | cons c0 rest0 =>
    have hnx : Kern.next k = (some c0, { k with oracle := rest0 }) := by
      simp [Kern.next, ho]
    rw [ho] at h
    obtain ⟨-, hr⟩ := List.cons.inj h
    rw [hnx]
    exact hr

/-- A byte write that succeeds in a memory also succeeds after
appending more regions. -/
theorem wr_isSome_append_left {m t : Mem} {a u : Nat}
    (h : (m.wr a u).isSome) : ((m ++ t).wr a u).isSome := by
  obtain ⟨i, r, hf, hrw⟩ := wr_isSome_iff.mp h
  exact wr_isSome_iff.mpr
    ⟨i, r, find_append_left m t a 0 hf, hrw⟩

/-- `StackRoom` transports down whole slots when every byte
write transports. -/
theorem stackRoom_transport_of_wr {s s' : X86Int.St}
    {k k' t sp : Nat}
    (hr4 : s.reg 4 = some sp) (hr4' : s'.reg 4 = some (sp - 8 * t))
    (hroom : StackRoom s k) (hk : t + k' ≤ k)
    (hge : 8 * k ≤ sp) (hspW : sp < W)
    (htr : ∀ a u, (s.mem.wr a u).isSome → (s'.mem.wr a u).isSome) :
    StackRoom s' k' := by
  intro sp'' h4'' i hi v
  have hsp'' : sp'' = sp - 8 * t := by
    rw [hr4'] at h4''
    exact (Option.some.inj h4'').symm
  have hmod : ∀ d : Nat, d ≤ sp → (sp + W - d) % W = sp - d := by
    intro d hd
    have hh : sp + W - d = (sp - d) + W := by omega
    rw [hh, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hA : (sp'' + W - 8 * (i + 1)) % W = sp - 8 * (t + i + 1) := by
    have ee : sp - 8 * t + W - 8 * (i + 1) =
        sp + W - 8 * (t + i + 1) := by omega
    rw [hsp'', ee, hmod (8 * (t + i + 1)) (by omega)]
  have hw64 := hroom sp hr4 (t + i) (by omega) v
  rw [hmod (8 * (t + i + 1)) (by omega)] at hw64
  rw [hA]
  exact wr64_isSome_of_forall_wr (fun j hj u =>
    htr _ _ (wr_isSome_of_wr64 hw64 hj u))

set_option maxHeartbeats 0 in
/-- The parser front half on an accepted file: from `loaded`
(`base + 0x134`, the state `loader_init_to_loaded` produces)
through the header checks, the data skip and the size guard,
to just after the words-table `mmap` (`base + 0x22d`). -/
theorem parser_front_ok {f path argv0 : List Nat} {top A : Nat}
    {pr : Prog} {s0 : X86Int.St}
    (hdec : Sporos.decode f = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfB : ∀ b ∈ f, b < 256)
    (hAmod : A % page = 0) (hApos : 0 < A)
    (hAbound : A + stage1Mf f ≤ 2 ^ 47)
    (hfreeA : (stage1Init f path argv0 [] [] top).mem.free A
      (stage1Mf f) = true)
    (hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base)
    (hstkhi : (X86Env.initStack top [argv0, path]).1.hi = top)
    (htop47 : top ≤ 2 ^ 47)
    (hrip0 : s0.rip = base + 0x134)
    (hr3 : s0.reg 3 = some A)
    (hr13 : s0.reg 13 = some f.length)
    (hr4 : s0.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24))
    (hr7 : ∃ v, s0.reg 7 = some v)
    (hr9 : ∃ v, s0.reg 9 = some v)
    (horacle : s0.k.oracle =
      [.at (A + arenaFileBytes f),
       .at (A + (arenaFileBytes f + arenaTableBytes f))])
    (hmem0 : ∃ c1 c2 buf', s0.mem = [c1, c2, buf'] ∧
      c1.base = base ∧ c1.hi = base + stage1Code.length ∧
      c2.base = (X86Env.initStack top [argv0, path]).1.base ∧
      c2.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
      buf'.base = A ∧ (∃ cap', buf'.hi = A + cap' ∧
        cap' ≤ arenaFileBytes f) ∧ buf'.rw = true ∧ c2.rw = true)
    (hrdN : s0.mem.rdN A f.length = some f)
    (hlen16 : s0.regs.length = 16)
    (hroom10 : StackRoom s0 10) :
    ∃ (k : Nat) (s' : X86Int.St) (m2 : Mem),
      run stage1Im k (Out.ok s0) = Out.ok s' ∧
      s'.rip = base + 0x22d ∧
      s'.reg 14 = some (A + arenaFileBytes f) ∧
      s'.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24 - 8) ∧
      s'.reg 3 = some A ∧ s'.reg 12 = some pr.main ∧
      s'.reg 13 = some f.length ∧
      s'.reg 15 = some (A + f.length) ∧
      s'.mem = m2 ++
        [⟨A + arenaFileBytes f,
          List.replicate (X86Env.pages (64 * pr.words.length)) 0,
          true⟩] ∧
      wshape m2 = wshape s0.mem ∧
      s'.mem.rdN A f.length = some f ∧
      rd64 s'.mem ((X86Env.initStack top [argv0, path]).2 + 24 - 8) =
        some ((A + 16 + dataBytes pr.data) % W) ∧
      s'.k.oracle =
        [.at (A + (arenaFileBytes f + arenaTableBytes f))] ∧
      s'.k.fds = s0.k.fds ∧ s'.k.files = s0.k.files ∧
      s'.k.stdin = s0.k.stdin ∧ s'.k.out = s0.k.out ∧
      (∃ v, s'.reg 7 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      StackRoom s' 1 ∧ s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
        (X86Env.initStack top [argv0, path]).2 + 24 ≤ x →
        m2.rdN x k = s0.mem.rdN x k) ∧
      (s0.mem.rdN base stage1Code.length = some stage1Code →
        s'.mem.rdN base stage1Code.length = some stage1Code) := by
  obtain ⟨hfenc, hbmn, hbnw, hbnd, hwB, hdB⟩ := decode_inv hdec hfB
  obtain ⟨hmn, -⟩ := checkProg_main hchk
  have hnw0 : pr.words.length ≠ 0 := by omega
  have hLenc : f.length =
      16 + dataBytes pr.data + wordsBytes pr.words := by
    rw [hfenc]; exact enc_length pr
  have hwb20 : 20 ≤ wordsBytes pr.words := by
    have h := wordsBytes_ge pr.words
    omega
  have hL20 : 20 ≤ f.length := by omega
  have hL16 : 16 ≤ f.length := by omega
  have hLe : f.length ≤ stage1Mf f := by
    have h1 : f.length ≤ arenaFileBytes f := by
      unfold arenaFileBytes
      have h2 := pages_ge (65536 + 2 * f.length)
      omega
    unfold stage1Mf
    omega
  have hAe : A + f.length < W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  have hrdNenc : s0.mem.rdN A pr.enc.length = some pr.enc := by
    rwa [← hfenc]
  have h20 : s0.mem.rdN A 20 = some (pr.enc.take 20) :=
    enc_head20 hrdNenc (by rw [← hfenc]; omega)
  obtain ⟨hmagic, hvmain, hvnw, hvnd⟩ := enc_header_vals pr
  have hmainv : leVal (((pr.enc.take 20).drop 4).take 4) % 2 ^ 32 =
      pr.main := by
    rw [hvmain]
    exact Nat.mod_eq_of_lt hbmn
  have hnwv : leVal (((pr.enc.take 20).drop 8).take 4) % 2 ^ 32 =
      pr.words.length := by
    rw [hvnw]
    exact Nat.mod_eq_of_lt hbnw
  have hndv : leVal (((pr.enc.take 20).drop 12).take 4) % 2 ^ 32 =
      pr.data.length := by
    rw [hvnd]
    exact Nat.mod_eq_of_lt hbnd
  obtain ⟨c1, c2, buf', hmem0eq, hc1b, hc1hi, hc2b, hc2hi, hbufb,
    ⟨cap', hbufhi, hbufbound⟩, hbufrw, hc2rw⟩ := hmem0
  -- SP0 arithmetic
  have hsp0eq : (X86Env.initStack top [argv0, path]).2 =
      c2.base + 4096 := by
    have h1 := initStack_sp top argv0 path
    rw [← hc2b] at h1
    have hsr : X86Env.stackRoom = 4096 := rfl
    rw [hsr] at h1
    exact h1
  have hSP0ge : c2.base + 4096 ≤
      (X86Env.initStack top [argv0, path]).2 + 24 := by omega
  have hSPv : (X86Env.initStack top [argv0, path]).2 + 24 =
      c2.base + 4120 := by omega
  have hSPle : (X86Env.initStack top [argv0, path]).2 + 24 ≤
      c2.hi := by
    have hlen := initStack_data_length top argv0 path
    have hh : c2.hi = c2.base +
        (X86Env.initStack top [argv0, path]).1.data.length := by
      have e : (X86Env.initStack top [argv0, path]).1.hi =
          (X86Env.initStack top [argv0, path]).1.base +
            (X86Env.initStack top [argv0, path]).1.data.length :=
        rfl
      rw [hc2hi, e, ← hc2b]
    have hsr : X86Env.stackRoom = 4096 := rfl
    have hpg := pages_ge
      (56 + ((argv0.length + 1) + (path.length + 1)) + 1)
    omega
  have hSP0W : (X86Env.initStack top [argv0, path]).2 + 24 < W := by
    have hW : W = 2 ^ 64 := rfl
    have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
    omega
  -- initial free facts per region
  have hfree0 := hfreeA
  rw [stage1Init_mem] at hfree0
  unfold X86Env.Mem.free at hfree0
  rw [List.all_eq_true] at hfree0
  have hfreecode := hfree0 ⟨base, stage1Code, false⟩ (by simp)
  have hfreestk := hfree0 (X86Env.initStack top [argv0, path]).1
    (by simp)
  rw [Bool.or_eq_true] at hfreecode hfreestk
  simp only [decide_eq_true_eq] at hfreecode hfreestk
  have hcodehi : (⟨base, stage1Code, false⟩ : Region).hi =
      base + stage1Code.length := by simp only [X86Env.Region.hi]
  have hfreecode2 : A + stage1Mf f ≤ base ∨
      base + stage1Code.length ≤ A := by
    have hcodeb : (⟨base, stage1Code, false⟩ : Region).base = base :=
      rfl
    rwa [hcodehi] at hfreecode
  have hpgnw : X86Env.pages (64 * pr.words.length) ≤
      arenaTableBytes f := by
    unfold arenaTableBytes
    apply pages_mono
    have h1 := wordsBytes_ge pr.words
    omega
  have hTMf : (A + arenaFileBytes f) +
        X86Env.pages (64 * pr.words.length) ≤ A + stage1Mf f := by
    unfold stage1Mf
    omega
  have hAframe : ∀ x k : Nat, A ≤ x → x + k ≤ A + f.length →
      x + k ≤ (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
        (X86Env.initStack top [argv0, path]).2 + 24 ≤ x := by
    intro x k hx hxk
    have hstkhi2 : (X86Env.initStack top [argv0, path]).1.hi = top :=
      hstkhi
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · have h2 : (X86Env.initStack top [argv0, path]).1.hi ≤ A := h
      exact Or.inr (by
        have h3 : (X86Env.initStack top [argv0, path]).2 + 24 ≤
            (X86Env.initStack top [argv0, path]).1.hi := by
          rw [← hc2hi]
          exact hSPle
        omega)
  -- header_a
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r15, hs1r6, hs1fr, hs1mem,
    hs1len, hs1k⟩ :=
    parser_header_a hrip0 hr3 hr13 hAe hL16 h20 hmagic hlen16
  have hs1r4 : s1.reg 4 = s0.reg 4 :=
    hs1fr 4 (by decide) (by decide) (by decide)
  have h20_1 : s1.mem.rdN A 20 = some (pr.enc.take 20) := by
    rw [hs1mem]; exact h20
  -- header_b
  obtain ⟨n2, s2, hrun2, hs2rip, hs2r12, hs2r10, hs2r1, hs2r6, hs2fr,
    hs2mem, hs2len, hs2k⟩ :=
    parser_header_b hs1rip hs1r6 hAe hL16 h20_1
      (by rw [hnwv]; exact hnw0) hs1len
  rw [hmainv] at hs2r12
  rw [hnwv] at hs2r10
  rw [hndv] at hs2r1
  have hs2r15 : s2.reg 15 = some (A + f.length) := by
    rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs1r15
  have hs2r4 : s2.reg 4 = s0.reg 4 := by
    rw [hs2fr 4 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs1r4
  have hs2r3 : s2.reg 3 = some A := by
    rw [hs2fr 3 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide),
      hs1fr 3 (by decide) (by decide) (by decide)]
    exact hr3
  have hs2r13 : s2.reg 13 = some f.length := by
    rw [hs2fr 13 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide),
      hs1fr 13 (by decide) (by decide) (by decide)]
    exact hr13
  -- data loop
  have hrdN2 : s2.mem.rdN A pr.enc.length = some pr.enc := by
    rw [hs2mem, hs1mem]; exact hrdNenc
  have hdataseq : DataSeq s2.mem (A + 16) (pr.data.map List.length)
      (A + 16 + dataBytes pr.data) :=
    DataSeq_of_rdN _ (enc_data_slice hrdN2) hdB
  obtain ⟨n3, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3r15, hs3fr, hs3mem,
    hs3len, hs3k⟩ :=
    parser_data_loop_ok hdataseq hs2rip
      (by rw [hs2r1, List.length_map]) hs2r6 hs2r15 (by omega) hAe
      (by
        have hW : W = 2 ^ 64 := rfl
        rw [List.length_map]
        omega) hs2len
  have hs3r10 : s3.reg 10 = some pr.words.length := by
    rw [hs3fr 10 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r10
  have hs3r4 : s3.reg 4 = s0.reg 4 := by
    rw [hs3fr 4 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r4
  -- guard
  have hguard : 20 * pr.words.length ≤
      (A + f.length) - (A + 16 + dataBytes pr.data) := by
    have h1 := wordsBytes_ge pr.words
    omega
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    parser_guard_ok hs3rip hs3r10 hs3r6 hs3r15 hbnw (by omega) hAe
      hguard hs3len
  have hmem4 : s4.mem = s0.mem := by
    rw [hs4mem, hs3mem, hs2mem, hs1mem]
  have hs4r4 : s4.reg 4 = s0.reg 4 := by
    rw [hs4fr 4 (by decide) (by decide)]
    exact hs3r4
  have hs4r6 : s4.reg 6 = some (A + 16 + dataBytes pr.data) := by
    rw [hs4fr 6 (by decide) (by decide)]
    exact hs3r6
  have hs4r10 : s4.reg 10 = some pr.words.length := by
    rw [hs4fr 10 (by decide) (by decide)]
    exact hs3r10
  have hs4r3 : s4.reg 3 = some A := by
    rw [hs4fr 3 (by decide) (by decide),
      hs3fr 3 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r3
  have hs4r12 : s4.reg 12 = some pr.main := by
    rw [hs4fr 12 (by decide) (by decide),
      hs3fr 12 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r12
  have hs4r13 : s4.reg 13 = some f.length := by
    rw [hs4fr 13 (by decide) (by decide),
      hs3fr 13 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r13
  have hs4r15 : s4.reg 15 = some (A + f.length) := by
    rw [hs4fr 15 (by decide) (by decide)]
    exact hs3r15
  have hk4 : s4.k = s0.k := by rw [hs4k, hs3k, hs2k, hs1k]
  have hroom4 : StackRoom s4 2 :=
    stackRoom_mono
      (StackRoom_of_mem hmem4 (by rw [hs4r4]) hroom10) (by decide)
  have hs4r7 : ∃ v, s4.reg 7 = some v := by
    obtain ⟨v, hv⟩ := hr7
    exact ⟨v, by
      rw [hs4fr 7 (by decide) (by decide),
        hs3fr 7 (by decide) (by decide) (by decide) (by decide),
        hs2fr 7 (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
        hs1fr 7 (by decide) (by decide) (by decide)]
      exact hv⟩
  have hs4r9 : ∃ v, s4.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by
      rw [hs4fr 9 (by decide) (by decide),
        hs3fr 9 (by decide) (by decide) (by decide) (by decide),
        hs2fr 9 (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
        hs1fr 9 (by decide) (by decide) (by decide)]
      exact hv⟩
  -- free table interval
  have hfreeT : s4.mem.free (A + arenaFileBytes f)
      (X86Env.pages (64 * pr.words.length)) = true := by
    rw [hmem4, hmem0eq]
    unfold X86Env.Mem.free
    rw [List.all_eq_true]
    intro r hr
    rw [Bool.or_eq_true]
    have hr' : c1 = r ∨ c2 = r ∨ buf' = r := by
      rcases (show r = c1 ∨ r = c2 ∨ r = buf' by simpa using hr) with
        h | h | h
      · exact Or.inl h.symm
      · exact Or.inr (Or.inl h.symm)
      · exact Or.inr (Or.inr h.symm)
    rcases hr' with rfl | rfl | rfl
    · rcases hfreecode2 with h | h
      · exact Or.inl (by
          rw [hc1b, decide_eq_true_eq]
          exact Nat.le_trans hTMf h)
      · exact Or.inr (by
          rw [hc1hi, decide_eq_true_eq]
          exact Nat.le_trans h (Nat.le_add_right A _))
    · rcases hfreestk with h | h
      · exact Or.inl (by
          rw [hc2b, decide_eq_true_eq]
          exact Nat.le_trans hTMf h)
      · exact Or.inr (by
          rw [hc2hi, decide_eq_true_eq]
          exact Nat.le_trans h (Nat.le_add_right A _))
    · exact Or.inr (by
        rw [decide_eq_true_eq, hbufhi]
        exact Nat.add_le_add_left hbufbound A)
  have hTpage : (A + arenaFileBytes f) % page = 0 := by
    have h1 : arenaFileBytes f % page = 0 := by
      unfold arenaFileBytes X86Env.pages
      exact Nat.mul_mod_left _ _
    rw [Nat.add_mod, hAmod, h1]
    simp
  have hor4 : s4.k.oracle =
      [.at (A + arenaFileBytes f),
       .at (A + (arenaFileBytes f + arenaTableBytes f))] := by
    rw [hk4]; exact horacle
  -- mmap1
  obtain ⟨s5, m2, hrun5, hs5rip, hs5r14, hs5r4, hs5mem, hm2fr,
    hrd64cur, hs5fr, hs5len, hs5k, hw2⟩ :=
    parser_mmap1_ok hs4rip hs4r6 hs4r10
      (by rw [hs4r4]; exact hr4) hs4r7 hs4r9 hroom4 (by omega) hSP0W
      hbnw hnw0 (by rw [hor4]; rfl) hTpage (by omega)
      (by omega) hfreeT hs4len
  have hrunA : run stage1Im (n1 + n2) (Out.ok s0) = Out.ok s2 := by
    rw [run_add, hrun1, hrun2]
  have hrunB : run stage1Im (n1 + n2 + n3) (Out.ok s0) = Out.ok s3 := by
    rw [show n1 + n2 + n3 = (n1 + n2) + n3 from rfl, run_add, hrunA,
      hrun3]
  have hrunC : run stage1Im (n1 + n2 + n3 + 11) (Out.ok s0) =
      Out.ok s4 := by
    rw [show n1 + n2 + n3 + 11 = (n1 + n2 + n3) + 11 from rfl, run_add,
      hrunB, hrun4]
  have hrunTot : run stage1Im (n1 + n2 + n3 + 11 + 20) (Out.ok s0) =
      Out.ok s5 := by
    rw [show n1 + n2 + n3 + 11 + 20 = (n1 + n2 + n3 + 11) + 20 from rfl,
      run_add, hrunC, hrun5]
  have hrdN5 : s5.mem.rdN A f.length = some f := by
    have hm2rd : m2.rdN A f.length = some f := by
      rw [hm2fr A f.length (hAframe A f.length (Nat.le_refl _)
        (by omega)), hmem4]
      exact hrdN
    rw [hs5mem]
    exact rdN_append_left hm2rd
  have hs5k2 : s5.k = (Kern.next s0.k).2 := by rw [hs5k, hk4]
  have hnx2 : (Kern.next s0.k).2 =
      { s0.k with oracle :=
        [.at (A + (arenaFileBytes f + arenaTableBytes f))] } := by
    cases ho : s0.k.oracle with
    | nil => simp [ho] at horacle
    | cons c0 rest0 =>
      have h2 : Kern.next s0.k =
          (some c0, { s0.k with oracle := rest0 }) := by
        simp [Kern.next, ho]
      rw [ho] at horacle
      obtain ⟨-, hr⟩ := List.cons.inj horacle
      rw [h2, hr]
  have hs5oracle : s5.k.oracle =
      [.at (A + (arenaFileBytes f + arenaTableBytes f))] := by
    have h1 : s5.k.oracle = (Kern.next s0.k).2.oracle := by
      rw [hs5k2]
    rw [h1, hnx2]
  have hroom5 : StackRoom s5 1 := by
    have hr4' : s5.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24 - 8 * 1) :=
      hs5r4
    refine stackRoom_transport_of_wr hr4 hr4' hroom10 (by decide)
      (by omega) hSP0W (fun a u hwr => ?_)
    have hwr4 : (s4.mem.wr a u).isSome := by rwa [hmem4]
    have h1 := wr_some_of_wshape hw2 hwr4
    rw [hs5mem]
    exact wr_isSome_append_left h1
  exact ⟨n1 + n2 + n3 + 11 + 20, s5, m2, hrunTot, hs5rip, hs5r14,
    hs5r4, by
      rw [hs5fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)]
      exact hs4r3, by
      rw [hs5fr 12 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)]
      exact hs4r12, by
      rw [hs5fr 13 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)]
      exact hs4r13, by
      rw [hs5fr 15 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide)]
      exact hs4r15, hs5mem, (by rw [← hmem4]; exact hw2), hrdN5, hrd64cur, hs5oracle, by
      rw [hs5k2, hnx2], by rw [hs5k2, hnx2], by rw [hs5k2, hnx2], by
      rw [hs5k2, hnx2], by
      obtain ⟨v, hv⟩ := hr7
      exact run_reg_some_ne11 _ (by decide) hv hrunTot, by
      obtain ⟨v, hv⟩ := hr9
      exact run_reg_some_ne11 _ (by decide) hv hrunTot, hroom5, hs5len,
    (fun x k hk => by rw [hm2fr x k hk, hmem4]), (fun h => by
      rw [hs5mem]
      apply rdN_append_left
      rw [hm2fr base _ (Or.inl (by omega)), hmem4]
      exact h)⟩

/-- Transport a writable `find` hit across equal write shapes. -/
theorem find_rw_of_wshape {m m' : Mem} {a : Nat}
    (hsh : wshape m' = wshape m)
    (h : ∃ i r, Mem.find m a 0 = some (i, r) ∧ r.rw = true) :
    ∃ i r, Mem.find m' a 0 = some (i, r) ∧ r.rw = true := by
  obtain ⟨i, r, hf, hrw⟩ := h
  have hw := find_wshape m m' hsh a 0
  obtain ⟨i', r', hf', -, hrw'⟩ :=
    (by simpa [hf] using hw : ∃ i' r',
      Mem.find m' a 0 = some (i', r') ∧ i' = i ∧ r'.rw = r.rw)
  exact ⟨i', r', hf', by rw [hrw']; exact hrw⟩

/-- `find` hits the second of three regions. -/
theorem find_c2_of_three {c1 c2 buf : Region} {a : Nat}
    (hdisj : c1.hi ≤ c2.base) (h1 : c2.base ≤ a) (h2 : a < c2.hi) :
    Mem.find [c1, c2, buf] a 0 = some (1, c2) := by
  have hg : Mem.find [c1, c2, buf] a 0 =
      (if c1.base ≤ a ∧ a < c1.hi then some (0, c1)
        else if c2.base ≤ a ∧ a < c2.hi then some (1, c2)
          else Mem.find [buf] a 2) := rfl
  rw [hg, if_neg (by omega), if_pos ⟨h1, h2⟩]

/-- `StackRoom` from a write shape whose second region is a
writable stack disjoint from the first. -/
theorem stackRoom_of_wshape_c2 {s : X86Int.St}
    {c1 c2 buf : Region} {extra : Mem} {sp k : Nat}
    (hsh : wshape s.mem = wshape ([c1, c2, buf] ++ extra))
    (hr4 : s.reg 4 = some sp) (hrw : c2.rw = true)
    (hdisj : c1.hi ≤ c2.base)
    (hlo : c2.base + 8 * k ≤ sp) (hhi : sp ≤ c2.hi)
    (hspW : sp < W) :
    StackRoom s k := by
  intro sp' h4' i hi v
  have hsp' : sp' = sp := by
    rw [hr4] at h4'
    exact (Option.some.inj h4').symm
  rw [hsp']
  have hmod : (sp + W - 8 * (i + 1)) % W = sp - 8 * (i + 1) := by
    have hh : sp + W - 8 * (i + 1) = (sp - 8 * (i + 1)) + W := by
      omega
    rw [hh, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  rw [hmod]
  refine wr64_isSome_of_forall_wr (fun j hj u => ?_)
  have href : ∃ ii r,
      Mem.find ([c1, c2, buf] ++ extra) (sp - 8 * (i + 1) + j) 0 =
        some (ii, r) ∧ r.rw = true :=
    ⟨1, c2, find_append_left [c1, c2, buf] extra _ 0
      (find_c2_of_three hdisj (by omega) (by omega)), hrw⟩
  obtain ⟨ii, r, hf, hrw2⟩ := find_rw_of_wshape hsh href
  exact wr_isSome_iff.mpr ⟨ii, r, hf, hrw2⟩

/-- The `ins` slot of the `j`-th word in a laid-out table. -/
theorem TabLay_ins_at : ∀ {m : Mem} {B S tbl c : Nat}
    {wds : List Word}, TabLay m B S tbl c wds →
    ∀ (j : Nat) (hj : j < wds.length),
      rd64 m (tbl + 64 * j) = some wds[j].ins := by
  intro m B S tbl c wds h
  induction wds generalizing tbl c with
  | nil => intro j hj; simp at hj
  | cons wd wds ih =>
    intro j hj
    obtain ⟨h0, -, -, -, -, -, -, -, -, -, hrest⟩ := h
    cases j with
    | zero =>
      simpa using h0
    | succ j =>
      have hj' : j < wds.length := by
        simp only [List.length_cons] at hj; omega
      have h1 := ih hrest j hj'
      have e : tbl + 64 * (j + 1) = tbl + 64 + 64 * j := by omega
      rw [e, List.getElem_cons_succ]
      exact h1

/-- Only the first four bytes of an eight-byte little-endian
value matter modulo `2^32`. -/
theorem leVal_first4_mod {bs : List Nat} (h : bs.length = 8) :
    leVal bs % 2 ^ 32 = leVal (bs.take 4) % 2 ^ 32 := by
  have hsplit : bs = bs.take 4 ++ bs.drop 4 :=
    (List.take_append_drop 4 bs).symm
  have htlen : (bs.take 4).length = 4 := by
    rw [List.length_take, h]; omega
  have hcong : leVal bs % 2 ^ 32 =
      leVal (bs.take 4 ++ bs.drop 4) % 2 ^ 32 :=
    congrArg (fun x => leVal x % 2 ^ 32) hsplit
  rw [hcong, leVal_append, htlen]
  show (leVal (bs.take 4) + 256 ^ 4 * leVal (bs.drop 4)) % 2 ^ 32 =
    leVal (bs.take 4) % 2 ^ 32
  exact Nat.add_mul_mod_self_left _ _ _

/-- An address below a region is outside it. -/
theorem not_mem_region_of_lt {r : Region} {a : Nat}
    (h : a < r.base) : ¬ (r.base ≤ a ∧ a < r.hi) :=
  fun hcon => absurd hcon.1 (Nat.not_le.mpr h)

/-- An address at or past a region's end is outside it. -/
theorem not_mem_region_of_le {r : Region} {a : Nat}
    (h : r.hi ≤ a) : ¬ (r.base ≤ a ∧ a < r.hi) :=
  fun hcon => absurd hcon.2 (Nat.not_lt.mpr h)

/-- The word-count field bytes at offset eight of a program
encoding. -/
theorem enc_drop8_take4 (pr : Prog) :
    (pr.enc.drop 8).take 4 = leBytes pr.words.length 4 := by
  have hf := enc_form pr
  have d4 : pr.enc.drop 4 =
      leBytes pr.main 4 ++ (leBytes pr.words.length 4 ++
        (leBytes pr.data.length 4 ++
          ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
            (pr.words.flatMap Word.enc)))) := by
    rw [hf, show (4 : Nat) =
      ([0x53, 0x50, 0x52, 0x31] : List Nat).length from rfl]
    exact List.drop_left
  have d8 : pr.enc.drop 8 =
      leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))) := by
    have e : pr.enc.drop 8 = (pr.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  rw [d8]
  exact take4_leBytes _ _

set_option maxHeartbeats 0 in
/-- The parser back half on an accepted file: from just after
the words-table `mmap` (`base + 0x22d`, the state
`parser_front_ok` produces) through the scratch `mmap`, the
setup, the whole word loop and the trailing checks, to the
checker entry (`base + 0x452`) with the table laid out. -/
theorem parser_back_ok {f path argv0 : List Nat} {top A : Nat}
    {pr : Prog} {s : X86Int.St} {c1 c2 buf' : Region} {cap' : Nat}
    (hdec : Sporos.decode f = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfB : ∀ b ∈ f, b < 256)
    (hAmod : A % page = 0) (hApos : 0 < A)
    (hAbound : A + stage1Mf f ≤ 2 ^ 47)
    (hfreeA : (stage1Init f path argv0 [] [] top).mem.free A
      (stage1Mf f) = true)
    (hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base)
    (hstkhi : (X86Env.initStack top [argv0, path]).1.hi = top)
    (htop47 : top ≤ 2 ^ 47)
    (hrip : s.rip = base + 0x22d)
    (hr14 : s.reg 14 = some (A + arenaFileBytes f))
    (hr4 : s.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24 - 8))
    (hr3 : s.reg 3 = some A)
    (hr12 : s.reg 12 = some pr.main)
    (hr13 : s.reg 13 = some f.length)
    (hr15 : s.reg 15 = some (A + f.length))
    (hwshape : wshape s.mem =
      wshape (([c1, c2, buf'] ++
        [⟨A + arenaFileBytes f,
          List.replicate (X86Env.pages (64 * pr.words.length)) 0,
          true⟩])))
    (hc1b : c1.base = base)
    (hc1hi : c1.hi = base + stage1Code.length)
    (hc2b : c2.base = (X86Env.initStack top [argv0, path]).1.base)
    (hc2hi : c2.hi = (X86Env.initStack top [argv0, path]).1.hi)
    (hbufb : buf'.base = A) (hbufhi : buf'.hi = A + cap')
    (hbufbound : cap' ≤ arenaFileBytes f)
    (hc2rw : c2.rw = true)
    (hrdN : s.mem.rdN A f.length = some f)
    (hcur : rd64 s.mem
      ((X86Env.initStack top [argv0, path]).2 + 24 - 8) =
        some ((A + 16 + dataBytes pr.data) % W))
    (horacle : s.k.oracle =
      [.at (A + (arenaFileBytes f + arenaTableBytes f))])
    (hr7 : ∃ v, s.reg 7 = some v)
    (hr9 : ∃ v, s.reg 9 = some v)
    (hroom1 : StackRoom s 1)
    (hlen16 : s.regs.length = 16) :
    ∃ (k : Nat) (s' : X86Int.St),
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧
      s'.reg 14 = some (A + arenaFileBytes f) ∧
      s'.reg 13 = some pr.words.length ∧
      s'.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24 + 48) ∧
      TabLay s'.mem A
        (A + (arenaFileBytes f + arenaTableBytes f))
        (A + arenaFileBytes f)
        (A + 16 + dataBytes pr.data) pr.words ∧
      StackRoom s' 6 ∧ s'.regs.length = 16 ∧
      wshape s'.mem = wshape ([c1, c2, buf'] ++
        [⟨A + arenaFileBytes f,
          List.replicate (X86Env.pages (64 * pr.words.length)) 0,
          true⟩,
         ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩]) ∧
      s'.k = (Kern.next s.k).2 ∧
      (s.mem.rdN base stage1Code.length = some stage1Code →
        s'.mem.rdN base stage1Code.length = some stage1Code) := by
  obtain ⟨hfenc, hbmn, hbnw, hbnd, hwB, hdB⟩ := decode_inv hdec hfB
  obtain ⟨hmn, hins0⟩ := checkProg_main hchk
  have hnw0 : pr.words.length ≠ 0 := by omega
  have hLenc : f.length =
      16 + dataBytes pr.data + wordsBytes pr.words := by
    rw [hfenc]; exact enc_length pr
  have hwb20 : 20 ≤ wordsBytes pr.words := by
    have h := wordsBytes_ge pr.words
    omega
  have hL20 : 20 ≤ f.length := by omega
  have hLe : f.length ≤ stage1Mf f := by
    have h1 : f.length ≤ arenaFileBytes f := by
      unfold arenaFileBytes
      have h2 := pages_ge (65536 + 2 * f.length)
      omega
    unfold stage1Mf
    omega
  have hAe : A + f.length < W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  have hrdNenc : s.mem.rdN A pr.enc.length = some pr.enc := by
    rwa [← hfenc]
  have hdisj : c1.hi ≤ c2.base := by
    rw [hc1hi, hc2b]; exact hdisjInit
  have hsp0eq : (X86Env.initStack top [argv0, path]).2 =
      c2.base + 4096 := by
    have h1 := initStack_sp top argv0 path
    rw [← hc2b] at h1
    have hsr : X86Env.stackRoom = 4096 := rfl
    rw [hsr] at h1
    exact h1
  have hstklen : 8192 ≤
      (X86Env.initStack top [argv0, path]).1.data.length := by
    rw [initStack_data_length]
    have hp : 4096 ≤ X86Env.pages
        (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
      unfold X86Env.pages X86Env.page
      have h1 : 1 ≤ (56 + ((argv0.length + 1) +
          (path.length + 1)) + 1 + 4095) / 4096 :=
        Nat.div_pos (by omega) (by decide)
      omega
    have hsr : X86Env.stackRoom = 4096 := rfl
    omega
  have hhival : c2.hi = c2.base +
      (X86Env.initStack top [argv0, path]).1.data.length := by
    have e : (X86Env.initStack top [argv0, path]).1.hi =
        (X86Env.initStack top [argv0, path]).1.base +
          (X86Env.initStack top [argv0, path]).1.data.length :=
      rfl
    rw [hc2hi, e, ← hc2b]
  have hSP1top :
      (X86Env.initStack top [argv0, path]).2 + 24 - 8 + 56 ≤
        c2.hi := by omega
  have hSP1W :
      (X86Env.initStack top [argv0, path]).2 + 24 - 8 < W := by
    have hW : W = 2 ^ 64 := rfl
    have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
    omega
  -- initial free facts per region
  have hfree0 := hfreeA
  rw [stage1Init_mem] at hfree0
  unfold X86Env.Mem.free at hfree0
  rw [List.all_eq_true] at hfree0
  have hfreecode := hfree0 ⟨base, stage1Code, false⟩ (by simp)
  have hfreestk := hfree0 (X86Env.initStack top [argv0, path]).1
    (by simp)
  rw [Bool.or_eq_true] at hfreecode hfreestk
  simp only [decide_eq_true_eq] at hfreecode hfreestk
  have hfreecode2 : A + stage1Mf f ≤ base ∨
      base + stage1Code.length ≤ A := by
    have hcodehi : (⟨base, stage1Code, false⟩ : Region).hi =
        base + stage1Code.length := by
      simp only [X86Env.Region.hi]
    rwa [hcodehi] at hfreecode
  have hpgnw : X86Env.pages (64 * pr.words.length) ≤
      arenaTableBytes f := by
    unfold arenaTableBytes
    apply pages_mono
    have h1 := wordsBytes_ge pr.words
    omega
  have hTMf : (A + arenaFileBytes f) +
        X86Env.pages (64 * pr.words.length) ≤ A + stage1Mf f := by
    unfold stage1Mf
    omega
  have hSplus : (A + (arenaFileBytes f + arenaTableBytes f)) +
        X86Env.pages f.length = A + stage1Mf f := by
    unfold stage1Mf arenaScratchBytes
    omega
  have hAframe2 : ∀ x k : Nat, A ≤ x → x + k ≤ A + f.length →
      x + k ≤ (X86Env.initStack top [argv0, path]).2 + 24 - 8 - 8 ∨
        (X86Env.initStack top [argv0, path]).2 + 24 - 8 ≤ x := by
    intro x k hx hxk
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  -- page alignment of the scratch base
  have hFilePage : arenaFileBytes f % page = 0 := by
    unfold arenaFileBytes X86Env.pages
    exact Nat.mul_mod_left _ _
  have hTabPage : arenaTableBytes f % page = 0 := by
    unfold arenaTableBytes X86Env.pages
    exact Nat.mul_mod_left _ _
  have hSpage : (A + (arenaFileBytes f + arenaTableBytes f)) % page =
      0 := by
    have hsum : (arenaFileBytes f + arenaTableBytes f) % page = 0 := by
      rw [Nat.add_mod, hFilePage, hTabPage]
      simp
    rw [Nat.add_mod, hAmod, hsum]
    simp
  -- the scratch interval is free in the current memory shape
  have hfree_c1 : (A + (arenaFileBytes f + arenaTableBytes f)) +
        X86Env.pages f.length ≤ c1.base ∨
      c1.hi ≤ A + (arenaFileBytes f + arenaTableBytes f) := by
    rcases hfreecode2 with h | h
    · exact Or.inl (by rw [hc1b]; omega)
    · exact Or.inr (by rw [hc1hi]; omega)
  have hfree_c2 : (A + (arenaFileBytes f + arenaTableBytes f)) +
        X86Env.pages f.length ≤ c2.base ∨
      c2.hi ≤ A + (arenaFileBytes f + arenaTableBytes f) := by
    rcases hfreestk with h | h
    · exact Or.inl (by rw [hc2b]; omega)
    · exact Or.inr (by rw [hc2hi]; omega)
  have hfree_buf : (A + (arenaFileBytes f + arenaTableBytes f)) +
        X86Env.pages f.length ≤ buf'.base ∨
      buf'.hi ≤ A + (arenaFileBytes f + arenaTableBytes f) := by
    exact Or.inr (by rw [hbufhi]; omega)
  have hfree3 : Mem.free [c1, c2, buf']
      (A + (arenaFileBytes f + arenaTableBytes f))
      (X86Env.pages f.length) = true := by
    rw [Mem.free, List.all_eq_true]
    intro r hr
    have hr2 : r = c1 ∨ r = c2 ∨ r = buf' := by simpa using hr
    rcases hr2 with rfl | rfl | rfl
    · simp only [Bool.or_eq_true, decide_eq_true_eq]; exact hfree_c1
    · simp only [Bool.or_eq_true, decide_eq_true_eq]; exact hfree_c2
    · simp only [Bool.or_eq_true, decide_eq_true_eq]; exact hfree_buf
  have hfree4 : Mem.free (([c1, c2, buf'] ++
        [⟨A + arenaFileBytes f,
          List.replicate (X86Env.pages (64 * pr.words.length)) 0,
          true⟩]))
      (A + (arenaFileBytes f + arenaTableBytes f))
      (X86Env.pages f.length) = true :=
    free_append hfree3 (Or.inr (by
      simp only [X86Env.Region.hi, List.length_replicate]
      omega))
  have hfreeS : s.mem.free
      (A + (arenaFileBytes f + arenaTableBytes f))
      (X86Env.pages f.length) = true := by
    rw [free_of_wshape hwshape]; exact hfree4
  -- mmap2: the scratch region
  obtain ⟨s6, m1, hrun6, hs6rip, hs6r11, hs6r4, hs6mem, hm1fr,
    hs6fr, hs6len, hs6k, hm1sh⟩ :=
    parser_mmap2_ok hrip hr13 hr4 hr7 hr9 hroom1 (by omega) hSP1W
      (by omega) (by rw [horacle]; rfl) hSpage (by omega)
      (by omega) hfreeS hlen16
  have hs6r14 : s6.reg 14 = some (A + arenaFileBytes f) := by
    rw [hs6fr 14 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr14
  have hs6r3 : s6.reg 3 = some A := by
    rw [hs6fr 3 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr3
  have hs6r12 : s6.reg 12 = some pr.main := by
    rw [hs6fr 12 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr12
  have hs6r13 : s6.reg 13 = some f.length := by
    rw [hs6fr 13 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr13
  have hs6r15 : s6.reg 15 = some (A + f.length) := by
    rw [hs6fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr15
  have hrdN6 : s6.mem.rdN A f.length = some f := by
    have hm1rd : m1.rdN A f.length = some f := by
      rw [hm1fr A f.length (hAframe2 A f.length (Nat.le_refl _)
        (by omega))]
      exact hrdN
    rw [hs6mem]
    exact rdN_append_left hm1rd
  have hrdNenc6 : s6.mem.rdN A pr.enc.length = some pr.enc := by
    have h := hrdN6
    rwa [hfenc] at h
  have hcur6 : rd64 s6.mem
      ((X86Env.initStack top [argv0, path]).2 + 24 - 8) =
        some (A + 16 + dataBytes pr.data) := by
    have h1 : rd64 s6.mem
        ((X86Env.initStack top [argv0, path]).2 + 24 - 8) =
          some ((A + 16 + dataBytes pr.data) % W) := by
      have h' : (s.mem.rdN
          ((X86Env.initStack top [argv0, path]).2 + 24 - 8) 8).map
            leVal = some ((A + 16 + dataBytes pr.data) % W) := hcur
      obtain ⟨bs, hbs, hle⟩ := Option.map_eq_some_iff.mp h'
      have hm1bs : m1.rdN
          ((X86Env.initStack top [argv0, path]).2 + 24 - 8) 8 =
            some bs := by
        rw [hm1fr _ 8 (Or.inr (Nat.le_refl _))]; exact hbs
      have h6bs : s6.mem.rdN
          ((X86Env.initStack top [argv0, path]).2 + 24 - 8) 8 =
            some bs := by
        rw [hs6mem]; exact rdN_append_left hm1bs
      unfold rd64
      rw [h6bs]
      exact congrArg some hle
    rwa [Nat.mod_eq_of_lt (by omega)] at h1
  -- the nwords field read at B+8
  obtain ⟨v, hnw6, hvmod⟩ : ∃ v, rd64 s6.mem (A + 8) = some v ∧
      v % 2 ^ 32 = pr.words.length := by
    have hok : 8 + 8 ≤ pr.enc.length := by
      rw [enc_length]; omega
    have hs := rdN_slice hrdNenc6 hok
    refine ⟨leVal ((pr.enc.drop 8).take 8), ?_, ?_⟩
    · unfold rd64
      rw [hs]
      rfl
    · have hlen8 : ((pr.enc.drop 8).take 8).length = 8 := by
        rw [rdN_length hs]
      have h1 := leVal_first4_mod hlen8
      have htake : ((pr.enc.drop 8).take 8).take 4 =
          (pr.enc.drop 8).take 4 := by
        rw [List.take_take, Nat.min_eq_left (by decide)]
      rw [htake, enc_drop8_take4] at h1
      rw [h1, leVal_leBytes]
      have h256 : (256 : Nat) ^ 4 = 2 ^ 32 := by decide
      have hlt : pr.words.length < 2 ^ 32 := by
        rw [← h256]; exact hbnw
      rw [h256, Nat.mod_mod]
      exact Nat.mod_eq_of_lt hlt
  -- write shape of the post-mmap2 memory, for stack reads/room
  have hsh6 : wshape s6.mem = wshape ([c1, c2, buf'] ++
      [⟨A + arenaFileBytes f,
        List.replicate (X86Env.pages (64 * pr.words.length)) 0, true⟩,
       ⟨A + (arenaFileBytes f + arenaTableBytes f),
        List.replicate (X86Env.pages f.length) 0, true⟩]) := by
    have happ : ∀ (m : Mem) (r : Region),
        wshape (m ++ [r]) = wshape m ++ [(r.base, r.hi, r.rw)] := by
      intro m r
      simp [wshape]
    rw [hs6mem, happ, hm1sh, hwshape]
    simp [wshape, List.append_assoc]
  have hrdstk : ∀ a : Nat, c2.base ≤ a → a < c2.hi →
      (s6.mem.rd a).isSome := by
    intro a h1 h2
    apply rd_isSome_of_find
    have href : ∃ ii r,
        Mem.find ([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩,
           ⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩]) a 0 =
          some (ii, r) ∧ r.rw = true :=
      ⟨1, c2, find_append_left [c1, c2, buf'] _ a 0
        (find_c2_of_three hdisj h1 h2), hc2rw⟩
    obtain ⟨ii, r, hf, -⟩ := find_rw_of_wshape hsh6 href
    exact ⟨ii, r, hf⟩
  have hpop : ∀ off : Nat, off ≤ 48 →
      ∃ w, rd64 s6.mem
        ((X86Env.initStack top [argv0, path]).2 + 24 - 8 + off) =
          some w := by
    intro off hoff
    refine rd64_some_of_forall_rd (fun j hj => ?_)
    apply hrdstk <;> omega
  obtain ⟨w1, hp1⟩ := hpop 8 (by decide)
  obtain ⟨w2, hp2⟩ := hpop 16 (by decide)
  obtain ⟨w3, hp3⟩ := hpop 24 (by decide)
  obtain ⟨w4, hp4⟩ := hpop 32 (by decide)
  obtain ⟨w5, hp5⟩ := hpop 40 (by decide)
  obtain ⟨w6, hp6⟩ := hpop 48 (by decide)
  -- setup: restore cursor, reload nwords, make stack room
  obtain ⟨s7, hrun7, hs7rip, hs7r6, hs7r7, hs7r13, hs7r4, -, hs7mem,
    hs7fr, hs7len, hs7k⟩ :=
    parser_setup_ok hs6rip hs6r4 hs6r14 hs6r3 (by
      have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
      have hW : W = 2 ^ 64 := rfl
      omega) (by omega) hcur6 hnw6 hp1 hp2 hp3 hp4 hp5 hp6 hs6len
  rw [hvmod] at hs7r13
  have hs7r4' : s7.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24 + 48) := by
    have e : (X86Env.initStack top [argv0, path]).2 + 24 - 8 + 56 =
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 := by omega
    rwa [e] at hs7r4
  have hs7r14 : s7.reg 14 = some (A + arenaFileBytes f) := by
    rw [hs7fr 14 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hs6r14
  have hs7r3 : s7.reg 3 = some A := by
    rw [hs7fr 3 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hs6r3
  have hs7r11 : s7.reg 11 =
      some (A + (arenaFileBytes f + arenaTableBytes f)) := by
    rw [hs7fr 11 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hs6r11
  have hs7r12 : s7.reg 12 = some pr.main := by
    rw [hs7fr 12 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hs6r12
  have hs7r15 : s7.reg 15 = some (A + f.length) := by
    rw [hs7fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hs6r15
  -- word-loop entry
  obtain ⟨s8, hrun8, hs8rip, hs8r1, hs8fr, hs8mem, hs8len, hs8k⟩ :=
    parser_words_init hs7rip hs7r13 hs7len
  have hs8r : ∀ i v, i ≠ 1 → s7.reg i = some v →
      s8.reg i = some v := by
    intro i v hi h
    rw [hs8fr i hi]; exact h
  -- geometry of the three arena parts and the final stack pointer
  have hLfile : f.length ≤ arenaFileBytes f := by
    unfold arenaFileBytes
    have h2 := pages_ge (65536 + 2 * f.length)
    omega
  have h64nw : 64 * pr.words.length ≤ arenaTableBytes f :=
    Nat.le_trans (pages_ge _) hpgnw
  have hpgL : f.length ≤ X86Env.pages f.length := pages_ge _
  have hspFbase : (X86Env.initStack top [argv0, path]).2 + 24 + 48 =
      c2.base + 4168 := by omega
  have hspFhi : (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
      c2.hi := by omega
  have hspFW : (X86Env.initStack top [argv0, path]).2 + 24 + 48 < W := by
    have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
    have hW : W = 2 ^ 64 := rfl
    omega
  have hsh8 : wshape s8.mem = wshape ([c1, c2, buf'] ++
      [⟨A + arenaFileBytes f,
        List.replicate (X86Env.pages (64 * pr.words.length)) 0, true⟩,
       ⟨A + (arenaFileBytes f + arenaTableBytes f),
        List.replicate (X86Env.pages f.length) 0, true⟩]) := by
    rw [hs8mem, hs7mem]; exact hsh6
  have hroom8 : StackRoom s8 2 :=
    stackRoom_of_wshape_c2 hsh8 (hs8r 4 _ (by decide) hs7r4') hc2rw
      hdisj (by omega) hspFhi hspFW
  have hlay8 : WordsLay s8.mem (A + 16 + dataBytes pr.data) pr.words := by
    apply WordsLay_of_rdN pr.words _ hwB
    rw [hs8mem, hs7mem]
    exact enc_words_slice hrdNenc6
  -- writability of the table and scratch regions in s8.mem
  have htabhi : (⟨A + arenaFileBytes f,
        List.replicate (X86Env.pages (64 * pr.words.length)) 0,
        true⟩ : Region).hi =
      A + arenaFileBytes f + X86Env.pages (64 * pr.words.length) := by
    simp only [X86Env.Region.hi, List.length_replicate]
  have hscrhi : (⟨A + (arenaFileBytes f + arenaTableBytes f),
        List.replicate (X86Env.pages f.length) 0, true⟩ : Region).hi =
      A + (arenaFileBytes f + arenaTableBytes f) +
        X86Env.pages f.length := by
    simp only [X86Env.Region.hi, List.length_replicate]
  have hnonmem3 : ∀ a : Nat,
      A + arenaFileBytes f ≤ a → a < A + stage1Mf f →
      ∀ r' ∈ [c1, c2, buf'], ¬ (r'.base ≤ a ∧ a < r'.hi) := by
    intro a ha1 ha2 r' hr'
    have hr2 : r' = c1 ∨ r' = c2 ∨ r' = buf' := by simpa using hr'
    rcases hr2 with rfl | rfl | rfl
    · rcases hfreecode2 with h | h
      · exact not_mem_region_of_lt (by rw [hc1b]; omega)
      · exact not_mem_region_of_le (by rw [hc1hi]; omega)
    · rcases hfreestk with h | h
      · exact not_mem_region_of_lt (by rw [hc2b]; omega)
      · exact not_mem_region_of_le (by rw [hc2hi]; omega)
    · exact not_mem_region_of_le (by rw [hbufhi]; omega)
  have hwrTab : ∀ (off v : Nat), off + 8 ≤ 64 * pr.words.length →
      (wr64 s8.mem (A + arenaFileBytes f + off) v).isSome := by
    intro off v hoff
    refine wr64_isSome_of_forall_wr (fun j hj u => ?_)
    have ha1 : A + arenaFileBytes f ≤ A + arenaFileBytes f + off + j := by
      omega
    have ha2 : A + arenaFileBytes f + off + j < A + stage1Mf f := by
      have hpg := pages_ge f.length
      omega
    have hfind1 : Mem.find (([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩]))
        (A + arenaFileBytes f + off + j) 0 =
        some (0 + [c1, c2, buf'].length,
          ⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩) :=
      find_append_last [c1, c2, buf'] 0 (hnonmem3 _ ha1 ha2)
        ⟨by omega, by
          rw [htabhi]
          have hpg := pages_ge (64 * pr.words.length)
          omega⟩
    have hfind2 : Mem.find ((([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩])) ++
          [⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩])
          (A + arenaFileBytes f + off + j) 0 =
        some (0 + [c1, c2, buf'].length,
          ⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩) :=
      find_append_left _ _ _ 0 hfind1
    rw [List.append_assoc] at hfind2
    have href : ∃ ii r,
        Mem.find ([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩,
           ⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩])
          (A + arenaFileBytes f + off + j) 0 = some (ii, r) ∧
          r.rw = true :=
      ⟨0 + [c1, c2, buf'].length, _, hfind2, rfl⟩
    obtain ⟨ii, r, hf, hrw2⟩ := find_rw_of_wshape hsh8 href
    exact wr_isSome_iff.mpr ⟨ii, r, hf, hrw2⟩
  have hwrScr : ∀ a v,
      A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data - A) ≤ a →
      a < A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data + wordsBytes pr.words - A) →
      (s8.mem.wr a v).isSome := by
    intro a v ha1 ha2
    have e1 : A + 16 + dataBytes pr.data + wordsBytes pr.words - A =
        f.length := by omega
    rw [e1] at ha2
    have hpgL := pages_ge f.length
    have hage : A + arenaFileBytes f ≤ a := by omega
    have halt : a < A + stage1Mf f := by omega
    have hnonmem : ∀ r' ∈ ([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩]),
        ¬ (r'.base ≤ a ∧ a < r'.hi) := by
      intro r' hr'
      rcases List.mem_append.1 hr' with h3 | hT
      · exact hnonmem3 a hage halt r' h3
      · have hr2 : r' = ⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩ := by simpa using hT
        rw [hr2]
        exact not_mem_region_of_le (by rw [htabhi]; omega)
    have hfind1 : Mem.find ((([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩])) ++
          [⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩]) a 0 =
        some (0 + (([c1, c2, buf'] ++
          [(⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩ : Region)])).length,
          ⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩) :=
      @find_append_last _ 0
        ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩ a hnonmem
        ⟨by
          show A + (arenaFileBytes f + arenaTableBytes f) ≤ a
          omega, by rw [hscrhi]; omega⟩
    have href : ∃ ii r,
        Mem.find ([c1, c2, buf'] ++
          [⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩,
           ⟨A + (arenaFileBytes f + arenaTableBytes f),
            List.replicate (X86Env.pages f.length) 0, true⟩]) a 0 =
          some (ii, r) ∧ r.rw = true := by
      refine ⟨0 + (([c1, c2, buf'] ++
          [(⟨A + arenaFileBytes f,
            List.replicate (X86Env.pages (64 * pr.words.length)) 0,
            true⟩ : Region)])).length,
        ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩, ?_, rfl⟩
      rwa [List.append_assoc] at hfind1
    obtain ⟨ii, r, hf, hrw2⟩ := find_rw_of_wshape hsh8 href
    exact wr_isSome_iff.mpr ⟨ii, r, hf, hrw2⟩
  -- disjointness disjunctions for the word loop
  have hdisjTB : A + arenaFileBytes f + 64 * pr.words.length ≤ A ∨
      A + f.length ≤ A + arenaFileBytes f := Or.inr (by omega)
  have hdisjTStk : A + arenaFileBytes f + 64 * pr.words.length ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 16 ∨
      (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
        A + arenaFileBytes f := by
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdisjTScr : A + arenaFileBytes f + 64 * pr.words.length ≤
        A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data - A) ∨
      A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data + wordsBytes pr.words - A) ≤
        A + arenaFileBytes f := Or.inl (by omega)
  have hdisjBStk : A + f.length ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 16 ∨
      (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤ A := by
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdisjScrB : A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data + wordsBytes pr.words - A) ≤
        A + 16 + dataBytes pr.data ∨
      A + 16 + dataBytes pr.data + wordsBytes pr.words ≤
        A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data - A) := Or.inr (by omega)
  have hdisjScrStk : A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data + wordsBytes pr.words - A) ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 16 ∨
      (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
        A + (arenaFileBytes f + arenaTableBytes f) +
          (A + 16 + dataBytes pr.data - A) := by
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  -- the whole word loop
  obtain ⟨k9, s9, hrun9, hs9rip, hs9r1, hs9r6, hs9r7, hs9r4, hs9tab,
    hs9fr, hs9len, hs9k, hs9sh, hs9room, hs9memfr⟩ :=
    parser_words_loop pr.words hs8rip hs8r1
      (hs8r 6 _ (by decide) hs7r6) (hs8r 7 _ (by decide) hs7r7)
      (hs8r 15 _ (by decide) hs7r15) (hs8r 4 _ (by decide) hs7r4')
      (hs8r 3 _ (by decide) hs7r3) (hs8r 11 _ (by decide) hs7r11)
      hlay8 (by omega) (by omega) hAe (by
        have hW : W = 2 ^ 64 := rfl
        omega) (by
        have hpg := pages_ge f.length
        have hW : W = 2 ^ 64 := rfl
        omega) (by
        have hW : W = 2 ^ 64 := rfl
        exact Nat.lt_trans hbnw (by rw [hW]; decide))
      hroom8 (by omega) hspFW hs8len hdisjTB hdisjTStk hdisjTScr
      hdisjBStk hdisjScrB hdisjScrStk hwrTab hwrScr
  -- loop head with no words left: to the trailing checks
  have hs9r : ∀ i v, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 10 → s8.reg i = some v → s9.reg i = some v := by
    intro i v h0 h1 h2 h4 h6 h7 h10 h
    rw [hs9fr i h0 h1 h2 h4 h6 h7 h10]; exact h
  obtain ⟨s10, hrun10, hs10rip, hs10fr, hs10mem, hs10len, hs10k⟩ :=
    parser_words_head_zero hs9rip hs9r1
  have hs10len16 : s10.regs.length = 16 := by rw [hs10len]; exact hs9len
  have hs10r : ∀ i v, s9.reg i = some v → s10.reg i = some v := by
    intro i v h
    rw [hs10fr i]; exact h
  have hs10r6 : s10.reg 6 =
      some (A + 16 + dataBytes pr.data + wordsBytes pr.words) :=
    hs10r 6 _ hs9r6
  have hs10r15 : s10.reg 15 = some (A + f.length) :=
    hs10r 15 _ (hs9r 15 _ (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (hs8r 15 _ (by decide) hs7r15))
  have hs10r12 : s10.reg 12 = some pr.main :=
    hs10r 12 _ (hs9r 12 _ (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (hs8r 12 _ (by decide) hs7r12))
  have hs10r13 : s10.reg 13 = some pr.words.length :=
    hs10r 13 _ (hs9r 13 _ (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (hs8r 13 _ (by decide) hs7r13))
  have hs10r14 : s10.reg 14 = some (A + arenaFileBytes f) :=
    hs10r 14 _ (hs9r 14 _ (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide)
      (hs8r 14 _ (by decide) hs7r14))
  have hs10r4 : s10.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24 + 48) :=
    hs10r 4 _ hs9r4
  have hslot : rd64 s10.mem
      (A + arenaFileBytes f + 64 * pr.main) = some 0 := by
    rw [hs10mem]
    have h1 := TabLay_ins_at hs9tab pr.main hmn
    rwa [hins0] at h1
  -- trailing checks into the checker entry
  obtain ⟨s11, hrun11, hs11rip, -, hs11fr, hs11mem, hs11len, hs11k⟩ :=
    parser_trailing_ok hs10rip hs10r6 hs10r15 hs10r12 hs10r13 hs10r14
      (by omega) (by omega) hAe hmn (by
        have hW : W = 2 ^ 64 := rfl
        omega) hslot hs10len16
  have hs11r : ∀ i v, i ≠ 0 → i ≠ 2 → s10.reg i = some v →
      s11.reg i = some v := by
    intro i v h0 h2 h
    rw [hs11fr i h0 h2]; exact h
  -- assemble the whole run
  have hrunA : run stage1Im (12 + 11) (Out.ok s) = Out.ok s7 := by
    rw [run_add, hrun6, hrun7]
  have hrunB : run stage1Im (12 + 11 + 1) (Out.ok s) = Out.ok s8 := by
    rw [show 12 + 11 + 1 = (12 + 11) + 1 from rfl, run_add, hrunA, hrun8]
  have hrunC : run stage1Im (12 + 11 + 1 + k9) (Out.ok s) = Out.ok s9 := by
    rw [show 12 + 11 + 1 + k9 = (12 + 11 + 1) + k9 from rfl, run_add,
      hrunB, hrun9]
  have hrunD : run stage1Im (12 + 11 + 1 + k9 + 3) (Out.ok s) =
      Out.ok s10 := by
    rw [show 12 + 11 + 1 + k9 + 3 = (12 + 11 + 1 + k9) + 3 from rfl,
      run_add, hrunC, hrun10]
  have hrunTot : run stage1Im (12 + 11 + 1 + k9 + 3 + 15) (Out.ok s) =
      Out.ok s11 := by
    rw [show 12 + 11 + 1 + k9 + 3 + 15 = (12 + 11 + 1 + k9 + 3) + 15
      from rfl, run_add, hrunD, hrun11]
  have hsh11 : wshape s11.mem = wshape ([c1, c2, buf'] ++
      [⟨A + arenaFileBytes f,
        List.replicate (X86Env.pages (64 * pr.words.length)) 0, true⟩,
       ⟨A + (arenaFileBytes f + arenaTableBytes f),
        List.replicate (X86Env.pages f.length) 0, true⟩]) := by
    rw [hs11mem, hs10mem, hs9sh]; exact hsh8
  have hroom11 : StackRoom s11 6 :=
    stackRoom_of_wshape_c2 hsh11
      (hs11r 4 _ (by decide) (by decide) hs10r4) hc2rw hdisj
      (by omega) hspFhi hspFW
  have hk11 : s11.k = (Kern.next s.k).2 := by
    rw [hs11k, hs10k, hs9k, hs8k, hs7k]; exact hs6k
  have hcode11 : s.mem.rdN base stage1Code.length =
        some stage1Code →
      s11.mem.rdN base stage1Code.length = some stage1Code := by
    intro h
    have hm1c : m1.rdN base stage1Code.length = some stage1Code := by
      rw [hm1fr base _ (Or.inl (by omega))]; exact h
    have hs6c : s6.mem.rdN base stage1Code.length =
        some stage1Code := by
      rw [hs6mem]; exact rdN_append_left hm1c
    have hs8c : s8.mem.rdN base stage1Code.length =
        some stage1Code := by
      rw [hs8mem, hs7mem]; exact hs6c
    have hs9c : s9.mem.rdN base stage1Code.length =
        some stage1Code := by
      rw [hs9memfr base _ (Or.inl (by omega)) ?_ ?_]; exact hs8c
      · rcases hfreecode2 with hf | hf
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
      · rcases hfreecode2 with hf | hf
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
    rw [hs11mem, hs10mem]; exact hs9c
  exact ⟨12 + 11 + 1 + k9 + 3 + 15, s11, hrunTot, hs11rip,
    hs11r 14 _ (by decide) (by decide) hs10r14,
    hs11r 13 _ (by decide) (by decide) hs10r13,
    hs11r 4 _ (by decide) (by decide) hs10r4,
    by rw [hs11mem, hs10mem]; exact hs9tab,
    hroom11, hs11len, hsh11, hk11, hcode11⟩

set_option maxHeartbeats 0 in
/-- The whole accept path from process entry to the checker
entry, under the witness oracle: loader, parser front half and
parser back half composed. -/
theorem parser_accept_ok {f path argv0 stdin : List Nat} {top : Nat}
    {pr : Prog}
    (hinit : Stage1InitOk f path argv0 top)
    (hroom : Room f path argv0 top)
    (hdec : Sporos.decode f = some pr)
    (hchk : Sporos.checkProg pr = true) :
    ∃ (A k : Nat) (s' : X86Int.St),
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
          top)) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧
      s'.reg 14 = some (A + arenaFileBytes f) ∧
      s'.reg 13 = some pr.words.length ∧
      s'.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24 + 48) ∧
      TabLay s'.mem A
        (A + (arenaFileBytes f + arenaTableBytes f))
        (A + arenaFileBytes f)
        (A + 16 + dataBytes pr.data) pr.words ∧
      StackRoom s' 6 ∧ s'.regs.length = 16 ∧
      s'.mem.rdN base stage1Code.length = some stage1Code ∧
      s'.k.oracle = [] ∧
      (∃ c1 c2 buf', wshape s'.mem = wshape ([c1, c2, buf'] ++
        [⟨A + arenaFileBytes f,
          List.replicate (X86Env.pages (64 * pr.words.length)) 0,
          true⟩,
         ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩]) ∧
        c1.base = base ∧ c1.hi = base + stage1Code.length ∧
        c2.base = (X86Env.initStack top [argv0, path]).1.base ∧
        c2.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
        c2.rw = true ∧ c1.hi ≤ c2.base) ∧
      (∃ t, run stage1Im 21
        (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
          top)) = Out.ok t ∧ OracleLegalAt t) ∧
      A % page = 0 ∧ 0 < A ∧ A + stage1Mf f ≤ 2 ^ 47 ∧
      (stage1Init f path argv0 [] [] top).mem.free A (stage1Mf f) =
        true := by
  obtain ⟨A, k0, s0, hAmod, hApos, hAbound, hfreeA, hrun0, hs0rip,
    hs0r3, -, hs0r13, -, hs0r4, -, -, -, hs0r9, hs0oracle, -,
    hmem0, hs0rdN, hs0len, hlegal, hs0room, hs0code⟩ :=
    loader_init_to_loaded hinit hroom
  have hfB : ∀ b ∈ f, b < 256 := hinit.1
  have htop47 : top ≤ 2 ^ 47 := hinit.2.2.2.2.2.2.2.1
  have hleInit : phVaddr + phMemsz ≤
      (X86Env.initStack top [argv0, path]).1.base :=
    hinit.2.2.2.2.2.2.2.2.1
  have hstkhi : (X86Env.initStack top [argv0, path]).1.hi = top :=
    hinit.2.2.2.2.2.2.2.2.2
  have hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base := by
    have h := hleInit
    rw [phVaddr_eq, phMemsz_eq] at h
    exact h
  obtain ⟨c1, c2, buf', hmem0eq, hc1b, hc1hi, hc2b, hc2hi, hbufb,
    ⟨cap', hbufhi, hbufbound⟩, hbufrw, hc2rw⟩ := hmem0
  have hr7 : ∃ v, s0.reg 7 = some v :=
    run_reg_some_ne11 _ (by decide)
      (stage1Init_reg_ne4 (i := 7) f path argv0 stdin
        (stage1Oracle f A) top (by decide) (by decide)) hrun0
  obtain ⟨k1, s1, m2, hrun1, hs1rip, hs1r14, hs1r4, hs1r3, hs1r12,
    hs1r13, hs1r15, hs1mem, hm2sh, hs1rdN, hs1cur, hs1oracle,
    -, -, -, -, hs1r7, hs1r9, hs1room, hs1len, -, hs1code⟩ :=
    parser_front_ok hdec hchk hfB hAmod hApos hAbound hfreeA
      hdisjInit hstkhi htop47 hs0rip hs0r3 hs0r13 hs0r4 hr7 hs0r9
      hs0oracle
      ⟨c1, c2, buf', hmem0eq, hc1b, hc1hi, hc2b, hc2hi, hbufb,
        ⟨cap', hbufhi, hbufbound⟩, hbufrw, hc2rw⟩
      hs0rdN hs0len hs0room
  have hwsh1 : wshape s1.mem = wshape (([c1, c2, buf'] ++
      [⟨A + arenaFileBytes f,
        List.replicate (X86Env.pages (64 * pr.words.length)) 0,
        true⟩])) := by
    have happ : ∀ (m : Mem) (r : Region),
        wshape (m ++ [r]) = wshape m ++ [(r.base, r.hi, r.rw)] := by
      intro m r
      simp [wshape]
    rw [hs1mem, happ, hm2sh, hmem0eq, happ]
  obtain ⟨k2, s2, hrun2, hs2rip, hs2r14, hs2r13, hs2r4, hs2tab,
    hs2room, hs2len, hs2sh, hs2k, hs2code⟩ :=
    parser_back_ok hdec hchk hfB hAmod hApos hAbound hfreeA
      hdisjInit hstkhi htop47 hs1rip hs1r14 hs1r4 hs1r3 hs1r12
      hs1r13 hs1r15 hwsh1 hc1b hc1hi hc2b hc2hi hbufb hbufhi
      hbufbound hc2rw hs1rdN hs1cur hs1oracle hs1r7 hs1r9 hs1room
      hs1len
  have hrunA : run stage1Im (k0 + k1)
      (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
        top)) = Out.ok s1 := by
    rw [run_add, hrun0, hrun1]
  have hrunTot : run stage1Im (k0 + k1 + k2)
      (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
        top)) = Out.ok s2 := by
    rw [show k0 + k1 + k2 = (k0 + k1) + k2 from rfl, run_add, hrunA,
      hrun2]
  have hor2 : s2.k.oracle = [] := by
    rw [hs2k]
    exact next_oracle_tail hs1oracle
  exact ⟨A, k0 + k1 + k2, s2, hrunTot, hs2rip, hs2r14, hs2r13, hs2r4,
    hs2tab, hs2room, hs2len, hs2code (hs1code hs0code), hor2,
    ⟨c1, c2, buf', hs2sh, hc1b, hc1hi, hc2b, hc2hi, hc2rw,
      by rw [hc1hi, hc2b]; exact hdisjInit⟩,
    hlegal, hAmod, hApos, hAbound, hfreeA⟩

/-- `AnnLayB` is `AnnLayout` at the same state (repackaged). -/
theorem AnnLayout_of_AnnLayB {s : X86Int.St} {p : Nat}
    {ann : List (Nat × Height)} (h : AnnLayB s.mem p ann) :
    AnnLayout s ann p := by
  intro i hi
  obtain ⟨h1, h2, h3, h4⟩ := h i hi
  exact ⟨⟨h1, h2, h3⟩, h4⟩

/-- Byte-bounded instructions have byte-bounded immediates. -/
theorem ImmOk_of_InstrBound : ∀ {ins : Instr}, InstrBound ins →
    ImmOk ins := by
  intro ins h
  cases ins <;> first | exact h | trivial

/-- Primitive effects are byte-sized. -/
theorem prim_eff_le : ∀ p : Prim, p.eff.1 ≤ 255 ∧ p.eff.2 ≤ 255 := by
  intro p
  cases p <;> decide

/-- A looked-up annotation height inherits the annotation bounds. -/
theorem ann_lookup_bounds {ann : List (Nat × Height)} {i : Nat}
    {h : Height} (hl : ann.lookup i = some h)
    (hb : ∀ x ∈ ann, AnnBound x) :
    h.d < 256 ^ 4 ∧ h.a < 256 ^ 4 := by
  obtain ⟨l1, l2, hann, -⟩ := List.lookup_eq_some_iff.mp hl
  have hmem : (i, h) ∈ ann := by
    rw [hann]; exact List.mem_append_right _ (.head _)
  have hb2 := hb (i, h) hmem
  exact ⟨hb2.2.1, hb2.2.2⟩

set_option maxHeartbeats 0 in
/-- Every height `hBnd` meets is below `W`: each `hstep` adds at
most `256^4 - 1` to either component, annotation heights and the
entry height are at most `256^4 - 1`, and there are fewer than
`256^4` instructions, so heights stay below `(256^4)^2 = W`. -/
theorem hBnd_of_bounds : ∀ (code : List Instr) {pr : Prog}
    {wd : Word} {i : Nat} {cur : Height},
    i + code.length < 256 ^ 4 →
    cur.d ≤ (i + 1) * (256 ^ 4 - 1) →
    cur.a ≤ (i + 1) * (256 ^ 4 - 1) →
    (∀ x ∈ wd.ann, AnnBound x) →
    (∀ w ∈ pr.words, w.outs < 256 ^ 4) →
    hBnd pr wd code i cur := by
  intro code
  induction code with
  | nil =>
    intro pr wd i cur _ _ _ _ _
    trivial
  | cons ins rest ih =>
    intro pr wd i cur hlen hcd hca hann houts
    have hB1 : 255 ≤ 256 ^ 4 - 1 := by decide
    have hkey : ∀ x : Nat, x ≤ (i + 1) * (256 ^ 4 - 1) → x + 2 < W := by
      intro x hx
      have h1 : (i + 1) * (256 ^ 4 - 1) ≤
          (256 ^ 4 - 1) * (256 ^ 4 - 1) :=
        Nat.mul_le_mul_right _ (by
          simp only [List.length_cons] at hlen; omega)
      have h2 : (256 ^ 4 - 1) * (256 ^ 4 - 1) + 2 < W := by
        have hW : W = 2 ^ 64 := rfl
        rw [hW]; decide
      omega
    have hkey1 : ∀ x : Nat, x ≤ (i + 1) * (256 ^ 4 - 1) → x + 1 < W :=
      fun x hx => by have h := hkey x hx; omega
    have hgetD : ∀ (j : Nat) (dflt : Height),
        dflt.d ≤ (i + 1) * (256 ^ 4 - 1) →
        dflt.a ≤ (i + 1) * (256 ^ 4 - 1) →
        ((wd.ann.lookup j).getD dflt).d ≤ (i + 1) * (256 ^ 4 - 1) ∧
          ((wd.ann.lookup j).getD dflt).a ≤ (i + 1) * (256 ^ 4 - 1) := by
      intro j dflt hd ha
      cases hl : wd.ann.lookup j with
      | none =>
        exact ⟨hd, ha⟩
      | some hh =>
        simp only [Option.getD_some]
        obtain ⟨hbd, hba⟩ := ann_lookup_bounds hl hann
        have hge : 256 ^ 4 - 1 ≤ (i + 1) * (256 ^ 4 - 1) := by
          have e : (i + 1) * (256 ^ 4 - 1) =
              (256 ^ 4 - 1) + i * (256 ^ 4 - 1) := by
            rw [Nat.succ_mul, Nat.add_comm]
          omega
        exact ⟨by omega, by omega⟩
    obtain ⟨hhd, hha⟩ := hgetD i cur hcd hca
    have hstepB : ∀ {h h' : Height},
        hstep pr wd h ins = some (some h') →
        h'.d ≤ h.d + (256 ^ 4 - 1) ∧ h'.a ≤ h.a + (256 ^ 4 - 1) := by
      intro h h' hs
      cases ins with
      | prim p =>
        simp only [hstep] at hs
        split at hs
        · split at hs
          · simp at hs
          · obtain rfl : h' = ⟨h.d - p.eff.1 + p.eff.2, h.a⟩ :=
              (Option.some.inj (Option.some.inj hs)).symm
            have he := prim_eff_le p
            exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | lit n =>
        simp only [hstep] at hs
        obtain rfl : h' = ⟨h.d + 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
        exact ⟨by dsimp only; omega, by dsimp only; omega⟩
      | str n =>
        simp only [hstep] at hs
        obtain rfl : h' = ⟨h.d + 2, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
        exact ⟨by dsimp only; omega, by dsimp only; omega⟩
      | pick n =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d + 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | tor =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d - 1, h.a + 1⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | fromr =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d + 1, h.a - 1⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | rfetch =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d + 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | locget n =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d + 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | locset n =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d - 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | jmp t =>
        simp only [hstep] at hs
        split at hs <;> simp at hs
      | jz t =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d - 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | jnz t =>
        simp only [hstep] at hs
        split at hs
        · obtain rfl : h' = ⟨h.d - 1, h.a⟩ := (Option.some.inj (Option.some.inj hs)).symm
          exact ⟨by dsimp only; omega, by dsimp only; omega⟩
        · simp at hs
      | call w =>
        cases hww : pr.words[w]? with
        | none =>
          simp only [hstep, hww] at hs
          simp at hs
        | some c =>
          by_cases hcond : c.ins ≤ h.d
          · have hs2 : hstep pr wd h (.call w) =
                some (some ⟨h.d - c.ins + c.outs, h.a⟩) := by
              simp only [hstep, hww, if_pos hcond]
            rw [hs2] at hs
            obtain rfl : h' = ⟨h.d - c.ins + c.outs, h.a⟩ :=
              (Option.some.inj (Option.some.inj hs)).symm
            obtain ⟨hlt, hget⟩ := List.getElem?_eq_some_iff.mp hww
            have hcmem : c ∈ pr.words := by
              rw [← hget]; exact List.getElem_mem hlt
            have hco := houts c hcmem
            exact ⟨by dsimp only; omega, by dsimp only; omega⟩
          · have hs2 : hstep pr wd h (.call w) = none := by
              simp only [hstep, hww, if_neg hcond]
            rw [hs2] at hs
            simp at hs
      | ret =>
        simp only [hstep] at hs
        split at hs <;> simp at hs
    have hlen' : i + 1 + rest.length < 256 ^ 4 := by
      simp only [List.length_cons] at hlen; omega
    show hBnd pr wd (ins :: rest) i cur
    unfold hBnd
    refine ⟨hkey _ hhd, hkey1 _ hha, ?_⟩
    cases hs : hstep pr wd ((wd.ann.lookup i).getD cur) ins with
    | none =>
      trivial
    | some o =>
      cases o with
      | none =>
        obtain ⟨hnd, hna⟩ := hgetD (i + 1) ⟨0, 0⟩
          (by dsimp only; omega) (by dsimp only; omega)
        have e : (i + 1 + 1) * (256 ^ 4 - 1) =
            (i + 1) * (256 ^ 4 - 1) + (256 ^ 4 - 1) := by
          rw [Nat.succ_mul]
        exact ih hlen'
          (by rw [e]; exact Nat.le_trans hnd (Nat.le_add_right _ _))
          (by rw [e]; exact Nat.le_trans hna (Nat.le_add_right _ _))
          hann houts
      | some h' =>
        obtain ⟨hbd, hba⟩ := hstepB hs
        have e : (i + 1 + 1) * (256 ^ 4 - 1) =
            (i + 1) * (256 ^ 4 - 1) + (256 ^ 4 - 1) := by
          rw [Nat.succ_mul]
        refine ⟨?_, ?_, ih hlen' (by omega) (by omega) hann houts⟩
        · have h1 : h'.d ≤ (i + 1 + 1) * (256 ^ 4 - 1) := by omega
          have h2 : (i + 1 + 1) * (256 ^ 4 - 1) ≤
              256 ^ 4 * (256 ^ 4 - 1) :=
            Nat.mul_le_mul_right _ (by omega)
          have h3 : 256 ^ 4 * (256 ^ 4 - 1) + 2 < W := by
            have hW : W = 2 ^ 64 := rfl
            rw [hW]; decide
          omega
        · have h1 : h'.a ≤ (i + 1 + 1) * (256 ^ 4 - 1) := by omega
          have h2 : (i + 1 + 1) * (256 ^ 4 - 1) ≤
              256 ^ 4 * (256 ^ 4 - 1) :=
            Nat.mul_le_mul_right _ (by omega)
          have h3 : 256 ^ 4 * (256 ^ 4 - 1) + 2 < W := by
            have hW : W = 2 ^ 64 := rfl
            rw [hW]; decide
          omega

/-- Per-index contents of a laid-out words table. -/
theorem TabLay_at : ∀ (wds : List Word) {m : Mem} {B S tbl c : Nat},
    TabLay m B S tbl c wds → c + wordsBytes wds < W →
    ∀ j (hj : j < wds.length), ∃ cj,
      c ≤ cj ∧
      cj + 20 + codeBytes wds[j].code + 12 * wds[j].ann.length ≤
        c + wordsBytes wds ∧
      rd64 m (tbl + 64 * j) = some wds[j].ins ∧
      rd64 m (tbl + 64 * j + 8) = some wds[j].outs ∧
      rd64 m (tbl + 64 * j + 16) = some wds[j].locals ∧
      rd64 m (tbl + 64 * j + 24) = some wds[j].code.length ∧
      rd64 m (tbl + 64 * j + 32) = some wds[j].ann.length ∧
      rd64 m (tbl + 64 * j + 40) = some (cj + 20) ∧
      rd64 m (tbl + 64 * j + 48) =
        some (cj + 20 + codeBytes wds[j].code) ∧
      CodeSeq m (cj + 20) wds[j].code ∧
      AnnLayB m (cj + 20 + codeBytes wds[j].code) wds[j].ann := by
  intro wds
  induction wds with
  | nil =>
    intro m B S tbl c _ _ j hj
    simp at hj
  | cons wd wds ih =>
    intro m B S tbl c htab hcW j hj
    obtain ⟨h0, h8, h16, h24, h32, h40, h48, hseq, -, hann, hrest⟩ :=
      htab
    have hwb : wordsBytes (wd :: wds) =
        (20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds := rfl
    cases j with
    | zero =>
      refine ⟨c, Nat.le_refl _, by
        show c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤
          c + wordsBytes (wd :: wds)
        omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hseq, hann⟩
      · simpa using h0
      · simpa using h8
      · simpa using h16
      · simpa using h24
      · simpa using h32
      · have hlt : c + 20 < W := by omega
        rw [Nat.mod_eq_of_lt hlt] at h40
        simpa using h40
      · have hlt : c + 20 + codeBytes wd.code < W := by omega
        rw [Nat.mod_eq_of_lt hlt] at h48
        simpa using h48
    | succ j =>
      have hj' : j < wds.length := by
        simp only [List.length_cons] at hj; omega
      have hcW' : c + 20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds < W := by omega
      obtain ⟨cj, hcj1, hcj2, t0, t8, t16, t24, t32, t40, t48, tseq,
        tann⟩ := ih hrest hcW' j hj'
      refine ⟨cj, by omega, by
        rw [List.getElem_cons_succ]
        omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [show tbl + 64 * (j + 1) = tbl + 64 + 64 * j by omega,
          List.getElem_cons_succ]
        exact t0
      · rw [show tbl + 64 * (j + 1) + 8 = tbl + 64 + 64 * j + 8 by omega,
          List.getElem_cons_succ]
        exact t8
      · rw [show tbl + 64 * (j + 1) + 16 = tbl + 64 + 64 * j + 16 by
          omega, List.getElem_cons_succ]
        exact t16
      · rw [show tbl + 64 * (j + 1) + 24 = tbl + 64 + 64 * j + 24 by
          omega, List.getElem_cons_succ]
        exact t24
      · rw [show tbl + 64 * (j + 1) + 32 = tbl + 64 + 64 * j + 32 by
          omega, List.getElem_cons_succ]
        exact t32
      · rw [show tbl + 64 * (j + 1) + 40 = tbl + 64 + 64 * j + 40 by
          omega]
        exact t40
      · rw [show tbl + 64 * (j + 1) + 48 = tbl + 64 + 64 * j + 48 by
          omega, List.getElem_cons_succ]
        exact t48
      · rwa [List.getElem_cons_succ]
      · rwa [List.getElem_cons_succ]

/-- The parser's table layout is the checker's `WordTab`, given
the arena disjointness the real layout provides. -/
theorem WordTab_of_TabLay {pr : Prog} {s : X86Int.St}
    {B S tbl c sp : Nat}
    (htab : TabLay s.mem B S tbl c pr.words)
    (hcW : c + wordsBytes pr.words < W)
    (hc0 : 0 < c)
    (hwB : ∀ w ∈ pr.words, WordBound w)
    (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (hdisjT : sp ≤ tbl ∨ tbl + 64 * pr.words.length ≤ sp - 48)
    (hdisjB : sp ≤ c ∨ c + wordsBytes pr.words ≤ sp - 48)
    (hdisjEff : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff) :
    WordTab pr tbl sp s := by
  have h256W : ∀ {n : Nat}, n < 256 ^ 4 → n < W := by
    intro n hn
    have hW : W = 2 ^ 64 := rfl
    exact Nat.lt_trans hn (by rw [hW]; decide)
  intro j hj
  obtain ⟨cj, hcj1, hcj2, t0, t8, t16, t24, t32, t40, t48, tseq,
    tann⟩ := TabLay_at pr.words htab hcW j hj
  have hwBw := hwB pr.words[j] (List.getElem_mem hj)
  obtain ⟨hbins, hbouts, -, hbcodelen, -, hbcode, hbann⟩ := hwBw
  have hws : WalkStatic pr pr.words[j] (tbl + 64 * j)
      (cj + 20 + codeBytes pr.words[j].code) sp tbl := by
    refine ⟨h256W hbcodelen, by omega, by omega, by omega, by omega,
      hspW, ?_, ?_, ?_, ?_⟩
    · rcases hdisjT with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · rcases hdisjB with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · rcases hdisjEff with h | h
      · exact Or.inl (by omega)
      · exact Or.inr h
    · intro n hn
      refine ⟨by omega, ?_⟩
      rcases hdisjT with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
  have hok : WordOk pr pr.words[j] (tbl + 64 * j)
      (cj + 20 + codeBytes pr.words[j].code) (cj + 20) sp tbl := by
    refine ⟨hws, h256W hbins, ?_, ?_, ?_, ?_, by omega, ?_, ?_⟩
    · intro x hx
      have hb := hbann x hx
      exact ⟨h256W hb.2.1, h256W hb.2.2⟩
    · intro x hx
      exact ImmOk_of_InstrBound (hbcode x hx)
    · exact hBnd_of_bounds pr.words[j].code (by omega)
        (by dsimp only; omega) (by dsimp only; omega) hbann
        (fun w hw => (hwB w hw).2.1)
    · rcases hdisjB with h | h
      · exact Or.inr (by omega)
      · exact Or.inl (by omega)
    · rcases hdisjT with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · rcases hdisjB with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
  exact ⟨cj + 20, cj + 20 + codeBytes pr.words[j].code, t0, t8, t16,
    t24, t32, t40, t48, tseq, AnnLayout_of_AnnLayB tann, hok⟩

/-- A state whose memory holds the whole code at `base` has the
Eff table. -/
theorem EffTable_of_code {s : X86Int.St}
    (h : s.mem.rdN base stage1Code.length = some stage1Code) :
    EffTable s := by
  intro p
  have hcode : p.code < 57 := prim_code_lt p
  have hok : (0x91d + 2 * p.code) + 2 ≤ stage1Code.length := by
    rw [stage1Code_length]; omega
  have hs := rdN_slice h hok
  have e : base + (0x91d + 2 * p.code) = aEff + 2 * p.code := by
    unfold aEff; omega
  rw [e] at hs
  rw [hs, eff_bytes p]

set_option maxHeartbeats 0 in
/-- The witness run accepts: under the witness oracle, a file
that decodes and checks runs from process entry to exit 0. -/
theorem witness_exit0 {f path argv0 stdin : List Nat} {top : Nat}
    {pr : Prog}
    (hinit : Stage1InitOk f path argv0 top)
    (hroom : Room f path argv0 top)
    (hdec : Sporos.decode f = some pr)
    (hchk : Sporos.checkProg pr = true) :
    ∃ A k, run stage1Im k
      (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
        top)) = Out.exit 0 := by
  have hfB : ∀ b ∈ f, b < 256 := hinit.1
  have htop47 : top ≤ 2 ^ 47 := hinit.2.2.2.2.2.2.2.1
  have hleInit : phVaddr + phMemsz ≤
      (X86Env.initStack top [argv0, path]).1.base :=
    hinit.2.2.2.2.2.2.2.2.1
  have hstkhi : (X86Env.initStack top [argv0, path]).1.hi = top :=
    hinit.2.2.2.2.2.2.2.2.2
  have hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base := by
    have h := hleInit
    rw [phVaddr_eq, phMemsz_eq] at h
    exact h
  obtain ⟨hfenc, -, hbnw, -, hwB, -⟩ := decode_inv hdec hfB
  have hLenc : f.length =
      16 + dataBytes pr.data + wordsBytes pr.words := by
    rw [hfenc]; exact enc_length pr
  have h64nw : 64 * pr.words.length ≤ arenaTableBytes f := by
    have h1 : X86Env.pages (64 * pr.words.length) ≤
        arenaTableBytes f := by
      unfold arenaTableBytes
      apply pages_mono
      have h2 := wordsBytes_ge pr.words
      omega
    exact Nat.le_trans (pages_ge _) h1
  have hLfile : f.length ≤ arenaFileBytes f := by
    unfold arenaFileBytes
    have h2 := pages_ge (65536 + 2 * f.length)
    omega
  obtain ⟨A, k0, s2, hrun0, hs2rip, hs2r14, hs2r13, hs2r4, hs2tab,
    hs2room, hs2len, hs2code, -,
    ⟨c1, c2, buf', -, hc1b, hc1hi, hc2b, hc2hi, -, hdisj12⟩, -,
    hAmod, hApos, hAbound, hfreeA⟩ :=
    parser_accept_ok hinit hroom hdec hchk
  -- the free interval against the initial stack region
  have hfree0 := hfreeA
  rw [stage1Init_mem] at hfree0
  unfold X86Env.Mem.free at hfree0
  rw [List.all_eq_true] at hfree0
  have hfreestk := hfree0
    (X86Env.initStack top [argv0, path]).1 (by simp)
  rw [Bool.or_eq_true] at hfreestk
  simp only [decide_eq_true_eq] at hfreestk
  have hsp0 : (X86Env.initStack top [argv0, path]).2 + 24 + 48 =
      c2.base + 4168 := by
    have h1 := initStack_sp top argv0 path
    rw [← hc2b] at h1
    have hsr : X86Env.stackRoom = 4096 := rfl
    omega
  have hspHi : (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
      c2.hi := by
    have hh : c2.hi = c2.base +
        (X86Env.initStack top [argv0, path]).1.data.length := by
      have e : (X86Env.initStack top [argv0, path]).1.hi =
          (X86Env.initStack top [argv0, path]).1.base +
            (X86Env.initStack top [argv0, path]).1.data.length := rfl
      rw [hc2hi, e, ← hc2b]
    have hlen : 8192 ≤
        (X86Env.initStack top [argv0, path]).1.data.length := by
      rw [initStack_data_length]
      have hp : 4096 ≤ X86Env.pages
          (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
        unfold X86Env.pages X86Env.page
        have h1 : 1 ≤ (56 + ((argv0.length + 1) +
            (path.length + 1)) + 1 + 4095) / 4096 :=
          Nat.div_pos (by omega) (by decide)
        omega
      have hsr : X86Env.stackRoom = 4096 := rfl
      omega
    omega
  have hspW : (X86Env.initStack top [argv0, path]).2 + 24 + 48 < W := by
    have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
    have hW : W = 2 ^ 64 := rfl
    omega
  have htblW : A + arenaFileBytes f + 64 * pr.words.length < W := by
    have hW : W = 2 ^ 64 := rfl
    unfold stage1Mf at hAbound
    omega
  have hMf : stage1Mf f =
      arenaFileBytes f + arenaTableBytes f + arenaScratchBytes f :=
    rfl
  have htab : ∀ j : Nat, j < pr.words.length →
      (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
          A + arenaFileBytes f + 64 * j ∨
        A + arenaFileBytes f + 64 * j + 64 ≤
          (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 48 := by
    intro j hj
    rcases hfreestk with h | h
    · exact Or.inr (by clear hs2tab hs2code; omega)
    · exact Or.inl (by clear hs2tab hs2code; omega)
  have hdisjT : (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
        A + arenaFileBytes f ∨
      A + arenaFileBytes f + 64 * pr.words.length ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 48 := by
    rcases hfreestk with h | h
    · exact Or.inr (by clear hs2tab hs2code htab; omega)
    · exact Or.inl (by clear hs2tab hs2code htab; omega)
  have hdisjB : (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
        A + 16 + dataBytes pr.data ∨
      A + 16 + dataBytes pr.data + wordsBytes pr.words ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 48 := by
    rcases hfreestk with h | h
    · exact Or.inr (by clear hs2tab hs2code htab; omega)
    · exact Or.inl (by clear hs2tab hs2code htab; omega)
  have hdisjEff : aEff + 114 ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 48 ∨
      (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤ aEff := by
    have step1 : aEff + 114 ≤ base + stage1Code.length := by
      show base + 0x91d + 114 ≤ base + stage1Code.length
      rw [Nat.add_assoc]
      exact Nat.add_le_add_left
        (by rw [stage1Code_length]; decide) base
    have step3 : (X86Env.initStack top [argv0, path]).1.base ≤
        (X86Env.initStack top [argv0, path]).2 + 24 + 48 - 48 := by
      have h1 := initStack_sp top argv0 path
      have hsr : X86Env.stackRoom = 4096 := rfl
      clear hs2tab hs2code htab hdisjT hdisjB
      omega
    exact Or.inl (Nat.le_trans step1 (Nat.le_trans hdisjInit step3))
  have hregs : RegsOk s2 := by
    apply run_regs_ok hrun0
    intro i hi16 _
    by_cases h4 : i = 4
    · subst h4
      exact ⟨_, stage1Init_reg4 f path argv0 stdin
        (stage1Oracle f A) top⟩
    · exact ⟨0, stage1Init_reg_ne4 f path argv0 stdin
        (stage1Oracle f A) top hi16 h4⟩
  have htab2 : WordTab pr (A + arenaFileBytes f)
      ((X86Env.initStack top [argv0, path]).2 + 24 + 48) s2 :=
    WordTab_of_TabLay hs2tab (by
      have hW : W = 2 ^ 64 := rfl
      unfold stage1Mf at hAbound
      omega) (by omega) hwB (by omega) hspW htblW hdisjT hdisjB
      hdisjEff
  obtain ⟨k1, hrun1⟩ :=
    checker_accepts_from_start stage1_allCode stage1_outerCode
      (by omega) hspW htblW htab hdisjEff hchk
      (s := s2)
      (by unfold aCheckStart; exact hs2rip) hs2r14 hs2r13 hs2r4
      (by omega) hregs hs2room (EffTable_of_code hs2code) htab2
  exact ⟨A, k0 + k1, by rw [run_add, hrun0, hrun1]⟩


end SporosX1
