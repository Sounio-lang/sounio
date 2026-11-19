/-
Stage-1 final proof, refusal phase: oracle-legality reduction
and the parser refusal segments composed to exit 1.  Split out
of `Bootstrap.SporosX1Final` unchanged; the three target
theorems remain in `Bootstrap.SporosX1Final`.
-/
import Bootstrap.SporosX1FinalParser

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-- `mmap` never exits. -/
theorem syscall9_ne_exit {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 c : Nat} :
    X86Env.syscall k m 9 a1 a2 a3 a4 a5 a6 ≠ some (.exit c) := by
  intro h
  simp only [X86Env.syscall] at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · cases hnext : Kern.next k with
      | mk oc k1 =>
        rw [hnext] at h
        simp only [] at h
        cases hp : X86Env.place m oc (X86Env.pages a2) with
        | none => simp [hp] at h
        | some o =>
          cases o with
          | none => simp [hp] at h
          | some a => simp [hp] at h

/-- `mremap` never exits. -/
theorem syscall25_ne_exit {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 c : Nat} :
    X86Env.syscall k m 25 a1 a2 a3 a4 a5 a6 ≠ some (.exit c) := by
  intro h
  cases hf : m.find a1 0 with
  | none =>
    simp only [X86Env.syscall, hf] at h
    split at h <;> simp at h
  | some ir =>
    obtain ⟨i, r⟩ := ir
    simp only [X86Env.syscall, hf] at h
    split at h
    · simp at h
    · split at h
      · simp at h
      · split at h
        · simp at h
        · cases hp : X86Env.place (m.eraseIdx i) (Kern.next k).1
              (X86Env.pages a3) with
          | none => simp [hp] at h
          | some o =>
            cases o with
            | none => simp [hp] at h
            | some a => simp [hp] at h

/-- A state whose step is not stuck is oracle-legal: at a memory
syscall the step succeeding (or even exiting, which the syscall
numbers exclude) forces every argument register to read and the
environment call to return a value. -/
theorem oracleLegalAt_of_step_ne_stuck {s : X86Int.St}
    (h : X86Int.step stage1Im s ≠ .stuck) : OracleLegalAt s := by
  intro hmem
  obtain ⟨hrip, hnr⟩ := hmem
  have hstep : X86Int.step stage1Im s =
      X86Int.exec .syscall 2 s := by
    simp only [X86Int.step, hrip]
  have hsome : ∀ {i : Nat},
      i = 0 ∨ i = 7 ∨ i = 6 ∨ i = 2 ∨ i = 10 ∨ i = 8 ∨ i = 9 →
      ∃ v, s.reg i = some v := by
    intro i hi
    cases hv : s.reg i with
    | some v => exact ⟨v, rfl⟩
    | none =>
      exfalso
      apply h
      rw [hstep]
      rcases hi with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        simp [X86Int.exec, hv]
  obtain ⟨nr, h0⟩ := hsome (Or.inl rfl)
  obtain ⟨a1, h7⟩ := hsome (Or.inr (Or.inl rfl))
  obtain ⟨a2, h6⟩ := hsome (Or.inr (Or.inr (Or.inl rfl)))
  obtain ⟨a3, h2⟩ := hsome (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
  obtain ⟨a4, h10⟩ :=
    hsome (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  obtain ⟨a5, h8⟩ :=
    hsome (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
  obtain ⟨a6, h9⟩ :=
    hsome (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl))))))
  cases hsc : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 with
  | none =>
    exfalso
    apply h
    rw [hstep]
    simp [X86Int.exec, h0, h7, h6, h2, h10, h8, h9, hsc]
  | some r =>
    cases r with
    | ret rax k' m' =>
      exact ⟨nr, a1, a2, a3, a4, a5, a6, rax, k', m',
        h0, h7, h6, h2, h10, h8, h9, hsc⟩
    | exit c =>
      rcases hnr with hn | hn
      · have hnr9 : nr = 9 := Option.some.inj (h0.symm.trans hn)
        subst hnr9
        exact absurd hsc syscall9_ne_exit
      · have hnr25 : nr = 25 := Option.some.inj (h0.symm.trans hn)
        subst hnr25
        exact absurd hsc syscall25_ne_exit

/-- A run from `stuck` stays stuck. -/
theorem run_stuck (n : Nat) :
    run stage1Im n .stuck = .stuck := by
  cases n <;> rfl

/-- A state on a run that continues without getting stuck is
oracle-legal. -/
theorem oracleLegalAt_of_continues {s : X86Int.St} {m : Nat}
    (hm : 0 < m) (h2 : run stage1Im m (Out.ok s) ≠ .stuck) :
    OracleLegalAt s := by
  apply oracleLegalAt_of_step_ne_stuck
  intro hstuck
  obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have he : run stage1Im (m' + 1) (Out.ok s) =
      run stage1Im m' (X86Int.step stage1Im s) := rfl
  rw [hstuck, run_stuck] at he
  exact h2 he

/-- Every state strictly before an ok-reached milestone is
oracle-legal. -/
theorem oracleLegalAt_of_prefix {s0 s sT : X86Int.St} {k K : Nat}
    (hk : k < K) (h1 : run stage1Im k (Out.ok s0) = Out.ok s)
    (hT : run stage1Im K (Out.ok s0) = Out.ok sT) :
    OracleLegalAt s := by
  have h2 : run stage1Im (K - k) (Out.ok s) = Out.ok sT := by
    have h := hT
    rw [show K = k + (K - k) by omega, run_add, h1] at h
    exact h
  refine oracleLegalAt_of_continues (m := K - k) (by omega) ?_
  rw [h2]
  simp

set_option maxHeartbeats 0 in
/-- Parser header refusal, short file: fewer than 16 bytes sends
the run from `loaded` to `aFail1` in six steps. -/
theorem parser_header_a_short {B L : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL : L < 16)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 6 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 6 → i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
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
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } :=
    step_alu_add (by rw [ht1rip]; exact hc.p137) ht1r15 ht1r13
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x13a := by
    rw [ht2]; show t1.rip + 3 = base + 0x13a; rw [ht1rip]
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
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 6 (some B) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p13a) ht2r3
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 6 (some B) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x13d := by
    rw [ht3]; show t2.rip + 3 = base + 0x13d; rw [ht2rip]
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
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p13d) ht3r13
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x141 := by
    rw [ht4]; show t3.rip + 4 = base + 0x141; rw [ht3rip]
  have h16W : 16 % W = 16 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some false := by
    have hcf : t4.fl.cf = some true := by
      rw [ht4]
      show (aluF .sub L (16 % W)).2.cf = some true
      rw [h16W]; exact aluF_sub_cf_true hL
    have hcc := cond_cc3 hcf
    simpa using hcc
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x143 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p141) ht4fl
    rw [ht4rip] at h
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x143 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x143 := by rw [ht5]
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t5)
      (by rw [ht5rip]; exact hc.p143)
    simpa using h
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht6rip : t6.rip = aFail1 := by
    rw [ht6]; show base + 0x124 = aFail1; rfl
  have ht6fr : ∀ i, i ≠ 6 → i ≠ 15 → t6.reg i = s.reg i := by
    intro i hi6 hi15
    have h4i : t4.reg i = s.reg i := by
      rw [ht4]; show t3.reg i = s.reg i; exact ht3fr i hi6 hi15
    have h5i : t5.reg i = s.reg i := by
      rw [ht5]; show t4.reg i = s.reg i; exact h4i
    rw [ht6]; show t5.reg i = s.reg i; exact h5i
  have ht6mem : t6.mem = s.mem := by
    have h4m : t4.mem = s.mem := by
      rw [ht4]; show t3.mem = s.mem; exact ht3mem
    have h5m : t5.mem = s.mem := by
      rw [ht5]; show t4.mem = s.mem; exact h4m
    rw [ht6]; show t5.mem = s.mem; exact h5m
  have ht6len : t6.regs.length = 16 := by
    have h4l : t4.regs.length = 16 := by
      rw [ht4]; show t3.regs.length = 16; exact ht3len
    have h5l : t5.regs.length = 16 := by
      rw [ht5]; show t4.regs.length = 16; exact h4l
    rw [ht6]; show t5.regs.length = 16; exact h5l
  have ht6k : t6.k = s.k := by
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
    have h4k : t4.k = s.k := by
      rw [ht4]; show t3.k = s.k; exact h3k
    have h5k : t5.k = s.k := by
      rw [ht5]
      show ({ t4 with rip := base + 0x143 }).k = s.k
      exact h4k
    rw [ht6]
    show ({ t5 with rip := (base + 0x124) }).k = s.k
    exact h5k
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
  exact ⟨t6, hrun, ht6rip, ht6fr, ht6mem, ht6len, ht6k⟩

set_option maxHeartbeats 0 in
/-- Parser header refusal, bad magic: from `loaded` to `aFail1`
in ten steps when the first four bytes are not `SPR1`. -/
theorem parser_header_a_magic {B L : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hmagic : leVal (bs.take 4) % 2 ^ 32 ≠ 827478099)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 10 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBW : B < W := by omega
  have hbslen : bs.length = 20 := rdN_length h20
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
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
  have hB8 : s.mem.rdN B 8 = some (bs.take 8) := by
    have h := rdN_slice h20 (show 0 + 8 ≤ 20 by decide)
    simpa using h
  have hld : rd64 t5.mem (ea B 0) = some (leVal (bs.take 8)) := by
    unfold rd64 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hBW, ht5mem, hB8]; rfl
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (leVal (bs.take 8))) with
        rip := t5.rip + 3 } :=
    step_ld (by rw [ht5rip]; exact hc.p148) ht5r6 hld
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (leVal (bs.take 8))) with
        rip := t5.rip + 3 } := ⟨_, rfl⟩
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
  have ht7r0 : t7.reg 0 = some (leVal (bs.take 4) % 2 ^ 32) := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 0 = some (leVal (bs.take 4) % 2 ^ 32)
    rw [reg_with_rip, reg_set_self (by rw [ht6len]; decide),
      leVal_take4_mod32 htake8len, htk]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t7.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht6fr i hi0 hi6 hi15
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
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7 with
        fl := (aluF .sub (leVal (bs.take 4) % 2 ^ 32)
          (827478099 % W)).2, rip := t7.rip + 7 } :=
    step_cmpI (by rw [ht7rip]; exact hc.p14d) ht7r0
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7 with
        fl := (aluF .sub (leVal (bs.take 4) % 2 ^ 32)
          (827478099 % W)).2, rip := t7.rip + 7 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x154 := by
    rw [ht8]; show t7.rip + 7 = base + 0x154; rw [ht7rip]
  have himmW : 827478099 % W = 827478099 :=
    Nat.mod_eq_of_lt (by unfold W; decide)
  have hz : (aluF .sub (leVal (bs.take 4) % 2 ^ 32)
      (827478099 % W)).1 ≠ 0 := by
    have hW : W = 2 ^ 64 := rfl
    have hM : leVal (bs.take 4) % 2 ^ 32 < 2 ^ 32 :=
      Nat.mod_lt _ (by decide)
    rw [aluF_sub_fst, himmW]
    by_cases hge : 827478099 ≤ leVal (bs.take 4) % 2 ^ 32
    · have e : leVal (bs.take 4) % 2 ^ 32 + W - 827478099 =
          (leVal (bs.take 4) % 2 ^ 32 - 827478099) + W := by omega
      rw [e, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      omega
    · rw [Nat.mod_eq_of_lt (by omega)]
      omega
  have ht8fl : X86Int.cond t8.fl 4 = some false := by
    have hzf : t8.fl.zf = some false := by
      rw [ht8]
      show (aluF .sub (leVal (bs.take 4) % 2 ^ 32)
        (827478099 % W)).2.zf = some false
      simp [aluF_zf, hz]
    have hcc := cond_cc4 hzf
    simpa using hcc
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8 with rip := base + 0x156 } := by
    have h := step_jcc (im := stage1Im) (s := t8)
      (by rw [ht8rip]; exact hc.p154) ht8fl
    rw [ht8rip] at h
    simpa using h
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8 with rip := base + 0x156 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x156 := by rw [ht9]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t9)
      (by rw [ht9rip]; exact hc.p156)
    simpa using h
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht10rip : t10.rip = aFail1 := by
    rw [ht10]; show base + 0x124 = aFail1; rfl
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t10.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    have h8i : t8.reg i = s.reg i := by
      rw [ht8]; show t7.reg i = s.reg i; exact ht7fr i hi0 hi6 hi15
    have h9i : t9.reg i = s.reg i := by
      rw [ht9]; show t8.reg i = s.reg i; exact h8i
    rw [ht10]; show t9.reg i = s.reg i; exact h9i
  have ht10mem : t10.mem = s.mem := by
    have h8m : t8.mem = s.mem := by
      rw [ht8]; show t7.mem = s.mem; exact ht7mem
    have h9m : t9.mem = s.mem := by
      rw [ht9]; show t8.mem = s.mem; exact h8m
    rw [ht10]; show t9.mem = s.mem; exact h9m
  have ht10len : t10.regs.length = 16 := by
    have h8l : t8.regs.length = 16 := by
      rw [ht8]; show t7.regs.length = 16; exact ht7len
    have h9l : t9.regs.length = 16 := by
      rw [ht9]; show t8.regs.length = 16; exact h8l
    rw [ht10]; show t9.regs.length = 16; exact h9l
  have ht10k : t10.k = s.k := by
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
    have h4k : t4.k = s.k := by
      rw [ht4]; show t3.k = s.k; exact h3k
    have h5k : t5.k = s.k := by
      rw [ht5]
      show ({ t4 with rip := base + 0x148 }).k = s.k
      exact h4k
    have h6k : t6.k = s.k := by
      rw [ht6]
      show ({ t5.set 0 (some (leVal (bs.take 8))) with
        rip := t5.rip + 3 }).k = s.k
      exact h5k
    have h7k : t7.k = s.k := by
      rw [ht7]
      show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 }).k = s.k
      exact h6k
    have h8k : t8.k = s.k := by
      rw [ht8]; show t7.k = s.k; exact h7k
    have h9k : t9.k = s.k := by
      rw [ht9]
      show ({ t8 with rip := base + 0x156 }).k = s.k
      exact h8k
    rw [ht10]
    show ({ t9 with rip := (base + 0x124) }).k = s.k
    exact h9k
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
  have hrun : run stage1Im 10 (Out.ok s) = Out.ok t10 := by
    rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r8,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
  exact ⟨t10, hrun, ht10rip, ht10fr, ht10mem, ht10len, ht10k⟩

set_option maxHeartbeats 0 in
/-- Parser header refusal, `nwords = 0`: from `base + 0x15b` to
`aFail1` in sixteen steps when the u32 at `B + 8` is zero. -/
theorem parser_header_b_nwords_zero {B L : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x15b) (hr6 : s.reg 6 = some B)
    (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hnw : leVal ((bs.drop 8).take 4) % 2 ^ 32 = 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 16 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBoff : ∀ o : Nat, o ≤ 12 → B + o < W := by
    intro o ho; omega
  obtain ⟨s1, hrun1, hs1rip, hs1r12, hs1r6, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    field_load (site := base + 0x15b) (off := 4) (dst := 12) (len4 := 3)
      (by exact hc.p15b)
      (by exact hc.p15e) (by exact hc.p162) (by exact hc.p165)
      hrip hr6 h20 (by decide) (by unfold W; decide) (hBoff 4 (by decide))
      (by decide) (by decide) hlen16
  have hs1rip' : s1.rip = base + 0x168 := by
    rw [hs1rip]
  have h20_1 : s1.mem.rdN B 20 = some bs := by rw [hs1mem]; exact h20
  obtain ⟨s2, hrun2, hs2rip, hs2r10, hs2r6, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    field_load (site := base + 0x168) (off := 8) (dst := 10) (len4 := 3)
      (by exact hc.p168) (by exact hc.p16b) (by exact hc.p16f)
      (by exact hc.p172)
      hs1rip' hs1r6 h20_1 (by decide) (by unfold W; decide)
      (hBoff 8 (by decide)) (by decide) (by decide) hs1len
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
  have hs3r10 : s3.reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
    rw [hs3fr 10 (by decide) (by decide) (by decide)]; exact hs2r10
  have e10 : X86Int.step stage1Im s3 =
      .ok { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } :=
    step_aluI_add (by rw [hs3rip']; exact hc.p181) hs3r6
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x185 := by
    rw [ht1]; show s3.rip + 4 = base + 0x185; rw [hs3rip']
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
  have ht2fl : X86Int.cond t2.fl 5 = some false := by
    have hzf : t2.fl.zf = some true := by
      rw [ht2]
      show (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
        (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self, hnw]
      rfl
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e12 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x18a } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p188) ht2fl
    rw [ht2rip] at h
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x18a } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x18a := by rw [ht3]
  have e13 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p18a)
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht4rip : t4.rip = aFail1 := by
    rw [ht4]; show base + 0x124 = aFail1; rfl
  have r10 : run stage1Im 1 (Out.ok s3) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s3, e10, ← ht1, run_zero]
  have r11 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e11, ← ht2, run_zero]
  have r12 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e12, ← ht3, run_zero]
  have r13 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e13, ← ht4, run_zero]
  have hrun : run stage1Im 16 (Out.ok s) = Out.ok t4 := by
    rw [show (16 : Nat) = 4 + 12 by omega, run_add stage1Im 4 12, hrun1,
      show (12 : Nat) = 4 + 8 by omega, run_add stage1Im 4 8, hrun2,
      show (8 : Nat) = 4 + 4 by omega, run_add stage1Im 4 4, hrun3,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r10,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r11,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r12, r13]
  have ht4k : t4.k = s.k := by
    have h2k : t2.k = s.k := by
      have h1k : t1.k = s.k := by
        rw [ht1]
        show (s3.set 6 (some (aluF .add B (16 % W)).1)).k = s.k
        exact hs3k.trans (hs2k.trans hs1k)
      rw [ht2]
      show t1.k = s.k
      exact h1k
    have h3k : t3.k = s.k := by
      rw [ht3]
      show ({ t2 with rip := base + 0x18a }).k = s.k
      exact h2k
    rw [ht4]
    show ({ t3 with rip := (base + 0x124) }).k = s.k
    exact h3k
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
      t4.reg i = s.reg i := by
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
    have h3i : t3.reg i = s.reg i := by
      rw [ht3]
      show ({ t2 with rip := base + 0x18a }).reg i = s.reg i
      rw [reg_with_rip]; exact h2i
    rw [ht4]
    show ({ t3 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h3i
  have ht4mem : t4.mem = s.mem := by
    have h2m : t2.mem = s.mem := by
      rw [ht2]; show t1.mem = s.mem; exact ht1mem
    have h3m : t3.mem = s.mem := by
      rw [ht3]
      show ({ t2 with rip := base + 0x18a }).mem = s.mem
      rw [mem_with_rip]; exact h2m
    rw [ht4]
    show ({ t3 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h3m
  have ht4len : t4.regs.length = 16 := by
    have h2l : t2.regs.length = 16 := by
      rw [ht2]; show t1.regs.length = 16; exact ht1len
    have h3l : t3.regs.length = 16 := by
      rw [ht3]
      show ({ t2 with rip := base + 0x18a }).regs.length = 16
      rw [len_with_rip]; exact h2l
    rw [ht4]
    show ({ t3 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h3l
  exact ⟨t4, hrun, ht4rip, ht4fr, ht4mem, ht4len, ht4k⟩

/-- `nwords = 0` exits 1: the header refusal composed with the
`fail1` exit block. -/
theorem parser_header_b_nwords_zero_exit1 {B L sp : Nat}
    {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x15b) (hr6 : s.reg 6 = some B)
    (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hnw : leVal ((bs.drop 8).take 4) % 2 ^ 32 = 0)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_header_b_nwords_zero hrip hr6 hBL hL16 h20 hnw hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨16 + k, by rw [run_add stage1Im 16 k, hrun, hk]⟩

set_option maxHeartbeats 0 in
/-- Data-loop refusal, short length field: fewer than four
bytes remain for the segment length, so the loop step reaches
`aFail1` in seven steps. -/
theorem parser_data_step_short {c e n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hce : c ≤ e) (hshort : e < c + 4)
    (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 7 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 2 → s'.reg i = s.reg i) ∧
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
  have ht5fl : X86Int.cond t5.fl 3 = some false := by
    have hcf : t5.fl.cf = some true := by
      rw [ht5]
      show (aluF .sub (e - c) (4 % W)).2.cf = some true
      rw [h4W]; exact aluF_sub_cf_true (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5 with rip := base + 0x1a0 } := by
    have h := step_jcc (im := stage1Im) (s := t5)
      (by rw [ht5rip]; exact hc.p19e) ht5fl
    rw [ht5rip] at h
    simpa using h
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5 with rip := base + 0x1a0 } := ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x1a0 := by rw [ht6]
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t6)
      (by rw [ht6rip]; exact hc.p1a0)
    simpa using h
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht7rip : t7.rip = aFail1 := by
    rw [ht7]; show base + 0x124 = aFail1; rfl
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
  have hrun : run stage1Im 7 (Out.ok s) = Out.ok t7 := by
    rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
  have ht7fr : ∀ i, i ≠ 2 → t7.reg i = s.reg i := by
    intro i hi2
    have h5i : t5.reg i = s.reg i := by
      rw [ht5]; show t4.reg i = s.reg i; exact ht4fr i hi2
    have h6i : t6.reg i = s.reg i := by
      rw [ht6]
      show ({ t5 with rip := base + 0x1a0 }).reg i = s.reg i
      rw [reg_with_rip]; exact h5i
    rw [ht7]
    show ({ t6 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h6i
  have ht7mem : t7.mem = s.mem := by
    have h5m : t5.mem = s.mem := by
      rw [ht5]; show t4.mem = s.mem; exact ht4mem
    have h6m : t6.mem = s.mem := by
      rw [ht6]
      show ({ t5 with rip := base + 0x1a0 }).mem = s.mem
      rw [mem_with_rip]; exact h5m
    rw [ht7]
    show ({ t6 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h6m
  have ht7len : t7.regs.length = 16 := by
    have h5l : t5.regs.length = 16 := by
      rw [ht5]; show t4.regs.length = 16; exact ht4len
    have h6l : t6.regs.length = 16 := by
      rw [ht6]
      show ({ t5 with rip := base + 0x1a0 }).regs.length = 16
      rw [len_with_rip]; exact h5l
    rw [ht7]
    show ({ t6 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h6l
  have ht7k : t7.k = s.k := by
    have h2k : t2.k = s.k := by
      rw [ht2]
      show ({ t1 with rip := base + 0x194 }).k = s.k
      rw [ht1]
    have h3k : t3.k = s.k := by
      rw [ht3]
      show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).k = s.k
      exact h2k
    have h4k : t4.k = s.k := by
      rw [ht4]
      show (t3.set 2 (some (aluF .sub e c).1)).k = s.k
      exact h3k
    have h5k : t5.k = s.k := by
      rw [ht5]; show t4.k = s.k; exact h4k
    have h6k : t6.k = s.k := by
      rw [ht6]
      show ({ t5 with rip := base + 0x1a0 }).k = s.k
      exact h5k
    rw [ht7]
    show ({ t6 with rip := (base + 0x124) }).k = s.k
    exact h6k
  exact ⟨t7, hrun, ht7rip, ht7fr, ht7mem, ht7len, ht7k⟩

set_option maxHeartbeats 0 in
/-- Data-loop refusal, payload overrun: the segment length
runs past the end of the file, so the loop step reaches
`aFail1` in fourteen steps. -/
theorem parser_data_step_overrun {c e d n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hce : c + 4 ≤ e)
    (hd : rd32 s.mem c = some d) (hover : e < c + 4 + d)
    (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 14 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
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
  have ht12fl : X86Int.cond t12.fl 3 = some false := by
    have hcf : t12.fl.cf = some true := by
      rw [ht12]
      show (aluF .sub (e - (c + 4)) d).2.cf = some true
      exact aluF_sub_cf_true (by omega)
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
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12 with rip := base + 0x1b9 } := by
    have h := step_jcc (im := stage1Im) (s := t12)
      (by rw [ht12rip]; exact hc.p1b7) ht12fl
    rw [ht12rip] at h
    simpa using h
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12 with rip := base + 0x1b9 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x1b9 := by rw [ht13]
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t13)
      (by rw [ht13rip]; exact hc.p1b9)
    simpa using h
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht14rip : t14.rip = aFail1 := by
    rw [ht14]; show base + 0x124 = aFail1; rfl
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
  have hrun : run stage1Im 14 (Out.ok s) = Out.ok t14 := by
    rw [show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r1,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r2,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r3,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r4,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r5,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r6,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r7,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r8,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r9,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r10,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r11,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r12,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r13, r14]
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t14.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    have h13i : t13.reg i = s.reg i := by
      rw [ht13]
      show ({ t12 with rip := base + 0x1b9 }).reg i = s.reg i
      rw [reg_with_rip]; exact ht12fr i hi0 hi2 hi6
    rw [ht14]
    show ({ t13 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h13i
  have ht14mem : t14.mem = s.mem := by
    have h13m : t13.mem = s.mem := by
      rw [ht13]
      show ({ t12 with rip := base + 0x1b9 }).mem = s.mem
      rw [mem_with_rip]; exact ht12mem
    rw [ht14]
    show ({ t13 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h13m
  have ht14len : t14.regs.length = 16 := by
    have h13l : t13.regs.length = 16 := by
      rw [ht13]
      show ({ t12 with rip := base + 0x1b9 }).regs.length = 16
      rw [len_with_rip]; exact ht12len
    rw [ht14]
    show ({ t13 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h13l
  have ht14k : t14.k = s.k := by
    have h2k : t2.k = s.k := by
      rw [ht2]
      show ({ t1 with rip := base + 0x194 }).k = s.k
      rw [ht1]
    have h3k : t3.k = s.k := by
      rw [ht3]
      show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).k = s.k
      exact h2k
    have h4k : t4.k = s.k := by
      rw [ht4]
      show (t3.set 2 (some (aluF .sub e c).1)).k = s.k
      exact h3k
    have h5k : t5.k = s.k := by
      rw [ht5]; show t4.k = s.k; exact h4k
    have h6k : t6.k = s.k := by
      rw [ht6]
      show ({ t5 with rip := base + 0x1a5 }).k = s.k
      exact h5k
    have h7k : t7.k = s.k := by
      rw [ht7]
      show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).k = s.k
      exact h6k
    have h8k : t8.k = s.k := by
      rw [ht8]
      show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).k = s.k
      exact h7k
    have h9k : t9.k = s.k := by
      rw [ht9]
      show (t8.set 6 (some (aluF .add c (4 % W)).1)).k = s.k
      exact h8k
    have h10k : t10.k = s.k := by
      rw [ht10]
      show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).k = s.k
      exact h9k
    have h11k : t11.k = s.k := by
      rw [ht11]
      show (t10.set 2 (some (aluF .sub e (c + 4)).1)).k = s.k
      exact h10k
    have h12k : t12.k = s.k := by
      rw [ht12]
      show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).k = s.k
      exact h11k
    have h13k : t13.k = s.k := by
      rw [ht13]
      show ({ t12 with rip := base + 0x1b9 }).k = s.k
      exact h12k
    rw [ht14]
    show ({ t13 with rip := (base + 0x124) }).k = s.k
    exact h13k
  exact ⟨t14, hrun, ht14rip, ht14fr, ht14mem, ht14len, ht14k⟩

/-- Short length field exits 1. -/
theorem parser_data_step_short_exit1 {c e n sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hce : c ≤ e) (hshort : e < c + 4)
    (heW : e < W) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_data_step_short hrip hr1 hr6 hr15 hn hce hshort heW hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨7 + k, by rw [run_add stage1Im 7 k, hrun, hk]⟩

/-- Payload overrun exits 1. -/
theorem parser_data_step_overrun_exit1 {c e d n sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hce : c + 4 ≤ e)
    (hd : rd32 s.mem c = some d) (hover : e < c + 4 + d)
    (heW : e < W) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_data_step_overrun hrip hr1 hr6 hr15 hn hce hd hover heW
      hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨14 + k, by rw [run_add stage1Im 14 k, hrun, hk]⟩


set_option maxHeartbeats 0 in
/-- Guard refusal: `20 * nwords` exceeds the bytes remaining,
so the guard reaches `aFail1` in twelve steps. -/
theorem parser_guard_fail {c e nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1c7) (hr10 : s.reg 10 = some nw)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hnw32 : nw < 2 ^ 32) (hce : c ≤ e) (heW : e < W)
    (hguard : e - c < 20 * nw) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
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
  have ht10fl : X86Int.cond t10.fl 6 = some false := by
    have hcf : t10.fl.cf = some false := by
      rw [ht10]
      show (aluF .sub (20 * nw) (e - c)).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hzval : (aluF .sub (20 * nw) (e - c)).1 = 20 * nw - (e - c) := by
      rw [aluF_sub_fst,
        show 20 * nw + W - (e - c) = (20 * nw - (e - c)) + W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have hzf : t10.fl.zf = some false := by
      rw [ht10]
      show (aluF .sub (20 * nw) (e - c)).2.zf = some false
      rw [aluF_zf, hzval]
      simp [show 20 * nw - (e - c) ≠ 0 by omega]
    simp [X86Int.cond, hcf, hzf]
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x1e7 } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p1e5) ht10fl
    rw [ht10rip] at h
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x1e7 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x1e7 := by rw [ht11]
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t11)
      (by rw [ht11rip]; exact hc.p1e7)
    simpa using h
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have ht12rip : t12.rip = aFail1 := by
    rw [ht12]; show base + 0x124 = aFail1; rfl
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
  have ht12k : t12.k = s.k := by
    have h11k : t11.k = s.k := by
      rw [ht11]
      show ({ t10 with rip := base + 0x1e7 }).k = s.k
      exact h10k
    rw [ht12]
    show ({ t11 with rip := (base + 0x124) }).k = s.k
    exact h11k
  refine ⟨t12, hrun, ht12rip, ?_, ?_, ?_, ht12k⟩
  · intro i hi0 hi2
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i; exact ht9fr i hi0 hi2
    have h11i : t11.reg i = s.reg i := by
      rw [ht11]
      show ({ t10 with rip := base + 0x1e7 }).reg i = s.reg i
      rw [reg_with_rip]; exact h10i
    rw [ht12]
    show ({ t11 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h11i
  · have h10m : t10.mem = s.mem := by
      rw [ht10]; show t9.mem = s.mem; exact ht9mem
    have h11m : t11.mem = s.mem := by
      rw [ht11]
      show ({ t10 with rip := base + 0x1e7 }).mem = s.mem
      rw [mem_with_rip]; exact h10m
    rw [ht12]
    show ({ t11 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h11m
  · have h10l : t10.regs.length = 16 := by
      rw [ht10]; show t9.regs.length = 16; exact ht9len
    have h11l : t11.regs.length = 16 := by
      rw [ht11]
      show ({ t10 with rip := base + 0x1e7 }).regs.length = 16
      rw [len_with_rip]; exact h10l
    rw [ht12]
    show ({ t11 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h11l

/-- Guard refusal exits 1. -/
theorem parser_guard_fail_exit1 {c e nw sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1c7) (hr10 : s.reg 10 = some nw)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hnw32 : nw < 2 ^ 32) (hce : c ≤ e) (heW : e < W)
    (hguard : e - c < 20 * nw) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_guard_fail hrip hr10 hr6 hr15 hnw32 hce heW hguard hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨12 + k, by rw [run_add stage1Im 12 k, hrun, hk]⟩

set_option maxHeartbeats 0 in
/-- `mmap` with a refusing oracle returns `-ENOMEM`, memory
unchanged, and consumes the oracle entry. -/
theorem syscall_mmap_nomem {k : Kern} {m : Mem} {a2 : Nat}
    (ha2 : a2 ≠ 0) (hhead : k.oracle.head? = some .nomem) :
    X86Env.syscall k m 9 0 a2 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next k).2 m) := by
  have hnext1 : (Kern.next k).1 = some .nomem := kern_next_of_head hhead
  simp only [X86Env.syscall]
  rw [if_neg (by simp), if_neg ha2]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    rfl

set_option maxHeartbeats 0 in
/-- The words-table `mmap`, oracle-refusal case: from
`base + 0x1ec` to `aFail2` in twenty steps when the oracle
head is `.nomem`.  The two saved stack slots are the only
memory change, recorded as `wshape` preservation plus the
stack room at the decremented `rsp` that `fail2` needs. -/
theorem parser_mmap1_nomem {c nw sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 20 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail2 ∧ s'.reg 4 = some (sp - 8) ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 1 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 := by
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
  have hsc : X86Env.syscall t16.k t16.mem 9 0 (64 * nw) PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next s.k).2 m2) := by
    rw [ht16k, ht16mem]
    exact syscall_mmap_nomem (by
      have : 0 < nw := Nat.pos_of_ne_zero hnw0; omega) hhead
  have e17 : X86Int.step stage1Im t16 =
      .ok { ((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2, rip := t16.rip + 2 } :=
    step_syscall_ret (by rw [ht16rip]; exact hc.p21e) ht16r0 ht16r7
      ht16r6 ht16r2 ht16r10 ht16r8 ht16r9 hsc
  obtain ⟨t17, ht17⟩ : ∃ t : X86Int.St, t =
      { ((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2, rip := t16.rip + 2 } :=
    ⟨_, rfl⟩
  have ht17rip : t17.rip = base + 0x220 := by
    rw [ht17]; show t16.rip + 2 = base + 0x220; rw [ht16rip]
  have ht17r0 : t17.reg 0 = some (X86Env.neg ENOMEM) := by
    rw [ht17]
    show (((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11
      none).reg 0 = some (X86Env.neg ENOMEM)
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht16len]; decide)]
  have ht17fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t17.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht17]
    show (((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht16fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht17r4 : t17.reg 4 = some (sp - 8) := by
    rw [ht17]
    show (((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11
      none).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide),
      ht16r 4 (Or.inr (Or.inr (Or.inl rfl)))]
    exact ht15r4
  have ht17mem : t17.mem = m2 := by
    rw [ht17]
  have ht17len : t17.regs.length = 16 := by
    rw [ht17]
    show (((t16.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t16.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht16len
  have ht17k : t17.k = (Kern.next s.k).2 := by
    rw [ht17]
  have e18 : X86Int.step stage1Im t17 =
      .ok { t17 with fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2, rip := t17.rip + 3 } :=
    step_test (by rw [ht17rip]; exact hc.p220) ht17r0 ht17r0
  obtain ⟨t18, ht18⟩ : ∃ t : X86Int.St, t =
      { t17 with fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2, rip := t17.rip + 3 } := ⟨_, rfl⟩
  have ht18rip : t18.rip = base + 0x223 := by
    rw [ht18]; show t17.rip + 3 = base + 0x223; rw [ht17rip]
  have hmsb : msb (X86Env.neg ENOMEM) = true := by
    unfold msb X86Env.neg ENOMEM W
    decide
  have ht18fl : X86Int.cond t18.fl 9 = some false := by
    have hsf : t18.fl.sf = some true := by
      rw [ht18]
      show (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf = some true
      have h1 : (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf = some (msb (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  have e19 : X86Int.step stage1Im t18 =
      .ok { t18 with rip := base + 0x225 } := by
    have h := step_jcc (im := stage1Im) (s := t18)
      (by rw [ht18rip]; exact hc.p223) ht18fl
    rw [ht18rip] at h
    simpa using h
  obtain ⟨t19, ht19⟩ : ∃ t : X86Int.St, t =
      { t18 with rip := base + 0x225 } := ⟨_, rfl⟩
  have ht19rip : t19.rip = base + 0x225 := by rw [ht19]
  have e20 : X86Int.step stage1Im t19 =
      .ok { t19 with rip := (base + 0x12c) } := by
    have h := step_jmp (im := stage1Im) (s := t19)
      (by rw [ht19rip]; exact hc.p225)
    simpa using h
  obtain ⟨t20, ht20⟩ : ∃ t : X86Int.St, t =
      { t19 with rip := (base + 0x12c) } := ⟨_, rfl⟩
  have ht20rip : t20.rip = aFail2 := by
    rw [ht20]; show base + 0x12c = aFail2; rfl
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
  have ht20r4 : t20.reg 4 = some (sp - 8) := by
    have h18r4 : t18.reg 4 = some (sp - 8) := by
      rw [ht18]; show t17.reg 4 = some (sp - 8); exact ht17r4
    have h19r4 : t19.reg 4 = some (sp - 8) := by
      rw [ht19]
      show ({ t18 with rip := base + 0x225 }).reg 4 = some (sp - 8)
      rw [reg_with_rip]; exact h18r4
    rw [ht20]
    show ({ t19 with rip := (base + 0x12c) }).reg 4 = some (sp - 8)
    rw [reg_with_rip]; exact h19r4
  have ht20mem : t20.mem = m2 := by
    have h18m : t18.mem = m2 := by
      rw [ht18]; show t17.mem = m2; exact ht17mem
    have h19m : t19.mem = m2 := by
      rw [ht19]
      show ({ t18 with rip := base + 0x225 }).mem = m2
      rw [mem_with_rip]; exact h18m
    rw [ht20]
    show ({ t19 with rip := (base + 0x12c) }).mem = m2
    rw [mem_with_rip]; exact h19m
  have ht20room : StackRoom t20 1 := by
    intro sp' hsp' i hi v
    have hi0 : i = 0 := by omega
    subst hi0
    have hsp'' : sp' = sp - 8 := by
      rw [ht20r4] at hsp'
      exact (Option.some.inj hsp').symm
    subst hsp''
    have h1 : (wr64 s.mem (sp - 16) v).isSome := by
      have h1 : (wr64 s.mem ((sp + W - 16) % W) v).isSome :=
        hroom sp hr4 1 (by decide) v
      rwa [hsp16mod] at h1
    rw [ht20mem]
    show (wr64 m2 ((sp - 8 + W - 8) % W) v).isSome
    rw [haddr2]
    exact wrN_some_of_wshape _ s.mem m2 hw2 _ h1
  refine ⟨t20, hrun, ht20rip, ht20r4, ?_, ht20room, ?_, ?_, ?_⟩
  · rw [ht20mem]; exact hw2
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h18i : t18.reg i = s.reg i := by
      rw [ht18]; show t17.reg i = s.reg i
      exact ht17fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h19i : t19.reg i = s.reg i := by
      rw [ht19]
      show ({ t18 with rip := base + 0x225 }).reg i = s.reg i
      rw [reg_with_rip]; exact h18i
    rw [ht20]
    show ({ t19 with rip := (base + 0x12c) }).reg i = s.reg i
    rw [reg_with_rip]; exact h19i
  · have h18l : t18.regs.length = 16 := by
      rw [ht18]; show t17.regs.length = 16; exact ht17len
    have h19l : t19.regs.length = 16 := by
      rw [ht19]
      show ({ t18 with rip := base + 0x225 }).regs.length = 16
      rw [len_with_rip]; exact h18l
    rw [ht20]
    show ({ t19 with rip := (base + 0x12c) }).regs.length = 16
    rw [len_with_rip]; exact h19l
  · have h18k : t18.k = (Kern.next s.k).2 := by
      rw [ht18]; show t17.k = (Kern.next s.k).2; exact ht17k
    have h19k : t19.k = (Kern.next s.k).2 := by
      rw [ht19]
      show ({ t18 with rip := base + 0x225 }).k = (Kern.next s.k).2
      exact h18k
    rw [ht20]
    show ({ t19 with rip := (base + 0x12c) }).k = (Kern.next s.k).2
    exact h19k

/-- Words-table `mmap` refusal exits 2. -/
theorem parser_mmap1_nomem_exit2 {c nw sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp40 : 40 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', -, hroom', -, -, -⟩ :=
    parser_mmap1_nomem hrip hr6 hr10 hr4 hr7 hr9 hroom (by omega)
      hspW hnw32 hnw0 hhead hlen16
  obtain ⟨k, hk⟩ := exit2_of_fail2 (sp := sp - 8) (by omega) (by omega)
    s' hrip' hr4' (run_regs_ok hrun hregs) hroom'
  exact ⟨20 + k, by rw [run_add stage1Im 20 k, hrun, hk]⟩

set_option maxHeartbeats 0 in
/-- The lengths-scratch `mmap`, oracle-refusal case: from
`base + 0x22d` to `aFail2` in twelve steps when the oracle
head is `.nomem`. -/
theorem parser_mmap2_nomem {L sp1 : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail2 ∧ s'.reg 4 = some sp1 ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 1 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 := by
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
  have hsc : X86Env.syscall t8.k t8.mem 9 0 L PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next s.k).2 m1) := by
    rw [ht8k, ht8mem]
    exact syscall_mmap_nomem hL hhead
  have e9 : X86Int.step stage1Im t8 =
      .ok { ((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1, rip := t8.rip + 2 } :=
    step_syscall_ret (by rw [ht8rip]; exact hc.p249) ht8r0 ht8r7
      ht8r6 ht8r2 ht8r10 ht8r8 ht8r9 hsc
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { ((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1, rip := t8.rip + 2 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x24b := by
    rw [ht9]; show t8.rip + 2 = base + 0x24b; rw [ht8rip]
  have ht9r0 : t9.reg 0 = some (X86Env.neg ENOMEM) := by
    rw [ht9]
    show (((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11
      none).reg 0 = some (X86Env.neg ENOMEM)
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht8len]; decide)]
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t9.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht9]
    show (((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht8fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht9r4 : t9.reg 4 = some sp1 := by
    rw [ht9]
    show (((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11
      none).reg 4 = some sp1
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide)]
    exact ht8r4
  have ht9mem : t9.mem = m1 := by
    rw [ht9]
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (((t8.set 0 (some (X86Env.neg ENOMEM))).set 1 (some (t8.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht8len
  have ht9k : t9.k = (Kern.next s.k).2 := by
    rw [ht9]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2, rip := t9.rip + 3 } :=
    step_test (by rw [ht9rip]; exact hc.p24b) ht9r0 ht9r0
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2, rip := t9.rip + 3 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x24e := by
    rw [ht10]; show t9.rip + 3 = base + 0x24e; rw [ht9rip]
  have hmsb : msb (X86Env.neg ENOMEM) = true := by
    unfold msb X86Env.neg ENOMEM W
    decide
  have ht10fl : X86Int.cond t10.fl 9 = some false := by
    have hsf : t10.fl.sf = some true := by
      rw [ht10]
      show (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf = some true
      have h1 : (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf = some (msb (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x250 } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p24e) ht10fl
    rw [ht10rip] at h
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x250 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x250 := by rw [ht11]
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11 with rip := (base + 0x12c) } := by
    have h := step_jmp (im := stage1Im) (s := t11)
      (by rw [ht11rip]; exact hc.p250)
    simpa using h
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11 with rip := (base + 0x12c) } := ⟨_, rfl⟩
  have ht12rip : t12.rip = aFail2 := by
    rw [ht12]; show base + 0x12c = aFail2; rfl
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
  have ht12r4 : t12.reg 4 = some sp1 := by
    have h10r4 : t10.reg 4 = some sp1 := by
      rw [ht10]; show t9.reg 4 = some sp1; exact ht9r4
    have h11r4 : t11.reg 4 = some sp1 := by
      rw [ht11]
      show ({ t10 with rip := base + 0x250 }).reg 4 = some sp1
      rw [reg_with_rip]; exact h10r4
    rw [ht12]
    show ({ t11 with rip := (base + 0x12c) }).reg 4 = some sp1
    rw [reg_with_rip]; exact h11r4
  have ht12mem : t12.mem = m1 := by
    have h10m : t10.mem = m1 := by
      rw [ht10]; show t9.mem = m1; exact ht9mem
    have h11m : t11.mem = m1 := by
      rw [ht11]
      show ({ t10 with rip := base + 0x250 }).mem = m1
      rw [mem_with_rip]; exact h10m
    rw [ht12]
    show ({ t11 with rip := (base + 0x12c) }).mem = m1
    rw [mem_with_rip]; exact h11m
  have h4eq : t12.reg 4 = s.reg 4 := by rw [ht12r4, hr4]
  refine ⟨t12, hrun, ht12rip, ht12r4, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht12mem]; exact hw1
  · exact StackRoom_of_wshape (by rw [ht12mem]; exact hw1) h4eq hroom
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i
      exact ht9fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h11i : t11.reg i = s.reg i := by
      rw [ht11]
      show ({ t10 with rip := base + 0x250 }).reg i = s.reg i
      rw [reg_with_rip]; exact h10i
    rw [ht12]
    show ({ t11 with rip := (base + 0x12c) }).reg i = s.reg i
    rw [reg_with_rip]; exact h11i
  · have h10l : t10.regs.length = 16 := by
      rw [ht10]; show t9.regs.length = 16; exact ht9len
    have h11l : t11.regs.length = 16 := by
      rw [ht11]
      show ({ t10 with rip := base + 0x250 }).regs.length = 16
      rw [len_with_rip]; exact h10l
    rw [ht12]
    show ({ t11 with rip := (base + 0x12c) }).regs.length = 16
    rw [len_with_rip]; exact h11l
  · have h10k : t10.k = (Kern.next s.k).2 := by
      rw [ht10]; show t9.k = (Kern.next s.k).2; exact ht9k
    have h11k : t11.k = (Kern.next s.k).2 := by
      rw [ht11]
      show ({ t10 with rip := base + 0x250 }).k = (Kern.next s.k).2
      exact h10k
    rw [ht12]
    show ({ t11 with rip := (base + 0x12c) }).k = (Kern.next s.k).2
    exact h11k

/-- Lengths-scratch `mmap` refusal exits 2. -/
theorem parser_mmap2_nomem_exit2 {L sp1 : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp32 : 32 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', -, hroom', -, -, -⟩ :=
    parser_mmap2_nomem hrip hr13 hr4 hr7 hr9 hroom (by omega) hspW
      hL hhead hlen16
  obtain ⟨k, hk⟩ := exit2_of_fail2 hsp32 hspW s' hrip' hr4'
    (run_regs_ok hrun hregs) hroom'
  exact ⟨12 + k, by rw [run_add stage1Im 12 k, hrun, hk]⟩

/-- Trailing refusal: cursor not at buffer end reaches
`aFail1` in three steps. -/
theorem parser_trailing_cursor_fail {c e : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hne : c ≠ e)
    (hcW : c < W) (heW : e < W) :
    ∃ s', run stage1Im 3 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } :=
    step_cmp (by rw [hrip]; exact hc.p419) hr6 hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x41c := by
    rw [ht1]; show s.rip + 3 = base + 0x41c; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 4 = some false := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .sub c e).2.zf = some false
      rw [aluF_zf]
      simp [sub_fst_ne_zero hcW heW hne]
    have hcc := cond_cc4 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x41e } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p41c) ht1fl
    rw [ht1rip] at h
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x41e } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x41e := by rw [ht2]
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p41e)
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  refine ⟨t3, by rw [show (3 : Nat) = 1 + 2 by omega,
    run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega,
    run_add stage1Im 1 1, r2, r3], ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht3]; show base + 0x124 = aFail1; rfl
  · intro i
    have h1i : t1.reg i = s.reg i := by
      rw [ht1]
      show s.reg i = s.reg i
      rfl
    have h2i : t2.reg i = s.reg i := by
      rw [ht2]
      show ({ t1 with rip := base + 0x41e }).reg i = s.reg i
      rw [reg_with_rip]; exact h1i
    rw [ht3]
    show ({ t2 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h2i
  · have h1m : t1.mem = s.mem := by rw [ht1]
    have h2m : t2.mem = s.mem := by
      rw [ht2]
      show ({ t1 with rip := base + 0x41e }).mem = s.mem
      rw [mem_with_rip]; exact h1m
    rw [ht3]
    show ({ t2 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h2m
  · have h1l : t1.regs.length = s.regs.length := by rw [ht1]
    have h2l : t2.regs.length = s.regs.length := by
      rw [ht2]
      show ({ t1 with rip := base + 0x41e }).regs.length = s.regs.length
      rw [len_with_rip]; exact h1l
    rw [ht3]
    show ({ t2 with rip := (base + 0x124) }).regs.length = s.regs.length
    rw [len_with_rip]; exact h2l
  · have h1k : t1.k = s.k := by rw [ht1]
    have h2k : t2.k = s.k := by
      rw [ht2]
      show ({ t1 with rip := base + 0x41e }).k = s.k
      exact h1k
    rw [ht3]
    show ({ t2 with rip := (base + 0x124) }).k = s.k
    exact h2k

/-- Trailing-cursor refusal exits 1. -/
theorem parser_trailing_cursor_fail_exit1 {c e sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hne : c ≠ e)
    (hcW : c < W) (heW : e < W)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_trailing_cursor_fail hrip hr6 hr15 hne hcW heW
  have hr4' : s'.reg 4 = some sp := by rw [hfr 4]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨3 + k, by rw [run_add stage1Im 3 k, hrun, hk]⟩

/-- Trailing refusal: `main` not below the word count
reaches `aFail1` in five steps. -/
theorem parser_trailing_main_ge {c e mn nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : nw ≤ mn) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 5 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ (∀ i, s'.reg i = s.reg i) ∧
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
  have ht3fl : X86Int.cond t3.fl 2 = some false := by
    have hcf : t3.fl.cf = some false := by
      rw [ht3]
      show (aluF .sub mn nw).2.cf = some false
      exact aluF_sub_cf (by omega)
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
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x428 } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p426) ht3fl
    rw [ht3rip] at h
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x428 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x428 := by rw [ht4]
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p428)
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := (base + 0x124) } := ⟨_, rfl⟩
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  refine ⟨t5, by rw [show (5 : Nat) = 1 + 4 by omega,
    run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega,
    run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega,
    run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega,
    run_add stage1Im 1 1, r4, r5], ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht5]; show base + 0x124 = aFail1; rfl
  · intro i
    have h4i : t4.reg i = s.reg i := by
      rw [ht4]
      show ({ t3 with rip := base + 0x428 }).reg i = s.reg i
      rw [reg_with_rip]; exact ht3fr i
    rw [ht5]
    show ({ t4 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h4i
  · have h4m : t4.mem = s.mem := by
      rw [ht4]
      show ({ t3 with rip := base + 0x428 }).mem = s.mem
      rw [mem_with_rip]; exact ht3mem
    rw [ht5]
    show ({ t4 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h4m
  · have h4l : t4.regs.length = 16 := by
      rw [ht4]
      show ({ t3 with rip := base + 0x428 }).regs.length = 16
      rw [len_with_rip]; exact ht3len
    rw [ht5]
    show ({ t4 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h4l
  · have h4k : t4.k = s.k := by rw [ht4]; exact ht3k
    rw [ht5]
    show ({ t4 with rip := (base + 0x124) }).k = s.k
    exact h4k

/-- `main` out of range exits 1. -/
theorem parser_trailing_main_ge_exit1 {c e mn nw sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : nw ≤ mn) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_trailing_main_ge hrip hr6 hr15 hr12 hr13 hceq hcW heW hmn
      hlen16
  have hr4' : s'.reg 4 = some sp := by rw [hfr 4]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨5 + k, by rw [run_add stage1Im 5 k, hrun, hk]⟩

/-- Trailing refusal: the main word's `ins` slot is
nonzero, reaching `aFail1` in sixteen steps. -/
theorem parser_trailing_slot_fail {c e mn nw T v : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw) (hr14 : s.reg 14 = some T)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : mn < nw) (hTW : T + 64 * mn < W)
    (hslot : rd64 s.mem (T + 64 * mn) = some v) (hv : v ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 16 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
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
  have hld : rd64 t12.mem (ea (T + 64 * mn) 0) = some v := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hTW, ht12mem]; exact hslot
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12.set 0 (some v) with rip := t12.rip + 3 } :=
    step_ld (by rw [ht12rip]; exact hc.p445) ht12r2 hld
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12.set 0 (some v) with rip := t12.rip + 3 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x448 := by
    rw [ht13]; show t12.rip + 3 = base + 0x448; rw [ht12rip]
  have ht13r0 : t13.reg 0 = some v := by
    rw [ht13]
    show ({ t12.set 0 (some v) with rip := t12.rip + 3 }).reg 0 = some v
    rw [reg_with_rip, reg_set_self (by rw [ht12len]; decide)]
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → t13.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht13]
    show ({ t12.set 0 (some v) with rip := t12.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht12fr i hi2
  have ht13mem : t13.mem = s.mem := by
    rw [ht13]
    show ({ t12.set 0 (some v) with rip := t12.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12.set 0 (some v) with rip := t12.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht12len
  have ht13k : t13.k = s.k := by rw [ht13]; exact ht12k
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with fl := (aluF .and v v).2, rip := t13.rip + 3 } :=
    step_test (by rw [ht13rip]; exact hc.p448) ht13r0 ht13r0
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with fl := (aluF .and v v).2, rip := t13.rip + 3 } :=
    ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x44b := by
    rw [ht14]; show t13.rip + 3 = base + 0x44b; rw [ht13rip]
  have ht14fl : X86Int.cond t14.fl 4 = some false := by
    have hzf : t14.fl.zf = some false := by
      rw [ht14]
      show (aluF .and v v).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hv]
    have hcc := cond_cc4 hzf
    simpa using hcc
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14 with rip := base + 0x44d } := by
    have h := step_jcc (im := stage1Im) (s := t14)
      (by rw [ht14rip]; exact hc.p44b) ht14fl
    rw [ht14rip] at h
    simpa using h
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14 with rip := base + 0x44d } := ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x44d := by rw [ht15]
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15 with rip := (base + 0x124) } := by
    have h := step_jmp (im := stage1Im) (s := t15)
      (by rw [ht15rip]; exact hc.p44d)
    simpa using h
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15 with rip := (base + 0x124) } := ⟨_, rfl⟩
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
  refine ⟨t16, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht16]; show base + 0x124 = aFail1; rfl
  · intro i hi0 hi2
    have h14i : t14.reg i = s.reg i := by
      rw [ht14]; show t13.reg i = s.reg i; exact ht13fr i hi0 hi2
    have h15i : t15.reg i = s.reg i := by
      rw [ht15]
      show ({ t14 with rip := base + 0x44d }).reg i = s.reg i
      rw [reg_with_rip]; exact h14i
    rw [ht16]
    show ({ t15 with rip := (base + 0x124) }).reg i = s.reg i
    rw [reg_with_rip]; exact h15i
  · have h14m : t14.mem = s.mem := by
      rw [ht14]; show t13.mem = s.mem; exact ht13mem
    have h15m : t15.mem = s.mem := by
      rw [ht15]
      show ({ t14 with rip := base + 0x44d }).mem = s.mem
      rw [mem_with_rip]; exact h14m
    rw [ht16]
    show ({ t15 with rip := (base + 0x124) }).mem = s.mem
    rw [mem_with_rip]; exact h15m
  · have h14l : t14.regs.length = 16 := by
      rw [ht14]; show t13.regs.length = 16; exact ht13len
    have h15l : t15.regs.length = 16 := by
      rw [ht15]
      show ({ t14 with rip := base + 0x44d }).regs.length = 16
      rw [len_with_rip]; exact h14l
    rw [ht16]
    show ({ t15 with rip := (base + 0x124) }).regs.length = 16
    rw [len_with_rip]; exact h15l
  · have h14k : t14.k = s.k := by
      rw [ht14]; show t13.k = s.k; exact ht13k
    have h15k : t15.k = s.k := by
      rw [ht15]
      show ({ t14 with rip := base + 0x44d }).k = s.k
      exact h14k
    rw [ht16]
    show ({ t15 with rip := (base + 0x124) }).k = s.k
    exact h15k

/-- Main-slot refusal exits 1. -/
theorem parser_trailing_slot_fail_exit1 {c e mn nw T v sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw) (hr14 : s.reg 14 = some T)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : mn < nw) (hTW : T + 64 * mn < W)
    (hslot : rd64 s.mem (T + 64 * mn) = some v) (hv : v ≠ 0)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_trailing_slot_fail hrip hr6 hr15 hr12 hr13 hr14 hceq hcW
      heW hmn hTW hslot hv hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨16 + k, by rw [run_add stage1Im 16 k, hrun, hk]⟩

/-- A parser `call RDC` site with fewer than four bytes left
exits 1: `call_rdc_fail` composed with the `fail1` block.  The
`fail1` slot is the second stack slot, hence `StackRoom s 2`
at entry and `40 ≤ sp`. -/
theorem call_rdc_fail_exit1 {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5))
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp40 : 40 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨n, s', hrun, hrip', hr4', -, -, -, -, hsh⟩ :=
    call_rdc_fail hsite hrip hr6 hr15 hce hcW heW
      (stackRoom_mono hroom (by decide)) hr4 (by omega) hspW hlen16
  have hroom' : StackRoom s' 1 := by
    intro sp' hsp' i hi v
    have hi0 : i = 0 := by omega
    subst hi0
    have hsp'' : sp' = sp - 8 := by
      rw [hr4'] at hsp'
      exact (Option.some.inj hsp').symm
    subst hsp''
    have h8mod : (sp + W - 8 * (1 + 1)) % W = sp - 16 := by
      have h : sp + W - 8 * (1 + 1) = (sp - 16) + W := by omega
      rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have h1 : (wr64 s.mem (sp - 16) v).isSome := by
      have h1 := hroom sp hr4 1 (by decide) v
      rwa [h8mod] at h1
    have haddr : (sp - 8 + W - 8 * (0 + 1)) % W = sp - 16 := by
      have h : sp - 8 + W - 8 * (0 + 1) = (sp - 16) + W := by omega
      rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    rw [haddr]
    exact wrN_some_of_wshape _ s.mem s'.mem hsh _ h1
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode (sp := sp - 8)
    (by omega) (by omega) s' hrip' hr4' (run_regs_ok hrun hregs)
    hroom'
  exact ⟨n + k, by rw [run_add stage1Im n k, hrun, hk]⟩

/-- Word-header `RDC` refusal, first `u32` field (`ins`): a word
body entered with fewer than four bytes left exits 1 at the very
first `call RDC` (`base + 0x2d6`).  Refusal mirror of
`parser_word_step`/`parser_word_fields` with only the branch
condition inverted. -/
theorem parser_word_step_field0_short_exit1 {c e sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hshort : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp40 : 40 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 :=
  call_rdc_fail_exit1 stage1_parserCode.p2d6 hrip hr6 hr15 hshort
    hcW heW hroom hr4 hsp40 hspW hlen16 hregs

/-- Condition outcome for `cmp op, k` with `k < op` under cc6. -/
theorem fl_ja {op k : Nat} (hop : op < W) (hk : k < W)
    (h : k < op) :
    X86Int.cond (aluF .sub op (k % W)).2 6 = some false := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hcf : (aluF .sub op (k % W)).2.cf = some false := by
    rw [hkmod]; exact aluF_sub_cf (by omega)
  have hzf : (aluF .sub op (k % W)).2.zf = some false := by
    rw [aluF_zf, hkmod]
    have hbeq : ((aluF .sub op k).1 == 0) = false := by
      cases h2 : ((aluF .sub op k).1 == 0) with
      | false => rfl
      | true =>
        rw [beq_iff_eq] at h2
        exact absurd h2 (sub_fst_ne_zero hop hk (by omega))
    rw [hbeq]
  simp [X86Int.cond, hcf, hzf]

/-- Length dispatch, invalid opcodes `57 ≤ op < 129`, `op ≠ 128`:
the chain reaches the `129` lower-bound test and falls to
`aFail1` in fifteen steps. -/
theorem parser_disp_invalid_lo {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 57 ≤ op) (hhi : op < 129) (h128 : op ≠ 128)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hopW : op < W :=
    Nat.lt_of_le_of_lt (show op ≤ 128 by omega) (by decide)
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip hs1r0
      (fl_jne hopW (by decide) h128)
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    cmpI_jcc_pair hc.p349 hc.p350 hs2rip hs2r0
      (fl_jne (k := 131) hopW (by decide) (by omega))
  have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    cmpI_jcc_pair hc.p357 hc.p35e hs3rip hs3r0
      (fl_jne (k := 132) hopW (by decide) (by omega))
  have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    cmpI_jcc_pair hc.p365 hc.p36c hs4rip hs4r0
      (fl_jne (k := 133) hopW (by decide) (by omega))
  have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
  obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
    cmpI_jcc_pair hc.p373 hc.p37a hs5rip hs5r0
      (fl_jne (k := 140) hopW (by decide) (by omega))
  have hs6r0 : s6.reg 0 = some op := by rw [hs6fr 0]; exact hs5r0
  obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
    cmpI_jcc_pair hc.p381 hc.p388 hs6rip hs6r0
      (fl_jb (by decide) hhi)
  have hs7rip' : s7.rip = base + 0x38a := hs7rip
  obtain ⟨s8, hrun8, hs8rip, hs8fr, hs8mem, hs8len, hs8k⟩ :=
    jmp_step hc.p38a hs7rip'
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok s8 := by
    rw [show (15 : Nat) = 2 + 13 by omega, run_add stage1Im 2 13, hrun1,
      show (13 : Nat) = 2 + 11 by omega, run_add stage1Im 2 11, hrun2,
      show (11 : Nat) = 2 + 9 by omega, run_add stage1Im 2 9, hrun3,
      show (9 : Nat) = 2 + 7 by omega, run_add stage1Im 2 7, hrun4,
      show (7 : Nat) = 2 + 5 by omega, run_add stage1Im 2 5, hrun5,
      show (5 : Nat) = 2 + 3 by omega, run_add stage1Im 2 3, hrun6,
      show (3 : Nat) = 2 + 1 by omega, run_add stage1Im 2 1, hrun7,
      hrun8]
  refine ⟨s8, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs8rip]; rfl
  · intro i
    rw [hs8fr i, hs7fr i, hs6fr i, hs5fr i, hs4fr i, hs3fr i, hs2fr i,
      hs1fr i]
  · rw [hs8mem, hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
  · rw [hs8len, hs7len, hs6len, hs5len, hs4len, hs3len, hs2len, hs1len]
    exact hlen16
  · rw [hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Length dispatch, invalid opcodes `141 ≤ op`: the chain
reaches the `139` upper-bound test and falls to `aFail1` in
seventeen steps. -/
theorem parser_disp_invalid_hi {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 141 ≤ op) (hopW : op < W)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 17 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip hs1r0
      (fl_jne (k := 128) hopW (by decide) (by omega))
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    cmpI_jcc_pair hc.p349 hc.p350 hs2rip hs2r0
      (fl_jne (k := 131) hopW (by decide) (by omega))
  have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    cmpI_jcc_pair hc.p357 hc.p35e hs3rip hs3r0
      (fl_jne (k := 132) hopW (by decide) (by omega))
  have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    cmpI_jcc_pair hc.p365 hc.p36c hs4rip hs4r0
      (fl_jne (k := 133) hopW (by decide) (by omega))
  have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
  obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
    cmpI_jcc_pair hc.p373 hc.p37a hs5rip hs5r0
      (fl_jne (k := 140) hopW (by decide) (by omega))
  have hs6r0 : s6.reg 0 = some op := by rw [hs6fr 0]; exact hs5r0
  obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
    cmpI_jcc_pair hc.p381 hc.p388 hs6rip hs6r0
      (fl_jae (by decide) (by omega))
  have hs7r0 : s7.reg 0 = some op := by rw [hs7fr 0]; exact hs6r0
  obtain ⟨s8, hrun8, hs8rip, hs8fr, hs8mem, hs8len, hs8k⟩ :=
    cmpI_jcc_pair hc.p38f hc.p396 hs7rip hs7r0
      (fl_ja hopW (by decide) (by omega))
  have hs8rip' : s8.rip = base + 0x398 := hs8rip
  obtain ⟨s9, hrun9, hs9rip, hs9fr, hs9mem, hs9len, hs9k⟩ :=
    jmp_step hc.p398 hs8rip'
  have hrun : run stage1Im 17 (Out.ok s) = Out.ok s9 := by
    rw [show (17 : Nat) = 2 + 15 by omega, run_add stage1Im 2 15, hrun1,
      show (15 : Nat) = 2 + 13 by omega, run_add stage1Im 2 13, hrun2,
      show (13 : Nat) = 2 + 11 by omega, run_add stage1Im 2 11, hrun3,
      show (11 : Nat) = 2 + 9 by omega, run_add stage1Im 2 9, hrun4,
      show (9 : Nat) = 2 + 7 by omega, run_add stage1Im 2 7, hrun5,
      show (7 : Nat) = 2 + 5 by omega, run_add stage1Im 2 5, hrun6,
      show (5 : Nat) = 2 + 3 by omega, run_add stage1Im 2 3, hrun7,
      show (3 : Nat) = 2 + 1 by omega, run_add stage1Im 2 1, hrun8,
      hrun9]
  refine ⟨s9, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs9rip]; rfl
  · intro i
    rw [hs9fr i, hs8fr i, hs7fr i, hs6fr i, hs5fr i, hs4fr i, hs3fr i,
      hs2fr i, hs1fr i]
  · rw [hs9mem, hs8mem, hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem,
      hs1mem]
  · rw [hs9len, hs8len, hs7len, hs6len, hs5len, hs4len, hs3len, hs2len,
      hs1len]
    exact hlen16
  · rw [hs9k, hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Invalid low opcode exits 1 at the dispatch. -/
theorem parser_disp_invalid_lo_exit1 {op sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 57 ≤ op) (hhi : op < 129) (h128 : op ≠ 128)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_disp_invalid_lo hrip hr0 hlo hhi h128 hlen16
  have hr4' : s'.reg 4 = some sp := by rw [hfr 4]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨15 + k, by rw [run_add stage1Im 15 k, hrun, hk]⟩

/-- Invalid high opcode exits 1 at the dispatch. -/
theorem parser_disp_invalid_hi_exit1 {op sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 141 ≤ op) (hopW : op < W)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_disp_invalid_hi hrip hr0 hlo hopW hlen16
  have hr4' : s'.reg 4 = some sp := by rw [hfr 4]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨17 + k, by rw [run_add stage1Im 17 k, hrun, hk]⟩

end SporosX1
