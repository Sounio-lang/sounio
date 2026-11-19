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


end SporosX1
