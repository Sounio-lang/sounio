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

/-- A state that is not at a memory syscall is oracle-legal
vacuously (`OracleLegalAt` is an implication, Defs:66). -/
theorem oracleLegalAt_of_not_atMemSyscall {s : X86Int.St}
    (h : ¬ AtMemSyscall s) : OracleLegalAt s :=
  fun hmem => absurd hmem h

/-- If the instruction at `rip` is not the syscall
instruction, the state is not at a memory syscall. -/
theorem not_atMemSyscall_of_stage1Im_ne_syscall {s : X86Int.St}
    (h : stage1Im s.rip ≠ some (.syscall, 2)) :
    ¬ AtMemSyscall s :=
  fun hmem => h hmem.1

theorem oracleLegalAt_of_stage1Im_ne_syscall {s : X86Int.St}
    (h : stage1Im s.rip ≠ some (.syscall, 2)) :
    OracleLegalAt s :=
  oracleLegalAt_of_not_atMemSyscall
    (not_atMemSyscall_of_stage1Im_ne_syscall h)

/-- The `loaded` milestone (`base + 0x134`, a `mov`) is
oracle-legal: it is not a syscall address
(`ParserCode.p134`). -/
theorem oracleLegalAt_of_rip_loaded {s : X86Int.St}
    (hrip : s.rip = base + 0x134) : OracleLegalAt s := by
  apply oracleLegalAt_of_not_atMemSyscall
  intro hmem
  have h1 : stage1Im (base + 0x134) = some (.syscall, 2) := by
    rw [← hrip]; exact hmem.1
  have h2 : stage1Im (base + 0x134) = some (.mov 15 3, 3) :=
    stage1_parserCode.p134
  rw [h2] at h1
  simp at h1

/-- `OracleAdmits` reduces to the stuck states: every
reachable state whose step is not stuck is already legal
(`oracleLegalAt_of_step_ne_stuck`, Refuse:64). -/
theorem oracleAdmits_of_stuck_cases {s0 : X86Int.St}
    (h : ∀ k s, run stage1Im k (Out.ok s0) = Out.ok s →
      X86Int.step stage1Im s = .stuck → OracleLegalAt s) :
    OracleAdmits s0 := by
  intro k s hr
  by_cases hs : X86Int.step stage1Im s = .stuck
  · exact h k s hr hs
  · exact oracleLegalAt_of_step_ne_stuck hs

/-- `OracleAdmits` from one ok milestone plus legality at
and after it: earlier states are prefixes of the milestone
(`oracleLegalAt_of_prefix`, Refuse:133). -/
theorem oracleAdmits_of_milestone {s0 sT : X86Int.St} {K : Nat}
    (hT : run stage1Im K (Out.ok s0) = Out.ok sT)
    (hpost : ∀ k s, K ≤ k →
      run stage1Im k (Out.ok s0) = Out.ok s → OracleLegalAt s) :
    OracleAdmits s0 := by
  intro k s hr
  rcases Nat.lt_or_ge k K with hk | hk
  · exact oracleLegalAt_of_prefix hk hr hT
  · exact hpost k s hk hr

/-- A run from `exit` stays at `exit` (cf. `run_stuck`,
Refuse:114). -/
theorem run_exit (n c : Nat) :
    run stage1Im n (.exit c) = .exit c := by
  cases n <;> rfl

/-- Termination implies admission: if the run from `s0`
reaches any exit, every reachable state is oracle-legal.
Strictly earlier states continue to a non-stuck outcome
(`oracleLegalAt_of_continues`, Refuse:120); at or after the
exit step the run is never `ok` again. -/
theorem oracleAdmits_of_exit {s0 : X86Int.St} {K c : Nat}
    (hK : run stage1Im K (Out.ok s0) = .exit c) :
    OracleAdmits s0 := by
  intro k s hr
  rcases Nat.lt_trichotomy k K with hk | hk | hk
  · have h2 : run stage1Im (K - k) (Out.ok s) = .exit c := by
      have h := hK
      rw [show K = k + (K - k) by omega, run_add, hr] at h
      exact h
    refine oracleLegalAt_of_continues (m := K - k) (by omega) ?_
    rw [h2]
    simp
  · subst hk
    rw [hr] at hK
    simp at hK
  · have h2 : run stage1Im k (Out.ok s0) = .exit c := by
      rw [show k = K + (k - K) by omega, run_add, hK, run_exit]
    rw [hr] at h2
    simp at h2

/-- The witness oracle starts with the file-buffer placement
(same proof as in `loader_init_to_loaded`, Loader:5268). -/
theorem stage1Oracle_head? (f : List Nat) (A : Nat) :
    (stage1Oracle f A).head? = some (.at A) := by
  simp [stage1Oracle, List.replicate_succ]

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

/-- Short header exits 1: the header refusal composed with the
`fail1` exit block. -/
theorem parser_header_a_short_exit1 {B L sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL : L < 16)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_header_a_short hrip hr3 hr13 hBL hL hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨6 + k, by rw [run_add stage1Im 6 k, hrun, hk]⟩

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

/-- Bad magic exits 1: the header refusal composed with the
`fail1` exit block. -/
theorem parser_header_a_magic_exit1 {B L sp : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hmagic : leVal (bs.take 4) % 2 ^ 32 ≠ 827478099)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨s', hrun, hrip', hfr, hmem, -, -⟩ :=
    parser_header_a_magic hrip hr3 hr13 hBL hL16 h20 hmagic hlen16
  have hr4' : s'.reg 4 = some sp := by
    rw [hfr 4 (by decide) (by decide) (by decide)]; exact hr4
  have h4eq : s'.reg 4 = s.reg 4 := by rw [hr4', hr4]
  obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32 hspW s'
    hrip' hr4' (run_regs_ok hrun hregs)
    (StackRoom_of_mem hmem h4eq hroom)
  exact ⟨10 + k, by rw [run_add stage1Im 10 k, hrun, hk]⟩

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

/-- Header classification for arbitrary bytes: from `loaded`,
either the run exits 1 (short file, bad magic or zero
`nwords`), or it reaches the data-loop head with the three
header fields in `r12`, `r10` and `rcx`.  The twenty-byte
window is the one the loaded buffer exposes. -/
theorem parser_header_classify {B L sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W)
    (hwin : ∃ bs, s.mem.rdN B 20 = some bs)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    (∃ k, run stage1Im k (Out.ok s) = Out.exit 1) ∨
    (∃ bs k s', s.mem.rdN B 20 = some bs ∧ 16 ≤ L ∧
      leVal (bs.take 4) % 2 ^ 32 = 827478099 ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 10 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0 ∧
      s'.reg 1 = some (leVal ((bs.drop 12).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some (B + 16) ∧ s'.reg 15 = some (B + L) ∧
      s'.reg 3 = some B ∧ s'.reg 13 = some L ∧
      s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) := by
  obtain ⟨bs, h20⟩ := hwin
  by_cases hL : L < 16
  · exact Or.inl (parser_header_a_short_exit1 hrip hr3 hr13 hBL hL
      hlen16 hr4 hsp32 hspW hregs hroom)
  · have hL16 : 16 ≤ L := by omega
    by_cases hmagic : leVal (bs.take 4) % 2 ^ 32 = 827478099
    · obtain ⟨n1, s1, hrun1, hs1rip, hs1r15, hs1r6, hs1fr, hs1mem,
          hs1len, hs1k⟩ :=
        parser_header_a hrip hr3 hr13 hBL hL16 h20 hmagic hlen16
      have hs1r4 : s1.reg 4 = some sp := by
        rw [hs1fr 4 (by decide) (by decide) (by decide)]; exact hr4
      have hs1r3 : s1.reg 3 = some B := by
        rw [hs1fr 3 (by decide) (by decide) (by decide)]; exact hr3
      have hs1r13 : s1.reg 13 = some L := by
        rw [hs1fr 13 (by decide) (by decide) (by decide)]; exact hr13
      have h20_1 : s1.mem.rdN B 20 = some bs := by
        rw [hs1mem]; exact h20
      by_cases hnw0 : leVal ((bs.drop 8).take 4) % 2 ^ 32 = 0
      · have h4eq : s1.reg 4 = s.reg 4 := by rw [hs1r4, hr4]
        obtain ⟨k, hk⟩ := parser_header_b_nwords_zero_exit1 hs1rip
          hs1r6 hBL hL16 h20_1 hnw0 hs1len hs1r4 hsp32 hspW
          (run_regs_ok hrun1 hregs)
          (StackRoom_of_mem hs1mem h4eq hroom)
        exact Or.inl ⟨n1 + k, by
          rw [run_add stage1Im n1 k, hrun1, hk]⟩
      · obtain ⟨n2, s2, hrun2, hs2rip, hs2r12, hs2r10, hs2r1, hs2r6,
            hs2fr, hs2mem, hs2len, hs2k⟩ :=
          parser_header_b hs1rip hs1r6 hBL hL16 h20_1 hnw0 hs1len
        have hs2r15 : s2.reg 15 = some (B + L) := by
          rw [hs2fr 15 (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide)]
          exact hs1r15
        have hs2r3 : s2.reg 3 = some B := by
          rw [hs2fr 3 (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide)]
          exact hs1r3
        have hs2r13 : s2.reg 13 = some L := by
          rw [hs2fr 13 (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide)]
          exact hs1r13
        have hs2r4 : s2.reg 4 = some sp := by
          rw [hs2fr 4 (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide)]
          exact hs1r4
        exact Or.inr ⟨bs, n1 + n2, s2, h20, hL16, hmagic,
          by rw [run_add stage1Im n1 n2, hrun1, hrun2],
          hs2rip, hs2r12, hs2r10, hnw0, hs2r1, hs2r6, hs2r15,
          hs2r3, hs2r13, hs2r4,
          fun i hi0 hi1 hi2 hi6 hi10 hi12 hi15 =>
            (hs2fr i hi0 hi1 hi2 hi6 hi10 hi12).trans
              (hs1fr i hi0 hi6 hi15),
          hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k⟩
    · exact Or.inl (parser_header_a_magic_exit1 hrip hr3 hr13 hBL
        hL16 h20 hmagic hlen16 hr4 hsp32 hspW hregs hroom)

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

/-- Data-loop classification for arbitrary bytes: from the loop
head, either all `n` length-prefixed segments fit and the loop
reaches `base + 0x1c7` describing them by a `DataSeq`, or a
short length field or a payload overrun reaches `aFail1`.
The only memory hypothesis is that a u32 is readable wherever
four bytes remain before `e`. -/
theorem parser_data_loop_classify :
    ∀ (n : Nat) {c e : Nat} {s : X86Int.St},
      s.rip = base + 0x18f → s.reg 1 = some n →
      s.reg 6 = some c → s.reg 15 = some e → c ≤ e → e < W →
      n < W →
      (∀ a, c ≤ a → a + 4 ≤ e → ∃ d, rd32 s.mem a = some d) →
      s.regs.length = 16 →
      (∃ ds c' k s', ds.length = n ∧ DataSeq s.mem c ds c' ∧
          c' ≤ e ∧ run stage1Im k (Out.ok s) = Out.ok s' ∧
          s'.rip = base + 0x1c7 ∧ s'.reg 1 = some 0 ∧
          s'.reg 6 = some c' ∧ s'.reg 15 = some e ∧
          (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 →
            s'.reg i = s.reg i) ∧
          s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) ∨
      (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
          s'.rip = aFail1 ∧ s'.reg 15 = some e ∧
          (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 →
            s'.reg i = s.reg i) ∧
          s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) := by
  intro n
  induction n with
  | zero =>
    intro c e s hrip hr1 hr6 hr15 hce heW _hnW _hrd hlen16
    obtain ⟨k, s', hrun, hsrip, hsr1, hsr6, hsr15, hsfr, hsmem,
        hslen, hsk⟩ :=
      parser_data_loop_ok (DataSeq.nil c) hrip hr1 hr6 hr15 hce heW
        (by simp [W]) hlen16
    exact Or.inl ⟨[], c, k, s', rfl, DataSeq.nil c, hce, hrun,
      hsrip, hsr1, hsr6, hsr15, hsfr, hsmem, hslen, hsk⟩
  | succ n ih =>
    intro c e s hrip hr1 hr6 hr15 hce heW hnW hrd hlen16
    by_cases hshort : e < c + 4
    · obtain ⟨s', hrun, hsrip, hsfr, hsmem, hslen, hsk⟩ :=
        parser_data_step_short hrip hr1 hr6 hr15 (by omega) hce
          hshort heW hlen16
      have hs15 : s'.reg 15 = some e := by
        rw [hsfr 15 (by decide)]; exact hr15
      exact Or.inr ⟨7, s', hrun, hsrip, hs15,
        fun i _ _ hi2 _ => hsfr i hi2, hsmem, hslen, hsk⟩
    · have hc4 : c + 4 ≤ e := by omega
      obtain ⟨d, hd⟩ := hrd c (Nat.le_refl c) hc4
      by_cases hover : e < c + 4 + d
      · obtain ⟨s', hrun, hsrip, hsfr, hsmem, hslen, hsk⟩ :=
          parser_data_step_overrun hrip hr1 hr6 hr15 (by omega) hc4
            hd hover heW hlen16
        have hs15 : s'.reg 15 = some e := by
          rw [hsfr 15 (by decide) (by decide) (by decide)]
          exact hr15
        exact Or.inr ⟨14, s', hrun, hsrip, hs15,
          fun i hi0 _ hi2 hi6 => hsfr i hi0 hi2 hi6,
          hsmem, hslen, hsk⟩
      · have hcd : c + 4 + d ≤ e := by omega
        obtain ⟨s1, hrun1, hs1rip, hs1r1, hs1r6, hs1r15, hs1fr,
            hs1mem, hs1len, hs1k⟩ :=
          parser_data_step_ok hrip hr1 hr6 hr15 (by omega) hnW hc4
            hd hcd heW hlen16
        have hs1r1' : s1.reg 1 = some n := by
          have h : n + 1 - 1 = n := by omega
          rw [h] at hs1r1
          exact hs1r1
        have hrd1 : ∀ a, c + 4 + d ≤ a → a + 4 ≤ e →
            ∃ d', rd32 s1.mem a = some d' := by
          intro a ha hb
          obtain ⟨d', hd'⟩ := hrd a (by omega) hb
          exact ⟨d', by rw [hs1mem]; exact hd'⟩
        rcases ih hs1rip hs1r1' hs1r6 hs1r15 hcd heW (by omega) hrd1
            hs1len with hleft | hright
        · obtain ⟨ds, c', k, s2, hdslen, hseq, hc'e, hrun2, hs2rip,
              hs2r1, hs2r6, hs2r15, hs2fr, hs2mem, hs2len, hs2k⟩ :=
            hleft
          have hrest : DataSeq s.mem (c + 4 + d) ds c' := by
            rw [hs1mem] at hseq
            exact hseq
          exact Or.inl ⟨d :: ds, c', 16 + k, s2,
            by simp [hdslen], DataSeq.cons hd hrest, hc'e,
            by rw [run_add stage1Im 16 k, hrun1, hrun2],
            hs2rip, hs2r1, hs2r6, hs2r15,
            fun i hi0 hi1 hi2 hi6 =>
              (hs2fr i hi0 hi1 hi2 hi6).trans
                (hs1fr i hi0 hi1 hi2 hi6),
            hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k⟩
        · obtain ⟨k, s2, hrun2, hs2rip, hs2r15, hs2fr, hs2mem,
              hs2len, hs2k⟩ := hright
          exact Or.inr ⟨16 + k, s2,
            by rw [run_add stage1Im 16 k, hrun1, hrun2],
            hs2rip, hs2r15,
            fun i hi0 hi1 hi2 hi6 =>
              (hs2fr i hi0 hi1 hi2 hi6).trans
                (hs1fr i hi0 hi1 hi2 hi6),
            hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k⟩

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

/-- Header, data loop and guard, classified together: from
`loaded`, either exit 1, or `base + 0x1ec` with the data
segments described by a `DataSeq` and the `20 * nwords` guard
satisfied. -/
theorem parser_front_to_guard_classify {B L sp : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W)
    (hwin : ∃ bs, s.mem.rdN B 20 = some bs)
    (hrd : ∀ a, B + 16 ≤ a → a + 4 ≤ B + L →
      ∃ d, rd32 s.mem a = some d)
    (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp) (hsp32 : 32 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom : StackRoom s 1) :
    (∃ k, run stage1Im k (Out.ok s) = Out.exit 1) ∨
    (∃ bs ds c k s', s.mem.rdN B 20 = some bs ∧ 16 ≤ L ∧
      leVal (bs.take 4) % 2 ^ 32 = 827478099 ∧
      ds.length = leVal ((bs.drop 12).take 4) % 2 ^ 32 ∧
      DataSeq s.mem (B + 16) ds c ∧ c ≤ B + L ∧
      20 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) ≤ B + L - c ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x1ec ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 10 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0 ∧
      s'.reg 1 = some 0 ∧ s'.reg 6 = some c ∧
      s'.reg 15 = some (B + L) ∧ s'.reg 3 = some B ∧
      s'.reg 13 = some L ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) := by
  rcases parser_header_classify hrip hr3 hr13 hBL hwin hlen16 hr4
      hsp32 hspW hregs hroom with hexit | hhead
  · exact Or.inl hexit
  · obtain ⟨bs, k1, s1, h20, hL16, hmagic, hrun1, hs1rip, hs1r12, hs1r10,
        hnw0, hs1r1, hs1r6, hs1r15, hs1r3, hs1r13, hs1r4, hs1fr,
        hs1mem, hs1len, hs1k⟩ := hhead
    have hs1regs : RegsOk s1 := run_regs_ok hrun1 hregs
    have h4eq1 : s1.reg 4 = s.reg 4 := by rw [hs1r4, hr4]
    have hs1room : StackRoom s1 1 :=
      StackRoom_of_mem hs1mem h4eq1 hroom
    have hndW : leVal ((bs.drop 12).take 4) % 2 ^ 32 < W := by
      have h := Nat.mod_lt (leVal ((bs.drop 12).take 4))
        (show 0 < 2 ^ 32 by decide)
      unfold W
      omega
    have hnw32 : leVal ((bs.drop 8).take 4) % 2 ^ 32 < 2 ^ 32 :=
      Nat.mod_lt _ (by decide)
    have hrd1 : ∀ a, B + 16 ≤ a → a + 4 ≤ B + L →
        ∃ d, rd32 s1.mem a = some d := by
      intro a ha hb
      obtain ⟨d, hd⟩ := hrd a ha hb
      exact ⟨d, by rw [hs1mem]; exact hd⟩
    rcases parser_data_loop_classify
        (leVal ((bs.drop 12).take 4) % 2 ^ 32) hs1rip hs1r1 hs1r6
        hs1r15 (by omega) (by omega) hndW hrd1 hs1len with
      hdata | hdatafail
    · obtain ⟨ds, c, k2, s2, hdslen, hseq, hcL, hrun2, hs2rip,
          hs2r1, hs2r6, hs2r15, hs2fr, hs2mem, hs2len, hs2k⟩ :=
        hdata
      have hs2r10 : s2.reg 10 =
          some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
        rw [hs2fr 10 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r10
      have hs2r12 : s2.reg 12 =
          some (leVal ((bs.drop 4).take 4) % 2 ^ 32) := by
        rw [hs2fr 12 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r12
      have hs2r3 : s2.reg 3 = some B := by
        rw [hs2fr 3 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r3
      have hs2r13 : s2.reg 13 = some L := by
        rw [hs2fr 13 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r13
      have hs2r4 : s2.reg 4 = some sp := by
        rw [hs2fr 4 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r4
      have hseq0 : DataSeq s.mem (B + 16) ds c := by
        rw [hs1mem] at hseq
        exact hseq
      by_cases hguard :
          20 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) ≤ B + L - c
      · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
          parser_guard_ok hs2rip hs2r10 hs2r6 hs2r15 hnw32 hcL
            (by omega) hguard hs2len
        have hreg3 : ∀ {i : Nat} {v : Nat}, i ≠ 0 → i ≠ 2 →
            s2.reg i = some v → s3.reg i = some v := by
          intro i v hi0 hi2 hv
          rw [hs3fr i hi0 hi2]
          exact hv
        exact Or.inr ⟨bs, ds, c, k1 + k2 + 11, s3, h20, hL16,
          hmagic, hdslen, hseq0, hcL, hguard,
          by rw [show k1 + k2 + 11 = (k1 + k2) + 11 by omega,
            run_add stage1Im (k1 + k2) 11,
            show run stage1Im (k1 + k2) (Out.ok s) = Out.ok s2 from by
              rw [run_add stage1Im k1 k2, hrun1, hrun2], hrun3],
          hs3rip,
          hreg3 (by decide) (by decide) hs2r12,
          hreg3 (by decide) (by decide) hs2r10, hnw0,
          hreg3 (by decide) (by decide) hs2r1,
          hreg3 (by decide) (by decide) hs2r6,
          hreg3 (by decide) (by decide) hs2r15,
          hreg3 (by decide) (by decide) hs2r3,
          hreg3 (by decide) (by decide) hs2r13,
          hreg3 (by decide) (by decide) hs2r4,
          fun i hi0 hi1 hi2 hi6 hi10 hi12 hi15 =>
            (hs3fr i hi0 hi2).trans
              ((hs2fr i hi0 hi1 hi2 hi6).trans
                (hs1fr i hi0 hi1 hi2 hi6 hi10 hi12 hi15)),
          (hs3mem.trans hs2mem).trans hs1mem, hs3len,
          (hs3k.trans hs2k).trans hs1k⟩
      · have hguardf : B + L - c <
            20 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by omega
        have hmem2 : s2.mem = s.mem := hs2mem.trans hs1mem
        have h4eq2 : s2.reg 4 = s.reg 4 := by rw [hs2r4, hr4]
        obtain ⟨k, hk⟩ := parser_guard_fail_exit1 hs2rip hs2r10
          hs2r6 hs2r15 hnw32 hcL (by omega) hguardf hs2len hs2r4
          hsp32 hspW (run_regs_ok hrun2 hs1regs)
          (StackRoom_of_mem hmem2 h4eq2 hroom)
        exact Or.inl ⟨k1 + k2 + k, by
          rw [show k1 + k2 + k = (k1 + k2) + k by omega,
            run_add stage1Im (k1 + k2) k,
            show run stage1Im (k1 + k2) (Out.ok s) = Out.ok s2 from by
              rw [run_add stage1Im k1 k2, hrun1, hrun2], hk]⟩
    · obtain ⟨k2, s2, hrun2, hs2rip, _hs2r15, hs2fr, hs2mem, hs2len,
          hs2k⟩ := hdatafail
      have hs2r4 : s2.reg 4 = some sp := by
        rw [hs2fr 4 (by decide) (by decide) (by decide)
          (by decide)]
        exact hs1r4
      have hmem2 : s2.mem = s.mem := hs2mem.trans hs1mem
      have h4eq2 : s2.reg 4 = s.reg 4 := by rw [hs2r4, hr4]
      obtain ⟨k, hk⟩ := exit1_of_fail1 stage1_refuseCode hsp32
        hspW s2 hs2rip hs2r4 (run_regs_ok hrun2 hs1regs)
        (StackRoom_of_mem hmem2 h4eq2 hroom)
      exact Or.inl ⟨k1 + k2 + k, by
        rw [show k1 + k2 + k = (k1 + k2) + k by omega,
          run_add stage1Im (k1 + k2) k,
          show run stage1Im (k1 + k2) (Out.ok s) = Out.ok s2 from by
            rw [run_add stage1Im k1 k2, hrun1, hrun2], hk]⟩

set_option maxRecDepth 8192 in
/-- Loaded state through both parser `mmap`s, for arbitrary
bytes: either exit 1 in the front half, or `base + 0x258` with
the words table at `r14`, the lengths scratch at `r11`, the
oracle exhausted, and the file still readable at `A`.  The
table size is bounded by the guard, so both placements fit
the `Room` arena. -/
theorem parser_loaded_to_scratch_classify {f : List Nat}
    {A sp : Nat} {c1 c2 buf : Region} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some A)
    (hr13 : s.reg 13 = some f.length) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (horacle : s.k.oracle =
      [.at (A + arenaFileBytes f),
       .at (A + (arenaFileBytes f + arenaTableBytes f))])
    (hmem : s.mem = [c1, c2, buf]) (hbufb : buf.base = A)
    (hbufhi : ∃ cap, buf.hi = A + cap ∧ cap ≤ arenaFileBytes f)
    (hrdN : s.mem.rdN A f.length = some f)
    (hwin : ∃ bs, s.mem.rdN A 20 = some bs)
    (hfreeInit : ∀ r ∈ ([c1, c2] : Mem),
      A + stage1Mf f ≤ r.base ∨ r.hi ≤ A)
    (hAmod : A % page = 0) (hApos : 0 < A)
    (hAbound : A + stage1Mf f ≤ 2 ^ 47)
    (hAframe : ∀ x k, A ≤ x → x + k ≤ A + f.length →
      x + k ≤ sp - 16 ∨ sp ≤ x)
    (hdisjc : c1.hi ≤ c2.base) (hc2rw : c2.rw = true)
    (hc2lo : c2.base + 8 * 6 ≤ sp - 8) (hc2hi2 : sp - 8 ≤ c2.hi)
    (hlen16 : s.regs.length = 16)
    (hsp80 : 80 ≤ sp) (hspW : sp < W)
    (hregs : RegsOk s) (hroom10 : StackRoom s 10) :
    (∃ k, run stage1Im k (Out.ok s) = Out.exit 1) ∨
    (∃ bs ds c k s', s.mem.rdN A 20 = some bs ∧ 16 ≤ f.length ∧
      leVal (bs.take 4) % 2 ^ 32 = 827478099 ∧
      ds.length = leVal ((bs.drop 12).take 4) % 2 ^ 32 ∧
      DataSeq s.mem (A + 16) ds c ∧ c ≤ A + f.length ∧
      20 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) ≤
        A + f.length - c ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x258 ∧
      s'.reg 14 = some (A + arenaFileBytes f) ∧
      s'.reg 11 =
        some (A + (arenaFileBytes f + arenaTableBytes f)) ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0 ∧
      s'.reg 3 = some A ∧ s'.reg 13 = some f.length ∧
      s'.reg 15 = some (A + f.length) ∧
      s'.reg 4 = some (sp - 8) ∧
      rd64 s'.mem (sp - 8) = some (c % W) ∧
      s'.mem.rdN A f.length = some f ∧
      s'.k.oracle = [] ∧
      wshape s'.mem = wshape ([c1, c2, buf] ++
        [⟨A + arenaFileBytes f,
          List.replicate
            (X86Env.pages
              (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
          true⟩,
         ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩]) ∧
      StackRoom s' 6 ∧ s'.regs.length = 16) := by
  generalize hTdef : A + arenaFileBytes f = T
  generalize hSdef : A + (arenaFileBytes f + arenaTableBytes f) = S
  rw [hTdef, hSdef] at horacle
  obtain ⟨capB0, hbufhi0, hbufcap⟩ := hbufhi
  have hLeM : f.length ≤ stage1Mf f := by
    have h1 := pages_ge (65536 + 2 * f.length)
    unfold stage1Mf arenaFileBytes at *
    omega
  have hAe : A + f.length < W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  have hrd : ∀ a, A + 16 ≤ a → a + 4 ≤ A + f.length →
      ∃ d, rd32 s.mem a = some d := by
    intro a ha hb
    have hs := rdN_slice hrdN (o := a - A) (k := 4) (by omega)
    rw [show A + (a - A) = a by omega] at hs
    exact ⟨leVal ((f.drop (a - A)).take 4), by
      show (s.mem.rdN a 4).map leVal = _
      rw [hs]
      rfl⟩
  rcases parser_front_to_guard_classify hrip hr3 hr13 hAe hwin
      hrd hlen16 hr4 (by omega) hspW hregs
      (stackRoom_mono hroom10 (by decide)) with hexit | hfront
  · exact Or.inl hexit
  · obtain ⟨bs, ds, c, k1, sg, h20, hL16, hmagic, hdslen, hseq, hcL,
        hguard, hrun1, hsgrip, hsg12, hsg10, hnw0, _hsg1, hsg6,
        hsg15, hsg3, hsg13, hsg4, hsgfr, hsgmem, hsglen, hsgk⟩ :=
      hfront
    generalize hnwdef : leVal ((bs.drop 8).take 4) % 2 ^ 32 = nw
    rw [hnwdef] at hsg10 hnw0 hguard
    have hcge : A + 16 ≤ c := DataSeq.le hseq
    have hnw32 : nw < 2 ^ 32 := by
      rw [← hnwdef]
      exact Nat.mod_lt _ (by decide)
    have hpgnw : X86Env.pages (64 * nw) ≤ arenaTableBytes f := by
      unfold arenaTableBytes
      apply pages_mono
      omega
    have hTMf : T + X86Env.pages (64 * nw) ≤ A + stage1Mf f := by
      unfold stage1Mf
      omega
    have hSplus : S + X86Env.pages f.length = A + stage1Mf f := by
      unfold stage1Mf arenaScratchBytes
      omega
    have hfree3 : ∀ a n, a + n ≤ A + stage1Mf f → buf.hi ≤ a →
        Mem.free [c1, c2, buf] a n = true := by
      intro a n hn hbuf
      rw [Mem.free, List.all_eq_true]
      intro r hr
      have hr2 : c1 = r ∨ c2 = r ∨ buf = r := by
        rcases (show r = c1 ∨ r = c2 ∨ r = buf by simpa using hr)
          with h | h | h
        · exact Or.inl h.symm
        · exact Or.inr (Or.inl h.symm)
        · exact Or.inr (Or.inr h.symm)
      simp only [Bool.or_eq_true, decide_eq_true_eq]
      rcases hr2 with rfl | rfl | rfl
      · rcases hfreeInit c1 (by simp) with h | h
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
      · rcases hfreeInit c2 (by simp) with h | h
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
      · exact Or.inr hbuf
    have hfreeT : sg.mem.free T (X86Env.pages (64 * nw)) = true := by
      rw [hsgmem, hmem]
      exact hfree3 T _ hTMf (by rw [hbufhi0]; omega)
    have hTpage : T % page = 0 := by
      have h1 : arenaFileBytes f % page = 0 := by
        unfold arenaFileBytes X86Env.pages
        exact Nat.mul_mod_left _ _
      rw [← hTdef, Nat.add_mod, hAmod, h1]
      simp
    have hsgr7 : ∃ v, sg.reg 7 = some v := by
      obtain ⟨v, hv⟩ := hr7
      exact ⟨v, by
        rw [hsgfr 7 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)]
        exact hv⟩
    have hsgr9 : ∃ v, sg.reg 9 = some v := by
      obtain ⟨v, hv⟩ := hr9
      exact ⟨v, by
        rw [hsgfr 9 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)]
        exact hv⟩
    have hsgroom : StackRoom sg 2 :=
      stackRoom_mono
        (StackRoom_of_mem hsgmem (by rw [hsg4, hr4]) hroom10)
        (by decide)
    have hsghead : sg.k.oracle.head? = some (.at T) := by
      rw [hsgk]
      show s.k.oracle.head? = some (.at T)
      rw [horacle]
      rfl
    obtain ⟨s2, m2, hrun2, hs2rip, hs2r14, hs2r4, hs2mem, hm2fr,
        hrd64cur, hs2fr, hs2len, hs2k, hw2⟩ :=
      parser_mmap1_ok hsgrip hsg6 hsg10 hsg4 hsgr7 hsgr9 hsgroom
        (by omega) hspW hnw32 hnw0 hsghead hTpage (by omega)
        (by omega) hfreeT hsglen
    have hrun12 : run stage1Im (k1 + 20) (Out.ok s) = Out.ok s2 := by
      rw [run_add stage1Im k1 20, hrun1, hrun2]
    have hw2' : wshape m2 = wshape s.mem := by
      rw [hsgmem] at hw2
      exact hw2
    have hs2room : StackRoom s2 1 := by
      have hr4' : s2.reg 4 = some (sp - 8 * 1) := by
        have h : sp - 8 * 1 = sp - 8 := by omega
        rw [h]
        exact hs2r4
      refine stackRoom_transport_of_wr hr4 hr4' hroom10 (by decide)
        (by omega) hspW (fun a u hwr => ?_)
      have h1 := wr_some_of_wshape hw2' hwr
      rw [hs2mem]
      exact wr_isSome_append_left h1
    have hnext1 : (Kern.next s.k).2 =
        { s.k with oracle := [.at S] } := by
      cases ho : s.k.oracle with
      | nil => simp [ho] at horacle
      | cons c0 rest0 =>
        have h2 : Kern.next s.k =
            (some c0, { s.k with oracle := rest0 }) := by
          simp [Kern.next, ho]
        rw [ho] at horacle
        obtain ⟨-, hr⟩ := List.cons.inj horacle
        rw [h2, hr]
    have hs2or : s2.k.oracle = [.at S] := by
      have h1 : s2.k = (Kern.next s.k).2 := by rw [hs2k, hsgk]
      rw [h1, hnext1]
    have hs2r13 : s2.reg 13 = some f.length := by
      rw [hs2fr 13 (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)]
      exact hsg13
    have hs2r7 : ∃ v, s2.reg 7 = some v := by
      obtain ⟨v, hv⟩ := hr7
      exact run_reg_some_ne11 _ (by decide) hv hrun12
    have hs2r9 : ∃ v, s2.reg 9 = some v := by
      obtain ⟨v, hv⟩ := hr9
      exact run_reg_some_ne11 _ (by decide) hv hrun12
    have hSpage : S % page = 0 := by
      have h1 : arenaFileBytes f % page = 0 := by
        unfold arenaFileBytes X86Env.pages
        exact Nat.mul_mod_left _ _
      have h2 : arenaTableBytes f % page = 0 := by
        unfold arenaTableBytes X86Env.pages
        exact Nat.mul_mod_left _ _
      have hsum : (arenaFileBytes f + arenaTableBytes f) % page =
          0 := by
        rw [Nat.add_mod, h1, h2]
        simp
      rw [← hSdef, Nat.add_mod, hAmod, hsum]
      simp
    have hfreeS3 : Mem.free [c1, c2, buf] S
        (X86Env.pages f.length) = true :=
      hfree3 S _ (by omega) (by rw [hbufhi0]; omega)
    have hfreeS4 : Mem.free ([c1, c2, buf] ++
          [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]) S
          (X86Env.pages f.length) = true :=
      free_append hfreeS3 (Or.inr (by
        simp only [X86Env.Region.hi, List.length_replicate]
        omega))
    have wshape_append_single : ∀ {x y : Mem} {r : Region},
        wshape x = wshape y → wshape (x ++ [r]) = wshape (y ++ [r]) := by
      intro x y r h
      simp only [wshape, List.map_append] at h ⊢
      rw [h]
    have hwsh2 : wshape s2.mem = wshape ([c1, c2, buf] ++
          [⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
            true⟩]) := by
      rw [hs2mem]
      exact wshape_append_single (by rw [hw2', hmem])
    have hfreeS : s2.mem.free S (X86Env.pages f.length) = true := by
      rw [free_of_wshape hwsh2]
      exact hfreeS4
    obtain ⟨s3, m1, hrun3, hs3rip, hs3r11, hs3r4, hs3mem, hm1fr,
        hs3fr, hs3len, hs3k, hm1sh⟩ :=
      parser_mmap2_ok hs2rip hs2r13 hs2r4 hs2r7 hs2r9 hs2room
        (by omega) (by omega) (by omega)
        (by rw [hs2or]; rfl) hSpage (by omega) (by omega) hfreeS
        hs2len
    have hrunTot : run stage1Im (k1 + 20 + 12) (Out.ok s) =
        Out.ok s3 := by
      rw [show k1 + 20 + 12 = (k1 + 20) + 12 by omega,
        run_add stage1Im (k1 + 20) 12, hrun12, hrun3]
    have hreg3 : ∀ {i : Nat} {v : Nat},
        i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → i ≠ 14 → i ≠ 12 → i ≠ 15 →
        s.reg i = some v → s3.reg i = some v := by
      intro i v hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11 hi14 hi12
        hi15 hv
      rw [hs3fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11,
        hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11 hi14,
        hsgfr i hi0 hi1 hi2 hi6 hi10 hi12 hi15]
      exact hv
    have hs3or : s3.k.oracle = [] := by
      have h1 : (Kern.next s2.k).2.oracle = [] := by
        simp [Kern.next, hs2or]
      rw [hs3k]
      exact h1
    have hm2rd : m2.rdN A f.length = some f := by
      rw [hm2fr A f.length
        (hAframe A f.length (Nat.le_refl _) (by omega)), hsgmem]
      exact hrdN
    have hs2rd : s2.mem.rdN A f.length = some f := by
      rw [hs2mem]
      exact rdN_append_left hm2rd
    have hm1rd : m1.rdN A f.length = some f := by
      have hcond : A + f.length ≤ sp - 8 - 8 ∨ sp - 8 ≤ A := by
        rcases hAframe A f.length (Nat.le_refl _) (by omega) with
          h | h
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
      rw [hm1fr A f.length hcond]
      exact hs2rd
    have hs3rd : s3.mem.rdN A f.length = some f := by
      rw [hs3mem]
      exact rdN_append_left hm1rd
    have hs3rd64 : rd64 s3.mem (sp - 8) = some (c % W) := by
      obtain ⟨bs8, hbs8, hle8⟩ := Option.map_eq_some_iff.mp hrd64cur
      have hm1bs : m1.rdN (sp - 8) 8 = some bs8 := by
        rw [hm1fr (sp - 8) 8 (Or.inr (Nat.le_refl _))]
        exact hbs8
      show (s3.mem.rdN (sp - 8) 8).map leVal = some (c % W)
      rw [hs3mem, rdN_append_left hm1bs]
      simp [hle8]
    have hwsh3 : wshape s3.mem = wshape ([c1, c2, buf] ++
          [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
           ⟨S, List.replicate (X86Env.pages f.length) 0,
            true⟩]) := by
      have e1 : wshape s3.mem = wshape (s2.mem ++
          [⟨S, List.replicate (X86Env.pages f.length) 0,
            true⟩]) := by
        rw [hs3mem]
        exact wshape_append_single hm1sh
      rw [e1]
      have e2 := wshape_append_single (r :=
        ⟨S, List.replicate (X86Env.pages f.length) 0, true⟩) hwsh2
      rw [e2]
      simp [List.append_assoc]
    have hs3room : StackRoom s3 6 :=
      stackRoom_of_wshape_c2 hwsh3 hs3r4 hc2rw hdisjc hc2lo hc2hi2
        (by omega)
    rw [← hnwdef] at hguard hnw0 hwsh3
    exact Or.inr ⟨bs, ds, c, k1 + 20 + 12, s3, h20, hL16, hmagic,
      hdslen,
      hseq, hcL, hguard, hrunTot, hs3rip,
      by
        rw [hs3fr 14 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide)]
        exact hs2r14,
      hs3r11,
      by
        rw [hs3fr 12 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide),
          hs2fr 12 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)]
        exact hsg12,
      hnw0,
      hreg3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)
        (by decide) hr3,
      hreg3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide)
        (by decide) hr13,
      by
        rw [hs3fr 15 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide),
          hs2fr 15 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)]
        exact hsg15,
      hs3r4, hs3rd64, hs3rd, hs3or, hwsh3, hs3room, hs3len⟩


/-- Parser setup and word-loop entry for a nonzero word
count: from `base + 0x258` to the first word body at
`base + 0x2d6`, with the saved cursor restored and the loop
counter loaded. -/
theorem parser_setup_to_word_body {sp1 c T B S e mn v : Nat}
    {w1 w2 w3 w4 w5 w6 : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x258) (hr4 : s.reg 4 = some sp1)
    (hr14 : s.reg 14 = some T) (hr3 : s.reg 3 = some B)
    (hr15 : s.reg 15 = some e) (hr11 : s.reg 11 = some S)
    (hr12 : s.reg 12 = some mn)
    (hspW : sp1 + 56 < W) (hBW : B + 8 < W)
    (hcur : rd64 s.mem sp1 = some c)
    (hnw : rd64 s.mem (B + 8) = some v)
    (hp1 : rd64 s.mem (sp1 + 8) = some w1)
    (hp2 : rd64 s.mem (sp1 + 16) = some w2)
    (hp3 : rd64 s.mem (sp1 + 24) = some w3)
    (hp4 : rd64 s.mem (sp1 + 32) = some w4)
    (hp5 : rd64 s.mem (sp1 + 40) = some w5)
    (hp6 : rd64 s.mem (sp1 + 48) = some w6)
    (hnw0 : v % 2 ^ 32 ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 14 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2d6 ∧
      s'.reg 1 = some (v % 2 ^ 32) ∧
      s'.reg 6 = some c ∧ s'.reg 7 = some T ∧
      s'.reg 13 = some (v % 2 ^ 32) ∧
      s'.reg 4 = some (sp1 + 56) ∧
      s'.reg 15 = some e ∧ s'.reg 3 = some B ∧
      s'.reg 11 = some S ∧ s'.reg 12 = some mn ∧
      s'.reg 14 = some T ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  obtain ⟨s1, hrun1, hs1rip, hs1r6, hs1r7, hs1r13, hs1r4, _hs1r0,
      hs1mem, hs1fr, hs1len, hs1k⟩ :=
    parser_setup_ok hrip hr4 hr14 hr3 hspW hBW hcur hnw hp1 hp2
      hp3 hp4 hp5 hp6 hlen16
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
  have hs1r12 : s1.reg 12 = some mn := by
    rw [hs1fr 12 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr12
  have hs1r14 : s1.reg 14 = some T := by
    rw [hs1fr 14 (by decide) (by decide) (by decide) (by decide)
      (by decide)]
    exact hr14
  obtain ⟨s2, hrun2, hs2rip, hs2r1, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    parser_words_init hs1rip hs1r13 hs1len
  have hreg2 : ∀ {i : Nat} {x : Nat}, i ≠ 1 →
      s1.reg i = some x → s2.reg i = some x := by
    intro i x hi hx
    rw [hs2fr i hi]
    exact hx
  have hnwW : v % 2 ^ 32 < W := by
    have h := Nat.mod_lt v (show 0 < 2 ^ 32 by decide)
    unfold W
    omega
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    parser_words_head_pos hs2rip hs2r1 hnw0 hnwW
  have hreg3 : ∀ {i : Nat} {x : Nat},
      s2.reg i = some x → s3.reg i = some x := by
    intro i x hx
    rw [hs3fr i]
    exact hx
  have hlen3 : s3.regs.length = 16 := by rw [hs3len]; exact hs2len
  refine ⟨s3, ?_, hs3rip,
    hreg3 hs2r1,
    hreg3 (hreg2 (by decide) hs1r6),
    hreg3 (hreg2 (by decide) hs1r7),
    hreg3 (hreg2 (by decide) hs1r13),
    hreg3 (hreg2 (by decide) hs1r4),
    hreg3 (hreg2 (by decide) hs1r15),
    hreg3 (hreg2 (by decide) hs1r3),
    hreg3 (hreg2 (by decide) hs1r11),
    hreg3 (hreg2 (by decide) hs1r12),
    hreg3 (hreg2 (by decide) hs1r14),
    (hs3mem.trans hs2mem).trans hs1mem, hlen3,
    (hs3k.trans hs2k).trans hs1k⟩
  rw [show (14 : Nat) = (11 + 1) + 2 by omega,
    run_add stage1Im (11 + 1) 2,
    show run stage1Im (11 + 1) (Out.ok s) = Out.ok s2 from by
      rw [run_add stage1Im 11 1, hrun1, hrun2], hrun3]

set_option maxHeartbeats 0 in
/-- Code-loop bounds refusal: no opcode byte remains at the
cursor, so the bounds check falls through to `aFail1` in six
steps. -/
theorem parser_code_bounds_short {c e : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x315) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c ≤ e) (hshort : e < c + 1)
    (hcW : c < W) (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 6 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 2 → s'.reg i = s.reg i) ∧
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
  have ht4fl : X86Int.cond t4.fl 3 = some false := by
    have hcf : t4.fl.cf = some true := by
      rw [ht4]
      show (aluF .sub (e - c) (1 % W)).2.cf = some true
      rw [h1W]; exact aluF_sub_cf_true (by omega)
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
  -- 0x326: jae not taken
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x328 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p326) ht4fl
    rw [ht4rip] at h
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x328 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x328 := by rw [ht5]
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4 with rip := base + 0x328 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht4fr i hi
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4 with rip := base + 0x328 }).mem = s.mem
    rw [mem_with_rip]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4 with rip := base + 0x328 }).regs.length = 16
    rw [len_with_rip]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  -- 0x328: jmp fail1
  obtain ⟨t6, r6, ht6rip, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    jmp_step hc.p328 ht5rip
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
  have hrun : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
    rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
  refine ⟨t6, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht6rip]; rfl
  · intro i hi
    rw [ht6fr i]; exact ht5fr i hi
  · rw [ht6mem]; exact ht5mem
  · rw [ht6len]; exact ht5len
  · rw [ht6k]; exact ht5k

set_option maxHeartbeats 0 in
/-- Code-loop tail refusal: the dispatched instruction length
runs past the end of the file, so the tail reaches `aFail1`
in five steps. -/
theorem parser_code_tail_overrun {c e l : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x3b3) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr10 : s.reg 10 = some l)
    (hce : c ≤ e) (hover : e < c + l) (hcW : c < W) (heW : e < W)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 5 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
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
  have ht3fl : X86Int.cond t3.fl 3 = some false := by
    have hcf : t3.fl.cf = some true := by
      rw [ht3]
      show (aluF .sub (e - c) l).2.cf = some true
      exact aluF_sub_cf_true (by omega)
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
  -- 0x3bc: jae not taken
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x3be } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p3bc) ht3fl
    rw [ht3rip] at h
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x3be } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x3be := by rw [ht4]
  have ht4fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t4.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht4]
    show ({ t3 with rip := base + 0x3be }).reg i = s.reg i
    rw [reg_with_rip, ht3fr i]; exact ht2fr i hi1 hi2 hi6
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3be }).mem = s.mem
    rw [mem_with_rip]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3be }).regs.length = 16
    rw [len_with_rip]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x3be: jmp fail1
  obtain ⟨t5, r5, ht5rip, ht5fr, ht5mem, ht5len, ht5k⟩ :=
    jmp_step hc.p3be ht4rip
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have hrun : run stage1Im 5 (Out.ok s) = Out.ok t5 := by
    rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
  refine ⟨t5, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht5rip]; rfl
  · intro i hi1 hi2 hi6
    rw [ht5fr i]; exact ht4fr i hi1 hi2 hi6
  · rw [ht5mem]; exact ht4mem
  · rw [ht5len]; exact ht4len
  · rw [ht5k]; exact ht4k

/-- Every opcode byte either has a dispatch length or lies
in one of the two invalid ranges the dispatch refusals cover. -/
theorem opLen_or_invalid (op : Nat) :
    (∃ l, OpLen op l) ∨ (57 ≤ op ∧ op < 129 ∧ op ≠ 128) ∨
      141 ≤ op := by
  by_cases h1 : op < 57
  · exact Or.inl ⟨1, Or.inl ⟨h1, rfl⟩⟩
  by_cases h2 : op = 128
  · exact Or.inl ⟨9, Or.inr (Or.inl ⟨h2, rfl⟩)⟩
  by_cases h3 : op = 131 ∨ op = 132 ∨ op = 133 ∨ op = 140
  · exact Or.inl ⟨1, Or.inr (Or.inr (Or.inl ⟨h3, rfl⟩))⟩
  by_cases h4 : 129 ≤ op ∧ op ≤ 139
  · have hne : op ≠ 131 ∧ op ≠ 132 ∧ op ≠ 133 := by
      refine ⟨fun h => h3 (Or.inl h), fun h => h3 (Or.inr (Or.inl h)),
        fun h => h3 (Or.inr (Or.inr (Or.inl h)))⟩
    exact Or.inl ⟨5, Or.inr (Or.inr (Or.inr
      ⟨⟨h4.1, h4.2, hne.1, hne.2.1, hne.2.2⟩, rfl⟩))⟩
  by_cases h5 : op < 129
  · exact Or.inr (Or.inl ⟨by omega, h5, h2⟩)
  · exact Or.inr (Or.inr (by
      have h140 : op ≠ 140 := fun h =>
        h3 (Or.inr (Or.inr (Or.inr h)))
      omega))

/-- An instruction with a prescribed encoding: from a dispatch
length for the head byte and the exact bytes, all below 256. -/
theorem instr_of_opLen {op l : Nat} {bs : List Nat}
    (hOp : OpLen op l) (hbs : bs.length = l)
    (hb : ∀ b ∈ bs, b < 256) (hhead : bs.head? = some op) :
    ∃ x : Instr, x.enc = bs ∧ InstrBound x := by
  cases bs with
  | nil => simp at hhead
  | cons b rest =>
    have hbop : b = op := by
      have h : (b :: rest).head? = some b := rfl
      rw [h] at hhead
      exact Option.some.inj hhead
    subst hbop
    have hrestb : ∀ x ∈ rest, x < 256 := fun x hx =>
      hb x (List.mem_cons_of_mem _ hx)
    rcases hOp with ⟨hop, hl⟩ | ⟨hop, hl⟩ | ⟨hop, hl⟩ | ⟨hop, hl⟩
    · have hrlen : rest.length = 0 := by
        rw [hl] at hbs
        simp only [List.length_cons] at hbs
        omega
      have hrest : rest = [] := (by cases rest with | nil => rfl | cons x xs => simp at hrlen)
      subst hrest
      have hlt : b < Prim.all.length := by
        rw [prim_all_length]; exact hop
      exact ⟨.prim (Prim.all[b]'hlt), by
        show [(Prim.all[b]'hlt).code] = [b]
        rw [prim_code_get b hlt], trivial⟩
    · subst hop
      have hrlen : rest.length = 8 := by
        rw [hl] at hbs
        simp only [List.length_cons] at hbs
        omega
      exact ⟨.lit (leVal rest), by
        show 0x80 :: leBytes (leVal rest) 8 = 0x80 :: rest
        rw [← hrlen]
        exact congrArg _ (leBytes_leVal hrestb), by
        show leVal rest < 256 ^ 8
        have h := leVal_bound hrestb
        rwa [hrlen] at h⟩
    · have hrlen : rest.length = 0 := by
        rw [hl] at hbs
        simp only [List.length_cons] at hbs
        omega
      have hrest : rest = [] := (by cases rest with | nil => rfl | cons x xs => simp at hrlen)
      subst hrest
      rcases hop with rfl | rfl | rfl | rfl
      · exact ⟨.tor, rfl, trivial⟩
      · exact ⟨.fromr, rfl, trivial⟩
      · exact ⟨.rfetch, rfl, trivial⟩
      · exact ⟨.ret, rfl, trivial⟩
    · obtain ⟨hlo, hhi, h131, h132, h133⟩ := hop
      have hrlen : rest.length = 4 := by
        rw [hl] at hbs
        simp only [List.length_cons] at hbs
        omega
      have key : ∀ (f : Nat → Instr),
          (∀ v, (f v).enc = b :: leBytes v 4) →
          (∀ v, v < 256 ^ 4 → InstrBound (f v)) →
          ∃ x : Instr, x.enc = b :: rest ∧ InstrBound x := by
        intro f hf hbnd
        refine ⟨f (leVal rest), by
          rw [hf, ← hrlen, leBytes_leVal hrestb], hbnd _ ?_⟩
        have h := leVal_bound hrestb
        rwa [hrlen] at h
      have hcases : b = 129 ∨ b = 130 ∨ b = 134 ∨ b = 135 ∨
          b = 136 ∨ b = 137 ∨ b = 138 ∨ b = 139 := by omega
      rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact key Instr.str (fun v => rfl) (fun v hv => hv)
      · exact key Instr.pick (fun v => rfl) (fun v hv => hv)
      · exact key Instr.locget (fun v => rfl) (fun v hv => hv)
      · exact key Instr.locset (fun v => rfl) (fun v hv => hv)
      · exact key Instr.jmp (fun v => rfl) (fun v hv => hv)
      · exact key Instr.jz (fun v => rfl) (fun v hv => hv)
      · exact key Instr.jnz (fun v => rfl) (fun v hv => hv)
      · exact key Instr.call (fun v => rfl) (fun v hv => hv)

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

theorem cond_cc8_of_sf_true {f : Flags} (hsf : f.sf = some true) :
    X86Int.cond f 8 = some true := by
  simp [X86Int.cond, hsf]

theorem cond_cc9_of_sf_true {f : Flags} (hsf : f.sf = some true) :
    X86Int.cond f 9 = some false := by
  simp [X86Int.cond, hsf]

/-- sf after `test (neg ENOMEM) (neg ENOMEM)`.  Verbatim copy of the
`hmsb`/`hsf` block in parser_mmap1_nomem (Refuse:2919-2927), factored
out so both loader sites share it. -/
theorem sf_test_neg_enomem :
    (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf =
      some true := by
  have hmsb : msb (X86Env.neg ENOMEM) = true := by
    unfold msb X86Env.neg ENOMEM W
    decide
  have h1 : (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2.sf =
      some (msb (aluF .and (X86Env.neg ENOMEM)
        (X86Env.neg ENOMEM)).1) := rfl
  rw [h1, aluF_and_fst_self, hmsb]

/-! ### §1. The missing primitive: `mremap` with a refusing oracle

Mirror of syscall_mremap_ret_of_place (Defs:563-589) with the
existential resolved to the explicit refusal return, exactly as
syscall_mmap_nomem (Refuse:2228-2240) mirrors
syscall_mmap_ret_of_place (Defs:546-561).  All guards of the accept
version are retained — they are evaluated BEFORE the oracle is
consulted (X86Env:229-236), so they are genuine preconditions of
the refusal too, not artifacts: without hfind/hbase/hlen/hrw/hgrow
the model syscall is `none` (stuck), never `-ENOMEM`.
NOT needed (contrast accept): any `place`/free/aligned/bound fact —
the `.nomem` branch of `place` ignores mem and size (X86Env:173).
a5/a6 are universally quantified and unused: at the growth site they
are the preserved-but-arbitrary v8/v9 (hg6r8/hg6r9, Loader:4003-4047;
the block never zeroes r8/r9, cf. loader_grow hr8/hr9 as bare
existentials, Loader:3724). -/

theorem syscall_mremap_nomem {k : Kern} {m : Mem}
    {a1 a2 a3 a5 a6 : Nat} {i : Nat} {r : Region}
    (hfind : m.find a1 0 = some (i, r))
    (hbase : r.base = a1) (hlen : r.data.length = X86Env.pages a2)
    (hrw : r.rw = true) (hgrow : ¬ X86Env.pages a3 < r.data.length)
    (hhead : k.oracle.head? = some .nomem) :
    X86Env.syscall k m 25 a1 a2 a3 MREMAP_MAYMOVE a5 a6 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next k).2 m) := by
  have hnext1 : (Kern.next k).1 = some .nomem :=
    kern_next_of_head hhead
  have hval : ¬ (r.base ≠ a1 ∨ r.data.length ≠ X86Env.pages a2 ∨
      ¬ r.rw) := by simp [hbase, hlen, hrw]
  -- Lines above: kern_next_of_head (Defs:538) and the `hval` shape of
  -- Defs:577-578 verbatim.  The simp invocation below is Defs:580-581
  -- verbatim; it discharges a4 (MREMAP_MAYMOVE), the find, hval
  -- (via hbase/hlen/hrw) and the not-shrink guard (hgrow).
  simp only [X86Env.syscall, hfind, hbase, hlen, hrw, hgrow,
    if_false]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    simp [X86Env.place, hgrow, hlen]
    omega

set_option maxHeartbeats 0 in
/-! ### §2. L1 — initial `mmap` refusal: `base+0xbf → aFail2` in 4 steps

Statement mirrors loader_mmap1_to_readloop's preconditions
(Loader:1910-1923) with: hhead `.at A` → `.nomem`; DELETED: the
second state `s`/hshape/hfree and hAal/hApos/hbound (only feed
place_at, Loader:1940-1954); hr13/hr14 dropped (unused in 4 steps;
hr14 only threaded to the read loop).  Post-state keeps reg4 and
mem IDENTICAL to the pre-state — this is what makes the exit-2
corollary's StackRoom transport trivial, unlike parser_mmap1_nomem
where sp moves by 8 and StackRoom needs wrN_some_of_wshape
(Refuse:3015-3030).

Proof = loader_mmap1_to_readloop lines 1972-2079 (b1,b2,b3) with
three replacements, then parser_mmap1_nomem lines 2929-2946 for the
fallthrough+jmp:
  R1. hsc (Loader:1955-1971) → one-liner via syscall_mmap_nomem:
        have hsc : X86Env.syscall s'.k s'.mem 9 0 65536 PROT_RW
            MAP_PRIVATE_ANON (W - 1) 0 =
          some (.ret (X86Env.neg ENOMEM) (Kern.next s'.k).2 s'.mem) :=
          syscall_mmap_nomem (by decide) hhead
      (hr2 : reg 2 = some 3 still discharges `some PROT_RW` by defeq,
      PROT_RW := 3, X86Env:166 — accept proof relies on the same
      defeq at Loader:1955-1978; MAP_PRIVATE_ANON := 0x22 = 34.)
  R2. b1 block (Loader:1972-2027): replace `some A` → `some (neg
      ENOMEM)` and mem `s'.mem ++ [⟨A,…⟩]` → `s'.mem`.  hb1r4 copy
      unchanged (Loader:1999-2006).  e1 via step_syscall_ret hc.dbf
      (Loader:500-511 signature).
  R3. b2 flag fact (Loader:2036-2045) →
        have hb2fl : X86Int.cond b2.fl 9 = some false :=
          cond_cc9_of_sf_true (by rw [hb2]; exact sf_test_neg_enomem)
      (hb2 def: fl := (aluF .and (neg ENOMEM) (neg ENOMEM)).2,
      rip := b1.rip + 3; e2 via step_test hc.dc1, Loader:2029-2031
      with A → neg ENOMEM.)
  R4. jcc NOT taken (mirror parser e19, Refuse:2929-2934 — note the
      `rw [hrip] at h; simpa using h` idiom, which produces the
      fallthrough `rip := base + 0xc6`):
        have e3 : X86Int.step stage1Im b2 =
            .ok { b2 with rip := base + 0xc6 } := by
          have h := step_jcc (im := stage1Im) (s := b2)
            (by rw [hb2rip]; exact hc.dc4) hb2fl
          rw [hb2rip] at h
          simpa using h
  R5. jmp (mirror parser e20, Refuse:2938-2946, hc.p225 → hc.dc6):
        have e4 : X86Int.step stage1Im b3 =
            .ok { b3 with rip := base + 0x12c } :=
          step_jmp (by rw [hb3rip]; exact hc.dc6)
        -- b4.rip = aFail2 by `show base + 0x12c = aFail2; rfl`
      (Accept's step_jmp use at Loader:4433-4435 has this direct
      form; parser wraps in simpa — either may be needed, adjust on
      elaboration.)
  Run assembly: r1..r4 one-step runs (run_succ_ok … run_zero idiom,
  e.g. Loader:4438-4439) composed by run_add as 4 = 1+3, 3 = 1+2,
  2 = 1+1 (tail of parser's hrun chain, Refuse:2992-2994).
  Frame fact: b1 writes regs 0,1,11 only (step_syscall_ret form,
  Loader:509); b2 writes fl/rip; b3/b4 write rip (`reg_with_rip`).
  Final reg4/mem/k/len: thread hb1r4 (r4), hb1mem := rfl-shape,
  hb1k, hlenb1 through rip/fl-only updates exactly as accept does
  for b2/b3 (Loader:2047-2079) and one more rip hop. -/

theorem loader_mmap1_nomem {sp0 : Nat} {s' : X86Int.St}
    (hrip : s'.rip = base + 0xbf) (hlen16 : s'.regs.length = 16)
    (hr0 : s'.reg 0 = some 9) (hr7 : s'.reg 7 = some 0)
    (hr6 : s'.reg 6 = some 65536) (hr2 : s'.reg 2 = some 3)
    (hr10 : s'.reg 10 = some 34) (hr8 : s'.reg 8 = some (W - 1))
    (hr9 : s'.reg 9 = some 0)
    (hr4 : s'.reg 4 = some (sp0 + 24))
    (hhead : s'.k.oracle.head? = some .nomem) :
    ∃ s'', run stage1Im 4 (Out.ok s') = Out.ok s'' ∧
      s''.rip = aFail2 ∧
      s''.reg 4 = some (sp0 + 24) ∧
      s''.mem = s'.mem ∧
      s''.k = (Kern.next s'.k).2 ∧
      s''.regs.length = 16 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 11 → s''.reg i = s'.reg i) := by
  have hc := stage1_loaderCode
  have hsc : X86Env.syscall s'.k s'.mem 9 0 65536 PROT_RW
        MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next s'.k).2 s'.mem) :=
    syscall_mmap_nomem (by decide) hhead
  have e1 : X86Int.step stage1Im s' =
      .ok { ((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
          (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2, mem := s'.mem,
        rip := s'.rip + 2 } :=
    step_syscall_ret (by rw [hrip]; exact hc.dbf) hr0 hr7 hr6 hr2
      hr10 hr8 hr9 hsc
  obtain ⟨b1, hb1⟩ : ∃ t : X86Int.St, t =
      { ((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
          (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2, mem := s'.mem,
        rip := s'.rip + 2 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = base + 0xc1 := by
    rw [hb1]; show s'.rip + 2 = base + 0xc1; rw [hrip]
  have hlenb1 : b1.regs.length = 16 := by
    rw [hb1]
    show (((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (s'.rip + 2))).set 11 none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen16
  have hb1r0 : b1.reg 0 = some (X86Env.neg ENOMEM) := by
    rw [hb1]
    show (((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (s'.rip + 2))).set 11 none).reg 0 =
      some (X86Env.neg ENOMEM)
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hb1r4 : b1.reg 4 = some (sp0 + 24) := by
    rw [hb1]
    show (((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (s'.rip + 2))).set 11 none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4
  have hb1fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 11 →
      b1.reg i = s'.reg i := by
    intro i hi0 hi1 hi11
    rw [hb1]
    show (((s'.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (s'.rip + 2))).set 11 none).reg i = s'.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
  have hb1mem : b1.mem = s'.mem := by rw [hb1]
  have hb1k : b1.k = (Kern.next s'.k).2 := by rw [hb1]
  -- step 2: test rax, rax (0xc1)
  have e2 : X86Int.step stage1Im b1 =
      .ok { b1 with
        fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2,
        rip := b1.rip + 3 } :=
    step_test (by rw [hb1rip]; exact hc.dc1) hb1r0 hb1r0
  obtain ⟨b2, hb2⟩ : ∃ t : X86Int.St, t =
      { b1 with
        fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2,
        rip := b1.rip + 3 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = base + 0xc4 := by
    rw [hb2]; show b1.rip + 3 = base + 0xc4; rw [hb1rip]
  have hb2fl : X86Int.cond b2.fl 9 = some false :=
    cond_cc9_of_sf_true (by rw [hb2]; exact sf_test_neg_enomem)
  have hb2r4 : b2.reg 4 = some (sp0 + 24) := by rw [hb2]; exact hb1r4
  have hb2mem : b2.mem = s'.mem := by rw [hb2]; exact hb1mem
  have hb2k : b2.k = (Kern.next s'.k).2 := by rw [hb2]; exact hb1k
  have hlenb2 : b2.regs.length = 16 := by rw [hb2]; exact hlenb1
  have hb2fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 11 →
      b2.reg i = s'.reg i := by
    intro i hi0 hi1 hi11; rw [hb2]; exact hb1fr i hi0 hi1 hi11
  -- step 3: jns 0xcb (0xc4) — NOT taken, sf = true
  have e3 : X86Int.step stage1Im b2 =
      .ok { b2 with rip := base + 0xc6 } := by
    have h := step_jcc (im := stage1Im) (s := b2)
      (by rw [hb2rip]; exact hc.dc4) hb2fl
    rw [hb2rip] at h
    simpa using h
  obtain ⟨b3, hb3⟩ : ∃ t : X86Int.St, t =
      { b2 with rip := base + 0xc6 } := ⟨_, rfl⟩
  have hb3rip : b3.rip = base + 0xc6 := by rw [hb3]
  have hb3r4 : b3.reg 4 = some (sp0 + 24) := by
    rw [hb3]
    show ({ b2 with rip := base + 0xc6 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hb2r4
  -- step 4: jmp fail2 (0xc6)
  have e4 : X86Int.step stage1Im b3 =
      .ok { b3 with rip := base + 0x12c } := by
    have h := step_jmp (im := stage1Im) (s := b3)
      (by rw [hb3rip]; exact hc.dc6)
    simpa using h
  obtain ⟨b4, hb4⟩ : ∃ t : X86Int.St, t =
      { b3 with rip := base + 0x12c } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s') = Out.ok b1 := by
    rw [run_succ_ok stage1Im 0 s', e1, ← hb1, run_zero]
  have r2 : run stage1Im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok stage1Im 0 b1, e2, ← hb2, run_zero]
  have r3 : run stage1Im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok stage1Im 0 b2, e3, ← hb3, run_zero]
  have r4 : run stage1Im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok stage1Im 0 b3, e4, ← hb4, run_zero]
  have hrun : run stage1Im 4 (Out.ok s') = Out.ok b4 := by
    rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
  refine ⟨b4, hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hb4]; show base + 0x12c = aFail2; rfl
  · rw [hb4]
    show ({ b3 with rip := base + 0x12c }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hb3r4
  · have h3m : b3.mem = s'.mem := by
      rw [hb3]
      show ({ b2 with rip := base + 0xc6 }).mem = s'.mem
      rw [mem_with_rip]; exact hb2mem
    rw [hb4]
    show ({ b3 with rip := base + 0x12c }).mem = s'.mem
    rw [mem_with_rip]; exact h3m
  · have h3k : b3.k = (Kern.next s'.k).2 := by
      rw [hb3]
      show ({ b2 with rip := base + 0xc6 }).k = (Kern.next s'.k).2
      exact hb2k
    rw [hb4]
    show ({ b3 with rip := base + 0x12c }).k = (Kern.next s'.k).2
    exact h3k
  · have h3l : b3.regs.length = 16 := by
      rw [hb3]
      show ({ b2 with rip := base + 0xc6 }).regs.length = 16
      rw [len_with_rip]; exact hlenb2
    rw [hb4]
    show ({ b3 with rip := base + 0x12c }).regs.length = 16
    rw [len_with_rip]; exact h3l
  · intro i hi0 hi1 hi11
    have h3i : b3.reg i = s'.reg i := by
      rw [hb3]
      show ({ b2 with rip := base + 0xc6 }).reg i = s'.reg i
      rw [reg_with_rip]; exact hb2fr i hi0 hi1 hi11
    rw [hb4]
    show ({ b3 with rip := base + 0x12c }).reg i = s'.reg i
    rw [reg_with_rip]; exact h3i

/-- L1 composed to exit 2.  Extra hypotheses are exactly
exit2_of_fail2's (Loader:5764-5767), specialised at sp := sp0+24:
`32 ≤ sp0 + 24` (i.e. 8 ≤ sp0 — keep in the sp0+24 form so `omega`
at the call site, where sp0 = stk.base + stackRoom, closes it like
parser_mmap1_nomem_exit2's `(by omega)`, Refuse:3076),
`sp0 + 24 < W`, RegsOk, StackRoom _ 1.  StackRoom transport is the
identity: StackRoom mentions only reg 4 and mem (SporosX1:77-79),
both unchanged by the segment. -/
theorem loader_mmap1_nomem_exit2 {sp0 : Nat} {s' : X86Int.St}
    (hrip : s'.rip = base + 0xbf) (hlen16 : s'.regs.length = 16)
    (hr0 : s'.reg 0 = some 9) (hr7 : s'.reg 7 = some 0)
    (hr6 : s'.reg 6 = some 65536) (hr2 : s'.reg 2 = some 3)
    (hr10 : s'.reg 10 = some 34) (hr8 : s'.reg 8 = some (W - 1))
    (hr9 : s'.reg 9 = some 0)
    (hr4 : s'.reg 4 = some (sp0 + 24))
    (hhead : s'.k.oracle.head? = some .nomem)
    (hregs : RegsOk s') (hroom : StackRoom s' 1)
    (hsp32 : 32 ≤ sp0 + 24) (hspW : sp0 + 24 < W) :
    ∃ k, run stage1Im k (Out.ok s') = Out.exit 2 := by
  obtain ⟨s'', hrun, hrip'', hr4'', hmem'', -, -, -⟩ :=
    loader_mmap1_nomem hrip hlen16 hr0 hr7 hr6 hr2 hr10 hr8 hr9 hr4
      hhead
  have hroom'' : StackRoom s'' 1 := by
    intro sp hsp i hi v
    have hsp_eq : sp = sp0 + 24 := by
      rw [hr4''] at hsp
      exact (Option.some.inj hsp).symm
    rw [hmem'']
    exact hroom sp (by rw [hsp_eq]; exact hr4) i hi v
  obtain ⟨k, hk⟩ := exit2_of_fail2 hsp32 hspW s'' hrip'' hr4''
    (run_regs_ok hrun hregs) hroom''
  exact ⟨4 + k, by rw [run_add stage1Im 4 k, hrun, hk]⟩

set_option maxHeartbeats 0 in
/-! ### §2b. L2 — growth `mremap` refusal: `base+0xfc → aFail2` in 9 steps

Mirror of `loader_grow` (Loader:3718ff) for a refusing oracle:
steps 1–6 (argument setup to `base+0x113`) are verbatim from the
accept proof; the syscall uses `syscall_mremap_nomem` (§1), so
memory is unchanged and no `place`/disjoint/bound fact is needed.
The `js aFail2` at `0x118` (hc.d118) is TAKEN on `-ENOMEM`
(`cond_cc8_of_sf_true` + `sf_test_neg_enomem`), landing directly
on `aFail2` — one step shorter than the accept fallthrough.
The `hfind` bundle is the exact precondition of the refusal
primitive, stated over `s.mem` abstractly (the read loop supplies
it from `find_third` over `[c1, c2, buf]`, as in `loader_grow`). -/
theorem loader_grow_nomem {A sp0 cap count : Nat}
    {s : X86Int.St} {i : Nat} {r : Region}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hfind : s.mem.find A 0 = some (i, r))
    (hbase : r.base = A) (hdatalen : r.data.length = cap)
    (hrw : r.rw = true)
    (hcapmod : cap % page = 0) (hcapW : cap + cap < W)
    (hhead : s.k.oracle.head? = some .nomem) :
    ∃ s', run stage1Im 9 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail2 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      s'.mem = s.mem ∧
      s'.k = (Kern.next s.k).2 ∧
      s'.regs.length = 16 := by
  have hc := stage1_loaderCode
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  -- step 1: mov eax, 25
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (25 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.dfc)
  obtain ⟨g1, hg1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (25 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have hg1rip : g1.rip = base + 0x101 := by
    rw [hg1]; show s.rip + 5 = base + 0x101; rw [hrip]
  have hlen1 : g1.regs.length = 16 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen16
  have hg1r0 : g1.reg 0 = some 25 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg1r3 : g1.reg 3 = some A := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hr3
  have hg1r12 : g1.reg 12 = some cap := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hr12
  have hg1mem : g1.mem = s.mem := by rw [hg1]; rfl
  have hg1k : g1.k = s.k := by rw [hg1]; rfl
  -- step 2: mov rdi, rbx
  have e2 : X86Int.step stage1Im g1 =
      .ok { g1.set 7 (some A) with rip := g1.rip + 3 } :=
    step_mov (by rw [hg1rip]; exact hc.d101) hg1r3
  obtain ⟨g2, hg2⟩ : ∃ t : X86Int.St, t =
      { g1.set 7 (some A) with rip := g1.rip + 3 } := ⟨_, rfl⟩
  have hg2rip : g2.rip = base + 0x104 := by
    rw [hg2]; show g1.rip + 3 = base + 0x104; rw [hg1rip]
  have hlen2 : g2.regs.length = 16 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hg2r7 : g2.reg 7 = some A := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg2r0 : g2.reg 0 = some 25 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hg1r0
  have hg2r12 : g2.reg 12 = some cap := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    exact hg1r12
  have hg2mem : g2.mem = s.mem := by rw [hg2]; exact hg1mem
  have hg2k : g2.k = s.k := by rw [hg2]; exact hg1k
  -- step 3: mov rsi, r12
  have e3 : X86Int.step stage1Im g2 =
      .ok { g2.set 6 (some cap) with rip := g2.rip + 3 } :=
    step_mov (by rw [hg2rip]; exact hc.d104) hg2r12
  obtain ⟨g3, hg3⟩ : ∃ t : X86Int.St, t =
      { g2.set 6 (some cap) with rip := g2.rip + 3 } := ⟨_, rfl⟩
  have hg3rip : g3.rip = base + 0x107 := by
    rw [hg3]; show g2.rip + 3 = base + 0x107; rw [hg2rip]
  have hlen3 : g3.regs.length = 16 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen2
  have hg3r6 : g3.reg 6 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg3r7 : g3.reg 7 = some A := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact hg2r7
  have hg3r0 : g3.reg 0 = some 25 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hg2r0
  have hg3r12 : g3.reg 12 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hg2r12
  have hg3mem : g3.mem = s.mem := by rw [hg3]; exact hg2mem
  have hg3k : g3.k = s.k := by rw [hg3]; exact hg2k
  -- step 4: mov rdx, r12
  have e4 : X86Int.step stage1Im g3 =
      .ok { g3.set 2 (some cap) with rip := g3.rip + 3 } :=
    step_mov (by rw [hg3rip]; exact hc.d107) hg3r12
  obtain ⟨g4, hg4⟩ : ∃ t : X86Int.St, t =
      { g3.set 2 (some cap) with rip := g3.rip + 3 } := ⟨_, rfl⟩
  have hg4rip : g4.rip = base + 0x10a := by
    rw [hg4]; show g3.rip + 3 = base + 0x10a; rw [hg3rip]
  have hlen4 : g4.regs.length = 16 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen3
  have hg4r2 : g4.reg 2 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg4r6 : g4.reg 6 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hg3r6
  have hg4r7 : g4.reg 7 = some A := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact hg3r7
  have hg4r0 : g4.reg 0 = some 25 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact hg3r0
  have hg4mem : g4.mem = s.mem := by rw [hg4]; exact hg3mem
  have hg4k : g4.k = s.k := by rw [hg4]; exact hg3k
  -- step 5: add rdx, rdx
  have hadd : (aluF .add cap cap).1 = cap + cap := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hcapW]
  have e5 : X86Int.step stage1Im g4 =
      .ok { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } :=
    step_alu_add (by rw [hg4rip]; exact hc.d10a) hg4r2 hg4r2
  obtain ⟨g5, hg5⟩ : ∃ t : X86Int.St, t =
      { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } := ⟨_, rfl⟩
  have hg5rip : g5.rip = base + 0x10d := by
    rw [hg5]; show g4.rip + 3 = base + 0x10d; rw [hg4rip]
  have hlen5 : g5.regs.length = 16 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).regs.length = 16
    rw [len_set]; exact hlen4
  have hg5r2 : g5.reg 2 = some (cap + cap) := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 2 = some (cap + cap)
    rw [reg_set_self (by omega), hadd]
  have hg5r6 : g5.reg 6 = some cap := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 6 = some cap
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]; exact hg4r6
  have hg5r7 : g5.reg 7 = some A := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 7 = some A
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]; exact hg4r7
  have hg5r0 : g5.reg 0 = some 25 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 0 = some 25
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]; exact hg4r0
  have hg5mem : g5.mem = s.mem := by rw [hg5]; exact hg4mem
  have hg5k : g5.k = s.k := by rw [hg5]; exact hg4k
  -- step 6: mov r10, 1 (MREMAP_MAYMOVE)
  have e6 : X86Int.step stage1Im g5 =
      .ok { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } :=
    step_movI (by rw [hg5rip]; exact hc.d10d)
  obtain ⟨g6, hg6⟩ : ∃ t : X86Int.St, t =
      { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } := ⟨_, rfl⟩
  have hg6rip : g6.rip = base + 0x113 := by
    rw [hg6]; show g5.rip + 6 = base + 0x113; rw [hg5rip]
  have hlen6 : g6.regs.length = 16 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen5
  have hg6r10 : g6.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg6r0 : g6.reg 0 = some 25 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    exact hg5r0
  have hg6r7 : g6.reg 7 = some A := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 7 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    exact hg5r7
  have hg6r6 : g6.reg 6 = some cap := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 6 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    exact hg5r6
  have hg6r2 : g6.reg 2 = some (cap + cap) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    exact hg5r2
  have hg6mem : g6.mem = s.mem := by rw [hg6]; exact hg5mem
  have hg6k : g6.k = s.k := by rw [hg6]; exact hg5k
  have hg6r8 : g6.reg 8 = some v8 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8v
  have hg6r9 : g6.reg 9 = some v9 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9v
  have hg6r13 : g6.reg 13 = some count := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hr13
  have hg6r14 : g6.reg 14 = some 3 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  have hg6r4 : g6.reg 4 = some (sp0 + 24) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4

  -- the refusing syscall at the g6 state
  have hpages : X86Env.pages cap = cap :=
    pages_eq_of_mod_zero hcapmod
  have hpages2 : X86Env.pages (cap + cap) = cap + cap :=
    pages_eq_of_mod_zero (by simp [Nat.add_mod, hcapmod])
  have hhead6 : g6.k.oracle.head? = some .nomem := by
    rw [hg6k]; exact hhead
  have hfind6 : g6.mem.find A 0 = some (i, r) := by
    rw [hg6mem]; exact hfind
  have hsc : X86Env.syscall g6.k g6.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE v8 v9 =
      some (.ret (X86Env.neg ENOMEM) (Kern.next g6.k).2 g6.mem) :=
    syscall_mremap_nomem hfind6 hbase
      (by rw [hdatalen, hpages]) hrw
      (by rw [hpages2, hdatalen]; omega) hhead6
  -- step 7: mremap refused (-ENOMEM), memory unchanged
  have e7 : X86Int.step stage1Im g6 =
      .ok { ((g6.set 0 (some (X86Env.neg ENOMEM))).set 1
          (some (g6.rip + 2))).set 11 none with
        k := (Kern.next g6.k).2, mem := g6.mem,
        rip := g6.rip + 2 } :=
    step_syscall_ret (by rw [hg6rip]; exact hc.d113) hg6r0 hg6r7
      hg6r6 hg6r2 hg6r10 hg6r8 hg6r9 hsc
  obtain ⟨g7, hg7⟩ : ∃ t : X86Int.St, t =
      { ((g6.set 0 (some (X86Env.neg ENOMEM))).set 1
          (some (g6.rip + 2))).set 11 none with
        k := (Kern.next g6.k).2, mem := g6.mem,
        rip := g6.rip + 2 } := ⟨_, rfl⟩
  have hg7rip : g7.rip = base + 0x115 := by
    rw [hg7]; show g6.rip + 2 = base + 0x115; rw [hg6rip]
  have hlen7 : g7.regs.length = 16 := by
    rw [hg7]
    show (((g6.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (g6.rip + 2))).set 11 none).regs.length = 16
    rw [len_set, len_set, len_set]; exact hlen6
  have hg7r0 : g7.reg 0 = some (X86Env.neg ENOMEM) := by
    rw [hg7]
    show (((g6.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (g6.rip + 2))).set 11 none).reg 0 =
      some (X86Env.neg ENOMEM)
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hg7r4 : g7.reg 4 = some (sp0 + 24) := by
    rw [hg7]
    show (((g6.set 0 (some (X86Env.neg ENOMEM))).set 1
      (some (g6.rip + 2))).set 11 none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hg6r4
  have hg7mem : g7.mem = s.mem := by rw [hg7]; exact hg6mem
  have hg7k : g7.k = (Kern.next s.k).2 := by
    rw [hg7]; show (Kern.next g6.k).2 = (Kern.next s.k).2
    rw [hg6k]
  -- step 8: test rax, rax (0x115)
  have e8 : X86Int.step stage1Im g7 =
      .ok { g7 with
        fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2,
        rip := g7.rip + 3 } :=
    step_test (by rw [hg7rip]; exact hc.d115) hg7r0 hg7r0
  obtain ⟨g8, hg8⟩ : ∃ t : X86Int.St, t =
      { g7 with
        fl := (aluF .and (X86Env.neg ENOMEM) (X86Env.neg ENOMEM)).2,
        rip := g7.rip + 3 } := ⟨_, rfl⟩
  have hg8rip : g8.rip = base + 0x118 := by
    rw [hg8]; show g7.rip + 3 = base + 0x118; rw [hg7rip]
  have hg8fl8 : X86Int.cond g8.fl 8 = some true :=
    cond_cc8_of_sf_true (by rw [hg8]; exact sf_test_neg_enomem)
  have hg8r4 : g8.reg 4 = some (sp0 + 24) := by rw [hg8]; exact hg7r4
  have hg8mem : g8.mem = s.mem := by rw [hg8]; exact hg7mem
  have hg8k : g8.k = (Kern.next s.k).2 := by rw [hg8]; exact hg7k
  have hlen8 : g8.regs.length = 16 := by rw [hg8]; exact hlen7
  -- step 9: js aFail2 (0x118) — TAKEN, sf = true
  have e9 : X86Int.step stage1Im g8 =
      .ok { g8 with rip := base + 0x12c } := by
    have h := step_jcc (im := stage1Im) (s := g8)
      (by rw [hg8rip]; exact hc.d118) hg8fl8
    simpa using h
  obtain ⟨g9, hg9⟩ : ∃ t : X86Int.St, t =
      { g8 with rip := base + 0x12c } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok g1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hg1, run_zero]
  have r2 : run stage1Im 1 (Out.ok g1) = Out.ok g2 := by
    rw [run_succ_ok stage1Im 0 g1, e2, ← hg2, run_zero]
  have r3 : run stage1Im 1 (Out.ok g2) = Out.ok g3 := by
    rw [run_succ_ok stage1Im 0 g2, e3, ← hg3, run_zero]
  have r4 : run stage1Im 1 (Out.ok g3) = Out.ok g4 := by
    rw [run_succ_ok stage1Im 0 g3, e4, ← hg4, run_zero]
  have r5 : run stage1Im 1 (Out.ok g4) = Out.ok g5 := by
    rw [run_succ_ok stage1Im 0 g4, e5, ← hg5, run_zero]
  have r6 : run stage1Im 1 (Out.ok g5) = Out.ok g6 := by
    rw [run_succ_ok stage1Im 0 g5, e6, ← hg6, run_zero]
  have r7 : run stage1Im 1 (Out.ok g6) = Out.ok g7 := by
    rw [run_succ_ok stage1Im 0 g6, e7, ← hg7, run_zero]
  have r8 : run stage1Im 1 (Out.ok g7) = Out.ok g8 := by
    rw [run_succ_ok stage1Im 0 g7, e8, ← hg8, run_zero]
  have r9 : run stage1Im 1 (Out.ok g8) = Out.ok g9 := by
    rw [run_succ_ok stage1Im 0 g8, e9, ← hg9, run_zero]
  have hrun : run stage1Im 9 (Out.ok s) = Out.ok g9 := by
    rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
  refine ⟨g9, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hg9]; show base + 0x12c = aFail2; rfl
  · rw [hg9]
    show ({ g8 with rip := base + 0x12c }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hg8r4
  · rw [hg9]
    show ({ g8 with rip := base + 0x12c }).mem = s.mem
    rw [mem_with_rip]; exact hg8mem
  · rw [hg9]
    show ({ g8 with rip := base + 0x12c }).k = (Kern.next s.k).2
    exact hg8k
  · rw [hg9]
    show ({ g8 with rip := base + 0x12c }).regs.length = 16
    rw [len_with_rip]; exact hlen8

/-- L2 composed to exit 2, with `exit2_of_fail2`'s extra
hypotheses at `sp := sp0+24`.  StackRoom transport is the
identity (reg 4 and mem unchanged), as in
`loader_mmap1_nomem_exit2`. -/
theorem loader_grow_nomem_exit2 {A sp0 cap count : Nat}
    {s : X86Int.St} {i : Nat} {r : Region}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hfind : s.mem.find A 0 = some (i, r))
    (hbase : r.base = A) (hdatalen : r.data.length = cap)
    (hrw : r.rw = true)
    (hcapmod : cap % page = 0) (hcapW : cap + cap < W)
    (hhead : s.k.oracle.head? = some .nomem)
    (hregs : RegsOk s) (hroom : StackRoom s 1)
    (hsp32 : 32 ≤ sp0 + 24) (hspW : sp0 + 24 < W) :
    ∃ k, run stage1Im k (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', hmem', -, -⟩ :=
    loader_grow_nomem hrip hlen16 hr3 hr12 hr13 hr14 hr4 hr8 hr9
      hfind hbase hdatalen hrw hcapmod hcapW hhead
  have hroom' : StackRoom s' 1 := by
    intro sp hsp i hi v
    have hsp_eq : sp = sp0 + 24 := by
      rw [hr4'] at hsp
      exact (Option.some.inj hsp).symm
    rw [hmem']
    exact hroom sp (by rw [hsp_eq]; exact hr4) i hi v
  obtain ⟨k, hk⟩ := exit2_of_fail2 hsp32 hspW s' hrip' hr4'
    (run_regs_ok hrun hregs) hroom'
  exact ⟨9 + k, by rw [run_add stage1Im 9 k, hrun, hk]⟩

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

set_option maxHeartbeats 0 in
/-- One code-loop iteration, classified for arbitrary bytes:
either the byte at the cursor starts a valid instruction that
fits before `e`, and the iteration is `parser_code_step` over
the constructed instruction, or the run reaches `aFail1`
(no byte left, invalid opcode, or instruction overrun). -/
theorem parser_code_step_classify {c e B S n : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some (n + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (hce : c ≤ e) (hcW : c < W) (heW : e < W)
    (hnW : n + 1 < W) (hBc : B ≤ c) (hSW : S + (c - B) < W)
    (hwr : ∀ l, 0 < l → c + l ≤ e →
      (s.mem.wr (S + (c - B)) l).isSome)
    (hrdN : ∀ l, l ≤ e - c →
      ∃ bs, s.mem.rdN c l = some bs ∧ ∀ b ∈ bs, b < 256)
    (hlen16 : s.regs.length = 16) :
    (∃ (x : Instr) (k : Nat) (s' : X86Int.St),
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧
      s'.reg 6 = some (c + x.enc.length) ∧ s'.reg 1 = some n ∧
      s'.mem.rd (S + (c - B)) = some x.enc.length ∧
      s.mem.rdN c x.enc.length = some x.enc ∧
      c + x.enc.length ≤ e ∧ InstrBound x ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c - B) + 1 ≤ y ∨ y + k ≤ S + (c - B) →
        s'.mem.rdN y k = s.mem.rdN y k)) ∨
    (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) := by
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    parser_code_head_pos hrip hr1 hnW
  have hs1len16 : s1.regs.length = 16 := by
    rw [hs1len]; exact hlen16
  have hs1r6 : s1.reg 6 = some c := by rw [hs1fr 6]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1fr 15]; exact hr15
  by_cases h1 : c + 1 ≤ e
  · obtain ⟨bs1, hbs1, hb1⟩ := hrdN 1 (by omega)
    have hbs1len : bs1.length = 1 := rdN_length hbs1
    obtain ⟨op, hopbs⟩ : ∃ op, bs1 = [op] := by
      cases bs1 with
      | nil => simp at hbs1len
      | cons b rest =>
        cases rest with
        | nil => exact ⟨b, rfl⟩
        | cons b2 rest2 => simp at hbs1len
    subst hopbs
    have hop256 : op < 256 := hb1 op (by simp)
    have hrd : s.mem.rd c = some op := by
      have h : s.mem.rdN c (0 + 1) = some (op :: []) := by
        simpa using hbs1
      exact (rdN_cons h).1
    have hs1rd : s1.mem.rd c = some op := by
      rw [hs1mem]; exact hrd
    obtain ⟨s2, hrun2, hs2rip, hs2r0, hs2r6, hs2fr, hs2mem,
        hs2len16, hs2k⟩ :=
      parser_code_bounds hs1rip hs1r6 hs1r15 h1 hcW heW hs1rd
        hs1len16
    have hrun12 : run stage1Im (2 + 6) (Out.ok s) = Out.ok s2 := by
      rw [run_add stage1Im 2 6, hrun1, hrun2]
    have hfr2 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s2.reg i = s.reg i := by
      intro i hi0 _ hi2 _ _
      rw [hs2fr i hi0 hi2, hs1fr i]
    have hmem2 : s2.mem = s.mem := hs2mem.trans hs1mem
    have hk2 : s2.k = s.k := hs2k.trans hs1k
    rcases opLen_or_invalid op with ⟨l, hOp⟩ | hlo | hhi
    · by_cases hfit : c + l ≤ e
      · obtain ⟨bsl, hbsl, hbl⟩ := hrdN l (by omega)
        have hlenl : bsl.length = l := rdN_length hbsl
        have hlpos : 0 < l := by
          rcases hOp with ⟨-, hl⟩ | ⟨-, hl⟩ | ⟨-, hl⟩ | ⟨-, hl⟩ <;>
            omega
        have hhead : bsl.head? = some op := by
          cases hbseq : bsl with
          | nil =>
            rw [hbseq] at hlenl
            simp at hlenl
            omega
          | cons b rest =>
            have hleq : l = rest.length + 1 := by
              rw [hbseq] at hlenl
              simp only [List.length_cons] at hlenl
              omega
            have hb1' : s.mem.rd c = some b := by
              rw [hbseq, hleq] at hbsl
              exact (rdN_cons hbsl).1
            have hb_eq : b = op := by
              rw [hrd] at hb1'
              exact (Option.some.inj hb1').symm
            subst hb_eq
            rfl
        obtain ⟨x, hx, hxb⟩ := instr_of_opLen hOp hlenl hbl hhead
        have henc : s.mem.rdN c x.enc.length = some x.enc := by
          rw [hx, hlenl]
          exact hbsl
        have hxlen : x.enc.length = l := by
          rw [hx]; exact hlenl
        have hxpos : 0 < x.enc.length := by
          obtain ⟨op', rest', heq', -⟩ := enc_head x
          rw [heq']; exact Nat.zero_lt_succ _
        obtain ⟨k, s', hrun, hsrip, hsr6, hsr1, hsrd, hsfr,
            hslen, hsk, hssh, hsmemfr⟩ :=
          parser_code_step hrip hr1 hr6 hr15 hr3 hr11 henc
            (by omega) hcW heW hnW hBc hSW (hwr _ hxpos (by omega))
            hlen16
        exact Or.inl ⟨x, k, s', hrun, hsrip, hsr6, hsr1, hsrd,
          henc, (by omega), hxb, hsfr, hslen, hsk, hssh, hsmemfr⟩
      · obtain ⟨nd, s3, hrun3, hs3rip, hs3r10, _hs3r0, hs3fr,
            hs3mem, hs3len16, hs3k⟩ :=
          parser_dispatch hs2rip hs2r0 hOp hs2len16
        have hs3r6 : s3.reg 6 = some c := by
          rw [hs3fr 6 (by decide)]; exact hs2r6
        have hs3r15 : s3.reg 15 = some e := by
          rw [hs3fr 15 (by decide), hs2fr 15 (by decide)
            (by decide), hs1fr 15]
          exact hr15
        obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len16,
            hs4k⟩ :=
          parser_code_tail_overrun hs3rip hs3r6 hs3r15 hs3r10 hce
            (by omega) hcW heW hs3len16
        exact Or.inr ⟨2 + 6 + nd + 5, s4, by
          rw [show 2 + 6 + nd + 5 = (2 + 6 + nd) + 5 by omega,
            run_add stage1Im (2 + 6 + nd) 5,
            show run stage1Im (2 + 6 + nd) (Out.ok s) =
              Out.ok s3 from by
              rw [show 2 + 6 + nd = (2 + 6) + nd by omega,
                run_add stage1Im (2 + 6) nd, hrun12, hrun3],
            hrun4],
          hs4rip,
          fun i hi0 hi1 hi2 hi6 hi10 =>
            (hs4fr i hi1 hi2 hi6).trans
              ((hs3fr i hi10).trans (hfr2 i hi0 hi1 hi2 hi6 hi10)),
          (hs4mem.trans hs3mem).trans hmem2, hs4len16,
          (hs4k.trans hs3k).trans hk2⟩
    · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len16, hs3k⟩ :=
        parser_disp_invalid_lo hs2rip hs2r0 hlo.1 hlo.2.1 hlo.2.2
          hs2len16
      exact Or.inr ⟨2 + 6 + 15, s3, by
        rw [show 2 + 6 + 15 = (2 + 6) + 15 by omega,
          run_add stage1Im (2 + 6) 15, hrun12, hrun3],
        hs3rip,
        fun i hi0 hi1 hi2 hi6 hi10 =>
          (hs3fr i).trans (hfr2 i hi0 hi1 hi2 hi6 hi10),
        hs3mem.trans hmem2, hs3len16, hs3k.trans hk2⟩
    · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len16, hs3k⟩ :=
        parser_disp_invalid_hi hs2rip hs2r0 hhi
          (by unfold W; omega) hs2len16
      exact Or.inr ⟨2 + 6 + 17, s3, by
        rw [show 2 + 6 + 17 = (2 + 6) + 17 by omega,
          run_add stage1Im (2 + 6) 17, hrun12, hrun3],
        hs3rip,
        fun i hi0 hi1 hi2 hi6 hi10 =>
          (hs3fr i).trans (hfr2 i hi0 hi1 hi2 hi6 hi10),
        hs3mem.trans hmem2, hs3len16, hs3k.trans hk2⟩
  · obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len16, hs2k⟩ :=
      parser_code_bounds_short hs1rip hs1r6 hs1r15 hce
        (by omega) hcW heW hs1len16
    exact Or.inr ⟨2 + 6, s2, by
      rw [run_add stage1Im 2 6, hrun1, hrun2],
      hs2rip,
      fun i _ _ hi2 _ _ => (hs2fr i hi2).trans (hs1fr i),
      hs2mem.trans hs1mem, hs2len16, hs2k.trans hs1k⟩


set_option maxHeartbeats 0 in
/-- The whole code loop, classified for arbitrary bytes: either
`n` instructions decode and fit before `e`, giving the list,
its `CodeSeq` and the `LenLay` scratch layout, or the run
reaches `aFail1` with memory outside the scratch byte range
written so far unchanged in shape. -/
theorem parser_code_loop_classify : ∀ (n : Nat) {c e B S : Nat}
    {s : X86Int.St},
    s.rip = base + 0x30b → s.reg 1 = some n →
    s.reg 6 = some c → s.reg 15 = some e →
    s.reg 3 = some B → s.reg 11 = some S →
    c ≤ e → c < W → e < W → n < W →
    B ≤ c → S + (e - B) < W →
    (S + (e - B) ≤ c ∨ e ≤ S + (c - B)) →
    (∀ a v, S + (c - B) ≤ a → a < S + (e - B) →
      (s.mem.wr a v).isSome) →
    (∀ a l, c ≤ a → l ≤ e - a →
      ∃ bs, s.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256) →
    s.regs.length = 16 →
    (∃ (code : List Instr) (k : Nat) (s' : X86Int.St),
      code.length = n ∧ c + codeBytes code ≤ e ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + codeBytes code) ∧
      CodeSeq s'.mem c code ∧ LenLay s'.mem B S c code ∧
      (∀ x ∈ code, InstrBound x) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c + codeBytes code - B) ≤ y ∨
        y + k ≤ S + (c - B) → s'.mem.rdN y k = s.mem.rdN y k)) ∨
    (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem) := by
  intro n
  induction n with
  | zero =>
    intro c e B S s hrip hr1 hr6 _ _ _ hce hcW _ _ _ _ _ _ _ hlen16
    refine Or.inl ⟨[], 0, s, rfl, by simpa using hce, rfl, hrip, hr1, ?_, trivial,
      trivial, fun x hx => (List.not_mem_nil hx).elim, ?_, hlen16, rfl, rfl, ?_⟩
    · show s.reg 6 = some (c + codeBytes [])
      rw [show codeBytes [] = 0 from rfl, Nat.add_zero]
      exact hr6
    · intro i _ _ _ _ _; exact rfl
    · intro y k _; exact rfl
  | succ n ih =>
    intro c e B S s hrip hr1 hr6 hr15 hr3 hr11 hce hcW heW hnW hBc
      hSW hdisj hwr hrdN hlen16
    have hrdNc : ∀ l, l ≤ e - c →
        ∃ bs, s.mem.rdN c l = some bs ∧ ∀ b ∈ bs, b < 256 :=
      fun l hl => hrdN c l (Nat.le_refl c) hl
    rcases parser_code_step_classify hrip hr1 hr6 hr15 hr3 hr11
        hce hcW heW hnW hBc (by omega)
        (fun l _ _ => hwr _ _ (Nat.le_refl _) (by omega)) hrdNc hlen16 with
      hstep | hfail
    · obtain ⟨x, k1, s1, hrun1, hs1rip, hs1r6, hs1r1, hs1rd, henc,
          hce1, hs1b, hs1fr, hs1len, hs1k, hs1sh, hs1memfr⟩ := hstep
      have hpos : 0 < x.enc.length := by
        obtain ⟨op, rest, heq, _⟩ := enc_head x
        rw [heq]; exact Nat.zero_lt_succ _
      have hs1r15 : s1.reg 15 = some e := by
        rw [hs1fr 15 (by decide) (by decide) (by decide)
          (by decide) (by decide)]
        exact hr15
      have hs1r3 : s1.reg 3 = some B := by
        rw [hs1fr 3 (by decide) (by decide) (by decide)
          (by decide) (by decide)]
        exact hr3
      have hs1r11 : s1.reg 11 = some S := by
        rw [hs1fr 11 (by decide) (by decide) (by decide)
          (by decide) (by decide)]
        exact hr11
      have hs1wr : ∀ a v, S + (c + x.enc.length - B) ≤ a →
          a < S + (e - B) → (s1.mem.wr a v).isSome := by
        intro a v ha1 ha2
        have h1 : (s.mem.wrN a [v]).isSome := by
          rw [← wr_eq_wrN]
          apply hwr <;> omega
        have h2 := wrN_some_of_wshape _ s.mem s1.mem hs1sh _ h1
        rwa [← wr_eq_wrN] at h2
      have hs1rdN : ∀ a l, c + x.enc.length ≤ a → l ≤ e - a →
          ∃ bs, s1.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256 := by
        intro a l ha hl
        obtain ⟨bs, hbs, hb⟩ := hrdN a l (by omega) hl
        refine ⟨bs, ?_, hb⟩
        rw [hs1memfr a l (by omega)]
        exact hbs
      rcases ih hs1rip hs1r1 hs1r6 hs1r15 hs1r3 hs1r11 (by omega)
          (by omega) heW (by omega) (by omega) (by omega)
          (by omega) hs1wr hs1rdN hs1len with hrest | hrestfail
      · obtain ⟨xs, k2, s2, hxslen, hs2fit, hrun2, hs2rip, hs2r1, hs2r6,
            hs2seq, hs2lay, hs2b, hs2fr, hs2len, hs2k, hs2sh, hs2memfr⟩ :=
          hrest
        have hrun : run stage1Im (k1 + k2) (Out.ok s) =
            Out.ok s2 := by
          rw [run_add stage1Im k1 k2, hrun1, hrun2]
        have hcb : codeBytes (x :: xs) =
            x.enc.length + codeBytes xs := rfl
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
        have hbcode : ∀ y ∈ x :: xs, InstrBound y := by
          intro y hy
          rcases List.mem_cons.mp hy with rfl | hy
          · exact hs1b
          · exact hs2b y hy
        refine Or.inl ⟨x :: xs, k1 + k2, s2, by simp [hxslen],
          by rw [hcb]; omega, hrun, hs2rip, hs2r1, ?_, hseq', hlay', hbcode, ?_, hs2len, ?_,
          ?_, ?_⟩
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
      · obtain ⟨k2, s2, hrun2, hs2rip, hs2fr, hs2len, hs2k,
            hs2sh⟩ := hrestfail
        exact Or.inr ⟨k1 + k2, s2, by
          rw [run_add stage1Im k1 k2, hrun1, hrun2],
          hs2rip,
          fun i hi0 hi1 hi2 hi6 hi10 =>
            (hs2fr i hi0 hi1 hi2 hi6 hi10).trans
              (hs1fr i hi0 hi1 hi2 hi6 hi10),
          hs2len, hs2k.trans hs1k, hs2sh.trans hs1sh⟩
    · obtain ⟨k, s', hrun, hsrip, hsfr, hsmem, hslen, hsk⟩ := hfail
      exact Or.inr ⟨k, s', hrun, hsrip, hsfr, hslen, hsk,
        congrArg wshape hsmem⟩


set_option maxHeartbeats 0 in
/-- Word-tail ann refusal: the annotation array of `12 * na`
bytes runs past the end of the file, so the bounds check at
the tail falls through to `aFail1`. -/
theorem parser_word_tail_short {sp1 p c e r na : Nat} {m48 : Mem}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x3db) (hr4 : s.reg 4 = some sp1)
    (hr6 : s.reg 6 = some c) (hr7 : s.reg 7 = some p)
    (hr15 : s.reg 15 = some e)
    (hpop : rd64 s.mem sp1 = some (r + 1))
    (hspW : sp1 + 8 < W) (hrW : r < W)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hnaW : 12 * na < W) (hc0 : c ≤ e)
    (hshort : e < c + 12 * na)
    (hslot32 : rd64 s.mem (p + 32) = some na)
    (hm48 : wr64 s.mem (p + 48) c = some m48)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ s'.reg 4 = some (sp1 + 8) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem := by
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
  have ht13fl : X86Int.cond t13.fl 3 = some false := by
    have hcf : t13.fl.cf = some true := by
      rw [ht13]
      show (aluF .sub (e - c) (12 * na)).2.cf = some true
      exact aluF_sub_cf_true (by omega)
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
  -- 0x402: jae not taken
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with rip := base + 0x404 } := by
    have h := step_jcc (im := stage1Im) (s := t13)
      (by rw [ht13rip]; exact hc.p402) ht13fl
    rw [ht13rip] at h
    simpa using h
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with rip := base + 0x404 } := ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x404 := by rw [ht14]
  have ht14r4 : t14.reg 4 = some (sp1 + 8) := by
    rw [ht14]
    show ({ t13 with rip := base + 0x404 }).reg 4 = some (sp1 + 8)
    rw [reg_with_rip, ht13fr 4, ht12fr 4 (by decide) (by decide)]
    exact ht3r4
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 →
      i ≠ 7 → t14.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht14]
    show ({ t13 with rip := base + 0x404 }).reg i = s.reg i
    rw [reg_with_rip, ht13fr i, ht12fr i hi0 hi2]
    exact ht3fr i hi0 hi1 hi2 hi4 hi6 hi7
  have ht14mem : t14.mem = m48 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x404 }).mem = m48
    rw [mem_with_rip]; exact ht13mem
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x404 }).regs.length = 16
    rw [len_with_rip]; exact ht13len
  have ht14k : t14.k = s.k := by rw [ht14]; exact ht13k
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  -- 0x404: jmp fail1
  obtain ⟨t15, r15, ht15rip, ht15fr, ht15mem, ht15len, ht15k⟩ :=
    jmp_step hc.p404 ht14rip
  have hrun13 : run stage1Im 13 (Out.ok s) = Out.ok t13 := by
    rw [show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r1,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, hrun2,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r3,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r4,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r5,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r6,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r7,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r8,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r9,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r10,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r11,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r12, r13]
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok t15 := by
    rw [show (15 : Nat) = 13 + 2 by omega, run_add stage1Im 13 2,
      hrun13, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
      r14, r15]
  refine ⟨t15, hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht15rip]; rfl
  · rw [ht15fr 4]; exact ht14r4
  · intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht15fr i]; exact ht14fr i hi0 hi1 hi2 hi4 hi6 hi7
  · rw [ht15len]; exact ht14len
  · rw [ht15k]; exact ht14k
  · rw [ht15mem, ht14mem]; exact wrN_wshape _ _ _ hm48


/-- Forward direction of `rdLE`: enough bytes decode to their
`leVal` and the remaining suffix. -/
theorem rdLE_forward {bs : List Nat} {k : Nat}
    (h : k ≤ bs.length) :
    rdLE bs k = some (leVal (bs.take k), bs.drop k) := by
  unfold rdLE
  rw [if_neg (by omega)]
  have hlen : (bs.take k).length = k := by
    rw [List.length_take, Nat.min_eq_left h]
  have h2 := fold_zip_range (bs.take k)
  rw [hlen] at h2
  rw [h2]

/-- Forward direction of `decAnn`: twelve bytes decode to the
entry read from their three `u32` slices. -/
theorem decAnn_forward {bs : List Nat} (h : 12 ≤ bs.length) :
    decAnn bs = some ((leVal (bs.take 4),
      ⟨leVal ((bs.drop 4).take 4), leVal ((bs.drop 8).take 4)⟩),
      bs.drop 12) := by
  have hstep : decAnn bs =
      (rdLE bs 4).bind (fun p =>
        (rdLE p.2 4).bind (fun q =>
          (rdLE q.2 4).bind
            (fun r => some ((p.1, ⟨q.1, r.1⟩), r.2)))) := rfl
  rw [hstep, Option.bind_eq_some_iff]
  refine ⟨(leVal (bs.take 4), bs.drop 4),
    rdLE_forward (by omega), ?_⟩
  rw [Option.bind_eq_some_iff]
  refine ⟨(leVal ((bs.drop 4).take 4), (bs.drop 4).drop 4),
    rdLE_forward (by rw [List.length_drop]; omega), ?_⟩
  rw [Option.bind_eq_some_iff]
  refine ⟨(leVal (((bs.drop 4).drop 4).take 4),
      ((bs.drop 4).drop 4).drop 4),
    rdLE_forward (by
      rw [List.length_drop, List.length_drop]; omega), ?_⟩
  simp only [List.drop_drop]

/-- Build the annotation list laid out at `p` from a whole-array
read: the list re-encodes to exactly the bytes read, every entry
is byte-bounded, the memory layout holds, and the semantic
decoder reads the same list. -/
theorem exists_ann_of_rdN : ∀ (na : Nat) {m : Mem} {p : Nat}
    {bs : List Nat},
    m.rdN p (12 * na) = some bs → (∀ b ∈ bs, b < 256) →
    ∃ ann : List (Nat × Height), ann.length = na ∧
      bs = ann.flatMap (fun x => leBytes x.1 4 ++
        leBytes x.2.d 4 ++ leBytes x.2.a 4) ∧
      (∀ x ∈ ann, AnnBound x) ∧ AnnLayB m p ann ∧
      decMany decAnn na bs = some (ann, []) := by
  intro na
  induction na with
  | zero =>
    intro m p bs h _
    have hlen : bs.length = 0 := rdN_length h
    have hbs : bs = [] := by
      cases bs with
      | nil => rfl
      | cons b t => simp at hlen
    subst hbs
    refine ⟨[], rfl, rfl, fun x hx => (List.not_mem_nil hx).elim,
      ?_, rfl⟩
    intro i hi
    exact (Nat.not_lt_zero i hi).elim
  | succ na ih =>
    intro m p bs h hb
    have hlen : bs.length = 12 * (na + 1) := rdN_length h
    have h12 : 12 ≤ bs.length := by omega
    have hdec : decAnn bs = some ((leVal (bs.take 4),
        ⟨leVal ((bs.drop 4).take 4), leVal ((bs.drop 8).take 4)⟩),
        bs.drop 12) := decAnn_forward h12
    obtain ⟨hhead, hbx⟩ := decAnn_inv hdec hb
    have htail : m.rdN (p + 12) (12 * na) = some (bs.drop 12) := by
      have hs := rdN_slice h (o := 12) (k := 12 * na) (by omega)
      rwa [List.take_of_length_le (by
        rw [List.length_drop, hlen]; omega)] at hs
    have hbt : ∀ b ∈ bs.drop 12, b < 256 := fun b hbm =>
      hb b (List.mem_of_mem_drop hbm)
    obtain ⟨ann, hannlen, hannflat, hannb, -, hanndec⟩ :=
      ih htail hbt
    have hflat : bs = ((leVal (bs.take 4),
        ⟨leVal ((bs.drop 4).take 4), leVal ((bs.drop 8).take 4)⟩) ::
        ann).flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4) := by
      rw [List.flatMap_cons, ← hannflat]
      exact hhead
    have hdecM : decMany decAnn (na + 1) bs =
        some ((leVal (bs.take 4),
          ⟨leVal ((bs.drop 4).take 4),
            leVal ((bs.drop 8).take 4)⟩) :: ann, []) := by
      have hstep : decMany decAnn (na + 1) bs =
          (decAnn bs).bind (fun q =>
            (decMany decAnn na q.2).bind
              (fun r => some (q.1 :: r.1, r.2))) := rfl
      rw [hstep, Option.bind_eq_some_iff]
      exact ⟨((leVal (bs.take 4),
          ⟨leVal ((bs.drop 4).take 4),
            leVal ((bs.drop 8).take 4)⟩), bs.drop 12), hdec, by
        rw [Option.bind_eq_some_iff]
        exact ⟨(ann, []), hanndec, rfl⟩⟩
    refine ⟨(leVal (bs.take 4),
        ⟨leVal ((bs.drop 4).take 4), leVal ((bs.drop 8).take 4)⟩) ::
        ann, by rw [List.length_cons, hannlen], hflat, ?_, ?_,
        hdecM⟩
    · intro y hy
      rcases List.mem_cons.mp hy with rfl | hy
      · exact hbx
      · exact hannb y hy
    · apply AnnLayB_of_rdN (m := m) (p := p)
      · rw [List.length_cons, hannlen, ← hflat]
        exact h
      · intro y hy
        rcases List.mem_cons.mp hy with rfl | hy
        · exact hbx
        · exact hannb y hy


set_option maxHeartbeats 0 in
/-- The five word header fields, classified: either all twenty
bytes fit and the fields are stored exactly as in
`parser_word_fields`, or the first short field's `call RDC`
reaches `aFail1` with `rsp` decremented by the call. -/
theorem parser_word_fields_classify {c e sp p : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr7 : s.reg 7 = some p)
    (hr4 : s.reg 4 = some sp)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hwr : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s.mem (p + off) v).isSome)
    (hrdN : ∀ a l, c ≤ a → l ≤ e - a →
      ∃ bs, s.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256) :
    (∃ bs, c + 20 ≤ e ∧ s.mem.rdN c 20 = some bs ∧
      ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x306 ∧ s'.reg 6 = some (c + 20) ∧
      s'.reg 4 = some sp ∧ s'.reg 7 = some p ∧
      rd64 s'.mem p = some (leVal (bs.take 4) % W) ∧
      rd64 s'.mem (p + 8) = some (leVal ((bs.drop 4).take 4) % W) ∧
      rd64 s'.mem (p + 16) = some (leVal ((bs.drop 8).take 4) % W) ∧
      rd64 s'.mem (p + 24) =
        some (leVal ((bs.drop 12).take 4) % W) ∧
      rd64 s'.mem (p + 32) =
        some (leVal ((bs.drop 16).take 4) % W) ∧
      rd64 s'.mem (p + 40) = some ((c + 20) % W) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      s'.mem.rdN c 20 = some bs ∧
      (∀ (off v : Nat), off + 8 ≤ 64 →
        (wr64 s'.mem (p + off) v).isSome) ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) →
        s'.mem.rdN x k = s.mem.rdN x k)) ∨
    (∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ s'.reg 4 = some (sp - 8) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem) := by
  have hc := stage1_parserCode
  have hbW : base + 0x400 < W := by decide
  by_cases hfit : c + 20 ≤ e
  · obtain ⟨bs, h20, -⟩ := hrdN c 20 (Nat.le_refl c) (by omega)
    exact Or.inl ⟨bs, hfit, h20,
      parser_word_fields hrip hr6 hr15 hr7 hr4 h20 hfit hcW heW
        hpW hroom hsp16 hspW hlen16 hEB hSB hES hwr⟩
  by_cases h4 : ¬ c + 4 ≤ e
  · -- field 0 is already short
    obtain ⟨n, s', hrun, hrip', hr4', hfr, -, hlen, hk, hsh⟩ :=
      call_rdc_fail hc.p2d6 hrip hr6 hr15 h4 hcW heW
        (stackRoom_mono hroom (by decide)) hr4 (by omega) hspW
        hlen16
    exact Or.inr ⟨n, s', hrun, hrip', hr4',
      fun i _ hi2 hi4 _ => hfr i hi2 hi4, hlen, hk, hsh⟩
  -- field 0 fits, store it
  obtain ⟨f0, hf0, -⟩ := hrdN c 4 (Nat.le_refl c) (by omega)
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r6, hs1r4, -, hs1fr, hs1len,
      hs1k, hs1sh, hs1room, -, hs1memfr⟩ :=
    field_store (site := base + 0x2d6) (off := 0) (slen := 3)
      hc.p2d6 hc.p2db (Nat.lt_of_le_of_lt (by omega) hbW) hrip hr6
      hr15 (by omega) hcW heW hf0 hroom hr4 hsp16 hspW hlen16 (by omega)
      hr7 (by omega) (by omega) (by
        have h := hwr 0 (leVal f0) (by decide)
        simpa using h)
  have hs1rip' : s1.rip = base + 0x2de := hs1rip
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have hs1r7 : s1.reg 7 = some p := by
    rw [hs1fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hr7
  by_cases h8 : ¬ c + 8 ≤ e
  · -- field 1 is short
    obtain ⟨n, s', hrun, hrip', hr4', hfr, -, hlen, hk, hsh⟩ :=
      call_rdc_fail hc.p2de hs1rip' hs1r6 hs1r15 (by omega)
        (by omega) heW (stackRoom_mono hs1room (by decide)) hs1r4
        (by omega) hspW hs1len
    exact Or.inr ⟨n1 + n, s', by
        rw [run_add stage1Im n1 n, hrun1, hrun], hrip', hr4',
      fun i hi0 hi2 hi4 hi6 =>
        (hfr i hi2 hi4).trans (hs1fr i hi0 hi2 hi4 hi6),
      hlen, hk.trans hs1k, hsh.trans hs1sh⟩
  -- field 1 fits, store it
  obtain ⟨f1, hf1, -⟩ := hrdN (c + 4) 4 (by omega) (by omega)
  have hf1' : s1.mem.rdN (c + 4) 4 = some f1 := by
    rw [hs1memfr (c + 4) 4 (by omega) (by omega)]; exact hf1
  have hw1 : ∀ v, (wr64 s1.mem (p + 8) v).isSome := by
    intro v
    exact wrN_some_of_wshape _ s.mem s1.mem hs1sh _
      (hwr 8 v (by decide))
  obtain ⟨n2, s2, hrun2, hs2rip, hs2r6, hs2r4, -, hs2fr, hs2len,
      hs2k, hs2sh, hs2room, -, hs2memfr⟩ :=
    field_store (site := base + 0x2de) (off := 8) (slen := 4)
      hc.p2de hc.p2e3 (Nat.lt_of_le_of_lt (by omega) hbW) hs1rip'
      hs1r6 hs1r15 (by omega) (by omega) heW hf1' hs1room hs1r4
      hsp16 hspW hs1len (by omega) hs1r7 (by omega) (by omega)
      (hw1 _)
  have hs2rip' : s2.rip = base + 0x2e7 := hs2rip
  have hs2r15 : s2.reg 15 = some e := by
    rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r15
  have hs2r7 : s2.reg 7 = some p := by
    rw [hs2fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r7
  have hs2r6' : s2.reg 6 = some (c + 8) := by
    have e1 : c + 4 + 4 = c + 8 := by omega
    rwa [e1] at hs2r6
  by_cases h12 : ¬ c + 12 ≤ e
  · -- field 2 is short
    obtain ⟨n, s', hrun, hrip', hr4', hfr, -, hlen, hk, hsh⟩ :=
      call_rdc_fail hc.p2e7 hs2rip' hs2r6' hs2r15 (by omega)
        (by omega) heW (stackRoom_mono hs2room (by decide)) hs2r4
        (by omega) hspW hs2len
    exact Or.inr ⟨n1 + n2 + n, s', by
        rw [show n1 + n2 + n = n1 + (n2 + n) by omega,
          run_add stage1Im n1 _, hrun1, run_add stage1Im n2 n,
          hrun2, hrun], hrip', hr4',
      fun i hi0 hi2 hi4 hi6 =>
        (hfr i hi2 hi4).trans
          ((hs2fr i hi0 hi2 hi4 hi6).trans (hs1fr i hi0 hi2 hi4 hi6)),
      hlen, (hk.trans hs2k).trans hs1k, (hsh.trans hs2sh).trans hs1sh⟩
  -- field 2 fits, store it
  obtain ⟨f2, hf2, -⟩ := hrdN (c + 8) 4 (by omega) (by omega)
  have hf2' : s2.mem.rdN (c + 8) 4 = some f2 := by
    rw [hs2memfr (c + 8) 4 (by omega) (by omega),
      hs1memfr (c + 8) 4 (by omega) (by omega)]
    exact hf2
  have hw2 : ∀ v, (wr64 s2.mem (p + 16) v).isSome := by
    intro v
    have h1 : (wr64 s1.mem (p + 16) v).isSome :=
      wrN_some_of_wshape _ s.mem s1.mem hs1sh _
        (hwr 16 v (by decide))
    exact wrN_some_of_wshape _ s1.mem s2.mem hs2sh _ h1
  obtain ⟨n3, s3, hrun3, hs3rip, hs3r6, hs3r4, -, hs3fr, hs3len,
      hs3k, hs3sh, hs3room, -, hs3memfr⟩ :=
    field_store (site := base + 0x2e7) (off := 16) (slen := 4)
      hc.p2e7 hc.p2ec (Nat.lt_of_le_of_lt (by omega) hbW) hs2rip'
      hs2r6' hs2r15 (by omega) (by omega) heW hf2' hs2room hs2r4
      hsp16 hspW hs2len (by omega) hs2r7 (by omega) (by omega)
      (hw2 _)
  have hs3rip' : s3.rip = base + 0x2f0 := hs3rip
  have hs3r15 : s3.reg 15 = some e := by
    rw [hs3fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r15
  have hs3r7 : s3.reg 7 = some p := by
    rw [hs3fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r7
  have hs3r6' : s3.reg 6 = some (c + 12) := by
    have e1 : c + 8 + 4 = c + 12 := by omega
    rwa [e1] at hs3r6
  by_cases h16 : ¬ c + 16 ≤ e
  · -- field 3 is short
    obtain ⟨n, s', hrun, hrip', hr4', hfr, -, hlen, hk, hsh⟩ :=
      call_rdc_fail hc.p2f0 hs3rip' hs3r6' hs3r15 (by omega)
        (by omega) heW (stackRoom_mono hs3room (by decide)) hs3r4
        (by omega) hspW hs3len
    exact Or.inr ⟨n1 + n2 + n3 + n, s', by
        rw [show n1 + n2 + n3 + n = n1 + (n2 + (n3 + n)) by omega,
          run_add stage1Im n1 _, hrun1, run_add stage1Im n2 _,
          hrun2, run_add stage1Im n3 n, hrun3, hrun], hrip', hr4',
      fun i hi0 hi2 hi4 hi6 =>
        (hfr i hi2 hi4).trans
          ((hs3fr i hi0 hi2 hi4 hi6).trans
            ((hs2fr i hi0 hi2 hi4 hi6).trans
              (hs1fr i hi0 hi2 hi4 hi6))),
      hlen, ((hk.trans hs3k).trans hs2k).trans hs1k,
      ((hsh.trans hs3sh).trans hs2sh).trans hs1sh⟩
  -- field 3 fits, store it
  obtain ⟨f3, hf3, -⟩ := hrdN (c + 12) 4 (by omega) (by omega)
  have hf3' : s3.mem.rdN (c + 12) 4 = some f3 := by
    rw [hs3memfr (c + 12) 4 (by omega) (by omega),
      hs2memfr (c + 12) 4 (by omega) (by omega),
      hs1memfr (c + 12) 4 (by omega) (by omega)]
    exact hf3
  have hw3 : ∀ v, (wr64 s3.mem (p + 24) v).isSome := by
    intro v
    have h1 : (wr64 s1.mem (p + 24) v).isSome :=
      wrN_some_of_wshape _ s.mem s1.mem hs1sh _
        (hwr 24 v (by decide))
    have h2 : (wr64 s2.mem (p + 24) v).isSome :=
      wrN_some_of_wshape _ s1.mem s2.mem hs2sh _ h1
    exact wrN_some_of_wshape _ s2.mem s3.mem hs3sh _ h2
  obtain ⟨n4, s4, hrun4, hs4rip, hs4r6, hs4r4, -, hs4fr, hs4len,
      hs4k, hs4sh, hs4room, -, -⟩ :=
    field_store (site := base + 0x2f0) (off := 24) (slen := 4)
      hc.p2f0 hc.p2f5 (Nat.lt_of_le_of_lt (by omega) hbW) hs3rip'
      hs3r6' hs3r15 (by omega) (by omega) heW hf3' hs3room hs3r4
      hsp16 hspW hs3len (by omega) hs3r7 (by omega) (by omega)
      (hw3 _)
  have hs4rip' : s4.rip = base + 0x2f9 := hs4rip
  have hs4r15 : s4.reg 15 = some e := by
    rw [hs4fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs3r15
  have hs4r6' : s4.reg 6 = some (c + 16) := by
    have e1 : c + 12 + 4 = c + 16 := by omega
    rwa [e1] at hs4r6
  by_cases h20f : ¬ c + 20 ≤ e
  · -- field 4 is short
    obtain ⟨n, s', hrun, hrip', hr4', hfr, -, hlen, hk, hsh⟩ :=
      call_rdc_fail hc.p2f9 hs4rip' hs4r6' hs4r15 (by omega)
        (by omega) heW (stackRoom_mono hs4room (by decide)) hs4r4
        (by omega) hspW hs4len
    exact Or.inr ⟨n1 + n2 + n3 + n4 + n, s', by
        rw [show n1 + n2 + n3 + n4 + n =
            n1 + (n2 + (n3 + (n4 + n))) by omega,
          run_add stage1Im n1 _, hrun1, run_add stage1Im n2 _,
          hrun2, run_add stage1Im n3 _, hrun3,
          run_add stage1Im n4 n, hrun4, hrun], hrip', hr4',
      fun i hi0 hi2 hi4 hi6 =>
        (hfr i hi2 hi4).trans
          ((hs4fr i hi0 hi2 hi4 hi6).trans
            ((hs3fr i hi0 hi2 hi4 hi6).trans
              ((hs2fr i hi0 hi2 hi4 hi6).trans
                (hs1fr i hi0 hi2 hi4 hi6)))),
      hlen, (((hk.trans hs4k).trans hs3k).trans hs2k).trans hs1k,
      (((hsh.trans hs4sh).trans hs3sh).trans hs2sh).trans hs1sh⟩
  exact (hfit (by omega)).elim

set_option maxHeartbeats 0 in
/-- One full word of the parser loop, classified for arbitrary
bytes: either a `Word` is built from the buffer (fields, code
loop and annotation array all fit), with the table entry, code
and length layouts left behind, or the run reaches `aFail1`
with the stack pointer at the entry value or one slot below. -/
theorem parser_word_step_classify {B e S sp p c r : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr1 : s.reg 1 = some (r + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr7 : s.reg 7 = some p) (hr4 : s.reg 4 = some sp)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hr1W : r + 1 < W) (hBc : B ≤ c)
    (hSW : S + (e - B) < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hScrB : S + (e - B) ≤ c ∨ e ≤ S + (c - B))
    (hScrE : S + (e - B) ≤ p ∨ p + 64 ≤ S + (c - B))
    (hScrStk : S + (e - B) ≤ sp - 8 ∨ sp ≤ S + (c - B))
    (hwrE : ∀ (off v : Nat), off + 8 ≤ 64 →
      (wr64 s.mem (p + off) v).isSome)
    (hwrS : ∀ a v, S + (c + 20 - B) ≤ a → a < S + (e - B) →
      (s.mem.wr a v).isSome)
    (hrdN : ∀ a l, c ≤ a → l ≤ e - a →
      ∃ bs, s.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256) :
    (∃ (wd : Word) (k : Nat) (s' : X86Int.St),
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some r ∧
      s'.reg 6 = some (c + 20 + codeBytes wd.code +
        12 * wd.ann.length) ∧
      s'.reg 7 = some (p + 64) ∧ s'.reg 4 = some sp ∧
      (∃ bs, s.mem.rdN c 20 = some bs ∧
        leVal (bs.take 4) % W = wd.ins ∧
        leVal ((bs.drop 4).take 4) % W = wd.outs ∧
        leVal ((bs.drop 8).take 4) % W = wd.locals ∧
        leVal ((bs.drop 12).take 4) % W = wd.code.length ∧
        leVal ((bs.drop 16).take 4) % W = wd.ann.length) ∧
      c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ e ∧
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
      AnnLayB s'.mem (c + 20 + codeBytes wd.code) wd.ann ∧
      WordBound wd ∧
      (∃ abs, s.mem.rdN (c + 20 + codeBytes wd.code)
          (12 * wd.ann.length) = some abs ∧
        decMany decAnn wd.ann.length abs = some (wd.ann, [])) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) →
        (S + (c + 20 + codeBytes wd.code - B) ≤ x ∨
          x + k ≤ S + (c + 20 - B)) →
        s'.mem.rdN x k = s.mem.rdN x k)) ∨
    (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (s'.reg 4 = some sp ∨ s'.reg 4 = some (sp - 8)) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem) := by
  by_cases hfit20 : ¬ c + 20 ≤ e
  · rcases parser_word_fields_classify hrip hr6 hr15 hr7 hr4 hcW
        heW hpW hroom hsp16 hspW hlen16 hEB hSB hES hwrE hrdN with
      ⟨bs, hfit, -⟩ | hffail
    · exact (hfit20 hfit).elim
    · obtain ⟨n, s', hrun, hrip', hr4', hfr, hlen, hk, hsh⟩ := hffail
      exact Or.inr ⟨n, s', hrun, hrip', Or.inr hr4',
        fun i hi0 _ hi2 hi4 hi6 _ _ => hfr i hi0 hi2 hi4 hi6,
        hlen, hk, hsh⟩
  have hc20 : c + 20 ≤ e := by omega
  obtain ⟨bs, h20, hb20⟩ := hrdN c 20 (Nat.le_refl c) (by omega)
  have hbslen : bs.length = 20 := rdN_length h20
  -- fields
  obtain ⟨nf, s1, hrunf, hs1rip, hs1r6, hs1r4, hs1r7, hs1s0, hs1s8,
      hs1s16, hs1s24, hs1s32, hs1s40, hs1fr, hs1len, hs1k, hs1sh,
      hs1room, hs1h20, hs1wr, hs1memfr⟩ :=
    parser_word_fields hrip hr6 hr15 hr7 hr4 h20 hc20 hcW heW
      hpW hroom hsp16 hspW hlen16 hEB hSB hES hwrE
  have hs1r1 : s1.reg 1 = some (r + 1) := by
    rw [hs1fr 1 (by decide) (by decide) (by decide) (by decide)]
    exact hr1
  have hroom1 : StackRoom s1 1 := by
    intro spx hspx i hi v
    exact hs1room spx hspx i (by omega) v
  -- enter
  obtain ⟨s2, hrun2, hs2rip, hs2r1, hs2r4, hs2pop, hs2s24, hs2fr,
      hs2len, hs2k, hs2sh, hs2memfr⟩ :=
    parser_code_enter hs1rip hs1r1 hs1r7 hs1r4 hroom1 (by omega)
      hspW (by omega) hs1s24 (by omega) hs1len
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
  have hs2wr : ∀ a v, S + (c + 20 - B) ≤ a → a < S + (e - B) →
      (s2.mem.wr a v).isSome := by
    intro a v ha1 ha2
    have h1 : (s.mem.wrN a [v]).isSome := by
      rw [← wr_eq_wrN]; exact hwrS a v ha1 ha2
    have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
    rwa [← wr_eq_wrN] at h2
  have hs2rdN : ∀ a l, c + 20 ≤ a → l ≤ e - a →
      ∃ bs, s2.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256 := by
    intro a l ha hl
    by_cases hl0 : l = 0
    · subst hl0
      exact ⟨[], rfl, fun b hb => (List.not_mem_nil hb).elim⟩
    obtain ⟨bs', hbs', hb'⟩ := hrdN a l (by omega) hl
    refine ⟨bs', ?_, hb'⟩
    rw [hs2memfr a l (by omega), hs1memfr a l (by omega) (by omega)]
    exact hbs'
  -- code loop, classified
  rcases parser_code_loop_classify _ hs2rip hs2r1 hs2r6 hs2r15
      hs2r3 hs2r11 (by omega) (by omega) heW
      (Nat.mod_lt _ (by unfold W; decide)) (by omega) hSW
      (by rcases hScrB with h | h <;> omega) hs2wr hs2rdN hs2len with
    hcl | hclfail
  · obtain ⟨code, k3, s3, hcodelen, hfit3, hrun3, hs3rip, hs3r1,
        hs3r6, hs3seq, hs3lay, hs3b, hs3fr, hs3len, hs3k, hs3sh,
        hs3memfr⟩ := hcl
    have slot3 : ∀ {off w : Nat}, off + 8 ≤ 48 →
        rd64 s2.mem (p + off) = some w →
        rd64 s3.mem (p + off) = some w := by
      intro off w hoff h
      unfold rd64
      rw [hs3memfr (p + off) 8 (by
        rcases hScrE with h | h <;> omega)]
      have h' := h
      unfold rd64 at h'
      exact h'
    have slot2 : ∀ {off w : Nat}, off + 8 ≤ 64 →
        rd64 s1.mem (p + off) = some w →
        rd64 s2.mem (p + off) = some w := by
      intro off w _ h
      unfold rd64
      rw [hs2memfr (p + off) 8 (by omega)]
      have h' := h
      unfold rd64 at h'
      exact h'
    have hs3s32 : rd64 s3.mem (p + 32) =
        some (leVal ((bs.drop 16).take 4) % W) :=
      slot3 (by decide) (slot2 (by decide) hs1s32)
    have hs3pop : rd64 s3.mem (sp - 8) = some (r + 1) := by
      unfold rd64
      rw [hs3memfr (sp - 8) 8 (by
        rcases hScrStk with h | h <;> omega)]
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
    have hs4r6 : s4.reg 6 = some (c + 20 + codeBytes code) := by
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
    have hs4s32 : rd64 s4.mem (p + 32) =
        some (leVal ((bs.drop 16).take 4) % W) := by
      rw [hs4mem]; exact hs3s32
    have hs4sh : wshape s4.mem = wshape s1.mem := by
      rw [hs4mem, hs3sh, hs2sh]
    obtain ⟨m48, hm48⟩ : ∃ m, wr64 s4.mem (p + 48)
        (c + 20 + codeBytes code) = some m := by
      have h1 : (s1.mem.wrN (p + 48)
          (bytes8 (c + 20 + codeBytes code))).isSome :=
        hs1wr 48 _ (by decide)
      have h2 := wrN_some_of_wshape _ s1.mem s4.mem hs4sh _ h1
      exact Option.isSome_iff_exists.mp h2
    have hnaW : 12 * (leVal ((bs.drop 16).take 4) % W) < W := by
      have hb4 : ∀ b ∈ (bs.drop 16).take 4, b < 256 :=
        fun b hbm => hb20 b
          (List.mem_of_mem_drop (List.mem_of_mem_take hbm))
      have hlen4 : ((bs.drop 16).take 4).length = 4 := by
        have h1 : (bs.drop 16).length = 4 := by
          rw [List.length_drop, hbslen]
        rw [List.take_of_length_le (by omega), h1]
      have hlt4 : leVal ((bs.drop 16).take 4) < 256 ^ 4 := by
        have h := leVal_bound hb4
        rwa [hlen4] at h
      have hmod : leVal ((bs.drop 16).take 4) % W =
          leVal ((bs.drop 16).take 4) :=
        Nat.mod_eq_of_lt (by unfold W; omega)
      rw [hmod]
      unfold W
      omega
    by_cases hfit : c + 20 + codeBytes code +
        12 * (leVal ((bs.drop 16).take 4) % W) ≤ e
    · -- annotation array fits: build the word and finish the tail
      obtain ⟨abs, habs, hab⟩ := hrdN (c + 20 + codeBytes code)
        (12 * (leVal ((bs.drop 16).take 4) % W)) (by omega)
        (by omega)
      obtain ⟨ann, hannlen, -, hannb, hannlay0, hanndec⟩ :=
        exists_ann_of_rdN _ habs hab
      obtain ⟨s5, hrun5, hs5rip, hs5r6, hs5r7, hs5r1, hs5r4, hs5s48,
          hs5s32, hs5fr, hs5len, hs5k, hs5sh, hs5memfr⟩ :=
        parser_word_tail hs4rip hs4r4 hs4r6 hs4r7 hs4r15 hs4pop
          (by omega) (by omega) (by omega) heW hpW hnaW (by omega)
          hs4s32 hm48 (by rw [hs4len]; exact hs3len)
      have hrun : run stage1Im (nf + 2 + k3 + 3 + 18) (Out.ok s) =
          Out.ok s5 := by
        rw [show nf + 2 + k3 + 3 + 18 = nf + (2 + (k3 + (3 + 18)))
            by omega,
          run_add stage1Im nf _, hrunf, run_add stage1Im 2 _, hrun2,
          run_add stage1Im k3 _, hrun3, run_add stage1Im 3 18, hrun4,
          hrun5]
      have slot5 : ∀ {off w : Nat}, off + 8 ≤ 48 →
          rd64 s1.mem (p + off) = some w →
          rd64 s5.mem (p + off) = some w := by
        intro off w hoff h
        have h3 : rd64 s3.mem (p + off) = some w :=
          slot3 hoff (slot2 (by omega) h)
        have h4 : rd64 s4.mem (p + off) = some w := by
          rw [hs4mem]; exact h3
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
      have hannlay1 : AnnLayB s1.mem (c + 20 + codeBytes code)
          ann :=
        AnnLayB_of_frame hannlay0 (fun x k _ _ =>
          hs1memfr x k (by omega) (by omega))
      have hannlay2 : AnnLayB s2.mem (c + 20 + codeBytes code)
          ann :=
        AnnLayB_of_frame hannlay1 (fun x k _ _ =>
          hs2memfr x k (by omega))
      have hannlay3 : AnnLayB s3.mem (c + 20 + codeBytes code)
          ann :=
        AnnLayB_of_frame hannlay2 (by
          intro x k _ _
          apply hs3memfr
          rcases hScrB with h | h <;> omega)
      have hannlay4 : AnnLayB s4.mem (c + 20 + codeBytes code)
          ann := by
        rw [hs4mem]; exact hannlay3
      have hannlay5 : AnnLayB s5.mem (c + 20 + codeBytes code)
          ann :=
        AnnLayB_of_frame hannlay4 (fun x k _ _ =>
          hs5memfr x k (by omega))
      have hfb : ∀ a : Nat, a ≤ 16 →
          leVal ((bs.drop a).take 4) % W < 256 ^ 4 := by
        intro a ha
        have hl : ((bs.drop a).take 4).length = 4 := by
          rw [List.length_take, List.length_drop, hbslen]
          omega
        have hb := leVal_bound (bs := (bs.drop a).take 4)
          (fun b hb => hb20 b
            (List.mem_of_mem_drop (List.mem_of_mem_take hb)))
        rw [hl] at hb
        have hm : leVal ((bs.drop a).take 4) % W =
            leVal ((bs.drop a).take 4) :=
          Nat.mod_eq_of_lt (by unfold W; omega)
        rwa [hm]
      have hwbound : WordBound (⟨leVal (bs.take 4) % W,
          leVal ((bs.drop 4).take 4) % W,
          leVal ((bs.drop 8).take 4) % W, code, ann⟩ : Word) :=
        ⟨by simpa using hfb 0 (by decide), hfb 4 (by decide),
          hfb 8 (by decide), by rw [hcodelen]; exact hfb 12 (by decide),
          by rw [hannlen]; exact hfb 16 (by decide), hs3b, hannb⟩
      refine Or.inl ⟨⟨leVal (bs.take 4) % W,
          leVal ((bs.drop 4).take 4) % W,
          leVal ((bs.drop 8).take 4) % W, code, ann⟩,
        nf + 2 + k3 + 3 + 18, s5, hrun, hs5rip, hs5r1, ?_, hs5r7,
        ?_, ⟨bs, h20, rfl, rfl, rfl, hcodelen.symm, hannlen.symm⟩,
        ?_, ?_, ?_, ?_, ?_, ?_, ?_, hs5s48, ?_, ?_, hannlay5, hwbound,
        ⟨abs, by rwa [hannlen], by rwa [hannlen]⟩, ?_, hs5len, ?_,
        hs5sh', hroom5, ?_⟩
      · show s5.reg 6 = some (c + 20 + codeBytes code +
          12 * ann.length)
        rw [hannlen]; exact hs5r6
      · have h1 : sp - 8 + 8 = sp := by omega
        rw [hs5r4, h1]
      · rw [hannlen]; exact hfit
      · exact slot5 (off := 0) (by decide) (by simpa using hs1s0)
      · exact slot5 (by decide) hs1s8
      · exact slot5 (by decide) hs1s16
      · have h := slot5 (by decide) hs1s24
        rwa [← hcodelen] at h
      · show rd64 s5.mem (p + 32) = some ann.length
        rw [hannlen]; exact hs5s32
      · exact slot5 (by decide) hs1s40
      · exact CodeSeq_of_frame hs3seq (fun y k hy hk => by
          have e1 : s5.mem.rdN y k = s4.mem.rdN y k :=
            hs5memfr y k (by
              rcases hEB with h | h
              · refine Or.inl ?_
                show p + 48 + 8 ≤ y
                omega
              · refine Or.inr ?_
                show y + k ≤ p + 48
                exact Nat.le_trans hk
                  (Nat.le_trans hfit3
                    (Nat.le_trans h (Nat.le_add_right p 48))))
          rw [e1, hs4mem])
      · apply LenLay_of_frame code (by omega) hs3lay
        intro a _ _
        have e1 : s5.mem.rd a = s4.mem.rd a :=
          rd_eq_of_rdN1 (hs5memfr a 1 (by omega))
        rw [e1, hs4mem]
      · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
        rw [hs5fr i hi0 hi1 hi2 hi4 hi6 hi7, hs4fr i,
          hs3fr i hi0 hi1 hi2 hi6 hi10,
          hs2fr i hi1 hi4]
        exact hs1fr i hi0 hi2 hi4 hi6
      · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
      · intro x k hd1 hd2 hd3
        rw [hs5memfr x k (by omega), hs4mem, hs3memfr x k hd3,
          hs2memfr x k (by omega), hs1memfr x k hd1 hd2]
    · -- annotation array overruns: the tail refuses
      obtain ⟨s5, hrun5, hs5rip, hs5r4, hs5fr, hs5len, hs5k,
          hs5sh⟩ :=
        parser_word_tail_short hs4rip hs4r4 hs4r6 hs4r7 hs4r15
          hs4pop (by omega) (by omega) (by omega) heW hpW hnaW
          hfit3 (by omega) hs4s32 hm48
          (by rw [hs4len]; exact hs3len)
      have hrun : run stage1Im (nf + 2 + k3 + 3 + 15) (Out.ok s) =
          Out.ok s5 := by
        rw [show nf + 2 + k3 + 3 + 15 = nf + (2 + (k3 + (3 + 15)))
            by omega,
          run_add stage1Im nf _, hrunf, run_add stage1Im 2 _, hrun2,
          run_add stage1Im k3 _, hrun3, run_add stage1Im 3 15, hrun4,
          hrun5]
      refine Or.inr ⟨nf + 2 + k3 + 3 + 15, s5, hrun, hs5rip,
        Or.inl ?_, ?_, hs5len, ?_, ?_⟩
      · have h1 : sp - 8 + 8 = sp := by omega
        rw [hs5r4, h1]
      · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
        rw [hs5fr i hi0 hi1 hi2 hi4 hi6 hi7, hs4fr i,
          hs3fr i hi0 hi1 hi2 hi6 hi10,
          hs2fr i hi1 hi4]
        exact hs1fr i hi0 hi2 hi4 hi6
      · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
      · rw [hs5sh, hs4mem, hs3sh, hs2sh, hs1sh]
  · -- the code loop itself refuses
    obtain ⟨k3, s3, hrun3, hs3rip, hs3fr, hs3len, hs3k, hs3sh⟩ :=
      hclfail
    have hrun : run stage1Im (nf + 2 + k3) (Out.ok s) =
        Out.ok s3 := by
      rw [show nf + 2 + k3 = nf + (2 + k3) by omega,
        run_add stage1Im nf _, hrunf, run_add stage1Im 2 k3, hrun2,
        hrun3]
    refine Or.inr ⟨nf + 2 + k3, s3, hrun, hs3rip, Or.inr ?_, ?_,
      hs3len, ?_, ?_⟩
    · rw [hs3fr 4 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hs2r4
    · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
      rw [hs3fr i hi0 hi1 hi2 hi6 hi10, hs2fr i hi1 hi4]
      exact hs1fr i hi0 hi2 hi4 hi6
    · rw [hs3k, hs2k, hs1k]
    · rw [hs3sh, hs2sh, hs1sh]



set_option maxHeartbeats 0 in
/-- The whole parser word loop, classified for arbitrary bytes:
either a word list of the given length is built from the buffer,
laid out in the table with its code, length and annotation
layouts, or the run reaches `aFail1` with the stack pointer at
the entry value or one slot below. -/
theorem parser_words_loop_classify : ∀ (n : Nat)
    {B e S sp p c : Nat} {s : X86Int.St},
    s.rip = base + 0x2cc → s.reg 1 = some n →
    s.reg 6 = some c → s.reg 7 = some p →
    s.reg 15 = some e → s.reg 4 = some sp →
    s.reg 3 = some B → s.reg 11 = some S →
    B ≤ c → c ≤ e → e < W →
    p + 64 * n < W → S + (e - B) < W → n < W →
    StackRoom s 2 → 16 ≤ sp → sp < W → s.regs.length = 16 →
    (p + 64 * n ≤ c ∨ e ≤ p) →
    (p + 64 * n ≤ sp - 16 ∨ sp ≤ p) →
    (S + (e - B) ≤ p ∨ p + 64 * n ≤ S + (c - B)) →
    (e ≤ sp - 16 ∨ sp ≤ c) →
    (S + (e - B) ≤ c ∨ e ≤ S + (c - B)) →
    (S + (e - B) ≤ sp - 16 ∨ sp ≤ S + (c - B)) →
    (∀ (off v : Nat), off + 8 ≤ 64 * n →
      (wr64 s.mem (p + off) v).isSome) →
    (∀ a v, S + (c - B) ≤ a → a < S + (e - B) →
      (s.mem.wr a v).isSome) →
    (∀ a l, c ≤ a → l ≤ e - a →
      ∃ bs, s.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256) →
    (∃ (wds : List Word) (k : Nat) (s' : X86Int.St),
      wds.length = n ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + wordsBytes wds) ∧
      s'.reg 7 = some (p + 64 * n) ∧
      s'.reg 4 = some sp ∧
      c + wordsBytes wds ≤ e ∧
      WordsLay s.mem c wds ∧ TabLay s'.mem B S p c wds ∧
      (∀ wd ∈ wds, WordBound wd) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 * n ≤ x ∨ x + k ≤ p) →
        (S + (c + wordsBytes wds - B) ≤ x ∨
          x + k ≤ S + (c - B)) →
        s'.mem.rdN x k = s.mem.rdN x k)) ∨
    (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (s'.reg 4 = some sp ∨ s'.reg 4 = some (sp - 8)) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem) := by
  intro n
  induction n with
  | zero =>
    intro B e S sp p c s hrip hr1 hr6 hr7 _ hr4 _ _ _ hce _ _ _ _
      hroom _ _ hlen16 _ _ _ _ _ _ _ _ _
    refine Or.inl ⟨[], 0, s, rfl, rfl, hrip, hr1, ?_, ?_, hr4, ?_,
      trivial, trivial, fun wd hwd => (List.not_mem_nil hwd).elim,
      ?_, hlen16, rfl, rfl, hroom, ?_⟩
    · show s.reg 6 = some (c + wordsBytes [])
      exact hr6
    · show s.reg 7 = some (p + 64 * 0)
      exact hr7
    · have hz : wordsBytes [] = 0 := rfl
      omega
    · intro i _ _ _ _ _ _ _; exact rfl
    · intro x k _ _ _; exact rfl
  | succ r ih =>
    intro B e S sp p c s hrip hr1 hr6 hr7 hr15 hr4 hr3 hr11 hBc hce
      heW hpW hSW hnW hroom hsp16 hspW hlen16 hTB hTStk hTScr hSB
      hScrB hScrStk hwrE hwrS hrdN
    have hcW : c < W := by omega
    -- head: enter the body
    obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
      parser_words_head_pos hrip hr1 (by omega) hnW
    have hs1len16 : s1.regs.length = 16 := by rw [hs1len]; exact hlen16
    have hs1r : ∀ i v, s.reg i = some v → s1.reg i = some v := by
      intro i v h
      rw [hs1fr i]; exact h
    -- one word step, classified
    rcases parser_word_step_classify hs1rip (hs1r 1 _ hr1)
        (hs1r 6 _ hr6) (hs1r 15 _ hr15) (hs1r 7 _ hr7)
        (hs1r 4 _ hr4) (hs1r 3 _ hr3) (hs1r 11 _ hr11) hcW heW
        (by omega) hnW hBc hSW
        (StackRoom_of_mem hs1mem (hs1fr 4) hroom)
        hsp16 hspW hs1len16 (by omega) hSB (by omega) hScrB
        (by omega) (by omega)
        (by
          intro off v hoff
          rw [hs1mem]
          exact hwrE off v (by omega))
        (by
          intro a v ha1 ha2
          rw [hs1mem]
          exact hwrS a v (by omega) ha2)
        (by
          intro a l ha hl
          obtain ⟨bs, hbs, hb⟩ := hrdN a l ha hl
          exact ⟨bs, by rwa [hs1mem], hb⟩) with hstep | hstepfail
    · obtain ⟨wd, k1, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r7, hs2r4,
          hs2head, hs2fit, hs2s0, hs2s8, hs2s16, hs2s24, hs2s32,
          hs2s40, hs2s48, hs2seq, hs2lay, hs2annlay, hs2annb, -,
          hs2fr, hs2len, hs2k, hs2sh, hs2room, hs2memfr⟩ := hstep
      have hs2sh' : wshape s2.mem = wshape s.mem := by
        rw [hs2sh, hs1mem]
      have hs2r15 : s2.reg 15 = some e := by
        rw [hs2fr 15 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide), hs1fr 15]
        exact hr15
      have hs2r3 : s2.reg 3 = some B := by
        rw [hs2fr 3 (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide), hs1fr 3]
        exact hr3
      have hs2r11 : s2.reg 11 = some S := by
        rw [hs2fr 11 (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide), hs1fr 11]
        exact hr11
      have hs2wrE : ∀ (off v : Nat), off + 8 ≤ 64 * r →
          (wr64 s2.mem (p + 64 + off) v).isSome := by
        intro off v hoff
        have h1 : (s.mem.wrN (p + 64 + off) (bytes8 v)).isSome := by
          have h2 := hwrE (64 + off) v (by omega)
          rwa [show p + (64 + off) = p + 64 + off by omega] at h2
        exact wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
      have hs2wrS : ∀ a v,
          S + (c + 20 + codeBytes wd.code + 12 * wd.ann.length -
            B) ≤ a → a < S + (e - B) → (s2.mem.wr a v).isSome := by
        intro a v ha1 ha2
        have h1 : (s.mem.wrN a [v]).isSome := by
          rw [← wr_eq_wrN]
          exact hwrS a v (by omega) ha2
        have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
        rwa [← wr_eq_wrN] at h2
      have hs2rdN : ∀ a l,
          c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ a →
          l ≤ e - a →
          ∃ bs, s2.mem.rdN a l = some bs ∧
            ∀ b ∈ bs, b < 256 := by
        intro a l ha hl
        by_cases hl0 : l = 0
        · subst hl0
          exact ⟨[], rfl, fun b hb => (List.not_mem_nil hb).elim⟩
        obtain ⟨bs, hbs, hb⟩ := hrdN a l (by omega) hl
        refine ⟨bs, ?_, hb⟩
        rw [hs2memfr a l (by omega) (by omega) (by omega), hs1mem]
        exact hbs
      -- induction over the rest
      rcases ih hs2rip hs2r1 hs2r6 hs2r7 hs2r15 hs2r4 hs2r3 hs2r11
          (by omega) hs2fit heW (by omega) hSW (by omega) hs2room
          hsp16 hspW hs2len (by omega) (by omega) (by omega)
          (by omega) (by omega) (by omega) hs2wrE hs2wrS hs2rdN with
        hrest | hrestfail
      · obtain ⟨wds, k2, s3, hwdslen, hrun3, hs3rip, hs3r1, hs3r6,
            hs3r7, hs3r4, hs3fit, hs3layW, hs3tab, hs3annb, hs3fr,
            hs3len, hs3k, hs3sh, hs3room, hs3memfr⟩ := hrest
        have hwb : wordsBytes (wd :: wds) =
            (20 + codeBytes wd.code + 12 * wd.ann.length) +
              wordsBytes wds := rfl
        have hrun : run stage1Im (2 + k1 + k2) (Out.ok s) =
            Out.ok s3 := by
          rw [show 2 + k1 + k2 = 2 + (k1 + k2) by omega,
            run_add stage1Im 2 _, hrun1, run_add stage1Im k1 k2,
            hrun2, hrun3]
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
          apply rd_eq_of_rdN1
          apply hs3memfr
          · rcases hScrStk with h | h
            · exact Or.inl (by omega)
            · exact Or.inr (by omega)
          · rcases hTScr with h | h
            · exact Or.inr (by omega)
            · exact Or.inl (by omega)
          · exact Or.inr (by omega)
        have hann3 : AnnLayB s3.mem (c + 20 + codeBytes wd.code)
            wd.ann := by
          apply AnnLayB_of_frame hs2annlay
          intro y k hy hk
          apply hs3memfr <;> omega
        have htab : TabLay s3.mem B S p c (wd :: wds) :=
          ⟨slot3 (off := 0) (by decide) (by simpa using hs2s0),
            slot3 (by decide) hs2s8, slot3 (by decide) hs2s16,
            slot3 (by decide) hs2s24, slot3 (by decide) hs2s32,
            slot3 (by decide) hs2s40, slot3 (by decide) hs2s48,
            hseq3, hlay3, hann3, hs3tab⟩
        have hlayW : WordsLay s.mem c (wd :: wds) := by
          obtain ⟨bs, h20, e0, e1, e2, e3, e4⟩ := hs2head
          refine ⟨⟨bs, by rwa [hs1mem] at h20, e0, e1, e2, e3, e4⟩,
            ?_, ?_, ?_⟩
          · apply CodeSeq_of_frame hs2seq
            intro y k hy hk
            rw [← hs1mem]
            exact (hs2memfr y k (by omega) (by omega)
              (by omega)).symm
          · apply AnnLayB_of_frame hs2annlay
            intro y k hy hk
            rw [← hs1mem]
            exact (hs2memfr y k (by omega) (by omega)
              (by omega)).symm
          · apply WordsLay_of_frame wds hs3layW
            intro y k hy hk
            rw [← hs1mem]
            exact (hs2memfr y k (by omega) (by omega)
              (by omega)).symm
        refine Or.inl ⟨wd :: wds, 2 + k1 + k2, s3, by
            rw [List.length_cons, hwdslen], hrun, hs3rip, hs3r1,
          ?_, ?_, hs3r4, by rw [hwb]; omega, hlayW, htab, ?_, ?_,
          hs3len, ?_, ?_, hs3room, ?_⟩
        · show s3.reg 6 = some (c + wordsBytes (wd :: wds))
          rw [hwb]
          have h := hs3r6
          rwa [show c + 20 + codeBytes wd.code +
              12 * wd.ann.length + wordsBytes wds =
            c + (20 + codeBytes wd.code + 12 * wd.ann.length +
              wordsBytes wds) by omega] at h
        · show s3.reg 7 = some (p + 64 * (r + 1))
          have h := hs3r7
          rwa [show p + 64 + 64 * r = p + 64 * (r + 1) by omega]
            at h
        · intro wd' hwd
          rcases List.mem_cons.mp hwd with rfl | hwd
          · exact hs2annb
          · exact hs3annb wd' hwd
        · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
          rw [hs3fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10,
            hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10, hs1fr i]
        · rw [hs3k, hs2k, hs1k]
        · rw [hs3sh, hs2sh, hs1mem]
        · intro x k hd1 hd2 hd3
          rw [hs3memfr x k hd1 (by omega) (by omega),
            hs2memfr x k hd1 (by omega) (by omega), hs1mem]
      · obtain ⟨k2, s3, hrun3, hs3rip, hs3r4, hs3fr, hs3len, hs3k,
            hs3sh⟩ := hrestfail
        exact Or.inr ⟨2 + k1 + k2, s3, by
            rw [show 2 + k1 + k2 = 2 + (k1 + k2) by omega,
              run_add stage1Im 2 _, hrun1, run_add stage1Im k1 k2,
              hrun2, hrun3], hs3rip, hs3r4,
          fun i hi0 hi1 hi2 hi4 hi6 hi7 hi10 =>
            (hs3fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10).trans
              ((hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10).trans
                (hs1fr i)),
          hs3len, (hs3k.trans hs2k).trans hs1k,
          by rw [hs3sh, hs2sh, hs1mem]⟩
    · obtain ⟨k1, s2, hrun2, hs2rip, hs2r4, hs2fr, hs2len, hs2k,
          hs2sh⟩ := hstepfail
      exact Or.inr ⟨2 + k1, s2, by
          rw [run_add stage1Im 2 k1, hrun1, hrun2], hs2rip, hs2r4,
        fun i hi0 hi1 hi2 hi4 hi6 hi7 hi10 =>
          (hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10).trans (hs1fr i),
        hs2len, hs2k.trans hs1k, by rw [hs2sh, hs1mem]⟩


/-- The trailing checks, classified: either the cursor is at
the buffer end, `main` is a valid word index with zero `ins`,
and control falls into the checker entry, or one of the three
checks reaches `aFail1` with memory unchanged. -/
theorem parser_trailing_classify {c e mn nw T : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw) (hr14 : s.reg 14 = some T)
    (hcW : c < W) (heW : e < W) (hTW : T + 64 * mn < W)
    (hrdslot : ∃ v, rd64 s.mem (T + 64 * mn) = some v)
    (hlen16 : s.regs.length = 16) :
    (∃ s' : X86Int.St,
      run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧ s'.reg 0 = some 0 ∧
      c = e ∧ mn < nw ∧ rd64 s.mem (T + 64 * mn) = some 0 ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) ∨
    (∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k) := by
  by_cases hne : c ≠ e
  · obtain ⟨s', hrun, hrip', hfr, hmem, hlen, hk⟩ :=
      parser_trailing_cursor_fail hrip hr6 hr15 hne hcW heW
    exact Or.inr ⟨3, s', hrun, hrip',
      fun i _ _ => hfr i, hmem, by rw [hlen]; exact hlen16, hk⟩
  have hceq : c = e := by omega
  by_cases hmn : ¬ mn < nw
  · obtain ⟨s', hrun, hrip', hfr, hmem, hlen, hk⟩ :=
      parser_trailing_main_ge hrip hr6 hr15 hr12 hr13 hceq hcW
        heW (by omega) hlen16
    exact Or.inr ⟨5, s', hrun, hrip',
      fun i _ _ => hfr i, hmem, hlen, hk⟩
  have hmn' : mn < nw := by omega
  obtain ⟨v, hvslot⟩ := hrdslot
  by_cases hv : v = 0
  · rw [hv] at hvslot
    obtain ⟨s', hrun, hrip', hr0, hfr, hmem, hlen, hk⟩ :=
      parser_trailing_ok hrip hr6 hr15 hr12 hr13 hr14 hceq hcW
        heW hmn' hTW hvslot hlen16
    exact Or.inl ⟨s', hrun, hrip', hr0, hceq, hmn', hvslot, hfr,
      hmem, hlen, hk⟩
  · obtain ⟨s', hrun, hrip', hfr, hmem, hlen, hk⟩ :=
      parser_trailing_slot_fail hrip hr6 hr15 hr12 hr13 hr14 hceq
        hcW heW hmn' hTW hvslot hv hlen16
    exact Or.inr ⟨16, s', hrun, hrip', hfr, hmem, hlen, hk⟩


/-- A writable `find` hit in a prefix of a reference memory,
transported across an append and then across equal write shapes. -/
theorem find_rw_of_wshape_append_left {m pre extra : Mem} {a : Nat}
    (hsh : wshape m = wshape (pre ++ extra))
    (h : ∃ i r, Mem.find pre a 0 = some (i, r) ∧ r.rw = true) :
    ∃ i r, Mem.find m a 0 = some (i, r) ∧ r.rw = true := by
  apply find_rw_of_wshape hsh
  obtain ⟨i, r, hf, hrw⟩ := h
  exact ⟨i, r, find_append_left pre extra a 0 hf, hrw⟩

/-- A writable `find` hit in the last region of a reference
memory, transported across equal write shapes. -/
theorem find_rw_of_wshape_append_last {m pre : Mem} {r : Region}
    {a : Nat}
    (hsh : wshape m = wshape (pre ++ [r]))
    (hnon : ∀ r' ∈ pre, ¬ (r'.base ≤ a ∧ a < r'.hi))
    (hmem : r.base ≤ a ∧ a < r.hi) (hrw : r.rw = true) :
    ∃ i r', Mem.find m a 0 = some (i, r') ∧ r'.rw = true := by
  apply find_rw_of_wshape hsh
  exact ⟨0 + pre.length, r, find_append_last pre 0 hnon hmem, hrw⟩

/-- Readability from a writable `find` hit in a prefix of a
reference memory of the same write shape. -/
theorem rd_isSome_of_wshape_append_left {m pre extra : Mem}
    {a : Nat}
    (hsh : wshape m = wshape (pre ++ extra))
    (h : ∃ i r, Mem.find pre a 0 = some (i, r) ∧ r.rw = true) :
    (m.rd a).isSome := by
  apply rd_isSome_of_find
  obtain ⟨i, r, hf, -⟩ :=
    find_rw_of_wshape_append_left hsh h
  exact ⟨i, r, hf⟩

/-- Writability from a writable `find` hit in a prefix of a
reference memory of the same write shape. -/
theorem wr_isSome_of_wshape_append_left {m pre extra : Mem}
    {a u : Nat}
    (hsh : wshape m = wshape (pre ++ extra))
    (h : ∃ i r, Mem.find pre a 0 = some (i, r) ∧ r.rw = true) :
    (m.wr a u).isSome := by
  obtain ⟨i, r, hf, hrw⟩ :=
    find_rw_of_wshape_append_left hsh h
  exact wr_isSome_iff.mpr ⟨i, r, hf, hrw⟩

/-- Writability from a writable `find` hit in the last region of
a reference memory of the same write shape. -/
theorem wr_isSome_of_wshape_append_last {m pre : Mem}
    {r : Region} {a u : Nat}
    (hsh : wshape m = wshape (pre ++ [r]))
    (hnon : ∀ r' ∈ pre, ¬ (r'.base ≤ a ∧ a < r'.hi))
    (hmem : r.base ≤ a ∧ a < r.hi) (hrw : r.rw = true) :
    (m.wr a u).isSome := by
  obtain ⟨i, r', hf, hrw'⟩ :=
    find_rw_of_wshape_append_last hsh hnon hmem hrw
  exact wr_isSome_iff.mpr ⟨i, r', hf, hrw'⟩


set_option maxHeartbeats 0 in
/-- E3: from the loaded state (file in the arena buffer, the two
parser placements next in the witness oracle) to the checker
entry, classified for arbitrary file bytes.  Success builds the
data segments and word list laid out by the parser, with the
cursor exactly at the file end and the main word's `ins` slot
zero; every refusal composes to exit 1. -/
theorem parser_loaded_to_checker_classify {f : List Nat}
    {A sp : Nat} {c1 c2 buf : Region} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some A)
    (hr13 : s.reg 13 = some f.length) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (horacle : s.k.oracle =
      [.at (A + arenaFileBytes f),
       .at (A + (arenaFileBytes f + arenaTableBytes f))])
    (hmem : s.mem = [c1, c2, buf]) (hbufb : buf.base = A)
    (hbufhi : ∃ cap, buf.hi = A + cap ∧ cap ≤ arenaFileBytes f)
    (hrdN : s.mem.rdN A f.length = some f)
    (hwin : ∃ bs, s.mem.rdN A 20 = some bs)
    (hfB : ∀ b ∈ f, b < 256)
    (hfreeInit : ∀ r ∈ ([c1, c2] : Mem),
      A + stage1Mf f ≤ r.base ∨ r.hi ≤ A)
    (hAmod : A % page = 0) (hApos : 0 < A)
    (hAbound : A + stage1Mf f ≤ 2 ^ 47)
    (hAframe : ∀ x k, A ≤ x → x + k ≤ A + f.length →
      x + k ≤ sp - 16 ∨ sp ≤ x)
    (hdisjc : c1.hi ≤ c2.base) (hc2rw : c2.rw = true)
    (hc2lo : c2.base + 8 * 6 ≤ sp - 8) (hc2hi2 : sp - 8 ≤ c2.hi)
    (hstackTop : sp + 48 ≤ c2.hi)
    (hlen16 : s.regs.length = 16)
    (hsp80 : 80 ≤ sp) (hspW : sp < W) (hsp48W : sp + 48 < W)
    (hregs : RegsOk s) (hroom10 : StackRoom s 10) :
    (∃ k, run stage1Im k (Out.ok s) = Out.exit 1) ∨
    (∃ bs ds wds c k s', s.mem.rdN A 20 = some bs ∧
      bs = f.take 20 ∧ 16 ≤ f.length ∧ leVal (bs.take 4) % 2 ^ 32 = 827478099 ∧
      ds.length = leVal ((bs.drop 12).take 4) % 2 ^ 32 ∧
      DataSeq s.mem (A + 16) ds c ∧ c ≤ A + f.length ∧
      20 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) ≤
        A + f.length - c ∧
      wds.length = leVal ((bs.drop 8).take 4) % 2 ^ 32 ∧
      c + wordsBytes wds = A + f.length ∧
      leVal ((bs.drop 4).take 4) % 2 ^ 32 <
        leVal ((bs.drop 8).take 4) % 2 ^ 32 ∧
      rd64 s'.mem (A + arenaFileBytes f +
        64 * (leVal ((bs.drop 4).take 4) % 2 ^ 32)) = some 0 ∧
      run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧
      s'.reg 0 = some 0 ∧ s'.reg 1 = some 0 ∧
      s'.reg 3 = some A ∧ s'.reg 4 = some (sp + 48) ∧
      s'.reg 6 = some (A + f.length) ∧
      s'.reg 7 = some (A + arenaFileBytes f +
        64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) ∧
      s'.reg 11 =
        some (A + (arenaFileBytes f + arenaTableBytes f)) ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 13 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      s'.reg 14 = some (A + arenaFileBytes f) ∧
      s'.reg 15 = some (A + f.length) ∧
      WordsLay s'.mem c wds ∧
      TabLay s'.mem A
        (A + (arenaFileBytes f + arenaTableBytes f))
        (A + arenaFileBytes f) c wds ∧
      (∀ wd ∈ wds, WordBound wd) ∧
      s'.mem.rdN A f.length = some f ∧
      s'.k.oracle = [] ∧ RegsOk s' ∧ StackRoom s' 6 ∧
      wshape s'.mem = wshape ([c1, c2, buf] ++
        [⟨A + arenaFileBytes f,
          List.replicate
            (X86Env.pages
              (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
          true⟩,
         ⟨A + (arenaFileBytes f + arenaTableBytes f),
          List.replicate (X86Env.pages f.length) 0, true⟩]) ∧
      s'.regs.length = 16) := by
  generalize hTdef : A + arenaFileBytes f = T
  generalize hSdef :
    A + (arenaFileBytes f + arenaTableBytes f) = Sc
  have hLeM : f.length ≤ stage1Mf f := by
    have h1 := pages_ge (65536 + 2 * f.length)
    unfold stage1Mf arenaFileBytes at *
    omega
  have hAe : A + f.length < W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  rcases parser_loaded_to_scratch_classify hrip hr3 hr13 hr4 hr7
      hr9 horacle hmem hbufb hbufhi hrdN hwin hfreeInit hAmod
      hApos hAbound hAframe hdisjc hc2rw hc2lo hc2hi2 hlen16
      hsp80 hspW hregs hroom10 with hexit | hscr
  · exact Or.inl hexit
  obtain ⟨bs, ds, c, k0, q, h20, hL16, hmagic, hdslen, hseq, hcL, hguard,
      hrun0, hqrip, hq14, hq11, hq12, hnw0, hq3, hq13, hq15, hq4,
      hqcur, hqfile, hqor, hwshq, hqroom6, hqlen⟩ := hscr
  obtain ⟨capB, hbufhi0, hbufcap⟩ := hbufhi
  have hcge : A + 16 ≤ c := DataSeq.le hseq
  have hL36 : 36 ≤ f.length := by omega
  have hbs : bs = f.take 20 := by
    have hs := rdN_slice hrdN (o := 0) (k := 20) (by omega)
    simp only [Nat.add_zero, List.drop_zero] at hs
    rw [h20] at hs
    exact Option.some.inj hs
  generalize hmn : leVal ((bs.drop 4).take 4) % 2 ^ 32 = mn
  generalize hnwdef : leVal ((bs.drop 8).take 4) % 2 ^ 32 = nw
  rw [hmn] at hq12
  rw [hnwdef] at hguard hnw0 hwshq
  rw [hTdef] at hq14
  rw [hSdef] at hq11
  rw [hTdef, hSdef] at hwshq
  have hmn32 : mn < 2 ^ 32 := by
    rw [← hmn]
    exact Nat.mod_lt _ (by decide)
  have hnw32 : nw < 2 ^ 32 := by
    rw [← hnwdef]
    exact Nat.mod_lt _ (by decide)
  have hcW : c < W := by omega
  have hcur : rd64 q.mem (sp - 8) = some c := by
    rwa [Nat.mod_eq_of_lt hcW] at hqcur
  -- the nwords reload value and its mask
  obtain ⟨v, hnv, hvmod⟩ : ∃ v, rd64 q.mem (A + 8) = some v ∧
      v % 2 ^ 32 = nw := by
    have hs := rdN_slice hqfile (o := 8) (k := 8) (by omega)
    have hlen8 : ((f.drop 8).take 8).length = 8 :=
      rdN_length hs
    have htake : ((f.drop 8).take 8).take 4 =
        (bs.drop 8).take 4 := by
      have e1 : ((f.drop 8).take 8).take 4 =
          (f.drop 8).take 4 := by
        simp [List.take_take]
      have e2 : (bs.drop 8).take 4 = (f.drop 8).take 4 := by
        simp [hbs, List.drop_take, List.take_take]
      rw [e1, e2]
    refine ⟨leVal ((f.drop 8).take 8), by
      show (q.mem.rdN (A + 8) 8).map leVal = _
      rw [hs]
      rfl, ?_⟩
    have hmod := leVal_first4_mod hlen8
    rw [htake, hnwdef] at hmod
    exact hmod
  -- saved initial-frame slots are readable in the stack region
  have hrdstk : ∀ a, c2.base ≤ a → a < c2.hi →
      (q.mem.rd a).isSome := by
    intro a h1 h2
    exact rd_isSome_of_wshape_append_left hwshq
      ⟨1, c2, find_c2_of_three hdisjc h1 h2, hc2rw⟩
  have hpop : ∀ j, j < 7 →
      ∃ w, rd64 q.mem (sp - 8 + 8 * j) = some w := by
    intro j hj
    apply rd64_some_of_forall_rd
    intro t ht
    apply hrdstk <;> omega
  obtain ⟨w1, hp1⟩ := hpop 1 (by decide)
  obtain ⟨w2, hp2⟩ := hpop 2 (by decide)
  obtain ⟨w3, hp3⟩ := hpop 3 (by decide)
  obtain ⟨w4, hp4⟩ := hpop 4 (by decide)
  obtain ⟨w5, hp5⟩ := hpop 5 (by decide)
  obtain ⟨w6, hp6⟩ := hpop 6 (by decide)
  -- setup, then copy the count into the loop register
  obtain ⟨a7, hrun7, ha7rip, ha7r6, ha7r7, ha7r13, ha7r4, -,
      ha7mem, ha7fr, ha7len, ha7k⟩ :=
    parser_setup_ok hqrip hq4 hq14 hq3 (by omega) (by omega) hcur
      hnv hp1 hp2 hp3 hp4 hp5 hp6 hqlen
  have ha7r13' : a7.reg 13 = some nw := by
    rwa [hvmod] at ha7r13
  have ha7r : ∀ i v, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      q.reg i = some v → a7.reg i = some v := by
    intro i v h0 h4 h6 h7 h13 h
    rw [ha7fr i h0 h4 h6 h7 h13]
    exact h
  obtain ⟨a8, hrun8, ha8rip, ha8r1, ha8fr, ha8mem, ha8len,
      ha8k⟩ :=
    parser_words_init ha7rip ha7r13' ha7len
  have ha8r : ∀ i v, i ≠ 1 → a7.reg i = some v →
      a8.reg i = some v := by
    intro i v hi h
    rw [ha8fr i hi]
    exact h
  have ha8memq : a8.mem = q.mem := by rw [ha8mem, ha7mem]
  have ha8r4 : a8.reg 4 = some (sp + 48) := by
    have h := ha8r 4 _ (by decide) ha7r4
    rwa [show sp - 8 + 56 = sp + 48 by omega] at h
  have ha8r3 : a8.reg 3 = some A :=
    ha8r 3 _ (by decide)
      (ha7r 3 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) hq3)
  have ha8r6 : a8.reg 6 = some c :=
    ha8r 6 _ (by decide) ha7r6
  have ha8r7 : a8.reg 7 = some T :=
    ha8r 7 _ (by decide) ha7r7
  have ha8r11 : a8.reg 11 = some Sc :=
    ha8r 11 _ (by decide)
      (ha7r 11 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) hq11)
  have ha8r12 : a8.reg 12 = some mn :=
    ha8r 12 _ (by decide)
      (ha7r 12 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) hq12)
  have ha8r13 : a8.reg 13 = some nw :=
    ha8r 13 _ (by decide) ha7r13'
  have ha8r14 : a8.reg 14 = some T :=
    ha8r 14 _ (by decide)
      (ha7r 14 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) hq14)
  have ha8r15 : a8.reg 15 = some (A + f.length) :=
    ha8r 15 _ (by decide)
      (ha7r 15 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) hq15)
  have hrunHead : run stage1Im (k0 + 11 + 1) (Out.ok s) =
      Out.ok a8 := by
    rw [show k0 + 11 + 1 = k0 + (11 + 1) by omega,
      run_add stage1Im k0 _, hrun0, run_add stage1Im 11 1, hrun7,
      hrun8]
  have hregs8 : RegsOk a8 := run_regs_ok hrunHead hregs
  have hsh8 : wshape a8.mem = wshape ([c1, c2, buf] ++
      [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
       ⟨Sc, List.replicate (X86Env.pages f.length) 0, true⟩]) := by
    rw [ha8memq]
    exact hwshq
  have hroom8 : StackRoom a8 2 :=
    stackRoom_of_wshape_c2 hsh8 ha8r4 hc2rw hdisjc (by omega)
      hstackTop (by omega)
  have ha8file : a8.mem.rdN A f.length = some f := by
    rw [ha8memq]
    exact hqfile
  -- arena geometry around the table and scratch bases
  have hMf : stage1Mf f = arenaFileBytes f + arenaTableBytes f +
      arenaScratchBytes f := rfl
  have hpgnw : X86Env.pages (64 * nw) ≤ arenaTableBytes f := by
    unfold arenaTableBytes
    apply pages_mono
    omega
  have h64nw : 64 * nw ≤ arenaTableBytes f :=
    Nat.le_trans (pages_ge _) hpgnw
  have hlenpg : f.length ≤ X86Env.pages f.length := pages_ge _
  have hlenfile : f.length ≤ arenaFileBytes f := by
    unfold arenaFileBytes
    have h := pages_ge (65536 + 2 * f.length)
    omega
  have hTEnd : T + arenaTableBytes f = Sc := by omega
  have hScEnd : Sc + X86Env.pages f.length = A + stage1Mf f := by
    unfold stage1Mf arenaScratchBytes
    omega
  have hTge : A + f.length ≤ T := by omega
  have hTabEnd : T + 64 * nw ≤ Sc := by omega
  have hfreec2 : A + stage1Mf f ≤ c2.base ∨ c2.hi ≤ A :=
    hfreeInit c2 (by simp)
  have hbufhiT : buf.hi ≤ T := by
    rw [hbufhi0, ← hTdef]
    omega
  have hnonmem3 : ∀ a, T ≤ a → a < A + stage1Mf f →
      ∀ r ∈ ([c1, c2, buf] : Mem),
        ¬ (r.base ≤ a ∧ a < r.hi) := by
    intro a ha1 ha2 r hr
    have hr2 : c1 = r ∨ c2 = r ∨ buf = r := by
      rcases (show r = c1 ∨ r = c2 ∨ r = buf by simpa using hr)
          with h | h | h
      · exact Or.inl h.symm
      · exact Or.inr (Or.inl h.symm)
      · exact Or.inr (Or.inr h.symm)
    rcases hr2 with rfl | rfl | rfl
    · rcases hfreeInit c1 (by simp) with h | h
      · exact not_mem_region_of_lt (by omega)
      · exact not_mem_region_of_le (by omega)
    · rcases hfreeInit c2 (by simp) with h | h
      · exact not_mem_region_of_lt (by omega)
      · exact not_mem_region_of_le (by omega)
    · exact not_mem_region_of_le (by omega)
  have hwsh8mid : wshape a8.mem =
      wshape (([c1, c2, buf] ++
        [⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
          true⟩]) ++
        [⟨Sc, List.replicate (X86Env.pages f.length) 0,
          true⟩]) := by
    rw [hsh8]
    congr 1
  have htabhi : (⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
        true⟩ : Region).hi = T + X86Env.pages (64 * nw) := by
    simp [X86Env.Region.hi]
  have hscrhi : (⟨Sc, List.replicate (X86Env.pages f.length) 0,
        true⟩ : Region).hi = Sc + X86Env.pages f.length := by
    simp [X86Env.Region.hi]
  have hwrTab : ∀ (off v : Nat), off + 8 ≤ 64 * nw →
      (wr64 a8.mem (T + off) v).isSome := by
    intro off v hoff
    apply wr64_isSome_of_forall_wr
    intro j hj u
    have hmem : (⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
          true⟩ : Region).base ≤ T + off + j ∧
        T + off + j <
          (⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
            true⟩ : Region).hi := by
      refine ⟨by show T ≤ T + off + j; omega, ?_⟩
      rw [htabhi]
      have h := pages_ge (64 * nw)
      omega
    exact wr_isSome_of_wshape_append_left hwsh8mid
      ⟨0 + ([c1, c2, buf] : Mem).length,
        ⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
        find_append_last [c1, c2, buf] 0
          (hnonmem3 _ (by omega) (by omega)) hmem, rfl⟩
  have hwrScr : ∀ a u, Sc + (c - A) ≤ a → a < Sc + f.length →
      (a8.mem.wr a u).isSome := by
    intro a u ha1 ha2
    apply wr_isSome_of_wshape_append_last hwsh8mid _ _ rfl
    · intro r hr
      rcases List.mem_append.mp hr with hr | hr
      · exact hnonmem3 a (by omega) (by omega) r hr
      · simp only [List.mem_singleton] at hr
        rw [hr]
        apply not_mem_region_of_le
        rw [htabhi]
        omega
    · refine ⟨by show Sc ≤ a; omega, ?_⟩
      rw [hscrhi]
      omega
  have ha8rdN : ∀ a l, c ≤ a → l ≤ A + f.length - a →
      ∃ bs, a8.mem.rdN a l = some bs ∧ ∀ b ∈ bs, b < 256 := by
    intro a l ha hl
    by_cases hl0 : l = 0
    · subst hl0
      exact ⟨[], rfl, fun b hb => (List.not_mem_nil hb).elim⟩
    have hs := rdN_slice ha8file (o := a - A) (k := l) (by omega)
    refine ⟨(f.drop (a - A)).take l, ?_, ?_⟩
    · rwa [show A + (a - A) = a by omega] at hs
    · intro b hb
      exact hfB b
        (List.mem_of_mem_drop (List.mem_of_mem_take hb))
  have hdisjTB : T + 64 * nw ≤ c ∨ A + f.length ≤ T :=
    Or.inr (by omega)
  have hdisjTStk : T + 64 * nw ≤ sp + 48 - 16 ∨ sp + 48 ≤ T := by
    rcases hfreec2 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdisjTScr : Sc + (A + f.length - A) ≤ T ∨
      T + 64 * nw ≤ Sc + (c - A) := Or.inr (by omega)
  have hdisjBStk : A + f.length ≤ sp + 48 - 16 ∨
      sp + 48 ≤ c := by
    rcases hfreec2 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdisjScrB : Sc + (A + f.length - A) ≤ c ∨
      A + f.length ≤ Sc + (c - A) := Or.inr (by omega)
  have hdisjScrStk : Sc + (A + f.length - A) ≤ sp + 48 - 16 ∨
      sp + 48 ≤ Sc + (c - A) := by
    rcases hfreec2 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  -- the whole word loop, classified
  rcases parser_words_loop_classify nw ha8rip ha8r1 ha8r6 ha8r7
      ha8r15 ha8r4 ha8r3 ha8r11 (by omega) hcL hAe (by
        have hW : W = 2 ^ 64 := rfl
        omega)
      (by
        have hW : W = 2 ^ 64 := rfl
        omega) (by
        have hW : W = 2 ^ 64 := rfl
        omega) hroom8 (by omega) (by omega) ha8len
      hdisjTB hdisjTStk hdisjTScr hdisjBStk hdisjScrB hdisjScrStk
      hwrTab (by
        intro a u ha1 ha2
        exact hwrScr a u ha1 (by omega)) ha8rdN with
    hloop | hloopfail
  rotate_left
  · -- a word refused: fall into exit 1
    obtain ⟨kF, sF, hrunF, hsFrip, hsFr4, -, hsFlen, hsFk,
        hsFsh⟩ := hloopfail
    have hrunToF : run stage1Im (k0 + 11 + 1 + kF) (Out.ok s) =
        Out.ok sF := by
      rw [show k0 + 11 + 1 + kF = (k0 + 11 + 1) + kF by omega,
        run_add stage1Im _ kF, hrunHead, hrunF]
    have hshF : wshape sF.mem = wshape ([c1, c2, buf] ++
        [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
         ⟨Sc, List.replicate (X86Env.pages f.length) 0,
           true⟩]) := by
      rw [hsFsh]
      exact hsh8
    have hregsF : RegsOk sF := run_regs_ok hrunToF hregs
    rcases hsFr4 with hr4F | hr4F
    · have hroomF : StackRoom sF 1 :=
        stackRoom_of_wshape_c2 hshF hr4F hc2rw hdisjc (by omega)
          hstackTop (by omega)
      obtain ⟨ke, hex⟩ := exit1_of_fail1 stage1_refuseCode
        (by omega) (by omega) sF hsFrip hr4F hregsF hroomF
      exact Or.inl ⟨k0 + 11 + 1 + kF + ke, by
        rw [run_add stage1Im _ ke, hrunToF, hex]⟩
    · have hr4F' : sF.reg 4 = some (sp + 40) := by
        rwa [show sp + 48 - 8 = sp + 40 by omega] at hr4F
      have hroomF : StackRoom sF 1 :=
        stackRoom_of_wshape_c2 hshF hr4F' hc2rw hdisjc (by omega)
          (by omega) (by omega)
      obtain ⟨ke, hex⟩ := exit1_of_fail1 stage1_refuseCode
        (by omega) (by omega) sF hsFrip hr4F' hregsF hroomF
      exact Or.inl ⟨k0 + 11 + 1 + kF + ke, by
        rw [run_add stage1Im _ ke, hrunToF, hex]⟩
  obtain ⟨wds, k9, a9, hwdslen, hrun9, ha9rip, ha9r1, ha9r6,
      ha9r7, ha9r4, ha9fit, ha9lay, ha9tab, ha9annb, ha9fr,
      ha9len, ha9k, ha9sh, ha9room, ha9memfr⟩ := hloop
  have ha9r : ∀ i v, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 →
      i ≠ 7 → i ≠ 10 → a8.reg i = some v → a9.reg i = some v := by
    intro i v h0 h1 h2 h4 h6 h7 h10 h
    rw [ha9fr i h0 h1 h2 h4 h6 h7 h10]
    exact h
  have hrunLoop : run stage1Im (k0 + 11 + 1 + k9) (Out.ok s) =
      Out.ok a9 := by
    rw [show k0 + 11 + 1 + k9 = (k0 + 11 + 1) + k9 by omega,
      run_add stage1Im _ k9, hrunHead, hrun9]
  -- loop head with no words left
  obtain ⟨a10, hrun10, ha10rip, ha10fr, ha10mem, ha10len,
      ha10k⟩ :=
    parser_words_head_zero ha9rip ha9r1
  have ha10len16 : a10.regs.length = 16 := by
    rw [ha10len]; exact ha9len
  have ha10r : ∀ i v, a9.reg i = some v → a10.reg i = some v := by
    intro i v h
    rw [ha10fr i]
    exact h
  have ha10r6 : a10.reg 6 = some (c + wordsBytes wds) :=
    ha10r 6 _ ha9r6
  have ha10r15 : a10.reg 15 = some (A + f.length) :=
    ha10r 15 _
      (ha9r 15 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r15)
  have ha10r12 : a10.reg 12 = some mn :=
    ha10r 12 _
      (ha9r 12 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r12)
  have ha10r13 : a10.reg 13 = some nw :=
    ha10r 13 _
      (ha9r 13 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r13)
  have ha10r14 : a10.reg 14 = some T :=
    ha10r 14 _
      (ha9r 14 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r14)
  have ha10r4 : a10.reg 4 = some (sp + 48) := ha10r 4 _ ha9r4
  have ha10r1 : a10.reg 1 = some 0 := ha10r 1 _ ha9r1
  have ha10r3 : a10.reg 3 = some A :=
    ha10r 3 _
      (ha9r 3 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r3)
  have ha10r7 : a10.reg 7 = some (T + 64 * nw) :=
    ha10r 7 _ ha9r7
  have ha10r11 : a10.reg 11 = some Sc :=
    ha10r 11 _
      (ha9r 11 _ (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) ha8r11)
  have hrunTrail : run stage1Im (k0 + 11 + 1 + k9 + 3) (Out.ok s) =
      Out.ok a10 := by
    rw [show k0 + 11 + 1 + k9 + 3 = (k0 + 11 + 1 + k9) + 3 by
      omega, run_add stage1Im _ 3, hrunLoop, hrun10]
  have hsh10 : wshape a10.mem = wshape ([c1, c2, buf] ++
      [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
       ⟨Sc, List.replicate (X86Env.pages f.length) 0,
         true⟩]) := by
    rw [ha10mem, ha9sh]
    exact hsh8
  -- a generic exit-1 fallthrough for the two direct trailing
  -- refusals, whose states keep the loop stack pointer
  have htrailExit : ∀ {z : X86Int.St} {kz : Nat},
      run stage1Im kz (Out.ok a10) = Out.ok z →
      z.rip = aFail1 → z.reg 4 = some (sp + 48) →
      z.mem = a10.mem →
      ∃ k, run stage1Im k (Out.ok s) = Out.exit 1 := by
    intro z kz hzrun hzrip hzr4 hzmem
    have hrunZ : run stage1Im (k0 + 11 + 1 + k9 + 3 + kz)
        (Out.ok s) = Out.ok z := by
      rw [run_add stage1Im _ kz, hrunTrail, hzrun]
    have hshz : wshape z.mem = wshape ([c1, c2, buf] ++
        [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
         ⟨Sc, List.replicate (X86Env.pages f.length) 0,
           true⟩]) := by
      rw [hzmem]
      exact hsh10
    have hroomz : StackRoom z 1 :=
      stackRoom_of_wshape_c2 hshz hzr4 hc2rw hdisjc (by omega)
        hstackTop (by omega)
    obtain ⟨ke, hex⟩ := exit1_of_fail1 stage1_refuseCode
      (by omega) (by omega) z hzrip hzr4
      (run_regs_ok hrunZ hregs) hroomz
    exact ⟨k0 + 11 + 1 + k9 + 3 + kz + ke, by
      rw [run_add stage1Im _ ke, hrunZ, hex]⟩
  by_cases hcurN : c + wordsBytes wds ≠ A + f.length
  · obtain ⟨z, hzrun, hzrip, hzfr, hzmem, -, hzk⟩ :=
      parser_trailing_cursor_fail ha10rip ha10r6 ha10r15 hcurN
        (by omega) hAe
    exact Or.inl (htrailExit hzrun hzrip (by rw [hzfr 4]; exact
      ha10r4) hzmem)
  have hcurE : c + wordsBytes wds = A + f.length := by omega
  by_cases hmnN : ¬ mn < nw
  · obtain ⟨z, hzrun, hzrip, hzfr, hzmem, -, hzk⟩ :=
      parser_trailing_main_ge ha10rip ha10r6 ha10r15 ha10r12
        ha10r13 hcurE (by omega) hAe (by omega) ha10len16
    exact Or.inl (htrailExit hzrun hzrip (by rw [hzfr 4]; exact
      ha10r4) hzmem)
  have hmnlt : mn < nw := by omega
  have hslot10 : rd64 a10.mem (T + 64 * mn) =
      some wds[mn].ins := by
    rw [ha10mem]
    have hj : mn < wds.length := by omega
    exact TabLay_ins_at ha9tab mn hj
  -- now the slot is readable, so the classified trailing applies
  rcases parser_trailing_classify ha10rip ha10r6 ha10r15 ha10r12
      ha10r13 ha10r14 (by rw [hcurE]; exact hAe) hAe (by
        have hW : W = 2 ^ 64 := rfl
        omega)
      ⟨wds[mn].ins, hslot10⟩ ha10len16 with htok | htfail
  rotate_left
  · obtain ⟨kz, z, hzrun, hzrip, hzfr, hzmem, -, -⟩ := htfail
    exact Or.inl (htrailExit hzrun hzrip
      (by rw [hzfr 4 (by decide) (by decide)]; exact ha10r4)
      hzmem)
  obtain ⟨a11, hrun11, ha11rip, ha11r0, -, -, ha11slot, ha11fr,
      ha11mem, ha11len, ha11k⟩ := htok
  have ha11r : ∀ i v, i ≠ 0 → i ≠ 2 → a10.reg i = some v →
      a11.reg i = some v := by
    intro i v h0 h2 h
    rw [ha11fr i h0 h2]
    exact h
  have hrunTot : run stage1Im (k0 + 11 + 1 + k9 + 3 + 15)
      (Out.ok s) = Out.ok a11 := by
    rw [run_add stage1Im _ 15, hrunTrail, hrun11]
  have hlay11 : WordsLay a11.mem c wds := by
    apply WordsLay_of_frame wds ha9lay
    intro x k hx hk
    rw [ha11mem, ha10mem]
    exact ha9memfr x k (by omega) (by omega) (by omega)
  have hfile11 : a11.mem.rdN A f.length = some f := by
    rw [ha11mem, ha10mem, ha9memfr A f.length ?_ ?_ ?_]
    · exact ha8file
    · rcases hfreec2 with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    · exact Or.inr (by omega)
    · exact Or.inr (by omega)
  have hsh11 : wshape a11.mem = wshape ([c1, c2, buf] ++
      [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩,
       ⟨Sc, List.replicate (X86Env.pages f.length) 0,
         true⟩]) := by
    rw [ha11mem]
    exact hsh10
  have ha11r4 : a11.reg 4 = some (sp + 48) :=
    ha11r 4 _ (by decide) (by decide) ha10r4
  have hroom11 : StackRoom a11 6 :=
    stackRoom_of_wshape_c2 hsh11 ha11r4 hc2rw hdisjc (by omega)
      hstackTop (by omega)
  have hslot11 : rd64 a11.mem (T + 64 * mn) = some 0 := by
    rwa [ha11mem]
  have htab11 : TabLay a11.mem A Sc T c wds := by
    rw [ha11mem, ha10mem]
    exact ha9tab
  refine Or.inr ⟨bs, ds, wds, c, k0 + 11 + 1 + k9 + 3 + 15, a11,
    h20, hbs, hL16, hmagic, hdslen, hseq, hcL, by
      rw [hnwdef]; exact hguard, by
      rw [hnwdef]; exact hwdslen, hcurE, by
      rw [hmn, hnwdef]; exact hmnlt, by
      rw [hmn]; exact hslot11, hrunTot, ha11rip, ha11r0, ?_, ?_,
    ha11r4, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hlay11, htab11, ha9annb,
    hfile11, ?_, run_regs_ok hrunTot hregs, hroom11, by
      rw [hnwdef]; exact hsh11, ha11len⟩
  · exact ha11r 1 _ (by decide) (by decide) ha10r1
  · exact ha11r 3 _ (by decide) (by decide) ha10r3
  · have h := ha11r 6 _ (by decide) (by decide) ha10r6
    rwa [hcurE] at h
  · rw [hnwdef]
    exact ha11r 7 _ (by decide) (by decide) ha10r7
  · exact ha11r 11 _ (by decide) (by decide) ha10r11
  · rw [hmn]
    exact ha11r 12 _ (by decide) (by decide) ha10r12
  · rw [hnwdef]
    exact ha11r 13 _ (by decide) (by decide) ha10r13
  · exact ha11r 14 _ (by decide) (by decide) ha10r14
  · exact ha11r 15 _ (by decide) (by decide) ha10r15
  · rw [ha11k, ha10k, ha9k, ha8k, ha7k]
    exact hqor


set_option maxHeartbeats 0 in
/-- The oracle-existence prefix: from process entry, under the
witness oracle for the `Room` interval, the run either exits 1
in the loader/parser, or reaches the checker entry with the
oracle fully consumed.  Together with
`oracleLegalAt_of_prefix`, every state before that milestone is
oracle-legal; `stage1Oracle_not_hitsNomem` discharges the
memory-refusal side for the whole run. -/
theorem stage1_oracle_prefix {f path argv0 stdin : List Nat}
    {top : Nat} (hinit : Stage1InitOk f path argv0 top)
    (hroom : Room f path argv0 top) :
    ∃ A, A % page = 0 ∧ 0 < A ∧ A + stage1Mf f ≤ 2 ^ 47 ∧
      (stage1Init f path argv0 [] [] top).mem.free A
          (stage1Mf f) = true ∧
      ((∃ k, run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin
            (stage1Oracle f A) top)) = Out.exit 1) ∨
       (∃ k s', run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin
            (stage1Oracle f A) top)) = Out.ok s' ∧
        s'.rip = base + 0x452 ∧ s'.k.oracle = [] ∧
        RegsOk s' ∧ StackRoom s' 6 ∧
        (∃ sp0, s'.reg 4 = some sp0 ∧ sp0 ≤ 2 ^ 48 ∧
          ∀ q ∈ X86Env.shape s'.mem, q.2 ≤ 2 ^ 48))) := by
  have hinit0 := hinit
  obtain ⟨hfB, -, -, -, -, -, -, htop47, hleInit, hstkhi⟩ := hinit0
  obtain ⟨A, k0, sL, hAmod, hApos, hAbound, hfree, hrun0, hLrip,
      hLr3, -, hLr13, -, hLr4, -, -, -, hLr9, hLoracle, -,
      hLmem, hLfile, hLwin, hLlen, -, hLroom, -⟩ :=
    loader_init_to_loaded hinit hroom
  obtain ⟨c1, c2, buf, hmL, hc1b, hc1hi, hc2b, hc2hi, hbufb,
      hbufhi, hbufrw, hc2rw⟩ := hLmem
  have hregsI : RegsOk
      (stage1Init f path argv0 stdin (stage1Oracle f A) top) := by
    intro i hi16 _
    by_cases h4 : i = 4
    · subst h4
      exact ⟨_, stage1Init_reg4 f path argv0 stdin
        (stage1Oracle f A) top⟩
    · exact ⟨0, stage1Init_reg_ne4 f path argv0 stdin
        (stage1Oracle f A) top hi16 h4⟩
  have hregsL : RegsOk sL := run_regs_ok hrun0 hregsI
  -- stack geometry of the initial frame
  have hspEq : (X86Env.initStack top [argv0, path]).2 + 24 =
      c2.base + 4120 := by
    have h1 := initStack_sp top argv0 path
    rw [← hc2b] at h1
    have hsr : X86Env.stackRoom = 4096 := rfl
    omega
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
  have hc2hi' : c2.hi = top := by rw [hc2hi]; exact hstkhi
  have hspHi : (X86Env.initStack top [argv0, path]).2 + 24 + 48 ≤
      c2.hi := by
    have hh : c2.hi = c2.base +
        (X86Env.initStack top [argv0, path]).1.data.length := by
      have e : (X86Env.initStack top [argv0, path]).1.hi =
          (X86Env.initStack top [argv0, path]).1.base +
            (X86Env.initStack top [argv0, path]).1.data.length :=
        rfl
      rw [hc2hi, e, ← hc2b]
    omega
  -- the free interval against the two initial regions
  have hfree0 := hfree
  rw [stage1Init_mem] at hfree0
  unfold X86Env.Mem.free at hfree0
  rw [List.all_eq_true] at hfree0
  have hfreecode := hfree0 ⟨base, stage1Code, false⟩ (by simp)
  have hfreestk := hfree0
    (X86Env.initStack top [argv0, path]).1 (by simp)
  rw [Bool.or_eq_true] at hfreecode hfreestk
  simp only [decide_eq_true_eq] at hfreecode hfreestk
  have hfreecode2 : A + stage1Mf f ≤ base ∨
      base + stage1Code.length ≤ A := by
    have hcodehi : (⟨base, stage1Code, false⟩ : Region).hi =
        base + stage1Code.length := by simp only [X86Env.Region.hi]
    rwa [hcodehi] at hfreecode
  have hfreeInit : ∀ r ∈ ([c1, c2] : Mem),
      A + stage1Mf f ≤ r.base ∨ r.hi ≤ A := by
    intro r hr
    have hr2 : r = c1 ∨ r = c2 := by simpa using hr
    rcases hr2 with rfl | rfl
    · rcases hfreecode2 with h | h
      · exact Or.inl (by rw [hc1b]; exact h)
      · exact Or.inr (by rw [hc1hi]; exact h)
    · rcases hfreestk with h | h
      · exact Or.inl (by rw [hc2b]; exact h)
      · exact Or.inr (by rw [hc2hi]; exact h)
  have hAframe : ∀ x k, A ≤ x → x + k ≤ A + f.length →
      x + k ≤ (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
        (X86Env.initStack top [argv0, path]).2 + 24 ≤ x := by
    intro x k hx hxk
    have hLe : f.length ≤ stage1Mf f := by
      have h1 : f.length ≤ arenaFileBytes f := by
        unfold arenaFileBytes
        have h2 := pages_ge (65536 + 2 * f.length)
        omega
      unfold stage1Mf
      omega
    rcases hfreestk with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdisjc : c1.hi ≤ c2.base := by
    have h := hleInit
    rw [phVaddr_eq, phMemsz_eq] at h
    rw [hc1hi, hc2b]
    exact h
  have hspW' : (X86Env.initStack top [argv0, path]).2 + 24 < W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  have hsp48W : (X86Env.initStack top [argv0, path]).2 + 24 + 48 <
      W := by
    have hW : W = 2 ^ 64 := rfl
    omega
  -- the classified parser from the loaded state
  rcases parser_loaded_to_checker_classify hLrip hLr3 hLr13 hLr4
      (hregsL 7 (by decide) (by decide)) hLr9 hLoracle hmL hbufb
      hbufhi hLfile hLwin hfB hfreeInit hAmod hApos hAbound
      hAframe hdisjc hc2rw (by omega) (by omega) (by omega) hLlen
      (by omega) hspW' hsp48W hregsL hLroom with hexit | hchk
  · exact ⟨A, hAmod, hApos, hAbound, hfree, Or.inl (by
      obtain ⟨k, hk⟩ := hexit
      exact ⟨k0 + k, by rw [run_add, hrun0, hk]⟩)⟩
  · obtain ⟨bs, ds, wds, c, k, s', -, -, -, -, -, hseq, -, -, hwdslen, hfit, -, -, hrun, hrip', -, -, -, hr4', -, -, -, -, -, -, -, -, -, -, -, hor, hregs', hroom', hwsh, -⟩ := hchk
    have hpgnw : X86Env.pages
        (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) ≤
        arenaTableBytes f := by
      rw [← hwdslen]
      unfold arenaTableBytes
      apply pages_mono
      have h1 := wordsBytes_ge wds
      have h2 : A + 16 ≤ c := DataSeq.le hseq
      omega
    have hshp : ∀ q ∈ X86Env.shape s'.mem, q.2 ≤ 2 ^ 48 := by
      have hwb : ∀ q ∈ wshape s'.mem, q.2.1 ≤ 2 ^ 48 := by
        rw [hwsh]
        intro q hq
        simp only [wshape, List.map_append, List.map_cons, List.map_nil,
          List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hq
        obtain ⟨cap, hcap, hcaple⟩ := hbufhi
        rcases hq with (h | h | h) | h | h
        · rw [h]
          show c1.hi ≤ 2 ^ 48
          rw [hc1hi, stage1Code_length]
          have hb : base = 4194304 := rfl
          rw [hb]
          decide
        · rw [h]
          show c2.hi ≤ 2 ^ 48
          rw [hc2hi']
          omega
        · rw [h]
          show buf.hi ≤ 2 ^ 48
          rw [hcap]
          have hM : arenaFileBytes f ≤ stage1Mf f := by
            unfold stage1Mf; omega
          omega
        · rw [h]
          simp only [X86Env.Region.hi, List.length_replicate]
          have hM : arenaFileBytes f + arenaTableBytes f ≤
              stage1Mf f := by
            unfold stage1Mf; omega
          omega
        · rw [h]
          simp only [X86Env.Region.hi, List.length_replicate]
          have hM : A + (arenaFileBytes f + arenaTableBytes f) +
              X86Env.pages f.length = A + stage1Mf f := by
            unfold stage1Mf arenaScratchBytes; omega
          omega
      intro q hq
      have hbridge : X86Env.shape s'.mem =
          (wshape s'.mem).map (fun q => (q.1, q.2.1)) := by
        simp [X86Env.shape, wshape, List.map_map]
      rw [hbridge] at hq
      obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
      exact hwb q' hq'
    exact ⟨A, hAmod, hApos, hAbound, hfree, Or.inr ⟨k0 + k, s', by
      rw [run_add, hrun0, hrun], hrip', hor, hregs', hroom',
      _, hr4', by omega, hshp⟩⟩

/-! ### Checker reachability: decode tables and abstract closure

Ported from spr1 (`SporosCheckBinCore`, commit `b887f1b51e`): the
checker region `base + 0x452 ... base + 2333` and the `ANNL`
subroutine are linear decode walks, kernel-checked (`walkOK`), and
carry an abstract state (stack slots above the frame, `reg 0` value)
whose successor closure is also kernel-checked (`annOK`).  Calls to
`L32` (`base + 622`) and `ANNL` (`base + 662`) are summarised in the
closure; jumps to `aFail1` leave the table.  The step lemmas below
turn this into: no state reachable from the checker entry is at a
memory syscall. -/


/-- Walk the code once: each entry decodes at its address from the
bytes that remain there (ported from spr1 `SporosCheckBinCore`). -/
def walkOK : List Nat → Nat → List (Nat × Ins × Nat) → Bool
  | _, _, [] => true
  | rest, a, (a', i, l) :: t =>
    a == a' && X86Dec.decode (fun j => rest[j]?) a == some (i, l) &&
      walkOK (rest.drop l) (a + l) t

theorem walkOK_sound (code : List Nat) :
    ∀ (tbl : List (Nat × Ins × Nat)) (rest : List Nat) (a : Nat),
    walkOK rest a tbl = true → base ≤ a → rest = code.drop (a - base) →
    ∀ e ∈ tbl, X86Dec.imageOf code base e.1 = some e.2
  | [], _, _, _, _, _, e, he => by simp at he
  | (a', i, l) :: t, rest, a, h, ha, hr, e, he => by
    simp only [walkOK, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, hd⟩, ht⟩ := h
    rcases List.mem_cons.1 he with rfl | he
    · show X86Dec.imageOf code base _ = some (i, l)
      unfold X86Dec.imageOf
      rw [if_neg (by omega), ← hd, hr]
      simp only [List.getElem?_drop]
    · refine walkOK_sound code t _ _ ht (by omega) ?_ e he
      rw [hr, List.drop_drop]
      exact congrArg (fun k => code.drop k) (by omega)

def projT (t : List (Nat × Ins × Nat × List (Option Nat) × Option Nat)) :
    List (Nat × Ins × Nat) :=
  t.map fun e => (e.1, e.2.1, e.2.2.1)

def chkAnn : List (Nat × Ins × Nat × List (Option Nat) × Option Nat) :=
[  (4195410, .mov 7 14, 3, [], none),
  (4195413, .mov 5 13, 3, [], none),
  (4195416, .test 5 5, 3, [], none),
  (4195419, .jcc 5 4195426, 2, [], none),
  (4195421, .jmp 4196629, 5, [], none),
  (4195426, .ld 0 7 24, 4, [], none),
  (4195430, .test 0 0, 3, [], none),
  (4195433, .jcc 5 4195440, 2, [], none),
  (4195435, .jmp 4194596, 5, [], none),
  (4195440, .ld 6 7 48, 4, [], none),
  (4195444, .ld 1 7 32, 4, [], none),
  (4195448, .test 1 1, 3, [], none),
  (4195451, .jcc 5 4195458, 2, [], none),
  (4195453, .jmp 4195578, 5, [], none),
  (4195458, .push 1, 1, [], none),
  (4195459, .push 6, 1, [none], none),
  (4195460, .mov 2 6, 3, [none, none], none),
  (4195463, .call 4194926, 5, [none, none], none),
  (4195468, .mov 12 0, 3, [none, none], none),
  (4195471, .ld 2 7 24, 4, [none, none], none),
  (4195475, .cmp 0 2, 3, [none, none], none),
  (4195478, .jcc 2 4195485, 2, [none, none], none),
  (4195480, .jmp 4194596, 5, [none, none], none),
  (4195485, .call 4194966, 5, [none, none], none),
  (4195490, .push 0, 1, [none, none], none),
  (4195491, .mov 2 0, 3, [none, none, none], none),
  (4195494, .aluI .add 2 4, 4, [none, none, none], none),
  (4195498, .call 4194926, 5, [none, none, none], none),
  (4195503, .mov 8 0, 3, [none, none, none], none),
  (4195506, .pop 2, 1, [none, none, none], none),
  (4195507, .aluI .add 2 8, 4, [none, none], none),
  (4195511, .call 4194926, 5, [none, none], none),
  (4195516, .mov 9 0, 3, [none, none], none),
  (4195519, .pop 6, 1, [none, none], none),
  (4195520, .mov 2 6, 3, [none], none),
  (4195523, .aluI .add 2 4, 4, [none], none),
  (4195527, .call 4194926, 5, [none], none),
  (4195532, .cmp 0 8, 3, [none], none),
  (4195535, .jcc 4 4195542, 2, [none], none),
  (4195537, .jmp 4194596, 5, [none], none),
  (4195542, .mov 2 6, 3, [none], none),
  (4195545, .aluI .add 2 8, 4, [none], none),
  (4195549, .call 4194926, 5, [none], none),
  (4195554, .cmp 0 9, 3, [none], none),
  (4195557, .jcc 4 4195564, 2, [none], none),
  (4195559, .jmp 4194596, 5, [none], none),
  (4195564, .aluI .add 6 12, 4, [none], none),
  (4195568, .pop 1, 1, [none], none),
  (4195569, .aluI .sub 1 1, 4, [], none),
  (4195573, .jmp 4195448, 5, [], none),
  (4195578, .ld 6 7 40, 4, [], none),
  (4195582, .alu .xor 1 1, 2, [], none),
  (4195584, .ld 8 7 0, 3, [], none),
  (4195587, .alu .xor 9 9, 3, [], none),
  (4195590, .alu .xor 12 12, 3, [], none),
  (4195593, .call 4194966, 5, [], none),
  (4195598, .test 0 0, 3, [], none),
  (4195601, .jcc 4 4195646, 2, [], none),
  (4195603, .push 0, 1, [], none),
  (4195604, .mov 2 0, 3, [none], none),
  (4195607, .aluI .add 2 4, 4, [none], none),
  (4195611, .call 4194926, 5, [none], none),
  (4195616, .cmp 0 8, 3, [none], none),
  (4195619, .jcc 4 4195626, 2, [none], none),
  (4195621, .jmp 4194596, 5, [none], none),
  (4195626, .pop 2, 1, [none], none),
  (4195627, .aluI .add 2 8, 4, [], none),
  (4195631, .call 4194926, 5, [], none),
  (4195636, .cmp 0 9, 3, [], none),
  (4195639, .jcc 4 4195646, 2, [], none),
  (4195641, .jmp 4194596, 5, [], none),
  (4195646, .ld 0 7 24, 4, [], none),
  (4195650, .cmp 1 0, 3, [], none),
  (4195653, .jcc 2 4195660, 2, [], none),
  (4195655, .jmp 4196616, 5, [], none),
  (4195660, .ld8 0 6 0, 3, [], none),
  (4195663, .cmpI 0 57, 4, [], none),
  (4195667, .jcc 3 4195674, 2, [], none),
  (4195669, .jmp 4196021, 5, [], none),
  (4195674, .cmpI 0 128, 7, [], none),
  (4195681, .jcc 5 4195688, 2, [], none),
  (4195683, .jmp 4196089, 5, [], none),
  (4195688, .cmpI 0 129, 7, [], none),
  (4195695, .jcc 5 4195702, 2, [], none),
  (4195697, .jmp 4196102, 5, [], none),
  (4195702, .cmpI 0 130, 7, [], none),
  (4195709, .jcc 5 4195716, 2, [], none),
  (4195711, .jmp 4196115, 5, [], none),
  (4195716, .cmpI 0 131, 7, [], none),
  (4195723, .jcc 5 4195730, 2, [], none),
  (4195725, .jmp 4196150, 5, [], none),
  (4195730, .cmpI 0 132, 7, [], none),
  (4195737, .jcc 5 4195744, 2, [], none),
  (4195739, .jmp 4196178, 5, [], none),
  (4195744, .cmpI 0 133, 7, [], none),
  (4195751, .jcc 5 4195758, 2, [], none),
  (4195753, .jmp 4196206, 5, [], none),
  (4195758, .cmpI 0 134, 7, [], none),
  (4195765, .jcc 5 4195772, 2, [], none),
  (4195767, .jmp 4196230, 5, [], none),
  (4195772, .cmpI 0 135, 7, [], none),
  (4195779, .jcc 5 4195786, 2, [], none),
  (4195781, .jmp 4196269, 5, [], none),
  (4195786, .cmpI 0 136, 7, [], none),
  (4195793, .jcc 5 4195800, 2, [], none),
  (4195795, .jmp 4196319, 5, [], none),
  (4195800, .cmpI 0 137, 7, [], none),
  (4195807, .jcc 5 4195814, 2, [], none),
  (4195809, .jmp 4196401, 5, [], none),
  (4195814, .cmpI 0 138, 7, [], none),
  (4195821, .jcc 5 4195828, 2, [], none),
  (4195823, .jmp 4196401, 5, [], none),
  (4195828, .cmpI 0 139, 7, [], none),
  (4195835, .jcc 5 4195842, 2, [], none),
  (4195837, .jmp 4196505, 5, [], none),
  (4195842, .cmpI 0 140, 7, [], none),
  (4195849, .jcc 5 4195856, 2, [], none),
  (4195851, .jmp 4196583, 5, [], none),
  (4195856, .jmp 4194596, 5, [], none),
  (4195861, .aluI .add 1 1, 4, [], none),
  (4195865, .ld 0 7 24, 4, [], none),
  (4195869, .cmp 1 0, 3, [], none),
  (4195872, .jcc 2 4195879, 2, [], none),
  (4195874, .jmp 4194596, 5, [], none),
  (4195879, .mov 12 1, 3, [], none),
  (4195882, .call 4194966, 5, [], none),
  (4195887, .test 0 0, 3, [], none),
  (4195890, .jcc 5 4195897, 2, [], none),
  (4195892, .jmp 4195646, 5, [], none),
  (4195897, .push 0, 1, [], none),
  (4195898, .mov 2 0, 3, [none], none),
  (4195901, .aluI .add 2 4, 4, [none], none),
  (4195905, .call 4194926, 5, [none], none),
  (4195910, .cmp 0 8, 3, [none], none),
  (4195913, .jcc 4 4195920, 2, [none], none),
  (4195915, .jmp 4194596, 5, [none], none),
  (4195920, .pop 2, 1, [none], none),
  (4195921, .aluI .add 2 8, 4, [], none),
  (4195925, .call 4194926, 5, [], none),
  (4195930, .cmp 0 9, 3, [], none),
  (4195933, .jcc 4 4195940, 2, [], none),
  (4195935, .jmp 4194596, 5, [], none),
  (4195940, .jmp 4195646, 5, [], none),
  (4195945, .aluI .add 1 1, 4, [], none),
  (4195949, .ld 0 7 24, 4, [], none),
  (4195953, .cmp 1 0, 3, [], none),
  (4195956, .jcc 2 4195963, 2, [], none),
  (4195958, .jmp 4195646, 5, [], none),
  (4195963, .mov 12 1, 3, [], none),
  (4195966, .call 4194966, 5, [], none),
  (4195971, .test 0 0, 3, [], none),
  (4195974, .jcc 5 4195987, 2, [], none),
  (4195976, .alu .xor 8 8, 3, [], none),
  (4195979, .alu .xor 9 9, 3, [], none),
  (4195982, .jmp 4195646, 5, [], none),
  (4195987, .push 0, 1, [], none),
  (4195988, .mov 2 0, 3, [none], none),
  (4195991, .aluI .add 2 4, 4, [none], none),
  (4195995, .call 4194926, 5, [none], none),
  (4196000, .mov 8 0, 3, [none], none),
  (4196003, .pop 2, 1, [none], none),
  (4196004, .aluI .add 2 8, 4, [], none),
  (4196008, .call 4194926, 5, [], none),
  (4196013, .mov 9 0, 3, [], none),
  (4196016, .jmp 4195646, 5, [], none),
  (4196021, .movI 2 4196637, 10, [], none),
  (4196031, .alu .add 2 0, 3, [], none),
  (4196034, .alu .add 2 0, 3, [], none),
  (4196037, .ld8 10 2 0, 4, [], none),
  (4196041, .cmp 8 10, 3, [], none),
  (4196044, .jcc 3 4196051, 2, [], none),
  (4196046, .jmp 4194596, 5, [], none),
  (4196051, .cmpI 0 50, 4, [], none),
  (4196055, .jcc 5 4196066, 2, [], none),
  (4196057, .aluI .add 6 1, 4, [], none),
  (4196061, .jmp 4195945, 5, [], none),
  (4196066, .aluI .add 2 1, 4, [], none),
  (4196070, .ld8 11 2 0, 4, [], none),
  (4196074, .alu .sub 8 10, 3, [], none),
  (4196077, .alu .add 8 11, 3, [], none),
  (4196080, .aluI .add 6 1, 4, [], none),
  (4196084, .jmp 4195861, 5, [], none),
  (4196089, .aluI .add 8 1, 4, [], none),
  (4196093, .aluI .add 6 9, 4, [], none),
  (4196097, .jmp 4195861, 5, [], none),
  (4196102, .aluI .add 8 2, 4, [], none),
  (4196106, .aluI .add 6 5, 4, [], none),
  (4196110, .jmp 4195861, 5, [], none),
  (4196115, .mov 2 6, 3, [], none),
  (4196118, .aluI .add 2 1, 4, [], none),
  (4196122, .call 4194926, 5, [], none),
  (4196127, .cmp 0 8, 3, [], none),
  (4196130, .jcc 2 4196137, 2, [], none),
  (4196132, .jmp 4194596, 5, [], none),
  (4196137, .aluI .add 8 1, 4, [], none),
  (4196141, .aluI .add 6 5, 4, [], none),
  (4196145, .jmp 4195861, 5, [], none),
  (4196150, .cmpI 8 1, 4, [], none),
  (4196154, .jcc 3 4196161, 2, [], none),
  (4196156, .jmp 4194596, 5, [], none),
  (4196161, .aluI .sub 8 1, 4, [], none),
  (4196165, .aluI .add 9 1, 4, [], none),
  (4196169, .aluI .add 6 1, 4, [], none),
  (4196173, .jmp 4195861, 5, [], none),
  (4196178, .cmpI 9 1, 4, [], none),
  (4196182, .jcc 3 4196189, 2, [], none),
  (4196184, .jmp 4194596, 5, [], none),
  (4196189, .aluI .add 8 1, 4, [], none),
  (4196193, .aluI .sub 9 1, 4, [], none),
  (4196197, .aluI .add 6 1, 4, [], none),
  (4196201, .jmp 4195861, 5, [], none),
  (4196206, .cmpI 9 1, 4, [], none),
  (4196210, .jcc 3 4196217, 2, [], none),
  (4196212, .jmp 4194596, 5, [], none),
  (4196217, .aluI .add 8 1, 4, [], none),
  (4196221, .aluI .add 6 1, 4, [], none),
  (4196225, .jmp 4195861, 5, [], none),
  (4196230, .mov 2 6, 3, [], none),
  (4196233, .aluI .add 2 1, 4, [], none),
  (4196237, .call 4194926, 5, [], none),
  (4196242, .ld 10 7 16, 4, [], none),
  (4196246, .cmp 0 10, 3, [], none),
  (4196249, .jcc 2 4196256, 2, [], none),
  (4196251, .jmp 4194596, 5, [], none),
  (4196256, .aluI .add 8 1, 4, [], none),
  (4196260, .aluI .add 6 5, 4, [], none),
  (4196264, .jmp 4195861, 5, [], none),
  (4196269, .mov 2 6, 3, [], none),
  (4196272, .aluI .add 2 1, 4, [], none),
  (4196276, .call 4194926, 5, [], none),
  (4196281, .cmpI 8 1, 4, [], none),
  (4196285, .jcc 3 4196292, 2, [], none),
  (4196287, .jmp 4194596, 5, [], none),
  (4196292, .ld 10 7 16, 4, [], none),
  (4196296, .cmp 0 10, 3, [], none),
  (4196299, .jcc 2 4196306, 2, [], none),
  (4196301, .jmp 4194596, 5, [], none),
  (4196306, .aluI .sub 8 1, 4, [], none),
  (4196310, .aluI .add 6 5, 4, [], none),
  (4196314, .jmp 4195861, 5, [], none),
  (4196319, .mov 2 6, 3, [], none),
  (4196322, .aluI .add 2 1, 4, [], none),
  (4196326, .call 4194926, 5, [], none),
  (4196331, .mov 12 0, 3, [], none),
  (4196334, .call 4194966, 5, [], none),
  (4196339, .test 0 0, 3, [], none),
  (4196342, .jcc 5 4196349, 2, [], none),
  (4196344, .jmp 4194596, 5, [], none),
  (4196349, .push 0, 1, [], none),
  (4196350, .mov 2 0, 3, [none], none),
  (4196353, .aluI .add 2 4, 4, [none], none),
  (4196357, .call 4194926, 5, [none], none),
  (4196362, .cmp 0 8, 3, [none], none),
  (4196365, .jcc 4 4196372, 2, [none], none),
  (4196367, .jmp 4194596, 5, [none], none),
  (4196372, .pop 2, 1, [none], none),
  (4196373, .aluI .add 2 8, 4, [], none),
  (4196377, .call 4194926, 5, [], none),
  (4196382, .cmp 0 9, 3, [], none),
  (4196385, .jcc 4 4196392, 2, [], none),
  (4196387, .jmp 4194596, 5, [], none),
  (4196392, .aluI .add 6 5, 4, [], none),
  (4196396, .jmp 4195945, 5, [], none),
  (4196401, .mov 2 6, 3, [], none),
  (4196404, .aluI .add 2 1, 4, [], none),
  (4196408, .call 4194926, 5, [], none),
  (4196413, .cmpI 8 1, 4, [], none),
  (4196417, .jcc 3 4196424, 2, [], none),
  (4196419, .jmp 4194596, 5, [], none),
  (4196424, .mov 12 0, 3, [], none),
  (4196427, .call 4194966, 5, [], none),
  (4196432, .test 0 0, 3, [], none),
  (4196435, .jcc 5 4196442, 2, [], none),
  (4196437, .jmp 4194596, 5, [], none),
  (4196442, .push 0, 1, [], none),
  (4196443, .mov 2 0, 3, [none], none),
  (4196446, .aluI .add 2 4, 4, [none], none),
  (4196450, .call 4194926, 5, [none], none),
  (4196455, .mov 10 8, 3, [none], none),
  (4196458, .aluI .sub 10 1, 4, [none], none),
  (4196462, .cmp 0 10, 3, [none], none),
  (4196465, .jcc 4 4196472, 2, [none], none),
  (4196467, .jmp 4194596, 5, [none], none),
  (4196472, .pop 2, 1, [none], none),
  (4196473, .aluI .add 2 8, 4, [], none),
  (4196477, .call 4194926, 5, [], none),
  (4196482, .cmp 0 9, 3, [], none),
  (4196485, .jcc 4 4196492, 2, [], none),
  (4196487, .jmp 4194596, 5, [], none),
  (4196492, .aluI .sub 8 1, 4, [], none),
  (4196496, .aluI .add 6 5, 4, [], none),
  (4196500, .jmp 4195861, 5, [], none),
  (4196505, .mov 2 6, 3, [], none),
  (4196508, .aluI .add 2 1, 4, [], none),
  (4196512, .call 4194926, 5, [], none),
  (4196517, .cmp 0 13, 3, [], none),
  (4196520, .jcc 2 4196527, 2, [], none),
  (4196522, .jmp 4194596, 5, [], none),
  (4196527, .mov 2 0, 3, [], none),
  (4196530, .alu .add 2 2, 3, [], none),
  (4196533, .alu .add 2 2, 3, [], none),
  (4196536, .alu .add 2 2, 3, [], none),
  (4196539, .alu .add 2 2, 3, [], none),
  (4196542, .alu .add 2 2, 3, [], none),
  (4196545, .alu .add 2 2, 3, [], none),
  (4196548, .alu .add 2 14, 3, [], none),
  (4196551, .ld 10 2 0, 3, [], none),
  (4196554, .cmp 8 10, 3, [], none),
  (4196557, .jcc 3 4196564, 2, [], none),
  (4196559, .jmp 4194596, 5, [], none),
  (4196564, .ld 11 2 8, 4, [], none),
  (4196568, .alu .sub 8 10, 3, [], none),
  (4196571, .alu .add 8 11, 3, [], none),
  (4196574, .aluI .add 6 5, 4, [], none),
  (4196578, .jmp 4195861, 5, [], none),
  (4196583, .ld 0 7 8, 4, [], none),
  (4196587, .cmp 8 0, 3, [], none),
  (4196590, .jcc 4 4196597, 2, [], none),
  (4196592, .jmp 4194596, 5, [], none),
  (4196597, .test 9 9, 3, [], none),
  (4196600, .jcc 4 4196607, 2, [], none),
  (4196602, .jmp 4194596, 5, [], none),
  (4196607, .aluI .add 6 1, 4, [], none),
  (4196611, .jmp 4195945, 5, [], none),
  (4196616, .aluI .add 7 64, 4, [], none),
  (4196620, .aluI .sub 5 1, 4, [], none),
  (4196624, .jmp 4195416, 5, [], none),
  (4196629, .pushI 60, 2, [], none),
  (4196631, .pop 0, 1, [some 60], none),
  (4196632, .pushI 0, 2, [], some 60),
  (4196634, .pop 7, 1, [some 0], some 60),
  (4196635, .syscall, 2, [], some 60)]

def annlAnn : List (Nat × Ins × Nat × List (Option Nat) × Option Nat) :=
[  (4194966, .push 1, 1, [], none),
  (4194967, .ld 0 7 48, 4, [none], none),
  (4194971, .ld 1 7 32, 4, [none], none),
  (4194975, .test 1 1, 3, [none], none),
  (4194978, .jcc 4 4195008, 2, [none], none),
  (4194980, .mov 2 0, 3, [none], none),
  (4194983, .push 0, 1, [none], none),
  (4194984, .call 4194926, 5, [none, none], none),
  (4194989, .pop 2, 1, [none, none], none),
  (4194990, .cmp 0 12, 3, [none], none),
  (4194993, .jcc 4 4195012, 2, [none], none),
  (4194995, .mov 0 2, 3, [none], none),
  (4194998, .aluI .add 0 12, 4, [none], none),
  (4195002, .aluI .sub 1 1, 4, [none], none),
  (4195006, .jmp 4194975, 2, [none], none),
  (4195008, .alu .xor 0 0, 2, [none], none),
  (4195010, .pop 1, 1, [none], some 0),
  (4195011, .ret, 1, [], some 0),
  (4195012, .mov 0 2, 3, [none], none),
  (4195015, .pop 1, 1, [none], none),
  (4195016, .ret, 1, [], none)]

set_option maxRecDepth 100000 in
theorem walk_chk_ok : walkOK (stage1Code.drop 0x452) (base + 0x452)
    (projT chkAnn) = true := by decide +kernel

set_option maxRecDepth 100000 in
theorem walk_annl_ok : walkOK (stage1Code.drop 662) (base + 662)
    (projT annlAnn) = true := by decide +kernel

theorem chk_codeAt : ∀ e ∈ projT chkAnn, stage1Im e.1 = some e.2 :=
  walkOK_sound stage1Code _ _ _ walk_chk_ok (by decide)
    (by rw [show base + 0x452 - base = 0x452 by unfold base; omega])

theorem annl_codeAt : ∀ e ∈ projT annlAnn, stage1Im e.1 = some e.2 :=
  walkOK_sound stage1Code _ _ _ walk_annl_ok (by decide)
    (by rw [show base + 662 - base = 662 by unfold base; omega])


/-- The register an instruction writes, if any (fall-through cases). -/
def destReg : Ins → Option Nat
  | .movI d _ | .mov d _ | .mov32 d _ | .alu _ d _ | .aluI _ d _
  | .shlI d _ | .shrI d _ | .ld d _ _ | .ld32 d _ _ | .ld8 d _ _
  | .lea d _ _ => some d
  | _ => none

/-- The abstract value of `reg 0` after a fall-through instruction. -/
def r0After : Ins → Option Nat → Option Nat
  | .movI d imm, r0 => if d == 0 then some (imm % W) else r0
  | .mov d r, r0 => if d == 0 then (if r == 0 then r0 else none) else r0
  | .mov32 d r, r0 =>
    if d == 0 then (if r == 0 then r0.map (· % 2 ^ 32) else none) else r0
  | .alu op d r, r0 =>
    if d == 0 then
      if r == 0 then
        match r0 with
        | some v => some (aluF op v v).1
        | none => if op = .xor then some 0 else none
      else none
    else r0
  | .aluI op d imm, r0 =>
    if d == 0 then r0.map (fun v => (aluF op v (imm % W)).1) else r0
  | .shlI d c, r0 => if d == 0 then r0.map (fun v => v * 2 ^ (c % 64) % W) else r0
  | .shrI d c, r0 => if d == 0 then r0.map (fun v => v / 2 ^ (c % 64)) else r0
  | .ld d _ _, r0 => if d == 0 then none else r0
  | .ld32 d _ _, r0 => if d == 0 then none else r0
  | .ld8 d _ _, r0 => if d == 0 then none else r0
  | .lea d _ _, r0 => if d == 0 then none else r0
  | _, r0 => r0

/-- Abstract successors of one entry. `mode = true` is the `ANNL`
subroutine (a `ret` on the empty abstract stack leaves the table;
calls go only to `L32`). Calls to the two subroutines are summarised:
control returns to the fall-through with the same abstract stack and
unknown `reg 0`. A jump to `aFail1` leaves the table. -/
def succs (mode : Bool) (a : Nat) (ins : Ins) (l : Nat)
    (σ : List (Option Nat)) (r0 : Option Nat) :
    Option (List (Nat × List (Option Nat) × Option Nat)) :=
  match ins with
  | .syscall => if r0 == some 60 then some [] else none
  | .jmp t => some [(t, σ, r0)]
  | .jcc _ t => some [(t, σ, r0), (a + l, σ, r0)]
  | .call t =>
    if (mode && t == base + 622) ||
        (!mode && (t == base + 622 || t == base + 662)) then
      some [(a + l, σ, none)]
    else none
  | .ret => if mode && σ.isEmpty then some [] else none
  | .pushI imm => some [(a + l, some (imm % W) :: σ, r0)]
  | .push r => some [(a + l, (if r == 0 then r0 else none) :: σ, r0)]
  | .pop d =>
    if d == 4 then none
    else match σ with
      | v :: σ' => some [(a + l, σ', if d == 0 then v else r0)]
      | [] => none
  | .st _ _ _ | .st8 _ _ _ | .jtab _ _ _ | .jmpR _ => none
  | i =>
    match destReg i with
    | some 4 => none
    | _ => some [(a + l, σ, r0After i r0)]

/-- Every abstract successor is an entry of the table with exactly
that abstract state, or the exit-1 block. -/
def annOK (mode : Bool)
    (t : List (Nat × Ins × Nat × List (Option Nat) × Option Nat)) : Bool :=
  t.all fun e =>
    match succs mode e.1 e.2.1 e.2.2.1 e.2.2.2.1 e.2.2.2.2 with
    | none => false
    | some ss => ss.all fun s => s.1 == aFail1 ||
        t.any fun e' => e'.1 == s.1 && e'.2.2.2.1 == s.2.1 &&
          e'.2.2.2.2 == s.2.2

set_option maxRecDepth 100000 in
theorem chk_ann_ok : annOK false chkAnn = true := by decide +kernel

theorem annl_ann_ok : annOK true annlAnn = true := by decide +kernel

/-! ### Invariants over the abstract tables -/

/-- The known stack slots at `sp`: slot `i` (at `sp + 8 * i`) holds
`v` whenever the abstract stack says `some v` there. -/
def stackAt (s : X86Int.St) (sp : Nat) (σ : List (Option Nat)) : Prop :=
  ∀ i v, σ[i]? = some (some v) → rd64 s.mem (sp + 8 * i) = some v

/-- Every region ends at or below `2 ^ 48`; this keeps the
`(sp + W - 8) % W` arithmetic of pushes and calls honest. -/
def shapeBnd (m : Mem) : Prop := ∀ p ∈ shape m, p.2 ≤ 2 ^ 48

theorem find_cond : ∀ (m : Mem) (a i j : Nat) (r : Region),
    Mem.find m a i = some (j, r) → r.base ≤ a ∧ a < r.hi ∧ r ∈ m
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | r :: rest, a, i, j, r', h => by
    simp only [Mem.find] at h
    split at h
    · obtain ⟨h1, h2⟩ := ‹r.base ≤ a ∧ a < r.hi›
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact ⟨h1, h2, List.mem_cons_self⟩
    · obtain ⟨h1, h2, h3⟩ := find_cond rest a (i + 1) j r' h
      exact ⟨h1, h2, List.mem_cons_of_mem _ h3⟩

theorem rd_lt_of_shapeBnd {m : Mem} {a b : Nat}
    (h : m.rd a = some b) (hb : shapeBnd m) : a < 2 ^ 48 := by
  simp only [Mem.rd, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨j, r⟩ := p
    obtain ⟨-, hlt, hmem⟩ := find_cond m a 0 j r hp
    have hsh : (r.base, r.hi) ∈ shape m :=
      List.mem_map.2 ⟨r, hmem, rfl⟩
    have := hb _ hsh
    omega

theorem wr_lt_of_shapeBnd {m m' : Mem} {a v : Nat}
    (h : m.wr a v = some m') (hb : shapeBnd m) : a < 2 ^ 48 := by
  simp only [Mem.wr, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨j, r⟩ := p
    obtain ⟨-, hlt, hmem⟩ := find_cond m a 0 j r hp
    have hsh : (r.base, r.hi) ∈ shape m :=
      List.mem_map.2 ⟨r, hmem, rfl⟩
    have := hb _ hsh
    omega

theorem wr64_lt_of_shapeBnd {m m' : Mem} {a v : Nat}
    (h : wr64 m a v = some m') (hb : shapeBnd m) : a < 2 ^ 48 := by
  unfold wr64 at h
  obtain ⟨b, bs, hbs⟩ : ∃ b bs, bytes8 v = b :: bs := by
    have hlen : (bytes8 v).length = 8 := by simp [bytes8]
    cases h : bytes8 v with
    | nil => rw [h] at hlen; simp at hlen
    | cons b bs => exact ⟨b, bs, rfl⟩
  rw [hbs] at h
  simp only [Mem.wrN, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i m1 hm1
    exact wr_lt_of_shapeBnd hm1 hb

theorem shapeBnd_wr64 {m m' : Mem} {a v : Nat}
    (h : wr64 m a v = some m') (hb : shapeBnd m) : shapeBnd m' := by
  have hs : shape m' = shape m := wrN_shape m a (bytes8 v) h
  intro p hp
  rw [hs] at hp
  exact hb p hp

theorem stackAt_wr64_below {s : X86Int.St} {m' : Mem} {a v sp : Nat}
    {σ : List (Option Nat)} (hm : wr64 s.mem a v = some m')
    (hstk : stackAt s sp σ) (hd : a + 8 ≤ sp) :
    stackAt { s with mem := m' } sp σ := by
  intro i w hi
  have h1 : rd64 s.mem (sp + 8 * i) = some w := hstk i w hi
  have h2 : rd64 m' (sp + 8 * i) = rd64 s.mem (sp + 8 * i) :=
    rd64_wr64_other hm (Or.inl (by omega))
  show rd64 m' (sp + 8 * i) = some w
  rw [h2, h1]

theorem stackAt_tail {s : X86Int.St} {sp : Nat} {v : Option Nat}
    {σ : List (Option Nat)} (h : stackAt s sp (v :: σ)) :
    stackAt s (sp + 8) σ := by
  intro i w hi
  have h1 : rd64 s.mem (sp + 8 * (i + 1)) = some w := by
    apply h
    rw [List.getElem?_cons_succ]
    exact hi
  have : sp + 8 + 8 * i = sp + 8 * (i + 1) := by omega
  rw [this]
  exact h1

theorem stackAt_head {s : X86Int.St} {sp v : Nat}
    {σ : List (Option Nat)} (h : stackAt s sp (some v :: σ)) :
    rd64 s.mem sp = some v := by
  have h1 := h 0 v rfl
  simpa using h1

/-- Successor information from the kernel-checked closure. -/
theorem annOK_succs {mode : Bool}
    {t : List (Nat × Ins × Nat × List (Option Nat) × Option Nat)}
    {e : Nat × Ins × Nat × List (Option Nat) × Option Nat}
    (hok : annOK mode t = true) (he : e ∈ t) :
    ∃ ss, succs mode e.1 e.2.1 e.2.2.1 e.2.2.2.1 e.2.2.2.2 = some ss ∧
      ∀ s ∈ ss, s.1 = aFail1 ∨
        ∃ e' ∈ t, e'.1 = s.1 ∧ e'.2.2.2.1 = s.2.1 ∧
          e'.2.2.2.2 = s.2.2 := by
  have h1 : (match succs mode e.1 e.2.1 e.2.2.1 e.2.2.2.1 e.2.2.2.2 with
      | none => false
      | some ss => ss.all fun s => s.1 == aFail1 ||
          t.any fun e' => e'.1 == s.1 && e'.2.2.2.1 == s.2.1 &&
            e'.2.2.2.2 == s.2.2) = true := by
    have := List.all_eq_true.1 hok e he
    exact this
  cases hs : succs mode e.1 e.2.1 e.2.2.1 e.2.2.2.1 e.2.2.2.2 with
  | none => rw [hs] at h1; simp at h1
  | some ss =>
    rw [hs] at h1
    refine ⟨ss, rfl, ?_⟩
    intro s hs_mem
    have h2 := List.all_eq_true.1 h1 s hs_mem
    rw [Bool.or_eq_true] at h2
    rcases h2 with h2 | h2
    · exact Or.inl (by simpa using h2)
    · rw [List.any_eq_true] at h2
      obtain ⟨e', he', h3⟩ := h2
      simp only [Bool.and_eq_true, beq_iff_eq] at h3
      exact Or.inr ⟨e', he', h3.1.1, h3.1.2, h3.2⟩

/-- The concrete decode of an annotated entry. -/
theorem chk_entry_decode {e : Nat × Ins × Nat × List (Option Nat) ×
    Option Nat} (he : e ∈ chkAnn) :
    stage1Im e.1 = some (e.2.1, e.2.2.1) := by
  have hmem : (e.1, e.2.1, e.2.2.1) ∈ projT chkAnn :=
    List.mem_map.2 ⟨e, he, rfl⟩
  exact chk_codeAt _ hmem

theorem annl_entry_decode {e : Nat × Ins × Nat × List (Option Nat) ×
    Option Nat} (he : e ∈ annlAnn) :
    stage1Im e.1 = some (e.2.1, e.2.2.1) := by
  have hmem : (e.1, e.2.1, e.2.2.1) ∈ projT annlAnn :=
    List.mem_map.2 ⟨e, he, rfl⟩
  exact annl_codeAt _ hmem

set_option maxRecDepth 100000 in
theorem chkAnn_addr_bnd : ∀ e ∈ chkAnn, e.1 + e.2.2.1 ≤ base + 2447 := by
  decide +kernel

set_option maxRecDepth 100000 in
theorem annlAnn_addr_bnd : ∀ e ∈ annlAnn, e.1 + e.2.2.1 ≤ base + 2447 := by
  decide +kernel
/-- The fall-through instructions: `exec` goes to `rip + len`,
writes at most one register, and never touches memory. -/
def isFall : Ins → Bool
  | .movI _ _ | .mov _ _ | .mov32 _ _ | .alu _ _ _ | .aluI _ _ _
  | .cmp _ _ | .cmpI _ _ | .test _ _ | .shlI _ _ | .shrI _ _
  | .ld _ _ _ | .ld32 _ _ _ | .ld8 _ _ _ | .lea _ _ _ => true
  | _ => false

theorem aluF_fst_lt {op : Alu} {a b : Nat} (ha : a < W) (hb : b < W) :
    (aluF op a b).1 < W := by
  have hW : W = 2 ^ 64 := rfl
  have hWpos : 0 < W := by omega
  cases op <;> simp only [aluF]
  · exact Nat.mod_lt _ hWpos
  · exact Nat.mod_lt _ hWpos
  · have h : Nat.land a b ≤ a := Nat.and_le_left; omega
  · rw [hW] at ha hb ⊢; exact Nat.or_lt_two_pow ha hb
  · rw [hW] at ha hb ⊢; exact Nat.xor_lt_two_pow ha hb

set_option maxHeartbeats 0 in
/-- Soundness of `r0After` for the fall-through instructions, with
the frame facts the abstract step needs. -/
theorem fall_sound {s s' : X86Int.St} {ins : Ins} {l : Nat}
    {r0 : Option Nat}
    (hfall : isFall ins = true) (hd4 : destReg ins ≠ some 4)
    (h0 : ∀ v, r0 = some v → s.reg 0 = some v)
    (h0W : ∀ v, r0 = some v → v < W)
    (hlen : 16 ≤ s.regs.length)
    (hexec : exec ins l s = .ok s') :
    s'.rip = s.rip + l ∧ s'.mem = s.mem ∧ s'.reg 4 = s.reg 4 ∧
    (∀ v, r0After ins r0 = some v → s'.reg 0 = some v) ∧
    (∀ v, r0After ins r0 = some v → v < W) ∧
    s'.regs.length = s.regs.length := by
  have hWpos : 0 < W := by have hW : W = 2 ^ 64 := rfl; omega
  have h00 : (0 : Nat) < s.regs.length := by omega
  have key : ∀ (d : Nat) (w : Nat) (t : X86Int.St),
      t = s.set d (some w) → d ≠ 4 →
      ({ t with rip := s.rip + l }).reg 4 = s.reg 4 := by
    intro d w t ht hd
    rw [reg_with_rip, ht, reg_set_other (show (4:Nat) ≠ d from fun h => hd h.symm)]
  cases ins with
  | movI d imm =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    simp only [exec] at hexec
    cases hexec
    refine ⟨rfl, rfl, key d (imm % W) _ rfl hd, ?_, ?_, ?_⟩
    · intro v hv
      simp only [r0After] at hv
      rw [reg_with_rip]
      by_cases hd0 : d = 0
      · subst hd0
        simp only [beq_self_eq_true, if_true] at hv
        rw [reg_set_self h00]; exact hv
      · simp only [beq_iff_eq, hd0, if_false] at hv
        rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
    · intro v hv
      simp only [r0After] at hv
      by_cases hd0 : d = 0
      · subst hd0
        simp only [beq_self_eq_true, if_true] at hv
        rw [Option.some.injEq] at hv
        rw [← hv]; exact Nat.mod_lt _ hWpos
      · simp only [beq_iff_eq, hd0, if_false] at hv
        exact h0W v hv
    · show (s.set d _).regs.length = s.regs.length
      simp [St.set]
  | mov d r =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hr : s.reg r with
    | none => simp [exec, hr, lift] at hexec
    | some w =>
      simp only [exec, hr, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, key d w _ rfl hd, ?_, ?_, ?_⟩
      · intro v hv
        simp only [r0After] at hv
        rw [reg_with_rip]
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          rw [reg_set_self h00]
          by_cases hr0 : r = 0
          · subst hr0
            simp only [beq_self_eq_true, if_true] at hv
            have := h0 v hv
            rw [hr] at this; exact this
          · simp only [beq_iff_eq, hr0, if_false] at hv
            simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          by_cases hr0 : r = 0
          · subst hr0
            simp only [beq_self_eq_true, if_true] at hv
            exact h0W v hv
          · simp only [beq_iff_eq, hr0, if_false] at hv
            simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | mov32 d r =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hr : s.reg r with
    | none => simp [exec, hr, lift] at hexec
    | some w =>
      simp only [exec, hr, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, key d (w % 2^32) _ rfl hd, ?_, ?_, ?_⟩
      · intro v hv
        simp only [r0After] at hv
        rw [reg_with_rip]
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          rw [reg_set_self h00]
          by_cases hr0 : r = 0
          · subst hr0
            simp only [beq_self_eq_true, if_true] at hv
            cases hr0v : r0 with
            | none => rw [hr0v] at hv; simp at hv
            | some u =>
              rw [hr0v] at hv
              simp only [Option.map_some, Option.some.injEq] at hv
              have hu := h0 u hr0v
              rw [hr] at hu
              rw [Option.some.injEq] at hu
              rw [hu, Option.some.injEq]; exact hv
          · simp only [beq_iff_eq, hr0, if_false] at hv
            simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          by_cases hr0 : r = 0
          · subst hr0
            simp only [beq_self_eq_true, if_true] at hv
            cases hr0v : r0 with
            | none => rw [hr0v] at hv; simp at hv
            | some u =>
              rw [hr0v] at hv
              simp only [Option.map_some, Option.some.injEq] at hv
              rw [← hv]
              have := h0W u hr0v
              have h1 : u % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by decide)
              have hW32 : 2 ^ 32 ≤ W := by
                have hW : W = 2 ^ 64 := rfl; omega
              omega
          · simp only [beq_iff_eq, hr0, if_false] at hv
            simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | alu op d r =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases ha : s.reg d with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      cases hb : s.reg r with
      | none => simp [exec, ha, hb, lift] at hexec
      | some y =>
        simp only [exec, ha, hb, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
        · show (s.set d _).reg 4 = s.reg 4
          rw [reg_set_other (show (4:Nat) ≠ d from fun h => hd h.symm)]
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · subst hd0
            simp only [beq_self_eq_true, if_true] at hv
            show (s.set 0 _).reg 0 = some v
            rw [reg_set_self h00]
            by_cases hr0 : r = 0
            · subst hr0
              simp only [beq_self_eq_true, if_true] at hv
              have hxy : x = y := Option.some.inj (ha.symm.trans hb)
              subst hxy
              cases hr0v : r0 with
              | none =>
                rw [hr0v] at hv
                dsimp only at hv
                by_cases hop : op = .xor
                · subst hop
                  rw [if_pos rfl] at hv
                  rw [← hv]
                  simp [aluF]
                · rw [if_neg hop] at hv
                  simp at hv
              | some u =>
                rw [hr0v] at hv
                dsimp only at hv
                have hu := h0 u hr0v
                rw [ha] at hu
                rw [Option.some.injEq] at hu
                subst hu
                rw [Option.some.injEq] at hv
                rw [Option.some.injEq]; exact hv
            · simp only [beq_iff_eq, hr0, if_false] at hv
              simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            show (s.set d _).reg 0 = some v
            rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · subst hd0
            simp only [beq_self_eq_true, if_true] at hv
            by_cases hr0 : r = 0
            · subst hr0
              simp only [beq_self_eq_true, if_true] at hv
              cases hr0v : r0 with
              | none =>
                rw [hr0v] at hv
                dsimp only at hv
                by_cases hop : op = .xor
                · subst hop
                  rw [if_pos rfl, Option.some.injEq] at hv; omega
                · rw [if_neg hop] at hv
                  simp at hv
              | some u =>
                rw [hr0v] at hv
                dsimp only at hv
                rw [Option.some.injEq] at hv
                rw [← hv]
                exact aluF_fst_lt (h0W u hr0v) (h0W u hr0v)
            · simp only [beq_iff_eq, hr0, if_false] at hv
              simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            exact h0W v hv
        · show (s.set d _).regs.length = s.regs.length
          simp [St.set]
  | aluI op d imm =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases ha : s.reg d with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      simp only [exec, ha, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
      · rw [reg_with_rip]
        show (s.set d _).reg 4 = s.reg 4
        rw [reg_set_other (show (4:Nat) ≠ d from fun h => hd h.symm)]
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          show (s.set 0 _).reg 0 = some v
          rw [reg_set_self h00]
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            have hu := h0 u hr0v
            have e1 : s.reg 0 = some x := ha
            rw [e1] at hu
            rw [Option.some.injEq] at hu
            subst hu
            rw [Option.some.injEq]; exact hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          show (s.set d _).reg 0 = some v
          rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            rw [← hv]
            exact aluF_fst_lt (h0W u hr0v) (Nat.mod_lt _ hWpos)
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | shlI d c =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases ha : s.reg d with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      simp only [exec, ha, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
      · rw [reg_with_rip]
        show (s.set d _).reg 4 = s.reg 4
        rw [reg_set_other (show (4:Nat) ≠ d from fun h => hd h.symm)]
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          show (s.set 0 _).reg 0 = some v
          rw [reg_set_self h00]
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            have hu := h0 u hr0v
            have e1 : s.reg 0 = some x := ha
            rw [e1] at hu
            rw [Option.some.injEq] at hu
            subst hu
            rw [Option.some.injEq]; exact hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          show (s.set d _).reg 0 = some v
          rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            rw [← hv]
            exact Nat.mod_lt _ hWpos
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | shrI d c =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases ha : s.reg d with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      simp only [exec, ha, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩
      · rw [reg_with_rip]
        show (s.set d _).reg 4 = s.reg 4
        rw [reg_set_other (show (4:Nat) ≠ d from fun h => hd h.symm)]
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          show (s.set 0 _).reg 0 = some v
          rw [reg_set_self h00]
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            have hu := h0 u hr0v
            have e1 : s.reg 0 = some x := ha
            rw [e1] at hu
            rw [Option.some.injEq] at hu
            subst hu
            rw [Option.some.injEq]; exact hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          show (s.set d _).reg 0 = some v
          rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]; exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · subst hd0
          simp only [beq_self_eq_true, if_true] at hv
          cases hr0v : r0 with
          | none => rw [hr0v] at hv; simp at hv
          | some u =>
            rw [hr0v] at hv
            simp only [Option.map_some, Option.some.injEq] at hv
            rw [← hv]
            have hu := h0W u hr0v
            have h := Nat.div_le_self u (2 ^ (c % 64)); omega
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | cmp a b =>
    cases ha : s.reg a with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      cases hb : s.reg b with
      | none => simp [exec, ha, hb, lift] at hexec
      | some y =>
        simp only [exec, ha, hb, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩
        · intro v hv
          simp only [r0After] at hv
          show s.reg 0 = some v
          exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          exact h0W v hv
        · rfl
  | cmpI a imm =>
    cases ha : s.reg a with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      simp only [exec, ha, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩
      · intro v hv
        simp only [r0After] at hv
        show s.reg 0 = some v
        exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        exact h0W v hv
      · rfl
  | test a b =>
    cases ha : s.reg a with
    | none => simp [exec, ha, lift] at hexec
    | some x =>
      cases hb : s.reg b with
      | none => simp [exec, ha, hb, lift] at hexec
      | some y =>
        simp only [exec, ha, hb, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩
        · intro v hv
          simp only [r0After] at hv
          show s.reg 0 = some v
          exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          exact h0W v hv
        · rfl
  | ld d b disp =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hb : s.reg b with
    | none => simp [exec, hb, lift] at hexec
    | some x =>
      cases hv : rd64 s.mem (ea x disp) with
      | none => simp [exec, hb, hv, lift] at hexec
      | some w =>
        simp only [exec, hb, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, key d w _ rfl hd, ?_, ?_, ?_⟩
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            rw [reg_with_rip, reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]
            exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            exact h0W v hv
        · show (s.set d _).regs.length = s.regs.length
          simp [St.set]
  | ld32 d b disp =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hb : s.reg b with
    | none => simp [exec, hb, lift] at hexec
    | some x =>
      cases hv : rd32 s.mem (ea x disp) with
      | none => simp [exec, hb, hv, lift] at hexec
      | some w =>
        simp only [exec, hb, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, key d w _ rfl hd, ?_, ?_, ?_⟩
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            rw [reg_with_rip, reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]
            exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            exact h0W v hv
        · show (s.set d _).regs.length = s.regs.length
          simp [St.set]
  | ld8 d b disp =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hb : s.reg b with
    | none => simp [exec, hb, lift] at hexec
    | some x =>
      cases hv : s.mem.rd (ea x disp) with
      | none => simp [exec, hb, hv, lift] at hexec
      | some w =>
        simp only [exec, hb, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine ⟨rfl, rfl, key d w _ rfl hd, ?_, ?_, ?_⟩
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            rw [reg_with_rip, reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]
            exact h0 v hv
        · intro v hv
          simp only [r0After] at hv
          by_cases hd0 : d = 0
          · simp only [beq_iff_eq, hd0, if_true] at hv
            simp at hv
          · simp only [beq_iff_eq, hd0, if_false] at hv
            exact h0W v hv
        · show (s.set d _).regs.length = s.regs.length
          simp [St.set]
  | lea d b disp =>
    have hd : d ≠ 4 := by
      intro h; apply hd4; rw [h]; rfl
    cases hb : s.reg b with
    | none => simp [exec, hb, lift] at hexec
    | some x =>
      simp only [exec, hb, lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨rfl, rfl, key d (ea x disp) _ rfl hd, ?_, ?_, ?_⟩
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · simp only [beq_iff_eq, hd0, if_true] at hv
          simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          rw [reg_with_rip, reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm)]
          exact h0 v hv
      · intro v hv
        simp only [r0After] at hv
        by_cases hd0 : d = 0
        · simp only [beq_iff_eq, hd0, if_true] at hv
          simp at hv
        · simp only [beq_iff_eq, hd0, if_false] at hv
          exact h0W v hv
      · show (s.set d _).regs.length = s.regs.length
        simp [St.set]
  | st _ _ _ => simp [isFall] at hfall
  | st8 _ _ _ => simp [isFall] at hfall
  | jmp _ => simp [isFall] at hfall
  | jcc _ _ => simp [isFall] at hfall
  | jtab _ _ _ => simp [isFall] at hfall
  | call _ => simp [isFall] at hfall
  | ret => simp [isFall] at hfall
  | jmpR _ => simp [isFall] at hfall
  | pushI _ => simp [isFall] at hfall
  | push _ => simp [isFall] at hfall
  | pop _ => simp [isFall] at hfall
  | syscall => simp [isFall] at hfall
/-- Every known abstract stack slot holds a 64-bit value. -/
def σBnd (σ : List (Option Nat)) : Prop :=
  ∀ (i v : Nat), σ[i]? = some (some v) → v < W

/-- The local content of an abstract state at one entry, relative
to the frame top `spT`: `sp + 8 * (abstract depth) = spT`. -/
def RipStack (s : X86Int.St) (a : Nat) (σ : List (Option Nat))
    (r0 : Option Nat) (spT : Nat) : Prop :=
  ∃ sp, s.rip = a ∧ s.reg 4 = some sp ∧ sp + 8 * σ.length = spT ∧
    stackAt s sp σ ∧ σBnd σ ∧ (∀ v, r0 = some v → s.reg 0 = some v) ∧
    (∀ v, r0 = some v → v < W) ∧ shapeBnd s.mem ∧
    16 ≤ s.regs.length

/-- A state of the checker proper, at a `chkAnn` entry, with frame
top `sp0` (the stack pointer at checker entry). -/
def ChkMain (sp0 : Nat) (s : X86Int.St) : Prop :=
  ∃ e ∈ chkAnn, RipStack s e.1 e.2.2.2.1 e.2.2.2.2 sp0

/-- Inside `L32`, called from the checker: the return slot holds
the address of a `chkAnn` entry `e` (with unknown `reg 0` on
return), and `e`'s abstract stack sits above it. -/
def ChkL32m (sp0 : Nat) (s : X86Int.St) : Prop :=
  (s.rip = aL32 ∨ s.rip = aL32 + 2) ∧
  ∃ sp e, e ∈ chkAnn ∧ s.reg 4 = some sp ∧
    rd64 s.mem sp = some e.1 ∧ stackAt s (sp + 8) e.2.2.2.1 ∧
    σBnd e.2.2.2.1 ∧
    sp + 8 + 8 * e.2.2.2.1.length = sp0 ∧ e.2.2.2.2 = none ∧
    shapeBnd s.mem ∧ 16 ≤ s.regs.length

/-- Inside `ANNL`, called from the checker: at an `annlAnn` entry,
with the `ANNL` return slot at `spA` holding a `chkAnn` entry
address, whose abstract stack sits above it. -/
def ChkAnnl (sp0 : Nat) (s : X86Int.St) : Prop :=
  ∃ e ∈ annlAnn, ∃ spA e0, e0 ∈ chkAnn ∧
    RipStack s e.1 e.2.2.2.1 e.2.2.2.2 spA ∧
    rd64 s.mem spA = some e0.1 ∧ stackAt s (spA + 8) e0.2.2.2.1 ∧
    σBnd e0.2.2.2.1 ∧
    spA + 8 + 8 * e0.2.2.2.1.length = sp0 ∧ e0.2.2.2.2 = none

/-- Inside `L32`, called from `ANNL`: like `ChkL32m`, but the
return goes to an `annlAnn` entry, with the `ANNL` frame (return
slot at `spA`, checker stack above) preserved below. -/
def ChkL32a (sp0 : Nat) (s : X86Int.St) : Prop :=
  (s.rip = aL32 ∨ s.rip = aL32 + 2) ∧
  ∃ sp spA e e0, e ∈ annlAnn ∧ e0 ∈ chkAnn ∧
    s.reg 4 = some sp ∧ rd64 s.mem sp = some e.1 ∧
    stackAt s (sp + 8) e.2.2.2.1 ∧ σBnd e.2.2.2.1 ∧
    sp + 8 + 8 * e.2.2.2.1.length = spA ∧
    rd64 s.mem spA = some e0.1 ∧ stackAt s (spA + 8) e0.2.2.2.1 ∧
    σBnd e0.2.2.2.1 ∧
    spA + 8 + 8 * e0.2.2.2.1.length = sp0 ∧ e0.2.2.2.2 = none ∧
    e.2.2.2.2 = none ∧ shapeBnd s.mem ∧ 16 ≤ s.regs.length

/-- Inside the `fail1` status-exit block, which is stack-polymorphic:
only the values the block itself pushes matter. -/
def ChkExit1 (s : X86Int.St) : Prop :=
  ((s.rip = aFail1 ∧ ∃ sp, s.reg 4 = some sp) ∨
   (s.rip = aFail1 + 2 ∧ ∃ sp, s.reg 4 = some sp ∧
      rd64 s.mem sp = some 60) ∨
   (s.rip = aFail1 + 3 ∧ s.reg 0 = some 60 ∧ ∃ sp, s.reg 4 = some sp) ∨
   (s.rip = aFail1 + 5 ∧ s.reg 0 = some 60 ∧ ∃ sp, s.reg 4 = some sp ∧
      rd64 s.mem sp = some 1) ∨
   (s.rip = aFail1 + 6 ∧ s.reg 0 = some 60)) ∧
  shapeBnd s.mem ∧ 16 ≤ s.regs.length

/-- Any state the checker can reach from its entry. -/
def ChkP (sp0 : Nat) (s : X86Int.St) : Prop :=
  ChkMain sp0 s ∨ ChkL32m sp0 s ∨ ChkAnnl sp0 s ∨ ChkL32a sp0 s ∨
    ChkExit1 s

theorem stackAt_cons {s : X86Int.St} {sp : Nat} {v : Option Nat}
    {σ : List (Option Nat)} :
    stackAt s sp (v :: σ) ↔
      (∀ w, v = some w → rd64 s.mem sp = some w) ∧
        stackAt s (sp + 8) σ := by
  constructor
  · intro h
    refine ⟨?_, stackAt_tail h⟩
    intro w hw
    have h1 := h 0 w (by simp [hw])
    simpa using h1
  · rintro ⟨h0, h1⟩ i w hi
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero] at hi
      obtain rfl : v = some w := Option.some.inj hi
      have := h0 w rfl
      simpa using this
    | succ i =>
      rw [List.getElem?_cons_succ] at hi
      have h2 := h1 i w hi
      have e : sp + 8 * (i + 1) = sp + 8 + 8 * i := by omega
      rw [e]
      exact h2

/-- The closure at a fall-through entry: no `rsp` write, and the
unique abstract successor. -/
theorem succs_fall {mode : Bool} {a l : Nat} {ins : Ins}
    {σ : List (Option Nat)} {r0 : Option Nat}
    {ss : List (Nat × List (Option Nat) × Option Nat)}
    (hfall : isFall ins = true)
    (hsuccs : succs mode a ins l σ r0 = some ss) :
    destReg ins ≠ some 4 ∧ ss = [(a + l, σ, r0After ins r0)] := by
  have hd4 : destReg ins ≠ some 4 := by
    intro h
    cases ins <;> simp [isFall] at hfall ⊢
    all_goals simp [succs, h] at hsuccs
  refine ⟨hd4, ?_⟩
  cases ins <;> simp only [succs, destReg] at hsuccs
  all_goals
    first
      | exact (Option.some.inj hsuccs).symm
      | (split at hsuccs
         · rename_i h
           exact absurd h hd4
         · exact (Option.some.inj hsuccs).symm)
      | simp [isFall] at hfall

/-- An `exec` of `syscall` with `reg 0 = 60` never continues: it
exits, or sticks on a missing argument register. -/
theorem syscall60_not_ok {s s' : X86Int.St} {l : Nat}
    (h0 : s.reg 0 = some 60) (h : exec .syscall l s = .ok s') :
    False := by
  cases h7 : s.reg 7 <;> cases h6 : s.reg 6 <;> cases h2 : s.reg 2 <;>
    cases h10 : s.reg 10 <;> cases h8 : s.reg 8 <;> cases h9 : s.reg 9 <;>
    simp [exec, h0, h7, h6, h2, h10, h8, h9, X86Env.syscall] at h
set_option maxHeartbeats 0 in
/-- One abstract step at one table entry.  Falls, jumps and stack
operations stay in the `RipStack` family at an abstract successor; a
call to `L32`/`ANNL` produces the subroutine frame; a `ret` (only
in `ANNL`) pops the return address the caller invariant carries. -/
theorem local_step {mode : Bool} {s s' : X86Int.St} {a l spT : Nat}
    {ins : Ins} {σ : List (Option Nat)} {r0 : Option Nat}
    {ss : List (Nat × List (Option Nat) × Option Nat)}
    (hdec : stage1Im a = some (ins, l)) (ha : a + l < W)
    (hspT : spT ≤ 2 ^ 48)
    (hin : RipStack s a σ r0 spT)
    (hsuccs : succs mode a ins l σ r0 = some ss)
    (hstep : step stage1Im s = .ok s') :
    (∃ t ∈ ss, RipStack s' t.1 t.2.1 t.2.2 spT ∧
      (∀ x, spT ≤ x → rd64 s'.mem x = rd64 s.mem x)) ∨
    (∃ t, ins = .call t ∧ (t = aL32 ∨ (mode = false ∧ t = aANNL)) ∧
      s'.rip = t ∧
      ∃ sp, s'.reg 4 = some sp ∧ rd64 s'.mem sp = some (a + l) ∧
        stackAt s' (sp + 8) σ ∧ σBnd σ ∧ sp + 8 + 8 * σ.length = spT ∧
        shapeBnd s'.mem ∧ 16 ≤ s'.regs.length ∧
        (a + l, σ, none) ∈ ss ∧
        (∀ x, spT ≤ x → rd64 s'.mem x = rd64 s.mem x)) ∨
    (ins = .ret ∧ σ = [] ∧
      ∃ sp t, s.reg 4 = some sp ∧ rd64 s.mem sp = some t ∧
        s'.rip = t ∧ s'.reg 4 = some (sp + 8) ∧ s'.mem = s.mem ∧
        shapeBnd s'.mem ∧ 16 ≤ s'.regs.length) := by
  obtain ⟨sp, hrip, h4, hsp, hstk, hσW, h0, h0W, hb, hlen⟩ := hin
  have hW : W = 2 ^ 64 := rfl
  have h04 : (4 : Nat) < s.regs.length := by omega
  have hexec : exec ins l s = .ok s' := by
    have h := hstep
    simp only [X86Int.step, hrip, hdec] at h
    exact h
  have hsp48 : sp ≤ 2 ^ 48 := by omega
  -- push arithmetic, where needed below
  have push_sp : ∀ {val : Nat} {m' : Mem},
      wr64 s.mem ((sp + W - 8) % W) val = some m' → 8 ≤ sp ∧
        (sp + W - 8) % W = sp - 8 := by
    intro val m' hm
    have hlt : (sp + W - 8) % W < 2 ^ 48 := wr64_lt_of_shapeBnd hm hb
    have h8 : 8 ≤ sp := by
      by_cases h8 : 8 ≤ sp
      · exact h8
      · have e : (sp + W - 8) % W = sp + W - 8 :=
          Nat.mod_eq_of_lt (by omega)
        omega
    refine ⟨h8, ?_⟩
    rw [show sp + W - 8 = (sp - 8) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
  cases ins with
  | movI d imm =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .movI d imm) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.movI d imm) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | mov d r =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .mov d r) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.mov d r) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | mov32 d r =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .mov32 d r) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.mov32 d r) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | alu op d r =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .alu op d r) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.alu op d r) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | aluI op d imm =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .aluI op d imm) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.aluI op d imm) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | cmp x y =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .cmp x y) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.cmp x y) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | cmpI x imm =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .cmpI x imm) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.cmpI x imm) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | test x y =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .test x y) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.test x y) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | shlI d c =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .shlI d c) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.shlI d c) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | shrI d c =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .shrI d c) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.shrI d c) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | ld d b disp =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .ld d b disp) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.ld d b disp) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | ld32 d b disp =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .ld32 d b disp) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.ld32 d b disp) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | ld8 d b disp =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .ld8 d b disp) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.ld8 d b disp) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | lea d b disp =>
    obtain ⟨hd4, hss⟩ := succs_fall (ins := .lea d b disp) rfl hsuccs
    obtain ⟨hrip', hmem, h4', hr0', hr0W', hlen'⟩ :=
      fall_sound rfl hd4 h0 h0W hlen hexec
    refine Or.inl ⟨(a + l, σ, r0After (.lea d b disp) r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, ?_, ?_, hsp, ?_, hσW, hr0', hr0W', ?_, ?_⟩, fun x _ => by rw [hmem]⟩
    · rw [hrip', hrip]
    · rw [h4']; exact h4
    · intro i v hi
      rw [hmem]
      exact hstk i v hi
    · rw [hmem]; exact hb
    · omega
  | jmp t =>
    have hss : ss = [(t, σ, r0)] := by
      simp only [succs] at hsuccs
      exact (Option.some.inj hsuccs).symm
    simp only [exec] at hexec
    cases hexec
    refine Or.inl ⟨(t, σ, r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp, rfl, ?_, hsp, hstk, hσW, h0, h0W, hb, hlen⟩, fun x _ => rfl⟩
    rw [reg_with_rip]; exact h4
  | jcc cc t =>
    have hss : ss = [(t, σ, r0), (a + l, σ, r0)] := by
      simp only [succs] at hsuccs
      exact (Option.some.inj hsuccs).symm
    cases hc : cond s.fl cc with
    | none => simp [exec, hc, lift] at hexec
    | some c =>
      simp only [exec, hc, lift, bind, Option.bind, pure] at hexec
      cases hexec
      by_cases hc2 : c = true
      · subst hc2
        simp only [if_true]
        refine Or.inl ⟨(t, σ, r0), by rw [hss]; simp, ⟨sp, by trivial, ?_, hsp, hstk, hσW, h0, h0W, hb, hlen⟩, by intro x _; trivial⟩
        rw [reg_with_rip]; exact h4
      · have : c = false := by cases c <;> simp_all
        subst this
        simp only [if_false]
        refine Or.inl ⟨(a + l, σ, r0), by rw [hss]; simp, ⟨sp, ?_, ?_, hsp, hstk, hσW, h0, h0W, hb, hlen⟩, by intro x _; trivial⟩
        · show s.rip + l = a + l
          rw [hrip]
        · rw [reg_with_rip]; exact h4
  | pushI imm =>
    have hss : ss = [(a + l, some (imm % W) :: σ, r0)] := by
      simp only [succs] at hsuccs
      exact (Option.some.inj hsuccs).symm
    cases hm' : wr64 s.mem ((sp + W - 8) % W) (imm % W) with
    | none => simp [exec, h4, hm', lift] at hexec
    | some m' =>
      obtain ⟨h8, hsp'⟩ := push_sp hm'
      simp only [exec, h4, hm', lift, bind, Option.bind, pure] at hexec
      cases hexec
      rw [hsp'] at hm'
      refine Or.inl ⟨(a + l, some (imm % W) :: σ, r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp - 8, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · show s.rip + l = a + l
        rw [hrip]
      · show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
        rw [reg_set_self h04, hsp']
      · show sp - 8 + 8 * (some (imm % W) :: σ).length = spT
        simp only [List.length_cons]
        omega
      · rw [stackAt_cons]
        refine ⟨?_, ?_⟩
        · intro w hw
          obtain rfl : w = imm % W := by
            have := hw; simp at this; exact this.symm
          show rd64 m' (sp - 8) = some (imm % W)
          have h1 := rd64_wr64 hm'
          rwa [Nat.mod_mod] at h1
        · have e : sp - 8 + 8 = sp := by omega
          rw [e]
          exact stackAt_wr64_below (s := s) hm' hstk (by omega)
      · intro i v hi
        cases i with
        | zero =>
          have h1 : imm % W = v := by simpa using hi
          rw [← h1]; exact Nat.mod_lt _ (by omega)
        | succ i =>
          rw [List.getElem?_cons_succ] at hi
          exact hσW i v hi
      · intro v hv
        have := h0 v hv
        show (s.set 4 (some ((sp + W - 8) % W))).reg 0 = some v
        rw [reg_set_other (show (0:Nat) ≠ 4 by decide)]
        exact this
      · intro v hv
        exact h0W v hv
      · exact shapeBnd_wr64 hm' hb
      · show (s.set 4 _).regs.length ≥ 16
        simp only [St.set, List.length_set]
        exact hlen
      · intro x hx
        show rd64 m' x = rd64 s.mem x
        exact rd64_wr64_other hm' (Or.inl (by omega))
  | push r =>
    have hss : ss = [(a + l, (if r == 0 then r0 else none) :: σ, r0)] := by
      simp only [succs] at hsuccs
      exact (Option.some.inj hsuccs).symm
    cases hr : s.reg r with
    | none => simp [exec, hr, lift] at hexec
    | some w =>
      cases hm' : wr64 s.mem ((sp + W - 8) % W) w with
      | none => simp [exec, hr, h4, hm', lift] at hexec
      | some m' =>
        obtain ⟨h8, hsp'⟩ := push_sp hm'
        simp only [exec, hr, h4, hm', lift, bind, Option.bind, pure] at hexec
        cases hexec
        rw [hsp'] at hm'
        refine Or.inl ⟨(a + l, (if r == 0 then r0 else none) :: σ, r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp - 8, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · show s.rip + l = a + l
          rw [hrip]
        · show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
          rw [reg_set_self h04, hsp']
        · show sp - 8 + 8 * ((if r == 0 then r0 else none) :: σ).length = spT
          simp only [List.length_cons]
          omega
        · rw [stackAt_cons]
          refine ⟨?_, ?_⟩
          · intro u hu
            have h1 := rd64_wr64 hm'
            by_cases hr0 : r = 0
            · subst hr0
              simp only [beq_self_eq_true, if_true] at hu
              have hu2 : s.reg 0 = some u := h0 u hu
              rw [hu2] at hr
              rw [Option.some.injEq] at hr
              have huW : u < W := h0W u hu
              rw [← hr] at h1
              rwa [Nat.mod_eq_of_lt huW] at h1
            · simp only [beq_iff_eq, hr0, if_false] at hu
              simp at hu
          · have e : sp - 8 + 8 = sp := by omega
            rw [e]
            exact stackAt_wr64_below (s := s) hm' hstk (by omega)
        · intro i v hi
          cases i with
          | zero =>
            by_cases hr0 : r = 0
            · subst hr0
              simp only [beq_self_eq_true, if_true] at hi
              have h1 : r0 = some v := by simpa using hi
              exact h0W v h1
            · simp only [beq_iff_eq, hr0, if_false] at hi
              simp at hi
          | succ i =>
            rw [List.getElem?_cons_succ] at hi
            exact hσW i v hi
        · intro v hv
          have := h0 v hv
          show (s.set 4 (some ((sp + W - 8) % W))).reg 0 = some v
          rw [reg_set_other (show (0:Nat) ≠ 4 by decide)]
          exact this
        · intro v hv
          exact h0W v hv
        · exact shapeBnd_wr64 hm' hb
        · show (s.set 4 _).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
        · intro x hx
          show rd64 m' x = rd64 s.mem x
          exact rd64_wr64_other hm' (Or.inl (by omega))
  | pop d =>
    cases hσ : σ with
    | nil =>
      simp [succs, hσ] at hsuccs
    | cons v0 σ' =>
      have hd4' : d ≠ 4 := by
        intro h
        subst h
        simp [succs, hσ] at hsuccs
      have hss : ss = [(a + l, σ', if d == 0 then v0 else r0)] := by
        simp only [succs, hσ] at hsuccs
        split at hsuccs
        · rename_i h
          simp at h
          exact absurd h hd4'
        · exact (Option.some.inj hsuccs).symm
      cases hv : rd64 s.mem sp with
      | none => simp [exec, h4, hv, lift] at hexec
      | some w =>
        have hlenσ : (v0 :: σ').length = σ'.length + 1 := rfl
        have hsp8 : (sp + 8) % W = sp + 8 := by
          apply Nat.mod_eq_of_lt
          have h1 : sp + 8 * (σ'.length + 1) = spT := by
            rw [hσ] at hsp; exact hsp
          omega
        simp only [exec, h4, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine Or.inl ⟨(a + l, σ', if d == 0 then v0 else r0), by rw [hss]; exact List.mem_singleton_self _, ⟨sp + 8, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hb, ?_⟩, fun x _ => rfl⟩
        · show s.rip + l = a + l
          rw [hrip]
        · show ((s.set 4 (some ((sp + 8) % W))).set d (some w)).reg 4 = some (sp + 8)
          rw [reg_set_other (show (4:Nat) ≠ d from fun h => hd4' h.symm),
            reg_set_self h04, hsp8]
        · show sp + 8 + 8 * σ'.length = spT
          have h1 : sp + 8 * (σ'.length + 1) = spT := by
            rw [hσ] at hsp; exact hsp
          omega
        · refine stackAt_tail (s := s) (sp := sp) (v := v0) (σ := σ') ?_
          rw [← hσ]; exact hstk
        · intro i v hi
          have h2 : σ[i + 1]? = some (some v) := by
            rw [hσ, List.getElem?_cons_succ]; exact hi
          exact hσW (i + 1) v h2
        · intro u hu
          by_cases hd0 : d = 0
          · subst hd0
            simp only [beq_self_eq_true, if_true] at hu
            show ((s.set 4 (some ((sp + 8) % W))).set 0 (some w)).reg 0 = some u
            rw [reg_set_self (by simp [St.set]; omega)]
            cases hv0 : v0 with
            | none => rw [hv0] at hu; simp at hu
            | some x =>
              have hx : rd64 s.mem sp = some x := by
                refine stackAt_head (s := s) (sp := sp) (v := x) (σ := σ') ?_
                rw [← hv0, ← hσ]; exact hstk
              have e1 : w = x := Option.some.inj (hv.symm.trans hx)
              have e2 : u = x := Option.some.inj (hu.symm.trans hv0)
              rw [e1, e2]
          · simp only [beq_iff_eq, hd0, if_false] at hu
            have := h0 u hu
            show ((s.set 4 (some ((sp + 8) % W))).set d (some w)).reg 0 = some u
            rw [reg_set_other (show (0:Nat) ≠ d from fun h => hd0 h.symm),
              reg_set_other (show (0:Nat) ≠ 4 by decide)]
            exact this
        · intro u hu
          by_cases hd0 : d = 0
          · subst hd0
            simp only [beq_self_eq_true, if_true] at hu
            cases hv0 : v0 with
            | none => rw [hv0] at hu; simp at hu
            | some x =>
              have hux : u = x := Option.some.inj (hu.symm.trans hv0)
              have hxW : x < W := hσW 0 x (by simp [hσ, hv0])
              omega
          · simp only [beq_iff_eq, hd0, if_false] at hu
            exact h0W u hu
        · show ((s.set 4 (some ((sp + 8) % W))).set d (some w)).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
  | call t =>
    simp only [succs] at hsuccs
    split at hsuccs
    · rename_i hcond
      have hss : ss = [(a + l, σ, none)] := (Option.some.inj hsuccs).symm
      have ht : t = aL32 ∨ (mode = false ∧ t = aANNL) := by
        rw [Bool.or_eq_true] at hcond
        rcases hcond with h | h
        · rw [Bool.and_eq_true] at h
          left
          have e : t = base + 622 := by simpa using h.2
          rw [e]
          rfl
        · rw [Bool.and_eq_true] at h
          have hm : mode = false := by
            have h1 := h.1
            cases mode with
            | false => rfl
            | true => simp at h1
          rw [Bool.or_eq_true] at h
          rcases h.2 with h2 | h2
          · left
            have e : t = base + 622 := by simpa using h2
            rw [e]; rfl
          · right
            refine ⟨hm, ?_⟩
            have e : t = base + 662 := by simpa using h2
            rw [e]; rfl
      cases hm' : wr64 s.mem ((sp + W - 8) % W) (s.rip + l) with
      | none => simp [exec, h4, hm', lift] at hexec
      | some m' =>
        obtain ⟨h8, hsp'⟩ := push_sp hm'
        simp only [exec, h4, hm', lift, bind, Option.bind, pure] at hexec
        cases hexec
        rw [hsp'] at hm'
        refine Or.inr (Or.inl ⟨t, rfl, ht, rfl, sp - 8, ?_, ?_, ?_, hσW, ?_, ?_, ?_, ?_, ?_⟩)
        · show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
          rw [reg_set_self h04, hsp']
        · show rd64 m' (sp - 8) = some (a + l)
          have h1 := rd64_wr64 hm'
          have hnx : s.rip + l = a + l := by rw [hrip]
          rw [hnx, Nat.mod_eq_of_lt ha] at h1
          exact h1
        · have e : sp - 8 + 8 = sp := by omega
          rw [e]
          exact stackAt_wr64_below (s := s) hm' hstk (by omega)
        · omega
        · exact shapeBnd_wr64 hm' hb
        · show (s.set 4 _).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
        · rw [hss]; exact List.mem_singleton_self _
        · intro x hx
          show rd64 m' x = rd64 s.mem x
          exact rd64_wr64_other hm' (Or.inl (by omega))
    · simp at hsuccs
  | ret =>
    cases hσ : σ with
    | cons v0 σ' =>
      simp [succs, hσ] at hsuccs
    | nil =>
      cases hv : rd64 s.mem sp with
      | none => simp [exec, h4, hv, lift] at hexec
      | some t =>
        have hsp8 : (sp + 8) % W = sp + 8 := by
          apply Nat.mod_eq_of_lt
          have h1 : sp + 8 * ([] : List (Option Nat)).length = spT := by
            rw [← hσ]; exact hsp
          simp only [List.length_nil] at h1
          omega
        simp only [exec, h4, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine Or.inr (Or.inr ⟨rfl, rfl, sp, t, h4, hv, rfl, ?_, rfl, hb, ?_⟩)
        · show (s.set 4 (some ((sp + 8) % W))).reg 4 = some (sp + 8)
          rw [reg_set_self h04, hsp8]
        · show (s.set 4 _).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
  | syscall =>
    simp only [succs] at hsuccs
    split at hsuccs
    · rename_i h60
      have hr0 : r0 = some 60 := by simpa using h60
      exact (syscall60_not_ok (h0 60 hr0) hexec).elim
    · simp at hsuccs
  | st _ _ _ => simp [succs] at hsuccs
  | st8 _ _ _ => simp [succs] at hsuccs
  | jtab _ _ _ => simp [succs] at hsuccs
  | jmpR _ => simp [succs] at hsuccs
set_option maxRecDepth 100000 in
theorem chkAnn_addr_ge : ∀ e ∈ chkAnn, base + 0x452 ≤ e.1 := by
  decide +kernel

set_option maxRecDepth 100000 in
theorem annlAnn_addr_ge : ∀ e ∈ annlAnn, base + 662 ≤ e.1 := by
  decide +kernel

set_option maxRecDepth 100000 in
theorem chkAnn_entry0 :
    (base + 0x452, .mov 7 14, 3, [], none) ∈ chkAnn := by
  decide +kernel

set_option maxRecDepth 100000 in
theorem annlAnn_entry0 :
    (aANNL, .push 1, 1, [], none) ∈ annlAnn := by
  decide +kernel

theorem stage1Im_l32_ret : stage1Im (aL32 + 2) = some (.ret, 1) := by
  decide +kernel

/-- The checker-entry state is a `ChkMain` at the first entry. -/
theorem chkMain_entry {s : X86Int.St} {sp0 : Nat}
    (hrip : s.rip = base + 0x452) (h4 : s.reg 4 = some sp0)
    (hb : shapeBnd s.mem) (hlen : 16 ≤ s.regs.length) :
    ChkMain sp0 s :=
  ⟨(base + 0x452, .mov 7 14, 3, [], none), chkAnn_entry0, sp0, hrip,
    h4, by simp, by intro i v hi; simp at hi,
    by intro i v hi; simp at hi, by intro v hv; simp at hv,
    by intro v hv; simp at hv, hb, hlen⟩

/-- Exit-1 from an abstract successor at `aFail1`. -/
theorem chkExit1_of_ripStack {s : X86Int.St} {σ : List (Option Nat)}
    {r0 : Option Nat} {spT : Nat}
    (h : RipStack s aFail1 σ r0 spT) : ChkExit1 s := by
  obtain ⟨sp, hrip, h4, -, -, -, -, -, hb, hlen⟩ := h
  exact ⟨Or.inl ⟨hrip, sp, h4⟩, hb, hlen⟩

set_option maxHeartbeats 0 in
theorem step_main {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s s' : X86Int.St} (hs : ChkMain sp0 s)
    (hstep : step stage1Im s = .ok s') : ChkP sp0 s' := by
  obtain ⟨e, he, hin⟩ := hs
  obtain ⟨ss, hssuccs, hsucc⟩ := annOK_succs (mode := false) chk_ann_ok he
  have ha : e.1 + e.2.2.1 < W := by
    have h1 := chkAnn_addr_bnd e he
    have hb : base = 0x400000 := rfl
    have hW : W = 2 ^ 64 := rfl
    omega
  rcases local_step (chk_entry_decode he) ha hsp0 hin hssuccs hstep with
    (⟨t, ht, hRip, -⟩ |
     ⟨t, -, ht2, hrip', sp', h4', hrd, hstk', hσW', hspEq, hb', hlen',
       hmem, -⟩ | ⟨hins, -, -⟩)
  · rcases hsucc t ht with hfail | ⟨e', he', ha', hσ', hr0'⟩
    · rw [hfail] at hRip
      exact Or.inr (Or.inr (Or.inr (Or.inr (chkExit1_of_ripStack hRip))))
    · exact Or.inl ⟨e', he', by rw [ha', hσ', hr0']; exact hRip⟩
  · rcases hsucc _ hmem with hfail | ⟨e', he', ha', hσ', hr0'⟩
    · have hge := chkAnn_addr_ge e he
      have hb : base = 0x400000 := rfl
      have hf : aFail1 = base + 0x124 := rfl
      omega
    · rcases ht2 with ht | ⟨-, ht⟩
      · refine Or.inr (Or.inl ⟨Or.inl ?_, sp', e', he', h4', ?_, ?_,
          ?_, ?_, hr0', hb', hlen'⟩)
        · rw [hrip', ht]
        · rw [ha']; exact hrd
        · rw [hσ']; exact hstk'
        · rw [hσ']; exact hσW'
        · rw [hσ']; exact hspEq
      · refine Or.inr (Or.inr (Or.inl ⟨(aANNL, .push 1, 1, [], none),
          annlAnn_entry0, sp', e', he', ⟨sp', ?_, h4', by simp, ?_, ?_,
          ?_, ?_, hb', hlen'⟩, ?_, ?_, ?_, ?_, hr0'⟩))
        · rw [hrip', ht]
        · intro i v hi; simp at hi
        · intro i v hi; simp at hi
        · intro v hv; simp at hv
        · intro v hv; simp at hv
        · rw [ha']; exact hrd
        · rw [hσ']; exact hstk'
        · rw [hσ']; exact hσW'
        · rw [hσ']; exact hspEq
  · rw [hins] at hssuccs
    simp [succs] at hssuccs

set_option maxHeartbeats 0 in
theorem step_annl {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s s' : X86Int.St} (hs : ChkAnnl sp0 s)
    (hstep : step stage1Im s = .ok s') : ChkP sp0 s' := by
  obtain ⟨e, he, spA, e0, he0, hin, hrdA, hstkA, hσWA, hspAeq, hr0A⟩ := hs
  obtain ⟨ss, hssuccs, hsucc⟩ := annOK_succs (mode := true) annl_ann_ok he
  have ha : e.1 + e.2.2.1 < W := by
    have h1 := annlAnn_addr_bnd e he
    have hb : base = 0x400000 := rfl
    have hW : W = 2 ^ 64 := rfl
    omega
  have hspA : spA ≤ 2 ^ 48 := by omega
  rcases local_step (annl_entry_decode he) ha hspA hin hssuccs hstep with
    (⟨t, ht, hRip, hframe⟩ |
     ⟨t, -, ht2, hrip', sp', h4', hrd, hstk', hσW', hspEq, hb', hlen',
       hmem, hframe⟩ |
     ⟨-, hσnil, sp1, t, h4s, hrds, hrip', h4'', hmem', hb', hlen'⟩)
  · rcases hsucc t ht with hfail | ⟨e', he', ha', hσ', hr0'⟩
    · rw [hfail] at hRip
      exact Or.inr (Or.inr (Or.inr (Or.inr (chkExit1_of_ripStack hRip))))
    · refine Or.inr (Or.inr (Or.inl ⟨e', he', spA, e0, he0, ?_, ?_, ?_,
        hσWA, hspAeq, hr0A⟩))
      · rw [ha', hσ', hr0']; exact hRip
      · rw [hframe spA (Nat.le_refl _)]; exact hrdA
      · intro i v hi
        rw [hframe _ (by omega)]
        exact hstkA i v hi
  · rcases hsucc _ hmem with hfail | ⟨e', he', ha', hσ', hr0'⟩
    · have hge := annlAnn_addr_ge e he
      have hb : base = 0x400000 := rfl
      have hf : aFail1 = base + 0x124 := rfl
      omega
    · have ht : t = aL32 := by
        rcases ht2 with h | ⟨h, -⟩
        · exact h
        · simp at h
      refine Or.inr (Or.inr (Or.inr (Or.inl ⟨Or.inl ?_, sp', spA, e', e0,
        he', he0, h4', ?_, ?_, ?_, ?_, ?_, ?_, hσWA, hspAeq, hr0A,
        hr0', hb', hlen'⟩)))
      · rw [hrip', ht]
      · rw [ha']; exact hrd
      · rw [hσ']; exact hstk'
      · rw [hσ']; exact hσW'
      · rw [hσ']; exact hspEq
      · rw [hframe spA (Nat.le_refl _)]; exact hrdA
      · intro i v hi
        rw [hframe _ (by omega)]
        exact hstkA i v hi
  · obtain ⟨sp, hrip, h4, hsp, -, -, -, -, -, -⟩ := hin
    have hsp1 : sp1 = sp := by
      rw [h4] at h4s
      exact (Option.some.inj h4s).symm
    have hspA2 : sp = spA := by
      rw [hσnil] at hsp
      simp at hsp
      omega
    have ht2 : t = e0.1 := by
      rw [hsp1, hspA2] at hrds
      rw [hrdA] at hrds
      exact (Option.some.inj hrds).symm
    refine Or.inl ⟨e0, he0, spA + 8, ?_, ?_, ?_, ?_, hσWA, ?_, ?_, hb',
      hlen'⟩
    · rw [hrip', ht2]
    · rw [hsp1, hspA2] at h4''
      exact h4''
    · omega
    · intro i v hi
      rw [hmem']
      exact hstkA i v hi
    · intro v hv
      rw [hr0A] at hv
      simp at hv
    · intro v hv
      rw [hr0A] at hv
      simp at hv

theorem step_l32m {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s s' : X86Int.St} (hs : ChkL32m sp0 s)
    (hstep : step stage1Im s = .ok s') : ChkP sp0 s' := by
  have hW : W = 2 ^ 64 := rfl
  obtain ⟨hrip, sp, e, he, h4, hrd, hstk, hσW, hspEq, hr0, hb, hlen⟩ := hs
  rcases hrip with hrip | hrip
  · have hexec : exec (.ld32 0 2 0) 2 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_l32.d0] at h
      exact h
    cases h2 : s.reg 2 with
    | none => simp [exec, h2, lift] at hexec
    | some x =>
      cases hv : rd32 s.mem (ea x 0) with
      | none => simp [exec, h2, hv, lift] at hexec
      | some w =>
        simp only [exec, h2, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine Or.inr (Or.inl ⟨Or.inr ?_, sp, e, he, ?_, hrd, hstk, hσW,
          hspEq, hr0, hb, ?_⟩)
        · show s.rip + 2 = aL32 + 2
          rw [hrip]
        · show (s.set 0 _).reg 4 = some sp
          rw [reg_set_other (show (4:Nat) ≠ 0 by decide)]
          exact h4
        · show (s.set 0 _).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
  · have hexec : exec .ret 1 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1Im_l32_ret] at h
      exact h
    simp only [exec, h4, hrd, lift, bind, Option.bind, pure] at hexec
    cases hexec
    refine Or.inl ⟨e, he, sp + 8, rfl, ?_, ?_, hstk, hσW, ?_, ?_, hb, ?_⟩
    · show (s.set 4 (some ((sp + 8) % W))).reg 4 = some (sp + 8)
      rw [reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
    · omega
    · intro v hv
      rw [hr0] at hv
      simp at hv
    · intro v hv
      rw [hr0] at hv
      simp at hv
    · show (s.set 4 _).regs.length ≥ 16
      simp only [St.set, List.length_set]
      exact hlen

theorem step_l32a {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s s' : X86Int.St} (hs : ChkL32a sp0 s)
    (hstep : step stage1Im s = .ok s') : ChkP sp0 s' := by
  have hW : W = 2 ^ 64 := rfl
  obtain ⟨hrip, sp, spA, e, e0, he, he0, h4, hrd, hstk, hσW, hspEq,
    hrdA, hstkA, hσWA, hspAeq, hr0A, hr0, hb, hlen⟩ := hs
  rcases hrip with hrip | hrip
  · have hexec : exec (.ld32 0 2 0) 2 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_l32.d0] at h
      exact h
    cases h2 : s.reg 2 with
    | none => simp [exec, h2, lift] at hexec
    | some x =>
      cases hv : rd32 s.mem (ea x 0) with
      | none => simp [exec, h2, hv, lift] at hexec
      | some w =>
        simp only [exec, h2, hv, lift, bind, Option.bind, pure] at hexec
        cases hexec
        refine Or.inr (Or.inr (Or.inr (Or.inl ⟨Or.inr ?_, sp, spA, e, e0,
          he, he0, ?_, hrd, hstk, hσW, hspEq, hrdA, hstkA, hσWA, hspAeq,
          hr0A, hr0, hb, ?_⟩)))
        · show s.rip + 2 = aL32 + 2
          rw [hrip]
        · show (s.set 0 _).reg 4 = some sp
          rw [reg_set_other (show (4:Nat) ≠ 0 by decide)]
          exact h4
        · show (s.set 0 _).regs.length ≥ 16
          simp only [St.set, List.length_set]
          exact hlen
  · have hexec : exec .ret 1 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1Im_l32_ret] at h
      exact h
    simp only [exec, h4, hrd, lift, bind, Option.bind, pure] at hexec
    cases hexec
    refine Or.inr (Or.inr (Or.inl ⟨e, he, spA, e0, he0,
      ⟨sp + 8, rfl, ?_, ?_, hstk, hσW, ?_, ?_, hb, ?_⟩,
      hrdA, hstkA, hσWA, hspAeq, hr0A⟩))
    · show (s.set 4 (some ((sp + 8) % W))).reg 4 = some (sp + 8)
      rw [reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
    · omega
    · intro v hv
      rw [hr0] at hv
      simp at hv
    · intro v hv
      rw [hr0] at hv
      simp at hv
    · show (s.set 4 _).regs.length ≥ 16
      simp only [St.set, List.length_set]
      exact hlen

/-- A readable 64-bit slot lies below `2 ^ 48` in a bounded memory. -/
theorem rd64_lt_of_shapeBnd {m : Mem} {a v : Nat}
    (h : rd64 m a = some v) (hb : shapeBnd m) : a < 2 ^ 48 := by
  have hrd1 : ∃ b, m.rd a = some b := by
    unfold rd64 at h
    cases hrd : m.rdN a 8 with
    | none => rw [hrd] at h; simp at h
    | some bs =>
      have h2 := hrd
      simp only [Mem.rdN, bind, Option.bind] at h2
      split at h2
      · rename_i hb'
        simp [hb'] at h2
      · rename_i b hb'
        exact ⟨b, hb'⟩
  obtain ⟨b, hb2⟩ := hrd1
  exact rd_lt_of_shapeBnd hb2 hb

set_option maxHeartbeats 0 in
theorem step_exit1 {s s' : X86Int.St} (hs : ChkExit1 s)
    (hstep : step stage1Im s = .ok s') : ChkExit1 s' := by
  obtain ⟨hdisj, hb, hlen⟩ := hs
  have hW : W = 2 ^ 64 := rfl
  rcases hdisj with (⟨hrip, sp, h4⟩ | ⟨hrip, sp, h4, hrd⟩ |
      ⟨hrip, hr0, sp, h4⟩ | ⟨hrip, hr0, sp, h4, hrd⟩ | ⟨hrip, hr0⟩)
  · -- pushI 60 at aFail1
    have hexec : exec (.pushI 60) 2 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_refuseCode.fail1.d0] at h
      exact h
    cases hm' : wr64 s.mem ((sp + W - 8) % W) (60 % W) with
    | none => simp [exec, h4, hm', lift] at hexec
    | some m' =>
      simp only [exec, h4, hm', lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨Or.inr (Or.inl ⟨?_, (sp + W - 8) % W, ?_, ?_⟩),
        shapeBnd_wr64 hm' hb, ?_⟩
      · show s.rip + 2 = aFail1 + 2
        rw [hrip]
      · show (s.set 4 (some ((sp + W - 8) % W))).reg 4 =
          some ((sp + W - 8) % W)
        rw [reg_set_self (by omega)]
      · show rd64 m' ((sp + W - 8) % W) = some 60
        have h1 := rd64_wr64 hm'
        have e : (60 % W) % W = 60 := by
          rw [Nat.mod_mod, Nat.mod_eq_of_lt (by omega)]
        rwa [e] at h1
      · show (s.set 4 _).regs.length ≥ 16
        simp only [St.set, List.length_set]
        exact hlen
  · -- pop 0 at aFail1 + 2
    have hlt : sp < 2 ^ 48 := rd64_lt_of_shapeBnd hrd hb
    have hexec : exec (.pop 0) 1 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_refuseCode.fail1.d1] at h
      exact h
    simp only [exec, h4, hrd, lift, bind, Option.bind, pure] at hexec
    cases hexec
    refine ⟨Or.inr (Or.inr (Or.inl ⟨?_, ?_, (sp + 8) % W, ?_⟩)), hb, ?_⟩
    · show s.rip + 1 = aFail1 + 3
      rw [hrip]
    · show ((s.set 4 (some ((sp + 8) % W))).set 0 (some 60)).reg 0 = some 60
      rw [reg_set_self (by simp only [St.set, List.length_set]; omega)]
    · show ((s.set 4 (some ((sp + 8) % W))).set 0 (some 60)).reg 4 =
        some ((sp + 8) % W)
      rw [reg_set_other (show (4:Nat) ≠ 0 by decide),
        reg_set_self (by omega)]
    · show ((s.set 4 (some ((sp + 8) % W))).set 0 (some 60)).regs.length ≥ 16
      simp only [St.set, List.length_set]
      exact hlen
  · -- pushI 1 at aFail1 + 3
    have hexec : exec (.pushI 1) 2 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_refuseCode.fail1.d2] at h
      exact h
    cases hm' : wr64 s.mem ((sp + W - 8) % W) (1 % W) with
    | none => simp [exec, h4, hm', lift] at hexec
    | some m' =>
      simp only [exec, h4, hm', lift, bind, Option.bind, pure] at hexec
      cases hexec
      refine ⟨Or.inr (Or.inr (Or.inr (Or.inl
        ⟨?_, ?_, (sp + W - 8) % W, ?_, ?_⟩))), shapeBnd_wr64 hm' hb, ?_⟩
      · show s.rip + 2 = aFail1 + 5
        rw [hrip]
      · show (s.set 4 (some ((sp + W - 8) % W))).reg 0 = some 60
        rw [reg_set_other (show (0:Nat) ≠ 4 by decide)]
        exact hr0
      · show (s.set 4 (some ((sp + W - 8) % W))).reg 4 =
          some ((sp + W - 8) % W)
        rw [reg_set_self (by omega)]
      · show rd64 m' ((sp + W - 8) % W) = some 1
        have h1 := rd64_wr64 hm'
        have e : (1 % W) % W = 1 := by
          rw [Nat.mod_mod, Nat.mod_eq_of_lt (by omega)]
        rwa [e] at h1
      · show (s.set 4 _).regs.length ≥ 16
        simp only [St.set, List.length_set]
        exact hlen
  · -- pop 7 at aFail1 + 5
    have hexec : exec (.pop 7) 1 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_refuseCode.fail1.d3] at h
      exact h
    simp only [exec, h4, hrd, lift, bind, Option.bind, pure] at hexec
    cases hexec
    refine ⟨Or.inr (Or.inr (Or.inr (Or.inr ⟨?_, ?_⟩))), hb, ?_⟩
    · show s.rip + 1 = aFail1 + 6
      rw [hrip]
    · show ((s.set 4 (some ((sp + 8) % W))).set 7 (some 1)).reg 0 = some 60
      rw [reg_set_other (show (0:Nat) ≠ 7 by decide),
        reg_set_other (show (0:Nat) ≠ 4 by decide)]
      exact hr0
    · show ((s.set 4 (some ((sp + 8) % W))).set 7 (some 1)).regs.length ≥ 16
      simp only [St.set, List.length_set]
      exact hlen
  · -- syscall at aFail1 + 6: never ok
    have hexec : exec .syscall 2 s = .ok s' := by
      have h := hstep
      simp only [X86Int.step, hrip, stage1_refuseCode.fail1.d4] at h
      exact h
    exact (syscall60_not_ok hr0 hexec).elim

theorem chk_step {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s s' : X86Int.St} (hs : ChkP sp0 s)
    (hstep : step stage1Im s = .ok s') : ChkP sp0 s' := by
  rcases hs with h | h | h | h | h
  · exact step_main hsp0 h hstep
  · exact step_l32m hsp0 h hstep
  · exact step_annl hsp0 h hstep
  · exact step_l32a hsp0 h hstep
  · exact Or.inr (Or.inr (Or.inr (Or.inr (step_exit1 h hstep))))

theorem chk_reachable {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s0 : X86Int.St} (h0 : ChkP sp0 s0) :
    ∀ k s, run stage1Im k (.ok s0) = .ok s → ChkP sp0 s := by
  intro k
  induction k generalizing s0 with
  | zero =>
    intro s h
    cases h
    exact h0
  | succ k ih =>
    intro s h
    have hrun : run stage1Im (k + 1) (.ok s0) =
        run stage1Im k (step stage1Im s0) := rfl
    cases hst : step stage1Im s0 with
    | ok s1 =>
      apply ih (chk_step hsp0 h0 hst)
      rwa [hrun, hst] at h
    | exit c =>
      rw [hrun, hst, run_exit] at h
      cases h
    | stuck =>
      rw [hrun, hst, run_stuck] at h
      cases h

/-- No state at a table entry is at a memory syscall: the only
`syscall` entry has abstract `reg 0 = 60`. -/
theorem not_atMem_of_entry {s : X86Int.St}
    {e : Nat × Ins × Nat × List (Option Nat) × Option Nat}
    {mode : Bool}
    {ss : List (Nat × List (Option Nat) × Option Nat)}
    (hdec : stage1Im e.1 = some (e.2.1, e.2.2.1))
    (hsuccs : succs mode e.1 e.2.1 e.2.2.1 e.2.2.2.1 e.2.2.2.2 = some ss)
    (hrip : s.rip = e.1)
    (h0 : ∀ v, e.2.2.2.2 = some v → s.reg 0 = some v)
    (hmem : AtMemSyscall s) : False := by
  obtain ⟨h1, h2⟩ := hmem
  rw [hrip, hdec] at h1
  have hins : e.2.1 = .syscall := by
    have := Option.some.inj h1
    exact congrArg Prod.fst this
  rw [hins] at hsuccs
  simp only [succs] at hsuccs
  split at hsuccs
  · rename_i h60
    have hr0 : e.2.2.2.2 = some 60 := by simpa using h60
    have h60 : s.reg 0 = some 60 := h0 60 hr0
    rcases h2 with h | h <;> rw [h60] at h <;> simp at h
  · simp at hsuccs

theorem chk_legal {sp0 : Nat} {s : X86Int.St} (h : ChkP sp0 s) :
    OracleLegalAt s := by
  apply oracleLegalAt_of_not_atMemSyscall
  intro hmem
  rcases h with h | h | h | h | h
  · obtain ⟨e, he, sp, hrip, -, -, -, -, h0, -, -, -⟩ := h
    obtain ⟨ss, hssuccs, -⟩ := annOK_succs (mode := false) chk_ann_ok he
    exact not_atMem_of_entry (chk_entry_decode he) hssuccs hrip h0 hmem
  · obtain ⟨hrip, -⟩ := h
    rcases hrip with hrip | hrip
    · have h1 := hmem.1
      rw [hrip, stage1_l32.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1Im_l32_ret] at h1
      simp at h1
  · obtain ⟨e, he, spA, e0, he0, hin, _rest⟩ := h
    obtain ⟨sp, hrip, -, -, -, -, h0, -, -, -⟩ := hin
    obtain ⟨ss, hssuccs, -⟩ := annOK_succs (mode := true) annl_ann_ok he
    exact not_atMem_of_entry (annl_entry_decode he) hssuccs hrip h0 hmem
  · obtain ⟨hrip, -⟩ := h
    rcases hrip with hrip | hrip
    · have h1 := hmem.1
      rw [hrip, stage1_l32.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1Im_l32_ret] at h1
      simp at h1
  · obtain ⟨hdisj, -, -⟩ := h
    rcases hdisj with (⟨hrip, -⟩ | ⟨hrip, -⟩ | ⟨hrip, -, -⟩ |
      ⟨hrip, -, -⟩ | ⟨hrip, hr0⟩)
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d1] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d2] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d3] at h1
      simp at h1
    · rcases hmem.2 with h | h <;> rw [hr0] at h <;> simp at h

/-! ### §3. Exit-2 composition infrastructure (toward
`stage1_final_exit2_target`)

`NoHitUpTo s n`: no state within the first `n` steps from `s`
is a nomem hit.  Segment lemmas below export this alongside
their runs so the first-hit case analysis composes. -/

def NoHitUpTo (s : X86Int.St) (n : Nat) : Prop :=
  ∀ j t, j ≤ n → run stage1Im j (Out.ok s) = Out.ok t →
    AtMemSyscall t → t.k.oracle.head? ≠ some .nomem

theorem noHitUpTo_add {s s' : X86Int.St} {n m : Nat}
    (h1 : NoHitUpTo s n)
    (hrun : run stage1Im n (Out.ok s) = Out.ok s')
    (h2 : NoHitUpTo s' m) : NoHitUpTo s (n + m) := by
  intro j t hj hr hmem
  rcases Nat.lt_or_ge j n with hjn | hjn
  · exact h1 j t (by omega) hr hmem
  · have hr' : run stage1Im (j - n) (Out.ok s') = Out.ok t := by
      have h := hr
      rw [show j = n + (j - n) by omega, run_add, hrun] at h
      exact h
    exact h2 (j - n) t (by omega) hr' hmem

theorem noHitUpTo_mono {s : X86Int.St} {n m : Nat}
    (h : NoHitUpTo s n) (hle : m ≤ n) : NoHitUpTo s m :=
  fun j t hj hr hmem => h j t (by omega) hr hmem

/-- Admission transports along a run. -/
theorem oracleAdmits_of_run {s0 s : X86Int.St} {k : Nat}
    (hadm : OracleAdmits s0)
    (hr : run stage1Im k (Out.ok s0) = Out.ok s) :
    OracleAdmits s := by
  intro j t hr'
  exact hadm (k + j) t (by rw [run_add, hr, hr'])

/-- The exit branch of `chk_legal` as a plain non-memsyscall
fact (same proof body). -/
theorem not_atMem_of_chkP {sp0 : Nat} {s : X86Int.St}
    (h : ChkP sp0 s) : ¬ AtMemSyscall s := by
  intro hmem
  rcases h with h | h | h | h | h
  · obtain ⟨e, he, sp, hrip, -, -, -, -, h0, -, -, -⟩ := h
    obtain ⟨ss, hssuccs, -⟩ := annOK_succs (mode := false) chk_ann_ok he
    exact not_atMem_of_entry (chk_entry_decode he) hssuccs hrip h0 hmem
  · obtain ⟨hrip, -⟩ := h
    rcases hrip with hrip | hrip
    · have h1 := hmem.1
      rw [hrip, stage1_l32.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1Im_l32_ret] at h1
      simp at h1
  · obtain ⟨e, he, spA, e0, he0, hin, _rest⟩ := h
    obtain ⟨sp, hrip, -, -, -, -, h0, -, -, -⟩ := hin
    obtain ⟨ss, hssuccs, -⟩ := annOK_succs (mode := true) annl_ann_ok he
    exact not_atMem_of_entry (annl_entry_decode he) hssuccs hrip h0 hmem
  · obtain ⟨hrip, -⟩ := h
    rcases hrip with hrip | hrip
    · have h1 := hmem.1
      rw [hrip, stage1_l32.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1Im_l32_ret] at h1
      simp at h1
  · obtain ⟨hdisj, -, -⟩ := h
    rcases hdisj with (⟨hrip, -⟩ | ⟨hrip, -⟩ | ⟨hrip, -, -⟩ |
      ⟨hrip, -, -⟩ | ⟨hrip, hr0⟩)
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d0] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d1] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d2] at h1
      simp at h1
    · have h1 := hmem.1
      rw [hrip, stage1_refuseCode.fail1.d3] at h1
      simp at h1
    · rcases hmem.2 with h | h <;> rw [hr0] at h <;> simp at h

/-- No nomem hit is reachable from a checker-invariant state,
whatever the remaining oracle is. -/
theorem not_hitsNomem_of_chkP {sp0 : Nat} (hsp0 : sp0 ≤ 2 ^ 48)
    {s0 : X86Int.St} (h0 : ChkP sp0 s0) : ¬ HitsNomem s0 := by
  rintro ⟨k, s, hr, hmem, -⟩
  exact not_atMem_of_chkP (chk_reachable hsp0 h0 k s hr) hmem

/-- Facts packed into a successful `.at` placement. -/
theorem place_at_facts {m : Mem} {a n : Nat}
    (h : X86Env.place m (some (.at a)) n = some (some a)) :
    a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧ m.free a n = true := by
  by_cases hc : a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧
      m.free a n = true
  · exact hc
  · simp [X86Env.place, hc] at h

theorem place_at_cases {m : Mem} {a n : Nat} :
    X86Env.place m (some (.at a)) n = none ∨
      X86Env.place m (some (.at a)) n = some (some a) := by
  by_cases hc : a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧
      m.free a n = true
  · exact Or.inr (place_at hc.1 hc.2.1 hc.2.2.1 hc.2.2.2)
  · exact Or.inl (by simp [X86Env.place, hc])

/-- The bounded status-exit block: exactly 5 steps, and no
interior state is a memory syscall.  Copy of `Status_exit`
(Comp:5897) with the existential resolved to `5` and the
interior clause added from its own `r1`–`r5` states. -/
theorem status_exit_bnd {im : Nat → Option (Ins × Nat)} {entry code : Nat}
    (fc : ExitCode im entry code) (hcode : code < W) :
    ∀ (sp a2 a3 a4 a5 a6 : Nat) (m1 m2 : Mem), 32 ≤ sp → sp < W →
    ∀ s, s.rip = entry → s.reg 4 = some sp →
      s.reg 6 = some a2 → s.reg 2 = some a3 → s.reg 10 = some a4 →
      s.reg 8 = some a5 → s.reg 9 = some a6 →
      wr64 s.mem ((sp + W - 8) % W) 60 = some m1 →
      wr64 m1 ((sp + W - 8) % W) (code % W) = some m2 →
      run im 5 (Out.ok s) = Out.exit (code % 256) ∧
      ∀ j t, j ≤ 5 → run im j (Out.ok s) = Out.ok t →
        ¬ (im t.rip = some (.syscall, 2) ∧
          (t.reg 0 = some 9 ∨ t.reg 0 = some 25)) := by
  intro sp a2 a3 a4 a5 a6 m1 m2 hsp32 hspW s hrip hr4 hr6 hr2 hr10 hr8 hr9 hm1 hm2
  have hslot : (sp + W - 8) % W = sp - 8 := slot_sub (k := 1) (by omega) hspW
  have hm1' : wr64 s.mem ((sp + W - 8) % W) (60 % W) = some m1 := hm1
  have hr60 : rd64 m1 ((sp + W - 8) % W) = some 60 := rd64_wr64 hm1'
  have hcodemod : code % W = code := Nat.mod_eq_of_lt hcode
  have hm2' : wr64 m1 ((sp + W - 8) % W) (code % W) = some m2 := hm2
  have hr0 : rd64 m2 ((sp + W - 8) % W) = some code := by
    have h := rd64_wr64 hm2
    have e : (code % W) % W = code := by
      rw [Nat.mod_mod]
      exact hcodemod
    exact e ▸ h
  have e1 : X86Int.step im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } :=
    step_pushI (by rw [hrip]; exact fc.d0) hr4 hm1'
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 2 := by
    rw [ha1]; show s.rip + 2 = entry + 2; rw [hrip]
  have ha1r4 : a1.reg 4 = some ((sp + W - 8) % W) := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    rw [reg_set_self (reg_some_len hr4)]
  have ha1mem : a1.mem = m1 := by rw [ha1]
  have e2 : X86Int.step im a1 =
      .ok { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
        rip := a1.rip + 1 } :=
    step_pop (by rw [ha1rip]; exact fc.d1) ha1r4 (by rw [ha1mem]; exact hr60)
  obtain ⟨a2s, ha2⟩ : ∃ u : X86Int.St, u =
    { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
      rip := a1.rip + 1 } := ⟨_, rfl⟩
  have hsp8 : ((sp + W - 8) % W + 8) % W = sp := by
    rw [hslot]
    have h1 : sp - 8 + 8 = sp := by omega
    rw [h1]; exact Nat.mod_eq_of_lt hspW
  have ha2rip : a2s.rip = entry + 3 := by
    rw [ha2]; show a1.rip + 1 = entry + 3; rw [ha1rip]
  have ha2r4 : a2s.reg 4 = some sp := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg 4 =
      some sp
    rw [reg_set_other (by decide), reg_set_self (by
      have e : a1.regs.length = s.regs.length := by
        rw [ha1]; exact List.length_set
      have h4len : 4 < s.regs.length := reg_some_len hr4
      omega), hsp8]
  have ha2r0 : a2s.reg 0 = some 60 := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg 0 =
      some 60
    rw [reg_set_self (by
      have e1 : (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
          a1.regs.length := List.length_set
      have e2 : a1.regs.length = s.regs.length := by
        rw [ha1]; exact List.length_set
      have h9len : 9 < s.regs.length := reg_some_len hr9
      omega)]
  have ha2mem : a2s.mem = m1 := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).mem = m1
    rw [mem_set, mem_set]; exact ha1mem
  have ga2 : ∀ q, q ≠ 0 → q ≠ 4 → a2s.reg q = s.reg q := by
    intro q h0 h4
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg q =
      s.reg q
    rw [reg_set_other h0, reg_set_other h4]
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other h4]
  have hm2s : wr64 a2s.mem ((sp + W - 8) % W) (code % W) = some m2 := by
    rw [ha2mem]; exact hm2'
  have e3 : X86Int.step im a2s =
      .ok { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    step_pushI (by rw [ha2rip]; exact fc.d2) ha2r4 hm2s
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 5 := by
    rw [ha3]; show a2s.rip + 2 = entry + 5; rw [ha2rip]
  have ha3r4 : a3.reg 4 = some ((sp + W - 8) % W) := by
    rw [ha3]
    show (a2s.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    rw [reg_set_self (by
      have e : a2s.regs.length = s.regs.length := by
        have e1 : a2s.regs.length = a1.regs.length := by
          rw [ha2]
          exact (List.length_set).trans (List.length_set)
        have e2 : a1.regs.length = s.regs.length := by
          rw [ha1]; exact List.length_set
        rw [e1, e2]
      have h4len : 4 < s.regs.length := reg_some_len hr4
      omega)]
  have ha3mem : a3.mem = m2 := by rw [ha3]
  have ga3 : ∀ q, q ≠ 4 → a3.reg q = a2s.reg q := by
    intro q h4
    rw [ha3]
    show (a2s.set 4 (some ((sp + W - 8) % W))).reg q = a2s.reg q
    rw [reg_set_other h4]
  have e4 : X86Int.step im a3 =
      .ok { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code) with
        rip := a3.rip + 1 } :=
    step_pop (by rw [ha3rip]; exact fc.d3) ha3r4 (by rw [ha3mem]; exact hr0)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code) with
      rip := a3.rip + 1 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 6 := by
    rw [ha4]; show a3.rip + 1 = entry + 6; rw [ha3rip]
  have ha4len : (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
      s.regs.length := by
    have e0 : (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
        a3.regs.length := List.length_set
    have e1 : a3.regs.length = a2s.regs.length := by
      rw [ha3]; exact List.length_set
    have e2 : a2s.regs.length = a1.regs.length := by
      rw [ha2]; exact (List.length_set).trans (List.length_set)
    have e3 : a1.regs.length = s.regs.length := by
      rw [ha1]; exact List.length_set
    rw [e0, e1, e2, e3]
  have ha4r0 : a4.reg 0 = some 60 := by
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg 0 =
      some 60
    rw [reg_set_other (by decide), reg_set_other (by decide), ga3 0 (by decide)]
    exact ha2r0
  have ha4r7 : a4.reg 7 = some code := by
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg 7 =
      some code
    rw [reg_set_self (by
      have h9len : 9 < s.regs.length := reg_some_len hr9
      omega)]
  have ha4r : ∀ q, q ≠ 0 → q ≠ 4 → q ≠ 7 → a4.reg q = s.reg q := by
    intro q h0 h4 h7
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg q =
      s.reg q
    rw [reg_set_other h7, reg_set_other h4, ga3 q h4]
    exact ga2 q h0 h4
  have e5 : X86Int.step im a4 = .exit (code % 256) :=
    step_syscall_exit (by rw [ha4rip]; exact fc.d4) ha4r0 ha4r7
      (by rw [ha4r 6 (by decide) (by decide) (by decide)]; exact hr6)
      (by rw [ha4r 2 (by decide) (by decide) (by decide)]; exact hr2)
      (by rw [ha4r 10 (by decide) (by decide) (by decide)]; exact hr10)
      (by rw [ha4r 8 (by decide) (by decide) (by decide)]; exact hr8)
      (by rw [ha4r 9 (by decide) (by decide) (by decide)]; exact hr9)
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2s := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2s) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2s, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.exit (code % 256) := by
    rw [run_succ_ok im 0 a4, e5, run_zero]
  have hrun5 : run im 5 (Out.ok s) = Out.exit (code % 256) := by
    show run im (1 + 1 + 1 + 1 + 1) (Out.ok s) = Out.exit (code % 256)
    rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3,
      run_add im 1 1, r4, r5]
  refine ⟨hrun5, ?_⟩
  intro j t hj hr hmem
  rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
    j = 5) with rfl | rfl | rfl | rfl | rfl | rfl
  · rw [run_zero] at hr
    simp only [Out.ok.injEq] at hr
    subst hr
    rw [hrip, fc.d0] at hmem
    simp at hmem
  · have ht : t = a1 := by
      have h := r1.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    rw [ha1rip, fc.d1] at hmem
    simp at hmem
  · have hr2 : run im 2 (Out.ok s) = Out.ok a2s := by
      rw [show (2 : Nat) = 1 + 1 by omega, run_add im 1 1, r1, r2]
    have ht : t = a2s := by have h := hr2.symm.trans hr; simp only [Out.ok.injEq] at h; exact h.symm
    subst ht
    rw [ha2rip, fc.d2] at hmem
    simp at hmem
  · have hr3 : run im 3 (Out.ok s) = Out.ok a3 := by
      rw [show (3 : Nat) = 1 + 2 by omega, run_add im 1 2, r1,
        show (2 : Nat) = 1 + 1 by omega, run_add im 1 1, r2, r3]
    have ht : t = a3 := by have h := hr3.symm.trans hr; simp only [Out.ok.injEq] at h; exact h.symm
    subst ht
    rw [ha3rip, fc.d3] at hmem
    simp at hmem
  · have hr4 : run im 4 (Out.ok s) = Out.ok a4 := by
      rw [show (4 : Nat) = 1 + 3 by omega, run_add im 1 3, r1,
        show (3 : Nat) = 1 + 2 by omega, run_add im 1 2, r2,
        show (2 : Nat) = 1 + 1 by omega, run_add im 1 1, r3, r4]
    have ht : t = a4 := by have h := hr4.symm.trans hr; simp only [Out.ok.injEq] at h; exact h.symm
    subst ht
    rcases hmem.2 with h | h <;> rw [ha4r0] at h <;> simp at h
  · rw [hrun5] at hr
    simp at hr

theorem exit2_of_fail2_bnd {sp : Nat} (hsp32 : 32 ≤ sp)
    (hspW : sp < W) :
    ∀ s, s.rip = aFail2 → s.reg 4 = some sp → RegsOk s →
      StackRoom s 1 →
      run stage1Im 5 (Out.ok s) = Out.exit 2 ∧
      NoHitUpTo s 5 := by
  intro s hrip hr4 hregs hroom
  obtain ⟨v6, hv6⟩ := hregs 6 (by decide) (by decide)
  obtain ⟨v2, hv2⟩ := hregs 2 (by decide) (by decide)
  obtain ⟨v10, hv10⟩ := hregs 10 (by decide) (by decide)
  obtain ⟨v8, hv8⟩ := hregs 8 (by decide) (by decide)
  obtain ⟨v9, hv9⟩ := hregs 9 (by decide) (by decide)
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp
    (hroom sp hr4 0 (by decide) 60)
  obtain ⟨m2, hm2s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm1s (hroom sp hr4 0 (by decide) (2 % W)))
  obtain ⟨h5, hno⟩ := status_exit_bnd stage1_exitCode_fail2
    (by unfold W; decide) sp v6 v2 v10 v8 v9 m1 m2 (by omega)
    hspW s hrip hr4 hv6 hv2 hv10 hv8 hv9 hm1s hm2s
  refine ⟨by simpa using h5, ?_⟩
  intro j t hj hr hmem
  have hnot := hno j t hj hr
  intro hhead
  exact hnot hmem


/-! ### §3b. Legality extraction at memory-syscall sites, and
bounded local exit-2 lemmas

Under `OracleAdmits`, the oracle head at a memory syscall
with valid arguments is `.at _` (with a successful placement)
or `.nomem`: a missing or `.count` head makes `place` return
`none`, hence the syscall `none`, contradicting legality. -/

theorem legal_mmap_ret {s : X86Int.St} {a2 : Nat}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 9) (hr7 : s.reg 7 = some 0)
    (hr6 : s.reg 6 = some a2) (hr2 : s.reg 2 = some 3)
    (hr10 : s.reg 10 = some 34) (hr8 : s.reg 8 = some (W - 1))
    (hr9 : s.reg 9 = some 0) :
    ∃ rax k' m', X86Env.syscall s.k s.mem 9 0 a2 3 34 (W - 1) 0 =
      some (.ret rax k' m') := by
  obtain ⟨nr, b1, b2, b3, b4, b5, b6, rax, k', m', h0, h7, h6, h2,
    h10, h8, h9, hsc⟩ := hleg hmem
  have e0 : nr = 9 := Option.some.inj (h0.symm.trans hr0)
  have e1 : b1 = 0 := Option.some.inj (h7.symm.trans hr7)
  have e2 : b2 = a2 := Option.some.inj (h6.symm.trans hr6)
  have e3 : b3 = 3 := Option.some.inj (h2.symm.trans hr2)
  have e4 : b4 = 34 := Option.some.inj (h10.symm.trans hr10)
  have e5 : b5 = W - 1 := Option.some.inj (h8.symm.trans hr8)
  have e6 : b6 = 0 := Option.some.inj (h9.symm.trans hr9)
  subst e0; subst e1; subst e2; subst e3; subst e4; subst e5;
    subst e6
  exact ⟨rax, k', m', hsc⟩

theorem syscall9_none_of_place_none {k : Kern} {m : Mem}
    {a2 : Nat} (ha2 : a2 ≠ 0)
    (hp : X86Env.place m (Kern.next k).1 (X86Env.pages a2) =
      none) :
    X86Env.syscall k m 9 0 a2 3 34 (W - 1) 0 = none := by
  simp only [X86Env.syscall]
  rw [if_neg (by simp [X86Env.PROT_RW, X86Env.MAP_PRIVATE_ANON]),
    if_neg ha2]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hp
    simp only [] at hp
    simp [hp]

theorem place_of_legal_mmap {s : X86Int.St} {A a2 : Nat}
    {rest : List Choice}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 9) (hr7 : s.reg 7 = some 0)
    (hr6 : s.reg 6 = some a2) (hr2 : s.reg 2 = some 3)
    (hr10 : s.reg 10 = some 34) (hr8 : s.reg 8 = some (W - 1))
    (hr9 : s.reg 9 = some 0) (ha2 : a2 ≠ 0)
    (horacle : s.k.oracle = .at A :: rest) :
    X86Env.place s.mem (some (.at A)) (X86Env.pages a2) =
      some (some A) := by
  rcases place_at_cases (m := s.mem) (a := A)
      (n := X86Env.pages a2) with hp | hp
  · exfalso
    obtain ⟨rax, k', m', hsc⟩ :=
      legal_mmap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
    have hnext1 : (Kern.next s.k).1 = some (.at A) := by
      simp [Kern.next, horacle]
    have hnone := syscall9_none_of_place_none ha2
      (by rw [hnext1]; exact hp)
    rw [hnone] at hsc
    simp at hsc
  · exact hp

theorem head_cases_of_legal_mmap {s : X86Int.St} {a2 : Nat}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 9) (hr7 : s.reg 7 = some 0)
    (hr6 : s.reg 6 = some a2) (hr2 : s.reg 2 = some 3)
    (hr10 : s.reg 10 = some 34) (hr8 : s.reg 8 = some (W - 1))
    (hr9 : s.reg 9 = some 0) (ha2 : a2 ≠ 0) :
    (∃ A rest, s.k.oracle = .at A :: rest) ∨
      s.k.oracle.head? = some .nomem := by
  cases hor : s.k.oracle with
  | nil =>
    exfalso
    obtain ⟨rax, k', m', hsc⟩ :=
      legal_mmap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
    have hp : X86Env.place s.mem (Kern.next s.k).1
        (X86Env.pages a2) = none := by
      have h1 : (Kern.next s.k).1 = none := by
        simp [Kern.next, hor]
      rw [h1]
      rfl
    rw [syscall9_none_of_place_none ha2 hp] at hsc
    simp at hsc
  | cons c rest =>
    cases c
    · rename_i A
      exact Or.inl ⟨A, rest, rfl⟩
    · exact Or.inr rfl
    · rename_i n
      exfalso
      obtain ⟨rax, k', m', hsc⟩ :=
        legal_mmap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
      have hp : X86Env.place s.mem (Kern.next s.k).1
          (X86Env.pages a2) = none := by
        have h1 : (Kern.next s.k).1 = some (.count n) := by
          simp [Kern.next, hor]
        rw [h1]
        rfl
      rw [syscall9_none_of_place_none ha2 hp] at hsc
      simp at hsc

theorem legal_mremap_ret {s : X86Int.St}
    {A cap a5 a6 : Nat}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 25) (hr7 : s.reg 7 = some A)
    (hr6 : s.reg 6 = some cap) (hr2 : s.reg 2 = some (cap + cap))
    (hr10 : s.reg 10 = some MREMAP_MAYMOVE)
    (hr8 : s.reg 8 = some a5) (hr9 : s.reg 9 = some a6) :
    ∃ rax k' m', X86Env.syscall s.k s.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE a5 a6 = some (.ret rax k' m') := by
  obtain ⟨nr, b1, b2, b3, b4, b5, b6, rax, k', m', h0, h7, h6, h2,
    h10, h8, h9, hsc⟩ := hleg hmem
  have e0 : nr = 25 := Option.some.inj (h0.symm.trans hr0)
  have e1 : b1 = A := Option.some.inj (h7.symm.trans hr7)
  have e2 : b2 = cap := Option.some.inj (h6.symm.trans hr6)
  have e3 : b3 = cap + cap := Option.some.inj (h2.symm.trans hr2)
  have e4 : b4 = MREMAP_MAYMOVE :=
    Option.some.inj (h10.symm.trans hr10)
  have e5 : b5 = a5 := Option.some.inj (h8.symm.trans hr8)
  have e6 : b6 = a6 := Option.some.inj (h9.symm.trans hr9)
  subst e0; subst e1; subst e2; subst e3; subst e4; subst e5;
    subst e6
  exact ⟨rax, k', m', hsc⟩

theorem syscall25_none_of_place_none {k : Kern} {m : Mem}
    {A cap a5 a6 i : Nat} {r : Region}
    (hfind : m.find A 0 = some (i, r)) (hbase : r.base = A)
    (hlen : r.data.length = X86Env.pages cap) (hrw : r.rw = true)
    (hshrink : ¬ X86Env.pages (cap + cap) < r.data.length)
    (hp : X86Env.place (m.eraseIdx i) (Kern.next k).1
      (X86Env.pages (cap + cap)) = none) :
    X86Env.syscall k m 25 A cap (cap + cap) MREMAP_MAYMOVE a5 a6 =
      none := by
  simp only [X86Env.syscall, hfind, hbase, hlen, hrw, hshrink,
    if_false]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hp
    simp only [] at hp
    simp [hp]

theorem place_of_legal_mremap {s : X86Int.St}
    {A A' cap a5 a6 i : Nat} {r : Region} {rest : List Choice}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 25) (hr7 : s.reg 7 = some A)
    (hr6 : s.reg 6 = some cap) (hr2 : s.reg 2 = some (cap + cap))
    (hr10 : s.reg 10 = some MREMAP_MAYMOVE)
    (hr8 : s.reg 8 = some a5) (hr9 : s.reg 9 = some a6)
    (hfind : s.mem.find A 0 = some (i, r)) (hbase : r.base = A)
    (hlen : r.data.length = X86Env.pages cap) (hrw : r.rw = true)
    (hshrink : ¬ X86Env.pages (cap + cap) < r.data.length)
    (horacle : s.k.oracle = .at A' :: rest) :
    X86Env.place (s.mem.eraseIdx i) (some (.at A'))
        (X86Env.pages (cap + cap)) = some (some A') := by
  rcases place_at_cases (m := s.mem.eraseIdx i) (a := A')
      (n := X86Env.pages (cap + cap)) with hp | hp
  · exfalso
    obtain ⟨rax, k', m', hsc⟩ :=
      legal_mremap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
    have hnext1 : (Kern.next s.k).1 = some (.at A') := by
      simp [Kern.next, horacle]
    have hnone := syscall25_none_of_place_none (a5 := a5)
      (a6 := a6) hfind hbase hlen hrw hshrink
      (by rw [hnext1]; exact hp)
    rw [hnone] at hsc
    simp at hsc
  · exact hp

theorem head_cases_of_legal_mremap {s : X86Int.St}
    {A cap a5 a6 i : Nat} {r : Region}
    (hleg : OracleLegalAt s) (hmem : AtMemSyscall s)
    (hr0 : s.reg 0 = some 25) (hr7 : s.reg 7 = some A)
    (hr6 : s.reg 6 = some cap) (hr2 : s.reg 2 = some (cap + cap))
    (hr10 : s.reg 10 = some MREMAP_MAYMOVE)
    (hr8 : s.reg 8 = some a5) (hr9 : s.reg 9 = some a6)
    (hfind : s.mem.find A 0 = some (i, r)) (hbase : r.base = A)
    (hlen : r.data.length = X86Env.pages cap) (hrw : r.rw = true)
    (hshrink : ¬ X86Env.pages (cap + cap) < r.data.length) :
    (∃ A' rest, s.k.oracle = .at A' :: rest) ∨
      s.k.oracle.head? = some .nomem := by
  cases hor : s.k.oracle with
  | nil =>
    exfalso
    obtain ⟨rax, k', m', hsc⟩ :=
      legal_mremap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
    have hp : X86Env.place (s.mem.eraseIdx i) (Kern.next s.k).1
        (X86Env.pages (cap + cap)) = none := by
      have h1 : (Kern.next s.k).1 = none := by
        simp [Kern.next, hor]
      rw [h1]
      rfl
    rw [syscall25_none_of_place_none hfind hbase hlen hrw hshrink
      hp] at hsc
    simp at hsc
  | cons c rest =>
    cases c
    · rename_i A'
      exact Or.inl ⟨A', rest, rfl⟩
    · exact Or.inr rfl
    · rename_i n
      exfalso
      obtain ⟨rax, k', m', hsc⟩ :=
        legal_mremap_ret hleg hmem hr0 hr7 hr6 hr2 hr10 hr8 hr9
      have hp : X86Env.place (s.mem.eraseIdx i)
          (Kern.next s.k).1 (X86Env.pages (cap + cap)) = none := by
        have h1 : (Kern.next s.k).1 = some (.count n) := by
          simp [Kern.next, hor]
        rw [h1]
        rfl
      rw [syscall25_none_of_place_none hfind hbase hlen hrw
        hshrink hp] at hsc
      simp at hsc

theorem loader_mmap1_nomem_exit2_bnd {sp0 : Nat} {s' : X86Int.St}
    (hrip : s'.rip = base + 0xbf) (hlen16 : s'.regs.length = 16)
    (hr0 : s'.reg 0 = some 9) (hr7 : s'.reg 7 = some 0)
    (hr6 : s'.reg 6 = some 65536) (hr2 : s'.reg 2 = some 3)
    (hr10 : s'.reg 10 = some 34) (hr8 : s'.reg 8 = some (W - 1))
    (hr9 : s'.reg 9 = some 0)
    (hr4 : s'.reg 4 = some (sp0 + 24))
    (hhead : s'.k.oracle.head? = some .nomem)
    (hregs : RegsOk s') (hroom : StackRoom s' 1)
    (hsp32 : 32 ≤ sp0 + 24) (hspW : sp0 + 24 < W) :
    run stage1Im 9 (Out.ok s') = Out.exit 2 := by
  obtain ⟨s'', hrun, hrip'', hr4'', hmem'', -, -, -⟩ :=
    loader_mmap1_nomem hrip hlen16 hr0 hr7 hr6 hr2 hr10 hr8 hr9 hr4
      hhead
  have hroom'' : StackRoom s'' 1 := by
    intro sp hsp i hi v
    have hsp_eq : sp = sp0 + 24 := by
      rw [hr4''] at hsp
      exact (Option.some.inj hsp).symm
    rw [hmem'']
    exact hroom sp (by rw [hsp_eq]; exact hr4) i hi v
  obtain ⟨hk, -⟩ := exit2_of_fail2_bnd hsp32 hspW s'' hrip''
    hr4'' (run_regs_ok hrun hregs) hroom''
  rw [show (9 : Nat) = 4 + 5 by omega, run_add stage1Im 4 5, hrun,
    hk]


theorem loader_grow_nomem_exit2_bnd {A sp0 cap count : Nat}
    {s : X86Int.St} {i : Nat} {r : Region}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hfind : s.mem.find A 0 = some (i, r))
    (hbase : r.base = A) (hdatalen : r.data.length = cap)
    (hrw : r.rw = true)
    (hcapmod : cap % page = 0) (hcapW : cap + cap < W)
    (hhead : s.k.oracle.head? = some .nomem)
    (hregs : RegsOk s) (hroom : StackRoom s 1)
    (hsp32 : 32 ≤ sp0 + 24) (hspW : sp0 + 24 < W) :
    run stage1Im 14 (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', hmem', -, -⟩ :=
    loader_grow_nomem hrip hlen16 hr3 hr12 hr13 hr14 hr4 hr8 hr9
      hfind hbase hdatalen hrw hcapmod hcapW hhead
  have hroom' : StackRoom s' 1 := by
    intro sp hsp i hi v
    have hsp_eq : sp = sp0 + 24 := by
      rw [hr4'] at hsp
      exact (Option.some.inj hsp).symm
    rw [hmem']
    exact hroom sp (by rw [hsp_eq]; exact hr4) i hi v
  obtain ⟨hk, -⟩ := exit2_of_fail2_bnd hsp32 hspW s' hrip' hr4'
    (run_regs_ok hrun hregs) hroom'
  rw [show (14 : Nat) = 9 + 5 by omega, run_add stage1Im 9 5, hrun,
    hk]


theorem parser_mmap1_nomem_exit2_bnd {c nw sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp40 : 40 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    run stage1Im 25 (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', -, hroom', -, -, -⟩ :=
    parser_mmap1_nomem hrip hr6 hr10 hr4 hr7 hr9 hroom (by omega)
      hspW hnw32 hnw0 hhead hlen16
  obtain ⟨hk, -⟩ := exit2_of_fail2_bnd (sp := sp - 8) (by omega)
    (by omega) s' hrip' hr4' (run_regs_ok hrun hregs) hroom'
  rw [show (25 : Nat) = 20 + 5 by omega, run_add stage1Im 20 5,
    hrun, hk]


theorem parser_mmap2_nomem_exit2_bnd {L sp1 : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp32 : 32 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hhead : s.k.oracle.head? = some .nomem)
    (hlen16 : s.regs.length = 16) (hregs : RegsOk s) :
    run stage1Im 17 (Out.ok s) = Out.exit 2 := by
  obtain ⟨s', hrun, hrip', hr4', -, hroom', -, -, -⟩ :=
    parser_mmap2_nomem hrip hr13 hr4 hr7 hr9 hroom (by omega) hspW
      hL hhead hlen16
  obtain ⟨hk, -⟩ := exit2_of_fail2_bnd hsp32 hspW s' hrip' hr4'
    (run_regs_ok hrun hregs) hroom'
  rw [show (17 : Nat) = 12 + 5 by omega, run_add stage1Im 12 5,
    hrun, hk]


/-! ### §3c. Generalized loader segments for arbitrary
placements, with `NoHitUpTo` companions -/

theorem kern_count_spec (k : Kern) (want avail : Nat) :
    (Kern.count k want avail).1 ≤ min want avail ∧
    (0 < min want avail → 0 < (Kern.count k want avail).1) ∧
    ((Kern.count k want avail).2 = k ∨
      ∃ c rest, k.oracle = .count c :: rest ∧
        (Kern.count k want avail).2 =
          { k with oracle := rest }) := by
  cases hor : k.oracle with
  | nil =>
    have hkc : Kern.count k want avail = (min want avail, k) := by
      simp [Kern.count, Kern.next, hor]
    rw [hkc]
    exact ⟨Nat.le_refl _, fun h => h, Or.inl rfl⟩
  | cons c rest =>
    cases c
    · have hkc : Kern.count k want avail =
          (min want avail, k) := by
        simp [Kern.count, Kern.next, hor]
      rw [hkc]
      exact ⟨Nat.le_refl _, fun h => h, Or.inl rfl⟩
    · have hkc : Kern.count k want avail =
          (min want avail, k) := by
        simp [Kern.count, Kern.next, hor]
      rw [hkc]
      exact ⟨Nat.le_refl _, fun h => h, Or.inl rfl⟩
    · rename_i n
      have hkc : Kern.count k want avail =
          (if 0 < n ∧ n ≤ min want avail
            then (n, { k with oracle := rest })
            else (min want avail, { k with oracle := rest })) := by
        simp [Kern.count, Kern.next, hor]
      rw [hkc]
      split
      · rename_i hc
        exact ⟨by omega, by intro _; omega,
          Or.inr ⟨n, rest, rfl, rfl⟩⟩
      · exact ⟨by omega, fun h => h,
          Or.inr ⟨n, rest, rfl, rfl⟩⟩

theorem syscall_read_file_any {k : Kern} {m m' : Mem}
    {f : List Nat} {buf count want : Nat}
    (hfd : k.fds = [some ⟨f, count⟩])
    (hwr : m.wrN buf
      ((f.drop count).take
        (Kern.count k want ((f.drop count).length)).1) =
      some m') :
    X86Env.syscall k m 0 3 buf want 0 0 0 =
      some (.ret (Kern.count k want ((f.drop count).length)).1
        { (Kern.count k want ((f.drop count).length)).2 with
          fds := [some ⟨f, count +
            (Kern.count k want ((f.drop count).length)).1⟩] }
        m') := by
  obtain ⟨n, k1, hkc⟩ : ∃ n k1,
      Kern.count k want ((f.drop count).length) = (n, k1) :=
    ⟨_, _, rfl⟩
  rw [hkc] at hwr
  simp only [] at hwr
  simp only [X86Env.syscall]
  rw [if_neg (by decide : ¬ ((3 : Nat) = 0)),
    if_pos (by decide : 3 ≤ 3)]
  rw [hfd]
  simp only [show (3 : Nat) - 3 = 0 from rfl,
    List.getElem?_cons_zero]
  rw [hkc]
  simp only []
  rw [hwr]
  rfl

theorem syscall_mremap_grow_any {k : Kern} {c1 c2 r : Region}
    {A A' cap a5 a6 : Nat} {rest : List Choice}
    (hfind : Mem.find [c1, c2, r] A 0 = some (2, r))
    (hbase : r.base = A) (hlen : r.data.length = cap)
    (hrw : r.rw = true)
    (hpages : X86Env.pages cap = cap)
    (hpages2 : X86Env.pages (cap + cap) = cap + cap)
    (horacle : k.oracle = .at A' :: rest)
    (hplace : X86Env.place [c1, c2] (some (.at A')) (cap + cap) =
      some (some A')) :
    X86Env.syscall k [c1, c2, r] 25 A cap (cap + cap) MREMAP_MAYMOVE
        a5 a6 =
      some (.ret A' { k with oracle := rest }
        [c1, c2, ⟨A', r.data ++ List.replicate cap 0, true⟩]) := by
  have hval : ¬ (r.base ≠ A ∨ r.data.length ≠ X86Env.pages cap ∨
      ¬ r.rw) := by simp [hbase, hlen, hrw, hpages]
  have hgrow : ¬ X86Env.pages (cap + cap) < r.data.length := by
    rw [hpages2, hlen]; omega
  have herase : List.eraseIdx [c1, c2, r] 2 = [c1, c2] := rfl
  have hplace' : X86Env.place (List.eraseIdx [c1, c2, r] 2)
      (some (.at A')) (X86Env.pages (cap + cap)) =
      some (some A') := by
    rw [herase, hpages2]; exact hplace
  have hnextv : Kern.next k = (some (.at A'), { k with oracle := rest }) := by
    simp [Kern.next, horacle]
  have hnext1 : (Kern.next k).1 = some (.at A') := by
    rw [hnextv]
  have hk1 : (Kern.next k).2 = { k with oracle := rest } := by
    rw [hnextv]
  simp only [X86Env.syscall, hfind, hbase, hlen, hrw]
  rw [if_neg (by simp), if_neg (by simp [hpages]),
    if_neg (by rw [hpages2]; omega : ¬ X86Env.pages (cap + cap) < cap)]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1 hk1
    simp only [] at hnext1 hk1
    subst hnext1
    subst hk1
    dsimp only []
    rw [hplace', herase]
    simp [hpages2, hlen]


theorem loader_mmap1_any {A sp0 : Nat}
    {s' : X86Int.St}
    (hrip : s'.rip = base + 0xbf) (hlen16 : s'.regs.length = 16)
    (hr0 : s'.reg 0 = some 9) (hr7 : s'.reg 7 = some 0)
    (hr6 : s'.reg 6 = some 65536) (hr2 : s'.reg 2 = some 3)
    (hr10 : s'.reg 10 = some 34) (hr8 : s'.reg 8 = some (W - 1))
    (hr9 : s'.reg 9 = some 0) (hr14 : s'.reg 14 = some 3)
    (hr4 : s'.reg 4 = some (sp0 + 24))
    (hr13 : ∃ v, s'.reg 13 = some v)
    (hhead : s'.k.oracle.head? = some (.at A))
    (hplace : X86Env.place s'.mem (some (.at A))
      (X86Env.pages 65536) = some (some A))
    (hAbound : A + 65536 ≤ 2 ^ 47) :
    ∃ s'', run stage1Im 6 (Out.ok s') = Out.ok s'' ∧
      s''.rip = base + 0xd7 ∧
      s''.reg 3 = some A ∧ s''.reg 12 = some 65536 ∧
      s''.reg 13 = some 0 ∧ s''.reg 14 = some 3 ∧
      s''.reg 4 = some (sp0 + 24) ∧
      (∃ v, s''.reg 0 = some v) ∧ (∃ v, s''.reg 10 = some v) ∧
      (∃ v, s''.reg 8 = some v) ∧ (∃ v, s''.reg 9 = some v) ∧
      s''.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] ∧
      s''.k = (Kern.next s'.k).2 ∧ s''.regs.length = 16 ∧
      NoHitUpTo s' 6 := by
  have hc := stage1_loaderCode
  have hpages : X86Env.pages 65536 = 65536 := by decide
  have hsc : X86Env.syscall s'.k s'.mem 9 0 65536 PROT_RW
        MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret A (Kern.next s'.k).2
        (s'.mem ++ [⟨A, List.replicate 65536 0, true⟩])) := by
    have hnext1 : (Kern.next s'.k).1 = some (.at A) :=
      kern_next_of_head hhead
    simp only [X86Env.syscall]
    rw [if_neg (by simp [PROT_RW, MAP_PRIVATE_ANON]),
      if_neg (by decide : ¬ ((65536 : Nat) = 0))]
    cases hnext : Kern.next s'.k with
    | mk c k1 =>
      rw [hnext] at hnext1
      simp only [] at hnext1
      subst hnext1
      simp only [hplace]
      rw [hpages]
      rfl
  have e1 : X86Int.step stage1Im s' =
      .ok { ((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2,
        mem := s'.mem ++ [⟨A, List.replicate 65536 0, true⟩],
        rip := s'.rip + 2 } :=
    step_syscall_ret (by rw [hrip]; exact hc.dbf) hr0 hr7 hr6 hr2 hr10
      hr8 hr9 hsc
  obtain ⟨b1, hb1⟩ : ∃ t : X86Int.St, t =
      { ((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2,
        mem := s'.mem ++ [⟨A, List.replicate 65536 0, true⟩],
        rip := s'.rip + 2 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = base + 0xc1 := by
    rw [hb1]; show s'.rip + 2 = base + 0xc1; rw [hrip]
  have hlenb1 : b1.regs.length = 16 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen16
  have hb1r0 : b1.reg 0 = some A := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 0 = some A
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hb1r4 : b1.reg 4 = some (sp0 + 24) := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4
  have hb1r14 : b1.reg 14 = some 3 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  obtain ⟨v13, hb1r13⟩ : ∃ v, b1.reg 13 = some v := by
    obtain ⟨v, hv⟩ := hr13
    exact ⟨v, by
      rw [hb1]
      show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
        none).reg 13 = some v
      rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
        reg_set_other (by decide : (13 : Nat) ≠ 1),
        reg_set_other (by decide : (13 : Nat) ≠ 0)]
      exact hv⟩
  have hb1mem : b1.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by
    rw [hb1]
  have hb1k : b1.k = (Kern.next s'.k).2 := by rw [hb1]
  -- test rax, rax: A < 2^63 so not negative
  have e2 : X86Int.step stage1Im b1 =
      .ok { b1 with fl := (aluF .and A A).2, rip := b1.rip + 3 } :=
    step_test (by rw [hb1rip]; exact hc.dc1) hb1r0 hb1r0
  obtain ⟨b2, hb2⟩ : ∃ t : X86Int.St, t =
      { b1 with fl := (aluF .and A A).2, rip := b1.rip + 3 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = base + 0xc4 := by
    rw [hb2]; show b1.rip + 3 = base + 0xc4; rw [hb1rip]
  have hb2fl : X86Int.cond b2.fl 9 = some true := by
    rw [hb2]
    show X86Int.cond (aluF .and A A).2 9 = some true
    apply cond_cc9_of_sf_false
    have hland : Nat.land A A = A := by simp
    have hmsb : X86Int.msb A = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt (by omega : A < 2 ^ 63)]
      decide
    simp [aluF, hland, hmsb]
  have hb2r0 : b2.reg 0 = some A := by rw [hb2]; exact hb1r0
  have hb2r4 : b2.reg 4 = some (sp0 + 24) := by rw [hb2]; exact hb1r4
  have hb2r14 : b2.reg 14 = some 3 := by rw [hb2]; exact hb1r14
  have hb2r13 : b2.reg 13 = some v13 := by rw [hb2]; exact hb1r13
  have hb2mem : b2.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb2]; exact hb1mem
  have hb2k : b2.k = (Kern.next s'.k).2 := by rw [hb2]; exact hb1k
  have hlenb2 : b2.regs.length = 16 := by
    rw [hb2]; show b1.regs.length = 16
    exact hlenb1
  -- jns 0xcb (taken)
  have e3 : X86Int.step stage1Im b2 = .ok { b2 with rip := base + 0xcb } := by
    have h := step_jcc (im := stage1Im) (s := b2)
      (by rw [hb2rip]; exact hc.dc4) hb2fl
    simpa using h
  obtain ⟨b3, hb3⟩ : ∃ t : X86Int.St, t =
      { b2 with rip := base + 0xcb } := ⟨_, rfl⟩
  have hb3rip : b3.rip = base + 0xcb := by rw [hb3]
  have hb3r0 : b3.reg 0 = some A := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 0 = some A
    rw [reg_with_rip]; exact hb2r0
  have hb3r4 : b3.reg 4 = some (sp0 + 24) := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hb2r4
  have hb3r14 : b3.reg 14 = some 3 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 14 = some 3
    rw [reg_with_rip]; exact hb2r14
  have hb3r13 : b3.reg 13 = some v13 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 13 = some v13
    rw [reg_with_rip]; exact hb2r13
  have hb3mem : b3.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb3]; exact hb2mem
  have hb3k : b3.k = (Kern.next s'.k).2 := by rw [hb3]; exact hb2k
  have hlenb3 : b3.regs.length = 16 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).regs.length = 16
    rw [len_with_rip]; exact hlenb2
  -- mov rbx, rax
  have e4 : X86Int.step stage1Im b3 =
      .ok { b3.set 3 (some A) with rip := b3.rip + 3 } :=
    step_mov (by rw [hb3rip]; exact hc.dcb) hb3r0
  obtain ⟨b4, hb4⟩ : ∃ t : X86Int.St, t =
      { b3.set 3 (some A) with rip := b3.rip + 3 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = base + 0xce := by
    rw [hb4]; show b3.rip + 3 = base + 0xce; rw [hb3rip]
  have hlenb4 : b4.regs.length = 16 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlenb3
  have hb4r3 : b4.reg 3 = some A := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hb4r4 : b4.reg 4 = some (sp0 + 24) := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 3)]
    exact hb3r4
  have hb4r14 : b4.reg 14 = some 3 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 3)]
    exact hb3r14
  have hb4r13 : b4.reg 13 = some v13 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 13 = some v13
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 3)]
    exact hb3r13
  have hb4mem : b4.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb4]; exact hb3mem
  have hb4k : b4.k = (Kern.next s'.k).2 := by rw [hb4]; exact hb3k
  -- mov r12d, 65536
  have e5 : X86Int.step stage1Im b4 =
      .ok { b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 } :=
    step_movI (by rw [hb4rip]; exact hc.dce)
  obtain ⟨b5, hb5⟩ : ∃ t : X86Int.St, t =
      { b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = base + 0xd4 := by
    rw [hb5]; show b4.rip + 6 = base + 0xd4; rw [hb4rip]
  have hlenb5 : b5.regs.length = 16 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlenb4
  have hb5r12 : b5.reg 12 = some 65536 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 12 = some 65536
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hb5r3 : b5.reg 3 = some A := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 12)]
    exact hb4r3
  have hb5r4 : b5.reg 4 = some (sp0 + 24) := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 12)]
    exact hb4r4
  have hb5r14 : b5.reg 14 = some 3 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 12)]
    exact hb4r14
  have hb5r13 : b5.reg 13 = some v13 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 13 =
      some v13
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 12)]
    exact hb4r13
  have hb5mem : b5.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb5]; exact hb4mem
  have hb5k : b5.k = (Kern.next s'.k).2 := by rw [hb5]; exact hb4k
  -- xor r13, r13
  have e6 : X86Int.step stage1Im b5 =
      .ok { b5.set 13 (some (aluF .xor v13 v13).1) with
        fl := (aluF .xor v13 v13).2, rip := b5.rip + 3 } :=
    step_alu_xor (by rw [hb5rip]; exact hc.dd4) hb5r13 hb5r13
  obtain ⟨b6, hb6⟩ : ∃ t : X86Int.St, t =
      { b5.set 13 (some (aluF .xor v13 v13).1) with
        fl := (aluF .xor v13 v13).2, rip := b5.rip + 3 } := ⟨_, rfl⟩
  have hb6rip : b6.rip = base + 0xd7 := by
    rw [hb6]; show b5.rip + 3 = base + 0xd7; rw [hb5rip]
  have hlenb6 : b6.regs.length = 16 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).regs.length = 16
    rw [len_set]
    exact hlenb5
  have hb6r13 : b6.reg 13 = some 0 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 13 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hb6r3 : b6.reg 3 = some A := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 13)]
    exact hb5r3
  have hb6r12 : b6.reg 12 = some 65536 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 12 = some 65536
    rw [reg_set_other (by decide : (12 : Nat) ≠ 13)]
    exact hb5r12
  have hb6r14 : b6.reg 14 = some 3 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 13)]
    exact hb5r14
  have hb6r4 : b6.reg 4 = some (sp0 + 24) := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact hb5r4
  have hb6mem : b6.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb6]; exact hb5mem
  have hb6k : b6.k = (Kern.next s'.k).2 := by rw [hb6]; exact hb5k
  have hb6r0 : b6.reg 0 = some A := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 0 = some A
    rw [reg_set_other (by decide : (0 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 0 = some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 0 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 0 = some A
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r0
  have hb1r10 : b1.reg 10 = some 34 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hr10
  have hb6r10 : b6.reg 10 = some 34 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 10 = some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 10 = some 34
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r10
  have hb1r8 : b1.reg 8 = some (W - 1) := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8
  have hb6r8 : b6.reg 8 = some (W - 1) := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 8 = some (W - 1)
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 8 =
      some (W - 1)
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 8 = some (W - 1)
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r8
  have hb1r9 : b1.reg 9 = some 0 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9
  have hb6r9 : b6.reg 9 = some 0 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 9 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 9 = some 0
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r9
  have r1 : run stage1Im 1 (Out.ok s') = Out.ok b1 := by
    rw [run_succ_ok stage1Im 0 s', e1, ← hb1, run_zero]
  have r2 : run stage1Im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok stage1Im 0 b1, e2, ← hb2, run_zero]
  have r3 : run stage1Im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok stage1Im 0 b2, e3, ← hb3, run_zero]
  have r4 : run stage1Im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok stage1Im 0 b3, e4, ← hb4, run_zero]
  have r5 : run stage1Im 1 (Out.ok b4) = Out.ok b5 := by
    rw [run_succ_ok stage1Im 0 b4, e5, ← hb5, run_zero]
  have r6 : run stage1Im 1 (Out.ok b5) = Out.ok b6 := by
    rw [run_succ_ok stage1Im 0 b5, e6, ← hb6, run_zero]
  have hrun : run stage1Im 6 (Out.ok s') = Out.ok b6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r5, r6]
  refine ⟨b6, hrun, hb6rip, hb6r3, hb6r12, hb6r13, hb6r14,
    hb6r4, ⟨A, hb6r0⟩, ⟨34, hb6r10⟩, ⟨W - 1, hb6r8⟩, ⟨0, hb6r9⟩,
    hb6mem, hb6k, hlenb6, ?_⟩
  intro j t hj hr hmem
  rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
    j = 5 ∨ j = 6) with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · rw [run_zero] at hr
    simp only [Out.ok.injEq] at hr
    subst hr
    rw [hhead]
    simp
  · have ht : t = b1 := by
      have h := r1.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb1rip, hc.dc1] at h1
    simp at h1
  · have hr2 : run stage1Im 2 (Out.ok s') = Out.ok b2 := by
      rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        r1, r2]
    have ht : t = b2 := by
      have h := hr2.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb2rip, hc.dc4] at h1
    simp at h1
  · have hr3 : run stage1Im 3 (Out.ok s') = Out.ok b3 := by
      rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
        r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        r2, r3]
    have ht : t = b3 := by
      have h := hr3.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb3rip, hc.dcb] at h1
    simp at h1
  · have hr4 : run stage1Im 4 (Out.ok s') = Out.ok b4 := by
      rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
        r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
        r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        r3, r4]
    have ht : t = b4 := by
      have h := hr4.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb4rip, hc.dce] at h1
    simp at h1
  · have hr5 : run stage1Im 5 (Out.ok s') = Out.ok b5 := by
      rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
        r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
        r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
        r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        r4, r5]
    have ht : t = b5 := by
      have h := hr5.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb5rip, hc.dd4] at h1
    simp at h1
  · have ht : t = b6 := by
      have h := hrun.symm.trans hr
      simp only [Out.ok.injEq] at h
      exact h.symm
    subst ht
    have h1 := hmem.1
    rw [hb6rip, hc.dd7] at h1
    simp at h1

theorem loader_read_once_any {f : List Nat} {A sp0 cap count capB : Nat}
    {c1 c2 buf : Region} {s : X86Int.St}
    (hrip : s.rip = base + 0xd7) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr0 : ∃ v, s.reg 0 = some v) (hr10 : ∃ v, s.reg 10 = some v)
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hmem : s.mem = [c1, c2, buf]) (hbufbase : buf.base = A)
    (hbufrw : buf.rw = true) (hbufhi : buf.hi = A + cap)
    (hdisj1 : c1.hi ≤ A ∨ A + capB ≤ c1.base)
    (hdisj2 : c2.hi ≤ A ∨ A + capB ≤ c2.base) (hcapB : cap ≤ capB)
    (hcontent : s.mem.rdN A count = some (f.take count))
    (hfd : s.k.fds = [some ⟨f, count⟩])
    (hfB : ∀ b ∈ f, b < 256)
    (hcount : count ≤ f.length) (hcountcap : count < cap)
    (hAcap : A + cap < W) (hcap63 : cap < 2 ^ 63) :
    ∃ n s', n ≤ cap - count ∧ n ≤ f.length - count ∧
      (count < f.length → 0 < n) ∧
      run stage1Im 10 (Out.ok s) = Out.ok s' ∧
      (s'.rip = base + 0x122 ∧ f.length = count ∨
        s'.rip = base + 0xf1 ∧ count < f.length) ∧
      s'.reg 0 = some n ∧
      s'.reg 3 = some A ∧ s'.reg 12 = some cap ∧
      s'.reg 13 = some count ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 10 = some v) ∧ (∃ v, s'.reg 8 = some v) ∧
      (∃ v, s'.reg 9 = some v) ∧
      s'.k = { (Kern.count s.k (cap - count)
          (f.length - count)).2 with
        fds := [some ⟨f, count + n⟩] } ∧
      (∃ buf', s'.mem = [c1, c2, buf'] ∧ buf'.base = A ∧
        buf'.hi = A + cap ∧ buf'.rw = true) ∧
      s'.mem.rdN A (count + n) = some (f.take (count + n)) ∧
      s'.regs.length = 16 ∧ NoHitUpTo s 10 := by
  have hc := stage1_loaderCode
  obtain ⟨n, hn⟩ : ∃ m : Nat,
      m = (Kern.count s.k (cap - count) (f.length - count)).1 :=
    ⟨_, rfl⟩
  have hspec := kern_count_spec s.k (cap - count) (f.length - count)
  have hnle : n ≤ min (cap - count) (f.length - count) := by
    rw [hn]
    exact hspec.1
  have hncap : n ≤ cap - count :=
    Nat.le_trans hnle (Nat.min_le_left _ _)
  have hnlen : n ≤ f.length - count :=
    Nat.le_trans hnle (Nat.min_le_right _ _)
  have hnpos : count < f.length → 0 < n := by
    intro hlt
    rw [hn]
    exact hspec.2.1 (by omega)
  have hdrop : (f.drop count).length = f.length - count := by
    rw [List.length_drop]
  have htake : (f.drop count).take
        (Kern.count s.k (cap - count) ((f.drop count).length)).1 =
      (f.drop count).take n := by
    rw [hdrop, ← hn]
  obtain ⟨v0, hr0v⟩ := hr0
  obtain ⟨v10, hr10v⟩ := hr10
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  have hbslen : ((f.drop count).take n).length = n := by
    rw [List.length_take, hdrop, Nat.min_eq_left hnlen]
  have hdwr1 : c1.hi ≤ buf.base ∨ buf.hi ≤ c1.base := by
    rcases hdisj1 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdwr2 : c2.hi ≤ buf.base ∨ buf.hi ≤ c2.base := by
    rcases hdisj2 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  obtain ⟨buf', hwr, hbuf'b, hbuf'hi, hbuf'rw⟩ :=
    wrN_third_some (c1 := c1) (c2 := c2) (r := buf) (a := A + count)
      ((f.drop count).take n) hdwr1 hdwr2 hbufrw (by omega) (by
      rw [hbslen, hbufhi]
      have hnn : n ≤ cap - count := hncap
      omega)
  have hwr' : s.mem.wrN (A + count) ((f.drop count).take n) =
      some [c1, c2, buf'] := by
    rwa [hmem] 
  have hsc : X86Env.syscall s.k s.mem 0 3 (A + count) (cap - count) 0 0 0 =
      some (.ret n
        { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] } [c1, c2, buf']) := by
    have h := syscall_read_file_any (k := s.k) (m := s.mem)
      (m' := [c1, c2, buf']) (f := f) (buf := A + count)
      (count := count) (want := cap - count) hfd (by
      rwa [htake])
    rwa [hdrop, ← hn] at h
  have hrdN' : Mem.rdN [c1, c2, buf'] A (count + n) =
      some (f.take (count + n)) := by
    have hpre : Mem.rdN [c1, c2, buf'] A count = some (f.take count) := by
      have h := rdN_wrN_prefix ((f.drop count).take n) hwr'
      rwa [hcontent] at h
    have hnew : Mem.rdN [c1, c2, buf'] (A + count) n =
        some ((f.drop count).take n) := by
      have h := rdN_wrN_same ((f.drop count).take n) s.mem (A + count) hwr'
      rw [hbslen] at h
      have hmapid : ∀ l : List Nat, (∀ b ∈ l, b < 256) →
          l.map (· % 256) = l := by
        intro l hl
        induction l with
        | nil => rfl
        | cons a l ih =>
          simp only [List.map_cons]
          rw [Nat.mod_eq_of_lt (hl a (by simp)),
            ih (fun b hb => hl b (List.mem_cons_of_mem a hb))]
      have hmap : ((f.drop count).take n).map (· % 256) =
          (f.drop count).take n :=
        hmapid _ (fun b hb => hfB b
          (List.mem_of_mem_drop (List.mem_of_mem_take hb)))
      rwa [hmap] at h
    have hcomb := rdN_combine (m := [c1, c2, buf']) (a := A) (n := count)
      (xs := f.take count) (by rw [List.length_take]; omega) hpre
      (k := n) (ys := (f.drop count).take n) hbslen hnew
    rw [List.take_add]
    exact hcomb
  -- steps 1-2: mov rsi, rbx; add rsi, r13
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 6 (some A) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.dd7) hr3
  obtain ⟨d1, hd1⟩ : ∃ t : X86Int.St, t =
      { s.set 6 (some A) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have hd1rip : d1.rip = base + 0xda := by
    rw [hd1]; show s.rip + 3 = base + 0xda; rw [hrip]
  have hlen1 : d1.regs.length = 16 := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen16
  have hd1r6 : d1.reg 6 = some A := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 6 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd1r13 : d1.reg 13 = some count := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    exact hr13
  have e2 : X86Int.step stage1Im d1 =
      .ok { d1.set 6 (some (aluF .add A count).1) with
        fl := (aluF .add A count).2, rip := d1.rip + 3 } :=
    step_alu_add (by rw [hd1rip]; exact hc.dda) hd1r6 hd1r13
  obtain ⟨d2, hd2⟩ : ∃ t : X86Int.St, t =
      { d1.set 6 (some (aluF .add A count).1) with
        fl := (aluF .add A count).2, rip := d1.rip + 3 } := ⟨_, rfl⟩
  have hd2rip : d2.rip = base + 0xdd := by
    rw [hd2]; show d1.rip + 3 = base + 0xdd; rw [hd1rip]
  have hlen2 : d2.regs.length = 16 := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).regs.length = 16
    rw [len_set]
    exact hlen1
  have hd2r6 : d2.reg 6 = some (A + count) := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 6 = some (A + count)
    rw [reg_set_self (by omega), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  -- steps 3-4: mov rdx, r12; sub rdx, r13
  have hd2r12 : d2.reg 12 = some cap := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hr12
  have e3 : X86Int.step stage1Im d2 =
      .ok { d2.set 2 (some cap) with rip := d2.rip + 3 } :=
    step_mov (by rw [hd2rip]; exact hc.ddd) hd2r12
  obtain ⟨d3, hd3⟩ : ∃ t : X86Int.St, t =
      { d2.set 2 (some cap) with rip := d2.rip + 3 } := ⟨_, rfl⟩
  have hd3rip : d3.rip = base + 0xe0 := by
    rw [hd3]; show d2.rip + 3 = base + 0xe0; rw [hd2rip]
  have hlen3 : d3.regs.length = 16 := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen2
  have hd3r2 : d3.reg 2 = some cap := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd3r13 : d3.reg 13 = some count := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 6)]
    exact hd1r13
  have e4 : X86Int.step stage1Im d3 =
      .ok { d3.set 2 (some (aluF .sub cap count).1) with
        fl := (aluF .sub cap count).2, rip := d3.rip + 3 } :=
    step_alu_sub (by rw [hd3rip]; exact hc.de0) hd3r2 hd3r13
  obtain ⟨d4, hd4⟩ : ∃ t : X86Int.St, t =
      { d3.set 2 (some (aluF .sub cap count).1) with
        fl := (aluF .sub cap count).2, rip := d3.rip + 3 } := ⟨_, rfl⟩
  have hd4rip : d4.rip = base + 0xe3 := by
    rw [hd4]; show d3.rip + 3 = base + 0xe3; rw [hd3rip]
  have hlen4 : d4.regs.length = 16 := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).regs.length = 16
    rw [len_set]
    exact hlen3
  have hd4r2 : d4.reg 2 = some (cap - count) := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 2 = some (cap - count)
    rw [reg_set_self (by omega), aluF_sub_fst]
    have h1 : cap + W - count = (cap - count) + W := by omega
    rw [h1, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  -- steps 5-6: xor eax, eax; mov edi, r14d
  have hd4r0 : d4.reg 0 = some v0 := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 0 = some v0
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 0 = some v0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 0 = some v0
    rw [reg_set_other (by decide : (0 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 0 = some v0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hr0v
  have e5 : X86Int.step stage1Im d4 =
      .ok { d4.set 0 (some (aluF .xor v0 v0).1) with
        fl := (aluF .xor v0 v0).2, rip := d4.rip + 2 } :=
    step_alu_xor (by rw [hd4rip]; exact hc.de3) hd4r0 hd4r0
  obtain ⟨d5, hd5⟩ : ∃ t : X86Int.St, t =
      { d4.set 0 (some (aluF .xor v0 v0).1) with
        fl := (aluF .xor v0 v0).2, rip := d4.rip + 2 } := ⟨_, rfl⟩
  have hd5rip : d5.rip = base + 0xe5 := by
    rw [hd5]; show d4.rip + 2 = base + 0xe5; rw [hd4rip]
  have hlen5 : d5.regs.length = 16 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).regs.length = 16
    rw [len_set]
    exact hlen4
  have hd5r0 : d5.reg 0 = some 0 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 0 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hd5r14 : d5.reg 14 = some 3 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    exact hr14
  have e6 : X86Int.step stage1Im d5 =
      .ok { d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 } :=
    step_mov32 (by rw [hd5rip]; exact hc.de5) hd5r14
  obtain ⟨d6, hd6⟩ : ∃ t : X86Int.St, t =
      { d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 } := ⟨_, rfl⟩
  have hd6rip : d6.rip = base + 0xe8 := by
    rw [hd6]; show d5.rip + 3 = base + 0xe8; rw [hd5rip]
  have hlen6 : d6.regs.length = 16 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with
      rip := d5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen5
  have hd6r7 : d6.reg 7 = some 3 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 7 =
      some 3
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd6r0 : d6.reg 0 = some 0 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 0 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hd5r0
  have hd6r6 : d6.reg 6 = some (A + count) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 6 =
      some (A + count)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 6 = some (A + count)
    rw [reg_set_other (by decide : (6 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 6 = some (A + count)
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 6 =
      some (A + count)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hd2r6
  have hd6r2 : d6.reg 2 = some (cap - count) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 2 =
      some (cap - count)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 2 = some (cap - count)
    rw [reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact hd4r2
  have hd6r10 : d6.reg 10 = some v10 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 10 =
      some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 10 = some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 10 = some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6)]
    exact hr10v
  have hd6r8 : d6.reg 8 = some v8 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    exact hr8v
  have hd6r9 : d6.reg 9 = some v9 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    exact hr9v
  have hd6mem : d6.mem = s.mem := by
    rw [hd6, hd5, hd4, hd3, hd2, hd1]
    rfl
  have hd6k : d6.k = s.k := by
    rw [hd6, hd5, hd4, hd3, hd2, hd1]
    rfl
  have hd6r3 : d6.reg 3 = some A := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 6)]
    exact hr3
  have hd6r12 : d6.reg 12 = some cap := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 2)]
    exact hd2r12
  have hd6r13 : d6.reg 13 = some count := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    exact hd3r13
  have hd6r4 : d6.reg 4 = some (sp0 + 24) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact hr4
  -- step 7: read(3, buf + count, cap - count)
  have hsc6 : X86Env.syscall d6.k d6.mem 0 3 (A + count) (cap - count)
        0 0 0 =
      some (.ret n { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] }
        [c1, c2, buf']) := by
    rw [hd6k, hd6mem]
    exact hsc
  have e7 : X86Int.step stage1Im d6 =
      .ok { ((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11 none with
        k := { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] },
        mem := [c1, c2, buf'], rip := d6.rip + 2 } :=
    step_syscall_ret (by rw [hd6rip]; exact hc.de8) hd6r0 hd6r7 hd6r6
      hd6r2 hd6r10 hd6r8 hd6r9 hsc6
  obtain ⟨d7, hd7⟩ : ∃ t : X86Int.St, t =
      { ((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11 none with
        k := { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] },
        mem := [c1, c2, buf'], rip := d6.rip + 2 } := ⟨_, rfl⟩
  have hd7rip : d7.rip = base + 0xea := by
    rw [hd7]; show d6.rip + 2 = base + 0xea; rw [hd6rip]
  have hlen7 : d7.regs.length = 16 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen6
  have hd7r0 : d7.reg 0 = some n := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hd7r3 : d7.reg 3 = some A := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 11),
      reg_set_other (by decide : (3 : Nat) ≠ 1),
      reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hd6r3
  have hd7r12 : d7.reg 12 = some cap := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 11),
      reg_set_other (by decide : (12 : Nat) ≠ 1),
      reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hd6r12
  have hd7r13 : d7.reg 13 = some count := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
      reg_set_other (by decide : (13 : Nat) ≠ 1),
      reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hd6r13
  have hd7r14 : d7.reg 14 = some 3 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact hd5r14
  have hd7r4 : d7.reg 4 = some (sp0 + 24) := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hd6r4
  have hd7r10 : d7.reg 10 = some v10 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hd6r10
  have hd7r8 : d7.reg 8 = some v8 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hd6r8
  have hd7r9 : d7.reg 9 = some v9 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hd6r9
  have hd7mem : d7.mem = [c1, c2, buf'] := by rw [hd7]
  have hd7k : d7.k = { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] } := by
    rw [hd7]
  -- step 8: test rax, rax
  have e8 : X86Int.step stage1Im d7 =
      .ok { d7 with fl := (aluF .and n n).2, rip := d7.rip + 3 } :=
    step_test (by rw [hd7rip]; exact hc.dea) hd7r0 hd7r0
  obtain ⟨d8, hd8⟩ : ∃ t : X86Int.St, t =
      { d7 with fl := (aluF .and n n).2, rip := d7.rip + 3 } := ⟨_, rfl⟩
  have hd8rip : d8.rip = base + 0xed := by
    rw [hd8]; show d7.rip + 3 = base + 0xed; rw [hd7rip]
  have hn63 : n < 2 ^ 63 := by
    have hle : n ≤ cap - count := hncap
    omega
  have hd8sf : d8.fl.sf = some false := by
    rw [hd8]
    show (aluF .and n n).2.sf = some false
    have hland : Nat.land n n = n := by simp
    have hmsb : X86Int.msb n = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt hn63]
      decide
    simp [aluF, hland, hmsb]
  have hd8zf : d8.fl.zf = some (n == 0) := by
    rw [hd8]
    show (aluF .and n n).2.zf = some (n == 0)
    rw [aluF_zf, aluF_and_fst_self]
  have hd8r0 : d8.reg 0 = some n := by rw [hd8]; exact hd7r0
  have hd8r3 : d8.reg 3 = some A := by rw [hd8]; exact hd7r3
  have hd8r12 : d8.reg 12 = some cap := by rw [hd8]; exact hd7r12
  have hd8r13 : d8.reg 13 = some count := by rw [hd8]; exact hd7r13
  have hd8r14 : d8.reg 14 = some 3 := by rw [hd8]; exact hd7r14
  have hd8r4 : d8.reg 4 = some (sp0 + 24) := by rw [hd8]; exact hd7r4
  have hd8r10 : d8.reg 10 = some v10 := by rw [hd8]; exact hd7r10
  have hd8r8 : d8.reg 8 = some v8 := by rw [hd8]; exact hd7r8
  have hd8r9 : d8.reg 9 = some v9 := by rw [hd8]; exact hd7r9
  have hd8mem : d8.mem = [c1, c2, buf'] := by rw [hd8]; exact hd7mem
  have hd8k : d8.k = { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] } := by rw [hd8]; exact hd7k
  have hlen8 : d8.regs.length = 16 := by rw [hd8]; exact hlen7
  -- step 9: js fail1 (not taken: n ≥ 0)
  have hd8fl8 : X86Int.cond d8.fl 8 = some false :=
    cond_cc8_of_sf_false hd8sf
  have e9 : X86Int.step stage1Im d8 =
      .ok { d8 with rip := d8.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := d8)
      (by rw [hd8rip]; exact hc.ded) hd8fl8
    simpa using h
  obtain ⟨d9, hd9⟩ : ∃ t : X86Int.St, t =
      { d8 with rip := d8.rip + 2 } := ⟨_, rfl⟩
  have hd9rip : d9.rip = base + 0xef := by
    rw [hd9]; show d8.rip + 2 = base + 0xef; rw [hd8rip]
  have hd9r0 : d9.reg 0 = some n := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 0 = some n
    rw [reg_with_rip]; exact hd8r0
  have hd9r3 : d9.reg 3 = some A := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 3 = some A
    rw [reg_with_rip]; exact hd8r3
  have hd9r12 : d9.reg 12 = some cap := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 12 = some cap
    rw [reg_with_rip]; exact hd8r12
  have hd9r13 : d9.reg 13 = some count := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 13 = some count
    rw [reg_with_rip]; exact hd8r13
  have hd9r14 : d9.reg 14 = some 3 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip]; exact hd8r14
  have hd9r4 : d9.reg 4 = some (sp0 + 24) := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hd8r4
  have hd9r10 : d9.reg 10 = some v10 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 10 = some v10
    rw [reg_with_rip]; exact hd8r10
  have hd9r8 : d9.reg 8 = some v8 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 8 = some v8
    rw [reg_with_rip]; exact hd8r8
  have hd9r9 : d9.reg 9 = some v9 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 9 = some v9
    rw [reg_with_rip]; exact hd8r9
  have hd9mem : d9.mem = [c1, c2, buf'] := by rw [hd9]; exact hd8mem
  have hd9k : d9.k = { (Kern.count s.k (cap - count) (f.length - count)).2 with
          fds := [some ⟨f, count + n⟩] } := by rw [hd9]; exact hd8k
  have hlen9 : d9.regs.length = 16 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact hlen8
  have hd9fl : d9.fl = d8.fl := by rw [hd9]
  -- step 10: jz loaded / fallthrough on n
  have hd9fl4 : X86Int.cond d9.fl 4 = some (n == 0) := by
    rw [hd9fl]
    exact cond_cc4 hd8zf
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok d1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hd1, run_zero]
  have r2 : run stage1Im 1 (Out.ok d1) = Out.ok d2 := by
    rw [run_succ_ok stage1Im 0 d1, e2, ← hd2, run_zero]
  have r3 : run stage1Im 1 (Out.ok d2) = Out.ok d3 := by
    rw [run_succ_ok stage1Im 0 d2, e3, ← hd3, run_zero]
  have r4 : run stage1Im 1 (Out.ok d3) = Out.ok d4 := by
    rw [run_succ_ok stage1Im 0 d3, e4, ← hd4, run_zero]
  have r5 : run stage1Im 1 (Out.ok d4) = Out.ok d5 := by
    rw [run_succ_ok stage1Im 0 d4, e5, ← hd5, run_zero]
  have r6 : run stage1Im 1 (Out.ok d5) = Out.ok d6 := by
    rw [run_succ_ok stage1Im 0 d5, e6, ← hd6, run_zero]
  have r7 : run stage1Im 1 (Out.ok d6) = Out.ok d7 := by
    rw [run_succ_ok stage1Im 0 d6, e7, ← hd7, run_zero]
  have r8 : run stage1Im 1 (Out.ok d7) = Out.ok d8 := by
    rw [run_succ_ok stage1Im 0 d7, e8, ← hd8, run_zero]
  have r9 : run stage1Im 1 (Out.ok d8) = Out.ok d9 := by
    rw [run_succ_ok stage1Im 0 d8, e9, ← hd9, run_zero]
  have hnohit9 : NoHitUpTo s 9 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
      j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.dd7] at h1
      simp at h1
    · have ht : t = d1 := by
        have h := r1.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd1rip, hc.dda] at h1
      simp at h1
    · have hr2 : run stage1Im 2 (Out.ok s) = Out.ok d2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r1, r2]
      have ht : t = d2 := by
        have h := hr2.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd2rip, hc.ddd] at h1
      simp at h1
    · have hr3 : run stage1Im 3 (Out.ok s) = Out.ok d3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r2, r3]
      have ht : t = d3 := by
        have h := hr3.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd3rip, hc.de0] at h1
      simp at h1
    · have hr4 : run stage1Im 4 (Out.ok s) = Out.ok d4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r3, r4]
      have ht : t = d4 := by
        have h := hr4.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd4rip, hc.de3] at h1
      simp at h1
    · have hr5 : run stage1Im 5 (Out.ok s) = Out.ok d5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
          r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r4, r5]
      have ht : t = d5 := by
        have h := hr5.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd5rip, hc.de5] at h1
      simp at h1
    · have hr6 : run stage1Im 6 (Out.ok s) = Out.ok d6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5,
          r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
          r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r5, r6]
      have ht : t = d6 := by
        have h := hr6.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      rcases hmem.2 with h | h <;> rw [hd6r0] at h <;> simp at h
    · have hr7 : run stage1Im 7 (Out.ok s) = Out.ok d7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6,
          r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5,
          r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
          r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r6, r7]
      have ht : t = d7 := by
        have h := hr7.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd7rip, hc.dea] at h1
      simp at h1
    · have hr8 : run stage1Im 8 (Out.ok s) = Out.ok d8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7,
          r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6,
          r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5,
          r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
          r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r7, r8]
      have ht : t = d8 := by
        have h := hr8.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd8rip, hc.ded] at h1
      simp at h1
    · have hr9 : run stage1Im 9 (Out.ok s) = Out.ok d9 := by
        rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8,
          r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7,
          r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6,
          r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5,
          r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4,
          r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3,
          r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r8, r9]
      have ht : t = d9 := by
        have h := hr9.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hd9rip, hc.def_] at h1
      simp at h1
  by_cases hn0 : n = 0
  · have hfl : X86Int.cond d9.fl 4 = some true := by
      rw [hd9fl4, hn0]
      decide
    have e10 : X86Int.step stage1Im d9 =
        .ok { d9 with rip := base + 0x122 } := by
      have h := step_jcc (im := stage1Im) (s := d9)
        (by rw [hd9rip]; exact hc.def_) hfl
      simpa using h
    obtain ⟨d10, hd10⟩ : ∃ t : X86Int.St, t =
        { d9 with rip := base + 0x122 } := ⟨_, rfl⟩
    have r10 : run stage1Im 1 (Out.ok d9) = Out.ok d10 := by
      rw [run_succ_ok stage1Im 0 d9, e10, ← hd10, run_zero]
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok d10 := by
      rw [show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r1,
        show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r2,
        show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r3,
        show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r4,
        show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r5,
        show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r6,
        show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r7,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r8,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r9, r10]
    have hfeq : f.length = count := by
      by_cases hlt : count < f.length
      · have hpos := hnpos hlt
        omega
      · omega
    refine ⟨n, d10, hncap, hnlen, hnpos, hrun,
      Or.inl ⟨by rw [hd10], hfeq⟩, ?_, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩, ?_, ?_, ?_⟩
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 0 = some n
      rw [reg_with_rip]; exact hd9r0
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 3 = some A
      rw [reg_with_rip]; exact hd9r3
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 12 = some cap
      rw [reg_with_rip]; exact hd9r12
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 13 = some count
      rw [reg_with_rip]; exact hd9r13
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 14 = some 3
      rw [reg_with_rip]; exact hd9r14
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hd9r4
    · refine ⟨v10, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 10 = some v10
      rw [reg_with_rip]; exact hd9r10
    · refine ⟨v8, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 8 = some v8
      rw [reg_with_rip]; exact hd9r8
    · refine ⟨v9, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 9 = some v9
      rw [reg_with_rip]; exact hd9r9
    · rw [hd10]
      exact hd9k
    · rw [hd10]
      exact hd9mem
    · rw [hd10]
      show Mem.rdN d9.mem A (count + n) = some (f.take (count + n))
      rw [hd9mem]
      exact hrdN'
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).regs.length = 16
      rw [len_with_rip]; exact hlen9
    · intro j t hj hr hmem
      rcases Nat.lt_or_ge j 10 with hj10 | hj10
      · exact hnohit9 j t (by omega) hr hmem
      · have hj10' : j = 10 := by omega
        subst hj10'
        have ht : t = d10 := by
          have h := hrun.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have hd10rip : d10.rip = base + 0x122 := by rw [hd10]
        have h1 := hmem.1
        rw [hd10rip, hc.d122] at h1
        simp at h1
  · have hfl : X86Int.cond d9.fl 4 = some false := by
      rw [hd9fl4]
      simp [hn0]
    have e10 : X86Int.step stage1Im d9 =
        .ok { d9 with rip := d9.rip + 2 } := by
      have h := step_jcc (im := stage1Im) (s := d9)
        (by rw [hd9rip]; exact hc.def_) hfl
      simpa using h
    obtain ⟨d10, hd10⟩ : ∃ t : X86Int.St, t =
        { d9 with rip := d9.rip + 2 } := ⟨_, rfl⟩
    have r10 : run stage1Im 1 (Out.ok d9) = Out.ok d10 := by
      rw [run_succ_ok stage1Im 0 d9, e10, ← hd10, run_zero]
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok d10 := by
      rw [show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r1,
        show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r2,
        show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r3,
        show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r4,
        show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r5,
        show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r6,
        show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r7,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r8,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r9, r10]
    have hflt : count < f.length := by omega
    refine ⟨n, d10, hncap, hnlen, hnpos, hrun, Or.inr ⟨by
        rw [hd10]; show d9.rip + 2 = base + 0xf1; rw [hd9rip], hflt⟩, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩,
      ?_, ?_, ?_⟩
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 0 = some n
      rw [reg_with_rip]; exact hd9r0
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 3 = some A
      rw [reg_with_rip]; exact hd9r3
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 12 = some cap
      rw [reg_with_rip]; exact hd9r12
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 13 = some count
      rw [reg_with_rip]; exact hd9r13
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 14 = some 3
      rw [reg_with_rip]; exact hd9r14
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hd9r4
    · refine ⟨v10, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 10 = some v10
      rw [reg_with_rip]; exact hd9r10
    · refine ⟨v8, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 8 = some v8
      rw [reg_with_rip]; exact hd9r8
    · refine ⟨v9, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 9 = some v9
      rw [reg_with_rip]; exact hd9r9
    · rw [hd10]
      exact hd9k
    · rw [hd10]
      exact hd9mem
    · rw [hd10]
      show Mem.rdN d9.mem A (count + n) = some (f.take (count + n))
      rw [hd9mem]
      exact hrdN'
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).regs.length = 16
      rw [len_with_rip]; exact hlen9
    · intro j t hj hr hmem
      rcases Nat.lt_or_ge j 10 with hj10 | hj10
      · exact hnohit9 j t (by omega) hr hmem
      · have hj10' : j = 10 := by omega
        subst hj10'
        have ht : t = d10 := by
          have h := hrun.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have hd10rip : d10.rip = base + 0xf1 := by
          rw [hd10]
          show d9.rip + 2 = base + 0xf1
          rw [hd9rip]
        have h1 := hmem.1
        rw [hd10rip, hc.df1] at h1
        simp at h1

theorem loader_count_update_nh {A sp0 cap count n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0xf1) (hlen16 : s.regs.length = 16)
    (hr0 : s.reg 0 = some n) (hr3 : s.reg 3 = some A)
    (hr12 : s.reg 12 = some cap) (hr13 : s.reg 13 = some count)
    (hr14 : s.reg 14 = some 3) (hr4 : s.reg 4 = some (sp0 + 24))
    (hr10 : ∃ v, s.reg 10 = some v) (hr8 : ∃ v, s.reg 8 = some v)
    (hr9 : ∃ v, s.reg 9 = some v)
    (hnle : count + n ≤ cap) (hcapW : cap < W) :
    ∃ s', run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      (s'.rip = base + 0xd7 ∧ count + n < cap ∨
        s'.rip = base + 0xfc ∧ count + n = cap) ∧
      s'.reg 13 = some (count + n) ∧ s'.reg 3 = some A ∧
      s'.reg 12 = some cap ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.mem = s.mem ∧ s'.k = s.k ∧ s'.regs.length = 16 ∧
      NoHitUpTo s 4 := by
  have hc := stage1_loaderCode
  have hadd : (aluF .add count n).1 = count + n := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  -- step 1: add r13, rax
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 13 (some (aluF .add count n).1) with
        fl := (aluF .add count n).2, rip := s.rip + 3 } :=
    step_alu_add (by rw [hrip]; exact hc.df1) hr13 hr0
  obtain ⟨u1, hu1⟩ : ∃ t : X86Int.St, t =
      { s.set 13 (some (aluF .add count n).1) with
        fl := (aluF .add count n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have hu1rip : u1.rip = base + 0xf4 := by
    rw [hu1]; show s.rip + 3 = base + 0xf4; rw [hrip]
  have hlen1 : u1.regs.length = 16 := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).regs.length = 16
    rw [len_set]; exact hlen16
  have hu1r13 : u1.reg 13 = some (count + n) := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 13 = some (count + n)
    rw [reg_set_self (by omega), hadd]
  have hu1r12 : u1.reg 12 = some cap := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 13)]; exact hr12
  have hu1r3 : u1.reg 3 = some A := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 13)]; exact hr3
  have hu1mem : u1.mem = s.mem := by rw [hu1]; rfl
  have hu1k : u1.k = s.k := by rw [hu1]; rfl
  -- step 2: mov rdx, r12
  have e2 : X86Int.step stage1Im u1 =
      .ok { u1.set 2 (some cap) with rip := u1.rip + 3 } :=
    step_mov (by rw [hu1rip]; exact hc.df4) hu1r12
  obtain ⟨u2, hu2⟩ : ∃ t : X86Int.St, t =
      { u1.set 2 (some cap) with rip := u1.rip + 3 } := ⟨_, rfl⟩
  have hu2rip : u2.rip = base + 0xf7 := by
    rw [hu2]; show u1.rip + 3 = base + 0xf7; rw [hu1rip]
  have hlen2 : u2.regs.length = 16 := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hu2r2 : u2.reg 2 = some cap := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hu2r13 : u2.reg 13 = some (count + n) := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 13 =
      some (count + n)
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    exact hu1r13
  have hu2r3 : u2.reg 3 = some A := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 2)]
    exact hu1r3
  have hu2r12 : u2.reg 12 = some cap := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 2)]
    exact hu1r12
  have hu2mem : u2.mem = s.mem := by rw [hu2]; exact hu1mem
  have hu2k : u2.k = s.k := by rw [hu2]; exact hu1k
  -- step 3: sub rdx, r13
  have hsub : (aluF .sub cap (count + n)).1 = cap - (count + n) := by
    rw [aluF_sub_fst,
      show cap + W - (count + n) = (cap - (count + n)) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have e3 : X86Int.step stage1Im u2 =
      .ok { u2.set 2 (some (aluF .sub cap (count + n)).1) with
        fl := (aluF .sub cap (count + n)).2, rip := u2.rip + 3 } :=
    step_alu_sub (by rw [hu2rip]; exact hc.df7) hu2r2 hu2r13
  obtain ⟨u3, hu3⟩ : ∃ t : X86Int.St, t =
      { u2.set 2 (some (aluF .sub cap (count + n)).1) with
        fl := (aluF .sub cap (count + n)).2, rip := u2.rip + 3 } :=
    ⟨_, rfl⟩
  have hu3rip : u3.rip = base + 0xfa := by
    rw [hu3]; show u2.rip + 3 = base + 0xfa; rw [hu2rip]
  have hlen3 : u3.regs.length = 16 := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).regs.length = 16
    rw [len_set]; exact hlen2
  have hu3zf : u3.fl.zf = some (cap - (count + n) == 0) := by
    rw [hu3]
    show (aluF .sub cap (count + n)).2.zf = some (cap - (count + n) == 0)
    rw [aluF_zf, hsub]
  have hu3r13 : u3.reg 13 = some (count + n) := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 13 =
      some (count + n)
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]; exact hu2r13
  have hu3r3 : u3.reg 3 = some A := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 2)]; exact hu2r3
  have hu3r12 : u3.reg 12 = some cap := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 2)]; exact hu2r12
  have hu3r14 : u3.reg 14 = some 3 := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 13)]; exact hr14
  have hu3r4 : u3.reg 4 = some (sp0 + 24) := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 4 =
      some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 13)]; exact hr4
  have hu3r0 : ∃ v, u3.reg 0 = some v := by
    refine ⟨n, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 13)]; exact hr0
  have hu3r10 : ∃ v, u3.reg 10 = some v := by
    obtain ⟨v, hv⟩ := hr10
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 10 = some v
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 10 = some v
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 10 = some v
    rw [reg_set_other (by decide : (10 : Nat) ≠ 13)]; exact hv
  have hu3r8 : ∃ v, u3.reg 8 = some v := by
    obtain ⟨v, hv⟩ := hr8
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 8 = some v
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 8 = some v
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 8 = some v
    rw [reg_set_other (by decide : (8 : Nat) ≠ 13)]; exact hv
  have hu3r9 : ∃ v, u3.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 9 = some v
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 9 = some v
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 9 = some v
    rw [reg_set_other (by decide : (9 : Nat) ≠ 13)]; exact hv
  have hu3mem : u3.mem = s.mem := by rw [hu3]; exact hu2mem
  have hu3k : u3.k = s.k := by rw [hu3]; exact hu2k
  -- step 4: jnz readloop / fall through to growth
  have hu3fl5 : X86Int.cond u3.fl 5 = some (!(cap - (count + n) == 0)) :=
    cond_cc5 hu3zf
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok u1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hu1, run_zero]
  have r2 : run stage1Im 1 (Out.ok u1) = Out.ok u2 := by
    rw [run_succ_ok stage1Im 0 u1, e2, ← hu2, run_zero]
  have r3 : run stage1Im 1 (Out.ok u2) = Out.ok u3 := by
    rw [run_succ_ok stage1Im 0 u2, e3, ← hu3, run_zero]
  have hnohit3 : NoHitUpTo s 3 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3) with
      rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.df1] at h1
      simp at h1
    · have ht : t = u1 := by
        have h := r1.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hu1rip, hc.df4] at h1
      simp at h1
    · have hr2 : run stage1Im 2 (Out.ok s) = Out.ok u2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r1, r2]
      have ht : t = u2 := by
        have h := hr2.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hu2rip, hc.df7] at h1
      simp at h1
    · have hr3 : run stage1Im 3 (Out.ok s) = Out.ok u3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r2, r3]
      have ht : t = u3 := by
        have h := hr3.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      subst ht
      have h1 := hmem.1
      rw [hu3rip, hc.dfa] at h1
      simp at h1
  by_cases hfull : count + n = cap
  · have hfl : X86Int.cond u3.fl 5 = some false := by
      rw [hu3fl5]
      simp [show cap - (count + n) = 0 by omega]
    have e4 : X86Int.step stage1Im u3 =
        .ok { u3 with rip := u3.rip + 2 } := by
      have h := step_jcc (im := stage1Im) (s := u3)
        (by rw [hu3rip]; exact hc.dfa) hfl
      simpa using h
    obtain ⟨u4, hu4⟩ : ∃ t : X86Int.St, t =
        { u3 with rip := u3.rip + 2 } := ⟨_, rfl⟩
    have r4 : run stage1Im 1 (Out.ok u3) = Out.ok u4 := by
      rw [run_succ_ok stage1Im 0 u3, e4, ← hu4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok u4 := by
      rw [show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r3, r4]
    refine ⟨u4, hrun, Or.inr ⟨by
        rw [hu4]; show u3.rip + 2 = base + 0xfc; rw [hu3rip], hfull⟩, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 13 = some (count + n)
      rw [reg_with_rip]; exact hu3r13
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 3 = some A
      rw [reg_with_rip]; exact hu3r3
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 12 = some cap
      rw [reg_with_rip]; exact hu3r12
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 14 = some 3
      rw [reg_with_rip]; exact hu3r14
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hu3r4
    · obtain ⟨v, hv⟩ := hu3r0
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 0 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r10
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 10 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r8
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 8 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r9
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 9 = some v
      rw [reg_with_rip]; exact hv
    · rw [hu4]; exact hu3mem
    · rw [hu4]; exact hu3k
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).regs.length = 16
      rw [len_with_rip]; exact hlen3
    · intro j t hj hr hmem
      rcases Nat.lt_or_ge j 4 with hj4 | hj4
      · exact hnohit3 j t (by omega) hr hmem
      · have hj4' : j = 4 := by omega
        subst hj4'
        have ht : t = u4 := by
          have h := hrun.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have hu4rip : u4.rip = base + 0xfc := by
          rw [hu4]
          show u3.rip + 2 = base + 0xfc
          rw [hu3rip]
        have h1 := hmem.1
        rw [hu4rip, hc.dfc] at h1
        simp at h1
  · have hfl : X86Int.cond u3.fl 5 = some true := by
      rw [hu3fl5]
      simp [show cap - (count + n) ≠ 0 by omega]
    have e4 : X86Int.step stage1Im u3 =
        .ok { u3 with rip := base + 0xd7 } := by
      have h := step_jcc (im := stage1Im) (s := u3)
        (by rw [hu3rip]; exact hc.dfa) hfl
      simpa using h
    obtain ⟨u4, hu4⟩ : ∃ t : X86Int.St, t =
        { u3 with rip := base + 0xd7 } := ⟨_, rfl⟩
    have r4 : run stage1Im 1 (Out.ok u3) = Out.ok u4 := by
      rw [run_succ_ok stage1Im 0 u3, e4, ← hu4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok u4 := by
      rw [show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r3, r4]
    refine ⟨u4, hrun, Or.inl ⟨by rw [hu4], by omega⟩, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 13 = some (count + n)
      rw [reg_with_rip]; exact hu3r13
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 3 = some A
      rw [reg_with_rip]; exact hu3r3
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 12 = some cap
      rw [reg_with_rip]; exact hu3r12
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 14 = some 3
      rw [reg_with_rip]; exact hu3r14
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hu3r4
    · obtain ⟨v, hv⟩ := hu3r0
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 0 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r10
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 10 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r8
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 8 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r9
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 9 = some v
      rw [reg_with_rip]; exact hv
    · rw [hu4]; exact hu3mem
    · rw [hu4]; exact hu3k
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).regs.length = 16
      rw [len_with_rip]; exact hlen3
    · intro j t hj hr hmem
      rcases Nat.lt_or_ge j 4 with hj4 | hj4
      · exact hnohit3 j t (by omega) hr hmem
      · have hj4' : j = 4 := by omega
        subst hj4'
        have ht : t = u4 := by
          have h := hrun.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have hu4rip : u4.rip = base + 0xd7 := by rw [hu4]
        have h1 := hmem.1
        rw [hu4rip, hc.dd7] at h1
        simp at h1

theorem loader_grow_any {f : List Nat} {A A' sp0 cap count : Nat}
    {c1 c2 buf : Region} {s : X86Int.St} {rest : List Choice}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hmem : s.mem = [c1, c2, buf]) (hbufbase : buf.base = A)
    (hbufrw : buf.rw = true) (hbufhi : buf.hi = A + cap)
    (hdisj1 : c1.hi ≤ A ∨ A + cap ≤ c1.base)
    (hdisj2 : c2.hi ≤ A ∨ A + cap ≤ c2.base)
    (hdata : buf.data.take count = f.take count)
    (hfd : s.k.fds = [some ⟨f, count⟩])
    (horacle : s.k.oracle = .at A' :: rest)
    (hcountcap : count ≤ cap)
    (hcapmod : cap % page = 0) (hcappos : 0 < cap)
    (hcapW : cap + cap < W) (hadm : OracleAdmits s) :
    ∃ s', run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0xd7 ∧
      s'.reg 3 = some A' ∧ s'.reg 12 = some (cap + cap) ∧
      s'.reg 13 = some count ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.k.oracle = rest ∧ s'.k.fds = [some ⟨f, count⟩] ∧
      (∃ buf', s'.mem = [c1, c2, buf'] ∧ buf'.base = A' ∧
        buf'.hi = A' + (cap + cap) ∧ buf'.rw = true ∧
        buf'.data.take count = f.take count) ∧
      s'.regs.length = 16 ∧ A' % page = 0 ∧ 0 < A' ∧
      A' + (cap + cap) ≤ 2 ^ 47 ∧
      (c1.hi ≤ A' ∨ A' + (cap + cap) ≤ c1.base) ∧
      (c2.hi ≤ A' ∨ A' + (cap + cap) ≤ c2.base) ∧
      NoHitUpTo s 12 := by
  have hc := stage1_loaderCode
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  have hdatalen : buf.data.length = cap := by
    have h := hbufhi
    unfold Region.hi at h
    omega
  have hpages : X86Env.pages cap = cap := pages_eq_of_mod_zero hcapmod
  have hpages2 : X86Env.pages (cap + cap) = cap + cap :=
    pages_eq_of_mod_zero (by simp [Nat.add_mod, hcapmod])
  have hfind : Mem.find [c1, c2, buf] A 0 = some (2, buf) := by
    have hd1 : c1.hi ≤ A ∨ A < c1.base := by
      rcases hdisj1 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    have hd2 : c2.hi ≤ A ∨ A < c2.base := by
      rcases hdisj2 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    exact find_third hd1 hd2 (by omega) (by rw [hbufhi]; omega)
  -- step 1: mov eax, 25
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (25 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.dfc)
  obtain ⟨g1, hg1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (25 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have hg1rip : g1.rip = base + 0x101 := by
    rw [hg1]; show s.rip + 5 = base + 0x101; rw [hrip]
  have hlen1 : g1.regs.length = 16 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen16
  have hg1r0 : g1.reg 0 = some 25 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg1r3 : g1.reg 3 = some A := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hr3
  have hg1r12 : g1.reg 12 = some cap := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hr12
  have hg1mem : g1.mem = s.mem := by rw [hg1]; rfl
  have hg1k : g1.k = s.k := by rw [hg1]; rfl
  -- step 2: mov rdi, rbx
  have e2 : X86Int.step stage1Im g1 =
      .ok { g1.set 7 (some A) with rip := g1.rip + 3 } :=
    step_mov (by rw [hg1rip]; exact hc.d101) hg1r3
  obtain ⟨g2, hg2⟩ : ∃ t : X86Int.St, t =
      { g1.set 7 (some A) with rip := g1.rip + 3 } := ⟨_, rfl⟩
  have hg2rip : g2.rip = base + 0x104 := by
    rw [hg2]; show g1.rip + 3 = base + 0x104; rw [hg1rip]
  have hlen2 : g2.regs.length = 16 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hg2r7 : g2.reg 7 = some A := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg2r0 : g2.reg 0 = some 25 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hg1r0
  have hg2r12 : g2.reg 12 = some cap := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    exact hg1r12
  have hg2mem : g2.mem = s.mem := by rw [hg2]; exact hg1mem
  have hg2k : g2.k = s.k := by rw [hg2]; exact hg1k
  -- step 3: mov rsi, r12
  have e3 : X86Int.step stage1Im g2 =
      .ok { g2.set 6 (some cap) with rip := g2.rip + 3 } :=
    step_mov (by rw [hg2rip]; exact hc.d104) hg2r12
  obtain ⟨g3, hg3⟩ : ∃ t : X86Int.St, t =
      { g2.set 6 (some cap) with rip := g2.rip + 3 } := ⟨_, rfl⟩
  have hg3rip : g3.rip = base + 0x107 := by
    rw [hg3]; show g2.rip + 3 = base + 0x107; rw [hg2rip]
  have hlen3 : g3.regs.length = 16 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen2
  have hg3r6 : g3.reg 6 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg3r7 : g3.reg 7 = some A := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact hg2r7
  have hg3r0 : g3.reg 0 = some 25 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hg2r0
  have hg3r12 : g3.reg 12 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hg2r12
  have hg3mem : g3.mem = s.mem := by rw [hg3]; exact hg2mem
  have hg3k : g3.k = s.k := by rw [hg3]; exact hg2k
  -- step 4: mov rdx, r12
  have e4 : X86Int.step stage1Im g3 =
      .ok { g3.set 2 (some cap) with rip := g3.rip + 3 } :=
    step_mov (by rw [hg3rip]; exact hc.d107) hg3r12
  obtain ⟨g4, hg4⟩ : ∃ t : X86Int.St, t =
      { g3.set 2 (some cap) with rip := g3.rip + 3 } := ⟨_, rfl⟩
  have hg4rip : g4.rip = base + 0x10a := by
    rw [hg4]; show g3.rip + 3 = base + 0x10a; rw [hg3rip]
  have hlen4 : g4.regs.length = 16 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen3
  have hg4r2 : g4.reg 2 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg4r6 : g4.reg 6 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hg3r6
  have hg4r7 : g4.reg 7 = some A := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact hg3r7
  have hg4r0 : g4.reg 0 = some 25 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact hg3r0
  have hg4mem : g4.mem = s.mem := by rw [hg4]; exact hg3mem
  have hg4k : g4.k = s.k := by rw [hg4]; exact hg3k
  -- step 5: add rdx, rdx
  have hadd : (aluF .add cap cap).1 = cap + cap := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hcapW]
  have e5 : X86Int.step stage1Im g4 =
      .ok { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } :=
    step_alu_add (by rw [hg4rip]; exact hc.d10a) hg4r2 hg4r2
  obtain ⟨g5, hg5⟩ : ∃ t : X86Int.St, t =
      { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } := ⟨_, rfl⟩
  have hg5rip : g5.rip = base + 0x10d := by
    rw [hg5]; show g4.rip + 3 = base + 0x10d; rw [hg4rip]
  have hlen5 : g5.regs.length = 16 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).regs.length = 16
    rw [len_set]; exact hlen4
  have hg5r2 : g5.reg 2 = some (cap + cap) := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 2 = some (cap + cap)
    rw [reg_set_self (by omega), hadd]
  have hg5r6 : g5.reg 6 = some cap := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 6 = some cap
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]; exact hg4r6
  have hg5r7 : g5.reg 7 = some A := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 7 = some A
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]; exact hg4r7
  have hg5r0 : g5.reg 0 = some 25 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 0 = some 25
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]; exact hg4r0
  have hg5mem : g5.mem = s.mem := by rw [hg5]; exact hg4mem
  have hg5k : g5.k = s.k := by rw [hg5]; exact hg4k
  -- step 6: mov r10, 1 (MREMAP_MAYMOVE)
  have e6 : X86Int.step stage1Im g5 =
      .ok { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } :=
    step_movI (by rw [hg5rip]; exact hc.d10d)
  obtain ⟨g6, hg6⟩ : ∃ t : X86Int.St, t =
      { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } := ⟨_, rfl⟩
  have hg6rip : g6.rip = base + 0x113 := by
    rw [hg6]; show g5.rip + 6 = base + 0x113; rw [hg5rip]
  have hlen6 : g6.regs.length = 16 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen5
  have hg6r10 : g6.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg6r0 : g6.reg 0 = some 25 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    exact hg5r0
  have hg6r7 : g6.reg 7 = some A := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 7 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    exact hg5r7
  have hg6r6 : g6.reg 6 = some cap := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 6 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    exact hg5r6
  have hg6r2 : g6.reg 2 = some (cap + cap) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    exact hg5r2
  have hg6mem : g6.mem = s.mem := by rw [hg6]; exact hg5mem
  have hg6k : g6.k = s.k := by rw [hg6]; exact hg5k
  have hg6r8 : g6.reg 8 = some v8 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8v
  have hg6r9 : g6.reg 9 = some v9 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9v
  have hg6r13 : g6.reg 13 = some count := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hr13
  have hg6r14 : g6.reg 14 = some 3 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  have hg6r4 : g6.reg 4 = some (sp0 + 24) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4
  -- legality at the syscall state gives the placement
  have q1 : run stage1Im 1 (Out.ok s) = Out.ok g1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hg1, run_zero]
  have q2 : run stage1Im 1 (Out.ok g1) = Out.ok g2 := by
    rw [run_succ_ok stage1Im 0 g1, e2, ← hg2, run_zero]
  have q3 : run stage1Im 1 (Out.ok g2) = Out.ok g3 := by
    rw [run_succ_ok stage1Im 0 g2, e3, ← hg3, run_zero]
  have q4 : run stage1Im 1 (Out.ok g3) = Out.ok g4 := by
    rw [run_succ_ok stage1Im 0 g3, e4, ← hg4, run_zero]
  have q5 : run stage1Im 1 (Out.ok g4) = Out.ok g5 := by
    rw [run_succ_ok stage1Im 0 g4, e5, ← hg5, run_zero]
  have q6 : run stage1Im 1 (Out.ok g5) = Out.ok g6 := by
    rw [run_succ_ok stage1Im 0 g5, e6, ← hg6, run_zero]
  have hqrun6 : run stage1Im 6 (Out.ok s) = Out.ok g6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, q1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, q2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, q3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, q4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, q5, q6]
  have hleg6 : OracleLegalAt g6 := hadm 6 g6 hqrun6
  have hmem6 : AtMemSyscall g6 :=
    ⟨by rw [hg6rip]; exact hc.d113, Or.inr hg6r0⟩
  have hfind6 : g6.mem.find A 0 = some (2, buf) := by
    rw [hg6mem]
    exact hmem.symm ▸ hfind
  have hplace : X86Env.place (g6.mem.eraseIdx 2) (some (.at A'))
        (X86Env.pages (cap + cap)) = some (some A') :=
    place_of_legal_mremap hleg6 hmem6 hg6r0 hg6r7 hg6r6 hg6r2
      hg6r10 hg6r8 hg6r9 hfind6 hbufbase
      (by rw [hdatalen, hpages]) hbufrw
      (by rw [hpages2, hdatalen]; omega)
      (by rw [hg6k]; exact horacle)
  have hplace2 : X86Env.place [c1, c2] (some (.at A'))
        (cap + cap) = some (some A') := by
    have h1 : g6.mem = [c1, c2, buf] := by
      rw [hg6mem]
      exact hmem
    rw [h1] at hplace
    rw [hpages2] at hplace
    exact hplace
  have hA'63 : A' < 2 ^ 63 := by
    have hf := place_at_facts hplace2
    omega
  have hpf := place_at_facts hplace2
  have hdisj1' : c1.hi ≤ A' ∨ A' + (cap + cap) ≤ c1.base := by
    rcases free_spec hpf.2.2.2 c1 (by simp) with h | h
    · exact Or.inr h
    · exact Or.inl h
  have hdisj2' : c2.hi ≤ A' ∨ A' + (cap + cap) ≤ c2.base := by
    rcases free_spec hpf.2.2.2 c2 (by simp) with h | h
    · exact Or.inr h
    · exact Or.inl h
  have hsc : X86Env.syscall s.k s.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE v8 v9 =
      some (.ret A' { s.k with oracle := rest }
        [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩]) := by
    rw [hmem]
    exact syscall_mremap_grow_any hfind hbufbase hdatalen hbufrw
      hpages hpages2 horacle hplace2
  have hsc6 : X86Env.syscall g6.k g6.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE v8 v9 =
      some (.ret A' { s.k with oracle := rest }
        [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩]) := by
    rw [hg6k, hg6mem]
    exact hsc
  -- step 7: mremap(buf, cap, 2 * cap, MREMAP_MAYMOVE)
  have e7 : X86Int.step stage1Im g6 =
      .ok { ((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11 none with
        k := { s.k with oracle := rest },
        mem := [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩],
        rip := g6.rip + 2 } :=
    step_syscall_ret (by rw [hg6rip]; exact hc.d113) hg6r0 hg6r7 hg6r6
      hg6r2 hg6r10 hg6r8 hg6r9 hsc6
  obtain ⟨g7, hg7⟩ : ∃ t : X86Int.St, t =
      { ((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11 none with
        k := { s.k with oracle := rest },
        mem := [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩],
        rip := g6.rip + 2 } := ⟨_, rfl⟩
  have hg7rip : g7.rip = base + 0x115 := by
    rw [hg7]; show g6.rip + 2 = base + 0x115; rw [hg6rip]
  have hlen7 : g7.regs.length = 16 := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact hlen6
  have hg7r0 : g7.reg 0 = some A' := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 0 = some A'
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hg7r2 : g7.reg 2 = some (cap + cap) := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 2 = some (cap + cap)
    rw [reg_set_other (by decide : (2 : Nat) ≠ 11),
      reg_set_other (by decide : (2 : Nat) ≠ 1),
      reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact hg6r2
  have hg7r13 : g7.reg 13 = some count := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
      reg_set_other (by decide : (13 : Nat) ≠ 1),
      reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hg6r13
  have hg7r14 : g7.reg 14 = some 3 := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hg6r14
  have hg7r4 : g7.reg 4 = some (sp0 + 24) := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hg6r4
  have hg7r10 : g7.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 10 = some MREMAP_MAYMOVE
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hg6r10
  have hg7r8 : g7.reg 8 = some v8 := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hg6r8
  have hg7r9 : g7.reg 9 = some v9 := by
    rw [hg7]
    show (((g6.set 0 (some A')).set 1 (some (g6.rip + 2))).set 11
      none).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hg6r9
  have hg7mem : g7.mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩] := by rw [hg7]
  have hg7k : g7.k = { s.k with oracle := rest } := by rw [hg7]
  -- step 8: test rax, rax
  have e8 : X86Int.step stage1Im g7 =
      .ok { g7 with fl := (aluF .and A' A').2, rip := g7.rip + 3 } :=
    step_test (by rw [hg7rip]; exact hc.d115) hg7r0 hg7r0
  obtain ⟨g8, hg8⟩ : ∃ t : X86Int.St, t =
      { g7 with fl := (aluF .and A' A').2, rip := g7.rip + 3 } := ⟨_, rfl⟩
  have hg8rip : g8.rip = base + 0x118 := by
    rw [hg8]; show g7.rip + 3 = base + 0x118; rw [hg7rip]
  have hg8sf : g8.fl.sf = some false := by
    rw [hg8]
    show (aluF .and A' A').2.sf = some false
    have hland : Nat.land A' A' = A' := by simp
    have hmsb : X86Int.msb A' = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt hA'63]
      decide
    simp [aluF, hland, hmsb]
  have hg8r0 : g8.reg 0 = some A' := by rw [hg8]; exact hg7r0
  have hg8r2 : g8.reg 2 = some (cap + cap) := by rw [hg8]; exact hg7r2
  have hg8r13 : g8.reg 13 = some count := by rw [hg8]; exact hg7r13
  have hg8r14 : g8.reg 14 = some 3 := by rw [hg8]; exact hg7r14
  have hg8r4 : g8.reg 4 = some (sp0 + 24) := by rw [hg8]; exact hg7r4
  have hg8r10 : g8.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg8]; exact hg7r10
  have hg8r8 : g8.reg 8 = some v8 := by rw [hg8]; exact hg7r8
  have hg8r9 : g8.reg 9 = some v9 := by rw [hg8]; exact hg7r9
  have hg8mem : g8.mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg8]; exact hg7mem
  have hg8k : g8.k = { s.k with oracle := rest } := by
    rw [hg8]; exact hg7k
  have hlen8 : g8.regs.length = 16 := by rw [hg8]; exact hlen7
  -- step 9: js fail2 (not taken: A ≥ 0)
  have hg8fl8 : X86Int.cond g8.fl 8 = some false :=
    cond_cc8_of_sf_false hg8sf
  have e9 : X86Int.step stage1Im g8 =
      .ok { g8 with rip := g8.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := g8)
      (by rw [hg8rip]; exact hc.d118) hg8fl8
    simpa using h
  obtain ⟨g9, hg9⟩ : ∃ t : X86Int.St, t =
      { g8 with rip := g8.rip + 2 } := ⟨_, rfl⟩
  have hg9rip : g9.rip = base + 0x11a := by
    rw [hg9]; show g8.rip + 2 = base + 0x11a; rw [hg8rip]
  have hg9r0 : g9.reg 0 = some A' := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 0 = some A'
    rw [reg_with_rip]; exact hg8r0
  have hg9r2 : g9.reg 2 = some (cap + cap) := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 2 = some (cap + cap)
    rw [reg_with_rip]; exact hg8r2
  have hg9r13 : g9.reg 13 = some count := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 13 = some count
    rw [reg_with_rip]; exact hg8r13
  have hg9r14 : g9.reg 14 = some 3 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip]; exact hg8r14
  have hg9r4 : g9.reg 4 = some (sp0 + 24) := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hg8r4
  have hg9r10 : g9.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip]; exact hg8r10
  have hg9r8 : g9.reg 8 = some v8 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 8 = some v8
    rw [reg_with_rip]; exact hg8r8
  have hg9r9 : g9.reg 9 = some v9 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 9 = some v9
    rw [reg_with_rip]; exact hg8r9
  have hg9mem : g9.mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg9]; exact hg8mem
  have hg9k : g9.k = { s.k with oracle := rest } := by
    rw [hg9]; exact hg8k
  have hlen9 : g9.regs.length = 16 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact hlen8
  -- step 10: mov rbx, rax
  have e10 : X86Int.step stage1Im g9 =
      .ok { g9.set 3 (some A') with rip := g9.rip + 3 } :=
    step_mov (by rw [hg9rip]; exact hc.d11a) hg9r0
  obtain ⟨g10, hg10⟩ : ∃ t : X86Int.St, t =
      { g9.set 3 (some A') with rip := g9.rip + 3 } := ⟨_, rfl⟩
  have hg10rip : g10.rip = base + 0x11d := by
    rw [hg10]; show g9.rip + 3 = base + 0x11d; rw [hg9rip]
  have hlen10 : g10.regs.length = 16 := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen9
  have hg10r3 : g10.reg 3 = some A' := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 3 = some A'
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg10r2 : g10.reg 2 = some (cap + cap) := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 3)]
    exact hg9r2
  have hg10r0 : g10.reg 0 = some A' := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 0 = some A'
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 3)]
    exact hg9r0
  have hg10r13 : g10.reg 13 = some count := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 3)]
    exact hg9r13
  have hg10r14 : g10.reg 14 = some 3 := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 3)]
    exact hg9r14
  have hg10r4 : g10.reg 4 = some (sp0 + 24) := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 3)]
    exact hg9r4
  have hg10r10 : g10.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 3)]
    exact hg9r10
  have hg10r8 : g10.reg 8 = some v8 := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 3)]
    exact hg9r8
  have hg10r9 : g10.reg 9 = some v9 := by
    rw [hg10]
    show ({ g9.set 3 (some A') with rip := g9.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 3)]
    exact hg9r9
  have hg10mem : g10.mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg10]; exact hg9mem
  have hg10k : g10.k = { s.k with oracle := rest } := by
    rw [hg10]; exact hg9k
  -- step 11: mov r12, rdx
  have e11 : X86Int.step stage1Im g10 =
      .ok { g10.set 12 (some (cap + cap)) with rip := g10.rip + 3 } :=
    step_mov (by rw [hg10rip]; exact hc.d11d) hg10r2
  obtain ⟨g11, hg11⟩ : ∃ t : X86Int.St, t =
      { g10.set 12 (some (cap + cap)) with rip := g10.rip + 3 } :=
    ⟨_, rfl⟩
  have hg11rip : g11.rip = base + 0x120 := by
    rw [hg11]; show g10.rip + 3 = base + 0x120; rw [hg10rip]
  have hlen11 : g11.regs.length = 16 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen10
  have hg11r12 : g11.reg 12 = some (cap + cap) := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 12 = some (cap + cap)
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg11r3 : g11.reg 3 = some A' := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 3 = some A'
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 12)]
    exact hg10r3
  have hg11r0 : g11.reg 0 = some A' := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 0 = some A'
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 12)]
    exact hg10r0
  have hg11r13 : g11.reg 13 = some count := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 12)]
    exact hg10r13
  have hg11r14 : g11.reg 14 = some 3 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 12)]
    exact hg10r14
  have hg11r4 : g11.reg 4 = some (sp0 + 24) := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 12)]
    exact hg10r4
  have hg11r10 : g11.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 12)]
    exact hg10r10
  have hg11r8 : g11.reg 8 = some v8 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 12)]
    exact hg10r8
  have hg11r9 : g11.reg 9 = some v9 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 12)]
    exact hg10r9
  have hg11mem : g11.mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg11]; exact hg10mem
  have hg11k : g11.k = { s.k with oracle := rest } := by
    rw [hg11]; exact hg10k
  -- step 12: jmp readloop
  have e12 : X86Int.step stage1Im g11 =
      .ok { g11 with rip := base + 0xd7 } :=
    step_jmp (by rw [hg11rip]; exact hc.d120)
  obtain ⟨g12, hg12⟩ : ∃ t : X86Int.St, t =
      { g11 with rip := base + 0xd7 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok g1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hg1, run_zero]
  have r2 : run stage1Im 1 (Out.ok g1) = Out.ok g2 := by
    rw [run_succ_ok stage1Im 0 g1, e2, ← hg2, run_zero]
  have r3 : run stage1Im 1 (Out.ok g2) = Out.ok g3 := by
    rw [run_succ_ok stage1Im 0 g2, e3, ← hg3, run_zero]
  have r4 : run stage1Im 1 (Out.ok g3) = Out.ok g4 := by
    rw [run_succ_ok stage1Im 0 g3, e4, ← hg4, run_zero]
  have r5 : run stage1Im 1 (Out.ok g4) = Out.ok g5 := by
    rw [run_succ_ok stage1Im 0 g4, e5, ← hg5, run_zero]
  have r6 : run stage1Im 1 (Out.ok g5) = Out.ok g6 := by
    rw [run_succ_ok stage1Im 0 g5, e6, ← hg6, run_zero]
  have r7 : run stage1Im 1 (Out.ok g6) = Out.ok g7 := by
    rw [run_succ_ok stage1Im 0 g6, e7, ← hg7, run_zero]
  have r8 : run stage1Im 1 (Out.ok g7) = Out.ok g8 := by
    rw [run_succ_ok stage1Im 0 g7, e8, ← hg8, run_zero]
  have r9 : run stage1Im 1 (Out.ok g8) = Out.ok g9 := by
    rw [run_succ_ok stage1Im 0 g8, e9, ← hg9, run_zero]
  have r10 : run stage1Im 1 (Out.ok g9) = Out.ok g10 := by
    rw [run_succ_ok stage1Im 0 g9, e10, ← hg10, run_zero]
  have r11 : run stage1Im 1 (Out.ok g10) = Out.ok g11 := by
    rw [run_succ_ok stage1Im 0 g10, e11, ← hg11, run_zero]
  have r12 : run stage1Im 1 (Out.ok g11) = Out.ok g12 := by
    rw [run_succ_ok stage1Im 0 g11, e12, ← hg12, run_zero]
  have hrun6 : run stage1Im 6 (Out.ok s) = Out.ok g6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r5, r6]
  have hrun : run stage1Im 12 (Out.ok s) = Out.ok g12 := by
    rw [show (12 : Nat) = 6 + 6 from rfl, run_add stage1Im 6 6, hrun6,
      show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r7,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r8,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r9,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r11, r12]
  have hhi' : (⟨A', buf.data ++ List.replicate cap 0, true⟩ :
      Region).hi = A' + (cap + cap) := by
    show A' + (buf.data ++ List.replicate cap 0).length = A' + (cap + cap)
    rw [List.length_append, List.length_replicate, hdatalen]
  have hdata' :
      (⟨A', buf.data ++ List.replicate cap 0, true⟩ :
        Region).data.take count = f.take count := by
    show (buf.data ++ List.replicate cap 0).take count =
      f.take count
    rw [List.take_append,
      show count - buf.data.length = 0 by omega, List.take_zero,
      List.append_nil]
    exact hdata
  have hnohit : NoHitUpTo s 12 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
      j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9 ∨ j = 10 ∨ j = 11 ∨
      j = 12) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.dfc] at h1
      simp at h1
    · have ht : t = g1 := by
        have h := (r1).symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg1rip, hc.d101] at h1
      simp at h1
    · have ht : t = g2 := by
        have hrj : run stage1Im 2 (Out.ok s) = Out.ok g2 := by
          rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg2rip, hc.d104] at h1
      simp at h1
    · have ht : t = g3 := by
        have hrj : run stage1Im 3 (Out.ok s) = Out.ok g3 := by
          rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg3rip, hc.d107] at h1
      simp at h1
    · have ht : t = g4 := by
        have hrj : run stage1Im 4 (Out.ok s) = Out.ok g4 := by
          rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg4rip, hc.d10a] at h1
      simp at h1
    · have ht : t = g5 := by
        have hrj : run stage1Im 5 (Out.ok s) = Out.ok g5 := by
          rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg5rip, hc.d10d] at h1
      simp at h1
    · have ht : t = g6 := by
        have h := hrun6.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      simp [hg6k, horacle]
    · have ht : t = g7 := by
        have hrj : run stage1Im 7 (Out.ok s) = Out.ok g7 := by
          rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg7rip, hc.d115] at h1
      simp at h1
    · have ht : t = g8 := by
        have hrj : run stage1Im 8 (Out.ok s) = Out.ok g8 := by
          rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg8rip, hc.d118] at h1
      simp at h1
    · have ht : t = g9 := by
        have hrj : run stage1Im 9 (Out.ok s) = Out.ok g9 := by
          rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg9rip, hc.d11a] at h1
      simp at h1
    · have ht : t = g10 := by
        have hrj : run stage1Im 10 (Out.ok s) = Out.ok g10 := by
          rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r8, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg10rip, hc.d11d] at h1
      simp at h1
    · have ht : t = g11 := by
        have hrj : run stage1Im 11 (Out.ok s) = Out.ok g11 := by
          rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have h1 := hmem.1
      rw [hg11rip, hc.d120] at h1
      simp at h1
    · have ht : t = g12 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem ⊢
      have hg12rip : g12.rip = base + 0xd7 := by rw [hg12]
      have h1 := hmem.1
      rw [hg12rip, hc.dd7] at h1
      simp at h1
  refine ⟨g12, hrun, by rw [hg12], ?_, ?_, ?_, ?_, ?_, ⟨A', ?_⟩,
    ⟨MREMAP_MAYMOVE, ?_⟩, ⟨v8, ?_⟩, ⟨v9, ?_⟩, ?_, ?_,
    ⟨⟨A', buf.data ++ List.replicate cap 0, true⟩, ?_, rfl, hhi',
      rfl, hdata'⟩,
    ?_, hpf.1, hpf.2.1, hpf.2.2.1, hdisj1', hdisj2', hnohit⟩
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 3 = some A'
    rw [reg_with_rip]; exact hg11r3
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 12 = some (cap + cap)
    rw [reg_with_rip]; exact hg11r12
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 13 = some count
    rw [reg_with_rip]; exact hg11r13
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 14 = some 3
    rw [reg_with_rip]; exact hg11r14
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hg11r4
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 0 = some A'
    rw [reg_with_rip]; exact hg11r0
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip]; exact hg11r10
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 8 = some v8
    rw [reg_with_rip]; exact hg11r8
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 9 = some v9
    rw [reg_with_rip]; exact hg11r9
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).k.oracle = rest
    rw [hg11k]
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).k.fds = [some ⟨f, count⟩]
    rw [hg11k]
    exact hfd
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).mem =
      [c1, c2, ⟨A', buf.data ++ List.replicate cap 0, true⟩]
    exact hg11mem
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).regs.length = 16
    rw [len_with_rip]; exact hlen11

/-- The first six steps of the growth block, to the `mremap`
syscall state at `base + 0x113` (prefix of `loader_grow_nomem`,
verbatim from `loader_grow`).  Used to apply legality
extraction at the site from loop-level admission. -/
theorem loader_grow_to_syscall {A sp0 cap count : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hcapW : cap + cap < W) :
    ∃ t, run stage1Im 6 (Out.ok s) = Out.ok t ∧
      t.rip = base + 0x113 ∧
      t.reg 0 = some 25 ∧ t.reg 7 = some A ∧
      t.reg 6 = some cap ∧ t.reg 2 = some (cap + cap) ∧
      t.reg 10 = some MREMAP_MAYMOVE ∧
      (∃ v, t.reg 8 = some v) ∧ (∃ v, t.reg 9 = some v) ∧
      t.reg 4 = some (sp0 + 24) ∧
      t.mem = s.mem ∧ t.k = s.k ∧ t.regs.length = 16 := by
  have hc := stage1_loaderCode
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  -- step 1: mov eax, 25
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (25 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.dfc)
  obtain ⟨g1, hg1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (25 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have hg1rip : g1.rip = base + 0x101 := by
    rw [hg1]; show s.rip + 5 = base + 0x101; rw [hrip]
  have hlen1 : g1.regs.length = 16 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen16
  have hg1r0 : g1.reg 0 = some 25 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg1r3 : g1.reg 3 = some A := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hr3
  have hg1r12 : g1.reg 12 = some cap := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hr12
  have hg1mem : g1.mem = s.mem := by rw [hg1]; rfl
  have hg1k : g1.k = s.k := by rw [hg1]; rfl
  -- step 2: mov rdi, rbx
  have e2 : X86Int.step stage1Im g1 =
      .ok { g1.set 7 (some A) with rip := g1.rip + 3 } :=
    step_mov (by rw [hg1rip]; exact hc.d101) hg1r3
  obtain ⟨g2, hg2⟩ : ∃ t : X86Int.St, t =
      { g1.set 7 (some A) with rip := g1.rip + 3 } := ⟨_, rfl⟩
  have hg2rip : g2.rip = base + 0x104 := by
    rw [hg2]; show g1.rip + 3 = base + 0x104; rw [hg1rip]
  have hlen2 : g2.regs.length = 16 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hg2r7 : g2.reg 7 = some A := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg2r0 : g2.reg 0 = some 25 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hg1r0
  have hg2r12 : g2.reg 12 = some cap := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    exact hg1r12
  have hg2mem : g2.mem = s.mem := by rw [hg2]; exact hg1mem
  have hg2k : g2.k = s.k := by rw [hg2]; exact hg1k
  -- step 3: mov rsi, r12
  have e3 : X86Int.step stage1Im g2 =
      .ok { g2.set 6 (some cap) with rip := g2.rip + 3 } :=
    step_mov (by rw [hg2rip]; exact hc.d104) hg2r12
  obtain ⟨g3, hg3⟩ : ∃ t : X86Int.St, t =
      { g2.set 6 (some cap) with rip := g2.rip + 3 } := ⟨_, rfl⟩
  have hg3rip : g3.rip = base + 0x107 := by
    rw [hg3]; show g2.rip + 3 = base + 0x107; rw [hg2rip]
  have hlen3 : g3.regs.length = 16 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen2
  have hg3r6 : g3.reg 6 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg3r7 : g3.reg 7 = some A := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact hg2r7
  have hg3r0 : g3.reg 0 = some 25 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hg2r0
  have hg3r12 : g3.reg 12 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hg2r12
  have hg3mem : g3.mem = s.mem := by rw [hg3]; exact hg2mem
  have hg3k : g3.k = s.k := by rw [hg3]; exact hg2k
  -- step 4: mov rdx, r12
  have e4 : X86Int.step stage1Im g3 =
      .ok { g3.set 2 (some cap) with rip := g3.rip + 3 } :=
    step_mov (by rw [hg3rip]; exact hc.d107) hg3r12
  obtain ⟨g4, hg4⟩ : ∃ t : X86Int.St, t =
      { g3.set 2 (some cap) with rip := g3.rip + 3 } := ⟨_, rfl⟩
  have hg4rip : g4.rip = base + 0x10a := by
    rw [hg4]; show g3.rip + 3 = base + 0x10a; rw [hg3rip]
  have hlen4 : g4.regs.length = 16 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen3
  have hg4r2 : g4.reg 2 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg4r6 : g4.reg 6 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hg3r6
  have hg4r7 : g4.reg 7 = some A := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact hg3r7
  have hg4r0 : g4.reg 0 = some 25 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact hg3r0
  have hg4mem : g4.mem = s.mem := by rw [hg4]; exact hg3mem
  have hg4k : g4.k = s.k := by rw [hg4]; exact hg3k
  -- step 5: add rdx, rdx
  have hadd : (aluF .add cap cap).1 = cap + cap := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hcapW]
  have e5 : X86Int.step stage1Im g4 =
      .ok { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } :=
    step_alu_add (by rw [hg4rip]; exact hc.d10a) hg4r2 hg4r2
  obtain ⟨g5, hg5⟩ : ∃ t : X86Int.St, t =
      { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } := ⟨_, rfl⟩
  have hg5rip : g5.rip = base + 0x10d := by
    rw [hg5]; show g4.rip + 3 = base + 0x10d; rw [hg4rip]
  have hlen5 : g5.regs.length = 16 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).regs.length = 16
    rw [len_set]; exact hlen4
  have hg5r2 : g5.reg 2 = some (cap + cap) := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 2 = some (cap + cap)
    rw [reg_set_self (by omega), hadd]
  have hg5r6 : g5.reg 6 = some cap := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 6 = some cap
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]; exact hg4r6
  have hg5r7 : g5.reg 7 = some A := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 7 = some A
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]; exact hg4r7
  have hg5r0 : g5.reg 0 = some 25 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 0 = some 25
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]; exact hg4r0
  have hg5mem : g5.mem = s.mem := by rw [hg5]; exact hg4mem
  have hg5k : g5.k = s.k := by rw [hg5]; exact hg4k
  -- step 6: mov r10, 1 (MREMAP_MAYMOVE)
  have e6 : X86Int.step stage1Im g5 =
      .ok { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } :=
    step_movI (by rw [hg5rip]; exact hc.d10d)
  obtain ⟨g6, hg6⟩ : ∃ t : X86Int.St, t =
      { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } := ⟨_, rfl⟩
  have hg6rip : g6.rip = base + 0x113 := by
    rw [hg6]; show g5.rip + 6 = base + 0x113; rw [hg5rip]
  have hlen6 : g6.regs.length = 16 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen5
  have hg6r10 : g6.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg6r0 : g6.reg 0 = some 25 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    exact hg5r0
  have hg6r7 : g6.reg 7 = some A := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 7 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    exact hg5r7
  have hg6r6 : g6.reg 6 = some cap := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 6 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    exact hg5r6
  have hg6r2 : g6.reg 2 = some (cap + cap) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    exact hg5r2
  have hg6mem : g6.mem = s.mem := by rw [hg6]; exact hg5mem
  have hg6k : g6.k = s.k := by rw [hg6]; exact hg5k
  have hg6r8 : g6.reg 8 = some v8 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8v
  have hg6r9 : g6.reg 9 = some v9 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9v
  have hg6r13 : g6.reg 13 = some count := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hr13
  have hg6r14 : g6.reg 14 = some 3 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  have hg6r4 : g6.reg 4 = some (sp0 + 24) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4

  have q1 : run stage1Im 1 (Out.ok s) = Out.ok g1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hg1, run_zero]
  have q2 : run stage1Im 1 (Out.ok g1) = Out.ok g2 := by
    rw [run_succ_ok stage1Im 0 g1, e2, ← hg2, run_zero]
  have q3 : run stage1Im 1 (Out.ok g2) = Out.ok g3 := by
    rw [run_succ_ok stage1Im 0 g2, e3, ← hg3, run_zero]
  have q4 : run stage1Im 1 (Out.ok g3) = Out.ok g4 := by
    rw [run_succ_ok stage1Im 0 g3, e4, ← hg4, run_zero]
  have q5 : run stage1Im 1 (Out.ok g4) = Out.ok g5 := by
    rw [run_succ_ok stage1Im 0 g4, e5, ← hg5, run_zero]
  have q6 : run stage1Im 1 (Out.ok g5) = Out.ok g6 := by
    rw [run_succ_ok stage1Im 0 g5, e6, ← hg6, run_zero]
  have hqrun6 : run stage1Im 6 (Out.ok s) = Out.ok g6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, q1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, q2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, q3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, q4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, q5, q6]
  exact ⟨g6, hqrun6, hg6rip, hg6r0, hg6r7, hg6r6, hg6r2, hg6r10,
    ⟨v8, hg6r8⟩, ⟨v9, hg6r9⟩, hg6r4, hg6mem, hg6k, hlen6⟩


/-! ### §3d. The read loop for arbitrary admitted oracles:
either a growth `mremap` is refused (exit 2 within bound) or
the file is loaded at `base + 0x134`, with buffer placements
varying per growth.  Strong induction on bytes remaining;
each read moves at least one byte (`kern_count_spec`), each
growth is followed by a progress read before the next growth,
so `1000 * (rem + 1)` bounds the steps crudely. -/
theorem loader_read_loop_any : ∀ rem : Nat,
    ∀ {f : List Nat} {A sp0 cap count : Nat}
      {c1 c2 buf : Region} {s : X86Int.St},
      f.length - count = rem →
      s.rip = base + 0xd7 → s.regs.length = 16 →
      s.reg 3 = some A → s.reg 12 = some cap →
      s.reg 13 = some count → s.reg 14 = some 3 →
      s.reg 4 = some (sp0 + 24) →
      (∃ v, s.reg 0 = some v) → (∃ v, s.reg 10 = some v) →
      (∃ v, s.reg 8 = some v) → (∃ v, s.reg 9 = some v) →
      s.mem = [c1, c2, buf] → buf.base = A → buf.rw = true →
      buf.hi = A + cap →
      (c1.hi ≤ A ∨ A + cap ≤ c1.base) →
      (c2.hi ≤ A ∨ A + cap ≤ c2.base) →
      buf.data.take count = f.take count →
      s.k.fds = [some ⟨f, count⟩] →
      (∀ b ∈ f, b < 256) →
      count ≤ f.length → count < cap →
      A % page = 0 → 0 < A → A + cap ≤ 2 ^ 47 →
      cap % page = 0 → 65536 ≤ cap →
      OracleAdmits s → RegsOk s →
      c1.hi ≤ c2.base → c2.rw = true →
      c2.base + 8 ≤ sp0 + 24 → sp0 + 24 ≤ c2.hi →
      32 ≤ sp0 + 24 → sp0 + 24 < W →
      (∃ k, k ≤ 1000 * (rem + 1) ∧
        run stage1Im k (Out.ok s) = Out.exit 2) ∨
      (∃ k A' cap' buf' s', k ≤ 1000 * (rem + 1) ∧
        run stage1Im k (Out.ok s) = Out.ok s' ∧
        s'.rip = base + 0x134 ∧
        s'.reg 3 = some A' ∧ s'.reg 12 = some cap' ∧
        s'.reg 13 = some f.length ∧ s'.reg 14 = some 3 ∧
        s'.reg 4 = some (sp0 + 24) ∧
        (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
        (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
        s'.mem = [c1, c2, buf'] ∧ buf'.base = A' ∧
        buf'.hi = A' + cap' ∧ buf'.rw = true ∧
        buf'.data.take f.length = f ∧
        (c1.hi ≤ A' ∨ A' + cap' ≤ c1.base) ∧
        (c2.hi ≤ A' ∨ A' + cap' ≤ c2.base) ∧
        A' % page = 0 ∧ 0 < A' ∧ A' + cap' ≤ 2 ^ 47 ∧
        cap' % page = 0 ∧ 65536 ≤ cap' ∧
        s'.k.fds = [some ⟨f, f.length⟩] ∧
        s'.regs.length = 16 ∧ NoHitUpTo s k) := by
  intro rem
  induction rem using Nat.strongRecOn with
  | _ rem ih =>
  intro f A sp0 cap count c1 c2 buf s hrem hrip hlen16 hr3 hr12
    hr13 hr14 hr4 hr0 hr10 hr8 hr9 hmem hbufbase hbufrw hbufhi
    hdisj1 hdisj2 hdata hfd hfB hcount hcountcap hAmod hApos
    hAbound hcapmod hcapge hadm hregs hdisjc hc2rw hc2lo hc2hi
    hsp32 hspW
  have hc := stage1_loaderCode
  have hW : W = 2 ^ 64 := rfl
  have hbuflen : buf.data.length = cap := by
    have h := hbufhi
    unfold Region.hi at h
    omega
  have hcontent : s.mem.rdN A count = some (f.take count) := by
    rw [hmem]
    have h := rdN_third_take hdisj1 hdisj2 hbufbase
      (by omega : count ≤ buf.data.length) (by omega)
    rwa [hdata] at h
  obtain ⟨n, s1, hncap, hnlen, hnpos, hrun1, hrip1, hs1r0, hs1r3,
    hs1r12, hs1r13, hs1r14, hs1r4, hs1r10, hs1r8, hs1r9, hs1k,
    ⟨buf1, hs1mem, hbuf1b, hbuf1hi, hbuf1rw⟩, hs1rdN, hs1len,
    hs1nh⟩ :=
    loader_read_once_any hrip hlen16 hr3 hr12 hr13 hr14 hr4 hr0
      hr10 hr8 hr9 hmem hbufbase hbufrw hbufhi hdisj1 hdisj2
      (Nat.le_refl cap) hcontent hfd hfB hcount hcountcap
      (by omega) (by omega)
  have hbuf1len : buf1.data.length = cap := by
    have h := hbuf1hi
    unfold Region.hi at h
    omega
  have hdata1 : buf1.data.take (count + n) =
      f.take (count + n) := by
    have h2 := rdN_third_take (r := buf1) hdisj1 hdisj2 hbuf1b
      (by omega : count + n ≤ buf1.data.length) (by omega)
    rw [hs1mem] at hs1rdN
    exact Option.some.inj (h2.symm.trans hs1rdN)
  have hs1fds : s1.k.fds = [some ⟨f, count + n⟩] := by
    have h := congrArg X86Env.Kern.fds hs1k
    exact h
  rcases hrip1 with ⟨hs1rip, hfeq⟩ | ⟨hs1rip, hflt⟩
  · -- EOF: one jump to loaded (0x134)
    have hn0 : n = 0 := by omega
    have e11 : X86Int.step stage1Im s1 =
        .ok { s1 with rip := base + 0x134 } := by
      have h := step_jmp (im := stage1Im) (s := s1)
        (by rw [hs1rip]; exact hc.d122)
      simpa using h
    obtain ⟨s2, hs2⟩ : ∃ t : X86Int.St, t =
        { s1 with rip := base + 0x134 } := ⟨_, rfl⟩
    have r11 : run stage1Im 1 (Out.ok s1) = Out.ok s2 := by
      rw [run_succ_ok stage1Im 0 s1, e11, ← hs2, run_zero]
    have hrun11 : run stage1Im 11 (Out.ok s) = Out.ok s2 := by
      rw [show (11 : Nat) = 10 + 1 by omega, run_add stage1Im 10 1,
        hrun1, r11]
    have hnh11 : NoHitUpTo s 11 :=
      noHitUpTo_add hs1nh hrun1 (show NoHitUpTo s1 1 from by
        intro j t hj hr hmem
        rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl
        · rw [run_zero] at hr
          simp only [Out.ok.injEq] at hr
          subst hr
          have h1 := hmem.1
          rw [hs1rip, hc.d122] at h1
          simp at h1
        · have ht : t = s2 := by
            have h := r11.symm.trans hr
            simp only [Out.ok.injEq] at h
            exact h.symm
          rw [ht] at hmem
          have hs2rip : s2.rip = base + 0x134 := by rw [hs2]
          have h1 := hmem.1
          rw [hs2rip, stage1_parserCode.p134] at h1
          simp at h1)
    have hdataF : buf1.data.take f.length = f := by
      have h := hdata1
      rw [show count + n = f.length by omega] at h
      rwa [List.take_length] at h
    refine Or.inr ⟨11, A, cap, buf1, s2, by omega, hrun11,
      by rw [hs2], ?_, ?_, ?_, ?_, ?_,
      ⟨n, by
        rw [hs2]
        show ({ s1 with rip := base + 0x134 }).reg 0 = some n
        rw [reg_with_rip]; exact hs1r0⟩,
      (by
        obtain ⟨v, hv⟩ := hs1r10
        exact ⟨v, by
          rw [hs2]
          show ({ s1 with rip := base + 0x134 }).reg 10 = some v
          rw [reg_with_rip]; exact hv⟩),
      (by
        obtain ⟨v, hv⟩ := hs1r8
        exact ⟨v, by
          rw [hs2]
          show ({ s1 with rip := base + 0x134 }).reg 8 = some v
          rw [reg_with_rip]; exact hv⟩),
      (by
        obtain ⟨v, hv⟩ := hs1r9
        exact ⟨v, by
          rw [hs2]
          show ({ s1 with rip := base + 0x134 }).reg 9 = some v
          rw [reg_with_rip]; exact hv⟩),
      by rw [hs2]; exact hs1mem, hbuf1b, hbuf1hi, hbuf1rw,
      hdataF, hdisj1, hdisj2, hAmod, hApos, hAbound, hcapmod,
      hcapge, ?_, by rw [hs2]; exact hs1len, hnh11⟩
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 3 = some A
      rw [reg_with_rip]; exact hs1r3
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 12 = some cap
      rw [reg_with_rip]; exact hs1r12
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 13 =
        some f.length
      rw [reg_with_rip]
      rw [hfeq]; exact hs1r13
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 14 = some 3
      rw [reg_with_rip]; exact hs1r14
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 4 =
        some (sp0 + 24)
      rw [reg_with_rip]; exact hs1r4
    · rw [hs2]
      have h := hs1fds
      rwa [show count + n = f.length by omega] at h
  · -- positive read: count update, then loop or grow
    have hn0 : 0 < n := hnpos hflt
    obtain ⟨s2, hrun2, hrip2, hs2r13, hs2r3, hs2r12, hs2r14,
      hs2r4, hs2r0, hs2r10, hs2r8, hs2r9, hs2mem, hs2k, hs2len,
      hs2nh⟩ :=
      loader_count_update_nh hs1rip hs1len hs1r0 hs1r3 hs1r12
        hs1r13 hs1r14 hs1r4 hs1r10 hs1r8 hs1r9 (by omega)
        (by omega)
    have hrun14 : run stage1Im 14 (Out.ok s) = Out.ok s2 := by
      rw [show (14 : Nat) = 10 + 4 by omega, run_add stage1Im 10 4,
        hrun1, hrun2]
    have hnh14 : NoHitUpTo s 14 :=
      noHitUpTo_add hs1nh hrun1 hs2nh
    have hadm2 : OracleAdmits s2 :=
      oracleAdmits_of_run hadm hrun14
    have hregs2 : RegsOk s2 := run_regs_ok hrun14 hregs
    have hs2mem1 : s2.mem = [c1, c2, buf1] := hs2mem.trans hs1mem
    have hs2fds : s2.k.fds = [some ⟨f, count + n⟩] := by
      rw [hs2k]
      exact hs1fds
    have hroom2 : StackRoom s2 1 :=
      stackRoom_of_c2 hs2mem1 hs2r4 hc2rw hdisjc hc2lo hc2hi hspW
    rcases hrip2 with ⟨hs2rip, hltcap⟩ | ⟨hs2rip, hfull⟩
    · -- back to loop head, fewer bytes remaining
      rcases ih (f.length - (count + n)) (by omega) rfl hs2rip
          hs2len hs2r3 hs2r12 hs2r13 hs2r14 hs2r4 hs2r0 hs2r10
          hs2r8 hs2r9 hs2mem1 hbuf1b hbuf1rw hbuf1hi hdisj1
          hdisj2 hdata1 hs2fds hfB (by omega) hltcap hAmod hApos
          hAbound hcapmod hcapge hadm2 hregs2 hdisjc hc2rw hc2lo
          hc2hi hsp32 hspW with hex | hload
      · obtain ⟨k', hk'b, hrk'⟩ := hex
        exact Or.inl ⟨14 + k', by omega, by
          rw [run_add stage1Im 14 k', hrun14, hrk']⟩
      · obtain ⟨k', A', cap', buf', s', hk'b, hrk', hs'rip, hs'r3,
          hs'r12, hs'r13, hs'r14, hs'r4, hs'r0, hs'r10, hs'r8,
          hs'r9, hs'mem, hbuf'b, hbuf'hi, hbuf'rw, hbuf'data,
          hdisj1', hdisj2', hA'mod, hA'pos, hA'bound, hcap'mod,
          hcap'ge, hs'fds, hs'len, hs'nh⟩ := hload
        exact Or.inr ⟨14 + k', A', cap', buf', s', by omega, by
          rw [run_add stage1Im 14 k', hrun14, hrk'], hs'rip,
          hs'r3, hs'r12, hs'r13, hs'r14, hs'r4, hs'r0, hs'r10,
          hs'r8, hs'r9, hs'mem, hbuf'b, hbuf'hi, hbuf'rw,
          hbuf'data, hdisj1', hdisj2', hA'mod, hA'pos, hA'bound,
          hcap'mod, hcap'ge, hs'fds, hs'len,
          noHitUpTo_add hnh14 hrun14 hs'nh⟩
    · -- buffer full: growth block at 0xfc
      have hfind2 : s2.mem.find A 0 = some (2, buf1) := by
        rw [hs2mem1]
        refine find_third ?_ ?_ (by omega) ?_
        · rcases hdisj1 with h | h
          · exact Or.inl h
          · exact Or.inr (by omega)
        · rcases hdisj2 with h | h
          · exact Or.inl h
          · exact Or.inr (by omega)
        · rw [hbuf1hi]
          omega
      have hlenpg : buf1.data.length = X86Env.pages cap := by
        rw [hbuf1len, pages_eq_of_mod_zero hcapmod]
      have hshrink2 : ¬ X86Env.pages (cap + cap) <
          buf1.data.length := by
        rw [pages_eq_of_mod_zero
          (by simp [Nat.add_mod, hcapmod]), hbuf1len]
        omega
      cases hs2or : s2.k.oracle with
      | nil =>
        exfalso
        obtain ⟨g6, hg6run, hg6rip, hg6r0, hg6r7, hg6r6, hg6r2,
          hg6r10, hg6r8, hg6r9, -, hg6mem, hg6k, -⟩ :=
          loader_grow_to_syscall hs2rip hs2len hs2r3 hs2r12 hs2r13
            hs2r14 hs2r4 hs2r8 hs2r9 (by omega)
        have hlegal6 : OracleLegalAt g6 :=
          (oracleAdmits_of_run hadm2 hg6run) 0 g6 rfl
        have hmem6 : AtMemSyscall g6 :=
          ⟨by rw [hg6rip]; exact hc.d113, Or.inr hg6r0⟩
        have hfind6 : g6.mem.find A 0 = some (2, buf1) := by
          rw [hg6mem]
          exact hfind2
        obtain ⟨v8, hv8⟩ := hg6r8
        obtain ⟨v9, hv9⟩ := hg6r9
        rcases head_cases_of_legal_mremap hlegal6 hmem6 hg6r0
            hg6r7 hg6r6 hg6r2 hg6r10 hv8 hv9 hfind6 hbuf1b
            hlenpg hbuf1rw hshrink2
            with ⟨A', rest, hor⟩ | hnomem
        · have h1 : g6.k.oracle = [] := by
            rw [hg6k]
            exact hs2or
          rw [h1] at hor
          simp at hor
        · have h1 : g6.k.oracle.head? = none := by
            have h2 : g6.k.oracle = [] := by
              rw [hg6k]
              exact hs2or
            rw [h2]
            rfl
          rw [h1] at hnomem
          simp at hnomem
      | cons c rest =>
        cases c
        · rename_i A'
          obtain ⟨s3, hrun3, hs3rip, hs3r3, hs3r12, hs3r13,
            hs3r14, hs3r4, hs3r0, hs3r10, hs3r8, hs3r9, hs3or,
            hs3fds, ⟨buf2, hs3mem, hbuf2b, hbuf2hi, hbuf2rw,
            hbuf2data⟩, hs3len, hA'mod, hA'pos, hA'bound,
            hdisj1', hdisj2', hs3nh⟩ :=
            loader_grow_any hs2rip hs2len hs2r3 hs2r12 hs2r13
              hs2r14 hs2r4 hs2r8 hs2r9 hs2mem1 hbuf1b hbuf1rw
              hbuf1hi hdisj1 hdisj2 hdata1 hs2fds hs2or
              (by omega) hcapmod (by omega) (by omega) hadm2
          have hrun26 : run stage1Im 26 (Out.ok s) = Out.ok s3 := by
            rw [show (26 : Nat) = 14 + 12 by omega,
              run_add stage1Im 14 12, hrun14, hrun3]
          have hnh26 : NoHitUpTo s 26 :=
            noHitUpTo_add hnh14 hrun14 hs3nh
          have hadm3 : OracleAdmits s3 :=
            oracleAdmits_of_run hadm hrun26
          have hregs3 : RegsOk s3 := run_regs_ok hrun26 hregs
          rcases ih (f.length - (count + n)) (by omega) rfl
              hs3rip hs3len hs3r3 hs3r12 hs3r13 hs3r14 hs3r4
              hs3r0 hs3r10 hs3r8 hs3r9 hs3mem hbuf2b hbuf2rw
              hbuf2hi hdisj1' hdisj2' hbuf2data hs3fds hfB
              (by omega) (by omega) hA'mod hA'pos hA'bound
              (by simp [Nat.add_mod, hcapmod]) (by omega) hadm3
              hregs3 hdisjc hc2rw hc2lo hc2hi hsp32 hspW with
            hex | hload
          · obtain ⟨k', hk'b, hrk'⟩ := hex
            exact Or.inl ⟨26 + k', by omega, by
              rw [run_add stage1Im 26 k', hrun26, hrk']⟩
          · obtain ⟨k', A'', cap'', buf'', s', hk'b, hrk', hs'rip,
              hs'r3, hs'r12, hs'r13, hs'r14, hs'r4, hs'r0, hs'r10,
              hs'r8, hs'r9, hs'mem, hbuf'b, hbuf'hi, hbuf'rw,
              hbuf'data, hdisj1'', hdisj2'', hA''mod, hA''pos,
              hA''bound, hcap''mod, hcap''ge, hs'fds, hs'len,
              hs'nh⟩ := hload
            exact Or.inr ⟨26 + k', A'', cap'', buf'', s',
              by omega, by
              rw [run_add stage1Im 26 k', hrun26, hrk'], hs'rip,
              hs'r3, hs'r12, hs'r13, hs'r14, hs'r4, hs'r0, hs'r10,
              hs'r8, hs'r9, hs'mem, hbuf'b, hbuf'hi, hbuf'rw,
              hbuf'data, hdisj1'', hdisj2'', hA''mod, hA''pos,
              hA''bound, hcap''mod, hcap''ge, hs'fds, hs'len,
              noHitUpTo_add hnh26 hrun26 hs'nh⟩
        · -- nomem at the growth mremap: exit 2 in 14 more steps
          have hhead2 : s2.k.oracle.head? = some .nomem := by
            rw [hs2or]
            rfl
          have hex := loader_grow_nomem_exit2_bnd hs2rip hs2len
            hs2r3 hs2r12 hs2r13 hs2r14 hs2r4 hs2r8 hs2r9 hfind2
            hbuf1b hbuf1len hbuf1rw hcapmod (by omega) hhead2
            hregs2 hroom2 hsp32 hspW
          exact Or.inl ⟨28, by omega, by
            rw [show (28 : Nat) = 14 + 14 by omega,
              run_add stage1Im 14 14, hrun14, hex]⟩
        · rename_i cn
          exfalso
          obtain ⟨g6, hg6run, hg6rip, hg6r0, hg6r7, hg6r6, hg6r2,
            hg6r10, hg6r8, hg6r9, -, hg6mem, hg6k, -⟩ :=
            loader_grow_to_syscall hs2rip hs2len hs2r3 hs2r12
              hs2r13 hs2r14 hs2r4 hs2r8 hs2r9 (by omega)
          have hlegal6 : OracleLegalAt g6 :=
            (oracleAdmits_of_run hadm2 hg6run) 0 g6 rfl
          have hmem6 : AtMemSyscall g6 :=
            ⟨by rw [hg6rip]; exact hc.d113, Or.inr hg6r0⟩
          have hfind6 : g6.mem.find A 0 = some (2, buf1) := by
            rw [hg6mem]
            exact hfind2
          obtain ⟨v8, hv8⟩ := hg6r8
          obtain ⟨v9, hv9⟩ := hg6r9
          rcases head_cases_of_legal_mremap hlegal6 hmem6 hg6r0
              hg6r7 hg6r6 hg6r2 hg6r10 hv8 hv9 hfind6 hbuf1b
              hlenpg hbuf1rw hshrink2
              with ⟨A', rest', hor⟩ | hnomem
          · have h1 : g6.k.oracle = .count cn :: rest := by
              rw [hg6k]
              exact hs2or
            rw [h1] at hor
            simp at hor
          · have h1 : g6.k.oracle.head? = some (.count cn) := by
              have h2 : g6.k.oracle = .count cn :: rest := by
                rw [hg6k]
                exact hs2or
              rw [h2]
              rfl
            rw [h1] at hnomem
            simp at hnomem

theorem loader_entry_nh {f path argv0 : List Nat} {oracle : List Choice}
    {top sp0 : Nat} {stdinL : List Nat} {stk : Region} {s : X86Int.St}
    (hstk : stk = (X86Env.initStack top [argv0, path]).1)
    (hsp0 : sp0 = (X86Env.initStack top [argv0, path]).2)
    (hrip : s.rip = base + 0x78) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp0)
    (hr0 : ∀ i, i ≠ 4 → i < 16 → s.reg i = some 0)
    (hmem : s.mem = [⟨base, stage1Code, false⟩, stk])
    (hk : s.k = ⟨[(path, f)], [], stdinL, [], oracle⟩)
    (hdisj : base + stage1Code.length ≤ stk.base)
    (hstkhi : stk.hi = top) (htop47 : top ≤ 2 ^ 47)
    (hpath0 : 0 ∉ path) (hpathlen : path.length < 4096) :
    ∃ s', run stage1Im 21 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0xbf ∧
      s'.reg 0 = some 9 ∧ s'.reg 7 = some 0 ∧ s'.reg 6 = some 65536 ∧
      s'.reg 2 = some 3 ∧ s'.reg 10 = some 34 ∧
      s'.reg 8 = some (W - 1) ∧ s'.reg 9 = some 0 ∧
      s'.reg 14 = some 3 ∧ s'.reg 4 = some (sp0 + 24) ∧
      s'.k = ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ ∧
      wshape s'.mem = wshape s.mem ∧ s'.regs.length = 16 ∧
      (∃ stk', s'.mem = [⟨base, stage1Code, false⟩, stk']) ∧
      ∀ j t, j ≤ 21 → run stage1Im j (Out.ok s) = Out.ok t →
        AtMemSyscall t → j = 21 ∧ t = s' := by
  have hc := stage1_loaderCode
  have hSR : X86Env.stackRoom = 4096 := rfl
  have hsp0eq : sp0 = stk.base + X86Env.stackRoom := by
    rw [hsp0, hstk]
    exact initStack_sp top argv0 path
  have hdatalen : stk.data.length =
      X86Env.stackRoom +
        X86Env.pages (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
    rw [hstk]
    exact initStack_data_length top argv0 path
  have hsp0le : sp0 ≤ top := by
    have h1 : stk.base + X86Env.stackRoom ≤ stk.hi := by
      unfold Region.hi
      omega
    omega
  have hspW : sp0 + 24 < W := by
    unfold W
    omega
  have hdisj' : (⟨base, stage1Code, false⟩ : Region).hi ≤ stk.base := by
    simp only [Region.hi]
    exact hdisj
  -- the path pointer and its bound
  have hP : sp0 + 56 + argv0.length + 1 < W := by
    have hb := initStack_rd_path (c := ⟨base, stage1Code, false⟩)
      (top := top) (x := argv0) (y := path) (by rw [← hstk]; exact hdisj')
      (i := 0) (by omega)
    rw [← hstk, ← hsp0] at hb
    obtain ⟨v, hv⟩ : ∃ v, Mem.rd [⟨base, stage1Code, false⟩, stk]
        (sp0 + 56 + argv0.length + 1 + 0) = some v := by
      cases hLi : (path ++ [0])[0]? with
      | some v => exact ⟨v, by rwa [hLi] at hb⟩
      | none =>
        cases path with
        | nil => simp at hLi
        | cons b t => simp at hLi
    obtain ⟨r, hr, -, hlt⟩ := rd_some_region hv
    have hr2 : r = ⟨base, stage1Code, false⟩ ∨ r = stk := by simpa using hr
    have hrhi : r.hi ≤ top := by
      rcases hr2 with rfl | rfl
      · have h1 : stk.base ≤ stk.hi := by
          unfold Region.hi
          omega
        omega
      · omega
    unfold W
    omega
  have hPmod : (sp0 + 56 + argv0.length + 1) % W =
      sp0 + 56 + argv0.length + 1 := Nat.mod_eq_of_lt hP
  have hA0mod : (sp0 + 56) % W = sp0 + 56 :=
    Nat.mod_eq_of_lt (by omega)
  -- step 1: pop rax (argc = 2)
  have hrd1 : rd64 s.mem sp0 = some 2 := by
    have h := initStack_rd64_argc (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0] at h
    rwa [hmem]
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.d78) hr4 hrd1
  obtain ⟨a1, ha1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = base + 0x79 := by
    rw [ha1]; show s.rip + 1 = base + 0x79; rw [hrip]
  have hlen1 : a1.regs.length = 16 := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen16
  have ha1r0 : a1.reg 0 = some 2 := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_self (by
      show 0 < (s.set 4 (some ((sp0 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha1r4 : a1.reg 4 = some (sp0 + 8) := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 4 = some (sp0 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha1mem : a1.mem = s.mem := by rw [ha1]; rfl
  have ha1k : a1.k = s.k := by rw [ha1]; rfl
  -- step 2: cmp rax, 2
  have e2 : X86Int.step stage1Im a1 =
      .ok { a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 } :=
    step_cmpI (by rw [ha1rip]; exact hc.d79) ha1r0
  obtain ⟨a2, ha2⟩ : ∃ t : X86Int.St, t =
      { a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = base + 0x7d := by
    rw [ha2]; show a1.rip + 4 = base + 0x7d; rw [ha1rip]
  have ha2fl : X86Int.cond a2.fl 13 = some true := by
    have h2W : (2 : Nat) % W = 2 := by decide
    rw [ha2]
    show X86Int.cond (aluF .sub 2 (2 % W)).2 13 = some true
    rw [h2W]
    apply cond_cc13_of_sf_eq_of (b := false) <;> decide
  have ha2r4 : a2.reg 4 = some (sp0 + 8) := by rw [ha2]; exact ha1r4
  have ha2mem : a2.mem = s.mem := by rw [ha2]; exact ha1mem
  have ha2k : a2.k = s.k := by rw [ha2]; exact ha1k
  -- step 3: jge 0x84 (taken)
  have e3 : X86Int.step stage1Im a2 = .ok { a2 with rip := base + 0x84 } := by
    have h := step_jcc (im := stage1Im) (s := a2)
      (by rw [ha2rip]; exact hc.d7d) ha2fl
    simpa using h
  obtain ⟨a3, ha3⟩ : ∃ t : X86Int.St, t =
      { a2 with rip := base + 0x84 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = base + 0x84 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp0 + 8) := by
    rw [ha3]; show ({ a2 with rip := base + 0x84 }).reg 4 = some (sp0 + 8)
    rw [reg_with_rip]; exact ha2r4
  have ha3mem : a3.mem = s.mem := by rw [ha3]; exact ha2mem
  have ha3k : a3.k = s.k := by rw [ha3]; exact ha2k
  have hlen3 : a3.regs.length = 16 := by
    rw [ha3]; show a2.regs.length = 16
    rw [ha2]; show a1.regs.length = 16
    exact hlen1
  -- steps 4-5: pop rsi twice (argv[0] discarded, argv[1] kept)
  have hrd4 : rd64 a3.mem (sp0 + 8) = some (sp0 + 56) := by
    have h := initStack_rd64_argv0 (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0, hA0mod] at h
    rwa [ha3mem, hmem]
  have e4 : X86Int.step stage1Im a3 =
      .ok { (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
        rip := a3.rip + 1 } :=
    step_pop (by rw [ha3rip]; exact hc.d84) ha3r4 hrd4
  obtain ⟨a4, ha4⟩ : ∃ t : X86Int.St, t =
      { (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
        rip := a3.rip + 1 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = base + 0x85 := by
    rw [ha4]; show a3.rip + 1 = base + 0x85; rw [ha3rip]
  have hlen4 : a4.regs.length = 16 := by
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen3
  have ha4r4 : a4.reg 4 = some (sp0 + 16) := by
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 4 = some (sp0 + 16)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha4mem : a4.mem = s.mem := by rw [ha4]; exact ha3mem
  have ha4k : a4.k = s.k := by rw [ha4]; exact ha3k
  have hrd5 : rd64 a4.mem (sp0 + 16) =
      some (sp0 + 56 + argv0.length + 1) := by
    have h := initStack_rd64_argv1 (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0, hPmod] at h
    rwa [ha4mem, hmem]
  have e5 : X86Int.step stage1Im a4 =
      .ok { (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
          (some (sp0 + 56 + argv0.length + 1)) with rip := a4.rip + 1 } :=
    step_pop (by rw [ha4rip]; exact hc.d85) ha4r4 hrd5
  obtain ⟨a5, ha5⟩ : ∃ t : X86Int.St, t =
      { (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
          (some (sp0 + 56 + argv0.length + 1)) with rip := a4.rip + 1 } :=
    ⟨_, rfl⟩
  have ha5rip : a5.rip = base + 0x86 := by
    rw [ha5]; show a4.rip + 1 = base + 0x86; rw [ha4rip]
  have hlen5 : a5.regs.length = 16 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen4
  have ha5r6 : a5.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 6 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_self (by
      show 6 < (a4.set 4 (some ((sp0 + 16 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha5r4 : a5.reg 4 = some (sp0 + 24) := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha5mem : a5.mem = s.mem := by rw [ha5]; exact ha4mem
  have ha5k : a5.k = s.k := by rw [ha5]; exact ha4k
  have ha5r2 : a5.reg 2 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 6),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 6),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 2 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 2 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 0),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    exact hr0 2 (by decide) (by decide)
  have ha5r10 : a5.reg 10 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 10 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 10 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 0),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    exact hr0 10 (by decide) (by decide)
  have ha5r8 : a5.reg 8 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 8 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 8 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    exact hr0 8 (by decide) (by decide)
  have ha5r9 : a5.reg 9 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 9 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 9 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    exact hr0 9 (by decide) (by decide)
  have reg_with_kmr : ∀ (t : X86Int.St) (k' : Kern) (m' : Mem) (x i : Nat),
      ({ t with k := k', mem := m', rip := x }).reg i = t.reg i :=
    fun _ _ _ _ _ => rfl
  -- step 6: mov eax, 2 (open)
  have e6 : X86Int.step stage1Im a5 =
      .ok { a5.set 0 (some (2 % W)) with rip := a5.rip + 5 } :=
    step_movI (by rw [ha5rip]; exact hc.d86)
  obtain ⟨a6, ha6⟩ : ∃ t : X86Int.St, t =
      { a5.set 0 (some (2 % W)) with rip := a5.rip + 5 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = base + 0x8b := by
    rw [ha6]; show a5.rip + 5 = base + 0x8b; rw [ha5rip]
  have hlen6 : a6.regs.length = 16 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen5
  have ha6r0 : a6.reg 0 = some 2 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha6r6 : a6.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 6 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 0)]
    exact ha5r6
  have ha6r2 : a6.reg 2 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact ha5r2
  have ha6r10 : a6.reg 10 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact ha5r10
  have ha6r8 : a6.reg 8 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact ha5r8
  have ha6r9 : a6.reg 9 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact ha5r9
  have ha6r4 : a6.reg 4 = some (sp0 + 24) := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha5r4
  have ha6mem : a6.mem = s.mem := by rw [ha6]; exact ha5mem
  have ha6k : a6.k = s.k := by rw [ha6]; exact ha5k
  -- step 7: mov rdi, rsi (the path)
  have e7 : X86Int.step stage1Im a6 =
      .ok { a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
        rip := a6.rip + 3 } :=
    step_mov (by rw [ha6rip]; exact hc.d8b) ha6r6
  obtain ⟨a7, ha7⟩ : ∃ t : X86Int.St, t =
      { a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
        rip := a6.rip + 3 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = base + 0x8e := by
    rw [ha7]; show a6.rip + 3 = base + 0x8e; rw [ha6rip]
  have hlen7 : a7.regs.length = 16 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen6
  have ha7r7 : a7.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha7r0 : a7.reg 0 = some 2 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact ha6r0
  have ha7r6 : a7.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 6 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    exact ha6r6
  have ha7r2 : a7.reg 2 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 7)]
    exact ha6r2
  have ha7r10 : a7.reg 10 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 7)]
    exact ha6r10
  have ha7r8 : a7.reg 8 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    exact ha6r8
  have ha7r9 : a7.reg 9 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    exact ha6r9
  have ha7r4 : a7.reg 4 = some (sp0 + 24) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ha6r4
  have ha7mem : a7.mem = s.mem := by rw [ha7]; exact ha6mem
  have ha7k : a7.k = s.k := by rw [ha7]; exact ha6k
  -- step 8: xor esi, esi
  have e8 : X86Int.step stage1Im a7 =
      .ok { a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a7.rip + 2 } :=
    step_alu_xor (by rw [ha7rip]; exact hc.d8e) ha7r6 ha7r6
  obtain ⟨a8, ha8⟩ : ∃ t : X86Int.St, t =
      { a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a7.rip + 2 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = base + 0x90 := by
    rw [ha8]; show a7.rip + 2 = base + 0x90; rw [ha7rip]
  have hlen8 : a8.regs.length = 16 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).regs.length = 16
    rw [len_set]
    exact hlen7
  have ha8r6 : a8.reg 6 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 6 = some 0
    rw [reg_set_self (by
      show 6 < a7.regs.length
      omega), aluF_xor_fst_self]
  have ha8r0 : a8.reg 0 = some 2 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 0 = some 2
    rw [reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact ha7r0
  have ha8r7 : a8.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact ha7r7
  have ha8r2 : a8.reg 2 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 2 = some 0
    rw [reg_set_other (by decide : (2 : Nat) ≠ 6)]
    exact ha7r2
  have ha8r10 : a8.reg 10 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 10 = some 0
    rw [reg_set_other (by decide : (10 : Nat) ≠ 6)]
    exact ha7r10
  have ha8r8 : a8.reg 8 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 8 = some 0
    rw [reg_set_other (by decide : (8 : Nat) ≠ 6)]
    exact ha7r8
  have ha8r9 : a8.reg 9 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 6)]
    exact ha7r9
  have ha8r4 : a8.reg 4 = some (sp0 + 24) := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact ha7r4
  have ha8mem : a8.mem = s.mem := by rw [ha8]; exact ha7mem
  have ha8k : a8.k = s.k := by rw [ha8]; exact ha7k
  -- step 9: xor edx, edx
  have e9 : X86Int.step stage1Im a8 =
      .ok { a8.set 2 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a8.rip + 2 } :=
    step_alu_xor (by rw [ha8rip]; exact hc.d90) ha8r2 ha8r2
  obtain ⟨a9, ha9⟩ : ∃ t : X86Int.St, t =
      { a8.set 2 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a8.rip + 2 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = base + 0x92 := by
    rw [ha9]; show a8.rip + 2 = base + 0x92; rw [ha8rip]
  have hlen9 : a9.regs.length = 16 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).regs.length = 16
    rw [len_set]
    exact hlen8
  have ha9r0 : a9.reg 0 = some 2 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 0 = some 2
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact ha8r0
  have ha9r7 : a9.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact ha8r7
  have ha9r6 : a9.reg 6 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 6 = some 0
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact ha8r6
  have ha9r2 : a9.reg 2 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 2 = some 0
    rw [reg_set_self (by show 2 < a8.regs.length; omega), aluF_xor_fst_self]
  have ha9r10 : a9.reg 10 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 10 = some 0
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    exact ha8r10
  have ha9r8 : a9.reg 8 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 8 = some 0
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    exact ha8r8
  have ha9r9 : a9.reg 9 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    exact ha8r9
  have ha9r4 : a9.reg 4 = some (sp0 + 24) := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    exact ha8r4
  have ha9mem : a9.mem = s.mem := by rw [ha9]; exact ha8mem
  have ha9k : a9.k = s.k := by rw [ha9]; exact ha8k
  -- step 10: open(path, O_RDONLY) = 3
  have hcstr : Mem.cstr a9.mem (sp0 + 56 + argv0.length + 1) 4096 =
      some path := by
    have h := initStack_cstr_path (by rw [← hstk]; exact hdisj')
      hpath0 hpathlen
    rw [← hstk, ← hsp0] at h
    rwa [ha9mem, hmem]
  have hsc : X86Env.syscall a9.k a9.mem 2
        (sp0 + 56 + argv0.length + 1) 0 0 0 0 0 =
      some (.ret 3 ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩
        a9.mem) := by
    rw [ha9k, hk]
    simp only [X86Env.syscall]
    rw [if_neg (by simp), hcstr]
    simp [X86Env.lookup, X86Env.freeSlot, X86Env.setSlot]
  have e10 : X86Int.step stage1Im a9 =
      .ok { ((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11 none with
        k := ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩,
        mem := a9.mem, rip := a9.rip + 2 } :=
    step_syscall_ret (by rw [ha9rip]; exact hc.d92) ha9r0 ha9r7 ha9r6
      ha9r2 ha9r10 ha9r8 ha9r9 hsc
  obtain ⟨a10, ha10⟩ : ∃ t : X86Int.St, t =
      { ((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11 none with
        k := ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩,
        mem := a9.mem, rip := a9.rip + 2 } := ⟨_, rfl⟩
  have ha10rip : a10.rip = base + 0x94 := by
    rw [ha10]; show a9.rip + 2 = base + 0x94; rw [ha9rip]
  have hlen10 : a10.regs.length = 16 := by
    rw [ha10]
    show (((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen9
  have ha10r0 : a10.reg 0 = some 3 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have ha10r7 : a10.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (7 : Nat) ≠ 11),
      reg_set_other (by decide : (7 : Nat) ≠ 1),
      reg_set_other (by decide : (7 : Nat) ≠ 0)]
    exact ha9r7
  have ha10r6 : a10.reg 6 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (6 : Nat) ≠ 11),
      reg_set_other (by decide : (6 : Nat) ≠ 1),
      reg_set_other (by decide : (6 : Nat) ≠ 0)]
    exact ha9r6
  have ha10r2 : a10.reg 2 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (2 : Nat) ≠ 11),
      reg_set_other (by decide : (2 : Nat) ≠ 1),
      reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact ha9r2
  have ha10r10 : a10.reg 10 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact ha9r10
  have ha10r8 : a10.reg 8 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact ha9r8
  have ha10r9 : a10.reg 9 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact ha9r9
  have ha10r4 : a10.reg 4 = some (sp0 + 24) := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha9r4
  have ha10mem : a10.mem = s.mem := by rw [ha10]; exact ha9mem
  have ha10k : a10.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by
    rw [ha10]
  -- step 11: test rax, rax (fd = 3, not negative)
  have e11 : X86Int.step stage1Im a10 =
      .ok { a10 with fl := (aluF .and 3 3).2, rip := a10.rip + 3 } :=
    step_test (by rw [ha10rip]; exact hc.d94) ha10r0 ha10r0
  obtain ⟨a11, ha11⟩ : ∃ t : X86Int.St, t =
      { a10 with fl := (aluF .and 3 3).2, rip := a10.rip + 3 } := ⟨_, rfl⟩
  have ha11rip : a11.rip = base + 0x97 := by
    rw [ha11]; show a10.rip + 3 = base + 0x97; rw [ha10rip]
  have ha11fl : X86Int.cond a11.fl 9 = some true := by
    rw [ha11]
    show X86Int.cond (aluF .and 3 3).2 9 = some true
    apply cond_cc9_of_sf_false
    decide
  have ha11r0 : a11.reg 0 = some 3 := by rw [ha11]; exact ha10r0
  have ha11r4 : a11.reg 4 = some (sp0 + 24) := by rw [ha11]; exact ha10r4
  have ha11mem : a11.mem = s.mem := by rw [ha11]; exact ha10mem
  have ha11k : a11.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha11]; exact ha10k
  have hlen11 : a11.regs.length = 16 := by
    rw [ha11]; show a10.regs.length = 16
    exact hlen10
  -- step 12: jns 0x9e (taken)
  have e12 : X86Int.step stage1Im a11 =
      .ok { a11 with rip := base + 0x9e } := by
    have h := step_jcc (im := stage1Im) (s := a11)
      (by rw [ha11rip]; exact hc.d97) ha11fl
    simpa using h
  obtain ⟨a12, ha12⟩ : ∃ t : X86Int.St, t =
      { a11 with rip := base + 0x9e } := ⟨_, rfl⟩
  have ha12rip : a12.rip = base + 0x9e := by rw [ha12]
  have ha12r0 : a12.reg 0 = some 3 := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).reg 0 = some 3
    rw [reg_with_rip]; exact ha11r0
  have ha12r4 : a12.reg 4 = some (sp0 + 24) := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact ha11r4
  have ha12mem : a12.mem = s.mem := by rw [ha12]; exact ha11mem
  have ha12k : a12.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha12]; exact ha11k
  have hlen12 : a12.regs.length = 16 := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).regs.length = 16
    rw [len_with_rip]; exact hlen11
  have ha12r7 : a12.reg 7 = some (sp0 + 56 + argv0.length + 1) := by rw [ha12, ha11]; exact ha10r7
  have ha12r6 : a12.reg 6 = some 0 := by rw [ha12, ha11]; exact ha10r6
  have ha12r2 : a12.reg 2 = some 0 := by rw [ha12, ha11]; exact ha10r2
  have ha12r10 : a12.reg 10 = some 0 := by rw [ha12, ha11]; exact ha10r10
  have ha12r8 : a12.reg 8 = some 0 := by rw [ha12, ha11]; exact ha10r8
  have ha12r9 : a12.reg 9 = some 0 := by rw [ha12, ha11]; exact ha10r9
  -- step 13: mov r14, rax (keep the fd)
  have e13 : X86Int.step stage1Im a12 =
      .ok { a12.set 14 (some 3) with rip := a12.rip + 3 } :=
    step_mov (by rw [ha12rip]; exact hc.d9e) ha12r0
  obtain ⟨a13, ha13⟩ : ∃ t : X86Int.St, t =
      { a12.set 14 (some 3) with rip := a12.rip + 3 } := ⟨_, rfl⟩
  have ha13rip : a13.rip = base + 0xa1 := by
    rw [ha13]; show a12.rip + 3 = base + 0xa1; rw [ha12rip]
  have hlen13 : a13.regs.length = 16 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen12
  have ha13r14 : a13.reg 14 = some 3 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha13r0 : a13.reg 0 = some 3 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 0 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 14)]
    exact ha12r0
  have ha13r7 : a13.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 7 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 14)]
    exact ha12r7
  have ha13r4 : a13.reg 4 = some (sp0 + 24) := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 14)]
    exact ha12r4
  have ha13mem : a13.mem = s.mem := by rw [ha13]; exact ha12mem
  have ha13k : a13.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha13]; exact ha12k
  -- step 14: mov eax, 9 (mmap)
  have e14 : X86Int.step stage1Im a13 =
      .ok { a13.set 0 (some (9 % W)) with rip := a13.rip + 5 } :=
    step_movI (by rw [ha13rip]; exact hc.da1)
  obtain ⟨a14, ha14⟩ : ∃ t : X86Int.St, t =
      { a13.set 0 (some (9 % W)) with rip := a13.rip + 5 } := ⟨_, rfl⟩
  have ha14rip : a14.rip = base + 0xa6 := by
    rw [ha14]; show a13.rip + 5 = base + 0xa6; rw [ha13rip]
  have hlen14 : a14.regs.length = 16 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen13
  have ha14r0 : a14.reg 0 = some 9 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha14r7 : a14.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 7 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 0)]
    exact ha13r7
  have ha14r14 : a14.reg 14 = some 3 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact ha13r14
  have ha14r4 : a14.reg 4 = some (sp0 + 24) := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha13r4
  have ha14mem : a14.mem = s.mem := by rw [ha14]; exact ha13mem
  have ha14k : a14.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha14]; exact ha13k
  -- step 15: xor edi, edi
  have e15 : X86Int.step stage1Im a14 =
      .ok { a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 } :=
    step_alu_xor (by rw [ha14rip]; exact hc.da6) ha14r7 ha14r7
  obtain ⟨a15, ha15⟩ : ∃ t : X86Int.St, t =
      { a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 } := ⟨_, rfl⟩
  have ha15rip : a15.rip = base + 0xa8 := by
    rw [ha15]; show a14.rip + 2 = base + 0xa8; rw [ha14rip]
  have hlen15 : a15.regs.length = 16 := by
    rw [ha15]
    show ({ a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1) with
      fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 }).regs.length = 16
    rw [len_set]
    exact hlen14
  have ha15r7 : a15.reg 7 = some 0 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 7 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have ha15r0 : a15.reg 0 = some 9 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact ha14r0
  have ha15r14 : a15.reg 14 = some 3 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact ha14r14
  have ha15r4 : a15.reg 4 = some (sp0 + 24) := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ha14r4
  have ha15mem : a15.mem = s.mem := by rw [ha15]; exact ha14mem
  have ha15k : a15.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha15]; exact ha14k
  -- steps 16-18: mov esi, 65536; mov edx, 3; mov r10d, 34
  have e16 : X86Int.step stage1Im a15 =
      .ok { a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 } :=
    step_movI (by rw [ha15rip]; exact hc.da8)
  obtain ⟨a16, ha16⟩ : ∃ t : X86Int.St, t =
      { a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 } := ⟨_, rfl⟩
  have ha16rip : a16.rip = base + 0xad := by
    rw [ha16]; show a15.rip + 5 = base + 0xad; rw [ha15rip]
  have hlen16 : a16.regs.length = 16 := by
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with
      rip := a15.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen15
  have ha16r6 : a16.reg 6 = some 65536 := by
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with
      rip := a15.rip + 5 }).reg 6 = some 65536
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have e17 : X86Int.step stage1Im a16 =
      .ok { a16.set 2 (some (3 % W)) with rip := a16.rip + 5 } :=
    step_movI (by rw [ha16rip]; exact hc.dad)
  obtain ⟨a17, ha17⟩ : ∃ t : X86Int.St, t =
      { a16.set 2 (some (3 % W)) with rip := a16.rip + 5 } := ⟨_, rfl⟩
  have ha17rip : a17.rip = base + 0xb2 := by
    rw [ha17]; show a16.rip + 5 = base + 0xb2; rw [ha16rip]
  have hlen17 : a17.regs.length = 16 := by
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen16
  have e18 : X86Int.step stage1Im a17 =
      .ok { a17.set 10 (some (34 % W)) with rip := a17.rip + 6 } :=
    step_movI (by rw [ha17rip]; exact hc.db2)
  obtain ⟨a18, ha18⟩ : ∃ t : X86Int.St, t =
      { a17.set 10 (some (34 % W)) with rip := a17.rip + 6 } := ⟨_, rfl⟩
  have ha18rip : a18.rip = base + 0xb8 := by
    rw [ha18]; show a17.rip + 6 = base + 0xb8; rw [ha17rip]
  have hlen18 : a18.regs.length = 16 := by
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with
      rip := a17.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen17
  have ha18mem : a18.mem = s.mem := by rw [ha18, ha17, ha16]; exact ha15mem
  have ha18k : a18.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha18, ha17, ha16]; exact ha15k
  have ha18r4 : a18.reg 4 = some (sp0 + 24) := by
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact ha15r4
  -- step 19: push -1
  have hwr : ∃ stk', wr64 a18.mem (sp0 + 16) ((W - 1) % W) =
      some [⟨base, stage1Code, false⟩, stk'] ∧ stk'.base = stk.base ∧
      stk'.hi = stk.hi ∧ stk'.rw = stk.rw := by
    rw [ha18mem, hmem]
    have h := wrN_second_some (r := stk) (a := sp0 + 16)
      (bytes8 ((W - 1) % W)) hdisj' (by
      rw [hstk]
      simp [X86Env.initStack]) (by omega) (by
      have hlen8 : (bytes8 ((W - 1) % W)).length = 8 := by simp [bytes8]
      rw [hlen8]
      have hge : X86Env.stackRoom + 24 ≤ stk.data.length := by
        rw [hdatalen]
        have := pages_ge (56 + ((argv0.length + 1) + (path.length + 1)) + 1)
        omega
      unfold Region.hi
      omega)
    obtain ⟨stk', h1, h2, h3, h4⟩ := h
    exact ⟨stk', h1, h2, h3, h4⟩
  obtain ⟨stk', hwr, hstk'b, hstk'hi, hstk'rw⟩ := hwr
  have hwshape : wshape [⟨base, stage1Code, false⟩, stk'] =
      wshape a18.mem := by
    have h := wrN_wshape (bytes8 ((W - 1) % W)) a18.mem (sp0 + 16) hwr
    exact h
  have e19 : X86Int.step stage1Im a18 =
      .ok { a18.set 4 (some ((sp0 + 24 + W - 8) % W)) with
        mem := [⟨base, stage1Code, false⟩, stk'], rip := a18.rip + 2 } :=
    step_pushI (by rw [ha18rip]; exact hc.db8) ha18r4 (by
      rw [show (sp0 + 24 + W - 8) % W = sp0 + 16 from by
        rw [show sp0 + 24 + W - 8 = (sp0 + 16) + W from by omega,
          Nat.add_mod_right]
        exact Nat.mod_eq_of_lt (by omega)]
      exact hwr)
  obtain ⟨a19, ha19⟩ : ∃ t : X86Int.St, t =
      { a18.set 4 (some ((sp0 + 24 + W - 8) % W)) with
        mem := [⟨base, stage1Code, false⟩, stk'], rip := a18.rip + 2 } :=
    ⟨_, rfl⟩
  have ha19rip : a19.rip = base + 0xba := by
    rw [ha19]; show a18.rip + 2 = base + 0xba; rw [ha18rip]
  have hlen19 : a19.regs.length = 16 := by
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).regs.length = 16
    rw [len_set]
    exact hlen18
  have ha19r4 : a19.reg 4 = some (sp0 + 16) := by
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 4 = some (sp0 + 16)
    rw [reg_set_self (by omega),
      show (sp0 + 24 + W - 8) % W = sp0 + 16 from by
        rw [show sp0 + 24 + W - 8 = (sp0 + 16) + W from by omega,
          Nat.add_mod_right]
        exact Nat.mod_eq_of_lt (by omega)]
  have ha19mem : a19.mem = [⟨base, stage1Code, false⟩, stk'] := by
    rw [ha19]
  have ha19k : a19.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha19]; exact ha18k
  -- step 20: pop r8 (the -1 just pushed)
  have hrd20 : rd64 a19.mem (sp0 + 16) = some (W - 1) := by
    rw [ha19mem]
    have h := rd64_wr64 hwr
    rw [Nat.mod_mod, Nat.mod_eq_of_lt (by unfold W; omega)] at h
    exact h
  have e20 : X86Int.step stage1Im a19 =
      .ok { (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := a19.rip + 2 } :=
    step_pop (by rw [ha19rip]; exact hc.dba) ha19r4 hrd20
  obtain ⟨a20, ha20⟩ : ∃ t : X86Int.St, t =
      { (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := a19.rip + 2 } := ⟨_, rfl⟩
  have ha20rip : a20.rip = base + 0xbc := by
    rw [ha20]; show a19.rip + 2 = base + 0xbc; rw [ha19rip]
  have hlen20 : a20.regs.length = 16 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen19
  have ha20r9 : a20.reg 9 = some 0 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 8),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 9 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 14)]
    exact ha12r9
  have ha20mem : a20.mem = [⟨base, stage1Code, false⟩, stk'] := by rw [ha20]; exact ha19mem
  have ha20k : a20.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha20]; exact ha19k
  -- step 21: xor r9d, r9d
  have e21 : X86Int.step stage1Im a20 =
      .ok { a20.set 9 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a20.rip + 3 } :=
    step_alu_xor (by rw [ha20rip]; exact hc.dbc) ha20r9 ha20r9
  obtain ⟨a21, ha21⟩ : ∃ t : X86Int.St, t =
      { a20.set 9 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a20.rip + 3 } := ⟨_, rfl⟩
  have ha21rip : a21.rip = base + 0xbf := by
    rw [ha21]; show a20.rip + 3 = base + 0xbf; rw [ha20rip]
  have hlen21 : a21.regs.length = 16 := by
    rw [ha21]
    show ({ a20.set 9 (some (aluF .xor 0 0).1) with
      fl := (aluF .xor 0 0).2, rip := a20.rip + 3 }).regs.length = 16
    rw [len_set]
    exact hlen20
  -- register conclusions at a21, by unwinding the set-chains
  have ha21r9 : a21.reg 9 = some 0 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 9 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have ha20r8 : a20.reg 8 = some (W - 1) := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 8 = some (W - 1)
    rw [reg_with_rip, reg_set_self (by
      show 8 < (a19.set 4 (some ((sp0 + 16 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha21r8 : a21.reg 8 = some (W - 1) := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 9)]
    exact ha20r8
  have ha20r4 : a20.reg 4 = some (sp0 + 24) := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 8),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha21r4 : a21.reg 4 = some (sp0 + 24) := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 9)]
    exact ha20r4
  have ha20r0 : a20.reg 0 = some 9 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 8),
      reg_set_other (by decide : (0 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact ha15r0
  have ha21r0 : a21.reg 0 = some 9 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 9)]
    exact ha20r0
  have ha20r7 : a20.reg 7 = some 0 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 8),
      reg_set_other (by decide : (7 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 7 = some 0
    rw [reg_set_other (by decide : (7 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact ha15r7
  have ha21r7 : a21.reg 7 = some 0 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 7 = some 0
    rw [reg_set_other (by decide : (7 : Nat) ≠ 9)]
    exact ha20r7
  have ha20r6 : a20.reg 6 = some 65536 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 6 = some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 8),
      reg_set_other (by decide : (6 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 6 = some 65536
    rw [reg_set_other (by decide : (6 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 6 =
      some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 6 =
      some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact ha16r6
  have ha21r6 : a21.reg 6 = some 65536 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 6 = some 65536
    rw [reg_set_other (by decide : (6 : Nat) ≠ 9)]
    exact ha20r6
  have ha20r2 : a20.reg 2 = some 3 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 8),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 2 = some 3
    rw [reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha21r2 : a21.reg 2 = some 3 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 2 = some 3
    rw [reg_set_other (by decide : (2 : Nat) ≠ 9)]
    exact ha20r2
  have ha20r10 : a20.reg 10 = some 34 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 10 = some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 8),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha21r10 : a21.reg 10 = some 34 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 9)]
    exact ha20r10
  have ha20r14 : a20.reg 14 = some 3 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 8),
      reg_set_other (by decide : (14 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact ha14r14
  have ha21r14 : a21.reg 14 = some 3 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 9)]
    exact ha20r14
  have ha21mem : a21.mem = [⟨base, stage1Code, false⟩, stk'] := by rw [ha21]; exact ha20mem
  have ha21k : a21.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha21]; exact ha20k
  -- fold the run
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ha1, run_zero]
  have r2 : run stage1Im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok stage1Im 0 a1, e2, ← ha2, run_zero]
  have r3 : run stage1Im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok stage1Im 0 a2, e3, ← ha3, run_zero]
  have r4 : run stage1Im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok stage1Im 0 a3, e4, ← ha4, run_zero]
  have r5 : run stage1Im 1 (Out.ok a4) = Out.ok a5 := by
    rw [run_succ_ok stage1Im 0 a4, e5, ← ha5, run_zero]
  have r6 : run stage1Im 1 (Out.ok a5) = Out.ok a6 := by
    rw [run_succ_ok stage1Im 0 a5, e6, ← ha6, run_zero]
  have r7 : run stage1Im 1 (Out.ok a6) = Out.ok a7 := by
    rw [run_succ_ok stage1Im 0 a6, e7, ← ha7, run_zero]
  have r8 : run stage1Im 1 (Out.ok a7) = Out.ok a8 := by
    rw [run_succ_ok stage1Im 0 a7, e8, ← ha8, run_zero]
  have r9 : run stage1Im 1 (Out.ok a8) = Out.ok a9 := by
    rw [run_succ_ok stage1Im 0 a8, e9, ← ha9, run_zero]
  have r10 : run stage1Im 1 (Out.ok a9) = Out.ok a10 := by
    rw [run_succ_ok stage1Im 0 a9, e10, ← ha10, run_zero]
  have r11 : run stage1Im 1 (Out.ok a10) = Out.ok a11 := by
    rw [run_succ_ok stage1Im 0 a10, e11, ← ha11, run_zero]
  have r12 : run stage1Im 1 (Out.ok a11) = Out.ok a12 := by
    rw [run_succ_ok stage1Im 0 a11, e12, ← ha12, run_zero]
  have r13 : run stage1Im 1 (Out.ok a12) = Out.ok a13 := by
    rw [run_succ_ok stage1Im 0 a12, e13, ← ha13, run_zero]
  have r14 : run stage1Im 1 (Out.ok a13) = Out.ok a14 := by
    rw [run_succ_ok stage1Im 0 a13, e14, ← ha14, run_zero]
  have r15 : run stage1Im 1 (Out.ok a14) = Out.ok a15 := by
    rw [run_succ_ok stage1Im 0 a14, e15, ← ha15, run_zero]
  have r16 : run stage1Im 1 (Out.ok a15) = Out.ok a16 := by
    rw [run_succ_ok stage1Im 0 a15, e16, ← ha16, run_zero]
  have r17 : run stage1Im 1 (Out.ok a16) = Out.ok a17 := by
    rw [run_succ_ok stage1Im 0 a16, e17, ← ha17, run_zero]
  have r18 : run stage1Im 1 (Out.ok a17) = Out.ok a18 := by
    rw [run_succ_ok stage1Im 0 a17, e18, ← ha18, run_zero]
  have r19 : run stage1Im 1 (Out.ok a18) = Out.ok a19 := by
    rw [run_succ_ok stage1Im 0 a18, e19, ← ha19, run_zero]
  have r20 : run stage1Im 1 (Out.ok a19) = Out.ok a20 := by
    rw [run_succ_ok stage1Im 0 a19, e20, ← ha20, run_zero]
  have r21 : run stage1Im 1 (Out.ok a20) = Out.ok a21 := by
    rw [run_succ_ok stage1Im 0 a20, e21, ← ha21, run_zero]
  have hrun : run stage1Im 21 (Out.ok s) = Out.ok a21 := by
    rw [show (21 : Nat) = 1 + 20 from rfl, run_add stage1Im 1 20, r1,
      show (20 : Nat) = 1 + 19 from rfl, run_add stage1Im 1 19, r2,
      show (19 : Nat) = 1 + 18 from rfl, run_add stage1Im 1 18, r3,
      show (18 : Nat) = 1 + 17 from rfl, run_add stage1Im 1 17, r4,
      show (17 : Nat) = 1 + 16 from rfl, run_add stage1Im 1 16, r5,
      show (16 : Nat) = 1 + 15 from rfl, run_add stage1Im 1 15, r6,
      show (15 : Nat) = 1 + 14 from rfl, run_add stage1Im 1 14, r7,
      show (14 : Nat) = 1 + 13 from rfl, run_add stage1Im 1 13, r8,
      show (13 : Nat) = 1 + 12 from rfl, run_add stage1Im 1 12, r9,
      show (12 : Nat) = 1 + 11 from rfl, run_add stage1Im 1 11, r10,
      show (11 : Nat) = 1 + 10 from rfl, run_add stage1Im 1 10, r11,
      show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r12,
      show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r13,
      show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r14,
      show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r15,
      show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r16,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r17,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r18,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r19,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r20, r21]
  have hchar : ∀ j t, j ≤ 21 →
      run stage1Im j (Out.ok s) = Out.ok t →
      AtMemSyscall t → j = 21 ∧ t = a21 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
      j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9 ∨ j = 10 ∨ j = 11 ∨
      j = 12 ∨ j = 13 ∨ j = 14 ∨ j = 15 ∨ j = 16 ∨ j = 17 ∨
      j = 18 ∨ j = 19 ∨ j = 20 ∨ j = 21) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.d78] at h1
      simp at h1
    · have hrj : run stage1Im 1 (Out.ok s) = Out.ok a1 := by
        rw [r1]
      have ht : t = a1 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha1rip, hc.d79] at h1
      simp at h1
    · have hrj : run stage1Im 2 (Out.ok s) = Out.ok a2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
      have ht : t = a2 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha2rip, hc.d7d] at h1
      simp at h1
    · have hrj : run stage1Im 3 (Out.ok s) = Out.ok a3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
      have ht : t = a3 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha3rip, hc.d84] at h1
      simp at h1
    · have hrj : run stage1Im 4 (Out.ok s) = Out.ok a4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
      have ht : t = a4 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha4rip, hc.d85] at h1
      simp at h1
    · have hrj : run stage1Im 5 (Out.ok s) = Out.ok a5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
      have ht : t = a5 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha5rip, hc.d86] at h1
      simp at h1
    · have hrj : run stage1Im 6 (Out.ok s) = Out.ok a6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
      have ht : t = a6 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha6rip, hc.d8b] at h1
      simp at h1
    · have hrj : run stage1Im 7 (Out.ok s) = Out.ok a7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
      have ht : t = a7 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha7rip, hc.d8e] at h1
      simp at h1
    · have hrj : run stage1Im 8 (Out.ok s) = Out.ok a8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
      have ht : t = a8 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha8rip, hc.d90] at h1
      simp at h1
    · have hrj : run stage1Im 9 (Out.ok s) = Out.ok a9 := by
        rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
      have ht : t = a9 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      rcases hmem.2 with h | h <;> rw [ha9r0] at h <;> simp at h
    · have hrj : run stage1Im 10 (Out.ok s) = Out.ok a10 := by
        rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r8, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
      have ht : t = a10 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha10rip, hc.d94] at h1
      simp at h1
    · have hrj : run stage1Im 11 (Out.ok s) = Out.ok a11 := by
        rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
      have ht : t = a11 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha11rip, hc.d97] at h1
      simp at h1
    · have hrj : run stage1Im 12 (Out.ok s) = Out.ok a12 := by
        rw [show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r1, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r2, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r3, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r4, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r5, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r6, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r7, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r8, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r9, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
      have ht : t = a12 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha12rip, hc.d9e] at h1
      simp at h1
    · have hrj : run stage1Im 13 (Out.ok s) = Out.ok a13 := by
        rw [show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r1, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r2, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r3, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r4, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r5, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r6, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r7, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r8, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r9, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r10, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r11, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r12, r13]
      have ht : t = a13 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha13rip, hc.da1] at h1
      simp at h1
    · have hrj : run stage1Im 14 (Out.ok s) = Out.ok a14 := by
        rw [show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r1, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r2, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r3, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r4, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r5, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r6, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r7, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r8, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r9, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r10, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r11, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r12, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r13, r14]
      have ht : t = a14 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha14rip, hc.da6] at h1
      simp at h1
    · have hrj : run stage1Im 15 (Out.ok s) = Out.ok a15 := by
        rw [show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r1, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r2, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r3, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r4, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r5, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r6, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r7, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r8, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r9, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r10, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r11, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r12, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r13, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r14, r15]
      have ht : t = a15 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha15rip, hc.da8] at h1
      simp at h1
    · have hrj : run stage1Im 16 (Out.ok s) = Out.ok a16 := by
        rw [show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r1, show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r2, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r3, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r4, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r5, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r6, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r7, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r8, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r9, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r10, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r11, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r12, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r13, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r14, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r15, r16]
      have ht : t = a16 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha16rip, hc.dad] at h1
      simp at h1
    · have hrj : run stage1Im 17 (Out.ok s) = Out.ok a17 := by
        rw [show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, r1, show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r2, show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r3, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r4, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r5, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r6, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r7, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r8, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r9, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r10, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r11, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r12, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r13, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r14, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r15, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r16, r17]
      have ht : t = a17 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha17rip, hc.db2] at h1
      simp at h1
    · have hrj : run stage1Im 18 (Out.ok s) = Out.ok a18 := by
        rw [show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, r1, show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, r2, show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r3, show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r4, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r5, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r6, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r7, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r8, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r9, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r10, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r11, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r12, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r13, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r14, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r15, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r16, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r17, r18]
      have ht : t = a18 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha18rip, hc.db8] at h1
      simp at h1
    · have hrj : run stage1Im 19 (Out.ok s) = Out.ok a19 := by
        rw [show (19 : Nat) = 1 + 18 by omega, run_add stage1Im 1 18, r1, show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, r2, show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, r3, show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r4, show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r5, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r6, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r7, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r8, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r9, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r10, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r11, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r12, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r13, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r14, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r15, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r16, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r17, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r18, r19]
      have ht : t = a19 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha19rip, hc.dba] at h1
      simp at h1
    · have hrj : run stage1Im 20 (Out.ok s) = Out.ok a20 := by
        rw [show (20 : Nat) = 1 + 19 by omega, run_add stage1Im 1 19, r1, show (19 : Nat) = 1 + 18 by omega, run_add stage1Im 1 18, r2, show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, r3, show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, r4, show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r5, show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r6, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r7, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r8, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r9, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r10, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r11, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r12, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r13, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r14, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r15, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r16, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r17, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r18, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r19, r20]
      have ht : t = a20 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ha20rip, hc.dbc] at h1
      simp at h1
    · have ht : t = a21 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      exact ⟨rfl, ht⟩
  exact ⟨a21, hrun, ha21rip, ha21r0, ha21r7, ha21r6, ha21r2, ha21r10,
    ha21r8, ha21r9, ha21r14, ha21r4, ha21k, by
      rw [ha21mem, hwshape, ha18mem], hlen21, ⟨stk', ha21mem⟩,
    hchar⟩


/-! ### §3e. Entry + initial `mmap` + read loop for an
arbitrary admitted oracle: exit 2 at the first refused
memory syscall, or loaded at `0x134`, with step bound and
`NoHitUpTo` over the whole run. -/
theorem loader_init_any {f path argv0 stdin : List Nat}
    {oracle : List Choice} {top : Nat}
    (hinit : Stage1InitOk f path argv0 top)
    (hadm : OracleAdmits
      (stage1Init f path argv0 stdin oracle top)) :
    (∃ k, k ≤ 100 + 1000 * (f.length + 1) ∧
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.exit 2) ∨
    (∃ k A' cap' c2 buf' s',
      k ≤ 100 + 1000 * (f.length + 1) ∧
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.ok s' ∧
      s'.rip = base + 0x134 ∧
      s'.reg 3 = some A' ∧ s'.reg 12 = some cap' ∧
      s'.reg 13 = some f.length ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.mem = [⟨base, stage1Code, false⟩, c2, buf'] ∧
      c2.base = (X86Env.initStack top [argv0, path]).1.base ∧
      c2.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
      c2.rw = true ∧
      buf'.base = A' ∧ buf'.hi = A' + cap' ∧ buf'.rw = true ∧
      buf'.data.take f.length = f ∧
      ((⟨base, stage1Code, false⟩ : Region).hi ≤ A' ∨
        A' + cap' ≤ base) ∧
      (c2.hi ≤ A' ∨ A' + cap' ≤ c2.base) ∧
      A' % page = 0 ∧ 0 < A' ∧ A' + cap' ≤ 2 ^ 47 ∧
      cap' % page = 0 ∧ 65536 ≤ cap' ∧
      s'.k.fds = [some ⟨f, f.length⟩] ∧
      s'.regs.length = 16 ∧ RegsOk s' ∧
      NoHitUpTo (stage1Init f path argv0 stdin oracle top) k) := by
  obtain ⟨hfB, -, -, hpath0, -, hpathlen, -, htop47, hleInit,
    hstkhi⟩ := hinit
  have hc := stage1_loaderCode
  have hW : W = 2 ^ 64 := rfl
  have hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base := by
    have h := hleInit
    rw [phVaddr_eq, phMemsz_eq] at h
    exact h
  have hk0 : (stage1Init f path argv0 stdin oracle top).k =
      ⟨[(path, f)], [], stdin, [], oracle⟩ := by
    simp [stage1Init]
  have hregs0 : RegsOk (stage1Init f path argv0 stdin oracle
      top) := by
    intro i hi _
    by_cases hi4 : i = 4
    · subst hi4
      exact ⟨_, stage1Init_reg4 _ _ _ _ _ _⟩
    · exact ⟨0, stage1Init_reg_ne4 _ _ _ _ _ _ hi hi4⟩
  obtain ⟨s1, hrun1, hs1rip, hs1r0, hs1r7, hs1r6, hs1r2, hs1r10,
    hs1r8, hs1r9, hs1r14, hs1r4, hs1k, hshape, hs1len,
    ⟨stk1, hs1memC⟩, hchar⟩ :=
    loader_entry_nh
      (stk := (X86Env.initStack top [argv0, path]).1)
      (sp0 := (X86Env.initStack top [argv0, path]).2)
      (stdinL := stdin) (oracle := oracle)
      rfl rfl (stage1Init_rip _ _ _ _ _ _)
      (stage1Init_regs_length _ _ _ _ _ _)
      (stage1Init_reg4 _ _ _ _ _ _)
      (fun i hi4 hi16 => stage1Init_reg_ne4 _ _ _ _ _ _ hi16 hi4)
      (stage1Init_mem _ _ _ _ _ _) hk0 hdisjInit hstkhi htop47
      hpath0 hpathlen
  have hsp0eq : (X86Env.initStack top [argv0, path]).2 = (X86Env.initStack top [argv0, path]).1.base + 4096 := by
    have h1 := initStack_sp top argv0 path
    have hsr : X86Env.stackRoom = 4096 := rfl
    rw [hsr] at h1
    exact h1
  have hdatalen : (X86Env.initStack top [argv0, path]).1.data.length =
      X86Env.stackRoom +
        X86Env.pages
          (56 + ((argv0.length + 1) + (path.length + 1)) + 1) :=
    initStack_data_length top argv0 path
  have hple : (X86Env.initStack top [argv0, path]).2 + 24 ≤ (X86Env.initStack top [argv0, path]).1.hi := by
    have hp : 4096 ≤ X86Env.pages
        (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
      unfold X86Env.pages X86Env.page
      have h1 : 1 ≤ (56 + ((argv0.length + 1) +
          (path.length + 1)) + 1 + 4095) / 4096 :=
        Nat.div_pos (by omega) (by decide)
      omega
    have hsr : X86Env.stackRoom = 4096 := rfl
    have hh : (X86Env.initStack top [argv0, path]).1.hi = (X86Env.initStack top [argv0, path]).1.base + (X86Env.initStack top [argv0, path]).1.data.length := rfl
    omega
  have hsp0W : (X86Env.initStack top [argv0, path]).2 + 24 < W := by
    rw [hstkhi] at hple
    omega
  have hshape2 : wshape s1.mem =
      [(base, base + stage1Code.length, false),
        ((X86Env.initStack top [argv0, path]).1.base, (X86Env.initStack top [argv0, path]).1.hi, (X86Env.initStack top [argv0, path]).1.rw)] := by
    rw [hshape]
    simp [stage1Init_mem, wshape, Region.hi]
  have hstk1 : stk1.base = (X86Env.initStack top [argv0, path]).1.base ∧ stk1.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
      stk1.rw = (X86Env.initStack top [argv0, path]).1.rw := by
    rw [hs1memC] at hshape2
    simp only [wshape, List.map_cons, List.map_nil,
      List.cons.injEq] at hshape2
    obtain ⟨h1, h2, -⟩ := hshape2
    simp only [Prod.mk.injEq] at h2
    exact ⟨h2.1, h2.2.1, h2.2.2⟩
  have hstkrw : (X86Env.initStack top [argv0, path]).1.rw = true := rfl
  have hroom1 : StackRoom s1 1 := by
    intro sp hsp i hi v
    have hsp_eq : sp = (X86Env.initStack top [argv0, path]).2 + 24 :=
      Option.some.inj (hsp.symm.trans hs1r4)
    have hi0 : i = 0 := by omega
    subst hi0
    rw [hsp_eq]
    have hmod : ((X86Env.initStack top [argv0, path]).2 + 24 + W - 8 * (0 + 1)) % W = (X86Env.initStack top [argv0, path]).2 + 16 := by
      have hh : (X86Env.initStack top [argv0, path]).2 + 24 + W - 8 * (0 + 1) = ((X86Env.initStack top [argv0, path]).2 + 16) + W := by
        omega
      rw [hh, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    rw [hmod, hs1memC]
    have hfind : ∀ a : Nat, stk1.base ≤ a → a < stk1.hi →
        Mem.find [⟨base, stage1Code, false⟩, stk1] a 0 =
          some (1, stk1) := by
      intro a h1 h2
      have hg : Mem.find [⟨base, stage1Code, false⟩, stk1] a 0 =
          (if (⟨base, stage1Code, false⟩ : Region).base ≤ a ∧
              a < (⟨base, stage1Code, false⟩ : Region).hi then
              some (0, ⟨base, stage1Code, false⟩)
            else if stk1.base ≤ a ∧ a < stk1.hi then
              some (1, stk1)
            else Mem.find [] a 2) := rfl
      have hc1b : (⟨base, stage1Code, false⟩ : Region).base =
          base := rfl
      have hc1hi : (⟨base, stage1Code, false⟩ : Region).hi =
          base + stage1Code.length := by simp only [Region.hi]
      rw [hg]
      rw [if_neg, if_pos ⟨h1, h2⟩]
      rw [hc1b, hc1hi]
      rw [hstk1.1] at h1
      omega
    refine wr64_isSome_of_forall_wr (fun j hj u => ?_)
    refine wr_isSome_iff.mpr ⟨1, stk1, ?_, ?_⟩
    · exact hfind _ (by rw [hstk1.1]; omega)
        (by rw [hstk1.2.1]; omega)
    · rw [hstk1.2.2]
      exact hstkrw
  have hregs1 : RegsOk s1 := run_regs_ok hrun1 hregs0
  have hadm1 : OracleAdmits s1 :=
    oracleAdmits_of_run hadm hrun1
  have hleg1 : OracleLegalAt s1 := hadm1 0 s1 rfl
  have hmem1 : AtMemSyscall s1 :=
    ⟨by rw [hs1rip]; exact hc.dbf, Or.inl hs1r0⟩
  rcases head_cases_of_legal_mmap hleg1 hmem1 hs1r0 hs1r7 hs1r6
      hs1r2 hs1r10 hs1r8 hs1r9 (by decide) with
    ⟨A, rest, horA⟩ | hhead
  · -- placement at A, then the generalized read loop
    have hplace : X86Env.place s1.mem (some (.at A))
        (X86Env.pages 65536) = some (some A) :=
      place_of_legal_mmap hleg1 hmem1 hs1r0 hs1r7 hs1r6 hs1r2
        hs1r10 hs1r8 hs1r9 (by decide) horA
    have hpf := place_at_facts hplace
    have hpg : X86Env.pages 65536 = 65536 :=
      pages_eq_of_mod_zero (by decide)
    obtain ⟨s2, hrun2, hs2rip, hs2r3, hs2r12, hs2r13, hs2r14,
      hs2r4, hs2r0, hs2r10, hs2r8, hs2r9, hs2mem, hs2k, hs2len,
      hs2nh⟩ :=
      loader_mmap1_any hs1rip hs1len hs1r0 hs1r7 hs1r6 hs1r2
        hs1r10 hs1r8 hs1r9 hs1r14 hs1r4
        (run_reg_some_ne11 21 (by decide)
          (stage1Init_reg_ne4 _ _ _ _ _ _ (by decide) (by decide))
          hrun1)
        (by rw [horA]; rfl) hplace (by rw [hpg] at hpf; omega)
    rw [hpg] at hpf
    have hs2mem3 : s2.mem =
        [⟨base, stage1Code, false⟩, stk1,
          ⟨A, List.replicate 65536 0, true⟩] := by
      rw [hs2mem, hs1memC]
      rfl
    have hs2fds : s2.k.fds = [some ⟨f, 0⟩] := by
      have hnx : (Kern.next s1.k).2.fds = s1.k.fds := by
        cases ho2 : s1.k.oracle with
        | nil => simp [Kern.next, ho2]
        | cons c rest => simp [Kern.next, ho2]
      rw [hs2k, hnx, hs1k]
    have hmemc1 : (⟨base, stage1Code, false⟩ : Region) ∈
        s1.mem := by
      rw [hs1memC]
      exact List.mem_cons_self
    have hdisjA1 : (⟨base, stage1Code, false⟩ : Region).hi ≤ A ∨
        A + 65536 ≤ base := by
      rcases free_spec hpf.2.2.2 ⟨base, stage1Code, false⟩
          hmemc1 with
        h | h
      · exact Or.inr h
      · exact Or.inl h
    have hmemstk1 : stk1 ∈ s1.mem := by
      rw [hs1memC]
      exact List.mem_cons_of_mem _ List.mem_cons_self
    have hdisjA2 : stk1.hi ≤ A ∨ A + 65536 ≤ stk1.base := by
      rcases free_spec hpf.2.2.2 stk1 hmemstk1 with
        h | h
      · exact Or.inr h
      · exact Or.inl h
    have hrun27 : run stage1Im 27
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.ok s2 := by
      rw [show (27 : Nat) = 21 + 6 by omega, run_add stage1Im 21 6,
        hrun1, hrun2]
    have hnh27 : NoHitUpTo
        (stage1Init f path argv0 stdin oracle top) 27 :=
      noHitUpTo_add (show NoHitUpTo
          (stage1Init f path argv0 stdin oracle top) 21 from by
        intro j t hj hr hmem
        obtain ⟨-, ht⟩ := hchar j t hj hr hmem
        exact hs2nh 0 t (Nat.zero_le _)
          (by rw [run_zero, ht]) hmem) hrun1 hs2nh
    have hadm2 : OracleAdmits s2 :=
      oracleAdmits_of_run hadm hrun27
    have hregs2 : RegsOk s2 := run_regs_ok hrun27 hregs0
    rcases loader_read_loop_any f.length rfl hs2rip hs2len hs2r3
        hs2r12 hs2r13 hs2r14 hs2r4 hs2r0 hs2r10 hs2r8 hs2r9
        hs2mem3 rfl rfl (by
          show A + (List.replicate 65536 0).length = A + 65536
          rw [List.length_replicate])
        hdisjA1 hdisjA2 (by simp) hs2fds hfB (Nat.zero_le _)
        (by decide) hpf.1 hpf.2.1 hpf.2.2.1 (by decide)
        (by decide) hadm2 hregs2
        (by
          have hc1hi : (⟨base, stage1Code, false⟩ : Region).hi =
              base + stage1Code.length := by
            simp only [Region.hi]
          rw [hc1hi, hstk1.1]; exact hdisjInit)
        (by rw [hstk1.2.2]; exact hstkrw)
        (by rw [hstk1.1]; omega)
        (by rw [hstk1.2.1]; exact hple) (by omega : 32 ≤ (X86Env.initStack top [argv0, path]).2 + 24) hsp0W with
      hex | hload
    · obtain ⟨k', hk'b, hrk'⟩ := hex
      exact Or.inl ⟨27 + k', by omega, by
        rw [run_add stage1Im 27 k', hrun27, hrk']⟩
    · obtain ⟨k', A', cap', buf', s', hk'b, hrk', hs'rip, hs'r3,
        hs'r12, hs'r13, hs'r14, hs'r4, hs'r0, hs'r10, hs'r8,
        hs'r9, hs'mem, hbuf'b, hbuf'hi, hbuf'rw, hbuf'data,
        hdisj1', hdisj2', hA'mod, hA'pos, hA'bound, hcap'mod,
        hcap'ge, hs'fds, hs'len, hs'nh⟩ := hload
      have hruntot : run stage1Im (27 + k')
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.ok s' := by
        rw [run_add stage1Im 27 k', hrun27, hrk']
      exact Or.inr ⟨27 + k', A', cap', stk1, buf', s', by omega,
        hruntot, hs'rip, hs'r3, hs'r12, hs'r13, hs'r14, hs'r4,
        hs'r0, hs'r10, hs'r8, hs'r9, hs'mem, hstk1.1, hstk1.2.1,
        (by rw [hstk1.2.2]; exact hstkrw), hbuf'b, hbuf'hi,
        hbuf'rw, hbuf'data, hdisj1', hdisj2', hA'mod, hA'pos,
        hA'bound, hcap'mod, hcap'ge, hs'fds, hs'len,
        run_regs_ok hruntot hregs0,
        noHitUpTo_add hnh27 hrun27 hs'nh⟩
  · -- nomem at the initial mmap: exit 2 in 9 more steps
    have hsp32 : 32 ≤ (X86Env.initStack top [argv0, path]).2 + 24 := by omega
    have hex := loader_mmap1_nomem_exit2_bnd hs1rip hs1len hs1r0
      hs1r7 hs1r6 hs1r2 hs1r10 hs1r8 hs1r9 hs1r4 hhead hregs1
      hroom1 hsp32 hsp0W
    exact Or.inl ⟨30, by omega, by
      rw [show (30 : Nat) = 21 + 9 by omega, run_add stage1Im 21 9,
        hrun1, hex]⟩

theorem parser_mmap1_to_syscall {c nw sp : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ m2 t, run stage1Im 16 (Out.ok s) = Out.ok t ∧
      t.rip = base + 0x21e ∧
      t.reg 0 = some 9 ∧ t.reg 7 = some 0 ∧
      t.reg 6 = some (64 * nw) ∧ t.reg 2 = some 3 ∧
      t.reg 10 = some 34 ∧ t.reg 8 = some (W - 1) ∧
      t.reg 9 = some 0 ∧
      t.mem = m2 ∧ wshape m2 = wshape s.mem ∧
      t.k = s.k ∧ t.regs.length = 16 := by
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
  have hrun16 : run stage1Im 16 (Out.ok s) = Out.ok t16 := by
    rw [show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r1,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r2,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, hrun3,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, hrun4,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, hrun5,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, hrun6,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, hrun7,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, hrun8,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r9,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r10,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r11,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r12,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r13,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r14,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r15, r16]
  exact ⟨m2, t16, hrun16, ht16rip, ht16r0, ht16r7, ht16r6, ht16r2,
    ht16r10, ht16r8, ht16r9, ht16mem, hw2, ht16k, ht16len⟩


theorem parser_mmap2_to_syscall {L sp1 : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ m1 t, run stage1Im 8 (Out.ok s) = Out.ok t ∧
      t.rip = base + 0x249 ∧
      t.reg 0 = some 9 ∧ t.reg 7 = some 0 ∧
      t.reg 6 = some L ∧ t.reg 2 = some 3 ∧
      t.reg 10 = some 34 ∧ t.reg 8 = some (W - 1) ∧
      t.reg 9 = some 0 ∧ t.reg 4 = some sp1 ∧
      t.mem = m1 ∧ wshape m1 = wshape s.mem ∧
      t.k = s.k ∧ t.regs.length = 16 := by
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
  have hrun8 : run stage1Im 8 (Out.ok s) = Out.ok t8 := by
    rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
  exact ⟨m1, t8, hrun8, ht8rip, ht8r0, ht8r7, ht8r6, ht8r2, ht8r10,
    ht8r8, ht8r9, ht8r4, ht8mem, hw1, ht8k, ht8len⟩


theorem rdc_body_ok_nh {c e sp : Nat} {s : X86Int.St}
    (hrip : s.rip = aRDC) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 1)
    (hr4 : s.reg 4 = some sp) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hframe : ∀ m, wr64 s.mem ((sp + W - 8) % W) (aRDC + 32) = some m →
      m.rdN c 4 = s.mem.rdN c 4) :
    ∃ s', run stage1Im 10 (Out.ok s) = Out.ok s' ∧
      s'.rip = aRDC + 0x24 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1, wr64 s.mem (sp - 8) (aRDC + 32) = some m1 ∧
        s'.mem = m1) ∧ NoHitUpTo s 10 := by
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
  have hnohit : NoHitUpTo s 10 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
      j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9 ∨ j = 10) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, haRDC, hc.p271] at h1
      simp at h1
    · have hrj : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
        rw [r1]
      have ht : t = t1 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht1rip, hc.p274] at h1
      simp at h1
    · have hrj : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
      have ht : t = t2 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht2rip, hc.p277] at h1
      simp at h1
    · have hrj : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
      have ht : t = t3 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht3rip, hc.p27e] at h1
      simp at h1
    · have hrj : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
      have ht : t = t4 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht4rip, hc.p282] at h1
      simp at h1
    · have hrj : run stage1Im 5 (Out.ok s) = Out.ok t5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
      have ht : t = t5 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht5rip, hc.p289] at h1
      simp at h1
    · have hrj : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
      have ht : t = t6 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht6rip, hc.p28c] at h1
      simp at h1
    · have hrj : run stage1Im 7 (Out.ok s) = Out.ok t7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
      have ht : t = t7 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hdec : stage1Im t7.rip = some (.ld32 0 2 0, 2) := by
        rw [ht7rip]; exact stage1_l32.d0
      rw [hdec] at h1
      simp at h1
    · have hrj : run stage1Im 8 (Out.ok s) = Out.ok t8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, hrun8]
      have ht : t = t8 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hdec : stage1Im t8.rip = some (.ret, 1) := by
        rw [ht8rip, haL32]; exact hc.p270
      rw [hdec] at h1
      simp at h1
    · have hrj : run stage1Im 9 (Out.ok s) = Out.ok t9 := by
        rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun8, r9]
      have ht : t = t9 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht9rip, hc.p291] at h1
      simp at h1
    · have ht : t = t10 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have ht10rip : t10.rip = base + 0x295 := by
        rw [ht10]
        show t9.rip + 4 = base + 0x295
        rw [ht9rip]
      have h1 := hmem.1
      rw [ht10rip, hc.p295] at h1
      simp at h1
  refine ⟨t10, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    hnohit⟩
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


theorem call_rdc_ok_nh {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5)) (hsiteW : site + 5 < W)
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hdisj : c + 4 ≤ sp - 16 ∨ sp ≤ c) :
    ∃ s', run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 5 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1 m2, wr64 s.mem (sp - 8) (site + 5) = some m1 ∧
        wr64 m1 (sp - 8 - 8) (aRDC + 32) = some m2 ∧ s'.mem = m2) ∧
      NoHitUpTo s 11 := by
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
  obtain ⟨s2, hrunb, hs2rip, hs2r0, hs2r6, hs2r4, hs2fr, hs2len,
      hs2memfr, hs2rd, hs2k, hs2memwr, hs2nh⟩ :=
    rdc_body_ok_nh hs1rip hs1r6 hs1r15 hce hcW heW hs1rd hs1room
      hs1r4 (by omega) (by omega) hs1len hs1frame
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
  have hrun : run stage1Im 12 (Out.ok s) = Out.ok s3 := by
    rw [show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11,
      r1, show (11 : Nat) = 10 + 1 by omega, run_add stage1Im 10 1,
      hrunb, r2]
  have hnh1 : NoHitUpTo s 1 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hsite] at h1
      simp at h1
    · have ht : t = s1 := by
        have h := r1.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hdec : stage1Im s1.rip = some (.mov 2 15, 3) := by
        rw [hs1rip]
        show stage1Im (base + 0x271) = some (.mov 2 15, 3)
        exact hc.p271
      rw [hdec] at h1
      simp at h1
  have hnh : NoHitUpTo s 11 := noHitUpTo_add hnh1 r1 hs2nh
  have hm1fr : ∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x →
      m1.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  refine ⟨s3, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hnh⟩
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

theorem field_load_nh {site B off dst len4 : Nat} {bs : List Nat}
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
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
      NoHitUpTo s 3 := by
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
  have hnohit : NoHitUpTo s 3 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3) with
      rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc1] at h1
      simp at h1
    · have ht : t = t1 := by
        have h := r1.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hdec : stage1Im t1.rip =
          some (.aluI .add 2 off, 4) := by
        rw [ht1rip]
        exact hc2
      rw [hdec] at h1
      simp at h1
    · have hr2 : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r1, r2]
      have ht : t = t2 := by
        have h := hr2.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hdec : stage1Im t2.rip = some (.ld 0 2 0, 3) := by
        rw [ht2rip]
        exact hc3
      rw [hdec] at h1
      simp at h1
    · have hr3 : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
          r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
          r2, r3]
      have ht : t = t3 := by
        have h := hr3.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht3rip, hc4] at h1
      simp at h1
  refine ⟨t4, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht4k, hnohit⟩
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


theorem parser_header_a_nh {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hmagic : leVal (bs.take 4) % 2 ^ 32 = 827478099)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 9 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x15b ∧ s'.reg 15 = some (B + L) ∧
      s'.reg 6 = some B ∧
      (∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
      NoHitUpTo s 9 := by
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
  have hnohit : NoHitUpTo s 9 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨
      j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.p134] at h1
      simp at h1
    · have hrj : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
        rw [r1]
      have ht : t = t1 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht1rip, hc.p137] at h1
      simp at h1
    · have hrj : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
      have ht : t = t2 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht2rip, hc.p13a] at h1
      simp at h1
    · have hrj : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
      have ht : t = t3 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht3rip, hc.p13d] at h1
      simp at h1
    · have hrj : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
      have ht : t = t4 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht4rip, hc.p141] at h1
      simp at h1
    · have hrj : run stage1Im 5 (Out.ok s) = Out.ok t5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
      have ht : t = t5 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht5rip, hc.p148] at h1
      simp at h1
    · have hrj : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
      have ht : t = t6 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht6rip, hc.p14b] at h1
      simp at h1
    · have hrj : run stage1Im 7 (Out.ok s) = Out.ok t7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
      have ht : t = t7 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht7rip, hc.p14d] at h1
      simp at h1
    · have hrj : run stage1Im 8 (Out.ok s) = Out.ok t8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
      have ht : t = t8 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht8rip, hc.p154] at h1
      simp at h1
    · have ht : t = t9 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have hdec : stage1Im t9.rip =
          some (.mov 2 6, 3) := by
        rw [ht9]
        exact hc.p15b
      have h1 := hmem.1
      rw [hdec] at h1
      simp at h1
  refine ⟨t9, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht9k, hnohit⟩
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


theorem parser_header_b_nh {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x15b) (hr6 : s.reg 6 = some B)
    (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hnw : leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 10 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      s'.reg 1 = some (leVal ((bs.drop 12).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some (B + 16) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
      NoHitUpTo s 15 := by
  have hc := stage1_parserCode
  have hBoff : ∀ o : Nat, o ≤ 12 → B + o < W := by
    intro o ho; omega
  -- main := u32 at B+4 (site 0x15b, exit 0x168)
  obtain ⟨s1, hrun1, hs1rip, hs1r12, hs1r6, hs1fr, hs1mem, hs1len, hs1k, hs1nh⟩ :=
    field_load_nh (site := base + 0x15b) (off := 4) (dst := 12) (len4 := 3)
      (by exact hc.p15b)
      (by exact hc.p15e) (by exact hc.p162) (by exact hc.p165)
      hrip hr6 h20 (by decide) (by unfold W; decide) (hBoff 4 (by decide))
      (by decide) (by decide) hlen16
  -- nwords := u32 at B+8 (site 0x168, exit 0x175)
  have hs1rip' : s1.rip = base + 0x168 := by
    rw [hs1rip]
  have h20_1 : s1.mem.rdN B 20 = some bs := by rw [hs1mem]; exact h20
  obtain ⟨s2, hrun2, hs2rip, hs2r10, hs2r6, hs2fr, hs2mem, hs2len, hs2k, hs2nh⟩ :=
    field_load_nh (site := base + 0x168) (off := 8) (dst := 10) (len4 := 3)
      (by exact hc.p168) (by exact hc.p16b) (by exact hc.p16f)
      (by exact hc.p172)
      hs1rip' hs1r6 h20_1 (by decide) (by unfold W; decide)
      (hBoff 8 (by decide)) (by decide) (by decide) hs1len
  -- ndata := u32 at B+12 (site 0x175, exit 0x181)
  have hs2rip' : s2.rip = base + 0x175 := by
    rw [hs2rip]
  have h20_2 : s2.mem.rdN B 20 = some bs := by rw [hs2mem]; exact h20_1
  obtain ⟨s3, hrun3, hs3rip, hs3r1, hs3r6, hs3fr, hs3mem, hs3len, hs3k, hs3nh⟩ :=
    field_load_nh (site := base + 0x175) (off := 12) (dst := 1) (len4 := 2)
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
  have hnohit : NoHitUpTo s 15 := by
    intro j t hj hr hmem
    rcases Nat.lt_or_ge j 4 with hja | hja
    · exact hs1nh j t (by omega) hr hmem
    rcases Nat.lt_or_ge j 8 with hjb | hjb
    · have hr' : run stage1Im (j - 4) (Out.ok s1) = Out.ok t := by
        have h := hr
        rw [show j = 4 + (j - 4) by omega, run_add stage1Im 4 _,
          hrun1] at h
        exact h
      by_cases hj4 : j = 4
      · have ht : t = s1 := by
          subst hj4
          simp only [Nat.add_sub_cancel_left] at hr'
          rw [run_zero] at hr'
          simp only [Out.ok.injEq] at hr'
          exact hr'.symm
        rw [ht] at hmem
        have h1 := hmem.1
        rw [hs1rip, hc.p168] at h1
        simp at h1
      · exact hs2nh (j - 4) t (by omega) hr' hmem
    rcases Nat.lt_or_ge j 12 with hjc | hjc
    · have hr8 : run stage1Im 8 (Out.ok s) = Out.ok s2 := by
        rw [show (8 : Nat) = 4 + 4 by omega, run_add stage1Im 4 4,
          hrun1, hrun2]
      have hr' : run stage1Im (j - 8) (Out.ok s2) = Out.ok t := by
        have h := hr
        rw [show j = 8 + (j - 8) by omega, run_add stage1Im 8 _,
          hr8] at h
        exact h
      by_cases hj8 : j = 8
      · have ht : t = s2 := by
          subst hj8
          simp only [Nat.add_sub_cancel_left] at hr'
          rw [run_zero] at hr'
          simp only [Out.ok.injEq] at hr'
          exact hr'.symm
        rw [ht] at hmem
        have h1 := hmem.1
        have hs2rip175 : s2.rip = base + 0x175 := by
          have h := hs2rip
          omega
        rw [hs2rip175, hc.p175] at h1
        simp at h1
      · exact hs3nh (j - 8) t (by omega) hr' hmem
    have hr12 : run stage1Im 12 (Out.ok s) = Out.ok s3 := by
      rw [show (12 : Nat) = 4 + 8 by omega, run_add stage1Im 4 8,
        hrun1, show (8 : Nat) = 4 + 4 by omega, run_add stage1Im 4 4,
        hrun2, hrun3]
    have hr' : run stage1Im (j - 12) (Out.ok s3) = Out.ok t := by
      have h := hr
      rw [show j = 12 + (j - 12) by omega, run_add stage1Im 12 _,
        hr12] at h
      exact h
    have hj3 : j - 12 = 0 ∨ j - 12 = 1 ∨ j - 12 = 2 ∨ j - 12 = 3 := by
      omega
    rcases hj3 with hj3 | hj3 | hj3 | hj3
    · have ht : t = s3 := by
        rw [hj3, run_zero] at hr'
        simp only [Out.ok.injEq] at hr'
        exact hr'.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have hs3rip181 : s3.rip = base + 0x181 := by
        have h := hs3rip'
        omega
      rw [hs3rip181, hc.p181] at h1
      simp at h1
    · have ht : t = t1 := by
        rw [hj3] at hr'
        have h := r10.symm.trans hr'
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht1rip, hc.p185] at h1
      simp at h1
    · have ht : t = t2 := by
        have hrr : run stage1Im 2 (Out.ok s3) = Out.ok t2 := by
          rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
            r10, r11]
        rw [hj3] at hr'
        have h := hrr.symm.trans hr'
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      have ht2rip : t2.rip = base + 0x188 := by
        rw [ht2]
        show t1.rip + 3 = base + 0x188
        rw [ht1rip]
      rw [ht2rip, hc.p188] at h1
      simp at h1
    · have ht : t = t3 := by
        have hrr : run stage1Im 3 (Out.ok s3) = Out.ok t3 := by
          rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2,
            r10, show (2 : Nat) = 1 + 1 by omega,
            run_add stage1Im 1 1, r11, r12]
        rw [hj3] at hr'
        have h := hrr.symm.trans hr'
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have ht3rip2 : t3.rip = base + 0x18f := by rw [ht3]
      have h1 := hmem.1
      rw [ht3rip2, hc.p18f] at h1
      simp at h1
  refine ⟨t3, hrun, ?_, ?_, ?_, ?_, ?_, ht3fr', ?_, ?_, ht3k, hnohit⟩
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

theorem parser_data_step_ok_nh {c e d n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hnW : n < W) (hce : c + 4 ≤ e)
    (hd : rd32 s.mem c = some d) (hcd : c + 4 + d ≤ e)
    (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 16 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧ s'.reg 1 = some (n - 1) ∧
      s'.reg 6 = some (c + 4 + d) ∧ s'.reg 15 = some e ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
      NoHitUpTo s 16 := by
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
  have hnohit : NoHitUpTo s 16 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9 ∨ j = 10 ∨ j = 11 ∨ j = 12 ∨ j = 13 ∨ j = 14 ∨ j = 15 ∨ j = 16) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.p18f] at h1
      simp at h1
    · have hrj : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
        rw [r1]
      have ht : t = t1 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht1rip, hc.p192] at h1
      simp at h1
    · have hrj : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
      have ht : t = t2 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht2rip, hc.p194] at h1
      simp at h1
    · have hrj : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
      have ht : t = t3 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht3rip, hc.p197] at h1
      simp at h1
    · have hrj : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
      have ht : t = t4 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht4rip, hc.p19a] at h1
      simp at h1
    · have hrj : run stage1Im 5 (Out.ok s) = Out.ok t5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
      have ht : t = t5 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht5rip, hc.p19e] at h1
      simp at h1
    · have hrj : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
      have ht : t = t6 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht6rip, hc.p1a5] at h1
      simp at h1
    · have hrj : run stage1Im 7 (Out.ok s) = Out.ok t7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
      have ht : t = t7 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht7rip, hc.p1a7] at h1
      simp at h1
    · have hrj : run stage1Im 8 (Out.ok s) = Out.ok t8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
      have ht : t = t8 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht8rip, hc.p1aa] at h1
      simp at h1
    · have hrj : run stage1Im 9 (Out.ok s) = Out.ok t9 := by
        rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
      have ht : t = t9 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht9rip, hc.p1ae] at h1
      simp at h1
    · have hrj : run stage1Im 10 (Out.ok s) = Out.ok t10 := by
        rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r8, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
      have ht : t = t10 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht10rip, hc.p1b1] at h1
      simp at h1
    · have hrj : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
        rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
      have ht : t = t11 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht11rip, hc.p1b4] at h1
      simp at h1
    · have hrj : run stage1Im 12 (Out.ok s) = Out.ok t12 := by
        rw [show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r1, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r2, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r3, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r4, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r5, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r6, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r7, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r8, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r9, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
      have ht : t = t12 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht12rip, hc.p1b7] at h1
      simp at h1
    · have hrj : run stage1Im 13 (Out.ok s) = Out.ok t13 := by
        rw [show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r1, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r2, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r3, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r4, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r5, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r6, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r7, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r8, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r9, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r10, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r11, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r12, r13]
      have ht : t = t13 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht13rip, hc.p1be] at h1
      simp at h1
    · have hrj : run stage1Im 14 (Out.ok s) = Out.ok t14 := by
        rw [show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r1, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r2, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r3, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r4, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r5, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r6, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r7, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r8, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r9, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r10, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r11, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r12, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r13, r14]
      have ht : t = t14 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht14rip, hc.p1c1] at h1
      simp at h1
    · have hrj : run stage1Im 15 (Out.ok s) = Out.ok t15 := by
        rw [show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r1, show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r2, show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r3, show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r4, show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r5, show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r6, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r7, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r8, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r9, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r10, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r11, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r12, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r13, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r14, r15]
      have ht : t = t15 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht15rip, hc.p1c5] at h1
      simp at h1
    · have ht : t = t16 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have hdec : stage1Im t16.rip = stage1Im (base + 0x18f) := by
        rw [ht16]
      have h1 := hmem.1
      rw [hdec, hc.p18f] at h1
      simp at h1
  refine ⟨t16, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ht16k, hnohit⟩
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

theorem parser_guard_ok_nh {c e nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1c7) (hr10 : s.reg 10 = some nw)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hnw32 : nw < 2 ^ 32) (hce : c ≤ e) (heW : e < W)
    (hguard : 20 * nw ≤ e - c) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x1ec ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
      NoHitUpTo s 11 := by
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
  have hnohit : NoHitUpTo s 11 := by
    intro j t hj hr hmem
    rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8 ∨ j = 9 ∨ j = 10 ∨ j = 11) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [run_zero] at hr
      simp only [Out.ok.injEq] at hr
      subst hr
      have h1 := hmem.1
      rw [hrip, hc.p1c7] at h1
      simp at h1
    · have hrj : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
        rw [r1]
      have ht : t = t1 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht1rip, hc.p1ca] at h1
      simp at h1
    · have hrj : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
        rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
      have ht : t = t2 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht2rip, hc.p1cd] at h1
      simp at h1
    · have hrj : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
        rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
      have ht : t = t3 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht3rip, hc.p1d0] at h1
      simp at h1
    · have hrj : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
        rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
      have ht : t = t4 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht4rip, hc.p1d3] at h1
      simp at h1
    · have hrj : run stage1Im 5 (Out.ok s) = Out.ok t5 := by
        rw [show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r1, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r2, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r3, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r4, r5]
      have ht : t = t5 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht5rip, hc.p1d6] at h1
      simp at h1
    · have hrj : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
        rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
      have ht : t = t6 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht6rip, hc.p1d9] at h1
      simp at h1
    · have hrj : run stage1Im 7 (Out.ok s) = Out.ok t7 := by
        rw [show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r1, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r2, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r3, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r4, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r6, r7]
      have ht : t = t7 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht7rip, hc.p1dc] at h1
      simp at h1
    · have hrj : run stage1Im 8 (Out.ok s) = Out.ok t8 := by
        rw [show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r1, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r2, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r3, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r4, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r5, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r7, r8]
      have ht : t = t8 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht8rip, hc.p1df] at h1
      simp at h1
    · have hrj : run stage1Im 9 (Out.ok s) = Out.ok t9 := by
        rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
      have ht : t = t9 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht9rip, hc.p1e2] at h1
      simp at h1
    · have hrj : run stage1Im 10 (Out.ok s) = Out.ok t10 := by
        rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1, show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2, show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3, show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4, show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5, show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6, show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7, show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r8, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
      have ht : t = t10 := by
        have h := hrj.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have h1 := hmem.1
      rw [ht10rip, hc.p1e5] at h1
      simp at h1
    · have ht : t = t11 := by
        have h := hrun.symm.trans hr
        simp only [Out.ok.injEq] at h
        exact h.symm
      rw [ht] at hmem
      have hdec : stage1Im t11.rip = stage1Im (base + 0x1ec) := by
        rw [ht11]
      have h1 := hmem.1
      rw [hdec, hc.p1ec] at h1
      simp at h1
  refine ⟨t11, hrun, ?_, ?_, ?_, ?_, ht11k, hnohit⟩
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
theorem parser_data_loop_ok_nh {e : Nat} :
    ∀ {c : Nat} {ds : List Nat} {c' : Nat} {s : X86Int.St},
      DataSeq s.mem c ds c' → s.rip = base + 0x18f →
      s.reg 1 = some ds.length → s.reg 6 = some c →
      s.reg 15 = some e → c' ≤ e → e < W → ds.length < W →
      s.regs.length = 16 →
      ∃ n, n ≤ 16 * (ds.length + 1) ∧ ∃ s',
        run stage1Im n (Out.ok s) = Out.ok s' ∧
        s'.rip = base + 0x1c7 ∧ s'.reg 1 = some 0 ∧
        s'.reg 6 = some c' ∧ s'.reg 15 = some e ∧
        (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
        s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k ∧
        NoHitUpTo s n := by
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
    have hnohit : NoHitUpTo s 2 := by
      intro j t hj hr hmem
      rcases (by omega : j = 0 ∨ j = 1 ∨ j = 2) with rfl | rfl | rfl
      · rw [run_zero] at hr
        simp only [Out.ok.injEq] at hr
        subst hr
        have h1 := hmem.1
        rw [hrip, hc.p18f] at h1
        simp at h1
      · have hrj : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
          rw [r1]
        have ht : t = t1 := by
          have h := hrj.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have h1 := hmem.1
        rw [ht1rip, hc.p192] at h1
        simp at h1
      · have hrj : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
          rw [hrun]
        have ht : t = t2 := by
          have h := hrj.symm.trans hr
          simp only [Out.ok.injEq] at h
          exact h.symm
        rw [ht] at hmem
        have h1 := hmem.1
        have ht2rip : t2.rip = base + 0x1c7 := by rw [ht2]
        rw [ht2rip, hc.p1c7] at h1
        simp at h1
    refine ⟨2, by omega, t2, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
      ht2k, hnohit⟩
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
          hs1len, hs1k, hnh1⟩ :=
        parser_data_step_ok_nh hrip hr1 hr6 hr15
          (by rw [hlencons]; omega) hlenW
          (by omega) hd hle heW hlen16
      have hs1r1' : s1.reg 1 = some ds.length := by
        rw [hlencons] at hs1r1
        simpa using hs1r1
      have hseq1 : DataSeq s1.mem (c + 4 + d) ds c' := by
        rw [hs1mem]; exact hrest
      obtain ⟨n2, hn2bnd, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r15,
          hs2fr, hs2mem, hs2len, hs2k, hnh2⟩ :=
        ih hseq1 hs1rip hs1r1' hs1r6 hs1r15 hc'e heW
          (by rw [hlencons] at hlenW; omega) hs1len
      have hrun : run stage1Im (16 + n2) (Out.ok s) = Out.ok s2 := by
        rw [run_add stage1Im 16 n2, hrun1, hrun2]
      refine ⟨16 + n2, ?_, s2, hrun, hs2rip, hs2r1, hs2r6, hs2r15,
        fun i hi0 hi1 hi2 hi6 =>
          (hs2fr i hi0 hi1 hi2 hi6).trans (hs1fr i hi0 hi1 hi2 hi6),
        hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k,
        noHitUpTo_add hnh1 hrun1 hnh2⟩
      rw [hlencons]; omega

/-- Parser front (header, data loop, guard) with step bound and
`NoHitUpTo` companion, from the loaded state at `0x134` to the
words-table syscall site state at `0x1ec`. -/
theorem parser_front_to_mmap1_nh {f path argv0 : List Nat}
    {top A : Nat} {pr : Prog} {s0 : X86Int.St}
    (hdec : Sporos.decode f = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfB : ∀ b ∈ f, b < 256)
    (htop47 : top ≤ 2 ^ 47)
    (hstkhi : (X86Env.initStack top [argv0, path]).1.hi = top)
    (hrip0 : s0.rip = base + 0x134)
    (hr3 : s0.reg 3 = some A) (hr13 : s0.reg 13 = some f.length)
    (hr4 : s0.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24))
    (hregs0 : RegsOk s0)
    (hmem0 : ∃ c1 c2 buf', s0.mem = [c1, c2, buf'] ∧
      c1.base = base ∧ c1.hi = base + stage1Code.length ∧
      c2.base = (X86Env.initStack top [argv0, path]).1.base ∧
      c2.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
      buf'.base = A ∧ (∃ cap', buf'.hi = A + cap' ∧
        A + cap' ≤ 2 ^ 47) ∧ buf'.rw = true ∧ c2.rw = true ∧
      buf'.data.take f.length = f)
    (hrdN : s0.mem.rdN A f.length = some f)
    (hlen16 : s0.regs.length = 16)
    (hroom10 : StackRoom s0 10) :
    ∃ N, N ≤ 35 + 16 * (pr.data.length + 1) ∧ ∃ s4,
      run stage1Im N (Out.ok s0) = Out.ok s4 ∧
      s4.rip = base + 0x1ec ∧
      s4.reg 6 = some (A + 16 + dataBytes pr.data) ∧
      s4.reg 10 = some pr.words.length ∧
      s4.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24) ∧
      s4.reg 3 = some A ∧ s4.reg 12 = some pr.main ∧
      s4.reg 13 = some f.length ∧
      s4.reg 15 = some (A + f.length) ∧
      s4.mem = s0.mem ∧ s4.k = s0.k ∧ s4.regs.length = 16 ∧
      RegsOk s4 ∧ StackRoom s4 2 ∧
      (∃ v, s4.reg 7 = some v) ∧ (∃ v, s4.reg 9 = some v) ∧
      NoHitUpTo s0 N := by
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
    ⟨cap', hbufhi, hbufcap⟩, hbufrw, hc2rw, htake⟩ := hmem0
  have hAe : A + f.length < W := by
    have hW : W = 2 ^ 64 := rfl
    have hcapE : cap' = buf'.data.length := by
      have e : buf'.hi = buf'.base + buf'.data.length := rfl
      rw [hbufb, hbufhi] at e
      omega
    have hflen : f.length ≤ cap' := by
      have h1 : (buf'.data.take f.length).length = f.length := by
        rw [htake]
      rw [List.length_take] at h1
      omega
    omega
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
  -- header_a
  obtain ⟨s1, hrun1, hs1rip, hs1r15, hs1r6, hs1fr, hs1mem,
    hs1len, hs1k, hnh1⟩ :=
    parser_header_a_nh hrip0 hr3 hr13 hAe hL16 h20 hmagic hlen16
  have hs1r4 : s1.reg 4 = s0.reg 4 :=
    hs1fr 4 (by decide) (by decide) (by decide)
  have h20_1 : s1.mem.rdN A 20 = some (pr.enc.take 20) := by
    rw [hs1mem]; exact h20
  -- header_b
  obtain ⟨s2, hrun2, hs2rip, hs2r12, hs2r10, hs2r1, hs2r6, hs2fr,
    hs2mem, hs2len, hs2k, hnh2⟩ :=
    parser_header_b_nh hs1rip hs1r6 hAe hL16 h20_1
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
  obtain ⟨n3, hn3bnd, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3r15, hs3fr,
    hs3mem, hs3len, hs3k, hnh3⟩ :=
    parser_data_loop_ok_nh hdataseq hs2rip
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
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k, hnh4⟩ :=
    parser_guard_ok_nh hs3rip hs3r10 hs3r6 hs3r15 hbnw (by omega) hAe
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
    obtain ⟨v, hv⟩ := hregs0 7 (by decide) (by decide)
    exact ⟨v, by
      rw [hs4fr 7 (by decide) (by decide),
        hs3fr 7 (by decide) (by decide) (by decide) (by decide),
        hs2fr 7 (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
        hs1fr 7 (by decide) (by decide) (by decide)]
      exact hv⟩
  have hs4r9 : ∃ v, s4.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hregs0 9 (by decide) (by decide)
    exact ⟨v, by
      rw [hs4fr 9 (by decide) (by decide),
        hs3fr 9 (by decide) (by decide) (by decide) (by decide),
        hs2fr 9 (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
        hs1fr 9 (by decide) (by decide) (by decide)]
      exact hv⟩
  have hNH : NoHitUpTo s0 (9 + (15 + (n3 + 11))) :=
    noHitUpTo_add hnh1 hrun1
      (noHitUpTo_add hnh2 hrun2
        (noHitUpTo_add hnh3 hrun3 hnh4))
  have hrunT : run stage1Im (9 + (15 + (n3 + 11))) (Out.ok s0) =
      Out.ok s4 := by
    rw [run_add stage1Im 9 (15 + (n3 + 11)), hrun1,
      show (15 + (n3 + 11) : Nat) = 15 + (n3 + 11) from rfl,
      run_add stage1Im 15 (n3 + 11), hrun2,
      run_add stage1Im n3 11, hrun3, hrun4]
  have hregs4 : RegsOk s4 :=
    run_regs_ok hrun4 (run_regs_ok hrun3 (run_regs_ok hrun2
      (run_regs_ok hrun1 hregs0)))
  have hs4r4' : s4.reg 4 =
      some ((X86Env.initStack top [argv0, path]).2 + 24) := by
    rw [hs4r4]; exact hr4
  have hn3' : n3 ≤ 16 * (pr.data.length + 1) := by
    simpa [List.length_map] using hn3bnd
  refine ⟨9 + (15 + (n3 + 11)), by omega, s4, hrunT, hs4rip, hs4r6,
    hs4r10, hs4r4', hs4r3, hs4r12, hs4r13, hs4r15, hmem4, hk4,
    hs4len, hregs4, hroom4, hs4r7, hs4r9, hNH⟩

end SporosX1
