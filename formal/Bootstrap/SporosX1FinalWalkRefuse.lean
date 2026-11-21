import Bootstrap.SporosX1FinalChkRefuse

/-!
# Sporos stage 1: bounded refusal for the code walk and opcode handlers

Bounded `Out.exit 1` refusal lemmas for every runtime failure check in the
single-pass walk (`tor`, `fromr`, `rfetch`, `pick`, `locget`, `locset`, `prim`,
`call`, `ret`, `jmp`, `jz`/`jnz`, and `fall_tail`), composed into `walk_refuseB`:
whenever `Sporos.walk pr wd code i cur = false`, machine execution from `WalkInv`
exits `1` within `(code.length + 1) * (200 + 60 * (wd.ann.length + 1))` steps.
-/

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-- Stack room at a lowered `rsp`, transported from 4 slots at the original `rsp`. -/
theorem StackRoom_shift4 {s u : X86Int.St} {sp L k : Nat}
    (hroom : StackRoom s 4) (hs4 : s.reg 4 = some sp)
    (hu4 : u.reg 4 = some (sp - L)) (hwsh : wshape u.mem = wshape s.mem)
    (hL : ∃ m, L = 8 * m ∧ L + 8 * k ≤ 32) (hsp : 32 ≤ sp) (hspW : sp < W) :
    StackRoom u k := by
  obtain ⟨m, rfl, hm⟩ := hL
  intro sp' hsp' i hi v
  have hsp'e : sp' = sp - 8 * m := by
    rw [hu4] at hsp'; exact Option.some.inj hsp'.symm
  subst hsp'e
  have e : ((sp - 8 * m) + W - 8 * (i + 1)) % W = sp - 8 * (m + i + 1) := by
    rw [slot_sub (by omega) (by omega)]; omega
  rw [e]
  have hh := hroom sp hs4 (m + i) (by omega) v
  have e2 : (sp + W - 8 * (m + i + 1)) % W = sp - 8 * (m + i + 1) :=
    slot_sub (by omega) hspW
  rw [e2] at hh
  exact wrN_some_of_wshape _ s.mem u.mem hwsh _ hh

/-- Shared 2-step `cmpI r 1` + not-taken `jcc 3 (entry + 11)` -> `jmp aFail1` at
`entry + 6` when `s.reg r = some 0`, exiting `1` in at most 8 steps. -/
private theorem run_cmp1_fail {im : Nat → Option (Ins × Nat)}
    (rc : RefuseCode im) {entry r sp : Nat}
    (h0 : im entry = some (.cmpI r 1, 4))
    (h1 : im (entry + 4) = some (.jcc 3 (entry + 11), 2))
    (hjmp : im (entry + 6) = some (.jmp aFail1, 5))
    (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = entry → s.reg r = some 0 → s.reg 4 = some sp →
      RegsOk s → StackRoom s 1 →
      ∃ m, m ≤ 8 ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hrip hr hr4 hregs hroom
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .sub 0 (1 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact h0) hr
  obtain ⟨s1, hs1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .sub 0 (1 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have hs1rip : s1.rip = entry + 4 := by rw [hs1]; show s.rip + 4 = entry + 4; rw [hrip]
  have hs1fl : X86Int.cond s1.fl 3 = some false := by
    have hcf : s1.fl.cf = some true := by
      rw [hs1]; exact aluF_sub_cf_true (by unfold W; decide)
    rw [cond_cc3 hcf]; rfl
  have e2 : X86Int.step im s1 = .ok { s1 with rip := entry + 6 } := by
    have hh := step_jcc (im := im) (s := s1) (by rw [hs1rip]; exact h1) hs1fl
    simpa [hs1rip] using hh
  obtain ⟨s2, hs2⟩ : ∃ u : X86Int.St, u = { s1 with rip := entry + 6 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok s1 := by rw [run_succ_ok im 0 s, e1, ← hs1, run_zero]
  have r2 : run im 1 (Out.ok s1) = Out.ok s2 := by rw [run_succ_ok im 0 s1, e2, ← hs2, run_zero]
  have hr2 : run im 2 (Out.ok s) = Out.ok s2 := by rw [run_add im 1 1, r1, r2]
  have hs2rip : s2.rip = entry + 6 := by rw [hs2]
  have hs2r4 : s2.reg 4 = some sp := by rw [hs2, hs1]; exact hr4
  have hs2mem : s2.mem = s.mem := by rw [hs2, hs1]
  obtain ⟨k, hk, hrun⟩ := jmp_fail1_exit1B rc hjmp hsp32 hspW s2 hs2rip hs2r4
    (run_regs_ok hr2 hregs) (StackRoom_of_mem hs2mem (by rw [hs2r4, hr4]) hroom)
  exact ⟨2 + k, (by omega), by rw [run_add im 2 k, hr2]; exact hrun⟩

/-- Shared 3-instruction immediate-load prefix (`mov rdx, rsi`; `add rdx, 1`; `call l32`)
reading the 32-bit immediate `n` at `c + 1` into `rax` in at most 6 steps. -/
private theorem run_load_imm32 {im : Nat → Option (Ins × Nat)}
    {entry l32 c n sp : Nat} (hl : L32Code im l32)
    (hdret : im (l32 + 2) = some (.ret, 1))
    (hd0 : im entry = some (.mov 2 6, 3))
    (hd1 : im (entry + 3) = some (.aluI .add 2 1, 4))
    (hd2 : im (entry + 7) = some (.call l32, 5))
    (hcW : c + 1 < W) (hsp8 : 8 ≤ sp) (hspW : sp < W) (hentry12 : entry + 12 < W)
    (hdisj : c + 5 ≤ sp - 8 ∨ sp ≤ c + 1) :
    ∀ s, s.rip = entry → s.reg 6 = some c → s.reg 4 = some sp →
      16 ≤ s.regs.length → (s.mem.rdN (c + 1) 4).map leVal = some n →
      StackRoom s 1 →
      ∃ m s', m ≤ 6 ∧ run im m (Out.ok s) = Out.ok s' ∧
        s'.rip = entry + 12 ∧ s'.reg 0 = some n ∧ s'.reg 4 = some sp ∧
        (∀ q, q ≠ 0 → q ≠ 2 → q ≠ 4 → s'.reg q = s.reg q) ∧
        (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  intro s hrip hr6 hr4 hlen hrd hroom
  have h1W : 1 < W := by unfold W; decide
  have hadd1 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W (by omega)
  have e1 : X86Int.step im s = .ok { s.set 2 (some c) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hd0) hr6
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some c) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 3 := by rw [ha1]; show s.rip + 3 = entry + 3; rw [hrip]
  have ha1r2 : a1.reg 2 = some c := by
    rw [ha1]; show (s.set 2 _).reg 2 = some c; rw [reg_set_self (by omega)]
  have ga1 : ∀ q, q ≠ 2 → a1.reg q = s.reg q := by
    intro q hq; rw [ha1]; show (s.set 2 _).reg q = s.reg q; rw [reg_set_other hq]
  have ha1mem : a1.mem = s.mem := by rw [ha1]; exact rfl
  have ha1len : a1.regs.length = s.regs.length := by
    rw [ha1]; show (s.set 2 _).regs.length = s.regs.length; exact List.length_set
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add c (1 % W)).1) with fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hd1) ha1r2
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add c (1 % W)).1) with fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 4 = entry + 7; rw [ha1rip]
  have ha2len : a2.regs.length = s.regs.length := by
    rw [ha2]; show (a1.set 2 _).regs.length = s.regs.length; exact List.length_set.trans ha1len
  have ha2r2 : a2.reg 2 = some (c + 1) := by
    rw [ha2]; show (a1.set 2 _).reg 2 = some (c + 1); rw [reg_set_self (by omega), hadd1]
  have ga2 : ∀ q, q ≠ 2 → a2.reg q = s.reg q := by
    intro q hq; rw [ha2]; show (a1.set 2 _).reg q = s.reg q; rw [reg_set_other hq]; exact ga1 q hq
  have ha2r4 : a2.reg 4 = some sp := by rw [ga2 4 (by decide)]; exact hr4
  have ha2mem : a2.mem = s.mem := by rw [ha2]; exact ha1mem
  obtain ⟨k, a3, hk, hrun3, ha3rip, ha3r0, ha3r4, ga3, ha3len, hfr3, hw3⟩ :=
    CallL32B hl hdret hd2 hcW hsp8 hspW (by omega) hdisj a2 ha2rip ha2r2 ha2r4
      (by rw [ha2mem]; exact hrd) (StackRoom_of_mem ha2mem (by rw [ha2r4, hr4]) hroom)
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have hr2 : run im 2 (Out.ok s) = Out.ok a2 := by rw [run_add im 1 1, r1, r2]
  refine ⟨2 + k, a3, (by omega), by rw [run_add im 2 k, hr2]; exact hrun3,
    (by omega), ha3r0, ha3r4, ?_, (by rw [← ha2mem]; exact hfr3), (by rw [← ha2mem]; exact hw3),
    ha3len.trans ha2len⟩
  intro q h0 h2 h4
  rw [ga3 q h0 h4, ga2 q h2]

/-- Single `add rdx, rdx` step doubling `reg 2` when `2 * v < W`. -/
private theorem run_double_r2 {im : Nat → Option (Ins × Nat)}
    {pc v : Nat} (hi : im pc = some (.alu .add 2 2, 3)) :
    ∀ s, s.rip = pc → s.reg 2 = some v → v + v < W → 16 ≤ s.regs.length →
      ∃ s', run im 1 (Out.ok s) = Out.ok s' ∧ s'.rip = pc + 3 ∧
        s'.reg 2 = some (2 * v) ∧ (∀ q, q ≠ 2 → s'.reg q = s.reg q) ∧
        s'.mem = s.mem ∧ s'.regs.length = s.regs.length := by
  intro s hrip hr2 h2v hlen
  have hdbl : (aluF .add v v).1 = 2 * v := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt h2v]; omega
  have e1 : X86Int.step im s =
      .ok { s.set 2 (some (aluF .add v v).1) with fl := (aluF .add v v).2, rip := s.rip + 3 } :=
    step_alu_add (by rw [hrip]; exact hi) hr2 hr2
  obtain ⟨s', hs'⟩ : ∃ u : X86Int.St, u =
    { s.set 2 (some (aluF .add v v).1) with fl := (aluF .add v v).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok s' := by rw [run_succ_ok im 0 s, e1, ← hs', run_zero]
  refine ⟨s', r1, by rw [hs']; show s.rip + 3 = pc + 3; rw [hrip], ?_, ?_, ?_, ?_⟩
  · rw [hs']; show (s.set 2 _).reg 2 = some (2 * v); rw [reg_set_self (by omega), hdbl]
  · intro q hq; rw [hs']; show (s.set 2 _).reg q = s.reg q; rw [reg_set_other hq]
  · rw [hs']; exact rfl
  · rw [hs']; show (s.set 2 _).regs.length = s.regs.length; exact List.length_set

/-- `tor` refuses when `hm.d = 0`. -/
theorem hand_tor_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.tor :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.tor = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 50 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have h1b : s.mem.rdN c 1 = some [131] := inv.csq.1
  have hbyte : s.mem.rd c = some 131 := (rdN_cons h1b).1
  have hd0 : hm.d = 0 := by
    by_cases h1 : 1 ≤ hm.d
    · simp [hstep, h1] at hs
    · omega
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, -, -, hs1r8, -, hs1r4, -, hs1mem⟩ :=
    WalkTor_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hs1r8' : s1.reg 8 = some 0 := by rw [hs1r8, hd0]
  obtain ⟨k2, hk2, hr2⟩ := run_cmp1_fail rc cfg.tor.d0 cfg.tor.d1 cr.torUnd st.sp32 st.spW
    s1 hrip1 hs1r8' hs1r4 (run_regs_ok hr1 hregs)
    (StackRoom_mono (StackRoom_of_mem hs1mem (by rw [hs1r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + k2, (by omega), by rw [run_add im k1 k2, hr1]; exact hr2⟩

/-- `fromr` refuses when `hm.a = 0`. -/
theorem hand_fromr_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.fromr :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.fromr = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 50 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have h1b : s.mem.rdN c 1 = some [132] := inv.csq.1
  have hbyte : s.mem.rd c = some 132 := (rdN_cons h1b).1
  have ha0 : hm.a = 0 := by
    by_cases h1 : 1 ≤ hm.a
    · simp [hstep, h1] at hs
    · omega
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, -, -, -, hs1r9, hs1r4, -, hs1mem⟩ :=
    WalkFromr_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hs1r9' : s1.reg 9 = some 0 := by rw [hs1r9, ha0]
  obtain ⟨k2, hk2, hr2⟩ := run_cmp1_fail rc cfg.fromr.d0 cfg.fromr.d1 cr.fromrUnd st.sp32 st.spW
    s1 hrip1 hs1r9' hs1r4 (run_regs_ok hr1 hregs)
    (StackRoom_mono (StackRoom_of_mem hs1mem (by rw [hs1r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + k2, (by omega), by rw [run_add im k1 k2, hr1]; exact hr2⟩

/-- `rfetch` refuses when `hm.a = 0`. -/
theorem hand_rfetch_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.rfetch :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.rfetch = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 50 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have h1b : s.mem.rdN c 1 = some [133] := inv.csq.1
  have hbyte : s.mem.rd c = some 133 := (rdN_cons h1b).1
  have ha0 : hm.a = 0 := by
    by_cases h1 : 1 ≤ hm.a
    · simp [hstep, h1] at hs
    · omega
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, -, -, -, hs1r9, hs1r4, -, hs1mem⟩ :=
    WalkRfetch_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hs1r9' : s1.reg 9 = some 0 := by rw [hs1r9, ha0]
  obtain ⟨k2, hk2, hr2⟩ := run_cmp1_fail rc cfg.rfetch.d0 cfg.rfetch.d1 cr.rfetchUnd st.sp32 st.spW
    s1 hrip1 hs1r9' hs1r4 (run_regs_ok hr1 hregs)
    (StackRoom_mono (StackRoom_of_mem hs1mem (by rw [hs1r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + k2, (by omega), by rw [run_add im k1 k2, hr1]; exact hr2⟩

/-- `pick n` refuses when `hm.d ≤ n`. -/
theorem hand_pick_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.pick n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.pick n) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 60 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hnle : hm.d ≤ n := by
    by_cases hn : n < hm.d
    · simp [hstep, hn] at hs
    · omega
  have himmok : n < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x82 :: leBytes n 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 130 := (rdN_cons h5).1
  have hc5W : c + 5 < W := c_add_len_lt inv rfl
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkPick_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, -, hw2, _⟩ :=
    run_load_imm32 cfg.l32 cfg.pick.dret cfg.pick.d0 cfg.pick.d1 cfg.pick.d2
      (by omega) (by have := st.sp32; omega) st.spW (by decide) (disjC5_of_inv inv rfl).2.symm
      s1 hrip1 hr61 hr41 hlen1 himm
      (StackRoom_mono (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room) (by decide))
  have hs2r8 : s2.reg 8 = some hm.d := by rw [gs2 8 (by decide) (by decide) (by decide)]; exact hr81
  have e3 : X86Int.step im s2 = .ok { s2 with fl := (aluF .sub n hm.d).2, rip := s2.rip + 3 } :=
    step_cmp (by rw [hrip2]; exact cfg.pick.d3) hs2r0 hs2r8
  obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2 with fl := (aluF .sub n hm.d).2, rip := s2.rip + 3 } := ⟨_, rfl⟩
  have hs3rip : s3.rip = aPick + 15 := by rw [hs3]; show s2.rip + 3 = aPick + 15; rw [hrip2]
  have hs3fl : X86Int.cond s3.fl 2 = some false := by
    have hcf : s3.fl.cf = some false := by rw [hs3]; exact aluF_sub_cf (by omega)
    exact cond_cc2 hcf
  have e4 : X86Int.step im s3 = .ok { s3 with rip := aPick + 17 } := by
    have hh := step_jcc (im := im) (s := s3) (by rw [hs3rip]; exact cfg.pick.d4) hs3fl
    simpa [hs3rip] using hh
  obtain ⟨s4, hs4⟩ : ∃ u : X86Int.St, u = { s3 with rip := aPick + 17 } := ⟨_, rfl⟩
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
  have hrun4 : run im (k1 + k2 + 2) (Out.ok s) = Out.ok s4 := by
    rw [run_add im (k1 + k2) 2, run_add im k1 k2, hr1, hr2, run_add im 1 1, r3, r4]
  have hs4rip : s4.rip = aPick + 17 := by rw [hs4]
  have hs4r4 : s4.reg 4 = some sp := by rw [hs4, hs3]; exact hs2r4
  have hws4 : wshape s4.mem = wshape s.mem := by rw [hs4, hs3, hw2, hm1]
  obtain ⟨k5, hk5, hr5⟩ := jmp_fail1_exit1B rc cr.pickOob st.sp32 st.spW s4 hs4rip hs4r4
    (run_regs_ok hrun4 hregs)
    (StackRoom_mono (StackRoom_of_wshape hws4 (by rw [hs4r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + k2 + 2 + k5, (by omega), by rw [run_add im (k1 + k2 + 2) k5, hrun4]; exact hr5⟩

/-- `locget n` refuses when `wd.locals ≤ n`. -/
theorem hand_locget_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.locget n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.locget n) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 60 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hnle : wd.locals ≤ n := by
    by_cases hn : n < wd.locals
    · simp [hstep, hn] at hs
    · omega
  have himmok : n < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x86 :: leBytes n 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 134 := (rdN_cons h5).1
  have hc5W : c + 5 < W := c_add_len_lt inv rfl
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkLocget_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, hfr2, hw2, hlen2⟩ :=
    run_load_imm32 cfg.l32 cfg.locget.dret cfg.locget.d0 cfg.locget.d1 cfg.locget.d2
      (by omega) (by have := st.sp32; omega) st.spW (by decide) (disjC5_of_inv inv rfl).2.symm
      s1 hrip1 hr61 hr41 hlen1 himm
      (StackRoom_mono (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room) (by decide))
  have hs2r7 : s2.reg 7 = some w := by rw [gs2 7 (by decide) (by decide) (by decide)]; exact hr71
  have hs2r9 : s2.reg 9 = some hm.a := by rw [gs2 9 (by decide) (by decide) (by decide)]; exact hr91
  have hdisjw8 : sp ≤ w + 16 ∨ w + 24 ≤ sp - 8 := by
    have h32 := st.sp32; cases st.disjw with | inl h => exact Or.inl (by omega) | inr h => exact Or.inr (by omega)
  have hfloc2 : rd64 s2.mem (w + 16) = some wd.locals := by
    unfold rd64; rw [hfr2 (w + 16) 8 (by cases hdisjw8 with | inl h => exact Or.inr h | inr h => exact Or.inl h), hm1]
    exact inv.floc
  have hea16 : ea w 16 = w + 16 := Nat.mod_eq_of_lt (by have := st.w56; omega)
  have e3 : X86Int.step im s2 = .ok { s2.set 10 (some wd.locals) with rip := s2.rip + 4 } :=
    step_ld (by rw [hrip2]; exact cfg.locget.d3) hs2r7 (by rw [hea16]; exact hfloc2)
  obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2.set 10 (some wd.locals) with rip := s2.rip + 4 } := ⟨_, rfl⟩
  have hs3rip : s3.rip = aLocget + 16 := by rw [hs3]; show s2.rip + 4 = aLocget + 16; rw [hrip2]
  have hs3r10 : s3.reg 10 = some wd.locals := by
    rw [hs3]; show (s2.set 10 _).reg 10 = some wd.locals; rw [reg_set_self (by omega)]
  have hs3r0 : s3.reg 0 = some n := by
    rw [hs3]; show (s2.set 10 _).reg 0 = some n; rw [reg_set_other (by decide)]; exact hs2r0
  have e4 : X86Int.step im s3 = .ok { s3 with fl := (aluF .sub n wd.locals).2, rip := s3.rip + 3 } :=
    step_cmp (by rw [hs3rip]; exact cfg.locget.d4) hs3r0 hs3r10
  obtain ⟨s4, hs4⟩ : ∃ u : X86Int.St, u = { s3 with fl := (aluF .sub n wd.locals).2, rip := s3.rip + 3 } := ⟨_, rfl⟩
  have hs4rip : s4.rip = aLocget + 19 := by rw [hs4]; show s3.rip + 3 = aLocget + 19; rw [hs3rip]
  have hs4fl : X86Int.cond s4.fl 2 = some false := by
    have hcf : s4.fl.cf = some false := by rw [hs4]; exact aluF_sub_cf (by omega)
    exact cond_cc2 hcf
  have e5 : X86Int.step im s4 = .ok { s4 with rip := aLocget + 21 } := by
    have hh := step_jcc (im := im) (s := s4) (by rw [hs4rip]; exact cfg.locget.d5) hs4fl
    simpa [hs4rip] using hh
  obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4 with rip := aLocget + 21 } := ⟨_, rfl⟩
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
  have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
  have hrun5 : run im (k1 + k2 + 3) (Out.ok s) = Out.ok s5 := by
    rw [run_add im (k1 + k2) 3, run_add im k1 k2, hr1, hr2, run_add im 1 2, r3, run_add im 1 1, r4, r5]
  have hs5rip : s5.rip = aLocget + 21 := by rw [hs5]
  have hs5r4 : s5.reg 4 = some sp := by rw [hs5, hs4, hs3]; show (s2.set 10 _).reg 4 = some sp; rw [reg_set_other (by decide)]; exact hs2r4
  have hws5 : wshape s5.mem = wshape s.mem := by rw [hs5, hs4, hs3]; show wshape s2.mem = _; rw [hw2, hm1]
  obtain ⟨k6, hk6, hr6⟩ := jmp_fail1_exit1B rc cr.locgetOob st.sp32 st.spW s5 hs5rip hs5r4
    (run_regs_ok hrun5 hregs)
    (StackRoom_mono (StackRoom_of_wshape hws5 (by rw [hs5r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + k2 + 3 + k6, (by omega), by rw [run_add im (k1 + k2 + 3) k6, hrun5]; exact hr6⟩

/-- `locset n` refuses when `hm.d = 0` or `wd.locals ≤ n`. -/
theorem hand_locset_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.locset n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.locset n) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 65 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hfail : hm.d = 0 ∨ wd.locals ≤ n := by
    by_cases h1 : 1 ≤ hm.d ∧ n < wd.locals
    · simp only [hstep] at hs; rw [if_pos h1] at hs; cases hs
    · omega
  have himmok : n < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x87 :: leBytes n 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 135 := (rdN_cons h5).1
  have hc5W : c + 5 < W := c_add_len_lt inv rfl
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkLocset_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, hfr2, hw2, hlen2⟩ :=
    run_load_imm32 cfg.l32 cfg.locset.dret cfg.locset.d0 cfg.locset.d1 cfg.locset.d2
      (by omega) (by have := st.sp32; omega) st.spW (by decide) (disjC5_of_inv inv rfl).2.symm
      s1 hrip1 hr61 hr41 hlen1 himm
      (StackRoom_mono (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room) (by decide))
  have hs2r8 : s2.reg 8 = some hm.d := by rw [gs2 8 (by decide) (by decide) (by decide)]; exact hr81
  have hs2r7 : s2.reg 7 = some w := by rw [gs2 7 (by decide) (by decide) (by decide)]; exact hr71
  have hs2r9 : s2.reg 9 = some hm.a := by rw [gs2 9 (by decide) (by decide) (by decide)]; exact hr91
  have e3 : X86Int.step im s2 =
      .ok { s2 with fl := (aluF .sub hm.d (1 % W)).2, rip := s2.rip + 4 } :=
    step_cmpI (by rw [hrip2]; exact cfg.locset.d3) hs2r8
  obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2 with fl := (aluF .sub hm.d (1 % W)).2, rip := s2.rip + 4 } := ⟨_, rfl⟩
  have hs3rip : s3.rip = aLocset + 16 := by rw [hs3]; show s2.rip + 4 = aLocset + 16; rw [hrip2]
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  by_cases hd0 : hm.d = 0
  · have hs3fl : X86Int.cond s3.fl 3 = some false := by
      have hcf : s3.fl.cf = some true := by
        rw [hs3, hd0]; exact aluF_sub_cf_true (by unfold W; decide)
      rw [cond_cc3 hcf]; rfl
    have e4 : X86Int.step im s3 = .ok { s3 with rip := aLocset + 18 } := by
      have hh := step_jcc (im := im) (s := s3) (by rw [hs3rip]; exact cfg.locset.d4) hs3fl
      simpa [hs3rip] using hh
    obtain ⟨s4, hs4⟩ : ∃ u : X86Int.St, u = { s3 with rip := aLocset + 18 } := ⟨_, rfl⟩
    have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
    have hrun4 : run im (k1 + k2 + 2) (Out.ok s) = Out.ok s4 := by
      rw [run_add im (k1 + k2) 2, run_add im k1 k2, hr1, hr2, run_add im 1 1, r3, r4]
    have hs4rip : s4.rip = aLocset + 18 := by rw [hs4]
    have hs4r4 : s4.reg 4 = some sp := by rw [hs4, hs3]; exact hs2r4
    have hws4 : wshape s4.mem = wshape s.mem := by rw [hs4, hs3, hw2, hm1]
    obtain ⟨k5, hk5, hr5⟩ := jmp_fail1_exit1B rc cr.locsetUnd st.sp32 st.spW s4 hs4rip hs4r4
      (run_regs_ok hrun4 hregs)
      (StackRoom_mono (StackRoom_of_wshape hws4 (by rw [hs4r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + k2 + 2 + k5, (by omega), by rw [run_add im (k1 + k2 + 2) k5, hrun4]; exact hr5⟩
  · have hnle : wd.locals ≤ n := by omega
    have hs3fl : X86Int.cond s3.fl 3 = some true := by
      have hcf : s3.fl.cf = some false := by
        rw [hs3]; exact aluF_sub_cf (by rw [show (1 : Nat) % W = 1 from Nat.mod_eq_of_lt (by decide)]; omega)
      exact cond_cc3 hcf
    have e4 : X86Int.step im s3 = .ok { s3 with rip := aLocset + 23 } := by
      have hh := step_jcc (im := im) (s := s3) (by rw [hs3rip]; exact cfg.locset.d4) hs3fl
      simpa using hh
    obtain ⟨s4, hs4⟩ : ∃ u : X86Int.St, u = { s3 with rip := aLocset + 23 } := ⟨_, rfl⟩
    have hs4rip : s4.rip = aLocset + 23 := by rw [hs4]
    have hs4r7 : s4.reg 7 = some w := by rw [hs4, hs3]; exact hs2r7
    have hs4r0 : s4.reg 0 = some n := by rw [hs4, hs3]; exact hs2r0
    have hs4r9 : s4.reg 9 = some hm.a := by rw [hs4, hs3]; exact hs2r9
    have hs4mem : s4.mem = s2.mem := by rw [hs4, hs3]
    have hs4len : 16 ≤ s4.regs.length := by rw [hs4, hs3]; show 16 ≤ s2.regs.length; omega
    have hdisjw8 : sp ≤ w + 16 ∨ w + 24 ≤ sp - 8 := by
      have h32 := st.sp32; cases st.disjw with | inl h => exact Or.inl (by omega) | inr h => exact Or.inr (by omega)
    have hfloc4 : rd64 s4.mem (w + 16) = some wd.locals := by
      rw [hs4mem]; unfold rd64
      rw [hfr2 (w + 16) 8 (by cases hdisjw8 with | inl h => exact Or.inr h | inr h => exact Or.inl h), hm1]
      exact inv.floc
    have hea16 : ea w 16 = w + 16 := Nat.mod_eq_of_lt (by have := st.w56; omega)
    have e5 : X86Int.step im s4 = .ok { s4.set 10 (some wd.locals) with rip := s4.rip + 4 } :=
      step_ld (by rw [hs4rip]; exact cfg.locset.d5) hs4r7 (by rw [hea16]; exact hfloc4)
    obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4.set 10 (some wd.locals) with rip := s4.rip + 4 } := ⟨_, rfl⟩
    have hs5rip : s5.rip = aLocset + 27 := by rw [hs5]; show s4.rip + 4 = aLocset + 27; rw [hs4rip]
    have hs5r10 : s5.reg 10 = some wd.locals := by
      rw [hs5]; show (s4.set 10 _).reg 10 = some wd.locals
      rw [reg_set_self (by omega)]
    have hs5r0 : s5.reg 0 = some n := by
      rw [hs5]; show (s4.set 10 _).reg 0 = some n; rw [reg_set_other (by decide)]; exact hs4r0
    have e6 : X86Int.step im s5 = .ok { s5 with fl := (aluF .sub n wd.locals).2, rip := s5.rip + 3 } :=
      step_cmp (by rw [hs5rip]; exact cfg.locset.d6) hs5r0 hs5r10
    obtain ⟨s6, hs6⟩ : ∃ u : X86Int.St, u = { s5 with fl := (aluF .sub n wd.locals).2, rip := s5.rip + 3 } := ⟨_, rfl⟩
    have hs6rip : s6.rip = aLocset + 30 := by rw [hs6]; show s5.rip + 3 = aLocset + 30; rw [hs5rip]
    have hs6fl : X86Int.cond s6.fl 2 = some false := by
      have hcf : s6.fl.cf = some false := by rw [hs6]; exact aluF_sub_cf (by omega)
      exact cond_cc2 hcf
    have e7 : X86Int.step im s6 = .ok { s6 with rip := aLocset + 32 } := by
      have hh := step_jcc (im := im) (s := s6) (by rw [hs6rip]; exact cfg.locset.d7) hs6fl
      simpa [hs6rip] using hh
    obtain ⟨s7, hs7⟩ : ∃ u : X86Int.St, u = { s6 with rip := aLocset + 32 } := ⟨_, rfl⟩
    have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
    have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
    have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
    have r7 : run im 1 (Out.ok s6) = Out.ok s7 := by rw [run_succ_ok im 0 s6, e7, ← hs7, run_zero]
    have hrun7 : run im (k1 + k2 + 5) (Out.ok s) = Out.ok s7 := by
      rw [run_add im (k1 + k2) 5, run_add im k1 k2, hr1, hr2,
        run_add im 1 4, r3, run_add im 1 3, r4, run_add im 1 2, r5, run_add im 1 1, r6, r7]
    have hs7rip : s7.rip = aLocset + 32 := by rw [hs7]
    have hs7r4 : s7.reg 4 = some sp := by
      rw [hs7, hs6, hs5]; show (s4.set 10 _).reg 4 = some sp
      rw [reg_set_other (by decide), hs4, hs3]; exact hs2r4
    have hws7 : wshape s7.mem = wshape s.mem := by rw [hs7, hs6, hs5]; show wshape s4.mem = _; rw [hs4mem, hw2, hm1]
    obtain ⟨k8, hk8, hr8⟩ := jmp_fail1_exit1B rc cr.locsetOob st.sp32 st.spW s7 hs7rip hs7r4
      (run_regs_ok hrun7 hregs)
      (StackRoom_mono (StackRoom_of_wshape hws7 (by rw [hs7r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + k2 + 5 + k8, (by omega), by rw [run_add im (k1 + k2 + 5) k8, hrun7]; exact hr8⟩

/-- `prim q` refuses when `hm.d < q.eff.1`. -/
theorem hand_prim_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {q : Prim} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.prim q :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.prim q) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 55 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hdlt : hm.d < q.eff.1 := by
    by_cases hg : q.eff.1 ≤ hm.d
    · by_cases he : q = .exit
      · subst he; simp [hstep] at hs; exact hs
      · simp [hstep, hg, he] at hs
    · omega
  have h1b : s.mem.rdN c 1 = some [q.code] := inv.csq.1
  have hbyte : s.mem.rd c = some q.code := (rdN_cons h1b).1
  obtain ⟨hrd1, -⟩ := rdN_two (inv.eff q)
  have hc57 := Prim_code_lt q
  have heffW : aEff + 2 * q.code + 2 ≤ W := by
    have h1 : aEff + 114 ≤ W := by decide
    omega
  obtain ⟨k1, s1, hk1, hr1, hrip1, hs1r0, -, -, -, hs1r8, -, hs1r4, -, hs1mem⟩ :=
    WalkPrim_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) q.code hc57 s inv.rip inv.r8 inv.r9
      inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  have heff : aEff % W = aEff := Nat.mod_eq_of_lt (by omega)
  have hadd1 : (aluF .add aEff q.code).1 = aEff + q.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have hadd2 : (aluF .add (aEff + q.code) q.code).1 = aEff + 2 * q.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have e1 : X86Int.step im s1 = .ok { s1.set 2 (some (aEff % W)) with rip := s1.rip + 10 } :=
    step_movI (by rw [hrip1]; exact cfg.prim.d0)
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s1.set 2 (some (aEff % W)) with rip := s1.rip + 10 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aPrim + 10 := by rw [ha1]; show s1.rip + 10 = aPrim + 10; rw [hrip1]
  have ha1r2 : a1.reg 2 = some aEff := by
    rw [ha1]; show (s1.set 2 _).reg 2 = some aEff; rw [reg_set_self (by omega), heff]
  have ga1 : ∀ qp, qp ≠ 2 → a1.reg qp = s1.reg qp := by
    intro qp hq; rw [ha1]; show (s1.set 2 _).reg qp = s1.reg qp; rw [reg_set_other hq]
  have ha1r0 : a1.reg 0 = some q.code := by rw [ga1 0 (by decide)]; exact hs1r0
  have ha1len : a1.regs.length = s1.regs.length := by
    rw [ha1]; show (s1.set 2 _).regs.length = s1.regs.length; exact List.length_set
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add aEff q.code).1) with fl := (aluF .add aEff q.code).2, rip := a1.rip + 3 } :=
    step_alu_add (by rw [ha1rip]; exact cfg.prim.d1) ha1r2 ha1r0
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add aEff q.code).1) with fl := (aluF .add aEff q.code).2, rip := a1.rip + 3 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = aPrim + 13 := by rw [ha2]; show a1.rip + 3 = aPrim + 13; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (aEff + q.code) := by
    rw [ha2]; show (a1.set 2 _).reg 2 = _; rw [reg_set_self (by omega), hadd1]
  have ga2 : ∀ qp, qp ≠ 2 → a2.reg qp = s1.reg qp := by
    intro qp hq; rw [ha2]; show (a1.set 2 _).reg qp = s1.reg qp; rw [reg_set_other hq]; exact ga1 qp hq
  have ha2r0 : a2.reg 0 = some q.code := by rw [ga2 0 (by decide)]; exact hs1r0
  have ha2len : a2.regs.length = s1.regs.length := by
    rw [ha2]; show (a1.set 2 _).regs.length = s1.regs.length; exact List.length_set.trans ha1len
  have e3 : X86Int.step im a2 = .ok { a2.set 2 (some (aluF .add (aEff + q.code) q.code).1) with fl := (aluF .add (aEff + q.code) q.code).2, rip := a2.rip + 3 } :=
    step_alu_add (by rw [ha2rip]; exact cfg.prim.d2) ha2r2 ha2r0
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 2 (some (aluF .add (aEff + q.code) q.code).1) with fl := (aluF .add (aEff + q.code) q.code).2, rip := a2.rip + 3 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = aPrim + 16 := by rw [ha3]; show a2.rip + 3 = aPrim + 16; rw [ha2rip]
  have ha3r2 : a3.reg 2 = some (aEff + 2 * q.code) := by
    rw [ha3]; show (a2.set 2 _).reg 2 = _; rw [reg_set_self (by omega), hadd2]
  have ga3 : ∀ qp, qp ≠ 2 → a3.reg qp = s1.reg qp := by
    intro qp hq; rw [ha3]; show (a2.set 2 _).reg qp = s1.reg qp; rw [reg_set_other hq]; exact ga2 qp hq
  have ha3mem : a3.mem = s.mem := by rw [ha3, ha2, ha1]; exact hs1mem
  have ha3len : a3.regs.length = s1.regs.length := by
    rw [ha3]; show (a2.set 2 _).regs.length = s1.regs.length; exact List.length_set.trans ha2len
  have hea0 : ea (aEff + 2 * q.code) 0 = aEff + 2 * q.code := Nat.mod_eq_of_lt (by omega)
  have e4 : X86Int.step im a3 = .ok { a3.set 10 (some q.eff.1) with rip := a3.rip + 4 } :=
    step_ld8 (by rw [ha3rip]; exact cfg.prim.d3) ha3r2 (by rw [ha3mem, hea0]; exact hrd1)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 10 (some q.eff.1) with rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = aPrim + 20 := by rw [ha4]; show a3.rip + 4 = aPrim + 20; rw [ha3rip]
  have ha4r10 : a4.reg 10 = some q.eff.1 := by
    rw [ha4]; show (a3.set 10 _).reg 10 = some q.eff.1; rw [reg_set_self (by omega)]
  have ha4r8 : a4.reg 8 = some hm.d := by
    rw [ha4]; show (a3.set 10 _).reg 8 = some hm.d; rw [reg_set_other (by decide), ga3 8 (by decide)]; exact hs1r8
  have ha4r4 : a4.reg 4 = some sp := by
    rw [ha4]; show (a3.set 10 _).reg 4 = some sp; rw [reg_set_other (by decide), ga3 4 (by decide)]; exact hs1r4
  have e5 : X86Int.step im a4 = .ok { a4 with fl := (aluF .sub hm.d q.eff.1).2, rip := a4.rip + 3 } :=
    step_cmp (by rw [ha4rip]; exact cfg.prim.d4) ha4r8 ha4r10
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with fl := (aluF .sub hm.d q.eff.1).2, rip := a4.rip + 3 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = aPrim + 23 := by rw [ha5]; show a4.rip + 3 = aPrim + 23; rw [ha4rip]
  have ha5fl : X86Int.cond a5.fl 3 = some false := by
    have hcf : a5.fl.cf = some true := by rw [ha5]; exact aluF_sub_cf_true hdlt
    rw [cond_cc3 hcf]; rfl
  have e6 : X86Int.step im a5 = .ok { a5 with rip := aPrim + 25 } := by
    have hh := step_jcc (im := im) (s := a5) (by rw [ha5rip]; exact cfg.prim.d5) ha5fl
    simpa [ha5rip] using hh
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := aPrim + 25 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s1) = Out.ok a1 := by rw [run_succ_ok im 0 s1, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have hrun6 : run im (k1 + 6) (Out.ok s) = Out.ok a6 := by
    rw [run_add im k1 6, hr1, run_add im 1 5, r1, run_add im 1 4, r2,
      run_add im 1 3, r3, run_add im 1 2, r4, run_add im 1 1, r5, r6]
  have ha6rip : a6.rip = aPrim + 25 := by rw [ha6]
  have ha6r4 : a6.reg 4 = some sp := by rw [ha6, ha5]; exact ha4r4
  have ha6mem : a6.mem = s.mem := by rw [ha6, ha5, ha4]; exact ha3mem
  obtain ⟨k7, hk7, hr7⟩ := jmp_fail1_exit1B rc cr.primUnd st.sp32 st.spW a6 ha6rip ha6r4
    (run_regs_ok hrun6 hregs)
    (StackRoom_mono (StackRoom_of_mem ha6mem (by rw [ha6r4, inv.r4]) inv.room) (by decide))
  exact ⟨k1 + 6 + k7, (by omega), by rw [run_add im (k1 + 6) k7, hrun6]; exact hr7⟩

/-- Helper for `hand_call_refuseB`: computes `tbl + n * 64`, loads `cw.ins`,
and refuses when `hm.d < cw.ins` via `cr.callUnd`. -/
private theorem call_und_refuse {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl n : Nat} {hm : Height}
    (st : WalkStatic pr wd w p sp tbl)
    (hnlt : n < pr.words.length) (hdund : hm.d < (pr.words[n]'hnlt).ins) :
    ∀ s2 : X86Int.St, s2.rip = aCall + 15 → s2.fl = (aluF .sub n pr.words.length).2 →
      s2.reg 0 = some n → s2.reg 8 = some hm.d → s2.reg 4 = some sp →
      s2.reg 14 = some tbl → 16 ≤ s2.regs.length → RegsOk s2 →
      rd64 s2.mem (tbl + n * 64) = some (pr.words[n]'hnlt).ins →
      StackRoom s2 1 →
      ∃ m, m ≤ 20 ∧ run im m (Out.ok s2) = Out.exit 1 := by
  intro s2 hrip2 hfl2 hr02 hr82 hr42 hr142 hlen2 hregs2 hins2 hroom2
  have htblB := (st.wordsB n hnlt).1
  have h14len : 14 < s2.regs.length := by omega
  have hs2cond : X86Int.cond s2.fl 2 = some true := by
    have hcf : s2.fl.cf = some true := by rw [hfl2]; exact aluF_sub_cf_true hnlt
    exact cond_cc2 hcf
  have e7 : X86Int.step im s2 = .ok { s2 with rip := aCall + 22 } := by
    have hh := step_jcc (im := im) (s := s2) (by rw [hrip2]; exact cfg.call.d4) hs2cond
    simpa using hh
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { s2 with rip := aCall + 22 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = aCall + 22 := by rw [ha7]
  have ha7r0 : a7.reg 0 = some n := by rw [ha7]; exact hr02
  have ha7r8 : a7.reg 8 = some hm.d := by rw [ha7]; exact hr82
  have ha7r4 : a7.reg 4 = some sp := by rw [ha7]; exact hr42
  have ha7r14 : a7.reg 14 = some tbl := by rw [ha7]; exact hr142
  have ha7mem : a7.mem = s2.mem := by rw [ha7]
  have e8 : X86Int.step im a7 = .ok { a7.set 2 (some n) with rip := a7.rip + 3 } :=
    step_mov (by rw [ha7rip]; exact cfg.call.d5) ha7r0
  obtain ⟨b0, hb0⟩ : ∃ u : X86Int.St, u = { a7.set 2 (some n) with rip := a7.rip + 3 } := ⟨_, rfl⟩
  have hb0rip : b0.rip = aCall + 25 := by rw [hb0]; show a7.rip + 3 = aCall + 25; rw [ha7rip]
  have hb0r2 : b0.reg 2 = some n := by
    rw [hb0]; show (a7.set 2 _).reg 2 = some n
    rw [reg_set_self (by show 2 < a7.regs.length; rw [ha7]; exact Nat.lt_of_lt_of_le (by decide) hlen2)]
  have hb0len : 16 ≤ b0.regs.length := by
    rw [hb0]; show 16 ≤ (a7.set 2 _).regs.length; rw [len_set, ha7]; exact hlen2
  have gb0 : ∀ q, q ≠ 2 → b0.reg q = a7.reg q := by
    intro q hq; rw [hb0]; show (a7.set 2 _).reg q = a7.reg q; rw [reg_set_other hq]
  have hb0mem : b0.mem = s2.mem := by rw [hb0]; exact ha7mem
  obtain ⟨b1, rb1, hb1rip, hb1r2, gb1, hb1mem, hb1len⟩ :=
    run_double_r2 (v := n) cfg.call.d6 b0 hb0rip hb0r2 (by omega) hb0len
  have hb1r2' : b1.reg 2 = some (2 * n) := hb1r2
  obtain ⟨b2, rb2, hb2rip, hb2r2, gb2, hb2mem, hb2len⟩ :=
    run_double_r2 (v := 2 * n) cfg.call.d7 b1 hb1rip hb1r2' (by omega) (by omega)
  have hb2r2' : b2.reg 2 = some (4 * n) := by rw [hb2r2]; congr 1; omega
  obtain ⟨b3, rb3, hb3rip, hb3r2, gb3, hb3mem, hb3len⟩ :=
    run_double_r2 (v := 4 * n) cfg.call.d8 b2 hb2rip hb2r2' (by omega) (by omega)
  have hb3r2' : b3.reg 2 = some (8 * n) := by rw [hb3r2]; congr 1; omega
  obtain ⟨b4, rb4, hb4rip, hb4r2, gb4, hb4mem, hb4len⟩ :=
    run_double_r2 (v := 8 * n) cfg.call.d9 b3 hb3rip hb3r2' (by omega) (by omega)
  have hb4r2' : b4.reg 2 = some (16 * n) := by rw [hb4r2]; congr 1; omega
  obtain ⟨b5, rb5, hb5rip, hb5r2, gb5, hb5mem, hb5len⟩ :=
    run_double_r2 (v := 16 * n) cfg.call.d10 b4 hb4rip hb4r2' (by omega) (by omega)
  have hb5r2' : b5.reg 2 = some (32 * n) := by rw [hb5r2]; congr 1; omega
  obtain ⟨b6, rb6, hb6rip, hb6r2, gb6, hb6mem, hb6len⟩ :=
    run_double_r2 (v := 32 * n) cfg.call.d11 b5 hb5rip hb5r2' (by omega) (by omega)
  have hb6r2' : b6.reg 2 = some (n * 64) := by rw [hb6r2]; congr 1; omega
  have gb6_s : ∀ q, q ≠ 2 → b6.reg q = a7.reg q := by
    intro q hq
    rw [gb6 q hq, gb5 q hq, gb4 q hq, gb3 q hq, gb2 q hq, gb1 q hq, gb0 q hq]
  have hb6r14 : b6.reg 14 = some tbl := by rw [gb6_s 14 (by decide)]; exact ha7r14
  have hb6mem' : b6.mem = s2.mem := by rw [hb6mem, hb5mem, hb4mem, hb3mem, hb2mem, hb1mem, hb0mem]
  have haddt : (aluF .add (n * 64) tbl).1 = tbl + n * 64 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega), Nat.add_comm]
  have eb7 : X86Int.step im b6 = .ok { b6.set 2 (some (aluF .add (n * 64) tbl).1) with fl := (aluF .add (n * 64) tbl).2, rip := b6.rip + 3 } :=
    step_alu_add (by rw [hb6rip]; exact cfg.call.d12) hb6r2' hb6r14
  obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u = { b6.set 2 (some (aluF .add (n * 64) tbl).1) with fl := (aluF .add (n * 64) tbl).2, rip := b6.rip + 3 } := ⟨_, rfl⟩
  have hb7rip : b7.rip = aCall + 46 := by rw [hb7]; show b6.rip + 3 = aCall + 46; rw [hb6rip]
  have hb7r2 : b7.reg 2 = some (tbl + n * 64) := by
    rw [hb7]; show (b6.set 2 _).reg 2 = _; rw [reg_set_self (by omega), haddt]
  have gb7 : ∀ q, q ≠ 2 → b7.reg q = a7.reg q := by
    intro q hq; rw [hb7]; show (b6.set 2 _).reg q = a7.reg q; rw [reg_set_other hq]; exact gb6_s q hq
  have hb7mem : b7.mem = s2.mem := by rw [hb7]; exact hb6mem'
  have hb7len : 16 ≤ b7.regs.length := by
    rw [hb7]; show 16 ≤ (b6.set 2 _).regs.length; rw [len_set]; omega
  have hea0 : ea (tbl + n * 64) 0 = tbl + n * 64 := Nat.mod_eq_of_lt (by omega)
  have eb8 : X86Int.step im b7 = .ok { b7.set 10 (some (pr.words[n]'hnlt).ins) with rip := b7.rip + 3 } :=
    step_ld (by rw [hb7rip]; exact cfg.call.d13) hb7r2 (by rw [hb7mem, hea0]; exact hins2)
  obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7.set 10 (some (pr.words[n]'hnlt).ins) with rip := b7.rip + 3 } := ⟨_, rfl⟩
  have hb8rip : b8.rip = aCall + 49 := by rw [hb8]; show b7.rip + 3 = aCall + 49; rw [hb7rip]
  have hb8r10 : b8.reg 10 = some (pr.words[n]'hnlt).ins := by
    rw [hb8]; show (b7.set 10 _).reg 10 = _; rw [reg_set_self (by omega)]
  have hb8r8 : b8.reg 8 = some hm.d := by
    rw [hb8]; show (b7.set 10 _).reg 8 = _; rw [reg_set_other (by decide), gb7 8 (by decide)]; exact ha7r8
  have hb8r4 : b8.reg 4 = some sp := by
    rw [hb8]; show (b7.set 10 _).reg 4 = _; rw [reg_set_other (by decide), gb7 4 (by decide)]; exact ha7r4
  have eb9 : X86Int.step im b8 = .ok { b8 with fl := (aluF .sub hm.d (pr.words[n]'hnlt).ins).2, rip := b8.rip + 3 } :=
    step_cmp (by rw [hb8rip]; exact cfg.call.d14) hb8r8 hb8r10
  obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u = { b8 with fl := (aluF .sub hm.d (pr.words[n]'hnlt).ins).2, rip := b8.rip + 3 } := ⟨_, rfl⟩
  have hb9rip : b9.rip = aCall + 52 := by rw [hb9]; show b8.rip + 3 = aCall + 52; rw [hb8rip]
  have hb9fl : X86Int.cond b9.fl 3 = some false := by
    have hcf : b9.fl.cf = some true := by rw [hb9]; exact aluF_sub_cf_true hdund
    rw [cond_cc3 hcf]; rfl
  have eb10 : X86Int.step im b9 = .ok { b9 with rip := aCall + 54 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact cfg.call.d15) hb9fl
    simpa [hb9rip] using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := aCall + 54 } := ⟨_, rfl⟩
  have r7 : run im 1 (Out.ok s2) = Out.ok a7 := by rw [run_succ_ok im 0 s2, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok b0 := by rw [run_succ_ok im 0 a7, e8, ← hb0, run_zero]
  have r15 : run im 1 (Out.ok b6) = Out.ok b7 := by rw [run_succ_ok im 0 b6, eb7, ← hb7, run_zero]
  have r16 : run im 1 (Out.ok b7) = Out.ok b8 := by rw [run_succ_ok im 0 b7, eb8, ← hb8, run_zero]
  have r17 : run im 1 (Out.ok b8) = Out.ok b9 := by rw [run_succ_ok im 0 b8, eb9, ← hb9, run_zero]
  have r18 : run im 1 (Out.ok b9) = Out.ok b10 := by rw [run_succ_ok im 0 b9, eb10, ← hb10, run_zero]
  have hrun_b0 : run im 2 (Out.ok s2) = Out.ok b0 := by rw [run_add im 1 1, r7, r8]
  have hrun_b1 : run im 3 (Out.ok s2) = Out.ok b1 := by rw [run_add im 2 1, hrun_b0, rb1]
  have hrun_b2 : run im 4 (Out.ok s2) = Out.ok b2 := by rw [run_add im 3 1, hrun_b1, rb2]
  have hrun_b3 : run im 5 (Out.ok s2) = Out.ok b3 := by rw [run_add im 4 1, hrun_b2, rb3]
  have hrun_b4 : run im 6 (Out.ok s2) = Out.ok b4 := by rw [run_add im 5 1, hrun_b3, rb4]
  have hrun_b5 : run im 7 (Out.ok s2) = Out.ok b5 := by rw [run_add im 6 1, hrun_b4, rb5]
  have hrun_b6 : run im 8 (Out.ok s2) = Out.ok b6 := by rw [run_add im 7 1, hrun_b5, rb6]
  have hrun_b10 : run im 12 (Out.ok s2) = Out.ok b10 := by
    rw [show (12 : Nat) = 8 + 4 by rfl, run_add im 8 4, hrun_b6,
      run_add im 1 3, r15, run_add im 1 2, r16, run_add im 1 1, r17, r18]
  have hb10rip : b10.rip = aCall + 54 := by rw [hb10]
  have hb10r4 : b10.reg 4 = some sp := by rw [hb10, hb9]; exact hb8r4
  have hb10mem : b10.mem = s2.mem := by rw [hb10, hb9, hb8]; exact hb7mem
  obtain ⟨k11, hk11, hr11⟩ := jmp_fail1_exit1B rc cr.callUnd st.sp32 st.spW b10 hb10rip hb10r4
    (run_regs_ok hrun_b10 hregs2) (StackRoom_of_mem hb10mem (by rw [hb10r4, hr42]) hroom2)
  exact ⟨12 + k11, (by omega), by rw [run_add im 12 k11, hrun_b10]; exact hr11⟩

/-- `call n` refuses when `pr.words.length ≤ n` or `hm.d < (pr.words[n]).ins`. -/
theorem hand_call_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.call n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.call n) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 75 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have himmok : n < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x8b :: leBytes n 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 139 := (rdN_cons h5).1
  have hc5W : c + 5 < W := c_add_len_lt inv rfl
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, hr61, _, hr81, _, hr41, hg1, hm1⟩ :=
    WalkCall_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, hfr2, hw2, hlen2⟩ :=
    run_load_imm32 cfg.l32 cfg.call.dret cfg.call.d0 cfg.call.d1 cfg.call.d2
      (by omega) (by have := st.sp32; omega) st.spW (by decide) (disjC5_of_inv inv rfl).2.symm
      s1 hrip1 hr61 hr41 hlen1 himm
      (StackRoom_mono (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room) (by decide))
  have hs2r13 : s2.reg 13 = some pr.words.length := by
    rw [gs2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)]; exact inv.r13
  have hs2r14 : s2.reg 14 = some tbl := by
    rw [gs2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)]; exact inv.r14
  have hs2r8 : s2.reg 8 = some hm.d := by
    rw [gs2 8 (by decide) (by decide) (by decide)]; exact hr81
  have e3 : X86Int.step im s2 =
      .ok { s2 with fl := (aluF .sub n pr.words.length).2, rip := s2.rip + 3 } :=
    step_cmp (by rw [hrip2]; exact cfg.call.d3) hs2r0 hs2r13
  obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2 with fl := (aluF .sub n pr.words.length).2, rip := s2.rip + 3 } := ⟨_, rfl⟩
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  have hrun3 : run im (k1 + k2 + 1) (Out.ok s) = Out.ok s3 := by
    rw [run_add im (k1 + k2) 1, run_add im k1 k2, hr1, hr2, r3]
  have hs3rip : s3.rip = aCall + 15 := by rw [hs3]; show s2.rip + 3 = aCall + 15; rw [hrip2]
  by_cases hnlt : n < pr.words.length
  · have hcw : pr.words[n]? = some (pr.words[n]'hnlt) := List.getElem?_eq_getElem hnlt
    have hdund : hm.d < (pr.words[n]'hnlt).ins := by
      by_cases hg : (pr.words[n]'hnlt).ins ≤ hm.d
      · simp [hstep, hcw, hg] at hs
      · omega
    have hdisjT : sp ≤ tbl + n * 64 ∨ tbl + n * 64 + 16 ≤ sp - 8 := by
      have h32 := st.sp32
      cases (st.wordsB n hnlt).2 with | inl h => exact Or.inl h | inr h => exact Or.inr (by omega)
    have hins3 : rd64 s3.mem (tbl + n * 64) = some (pr.words[n]'hnlt).ins := by
      rw [hs3]; unfold rd64
      rw [hfr2 (tbl + n * 64) 8 (by cases hdisjT with | inl h => exact Or.inr h | inr h => exact Or.inl (by omega)), hm1]
      exact (inv.words n hnlt).1
    have hws3 : wshape s3.mem = wshape s.mem := by rw [hs3, hw2, hm1]
    have hs3r4 : s3.reg 4 = some sp := by rw [hs3]; exact hs2r4
    have hs3len : 16 ≤ s3.regs.length := by rw [hs3]; show 16 ≤ s2.regs.length; omega
    obtain ⟨k4, hk4, hr4⟩ := call_und_refuse cfg rc cr st hnlt hdund s3 hs3rip (by rw [hs3])
      (by rw [hs3]; exact hs2r0) (by rw [hs3]; exact hs2r8) hs3r4
      (by rw [hs3]; exact hs2r14) hs3len (run_regs_ok hrun3 hregs) hins3
      (StackRoom_mono (StackRoom_of_wshape hws3 (by rw [hs3r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + k2 + 1 + k4, (by omega), by rw [run_add im (k1 + k2 + 1) k4, hrun3]; exact hr4⟩
  · have hs3fl : X86Int.cond s3.fl 2 = some false := by
      have hcf : s3.fl.cf = some false := by rw [hs3]; exact aluF_sub_cf hnlt
      exact cond_cc2 hcf
    have e4 : X86Int.step im s3 = .ok { s3 with rip := aCall + 17 } := by
      have hh := step_jcc (im := im) (s := s3) (by rw [hs3rip]; exact cfg.call.d4) hs3fl
      simpa [hs3rip] using hh
    obtain ⟨s4, hs4⟩ : ∃ u : X86Int.St, u = { s3 with rip := aCall + 17 } := ⟨_, rfl⟩
    have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
    have hrun4 : run im (k1 + k2 + 2) (Out.ok s) = Out.ok s4 := by
      rw [run_add im (k1 + k2 + 1) 1, hrun3, r4]
    have hs4rip : s4.rip = aCall + 17 := by rw [hs4]
    have hs4r4 : s4.reg 4 = some sp := by rw [hs4, hs3]; exact hs2r4
    have hws4 : wshape s4.mem = wshape s.mem := by rw [hs4, hs3, hw2, hm1]
    obtain ⟨k5, hk5, hr5⟩ := jmp_fail1_exit1B rc cr.callOob st.sp32 st.spW s4 hs4rip hs4r4
      (run_regs_ok hrun4 hregs)
      (StackRoom_mono (StackRoom_of_wshape hws4 (by rw [hs4r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + k2 + 2 + k5, (by omega), by rw [run_add im (k1 + k2 + 2) k5, hrun4]; exact hr5⟩

/-- `ret` refuses when `hm ≠ ⟨wd.outs, 0⟩`. -/
theorem hand_ret_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (houtW : wd.outs < W) {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.ret :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.ret = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 55 ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hdW : hm.d < W := by have h1 := hBnd_head inv.hb; rw [inv.hlink.symm] at h1; omega
  have hne : hm ≠ ⟨wd.outs, 0⟩ := by
    intro heq; simp [hstep, heq] at hs
  have h1b : s.mem.rdN c 1 = some [140] := inv.csq.1
  have hbyte : s.mem.rd c = some 140 := (rdN_cons h1b).1
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, -, -, hs1r7, hs1r8, hs1r9, hs1r4, -, hs1mem⟩ :=
    WalkRet_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  have hea8 : ea w 8 = w + 8 := Nat.mod_eq_of_lt (by have := st.w56; omega)
  have e1 : X86Int.step im s1 = .ok { s1.set 0 (some wd.outs) with rip := s1.rip + 4 } :=
    step_ld (by rw [hrip1]; exact cfg.ret.d0) hs1r7 (by rw [hs1mem, hea8]; exact inv.fouts)
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s1.set 0 (some wd.outs) with rip := s1.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aRet + 4 := by rw [ha1]; show s1.rip + 4 = aRet + 4; rw [hrip1]
  have ha1r0 : a1.reg 0 = some wd.outs := by
    rw [ha1]; show (s1.set 0 _).reg 0 = some wd.outs; rw [reg_set_self (by omega)]
  have ga1 : ∀ q, q ≠ 0 → a1.reg q = s1.reg q := by
    intro q hq; rw [ha1]; show (s1.set 0 _).reg q = s1.reg q; rw [reg_set_other hq]
  have ha1r8 : a1.reg 8 = some hm.d := by rw [ga1 8 (by decide)]; exact hs1r8
  have ha1r9 : a1.reg 9 = some hm.a := by rw [ga1 9 (by decide)]; exact hs1r9
  have ha1r4 : a1.reg 4 = some sp := by rw [ga1 4 (by decide)]; exact hs1r4
  have ha1mem : a1.mem = s.mem := by rw [ha1]; exact hs1mem
  have e2 : X86Int.step im a1 = .ok { a1 with fl := (aluF .sub hm.d wd.outs).2, rip := a1.rip + 3 } :=
    step_cmp (by rw [ha1rip]; exact cfg.ret.d1) ha1r8 ha1r0
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with fl := (aluF .sub hm.d wd.outs).2, rip := a1.rip + 3 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = aRet + 7 := by rw [ha2]; show a1.rip + 3 = aRet + 7; rw [ha1rip]
  have r1 : run im 1 (Out.ok s1) = Out.ok a1 := by rw [run_succ_ok im 0 s1, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  by_cases hdeq : hm.d = wd.outs
  · have hane : hm.a ≠ 0 := by intro ha0; apply hne; cases hm; simp at hdeq ha0; simp [hdeq, ha0]
    have ha2fl : X86Int.cond a2.fl 4 = some true := by
      have hzf : a2.fl.zf = some true := by rw [ha2]; exact aluF_sub_zf_eq hdW houtW hdeq
      exact cond_cc4 hzf
    have e3 : X86Int.step im a2 = .ok { a2 with rip := aRet + 14 } := by
      have hh := step_jcc (im := im) (s := a2) (by rw [ha2rip]; exact cfg.ret.d2) ha2fl
      simpa using hh
    obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := aRet + 14 } := ⟨_, rfl⟩
    have ha3rip : a3.rip = aRet + 14 := by rw [ha3]
    have ha3r9 : a3.reg 9 = some hm.a := by rw [ha3, ha2]; exact ha1r9
    have e4 : X86Int.step im a3 = .ok { a3 with fl := (aluF .and hm.a hm.a).2, rip := a3.rip + 3 } :=
      step_test (by rw [ha3rip]; exact cfg.ret.d3) ha3r9 ha3r9
    obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3 with fl := (aluF .and hm.a hm.a).2, rip := a3.rip + 3 } := ⟨_, rfl⟩
    have ha4rip : a4.rip = aRet + 17 := by rw [ha4]; show a3.rip + 3 = aRet + 17; rw [ha3rip]
    have ha4fl : X86Int.cond a4.fl 4 = some false := by
      have hz : a4.fl.zf = some false := by
        rw [ha4]; show (aluF .and hm.a hm.a).2.zf = some false; rw [aluF_zf, aluF_and_fst_self]; simp [hane]
      exact cond_cc4 hz
    have e5 : X86Int.step im a4 = .ok { a4 with rip := aRet + 19 } := by
      have hh := step_jcc (im := im) (s := a4) (by rw [ha4rip]; exact cfg.ret.d4) ha4fl
      simpa [ha4rip] using hh
    obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := aRet + 19 } := ⟨_, rfl⟩
    have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
    have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
    have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
    have hrun5 : run im (k1 + 5) (Out.ok s) = Out.ok a5 := by
      rw [run_add im k1 5, hr1, run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3, run_add im 1 1, r4, r5]
    have ha5rip : a5.rip = aRet + 19 := by rw [ha5]
    have ha5r4 : a5.reg 4 = some sp := by rw [ha5, ha4, ha3, ha2]; exact ha1r4
    have ha5mem : a5.mem = s.mem := by rw [ha5, ha4, ha3, ha2]; exact ha1mem
    obtain ⟨k6, hk6, hr6⟩ := jmp_fail1_exit1B rc cr.retA st.sp32 st.spW a5 ha5rip ha5r4
      (run_regs_ok hrun5 hregs) (StackRoom_mono (StackRoom_of_mem ha5mem (by rw [ha5r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + 5 + k6, (by omega), by rw [run_add im (k1 + 5) k6, hrun5]; exact hr6⟩
  · have ha2fl : X86Int.cond a2.fl 4 = some false := by
      have hzf : a2.fl.zf = some false := by rw [ha2]; exact aluF_sub_zf_ne hdW houtW hdeq
      exact cond_cc4 hzf
    have e3 : X86Int.step im a2 = .ok { a2 with rip := aRet + 9 } := by
      have hh := step_jcc (im := im) (s := a2) (by rw [ha2rip]; exact cfg.ret.d2) ha2fl
      simpa [ha2rip] using hh
    obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := aRet + 9 } := ⟨_, rfl⟩
    have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
    have hrun3 : run im (k1 + 3) (Out.ok s) = Out.ok a3 := by
      rw [run_add im k1 3, hr1, run_add im 1 2, r1, run_add im 1 1, r2, r3]
    have ha3rip : a3.rip = aRet + 9 := by rw [ha3]
    have ha3r4 : a3.reg 4 = some sp := by rw [ha3, ha2]; exact ha1r4
    have ha3mem : a3.mem = s.mem := by rw [ha3, ha2]; exact ha1mem
    obtain ⟨k4, hk4, hr4⟩ := jmp_fail1_exit1B rc cr.retD st.sp32 st.spW a3 ha3rip ha3r4
      (run_regs_ok hrun3 hregs) (StackRoom_mono (StackRoom_of_mem ha3mem (by rw [ha3r4, inv.r4]) inv.room) (by decide))
    exact ⟨k1 + 3 + k4, (by omega), by rw [run_add im (k1 + 3) k4, hrun3]; exact hr4⟩

set_option maxHeartbeats 800000 in
/-- `jmp t` refuses when `wd.ann.lookup t ≠ some hm`. -/
theorem hand_jmp_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jmp t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jmp t) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 100 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hne : wd.ann.lookup t ≠ some hm := by
    intro hlk; simp [hstep, hlk] at hs
  have hbhead := hBnd_head inv.hb; rw [inv.hlink.symm] at hbhead
  have hdW : hm.d < W := by omega
  have haW : hm.a < W := by omega
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have htW : t < W := Nat.lt_trans himmok (by decide)
  have h5 : s.mem.rdN c 5 = some (0x88 :: leBytes t 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 136 := (rdN_cons h5).1
  have hc5W : c + 5 < W := c_add_len_lt inv rfl
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkJmp_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, hfr2, hw2, hlen2⟩ :=
    run_load_imm32 cfg.l32 cfg.jmp.dret cfg.jmp.d0 cfg.jmp.d1 cfg.jmp.d2
      (by omega) (by omega) st.spW (by decide) (disjC5_of_inv inv rfl).2.symm
      s1 hrip1 hr61 hr41 hlen1 himm
      (StackRoom_mono (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room) (by decide))
  have hs2r1 : s2.reg 1 = some i := by rw [gs2 1 (by decide) (by decide) (by decide)]; exact hr11
  have hs2r7 : s2.reg 7 = some w := by rw [gs2 7 (by decide) (by decide) (by decide)]; exact hr71
  have hs2r8 : s2.reg 8 = some hm.d := by rw [gs2 8 (by decide) (by decide) (by decide)]; exact hr81
  have hs2r9 : s2.reg 9 = some hm.a := by rw [gs2 9 (by decide) (by decide) (by decide)]; exact hr91
  have hlen2' : 16 ≤ s2.regs.length := by omega
  have hroom2 : StackRoom s2 4 := StackRoom_of_wshape (hw2.trans (congrArg wshape hm1)) (by rw [hs2r4, inv.r4]) inv.room
  have hlay2 : AnnLayout s2 wd.ann p := AnnLayout_of_frame inv.lay (by
    intro x k hx1 hx2; rw [hfr2 x k (by cases st.disjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega)), hm1])
  have hfp2 : rd64 s2.mem (w + 48) = some p := by
    unfold rd64; rw [hfr2 (w + 48) 8 (by cases st.disjw with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega)), hm1]
    exact inv.fannp
  have hfn2 : rd64 s2.mem (w + 32) = some wd.ann.length := by
    unfold rd64; rw [hfr2 (w + 32) 8 (by cases st.disjw with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega)), hm1]
    exact inv.fnann
  -- mov r12, rax
  have e3 : X86Int.step im s2 = .ok { s2.set 12 (some t) with rip := s2.rip + 3 } :=
    step_mov (by rw [hrip2]; exact cfg.jmp.d3) hs2r0
  obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2.set 12 (some t) with rip := s2.rip + 3 } := ⟨_, rfl⟩
  have hs3rip : s3.rip = aJmp + 15 := by rw [hs3]; show s2.rip + 3 = aJmp + 15; rw [hrip2]
  have hs3r12 : s3.reg 12 = some t := by rw [hs3]; show (s2.set 12 _).reg 12 = _; rw [reg_set_self (by omega)]
  have gs3 : ∀ q, q ≠ 12 → s3.reg q = s2.reg q := by
    intro q hq; rw [hs3]; show (s2.set 12 _).reg q = s2.reg q; rw [reg_set_other hq]
  have hs3mem : s3.mem = s2.mem := by rw [hs3]; exact rfl
  have hs3r1 : s3.reg 1 = some i := by rw [gs3 1 (by decide)]; exact hs2r1
  have hs3r7 : s3.reg 7 = some w := by rw [gs3 7 (by decide)]; exact hs2r7
  have hs3r4 : s3.reg 4 = some sp := by rw [gs3 4 (by decide)]; exact hs2r4
  have hs3r8 : s3.reg 8 = some hm.d := by rw [gs3 8 (by decide)]; exact hs2r8
  have hs3r9 : s3.reg 9 = some hm.a := by rw [gs3 9 (by decide)]; exact hs2r9
  have hs3len : 16 ≤ s3.regs.length := by rw [hs3]; show 16 ≤ (s2.set 12 _).regs.length; rw [len_set]; exact hlen2'
  -- call ANNL
  have annls := ANNL_tripleB cfg.annl cfg.l32 (by decide)
  obtain ⟨k4, s4, hk4, hr4, hs4rip, hpost, _, hs4r4, _, _, _, gs4, hfr4seg, hex0, hloc4, hw4seg, hlen4⟩ :=
    CallANNLB cfg.annl cfg.l32 (by decide) cfg.jmp.d4 (by decide) htW st.p0 st.w56
      st.pann st.sp32 st.spW s3 hs3rip hs3r12 hs3r7 hs3r4 hs3r1 (by rw [hs3mem]; exact hfp2)
      (by rw [hs3mem]; exact hfn2) (AnnLayout_of_mem hs3mem hlay2)
      (StackRoom_of_mem hs3mem (by rw [hs3r4, hs2r4]) hroom2) st.disjw st.disjp
  obtain ⟨e, hs4r0⟩ := hex0
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  have hrun4 : run im (k1 + k2 + 1 + k4) (Out.ok s) = Out.ok s4 := by
    rw [run_add im (k1 + k2 + 1) k4, run_add im (k1 + k2) 1, run_add im k1 k2, hr1, hr2, r3, hr4]
  have hs4r8 : s4.reg 8 = some hm.d := by rw [gs4 8 (by decide) (by decide) (by decide) (by decide)]; exact hs3r8
  have hs4r9 : s4.reg 9 = some hm.a := by rw [gs4 9 (by decide) (by decide) (by decide) (by decide)]; exact hs3r9
  have hws4 : wshape s4.mem = wshape s.mem := by rw [hw4seg, hs3mem, hw2, hm1]
  have hroom4 : StackRoom s4 4 := StackRoom_of_wshape hws4 (by rw [hs4r4, inv.r4]) inv.room
  have hlen4s : 16 ≤ s4.regs.length := by rw [hlen4]; exact hs3len
  have h4W : 4 < W := by decide
  have h8W : 8 < W := by decide
  have hs4rip' : s4.rip = aJmp + 20 := by rw [hs4rip]
  cases hlk : wd.ann.lookup t with
  | none =>
    have he0 : e = 0 := by
      cases hpost e hs4r0 with
      | inl hb => exact hb.1
      | inr hb => obtain ⟨d1, a1, -, -, hlk1⟩ := hb; rw [hlk] at hlk1; simp at hlk1
    have hs4r0' : s4.reg 0 = some 0 := by rw [hs4r0, he0]
    have e5 : X86Int.step im s4 = .ok { s4 with fl := (aluF .and 0 0).2, rip := s4.rip + 3 } :=
      step_test (by rw [hs4rip']; exact cfg.jmp.d5) hs4r0' hs4r0'
    obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4 with fl := (aluF .and 0 0).2, rip := s4.rip + 3 } := ⟨_, rfl⟩
    have hs5rip : s5.rip = aJmp + 23 := by rw [hs5]; show s4.rip + 3 = aJmp + 23; rw [hs4rip']
    have hs5fl : X86Int.cond s5.fl 5 = some false := by
      have hz : s5.fl.zf = some true := by
        rw [hs5]; show (aluF .and 0 0).2.zf = some true; rw [aluF_zf, aluF_and_fst_self]; exact rfl
      rw [cond_cc5 hz]; rfl
    have e6 : X86Int.step im s5 = .ok { s5 with rip := aJmp + 25 } := by
      have hh := step_jcc (im := im) (s := s5) (by rw [hs5rip]; exact cfg.jmp.d6) hs5fl
      simpa [hs5rip] using hh
    obtain ⟨s6, hs6⟩ : ∃ u : X86Int.St, u = { s5 with rip := aJmp + 25 } := ⟨_, rfl⟩
    have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
    have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
    have hrun6 : run im ((k1 + k2 + 1 + k4) + 2) (Out.ok s) = Out.ok s6 := by
      rw [run_add im _ 2, hrun4, run_add im 1 1, r5, r6]
    have hs6rip : s6.rip = aJmp + 25 := by rw [hs6]
    have hs6r4 : s6.reg 4 = some sp := by rw [hs6, hs5]; exact hs4r4
    have hws6 : wshape s6.mem = wshape s.mem := by rw [hs6, hs5]; exact hws4
    obtain ⟨k7, hk7, hr7⟩ := jmp_fail1_exit1B rc cr.jmpUnann st.sp32 st.spW s6 hs6rip hs6r4
      (run_regs_ok hrun6 hregs) (StackRoom_mono (StackRoom_of_wshape hws6 (by rw [hs6r4, inv.r4]) inv.room) (by decide))
    exact ⟨(k1 + k2 + 1 + k4) + 2 + k7, (by omega), by rw [run_add im _ k7, hrun6]; exact hr7⟩
  | some a =>
    have hnea : a ≠ hm := fun heq => hne (by rw [hlk, heq])
    obtain ⟨adW, aaW⟩ := hhW (t, a) (lookup_mem hlk)
    obtain ⟨he0, hat4⟩ : e ≠ 0 ∧ AnnAt s4 e t a.d a.a := by
      cases hpost e hs4r0 with
      | inl hb => rw [hb.2] at hlk; simp at hlk
      | inr hb =>
        obtain ⟨d1, a1, hne0, hat, hlk1⟩ := hb
        have hhh : (⟨d1, a1⟩ : Height) = a := Option.some.inj (hlk1.symm.trans hlk)
        exact ⟨hne0, by rwa [show d1 = a.d from congrArg Height.d hhh, show a1 = a.a from congrArg Height.a hhh] at hat⟩
    obtain ⟨helo, hehi⟩ := hloc4 e hs4r0 he0
    have hpann := st.pann
    have hdisjp := st.disjp
    have hspW := st.spW
    have hsp32 := st.sp32
    -- test rax, rax ; jnz taken -> push rax -> CallL32B a.d
    have e5 : X86Int.step im s4 = .ok { s4 with fl := (aluF .and e e).2, rip := s4.rip + 3 } :=
      step_test (by rw [hs4rip']; exact cfg.jmp.d5) hs4r0 hs4r0
    obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4 with fl := (aluF .and e e).2, rip := s4.rip + 3 } := ⟨_, rfl⟩
    have hs5rip : s5.rip = aJmp + 23 := by rw [hs5]; show s4.rip + 3 = aJmp + 23; rw [hs4rip']
    have hs5fl : X86Int.cond s5.fl 5 = some true := by
      have hz : s5.fl.zf = some false := by
        rw [hs5]; show (aluF .and e e).2.zf = some false; rw [aluF_zf, aluF_and_fst_self]; simp [he0]
      rw [cond_cc5 hz]; rfl
    have e6 : X86Int.step im s5 = .ok { s5 with rip := aJmp + 30 } := by
      have hh := step_jcc (im := im) (s := s5) (by rw [hs5rip]; exact cfg.jmp.d6) hs5fl
      simpa using hh
    obtain ⟨s6, hs6⟩ : ∃ u : X86Int.St, u = { s5 with rip := aJmp + 30 } := ⟨_, rfl⟩
    have hs6rip : s6.rip = aJmp + 30 := by rw [hs6]
    have hs6r0 : s6.reg 0 = some e := by rw [hs6, hs5]; exact hs4r0
    have hs6r4 : s6.reg 4 = some sp := by rw [hs6, hs5]; exact hs4r4
    have hs6r8 : s6.reg 8 = some hm.d := by rw [hs6, hs5]; exact hs4r8
    have hs6r9 : s6.reg 9 = some hm.a := by rw [hs6, hs5]; exact hs4r9
    have hs6mem : s6.mem = s4.mem := by rw [hs6, hs5]
    have hs6len : 16 ≤ s6.regs.length := by rw [hs6, hs5]; exact hlen4s
    have hslot8 : (sp + W - 8) % W = sp - 8 := by
      rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) st.spW]
    obtain ⟨m7, hm7s⟩ := Option.isSome_iff_exists.mp
      (by have hh := hroom4 sp hs4r4 0 (by decide) e; rw [hslot8, ← hs6mem] at hh; exact hh)
    have hm7p : wr64 s6.mem ((sp + W - 8) % W) e = some m7 := by rw [hslot8]; exact hm7s
    have e7 : X86Int.step im s6 = .ok { s6.set 4 (some ((sp + W - 8) % W)) with mem := m7, rip := s6.rip + 1 } :=
      step_push (by rw [hs6rip]; exact cfg.jmp.d7) hs6r0 hs6r4 hm7p
    obtain ⟨s7, hs7⟩ : ∃ u : X86Int.St, u = { s6.set 4 (some ((sp + W - 8) % W)) with mem := m7, rip := s6.rip + 1 } := ⟨_, rfl⟩
    have hs7rip : s7.rip = aJmp + 31 := by rw [hs7]; show s6.rip + 1 = aJmp + 31; rw [hs6rip]
    have hs7r4 : s7.reg 4 = some (sp - 8) := by
      rw [hs7]; show (s6.set 4 _).reg 4 = some (sp - 8)
      rw [reg_set_self (Nat.lt_of_lt_of_le (by decide) hs6len), hslot8]
    have gs7 : ∀ q, q ≠ 4 → s7.reg q = s6.reg q := by
      intro q hq; rw [hs7]; show (s6.set 4 _).reg q = s6.reg q; rw [reg_set_other hq]
    have hs7r0 : s7.reg 0 = some e := by rw [gs7 0 (by decide)]; exact hs6r0
    have hs7mem : s7.mem = m7 := by rw [hs7]
    have hlen7 : 16 ≤ s7.regs.length := by rw [hs7]; show 16 ≤ (s6.set 4 _).regs.length; rw [len_set]; exact hs6len
    have hws7 : wshape s7.mem = wshape s.mem := by rw [hs7mem]; exact (wrN_wshape _ _ _ hm7s).trans (by rw [hs6mem]; exact hws4)
    have hat7 : AnnAt s7 e t a.d a.a :=
      ⟨by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.1,
       by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.2.1,
       by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.2.2⟩
    have hslot8_7 : rd64 s7.mem (sp - 8) = some e := by
      have hv := rd64_wr64 hm7s; rw [Nat.mod_eq_of_lt (by omega)] at hv; rw [hs7mem]; exact hv
    have e8 : X86Int.step im s7 = .ok { s7.set 2 (some e) with rip := s7.rip + 3 } :=
      step_mov (by rw [hs7rip]; exact cfg.jmp.d8) hs7r0
    obtain ⟨s8, hs8⟩ : ∃ u : X86Int.St, u = { s7.set 2 (some e) with rip := s7.rip + 3 } := ⟨_, rfl⟩
    have hs8rip : s8.rip = aJmp + 34 := by rw [hs8]; show s7.rip + 3 = aJmp + 34; rw [hs7rip]
    have hs8r2 : s8.reg 2 = some e := by rw [hs8]; show (s7.set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
    have gs8 : ∀ q, q ≠ 2 → s8.reg q = s7.reg q := by
      intro q hq; rw [hs8]; show (s7.set 2 _).reg q = s7.reg q; rw [reg_set_other hq]
    have hlen8 : 16 ≤ s8.regs.length := by rw [hs8]; show 16 ≤ (s7.set 2 _).regs.length; rw [len_set]; exact hlen7
    have hadd4 : (aluF .add e (4 % W)).1 = e + 4 := aluI_add_val h4W (by omega)
    have e9 : X86Int.step im s8 = .ok { s8.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := s8.rip + 4 } :=
      step_aluI_add (by rw [hs8rip]; exact cfg.jmp.d9) hs8r2
    obtain ⟨s9, hs9⟩ : ∃ u : X86Int.St, u = { s8.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := s8.rip + 4 } := ⟨_, rfl⟩
    have hs9rip : s9.rip = aJmp + 38 := by rw [hs9]; show s8.rip + 4 = aJmp + 38; rw [hs8rip]
    have hs9r2 : s9.reg 2 = some (e + 4) := by rw [hs9]; show (s8.set 2 _).reg 2 = some (e + 4); rw [reg_set_self (by omega), hadd4]
    have gs9 : ∀ q, q ≠ 2 → s9.reg q = s8.reg q := by
      intro q hq; rw [hs9]; show (s8.set 2 _).reg q = s8.reg q; rw [reg_set_other hq]
    have hlen9 : 16 ≤ s9.regs.length := by rw [hs9]; show 16 ≤ (s8.set 2 _).regs.length; rw [len_set]; exact hlen8
    have hs9r4 : s9.reg 4 = some (sp - 8) := by rw [gs9 4 (by decide), gs8 4 (by decide)]; exact hs7r4
    have hs9mem : s9.mem = m7 := by rw [hs9, hs8]; exact hs7mem
    have hws9 : wshape s9.mem = wshape s.mem := by rw [hs9mem, ← hs7mem]; exact hws7
    have hroom9 : StackRoom s9 1 :=
      StackRoom_shift4 hroom4 hs4r4 hs9r4 (hws9.trans hws4.symm) ⟨1, by omega, by omega⟩ hsp32 hspW
    obtain ⟨k10, s10, hk10, hr10, hs10rip, hs10r0, hs10r4, gs10, hlen10, hfr10seg, hw10seg⟩ :=
      CallL32B cfg.l32 cfg.jmp.dret cfg.jmp.d10 (by omega) (by omega) (by omega) (by decide)
        (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
        s9 hs9rip hs9r2 hs9r4 (by rw [hs9mem, ← hs7mem]; exact hat7.2.1) hroom9
    have hs10r8 : s10.reg 8 = some hm.d := by
      rw [gs10 8 (by decide) (by decide), gs9 8 (by decide), gs8 8 (by decide), gs7 8 (by decide)]; exact hs6r8
    have hs10r9 : s10.reg 9 = some hm.a := by
      rw [gs10 9 (by decide) (by decide), gs9 9 (by decide), gs8 9 (by decide), gs7 9 (by decide)]; exact hs6r9
    have hws10 : wshape s10.mem = wshape s.mem := hw10seg.trans hws9
    have e11 : X86Int.step im s10 = .ok { s10 with fl := (aluF .sub a.d hm.d).2, rip := s10.rip + 3 } :=
      step_cmp (by rw [hs10rip]; exact cfg.jmp.d11) hs10r0 hs10r8
    obtain ⟨s11, hs11⟩ : ∃ u : X86Int.St, u = { s10 with fl := (aluF .sub a.d hm.d).2, rip := s10.rip + 3 } := ⟨_, rfl⟩
    have hs11rip : s11.rip = aJmp + 46 := by rw [hs11]; show s10.rip + 3 = aJmp + 46; rw [hs10rip]
    have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
    have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
    have r7 : run im 1 (Out.ok s6) = Out.ok s7 := by rw [run_succ_ok im 0 s6, e7, ← hs7, run_zero]
    have r8 : run im 1 (Out.ok s7) = Out.ok s8 := by rw [run_succ_ok im 0 s7, e8, ← hs8, run_zero]
    have r9 : run im 1 (Out.ok s8) = Out.ok s9 := by rw [run_succ_ok im 0 s8, e9, ← hs9, run_zero]
    have r11 : run im 1 (Out.ok s10) = Out.ok s11 := by rw [run_succ_ok im 0 s10, e11, ← hs11, run_zero]
    have hrun9 : run im ((k1 + k2 + 1 + k4) + 5) (Out.ok s) = Out.ok s9 := by
      rw [run_add im _ 5, hrun4, run_add im 1 4, r5, run_add im 1 3, r6, run_add im 1 2, r7, run_add im 1 1, r8, r9]
    have hrun10 : run im ((k1 + k2 + 1 + k4) + 5 + k10) (Out.ok s) = Out.ok s10 := by
      rw [run_add im _ k10, hrun9, hr10]
    have hrun11 : run im ((k1 + k2 + 1 + k4) + 5 + k10 + 1) (Out.ok s) = Out.ok s11 := by
      rw [run_add im _ 1, hrun10, r11]
    by_cases hdeq : a.d = hm.d
    · have ha : a.a ≠ hm.a := fun ha => hnea (by cases a; cases hm; simp at hdeq ha; simp [hdeq, ha])
      have hs11fl : X86Int.cond s11.fl 4 = some true := by
        have hzf : s11.fl.zf = some true := by rw [hs11]; exact aluF_sub_zf_eq adW hdW hdeq
        exact cond_cc4 hzf
      have e12 : X86Int.step im s11 = .ok { s11 with rip := aJmp + 53 } := by
        have hh := step_jcc (im := im) (s := s11) (by rw [hs11rip]; exact cfg.jmp.d12) hs11fl
        simpa using hh
      obtain ⟨s12, hs12⟩ : ∃ u : X86Int.St, u = { s11 with rip := aJmp + 53 } := ⟨_, rfl⟩
      have hs12rip : s12.rip = aJmp + 53 := by rw [hs12]
      have hs12r4 : s12.reg 4 = some (sp - 8) := by rw [hs12, hs11]; exact hs10r4
      have hs12mem : s12.mem = s10.mem := by rw [hs12, hs11]
      have hslot12 : rd64 s12.mem (sp - 8) = some e := by
        rw [hs12mem]; unfold rd64; rw [hfr10seg (sp - 8) 8 (by omega), hs9mem, ← hs7mem]; exact hslot8_7
      have hp8 : ((sp - 8) + 8) % W = sp := slot_add8 (by omega) st.spW
      have e13 : X86Int.step im s12 = .ok { (s12.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := s12.rip + 1 } :=
        step_pop (by rw [hs12rip]; exact cfg.jmp.d13) hs12r4 hslot12
      obtain ⟨s13, hs13⟩ : ∃ u : X86Int.St, u = { (s12.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := s12.rip + 1 } := ⟨_, rfl⟩
      have hs13rip : s13.rip = aJmp + 54 := by rw [hs13]; show s12.rip + 1 = aJmp + 54; rw [hs12rip]
      have hs12len : 16 ≤ s12.regs.length := by rw [hs12, hs11, hlen10]; exact hlen9
      have hs13r4 : s13.reg 4 = some sp := by
        rw [hs13]; show ((s12.set 4 _).set 2 _).reg 4 = some sp
        rw [reg_set_other (by decide), reg_set_self (by omega), hp8]
      have hs13r2 : s13.reg 2 = some e := by
        rw [hs13]; show ((s12.set 4 _).set 2 _).reg 2 = some e
        rw [reg_set_self (by show 2 < (s12.set 4 _).regs.length; rw [len_set]; exact Nat.lt_of_lt_of_le (by decide) hs12len)]
      have gs13 : ∀ q, q ≠ 2 → q ≠ 4 → s13.reg q = s12.reg q := by
        intro q h2 h4; rw [hs13]; show ((s12.set 4 _).set 2 _).reg q = s12.reg q
        rw [reg_set_other h2, reg_set_other h4]
      have hlen13 : 16 ≤ s13.regs.length := by
        rw [hs13]; show 16 ≤ ((s12.set 4 _).set 2 _).regs.length; rw [len_set, len_set]; exact hs12len
      have hadd8 : (aluF .add e (8 % W)).1 = e + 8 := aluI_add_val h8W (by omega)
      have e14 : X86Int.step im s13 = .ok { s13.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := s13.rip + 4 } :=
        step_aluI_add (by rw [hs13rip]; exact cfg.jmp.d14) hs13r2
      obtain ⟨s14, hs14⟩ : ∃ u : X86Int.St, u = { s13.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := s13.rip + 4 } := ⟨_, rfl⟩
      have hs14rip : s14.rip = aJmp + 58 := by rw [hs14]; show s13.rip + 4 = aJmp + 58; rw [hs13rip]
      have hs14r2 : s14.reg 2 = some (e + 8) := by rw [hs14]; show (s13.set 2 _).reg 2 = some (e + 8); rw [reg_set_self (by omega), hadd8]
      have gs14 : ∀ q, q ≠ 2 → s14.reg q = s13.reg q := by
        intro q hq; rw [hs14]; show (s13.set 2 _).reg q = s13.reg q; rw [reg_set_other hq]
      have hs14r4 : s14.reg 4 = some sp := by rw [gs14 4 (by decide)]; exact hs13r4
      have hs14r9 : s14.reg 9 = some hm.a := by rw [gs14 9 (by decide), gs13 9 (by decide) (by decide), hs12, hs11]; exact hs10r9
      have hs13mem : s13.mem = s12.mem := by rw [hs13]; rfl
      have hs14mem : s14.mem = s10.mem := by rw [hs14]; exact hs13mem.trans hs12mem
      have hws14 : wshape s14.mem = wshape s.mem := by rw [hs14mem]; exact hws10
      have hat14a : (s14.mem.rdN (e + 8) 4).map leVal = some a.a := by
        rw [hs14mem, hfr10seg (e + 8) 4 (by omega), hs9mem, ← hs7mem]; exact hat7.2.2
      obtain ⟨k15, s15, hk15, hr15, hs15rip, hs15r0, hs15r4, gs15, _, _, hw15seg⟩ :=
        CallL32B cfg.l32 cfg.jmp.dret cfg.jmp.d15 (by omega) (by omega) hspW (by decide)
          (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
          s14 hs14rip hs14r2 hs14r4 hat14a
          (StackRoom_shift4 hroom4 hs4r4 (L := 0) (by simpa using hs14r4) (hws14.trans hws4.symm) ⟨0, rfl, by decide⟩ hsp32 hspW)
      have hs15r9 : s15.reg 9 = some hm.a := by rw [gs15 9 (by decide) (by decide)]; exact hs14r9
      have hws15 : wshape s15.mem = wshape s.mem := hw15seg.trans hws14
      have e16 : X86Int.step im s15 = .ok { s15 with fl := (aluF .sub a.a hm.a).2, rip := s15.rip + 3 } :=
        step_cmp (by rw [hs15rip]; exact cfg.jmp.d16) hs15r0 hs15r9
      obtain ⟨s16, hs16⟩ : ∃ u : X86Int.St, u = { s15 with fl := (aluF .sub a.a hm.a).2, rip := s15.rip + 3 } := ⟨_, rfl⟩
      have hs16rip : s16.rip = aJmp + 66 := by rw [hs16]; show s15.rip + 3 = aJmp + 66; rw [hs15rip]
      have hs16fl : X86Int.cond s16.fl 4 = some false := by
        have hzf : s16.fl.zf = some false := by rw [hs16]; exact aluF_sub_zf_ne aaW haW ha
        exact cond_cc4 hzf
      have e17 : X86Int.step im s16 = .ok { s16 with rip := aJmp + 68 } := by
        have hh := step_jcc (im := im) (s := s16) (by rw [hs16rip]; exact cfg.jmp.d17) hs16fl
        simpa [hs16rip] using hh
      obtain ⟨s17, hs17⟩ : ∃ u : X86Int.St, u = { s16 with rip := aJmp + 68 } := ⟨_, rfl⟩
      have r12 : run im 1 (Out.ok s11) = Out.ok s12 := by rw [run_succ_ok im 0 s11, e12, ← hs12, run_zero]
      have r13 : run im 1 (Out.ok s12) = Out.ok s13 := by rw [run_succ_ok im 0 s12, e13, ← hs13, run_zero]
      have r14 : run im 1 (Out.ok s13) = Out.ok s14 := by rw [run_succ_ok im 0 s13, e14, ← hs14, run_zero]
      have r16 : run im 1 (Out.ok s15) = Out.ok s16 := by rw [run_succ_ok im 0 s15, e16, ← hs16, run_zero]
      have r17 : run im 1 (Out.ok s16) = Out.ok s17 := by rw [run_succ_ok im 0 s16, e17, ← hs17, run_zero]
      have hrun14 : run im (((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 3) (Out.ok s) = Out.ok s14 := by
        rw [run_add im _ 3, hrun11, run_add im 1 2, r12, run_add im 1 1, r13, r14]
      have hrun15 : run im (((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 3 + k15) (Out.ok s) = Out.ok s15 := by
        rw [run_add im _ k15, hrun14, hr15]
      have hrun17 : run im (((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 3 + k15 + 2) (Out.ok s) = Out.ok s17 := by
        rw [run_add im _ 2, hrun15, run_add im 1 1, r16, r17]
      have hs17rip : s17.rip = aJmp + 68 := by rw [hs17]
      have hs17r4 : s17.reg 4 = some sp := by rw [hs17, hs16]; exact hs15r4
      have hws17 : wshape s17.mem = wshape s.mem := by rw [hs17, hs16]; exact hws15
      obtain ⟨k18, hk18, hr18⟩ := jmp_fail1_exit1B rc cr.jmpAnnA st.sp32 st.spW s17 hs17rip hs17r4
        (run_regs_ok hrun17 hregs)
        (StackRoom_mono (StackRoom_of_wshape hws17 (by rw [hs17r4, inv.r4]) inv.room) (by decide))
      exact ⟨(((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 3 + k15 + 2) + k18, (by omega),
        by rw [run_add im _ k18, hrun17]; exact hr18⟩
    · -- a.d ≠ hm.d -> cr.jmpAnnD (stack at sp - 8)
      have hs11fl : X86Int.cond s11.fl 4 = some false := by
        have hzf : s11.fl.zf = some false := by rw [hs11]; exact aluF_sub_zf_ne adW hdW hdeq
        exact cond_cc4 hzf
      have e12 : X86Int.step im s11 = .ok { s11 with rip := aJmp + 48 } := by
        have hh := step_jcc (im := im) (s := s11) (by rw [hs11rip]; exact cfg.jmp.d12) hs11fl
        simpa [hs11rip] using hh
      obtain ⟨s12, hs12⟩ : ∃ u : X86Int.St, u = { s11 with rip := aJmp + 48 } := ⟨_, rfl⟩
      have r12 : run im 1 (Out.ok s11) = Out.ok s12 := by rw [run_succ_ok im 0 s11, e12, ← hs12, run_zero]
      have hrun12 : run im (((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 1) (Out.ok s) = Out.ok s12 := by
        rw [run_add im _ 1, hrun11, r12]
      have hs12rip : s12.rip = aJmp + 48 := by rw [hs12]
      have hs12r4 : s12.reg 4 = some (sp - 8) := by rw [hs12, hs11]; exact hs10r4
      have hws12 : wshape s12.mem = wshape s.mem := by rw [hs12, hs11]; exact hws10
      obtain ⟨k13, hk13, hr13⟩ := jmp_fail1_exit1B rc cr.jmpAnnD (by omega) (by omega) s12 hs12rip hs12r4
        (run_regs_ok hrun12 hregs)
        (StackRoom_shift4 hroom4 hs4r4 hs12r4 (hws12.trans hws4.symm) ⟨1, by omega, by omega⟩ hsp32 hspW)
      exact ⟨(((k1 + k2 + 1 + k4) + 5 + k10 + 1) + 1) + k13, (by omega),
        by rw [run_add im _ k13, hrun12]; exact hr13⟩

set_option maxHeartbeats 800000 in
/-- Shared refusal handler at `aJz` (used for both `jz` and `jnz`) when
`hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩`. -/
private theorem jz_handler_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl c t : Nat} {hm : Height}
    (st : WalkStatic pr wd w p sp tbl) (hsp48 : 48 ≤ sp)
    (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    (hdW : hm.d < W) (haW : hm.a < W) (htW : t < W)
    (hc5W : c + 5 < W) (hdisjC5 : sp ≤ c + 1 ∨ c + 5 ≤ sp - 8)
    (hfail : hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩) :
    ∀ s1 : X86Int.St, s1.rip = aJz → s1.reg 8 = some hm.d → s1.reg 9 = some hm.a →
      s1.reg 6 = some c → s1.reg 4 = some sp → s1.reg 7 = some w → s1.reg 1 = some i →
      16 ≤ s1.regs.length → RegsOk s1 →
      (s1.mem.rdN (c + 1) 4).map leVal = some t →
      rd64 s1.mem (w + 48) = some p → rd64 s1.mem (w + 32) = some wd.ann.length →
      AnnLayout s1 wd.ann p → StackRoom s1 4 →
      ∃ m, m ≤ 60 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s1) = Out.exit 1 := by
  intro s1 hrip1 hr81 hr91 hr61 hr41 hr71 hr11 hlen1 hregs1 himm hfp1 hfn1 hlay1 hroom1
  obtain ⟨k2, s2, hk2, hr2, hrip2, hs2r0, hs2r4, gs2, hfr2, hw2, hlen2⟩ :=
    run_load_imm32 cfg.l32 cfg.jz.dret cfg.jz.d0 cfg.jz.d1 cfg.jz.d2
      (by omega) (by omega) st.spW (by decide) hdisjC5.symm
      s1 hrip1 hr61 hr41 hlen1 himm (StackRoom_mono hroom1 (by decide))
  have hs2r8 : s2.reg 8 = some hm.d := by rw [gs2 8 (by decide) (by decide) (by decide)]; exact hr81
  have hs2r9 : s2.reg 9 = some hm.a := by rw [gs2 9 (by decide) (by decide) (by decide)]; exact hr91
  have hs2r7 : s2.reg 7 = some w := by rw [gs2 7 (by decide) (by decide) (by decide)]; exact hr71
  have hs2r1 : s2.reg 1 = some i := by rw [gs2 1 (by decide) (by decide) (by decide)]; exact hr11
  have e2a : X86Int.step im s2 = .ok { s2 with fl := (aluF .sub hm.d (1 % W)).2, rip := s2.rip + 4 } :=
    step_cmpI (by rw [hrip2]; exact cfg.jz.d2a) hs2r8
  obtain ⟨s2a, hs2a⟩ : ∃ u : X86Int.St, u = { s2 with fl := (aluF .sub hm.d (1 % W)).2, rip := s2.rip + 4 } := ⟨_, rfl⟩
  have hs2arip : s2a.rip = aJz + 16 := by rw [hs2a]; show s2.rip + 4 = aJz + 16; rw [hrip2]
  have r2a : run im 1 (Out.ok s2) = Out.ok s2a := by rw [run_succ_ok im 0 s2, e2a, ← hs2a, run_zero]
  by_cases hd0 : hm.d = 0
  · have hs2afl : X86Int.cond s2a.fl 3 = some false := by
      have hcf : s2a.fl.cf = some true := by rw [hs2a, hd0]; exact aluF_sub_cf_true (by unfold W; decide)
      rw [cond_cc3 hcf]; rfl
    have e2b : X86Int.step im s2a = .ok { s2a with rip := aJz + 18 } := by
      have hh := step_jcc (im := im) (s := s2a) (by rw [hs2arip]; exact cfg.jz.d2b) hs2afl
      simpa [hs2arip] using hh
    obtain ⟨s2b, hs2b⟩ : ∃ u : X86Int.St, u = { s2a with rip := aJz + 18 } := ⟨_, rfl⟩
    have r2b : run im 1 (Out.ok s2a) = Out.ok s2b := by rw [run_succ_ok im 0 s2a, e2b, ← hs2b, run_zero]
    have hrun2b : run im (k2 + 2) (Out.ok s1) = Out.ok s2b := by
      rw [run_add im k2 2, hr2, run_add im 1 1, r2a, r2b]
    have hs2brip : s2b.rip = aJz + 18 := by rw [hs2b]
    have hs2br4 : s2b.reg 4 = some sp := by rw [hs2b, hs2a]; exact hs2r4
    have hws2b : wshape s2b.mem = wshape s1.mem := by rw [hs2b, hs2a]; exact hw2
    obtain ⟨k3, hk3, hr3⟩ := jmp_fail1_exit1B rc cr.jzUnd st.sp32 st.spW s2b hs2brip hs2br4
      (run_regs_ok hrun2b hregs1)
      (StackRoom_mono (StackRoom_of_wshape hws2b (by rw [hs2br4, hr41]) hroom1) (by decide))
    exact ⟨k2 + 2 + k3, (by omega), by rw [run_add im (k2 + 2) k3, hrun2b]; exact hr3⟩
  · have hne : wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩ := by
      cases hfail with
      | inl h => exact absurd h hd0
      | inr h => exact h
    have hs2afl : X86Int.cond s2a.fl 3 = some true := by
      have hcf : s2a.fl.cf = some false := by
        rw [hs2a]; exact aluF_sub_cf (by rw [show (1 : Nat) % W = 1 from Nat.mod_eq_of_lt (by decide)]; omega)
      exact cond_cc3 hcf
    have e2b : X86Int.step im s2a = .ok { s2a with rip := aJz + 23 } := by
      have hh := step_jcc (im := im) (s := s2a) (by rw [hs2arip]; exact cfg.jz.d2b) hs2afl
      simpa using hh
    obtain ⟨s2b, hs2b⟩ : ∃ u : X86Int.St, u = { s2a with rip := aJz + 23 } := ⟨_, rfl⟩
    have hs2brip : s2b.rip = aJz + 23 := by rw [hs2b]
    have hs2br0 : s2b.reg 0 = some t := by rw [hs2b, hs2a]; exact hs2r0
    have hs2br4 : s2b.reg 4 = some sp := by rw [hs2b, hs2a]; exact hs2r4
    have hs2br7 : s2b.reg 7 = some w := by rw [hs2b, hs2a]; exact hs2r7
    have hs2br1 : s2b.reg 1 = some i := by rw [hs2b, hs2a]; exact hs2r1
    have hs2br8 : s2b.reg 8 = some hm.d := by rw [hs2b, hs2a]; exact hs2r8
    have hs2br9 : s2b.reg 9 = some hm.a := by rw [hs2b, hs2a]; exact hs2r9
    have hs2bmem : s2b.mem = s2.mem := by rw [hs2b, hs2a]
    have hlen2b : 16 ≤ s2b.regs.length := by rw [hs2b, hs2a]; show 16 ≤ s2.regs.length; omega
    -- mov r12, rax
    have e3 : X86Int.step im s2b = .ok { s2b.set 12 (some t) with rip := s2b.rip + 3 } :=
      step_mov (by rw [hs2brip]; exact cfg.jz.d3) hs2br0
    obtain ⟨s3, hs3⟩ : ∃ u : X86Int.St, u = { s2b.set 12 (some t) with rip := s2b.rip + 3 } := ⟨_, rfl⟩
    have hs3rip : s3.rip = aJz + 26 := by rw [hs3]; show s2b.rip + 3 = aJz + 26; rw [hs2brip]
    have hs3r12 : s3.reg 12 = some t := by rw [hs3]; show (s2b.set 12 _).reg 12 = _; rw [reg_set_self (by omega)]
    have gs3 : ∀ q, q ≠ 12 → s3.reg q = s2b.reg q := by
      intro q hq; rw [hs3]; show (s2b.set 12 _).reg q = s2b.reg q; rw [reg_set_other hq]
    have hs3mem : s3.mem = s2.mem := by rw [hs3]; exact hs2bmem
    have hs3r1 : s3.reg 1 = some i := by rw [gs3 1 (by decide)]; exact hs2br1
    have hs3r7 : s3.reg 7 = some w := by rw [gs3 7 (by decide)]; exact hs2br7
    have hs3r4 : s3.reg 4 = some sp := by rw [gs3 4 (by decide)]; exact hs2br4
    have hs3r8 : s3.reg 8 = some hm.d := by rw [gs3 8 (by decide)]; exact hs2br8
    have hs3r9 : s3.reg 9 = some hm.a := by rw [gs3 9 (by decide)]; exact hs2br9
    have hs3len : 16 ≤ s3.regs.length := by rw [hs3]; show 16 ≤ (s2b.set 12 _).regs.length; rw [len_set]; exact hlen2b
    have hfp3 : rd64 s3.mem (w + 48) = some p := by
      rw [hs3mem]; unfold rd64; rw [hfr2 (w + 48) 8 (by cases st.disjw with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))]
      exact hfp1
    have hfn3 : rd64 s3.mem (w + 32) = some wd.ann.length := by
      rw [hs3mem]; unfold rd64; rw [hfr2 (w + 32) 8 (by cases st.disjw with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))]
      exact hfn1
    have hlay3 : AnnLayout s3 wd.ann p :=
      AnnLayout_of_mem hs3mem (AnnLayout_of_frame hlay1 (by
        intro x k hx1 hx2; exact hfr2 x k (by cases st.disjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))))
    have hws3 : wshape s3.mem = wshape s1.mem := by rw [hs3mem]; exact hw2
    have hroom3 : StackRoom s3 4 := StackRoom_of_wshape hws3 (by rw [hs3r4, hr41]) hroom1
    -- call ANNL
    obtain ⟨k4, s4, hk4, hr4, hs4rip, hpost, _, hs4r4, _, _, _, gs4, _, hex0, hloc4, hw4seg, hlen4⟩ :=
      CallANNLB cfg.annl cfg.l32 (by decide) cfg.jz.d4 (by decide) htW st.p0 st.w56
        st.pann st.sp32 st.spW s3 hs3rip hs3r12 hs3r7 hs3r4 hs3r1 hfp3 hfn3 hlay3 hroom3 st.disjw st.disjp
    obtain ⟨e, hs4r0⟩ := hex0
    have r2b : run im 1 (Out.ok s2a) = Out.ok s2b := by rw [run_succ_ok im 0 s2a, e2b, ← hs2b, run_zero]
    have r3 : run im 1 (Out.ok s2b) = Out.ok s3 := by rw [run_succ_ok im 0 s2b, e3, ← hs3, run_zero]
    have hrun3 : run im (k2 + 3) (Out.ok s1) = Out.ok s3 := by
      rw [run_add im k2 3, hr2, run_add im 1 2, r2a, run_add im 1 1, r2b, r3]
    have hrun4 : run im (k2 + 3 + k4) (Out.ok s1) = Out.ok s4 := by
      rw [run_add im (k2 + 3) k4, hrun3, hr4]
    have hs4r8 : s4.reg 8 = some hm.d := by rw [gs4 8 (by decide) (by decide) (by decide) (by decide)]; exact hs3r8
    have hs4r9 : s4.reg 9 = some hm.a := by rw [gs4 9 (by decide) (by decide) (by decide) (by decide)]; exact hs3r9
    have hws4 : wshape s4.mem = wshape s1.mem := hw4seg.trans hws3
    have hroom4 : StackRoom s4 4 := StackRoom_of_wshape hws4 (by rw [hs4r4, hr41]) hroom1
    have hlen4s : 16 ≤ s4.regs.length := by rw [hlen4]; exact hs3len
    have h1W : 1 < W := by decide
    have h4W : 4 < W := by decide
    have h8W : 8 < W := by decide
    have hs4rip' : s4.rip = aJz + 31 := by rw [hs4rip]
    cases hlk : wd.ann.lookup t with
    | none =>
      have he0 : e = 0 := by
        cases hpost e hs4r0 with
        | inl hb => exact hb.1
        | inr hb => obtain ⟨d1, a1, -, -, hlk1⟩ := hb; rw [hlk] at hlk1; simp at hlk1
      have hs4r0' : s4.reg 0 = some 0 := by rw [hs4r0, he0]
      have e5 : X86Int.step im s4 = .ok { s4 with fl := (aluF .and 0 0).2, rip := s4.rip + 3 } :=
        step_test (by rw [hs4rip']; exact cfg.jz.d5) hs4r0' hs4r0'
      obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4 with fl := (aluF .and 0 0).2, rip := s4.rip + 3 } := ⟨_, rfl⟩
      have hs5rip : s5.rip = aJz + 34 := by rw [hs5]; show s4.rip + 3 = aJz + 34; rw [hs4rip']
      have hs5fl : X86Int.cond s5.fl 5 = some false := by
        have hz : s5.fl.zf = some true := by
          rw [hs5]; show (aluF .and 0 0).2.zf = some true; rw [aluF_zf, aluF_and_fst_self]; exact rfl
        rw [cond_cc5 hz]; rfl
      have e6 : X86Int.step im s5 = .ok { s5 with rip := aJz + 36 } := by
        have hh := step_jcc (im := im) (s := s5) (by rw [hs5rip]; exact cfg.jz.d6) hs5fl
        simpa [hs5rip] using hh
      obtain ⟨s6, hs6⟩ : ∃ u : X86Int.St, u = { s5 with rip := aJz + 36 } := ⟨_, rfl⟩
      have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
      have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
      have hrun6 : run im ((k2 + 3 + k4) + 2) (Out.ok s1) = Out.ok s6 := by
        rw [run_add im _ 2, hrun4, run_add im 1 1, r5, r6]
      have hs6rip : s6.rip = aJz + 36 := by rw [hs6]
      have hs6r4 : s6.reg 4 = some sp := by rw [hs6, hs5]; exact hs4r4
      have hws6 : wshape s6.mem = wshape s1.mem := by rw [hs6, hs5]; exact hws4
      obtain ⟨k7, hk7, hr7⟩ := jmp_fail1_exit1B rc cr.jzUnann st.sp32 st.spW s6 hs6rip hs6r4
        (run_regs_ok hrun6 hregs1) (StackRoom_mono (StackRoom_of_wshape hws6 (by rw [hs6r4, hr41]) hroom1) (by decide))
      exact ⟨(k2 + 3 + k4) + 2 + k7, (by omega), by rw [run_add im _ k7, hrun6]; exact hr7⟩
    | some a =>
      have hnea : a ≠ ⟨hm.d - 1, hm.a⟩ := fun heq => hne (by rw [hlk, heq])
      obtain ⟨adW, aaW⟩ := hhW (t, a) (lookup_mem hlk)
      obtain ⟨he0, hat4⟩ : e ≠ 0 ∧ AnnAt s4 e t a.d a.a := by
        cases hpost e hs4r0 with
        | inl hb => rw [hb.2] at hlk; simp at hlk
        | inr hb =>
          obtain ⟨d1, a1, hne0, hat, hlk1⟩ := hb
          have hhh : (⟨d1, a1⟩ : Height) = a := Option.some.inj (hlk1.symm.trans hlk)
          exact ⟨hne0, by rwa [show d1 = a.d from congrArg Height.d hhh, show a1 = a.a from congrArg Height.a hhh] at hat⟩
      obtain ⟨helo, hehi⟩ := hloc4 e hs4r0 he0
      have hpann := st.pann
      have hdisjp := st.disjp
      have hspW := st.spW
      have hsp32 := st.sp32
      -- test ; jnz taken -> push rax -> CallL32B a.d
      have e5 : X86Int.step im s4 = .ok { s4 with fl := (aluF .and e e).2, rip := s4.rip + 3 } :=
        step_test (by rw [hs4rip']; exact cfg.jz.d5) hs4r0 hs4r0
      obtain ⟨s5, hs5⟩ : ∃ u : X86Int.St, u = { s4 with fl := (aluF .and e e).2, rip := s4.rip + 3 } := ⟨_, rfl⟩
      have hs5rip : s5.rip = aJz + 34 := by rw [hs5]; show s4.rip + 3 = aJz + 34; rw [hs4rip']
      have hs5fl : X86Int.cond s5.fl 5 = some true := by
        have hz : s5.fl.zf = some false := by
          rw [hs5]; show (aluF .and e e).2.zf = some false; rw [aluF_zf, aluF_and_fst_self]; simp [he0]
        rw [cond_cc5 hz]; rfl
      have e6 : X86Int.step im s5 = .ok { s5 with rip := aJz + 41 } := by
        have hh := step_jcc (im := im) (s := s5) (by rw [hs5rip]; exact cfg.jz.d6) hs5fl
        simpa using hh
      obtain ⟨s6, hs6⟩ : ∃ u : X86Int.St, u = { s5 with rip := aJz + 41 } := ⟨_, rfl⟩
      have hs6rip : s6.rip = aJz + 41 := by rw [hs6]
      have hs6r0 : s6.reg 0 = some e := by rw [hs6, hs5]; exact hs4r0
      have hs6r4 : s6.reg 4 = some sp := by rw [hs6, hs5]; exact hs4r4
      have hs6r8 : s6.reg 8 = some hm.d := by rw [hs6, hs5]; exact hs4r8
      have hs6r9 : s6.reg 9 = some hm.a := by rw [hs6, hs5]; exact hs4r9
      have hs6mem : s6.mem = s4.mem := by rw [hs6, hs5]
      have hs6len : 16 ≤ s6.regs.length := by rw [hs6, hs5]; exact hlen4s
      have hslot8 : (sp + W - 8) % W = sp - 8 := by
        rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) st.spW]
      obtain ⟨m7, hm7s⟩ := Option.isSome_iff_exists.mp
        (by have hh := hroom4 sp hs4r4 0 (by decide) e; rw [hslot8, ← hs6mem] at hh; exact hh)
      have hm7p : wr64 s6.mem ((sp + W - 8) % W) e = some m7 := by rw [hslot8]; exact hm7s
      have e7 : X86Int.step im s6 = .ok { s6.set 4 (some ((sp + W - 8) % W)) with mem := m7, rip := s6.rip + 1 } :=
        step_push (by rw [hs6rip]; exact cfg.jz.d7) hs6r0 hs6r4 hm7p
      obtain ⟨s7, hs7⟩ : ∃ u : X86Int.St, u = { s6.set 4 (some ((sp + W - 8) % W)) with mem := m7, rip := s6.rip + 1 } := ⟨_, rfl⟩
      have hs7rip : s7.rip = aJz + 42 := by rw [hs7]; show s6.rip + 1 = aJz + 42; rw [hs6rip]
      have hs7r4 : s7.reg 4 = some (sp - 8) := by
        rw [hs7]; show (s6.set 4 _).reg 4 = some (sp - 8)
        rw [reg_set_self (Nat.lt_of_lt_of_le (by decide) hs6len), hslot8]
      have gs7 : ∀ q, q ≠ 4 → s7.reg q = s6.reg q := by
        intro q hq; rw [hs7]; show (s6.set 4 _).reg q = s6.reg q; rw [reg_set_other hq]
      have hs7r0 : s7.reg 0 = some e := by rw [gs7 0 (by decide)]; exact hs6r0
      have hs7mem : s7.mem = m7 := by rw [hs7]
      have hlen7 : 16 ≤ s7.regs.length := by rw [hs7]; show 16 ≤ (s6.set 4 _).regs.length; rw [len_set]; exact hs6len
      have hws7 : wshape s7.mem = wshape s1.mem := by rw [hs7mem]; exact (wrN_wshape _ _ _ hm7s).trans (by rw [hs6mem]; exact hws4)
      have hat7 : AnnAt s7 e t a.d a.a :=
        ⟨by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.1,
         by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.2.1,
         by rw [hs7mem, rdN_wr64_other hm7s (by omega), hs6mem]; exact hat4.2.2⟩
      have hslot8_7 : rd64 s7.mem (sp - 8) = some e := by
        have hv := rd64_wr64 hm7s; rw [Nat.mod_eq_of_lt (by omega)] at hv; rw [hs7mem]; exact hv
      have e8 : X86Int.step im s7 = .ok { s7.set 2 (some e) with rip := s7.rip + 3 } :=
        step_mov (by rw [hs7rip]; exact cfg.jz.d8) hs7r0
      obtain ⟨s8, hs8⟩ : ∃ u : X86Int.St, u = { s7.set 2 (some e) with rip := s7.rip + 3 } := ⟨_, rfl⟩
      have hs8rip : s8.rip = aJz + 45 := by rw [hs8]; show s7.rip + 3 = aJz + 45; rw [hs7rip]
      have hs8r2 : s8.reg 2 = some e := by rw [hs8]; show (s7.set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
      have gs8 : ∀ q, q ≠ 2 → s8.reg q = s7.reg q := by
        intro q hq; rw [hs8]; show (s7.set 2 _).reg q = s7.reg q; rw [reg_set_other hq]
      have hlen8 : 16 ≤ s8.regs.length := by rw [hs8]; show 16 ≤ (s7.set 2 _).regs.length; rw [len_set]; exact hlen7
      have hadd4 : (aluF .add e (4 % W)).1 = e + 4 := aluI_add_val h4W (by omega)
      have e9 : X86Int.step im s8 = .ok { s8.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := s8.rip + 4 } :=
        step_aluI_add (by rw [hs8rip]; exact cfg.jz.d9) hs8r2
      obtain ⟨s9, hs9⟩ : ∃ u : X86Int.St, u = { s8.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := s8.rip + 4 } := ⟨_, rfl⟩
      have hs9rip : s9.rip = aJz + 49 := by rw [hs9]; show s8.rip + 4 = aJz + 49; rw [hs8rip]
      have hs9r2 : s9.reg 2 = some (e + 4) := by rw [hs9]; show (s8.set 2 _).reg 2 = some (e + 4); rw [reg_set_self (by omega), hadd4]
      have gs9 : ∀ q, q ≠ 2 → s9.reg q = s8.reg q := by
        intro q hq; rw [hs9]; show (s8.set 2 _).reg q = s8.reg q; rw [reg_set_other hq]
      have hlen9 : 16 ≤ s9.regs.length := by rw [hs9]; show 16 ≤ (s8.set 2 _).regs.length; rw [len_set]; exact hlen8
      have hs9r4 : s9.reg 4 = some (sp - 8) := by rw [gs9 4 (by decide), gs8 4 (by decide)]; exact hs7r4
      have hs9mem : s9.mem = m7 := by rw [hs9, hs8]; exact hs7mem
      have hws9 : wshape s9.mem = wshape s1.mem := by rw [hs9mem, ← hs7mem]; exact hws7
      have hroom9 : StackRoom s9 1 :=
        StackRoom_shift4 hroom4 hs4r4 hs9r4 (hws9.trans hws4.symm) ⟨1, by omega, by omega⟩ hsp32 hspW
      obtain ⟨k10, s10, hk10, hr10, hs10rip, hs10r0, hs10r4, gs10, hlen10, hfr10seg, hw10seg⟩ :=
        CallL32B cfg.l32 cfg.jz.dret cfg.jz.d10 (by omega) (by omega) (by omega) (by decide)
          (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
          s9 hs9rip hs9r2 hs9r4 (by rw [hs9mem, ← hs7mem]; exact hat7.2.1) hroom9
      have hs10r8 : s10.reg 8 = some hm.d := by
        rw [gs10 8 (by decide) (by decide), gs9 8 (by decide), gs8 8 (by decide), gs7 8 (by decide)]; exact hs6r8
      have hs10r9 : s10.reg 9 = some hm.a := by
        rw [gs10 9 (by decide) (by decide), gs9 9 (by decide), gs8 9 (by decide), gs7 9 (by decide)]; exact hs6r9
      have hws10 : wshape s10.mem = wshape s1.mem := hw10seg.trans hws9
      have hlen10s : 16 ≤ s10.regs.length := by rw [hlen10]; exact hlen9
      -- mov r10, r8 ; sub r10, 1 ; cmp rax, r10
      have e10a : X86Int.step im s10 = .ok { s10.set 10 (some hm.d) with rip := s10.rip + 3 } :=
        step_mov (by rw [hs10rip]; exact cfg.jz.d10a) hs10r8
      obtain ⟨s10a, hs10a⟩ : ∃ u : X86Int.St, u = { s10.set 10 (some hm.d) with rip := s10.rip + 3 } := ⟨_, rfl⟩
      have hs10arip : s10a.rip = aJz + 57 := by rw [hs10a]; show s10.rip + 3 = aJz + 57; rw [hs10rip]
      have hs10ar10 : s10a.reg 10 = some hm.d := by rw [hs10a]; show (s10.set 10 _).reg 10 = some hm.d; rw [reg_set_self (by omega)]
      have gs10a : ∀ q, q ≠ 10 → s10a.reg q = s10.reg q := by
        intro q hq; rw [hs10a]; show (s10.set 10 _).reg q = s10.reg q; rw [reg_set_other hq]
      have hlen10a : 16 ≤ s10a.regs.length := by rw [hs10a]; show 16 ≤ (s10.set 10 _).regs.length; rw [len_set]; exact hlen10s
      have hsub1 : (aluF .sub hm.d (1 % W)).1 = hm.d - 1 := by
        rw [aluF_sub_fst, show (1 : Nat) % W = 1 from Nat.mod_eq_of_lt (by decide),
          show hm.d + W - 1 = (hm.d - 1) + W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      have e10b : X86Int.step im s10a = .ok { s10a.set 10 (some (aluF .sub hm.d (1 % W)).1) with fl := (aluF .sub hm.d (1 % W)).2, rip := s10a.rip + 4 } :=
        step_aluI_sub (by rw [hs10arip]; exact cfg.jz.d10b) hs10ar10
      obtain ⟨s10b, hs10b⟩ : ∃ u : X86Int.St, u = { s10a.set 10 (some (aluF .sub hm.d (1 % W)).1) with fl := (aluF .sub hm.d (1 % W)).2, rip := s10a.rip + 4 } := ⟨_, rfl⟩
      have hs10brip : s10b.rip = aJz + 61 := by rw [hs10b]; show s10a.rip + 4 = aJz + 61; rw [hs10arip]
      have hs10br10 : s10b.reg 10 = some (hm.d - 1) := by rw [hs10b]; show (s10a.set 10 _).reg 10 = _; rw [reg_set_self (by omega), hsub1]
      have gs10b : ∀ q, q ≠ 10 → s10b.reg q = s10a.reg q := by
        intro q hq; rw [hs10b]; show (s10a.set 10 _).reg q = s10a.reg q; rw [reg_set_other hq]
      have hs10br0 : s10b.reg 0 = some a.d := by rw [gs10b 0 (by decide), gs10a 0 (by decide)]; exact hs10r0
      have hs10br4 : s10b.reg 4 = some (sp - 8) := by rw [gs10b 4 (by decide), gs10a 4 (by decide)]; exact hs10r4
      have hs10br9 : s10b.reg 9 = some hm.a := by rw [gs10b 9 (by decide), gs10a 9 (by decide)]; exact hs10r9
      have hs10bmem : s10b.mem = s10.mem := by rw [hs10b, hs10a]; exact rfl
      have hlen10b : 16 ≤ s10b.regs.length := by rw [hs10b]; show 16 ≤ (s10a.set 10 _).regs.length; rw [len_set]; exact hlen10a
      have e11 : X86Int.step im s10b = .ok { s10b with fl := (aluF .sub a.d (hm.d - 1)).2, rip := s10b.rip + 3 } :=
        step_cmp (by rw [hs10brip]; exact cfg.jz.d11) hs10br0 hs10br10
      obtain ⟨s11, hs11⟩ : ∃ u : X86Int.St, u = { s10b with fl := (aluF .sub a.d (hm.d - 1)).2, rip := s10b.rip + 3 } := ⟨_, rfl⟩
      have hs11rip : s11.rip = aJz + 64 := by rw [hs11]; show s10b.rip + 3 = aJz + 64; rw [hs10brip]
      have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
      have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
      have r7 : run im 1 (Out.ok s6) = Out.ok s7 := by rw [run_succ_ok im 0 s6, e7, ← hs7, run_zero]
      have r8 : run im 1 (Out.ok s7) = Out.ok s8 := by rw [run_succ_ok im 0 s7, e8, ← hs8, run_zero]
      have r9 : run im 1 (Out.ok s8) = Out.ok s9 := by rw [run_succ_ok im 0 s8, e9, ← hs9, run_zero]
      have r10a : run im 1 (Out.ok s10) = Out.ok s10a := by rw [run_succ_ok im 0 s10, e10a, ← hs10a, run_zero]
      have r10b : run im 1 (Out.ok s10a) = Out.ok s10b := by rw [run_succ_ok im 0 s10a, e10b, ← hs10b, run_zero]
      have r11 : run im 1 (Out.ok s10b) = Out.ok s11 := by rw [run_succ_ok im 0 s10b, e11, ← hs11, run_zero]
      have hrun9 : run im ((k2 + 3 + k4) + 5) (Out.ok s1) = Out.ok s9 := by
        rw [run_add im _ 5, hrun4, run_add im 1 4, r5, run_add im 1 3, r6, run_add im 1 2, r7, run_add im 1 1, r8, r9]
      have hrun10 : run im ((k2 + 3 + k4) + 5 + k10) (Out.ok s1) = Out.ok s10 := by
        rw [run_add im _ k10, hrun9, hr10]
      have hrun11 : run im ((k2 + 3 + k4) + 5 + k10 + 3) (Out.ok s1) = Out.ok s11 := by
        rw [run_add im _ 3, hrun10, run_add im 1 2, r10a, run_add im 1 1, r10b, r11]
      by_cases hdeq : a.d = hm.d - 1
      · have ha : a.a ≠ hm.a := fun ha => hnea (by cases a; cases hm; simp at hdeq ha; simp [hdeq, ha])
        have hs11fl : X86Int.cond s11.fl 4 = some true := by
          have hzf : s11.fl.zf = some true := by rw [hs11]; exact aluF_sub_zf_eq adW (by omega) hdeq
          exact cond_cc4 hzf
        have e12 : X86Int.step im s11 = .ok { s11 with rip := aJz + 71 } := by
          have hh := step_jcc (im := im) (s := s11) (by rw [hs11rip]; exact cfg.jz.d12) hs11fl
          simpa using hh
        obtain ⟨s12, hs12⟩ : ∃ u : X86Int.St, u = { s11 with rip := aJz + 71 } := ⟨_, rfl⟩
        have hs12rip : s12.rip = aJz + 71 := by rw [hs12]
        have hs12r4 : s12.reg 4 = some (sp - 8) := by rw [hs12, hs11]; exact hs10br4
        have hs12mem : s12.mem = s10.mem := by rw [hs12, hs11]; exact hs10bmem
        have hslot12 : rd64 s12.mem (sp - 8) = some e := by
          rw [hs12mem]; unfold rd64; rw [hfr10seg (sp - 8) 8 (by omega), hs9mem, ← hs7mem]; exact hslot8_7
        have hp8 : ((sp - 8) + 8) % W = sp := slot_add8 (by omega) st.spW
        have e13 : X86Int.step im s12 = .ok { (s12.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := s12.rip + 1 } :=
          step_pop (by rw [hs12rip]; exact cfg.jz.d13) hs12r4 hslot12
        obtain ⟨s13, hs13⟩ : ∃ u : X86Int.St, u = { (s12.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := s12.rip + 1 } := ⟨_, rfl⟩
        have hs13rip : s13.rip = aJz + 72 := by rw [hs13]; show s12.rip + 1 = aJz + 72; rw [hs12rip]
        have hs12len : 16 ≤ s12.regs.length := by rw [hs12, hs11]; exact hlen10b
        have hs13r4 : s13.reg 4 = some sp := by
          rw [hs13]; show ((s12.set 4 _).set 2 _).reg 4 = some sp
          rw [reg_set_other (by decide), reg_set_self (by omega), hp8]
        have hs13r2 : s13.reg 2 = some e := by
          rw [hs13]; show ((s12.set 4 _).set 2 _).reg 2 = some e
          rw [reg_set_self (by show 2 < (s12.set 4 _).regs.length; rw [len_set]; exact Nat.lt_of_lt_of_le (by decide) hs12len)]
        have gs13 : ∀ q, q ≠ 2 → q ≠ 4 → s13.reg q = s12.reg q := by
          intro q h2 h4; rw [hs13]; show ((s12.set 4 _).set 2 _).reg q = s12.reg q
          rw [reg_set_other h2, reg_set_other h4]
        have hlen13 : 16 ≤ s13.regs.length := by
          rw [hs13]; show 16 ≤ ((s12.set 4 _).set 2 _).regs.length; rw [len_set, len_set]; exact hs12len
        have hadd8 : (aluF .add e (8 % W)).1 = e + 8 := aluI_add_val h8W (by omega)
        have e14 : X86Int.step im s13 = .ok { s13.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := s13.rip + 4 } :=
          step_aluI_add (by rw [hs13rip]; exact cfg.jz.d14) hs13r2
        obtain ⟨s14, hs14⟩ : ∃ u : X86Int.St, u = { s13.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := s13.rip + 4 } := ⟨_, rfl⟩
        have hs14rip : s14.rip = aJz + 76 := by rw [hs14]; show s13.rip + 4 = aJz + 76; rw [hs13rip]
        have hs14r2 : s14.reg 2 = some (e + 8) := by rw [hs14]; show (s13.set 2 _).reg 2 = some (e + 8); rw [reg_set_self (by omega), hadd8]
        have gs14 : ∀ q, q ≠ 2 → s14.reg q = s13.reg q := by
          intro q hq; rw [hs14]; show (s13.set 2 _).reg q = s13.reg q; rw [reg_set_other hq]
        have hs14r4 : s14.reg 4 = some sp := by rw [gs14 4 (by decide)]; exact hs13r4
        have hs14r9 : s14.reg 9 = some hm.a := by rw [gs14 9 (by decide), gs13 9 (by decide) (by decide), hs12, hs11]; exact hs10br9
        have hs13mem : s13.mem = s12.mem := by rw [hs13]; rfl
        have hs14mem : s14.mem = s10.mem := by rw [hs14]; exact hs13mem.trans hs12mem
        have hws14 : wshape s14.mem = wshape s1.mem := by rw [hs14mem]; exact hws10
        have hat14a : (s14.mem.rdN (e + 8) 4).map leVal = some a.a := by
          rw [hs14mem, hfr10seg (e + 8) 4 (by omega), hs9mem, ← hs7mem]; exact hat7.2.2
        obtain ⟨k15, s15, hk15, hr15, hs15rip, hs15r0, hs15r4, gs15, _, _, hw15seg⟩ :=
          CallL32B cfg.l32 cfg.jz.dret cfg.jz.d15 (by omega) (by omega) hspW (by decide)
            (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
            s14 hs14rip hs14r2 hs14r4 hat14a
            (StackRoom_shift4 hroom4 hs4r4 (L := 0) (by simpa using hs14r4) (hws14.trans hws4.symm) ⟨0, rfl, by decide⟩ hsp32 hspW)
        have hs15r9 : s15.reg 9 = some hm.a := by rw [gs15 9 (by decide) (by decide)]; exact hs14r9
        have hws15 : wshape s15.mem = wshape s1.mem := hw15seg.trans hws14
        have e16 : X86Int.step im s15 = .ok { s15 with fl := (aluF .sub a.a hm.a).2, rip := s15.rip + 3 } :=
          step_cmp (by rw [hs15rip]; exact cfg.jz.d16) hs15r0 hs15r9
        obtain ⟨s16, hs16⟩ : ∃ u : X86Int.St, u = { s15 with fl := (aluF .sub a.a hm.a).2, rip := s15.rip + 3 } := ⟨_, rfl⟩
        have hs16rip : s16.rip = aJz + 84 := by rw [hs16]; show s15.rip + 3 = aJz + 84; rw [hs15rip]
        have hs16fl : X86Int.cond s16.fl 4 = some false := by
          have hzf : s16.fl.zf = some false := by rw [hs16]; exact aluF_sub_zf_ne aaW haW ha
          exact cond_cc4 hzf
        have e17 : X86Int.step im s16 = .ok { s16 with rip := aJz + 86 } := by
          have hh := step_jcc (im := im) (s := s16) (by rw [hs16rip]; exact cfg.jz.d17) hs16fl
          simpa [hs16rip] using hh
        obtain ⟨s17, hs17⟩ : ∃ u : X86Int.St, u = { s16 with rip := aJz + 86 } := ⟨_, rfl⟩
        have r12 : run im 1 (Out.ok s11) = Out.ok s12 := by rw [run_succ_ok im 0 s11, e12, ← hs12, run_zero]
        have r13 : run im 1 (Out.ok s12) = Out.ok s13 := by rw [run_succ_ok im 0 s12, e13, ← hs13, run_zero]
        have r14 : run im 1 (Out.ok s13) = Out.ok s14 := by rw [run_succ_ok im 0 s13, e14, ← hs14, run_zero]
        have r16 : run im 1 (Out.ok s15) = Out.ok s16 := by rw [run_succ_ok im 0 s15, e16, ← hs16, run_zero]
        have r17 : run im 1 (Out.ok s16) = Out.ok s17 := by rw [run_succ_ok im 0 s16, e17, ← hs17, run_zero]
        have hrun14 : run im (((k2 + 3 + k4) + 5 + k10 + 3) + 3) (Out.ok s1) = Out.ok s14 := by
          rw [run_add im _ 3, hrun11, run_add im 1 2, r12, run_add im 1 1, r13, r14]
        have hrun15 : run im (((k2 + 3 + k4) + 5 + k10 + 3) + 3 + k15) (Out.ok s1) = Out.ok s15 := by
          rw [run_add im _ k15, hrun14, hr15]
        have hrun17 : run im (((k2 + 3 + k4) + 5 + k10 + 3) + 3 + k15 + 2) (Out.ok s1) = Out.ok s17 := by
          rw [run_add im _ 2, hrun15, run_add im 1 1, r16, r17]
        have hs17rip : s17.rip = aJz + 86 := by rw [hs17]
        have hs17r4 : s17.reg 4 = some sp := by rw [hs17, hs16]; exact hs15r4
        have hws17 : wshape s17.mem = wshape s1.mem := by rw [hs17, hs16]; exact hws15
        obtain ⟨k18, hk18, hr18⟩ := jmp_fail1_exit1B rc cr.jzAnnA st.sp32 st.spW s17 hs17rip hs17r4
          (run_regs_ok hrun17 hregs1)
          (StackRoom_mono (StackRoom_of_wshape hws17 (by rw [hs17r4, hr41]) hroom1) (by decide))
        exact ⟨(((k2 + 3 + k4) + 5 + k10 + 3) + 3 + k15 + 2) + k18, (by omega),
          by rw [run_add im _ k18, hrun17]; exact hr18⟩
      · -- a.d ≠ hm.d - 1 -> cr.jzAnnD
        have hs11fl : X86Int.cond s11.fl 4 = some false := by
          have hzf : s11.fl.zf = some false := by rw [hs11]; exact aluF_sub_zf_ne adW (by omega) hdeq
          exact cond_cc4 hzf
        have e12 : X86Int.step im s11 = .ok { s11 with rip := aJz + 66 } := by
          have hh := step_jcc (im := im) (s := s11) (by rw [hs11rip]; exact cfg.jz.d12) hs11fl
          simpa [hs11rip] using hh
        obtain ⟨s12, hs12⟩ : ∃ u : X86Int.St, u = { s11 with rip := aJz + 66 } := ⟨_, rfl⟩
        have r12 : run im 1 (Out.ok s11) = Out.ok s12 := by rw [run_succ_ok im 0 s11, e12, ← hs12, run_zero]
        have hrun12 : run im (((k2 + 3 + k4) + 5 + k10 + 3) + 1) (Out.ok s1) = Out.ok s12 := by
          rw [run_add im _ 1, hrun11, r12]
        have hs12rip : s12.rip = aJz + 66 := by rw [hs12]
        have hs12r4 : s12.reg 4 = some (sp - 8) := by rw [hs12, hs11]; exact hs10br4
        have hws12 : wshape s12.mem = wshape s1.mem := by rw [hs12, hs11, hs10bmem]; exact hws10
        obtain ⟨k13, hk13, hr13⟩ := jmp_fail1_exit1B rc cr.jzAnnD (by omega) (by omega) s12 hs12rip hs12r4
          (run_regs_ok hrun12 hregs1)
          (StackRoom_shift4 hroom4 hs4r4 hs12r4 (hws12.trans hws4.symm) ⟨1, by omega, by omega⟩ hsp32 hspW)
        exact ⟨(((k2 + 3 + k4) + 5 + k10 + 3) + 1) + k13, (by omega),
          by rw [run_add im _ k13, hrun12]; exact hr13⟩

/-- `jz t` refuses when `hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩`. -/
theorem hand_jz_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jz t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jz t) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 100 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hfail : hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩ := by
    by_cases h1 : 1 ≤ hm.d ∧ wd.ann.lookup t = some ⟨hm.d - 1, hm.a⟩
    · simp only [hstep] at hs; rw [if_pos h1] at hs; cases hs
    · by_cases hd : hm.d = 0
      · exact Or.inl hd
      · exact Or.inr fun heq => h1 ⟨by omega, heq⟩
  have hbhead := hBnd_head inv.hb; rw [inv.hlink.symm] at hbhead
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x89 :: leBytes t 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 137 := (rdN_cons h5).1
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkJz_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, hk2, hr2⟩ := jz_handler_refuseB cfg rc cr st hsp48 hhW (by omega) (by omega)
    (Nat.lt_trans himmok (by decide)) (c_add_len_lt inv rfl) (disjC5_of_inv inv rfl).2 hfail
    s1 hrip1 hr81 hr91 hr61 hr41 hr71 hr11 hlen1 (run_regs_ok hr1 hregs) himm
    (by rw [hm1]; exact inv.fannp) (by rw [hm1]; exact inv.fnann)
    (AnnLayout_of_mem hm1 inv.lay) (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room)
  exact ⟨k1 + k2, (by omega), by rw [run_add im k1 k2, hr1]; exact hr2⟩

/-- `jnz t` refuses when `hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩`. -/
theorem hand_jnz_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jnz t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jnz t) = none) (hregs : RegsOk s) :
    ∃ m, m ≤ 100 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  have hfail : hm.d = 0 ∨ wd.ann.lookup t ≠ some ⟨hm.d - 1, hm.a⟩ := by
    by_cases h1 : 1 ≤ hm.d ∧ wd.ann.lookup t = some ⟨hm.d - 1, hm.a⟩
    · simp only [hstep] at hs; rw [if_pos h1] at hs; cases hs
    · by_cases hd : hm.d = 0
      · exact Or.inl hd
      · exact Or.inr fun heq => h1 ⟨by omega, heq⟩
  have hbhead := hBnd_head inv.hb; rw [inv.hlink.symm] at hbhead
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have h5 : s.mem.rdN c 5 = some (0x8a :: leBytes t 4) := inv.csq.1
  have hbyte : s.mem.rd c = some 138 := (rdN_cons h5).1
  obtain ⟨k1, s1, hk1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, -, hm1⟩ :=
    WalkJnz_tripleB cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]; obtain ⟨-, himm0⟩ := imm32_of_enc h5; rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hlen1 : 16 ≤ s1.regs.length := by rw [run_regs_length k1 hr1]; exact inv.len16
  obtain ⟨k2, hk2, hr2⟩ := jz_handler_refuseB cfg rc cr st hsp48 hhW (by omega) (by omega)
    (Nat.lt_trans himmok (by decide)) (c_add_len_lt inv rfl) (disjC5_of_inv inv rfl).2 hfail
    s1 hrip1 hr81 hr91 hr61 hr41 hr71 hr11 hlen1 (run_regs_ok hr1 hregs) himm
    (by rw [hm1]; exact inv.fannp) (by rw [hm1]; exact inv.fnann)
    (AnnLayout_of_mem hm1 inv.lay) (StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room)
  exact ⟨k1 + k2, (by omega), by rw [run_add im k1 k2, hr1]; exact hr2⟩

/-- `fall_tail` refuses when the falling-through instruction is the last instruction (`i + 1 = wd.code.length`). -/
theorem fall_end_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {i c' : Nat} {h' : Height} {s s2 : X86Int.St}
    (hi1_eq : i + 1 = wd.code.length)
    (hrip2 : s2.rip = aFallTail) (hpost : HPost s s2 h' c' sp)
    (hr1 : s.reg 1 = some i) (hr7 : s.reg 7 = some w) (hfnc : rd64 s.mem (w + 24) = some wd.code.length)
    (h16 : 16 ≤ s.regs.length) (hroom : StackRoom s 4) (hsp4 : s.reg 4 = some sp)
    (hregs2 : RegsOk s2) :
    ∃ m, m ≤ 20 ∧ run im m (Out.ok s2) = Out.exit 1 := by
  have hdone : ¬ i + 1 < wd.code.length := by omega
  obtain ⟨-, -, -, hs2r4, hs2r1, hs2r7, -, -, hfr2, hsh2, hlen2, -⟩ := hpost
  have hr1' : s2.reg 1 = some i := by rw [hs2r1, hr1]
  have hr7' : s2.reg 7 = some w := by rw [hs2r7, hr7]
  have h16' : 16 ≤ s2.regs.length := by rw [hlen2]; exact h16
  have hroom2 : StackRoom s2 4 := StackRoom_of_wshape hsh2 (by rw [hs2r4, hsp4]) hroom
  have hfnc2 : rd64 s2.mem (w + 24) = some wd.code.length :=
    rd64_of_frame_of_disjw st.disjw hfr2 (by decide) hfnc
  have hclenW := st.clenW
  have hiW : i < W := by omega
  have hadd1 : (aluF .add i (1 % W)).1 = i + 1 := aluI_add_val (by unfold W; decide) (by omega)
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by have := st.w56; omega)
  have e1 : X86Int.step im s2 = .ok { s2.set 1 (some (aluF .add i (1 % W)).1) with fl := (aluF .add i (1 % W)).2, rip := s2.rip + 4 } :=
    step_aluI_add (by rw [hrip2]; exact cfg.fall.d0) hr1'
  obtain ⟨f1, hf1⟩ : ∃ u : X86Int.St, u = { s2.set 1 (some (aluF .add i (1 % W)).1) with fl := (aluF .add i (1 % W)).2, rip := s2.rip + 4 } := ⟨_, rfl⟩
  have hf1rip : f1.rip = aFallTail + 4 := by rw [hf1]; show s2.rip + 4 = aFallTail + 4; rw [hrip2]
  have hf1r1 : f1.reg 1 = some (i + 1) := by rw [hf1]; show (s2.set 1 _).reg 1 = some (i + 1); rw [reg_set_self (by omega), hadd1]
  have hf1r7 : f1.reg 7 = some w := by rw [hf1]; show (s2.set 1 _).reg 7 = some w; rw [reg_set_other (by decide)]; exact hr7'
  have hf1r4 : f1.reg 4 = some sp := by rw [hf1]; show (s2.set 1 _).reg 4 = some sp; rw [reg_set_other (by decide)]; exact hs2r4
  have hf1mem : f1.mem = s2.mem := by rw [hf1]; exact rfl
  have hf1len : 16 ≤ f1.regs.length := by rw [hf1]; show 16 ≤ (s2.set 1 _).regs.length; rw [len_set]; exact h16'
  have e2 : X86Int.step im f1 = .ok { f1.set 0 (some wd.code.length) with rip := f1.rip + 4 } :=
    step_ld (by rw [hf1rip]; exact cfg.fall.d1) hf1r7 (by rw [hf1mem, hea24]; exact hfnc2)
  obtain ⟨f2, hf2⟩ : ∃ u : X86Int.St, u = { f1.set 0 (some wd.code.length) with rip := f1.rip + 4 } := ⟨_, rfl⟩
  have hf2rip : f2.rip = aFallTail + 8 := by rw [hf2]; show f1.rip + 4 = aFallTail + 8; rw [hf1rip]
  have hf2r0 : f2.reg 0 = some wd.code.length := by rw [hf2]; show (f1.set 0 _).reg 0 = some wd.code.length; rw [reg_set_self (by omega)]
  have hf2r1 : f2.reg 1 = some (i + 1) := by rw [hf2]; show (f1.set 0 _).reg 1 = some (i + 1); rw [reg_set_other (by decide)]; exact hf1r1
  have hf2r4 : f2.reg 4 = some sp := by rw [hf2]; show (f1.set 0 _).reg 4 = some sp; rw [reg_set_other (by decide)]; exact hf1r4
  have hf2mem : f2.mem = s2.mem := by rw [hf2]; exact hf1mem
  have e3 : X86Int.step im f2 = .ok { f2 with fl := (aluF .sub (i + 1) wd.code.length).2, rip := f2.rip + 3 } :=
    step_cmp (by rw [hf2rip]; exact cfg.fall.d2) hf2r1 hf2r0
  obtain ⟨f3, hf3⟩ : ∃ u : X86Int.St, u = { f2 with fl := (aluF .sub (i + 1) wd.code.length).2, rip := f2.rip + 3 } := ⟨_, rfl⟩
  have hf3rip : f3.rip = aFallTail + 11 := by rw [hf3]; show f2.rip + 3 = aFallTail + 11; rw [hf2rip]
  have hf3fl : X86Int.cond f3.fl 2 = some false := by
    have hcf : f3.fl.cf = some false := by rw [hf3]; exact aluF_sub_cf hdone
    exact cond_cc2 hcf
  have e4 : X86Int.step im f3 = .ok { f3 with rip := aFallTail + 13 } := by
    have hh := step_jcc (im := im) (s := f3) (by rw [hf3rip]; exact cfg.fall.d3) hf3fl
    simpa [hf3rip] using hh
  obtain ⟨f4, hf4⟩ : ∃ u : X86Int.St, u = { f3 with rip := aFallTail + 13 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s2) = Out.ok f1 := by rw [run_succ_ok im 0 s2, e1, ← hf1, run_zero]
  have r2 : run im 1 (Out.ok f1) = Out.ok f2 := by rw [run_succ_ok im 0 f1, e2, ← hf2, run_zero]
  have r3 : run im 1 (Out.ok f2) = Out.ok f3 := by rw [run_succ_ok im 0 f2, e3, ← hf3, run_zero]
  have r4 : run im 1 (Out.ok f3) = Out.ok f4 := by rw [run_succ_ok im 0 f3, e4, ← hf4, run_zero]
  have hrun4 : run im 4 (Out.ok s2) = Out.ok f4 := by
    rw [show (4 : Nat) = 1 + 3 by rfl, run_add im 1 3, r1, run_add im 1 2, r2, run_add im 1 1, r3, r4]
  have hf4rip : f4.rip = aFallTail + 13 := by rw [hf4]
  have hf4r4 : f4.reg 4 = some sp := by rw [hf4, hf3]; exact hf2r4
  have hf4mem : f4.mem = s2.mem := by rw [hf4, hf3]; exact hf2mem
  obtain ⟨k5, hk5, hr5⟩ := jmp_fail1_exit1B rc cr.fallEnd st.sp32 st.spW f4 hf4rip hf4r4
    (run_regs_ok hrun4 hregs2) (StackRoom_mono (StackRoom_of_mem hf4mem (by rw [hf4r4, hs2r4]) hroom2) (by decide))
  exact ⟨4 + k5, (by omega), by rw [run_add im 4 k5, hrun4]; exact hr5⟩

set_option maxHeartbeats 800000 in
/-- `fall_tail` refuses when `i + 1 < wd.code.length` and `wd.ann.lookup (i + 1) = some a` with `a ≠ h'`. -/
theorem fall_ann_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    {i c' : Nat} {h' a : Height} {s s2 : X86Int.St}
    (hi1 : i + 1 < wd.code.length) (hlk : wd.ann.lookup (i + 1) = some a) (hne : a ≠ h')
    (hd'W : h'.d < W) (ha'W : h'.a < W)
    (hrip2 : s2.rip = aFallTail) (hpost : HPost s s2 h' c' sp)
    (hr1 : s.reg 1 = some i) (hr7 : s.reg 7 = some w)
    (hf24 : rd64 s.mem (w + 24) = some wd.code.length)
    (hf32 : rd64 s.mem (w + 32) = some wd.ann.length)
    (hf48 : rd64 s.mem (w + 48) = some p)
    (hlay : AnnLayout s wd.ann p) (h16 : 16 ≤ s.regs.length) (hroom : StackRoom s 4)
    (hsp4 : s.reg 4 = some sp) (hregs2 : RegsOk s2) :
    ∃ m, m ≤ 50 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s2) = Out.exit 1 := by
  obtain ⟨adW, aaW⟩ := hhW (i + 1, a) (lookup_mem hlk)
  obtain ⟨hs2r8, hs2r9, -, hs2r4, hs2r1, hs2r7, -, -, hfr2, hsh2, hlen2, -⟩ := hpost
  have hr1' : s2.reg 1 = some i := by rw [hs2r1, hr1]
  have hr7' : s2.reg 7 = some w := by rw [hs2r7, hr7]
  have h16' : 16 ≤ s2.regs.length := by rw [hlen2]; exact h16
  have hroom2 : StackRoom s2 4 := StackRoom_of_wshape hsh2 (by rw [hs2r4, hsp4]) hroom
  have hlay2 : AnnLayout s2 wd.ann p := AnnLayout_of_frame hlay (by
    intro x k hx1 hx2; exact hfr2 x k (by cases st.disjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega)))
  have hf24_2 := rd64_of_frame_of_disjw st.disjw hfr2 (by decide) hf24
  have hf32_2 := rd64_of_frame_of_disjw st.disjw hfr2 (by decide) hf32
  have hf48_2 := rd64_of_frame_of_disjw st.disjw hfr2 (by decide) hf48
  have hadd1 : (aluF .add i (1 % W)).1 = i + 1 := aluI_add_val (by unfold W; decide) (by have := st.clenW; omega)
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by have := st.w56; omega)
  -- add 1; ld; cmp; jb +18; mov 12, 1
  have e1 : X86Int.step im s2 = .ok { s2.set 1 (some (aluF .add i (1 % W)).1) with fl := (aluF .add i (1 % W)).2, rip := s2.rip + 4 } :=
    step_aluI_add (by rw [hrip2]; exact cfg.fall.d0) hr1'
  obtain ⟨f1, hf1⟩ : ∃ u : X86Int.St, u = { s2.set 1 (some (aluF .add i (1 % W)).1) with fl := (aluF .add i (1 % W)).2, rip := s2.rip + 4 } := ⟨_, rfl⟩
  have hf1rip : f1.rip = aFallTail + 4 := by rw [hf1]; show s2.rip + 4 = aFallTail + 4; rw [hrip2]
  have hf1r1 : f1.reg 1 = some (i + 1) := by rw [hf1]; show (s2.set 1 _).reg 1 = some (i + 1); rw [reg_set_self (by omega), hadd1]
  have gs1 : ∀ q, q ≠ 1 → f1.reg q = s2.reg q := by
    intro q hq; rw [hf1]; show (s2.set 1 _).reg q = s2.reg q; rw [reg_set_other hq]
  have hf1mem : f1.mem = s2.mem := by rw [hf1]; rfl
  have hf1len : 16 ≤ f1.regs.length := by rw [hf1]; show 16 ≤ (s2.set 1 _).regs.length; rw [len_set]; exact h16'
  have e2 : X86Int.step im f1 = .ok { f1.set 0 (some wd.code.length) with rip := f1.rip + 4 } :=
    step_ld (by rw [hf1rip]; exact cfg.fall.d1) (by rw [gs1 7 (by decide)]; exact hr7')
      (by rw [hf1, mem_set, hea24]; exact hf24_2)
  obtain ⟨f2, hf2⟩ : ∃ u : X86Int.St, u = { f1.set 0 (some wd.code.length) with rip := f1.rip + 4 } := ⟨_, rfl⟩
  have hf2rip : f2.rip = aFallTail + 8 := by rw [hf2]; show f1.rip + 4 = aFallTail + 8; rw [hf1rip]
  have hf2r0 : f2.reg 0 = some wd.code.length := by rw [hf2]; show (f1.set 0 _).reg 0 = some wd.code.length; rw [reg_set_self (by omega)]
  have gs2 : ∀ q, q ≠ 0 → f2.reg q = f1.reg q := by
    intro q hq; rw [hf2]; show (f1.set 0 _).reg q = f1.reg q; rw [reg_set_other hq]
  have hf2r1 : f2.reg 1 = some (i + 1) := by rw [gs2 1 (by decide)]; exact hf1r1
  have e3 : X86Int.step im f2 = .ok { f2 with fl := (aluF .sub (i + 1) wd.code.length).2, rip := f2.rip + 3 } :=
    step_cmp (by rw [hf2rip]; exact cfg.fall.d2) hf2r1 hf2r0
  obtain ⟨f3, hf3⟩ : ∃ u : X86Int.St, u = { f2 with fl := (aluF .sub (i + 1) wd.code.length).2, rip := f2.rip + 3 } := ⟨_, rfl⟩
  have hf3rip : f3.rip = aFallTail + 11 := by rw [hf3]; show f2.rip + 3 = aFallTail + 11; rw [hf2rip]
  have hf3fl : X86Int.cond f3.fl 2 = some true := by
    have hcf : f3.fl.cf = some true := by rw [hf3]; exact aluF_sub_cf_true hi1
    exact cond_cc2 hcf
  have e4 : X86Int.step im f3 = .ok { f3 with rip := aFallTail + 18 } := by
    have hh := step_jcc (im := im) (s := f3) (by rw [hf3rip]; exact cfg.fall.d3) hf3fl
    simpa using hh
  obtain ⟨f4, hf4⟩ : ∃ u : X86Int.St, u = { f3 with rip := aFallTail + 18 } := ⟨_, rfl⟩
  have hf4rip : f4.rip = aFallTail + 18 := by rw [hf4]
  have hf4r1 : f4.reg 1 = some (i + 1) := by rw [hf4, hf3]; exact hf2r1
  have hf4len : 16 ≤ f4.regs.length := by rw [hf4, hf3, hf2]; show 16 ≤ (f1.set 0 _).regs.length; rw [len_set]; exact hf1len
  have e5 : X86Int.step im f4 = .ok { f4.set 12 (some (i + 1)) with rip := f4.rip + 3 } :=
    step_mov (by rw [hf4rip]; exact cfg.fall.d4) hf4r1
  obtain ⟨f5, hf5⟩ : ∃ u : X86Int.St, u = { f4.set 12 (some (i + 1)) with rip := f4.rip + 3 } := ⟨_, rfl⟩
  have hf5rip : f5.rip = aFallTail + 21 := by rw [hf5]; show f4.rip + 3 = aFallTail + 21; rw [hf4rip]
  have hf5r12 : f5.reg 12 = some (i + 1) := by rw [hf5]; show (f4.set 12 _).reg 12 = _; rw [reg_set_self (by omega)]
  have gs5 : ∀ q, q ≠ 12 → f5.reg q = f4.reg q := by
    intro q hq; rw [hf5]; show (f4.set 12 _).reg q = f4.reg q; rw [reg_set_other hq]
  have gs4 : ∀ q, q ≠ 0 → q ≠ 1 → f4.reg q = s2.reg q := by
    intro q h0 h1
    have h42 : f4.reg q = f2.reg q := by rw [hf4, hf3]; rfl
    rw [h42, gs2 q h0, gs1 q h1]
  have hf5r7 : f5.reg 7 = some w := by rw [gs5 7 (by decide), gs4 7 (by decide) (by decide)]; exact hr7'
  have hf5r4 : f5.reg 4 = some sp := by rw [gs5 4 (by decide), gs4 4 (by decide) (by decide)]; exact hs2r4
  have hf5r1 : f5.reg 1 = some (i + 1) := by rw [gs5 1 (by decide)]; exact hf4r1
  have hf5r8 : f5.reg 8 = some h'.d := by rw [gs5 8 (by decide), gs4 8 (by decide) (by decide)]; exact hs2r8
  have hf5r9 : f5.reg 9 = some h'.a := by rw [gs5 9 (by decide), gs4 9 (by decide) (by decide)]; exact hs2r9
  have hf5mem : f5.mem = s2.mem := by rw [hf5, hf4, hf3, hf2]; exact hf1mem
  -- call ANNL
  have hi1W : i + 1 < W := by have := st.clenW; omega
  obtain ⟨k6, f6, hk6, hr6, hf6rip, hpost6, _, hf6r4, _, _, _, gs6, _, hex0, hloc6, hw6seg, hlen6⟩ :=
    CallANNLB cfg.annl cfg.l32 (by decide) cfg.fall.d5 (by decide) hi1W st.p0 st.w56
      st.pann st.sp32 st.spW f5 hf5rip hf5r12 hf5r7 hf5r4 hf5r1
      (by rw [hf5mem]; exact hf48_2) (by rw [hf5mem]; exact hf32_2)
      (AnnLayout_of_mem hf5mem hlay2) (StackRoom_of_mem hf5mem (by rw [hf5r4, hs2r4]) hroom2)
      st.disjw st.disjp
  obtain ⟨e, hf6r0⟩ := hex0
  obtain ⟨he0, hat6⟩ : e ≠ 0 ∧ AnnAt f6 e (i + 1) a.d a.a := by
    cases hpost6 e hf6r0 with
    | inl hb => rw [hb.2] at hlk; simp at hlk
    | inr hb =>
      obtain ⟨d1, a1, hne0, hat, hlk1⟩ := hb
      have hhh : (⟨d1, a1⟩ : Height) = a := Option.some.inj (hlk1.symm.trans hlk)
      exact ⟨hne0, by rwa [show d1 = a.d from congrArg Height.d hhh, show a1 = a.a from congrArg Height.a hhh] at hat⟩
  obtain ⟨helo, hehi⟩ := hloc6 e hf6r0 he0
  have r1 : run im 1 (Out.ok s2) = Out.ok f1 := by rw [run_succ_ok im 0 s2, e1, ← hf1, run_zero]
  have r2 : run im 1 (Out.ok f1) = Out.ok f2 := by rw [run_succ_ok im 0 f1, e2, ← hf2, run_zero]
  have r3 : run im 1 (Out.ok f2) = Out.ok f3 := by rw [run_succ_ok im 0 f2, e3, ← hf3, run_zero]
  have r4 : run im 1 (Out.ok f3) = Out.ok f4 := by rw [run_succ_ok im 0 f3, e4, ← hf4, run_zero]
  have r5 : run im 1 (Out.ok f4) = Out.ok f5 := by rw [run_succ_ok im 0 f4, e5, ← hf5, run_zero]
  have hrun6 : run im (5 + k6) (Out.ok s2) = Out.ok f6 := by
    rw [run_add im 5 k6, run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3,
      run_add im 1 1, r4, r5, hr6]
  have hf6r8 : f6.reg 8 = some h'.d := by rw [gs6 8 (by decide) (by decide) (by decide) (by decide)]; exact hf5r8
  have hf6r9 : f6.reg 9 = some h'.a := by rw [gs6 9 (by decide) (by decide) (by decide) (by decide)]; exact hf5r9
  have hwf6 : wshape f6.mem = wshape s2.mem := by rw [hw6seg, hf5mem]
  have hroom6 : StackRoom f6 4 := StackRoom_of_wshape hwf6 (by rw [hf6r4, hs2r4]) hroom2
  have hf5len : 16 ≤ f5.regs.length := by rw [hf5]; show 16 ≤ (f4.set 12 _).regs.length; rw [len_set]; exact hf4len
  have hlen6s : 16 ≤ f6.regs.length := by rw [hlen6]; exact hf5len
  have h4W : 4 < W := by decide
  have h8W : 8 < W := by decide
  have hpann := st.pann
  have hdisjp := st.disjp
  have hspW := st.spW
  have hsp32 := st.sp32
  have hf6rip' : f6.rip = aFallTail + 26 := by rw [hf6rip]
  -- test ; jnz taken -> push -> CallL32B a.d
  have e7 : X86Int.step im f6 = .ok { f6 with fl := (aluF .and e e).2, rip := f6.rip + 3 } :=
    step_test (by rw [hf6rip']; exact cfg.fall.d6) hf6r0 hf6r0
  obtain ⟨f7, hf7⟩ : ∃ u : X86Int.St, u = { f6 with fl := (aluF .and e e).2, rip := f6.rip + 3 } := ⟨_, rfl⟩
  have hf7rip : f7.rip = aFallTail + 29 := by rw [hf7]; show f6.rip + 3 = aFallTail + 29; rw [hf6rip']
  have hf7fl : X86Int.cond f7.fl 5 = some true := by
    have hz : f7.fl.zf = some false := by
      rw [hf7]; show (aluF .and e e).2.zf = some false; rw [aluF_zf, aluF_and_fst_self]; simp [he0]
    rw [cond_cc5 hz]; rfl
  have e8 : X86Int.step im f7 = .ok { f7 with rip := aFallTail + 36 } := by
    have hh := step_jcc (im := im) (s := f7) (by rw [hf7rip]; exact cfg.fall.d7) hf7fl
    simpa using hh
  obtain ⟨f8, hf8⟩ : ∃ u : X86Int.St, u = { f7 with rip := aFallTail + 36 } := ⟨_, rfl⟩
  have hf8rip : f8.rip = aFallTail + 36 := by rw [hf8]
  have hf8r0 : f8.reg 0 = some e := by rw [hf8, hf7]; exact hf6r0
  have hf8r4 : f8.reg 4 = some sp := by rw [hf8, hf7]; exact hf6r4
  have hf8r8 : f8.reg 8 = some h'.d := by rw [hf8, hf7]; exact hf6r8
  have hf8r9 : f8.reg 9 = some h'.a := by rw [hf8, hf7]; exact hf6r9
  have hf8mem : f8.mem = f6.mem := by rw [hf8, hf7]
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) st.spW]
  obtain ⟨m9, hm9s⟩ := Option.isSome_iff_exists.mp
    (by have hh := hroom6 sp hf6r4 0 (by decide) e; rw [hslot8, ← hf8mem] at hh; exact hh)
  have hm9p : wr64 f8.mem ((sp + W - 8) % W) e = some m9 := by rw [hslot8]; exact hm9s
  have hf8len : 16 ≤ f8.regs.length := by rw [hf8, hf7]; exact hlen6s
  have e9 : X86Int.step im f8 = .ok { f8.set 4 (some ((sp + W - 8) % W)) with mem := m9, rip := f8.rip + 1 } :=
    step_push (by rw [hf8rip]; exact cfg.fall.d9) hf8r0 hf8r4 hm9p
  obtain ⟨f9, hf9⟩ : ∃ u : X86Int.St, u = { f8.set 4 (some ((sp + W - 8) % W)) with mem := m9, rip := f8.rip + 1 } := ⟨_, rfl⟩
  have hf9rip : f9.rip = aFallTail + 37 := by rw [hf9]; show f8.rip + 1 = aFallTail + 37; rw [hf8rip]
  have hf9r4 : f9.reg 4 = some (sp - 8) := by
    rw [hf9]; show (f8.set 4 _).reg 4 = some (sp - 8)
    rw [reg_set_self (Nat.lt_of_lt_of_le (by decide) hf8len), hslot8]
  have gs9 : ∀ q, q ≠ 4 → f9.reg q = f8.reg q := by
    intro q hq; rw [hf9]; show (f8.set 4 _).reg q = f8.reg q; rw [reg_set_other hq]
  have hf9r0 : f9.reg 0 = some e := by rw [gs9 0 (by decide)]; exact hf8r0
  have hf9mem : f9.mem = m9 := by rw [hf9]
  have hlen9 : 16 ≤ f9.regs.length := by rw [hf9]; show 16 ≤ (f8.set 4 _).regs.length; rw [len_set]; exact hf8len
  have hwf9 : wshape f9.mem = wshape s2.mem := by rw [hf9mem]; exact (wrN_wshape _ _ _ hm9s).trans (by rw [hf8mem]; exact hwf6)
  have hat9 : AnnAt f9 e (i + 1) a.d a.a :=
    ⟨by rw [hf9mem, rdN_wr64_other hm9s (by omega), hf8mem]; exact hat6.1,
     by rw [hf9mem, rdN_wr64_other hm9s (by omega), hf8mem]; exact hat6.2.1,
     by rw [hf9mem, rdN_wr64_other hm9s (by omega), hf8mem]; exact hat6.2.2⟩
  have hslot8_9 : rd64 f9.mem (sp - 8) = some e := by
    have hv := rd64_wr64 hm9s; rw [Nat.mod_eq_of_lt (by omega)] at hv; rw [hf9mem]; exact hv
  have e10 : X86Int.step im f9 = .ok { f9.set 2 (some e) with rip := f9.rip + 3 } :=
    step_mov (by rw [hf9rip]; exact cfg.fall.d10) hf9r0
  obtain ⟨f10, hf10⟩ : ∃ u : X86Int.St, u = { f9.set 2 (some e) with rip := f9.rip + 3 } := ⟨_, rfl⟩
  have hf10rip : f10.rip = aFallTail + 40 := by rw [hf10]; show f9.rip + 3 = aFallTail + 40; rw [hf9rip]
  have hf10r2 : f10.reg 2 = some e := by rw [hf10]; show (f9.set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
  have gs10 : ∀ q, q ≠ 2 → f10.reg q = f9.reg q := by
    intro q hq; rw [hf10]; show (f9.set 2 _).reg q = f9.reg q; rw [reg_set_other hq]
  have hlen10 : 16 ≤ f10.regs.length := by rw [hf10]; show 16 ≤ (f9.set 2 _).regs.length; rw [len_set]; exact hlen9
  have hadd4 : (aluF .add e (4 % W)).1 = e + 4 := aluI_add_val h4W (by omega)
  have e11 : X86Int.step im f10 = .ok { f10.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := f10.rip + 4 } :=
    step_aluI_add (by rw [hf10rip]; exact cfg.fall.d11) hf10r2
  obtain ⟨f11, hf11⟩ : ∃ u : X86Int.St, u = { f10.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := f10.rip + 4 } := ⟨_, rfl⟩
  have hf11rip : f11.rip = aFallTail + 44 := by rw [hf11]; show f10.rip + 4 = aFallTail + 44; rw [hf10rip]
  have hf11r2 : f11.reg 2 = some (e + 4) := by rw [hf11]; show (f10.set 2 _).reg 2 = some (e + 4); rw [reg_set_self (by omega), hadd4]
  have gs11 : ∀ q, q ≠ 2 → f11.reg q = f10.reg q := by
    intro q hq; rw [hf11]; show (f10.set 2 _).reg q = f10.reg q; rw [reg_set_other hq]
  have hf11len : 16 ≤ f11.regs.length := by rw [hf11]; show 16 ≤ (f10.set 2 _).regs.length; rw [len_set]; exact hlen10
  have hf11r4 : f11.reg 4 = some (sp - 8) := by rw [gs11 4 (by decide), gs10 4 (by decide)]; exact hf9r4
  have hf11mem : f11.mem = m9 := by rw [hf11, hf10]; exact hf9mem
  have hwf11 : wshape f11.mem = wshape s2.mem := by rw [hf11mem, ← hf9mem]; exact hwf9
  have hroom11 : StackRoom f11 1 :=
    StackRoom_shift4 hroom6 hf6r4 hf11r4 (hwf11.trans hwf6.symm) ⟨1, by omega, by omega⟩ hsp32 hspW
  obtain ⟨k12, f12, hk12, hr12, hf12rip, hf12r0, hf12r4, gs12, hlen12, hfr12seg, hw12seg⟩ :=
    CallL32B cfg.l32 cfg.fall.dret cfg.fall.d12 (by omega) (by omega) (by omega) (by decide)
      (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
      f11 hf11rip hf11r2 hf11r4 (by rw [hf11mem, ← hf9mem]; exact hat9.2.1) hroom11
  have hf12r8 : f12.reg 8 = some h'.d := by
    rw [gs12 8 (by decide) (by decide), gs11 8 (by decide), gs10 8 (by decide), gs9 8 (by decide)]; exact hf8r8
  have hf12r9 : f12.reg 9 = some h'.a := by
    rw [gs12 9 (by decide) (by decide), gs11 9 (by decide), gs10 9 (by decide), gs9 9 (by decide)]; exact hf8r9
  have hwf12 : wshape f12.mem = wshape s2.mem := hw12seg.trans hwf11
  have e13 : X86Int.step im f12 = .ok { f12 with fl := (aluF .sub a.d h'.d).2, rip := f12.rip + 3 } :=
    step_cmp (by rw [hf12rip]; exact cfg.fall.d13) hf12r0 hf12r8
  obtain ⟨f13, hf13⟩ : ∃ u : X86Int.St, u = { f12 with fl := (aluF .sub a.d h'.d).2, rip := f12.rip + 3 } := ⟨_, rfl⟩
  have hf13rip : f13.rip = aFallTail + 52 := by rw [hf13]; show f12.rip + 3 = aFallTail + 52; rw [hf12rip]
  have r7 : run im 1 (Out.ok f6) = Out.ok f7 := by rw [run_succ_ok im 0 f6, e7, ← hf7, run_zero]
  have r8 : run im 1 (Out.ok f7) = Out.ok f8 := by rw [run_succ_ok im 0 f7, e8, ← hf8, run_zero]
  have r9 : run im 1 (Out.ok f8) = Out.ok f9 := by rw [run_succ_ok im 0 f8, e9, ← hf9, run_zero]
  have r10 : run im 1 (Out.ok f9) = Out.ok f10 := by rw [run_succ_ok im 0 f9, e10, ← hf10, run_zero]
  have r11 : run im 1 (Out.ok f10) = Out.ok f11 := by rw [run_succ_ok im 0 f10, e11, ← hf11, run_zero]
  have r13 : run im 1 (Out.ok f12) = Out.ok f13 := by rw [run_succ_ok im 0 f12, e13, ← hf13, run_zero]
  have hrun11 : run im ((5 + k6) + 5) (Out.ok s2) = Out.ok f11 := by
    rw [run_add im _ 5, hrun6, run_add im 1 4, r7, run_add im 1 3, r8, run_add im 1 2, r9, run_add im 1 1, r10, r11]
  have hrun12 : run im ((5 + k6) + 5 + k12) (Out.ok s2) = Out.ok f12 := by
    rw [run_add im _ k12, hrun11, hr12]
  have hrun13 : run im ((5 + k6) + 5 + k12 + 1) (Out.ok s2) = Out.ok f13 := by
    rw [run_add im _ 1, hrun12, r13]
  by_cases hdeq : a.d = h'.d
  · have ha : a.a ≠ h'.a := fun ha => hne (by cases a; cases h'; simp at hdeq ha; simp [hdeq, ha])
    have hf13fl : X86Int.cond f13.fl 4 = some true := by
      have hzf : f13.fl.zf = some true := by rw [hf13]; exact aluF_sub_zf_eq adW hd'W hdeq
      exact cond_cc4 hzf
    have e14 : X86Int.step im f13 = .ok { f13 with rip := aFallTail + 59 } := by
      have hh := step_jcc (im := im) (s := f13) (by rw [hf13rip]; exact cfg.fall.d14) hf13fl
      simpa using hh
    obtain ⟨f14, hf14⟩ : ∃ u : X86Int.St, u = { f13 with rip := aFallTail + 59 } := ⟨_, rfl⟩
    have hf14rip : f14.rip = aFallTail + 59 := by rw [hf14]
    have hf14r4 : f14.reg 4 = some (sp - 8) := by rw [hf14, hf13]; exact hf12r4
    have hf14mem : f14.mem = f12.mem := by rw [hf14, hf13]
    have hslot14 : rd64 f14.mem (sp - 8) = some e := by
      rw [hf14mem]; unfold rd64; rw [hfr12seg (sp - 8) 8 (by omega), hf11mem, ← hf9mem]; exact hslot8_9
    have hp8 : ((sp - 8) + 8) % W = sp := slot_add8 (by omega) st.spW
    have e15 : X86Int.step im f14 = .ok { (f14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := f14.rip + 1 } :=
      step_pop (by rw [hf14rip]; exact cfg.fall.d15) hf14r4 hslot14
    obtain ⟨f15, hf15⟩ : ∃ u : X86Int.St, u = { (f14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := f14.rip + 1 } := ⟨_, rfl⟩
    have hf15rip : f15.rip = aFallTail + 60 := by rw [hf15]; show f14.rip + 1 = aFallTail + 60; rw [hf14rip]
    have hf14len : 16 ≤ f14.regs.length := by rw [hf14, hf13, hlen12]; exact hf11len
    have hf15r4 : f15.reg 4 = some sp := by
      rw [hf15]; show ((f14.set 4 _).set 2 _).reg 4 = some sp
      rw [reg_set_other (by decide), reg_set_self (by omega), hp8]
    have hf15r2 : f15.reg 2 = some e := by
      rw [hf15]; show ((f14.set 4 _).set 2 _).reg 2 = some e
      rw [reg_set_self (by show 2 < (f14.set 4 _).regs.length; rw [len_set]; exact Nat.lt_of_lt_of_le (by decide) hf14len)]
    have gs15 : ∀ q, q ≠ 2 → q ≠ 4 → f15.reg q = f14.reg q := by
      intro q h2 h4; rw [hf15]; show ((f14.set 4 _).set 2 _).reg q = f14.reg q
      rw [reg_set_other h2, reg_set_other h4]
    have hlen15 : 16 ≤ f15.regs.length := by
      rw [hf15]; show 16 ≤ ((f14.set 4 _).set 2 _).regs.length; rw [len_set, len_set]; exact hf14len
    have hadd8 : (aluF .add e (8 % W)).1 = e + 8 := aluI_add_val h8W (by omega)
    have e16 : X86Int.step im f15 = .ok { f15.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := f15.rip + 4 } :=
      step_aluI_add (by rw [hf15rip]; exact cfg.fall.d16) hf15r2
    obtain ⟨f16, hf16⟩ : ∃ u : X86Int.St, u = { f15.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := f15.rip + 4 } := ⟨_, rfl⟩
    have hf16rip : f16.rip = aFallTail + 64 := by rw [hf16]; show f15.rip + 4 = aFallTail + 64; rw [hf15rip]
    have hf16r2 : f16.reg 2 = some (e + 8) := by rw [hf16]; show (f15.set 2 _).reg 2 = some (e + 8); rw [reg_set_self (by omega), hadd8]
    have gs16 : ∀ q, q ≠ 2 → f16.reg q = f15.reg q := by
      intro q hq; rw [hf16]; show (f15.set 2 _).reg q = f15.reg q; rw [reg_set_other hq]
    have hf16r4 : f16.reg 4 = some sp := by rw [gs16 4 (by decide)]; exact hf15r4
    have hf16r9 : f16.reg 9 = some h'.a := by rw [gs16 9 (by decide), gs15 9 (by decide) (by decide), hf14, hf13]; exact hf12r9
    have hf15mem : f15.mem = f14.mem := by rw [hf15]; rfl
    have hf16mem : f16.mem = f12.mem := by rw [hf16]; exact hf15mem.trans hf14mem
    have hwf16 : wshape f16.mem = wshape s2.mem := by rw [hf16mem]; exact hwf12
    have hat16a : (f16.mem.rdN (e + 8) 4).map leVal = some a.a := by
      rw [hf16mem, hfr12seg (e + 8) 4 (by omega), hf11mem, ← hf9mem]; exact hat9.2.2
    obtain ⟨k17, f17, hk17, hr17, hf17rip, hf17r0, hf17r4, gs17, _, _, hw17seg⟩ :=
      CallL32B cfg.l32 cfg.fall.dret cfg.fall.d17 (by omega) (by omega) hspW (by decide)
        (by cases hdisjp with | inl h => exact Or.inr (by omega) | inr h => exact Or.inl (by omega))
        f16 hf16rip hf16r2 hf16r4 hat16a
        (StackRoom_shift4 hroom6 hf6r4 (L := 0) (by simpa using hf16r4) (hwf16.trans hwf6.symm) ⟨0, rfl, by decide⟩ hsp32 hspW)
    have hf17r9 : f17.reg 9 = some h'.a := by rw [gs17 9 (by decide) (by decide)]; exact hf16r9
    have hwf17 : wshape f17.mem = wshape s2.mem := hw17seg.trans hwf16
    have e18 : X86Int.step im f17 = .ok { f17 with fl := (aluF .sub a.a h'.a).2, rip := f17.rip + 3 } :=
      step_cmp (by rw [hf17rip]; exact cfg.fall.d18) hf17r0 hf17r9
    obtain ⟨f18, hf18⟩ : ∃ u : X86Int.St, u = { f17 with fl := (aluF .sub a.a h'.a).2, rip := f17.rip + 3 } := ⟨_, rfl⟩
    have hf18rip : f18.rip = aFallTail + 72 := by rw [hf18]; show f17.rip + 3 = aFallTail + 72; rw [hf17rip]
    have hf18fl : X86Int.cond f18.fl 4 = some false := by
      have hzf : f18.fl.zf = some false := by rw [hf18]; exact aluF_sub_zf_ne aaW ha'W ha
      exact cond_cc4 hzf
    have e19 : X86Int.step im f18 = .ok { f18 with rip := aFallTail + 74 } := by
      have hh := step_jcc (im := im) (s := f18) (by rw [hf18rip]; exact cfg.fall.d19) hf18fl
      simpa [hf18rip] using hh
    obtain ⟨f19, hf19⟩ : ∃ u : X86Int.St, u = { f18 with rip := aFallTail + 74 } := ⟨_, rfl⟩
    have r14 : run im 1 (Out.ok f13) = Out.ok f14 := by rw [run_succ_ok im 0 f13, e14, ← hf14, run_zero]
    have r15 : run im 1 (Out.ok f14) = Out.ok f15 := by rw [run_succ_ok im 0 f14, e15, ← hf15, run_zero]
    have r16 : run im 1 (Out.ok f15) = Out.ok f16 := by rw [run_succ_ok im 0 f15, e16, ← hf16, run_zero]
    have r18 : run im 1 (Out.ok f17) = Out.ok f18 := by rw [run_succ_ok im 0 f17, e18, ← hf18, run_zero]
    have r19 : run im 1 (Out.ok f18) = Out.ok f19 := by rw [run_succ_ok im 0 f18, e19, ← hf19, run_zero]
    have hrun16 : run im (((5 + k6) + 5 + k12 + 1) + 3) (Out.ok s2) = Out.ok f16 := by
      rw [run_add im _ 3, hrun13, run_add im 1 2, r14, run_add im 1 1, r15, r16]
    have hrun17 : run im (((5 + k6) + 5 + k12 + 1) + 3 + k17) (Out.ok s2) = Out.ok f17 := by
      rw [run_add im _ k17, hrun16, hr17]
    have hrun19 : run im (((5 + k6) + 5 + k12 + 1) + 3 + k17 + 2) (Out.ok s2) = Out.ok f19 := by
      rw [run_add im _ 2, hrun17, run_add im 1 1, r18, r19]
    have hf19rip : f19.rip = aFallTail + 74 := by rw [hf19]
    have hf19r4 : f19.reg 4 = some sp := by rw [hf19, hf18]; exact hf17r4
    have hwf19 : wshape f19.mem = wshape s2.mem := by rw [hf19, hf18]; exact hwf17
    obtain ⟨k20, hk20, hr20⟩ := jmp_fail1_exit1B rc cr.fallAnnA st.sp32 st.spW f19 hf19rip hf19r4
      (run_regs_ok hrun19 hregs2)
      (StackRoom_mono (StackRoom_of_wshape hwf19 (by rw [hf19r4, hs2r4]) hroom2) (by decide))
    exact ⟨(((5 + k6) + 5 + k12 + 1) + 3 + k17 + 2) + k20, (by omega),
      by rw [run_add im _ k20, hrun19]; exact hr20⟩
  · -- a.d ≠ h'.d -> cr.fallAnnD
    have hf13fl : X86Int.cond f13.fl 4 = some false := by
      have hzf : f13.fl.zf = some false := by rw [hf13]; exact aluF_sub_zf_ne adW hd'W hdeq
      exact cond_cc4 hzf
    have e14 : X86Int.step im f13 = .ok { f13 with rip := aFallTail + 54 } := by
      have hh := step_jcc (im := im) (s := f13) (by rw [hf13rip]; exact cfg.fall.d14) hf13fl
      simpa [hf13rip] using hh
    obtain ⟨f14, hf14⟩ : ∃ u : X86Int.St, u = { f13 with rip := aFallTail + 54 } := ⟨_, rfl⟩
    have r14 : run im 1 (Out.ok f13) = Out.ok f14 := by rw [run_succ_ok im 0 f13, e14, ← hf14, run_zero]
    have hrun14 : run im (((5 + k6) + 5 + k12 + 1) + 1) (Out.ok s2) = Out.ok f14 := by
      rw [run_add im _ 1, hrun13, r14]
    have hf14rip : f14.rip = aFallTail + 54 := by rw [hf14]
    have hf14r4 : f14.reg 4 = some (sp - 8) := by rw [hf14, hf13]; exact hf12r4
    have hwf14 : wshape f14.mem = wshape s2.mem := by rw [hf14, hf13]; exact hwf12
    obtain ⟨k15, hk15, hr15⟩ := jmp_fail1_exit1B rc cr.fallAnnD (by omega) (by omega) f14 hf14rip hf14r4
      (run_regs_ok hrun14 hregs2)
      (StackRoom_shift4 hroom6 hf6r4 hf14r4 (hwf14.trans hwf6.symm) ⟨1, by omega, by omega⟩ hsp32 hspW)
    exact ⟨(((5 + k6) + 5 + k12 + 1) + 1) + k15, (by omega),
      by rw [run_add im _ k15, hrun14]; exact hr15⟩

private theorem walk_refuse_single_le {ys_len ann_len m : Nat}
    (hm : m ≤ 120 + 20 * (ann_len + 1)) :
    m ≤ (ys_len + 1 + 1) * (200 + 60 * (ann_len + 1)) := by
  have h1 : 1 ≤ ys_len + 1 + 1 := by omega
  have h2 := Nat.mul_le_mul_right (200 + 60 * (ann_len + 1)) h1
  omega

private theorem walk_refuse_ann_le {ys_len ann_len kh kt : Nat}
    (hkh : kh ≤ 80 + 20 * (ann_len + 1)) (hkt : kt ≤ 50 + 20 * (ann_len + 1)) :
    kh + kt ≤ (ys_len + 1 + 1 + 1) * (200 + 60 * (ann_len + 1)) := by
  have h1 : 1 ≤ ys_len + 1 + 1 + 1 := by omega
  have h2 := Nat.mul_le_mul_right (200 + 60 * (ann_len + 1)) h1
  omega

private theorem walk_refuse_step_le {ys_len ann_len k3 k4 : Nat}
    (hk3 : k3 ≤ 156 + 60 * (ann_len + 1))
    (hk4 : k4 ≤ (ys_len + 1 + 1) * (200 + 60 * (ann_len + 1))) :
    k3 + k4 ≤ (ys_len + 1 + 1 + 1) * (200 + 60 * (ann_len + 1)) := by
  have h3 : k3 ≤ 200 + 60 * (ann_len + 1) := by omega
  have h := Nat.add_le_add h3 hk4
  have e : (ys_len + 1 + 1 + 1) * (200 + 60 * (ann_len + 1)) =
      (200 + 60 * (ann_len + 1)) + (ys_len + 1 + 1) * (200 + 60 * (ann_len + 1)) := by
    rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]
  rw [e]
  exact h

/-- Shared fall-through continuation helper for `walk_refuseB`. -/
private theorem walk_refuse_some_cont {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W)
    {ins : Instr} {rest : List Instr} {i c len kh : Nat} {cur hm h' : Height}
    {s s2 : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hregs : RegsOk s)
    (hs : hstep pr wd hm ins = some (some h'))
    (hlen : ins.enc.length = len)
    (hrh : run im kh (Out.ok s) = Out.ok s2)
    (hkh : kh ≤ 80 + 20 * (wd.ann.length + 1))
    (hrip2 : s2.rip = aFallTail)
    (hpost : HPost s s2 h' (c + len) sp)
    (hw : Sporos.walk pr wd (ins :: rest) i cur = false)
    (ih : ∀ (i' c' : Nat) (cur' hm' : Height) (s' : X86Int.St),
      WalkInv pr wd w p sp tbl s' rest i' c' cur' hm' → RegsOk s' →
      Sporos.walk pr wd rest i' cur' = false →
      ∃ m, m ≤ (rest.length + 1) * (200 + 60 * (wd.ann.length + 1)) ∧
        run im m (Out.ok s') = Out.exit 1) :
    ∃ m, m ≤ ((ins :: rest).length + 1) * (200 + 60 * (wd.ann.length + 1)) ∧
      run im m (Out.ok s) = Out.exit 1 := by
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some (some h') := by
    rw [← inv.hlink]; exact hs
  obtain ⟨hbd, hba, -⟩ := hBnd_some_some inv.hb hs0
  have hd'W : h'.d < W := by omega
  have ha'W : h'.a < W := by omega
  cases rest with
  | nil =>
    have hi1_eq : i + 1 = wd.code.length := by
      have h := inv.isum; simp only [List.length_cons, List.length_nil] at h; exact h
    obtain ⟨kt, hkt, hrt⟩ := fall_end_refuseB cfg rc cr st hi1_eq hrip2 hpost
      inv.r1 inv.r7 inv.fnc inv.len16 inv.room inv.r4 (run_regs_ok hrh hregs)
    exact ⟨kh + kt, by simp only [List.length_cons, List.length_nil]; omega,
      by rw [run_add im kh kt, hrh]; exact hrt⟩
  | cons y ys =>
    have hi1 : i + 1 < wd.code.length := by
      have h := inv.isum; simp only [List.length_cons] at h; omega
    cases hmatch : (match wd.ann.lookup (i + 1) with | some a => a == h' | none => true) with
    | false =>
      cases hl1 : wd.ann.lookup (i + 1) with
      | none => simp [hl1] at hmatch
      | some a =>
        have hne : a ≠ h' := fun heq => by simp [hl1, heq] at hmatch
        obtain ⟨kt, hkt, hrt⟩ := fall_ann_refuseB cfg rc cr st hsp48 hhW hi1 hl1 hne hd'W ha'W
          hrip2 hpost inv.r1 inv.r7 inv.fnc inv.fnann inv.fannp inv.lay inv.len16 inv.room
          inv.r4 (run_regs_ok hrh hregs)
        have hle := walk_refuse_ann_le (ys_len := ys.length) hkh hkt
        exact ⟨kh + kt, by simpa only [List.length_cons] using hle,
          by rw [run_add im kh kt, hrh]; exact hrt⟩
    | true =>
      have hann1 : wd.ann.lookup (i + 1) = none ∨ wd.ann.lookup (i + 1) = some h' := by
        cases hl1 : wd.ann.lookup (i + 1) with
        | none => exact Or.inl rfl
        | some a =>
          have hbe : (a == h') = true := by simpa [hl1] using hmatch
          exact Or.inr (by rw [height_eq_of_beq hbe])
      have hwrest : Sporos.walk pr wd (y :: ys) (i + 1) h' = false := by
        change (match hstep pr wd ((wd.ann.lookup i).getD cur) ins with
          | none => false
          | some none => Sporos.walk pr wd (y :: ys) (i + 1) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
          | some (some h1) =>
            (match wd.ann.lookup (i + 1) with | some a => a == h1 | none => true) &&
              Sporos.walk pr wd (y :: ys) (i + 1) h1) = false at hw
        rw [hs0] at hw
        dsimp only at hw
        rcases hann1 with hl1 | hl1
        · rw [hl1] at hw; exact hw
        · rw [hl1] at hw
          dsimp only at hw
          have hb : (h' == h') = true := by cases h'; simp [BEq.beq]
          rw [hb] at hw; exact hw
      obtain ⟨k3, s3, hk3, hr3, inv3, -, -, -⟩ :=
        fall_finish1B cfg st inv hs hann1 hi1 hlen hrh hkh hrip2 hpost
      obtain ⟨k4, hk4, hr4⟩ := ih (i + 1) (c + len) h' h' s3 inv3 (run_regs_ok hr3 hregs) hwrest
      have hle := walk_refuse_step_le hk3 hk4
      exact ⟨k3 + k4, by simpa only [List.length_cons] using hle,
        by rw [run_add im k3 k4, hr3]; exact hr4⟩

/-- Shared non-fall-through continuation helper for `walk_refuseB`. -/
private theorem walk_refuse_none_cont {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {rest : List Instr} {i c len kh : Nat} {cur hm : Height}
    {s s2 : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hregs : RegsOk s)
    (hs : hstep pr wd hm ins = some none)
    (hlen : ins.enc.length = len)
    (hrh : run im kh (Out.ok s) = Out.ok s2)
    (hkh : kh ≤ 80 + 20 * (wd.ann.length + 1))
    (hrip2 : s2.rip = aNoneTail)
    (hpost : HPost s s2 hm (c + len) sp)
    (hw : Sporos.walk pr wd (ins :: rest) i cur = false)
    (ih : ∀ (i' c' : Nat) (cur' hm' : Height) (s' : X86Int.St),
      WalkInv pr wd w p sp tbl s' rest i' c' cur' hm' → RegsOk s' →
      Sporos.walk pr wd rest i' cur' = false →
      ∃ m, m ≤ (rest.length + 1) * (200 + 60 * (wd.ann.length + 1)) ∧
        run im m (Out.ok s') = Out.exit 1) :
    ∃ m, m ≤ ((ins :: rest).length + 1) * (200 + 60 * (wd.ann.length + 1)) ∧
      run im m (Out.ok s) = Out.exit 1 := by
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some none := by
    rw [← inv.hlink]; exact hs
  cases rest with
  | nil =>
    simp only [Sporos.walk, hs0] at hw
    cases hw
  | cons y ys =>
    have hi1 : i + 1 < wd.code.length := by
      have h := inv.isum; simp only [List.length_cons] at h; omega
    have hwrest : Sporos.walk pr wd (y :: ys) (i + 1) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) = false := by
      simp only [Sporos.walk, hs0] at hw; exact hw
    obtain ⟨k3, s3, hk3, hr3, inv3, -, -, -⟩ :=
      none_finish_cons1B cfg st inv hs hi1 hlen hrh hkh hrip2 hpost
    obtain ⟨k4, hk4, hr4⟩ :=
      ih (i + 1) (c + len) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
        s3 inv3 (run_regs_ok hr3 hregs) hwrest
    have hle := walk_refuse_step_le hk3 hk4
    exact ⟨k3 + k4, by simpa only [List.length_cons] using hle,
      by rw [run_add im k3 k4, hr3]; exact hr4⟩

/-- Bounded refusal direction of the single-pass code walk: whenever
`Sporos.walk pr wd code i cur = false`, starting from `WalkInv` the checker exits `1`
within `(code.length + 1) * (200 + 60 * (wd.ann.length + 1))` steps. -/
theorem walk_refuseB {im : Nat → Option (Ins × Nat)}
    (cfg : AllCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    (hsp48 : 48 ≤ sp) (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W) (houtW : wd.outs < W) :
    ∀ (code : List Instr) (i c : Nat) (cur hm : Height) (s : X86Int.St),
      WalkInv pr wd w p sp tbl s code i c cur hm → RegsOk s →
      Sporos.walk pr wd code i cur = false →
      ∃ m, m ≤ (code.length + 1) * (200 + 60 * (wd.ann.length + 1)) ∧
        run im m (Out.ok s) = Out.exit 1 := by
  intro code
  induction code with
  | nil =>
    intro i c cur hm s _ _ hw
    simp [Sporos.walk] at hw
  | cons ins rest ih =>
    intro i c cur hm s inv hregs hw
    cases ins with
    | prim pq =>
      cases hs : hstep pr wd hm (Instr.prim pq) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_prim_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_prim_exitB cfg st inv hs
          exact walk_refuse_none_cont cfg st inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
        | some h' =>
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_prim_fallB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | lit n =>
      have hs : hstep pr wd hm (Instr.lit n) = some (some ⟨hm.d + 1, hm.a⟩) := rfl
      obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_litB cfg st inv
      exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | str n =>
      have hs : hstep pr wd hm (Instr.str n) = some (some ⟨hm.d + 2, hm.a⟩) := rfl
      obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_strB cfg st inv
      exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | pick n =>
      cases hs : hstep pr wd hm (Instr.pick n) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_pick_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : n < hm.d <;> simp [hstep, hn] at hs
        | some h' =>
          have heq : h' = ⟨hm.d + 1, hm.a⟩ := (hstep_pick hs).2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_pickB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | tor =>
      cases hs : hstep pr wd hm Instr.tor with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_tor_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.d <;> simp [hstep, hn] at hs
        | some h' =>
          have heq : h' = ⟨hm.d - 1, hm.a + 1⟩ := (hstep_tor hs).2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_torB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | fromr =>
      cases hs : hstep pr wd hm Instr.fromr with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_fromr_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.a <;> simp [hstep, hn] at hs
        | some h' =>
          have heq : h' = ⟨hm.d + 1, hm.a - 1⟩ := (hstep_fromr hs).2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_fromrB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | rfetch =>
      cases hs : hstep pr wd hm Instr.rfetch with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_rfetch_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.a <;> simp [hstep, hn] at hs
        | some h' =>
          have heq : h' = ⟨hm.d + 1, hm.a⟩ := (hstep_rfetch hs).2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_rfetchB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | locget n =>
      cases hs : hstep pr wd hm (Instr.locget n) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_locget_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : n < wd.locals <;> simp [hstep, hn] at hs
        | some h' =>
          have heq : h' = ⟨hm.d + 1, hm.a⟩ := (hstep_locget hs).2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_locgetB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | locset n =>
      cases hs : hstep pr wd hm (Instr.locset n) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_locset_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.d ∧ n < wd.locals
          · simp only [hstep] at hs; rw [if_pos hn] at hs; cases hs
          · simp only [hstep] at hs; rw [if_neg hn] at hs; cases hs
        | some h' =>
          have heq : h' = ⟨hm.d - 1, hm.a⟩ := (hstep_locset hs).2.2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_locsetB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | jmp t =>
      cases hs : hstep pr wd hm (Instr.jmp t) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_jmp_refuseB cfg rc cr st hsp48 hhW inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | some h' =>
          exfalso; by_cases hn : wd.ann.lookup t = some hm
          · simp only [hstep] at hs; rw [if_pos hn] at hs; cases hs
          · simp only [hstep] at hs; rw [if_neg hn] at hs; cases hs
        | none =>
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_jmpB cfg st inv hs
          exact walk_refuse_none_cont cfg st inv hregs hs rfl hrh hkh hrip2 hpost hw ih
    | jz t =>
      cases hs : hstep pr wd hm (Instr.jz t) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_jz_refuseB cfg rc cr st hsp48 hhW inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.d ∧ wd.ann.lookup t = some ⟨hm.d - 1, hm.a⟩
          · simp only [hstep] at hs; rw [if_pos hn] at hs; cases hs
          · simp only [hstep] at hs; rw [if_neg hn] at hs; cases hs
        | some h' =>
          have heq : h' = ⟨hm.d - 1, hm.a⟩ := (hstep_jz hs).2.2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_jzB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh hkh hrip2 hpost hw ih
    | jnz t =>
      cases hs : hstep pr wd hm (Instr.jnz t) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_jnz_refuseB cfg rc cr st hsp48 hhW inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso; by_cases hn : 1 ≤ hm.d ∧ wd.ann.lookup t = some ⟨hm.d - 1, hm.a⟩
          · simp only [hstep] at hs; rw [if_pos hn] at hs; cases hs
          · simp only [hstep] at hs; rw [if_neg hn] at hs; cases hs
        | some h' =>
          have heq : h' = ⟨hm.d - 1, hm.a⟩ := (hstep_jnz hs).2.2
          subst heq
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_jnzB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh hkh hrip2 hpost hw ih
    | call cwidx =>
      cases hs : hstep pr wd hm (Instr.call cwidx) with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_call_refuseB cfg rc cr st inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | none =>
          exfalso
          cases hcw : pr.words[cwidx]? with
          | none => simp [hstep, hcw] at hs
          | some cw => by_cases hg : cw.ins ≤ hm.d <;> simp [hstep, hcw, hg] at hs
        | some h' =>
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_callB cfg st inv hs
          exact walk_refuse_some_cont cfg rc cr st hsp48 hhW inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih
    | ret =>
      cases hs : hstep pr wd hm Instr.ret with
      | none =>
        obtain ⟨m, hm, hr⟩ := hand_ret_refuseB cfg rc cr st houtW inv hs hregs
        exact ⟨m, walk_refuse_single_le (by omega), hr⟩
      | some o =>
        cases o with
        | some h' =>
          exfalso; by_cases heq : hm = ⟨wd.outs, 0⟩ <;> simp [hstep, heq] at hs
        | none =>
          obtain ⟨kh, s2, hkh, hrh, hrip2, hpost⟩ := hand_retB cfg st inv hs
          exact walk_refuse_none_cont cfg st inv hregs hs rfl hrh (by omega) hrip2 hpost hw ih

end SporosX1
