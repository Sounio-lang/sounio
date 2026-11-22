import Bootstrap.SporosVmOpsData2

/-!
# Native x86-64 VM Stack Transfer Operations (`O_pick`, `O_tor`, `O_fromr`, `O_rfetch`)

Simulation proofs over `X86Vm` for `O_pick`, `O_tor`, `O_fromr`, and `O_rfetch`.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmOpsData

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase
  SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

local instance : Inhabited Sporos.Word := ⟨⟨0, 0, 0, [], []⟩⟩
local instance : Inhabited Sporos.Instr := ⟨.ret⟩
local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### `O_pick` and `O_tor` simulation specs -/

theorem O_pick_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {n : Nat}
    (hrip : s.rip = instrEntry (.pick n)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .pick n)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.pick n)).2) :
    OpPostOutcome pr ls s := by
  have hnW : n < X86Env.W := by
    have hf := flatProgFit_ins_at (vc.code_flat h).2.2.2.1 hnw hpc; rw [hins] at hf; exact hf
  have h264' : rd64 s.mem (ea ctxBase 264) = some n := by
    simpa [instrPair, Nat.mod_eq_of_lt hnW] using h264
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some n) with rip := 4199496 } := by
    rw [step_at_vc 16 hrip (by decide : (SporosVmBin.O_pick, (.ofInt (.ld 0 15 264)), 7) ∈ vmTableChunk 16),
      exec_ld h15 h264', hrip]; rfl
  let s1 := { s.set 0 (some n) with rip := 4199496 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hdsLen1 : rd64 s1.mem (ea ctxBase 8) = some ls.ds.length := vc.rd_ctx_dsLen (s := s) h
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds.length) with rip := 4199500 } := by
    rw [step_at_vc 16 rfl (by decide : (4199496, (.ofInt (.ld 1 15 8)), 4) ∈ vmTableChunk 16),
      exec_ld hs1_15 hdsLen1]
  let s2 := { s1.set 1 (some ls.ds.length) with rip := 4199500 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some n := by
    change (s1.set 1 _).reg 0 = some n; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr (by decide) _
  have hs2_1 : s2.reg 1 = some ls.ds.length := reg_set_self_ok' hr1 (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub n ls.ds.length).2, rip := 4199503 } := by
    rw [step_at_vc 16 rfl (by decide : (4199500, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 16),
      exec_cmp hs2_0 hs2_1]
  let s3 := { s2 with fl := (X86Int.aluF .sub n ls.ds.length).2, rip := 4199503 }
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  unfold OpPostOutcome
  by_cases hn : n < ls.ds.length
  · have hget : ls.ds[n]? = some ls.ds[n]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hn]; rfl
    let v := ls.ds[n]!
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with ds := SporosProvLen.ltag ls.pc v :: ls.ds, rs := { f with pc := f.pc + 1 } :: rest } := by
      unfold SporosProvLen.lstep; rw [hrs]; dsimp only; rw [hfetch]; dsimp only; rw [hget]
      simp [SporosProvLen.LSt.setTop, v]
    rw [hlstep]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4199509 } := by
      rw [step_at_vc 16 rfl (by decide : (4199503, (.ofInt (.jcc 3 4201048)), 6) ∈ vmTableChunk 16),
        exec_jcc (cond_aluF_sub_ae n ls.ds.length)]
      dsimp [s3]; rw [decide_eq_false (by omega)]; rfl
    let s4 := { s3 with rip := 4199509 }
    have ⟨⟨_, _, hds_len, _, hds_hi, hds_rel, hds_le⟩, _, _⟩ := vc.arena_bounds h
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hds_ltW : ls.ds.length * 16 < X86Env.W := by simp only [Region.hi, hds_len] at hds_hi; omega
    have hsub1 : (X86Int.aluF .sub ls.ds.length n).1 = ls.ds.length - n := by
      rw [aluF_sub_val,
        show ls.ds.length + X86Env.W - n = (ls.ds.length - n) + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have hs4_1 : s4.reg 1 = some ls.ds.length := hs2_1
    have hs4_0 : s4.reg 0 = some n := hs2_0
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 1 (some (ls.ds.length - n)) with fl := (X86Int.aluF .sub ls.ds.length n).2, rip := 4199512 } := by
      rw [step_at_vc 16 rfl (by decide : (4199509, (.ofInt (.alu .sub 1 0)), 3) ∈ vmTableChunk 16),
        exec_alu hs4_1 hs4_0]; dsimp only; rw [hsub1]
    let s5 := { s4.set 1 (some (ls.ds.length - n)) with fl := (X86Int.aluF .sub ls.ds.length n).2, rip := 4199512 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_1 : s5.reg 1 = some (ls.ds.length - n) := reg_set_self_ok' hr2 (by decide) _
    have hsub2 : (X86Int.aluF .sub (ls.ds.length - n) (1 % X86Env.W)).1 = ls.ds.length - 1 - n := by
      rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val,
        show ls.ds.length - n + X86Env.W - 1 = (ls.ds.length - 1 - n) + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5.set 1 (some (ls.ds.length - 1 - n)) with
              fl := (X86Int.aluF .sub (ls.ds.length - n) (1 % X86Env.W)).2, rip := 4199516 } := by
      rw [step_at_vc 16 rfl (by decide : (4199512, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 16),
        exec_aluI hs5_1]; dsimp only; rw [hsub2]
    let s6 := { s5.set 1 (some (ls.ds.length - 1 - n)) with
                fl := (X86Int.aluF .sub (ls.ds.length - n) (1 % X86Env.W)).2, rip := 4199516 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_1 : s6.reg 1 = some (ls.ds.length - 1 - n) := reg_set_self_ok' hr5 (by decide) _
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 1 (some ((ls.ds.length - 1 - n) * 16)) with fl := Flags.undef, rip := 4199520 } := by
      rw [step_at_vc 16 rfl (by decide : (4199516, (.ofInt (.shlI 1 4)), 4) ∈ vmTableChunk 16),
        exec_shlI_4_lt hs6_1 (by omega)]
    let s7 := { s6.set 1 (some ((ls.ds.length - 1 - n) * 16)) with fl := Flags.undef, rip := 4199520 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
    have hs7_1 : s7.reg 1 = some ((ls.ds.length - 1 - n) * 16) := reg_set_self_ok' hr6 (by decide) _
    have hs7_15 : s7.reg 15 = some ctxBase := by
      change (s6.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s5.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s4.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
    have hdsBase7 : rd64 s7.mem (ea ctxBase 0) = some vc.dsReg.base := vc.rd_ctx_dsBase (s := s) h
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some vc.dsReg.base) with rip := 4199523 } := by
      rw [step_at_vc 16 rfl (by decide : (4199520, (.ofInt (.ld 2 15 0)), 3) ∈ vmTableChunk 16),
        exec_ld hs7_15 hdsBase7]
    let s8 := { s7.set 2 (some vc.dsReg.base) with rip := 4199523 }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
    have hs8_2 : s8.reg 2 = some vc.dsReg.base := reg_set_self_ok' hr7 (by decide) _
    have hs8_1 : s8.reg 1 = some ((ls.ds.length - 1 - n) * 16) := by
      change (s7.set 2 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs7_1
    have hpair := rd64_pairList_top h.1.2 (vc.mem_ds h) hds_hi hds_rel hn
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 7 (some v.w) with rip := 4199527 } := by
      rw [step_at_vc 16 rfl (by decide : (4199523, (.ldIdx .w64 7 2 1 1 0), 4) ∈ vmTableChunk 16),
        exec_ldIdx hs8_2 hs8_1 hpair.1]
    let s9 := { s8.set 7 (some v.w) with rip := 4199527 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_2 : s9.reg 2 = some vc.dsReg.base := by
      change (s8.set 7 _).reg 2 = some _; rw [reg_set_ne (by decide)]; exact hs8_2
    have hs9_1 : s9.reg 1 = some ((ls.ds.length - 1 - n) * 16) := by
      change (s8.set 7 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs8_1
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 6 (some v.plen) with rip := 4199532 } := by
      rw [step_at_vc 16 rfl (by decide : (4199527, (.ldIdx .w64 6 2 1 1 8), 5) ∈ vmTableChunk 16),
        exec_ldIdx hs9_2 hs9_1 hpair.2]
    let s10 := { s9.set 6 (some v.plen) with rip := 4199532 }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
    have hs10_6 : s10.reg 6 = some v.plen := reg_set_self_ok' hr9 (by decide) _
    have hs10_7 : s10.reg 7 = some v.w := by
      change (s9.set 6 _).reg 7 = some v.w; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr8 (by decide) _
    have hwf10 : vc.wf pr ls s10 := vc.wf_step_misc h rfl hr10
      (by change (s9.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s8.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s7.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s6.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s5.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s4.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs3_4)
      (by change (s9.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s8.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s7.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs7_15)
    obtain ⟨s11, hrch11, hrip11, _, hk11, hwf11, hs11_6, hs11_oth⟩ :=
      step_tag_prov (a := 4199532) (chk := 16) (rdst := 6) (rtmp := 8) rfl hwf10 hs10_6
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)
    have hrch11_all : Reaches vmImage s s11 :=
      Reaches.trans hrch3 (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
        (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10 hrch11)))))))
    have hvW : v.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt hn)).1
    have hpW : (v.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [M_eq_W']; exact lt_W_of_mod _
    rcases step_call_ds_push_vc (step_at_vc 16 hrip11 (by decide : (4199546, (.ofInt (.call 4201446)), 5) ∈ vmTableChunk 16))
        hwf11 (oracleAdmits_of_reaches hor hrch11_all) (by rw [hrip11]; omega)
        (by rw [hs11_oth 7 (by decide) (by decide)]; exact hs10_7) hs11_6 hvW hpW with
      ⟨vc12, s12, hrch12, hrip12, hk12, hor12, hwf12, _⟩ | ⟨hhit, hends⟩
    · rcases adv_pc_finish_vc (a := 4199551) (chk := 16) (by rw [hrip12, hrip11]) hwf12 hor12 hrs hpcW
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) with
        ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
      · exact Or.inl ⟨sf, Reaches.trans hrch11_all (Reaches.trans hrch12 hrchf),
          by rw [hkf, hk12, hk11]; rfl, horf, hrelf⟩
      · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch11_all hrch12) hhit,
          Ends.of_reaches (Reaches.trans hrch11_all hrch12) hends⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch11_all hhit, Ends.of_reaches hrch11_all hends⟩
  · have hnone : ls.ds[n]? = none := List.getElem?_eq_none (by omega)
    have hlstep : SporosProvLen.lstep pr ls = .fail .underflow := by
      simp [SporosProvLen.lstep, hrs, hfetch, hnone]
    rw [hlstep]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := errEntry .underflow } := by
      rw [step_at_vc 16 rfl (by decide : (4199503, (.ofInt (.jcc 3 4201048)), 6) ∈ vmTableChunk 16),
        exec_jcc (cond_aluF_sub_ae n ls.ds.length)]
      dsimp [s3]; rw [decide_eq_true (by omega)]; rfl
    let s4 := { s3 with rip := errEntry .underflow }
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr2 hs3_4 hs3_15
    exact lift_fail_outcome (Reaches.trans hrch3 (Reaches.step h04)) rfl
      (L_fail_from_vc_spec (e := .underflow) rfl hwf4)

theorem O_tor_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = instrEntry .tor) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .tor)
    (hpcW : f.pc + 1 < X86Env.W)
    (_h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair .tor).2) :
    OpPostOutcome pr ls s := by
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds.length) with rip := 4199591 } := by
    rw [step_at_vc 16 hrip (by decide : (SporosVmBin.O_tor, (.ofInt (.ld 1 15 8)), 4) ∈ vmTableChunk 16),
      exec_ld h15 (vc.rd_ctx_dsLen h), hrip]; rfl
  let s1 := { s.set 1 (some ls.ds.length) with rip := 4199591 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_1 : s1.reg 1 = some ls.ds.length := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := 4199594 } := by
    rw [step_at_vc 16 rfl (by decide : (4199591, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 16),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := 4199594 }
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  unfold OpPostOutcome
  cases hds : ls.ds with
  | nil =>
    have hlstep : SporosProvLen.lstep pr ls = .fail .underflow := by
      simp [SporosProvLen.lstep, hrs, hfetch, hds]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := errEntry .underflow } := by
      rw [step_at_vc 16 rfl (by decide : (4199594, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 16),
        exec_jcc (cond_aluF_and_e ls.ds.length ls.ds.length)]
      simp [s2, hds]; rfl
    exact lift_fail_outcome (Reaches.trans hrch2 (Reaches.step h03)) rfl
      (L_fail_from_vc_spec (e := .underflow) rfl (vc.wf_step_misc h rfl hr1 hs2_4 hs2_15))
  | cons v r =>
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with ds := r, aux := SporosProvLen.ltag ls.pc v :: ls.aux,
                        rs := { f with pc := f.pc + 1 } :: rest } := by
      simp [SporosProvLen.lstep, hrs, hfetch, hds, SporosProvLen.LSt.setTop]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199600 } := by
      rw [step_at_vc 16 rfl (by decide : (4199594, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 16),
        exec_jcc (cond_aluF_and_e ls.ds.length ls.ds.length)]
      simp [s2, hds]
    let s3 := { s2 with rip := 4199600 }
    have ⟨⟨_, _, hds_len, _, hds_hi, hds_rel, hds_le⟩, _, _⟩ := vc.arena_bounds h
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hds_ltW : ls.ds.length * 16 < X86Env.W := by simp only [Region.hi, hds_len] at hds_hi; omega
    have hlen_eq : ls.ds.length = r.length + 1 := by simp [hds]
    have hsub1 : (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).1 = r.length := by
      rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val,
        show ls.ds.length + X86Env.W - 1 = r.length + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have hs3_1 : s3.reg 1 = some ls.ds.length := hs1_1
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).2, rip := 4199604 } := by
      rw [step_at_vc 16 rfl (by decide : (4199600, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 16),
        exec_aluI hs3_1]; dsimp only; rw [hsub1]
    let s4 := { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).2, rip := 4199604 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some r.length := reg_set_self_ok' hr1 (by decide) _
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4
      (by change (s3.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4)
      (by change (s3.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15)
    obtain ⟨s5, h05, hrip5, hk5, hregs5, _, hwf5⟩ :=
      wf_dec_dsLen (step_at_vc 16 rfl (by decide : (4199604, (.ofInt (.st 15 8 1)), 4) ∈ vmTableChunk 16))
        hwf4 hds hs4_1
    let vc5 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 r.length) } }
    have hs5_1 : s5.reg 1 = some r.length := by simp only [X86Int.St.reg, hregs5]; exact hs4_1
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5.set 1 (some (r.length * 16)) with fl := Flags.undef, rip := 4199612 } := by
      rw [step_at_vc 16 hrip5 (by decide : (4199608, (.ofInt (.shlI 1 4)), 4) ∈ vmTableChunk 16),
        exec_shlI_4_lt hs5_1 (by omega), hrip5]
    let s6 := { s5.set 1 (some (r.length * 16)) with fl := Flags.undef, rip := 4199612 }
    have hr6 : RegsOk s6 := regsOk_set hwf5.2.2.2.1 (by decide) _
    have hs6_1 : s6.reg 1 = some (r.length * 16) := reg_set_self_ok' hwf5.2.2.2.1 (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf5.2.2.2.2.2.1
    have hdsBase6 : rd64 s6.mem (ea ctxBase 0) = some vc.dsReg.base := vc5.rd_ctx_dsBase (s := s5) hwf5
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 2 (some vc.dsReg.base) with rip := 4199615 } := by
      rw [step_at_vc 16 rfl (by decide : (4199612, (.ofInt (.ld 2 15 0)), 3) ∈ vmTableChunk 16),
        exec_ld hs6_15 hdsBase6]
    let s7 := { s6.set 2 (some vc.dsReg.base) with rip := 4199615 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
    have hs7_2 : s7.reg 2 = some vc.dsReg.base := reg_set_self_ok' hr6 (by decide) _
    have hs7_1 : s7.reg 1 = some (r.length * 16) := by
      change (s6.set 2 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs6_1
    have hpair := rd64_pairList_top hwf5.1.2 (vc5.mem_ds hwf5) hds_hi hds_rel (k := 0) (by simp [hds])
    have hpair0 : rdMem .w64 s7.mem (eaIdx vc.dsReg.base (r.length * 16) 1 0) = some v.w ∧
        rdMem .w64 s7.mem (eaIdx vc.dsReg.base (r.length * 16) 1 8) = some v.plen := by
      simpa [s7, s6, vc5, hlen_eq, hds] using hpair
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 7 (some v.w) with rip := 4199619 } := by
      rw [step_at_vc 16 rfl (by decide : (4199615, (.ldIdx .w64 7 2 1 1 0), 4) ∈ vmTableChunk 16),
        exec_ldIdx hs7_2 hs7_1 hpair0.1]
    let s8 := { s7.set 7 (some v.w) with rip := 4199619 }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
    have hs8_2 : s8.reg 2 = some vc.dsReg.base := by
      change (s7.set 7 _).reg 2 = some _; rw [reg_set_ne (by decide)]; exact hs7_2
    have hs8_1 : s8.reg 1 = some (r.length * 16) := by
      change (s7.set 7 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs7_1
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 6 (some v.plen) with rip := 4199624 } := by
      rw [step_at_vc 17 rfl (by decide : (4199619, (.ldIdx .w64 6 2 1 1 8), 5) ∈ vmTableChunk 17),
        exec_ldIdx hs8_2 hs8_1 hpair0.2]
    let s9 := { s8.set 6 (some v.plen) with rip := 4199624 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_6 : s9.reg 6 = some v.plen := reg_set_self_ok' hr8 (by decide) _
    have hs9_7 : s9.reg 7 = some v.w := by
      change (s8.set 6 _).reg 7 = some v.w; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr7 (by decide) _
    have hwf9 : vc5.wf pr { ls with ds := r } s9 := vc5.wf_step_misc hwf5 rfl hr9
      (by change (s8.set 6 _).reg 4 = some vc5.sp; rw [reg_set_ne (by decide)]
          change (s7.set 7 _).reg 4 = some vc5.sp; rw [reg_set_ne (by decide)]
          change (s6.set 2 _).reg 4 = some vc5.sp; rw [reg_set_ne (by decide)]
          change (s5.set 1 _).reg 4 = some vc5.sp; rw [reg_set_ne (by decide)]; exact hwf5.2.2.2.2.1)
      (by change (s8.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s7.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s6.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs6_15)
    obtain ⟨s10, hrch10, hrip10, _, hk10, hwf10, hs10_6, hs10_oth⟩ :=
      step_tag_prov (a := 4199624) (chk := 17) (rdst := 6) (rtmp := 8) rfl hwf9 hs9_6
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)
    have hrch10_all : Reaches vmImage s s10 :=
      Reaches.trans hrch2 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step_trans h05
        (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09 hrch10)))))))
    have hvW : v.w < X86Env.W := (wf_ds_lt_W h (by rw [hds]; exact List.mem_cons_self ..)).1
    have hpW : (v.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [M_eq_W']; exact lt_W_of_mod _
    rcases step_call_aux_push_vc (step_at_vc 17 hrip10 (by decide : (4199638, (.ofInt (.call 4201534)), 5) ∈ vmTableChunk 17))
        hwf10 (oracleAdmits_of_reaches hor hrch10_all) (by rw [hrip10]; omega)
        (by rw [hs10_oth 7 (by decide) (by decide)]; exact hs9_7) hs10_6 hvW hpW with
      ⟨vc11, s11, hrch11, hrip11, hk11, hor11, hwf11⟩ | ⟨hhit, hends⟩
    · rcases adv_pc_finish_vc (a := 4199643) (chk := 17) (by rw [hrip11, hrip10]) hwf11 hor11 hrs hpcW
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) with
        ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
      · exact Or.inl ⟨sf, Reaches.trans hrch10_all (Reaches.trans hrch11 hrchf),
          by rw [hkf, hk11, hk10]; exact congrArg X86Env.Kern.out hk5, horf, hrelf⟩
      · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch10_all hrch11) hhit,
          Ends.of_reaches (Reaches.trans hrch10_all hrch11) hends⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch10_all hhit, Ends.of_reaches hrch10_all hends⟩

/-! ### Shared aux-top-to-ds tail (`O_fromr`, `O_rfetch`) and specs -/

private theorem aux_top_to_ds_adv_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {v : SporosProvLen.LVal} {r : List SporosProvLen.LVal}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hr1 : s.reg 1 = some r.length)
    (haux_rel : PairListRel vc.auxReg.data (r.length + 1) (v :: r))
    (hvW : v.w < X86Env.W) (haW : a + 36 < X86Env.W)
    (hm1 : (a, (.ofInt (.shlI 1 4)), 4) ∈ vmTableChunk chk)
    (hm2 : (a + 4, (.ofInt (.ld 2 15 24)), 4) ∈ vmTableChunk chk)
    (hm3 : (a + 8, (.ldIdx .w64 7 2 1 1 0), 4) ∈ vmTableChunk chk)
    (hm4 : (a + 12, (.ldIdx .w64 6 2 1 1 8), 5) ∈ vmTableChunk chk)
    (hm5 : (a + 17, (.ofInt (.ld 8 15 224)), 7) ∈ vmTableChunk chk)
    (hm6 : (a + 24, (.ofInt (.alu .add 6 8)), 3) ∈ vmTableChunk chk)
    (hm7 : (a + 27, (.ofInt (.aluI .add 6 1)), 4) ∈ vmTableChunk chk)
    (hm8 : (a + 31, (.ofInt (.call SporosVmBin.fn_ds_push)), 5) ∈ vmTableChunk chk)
    (hp1 : (a + 36, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk chk)
    (hp2 : (a + 40, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk chk)
    (hp3 : (a + 44, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk chk)
    (hp4 : (a + 48, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk chk)
    (hp5 : (a + 52, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk chk)
    (hp6 : (a + 55, (.ofInt (.ld 1 0 8)), 4) ∈ vmTableChunk chk)
    (hp7 : (a + 59, (.ofInt (.aluI .add 1 1)), 4) ∈ vmTableChunk chk)
    (hp8 : (a + 63, (.ofInt (.st 0 8 1)), 4) ∈ vmTableChunk chk)
    (hp9 : (a + 67, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk chk) :
    let ls' := { ls with ds := SporosProvLen.ltag ls.pc v :: ls.ds, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have ⟨_, ⟨_, _, haux_len, _, haux_hi, _, _⟩, _⟩ := vc.arena_bounds h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hr_ltW : r.length * 16 < X86Env.W := by
    have := haux_rel.2.1; simp only [Region.hi] at haux_hi; omega
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 1 (some (r.length * 16)) with fl := Flags.undef, rip := a + 4 } := by
    rw [step_at_vc chk hrip hm1, exec_shlI_4_lt hr1 hr_ltW, hrip]
  let s1 := { s.set 1 (some (r.length * 16)) with fl := Flags.undef, rip := a + 4 }
  have hr1' : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_1 : s1.reg 1 = some (r.length * 16) := reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.2.1
  have hauxBase1 : rd64 s1.mem (ea ctxBase 24) = some vc.auxReg.base := rd_ctx_auxBase (s := s) h
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some vc.auxReg.base) with rip := a + 8 } := by
    rw [step_at_vc chk rfl hm2, exec_ld hs1_15 hauxBase1]
  let s2 := { s1.set 2 (some vc.auxReg.base) with rip := a + 8 }
  have hr2 : RegsOk s2 := regsOk_set hr1' (by decide) _
  have hs2_2 : s2.reg 2 = some vc.auxReg.base := reg_set_self_ok' hr1' (by decide) _
  have hs2_1 : s2.reg 1 = some (r.length * 16) := by
    change (s1.set 2 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs1_1
  have hpair := rd64_pairList_top h.1.2 (vc.mem_aux h) haux_hi haux_rel (k := 0) (by simp)
  have hpair0 : rdMem .w64 s2.mem (eaIdx vc.auxReg.base (r.length * 16) 1 0) = some v.w ∧
      rdMem .w64 s2.mem (eaIdx vc.auxReg.base (r.length * 16) 1 8) = some v.plen := by
    simpa [s2, s1] using hpair
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 7 (some v.w) with rip := a + 12 } := by
    rw [step_at_vc chk rfl hm3, exec_ldIdx hs2_2 hs2_1 hpair0.1]
  let s3 := { s2.set 7 (some v.w) with rip := a + 12 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_2 : s3.reg 2 = some vc.auxReg.base := by
    change (s2.set 7 _).reg 2 = some _; rw [reg_set_ne (by decide)]; exact hs2_2
  have hs3_1 : s3.reg 1 = some (r.length * 16) := by
    change (s2.set 7 _).reg 1 = some _; rw [reg_set_ne (by decide)]; exact hs2_1
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 6 (some v.plen) with rip := a + 17 } := by
    rw [step_at_vc chk rfl hm4, exec_ldIdx hs3_2 hs3_1 hpair0.2]
  let s4 := { s3.set 6 (some v.plen) with rip := a + 17 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_6 : s4.reg 6 = some v.plen := reg_set_self_ok' hr3 (by decide) _
  have hs4_7 : s4.reg 7 = some v.w := by
    change (s3.set 6 _).reg 7 = some v.w; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr2 (by decide) _
  have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4
    (by change (s3.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s2.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s1.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
    (by change (s3.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s2.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s1.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15)
  obtain ⟨s5, hrch5, hrip5, _, hk5, hwf5, hs5_6, hs5_oth⟩ :=
    step_tag_prov (a := a + 17) (chk := chk) (rdst := 6) (rtmp := 8) rfl hwf4 hs4_6
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      hm5 (by simpa [Nat.add_assoc] using hm6) (by simpa [Nat.add_assoc] using hm7)
  have hrch5_all : Reaches vmImage s s5 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04 hrch5)))
  have hpW : (v.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [M_eq_W']; exact lt_W_of_mod _
  have hrip5' : s5.rip = a + 31 := by rw [hrip5]
  rcases step_call_ds_push_vc (step_at_vc chk hrip5' hm8) hwf5 (oracleAdmits_of_reaches hor hrch5_all)
      (by rw [hrip5']; omega) (by rw [hs5_oth 7 (by decide) (by decide)]; exact hs4_7) hs5_6 hvW hpW with
    ⟨vc6, s6, hrch6, hrip6, hk6, hor6, hwf6, _⟩ | ⟨hhit, hends⟩
  · have hrip6' : s6.rip = a + 36 := by rw [hrip6, hrip5']
    rcases adv_pc_finish_vc (a := a + 36) (chk := chk) hrip6' hwf6 hor6 hrs hpcW
        hp1 (by simpa [Nat.add_assoc] using hp2) (by simpa [Nat.add_assoc] using hp3)
        (by simpa [Nat.add_assoc] using hp4) (by simpa [Nat.add_assoc] using hp5)
        (by simpa [Nat.add_assoc] using hp6) (by simpa [Nat.add_assoc] using hp7)
        (by simpa [Nat.add_assoc] using hp8) (by simpa [Nat.add_assoc] using hp9) with
      ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch5_all (Reaches.trans hrch6 hrchf),
        by rw [hkf, hk6, hk5]; rfl, horf, hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch5_all hrch6) hhit,
        Ends.of_reaches (Reaches.trans hrch5_all hrch6) hends⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch5_all hhit, Ends.of_reaches hrch5_all hends⟩

theorem O_fromr_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = instrEntry .fromr) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .fromr)
    (hpcW : f.pc + 1 < X86Env.W)
    (_h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair .fromr).2) :
    OpPostOutcome pr ls s := by
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.aux.length) with rip := 4199683 } := by
    rw [step_at_vc 17 hrip (by decide : (SporosVmBin.O_fromr, (.ofInt (.ld 1 15 32)), 4) ∈ vmTableChunk 17),
      exec_ld h15 (rd_ctx_auxLen h), hrip]; rfl
  let s1 := { s.set 1 (some ls.aux.length) with rip := 4199683 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_1 : s1.reg 1 = some ls.aux.length := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and ls.aux.length ls.aux.length).2, rip := 4199686 } := by
    rw [step_at_vc 17 rfl (by decide : (4199683, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 17),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.aux.length ls.aux.length).2, rip := 4199686 }
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  unfold OpPostOutcome
  cases haux : ls.aux with
  | nil =>
    have hlstep : SporosProvLen.lstep pr ls = .fail .auxUnderflow := by
      simp [SporosProvLen.lstep, hrs, hfetch, haux]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := errEntry .auxUnderflow } := by
      rw [step_at_vc 17 rfl (by decide : (4199686, (.ofInt (.jcc 4 4201065)), 6) ∈ vmTableChunk 17),
        exec_jcc (cond_aluF_and_e ls.aux.length ls.aux.length)]
      simp [s2, haux]; rfl
    exact lift_fail_outcome (Reaches.trans hrch2 (Reaches.step h03)) rfl
      (L_fail_from_vc_spec (e := .auxUnderflow) rfl (vc.wf_step_misc h rfl hr1 hs2_4 hs2_15))
  | cons v r =>
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with aux := r, ds := SporosProvLen.ltag ls.pc v :: ls.ds,
                        rs := { f with pc := f.pc + 1 } :: rest } := by
      simp [SporosProvLen.lstep, hrs, hfetch, haux, SporosProvLen.LSt.setTop]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199692 } := by
      rw [step_at_vc 17 rfl (by decide : (4199686, (.ofInt (.jcc 4 4201065)), 6) ∈ vmTableChunk 17),
        exec_jcc (cond_aluF_and_e ls.aux.length ls.aux.length)]
      simp [s2, haux]
    let s3 := { s2 with rip := 4199692 }
    have ⟨_, ⟨_, _, haux_len, _, haux_hi, haux_rel, haux_le⟩, _⟩ := vc.arena_bounds h
    have hW : X86Env.W = 2 ^ 64 := rfl
    have haux_ltW : ls.aux.length * 16 < X86Env.W := by simp only [Region.hi, haux_len] at haux_hi; omega
    have hlen_eq : ls.aux.length = r.length + 1 := by simp [haux]
    have hsub1 : (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).1 = r.length := by
      rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val,
        show ls.aux.length + X86Env.W - 1 = r.length + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have hs3_1 : s3.reg 1 = some ls.aux.length := hs1_1
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).2, rip := 4199696 } := by
      rw [step_at_vc 17 rfl (by decide : (4199692, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 17),
        exec_aluI hs3_1]; dsimp only; rw [hsub1]
    let s4 := { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).2, rip := 4199696 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some r.length := reg_set_self_ok' hr1 (by decide) _
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4
      (by change (s3.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4)
      (by change (s3.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15)
    obtain ⟨s5, h05, hrip5, hk5, hregs5, _, hwf5⟩ :=
      wf_dec_auxLen (step_at_vc 17 rfl (by decide : (4199696, (.ofInt (.st 15 32 1)), 4) ∈ vmTableChunk 17))
        hwf4 haux hs4_1
    have hrch5 : Reaches vmImage s s5 :=
      Reaches.trans hrch2 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step h05)))
    have hvW : v.w < X86Env.W :=
      ((vc.code_flat h).2.2.2.2.2.1 v (by rw [haux]; exact List.mem_cons_self ..)).1
    have haux_rel' : PairListRel vc.auxReg.data (r.length + 1) (v :: r) := by
      simpa [hlen_eq, haux] using haux_rel
    exact lift_next_outcome hrch5 hk5
      (aux_top_to_ds_adv_spec (a := 4199700) (chk := 17) hrip5 hwf5
        (oracleAdmits_of_reaches hor hrch5) hrs hpcW
        (by simp only [X86Int.St.reg, hregs5]; exact hs4_1) haux_rel' hvW (by omega)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide))

theorem O_rfetch_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = instrEntry .rfetch) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .rfetch)
    (hpcW : f.pc + 1 < X86Env.W)
    (_h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair .rfetch).2) :
    OpPostOutcome pr ls s := by
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.aux.length) with rip := 4199776 } := by
    rw [step_at_vc 17 hrip (by decide : (SporosVmBin.O_rfetch, (.ofInt (.ld 1 15 32)), 4) ∈ vmTableChunk 17),
      exec_ld h15 (rd_ctx_auxLen h), hrip]; rfl
  let s1 := { s.set 1 (some ls.aux.length) with rip := 4199776 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_1 : s1.reg 1 = some ls.aux.length := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and ls.aux.length ls.aux.length).2, rip := 4199779 } := by
    rw [step_at_vc 17 rfl (by decide : (4199776, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 17),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.aux.length ls.aux.length).2, rip := 4199779 }
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  unfold OpPostOutcome
  cases haux : ls.aux with
  | nil =>
    have hlstep : SporosProvLen.lstep pr ls = .fail .auxUnderflow := by
      simp [SporosProvLen.lstep, hrs, hfetch, haux]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := errEntry .auxUnderflow } := by
      rw [step_at_vc 17 rfl (by decide : (4199779, (.ofInt (.jcc 4 4201065)), 6) ∈ vmTableChunk 17),
        exec_jcc (cond_aluF_and_e ls.aux.length ls.aux.length)]
      simp [s2, haux]; rfl
    exact lift_fail_outcome (Reaches.trans hrch2 (Reaches.step h03)) rfl
      (L_fail_from_vc_spec (e := .auxUnderflow) rfl (vc.wf_step_misc h rfl hr1 hs2_4 hs2_15))
  | cons v r =>
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with ds := SporosProvLen.ltag ls.pc v :: ls.ds, rs := { f with pc := f.pc + 1 } :: rest } := by
      simp [SporosProvLen.lstep, hrs, hfetch, haux, SporosProvLen.LSt.setTop]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199785 } := by
      rw [step_at_vc 17 rfl (by decide : (4199779, (.ofInt (.jcc 4 4201065)), 6) ∈ vmTableChunk 17),
        exec_jcc (cond_aluF_and_e ls.aux.length ls.aux.length)]
      simp [s2, haux]
    let s3 := { s2 with rip := 4199785 }
    have ⟨_, ⟨_, _, haux_len, _, haux_hi, haux_rel, haux_le⟩, _⟩ := vc.arena_bounds h
    have hW : X86Env.W = 2 ^ 64 := rfl
    have haux_ltW : ls.aux.length * 16 < X86Env.W := by simp only [Region.hi, haux_len] at haux_hi; omega
    have hlen_eq : ls.aux.length = r.length + 1 := by simp [haux]
    have hsub1 : (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).1 = r.length := by
      rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val,
        show ls.aux.length + X86Env.W - 1 = r.length + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have hs3_1 : s3.reg 1 = some ls.aux.length := hs1_1
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).2, rip := 4199789 } := by
      rw [step_at_vc 17 rfl (by decide : (4199785, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 17),
        exec_aluI hs3_1]; dsimp only; rw [hsub1]
    let s4 := { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub ls.aux.length (1 % X86Env.W)).2, rip := 4199789 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some r.length := reg_set_self_ok' hr1 (by decide) _
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4
      (by change (s3.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4)
      (by change (s3.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15)
    have hrch4 : Reaches vmImage s s4 :=
      Reaches.trans hrch2 (Reaches.step_trans h03 (Reaches.step h04))
    have hvW : v.w < X86Env.W :=
      ((vc.code_flat h).2.2.2.2.2.1 v (by rw [haux]; exact List.mem_cons_self ..)).1
    have haux_rel' : PairListRel vc.auxReg.data (r.length + 1) (v :: r) := by
      simpa [hlen_eq, haux] using haux_rel
    exact lift_next_outcome hrch4 rfl
      (aux_top_to_ds_adv_spec (a := 4199789) (chk := 17) rfl hwf4
        (oracleAdmits_of_reaches hor hrch4) hrs hpcW hs4_1 haux_rel' hvW (by omega)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide))

end SporosVmOpsData
