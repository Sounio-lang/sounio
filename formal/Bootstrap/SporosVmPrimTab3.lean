import Bootstrap.SporosVmPrimTab2

/-!
# Native x86-64 VM Table Insertion Primitive (`P_tput`)

Simulation proof over `X86Vm` for `P_tput` (primitive 45), handling bounds
checking (`L_fail_badTable`), optional `kv` arena growth via `fn_mremap`, and
dispatch to `tp_room_spec`.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimTab

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

attribute [local instance] instInhabitedLVal instInhabitedLFrame

/-! ### `P_tput` (45) -/

theorem P_tput_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_tput) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w) :
    PrimPostOutcome pr ls f rest s .tput := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  let v := ls.ds[0]!
  let k := ls.ds[1]!
  let tb := ls.ds[2]!
  obtain ⟨⟨htb_rw, _, _, _, _⟩, ⟨hkv_rw, hkv_pos, hkv_len, hkv_pg, hkv_hi, hkv_le⟩,
    ⟨_, hsl136, _, hsl152, hsl160, hsl168⟩, htabs_rel, _⟩ := wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hrd136 : rd64 s.mem (ea ctxBase 136) = some ls.tabs.length := vc.rd64_ctx_of_eq h (by decide) hsl136
  have hrd152 : rd64 s.mem (ea ctxBase 152) = some vc.kvReg.base := vc.rd64_ctx_of_eq h (by decide) hsl152
  have hrd160 : rd64 s.mem (ea ctxBase 160) = some vc.kvLen := vc.rd64_ctx_of_eq h (by decide) hsl160
  have hrd168 : rd64 s.mem (ea ctxBase 168) = some vc.kvCap := vc.rd64_ctx_of_eq h (by decide) hsl168
  have htake : ls.ds.take 3 = [v, k, tb] := by
    match hd : ls.ds with
    | a :: b :: c :: _ => simp [v, k, tb, hd, List.take]
    | [] | [_] | [_, _] => simp [hd] at hle
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some tb.w) with rip := 4198402 } := by
    rw [step_at_vc 13 hrip (by decide : (SporosVmBin.P_tput, (.ofInt (.ld 12 15 328)), 7) ∈ vmTableChunk 13),
      exec_ld h15 h328, hrip]; rfl
  let s1 := { s.set 12 (some tb.w) with rip := 4198402 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_12 : s1.reg 12 = some tb.w := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 12 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.tabs.length) with rip := 4198409 } := by
    rw [step_at_vc 13 rfl (by decide : (4198402, (.ofInt (.ld 1 15 136)), 7) ∈ vmTableChunk 13),
      exec_ld hs1_15 hrd136]
  let s2 := { s1.set 1 (some ls.tabs.length) with rip := 4198409 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some ls.tabs.length := reg_set_self_ok' hr1 (by decide) _
  have hs2_12 : s2.reg 12 = some tb.w := by
    change (s1.set 1 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs1_12
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 12 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub tb.w ls.tabs.length).2, rip := 4198412 } := by
    rw [step_at_vc 13 rfl (by decide : (4198409, (.ofInt (.cmp 12 1)), 3) ∈ vmTableChunk 13),
      exec_cmp hs2_12 hs2_1]
  let s3 := { s2 with fl := (X86Int.aluF .sub tb.w ls.tabs.length).2, rip := 4198412 }
  by_cases htb : tb.w < ls.tabs.length
  · have hnotae : ¬ ls.tabs.length ≤ tb.w := by omega
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198418 } := by
      rw [step_at_vc 13 rfl (by decide : (4198412, (.ofInt (.jcc 3 4201184)), 6) ∈ vmTableChunk 13),
        exec_jcc (cond_aluF_sub_ae tb.w ls.tabs.length)]
      simp [s3, hnotae]
    let s4 := { s3 with rip := 4198418 }
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some vc.kvLen) with rip := 4198425 } := by
      rw [step_at_vc 13 rfl (by decide : (4198418, (.ofInt (.ld 0 15 160)), 7) ∈ vmTableChunk 13),
        exec_ld (show s4.reg 15 = some ctxBase from hs2_15) hrd160]
    let s5 := { s4.set 0 (some vc.kvLen) with rip := 4198425 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_0 : s5.reg 0 = some vc.kvLen := reg_set_self_ok' hr2 (by decide) _
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some vc.kvCap) with rip := 4198432 } := by
      rw [step_at_vc 13 rfl (by decide : (4198425, (.ofInt (.ld 1 15 168)), 7) ∈ vmTableChunk 13),
        exec_ld hs5_15 hrd168]
    let s6 := { s5.set 1 (some vc.kvCap) with rip := 4198432 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_1 : s6.reg 1 = some vc.kvCap := reg_set_self_ok' hr5 (by decide) _
    have hs6_0 : s6.reg 0 = some vc.kvLen := by
      change (s5.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]; exact hs5_0
    have hs6_12 : s6.reg 12 = some tb.w := by
      change (s5.set 1 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
      change (s4.set 0 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs2_12
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change (s5.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s4.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6 with fl := (X86Int.aluF .sub vc.kvLen vc.kvCap).2, rip := 4198435 } := by
      rw [step_at_vc 13 rfl (by decide : (4198432, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 13),
        exec_cmp hs6_0 hs6_1]
    let s7 := { s6 with fl := (X86Int.aluF .sub vc.kvLen vc.kvCap).2, rip := 4198435 }
    by_cases hroom : vc.kvLen < vc.kvCap
    · have h08 : X86Vm.step vmImage s7 = .ok { s7 with rip := 4198488 } := by
        rw [step_at_vc 13 rfl (by decide : (4198435, (.ofInt (.jcc 2 4198488)), 2) ∈ vmTableChunk 13),
          exec_jcc (cond_aluF_sub_b vc.kvLen vc.kvCap)]
        simp [s7, hroom]
      let s8 := { s7 with rip := 4198488 }
      have hrch8 : Reaches vmImage s s8 :=
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
          (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08)))))))
      have hwf8 : vc.wf pr ls s8 := vc.wf_step_misc h rfl hr6 hs6_4 hs6_15
      exact lift_primPostOutcome hrch8 rfl
        (tp_room_spec rfl hle hwf8 (oracleAdmits_of_reaches hor hrch8) hrs hpcW htb hroom hs6_12
          h272 h280 h288 h296 h304 h312)
    · have h08 : X86Vm.step vmImage s7 = .ok { s7 with rip := 4198437 } := by
        rw [step_at_vc 13 rfl (by decide : (4198435, (.ofInt (.jcc 2 4198488)), 2) ∈ vmTableChunk 13),
          exec_jcc (cond_aluF_sub_b vc.kvLen vc.kvCap)]
        simp [s7, hroom]
      let s8 := { s7 with rip := 4198437 }
      have h09 : X86Vm.step vmImage s8 = .ok { s8.set 7 (some vc.kvReg.base) with rip := 4198444 } := by
        rw [step_at_vc 13 rfl (by decide : (4198437, (.ofInt (.ld 7 15 152)), 7) ∈ vmTableChunk 13),
          exec_ld (show s8.reg 15 = some ctxBase from hs6_15) hrd152]
      let s9 := { s8.set 7 (some vc.kvReg.base) with rip := 4198444 }
      have hr9 : RegsOk s9 := regsOk_set hr6 (by decide) _
      have hs9_7 : s9.reg 7 = some vc.kvReg.base := reg_set_self_ok' hr6 (by decide) _
      have hs9_1 : s9.reg 1 = some vc.kvCap := by
        change (s8.set 7 _).reg 1 = some vc.kvCap; rw [reg_set_ne (by decide)]; exact hs6_1
      have h10 : X86Vm.step vmImage s9 = .ok { s9.set 6 (some vc.kvCap) with rip := 4198447 } := by
        rw [step_at_vc 13 rfl (by decide : (4198444, (.ofInt (.mov 6 1)), 3) ∈ vmTableChunk 13), exec_mov hs9_1]
      let s10 := { s9.set 6 (some vc.kvCap) with rip := 4198447 }
      have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
      have hs10_6 : s10.reg 6 = some vc.kvCap := reg_set_self_ok' hr9 (by decide) _
      have hcap64W : vc.kvCap * 64 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
      have h11 : X86Vm.step vmImage s10 =
          .ok { s10.set 6 (some (vc.kvCap * 32)) with fl := X86Int.Flags.undef, rip := 4198451 } := by
        rw [step_at_vc 13 rfl (by decide : (4198447, (.ofInt (.shlI 6 5)), 4) ∈ vmTableChunk 13),
          exec_shlI_lt (c := 5) hs10_6 (by rw [show 2 ^ (5 % 64) = 32 from by decide]; omega)]
      let s11 := { s10.set 6 (some (vc.kvCap * 32)) with fl := X86Int.Flags.undef, rip := 4198451 }
      have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
      have hs11_6 : s11.reg 6 = some (vc.kvCap * 32) := reg_set_self_ok' hr10 (by decide) _
      have h12 : X86Vm.step vmImage s11 = .ok { s11.set 2 (some (vc.kvCap * 32)) with rip := 4198454 } := by
        rw [step_at_vc 13 rfl (by decide : (4198451, (.ofInt (.mov 2 6)), 3) ∈ vmTableChunk 13), exec_mov hs11_6]
      let s12 := { s11.set 2 (some (vc.kvCap * 32)) with rip := 4198454 }
      have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) _
      have hs12_2 : s12.reg 2 = some (vc.kvCap * 32) := reg_set_self_ok' hr11 (by decide) _
      have h13 : X86Vm.step vmImage s12 =
          .ok { s12.set 2 (some (vc.kvCap * 64)) with fl := X86Int.Flags.undef, rip := 4198458 } := by
        rw [step_at_vc 13 rfl (by decide : (4198454, (.ofInt (.shlI 2 1)), 4) ∈ vmTableChunk 13),
          exec_shlI_1_lt hs12_2 (by omega)]
        simp [s12, show vc.kvCap * 32 * 2 = vc.kvCap * 64 by omega]
      let s13 := { s12.set 2 (some (vc.kvCap * 64)) with fl := X86Int.Flags.undef, rip := 4198458 }
      have hr13 : RegsOk s13 := regsOk_set hr12' (by decide) _
      have hs13_2 : s13.reg 2 = some (vc.kvCap * 64) := reg_set_self_ok' hr12' (by decide) _
      have hs13_6 : s13.reg 6 = some (vc.kvCap * 32) := by
        change (s12.set 2 _).reg 6 = some (vc.kvCap * 32); rw [reg_set_ne (by decide)]
        change (s11.set 2 _).reg 6 = some (vc.kvCap * 32); rw [reg_set_ne (by decide)]; exact hs11_6
      have hs13_7 : s13.reg 7 = some vc.kvReg.base := by
        change (s12.set 2 _).reg 7 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
        change (s11.set 2 _).reg 7 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
        change (s10.set 6 _).reg 7 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
        change (s9.set 6 _).reg 7 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hs9_7
      have hs13_12 : s13.reg 12 = some tb.w := by
        change (s12.set 2 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
        change (s11.set 2 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
        change (s10.set 6 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
        change (s9.set 6 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
        change (s8.set 7 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs6_12
      have hs13_4 : s13.reg 4 = some vc.sp := by
        change (s12.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s11.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s10.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s9.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s8.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs6_4
      have hs13_15 : s13.reg 15 = some ctxBase := by
        change (s12.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s11.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s10.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s9.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s8.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs6_15
      have hwf13 : vc.wf pr ls s13 := vc.wf_step_misc h rfl hr13 hs13_4 hs13_15
      obtain ⟨s14, h14, hrip14, hk14, _, hcm14, _, _, hr14, hs14_4, hs14_15, hs14_oth,
          hstk14, hstk14_b, hstk14_rw, hstk14_len, h16, hsp14W, hstk14_lo, hstk14_hi, hrd_ret, hoth14⟩ :=
        step_call_stk (vc := vc) (pr := pr) (ls := ls) (s := s13) (t := SporosVmBin.fn_mremap)
          (step_at_vc 13 rfl (by decide : (4198458, (.ofInt (.call 4201363)), 5) ∈ vmTableChunk 13))
          hwf13 (by change 4198458 + 5 < X86Env.W; omega)
      have hrch14 : Reaches vmImage s s14 :=
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
          (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
            (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step_trans h12
              (Reaches.step_trans h13 (Reaches.step h14)))))))))))))
      have hne_stk_kv : vc.stk.base ≠ vc.kvReg.base :=
        vc.ne_bases h (i := 1) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hne_ctx_kv : vc.ctxReg.base ≠ vc.kvReg.base :=
        vc.ne_bases h (i := 2) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hne_tb_kv : vc.tabsReg.base ≠ vc.kvReg.base :=
        vc.ne_bases h (i := 12) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hne_stk_ctx : vc.stk.base ≠ vc.ctxReg.base :=
        vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hne_stk_tb : vc.stk.base ≠ vc.tabsReg.base :=
        vc.ne_bases h (i := 1) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hne_ctx_tb : vc.ctxReg.base ≠ vc.tabsReg.base :=
        vc.ne_bases h (i := 2) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
      have hkv14 : vc.kvReg ∈ s14.mem := hoth14 vc.kvReg (vc.mem_kv hwf13) (Ne.symm hne_stk_kv)
      obtain ⟨idx, hfind14⟩ := mem_find_of_mem s14.mem vc.kvReg vc.kvReg.base 0 hcm14.2.2 hkv14
        (Nat.le_refl _) (by simp only [Region.hi]; omega)
      rw [Nat.zero_add] at hfind14
      have hpg32 : X86Env.pages (vc.kvCap * 32) = vc.kvCap * 32 := pages_of_dvd hkv_pg
      have hdvd64 : (vc.kvCap * 64) % X86Env.page = 0 := by
        rw [show vc.kvCap * 64 = 2 * (vc.kvCap * 32) by omega, Nat.mul_mod, hkv_pg]; simp
      have hpg64 : X86Env.pages (vc.kvCap * 64) = vc.kvCap * 64 := pages_of_dvd hdvd64
      have hmremap := call_mremap_oracle_spec hrip14 hr14 (oracleAdmits_of_reaches hor hrch14)
        hs14_4 (by omega) hsp14W hcm14.2 hstk14 hstk14_rw (by rw [hstk14_b]; exact hne_stk_kv)
        (by omega) hstk14_hi hrd_ret
        (by rw [hs14_oth 7 (by decide)]; exact hs13_7)
        (by rw [hs14_oth 6 (by decide)]; exact hs13_6)
        (by rw [hs14_oth 2 (by decide)]; exact hs13_2)
        hfind14 rfl (by rw [hkv_len, hpg32]) hkv_rw (by rw [hkv_len, hpg64]; omega)
      rcases hmremap with ⟨a, s15, hrch15, _, _, _, ha_lim, hfree, hrip15, hk15, hmem15, hmwf15,
          hr15, hs15_0, hs15_4, _, hs15_oth⟩ | ⟨hhit, hends⟩
      · have hsb := vc.sp_bounds h
        have hs15_4' : s15.reg 4 = some vc.sp := by
          simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hs15_4
        have hs15_15 : s15.reg 15 = some ctxBase := by
          rw [hs15_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide)]; exact hs14_15
        have hs15_12 : s15.reg 12 = some tb.w := by
          rw [hs15_oth 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide), hs14_oth 12 (by decide)]; exact hs13_12
        have hk15_out : s15.k.out = s.k.out := by
          have hk13 : s13.k = s.k := rfl
          rw [hk15, hk14, hk13]; unfold X86Env.Kern.next; cases s.k.oracle <;> rfl
        have hget14 : s14.mem[idx]? = some vc.kvReg := by
          simpa using (mem_find_idx_lt s14.mem vc.kvReg.base 0 idx vc.kvReg hfind14).2.2
        rw [hpg64] at hfree ha_lim
        let stk14 : Region := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (4198458 + 5)) }
        have hstk14_hi_eq : stk14.hi = vc.stk.hi := by simp [stk14, Region.hi, writeBytes_length]
        obtain ⟨_, hfree_all⟩ := mremap_find_after_stack_write (n := vc.kvCap * 64)
          h.1.2 (vc.mem_stk h) hstk14_b hstk14_hi_eq hne_stk_kv hstk14 hoth14 hget14 rfl hfree
          (by omega) h.2.2.1
        have hoth15 : ∀ u ∈ s14.mem, u.base ≠ vc.kvReg.base → u ∈ s15.mem := by
          intro u hu hne
          rw [hmem15]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base _ _ hget14 hu hne)
        let newKv : Region := ⟨a, vc.kvReg.data ++ List.replicate (vc.kvCap * 32) 0, true⟩
        have hnewKv_mem : newKv ∈ s15.mem := by
          rw [hmem15, hpg64, hkv_len, show vc.kvCap * 64 - vc.kvCap * 32 = vc.kvCap * 32 by omega]
          exact List.mem_append_right _ (List.mem_cons_self ..)
        have hctx15_mem : vc.ctxReg ∈ s15.mem :=
          hoth15 vc.ctxReg (hoth14 vc.ctxReg (vc.mem_ctx hwf13) (Ne.symm hne_stk_ctx)) hne_ctx_kv
        have htb15_mem : vc.tabsReg ∈ s15.mem :=
          hoth15 vc.tabsReg (hoth14 vc.tabsReg (vc.mem_tabs hwf13) (Ne.symm hne_stk_tb)) hne_tb_kv
        have hstk15_mem : stk14 ∈ s15.mem := hoth15 stk14 hstk14 (by rw [hstk14_b]; exact hne_stk_kv)
        have ha_ne_ctx : a ≠ vc.ctxReg.base :=
          mem_free_ne_base (by omega) (h.2.2.1 vc.ctxReg (vc.mem_ctx h)) (hfree_all vc.ctxReg (vc.mem_ctx h) hne_ctx_kv)
        have hea152 : ea ctxBase 152 = ctxBase + 152 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
        have hea168 : ea ctxBase 168 = ctxBase + 168 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
        obtain ⟨_, m16, hwr16, _, _, hmwf16, hctx16_mem, hoth16⟩ :=
          wr64_region hmwf15 hctx15_mem hctx_b.2.1 (a := ctxBase + 152)
            (by rw [hctx_b.1]; omega) (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega) (v := a)
        have hwr16' : wr64 s15.mem (ea ctxBase 152) a = some m16 := by rw [hea152]; exact hwr16
        have hrip15' : s15.rip = 4198463 := hrip15
        have h16s : X86Vm.step vmImage s15 = .ok { s15 with mem := m16, rip := 4198470 } := by
          rw [step_at_vc 13 hrip15' (by decide : (4198463, (.ofInt (.st 15 152 0)), 7) ∈ vmTableChunk 13),
            exec_st hs15_15 hs15_0 hwr16', hrip15']
        let s16 := { s15 with mem := m16, rip := 4198470 }
        have hsub152 : ctxBase + 152 - vc.ctxReg.base = 152 := by rw [hctx_b.1]; omega
        rw [hsub152] at hctx16_mem
        let ctx16 : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 152 (bytes8 a) }
        have hrd168_16 : rd64 s16.mem (ea ctxBase 168) = some vc.kvCap := by
          rw [hea168]
          refine rd64_of_mem hmwf16 hctx16_mem (by dsimp [ctx16]; rw [hctx_b.1]; omega)
            (by simp only [Region.hi, hctx_b.1, writeBytes_length]; have := hctx_b.2.2.1; omega) ?_
          dsimp only [ctx16]
          rw [show ctxBase + 168 - vc.ctxReg.base = 168 by rw [hctx_b.1]; omega,
              writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
                (by have := hctx_b.2.2.1; omega), hsl168]
        have h17 : X86Vm.step vmImage s16 = .ok { s16.set 1 (some vc.kvCap) with rip := 4198477 } := by
          rw [step_at_vc 13 rfl (by decide : (4198470, (.ofInt (.ld 1 15 168)), 7) ∈ vmTableChunk 13),
            exec_ld (show s16.reg 15 = some ctxBase from hs15_15) hrd168_16]
        let s17 := { s16.set 1 (some vc.kvCap) with rip := 4198477 }
        have hr17 : RegsOk s17 := regsOk_set hr15 (by decide) _
        have hs17_1 : s17.reg 1 = some vc.kvCap := reg_set_self_ok' hr15 (by decide) _
        have h18 : X86Vm.step vmImage s17 =
            .ok { s17.set 1 (some (vc.kvCap * 2)) with fl := X86Int.Flags.undef, rip := 4198481 } := by
          rw [step_at_vc 13 rfl (by decide : (4198477, (.ofInt (.shlI 1 1)), 4) ∈ vmTableChunk 13),
            exec_shlI_1_lt hs17_1 (by omega)]
        let s18 := { s17.set 1 (some (vc.kvCap * 2)) with fl := X86Int.Flags.undef, rip := 4198481 }
        have hr18 : RegsOk s18 := regsOk_set hr17 (by decide) _
        have hs18_1 : s18.reg 1 = some (vc.kvCap * 2) := reg_set_self_ok' hr17 (by decide) _
        have hs18_4 : s18.reg 4 = some vc.sp := by
          change (s17.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s16.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs15_4'
        have hs18_12 : s18.reg 12 = some tb.w := by
          change (s17.set 1 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]
          change (s16.set 1 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs15_12
        have hs18_15 : s18.reg 15 = some ctxBase := by
          change (s17.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s16.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs15_15
        obtain ⟨_, m19, hwr19, _, _, hmwf19, hctx19_mem, hoth19⟩ :=
          wr64_region hmwf16 hctx16_mem hctx_b.2.1 (a := ctxBase + 168)
            (by dsimp [ctx16]; rw [hctx_b.1]; omega)
            (by simp only [Region.hi, hctx_b.1, writeBytes_length]; have := hctx_b.2.2.1; omega)
            (v := vc.kvCap * 2)
        have hwr19' : wr64 s18.mem (ea ctxBase 168) (vc.kvCap * 2) = some m19 := by rw [hea168]; exact hwr19
        have h19 : X86Vm.step vmImage s18 = .ok { s18 with mem := m19, rip := 4198488 } := by
          rw [step_at_vc 13 rfl (by decide : (4198481, (.ofInt (.st 15 168 1)), 7) ∈ vmTableChunk 13),
            exec_st hs18_15 hs18_1 hwr19']
        let s19 := { s18 with mem := m19, rip := 4198488 }
        have hrch_tot : Reaches vmImage s s19 :=
          Reaches.trans hrch14 (Reaches.trans hrch15
            (Reaches.step_trans h16s (Reaches.step_trans h17 (Reaches.step_trans h18 (Reaches.step h19)))))
        have ⟨hbl19, hpos19⟩ := reaches_bytesLt_len_pos h.2.1 h.2.2.1 hrch_tot
        have hsub168 : ctxBase + 168 - ctx16.base = 168 := by dsimp [ctx16]; rw [hctx_b.1]; omega
        rw [hsub168] at hctx19_mem
        let ctx19 : Region := { ctx16 with data := writeBytes ctx16.data 168 (bytes8 (vc.kvCap * 2)) }
        have hnewKv19 : newKv ∈ m19 := hoth19 newKv (hoth16 newKv hnewKv_mem ha_ne_ctx) ha_ne_ctx
        have htb19 : vc.tabsReg ∈ m19 := hoth19 vc.tabsReg (hoth16 vc.tabsReg htb15_mem (Ne.symm hne_ctx_tb)) (Ne.symm hne_ctx_tb)
        have hstk19 : stk14 ∈ m19 := hoth19 stk14 (hoth16 stk14 hstk15_mem (by rw [hstk14_b]; exact hne_stk_ctx))
          (by rw [hstk14_b]; exact hne_stk_ctx)
        have hc152' : leVal ((ctx19.data.drop 152).take 8) = newKv.base := by
          dsimp only [ctx19, ctx16, newKv]
          rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide))
                (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
              leVal_writeBytes_same _ _ _ (by have := hctx_b.2.2.1; omega), Nat.mod_eq_of_lt (by omega)]
        have hc160' : leVal ((ctx19.data.drop 160).take 8) = vc.kvLen := by
          dsimp only [ctx19, ctx16]
          rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide))
                (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
              writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
                (by have := hctx_b.2.2.1; omega), hsl160]
        have hc168' : leVal ((ctx19.data.drop 168).take 8) = vc.kvCap * 2 := by
          dsimp only [ctx19, ctx16]
          rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
              Nat.mod_eq_of_lt (by omega)]
        have hctx_oth' : ∀ j, (j < 152 ∨ 176 ≤ j) → ctx19.data[j]? = vc.ctxReg.data[j]? := by
          intro j hj; dsimp only [ctx19, ctx16]
          rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
              writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
        obtain ⟨hwf19, hpres19⟩ := wf_update_kv_tabs h hmwf19 hbl19 hpos19 hr18 hs18_4 hs18_15
          htb19 rfl htb_rw rfl
          hnewKv19 rfl (by omega)
          (by simp [newKv, List.length_append, List.length_replicate, hkv_len]; omega)
          (by rw [show vc.kvCap * 2 * 32 = vc.kvCap * 64 by omega]; exact hdvd64)
          (by simp [newKv, Region.hi, List.length_append, List.length_replicate, hkv_len]; omega) (by omega)
          hctx19_mem rfl rfl (by dsimp [ctx19, ctx16]; rw [writeBytes_length, writeBytes_length])
          hc152' hc160' hc168' hctx_oth'
          hstk19 hstk14_b rfl hstk14_len
          (fun u hu hne => Ne.symm (mem_free_ne_base (by omega) (h.2.2.1 u hu) (hfree_all u hu hne)))
          (fun u hu h1 _ h3 h4 => hoth19 u (hoth16 u (hoth15 u (hoth14 u hu h4) h1) h3) h3)
          rfl (tabsRel_extend_kv_zero htabs_rel (by omega)) (vc.code_flat h).2.2.2.2.2.2.2.2.1
        have hfin := tp_room_spec rfl hle hwf19 (oracleAdmits_of_reaches hor hrch_tot) hrs hpcW htb
          (by dsimp; omega) hs18_12
          (hpres19 272 _ (Or.inr (by decide)) (by decide) h272)
          (hpres19 280 3 (Or.inr (by decide)) (by decide) h280)
          (hpres19 288 0 (Or.inr (by decide)) (by decide) h288)
          (hpres19 296 v.w (Or.inr (by decide)) (by decide) h296)
          (hpres19 304 v.plen (Or.inr (by decide)) (by decide) h304)
          (hpres19 312 k.w (Or.inr (by decide)) (by decide) h312)
        unfold PrimPostOutcome at hfin ⊢
        simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, dif_pos htb] at hfin ⊢
        exact lift_next_outcome_out hrch_tot hk15_out hfin
      · unfold PrimPostOutcome
        simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, dif_pos htb]
        exact Or.inr ⟨hitsNomem_of_reaches hrch14 hhit, Ends.of_reaches hrch14 hends⟩
  · have hae : ls.tabs.length ≤ tb.w := by omega
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := errEntry .badTable } := by
      rw [step_at_vc 13 rfl (by decide : (4198412, (.ofInt (.jcc 3 4201184)), 6) ∈ vmTableChunk 13),
        exec_jcc (cond_aluF_sub_ae tb.w ls.tabs.length)]
      simp [s3, hae, errEntry, SporosVmBin.L_fail_badTable]
    let s4 := { s3 with rip := errEntry .badTable }
    have hrch4 : Reaches vmImage s s4 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr2 hs2_4 hs2_15
    unfold PrimPostOutcome
    simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, dif_neg htb]
    exact lift_fail_outcome hrch4 rfl (L_fail_from_vc_spec (e := .badTable) rfl hwf4)

end SporosVmPrimTab
