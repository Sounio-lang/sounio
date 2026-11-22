import Bootstrap.SporosVmPrimTabBase

/-!
# Native x86-64 VM Table Allocation Primitive (`P_tnew`)

Simulation proof over `X86Vm` for `P_tnew` (primitive 44), including both the
in-place fast path (`tn_room_spec`) and the `fn_mremap` growth path.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimTab

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

attribute [local instance] instInhabitedLVal instInhabitedLFrame

/-! ### `P_tnew` (44) -/

private theorem tn_room_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = 4198351) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hroom : ls.tabs.length < vc.tabsCap)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .tnew := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  obtain ⟨⟨htb_rw, htb_pos, htb_len, htb_pg, htb_hi⟩, _, ⟨hsl128, hsl136, hsl144, _, _, _⟩, htabs_rel, _⟩ :=
    wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hrd136 : rd64 s.mem (ea ctxBase 136) = some ls.tabs.length := vc.rd64_ctx_of_eq h (by decide) hsl136
  have hrd128 : rd64 s.mem (ea ctxBase 128) = some vc.tabsReg.base := vc.rd64_ctx_of_eq h (by decide) hsl128
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.tabs.length) with rip := 4198358 } := by
    rw [step_at_vc 13 hrip (by decide : (4198351, (.ofInt (.ld 0 15 136)), 7) ∈ vmTableChunk 13),
      exec_ld h15 hrd136, hrip]
  let s1 := { s.set 0 (some ls.tabs.length) with rip := 4198358 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_0 : s1.reg 0 = some ls.tabs.length := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some vc.tabsReg.base) with rip := 4198365 } := by
    rw [step_at_vc 13 rfl (by decide : (4198358, (.ofInt (.ld 2 15 128)), 7) ∈ vmTableChunk 13),
      exec_ld hs1_15 hrd128]
  let s2 := { s1.set 2 (some vc.tabsReg.base) with rip := 4198365 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_2 : s2.reg 2 = some vc.tabsReg.base := reg_set_self_ok' hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some ls.tabs.length := by
    change (s1.set 2 _).reg 0 = some ls.tabs.length; rw [reg_set_ne (by decide)]; exact hs1_0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some 18446744073709551615) with rip := 4198375 } := by
    rw [step_at_vc 13 rfl (by decide : (4198365, (.ofInt (.movI 1 18446744073709551615)), 10) ∈ vmTableChunk 13),
      exec_movI_lt (by omega)]
  let s3 := { s2.set 1 (some 18446744073709551615) with rip := 4198375 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_1 : s3.reg 1 = some 18446744073709551615 := reg_set_self_ok' hr2 (by decide) _
  have hs3_2 : s3.reg 2 = some vc.tabsReg.base := by
    change (s2.set 1 _).reg 2 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]; exact hs2_2
  have hs3_0 : s3.reg 0 = some ls.tabs.length := by
    change (s2.set 1 _).reg 0 = some ls.tabs.length; rw [reg_set_ne (by decide)]; exact hs2_0
  have hea_tb : eaIdx vc.tabsReg.base ls.tabs.length 8 0 = vc.tabsReg.base + ls.tabs.length * 8 := by
    have hltW : vc.tabsReg.base + 8 * ls.tabs.length + 0 < X86Env.W := by
      simp only [Region.hi] at htb_hi; omega
    rw [eaIdx_add_eq hltW]; omega
  obtain ⟨_, m4, hwr4, _, _, hmwf4, htb4_mem, hoth4⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_tabs h) htb_rw (a := vc.tabsReg.base + ls.tabs.length * 8)
      (Nat.le_add_right _ _) (by simp only [Region.hi]; omega) (v := 18446744073709551615)
  have hwr4' : wrMem .w64 s3.mem (eaIdx vc.tabsReg.base ls.tabs.length 8 0) 18446744073709551615 = some m4 := by
    rw [hea_tb]; exact hwr4
  have h04 : X86Vm.step vmImage s3 = .ok { s3 with mem := m4, rip := 4198379 } := by
    rw [step_at_vc 13 rfl (by decide : (4198375, (.stIdx .w64 2 0 8 0 1), 4) ∈ vmTableChunk 13),
      exec_stIdx hs3_2 hs3_0 hs3_1 hwr4']
  let s4 := { s3 with mem := m4, rip := 4198379 }
  have hs4_0 : s4.reg 0 = some ls.tabs.length := hs3_0
  have hlen1W : ls.tabs.length + 1 < X86Env.W := by simp only [Region.hi] at htb_hi; omega
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 1 (some (ls.tabs.length + 1)) with rip := 4198383 } := by
    rw [step_at_vc 13 rfl (by decide : (4198379, (.ofInt (.lea 1 0 1)), 4) ∈ vmTableChunk 13),
      exec_lea hs4_0, ea_add_eq hlen1W]
  let s5 := { s4.set 1 (some (ls.tabs.length + 1)) with rip := 4198383 }
  have hr5 : RegsOk s5 := regsOk_set hr3 (by decide) _
  have hs5_1 : s5.reg 1 = some (ls.tabs.length + 1) := reg_set_self_ok' hr3 (by decide) _
  have hs5_0 : s5.reg 0 = some ls.tabs.length := by
    change (s4.set 1 _).reg 0 = some ls.tabs.length; rw [reg_set_ne (by decide)]; exact hs4_0
  have hs5_4 : s5.reg 4 = some vc.sp := by
    change (s4.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hne_ctx_tb : vc.ctxReg.base ≠ vc.tabsReg.base :=
    vc.ne_bases h (i := 2) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hctx4_mem : vc.ctxReg ∈ m4 := hoth4 vc.ctxReg (vc.mem_ctx h) hne_ctx_tb
  have hea136 : ea ctxBase 136 = ctxBase + 136 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
  obtain ⟨_, m6, hwr6, _, _, hmwf6, hctx6_mem, hoth6⟩ :=
    wr64_region hmwf4 hctx4_mem hctx_b.2.1 (a := ctxBase + 136)
      (by rw [hctx_b.1]; omega) (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega)
      (v := ls.tabs.length + 1)
  have hwr6' : wr64 s5.mem (ea ctxBase 136) (ls.tabs.length + 1) = some m6 := by rw [hea136]; exact hwr6
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with mem := m6, rip := 4198390 } := by
    rw [step_at_vc 13 rfl (by decide : (4198383, (.ofInt (.st 15 136 1)), 7) ∈ vmTableChunk 13),
      exec_st hs5_15 hs5_1 hwr6']
  let s6 := { s5 with mem := m6, rip := 4198390 }
  have hrch6 : Reaches vmImage s s6 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step h06)))))
  have ⟨hbl6, hpos6⟩ := reaches_bytesLt_len_pos h.2.1 h.2.2.1 hrch6
  let tabsReg' : Region := { vc.tabsReg with data := writeBytes vc.tabsReg.data (ls.tabs.length * 8) (bytes8 18446744073709551615) }
  let ctx' : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 136 (bytes8 (ls.tabs.length + 1)) }
  have hsub_tb : vc.tabsReg.base + ls.tabs.length * 8 - vc.tabsReg.base = ls.tabs.length * 8 := by omega
  have hsub_ctx : ctxBase + 136 - vc.ctxReg.base = 136 := by rw [hctx_b.1]; omega
  rw [hsub_tb] at htb4_mem; rw [hsub_ctx] at hctx6_mem
  have htb6_mem : tabsReg' ∈ m6 := hoth6 tabsReg' htb4_mem (Ne.symm hne_ctx_tb)
  have hstk6_mem : vc.stk ∈ m6 := by
    have hne1 : vc.stk.base ≠ vc.tabsReg.base :=
      vc.ne_bases h (i := 1) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hne2 : vc.stk.base ≠ vc.ctxReg.base :=
      vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    exact hoth6 vc.stk (hoth4 vc.stk (vc.mem_stk h) hne1) hne2
  have hc128' : leVal ((ctx'.data.drop 128).take 8) = tabsReg'.base := by
    dsimp only [ctx', tabsReg']
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by have := hctx_b.2.2.1; omega), hsl128]
  have hc136' : leVal ((ctx'.data.drop 136).take 8) = (ls.tabs ++ [[]]).length := by
    dsimp only [ctx']
    rw [leVal_writeBytes_same _ _ _ (by have := hctx_b.2.2.1; omega), Nat.mod_eq_of_lt hlen1W]
    simp
  have hc144' : leVal ((ctx'.data.drop 144).take 8) = vc.tabsCap := by
    dsimp only [ctx']
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
      (by have := hctx_b.2.2.1; omega), hsl144]
  have hctx_oth' : ∀ j, (j < 128 ∨ 152 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]? := by
    intro j hj; exact writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)
  have hwlt_tabs' : ∀ tb ∈ ls.tabs ++ [[]], ∀ kv ∈ tb, kv.2.w < Sporos.M ∧ kv.2.plen < Sporos.M := by
    intro tb htb kv hkv
    rw [List.mem_append, List.mem_singleton] at htb
    rcases htb with htb | rfl
    · exact (vc.code_flat h).2.2.2.2.2.2.2.2.1 tb htb kv hkv
    · cases hkv
  obtain ⟨hwf6, hpres6⟩ := wf_update_tabs h hmwf6 hbl6 hpos6 hr5 hs5_4 hs5_15
    htb6_mem htb_rw htb_pos (by dsimp [tabsReg']; rw [writeBytes_length]; exact htb_len) htb_pg
    (by simpa only [Region.hi, tabsReg', writeBytes_length] using htb_hi)
    hctx6_mem rfl rfl (writeBytes_length _ _ _) hc128' hc136' hc144' hctx_oth'
    hstk6_mem rfl rfl rfl (fun _ _ hne => hne)
    (fun u hu h1 h2 _ => hoth6 u (hoth4 u hu h1) h2)
    (tabsRel_append_empty htabs_rel (by omega)) (by simp; omega) hwlt_tabs'
  have hqW : SporosProvLen.lq (ls.ds.take 0) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hfin := jmp_mk1_finish_spec
    (step_at_vc 13 rfl (by decide : (4198390, (.ofInt (.jmp 4195210)), 5) ∈ vmTableChunk 13))
    hwf6 (oracleAdmits_of_reaches hor hrch6) hrs hpcW (Nat.zero_le _) hs5_0
    (hpres6 272 _ (Or.inr (by decide)) (by decide) h272) hqW
    (hpres6 280 0 (Or.inr (by decide)) (by decide) h280)
    (hpres6 288 1 (Or.inr (by decide)) (by decide) h288)
  have hres := lift_next_outcome hrch6 rfl hfin
  unfold PrimPostOutcome
  have hexec : SporosProvLen.lexecPrim .tnew (ls.ds.take Sporos.Prim.tnew.eff.1) ls =
      .ok [⟨Sporos.wrap ls.tabs.length, SporosProvLen.lq (ls.ds.take 0) ls.pc⟩]
        { ls with tabs := ls.tabs ++ [[]] } := by
    simp [Sporos.Prim.eff, SporosProvLen.lexecPrim]
  rw [hexec]
  simpa [Sporos.Prim.eff, Sporos.wrap, M_eq_W'] using hres

theorem P_tnew_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_tnew)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .tnew := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  obtain ⟨⟨htb_rw, htb_pos, htb_len, htb_pg, htb_hi⟩, _, ⟨hsl128, hsl136, hsl144, _, _, _⟩, htabs_rel, htabs_le⟩ :=
    wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hrd128 : rd64 s.mem (ea ctxBase 128) = some vc.tabsReg.base := vc.rd64_ctx_of_eq h (by decide) hsl128
  have hrd136 : rd64 s.mem (ea ctxBase 136) = some ls.tabs.length := vc.rd64_ctx_of_eq h (by decide) hsl136
  have hrd144 : rd64 s.mem (ea ctxBase 144) = some vc.tabsCap := vc.rd64_ctx_of_eq h (by decide) hsl144
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.tabs.length) with rip := 4198288 } := by
    rw [step_at_vc 12 hrip (by decide : (SporosVmBin.P_tnew, (.ofInt (.ld 0 15 136)), 7) ∈ vmTableChunk 12),
      exec_ld h15 hrd136, hrip]; rfl
  let s1 := { s.set 0 (some ls.tabs.length) with rip := 4198288 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_0 : s1.reg 0 = some ls.tabs.length := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some vc.tabsCap) with rip := 4198295 } := by
    rw [step_at_vc 12 rfl (by decide : (4198288, (.ofInt (.ld 1 15 144)), 7) ∈ vmTableChunk 12),
      exec_ld hs1_15 hrd144]
  let s2 := { s1.set 1 (some vc.tabsCap) with rip := 4198295 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some vc.tabsCap := reg_set_self_ok' hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some ls.tabs.length := by
    change (s1.set 1 _).reg 0 = some ls.tabs.length; rw [reg_set_ne (by decide)]; exact hs1_0
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub ls.tabs.length vc.tabsCap).2, rip := 4198298 } := by
    rw [step_at_vc 12 rfl (by decide : (4198295, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 12),
      exec_cmp hs2_0 hs2_1]
  let s3 := { s2 with fl := (X86Int.aluF .sub ls.tabs.length vc.tabsCap).2, rip := 4198298 }
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  by_cases hroom : ls.tabs.length < vc.tabsCap
  · have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198351 } := by
      rw [step_at_vc 12 rfl (by decide : (4198298, (.ofInt (.jcc 2 4198351)), 2) ∈ vmTableChunk 12),
        exec_jcc (cond_aluF_sub_b ls.tabs.length vc.tabsCap)]
      simp [s3, hroom]
    let s4 := { s3 with rip := 4198351 }
    have hrch4 : Reaches vmImage s s4 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr2 hs3_4 hs3_15
    exact lift_primPostOutcome hrch4 rfl
      (tn_room_spec rfl hwf4 (oracleAdmits_of_reaches hor hrch4) hrs hpcW hroom h272 h280 h288)
  · have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198300 } := by
      rw [step_at_vc 12 rfl (by decide : (4198298, (.ofInt (.jcc 2 4198351)), 2) ∈ vmTableChunk 12),
        exec_jcc (cond_aluF_sub_b ls.tabs.length vc.tabsCap)]
      simp [s3, hroom]
    let s4 := { s3 with rip := 4198300 }
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 7 (some vc.tabsReg.base) with rip := 4198307 } := by
      rw [step_at_vc 12 rfl (by decide : (4198300, (.ofInt (.ld 7 15 128)), 7) ∈ vmTableChunk 12),
        exec_ld (show s4.reg 15 = some ctxBase from hs3_15) hrd128]
    let s5 := { s4.set 7 (some vc.tabsReg.base) with rip := 4198307 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_7 : s5.reg 7 = some vc.tabsReg.base := reg_set_self_ok' hr2 (by decide) _
    have hs5_1 : s5.reg 1 = some vc.tabsCap := by
      change (s4.set 7 _).reg 1 = some vc.tabsCap; rw [reg_set_ne (by decide)]; exact hs2_1
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 6 (some vc.tabsCap) with rip := 4198310 } := by
      rw [step_at_vc 12 rfl (by decide : (4198307, (.ofInt (.mov 6 1)), 3) ∈ vmTableChunk 12), exec_mov hs5_1]
    let s6 := { s5.set 6 (some vc.tabsCap) with rip := 4198310 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_6 : s6.reg 6 = some vc.tabsCap := reg_set_self_ok' hr5 (by decide) _
    have hcap16W : vc.tabsCap * 16 < X86Env.W := by simp only [Region.hi] at htb_hi; omega
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 6 (some (vc.tabsCap * 8)) with fl := X86Int.Flags.undef, rip := 4198314 } := by
      rw [step_at_vc 12 rfl (by decide : (4198310, (.ofInt (.shlI 6 3)), 4) ∈ vmTableChunk 12),
        exec_shlI_lt (c := 3) hs6_6 (by rw [show 2 ^ (3 % 64) = 8 from by decide]; omega)]
    let s7 := { s6.set 6 (some (vc.tabsCap * 8)) with fl := X86Int.Flags.undef, rip := 4198314 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
    have hs7_6 : s7.reg 6 = some (vc.tabsCap * 8) := reg_set_self_ok' hr6 (by decide) _
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some (vc.tabsCap * 8)) with rip := 4198317 } := by
      rw [step_at_vc 12 rfl (by decide : (4198314, (.ofInt (.mov 2 6)), 3) ∈ vmTableChunk 12), exec_mov hs7_6]
    let s8 := { s7.set 2 (some (vc.tabsCap * 8)) with rip := 4198317 }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
    have hs8_2 : s8.reg 2 = some (vc.tabsCap * 8) := reg_set_self_ok' hr7 (by decide) _
    have h09 : X86Vm.step vmImage s8 =
        .ok { s8.set 2 (some (vc.tabsCap * 16)) with fl := X86Int.Flags.undef, rip := 4198321 } := by
      rw [step_at_vc 12 rfl (by decide : (4198317, (.ofInt (.shlI 2 1)), 4) ∈ vmTableChunk 12),
        exec_shlI_1_lt hs8_2 (by omega)]
      simp [s8, show vc.tabsCap * 8 * 2 = vc.tabsCap * 16 by omega]
    let s9 := { s8.set 2 (some (vc.tabsCap * 16)) with fl := X86Int.Flags.undef, rip := 4198321 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_2 : s9.reg 2 = some (vc.tabsCap * 16) := reg_set_self_ok' hr8 (by decide) _
    have hs9_6 : s9.reg 6 = some (vc.tabsCap * 8) := by
      change (s8.set 2 _).reg 6 = some (vc.tabsCap * 8); rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 6 = some (vc.tabsCap * 8); rw [reg_set_ne (by decide)]; exact hs7_6
    have hs9_7 : s9.reg 7 = some vc.tabsReg.base := by
      change (s8.set 2 _).reg 7 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 7 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]
      change (s6.set 6 _).reg 7 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]
      change (s5.set 6 _).reg 7 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]; exact hs5_7
    have hs9_4 : s9.reg 4 = some vc.sp := by
      change (s8.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s6.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s5.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s4.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs3_4
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s6.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s5.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s4.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
    have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc h rfl hr9 hs9_4 hs9_15
    obtain ⟨s10, h10, hrip10, hk10, _, hcm10, _, _, hr10, hs10_4, hs10_15, hs10_oth,
        hstk10, hstk10_b, hstk10_rw, hstk10_len, h16, hsp10W, hstk10_lo, hstk10_hi, hrd_ret, hoth10⟩ :=
      step_call_stk (vc := vc) (pr := pr) (ls := ls) (s := s9) (t := SporosVmBin.fn_mremap)
        (step_at_vc 12 rfl (by decide : (4198321, (.ofInt (.call 4201363)), 5) ∈ vmTableChunk 12))
        hwf9 (by change 4198321 + 5 < X86Env.W; omega)
    have hrch10 : Reaches vmImage s s10 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step h10)))))))))
    have hne_stk_tb : vc.stk.base ≠ vc.tabsReg.base :=
      vc.ne_bases h (i := 1) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hne_ctx_tb : vc.ctxReg.base ≠ vc.tabsReg.base :=
      vc.ne_bases h (i := 2) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hne_stk_ctx : vc.stk.base ≠ vc.ctxReg.base :=
      vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have htb10 : vc.tabsReg ∈ s10.mem := hoth10 vc.tabsReg (vc.mem_tabs hwf9) (Ne.symm hne_stk_tb)
    obtain ⟨idx, hfind10⟩ := mem_find_of_mem s10.mem vc.tabsReg vc.tabsReg.base 0 hcm10.2.2 htb10
      (Nat.le_refl _) (by simp only [Region.hi]; omega)
    rw [Nat.zero_add] at hfind10
    have hpg8 : X86Env.pages (vc.tabsCap * 8) = vc.tabsCap * 8 := pages_of_dvd htb_pg
    have hdvd16 : (vc.tabsCap * 16) % X86Env.page = 0 := by
      rw [show vc.tabsCap * 16 = 2 * (vc.tabsCap * 8) by omega, Nat.mul_mod, htb_pg]; simp
    have hpg16 : X86Env.pages (vc.tabsCap * 16) = vc.tabsCap * 16 := pages_of_dvd hdvd16
    have hmremap := call_mremap_oracle_spec hrip10 hr10 (oracleAdmits_of_reaches hor hrch10)
      hs10_4 (by omega) hsp10W hcm10.2 hstk10 hstk10_rw (by rw [hstk10_b]; exact hne_stk_tb)
      (by omega) hstk10_hi hrd_ret
      (by rw [hs10_oth 7 (by decide)]; exact hs9_7)
      (by rw [hs10_oth 6 (by decide)]; exact hs9_6)
      (by rw [hs10_oth 2 (by decide)]; exact hs9_2)
      hfind10 rfl (by rw [htb_len, hpg8]) htb_rw (by rw [htb_len, hpg16]; omega)
    rcases hmremap with ⟨a, s11, hrch11, _, _, _, ha_lim, hfree, hrip11, hk11, hmem11, hmwf11,
        hr11, hs11_0, hs11_4, _, hs11_oth⟩ | ⟨hhit, hends⟩
    · have hsb := vc.sp_bounds h
      have hs11_4' : s11.reg 4 = some vc.sp := by
        simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hs11_4
      have hs11_15 : s11.reg 15 = some ctxBase := by
        rw [hs11_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide)]; exact hs10_15
      have hk11_out : s11.k.out = s.k.out := by
        have hk9 : s9.k = s.k := rfl
        rw [hk11, hk10, hk9]; unfold X86Env.Kern.next; cases s.k.oracle <;> rfl
      have hget10 : s10.mem[idx]? = some vc.tabsReg := by
        simpa using (mem_find_idx_lt s10.mem vc.tabsReg.base 0 idx vc.tabsReg hfind10).2.2
      rw [hpg16] at hfree ha_lim
      let stk10 : Region := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (4198321 + 5)) }
      have hstk10_hi_eq : stk10.hi = vc.stk.hi := by simp [stk10, Region.hi, writeBytes_length]
      obtain ⟨_, hfree_all⟩ := mremap_find_after_stack_write (n := vc.tabsCap * 16)
        h.1.2 (vc.mem_stk h) hstk10_b hstk10_hi_eq hne_stk_tb hstk10 hoth10 hget10 rfl hfree
        (by omega) h.2.2.1
      have hoth11 : ∀ u ∈ s10.mem, u.base ≠ vc.tabsReg.base → u ∈ s11.mem := by
        intro u hu hne
        rw [hmem11]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base _ _ hget10 hu hne)
      let newTb : Region := ⟨a, vc.tabsReg.data ++ List.replicate (vc.tabsCap * 8) 0, true⟩
      have hnewTb_mem : newTb ∈ s11.mem := by
        rw [hmem11, hpg16, htb_len, show vc.tabsCap * 16 - vc.tabsCap * 8 = vc.tabsCap * 8 by omega]
        exact List.mem_append_right _ (List.mem_cons_self ..)
      have hctx11_mem : vc.ctxReg ∈ s11.mem :=
        hoth11 vc.ctxReg (hoth10 vc.ctxReg (vc.mem_ctx hwf9) (Ne.symm hne_stk_ctx)) hne_ctx_tb
      have hstk11_mem : stk10 ∈ s11.mem := hoth11 stk10 hstk10 (by rw [hstk10_b]; exact hne_stk_tb)
      have ha_ne_ctx : a ≠ vc.ctxReg.base :=
        mem_free_ne_base (by omega) (h.2.2.1 vc.ctxReg (vc.mem_ctx h)) (hfree_all vc.ctxReg (vc.mem_ctx h) hne_ctx_tb)
      have hea128 : ea ctxBase 128 = ctxBase + 128 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
      have hea144 : ea ctxBase 144 = ctxBase + 144 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
      obtain ⟨_, m12, hwr12, _, _, hmwf12, hctx12_mem, hoth12⟩ :=
        wr64_region hmwf11 hctx11_mem hctx_b.2.1 (a := ctxBase + 128)
          (by rw [hctx_b.1]; omega) (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega) (v := a)
      have hwr12' : wr64 s11.mem (ea ctxBase 128) a = some m12 := by rw [hea128]; exact hwr12
      have hrip11' : s11.rip = 4198326 := hrip11
      have h12 : X86Vm.step vmImage s11 = .ok { s11 with mem := m12, rip := 4198333 } := by
        rw [step_at_vc 12 hrip11' (by decide : (4198326, (.ofInt (.st 15 128 0)), 7) ∈ vmTableChunk 12),
          exec_st hs11_15 hs11_0 hwr12', hrip11']
      let s12 := { s11 with mem := m12, rip := 4198333 }
      have hsub128 : ctxBase + 128 - vc.ctxReg.base = 128 := by rw [hctx_b.1]; omega
      rw [hsub128] at hctx12_mem
      let ctx12 : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 128 (bytes8 a) }
      have hrd144_12 : rd64 s12.mem (ea ctxBase 144) = some vc.tabsCap := by
        rw [hea144]
        refine rd64_of_mem hmwf12 hctx12_mem (by dsimp [ctx12]; rw [hctx_b.1]; omega)
          (by simp only [Region.hi, hctx_b.1, writeBytes_length]; have := hctx_b.2.2.1; omega) ?_
        dsimp only [ctx12]
        rw [show ctxBase + 144 - vc.ctxReg.base = 144 by rw [hctx_b.1]; omega,
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
              (by have := hctx_b.2.2.1; omega), hsl144]
      have h13 : X86Vm.step vmImage s12 = .ok { s12.set 1 (some vc.tabsCap) with rip := 4198340 } := by
        rw [step_at_vc 13 rfl (by decide : (4198333, (.ofInt (.ld 1 15 144)), 7) ∈ vmTableChunk 13),
          exec_ld (show s12.reg 15 = some ctxBase from hs11_15) hrd144_12]
      let s13 := { s12.set 1 (some vc.tabsCap) with rip := 4198340 }
      have hr13 : RegsOk s13 := regsOk_set hr11 (by decide) _
      have hs13_1 : s13.reg 1 = some vc.tabsCap := reg_set_self_ok' hr11 (by decide) _
      have h14 : X86Vm.step vmImage s13 =
          .ok { s13.set 1 (some (vc.tabsCap * 2)) with fl := X86Int.Flags.undef, rip := 4198344 } := by
        rw [step_at_vc 13 rfl (by decide : (4198340, (.ofInt (.shlI 1 1)), 4) ∈ vmTableChunk 13),
          exec_shlI_1_lt hs13_1 (by omega)]
      let s14 := { s13.set 1 (some (vc.tabsCap * 2)) with fl := X86Int.Flags.undef, rip := 4198344 }
      have hr14 : RegsOk s14 := regsOk_set hr13 (by decide) _
      have hs14_1 : s14.reg 1 = some (vc.tabsCap * 2) := reg_set_self_ok' hr13 (by decide) _
      have hs14_4 : s14.reg 4 = some vc.sp := by
        change (s13.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s12.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs11_4'
      have hs14_15 : s14.reg 15 = some ctxBase := by
        change (s13.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s12.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs11_15
      obtain ⟨_, m15, hwr15, _, _, hmwf15, hctx15_mem, hoth15⟩ :=
        wr64_region hmwf12 hctx12_mem hctx_b.2.1 (a := ctxBase + 144)
          (by dsimp [ctx12]; rw [hctx_b.1]; omega)
          (by simp only [Region.hi, hctx_b.1, writeBytes_length]; have := hctx_b.2.2.1; omega)
          (v := vc.tabsCap * 2)
      have hwr15' : wr64 s14.mem (ea ctxBase 144) (vc.tabsCap * 2) = some m15 := by rw [hea144]; exact hwr15
      have h15s : X86Vm.step vmImage s14 = .ok { s14 with mem := m15, rip := 4198351 } := by
        rw [step_at_vc 13 rfl (by decide : (4198344, (.ofInt (.st 15 144 1)), 7) ∈ vmTableChunk 13),
          exec_st hs14_15 hs14_1 hwr15']
      let s15 := { s14 with mem := m15, rip := 4198351 }
      have hrch_tot : Reaches vmImage s s15 :=
        Reaches.trans hrch10 (Reaches.trans hrch11
          (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.step_trans h14 (Reaches.step h15s)))))
      have ⟨hbl15, hpos15⟩ := reaches_bytesLt_len_pos h.2.1 h.2.2.1 hrch_tot
      have hsub144 : ctxBase + 144 - ctx12.base = 144 := by dsimp [ctx12]; rw [hctx_b.1]; omega
      rw [hsub144] at hctx15_mem
      let ctx15 : Region := { ctx12 with data := writeBytes ctx12.data 144 (bytes8 (vc.tabsCap * 2)) }
      have hnewTb15 : newTb ∈ m15 := hoth15 newTb (hoth12 newTb hnewTb_mem ha_ne_ctx) ha_ne_ctx
      have hstk15 : stk10 ∈ m15 := hoth15 stk10 (hoth12 stk10 hstk11_mem (by rw [hstk10_b]; exact hne_stk_ctx))
        (by rw [hstk10_b]; exact hne_stk_ctx)
      have hc128' : leVal ((ctx15.data.drop 128).take 8) = newTb.base := by
        dsimp only [ctx15, ctx12, newTb]
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide))
              (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
            leVal_writeBytes_same _ _ _ (by have := hctx_b.2.2.1; omega), Nat.mod_eq_of_lt (by omega)]
      have hc136' : leVal ((ctx15.data.drop 136).take 8) = ls.tabs.length := by
        dsimp only [ctx15, ctx12]
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide))
              (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
              (by have := hctx_b.2.2.1; omega), hsl136]
      have hc144' : leVal ((ctx15.data.drop 144).take 8) = vc.tabsCap * 2 := by
        dsimp only [ctx15, ctx12]
        rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; have := hctx_b.2.2.1; omega),
            Nat.mod_eq_of_lt (by omega)]
      have hctx_oth' : ∀ j, (j < 128 ∨ 152 ≤ j) → ctx15.data[j]? = vc.ctxReg.data[j]? := by
        intro j hj; dsimp only [ctx15, ctx12]
        rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
      obtain ⟨hwf15, hpres15⟩ := wf_update_tabs h hmwf15 hbl15 hpos15 hr14 hs14_4 hs14_15
        hnewTb15 rfl (by omega)
        (by simp [newTb, List.length_append, List.length_replicate, htb_len]; omega)
        (by rw [show vc.tabsCap * 2 * 8 = vc.tabsCap * 16 by omega]; exact hdvd16)
        (by simp [newTb, Region.hi, List.length_append, List.length_replicate, htb_len]; omega)
        hctx15_mem rfl rfl (by dsimp [ctx15, ctx12]; rw [writeBytes_length, writeBytes_length])
        hc128' hc136' hc144' hctx_oth'
        hstk15 hstk10_b rfl hstk10_len
        (fun u hu hne => Ne.symm (mem_free_ne_base (by omega) (h.2.2.1 u hu) (hfree_all u hu hne)))
        (fun u hu h1 h2 h3 => hoth15 u (hoth12 u (hoth11 u (hoth10 u hu h3) h1) h2) h2)
        (tabsRel_extend_tabs_zero htabs_rel) (by omega) (vc.code_flat h).2.2.2.2.2.2.2.2.1
      have hfin := tn_room_spec rfl hwf15 (oracleAdmits_of_reaches hor hrch_tot) hrs hpcW (by dsimp; omega)
        (hpres15 272 _ (Or.inr (by decide)) (by decide) h272)
        (hpres15 280 0 (Or.inr (by decide)) (by decide) h280)
        (hpres15 288 1 (Or.inr (by decide)) (by decide) h288)
      unfold PrimPostOutcome at hfin ⊢
      have hexec : SporosProvLen.lexecPrim .tnew (ls.ds.take Sporos.Prim.tnew.eff.1) ls =
          .ok [⟨Sporos.wrap ls.tabs.length, SporosProvLen.lq (ls.ds.take 0) ls.pc⟩]
            { ls with tabs := ls.tabs ++ [[]] } := by
        simp [Sporos.Prim.eff, SporosProvLen.lexecPrim]
      rw [hexec] at hfin ⊢
      exact lift_next_outcome_out hrch_tot hk11_out hfin
    · unfold PrimPostOutcome
      have hexec : SporosProvLen.lexecPrim .tnew (ls.ds.take Sporos.Prim.tnew.eff.1) ls =
          .ok [⟨Sporos.wrap ls.tabs.length, SporosProvLen.lq (ls.ds.take 0) ls.pc⟩]
            { ls with tabs := ls.tabs ++ [[]] } := by
        simp [Sporos.Prim.eff, SporosProvLen.lexecPrim]
      rw [hexec]
      exact Or.inr ⟨hitsNomem_of_reaches hrch10 hhit, Ends.of_reaches hrch10 hends⟩

end SporosVmPrimTab
