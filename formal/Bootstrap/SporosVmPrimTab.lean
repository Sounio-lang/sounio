import Bootstrap.SporosVmPrimTab1
import Bootstrap.SporosVmPrimTab3

/-!
# Native x86-64 VM Table Primitives (`P_tnew`, `P_tput`, `P_tget`)

Umbrella module for all three table primitives: re-exports `P_tnew_spec`
(from `Bootstrap.SporosVmPrimTab1`) and `P_tput_spec` (from
`Bootstrap.SporosVmPrimTab3`), and proves `P_tget_spec` (primitive 46) via
structural induction on `ChainRel` matching `SporosProvLen.ltfind`.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimTab

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

attribute [local instance] instInhabitedLVal instInhabitedLFrame

/-! ### `P_tget` (46) -/

private theorem tg_hit_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {idx : Nat} {v : SporosProvLen.LVal}
    (hrip : s.rip = 4198675) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hidx_lt : idx * 32 + 32 ≤ vc.kvReg.data.length)
    (hvw : leVal ((vc.kvReg.data.drop (idx * 32 + 8)).take 8) = v.w)
    (hvp : leVal ((vc.kvReg.data.drop (idx * 32 + 16)).take 8) = v.plen)
    (hr8 : s.reg 8 = some vc.kvReg.base) (hr1 : s.reg 1 = some (idx * 32))
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 2) :
    let q := SporosProvLen.lq (ls.ds.take 2) ls.pc
    let outs : List SporosProvLen.LVal := [⟨Sporos.wrap 1, q⟩, ⟨v.w, (v.plen + q) % Sporos.M⟩]
    let ls' := { ls with ds := outs ++ ls.ds.drop 2, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  let q := SporosProvLen.lq (ls.ds.take 2) ls.pc
  have hqW : q < X86Env.W := by dsimp [q, SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hvwW : v.w < X86Env.W := by
    rw [← hvw]; exact leVal_take8_lt_W fun b hb => h.2.1 vc.kvReg (vc.mem_kv h) b (List.mem_of_mem_drop hb)
  have hkv_hi : vc.kvReg.hi ≤ 2 ^ 47 := (wf_tabs_info h).2.1.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 10 (some 1) with rip := 4198681 } := by
    rw [step_at_vc 14 hrip (by decide : (4198675, (.ofInt (.movI 10 1)), 6) ∈ vmTableChunk 14),
      exec_movI_lt (by omega), hrip]
  let s1 := { s.set 10 (some 1) with rip := 4198681 }
  have hr1' : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_10 : s1.reg 10 = some 1 := reg_set_self_ok' hr (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_8 : s1.reg 8 = some vc.kvReg.base := by
    change (s.set 10 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hr8
  have hs1_1 : s1.reg 1 = some (idx * 32) := by
    change (s.set 10 _).reg 1 = some (idx * 32); rw [reg_set_ne (by decide)]; exact hr1
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1' hs1_4 hs1_15
  obtain ⟨s2, h02, hrip2, hk2, hregs2, _, hwf2, h360_2, hpres2⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198681, (.ofInt (.st 15 360 10)), 7) ∈ vmTableChunk 14))
      hwf1 hs1_10 (by decide) (by decide)
  have hs2_15 : s2.reg 15 = some ctxBase := hwf2.2.2.2.2.2.1
  have h272_2 : rd64 s2.mem (ea ctxBase 272) = some q := hpres2 272 q (Or.inl (by decide)) (by decide) h272
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 11 (some q) with rip := 4198695 } := by
    rw [step_at_vc 14 hrip2 (by decide : (4198688, (.ofInt (.ld 11 15 272)), 7) ∈ vmTableChunk 14),
      exec_ld hs2_15 h272_2, hrip2]
  let s3 := { s2.set 11 (some q) with rip := 4198695 }
  have hr3 : RegsOk s3 := regsOk_set hwf2.2.2.2.1 (by decide) _
  have hs3_11 : s3.reg 11 = some q := reg_set_self_ok' hwf2.2.2.2.1 (by decide) _
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 11 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.1
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 11 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hs3_8 : s3.reg 8 = some vc.kvReg.base := by
    change (s2.set 11 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
    simp only [X86Int.St.reg, hregs2]; exact hs1_8
  have hs3_1 : s3.reg 1 = some (idx * 32) := by
    change (s2.set 11 _).reg 1 = some (idx * 32); rw [reg_set_ne (by decide)]
    simp only [X86Int.St.reg, hregs2]; exact hs1_1
  let vc2 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 1) } }
  have hwf3 : vc2.wf pr ls s3 := vc2.wf_step_misc hwf2 rfl hr3 hs3_4 hs3_15
  obtain ⟨s4, h04, hrip4, hk4, hregs4, _, hwf4, h368_4, hpres4⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198695, (.ofInt (.st 15 368 11)), 7) ∈ vmTableChunk 14))
      hwf3 hs3_11 (by decide) (by decide)
  have hs4_8 : s4.reg 8 = some vc.kvReg.base := by simp only [X86Int.St.reg, hregs4]; exact hs3_8
  have hs4_1 : s4.reg 1 = some (idx * 32) := by simp only [X86Int.St.reg, hregs4]; exact hs3_1
  have hs4_11 : s4.reg 11 = some q := by simp only [X86Int.St.reg, hregs4]; exact hs3_11
  let vc4 := { vc2 with ctxReg := { vc2.ctxReg with data := writeBytes vc2.ctxReg.data 368 (bytes8 q) } }
  have hea8 : eaIdx vc.kvReg.base (idx * 32) 1 8 = vc.kvReg.base + (idx * 32 + 8) := by
    have hltW : vc.kvReg.base + 1 * (idx * 32) + 8 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
    rw [eaIdx_add_eq hltW]; omega
  have hrd_vw : rdMem .w64 s4.mem (eaIdx vc.kvReg.base (idx * 32) 1 8) = some v.w := by
    rw [hea8]
    exact rdMem_w64_at hwf4.1.2 (vc4.mem_kv hwf4) (Nat.le_add_right _ _)
      (by simp only [Region.hi, vc4, vc2]; omega) (by rw [Nat.add_sub_cancel_left]; exact hvw)
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some v.w) with rip := 4198707 } := by
    rw [step_at_vc 14 hrip4 (by decide : (4198702, (.ldIdx .w64 0 8 1 1 8), 5) ∈ vmTableChunk 14),
      exec_ldIdx hs4_8 hs4_1 hrd_vw, hrip4]
  let s5 := { s4.set 0 (some v.w) with rip := 4198707 }
  have hr5 : RegsOk s5 := regsOk_set hwf4.2.2.2.1 (by decide) _
  have hs5_0 : s5.reg 0 = some v.w := reg_set_self_ok' hwf4.2.2.2.1 (by decide) _
  have hs5_4 : s5.reg 4 = some vc.sp := by
    change (s4.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.1
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.2.1
  have hs5_8 : s5.reg 8 = some vc.kvReg.base := by
    change (s4.set 0 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hs4_8
  have hs5_1 : s5.reg 1 = some (idx * 32) := by
    change (s4.set 0 _).reg 1 = some (idx * 32); rw [reg_set_ne (by decide)]; exact hs4_1
  have hs5_11 : s5.reg 11 = some q := by
    change (s4.set 0 _).reg 11 = some q; rw [reg_set_ne (by decide)]; exact hs4_11
  have hwf5 : vc4.wf pr ls s5 := vc4.wf_step_misc hwf4 rfl hr5 hs5_4 hs5_15
  obtain ⟨s6, h06, hrip6, hk6, hregs6, _, hwf6, h376_6, hpres6⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198707, (.ofInt (.st 15 376 0)), 7) ∈ vmTableChunk 14))
      hwf5 hs5_0 (by decide) (by decide)
  have hs6_8 : s6.reg 8 = some vc.kvReg.base := by simp only [X86Int.St.reg, hregs6]; exact hs5_8
  have hs6_1 : s6.reg 1 = some (idx * 32) := by simp only [X86Int.St.reg, hregs6]; exact hs5_1
  have hs6_11 : s6.reg 11 = some q := by simp only [X86Int.St.reg, hregs6]; exact hs5_11
  let vc6 := { vc4 with ctxReg := { vc4.ctxReg with data := writeBytes vc4.ctxReg.data 376 (bytes8 v.w) } }
  have hea16 : eaIdx vc.kvReg.base (idx * 32) 1 16 = vc.kvReg.base + (idx * 32 + 16) := by
    have hltW : vc.kvReg.base + 1 * (idx * 32) + 16 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
    rw [eaIdx_add_eq hltW]; omega
  have hrd_vp : rdMem .w64 s6.mem (eaIdx vc.kvReg.base (idx * 32) 1 16) = some v.plen := by
    rw [hea16]
    exact rdMem_w64_at hwf6.1.2 (vc6.mem_kv hwf6) (Nat.le_add_right _ _)
      (by simp only [Region.hi, vc6, vc4, vc2]; omega) (by rw [Nat.add_sub_cancel_left]; exact hvp)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 0 (some v.plen) with rip := 4198719 } := by
    rw [step_at_vc 14 hrip6 (by decide : (4198714, (.ldIdx .w64 0 8 1 1 16), 5) ∈ vmTableChunk 14),
      exec_ldIdx hs6_8 hs6_1 hrd_vp, hrip6]
  let s7 := { s6.set 0 (some v.plen) with rip := 4198719 }
  have hr7 : RegsOk s7 := regsOk_set hwf6.2.2.2.1 (by decide) _
  have hs7_0 : s7.reg 0 = some v.plen := reg_set_self_ok' hwf6.2.2.2.1 (by decide) _
  have hs7_11 : s7.reg 11 = some q := by
    change (s6.set 0 _).reg 11 = some q; rw [reg_set_ne (by decide)]; exact hs6_11
  let vp' := (v.plen + q) % X86Env.W
  have h08 : X86Vm.step vmImage s7 =
      .ok { s7.set 0 (some vp') with fl := (X86Int.aluF .add v.plen q).2, rip := 4198722 } := by
    have hadd_vp' : X86Int.aluF .add v.plen q = (vp', (X86Int.aluF .add v.plen q).2) :=
      Prod.ext (aluF_add_val _ _) rfl
    rw [step_at_vc 14 rfl (by decide : (4198719, (.ofInt (.alu .add 0 11)), 3) ∈ vmTableChunk 14),
      exec_alu hs7_0 hs7_11, hadd_vp']
  let s8 := { s7.set 0 (some vp') with fl := (X86Int.aluF .add v.plen q).2, rip := 4198722 }
  have hr8' : RegsOk s8 := regsOk_set hr7 (by decide) _
  have hs8_0 : s8.reg 0 = some vp' := reg_set_self_ok' hr7 (by decide) _
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s6.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf6.2.2.2.2.1
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s6.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf6.2.2.2.2.2.1
  have hwf8 : vc6.wf pr ls s8 := vc6.wf_step_misc hwf6 rfl hr8' hs8_4 hs8_15
  obtain ⟨s9, h09, hrip9, hk9, _, _, hwf9, h384_9, hpres9⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198722, (.ofInt (.st 15 384 0)), 7) ∈ vmTableChunk 14))
      hwf8 hs8_0 (by decide) (by decide)
  have hrch9 : Reaches vmImage s s9 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step h09))))))))
  have hk9_eq : s9.k = s.k := by rw [hk9]; exact hk6.trans (hk4.trans hk2)
  have h280_9 : rd64 s9.mem (ea ctxBase 280) = some 2 :=
    hpres9 280 2 (Or.inl (by decide)) (by decide) (hpres6 280 2 (Or.inl (by decide)) (by decide)
      (hpres4 280 2 (Or.inl (by decide)) (by decide) (hpres2 280 2 (Or.inl (by decide)) (by decide) h280)))
  have h288_9 : rd64 s9.mem (ea ctxBase 288) = some 2 :=
    hpres9 288 2 (Or.inl (by decide)) (by decide) (hpres6 288 2 (Or.inl (by decide)) (by decide)
      (hpres4 288 2 (Or.inl (by decide)) (by decide) (hpres2 288 2 (Or.inl (by decide)) (by decide) h288)))
  have h360_9 : rd64 s9.mem (ea ctxBase 360) = some 1 :=
    hpres9 360 1 (Or.inl (by decide)) (by decide) (hpres6 360 1 (Or.inl (by decide)) (by decide)
      (hpres4 360 1 (Or.inl (by decide)) (by decide) (by change rd64 s2.mem _ = _; simpa [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W)] using h360_2)))
  have h368_9 : rd64 s9.mem (ea ctxBase 368) = some q :=
    hpres9 368 q (Or.inl (by decide)) (by decide) (hpres6 368 q (Or.inl (by decide)) (by decide)
      (by change rd64 s4.mem _ = _; simpa [Nat.mod_eq_of_lt hqW] using h368_4))
  have h376_9 : rd64 s9.mem (ea ctxBase 376) = some v.w :=
    hpres9 376 v.w (Or.inl (by decide)) (by decide) (by change rd64 s6.mem _ = _; simpa [Nat.mod_eq_of_lt hvwW] using h376_6)
  have h384_9' : rd64 s9.mem (ea ctxBase 384) = some vp' := by
    simpa [vp', Nat.mod_mod] using h384_9
  let outs : List SporosProvLen.LVal := [⟨Sporos.wrap 1, q⟩, ⟨v.w, (v.plen + q) % Sporos.M⟩]
  have hfin := jmp_epilogue0_finish_spec (outs := outs)
    (step_at_vc 14 hrip9 (by decide : (4198729, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 14))
    hwf9 (oracleAdmits_of_reaches hor hrch9) hrs hpcW hle (by simp [outs])
    (by intro u hu; simp only [outs, List.mem_cons, List.not_mem_nil, or_false] at hu
        rcases hu with rfl | rfl
        · exact ⟨by rw [Sporos.wrap, M_eq_W']; exact lt_W_of_mod _, hqW⟩
        · exact ⟨hvwW, by rw [M_eq_W']; exact lt_W_of_mod _⟩)
    h280_9 h288_9
    (by intro i hi
        have hcases : i = 0 ∨ i = 1 := by simp [outs] at hi; omega
        rcases hcases with rfl | rfl
        · exact ⟨by simpa [outs, Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt (by omega : 1 < X86Env.W)] using h360_9, h368_9⟩
        · exact ⟨h376_9, by simpa [outs, vp', M_eq_W'] using h384_9'⟩)
  exact lift_next_outcome hrch9 hk9_eq hfin

private theorem tg_miss_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = 4198734) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 2) :
    let q := SporosProvLen.lq (ls.ds.take 2) ls.pc
    let outs : List SporosProvLen.LVal := [⟨Sporos.wrap 0, q⟩, ⟨Sporos.wrap 0, q⟩]
    let ls' := { ls with ds := outs ++ ls.ds.drop 2, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  let q := SporosProvLen.lq (ls.ds.take 2) ls.pc
  have hqW : q < X86Env.W := by dsimp [q, SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hr : RegsOk s := h.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  obtain ⟨r0, hr0⟩ := reg_of_ok hr 0 (by decide) (by decide)
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4198736 } := by
    rw [step_at_vc 14 hrip (by decide : (4198734, (.ofInt (.alu .xor 0 0)), 2) ∈ vmTableChunk 14),
      exec_alu_xor_self hr0, hrip]
  let s1 := { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4198736 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_0 : s1.reg 0 = some 0 := reg_set_self_ok' hr (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, h02, hrip2, hk2, hregs2, _, hwf2, h360_2, hpres2⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198736, (.ofInt (.st 15 360 0)), 7) ∈ vmTableChunk 14))
      hwf1 hs1_0 (by decide) (by decide)
  have hs2_0 : s2.reg 0 = some 0 := by simp only [X86Int.St.reg, hregs2]; exact hs1_0
  obtain ⟨s3, h03, hrip3, hk3, _, _, hwf3, h376_3, hpres3⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 hrip2 (by decide : (4198743, (.ofInt (.st 15 376 0)), 7) ∈ vmTableChunk 14))
      hwf2 hs2_0 (by decide) (by decide)
  have h272_3 : rd64 s3.mem (ea ctxBase 272) = some q :=
    hpres3 272 q (Or.inl (by decide)) (by decide) (hpres2 272 q (Or.inl (by decide)) (by decide) h272)
  have hrip3' : s3.rip = 4198750 := by rw [hrip3, hrip2]
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 11 (some q) with rip := 4198757 } := by
    rw [step_at_vc 14 hrip3' (by decide : (4198750, (.ofInt (.ld 11 15 272)), 7) ∈ vmTableChunk 14),
      exec_ld hwf3.2.2.2.2.2.1 h272_3, hrip3']
  let s4 := { s3.set 11 (some q) with rip := 4198757 }
  have hr4 : RegsOk s4 := regsOk_set hwf3.2.2.2.1 (by decide) _
  have hs4_11 : s4.reg 11 = some q := reg_set_self_ok' hwf3.2.2.2.1 (by decide) _
  have hs4_4 : s4.reg 4 = some vc.sp := by
    change (s3.set 11 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf3.2.2.2.2.1
  have hs4_15 : s4.reg 15 = some ctxBase := by
    change (s3.set 11 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf3.2.2.2.2.2.1
  let vc2 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 0) } }
  let vc3 := { vc2 with ctxReg := { vc2.ctxReg with data := writeBytes vc2.ctxReg.data 376 (bytes8 0) } }
  have hwf4 : vc3.wf pr ls s4 := vc3.wf_step_misc hwf3 rfl hr4 hs4_4 hs4_15
  obtain ⟨s5, h05, hrip5, hk5, hregs5, _, hwf5, h368_5, hpres5⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 rfl (by decide : (4198757, (.ofInt (.st 15 368 11)), 7) ∈ vmTableChunk 14))
      hwf4 hs4_11 (by decide) (by decide)
  have hs5_11 : s5.reg 11 = some q := by simp only [X86Int.St.reg, hregs5]; exact hs4_11
  obtain ⟨s6, h06, hrip6, hk6, _, _, hwf6, h384_6, hpres6⟩ :=
    step_wr_ctx_scratch (step_at_vc 14 hrip5 (by decide : (4198764, (.ofInt (.st 15 384 11)), 7) ∈ vmTableChunk 14))
      hwf5 hs5_11 (by decide) (by decide)
  have hrch6 : Reaches vmImage s s6 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step h06)))))
  have hk6_eq : s6.k = s.k := by rw [hk6]; exact hk5.trans (hk3.trans hk2)
  have h280_6 : rd64 s6.mem (ea ctxBase 280) = some 2 :=
    hpres6 280 2 (Or.inl (by decide)) (by decide) (hpres5 280 2 (Or.inl (by decide)) (by decide)
      (hpres3 280 2 (Or.inl (by decide)) (by decide) (hpres2 280 2 (Or.inl (by decide)) (by decide) h280)))
  have h288_6 : rd64 s6.mem (ea ctxBase 288) = some 2 :=
    hpres6 288 2 (Or.inl (by decide)) (by decide) (hpres5 288 2 (Or.inl (by decide)) (by decide)
      (hpres3 288 2 (Or.inl (by decide)) (by decide) (hpres2 288 2 (Or.inl (by decide)) (by decide) h288)))
  have h360_6 : rd64 s6.mem (ea ctxBase 360) = some 0 :=
    hpres6 360 0 (Or.inl (by decide)) (by decide) (hpres5 360 0 (Or.inl (by decide)) (by decide)
      (hpres3 360 0 (Or.inl (by decide)) (by decide) (by simpa [Nat.zero_mod] using h360_2)))
  have h368_6 : rd64 s6.mem (ea ctxBase 368) = some q :=
    hpres6 368 q (Or.inl (by decide)) (by decide) (by simpa [Nat.mod_eq_of_lt hqW] using h368_5)
  have h376_6 : rd64 s6.mem (ea ctxBase 376) = some 0 :=
    hpres6 376 0 (Or.inl (by decide)) (by decide) (hpres5 376 0 (Or.inr (by decide)) (by decide)
      (by change rd64 s3.mem _ = _; simpa [Nat.zero_mod] using h376_3))
  have h384_6' : rd64 s6.mem (ea ctxBase 384) = some q := by simpa [Nat.mod_eq_of_lt hqW] using h384_6
  have hrip6' : s6.rip = 4198771 := by rw [hrip6, hrip5]
  let outs : List SporosProvLen.LVal := [⟨Sporos.wrap 0, q⟩, ⟨Sporos.wrap 0, q⟩]
  have hfin := jmp_epilogue0_finish_spec (outs := outs)
    (step_at_vc 14 hrip6' (by decide : (4198771, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 14))
    hwf6 (oracleAdmits_of_reaches hor hrch6) hrs hpcW hle (by simp [outs])
    (by intro u hu; simp only [outs, List.mem_cons, List.not_mem_nil, or_false] at hu
        rcases hu with rfl | rfl <;> exact ⟨by rw [Sporos.wrap, M_eq_W']; exact lt_W_of_mod _, hqW⟩)
    h280_6 h288_6
    (by intro i hi
        have hcases : i = 0 ∨ i = 1 := by simp [outs] at hi; omega
        rcases hcases with rfl | rfl
        · exact ⟨by simpa [outs, Sporos.wrap, M_eq_W', Nat.zero_mod] using h360_6, h368_6⟩
        · exact ⟨by simpa [outs, Sporos.wrap, M_eq_W', Nat.zero_mod] using h376_6, h384_6'⟩)
  exact lift_next_outcome hrch6 hk6_eq hfin

private theorem tg_loop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {k : Nat}
    {idx : Nat} {entries : List (Nat × SporosProvLen.LVal)}
    (hc : ChainRel vc.kvReg.data vc.kvLen idx entries) :
    ∀ {s : X86Vm.St},
      s.rip = 4198647 → 2 ≤ ls.ds.length → vc.wf pr ls s → OracleAdmits s →
      ls.rs = f :: rest → f.pc + 1 < X86Env.W → k < X86Env.W →
      s.reg 0 = some idx → s.reg 8 = some vc.kvReg.base → s.reg 2 = some k →
      s.reg 6 = some 18446744073709551615 →
      rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc) →
      rd64 s.mem (ea ctxBase 280) = some 2 → rd64 s.mem (ea ctxBase 288) = some 2 →
      let q := SporosProvLen.lq (ls.ds.take 2) ls.pc
      let outs := match SporosProvLen.ltfind entries k with
        | some v => [⟨Sporos.wrap 1, q⟩, ⟨v.w, (v.plen + q) % Sporos.M⟩]
        | none => [⟨Sporos.wrap 0, q⟩, ⟨Sporos.wrap 0, q⟩]
      let ls' := { ls with ds := outs ++ ls.ds.drop 2, rs := { f with pc := f.pc + 1 } :: rest }
      (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
        (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  induction hc with
  | nil =>
    intro s hrip hle h hor hrs hpcW _ hr0 _ _ hr6 h272 h280 h288
    have hr : RegsOk s := h.2.2.2.1
    have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
    have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
    have h01 : X86Vm.step vmImage s =
        .ok { s with fl := (X86Int.aluF .sub (X86Env.W - 1) 18446744073709551615).2, rip := 4198650 } := by
      rw [step_at_vc 13 hrip (by decide : (4198647, (.ofInt (.cmp 0 6)), 3) ∈ vmTableChunk 13),
        exec_cmp hr0 hr6, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub (X86Env.W - 1) 18446744073709551615).2, rip := 4198650 }
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4198734 } := by
      rw [step_at_vc 13 rfl (by decide : (4198650, (.ofInt (.jcc 4 4198734)), 2) ∈ vmTableChunk 13),
        exec_jcc (cond_aluF_sub_e (X86Env.W - 1) 18446744073709551615 (by omega) (by omega))]
      simp [s1, hW]
    let s2 := { s1 with rip := 4198734 }
    have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
    have hwf2 : vc.wf pr ls s2 := vc.wf_step_misc h rfl hr h4 h15
    simpa [SporosProvLen.ltfind] using
      lift_next_outcome hrch2 rfl (tg_miss_spec rfl hle hwf2 (oracleAdmits_of_reaches hor hrch2) hrs hpcW h272 h280 h288)
  | @cons idx next ek ev tail _ hlen hek hvw hvp hnx _ ih =>
    intro s hrip hle h hor hrs hpcW hkW hr0 hr8 hr2 hr6 h272 h280 h288
    have hr : RegsOk s := h.2.2.2.1
    have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
    have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
    have hkv_hi : vc.kvReg.hi ≤ 2 ^ 47 := (wf_tabs_info h).2.1.2.2.2.2.1
    have hidxW : idx < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
    have hidx_ne : idx ≠ 18446744073709551615 := by simp only [Region.hi] at hkv_hi; omega
    have hekW : ek < X86Env.W := by
      rw [← hek]; exact leVal_take8_lt_W fun b hb => h.2.1 vc.kvReg (vc.mem_kv h) b (List.mem_of_mem_drop hb)
    have h01 : X86Vm.step vmImage s =
        .ok { s with fl := (X86Int.aluF .sub idx 18446744073709551615).2, rip := 4198650 } := by
      rw [step_at_vc 13 hrip (by decide : (4198647, (.ofInt (.cmp 0 6)), 3) ∈ vmTableChunk 13),
        exec_cmp hr0 hr6, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub idx 18446744073709551615).2, rip := 4198650 }
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4198652 } := by
      rw [step_at_vc 13 rfl (by decide : (4198650, (.ofInt (.jcc 4 4198734)), 2) ∈ vmTableChunk 13),
        exec_jcc (cond_aluF_sub_e idx 18446744073709551615 hidxW (by omega))]
      simp [s1, hidx_ne]
    let s2 := { s1 with rip := 4198652 }
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some idx) with rip := 4198655 } := by
      rw [step_at_vc 13 rfl (by decide : (4198652, (.ofInt (.mov 1 0)), 3) ∈ vmTableChunk 13),
        exec_mov (show s2.reg 0 = some idx from hr0)]
    let s3 := { s2.set 1 (some idx) with rip := 4198655 }
    have hr3 : RegsOk s3 := regsOk_set hr (by decide) _
    have hs3_1 : s3.reg 1 = some idx := reg_set_self_ok' hr (by decide) _
    have hidx32W : idx * 32 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 1 (some (idx * 32)) with fl := X86Int.Flags.undef, rip := 4198659 } := by
      rw [step_at_vc 13 rfl (by decide : (4198655, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk 13),
        exec_shlI_lt (c := 5) hs3_1 (by rw [show 2 ^ (5 % 64) = 32 from by decide]; exact hidx32W)]
    let s4 := { s3.set 1 (some (idx * 32)) with fl := X86Int.Flags.undef, rip := 4198659 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
    have hs4_1 : s4.reg 1 = some (idx * 32) := reg_set_self_ok' hr3 (by decide) _
    have hs4_8 : s4.reg 8 = some vc.kvReg.base := by
      change (s3.set 1 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
      change (s2.set 1 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hr8
    have hea0 : eaIdx vc.kvReg.base (idx * 32) 1 0 = vc.kvReg.base + idx * 32 := by
      have hltW : vc.kvReg.base + 1 * (idx * 32) + 0 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_k : rdMem .w64 s4.mem (eaIdx vc.kvReg.base (idx * 32) 1 0) = some ek := by
      have hkv_in : vc.kvReg ∈ s.mem := vc.mem_kv h
      rw [hea0]
      exact rdMem_w64_at h.1.2 hkv_in (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hek)
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 10 (some ek) with rip := 4198663 } := by
      rw [step_at_vc 13 rfl (by decide : (4198659, (.ldIdx .w64 10 8 1 1 0), 4) ∈ vmTableChunk 13),
        exec_ldIdx hs4_8 hs4_1 hrd_k]
    let s5 := { s4.set 10 (some ek) with rip := 4198663 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
    have hs5_10 : s5.reg 10 = some ek := reg_set_self_ok' hr4 (by decide) _
    have hs5_2 : s5.reg 2 = some k := by
      change (s4.set 10 _).reg 2 = some k; rw [reg_set_ne (by decide)]
      change (s3.set 1 _).reg 2 = some k; rw [reg_set_ne (by decide)]
      change (s2.set 1 _).reg 2 = some k; rw [reg_set_ne (by decide)]; exact hr2
    have hs5_8 : s5.reg 8 = some vc.kvReg.base := by
      change (s4.set 10 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hs4_8
    have hs5_1 : s5.reg 1 = some (idx * 32) := by
      change (s4.set 10 _).reg 1 = some (idx * 32); rw [reg_set_ne (by decide)]; exact hs4_1
    have hs5_6 : s5.reg 6 = some 18446744073709551615 := by
      change (s4.set 10 _).reg 6 = some 18446744073709551615; rw [reg_set_ne (by decide)]
      change (s3.set 1 _).reg 6 = some 18446744073709551615; rw [reg_set_ne (by decide)]
      change (s2.set 1 _).reg 6 = some 18446744073709551615; rw [reg_set_ne (by decide)]; exact hr6
    have hs5_4 : s5.reg 4 = some vc.sp := by
      change (s4.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s3.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s2.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s3.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s2.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5 with fl := (X86Int.aluF .sub ek k).2, rip := 4198666 } := by
      rw [step_at_vc 13 rfl (by decide : (4198663, (.ofInt (.cmp 10 2)), 3) ∈ vmTableChunk 13),
        exec_cmp hs5_10 hs5_2]
    let s6 := { s5 with fl := (X86Int.aluF .sub ek k).2, rip := 4198666 }
    by_cases hmatch : ek = k
    · have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4198675 } := by
        rw [step_at_vc 13 rfl (by decide : (4198666, (.ofInt (.jcc 4 4198675)), 2) ∈ vmTableChunk 13),
          exec_jcc (cond_aluF_sub_e ek k hekW hkW)]
        simp [s6, hmatch]
      let s7 := { s6 with rip := 4198675 }
      have hrch7 : Reaches vmImage s s7 :=
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
          (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step h07))))))
      have hwf7 : vc.wf pr ls s7 := vc.wf_step_misc h rfl hr5 hs5_4 hs5_15
      have hfin := tg_hit_spec rfl hle hwf7 (oracleAdmits_of_reaches hor hrch7) hrs hpcW hlen hvw hvp
        hs5_8 hs5_1 h272 h280 h288
      simpa [SporosProvLen.ltfind, hmatch] using lift_next_outcome hrch7 rfl hfin
    · have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4198668 } := by
        rw [step_at_vc 13 rfl (by decide : (4198666, (.ofInt (.jcc 4 4198675)), 2) ∈ vmTableChunk 13),
          exec_jcc (cond_aluF_sub_e ek k hekW hkW)]
        simp [s6, hmatch]
      let s7 := { s6 with rip := 4198668 }
      have hea24 : eaIdx vc.kvReg.base (idx * 32) 1 24 = vc.kvReg.base + (idx * 32 + 24) := by
        have hltW : vc.kvReg.base + 1 * (idx * 32) + 24 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
        rw [eaIdx_add_eq hltW]; omega
      have hrd_nx : rdMem .w64 s7.mem (eaIdx vc.kvReg.base (idx * 32) 1 24) = some next := by
        have hkv_in : vc.kvReg ∈ s.mem := vc.mem_kv h
        rw [hea24]
        exact rdMem_w64_at h.1.2 hkv_in (Nat.le_add_right _ _)
          (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hnx)
      have h08 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some next) with rip := 4198673 } := by
        rw [step_at_vc 14 rfl (by decide : (4198668, (.ldIdx .w64 0 8 1 1 24), 5) ∈ vmTableChunk 14),
          exec_ldIdx (show s7.reg 8 = some vc.kvReg.base from hs5_8)
            (show s7.reg 1 = some (idx * 32) from hs5_1) hrd_nx]
      let s8 := { s7.set 0 (some next) with rip := 4198673 }
      have hr8' : RegsOk s8 := regsOk_set hr5 (by decide) _
      have hs8_0 : s8.reg 0 = some next := reg_set_self_ok' hr5 (by decide) _
      have hs8_8 : s8.reg 8 = some vc.kvReg.base := by
        change (s7.set 0 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hs5_8
      have hs8_2 : s8.reg 2 = some k := by
        change (s7.set 0 _).reg 2 = some k; rw [reg_set_ne (by decide)]; exact hs5_2
      have hs8_6 : s8.reg 6 = some 18446744073709551615 := by
        change (s7.set 0 _).reg 6 = some 18446744073709551615; rw [reg_set_ne (by decide)]; exact hs5_6
      have hs8_4 : s8.reg 4 = some vc.sp := by
        change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs5_4
      have hs8_15 : s8.reg 15 = some ctxBase := by
        change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
      have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4198647 } := by
        rw [step_at_vc 14 rfl (by decide : (4198673, (.ofInt (.jmp 4198647)), 2) ∈ vmTableChunk 14), exec_jmp]
      let s9 := { s8 with rip := 4198647 }
      have hrch9 : Reaches vmImage s s9 :=
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
          (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
            (Reaches.step h09))))))))
      have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc h rfl hr8' hs8_4 hs8_15
      have hrec := ih rfl hle hwf9 (oracleAdmits_of_reaches hor hrch9) hrs hpcW hkW hs8_0 hs8_8 hs8_2 hs8_6
        h272 h280 h288
      simpa [SporosProvLen.ltfind, hmatch] using lift_next_outcome hrch9 rfl hrec

theorem P_tget_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_tget) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 2)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .tget := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  let k := ls.ds[0]!
  let tb := ls.ds[1]!
  have hkW : k.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega))).1
  obtain ⟨⟨_, _, htb_len, _, htb_hi⟩, _, ⟨hsl128, hsl136, _, hsl152, _, _⟩, htabs_rel, htabs_le⟩ :=
    wf_tabs_info h
  have hrd128 : rd64 s.mem (ea ctxBase 128) = some vc.tabsReg.base := vc.rd64_ctx_of_eq h (by decide) hsl128
  have hrd136 : rd64 s.mem (ea ctxBase 136) = some ls.tabs.length := vc.rd64_ctx_of_eq h (by decide) hsl136
  have hrd152 : rd64 s.mem (ea ctxBase 152) = some vc.kvReg.base := vc.rd64_ctx_of_eq h (by decide) hsl152
  have htake : ls.ds.take 2 = [k, tb] := by
    match hd : ls.ds with
    | a :: b :: _ => simp [k, tb, hd, List.take]
    | [] | [_] => simp [hd] at hle
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some tb.w) with rip := 4198596 } := by
    rw [step_at_vc 13 hrip (by decide : (SporosVmBin.P_tget, (.ofInt (.ld 12 15 312)), 7) ∈ vmTableChunk 13),
      exec_ld h15 h312, hrip]; rfl
  let s1 := { s.set 12 (some tb.w) with rip := 4198596 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_12 : s1.reg 12 = some tb.w := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 12 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.tabs.length) with rip := 4198603 } := by
    rw [step_at_vc 13 rfl (by decide : (4198596, (.ofInt (.ld 1 15 136)), 7) ∈ vmTableChunk 13),
      exec_ld hs1_15 hrd136]
  let s2 := { s1.set 1 (some ls.tabs.length) with rip := 4198603 }
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
      .ok { s2 with fl := (X86Int.aluF .sub tb.w ls.tabs.length).2, rip := 4198606 } := by
    rw [step_at_vc 13 rfl (by decide : (4198603, (.ofInt (.cmp 12 1)), 3) ∈ vmTableChunk 13),
      exec_cmp hs2_12 hs2_1]
  let s3 := { s2 with fl := (X86Int.aluF .sub tb.w ls.tabs.length).2, rip := 4198606 }
  by_cases htb : tb.w < ls.tabs.length
  · have hnotae : ¬ ls.tabs.length ≤ tb.w := by omega
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198612 } := by
      rw [step_at_vc 13 rfl (by decide : (4198606, (.ofInt (.jcc 3 4201184)), 6) ∈ vmTableChunk 13),
        exec_jcc (cond_aluF_sub_ae tb.w ls.tabs.length)]
      simp [s3, hnotae]
    let s4 := { s3 with rip := 4198612 }
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 9 (some vc.tabsReg.base) with rip := 4198619 } := by
      rw [step_at_vc 13 rfl (by decide : (4198612, (.ofInt (.ld 9 15 128)), 7) ∈ vmTableChunk 13),
        exec_ld (show s4.reg 15 = some ctxBase from hs2_15) hrd128]
    let s5 := { s4.set 9 (some vc.tabsReg.base) with rip := 4198619 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_9 : s5.reg 9 = some vc.tabsReg.base := reg_set_self_ok' hr2 (by decide) _
    have hs5_12 : s5.reg 12 = some tb.w := by
      change (s4.set 9 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs2_12
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
    let head := leVal ((vc.tabsReg.data.drop (tb.w * 8)).take 8)
    have hea_tb : eaIdx vc.tabsReg.base tb.w 8 0 = vc.tabsReg.base + tb.w * 8 := by
      have hltW : vc.tabsReg.base + 8 * tb.w + 0 < X86Env.W := by simp only [Region.hi] at htb_hi; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_head : rdMem .w64 s5.mem (eaIdx vc.tabsReg.base tb.w 8 0) = some head := by
      have htb_in : vc.tabsReg ∈ s.mem := vc.mem_tabs h
      rw [hea_tb]
      exact rdMem_w64_at h.1.2 htb_in (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left])
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 0 (some head) with rip := 4198623 } := by
      rw [step_at_vc 13 rfl (by decide : (4198619, (.ldIdx .w64 0 9 12 8 0), 4) ∈ vmTableChunk 13),
        exec_ldIdx hs5_9 hs5_12 hrd_head]
    let s6 := { s5.set 0 (some head) with rip := 4198623 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_0 : s6.reg 0 = some head := reg_set_self_ok' hr5 (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 8 (some vc.kvReg.base) with rip := 4198630 } := by
      rw [step_at_vc 13 rfl (by decide : (4198623, (.ofInt (.ld 8 15 152)), 7) ∈ vmTableChunk 13),
        exec_ld hs6_15 hrd152]
    let s7 := { s6.set 8 (some vc.kvReg.base) with rip := 4198630 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
    have hs7_8 : s7.reg 8 = some vc.kvReg.base := reg_set_self_ok' hr6 (by decide) _
    have hs7_15 : s7.reg 15 = some ctxBase := by
      change (s6.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs6_15
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some k.w) with rip := 4198637 } := by
      rw [step_at_vc 13 rfl (by decide : (4198630, (.ofInt (.ld 2 15 296)), 7) ∈ vmTableChunk 13),
        exec_ld hs7_15 h296]
    let s8 := { s7.set 2 (some k.w) with rip := 4198637 }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
    have hs8_2 : s8.reg 2 = some k.w := reg_set_self_ok' hr7 (by decide) _
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 6 (some 18446744073709551615) with rip := 4198647 } := by
      rw [step_at_vc 13 rfl (by decide : (4198637, (.ofInt (.movI 6 18446744073709551615)), 10) ∈ vmTableChunk 13),
        exec_movI_lt (by omega)]
    let s9 := { s8.set 6 (some 18446744073709551615) with rip := 4198647 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_6 : s9.reg 6 = some 18446744073709551615 := reg_set_self_ok' hr8 (by decide) _
    have hs9_2 : s9.reg 2 = some k.w := by
      change (s8.set 6 _).reg 2 = some k.w; rw [reg_set_ne (by decide)]; exact hs8_2
    have hs9_8 : s9.reg 8 = some vc.kvReg.base := by
      change (s8.set 6 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 8 = some vc.kvReg.base; rw [reg_set_ne (by decide)]; exact hs7_8
    have hs9_0 : s9.reg 0 = some head := by
      change (s8.set 6 _).reg 0 = some head; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 0 = some head; rw [reg_set_ne (by decide)]
      change (s6.set 8 _).reg 0 = some head; rw [reg_set_ne (by decide)]; exact hs6_0
    have hs9_4 : s9.reg 4 = some vc.sp := by
      change (s8.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s6.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s5.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s4.set 9 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s7.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs7_15
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step h09))))))))
    have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc h rfl hr9 hs9_4 hs9_15
    have hloop := tg_loop_spec (htabs_rel.2 tb.w htb) rfl hle hwf9 (oracleAdmits_of_reaches hor hrch9)
      hrs hpcW hkW hs9_0 hs9_8 hs9_2 hs9_6 h272 h280 h288
    have hres := lift_next_outcome hrch9 rfl hloop
    unfold PrimPostOutcome
    have hget_eq : ls.tabs[tb.w]'htb = ls.tabs[tb.w]! := (getElem!_pos ls.tabs tb.w htb).symm
    simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, dif_pos htb, hget_eq]
    rw [htake] at hres
    cases hfind : SporosProvLen.ltfind ls.tabs[tb.w]! k.w <;>
      simpa [← List.getElem!_eq_getElem?_getD, hfind] using hres
  · have hae : ls.tabs.length ≤ tb.w := by omega
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := errEntry .badTable } := by
      rw [step_at_vc 13 rfl (by decide : (4198606, (.ofInt (.jcc 3 4201184)), 6) ∈ vmTableChunk 13),
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
