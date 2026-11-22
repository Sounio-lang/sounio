import Bootstrap.SporosVmPrimTabBase

/-!
# Native x86-64 VM Table Insertion Fast Path (`tp_room_spec`)

Simulation proof over `X86Vm` for the post-capacity-check body of `P_tput`
(starting at `tp_room = 4198488`), writing the 32-byte kv node, updating
`tab_heads[tb]`, incrementing `ctx[160]`, and finishing via `L_prim_epilogue`.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimTab

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

attribute [local instance] instInhabitedLVal instInhabitedLFrame

/-! ### `tp_room` stage 1: compute `nodePtr = kvBase + kvLen * 32` (`4198488..4198512`) -/

private theorem tp_room_addr_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {tbw : Nat}
    (hrip : s.rip = 4198488) (h : vc.wf pr ls s) (hroom : vc.kvLen < vc.kvCap)
    (hr12 : s.reg 12 = some tbw) :
    ∃ s5, Reaches vmImage s s5 ∧ s5.rip = 4198512 ∧ s5.mem = s.mem ∧ s5.k = s.k ∧
      vc.wf pr ls s5 ∧ s5.reg 0 = some vc.kvLen ∧
      s5.reg 8 = some (vc.kvReg.base + vc.kvLen * 32) ∧ s5.reg 12 = some tbw := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  obtain ⟨_, ⟨_, _, hkv_len, _, hkv_hi, _⟩, ⟨_, _, _, hsl152, hsl160, _⟩, _, _⟩ := wf_tabs_info h
  have hrd152 : rd64 s.mem (ea ctxBase 152) = some vc.kvReg.base := vc.rd64_ctx_of_eq h (by decide) hsl152
  have hrd160 : rd64 s.mem (ea ctxBase 160) = some vc.kvLen := vc.rd64_ctx_of_eq h (by decide) hsl160
  have hkv_hi_W : vc.kvReg.base + vc.kvLen * 32 + 32 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some vc.kvLen) with rip := 4198495 } := by
    rw [step_at_vc 13 hrip (by decide : (4198488, (.ofInt (.ld 0 15 160)), 7) ∈ vmTableChunk 13),
      exec_ld h15 hrd160, hrip]
  let s1 := { s.set 0 (some vc.kvLen) with rip := 4198495 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_0 : s1.reg 0 = some vc.kvLen := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some vc.kvLen) with rip := 4198498 } := by
    rw [step_at_vc 13 rfl (by decide : (4198495, (.ofInt (.mov 2 0)), 3) ∈ vmTableChunk 13), exec_mov hs1_0]
  let s2 := { s1.set 2 (some vc.kvLen) with rip := 4198498 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_2 : s2.reg 2 = some vc.kvLen := reg_set_self_ok' hr1 (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 2 (some (vc.kvLen * 32)) with fl := X86Int.Flags.undef, rip := 4198502 } := by
    rw [step_at_vc 13 rfl (by decide : (4198498, (.ofInt (.shlI 2 5)), 4) ∈ vmTableChunk 13),
      exec_shlI_lt (c := 5) hs2_2 (by rw [show 2 ^ (5 % 64) = 32 from by decide]; omega)]
  let s3 := { s2.set 2 (some (vc.kvLen * 32)) with fl := X86Int.Flags.undef, rip := 4198502 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_2 : s3.reg 2 = some (vc.kvLen * 32) := reg_set_self_ok' hr2 (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 8 (some vc.kvReg.base) with rip := 4198509 } := by
    rw [step_at_vc 13 rfl (by decide : (4198502, (.ofInt (.ld 8 15 152)), 7) ∈ vmTableChunk 13),
      exec_ld hs3_15 hrd152]
  let s4 := { s3.set 8 (some vc.kvReg.base) with rip := 4198509 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_8 : s4.reg 8 = some vc.kvReg.base := reg_set_self_ok' hr3 (by decide) _
  have hs4_2 : s4.reg 2 = some (vc.kvLen * 32) := by
    change (s3.set 8 _).reg 2 = some (vc.kvLen * 32); rw [reg_set_ne (by decide)]; exact hs3_2
  let nodePtr := vc.kvReg.base + vc.kvLen * 32
  have hadd_ptr : X86Int.aluF .add vc.kvReg.base (vc.kvLen * 32) =
      (nodePtr, (X86Int.aluF .add vc.kvReg.base (vc.kvLen * 32)).2) :=
    Prod.ext (by rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]) rfl
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4.set 8 (some nodePtr) with fl := (X86Int.aluF .add vc.kvReg.base (vc.kvLen * 32)).2, rip := 4198512 } := by
    rw [step_at_vc 13 rfl (by decide : (4198509, (.ofInt (.alu .add 8 2)), 3) ∈ vmTableChunk 13),
      exec_alu hs4_8 hs4_2, hadd_ptr]
  let s5 := { s4.set 8 (some nodePtr) with fl := (X86Int.aluF .add vc.kvReg.base (vc.kvLen * 32)).2, rip := 4198512 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
  have hs5_8 : s5.reg 8 = some nodePtr := reg_set_self_ok' hr4 (by decide) _
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s3.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
  have hs5_0 : s5.reg 0 = some vc.kvLen := by
    change (s4.set 8 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s3.set 8 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s2.set 2 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs5_12 : s5.reg 12 = some tbw := by
    change (s4.set 8 _).reg 12 = some tbw; rw [reg_set_ne (by decide)]
    change (s3.set 8 _).reg 12 = some tbw; rw [reg_set_ne (by decide)]
    change (s2.set 2 _).reg 12 = some tbw; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 12 = some tbw; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 12 = some tbw; rw [reg_set_ne (by decide)]; exact hr12
  have hs5_4 : s5.reg 4 = some vc.sp := by
    change (s4.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s3.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s2.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  exact ⟨s5,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step h05)))),
    rfl, rfl, rfl, vc.wf_step_misc h rfl hr5 hs5_4 hs5_15, hs5_0, hs5_8, hs5_12⟩

/-! ### `tp_room` stage 2: store `k`, `v.w`, `v.plen + q` into `kv[kvLen]` (`4198512..4198554`) -/

private theorem tp_room_write_kv_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St}
    (hrip : s.rip = 4198512) (h : vc.wf pr ls s) (hroom : vc.kvLen < vc.kvCap)
    (hr0 : s.reg 0 = some vc.kvLen)
    (hr8 : s.reg 8 = some (vc.kvReg.base + vc.kvLen * 32))
    (hr12 : s.reg 12 = some ls.ds[2]!.w)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    let v := ls.ds[0]!
    let k := ls.ds[1]!
    let vp := (v.plen + SporosProvLen.lq (ls.ds.take 3) ls.pc) % X86Env.W
    let d1 := writeBytes vc.kvReg.data (vc.kvLen * 32) (bytes8 k.w)
    let d2 := writeBytes d1 (vc.kvLen * 32 + 8) (bytes8 v.w)
    let d3 := writeBytes d2 (vc.kvLen * 32 + 16) (bytes8 vp)
    ∃ s8, Reaches vmImage s s8 ∧ s8.rip = 4198554 ∧ s8.k = s.k ∧ s8.mem.wf ∧ RegsOk s8 ∧
      s8.reg 0 = some vc.kvLen ∧ s8.reg 4 = some vc.sp ∧
      s8.reg 8 = some (vc.kvReg.base + vc.kvLen * 32) ∧ s8.reg 12 = some ls.ds[2]!.w ∧
      s8.reg 15 = some ctxBase ∧ { vc.kvReg with data := d3 } ∈ s8.mem ∧
      (∀ u ∈ s.mem, u.base ≠ vc.kvReg.base → u ∈ s8.mem) := by
  intro v k vp d1 d2 d3
  let kv1 : Region := { vc.kvReg with data := d1 }
  let kv2 : Region := { vc.kvReg with data := d2 }
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  let q := SporosProvLen.lq (ls.ds.take 3) ls.pc
  let nodePtr := vc.kvReg.base + vc.kvLen * 32
  obtain ⟨_, ⟨hkv_rw, _, hkv_len, _, hkv_hi, _⟩, _, _, _⟩ := wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hkv_hi_W : vc.kvReg.base + vc.kvLen * 32 + 32 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
  have hne_ctx_kv : vc.ctxReg.base ≠ vc.kvReg.base :=
    vc.ne_bases h (i := 2) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some k.w) with rip := 4198519 } := by
    rw [step_at_vc 13 hrip (by decide : (4198512, (.ofInt (.ld 1 15 312)), 7) ∈ vmTableChunk 13),
      exec_ld h15 h312, hrip]
  let s1 := { s.set 1 (some k.w) with rip := 4198519 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_1 : s1.reg 1 = some k.w := reg_set_self_ok' hr (by decide) _
  have hs1_8 : s1.reg 8 = some nodePtr := by
    change (s.set 1 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]; exact hr8
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  obtain ⟨_, m2, hwr2, _, _, hmwf2, hkv2_mem, hoth2⟩ :=
    wr64_region h.1.2 (vc.mem_kv h) hkv_rw (a := nodePtr)
      (by dsimp [nodePtr]; omega) (by simp only [Region.hi, nodePtr]; omega) (v := k.w)
  have hwr2' : wr64 s1.mem (ea nodePtr 0) k.w = some m2 := by rw [ea_add_eq (by omega)]; simpa [s1] using hwr2
  have h02 : X86Vm.step vmImage s1 = .ok { s1 with mem := m2, rip := 4198522 } := by
    rw [step_at_vc 13 rfl (by decide : (4198519, (.ofInt (.st 8 0 1)), 3) ∈ vmTableChunk 13),
      exec_st hs1_8 hs1_1 hwr2']
  let s2 := { s1 with mem := m2, rip := 4198522 }
  have hsub0 : nodePtr - vc.kvReg.base = vc.kvLen * 32 := by dsimp [nodePtr]; omega
  rw [hsub0] at hkv2_mem
  have hctx2 : vc.ctxReg ∈ m2 := hoth2 vc.ctxReg (vc.mem_ctx h) hne_ctx_kv
  have hrd_ctx2 : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      rd64 m2 (ea ctxBase m0) = some u := by
    intro m0 u hlim hrd0
    rw [ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)]
    refine rd64_of_mem hmwf2 hctx2 (by rw [hctx_b.1]; omega)
      (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega) ?_
    rw [hctx_b.1, Nat.add_sub_cancel_left]; exact ctx_leVal_of_rd64 h hlim hrd0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some v.w) with rip := 4198529 } := by
    rw [step_at_vc 13 rfl (by decide : (4198522, (.ofInt (.ld 1 15 296)), 7) ∈ vmTableChunk 13),
      exec_ld (show s2.reg 15 = some ctxBase from hs1_15) (hrd_ctx2 296 v.w (by decide) h296)]
  let s3 := { s2.set 1 (some v.w) with rip := 4198529 }
  have hr3 : RegsOk s3 := regsOk_set hr1 (by decide) _
  have hs3_1 : s3.reg 1 = some v.w := reg_set_self_ok' hr1 (by decide) _
  have hs3_8 : s3.reg 8 = some nodePtr := by
    change (s2.set 1 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]; exact hs1_8
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  obtain ⟨_, m4, hwr4, _, _, hmwf4, hkv4_mem, hoth4⟩ :=
    wr64_region hmwf2 hkv2_mem hkv_rw (a := nodePtr + 8)
      (by dsimp [kv1, nodePtr]; omega) (by simp only [Region.hi, nodePtr, writeBytes_length]; omega) (v := v.w)
  have hwr4' : wr64 s3.mem (ea nodePtr 8) v.w = some m4 := by rw [ea_add_eq (by omega)]; exact hwr4
  have h04 : X86Vm.step vmImage s3 = .ok { s3 with mem := m4, rip := 4198533 } := by
    rw [step_at_vc 13 rfl (by decide : (4198529, (.ofInt (.st 8 8 1)), 4) ∈ vmTableChunk 13),
      exec_st hs3_8 hs3_1 hwr4']
  let s4 := { s3 with mem := m4, rip := 4198533 }
  have hsub8 : nodePtr + 8 - kv1.base = vc.kvLen * 32 + 8 := by dsimp [kv1, nodePtr]; omega
  rw [hsub8] at hkv4_mem
  have hctx4 : vc.ctxReg ∈ m4 := hoth4 vc.ctxReg hctx2 hne_ctx_kv
  have hrd_ctx4 : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      rd64 m4 (ea ctxBase m0) = some u := by
    intro m0 u hlim hrd0
    rw [ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)]
    refine rd64_of_mem hmwf4 hctx4 (by rw [hctx_b.1]; omega)
      (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega) ?_
    rw [hctx_b.1, Nat.add_sub_cancel_left]; exact ctx_leVal_of_rd64 h hlim hrd0
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 1 (some v.plen) with rip := 4198540 } := by
    rw [step_at_vc 13 rfl (by decide : (4198533, (.ofInt (.ld 1 15 304)), 7) ∈ vmTableChunk 13),
      exec_ld (show s4.reg 15 = some ctxBase from hs3_15) (hrd_ctx4 304 v.plen (by decide) h304)]
  let s5 := { s4.set 1 (some v.plen) with rip := 4198540 }
  have hr5 : RegsOk s5 := regsOk_set hr3 (by decide) _
  have hs5_1 : s5.reg 1 = some v.plen := reg_set_self_ok' hr3 (by decide) _
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 2 (some q) with rip := 4198547 } := by
    rw [step_at_vc 13 rfl (by decide : (4198540, (.ofInt (.ld 2 15 272)), 7) ∈ vmTableChunk 13),
      exec_ld hs5_15 (hrd_ctx4 272 q (by decide) h272)]
  let s6 := { s5.set 2 (some q) with rip := 4198547 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
  have hs6_2 : s6.reg 2 = some q := reg_set_self_ok' hr5 (by decide) _
  have hs6_1 : s6.reg 1 = some v.plen := by
    change (s5.set 2 _).reg 1 = some v.plen; rw [reg_set_ne (by decide)]; exact hs5_1
  have hadd_vp : X86Int.aluF .add v.plen q = (vp, (X86Int.aluF .add v.plen q).2) :=
    Prod.ext (aluF_add_val _ _) rfl
  have h07 : X86Vm.step vmImage s6 =
      .ok { s6.set 1 (some vp) with fl := (X86Int.aluF .add v.plen q).2, rip := 4198550 } := by
    rw [step_at_vc 13 rfl (by decide : (4198547, (.ofInt (.alu .add 1 2)), 3) ∈ vmTableChunk 13),
      exec_alu hs6_1 hs6_2, hadd_vp]
  let s7 := { s6.set 1 (some vp) with fl := (X86Int.aluF .add v.plen q).2, rip := 4198550 }
  have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
  have hs7_1 : s7.reg 1 = some vp := reg_set_self_ok' hr6 (by decide) _
  have hs7_8 : s7.reg 8 = some nodePtr := by
    change (s6.set 1 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]
    change (s5.set 2 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]
    change (s4.set 1 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]; exact hs3_8
  have hs7_15 : s7.reg 15 = some ctxBase := by
    change (s6.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s5.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
  have hs7_0 : s7.reg 0 = some vc.kvLen := by
    change (s6.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s5.set 2 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s4.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]; exact hr0
  have hs7_12 : s7.reg 12 = some ls.ds[2]!.w := by
    change (s6.set 1 _).reg 12 = some ls.ds[2]!.w; rw [reg_set_ne (by decide)]
    change (s5.set 2 _).reg 12 = some ls.ds[2]!.w; rw [reg_set_ne (by decide)]
    change (s4.set 1 _).reg 12 = some ls.ds[2]!.w; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 12 = some ls.ds[2]!.w; rw [reg_set_ne (by decide)]
    change (s.set 1 _).reg 12 = some ls.ds[2]!.w; rw [reg_set_ne (by decide)]; exact hr12
  have hs7_4 : s7.reg 4 = some vc.sp := by
    change (s6.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s5.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s4.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  obtain ⟨_, m8, hwr8, _, _, hmwf8, hkv8_mem, hoth8⟩ :=
    wr64_region hmwf4 hkv4_mem hkv_rw (a := nodePtr + 16)
      (by dsimp [kv2, kv1, nodePtr]; omega)
      (by simp only [Region.hi, nodePtr, writeBytes_length]; omega) (v := vp)
  have hwr8' : wr64 s7.mem (ea nodePtr 16) vp = some m8 := by rw [ea_add_eq (by omega)]; exact hwr8
  have h08 : X86Vm.step vmImage s7 = .ok { s7 with mem := m8, rip := 4198554 } := by
    rw [step_at_vc 13 rfl (by decide : (4198550, (.ofInt (.st 8 16 1)), 4) ∈ vmTableChunk 13),
      exec_st hs7_8 hs7_1 hwr8']
  let s8 := { s7 with mem := m8, rip := 4198554 }
  have hsub16 : nodePtr + 16 - kv2.base = vc.kvLen * 32 + 16 := by dsimp [kv2, kv1, nodePtr]; omega
  rw [hsub16] at hkv8_mem
  refine ⟨s8,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08))))))),
    rfl, rfl, hmwf8, hr7, hs7_0, hs7_4, hs7_8, hs7_12, hs7_15, hkv8_mem, ?_⟩
  intro u hu hne
  exact hoth8 u (hoth4 u (hoth2 u hu hne) hne) hne

/-! ### `tp_room` stage 3: link `oldHead`, update `tab_heads[tb]` and `ctx[160]` (`4198554..4198584`) -/

private theorem tp_room_link_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s0 s13 : X86Vm.St} {d3 : List Nat}
    (h : vc.wf pr ls s0) (htb : ls.ds[2]!.w < ls.tabs.length) (hroom : vc.kvLen < vc.kvCap)
    (hd3_len : d3.length = vc.kvCap * 32)
    (hrip13 : s13.rip = 4198554) (hmwf13 : s13.mem.wf) (hr13 : RegsOk s13)
    (hs13_0 : s13.reg 0 = some vc.kvLen) (hs13_4 : s13.reg 4 = some vc.sp)
    (hs13_8 : s13.reg 8 = some (vc.kvReg.base + vc.kvLen * 32))
    (hs13_12 : s13.reg 12 = some ls.ds[2]!.w) (hs13_15 : s13.reg 15 = some ctxBase)
    (hkv13_mem : { vc.kvReg with data := d3 } ∈ s13.mem)
    (hoth13 : ∀ u ∈ s0.mem, u.base ≠ vc.kvReg.base → u ∈ s13.mem) :
    let tb := ls.ds[2]!
    let oldHead := leVal ((vc.tabsReg.data.drop (tb.w * 8)).take 8)
    let kv4 : Region := { vc.kvReg with data := writeBytes d3 (vc.kvLen * 32 + 24) (bytes8 oldHead) }
    let tabsReg' : Region := { vc.tabsReg with data := writeBytes vc.tabsReg.data (tb.w * 8) (bytes8 vc.kvLen) }
    let ctx' : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 160 (bytes8 (vc.kvLen + 1)) }
    ∃ s19, Reaches vmImage s13 s19 ∧ s19.rip = 4198584 ∧ s19.k = s13.k ∧ s19.mem.wf ∧
      RegsOk s19 ∧ s19.reg 4 = some vc.sp ∧ s19.reg 15 = some ctxBase ∧
      kv4 ∈ s19.mem ∧ tabsReg' ∈ s19.mem ∧ ctx' ∈ s19.mem ∧ vc.stk ∈ s19.mem ∧
      (∀ u ∈ s0.mem, u.base ≠ vc.kvReg.base → u.base ≠ vc.tabsReg.base →
        u.base ≠ vc.ctxReg.base → u ∈ s19.mem) := by
  intro tb oldHead kv4 tabsReg' ctx'
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  let nodePtr := vc.kvReg.base + vc.kvLen * 32
  let kv3 : Region := { vc.kvReg with data := d3 }
  obtain ⟨⟨htb_rw, _, htb_len, _, htb_hi⟩, ⟨hkv_rw, _, hkv_len, _, hkv_hi, _⟩,
    ⟨hsl128, _, _, _, _, _⟩, _, _⟩ := wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hkv_hi_W : vc.kvReg.base + vc.kvLen * 32 + 32 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
  have hne_ctx_kv : vc.ctxReg.base ≠ vc.kvReg.base :=
    vc.ne_bases h (i := 2) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_tb_kv : vc.tabsReg.base ≠ vc.kvReg.base :=
    vc.ne_bases h (i := 12) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_ctx_tb : vc.ctxReg.base ≠ vc.tabsReg.base :=
    vc.ne_bases h (i := 2) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_kv : vc.stk.base ≠ vc.kvReg.base :=
    vc.ne_bases h (i := 1) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_tb : vc.stk.base ≠ vc.tabsReg.base :=
    vc.ne_bases h (i := 1) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ctx : vc.stk.base ≠ vc.ctxReg.base :=
    vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hctx13 : vc.ctxReg ∈ s13.mem := hoth13 vc.ctxReg (vc.mem_ctx h) hne_ctx_kv
  have htb13 : vc.tabsReg ∈ s13.mem := hoth13 vc.tabsReg (vc.mem_tabs h) hne_tb_kv
  have hrd128_13 : rd64 s13.mem (ea ctxBase 128) = some vc.tabsReg.base := by
    rw [ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)]
    refine rd64_of_mem hmwf13 hctx13 (by rw [hctx_b.1]; omega)
      (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega) ?_
    rw [hctx_b.1, Nat.add_sub_cancel_left]; exact hsl128
  have h14 : X86Vm.step vmImage s13 = .ok { s13.set 9 (some vc.tabsReg.base) with rip := 4198561 } := by
    rw [step_at_vc 13 hrip13 (by decide : (4198554, (.ofInt (.ld 9 15 128)), 7) ∈ vmTableChunk 13),
      exec_ld hs13_15 hrd128_13, hrip13]
  let s14 := { s13.set 9 (some vc.tabsReg.base) with rip := 4198561 }
  have hr14 : RegsOk s14 := regsOk_set hr13 (by decide) _
  have hs14_9 : s14.reg 9 = some vc.tabsReg.base := reg_set_self_ok' hr13 (by decide) _
  have hs14_12 : s14.reg 12 = some tb.w := by
    change (s13.set 9 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs13_12
  have htb' : tb.w < ls.tabs.length := htb
  have hea_tb : eaIdx vc.tabsReg.base tb.w 8 0 = vc.tabsReg.base + tb.w * 8 := by
    have hltW : vc.tabsReg.base + 8 * tb.w + 0 < X86Env.W := by simp only [Region.hi] at htb_hi; omega
    rw [eaIdx_add_eq hltW]; omega
  have hrd_head : rdMem .w64 s14.mem (eaIdx vc.tabsReg.base tb.w 8 0) = some oldHead := by
    rw [hea_tb]
    exact rdMem_w64_at hmwf13 htb13 (Nat.le_add_right _ _) (by simp only [Region.hi]; omega)
      (by rw [Nat.add_sub_cancel_left])
  have h15s : X86Vm.step vmImage s14 = .ok { s14.set 1 (some oldHead) with rip := 4198565 } := by
    rw [step_at_vc 13 rfl (by decide : (4198561, (.ldIdx .w64 1 9 12 8 0), 4) ∈ vmTableChunk 13),
      exec_ldIdx hs14_9 hs14_12 hrd_head]
  let s15 := { s14.set 1 (some oldHead) with rip := 4198565 }
  have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) _
  have hs15_1 : s15.reg 1 = some oldHead := reg_set_self_ok' hr14 (by decide) _
  have hs15_8 : s15.reg 8 = some nodePtr := by
    change (s14.set 1 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]
    change (s13.set 9 _).reg 8 = some nodePtr; rw [reg_set_ne (by decide)]; exact hs13_8
  have hs15_9 : s15.reg 9 = some vc.tabsReg.base := by
    change (s14.set 1 _).reg 9 = some vc.tabsReg.base; rw [reg_set_ne (by decide)]; exact hs14_9
  have hs15_12 : s15.reg 12 = some tb.w := by
    change (s14.set 1 _).reg 12 = some tb.w; rw [reg_set_ne (by decide)]; exact hs14_12
  have hs15_0 : s15.reg 0 = some vc.kvLen := by
    change (s14.set 1 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]
    change (s13.set 9 _).reg 0 = some vc.kvLen; rw [reg_set_ne (by decide)]; exact hs13_0
  obtain ⟨_, m16, hwr16, _, _, hmwf16, hkv16_mem, hoth16⟩ :=
    wr64_region hmwf13 hkv13_mem hkv_rw (a := nodePtr + 24)
      (by dsimp [kv3, nodePtr]; omega)
      (by simp only [Region.hi, nodePtr, hd3_len]; omega) (v := oldHead)
  have hwr16' : wr64 s15.mem (ea nodePtr 24) oldHead = some m16 := by rw [ea_add_eq (by omega)]; exact hwr16
  have h16 : X86Vm.step vmImage s15 = .ok { s15 with mem := m16, rip := 4198569 } := by
    rw [step_at_vc 13 rfl (by decide : (4198565, (.ofInt (.st 8 24 1)), 4) ∈ vmTableChunk 13),
      exec_st hs15_8 hs15_1 hwr16']
  let s16 := { s15 with mem := m16, rip := 4198569 }
  have hsub24 : nodePtr + 24 - kv3.base = vc.kvLen * 32 + 24 := by dsimp [kv3, nodePtr]; omega
  rw [hsub24] at hkv16_mem
  have htb16 : vc.tabsReg ∈ m16 := hoth16 vc.tabsReg htb13 hne_tb_kv
  obtain ⟨_, m17, hwr17, _, _, hmwf17, htb17_mem, hoth17⟩ :=
    wrMem_w64_region hmwf16 htb16 htb_rw (a := vc.tabsReg.base + tb.w * 8)
      (Nat.le_add_right _ _) (by simp only [Region.hi]; omega) (v := vc.kvLen)
  have hwr17' : wrMem .w64 s16.mem (eaIdx vc.tabsReg.base tb.w 8 0) vc.kvLen = some m17 := by
    rw [hea_tb]; exact hwr17
  have h17 : X86Vm.step vmImage s16 = .ok { s16 with mem := m17, rip := 4198573 } := by
    rw [step_at_vc 13 rfl (by decide : (4198569, (.stIdx .w64 9 12 8 0 0), 4) ∈ vmTableChunk 13),
      exec_stIdx (show s16.reg 9 = some vc.tabsReg.base from hs15_9)
        (show s16.reg 12 = some tb.w from hs15_12)
        (show s16.reg 0 = some vc.kvLen from hs15_0) hwr17']
  let s17 := { s16 with mem := m17, rip := 4198573 }
  have hkvLen1W : vc.kvLen + 1 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
  have h18 : X86Vm.step vmImage s17 =
      .ok { s17.set 0 (some (vc.kvLen + 1)) with fl := (X86Int.aluF .add vc.kvLen 1).2, rip := 4198577 } := by
    rw [step_at_vc 13 rfl (by decide : (4198573, (.ofInt (.aluI .add 0 1)), 4) ∈ vmTableChunk 13),
      exec_aluI_add_lt (show s17.reg 0 = some vc.kvLen from hs15_0) (by omega), Nat.mod_eq_of_lt hkvLen1W]
  let s18 := { s17.set 0 (some (vc.kvLen + 1)) with fl := (X86Int.aluF .add vc.kvLen 1).2, rip := 4198577 }
  have hr18 : RegsOk s18 := regsOk_set hr15 (by decide) _
  have hs18_0 : s18.reg 0 = some (vc.kvLen + 1) := reg_set_self_ok' hr15 (by decide) _
  have hs18_4 : s18.reg 4 = some vc.sp := by
    change (s17.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s14.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s13.set 9 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs13_4
  have hs18_15 : s18.reg 15 = some ctxBase := by
    change (s17.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s14.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s13.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs13_15
  have hctx17 : vc.ctxReg ∈ m17 := hoth17 vc.ctxReg (hoth16 vc.ctxReg hctx13 hne_ctx_kv) hne_ctx_tb
  have hea160 : ea ctxBase 160 = ctxBase + 160 := ea_add_eq (by have := hctx_b.2.2.2.2.1; omega)
  obtain ⟨_, m19, hwr19, _, _, hmwf19, hctx19_mem, hoth19⟩ :=
    wr64_region hmwf17 hctx17 hctx_b.2.1 (a := ctxBase + 160)
      (by rw [hctx_b.1]; omega) (by simp only [Region.hi, hctx_b.1]; have := hctx_b.2.2.1; omega)
      (v := vc.kvLen + 1)
  have hwr19' : wr64 s18.mem (ea ctxBase 160) (vc.kvLen + 1) = some m19 := by rw [hea160]; exact hwr19
  have h19 : X86Vm.step vmImage s18 = .ok { s18 with mem := m19, rip := 4198584 } := by
    rw [step_at_vc 13 rfl (by decide : (4198577, (.ofInt (.st 15 160 0)), 7) ∈ vmTableChunk 13),
      exec_st hs18_15 hs18_0 hwr19']
  let s19 := { s18 with mem := m19, rip := 4198584 }
  have hsub_tb : vc.tabsReg.base + tb.w * 8 - vc.tabsReg.base = tb.w * 8 := by omega
  have hsub_ctx : ctxBase + 160 - vc.ctxReg.base = 160 := by rw [hctx_b.1]; omega
  rw [hsub_tb] at htb17_mem; rw [hsub_ctx] at hctx19_mem
  refine ⟨s19,
    Reaches.step_trans h14 (Reaches.step_trans h15s (Reaches.step_trans h16
      (Reaches.step_trans h17 (Reaches.step_trans h18 (Reaches.step h19))))),
    rfl, rfl, hmwf19, hr18, hs18_4, hs18_15,
    hoth19 kv4 (hoth17 kv4 hkv16_mem (Ne.symm hne_tb_kv)) (Ne.symm hne_ctx_kv),
    hoth19 tabsReg' htb17_mem (Ne.symm hne_ctx_tb), hctx19_mem,
    hoth19 vc.stk (hoth17 vc.stk (hoth16 vc.stk (hoth13 vc.stk (vc.mem_stk h) hne_stk_kv)
      hne_stk_kv) hne_stk_tb) hne_stk_ctx, ?_⟩
  intro u hu h1 h2 h3
  exact hoth19 u (hoth17 u (hoth16 u (hoth13 u hu h1) h1) h2) h3

/-! ### `tp_room` full spec (`4198488..4198589`) -/

theorem tp_room_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = 4198488) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (htb : ls.ds[2]!.w < ls.tabs.length) (hroom : vc.kvLen < vc.kvCap)
    (hr12 : s.reg 12 = some ls.ds[2]!.w)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .tput := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcs : ctxSize = 4096 := rfl
  let v := ls.ds[0]!
  let k := ls.ds[1]!
  let tb := ls.ds[2]!
  let q := SporosProvLen.lq (ls.ds.take 3) ls.pc
  let vp := (v.plen + q) % X86Env.W
  have hvW : v.w < X86Env.W ∧ v.plen < X86Env.W := wf_ds_lt_W h (getElem!_mem_of_lt (by omega))
  have hkW : k.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega))).1
  obtain ⟨⟨htb_rw, _, _, _, _⟩, ⟨hkv_rw, hkv_pos, hkv_len, hkv_pg, hkv_hi, _⟩,
    ⟨_, _, _, hsl152, _, hsl168⟩, htabs_rel, _⟩ := wf_tabs_info h
  have hctx_b := vc.ctx_bounds h
  have hkvLen1W : vc.kvLen + 1 < X86Env.W := by simp only [Region.hi] at hkv_hi; omega
  obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hwf5, hs5_0, hs5_8, hs5_12⟩ :=
    tp_room_addr_spec hrip h hroom hr12
  obtain ⟨s13, hrch5_13, hrip13, hk13, hmwf13, hr13, hs13_0, hs13_4, hs13_8, hs13_12, hs13_15,
      hkv13_mem, hoth13⟩ :=
    tp_room_write_kv_spec hrip5 hwf5 hroom hs5_0 hs5_8 hs5_12
      (hmem5 ▸ h272) (hmem5 ▸ h296) (hmem5 ▸ h304) (hmem5 ▸ h312)
  let d1 := writeBytes vc.kvReg.data (vc.kvLen * 32) (bytes8 k.w)
  let d2 := writeBytes d1 (vc.kvLen * 32 + 8) (bytes8 v.w)
  let d3 := writeBytes d2 (vc.kvLen * 32 + 16) (bytes8 vp)
  have hd3_len : d3.length = vc.kvCap * 32 := by
    dsimp only [d3, d2, d1]; simp only [writeBytes_length]; exact hkv_len
  obtain ⟨s19, hrch13_19, hrip19, hk19', hmwf19, hr19, hs19_4, hs19_15,
      hkv19_mem, htb19_mem, hctx19_mem, hstk19_mem, hoth19⟩ :=
    tp_room_link_spec h htb hroom hd3_len hrip13 hmwf13 hr13 hs13_0 hs13_4 hs13_8 hs13_12 hs13_15
      hkv13_mem (fun u hu hne => hoth13 u (hmem5 ▸ hu) hne)
  have hrch19 : Reaches vmImage s s19 := (hrch5.trans hrch5_13).trans hrch13_19
  have hk19 : s19.k = s.k := by rw [hk19', hk13, hk5]
  have ⟨hbl19, hpos19⟩ := reaches_bytesLt_len_pos h.2.1 h.2.2.1 hrch19
  let oldHead := leVal ((vc.tabsReg.data.drop (tb.w * 8)).take 8)
  let kv4 : Region := { vc.kvReg with data := writeBytes d3 (vc.kvLen * 32 + 24) (bytes8 oldHead) }
  let ctx' : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 160 (bytes8 (vc.kvLen + 1)) }
  have holdW : oldHead < X86Env.W :=
    leVal_take8_lt_W fun b hb => h.2.1 vc.tabsReg (vc.mem_tabs h) b (List.mem_of_mem_drop hb)
  have hvpW : vp < X86Env.W := lt_W_of_mod _
  let tabs' := ls.tabs.set tb.w ((k.w, ⟨v.w, vp⟩) :: ls.tabs[tb.w]!)
  have hwlt_tabs' : ∀ t ∈ tabs', ∀ kv ∈ t, kv.2.w < Sporos.M ∧ kv.2.plen < Sporos.M := by
    intro t ht kv hkv
    rcases mem_set_mem_cases ht with ht0 | rfl
    · exact (vc.code_flat h).2.2.2.2.2.2.2.2.1 t ht0 kv hkv
    · simp only [List.mem_cons] at hkv
      rcases hkv with rfl | hkv
      · exact ⟨by rw [M_eq_W']; exact hvW.1, by rw [M_eq_W']; exact hvpW⟩
      · exact (vc.code_flat h).2.2.2.2.2.2.2.2.1 ls.tabs[tb.w]! (getElem!_mem_of_lt htb) kv hkv
  have hc152' : leVal ((ctx'.data.drop 152).take 8) = kv4.base := by
    dsimp only [ctx', kv4]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by have := hctx_b.2.2.1; omega), hsl152]
  have hc160' : leVal ((ctx'.data.drop 160).take 8) = vc.kvLen + 1 := by
    dsimp only [ctx']
    rw [leVal_writeBytes_same _ _ _ (by have := hctx_b.2.2.1; omega), Nat.mod_eq_of_lt hkvLen1W]
  have hc168' : leVal ((ctx'.data.drop 168).take 8) = vc.kvCap := by
    dsimp only [ctx']
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide))
      (by have := hctx_b.2.2.1; omega), hsl168]
  have hctx_oth' : ∀ j, (j < 152 ∨ 176 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]? := by
    intro j hj; exact writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)
  have hkv4_len : kv4.data.length = vc.kvCap * 32 := by
    dsimp only [kv4]; rw [writeBytes_length]; exact hd3_len
  have htabs_rel' := tabsRel_cons_entry (k := k.w) (vw := v.w) (vp := vp) (oldHead := oldHead)
    htabs_rel htb (by omega) hkW hvW.1 hvpW holdW (by omega) rfl
  let tabsReg' : Region := { vc.tabsReg with data := writeBytes vc.tabsReg.data (tb.w * 8) (bytes8 vc.kvLen) }
  have hkv4_hi : kv4.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, kv4, writeBytes_length, hd3_len, hkv_len] using hkv_hi
  have hupd := wf_update_kv_tabs (tabsReg' := tabsReg') (kvReg' := kv4) (ctx' := ctx') (stk' := vc.stk)
    (kvLen' := vc.kvLen + 1) (kvCap' := vc.kvCap) (tabs' := tabs')
    h hmwf19 hbl19 hpos19 hr19 hs19_4 hs19_15
    htb19_mem rfl htb_rw (writeBytes_length _ _ _)
    hkv19_mem hkv_rw hkv_pos hkv4_len hkv_pg hkv4_hi (by omega)
    hctx19_mem rfl rfl (writeBytes_length _ _ _) hc152' hc160' hc168' hctx_oth'
    hstk19_mem rfl rfl rfl (fun _ _ hne => hne)
    (fun u hu h1 h2 h3 _ => hoth19 u hu h1 h2 h3)
    (List.length_set ..) (by simpa only [tabsReg', kv4, d3, d2, d1, tabs'] using htabs_rel') hwlt_tabs'
  rcases hupd with ⟨hwf19, hpres19⟩
  have hfin := jmp_epilogue0_finish_spec (outs := [])
    (step_at_vc 13 hrip19 (by decide : (4198584, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 13))
    hwf19 (oracleAdmits_of_reaches hor hrch19) hrs hpcW hle (by decide) (by intro _ hv; cases hv)
    (hpres19 280 3 (Or.inr (by decide)) (by decide) h280)
    (hpres19 288 0 (Or.inr (by decide)) (by decide) h288) (by intro _ hi; cases hi)
  have hres := lift_next_outcome hrch19 hk19 hfin
  unfold PrimPostOutcome
  have htake : ls.ds.take 3 = [v, k, tb] := by
    match hd : ls.ds with
    | a :: b :: c :: _ => simp [v, k, tb, hd, List.take]
    | [] | [_] | [_, _] => simp [hd] at hle
  have hexec : SporosProvLen.lexecPrim .tput (ls.ds.take Sporos.Prim.tput.eff.1) ls =
      .ok [] { ls with tabs := tabs' } := by
    have htb' : tb.w < ls.tabs.length := htb
    have hget_eq : ls.tabs[tb.w]'htb' = ls.tabs[tb.w]! := (getElem!_pos ls.tabs tb.w htb').symm
    simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, dif_pos htb', hget_eq, tabs', vp, M_eq_W', q]
  rw [hexec]
  simpa [Sporos.Prim.eff] using hres

end SporosVmPrimTab
