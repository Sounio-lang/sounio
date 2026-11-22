import Bootstrap.SporosVmOpsCall2

/-!
# Native x86-64 VM Call Instruction Proof (`O_call`)

Simulation proof over `X86Vm` for `Sporos.Instr.call w` (`O_call`, `0x4017bc`):
3-way voted metadata loads (`nw`, `nd`, `wd.ins`, `wd.locals`), bounds/underflow
checks, return-stack and locals-arena growth, caller PC advance, zero-initialized
callee locals (`4200816: xor rbx, rbx; mov r11, rbx`), and frame push.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmOpsCtrl

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase
  SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

local instance : Inhabited Sporos.Word := ⟨⟨0, 0, 0, [], []⟩⟩
local instance : Inhabited Sporos.Instr := ⟨.ret⟩
local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

private theorem mc21_851 : (4200851, (.ofInt (.mov 0 9)), 3) ∈ vmTableChunk 21 := by decide
private theorem mc21_854 : (4200854, (.ofInt (.alu .add 0 13)), 3) ∈ vmTableChunk 21 := by decide
private theorem mc21_857 : (4200857, (.ofInt (.st 15 80 0)), 4) ∈ vmTableChunk 21 := by decide
private theorem mc21_861 : (4200861, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk 21 := by decide
private theorem mc21_865 : (4200865, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk 21 := by decide
private theorem mc21_869 : (4200869, (.ofInt (.mov 2 1)), 3) ∈ vmTableChunk 21 := by decide
private theorem mc21_872 : (4200872, (.ofInt (.shlI 2 5)), 4) ∈ vmTableChunk 21 := by decide
private theorem mc21_876 : (4200876, (.stIdx .w64 0 2 1 0 12), 4) ∈ vmTableChunk 21 := by decide
private theorem mc21_880 : (4200880, (.stIdx .w64 0 2 1 8 11), 5) ∈ vmTableChunk 21 := by decide
private theorem mc21_885 : (4200885, (.stIdx .w64 0 2 1 16 9), 5) ∈ vmTableChunk 21 := by decide
private theorem mc21_890 : (4200890, (.stIdx .w64 0 2 1 24 13), 5) ∈ vmTableChunk 21 := by decide

/-- Steps `4200851..4200876` of frame push: updates `ctx[80]` and prepares
`r0 = rsReg.base`, `r1 = rs.length`, `r2 = rs.length * 32`. -/
private theorem call_push_frame_prep_spec {vc : VmCtx} {pr : Sporos.Prog} {ls1 : SporosProvLen.LSt}
    {s : X86Vm.St} {w nLocs : Nat}
    (hrip : s.rip = 4200851) (h : vc.wf pr ls1 s)
    (hrs_lt : ls1.rs.length < vc.rsCap) (hloc_cap : rsLocBase ls1.rs + nLocs ≤ vc.locCap)
    (hr9 : s.reg 9 = some (rsLocBase ls1.rs)) (hr11 : s.reg 11 = some 0)
    (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs) :
    ∃ (ctx1 : Region) (s7 : X86Vm.St),
      Reaches vmImage s s7 ∧ s7.rip = 4200876 ∧ s7.k = s.k ∧
      s7.mem.wf ∧ BytesLt s7.mem ∧ (∀ u ∈ s7.mem, 0 < u.data.length) ∧ RegsOk s7 ∧
      s7.reg 0 = some vc.rsReg.base ∧ s7.reg 1 = some ls1.rs.length ∧
      s7.reg 2 = some (ls1.rs.length * 32) ∧ s7.reg 4 = some vc.sp ∧
      s7.reg 9 = some (rsLocBase ls1.rs) ∧ s7.reg 11 = some 0 ∧
      s7.reg 12 = some w ∧ s7.reg 13 = some nLocs ∧ s7.reg 15 = some ctxBase ∧
      ctx1 = { vc.ctxReg with data := writeBytes vc.ctxReg.data 80 (bytes8 (rsLocBase ls1.rs + nLocs)) } ∧
      ctx1 ∈ s7.mem ∧ vc.rsReg ∈ s7.mem ∧
      (∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u ∈ s7.mem) := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨_, _, hrsLenEq, _, hrsHi, _, _, hlocLenEq, _, hlocHi, _, _, _⟩ := (vc.arena_bounds h).2.2
  have ⟨s48, s56, _, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
  let s1 := { s.set 0 (some (rsLocBase ls1.rs)) with rip := 4200854 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by rw [step_at_vc 21 hrip mc21_851, exec_mov hr9, hrip]
  have hr1 := regsOk_set hr (r := 0) (by decide) (rsLocBase ls1.rs)
  have hs1_0 : s1.reg 0 = some (rsLocBase ls1.rs) := reg_set_self_ok' hr (by decide) _
  have hs1_13 : s1.reg 13 = some nLocs := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13
  let s2 := { s1.set 0 (some (rsLocBase ls1.rs + nLocs)) with fl := (X86Int.aluF .add (rsLocBase ls1.rs) nLocs).2, rip := 4200857 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 21 rfl mc21_854, exec_alu hs1_0 hs1_13]; dsimp only [s2]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  have hr2 := regsOk_set hr1 (r := 0) (by decide) (rsLocBase ls1.rs + nLocs)
  have hs2_0 : s2.reg 0 = some (rsLocBase ls1.rs + nLocs) := reg_set_self_ok' hr1 (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  have hea80 : ea ctxBase 80 = ctxBase + 80 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  have hea48 : ea ctxBase 48 = ctxBase + 48 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  have hea56 : ea ctxBase 56 = ctxBase + 56 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  obtain ⟨_, m1, hwr1, _, _, hm1wf, hctx1_m, hoth1⟩ := wr64_region h.1.2 (vc.mem_ctx h) hcb.2.1
    (a := ctxBase + 80) (v := rsLocBase ls1.rs + nLocs) (by rw [hcb.1]; omega) (by simp only [Region.hi, hcb.1]; omega)
  let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 80 (bytes8 (rsLocBase ls1.rs + nLocs)) }
  have hctx1_m' : ctx1 ∈ m1 := by simpa [ctx1, hcb.1] using hctx1_m
  let s3 := { s2 with mem := m1, rip := 4200861 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by rw [step_at_vc 21 rfl mc21_857, exec_st hs2_15 hs2_0 (by rw [hea80]; exact hwr1)]
  have hrd48 : rd64 m1 (ea ctxBase 48) = some vc.rsReg.base := by
    rw [hea48]; refine rd64_of_mem hm1wf hctx1_m' (by rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega) ?_
    dsimp [ctx1]; rw [hcb.1, Nat.add_sub_cancel_left, writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by omega)]
    exact ctx_leVal_of_rd64 h (by decide) s48
  have hrd56 : rd64 m1 (ea ctxBase 56) = some ls1.rs.length := by
    rw [hea56]; refine rd64_of_mem hm1wf hctx1_m' (by rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega) ?_
    dsimp [ctx1]; rw [hcb.1, Nat.add_sub_cancel_left, writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by omega)]
    exact ctx_leVal_of_rd64 h (by decide) s56
  let s4 := { s3.set 0 (some vc.rsReg.base) with rip := 4200865 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by rw [step_at_vc 21 rfl mc21_861, exec_ld (s := s3) hs2_15 hrd48]
  have hr4 := regsOk_set hr2 (r := 0) (by decide) vc.rsReg.base
  have hs4_15 : s4.reg 15 = some ctxBase := by simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs2_15
  let s5 := { s4.set 1 (some ls1.rs.length) with rip := 4200869 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by rw [step_at_vc 21 rfl mc21_865, exec_ld hs4_15 hrd56]
  have hr5 := regsOk_set hr4 (r := 1) (by decide) ls1.rs.length
  have hs5_1 : s5.reg 1 = some ls1.rs.length := reg_set_self_ok' hr4 (by decide) _
  let s6 := { s5.set 2 (some ls1.rs.length) with rip := 4200872 }
  have h06 : X86Vm.step vmImage s5 = .ok s6 := by rw [step_at_vc 21 rfl mc21_869, exec_mov hs5_1]
  have hr6 := regsOk_set hr5 (r := 2) (by decide) ls1.rs.length
  have hs6_2 : s6.reg 2 = some ls1.rs.length := reg_set_self_ok' hr5 (by decide) _
  let s7 := { s6.set 2 (some (ls1.rs.length * 32)) with fl := X86Int.Flags.undef, rip := 4200876 }
  have h07 : X86Vm.step vmImage s6 = .ok s7 := by
    rw [step_at_vc 21 rfl mc21_872, exec_shlI_lt hs6_2 (by rw [show 2 ^ (5 % 64) = 32 by decide]; omega)]
  have hr7 := regsOk_set hr6 (r := 2) (by decide) (ls1.rs.length * 32)
  have hreach7 : Reaches vmImage s s7 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step h07))))))
  have hrs_ne_ctx : vc.rsReg.base ≠ vc.ctxReg.base := vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  refine ⟨ctx1, s7, hreach7, rfl, rfl, hm1wf, bytesLt_wr64 h.2.1 hwr1, len_pos_wrN h.2.2.1 hwr1, hr7,
    ?_, ?_, reg_set_self_ok' hr6 (by decide) _, ?_, ?_, ?_, ?_, ?_, ?_, rfl, hctx1_m', hoth1 _ (vc.mem_rs h) hrs_ne_ctx, hoth1⟩
  · simp only [s7, s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact reg_set_self_ok' hr2 (by decide) _
  · simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 1)]; exact hs5_1
  · simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp
  · simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 9), List.getElem?_set_ne (by decide : 1 ≠ 9), List.getElem?_set_ne (by decide : 0 ≠ 9)]; exact hr9
  · simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 11), List.getElem?_set_ne (by decide : 1 ≠ 11), List.getElem?_set_ne (by decide : 0 ≠ 11)]; exact hr11
  · simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 12), List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hr12
  · simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 13), List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13
  · simp only [s7, s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15), List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs4_15

/-- Single indexed qword store `stIdx .w64 0 2 1 disp rv` into region `r`,
keeping `r'` abstract so chained writes avoid `writeBytes` unfolding in `whnf`. -/
private theorem step_stIdx_rs {s : X86Vm.St} {r : Region} {disp rv v base off a len : Nat}
    (hrip : s.rip = a) (hstep : (a, (.stIdx .w64 0 2 1 disp rv), len) ∈ vmTableChunk 21)
    (hmwf : s.mem.wf) (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hr0 : s.reg 0 = some base) (hr2 : s.reg 2 = some off) (hrv : s.reg rv = some v)
    (hr_m : r ∈ s.mem) (hr_b : r.base = base) (hr_rw : r.rw = true)
    (hhi : off + disp + 8 ≤ r.data.length) (hW : base + (off + disp) + 8 < X86Env.W) :
    ∃ (r' : Region) (s' : X86Vm.St),
      X86Vm.step vmImage s = .ok s' ∧ s'.rip = a + len ∧ s'.regs = s.regs ∧ s'.k = s.k ∧
      s'.mem.wf ∧ BytesLt s'.mem ∧ (∀ u ∈ s'.mem, 0 < u.data.length) ∧
      r' = { r with data := writeBytes r.data (off + disp) (bytes8 v) } ∧
      r' ∈ s'.mem ∧ r'.base = base ∧ r'.rw = true ∧ r'.data.length = r.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ base → u ∈ s'.mem) := by
  obtain ⟨_, m', hwr, _, _, hm'wf, hr'_m, hoth⟩ := wrMem_w64_region hmwf hr_m hr_rw
    (a := base + (off + disp)) (v := v) (by omega) (by simp only [Region.hi, hr_b]; omega)
  let r' := { r with data := writeBytes r.data (off + disp) (bytes8 v) }
  have hr'_m' : r' ∈ m' := by
    have hsub : base + (off + disp) - r.base = off + disp := by omega
    rw [hsub] at hr'_m; exact hr'_m
  let s' := { s with mem := m', rip := a + len }
  have hst : X86Vm.step vmImage s = .ok s' := by
    rw [step_at_vc 21 hrip hstep,
      exec_stIdx hr0 hr2 hrv (by rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_assoc]; exact hwr), hrip]
  exact ⟨r', s', hst, rfl, rfl, rfl, hm'wf, bytesLt_wr64 hbl hwr, len_pos_wrMem hpos hwr,
    rfl, hr'_m', hr_b, hr_rw, by dsimp [r']; rw [writeBytes_length], fun u hu hne => hoth u hu (by omega)⟩

/-- Steps `4200876..4200895` of frame push: writes the 4 frame qwords
`(w, 0, lb, nLocs)` into `rsReg[rs.length * 32 .. +24]`. -/
private theorem call_push_frame_rs_spec {vc : VmCtx} {pr : Sporos.Prog} {ls1 : SporosProvLen.LSt}
    {s : X86Vm.St} {w nLocs : Nat}
    (hrip : s.rip = 4200851) (h : vc.wf pr ls1 s)
    (hrs_lt : ls1.rs.length < vc.rsCap) (hloc_cap : rsLocBase ls1.rs + nLocs ≤ vc.locCap)
    (hr9 : s.reg 9 = some (rsLocBase ls1.rs)) (hr11 : s.reg 11 = some 0)
    (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs) :
    ∃ (ctx1 rs5 : Region) (s11 : X86Vm.St),
      Reaches vmImage s s11 ∧ s11.rip = 4200895 ∧ s11.k = s.k ∧
      s11.mem.wf ∧ BytesLt s11.mem ∧ (∀ u ∈ s11.mem, 0 < u.data.length) ∧
      RegsOk s11 ∧ s11.reg 1 = some ls1.rs.length ∧ s11.reg 4 = some vc.sp ∧ s11.reg 15 = some ctxBase ∧
      ctx1 = { vc.ctxReg with data := writeBytes vc.ctxReg.data 80 (bytes8 (rsLocBase ls1.rs + nLocs)) } ∧
      ctx1 ∈ s11.mem ∧
      rs5 = { vc.rsReg with data := writeBytes (writeBytes (writeBytes (writeBytes vc.rsReg.data
        (ls1.rs.length * 32) (bytes8 w)) (ls1.rs.length * 32 + 8) (bytes8 0))
        (ls1.rs.length * 32 + 16) (bytes8 (rsLocBase ls1.rs))) (ls1.rs.length * 32 + 24) (bytes8 nLocs) } ∧
      rs5 ∈ s11.mem ∧
      (∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u.base ≠ vc.rsReg.base → u ∈ s11.mem) := by
  have ⟨hrsRw, _, hrsLenEq, _, hrsHi, _⟩ := (vc.arena_bounds h).2.2
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  have hrs_ne_ctx : vc.rsReg.base ≠ vc.ctxReg.base := vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  obtain ⟨ctx1, s7, hreach7, hrip7, hk7, hm1wf, hbl1, hpos1, hr7,
    hs7_0, hs7_1, hs7_2, hs7_4, hs7_9, hs7_11, hs7_12, hs7_13, hs7_15, hctx1_eq, hctx1_m', hrs_in_m1, hoth1⟩ :=
    call_push_frame_prep_spec hrip h hrs_lt hloc_cap hr9 hr11 hr12 hr13
  have hreg_eq : ∀ {a b : X86Vm.St} {r}, a.regs = b.regs → a.reg r = b.reg r := fun h => by simp [X86Int.St.reg, h]
  obtain ⟨rs2, s8, h08, hrip8, hregs8, hk8, hm2wf, hbl2, hpos2, hrs2_eq, hrs2_m, hrs2_b, hrs2_rw, hrs2_l, hoth2⟩ :=
    step_stIdx_rs hrip7 mc21_876 hm1wf hbl1 hpos1 hs7_0 hs7_2 hs7_12 hrs_in_m1 rfl hrsRw (by omega) (by omega)
  obtain ⟨rs3, s9, h09, hrip9, hregs9, hk9, hm3wf, hbl3, hpos3, hrs3_eq, hrs3_m, hrs3_b, hrs3_rw, hrs3_l, hoth3⟩ :=
    step_stIdx_rs hrip8 mc21_880 hm2wf hbl2 hpos2 (by rw [hreg_eq hregs8]; exact hs7_0)
      (by rw [hreg_eq hregs8]; exact hs7_2) (by rw [hreg_eq hregs8]; exact hs7_11) hrs2_m hrs2_b hrs2_rw (by omega) (by omega)
  obtain ⟨rs4, s10, h10, hrip10, hregs10, hk10, hm4wf, hbl4, hpos4, hrs4_eq, hrs4_m, hrs4_b, hrs4_rw, hrs4_l, hoth4⟩ :=
    step_stIdx_rs hrip9 mc21_885 hm3wf hbl3 hpos3 (by rw [hreg_eq (hregs9.trans hregs8)]; exact hs7_0)
      (by rw [hreg_eq (hregs9.trans hregs8)]; exact hs7_2) (by rw [hreg_eq (hregs9.trans hregs8)]; exact hs7_9) hrs3_m hrs3_b hrs3_rw (by omega) (by omega)
  obtain ⟨rs5, s11, h11, hrip11, hregs11, hk11, hm5wf, hbl5, hpos5, hrs5_eq, hrs5_m, _, _, _, hoth5⟩ :=
    step_stIdx_rs hrip10 mc21_890 hm4wf hbl4 hpos4 (by rw [hreg_eq (hregs10.trans (hregs9.trans hregs8))]; exact hs7_0)
      (by rw [hreg_eq (hregs10.trans (hregs9.trans hregs8))]; exact hs7_2)
      (by rw [hreg_eq (hregs10.trans (hregs9.trans hregs8))]; exact hs7_13) hrs4_m hrs4_b hrs4_rw (by omega) (by omega)
  have hregs_all : s11.regs = s7.regs := hregs11.trans (hregs10.trans (hregs9.trans hregs8))
  have hctx1_b : ctx1.base = vc.ctxReg.base := by rw [hctx1_eq]
  have hctx1_in_m5 : ctx1 ∈ s11.mem :=
    hoth5 _ (hoth4 _ (hoth3 _ (hoth2 _ hctx1_m' (by rw [hctx1_b]; exact hrs_ne_ctx.symm)) (by rw [hctx1_b]; exact hrs_ne_ctx.symm))
      (by rw [hctx1_b]; exact hrs_ne_ctx.symm)) (by rw [hctx1_b]; exact hrs_ne_ctx.symm)
  have hrs5_full : rs5 = { vc.rsReg with data := writeBytes (writeBytes (writeBytes (writeBytes vc.rsReg.data
      (ls1.rs.length * 32) (bytes8 w)) (ls1.rs.length * 32 + 8) (bytes8 0))
      (ls1.rs.length * 32 + 16) (bytes8 (rsLocBase ls1.rs))) (ls1.rs.length * 32 + 24) (bytes8 nLocs) } := by
    rw [hrs5_eq, hrs4_eq, hrs3_eq, hrs2_eq, Nat.add_zero]
  refine ⟨ctx1, rs5, s11,
    Reaches.trans hreach7 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step h11)))),
    hrip11, hk11.trans (hk10.trans (hk9.trans (hk8.trans hk7))), hm5wf, hbl5, hpos5,
    by simpa [RegsOk, X86Int.St.reg, hregs_all] using hr7,
    by rw [hreg_eq hregs_all]; exact hs7_1, by rw [hreg_eq hregs_all]; exact hs7_4, by rw [hreg_eq hregs_all]; exact hs7_15,
    hctx1_eq, hctx1_in_m5, hrs5_full, hrs5_m, fun u hu hc hr' => hoth5 _ (hoth4 _ (hoth3 _ (hoth2 _ (hoth1 u hu hc) hr') hr') hr') hr'⟩

/-- Frame push & jump to `L_vm_loop` (`4200851..4200903`) at the tail of `O_call`. -/
private theorem call_push_frame_spec {vc : VmCtx} {pr : Sporos.Prog} {ls1 : SporosProvLen.LSt}
    {s : X86Vm.St} {w nLocs dbase abase : Nat}
    (hrip : s.rip = 4200851) (h : vc.wf pr ls1 s) (hwW : w < X86Env.W)
    (hrs_lt : ls1.rs.length < vc.rsCap) (hloc_cap : rsLocBase ls1.rs + nLocs ≤ vc.locCap)
    (hr9 : s.reg 9 = some (rsLocBase ls1.rs)) (hr11 : s.reg 11 = some 0)
    (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs)
    (hinit : ∀ j, j < nLocs →
      leVal ((vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8) = 0 ∧
      leVal ((vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8) = (ls1.pc + 1) % X86Env.W) :
    let callee : SporosProvLen.LFrame := ⟨w, 0, List.replicate nLocs ⟨0, (ls1.pc + 1) % Sporos.M⟩, dbase, abase⟩
    ∃ s', Reaches vmImage s s' ∧ s'.k = s.k ∧ VmRel pr { ls1 with rs := callee :: ls1.rs } s' := by
  let callee : SporosProvLen.LFrame := ⟨w, 0, List.replicate nLocs ⟨0, (ls1.pc + 1) % Sporos.M⟩, dbase, abase⟩
  have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, _, _⟩ := (vc.arena_bounds h).2.2
  have ⟨s48, _, s64, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  have ⟨s72, _, s88⟩ := wf_rd_loc_slots h
  have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
  have hlb_new : rsLocBase (callee :: ls1.rs) = rsLocBase ls1.rs + nLocs := by
    simp only [rsLocBase, List.map_cons, List.foldl_cons, Nat.zero_add, callee, List.length_replicate]
    rw [list_foldl_add]; omega
  obtain ⟨ctx1, rs5, s11, hreach11, hrip11, hk11, hm5wf, hbl5, hpos5, hr11', hs11_1, hs11_4, hs11_15,
    hctx1_eq, hctx1_m, hrs5_eq, hrs5_m, hoth5⟩ :=
    call_push_frame_rs_spec hrip h hrs_lt hloc_cap hr9 hr11 hr12 hr13
  let s12 := { s11.set 1 (some (ls1.rs.length + 1)) with fl := (X86Int.aluF .add ls1.rs.length 1).2, rip := 4200899 }
  have h12 : X86Vm.step vmImage s11 = .ok s12 := by
    rw [step_at_vc 21 hrip11 (by decide : (4200895, (.ofInt (.aluI .add 1 1)), 4) ∈ vmTableChunk 21),
      exec_aluI_add_lt hs11_1 (by omega), Nat.mod_eq_of_lt (by omega), hrip11]
  have hr12' := regsOk_set hr11' (r := 1) (by decide) (ls1.rs.length + 1)
  have hs12_1 : s12.reg 1 = some (ls1.rs.length + 1) := reg_set_self_ok' hr11' (by decide) _
  have hs12_15 : s12.reg 15 = some ctxBase := by
    simp only [s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs11_15
  have hea56 : ea ctxBase 56 = ctxBase + 56 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  have hctx1_b : ctx1.base = ctxBase := by rw [hctx1_eq]; exact hcb.1
  have hctx1_rw : ctx1.rw = true := by rw [hctx1_eq]; exact hcb.2.1
  have hctx1_l : ctx1.data.length = vc.ctxReg.data.length := by rw [hctx1_eq]; dsimp; rw [writeBytes_length]
  obtain ⟨_, m6, hwr6, _, _, hm6wf, hctx2_m, hoth6⟩ := wr64_region hm5wf hctx1_m hctx1_rw
    (a := ctxBase + 56) (v := ls1.rs.length + 1) (by rw [hctx1_b]; omega)
    (by simp only [Region.hi, hctx1_b, hctx1_l]; omega)
  let ctx2 := { vc.ctxReg with data := writeBytes ctx1.data 56 (bytes8 (ls1.rs.length + 1)) }
  have hctx2_m' : ctx2 ∈ m6 := by rw [hctx1_b, Nat.add_sub_cancel_left] at hctx2_m; simpa [ctx2, hctx1_eq, hcb.1] using hctx2_m
  let s13 := { s12 with mem := m6, rip := 4200903 }
  have h13 : X86Vm.step vmImage s12 = .ok s13 := by
    rw [step_at_vc 21 rfl (by decide : (4200899, (.ofInt (.st 15 56 1)), 4) ∈ vmTableChunk 21), exec_st hs12_15 hs12_1 (by rw [hea56]; exact hwr6)]
  let s14 := { s13 with rip := SporosVmBin.L_vm_loop }
  have h14 : X86Vm.step vmImage s13 = .ok s14 := by
    rw [step_at_vc 21 rfl (by decide : (4200903, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk 21), exec_jmp]; rfl
  have hreach14 : Reaches vmImage s s14 := Reaches.trans hreach11 (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.step h14)))
  have hoth_all : ∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u.base ≠ vc.rsReg.base → u ∈ m6 := fun u hu hc hr' =>
    hoth6 _ (hoth5 u hu hc hr') (by rw [hctx1_eq]; exact hc)
  have hdisj : ∀ off, (off + 8 ≤ 56 ∨ (64 ≤ off ∧ off + 8 ≤ 80) ∨ 88 ≤ off) → off + 8 ≤ ctxSize →
      leVal ((ctx2.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8) := fun off ho hl => by
    dsimp [ctx2, ctxSize] at hl ⊢; rw [hctx1_eq]; dsimp
    have h1 : off + 8 ≤ 56 ∨ 56 + (bytes8 (ls1.rs.length + 1)).length ≤ off := by
      rw [bytes8_length]; rcases ho with ha | ⟨ha, _⟩ | ha
      · exact Or.inl ha
      · exact Or.inr (by omega)
      · exact Or.inr (by omega)
    have h2 : off + 8 ≤ 80 ∨ 80 + (bytes8 (rsLocBase ls1.rs + nLocs)).length ≤ off := by
      rw [bytes8_length]; rcases ho with ha | ⟨_, hb⟩ | ha
      · exact Or.inl (by omega)
      · exact Or.inl hb
      · exact Or.inr (by omega)
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ h1 (by rw [writeBytes_length]; omega),
        writeBytes_drop_take_disjoint _ _ _ _ _ h2 (by omega)]
  have hrs5_len : rs5.data.length = vc.rsCap * 32 := by
    rw [hrs5_eq]; dsimp; rw [writeBytes_length, writeBytes_length, writeBytes_length, writeBytes_length]; exact hrsLenEq
  have hrel_rs5 : RsLocRel rs5.data vc.locReg.data (rsLocBase ls1.rs) ls1.rs := by
    rw [hrs5_eq]; dsimp
    have h2 := rsLocRel_write_rs_after hrsRel (off := ls1.rs.length * 32) (bs := bytes8 w) (by omega) (bytes8_length _)
    have h3 := rsLocRel_write_rs_after h2 (off := ls1.rs.length * 32 + 8) (bs := bytes8 0) (by omega) (bytes8_length _)
    have h4 := rsLocRel_write_rs_after h3 (off := ls1.rs.length * 32 + 16) (bs := bytes8 (rsLocBase ls1.rs)) (by omega) (bytes8_length _)
    exact rsLocRel_write_rs_after h4 (off := ls1.rs.length * 32 + 24) (bs := bytes8 nLocs) (by omega) (bytes8_length _)
  have hrel_new : RsLocRel rs5.data vc.locReg.data (rsLocBase (callee :: ls1.rs)) (callee :: ls1.rs) := by
    simp only [RsLocRel, callee, List.length_replicate]
    refine ⟨hlb_new, by rw [hrs5_len]; omega, by rw [hlocLenEq]; omega, ?_, ?_, ?_, ?_, ?_, hrel_rs5⟩
    · rw [hrs5_eq]; dsimp
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length, writeBytes_length, writeBytes_length]; omega),
        writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length, writeBytes_length]; omega),
        writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
        leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt hwW]
    · rw [hrs5_eq]; dsimp
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length, writeBytes_length, writeBytes_length]; omega),
        writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length, writeBytes_length]; omega),
        leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by decide)]
    · rw [hrs5_eq]; dsimp
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length, writeBytes_length, writeBytes_length]; omega),
        leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
    · rw [hrs5_eq]; dsimp
      rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length, writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
    · intro j hj
      have hget : (List.replicate nLocs (⟨0, (ls1.pc + 1) % Sporos.M⟩ : SporosProvLen.LVal))[j]! = ⟨0, (ls1.pc + 1) % Sporos.M⟩ := by
        rw [List.getElem!_eq_getElem?_getD, List.getElem?_replicate, if_pos hj]; rfl
      rw [hget]; simpa [M_eq_W'] using hinit j hj
  have hwlt_rs : ∀ g ∈ callee :: ls1.rs, ∀ v ∈ g.locs, v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro g hg v hv
    rcases List.mem_cons.mp hg with rfl | hg
    · simp only [callee, List.mem_replicate] at hv; rcases hv with ⟨_, rfl⟩
      exact ⟨by show 0 < X86Env.W; omega, by show (ls1.pc + 1) % Sporos.M < X86Env.W; rw [← M_eq_W']; exact Nat.mod_lt _ (by rw [M_eq_W']; omega)⟩
    · exact (vc.code_flat h).2.2.2.2.2.2.1 g hg v hv
  have hsp14 : s14.reg 4 = some vc.sp := by
    simp only [s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hs11_4
  have hrs_ne_ctx : vc.rsReg.base ≠ vc.ctxReg.base := vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hrs5_b : rs5.base = vc.rsReg.base := by rw [hrs5_eq]
  have hwf14 : ({ vc with ctxReg := ctx2, rsReg := rs5 } : VmCtx).wf pr { ls1 with rs := callee :: ls1.rs } s14 :=
    wf_update_rs_loc h hm6wf (bytesLt_wr64 hbl5 hwr6) (len_pos_wrN hpos5 hwr6) hr12' hsp14 hs12_15
      (hoth_all _ (vc.mem_stk h) (vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (vc.ne_bases h (i := 1) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))) rfl rfl rfl
      hctx2_m' rfl rfl (by dsimp [ctx2]; rw [writeBytes_length, hctx1_l])
      (by rw [hdisj 48 (Or.inl (by decide)) (by decide), hrs5_b]; exact ctx_leVal_of_rd64 h (by decide) s48)
      (by dsimp [ctx2, List.length_cons]; rw [leVal_writeBytes_same _ _ _ (by rw [hctx1_l]; omega), Nat.mod_eq_of_lt (by omega)])
      (by rw [hdisj 64 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s64)
      (by rw [hdisj 72 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s72)
      (by rw [hlb_new]; dsimp [ctx2]; rw [hctx1_eq]; dsimp
          rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)])
      (by rw [hdisj 88 (Or.inr (Or.inr (by decide))) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s88)
      (fun off ho hl => hdisj off (by rcases ho with h1 | h1; exact Or.inl (by omega); exact Or.inr (Or.inr (by omega))) hl)
      (hoth6 rs5 hrs5_m (by rw [hrs5_b, hctx1_b, ← hcb.1]; exact hrs_ne_ctx)) (by rw [hrs5_eq]; exact hrsRw) hrsPos hrs5_len hrsPg
      (by show rs5.base + rs5.data.length ≤ 2 ^ 47; rw [hrs5_b, hrs5_len, ← hrsLenEq]; exact hrsHi)
      (hoth_all _ (vc.mem_loc h) (vc.ne_bases h (i := 9) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (vc.ne_bases h (i := 9) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)))
      hlocRw hlocPos hlocLenEq hlocPg hlocHi hrel_new (by simp; omega) (by rw [hlb_new]; exact hloc_cap)
      hwlt_rs (by simpa [VmCtx.regions, hrs5_b] using h.2.2.2.2.2.2.1) (fun u hu _ h2 h3 _ => hoth_all u hu h2 h3)
  exact ⟨s14, hreach14, hk11, rfl, _, hwf14⟩

/-- Simulation proof for `Sporos.Instr.call w` (`O_call`, `0x4017bc`). -/
theorem O_call_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {w : Nat}
    (hrip : s.rip = instrEntry (.call w)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length)
    (hins : pr.words[f.word]!.code[f.pc]! = .call w)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.call w)).2) :
    CtrlOutcome pr ls s := by
  have hfetch := fetch_eq_some hnw hpc hins
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨hf1, hf2, hf3, hfit, _⟩ := vc.code_flat h
  have ⟨_, _, _, _, hc1, hc2, hc3⟩ := vc.rd_ctx_rs_code_slots h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hwd_f : pr.words[f.word]! ∈ pr.words := getElem!_mem_of_lt hnw
  have hins_f : (.call w : Sporos.Instr) ∈ pr.words[f.word]!.code := hins ▸ getElem!_mem_of_lt hpc
  have hwW : w < X86Env.W := (hfit.2.2.2.2.2 _ hwd_f).2.2.2.2 _ hins_f
  have h264' : rd64 s.mem (ea ctxBase 264) = some w := by simpa [instrPair, Nat.mod_eq_of_lt hwW] using h264
  have hnwW : pr.words.length < X86Env.W := hfit.2.1
  have hndW : pr.data.length < X86Env.W := hfit.2.2.1
  obtain ⟨s1, hr1, hrip1, hm1, hk1, hreg1, hs1_0, _, _, hs1_o⟩ := tmr_ld_vote_eq_spec
    (a := SporosVmBin.O_call) (done := 4200436) (disp := 8) (v := pr.words.length) hrip hr h15
    (mem_vmTable_of_chunk 19 rfl (by decide)) (mem_vmTable_of_chunk 19 rfl (by decide))
    (mem_vmTable_of_chunk 19 rfl (by decide)) (mem_vmTable_of_chunk 19 rfl (by decide))
    (mem_vmTable_of_chunk 19 rfl (by decide)) (mem_vmTable_of_chunk 19 rfl (by decide))
    (mem_vmTable_of_chunk 19 rfl (by decide)) (mem_vmTable_of_chunk 19 rfl (by decide))
    hc1 hc2 hc3 (rd64_flat_nw h.1.2 (vc.mem_code1 h) hf1 hfit) (rd64_flat_nw h.1.2 (vc.mem_code2 h) hf2 hfit)
    (rd64_flat_nw h.1.2 (vc.mem_code3 h) hf3 hfit) hnwW
  have hs1_15 : s1.reg 15 = some ctxBase := by rw [hs1_o 15 (by decide) (by decide) (by decide) (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by rw [hs1_o 4 (by decide) (by decide) (by decide) (by decide)]; exact hsp
  let s2 := { s1.set 12 (some w) with rip := 4200443 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 20 hrip1 (by decide : (4200436, (.ofInt (.ld 12 15 264)), 7) ∈ vmTableChunk 20), exec_ld hs1_15 (hm1 ▸ h264'), hrip1]
  have hreg2 := regsOk_set hreg1 (r := 12) (by decide) w
  have hs2_12 : s2.reg 12 = some w := reg_set_self_ok' hreg1 (by decide) _
  have hs2_0 : s2.reg 0 = some pr.words.length := by simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 0)]; exact hs1_0
  let s3 := { s2 with fl := (X86Int.aluF .sub w pr.words.length).2, rip := 4200446 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 20 rfl (by decide : (4200443, (.ofInt (.cmp 12 0)), 3) ∈ vmTableChunk 20), exec_cmp hs2_12 hs2_0]
  have hreach3 : Reaches vmImage s s3 := Reaches.trans hr1 (Reaches.step_trans h02 (Reaches.step h03))
  have hs3_4 : s3.reg 4 = some vc.sp := by simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 4)]; exact hs1_4
  have hs3_15 : s3.reg 15 = some ctxBase := by simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]; exact hs1_15
  unfold CtrlOutcome; simp only [SporosProvLen.lstep, hrs, hfetch]
  by_cases hw : w < pr.words.length
  · have hw_get : pr.words[w]? = some (pr.words[w]'hw) := List.getElem?_eq_getElem hw
    let wd := pr.words[w]'hw
    have hwd_eq : pr.words[w]! = wd := by rw [List.getElem!_eq_getElem?_getD, hw_get]; rfl
    have hwd_fit := hfit.2.2.2.2.2 wd (List.getElem_mem hw)
    rw [hw_get]; dsimp only
    let s4 := { s3 with rip := 4200452 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 20 rfl (by decide : (4200446, (.ofInt (.jcc 3 4201116)), 6) ∈ vmTableChunk 20),
        exec_jcc (cond_aluF_sub_ae w pr.words.length)]; simp [show ¬ pr.words.length ≤ w by omega, s3, s4]
    obtain ⟨s5, hr5, hrip5, hm5, hk5, hreg5, hs5_0, _, _, hs5_o⟩ := tmr_ld_vote_eq_spec
      (a := 4200452) (done := 4200508) (disp := 16) (v := pr.data.length) (s := s4) rfl hreg2 hs3_15
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (hm1 ▸ hc1) (hm1 ▸ hc2) (hm1 ▸ hc3)
      (hm1 ▸ rd64_flat_nd h.1.2 (vc.mem_code1 h) hf1 hfit) (hm1 ▸ rd64_flat_nd h.1.2 (vc.mem_code2 h) hf2 hfit)
      (hm1 ▸ rd64_flat_nd h.1.2 (vc.mem_code3 h) hf3 hfit) hndW
    have hboff_bound : w * 64 + pr.data.length * 16 + 32 < 2 ^ 47 := by
      have hfq : (flatProgQwords pr).length * 8 < 2 ^ 47 := hfit.2.2.2.1
      have htot : 4 + 2 * pr.data.length + 8 * pr.words.length ≤ (flatProgQwords pr).length := by
        simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len]; omega
      omega
    have hs5_12 : s5.reg 12 = some w := by rw [hs5_o 12 (by decide) (by decide) (by decide) (by decide)]; exact hs2_12
    have hs5_15 : s5.reg 15 = some ctxBase := by rw [hs5_o 15 (by decide) (by decide) (by decide) (by decide)]; exact hs3_15
    have hs5_4 : s5.reg 4 = some vc.sp := by rw [hs5_o 4 (by decide) (by decide) (by decide) (by decide)]; exact hs3_4
    let s6 := { s5.set 0 (some (pr.data.length * 16)) with fl := X86Int.Flags.undef, rip := 4200512 }
    have h06 : X86Vm.step vmImage s5 = .ok s6 := by
      rw [step_at_vc 20 hrip5 (by decide : (4200508, (.ofInt (.shlI 0 4)), 4) ∈ vmTableChunk 20),
        exec_shlI_lt hs5_0 (by rw [show 2 ^ (4 % 64) = 16 by decide]; omega), hrip5]
    have hreg6 := regsOk_set hreg5 (r := 0) (by decide) (pr.data.length * 16)
    have hs6_12 : s6.reg 12 = some w := by simp only [s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hs5_12
    let s7 := { s6.set 8 (some w) with rip := 4200515 }
    have h07 : X86Vm.step vmImage s6 = .ok s7 := by
      rw [step_at_vc 20 rfl (by decide : (4200512, (.ofInt (.mov 8 12)), 3) ∈ vmTableChunk 20), exec_mov hs6_12]
    have hreg7 := regsOk_set hreg6 (r := 8) (by decide) w
    have hs7_8 : s7.reg 8 = some w := reg_set_self_ok' hreg6 (by decide) _
    let s8 := { s7.set 8 (some (w * 64)) with fl := X86Int.Flags.undef, rip := 4200519 }
    have h08 : X86Vm.step vmImage s7 = .ok s8 := by
      rw [step_at_vc 20 rfl (by decide : (4200515, (.ofInt (.shlI 8 6)), 4) ∈ vmTableChunk 20),
        exec_shlI_lt hs7_8 (by rw [show 2 ^ (6 % 64) = 64 by decide]; omega)]
    have hreg8 := regsOk_set hreg7 (r := 8) (by decide) (w * 64)
    have hs8_8 : s8.reg 8 = some (w * 64) := reg_set_self_ok' hreg7 (by decide) _
    have hs8_0 : s8.reg 0 = some (pr.data.length * 16) := by
      simp only [s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 0)]; exact reg_set_self_ok' hreg5 (by decide) _
    let s9 := { s8.set 8 (some (w * 64 + pr.data.length * 16)) with fl := (X86Int.aluF .add (w * 64) (pr.data.length * 16)).2, rip := 4200522 }
    have h09 : X86Vm.step vmImage s8 = .ok s9 := by
      rw [step_at_vc 20 rfl (by decide : (4200519, (.ofInt (.alu .add 8 0)), 3) ∈ vmTableChunk 20), exec_alu hs8_8 hs8_0]
      dsimp only [s9]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
    have hreg9 := regsOk_set hreg8 (r := 8) (by decide) (w * 64 + pr.data.length * 16)
    have hs9_8 : s9.reg 8 = some (w * 64 + pr.data.length * 16) := reg_set_self_ok' hreg8 (by decide) _
    let boff := w * 64 + pr.data.length * 16 + 32
    let s10 := { s9.set 8 (some boff) with fl := (X86Int.aluF .add (w * 64 + pr.data.length * 16) 32).2, rip := 4200526 }
    have h10 : X86Vm.step vmImage s9 = .ok s10 := by
      rw [step_at_vc 20 rfl (by decide : (4200522, (.ofInt (.aluI .add 8 32)), 4) ∈ vmTableChunk 20),
        exec_aluI_add_lt hs9_8 (by omega), Nat.mod_eq_of_lt (by omega)]
    have hreg10 := regsOk_set hreg9 (r := 8) (by decide) boff
    have hs10_8 : s10.reg 8 = some boff := reg_set_self_ok' hreg9 (by decide) _
    have hs10_15 : s10.reg 15 = some ctxBase := by
      simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 15),
        List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs5_15
    have hs10_4 : s10.reg 4 = some vc.sp := by
      simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 4),
        List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hs5_4
    have hs10_12 : s10.reg 12 = some w := by
      simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 12),
        List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hs5_12
    have hfw1 := rd64_flat_word h.1.2 (vc.mem_code1 h) hf1 hfit hw
    have hfw2 := rd64_flat_word h.1.2 (vc.mem_code2 h) hf2 hfit hw
    have hfw3 := rd64_flat_word h.1.2 (vc.mem_code3 h) hf3 hfit hw
    have hm10 : s10.mem = s.mem := hm5.trans hm1
    obtain ⟨s11, hr11, hrip11, hm11, hk11, hreg11, hs11_0, _, _, hs11_o⟩ := tmr_ldIdx8_vote_eq_spec
      (a := 4200526) (done := 4200582) (disp := 0) (ldLen := 4) (r8v := boff) (v := wd.ins) (s := s10)
      rfl hreg10 hs10_15 hs10_8
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (hm10 ▸ hc1) (hm10 ▸ hc2) (hm10 ▸ hc3)
      (hm10 ▸ hwd_eq ▸ hfw1.1) (hm10 ▸ hwd_eq ▸ hfw2.1) (hm10 ▸ hwd_eq ▸ hfw3.1) hwd_fit.1
    have hs11_15 : s11.reg 15 = some ctxBase := by rw [hs11_o 15 (by decide) (by decide) (by decide) (by decide)]; exact hs10_15
    have hs11_4 : s11.reg 4 = some vc.sp := by rw [hs11_o 4 (by decide) (by decide) (by decide) (by decide)]; exact hs10_4
    have hs11_8 : s11.reg 8 = some boff := by rw [hs11_o 8 (by decide) (by decide) (by decide) (by decide)]; exact hs10_8
    have hs11_12 : s11.reg 12 = some w := by rw [hs11_o 12 (by decide) (by decide) (by decide) (by decide)]; exact hs10_12
    have hm11' : s11.mem = s.mem := hm11.trans hm10
    let s12 := { s11.set 1 (some ls.ds.length) with rip := 4200586 }
    have h12 : X86Vm.step vmImage s11 = .ok s12 := by
      rw [step_at_vc 20 hrip11 (by decide : (4200582, (.ofInt (.ld 1 15 8)), 4) ∈ vmTableChunk 20),
        exec_ld hs11_15 (hm11' ▸ vc.rd_ctx_dsLen h), hrip11]
    have hreg12 := regsOk_set hreg11 (r := 1) (by decide) ls.ds.length
    have hs12_1 : s12.reg 1 = some ls.ds.length := reg_set_self_ok' hreg11 (by decide) _
    have hs12_0 : s12.reg 0 = some wd.ins := by simp only [s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs11_0
    let s13 := { s12 with fl := (X86Int.aluF .sub ls.ds.length wd.ins).2, rip := 4200589 }
    have h13 : X86Vm.step vmImage s12 = .ok s13 := by
      rw [step_at_vc 20 rfl (by decide : (4200586, (.ofInt (.cmp 1 0)), 3) ∈ vmTableChunk 20), exec_cmp hs12_1 hs12_0]
    have hreach13 : Reaches vmImage s s13 :=
      Reaches.trans hreach3 (Reaches.step_trans h04 (Reaches.trans hr5 (Reaches.step_trans h06
        (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10
          (Reaches.trans hr11 (Reaches.step_trans h12 (Reaches.step h13))))))))))
    have hk13 : s13.k = s.k := hk11.trans (hk5.trans hk1)
    have hs13_4 : s13.reg 4 = some vc.sp := by simp only [s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hs11_4
    have hs13_15 : s13.reg 15 = some ctxBase := by simp only [s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs11_15
    by_cases hins_lt : ls.ds.length < wd.ins
    · rw [if_pos hins_lt]
      let s14 := { s13 with rip := SporosVmBin.L_fail_underflow }
      have h14 : X86Vm.step vmImage s13 = .ok s14 := by
        rw [step_at_vc 20 rfl (by decide : (4200589, (.ofInt (.jcc 2 4201048)), 6) ∈ vmTableChunk 20),
          exec_jcc (cond_aluF_sub_b ls.ds.length wd.ins)]; simp [hins_lt, s13, s14]; rfl
      have hwf14 : vc.wf pr ls s14 := vc.wf_step_misc h hm11' hreg12 hs13_4 hs13_15
      obtain ⟨se, hre, he, hke⟩ := L_fail_from_vc_spec (e := .underflow) rfl hwf14
      exact ⟨se, Reaches.trans hreach13 (Reaches.step_trans h14 hre), he, hke.trans hk13⟩
    · rw [if_neg hins_lt]
      let s14 := { s13 with rip := 4200595 }
      have h14 : X86Vm.step vmImage s13 = .ok s14 := by
        rw [step_at_vc 20 rfl (by decide : (4200589, (.ofInt (.jcc 2 4201048)), 6) ∈ vmTableChunk 20),
          exec_jcc (cond_aluF_sub_b ls.ds.length wd.ins)]; simp [hins_lt, s13, s14]
      have hs14_8 : s14.reg 8 = some boff := by simp only [s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 8)]; exact hs11_8
      have hs14_12 : s14.reg 12 = some w := by simp only [s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12)]; exact hs11_12
      obtain ⟨s15, hr15, hrip15, hm15, hk15, hreg15, hs15_0, _, _, hs15_o⟩ := tmr_ldIdx8_vote_eq_spec
        (a := 4200595) (done := 4200654) (disp := 16) (ldLen := 5) (r8v := boff) (v := wd.locals) (s := s14)
        rfl hreg12 hs13_15 hs14_8
        (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
        (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
        (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
        (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
        (hm11' ▸ hc1) (hm11' ▸ hc2) (hm11' ▸ hc3)
        (hm11' ▸ hwd_eq ▸ hfw1.2.1) (hm11' ▸ hwd_eq ▸ hfw2.2.1) (hm11' ▸ hwd_eq ▸ hfw3.2.1) (by have := hwd_fit.2.2.1; omega)
      let s16 := { s15.set 13 (some wd.locals) with rip := 4200657 }
      have h16 : X86Vm.step vmImage s15 = .ok s16 := by
        rw [step_at_vc 20 hrip15 (by decide : (4200654, (.ofInt (.mov 13 0)), 3) ∈ vmTableChunk 20), exec_mov hs15_0, hrip15]
      have hreg16 := regsOk_set hreg15 (r := 13) (by decide) wd.locals
      have hreach16 : Reaches vmImage s s16 := Reaches.trans hreach13 (Reaches.step_trans h14 (Reaches.trans hr15 (Reaches.step h16)))
      have hs16_4 : s16.reg 4 = some vc.sp := by
        change (s15.set 13 (some wd.locals)).reg 4 = some vc.sp
        rw [reg_set_ne (by decide : (13 : Nat) ≠ 4), hs15_o 4 (by decide) (by decide) (by decide) (by decide)]; exact hs13_4
      have hs16_15 : s16.reg 15 = some ctxBase := by
        change (s15.set 13 (some wd.locals)).reg 15 = some ctxBase
        rw [reg_set_ne (by decide : (13 : Nat) ≠ 15), hs15_o 15 (by decide) (by decide) (by decide) (by decide)]; exact hs13_15
      have hs16_12 : s16.reg 12 = some w := by
        change (s15.set 13 (some wd.locals)).reg 12 = some w
        rw [reg_set_ne (by decide : (13 : Nat) ≠ 12), hs15_o 12 (by decide) (by decide) (by decide) (by decide)]; exact hs14_12
      have hs16_13 : s16.reg 13 = some wd.locals := reg_set_self_ok' hreg15 (by decide) _
      have hwf16 : vc.wf pr ls s16 := vc.wf_step_misc h (hm15.trans hm11') hreg16 hs16_4 hs16_15
      have hk16 : s16.k = s.k := hk15.trans hk13
      rcases call_grow_rs_spec rfl hwf16 (oracleAdmits_of_reaches hor hreach16) hs16_12 hs16_13 with
        ⟨vc1, s17, hr17, hrip17, hk17, hor17, hwf17, hrs_lt17, h12_17, h13_17⟩ | ⟨hhit, hend⟩
      · rcases call_locs_chk_loop_spec hwd_fit.2.2.1 (rsLocBase ls.rs + wd.locals - vc1.locCap) vc1 s17 (Nat.le_refl _)
          hrip17 hwf17 hor17 h12_17 h13_17 with
          ⟨vc2, s18, hr18, hrip18, hk18, hor18, hwf18, hrsc18, hlocc18, h12_18, h13_18⟩ | ⟨hhit, hend⟩
        · obtain ⟨s19, hr19, hrip19, hk19, hwf19, h12_19, h13_19⟩ :=
            call_adv_caller_pc_spec hrip18 hwf18 hrs hpcW h12_18 h13_18
          let ls1 := { ls with rs := { f with pc := f.pc + 1 } :: rest }
          let vc3 := { vc2 with rsReg := { vc2.rsReg with data := writeBytes vc2.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) } }
          have hlb_eq : rsLocBase ls1.rs = rsLocBase ls.rs := by simp [ls1, hrs, rsLocBase]
          have hreg19 := hwf19.2.2.2.1; have hsp19 := hwf19.2.2.2.2.1; have h15_19 := hwf19.2.2.2.2.2.1
          have ⟨hlocB19, hlocL19, _⟩ := wf_rd_loc_slots hwf19
          have hpc19 := rd_ctx_pc hwf19
          let s20 := { s19.set 8 (some vc3.locReg.base) with rip := 4200801 }
          have h20 : X86Vm.step vmImage s19 = .ok s20 := by
            rw [step_at_vc 21 hrip19 (by decide : (4200797, (.ofInt (.ld 8 15 72)), 4) ∈ vmTableChunk 21), exec_ld h15_19 hlocB19, hrip19]
          have hreg20 := regsOk_set hreg19 (r := 8) (by decide) vc3.locReg.base
          have hs20_15 : s20.reg 15 = some ctxBase := by simp only [s20, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 15)]; exact h15_19
          let s21 := { s20.set 9 (some (rsLocBase ls1.rs)) with rip := 4200805 }
          have h21 : X86Vm.step vmImage s20 = .ok s21 := by
            rw [step_at_vc 21 rfl (by decide : (4200801, (.ofInt (.ld 9 15 80)), 4) ∈ vmTableChunk 21), exec_ld hs20_15 hlocL19]
          have hreg21 := regsOk_set hreg20 (r := 9) (by decide) (rsLocBase ls1.rs)
          have hs21_15 : s21.reg 15 = some ctxBase := by simp only [s21, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 9 ≠ 15)]; exact hs20_15
          let s22 := { s21.set 10 (some ls1.pc) with rip := 4200812 }
          have h22 : X86Vm.step vmImage s21 = .ok s22 := by
            rw [step_at_vc 21 rfl (by decide : (4200805, (.ofInt (.ld 10 15 224)), 7) ∈ vmTableChunk 21), exec_ld hs21_15 hpc19]
          have hreg22 := regsOk_set hreg21 (r := 10) (by decide) ls1.pc
          have hs22_10 : s22.reg 10 = some ls1.pc := reg_set_self_ok' hreg21 (by decide) _
          let s23 := { s22.set 10 (some ((ls1.pc + 1) % X86Env.W)) with fl := (X86Int.aluF .add ls1.pc 1).2, rip := 4200816 }
          have h23 : X86Vm.step vmImage s22 = .ok s23 := by
            rw [step_at_vc 21 rfl (by decide : (4200812, (.ofInt (.aluI .add 10 1)), 4) ∈ vmTableChunk 21),
              exec_aluI_add_lt hs22_10 (by decide : 1 < X86Env.W)]
          have hreg23 := regsOk_set hreg22 (r := 10) (by decide) ((ls1.pc + 1) % X86Env.W)
          obtain ⟨v3, hv3⟩ := regs_get_of_ok hreg23 3 (by decide) (by decide)
          let s24 := { s23.set 3 (some 0) with fl := (X86Int.aluF .xor v3 v3).2, rip := 4200818 }
          have h24 : X86Vm.step vmImage s23 = .ok s24 := by
            rw [step_at_vc 21 rfl (by decide : (4200816, (.ofInt (.alu .xor 3 3)), 2) ∈ vmTableChunk 21), exec_alu_xor_self (s := s23) hv3]
          have hreg24 := regsOk_set hreg23 (r := 3) (by decide) 0
          have hs24_3 : s24.reg 3 = some 0 := reg_set_self_ok' hreg23 (by decide) _
          let s25 := { s24.set 11 (some 0) with rip := 4200821 }
          have h25 : X86Vm.step vmImage s24 = .ok s25 := by
            rw [step_at_vc 21 rfl (by decide : (4200818, (.ofInt (.mov 11 3)), 3) ∈ vmTableChunk 21), exec_mov hs24_3]
          have hreg25 := regsOk_set hreg24 (r := 11) (by decide) 0
          have hreach25 : Reaches vmImage s19 s25 :=
            Reaches.step_trans h20 (Reaches.step_trans h21 (Reaches.step_trans h22 (Reaches.step_trans h23 (Reaches.step_trans h24 (Reaches.step h25)))))
          have hsp25 : s25.reg 4 = some vc3.sp := by
            simp only [s25, s24, s23, s22, s21, s20, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 4),
              List.getElem?_set_ne (by decide : 3 ≠ 4), List.getElem?_set_ne (by decide : 10 ≠ 4),
              List.getElem?_set_ne (by decide : 9 ≠ 4), List.getElem?_set_ne (by decide : 8 ≠ 4)]; exact hsp19
          have h15_25 : s25.reg 15 = some ctxBase := by
            simp only [s25, s24, s23, s22, s21, s20, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 15),
              List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 10 ≠ 15),
              List.getElem?_set_ne (by decide : 9 ≠ 15), List.getElem?_set_ne (by decide : 8 ≠ 15)]; exact h15_19
          have hwf25 : vc3.wf pr ls1 s25 := vc3.wf_step_misc hwf19 rfl hreg25 hsp25 h15_25
          have hr25_8 : s25.reg 8 = some vc3.locReg.base := by
            simp only [s25, s24, s23, s22, s21, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 8),
              List.getElem?_set_ne (by decide : 3 ≠ 8), List.getElem?_set_ne (by decide : 10 ≠ 8),
              List.getElem?_set_ne (by decide : 9 ≠ 8)]; exact reg_set_self_ok' hreg19 (by decide) _
          have hr25_9 : s25.reg 9 = some (rsLocBase ls1.rs) := by
            simp only [s25, s24, s23, s22, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 9),
              List.getElem?_set_ne (by decide : 3 ≠ 9), List.getElem?_set_ne (by decide : 10 ≠ 9)]; exact reg_set_self_ok' hreg20 (by decide) _
          have hr25_10 : s25.reg 10 = some ((ls1.pc + 1) % X86Env.W) := by
            simp only [s25, s24, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 10),
              List.getElem?_set_ne (by decide : 3 ≠ 10)]; exact reg_set_self_ok' hreg22 (by decide) _
          have hr25_11 : s25.reg 11 = some 0 := reg_set_self_ok' hreg24 (by decide) _
          have hr25_3 : s25.reg 3 = some (wd.locals - wd.locals) := by
            rw [Nat.sub_self]; simp only [s25, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 3)]; exact hs24_3
          have hr25_12 : s25.reg 12 = some w := by
            simp only [s25, s24, s23, s22, s21, s20, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 12),
              List.getElem?_set_ne (by decide : 3 ≠ 12), List.getElem?_set_ne (by decide : 10 ≠ 12),
              List.getElem?_set_ne (by decide : 9 ≠ 12), List.getElem?_set_ne (by decide : 8 ≠ 12)]; exact h12_19
          have hr25_13 : s25.reg 13 = some wd.locals := by
            simp only [s25, s24, s23, s22, s21, s20, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 13),
              List.getElem?_set_ne (by decide : 3 ≠ 13), List.getElem?_set_ne (by decide : 10 ≠ 13),
              List.getElem?_set_ne (by decide : 9 ≠ 13), List.getElem?_set_ne (by decide : 8 ≠ 13)]; exact h13_19
          obtain ⟨locReg', s26, hr26, hrip26, hk26, _, hwf26, h26_9, h26_11, h26_12, h26_13, hinit26⟩ :=
            call_init_locs_loop_spec (Nat.mod_lt _ (by decide)) wd.locals vc3 s25 (Nat.le_refl _)
              (by simpa [vc3, hlb_eq] using hlocc18) rfl hwf25 hr25_8 hr25_9 hr25_10 hr25_11 hr25_3 hr25_12 hr25_13
              (by intro j hj; omega)
          obtain ⟨s27, hr27, hk27, hvm27⟩ := call_push_frame_spec (dbase := ls.ds.length - wd.ins) (abase := ls.aux.length)
            hrip26 hwf26 hwW (by simpa [ls1, hrs, vc3, hrsc18] using hrs_lt17) (by simpa [vc3, hlb_eq] using hlocc18)
            h26_9 h26_11 h26_12 h26_13 hinit26
          have htot : Reaches vmImage s s27 :=
            Reaches.trans hreach16 (Reaches.trans hr17 (Reaches.trans hr18 (Reaches.trans hr19
              (Reaches.trans hreach25 (Reaches.trans hr26 hr27)))))
          exact Or.inl ⟨s27, htot, by rw [hk27, hk26]; change s19.k.out = s.k.out; rw [hk19, hk18, hk17, hk16],
            oracleAdmits_of_reaches hor htot, hvm27⟩
        · have hpre := Reaches.trans hreach16 hr17
          exact Or.inr ⟨hitsNomem_of_reaches hpre hhit, Ends.of_reaches hpre hend⟩
      · exact Or.inr ⟨hitsNomem_of_reaches hreach16 hhit, Ends.of_reaches hreach16 hend⟩
  · have hw_none : pr.words[w]? = none := List.getElem?_eq_none (Nat.le_of_not_lt hw)
    rw [hw_none]; dsimp only
    let s4 := { s3 with rip := SporosVmBin.L_fail_badWord }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 20 rfl (by decide : (4200446, (.ofInt (.jcc 3 4201116)), 6) ∈ vmTableChunk 20),
        exec_jcc (cond_aluF_sub_ae w pr.words.length)]; simp [show pr.words.length ≤ w by omega, s3, s4]; rfl
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h hm1 hreg2 hs3_4 hs3_15
    obtain ⟨se, hre, he, hke⟩ := L_fail_from_vc_spec (e := .badWord) rfl hwf4
    exact ⟨se, Reaches.trans hreach3 (Reaches.step_trans h04 hre), he, hke.trans hk1⟩

end SporosVmOpsCtrl
