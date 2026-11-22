import Bootstrap.SporosVmOpsJmp

/-!
# Native x86-64 VM Control-Flow Instruction Proofs (`O_jmp`, `O_jz`, `O_jnz`, `O_call`, `O_ret`)

Top-level control-flow module proving `O_ret_spec` (`0x4019cc`) and transitively
re-exporting `CtrlOutcome`, `O_jmp_spec`, `O_jz_spec`, `O_jnz_spec`, and
`O_call_spec` in `namespace SporosVmOpsCtrl`.

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

/-- Simulation proof for `Sporos.Instr.ret` (`O_ret`, `0x4019cc`). -/
theorem O_ret_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = instrEntry .ret) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length)
    (hins : pr.words[f.word]!.code[f.pc]! = .ret)
    (_hpcW : f.pc + 1 < X86Env.W)
    (_h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair .ret).2) : CtrlOutcome pr ls s := by
  have hfetch := fetch_eq_some hnw hpc hins
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, hrsLe, hlocLe⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, hrsCap, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  have ⟨s72, _, s88⟩ := wf_rd_loc_slots h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  rw [hrs] at hrsLen hrsRel hrsLe hlocLe
  let s1 := { s.set 1 (some (rest.length + 1)) with rip := 4200912 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_vc 21 hrip (by decide : (4200908, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk 21),
      exec_ld h15 (by simpa [List.length_cons] using hrsLen), hrip]; rfl
  have hr1 := regsOk_set hr (r := 1) (by decide) (rest.length + 1)
  have hs1_1 : s1.reg 1 = some (rest.length + 1) := reg_set_self_ok' hr (by decide) _
  let s2 := { s1 with fl := (X86Int.aluF .sub (rest.length + 1) (1 % X86Env.W)).2, rip := 4200916 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 21 rfl (by decide : (4200912, (.ofInt (.cmpI 1 1)), 4) ∈ vmTableChunk 21), exec_cmpI hs1_1]
  have h1_mod : 1 % X86Env.W = 1 := by decide
  have hcond_a := cond_aluF_sub_a (rest.length + 1) 1 (by simp only [List.length_cons] at hrsLe; omega) (by decide)
  unfold CtrlOutcome; simp only [SporosProvLen.lstep, hrs, hfetch]
  cases rest with
  | nil =>
    let s3 := { s2 with rip := 4200918 }
    have h03 : X86Vm.step vmImage s2 = .ok s3 := by
      rw [step_at_vc 21 rfl (by decide : (4200916, (.ofInt (.jcc 7 4200932)), 2) ∈ vmTableChunk 21),
        exec_jcc (by simpa [s2, h1_mod] using hcond_a)]; simp [s2, s3]
    obtain ⟨v0, hv0⟩ := regs_get_of_ok hr1 0 (by decide) (by decide)
    let s4 := { s3.set 0 (some 0) with fl := (X86Int.aluF .xor v0 v0).2, rip := 4200920 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 21 rfl (by decide : (4200918, (.ofInt (.alu .xor 0 0)), 2) ∈ vmTableChunk 21), exec_alu_xor_self (s := s3) hv0]
    have hr4 := regsOk_set hr1 (r := 0) (by decide) 0
    have hs4_0 : s4.reg 0 = some 0 := reg_set_self_ok' hr1 (by decide) _
    have hs4_4 : s4.reg 4 = some vc.sp := by
      simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp
    have hs4_15 : s4.reg 15 = some ctxBase := by
      simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact h15
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4 hs4_4 hs4_15
    obtain ⟨_, m5, _, _, h05, hwf5, hc448, _⟩ := vc.wr64_ctx_scratch_slot hwf4 448 0 4200927
      (by decide) (by decide) hs4_0 (fun hwr => by
        rw [step_at_vc 21 rfl (by decide : (4200920, (.ofInt (.st 15 448 0)), 7) ∈ vmTableChunk 21), exec_st hs4_15 hs4_0 hwr])
    let s5 := { s4 with mem := m5, rip := 4200927 }
    let s6 := { s5 with rip := SporosVmBin.L_vm_halt }
    have h06 : X86Vm.step vmImage s5 = .ok s6 := by
      rw [step_at_vc 21 rfl (by decide : (4200927, (.ofInt (.jmp 4200962)), 5) ∈ vmTableChunk 21), exec_jmp]; rfl
    let vc5 : VmCtx := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 448 (bytes8 0) } }
    obtain ⟨_, _, _, _, _, houtHi, _, _, _, hc200, hc208, _, _, _, horel, _⟩ := wf_io_info (vc := vc5) hwf5
    have hcb5 := vc5.ctx_bounds hwf5
    obtain ⟨se, hre, he, hke⟩ := L_vm_halt_spec (vc := vc5) (_pr := pr) (ls := ls) (s := s6) (exitCode := 0)
      rfl hwf5.1 hwf5.2.1 hwf5.2.2.2.1 hs4_15 (vc5.mem_ctx hwf5) hcb5.1 hcb5.2.2.1
      (vc5.mem_out hwf5) houtHi hc200 hc208 (by simpa [vc5] using hc448) horel
    exact ⟨se, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 hre))))), he, hke⟩
  | cons g rest' =>
    let rest := g :: rest'
    have hrest_le : rest.length + 1 ≤ vc.rsCap := by simpa [rest] using hrsLe
    have hrest_lb : rsLocBase rest = rsLocBase (g :: rest') := rfl
    let s3 := { s2 with rip := 4200932 }
    have h03 : X86Vm.step vmImage s2 = .ok s3 := by
      rw [step_at_vc 21 rfl (by decide : (4200916, (.ofInt (.jcc 7 4200932)), 2) ∈ vmTableChunk 21),
        exec_jcc (by simpa [s2, h1_mod] using hcond_a)]; simp [s2, s3]
    have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
      rw [show rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    let s4 := { s3.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := 4200936 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 22 rfl (by decide : (4200932, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 22), exec_aluI (s := s3) hs1_1]
      dsimp only [s4, rest]; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub1]
    have hr4 := regsOk_set hr1 (r := 1) (by decide) rest.length
    have hs4_1 : s4.reg 1 = some rest.length := reg_set_self_ok' hr1 (by decide) _
    have hs4_15 : s4.reg 15 = some ctxBase := by
      simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact h15
    have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
    have hea56 : ea ctxBase 56 = ctxBase + 56 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
    have hea48 : ea ctxBase 48 = ctxBase + 48 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
    have hea80 : ea ctxBase 80 = ctxBase + 80 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
    obtain ⟨_, m1, hwr1, _, _, hm1wf, hctx1_m, hoth1⟩ := wr64_region h.1.2 (vc.mem_ctx h) hcb.2.1
      (a := ctxBase + 56) (v := rest.length) (by rw [hcb.1]; omega) (by simp only [Region.hi, hcb.1]; omega)
    let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 56 (bytes8 rest.length) }
    have hctx1_m' : ctx1 ∈ m1 := by simpa [ctx1, hcb.1] using hctx1_m
    let s5 := { s4 with mem := m1, rip := 4200940 }
    have h05 : X86Vm.step vmImage s4 = .ok s5 := by
      rw [step_at_vc 22 rfl (by decide : (4200936, (.ofInt (.st 15 56 1)), 4) ∈ vmTableChunk 22), exec_st (s := s4) hs4_15 hs4_1 (by rw [hea56]; exact hwr1)]
    let s6 := { s5.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := 4200944 }
    have h06 : X86Vm.step vmImage s5 = .ok s6 := by
      rw [step_at_vc 22 rfl (by decide : (4200940, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk 22),
        exec_shlI_lt (s := s5) hs4_1 (by rw [show 2 ^ (5 % 64) = 32 by decide]; omega)]
    have hr6 := regsOk_set hr4 (r := 1) (by decide) (rest.length * 32)
    have hs6_1 : s6.reg 1 = some (rest.length * 32) := reg_set_self_ok' hr4 (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs4_15
    have hrd48 : rd64 m1 (ea ctxBase 48) = some vc.rsReg.base := by
      rw [hea48]; refine rd64_of_mem hm1wf hctx1_m' (by rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega) ?_
      dsimp [ctx1]; rw [hcb.1, Nat.add_sub_cancel_left, writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by omega)]
      exact ctx_leVal_of_rd64 h (by decide) hrsBase
    let s7 := { s6.set 0 (some vc.rsReg.base) with rip := 4200948 }
    have h07 : X86Vm.step vmImage s6 = .ok s7 := by
      rw [step_at_vc 22 rfl (by decide : (4200944, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk 22), exec_ld hs6_15 hrd48]
    have hr7 := regsOk_set hr6 (r := 0) (by decide) vc.rsReg.base
    have hs7_0 : s7.reg 0 = some vc.rsReg.base := reg_set_self_ok' hr6 (by decide) _
    have hs7_1 : s7.reg 1 = some (rest.length * 32) := by simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 1)]; exact hs6_1
    have hrs_m1 : vc.rsReg ∈ m1 := hoth1 _ (vc.mem_rs h) (vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    simp only [RsLocRel] at hrsRel
    have hrd_lb : rdMem .w64 m1 (eaIdx vc.rsReg.base (rest.length * 32) 1 16) = some (rsLocBase rest) := by
      have := hrsRel.2.1; rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_assoc]
      exact rdMem_w64_of_mem hm1wf hrs_m1 (by omega) (by simp only [Region.hi]; omega) (by simpa [rest, Nat.add_sub_cancel_left] using hrsRel.2.2.2.2.2.1)
    let s8 := { s7.set 2 (some (rsLocBase rest)) with rip := 4200953 }
    have h08 : X86Vm.step vmImage s7 = .ok s8 := by
      rw [step_at_vc 22 rfl (by decide : (4200948, (.ldIdx .w64 2 0 1 1 16), 5) ∈ vmTableChunk 22), exec_ldIdx hs7_0 hs7_1 hrd_lb]
    have hr8 := regsOk_set hr7 (r := 2) (by decide) (rsLocBase rest)
    have hs8_2 : s8.reg 2 = some (rsLocBase rest) := reg_set_self_ok' hr7 (by decide) _
    have hs8_15 : s8.reg 15 = some ctxBase := by
      simp only [s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs6_15
    obtain ⟨_, m2, hwr2, _, _, hm2wf, hctx2_m, hoth2⟩ := wr64_region hm1wf hctx1_m' hcb.2.1
      (a := ctxBase + 80) (v := rsLocBase rest) (by dsimp [ctx1]; rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega)
    let ctx2 := { vc.ctxReg with data := writeBytes ctx1.data 80 (bytes8 (rsLocBase rest)) }
    have hctx2_m' : ctx2 ∈ m2 := by simpa [ctx2, ctx1, hcb.1] using hctx2_m
    let s9 := { s8 with mem := m2, rip := 4200957 }
    have h09 : X86Vm.step vmImage s8 = .ok s9 := by
      rw [step_at_vc 22 rfl (by decide : (4200953, (.ofInt (.st 15 80 2)), 4) ∈ vmTableChunk 22), exec_st hs8_15 hs8_2 (by rw [hea80]; exact hwr2)]
    let s10 := { s9 with rip := SporosVmBin.L_vm_loop }
    have h10 : X86Vm.step vmImage s9 = .ok s10 := by
      rw [step_at_vc 22 rfl (by decide : (4200957, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk 22), exec_jmp]; rfl
    have hreach : Reaches vmImage s s10 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step h10)))))))))
    have hoth_all : ∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u ∈ m2 := fun u hu hc =>
      hoth2 _ (hoth1 u hu hc) (by simpa [ctx1] using hc)
    have hdisj : ∀ off, (off + 8 ≤ 56 ∨ (64 ≤ off ∧ off + 8 ≤ 80) ∨ 88 ≤ off) → off + 8 ≤ ctxSize →
        leVal ((ctx2.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8) := fun off ho hl => by
      dsimp [ctx2, ctx1, ctxSize] at hl ⊢
      have h1 : off + 8 ≤ 80 ∨ 80 + (bytes8 (rsLocBase rest)).length ≤ off := by
        rw [bytes8_length]; rcases ho with ha | ⟨_, hb⟩ | ha
        · exact Or.inl (by omega)
        · exact Or.inl hb
        · exact Or.inr (by omega)
      have h2 : off + 8 ≤ 56 ∨ 56 + (bytes8 rest.length).length ≤ off := by
        rw [bytes8_length]; rcases ho with ha | ⟨ha, _⟩ | ha
        · exact Or.inl ha
        · exact Or.inr (by omega)
        · exact Or.inr (by omega)
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ h1 (by rw [writeBytes_length]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ h2 (by omega)]
    have hsp10 : s10.reg 4 = some vc.sp := by
      simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp
    have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
    have hlb_le : rsLocBase rest ≤ vc.locCap := by have := hrsRel.1; omega
    have hwf10 : ({ vc with ctxReg := ctx2 } : VmCtx).wf pr { ls with rs := rest } s10 :=
      wf_update_rs_loc h hm2wf (bytesLt_wr64 (bytesLt_wr64 h.2.1 hwr1) hwr2) (len_pos_wrN (len_pos_wrN h.2.2.1 hwr1) hwr2)
        hr8 hsp10 hs8_15
        (hoth_all _ (vc.mem_stk h) (vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))) rfl rfl rfl
        hctx2_m' rfl rfl (by dsimp [ctx2, ctx1]; rw [writeBytes_length, writeBytes_length])
        (by rw [hdisj 48 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) hrsBase)
        (by dsimp [ctx2, ctx1]; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)])
        (by rw [hdisj 64 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) hrsCap)
        (by rw [hdisj 72 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s72)
        (by dsimp [ctx2, ctx1]; rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)])
        (by rw [hdisj 88 (Or.inr (Or.inr (by decide))) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s88)
        (fun off ho hl => hdisj off (by rcases ho with h1 | h1; exact Or.inl (by omega); exact Or.inr (Or.inr (by omega))) hl)
        (hoth_all _ (vc.mem_rs h) (vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)))
        hrsRw hrsPos hrsLenEq hrsPg hrsHi
        (hoth_all _ (vc.mem_loc h) (vc.ne_bases h (i := 9) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)))
        hlocRw hlocPos hlocLenEq hlocPg hlocHi hrsRel.2.2.2.2.2.2.2.2 (by omega) hlb_le
        (fun x hx v hv => (vc.code_flat h).2.2.2.2.2.2.1 x (by rw [hrs]; exact List.mem_cons_of_mem _ hx) v hv)
        h.2.2.2.2.2.2.1 (fun u hu _ h2 _ _ => hoth_all u hu h2)
    exact Or.inl ⟨s10, hreach, rfl, oracleAdmits_of_reaches hor hreach, rfl, _, hwf10⟩

end SporosVmOpsCtrl
