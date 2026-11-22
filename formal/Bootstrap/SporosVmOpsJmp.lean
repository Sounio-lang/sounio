import Bootstrap.SporosVmOpsCall

/-!
# Native x86-64 VM Jump & Conditional-Branch Proofs (`O_jmp`, `O_jz`, `O_jnz`)

Simulation proofs over `X86Vm` for `Sporos.Instr.jmp t` (`O_jmp`, `0x4016c1`),
`Sporos.Instr.jz t` (`O_jz`, `0x4016e4`), and `Sporos.Instr.jnz t` (`O_jnz`,
`0x401750`), sharing `wf_update_jcc` and `O_jcc_core_spec`.

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

/-- Simulation proof for `Sporos.Instr.jmp t` (`O_jmp`, `0x4016c1`). -/
theorem O_jmp_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {t : Nat}
    (hrip : s.rip = instrEntry (.jmp t)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length)
    (hins : pr.words[f.word]!.code[f.pc]! = .jmp t)
    (_hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.jmp t)).2) :
    CtrlOutcome pr ls s := by
  have hfetch := fetch_eq_some hnw hpc hins
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, hrsLe, hlocLe⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, hrsCap, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  have ⟨s72, s80, s88⟩ := wf_rd_loc_slots h
  have ⟨_, _, _, hfit, hwlt⟩ := vc.code_flat h
  have htW : t < X86Env.W := (hfit.2.2.2.2.2 _ (getElem!_mem_of_lt hnw)).2.2.2.2 _ (hins ▸ getElem!_mem_of_lt hpc)
  have h264' : rd64 s.mem (ea ctxBase 264) = some t := by simpa [instrPair, Nat.mod_eq_of_lt htW] using h264
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  rw [hrs] at hrsLen hrsRel hrsLe hlocLe
  have hrl_len : (rest.length + 1) * 32 ≤ vc.rsReg.data.length := by rw [hrsLenEq]; simp only [List.length_cons] at hrsLe; omega
  let s1 := { s.set 0 (some vc.rsReg.base) with rip := 4200133 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_vc 18 hrip (by decide : (4200129, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk 18), exec_ld h15 hrsBase, hrip]; rfl
  have hr1 := regsOk_set hr (r := 0) (by decide) vc.rsReg.base
  have hs1_0 : s1.reg 0 = some vc.rsReg.base := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  let s2 := { s1.set 1 (some (rest.length + 1)) with rip := 4200137 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 18 rfl (by decide : (4200133, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk 18), exec_ld hs1_15 (by simpa [s1, List.length_cons] using hrsLen)]
  have hr2 := regsOk_set hr1 (r := 1) (by decide) (rest.length + 1)
  have hs2_1 : s2.reg 1 = some (rest.length + 1) := reg_set_self_ok' hr1 (by decide) _
  have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
    rw [show rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  let s3 := { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := 4200141 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 18 rfl (by decide : (4200137, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 18), exec_aluI hs2_1]
    dsimp only [s3]; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub1]
  have hr3 := regsOk_set hr2 (r := 1) (by decide) rest.length
  have hs3_1 : s3.reg 1 = some rest.length := reg_set_self_ok' hr2 (by decide) _
  let s4 := { s3.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := 4200145 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by
    rw [step_at_vc 18 rfl (by decide : (4200141, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk 18),
      exec_shlI_lt (c := 5) hs3_1 (by rw [show 2 ^ (5 % 64) = 32 by decide]; omega)]
  have hr4 := regsOk_set hr3 (r := 1) (by decide) (rest.length * 32)
  have hs4_0 : s4.reg 0 = some vc.rsReg.base := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs1_0
  have hs4_1 : s4.reg 1 = some (rest.length * 32) := reg_set_self_ok' hr3 (by decide) _
  let s5 := { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := 4200148 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by
    rw [step_at_vc 18 rfl (by decide : (4200145, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk 18), exec_alu hs4_0 hs4_1]
    dsimp only [s5]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  have hr5 := regsOk_set hr4 (r := 0) (by decide) (vc.rsReg.base + rest.length * 32)
  have hs5_0 : s5.reg 0 = some (vc.rsReg.base + rest.length * 32) := reg_set_self_ok' hr4 (by decide) _
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs1_15
  let s6 := { s5.set 2 (some t) with rip := 4200155 }
  have h06 : X86Vm.step vmImage s5 = .ok s6 := by
    rw [step_at_vc 18 rfl (by decide : (4200148, (.ofInt (.ld 2 15 264)), 7) ∈ vmTableChunk 18), exec_ld hs5_15 h264']
  have hr6 := regsOk_set hr5 (r := 2) (by decide) t
  have hs6_0 : s6.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
    simp only [s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]; exact hs5_0
  have hs6_2 : s6.reg 2 = some t := reg_set_self_ok' hr5 (by decide) _
  obtain ⟨_, m', hwr, _, _, hm'wf, hrs'_mem, hoth⟩ := wr64_region (m := s.mem) (r := vc.rsReg)
    (a := ea (vc.rsReg.base + rest.length * 32) 8) (v := t) h.1.2 (vc.mem_rs h) hrsRw
    (by rw [ea_add_eq (by omega)]; omega) (by rw [ea_add_eq (by omega)]; simp only [Region.hi]; omega)
  let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 t) }
  have hrs'_mem' : rs' ∈ m' := by
    rw [ea_add_eq (by omega), show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] at hrs'_mem
    exact hrs'_mem
  let s7 := { s6 with mem := m', rip := 4200159 }
  have h07 : X86Vm.step vmImage s6 = .ok s7 := by
    rw [step_at_vc 18 rfl (by decide : (4200155, (.ofInt (.st 0 8 2)), 4) ∈ vmTableChunk 18), exec_st (s := s6) hs6_0 hs6_2 hwr]
  let s8 := { s7 with rip := SporosVmBin.L_vm_loop }
  have h08 : X86Vm.step vmImage s7 = .ok s8 := by
    rw [step_at_vc 19 rfl (by decide : (4200159, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk 19), exec_jmp]; rfl
  have hreach : Reaches vmImage s s8 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08)))))))
  have hsp8 : s8.reg 4 = some vc.sp := by
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp
  have h15_8 : s8.reg 15 = some ctxBase := by
    simp only [s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs5_15
  have hlb_eq : rsLocBase ({ f with pc := t } :: rest) = rsLocBase (f :: rest) := by simp [rsLocBase]
  have hwf8 : ({ vc with rsReg := rs' } : VmCtx).wf pr { ls with rs := { f with pc := t } :: rest } s8 :=
    wf_update_rs_loc h hm'wf (bytesLt_wr64 h.2.1 hwr) (len_pos_wrN h.2.2.1 hwr) hr6 hsp8 h15_8
      (hoth _ (vc.mem_stk h) (vc.ne_bases h (i := 1) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))) rfl rfl rfl
      (hoth _ (vc.mem_ctx h) (vc.ne_bases h (i := 2) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))) rfl rfl rfl
      (ctx_leVal_of_rd64 h (by decide) hrsBase) (by simpa [hrs] using ctx_leVal_of_rd64 h (by decide) (vc.rd_ctx_rs_code_slots h).2.1)
      (ctx_leVal_of_rd64 h (by decide) hrsCap) (ctx_leVal_of_rd64 h (by decide) s72)
      (by rw [hlb_eq, ← hrs]; exact ctx_leVal_of_rd64 h (by decide) s80) (ctx_leVal_of_rd64 h (by decide) s88)
      (fun _ _ _ => rfl) hrs'_mem' hrsRw hrsPos (by dsimp [rs']; rw [writeBytes_length]; exact hrsLenEq) hrsPg
      (by simpa [Region.hi, rs', writeBytes_length] using hrsHi)
      (hoth _ (vc.mem_loc h) (vc.ne_bases h (i := 9) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)))
      hlocRw hlocPos hlocLenEq hlocPg hlocHi (by rw [hlb_eq]; exact rsLocRel_setTop_pc hrsRel hrl_len htW)
      (by simpa using hrsLe) (by rw [hlb_eq]; exact hlocLe)
      (fun g hg v hv => by
        have hw := hwlt.2.2.1; rw [hrs] at hw
        rcases List.mem_cons.mp hg with rfl | hg
        · exact hw f (by simp) v hv
        · exact hw g (by simp [hg]) v hv)
      h.2.2.2.2.2.2.1 (fun u hu _ _ h3 _ => hoth u hu h3)
  unfold CtrlOutcome; simp only [SporosProvLen.lstep, hrs, hfetch, SporosProvLen.LSt.setTop]
  exact Or.inl ⟨s8, hreach, rfl, oracleAdmits_of_reaches hor hreach, rfl, _, hwf8⟩

/-- `VmCtx.wf` preservation for `O_jz`/`O_jnz`: pops 1 value from `ds`, updates
`ctx[8]` and `ctx[224]` (`ls.pc`), and sets top frame `pc` in `rsReg`. -/
private theorem wf_update_jcc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {ctx' rs' : Region} {c : SporosProvLen.LVal} {r : List SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {newPc newLPc : Nat}
    (h : vc.wf pr ls s) (hds : ls.ds = c :: r) (hrs : ls.rs = f :: rest)
    (hnewLPc : newLPc < X86Env.W) (hnewPc : newPc < X86Env.W)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hctx_eq : ctx' = { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 8 (bytes8 r.length)) 224 (bytes8 newLPc) })
    (hrs_eq : rs' = { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 newPc) })
    (hctx_m : ctx' ∈ s'.mem) (hrs_m : rs' ∈ s'.mem)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u.base ≠ vc.rsReg.base → u ∈ s'.mem) :
    ({ vc with ctxReg := ctx', rsReg := rs' } : VmCtx).wf pr
      { ls with ds := r, pc := newLPc, rs := { f with pc := newPc } :: rest } s' := by
  have hcode_ne_ctx : codeRegion.base ≠ vc.ctxReg.base := vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_rs : codeRegion.base ≠ vc.rsReg.base := vc.ne_bases h (i := 0) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1 hcode_ne_ctx hcode_ne_rs, hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, _, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, _, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hcl : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hdsHi' : vc.dsReg.base + vc.dsReg.data.length ≤ 2 ^ 47 := hds5
  have hmem' : ∀ u ∈ ({ vc with ctxReg := ctx', rsReg := rs' } : VmCtx).regions, u ∈ s'.mem := by
    intro u hu; simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    have hnd0 := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd0
    rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hctx_m | exact hrs_m
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
  have hdisj : ∀ off, (off + 8 ≤ 8 ∨ (16 ≤ off ∧ off + 8 ≤ 224) ∨ 232 ≤ off) → off + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8) := fun off ho hl => by
    rw [hctx_eq]; dsimp [ctxSize] at hl ⊢
    have h1 : off + 8 ≤ 224 ∨ 224 + (bytes8 newLPc).length ≤ off := by
      rw [bytes8_length]; rcases ho with ha | ⟨_, hb⟩ | ha
      · exact Or.inl (by omega)
      · exact Or.inl hb
      · exact Or.inr (by omega)
    have h2 : off + 8 ≤ 8 ∨ 8 + (bytes8 r.length).length ≤ off := by
      rw [bytes8_length]; rcases ho with ha | ⟨ha, _⟩ | ha
      · exact Or.inl ha
      · exact Or.inr (by omega)
      · exact Or.inr (by omega)
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ h1 (by rw [writeBytes_length]; omega),
        writeBytes_drop_take_disjoint _ _ _ _ _ h2 (by omega)]
  have keep : ∀ {off val : Nat}, (off + 8 ≤ 8 ∨ (16 ≤ off ∧ off + 8 ≤ 224) ∨ 232 ≤ off) → off + 8 ≤ ctxSize →
      leVal ((vc.ctxReg.data.drop off).take 8) = val → leVal ((ctx'.data.drop off).take 8) = val :=
    fun h1 h2 hv => (hdisj _ h1 h2).trans hv
  have hlb_eq : rsLocBase ({ f with pc := newPc } :: rest) = rsLocBase ls.rs := by simp [hrs, rsLocBase]
  have hpds' : PairListRel vc.dsReg.data r.length r := by
    have hdrop := pairListRel_drop hpds (k := 1) (by simp [hds])
    simpa [hds] using hdrop
  have hrsrel' : RsLocRel rs'.data vc.locReg.data (rsLocBase ({ f with pc := newPc } :: rest)) ({ f with pc := newPc } :: rest) := by
    rw [hrs_eq, hlb_eq, hrs]
    exact rsLocRel_setTop_pc (hrs ▸ hrsrel) (by rw [hrs3]; have := hrs ▸ hrsle; simp only [List.length_cons] at this; omega) hnewPc
  refine ⟨hcm', hbl', hpos', hr', hsp', h15', by simpa [VmCtx.regions, hctx_eq, hrs_eq] using hnodup, hmem',
    hstk1, hstk2, hstk3, hstk4,
    by rw [hctx_eq]; exact hctx1, by rw [hctx_eq]; exact hctx2,
    by rw [hctx_eq]; dsimp; rw [writeBytes_length, writeBytes_length]; exact hctx3,
    by rw [hctx_eq]; simpa [Region.hi, writeBytes_length] using hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    by rw [hrs_eq]; exact hrs1, hrs2, by rw [hrs_eq]; dsimp; rw [writeBytes_length]; exact hrs3, hrs4,
    by rw [hrs_eq]; simpa [Region.hi, writeBytes_length] using hrs5,
    hloc1, hloc2, hloc3, hloc4, hloc5, hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    keep (Or.inl (by decide)) (by decide) hsl0,
    by rw [hctx_eq]; dsimp; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega),
        leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by rw [hds] at hdsle; simp at hdsle; omega)],
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl2,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl3,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl4,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl5,
    by rw [hrs_eq]; exact keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl6,
    by simpa [hrs] using keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl7,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl8,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl9,
    by rw [hlb_eq]; exact keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl10,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl11,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl12,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl13,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl14,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl15,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl16,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl17,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl18,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl19,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl20,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl21,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl22,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl23,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl24,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl25,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl26,
    keep (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide) hsl27,
    by rw [hctx_eq]; dsimp; rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt hnewLPc],
    keep (Or.inr (Or.inr (by decide))) (by decide) hsl29,
    keep (Or.inr (Or.inr (by decide))) (by decide) hsl30,
    keep (Or.inr (Or.inr (by decide))) (by decide) hsl31,
    hpds', by dsimp; rw [hds] at hdsle; simp at hdsle; omega, hpaux, hauxle, hrsrel',
    by rw [hrs] at hrsle; simpa using hrsle, by rw [hlb_eq]; exact hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
    ⟨fun v hv => hwlt.1 v (by rw [hds]; exact List.mem_cons_of_mem _ hv), hwlt.2.1,
      fun g hg v hv => by have hw := hwlt.2.2.1; rw [hrs] at hw; rcases List.mem_cons.mp hg with rfl | hg; exact hw f (by simp) v hv; exact hw g (by simp [hg]) v hv,
      hwlt.2.2.2.1, hwlt.2.2.2.2.1, hnewLPc⟩, hfit⟩

/-- First 13 steps of `O_jz`/`O_jnz` (`a..a+58`) on non-empty `ds = c :: r`:
pops `c` from `ds`, updates `ctx[8] := r.length` and `ctx[224] := newLPc`,
returning abstract `ctx2`, `s13` with `r8 = c.w`. -/
private theorem O_jcc_pop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {c : SporosProvLen.LVal} {r : List SporosProvLen.LVal} {a : Nat}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hds : ls.ds = c :: r)
    (hm1 : (a, (.ofInt (.ld 1 15 8)), 4) ∈ vmTableChunk 19)
    (hm2 : (a + 4, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 19)
    (hm3 : (a + 7, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 19)
    (hm4 : (a + 13, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 19)
    (hm5 : (a + 17, (.ofInt (.st 15 8 1)), 4) ∈ vmTableChunk 19)
    (hm6 : (a + 21, (.ofInt (.shlI 1 4)), 4) ∈ vmTableChunk 19)
    (hm7 : (a + 25, (.ofInt (.ld 2 15 0)), 3) ∈ vmTableChunk 19)
    (hm8 : (a + 28, (.ldIdx .w64 8 2 1 1 0), 4) ∈ vmTableChunk 19)
    (hm9 : (a + 32, (.ldIdx .w64 9 2 1 1 8), 5) ∈ vmTableChunk 19)
    (hm10 : (a + 37, (.ofInt (.ld 10 15 224)), 7) ∈ vmTableChunk 19)
    (hm11 : (a + 44, (.ofInt (.alu .add 10 9)), 3) ∈ vmTableChunk 19)
    (hm12 : (a + 47, (.ofInt (.aluI .add 10 1)), 4) ∈ vmTableChunk 19)
    (hm13 : (a + 51, (.ofInt (.st 15 224 10)), 7) ∈ vmTableChunk 19) :
    let newLPc := (ls.pc + c.plen + 1) % Sporos.M
    ∃ (ctx2 : Region) (s13 : X86Vm.St),
      Reaches vmImage s s13 ∧ s13.rip = a + 58 ∧ s13.k = s.k ∧
      s13.mem.wf ∧ BytesLt s13.mem ∧ (∀ u ∈ s13.mem, 0 < u.data.length) ∧ RegsOk s13 ∧
      s13.reg 4 = some vc.sp ∧ s13.reg 8 = some c.w ∧ s13.reg 15 = some ctxBase ∧
      ctx2 = { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 8 (bytes8 r.length)) 224 (bytes8 newLPc) } ∧
      ctx2 ∈ s13.mem ∧
      (∀ u ∈ s.mem, u.base ≠ vc.ctxReg.base → u ∈ s13.mem) := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have hW : X86Env.W = 2 ^ 64 := rfl
  have ⟨_, _, hdsLenEq, _, hdsHi, hpds, hdsLe⟩ := (vc.arena_bounds h).1
  have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
  have hdsHi' : vc.dsReg.base + vc.dsReg.data.length ≤ 2 ^ 47 := hdsHi
  let s1 := { s.set 1 (some ls.ds.length) with rip := a + 4 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by rw [step_at_vc 19 hrip hm1, exec_ld h15 (vc.rd_ctx_dsLen h), hrip]
  have hr1 := regsOk_set hr (r := 1) (by decide) ls.ds.length
  have hs1_1 : s1.reg 1 = some ls.ds.length := reg_set_self_ok' hr (by decide) _
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := a + 7 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by rw [step_at_vc 19 rfl hm2, exec_test hs1_1 hs1_1]
  have hsp2 : s2.reg 4 = some vc.sp := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp
  have h15_2 : s2.reg 15 = some ctxBase := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact h15
  let s3 := { s2 with rip := a + 13 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 19 rfl hm3, exec_jcc (cond_aluF_and_self_e ls.ds.length)]; simp [hds, s2, s3]
  have hs3_1 : s3.reg 1 = some (r.length + 1) := by change s1.reg 1 = _; simpa [hds] using hs1_1
  have hsub_r : (r.length + 1 + X86Env.W - 1) % X86Env.W = r.length := by
    rw [hds] at hdsLe; simp at hdsLe; rw [show r.length + 1 + X86Env.W - 1 = r.length + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  let s4 := { s3.set 1 (some r.length) with fl := (X86Int.aluF .sub (r.length + 1) 1).2, rip := a + 17 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by
    rw [step_at_vc 19 rfl hm4, exec_aluI hs3_1]; dsimp only [s4]; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub_r]
  have hr4 := regsOk_set hr1 (r := 1) (by decide) r.length
  have hs4_1 : s4.reg 1 = some r.length := reg_set_self_ok' hr1 (by decide) _
  have hs4_15 : s4.reg 15 = some ctxBase := by simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact h15_2
  have hea8 : ea ctxBase 8 = ctxBase + 8 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  have hea224 : ea ctxBase 224 = ctxBase + 224 := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
  obtain ⟨_, m1, hwr1, _, _, hm1wf, hctx1_m, hoth1⟩ := wr64_region h.1.2 (vc.mem_ctx h) hcb.2.1
    (a := ctxBase + 8) (v := r.length) (by rw [hcb.1]; omega) (by simp only [Region.hi, hcb.1]; omega)
  let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 r.length) }
  have hctx1_m' : ctx1 ∈ m1 := by simpa [ctx1, hcb.1] using hctx1_m
  let s5 := { s4 with mem := m1, rip := a + 21 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by rw [step_at_vc 19 rfl hm5, exec_st (s := s4) hs4_15 hs4_1 (by rw [hea8]; exact hwr1)]
  have hrlen_lt : r.length * 16 + 16 ≤ vc.dsReg.data.length := by rw [hds] at hdsLe; simp at hdsLe; omega
  let s6 := { s5.set 1 (some (r.length * 16)) with fl := X86Int.Flags.undef, rip := a + 25 }
  have h06 : X86Vm.step vmImage s5 = .ok s6 := by
    rw [step_at_vc 19 rfl hm6, exec_shlI_lt (s := s5) hs4_1 (by rw [show 2 ^ (4 % 64) = 16 by decide]; omega)]
  have hr6 := regsOk_set hr4 (r := 1) (by decide) (r.length * 16)
  have hs6_1 : s6.reg 1 = some (r.length * 16) := reg_set_self_ok' hr4 (by decide) _
  have hs6_15 : s6.reg 15 = some ctxBase := by simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs4_15
  have hrd_c1 : ∀ off v, (off + 8 ≤ 8 ∨ 16 ≤ off) → off + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase off) = some v → rd64 m1 (ea ctxBase off) = some v := fun off v ho hl hv => by
    dsimp [ctxSize] at hl; rw [ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)]
    refine rd64_of_mem hm1wf hctx1_m' (by rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega) ?_
    dsimp [ctx1]; rw [hcb.1, Nat.add_sub_cancel_left, writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; rcases ho with h1 | h1; exact Or.inl h1; exact Or.inr h1) (by omega)]
    exact ctx_leVal_of_rd64 h (by dsimp [ctxSize]; omega) hv
  let s7 := { s6.set 2 (some vc.dsReg.base) with rip := a + 28 }
  have h07 : X86Vm.step vmImage s6 = .ok s7 := by
    rw [step_at_vc 19 rfl hm7, exec_ld hs6_15 (hrd_c1 0 _ (Or.inl (by decide)) (by decide) (vc.rd_ctx_dsBase h))]
  have hr7 := regsOk_set hr6 (r := 2) (by decide) vc.dsReg.base
  have hs7_2 : s7.reg 2 = some vc.dsReg.base := reg_set_self_ok' hr6 (by decide) _
  have hs7_1 : s7.reg 1 = some (r.length * 16) := by simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 1)]; exact hs6_1
  have hds_m1 : vc.dsReg ∈ m1 := hoth1 _ (vc.mem_ds h) (vc.ne_bases h (i := 6) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
  have htop := pairListRel_get_top hpds (k := 0) (by simp [hds])
  simp only [hds, List.length_cons, Nat.add_sub_cancel, Nat.sub_zero, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero, Option.getD_some] at htop
  have hrd_cw : rdMem .w64 m1 (eaIdx vc.dsReg.base (r.length * 16) 1 0) = some c.w := by
    rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_zero]
    exact rdMem_w64_of_mem hm1wf hds_m1 (by omega) (by simp only [Region.hi]; omega) (by simpa [Nat.add_sub_cancel_left] using htop.1)
  have hrd_cp : rdMem .w64 m1 (eaIdx vc.dsReg.base (r.length * 16) 1 8) = some c.plen := by
    rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_assoc]
    exact rdMem_w64_of_mem hm1wf hds_m1 (by omega) (by simp only [Region.hi]; omega) (by simpa [Nat.add_sub_cancel_left] using htop.2)
  let s8 := { s7.set 8 (some c.w) with rip := a + 32 }
  have h08 : X86Vm.step vmImage s7 = .ok s8 := by rw [step_at_vc 19 rfl hm8, exec_ldIdx hs7_2 hs7_1 hrd_cw]
  have hr8 := regsOk_set hr7 (r := 8) (by decide) c.w
  have hs8_2 : s8.reg 2 = some vc.dsReg.base := by simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 2)]; exact hs7_2
  have hs8_1 : s8.reg 1 = some (r.length * 16) := by simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 1)]; exact hs7_1
  let s9 := { s8.set 9 (some c.plen) with rip := a + 37 }
  have h09 : X86Vm.step vmImage s8 = .ok s9 := by rw [step_at_vc 19 rfl hm9, exec_ldIdx hs8_2 hs8_1 hrd_cp]
  have hr9 := regsOk_set hr8 (r := 9) (by decide) c.plen
  have hs9_15 : s9.reg 15 = some ctxBase := by
    simp only [s9, s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 9 ≠ 15),
      List.getElem?_set_ne (by decide : 8 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs6_15
  let s10 := { s9.set 10 (some ls.pc) with rip := a + 44 }
  have h10 : X86Vm.step vmImage s9 = .ok s10 := by
    rw [step_at_vc 19 rfl hm10, exec_ld hs9_15 (hrd_c1 224 _ (Or.inr (by decide)) (by decide) (rd_ctx_pc h))]
  have hr10 := regsOk_set hr9 (r := 10) (by decide) ls.pc
  have hs10_10 : s10.reg 10 = some ls.pc := reg_set_self_ok' hr9 (by decide) _
  have hs10_9 : s10.reg 9 = some c.plen := by simp only [s10, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 10 ≠ 9)]; exact reg_set_self_ok' hr8 (by decide) _
  let s11 := { s10.set 10 (some ((ls.pc + c.plen) % X86Env.W)) with fl := (X86Int.aluF .add ls.pc c.plen).2, rip := a + 47 }
  have h11 : X86Vm.step vmImage s10 = .ok s11 := by
    rw [step_at_vc 19 rfl hm11, exec_alu hs10_10 hs10_9]; dsimp only [s11]; rw [aluF_add_val]
  have hr11 := regsOk_set hr10 (r := 10) (by decide) ((ls.pc + c.plen) % X86Env.W)
  have hs11_10 : s11.reg 10 = some ((ls.pc + c.plen) % X86Env.W) := reg_set_self_ok' hr10 (by decide) _
  let newLPc := (ls.pc + c.plen + 1) % Sporos.M
  let s12 := { s11.set 10 (some newLPc) with fl := (X86Int.aluF .add ((ls.pc + c.plen) % X86Env.W) (1 % X86Env.W)).2, rip := a + 51 }
  have h12 : X86Vm.step vmImage s11 = .ok s12 := by
    rw [step_at_vc 19 rfl hm12, exec_aluI hs11_10]; dsimp only [s12, newLPc]; rw [aluF_add_val, add_pc_1_mod_eq]
  have hr12 := regsOk_set hr11 (r := 10) (by decide) newLPc
  have hs12_10 : s12.reg 10 = some newLPc := reg_set_self_ok' hr11 (by decide) _
  have hs12_15 : s12.reg 15 = some ctxBase := by
    simp only [s12, s11, s10, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 10 ≠ 15)]; exact hs9_15
  obtain ⟨_, m2, hwr2, _, _, hm2wf, hctx2_m, hoth2⟩ := wr64_region hm1wf hctx1_m' hcb.2.1
    (a := ctxBase + 224) (v := newLPc) (by dsimp [ctx1]; rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega)
  let ctx2 := { vc.ctxReg with data := writeBytes ctx1.data 224 (bytes8 newLPc) }
  have hctx2_m' : ctx2 ∈ m2 := by simpa [ctx2, ctx1, hcb.1] using hctx2_m
  let s13 := { s12 with mem := m2, rip := a + 58 }
  have h13 : X86Vm.step vmImage s12 = .ok s13 := by rw [step_at_vc 19 rfl hm13, exec_st hs12_15 hs12_10 (by rw [hea224]; exact hwr2)]
  have hreach13 : Reaches vmImage s s13 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step_trans h12 (Reaches.step h13))))))))))))
  refine ⟨ctx2, s13, hreach13, rfl, rfl, hm2wf, bytesLt_wr64 (bytesLt_wr64 h.2.1 hwr1) hwr2,
    len_pos_wrN (len_pos_wrN h.2.2.1 hwr1) hwr2, hr12, ?_, ?_, hs12_15, rfl, hctx2_m',
    fun u hu hc => hoth2 _ (hoth1 u hu hc) (by simpa [ctx1] using hc)⟩
  · simp only [s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 4), List.getElem?_set_ne (by decide : 9 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp2
  · simp only [s13, s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 8), List.getElem?_set_ne (by decide : 9 ≠ 8)]; exact reg_set_self_ok' hr7 (by decide) _

/-- Shared 25-instruction simulation for `O_jz` (`a = 4200164`, `isJz = true`)
and `O_jnz` (`a = 4200272`, `isJz = false`). -/
private theorem O_jcc_core_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {a t : Nat} (isJz : Bool)
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (htW : t < X86Env.W) (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some t)
    (hm1 : (a, (.ofInt (.ld 1 15 8)), 4) ∈ vmTableChunk 19)
    (hm2 : (a + 4, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 19)
    (hm3 : (a + 7, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 19)
    (hm4 : (a + 13, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 19)
    (hm5 : (a + 17, (.ofInt (.st 15 8 1)), 4) ∈ vmTableChunk 19)
    (hm6 : (a + 21, (.ofInt (.shlI 1 4)), 4) ∈ vmTableChunk 19)
    (hm7 : (a + 25, (.ofInt (.ld 2 15 0)), 3) ∈ vmTableChunk 19)
    (hm8 : (a + 28, (.ldIdx .w64 8 2 1 1 0), 4) ∈ vmTableChunk 19)
    (hm9 : (a + 32, (.ldIdx .w64 9 2 1 1 8), 5) ∈ vmTableChunk 19)
    (hm10 : (a + 37, (.ofInt (.ld 10 15 224)), 7) ∈ vmTableChunk 19)
    (hm11 : (a + 44, (.ofInt (.alu .add 10 9)), 3) ∈ vmTableChunk 19)
    (hm12 : (a + 47, (.ofInt (.aluI .add 10 1)), 4) ∈ vmTableChunk 19)
    (hm13 : (a + 51, (.ofInt (.st 15 224 10)), 7) ∈ vmTableChunk 19)
    (hm14 : (a + 58, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk 19)
    (hm15 : (a + 62, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk 19)
    (hm16 : (a + 66, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 19)
    (hm17 : (a + 70, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk 19)
    (hm18 : (a + 74, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk 19)
    (hm19 : (a + 77, (.ofInt (.ld 2 0 8)), 4) ∈ vmTableChunk 19)
    (hm20 : (a + 81, (.ofInt (.aluI .add 2 1)), 4) ∈ vmTableChunk 19)
    (hm21 : (a + 85, (.ofInt (.ld 6 15 264)), 7) ∈ vmTableChunk 19)
    (hm22 : (a + 92, (.ofInt (.test 8 8)), 3) ∈ vmTableChunk 19)
    (hm23 : (a + 95, (.cmovcc (if isJz then 4 else 5) 2 6), 4) ∈ vmTableChunk 19)
    (hm24 : (a + 99, (.ofInt (.st 0 8 2)), 4) ∈ vmTableChunk 19)
    (hm25 : (a + 103, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk 19) :
    match ls.ds with
    | [] => ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k
    | c :: r =>
      let ls' : SporosProvLen.LSt := { ls with
        ds := r, pc := (ls.pc + c.plen + 1) % Sporos.M,
        rs := { f with pc := if isJz then (if c.w % Sporos.M = 0 then t else f.pc + 1) else (if c.w % Sporos.M = 0 then f.pc + 1 else t) } :: rest }
      ∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s' := by
  cases hds : ls.ds with
  | nil =>
    have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
    let s1 := { s.set 1 (some ls.ds.length) with rip := a + 4 }
    have h01 : X86Vm.step vmImage s = .ok s1 := by rw [step_at_vc 19 hrip hm1, exec_ld h15 (vc.rd_ctx_dsLen h), hrip]
    have hr1 := regsOk_set hr (r := 1) (by decide) ls.ds.length
    have hs1_1 : s1.reg 1 = some ls.ds.length := reg_set_self_ok' hr (by decide) _
    let s2 := { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := a + 7 }
    have h02 : X86Vm.step vmImage s1 = .ok s2 := by rw [step_at_vc 19 rfl hm2, exec_test hs1_1 hs1_1]
    have hsp2 : s2.reg 4 = some vc.sp := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hsp
    have h15_2 : s2.reg 15 = some ctxBase := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact h15
    let s3 := { s2 with rip := SporosVmBin.L_fail_underflow }
    have h03 : X86Vm.step vmImage s2 = .ok s3 := by
      rw [step_at_vc 19 rfl hm3, exec_jcc (cond_aluF_and_self_e ls.ds.length)]; simp [hds, s2, s3]; rfl
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hsp2 h15_2
    obtain ⟨se, hre, he, hke⟩ := L_fail_from_vc_spec (e := .underflow) rfl hwf3
    exact ⟨se, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 hre)), he, hke⟩
  | cons c r =>
    have hW : X86Env.W = 2 ^ 64 := rfl
    have ⟨hrsRw, _, _, _, hrsHi, _, _, _, _, _, hrsRel, hrsLe, _⟩ := (vc.arena_bounds h).2.2
    have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
    have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
    have hc_lt := wf_ds_lt_W h (v := c) (by rw [hds]; exact List.mem_cons_self ..)
    obtain ⟨ctx2, s13, hreach13, hrip13, hk13, hm2wf, hbl2, hpos2, hr12, hs13_4, hs13_8, hs13_15, hctx2_eq, hctx2_m', hoth2⟩ :=
      O_jcc_pop_spec hrip h hds hm1 hm2 hm3 hm4 hm5 hm6 hm7 hm8 hm9 hm10 hm11 hm12 hm13
    have hrd_c2 : ∀ off v, ((16 ≤ off ∧ off + 8 ≤ 224) ∨ 232 ≤ off) → off + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase off) = some v → rd64 s13.mem (ea ctxBase off) = some v := fun off v ho hl hv => by
      dsimp [ctxSize] at hl; rw [ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)]
      refine rd64_of_mem hm2wf hctx2_m' (by rw [hctx2_eq, hcb.1]; omega) (by simp only [Region.hi, hctx2_eq, hcb.1, writeBytes_length]; omega) ?_
      rw [hctx2_eq]; dsimp; rw [hcb.1, Nat.add_sub_cancel_left]
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; rcases ho with ⟨_, hb⟩ | ha; exact Or.inl hb; exact Or.inr ha) (by rw [writeBytes_length]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; rcases ho with ⟨ha, _⟩ | ha; exact Or.inr (by omega); exact Or.inr (by omega)) (by omega)]
      exact ctx_leVal_of_rd64 h (by dsimp [ctxSize]; omega) hv
    have ⟨hrsBase, hrsLen, _, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
    rw [hrs] at hrsLen hrsRel hrsLe
    simp only [RsLocRel] at hrsRel
    let s14 := { s13.set 0 (some vc.rsReg.base) with rip := a + 62 }
    have h14 : X86Vm.step vmImage s13 = .ok s14 := by
      rw [step_at_vc 19 hrip13 hm14, exec_ld hs13_15 (hrd_c2 48 _ (Or.inl ⟨by decide, by decide⟩) (by decide) hrsBase), hrip13]
    have hr14 := regsOk_set hr12 (r := 0) (by decide) vc.rsReg.base
    have hs14_0 : s14.reg 0 = some vc.rsReg.base := reg_set_self_ok' hr12 (by decide) _
    have hs14_15 : s14.reg 15 = some ctxBase := by
      change (s13.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide : (0 : Nat) ≠ 15)]; exact hs13_15
    let s15 := { s14.set 1 (some (rest.length + 1)) with rip := a + 66 }
    have h15s : X86Vm.step vmImage s14 = .ok s15 := by
      rw [step_at_vc 19 rfl hm15, exec_ld hs14_15 (hrd_c2 56 _ (Or.inl ⟨by decide, by decide⟩) (by decide) (by simpa [List.length_cons] using hrsLen))]
    have hr15 := regsOk_set hr14 (r := 1) (by decide) (rest.length + 1)
    have hs15_1 : s15.reg 1 = some (rest.length + 1) := reg_set_self_ok' hr14 (by decide) _
    have hsub_rs : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
      have := hrsRel.2.1; rw [show rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    let s16 := { s15.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := a + 70 }
    have h16 : X86Vm.step vmImage s15 = .ok s16 := by
      rw [step_at_vc 19 rfl hm16, exec_aluI hs15_1]; dsimp only [s16]; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub_rs]
    have hr16 := regsOk_set hr15 (r := 1) (by decide) rest.length
    have hs16_1 : s16.reg 1 = some rest.length := reg_set_self_ok' hr15 (by decide) _
    let s17 := { s16.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := a + 74 }
    have h17 : X86Vm.step vmImage s16 = .ok s17 := by
      have := hrsRel.2.1; rw [step_at_vc 19 rfl hm17, exec_shlI_lt (c := 5) hs16_1 (by rw [show 2 ^ (5 % 64) = 32 by decide]; omega)]
    have hr17 := regsOk_set hr16 (r := 1) (by decide) (rest.length * 32)
    have hs17_0 : s17.reg 0 = some vc.rsReg.base := by
      simp only [s17, s16, s15, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs14_0
    have hs17_1 : s17.reg 1 = some (rest.length * 32) := reg_set_self_ok' hr16 (by decide) _
    let s18 := { s17.set 0 (some (vc.rsReg.base + rest.length * 32)) with fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := a + 77 }
    have h18 : X86Vm.step vmImage s17 = .ok s18 := by
      have := hrsRel.2.1; rw [step_at_vc 19 rfl hm18, exec_alu hs17_0 hs17_1]; dsimp only [s18]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
    have hr18 := regsOk_set hr17 (r := 0) (by decide) (vc.rsReg.base + rest.length * 32)
    have hs18_0 : s18.reg 0 = some (vc.rsReg.base + rest.length * 32) := reg_set_self_ok' hr17 (by decide) _
    have hrs_ne_ctx : vc.rsReg.base ≠ vc.ctxReg.base := vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hrs_m2 : vc.rsReg ∈ s13.mem := hoth2 _ (vc.mem_rs h) hrs_ne_ctx
    have hrd_fpc : rd64 s13.mem (ea (vc.rsReg.base + rest.length * 32) 8) = some f.pc := by
      have := hrsRel.2.1; rw [ea_add_eq (by omega)]
      exact rd64_of_mem hm2wf hrs_m2 (by omega) (by simp only [Region.hi]; omega)
        (by simpa [show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] using hrsRel.2.2.2.2.1)
    let s19 := { s18.set 2 (some f.pc) with rip := a + 81 }
    have h19 : X86Vm.step vmImage s18 = .ok s19 := by rw [step_at_vc 19 rfl hm19, exec_ld hs18_0 hrd_fpc]
    have hr19 := regsOk_set hr18 (r := 2) (by decide) f.pc
    have hs19_2 : s19.reg 2 = some f.pc := reg_set_self_ok' hr18 (by decide) _
    let s20 := { s19.set 2 (some (f.pc + 1)) with fl := (X86Int.aluF .add f.pc 1).2, rip := a + 85 }
    have h20 : X86Vm.step vmImage s19 = .ok s20 := by
      rw [step_at_vc 19 rfl hm20, exec_aluI_add_lt hs19_2 (by omega), Nat.mod_eq_of_lt hpcW]
    have hr20 := regsOk_set hr19 (r := 2) (by decide) (f.pc + 1)
    have hs20_15 : s20.reg 15 = some ctxBase := by
      simp only [s20, s19, s18, s17, s16, s15, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15),
        List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs14_15
    let s21 := { s20.set 6 (some t) with rip := a + 92 }
    have h21 : X86Vm.step vmImage s20 = .ok s21 := by
      rw [step_at_vc 19 rfl hm21, exec_ld hs20_15 (hrd_c2 264 _ (Or.inr (by decide)) (by decide) h264)]
    have hr21 := regsOk_set hr20 (r := 6) (by decide) t
    have hs21_8 : s21.reg 8 = some c.w := by
      change ((((((((s13.set 0 _).set 1 _).set 1 _).set 1 _).set 0 _).set 2 _).set 2 _).set 6 _).reg 8 = some c.w
      rw [reg_set_ne (by decide : (6 : Nat) ≠ 8), reg_set_ne (by decide : (2 : Nat) ≠ 8), reg_set_ne (by decide : (2 : Nat) ≠ 8),
        reg_set_ne (by decide : (0 : Nat) ≠ 8), reg_set_ne (by decide : (1 : Nat) ≠ 8), reg_set_ne (by decide : (1 : Nat) ≠ 8),
        reg_set_ne (by decide : (1 : Nat) ≠ 8), reg_set_ne (by decide : (0 : Nat) ≠ 8)]; exact hs13_8
    let s22 := { s21 with fl := (X86Int.aluF .and c.w c.w).2, rip := a + 95 }
    have h22 : X86Vm.step vmImage s21 = .ok s22 := by rw [step_at_vc 19 rfl hm22, exec_test hs21_8 hs21_8]
    have hs22_6 : s22.reg 6 = some t := reg_set_self_ok' hr20 (by decide) _
    have hs22_2 : s22.reg 2 = some (f.pc + 1) := by
      simp only [s22, s21, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 2)]; exact reg_set_self_ok' hr19 (by decide) _
    let newPc := if isJz then (if c.w % Sporos.M = 0 then t else f.pc + 1) else (if c.w % Sporos.M = 0 then f.pc + 1 else t)
    have hcmov_val : (if (if isJz then decide (c.w = 0) else decide (c.w ≠ 0)) then t else f.pc + 1) = newPc := by
      have hcw_mod : c.w % Sporos.M = c.w := Nat.mod_eq_of_lt (by simpa [M_eq_W'] using hc_lt.1)
      cases isJz <;> by_cases hz : c.w = 0 <;> simp [newPc, hcw_mod, hz]
    have hcond_cmov : X86Int.cond s22.fl (if isJz then 4 else 5) = some (if isJz then decide (c.w = 0) else decide (c.w ≠ 0)) := by
      cases isJz
      · exact cond_aluF_and_self_ne c.w
      · exact cond_aluF_and_self_e c.w
    let s23 := { s22.set 2 (some newPc) with rip := a + 99 }
    have h23 : X86Vm.step vmImage s22 = .ok s23 := by
      rw [step_at_vc 19 rfl hm23, exec_cmovcc hcond_cmov hs22_6 hs22_2, hcmov_val]
    have hr23 := regsOk_set hr21 (r := 2) (by decide) newPc
    have hs23_0 : s23.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
      simp only [s23, s22, s21, s20, s19, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0),
        List.getElem?_set_ne (by decide : 6 ≠ 0)]; exact hs18_0
    have hs23_2 : s23.reg 2 = some newPc := reg_set_self_ok' hr21 (by decide) _
    obtain ⟨_, m3, hwr3, _, _, hm3wf, hrs'_m, hoth3⟩ := wr64_region hm2wf hrs_m2 hrsRw
      (a := ea (vc.rsReg.base + rest.length * 32) 8) (v := newPc)
      (by have := hrsRel.2.1; rw [ea_add_eq (by omega)]; omega)
      (by have := hrsRel.2.1; rw [ea_add_eq (by omega)]; simp only [Region.hi]; omega)
    let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 newPc) }
    have hrs'_m' : rs' ∈ m3 := by
      have := hrsRel.2.1
      rw [ea_add_eq (by omega), show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] at hrs'_m
      exact hrs'_m
    let s24 := { s23 with mem := m3, rip := a + 103 }
    have h24 : X86Vm.step vmImage s23 = .ok s24 := by rw [step_at_vc 19 rfl hm24, exec_st (s := s23) hs23_0 hs23_2 hwr3]
    let s25 := { s24 with rip := SporosVmBin.L_vm_loop }
    have h25 : X86Vm.step vmImage s24 = .ok s25 := by rw [step_at_vc 19 rfl hm25, exec_jmp]; rfl
    have hreach : Reaches vmImage s s25 :=
      Reaches.trans hreach13 (Reaches.step_trans h14 (Reaches.step_trans h15s (Reaches.step_trans h16
        (Reaches.step_trans h17 (Reaches.step_trans h18 (Reaches.step_trans h19 (Reaches.step_trans h20
          (Reaches.step_trans h21 (Reaches.step_trans h22 (Reaches.step_trans h23
            (Reaches.step_trans h24 (Reaches.step h25))))))))))))
    have hsp25 : s25.reg 4 = some vc.sp := by
      change (((((((((s13.set 0 _).set 1 _).set 1 _).set 1 _).set 0 _).set 2 _).set 2 _).set 6 _).set 2 _).reg 4 = some vc.sp
      rw [reg_set_ne (by decide : (2 : Nat) ≠ 4), reg_set_ne (by decide : (6 : Nat) ≠ 4), reg_set_ne (by decide : (2 : Nat) ≠ 4),
        reg_set_ne (by decide : (2 : Nat) ≠ 4), reg_set_ne (by decide : (0 : Nat) ≠ 4), reg_set_ne (by decide : (1 : Nat) ≠ 4),
        reg_set_ne (by decide : (1 : Nat) ≠ 4), reg_set_ne (by decide : (1 : Nat) ≠ 4), reg_set_ne (by decide : (0 : Nat) ≠ 4)]; exact hs13_4
    have h15_25 : s25.reg 15 = some ctxBase := by
      simp only [s25, s24, s23, s22, s21, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15),
        List.getElem?_set_ne (by decide : 6 ≠ 15)]; exact hs20_15
    have hnewPcW : newPc < X86Env.W := by dsimp [newPc]; cases isJz <;> simp <;> split <;> assumption
    have hctx2_b : ctx2.base = vc.ctxReg.base := by rw [hctx2_eq]
    have hwf25 := wf_update_jcc (s' := s25) h hds hrs (Nat.mod_lt _ (by decide)) hnewPcW hm3wf
      (bytesLt_wr64 hbl2 hwr3) (len_pos_wrN hpos2 hwr3)
      hr23 hsp25 h15_25 hctx2_eq rfl (hoth3 ctx2 hctx2_m' (by rw [hctx2_b]; exact hrs_ne_ctx.symm)) hrs'_m'
      (fun u hu hc hr' => hoth3 _ (hoth2 u hu hc) hr')
    exact ⟨s25, hreach, by change s13.k.out = s.k.out; rw [hk13], oracleAdmits_of_reaches hor hreach, rfl, _, hwf25⟩

/-- Simulation proof for `Sporos.Instr.jz t` (`O_jz`, `0x4016e4`). -/
theorem O_jz_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {t : Nat}
    (hrip : s.rip = instrEntry (.jz t)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length)
    (hins : pr.words[f.word]!.code[f.pc]! = .jz t) (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.jz t)).2) : CtrlOutcome pr ls s := by
  have hfetch := fetch_eq_some hnw hpc hins
  have htW : t < X86Env.W := ((vc.code_flat h).2.2.2.1.2.2.2.2.2 _ (getElem!_mem_of_lt hnw)).2.2.2.2 _ (hins ▸ getElem!_mem_of_lt hpc)
  have h264' : rd64 s.mem (ea ctxBase 264) = some t := by simpa [instrPair, Nat.mod_eq_of_lt htW] using h264
  have hcore := O_jcc_core_spec (a := 4200164) true hrip h hor hrs htW hpcW h264'
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  unfold CtrlOutcome; simp only [SporosProvLen.lstep, hrs, hfetch]
  cases hds : ls.ds with
  | nil => simpa [hds] using hcore
  | cons c r => simp only [hds] at hcore; exact Or.inl (by simpa [SporosProvLen.LSt.setTop, hrs] using hcore)

/-- Simulation proof for `Sporos.Instr.jnz t` (`O_jnz`, `0x401750`). -/
theorem O_jnz_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {t : Nat}
    (hrip : s.rip = instrEntry (.jnz t)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length)
    (hins : pr.words[f.word]!.code[f.pc]! = .jnz t) (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.jnz t)).2) : CtrlOutcome pr ls s := by
  have hfetch := fetch_eq_some hnw hpc hins
  have htW : t < X86Env.W := ((vc.code_flat h).2.2.2.1.2.2.2.2.2 _ (getElem!_mem_of_lt hnw)).2.2.2.2 _ (hins ▸ getElem!_mem_of_lt hpc)
  have h264' : rd64 s.mem (ea ctxBase 264) = some t := by simpa [instrPair, Nat.mod_eq_of_lt htW] using h264
  have hcore := O_jcc_core_spec (a := 4200272) false hrip h hor hrs htW hpcW h264'
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  unfold CtrlOutcome; simp only [SporosProvLen.lstep, hrs, hfetch]
  cases hds : ls.ds with
  | nil => simpa [hds] using hcore
  | cons c r => simp only [hds] at hcore; exact Or.inl (by simpa [SporosProvLen.LSt.setTop, hrs] using hcore)

end SporosVmOpsCtrl
