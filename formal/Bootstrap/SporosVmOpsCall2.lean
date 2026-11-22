import Bootstrap.SporosVmOpsCall1

/-!
# Native x86-64 VM `O_call` Locals-Growth, Caller-PC, and Locals-Init Helpers

Mid-stage helper specifications for `Sporos.Instr.call w` (`O_call`, `0x4017bc`):
locals-arena growth loop (`call_locs_chk_loop_spec`), caller PC advance
(`call_adv_caller_pc_spec`), and zero-initialized callee locals loop
(`call_init_locs_loop_spec`).

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

private theorem step_locs_chk_hdr {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w nLocs : Nat} (hnLocs : nLocs < 2 ^ 44)
    (hrip : s.rip = 4200709) (h : vc.wf pr ls s)
    (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs) :
    ∃ s5, Reaches vmImage s s5 ∧
      s5.rip = (if rsLocBase ls.rs + nLocs ≤ vc.locCap then 4200766 else 4200725) ∧
      s5.mem = s.mem ∧ s5.k = s.k ∧ vc.wf pr ls s5 ∧
      s5.reg 1 = some vc.locCap ∧ s5.reg 12 = some w ∧ s5.reg 13 = some nLocs := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨_, _, _, _, _, _, _, hlocLenEq, _, hlocHi, _, _, hlocLe⟩ := (vc.arena_bounds h).2.2
  have ⟨_, hlocLen, hlocCap⟩ := wf_rd_loc_slots h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
  let s1 := { s.set 0 (some (rsLocBase ls.rs)) with rip := 4200713 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_vc 21 hrip (by decide : (4200709, (.ofInt (.ld 0 15 80)), 4) ∈ vmTableChunk 21), exec_ld h15 hlocLen, hrip]
  have hr1 := regsOk_set hr (r := 0) (by decide) (rsLocBase ls.rs)
  have hs1_0 : s1.reg 0 = some (rsLocBase ls.rs) := reg_set_self_ok' hr (by decide) _
  have hs1_13 : s1.reg 13 = some nLocs := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13
  let s2 := { s1.set 0 (some (rsLocBase ls.rs + nLocs)) with fl := (X86Int.aluF .add (rsLocBase ls.rs) nLocs).2, rip := 4200716 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 21 rfl (by decide : (4200713, (.ofInt (.alu .add 0 13)), 3) ∈ vmTableChunk 21), exec_alu hs1_0 hs1_13]
    dsimp only [s2]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  have hr2 := regsOk_set hr1 (r := 0) (by decide) (rsLocBase ls.rs + nLocs)
  have hs2_15 : s2.reg 15 = some ctxBase := by simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  let s3 := { s2.set 1 (some vc.locCap) with rip := 4200720 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 21 rfl (by decide : (4200716, (.ofInt (.ld 1 15 88)), 4) ∈ vmTableChunk 21), exec_ld hs2_15 hlocCap]
  have hr3 := regsOk_set hr2 (r := 1) (by decide) vc.locCap
  have hs3_0 : s3.reg 0 = some (rsLocBase ls.rs + nLocs) := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact reg_set_self_ok' hr1 (by decide) _
  have hs3_1 : s3.reg 1 = some vc.locCap := reg_set_self_ok' hr2 (by decide) _
  let s4 := { s3 with fl := (X86Int.aluF .sub (rsLocBase ls.rs + nLocs) vc.locCap).2, rip := 4200723 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by
    rw [step_at_vc 21 rfl (by decide : (4200720, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 21), exec_cmp hs3_0 hs3_1]
  let s5 := { s4 with rip := if rsLocBase ls.rs + nLocs ≤ vc.locCap then 4200766 else 4200725 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by
    rw [step_at_vc 21 rfl (by decide : (4200723, (.ofInt (.jcc 6 4200766)), 2) ∈ vmTableChunk 21),
      exec_jcc (cond_aluF_sub_be _ _ (by omega) (by omega))]
    by_cases hle : rsLocBase ls.rs + nLocs ≤ vc.locCap <;> simp [hle, s4, s5]
  have hsp5 : s5.reg 4 = some vc.sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp
  have h15_5 : s5.reg 15 = some ctxBase := by simp only [s5, s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs2_15
  refine ⟨s5, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step h05)))),
    rfl, rfl, rfl, vc.wf_step_misc h rfl hr3 hsp5 h15_5, hs3_1, ?_, ?_⟩
  · simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hr12
  · simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13

/-- `O_call` locals-arena growth loop (`4200709..4200766`): loops doubling
`locReg` via `fn_mremap` until `rsLocBase ls.rs + nLocs ≤ vc'.locCap`. -/
theorem call_locs_chk_loop_spec {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {w nLocs : Nat}
    (hnLocs : nLocs < 2 ^ 44) :
    ∀ (fuel : Nat) (vc : VmCtx) (s : X86Vm.St),
      rsLocBase ls.rs + nLocs - vc.locCap ≤ fuel →
      s.rip = 4200709 → vc.wf pr ls s → OracleAdmits s →
      s.reg 12 = some w → s.reg 13 = some nLocs →
      (∃ (vc' : VmCtx) (s' : X86Vm.St), Reaches vmImage s s' ∧ s'.rip = 4200766 ∧ s'.k.out = s.k.out ∧
        OracleAdmits s' ∧ vc'.wf pr ls s' ∧ vc'.rsCap = vc.rsCap ∧
        rsLocBase ls.rs + nLocs ≤ vc'.locCap ∧
        s'.reg 12 = some w ∧ s'.reg 13 = some nLocs) ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  intro fuel
  induction fuel with
  | zero =>
    intro vc s hfuel hrip h hor hr12 hr13
    have hle : rsLocBase ls.rs + nLocs ≤ vc.locCap := by omega
    obtain ⟨s5, hr5, hrip5, _, hk5, hwf5, _, h12_5, h13_5⟩ := step_locs_chk_hdr hnLocs hrip h hr12 hr13
    rw [if_pos hle] at hrip5
    exact Or.inl ⟨vc, s5, hr5, hrip5, by rw [hk5], oracleAdmits_of_reaches hor hr5, hwf5, rfl, hle, h12_5, h13_5⟩
  | succ fuel' ih =>
    intro vc s hfuel hrip h hor hr12 hr13
    by_cases hle : rsLocBase ls.rs + nLocs ≤ vc.locCap
    · exact ih vc s (by omega) hrip h hor hr12 hr13
    · obtain ⟨s5, hr5, hrip5, hm5, hk5, hwf5, hr1_5, hr12_5, hr13_5⟩ := step_locs_chk_hdr hnLocs hrip h hr12 hr13
      rw [if_neg hle] at hrip5
      have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, hrsLe, _⟩ := (vc.arena_bounds h).2.2
      have ⟨hlocBase, hlocLen, hlocCap⟩ := wf_rd_loc_slots h
      have ⟨s48, s56, s64, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
      have hW : X86Env.W = 2 ^ 64 := rfl
      have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
      rcases step_mremap_ctx_double (a := 4200725) (offB := 72) (offC := 88) (sh := 4) (cap := vc.locCap) (reg := vc.locReg)
        hrip5 hwf5 (oracleAdmits_of_reaches hor hr5) hr1_5 (by omega) (by decide) (by decide) (by decide)
        (vc.mem_loc hwf5) hlocRw hlocPos hlocLenEq hlocPg hlocHi
        (vc.ne_bases h (i := 1) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (vc.ne_bases h (i := 2) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (hm5 ▸ hlocBase) (hm5 ▸ hlocCap)
        (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
        (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
        (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
        (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
        (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide)) with
        ⟨nb, stk1, ctx2, locReg', s6, hrm, hrip6, hk6, _, hmwf6, hbl6, hpos6, hr6, hsp6, h15_6, hroth,
          hstk_m, hstkb, hstkrw, hstkl, hctx2_eq, hctx2_m, hloc'_eq, hloc'_m, hnb_hi, hnb_ne, hoth⟩ | ⟨hhit, hend⟩
      · let s7 := { s6 with rip := 4200709 }
        have h07 : X86Vm.step vmImage s6 = .ok s7 := by
          rw [step_at_vc 21 hrip6 (by decide : (4200764, (.ofInt (.jmp 4200709)), 2) ∈ vmTableChunk 21), exec_jmp]
        have hreach7 := Reaches.trans hr5 (Reaches.trans hrm (Reaches.step h07))
        have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
        have hrs_m : vc.rsReg ∈ s6.mem := hoth vc.rsReg (vc.mem_rs hwf5)
          (vc.ne_bases h (i := 8) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
          (vc.ne_bases h (i := 8) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
          (vc.ne_bases h (i := 8) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        let vc1 : VmCtx := { vc with stk := stk1, ctxReg := ctx2, locReg := locReg', locCap := 2 * vc.locCap }
        have hnd' : (vc1.regions.map Region.base).Nodup := by
          let pfx : List Region := [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3, vc.dsReg, vc.auxReg, vc.rsReg]
          let sfx : List Region := [vc.memwReg, vc.mempReg, vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]
          have hnd0 : (pfx.map Region.base ++ vc.locReg.base :: sfx.map Region.base).Nodup := by simpa [pfx, sfx, VmCtx.regions] using h.2.2.2.2.2.2.1
          have hgoal := nodup_replace_at (b := nb) hnd0 (fun x hx => by
            rw [List.mem_append, List.mem_map, List.mem_map] at hx
            rcases hx with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
            · have hu_in : u ∈ vc.regions := by simp only [pfx, List.mem_cons, List.not_mem_nil, or_false] at hu; rcases hu with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> simp [VmCtx.regions]
              exact hnb_ne u (hwf5.2.2.2.2.2.2.2.1 u hu_in) (by rw [List.nodup_append] at hnd0; exact hnd0.2.2 u.base (List.mem_map_of_mem hu) vc.locReg.base (List.mem_cons_self ..))
            · have hu_in : u ∈ vc.regions := by simp only [sfx, List.mem_cons, List.not_mem_nil, or_false] at hu; rcases hu with rfl|rfl|rfl|rfl|rfl|rfl <;> simp [VmCtx.regions]
              exact hnb_ne u (hwf5.2.2.2.2.2.2.2.1 u hu_in) (by rw [List.nodup_append, List.nodup_cons] at hnd0; intro heq; exact hnd0.2.1.1 (heq ▸ List.mem_map_of_mem hu)))
          simpa [vc1, pfx, sfx, VmCtx.regions, hstkb, hctx2_eq, hloc'_eq] using hgoal
        have hdisj : ∀ off, (off + 8 ≤ 72 ∨ (80 ≤ off ∧ off + 8 ≤ 88) ∨ 96 ≤ off) → off + 8 ≤ ctxSize →
            leVal ((ctx2.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8) := fun off ho hl => by
          rw [hctx2_eq]; dsimp [ctxSize] at hl ⊢
          have hc1 : off + 8 ≤ 88 ∨ 88 + (bytes8 (2 * vc.locCap)).length ≤ off := by
            rw [bytes8_length]; rcases ho with h1 | ⟨h1, h2⟩ | h1
            · exact Or.inl (by omega)
            · exact Or.inl (by omega)
            · exact Or.inr (by omega)
          have hc2 : off + 8 ≤ 72 ∨ 72 + (bytes8 nb).length ≤ off := by
            rw [bytes8_length]; rcases ho with h1 | ⟨h1, h2⟩ | h1
            · exact Or.inl (by omega)
            · exact Or.inr (by omega)
            · exact Or.inr (by omega)
          rw [writeBytes_drop_take_disjoint _ _ _ _ _ hc1 (by rw [writeBytes_length]; omega),
              writeBytes_drop_take_disjoint _ _ _ _ _ hc2 (by omega)]
        have hwf7 : vc1.wf pr ls s7 := wf_update_rs_loc h hmwf6 hbl6 hpos6 hr6 hsp6 h15_6 hstk_m hstkb hstkrw hstkl
          hctx2_m (by rw [hctx2_eq]) (by rw [hctx2_eq]) (by rw [hctx2_eq]; dsimp; rw [writeBytes_length, writeBytes_length])
          (by rw [hdisj 48 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s48)
          (by rw [hdisj 56 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s56)
          (by rw [hdisj 64 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s64)
          (by rw [hctx2_eq, hloc'_eq]; dsimp; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega), leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)])
          (by rw [hdisj 80 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) hlocLen)
          (by rw [hctx2_eq]; dsimp; rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)])
          (fun off ho hl => hdisj off (by rcases ho with h1 | h1; exact Or.inl (by omega); exact Or.inr (Or.inr h1)) hl)
          hrs_m hrsRw hrsPos hrsLenEq hrsPg hrsHi
          hloc'_m (by rw [hloc'_eq]) (by omega) (by rw [hloc'_eq]; dsimp; rw [List.length_append, List.length_replicate, hlocLenEq]; omega)
          (by rw [show (2 * vc.locCap) * 16 = 2 * (vc.locCap * 16) by omega, Nat.mul_mod, hlocPg]; decide)
          (by rw [hloc'_eq]; simpa [Region.hi, List.length_append, List.length_replicate, hlocLenEq, show vc.locCap * 16 + vc.locCap * 16 = 2 * (vc.locCap * 16) by omega] using hnb_hi)
          (by rw [hloc'_eq]; exact rsLocRel_extend_loc_zero hrsRel (by rw [hlocLenEq]; omega)) hrsLe (by omega)
          (vc.code_flat h).2.2.2.2.2.2.1 hnd' (fun u hu h1 h2 _ h4 => hoth u (hm5 ▸ hu) h1 h2 h4)
        have h12_7 : s7.reg 12 = some w := by
          change s6.reg 12 = some w
          rw [hroth 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr12_5
        have h13_7 : s7.reg 13 = some nLocs := by
          change s6.reg 13 = some nLocs
          rw [hroth 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr13_5
        rcases ih vc1 s7 (by dsimp [vc1]; omega) rfl hwf7 (oracleAdmits_of_reaches hor hreach7) h12_7 h13_7 with
          ⟨vc', s', hrt, hrip', hk', hor', hwf', hrsc', hlocc', h12', h13'⟩ | ⟨hhit', hend'⟩
        · exact Or.inl ⟨vc', s', Reaches.trans hreach7 hrt, hrip', by rw [hk']; change s6.k.out = s.k.out; rw [hk6, hk5], hor', hwf', hrsc', hlocc', h12', h13'⟩
        · exact Or.inr ⟨hitsNomem_of_reaches hreach7 hhit', Ends.of_reaches hreach7 hend'⟩
      · exact Or.inr ⟨hitsNomem_of_reaches hr5 hhit, Ends.of_reaches hr5 hend⟩

/-- Straight-line 8-step caller PC advance (`4200766..4200797`) inside `O_call`,
advancing top-frame `f.pc` to `f.pc + 1` while preserving `r12` and `r13`. -/
theorem call_adv_caller_pc_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {w nLocs : Nat}
    (hrip : s.rip = 4200766) (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest)
    (hpcW : f.pc + 1 < X86Env.W) (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs) :
    let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) }
    let ls' := { ls with rs := { f with pc := f.pc + 1 } :: rest }
    ∃ s', Reaches vmImage s s' ∧ s'.rip = 4200797 ∧ s'.k = s.k ∧
      ({ vc with rsReg := rs' } : VmCtx).wf pr ls' s' ∧
      s'.reg 12 = some w ∧ s'.reg 13 = some nLocs := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨hrsRw, _, _, _, hrsHi, _, _, _, _, _, hrs_rel, hrs_le, _⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, _, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  rw [hrs] at hrsLen hrs_rel hrs_le
  simp only [RsLocRel] at hrs_rel
  rcases hrs_rel with ⟨_, hrl_len, _, _, hfpc_lt, _, _, _, _⟩
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
  let s1 := { s.set 0 (some vc.rsReg.base) with rip := 4200770 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_vc 21 hrip (by decide : (4200766, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk 21), exec_ld h15 hrsBase, hrip]
  have hr1 := regsOk_set hr (r := 0) (by decide) vc.rsReg.base
  have hs1_0 : s1.reg 0 = some vc.rsReg.base := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  let s2 := { s1.set 1 (some (rest.length + 1)) with rip := 4200774 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 21 rfl (by decide : (4200770, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk 21), exec_ld hs1_15 (by simpa [s1, List.length_cons] using hrsLen)]
  have hr2 := regsOk_set hr1 (r := 1) (by decide) (rest.length + 1)
  have hs2_1 : s2.reg 1 = some (rest.length + 1) := reg_set_self_ok' hr1 (by decide) _
  have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
    rw [show rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  let s3 := { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := 4200778 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 21 rfl (by decide : (4200774, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk 21), exec_aluI hs2_1]
    dsimp only [s3]; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub1]
  have hr3 := regsOk_set hr2 (r := 1) (by decide) rest.length
  have hs3_1 : s3.reg 1 = some rest.length := reg_set_self_ok' hr2 (by decide) _
  let s4 := { s3.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := 4200782 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by
    rw [step_at_vc 21 rfl (by decide : (4200778, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk 21),
      exec_shlI_lt (c := 5) hs3_1 (by rw [show 2 ^ (5 % 64) = 32 by decide]; omega)]
  have hr4 := regsOk_set hr3 (r := 1) (by decide) (rest.length * 32)
  have hs4_0 : s4.reg 0 = some vc.rsReg.base := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs1_0
  have hs4_1 : s4.reg 1 = some (rest.length * 32) := reg_set_self_ok' hr3 (by decide) _
  let s5 := { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := 4200785 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by
    rw [step_at_vc 21 rfl (by decide : (4200782, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk 21), exec_alu hs4_0 hs4_1]
    dsimp only [s5]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  have hr5 := regsOk_set hr4 (r := 0) (by decide) (vc.rsReg.base + rest.length * 32)
  have hs5_0 : s5.reg 0 = some (vc.rsReg.base + rest.length * 32) := reg_set_self_ok' hr4 (by decide) _
  have hrd_pc : rd64 s5.mem (ea (vc.rsReg.base + rest.length * 32) 8) = some f.pc := by
    change rd64 s.mem (ea (vc.rsReg.base + rest.length * 32) 8) = some f.pc
    rw [ea_add_eq (by omega)]
    exact rd64_of_mem h.1.2 (vc.mem_rs h) (by omega) (by simp only [Region.hi]; omega)
      (by simpa [show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] using hfpc_lt)
  let s6 := { s5.set 1 (some f.pc) with rip := 4200789 }
  have h06 : X86Vm.step vmImage s5 = .ok s6 := by
    rw [step_at_vc 21 rfl (by decide : (4200785, (.ofInt (.ld 1 0 8)), 4) ∈ vmTableChunk 21), exec_ld hs5_0 hrd_pc]
  have hr6 := regsOk_set hr5 (r := 1) (by decide) f.pc
  have hs6_1 : s6.reg 1 = some f.pc := reg_set_self_ok' hr5 (by decide) _
  let s7 := { s6.set 1 (some (f.pc + 1)) with fl := (X86Int.aluF .add f.pc 1).2, rip := 4200793 }
  have h07 : X86Vm.step vmImage s6 = .ok s7 := by
    rw [step_at_vc 21 rfl (by decide : (4200789, (.ofInt (.aluI .add 1 1)), 4) ∈ vmTableChunk 21),
      exec_aluI_add_lt hs6_1 (by omega), Nat.mod_eq_of_lt hpcW]
  have hr7 := regsOk_set hr6 (r := 1) (by decide) (f.pc + 1)
  have hs7_0 : s7.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
    simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs5_0
  have hs7_1 : s7.reg 1 = some (f.pc + 1) := reg_set_self_ok' hr6 (by decide) _
  obtain ⟨_, m', hwr, _, _, _, hrs'_mem, hoth⟩ := wr64_region (m := s.mem) (r := vc.rsReg)
    (a := ea (vc.rsReg.base + rest.length * 32) 8) (v := f.pc + 1) h.1.2 (vc.mem_rs h) hrsRw
    (by rw [ea_add_eq (by omega)]; omega) (by rw [ea_add_eq (by omega)]; simp only [Region.hi]; omega)
  let s8 := { s7 with mem := m', rip := 4200797 }
  have h08 : X86Vm.step vmImage s7 = .ok s8 := by
    rw [step_at_vc 21 rfl (by decide : (4200793, (.ofInt (.st 0 8 1)), 4) ∈ vmTableChunk 21), exec_st (s := s7) hs7_0 hs7_1 hwr]
  have hwr' : wr64 s.mem (ea vc.rsReg.base (rest.length * 32 + 8)) (f.pc + 1) = some m' := by
    rw [ea_add_eq (by omega)] at hwr ⊢
    simpa [Nat.add_assoc] using hwr
  have hrs'_mem' : { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) } ∈ m' := by
    rw [ea_add_eq (by omega), show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] at hrs'_mem
    exact hrs'_mem
  have hsp8 : s8.reg 4 = some vc.sp := by
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp
  have h15_8 : s8.reg 15 = some ctxBase := by
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  refine ⟨s8, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
    (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08))))))),
    rfl, rfl, wf_write_rs_top_pc (s' := s8) h hrs hpcW hwr' hrs'_mem' hoth rfl hr7 hsp8 h15_8, ?_, ?_⟩
  · simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hr12
  · simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13

/-- Callee locals zero-initialization loop (`4200821..4200851`): writes `nLocs`
pairs `(0, tag)` at `locReg[(lb + j) * 16]`, preserving `vc.wf pr ls1` and
returning with all `j < nLocs` initialized. -/
theorem call_init_locs_loop_spec {pr : Sporos.Prog} {ls1 : SporosProvLen.LSt}
    {w nLocs tag : Nat} (htag : tag < X86Env.W) :
    ∀ (rem : Nat) (vc : VmCtx) (s : X86Vm.St),
      rem ≤ nLocs → rsLocBase ls1.rs + nLocs ≤ vc.locCap →
      s.rip = 4200821 → vc.wf pr ls1 s →
      s.reg 8 = some vc.locReg.base → s.reg 9 = some (rsLocBase ls1.rs) →
      s.reg 10 = some tag → s.reg 11 = some 0 → s.reg 3 = some (nLocs - rem) →
      s.reg 12 = some w → s.reg 13 = some nLocs →
      (∀ j, j < nLocs - rem →
        leVal ((vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8) = 0 ∧
        leVal ((vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8) = tag) →
      ∃ (locReg' : Region) (s' : X86Vm.St),
        Reaches vmImage s s' ∧ s'.rip = 4200851 ∧ s'.k = s.k ∧
        locReg'.base = vc.locReg.base ∧
        ({ vc with locReg := locReg' } : VmCtx).wf pr ls1 s' ∧
        s'.reg 9 = some (rsLocBase ls1.rs) ∧ s'.reg 11 = some 0 ∧
        s'.reg 12 = some w ∧ s'.reg 13 = some nLocs ∧
        (∀ j, j < nLocs →
          leVal ((locReg'.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8) = 0 ∧
          leVal ((locReg'.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8) = tag) := by
  intro rem
  induction rem with
  | zero =>
    intro vc s _ _ hrip h hr8 hr9 hr10 hr11 hr3 hr12 hr13 hinit
    rw [Nat.sub_zero] at hr3 hinit
    have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
    let s1 := { s with fl := (X86Int.aluF .sub nLocs nLocs).2, rip := 4200824 }
    have h01 : X86Vm.step vmImage s = .ok s1 := by
      rw [step_at_vc 21 hrip (by decide : (4200821, (.ofInt (.cmp 3 13)), 3) ∈ vmTableChunk 21), exec_cmp hr3 hr13, hrip]
    let s2 := { s1 with rip := 4200851 }
    have h02 : X86Vm.step vmImage s1 = .ok s2 := by
      rw [step_at_vc 21 rfl (by decide : (4200824, (.ofInt (.jcc 3 4200851)), 2) ∈ vmTableChunk 21),
        exec_jcc (cond_aluF_sub_ae nLocs nLocs)]; simp [s1, s2]
    exact ⟨vc.locReg, s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl,
      vc.wf_step_misc h rfl hr hsp h15, hr9, hr11, hr12, hr13, hinit⟩
  | succ rem' ih =>
    intro vc s hrem hcap hrip h hr8 hr9 hr10 hr11 hr3 hr12 hr13 hinit
    let i := nLocs - (rem' + 1)
    have hi_lt : i < nLocs := by omega
    have hi_succ : i + 1 = nLocs - rem' := by omega
    have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
    have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, hrsLe, hlocLe⟩ := (vc.arena_bounds h).2.2
    have ⟨s48, s56, s64, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
    have ⟨s72, s80, s88⟩ := wf_rd_loc_slots h
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hlocHi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
    let s1 := { s with fl := (X86Int.aluF .sub i nLocs).2, rip := 4200824 }
    have h01 : X86Vm.step vmImage s = .ok s1 := by
      rw [step_at_vc 21 hrip (by decide : (4200821, (.ofInt (.cmp 3 13)), 3) ∈ vmTableChunk 21), exec_cmp hr3 hr13, hrip]
    let s2 := { s1 with rip := 4200826 }
    have h02 : X86Vm.step vmImage s1 = .ok s2 := by
      rw [step_at_vc 21 rfl (by decide : (4200824, (.ofInt (.jcc 3 4200851)), 2) ∈ vmTableChunk 21),
        exec_jcc (cond_aluF_sub_ae i nLocs)]; simp [show ¬ nLocs ≤ i by omega, s1, s2]
    let s3 := { s2.set 2 (some (rsLocBase ls1.rs)) with rip := 4200829 }
    have h03 : X86Vm.step vmImage s2 = .ok s3 := by
      rw [step_at_vc 21 rfl (by decide : (4200826, (.ofInt (.mov 2 9)), 3) ∈ vmTableChunk 21), exec_mov (s := s2) hr9]
    have hr3' := regsOk_set hr (r := 2) (by decide) (rsLocBase ls1.rs)
    have hs3_2 : s3.reg 2 = some (rsLocBase ls1.rs) := reg_set_self_ok' hr (by decide) _
    have hs3_3 : s3.reg 3 = some i := by simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 3)]; exact hr3
    let s4 := { s3.set 2 (some (rsLocBase ls1.rs + i)) with fl := (X86Int.aluF .add (rsLocBase ls1.rs) i).2, rip := 4200832 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 21 rfl (by decide : (4200829, (.ofInt (.alu .add 2 3)), 3) ∈ vmTableChunk 21), exec_alu hs3_2 hs3_3]
      dsimp only [s4]; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
    have hr4 := regsOk_set hr3' (r := 2) (by decide) (rsLocBase ls1.rs + i)
    have hs4_2 : s4.reg 2 = some (rsLocBase ls1.rs + i) := reg_set_self_ok' hr3' (by decide) _
    let s5 := { s4.set 2 (some ((rsLocBase ls1.rs + i) * 16)) with fl := X86Int.Flags.undef, rip := 4200836 }
    have h05 : X86Vm.step vmImage s4 = .ok s5 := by
      rw [step_at_vc 21 rfl (by decide : (4200832, (.ofInt (.shlI 2 4)), 4) ∈ vmTableChunk 21),
        exec_shlI_lt hs4_2 (by rw [show 2 ^ (4 % 64) = 16 by decide]; omega)]
    have hr5 := regsOk_set hr4 (r := 2) (by decide) ((rsLocBase ls1.rs + i) * 16)
    have hs5_2 : s5.reg 2 = some ((rsLocBase ls1.rs + i) * 16) := reg_set_self_ok' hr4 (by decide) _
    have hs5_8 : s5.reg 8 = some vc.locReg.base := by
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 8)]; exact hr8
    have hs5_11 : s5.reg 11 = some 0 := by
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 11)]; exact hr11
    have hea0 : eaIdx vc.locReg.base ((rsLocBase ls1.rs + i) * 16) 1 0 = vc.locReg.base + (rsLocBase ls1.rs + i) * 16 := by
      rw [eaIdx_add_eq (by omega)]; omega
    have hea8 : eaIdx vc.locReg.base ((rsLocBase ls1.rs + i) * 16) 1 8 = vc.locReg.base + ((rsLocBase ls1.rs + i) * 16 + 8) := by
      rw [eaIdx_add_eq (by omega)]; omega
    obtain ⟨_, m1, hwr1, _, _, hm1wf, hloc1_m, hoth1⟩ := wrMem_w64_region h.1.2 (vc.mem_loc h) hlocRw
      (a := vc.locReg.base + (rsLocBase ls1.rs + i) * 16) (v := 0) (by omega) (by simp only [Region.hi]; omega)
    let loc1 := { vc.locReg with data := writeBytes vc.locReg.data ((rsLocBase ls1.rs + i) * 16) (bytes8 0) }
    have hloc1_m' : loc1 ∈ m1 := by simpa [loc1, Nat.add_sub_cancel_left] using hloc1_m
    let s6 := { s5 with mem := m1, rip := 4200840 }
    have h06 : X86Vm.step vmImage s5 = .ok s6 := by
      rw [step_at_vc 21 rfl (by decide : (4200836, (.stIdx .w64 8 2 1 0 11), 4) ∈ vmTableChunk 21),
        exec_stIdx hs5_8 hs5_2 hs5_11 (by rw [hea0]; exact hwr1)]
    have hs6_10 : s6.reg 10 = some tag := by
      simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 10)]; exact hr10
    obtain ⟨_, m2, hwr2, _, _, hm2wf, hloc2_m, hoth2⟩ := wrMem_w64_region hm1wf hloc1_m' hlocRw
      (a := vc.locReg.base + ((rsLocBase ls1.rs + i) * 16 + 8)) (v := tag)
      (by dsimp [loc1]; omega) (by simp only [Region.hi, loc1, writeBytes_length]; omega)
    let loc2 := { vc.locReg with data := writeBytes loc1.data ((rsLocBase ls1.rs + i) * 16 + 8) (bytes8 tag) }
    have hloc2_m' : loc2 ∈ m2 := by simpa [loc2, loc1, Nat.add_sub_cancel_left] using hloc2_m
    let s7 := { s6 with mem := m2, rip := 4200845 }
    have h07 : X86Vm.step vmImage s6 = .ok s7 := by
      rw [step_at_vc 21 rfl (by decide : (4200840, (.stIdx .w64 8 2 1 8 10), 5) ∈ vmTableChunk 21),
        exec_stIdx (s := s6) hs5_8 hs5_2 hs6_10 (by rw [hea8]; exact hwr2)]
    have hs7_3 : s7.reg 3 = some i := by
      simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 3)]; exact hr3
    let s8 := { s7.set 3 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4200849 }
    have h08 : X86Vm.step vmImage s7 = .ok s8 := by
      rw [step_at_vc 21 rfl (by decide : (4200845, (.ofInt (.aluI .add 3 1)), 4) ∈ vmTableChunk 21),
        exec_aluI_add_lt hs7_3 (by omega), Nat.mod_eq_of_lt (by omega)]
    have hr8' := regsOk_set hr5 (r := 3) (by decide) (i + 1)
    let s9 := { s8 with rip := 4200821 }
    have h09 : X86Vm.step vmImage s8 = .ok s9 := by
      rw [step_at_vc 21 rfl (by decide : (4200849, (.ofInt (.jmp 4200821)), 2) ∈ vmTableChunk 21), exec_jmp]
    have hreach9 : Reaches vmImage s s9 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))))))
    have hoth_all : ∀ u ∈ s.mem, u.base ≠ vc.locReg.base → u ∈ m2 := fun u hu hne =>
      hoth2 u (hoth1 u hu hne) (by simpa [loc1] using hne)
    have hstk_m : vc.stk ∈ m2 := hoth_all _ (vc.mem_stk h) (vc.ne_bases h (i := 1) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    have hctx_m : vc.ctxReg ∈ m2 := hoth_all _ (vc.mem_ctx h) (vc.ne_bases h (i := 2) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    have hrs_m : vc.rsReg ∈ m2 := hoth_all _ (vc.mem_rs h) (vc.ne_bases h (i := 8) (j := 9) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    have hsp9 : s9.reg 4 = some vc.sp := by
      simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 3 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4)]; exact hsp
    have h15_9 : s9.reg 15 = some ctxBase := by
      simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
    have hrel2 : RsLocRel vc.rsReg.data loc2.data (rsLocBase ls1.rs) ls1.rs :=
      rsLocRel_write_loc_after (rsLocRel_write_loc_after hrsRel (by omega)) (by omega)
    have hwf9 : ({ vc with locReg := loc2 } : VmCtx).wf pr ls1 s9 :=
      wf_update_rs_loc h hm2wf (bytesLt_wr64 (bytesLt_wr64 h.2.1 hwr1) hwr2) (len_pos_wrN (len_pos_wrN h.2.2.1 hwr1) hwr2)
        hr8' hsp9 h15_9 hstk_m rfl rfl rfl hctx_m rfl rfl rfl
        (ctx_leVal_of_rd64 h (by decide) s48) (ctx_leVal_of_rd64 h (by decide) s56) (ctx_leVal_of_rd64 h (by decide) s64)
        (ctx_leVal_of_rd64 h (by decide) s72) (ctx_leVal_of_rd64 h (by decide) s80) (ctx_leVal_of_rd64 h (by decide) s88)
        (fun _ _ _ => rfl) hrs_m hrsRw hrsPos hrsLenEq hrsPg hrsHi
        hloc2_m' hlocRw hlocPos (by dsimp [loc2, loc1]; rw [writeBytes_length, writeBytes_length]; exact hlocLenEq) hlocPg
        (by simpa [Region.hi, loc2, loc1, writeBytes_length] using hlocHi)
        hrel2 hrsLe hlocLe (vc.code_flat h).2.2.2.2.2.2.1 h.2.2.2.2.2.2.1 (fun u hu _ _ _ h4 => hoth_all u hu h4)
    have hinit' : ∀ j, j < nLocs - rem' →
        leVal ((loc2.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8) = 0 ∧
        leVal ((loc2.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8) = tag := by
      intro j hj
      by_cases hji : j < i
      · have hprev := hinit j hji
        have hd0 : ((loc2.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8) = (vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16)).take 8 := by
          dsimp [loc2, loc1]; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
        have hd8 : ((loc2.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8) = (vc.locReg.data.drop ((rsLocBase ls1.rs + j) * 16 + 8)).take 8 := by
          dsimp [loc2, loc1]; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
        rw [hd0, hd8]; exact hprev
      · have hjeq : j = i := by omega
        subst hjeq; dsimp [loc2, loc1]
        constructor
        · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by decide)]
        · rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt htag]
    have hreg9 : s9.reg 8 = some loc2.base ∧ s9.reg 9 = some (rsLocBase ls1.rs) ∧ s9.reg 10 = some tag ∧
        s9.reg 11 = some 0 ∧ s9.reg 3 = some (nLocs - rem') ∧ s9.reg 12 = some w ∧ s9.reg 13 = some nLocs := by
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 8), List.getElem?_set_ne (by decide : 2 ≠ 8)]; exact hr8
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 9), List.getElem?_set_ne (by decide : 2 ≠ 9)]; exact hr9
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 10), List.getElem?_set_ne (by decide : 2 ≠ 10)]; exact hr10
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 11), List.getElem?_set_ne (by decide : 2 ≠ 11)]; exact hr11
      · rw [← hi_succ]; exact reg_set_self_ok' hr5 (by decide) _
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12)]; exact hr12
      · simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 13), List.getElem?_set_ne (by decide : 2 ≠ 13)]; exact hr13
    obtain ⟨locReg', s', hrt, hrip', hk', hlb', hwf', h9', h11', h12', h13', hall'⟩ :=
      ih { vc with locReg := loc2 } s9 (by omega) hcap rfl hwf9 hreg9.1 hreg9.2.1 hreg9.2.2.1
        hreg9.2.2.2.1 hreg9.2.2.2.2.1 hreg9.2.2.2.2.2.1 hreg9.2.2.2.2.2.2 hinit'
    exact ⟨locReg', s', Reaches.trans hreach9 hrt, hrip', hk', hlb', hwf', h9', h11', h12', h13', hall'⟩

end SporosVmOpsCtrl
