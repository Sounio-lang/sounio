import Bootstrap.SporosVmOpsData1

/-!
# Native x86-64 VM Aux-Stack Helpers & `O_lit`, `O_str` (`SporosVmOpsData2`)

Auxiliary-stack `VmCtx` update helpers (`wf_update_aux_push`, `step_call_aux_push_vc`,
`wf_dec_auxLen`) plus simulation proofs over `X86Vm` for `O_lit` and `O_str`.

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

/-! ### `VmCtx` updates for `fn_aux_push`, `aux_len` decrement, and `ctx`-slot push -/

theorem wf_update_aux_push {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {auxReg' ctx' stk' : Region} {auxCap' : Nat} {v : SporosProvLen.LVal}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (haux_mem : auxReg' ∈ s'.mem) (haux_rw : auxReg'.rw = true) (haux_hi : auxReg'.hi ≤ 2 ^ 47)
    (haux_cap_ge : 4 ≤ auxCap') (haux_len_eq : auxReg'.data.length = auxCap' * 16)
    (haux_pg : (auxCap' * 16) % X86Env.page = 0)
    (hctx_mem : ctx' ∈ s'.mem) (hctx_base : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_len : ctx'.data.length = vc.ctxReg.data.length)
    (hc24 : leVal ((ctx'.data.drop 24).take 8) = auxReg'.base)
    (hc32 : leVal ((ctx'.data.drop 32).take 8) = (v :: ls.aux).length)
    (hc40 : leVal ((ctx'.data.drop 40).take 8) = auxCap')
    (hctx_oth : ∀ j, (j < 24 ∨ 48 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]?)
    (hstk_mem : stk' ∈ s'.mem) (hstk_base : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_len : stk'.data.length = vc.stk.data.length)
    (haux_ne : ∀ u ∈ s.mem, u.base ≠ vc.auxReg.base → u.base ≠ auxReg'.base)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.auxReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s'.mem)
    (haux_rel : PairListRel auxReg'.data (v :: ls.aux).length (v :: ls.aux))
    (haux_le : (v :: ls.aux).length ≤ auxCap')
    (hvW : v.w < X86Env.W ∧ v.plen < X86Env.W) :
    let vc' := { vc with stk := stk', ctxReg := ctx', auxReg := auxReg', auxCap := auxCap' }
    vc'.wf pr { ls with aux := v :: ls.aux } s' := by
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1
    (vc.ne_bases h (i := 0) (j := 7) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    (vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    (vc.ne_bases h (i := 0) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)), hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, _, _, _, _, _,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, _, _, _, hsl6, hsl7, hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, _, _, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', auxReg := auxReg', auxCap := auxCap' }
  have hmem_all : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem | exact hctx_mem | exact haux_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    let l1 : List Region := [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3, vc.dsReg]
    let l2 : List Region := [vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg,
      vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]
    have hnd_old : (l1.map Region.base ++ vc.auxReg.base :: l2.map Region.base).Nodup := by
      simpa [l1, l2, VmCtx.regions] using hnodup
    have hgoal : (l1.map Region.base ++ auxReg'.base :: l2.map Region.base).Nodup := by
      refine nodup_replace_at hnd_old ?_
      intro x hx
      rw [← List.map_append, List.mem_map] at hx
      rcases hx with ⟨u, hu, rfl⟩
      have hu_in : u ∈ vc.regions := by
        simp only [VmCtx.regions]
        simp only [l1, l2, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hu
        rcases hu with (rfl | rfl | rfl | rfl | rfl | rfl | rfl) | (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;> simp
      have hu_ne : u.base ≠ vc.auxReg.base := by
        rw [List.nodup_append, List.nodup_cons] at hnd_old
        rcases List.mem_append.mp hu with h1 | h2
        · exact hnd_old.2.2 u.base (List.mem_map_of_mem h1) vc.auxReg.base (List.mem_cons_self ..)
        · intro heq; exact hnd_old.2.1.1 (heq ▸ List.mem_map_of_mem h2)
      exact haux_ne u (hmem u hu_in) hu_ne
    simpa [vc', l1, l2, VmCtx.regions, hstk_base, hctx_base] using hgoal
  have hslot : ∀ m0, (m0 + 8 ≤ 24 ∨ 48 ≤ m0) →
      leVal ((ctx'.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
    intro m0 hm0; congr 1
    exact drop_take_eq_of_getElem?_eq fun j hj1 hj2 => hctx_oth j (by omega)
  have hwlt_aux' : ∀ x ∈ v :: ls.aux, x.w < Sporos.M ∧ x.plen < Sporos.M := by
    intro x hx; rcases List.mem_cons.mp hx with rfl | hx
    · exact hvW
    · exact hwlt.2.1 x hx
  exact ⟨hcm', hbl', hpos', hr', hsp', h15', hnodup', hmem_all,
    by rw [hstk_rw]; exact hstk1, by rw [hstk_base]; exact hstk2,
    by simpa only [vc', Region.hi, hstk_base, hstk_len] using hstk3, hstk4,
    by rw [hctx_base]; exact hctx1, by rw [hctx_rw]; exact hctx2,
    by rw [hctx_len]; exact hctx3, by simpa only [vc', Region.hi, hctx_base, hctx_len] using hctx4,
    hf1, hf2, hf3, hds1, hds2, hds3, hds4, hds5, haux_rw, haux_cap_ge, haux_len_eq, haux_pg, haux_hi,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5, hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6, hinp1, hout1, hout2, hout3, hout4, hout5,
    by rw [hslot _ (Or.inl (by decide))]; exact hsl0, by rw [hslot _ (Or.inl (by decide))]; exact hsl1,
    by rw [hslot _ (Or.inl (by decide))]; exact hsl2, hc24, hc32, hc40,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl6, by rw [hslot _ (Or.inr (by decide))]; exact hsl7,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl8, by rw [hslot _ (Or.inr (by decide))]; exact hsl9,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl10, by rw [hslot _ (Or.inr (by decide))]; exact hsl11,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl12, by rw [hslot _ (Or.inr (by decide))]; exact hsl13,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl14, by rw [hslot _ (Or.inr (by decide))]; exact hsl15,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl16, by rw [hslot _ (Or.inr (by decide))]; exact hsl17,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl18, by rw [hslot _ (Or.inr (by decide))]; exact hsl19,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl20, by rw [hslot _ (Or.inr (by decide))]; exact hsl21,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl22, by rw [hslot _ (Or.inr (by decide))]; exact hsl23,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl24, by rw [hslot _ (Or.inr (by decide))]; exact hsl25,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl26, by rw [hslot _ (Or.inr (by decide))]; exact hsl27,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl28, by rw [hslot _ (Or.inr (by decide))]; exact hsl29,
    by rw [hslot _ (Or.inr (by decide))]; exact hsl30, by rw [hslot _ (Or.inr (by decide))]; exact hsl31,
    hpds, hdsle, haux_rel, haux_le, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
    ⟨hwlt.1, hwlt_aux', hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩

theorem step_call_aux_push_vc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w p : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.call SporosVmBin.fn_aux_push)) 5 s)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hretW : s.rip + 5 < X86Env.W)
    (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hwW : w < X86Env.W) (hpW : p < X86Env.W) :
    (∃ (vc' : VmCtx) (s' : X86Vm.St),
      Reaches vmImage s s' ∧ s'.rip = s.rip + 5 ∧ s'.k.out = s.k.out ∧
      OracleAdmits s' ∧ vc'.wf pr { ls with aux := ⟨w, p⟩ :: ls.aux } s') ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  obtain ⟨s1, h01, hrip1, hk1, _, hcm1, hbl1, hpos1, hr1, hs1_4, hs1_15, hs1_oth,
      hstk1, hstk1_b, hstk1_rw, hstk1_len, h16, hsp1W, hstk1_lo, hstk1_hi, hrd_ret, hoth1⟩ :=
    step_call_stk hstep h hretW
  have hrch1 : Reaches vmImage s s1 := Reaches.step h01
  have hctx_b := vc.ctx_bounds h
  have ⟨_, ⟨haux_rw, haux_cap4, haux_len, haux_pg, haux_hi, haux_rel, haux_le⟩, _⟩ := vc.arena_bounds h
  have ⟨_, _, _, hc24, hc32, hc40, _⟩ := wf_aux_loc_slots h
  have hne_aux_ctx : vc.auxReg.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 7) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_aux : vc.stk.base ≠ vc.auxReg.base :=
    vc.ne_bases h (i := 1) (j := 7) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ctx : vc.stk.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcall := call_aux_push_oracle_spec (s := s1) (sp := vc.sp - 8) (retAddr := s.rip + 5)
    (ctxBase := ctxBase) (auxBase := vc.auxReg.base) (auxLen := ls.aux.length) (auxCap := vc.auxCap)
    (w := w) (p := p) (ctx := vc.ctxReg) (auxReg := vc.auxReg)
    (stk := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (s.rip + 5)) })
    hrip1 hr1 (oracleAdmits_of_reaches hor hrch1) hbl1 hpos1 hs1_4 hs1_15
    (by rw [hs1_oth 7 (by decide)]; exact hr7) (by rw [hs1_oth 6 (by decide)]; exact hr6)
    (hoth1 vc.ctxReg (vc.mem_ctx h) (by rw [hctx_b.1]; exact Ne.symm hne_stk_ctx))
    hctx_b.1 hctx_b.2.1 hctx_b.2.2.1
    (hoth1 vc.auxReg (vc.mem_aux h) (Ne.symm hne_stk_aux)) rfl haux_rw haux_hi hne_aux_ctx
    hstk1 hstk1_rw (by rw [hstk1_b]; exact hne_stk_aux) (by rw [hstk1_b]; exact hne_stk_ctx)
    h16 hsp1W hstk1_lo hstk1_hi haux_le (by omega) haux_len haux_pg hctx_b.2.2.2.2.1
    hcm1.2 hrd_ret hc24 hc32 hc40
  rcases hcall with ⟨auxReg', ctx', stk', m2, s2, auxCap', hrch2, hrip2, hm2, hmwf2, hbl2, hpos2,
      hr2, hor2, hk2, hsp2, hcap_le, hlen_le, haux_pg', haux2_mem, haux2_rw, haux2_hi, haux2_data,
      hctx2_mem, hctx2_b, hctx2_rw, hctx2_len, hc24', hc32', hc40', hctx2_oth,
      hstk2_mem, hstk2_b, hstk2_rw, hstk2_len, haux2_ne, hoth2, hreg2_oth⟩ | ⟨hhit, hends⟩
  · have hsp2' : s2.reg 4 = some vc.sp := by
      have hsb := vc.sp_bounds h
      simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hsp2
    have hs2_15 : s2.reg 15 = some ctxBase := by
      rw [hreg2_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hs1_15
    have haux_len_eq' : auxReg'.data.length = auxCap' * 16 := by
      rw [haux2_data, writeBytes_length, writeBytes_length, List.length_append, List.length_replicate, haux_len]
      omega
    have haux_rel' : PairListRel auxReg'.data ((⟨w, p⟩ : SporosProvLen.LVal) :: ls.aux).length (⟨w, p⟩ :: ls.aux) := by
      rw [haux2_data]
      simpa using pairListRel_push (pairListRel_extend_zero (ext := (auxCap' - vc.auxCap) * 16) haux_rel (by rw [haux_len]; omega))
        ⟨w, p⟩ hwW hpW (by rw [List.length_append, List.length_replicate, haux_len]; omega)
    have haux_ne_s : ∀ u ∈ s.mem, u.base ≠ vc.auxReg.base → u.base ≠ auxReg'.base := by
      intro u hu hne
      by_cases hstku : u.base = vc.stk.base
      · rw [hstku, ← hstk1_b]; exact haux2_ne _ hstk1 (by rw [hstk1_b]; exact hne_stk_aux)
      · exact haux2_ne u (hoth1 u hu hstku) hne
    have hoth_s : ∀ u ∈ s.mem, u.base ≠ vc.auxReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s2.mem := by
      intro u hu h1 h2 h3; rw [hm2]
      exact hoth2 u (hoth1 u hu h3) h1 (by rw [← hctx_b.1]; exact h2) (by rw [hstk1_b]; exact h3)
    have hwf2 := wf_update_aux_push h (hm2 ▸ hmwf2) (hm2 ▸ hbl2) (hm2 ▸ hpos2)
      hr2 hsp2' hs2_15 (hm2 ▸ haux2_mem) haux2_rw haux2_hi (by omega) haux_len_eq' haux_pg'
      (hm2 ▸ hctx2_mem) hctx2_b hctx2_rw hctx2_len hc24' (by simpa using hc32') hc40' hctx2_oth
      (hm2 ▸ hstk2_mem) (by rw [hstk2_b, hstk1_b]) hstk2_rw (by rw [hstk2_len, hstk1_len])
      haux_ne_s hoth_s haux_rel' (by simpa using hlen_le) ⟨hwW, hpW⟩
    exact Or.inl ⟨_, s2, Reaches.trans hrch1 hrch2, hrip2, hk2.trans (by rw [hk1]), hor2, hwf2⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch1 hhit, Ends.of_reaches hrch1 hends⟩

theorem wf_dec_auxLen {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {len r : Nat} {v : SporosProvLen.LVal} {restAux : List SporosProvLen.LVal}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 32 r)) len s)
    (h : vc.wf pr ls s) (haux : ls.aux = v :: restAux) (hr : s.reg r = some restAux.length) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 32 (bytes8 restAux.length) }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧ s'.rip = s.rip + len ∧ s'.k = s.k ∧
      s'.regs = s.regs ∧ s'.fl = s.fl ∧
      { vc with ctxReg := ctx' }.wf pr { ls with aux := restAux } s' := by
  have hb := vc.ctx_bounds h
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hea : ea ctxBase 32 = ctxBase + 32 := ea_add_eq (by have : ctxSize = 4096 := rfl; have : X86Env.W = 2 ^ 64 := rfl; omega)
  obtain ⟨_, m', hwr, _, _, _, hmem', hoth⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_ctx h) hb.2.1 (a := ctxBase + 32)
      (by rw [hb.1]; omega) (by simp only [Region.hi, hb.1]; have : ctxSize = 4096 := rfl; omega)
      (v := restAux.length)
  have hsub : ctxBase + 32 - vc.ctxReg.base = 32 := by rw [hb.1]; omega
  rw [hsub] at hmem'
  rw [← hea] at hwr
  have hcm' := h.1.wr64_preserve hwr
  have hbl' := bytesLt_wr64 h.2.1 hwr
  have hpos' := len_pos_wrN h.2.2.1 hwr
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 32 (bytes8 restAux.length) }
  let vc' := { vc with ctxReg := ctx' }
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by rw [hstep, exec_st h15 hr hwr]
  obtain ⟨_, _, _, hregs, hsp, h15s, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3, htb1, htb2, htb3, htb4, htb5,
    hkv1, hkv2, hkv3, hkv4, hkv5, hkv6, hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, _, hsl5, hsl6, hsl7, hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hmem_all : ∀ u ∈ vc'.regions, u ∈ s'.mem := by
    intro u hu
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    rcases hu with rfl | rfl | huc | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | (rw [huc]; exact hmem')
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrest_ltW : restAux.length < X86Env.W := by
    simp only [Region.hi, haux3] at haux5; rw [haux] at hauxle; simp at hauxle; omega
  have hslot : ∀ j, (j + 8 ≤ 32 ∨ 40 ≤ j) → j + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop j).take 8) = leVal ((vc.ctxReg.data.drop j).take 8) := by
    intro j hj hlim; dsimp [ctx']
    rw [writeBytes_drop_take_disjoint vc.ctxReg.data 32 j 8 (bytes8 restAux.length)
      (by rw [bytes8_length]; exact hj) (by have : ctxSize = 4096 := rfl; omega)]
  have hsl4' : leVal ((ctx'.data.drop 32).take 8) = restAux.length := by
    dsimp [ctx']
    rw [leVal_writeBytes_same vc.ctxReg.data 32 restAux.length (by have : ctxSize = 4096 := rfl; omega),
      Nat.mod_eq_of_lt hrest_ltW]
  have hpaux' : PairListRel vc.auxReg.data restAux.length restAux := by
    have hd := pairListRel_drop hpaux (k := 1) (by simp [haux]); simpa [haux] using hd
  have hauxle' : restAux.length ≤ vc.auxCap := by rw [haux] at hauxle; simp at hauxle; omega
  have hwf' : vc'.wf pr { ls with aux := restAux } s' :=
    ⟨hcm', hbl', hpos', hregs, hsp, h15s, by simpa [vc', ctx', VmCtx.regions] using hnodup, hmem_all,
      hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, by dsimp [vc', ctx']; rw [writeBytes_length]; exact hctx3,
      by simpa only [vc', Region.hi, ctx', writeBytes_length] using hctx4, hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3, htb1, htb2, htb3, htb4, htb5,
      hkv1, hkv2, hkv3, hkv4, hkv5, hkv6, hinp1, hout1, hout2, hout3, hout4, hout5,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl0,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl1,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl2,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl3, hsl4',
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl5,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl6,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl7,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl8,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl9,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl10,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl11,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl12,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl13,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl14,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl15,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl16,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl17,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl18,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl19,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl20,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl21,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl22,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl23,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl24,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl25,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl26,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl27,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl31,
      hpds, hdsle, hpaux', hauxle', hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
      ⟨hwlt.1, fun x hx => hwlt.2.1 x (by rw [haux]; exact List.mem_cons_of_mem _ hx),
       hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩
  exact ⟨s', hs, rfl, rfl, rfl, rfl, hwf'⟩

private theorem step_push_ctx_val_vc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk off v : Nat}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hoff : rd64 s.mem (ea ctxBase off) = some v) (hvW : v < X86Env.W)
    (haW : a + 23 < X86Env.W)
    (hm1 : (a, (.ofInt (.ld 7 15 off)), 7) ∈ vmTableChunk chk)
    (hm2 : (a + 7, (.ofInt (.ld 6 15 224)), 7) ∈ vmTableChunk chk)
    (hm3 : (a + 14, (.ofInt (.aluI .add 6 1)), 4) ∈ vmTableChunk chk)
    (hm4 : (a + 18, (.ofInt (.call SporosVmBin.fn_ds_push)), 5) ∈ vmTableChunk chk) :
    (∃ (vc' : VmCtx) (s' : X86Vm.St),
      Reaches vmImage s s' ∧ s'.rip = a + 23 ∧ s'.k.out = s.k.out ∧
      OracleAdmits s' ∧ vc'.wf pr { ls with ds := ⟨v, (ls.pc + 1) % Sporos.M⟩ :: ls.ds } s' ∧
      (∀ m0 u, 24 ≤ m0 → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 7 (some v) with rip := a + 7 } := by
    rw [step_at_vc chk hrip hm1, exec_ld h15 hoff, hrip]
  let s1 := { s.set 7 (some v) with rip := a + 7 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hpc1 : rd64 s1.mem (ea ctxBase 224) = some ls.pc := rd_ctx_pc (s := s) h
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 6 (some ls.pc) with rip := a + 14 } := by
    rw [step_at_vc chk rfl hm2, exec_ld hs1_15 hpc1]
  let s2 := { s1.set 6 (some ls.pc) with rip := a + 14 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_6 : s2.reg 6 = some ls.pc := reg_set_self_ok' hr1 (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 6 (some ((ls.pc + 1) % Sporos.M)) with
            fl := (X86Int.aluF .add ls.pc 1).2, rip := a + 18 } := by
    rw [step_at_vc chk rfl hm3, exec_aluI_add_lt hs2_6 (by have : X86Env.W = 2 ^ 64 := rfl; omega)]
    rfl
  let s3 := { s2.set 6 (some ((ls.pc + 1) % Sporos.M)) with
              fl := (X86Int.aluF .add ls.pc 1).2, rip := a + 18 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_7 : s3.reg 7 = some v := by
    change (s2.set 6 _).reg 7 = some v; rw [reg_set_ne (by decide)]
    change (s1.set 6 _).reg 7 = some v; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok' hr (by decide) _
  have hs3_6 : s3.reg 6 = some ((ls.pc + 1) % Sporos.M) := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3
    (by change (s2.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s1.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4)
    (by change (s2.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s1.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15)
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hpW : (ls.pc + 1) % Sporos.M < X86Env.W := by rw [M_eq_W']; exact lt_W_of_mod _
  rcases step_call_ds_push_vc (step_at_vc chk rfl hm4) hwf3 (oracleAdmits_of_reaches hor hrch3)
      (by dsimp [s3]; omega) hs3_7 hs3_6 hvW hpW with
    ⟨vc4, s4, hrch4, hrip4, hk4, hor4, hwf4, hpres4⟩ | ⟨hhit, hends⟩
  · exact Or.inl ⟨vc4, s4, Reaches.trans hrch3 hrch4, by simpa [s3] using hrip4, hk4, hor4, hwf4, hpres4⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch3 hhit, Ends.of_reaches hrch3 hends⟩

/-! ### `O_lit` and `O_str` simulation specs -/

theorem O_lit_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {n : Nat}
    (hrip : s.rip = instrEntry (.lit n)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .lit n)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.lit n)).2) :
    OpPostOutcome pr ls s := by
  have hnW : n < X86Env.W := by
    have hf := flatProgFit_ins_at (vc.code_flat h).2.2.2.1 hnw hpc; rw [hins] at hf; exact hf
  have h264' : rd64 s.mem (ea ctxBase 264) = some n := by
    simpa [instrPair, Nat.mod_eq_of_lt hnW] using h264
  have hfetch := fetch_eq_of_lt hnw hpc hins
  unfold OpPostOutcome
  have hlstep : SporosProvLen.lstep pr ls =
      .next { ls with ds := ⟨n, (ls.pc + 1) % Sporos.M⟩ :: ls.ds, rs := { f with pc := f.pc + 1 } :: rest } := by
    simp [SporosProvLen.lstep, hrs, hfetch, SporosProvLen.LSt.setTop, Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt hnW]
  rw [hlstep]
  have hW : X86Env.W = 2 ^ 64 := rfl
  rcases step_push_ctx_val_vc (a := 4199139) (chk := 15) (off := 264) hrip h hor h264' hnW (by omega)
      (by decide) (by decide) (by decide) (by decide) with
    ⟨vc1, s1, hrch1, hrip1, hk1, hor1, hwf1, _⟩ | ⟨hhit, hends⟩
  · rcases adv_pc_finish_vc (a := 4199162) (chk := 15) hrip1 hwf1 hor1 hrs hpcW
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) with
      ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch1 hrchf, hkf.trans hk1, horf, hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch1 hhit, Ends.of_reaches hrch1 hends⟩
  · exact Or.inr ⟨hhit, hends⟩

theorem O_str_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {i : Nat}
    (hrip : s.rip = instrEntry (.str i)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .str i)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.str i)).2) :
    OpPostOutcome pr ls s := by
  have ⟨hf1, hf2, hf3, hfit, _⟩ := vc.code_flat h
  have hiW : i < X86Env.W := by
    have hf := flatProgFit_ins_at hfit hnw hpc; rw [hins] at hf; exact hf
  have h264' : rd64 s.mem (ea ctxBase 264) = some i := by
    simpa [instrPair, Nat.mod_eq_of_lt hiW] using h264
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have ⟨_, _, _, _, hc1, hc2, hc3⟩ := vc.rd_ctx_rs_code_slots h
  have hnd1 := rd64_flat_nd h.1.2 (vc.mem_code1 h) hf1 hfit
  have hnd2 := rd64_flat_nd h.1.2 (vc.mem_code2 h) hf2 hfit
  have hnd3 := rd64_flat_nd h.1.2 (vc.mem_code3 h) hf3 hfit
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_0, _, _, hs1_oth⟩ :=
    tmr_ld_vote_eq_spec (a := 4199198) (done := 4199254) (disp := 16) hrip h.2.2.2.1 h.2.2.2.2.2.1
      (mem_vmTable_of_vc 15 (by decide)) (mem_vmTable_of_vc 15 (by decide))
      (mem_vmTable_of_vc 15 (by decide)) (mem_vmTable_of_vc 15 (by decide))
      (mem_vmTable_of_vc 15 (by decide)) (mem_vmTable_of_vc 15 (by decide))
      (mem_vmTable_of_vc 15 (by decide)) (mem_vmTable_of_vc 15 (by decide))
      hc1 hc2 hc3 hnd1 hnd2 hnd3 hfit.2.2.1
  have hs1_15 : s1.reg 15 = some ctxBase := by
    rw [hs1_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.2.1
  have hs1_4 : s1.reg 4 = some vc.sp := by
    rw [hs1_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.1
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 8 (some i) with rip := 4199261 } := by
    rw [step_at_vc 15 hrip1 (by decide : (4199254, (.ofInt (.ld 8 15 264)), 7) ∈ vmTableChunk 15),
      exec_ld hs1_15 (by rw [hmem1]; exact h264'), hrip1]
  let s2 := { s1.set 8 (some i) with rip := 4199261 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_8 : s2.reg 8 = some i := reg_set_self_ok' hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some pr.data.length := by
    change (s1.set 8 _).reg 0 = some pr.data.length; rw [reg_set_ne (by decide)]; exact hs1_0
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub i pr.data.length).2, rip := 4199264 } := by
    rw [step_at_vc 15 rfl (by decide : (4199261, (.ofInt (.cmp 8 0)), 3) ∈ vmTableChunk 15),
      exec_cmp hs2_8 hs2_0]
  let s3 := { s2 with fl := (X86Int.aluF .sub i pr.data.length).2, rip := 4199264 }
  have hrch3 : Reaches vmImage s s3 := Reaches.trans hrch1 (Reaches.step_trans h02 (Reaches.step h03))
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s1.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s1.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  unfold OpPostOutcome
  by_cases hi : i < pr.data.length
  · have hget : pr.data[i]? = some pr.data[i]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hi]; rfl
    let addr := ((pr.data.take i).map List.length).foldl (· + ·) 0
    let dlen := pr.data[i]!.length
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with ds := ⟨dlen, (ls.pc + 1) % Sporos.M⟩ :: ⟨addr, (ls.pc + 1) % Sporos.M⟩ :: ls.ds,
                        rs := { f with pc := f.pc + 1 } :: rest } := by
      unfold SporosProvLen.lstep; rw [hrs]; dsimp only; rw [hfetch]; dsimp only; rw [hget]
      simp [SporosProvLen.LSt.setTop, addr, dlen]
    rw [hlstep]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4199270 } := by
      rw [step_at_vc 15 rfl (by decide : (4199264, (.ofInt (.jcc 3 4201150)), 6) ∈ vmTableChunk 15),
        exec_jcc (cond_aluF_sub_ae i pr.data.length)]
      dsimp [s3]; rw [decide_eq_false (by omega)]; rfl
    let s4 := { s3 with rip := 4199270 }
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hfq : 4 + 2 * pr.data.length ≤ (flatProgQwords pr).length := by
      simp [flatProgQwords, flatProgHeader, dataMetaQwords_len]; omega
    have hsz : (flatProgQwords pr).length * 8 < 2 ^ 47 := hfit.2.2.2.1
    have hs4_8 : s4.reg 8 = some i := hs2_8
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 8 (some (i * 16)) with fl := Flags.undef, rip := 4199274 } := by
      rw [step_at_vc 15 rfl (by decide : (4199270, (.ofInt (.shlI 8 4)), 4) ∈ vmTableChunk 15),
        exec_shlI_4_lt hs4_8 (by omega)]
    let s5 := { s4.set 8 (some (i * 16)) with fl := Flags.undef, rip := 4199274 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_8 : s5.reg 8 = some (i * 16) := reg_set_self_ok' hr2 (by decide) _
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5.set 8 (some (i * 16 + 32)) with fl := (X86Int.aluF .add (i * 16) 32).2, rip := 4199278 } := by
      rw [step_at_vc 15 rfl (by decide : (4199274, (.ofInt (.aluI .add 8 32)), 4) ∈ vmTableChunk 15),
        exec_aluI_add_lt hs5_8 (by omega), Nat.mod_eq_of_lt (by omega)]
    let s6 := { s5.set 8 (some (i * 16 + 32)) with fl := (X86Int.aluF .add (i * 16) 32).2, rip := 4199278 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_8 : s6.reg 8 = some (i * 16 + 32) := reg_set_self_ok' hr5 (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s4.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change (s5.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s4.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs3_4
    have hflat_bound := data_addr_len_le_flatten pr.data i hi
    have haddrW : addr < X86Env.W := by have := hfit.2.2.2.2.1; omega
    have hdlenW : dlen < X86Env.W := by have := hfit.2.2.2.2.1; omega
    have hd1 := rd64_flat_data h.1.2 (vc.mem_code1 h) hf1 hfit hi
    have hd2 := rd64_flat_data h.1.2 (vc.mem_code2 h) hf2 hfit hi
    have hd3 := rd64_flat_data h.1.2 (vc.mem_code3 h) hf3 hfit hi
    have hmem6 : s6.mem = s.mem := hmem1
    obtain ⟨s7, hrch7, hrip7, hmem7, hk7, hr7, hs7_0, _, _, hs7_oth⟩ :=
      tmr_ldIdx8_vote_eq_spec (a := 4199278) (done := 4199334) (disp := 0) (ldLen := 4) (chk := 15)
        rfl hr6 hs6_15 hs6_8 (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)
        (by rw [hmem6]; exact hc1) (by rw [hmem6]; exact hc2) (by rw [hmem6]; exact hc3)
        (by rw [hmem6]; exact hd1.1) (by rw [hmem6]; exact hd2.1) (by rw [hmem6]; exact hd3.1) haddrW
    have hwf7 : vc.wf pr ls s7 := vc.wf_step_misc h (hmem7.trans hmem1) hr7
      (by rw [hs7_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hs6_4)
      (by rw [hs7_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hs6_15)
    obtain ⟨s8, h08, hrip8, hk8, hregs8, _, hwf8, h616_8, _⟩ :=
      step_wr_ctx_scratch (step_at_vc 16 hrip7 (by decide : (4199334, (.ofInt (.st 15 616 0)), 7) ∈ vmTableChunk 16))
        hwf7 hs7_0 (by decide) (by decide)
    let vc8 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 616 (bytes8 addr) } }
    have hs8_8 : s8.reg 8 = some (i * 16 + 32) := by
      have h78 : s7.reg 8 = some (i * 16 + 32) := by
        rw [hs7_oth 8 (by decide) (by decide) (by decide) (by decide)]; exact hs6_8
      change s8.regs[8]?.getD none = some (i * 16 + 32); rw [hregs8]; exact h78
    have ⟨_, _, _, _, hc1_8, hc2_8, hc3_8⟩ := vc8.rd_ctx_rs_code_slots hwf8
    have hd1_8 := rd64_flat_data hwf8.1.2 (vc8.mem_code1 hwf8) hf1 hfit hi
    have hd2_8 := rd64_flat_data hwf8.1.2 (vc8.mem_code2 hwf8) hf2 hfit hi
    have hd3_8 := rd64_flat_data hwf8.1.2 (vc8.mem_code3 hwf8) hf3 hfit hi
    obtain ⟨s9, hrch9, hrip9, hmem9, hk9, hr9, hs9_0, _, _, hs9_oth⟩ :=
      tmr_ldIdx8_vote_eq_spec (a := 4199341) (done := 4199400) (disp := 8) (ldLen := 5) (chk := 16)
        (by rw [hrip8, hrip7]) hwf8.2.2.2.1 hwf8.2.2.2.2.2.1 hs8_8
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        hc1_8 hc2_8 hc3_8 hd1_8.2 hd2_8.2 hd3_8.2 hdlenW
    have hwf9 : vc8.wf pr ls s9 := vc8.wf_step_misc hwf8 hmem9 hr9
      (by rw [hs9_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hwf8.2.2.2.2.1)
      (by rw [hs9_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hwf8.2.2.2.2.2.1)
    obtain ⟨s10, h10, hrip10, hk10, _, _, hwf10, h624_10, hpres10⟩ :=
      step_wr_ctx_scratch (step_at_vc 16 hrip9 (by decide : (4199400, (.ofInt (.st 15 624 0)), 7) ∈ vmTableChunk 16))
        hwf9 hs9_0 (by decide) (by decide)
    have h616_10 : rd64 s10.mem (ea ctxBase 616) = some addr :=
      hpres10 616 addr (Or.inl (by decide)) (by decide)
        (by rw [hmem9, Nat.mod_eq_of_lt haddrW] at *; exact h616_8)
    have h624_10' : rd64 s10.mem (ea ctxBase 624) = some dlen := by
      rwa [Nat.mod_eq_of_lt hdlenW] at h624_10
    have hrch10 : Reaches vmImage s s10 :=
      Reaches.trans hrch3 (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
        (Reaches.trans hrch7 (Reaches.step_trans h08 (Reaches.trans hrch9 (Reaches.step h10)))))))
    have hk10_eq : s10.k = s.k := by rw [hk10, hk9, hk8, hk7]; exact hk1
    rcases step_push_ctx_val_vc (a := 4199407) (chk := 16) (off := 616) (by rw [hrip10, hrip9])
        hwf10 (oracleAdmits_of_reaches hor hrch10) h616_10 haddrW (by omega)
        (by decide) (by decide) (by decide) (by decide) with
      ⟨vc11, s11, hrch11, hrip11, hk11, hor11, hwf11, hpres11⟩ | ⟨hhit, hends⟩
    · have h624_11 := hpres11 624 dlen (by decide) (by decide) h624_10'
      rcases step_push_ctx_val_vc (a := 4199430) (chk := 16) (off := 624) hrip11
          hwf11 hor11 h624_11 hdlenW (by omega)
          (by decide) (by decide) (by decide) (by decide) with
        ⟨vc12, s12, hrch12, hrip12, hk12, hor12, hwf12, _⟩ | ⟨hhit, hends⟩
      · rcases adv_pc_finish_vc (a := 4199453) (chk := 16) hrip12 hwf12 hor12 hrs hpcW
            (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) with
          ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
        · exact Or.inl ⟨sf, Reaches.trans hrch10 (Reaches.trans hrch11 (Reaches.trans hrch12 hrchf)),
            by rw [hkf, hk12, hk11, hk10_eq], horf, hrelf⟩
        · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch10 (Reaches.trans hrch11 hrch12)) hhit,
            Ends.of_reaches (Reaches.trans hrch10 (Reaches.trans hrch11 hrch12)) hends⟩
      · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch10 hrch11) hhit,
          Ends.of_reaches (Reaches.trans hrch10 hrch11) hends⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch10 hhit, Ends.of_reaches hrch10 hends⟩
  · have hnone : pr.data[i]? = none := List.getElem?_eq_none (by omega)
    have hlstep : SporosProvLen.lstep pr ls = .fail .bounds := by
      simp [SporosProvLen.lstep, hrs, hfetch, hnone]
    rw [hlstep]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := errEntry .bounds } := by
      rw [step_at_vc 15 rfl (by decide : (4199264, (.ofInt (.jcc 3 4201150)), 6) ∈ vmTableChunk 15),
        exec_jcc (cond_aluF_sub_ae i pr.data.length)]
      dsimp [s3]; rw [decide_eq_true (by omega)]; rfl
    let s4 := { s3 with rip := errEntry .bounds }
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h hmem1 hr2 hs3_4 hs3_15
    exact lift_fail_outcome (Reaches.trans hrch3 (Reaches.step h04)) hk1
      (L_fail_from_vc_spec (e := .bounds) rfl hwf4)

end SporosVmOpsData
