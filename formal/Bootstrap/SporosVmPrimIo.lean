import Bootstrap.SporosVmPrimMem

/-!
# Native x86-64 VM I/O Primitive Proofs (`P_getb`, `P_putb`, `P_putn`)

Simulation proofs over `X86Vm` for the 3 I/O primitives (`getb`, `putb`, `putn`),
plus reusable combinators (`nodup_replace_at`, `drop_take_eq_of_getElem?_eq`,
`step_call_stk`, `step_wr_ctx_ipos`, `wf_update_out`) shared across table and
non-primitive instruction proofs.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimIo

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### Reusable `Nodup`, `ctx`-slice, `call`, and `VmCtx` I/O combinators -/

theorem nodup_replace_at {α : Type} {l1 l2 : List α} {a b : α}
    (h : (l1 ++ a :: l2).Nodup) (hb : ∀ x ∈ l1 ++ l2, x ≠ b) :
    (l1 ++ b :: l2).Nodup := by
  rw [List.nodup_append] at h ⊢
  rcases h with ⟨h1, h2, h12⟩
  rw [List.nodup_cons] at h2 ⊢
  refine ⟨h1, ⟨fun hmem => hb b (List.mem_append_right _ hmem) rfl, h2.2⟩, fun x hx y hy => ?_⟩
  rcases List.mem_cons.mp hy with rfl | hy
  · exact hb x (List.mem_append_left _ hx)
  · exact h12 x hx y (List.mem_cons_of_mem _ hy)

theorem drop_take_eq_of_getElem?_eq {xs ys : List Nat} {m0 k : Nat}
    (heq : ∀ j, m0 ≤ j → j < m0 + k → xs[j]? = ys[j]?) :
    (xs.drop m0).take k = (ys.drop m0).take k := by
  apply List.ext_getElem?
  intro i
  by_cases hi : i < k
  · rw [List.getElem?_take_of_lt hi, List.getElem?_take_of_lt hi,
      List.getElem?_drop, List.getElem?_drop]
    exact heq (m0 + i) (Nat.le_add_right _ _) (by omega)
  · rw [List.getElem?_eq_none (by simp only [List.length_take]; omega),
      List.getElem?_eq_none (by simp only [List.length_take]; omega)]

theorem wf_io_info {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    vc.inpReg.hi ≤ 2 ^ 47 ∧
    vc.outReg.rw = true ∧ 8 ≤ vc.outCap ∧ vc.outReg.data.length = vc.outCap ∧
      vc.outCap % X86Env.page = 0 ∧ vc.outReg.hi ≤ 2 ^ 47 ∧
    leVal ((vc.ctxReg.data.drop 176).take 8) = vc.inpReg.base ∧
    leVal ((vc.ctxReg.data.drop 184).take 8) = vc.inpLen ∧
    leVal ((vc.ctxReg.data.drop 192).take 8) = ls.ipos ∧
    leVal ((vc.ctxReg.data.drop 200).take 8) = vc.outReg.base ∧
    leVal ((vc.ctxReg.data.drop 208).take 8) = ls.out.length ∧
    leVal ((vc.ctxReg.data.drop 216).take 8) = vc.outCap ∧
    leVal ((vc.ctxReg.data.drop 224).take 8) = ls.pc ∧
    InpRel vc.inpReg.data ls.ipos vc.inpLen ls.inp ∧
    OutRel vc.outReg.data ls.out.length ls.out ∧ ls.out.length ≤ vc.outCap := by
  rcases h with ⟨_, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, _, _, _,
    _, _, _, _, _, _, _, _, _, _, hinprel, houtrel, houtle, _, _⟩
  exact ⟨hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hinprel, houtrel, houtle⟩

theorem step_call_stk {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {t : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.call t)) 5 s)
    (h : vc.wf pr ls s) (hretW : s.rip + 5 < X86Env.W) :
    let sp1 := vc.sp - 8
    let stk1 := { vc.stk with data := writeBytes vc.stk.data (sp1 - vc.stk.base) (bytes8 (s.rip + 5)) }
    ∃ s1,
      X86Vm.step vmImage s = .ok s1 ∧
      s1.rip = t ∧ s1.k = s.k ∧ s1.fl = s.fl ∧
      CodeMem s1.mem ∧ BytesLt s1.mem ∧ (∀ u ∈ s1.mem, 0 < u.data.length) ∧
      RegsOk s1 ∧ s1.reg 4 = some sp1 ∧ s1.reg 15 = some ctxBase ∧
      (∀ r, r ≠ 4 → s1.reg r = s.reg r) ∧
      stk1 ∈ s1.mem ∧ stk1.base = vc.stk.base ∧ stk1.rw = true ∧
      stk1.data.length = vc.stk.data.length ∧
      16 ≤ sp1 ∧ sp1 < X86Env.W ∧ stk1.base ≤ sp1 - 16 ∧ sp1 + 8 ≤ stk1.hi ∧
      rd64 s1.mem sp1 = some (s.rip + 5) ∧
      (∀ u ∈ s.mem, u.base ≠ vc.stk.base → u ∈ s1.mem) := by
  have hsb := vc.sp_bounds h
  have hstkrw : vc.stk.rw = true := h.2.2.2.2.2.2.2.2.1
  have hsp : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hsp1_eq : (vc.sp + X86Env.W - 8) % X86Env.W = vc.sp - 8 := hsb.2.2.2.2.2.2
  have hlo : vc.stk.base ≤ vc.sp - 8 := by omega
  have hhi : (vc.sp - 8) + 8 ≤ vc.stk.hi := by omega
  obtain ⟨_, m1, hwr, _, _, hm1wf, hstk1, hoth⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_stk h) hstkrw hlo hhi (v := s.rip + 5)
  have hwr' : wr64 s.mem ((vc.sp + X86Env.W - 8) % X86Env.W) (s.rip + 5) = some m1 := by
    rw [hsp1_eq]; exact hwr
  have hs1 : X86Vm.step vmImage s = .ok { s.set 4 (some (vc.sp - 8)) with mem := m1, rip := t } := by
    rw [hstep, exec_call hsp hwr', hsp1_eq]
  let s1 := { s.set 4 (some (vc.sp - 8)) with mem := m1, rip := t }
  have hrd : rd64 m1 (vc.sp - 8) = some (s.rip + 5) := by
    refine rd64_of_mem hm1wf hstk1 hlo (by simp only [Region.hi, writeBytes_length]; exact hhi) ?_
    have hlen : vc.sp - 8 - vc.stk.base + 8 ≤ vc.stk.data.length := by
      have := hsb.2.1; simp only [Region.hi] at this; omega
    rw [leVal_writeBytes_same _ _ _ hlen, Nat.mod_eq_of_lt hretW]
  refine ⟨s1, hs1, rfl, rfl, rfl,
    h.1.wr64_preserve hwr', bytesLt_wr64 h.2.1 hwr', len_pos_wrN h.2.2.1 hwr',
    regsOk_set hreg (by decide) _, reg_set_self_ok' hreg (by decide) _,
    by change (s.set 4 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15,
    fun r hr => by change (s.set 4 _).reg r = s.reg r; exact reg_set_ne hr.symm _,
    hstk1, rfl, hstkrw, writeBytes_length _ _ _,
    by omega, by omega, by dsimp; omega,
    by simp only [Region.hi, writeBytes_length]; have := hsb.2.1; simp only [Region.hi] at this; omega,
    hrd, hoth⟩

theorem step_wr_ctx_ipos {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {len r c : Nat} {rest : List Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 192 r)) len s)
    (h : vc.wf pr ls s) (hr : s.reg r = some (ls.ipos + 1))
    (hinp : ls.inp = c :: rest) (hW : ls.ipos + 1 < X86Env.W) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 192 (bytes8 (ls.ipos + 1)) }
    let vc' := { vc with ctxReg := ctx' }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧
      s'.rip = s.rip + len ∧ s'.k = s.k ∧ s'.regs = s.regs ∧ s'.fl = s.fl ∧
      vc'.wf pr { ls with inp := rest, ipos := ls.ipos + 1 } s' ∧
      (∀ m0 u, (m0 + 8 ≤ 192 ∨ 200 ≤ m0) → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hb := vc.ctx_bounds h
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hea : ea ctxBase 192 = ctxBase + 192 := ea_add_eq (by have : ctxSize = 4096 := rfl; omega)
  obtain ⟨_, m', hwr, _, _, _, hmem', hoth⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_ctx h) hb.2.1 (a := ctxBase + 192)
      (by rw [hb.1]; omega) (by simp only [Region.hi, hb.1]; have : ctxSize = 4096 := rfl; omega)
      (v := ls.ipos + 1)
  have hsub : ctxBase + 192 - vc.ctxReg.base = 192 := by rw [hb.1]; omega
  rw [hsub] at hmem'
  rw [← hea] at hwr
  have hcm' := h.1.wr64_preserve hwr
  have hbl' := bytesLt_wr64 h.2.1 hwr
  have hpos' := len_pos_wrN h.2.2.1 hwr
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 192 (bytes8 (ls.ipos + 1)) }
  let vc' := { vc with ctxReg := ctx' }
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by rw [hstep, exec_st h15 hr hwr]
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  obtain ⟨_hcm, _hblt, _hpos, hregs, hsp, h15s, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    _hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    simpa [vc', ctx', VmCtx.regions] using hnodup
  have hmem_all : ∀ u ∈ vc'.regions, u ∈ s'.mem := by
    intro u hu
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    rcases hu with rfl | rfl | huc | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | (rw [huc]; exact hmem')
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hctx3' : ctxSize ≤ ctx'.data.length := by dsimp [ctx']; rw [writeBytes_length]; exact hctx3
  have hctx4' : ctx'.hi ≤ 2 ^ 47 := by simpa only [Region.hi, ctx', writeBytes_length] using hctx4
  have hslot : ∀ j, (j + 8 ≤ 192 ∨ 200 ≤ j) → j + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop j).take 8) = leVal ((vc.ctxReg.data.drop j).take 8) := by
    intro j hj hlim
    dsimp [ctx']
    rw [writeBytes_drop_take_disjoint vc.ctxReg.data 192 j 8 (bytes8 (ls.ipos + 1))
      (by rw [bytes8_length]; exact hj) (by have : ctxSize = 4096 := rfl; omega)]
  have hsl24' : leVal ((ctx'.data.drop 192).take 8) = ls.ipos + 1 := by
    dsimp [ctx']
    rw [leVal_writeBytes_same vc.ctxReg.data 192 (ls.ipos + 1) (by have : ctxSize = 4096 := rfl; omega),
      Nat.mod_eq_of_lt hW]
  have hinprel' : InpRel vc.inpReg.data (ls.ipos + 1) vc.inpLen rest := by
    obtain ⟨hsum, hlen, hget⟩ := hinprel
    rw [hinp] at hsum hget
    refine ⟨by simp at hsum; omega, hlen, ?_⟩
    intro j hj
    have h1 := hget (j + 1) (by simp; omega)
    rw [show ls.ipos + (j + 1) = ls.ipos + 1 + j by omega] at h1
    simpa [List.getElem!_cons_succ] using h1
  have hwf' : vc'.wf pr { ls with inp := rest, ipos := ls.ipos + 1 } s' :=
    ⟨hcm', hbl', hpos', hregs, hsp, h15s, hnodup', hmem_all,
      hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3', hctx4', hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
      hinp1, hout1, hout2, hout3, hout4, hout5,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl0,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl1,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl2,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl3,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl4,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl5,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl6,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl7,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl8,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl9,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl10,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl11,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl12,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl13,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl14,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl15,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl16,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl17,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl18,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl19,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl20,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl21,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl22,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl23,
      hsl24',
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl25,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl26,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl27,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl31,
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel', houtrel, houtle,
      ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨s', hs, rfl, rfl, rfl, rfl, hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0 hmlim, hle_ctx m0 u hmlim hprev])

theorem wf_update_out {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {outReg' ctx' stk' : Region} {outCap' : Nat} {out' : List Nat}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hout_mem : outReg' ∈ s'.mem) (hout_rw : outReg'.rw = true) (hout_hi : outReg'.hi ≤ 2 ^ 47)
    (hout_cap_ge : 8 ≤ outCap') (hout_len_eq : outReg'.data.length = outCap')
    (hout_pg : outCap' % X86Env.page = 0)
    (hctx_mem : ctx' ∈ s'.mem) (hctx_base : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_len : ctx'.data.length = vc.ctxReg.data.length)
    (hc200 : leVal ((ctx'.data.drop 200).take 8) = outReg'.base)
    (hc208 : leVal ((ctx'.data.drop 208).take 8) = out'.length)
    (hc216 : leVal ((ctx'.data.drop 216).take 8) = outCap')
    (hctx_oth : ∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]?)
    (hstk_mem : stk' ∈ s'.mem) (hstk_base : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_len : stk'.data.length = vc.stk.data.length)
    (hout_ne : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ outReg'.base)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s'.mem)
    (hout_rel : OutRel outReg'.data out'.length out') (hout_le : out'.length ≤ outCap') :
    let vc' := { vc with stk := stk', ctxReg := ctx', outReg := outReg', outCap := outCap' }
    vc'.wf pr { ls with out := out' } s' ∧
    (∀ m0 u, (m0 + 8 ≤ 200 ∨ 224 ≤ m0) → m0 + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  have hcode_ne_out : codeRegion.base ≠ vc.outReg.base :=
    vc.ne_bases h (i := 0) (j := 15) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_ctx : codeRegion.base ≠ vc.ctxReg.base :=
    vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_stk : codeRegion.base ≠ vc.stk.base :=
    vc.ne_bases h (i := 0) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1 hcode_ne_out hcode_ne_ctx hcode_ne_stk, hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, _hout1, _hout2, _hout3, _hout4, _hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, _hsl25, _hsl26, _hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, _houtrel, _houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', outReg := outReg', outCap := outCap' }
  have hmem_all : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem
      | exact hctx_mem
      | exact hout_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    let pfx : List Region :=
      [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3, vc.dsReg, vc.auxReg,
       vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg, vc.tabsReg, vc.kvReg, vc.inpReg]
    have hnd_old : (pfx.map Region.base ++ vc.outReg.base :: []).Nodup := by
      simpa [pfx, VmCtx.regions] using hnodup
    have hgoal : (pfx.map Region.base ++ outReg'.base :: []).Nodup := by
      refine nodup_replace_at hnd_old ?_
      intro x hx
      simp only [List.append_nil, List.mem_map] at hx
      rcases hx with ⟨u, hu, rfl⟩
      have hu_in : u ∈ vc.regions := by
        simp only [VmCtx.regions]
        simp only [pfx, List.mem_cons, List.not_mem_nil, or_false] at hu
        rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
          simp
      have hu_mem : u ∈ s.mem := hmem u hu_in
      have hu_ne : u.base ≠ vc.outReg.base := by
        rw [List.nodup_append] at hnd_old
        exact hnd_old.2.2 u.base (List.mem_map_of_mem hu) vc.outReg.base (List.mem_cons_self ..)
      exact hout_ne u hu_mem hu_ne
    simpa [vc', pfx, VmCtx.regions, hstk_base, hctx_base] using hgoal
  have hslot : ∀ m0, (m0 + 8 ≤ 200 ∨ 224 ≤ m0) →
      leVal ((ctx'.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
    intro m0 hm0
    congr 1
    exact drop_take_eq_of_getElem?_eq fun j hj1 hj2 => hctx_oth j (by omega)
  have hwf' : vc'.wf pr { ls with out := out' } s' :=
    ⟨hcm', hbl', hpos', hr', hsp', h15', hnodup', hmem_all,
      by rw [hstk_rw]; exact hstk1,
      by rw [hstk_base]; exact hstk2,
      by simpa only [vc', Region.hi, hstk_base, hstk_len] using hstk3,
      hstk4,
      by rw [hctx_base]; exact hctx1,
      by rw [hctx_rw]; exact hctx2,
      by rw [hctx_len]; exact hctx3,
      by simpa only [vc', Region.hi, hctx_base, hctx_len] using hctx4,
      hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
      hinp1, hout_rw, hout_cap_ge, hout_len_eq, hout_pg, hout_hi,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl0,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl1,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl2,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl3,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl4,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl5,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl6,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl7,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl8,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl9,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl10,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl11,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl12,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl13,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl14,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl15,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl16,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl17,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl18,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl19,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl20,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl21,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl22,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl23,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl24,
      hc200, hc208, hc216,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl31,
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel, hout_rel, hout_le,
      ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0, hle_ctx m0 u hmlim hprev])

/-! ### `P_getb`, `P_putb`, `P_putn` -/

theorem P_getb_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_getb)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .getb := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  obtain ⟨hinp_hi, _, _, _, _, _, hsl176, hsl184, hsl192, _, _, _, hsl224, hinp_rel, _, _⟩ := wf_io_info h
  have hrd192 : rd64 s.mem (ea ctxBase 192) = some ls.ipos := vc.rd64_ctx_of_eq h (by decide) hsl192
  have hrd184 : rd64 s.mem (ea ctxBase 184) = some vc.inpLen := vc.rd64_ctx_of_eq h (by decide) hsl184
  have hrd176 : rd64 s.mem (ea ctxBase 176) = some vc.inpReg.base := vc.rd64_ctx_of_eq h (by decide) hsl176
  have hrd224 : rd64 s.mem (ea ctxBase 224) = some ls.pc := vc.rd64_ctx_of_eq h (by decide) hsl224
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ipos) with rip := 4198783 } := by
    rw [step_at_vc 14 hrip (by decide : (SporosVmBin.P_getb, (.ofInt (.ld 1 15 192)), 7) ∈ vmTableChunk 14),
      exec_ld h15 hrd192, hrip]; rfl
  let s1 := { s.set 1 (some ls.ipos) with rip := 4198783 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_1 : s1.reg 1 = some ls.ipos := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some vc.inpLen) with rip := 4198790 } := by
    rw [step_at_vc 14 rfl (by decide : (4198783, (.ofInt (.ld 2 15 184)), 7) ∈ vmTableChunk 14),
      exec_ld hs1_15 hrd184]
  let s2 := { s1.set 2 (some vc.inpLen) with rip := 4198790 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some ls.ipos := by
    change (s1.set 2 _).reg 1 = some ls.ipos; rw [reg_set_ne (by decide)]; exact hs1_1
  have hs2_2 : s2.reg 2 = some vc.inpLen := reg_set_self_ok' hr1 (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub ls.ipos vc.inpLen).2, rip := 4198793 } := by
    rw [step_at_vc 14 rfl (by decide : (4198790, (.ofInt (.cmp 1 2)), 3) ∈ vmTableChunk 14),
      exec_cmp hs2_1 hs2_2]
  let s3 := { s2 with fl := (X86Int.aluF .sub ls.ipos vc.inpLen).2, rip := 4198793 }
  have hcb : X86Int.cond s3.fl 2 = some (decide (ls.ipos < vc.inpLen)) := by
    dsimp [s3]; exact cond_aluF_sub_b ls.ipos vc.inpLen
  have hW47 : 2 ^ 47 < X86Env.W := by decide
  have hipos_le : ls.ipos + ls.inp.length = vc.inpLen := hinp_rel.1
  have hinpLen_le : vc.inpLen ≤ vc.inpReg.data.length := hinp_rel.2.1
  have hinpLen_ltW : vc.inpLen < X86Env.W := by simp only [Region.hi] at hinp_hi; omega
  cases hinp_eq : ls.inp with
  | nil =>
    have hnotlt : ¬ ls.ipos < vc.inpLen := by simp [hinp_eq] at hipos_le; omega
    have hcb_f : X86Int.cond s3.fl 2 = some false := by rw [hcb, decide_eq_false hnotlt]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198795 } := by
      rw [step_at_vc 14 rfl (by decide : (4198793, (.ofInt (.jcc 2 4198810)), 2) ∈ vmTableChunk 14),
        exec_jcc hcb_f]; rfl
    let s4 := { s3 with rip := 4198795 }
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 0 (some 18446744073709551615) with rip := 4198805 } := by
      rw [step_at_vc 14 rfl (by decide : (4198795, (.ofInt (.movI 0 18446744073709551615)), 10) ∈ vmTableChunk 14),
        exec_movI_lt (by decide : 18446744073709551615 < X86Env.W)]
    let s5 := { s4.set 0 (some 18446744073709551615) with rip := 4198805 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_0 : s5.reg 0 = some 18446744073709551615 := reg_set_self_ok' hr2 (by decide) _
    have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := SporosVmBin.L_prim_epilogue_mk1 } := by
      rw [step_at_vc 14 rfl (by decide : (4198805, (.ofInt (.jmp 4195210)), 5) ∈ vmTableChunk 14), exec_jmp]
      rfl
    let s6 := { s5 with rip := SporosVmBin.L_prim_epilogue_mk1 }
    have hrch6 : Reaches vmImage s s6 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step h06)))))
    have hwf6 : vc.wf pr ls s6 :=
      vc.wf_step_misc h rfl hr5
        (by change (s4.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s1.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
        (by change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s1.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            exact hs1_15)
    have hqW : SporosProvLen.lq (ls.ds.take 0) ls.pc < X86Env.W := by
      simp [SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by decide)
    obtain ⟨ctxData', s7, hrch7, hrip7, hk7, hwf7, _, h360_7, h368_7, hrest7⟩ :=
      L_prim_epilogue_mk1_spec (vc := vc) (pr := pr) (ls := ls) (s := s6)
        (w := 18446744073709551615) (q := SporosProvLen.lq (ls.ds.take 0) ls.pc)
        rfl hwf6 hs5_0 h272 hqW
    let vc7 := { vc with ctxReg := { vc.ctxReg with data := ctxData' } }
    have h280_7 : rd64 s7.mem (ea ctxBase 280) = some 0 :=
      vc7.rd64_ctx_of_eq hwf7 (by decide)
        (by rw [hrest7 280 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) h280)
    have h288_7 : rd64 s7.mem (ea ctxBase 288) = some 1 :=
      vc7.rd64_ctx_of_eq hwf7 (by decide)
        (by rw [hrest7 288 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) h288)
    have hrch07 : Reaches vmImage s s7 := Reaches.trans hrch6 hrch7
    have hor7 : OracleAdmits s7 := oracleAdmits_of_reaches hor hrch07
    let v_eof : SporosProvLen.LVal := ⟨Sporos.wrap (Sporos.M - 1), SporosProvLen.lq [] ls.pc⟩
    have hv_eof_w : v_eof.w = 18446744073709551615 := by dsimp [v_eof]; decide
    have hv_eof_p : v_eof.plen = SporosProvLen.lq (ls.ds.take 0) ls.pc := rfl
    have hepi := L_prim_epilogue_oracle_spec (vc := vc7) (pr := pr) (ls := ls) (s := s7)
      (e_in := 0) (outs := [v_eof]) (f := f) (rest := rest)
      hrip7 hwf7 hor7 hrs hpcW (Nat.zero_le _) (by simp)
      (by intro v hv; simp at hv; subst hv; exact ⟨by rw [hv_eof_w]; decide, hqW⟩)
      h280_7 h288_7
      (by intro i hi; simp at hi; subst hi
          refine ⟨by simpa [hv_eof_w, Nat.mod_eq_of_lt (by decide : 18446744073709551615 < X86Env.W)] using h360_7,
                  by simpa [hv_eof_p] using h368_7⟩)
    unfold PrimPostOutcome
    have hexec : SporosProvLen.lexecPrim .getb (ls.ds.take (Sporos.Prim.getb.eff.1)) ls = .ok [v_eof] ls := by
      simp [Sporos.Prim.eff, SporosProvLen.lexecPrim, hinp_eq, v_eof]
    rw [hexec]
    rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch07 hrchf, by rw [hkf, hk7]; rfl, horf, by simpa [Sporos.Prim.eff] using hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch07 hhit, Ends.of_reaches hrch07 hends⟩
  | cons c inpRest =>
    have hlt : ls.ipos < vc.inpLen := by simp [hinp_eq] at hipos_le; omega
    have hcb_t : X86Int.cond s3.fl 2 = some true := by rw [hcb, decide_eq_true hlt]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4198810 } := by
      rw [step_at_vc 14 rfl (by decide : (4198793, (.ofInt (.jcc 2 4198810)), 2) ∈ vmTableChunk 14),
        exec_jcc hcb_t]; rfl
    let s4 := { s3 with rip := 4198810 }
    have hs4_15 : s4.reg 15 = some ctxBase := by
      change (s1.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 8 (some vc.inpReg.base) with rip := 4198817 } := by
      rw [step_at_vc 14 rfl (by decide : (4198810, (.ofInt (.ld 8 15 176)), 7) ∈ vmTableChunk 14),
        exec_ld hs4_15 hrd176]
    let s5 := { s4.set 8 (some vc.inpReg.base) with rip := 4198817 }
    have hr5 : RegsOk s5 := regsOk_set hr2 (by decide) _
    have hs5_8 : s5.reg 8 = some vc.inpReg.base := reg_set_self_ok' hr2 (by decide) _
    have hs5_1 : s5.reg 1 = some ls.ipos := by
      change (s4.set 8 _).reg 1 = some ls.ipos; rw [reg_set_ne (by decide)]; exact hs2_1
    have hbyte : vc.inpReg.data[ls.ipos]? = some (c % 256) := by
      have h0 := hinp_rel.2.2 0 (by simp [hinp_eq])
      simpa [hinp_eq] using h0
    have hea_inp : eaIdx vc.inpReg.base ls.ipos 1 0 = vc.inpReg.base + ls.ipos := by
      have hltW : vc.inpReg.base + 1 * ls.ipos + 0 < X86Env.W := by
        simp only [Region.hi] at hinp_hi; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_b : rdMem .w8 s5.mem (eaIdx vc.inpReg.base ls.ipos 1 0) = some (c % 256) := by
      change rdMem .w8 s.mem _ = some (c % 256)
      rw [hea_inp]
      exact rdMem_w8_at h.1.2 (vc.mem_inp h) (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hbyte)
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 0 (some (c % 256)) with rip := 4198822 } := by
      rw [step_at_vc 14 rfl (by decide : (4198817, (.ldIdx .w8 0 8 1 1 0), 5) ∈ vmTableChunk 14),
        exec_ldIdx hs5_8 hs5_1 hrd_b]
    let s6 := { s5.set 0 (some (c % 256)) with rip := 4198822 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_1 : s6.reg 1 = some ls.ipos := by
      change (s5.set 0 _).reg 1 = some ls.ipos; rw [reg_set_ne (by decide)]; exact hs5_1
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 1 (some (ls.ipos + 1)) with fl := (X86Int.aluF .add ls.ipos 1).2, rip := 4198826 } := by
      rw [step_at_vc 14 rfl (by decide : (4198822, (.ofInt (.aluI .add 1 1)), 4) ∈ vmTableChunk 14),
        exec_aluI_add_lt hs6_1 (by decide), Nat.mod_eq_of_lt (by omega : ls.ipos + 1 < X86Env.W)]
    let s7 := { s6.set 1 (some (ls.ipos + 1)) with fl := (X86Int.aluF .add ls.ipos 1).2, rip := 4198826 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
    have hs7_1 : s7.reg 1 = some (ls.ipos + 1) := reg_set_self_ok' hr6 (by decide) _
    have hs7_0 : s7.reg 0 = some (c % 256) := by
      change (s6.set 1 _).reg 0 = some (c % 256); rw [reg_set_ne (by decide)]
      exact reg_set_self_ok' hr5 (by decide) _
    have hwf7 : vc.wf pr ls s7 :=
      vc.wf_step_misc h rfl hr7
        (by change (s6.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s5.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s4.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s1.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
            change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
        (by change (s6.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s5.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s4.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs4_15)
    let ls1 : SporosProvLen.LSt := { ls with inp := inpRest, ipos := ls.ipos + 1 }
    obtain ⟨s8, h08, hrip8, hk8, hregs8, _, hwf8, hpres8⟩ :=
      step_wr_ctx_ipos (vc := vc) (pr := pr) (ls := ls) (s := s7) (len := 7) (r := 1) (c := c) (rest := inpRest)
        (step_at_vc 14 rfl (by decide : (4198826, (.ofInt (.st 15 192 1)), 7) ∈ vmTableChunk 14))
        hwf7 hs7_1 hinp_eq (by omega)
    let vc8 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 192 (bytes8 (ls.ipos + 1)) } }
    have hs8_0 : s8.reg 0 = some (c % 256) := by
      simp only [X86Int.St.reg, hregs8]; exact hs7_0
    have h280_8 : rd64 s8.mem (ea ctxBase 280) = some 0 := hpres8 280 0 (Or.inr (by decide)) (by decide) h280
    have h288_8 : rd64 s8.mem (ea ctxBase 288) = some 1 := hpres8 288 1 (Or.inr (by decide)) (by decide) h288
    have h224_8 : rd64 s8.mem (ea ctxBase 224) = some ls.pc := hpres8 224 ls.pc (Or.inr (by decide)) (by decide) hrd224
    obtain ⟨_, m9, _, _, h09, hwf9, h360_9, hrest9⟩ :=
      vc8.wr64_ctx_scratch_slot hwf8 360 _ 4198840 (by decide) (by decide) hs8_0
        (fun {m'} hwr => by
          rw [step_at_vc 14 hrip8 (by decide : (4198833, (.ofInt (.st 15 360 0)), 7) ∈ vmTableChunk 14),
            exec_st hwf8.2.2.2.2.2.1 hs8_0 hwr, hrip8])
    let s9 := { s8 with mem := m9, rip := 4198840 }
    let vc9 := { vc8 with ctxReg := { vc8.ctxReg with data := writeBytes vc8.ctxReg.data 360 (bytes8 (c % 256)) } }
    have h224_9 : rd64 s9.mem (ea ctxBase 224) = some ls.pc :=
      vc9.rd64_ctx_of_eq hwf9 (by decide)
        (by rw [hrest9 224 (Or.inl (by decide)) (by decide)]; exact ctx_leVal_of_rd64 hwf8 (by decide) h224_8)
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some ls.pc) with rip := 4198847 } := by
      rw [step_at_vc 14 rfl (by decide : (4198840, (.ofInt (.ld 2 15 224)), 7) ∈ vmTableChunk 14),
        exec_ld hwf9.2.2.2.2.2.1 h224_9]
    let s10 := { s9.set 2 (some ls.pc) with rip := 4198847 }
    have hr10 : RegsOk s10 := regsOk_set hwf9.2.2.2.1 (by decide) _
    have hs10_2 : s10.reg 2 = some ls.pc := reg_set_self_ok' hwf9.2.2.2.1 (by decide) _
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10.set 2 (some ((ls.pc + 1) % X86Env.W)) with fl := (X86Int.aluF .add ls.pc 1).2, rip := 4198851 } := by
      rw [step_at_vc 14 rfl (by decide : (4198847, (.ofInt (.aluI .add 2 1)), 4) ∈ vmTableChunk 14),
        exec_aluI_add_lt hs10_2 (by decide)]
    let s11 := { s10.set 2 (some ((ls.pc + 1) % X86Env.W)) with fl := (X86Int.aluF .add ls.pc 1).2, rip := 4198851 }
    have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
    have hs11_2 : s11.reg 2 = some ((ls.pc + 1) % X86Env.W) := reg_set_self_ok' hr10 (by decide) _
    have hwf11 : vc9.wf pr ls1 s11 :=
      vc9.wf_step_misc hwf9 rfl hr11
        (by change (s10.set 2 _).reg 4 = some vc9.sp; rw [reg_set_ne (by decide)]
            change (s9.set 2 _).reg 4 = some vc9.sp; rw [reg_set_ne (by decide)]; exact hwf9.2.2.2.2.1)
        (by change (s10.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s9.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf9.2.2.2.2.2.1)
    obtain ⟨_, m12, _, _, h12, hwf12, h368_12, hrest12⟩ :=
      vc9.wr64_ctx_scratch_slot hwf11 368 _ 4198858 (by decide) (by decide) hs11_2
        (fun {m'} hwr => by
          rw [step_at_vc 14 rfl (by decide : (4198851, (.ofInt (.st 15 368 2)), 7) ∈ vmTableChunk 14),
            exec_st hwf11.2.2.2.2.2.1 hs11_2 hwr])
    let s12 := { s11 with mem := m12, rip := 4198858 }
    let vc12 := { vc9 with ctxReg := { vc9.ctxReg with data := writeBytes vc9.ctxReg.data 368 (bytes8 ((ls.pc + 1) % X86Env.W)) } }
    have h13 : X86Vm.step vmImage s12 = .ok { s12 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 14 rfl (by decide : (4198858, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 14), exec_jmp]
      rfl
    let s13 := { s12 with rip := SporosVmBin.L_prim_epilogue }
    have hwf13 : vc12.wf pr ls1 s13 := vc12.wf_step_misc hwf12 rfl hwf12.2.2.2.1 hwf12.2.2.2.2.1 hwf12.2.2.2.2.2.1
    have hrch13 : Reaches vmImage s s13 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09
            (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step_trans h12
              (Reaches.step h13))))))))))))
    have hor13 : OracleAdmits s13 := oracleAdmits_of_reaches hor hrch13
    have h280_13 : rd64 s13.mem (ea ctxBase 280) = some 0 :=
      vc12.rd64_ctx_of_eq hwf13 (by decide)
        (by rw [hrest12 280 (Or.inl (by decide)) (by decide), hrest9 280 (Or.inl (by decide)) (by decide)]
            exact ctx_leVal_of_rd64 hwf8 (by decide) h280_8)
    have h288_13 : rd64 s13.mem (ea ctxBase 288) = some 1 :=
      vc12.rd64_ctx_of_eq hwf13 (by decide)
        (by rw [hrest12 288 (Or.inl (by decide)) (by decide), hrest9 288 (Or.inl (by decide)) (by decide)]
            exact ctx_leVal_of_rd64 hwf8 (by decide) h288_8)
    have hc256W : c % 256 < X86Env.W := by
      have h1 : c % 256 < 256 := Nat.mod_lt c (by decide)
      have hW : X86Env.W = 2 ^ 64 := rfl
      omega
    have h360_13 : rd64 s13.mem (ea ctxBase 360) = some (c % 256) :=
      vc12.rd64_ctx_of_eq hwf13 (by decide)
        (by rw [hrest12 360 (Or.inl (by decide)) (by decide), h360_9, Nat.mod_eq_of_lt hc256W])
    have h368_13 : rd64 s13.mem (ea ctxBase 368) = some ((ls.pc + 1) % X86Env.W) :=
      vc12.rd64_ctx_of_eq hwf13 (by decide) (by rw [h368_12, Nat.mod_mod])
    let v_b : SporosProvLen.LVal := ⟨c % 256, (ls.pc + 1) % Sporos.M⟩
    have hepi := L_prim_epilogue_oracle_spec (vc := vc12) (pr := pr) (ls := ls1) (s := s13)
      (e_in := 0) (outs := [v_b]) (f := f) (rest := rest)
      rfl hwf13 hor13 hrs hpcW (Nat.zero_le _) (by simp)
      (by intro v hv; simp at hv; subst hv
          exact ⟨hc256W, by dsimp [v_b]; rw [M_eq_W']; exact Nat.mod_lt _ (by decide)⟩)
      h280_13 h288_13
      (by intro i hi; simp at hi; subst hi
          exact ⟨h360_13, by simpa [v_b, M_eq_W'] using h368_13⟩)
    unfold PrimPostOutcome
    have hexec : SporosProvLen.lexecPrim .getb (ls.ds.take (Sporos.Prim.getb.eff.1)) ls = .ok [v_b] ls1 := by
      simp [Sporos.Prim.eff, SporosProvLen.lexecPrim, hinp_eq, v_b, ls1]
    rw [hexec]
    rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch13 hrchf, by rw [hkf]; exact congrArg X86Env.Kern.out hk8, horf,
        by simpa [Sporos.Prim.eff, ls1] using hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch13 hhit, Ends.of_reaches hrch13 hends⟩

theorem P_putb_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_putb) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .putb := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 7 (some ls.ds[0]!.w) with rip := 4198870 } := by
    rw [step_at_vc 14 hrip (by decide : (SporosVmBin.P_putb, (.ofInt (.ld 7 15 296)), 7) ∈ vmTableChunk 14),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 7 (some ls.ds[0]!.w) with rip := 4198870 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_7 : s1.reg 7 = some ls.ds[0]!.w := reg_set_self_ok' hr (by decide) _
  have hwf1 : vc.wf pr ls s1 :=
    vc.wf_step_misc h rfl hr1
      (by change (s.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15)
  obtain ⟨s2, h02, hrip2, hk2, _, hcm2, hbl2, hpos2, hr2, hs2_4, hs2_15, hs2_oth,
      hstk2, hstk2_b, hstk2_rw, hstk2_len, h16, hsp2W, hstk2_lo, hstk2_hi, hrd_ret, hoth2⟩ :=
    step_call_stk (vc := vc) (pr := pr) (ls := ls) (s := s1) (t := SporosVmBin.fn_out_push_byte)
      (step_at_vc 14 rfl (by decide : (4198870, (.ofInt (.call 4201625)), 5) ∈ vmTableChunk 14))
      hwf1 (by change 4198870 + 5 < X86Env.W; decide)
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hor2 : OracleAdmits s2 := oracleAdmits_of_reaches hor hrch2
  have hs2_7 : s2.reg 7 = some ls.ds[0]!.w := by rw [hs2_oth 7 (by decide)]; exact hs1_7
  have hctx_b := vc.ctx_bounds h
  obtain ⟨_, hout_rw, hout_cap8, hout_len, hout_pg, hout_hi, _, _, _, hsl200, hsl208, hsl216, _, _, hout_rel, hout_le⟩ :=
    wf_io_info h
  have hne_out_ctx : vc.outReg.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 15) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_out : vc.stk.base ≠ vc.outReg.base :=
    vc.ne_bases h (i := 1) (j := 15) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ctx : vc.stk.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hctx2 : vc.ctxReg ∈ s2.mem := hoth2 vc.ctxReg (vc.mem_ctx hwf1) (by rw [hctx_b.1]; exact Ne.symm hne_stk_ctx)
  have hout2 : vc.outReg ∈ s2.mem := hoth2 vc.outReg (vc.mem_out hwf1) (Ne.symm hne_stk_out)
  have hcall := call_out_push_byte_oracle_spec (s := s2) (sp := vc.sp - 8) (retAddr := 4198875)
    (ctxBase := ctxBase) (outBase := vc.outReg.base) (outLen := ls.out.length) (outCap := vc.outCap)
    (b := ls.ds[0]!.w) (ctx := vc.ctxReg) (outReg := vc.outReg)
    (stk := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (4198870 + 5)) })
    hrip2 hr2 hor2 hbl2 hpos2 hs2_4 hs2_15 hs2_7
    hctx2 hctx_b.1 hctx_b.2.1 hctx_b.2.2.1
    hout2 rfl hout_rw hout_hi hne_out_ctx
    hstk2 hstk2_rw (by rw [hstk2_b]; exact hne_stk_out) (by rw [hstk2_b]; exact hne_stk_ctx)
    h16 hsp2W hstk2_lo hstk2_hi hout_le (by omega) hout_len hout_pg hctx_b.2.2.2.2.1
    hcm2.2 hrd_ret hsl200 hsl208 hsl216
  unfold PrimPostOutcome
  have htake : ls.ds.take 1 = [ls.ds[0]!] := by
    cases hd : ls.ds with
    | nil => simp [hd] at hle
    | cons a r => simp [List.take]
  have hexec : SporosProvLen.lexecPrim .putb (ls.ds.take (Sporos.Prim.putb.eff.1)) ls =
      .ok [] { ls with out := (ls.ds[0]!.w % 256) :: ls.out } := by
    simp [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  rw [hexec]
  rcases hcall with ⟨outReg', ctx', stk', m3, s3, outCap', hrch3, hrip3, hm3, hmwf3, hbl3, hpos3,
      hr3, hor3, hk3, hsp3, hcap_le, hlen_le, hout_pg', hout3_mem, hout3_rw, hout3_hi, hout3_data,
      hctx3_mem, hctx3_b, hctx3_rw, hctx3_len, hc200', hc208', hc216', hctx3_oth,
      hstk3_mem, hstk3_b, hstk3_rw, hstk3_len, hout3_ne, hoth3, hreg3_oth⟩ | ⟨hhit, hends⟩
  · have hsp3' : s3.reg 4 = some vc.sp := by
      have hsb := vc.sp_bounds h
      simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hsp3
    have hs3_15 : s3.reg 15 = some ctxBase := by
      rw [hreg3_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hs2_15
    let out' : List Nat := (ls.ds[0]!.w % 256) :: ls.out
    have hout_len_eq' : outReg'.data.length = outCap' := by
      rw [hout3_data, writeBytes_length, List.length_append, List.length_replicate, hout_len]; omega
    have hout_rel' : OutRel outReg'.data out'.length out' := by
      rw [hout3_data]
      simpa [out'] using outRel_push_byte (outRel_extend_zero (ext := outCap' - vc.outCap) hout_rel)
        (by rw [List.length_append, List.length_replicate, hout_len]; omega)
    have hout_ne_s : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ outReg'.base := by
      intro u hu hne
      by_cases hstku : u.base = vc.stk.base
      · rw [hstku, ← hstk2_b]
        exact hout3_ne _ hstk2 (by rw [hstk2_b]; exact hne_stk_out)
      · exact hout3_ne u (hoth2 u hu hstku) hne
    have hoth_s : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s3.mem := by
      intro u hu h1 h2 h3
      rw [hm3]
      exact hoth3 u (hoth2 u hu h3) h1 (by rw [← hctx_b.1]; exact h2) (by rw [hstk2_b]; exact h3)
    obtain ⟨hwf3, hpres3⟩ := wf_update_out h (by rw [hm3]; exact hmwf3) (by rw [hm3]; exact hbl3)
      (by rw [hm3]; exact hpos3) hr3 hsp3' hs3_15
      (by rw [hm3]; exact hout3_mem) hout3_rw hout3_hi (by omega) hout_len_eq' hout_pg'
      (by rw [hm3]; exact hctx3_mem) hctx3_b hctx3_rw hctx3_len
      hc200' (by simpa [out'] using hc208') hc216' hctx3_oth
      (by rw [hm3]; exact hstk3_mem) (by rw [hstk3_b, hstk2_b]) (by rw [hstk3_rw])
      (by rw [hstk3_len, hstk2_len]) hout_ne_s hoth_s hout_rel' (by simpa [out'] using hlen_le)
    let vc3 := { vc with stk := stk', ctxReg := ctx', outReg := outReg', outCap := outCap' }
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 14 hrip3 (by decide : (4198875, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 14), exec_jmp]
      rfl
    let s4 := { s3 with rip := SporosVmBin.L_prim_epilogue }
    have hwf4 : vc3.wf pr { ls with out := out' } s4 :=
      vc3.wf_step_misc hwf3 rfl hr3 hsp3' hs3_15
    have hrch04 : Reaches vmImage s s4 := Reaches.trans hrch2 (Reaches.trans hrch3 (Reaches.step h04))
    have hor4 : OracleAdmits s4 := oracleAdmits_of_reaches hor hrch04
    have h280_4 : rd64 s4.mem (ea ctxBase 280) = some 1 := hpres3 280 1 (Or.inr (by decide)) (by decide) h280
    have h288_4 : rd64 s4.mem (ea ctxBase 288) = some 0 := hpres3 288 0 (Or.inr (by decide)) (by decide) h288
    have hepi := L_prim_epilogue_oracle_spec (vc := vc3) (pr := pr) (ls := { ls with out := out' }) (s := s4)
      (e_in := 1) (outs := []) (f := f) (rest := rest)
      rfl hwf4 hor4 hrs hpcW hle (by decide) (by intro _ hv; cases hv) h280_4 h288_4
      (by intro _ hi; cases hi)
    rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch04 hrchf, by rw [hkf]; exact hk3.trans (by rw [hk2]; rfl), horf,
        by simpa [Sporos.Prim.eff, out'] using hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch04 hhit, Ends.of_reaches hrch04 hends⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch2 hhit, Ends.of_reaches hrch2 hends⟩

theorem P_putn_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_putn) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .putn := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 7 (some ls.ds[0]!.w) with rip := 4198887 } := by
    rw [step_at_vc 14 hrip (by decide : (SporosVmBin.P_putn, (.ofInt (.ld 7 15 296)), 7) ∈ vmTableChunk 14),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 7 (some ls.ds[0]!.w) with rip := 4198887 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_7 : s1.reg 7 = some ls.ds[0]!.w := reg_set_self_ok' hr (by decide) _
  have hwf1 : vc.wf pr ls s1 :=
    vc.wf_step_misc h rfl hr1
      (by change (s.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15)
  obtain ⟨s2, h02, hrip2, hk2, _, hcm2, hbl2, hpos2, hr2, hs2_4, hs2_15, hs2_oth,
      hstk2, hstk2_b, hstk2_rw, hstk2_len, h16, hsp2W, hstk2_lo, hstk2_hi, hrd_ret, hoth2⟩ :=
    step_call_stk (vc := vc) (pr := pr) (ls := ls) (s := s1) (t := SporosVmBin.fn_out_push_u64)
      (step_at_vc 14 rfl (by decide : (4198887, (.ofInt (.call 4201724)), 5) ∈ vmTableChunk 14))
      hwf1 (by change 4198887 + 5 < X86Env.W; decide)
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hor2 : OracleAdmits s2 := oracleAdmits_of_reaches hor hrch2
  have hs2_7 : s2.reg 7 = some ls.ds[0]!.w := by rw [hs2_oth 7 (by decide)]; exact hs1_7
  have hctx_b := vc.ctx_bounds h
  obtain ⟨_, hout_rw, hout_cap8, hout_len, hout_pg, hout_hi, _, _, _, hsl200, hsl208, hsl216, _, _, hout_rel, hout_le⟩ :=
    wf_io_info h
  have hne_out_ctx : vc.outReg.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 15) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_out : vc.stk.base ≠ vc.outReg.base :=
    vc.ne_bases h (i := 1) (j := 15) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ctx : vc.stk.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hctx2 : vc.ctxReg ∈ s2.mem := hoth2 vc.ctxReg (vc.mem_ctx hwf1) (by rw [hctx_b.1]; exact Ne.symm hne_stk_ctx)
  have hout2 : vc.outReg ∈ s2.mem := hoth2 vc.outReg (vc.mem_out hwf1) (Ne.symm hne_stk_out)
  have hcall := call_out_push_u64_oracle_spec (s := s2) (sp := vc.sp - 8) (retAddr := 4198892)
    (ctxBase := ctxBase) (outBase := vc.outReg.base) (outLen := ls.out.length) (outCap := vc.outCap)
    (v := ls.ds[0]!.w) (ctx := vc.ctxReg) (outReg := vc.outReg)
    (stk := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (4198887 + 5)) })
    hrip2 hr2 hor2 hbl2 hpos2 hs2_4 hs2_15 hs2_7
    hctx2 hctx_b.1 hctx_b.2.1 hctx_b.2.2.1
    hout2 rfl hout_rw hout_hi hne_out_ctx
    hstk2 hstk2_rw (by rw [hstk2_b]; exact hne_stk_out) (by rw [hstk2_b]; exact hne_stk_ctx)
    h16 hsp2W hstk2_lo hstk2_hi hout_le hout_cap8 hout_len hout_pg hctx_b.2.2.2.2.1
    hcm2.2 hrd_ret hsl200 hsl208 hsl216
  unfold PrimPostOutcome
  have htake : ls.ds.take 1 = [ls.ds[0]!] := by
    cases hd : ls.ds with
    | nil => simp [hd] at hle
    | cons a r => simp [List.take]
  let out' : List Nat := (bytes8 ls.ds[0]!.w).reverse ++ ls.out
  have hexec : SporosProvLen.lexecPrim .putn (ls.ds.take (Sporos.Prim.putn.eff.1)) ls =
      .ok [] { ls with out := out' } := by
    simp [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim, out', X86Int.bytes8]
  rw [hexec]
  rcases hcall with ⟨outReg', ctx', stk', m3, s3, outCap', hrch3, hrip3, hm3, hmwf3, hbl3, hpos3,
      hr3, hor3, hk3, hsp3, hcap_le, hlen_le, hout_pg', hout3_mem, hout3_rw, hout3_hi, hout3_data,
      hctx3_mem, hctx3_b, hctx3_rw, hctx3_len, hc200', hc208', hc216', hctx3_oth,
      hstk3_mem, hstk3_b, hstk3_rw, hstk3_len, hout3_ne, hoth3, hreg3_oth⟩ | ⟨hhit, hends⟩
  · have hsp3' : s3.reg 4 = some vc.sp := by
      have hsb := vc.sp_bounds h
      simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hsp3
    have hs3_15 : s3.reg 15 = some ctxBase := by
      rw [hreg3_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hs2_15
    have hout'_len : out'.length = ls.out.length + 8 := by
      simp [out', bytes8_length]; omega
    have hout_len_eq' : outReg'.data.length = outCap' := by
      rw [hout3_data, writeBytes_length, List.length_append, List.length_replicate, hout_len]; omega
    have hout_rel' : OutRel outReg'.data out'.length out' := by
      rw [hout3_data, hout'_len]
      exact outRel_push_u64 (outRel_extend_zero (ext := outCap' - vc.outCap) hout_rel)
        (by rw [List.length_append, List.length_replicate, hout_len]; omega)
    have hout_ne_s : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ outReg'.base := by
      intro u hu hne
      by_cases hstku : u.base = vc.stk.base
      · rw [hstku, ← hstk2_b]
        exact hout3_ne _ hstk2 (by rw [hstk2_b]; exact hne_stk_out)
      · exact hout3_ne u (hoth2 u hu hstku) hne
    have hoth_s : ∀ u ∈ s.mem, u.base ≠ vc.outReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s3.mem := by
      intro u hu h1 h2 h3
      rw [hm3]
      exact hoth3 u (hoth2 u hu h3) h1 (by rw [← hctx_b.1]; exact h2) (by rw [hstk2_b]; exact h3)
    obtain ⟨hwf3, hpres3⟩ := wf_update_out h (by rw [hm3]; exact hmwf3) (by rw [hm3]; exact hbl3)
      (by rw [hm3]; exact hpos3) hr3 hsp3' hs3_15
      (by rw [hm3]; exact hout3_mem) hout3_rw hout3_hi (by omega) hout_len_eq' hout_pg'
      (by rw [hm3]; exact hctx3_mem) hctx3_b hctx3_rw hctx3_len
      hc200' (by rw [hout'_len]; exact hc208') hc216' hctx3_oth
      (by rw [hm3]; exact hstk3_mem) (by rw [hstk3_b, hstk2_b]) (by rw [hstk3_rw])
      (by rw [hstk3_len, hstk2_len]) hout_ne_s hoth_s hout_rel' (by rw [hout'_len]; exact hlen_le)
    let vc3 := { vc with stk := stk', ctxReg := ctx', outReg := outReg', outCap := outCap' }
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 14 hrip3 (by decide : (4198892, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk 14), exec_jmp]
      rfl
    let s4 := { s3 with rip := SporosVmBin.L_prim_epilogue }
    have hwf4 : vc3.wf pr { ls with out := out' } s4 :=
      vc3.wf_step_misc hwf3 rfl hr3 hsp3' hs3_15
    have hrch04 : Reaches vmImage s s4 := Reaches.trans hrch2 (Reaches.trans hrch3 (Reaches.step h04))
    have hor4 : OracleAdmits s4 := oracleAdmits_of_reaches hor hrch04
    have h280_4 : rd64 s4.mem (ea ctxBase 280) = some 1 := hpres3 280 1 (Or.inr (by decide)) (by decide) h280
    have h288_4 : rd64 s4.mem (ea ctxBase 288) = some 0 := hpres3 288 0 (Or.inr (by decide)) (by decide) h288
    have hepi := L_prim_epilogue_oracle_spec (vc := vc3) (pr := pr) (ls := { ls with out := out' }) (s := s4)
      (e_in := 1) (outs := []) (f := f) (rest := rest)
      rfl hwf4 hor4 hrs hpcW hle (by decide) (by intro _ hv; cases hv) h280_4 h288_4
      (by intro _ hi; cases hi)
    rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch04 hrchf, by rw [hkf]; exact hk3.trans (by rw [hk2]; rfl), horf,
        by simpa [Sporos.Prim.eff, out'] using hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch04 hhit, Ends.of_reaches hrch04 hends⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch2 hhit, Ends.of_reaches hrch2 hends⟩

end SporosVmPrimIo
