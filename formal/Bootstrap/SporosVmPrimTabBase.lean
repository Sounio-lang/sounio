import Bootstrap.SporosVmPrimIo

/-!
# Native x86-64 VM Table Primitive Infrastructure

Shared `ChainRel`, `TabsRel`, outcome-lifting, and `VmCtx` update combinators
used by the table primitive simulation proofs (`P_tnew`, `P_tput`, `P_tget`).

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimTab

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### Reusable `ChainRel`, `TabsRel`, outcome-lifting, and `VmCtx` combinators -/

theorem lift_next_outcome_out {pr : Sporos.Prog} {ls'' : SporosProvLen.LSt} {s₀ s₁ : X86Vm.St}
    (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k.out = s₀.k.out) :
    ((∃ s', Reaches vmImage s₁ s' ∧ s'.k.out = s₁.k.out ∧ OracleAdmits s' ∧ VmRel pr ls'' s') ∨
      (HitsNomem s₁ ∧ Ends vmImage s₁ (.exit 2))) →
    ((∃ s', Reaches vmImage s₀ s' ∧ s'.k.out = s₀.k.out ∧ OracleAdmits s' ∧ VmRel pr ls'' s') ∨
      (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2)))
  | Or.inl ⟨s', hr', hk', hor', hrel'⟩ =>
    Or.inl ⟨s', Reaches.trans hrch hr', hk'.trans hk, hor', hrel'⟩
  | Or.inr ⟨hhit, hends⟩ =>
    Or.inr ⟨hitsNomem_of_reaches hrch hhit, Ends.of_reaches hrch hends⟩

theorem chainRel_mono_len {kv : List Nat} {kvLen kvLen' idx : Nat} {l : List (Nat × SporosProvLen.LVal)}
    (hc : ChainRel kv kvLen idx l) (hle : kvLen ≤ kvLen') :
    ChainRel kv kvLen' idx l := by
  induction hc with
  | nil => exact .nil
  | @cons idx next k v rest hidx hlen hk hw hp hnx _ ih =>
    exact .cons (Nat.lt_of_lt_of_le hidx hle) hlen hk hw hp hnx ih

theorem tabsRel_extend_tabs_zero {tabsData kvData : List Nat} {kvLen ext : Nat}
    {tabs : List (List (Nat × SporosProvLen.LVal))}
    (ht : TabsRel tabsData kvData kvLen tabs) :
    TabsRel (tabsData ++ List.replicate ext 0) kvData kvLen tabs := by
  refine ⟨by rw [List.length_append, List.length_replicate]; exact Nat.le_trans ht.1 (Nat.le_add_right _ _), ?_⟩
  intro t hlt
  have hlen := ht.1
  have hslice : (((tabsData ++ List.replicate ext 0).drop (t * 8)).take 8) = (tabsData.drop (t * 8)).take 8 := by
    rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
  rw [hslice]
  exact ht.2 t hlt

theorem tabsRel_extend_kv_zero {tabsData kvData : List Nat} {kvLen ext : Nat}
    {tabs : List (List (Nat × SporosProvLen.LVal))}
    (ht : TabsRel tabsData kvData kvLen tabs) (hkv : kvLen * 32 ≤ kvData.length) :
    TabsRel tabsData (kvData ++ List.replicate ext 0) kvLen tabs :=
  ⟨ht.1, fun t hlt => chainRel_extend_zero (ht.2 t hlt) hkv⟩

theorem tabsRel_append_empty {tabsData kvData : List Nat} {kvLen : Nat}
    {tabs : List (List (Nat × SporosProvLen.LVal))}
    (ht : TabsRel tabsData kvData kvLen tabs) (hlen : (tabs.length + 1) * 8 ≤ tabsData.length) :
    TabsRel (writeBytes tabsData (tabs.length * 8) (bytes8 18446744073709551615)) kvData kvLen (tabs ++ [[]]) := by
  refine ⟨by simp [writeBytes_length]; omega, ?_⟩
  intro t hlt
  simp only [List.length_append, List.length_singleton] at hlt
  by_cases hlt0 : t < tabs.length
  · have hget : (tabs ++ [[]])[t]! = tabs[t]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD, List.getElem?_append_left hlt0]
    rw [hget, writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
    exact ht.2 t hlt0
  · have heq : t = tabs.length := by omega
    subst heq
    have hget : (tabs ++ [[]])[tabs.length]! = [] := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]; rfl
    have hval : leVal (((writeBytes tabsData (tabs.length * 8) (bytes8 18446744073709551615)).drop
        (tabs.length * 8)).take 8) = X86Env.W - 1 := by
      have hW : X86Env.W = 2 ^ 64 := rfl
      rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
      omega
    rw [hget, hval]
    exact .nil

theorem tabsRel_cons_entry {tabsData kvData : List Nat} {kvLen tb k vw vp oldHead : Nat}
    {tabs : List (List (Nat × SporosProvLen.LVal))}
    (ht : TabsRel tabsData kvData kvLen tabs)
    (htb : tb < tabs.length) (hkv_room : (kvLen + 1) * 32 ≤ kvData.length)
    (hkW : k < X86Env.W) (hvwW : vw < X86Env.W) (hvpW : vp < X86Env.W)
    (holdW : oldHead < X86Env.W) (hkvLenW : kvLen < X86Env.W)
    (hold : leVal ((tabsData.drop (tb * 8)).take 8) = oldHead) :
    let kv1 := writeBytes kvData (kvLen * 32) (bytes8 k)
    let kv2 := writeBytes kv1 (kvLen * 32 + 8) (bytes8 vw)
    let kv3 := writeBytes kv2 (kvLen * 32 + 16) (bytes8 vp)
    let kv4 := writeBytes kv3 (kvLen * 32 + 24) (bytes8 oldHead)
    let tabs' := writeBytes tabsData (tb * 8) (bytes8 kvLen)
    TabsRel tabs' kv4 (kvLen + 1) (tabs.set tb ((k, ⟨vw, vp⟩) :: tabs[tb]!)) := by
  intro kv1 kv2 kv3 kv4 tabs'
  have hlen1 : kv1.length = kvData.length := writeBytes_length _ _ _
  have hlen2 : kv2.length = kvData.length := by rw [writeBytes_length, hlen1]
  have hlen3 : kv3.length = kvData.length := by rw [writeBytes_length, hlen2]
  have hlen4 : kv4.length = kvData.length := by rw [writeBytes_length, hlen3]
  have hlift : ∀ {idx : Nat} {l : List (Nat × SporosProvLen.LVal)},
      ChainRel kvData kvLen idx l → ChainRel kv4 (kvLen + 1) idx l := by
    intro idx l hc
    exact chainRel_mono_len
      (chainRel_write_after (chainRel_write_after (chainRel_write_after
        (chainRel_write_after hc (Nat.le_refl _)) (by omega)) (by omega)) (by omega))
      (Nat.le_succ _)
  refine ⟨by simp [tabs', writeBytes_length]; exact ht.1, ?_⟩
  intro t hlt
  simp only [List.length_set] at hlt
  by_cases heq : t = tb
  · subst heq
    have hget : (tabs.set t ((k, ⟨vw, vp⟩) :: tabs[t]!))[t]! = (k, ⟨vw, vp⟩) :: tabs[t]! := by
      rw [List.getElem!_eq_getElem?_getD, mem_set_self_get _ t _ htb]; rfl
    have hhead : leVal ((tabs'.drop (t * 8)).take 8) = kvLen := by
      dsimp only [tabs']
      rw [leVal_writeBytes_same _ _ _ (by have := ht.1; omega), Nat.mod_eq_of_lt hkvLenW]
    have hq0 : leVal ((kv4.drop (kvLen * 32)).take 8) = k := by
      dsimp only [kv4, kv3, kv2, kv1]
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen3]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen2]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen1]; omega),
          leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt hkW]
    have hq1 : leVal ((kv4.drop (kvLen * 32 + 8)).take 8) = vw := by
      dsimp only [kv4, kv3, kv2]
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen3]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen2]; omega),
          leVal_writeBytes_same _ _ _ (by rw [hlen1]; omega), Nat.mod_eq_of_lt hvwW]
    have hq2 : leVal ((kv4.drop (kvLen * 32 + 16)).take 8) = vp := by
      dsimp only [kv4, kv3]
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [hlen3]; omega),
          leVal_writeBytes_same _ _ _ (by rw [hlen2]; omega), Nat.mod_eq_of_lt hvpW]
    have hq3 : leVal ((kv4.drop (kvLen * 32 + 24)).take 8) = oldHead := by
      dsimp only [kv4]
      rw [leVal_writeBytes_same _ _ _ (by rw [hlen3]; omega), Nat.mod_eq_of_lt holdW]
    have htail : ChainRel kv4 (kvLen + 1) oldHead tabs[t]! := by
      have h0 := ht.2 t htb
      rw [hold] at h0
      exact hlift h0
    rw [hget, hhead]
    exact .cons (Nat.lt_succ_self _) (by rw [hlen4]; omega) hq0 hq1 hq2 hq3 htail
  · have hget : (tabs.set tb ((k, ⟨vw, vp⟩) :: tabs[tb]!))[t]! = tabs[t]! := by
      simp only [List.getElem!_eq_getElem?_getD, mem_set_ne_get _ _ (Ne.symm heq)]
    have hhead : leVal ((tabs'.drop (t * 8)).take 8) = leVal ((tabsData.drop (t * 8)).take 8) := by
      dsimp only [tabs']
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by have := ht.1; omega)]
    rw [hget, hhead]
    exact hlift (ht.2 t hlt)

theorem wf_tabs_info {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    (vc.tabsReg.rw = true ∧ 0 < vc.tabsCap ∧ vc.tabsReg.data.length = vc.tabsCap * 8 ∧
      (vc.tabsCap * 8) % X86Env.page = 0 ∧ vc.tabsReg.hi ≤ 2 ^ 47) ∧
    (vc.kvReg.rw = true ∧ 0 < vc.kvCap ∧ vc.kvReg.data.length = vc.kvCap * 32 ∧
      (vc.kvCap * 32) % X86Env.page = 0 ∧ vc.kvReg.hi ≤ 2 ^ 47 ∧ vc.kvLen ≤ vc.kvCap) ∧
    (leVal ((vc.ctxReg.data.drop 128).take 8) = vc.tabsReg.base ∧
     leVal ((vc.ctxReg.data.drop 136).take 8) = ls.tabs.length ∧
     leVal ((vc.ctxReg.data.drop 144).take 8) = vc.tabsCap ∧
     leVal ((vc.ctxReg.data.drop 152).take 8) = vc.kvReg.base ∧
     leVal ((vc.ctxReg.data.drop 160).take 8) = vc.kvLen ∧
     leVal ((vc.ctxReg.data.drop 168).take 8) = vc.kvCap) ∧
    TabsRel vc.tabsReg.data vc.kvReg.data vc.kvLen ls.tabs ∧ ls.tabs.length ≤ vc.tabsCap := by
  rcases h with ⟨_, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, htabsrel, htabsle, _, _, _, _, _⟩
  exact ⟨⟨htb1, htb2, htb3, htb4, htb5⟩, ⟨hkv1, hkv2, hkv3, hkv4, hkv5, hkv6⟩,
    ⟨hsl16, hsl17, hsl18, hsl19, hsl20, hsl21⟩, htabsrel, htabsle⟩

theorem wf_update_tabs {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {tabsReg' ctx' stk' : Region} {tabsCap' : Nat}
    {tabs' : List (List (Nat × SporosProvLen.LVal))}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (htb_mem : tabsReg' ∈ s'.mem) (htb_rw : tabsReg'.rw = true) (htb_pos : 0 < tabsCap')
    (htb_len : tabsReg'.data.length = tabsCap' * 8) (htb_pg : (tabsCap' * 8) % X86Env.page = 0)
    (htb_hi : tabsReg'.hi ≤ 2 ^ 47)
    (hctx_mem : ctx' ∈ s'.mem) (hctx_base : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_len : ctx'.data.length = vc.ctxReg.data.length)
    (hc128 : leVal ((ctx'.data.drop 128).take 8) = tabsReg'.base)
    (hc136 : leVal ((ctx'.data.drop 136).take 8) = tabs'.length)
    (hc144 : leVal ((ctx'.data.drop 144).take 8) = tabsCap')
    (hctx_oth : ∀ j, (j < 128 ∨ 152 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]?)
    (hstk_mem : stk' ∈ s'.mem) (hstk_base : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_len : stk'.data.length = vc.stk.data.length)
    (htb_ne : ∀ u ∈ s.mem, u.base ≠ vc.tabsReg.base → u.base ≠ tabsReg'.base)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.tabsReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s'.mem)
    (htabs_rel : TabsRel tabsReg'.data vc.kvReg.data vc.kvLen tabs') (htabs_le : tabs'.length ≤ tabsCap')
    (hwlt_tabs : ∀ tb ∈ tabs', ∀ kv ∈ tb, kv.2.w < Sporos.M ∧ kv.2.plen < Sporos.M) :
    let vc' := { vc with stk := stk', ctxReg := ctx', tabsReg := tabsReg', tabsCap := tabsCap' }
    vc'.wf pr { ls with tabs := tabs' } s' ∧
    (∀ m0 u, (m0 + 8 ≤ 128 ∨ 152 ≤ m0) → m0 + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  have hcode_ne_tb : codeRegion.base ≠ vc.tabsReg.base :=
    vc.ne_bases h (i := 0) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_ctx : codeRegion.base ≠ vc.ctxReg.base :=
    vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_stk : codeRegion.base ≠ vc.stk.base :=
    vc.ne_bases h (i := 0) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1 hcode_ne_tb hcode_ne_ctx hcode_ne_stk, hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    _, _, _, _, _, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    _, _, _, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, _, _, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', tabsReg := tabsReg', tabsCap := tabsCap' }
  have hmem_all : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem
      | exact hctx_mem
      | exact htb_mem
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
       vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg]
    let sfx : List Region := [vc.kvReg, vc.inpReg, vc.outReg]
    have hnd_old : (pfx.map Region.base ++ vc.tabsReg.base :: sfx.map Region.base).Nodup := by
      simpa [pfx, sfx, VmCtx.regions] using hnodup
    have hgoal : (pfx.map Region.base ++ tabsReg'.base :: sfx.map Region.base).Nodup := by
      refine nodup_replace_at hnd_old ?_
      intro x hx
      rw [List.mem_append, List.mem_map, List.mem_map] at hx
      have hmem_ne : ∀ u, (u ∈ pfx ∨ u ∈ sfx) → u ∈ s.mem ∧ u.base ≠ vc.tabsReg.base := by
        intro u hu
        have hu_in : u ∈ vc.regions := by
          simp only [VmCtx.regions]
          rcases hu with hu | hu
          · simp only [pfx, List.mem_cons, List.not_mem_nil, or_false] at hu
            rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
          · simp only [sfx, List.mem_cons, List.not_mem_nil, or_false] at hu
            rcases hu with rfl | rfl | rfl <;> simp
        refine ⟨hmem u hu_in, ?_⟩
        rw [List.nodup_append] at hnd_old
        rcases hu with hu | hu
        · exact hnd_old.2.2 u.base (List.mem_map_of_mem hu) vc.tabsReg.base (List.mem_cons_self ..)
        · rw [List.nodup_cons] at hnd_old
          intro heq; exact hnd_old.2.1.1 (heq ▸ List.mem_map_of_mem hu)
      rcases hx with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · exact htb_ne u (hmem_ne u (Or.inl hu)).1 (hmem_ne u (Or.inl hu)).2
      · exact htb_ne u (hmem_ne u (Or.inr hu)).1 (hmem_ne u (Or.inr hu)).2
    simpa [vc', pfx, sfx, VmCtx.regions, hstk_base, hctx_base] using hgoal
  have hslot : ∀ m0, (m0 + 8 ≤ 128 ∨ 152 ≤ m0) →
      leVal ((ctx'.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
    intro m0 hm0
    congr 1
    exact drop_take_eq_of_getElem?_eq fun j hj1 hj2 => hctx_oth j (by omega)
  have hwf' : vc'.wf pr { ls with tabs := tabs' } s' :=
    ⟨hcm', hbl', hpos', hr', hsp', h15', hnodup', hmem_all,
      by rw [hstk_rw]; exact hstk1,
      by rw [hstk_base]; exact hstk2,
      by simpa only [vc', Region.hi, hstk_base, hstk_len] using hstk3, hstk4,
      by rw [hctx_base]; exact hctx1,
      by rw [hctx_rw]; exact hctx2,
      by rw [hctx_len]; exact hctx3,
      by simpa only [vc', Region.hi, hctx_base, hctx_len] using hctx4,
      hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb_rw, htb_pos, htb_len, htb_pg, htb_hi,
      hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
      hinp1, hout1, hout2, hout3, hout4, hout5,
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
      hc128, hc136, hc144,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl19,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl20,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl21,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl22,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl23,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl24,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl25,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl26,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl27,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl31,
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabs_rel, htabs_le, hinprel, houtrel, houtle,
      ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt_tabs, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0, hle_ctx m0 u hmlim hprev])

theorem wf_update_kv_tabs {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {tabsReg' kvReg' ctx' stk' : Region} {kvLen' kvCap' : Nat}
    {tabs' : List (List (Nat × SporosProvLen.LVal))}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (htb_mem : tabsReg' ∈ s'.mem) (htb_base : tabsReg'.base = vc.tabsReg.base)
    (htb_rw : tabsReg'.rw = true) (htb_len : tabsReg'.data.length = vc.tabsReg.data.length)
    (hkv_mem : kvReg' ∈ s'.mem) (hkv_rw : kvReg'.rw = true) (hkv_pos : 0 < kvCap')
    (hkv_len : kvReg'.data.length = kvCap' * 32) (hkv_pg : (kvCap' * 32) % X86Env.page = 0)
    (hkv_hi : kvReg'.hi ≤ 2 ^ 47) (hkv_le : kvLen' ≤ kvCap')
    (hctx_mem : ctx' ∈ s'.mem) (hctx_base : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_len : ctx'.data.length = vc.ctxReg.data.length)
    (hc152 : leVal ((ctx'.data.drop 152).take 8) = kvReg'.base)
    (hc160 : leVal ((ctx'.data.drop 160).take 8) = kvLen')
    (hc168 : leVal ((ctx'.data.drop 168).take 8) = kvCap')
    (hctx_oth : ∀ j, (j < 152 ∨ 176 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]?)
    (hstk_mem : stk' ∈ s'.mem) (hstk_base : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_len : stk'.data.length = vc.stk.data.length)
    (hkv_ne : ∀ u ∈ s.mem, u.base ≠ vc.kvReg.base → u.base ≠ kvReg'.base)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.kvReg.base → u.base ≠ vc.tabsReg.base →
      u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s'.mem)
    (htabs_len_eq : tabs'.length = ls.tabs.length)
    (htabs_rel : TabsRel tabsReg'.data kvReg'.data kvLen' tabs')
    (hwlt_tabs : ∀ tb ∈ tabs', ∀ kv ∈ tb, kv.2.w < Sporos.M ∧ kv.2.plen < Sporos.M) :
    let vc' := { vc with stk := stk', ctxReg := ctx', tabsReg := tabsReg', kvReg := kvReg',
                         kvLen := kvLen', kvCap := kvCap' }
    vc'.wf pr { ls with tabs := tabs' } s' ∧
    (∀ m0 u, (m0 + 8 ≤ 152 ∨ 176 ≤ m0) → m0 + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  have hcode_ne_kv : codeRegion.base ≠ vc.kvReg.base :=
    vc.ne_bases h (i := 0) (j := 13) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_tb : codeRegion.base ≠ vc.tabsReg.base :=
    vc.ne_bases h (i := 0) (j := 12) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_ctx : codeRegion.base ≠ vc.ctxReg.base :=
    vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcode_ne_stk : codeRegion.base ≠ vc.stk.base :=
    vc.ne_bases h (i := 0) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1 hcode_ne_kv hcode_ne_tb hcode_ne_ctx hcode_ne_stk, hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    _, htb2, htb3, htb4, htb5, _, _, _, _, _, _,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, _, _, _, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, _, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', tabsReg := tabsReg', kvReg := kvReg',
                               kvLen := kvLen', kvCap := kvCap' }
  have hmem_all : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem
      | exact hctx_mem
      | exact htb_mem
      | exact hkv_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    let pfx : List Region :=
      [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3, vc.dsReg, vc.auxReg,
       vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg, vc.tabsReg]
    let sfx : List Region := [vc.inpReg, vc.outReg]
    have hnd_old : (pfx.map Region.base ++ vc.kvReg.base :: sfx.map Region.base).Nodup := by
      simpa [pfx, sfx, VmCtx.regions] using hnodup
    have hgoal : (pfx.map Region.base ++ kvReg'.base :: sfx.map Region.base).Nodup := by
      refine nodup_replace_at hnd_old ?_
      intro x hx
      rw [List.mem_append, List.mem_map, List.mem_map] at hx
      have hmem_ne : ∀ u, (u ∈ pfx ∨ u ∈ sfx) → u ∈ s.mem ∧ u.base ≠ vc.kvReg.base := by
        intro u hu
        have hu_in : u ∈ vc.regions := by
          simp only [VmCtx.regions]
          rcases hu with hu | hu
          · simp only [pfx, List.mem_cons, List.not_mem_nil, or_false] at hu
            rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
          · simp only [sfx, List.mem_cons, List.not_mem_nil, or_false] at hu
            rcases hu with rfl | rfl <;> simp
        refine ⟨hmem u hu_in, ?_⟩
        rw [List.nodup_append] at hnd_old
        rcases hu with hu | hu
        · exact hnd_old.2.2 u.base (List.mem_map_of_mem hu) vc.kvReg.base (List.mem_cons_self ..)
        · rw [List.nodup_cons] at hnd_old
          intro heq; exact hnd_old.2.1.1 (heq ▸ List.mem_map_of_mem hu)
      rcases hx with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
      · exact hkv_ne u (hmem_ne u (Or.inl hu)).1 (hmem_ne u (Or.inl hu)).2
      · exact hkv_ne u (hmem_ne u (Or.inr hu)).1 (hmem_ne u (Or.inr hu)).2
    simpa [vc', pfx, sfx, VmCtx.regions, hstk_base, hctx_base, htb_base] using hgoal
  have hslot : ∀ m0, (m0 + 8 ≤ 152 ∨ 176 ≤ m0) →
      leVal ((ctx'.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
    intro m0 hm0
    congr 1
    exact drop_take_eq_of_getElem?_eq fun j hj1 hj2 => hctx_oth j (by omega)
  have hwf' : vc'.wf pr { ls with tabs := tabs' } s' :=
    ⟨hcm', hbl', hpos', hr', hsp', h15', hnodup', hmem_all,
      by rw [hstk_rw]; exact hstk1,
      by rw [hstk_base]; exact hstk2,
      by simpa only [vc', Region.hi, hstk_base, hstk_len] using hstk3, hstk4,
      by rw [hctx_base]; exact hctx1,
      by rw [hctx_rw]; exact hctx2,
      by rw [hctx_len]; exact hctx3,
      by simpa only [vc', Region.hi, hctx_base, hctx_len] using hctx4,
      hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb_rw, htb2, by rw [htb_len]; exact htb3, htb4,
      by simpa only [vc', Region.hi, htb_base, htb_len] using htb5,
      hkv_rw, hkv_pos, hkv_len, hkv_pg, hkv_hi, hkv_le,
      hinp1, hout1, hout2, hout3, hout4, hout5,
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
      by rw [hslot _ (Or.inl (by decide)), htb_base]; exact hsl16,
      by rw [hslot _ (Or.inl (by decide)), htabs_len_eq]; exact hsl17,
      by rw [hslot _ (Or.inl (by decide))]; exact hsl18,
      hc152, hc160, hc168,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl22,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl23,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl24,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl25,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl26,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl27,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide))]; exact hsl31,
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabs_rel, by rw [htabs_len_eq]; exact htabsle, hinprel, houtrel, houtle,
      ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt_tabs, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0, hle_ctx m0 u hmlim hprev])

end SporosVmPrimTab
