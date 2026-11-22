import Bootstrap.SporosVmPrimIo

/-!
# Native x86-64 VM Control-Flow Support & `O_call` Arena Growth

Shared control-flow outcome (`CtrlOutcome`), unified `VmCtx.wf` updater for
`rs`/`loc` arena and frame-stack transitions (`wf_update_rs_loc`), index-reg-8
3-way code-read voter (`tmr_ldIdx8_vote_eq_spec`), shared `mremap` doubling
step (`step_mremap_ctx_double`), and `O_call` return-stack growth
(`call_grow_rs_spec`).

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

/-- Post-step simulation outcome matching `SporosProvLen.lstep` across control
instructions (`O_jmp`, `O_jz`, `O_jnz`, `O_call`, `O_ret`). -/
def CtrlOutcome (pr : Sporos.Prog) (ls : SporosProvLen.LSt) (s : X86Vm.St) : Prop :=
  match SporosProvLen.lstep pr ls with
  | .next ls' =>
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2))
  | .halt c out =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit (c % 256) ∧
      s_exit.k.out = s.k.out ++ out.reverse
  | .fail _ =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k

theorem fetch_eq_some {pr : Sporos.Prog} {w pc : Nat} {ins : Sporos.Instr}
    (hnw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length)
    (hins : pr.words[w]!.code[pc]! = ins) : Sporos.fetch pr w pc = some ins := by
  unfold Sporos.fetch
  have hw : pr.words[w]? = some (pr.words[w]'hnw) := List.getElem?_eq_getElem hnw
  have hw' : pr.words[w]! = pr.words[w]'hnw := by rw [List.getElem!_eq_getElem?_getD, hw]; rfl
  rw [hw'] at hpc hins
  simp only [hw, bind, Option.bind, List.getElem?_eq_getElem hpc]
  simpa [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hpc] using hins

theorem kern_next_out (k : X86Env.Kern) : (k.next).2.out = k.out := by
  cases h : k.oracle <;> simp [X86Env.Kern.next, h]

theorem wf_rd_loc_slots {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    rd64 s.mem (ea ctxBase 72) = some vc.locReg.base ∧
    rd64 s.mem (ea ctxBase 80) = some (rsLocBase ls.rs) ∧
    rd64 s.mem (ea ctxBase 88) = some vc.locCap := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _,
    _, _, _, _, _, _, _, s72, s80, s88, _⟩ := h
  exact ⟨vc.rd64_ctx_of_eq h (by decide) s72,
    vc.rd64_ctx_of_eq h (by decide) s80,
    vc.rd64_ctx_of_eq h (by decide) s88⟩

/-- Unified `VmCtx.wf` preservation lemma for updates touching only `stk`,
`ctx[48..96)`, `rsReg`/`rsCap`, `locReg`/`locCap`, and `ls.rs`. -/
theorem wf_update_rs_loc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {stk' ctx' rsReg' locReg' : Region} {rsCap' locCap' : Nat}
    {rs' : List SporosProvLen.LFrame}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hstk_mem : stk' ∈ s'.mem) (hstk_b : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_l : stk'.data.length = vc.stk.data.length)
    (hctx_mem : ctx' ∈ s'.mem) (hctx_b : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_l : ctx'.data.length = vc.ctxReg.data.length)
    (hc48 : leVal ((ctx'.data.drop 48).take 8) = rsReg'.base)
    (hc56 : leVal ((ctx'.data.drop 56).take 8) = rs'.length)
    (hc64 : leVal ((ctx'.data.drop 64).take 8) = rsCap')
    (hc72 : leVal ((ctx'.data.drop 72).take 8) = locReg'.base)
    (hc80 : leVal ((ctx'.data.drop 80).take 8) = rsLocBase rs')
    (hc88 : leVal ((ctx'.data.drop 88).take 8) = locCap')
    (hctx_oth : ∀ off, (off + 8 ≤ 48 ∨ 96 ≤ off) → off + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8))
    (hrs_mem : rsReg' ∈ s'.mem) (hrs_rw : rsReg'.rw = true) (hrs_pos : 0 < rsCap')
    (hrs_l : rsReg'.data.length = rsCap' * 32) (hrs_pg : (rsCap' * 32) % X86Env.page = 0)
    (hrs_hi : rsReg'.hi ≤ 2 ^ 47)
    (hloc_mem : locReg' ∈ s'.mem) (hloc_rw : locReg'.rw = true) (hloc_pos : 0 < locCap')
    (hloc_l : locReg'.data.length = locCap' * 16) (hloc_pg : (locCap' * 16) % X86Env.page = 0)
    (hloc_hi : locReg'.hi ≤ 2 ^ 47)
    (hrel : RsLocRel rsReg'.data locReg'.data (rsLocBase rs') rs')
    (hrs_le : rs'.length ≤ rsCap') (hloc_le : rsLocBase rs' ≤ locCap')
    (hwlt_rs : ∀ f ∈ rs', ∀ v ∈ f.locs, v.w < X86Env.W ∧ v.plen < X86Env.W)
    (hnd' : (VmCtx.regions { vc with stk := stk', ctxReg := ctx', rsReg := rsReg', rsCap := rsCap', locReg := locReg', locCap := locCap' } |>.map Region.base).Nodup)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.stk.base → u.base ≠ vc.ctxReg.base →
      u.base ≠ vc.rsReg.base → u.base ≠ vc.locReg.base → u ∈ s'.mem) :
    let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', rsReg := rsReg', rsCap := rsCap', locReg := locReg', locCap := locCap' }
    vc'.wf pr { ls with rs := rs' } s' := by
  have hcode_ne : ∀ (j : Nat) (hj : j < vc.regions.length), 0 ≠ j → codeRegion.base ≠ vc.regions[j].base :=
    fun j hj hne => vc.ne_bases h (i := 0) (j := j) (by simp [VmCtx.regions]) hj hne
  have hcm' : CodeMem s'.mem :=
    ⟨hoth codeRegion h.1.1 (hcode_ne 1 (by simp [VmCtx.regions]) (by decide)) (hcode_ne 2 (by simp [VmCtx.regions]) (by decide))
      (hcode_ne 8 (by simp [VmCtx.regions]) (by decide)) (hcode_ne 9 (by simp [VmCtx.regions]) (by decide)), hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    _, _, _, _, _, _, _, _, _, _,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, _, _, _, _, _, _,
    hsl12, hsl13, hsl14, hsl15, hsl16, hsl17, hsl18, hsl19, hsl20, hsl21,
    hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, _, _, _,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', rsReg := rsReg', rsCap := rsCap', locReg := locReg', locCap := locCap' }
  have hmem' : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr; simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    have hnd0 := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd0
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem | exact hctx_mem | exact hrs_mem | exact hloc_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
          (by intro heq; rw [heq] at hnd0; simp [List.nodup_cons] at hnd0)
  have keep : ∀ {off val : Nat}, (off + 8 ≤ 48 ∨ 96 ≤ off) → off + 8 ≤ ctxSize →
      leVal ((vc.ctxReg.data.drop off).take 8) = val → leVal ((ctx'.data.drop off).take 8) = val :=
    fun h1 h2 hv => (hctx_oth _ h1 h2).trans hv
  refine ⟨hcm', hbl', hpos', hr', hsp', h15', hnd', hmem',
    hstk_rw ▸ hstk1, hstk_b ▸ hstk2, by simpa [Region.hi, hstk_b, hstk_l] using hstk3, hstk4,
    hctx_b ▸ hctx1, hctx_rw ▸ hctx2, hctx_l ▸ hctx3, by simpa [Region.hi, hctx_b, hctx_l] using hctx4,
    hf1, hf2, hf3, hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs_rw, hrs_pos, hrs_l, hrs_pg, hrs_hi, hloc_rw, hloc_pos, hloc_l, hloc_pg, hloc_hi,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    keep (Or.inl (by decide)) (by decide) hsl0, keep (Or.inl (by decide)) (by decide) hsl1,
    keep (Or.inl (by decide)) (by decide) hsl2, keep (Or.inl (by decide)) (by decide) hsl3,
    keep (Or.inl (by decide)) (by decide) hsl4, keep (Or.inl (by decide)) (by decide) hsl5,
    hc48, hc56, hc64, hc72, hc80, hc88,
    keep (Or.inr (by decide)) (by decide) hsl12, keep (Or.inr (by decide)) (by decide) hsl13,
    keep (Or.inr (by decide)) (by decide) hsl14, keep (Or.inr (by decide)) (by decide) hsl15,
    keep (Or.inr (by decide)) (by decide) hsl16, keep (Or.inr (by decide)) (by decide) hsl17,
    keep (Or.inr (by decide)) (by decide) hsl18, keep (Or.inr (by decide)) (by decide) hsl19,
    keep (Or.inr (by decide)) (by decide) hsl20, keep (Or.inr (by decide)) (by decide) hsl21,
    keep (Or.inr (by decide)) (by decide) hsl22, keep (Or.inr (by decide)) (by decide) hsl23,
    keep (Or.inr (by decide)) (by decide) hsl24, keep (Or.inr (by decide)) (by decide) hsl25,
    keep (Or.inr (by decide)) (by decide) hsl26, keep (Or.inr (by decide)) (by decide) hsl27,
    keep (Or.inr (by decide)) (by decide) hsl28, keep (Or.inr (by decide)) (by decide) hsl29,
    keep (Or.inr (by decide)) (by decide) hsl30, keep (Or.inr (by decide)) (by decide) hsl31,
    hpds, hdsle, hpaux, hauxle, hrel, hrs_le, hloc_le,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
    ⟨hwlt.1, hwlt.2.1, hwlt_rs, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩

/-- 3-replica indexed code read using index register `r8` (`O_call` word-meta loads,
parameterized by `ldLen` = 4 for disp 0 or 5 for disp 16) + inline voter. -/
theorem tmr_ldIdx8_vote_eq_spec {a done disp ldLen c1 c2 c3 r8v v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase) (hr8 : s.reg 8 = some r8v)
    (hm1 : (a, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 7, (.ldIdx .w64 0 2 8 1 disp), ldLen) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 7 + ldLen, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable)
    (hm4 : (a + 14 + ldLen, (.ldIdx .w64 3 2 8 1 disp), ldLen) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 14 + 2 * ldLen, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable)
    (hm6 : (a + 21 + 2 * ldLen, (.ldIdx .w64 1 2 8 1 disp), ldLen) ∈ SporosVmBin.vmTable)
    (hm7 : (a + 21 + 3 * ldLen, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable)
    (hm8 : (a + 24 + 3 * ldLen, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable)
    (hc1 : rd64 s.mem (ea ctxBase 232) = some c1) (hc2 : rd64 s.mem (ea ctxBase 240) = some c2)
    (hc3 : rd64 s.mem (ea ctxBase 248) = some c3)
    (hv1 : rd64 s.mem (eaIdx c1 r8v 1 disp) = some v) (hv2 : rd64 s.mem (eaIdx c2 r8v 1 disp) = some v)
    (hv3 : rd64 s.mem (eaIdx c3 r8v 1 disp) = some v) (hv : v < X86Env.W) :
    ∃ s', Reaches vmImage s s' ∧ s'.rip = done ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 3 → s'.reg r = s.reg r) := by
  let s1 := { s.set 2 (some c1) with rip := a + 7 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by rw [step_of_table (hrip ▸ hm1), exec_ld h15 hc1, hrip]
  have hr1 : RegsOk s1 := regsOk_set hr (r := 2) (by decide) c1
  have hs1_2 : s1.reg 2 = some c1 := reg_set_self_ok' hr (by decide) _
  have hs1_8 : s1.reg 8 = some r8v := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 8)]; exact hr8
  let s2 := { s1.set 0 (some v) with rip := a + 7 + ldLen }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_of_table hm2, exec_ldIdx hs1_2 hs1_8 (show rdMem .w64 s1.mem (eaIdx c1 r8v 1 disp) = some v from hv1)]
  have hr2 : RegsOk s2 := regsOk_set hr1 (r := 0) (by decide) v
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  have hs2_8 : s2.reg 8 = some r8v := by simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 8)]; exact hs1_8
  let s3 := { s2.set 2 (some c2) with rip := a + 7 + ldLen + 7 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by rw [step_of_table hm3, exec_ld hs2_15 hc2]
  have hr3 : RegsOk s3 := regsOk_set hr2 (r := 2) (by decide) c2
  have hs3_2 : s3.reg 2 = some c2 := reg_set_self_ok' hr2 (by decide) _
  have hs3_8 : s3.reg 8 = some r8v := by simp only [s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 8)]; exact hs2_8
  have hs3_rip : s3.rip = a + 14 + ldLen := by dsimp only [s3]; omega
  let s4 := { s3.set 3 (some v) with rip := a + 14 + ldLen + ldLen }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by
    rw [step_of_table (hs3_rip ▸ hm4), exec_ldIdx hs3_2 hs3_8 (show rdMem .w64 s3.mem (eaIdx c2 r8v 1 disp) = some v from hv2), hs3_rip]
  have hr4 : RegsOk s4 := regsOk_set hr3 (r := 3) (by decide) v
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs2_15
  have hs4_8 : s4.reg 8 = some r8v := by simp only [s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 8)]; exact hs3_8
  have hs4_rip : s4.rip = a + 14 + 2 * ldLen := by dsimp only [s4]; omega
  let s5 := { s4.set 2 (some c3) with rip := a + 14 + 2 * ldLen + 7 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by rw [step_of_table (hs4_rip ▸ hm5), exec_ld hs4_15 hc3, hs4_rip]
  have hr5 : RegsOk s5 := regsOk_set hr4 (r := 2) (by decide) c3
  have hs5_2 : s5.reg 2 = some c3 := reg_set_self_ok' hr4 (by decide) _
  have hs5_8 : s5.reg 8 = some r8v := by simp only [s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 8)]; exact hs4_8
  have hs5_rip : s5.rip = a + 21 + 2 * ldLen := by dsimp only [s5]; omega
  let s6 := { s5.set 1 (some v) with rip := a + 21 + 2 * ldLen + ldLen }
  have h06 : X86Vm.step vmImage s5 = .ok s6 := by
    rw [step_of_table (hs5_rip ▸ hm6), exec_ldIdx hs5_2 hs5_8 (show rdMem .w64 s5.mem (eaIdx c3 r8v 1 disp) = some v from hv3), hs5_rip]
  have hr6 : RegsOk s6 := regsOk_set hr5 (r := 1) (by decide) v
  have hs6_0 : s6.reg 0 = some v := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0),
      List.getElem?_set_ne (by decide : 2 ≠ 0), List.getElem?_set_ne (by decide : 3 ≠ 0)]
    exact reg_set_self_ok' hr1 (by decide) _
  have hs6_3 : s6.reg 3 = some v := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 3),
      List.getElem?_set_ne (by decide : 2 ≠ 3)]
    exact reg_set_self_ok' hr3 (by decide) _
  have hs6_1 : s6.reg 1 = some v := reg_set_self_ok' hr5 (by decide) _
  have hs6_rip : s6.rip = a + 21 + 3 * ldLen := by dsimp only [s6]; omega
  have hvote := tmr_inline_vote_eq_spec hs6_rip hm7
    (by simpa [show a + 21 + 3 * ldLen + 3 = a + 24 + 3 * ldLen by omega] using hm8) hs6_0 hs6_3 hv
  refine ⟨_, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
    (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 hvote))))),
    rfl, rfl, rfl, hr6, hs6_0, hs6_1, hs6_3, fun r h0r h1r h2r h3r => ?_⟩
  simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne (Ne.symm h1r), List.getElem?_set_ne (Ne.symm h2r),
    List.getElem?_set_ne (Ne.symm h3r), List.getElem?_set_ne (Ne.symm h0r)]

/-- Shared 10-instruction arena-doubling `mremap` + `ctx` update sequence used
by both `call_grow_rs` (`offB=48, offC=64, sh=5`) and `call_locs_chk`
(`offB=72, offC=88, sh=4`). -/
theorem step_mremap_ctx_double {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a offB offC sh cap : Nat} {reg : Region}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hr1 : s.reg 1 = some cap) (haW : a + 23 < X86Env.W)
    (hoffB : offB + 8 ≤ offC) (hoffC : offC + 8 ≤ ctxSize) (hsh : sh < 64)
    (hreg_in : reg ∈ s.mem) (hreg_rw : reg.rw = true) (hcap_pos : 0 < cap)
    (hreg_len : reg.data.length = cap * 2 ^ sh) (hreg_pg : (cap * 2 ^ sh) % X86Env.page = 0)
    (hreg_hi : reg.hi ≤ 2 ^ 47) (hstk_ne : vc.stk.base ≠ reg.base) (hctx_ne : vc.ctxReg.base ≠ reg.base)
    (hrdB : rd64 s.mem (ea ctxBase offB) = some reg.base)
    (hrdC : rd64 s.mem (ea ctxBase offC) = some cap)
    (hm1 : (a, (.ofInt (.ld 7 15 offB)), 4) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 4, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 7, (.ofInt (.shlI 6 sh)), 4) ∈ SporosVmBin.vmTable)
    (hm4 : (a + 11, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 14, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable)
    (hm6 : (a + 18, (.ofInt (.call SporosVmBin.fn_mremap)), 5) ∈ SporosVmBin.vmTable)
    (hm7 : (a + 23, (.ofInt (.st 15 offB 0)), 4) ∈ SporosVmBin.vmTable)
    (hm8 : (a + 27, (.ofInt (.ld 1 15 offC)), 4) ∈ SporosVmBin.vmTable)
    (hm9 : (a + 31, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable)
    (hm10 : (a + 35, (.ofInt (.st 15 offC 1)), 4) ∈ SporosVmBin.vmTable) :
    (∃ (newBase : Nat) (stk1 ctx2 reg' : Region) (s' : X86Vm.St),
      Reaches vmImage s s' ∧ s'.rip = a + 39 ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧
      s'.mem.wf ∧ BytesLt s'.mem ∧ (∀ u ∈ s'.mem, 0 < u.data.length) ∧
      RegsOk s' ∧ s'.reg 4 = some vc.sp ∧ s'.reg 15 = some ctxBase ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r) ∧
      stk1 ∈ s'.mem ∧ stk1.base = vc.stk.base ∧ stk1.rw = vc.stk.rw ∧ stk1.data.length = vc.stk.data.length ∧
      ctx2 = { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data offB (bytes8 newBase)) offC (bytes8 (2 * cap)) } ∧
      ctx2 ∈ s'.mem ∧
      reg' = ⟨newBase, reg.data ++ List.replicate (cap * 2 ^ sh) 0, true⟩ ∧
      reg' ∈ s'.mem ∧ newBase + 2 * (cap * 2 ^ sh) ≤ 2 ^ 47 ∧
      (∀ u ∈ s.mem, u.base ≠ reg.base → u.base ≠ newBase) ∧
      (∀ u ∈ s.mem, u.base ≠ vc.stk.base → u.base ≠ vc.ctxReg.base → u.base ≠ reg.base → u ∈ s'.mem)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hhi' : reg.base + reg.data.length ≤ 2 ^ 47 := hreg_hi
  have hpow_pos : 0 < 2 ^ sh := Nat.two_pow_pos sh
  let s1 := { s.set 7 (some reg.base) with rip := a + 4 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by rw [step_of_table (hrip ▸ hm1), exec_ld h15 hrdB, hrip]
  have hr1' := regsOk_set hr (r := 7) (by decide) reg.base
  have hs1_1 : s1.reg 1 = some cap := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]; exact hr1
  let s2 := { s1.set 6 (some cap) with rip := a + 7 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by rw [step_of_table hm2, exec_mov hs1_1]
  have hr2 := regsOk_set hr1' (r := 6) (by decide) cap
  have hs2_6 : s2.reg 6 = some cap := reg_set_self_ok' hr1' (by decide) _
  have hsh_mod : sh % 64 = sh := Nat.mod_eq_of_lt hsh
  let s3 := { s2.set 6 (some (cap * 2 ^ sh)) with fl := X86Int.Flags.undef, rip := a + 11 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_of_table hm3, exec_shlI_lt hs2_6 (by rw [hsh_mod]; omega)]; simp only [s3, s2, hsh_mod]
  have hr3 := regsOk_set hr2 (r := 6) (by decide) (cap * 2 ^ sh)
  have hs3_6 : s3.reg 6 = some (cap * 2 ^ sh) := reg_set_self_ok' hr2 (by decide) _
  let s4 := { s3.set 2 (some (cap * 2 ^ sh)) with rip := a + 14 }
  have h04 : X86Vm.step vmImage s3 = .ok s4 := by rw [step_of_table hm4, exec_mov hs3_6]
  have hr4 := regsOk_set hr3 (r := 2) (by decide) (cap * 2 ^ sh)
  have hs4_2 : s4.reg 2 = some (cap * 2 ^ sh) := reg_set_self_ok' hr3 (by decide) _
  let s5 := { s4.set 2 (some (2 * (cap * 2 ^ sh))) with fl := X86Int.Flags.undef, rip := a + 18 }
  have h05 : X86Vm.step vmImage s4 = .ok s5 := by
    rw [step_of_table hm5, exec_shlI_lt hs4_2 (by rw [show 2 ^ (1 % 64) = 2 by decide]; omega)]
    simp only [s5, s4, show 2 ^ (1 % 64) = 2 by decide, Nat.mul_comm (cap * 2 ^ sh) 2]
  have hr5 := regsOk_set hr4 (r := 2) (by decide) (2 * (cap * 2 ^ sh))
  have hreach5 : Reaches vmImage s s5 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step h05))))
  have hsp5 : s5.reg 4 = some vc.sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 6 ≠ 4), List.getElem?_set_ne (by decide : 7 ≠ 4)]; exact hsp
  have h15_5 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 7 ≠ 15)]; exact h15
  have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc h rfl hr5 hsp5 h15_5
  obtain ⟨sc, hcall, hcrip, hck, _, hccm, hcbl, hcpos, hcr, hcsp, hc15, hcoth, hstk1, hstkb, hstkrw, hstkl,
    _, hspW, _, hstkhi, hret, hoth_stk⟩ :=
    step_call_stk (t := SporosVmBin.fn_mremap) (step_of_table hm6) hwf5 (by dsimp [s5]; omega)
  have hrc : Reaches vmImage s sc := Reaches.trans hreach5 (Reaches.step hcall)
  have hreg_sc : reg ∈ sc.mem := hoth_stk reg hreg_in hstk_ne.symm
  obtain ⟨idx, hfind⟩ := mem_find_of_mem sc.mem reg reg.base 0 hccm.2.2 hreg_sc (Nat.le_refl _)
    (by simp [Region.hi, hreg_len]; exact Nat.mul_pos hcap_pos hpow_pos)
  simp only [Nat.zero_add] at hfind
  have hpg1 : X86Env.pages (cap * 2 ^ sh) = cap * 2 ^ sh := pages_of_dvd hreg_pg
  have hpg2 : X86Env.pages (2 * (cap * 2 ^ sh)) = 2 * (cap * 2 ^ sh) := pages_of_dvd (by rw [Nat.mul_mod, hreg_pg]; decide)
  have hsc7 : sc.reg 7 = some reg.base := by
    rw [hcoth 7 (by decide)]; simp only [s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 7), List.getElem?_set_ne (by decide : 6 ≠ 7)]; exact reg_set_self_ok' hr (by decide) _
  have hsc6 : sc.reg 6 = some (cap * 2 ^ sh) := by
    rw [hcoth 6 (by decide)]; simp only [s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs3_6
  have hsc2 : sc.reg 2 = some (2 * (cap * 2 ^ sh)) := by rw [hcoth 2 (by decide)]; exact reg_set_self_ok' hr4 (by decide) _
  rcases call_mremap_oracle_spec hcrip hcr (oracleAdmits_of_reaches hor hrc) hcsp (by omega) hspW hccm.2
    hstk1 hstkrw (by rw [hstkb]; exact hstk_ne) (by omega) hstkhi hret hsc7 hsc6 hsc2 hfind rfl
    (by rw [hpg1]; exact hreg_len) hreg_rw (by rw [hpg2, hreg_len]; omega) with
    ⟨newBase, sm, hrm, _, _, _, halim, hfree, hmrip, hmk, hmmem, hmwf, hmr, hm0, hm4, _, hmoth⟩ | ⟨hhit, hend⟩
  · have hget_idx : sc.mem[idx]? = some reg := (mem_find_idx_lt sc.mem reg.base 0 idx reg hfind).2.2
    have hdisj := mremap_find_after_stack_write h.1.2 (vc.mem_stk h) hstkb (by simp [Region.hi, hstkl])
      hstk_ne hstk1 hoth_stk hget_idx rfl hfree (by rw [hpg2]; exact Nat.mul_pos (by decide) (Nat.mul_pos hcap_pos hpow_pos)) h.2.2.1
    have hbl_m : BytesLt sm.mem := by
      rw [hmmem]; intro u hu; rcases List.mem_append.mp hu with hu | hu
      · exact hcbl u (List.mem_of_mem_eraseIdx hu)
      · simp only [List.mem_singleton] at hu; subst hu; intro b hb
        rcases List.mem_append.mp hb with hb | hb
        · exact h.2.1 reg hreg_in b hb
        · exact bytesLt_replicate_zero _ b hb
    have hpos_m : ∀ u ∈ sm.mem, 0 < u.data.length := by
      rw [hmmem]; intro u hu; rcases List.mem_append.mp hu with hu | hu
      · exact hcpos u (List.mem_of_mem_eraseIdx hu)
      · simp only [List.mem_singleton] at hu; subst hu; simp only [List.length_append, hreg_len]
        exact Nat.lt_of_lt_of_le (Nat.mul_pos hcap_pos hpow_pos) (Nat.le_add_right _ _)
    have hctx_ne_stk : vc.ctxReg.base ≠ vc.stk.base := vc.ne_bases h (i := 2) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hctx_in_m : vc.ctxReg ∈ sm.mem := by
      rw [hmmem]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base sc.mem idx hget_idx (hoth_stk _ (vc.mem_ctx hwf5) hctx_ne_stk) hctx_ne)
    have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hoffC hcl
    have hm15 : sm.reg 15 = some ctxBase := by
      rw [hmoth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hc15
    have heaB : ea ctxBase offB = ctxBase + offB := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
    have heaC : ea ctxBase offC = ctxBase + offC := ea_add_eq (by have : ctxBase = 0x600000 := rfl; omega)
    obtain ⟨_, m1, hwr1, _, _, hm1wf, hctx1_m, hoth1⟩ := wr64_region hmwf hctx_in_m hcb.2.1 (a := ctxBase + offB) (v := newBase)
      (by rw [hcb.1]; omega) (by simp only [Region.hi, hcb.1]; omega)
    let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data offB (bytes8 newBase) }
    have hctx1_m' : ctx1 ∈ m1 := by simpa [ctx1, hcb.1] using hctx1_m
    have hmrip' : sm.rip = a + 23 := by simpa [s5] using hmrip
    let s6 := { sm with mem := m1, rip := a + 27 }
    have h06 : X86Vm.step vmImage sm = .ok s6 := by
      rw [step_of_table (hmrip' ▸ hm7), exec_st hm15 hm0 (by rw [heaB]; exact hwr1), hmrip']
    have hrdC1 : rd64 m1 (ea ctxBase offC) = some cap := by
      rw [heaC]; refine rd64_of_mem hm1wf hctx1_m' (by rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega) ?_
      have hle := ctx_leVal_of_rd64 h (by dsimp [ctxSize]; omega) hrdC
      dsimp only [ctx1]; rw [hcb.1, Nat.add_sub_cancel_left]
      rw [writeBytes_drop_take_disjoint vc.ctxReg.data offB offC 8 (bytes8 newBase) (by rw [bytes8_length]; exact Or.inr hoffB) (by omega)]; exact hle
    let s7 := { s6.set 1 (some cap) with rip := a + 31 }
    have h07 : X86Vm.step vmImage s6 = .ok s7 := by rw [step_of_table hm8, exec_ld (s := s6) hm15 hrdC1]
    have hr7 := regsOk_set hmr (r := 1) (by decide) cap
    have hs7_1 : s7.reg 1 = some cap := reg_set_self_ok' hmr (by decide) _
    have hcapW : cap < 2 ^ 47 := by have : cap ≤ cap * 2 ^ sh := Nat.le_mul_of_pos_right cap hpow_pos; omega
    let s8 := { s7.set 1 (some (2 * cap)) with fl := X86Int.Flags.undef, rip := a + 35 }
    have h08 : X86Vm.step vmImage s7 = .ok s8 := by
      rw [step_of_table hm9, exec_shlI_lt hs7_1 (by rw [show 2 ^ (1 % 64) = 2 by decide]; omega)]
      simp only [s8, s7, show 2 ^ (1 % 64) = 2 by decide, Nat.mul_comm cap 2]
    have hr8 := regsOk_set hr7 (r := 1) (by decide) (2 * cap)
    have hs8_1 : s8.reg 1 = some (2 * cap) := reg_set_self_ok' hr7 (by decide) _
    have hs8_15 : s8.reg 15 = some ctxBase := by simp only [s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hm15
    obtain ⟨_, m2, hwr2, _, _, hm2wf, hctx2_m, hoth2⟩ := wr64_region hm1wf hctx1_m' hcb.2.1 (a := ctxBase + offC) (v := 2 * cap)
      (by dsimp [ctx1]; rw [hcb.1]; omega) (by simp only [Region.hi, ctx1, hcb.1, writeBytes_length]; omega)
    let ctx2 := { vc.ctxReg with data := writeBytes ctx1.data offC (bytes8 (2 * cap)) }
    have hctx2_m' : ctx2 ∈ m2 := by simpa [ctx2, ctx1, hcb.1] using hctx2_m
    let s9 := { s8 with mem := m2, rip := a + 39 }
    have h09 : X86Vm.step vmImage s8 = .ok s9 := by rw [step_of_table hm10, exec_st hs8_15 hs8_1 (by rw [heaC]; exact hwr2)]
    have hreach9 : Reaches vmImage s s9 := Reaches.trans hrc (Reaches.trans hrm
      (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09)))))
    have hne_all : ∀ u ∈ s.mem, u.base ≠ reg.base → u.base ≠ newBase := fun u hu hne =>
      (mem_free_ne_base (by rw [hpg2]; exact Nat.mul_pos (by decide) (Nat.mul_pos hcap_pos hpow_pos))
        (h.2.2.1 u hu) (hdisj.2 u hu hne)).symm
    let reg' : Region := ⟨newBase, reg.data ++ List.replicate (cap * 2 ^ sh) 0, true⟩
    have hreg'_eq : (⟨newBase, reg.data ++ List.replicate (X86Env.pages (2 * (cap * 2 ^ sh)) - reg.data.length) 0, true⟩ : Region) = reg' := by
      simp only [reg', hpg2, hreg_len, show 2 * (cap * 2 ^ sh) - cap * 2 ^ sh = cap * 2 ^ sh by omega]
    have hnb_ne_ctx : newBase ≠ vc.ctxReg.base := (hne_all vc.ctxReg (vc.mem_ctx h) hctx_ne).symm
    let stk1 := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (s5.rip + 5)) }
    refine Or.inl ⟨newBase, stk1, ctx2, reg', s9, hreach9, rfl,
      by change sm.k.out = s.k.out; rw [hmk, kern_next_out, hck]; rfl,
      oracleAdmits_of_reaches hor hreach9, hm2wf,
      bytesLt_wr64 (bytesLt_wr64 hbl_m hwr1) hwr2, len_pos_wrN (len_pos_wrN hpos_m hwr1) hwr2,
      hr8, ?_, hs8_15, ?_, ?_, hstkb, rfl, hstkl, rfl, hctx2_m', rfl, ?_, by rw [← hpg2]; exact halim, hne_all, ?_⟩
    · have hspb := vc.sp_bounds h
      have hsp_eq : (vc.sp - 8 + 8) % X86Env.W = vc.sp := by rw [Nat.sub_add_cancel (by omega), Nat.mod_eq_of_lt hspb.2.2.1]
      change ((s6.set 1 _).set 1 _).reg 4 = some vc.sp
      rw [reg_set_ne (by decide : (1 : Nat) ≠ 4), reg_set_ne (by decide : (1 : Nat) ≠ 4)]
      change sm.reg 4 = some vc.sp; rw [hm4, hsp_eq]
    · intro r h0r h1r h2r h4r h6r h7r h8r h9r h10r h11r
      change ((s6.set 1 _).set 1 _).reg r = s.reg r
      rw [reg_set_ne (Ne.symm h1r), reg_set_ne (Ne.symm h1r)]
      change sm.reg r = s.reg r
      rw [hmoth r h0r h1r h2r h4r h6r h7r h8r h9r h10r h11r, hcoth r h4r]
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (Ne.symm h2r), List.getElem?_set_ne (Ne.symm h6r), List.getElem?_set_ne (Ne.symm h7r)]
    · exact hoth2 stk1 (hoth1 stk1 (by rw [hmmem]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base sc.mem idx hget_idx hstk1 (by rw [hstkb]; exact hstk_ne))) (by rw [hstkb]; exact hctx_ne_stk.symm)) (by rw [hstkb]; simpa [ctx1] using hctx_ne_stk.symm)
    · exact hoth2 reg' (hoth1 reg' (by rw [hmmem, hreg'_eq]; simp) hnb_ne_ctx) (by simpa [ctx1] using hnb_ne_ctx)
    · intro u hu h1 h2 h3
      exact hoth2 u (hoth1 u (by rw [hmmem]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base sc.mem idx hget_idx (hoth_stk u hu h1) h3)) h2) (by simpa [ctx1] using h2)
  · exact Or.inr ⟨hitsNomem_of_reaches hrc hhit, Ends.of_reaches hrc hend⟩

/-- `O_call` return-stack growth check (`4200657..4200709`): ensures
`ls.rs.length < vc'.rsCap`, doubling `rsReg` via `fn_mremap` if full. -/
theorem call_grow_rs_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w nLocs : Nat}
    (hrip : s.rip = 4200657) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some nLocs) :
    (∃ (vc' : VmCtx) (s' : X86Vm.St), Reaches vmImage s s' ∧ s'.rip = 4200709 ∧ s'.k.out = s.k.out ∧
      OracleAdmits s' ∧ vc'.wf pr ls s' ∧ ls.rs.length < vc'.rsCap ∧
      s'.reg 12 = some w ∧ s'.reg 13 = some nLocs) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hr := h.2.2.2.1; have hsp := h.2.2.2.2.1; have h15 := h.2.2.2.2.2.1
  have ⟨hrsRw, hrsPos, hrsLenEq, hrsPg, hrsHi, hlocRw, hlocPos, hlocLenEq, hlocPg, hlocHi, hrsRel, hrsLe, hlocLe⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, hrsCap, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  have ⟨s72, s80, s88⟩ := wf_rd_loc_slots h
  have hW : X86Env.W = 2 ^ 64 := rfl
  let s1 := { s.set 0 (some ls.rs.length) with rip := 4200661 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_vc 20 hrip (by decide : (4200657, (.ofInt (.ld 0 15 56)), 4) ∈ vmTableChunk 20), exec_ld h15 hrsLen, hrip]
  have hr1 := regsOk_set hr (r := 0) (by decide) ls.rs.length
  have hs1_15 : s1.reg 15 = some ctxBase := by simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  let s2 := { s1.set 1 (some vc.rsCap) with rip := 4200665 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_vc 20 rfl (by decide : (4200661, (.ofInt (.ld 1 15 64)), 4) ∈ vmTableChunk 20), exec_ld hs1_15 hrsCap]
  have hr2 := regsOk_set hr1 (r := 1) (by decide) vc.rsCap
  have hs2_0 : s2.reg 0 = some ls.rs.length := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact reg_set_self_ok' hr (by decide) _
  have hs2_1 : s2.reg 1 = some vc.rsCap := reg_set_self_ok' hr1 (by decide) _
  let s3 := { s2 with fl := (X86Int.aluF .sub ls.rs.length vc.rsCap).2, rip := 4200668 }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_vc 20 rfl (by decide : (4200665, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 20), exec_cmp hs2_0 hs2_1]
  have hreach3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hsp3 : s3.reg 4 = some vc.sp := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp
  have h15_3 : s3.reg 15 = some ctxBase := by simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs1_15
  have hr12_3 : s3.reg 12 = some w := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hr12
  have hr13_3 : s3.reg 13 = some nLocs := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hr13
  by_cases hlt : ls.rs.length < vc.rsCap
  · let s4 := { s3 with rip := 4200709 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 20 rfl (by decide : (4200668, (.ofInt (.jcc 2 4200709)), 2) ∈ vmTableChunk 20), exec_jcc (cond_aluF_sub_b ls.rs.length vc.rsCap)]; simp [hlt, s3, s4]
    have hreach4 := Reaches.trans hreach3 (Reaches.step h04)
    exact Or.inl ⟨vc, s4, hreach4, rfl, rfl, oracleAdmits_of_reaches hor hreach4, vc.wf_step_misc h rfl hr2 hsp3 h15_3, hlt, hr12_3, hr13_3⟩
  · let s4 := { s3 with rip := 4200670 }
    have h04 : X86Vm.step vmImage s3 = .ok s4 := by
      rw [step_at_vc 20 rfl (by decide : (4200668, (.ofInt (.jcc 2 4200709)), 2) ∈ vmTableChunk 20), exec_jcc (cond_aluF_sub_b ls.rs.length vc.rsCap)]; simp [hlt, s3, s4]
    have hreach4 := Reaches.trans hreach3 (Reaches.step h04)
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr2 hsp3 h15_3
    rcases step_mremap_ctx_double (a := 4200670) (offB := 48) (offC := 64) (sh := 5) (cap := vc.rsCap) (reg := vc.rsReg)
      rfl hwf4 (oracleAdmits_of_reaches hor hreach4) hs2_1 (by omega) (by decide) (by decide) (by decide)
      (vc.mem_rs hwf4) hrsRw hrsPos hrsLenEq hrsPg hrsHi
      (vc.ne_bases h (i := 1) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
      (vc.ne_bases h (i := 2) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
      hrsBase hrsCap
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 20 rfl (by decide))
      (mem_vmTable_of_chunk 20 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
      (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide))
      (mem_vmTable_of_chunk 21 rfl (by decide)) (mem_vmTable_of_chunk 21 rfl (by decide)) with
      ⟨nb, stk1, ctx2, rsReg', s', hrm, hrip', hk', hor', hmwf', hbl', hpos', hr', hsp', h15', hroth,
        hstk_m, hstkb, hstkrw, hstkl, hctx2_eq, hctx2_m, hrs'_eq, hrs'_m, hnb_hi, hnb_ne, hoth⟩ | ⟨hhit, hend⟩
    · have hreach' := Reaches.trans hreach4 hrm
      have hcb := vc.ctx_bounds h; have hcl := hcb.2.2.1; dsimp [ctxSize] at hcl
      have hrsHi' : vc.rsReg.base + vc.rsReg.data.length ≤ 2 ^ 47 := hrsHi
      have hloc_m : vc.locReg ∈ s'.mem := hoth vc.locReg (vc.mem_loc hwf4)
        (vc.ne_bases h (i := 9) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (vc.ne_bases h (i := 9) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
        (vc.ne_bases h (i := 9) (j := 8) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
      let vc' : VmCtx := { vc with stk := stk1, ctxReg := ctx2, rsReg := rsReg', rsCap := 2 * vc.rsCap }
      have hnd' : (vc'.regions.map Region.base).Nodup := by
        let pfx : List Region := [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3, vc.dsReg, vc.auxReg]
        let sfx : List Region := [vc.locReg, vc.memwReg, vc.mempReg, vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]
        have hnd0 : (pfx.map Region.base ++ vc.rsReg.base :: sfx.map Region.base).Nodup := by simpa [pfx, sfx, VmCtx.regions] using h.2.2.2.2.2.2.1
        have hgoal := nodup_replace_at (b := nb) hnd0 (fun x hx => by
          rw [List.mem_append, List.mem_map, List.mem_map] at hx
          rcases hx with ⟨u, hu, rfl⟩ | ⟨u, hu, rfl⟩
          · have hu_in : u ∈ vc.regions := by simp only [pfx, List.mem_cons, List.not_mem_nil, or_false] at hu; rcases hu with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> simp [VmCtx.regions]
            exact hnb_ne u (hwf4.2.2.2.2.2.2.2.1 u hu_in) (by rw [List.nodup_append] at hnd0; exact hnd0.2.2 u.base (List.mem_map_of_mem hu) vc.rsReg.base (List.mem_cons_self ..))
          · have hu_in : u ∈ vc.regions := by simp only [sfx, List.mem_cons, List.not_mem_nil, or_false] at hu; rcases hu with rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> simp [VmCtx.regions]
            exact hnb_ne u (hwf4.2.2.2.2.2.2.2.1 u hu_in) (by rw [List.nodup_append, List.nodup_cons] at hnd0; intro heq; exact hnd0.2.1.1 (heq ▸ List.mem_map_of_mem hu)))
        simpa [vc', pfx, sfx, VmCtx.regions, hstkb, hctx2_eq, hrs'_eq] using hgoal
      have hdisj : ∀ off, (off + 8 ≤ 48 ∨ (56 ≤ off ∧ off + 8 ≤ 64) ∨ 72 ≤ off) → off + 8 ≤ ctxSize →
          leVal ((ctx2.data.drop off).take 8) = leVal ((vc.ctxReg.data.drop off).take 8) := fun off ho hl => by
        rw [hctx2_eq]; dsimp [ctxSize] at hl ⊢
        have hc1 : off + 8 ≤ 64 ∨ 64 + (bytes8 (2 * vc.rsCap)).length ≤ off := by
          rw [bytes8_length]; rcases ho with h1 | ⟨h1, h2⟩ | h1
          · exact Or.inl (by omega)
          · exact Or.inl (by omega)
          · exact Or.inr (by omega)
        have hc2 : off + 8 ≤ 48 ∨ 48 + (bytes8 nb).length ≤ off := by
          rw [bytes8_length]; rcases ho with h1 | ⟨h1, h2⟩ | h1
          · exact Or.inl (by omega)
          · exact Or.inr (by omega)
          · exact Or.inr (by omega)
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ hc1 (by rw [writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ hc2 (by omega)]
      have hwf' : vc'.wf pr ls s' := wf_update_rs_loc h hmwf' hbl' hpos' hr' hsp' h15' hstk_m hstkb hstkrw hstkl
        hctx2_m (by rw [hctx2_eq]) (by rw [hctx2_eq]) (by rw [hctx2_eq]; dsimp; rw [writeBytes_length, writeBytes_length])
        (by rw [hctx2_eq, hrs'_eq]; dsimp; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega), leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)])
        (by rw [hdisj 56 (Or.inr (Or.inl ⟨by decide, by decide⟩)) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) hrsLen)
        (by rw [hctx2_eq]; dsimp; rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)])
        (by rw [hdisj 72 (Or.inr (Or.inr (by decide))) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s72)
        (by rw [hdisj 80 (Or.inr (Or.inr (by decide))) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s80)
        (by rw [hdisj 88 (Or.inr (Or.inr (by decide))) (by decide)]; exact ctx_leVal_of_rd64 h (by decide) s88)
        (fun off ho hl => hdisj off (by rcases ho with h1 | h1; exact Or.inl h1; exact Or.inr (Or.inr (by omega))) hl)
        hrs'_m (by rw [hrs'_eq]) (by omega) (by rw [hrs'_eq]; dsimp; rw [List.length_append, List.length_replicate, hrsLenEq]; omega)
        (by rw [show (2 * vc.rsCap) * 32 = 2 * (vc.rsCap * 32) by omega, Nat.mul_mod, hrsPg]; decide)
        (by rw [hrs'_eq]; simpa [Region.hi, List.length_append, List.length_replicate, hrsLenEq, show vc.rsCap * 32 + vc.rsCap * 32 = 2 * (vc.rsCap * 32) by omega] using hnb_hi)
        hloc_m hlocRw hlocPos hlocLenEq hlocPg hlocHi
        (by rw [hrs'_eq]; exact rsLocRel_extend_rs_zero hrsRel (by rw [hrsLenEq]; omega)) (by omega) hlocLe
        (vc.code_flat h).2.2.2.2.2.2.1 hnd' (fun u hu h1 h2 h3 _ => hoth u hu h1 h2 h3)
      exact Or.inl ⟨vc', s', hreach', hrip', hk', hor', hwf', by dsimp [vc']; omega,
        by rw [hroth 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr12_3,
        by rw [hroth 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr13_3⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hreach4 hhit, Ends.of_reaches hreach4 hend⟩

end SporosVmOpsCtrl
