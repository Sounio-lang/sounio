import Bootstrap.SporosVmPrimHeap
import Bootstrap.SporosVmPrimTab
import Bootstrap.SporosVmOpsData1

/-!
# Native x86-64 VM Exhaustive 57-Primitive Dispatch (`SporosVmPrimDispatch`)

Connects the shared primitive prologue (`L_prim_prologue_spec`) to all 57
per-primitive simulation theorems (`P_dup_spec` through `P_demand_spec`) across
`SporosVmPrimBase`, `SporosVmPrimMem`, `SporosVmPrimIo`, `SporosVmPrimBlock`,
`SporosVmPrimHeap`, and `SporosVmPrimTab`, lifting the result to the uniform
`SporosVmOpsData.OpPostOutcome` step predicate.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimDispatch

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase
  SporosVmPrimBase SporosVmPrimMem SporosVmPrimIo SporosVmPrimBlock
  SporosVmPrimHeap SporosVmPrimTab SporosVmOpsData

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

private theorem lst_st_rs_eq {ls ls' : SporosProvLen.LSt} {a n v q : Nat}
    (h : ls.st a n v q = some ls') : ls'.rs = ls.rs := by
  unfold SporosProvLen.LSt.st at h
  split at h <;> simp at h
  subst h; rfl

theorem lexecPrim_rs_eq (p : Sporos.Prim) (args : List SporosProvLen.LVal)
    (ls : SporosProvLen.LSt) {res : List SporosProvLen.LVal} {ls' : SporosProvLen.LSt}
    (h : SporosProvLen.lexecPrim p args ls = .ok res ls') :
    ls'.rs = ls.rs := by
  unfold SporosProvLen.lexecPrim at h
  dsimp only at h
  split at h
  all_goals first
    | (cases h; all_goals first | rfl | exact lst_st_rs_eq ‹_›)
    | (split at h <;> first
        | (cases h; all_goals first | rfl | exact lst_st_rs_eq ‹_›)
        | (split at h <;> cases h; all_goals rfl))

theorem lift_primPost_to_opPost {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {s s' : X86Vm.St} {p : Sporos.Prim}
    (hrs : ls.rs = f :: rest)
    (hfetch : Sporos.fetch pr f.word f.pc = some (.prim p))
    (hle : p.eff.1 ≤ ls.ds.length)
    (hrch : Reaches vmImage s s') (hk : s'.k = s.k)
    (hpost : PrimPostOutcome pr ls f rest s' p) :
    OpPostOutcome pr ls s := by
  unfold OpPostOutcome
  unfold SporosProvLen.lstep
  rw [hrs]; dsimp only
  rw [hfetch]; dsimp only
  rw [if_neg (Nat.not_lt.mpr hle)]
  unfold PrimPostOutcome at hpost
  cases hlex : SporosProvLen.lexecPrim p (ls.ds.take p.eff.1) ls with
  | ok res ls' =>
    rw [hlex] at hpost
    dsimp only at hpost ⊢
    have hrs' : ls'.rs = f :: rest := (lexecPrim_rs_eq p _ ls hlex).trans hrs
    have htop : ({ ls' with ds := res ++ ls.ds.drop p.eff.1 } : SporosProvLen.LSt).setTop
        { f with pc := f.pc + 1 } =
        { ls' with ds := res ++ ls.ds.drop p.eff.1, rs := { f with pc := f.pc + 1 } :: rest } := by
      simp [SporosProvLen.LSt.setTop, hrs']
    rw [htop]
    exact lift_next_outcome hrch hk hpost
  | halt code =>
    rw [hlex] at hpost
    obtain ⟨sx, hrx, hsx, hkx⟩ := hpost
    exact ⟨sx, Reaches.trans hrch hrx, hsx, by rw [hkx, hk]⟩
  | fail e =>
    rw [hlex] at hpost
    exact lift_fail_outcome hrch hk hpost

/-- Exhaustive 57-primitive dispatch from `L_prim_prologue` to `OpPostOutcome`. -/
theorem prim_dispatch_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (p : Sporos.Prim)
    (hrip : s.rip = SporosVmBin.L_prim_prologue)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest)
    (hfetch : Sporos.fetch pr f.word f.pc = some (.prim p))
    (hpcW : f.pc + 1 < X86Env.W)
    (hr0 : s.reg 0 = some (primCode p)) :
    OpPostOutcome pr ls s := by
  have hpro := L_prim_prologue_spec p hrip h hr0
  by_cases hlt : ls.ds.length < p.eff.1
  · rw [if_pos hlt] at hpro
    unfold OpPostOutcome SporosProvLen.lstep
    rw [hrs]; dsimp only
    rw [hfetch]; dsimp only
    rw [if_pos hlt]
    exact hpro
  · have hle : p.eff.1 ≤ ls.ds.length := Nat.le_of_not_lt hlt
    rw [if_neg hlt] at hpro
    obtain ⟨ctxData', s', hrch, hrip', hk', hwf', _, h272, h280, h288, h1, h2, h3, h4, _⟩ := hpro
    have hor' : OracleAdmits s' := oracleAdmits_of_reaches hor hrch
    apply lift_primPost_to_opPost hrs hfetch hle hrch hk'
    cases p with
    | dup => exact P_dup_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2
    | drop => exact P_drop_spec hrip' hle hwf' hor' hrs hpcW h280 h288
    | swap => exact P_swap_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2 (h2 (by decide)).1 (h2 (by decide)).2
    | over => exact P_over_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2 (h2 (by decide)).1 (h2 (by decide)).2
    | rot => exact P_rot_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2 (h2 (by decide)).1 (h2 (by decide)).2 (h3 (by decide)).1 (h3 (by decide)).2
    | nip => exact P_nip_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2
    | tuck => exact P_tuck_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h1 (by decide)).2 (h2 (by decide)).1 (h2 (by decide)).2
    | add => exact P_add_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | sub => exact P_sub_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | mul => exact P_mul_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | divu => exact P_divu_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | modu => exact P_modu_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | divs => exact P_divs_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | mods => exact P_mods_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | neg => exact P_neg_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | band => exact P_band_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | bor => exact P_bor_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | bxor => exact P_bxor_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | bnot => exact P_bnot_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | shl => exact P_shl_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | shr => exact P_shr_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | sar => exact P_sar_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | eq => exact P_eq_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ne => exact P_ne_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ltu => exact P_ltu_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | lts => exact P_lts_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | leu => exact P_leu_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | les => exact P_les_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ld8 => exact P_ld8_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | st8 => exact P_st8_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ld16 => exact P_ld16_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | st16 => exact P_st16_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ld32 => exact P_ld32_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | st32 => exact P_st32_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | ld64 => exact P_ld64_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | st64 => exact P_st64_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | copy => exact P_copy_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1 (h3 (by decide)).1
    | fill => exact P_fill_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1 (h3 (by decide)).1
    | mcmp => exact P_mcmp_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1 (h3 (by decide)).1
    | alloc => exact P_alloc_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1
    | mark => exact P_mark_spec hrip' hwf' hor' hrs hpcW h272 h280 h288
    | release => exact P_release_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1
    | streq => exact P_streq_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1 (h3 (by decide)).1 (h4 (by decide)).1
    | strhash => exact P_strhash_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | tnew => exact P_tnew_spec hrip' hwf' hor' hrs hpcW h272 h280 h288
    | tput => exact P_tput_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h1 (by decide)).2 (h2 (by decide)).1 (h3 (by decide)).1
    | tget => exact P_tget_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h2 (by decide)).1
    | getb => exact P_getb_spec hrip' hwf' hor' hrs hpcW h272 h280 h288
    | putb => exact P_putb_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1
    | putn => exact P_putn_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1
    | exit => exact P_exit_spec hrip' hle hwf' (h1 (by decide)).1
    | clock => exact P_clock_spec hrip' hwf' hor' hrs hpcW h272 h280 h288
    | core => exact P_core_spec hrip' hwf' hor' hrs hpcW h272 h280 h288
    | sensor => exact P_sensor_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288
    | provget => exact P_provget_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h1 (by decide)).2
    | confget => exact P_confget_spec hrip' hle hwf' hor' hrs hpcW h272 h280 h288 (h1 (by decide)).1 (h1 (by decide)).2
    | demand => exact P_demand_spec hrip' hle hwf' hor' hrs hpcW h280 h288 (h1 (by decide)).1 (h2 (by decide)).1 (h2 (by decide)).2

end SporosVmPrimDispatch
