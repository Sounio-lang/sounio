import Bootstrap.SporosVmPrimBlock

/-!
# Native x86-64 VM Heap & String Primitive Proofs (`P_alloc`, `P_mark`, `P_release`, `P_streq`, `P_strhash`)

Simulation proofs over `X86Vm` for the 5 arena-heap and string primitives
(`alloc`, `mark`, `release`, `streq`, `strhash`), completing the 54 non-table
primitives (with `SporosVmPrimTab.lean` supplying the final 3 table primitives).

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimHeap

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem SporosVmPrimBlock

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### Reusable `ctx[120]` (`hp`) write helper and FNV-1a lemmas -/

theorem step_wr_ctx_hp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {r hp' len : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 120 r)) len s)
    (h : vc.wf pr ls s) (hr : s.reg r = some hp') (hW : hp' < X86Env.W) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 120 (bytes8 hp') }
    let vc' := { vc with ctxReg := ctx' }
    ∃ s', X86Vm.step vmImage s = .ok s' ∧ s'.rip = s.rip + len ∧ s'.k = s.k ∧
      s'.regs = s.regs ∧ s'.fl = s.fl ∧ vc'.wf pr { ls with hp := hp' } s' ∧
      (∀ m0 u, (m0 + 8 ≤ 120 ∨ 128 ≤ m0) → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hcb := vc.ctx_bounds h
  have hcs : ctxSize = 4096 := rfl
  have hcb0 : ctxBase = 0x600000 := rfl
  have hW_nat : X86Env.W = 2 ^ 64 := rfl
  have hea120 : ea ctxBase 120 = ctxBase + 120 := by rw [ea_add_eq (by omega)]
  obtain ⟨_, m', hwr, _, _, hwf', hmem', hoth⟩ :=
    wr64_region h.1.2 (vc.mem_ctx h) hcb.2.1 (a := ctxBase + 120) (v := hp')
      (by rw [hcb.1]; omega) (by simp only [Region.hi, hcb.1]; omega)
  have hwr' : wr64 s.mem (ea ctxBase 120) hp' = some m' := by rw [hea120]; exact hwr
  have hcm' : CodeMem m' := h.1.wr64_preserve hwr'
  have hbl' : BytesLt m' := bytesLt_wr64 h.2.1 hwr'
  have hpos' : ∀ u ∈ m', 0 < u.data.length := len_pos_wrN h.2.2.1 hwr'
  rw [hcb.1, Nat.add_sub_cancel_left] at hmem'
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 120 (bytes8 hp') }
  have hmem'' : ctx' ∈ m' := by simpa [ctx', hcb.1] using hmem'
  let vc' := { vc with ctxReg := ctx' }
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by rw [hstep, exec_st h15 hr hwr']
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
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, _hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    simpa [vc', ctx', VmCtx.regions] using hnodup
  have hmem_all : ∀ u ∈ vc'.regions, u ∈ s'.mem := by
    intro u hu
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    rcases hu with rfl | rfl | huc | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | (rw [huc]; exact hmem'')
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hctx3' : ctxSize ≤ ctx'.data.length := by dsimp [ctx']; rw [writeBytes_length]; exact hctx3
  have hctx4' : ctx'.hi ≤ 2 ^ 47 := by simpa only [Region.hi, ctx', writeBytes_length] using hctx4
  have hslot : ∀ j, (j + 8 ≤ 120 ∨ 128 ≤ j) → j + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop j).take 8) = leVal ((vc.ctxReg.data.drop j).take 8) := by
    intro j hj hlim
    dsimp [ctx']
    rw [writeBytes_drop_take_disjoint vc.ctxReg.data 120 j 8 (bytes8 hp')
      (by rw [bytes8_length]; exact hj) (by omega)]
  have hsl15' : leVal ((ctx'.data.drop 120).take 8) = hp' := by
    dsimp [ctx']
    rw [leVal_writeBytes_same vc.ctxReg.data 120 hp' (by omega), Nat.mod_eq_of_lt hW]
  have hwf' : vc'.wf pr { ls with hp := hp' } s' :=
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
      hsl15',
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
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩
  refine ⟨s', hs, rfl, rfl, rfl, rfl, hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0 hmlim, hle_ctx m0 u hmlim hprev])

private theorem fnv_lt_M (bs : List Nat) : Sporos.fnv bs < Sporos.M := by
  unfold Sporos.fnv
  suffices ∀ (xs : List Nat) (acc : Nat), acc < Sporos.M →
      xs.foldl (fun h b => (Nat.xor h b) * 1099511628211 % Sporos.M) acc < Sporos.M by
    exact this bs 14695981039346656037 (by rw [M_eq_W']; change 14695981039346656037 < 2 ^ 64; omega)
  intro xs
  induction xs with
  | nil => intro acc ha; exact ha
  | cons b xs ih =>
    intro acc _
    exact ih _ (Nat.mod_lt _ (by rw [M_eq_W']; change 0 < 2 ^ 64; omega))

private theorem foldl_append_singleton {α β : Type} (g : α → β → α) (init : α) (xs : List β) (x : β) :
    (xs ++ [x]).foldl g init = g (xs.foldl g init) x := by
  rw [List.foldl_append]; rfl

private theorem fnv_range_step (f : Nat → Nat) (i : Nat) :
    Sporos.fnv ((List.range (i + 1)).map f) =
      (Nat.xor (Sporos.fnv ((List.range i).map f)) (f i) * 1099511628211) % Sporos.M := by
  rw [List.range_succ, List.map_append, List.map_cons, List.map_nil]
  exact foldl_append_singleton _ _ ((List.range i).map f) (f i)

/-! ### `P_alloc`, `P_mark`, `P_release` -/

theorem P_alloc_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_alloc) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .alloc := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hmi := wf_mem_info h
  have hhpW : ls.hp < X86Env.W := hmi.2.2.2
  have htake : ls.ds.take 1 = [ls.ds[0]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hnW : ls.ds[0]!.w < X86Env.W := by
    rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hhp_rd : rd64 s.mem (ea ctxBase 120) = some ls.hp := SporosVmPrimBase.rd_ctx_hp h
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := SporosVmPrimBase.rd_ctx_msize h
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.hp) with rip := 4197856 } := by
    rw [step_at_vc 11 hrip (by decide : (SporosVmBin.P_alloc, (.ofInt (.ld 0 15 120)), 4) ∈ vmTableChunk 11),
      exec_ld h15 hhp_rd, hrip]; rfl
  let s1 := { s.set 0 (some ls.hp) with rip := 4197856 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := 4197863 } := by
    rw [step_at_vc 11 rfl (by decide : (4197856, (.ofInt (.ld 1 15 296)), 7) ∈ vmTableChunk 11), exec_ld hs1_15 h296]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := 4197863 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 10 (some ls.msize) with rip := 4197867 } := by
    rw [step_at_vc 11 rfl (by decide : (4197863, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 11), exec_ld hs2_15 hms]
  let s3 := { s2.set 10 (some ls.msize) with rip := 4197867 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hwf3 : vc.wf pr ls s3 :=
    vc.wf_step_misc h rfl hr3_ok
      (by change (s2.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s2.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15)
  have hs3_0 : s3.reg 0 = some ls.hp := by
    change (s2.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs3_1 : s3.reg 1 = some ls.ds[0]!.w := by
    change (s2.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs3_10 : s3.reg 10 = some ls.msize := reg_set_self_ok' hr2_ok (by decide) _
  have hrc := range_check_spec (rip0 := 4197867) (c0 := 11) (c1 := 11) (c2 := 11) (c3 := 11) (c4 := 11) (c5 := 11)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf3 hs3_0 hs3_1 hs3_10 hhpW hnW (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases hok : ls.hp + ls.ds[0]!.w ≤ ls.msize
  · rw [if_pos hok] at hrc ⊢
    obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4_ok, hs4_4, hs4_15, _, hs4_oth⟩ := hrc
    have hs4_0 : s4.reg 0 = some ls.hp := by rw [hs4_oth 0 (by decide)]; exact hs3_0
    have hs4_1 : s4.reg 1 = some ls.ds[0]!.w := by rw [hs4_oth 1 (by decide)]; exact hs3_1
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some ls.hp) with rip := 4197894 } := by
      rw [step_at_vc 11 hrip4 (by decide : (4197891, (.ofInt (.mov 2 0)), 3) ∈ vmTableChunk 11),
        exec_mov hs4_0, hrip4]
    let s5 := { s4.set 2 (some ls.hp) with rip := 4197894 }
    have hr5_ok : RegsOk s5 := regsOk_set hr4_ok (by decide) _
    have hs5_2 : s5.reg 2 = some ls.hp := reg_set_self_ok' hr4_ok (by decide) _
    have hs5_1 : s5.reg 1 = some ls.ds[0]!.w := by change (s4.set 2 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact hs4_1
    have hhp_addW : ls.hp + ls.ds[0]!.w < X86Env.W := by
      have : ls.msize < 2 ^ 44 := hmi.1.2.1; have : X86Env.W = 2 ^ 64 := rfl; omega
    have hadd_val : (X86Int.aluF .add ls.hp ls.ds[0]!.w).1 = ls.hp + ls.ds[0]!.w := by
      rw [aluF_add_val, Nat.mod_eq_of_lt hhp_addW]
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5.set 2 (some (ls.hp + ls.ds[0]!.w)) with fl := (X86Int.aluF .add ls.hp ls.ds[0]!.w).2, rip := 4197897 } := by
      rw [step_at_vc 11 rfl (by decide : (4197894, (.ofInt (.alu .add 2 1)), 3) ∈ vmTableChunk 11),
        exec_alu hs5_2 hs5_1, ← hadd_val]
    let s6 := { s5.set 2 (some (ls.hp + ls.ds[0]!.w)) with fl := (X86Int.aluF .add ls.hp ls.ds[0]!.w).2, rip := 4197897 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_2 : s6.reg 2 = some (ls.hp + ls.ds[0]!.w) := reg_set_self_ok' hr5_ok (by decide) _
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change (s5.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]
      change (s4.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact hs4_4
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s4.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs4_15
    have hs6_0 : s6.reg 0 = some ls.hp := by
      change (s5.set 2 _).reg 0 = _; rw [reg_set_ne (by decide)]
      change (s4.set 2 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs4_0
    have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc hwf3 hmem4 hr6_ok hs6_4 hs6_15
    obtain ⟨s7, h07, hrip7, hk7, hregs7, _, hwf7, hpres7⟩ :=
      step_wr_ctx_hp (step_at_vc 11 rfl (by decide : (4197897, (.ofInt (.st 15 120 2)), 4) ∈ vmTableChunk 11))
        hwf6 hs6_2 hhp_addW
    have hrch07 : Reaches vmImage s s7 :=
      Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step h07))))
    have hqW : SporosProvLen.lq [ls.ds[0]!] ls.pc < X86Env.W := by
      simp [SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)
    have h272_7 : rd64 s7.mem (ea ctxBase 272) = some (SporosProvLen.lq [ls.ds[0]!] ls.pc) :=
      hpres7 272 _ (Or.inr (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; change rd64 s.mem _ = _; rw [htake] at h272; exact h272)
    have h280_7 : rd64 s7.mem (ea ctxBase 280) = some 1 :=
      hpres7 280 _ (Or.inr (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact h280)
    have h288_7 : rd64 s7.mem (ea ctxBase 288) = some 1 :=
      hpres7 288 _ (Or.inr (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact h288)
    have hfin := jmp_mk1_finish_spec
      (step_at_vc 11 hrip7 (by decide : (4197901, (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)), 5) ∈ vmTableChunk 11))
      hwf7 (oracleAdmits_of_reaches hor hrch07) hrs hpcW hle
      (by simp only [X86Int.St.reg, hregs7]; exact hs6_0) h272_7 hqW h280_7 h288_7
    have hwrap : Sporos.wrap ls.hp = ls.hp % X86Env.W := by simp [Sporos.wrap, M_eq_W']
    simpa [hwrap, List.cons_append] using lift_next_outcome hrch07 (by rw [hk7]; exact hk4) hfin
  · rw [if_neg hok] at hrc ⊢
    exact lift_fail_outcome hrch3 rfl hrc

theorem P_mark_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_mark)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .mark := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hhp_rd : rd64 s.mem (ea ctxBase 120) = some ls.hp := SporosVmPrimBase.rd_ctx_hp h
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.hp) with rip := 4197910 } := by
    rw [step_at_vc 11 hrip (by decide : (SporosVmBin.P_mark, (.ofInt (.ld 0 15 120)), 4) ∈ vmTableChunk 11),
      exec_ld h15 hhp_rd, hrip]; rfl
  let s1 := { s.set 0 (some ls.hp) with rip := 4197910 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_0 : s1.reg 0 = some ls.hp := reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hwf1 : vc.wf pr ls s1 :=
    vc.wf_step_misc h rfl hr1_ok
      (by change (s.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15)
  have hqW : SporosProvLen.lq [] ls.pc < X86Env.W := by
    simp [SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)
  have hfin := jmp_mk1_finish_spec
    (step_at_vc 11 rfl (by decide : (4197910, (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)), 5) ∈ vmTableChunk 11))
    hwf1 (oracleAdmits_of_reaches hor (Reaches.step h01)) hrs hpcW (Nat.zero_le _)
    hs1_0 h272 hqW h280 h288
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, List.take_zero, SporosProvLen.lexecPrim]
  have hwrap : Sporos.wrap ls.hp = ls.hp % X86Env.W := by simp [Sporos.wrap, M_eq_W']
  simpa [hwrap, List.cons_append] using lift_next_outcome (Reaches.step h01) rfl hfin

theorem P_release_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_release) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .release := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hmi := wf_mem_info h
  have hhpW : ls.hp < X86Env.W := hmi.2.2.2
  have htake : ls.ds.take 1 = [ls.ds[0]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hmW : ls.ds[0]!.w < X86Env.W := by
    rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hhp_rd : rd64 s.mem (ea ctxBase 120) = some ls.hp := SporosVmPrimBase.rd_ctx_hp h
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4197922 } := by
    rw [step_at_vc 11 hrip (by decide : (SporosVmBin.P_release, (.ofInt (.ld 0 15 296)), 7) ∈ vmTableChunk 11),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4197922 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.hp) with rip := 4197926 } := by
    rw [step_at_vc 11 rfl (by decide : (4197922, (.ofInt (.ld 1 15 120)), 4) ∈ vmTableChunk 11), exec_ld hs1_15 hhp_rd]
  let s2 := { s1.set 1 (some ls.hp) with rip := 4197926 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_0 : s2.reg 0 = some ls.ds[0]!.w := by
    change (s1.set 1 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs2_1 : s2.reg 1 = some ls.hp := reg_set_self_ok' hr1_ok (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2 with fl := (X86Int.aluF .sub ls.ds[0]!.w ls.hp).2, rip := 4197929 } := by
    rw [step_at_vc 11 rfl (by decide : (4197926, (.ofInt (.cmp 0 1)), 3) ∈ vmTableChunk 11), exec_cmp hs2_0 hs2_1]
  let s3 := { s2 with fl := (X86Int.aluF .sub ls.ds[0]!.w ls.hp).2, rip := 4197929 }
  have hca : X86Int.cond s3.fl 7 = some (decide (ls.hp < ls.ds[0]!.w)) := by
    dsimp [s3]; exact cond_aluF_sub_a _ _ hmW hhpW
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases hle_hp : ls.ds[0]!.w ≤ ls.hp
  · rw [if_pos hle_hp]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4197935 } := by
      rw [step_at_vc 11 rfl (by decide : (4197929, (.ofInt (.jcc 7 4201150)), 6) ∈ vmTableChunk 11),
        exec_jcc (by rw [hca, decide_eq_false (by omega)])]; rfl
    let s4 := { s3 with rip := 4197935 }
    have hwf4 : vc.wf pr ls s4 :=
      vc.wf_step_misc h rfl hr2_ok
        (by change (s1.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]
            change (s.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
        (by change (s1.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15)
    obtain ⟨s5, h05, hrip5, hk5, _, _, hwf5, hpres5⟩ :=
      step_wr_ctx_hp (step_at_vc 11 rfl (by decide : (4197935, (.ofInt (.st 15 120 0)), 4) ∈ vmTableChunk 11))
        hwf4 hs2_0 hmW
    have hrch05 : Reaches vmImage s s5 := Reaches.trans hrch3 (Reaches.step_trans h04 (Reaches.step h05))
    have h280_5 : rd64 s5.mem (ea ctxBase 280) = some 1 := hpres5 280 _ (Or.inr (by decide)) (by decide) h280
    have h288_5 : rd64 s5.mem (ea ctxBase 288) = some 0 := hpres5 288 _ (Or.inr (by decide)) (by decide) h288
    have hfin := jmp_epilogue0_finish_spec
      (outs := [])
      (step_at_vc 11 hrip5 (by decide : (4197939, (.ofInt (.jmp SporosVmBin.L_prim_epilogue)), 5) ∈ vmTableChunk 11))
      hwf5 (oracleAdmits_of_reaches hor hrch05) hrs hpcW hle (by decide)
      (by intro _ hv; cases hv) h280_5 h288_5 (by intro _ hi; cases hi)
    exact lift_next_outcome hrch05 hk5 hfin
  · rw [if_neg hle_hp]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := SporosVmBin.L_fail_bounds } := by
      rw [step_at_vc 11 rfl (by decide : (4197929, (.ofInt (.jcc 7 4201150)), 6) ∈ vmTableChunk 11),
        exec_jcc (by rw [hca, decide_eq_true (by omega)])]; rfl
    let s4 := { s3 with rip := SporosVmBin.L_fail_bounds }
    have hwf4 : vc.wf pr ls s4 :=
      vc.wf_step_misc h rfl hr2_ok
        (by change (s1.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]
            change (s.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
        (by change (s1.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15)
    exact lift_fail_outcome (Reaches.trans hrch3 (Reaches.step h04)) rfl (L_fail_from_vc_spec (e := .bounds) rfl hwf4)

/-! ### `P_streq`, `P_strhash` -/

theorem P_streq_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_streq) (hle : 4 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 4) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 4) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w)
    (h344 : rd64 s.mem (ea ctxBase 344) = some ls.ds[3]!.w) :
    PrimPostOutcome pr ls f rest s .streq := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have htake : ls.ds.take 4 = [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!, ls.ds[3]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 =>
      cases hd1 : r0 with
      | nil => simp [hd0, hd1] at hle
      | cons a1 r1 =>
        cases hd2 : r1 with
        | nil => simp [hd0, hd1, hd2] at hle
        | cons a2 r2 =>
          cases hd3 : r2 with
          | nil => simp [hd0, hd1, hd2, hd3] at hle
          | cons a3 r3 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hl2W : ls.ds[0]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have ha2W : ls.ds[1]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hl1W : ls.ds[2]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have ha1W : ls.ds[3]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := SporosVmPrimBase.rd_ctx_msize h
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[3]!.w) with rip := 4197951 } := by
    rw [step_at_vc 11 hrip (by decide : (SporosVmBin.P_streq, (.ofInt (.ld 0 15 344)), 7) ∈ vmTableChunk 11),
      exec_ld h15 h344, hrip]; rfl
  let s1 := { s.set 0 (some ls.ds[3]!.w) with rip := 4197951 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[2]!.w) with rip := 4197958 } := by
    rw [step_at_vc 11 rfl (by decide : (4197951, (.ofInt (.ld 1 15 328)), 7) ∈ vmTableChunk 11), exec_ld hs1_15 h328]
  let s2 := { s1.set 1 (some ls.ds[2]!.w) with rip := 4197958 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 10 (some ls.msize) with rip := 4197962 } := by
    rw [step_at_vc 11 rfl (by decide : (4197958, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 11), exec_ld hs2_15 hms]
  let s3 := { s2.set 10 (some ls.msize) with rip := 4197962 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hwf3 : vc.wf pr ls s3 :=
    vc.wf_step_misc h rfl hr3_ok
      (by change (s2.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s2.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15)
  have hs3_0 : s3.reg 0 = some ls.ds[3]!.w := by
    change (s2.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs3_1 : s3.reg 1 = some ls.ds[2]!.w := by
    change (s2.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs3_10 : s3.reg 10 = some ls.msize := reg_set_self_ok' hr2_ok (by decide) _
  have hrc1 := range_check_spec (rip0 := 4197962) (c0 := 11) (c1 := 11) (c2 := 11) (c3 := 11) (c4 := 11) (c5 := 11)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf3 hs3_0 hs3_1 hs3_10 ha1W hl1W (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases ha1_ok : ls.ds[3]!.w + ls.ds[2]!.w ≤ ls.msize
  · rw [if_pos ha1_ok] at hrc1
    obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4_ok, hs4_4, hs4_15, _, hs4_oth⟩ := hrc1
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc hwf3 hmem4 hr4_ok hs4_4 hs4_15
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some ls.ds[1]!.w) with rip := 4197993 } := by
      rw [step_at_vc 11 hrip4 (by decide : (4197986, (.ofInt (.ld 2 15 312)), 7) ∈ vmTableChunk 11),
        exec_ld hs4_15 (by rw [hmem4]; exact h312), hrip4]
    let s5 := { s4.set 2 (some ls.ds[1]!.w) with rip := 4197993 }
    have hr5_ok : RegsOk s5 := regsOk_set hr4_ok (by decide) _
    have hs5_15 : s5.reg 15 = some ctxBase := by change (s4.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs4_15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 8 (some ls.ds[0]!.w) with rip := 4198000 } := by
      rw [step_at_vc 11 rfl (by decide : (4197993, (.ofInt (.ld 8 15 296)), 7) ∈ vmTableChunk 11),
        exec_ld hs5_15 (by change rd64 s4.mem _ = _; rw [hmem4]; exact h296)]
    let s6 := { s5.set 8 (some ls.ds[0]!.w) with rip := 4198000 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by change (s5.set 8 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs5_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 10 (some ls.msize) with rip := 4198004 } := by
      rw [step_at_vc 11 rfl (by decide : (4198000, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 11),
        exec_ld hs6_15 (by change rd64 s4.mem _ = _; rw [hmem4]; exact hms)]
    let s7 := { s6.set 10 (some ls.msize) with rip := 4198004 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hrch07 : Reaches vmImage s s7 :=
      Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step h07))))
    have hwf7 : vc.wf pr ls s7 :=
      vc.wf_step_misc hwf4 rfl hr7_ok
        (by change (s6.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
            change (s5.set 8 _).reg 4 = _; rw [reg_set_ne (by decide)]
            change (s4.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact hs4_4)
        (by change (s6.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs6_15)
    have hs7_0 : s7.reg 0 = some ls.ds[3]!.w := by
      change (s6.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
      change (s5.set 8 _).reg 0 = _; rw [reg_set_ne (by decide)]
      change (s4.set 2 _).reg 0 = _; rw [reg_set_ne (by decide), hs4_oth 0 (by decide)]; exact hs3_0
    have hs7_1 : s7.reg 1 = some ls.ds[2]!.w := by
      change (s6.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]
      change (s5.set 8 _).reg 1 = _; rw [reg_set_ne (by decide)]
      change (s4.set 2 _).reg 1 = _; rw [reg_set_ne (by decide), hs4_oth 1 (by decide)]; exact hs3_1
    have hs7_2 : s7.reg 2 = some ls.ds[1]!.w := by
      change (s6.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]
      change (s5.set 8 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr4_ok (by decide) _
    have hs7_8 : s7.reg 8 = some ls.ds[0]!.w := by
      change (s6.set 10 _).reg 8 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr5_ok (by decide) _
    have hs7_10 : s7.reg 10 = some ls.msize := reg_set_self_ok' hr6_ok (by decide) _
    have hrc2 := range_check_spec (rip0 := 4198004) (c0 := 11) (c1 := 11) (c2 := 11) (c3 := 11) (c4 := 11) (c5 := 11)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      rfl hwf7 hs7_2 hs7_8 hs7_10 ha2W hl2W (by decide) (by decide)
    by_cases ha2_ok : ls.ds[1]!.w + ls.ds[0]!.w ≤ ls.msize
    · rw [if_pos ⟨ha1_ok, ha2_ok⟩]
      rw [if_pos ha2_ok] at hrc2
      obtain ⟨s8, hrch8, hrip8, hmem8, hk8, hr8_ok, hs8_4, hs8_15, hs8_11, hs8_oth⟩ := hrc2
      have hrch08 : Reaches vmImage s s8 := Reaches.trans hrch07 hrch8
      have hmem08 : s8.mem = s.mem := by rw [hmem8]; exact hmem4
      have hk08 : s8.k = s.k := by rw [hk8]; exact hk4
      have hs8_0 : s8.reg 0 = some ls.ds[3]!.w := by rw [hs8_oth 0 (by decide)]; exact hs7_0
      have hs8_1 : s8.reg 1 = some ls.ds[2]!.w := by rw [hs8_oth 1 (by decide)]; exact hs7_1
      have hs8_2 : s8.reg 2 = some ls.ds[1]!.w := by rw [hs8_oth 2 (by decide)]; exact hs7_2
      have hs8_8 : s8.reg 8 = some ls.ds[0]!.w := by rw [hs8_oth 8 (by decide)]; exact hs7_8
      have h09 : X86Vm.step vmImage s8 =
          .ok { s8 with fl := (X86Int.aluF .sub ls.ds[2]!.w ls.ds[0]!.w).2, rip := 4198031 } := by
        rw [step_at_vc 11 hrip8 (by decide : (4198028, (.ofInt (.cmp 1 8)), 3) ∈ vmTableChunk 11),
          exec_cmp hs8_1 hs8_8, hrip8]
      let s9 := { s8 with fl := (X86Int.aluF .sub ls.ds[2]!.w ls.ds[0]!.w).2, rip := 4198031 }
      have hce : X86Int.cond s9.fl 4 = some (decide (ls.ds[2]!.w = ls.ds[0]!.w)) := by
        dsimp [s9]; exact cond_aluF_sub_e _ _ hl1W hl2W
      have hqW : SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!, ls.ds[3]!] ls.pc < X86Env.W := by
        simp [SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)
      by_cases hleq : ls.ds[2]!.w = ls.ds[0]!.w
      · have h10 : X86Vm.step vmImage s9 = .ok { s9 with rip := 4198040 } := by
          rw [step_at_vc 11 rfl (by decide : (4198031, (.ofInt (.jcc 4 4198040)), 2) ∈ vmTableChunk 11),
            exec_jcc (by rw [hce, decide_eq_true hleq])]; rfl
        let s10 := { s9 with rip := 4198040 }
        have hrch10 : Reaches vmImage s s10 := Reaches.trans hrch08 (Reaches.step_trans h09 (Reaches.step h10))
        have hwf10 : vc.wf pr ls s10 := vc.wf_step_misc hwf7 hmem8 hr8_ok hs8_4 hs8_15
        have hfin := mem_cmp_finish_spec (vc := vc) (pr := pr) (ls := ls) (s := s10) (f := f) (rest := rest)
          (x := ls.ds[3]!.w) (y := ls.ds[1]!.w) (n := ls.ds[2]!.w)
          (q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!, ls.ds[3]!] ls.pc) (e_in := 4)
          (rip0 := 4198040) (c_ld8 := 11) (c_pre := 12) (c_body := 12)
          rfl hwf10 (oracleAdmits_of_reaches hor hrch10) hrs hpcW hle ha1_ok (by omega) hqW
          hs8_0 hs8_1 hs8_2 hs8_11
          (by change rd64 s8.mem _ = _; rw [hmem08]; rw [htake] at h272; exact h272)
          (by change rd64 s8.mem _ = _; rw [hmem08]; exact h280)
          (by change rd64 s8.mem _ = _; rw [hmem08]; exact h288)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        have hres := lift_next_outcome hrch10 hk08 hfin
        simp only [hleq, BEq.refl, Bool.true_and, ↓reduceIte] at hres ⊢
        exact hres
      · have h10 : X86Vm.step vmImage s9 = .ok { s9 with rip := 4198033 } := by
          rw [step_at_vc 11 rfl (by decide : (4198031, (.ofInt (.jcc 4 4198040)), 2) ∈ vmTableChunk 11),
            exec_jcc (by rw [hce, decide_eq_false hleq])]; rfl
        let s10 := { s9 with rip := 4198033 }
        have h11 : X86Vm.step vmImage s10 =
            .ok { s10.set 0 (some 0) with fl := (X86Int.aluF .xor ls.ds[3]!.w ls.ds[3]!.w).2, rip := 4198035 } := by
          rw [step_at_vc 11 rfl (by decide : (4198033, (.ofInt (.alu .xor 0 0)), 2) ∈ vmTableChunk 11),
            exec_alu_xor_self (s := s10) hs8_0]
        let s11 := { s10.set 0 (some 0) with fl := (X86Int.aluF .xor ls.ds[3]!.w ls.ds[3]!.w).2, rip := 4198035 }
        have hr11_ok : RegsOk s11 := regsOk_set hr8_ok (by decide) _
        have hrch11 : Reaches vmImage s s11 :=
          Reaches.trans hrch08 (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step h11)))
        have hwf11 : vc.wf pr ls s11 :=
          vc.wf_step_misc hwf7 hmem8 hr11_ok
            (by change (s10.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact hs8_4)
            (by change (s10.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs8_15)
        have hfin := jmp_mk1_finish_spec
          (step_at_vc 11 rfl (by decide : (4198035, (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)), 5) ∈ vmTableChunk 11))
          hwf11 (oracleAdmits_of_reaches hor hrch11) hrs hpcW hle
          (reg_set_self_ok' hr8_ok (by decide) _)
          (by change rd64 s8.mem _ = _; rw [hmem08]; rw [htake] at h272; exact h272) hqW
          (by change rd64 s8.mem _ = _; rw [hmem08]; exact h280)
          (by change rd64 s8.mem _ = _; rw [hmem08]; exact h288)
        have hbeq_f : (ls.ds[2]!.w == ls.ds[0]!.w) = false := beq_false_of_ne hleq
        have hqmod : (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!, ls.ds[3]!] ls.pc + 0) % Sporos.M =
            SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!, ls.ds[3]!] ls.pc := by
          rw [Nat.add_zero, M_eq_W', Nat.mod_eq_of_lt hqW]
        have hzmod : 0 % X86Env.W = 0 := Nat.zero_mod _
        have hres := lift_next_outcome hrch11 hk08 hfin
        simp only [hleq, hbeq_f, Bool.false_and, Sporos.bool, Bool.false_eq_true, ↓reduceIte, hqmod, List.cons_append, List.nil_append]
        simpa only [hzmod] using hres
    · rw [if_neg (fun hc => ha2_ok hc.2)]
      rw [if_neg ha2_ok] at hrc2
      exact lift_fail_outcome hrch07 hk4 hrc2
  · rw [if_neg (fun hc => ha1_ok hc.1)]
    rw [if_neg ha1_ok] at hrc1
    exact lift_fail_outcome hrch3 rfl hrc1

private theorem strhash_loop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {a n q : Nat} (ha_in : a + n ≤ ls.msize) :
    ∀ (rem i hacc pacc : Nat) (s : X86Vm.St),
      i + rem = n → s.rip = 4198232 → vc.wf pr ls s →
      s.reg 0 = some hacc → s.reg 1 = some n → s.reg 2 = some 1099511628211 →
      s.reg 6 = some (vc.memwReg.base + a) → s.reg 7 = some (vc.mempReg.base + a * 8) →
      s.reg 10 = some (pacc % X86Env.W) → s.reg 11 = some i →
      hacc = Sporos.fnv ((List.range i).map fun k => (ls.mem (a + k)).w % 256) →
      pacc % X86Env.W = (q + (List.range i).foldr (fun k acc => (ls.mem (a + k)).plen + acc) 0) % X86Env.W →
      ∃ s', Reaches vmImage s s' ∧ s'.rip = 4198262 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        vc.wf pr ls s' ∧
        s'.reg 0 = some (Sporos.fnv ((List.range n).map fun k => (ls.mem (a + k)).w % 256)) ∧
        s'.reg 10 = some ((q + (List.range n).foldr (fun k acc => (ls.mem (a + k)).plen + acc) 0) % X86Env.W) := by
  intro rem
  induction rem with
  | zero =>
    intro i hacc pacc s hirem hrip h hr0 hr1 _ _ _ hr10 hr11 hhacc hpacc
    have hi_eq : i = n := by omega
    subst hi_eq
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i i).2, rip := 4198235 } := by
      rw [step_at_vc 12 hrip (by decide : (4198232, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 12),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i i).2, rip := 4198235 }
    have hcae : X86Int.cond s1.fl 3 = some true := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_true (Nat.le_refl _)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4198262 } := by
      rw [step_at_vc 12 rfl (by decide : (4198235, (.ofInt (.jcc 3 4198262)), 2) ∈ vmTableChunk 12),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := 4198262 }
    exact ⟨s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl,
      vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1,
      by rw [← hhacc]; exact hr0, by rw [← hpacc]; exact hr10⟩
  | succ rem ih =>
    intro i hacc pacc s hirem hrip h hr0 hr1 hr2 hr6 hr7 hr10 hr11 hhacc hpacc
    have hmi := wf_mem_info h
    have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
    have hW44 : 2 ^ 44 < X86Env.W := by change 2 ^ 44 < 2 ^ 64; omega
    have hai : a + i < ls.msize := by omega
    have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i n).2, rip := 4198235 } := by
      rw [step_at_vc 12 hrip (by decide : (4198232, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 12),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i n).2, rip := 4198235 }
    have hcae : X86Int.cond s1.fl 3 = some false := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_false (by omega)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4198237 } := by
      rw [step_at_vc 12 rfl (by decide : (4198235, (.ofInt (.jcc 3 4198262)), 2) ∈ vmTableChunk 12),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := 4198237 }
    have hea_w : eaIdx (vc.memwReg.base + a) i 1 0 = vc.memwReg.base + (a + i) := by
      have hltW : vc.memwReg.base + a + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_p : eaIdx (vc.mempReg.base + a * 8) i 8 0 = vc.mempReg.base + (a + i) * 8 := by
      have hltW : vc.mempReg.base + a * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_b : rdMem .w8 s2.mem (eaIdx (vc.memwReg.base + a) i 1 0) = some ((ls.mem (a + i)).w % 256) := by
      change rdMem .w8 s.mem _ = _; rw [hea_w]; exact rd_mem_byte_at h hai
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 3 (some ((ls.mem (a + i)).w % 256)) with rip := 4198242 } := by
      rw [step_at_vc 12 rfl (by decide : (4198237, (.ldIdx .w8 3 6 11 1 0), 5) ∈ vmTableChunk 12),
        exec_ldIdx (s := s2) hr6 hr11 hrd_b]
    let s3 := { s2.set 3 (some ((ls.mem (a + i)).w % 256)) with rip := 4198242 }
    have hr3_ok : RegsOk s3 := regsOk_set h.2.2.2.1 (by decide) _
    have hs3_0 : s3.reg 0 = some hacc := by change (s2.set 3 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hr0
    have hs3_3 : s3.reg 3 = some ((ls.mem (a + i)).w % 256) := reg_set_self_ok' h.2.2.2.1 (by decide) _
    have hxor_val : (X86Int.aluF .xor hacc ((ls.mem (a + i)).w % 256)).1 = Nat.xor hacc ((ls.mem (a + i)).w % 256) :=
      aluF_xor_val _ _
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 0 (some (Nat.xor hacc ((ls.mem (a + i)).w % 256))) with
              fl := (X86Int.aluF .xor hacc ((ls.mem (a + i)).w % 256)).2, rip := 4198245 } := by
      rw [step_at_vc 12 rfl (by decide : (4198242, (.ofInt (.alu .xor 0 3)), 3) ∈ vmTableChunk 12),
        exec_alu hs3_0 hs3_3, ← hxor_val]
    let s4 := { s3.set 0 (some (Nat.xor hacc ((ls.mem (a + i)).w % 256))) with
                fl := (X86Int.aluF .xor hacc ((ls.mem (a + i)).w % 256)).2, rip := 4198245 }
    have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
    have hs4_0 : s4.reg 0 = some (Nat.xor hacc ((ls.mem (a + i)).w % 256)) := reg_set_self_ok' hr3_ok (by decide) _
    have hs4_2 : s4.reg 2 = some 1099511628211 := by
      change (s3.set 0 _).reg 2 = _; rw [reg_set_ne (by decide)]
      change (s2.set 3 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hr2
    let hacc' := (Nat.xor hacc ((ls.mem (a + i)).w % 256) * 1099511628211) % X86Env.W
    have himul_eq : ((Nat.xor hacc ((ls.mem (a + i)).w % 256) % X86Env.W) * (1099511628211 % X86Env.W)) % X86Env.W = hacc' :=
      (Nat.mul_mod _ _ X86Env.W).symm
    let ov := toSigned64 (Nat.xor hacc ((ls.mem (a + i)).w % 256)) * toSigned64 1099511628211 != toSigned64 hacc'
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 0 (some hacc') with fl := ⟨some ov, none, none, some ov⟩, rip := 4198249 } := by
      rw [step_at_vc 12 rfl (by decide : (4198245, (.imul 0 2), 4) ∈ vmTableChunk 12),
        exec_imul hs4_0 hs4_2]
      simp only [himul_eq]; rfl
    let s5 := { s4.set 0 (some hacc') with fl := ⟨some ov, none, none, some ov⟩, rip := 4198249 }
    have hr5_ok : RegsOk s5 := regsOk_set hr4_ok (by decide) _
    have hs5_7 : s5.reg 7 = some (vc.mempReg.base + a * 8) := by
      change (s4.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]
      change (s2.set 3 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hr7
    have hs5_11 : s5.reg 11 = some i := by
      change (s4.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s2.set 3 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hr11
    have hrd_p : rdMem .w64 s5.mem (eaIdx (vc.mempReg.base + a * 8) i 8 0) = some ((ls.mem (a + i)).plen % X86Env.W) := by
      change rdMem .w64 s.mem _ = _; rw [hea_p]; exact rd_mem_plen_at h hai
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 3 (some ((ls.mem (a + i)).plen % X86Env.W)) with rip := 4198253 } := by
      rw [step_at_vc 12 rfl (by decide : (4198249, (.ldIdx .w64 3 7 11 8 0), 4) ∈ vmTableChunk 12),
        exec_ldIdx hs5_7 hs5_11 hrd_p]
    let s6 := { s5.set 3 (some ((ls.mem (a + i)).plen % X86Env.W)) with rip := 4198253 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_3 : s6.reg 3 = some ((ls.mem (a + i)).plen % X86Env.W) := reg_set_self_ok' hr5_ok (by decide) _
    have hs6_10 : s6.reg 10 = some (pacc % X86Env.W) := by
      change (s5.set 3 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s4.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s2.set 3 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hr10
    let pacc' := pacc + (ls.mem (a + i)).plen
    have hadd_p : (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (a + i)).plen % X86Env.W)).1 = pacc' % X86Env.W := by
      dsimp [pacc']; simp only [aluF_add_val, Nat.mod_add_mod, Nat.add_mod_mod]
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 10 (some (pacc' % X86Env.W)) with
              fl := (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (a + i)).plen % X86Env.W)).2, rip := 4198256 } := by
      rw [step_at_vc 12 rfl (by decide : (4198253, (.ofInt (.alu .add 10 3)), 3) ∈ vmTableChunk 12),
        exec_alu hs6_10 hs6_3, ← hadd_p]
    let s7 := { s6.set 10 (some (pacc' % X86Env.W)) with
                fl := (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (a + i)).plen % X86Env.W)).2, rip := 4198256 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hs7_11 : s7.reg 11 = some i := by
      change (s6.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s5.set 3 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs5_11
    have hi1W : i + 1 < X86Env.W := by have : ls.msize < 2 ^ 44 := hmi.1.2.1; omega
    have h08 : X86Vm.step vmImage s7 =
        .ok { s7.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4198260 } := by
      rw [step_at_vc 12 rfl (by decide : (4198256, (.ofInt (.aluI .add 11 1)), 4) ∈ vmTableChunk 12),
        exec_aluI_add_lt hs7_11 (by decide), Nat.mod_eq_of_lt hi1W]
    let s8 := { s7.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4198260 }
    have hr8_ok : RegsOk s8 := regsOk_set hr7_ok (by decide) _
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4198232 } := by
      rw [step_at_vc 12 rfl (by decide : (4198260, (.ofInt (.jmp 4198232)), 2) ∈ vmTableChunk 12), exec_jmp]
    let s9 := { s8 with rip := 4198232 }
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))))))
    have hs9_keep : ∀ r, r ≠ 0 → r ≠ 3 → r ≠ 10 → r ≠ 11 → s9.reg r = s.reg r := by
      intro r h0 h3 h10' h11'
      change (s7.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]
      change (s6.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
      change (s5.set 3 _).reg r = _; rw [reg_set_ne h3.symm]
      change (s4.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s3.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s2.set 3 _).reg r = _; rw [reg_set_ne h3.symm]; rfl
    have hwf9 : vc.wf pr ls s9 :=
      vc.wf_step_misc h rfl hr8_ok
        (by rw [hs9_keep 4 (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.1)
        (by rw [hs9_keep 15 (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.2.1)
    have hhacc' : hacc' = Sporos.fnv ((List.range (i + 1)).map fun k => (ls.mem (a + k)).w % 256) := by
      rw [fnv_range_step, ← hhacc, M_eq_W']
    have hpacc' : pacc' % X86Env.W = (q + (List.range (i + 1)).foldr (fun k acc => (ls.mem (a + k)).plen + acc) 0) % X86Env.W := by
      rw [foldr_range_step (fun k => (ls.mem (a + k)).plen) i]
      dsimp [pacc']
      rw [← Nat.mod_add_mod pacc, hpacc, Nat.mod_add_mod, Nat.add_assoc]
    obtain ⟨sf, hrchf, hripf, hmemf, hkf, hwff, hsf0, hsf10⟩ :=
      ih (i + 1) hacc' pacc' s9 (by omega) rfl hwf9
        (by change (s7.set 11 _).reg 0 = _; rw [reg_set_ne (by decide)]
            change (s6.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
            change (s5.set 3 _).reg 0 = _; rw [reg_set_ne (by decide)]
            exact reg_set_self_ok' hr4_ok (by decide) _)
        (by rw [hs9_keep 1 (by decide) (by decide) (by decide) (by decide)]; exact hr1)
        (by rw [hs9_keep 2 (by decide) (by decide) (by decide) (by decide)]; exact hr2)
        (by rw [hs9_keep 6 (by decide) (by decide) (by decide) (by decide)]; exact hr6)
        (by rw [hs9_keep 7 (by decide) (by decide) (by decide) (by decide)]; exact hr7)
        (by change (s7.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr6_ok (by decide) _)
        (reg_set_self_ok' hr7_ok (by decide) _) hhacc' hpacc'
    exact ⟨sf, Reaches.trans hrch9 hrchf, hripf, hmemf, hkf, hwff, hsf0, hsf10⟩

theorem P_strhash_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_strhash) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .strhash := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hmi := wf_mem_info h
  have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
  have htake : ls.ds.take 2 = [ls.ds[0]!, ls.ds[1]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 =>
      cases hd1 : r0 with
      | nil => simp [hd0, hd1] at hle
      | cons a1 r1 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hlW : ls.ds[0]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have haW : ls.ds[1]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := SporosVmPrimBase.rd_ctx_msize h
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4198151 } := by
    rw [step_at_vc 12 hrip (by decide : (SporosVmBin.P_strhash, (.ofInt (.ld 1 15 296)), 7) ∈ vmTableChunk 12),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4198151 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some ls.ds[1]!.w) with rip := 4198158 } := by
    rw [step_at_vc 12 rfl (by decide : (4198151, (.ofInt (.ld 0 15 312)), 7) ∈ vmTableChunk 12), exec_ld hs1_15 h312]
  let s2 := { s1.set 0 (some ls.ds[1]!.w) with rip := 4198158 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 10 (some ls.msize) with rip := 4198162 } := by
    rw [step_at_vc 12 rfl (by decide : (4198158, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 12), exec_ld hs2_15 hms]
  let s3 := { s2.set 10 (some ls.msize) with rip := 4198162 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03))
  have hwf3 : vc.wf pr ls s3 :=
    vc.wf_step_misc h rfl hr3_ok
      (by change (s2.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s2.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15)
  have hs3_0 : s3.reg 0 = some ls.ds[1]!.w := by
    change (s2.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs3_1 : s3.reg 1 = some ls.ds[0]!.w := by
    change (s2.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s1.set 0 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs3_10 : s3.reg 10 = some ls.msize := reg_set_self_ok' hr2_ok (by decide) _
  have hrc := range_check_spec (rip0 := 4198162) (c0 := 12) (c1 := 12) (c2 := 12) (c3 := 12) (c4 := 12) (c5 := 12)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf3 hs3_0 hs3_1 hs3_10 haW hlW (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases ha_ok : ls.ds[1]!.w + ls.ds[0]!.w ≤ ls.msize
  · rw [if_pos ha_ok] at hrc ⊢
    obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4_ok, hs4_4, hs4_15, hs4_11, hs4_oth⟩ := hrc
    have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 8 (some vc.memwReg.base) with rip := 4198190 } := by
      rw [step_at_vc 12 hrip4 (by decide : (4198186, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk 12),
        exec_ld hs4_15 (by rw [hmem4]; exact rd_ctx_memwBase hwf3), hrip4]
    let s5 := { s4.set 8 (some vc.memwReg.base) with rip := 4198190 }
    have hr5_ok : RegsOk s5 := regsOk_set hr4_ok (by decide) _
    have hs5_8 : s5.reg 8 = some vc.memwReg.base := reg_set_self_ok' hr4_ok (by decide) _
    have hs5_0 : s5.reg 0 = some ls.ds[1]!.w := by change (s4.set 8 _).reg 0 = _; rw [reg_set_ne (by decide), hs4_oth 0 (by decide)]; exact hs3_0
    have hea6 : eaIdx vc.memwReg.base ls.ds[1]!.w 1 0 = vc.memwReg.base + ls.ds[1]!.w := by
      have hltW : vc.memwReg.base + 1 * ls.ds[1]!.w + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 6 (some (vc.memwReg.base + ls.ds[1]!.w)) with rip := 4198194 } := by
      rw [step_at_vc 12 rfl (by decide : (4198190, (.leaIdx 6 8 0 1 0), 4) ∈ vmTableChunk 12), exec_leaIdx hs5_8 hs5_0, hea6]
    let s6 := { s5.set 6 (some (vc.memwReg.base + ls.ds[1]!.w)) with rip := 4198194 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 6 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s4.set 8 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs4_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 9 (some vc.mempReg.base) with rip := 4198198 } := by
      rw [step_at_vc 12 rfl (by decide : (4198194, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk 12),
        exec_ld hs6_15 (by change rd64 s4.mem _ = _; rw [hmem4]; exact rd_ctx_mempBase hwf3)]
    let s7 := { s6.set 9 (some vc.mempReg.base) with rip := 4198198 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hs7_9 : s7.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr6_ok (by decide) _
    have hs7_0 : s7.reg 0 = some ls.ds[1]!.w := by
      change (s6.set 9 _).reg 0 = _; rw [reg_set_ne (by decide)]
      change (s5.set 6 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs5_0
    have hea7 : eaIdx vc.mempReg.base ls.ds[1]!.w 8 0 = vc.mempReg.base + ls.ds[1]!.w * 8 := by
      have hltW : vc.mempReg.base + 8 * ls.ds[1]!.w + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 7 (some (vc.mempReg.base + ls.ds[1]!.w * 8)) with rip := 4198202 } := by
      rw [step_at_vc 12 rfl (by decide : (4198198, (.leaIdx 7 9 0 8 0), 4) ∈ vmTableChunk 12), exec_leaIdx hs7_9 hs7_0, hea7]
    let s8 := { s7.set 7 (some (vc.mempReg.base + ls.ds[1]!.w * 8)) with rip := 4198202 }
    have hr8_ok : RegsOk s8 := regsOk_set hr7_ok (by decide) _
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 0 (some 14695981039346656037) with rip := 4198212 } := by
      rw [step_at_vc 12 rfl (by decide : (4198202, (.ofInt (.movI 0 14695981039346656037)), 10) ∈ vmTableChunk 12),
        exec_movI_lt (by change 14695981039346656037 < 2 ^ 64; omega)]
    let s9 := { s8.set 0 (some 14695981039346656037) with rip := 4198212 }
    have hr9_ok : RegsOk s9 := regsOk_set hr8_ok (by decide) _
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some 1099511628211) with rip := 4198222 } := by
      rw [step_at_vc 12 rfl (by decide : (4198212, (.ofInt (.movI 2 1099511628211)), 10) ∈ vmTableChunk 12),
        exec_movI_lt (by change 1099511628211 < 2 ^ 64; omega)]
    let s10 := { s9.set 2 (some 1099511628211) with rip := 4198222 }
    have hr10_ok : RegsOk s10 := regsOk_set hr9_ok (by decide) _
    have hs10_15 : s10.reg 15 = some ctxBase := by
      change (s9.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s8.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s7.set 7 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s6.set 9 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs6_15
    let q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc
    have hqW : q < X86Env.W := by simp [q, SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)
    have h11 : X86Vm.step vmImage s10 = .ok { s10.set 10 (some q) with rip := 4198229 } := by
      rw [step_at_vc 12 rfl (by decide : (4198222, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk 12),
        exec_ld hs10_15 (by change rd64 s4.mem _ = _; rw [hmem4]; change rd64 s.mem _ = _; rw [htake] at h272; exact h272)]
    let s11 := { s10.set 10 (some q) with rip := 4198229 }
    have hr11_ok : RegsOk s11 := regsOk_set hr10_ok (by decide) _
    have hs11_11 : s11.reg 11 = some (ls.msize - ls.ds[1]!.w) := by
      change (s10.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s9.set 2 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s8.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s7.set 7 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s6.set 9 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s5.set 6 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s4.set 8 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs4_11
    have h12 : X86Vm.step vmImage s11 =
        .ok { s11.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[1]!.w) (ls.msize - ls.ds[1]!.w)).2, rip := 4198232 } := by
      rw [step_at_vc 12 rfl (by decide : (4198229, (.ofInt (.alu .xor 11 11)), 3) ∈ vmTableChunk 12), exec_alu_xor_self hs11_11]
    let s12 := { s11.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[1]!.w) (ls.msize - ls.ds[1]!.w)).2, rip := 4198232 }
    have hr12_ok : RegsOk s12 := regsOk_set hr11_ok (by decide) _
    have hrch12 : Reaches vmImage s s12 :=
      Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h05 (Reaches.step_trans h06
        (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09
          (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step h12)))))))))
    have hs12_keep : ∀ r, r ≠ 0 → r ≠ 2 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s12.reg r = s4.reg r := by
      intro r h0 h2 h6 h7 h8 h9 h10' h11'
      change (s11.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]
      change (s10.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
      change (s9.set 2 _).reg r = _; rw [reg_set_ne h2.symm]
      change (s8.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s7.set 7 _).reg r = _; rw [reg_set_ne h7.symm]
      change (s6.set 9 _).reg r = _; rw [reg_set_ne h9.symm]
      change (s5.set 6 _).reg r = _; rw [reg_set_ne h6.symm]
      change (s4.set 8 _).reg r = _; rw [reg_set_ne h8.symm]
    have hwf12 : vc.wf pr ls s12 :=
      vc.wf_step_misc hwf3 hmem4 hr12_ok
        (by rw [hs12_keep 4 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs4_4)
        (by rw [hs12_keep 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs4_15)
    obtain ⟨s13, hrch13, hrip13, hmem13, hk13, hwf13, hs13_0, hs13_10⟩ :=
      strhash_loop_spec (q := q) ha_ok ls.ds[0]!.w 0 14695981039346656037 q s12 (Nat.zero_add _) rfl hwf12
        (by change (s11.set 11 _).reg 0 = _; rw [reg_set_ne (by decide)]
            change (s10.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
            change (s9.set 2 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr8_ok (by decide) _)
        (by rw [hs12_keep 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hs4_oth 1 (by decide)]; exact hs3_1)
        (by change (s11.set 11 _).reg 2 = _; rw [reg_set_ne (by decide)]
            change (s10.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr9_ok (by decide) _)
        (by change (s11.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s10.set 10 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s9.set 2 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s8.set 0 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s7.set 7 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s6.set 9 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr5_ok (by decide) _)
        (by change (s11.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s10.set 10 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s9.set 2 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s8.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr7_ok (by decide) _)
        (by change (s11.set 11 _).reg 10 = _; rw [reg_set_ne (by decide), Nat.mod_eq_of_lt hqW]; exact reg_set_self_ok' hr10_ok (by decide) _)
        (reg_set_self_ok' hr11_ok (by decide) _) (by simp [Sporos.fnv]) (by simp)
    let hv := Sporos.fnv ((List.range ls.ds[0]!.w).map fun k => (ls.mem (ls.ds[1]!.w + k)).w % 256)
    let pm := (List.range ls.ds[0]!.w).foldr (fun k acc => (ls.mem (ls.ds[1]!.w + k)).plen + acc) 0
    obtain ⟨_, m14, _, _, h14, hwf14, h360_14, hrest14⟩ :=
      vc.wr64_ctx_scratch_slot hwf13 360 hv 4198269 (by decide) (by decide) hs13_0
        (fun {m'} hwr => by
          rw [step_at_vc 12 hrip13 (by decide : (4198262, (.ofInt (.st 15 360 0)), 7) ∈ vmTableChunk 12),
            exec_st hwf13.2.2.2.2.2.1 hs13_0 hwr, hrip13])
    let s14 := { s13 with mem := m14, rip := 4198269 }
    let vc14 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 hv) } }
    obtain ⟨_, m15, _, _, h15', hwf15, h368_15, hrest15⟩ :=
      vc14.wr64_ctx_scratch_slot hwf14 368 ((q + pm) % X86Env.W) 4198276 (by decide) (by decide) hs13_10
        (fun {m'} hwr => by
          rw [step_at_vc 12 rfl (by decide : (4198269, (.ofInt (.st 15 368 10)), 7) ∈ vmTableChunk 12),
            exec_st hwf14.2.2.2.2.2.1 hs13_10 hwr])
    let s15 := { s14 with mem := m15, rip := 4198276 }
    let vc15 := { vc14 with ctxReg := { vc14.ctxReg with data := writeBytes vc14.ctxReg.data 368 (bytes8 ((q + pm) % X86Env.W)) } }
    have hrch15 : Reaches vmImage s s15 :=
      Reaches.trans hrch12 (Reaches.trans hrch13 (Reaches.step_trans h14 (Reaches.step h15')))
    have hhvW : hv < X86Env.W := by rw [← M_eq_W']; exact fnv_lt_M _
    have h280_15 : rd64 s15.mem (ea ctxBase 280) = some 2 :=
      vc15.rd64_ctx_of_eq hwf15 (by decide)
        (by rw [hrest15 280 (Or.inl (by decide)) (by decide), hrest14 280 (Or.inl (by decide)) (by decide)]
            exact ctx_leVal_of_rd64 h (by decide) h280)
    have h288_15 : rd64 s15.mem (ea ctxBase 288) = some 1 :=
      vc15.rd64_ctx_of_eq hwf15 (by decide)
        (by rw [hrest15 288 (Or.inl (by decide)) (by decide), hrest14 288 (Or.inl (by decide)) (by decide)]
            exact ctx_leVal_of_rd64 h (by decide) h288)
    have h360_15 : rd64 s15.mem (ea ctxBase 360) = some hv :=
      vc15.rd64_ctx_of_eq hwf15 (by decide)
        (by rw [hrest15 360 (Or.inl (by decide)) (by decide), h360_14, Nat.mod_eq_of_lt hhvW])
    have h368_15 : rd64 s15.mem (ea ctxBase 368) = some ((q + pm) % X86Env.W) :=
      vc15.rd64_ctx_of_eq hwf15 (by decide)
        (by rw [h368_15, Nat.mod_eq_of_lt (Nat.mod_lt _ (by change 0 < 2 ^ 64; omega))])
    have hwrap_hv : Sporos.wrap hv = hv := by simp [Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt hhvW]
    have hfin := jmp_epilogue0_finish_spec
      (outs := [⟨Sporos.wrap hv, (q + pm) % Sporos.M⟩])
      (step_at_vc 12 rfl (by decide : (4198276, (.ofInt (.jmp SporosVmBin.L_prim_epilogue)), 5) ∈ vmTableChunk 12))
      hwf15 (oracleAdmits_of_reaches hor hrch15) hrs hpcW hle (by simp)
      (by intro v hv'; simp only [List.mem_singleton] at hv'; subst hv'
          exact ⟨by rw [hwrap_hv]; exact hhvW, by rw [M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)⟩)
      h280_15 h288_15
      (by intro i hi; have hi0 : i = 0 := by simpa using hi
          subst hi0; simp only [Nat.zero_mul, Nat.add_zero, List.getElem_cons_zero, hwrap_hv, M_eq_W']
          exact ⟨h360_15, h368_15⟩)
    exact lift_next_outcome hrch15 (by rw [hk13]; exact hk4) hfin
  · rw [if_neg ha_ok] at hrc ⊢
    exact lift_fail_outcome hrch3 rfl hrc

end SporosVmPrimHeap
