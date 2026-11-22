import Bootstrap.SporosVmOpsData3

/-!
# Native x86-64 VM Local Variable Operations (`O_locget`, `O_locset`) & Entrypoint

Simulation proofs over `X86Vm` for `O_locget` and `O_locset`, plus re-export of
`SporosVmOpsData.OpPostOutcome` and all 8 data/stack/local instruction specs
(`O_lit_spec`, `O_str_spec`, `O_pick_spec`, `O_tor_spec`, `O_fromr_spec`,
`O_rfetch_spec`, `O_locget_spec`, `O_locset_spec`).

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

/-! ### Shared top-frame local check & local-slot address/update helpers -/

private theorem step_top_frame_check_local {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk1 chk2 i : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest)
    (h264 : rd64 s.mem (ea ctxBase 264) = some i)
    (hm1 : (a, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk chk1)
    (hm2 : (a + 4, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk chk1)
    (hm3 : (a + 8, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk chk1)
    (hm4 : (a + 12, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk chk1)
    (hm5 : (a + 16, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk chk1)
    (hm6 : (a + 19, (.ofInt (.ld 2 15 264)), 7) ∈ vmTableChunk chk1)
    (hm7 : (a + 26, (.ofInt (.ld 8 0 24)), 4) ∈ vmTableChunk chk1)
    (hm8 : (a + 30, (.ofInt (.cmp 2 8)), 3) ∈ vmTableChunk chk2)
    (hm9 : (a + 33, (.ofInt (.jcc 3 SporosVmBin.L_fail_badLocal)), 6) ∈ vmTableChunk chk2) :
    if i < f.locs.length then
      ∃ s',
        Reaches vmImage s s' ∧ s'.rip = a + 39 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        vc.wf pr ls s' ∧
        s'.reg 0 = some (vc.rsReg.base + rest.length * 32) ∧
        s'.reg 2 = some i ∧
        (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 8 → s'.reg r = s.reg r)
    else
      ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have ⟨_, _, _, _, hrsHi, _, _, _, _, _, hrs_rel, _, _⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, _, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  rw [hrs] at hrsLen hrs_rel
  simp only [RsLocRel] at hrs_rel
  rcases hrs_rel with ⟨_, hrl_len, _, _, _, _, hnlocs, _, _⟩
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hlenW : rest.length + 1 < X86Env.W := by simp only [Region.hi] at hrsHi; omega
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some vc.rsReg.base) with rip := a + 4 } := by
    rw [step_at_vc chk1 hrip hm1, exec_ld h15 hrsBase, hrip]
  let s1 := { s.set 0 (some vc.rsReg.base) with rip := a + 4 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_0 : s1.reg 0 = some vc.rsReg.base := reg_set_self_ok' hr (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hrsLen' : rd64 s1.mem (ea ctxBase 56) = some (rest.length + 1) := by
    simpa [s1, List.length_cons] using hrsLen
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some (rest.length + 1)) with rip := a + 8 } := by
    rw [step_at_vc chk1 rfl hm2, exec_ld hs1_15 hrsLen']
  let s2 := { s1.set 1 (some (rest.length + 1)) with rip := a + 8 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some (rest.length + 1) := reg_set_self_ok' hr1 (by decide) _
  have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
    rw [show rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := a + 12 } := by
    rw [step_at_vc chk1 rfl hm3, exec_aluI hs2_1]
    dsimp only; rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val, hsub1]
  let s3 := { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := a + 12 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_1 : s3.reg 1 = some rest.length := reg_set_self_ok' hr2 (by decide) _
  have hshW : rest.length * 32 < X86Env.W := by simp only [Region.hi] at hrsHi; omega
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 1 (some (rest.length * 32)) with fl := Flags.undef, rip := a + 16 } := by
    rw [step_at_vc chk1 rfl hm4,
      exec_shlI_lt (c := 5) hs3_1 (by rw [show 2 ^ (5 % 64) = 32 from by decide]; exact hshW)]
  let s4 := { s3.set 1 (some (rest.length * 32)) with fl := Flags.undef, rip := a + 16 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_0 : s4.reg 0 = some vc.rsReg.base := by
    change (s3.set 1 _).reg 0 = some _; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 0 = some _; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 0 = some _; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs4_1 : s4.reg 1 = some (rest.length * 32) := reg_set_self_ok' hr3 (by decide) _
  have hrs_off_W : vc.rsReg.base + rest.length * 32 + 32 < X86Env.W := by
    simp only [Region.hi] at hrsHi; omega
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with
            fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := a + 19 } := by
    rw [step_at_vc chk1 rfl hm5, exec_alu hs4_0 hs4_1]
    dsimp only; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  let s5 := { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with
              fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := a + 19 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
  have hs5_0 : s5.reg 0 = some (vc.rsReg.base + rest.length * 32) := reg_set_self_ok' hr4 (by decide) _
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s3.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have h264_5 : rd64 s5.mem (ea ctxBase 264) = some i := h264
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 2 (some i) with rip := a + 26 } := by
    rw [step_at_vc chk1 rfl hm6, exec_ld hs5_15 h264_5]
  let s6 := { s5.set 2 (some i) with rip := a + 26 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
  have hs6_0 : s6.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
    change (s5.set 2 _).reg 0 = some _; rw [reg_set_ne (by decide)]; exact hs5_0
  have hs6_2 : s6.reg 2 = some i := reg_set_self_ok' hr5 (by decide) _
  have hrd_nlocs : rd64 s6.mem (ea (vc.rsReg.base + rest.length * 32) 24) = some f.locs.length := by
    rw [ea_add_eq (by omega)]
    exact rd64_of_mem h.1.2 (vc.mem_rs (s := s) h) (by omega)
      (by simp only [Region.hi] at hrsHi ⊢; omega)
      (by rw [show vc.rsReg.base + rest.length * 32 + 24 - vc.rsReg.base = rest.length * 32 + 24 by omega]; exact hnlocs)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 8 (some f.locs.length) with rip := a + 30 } := by
    rw [step_at_vc chk1 rfl hm7, exec_ld hs6_0 hrd_nlocs]
  let s7 := { s6.set 8 (some f.locs.length) with rip := a + 30 }
  have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) _
  have hs7_2 : s7.reg 2 = some i := by
    change (s6.set 8 _).reg 2 = some i; rw [reg_set_ne (by decide)]; exact hs6_2
  have hs7_8 : s7.reg 8 = some f.locs.length := reg_set_self_ok' hr6 (by decide) _
  have h08 : X86Vm.step vmImage s7 =
      .ok { s7 with fl := (X86Int.aluF .sub i f.locs.length).2, rip := a + 33 } := by
    rw [step_at_vc chk2 rfl hm8, exec_cmp hs7_2 hs7_8]
  let s8 := { s7 with fl := (X86Int.aluF .sub i f.locs.length).2, rip := a + 33 }
  have hrch8 : Reaches vmImage s s8 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08)))))))
  have hoth : ∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 8 → s8.reg r = s.reg r := by
    intro r h0 h1 h2 h8
    change (s6.set 8 _).reg r = s.reg r; rw [reg_set_ne h8.symm]
    change (s5.set 2 _).reg r = s.reg r; rw [reg_set_ne h2.symm]
    change (s4.set 0 _).reg r = s.reg r; rw [reg_set_ne h0.symm]
    change (s3.set 1 _).reg r = s.reg r; rw [reg_set_ne h1.symm]
    change (s2.set 1 _).reg r = s.reg r; rw [reg_set_ne h1.symm]
    change (s1.set 1 _).reg r = s.reg r; rw [reg_set_ne h1.symm]
    change (s.set 0 _).reg r = s.reg r; rw [reg_set_ne h0.symm]
  have hs8_4 : s8.reg 4 = some vc.sp := by rw [hoth 4 (by decide) (by decide) (by decide) (by decide)]; exact h4
  have hs8_15 : s8.reg 15 = some ctxBase := by rw [hoth 15 (by decide) (by decide) (by decide) (by decide)]; exact h15
  by_cases hi : i < f.locs.length
  · rw [if_pos hi]
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := a + 39 } := by
      rw [step_at_vc chk2 rfl hm9, exec_jcc (cond_aluF_sub_ae i f.locs.length)]
      dsimp [s8]; rw [decide_eq_false (by omega)]; rfl
    let s9 := { s8 with rip := a + 39 }
    have hs9_0 : s9.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
      change (s6.set 8 _).reg 0 = some _; rw [reg_set_ne (by decide)]; exact hs6_0
    exact ⟨s9, Reaches.trans hrch8 (Reaches.step h09), rfl, rfl, rfl,
      vc.wf_step_misc h rfl hr7 hs8_4 hs8_15, hs9_0, hs7_2, hoth⟩
  · rw [if_neg hi]
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := errEntry .badLocal } := by
      rw [step_at_vc chk2 rfl hm9, exec_jcc (cond_aluF_sub_ae i f.locs.length)]
      dsimp [s8]; rw [decide_eq_true (by omega)]; rfl
    exact lift_fail_outcome (Reaches.trans hrch8 (Reaches.step h09)) rfl
      (L_fail_from_vc_spec (e := .badLocal) rfl (vc.wf_step_misc h rfl hr7 hs8_4 hs8_15))

private theorem step_loc_addr {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk i : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest) (hi : i < f.locs.length)
    (hr0 : s.reg 0 = some (vc.rsReg.base + rest.length * 32)) (hr2 : s.reg 2 = some i)
    (hm1 : (a, (.ofInt (.ld 9 0 16)), 4) ∈ vmTableChunk chk)
    (hm2 : (a + 4, (.ofInt (.alu .add 9 2)), 3) ∈ vmTableChunk chk)
    (hm3 : (a + 7, (.ofInt (.shlI 9 4)), 4) ∈ vmTableChunk chk)
    (hm4 : (a + 11, (.ofInt (.ld 10 15 72)), 4) ∈ vmTableChunk chk) :
    ∃ s',
      Reaches vmImage s s' ∧ s'.rip = a + 15 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      vc.wf pr ls s' ∧
      s'.reg 9 = some ((rsLocBase rest + i) * 16) ∧
      s'.reg 10 = some vc.locReg.base ∧
      (∀ r, r ≠ 9 → r ≠ 10 → s'.reg r = s.reg r) := by
  have ⟨_, _, _, _, hrsHi, _, _, _, _, hlocHi, hrs_rel, _, _⟩ := (vc.arena_bounds h).2.2
  rw [hrs] at hrs_rel
  simp only [RsLocRel] at hrs_rel
  rcases hrs_rel with ⟨_, hrl_len, hloc_len, _, _, hlb, _, _, _⟩
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hrs_off_W : vc.rsReg.base + rest.length * 32 + 24 < X86Env.W := by
    simp only [Region.hi] at hrsHi; omega
  have hloc_bound : (rsLocBase rest + i) * 16 + 16 ≤ vc.locReg.data.length := by omega
  have hloc_ltW : (rsLocBase rest + i) * 16 < X86Env.W := by simp only [Region.hi] at hlocHi; omega
  have hrd_lb : rd64 s.mem (ea (vc.rsReg.base + rest.length * 32) 16) = some (rsLocBase rest) := by
    rw [ea_add_eq (by omega)]
    exact rd64_of_mem h.1.2 (vc.mem_rs h) (by omega)
      (by simp only [Region.hi] at hrsHi ⊢; omega)
      (by rw [show vc.rsReg.base + rest.length * 32 + 16 - vc.rsReg.base = rest.length * 32 + 16 by omega]; exact hlb)
  have h01 : X86Vm.step vmImage s = .ok { s.set 9 (some (rsLocBase rest)) with rip := a + 4 } := by
    rw [step_at_vc chk hrip hm1, exec_ld hr0 hrd_lb, hrip]
  let s1 := { s.set 9 (some (rsLocBase rest)) with rip := a + 4 }
  have hr1 : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_9 : s1.reg 9 = some (rsLocBase rest) := reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs1_2 : s1.reg 2 = some i := by
    change (s.set 9 _).reg 2 = some i; rw [reg_set_ne (by decide)]; exact hr2
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 9 (some (rsLocBase rest + i)) with
            fl := (X86Int.aluF .add (rsLocBase rest) i).2, rip := a + 7 } := by
    rw [step_at_vc chk rfl hm2, exec_alu hs1_9 hs1_2]
    dsimp only; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  let s2 := { s1.set 9 (some (rsLocBase rest + i)) with
              fl := (X86Int.aluF .add (rsLocBase rest) i).2, rip := a + 7 }
  have hr2' : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_9 : s2.reg 9 = some (rsLocBase rest + i) := reg_set_self_ok' hr1 (by decide) _
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 9 (some ((rsLocBase rest + i) * 16)) with fl := Flags.undef, rip := a + 11 } := by
    rw [step_at_vc chk rfl hm3, exec_shlI_4_lt hs2_9 hloc_ltW]
  let s3 := { s2.set 9 (some ((rsLocBase rest + i) * 16)) with fl := Flags.undef, rip := a + 11 }
  have hr3 : RegsOk s3 := regsOk_set hr2' (by decide) _
  have hs3_9 : s3.reg 9 = some ((rsLocBase rest + i) * 16) := reg_set_self_ok' hr2' (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.2.1
  have hlocBase3 : rd64 s3.mem (ea ctxBase 72) = some vc.locReg.base := rd_ctx_locBase (s := s) h
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some vc.locReg.base) with rip := a + 15 } := by
    rw [step_at_vc chk rfl hm4, exec_ld hs3_15 hlocBase3]
  let s4 := { s3.set 10 (some vc.locReg.base) with rip := a + 15 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hoth : ∀ r, r ≠ 9 → r ≠ 10 → s4.reg r = s.reg r := by
    intro r h9 h10
    change (s3.set 10 _).reg r = s.reg r; rw [reg_set_ne h10.symm]
    change (s2.set 9 _).reg r = s.reg r; rw [reg_set_ne h9.symm]
    change (s1.set 9 _).reg r = s.reg r; rw [reg_set_ne h9.symm]
    change (s.set 9 _).reg r = s.reg r; rw [reg_set_ne h9.symm]
  have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4
    (by rw [hoth 4 (by decide) (by decide)]; exact h.2.2.2.2.1)
    (by rw [hoth 15 (by decide) (by decide)]; exact h.2.2.2.2.2.1)
  exact ⟨s4, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04))),
    rfl, rfl, rfl, hwf4, by change (s3.set 10 _).reg 9 = some _; rw [reg_set_ne (by decide)]; exact hs3_9,
    reg_set_self_ok' hr3 (by decide) _, hoth⟩

theorem wf_write_loc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {i : Nat} {v : SporosProvLen.LVal} {loc' : Region}
    (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest) (hi : i < f.locs.length)
    (hvW : v.w < X86Env.W ∧ v.plen < X86Env.W)
    (hcm' : CodeMem s'.mem) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hloc_eq : loc' = { vc.locReg with data := writeBytes (writeBytes vc.locReg.data
        ((rsLocBase rest + i) * 16) (bytes8 v.w)) ((rsLocBase rest + i) * 16 + 8) (bytes8 v.plen) })
    (hloc_mem : loc' ∈ s'.mem)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.locReg.base → u ∈ s'.mem) :
    { vc with locReg := loc' }.wf pr { ls with rs := { f with locs := f.locs.set i v } :: rest } s' := by
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3, htb1, htb2, htb3, htb4, htb5,
    hkv1, hkv2, hkv3, hkv4, hkv5, hkv6, hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7, hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hloc_b : loc'.base = vc.locReg.base := by rw [hloc_eq]
  have hloc_rw : loc'.rw = true := by rw [hloc_eq]; exact hloc1
  have hloc_len : loc'.data.length = vc.locCap * 16 := by
    rw [hloc_eq]; dsimp only; rw [writeBytes_length, writeBytes_length]; exact hloc3
  have hloc_hi : loc'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, hloc_b, hloc_len, ← hloc3] using hloc5
  have hbase_eq : rsLocBase ({ f with locs := f.locs.set i v } :: rest) = rsLocBase ls.rs := by
    rw [hrs]; simp [rsLocBase]
  have hrsrel' : RsLocRel vc.rsReg.data loc'.data (rsLocBase ({ f with locs := f.locs.set i v } :: rest))
      ({ f with locs := f.locs.set i v } :: rest) := by
    rw [hbase_eq, hrs, hloc_eq]; rw [hrs] at hrsrel
    have htmp := rsLocRel_set_top_loc hrsrel hi hvW.1 hvW.2
    dsimp only at htmp ⊢
    exact htmp
  have hmem_all : ∀ u ∈ ({ vc with locReg := loc' } : VmCtx).regions, u ∈ s'.mem := by
    intro u hu
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hloc_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hwlt_rs' : ∀ fr ∈ ({ f with locs := f.locs.set i v } :: rest), ∀ x ∈ fr.locs, x.w < Sporos.M ∧ x.plen < Sporos.M := by
    intro fr hfr x hx
    rcases List.mem_cons.mp hfr with rfl | hfr
    · rcases mem_set_mem_cases hx with hx0 | rfl
      · exact hwlt.2.2.1 f (by rw [hrs]; exact List.mem_cons_self ..) x hx0
      · exact hvW
    · exact hwlt.2.2.1 fr (by rw [hrs]; exact List.mem_cons_of_mem _ hfr) x hx
  exact ⟨hcm', hbl', hpos', hr', hsp', h15', by simpa [VmCtx.regions, hloc_b] using hnodup, hmem_all,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc_rw, hloc2, hloc_len, hloc4, hloc_hi,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3, htb1, htb2, htb3, htb4, htb5,
    hkv1, hkv2, hkv3, hkv4, hkv5, hkv6, hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, by simpa [hrs] using hsl7,
    hsl8, by rw [hloc_b]; exact hsl9, by rw [hbase_eq]; exact hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23, hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel', by simpa [hrs] using hrsle, by rw [hbase_eq]; exact hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
    ⟨hwlt.1, hwlt.2.1, hwlt_rs', hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩

/-! ### `O_locget` and `O_locset` simulation specs -/

theorem O_locget_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {i : Nat}
    (hrip : s.rip = instrEntry (.locget i)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .locget i)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.locget i)).2) :
    OpPostOutcome pr ls s := by
  have hiW : i < X86Env.W := by
    have hf := flatProgFit_ins_at (vc.code_flat h).2.2.2.1 hnw hpc; rw [hins] at hf; exact hf
  have h264' : rd64 s.mem (ea ctxBase 264) = some i := by
    simpa [instrPair, Nat.mod_eq_of_lt hiW] using h264
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have hchk := step_top_frame_check_local (a := 4199861) (chk1 := 17) (chk2 := 18) hrip h hrs h264'
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  unfold OpPostOutcome
  by_cases hi : i < f.locs.length
  · rw [if_pos hi] at hchk
    obtain ⟨s1, hrch1, hrip1, _, hk1, hwf1, hs1_0, hs1_2, _⟩ := hchk
    have hget : f.locs[i]? = some f.locs[i]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hi]; rfl
    let v := f.locs[i]!
    have hlstep : SporosProvLen.lstep pr ls =
        .next { ls with ds := SporosProvLen.ltag ls.pc v :: ls.ds, rs := { f with pc := f.pc + 1 } :: rest } := by
      unfold SporosProvLen.lstep; rw [hrs]; dsimp only; rw [hfetch]; dsimp only; rw [hget]
      simp [SporosProvLen.LSt.setTop, v]
    rw [hlstep]
    obtain ⟨s2, hrch2, hrip2, _, hk2, hwf2, hs2_9, hs2_10, _⟩ :=
      step_loc_addr (a := 4199900) (chk := 18) hrip1 hwf1 hrs hi hs1_0 hs1_2
        (by decide) (by decide) (by decide) (by decide)
    have ⟨_, _, _, _, _, _, _, _, _, hlocHi, hrs_rel, _, _⟩ := (vc.arena_bounds hwf2).2.2
    rw [hrs] at hrs_rel; simp only [RsLocRel] at hrs_rel
    rcases hrs_rel with ⟨_, _, hloc_len, _, _, _, _, hlocs_rel, _⟩
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hloc_bound : (rsLocBase rest + i) * 16 + 16 ≤ vc.locReg.data.length := by omega
    have hloc_hi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
    have hpair_w : rdMem .w64 s2.mem (eaIdx vc.locReg.base ((rsLocBase rest + i) * 16) 1 0) = some v.w := by
      rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_zero]
      exact rdMem_w64_at hwf2.1.2 (vc.mem_loc hwf2) (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact (hlocs_rel i hi).1)
    have hpair_p : rdMem .w64 s2.mem (eaIdx vc.locReg.base ((rsLocBase rest + i) * 16) 1 8) = some v.plen := by
      rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_assoc]
      exact rdMem_w64_at hwf2.1.2 (vc.mem_loc hwf2) (by omega)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact (hlocs_rel i hi).2)
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 7 (some v.w) with rip := 4199919 } := by
      rw [step_at_vc 18 hrip2 (by decide : (4199915, (.ldIdx .w64 7 10 9 1 0), 4) ∈ vmTableChunk 18),
        exec_ldIdx hs2_10 hs2_9 hpair_w, hrip2]
    let s3 := { s2.set 7 (some v.w) with rip := 4199919 }
    have hr3 : RegsOk s3 := regsOk_set hwf2.2.2.2.1 (by decide) _
    have hs3_10 : s3.reg 10 = some vc.locReg.base := by
      change (s2.set 7 _).reg 10 = some _; rw [reg_set_ne (by decide)]; exact hs2_10
    have hs3_9 : s3.reg 9 = some ((rsLocBase rest + i) * 16) := by
      change (s2.set 7 _).reg 9 = some _; rw [reg_set_ne (by decide)]; exact hs2_9
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 6 (some v.plen) with rip := 4199924 } := by
      rw [step_at_vc 18 rfl (by decide : (4199919, (.ldIdx .w64 6 10 9 1 8), 5) ∈ vmTableChunk 18),
        exec_ldIdx hs3_10 hs3_9 hpair_p]
    let s4 := { s3.set 6 (some v.plen) with rip := 4199924 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
    have hs4_6 : s4.reg 6 = some v.plen := reg_set_self_ok' hr3 (by decide) _
    have hs4_7 : s4.reg 7 = some v.w := by
      change (s3.set 6 _).reg 7 = some v.w; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hwf2.2.2.2.1 (by decide) _
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc hwf2 rfl hr4
      (by change (s3.set 6 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
          change (s2.set 7 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.1)
      (by change (s3.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
          change (s2.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.2.1)
    obtain ⟨s5, hrch5, hrip5, _, hk5, hwf5, hs5_6, hs5_oth⟩ :=
      step_tag_prov (a := 4199924) (chk := 18) (rdst := 6) (rtmp := 11) rfl hwf4 hs4_6
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)
    have hrch5_all : Reaches vmImage s s5 :=
      Reaches.trans hrch1 (Reaches.trans hrch2 (Reaches.step_trans h03 (Reaches.step_trans h04 hrch5)))
    have hvW : v.w < X86Env.W :=
      ((vc.code_flat h).2.2.2.2.2.2.1 f (by rw [hrs]; exact List.mem_cons_self ..) v (getElem!_mem_of_lt hi)).1
    have hpW : (v.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [M_eq_W']; exact lt_W_of_mod _
    rcases step_call_ds_push_vc (step_at_vc 18 hrip5 (by decide : (4199938, (.ofInt (.call 4201446)), 5) ∈ vmTableChunk 18))
        hwf5 (oracleAdmits_of_reaches hor hrch5_all) (by rw [hrip5]; omega)
        (by rw [hs5_oth 7 (by decide) (by decide)]; exact hs4_7) hs5_6 hvW hpW with
      ⟨vc6, s6, hrch6, hrip6, hk6, hor6, hwf6, _⟩ | ⟨hhit, hends⟩
    · rcases adv_pc_finish_vc (a := 4199943) (chk := 18) (by rw [hrip6, hrip5]) hwf6 hor6 hrs hpcW
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide) with
        ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
      · exact Or.inl ⟨sf, Reaches.trans hrch5_all (Reaches.trans hrch6 hrchf),
          by rw [hkf, hk6, hk5]; change s2.k.out = s.k.out; rw [hk2, hk1], horf, hrelf⟩
      · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.trans hrch5_all hrch6) hhit,
          Ends.of_reaches (Reaches.trans hrch5_all hrch6) hends⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch5_all hhit, Ends.of_reaches hrch5_all hends⟩
  · rw [if_neg hi] at hchk
    have hnone : f.locs[i]? = none := List.getElem?_eq_none (by omega)
    have hlstep : SporosProvLen.lstep pr ls = .fail .badLocal := by
      simp [SporosProvLen.lstep, hrs, hfetch, hnone]
    rw [hlstep]; exact hchk

theorem O_locset_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame} {i : Nat}
    (hrip : s.rip = instrEntry (.locset i)) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hnw : f.word < pr.words.length)
    (hpc : f.pc < pr.words[f.word]!.code.length) (hins : pr.words[f.word]!.code[f.pc]! = .locset i)
    (hpcW : f.pc + 1 < X86Env.W)
    (h264 : rd64 s.mem (ea ctxBase 264) = some (instrPair (.locset i)).2) :
    OpPostOutcome pr ls s := by
  have hiW : i < X86Env.W := by
    have hf := flatProgFit_ins_at (vc.code_flat h).2.2.2.1 hnw hpc; rw [hins] at hf; exact hf
  have h264' : rd64 s.mem (ea ctxBase 264) = some i := by
    simpa [instrPair, Nat.mod_eq_of_lt hiW] using h264
  have hfetch := fetch_eq_of_lt hnw hpc hins
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some ls.ds.length) with rip := 4199983 } := by
    rw [step_at_vc 18 hrip (by decide : (SporosVmBin.O_locset, (.ofInt (.ld 12 15 8)), 4) ∈ vmTableChunk 18),
      exec_ld h15 (vc.rd_ctx_dsLen h), hrip]; rfl
  let s1 := { s.set 12 (some ls.ds.length) with rip := 4199983 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) _
  have hs1_12 : s1.reg 12 = some ls.ds.length := reg_set_self_ok' hr (by decide) _
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := 4199986 } := by
    rw [step_at_vc 18 rfl (by decide : (4199983, (.ofInt (.test 12 12)), 3) ∈ vmTableChunk 18),
      exec_test hs1_12 hs1_12]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds.length ls.ds.length).2, rip := 4199986 }
  have hrch2 : Reaches vmImage s s2 := Reaches.step_trans h01 (Reaches.step h02)
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s.set 12 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s.set 12 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  unfold OpPostOutcome
  cases hds : ls.ds with
  | nil =>
    have hlstep : SporosProvLen.lstep pr ls = .fail .underflow := by
      simp [SporosProvLen.lstep, hrs, hfetch, hds]
    rw [hlstep]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := errEntry .underflow } := by
      rw [step_at_vc 18 rfl (by decide : (4199986, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 18),
        exec_jcc (cond_aluF_and_e ls.ds.length ls.ds.length)]
      simp [s2, hds]; rfl
    exact lift_fail_outcome (Reaches.trans hrch2 (Reaches.step h03)) rfl
      (L_fail_from_vc_spec (e := .underflow) rfl (vc.wf_step_misc h rfl hr1 hs2_4 hs2_15))
  | cons v r =>
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199992 } := by
      rw [step_at_vc 18 rfl (by decide : (4199986, (.ofInt (.jcc 4 4201048)), 6) ∈ vmTableChunk 18),
        exec_jcc (cond_aluF_and_e ls.ds.length ls.ds.length)]
      simp [s2, hds]
    let s3 := { s2 with rip := 4199992 }
    have hrch3 : Reaches vmImage s s3 := Reaches.trans hrch2 (Reaches.step h03)
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hs2_4 hs2_15
    have hchk := step_top_frame_check_local (a := 4199992) (chk1 := 18) (chk2 := 18) rfl hwf3 hrs h264'
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    by_cases hi : i < f.locs.length
    · rw [if_pos hi] at hchk
      obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hwf4, hs4_0, hs4_2, hs4_oth⟩ := hchk
      let u := SporosProvLen.ltag ls.pc v
      have hlstep : SporosProvLen.lstep pr ls =
          .next { ls with ds := r, rs := { f with pc := f.pc + 1, locs := f.locs.set i u } :: rest } := by
        simp [SporosProvLen.lstep, hrs, hfetch, hds, hi, SporosProvLen.LSt.setTop, u]
      rw [hlstep]
      have ⟨⟨_, _, hds_len, _, hds_hi, hds_rel, hds_le⟩, _, _⟩ := vc.arena_bounds h
      have hW : X86Env.W = 2 ^ 64 := rfl
      have hds_ltW : ls.ds.length * 16 < X86Env.W := by simp only [Region.hi, hds_len] at hds_hi; omega
      have hlen_eq : ls.ds.length = r.length + 1 := by simp [hds]
      have hs4_12 : s4.reg 12 = some ls.ds.length := by
        rw [hs4_oth 12 (by decide) (by decide) (by decide) (by decide)]; exact hs1_12
      have hsub1 : (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).1 = r.length := by
        rw [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), aluF_sub_val,
          show ls.ds.length + X86Env.W - 1 = r.length + X86Env.W by omega,
          Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      have h05 : X86Vm.step vmImage s4 =
          .ok { s4.set 12 (some r.length) with fl := (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).2, rip := 4200035 } := by
        rw [step_at_vc 18 hrip4 (by decide : (4200031, (.ofInt (.aluI .sub 12 1)), 4) ∈ vmTableChunk 18),
          exec_aluI hs4_12, hrip4]; dsimp only; rw [hsub1]
      let s5 := { s4.set 12 (some r.length) with fl := (X86Int.aluF .sub ls.ds.length (1 % X86Env.W)).2, rip := 4200035 }
      have hr5 : RegsOk s5 := regsOk_set hwf4.2.2.2.1 (by decide) _
      have hs5_12 : s5.reg 12 = some r.length := reg_set_self_ok' hwf4.2.2.2.1 (by decide) _
      have hs5_0 : s5.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
        change (s4.set 12 _).reg 0 = some _; rw [reg_set_ne (by decide)]; exact hs4_0
      have hs5_2 : s5.reg 2 = some i := by
        change (s4.set 12 _).reg 2 = some i; rw [reg_set_ne (by decide)]; exact hs4_2
      have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc hwf4 rfl hr5
        (by change (s4.set 12 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.1)
        (by change (s4.set 12 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.2.1)
      obtain ⟨s6, h06, hrip6, hk6, hregs6, _, hwf6⟩ :=
        wf_dec_dsLen (step_at_vc 18 rfl (by decide : (4200035, (.ofInt (.st 15 8 12)), 4) ∈ vmTableChunk 18))
          hwf5 hds hs5_12
      let vc6 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 r.length) } }
      have hs6_12 : s6.reg 12 = some r.length := by simp only [X86Int.St.reg, hregs6]; exact hs5_12
      have hs6_0 : s6.reg 0 = some (vc.rsReg.base + rest.length * 32) := by simp only [X86Int.St.reg, hregs6]; exact hs5_0
      have hs6_2 : s6.reg 2 = some i := by simp only [X86Int.St.reg, hregs6]; exact hs5_2
      have h07 : X86Vm.step vmImage s6 =
          .ok { s6.set 12 (some (r.length * 16)) with fl := Flags.undef, rip := 4200043 } := by
        rw [step_at_vc 18 hrip6 (by decide : (4200039, (.ofInt (.shlI 12 4)), 4) ∈ vmTableChunk 18),
          exec_shlI_4_lt hs6_12 (by omega), hrip6]
      let s7 := { s6.set 12 (some (r.length * 16)) with fl := Flags.undef, rip := 4200043 }
      have hr7 : RegsOk s7 := regsOk_set hwf6.2.2.2.1 (by decide) _
      have hs7_12 : s7.reg 12 = some (r.length * 16) := reg_set_self_ok' hwf6.2.2.2.1 (by decide) _
      have hs7_15 : s7.reg 15 = some ctxBase := by
        change (s6.set 12 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf6.2.2.2.2.2.1
      have hdsBase7 : rd64 s7.mem (ea ctxBase 0) = some vc.dsReg.base := vc6.rd_ctx_dsBase (s := s6) hwf6
      have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some vc.dsReg.base) with rip := 4200046 } := by
        rw [step_at_vc 18 rfl (by decide : (4200043, (.ofInt (.ld 6 15 0)), 3) ∈ vmTableChunk 18),
          exec_ld hs7_15 hdsBase7]
      let s8 := { s7.set 6 (some vc.dsReg.base) with rip := 4200046 }
      have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
      have hs8_6 : s8.reg 6 = some vc.dsReg.base := reg_set_self_ok' hr7 (by decide) _
      have hs8_12 : s8.reg 12 = some (r.length * 16) := by
        change (s7.set 6 _).reg 12 = some _; rw [reg_set_ne (by decide)]; exact hs7_12
      have hpair := rd64_pairList_top hwf6.1.2 (vc6.mem_ds hwf6) hds_hi hds_rel (k := 0) (by simp [hds])
      have hpair0 : rdMem .w64 s8.mem (eaIdx vc.dsReg.base (r.length * 16) 1 0) = some v.w ∧
          rdMem .w64 s8.mem (eaIdx vc.dsReg.base (r.length * 16) 1 8) = some v.plen := by
        simpa [s8, s7, vc6, hlen_eq, hds] using hpair
      have h09 : X86Vm.step vmImage s8 = .ok { s8.set 7 (some v.w) with rip := 4200050 } := by
        rw [step_at_vc 18 rfl (by decide : (4200046, (.ldIdx .w64 7 6 12 1 0), 4) ∈ vmTableChunk 18),
          exec_ldIdx hs8_6 hs8_12 hpair0.1]
      let s9 := { s8.set 7 (some v.w) with rip := 4200050 }
      have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
      have hs9_6 : s9.reg 6 = some vc.dsReg.base := by
        change (s8.set 7 _).reg 6 = some _; rw [reg_set_ne (by decide)]; exact hs8_6
      have hs9_12 : s9.reg 12 = some (r.length * 16) := by
        change (s8.set 7 _).reg 12 = some _; rw [reg_set_ne (by decide)]; exact hs8_12
      have h10 : X86Vm.step vmImage s9 = .ok { s9.set 13 (some v.plen) with rip := 4200055 } := by
        rw [step_at_vc 18 rfl (by decide : (4200050, (.ldIdx .w64 13 6 12 1 8), 5) ∈ vmTableChunk 18),
          exec_ldIdx hs9_6 hs9_12 hpair0.2]
      let s10 := { s9.set 13 (some v.plen) with rip := 4200055 }
      have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
      have hs10_13 : s10.reg 13 = some v.plen := reg_set_self_ok' hr9 (by decide) _
      have hs10_7 : s10.reg 7 = some v.w := by
        change (s9.set 13 _).reg 7 = some v.w; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr8 (by decide) _
      have hs10_0 : s10.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
        change (s9.set 13 _).reg 0 = some _; rw [reg_set_ne (by decide)]
        change (s8.set 7 _).reg 0 = some _; rw [reg_set_ne (by decide)]
        change (s7.set 6 _).reg 0 = some _; rw [reg_set_ne (by decide)]
        change (s6.set 12 _).reg 0 = some _; rw [reg_set_ne (by decide)]; exact hs6_0
      have hs10_2 : s10.reg 2 = some i := by
        change (s9.set 13 _).reg 2 = some i; rw [reg_set_ne (by decide)]
        change (s8.set 7 _).reg 2 = some i; rw [reg_set_ne (by decide)]
        change (s7.set 6 _).reg 2 = some i; rw [reg_set_ne (by decide)]
        change (s6.set 12 _).reg 2 = some i; rw [reg_set_ne (by decide)]; exact hs6_2
      have hwf10 : vc6.wf pr { ls with ds := r } s10 := vc6.wf_step_misc hwf6 rfl hr10
        (by change (s9.set 13 _).reg 4 = some vc6.sp; rw [reg_set_ne (by decide)]
            change (s8.set 7 _).reg 4 = some vc6.sp; rw [reg_set_ne (by decide)]
            change (s7.set 6 _).reg 4 = some vc6.sp; rw [reg_set_ne (by decide)]
            change (s6.set 12 _).reg 4 = some vc6.sp; rw [reg_set_ne (by decide)]; exact hwf6.2.2.2.2.1)
        (by change (s9.set 13 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s8.set 7 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
            change (s7.set 6 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs7_15)
      obtain ⟨s11, hrch11, hrip11, _, hk11, hwf11, hs11_13, hs11_oth⟩ :=
        step_tag_prov (a := 4200055) (chk := 18) (rdst := 13) (rtmp := 11) rfl hwf10 hs10_13
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide)
      obtain ⟨s12, hrch12, hrip12, _, hk12, hwf12, hs12_9, hs12_10, hs12_oth⟩ :=
        step_loc_addr (a := 4200069) (chk := 18) hrip11 hwf11 hrs hi
          (by rw [hs11_oth 0 (by decide) (by decide)]; exact hs10_0)
          (by rw [hs11_oth 2 (by decide) (by decide)]; exact hs10_2)
          (by decide) (by decide) (by decide) (by decide)
      have hs12_7 : s12.reg 7 = some u.w := by
        rw [hs12_oth 7 (by decide) (by decide), hs11_oth 7 (by decide) (by decide)]; exact hs10_7
      have hs12_13 : s12.reg 13 = some u.plen := by
        rw [hs12_oth 13 (by decide) (by decide)]; exact hs11_13
      have ⟨_, _, _, _, _, hlocRw, _, _, _, hlocHi, hrs_rel, _, _⟩ := (vc6.arena_bounds hwf12).2.2
      rw [hrs] at hrs_rel; simp only [RsLocRel] at hrs_rel
      rcases hrs_rel with ⟨_, _, hloc_len, _, _, _, _, _, _⟩
      dsimp only [vc6] at hlocRw hlocHi hloc_len
      have hloc_bound : (rsLocBase rest + i) * 16 + 16 ≤ vc.locReg.data.length := by omega
      have hloc_hi' : vc.locReg.base + vc.locReg.data.length ≤ 2 ^ 47 := hlocHi
      have hea0 : eaIdx vc.locReg.base ((rsLocBase rest + i) * 16) 1 0 =
          vc.locReg.base + (rsLocBase rest + i) * 16 := by
        rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_zero]
      have hea8 : eaIdx vc.locReg.base ((rsLocBase rest + i) * 16) 1 8 =
          vc.locReg.base + (rsLocBase rest + i) * 16 + 8 := by
        rw [eaIdx_add_eq (by omega), Nat.one_mul]
      obtain ⟨_, m13, hwr13, _, _, hmwf13, hloc13, hoth13⟩ :=
        wrMem_w64_region hwf12.1.2 (vc6.mem_loc hwf12) hlocRw (a := vc.locReg.base + (rsLocBase rest + i) * 16)
          (Nat.le_add_right _ _) (by simp only [Region.hi, vc6]; omega) (v := u.w)
      rw [Nat.add_sub_cancel_left] at hloc13
      have h13 : X86Vm.step vmImage s12 = .ok { s12 with mem := m13, rip := 4200088 } := by
        rw [step_at_vc 18 hrip12 (by decide : (4200084, (.stIdx .w64 10 9 1 0 7), 4) ∈ vmTableChunk 18),
          exec_stIdx hs12_10 hs12_9 hs12_7 (by rw [hea0]; exact hwr13), hrip12]
      let s13 := { s12 with mem := m13, rip := 4200088 }
      let loc1 := { vc.locReg with data := writeBytes vc.locReg.data ((rsLocBase rest + i) * 16) (bytes8 u.w) }
      obtain ⟨_, m14, hwr14, _, _, hmwf14, hloc14, hoth14⟩ :=
        wrMem_w64_region hmwf13 hloc13 hlocRw (a := vc.locReg.base + (rsLocBase rest + i) * 16 + 8)
          (by dsimp [vc6]; omega) (by simp only [Region.hi, vc6, writeBytes_length]; omega) (v := u.plen)
      have hsub14 : vc.locReg.base + (rsLocBase rest + i) * 16 + 8 - loc1.base = (rsLocBase rest + i) * 16 + 8 := by
        dsimp [loc1]; omega
      rw [hsub14] at hloc14
      have hs13_10 : s13.reg 10 = some vc.locReg.base := hs12_10
      have hs13_9 : s13.reg 9 = some ((rsLocBase rest + i) * 16) := hs12_9
      have hs13_13 : s13.reg 13 = some u.plen := hs12_13
      have h14 : X86Vm.step vmImage s13 = .ok { s13 with mem := m14, rip := 4200093 } := by
        rw [step_at_vc 18 rfl (by decide : (4200088, (.stIdx .w64 10 9 1 8 13), 5) ∈ vmTableChunk 18),
          exec_stIdx hs13_10 hs13_9 hs13_13 (by rw [hea8]; exact hwr14)]
      let s14 := { s13 with mem := m14, rip := 4200093 }
      have hcm13 := hwf12.1.wr64_preserve hwr13
      have hbl13 := bytesLt_wr64 hwf12.2.1 hwr13
      have hpos13 := len_pos_wrN hwf12.2.2.1 hwr13
      have huW : u.w < X86Env.W ∧ u.plen < X86Env.W :=
        ⟨by change v.w < X86Env.W; exact (wf_ds_lt_W h (by rw [hds]; exact List.mem_cons_self ..)).1,
         by dsimp [u, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
      have hwf14 := wf_write_loc (vc := vc6) (s' := s14) hwf12 hrs hi huW
        (hcm13.wr64_preserve hwr14) (bytesLt_wr64 hbl13 hwr14) (len_pos_wrN hpos13 hwr14)
        hwf12.2.2.2.1 hwf12.2.2.2.2.1 hwf12.2.2.2.2.2.1 rfl hloc14
        (fun w hw hne => hoth14 w (hoth13 w hw hne) hne)
      have hrch14_all : Reaches vmImage s s14 :=
        Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10
            (Reaches.trans hrch11 (Reaches.trans hrch12 (Reaches.step_trans h13 (Reaches.step h14)))))))))))
      have hk14 : s14.k = s.k := by
        change s12.k = s.k; rw [hk12, hk11]; change s6.k = s.k; rw [hk6]; exact hk4
      exact lift_next_outcome hrch14_all hk14
        (adv_pc_finish_vc (a := 4200093) (chk := 18) rfl hwf14 (oracleAdmits_of_reaches hor hrch14_all)
          rfl hpcW (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide))
    · rw [if_neg hi] at hchk
      have hlstep : SporosProvLen.lstep pr ls = .fail .badLocal := by
        simp [SporosProvLen.lstep, hrs, hfetch, hds, hi]
      rw [hlstep]
      exact lift_fail_outcome hrch3 rfl hchk

end SporosVmOpsData
