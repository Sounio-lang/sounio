import Bootstrap.SporosVmPrimMem

/-!
# Native x86-64 VM Block Memory Primitive Proofs (`P_copy`, `P_fill`, `P_mcmp`)

Simulation proofs over `X86Vm` for the 3 block-memory primitives (`copy`, `fill`,
`mcmp`), plus reusable `rd_ctx_msize`, `memRel_set_byte`, `all_range_step`,
`foldr_range_step`, and `mem_cmp_finish_spec` (shared with `P_streq` in
`SporosVmPrimHeap.lean`).

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmPrimBlock

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase SporosVmPrimMem

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### Reusable memory-update, range-fold, and comparison-loop helpers -/

theorem rd_ctx_msize {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 112) = some ls.msize := by
  have hs112 : leVal ((vc.ctxReg.data.drop 112).take 8) = ls.msize := by
    obtain ⟨_, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsl14, _⟩ := h
    exact hsl14
  exact vc.rd64_ctx_of_eq h (by decide) hs112

theorem rd_ctx_hp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 120) = some ls.hp := by
  have hs120 : leVal ((vc.ctxReg.data.drop 120).take 8) = ls.hp := by
    obtain ⟨_, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsl15, _⟩ := h
    exact hsl15
  exact vc.rd64_ctx_of_eq h (by decide) hs120

theorem memRel_set_byte {mw mp : List Nat} {msize k v q : Nat} {mem : SporosProvLen.LMem}
    (hrel : MemRel mw mp msize mem) (hk : k < msize) :
    MemRel (writeBytes mw k [v % 256]) (writeBytes mp (k * 8) (bytes8 q)) msize
      (fun x => if x = k then ⟨v, q⟩ else mem x) := by
  obtain ⟨hmw, hmp, hget⟩ := hrel
  refine ⟨by rw [writeBytes_length]; exact hmw, by rw [writeBytes_length]; exact hmp, ?_⟩
  intro x hx
  by_cases hxk : x = k
  · subst hxk
    refine ⟨?_, ?_⟩
    · rw [writeBytes_at_self _ _ _ (Nat.mod_lt _ (by decide)) (by omega)]
      simp
    · rw [leVal_writeBytes_same _ _ _ (by omega)]
      simp
  · have hprev := hget x hx
    refine ⟨?_, ?_⟩
    · rw [writeBytes_getElem?_disjoint _ _ _ _ (by simp; omega)]
      simpa [hxk] using hprev.1
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _
        (by rw [bytes8_length]; omega) (by omega)]
      simpa [hxk] using hprev.2

theorem all_range_step (f : Nat → Bool) (i : Nat) :
    (List.range (i + 1)).all f = ((List.range i).all f && f i) := by
  rw [List.range_succ, List.all_append]; simp

private theorem foldr_add_shift (g : Nat → Nat) (xs : List Nat) (a : Nat) :
    xs.foldr (fun k acc => g k + acc) a = xs.foldr (fun k acc => g k + acc) 0 + a := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Nat.add_assoc]

theorem foldr_range_step (g : Nat → Nat) (i : Nat) :
    (List.range (i + 1)).foldr (fun k acc => g k + acc) 0 =
      (List.range i).foldr (fun k acc => g k + acc) 0 + g i := by
  rw [List.range_succ, List.foldr_append]
  simpa using foldr_add_shift g (List.range i) (g i)

theorem rd_mem_byte_at {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {idx : Nat} (hidx : idx < ls.msize) :
    rdMem .w8 s.mem (vc.memwReg.base + idx) = some ((ls.mem idx).w % 256) := by
  have hmi := wf_mem_info h
  have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
  have hrel := hmi.2.2.1.2.2 idx hidx
  exact rdMem_w8_at h.1.2 (vc.mem_memw h) (Nat.le_add_right _ _)
    (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hrel.1)

theorem rd_mem_plen_at {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {idx : Nat} (hidx : idx < ls.msize) :
    rdMem .w64 s.mem (vc.mempReg.base + idx * 8) = some ((ls.mem idx).plen % X86Env.W) := by
  have hmi := wf_mem_info h
  have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
  have hrel := hmi.2.2.1.2.2 idx hidx
  exact rdMem_w64_at h.1.2 (vc.mem_memp h) (Nat.le_add_right _ _)
    (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hrel.2)

theorem wr_mem_pair_step {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {k v q : Nat}
    (h : vc.wf pr ls s) (hk : k < ls.msize) (hvM : v < Sporos.M) (hqM : q < Sporos.M) :
    let mw' := { vc.memwReg with data := writeBytes vc.memwReg.data k [v % 256] }
    let mp' := { vc.mempReg with data := writeBytes vc.mempReg.data (k * 8) (bytes8 q) }
    ∃ m1 m2,
      wrMem .w8 s.mem (vc.memwReg.base + k) v = some m1 ∧
      wrMem .w64 m1 (vc.mempReg.base + k * 8) q = some m2 ∧
      CodeMem m2 ∧ BytesLt m2 ∧ (∀ u ∈ m2, 0 < u.data.length) ∧
      mw' ∈ m2 ∧ mp' ∈ m2 ∧
      (∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u.base ≠ vc.mempReg.base → u ∈ m2) ∧
      ({ vc with memwReg := mw', mempReg := mp' }.wf pr
        { ls with mem := fun x => if x = k then ⟨v, q⟩ else ls.mem x }
        { s with mem := m2 }) := by
  have hmi := wf_mem_info h
  have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
  have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
  have hne_mw_mp : vc.memwReg.base ≠ vc.mempReg.base :=
    vc.ne_bases h (i := 10) (j := 11) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  obtain ⟨idx1, m1, hwr1, _, hm1_eq, hm1wf, hmw1, hoth1⟩ :=
    wrMem_w8_region h.1.2 (vc.mem_memw h) hmi.1.1 (a := vc.memwReg.base + k) (Nat.le_add_right _ _)
      (by simp only [Region.hi]; omega) (v := v)
  rw [Nat.add_sub_cancel_left] at hmw1
  have hmp_m1 : vc.mempReg ∈ m1 := hoth1 vc.mempReg (vc.mem_memp h) (Ne.symm hne_mw_mp)
  obtain ⟨idx2, m2, hwr2, _, hm2_eq, hm2wf, hmp2, hoth2⟩ :=
    wrMem_w64_region hm1wf hmp_m1 hmi.2.1.1 (a := vc.mempReg.base + k * 8) (Nat.le_add_right _ _)
      (by simp only [Region.hi]; omega) (v := q)
  rw [Nat.add_sub_cancel_left] at hmp2
  have hcm1 : CodeMem m1 := by
    refine ⟨?_, hm1wf⟩
    have hne : codeRegion.base ≠ vc.memwReg.base :=
      vc.ne_bases h (i := 0) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    exact hoth1 codeRegion h.1.1 hne
  have hcm2 : CodeMem m2 := hcm1.wr64_preserve hwr2
  have hbl1 : BytesLt m1 := bytesLt_wrMem h.2.1 h.1.2 (vc.mem_memw h) hmi.1.1
    (Nat.le_add_right _ _) (by simp only [Region.hi]; omega) hwr1
  have hbl2 : BytesLt m2 := bytesLt_wr64 hbl1 hwr2
  have hpos1 : ∀ u ∈ m1, 0 < u.data.length := by
    intro u hu; rw [hm1_eq] at hu
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp hu
    by_cases hje : j = idx1
    · subst hje; simp only [List.getElem_set_self, writeBytes_length]; exact h.2.2.1 vc.memwReg (vc.mem_memw h)
    · rw [List.getElem_set_ne (Ne.symm hje)]; exact h.2.2.1 _ (List.getElem_mem _)
  have hpos2 : ∀ u ∈ m2, 0 < u.data.length := by
    intro u hu; rw [hm2_eq] at hu
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp hu
    by_cases hje : j = idx2
    · subst hje; simp only [List.getElem_set_self, writeBytes_length]; exact hpos1 vc.mempReg hmp_m1
    · rw [List.getElem_set_ne (Ne.symm hje)]; exact hpos1 _ (List.getElem_mem _)
  have hmw2 : { vc.memwReg with data := writeBytes vc.memwReg.data k [v % 256] } ∈ m2 :=
    hoth2 _ hmw1 hne_mw_mp
  have hoth12 : ∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u.base ≠ vc.mempReg.base → u ∈ m2 :=
    fun u hu h1 h2 => hoth2 u (hoth1 u hu h1) h2
  have hrel' := memRel_set_byte hmi.2.2.1 hk (v := v) (q := q)
  have hwlt_old := (vc.code_flat h).2.2.2.2.2.2.2.1
  have hwlt' : ∀ a, (if a = k then (⟨v, q⟩ : SporosProvLen.LVal) else ls.mem a).w < Sporos.M ∧
      (if a = k then (⟨v, q⟩ : SporosProvLen.LVal) else ls.mem a).plen < Sporos.M := by
    intro a; by_cases ha : a = k
    · simp [ha, hvM, hqM]
    · simp [ha]; exact hwlt_old a
  have hwf' := wf_write_mem
    (mw' := { vc.memwReg with data := writeBytes vc.memwReg.data k [v % 256] })
    (mp' := { vc.mempReg with data := writeBytes vc.mempReg.data (k * 8) (bytes8 q) })
    (s' := { s with mem := m2 }) h hcm2 hbl2 hpos2 h.2.2.2.1
    h.2.2.2.2.1 h.2.2.2.2.2.1 rfl hmi.1.1 (writeBytes_length _ _ _)
    rfl hmi.2.1.1 (writeBytes_length _ _ _) hmw2 hmp2 hoth12 hrel' hwlt'
  exact ⟨m1, m2, hwr1, hwr2, hcm2, hbl2, hpos2, hmw2, hmp2, hoth12, hwf'⟩

/-! ### Shared comparison-loop & epilogue (`P_mcmp` and `P_streq`) -/

theorem mem_cmp_loop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {x y n q rip_loop chk : Nat}
    (hx_in : x + n ≤ ls.msize) (hy_in : y + n ≤ ls.msize)
    (hm_cmp : (rip_loop, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk chk)
    (hm_jae : (rip_loop + 3, (.ofInt (.jcc 3 (rip_loop + 43))), 2) ∈ vmTableChunk chk)
    (hm_ldx : (rip_loop + 5, (.ldIdx .w8 0 6 11 1 0), 5) ∈ vmTableChunk chk)
    (hm_ldy : (rip_loop + 10, (.ldIdx .w8 2 7 11 1 0), 5) ∈ vmTableChunk chk)
    (hm_cmpb : (rip_loop + 15, (.ofInt (.cmp 0 2)), 3) ∈ vmTableChunk chk)
    (hm_cmov : (rip_loop + 18, (.cmovcc 5 3 14), 4) ∈ vmTableChunk chk)
    (hm_ldpx : (rip_loop + 22, (.ldIdx .w64 0 12 11 8 0), 4) ∈ vmTableChunk chk)
    (hm_addx : (rip_loop + 26, (.ofInt (.alu .add 10 0)), 3) ∈ vmTableChunk chk)
    (hm_ldpy : (rip_loop + 29, (.ldIdx .w64 0 13 11 8 0), 5) ∈ vmTableChunk chk)
    (hm_addy : (rip_loop + 34, (.ofInt (.alu .add 10 0)), 3) ∈ vmTableChunk chk)
    (hm_inc : (rip_loop + 37, (.ofInt (.aluI .add 11 1)), 4) ∈ vmTableChunk chk)
    (hm_jmp : (rip_loop + 41, (.ofInt (.jmp rip_loop)), 2) ∈ vmTableChunk chk) :
    ∀ (rem i : Nat) (bacc : Bool) (pacc : Nat) (s : X86Vm.St),
      i + rem = n →
      s.rip = rip_loop → vc.wf pr ls s →
      s.reg 1 = some n → s.reg 3 = some (Sporos.bool bacc) →
      s.reg 6 = some (vc.memwReg.base + x) → s.reg 7 = some (vc.memwReg.base + y) →
      s.reg 10 = some (pacc % X86Env.W) → s.reg 11 = some i →
      s.reg 12 = some (vc.mempReg.base + x * 8) → s.reg 13 = some (vc.mempReg.base + y * 8) →
      s.reg 14 = some 0 →
      bacc = (List.range i).all (fun k => (ls.mem (x + k)).w % 256 == (ls.mem (y + k)).w % 256) →
      pacc % X86Env.W = (q + (List.range i).foldr (fun k acc => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen + acc) 0) % X86Env.W →
      ∃ s', Reaches vmImage s s' ∧ s'.rip = rip_loop + 43 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        vc.wf pr ls s' ∧
        s'.reg 3 = some (Sporos.bool ((List.range n).all fun k => (ls.mem (x + k)).w % 256 == (ls.mem (y + k)).w % 256)) ∧
        s'.reg 10 = some ((q + (List.range n).foldr (fun k acc => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen + acc) 0) % X86Env.W) := by
  intro rem
  induction rem with
  | zero =>
    intro i bacc pacc s hirem hrip h hr1 hr3 hr6 hr7 hr10 hr11 hr12 hr13 hr14 hbacc hpacc
    have hi_eq : i = n := by omega
    subst hi_eq
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i i).2, rip := rip_loop + 3 } := by
      rw [step_at_vc chk hrip hm_cmp, exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i i).2, rip := rip_loop + 3 }
    have hcae : X86Int.cond s1.fl 3 = some true := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_true (Nat.le_refl _)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := rip_loop + 43 } := by
      rw [step_at_vc chk rfl hm_jae, exec_jcc hcae]; rfl
    let s2 := { s1 with rip := rip_loop + 43 }
    refine ⟨s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl,
      vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1, ?_, ?_⟩
    · simpa [s2, s1, X86Int.St.reg, hbacc] using hr3
    · simpa [s2, s1, X86Int.St.reg, hpacc] using hr10
  | succ rem ih =>
    intro i bacc pacc s hirem hrip h hr1 hr3 hr6 hr7 hr10 hr11 hr12 hr13 hr14 hbacc hpacc
    have hmi := wf_mem_info h
    have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
    have hW44 : 2 ^ 44 < X86Env.W := by change 2 ^ 44 < 2 ^ 64; omega
    have hxi : x + i < ls.msize := by omega
    have hyi : y + i < ls.msize := by omega
    have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i n).2, rip := rip_loop + 3 } := by
      rw [step_at_vc chk hrip hm_cmp, exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i n).2, rip := rip_loop + 3 }
    have hcae : X86Int.cond s1.fl 3 = some false := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_false (by omega)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := rip_loop + 5 } := by
      rw [step_at_vc chk rfl hm_jae, exec_jcc hcae]; rfl
    let s2 := { s1 with rip := rip_loop + 5 }
    have hea_x : eaIdx (vc.memwReg.base + x) i 1 0 = vc.memwReg.base + (x + i) := by
      have hltW : vc.memwReg.base + x + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_x : rdMem .w8 s2.mem (eaIdx (vc.memwReg.base + x) i 1 0) = some ((ls.mem (x + i)).w % 256) := by
      change rdMem .w8 s.mem _ = _; rw [hea_x]; exact rd_mem_byte_at h hxi
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some ((ls.mem (x + i)).w % 256)) with rip := rip_loop + 10 } := by
      rw [step_at_vc chk rfl hm_ldx, exec_ldIdx (s := s2) hr6 hr11 hrd_x]
    let s3 := { s2.set 0 (some ((ls.mem (x + i)).w % 256)) with rip := rip_loop + 10 }
    have hr3_ok : RegsOk s3 := regsOk_set h.2.2.2.1 (by decide) _
    have hs3_0 : s3.reg 0 = some ((ls.mem (x + i)).w % 256) := reg_set_self_ok' h.2.2.2.1 (by decide) _
    have hs3_7 : s3.reg 7 = some (vc.memwReg.base + y) := by
      change (s2.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hr7
    have hs3_11 : s3.reg 11 = some i := by
      change (s2.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hr11
    have hea_y : eaIdx (vc.memwReg.base + y) i 1 0 = vc.memwReg.base + (y + i) := by
      have hltW : vc.memwReg.base + y + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_y : rdMem .w8 s3.mem (eaIdx (vc.memwReg.base + y) i 1 0) = some ((ls.mem (y + i)).w % 256) := by
      change rdMem .w8 s.mem _ = _; rw [hea_y]; exact rd_mem_byte_at h hyi
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 2 (some ((ls.mem (y + i)).w % 256)) with rip := rip_loop + 15 } := by
      rw [step_at_vc chk rfl hm_ldy, exec_ldIdx hs3_7 hs3_11 hrd_y]
    let s4 := { s3.set 2 (some ((ls.mem (y + i)).w % 256)) with rip := rip_loop + 15 }
    have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
    have hs4_0 : s4.reg 0 = some ((ls.mem (x + i)).w % 256) := by
      change (s3.set 2 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs3_0
    have hs4_2 : s4.reg 2 = some ((ls.mem (y + i)).w % 256) := reg_set_self_ok' hr3_ok (by decide) _
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4 with fl := (X86Int.aluF .sub ((ls.mem (x + i)).w % 256) ((ls.mem (y + i)).w % 256)).2, rip := rip_loop + 18 } := by
      rw [step_at_vc chk rfl hm_cmpb, exec_cmp hs4_0 hs4_2]
    let s5 := { s4 with fl := (X86Int.aluF .sub ((ls.mem (x + i)).w % 256) ((ls.mem (y + i)).w % 256)).2, rip := rip_loop + 18 }
    have hbxW : (ls.mem (x + i)).w % 256 < X86Env.W := by have := Nat.mod_lt (ls.mem (x + i)).w (by decide : 0 < 256); omega
    have hbyW : (ls.mem (y + i)).w % 256 < X86Env.W := by have := Nat.mod_lt (ls.mem (y + i)).w (by decide : 0 < 256); omega
    have hcne : X86Int.cond s5.fl 5 = some (decide ((ls.mem (x + i)).w % 256 ≠ (ls.mem (y + i)).w % 256)) := by
      dsimp [s5]; exact cond_aluF_sub_ne _ _ hbxW hbyW
    have hs5_3 : s5.reg 3 = some (Sporos.bool bacc) := by
      change (s3.set 2 _).reg 3 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 3 = _; rw [reg_set_ne (by decide)]; exact hr3
    have hs5_14 : s5.reg 14 = some 0 := by
      change (s3.set 2 _).reg 14 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 14 = _; rw [reg_set_ne (by decide)]; exact hr14
    let bacc' : Bool := bacc && ((ls.mem (x + i)).w % 256 == (ls.mem (y + i)).w % 256)
    have hcmov_val : (if decide ((ls.mem (x + i)).w % 256 ≠ (ls.mem (y + i)).w % 256) then 0 else Sporos.bool bacc) =
        Sporos.bool bacc' := by
      dsimp [bacc']
      by_cases heq : (ls.mem (x + i)).w % 256 = (ls.mem (y + i)).w % 256
      · simp [heq]
      · simp [heq, Sporos.bool]
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 3 (some (Sporos.bool bacc')) with rip := rip_loop + 22 } := by
      rw [step_at_vc chk rfl hm_cmov, exec_cmovcc hcne hs5_14 hs5_3, hcmov_val]
    let s6 := { s5.set 3 (some (Sporos.bool bacc')) with rip := rip_loop + 22 }
    have hr6_ok : RegsOk s6 := regsOk_set hr4_ok (by decide) _
    have hs6_12 : s6.reg 12 = some (vc.mempReg.base + x * 8) := by
      change (s5.set 3 _).reg 12 = _; rw [reg_set_ne (by decide)]
      change (s3.set 2 _).reg 12 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 12 = _; rw [reg_set_ne (by decide)]; exact hr12
    have hs6_11 : s6.reg 11 = some i := by
      change (s5.set 3 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s3.set 2 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs3_11
    have hea_px : eaIdx (vc.mempReg.base + x * 8) i 8 0 = vc.mempReg.base + (x + i) * 8 := by
      have hltW : vc.mempReg.base + x * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_px : rdMem .w64 s6.mem (eaIdx (vc.mempReg.base + x * 8) i 8 0) = some ((ls.mem (x + i)).plen % X86Env.W) := by
      change rdMem .w64 s.mem _ = _; rw [hea_px]; exact rd_mem_plen_at h hxi
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 0 (some ((ls.mem (x + i)).plen % X86Env.W)) with rip := rip_loop + 26 } := by
      rw [step_at_vc chk rfl hm_ldpx, exec_ldIdx hs6_12 hs6_11 hrd_px]
    let s7 := { s6.set 0 (some ((ls.mem (x + i)).plen % X86Env.W)) with rip := rip_loop + 26 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hs7_0 : s7.reg 0 = some ((ls.mem (x + i)).plen % X86Env.W) := reg_set_self_ok' hr6_ok (by decide) _
    have hs7_10 : s7.reg 10 = some (pacc % X86Env.W) := by
      change (s6.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s5.set 3 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s3.set 2 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hr10
    have hadd1_eq : (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (x + i)).plen % X86Env.W)).1 =
        (pacc + (ls.mem (x + i)).plen) % X86Env.W := by
      simp only [aluF_add_val, Nat.mod_add_mod, Nat.add_mod_mod]
    have h08 : X86Vm.step vmImage s7 =
        .ok { s7.set 10 (some ((pacc + (ls.mem (x + i)).plen) % X86Env.W)) with
              fl := (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (x + i)).plen % X86Env.W)).2, rip := rip_loop + 29 } := by
      rw [step_at_vc chk rfl hm_addx, exec_alu hs7_10 hs7_0, ← hadd1_eq]
    let s8 := { s7.set 10 (some ((pacc + (ls.mem (x + i)).plen) % X86Env.W)) with
                fl := (X86Int.aluF .add (pacc % X86Env.W) ((ls.mem (x + i)).plen % X86Env.W)).2, rip := rip_loop + 29 }
    have hr8_ok : RegsOk s8 := regsOk_set hr7_ok (by decide) _
    have hs8_10 : s8.reg 10 = some ((pacc + (ls.mem (x + i)).plen) % X86Env.W) := reg_set_self_ok' hr7_ok (by decide) _
    have hs8_13 : s8.reg 13 = some (vc.mempReg.base + y * 8) := by
      change (s7.set 10 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s6.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s5.set 3 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s3.set 2 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]; exact hr13
    have hs8_11 : s8.reg 11 = some i := by
      change (s7.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s6.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs6_11
    have hea_py : eaIdx (vc.mempReg.base + y * 8) i 8 0 = vc.mempReg.base + (y + i) * 8 := by
      have hltW : vc.mempReg.base + y * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_py : rdMem .w64 s8.mem (eaIdx (vc.mempReg.base + y * 8) i 8 0) = some ((ls.mem (y + i)).plen % X86Env.W) := by
      change rdMem .w64 s.mem _ = _; rw [hea_py]; exact rd_mem_plen_at h hyi
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 0 (some ((ls.mem (y + i)).plen % X86Env.W)) with rip := rip_loop + 34 } := by
      rw [step_at_vc chk rfl hm_ldpy, exec_ldIdx hs8_13 hs8_11 hrd_py]
    let s9 := { s8.set 0 (some ((ls.mem (y + i)).plen % X86Env.W)) with rip := rip_loop + 34 }
    have hr9_ok : RegsOk s9 := regsOk_set hr8_ok (by decide) _
    have hs9_0 : s9.reg 0 = some ((ls.mem (y + i)).plen % X86Env.W) := reg_set_self_ok' hr8_ok (by decide) _
    have hs9_10 : s9.reg 10 = some ((pacc + (ls.mem (x + i)).plen) % X86Env.W) := by
      change (s8.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hs8_10
    let pacc' : Nat := pacc + (ls.mem (x + i)).plen + (ls.mem (y + i)).plen
    have hadd2_eq : (X86Int.aluF .add ((pacc + (ls.mem (x + i)).plen) % X86Env.W) ((ls.mem (y + i)).plen % X86Env.W)).1 =
        pacc' % X86Env.W := by
      dsimp [pacc']; simp only [aluF_add_val, Nat.mod_add_mod, Nat.add_mod_mod]
    have h10 : X86Vm.step vmImage s9 =
        .ok { s9.set 10 (some (pacc' % X86Env.W)) with
              fl := (X86Int.aluF .add ((pacc + (ls.mem (x + i)).plen) % X86Env.W) ((ls.mem (y + i)).plen % X86Env.W)).2, rip := rip_loop + 37 } := by
      rw [step_at_vc chk rfl hm_addy, exec_alu hs9_10 hs9_0, ← hadd2_eq]
    let s10 := { s9.set 10 (some (pacc' % X86Env.W)) with
                 fl := (X86Int.aluF .add ((pacc + (ls.mem (x + i)).plen) % X86Env.W) ((ls.mem (y + i)).plen % X86Env.W)).2, rip := rip_loop + 37 }
    have hr10_ok : RegsOk s10 := regsOk_set hr9_ok (by decide) _
    have hs10_11 : s10.reg 11 = some i := by
      change (s9.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s8.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs8_11
    have hi1W : i + 1 < X86Env.W := by have : ls.msize < 2 ^ 44 := hmi.1.2.1; omega
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := rip_loop + 41 } := by
      rw [step_at_vc chk rfl hm_inc, exec_aluI_add_lt hs10_11 (by decide), Nat.mod_eq_of_lt hi1W]
    let s11 := { s10.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := rip_loop + 41 }
    have hr11_ok : RegsOk s11 := regsOk_set hr10_ok (by decide) _
    have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := rip_loop } := by
      rw [step_at_vc chk rfl hm_jmp, exec_jmp]
    let s12 := { s11 with rip := rip_loop }
    have hrch12 : Reaches vmImage s s12 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09
            (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step h12)))))))))))
    have hreg_pres : ∀ r, r ≠ 0 → r ≠ 2 → r ≠ 3 → r ≠ 10 → r ≠ 11 → s12.reg r = s.reg r := by
      intro r h0 h2 h3 h10' h11'
      change (s10.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]
      change (s9.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
      change (s8.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s7.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
      change (s6.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s5.set 3 _).reg r = _; rw [reg_set_ne h3.symm]
      change (s3.set 2 _).reg r = _; rw [reg_set_ne h2.symm]
      change (s2.set 0 _).reg r = _; rw [reg_set_ne h0.symm]; rfl
    have hwf12 : vc.wf pr ls s12 :=
      vc.wf_step_misc h rfl hr11_ok
        (by rw [hreg_pres 4 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.1)
        (by rw [hreg_pres 15 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.2.1)
    have hs12_3 : s12.reg 3 = some (Sporos.bool bacc') := by
      change (s10.set 11 _).reg 3 = _; rw [reg_set_ne (by decide)]
      change (s9.set 10 _).reg 3 = _; rw [reg_set_ne (by decide)]
      change (s8.set 0 _).reg 3 = _; rw [reg_set_ne (by decide)]
      change (s7.set 10 _).reg 3 = _; rw [reg_set_ne (by decide)]
      change (s6.set 0 _).reg 3 = _; rw [reg_set_ne (by decide)]
      exact reg_set_self_ok' hr4_ok (by decide) _
    have hs12_10 : s12.reg 10 = some (pacc' % X86Env.W) := by
      change (s10.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]
      exact reg_set_self_ok' hr9_ok (by decide) _
    have hs12_11 : s12.reg 11 = some (i + 1) := reg_set_self_ok' hr10_ok (by decide) _
    have hbacc' : bacc' = (List.range (i + 1)).all (fun k => (ls.mem (x + k)).w % 256 == (ls.mem (y + k)).w % 256) := by
      rw [all_range_step, ← hbacc]
    have hpacc' : pacc' % X86Env.W =
        (q + (List.range (i + 1)).foldr (fun k acc => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen + acc) 0) % X86Env.W := by
      dsimp [pacc']
      have hf := foldr_range_step (fun k => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen) i
      have hfold_eq : (List.range (i + 1)).foldr (fun k acc => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen + acc) 0 =
          (List.range i).foldr (fun k acc => (ls.mem (x + k)).plen + (ls.mem (y + k)).plen + acc) 0 +
            ((ls.mem (x + i)).plen + (ls.mem (y + i)).plen) := by
        simpa [Nat.add_assoc] using hf
      rw [hfold_eq, show pacc + (ls.mem (x + i)).plen + (ls.mem (y + i)).plen =
          pacc + ((ls.mem (x + i)).plen + (ls.mem (y + i)).plen) by omega]
      rw [← mod_add_left, hpacc, mod_add_left]
      congr 1; omega
    obtain ⟨sf, hrchf, hripf, hmemf, hkf, hwff, h3f, h10f⟩ :=
      ih (i + 1) bacc' pacc' s12 (by omega) rfl hwf12
        (by rw [hreg_pres 1 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr1)
        hs12_3
        (by rw [hreg_pres 6 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr6)
        (by rw [hreg_pres 7 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr7)
        hs12_10 hs12_11
        (by rw [hreg_pres 12 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr12)
        (by rw [hreg_pres 13 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr13)
        (by rw [hreg_pres 14 (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr14)
        hbacc' hpacc'
    exact ⟨sf, Reaches.trans hrch12 hrchf, hripf, hmemf, hkf, hwff, h3f, h10f⟩

theorem mem_cmp_finish_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {x y n q e_in rip0 c_ld8 c_pre c_body v11 : Nat}
    (hrip : s.rip = rip0) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hein : e_in ≤ ls.ds.length) (hx_in : x + n ≤ ls.msize) (hy_in : y + n ≤ ls.msize)
    (hqW : q < X86Env.W)
    (hr0 : s.reg 0 = some x) (hr1 : s.reg 1 = some n) (hr2 : s.reg 2 = some y)
    (hr11 : s.reg 11 = some v11)
    (h272 : rd64 s.mem (ea ctxBase 272) = some q)
    (h280 : rd64 s.mem (ea ctxBase 280) = some e_in)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (hm_ld8 : (rip0, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk c_ld8)
    (hm_l6 : (rip0 + 4, (.leaIdx 6 8 0 1 0), 4) ∈ vmTableChunk c_pre)
    (hm_l7 : (rip0 + 8, (.leaIdx 7 8 2 1 0), 4) ∈ vmTableChunk c_pre)
    (hm_ld9 : (rip0 + 12, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk c_pre)
    (hm_l12 : (rip0 + 16, (.leaIdx 12 9 0 8 0), 4) ∈ vmTableChunk c_pre)
    (hm_l13 : (rip0 + 20, (.leaIdx 13 9 2 8 0), 4) ∈ vmTableChunk c_body)
    (hm_m3 : (rip0 + 24, (.ofInt (.movI 3 1)), 5) ∈ vmTableChunk c_body)
    (hm_lq : (rip0 + 29, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk c_body)
    (hm_z11 : (rip0 + 36, (.ofInt (.alu .xor 11 11)), 3) ∈ vmTableChunk c_body)
    (hm_z14 : (rip0 + 39, (.ofInt (.alu .xor 14 14)), 3) ∈ vmTableChunk c_body)
    (hm_cmp : (rip0 + 42, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk c_body)
    (hm_jae : (rip0 + 45, (.ofInt (.jcc 3 (rip0 + 85))), 2) ∈ vmTableChunk c_body)
    (hm_ldx : (rip0 + 47, (.ldIdx .w8 0 6 11 1 0), 5) ∈ vmTableChunk c_body)
    (hm_ldy : (rip0 + 52, (.ldIdx .w8 2 7 11 1 0), 5) ∈ vmTableChunk c_body)
    (hm_cmpb : (rip0 + 57, (.ofInt (.cmp 0 2)), 3) ∈ vmTableChunk c_body)
    (hm_cmov : (rip0 + 60, (.cmovcc 5 3 14), 4) ∈ vmTableChunk c_body)
    (hm_ldpx : (rip0 + 64, (.ldIdx .w64 0 12 11 8 0), 4) ∈ vmTableChunk c_body)
    (hm_addx : (rip0 + 68, (.ofInt (.alu .add 10 0)), 3) ∈ vmTableChunk c_body)
    (hm_ldpy : (rip0 + 71, (.ldIdx .w64 0 13 11 8 0), 5) ∈ vmTableChunk c_body)
    (hm_addy : (rip0 + 76, (.ofInt (.alu .add 10 0)), 3) ∈ vmTableChunk c_body)
    (hm_inc : (rip0 + 79, (.ofInt (.aluI .add 11 1)), 4) ∈ vmTableChunk c_body)
    (hm_jmp : (rip0 + 83, (.ofInt (.jmp (rip0 + 42))), 2) ∈ vmTableChunk c_body)
    (hm_st0 : (rip0 + 85, (.ofInt (.st 15 360 3)), 7) ∈ vmTableChunk c_body)
    (hm_st1 : (rip0 + 92, (.ofInt (.st 15 368 10)), 7) ∈ vmTableChunk c_body)
    (hm_epi : (rip0 + 99, (.ofInt (.jmp 4195233)), 5) ∈ vmTableChunk c_body) :
    let eqb := (List.range n).all fun i => (ls.mem (x + i)).w % 256 == (ls.mem (y + i)).w % 256
    let pm := (List.range n).foldr (fun i acc => (ls.mem (x + i)).plen + (ls.mem (y + i)).plen + acc) 0
    let ls' := { ls with ds := [⟨Sporos.bool eqb, (q + pm) % Sporos.M⟩] ++ ls.ds.drop e_in,
                         rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hmi := wf_mem_info h
  have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
  have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
  have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 8 (some vc.memwReg.base) with rip := rip0 + 4 } := by
    rw [step_at_vc c_ld8 hrip hm_ld8, exec_ld h15 (rd_ctx_memwBase h), hrip]
  let s1 := { s.set 8 (some vc.memwReg.base) with rip := rip0 + 4 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_8 : s1.reg 8 = some vc.memwReg.base := reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs1_0 : s1.reg 0 = some x := by change (s.set 8 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hr0
  have hea6 : eaIdx vc.memwReg.base x 1 0 = vc.memwReg.base + x := by
    have hltW : vc.memwReg.base + 1 * x + 0 < X86Env.W := by
      have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
    rw [eaIdx_add_eq hltW]; omega
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 6 (some (vc.memwReg.base + x)) with rip := rip0 + 8 } := by
    rw [step_at_vc c_pre rfl hm_l6, exec_leaIdx hs1_8 hs1_0, hea6]
  let s2 := { s1.set 6 (some (vc.memwReg.base + x)) with rip := rip0 + 8 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_8 : s2.reg 8 = some vc.memwReg.base := by change (s1.set 6 _).reg 8 = _; rw [reg_set_ne (by decide)]; exact hs1_8
  have hs2_2 : s2.reg 2 = some y := by
    change (s1.set 6 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s.set 8 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hr2
  have hea7 : eaIdx vc.memwReg.base y 1 0 = vc.memwReg.base + y := by
    have hltW : vc.memwReg.base + 1 * y + 0 < X86Env.W := by
      have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
    rw [eaIdx_add_eq hltW]; omega
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 7 (some (vc.memwReg.base + y)) with rip := rip0 + 12 } := by
    rw [step_at_vc c_pre rfl hm_l7, exec_leaIdx hs2_8 hs2_2, hea7]
  let s3 := { s2.set 7 (some (vc.memwReg.base + y)) with rip := rip0 + 12 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 7 _).reg 15 = _; rw [reg_set_ne (by decide)]
    change (s1.set 6 _).reg 15 = _; rw [reg_set_ne (by decide)]
    change (s.set 8 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 9 (some vc.mempReg.base) with rip := rip0 + 16 } := by
    rw [step_at_vc c_pre rfl hm_ld9, exec_ld hs3_15 (by change rd64 s.mem _ = _; exact rd_ctx_mempBase h)]
  let s4 := { s3.set 9 (some vc.mempReg.base) with rip := rip0 + 16 }
  have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
  have hs4_9 : s4.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr3_ok (by decide) _
  have hs4_0 : s4.reg 0 = some x := by
    change (s3.set 9 _).reg 0 = _; rw [reg_set_ne (by decide)]
    change (s2.set 7 _).reg 0 = _; rw [reg_set_ne (by decide)]
    change (s1.set 6 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs1_0
  have hea12 : eaIdx vc.mempReg.base x 8 0 = vc.mempReg.base + x * 8 := by
    have hltW : vc.mempReg.base + 8 * x + 0 < X86Env.W := by
      have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
    rw [eaIdx_add_eq hltW]; omega
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 12 (some (vc.mempReg.base + x * 8)) with rip := rip0 + 20 } := by
    rw [step_at_vc c_pre rfl hm_l12, exec_leaIdx hs4_9 hs4_0, hea12]
  let s5 := { s4.set 12 (some (vc.mempReg.base + x * 8)) with rip := rip0 + 20 }
  have hr5_ok : RegsOk s5 := regsOk_set hr4_ok (by decide) _
  have hs5_9 : s5.reg 9 = some vc.mempReg.base := by change (s4.set 12 _).reg 9 = _; rw [reg_set_ne (by decide)]; exact hs4_9
  have hs5_2 : s5.reg 2 = some y := by
    change (s4.set 12 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s3.set 9 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s2.set 7 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hs2_2
  have hea13 : eaIdx vc.mempReg.base y 8 0 = vc.mempReg.base + y * 8 := by
    have hltW : vc.mempReg.base + 8 * y + 0 < X86Env.W := by
      have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
    rw [eaIdx_add_eq hltW]; omega
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 13 (some (vc.mempReg.base + y * 8)) with rip := rip0 + 24 } := by
    rw [step_at_vc c_body rfl hm_l13, exec_leaIdx hs5_9 hs5_2, hea13]
  let s6 := { s5.set 13 (some (vc.mempReg.base + y * 8)) with rip := rip0 + 24 }
  have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 3 (some 1) with rip := rip0 + 29 } := by
    rw [step_at_vc c_body rfl hm_m3, exec_movI_lt (by have : X86Env.W = 2 ^ 64 := rfl; omega)]
  let s7 := { s6.set 3 (some 1) with rip := rip0 + 29 }
  have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
  have hs7_15 : s7.reg 15 = some ctxBase := by
    change (s6.set 3 _).reg 15 = _; rw [reg_set_ne (by decide)]
    change (s5.set 13 _).reg 15 = _; rw [reg_set_ne (by decide)]
    change (s4.set 12 _).reg 15 = _; rw [reg_set_ne (by decide)]
    change (s3.set 9 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs3_15
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 10 (some q) with rip := rip0 + 36 } := by
    rw [step_at_vc c_body rfl hm_lq, exec_ld hs7_15 h272]
  let s8 := { s7.set 10 (some q) with rip := rip0 + 36 }
  have hr8_ok : RegsOk s8 := regsOk_set hr7_ok (by decide) _
  have hs8_11 : s8.reg 11 = some v11 := by
    change (s7.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s6.set 3 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s5.set 13 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s4.set 12 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s3.set 9 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s2.set 7 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s1.set 6 _).reg 11 = _; rw [reg_set_ne (by decide)]
    change (s.set 8 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hr11
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 11 (some 0) with fl := (X86Int.aluF .xor v11 v11).2, rip := rip0 + 39 } := by
    rw [step_at_vc c_body rfl hm_z11, exec_alu_xor_self hs8_11]
  let s9 := { s8.set 11 (some 0) with fl := (X86Int.aluF .xor v11 v11).2, rip := rip0 + 39 }
  have hr9_ok : RegsOk s9 := regsOk_set hr8_ok (by decide) _
  obtain ⟨v14, hv14⟩ := regs_get_of_ok hr9_ok 14 (by decide) (by decide)
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 14 (some 0) with fl := (X86Int.aluF .xor v14 v14).2, rip := rip0 + 42 } := by
    rw [step_at_vc c_body rfl hm_z14, exec_alu_xor_self hv14]
  let s10 := { s9.set 14 (some 0) with fl := (X86Int.aluF .xor v14 v14).2, rip := rip0 + 42 }
  have hr10_ok : RegsOk s10 := regsOk_set hr9_ok (by decide) _
  have hrch10 : Reaches vmImage s s10 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
        (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step h10)))))))))
  have hs10_keep : ∀ r, r ≠ 3 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → r ≠ 14 →
      s10.reg r = s.reg r := by
    intro r h3 h6 h7 h8 h9 h10' h11 h12 h13 h14
    change (s9.set 14 _).reg r = _; rw [reg_set_ne h14.symm]
    change (s8.set 11 _).reg r = _; rw [reg_set_ne h11.symm]
    change (s7.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
    change (s6.set 3 _).reg r = _; rw [reg_set_ne h3.symm]
    change (s5.set 13 _).reg r = _; rw [reg_set_ne h13.symm]
    change (s4.set 12 _).reg r = _; rw [reg_set_ne h12.symm]
    change (s3.set 9 _).reg r = _; rw [reg_set_ne h9.symm]
    change (s2.set 7 _).reg r = _; rw [reg_set_ne h7.symm]
    change (s1.set 6 _).reg r = _; rw [reg_set_ne h6.symm]
    change (s.set 8 _).reg r = _; rw [reg_set_ne h8.symm]
  have hwf10 : vc.wf pr ls s10 :=
    vc.wf_step_misc h rfl hr10_ok
      (by rw [hs10_keep 4 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.1)
      (by rw [hs10_keep 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact h15)
  obtain ⟨s11, hrch11, hrip11, hmem11, hk11, hwf11, hs11_3, hs11_10⟩ :=
    mem_cmp_loop_spec (q := q) hx_in hy_in hm_cmp (by simpa [show rip0 + 42 + 43 = rip0 + 85 by omega] using hm_jae)
      hm_ldx hm_ldy hm_cmpb hm_cmov hm_ldpx hm_addx hm_ldpy hm_addy hm_inc hm_jmp
      n 0 true q s10 (Nat.zero_add n) rfl hwf10
      (by rw [hs10_keep 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hr1)
      (by change (s9.set 14 _).reg 3 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 3 = _; rw [reg_set_ne (by decide)]
          change (s7.set 10 _).reg 3 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr6_ok (by decide) _)
      (by change (s9.set 14 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s7.set 10 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s6.set 3 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s5.set 13 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s4.set 12 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s3.set 9 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s2.set 7 _).reg 6 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr1_ok (by decide) _)
      (by change (s9.set 14 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s7.set 10 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s6.set 3 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s5.set 13 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s4.set 12 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s3.set 9 _).reg 7 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr2_ok (by decide) _)
      (by change (s9.set 14 _).reg 10 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]
          rw [Nat.mod_eq_of_lt hqW]; exact reg_set_self_ok' hr7_ok (by decide) _)
      (by change (s9.set 14 _).reg 11 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr8_ok (by decide) _)
      (by change (s9.set 14 _).reg 12 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 12 = _; rw [reg_set_ne (by decide)]
          change (s7.set 10 _).reg 12 = _; rw [reg_set_ne (by decide)]
          change (s6.set 3 _).reg 12 = _; rw [reg_set_ne (by decide)]
          change (s5.set 13 _).reg 12 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr4_ok (by decide) _)
      (by change (s9.set 14 _).reg 13 = _; rw [reg_set_ne (by decide)]
          change (s8.set 11 _).reg 13 = _; rw [reg_set_ne (by decide)]
          change (s7.set 10 _).reg 13 = _; rw [reg_set_ne (by decide)]
          change (s6.set 3 _).reg 13 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr5_ok (by decide) _)
      (reg_set_self_ok' hr9_ok (by decide) _)
      (by simp) (by simp)
  have hrip11' : s11.rip = rip0 + 85 := by omega
  let eqb := (List.range n).all fun i => (ls.mem (x + i)).w % 256 == (ls.mem (y + i)).w % 256
  let pm := (List.range n).foldr (fun i acc => (ls.mem (x + i)).plen + (ls.mem (y + i)).plen + acc) 0
  obtain ⟨_, m12, _, _, h12, hwf12, h360_12, hrest12⟩ :=
    vc.wr64_ctx_scratch_slot hwf11 360 (Sporos.bool eqb) (rip0 + 92) (by decide) (by decide) hs11_3
      (fun {m'} hwr => by
        rw [step_at_vc c_body hrip11' hm_st0, exec_st hwf11.2.2.2.2.2.1 hs11_3 hwr, hrip11'])
  let s12 := { s11 with mem := m12, rip := rip0 + 92 }
  let vc12 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 (Sporos.bool eqb)) } }
  obtain ⟨_, m13, _, _, h13, hwf13, h368_13, hrest13⟩ :=
    vc12.wr64_ctx_scratch_slot hwf12 368 ((q + pm) % X86Env.W) (rip0 + 99) (by decide) (by decide) hs11_10
      (fun {m'} hwr => by
        rw [step_at_vc c_body rfl hm_st1, exec_st hwf12.2.2.2.2.2.1 hs11_10 hwr])
  let s13 := { s12 with mem := m13, rip := rip0 + 99 }
  let vc13 := { vc12 with ctxReg := { vc12.ctxReg with data := writeBytes vc12.ctxReg.data 368 (bytes8 ((q + pm) % X86Env.W)) } }
  have h14 : X86Vm.step vmImage s13 = .ok { s13 with rip := SporosVmBin.L_prim_epilogue } := by
    rw [step_at_vc c_body rfl hm_epi, exec_jmp]; rfl
  let s14 := { s13 with rip := SporosVmBin.L_prim_epilogue }
  have hwf14 : vc13.wf pr ls s14 := vc13.wf_step_misc hwf13 rfl hwf13.2.2.2.1 hwf13.2.2.2.2.1 hwf13.2.2.2.2.2.1
  have hrch14 : Reaches vmImage s s14 :=
    Reaches.trans hrch10 (Reaches.trans hrch11 (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.step h14))))
  have hor14 : OracleAdmits s14 := oracleAdmits_of_reaches hor hrch14
  have h280_14 : rd64 s14.mem (ea ctxBase 280) = some e_in :=
    vc13.rd64_ctx_of_eq hwf14 (by decide)
      (by rw [hrest13 280 (Or.inl (by decide)) (by decide), hrest12 280 (Or.inl (by decide)) (by decide)]
          exact ctx_leVal_of_rd64 h (by decide) h280)
  have h288_14 : rd64 s14.mem (ea ctxBase 288) = some 1 :=
    vc13.rd64_ctx_of_eq hwf14 (by decide)
      (by rw [hrest13 288 (Or.inl (by decide)) (by decide), hrest12 288 (Or.inl (by decide)) (by decide)]
          exact ctx_leVal_of_rd64 h (by decide) h288)
  have hboolW : Sporos.bool eqb < X86Env.W := by
    cases eqb <;> simp [Sporos.bool] <;> (have : X86Env.W = 2 ^ 64 := rfl; omega)
  have h360_14 : rd64 s14.mem (ea ctxBase 360) = some (Sporos.bool eqb) :=
    vc13.rd64_ctx_of_eq hwf14 (by decide)
      (by rw [hrest13 360 (Or.inl (by decide)) (by decide), h360_12, Nat.mod_eq_of_lt hboolW])
  have h368_14 : rd64 s14.mem (ea ctxBase 368) = some ((q + pm) % X86Env.W) :=
    vc13.rd64_ctx_of_eq hwf14 (by decide) (by rw [h368_13, Nat.mod_mod])
  let resv : SporosProvLen.LVal := ⟨Sporos.bool eqb, (q + pm) % Sporos.M⟩
  have hepi := L_prim_epilogue_oracle_spec (vc := vc13) (pr := pr) (ls := ls) (s := s14)
    (e_in := e_in) (outs := [resv]) (f := f) (rest := rest)
    rfl hwf14 hor14 hrs hpcW hein (by simp)
    (by intro v hv; simp at hv; subst hv
        exact ⟨hboolW, by dsimp [resv]; rw [M_eq_W']; exact Nat.mod_lt _ (by decide)⟩)
    h280_14 h288_14
    (by intro i hi; simp at hi; subst hi
        exact ⟨h360_14, by dsimp [resv]; rw [M_eq_W']; exact h368_14⟩)
  rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
  · exact Or.inl ⟨sf, Reaches.trans hrch14 hrchf, by rw [hkf]; exact congrArg X86Env.Kern.out hk11, horf, hrelf⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch14 hhit, Ends.of_reaches hrch14 hends⟩

/-! ### `P_copy`, `P_fill`, `P_mcmp` -/

private theorem copy_fwd_loop_spec {pr : Sporos.Prog} {ls₀ : SporosProvLen.LSt}
    {src dst n q : Nat} (hsrc : src + n ≤ ls₀.msize) (hdst : dst + n ≤ ls₀.msize)
    (hle : dst ≤ src) :
    ∀ (rem i : Nat) (vc : VmCtx) (s : X86Vm.St),
      i + rem = n →
      let mem_i : SporosProvLen.LMem := fun x =>
        if dst ≤ x ∧ x < dst + i then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
        else ls₀.mem x
      s.rip = 4197498 → vc.wf pr { ls₀ with mem := mem_i } s →
      s.reg 1 = some n → s.reg 6 = some (vc.memwReg.base + src) →
      s.reg 7 = some (vc.memwReg.base + dst) → s.reg 10 = some q → s.reg 11 = some i →
      s.reg 12 = some (vc.mempReg.base + src * 8) → s.reg 13 = some (vc.mempReg.base + dst * 8) →
      ∃ (vc' : VmCtx) (s' : X86Vm.St),
        Reaches vmImage s s' ∧ s'.rip = SporosVmBin.L_prim_epilogue ∧ s'.k = s.k ∧
        vc'.ctxReg = vc.ctxReg ∧
        vc'.wf pr { ls₀ with mem := (fun x =>
          if dst ≤ x ∧ x < dst + n then
            ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
          else ls₀.mem x) } s' := by
  intro rem
  induction rem with
  | zero =>
    intro i vc s hirem _ hrip h hr1 _ _ _ hr11 _ _
    have hi_eq : i = n := by omega
    subst hi_eq
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i i).2, rip := 4197501 } := by
      rw [step_at_vc 10 hrip (by decide : (4197498, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 10),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i i).2, rip := 4197501 }
    have hcae : X86Int.cond s1.fl 3 = some true := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_true (Nat.le_refl _)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 10 rfl (by decide : (4197501, (.ofInt (.jcc 3 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := SporosVmBin.L_prim_epilogue }
    exact ⟨vc, s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl,
      vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1⟩
  | succ rem ih =>
    intro i vc s hirem mem_i hrip h hr1 hr6 hr7 hr10 hr11 hr12 hr13
    have hmi := wf_mem_info h
    have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
    have hW44 : 2 ^ 44 < X86Env.W := by change 2 ^ 44 < 2 ^ 64; omega
    have hsi : src + i < ls₀.msize := by omega
    have hdi : dst + i < ls₀.msize := by omega
    have hmw_len : ls₀.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls₀.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have hmem_src : mem_i (src + i) = ls₀.mem (src + i) := by
      dsimp [mem_i]; rw [if_neg (by omega)]
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i n).2, rip := 4197501 } := by
      rw [step_at_vc 10 hrip (by decide : (4197498, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 10),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i n).2, rip := 4197501 }
    have hcae : X86Int.cond s1.fl 3 = some false := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_false (by omega)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4197507 } := by
      rw [step_at_vc 10 rfl (by decide : (4197501, (.ofInt (.jcc 3 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := 4197507 }
    have hea_sw : eaIdx (vc.memwReg.base + src) i 1 0 = vc.memwReg.base + (src + i) := by
      have hltW : vc.memwReg.base + src + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_dw : eaIdx (vc.memwReg.base + dst) i 1 0 = vc.memwReg.base + (dst + i) := by
      have hltW : vc.memwReg.base + dst + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_sp : eaIdx (vc.mempReg.base + src * 8) i 8 0 = vc.mempReg.base + (src + i) * 8 := by
      have hltW : vc.mempReg.base + src * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_dp : eaIdx (vc.mempReg.base + dst * 8) i 8 0 = vc.mempReg.base + (dst + i) * 8 := by
      have hltW : vc.mempReg.base + dst * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_w : rdMem .w8 s2.mem (eaIdx (vc.memwReg.base + src) i 1 0) = some ((ls₀.mem (src + i)).w % 256) := by
      change rdMem .w8 s.mem _ = _; rw [hea_sw]; simpa [hmem_src] using rd_mem_byte_at h hsi
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some ((ls₀.mem (src + i)).w % 256)) with rip := 4197512 } := by
      rw [step_at_vc 10 rfl (by decide : (4197507, (.ldIdx .w8 0 6 11 1 0), 5) ∈ vmTableChunk 10),
        exec_ldIdx (s := s2) hr6 hr11 hrd_w]
    let s3 := { s2.set 0 (some ((ls₀.mem (src + i)).w % 256)) with rip := 4197512 }
    have hr3_ok : RegsOk s3 := regsOk_set h.2.2.2.1 (by decide) _
    have hs3_0 : s3.reg 0 = some ((ls₀.mem (src + i)).w % 256) := reg_set_self_ok' h.2.2.2.1 (by decide) _
    have hs3_7 : s3.reg 7 = some (vc.memwReg.base + dst) := by
      change (s2.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hr7
    have hs3_11 : s3.reg 11 = some i := by
      change (s2.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hr11
    have hwlt_src := (vc.code_flat h).2.2.2.2.2.2.2.1 (src + i)
    dsimp only at hwlt_src
    rw [hmem_src] at hwlt_src
    let q' : Nat := ((ls₀.mem (src + i)).plen + q) % Sporos.M
    have hq'M : q' < Sporos.M := Nat.mod_lt _ (by rw [M_eq_W']; change 0 < 2 ^ 64; omega)
    obtain ⟨m1, m2, hwr1, hwr2, _, _, _, _, _, _, hwf_pair⟩ :=
      wr_mem_pair_step h hdi (v := (ls₀.mem (src + i)).w) (q := q') hwlt_src.1 hq'M
    have hwr1_b : wrMem .w8 s3.mem (eaIdx (vc.memwReg.base + dst) i 1 0) ((ls₀.mem (src + i)).w % 256) = some m1 := by
      change wrMem .w8 s.mem _ _ = _; rw [hea_dw]
      simpa [X86Vm.wrMem, Nat.mod_mod] using hwr1
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with mem := m1, rip := 4197516 } := by
      rw [step_at_vc 10 rfl (by decide : (4197512, (.stIdx .w8 7 11 1 0 0), 4) ∈ vmTableChunk 10),
        exec_stIdx hs3_7 hs3_11 hs3_0 hwr1_b]
    let s4 := { s3 with mem := m1, rip := 4197516 }
    have hne_mw_mp : vc.memwReg.base ≠ vc.mempReg.base :=
      vc.ne_bases h (i := 10) (j := 11) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hmw1_wf : m1.wf := by
      obtain ⟨_, m1', hwr1', _, _, hm1'wf, _, _⟩ :=
        wrMem_w8_region h.1.2 (vc.mem_memw h) hmi.1.1 (a := vc.memwReg.base + (dst + i)) (Nat.le_add_right _ _)
          (by simp only [Region.hi]; omega) (v := (ls₀.mem (src + i)).w)
      rw [hwr1] at hwr1'; cases hwr1'; exact hm1'wf
    have hmp_in_m1 : vc.mempReg ∈ m1 := by
      obtain ⟨_, m1', hwr1', _, _, _, _, hoth1'⟩ :=
        wrMem_w8_region h.1.2 (vc.mem_memw h) hmi.1.1 (a := vc.memwReg.base + (dst + i)) (Nat.le_add_right _ _)
          (by simp only [Region.hi]; omega) (v := (ls₀.mem (src + i)).w)
      rw [hwr1] at hwr1'; cases hwr1'
      exact hoth1' vc.mempReg (vc.mem_memp h) (Ne.symm hne_mw_mp)
    have hrd_p : rdMem .w64 m1 (eaIdx (vc.mempReg.base + src * 8) i 8 0) = some ((ls₀.mem (src + i)).plen % X86Env.W) := by
      rw [hea_sp]
      have hrel := (hmi.2.2.1.2.2 (src + i) hsi).2
      dsimp only at hrel
      rw [hmem_src] at hrel
      exact rdMem_w64_at hmw1_wf hmp_in_m1 (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hrel)
    have hs4_12 : s4.reg 12 = some (vc.mempReg.base + src * 8) := by
      change (s2.set 0 _).reg 12 = _; rw [reg_set_ne (by decide)]; exact hr12
    have h05 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some ((ls₀.mem (src + i)).plen % X86Env.W)) with rip := 4197520 } := by
      rw [step_at_vc 10 rfl (by decide : (4197516, (.ldIdx .w64 0 12 11 8 0), 4) ∈ vmTableChunk 10),
        exec_ldIdx (s := s4) hs4_12 hs3_11 hrd_p]
    let s5 := { s4.set 0 (some ((ls₀.mem (src + i)).plen % X86Env.W)) with rip := 4197520 }
    have hr5_ok : RegsOk s5 := regsOk_set hr3_ok (by decide) _
    have hs5_0 : s5.reg 0 = some ((ls₀.mem (src + i)).plen % X86Env.W) := reg_set_self_ok' hr3_ok (by decide) _
    have hs5_10 : s5.reg 10 = some q := by
      change (s4.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hr10
    have hadd_q : (X86Int.aluF .add ((ls₀.mem (src + i)).plen % X86Env.W) q).1 = q' := by
      dsimp [q']; simp only [aluF_add_val, Nat.mod_add_mod, M_eq_W']
    have h06 : X86Vm.step vmImage s5 =
        .ok { s5.set 0 (some q') with fl := (X86Int.aluF .add ((ls₀.mem (src + i)).plen % X86Env.W) q).2, rip := 4197523 } := by
      rw [step_at_vc 10 rfl (by decide : (4197520, (.ofInt (.alu .add 0 10)), 3) ∈ vmTableChunk 10),
        exec_alu hs5_0 hs5_10, ← hadd_q]
    let s6 := { s5.set 0 (some q') with fl := (X86Int.aluF .add ((ls₀.mem (src + i)).plen % X86Env.W) q).2, rip := 4197523 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_0 : s6.reg 0 = some q' := reg_set_self_ok' hr5_ok (by decide) _
    have hs6_13 : s6.reg 13 = some (vc.mempReg.base + dst * 8) := by
      change (s5.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s4.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s2.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]; exact hr13
    have hs6_11 : s6.reg 11 = some i := by
      change (s5.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s4.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs3_11
    have hwr2' : wrMem .w64 s6.mem (eaIdx (vc.mempReg.base + dst * 8) i 8 0) q' = some m2 := by
      change wrMem .w64 m1 _ _ = _; rw [hea_dp]; exact hwr2
    have h07 : X86Vm.step vmImage s6 = .ok { s6 with mem := m2, rip := 4197528 } := by
      rw [step_at_vc 10 rfl (by decide : (4197523, (.stIdx .w64 13 11 8 0 0), 5) ∈ vmTableChunk 10),
        exec_stIdx hs6_13 hs6_11 hs6_0 hwr2']
    let s7 := { s6 with mem := m2, rip := 4197528 }
    have hi1W : i + 1 < X86Env.W := by have : ls₀.msize < 2 ^ 44 := hmi.1.2.1; omega
    have h08 : X86Vm.step vmImage s7 =
        .ok { s7.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4197532 } := by
      rw [step_at_vc 10 rfl (by decide : (4197528, (.ofInt (.aluI .add 11 1)), 4) ∈ vmTableChunk 10),
        exec_aluI_add_lt (s := s7) hs6_11 (by decide), Nat.mod_eq_of_lt hi1W]
    let s8 := { s7.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4197532 }
    have hr8_ok : RegsOk s8 := regsOk_set hr6_ok (by decide) _
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4197498 } := by
      rw [step_at_vc 10 rfl (by decide : (4197532, (.ofInt (.jmp 4197498)), 2) ∈ vmTableChunk 10), exec_jmp]
    let s9 := { s8 with rip := 4197498 }
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))))))
    have hs9_keep : ∀ r, r ≠ 0 → r ≠ 11 → s9.reg r = s.reg r := by
      intro r h0 h11'
      change (s7.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]
      change (s5.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s4.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s2.set 0 _).reg r = _; rw [reg_set_ne h0.symm]; rfl
    have hmem_step : (fun x => if x = dst + i then (⟨(ls₀.mem (src + i)).w, q'⟩ : SporosProvLen.LVal) else mem_i x) =
        (fun x => if dst ≤ x ∧ x < dst + (i + 1) then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩ else ls₀.mem x) := by
      funext x
      by_cases h1 : x = dst + i
      · subst h1; simp [q', Nat.add_sub_cancel_left]
      · dsimp [mem_i]
        by_cases h2 : dst ≤ x ∧ x < dst + i
        · have h3 : dst ≤ x ∧ x < dst + (i + 1) := by omega
          simp [h1, h2, h3]
        · have h3 : ¬ (dst ≤ x ∧ x < dst + (i + 1)) := by omega
          simp [h1, h2, h3]
    let vc1 := { vc with memwReg := { vc.memwReg with data := writeBytes vc.memwReg.data (dst + i) [(ls₀.mem (src + i)).w % 256] },
                         mempReg := { vc.mempReg with data := writeBytes vc.mempReg.data ((dst + i) * 8) (bytes8 q') } }
    have hwf9 : vc1.wf pr { ls₀ with mem := (fun x =>
        if dst ≤ x ∧ x < dst + (i + 1) then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩ else ls₀.mem x) } s9 := by
      rw [← hmem_step]
      exact vc1.wf_step_misc hwf_pair rfl hr8_ok
        (by rw [hs9_keep 4 (by decide) (by decide)]; exact h.2.2.2.2.1)
        (by rw [hs9_keep 15 (by decide) (by decide)]; exact h.2.2.2.2.2.1)
    obtain ⟨vc', sf, hrchf, hripf, hkf, hctxf, hwff⟩ :=
      ih (i + 1) vc1 s9 (by omega) rfl hwf9
        (by rw [hs9_keep 1 (by decide) (by decide)]; exact hr1)
        (by rw [hs9_keep 6 (by decide) (by decide)]; exact hr6)
        (by rw [hs9_keep 7 (by decide) (by decide)]; exact hr7)
        (by rw [hs9_keep 10 (by decide) (by decide)]; exact hr10)
        (reg_set_self_ok' hr6_ok (by decide) _)
        (by rw [hs9_keep 12 (by decide) (by decide)]; exact hr12)
        (by rw [hs9_keep 13 (by decide) (by decide)]; exact hr13)
    exact ⟨vc', sf, Reaches.trans hrch9 hrchf, hripf, hkf, hctxf, hwff⟩

private theorem copy_bkw_loop_spec {pr : Sporos.Prog} {ls₀ : SporosProvLen.LSt}
    {src dst n q : Nat} (hdst : dst + n ≤ ls₀.msize)
    (hlt : src < dst) :
    ∀ (k : Nat) (vc : VmCtx) (s : X86Vm.St),
      k ≤ n →
      let mem_k : SporosProvLen.LMem := fun x =>
        if dst + k ≤ x ∧ x < dst + n then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
        else ls₀.mem x
      s.rip = 4197537 → vc.wf pr { ls₀ with mem := mem_k } s →
      s.reg 6 = some (vc.memwReg.base + src) →
      s.reg 7 = some (vc.memwReg.base + dst) → s.reg 10 = some q → s.reg 11 = some k →
      s.reg 12 = some (vc.mempReg.base + src * 8) → s.reg 13 = some (vc.mempReg.base + dst * 8) →
      ∃ (vc' : VmCtx) (s' : X86Vm.St),
        Reaches vmImage s s' ∧ s'.rip = SporosVmBin.L_prim_epilogue ∧ s'.k = s.k ∧
        vc'.ctxReg = vc.ctxReg ∧
        vc'.wf pr { ls₀ with mem := (fun x =>
          if dst ≤ x ∧ x < dst + n then
            ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
          else ls₀.mem x) } s' := by
  intro k
  induction k with
  | zero =>
    intro vc s _ mem_0 hrip h _ _ _ hr11 _ _
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .and 0 0).2, rip := 4197540 } := by
      rw [step_at_vc 10 hrip (by decide : (4197537, (.ofInt (.test 11 11)), 3) ∈ vmTableChunk 10),
        exec_test hr11 hr11, hrip]
    let s1 := { s with fl := (X86Int.aluF .and 0 0).2, rip := 4197540 }
    have hce : X86Int.cond s1.fl 4 = some true := by
      dsimp [s1]; rw [cond_aluF_and_self_e]; rfl
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 10 rfl (by decide : (4197540, (.ofInt (.jcc 4 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hce]; rfl
    let s2 := { s1 with rip := SporosVmBin.L_prim_epilogue }
    have hwf2 : vc.wf pr { ls₀ with mem := (fun x =>
        if dst ≤ x ∧ x < dst + n then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
        else ls₀.mem x) } s2 := by
      have hmeq : mem_0 = fun x => if dst ≤ x ∧ x < dst + n then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩
        else ls₀.mem x := by funext x; simp [mem_0]
      rw [← hmeq]
      exact vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1
    exact ⟨vc, s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl, hwf2⟩
  | succ k ih =>
    intro vc s hkn mem_k hrip h hr6 hr7 hr10 hr11 hr12 hr13
    have hmi := wf_mem_info h
    have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
    have hW44 : 2 ^ 44 < X86Env.W := by change 2 ^ 44 < 2 ^ 64; omega
    have hsk : src + k < ls₀.msize := by omega
    have hdk : dst + k < ls₀.msize := by omega
    have hmw_len : ls₀.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls₀.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have hmem_src : mem_k (src + k) = ls₀.mem (src + k) := by
      dsimp [mem_k]; rw [if_neg (by omega)]
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .and (k + 1) (k + 1)).2, rip := 4197540 } := by
      rw [step_at_vc 10 hrip (by decide : (4197537, (.ofInt (.test 11 11)), 3) ∈ vmTableChunk 10),
        exec_test hr11 hr11, hrip]
    let s1 := { s with fl := (X86Int.aluF .and (k + 1) (k + 1)).2, rip := 4197540 }
    have hce : X86Int.cond s1.fl 4 = some false := by
      dsimp [s1]; rw [cond_aluF_and_self_e, decide_eq_false (Nat.succ_ne_zero k)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4197546 } := by
      rw [step_at_vc 10 rfl (by decide : (4197540, (.ofInt (.jcc 4 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hce]; rfl
    let s2 := { s1 with rip := 4197546 }
    have hk1W : k + 1 < X86Env.W := by have : ls₀.msize < 2 ^ 44 := hmi.1.2.1; omega
    have hsub_val : (X86Int.aluF .sub (k + 1) 1).1 = k := by
      simp only [X86Int.aluF]
      have : k + 1 + X86Env.W - 1 = k + X86Env.W := by omega
      rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have h03 : X86Vm.step vmImage s2 =
        .ok { s2.set 11 (some k) with fl := (X86Int.aluF .sub (k + 1) 1).2, rip := 4197550 } := by
      rw [step_at_vc 10 rfl (by decide : (4197546, (.ofInt (.aluI .sub 11 1)), 4) ∈ vmTableChunk 10),
        exec_aluI (s := s2) hr11]
      simp only [Nat.mod_eq_of_lt (by omega : 1 < X86Env.W), hsub_val]; rfl
    let s3 := { s2.set 11 (some k) with fl := (X86Int.aluF .sub (k + 1) 1).2, rip := 4197550 }
    have hr3_ok : RegsOk s3 := regsOk_set h.2.2.2.1 (by decide) _
    have hs3_11 : s3.reg 11 = some k := reg_set_self_ok' h.2.2.2.1 (by decide) _
    have hs3_6 : s3.reg 6 = some (vc.memwReg.base + src) := by
      change (s2.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact hr6
    have hea_sw : eaIdx (vc.memwReg.base + src) k 1 0 = vc.memwReg.base + (src + k) := by
      have hltW : vc.memwReg.base + src + 1 * k + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_dw : eaIdx (vc.memwReg.base + dst) k 1 0 = vc.memwReg.base + (dst + k) := by
      have hltW : vc.memwReg.base + dst + 1 * k + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_sp : eaIdx (vc.mempReg.base + src * 8) k 8 0 = vc.mempReg.base + (src + k) * 8 := by
      have hltW : vc.mempReg.base + src * 8 + 8 * k + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_dp : eaIdx (vc.mempReg.base + dst * 8) k 8 0 = vc.mempReg.base + (dst + k) * 8 := by
      have hltW : vc.mempReg.base + dst * 8 + 8 * k + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hrd_w : rdMem .w8 s3.mem (eaIdx (vc.memwReg.base + src) k 1 0) = some ((ls₀.mem (src + k)).w % 256) := by
      change rdMem .w8 s.mem _ = _; rw [hea_sw]; simpa [hmem_src] using rd_mem_byte_at h hsk
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some ((ls₀.mem (src + k)).w % 256)) with rip := 4197555 } := by
      rw [step_at_vc 10 rfl (by decide : (4197550, (.ldIdx .w8 0 6 11 1 0), 5) ∈ vmTableChunk 10),
        exec_ldIdx (s := s3) hs3_6 hs3_11 hrd_w]
    let s4 := { s3.set 0 (some ((ls₀.mem (src + k)).w % 256)) with rip := 4197555 }
    have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
    have hs4_0 : s4.reg 0 = some ((ls₀.mem (src + k)).w % 256) := reg_set_self_ok' hr3_ok (by decide) _
    have hs4_7 : s4.reg 7 = some (vc.memwReg.base + dst) := by
      change (s3.set 0 _).reg 7 = _; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hr7
    have hs4_11 : s4.reg 11 = some k := by change (s3.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs3_11
    have hwlt_src := (vc.code_flat h).2.2.2.2.2.2.2.1 (src + k)
    dsimp only at hwlt_src
    rw [hmem_src] at hwlt_src
    let q' : Nat := ((ls₀.mem (src + k)).plen + q) % Sporos.M
    have hq'M : q' < Sporos.M := Nat.mod_lt _ (by rw [M_eq_W']; change 0 < 2 ^ 64; omega)
    obtain ⟨m1, m2, hwr1, hwr2, _, _, _, _, _, _, hwf_pair⟩ :=
      wr_mem_pair_step h hdk (v := (ls₀.mem (src + k)).w) (q := q') hwlt_src.1 hq'M
    have hwr1_b : wrMem .w8 s4.mem (eaIdx (vc.memwReg.base + dst) k 1 0) ((ls₀.mem (src + k)).w % 256) = some m1 := by
      change wrMem .w8 s.mem _ _ = _; rw [hea_dw]
      simpa [X86Vm.wrMem, Nat.mod_mod] using hwr1
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with mem := m1, rip := 4197559 } := by
      rw [step_at_vc 10 rfl (by decide : (4197555, (.stIdx .w8 7 11 1 0 0), 4) ∈ vmTableChunk 10),
        exec_stIdx hs4_7 hs4_11 hs4_0 hwr1_b]
    let s5 := { s4 with mem := m1, rip := 4197559 }
    have hne_mw_mp : vc.memwReg.base ≠ vc.mempReg.base :=
      vc.ne_bases h (i := 10) (j := 11) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hmw1_wf : m1.wf := by
      obtain ⟨_, m1', hwr1', _, _, hm1'wf, _, _⟩ :=
        wrMem_w8_region h.1.2 (vc.mem_memw h) hmi.1.1 (a := vc.memwReg.base + (dst + k)) (Nat.le_add_right _ _)
          (by simp only [Region.hi]; omega) (v := (ls₀.mem (src + k)).w)
      rw [hwr1] at hwr1'; cases hwr1'; exact hm1'wf
    have hmp_in_m1 : vc.mempReg ∈ m1 := by
      obtain ⟨_, m1', hwr1', _, _, _, _, hoth1'⟩ :=
        wrMem_w8_region h.1.2 (vc.mem_memw h) hmi.1.1 (a := vc.memwReg.base + (dst + k)) (Nat.le_add_right _ _)
          (by simp only [Region.hi]; omega) (v := (ls₀.mem (src + k)).w)
      rw [hwr1] at hwr1'; cases hwr1'
      exact hoth1' vc.mempReg (vc.mem_memp h) (Ne.symm hne_mw_mp)
    have hrd_p : rdMem .w64 m1 (eaIdx (vc.mempReg.base + src * 8) k 8 0) = some ((ls₀.mem (src + k)).plen % X86Env.W) := by
      rw [hea_sp]
      have hrel := (hmi.2.2.1.2.2 (src + k) hsk).2
      dsimp only at hrel
      rw [hmem_src] at hrel
      exact rdMem_w64_at hmw1_wf hmp_in_m1 (Nat.le_add_right _ _)
        (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact hrel)
    have hs5_12 : s5.reg 12 = some (vc.mempReg.base + src * 8) := by
      change (s3.set 0 _).reg 12 = _; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 12 = _; rw [reg_set_ne (by decide)]; exact hr12
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 0 (some ((ls₀.mem (src + k)).plen % X86Env.W)) with rip := 4197563 } := by
      rw [step_at_vc 10 rfl (by decide : (4197559, (.ldIdx .w64 0 12 11 8 0), 4) ∈ vmTableChunk 10),
        exec_ldIdx (s := s5) hs5_12 hs4_11 hrd_p]
    let s6 := { s5.set 0 (some ((ls₀.mem (src + k)).plen % X86Env.W)) with rip := 4197563 }
    have hr6_ok : RegsOk s6 := regsOk_set hr4_ok (by decide) _
    have hs6_0 : s6.reg 0 = some ((ls₀.mem (src + k)).plen % X86Env.W) := reg_set_self_ok' hr4_ok (by decide) _
    have hs6_10 : s6.reg 10 = some q := by
      change (s5.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 10 = _; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hr10
    have hadd_q : (X86Int.aluF .add ((ls₀.mem (src + k)).plen % X86Env.W) q).1 = q' := by
      dsimp [q']; simp only [aluF_add_val, Nat.mod_add_mod, M_eq_W']
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 0 (some q') with fl := (X86Int.aluF .add ((ls₀.mem (src + k)).plen % X86Env.W) q).2, rip := 4197566 } := by
      rw [step_at_vc 10 rfl (by decide : (4197563, (.ofInt (.alu .add 0 10)), 3) ∈ vmTableChunk 10),
        exec_alu hs6_0 hs6_10, ← hadd_q]
    let s7 := { s6.set 0 (some q') with fl := (X86Int.aluF .add ((ls₀.mem (src + k)).plen % X86Env.W) q).2, rip := 4197566 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hs7_0 : s7.reg 0 = some q' := reg_set_self_ok' hr6_ok (by decide) _
    have hs7_13 : s7.reg 13 = some (vc.mempReg.base + dst * 8) := by
      change (s6.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s5.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 13 = _; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 13 = _; rw [reg_set_ne (by decide)]; exact hr13
    have hs7_11 : s7.reg 11 = some k := by
      change (s6.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s5.set 0 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs4_11
    have hwr2' : wrMem .w64 s7.mem (eaIdx (vc.mempReg.base + dst * 8) k 8 0) q' = some m2 := by
      change wrMem .w64 m1 _ _ = _; rw [hea_dp]; exact hwr2
    have h08 : X86Vm.step vmImage s7 = .ok { s7 with mem := m2, rip := 4197571 } := by
      rw [step_at_vc 10 rfl (by decide : (4197566, (.stIdx .w64 13 11 8 0 0), 5) ∈ vmTableChunk 10),
        exec_stIdx hs7_13 hs7_11 hs7_0 hwr2']
    let s8 := { s7 with mem := m2, rip := 4197571 }
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4197537 } := by
      rw [step_at_vc 10 rfl (by decide : (4197571, (.ofInt (.jmp 4197537)), 2) ∈ vmTableChunk 10), exec_jmp]
    let s9 := { s8 with rip := 4197537 }
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
          (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))))))
    have hs9_keep : ∀ r, r ≠ 0 → r ≠ 11 → s9.reg r = s.reg r := by
      intro r h0 h11'
      change (s6.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s5.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s3.set 0 _).reg r = _; rw [reg_set_ne h0.symm]
      change (s2.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]; rfl
    have hmem_step : (fun x => if x = dst + k then (⟨(ls₀.mem (src + k)).w, q'⟩ : SporosProvLen.LVal) else mem_k x) =
        (fun x => if dst + k ≤ x ∧ x < dst + n then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩ else ls₀.mem x) := by
      funext x
      by_cases h1 : x = dst + k
      · subst h1; simp [q', Nat.add_sub_cancel_left]; omega
      · dsimp [mem_k]
        by_cases h2 : dst + (k + 1) ≤ x ∧ x < dst + n
        · have h3 : dst + k ≤ x ∧ x < dst + n := by omega
          simp [h1, h2, h3]
        · have h3 : ¬ (dst + k ≤ x ∧ x < dst + n) := by omega
          simp [h1, h2, h3]
    let vc1 := { vc with memwReg := { vc.memwReg with data := writeBytes vc.memwReg.data (dst + k) [(ls₀.mem (src + k)).w % 256] },
                         mempReg := { vc.mempReg with data := writeBytes vc.mempReg.data ((dst + k) * 8) (bytes8 q') } }
    have hwf9 : vc1.wf pr { ls₀ with mem := (fun x =>
        if dst + k ≤ x ∧ x < dst + n then
          ⟨(ls₀.mem (src + (x - dst))).w, ((ls₀.mem (src + (x - dst))).plen + q) % Sporos.M⟩ else ls₀.mem x) } s9 := by
      rw [← hmem_step]
      exact vc1.wf_step_misc hwf_pair rfl hr7_ok
        (by rw [hs9_keep 4 (by decide) (by decide)]; exact h.2.2.2.2.1)
        (by rw [hs9_keep 15 (by decide) (by decide)]; exact h.2.2.2.2.2.1)
    obtain ⟨vc', sf, hrchf, hripf, hkf, hctxf, hwff⟩ :=
      ih vc1 s9 (by omega) rfl hwf9
        (by rw [hs9_keep 6 (by decide) (by decide)]; exact hr6)
        (by rw [hs9_keep 7 (by decide) (by decide)]; exact hr7)
        (by rw [hs9_keep 10 (by decide) (by decide)]; exact hr10)
        hs7_11
        (by rw [hs9_keep 12 (by decide) (by decide)]; exact hr12)
        (by rw [hs9_keep 13 (by decide) (by decide)]; exact hr13)
    exact ⟨vc', sf, Reaches.trans hrch9 hrchf, hripf, hkf, hctxf, hwff⟩

theorem P_copy_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_copy) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w) :
    PrimPostOutcome pr ls f rest s .copy := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hmi := wf_mem_info h
  have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
  have htake : ls.ds.take 3 = [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 =>
      cases hd1 : r0 with
      | nil => simp [hd0, hd1] at hle
      | cons a1 r1 =>
        cases hd2 : r1 with
        | nil => simp [hd0, hd1, hd2] at hle
        | cons a2 r2 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hnW : ls.ds[0]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hdstW : ls.ds[1]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hsrcW : ls.ds[2]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4197380 } := by
    rw [step_at_vc 9 hrip (by decide : (SporosVmBin.P_copy, (.ofInt (.ld 1 15 296)), 7) ∈ vmTableChunk 9),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4197380 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some ls.ds[1]!.w) with rip := 4197387 } := by
    rw [step_at_vc 9 rfl (by decide : (4197380, (.ofInt (.ld 2 15 312)), 7) ∈ vmTableChunk 9), exec_ld hs1_15 h312]
  let s2 := { s1.set 2 (some ls.ds[1]!.w) with rip := 4197387 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some ls.ds[2]!.w) with rip := 4197394 } := by
    rw [step_at_vc 9 rfl (by decide : (4197387, (.ofInt (.ld 0 15 328)), 7) ∈ vmTableChunk 9), exec_ld hs2_15 h328]
  let s3 := { s2.set 0 (some ls.ds[2]!.w) with rip := 4197394 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by change (s2.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := rd_ctx_msize h
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some ls.msize) with rip := 4197398 } := by
    rw [step_at_vc 9 rfl (by decide : (4197394, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 9), exec_ld hs3_15 hms]
  let s4 := { s3.set 10 (some ls.msize) with rip := 4197398 }
  have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
  have hrch4 : Reaches vmImage s s4 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
  have hwf4 : vc.wf pr ls s4 :=
    vc.wf_step_misc h rfl hr4_ok
      (by change (s3.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s2.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s3.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs3_15)
  have hs4_0 : s4.reg 0 = some ls.ds[2]!.w := by change (s3.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr2_ok (by decide) _
  have hs4_1 : s4.reg 1 = some ls.ds[0]!.w := by
    change (s3.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs4_2 : s4.reg 2 = some ls.ds[1]!.w := by
    change (s3.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs4_10 : s4.reg 10 = some ls.msize := reg_set_self_ok' hr3_ok (by decide) _
  have hrc1 := range_check_spec (rip0 := 4197398) (c0 := 9) (c1 := 9) (c2 := 9) (c3 := 9) (c4 := 9) (c5 := 9)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf4 hs4_0 hs4_1 hs4_10 hsrcW hnW (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases hsrc_ok : ls.ds[2]!.w + ls.ds[0]!.w ≤ ls.msize
  · rw [if_pos hsrc_ok] at hrc1
    obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hr5_ok, hs5_4, hs5_15, _, hs5_oth⟩ := hrc1
    have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc hwf4 hmem5 hr5_ok hs5_4 hs5_15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 10 (some ls.msize) with rip := 4197426 } := by
      rw [step_at_vc 9 hrip5 (by decide : (4197422, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 9),
        exec_ld hs5_15 (by rw [hmem5]; exact hms), hrip5]
    let s6 := { s5.set 10 (some ls.msize) with rip := 4197426 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hwf6 : vc.wf pr ls s6 :=
      vc.wf_step_misc hwf5 rfl hr6_ok
        (by change (s5.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact hs5_4)
        (by change (s5.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs5_15)
    have hs6_0 : s6.reg 0 = some ls.ds[2]!.w := by change (s5.set 10 _).reg 0 = _; rw [reg_set_ne (by decide), hs5_oth 0 (by decide)]; exact hs4_0
    have hs6_1 : s6.reg 1 = some ls.ds[0]!.w := by change (s5.set 10 _).reg 1 = _; rw [reg_set_ne (by decide), hs5_oth 1 (by decide)]; exact hs4_1
    have hs6_2 : s6.reg 2 = some ls.ds[1]!.w := by change (s5.set 10 _).reg 2 = _; rw [reg_set_ne (by decide), hs5_oth 2 (by decide)]; exact hs4_2
    have hs6_10 : s6.reg 10 = some ls.msize := reg_set_self_ok' hr5_ok (by decide) _
    have hrc2 := range_check_spec (rip0 := 4197426) (c0 := 9) (c1 := 9) (c2 := 9) (c3 := 9) (c4 := 9) (c5 := 9)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      rfl hwf6 hs6_2 hs6_1 hs6_10 hdstW hnW (by decide) (by decide)
    by_cases hdst_ok : ls.ds[1]!.w + ls.ds[0]!.w ≤ ls.msize
    · rw [if_pos ⟨hsrc_ok, hdst_ok⟩]
      rw [if_pos hdst_ok] at hrc2
      obtain ⟨s7, hrch7, hrip7, hmem7, hk7, hr7_ok, hs7_4, hs7_15, hs7_11, hs7_oth⟩ := hrc2
      have hrch07 : Reaches vmImage s s7 := Reaches.trans hrch4 (Reaches.trans hrch5 (Reaches.step_trans h06 hrch7))
      have hmem07 : s7.mem = s.mem := by rw [hmem7]; exact hmem5
      have hk07 : s7.k = s.k := by rw [hk7]; exact hk5
      have hwf7 : vc.wf pr ls s7 := vc.wf_step_misc hwf6 hmem7 hr7_ok hs7_4 hs7_15
      have hs7_0 : s7.reg 0 = some ls.ds[2]!.w := by rw [hs7_oth 0 (by decide)]; exact hs6_0
      have hs7_1 : s7.reg 1 = some ls.ds[0]!.w := by rw [hs7_oth 1 (by decide)]; exact hs6_1
      have hs7_2 : s7.reg 2 = some ls.ds[1]!.w := by rw [hs7_oth 2 (by decide)]; exact hs6_2
      have h08 : X86Vm.step vmImage s7 = .ok { s7 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4197453 } := by
        rw [step_at_vc 9 hrip7 (by decide : (4197450, (.ofInt (.test 1 1)), 3) ∈ vmTableChunk 9),
          exec_test hs7_1 hs7_1, hrip7]
      let s8 := { s7 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4197453 }
      by_cases hn0 : ls.ds[0]!.w = 0
      · have hce : X86Int.cond s8.fl 4 = some true := by dsimp [s8]; rw [cond_aluF_and_self_e, decide_eq_true hn0]
        have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := SporosVmBin.L_prim_epilogue } := by
          rw [step_at_vc 9 rfl (by decide : (4197453, (.ofInt (.jcc 4 4195233)), 6) ∈ vmTableChunk 9), exec_jcc hce]; rfl
        let s9 := { s8 with rip := SporosVmBin.L_prim_epilogue }
        have hrch09 : Reaches vmImage s s9 := Reaches.trans hrch07 (Reaches.step_trans h08 (Reaches.step h09))
        have hmem_id : (fun x => if ls.ds[1]!.w ≤ x ∧ x < ls.ds[1]!.w + ls.ds[0]!.w then
            ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w,
             ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] ls.pc) % Sporos.M⟩
            else ls.mem x) = ls.mem := by funext x; rw [if_neg (by omega)]
        rw [hmem_id]
        have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc hwf7 rfl hr7_ok hs7_4 hs7_15
        have hepi := L_prim_epilogue_oracle_spec (vc := vc) (pr := pr) (ls := ls) (s := s9)
          (e_in := 3) (outs := []) (f := f) (rest := rest)
          rfl hwf9 (oracleAdmits_of_reaches hor hrch09) hrs hpcW hle (by decide)
          (by intro _ hv; cases hv) (by change rd64 s7.mem _ = _; rw [hmem07]; exact h280)
          (by change rd64 s7.mem _ = _; rw [hmem07]; exact h288) (by intro _ hi; cases hi)
        rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
        · exact Or.inl ⟨sf, Reaches.trans hrch09 hrchf, by rw [hkf]; exact congrArg X86Env.Kern.out hk07, horf, hrelf⟩
        · exact Or.inr ⟨hitsNomem_of_reaches hrch09 hhit, Ends.of_reaches hrch09 hends⟩
      · have hce : X86Int.cond s8.fl 4 = some false := by dsimp [s8]; rw [cond_aluF_and_self_e, decide_eq_false hn0]
        have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4197459 } := by
          rw [step_at_vc 9 rfl (by decide : (4197453, (.ofInt (.jcc 4 4195233)), 6) ∈ vmTableChunk 9), exec_jcc hce]; rfl
        let s9 := { s8 with rip := 4197459 }
        have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
        have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
        have h10 : X86Vm.step vmImage s9 = .ok { s9.set 8 (some vc.memwReg.base) with rip := 4197463 } := by
          rw [step_at_vc 9 rfl (by decide : (4197459, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk 9),
            exec_ld (s := s9) hs7_15 (by change rd64 s7.mem _ = _; rw [hmem07]; exact rd_ctx_memwBase h)]
        let s10 := { s9.set 8 (some vc.memwReg.base) with rip := 4197463 }
        have hr10_ok : RegsOk s10 := regsOk_set hr7_ok (by decide) _
        have hs10_8 : s10.reg 8 = some vc.memwReg.base := reg_set_self_ok' hr7_ok (by decide) _
        have hs10_0 : s10.reg 0 = some ls.ds[2]!.w := by change (s9.set 8 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs7_0
        have hea6 : eaIdx vc.memwReg.base ls.ds[2]!.w 1 0 = vc.memwReg.base + ls.ds[2]!.w := by
          have hltW : vc.memwReg.base + 1 * ls.ds[2]!.w + 0 < X86Env.W := by
            have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
          rw [eaIdx_add_eq hltW]; omega
        have h11 : X86Vm.step vmImage s10 = .ok { s10.set 6 (some (vc.memwReg.base + ls.ds[2]!.w)) with rip := 4197467 } := by
          rw [step_at_vc 9 rfl (by decide : (4197463, (.leaIdx 6 8 0 1 0), 4) ∈ vmTableChunk 9), exec_leaIdx hs10_8 hs10_0, hea6]
        let s11 := { s10.set 6 (some (vc.memwReg.base + ls.ds[2]!.w)) with rip := 4197467 }
        have hr11_ok : RegsOk s11 := regsOk_set hr10_ok (by decide) _
        have hs11_8 : s11.reg 8 = some vc.memwReg.base := by change (s10.set 6 _).reg 8 = _; rw [reg_set_ne (by decide)]; exact hs10_8
        have hs11_2 : s11.reg 2 = some ls.ds[1]!.w := by
          change (s10.set 6 _).reg 2 = _; rw [reg_set_ne (by decide)]
          change (s9.set 8 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hs7_2
        have hea7 : eaIdx vc.memwReg.base ls.ds[1]!.w 1 0 = vc.memwReg.base + ls.ds[1]!.w := by
          have hltW : vc.memwReg.base + 1 * ls.ds[1]!.w + 0 < X86Env.W := by
            have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
          rw [eaIdx_add_eq hltW]; omega
        have h12 : X86Vm.step vmImage s11 = .ok { s11.set 7 (some (vc.memwReg.base + ls.ds[1]!.w)) with rip := 4197471 } := by
          rw [step_at_vc 9 rfl (by decide : (4197467, (.leaIdx 7 8 2 1 0), 4) ∈ vmTableChunk 9), exec_leaIdx hs11_8 hs11_2, hea7]
        let s12 := { s11.set 7 (some (vc.memwReg.base + ls.ds[1]!.w)) with rip := 4197471 }
        have hr12_ok : RegsOk s12 := regsOk_set hr11_ok (by decide) _
        have hs12_15 : s12.reg 15 = some ctxBase := by
          change (s11.set 7 _).reg 15 = _; rw [reg_set_ne (by decide)]
          change (s10.set 6 _).reg 15 = _; rw [reg_set_ne (by decide)]
          change (s9.set 8 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs7_15
        have h13 : X86Vm.step vmImage s12 = .ok { s12.set 9 (some vc.mempReg.base) with rip := 4197475 } := by
          rw [step_at_vc 9 rfl (by decide : (4197471, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk 9),
            exec_ld hs12_15 (by change rd64 s7.mem _ = _; rw [hmem07]; exact rd_ctx_mempBase h)]
        let s13 := { s12.set 9 (some vc.mempReg.base) with rip := 4197475 }
        have hr13_ok : RegsOk s13 := regsOk_set hr12_ok (by decide) _
        have hs13_9 : s13.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr12_ok (by decide) _
        have hs13_0 : s13.reg 0 = some ls.ds[2]!.w := by
          change (s12.set 9 _).reg 0 = _; rw [reg_set_ne (by decide)]
          change (s11.set 7 _).reg 0 = _; rw [reg_set_ne (by decide)]
          change (s10.set 6 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs10_0
        have hea12 : eaIdx vc.mempReg.base ls.ds[2]!.w 8 0 = vc.mempReg.base + ls.ds[2]!.w * 8 := by
          have hltW : vc.mempReg.base + 8 * ls.ds[2]!.w + 0 < X86Env.W := by
            have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
          rw [eaIdx_add_eq hltW]; omega
        have h14 : X86Vm.step vmImage s13 = .ok { s13.set 12 (some (vc.mempReg.base + ls.ds[2]!.w * 8)) with rip := 4197479 } := by
          rw [step_at_vc 9 rfl (by decide : (4197475, (.leaIdx 12 9 0 8 0), 4) ∈ vmTableChunk 9), exec_leaIdx hs13_9 hs13_0, hea12]
        let s14 := { s13.set 12 (some (vc.mempReg.base + ls.ds[2]!.w * 8)) with rip := 4197479 }
        have hr14_ok : RegsOk s14 := regsOk_set hr13_ok (by decide) _
        have hs14_9 : s14.reg 9 = some vc.mempReg.base := by change (s13.set 12 _).reg 9 = _; rw [reg_set_ne (by decide)]; exact hs13_9
        have hs14_2 : s14.reg 2 = some ls.ds[1]!.w := by
          change (s13.set 12 _).reg 2 = _; rw [reg_set_ne (by decide)]
          change (s12.set 9 _).reg 2 = _; rw [reg_set_ne (by decide)]
          change (s11.set 7 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hs11_2
        have hea13 : eaIdx vc.mempReg.base ls.ds[1]!.w 8 0 = vc.mempReg.base + ls.ds[1]!.w * 8 := by
          have hltW : vc.mempReg.base + 8 * ls.ds[1]!.w + 0 < X86Env.W := by
            have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
          rw [eaIdx_add_eq hltW]; omega
        have h15' : X86Vm.step vmImage s14 = .ok { s14.set 13 (some (vc.mempReg.base + ls.ds[1]!.w * 8)) with rip := 4197483 } := by
          rw [step_at_vc 9 rfl (by decide : (4197479, (.leaIdx 13 9 2 8 0), 4) ∈ vmTableChunk 9), exec_leaIdx hs14_9 hs14_2, hea13]
        let s15 := { s14.set 13 (some (vc.mempReg.base + ls.ds[1]!.w * 8)) with rip := 4197483 }
        have hr15_ok : RegsOk s15 := regsOk_set hr14_ok (by decide) _
        have hs15_15 : s15.reg 15 = some ctxBase := by
          change (s14.set 13 _).reg 15 = _; rw [reg_set_ne (by decide)]
          change (s13.set 12 _).reg 15 = _; rw [reg_set_ne (by decide)]
          change (s12.set 9 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs12_15
        let q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] ls.pc
        have h16 : X86Vm.step vmImage s15 = .ok { s15.set 10 (some q) with rip := 4197490 } := by
          rw [step_at_vc 9 rfl (by decide : (4197483, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk 9),
            exec_ld hs15_15 (by change rd64 s7.mem _ = _; rw [hmem07]; rw [htake] at h272; exact h272)]
        let s16 := { s15.set 10 (some q) with rip := 4197490 }
        have hr16_ok : RegsOk s16 := regsOk_set hr15_ok (by decide) _
        have hs16_2 : s16.reg 2 = some ls.ds[1]!.w := by
          change (s15.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]
          change (s14.set 13 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hs14_2
        have hs16_0 : s16.reg 0 = some ls.ds[2]!.w := by
          change (s15.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]
          change (s14.set 13 _).reg 0 = _; rw [reg_set_ne (by decide)]
          change (s13.set 12 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs13_0
        have h17 : X86Vm.step vmImage s16 =
            .ok { s16 with fl := (X86Int.aluF .sub ls.ds[1]!.w ls.ds[2]!.w).2, rip := 4197493 } := by
          rw [step_at_vc 9 rfl (by decide : (4197490, (.ofInt (.cmp 2 0)), 3) ∈ vmTableChunk 9), exec_cmp hs16_2 hs16_0]
        let s17 := { s16 with fl := (X86Int.aluF .sub ls.ds[1]!.w ls.ds[2]!.w).2, rip := 4197493 }
        have hrch17 : Reaches vmImage s s17 :=
          Reaches.trans hrch07 (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10
            (Reaches.step_trans h11 (Reaches.step_trans h12 (Reaches.step_trans h13
              (Reaches.step_trans h14 (Reaches.step_trans h15' (Reaches.step_trans h16 (Reaches.step h17))))))))))
        have hs17_keep : ∀ r, r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 12 → r ≠ 13 → s17.reg r = s7.reg r := by
          intro r h6 h7 h8 h9 h10' h12' h13'
          change (s15.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
          change (s14.set 13 _).reg r = _; rw [reg_set_ne h13'.symm]
          change (s13.set 12 _).reg r = _; rw [reg_set_ne h12'.symm]
          change (s12.set 9 _).reg r = _; rw [reg_set_ne h9.symm]
          change (s11.set 7 _).reg r = _; rw [reg_set_ne h7.symm]
          change (s10.set 6 _).reg r = _; rw [reg_set_ne h6.symm]
          change (s9.set 8 _).reg r = _; rw [reg_set_ne h8.symm]; rfl
        have hs17_6 : s17.reg 6 = some (vc.memwReg.base + ls.ds[2]!.w) := by
          change (s15.set 10 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s14.set 13 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s13.set 12 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s12.set 9 _).reg 6 = _; rw [reg_set_ne (by decide)]
          change (s11.set 7 _).reg 6 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr10_ok (by decide) _
        have hs17_7 : s17.reg 7 = some (vc.memwReg.base + ls.ds[1]!.w) := by
          change (s15.set 10 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s14.set 13 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s13.set 12 _).reg 7 = _; rw [reg_set_ne (by decide)]
          change (s12.set 9 _).reg 7 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr11_ok (by decide) _
        have hs17_10 : s17.reg 10 = some q := reg_set_self_ok' hr15_ok (by decide) _
        have hs17_12 : s17.reg 12 = some (vc.mempReg.base + ls.ds[2]!.w * 8) := by
          change (s15.set 10 _).reg 12 = _; rw [reg_set_ne (by decide)]
          change (s14.set 13 _).reg 12 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr13_ok (by decide) _
        have hs17_13 : s17.reg 13 = some (vc.mempReg.base + ls.ds[1]!.w * 8) := by
          change (s15.set 10 _).reg 13 = _; rw [reg_set_ne (by decide)]
          exact reg_set_self_ok' hr14_ok (by decide) _
        have hca : X86Int.cond s17.fl 7 = some (decide (ls.ds[2]!.w < ls.ds[1]!.w)) := by
          dsimp [s17]; exact cond_aluF_sub_a _ _ hdstW hsrcW
        let ls_done : SporosProvLen.LSt := { ls with mem := (fun x =>
          if ls.ds[1]!.w ≤ x ∧ x < ls.ds[1]!.w + ls.ds[0]!.w then
            ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w, ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + q) % Sporos.M⟩
          else ls.mem x) }
        have hloop_res : ∃ (vc' : VmCtx) (sf : X86Vm.St), Reaches vmImage s17 sf ∧ sf.rip = SporosVmBin.L_prim_epilogue ∧
            sf.k = s17.k ∧ vc'.ctxReg = vc.ctxReg ∧ vc'.wf pr ls_done sf := by
          by_cases hgt : ls.ds[2]!.w < ls.ds[1]!.w
          · have h18 : X86Vm.step vmImage s17 = .ok { s17 with rip := 4197534 } := by
              rw [step_at_vc 9 rfl (by decide : (4197493, (.ofInt (.jcc 7 4197534)), 2) ∈ vmTableChunk 9),
                exec_jcc (by rw [hca, decide_eq_true hgt])]; rfl
            let s18 := { s17 with rip := 4197534 }
            have hs18_1 : s18.reg 1 = some ls.ds[0]!.w := by
              change s17.reg 1 = _; rw [hs17_keep 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_1
            have h19 : X86Vm.step vmImage s18 = .ok { s18.set 11 (some ls.ds[0]!.w) with rip := 4197537 } := by
              rw [step_at_vc 10 rfl (by decide : (4197534, (.ofInt (.mov 11 1)), 3) ∈ vmTableChunk 10), exec_mov hs18_1]
            let s19 := { s18.set 11 (some ls.ds[0]!.w) with rip := 4197537 }
            have hr19_ok : RegsOk s19 := regsOk_set hr16_ok (by decide) _
            have hmem_init : (fun x => if ls.ds[1]!.w + ls.ds[0]!.w ≤ x ∧ x < ls.ds[1]!.w + ls.ds[0]!.w then
                ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w, ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + q) % Sporos.M⟩
                else ls.mem x) = ls.mem := by funext x; rw [if_neg (by omega)]
            have hwf19 : vc.wf pr { ls with mem := (fun x =>
                if ls.ds[1]!.w + ls.ds[0]!.w ≤ x ∧ x < ls.ds[1]!.w + ls.ds[0]!.w then
                  ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w, ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + q) % Sporos.M⟩
                else ls.mem x) } s19 := by
              rw [hmem_init]
              exact vc.wf_step_misc hwf7 rfl hr19_ok
                (by change (s18.set 11 _).reg 4 = _; rw [reg_set_ne (by decide)]
                    change s17.reg 4 = _; rw [hs17_keep 4 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_4)
                (by change (s18.set 11 _).reg 15 = _; rw [reg_set_ne (by decide)]
                    change s17.reg 15 = _; rw [hs17_keep 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_15)
            obtain ⟨vc', sf, hrchf, hripf, hkf, hctxf, hwff⟩ :=
              copy_bkw_loop_spec hdst_ok hgt ls.ds[0]!.w vc s19 (Nat.le_refl _) rfl hwf19
                (by change (s18.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact hs17_6)
                (by change (s18.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hs17_7)
                (by change (s18.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hs17_10)
                (reg_set_self_ok' hr16_ok (by decide) _)
                (by change (s18.set 11 _).reg 12 = _; rw [reg_set_ne (by decide)]; exact hs17_12)
                (by change (s18.set 11 _).reg 13 = _; rw [reg_set_ne (by decide)]; exact hs17_13)
            exact ⟨vc', sf, Reaches.step_trans h18 (Reaches.step_trans h19 hrchf), hripf, hkf, hctxf, hwff⟩
          · have h18 : X86Vm.step vmImage s17 = .ok { s17 with rip := 4197495 } := by
              rw [step_at_vc 9 rfl (by decide : (4197493, (.ofInt (.jcc 7 4197534)), 2) ∈ vmTableChunk 9),
                exec_jcc (by rw [hca, decide_eq_false hgt])]; rfl
            let s18 := { s17 with rip := 4197495 }
            have hs18_11 : s18.reg 11 = some (ls.msize - ls.ds[1]!.w) := by
              change s17.reg 11 = _; rw [hs17_keep 11 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_11
            have h19 : X86Vm.step vmImage s18 = .ok { s18.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[1]!.w) (ls.msize - ls.ds[1]!.w)).2, rip := 4197498 } := by
              rw [step_at_vc 10 rfl (by decide : (4197495, (.ofInt (.alu .xor 11 11)), 3) ∈ vmTableChunk 10), exec_alu_xor_self hs18_11]
            let s19 := { s18.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[1]!.w) (ls.msize - ls.ds[1]!.w)).2, rip := 4197498 }
            have hr19_ok : RegsOk s19 := regsOk_set hr16_ok (by decide) _
            have hmem_init : (fun x => if ls.ds[1]!.w ≤ x ∧ x < ls.ds[1]!.w + 0 then
                ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w, ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + q) % Sporos.M⟩
                else ls.mem x) = ls.mem := by funext x; rw [if_neg (by omega)]
            have hwf19 : vc.wf pr { ls with mem := (fun x =>
                if ls.ds[1]!.w ≤ x ∧ x < ls.ds[1]!.w + 0 then
                  ⟨(ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).w, ((ls.mem (ls.ds[2]!.w + (x - ls.ds[1]!.w))).plen + q) % Sporos.M⟩
                else ls.mem x) } s19 := by
              rw [hmem_init]
              exact vc.wf_step_misc hwf7 rfl hr19_ok
                (by change (s18.set 11 _).reg 4 = _; rw [reg_set_ne (by decide)]
                    change s17.reg 4 = _; rw [hs17_keep 4 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_4)
                (by change (s18.set 11 _).reg 15 = _; rw [reg_set_ne (by decide)]
                    change s17.reg 15 = _; rw [hs17_keep 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_15)
            obtain ⟨vc', sf, hrchf, hripf, hkf, hctxf, hwff⟩ :=
              copy_fwd_loop_spec hsrc_ok hdst_ok (Nat.le_of_not_lt hgt) ls.ds[0]!.w 0 vc s19 (Nat.zero_add _) rfl hwf19
                (by change (s18.set 11 _).reg 1 = _; rw [reg_set_ne (by decide)]
                    change s17.reg 1 = _; rw [hs17_keep 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs7_1)
                (by change (s18.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact hs17_6)
                (by change (s18.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hs17_7)
                (by change (s18.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hs17_10)
                (reg_set_self_ok' hr16_ok (by decide) _)
                (by change (s18.set 11 _).reg 12 = _; rw [reg_set_ne (by decide)]; exact hs17_12)
                (by change (s18.set 11 _).reg 13 = _; rw [reg_set_ne (by decide)]; exact hs17_13)
            exact ⟨vc', sf, Reaches.step_trans h18 (Reaches.step_trans h19 hrchf), hripf, hkf, hctxf, hwff⟩
        obtain ⟨vc', s_epi, hrch_epi, hrip_epi, hk_epi, hctx_eq, hwf_epi⟩ := hloop_res
        have hrch0_epi : Reaches vmImage s s_epi := Reaches.trans hrch17 hrch_epi
        have h280_epi : rd64 s_epi.mem (ea ctxBase 280) = some 3 :=
          vc'.rd64_ctx_of_eq hwf_epi (by decide) (by rw [hctx_eq]; exact ctx_leVal_of_rd64 h (by decide) h280)
        have h288_epi : rd64 s_epi.mem (ea ctxBase 288) = some 0 :=
          vc'.rd64_ctx_of_eq hwf_epi (by decide) (by rw [hctx_eq]; exact ctx_leVal_of_rd64 h (by decide) h288)
        have hepi := L_prim_epilogue_oracle_spec (vc := vc') (pr := pr) (ls := ls_done) (s := s_epi)
          (e_in := 3) (outs := []) (f := f) (rest := rest)
          hrip_epi hwf_epi (oracleAdmits_of_reaches hor hrch0_epi) hrs hpcW hle (by decide)
          (by intro _ hv; cases hv) h280_epi h288_epi (by intro _ hi; cases hi)
        rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
        · exact Or.inl ⟨sf, Reaches.trans hrch0_epi hrchf, by rw [hkf, hk_epi]; exact congrArg X86Env.Kern.out hk07, horf, hrelf⟩
        · exact Or.inr ⟨hitsNomem_of_reaches hrch0_epi hhit, Ends.of_reaches hrch0_epi hends⟩
    · rw [if_neg (fun hc => hdst_ok hc.2)]
      rw [if_neg hdst_ok] at hrc2
      obtain ⟨se, hrche, hexit, hke⟩ := hrc2
      exact ⟨se, Reaches.trans hrch4 (Reaches.trans hrch5 (Reaches.step_trans h06 hrche)), hexit, by rw [hke]; exact hk5⟩
  · rw [if_neg (fun hc => hsrc_ok hc.1)]
    rw [if_neg hsrc_ok] at hrc1
    obtain ⟨se, hrche, hexit, hke⟩ := hrc1
    exact ⟨se, Reaches.trans hrch4 hrche, hexit, hke⟩

private theorem fill_loop_spec {pr : Sporos.Prog} {ls₀ : SporosProvLen.LSt}
    {dst n b q : Nat} (hdst : dst + n ≤ ls₀.msize) (hqM : q < Sporos.M) :
    ∀ (rem i : Nat) (vc : VmCtx) (s : X86Vm.St),
      i + rem = n →
      let mem_i : SporosProvLen.LMem := fun x =>
        if dst ≤ x ∧ x < dst + i then ⟨b % 256, q⟩ else ls₀.mem x
      s.rip = 4197648 → vc.wf pr { ls₀ with mem := mem_i } s →
      s.reg 1 = some n → s.reg 2 = some b →
      s.reg 6 = some (vc.mempReg.base + dst * 8) →
      s.reg 7 = some (vc.memwReg.base + dst) →
      s.reg 10 = some q → s.reg 11 = some i →
      ∃ (vc' : VmCtx) (s' : X86Vm.St),
        Reaches vmImage s s' ∧ s'.rip = SporosVmBin.L_prim_epilogue ∧ s'.k = s.k ∧
        vc'.ctxReg = vc.ctxReg ∧
        vc'.wf pr { ls₀ with mem := (fun x =>
          if dst ≤ x ∧ x < dst + n then ⟨b % 256, q⟩ else ls₀.mem x) } s' := by
  intro rem
  induction rem with
  | zero =>
    intro i vc s hirem _ hrip h hr1 _ _ _ _ hr11
    have hi_eq : i = n := by omega
    subst hi_eq
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i i).2, rip := 4197651 } := by
      rw [step_at_vc 10 hrip (by decide : (4197648, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 10),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i i).2, rip := 4197651 }
    have hcae : X86Int.cond s1.fl 3 = some true := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_true (Nat.le_refl _)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := SporosVmBin.L_prim_epilogue } := by
      rw [step_at_vc 10 rfl (by decide : (4197651, (.ofInt (.jcc 3 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := SporosVmBin.L_prim_epilogue }
    exact ⟨vc, s2, Reaches.step_trans h01 (Reaches.step h02), rfl, rfl, rfl,
      vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1⟩
  | succ rem ih =>
    intro i vc s hirem mem_i hrip h hr1 hr2 hr6 hr7 hr10 hr11
    have hmi := wf_mem_info h
    have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
    have hW44 : 2 ^ 44 < X86Env.W := by change 2 ^ 44 < 2 ^ 64; omega
    have hdi : dst + i < ls₀.msize := by omega
    have hmw_len : ls₀.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls₀.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have h01 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub i n).2, rip := 4197651 } := by
      rw [step_at_vc 10 hrip (by decide : (4197648, (.ofInt (.cmp 11 1)), 3) ∈ vmTableChunk 10),
        exec_cmp hr11 hr1, hrip]
    let s1 := { s with fl := (X86Int.aluF .sub i n).2, rip := 4197651 }
    have hcae : X86Int.cond s1.fl 3 = some false := by
      dsimp [s1]; rw [cond_aluF_sub_ae, decide_eq_false (by omega)]
    have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4197657 } := by
      rw [step_at_vc 10 rfl (by decide : (4197651, (.ofInt (.jcc 3 4195233)), 6) ∈ vmTableChunk 10),
        exec_jcc hcae]; rfl
    let s2 := { s1 with rip := 4197657 }
    have hea_dw : eaIdx (vc.memwReg.base + dst) i 1 0 = vc.memwReg.base + (dst + i) := by
      have hltW : vc.memwReg.base + dst + 1 * i + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hea_dp : eaIdx (vc.mempReg.base + dst * 8) i 8 0 = vc.mempReg.base + (dst + i) * 8 := by
      have hltW : vc.mempReg.base + dst * 8 + 8 * i + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have hb256M : b % 256 < Sporos.M := by
      have h1 := Nat.mod_lt b (by decide : 0 < 256); rw [M_eq_W']; have : X86Env.W = 2 ^ 64 := rfl; omega
    obtain ⟨m1, m2, hwr1, hwr2, _, _, _, _, _, _, hwf_pair⟩ :=
      wr_mem_pair_step h hdi (v := b % 256) (q := q) hb256M hqM
    have hwr1_b : wrMem .w8 s2.mem (eaIdx (vc.memwReg.base + dst) i 1 0) b = some m1 := by
      change wrMem .w8 s.mem _ _ = _; rw [hea_dw]
      simpa [X86Vm.wrMem, Nat.mod_mod] using hwr1
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with mem := m1, rip := 4197661 } := by
      rw [step_at_vc 10 rfl (by decide : (4197657, (.stIdx .w8 7 11 1 0 2), 4) ∈ vmTableChunk 10),
        exec_stIdx (s := s2) hr7 hr11 hr2 hwr1_b]
    let s3 := { s2 with mem := m1, rip := 4197661 }
    have hwr2' : wrMem .w64 s3.mem (eaIdx (vc.mempReg.base + dst * 8) i 8 0) q = some m2 := by
      change wrMem .w64 m1 _ _ = _; rw [hea_dp]; exact hwr2
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with mem := m2, rip := 4197665 } := by
      rw [step_at_vc 10 rfl (by decide : (4197661, (.stIdx .w64 6 11 8 0 10), 4) ∈ vmTableChunk 10),
        exec_stIdx (s := s3) hr6 hr11 hr10 hwr2']
    let s4 := { s3 with mem := m2, rip := 4197665 }
    have hi1W : i + 1 < X86Env.W := by have : ls₀.msize < 2 ^ 44 := hmi.1.2.1; omega
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4197669 } := by
      rw [step_at_vc 10 rfl (by decide : (4197665, (.ofInt (.aluI .add 11 1)), 4) ∈ vmTableChunk 10),
        exec_aluI_add_lt (s := s4) hr11 (by decide), Nat.mod_eq_of_lt hi1W]
    let s5 := { s4.set 11 (some (i + 1)) with fl := (X86Int.aluF .add i 1).2, rip := 4197669 }
    have hr5_ok : RegsOk s5 := regsOk_set h.2.2.2.1 (by decide) _
    have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4197648 } := by
      rw [step_at_vc 10 rfl (by decide : (4197669, (.ofInt (.jmp 4197648)), 2) ∈ vmTableChunk 10), exec_jmp]
    let s6 := { s5 with rip := 4197648 }
    have hrch6 : Reaches vmImage s s6 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step h06)))))
    have hmem_step : (fun x => if x = dst + i then (⟨b % 256, q⟩ : SporosProvLen.LVal) else mem_i x) =
        (fun x => if dst ≤ x ∧ x < dst + (i + 1) then ⟨b % 256, q⟩ else ls₀.mem x) := by
      funext x
      by_cases h1 : x = dst + i
      · subst h1; simp
      · dsimp [mem_i]
        by_cases h2 : dst ≤ x ∧ x < dst + i
        · have h3 : dst ≤ x ∧ x < dst + (i + 1) := by omega
          simp [h1, h2, h3]
        · have h3 : ¬ (dst ≤ x ∧ x < dst + (i + 1)) := by omega
          simp [h1, h2, h3]
    let vc1 := { vc with memwReg := { vc.memwReg with data := writeBytes vc.memwReg.data (dst + i) [(b % 256) % 256] },
                         mempReg := { vc.mempReg with data := writeBytes vc.mempReg.data ((dst + i) * 8) (bytes8 q) } }
    have hwf6 : vc1.wf pr { ls₀ with mem := (fun x =>
        if dst ≤ x ∧ x < dst + (i + 1) then ⟨b % 256, q⟩ else ls₀.mem x) } s6 := by
      rw [← hmem_step]
      exact vc1.wf_step_misc hwf_pair rfl hr5_ok
        (by change (s4.set 11 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
        (by change (s4.set 11 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.2.1)
    obtain ⟨vc', sf, hrchf, hripf, hkf, hctxf, hwff⟩ :=
      ih (i + 1) vc1 s6 (by omega) rfl hwf6
        (by change (s4.set 11 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact hr1)
        (by change (s4.set 11 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact hr2)
        (by change (s4.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact hr6)
        (by change (s4.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact hr7)
        (by change (s4.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact hr10)
        (reg_set_self_ok' h.2.2.2.1 (by decide) _)
    exact ⟨vc', sf, Reaches.trans hrch6 hrchf, hripf, hkf, hctxf, hwff⟩

theorem P_fill_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_fill) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w) :
    PrimPostOutcome pr ls f rest s .fill := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hmi := wf_mem_info h
  have hW47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
  have htake : ls.ds.take 3 = [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 =>
      cases hd1 : r0 with
      | nil => simp [hd0, hd1] at hle
      | cons a1 r1 =>
        cases hd2 : r1 with
        | nil => simp [hd0, hd1, hd2] at hle
        | cons a2 r2 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hnW : ls.ds[1]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hdstW : ls.ds[2]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some ls.ds[0]!.w) with rip := 4197580 } := by
    rw [step_at_vc 10 hrip (by decide : (SporosVmBin.P_fill, (.ofInt (.ld 2 15 296)), 7) ∈ vmTableChunk 10),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 2 (some ls.ds[0]!.w) with rip := 4197580 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[1]!.w) with rip := 4197587 } := by
    rw [step_at_vc 10 rfl (by decide : (4197580, (.ofInt (.ld 1 15 312)), 7) ∈ vmTableChunk 10), exec_ld hs1_15 h312]
  let s2 := { s1.set 1 (some ls.ds[1]!.w) with rip := 4197587 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some ls.ds[2]!.w) with rip := 4197594 } := by
    rw [step_at_vc 10 rfl (by decide : (4197587, (.ofInt (.ld 0 15 328)), 7) ∈ vmTableChunk 10), exec_ld hs2_15 h328]
  let s3 := { s2.set 0 (some ls.ds[2]!.w) with rip := 4197594 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by change (s2.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := rd_ctx_msize h
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some ls.msize) with rip := 4197598 } := by
    rw [step_at_vc 10 rfl (by decide : (4197594, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 10), exec_ld hs3_15 hms]
  let s4 := { s3.set 10 (some ls.msize) with rip := 4197598 }
  have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
  have hrch4 : Reaches vmImage s s4 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
  have hwf4 : vc.wf pr ls s4 :=
    vc.wf_step_misc h rfl hr4_ok
      (by change (s3.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s2.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s3.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs3_15)
  have hs4_0 : s4.reg 0 = some ls.ds[2]!.w := by change (s3.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr2_ok (by decide) _
  have hs4_1 : s4.reg 1 = some ls.ds[1]!.w := by
    change (s3.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs4_2 : s4.reg 2 = some ls.ds[0]!.w := by
    change (s3.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs4_10 : s4.reg 10 = some ls.msize := reg_set_self_ok' hr3_ok (by decide) _
  have hrc := range_check_spec (rip0 := 4197598) (c0 := 10) (c1 := 10) (c2 := 10) (c3 := 10) (c4 := 10) (c5 := 10)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf4 hs4_0 hs4_1 hs4_10 hdstW hnW (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases hdst_ok : ls.ds[2]!.w + ls.ds[1]!.w ≤ ls.msize
  · rw [if_pos hdst_ok] at hrc ⊢
    obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hr5_ok, hs5_4, hs5_15, hs5_11, hs5_oth⟩ := hrc
    have hmw_len : ls.msize ≤ vc.memwReg.data.length := hmi.2.2.1.1
    have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hmi.2.2.1.2.1
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 8 (some vc.memwReg.base) with rip := 4197626 } := by
      rw [step_at_vc 10 hrip5 (by decide : (4197622, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk 10),
        exec_ld hs5_15 (by rw [hmem5]; exact rd_ctx_memwBase hwf4), hrip5]
    let s6 := { s5.set 8 (some vc.memwReg.base) with rip := 4197626 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hs6_8 : s6.reg 8 = some vc.memwReg.base := reg_set_self_ok' hr5_ok (by decide) _
    have hs6_0 : s6.reg 0 = some ls.ds[2]!.w := by change (s5.set 8 _).reg 0 = _; rw [reg_set_ne (by decide), hs5_oth 0 (by decide)]; exact hs4_0
    have hea7 : eaIdx vc.memwReg.base ls.ds[2]!.w 1 0 = vc.memwReg.base + ls.ds[2]!.w := by
      have hltW : vc.memwReg.base + 1 * ls.ds[2]!.w + 0 < X86Env.W := by
        have := hmi.1.2.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some (vc.memwReg.base + ls.ds[2]!.w)) with rip := 4197630 } := by
      rw [step_at_vc 10 rfl (by decide : (4197626, (.leaIdx 7 8 0 1 0), 4) ∈ vmTableChunk 10), exec_leaIdx hs6_8 hs6_0, hea7]
    let s7 := { s6.set 7 (some (vc.memwReg.base + ls.ds[2]!.w)) with rip := 4197630 }
    have hr7_ok : RegsOk s7 := regsOk_set hr6_ok (by decide) _
    have hs7_15 : s7.reg 15 = some ctxBase := by
      change (s6.set 7 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s5.set 8 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs5_15
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 9 (some vc.mempReg.base) with rip := 4197634 } := by
      rw [step_at_vc 10 rfl (by decide : (4197630, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk 10),
        exec_ld hs7_15 (by change rd64 s5.mem _ = _; rw [hmem5]; exact rd_ctx_mempBase hwf4)]
    let s8 := { s7.set 9 (some vc.mempReg.base) with rip := 4197634 }
    have hr8_ok : RegsOk s8 := regsOk_set hr7_ok (by decide) _
    have hs8_9 : s8.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr7_ok (by decide) _
    have hs8_0 : s8.reg 0 = some ls.ds[2]!.w := by
      change (s7.set 9 _).reg 0 = _; rw [reg_set_ne (by decide)]
      change (s6.set 7 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact hs6_0
    have hea6 : eaIdx vc.mempReg.base ls.ds[2]!.w 8 0 = vc.mempReg.base + ls.ds[2]!.w * 8 := by
      have hltW : vc.mempReg.base + 8 * ls.ds[2]!.w + 0 < X86Env.W := by
        have := hmi.2.1.2.2.1; simp only [Region.hi] at this; omega
      rw [eaIdx_add_eq hltW]; omega
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 6 (some (vc.mempReg.base + ls.ds[2]!.w * 8)) with rip := 4197638 } := by
      rw [step_at_vc 10 rfl (by decide : (4197634, (.leaIdx 6 9 0 8 0), 4) ∈ vmTableChunk 10), exec_leaIdx hs8_9 hs8_0, hea6]
    let s9 := { s8.set 6 (some (vc.mempReg.base + ls.ds[2]!.w * 8)) with rip := 4197638 }
    have hr9_ok : RegsOk s9 := regsOk_set hr8_ok (by decide) _
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 6 _).reg 15 = _; rw [reg_set_ne (by decide)]
      change (s7.set 9 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs7_15
    let q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] ls.pc
    have hqM : q < Sporos.M := by simp [q, SporosProvLen.lq]; exact Nat.mod_lt _ (by rw [M_eq_W']; change 0 < 2 ^ 64; omega)
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 10 (some q) with rip := 4197645 } := by
      rw [step_at_vc 10 rfl (by decide : (4197638, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk 10),
        exec_ld hs9_15 (by change rd64 s5.mem _ = _; rw [hmem5]; change rd64 s.mem _ = _; rw [htake] at h272; exact h272)]
    let s10 := { s9.set 10 (some q) with rip := 4197645 }
    have hr10_ok : RegsOk s10 := regsOk_set hr9_ok (by decide) _
    have hs10_11 : s10.reg 11 = some (ls.msize - ls.ds[2]!.w) := by
      change (s9.set 10 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s8.set 6 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s7.set 9 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s6.set 7 _).reg 11 = _; rw [reg_set_ne (by decide)]
      change (s5.set 8 _).reg 11 = _; rw [reg_set_ne (by decide)]; exact hs5_11
    have h11 : X86Vm.step vmImage s10 = .ok { s10.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[2]!.w) (ls.msize - ls.ds[2]!.w)).2, rip := 4197648 } := by
      rw [step_at_vc 10 rfl (by decide : (4197645, (.ofInt (.alu .xor 11 11)), 3) ∈ vmTableChunk 10), exec_alu_xor_self hs10_11]
    let s11 := { s10.set 11 (some 0) with fl := (X86Int.aluF .xor (ls.msize - ls.ds[2]!.w) (ls.msize - ls.ds[2]!.w)).2, rip := 4197648 }
    have hr11_ok : RegsOk s11 := regsOk_set hr10_ok (by decide) _
    have hrch11 : Reaches vmImage s s11 :=
      Reaches.trans hrch4 (Reaches.trans hrch5 (Reaches.step_trans h06 (Reaches.step_trans h07
        (Reaches.step_trans h08 (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step h11)))))))
    have hs11_keep : ∀ r, r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s11.reg r = s5.reg r := by
      intro r h6 h7 h8 h9 h10' h11'
      change (s10.set 11 _).reg r = _; rw [reg_set_ne h11'.symm]
      change (s9.set 10 _).reg r = _; rw [reg_set_ne h10'.symm]
      change (s8.set 6 _).reg r = _; rw [reg_set_ne h6.symm]
      change (s7.set 9 _).reg r = _; rw [reg_set_ne h9.symm]
      change (s6.set 7 _).reg r = _; rw [reg_set_ne h7.symm]
      change (s5.set 8 _).reg r = _; rw [reg_set_ne h8.symm]
    have hmem_init : (fun x => if ls.ds[2]!.w ≤ x ∧ x < ls.ds[2]!.w + 0 then (⟨ls.ds[0]!.w % 256, q⟩ : SporosProvLen.LVal) else ls.mem x) = ls.mem := by
      funext x; rw [if_neg (by omega)]
    have hwf11 : vc.wf pr { ls with mem := (fun x => if ls.ds[2]!.w ≤ x ∧ x < ls.ds[2]!.w + 0 then ⟨ls.ds[0]!.w % 256, q⟩ else ls.mem x) } s11 := by
      rw [hmem_init]
      exact vc.wf_step_misc hwf4 hmem5 hr11_ok
        (by rw [hs11_keep 4 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs5_4)
        (by rw [hs11_keep 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs5_15)
    obtain ⟨vc', s_epi, hrch_epi, hrip_epi, hk_epi, hctx_eq, hwf_epi⟩ :=
      fill_loop_spec hdst_ok hqM ls.ds[1]!.w 0 vc s11 (Nat.zero_add _) rfl hwf11
        (by rw [hs11_keep 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hs5_oth 1 (by decide)]; exact hs4_1)
        (by rw [hs11_keep 2 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hs5_oth 2 (by decide)]; exact hs4_2)
        (by change (s10.set 11 _).reg 6 = _; rw [reg_set_ne (by decide)]
            change (s9.set 10 _).reg 6 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr8_ok (by decide) _)
        (by change (s10.set 11 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s9.set 10 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s8.set 6 _).reg 7 = _; rw [reg_set_ne (by decide)]
            change (s7.set 9 _).reg 7 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr6_ok (by decide) _)
        (by change (s10.set 11 _).reg 10 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr9_ok (by decide) _)
        (reg_set_self_ok' hr10_ok (by decide) _)
    have hrch0_epi : Reaches vmImage s s_epi := Reaches.trans hrch11 hrch_epi
    have h280_epi : rd64 s_epi.mem (ea ctxBase 280) = some 3 :=
      vc'.rd64_ctx_of_eq hwf_epi (by decide) (by rw [hctx_eq]; exact ctx_leVal_of_rd64 h (by decide) h280)
    have h288_epi : rd64 s_epi.mem (ea ctxBase 288) = some 0 :=
      vc'.rd64_ctx_of_eq hwf_epi (by decide) (by rw [hctx_eq]; exact ctx_leVal_of_rd64 h (by decide) h288)
    have hepi := L_prim_epilogue_oracle_spec (vc := vc') (pr := pr) (ls := { ls with mem := (fun x =>
        if ls.ds[2]!.w ≤ x ∧ x < ls.ds[2]!.w + ls.ds[1]!.w then ⟨ls.ds[0]!.w % 256, q⟩ else ls.mem x) })
      (s := s_epi) (e_in := 3) (outs := []) (f := f) (rest := rest)
      hrip_epi hwf_epi (oracleAdmits_of_reaches hor hrch0_epi) hrs hpcW hle (by decide)
      (by intro _ hv; cases hv) h280_epi h288_epi (by intro _ hi; cases hi)
    rcases hepi with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
    · exact Or.inl ⟨sf, Reaches.trans hrch0_epi hrchf, by rw [hkf, hk_epi]; exact congrArg X86Env.Kern.out hk5, horf, hrelf⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch0_epi hhit, Ends.of_reaches hrch0_epi hends⟩
  · rw [if_neg hdst_ok] at hrc ⊢
    obtain ⟨se, hrche, hexit, hke⟩ := hrc
    exact ⟨se, Reaches.trans hrch4 hrche, hexit, hke⟩

theorem P_mcmp_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_mcmp) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 3) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w) :
    PrimPostOutcome pr ls f rest s .mcmp := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have htake : ls.ds.take 3 = [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] := by
    cases hd0 : ls.ds with
    | nil => simp [hd0] at hle
    | cons a0 r0 =>
      cases hd1 : r0 with
      | nil => simp [hd0, hd1] at hle
      | cons a1 r1 =>
        cases hd2 : r1 with
        | nil => simp [hd0, hd1, hd2] at hle
        | cons a2 r2 => simp [List.take]
  have hwlt_ds := (vc.code_flat h).2.2.2.2.1
  have hnW : ls.ds[0]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hyW : ls.ds[1]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have hxW : ls.ds[2]!.w < X86Env.W := by rw [← M_eq_W']; exact (hwlt_ds _ (by rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]; exact List.getElem_mem _)).1
  have h01 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4197678 } := by
    rw [step_at_vc 10 hrip (by decide : (SporosVmBin.P_mcmp, (.ofInt (.ld 1 15 296)), 7) ∈ vmTableChunk 10),
      exec_ld h15 h296, hrip]; rfl
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4197678 }
  have hr1_ok : RegsOk s1 := regsOk_set h.2.2.2.1 (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by change (s.set 1 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact h15
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some ls.ds[1]!.w) with rip := 4197685 } := by
    rw [step_at_vc 10 rfl (by decide : (4197678, (.ofInt (.ld 2 15 312)), 7) ∈ vmTableChunk 10), exec_ld hs1_15 h312]
  let s2 := { s1.set 2 (some ls.ds[1]!.w) with rip := 4197685 }
  have hr2_ok : RegsOk s2 := regsOk_set hr1_ok (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by change (s1.set 2 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs1_15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some ls.ds[2]!.w) with rip := 4197692 } := by
    rw [step_at_vc 10 rfl (by decide : (4197685, (.ofInt (.ld 0 15 328)), 7) ∈ vmTableChunk 10), exec_ld hs2_15 h328]
  let s3 := { s2.set 0 (some ls.ds[2]!.w) with rip := 4197692 }
  have hr3_ok : RegsOk s3 := regsOk_set hr2_ok (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by change (s2.set 0 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs2_15
  have hms : rd64 s.mem (ea ctxBase 112) = some ls.msize := rd_ctx_msize h
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some ls.msize) with rip := 4197696 } := by
    rw [step_at_vc 10 rfl (by decide : (4197692, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 10), exec_ld hs3_15 hms]
  let s4 := { s3.set 10 (some ls.msize) with rip := 4197696 }
  have hr4_ok : RegsOk s4 := regsOk_set hr3_ok (by decide) _
  have hrch4 : Reaches vmImage s s4 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
  have hwf4 : vc.wf pr ls s4 :=
    vc.wf_step_misc h rfl hr4_ok
      (by change (s3.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s2.set 0 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s1.set 2 _).reg 4 = _; rw [reg_set_ne (by decide)]
          change (s.set 1 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact h.2.2.2.2.1)
      (by change (s3.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs3_15)
  have hs4_0 : s4.reg 0 = some ls.ds[2]!.w := by change (s3.set 10 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr2_ok (by decide) _
  have hs4_1 : s4.reg 1 = some ls.ds[0]!.w := by
    change (s3.set 10 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 1 = _; rw [reg_set_ne (by decide)]
    change (s1.set 2 _).reg 1 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' h.2.2.2.1 (by decide) _
  have hs4_2 : s4.reg 2 = some ls.ds[1]!.w := by
    change (s3.set 10 _).reg 2 = _; rw [reg_set_ne (by decide)]
    change (s2.set 0 _).reg 2 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' hr1_ok (by decide) _
  have hs4_10 : s4.reg 10 = some ls.msize := reg_set_self_ok' hr3_ok (by decide) _
  have hrc1 := range_check_spec (rip0 := 4197696) (c0 := 10) (c1 := 10) (c2 := 10) (c3 := 10) (c4 := 10) (c5 := 10)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hwf4 hs4_0 hs4_1 hs4_10 hxW hnW (by decide) (by decide)
  unfold PrimPostOutcome
  simp only [Sporos.Prim.eff, htake, SporosProvLen.lexecPrim]
  by_cases hx_ok : ls.ds[2]!.w + ls.ds[0]!.w ≤ ls.msize
  · rw [if_pos hx_ok] at hrc1
    obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hr5_ok, hs5_4, hs5_15, _, hs5_oth⟩ := hrc1
    have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc hwf4 hmem5 hr5_ok hs5_4 hs5_15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 10 (some ls.msize) with rip := 4197724 } := by
      rw [step_at_vc 10 hrip5 (by decide : (4197720, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk 10),
        exec_ld hs5_15 (by rw [hmem5]; exact hms), hrip5]
    let s6 := { s5.set 10 (some ls.msize) with rip := 4197724 }
    have hr6_ok : RegsOk s6 := regsOk_set hr5_ok (by decide) _
    have hwf6 : vc.wf pr ls s6 :=
      vc.wf_step_misc hwf5 rfl hr6_ok
        (by change (s5.set 10 _).reg 4 = _; rw [reg_set_ne (by decide)]; exact hs5_4)
        (by change (s5.set 10 _).reg 15 = _; rw [reg_set_ne (by decide)]; exact hs5_15)
    have hs6_0 : s6.reg 0 = some ls.ds[2]!.w := by change (s5.set 10 _).reg 0 = _; rw [reg_set_ne (by decide), hs5_oth 0 (by decide)]; exact hs4_0
    have hs6_1 : s6.reg 1 = some ls.ds[0]!.w := by change (s5.set 10 _).reg 1 = _; rw [reg_set_ne (by decide), hs5_oth 1 (by decide)]; exact hs4_1
    have hs6_2 : s6.reg 2 = some ls.ds[1]!.w := by change (s5.set 10 _).reg 2 = _; rw [reg_set_ne (by decide), hs5_oth 2 (by decide)]; exact hs4_2
    have hs6_10 : s6.reg 10 = some ls.msize := reg_set_self_ok' hr5_ok (by decide) _
    have hrc2 := range_check_spec (rip0 := 4197724) (c0 := 10) (c1 := 10) (c2 := 10) (c3 := 10) (c4 := 10) (c5 := 10)
      (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      rfl hwf6 hs6_2 hs6_1 hs6_10 hyW hnW (by decide) (by decide)
    by_cases hy_ok : ls.ds[1]!.w + ls.ds[0]!.w ≤ ls.msize
    · rw [if_pos ⟨hx_ok, hy_ok⟩]
      rw [if_pos hy_ok] at hrc2
      obtain ⟨s7, hrch7, hrip7, hmem7, hk7, hr7_ok, hs7_4, hs7_15, hs7_11, hs7_oth⟩ := hrc2
      have hrch07 : Reaches vmImage s s7 := Reaches.trans hrch4 (Reaches.trans hrch5 (Reaches.step_trans h06 hrch7))
      have hmem07 : s7.mem = s.mem := by rw [hmem7]; exact hmem5
      have hk07 : s7.k = s.k := by rw [hk7]; exact hk5
      have hwf7 : vc.wf pr ls s7 := vc.wf_step_misc hwf6 hmem7 hr7_ok hs7_4 hs7_15
      have hqW : SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] ls.pc < X86Env.W := by
        simp [SporosProvLen.lq, M_eq_W']; exact Nat.mod_lt _ (by change 0 < 2 ^ 64; omega)
      have hfin := mem_cmp_finish_spec (vc := vc) (pr := pr) (ls := ls) (s := s7) (f := f) (rest := rest)
        (x := ls.ds[2]!.w) (y := ls.ds[1]!.w) (n := ls.ds[0]!.w)
        (q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!, ls.ds[2]!] ls.pc) (e_in := 3)
        (rip0 := 4197748) (c_ld8 := 10) (c_pre := 10) (c_body := 11)
        hrip7 hwf7 (oracleAdmits_of_reaches hor hrch07) hrs hpcW hle hx_ok hy_ok hqW
        (by rw [hs7_oth 0 (by decide)]; exact hs6_0)
        (by rw [hs7_oth 1 (by decide)]; exact hs6_1)
        (by rw [hs7_oth 2 (by decide)]; exact hs6_2)
        hs7_11
        (by rw [hmem07]; rw [htake] at h272; exact h272)
        (by rw [hmem07]; exact h280) (by rw [hmem07]; exact h288)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      rcases hfin with ⟨sf, hrchf, hkf, horf, hrelf⟩ | ⟨hhit, hends⟩
      · exact Or.inl ⟨sf, Reaches.trans hrch07 hrchf, by rw [hkf]; exact congrArg X86Env.Kern.out hk07, horf, hrelf⟩
      · exact Or.inr ⟨hitsNomem_of_reaches hrch07 hhit, Ends.of_reaches hrch07 hends⟩
    · rw [if_neg (fun hc => hy_ok hc.2)]
      rw [if_neg hy_ok] at hrc2
      obtain ⟨se, hrche, hexit, hke⟩ := hrc2
      exact ⟨se, Reaches.trans hrch4 (Reaches.trans hrch5 (Reaches.step_trans h06 hrche)), hexit, by rw [hke]; exact hk5⟩
  · rw [if_neg (fun hc => hx_ok hc.1)]
    rw [if_neg hx_ok] at hrc1
    obtain ⟨se, hrche, hexit, hke⟩ := hrc1
    exact ⟨se, Reaches.trans hrch4 hrche, hexit, hke⟩

end SporosVmPrimBlock
