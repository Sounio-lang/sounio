import Bootstrap.SporosVmPrimIo

/-!
# Native x86-64 VM Data/Stack/Local Instruction Infrastructure (`SporosVmOpsData1`)

Defines `OpPostOutcome` (the uniform post-dispatch simulation outcome predicate for all
non-primitive VM instructions) together with shared combinators for data-stack pushes
and decrements, triplicated indexed code loads (`O_str`), provenance tagging, and
`adv_pc + jmp L_vm_loop` tails.

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

/-- Uniform post-dispatch simulation outcome for non-primitive VM instructions. -/
def OpPostOutcome (pr : Sporos.Prog) (ls : SporosProvLen.LSt) (s : X86Vm.St) : Prop :=
  match SporosProvLen.lstep pr ls with
  | .next ls' =>
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2))
  | .halt c out =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit (c % 256) ∧
      s_exit.k.out = s.k.out ++ out.reverse
  | .fail _ =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k

/-! ### General table, fetch, `FlatProgFit`, and context helpers -/

theorem mem_vmTable_of_vc (k : Nat) {e : Nat × X86Vm.Ins × Nat}
    (he : e ∈ vmTableChunk k) : e ∈ SporosVmBin.vmTable := by
  unfold vmTableChunk at he
  cases hc : SporosVmBin.vmTableChunks[k]? with
  | none => rw [hc] at he; cases he
  | some p => rw [hc] at he; exact mem_vmTable_of_chunk k hc he

theorem fetch_eq_of_lt {pr : Sporos.Prog} {w pc : Nat} {ins : Sporos.Instr}
    (hnw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length)
    (hins : pr.words[w]!.code[pc]! = ins) :
    Sporos.fetch pr w pc = some ins := by
  unfold Sporos.fetch
  have hw := List.getElem?_eq_getElem hnw
  have hwd : pr.words[w]! = pr.words[w]'hnw := by
    rw [List.getElem!_eq_getElem?_getD, hw]; rfl
  rw [hwd] at hpc hins
  have hc := List.getElem?_eq_getElem hpc
  have hic : (pr.words[w]'hnw).code[pc]! = (pr.words[w]'hnw).code[pc]'hpc := by
    rw [List.getElem!_eq_getElem?_getD, hc]; rfl
  rw [hic] at hins
  simp [hw, hc, hins]

theorem flatProgFit_ins_at {pr : Sporos.Prog} (hfit : FlatProgFit pr) {w pc : Nat}
    (hnw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length) :
    match pr.words[w]!.code[pc]! with
    | .lit n => n < X86Env.W
    | .str i => i < X86Env.W
    | .pick n => n < X86Env.W
    | .locget i => i < X86Env.W
    | .locset i => i < X86Env.W
    | .jmp t => t < X86Env.W
    | .jz t => t < X86Env.W
    | .jnz t => t < X86Env.W
    | .call w => w < X86Env.W
    | _ => True := by
  have hwd_mem : pr.words[w]! ∈ pr.words := getElem!_mem_of_lt hnw
  have hins_mem : pr.words[w]!.code[pc]! ∈ pr.words[w]!.code := getElem!_mem_of_lt hpc
  exact (hfit.2.2.2.2.2 _ hwd_mem).2.2.2.2 _ hins_mem

theorem data_addr_len_le_flatten : ∀ (ds : List (List Nat)) (i : Nat), i < ds.length →
    ((ds.take i).map List.length).foldl (· + ·) 0 + ds[i]!.length ≤ ds.flatten.length
  | [], _, hi => by simp at hi
  | d :: ds, 0, _ => by
    simp only [List.take_zero, List.map_nil, List.foldl_nil, Nat.zero_add,
      List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero, Option.getD_some,
      List.flatten_cons, List.length_append]
    omega
  | d :: ds, k + 1, hi => by
    have hk : k < ds.length := by simp at hi; omega
    have ih := data_addr_len_le_flatten ds k hk
    simp only [List.take_succ_cons, List.map_cons, List.foldl_cons, Nat.zero_add,
      List.getElem!_eq_getElem?_getD, List.getElem?_cons_succ,
      List.flatten_cons, List.length_append] at ih ⊢
    rw [list_foldl_add]
    omega

theorem wf_aux_loc_slots {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    leVal ((vc.ctxReg.data.drop 0).take 8) = vc.dsReg.base ∧
    leVal ((vc.ctxReg.data.drop 8).take 8) = ls.ds.length ∧
    leVal ((vc.ctxReg.data.drop 16).take 8) = vc.dsCap ∧
    leVal ((vc.ctxReg.data.drop 24).take 8) = vc.auxReg.base ∧
    leVal ((vc.ctxReg.data.drop 32).take 8) = ls.aux.length ∧
    leVal ((vc.ctxReg.data.drop 40).take 8) = vc.auxCap ∧
    leVal ((vc.ctxReg.data.drop 72).take 8) = vc.locReg.base := by
  obtain ⟨_, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    s0, s8, s16, s24, s32, s40, _, _, _, s72, _⟩ := h
  exact ⟨s0, s8, s16, s24, s32, s40, s72⟩

theorem rd_ctx_auxBase {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 24) = some vc.auxReg.base :=
  vc.rd64_ctx_of_eq h (by decide) (wf_aux_loc_slots h).2.2.2.1

theorem rd_ctx_auxLen {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 32) = some ls.aux.length :=
  vc.rd64_ctx_of_eq h (by decide) (wf_aux_loc_slots h).2.2.2.2.1

theorem rd_ctx_locBase {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 72) = some vc.locReg.base :=
  vc.rd64_ctx_of_eq h (by decide) (wf_aux_loc_slots h).2.2.2.2.2.2

theorem rd64_pairList_top {m : Mem} {r : Region} {len k : Nat} {xs : List SporosProvLen.LVal}
    (hmwf : m.wf) (hrm : r ∈ m) (hhi : r.hi ≤ 2 ^ 47)
    (hrel : PairListRel r.data len xs) (hk : k < xs.length) :
    rdMem .w64 m (eaIdx r.base ((len - 1 - k) * 16) 1 0) = some xs[k]!.w ∧
    rdMem .w64 m (eaIdx r.base ((len - 1 - k) * 16) 1 8) = some xs[k]!.plen := by
  have hW : X86Env.W = 2 ^ 64 := rfl
  have htop := pairListRel_get_top hrel hk
  have hlen_eq : xs.length = len := hrel.1
  have hbound : (len - 1 - k) * 16 + 16 ≤ r.data.length := by have := hrel.2.1; omega
  have hhi' : r.base + r.data.length ≤ 2 ^ 47 := hhi
  constructor
  · rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_zero]
    exact rdMem_w64_at hmwf hrm (Nat.le_add_right _ _)
      (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact htop.1)
  · rw [eaIdx_add_eq (by omega), Nat.one_mul, Nat.add_assoc]
    exact rdMem_w64_at hmwf hrm (by omega)
      (by simp only [Region.hi]; omega) (by rw [Nat.add_sub_cancel_left]; exact htop.2)

theorem adv_pc_finish_vc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hm1 : (a, (.ofInt (.ld 0 15 48)), 4) ∈ vmTableChunk chk)
    (hm2 : (a + 4, (.ofInt (.ld 1 15 56)), 4) ∈ vmTableChunk chk)
    (hm3 : (a + 8, (.ofInt (.aluI .sub 1 1)), 4) ∈ vmTableChunk chk)
    (hm4 : (a + 12, (.ofInt (.shlI 1 5)), 4) ∈ vmTableChunk chk)
    (hm5 : (a + 16, (.ofInt (.alu .add 0 1)), 3) ∈ vmTableChunk chk)
    (hm6 : (a + 19, (.ofInt (.ld 1 0 8)), 4) ∈ vmTableChunk chk)
    (hm7 : (a + 23, (.ofInt (.aluI .add 1 1)), 4) ∈ vmTableChunk chk)
    (hm8 : (a + 27, (.ofInt (.st 0 8 1)), 4) ∈ vmTableChunk chk)
    (hm9 : (a + 31, (.ofInt (.jmp 4194480)), 5) ∈ vmTableChunk chk) :
    let ls' := { ls with rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  obtain ⟨s', hrch, _, hk, _, hrel⟩ := adv_pc_loop_spec hrip h hrs hpcW
    (mem_vmTable_of_vc chk hm1) (mem_vmTable_of_vc chk hm2) (mem_vmTable_of_vc chk hm3)
    (mem_vmTable_of_vc chk hm4) (mem_vmTable_of_vc chk hm5) (mem_vmTable_of_vc chk hm6)
    (mem_vmTable_of_vc chk hm7) (mem_vmTable_of_vc chk hm8) (mem_vmTable_of_vc chk hm9)
  exact Or.inl ⟨s', hrch, by rw [hk], oracleAdmits_of_reaches hor hrch, hrel⟩

theorem step_tag_prov {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a chk rdst rtmp p : Nat}
    (hrip : s.rip = a) (h : vc.wf pr ls s)
    (hrdst : s.reg rdst = some p) (hlt_d : rdst < 16) (hlt_t : rtmp < 16)
    (hne_dt : rdst ≠ rtmp) (hd4 : rdst ≠ 4) (hd15 : rdst ≠ 15) (ht4 : rtmp ≠ 4) (ht15 : rtmp ≠ 15)
    (hm1 : (a, (.ofInt (.ld rtmp 15 224)), 7) ∈ vmTableChunk chk)
    (hm2 : (a + 7, (.ofInt (.alu .add rdst rtmp)), 3) ∈ vmTableChunk chk)
    (hm3 : (a + 10, (.ofInt (.aluI .add rdst 1)), 4) ∈ vmTableChunk chk) :
    ∃ s',
      Reaches vmImage s s' ∧ s'.rip = a + 14 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      vc.wf pr ls s' ∧
      s'.reg rdst = some ((p + ls.pc + 1) % Sporos.M) ∧
      (∀ r, r ≠ rdst → r ≠ rtmp → s'.reg r = s.reg r) := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hr : RegsOk s := h.2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set rtmp (some ls.pc) with rip := a + 7 } := by
    rw [step_at_vc chk hrip hm1, exec_ld h15 (rd_ctx_pc h), hrip]
  let s1 := { s.set rtmp (some ls.pc) with rip := a + 7 }
  have hr1 : RegsOk s1 := regsOk_set hr hlt_t _
  have hs1_t : s1.reg rtmp = some ls.pc := reg_set_self_ok' hr hlt_t _
  have hs1_d : s1.reg rdst = some p := by
    change (s.set rtmp _).reg rdst = some p; rw [reg_set_ne hne_dt.symm]; exact hrdst
  have hadd0 : X86Int.aluF .add p ls.pc = ((p + ls.pc) % X86Env.W, (X86Int.aluF .add p ls.pc).2) :=
    Prod.ext (aluF_add_val p ls.pc) rfl
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set rdst (some ((p + ls.pc) % X86Env.W)) with
            fl := (X86Int.aluF .add p ls.pc).2, rip := a + 10 } := by
    rw [step_at_vc chk rfl hm2, exec_alu hs1_d hs1_t, hadd0]
  let s2 := { s1.set rdst (some ((p + ls.pc) % X86Env.W)) with
              fl := (X86Int.aluF .add p ls.pc).2, rip := a + 10 }
  have hr2 : RegsOk s2 := regsOk_set hr1 hlt_d _
  have hs2_d : s2.reg rdst = some ((p + ls.pc) % X86Env.W) := reg_set_self_ok' hr1 hlt_d _
  have hadd1 : X86Int.aluF .add ((p + ls.pc) % X86Env.W) (1 % X86Env.W) =
      ((p + ls.pc + 1) % Sporos.M, (X86Int.aluF .add ((p + ls.pc) % X86Env.W) (1 % X86Env.W)).2) :=
    Prod.ext (by rw [aluF_add_val]; exact add_pc_1_mod_eq) rfl
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set rdst (some ((p + ls.pc + 1) % Sporos.M)) with
            fl := (X86Int.aluF .add ((p + ls.pc) % X86Env.W) (1 % X86Env.W)).2, rip := a + 14 } := by
    rw [step_at_vc chk rfl hm3, exec_aluI hs2_d, hadd1]
  let s3 := { s2.set rdst (some ((p + ls.pc + 1) % Sporos.M)) with
              fl := (X86Int.aluF .add ((p + ls.pc) % X86Env.W) (1 % X86Env.W)).2, rip := a + 14 }
  have hr3 : RegsOk s3 := regsOk_set hr2 hlt_d _
  have hoth : ∀ r, r ≠ rdst → r ≠ rtmp → s3.reg r = s.reg r := by
    intro r hnd hnt
    change (s2.set rdst _).reg r = s.reg r; rw [reg_set_ne hnd.symm]
    change (s1.set rdst _).reg r = s.reg r; rw [reg_set_ne hnd.symm]
    change (s.set rtmp _).reg r = s.reg r; rw [reg_set_ne hnt.symm]
  have hwf3 : vc.wf pr ls s3 :=
    vc.wf_step_misc h rfl hr3 (by rw [hoth 4 hd4.symm ht4.symm]; exact h4)
      (by rw [hoth 15 hd15.symm ht15.symm]; exact h15)
  exact ⟨s3, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step h03)),
    rfl, rfl, rfl, hwf3, reg_set_self_ok' hr2 hlt_d _, hoth⟩

/-! ### TMR indexed code load voter with index register `r8` (`O_str`) -/

theorem tmr_ldIdx8_vote_eq_spec {a done disp ldLen chk c1 c2 c3 r8v v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase) (hr8 : s.reg 8 = some r8v)
    (hm1 : (a, (.ofInt (.ld 2 15 232)), 7) ∈ vmTableChunk chk)
    (hm2 : (a + 7, (.ldIdx .w64 0 2 8 1 disp), ldLen) ∈ vmTableChunk chk)
    (hm3 : (a + 7 + ldLen, (.ofInt (.ld 2 15 240)), 7) ∈ vmTableChunk chk)
    (hm4 : (a + 14 + ldLen, (.ldIdx .w64 3 2 8 1 disp), ldLen) ∈ vmTableChunk chk)
    (hm5 : (a + 14 + 2 * ldLen, (.ofInt (.ld 2 15 248)), 7) ∈ vmTableChunk chk)
    (hm6 : (a + 21 + 2 * ldLen, (.ldIdx .w64 1 2 8 1 disp), ldLen) ∈ vmTableChunk chk)
    (hm7 : (a + 21 + 3 * ldLen, (.ofInt (.cmp 0 3)), 3) ∈ vmTableChunk chk)
    (hm8 : (a + 24 + 3 * ldLen, (.ofInt (.jcc 4 done)), 2) ∈ vmTableChunk chk)
    (hc1 : rd64 s.mem (ea ctxBase 232) = some c1)
    (hc2 : rd64 s.mem (ea ctxBase 240) = some c2)
    (hc3 : rd64 s.mem (ea ctxBase 248) = some c3)
    (hv1 : rd64 s.mem (eaIdx c1 r8v 1 disp) = some v)
    (hv2 : rd64 s.mem (eaIdx c2 r8v 1 disp) = some v)
    (hv3 : rd64 s.mem (eaIdx c3 r8v 1 disp) = some v)
    (hv : v < X86Env.W) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = done ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 3 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some c1) with rip := a + 7 } := by
    rw [step_at_vc chk hrip hm1, exec_ld h15 hc1, hrip]
  let s1 := { s.set 2 (some c1) with rip := a + 7 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) c1
  have hs1_2 : s1.reg 2 = some c1 := reg_set_self_ok' hr (by decide) _
  have hs1_8 : s1.reg 8 = some r8v := by
    change (s.set 2 _).reg 8 = some r8v; rw [reg_set_ne (by decide)]; exact hr8
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some v) with rip := a + 7 + ldLen } := by
    rw [step_at_vc chk rfl hm2, exec_ldIdx hs1_2 hs1_8 (show rdMem .w64 s1.mem _ = some v from hv1)]
  let s2 := { s1.set 0 (some v) with rip := a + 7 + ldLen }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) v
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs2_8 : s2.reg 8 = some r8v := by
    change (s1.set 0 _).reg 8 = some r8v; rw [reg_set_ne (by decide)]; exact hs1_8
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some c2) with rip := a + 14 + ldLen } := by
    rw [step_at_vc chk rfl hm3, exec_ld hs2_15 hc2, show a + 7 + ldLen + 7 = a + 14 + ldLen by omega]
  let s3 := { s2.set 2 (some c2) with rip := a + 14 + ldLen }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) c2
  have hs3_2 : s3.reg 2 = some c2 := reg_set_self_ok' hr2 (by decide) _
  have hs3_8 : s3.reg 8 = some r8v := by
    change (s2.set 2 _).reg 8 = some r8v; rw [reg_set_ne (by decide)]; exact hs2_8
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 3 (some v) with rip := a + 14 + 2 * ldLen } := by
    rw [step_at_vc chk rfl hm4, exec_ldIdx hs3_2 hs3_8 (show rdMem .w64 s3.mem _ = some v from hv2),
      show a + 14 + ldLen + ldLen = a + 14 + 2 * ldLen by omega]
  let s4 := { s3.set 3 (some v) with rip := a + 14 + 2 * ldLen }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) v
  have hs4_15 : s4.reg 15 = some ctxBase := by
    change (s3.set 3 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s2.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hs4_8 : s4.reg 8 = some r8v := by
    change (s3.set 3 _).reg 8 = some r8v; rw [reg_set_ne (by decide)]; exact hs3_8
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some c3) with rip := a + 21 + 2 * ldLen } := by
    rw [step_at_vc chk rfl hm5, exec_ld hs4_15 hc3,
      show a + 14 + 2 * ldLen + 7 = a + 21 + 2 * ldLen by omega]
  let s5 := { s4.set 2 (some c3) with rip := a + 21 + 2 * ldLen }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) c3
  have hs5_2 : s5.reg 2 = some c3 := reg_set_self_ok' hr4 (by decide) _
  have hs5_8 : s5.reg 8 = some r8v := by
    change (s4.set 2 _).reg 8 = some r8v; rw [reg_set_ne (by decide)]; exact hs4_8
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some v) with rip := a + 21 + 3 * ldLen } := by
    rw [step_at_vc chk rfl hm6, exec_ldIdx hs5_2 hs5_8 (show rdMem .w64 s5.mem _ = some v from hv3),
      show a + 21 + 2 * ldLen + ldLen = a + 21 + 3 * ldLen by omega]
  let s6 := { s5.set 1 (some v) with rip := a + 21 + 3 * ldLen }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) v
  have hs6_0 : s6.reg 0 = some v := by
    change (s5.set 1 _).reg 0 = some v; rw [reg_set_ne (by decide)]
    change (s4.set 2 _).reg 0 = some v; rw [reg_set_ne (by decide)]
    change (s3.set 3 _).reg 0 = some v; rw [reg_set_ne (by decide)]
    change (s2.set 2 _).reg 0 = some v; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok' hr1 (by decide) _
  have hs6_3 : s6.reg 3 = some v := by
    change (s5.set 1 _).reg 3 = some v; rw [reg_set_ne (by decide)]
    change (s4.set 2 _).reg 3 = some v; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok' hr3 (by decide) _
  have hs6_1 : s6.reg 1 = some v := reg_set_self_ok' hr5 (by decide) _
  have hm8' : (a + 21 + 3 * ldLen + 3, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable := by
    rw [show a + 21 + 3 * ldLen + 3 = a + 24 + 3 * ldLen by omega]
    exact mem_vmTable_of_vc chk hm8
  have hvote := tmr_inline_vote_eq_spec (a := a + 21 + 3 * ldLen) rfl
    (mem_vmTable_of_vc chk hm7) hm8' hs6_0 hs6_3 hv
  refine ⟨{ s6 with fl := (X86Int.aluF .sub v v).2, rip := done },
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 hvote))))),
    rfl, rfl, rfl, hr6, hs6_0, hs6_1, hs6_3, ?_⟩
  intro r h0 h1 h2 h3
  change (s5.set 1 _).reg r = s.reg r; rw [reg_set_ne h1.symm]
  change (s4.set 2 _).reg r = s.reg r; rw [reg_set_ne h2.symm]
  change (s3.set 3 _).reg r = s.reg r; rw [reg_set_ne h3.symm]
  change (s2.set 2 _).reg r = s.reg r; rw [reg_set_ne h2.symm]
  change (s1.set 0 _).reg r = s.reg r; rw [reg_set_ne h0.symm]
  change (s.set 2 _).reg r = s.reg r; rw [reg_set_ne h2.symm]

/-! ### `VmCtx` updates for `fn_ds_push` and `ds_len` decrement -/

theorem wf_update_ds_push {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {dsReg' ctx' stk' : Region} {dsCap' : Nat} {v : SporosProvLen.LVal}
    (h : vc.wf pr ls s)
    (hmwf' : s'.mem.wf) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hds_mem : dsReg' ∈ s'.mem) (hds_rw : dsReg'.rw = true) (hds_hi : dsReg'.hi ≤ 2 ^ 47)
    (hds_cap_ge : 4 ≤ dsCap') (hds_len_eq : dsReg'.data.length = dsCap' * 16)
    (hds_pg : (dsCap' * 16) % X86Env.page = 0)
    (hctx_mem : ctx' ∈ s'.mem) (hctx_base : ctx'.base = vc.ctxReg.base)
    (hctx_rw : ctx'.rw = vc.ctxReg.rw) (hctx_len : ctx'.data.length = vc.ctxReg.data.length)
    (hc0 : leVal ((ctx'.data.drop 0).take 8) = dsReg'.base)
    (hc8 : leVal ((ctx'.data.drop 8).take 8) = (v :: ls.ds).length)
    (hc16 : leVal ((ctx'.data.drop 16).take 8) = dsCap')
    (hctx_oth : ∀ j, (j < 0 ∨ 24 ≤ j) → ctx'.data[j]? = vc.ctxReg.data[j]?)
    (hstk_mem : stk' ∈ s'.mem) (hstk_base : stk'.base = vc.stk.base)
    (hstk_rw : stk'.rw = vc.stk.rw) (hstk_len : stk'.data.length = vc.stk.data.length)
    (hds_ne : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ dsReg'.base)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s'.mem)
    (hds_rel : PairListRel dsReg'.data (v :: ls.ds).length (v :: ls.ds))
    (hds_le : (v :: ls.ds).length ≤ dsCap')
    (hvW : v.w < X86Env.W ∧ v.plen < X86Env.W) :
    let vc' := { vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := dsCap' }
    vc'.wf pr { ls with ds := v :: ls.ds } s' ∧
    (∀ m0 u, 24 ≤ m0 → m0 + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  have hcm' : CodeMem s'.mem := ⟨hoth codeRegion h.1.1
    (vc.ne_bases h (i := 0) (j := 6) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    (vc.ne_bases h (i := 0) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide))
    (vc.ne_bases h (i := 0) (j := 1) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)), hmwf'⟩
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    _, _, _, _, _, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    _, _, _, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    _, _, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let vc' : VmCtx := { vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := dsCap' }
  have hmem_all : ∀ r ∈ vc'.regions, r ∈ s'.mem := by
    intro r hr
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hstk_mem | exact hctx_mem | exact hds_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    let l1 : List Region := [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3]
    let l2 : List Region := [vc.auxReg, vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg,
      vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]
    have hnd_old : (l1.map Region.base ++ vc.dsReg.base :: l2.map Region.base).Nodup := by
      simpa [l1, l2, VmCtx.regions] using hnodup
    have hgoal : (l1.map Region.base ++ dsReg'.base :: l2.map Region.base).Nodup := by
      refine nodup_replace_at hnd_old ?_
      intro x hx
      rw [← List.map_append, List.mem_map] at hx
      rcases hx with ⟨u, hu, rfl⟩
      have hu_in : u ∈ vc.regions := by
        simp only [VmCtx.regions]
        simp only [l1, l2, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hu
        rcases hu with (rfl | rfl | rfl | rfl | rfl | rfl) | (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;> simp
      have hu_ne : u.base ≠ vc.dsReg.base := by
        rw [List.nodup_append, List.nodup_cons] at hnd_old
        rcases List.mem_append.mp hu with h1 | h2
        · exact hnd_old.2.2 u.base (List.mem_map_of_mem h1) vc.dsReg.base (List.mem_cons_self ..)
        · intro heq; exact hnd_old.2.1.1 (heq ▸ List.mem_map_of_mem h2)
      exact hds_ne u (hmem u hu_in) hu_ne
    simpa [vc', l1, l2, VmCtx.regions, hstk_base, hctx_base] using hgoal
  have hslot : ∀ m0, 24 ≤ m0 →
      leVal ((ctx'.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
    intro m0 hm0; congr 1
    exact drop_take_eq_of_getElem?_eq fun j hj1 _ => hctx_oth j (Or.inr (by omega))
  have hwlt_ds' : ∀ x ∈ v :: ls.ds, x.w < Sporos.M ∧ x.plen < Sporos.M := by
    intro x hx; rcases List.mem_cons.mp hx with rfl | hx
    · exact hvW
    · exact hwlt.1 x hx
  have hwf' : vc'.wf pr { ls with ds := v :: ls.ds } s' :=
    ⟨hcm', hbl', hpos', hr', hsp', h15', hnodup', hmem_all,
      by rw [hstk_rw]; exact hstk1, by rw [hstk_base]; exact hstk2,
      by simpa only [vc', Region.hi, hstk_base, hstk_len] using hstk3, hstk4,
      by rw [hctx_base]; exact hctx1, by rw [hctx_rw]; exact hctx2,
      by rw [hctx_len]; exact hctx3, by simpa only [vc', Region.hi, hctx_base, hctx_len] using hctx4,
      hf1, hf2, hf3, hds_rw, hds_cap_ge, hds_len_eq, hds_pg, hds_hi,
      haux1, haux2, haux3, haux4, haux5, hrs1, hrs2, hrs3, hrs4, hrs5,
      hloc1, hloc2, hloc3, hloc4, hloc5, hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
      hinp1, hout1, hout2, hout3, hout4, hout5, hc0, hc8, hc16,
      by rw [hslot _ (by decide)]; exact hsl3, by rw [hslot _ (by decide)]; exact hsl4,
      by rw [hslot _ (by decide)]; exact hsl5, by rw [hslot _ (by decide)]; exact hsl6,
      by rw [hslot _ (by decide)]; exact hsl7, by rw [hslot _ (by decide)]; exact hsl8,
      by rw [hslot _ (by decide)]; exact hsl9, by rw [hslot _ (by decide)]; exact hsl10,
      by rw [hslot _ (by decide)]; exact hsl11, by rw [hslot _ (by decide)]; exact hsl12,
      by rw [hslot _ (by decide)]; exact hsl13, by rw [hslot _ (by decide)]; exact hsl14,
      by rw [hslot _ (by decide)]; exact hsl15, by rw [hslot _ (by decide)]; exact hsl16,
      by rw [hslot _ (by decide)]; exact hsl17, by rw [hslot _ (by decide)]; exact hsl18,
      by rw [hslot _ (by decide)]; exact hsl19, by rw [hslot _ (by decide)]; exact hsl20,
      by rw [hslot _ (by decide)]; exact hsl21, by rw [hslot _ (by decide)]; exact hsl22,
      by rw [hslot _ (by decide)]; exact hsl23, by rw [hslot _ (by decide)]; exact hsl24,
      by rw [hslot _ (by decide)]; exact hsl25, by rw [hslot _ (by decide)]; exact hsl26,
      by rw [hslot _ (by decide)]; exact hsl27, by rw [hslot _ (by decide)]; exact hsl28,
      by rw [hslot _ (by decide)]; exact hsl29, by rw [hslot _ (by decide)]; exact hsl30,
      by rw [hslot _ (by decide)]; exact hsl31,
      hds_rel, hds_le, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
      ⟨hwlt_ds', hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨hwf', fun m0 u hm0 hmlim hprev =>
    vc'.rd64_ctx_of_eq hwf' hmlim (by rw [hslot m0 hm0, hle_ctx m0 u hmlim hprev])⟩

theorem step_call_ds_push_vc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w p : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.call SporosVmBin.fn_ds_push)) 5 s)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hretW : s.rip + 5 < X86Env.W)
    (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hwW : w < X86Env.W) (hpW : p < X86Env.W) :
    (∃ (vc' : VmCtx) (s' : X86Vm.St),
      Reaches vmImage s s' ∧ s'.rip = s.rip + 5 ∧ s'.k.out = s.k.out ∧
      OracleAdmits s' ∧ vc'.wf pr { ls with ds := ⟨w, p⟩ :: ls.ds } s' ∧
      (∀ m0 u, 24 ≤ m0 → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  obtain ⟨s1, h01, hrip1, hk1, _, hcm1, hbl1, hpos1, hr1, hs1_4, hs1_15, hs1_oth,
      hstk1, hstk1_b, hstk1_rw, hstk1_len, h16, hsp1W, hstk1_lo, hstk1_hi, hrd_ret, hoth1⟩ :=
    step_call_stk hstep h hretW
  have hrch1 : Reaches vmImage s s1 := Reaches.step h01
  have hctx_b := vc.ctx_bounds h
  have ⟨⟨hds_rw, hds_cap4, hds_len, hds_pg, hds_hi, hds_rel, hds_le⟩, _, _⟩ := vc.arena_bounds h
  have ⟨hc0, hc8, hc16, _, _, _, _⟩ := wf_aux_loc_slots h
  have hne_ds_ctx : vc.dsReg.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 6) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ds : vc.stk.base ≠ vc.dsReg.base :=
    vc.ne_bases h (i := 1) (j := 6) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_stk_ctx : vc.stk.base ≠ ctxBase := by
    rw [← hctx_b.1]
    exact vc.ne_bases h (i := 1) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hcall := call_ds_push_oracle_spec (s := s1) (sp := vc.sp - 8) (retAddr := s.rip + 5)
    (ctxBase := ctxBase) (dsBase := vc.dsReg.base) (dsLen := ls.ds.length) (dsCap := vc.dsCap)
    (w := w) (p := p) (ctx := vc.ctxReg) (dsReg := vc.dsReg)
    (stk := { vc.stk with data := writeBytes vc.stk.data (vc.sp - 8 - vc.stk.base) (bytes8 (s.rip + 5)) })
    hrip1 hr1 (oracleAdmits_of_reaches hor hrch1) hbl1 hpos1 hs1_4 hs1_15
    (by rw [hs1_oth 7 (by decide)]; exact hr7) (by rw [hs1_oth 6 (by decide)]; exact hr6)
    (hoth1 vc.ctxReg (vc.mem_ctx h) (by rw [hctx_b.1]; exact Ne.symm hne_stk_ctx))
    hctx_b.1 hctx_b.2.1 hctx_b.2.2.1
    (hoth1 vc.dsReg (vc.mem_ds h) (Ne.symm hne_stk_ds)) rfl hds_rw hds_hi hne_ds_ctx
    hstk1 hstk1_rw (by rw [hstk1_b]; exact hne_stk_ds) (by rw [hstk1_b]; exact hne_stk_ctx)
    h16 hsp1W hstk1_lo hstk1_hi hds_le (by omega) hds_len hds_pg hctx_b.2.2.2.2.1
    hcm1.2 hrd_ret hc0 hc8 hc16
  rcases hcall with ⟨dsReg', ctx', stk', m2, s2, dsCap', hrch2, hrip2, hm2, hmwf2, hbl2, hpos2,
      hr2, hor2, hk2, hsp2, hcap_le, hlen_le, hds_pg', hds2_mem, hds2_rw, hds2_hi, hds2_data,
      hctx2_mem, hctx2_b, hctx2_rw, hctx2_len, hc0', hc8', hc16', hctx2_oth,
      hstk2_mem, hstk2_b, hstk2_rw, hstk2_len, hds2_ne, hoth2, hreg2_oth⟩ | ⟨hhit, hends⟩
  · have hsp2' : s2.reg 4 = some vc.sp := by
      have hsb := vc.sp_bounds h
      simpa [show vc.sp - 8 + 8 = vc.sp by omega, Nat.mod_eq_of_lt hsb.2.2.1] using hsp2
    have hs2_15 : s2.reg 15 = some ctxBase := by
      rw [hreg2_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hs1_15
    have hds_len_eq' : dsReg'.data.length = dsCap' * 16 := by
      rw [hds2_data, writeBytes_length, writeBytes_length, List.length_append, List.length_replicate, hds_len]
      omega
    have hds_rel' : PairListRel dsReg'.data ((⟨w, p⟩ : SporosProvLen.LVal) :: ls.ds).length (⟨w, p⟩ :: ls.ds) := by
      rw [hds2_data]
      simpa using pairListRel_push (pairListRel_extend_zero (ext := (dsCap' - vc.dsCap) * 16) hds_rel (by rw [hds_len]; omega))
        ⟨w, p⟩ hwW hpW (by rw [List.length_append, List.length_replicate, hds_len]; omega)
    have hds_ne_s : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ dsReg'.base := by
      intro u hu hne
      by_cases hstku : u.base = vc.stk.base
      · rw [hstku, ← hstk1_b]; exact hds2_ne _ hstk1 (by rw [hstk1_b]; exact hne_stk_ds)
      · exact hds2_ne u (hoth1 u hu hstku) hne
    have hoth_s : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ vc.ctxReg.base → u.base ≠ vc.stk.base → u ∈ s2.mem := by
      intro u hu h1 h2 h3; rw [hm2]
      exact hoth2 u (hoth1 u hu h3) h1 (by rw [← hctx_b.1]; exact h2) (by rw [hstk1_b]; exact h3)
    obtain ⟨hwf2, hpres2⟩ := wf_update_ds_push h (hm2 ▸ hmwf2) (hm2 ▸ hbl2) (hm2 ▸ hpos2)
      hr2 hsp2' hs2_15 (hm2 ▸ hds2_mem) hds2_rw hds2_hi (by omega) hds_len_eq' hds_pg'
      (hm2 ▸ hctx2_mem) hctx2_b hctx2_rw hctx2_len hc0' (by simpa using hc8') hc16' hctx2_oth
      (hm2 ▸ hstk2_mem) (by rw [hstk2_b, hstk1_b]) hstk2_rw (by rw [hstk2_len, hstk1_len])
      hds_ne_s hoth_s hds_rel' (by simpa using hlen_le) ⟨hwW, hpW⟩
    exact Or.inl ⟨_, s2, Reaches.trans hrch1 hrch2, hrip2, hk2.trans (by rw [hk1]), hor2, hwf2, hpres2⟩
  · exact Or.inr ⟨hitsNomem_of_reaches hrch1 hhit, Ends.of_reaches hrch1 hends⟩

theorem wf_dec_dsLen {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {len r : Nat} {v : SporosProvLen.LVal} {restDs : List SporosProvLen.LVal}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 8 r)) len s)
    (h : vc.wf pr ls s) (hds : ls.ds = v :: restDs) (hr : s.reg r = some restDs.length) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 restDs.length) }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧ s'.rip = s.rip + len ∧ s'.k = s.k ∧
      s'.regs = s.regs ∧ s'.fl = s.fl ∧
      { vc with ctxReg := ctx' }.wf pr { ls with ds := restDs } s' := by
  have hb := vc.ctx_bounds h
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hea : ea ctxBase 8 = ctxBase + 8 := ea_add_eq (by have : ctxSize = 4096 := rfl; have : X86Env.W = 2 ^ 64 := rfl; omega)
  obtain ⟨_, m', hwr, _, _, _, hmem', hoth⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_ctx h) hb.2.1 (a := ctxBase + 8)
      (by rw [hb.1]; omega) (by simp only [Region.hi, hb.1]; have : ctxSize = 4096 := rfl; omega)
      (v := restDs.length)
  have hsub : ctxBase + 8 - vc.ctxReg.base = 8 := by rw [hb.1]; omega
  rw [hsub] at hmem'
  rw [← hea] at hwr
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by rw [hstep, exec_st h15 hr hwr]
  have hdrop_len : ls.ds.length - 1 = restDs.length := by simp [hds]
  have hdrop_eq : ls.ds.drop 1 = restDs := by simp [hds]
  have hwf' := wf_write_ctx8_drop (s' := s') (e_in := 1) h (by simp [hds])
    (by rw [hdrop_len]; exact hwr) (by rw [hdrop_len]; exact hmem')
    (fun u hu hne => hoth u hu (by rw [hb.1]; exact hne)) rfl h.2.2.2.1 h.2.2.2.2.1 h15
  rw [hdrop_len, hdrop_eq] at hwf'
  exact ⟨s', hs, rfl, rfl, rfl, rfl, hwf'⟩

theorem rsLocRel_set_top_loc {rsData locData : List Nat} {locLen i : Nat} {v : SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hr : RsLocRel rsData locData locLen (f :: rest))
    (hi : i < f.locs.length) (hv_w : v.w < X86Env.W) (hv_p : v.plen < X86Env.W) :
    let lb := rsLocBase rest
    let locData' := writeBytes (writeBytes locData ((lb + i) * 16) (bytes8 v.w))
      ((lb + i) * 16 + 8) (bytes8 v.plen)
    RsLocRel rsData locData' locLen ({ f with locs := f.locs.set i v } :: rest) := by
  simp only [RsLocRel] at hr ⊢
  rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc, hlb, hn, hlocs, hrest⟩
  refine ⟨by simp [hl], hrsLen,
    by simp only [List.length_set]; rw [writeBytes_length, writeBytes_length]; exact hlocLen,
    hw, hpc, hlb, by simp [hn], ?_,
    rsLocRel_write_loc_after (rsLocRel_write_loc_after hrest (by omega)) (by omega)⟩
  intro j hj
  simp only [List.length_set] at hj
  by_cases heq : j = i
  · subst heq
    have hdisj := writeBytes_drop_take_disjoint
      (writeBytes locData ((rsLocBase rest + j) * 16) (bytes8 v.w))
      ((rsLocBase rest + j) * 16 + 8) ((rsLocBase rest + j) * 16) 8 (bytes8 v.plen)
      (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega)
    have hw0 := leVal_writeBytes_same locData ((rsLocBase rest + j) * 16) v.w (by omega)
    have hw1 := leVal_writeBytes_same (writeBytes locData ((rsLocBase rest + j) * 16) (bytes8 v.w))
      ((rsLocBase rest + j) * 16 + 8) v.plen (by rw [writeBytes_length]; omega)
    have hget : (f.locs.set j v)[j]! = v := by
      rw [List.getElem!_eq_getElem?_getD, mem_set_self_get _ j v hi]; rfl
    rw [hdisj, hw0, hw1, hget, Nat.mod_eq_of_lt hv_w, Nat.mod_eq_of_lt hv_p]
    exact ⟨rfl, rfl⟩
  · have hget : (f.locs.set i v)[j]! = f.locs[j]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD, mem_set_ne_get _ v (Ne.symm heq)]
    rw [hget]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
      writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega),
      writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega),
      writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
    exact hlocs j hj

end SporosVmOpsData
