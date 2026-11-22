import Bootstrap.SporosVmPrimDispatch
import Bootstrap.SporosVmOpsData
import Bootstrap.SporosVmOpsCtrl

/-!
# Native x86-64 VM Full Step & Multi-Step Simulation (`SporosVmSim`)

Connects the native x86-64 VM image (`vmImage`, verified against committed
`bootstrap/model/sporos_vm` bytes by `SporosVmBin.table_ok` and
`SporosVmCheckBytes`) across all 14 `Sporos.Instr` constructors (all 57
primitives + 8 data/stack/local ops + 5 control-flow ops) to:

- `sporos_vm_step_sim`: single-step simulation against `SporosProvLen.lstep`,
- `sporos_vm_lrun_sim`: multi-step simulation against `SporosProvLen.lrun`,
- `sporos_vm_sim_noPhys`: end-to-end simulation down to `Sporos.run` under
  `Sporos.noPhys`,
- `sporos_vm_sim_noPhys_noNomem`: unconditional exit/output equivalence whenever
  the memory-allocator oracle contains no `.nomem` entry,
- `sporos_vm_sim_noPhys_checked`: composition with `Sporos.run_safe` /
  `SporosProvLen.lrun_safe` ruling out static checker failures on checked
  programs.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmSim

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase
  SporosVmPrimBase SporosVmPrimDispatch SporosVmOpsData SporosVmOpsCtrl

local instance : Inhabited Sporos.Word := ⟨⟨0, 0, 0, [], []⟩⟩
local instance : Inhabited Sporos.Instr := ⟨.ret⟩
local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### Oracle monotonicity (`no .nomem` eliminates the exit-2 disjunct) -/

private theorem kern_next_oracle_sub (k : X86Env.Kern) :
    ∀ c ∈ (k.next.2).oracle, c ∈ k.oracle := by
  intro c hc
  cases ho : k.oracle with
  | nil => simp [X86Env.Kern.next, ho] at hc ⊢
  | cons x rest => simp [X86Env.Kern.next, ho] at hc ⊢; exact Or.inr hc

private theorem kern_count_oracle_sub (k : X86Env.Kern) (want avail : Nat) :
    ∀ c ∈ (k.count want avail).2.oracle, c ∈ k.oracle := by
  intro c hc
  cases hnx : k.next with
  | mk oc k1 =>
    have hsub : ∀ x ∈ k1.oracle, x ∈ k.oracle := by
      have h0 := kern_next_oracle_sub k
      rw [hnx] at h0
      exact h0
    unfold X86Env.Kern.count at hc
    rw [hnx] at hc
    cases oc with
    | none => exact hc
    | some ch =>
      cases ch with
      | count n =>
        dsimp only at hc
        split at hc <;> exact hsub c hc
      | _ => exact hc

private theorem lift_ok_eq' {o : Option X86Int.St} {s' : X86Vm.St}
    (h : X86Int.lift o = .ok s') : o = some s' := by
  cases o with
  | none => simp [X86Int.lift] at h
  | some s1 => simp [X86Int.lift] at h; subst h; rfl

private theorem syscall_oracle_sub {k k' : X86Env.Kern} {m m' : Mem}
    {nr a1 a2 a3 a4 a5 a6 rax : Nat}
    (h : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) :
    ∀ c ∈ k'.oracle, c ∈ k.oracle := by
  intro c hc
  unfold X86Env.syscall at h
  split at h
  · -- nr = 0 (read)
    split at h
    · simp only [bind, Option.bind, pure] at h
      cases hwr : m.wrN a2 _ with
      | none => rw [hwr] at h; cases h
      | some m1 => rw [hwr] at h; cases h; exact kern_count_oracle_sub k a3 k.stdin.length c hc
    · split at h
      · split at h
        · simp only [bind, Option.bind, pure] at h
          cases hwr : m.wrN a2 _ with
          | none => rw [hwr] at h; cases h
          | some m1 => rw [hwr] at h; cases h; exact kern_count_oracle_sub k _ _ c hc
        · cases h; exact hc
      · cases h; exact hc
  · -- nr = 1 (write)
    split at h
    · simp only [bind, Option.bind, pure] at h
      cases hrd : m.rdN a2 _ with
      | none => rw [hrd] at h; cases h
      | some bs => rw [hrd] at h; cases h; exact kern_count_oracle_sub k a3 a3 c hc
    · cases h; exact hc
  · -- nr = 2 (open)
    split at h
    · cases h
    · simp only [bind, Option.bind, pure] at h
      cases hcs : m.cstr a1 4096 with
      | none => rw [hcs] at h; cases h
      | some name =>
        rw [hcs] at h; dsimp only at h
        cases hlu : X86Env.lookup k.files name with
        | none => rw [hlu] at h; cases h; exact hc
        | some data => rw [hlu] at h; cases h; exact hc
  · -- nr = 3 (close)
    split at h <;> (cases h; exact hc)
  · -- nr = 9 (mmap)
    split at h
    · cases h
    · split at h
      · cases h; exact hc
      · simp only [bind, Option.bind, pure] at h
        cases hpl : X86Env.place m k.next.1 (X86Env.pages a2) with
        | none => rw [hpl] at h; cases h
        | some oa =>
          rw [hpl] at h; dsimp only at h
          cases oa with
          | none => cases h; exact kern_next_oracle_sub k c hc
          | some a => cases h; exact kern_next_oracle_sub k c hc
  · -- nr = 11 (munmap)
    split at h
    · split at h
      · cases h; exact hc
      · cases h
    · cases h
  · -- nr = 25 (mremap)
    split at h
    · cases h
    · split at h
      · split at h
        · cases h
        · dsimp only at h
          split at h
          · cases h
          · simp only [bind, Option.bind, pure] at h
            cases hpl : X86Env.place _ k.next.1 (X86Env.pages a3) with
            | none => rw [hpl] at h; cases h
            | some oa =>
              rw [hpl] at h; dsimp only at h
              cases oa with
              | none => cases h; exact kern_next_oracle_sub k c hc
              | some a => cases h; exact kern_next_oracle_sub k c hc
      · cases h
  · -- nr = 60 (exit)
    cases h
  · -- other
    cases h

private theorem int_exec_oracle_sub {ii : X86Int.Ins} {len : Nat} {s s' : X86Vm.St}
    (h : X86Vm.exec (.ofInt ii) len s = .ok s') :
    ∀ c ∈ s'.k.oracle, c ∈ s.k.oracle := by
  intro c hc
  cases ii with
  | syscall =>
    simp only [X86Vm.exec, X86Int.exec] at h
    cases h0 : s.reg 0 <;> simp [h0] at h
    cases h7 : s.reg 7 <;> simp [h7] at h
    cases h6 : s.reg 6 <;> simp [h6] at h
    cases h2 : s.reg 2 <;> simp [h2] at h
    cases h10 : s.reg 10 <;> simp [h10] at h
    cases h8 : s.reg 8 <;> simp [h8] at h
    cases h9 : s.reg 9 <;> simp [h9] at h
    cases hsy : X86Env.syscall s.k s.mem _ _ _ _ _ _ _ with
    | none => rw [hsy] at h; cases h
    | some sr =>
      cases sr with
      | exit _ => rw [hsy] at h; cases h
      | ret rax k' m' =>
        rw [hsy] at h; cases h
        exact syscall_oracle_sub hsy c hc
  | movI _ _ | jmp _ =>
    cases h; exact hc
  | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _ | test _ _
  | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | st _ _ _ | st8 _ _ _
  | lea _ _ _ | jcc _ _ | jtab _ _ _ | jmpR _ | call _ | ret | push _ | pushI _ | pop _ =>
    have ho := lift_ok_eq' h
    simp only [bind, Option.bind, pure] at ho
    repeat' (split at ho <;> simp at ho)
    cases ho; exact hc

theorem step_oracle_sub {im : Nat → Option (X86Vm.Ins × Nat)} {s s' : X86Vm.St}
    (h : X86Vm.step im s = .ok s') :
    ∀ c ∈ s'.k.oracle, c ∈ s.k.oracle := by
  simp only [X86Vm.step] at h
  cases him : im s.rip with
  | none => rw [him] at h; cases h
  | some p =>
    rcases p with ⟨ins, len⟩
    rw [him] at h
    cases ins with
    | ofInt ii => exact int_exec_oracle_sub h
    | div _ | idiv _ =>
      have ho := lift_ok_eq' h
      simp only [bind, Option.bind, pure] at ho
      repeat' (split at ho <;> simp at ho)
      rcases ho with ⟨_, _, ho⟩
      rw [← ho]; intro c hc; exact hc
    | mul _ | imul _ _ | cqo | neg _ | not _ | shlR _ | shrR _ | sarR _
    | sarI _ _ | ld16 _ _ _ | st16 _ _ _ | st32 _ _ _ | ldIdx _ _ _ _ _ _
    | stIdx _ _ _ _ _ _ | leaIdx _ _ _ _ _ | testI _ _ | cmpAlI _ | testAlI _
    | setcc _ _ | cmovcc _ _ _ =>
      have ho := lift_ok_eq' h
      simp only [bind, Option.bind, pure] at ho
      repeat' (split at ho <;> simp at ho)
      rw [← ho]; intro c hc; exact hc

theorem run_oracle_sub {im : Nat → Option (X86Vm.Ins × Nat)} {k : Nat} {s s' : X86Vm.St}
    (h : X86Vm.run im k (.ok s) = .ok s') :
    ∀ c ∈ s'.k.oracle, c ∈ s.k.oracle := by
  induction k generalizing s with
  | zero => simp [X86Vm.run] at h; subst h; exact fun _ hc => hc
  | succ k ih =>
    simp only [X86Vm.run] at h
    cases hst : X86Vm.step im s with
    | ok s1 =>
      rw [hst] at h
      exact fun c hc => step_oracle_sub hst c (ih h c hc)
    | exit _ => rw [hst] at h; cases k <;> simp [X86Vm.run] at h
    | stuck => rw [hst] at h; cases k <;> simp [X86Vm.run] at h

/-- If the initial oracle contains no `.nomem` entry, `HitsNomem` is impossible. -/
theorem not_hitsNomem_of_no_nomem {s₀ : X86Vm.St}
    (hno : ∀ c ∈ s₀.k.oracle, c ≠ .nomem) : ¬ HitsNomem s₀ := by
  intro ⟨k, s, hk, _, hhd⟩
  cases ho : s.k.oracle with
  | nil => rw [ho] at hhd; cases hhd
  | cons c rest =>
    rw [ho] at hhd; cases hhd
    have hmem : X86Env.Choice.nomem ∈ s.k.oracle := by rw [ho]; exact List.mem_cons_self
    exact hno .nomem (run_oracle_sub hk .nomem hmem) rfl

/-! ### Single-step simulation across all 14 `Sporos.Instr` constructors -/

private theorem lift_opPostOutcome {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s₀ s₁ : X86Vm.St}
    (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k = s₀.k)
    (hout : OpPostOutcome pr ls s₁) : OpPostOutcome pr ls s₀ := by
  unfold OpPostOutcome at hout ⊢
  cases hl : SporosProvLen.lstep pr ls with
  | next ls' => rw [hl] at hout; exact lift_next_outcome hrch hk hout
  | halt c out =>
    rw [hl] at hout
    obtain ⟨sx, hrx, hsx, hkx⟩ := hout
    exact ⟨sx, Reaches.trans hrch hrx, hsx, by rw [hkx, hk]⟩
  | fail e => rw [hl] at hout; exact lift_fail_outcome hrch hk hout

private theorem fetch_none_of_not_word {pr : Sporos.Prog} {w pc : Nat}
    (hnw : ¬ w < pr.words.length) : Sporos.fetch pr w pc = none := by
  unfold Sporos.fetch
  rw [List.getElem?_eq_none (Nat.le_of_not_lt hnw)]
  rfl

private theorem fetch_none_of_not_pc {pr : Sporos.Prog} {w pc : Nat}
    (hnw : w < pr.words.length) (hpc : ¬ pc < pr.words[w]!.code.length) :
    Sporos.fetch pr w pc = none := by
  unfold Sporos.fetch
  have hw := List.getElem?_eq_getElem hnw
  have hwd : pr.words[w]! = pr.words[w]'hnw := by
    rw [List.getElem!_eq_getElem?_getD, hw]; rfl
  rw [hwd] at hpc
  simp [hw, List.getElem?_eq_none (Nat.le_of_not_lt hpc)]

/-- Full single-step simulation: one iteration of `L_vm_loop` on `sporos_vm` simulates
`SporosProvLen.lstep` across every instruction (all 57 primitives + 13 core ops). -/
theorem sporos_vm_step_sim {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s₀ : X86Vm.St}
    (hrel : VmRel pr ls s₀) (hor : OracleAdmits s₀) :
    OpPostOutcome pr ls s₀ := by
  have ⟨_, vc, hvc⟩ := hrel
  have hdisp := L_vm_loop_dispatch_spec hrel hvc
  cases hrs : ls.rs with
  | nil =>
    rw [hrs] at hdisp
    unfold OpPostOutcome SporosProvLen.lstep
    rw [hrs]
    exact hdisp
  | cons f rest =>
    rw [hrs] at hdisp
    dsimp only at hdisp
    by_cases hnw : f.word < pr.words.length
    · rw [dif_pos hnw] at hdisp
      by_cases hpc : f.pc < pr.words[f.word]!.code.length
      · rw [dif_pos hpc] at hdisp
        obtain ⟨s1, hrch1, hrip1, hk1, hwf1, hr0_1, _, hc264⟩ := hdisp
        have hor1 : OracleAdmits s1 := oracleAdmits_of_reaches hor hrch1
        let ins := pr.words[f.word]!.code[f.pc]!
        let ctx1 : Region :=
          { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 264 (bytes8 (instrPair ins).2)) 256 (bytes8 (instrPair ins).1) }
        let vc1 : VmCtx := { vc with ctxReg := ctx1 }
        have h264 : rd64 s1.mem (ea ctxBase 264) = some (instrPair pr.words[f.word]!.code[f.pc]!).2 :=
          vc1.rd64_ctx_of_eq hwf1 (by decide) hc264
        have hwd_mem : pr.words[f.word]! ∈ pr.words := getElem!_mem_of_lt hnw
        have hcode_lt : pr.words[f.word]!.code.length < X86Env.W :=
          ((vc.code_flat hvc).2.2.2.1.2.2.2.2.2 _ hwd_mem).2.2.2.1
        have hpcW : f.pc + 1 < X86Env.W := by omega
        apply lift_opPostOutcome hrch1 hk1
        cases hins : pr.words[f.word]!.code[f.pc]! with
        | prim p =>
          rw [hins] at hrip1 hr0_1
          exact prim_dispatch_spec p hrip1 hwf1 hor1 hrs (fetch_eq_of_lt hnw hpc hins) hpcW hr0_1
        | lit n =>
          rw [hins] at hrip1 h264
          exact O_lit_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | str i =>
          rw [hins] at hrip1 h264
          exact O_str_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | pick n =>
          rw [hins] at hrip1 h264
          exact O_pick_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | tor =>
          rw [hins] at hrip1 h264
          exact O_tor_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | fromr =>
          rw [hins] at hrip1 h264
          exact O_fromr_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | rfetch =>
          rw [hins] at hrip1 h264
          exact O_rfetch_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | locget i =>
          rw [hins] at hrip1 h264
          exact O_locget_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | locset i =>
          rw [hins] at hrip1 h264
          exact O_locset_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | jmp t =>
          rw [hins] at hrip1 h264
          exact O_jmp_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | jz t =>
          rw [hins] at hrip1 h264
          exact O_jz_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | jnz t =>
          rw [hins] at hrip1 h264
          exact O_jnz_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | call w =>
          rw [hins] at hrip1 h264
          exact O_call_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
        | ret =>
          rw [hins] at hrip1 h264
          exact O_ret_spec hrip1 hwf1 hor1 hrs hnw hpc hins hpcW h264
      · rw [dif_neg hpc] at hdisp
        unfold OpPostOutcome SporosProvLen.lstep
        rw [hrs]; dsimp only
        rw [fetch_none_of_not_pc hnw hpc]
        exact hdisp
    · rw [dif_neg hnw] at hdisp
      unfold OpPostOutcome SporosProvLen.lstep
      rw [hrs]; dsimp only
      rw [fetch_none_of_not_word hnw]
      exact hdisp

/-! ### Multi-step simulation (`SporosProvLen.lrun` and `Sporos.run` under `noPhys`) -/

/-- Multi-step simulation outcome matching `SporosProvLen.lrun pr n ls`. -/
def LRunOutcome (pr : Sporos.Prog) (n : Nat) (ls : SporosProvLen.LSt) (s₀ : X86Vm.St) : Prop :=
  match SporosProvLen.lrun pr n ls with
  | .next ls' =>
    (∃ s', Reaches vmImage s₀ s' ∧ s'.k.out = s₀.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2))
  | .halt c out =>
    (∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit (c % 256) ∧
        Ends vmImage s₀ (.exit (c % 256)) ∧ s_exit.k.out = s₀.k.out ++ out.reverse) ∨
      (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2))
  | .fail _ =>
    (∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧
        Ends vmImage s₀ (.exit 1) ∧ s_exit.k.out = s₀.k.out) ∨
      (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2))

private theorem lift_lrunOutcome {pr : Sporos.Prog} {n : Nat} {ls : SporosProvLen.LSt}
    {s₀ s₁ : X86Vm.St}
    (hrch : Reaches vmImage s₀ s₁) (hout_eq : s₁.k.out = s₀.k.out)
    (h : LRunOutcome pr n ls s₁) : LRunOutcome pr n ls s₀ := by
  unfold LRunOutcome at h ⊢
  cases hl : SporosProvLen.lrun pr n ls with
  | next ls' =>
    rw [hl] at h
    rcases h with ⟨s', hr', hk', hor', hrel'⟩ | ⟨hh, he⟩
    · exact Or.inl ⟨s', Reaches.trans hrch hr', hk'.trans hout_eq, hor', hrel'⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch hh, Ends.of_reaches hrch he⟩
  | halt c out =>
    rw [hl] at h
    rcases h with ⟨sx, hrx, hsx, _, hkx⟩ | ⟨hh, he⟩
    · have htot := Reaches.trans hrch hrx
      exact Or.inl ⟨sx, htot, hsx, Ends.of_reaches htot (Ends.step hsx), by rw [hkx, hout_eq]⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch hh, Ends.of_reaches hrch he⟩
  | fail e =>
    rw [hl] at h
    rcases h with ⟨sx, hrx, hsx, _, hkx⟩ | ⟨hh, he⟩
    · have htot := Reaches.trans hrch hrx
      exact Or.inl ⟨sx, htot, hsx, Ends.of_reaches htot (Ends.step hsx), hkx.trans hout_eq⟩
    · exact Or.inr ⟨hitsNomem_of_reaches hrch hh, Ends.of_reaches hrch he⟩

/-- Multi-step simulation of `SporosProvLen.lrun` by the native x86-64 `sporos_vm` binary. -/
theorem sporos_vm_lrun_sim (pr : Sporos.Prog) (n : Nat) {ls : SporosProvLen.LSt} {s₀ : X86Vm.St}
    (hrel : VmRel pr ls s₀) (hor : OracleAdmits s₀) :
    LRunOutcome pr n ls s₀ := by
  induction n generalizing ls s₀ with
  | zero =>
    unfold LRunOutcome SporosProvLen.lrun
    exact Or.inl ⟨s₀, Reaches.refl vmImage s₀, rfl, hor, hrel⟩
  | succ n ih =>
    have hstep := sporos_vm_step_sim hrel hor
    unfold OpPostOutcome at hstep
    unfold LRunOutcome SporosProvLen.lrun
    cases hls : SporosProvLen.lstep pr ls with
    | next ls1 =>
      rw [hls] at hstep
      dsimp only
      rcases hstep with ⟨s1, hr1, hk1, hor1, hrel1⟩ | ⟨hh, he⟩
      · exact lift_lrunOutcome hr1 hk1 (ih hrel1 hor1)
      · cases SporosProvLen.lrun pr n ls1 with
        | next _ => exact Or.inr ⟨hh, he⟩
        | halt _ _ => exact Or.inr ⟨hh, he⟩
        | fail _ => exact Or.inr ⟨hh, he⟩
    | halt c out =>
      rw [hls] at hstep
      obtain ⟨sx, hrx, hsx, hkx⟩ := hstep
      exact Or.inl ⟨sx, hrx, hsx, Ends.of_reaches hrx (Ends.step hsx), hkx⟩
    | fail e =>
      rw [hls] at hstep
      obtain ⟨sx, hrx, hsx, hkx⟩ := hstep
      exact Or.inl ⟨sx, hrx, hsx, Ends.of_reaches hrx (Ends.step hsx), by rw [hkx]⟩

/-- End-to-end simulation of the reference `Sporos.run pr Sporos.noPhys` semantics by
the committed x86-64 `sporos_vm` binary (`vmImage`), composing `sporos_vm_lrun_sim` with
`SporosProvLen.lrun_sim_init`. -/
theorem sporos_vm_sim_noPhys (pr : Sporos.Prog) (msize : Nat) (inp : List Nat) (n : Nat)
    {s₀ : X86Vm.St}
    (hrel : VmRel pr (SporosProvLen.linit pr msize inp) s₀)
    (hor : OracleAdmits s₀) :
    match Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
    | .next st' =>
      (∃ ls' s', SporosProvLen.Rel st' ls' ∧
          Reaches vmImage s₀ s' ∧ s'.k.out = s₀.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
        (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2))
    | .halt c out =>
      (∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit (c % 256) ∧
          Ends vmImage s₀ (.exit (c % 256)) ∧
          s_exit.k.out = s₀.k.out ++ (out.map Prod.fst).reverse) ∨
        (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2))
    | .fail _ =>
      (∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧
          Ends vmImage s₀ (.exit 1) ∧ s_exit.k.out = s₀.k.out) ∨
        (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2)) := by
  have hlr := sporos_vm_lrun_sim pr n hrel hor
  have hsim := SporosProvLen.lrun_sim_init pr msize inp n
  unfold LRunOutcome at hlr
  cases hr : Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
  | next st' =>
    rw [hr] at hsim
    cases hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) with
    | next ls' =>
      rw [hl] at hsim hlr
      cases hsim with
      | next hrel_st =>
        rcases hlr with ⟨s', h1, h2, h3, h4⟩ | hnm
        · exact Or.inl ⟨ls', s', hrel_st, h1, h2, h3, h4⟩
        · exact Or.inr hnm
    | halt _ _ => rw [hl] at hsim; cases hsim
    | fail _ => rw [hl] at hsim; cases hsim
  | halt c out =>
    rw [hr] at hsim
    cases hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) with
    | next _ => rw [hl] at hsim; cases hsim
    | halt c' out' =>
      rw [hl] at hsim hlr
      cases hsim with
      | halt ho =>
        rw [← ho] at hlr
        exact hlr
    | fail _ => rw [hl] at hsim; cases hsim
  | fail e =>
    rw [hr] at hsim
    cases hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) with
    | next _ => rw [hl] at hsim; cases hsim
    | halt _ _ => rw [hl] at hsim; cases hsim
    | fail e' =>
      rw [hl] at hsim hlr
      exact hlr

/-- Unconditional end-to-end simulation when the oracle contains no `.nomem` entry. -/
theorem sporos_vm_sim_noPhys_noNomem (pr : Sporos.Prog) (msize : Nat) (inp : List Nat) (n : Nat)
    {s₀ : X86Vm.St}
    (hrel : VmRel pr (SporosProvLen.linit pr msize inp) s₀)
    (hor : OracleAdmits s₀)
    (hno : ∀ c ∈ s₀.k.oracle, c ≠ .nomem) :
    match Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
    | .next st' =>
      ∃ ls' s', SporosProvLen.Rel st' ls' ∧
        Reaches vmImage s₀ s' ∧ s'.k.out = s₀.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s'
    | .halt c out =>
      ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit (c % 256) ∧
        Ends vmImage s₀ (.exit (c % 256)) ∧
        s_exit.k.out = s₀.k.out ++ (out.map Prod.fst).reverse
    | .fail _ =>
      ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧
        Ends vmImage s₀ (.exit 1) ∧ s_exit.k.out = s₀.k.out := by
  have h := sporos_vm_sim_noPhys pr msize inp n hrel hor
  have hnn := not_hitsNomem_of_no_nomem hno
  cases hr : Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
  | next _ => rw [hr] at h; exact h.resolve_right (fun ⟨hh, _⟩ => hnn hh)
  | halt _ _ => rw [hr] at h; exact h.resolve_right (fun ⟨hh, _⟩ => hnn hh)
  | fail _ => rw [hr] at h; exact h.resolve_right (fun ⟨hh, _⟩ => hnn hh)

/-- Checked programs never trigger static checker errors (`.underflow`, `.auxUnderflow`,
`.badLocal`, `.badJump`, `.badWord`, `.noFrame`), so any `sporos_vm` exit-1 runtime
failure on a checked program corresponds solely to a dynamic runtime error
(`.bounds`, `.divZero`, `.badTable`, or `.demand`). -/
theorem sporos_vm_sim_checked_no_static_fail (pr : Sporos.Prog) (hp : Sporos.checkProg pr = true)
    (msize : Nat) (inp : List Nat) (n : Nat) (e : Sporos.Err)
    (hfail : Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) = .fail e) :
    e = .bounds ∨ e = .divZero ∨ e = .badTable ∨ e = .demand := by
  have hlf : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .fail e :=
    (SporosProvLen.lrun_sim_fail pr msize inp n e).mpr hfail
  have hsafe := SporosProvLen.lrun_safe pr hp msize inp n e hlf
  cases e with
  | bounds => exact Or.inl rfl
  | divZero => exact Or.inr (Or.inl rfl)
  | badTable => exact Or.inr (Or.inr (Or.inl rfl))
  | demand => exact Or.inr (Or.inr (Or.inr rfl))
  | underflow | auxUnderflow | badLocal | badJump | badWord | noFrame =>
    cases hsafe

end SporosVmSim
