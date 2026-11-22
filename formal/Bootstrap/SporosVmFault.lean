import Bootstrap.SporosVmSimBase
import Bootstrap.TMR

/-!
# Dispatch-Boundary TMR Disturbance Safety for `sporos_vm`

Formal verification of the triplicated flat-code majority voters in `sporos_vm`
(`bootstrap/model/sporos_vm`, `SporosVmBin`) under architectural execution
(`X86Vm.run vmImage`) and the 4-channel physical disturbance model
(`X86Vm.prun vmAlts (r.state hr) r.fetch vmImage` over `PDist`).

## Disturbance Model and Scope
`sporos_vm` keeps three read-only replicas (`code1`, `code2`, `code3`, at
`ctx[232]`, `ctx[240]`, `ctx[248]`) of `encodeFlatProg pr`. Every code-read
boundary loads all three replica words into `(rax, rbx, rcx)` (`regs 0, 3, 1`)
and votes them via `SporosVmBin.L_vm_vote` (`0x40024a`, jumping to `L_vm_dispatch`
on majority or `L_exit3` on 3-way disagreement) or one of the 12 inline 3-way
voters (`L_vm_loop`: 5, `O_str`: 3, `O_call`: 4), masking any 1-of-3 replica
perturbation and exiting with code `3` at `SporosVmBin.L_exit3` (`0x400264`)
on 3-way pairwise disagreement.

## Explicitly Excluded Disturbances
- **Pre-triplication loader execution**: `mmap`, file `read`, decode, check,
  and initial table copy into `code1`/`code2`/`code3` before `L_vm_loop`.
- **Simultaneous multi-replica perturbations (`≥ 2` of 3)** at the same offset.
- **Untriplicated post-dispatch state**: `ds`, `aux`, `rs`, `locs`, `memw`,
  `memp`, `tabs`, `kv`, `inp`, `out`, `ctx`, and post-dispatch ALU/control.

## Root A Framing
Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
`spr1_check` is an independent second check (not Root A or Root A's trust base);
`sporos_vm` is the post-admission native x86-64 VM paired with `Sporos.decode`
and `Sporos.checkProg`.
-/

namespace SporosVmFault

open X86Env X86Int X86Vm PDist TMR SporosVmCore SporosVmExec SporosVmSimBase

/-- Exact architectural step count of `L_vm_vote` (`0x40024a`) on `(a, b, c)`:
`5` (`a = b`), `7` (`a ≠ b ∧ a = c`), `10` (`a ≠ b ∧ a ≠ c ∧ b = c`), or `9`
(3-way pairwise disagreement -> `.exit 3`). -/
def voteSteps (a b c : Nat) : Nat :=
  if a = b then 5 else if a = c then 7 else if b = c then 10 else 9

theorem voteSteps_le_10 (a b c : Nat) : voteSteps a b c ≤ 10 := by
  unfold voteSteps; repeat (split <;> try omega)

private theorem maj_third {a b c v : Nat} (hab : a ≠ b) (hac : a ≠ c)
    (h : (a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) : b = c ∧ b = v := by
  rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact absurd (h1.trans h2.symm) hab
  · exact absurd (h1.trans h2.symm) hac
  · exact ⟨h1.trans h2.symm, h1⟩

private theorem cmp_jcc_run2 {r1 r2 cc a tgt x y : Nat} {P : Prop} [Decidable P] {s : X86Vm.St}
    (hrip : s.rip = a) (h1 : s.reg r1 = some x) (h2 : s.reg r2 = some y)
    (hm1 : (a, (.ofInt (.cmp r1 r2)), 3) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 3, (.ofInt (.jcc cc tgt)), 2) ∈ SporosVmBin.vmTable)
    (hcond : X86Int.cond (X86Int.aluF .sub x y).2 cc = some (decide P)) :
    X86Vm.run vmImage 2 (.ok s) =
      .ok { s with fl := (X86Int.aluF .sub x y).2, rip := if P then tgt else a + 5 } := by
  have hs1 : X86Vm.step vmImage s = .ok { s with fl := (X86Int.aluF .sub x y).2, rip := a + 3 } := by
    rw [step_of_table (hrip ▸ hm1), exec_cmp h1 h2, hrip]
  have hs2 : X86Vm.step vmImage { s with fl := (X86Int.aluF .sub x y).2, rip := a + 3 } =
      .ok { s with fl := (X86Int.aluF .sub x y).2, rip := if P then tgt else a + 5 } := by
    rw [step_of_table hm2, exec_jcc hcond]; by_cases hp : P <;> simp [hp]
  rw [X86Vm.run, hs1, X86Vm.run, hs2, X86Vm.run]

/-- Exact `L_exit3` (`0x400264`) termination in 3 steps with `.exit 3`. -/
theorem L_exit3_run_spec {s : X86Vm.St} (hrip : s.rip = SporosVmBin.L_exit3) (hr : RegsOk s) :
    X86Vm.run vmImage 3 (.ok s) = .exit 3 := by
  let s1 := { s.set 7 (some 3) with rip := 4194921 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_chunk 1 hrip rfl (by decide : (SporosVmBin.L_exit3, (.ofInt (.movI 7 3)), 5) ∈ SporosVmBin.vmTable_1),
      exec_movI_lt (by decide), hrip]; rfl
  have hr1 : RegsOk s1 := regsOk_set hr (r := 7) (by decide) 3
  let s2 := { s1.set 0 (some 60) with rip := 4194926 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194921, (.ofInt (.movI 0 60)), 5) ∈ SporosVmBin.vmTable_1),
      exec_movI_lt (by decide)]
  have hr2 : RegsOk s2 := regsOk_set hr1 (r := 0) (by decide) 60
  obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
    regsOk_syscall_regs hr2
  have hv0' : v0 = 60 := Option.some.inj (hv0.symm.trans (by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]))
  have hv7' : v7 = 3 := Option.some.inj (hv7.symm.trans (by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr.1]; decide)]))
  subst hv0' hv7'
  have h03 : X86Vm.step vmImage s2 = .exit 3 := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194926, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_1),
      exec_syscall_exit (c := 3) hv0 hv7 hv6 hv2 hv10 hv8 hv9 (by simp [X86Env.syscall])]
  rw [X86Vm.run, h01, X86Vm.run, h02, X86Vm.run, h03, X86Vm.run]

theorem L_exit3_spec {s : X86Vm.St} (hrip : s.rip = SporosVmBin.L_exit3) (hr : RegsOk s) :
    Ends vmImage s (.exit 3) := ⟨3, L_exit3_run_spec hrip hr⟩

/-! ### Canonical 38-byte TMR voter at `L_vm_vote` (`0x40024a`) -/

private theorem L_vm_vote_repair_tail {v : Nat} {s : X86Vm.St}
    (hrip : s.rip = 4194908) (hr : RegsOk s) (h0 : s.reg 0 = some v) :
    let s' := { (s.set 3 (some v)).set 1 (some v) with rip := SporosVmBin.L_vm_dispatch }
    X86Vm.run vmImage 3 (.ok s) = .ok s' ∧ s'.rip = SporosVmBin.L_vm_dispatch ∧
    s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧ s'.reg 0 = some v ∧ s'.reg 1 = some v ∧
    s'.reg 3 = some v ∧ (∀ r, r ≠ 1 → r ≠ 3 → s'.reg r = s.reg r) := by
  let s1 := { s.set 3 (some v) with rip := 4194911 }
  have h01 : X86Vm.step vmImage s = .ok s1 := by
    rw [step_at_chunk 1 hrip rfl (by decide : (4194908, (.ofInt (.mov 3 0)), 3) ∈ SporosVmBin.vmTable_1), exec_mov h0, hrip]
  have hr1 : RegsOk s1 := regsOk_set hr (r := 3) (by decide) v
  have hs1_0 : s1.reg 0 = some v := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 0)]; exact h0
  let s2 := { s1.set 1 (some v) with rip := 4194914 }
  have h02 : X86Vm.step vmImage s1 = .ok s2 := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194911, (.ofInt (.mov 1 0)), 3) ∈ SporosVmBin.vmTable_1), exec_mov hs1_0]
  let s3 := { s2 with rip := SporosVmBin.L_vm_dispatch }
  have h03 : X86Vm.step vmImage s2 = .ok s3 := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194914, (.ofInt (.jmp 4194928)), 2) ∈ SporosVmBin.vmTable_1), exec_jmp]; rfl
  refine ⟨by rw [X86Vm.run, h01, X86Vm.run, h02, X86Vm.run, h03, X86Vm.run]; rfl,
    rfl, rfl, rfl, regsOk_set hr1 (r := 1) (by decide) v, ?_, ?_, ?_, ?_⟩
  · simp only [X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0),
      List.getElem?_set_ne (by decide : 3 ≠ 0)]; exact h0
  · simp only [X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by simp [hr.1])]
  · simp only [X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 3)]
    rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  · intro r h1 h3; simp only [X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (Ne.symm h1), List.getElem?_set_ne (Ne.symm h3)]

/-- Architectural 1-of-3 perturbation masking at `L_vm_vote` (`0x40024a`):
repairs `(r0, r1, r3)` to `v` at `L_vm_dispatch` in `voteSteps a b c ≤ 10` steps. -/
theorem L_vm_vote_majority_spec {a b c v : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hr : RegsOk s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hmaj : (a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) :
    ∃ s', X86Vm.run vmImage (voteSteps a b c) (.ok s) = .ok s' ∧
      s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 3 → s'.reg r = s.reg r) := by
  have hc1 := cmp_jcc_run2 (cc := 4) (a := 4194890) (tgt := 4194908) hrip h0 h3
    (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_e a b ha hb)
  by_cases hab : a = b
  · have hav : a = v := by rcases hmaj with ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    subst hab; subst hav; simp only [↓reduceIte] at hc1
    have ht := L_vm_vote_repair_tail (v := a) (s := { s with fl := (X86Int.aluF .sub a a).2, rip := 4194908 }) rfl hr h0
    exact ⟨_, by rw [show voteSteps a a c = 2 + 3 by simp [voteSteps], X86Vm.run_add, hc1]; exact ht.1,
      ht.2.1, ht.2.2.1, ht.2.2.2.1, ht.2.2.2.2.1, ht.2.2.2.2.2.1, ht.2.2.2.2.2.2.1,
      ht.2.2.2.2.2.2.2.1, fun r _ hr1 hr3 => ht.2.2.2.2.2.2.2.2 r hr1 hr3⟩
  · simp only [hab, ↓reduceIte] at hc1
    have hc2 := cmp_jcc_run2 (cc := 4) (a := 4194895) (tgt := 4194908)
      (s := { s with fl := (X86Int.aluF .sub a b).2, rip := 4194895 }) rfl h0 h1
      (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_e a c ha hc)
    by_cases hac : a = c
    · have hav : a = v := by rcases hmaj with ⟨h, _⟩ | ⟨h, _⟩ | ⟨_, h⟩ <;> omega
      subst hac; subst hav; simp only [↓reduceIte] at hc2
      have ht := L_vm_vote_repair_tail (v := a) (s := { s with fl := (X86Int.aluF .sub a a).2, rip := 4194908 }) rfl hr h0
      exact ⟨_, by rw [show voteSteps a b a = 2 + (2 + 3) by simp [voteSteps, hab],
          X86Vm.run_add, hc1, X86Vm.run_add, hc2]; exact ht.1,
        ht.2.1, ht.2.2.1, ht.2.2.2.1, ht.2.2.2.2.1, ht.2.2.2.2.2.1, ht.2.2.2.2.2.2.1,
        ht.2.2.2.2.2.2.2.1, fun r _ hr1 hr3 => ht.2.2.2.2.2.2.2.2 r hr1 hr3⟩
    · rcases maj_third hab hac hmaj with ⟨rfl, rfl⟩; simp only [hac, ↓reduceIte] at hc2
      have hc3 := cmp_jcc_run2 (cc := 5) (a := 4194900) (tgt := 4194916)
        (s := { s with fl := (X86Int.aluF .sub a b).2, rip := 4194900 }) rfl h3 h1
        (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_ne b b hb hb)
      simp only [ne_eq, not_true_eq_false, ↓reduceIte] at hc3
      let s6 := { s with fl := (X86Int.aluF .sub b b).2, rip := 4194905 }
      let s7 := { s6.set 0 (some b) with rip := 4194908 }
      have hmov : X86Vm.step vmImage s6 = .ok s7 := by
        rw [step_at_chunk 1 rfl rfl (by decide : (4194905, (.ofInt (.mov 0 3)), 3) ∈ SporosVmBin.vmTable_1), exec_mov (s := s6) h3]
      have hs7_0 : s7.reg 0 = some b := by
        simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
      have ht := L_vm_vote_repair_tail (v := b) (s := s7) rfl (regsOk_set hr (r := 0) (by decide) b) hs7_0
      refine ⟨_, by rw [show voteSteps a b b = 2 + (2 + (2 + (1 + 3))) by simp [voteSteps, hab],
          X86Vm.run_add, hc1, X86Vm.run_add, hc2, X86Vm.run_add, hc3, X86Vm.run_add, X86Vm.run, hmov, X86Vm.run]; exact ht.1,
        ht.2.1, ht.2.2.1, ht.2.2.2.1, ht.2.2.2.2.1, ht.2.2.2.2.2.1, ht.2.2.2.2.2.2.1, ht.2.2.2.2.2.2.2.1, fun r hr0 hr1 hr3 => ?_⟩
      rw [ht.2.2.2.2.2.2.2.2 r hr1 hr3]; simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (Ne.symm hr0)]

/-- Architectural 3-way pairwise disagreement refusal at `L_vm_vote`: exits `3` in 9 steps. -/
theorem L_vm_vote_disagree_run_spec {a b c : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hr : RegsOk s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : X86Vm.run vmImage 9 (.ok s) = .exit 3 := by
  have hc1 := cmp_jcc_run2 (cc := 4) (a := 4194890) (tgt := 4194908) hrip h0 h3
    (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_e a b ha hb)
  have hc2 := cmp_jcc_run2 (cc := 4) (a := 4194895) (tgt := 4194908)
    (s := { s with fl := (X86Int.aluF .sub a b).2, rip := 4194895 }) rfl h0 h1
    (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_e a c ha hc)
  have hc3 := cmp_jcc_run2 (cc := 5) (a := 4194900) (tgt := 4194916)
    (s := { s with fl := (X86Int.aluF .sub a c).2, rip := 4194900 }) rfl h3 h1
    (mem_vmTable_of_chunk 1 rfl (by decide)) (mem_vmTable_of_chunk 1 rfl (by decide)) (cond_aluF_sub_ne b c hb hc)
  simp only [hab, hac, hbc, ne_eq, not_false_eq_true, ↓reduceIte] at hc1 hc2 hc3
  rw [show (9 : Nat) = 2 + (2 + (2 + 3)) from rfl, X86Vm.run_add, hc1, X86Vm.run_add, hc2, X86Vm.run_add, hc3]
  exact L_exit3_run_spec rfl hr

theorem L_vm_vote_disagree_spec {a b c : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hr : RegsOk s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : Ends vmImage s (.exit 3) :=
  ⟨9, L_vm_vote_disagree_run_spec hrip hr h0 h3 h1 ha hb hc hab hac hbc⟩

/-! ### Inline 3-way code-read voters (`L_vm_loop`, `O_str`, `O_call`) -/

/-- The 8-instruction inline 3-way voter at address `a`, landing at `a + 23`
on majority or jumping to `L_exit3` on 3-way pairwise disagreement. -/
def InlineVoteAt (a : Nat) : Prop :=
  (a, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable ∧ (a + 3, (.ofInt (.jcc 4 (a + 23))), 2) ∈ SporosVmBin.vmTable ∧
  (a + 5, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable ∧ (a + 8, (.ofInt (.jcc 4 (a + 23))), 2) ∈ SporosVmBin.vmTable ∧
  (a + 10, (.ofInt (.cmp 3 1)), 3) ∈ SporosVmBin.vmTable ∧ (a + 13, (.ofInt (.jcc 4 (a + 20))), 2) ∈ SporosVmBin.vmTable ∧
  (a + 15, (.ofInt (.jmp SporosVmBin.L_exit3)), 5) ∈ SporosVmBin.vmTable ∧ (a + 20, (.ofInt (.mov 0 3)), 3) ∈ SporosVmBin.vmTable

theorem inline_vote_majority_spec {a x1 x2 x3 v : Nat} {s : X86Vm.St}
    (hiv : InlineVoteAt a) (hrip : s.rip = a) (hr : RegsOk s)
    (h0 : s.reg 0 = some x1) (h3 : s.reg 3 = some x2) (h1 : s.reg 1 = some x3)
    (hx1 : x1 < X86Env.W) (hx2 : x2 < X86Env.W) (hx3 : x3 < X86Env.W)
    (hmaj : (x1 = v ∧ x2 = v) ∨ (x1 = v ∧ x3 = v) ∨ (x2 = v ∧ x3 = v)) :
    ∃ s', Reaches vmImage s s' ∧ s'.rip = a + 23 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg 0 = some v ∧ (∀ r, r ≠ 0 → s'.reg r = s.reg r) := by
  obtain ⟨hm0, hm1, hm2, hm3, hm4, hm5, _, hm7⟩ := hiv
  have hc1 := cmp_jcc_run2 hrip h0 h3 hm0 hm1 (cond_aluF_sub_e x1 x2 hx1 hx2)
  by_cases h12 : x1 = x2
  · have hv : x1 = v := by rcases hmaj with ⟨h, _⟩ | ⟨h, _⟩ | ⟨h, _⟩ <;> omega
    subst h12; subst hv; simp only [↓reduceIte] at hc1
    exact ⟨_, ⟨2, hc1⟩, rfl, rfl, rfl, hr, h0, fun _ _ => rfl⟩
  · simp only [h12, ↓reduceIte] at hc1
    have hc2 := cmp_jcc_run2 (a := a + 5) (tgt := a + 23)
      (s := { s with fl := (X86Int.aluF .sub x1 x2).2, rip := a + 5 }) rfl h0 h1 hm2
      (by simpa [Nat.add_assoc] using hm3) (cond_aluF_sub_e x1 x3 hx1 hx3)
    by_cases h13 : x1 = x3
    · have hv : x1 = v := by rcases hmaj with ⟨h, _⟩ | ⟨h, _⟩ | ⟨_, h⟩ <;> omega
      subst h13; subst hv; simp only [↓reduceIte] at hc2
      exact ⟨_, Reaches.trans ⟨2, hc1⟩ ⟨2, hc2⟩, rfl, rfl, rfl, hr, h0, fun _ _ => rfl⟩
    · rcases maj_third h12 h13 hmaj with ⟨rfl, rfl⟩; simp only [h13, ↓reduceIte] at hc2
      have hc3 := cmp_jcc_run2 (a := a + 10) (tgt := a + 20)
        (s := { s with fl := (X86Int.aluF .sub x1 x2).2, rip := a + 10 }) rfl h3 h1 hm4
        (by simpa [Nat.add_assoc] using hm5) (cond_aluF_sub_e x2 x2 hx2 hx2)
      simp only [↓reduceIte] at hc3
      let s6 := { s with fl := (X86Int.aluF .sub x2 x2).2, rip := a + 20 }
      let s7 := { s6.set 0 (some x2) with rip := a + 23 }
      have hmov : X86Vm.step vmImage s6 = .ok s7 := by rw [step_of_table hm7, exec_mov (s := s6) h3]
      refine ⟨s7, Reaches.trans ⟨2, hc1⟩ (Reaches.trans ⟨2, by simpa [Nat.add_assoc] using hc2⟩
        (Reaches.trans ⟨2, hc3⟩ (Reaches.step hmov))), rfl, rfl, rfl, regsOk_set hr (r := 0) (by decide) x2, ?_, ?_⟩
      · simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
      · intro r hr0; simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (Ne.symm hr0)]

theorem inline_vote_disagree_spec {a x1 x2 x3 : Nat} {s : X86Vm.St}
    (hiv : InlineVoteAt a) (hrip : s.rip = a) (hr : RegsOk s)
    (h0 : s.reg 0 = some x1) (h3 : s.reg 3 = some x2) (h1 : s.reg 1 = some x3)
    (hx1 : x1 < X86Env.W) (hx2 : x2 < X86Env.W) (hx3 : x3 < X86Env.W)
    (h12 : x1 ≠ x2) (h13 : x1 ≠ x3) (h23 : x2 ≠ x3) : Ends vmImage s (.exit 3) := by
  obtain ⟨hm0, hm1, hm2, hm3, hm4, hm5, hm6, _⟩ := hiv
  have hc1 := cmp_jcc_run2 hrip h0 h3 hm0 hm1 (cond_aluF_sub_e x1 x2 hx1 hx2)
  have hc2 := cmp_jcc_run2 (a := a + 5) (tgt := a + 23)
    (s := { s with fl := (X86Int.aluF .sub x1 x2).2, rip := a + 5 }) rfl h0 h1 hm2
    (by simpa [Nat.add_assoc] using hm3) (cond_aluF_sub_e x1 x3 hx1 hx3)
  have hc3 := cmp_jcc_run2 (a := a + 10) (tgt := a + 20)
    (s := { s with fl := (X86Int.aluF .sub x1 x3).2, rip := a + 10 }) rfl h3 h1 hm4
    (by simpa [Nat.add_assoc] using hm5) (cond_aluF_sub_e x2 x3 hx2 hx3)
  simp only [h12, h13, h23, ↓reduceIte] at hc1 hc2 hc3
  let s6 := { s with fl := (X86Int.aluF .sub x2 x3).2, rip := a + 15 }
  have hjmp : X86Vm.step vmImage s6 = .ok { s6 with rip := SporosVmBin.L_exit3 } := by rw [step_of_table hm6, exec_jmp]
  exact Ends.of_reaches (Reaches.trans ⟨2, hc1⟩ (Reaches.trans ⟨2, by simpa [Nat.add_assoc] using hc2⟩
    (Reaches.trans ⟨2, by simpa [Nat.add_assoc] using hc3⟩ (Reaches.step hjmp)))) (L_exit3_spec rfl hr)

private theorem ld_replica_step {a off dst disp c x : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase)
    (hdst16 : dst < 16) (hdst15 : dst ≠ 15)
    (hm1 : (a, (.ofInt (.ld 2 15 off)), 7) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 7, (.ofInt (.ld dst 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hc : rd64 s.mem (ea ctxBase off) = some c) (hv : rd64 s.mem (ea c disp) = some x) :
    ∃ s', Reaches vmImage s s' ∧ s'.rip = a + 11 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg dst = some x ∧ s'.reg 15 = some ctxBase ∧
      (∀ r, r ≠ 2 → r ≠ dst → s'.reg r = s.reg r) := by
  let s1 := { s.set 2 (some c) with rip := a + 7 }
  have h1 : X86Vm.step vmImage s = .ok s1 := by rw [step_of_table (hrip ▸ hm1), exec_ld h15 hc, hrip]
  have hr1 : RegsOk s1 := regsOk_set hr (r := 2) (by decide) c
  have hs1_2 : s1.reg 2 = some c := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  let s2 := { s1.set dst (some x) with rip := a + 7 + 4 }
  have h2 : X86Vm.step vmImage s1 = .ok s2 := by rw [step_of_table hm2, exec_ld hs1_2 hv]
  refine ⟨s2, Reaches.step_trans h1 (Reaches.step h2), rfl, rfl, rfl, regsOk_set hr1 hdst16 x, ?_, ?_, ?_⟩
  · simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; exact hdst16)]
  · simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne hdst15,
      List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  · intro r hr2 hrd; simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (Ne.symm hrd), List.getElem?_set_ne (Ne.symm hr2)]

/-- Constant-offset 3-replica code read (`code1`/`code2`/`code3` at `+disp`) +
inline voter masks any 1-of-3 replica perturbation to `r0 = some v`. -/
theorem tmr_code_ld_majority_spec {a disp c1 c2 c3 x1 x2 x3 v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase)
    (hm1 : (a, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable) (hm2 : (a + 7, (.ofInt (.ld 0 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 11, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable) (hm4 : (a + 18, (.ofInt (.ld 3 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 22, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable) (hm6 : (a + 29, (.ofInt (.ld 1 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hiv : InlineVoteAt (a + 33)) (hc1 : rd64 s.mem (ea ctxBase 232) = some c1) (hc2 : rd64 s.mem (ea ctxBase 240) = some c2)
    (hc3 : rd64 s.mem (ea ctxBase 248) = some c3) (hv1 : rd64 s.mem (ea c1 disp) = some x1)
    (hv2 : rd64 s.mem (ea c2 disp) = some x2) (hv3 : rd64 s.mem (ea c3 disp) = some x3)
    (hx1 : x1 < X86Env.W) (hx2 : x2 < X86Env.W) (hx3 : x3 < X86Env.W)
    (hmaj : (x1 = v ∧ x2 = v) ∨ (x1 = v ∧ x3 = v) ∨ (x2 = v ∧ x3 = v)) :
    ∃ s', Reaches vmImage s s' ∧ s'.rip = a + 56 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg 0 = some v ∧ (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 3 → s'.reg r = s.reg r) := by
  obtain ⟨s1, hr1, hrip1, hm1', hk1, hok1, h1_0, h1_15, h1_o⟩ :=
    ld_replica_step (dst := 0) hrip hr h15 (by decide) (by decide) hm1 hm2 hc1 hv1
  obtain ⟨s2, hr2, hrip2, hm2', hk2, hok2, h2_3, h2_15, h2_o⟩ :=
    ld_replica_step (dst := 3) (a := a + 11) hrip1 hok1 h1_15 (by decide) (by decide)
      hm3 (by simpa [Nat.add_assoc] using hm4) (hm1' ▸ hc2) (hm1' ▸ hv2)
  obtain ⟨s3, hr3, hrip3, hm3', hk3, hok3, h3_1, _, h3_o⟩ :=
    ld_replica_step (dst := 1) (a := a + 22) (by omega) hok2 h2_15 (by decide) (by decide)
      (by simpa [Nat.add_assoc] using hm5) (by simpa [Nat.add_assoc] using hm6) (hm2' ▸ hm1' ▸ hc3) (hm2' ▸ hm1' ▸ hv3)
  have h3_0 : s3.reg 0 = some x1 := by rw [h3_o 0 (by decide) (by decide), h2_o 0 (by decide) (by decide), h1_0]
  have h3_3 : s3.reg 3 = some x2 := by rw [h3_o 3 (by decide) (by decide), h2_3]
  obtain ⟨s4, hr4, hrip4, hm4', hk4, hok4, hs4_0, hs4_o⟩ :=
    inline_vote_majority_spec hiv (by omega) hok3 h3_0 h3_3 h3_1 hx1 hx2 hx3 hmaj
  exact ⟨s4, Reaches.trans hr1 (Reaches.trans hr2 (Reaches.trans hr3 hr4)), by omega,
    by rw [hm4', hm3', hm2', hm1'], by rw [hk4, hk3, hk2, hk1], hok4, hs4_0,
    fun r h0r h1r h2r h3r => by rw [hs4_o r h0r, h3_o r h2r h1r, h2_o r h2r h3r, h1_o r h2r h0r]⟩

/-- All 12 inline 3-way code-read voter addresses (`L_vm_loop`, `O_str`, `O_call`). -/
def allInlineVoterAddrs : List Nat :=
  [4194548, 4194613, 4194690, 4194758, 4194827, 4199231, 4199311, 4199377, 4200413, 4200485, 4200559, 4200631]

private theorem iv_chunk (k : Nat) {c : List (Nat × X86Vm.Ins × Nat)} {b a : Nat}
    (hc : SporosVmBin.vmTableChunks[k]? = some (b, c))
    (h0 : (a, (.ofInt (.cmp 0 3)), 3) ∈ c := by decide) (h1 : (a + 3, (.ofInt (.jcc 4 (a + 23))), 2) ∈ c := by decide)
    (h2 : (a + 5, (.ofInt (.cmp 0 1)), 3) ∈ c := by decide) (h3 : (a + 8, (.ofInt (.jcc 4 (a + 23))), 2) ∈ c := by decide)
    (h4 : (a + 10, (.ofInt (.cmp 3 1)), 3) ∈ c := by decide) (h5 : (a + 13, (.ofInt (.jcc 4 (a + 20))), 2) ∈ c := by decide)
    (h6 : (a + 15, (.ofInt (.jmp SporosVmBin.L_exit3)), 5) ∈ c := by decide)
    (h7 : (a + 20, (.ofInt (.mov 0 3)), 3) ∈ c := by decide) : InlineVoteAt a :=
  ⟨mem_vmTable_of_chunk k hc h0, mem_vmTable_of_chunk k hc h1, mem_vmTable_of_chunk k hc h2,
   mem_vmTable_of_chunk k hc h3, mem_vmTable_of_chunk k hc h4, mem_vmTable_of_chunk k hc h5,
   mem_vmTable_of_chunk k hc h6, mem_vmTable_of_chunk k hc h7⟩

theorem all_inline_voters_ok : ∀ a ∈ allInlineVoterAddrs, InlineVoteAt a := by
  intro a ha; simp only [allInlineVoterAddrs, List.mem_cons, List.not_mem_nil, or_false] at ha
  rcases ha with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact iv_chunk 0 rfl | exact iv_chunk 1 rfl | exact iv_chunk 15 rfl | exact iv_chunk 16 rfl | exact iv_chunk 20 rfl
    | exact ⟨mem_vmTable_of_chunk 19 rfl (by decide), mem_vmTable_of_chunk 19 rfl (by decide),
             mem_vmTable_of_chunk 19 rfl (by decide), mem_vmTable_of_chunk 19 rfl (by decide),
             mem_vmTable_of_chunk 19 rfl (by decide), mem_vmTable_of_chunk 19 rfl (by decide),
             mem_vmTable_of_chunk 19 rfl (by decide), mem_vmTable_of_chunk 20 rfl (by decide)⟩

/-! ### 4-channel physical disturbance window over `X86Vm.prun` -/
/-- State at `L_vm_vote` with loaded replica words `(a, b, c)` in `(r0, r3, r1)`. -/
def vmVoteState (s : X86Vm.St) (a b c : Nat) : X86Vm.St :=
  { ((s.set 0 (some a)).set 3 (some b)).set 1 (some c) with rip := SporosVmBin.L_vm_vote }
theorem regsOk_vmVoteState {s : X86Vm.St} (hr : RegsOk s) (a b c : Nat) : RegsOk (vmVoteState s a b c) :=
  regsOk_set (regsOk_set (regsOk_set hr (r := 0) (by decide) a) (r := 3) (by decide) b) (r := 1) (by decide) c
/-- Physical 4-channel execution of `L_vm_vote` over `vmAlts` + `vmImage`. -/
def vmVotePhys (r : X86Vm.Rates) (hr : r.ok) (s : X86Vm.St) (a b c : Nat) : PDist.PDF X86Vm.Out :=
  X86Vm.prun vmAlts (r.state hr) r.fetch vmImage (voteSteps a b c) (.ok s)
private theorem rates_total_nonneg (r : X86Vm.Rates) (hr : r.ok) : 0 ≤ r.total :=
  X86Int.nonneg_add hr.1.1 (X86Int.nonneg_add hr.2.1.1 (X86Int.nonneg_add hr.2.2.1.1 hr.2.2.2.1))
theorem vmVotePhys_ok (r : X86Vm.Rates) (hr : r.ok) (s : X86Vm.St) (a b c : Nat) : (vmVotePhys r hr s a b c).ok :=
  X86Vm.prun_ok vmAlts (r.state hr) hr.1 vmImage (voteSteps a b c) (.ok s)
theorem vmVotePhys_fm_le (r : X86Vm.Rates) (hr : r.ok) (s : X86Vm.St) (a b c : Nat) :
    PDist.fm (vmVotePhys r hr s a b c) ≤ 10 * r.total :=
  Rat.le_trans (X86Vm.fm_prun vmAlts (r.state hr) hr.1 vmImage (voteSteps a b c) (.ok s))
    (Rat.mul_le_mul_of_nonneg_right (Rat.natCast_le_natCast.mpr (voteSteps_le_10 a b c)) (rates_total_nonneg r hr))
theorem vmVotePhys_majority (r : X86Vm.Rates) (hr : r.ok) {a b c v : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hreg : RegsOk s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hmaj : (a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) :
    (vmVotePhys r hr s a b c).ok ∧ PDist.fm (vmVotePhys r hr s a b c) ≤ 10 * r.total ∧
    ∃ s', PDist.faultFree (vmVotePhys r hr s a b c) = [.ok s'] ∧
      s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v := by
  obtain ⟨s', hrun, hrip', hmem', hk', hr', hs0', hs1', hs3', _⟩ :=
    L_vm_vote_majority_spec hrip hreg h0 h3 h1 ha hb hc hmaj
  exact ⟨vmVotePhys_ok r hr s a b c, vmVotePhys_fm_le r hr s a b c, s',
    by rw [vmVotePhys, X86Vm.faultFree_prun, hrun], hrip', hmem', hk', hr', hs0', hs1', hs3'⟩
theorem vmVotePhys_disagree (r : X86Vm.Rates) (hr : r.ok) {a b c : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hreg : RegsOk s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (vmVotePhys r hr s a b c).ok ∧ PDist.faultFree (vmVotePhys r hr s a b c) = [.exit 3] ∧
    PDist.fm (vmVotePhys r hr s a b c) ≤ 9 * r.total := by
  have hvs : voteSteps a b c = 9 := by simp [voteSteps, hab, hac, hbc]
  exact ⟨vmVotePhys_ok r hr s a b c,
    by rw [vmVotePhys, hvs, X86Vm.faultFree_prun, L_vm_vote_disagree_run_spec hrip hreg h0 h3 h1 ha hb hc hab hac hbc],
    by rw [vmVotePhys, hvs]; exact X86Vm.fm_prun vmAlts (r.state hr) hr.1 vmImage 9 (.ok s)⟩
/-- Triplicated code-word channel `C` fed into physical `L_vm_vote` over `X86Vm.prun`. -/
def vmDispatchVotedPhys (C : TMR.Channel Nat) (r : X86Vm.Rates) (hr : r.ok) (x : Nat) (s : X86Vm.St) : PDist.PDF X86Vm.Out :=
  TMR.bindW (C.step x) fun a => TMR.bindW (C.step x) fun b => TMR.bindW (C.step x) fun c =>
    PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some (C.f x))) 1 (vmVotePhys r hr (vmVoteState s a.1 b.1 c.1) a.1 b.1 c.1)
private theorem vote_eq_some_maj {a b c v : Nat} (h : TMR.vote a b c = some v) :
    (a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v) := by
  unfold TMR.vote at h
  by_cases h1 : a = b
  · rw [if_pos (Or.inl h1)] at h; exact Or.inl ⟨Option.some.inj h, h1 ▸ Option.some.inj h⟩
  · by_cases h2 : a = c
    · rw [if_pos (Or.inr h2)] at h; exact Or.inr (Or.inl ⟨Option.some.inj h, h2 ▸ Option.some.inj h⟩)
    · rw [if_neg (fun ho => ho.elim h1 h2)] at h
      by_cases h3 : b = c
      · rw [if_pos h3] at h; exact Or.inr (Or.inr ⟨Option.some.inj h, h3 ▸ Option.some.inj h⟩)
      · rw [if_neg h3] at h; exact nomatch h
private theorem sumW_add_const_ok {α : Type} {D : PDist.PDF α} (hD : D.ok) (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + k := by
  have go : ∀ (l : PDist.PDF α), TMR.sumW l (fun x => φ x + k) = TMR.sumW l φ + PDist.mass l * k := by
    intro l; induction l with
    | nil => show (0 : Rat) = 0 + 0 * k; rw [Rat.zero_mul, Rat.add_zero]
    | cons x r ih =>
      show x.2 * (φ x.1 + k) + TMR.sumW r (fun y => φ y + k) = (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 + PDist.mass r) * k
      rw [ih, Rat.mul_add, Rat.add_mul]; grind
  rw [go, hD.2, Rat.one_mul]
private theorem tri_sumW_add_const {D : PDist.PDF Nat} (hD : D.ok) (g : Nat) (qv : Rat) :
    TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
      TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv))) = PDist.fm (TMR.tri D g) + qv := by
  have hinner : ∀ a b : Nat × Bool, TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv) =
      TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))) + qv := fun _ _ => sumW_add_const_ok hD _ _
  have hmid : ∀ a : Nat × Bool, TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv)) =
      TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)))) + qv :=
    fun a => by rw [TMR.sumW_congr D (fun b => hinner a b), sumW_add_const_ok hD]
  have hm : ∀ v : Option Nat, PDist.fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) :=
    fun v => by show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _; rw [Rat.add_zero]; rfl
  rw [TMR.sumW_congr D hmid, sumW_add_const_ok hD]; simp only [TMR.tri, TMR.fm_bindW, hm]

/-- Composed dispatch-boundary TMR bound over `X86Vm.prun`:
total disturbance probability is at most `3 * (C.q * C.q) + 10 * r.total`. -/
theorem vm_dispatch_tri_phys_bound (C : TMR.Channel Nat) (r : X86Vm.Rates) (hr : r.ok)
    (x : Nat) (s : X86Vm.St) (hreg : RegsOk s) (hW : ∀ y ∈ C.step x, y.1.1 < X86Env.W) :
    (vmDispatchVotedPhys C r hr x s).ok ∧
    (∀ u ∈ PDist.faultFree (vmDispatchVotedPhys C r hr x s),
      ∃ s', u = .ok s' ∧ s'.rip = SporosVmBin.L_vm_dispatch ∧
        s'.reg 0 = some (C.f x) ∧ s'.reg 1 = some (C.f x) ∧ s'.reg 3 = some (C.f x)) ∧
    PDist.fm (vmDispatchVotedPhys C r hr x s) ≤ 3 * (C.q * C.q) + 10 * r.total := by
  have hD := C.ok x
  have hrt0 : 0 ≤ 10 * r.total := Rat.mul_nonneg (by decide) (rates_total_nonneg r hr)
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · refine TMR.bindW_nonneg hD.1 fun a => TMR.bindW_nonneg hD.1 fun b =>
      TMR.bindW_nonneg hD.1 fun c => fun y hy => ?_
    simp only [PDist.tagW, List.mem_map] at hy; obtain ⟨z, hz, rfl⟩ := hy
    simpa [Rat.one_mul] using (vmVotePhys_ok r hr (vmVoteState s a.1 b.1 c.1) a.1 b.1 c.1).1 z hz
  · have hm : ∀ a b c, PDist.mass (PDist.tagW (decide (TMR.vote a b c ≠ some (C.f x))) 1
        (vmVotePhys r hr (vmVoteState s a b c) a b c)) = 1 := fun a b c => by
      rw [PDist.mass_tagW, (vmVotePhys_ok r hr (vmVoteState s a b c) a b c).2, Rat.one_mul]
    simp only [vmDispatchVotedPhys, TMR.mass_bindW, hm, TMR.sumW_const, hD.2, Rat.one_mul]
  · intro u hu
    obtain ⟨a, ha, hu⟩ := TMR.mem_faultFree_bindW hu
    obtain ⟨b, hb, hu⟩ := TMR.mem_faultFree_bindW hu
    obtain ⟨c, hc, hu⟩ := TMR.mem_faultFree_bindW hu
    rw [PDist.faultFree_tagW] at hu; split at hu
    · simp at hu
    · next hne =>
      have hmaj := vote_eq_some_maj (by simpa using hne : TMR.vote a.1.1 b.1.1 c.1.1 = some (C.f x))
      have hr0' : (vmVoteState s a.1.1 b.1.1 c.1.1).reg 0 = some a.1.1 := by
        simp only [vmVoteState, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0),
          List.getElem?_set_ne (by decide : 3 ≠ 0)]; rw [getElem?_set_getD_self (by rw [hreg.1]; decide)]
      have hr3' : (vmVoteState s a.1.1 b.1.1 c.1.1).reg 3 = some b.1.1 := by
        simp only [vmVoteState, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 3)]
        rw [getElem?_set_getD_self (by simp [hreg.1])]
      have hr1' : (vmVoteState s a.1.1 b.1.1 c.1.1).reg 1 = some c.1.1 := by
        simp only [vmVoteState, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by simp [hreg.1])]
      obtain ⟨_, _, s', hff, hrip', _, _, _, hs0', hs1', hs3'⟩ :=
        vmVotePhys_majority r hr rfl (regsOk_vmVoteState hreg a.1.1 b.1.1 c.1.1)
          hr0' hr3' hr1' (hW a ha) (hW b hb) (hW c hc) hmaj
      rw [hff] at hu; simp only [List.mem_singleton] at hu; exact ⟨s', hu, hrip', hs0', hs1', hs3'⟩
  · rw [vmDispatchVotedPhys, TMR.fm_bindW]; simp only [TMR.fm_bindW]
    have hcell : ∀ a b c : Nat, PDist.fm (PDist.tagW (decide (TMR.vote a b c ≠ some (C.f x))) 1
        (vmVotePhys r hr (vmVoteState s a b c) a b c)) ≤
        TMR.ι (decide (TMR.vote a b c ≠ some (C.f x))) + 10 * r.total := fun a b c => by
      rw [PDist.fm_tagW]
      by_cases ht : decide (TMR.vote a b c ≠ some (C.f x)) = true
      · rw [if_pos ht, Rat.one_mul, (vmVotePhys_ok r hr _ a b c).2]; simp only [TMR.ι, ht, ↓reduceIte]
        have := add_le_add' (Rat.le_refl (a := (1 : Rat))) hrt0; rwa [Rat.add_zero] at this
      · have hf : decide (TMR.vote a b c ≠ some (C.f x)) = false := by
          cases h : decide (TMR.vote a b c ≠ some (C.f x)) <;> simp_all
        rw [if_neg ht, Rat.one_mul]; simp only [TMR.ι, hf, Bool.false_eq_true, ↓reduceIte, Rat.zero_add]
        exact vmVotePhys_fm_le r hr _ a b c
    have h1 : TMR.sumW (C.step x) (fun a => TMR.sumW (C.step x) (fun b => TMR.sumW (C.step x) (fun c =>
        PDist.fm (PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some (C.f x))) 1
          (vmVotePhys r hr (vmVoteState s a.1 b.1 c.1) a.1 b.1 c.1))))) ≤
        TMR.sumW (C.step x) (fun a => TMR.sumW (C.step x) (fun b => TMR.sumW (C.step x) (fun c =>
          TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some (C.f x))) + 10 * r.total))) :=
      TMR.sumW_le hD.1 fun a _ => TMR.sumW_le hD.1 fun b _ => TMR.sumW_le hD.1 fun c _ => hcell a.1.1 b.1.1 c.1.1
    refine Rat.le_trans h1 ?_; rw [tri_sumW_add_const hD (C.f x) (10 * r.total)]
    exact add_le_add' (TMR.tri_err C x) Rat.le_refl

/-! ### Top-level dispatch-boundary TMR safety theorem -/

/-- Dispatch-boundary TMR safety of `sporos_vm`: 1-of-3 replica perturbation
masking and 3-way disagreement `.exit 3` refusal at `L_vm_vote` (under both
`X86Vm.run` and 4-channel `X86Vm.prun`) and across all 12 inline code voters. -/
theorem sporos_vm_dispatch_tmr_safe
    (cf cr cm cl : PDist.Iv) (r : X86Vm.Rates) (hr : r.ok)
    (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem) (hl : cl.mem r.flag) :
    (∀ (s : X86Vm.St) (a b c v : Nat),
      s.rip = SporosVmBin.L_vm_vote → RegsOk s → s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
      a < X86Env.W → b < X86Env.W → c < X86Env.W →
      ((a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) →
      (vmVotePhys r hr s a b c).ok ∧
      PDist.fm (vmVotePhys r hr s a b c) ≤ 10 * (cf.hi + (cr.hi + (cm.hi + cl.hi))) ∧
      ∃ s', X86Vm.run vmImage (voteSteps a b c) (.ok s) = .ok s' ∧
        PDist.faultFree (vmVotePhys r hr s a b c) = [.ok s'] ∧
        s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v) ∧
    (∀ (s : X86Vm.St) (a b c : Nat),
      s.rip = SporosVmBin.L_vm_vote → RegsOk s → s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
      a < X86Env.W → b < X86Env.W → c < X86Env.W → a ≠ b → a ≠ c → b ≠ c →
      Ends vmImage s (.exit 3) ∧ PDist.faultFree (vmVotePhys r hr s a b c) = [.exit 3] ∧
      PDist.fm (vmVotePhys r hr s a b c) ≤ 9 * (cf.hi + (cr.hi + (cm.hi + cl.hi)))) ∧
    (∀ addr ∈ allInlineVoterAddrs, InlineVoteAt addr ∧
      (∀ (s : X86Vm.St) (x1 x2 x3 v : Nat),
        s.rip = addr → RegsOk s → s.reg 0 = some x1 → s.reg 3 = some x2 → s.reg 1 = some x3 →
        x1 < X86Env.W → x2 < X86Env.W → x3 < X86Env.W →
        ((x1 = v ∧ x2 = v) ∨ (x1 = v ∧ x3 = v) ∨ (x2 = v ∧ x3 = v)) →
        ∃ s', Reaches vmImage s s' ∧ s'.rip = addr + 23 ∧ s'.mem = s.mem ∧
          s'.k = s.k ∧ RegsOk s' ∧ s'.reg 0 = some v) ∧
      (∀ (s : X86Vm.St) (x1 x2 x3 : Nat),
        s.rip = addr → RegsOk s → s.reg 0 = some x1 → s.reg 3 = some x2 → s.reg 1 = some x3 →
        x1 < X86Env.W → x2 < X86Env.W → x3 < X86Env.W →
        x1 ≠ x2 → x1 ≠ x3 → x2 ≠ x3 → Ends vmImage s (.exit 3))) := by
  have htot_le : r.total ≤ cf.hi + (cr.hi + (cm.hi + cl.hi)) :=
    add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))
  refine ⟨?_, ?_, ?_⟩
  · intro s a b c v hrip hreg h0 h3 h1 ha hb hc hmaj
    obtain ⟨hok, hfm, s', hff, hrip', hmem', hk', _, hs0', hs1', hs3'⟩ :=
      vmVotePhys_majority r hr hrip hreg h0 h3 h1 ha hb hc hmaj
    have hrun : X86Vm.run vmImage (voteSteps a b c) (.ok s) = .ok s' := by
      rw [vmVotePhys, X86Vm.faultFree_prun] at hff; exact (List.cons.inj hff).1
    exact ⟨hok, Rat.le_trans hfm (Rat.mul_le_mul_of_nonneg_left htot_le (by decide)),
      s', hrun, hff, hrip', hmem', hk', hs0', hs1', hs3'⟩
  · intro s a b c hrip hreg h0 h3 h1 ha hb hc hab hac hbc
    obtain ⟨_, hff, hfm⟩ := vmVotePhys_disagree r hr hrip hreg h0 h3 h1 ha hb hc hab hac hbc
    exact ⟨L_vm_vote_disagree_spec hrip hreg h0 h3 h1 ha hb hc hab hac hbc,
      hff, Rat.le_trans hfm (Rat.mul_le_mul_of_nonneg_left htot_le (by decide))⟩
  · intro addr haddr
    have hiv := all_inline_voters_ok addr haddr
    refine ⟨hiv, fun s x1 x2 x3 v hrip hreg h0 h3 h1 hx1 hx2 hx3 hmaj => ?_,
      fun s x1 x2 x3 hrip hreg h0 h3 h1 hx1 hx2 hx3 h12 h13 h23 =>
        inline_vote_disagree_spec hiv hrip hreg h0 h3 h1 hx1 hx2 hx3 h12 h13 h23⟩
    obtain ⟨s', hrch, hrip', hmem', hk', hr', hs0', _⟩ :=
      inline_vote_majority_spec hiv hrip hreg h0 h3 h1 hx1 hx2 hx3 hmaj
    exact ⟨s', hrch, hrip', hmem', hk', hr', hs0'⟩

end SporosVmFault
