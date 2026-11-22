import Bootstrap.SporosVmSim
import Bootstrap.SporosVmReady
import Bootstrap.SporosVmFault

/-!
# Ground Non-Vacuity & End-to-End Execution Witnesses (`SporosVmWitness`)

Instantiates `SporosVmReady.mkVmReadyState_rel`, `SporosVmSim.sporos_vm_sim_noPhys`,
`SporosVmSim.sporos_vm_sim_noPhys_noNomem`, and `SporosVmFault.sporos_vm_dispatch_tmr_safe`
on concrete Sporos programs (`Sporos.echo`, `Sporos.Examples.hello`,
`Sporos.Examples.upper`, `Sporos.Examples.fib`, and `SporosProvLen.provDemo`),
confirming non-vacuity of the simulation hypotheses (`VmRel`, `FlatProgFit`,
`readyEndAddr ≤ 2^47`) and end-to-end exit-code / stdout equivalence down to the
committed `12,945`-byte native x86-64 `sporos_vm` image.

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmWitness

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase
  SporosVmReady SporosVmSim SporosVmFault

/-! ### Decidable checker for `FlatProgFit` -/

def instrFitBool : Sporos.Instr → Bool
  | .lit n => decide (n < X86Env.W)
  | .str i => decide (i < X86Env.W)
  | .pick n => decide (n < X86Env.W)
  | .locget i => decide (i < X86Env.W)
  | .locset i => decide (i < X86Env.W)
  | .jmp t => decide (t < X86Env.W)
  | .jz t => decide (t < X86Env.W)
  | .jnz t => decide (t < X86Env.W)
  | .call w => decide (w < X86Env.W)
  | _ => true

def wordFitBool (wd : Sporos.Word) : Bool :=
  decide (wd.ins < X86Env.W) &&
  decide (wd.outs < X86Env.W) &&
  decide (wd.locals < 2 ^ 44) &&
  decide (wd.code.length < X86Env.W) &&
  wd.code.all instrFitBool

def flatProgFitBool (pr : Sporos.Prog) : Bool :=
  decide (pr.main < X86Env.W) &&
  decide (pr.words.length < X86Env.W) &&
  decide (pr.data.length < X86Env.W) &&
  decide ((flatProgQwords pr).length * 8 < 2 ^ 47) &&
  decide (pr.data.flatten.length < X86Env.W) &&
  pr.words.all wordFitBool

theorem flatProgFit_of_bool {pr : Sporos.Prog} (h : flatProgFitBool pr = true) :
    FlatProgFit pr := by
  unfold flatProgFitBool at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, hall⟩ := h
  refine ⟨h1, h2, h3, h4, h5, ?_⟩
  intro wd hwd
  have hw := hall wd hwd
  unfold wordFitBool at hw
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hw
  obtain ⟨⟨⟨⟨hi, ho⟩, hl⟩, hc⟩, hins⟩ := hw
  refine ⟨hi, ho, hl, hc, ?_⟩
  intro ins hi_mem
  have hbi := hins ins hi_mem
  cases ins <;> simp [instrFitBool] at hbi ⊢ <;> exact hbi

/-! ### Helper combinators from `SporosProvLen` concrete halt/fail witnesses -/

private theorem ref_halt_of_lrun_witness {pr : Sporos.Prog} {msize n code : Nat}
    {inp expected : List Nat}
    (hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .halt code expected.reverse) :
    ∃ out, Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) = .halt code out ∧
      (out.map Prod.fst).reverse = expected := by
  obtain ⟨out, hout, hmap⟩ := (SporosProvLen.lrun_sim_halt pr msize inp n code expected.reverse).mp hl
  exact ⟨out, hout, by rw [hmap, List.reverse_reverse]⟩

private theorem vm_halt_of_ready_witness {pr : Sporos.Prog} {msize n code : Nat}
    {inp expected : List Nat} (oracle : List X86Env.Choice)
    (hfit : FlatProgFit pr) (hmsize : msize < 2 ^ 44)
    (hend : readyEndAddr pr msize inp ≤ 2 ^ 47)
    (hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .halt code expected.reverse)
    (hor : OracleAdmits (mkVmReadyState pr msize inp oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState pr msize inp oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧
      X86Vm.step vmImage s_exit = .exit (code % 256) ∧
      Ends vmImage s₀ (.exit (code % 256)) ∧
      s_exit.k.out = expected := by
  intro s₀
  obtain ⟨out, hrun, hrev⟩ := ref_halt_of_lrun_witness hl
  have hrel := mkVmReadyState_rel pr msize inp oracle hfit hmsize hend
  have hsim := sporos_vm_sim_noPhys_noNomem pr msize inp n hrel hor hno
  rw [hrun] at hsim
  obtain ⟨s_exit, hreach, hstep, hends, hout⟩ := hsim
  refine ⟨s_exit, hreach, hstep, hends, ?_⟩
  rw [hout, hrev]
  rfl

private theorem vm_fail_of_ready_witness {pr : Sporos.Prog} {msize n : Nat}
    {inp : List Nat} {e : Sporos.Err} (oracle : List X86Env.Choice)
    (hfit : FlatProgFit pr) (hmsize : msize < 2 ^ 44)
    (hend : readyEndAddr pr msize inp ≤ 2 ^ 47)
    (hl : SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .fail e)
    (hor : OracleAdmits (mkVmReadyState pr msize inp oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState pr msize inp oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧
      X86Vm.step vmImage s_exit = .exit 1 ∧
      Ends vmImage s₀ (.exit 1) ∧
      s_exit.k.out = [] := by
  intro s₀
  have hrun : Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) = .fail e :=
    (SporosProvLen.lrun_sim_fail pr msize inp n e).mp hl
  have hrel := mkVmReadyState_rel pr msize inp oracle hfit hmsize hend
  have hsim := sporos_vm_sim_noPhys_noNomem pr msize inp n hrel hor hno
  rw [hrun] at hsim
  exact hsim

/-! ### Fast `readyEndAddr` bound helper (avoids expanding `bytes8` flatMap) -/

private theorem readyCodeLen_eq (pr : Sporos.Prog) :
    readyCodeLen pr = X86Env.pages (max ((flatProgQwords pr).length * 8) 1) := by
  unfold readyCodeLen encodeFlatProg
  rw [flatMap_bytes8_length]

private theorem readyEndAddr_le_of_eq {pr : Sporos.Prog} {msize : Nat} {inp : List Nat}
    (h : 0x1000000 +
      3 * X86Env.pages (max ((flatProgQwords pr).length * 8) 1) +
      3 * 4096 + readyLocPages pr + readyMemwLen msize + readyMempLen msize +
      2 * 4096 + readyInpPages inp + 4096 ≤ 2 ^ 47) :
    readyEndAddr pr msize inp ≤ 2 ^ 47 := by
  unfold readyEndAddr
  rw [readyCodeLen_eq]
  exact h

/-! ### Ground `FlatProgFit` and `VmRel` non-vacuity witnesses -/

theorem echo_fit : FlatProgFit Sporos.echo := flatProgFit_of_bool (by decide)
theorem hello_fit : FlatProgFit Sporos.Examples.hello := flatProgFit_of_bool (by decide)
theorem upper_fit : FlatProgFit Sporos.Examples.upper := flatProgFit_of_bool (by decide)
theorem fib_fit : FlatProgFit Sporos.Examples.fib := flatProgFit_of_bool (by decide)
theorem provDemo_fit : FlatProgFit SporosProvLen.provDemo := flatProgFit_of_bool (by decide)

theorem echo_ready_rel (oracle : List X86Env.Choice) :
    VmRel Sporos.echo (SporosProvLen.linit Sporos.echo 0 [104, 105])
      (mkVmReadyState Sporos.echo 0 [104, 105] oracle) :=
  mkVmReadyState_rel Sporos.echo 0 [104, 105] oracle echo_fit (by decide) (readyEndAddr_le_of_eq (by decide))

theorem hello_ready_rel (oracle : List X86Env.Choice) :
    VmRel Sporos.Examples.hello (SporosProvLen.linit Sporos.Examples.hello 64 [])
      (mkVmReadyState Sporos.Examples.hello 64 [] oracle) :=
  mkVmReadyState_rel Sporos.Examples.hello 64 [] oracle hello_fit (by decide) (readyEndAddr_le_of_eq (by decide))

theorem upper_ready_rel (oracle : List X86Env.Choice) :
    VmRel Sporos.Examples.upper (SporosProvLen.linit Sporos.Examples.upper 0 [97, 66, 49, 122])
      (mkVmReadyState Sporos.Examples.upper 0 [97, 66, 49, 122] oracle) :=
  mkVmReadyState_rel Sporos.Examples.upper 0 [97, 66, 49, 122] oracle upper_fit (by decide) (readyEndAddr_le_of_eq (by decide))

theorem fib_ready_rel (oracle : List X86Env.Choice) :
    VmRel Sporos.Examples.fib (SporosProvLen.linit Sporos.Examples.fib 0 [])
      (mkVmReadyState Sporos.Examples.fib 0 [] oracle) :=
  mkVmReadyState_rel Sporos.Examples.fib 0 [] oracle fib_fit (by decide) (readyEndAddr_le_of_eq (by decide))

theorem provDemo_ready_rel (inp : List Nat) (hinp : inp.length ≤ 4096)
    (oracle : List X86Env.Choice) :
    VmRel SporosProvLen.provDemo (SporosProvLen.linit SporosProvLen.provDemo 0 inp)
      (mkVmReadyState SporosProvLen.provDemo 0 inp oracle) := by
  apply mkVmReadyState_rel SporosProvLen.provDemo 0 inp oracle provDemo_fit (by decide)
  apply readyEndAddr_le_of_eq
  have h1 : X86Env.pages (max ((flatProgQwords SporosProvLen.provDemo).length * 8) 1) = 4096 := by decide
  have h2 : readyLocPages SporosProvLen.provDemo = 4096 := by decide
  have h3 : readyMemwLen 0 = 4096 := by decide
  have h4 : readyMempLen 0 = 4096 := by decide
  have h5 : readyInpPages inp ≤ 4096 := by
    unfold readyInpPages X86Env.pages
    have hp : X86Env.page = 4096 := rfl
    rw [hp]
    omega
  rw [h1, h2, h3, h4]
  omega

/-! ### Concrete end-to-end x86-64 `sporos_vm` execution witnesses -/

def helloBytes : List Nat := "Hello from Sporos!\n".toList.map Char.toNat

def fibBytes : List Nat :=
  "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n".toList.map Char.toNat

private def lresHaltEq (r : SporosProvLen.LRes) (c : Nat) (stdout : List Nat) : Bool :=
  match r with
  | .halt rc out => rc == c && out.reverse == stdout
  | _ => false

private def lresFailEq (r : SporosProvLen.LRes) (e : Sporos.Err) : Bool :=
  match r with
  | .fail err => err == e
  | _ => false

private theorem eq_halt_of_lresHaltEq {r : SporosProvLen.LRes} {c : Nat} {stdout : List Nat}
    (h : lresHaltEq r c stdout = true) : r = .halt c stdout.reverse := by
  cases r with
  | next _ => simp [lresHaltEq] at h
  | fail _ => simp [lresHaltEq] at h
  | halt rc out =>
    simp only [lresHaltEq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, ho⟩ := h
    rw [← ho, List.reverse_reverse]

private theorem eq_fail_of_lresFailEq {r : SporosProvLen.LRes} {e : Sporos.Err}
    (h : lresFailEq r e = true) : r = .fail e := by
  cases r with
  | next _ => simp [lresFailEq] at h
  | halt _ _ => simp [lresFailEq] at h
  | fail err =>
    simp only [lresFailEq, beq_iff_eq] at h
    rw [h]

private theorem lrun_echo_hi :
    SporosProvLen.lrun Sporos.echo 40 (SporosProvLen.linit Sporos.echo 0 [104, 105]) =
      .halt 0 [104, 105].reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_hello :
    SporosProvLen.lrun Sporos.Examples.hello 300 (SporosProvLen.linit Sporos.Examples.hello 64 []) =
      .halt 0 helloBytes.reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_upper :
    SporosProvLen.lrun Sporos.Examples.upper 100
      (SporosProvLen.linit Sporos.Examples.upper 0 [97, 66, 49, 122]) =
      .halt 0 [65, 66, 49, 90].reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_fib :
    SporosProvLen.lrun Sporos.Examples.fib 2000 (SporosProvLen.linit Sporos.Examples.fib 0 []) =
      .halt 0 fibBytes.reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_provDemo_nonzero :
    SporosProvLen.lrun SporosProvLen.provDemo 30
      (SporosProvLen.linit SporosProvLen.provDemo 0 [65, 0]) =
      .halt 0 [0, 14, 66].reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_provDemo_zero :
    SporosProvLen.lrun SporosProvLen.provDemo 30
      (SporosProvLen.linit SporosProvLen.provDemo 0 [0, 0]) =
      .halt 0 [0, 6, 0].reverse :=
  eq_halt_of_lresHaltEq (by decide +kernel)

private theorem lrun_provDemo_demand :
    SporosProvLen.lrun SporosProvLen.provDemo 30
      (SporosProvLen.linit SporosProvLen.provDemo 0 [65, 1]) =
      .fail .demand :=
  eq_fail_of_lresFailEq (by decide +kernel)

/-- End-to-end witness: `sporos_vm` runs `echo` on `"hi"`, prints `"hi"`, and exits 0. -/
theorem witness_vm_echo (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState Sporos.echo 0 [104, 105] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState Sporos.echo 0 [104, 105] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = [104, 105] :=
  vm_halt_of_ready_witness oracle echo_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_echo_hi hor hno

/-- End-to-end witness: `sporos_vm` runs `hello`, prints `"Hello from Sporos!\n"`, and exits 0. -/
theorem witness_vm_hello (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState Sporos.Examples.hello 64 [] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState Sporos.Examples.hello 64 [] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = helloBytes :=
  vm_halt_of_ready_witness oracle hello_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_hello hor hno

/-- End-to-end witness: `sporos_vm` runs `upper` on `"aB1z"`, prints `"AB1Z"`, and exits 0. -/
theorem witness_vm_upper (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState Sporos.Examples.upper 0 [97, 66, 49, 122] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState Sporos.Examples.upper 0 [97, 66, 49, 122] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = [65, 66, 49, 90] :=
  vm_halt_of_ready_witness oracle upper_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_upper hor hno

/-- End-to-end witness: `sporos_vm` runs `fib`, prints the first 20 Fibonacci numbers, and exits 0. -/
theorem witness_vm_fib (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState Sporos.Examples.fib 0 [] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState Sporos.Examples.fib 0 [] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = fibBytes :=
  vm_halt_of_ready_witness oracle fib_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_fib hor hno

/-- End-to-end provenance witness: `provDemo` on `[65, 0]` emits `[0, 14, 66]` and exits 0. -/
theorem witness_vm_provDemo_nonzero (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState SporosProvLen.provDemo 0 [65, 0] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState SporosProvLen.provDemo 0 [65, 0] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = [0, 14, 66] :=
  vm_halt_of_ready_witness oracle provDemo_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_provDemo_nonzero hor hno

/-- End-to-end provenance witness: `provDemo` on `[0, 0]` emits `[0, 6, 0]` and exits 0. -/
theorem witness_vm_provDemo_zero (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState SporosProvLen.provDemo 0 [0, 0] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState SporosProvLen.provDemo 0 [0, 0] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 0 ∧
      Ends vmImage s₀ (.exit 0) ∧ s_exit.k.out = [0, 6, 0] :=
  vm_halt_of_ready_witness oracle provDemo_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_provDemo_zero hor hno

/-- End-to-end provenance-demand failure witness: `provDemo` on `[65, 1]` exits 1 cleanly. -/
theorem witness_vm_provDemo_demand_fail (oracle : List X86Env.Choice)
    (hor : OracleAdmits (mkVmReadyState SporosProvLen.provDemo 0 [65, 1] oracle))
    (hno : ∀ c ∈ oracle, c ≠ .nomem) :
    let s₀ := mkVmReadyState SporosProvLen.provDemo 0 [65, 1] oracle
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧
      Ends vmImage s₀ (.exit 1) ∧ s_exit.k.out = [] :=
  vm_fail_of_ready_witness oracle provDemo_fit (by decide) (readyEndAddr_le_of_eq (by decide)) lrun_provDemo_demand hor hno

/-! ### Ground TMR dispatch voter witnesses (single-replica masking + 3-way exit 3) -/

private def mkVoteTestState (a b c : Nat) : X86Vm.St :=
  let s0 := mkVmReadyState Sporos.echo 0 [] []
  { ((s0.set 0 (some a)).set 3 (some b)).set 1 (some c) with rip := SporosVmBin.L_vm_vote }

private theorem mkVoteTestState_regsOk (a b c : Nat) : RegsOk (mkVoteTestState a b c) := by
  have hr0 : RegsOk (mkVmReadyState Sporos.echo 0 [] []) :=
    (mkVmReadyCtx_wf Sporos.echo 0 [] [] echo_fit (by decide) (readyEndAddr_le_of_eq (by decide))).2.2.2.1
  exact regsOk_set (regsOk_set (regsOk_set hr0 (by decide) a) (by decide) b) (by decide) c

/-- Ground TMR masking witness: replica 2 disturbed (`0x55` XOR) still votes `42` into `L_vm_dispatch`. -/
theorem witness_tmr_vote_single_disturbance :
    let s := mkVoteTestState 42 (Nat.xor 42 0x55) 42
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      s'.reg 0 = some 42 ∧ s'.reg 1 = some 42 ∧ s'.reg 3 = some 42 := by
  obtain ⟨s', hrun, hrip, hmem, hk, _, hr0, hr1, hr3, _⟩ :=
    L_vm_vote_majority_spec (s := mkVoteTestState 42 (Nat.xor 42 0x55) 42)
      (a := 42) (b := Nat.xor 42 0x55) (c := 42) (v := 42)
      rfl (mkVoteTestState_regsOk _ _ _) rfl rfl rfl
      (by decide) (by decide) (by decide) (Or.inr (Or.inl ⟨rfl, rfl⟩))
  exact ⟨s', ⟨_, hrun⟩, hrip, hmem, hk, hr0, hr1, hr3⟩

/-- Ground TMR trap witness: 3-way replica disagreement (`42`, `99`, `7`) exits with status 3. -/
theorem witness_tmr_vote_three_way_exit3 :
    Ends vmImage (mkVoteTestState 42 99 7) (.exit 3) :=
  L_vm_vote_disagree_spec (s := mkVoteTestState 42 99 7) (a := 42) (b := 99) (c := 7)
    rfl (mkVoteTestState_regsOk _ _ _) rfl rfl rfl
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

end SporosVmWitness
