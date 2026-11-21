/-
SporosCheckArm64Bin: top-level machine-checked correctness, completeness,
refusal soundness, non-vacuity witnesses, and unconditional 4-channel
single-disturbance immunity of the committed `47,636`-byte static
`EM_AARCH64` ELF binary `bootstrap/model/spr1_check_arm64` over
`AArch64Phys` / `AArch64Env`.

Note on trust-base framing:
  Root A is MUSE's hand-written stage-1 checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
  `spr1_check_arm64` is an independent second check of stage 1 over AArch64,
  NOT Root A or Root A's trust base.
-/
import Bootstrap.SporosCheckArm64BinInit
import Bootstrap.SporosCheckArm64BinFault

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-! ### Top-level ELF-entry execution outcome -/

private theorem ends_exit_unique {s : St} {c1 c2 n : Nat}
    (he : Ends s (· = .exit c1)) (hr : run im n (.ok s) = .exit c2) : c1 = c2 := by
  obtain ⟨m, hm⟩ := he
  exact run_exit_unique hm hr

private theorem ends_stuck_ne_exit {s : St} {c n : Nat}
    (he : Ends s (· = .stuck)) (hr : run im n (.ok s) = .exit c) : False := by
  obtain ⟨m, hm⟩ := he
  by_cases hle : m ≤ n
  · have h := congrArg (run im (n - m)) hm
    rw [← run_add, show m + (n - m) = n by omega, hr, run_stuck] at h; cases h
  · have h := congrArg (run im (m - n)) hr
    rw [← run_add, show n + (m - n) = m by omega, hm, run_exit] at h; cases h

theorem spr1_arm64_bin_outcome {bs : List Nat} {o : List Choice}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o) :
    Ends (initState bs o) (fun r => r = .stuck ∨ r = .exit 6 ∨ r = .exit (expectedExit bs)) := by
  rcases init_spec hby ho with hst | hnm | htr
  · obtain ⟨n, hn⟩ := hst; exact ⟨n, Or.inl hn⟩
  · obtain ⟨n, hn⟩ := hnm; exact ⟨n, Or.inr (Or.inl hn)⟩
  · obtain ⟨n, hn⟩ := htr.ends fun _ ht => three_pass_ends ht
    exact ⟨n, Or.inr (Or.inr hn)⟩

/-! ### Soundness, `Sporos.step_safe` / `Sporos.run_safe` composition, and refusal soundness -/

/-- If `spr1_check_arm64` exits `0` from ELF entry `initState bs o`, then `bs` decodes
to a valid `Sporos` program `pr` passing `Sporos.checkProg pr = true`. -/
theorem spr1_arm64_bin_sound {bs : List Nat} {o : List Choice} {n : Nat}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o)
    (h0 : run im n (.ok (initState bs o)) = .exit 0) :
    ∃ pr, Sporos.decode bs = some pr ∧ Sporos.checkProg pr = true := by
  rcases init_spec hby ho with hst | hnm | htr
  · exact (ends_stuck_ne_exit hst h0).elim
  · exact absurd (ends_exit_unique hnm h0) (by decide)
  · have hex := ends_exit_unique (htr.ends fun _ ht => three_pass_ends ht) h0
    unfold expectedExit at hex
    cases hcf : checkFlat bs with
    | none => rw [hcf] at hex; cases hex
    | some ok =>
      cases ok
      · rw [hcf] at hex; cases hex
      · exact checkFlat_sound hcf

/-- Accepting exit `0` from `spr1_check_arm64` guarantees single-step safety of the decoded
`Sporos` program (`Sporos.step_safe`). -/
theorem spr1_arm64_bin_safe {bs : List Nat} {o : List Choice} {n : Nat}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o)
    (h0 : run im n (.ok (initState bs o)) = .exit 0) (ph : Sporos.Phys) :
    ∃ pr, Sporos.decode bs = some pr ∧
      ∀ (s : Sporos.St), Sporos.Inv pr s → Sporos.Safe pr (Sporos.step pr ph s) := by
  obtain ⟨pr, hdec, hchk⟩ := spr1_arm64_bin_sound hby ho h0
  exact ⟨pr, hdec, fun s hi => Sporos.step_safe pr ph hchk s hi⟩

/-- Accepting exit `0` from `spr1_check_arm64` guarantees multi-step run safety of the
decoded `Sporos` program (`Sporos.run_safe`): no static error can occur. -/
theorem spr1_arm64_bin_run_safe {bs : List Nat} {o : List Choice} {n : Nat}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o)
    (h0 : run im n (.ok (initState bs o)) = .exit 0)
    (ph : Sporos.Phys) (msize : Nat) (inp : List Nat) (m : Nat) :
    ∃ pr, Sporos.decode bs = some pr ∧
      ∀ e, Sporos.run pr ph m (Sporos.init pr msize inp) = .fail e → e.static = false := by
  obtain ⟨pr, hdec, hchk⟩ := spr1_arm64_bin_sound hby ho h0
  exact ⟨pr, hdec, fun e hr => Sporos.run_safe pr ph hchk msize inp m e hr⟩

/-- If `spr1_check_arm64` exits `5` (reject) from ELF entry `initState bs o`, then
no `Sporos` program decoding from `bs` can satisfy `Sporos.checkProg pr = true`. -/
theorem spr1_arm64_bin_refuse_sound {bs : List Nat} {o : List Choice} {n : Nat}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o)
    (h5 : run im n (.ok (initState bs o)) = .exit 5) :
    ∀ pr, Sporos.decode bs = some pr → Sporos.checkProg pr = false := by
  rcases init_spec hby ho with hst | hnm | htr
  · exact (ends_stuck_ne_exit hst h5).elim
  · exact absurd (ends_exit_unique hnm h5) (by decide)
  · have hex := ends_exit_unique (htr.ends fun _ ht => three_pass_ends ht) h5
    unfold expectedExit at hex
    cases hcf : checkFlat bs with
    | none => rw [hcf] at hex; cases hex
    | some ok =>
      cases ok
      · exact checkFlat_refuse_sound hcf
      · rw [hcf] at hex; cases hex

/-! ### Completeness and `TripledState` non-vacuity -/

/-- Non-vacuity witness for `TripledState`: for every valid-byte stream `bs` with
`bs.length < 2 ^ 45`, `goodOracle bs` drives ELF entry `initState bs (goodOracle bs)`
deterministically into a `TripledState bs`. -/
theorem mkTripledState_exists {bs : List Nat} (hby : ∀ b ∈ bs, b < 256) (hlen : bs.length < 2 ^ 45) :
    ∃ s, TripledState bs s := by
  obtain ⟨_, s, _, ht⟩ := init_spec_complete hby hlen
  exact ⟨s, ht⟩

theorem spr1_arm64_bin_complete_of_checkFlat {bs : List Nat}
    (hby : ∀ b ∈ bs, b < 256) (hlen : bs.length < 2 ^ 45) :
    OracleWf (goodOracle bs) ∧
    Ends (initState bs (goodOracle bs)) (· = .exit (expectedExit bs)) :=
  ⟨goodOracle_wf bs, (init_spec_complete hby hlen).ends fun _ ht => three_pass_ends ht⟩

/-- Completeness: any `Sporos` program `pr` encoded as `bs` with `Sporos.checkProg pr = true`,
`Fits48 bs pr`, and `bs.length < 2 ^ 45` is accepted (`exit 0`) by `spr1_check_arm64`
under `goodOracle bs`. -/
theorem spr1_arm64_bin_complete {bs : List Nat} {pr : Sporos.Prog}
    (hdec : Sporos.decode bs = some pr) (hchk : Sporos.checkProg pr = true)
    (hfit : Fits48 bs pr) (hlen : bs.length < 2 ^ 45) :
    ∃ o, OracleWf o ∧ Ends (initState bs o) (· = .exit 0) := by
  have hcf := checkFlat_complete hdec hchk hfit
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat hfit.1 hlen
  exact ⟨goodOracle bs, hwf, by simpa [expectedExit, hcf] using hend⟩

/-! ### Ground ELF-entry witnesses (4 canonical examples, refusals, `HMAX` overflow, `-ENOMEM`) -/

private theorem bytes_of_checkFlat_some {bs : List Nat} {b : Bool} (h : checkFlat bs = some b) :
    ∀ x ∈ bs, x < 256 := by
  unfold checkFlat checkFlatRes at h
  by_cases hall : bs.all (· < 256) = true
  · exact fun x hx => of_decide_eq_true (List.all_eq_true.1 hall x hx)
  · simp [hall] at h

theorem spr1_arm64_bin_echo_accept :
    ∃ o, OracleWf o ∧ Ends (initState Sporos.echo.enc o) (· = .exit 0) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_echo_valid) (by decide +kernel)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_echo_valid] using hend⟩

theorem spr1_arm64_bin_hello_accept :
    ∃ o, OracleWf o ∧ Ends (initState Sporos.Examples.hello.enc o) (· = .exit 0) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_hello_valid) (by decide +kernel)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_hello_valid] using hend⟩

theorem spr1_arm64_bin_upper_accept :
    ∃ o, OracleWf o ∧ Ends (initState Sporos.Examples.upper.enc o) (· = .exit 0) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_upper_valid) (by decide +kernel)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_upper_valid] using hend⟩

theorem spr1_arm64_bin_fib_accept :
    ∃ o, OracleWf o ∧ Ends (initState Sporos.Examples.fib.enc o) (· = .exit 0) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_fib_valid) (by decide +kernel)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_fib_valid] using hend⟩

theorem spr1_arm64_bin_empty_refuse :
    ∃ o, OracleWf o ∧ Ends (initState [] o) (· = .exit 5) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat (bs := []) (fun _ h => by cases h) (by decide)
  exact ⟨goodOracle [], hwf, by simpa [expectedExit, checkFlat_empty_refused] using hend⟩

theorem spr1_arm64_bin_badMagic_refuse :
    ∃ o, OracleWf o ∧ Ends (initState [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] o) (· = .exit 5) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_badMagic_refused) (by decide)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_badMagic_refused] using hend⟩

theorem spr1_arm64_bin_underflow_refuse :
    ∃ o, OracleWf o ∧
      Ends (initState (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.prim .add, .ret], ann := [] }], data := [], main := 0 }) o)
        (· = .exit 5) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_underflow_refused) (by decide +kernel)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_underflow_refused] using hend⟩

theorem spr1_arm64_bin_badOpcode_refuse :
    ∃ o, OracleWf o ∧
      Ends (initState
        [ 0x53, 0x50, 0x52, 0x31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0xff ] o)
        (· = .exit 5) := by
  have ⟨hwf, hend⟩ := spr1_arm64_bin_complete_of_checkFlat
    (bytes_of_checkFlat_some checkFlat_badOpcode_refused) (by decide)
  exact ⟨goodOracle _, hwf, by simpa [expectedExit, checkFlat_badOpcode_refused] using hend⟩

/-- Non-vacuous unconditional 4-channel single-disturbance witness: the empty invalid image
reaches a concrete `TripledState [] s0` on which `spr1_arm64_bin_no_single_fault_false_accept`
holds across all 4 channels (`fetch`, `reg`, `mem`, `nzcv`). -/
theorem spr1_arm64_bin_fault_witness_nonvacuous :
    ∃ s0, TripledState [] s0 ∧
      (∀ n o', OneFaultRun alts im n (.ok s0) o' → o' ≠ .exit 0) ∧
      (∀ (r : Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (Out × Bool) × Rat),
        y ∈ prunWithFaultAt alts r hr ch im (some fi) n (.ok s0) → 0 < y.2 → y.1.1 ≠ .exit 0) := by
  obtain ⟨s0, ht⟩ := mkTripledState_exists (bs := []) (fun _ h => by cases h) (by decide)
  have hne : ∀ p, Sporos.decode [] = some p → Sporos.checkProg p ≠ true := fun p hd => by
    rw [checkFlat_refuse_sound checkFlat_empty_refused p hd]; decide
  obtain ⟨hdet, hprob⟩ := spr1_arm64_bin_no_single_fault_false_accept ht hne
  exact ⟨s0, ht, hdet, hprob⟩

/-- Explicit `-ENOMEM` witness from ELF entry: if `mmap` returns `-ENOMEM`,
`spr1_check_arm64` deterministically exits `6`. -/
theorem spr1_arm64_bin_nomem_witness :
    OracleWf [.nomem] ∧ Ends (initState [] [.nomem]) (· = .exit 6) :=
  ⟨fun c hc => by simp only [List.mem_singleton] at hc; subst hc; trivial, init_nomem_exit6 []⟩

/-- Valid-byte `HMAX` overflow witness: an invalid-height input with `checkFlat bs = none`
causes every post-triplication `TripledState bs` of `spr1_check_arm64` to exit `6`. -/
theorem spr1_arm64_bin_hmax_overflow_witness :
    ∃ bs, (∀ b ∈ bs, b < 256) ∧ bs.length < SporosCheckArm64Flat.W ∧ checkFlat bs = none ∧
      ∀ s, TripledState bs s → Ends s (· = .exit 6) := by
  obtain ⟨bs, hby, hlen, hnone⟩ := checkFlat_hmax_overflow_witness
  exact ⟨bs, hby, hlen, hnone, fun s ht => by
    simpa [expectedExit, hnone] using three_pass_ends ht⟩

end SporosCheckArm64Bin
