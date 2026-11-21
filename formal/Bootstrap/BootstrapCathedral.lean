import Sporos.Bytecode
import Sporos.Check
import Sporos.Run
import Sporos.DagGC
import Sporos.DagHash
import Sporos.Examples
import Bootstrap.Formats
import Bootstrap.FormatsC
import Bootstrap.SeedA
import Bootstrap.SeedB
import Bootstrap.SeedC
import Bootstrap.SporosX1Image
import Bootstrap.X86Bridge
import Bootstrap.SporosProvLen
import Bootstrap.SporosCheckArm64Bin
import Bootstrap.SporosVmBin
import Bootstrap.SporosX1Final

/-!
# BootstrapCathedral: End-to-End Multi-ISA Verified Bootstrapping Unification

Unifies the four verified pillars of the Sounio / Sporos self-hosting cathedral
under `leanprover/lean4:v4.33.0`:

1. **Pillar 1 — Tri-Root Multi-ISA Seed Genesis**:
   Cross-ISA grammar agreement and uniqueness across x86-64 `seed-a` (Format A hex),
   x86-64 `seed-b` (Format B octal), and AArch64 `seed-c` (Format C binary), together
   with kernel-checked byte/decode identity links for the committed ELF artifacts
   (`seed-a`, `seed-b`, `seed-c`, Root A `sporos-stage1`, `sporos_vm`, `spr1_check_arm64`).

2. **Pillar 2 — AArch64 4-Channel Fault-Immune Admission -> Semantic Safety ->
   Provenance Hash-DAG & 64-Bit Length Bisimulation -> Native x86-64 TMR Execution**:
   For any raw byte list `f`, acceptance by `spr1_check_arm64` yields decoded `pr`
   with `Sporos.checkProg pr = true`, static-error freedom (`Sporos.run_safe` /
   `SporosProvLen.lrun_safe`), provenance bisimulation (`SporosProvLen.lrun_sim_init`,
   `Sporos.drun_init_hashcons`, `Sporos.drunC_init`), and 12,945-byte native x86-64
   `sporos_vm` simulation + inline TMR dispatch voter bounds; while any rejected `f`
   exhibits a concrete `TripledState` immune to any single-disturbance exit-0 false
   accept across all 4 physical channels (`fetch`, `reg`, `mem`, `nzcv`).

3. **Pillar 3 — Ground Non-Vacuity & Fault-Control Witnesses**:
   End-to-end concrete kernel-checked witnesses on `echo`, `fib`, `provDemo`,
   invalid/corrupted byte slices, AArch64 4-channel fault non-vacuity, and x86-64
   TMR majority masking / 3-way disagreement traps.

4. **Pillar 4 — Dual-ISA Stage-1 Equivalence & Root A End-to-End Cathedral**:
   Cross-ISA extensional equivalence between Root A's hand-written 2,447-byte x86-64
   `stage1Im` (bounded within cubic `stage1Bound f`, with `-ENOMEM` exit-2 totality and
   admitting oracle existence) and AArch64's 47,636-byte TMR-hardened `spr1_check_arm64`,
   composing Root A admission directly with semantic safety, provenance bisimulation,
   native x86-64 `sporos_vm` + TMR execution, and ground witnesses (`echo`, `fib`, empty,
   corrupted).
-/

namespace BootstrapCathedral

/-! ## Pillar 1: Tri-Root Multi-ISA Seed Genesis & Committed Binary Identity -/

/-- Whenever Format-A listing `LA`, Format-B listing `LB`, and Format-C listing `LC`
decode via `decA`, `decB`, and `decC` to the same byte stream `bs`, all three
independent ISA seeds (`x86-64` `seed-a`, `x86-64` `seed-b`, `AArch64` `seed-c`)
execute to `.exit 0 bs`, and `bs` is the unique byte stream any of them can emit. -/
theorem tri_root_seed_genesis
    (LA LB LC bs : List Nat)
    (hA : ∀ c ∈ LA, c < 256)
    (hB : ∀ c ∈ LB, c < 256)
    (hC : ∀ c ∈ LC, c < 256)
    (hdA : X86Min.Formats.decA LA false none [] = some bs)
    (hdB : X86Min.Formats.decB LB false 0 0 [] = some bs)
    (hdC : Bootstrap.FormatsC.decC LC false 0 0 [] = some bs) :
    X86Min.Formats.HexA LA none bs ∧
    X86Min.Formats.OctB LB bs ∧
    Bootstrap.FormatsC.BinC LC bs ∧
    (∃ FA, X86Min.exec FA X86Min.Data.seedA LA = .exit 0 bs) ∧
    (∀ FA bs', X86Min.exec FA X86Min.Data.seedA LA = .exit 0 bs' → bs' = bs) ∧
    (∃ FB, X86Min.exec FB X86Min.Data.seedB LB = .exit 0 bs) ∧
    (∀ FB bs', X86Min.exec FB X86Min.Data.seedB LB = .exit 0 bs' → bs' = bs) ∧
    (∃ FC, AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 0 bs) ∧
    (∀ FC bs', AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 0 bs' → bs' = bs) := by
  have hgA : X86Min.Formats.HexA LA none bs := (X86Min.Formats.decA_iff LA bs).mp hdA
  have hgB : X86Min.Formats.OctB LB bs := (X86Min.Formats.decB_iff LB bs).mp hdB
  have hgC : Bootstrap.FormatsC.BinC LC bs := (Bootstrap.FormatsC.decC_iff LC bs).mp hdC
  have heA : ∃ FA, X86Min.exec FA X86Min.Data.seedA LA = .exit 0 bs :=
    (X86Min.SeedA.seedA_correct LA bs hA).mpr hgA
  have heB : ∃ FB, X86Min.exec FB X86Min.Data.seedB LB = .exit 0 bs :=
    (X86Min.SeedB.seedB_correct LB bs hB).mpr hgB
  have heC : ∃ FC, AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 0 bs :=
    (AArch64Min.SeedC.seedC_correct LC bs hC).mpr hgC
  refine ⟨hgA, hgB, hgC, heA, ?_, heB, ?_, heC, ?_⟩
  · intro FA bs' hrun
    have hgA' : X86Min.Formats.HexA LA none bs' :=
      (X86Min.SeedA.seedA_correct LA bs' hA).mp ⟨FA, hrun⟩
    exact X86Min.Formats.hexA_unique hgA' hgA
  · intro FB bs' hrun
    have hgB' : X86Min.Formats.OctB LB bs' :=
      (X86Min.SeedB.seedB_correct LB bs' hB).mp ⟨FB, hrun⟩
    exact X86Min.Formats.octB_unique hgB' hgB
  · intro FC bs' hrun
    have hgC' : Bootstrap.FormatsC.BinC LC bs' :=
      (AArch64Min.SeedC.seedC_correct LC bs' hC).mp ⟨FC, hrun⟩
    exact Bootstrap.FormatsC.binC_unique hgC' hgC

/-- Bidirectional cross-ISA equivalence between x86-64 `seed-a`, x86-64 `seed-b`,
and AArch64 `seed-c` executions and their three independent grammar specifications. -/
theorem tri_root_cross_isa_iff
    (LA LB LC bs : List Nat)
    (hA : ∀ c ∈ LA, c < 256)
    (hB : ∀ c ∈ LB, c < 256)
    (hC : ∀ c ∈ LC, c < 256) :
    ((∃ FA, X86Min.exec FA X86Min.Data.seedA LA = .exit 0 bs) ∧
     (∃ FB, X86Min.exec FB X86Min.Data.seedB LB = .exit 0 bs) ∧
     (∃ FC, AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 0 bs)) ↔
    (X86Min.Formats.HexA LA none bs ∧
     X86Min.Formats.OctB LB bs ∧
     Bootstrap.FormatsC.BinC LC bs) := by
  rw [X86Min.SeedA.seedA_correct LA bs hA,
      X86Min.SeedB.seedB_correct LB bs hB,
      AArch64Min.SeedC.seedC_correct LC bs hC]

/-- If three valid listings `LA`, `LB`, `LC` across Format A, Format B, and Format C
decode via `decA`/`decB`/`decC` to identical target bytes, then any `.exit 0` outputs
emitted by `seed-a` (x86-64), `seed-b` (x86-64), and `seed-c` (AArch64) are identical. -/
theorem tri_root_cross_isa_output_agreement
    (LA LB LC bs bsA bsB bsC : List Nat)
    (hA : ∀ c ∈ LA, c < 256)
    (hB : ∀ c ∈ LB, c < 256)
    (hC : ∀ c ∈ LC, c < 256)
    (hdA : X86Min.Formats.decA LA false none [] = some bs)
    (hdB : X86Min.Formats.decB LB false 0 0 [] = some bs)
    (hdC : Bootstrap.FormatsC.decC LC false 0 0 [] = some bs)
    {FA FB FC : Nat}
    (hrunA : X86Min.exec FA X86Min.Data.seedA LA = .exit 0 bsA)
    (hrunB : X86Min.exec FB X86Min.Data.seedB LB = .exit 0 bsB)
    (hrunC : AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 0 bsC) :
    bsA = bs ∧ bsB = bs ∧ bsC = bs ∧ bsA = bsB ∧ bsB = bsC := by
  have hg := tri_root_seed_genesis LA LB LC bs hA hB hC hdA hdB hdC
  have eA : bsA = bs := hg.2.2.2.2.1 FA bsA hrunA
  have eB : bsB = bs := hg.2.2.2.2.2.2.1 FB bsB hrunB
  have eC : bsC = bs := hg.2.2.2.2.2.2.2.2 FC bsC hrunC
  exact ⟨eA, eB, eC, eA.trans eB.symm, eB.trans eC.symm⟩

/-- Refusal symmetry across all three roots: malformed inputs deterministically exit `1`
and can never emit `.exit 0` under any fuel bound. -/
theorem tri_root_cross_isa_refuses
    (LA LB LC : List Nat)
    (hA : ∀ c ∈ LA, c < 256)
    (hB : ∀ c ∈ LB, c < 256)
    (hC : ∀ c ∈ LC, c < 256)
    (hiA : ∀ bs, ¬ X86Min.Formats.HexA LA none bs)
    (hiB : ∀ bs, ¬ X86Min.Formats.OctB LB bs)
    (hiC : ∀ bs, ¬ Bootstrap.FormatsC.BinC LC bs) :
    (∃ FA x, X86Min.exec FA X86Min.Data.seedA LA = .exit 1 x) ∧
    (∀ FA bs, X86Min.exec FA X86Min.Data.seedA LA ≠ .exit 0 bs) ∧
    (∃ FB x, X86Min.exec FB X86Min.Data.seedB LB = .exit 1 x) ∧
    (∀ FB bs, X86Min.exec FB X86Min.Data.seedB LB ≠ .exit 0 bs) ∧
    (∃ FC x, AArch64Min.exec FC AArch64Min.SeedC.seedC LC = .exit 1 x) ∧
    (∀ FC bs, AArch64Min.exec FC AArch64Min.SeedC.seedC LC ≠ .exit 0 bs) := by
  refine ⟨X86Min.SeedA.seedA_refuses LA hA hiA, ?_,
          X86Min.SeedB.seedB_refuses LB hB hiB, ?_,
          AArch64Min.SeedC.seedC_refuses LC hC hiC, ?_⟩
  · intro FA bs h0
    exact hiA bs ((X86Min.SeedA.seedA_correct LA bs hA).mp ⟨FA, h0⟩)
  · intro FB bs h0
    exact hiB bs ((X86Min.SeedB.seedB_correct LB bs hB).mp ⟨FB, h0⟩)
  · intro FC bs h0
    exact hiC bs ((AArch64Min.SeedC.seedC_correct LC bs hC).mp ⟨FC, h0⟩)

private theorem seedAHex_decA :
    X86Min.Formats.decA X86Min.Data.seedAHex false none [] = some X86Min.Data.seedA := by
  decide +kernel

private theorem seedAOct_decB :
    X86Min.Formats.decB X86Min.Data.seedAOct false 0 0 [] = some X86Min.Data.seedA := by
  decide +kernel

private theorem seedBHex_decA :
    X86Min.Formats.decA X86Min.Data.seedBHex false none [] = some X86Min.Data.seedB := by
  decide +kernel

private theorem seedBOct_decB :
    X86Min.Formats.decB X86Min.Data.seedBOct false 0 0 [] = some X86Min.Data.seedB := by
  decide +kernel

private theorem seedAHex_bytes : ∀ c ∈ X86Min.Data.seedAHex, c < 256 := by
  have h : X86Min.Data.seedAHex.all (· < 256) = true := by decide +kernel
  exact fun c hc => of_decide_eq_true (List.all_eq_true.mp h c hc)

private theorem seedAOct_bytes : ∀ c ∈ X86Min.Data.seedAOct, c < 256 := by
  have h : X86Min.Data.seedAOct.all (· < 256) = true := by decide +kernel
  exact fun c hc => of_decide_eq_true (List.all_eq_true.mp h c hc)

private theorem seedBHex_bytes : ∀ c ∈ X86Min.Data.seedBHex, c < 256 := by
  have h : X86Min.Data.seedBHex.all (· < 256) = true := by decide +kernel
  exact fun c hc => of_decide_eq_true (List.all_eq_true.mp h c hc)

private theorem seedBOct_bytes : ∀ c ∈ X86Min.Data.seedBOct, c < 256 := by
  have h : X86Min.Data.seedBOct.all (· < 256) = true := by decide +kernel
  exact fun c hc => of_decide_eq_true (List.all_eq_true.mp h c hc)

private theorem stage1HexChunks_bytes :
    (SporosX1.stage1HexChunks.all fun ch => ch.all (· < 256)) = true := by
  decide +kernel

private theorem sporosStage1Hex_bytes : ∀ c ∈ X86Min.Data.sporosStage1Hex, c < 256 := by
  intro c hc
  rw [SporosX1.hex_flatten] at hc
  obtain ⟨ch, hch_mem, hc_mem⟩ := List.mem_flatten.mp hc
  have hch := List.all_eq_true.mp stage1HexChunks_bytes ch hch_mem
  exact of_decide_eq_true (List.all_eq_true.mp hch c hc_mem)

/-- Kernel-checked byte, decode, and ELF load/table identities linking all committed
bootstrap artifacts (`seed-a`, `seed-b`, `seed-c`, Root A `sporos-stage1`, `spr1_check_arm64`,
and `sporos_vm`). -/
theorem committed_binaries_identity_links (alts : X86Int.Alts) (r : X86Int.Rates) (hr : r.ok) :
    (∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.seedAHex = .exit 0 X86Min.Data.seedA) ∧
    (∃ F, X86Min.exec F X86Min.Data.seedB X86Min.Data.seedAOct = .exit 0 X86Min.Data.seedA) ∧
    (∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.seedBHex = .exit 0 X86Min.Data.seedB) ∧
    (∃ F, X86Min.exec F X86Min.Data.seedB X86Min.Data.seedBOct = .exit 0 X86Min.Data.seedB) ∧
    (∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.sporosStage1Hex = .exit 0 SporosX1.stage1Code) ∧
    SporosX1.stage1Code.length = 2447 ∧
    SporosX1.phFilesz = SporosX1.stage1Code.length ∧
    SporosX1.phMemsz = SporosX1.stage1Code.length ∧
    AArch64Min.SeedC.seedC.length = 340 ∧
    (∀ L, AArch64Min.load AArch64Min.SeedC.seedC 0x7ffffffde000 L =
      some (AArch64Min.SeedC.imC, { AArch64Min.SeedC.s0 with inp := L })) ∧
    SporosCheckArm64Bin.spr1Arm64Bytes.length = 47636 ∧
    (∀ e ∈ SporosCheckArm64Bin.spr1Arm64Table, SporosCheckArm64Bin.im e.1 = some e.2) ∧
    SporosVmBin.vmBytes.length = 12945 ∧
    (∀ e ∈ SporosVmBin.vmTable, SporosVmCore.vmImage e.1 = some e.2) ∧
    (X86Min.exec 300 X86Bridge.loaderElf X86Bridge.stage1Code = .exit 0 X86Bridge.stage1Code ∧
      X86Int.run X86Bridge.loaderImInt 300 (.ok X86Bridge.loaderStInt) = .exit 0 ∧
      X86Int.run X86Bridge.stage1Im 10 (.ok X86Bridge.stage1StInt) = .exit 42) := by
  have hsAA : ∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.seedAHex = .exit 0 X86Min.Data.seedA :=
    (X86Min.SeedA.seedA_correct _ _ seedAHex_bytes).mpr
      ((X86Min.Formats.decA_iff _ _).mp seedAHex_decA)
  have hsBA : ∃ F, X86Min.exec F X86Min.Data.seedB X86Min.Data.seedAOct = .exit 0 X86Min.Data.seedA :=
    (X86Min.SeedB.seedB_correct _ _ seedAOct_bytes).mpr
      ((X86Min.Formats.decB_iff _ _).mp seedAOct_decB)
  have hsAB : ∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.seedBHex = .exit 0 X86Min.Data.seedB :=
    (X86Min.SeedA.seedA_correct _ _ seedBHex_bytes).mpr
      ((X86Min.Formats.decA_iff _ _).mp seedBHex_decA)
  have hsBB : ∃ F, X86Min.exec F X86Min.Data.seedB X86Min.Data.seedBOct = .exit 0 X86Min.Data.seedB :=
    (X86Min.SeedB.seedB_correct _ _ seedBOct_bytes).mpr
      ((X86Min.Formats.decB_iff _ _).mp seedBOct_decB)
  have hs1 : ∃ F, X86Min.exec F X86Min.Data.seedA X86Min.Data.sporosStage1Hex = .exit 0 SporosX1.stage1Code :=
    (X86Min.SeedA.seedA_correct _ _ sporosStage1Hex_bytes).mpr
      ((X86Min.Formats.decA_iff _ _).mp SporosX1.stage1Code_decA)
  have hbr := X86Bridge.seed_to_stage1_bridge_witness alts r hr
  refine ⟨hsAA, hsBA, hsAB, hsBB, hs1,
    SporosX1.stage1Code_length, SporosX1.phFilesz_eq, SporosX1.phMemsz_eq,
    by decide +kernel, AArch64Min.SeedC.load_C,
    SporosCheckArm64Bin.spr1Arm64Bytes_len, SporosCheckArm64Bin.table_ok,
    SporosVmCore.vmBytes_len_eq, SporosVmBin.table_ok,
    ⟨hbr.1, hbr.2.1, hbr.2.2.2.2⟩⟩


/-! ## Pillar 2: AArch64 4-Channel Fault-Immune Admission -> Semantic Safety ->
Provenance Bisimulation -> Native x86-64 TMR `sporos_vm` Execution -/

/-- Shared post-admission semantic safety, provenance bisimulation, and native
12,945-byte x86-64 `sporos_vm` simulation theorem parameterized by `checkProg pr = true`. -/
private theorem checked_prog_safety_and_vm {pr : Sporos.Prog}
    (hchk : Sporos.checkProg pr = true) :
    (∀ (ph : Sporos.Phys) (msize : Nat) (inp : List Nat) (n : Nat),
      (∀ e, Sporos.run pr ph n (Sporos.init pr msize inp) = .fail e → e.static = false) ∧
      (∀ e, SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .fail e → e.static = false) ∧
      SporosProvLen.ResRel (Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp))
        (SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp)) ∧
      (Sporos.drun Sporos.Alloc.hashcons pr ph n (Sporos.dinit pr msize inp)).abs =
        Sporos.run pr ph n (Sporos.init pr msize inp) ∧
      (∀ (A : Sporos.Alloc) (pol : Sporos.DSt → Bool),
        (Sporos.drunC A pr ph pol n (Sporos.dinit pr msize inp)).abs =
          Sporos.run pr ph n (Sporos.init pr msize inp))) ∧
    (∀ (msize : Nat) (inp : List Nat) (oracle : List X86Env.Choice),
      SporosVmSimBase.FlatProgFit pr → msize < 2 ^ 44 →
      SporosVmReady.readyEndAddr pr msize inp ≤ 2 ^ 47 →
      let s₀ := SporosVmReady.mkVmReadyState pr msize inp oracle
      SporosVmSimBase.OracleAdmits s₀ → (∀ c ∈ oracle, c ≠ .nomem) →
      SporosVmExec.RegsOk s₀ ∧
      (∀ n,
        (match Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
        | .next st' =>
          ∃ ls' s', SporosProvLen.Rel st' ls' ∧
            SporosVmCore.Reaches SporosVmCore.vmImage s₀ s' ∧
            s'.k.out = s₀.k.out ∧
            SporosVmSimBase.OracleAdmits s' ∧
            SporosVmSimBase.VmRel pr ls' s'
        | .halt c out =>
          ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
            X86Vm.step SporosVmCore.vmImage s_exit = .exit (c % 256) ∧
            SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit (c % 256)) ∧
            s_exit.k.out = s₀.k.out ++ (out.map Prod.fst).reverse
        | .fail e =>
          (e = .bounds ∨ e = .divZero ∨ e = .badTable ∨ e = .demand) ∧
          ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
            X86Vm.step SporosVmCore.vmImage s_exit = .exit 1 ∧
            SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 1) ∧
            s_exit.k.out = s₀.k.out))) := by
  refine ⟨?_, ?_⟩
  · intro ph msize inp n
    exact ⟨fun e hf => Sporos.run_safe pr ph hchk msize inp n e hf,
      fun e hf => SporosProvLen.lrun_safe pr hchk msize inp n e hf,
      SporosProvLen.lrun_sim_init pr msize inp n,
      Sporos.drun_init_hashcons pr ph msize inp n,
      fun A pol => Sporos.drunC_init A pr ph pol msize inp n⟩
  · intro msize inp oracle hfit hmsize hend s₀ hor hno
    have hwf := SporosVmReady.mkVmReadyCtx_wf pr msize inp oracle hfit hmsize hend
    have hrel := SporosVmReady.mkVmReadyState_rel pr msize inp oracle hfit hmsize hend
    refine ⟨hwf.2.2.2.1, fun n => ?_⟩
    have hsim := SporosVmSim.sporos_vm_sim_noPhys_noNomem pr msize inp n hrel hor hno
    cases hrun : Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
    | next st' => rw [hrun] at hsim; exact hsim
    | halt c out => rw [hrun] at hsim; exact hsim
    | fail e =>
      rw [hrun] at hsim
      exact ⟨SporosVmSim.sporos_vm_sim_checked_no_static_fail pr hchk msize inp n e hrun, hsim⟩

/-- Shared inline TMR dispatch voter bounds for the committed 12,945-byte `sporos_vm` ELF. -/
private theorem vm_tmr_voter_safety :
    SporosVmBin.vmBytes.length = 12945 ∧
    (∀ e ∈ SporosVmBin.vmTable, SporosVmCore.vmImage e.1 = some e.2) ∧
    (∀ a ∈ SporosVmFault.allInlineVoterAddrs, SporosVmFault.InlineVoteAt a) ∧
    (∀ (C : TMR.Channel Nat) (r : X86Vm.Rates) (hr : r.ok) (x : Nat) (s : X86Vm.St),
      SporosVmExec.RegsOk s → (∀ y ∈ C.step x, y.1.1 < X86Env.W) →
      (SporosVmFault.vmDispatchVotedPhys C r hr x s).ok ∧
      PDist.fm (SporosVmFault.vmDispatchVotedPhys C r hr x s) ≤ 3 * (C.q * C.q) + 10 * r.total) ∧
    (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c v : Nat} {s : X86Vm.St},
      s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
      s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
      a < X86Env.W → b < X86Env.W → c < X86Env.W →
      ((a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) →
      (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
      PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 10 * r.total ∧
      ∃ s', PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.ok s'] ∧
        s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        SporosVmExec.RegsOk s' ∧ s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v) ∧
    (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c : Nat} {s : X86Vm.St},
      s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
      s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
      a < X86Env.W → b < X86Env.W → c < X86Env.W →
      a ≠ b → a ≠ c → b ≠ c →
      (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
      PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.exit 3] ∧
      PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 9 * r.total) := by
  refine ⟨SporosVmCore.vmBytes_len_eq, SporosVmBin.table_ok, SporosVmFault.all_inline_voters_ok,
    ?_, ?_, ?_⟩
  · intro C r hr x s hreg hW
    have hb := SporosVmFault.vm_dispatch_tri_phys_bound C r hr x s hreg hW
    exact ⟨hb.1, hb.2.2⟩
  · intro r hr a b c v s hrip hreg h0 h3 h1 ha hb hc hmaj
    exact SporosVmFault.vmVotePhys_majority r hr hrip hreg h0 h3 h1 ha hb hc hmaj
  · intro r hr a b c s hrip hreg h0 h3 h1 ha hb hc hab hac hbc
    exact SporosVmFault.vmVotePhys_disagree r hr hrip hreg h0 h3 h1 ha hb hc hab hac hbc

/-- Shared AArch64 4-channel single-disturbance immunity witness for rejected byte slices. -/
private theorem arm64_refused_single_fault_immune (f : List Nat)
    (hby : ∀ b ∈ f, b < 256) (hlen : f.length < 2 ^ 45)
    (hrej : ∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr ≠ true) :
    ∃ s0, SporosCheckArm64Bin.TripledState f s0 ∧
      (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
        o' ≠ .exit 0) ∧
      (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
        y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
          SporosCheckArm64Bin.im (some fi) n (.ok s0) →
        0 < y.2 → y.1.1 ≠ .exit 0) := by
  obtain ⟨s0, ht⟩ := SporosCheckArm64Bin.mkTripledState_exists hby hlen
  obtain ⟨h1, h2⟩ := SporosCheckArm64Bin.spr1_arm64_bin_no_single_fault_false_accept ht hrej
  exact ⟨s0, ht, h1, h2⟩

/-- Crown Theorem (`arm64_admit_to_x86vm_cathedral`):
For ANY raw byte slice `f : List Nat` (`b < 256`, `f.length < 2^45`):
1. Acceptance by the committed 47,636-byte AArch64 checker `spr1_check_arm64` yields a
   decoded `pr` satisfying `Sporos.checkProg pr = true`, static-error freedom across all
   physical oracles and provenance models (`Sporos.run_safe`, `SporosProvLen.lrun_safe`),
   64-bit provenance-length bisimulation (`SporosProvLen.lrun_sim_init`), hash-consed
   and GC-enabled provenance DAG bisimulation (`Sporos.drun_init_hashcons`, `Sporos.drunC_init`),
   and exact simulation by the committed 12,945-byte native x86-64 `sporos_vm` ELF from
   `mkVmReadyState` under `FlatProgFit pr` (`sporos_vm_sim_noPhys_noNomem`,
   `sporos_vm_sim_checked_no_static_fail`), together with inline TMR dispatch voter
   safety (`vmVotePhys_majority`, `vmVotePhys_disagree`, `vm_dispatch_tri_phys_bound`,
   `all_inline_voters_ok`);
2. Any invalid byte slice `f` (`∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr ≠ true`)
   constructively inhabits `TripledState f s0` and is unconditionally immune to any single
   physical disturbance across all 4 channels (`fetch`, `reg`, `mem`, `nzcv`) causing `.exit 0`;
3. Every well-typed `Fits48` program is deterministically accepted under `goodOracle f`. -/
theorem arm64_admit_to_x86vm_cathedral
    (f : List Nat) (hby : ∀ b ∈ f, b < 256) (hlen : f.length < 2 ^ 45) :
    -- (a1) Soundness + (b) Safety & Provenance + (c1) Native x86-64 `sporos_vm` simulation
    (∀ (o : List AArch64Env.Choice), AArch64Env.OracleWf o →
      SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0) →
      ∃ pr, Sporos.decode f = some pr ∧ Sporos.checkProg pr = true ∧
        (∀ (ph : Sporos.Phys) (msize : Nat) (inp : List Nat) (n : Nat),
          (∀ e, Sporos.run pr ph n (Sporos.init pr msize inp) = .fail e → e.static = false) ∧
          (∀ e, SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .fail e → e.static = false) ∧
          SporosProvLen.ResRel (Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp))
            (SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp)) ∧
          (Sporos.drun Sporos.Alloc.hashcons pr ph n (Sporos.dinit pr msize inp)).abs =
            Sporos.run pr ph n (Sporos.init pr msize inp) ∧
          (∀ (A : Sporos.Alloc) (pol : Sporos.DSt → Bool),
            (Sporos.drunC A pr ph pol n (Sporos.dinit pr msize inp)).abs =
              Sporos.run pr ph n (Sporos.init pr msize inp))) ∧
        (∀ (msize : Nat) (inp : List Nat) (oracle : List X86Env.Choice),
          SporosVmSimBase.FlatProgFit pr → msize < 2 ^ 44 →
          SporosVmReady.readyEndAddr pr msize inp ≤ 2 ^ 47 →
          let s₀ := SporosVmReady.mkVmReadyState pr msize inp oracle
          SporosVmSimBase.OracleAdmits s₀ → (∀ c ∈ oracle, c ≠ .nomem) →
          SporosVmExec.RegsOk s₀ ∧
          (∀ n,
            (match Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
            | .next st' =>
              ∃ ls' s', SporosProvLen.Rel st' ls' ∧
                SporosVmCore.Reaches SporosVmCore.vmImage s₀ s' ∧
                s'.k.out = s₀.k.out ∧
                SporosVmSimBase.OracleAdmits s' ∧
                SporosVmSimBase.VmRel pr ls' s'
            | .halt c out =>
              ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
                X86Vm.step SporosVmCore.vmImage s_exit = .exit (c % 256) ∧
                SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit (c % 256)) ∧
                s_exit.k.out = s₀.k.out ++ (out.map Prod.fst).reverse
            | .fail e =>
              (e = .bounds ∨ e = .divZero ∨ e = .badTable ∨ e = .demand) ∧
              ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
                X86Vm.step SporosVmCore.vmImage s_exit = .exit 1 ∧
                SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 1) ∧
                s_exit.k.out = s₀.k.out)))) ∧
    -- (c2) Committed 12,945-byte `sporos_vm` inline TMR voter & physical dispatch bounds
    (SporosVmBin.vmBytes.length = 12945 ∧
      (∀ e ∈ SporosVmBin.vmTable, SporosVmCore.vmImage e.1 = some e.2) ∧
      (∀ a ∈ SporosVmFault.allInlineVoterAddrs, SporosVmFault.InlineVoteAt a) ∧
      (∀ (C : TMR.Channel Nat) (r : X86Vm.Rates) (hr : r.ok) (x : Nat) (s : X86Vm.St),
        SporosVmExec.RegsOk s → (∀ y ∈ C.step x, y.1.1 < X86Env.W) →
        (SporosVmFault.vmDispatchVotedPhys C r hr x s).ok ∧
        PDist.fm (SporosVmFault.vmDispatchVotedPhys C r hr x s) ≤ 3 * (C.q * C.q) + 10 * r.total) ∧
      (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c v : Nat} {s : X86Vm.St},
        s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
        s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
        a < X86Env.W → b < X86Env.W → c < X86Env.W →
        ((a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) →
        (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
        PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 10 * r.total ∧
        ∃ s', PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.ok s'] ∧
          s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
          SporosVmExec.RegsOk s' ∧ s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v) ∧
      (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c : Nat} {s : X86Vm.St},
        s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
        s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
        a < X86Env.W → b < X86Env.W → c < X86Env.W →
        a ≠ b → a ≠ c → b ≠ c →
        (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
        PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.exit 3] ∧
        PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 9 * r.total)) ∧
    -- (a2) Unconditional 4-channel single-disturbance no-false-accept on rejected bytes
    ((∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr ≠ true) →
      ∃ s0, SporosCheckArm64Bin.TripledState f s0 ∧
        (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
          o' ≠ .exit 0) ∧
        (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
          y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
            SporosCheckArm64Bin.im (some fi) n (.ok s0) →
          0 < y.2 → y.1.1 ≠ .exit 0)) ∧
    -- (a3) Completeness for all well-typed `Fits48` programs
    (∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr = true →
      SporosCheckArm64Flat.Fits48 f pr →
      ∃ o, AArch64Env.OracleWf o ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) := by
  refine ⟨?_, vm_tmr_voter_safety, arm64_refused_single_fault_immune f hby hlen, ?_⟩
  · intro o ho hacc
    obtain ⟨n0, h0⟩ := hacc
    obtain ⟨pr, hdec, hchk⟩ := SporosCheckArm64Bin.spr1_arm64_bin_sound hby ho h0
    obtain ⟨hsaf, hvm⟩ := checked_prog_safety_and_vm hchk
    exact ⟨pr, hdec, hchk, hsaf, hvm⟩
  · intro pr hdec hchk hfit
    exact SporosCheckArm64Bin.spr1_arm64_bin_complete hdec hchk hfit hlen


/-! ## Pillar 3: Ground Non-Vacuity & Concrete Cross-ISA Execution Witnesses -/

private theorem bytes_lt_256_of_all {f : List Nat} (h : f.all (· < 256) = true) :
    ∀ b ∈ f, b < 256 :=
  fun b hb => of_decide_eq_true (List.all_eq_true.mp h b hb)

private theorem echo_bytes_lt_256 : ∀ b ∈ Sporos.echo.enc, b < 256 := by
  have h : Sporos.echo.enc.all (· < 256) = true := by decide +kernel
  exact bytes_lt_256_of_all h

private theorem fib_bytes_lt_256 : ∀ b ∈ Sporos.Examples.fib.enc, b < 256 := by
  have h : Sporos.Examples.fib.enc.all (· < 256) = true := by decide +kernel
  exact bytes_lt_256_of_all h

private theorem arm64_goodOracle_of_checkFlat (f : List Nat)
    (hby : ∀ b ∈ f, b < 256) (hlen : f.length < 2 ^ 45)
    (hchk : SporosCheckArm64Flat.checkFlat f = some true) :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle f) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f
      (SporosCheckArm64Bin.goodOracle f)) (· = .exit 0) := by
  have he : SporosCheckArm64Bin.expectedExit f = 0 := by
    simp [SporosCheckArm64Bin.expectedExit, hchk]
  have hc := SporosCheckArm64Bin.spr1_arm64_bin_complete_of_checkFlat hby hlen
  rwa [he] at hc

private theorem arm64_goodOracle_refuses_of_expected5 (f : List Nat)
    (hby : ∀ b ∈ f, b < 256) (hlen : f.length < 2 ^ 45)
    (he : SporosCheckArm64Bin.expectedExit f = 5) :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle f) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f
      (SporosCheckArm64Bin.goodOracle f)) (· = .exit 5) := by
  have hc := SporosCheckArm64Bin.spr1_arm64_bin_complete_of_checkFlat hby hlen
  rwa [he] at hc

private theorem echo_arm64_accepts :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.echo.enc) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.echo.enc
      (SporosCheckArm64Bin.goodOracle Sporos.echo.enc)) (· = .exit 0) :=
  arm64_goodOracle_of_checkFlat Sporos.echo.enc echo_bytes_lt_256 (by decide +kernel)
    SporosCheckArm64Flat.checkFlat_echo_valid

private theorem fib_arm64_accepts :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.Examples.fib.enc
      (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc)) (· = .exit 0) :=
  arm64_goodOracle_of_checkFlat Sporos.Examples.fib.enc fib_bytes_lt_256 (by decide +kernel)
    SporosCheckArm64Flat.checkFlat_fib_valid

private theorem empty_arm64_refuses :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle []) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState []
      (SporosCheckArm64Bin.goodOracle [])) (· = .exit 5) :=
  arm64_goodOracle_refuses_of_expected5 [] (fun _ h => by simp at h) (by decide) (by decide)

private theorem bad_arm64_refuses :
    AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle [0x53, 0x50, 0x52, 0x00]) ∧
    SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState [0x53, 0x50, 0x52, 0x00]
      (SporosCheckArm64Bin.goodOracle [0x53, 0x50, 0x52, 0x00])) (· = .exit 5) :=
  arm64_goodOracle_refuses_of_expected5 [0x53, 0x50, 0x52, 0x00] (by decide) (by decide) (by decide)

/-- Ground witnesses spanning `seed-c` (AArch64), `spr1_check_arm64` (47,636B accept/refuse/4-channel
fault immunity), and `sporos_vm` (12,945B `echo`, `fib`, provenance `provDemo`, TMR 2-of-3 masking,
and 3-way disagreement `exit 3`). -/
theorem cathedral_ground_witnesses
    (oracle_echo : List X86Env.Choice)
    (hor_echo : SporosVmSimBase.OracleAdmits
      (SporosVmReady.mkVmReadyState Sporos.echo 0 [104, 105] oracle_echo))
    (hno_echo : ∀ c ∈ oracle_echo, c ≠ .nomem)
    (oracle_fib : List X86Env.Choice)
    (hor_fib : SporosVmSimBase.OracleAdmits
      (SporosVmReady.mkVmReadyState Sporos.Examples.fib 0 [] oracle_fib))
    (hno_fib : ∀ c ∈ oracle_fib, c ≠ .nomem)
    (oracle_prov : List X86Env.Choice)
    (hor_prov : SporosVmSimBase.OracleAdmits
      (SporosVmReady.mkVmReadyState SporosProvLen.provDemo 0 [65, 0] oracle_prov))
    (hno_prov : ∀ c ∈ oracle_prov, c ≠ .nomem)
    (oracle_dem : List X86Env.Choice)
    (hor_dem : SporosVmSimBase.OracleAdmits
      (SporosVmReady.mkVmReadyState SporosProvLen.provDemo 0 [65, 1] oracle_dem))
    (hno_dem : ∀ c ∈ oracle_dem, c ≠ .nomem) :
    -- 1. AArch64 `seed-c` ground witnesses
    (AArch64Min.exec 200 AArch64Min.SeedC.seedC [] = .exit 0 [] ∧
      AArch64Min.exec 1000 AArch64Min.SeedC.seedC [13] = .exit 1 []) ∧
    -- 2. AArch64 `spr1_check_arm64` ground accept, refuse, and 4-channel fault-control witnesses
    ((AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.echo.enc) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.echo.enc
          (SporosCheckArm64Bin.goodOracle Sporos.echo.enc)) (· = .exit 0)) ∧
      (AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.Examples.fib.enc
          (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc)) (· = .exit 0)) ∧
      SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState []
        (SporosCheckArm64Bin.goodOracle [])) (· = .exit 5) ∧
      SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState [0x53, 0x50, 0x52, 0x00]
        (SporosCheckArm64Bin.goodOracle [0x53, 0x50, 0x52, 0x00])) (· = .exit 5) ∧
      (∃ s0, SporosCheckArm64Bin.TripledState [0x53, 0x50, 0x52, 0x00] s0 ∧
        (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
          o' ≠ .exit 0) ∧
        (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
          y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
            SporosCheckArm64Bin.im (some fi) n (.ok s0) →
          0 < y.2 → y.1.1 ≠ .exit 0))) ∧
    -- 3. x86-64 `sporos_vm` execution & TMR voter ground witnesses
    ((let s₀ := SporosVmReady.mkVmReadyState Sporos.echo 0 [104, 105] oracle_echo
      ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
        X86Vm.step SporosVmCore.vmImage s_exit = .exit 0 ∧
        SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 0) ∧ s_exit.k.out = [104, 105]) ∧
     (let s₀ := SporosVmReady.mkVmReadyState Sporos.Examples.fib 0 [] oracle_fib
      ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
        X86Vm.step SporosVmCore.vmImage s_exit = .exit 0 ∧
        SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 0) ∧
        s_exit.k.out = SporosVmWitness.fibBytes) ∧
     (let s₀ := SporosVmReady.mkVmReadyState SporosProvLen.provDemo 0 [65, 0] oracle_prov
      ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
        X86Vm.step SporosVmCore.vmImage s_exit = .exit 0 ∧
        SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 0) ∧ s_exit.k.out = [0, 14, 66]) ∧
     (let s₀ := SporosVmReady.mkVmReadyState SporosProvLen.provDemo 0 [65, 1] oracle_dem
      ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
        X86Vm.step SporosVmCore.vmImage s_exit = .exit 1 ∧
        SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 1) ∧ s_exit.k.out = []) ∧
     (let s : X86Vm.St :=
        { (((SporosVmReady.mkVmReadyState Sporos.echo 0 [] []).set 0 (some 42)).set 3
          (some (Nat.xor 42 0x55))).set 1 (some 42) with rip := SporosVmBin.L_vm_vote }
      ∃ s', SporosVmCore.Reaches SporosVmCore.vmImage s s' ∧
        s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        s'.reg 0 = some 42 ∧ s'.reg 1 = some 42 ∧ s'.reg 3 = some 42) ∧
     (let s : X86Vm.St :=
        { (((SporosVmReady.mkVmReadyState Sporos.echo 0 [] []).set 0 (some 42)).set 3
          (some 99)).set 1 (some 7) with rip := SporosVmBin.L_vm_vote }
      SporosVmCore.Ends SporosVmCore.vmImage s (.exit 3))) := by
  have hrej_bad : ∀ p, Sporos.decode [0x53, 0x50, 0x52, 0x00] = some p → Sporos.checkProg p ≠ true := by
    have hd : Sporos.decode [0x53, 0x50, 0x52, 0x00] = none := by decide
    intro p hp
    rw [hd] at hp
    cases hp
  have hflt_bad := arm64_refused_single_fault_immune [0x53, 0x50, 0x52, 0x00]
    (by decide) (by decide) hrej_bad
  have hc_cr : AArch64Min.exec 1000 AArch64Min.SeedC.seedC [13] = .exit 1 [] := by
    rw [AArch64Min.exec_of_load (AArch64Min.SeedC.load_C _)]
    decide +kernel
  refine ⟨⟨AArch64Min.SeedC.c_empty, hc_cr⟩,
    ⟨echo_arm64_accepts, fib_arm64_accepts, empty_arm64_refuses.2, bad_arm64_refuses.2, hflt_bad⟩,
    SporosVmWitness.witness_vm_echo oracle_echo hor_echo hno_echo,
    SporosVmWitness.witness_vm_fib oracle_fib hor_fib hno_fib,
    SporosVmWitness.witness_vm_provDemo_nonzero oracle_prov hor_prov hno_prov,
    SporosVmWitness.witness_vm_provDemo_demand_fail oracle_dem hor_dem hno_dem,
    SporosVmWitness.witness_tmr_vote_single_disturbance,
    SporosVmWitness.witness_tmr_vote_three_way_exit3⟩


/-! ## Pillar 4: Dual-ISA Stage-1 Equivalence & Root A End-to-End Cathedral -/

/-- Pillar 4 (`dual_isa_stage1_cathedral_iff`):
Cross-ISA extensional equivalence between Root A's hand-written 2,447-byte x86-64
checker `SporosX1.stage1Im` (bounded within cubic `SporosX1.stage1Bound f`) and AArch64's
47,636-byte TMR-hardened checker `SporosCheckArm64Bin`:
1. Unconditional AArch64 `.exit 0` admission implies bounded Root A x86-64 `.exit 0`;
2. Unconditional Root A x86-64 bounded `.exit 0 ↔ Accepted f` and `.exit 1 ↔ ¬ Accepted f`;
3. Root A x86-64 `.exit 1` implies AArch64 never exits `0` under any well-formed oracle and
   inhabits a concrete `TripledState` immune to all 4-channel single physical faults;
4. For every program satisfying `Fits48 f pr`, Root A x86-64 `.exit 0` and `.exit 1`
   are bidirectional with AArch64 `spr1_check_arm64`;
5. Bounded Root A `.exit 0 ∨ .exit 1` totality holds within `SporosX1.stage1Bound f`. -/
theorem dual_isa_stage1_cathedral_iff
    (f path argv0 stdin : List Nat) (oracle : List X86Env.Choice) (top : Nat)
    (hinit : SporosX1.Stage1InitOk f path argv0 top)
    (horacle : SporosX1.OracleAdmits (SporosX1.stage1Init f path argv0 stdin oracle top))
    (hnomem : ¬ SporosX1.HitsNomem (SporosX1.stage1Init f path argv0 stdin oracle top))
    (hlen : f.length < 2 ^ 45) :
    ((∃ o : List AArch64Env.Choice, AArch64Env.OracleWf o ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) →
      (∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0)) ∧
    ((∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0) ↔
      SporosX1.Accepted f) ∧
    ((∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1) ↔
      ¬ SporosX1.Accepted f) ∧
    ((∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1) →
      (∀ o : List AArch64Env.Choice, AArch64Env.OracleWf o →
        ¬ SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) ∧
      ∃ s0, SporosCheckArm64Bin.TripledState f s0 ∧
        (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
          o' ≠ .exit 0) ∧
        (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
          y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
            SporosCheckArm64Bin.im (some fi) n (.ok s0) →
          0 < y.2 → y.1.1 ≠ .exit 0)) ∧
    ((∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr = true →
        SporosCheckArm64Flat.Fits48 f pr) →
      (((∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0) ↔
        (∃ o : List AArch64Env.Choice, AArch64Env.OracleWf o ∧
          SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0))) ∧
       ((∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1) ↔
        (∀ o : List AArch64Env.Choice, AArch64Env.OracleWf o →
          ¬ SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0))))) ∧
    (∃ k, k ≤ SporosX1.stage1Bound f ∧
      (X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0 ∨
       X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1)) := by
  have hfin := SporosX1.stage1_final_target f path argv0 stdin oracle top hinit horacle hnomem
  have harm := arm64_admit_to_x86vm_cathedral f hinit.1 hlen
  have hArm0ToRoot0 :
      (∃ o : List AArch64Env.Choice, AArch64Env.OracleWf o ∧
          SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) →
        (∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0) := by
    intro ⟨o, ho, he⟩
    obtain ⟨pr, hdec, hchk, -, -⟩ := harm.1 o ho he
    exact hfin.1.mpr ⟨pr, hdec, hchk⟩
  have hRoot1ToNoArm0 :
      (∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1) →
        ∀ o : List AArch64Env.Choice, AArch64Env.OracleWf o →
          ¬ SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0) := by
    intro h1 o ho he
    have hno : ¬ SporosX1.Accepted f := hfin.2.1.mp h1
    obtain ⟨pr, hdec, hchk, -, -⟩ := harm.1 o ho he
    exact hno ⟨pr, hdec, hchk⟩
  refine ⟨hArm0ToRoot0, hfin.1, hfin.2.1, ?_, ?_, hfin.2.2⟩
  · intro h1
    have hno : ¬ SporosX1.Accepted f := hfin.2.1.mp h1
    have hrej : ∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr ≠ true :=
      fun pr hd hc => hno ⟨pr, hd, hc⟩
    exact ⟨hRoot1ToNoArm0 h1, harm.2.2.1 hrej⟩
  · intro hfits
    refine ⟨⟨fun hr0 => ?_, hArm0ToRoot0⟩, ⟨hRoot1ToNoArm0, fun hna => ?_⟩⟩
    · obtain ⟨pr, hdec, hchk⟩ := hfin.1.mp hr0
      exact harm.2.2.2 pr hdec hchk (hfits pr hdec hchk)
    · apply hfin.2.1.mpr
      intro ⟨pr, hdec, hchk⟩
      obtain ⟨o, ho, he⟩ := harm.2.2.2 pr hdec hchk (hfits pr hdec hchk)
      exact hna o ho he

/-- Crown Theorem (`rootA_admit_to_x86vm_cathedral`):
End-to-end Root A x86-64 admission -> Cross-ISA equivalence -> Semantic Safety ->
Provenance Bisimulation -> Native 12,945B x86-64 `sporos_vm` + TMR Execution Cathedral.
Under `Stage1InitOk f path argv0 top`, `f.length ≤ 2 ^ 40`, and `Room f path argv0 top`:
1. A non-refusing admitting kernel oracle exists constructively
   (`SporosX1.stage1_oracle_exists_target`);
2. For every non-refusing admitting oracle `oracle`, `stage1Im` terminates in `.exit 0 ∨ .exit 1`
   within cubic `SporosX1.stage1Bound f`, agrees with AArch64 `spr1_check_arm64`, and every
   `.exit 0` admission yields decoded `pr` with `Sporos.checkProg pr = true`, static-error
   freedom (`Sporos.run_safe`, `SporosProvLen.lrun_safe`), provenance bisimulations
   (`lrun_sim_init`, `drun_init_hashcons`, `drunC_init`), native x86-64 `sporos_vm` simulation
   from `mkVmReadyState` under `FlatProgFit pr`, and inline TMR dispatch voter safety bounds;
3. Any oracle whose run triggers `-ENOMEM` (`HitsNomem`) deterministically exits `2` within
   `SporosX1.stage1Bound f` (`SporosX1.stage1_final_exit2_target`). -/
theorem rootA_admit_to_x86vm_cathedral
    (f path argv0 stdin : List Nat) (top : Nat)
    (hinit : SporosX1.Stage1InitOk f path argv0 top)
    (hsize : f.length ≤ 2 ^ 40)
    (hroom : SporosX1.Room f path argv0 top) :
    -- 1. Non-vacuous OS oracle existence
    (∃ oracle_good,
      SporosX1.OracleAdmits (SporosX1.stage1Init f path argv0 stdin oracle_good top) ∧
      ¬ SporosX1.HitsNomem (SporosX1.stage1Init f path argv0 stdin oracle_good top)) ∧
    -- 2. Full dual-ISA + post-admission safety/provenance/VM cathedral for any non-refusing oracle
    (∀ oracle,
      SporosX1.OracleAdmits (SporosX1.stage1Init f path argv0 stdin oracle top) →
      ¬ SporosX1.HitsNomem (SporosX1.stage1Init f path argv0 stdin oracle top) →
      ((∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0) →
        ∃ pr, Sporos.decode f = some pr ∧ Sporos.checkProg pr = true ∧
          (∀ (ph : Sporos.Phys) (msize : Nat) (inp : List Nat) (n : Nat),
            (∀ e, Sporos.run pr ph n (Sporos.init pr msize inp) = .fail e → e.static = false) ∧
            (∀ e, SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp) = .fail e → e.static = false) ∧
            SporosProvLen.ResRel (Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp))
              (SporosProvLen.lrun pr n (SporosProvLen.linit pr msize inp)) ∧
            (Sporos.drun Sporos.Alloc.hashcons pr ph n (Sporos.dinit pr msize inp)).abs =
              Sporos.run pr ph n (Sporos.init pr msize inp) ∧
            (∀ (A : Sporos.Alloc) (pol : Sporos.DSt → Bool),
              (Sporos.drunC A pr ph pol n (Sporos.dinit pr msize inp)).abs =
                Sporos.run pr ph n (Sporos.init pr msize inp))) ∧
          (∀ (msize : Nat) (inp : List Nat) (vm_oracle : List X86Env.Choice),
            SporosVmSimBase.FlatProgFit pr → msize < 2 ^ 44 →
            SporosVmReady.readyEndAddr pr msize inp ≤ 2 ^ 47 →
            let s₀ := SporosVmReady.mkVmReadyState pr msize inp vm_oracle
            SporosVmSimBase.OracleAdmits s₀ → (∀ c ∈ vm_oracle, c ≠ .nomem) →
            SporosVmExec.RegsOk s₀ ∧
            (∀ n,
              (match Sporos.run pr Sporos.noPhys n (Sporos.init pr msize inp) with
              | .next st' =>
                ∃ ls' s', SporosProvLen.Rel st' ls' ∧
                  SporosVmCore.Reaches SporosVmCore.vmImage s₀ s' ∧
                  s'.k.out = s₀.k.out ∧
                  SporosVmSimBase.OracleAdmits s' ∧
                  SporosVmSimBase.VmRel pr ls' s'
              | .halt c out =>
                ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
                  X86Vm.step SporosVmCore.vmImage s_exit = .exit (c % 256) ∧
                  SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit (c % 256)) ∧
                  s_exit.k.out = s₀.k.out ++ (out.map Prod.fst).reverse
              | .fail e =>
                (e = .bounds ∨ e = .divZero ∨ e = .badTable ∨ e = .demand) ∧
                ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
                  X86Vm.step SporosVmCore.vmImage s_exit = .exit 1 ∧
                  SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 1) ∧
                  s_exit.k.out = s₀.k.out)))) ∧
      -- Inline `sporos_vm` TMR dispatch voter bounds
      (SporosVmBin.vmBytes.length = 12945 ∧
        (∀ e ∈ SporosVmBin.vmTable, SporosVmCore.vmImage e.1 = some e.2) ∧
        (∀ a ∈ SporosVmFault.allInlineVoterAddrs, SporosVmFault.InlineVoteAt a) ∧
        (∀ (C : TMR.Channel Nat) (r : X86Vm.Rates) (hr : r.ok) (x : Nat) (s : X86Vm.St),
          SporosVmExec.RegsOk s → (∀ y ∈ C.step x, y.1.1 < X86Env.W) →
          (SporosVmFault.vmDispatchVotedPhys C r hr x s).ok ∧
          PDist.fm (SporosVmFault.vmDispatchVotedPhys C r hr x s) ≤ 3 * (C.q * C.q) + 10 * r.total) ∧
        (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c v : Nat} {s : X86Vm.St},
          s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
          s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
          a < X86Env.W → b < X86Env.W → c < X86Env.W →
          ((a = v ∧ b = v) ∨ (a = v ∧ c = v) ∨ (b = v ∧ c = v)) →
          (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
          PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 10 * r.total ∧
          ∃ s', PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.ok s'] ∧
            s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
            SporosVmExec.RegsOk s' ∧ s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v) ∧
        (∀ (r : X86Vm.Rates) (hr : r.ok) {a b c : Nat} {s : X86Vm.St},
          s.rip = SporosVmBin.L_vm_vote → SporosVmExec.RegsOk s →
          s.reg 0 = some a → s.reg 3 = some b → s.reg 1 = some c →
          a < X86Env.W → b < X86Env.W → c < X86Env.W →
          a ≠ b → a ≠ c → b ≠ c →
          (SporosVmFault.vmVotePhys r hr s a b c).ok ∧
          PDist.faultFree (SporosVmFault.vmVotePhys r hr s a b c) = [.exit 3] ∧
          PDist.fm (SporosVmFault.vmVotePhys r hr s a b c) ≤ 9 * r.total)) ∧
      -- Dual-ISA stage-1 equivalence + AArch64 4-channel fault-immune refusal + bounded totality
      (((∃ o : List AArch64Env.Choice, AArch64Env.OracleWf o ∧
          SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) →
        (∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0)) ∧
       ((∃ k, k ≤ SporosX1.stage1Bound f ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1) →
        (∀ o : List AArch64Env.Choice, AArch64Env.OracleWf o →
          ¬ SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)) ∧
        ∃ s0, SporosCheckArm64Bin.TripledState f s0 ∧
          (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
            o' ≠ .exit 0) ∧
          (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
            y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
              SporosCheckArm64Bin.im (some fi) n (.ok s0) →
            0 < y.2 → y.1.1 ≠ .exit 0)) ∧
       ((∀ pr, Sporos.decode f = some pr → Sporos.checkProg pr = true →
          SporosCheckArm64Flat.Fits48 f pr) →
        ((∃ k, k ≤ SporosX1.stage1Bound f ∧
            X86Int.run SporosX1.stage1Im k
              (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0) ↔
          (∃ o : List AArch64Env.Choice, AArch64Env.OracleWf o ∧
            SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f o) (· = .exit 0)))) ∧
       (∃ k, k ≤ SporosX1.stage1Bound f ∧
        (X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 0 ∨
         X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 1)))) ∧
    -- 3. Root A `-ENOMEM` exit-2 completeness
    (∀ oracle,
      SporosX1.OracleAdmits (SporosX1.stage1Init f path argv0 stdin oracle top) →
      SporosX1.HitsNomem (SporosX1.stage1Init f path argv0 stdin oracle top) →
      ∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f path argv0 stdin oracle top)) = .exit 2) := by
  refine ⟨SporosX1.stage1_oracle_exists_target f path argv0 stdin top hinit hsize hroom,
    ?_, fun oracle horacle hhit =>
      SporosX1.stage1_final_exit2_target f path argv0 stdin oracle top hinit horacle hhit⟩
  intro oracle horacle hnomem
  have hlen : f.length < 2 ^ 45 := by omega
  have hdual := dual_isa_stage1_cathedral_iff f path argv0 stdin oracle top hinit horacle hnomem hlen
  refine ⟨?_, vm_tmr_voter_safety,
    hdual.1, hdual.2.2.2.1, fun hfits => (hdual.2.2.2.2.1 hfits).1, hdual.2.2.2.2.2⟩
  intro hr0
  obtain ⟨pr, hdec, hchk⟩ := hdual.2.1.mp hr0
  obtain ⟨hsaf, hvm⟩ := checked_prog_safety_and_vm hchk
  exact ⟨pr, hdec, hchk, hsaf, hvm⟩

private theorem concrete_stk_base :
    (X86Env.initStack 0x100000000 [[115], [47, 112]]).1.base = 0x100000000 - 8192 := by
  simp only [X86Env.initStack]
  congr 1

private theorem concrete_stk_hi :
    (X86Env.initStack 0x100000000 [[115], [47, 112]]).1.hi = 0x100000000 := by
  have hlen : (X86Env.initStack 0x100000000 [[115], [47, 112]]).1.data.length = 8192 := by
    rw [SporosX1.initStack_data_length]
    decide
  unfold X86Env.Region.hi
  rw [concrete_stk_base, hlen]

private theorem stage1_concrete_init_ok (f : List Nat) (hf : ∀ b ∈ f, b < 256) :
    SporosX1.Stage1InitOk f [47, 112] [115] 0x100000000 := by
  refine ⟨hf, by decide, by decide, by decide, by decide, by decide, by decide, by decide, ?_,
    concrete_stk_hi⟩
  have hbase : SporosX1.base = 0x400000 := rfl
  rw [SporosX1.phVaddr_eq, hbase, SporosX1.phMemsz_eq, SporosX1.stage1Code_length,
    concrete_stk_base]
  decide

private theorem stage1_concrete_room (f : List Nat) (hf_len : f.length ≤ 10000) :
    SporosX1.Room f [47, 112] [115] 0x100000000 := by
  have hmf := SporosX1.stage1Mf_le f
  refine ⟨0x10000000, by decide, by decide, by omega, ?_⟩
  have hbase : SporosX1.base = 0x400000 := rfl
  have hhi0 : (⟨SporosX1.phVaddr, SporosX1.loadSegment, false⟩ : X86Env.Region).hi ≤
      0x10000000 := by
    simp only [X86Env.Region.hi, SporosX1.phVaddr_eq, hbase, SporosX1.loadSegment_eq,
      SporosX1.stage1Code_length]
    decide
  have hbase1 : 0x10000000 + SporosX1.stage1Mf f ≤
      (X86Env.initStack 0x100000000 [[115], [47, 112]]).1.base := by
    rw [concrete_stk_base]
    omega
  have hmem : (SporosX1.stage1Init f [47, 112] [115] [] [] 0x100000000).mem =
      [⟨SporosX1.phVaddr, SporosX1.loadSegment, false⟩,
       (X86Env.initStack 0x100000000 [[115], [47, 112]]).1] := rfl
  rw [hmem]
  simp only [X86Env.Mem.free, List.all_cons, List.all_nil, Bool.and_true,
    Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]
  exact ⟨Or.inr hhi0, Or.inl hbase1⟩

private theorem stage1_concrete_accept_witness (f : List Nat)
    (hby : ∀ b ∈ f, b < 256) (hlen_small : f.length ≤ 10000)
    (harm : AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle f) ∧
      SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState f
        (SporosCheckArm64Bin.goodOracle f)) (· = .exit 0)) :
    (∃ oracle_x86,
      SporosX1.OracleAdmits (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000) ∧
      ¬ SporosX1.HitsNomem (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000) ∧
      ∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 0) ∧
    SporosX1.Accepted f := by
  have hinit := stage1_concrete_init_ok f hby
  have hroom := stage1_concrete_room f hlen_small
  have hcat := rootA_admit_to_x86vm_cathedral f [47, 112] [115] [] 0x100000000
    hinit (by omega) hroom
  obtain ⟨oracle_x86, hor, hno⟩ := hcat.1
  have hs := hcat.2.1 oracle_x86 hor hno
  have hk := hs.2.2.1 ⟨SporosCheckArm64Bin.goodOracle f, harm.1, harm.2⟩
  obtain ⟨n0, hn0⟩ := harm.2
  obtain ⟨pr, hdec, hchk⟩ := SporosCheckArm64Bin.spr1_arm64_bin_sound hby harm.1 hn0
  exact ⟨⟨oracle_x86, hor, hno, hk⟩, ⟨pr, hdec, hchk⟩⟩

private theorem stage1_concrete_refuse_witness (f : List Nat)
    (hby : ∀ b ∈ f, b < 256) (hlen_small : f.length ≤ 10000)
    (hdec : Sporos.decode f = none) :
    (∃ oracle_x86,
      SporosX1.OracleAdmits (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000) ∧
      ¬ SporosX1.HitsNomem (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000) ∧
      ∃ k, k ≤ SporosX1.stage1Bound f ∧
        X86Int.run SporosX1.stage1Im k
          (.ok (SporosX1.stage1Init f [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 1) ∧
    ¬ SporosX1.Accepted f ∧
    (∃ s0, SporosCheckArm64Bin.TripledState f s0 ∧
      (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
        o' ≠ .exit 0) ∧
      (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
        y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
          SporosCheckArm64Bin.im (some fi) n (.ok s0) →
        0 < y.2 → y.1.1 ≠ .exit 0)) := by
  have hnotacc : ¬ SporosX1.Accepted f := fun ⟨p, hp, _⟩ => by rw [hdec] at hp; cases hp
  have hinit := stage1_concrete_init_ok f hby
  have hroom := stage1_concrete_room f hlen_small
  obtain ⟨oracle_x86, hor, hno⟩ :=
    SporosX1.stage1_oracle_exists_target f [47, 112] [115] [] 0x100000000 hinit (by omega) hroom
  have hdual := dual_isa_stage1_cathedral_iff f [47, 112] [115] [] oracle_x86 0x100000000
    hinit hor hno (by omega)
  have hr1 := hdual.2.2.1.mpr hnotacc
  exact ⟨⟨oracle_x86, hor, hno, hr1⟩, hnotacc, (hdual.2.2.2.1 hr1).2⟩

/-- Pillar 4 Ground Witnesses (`stage1_cathedral_ground_witnesses`):
Zero-hypothesis ground non-vacuity bridging Root A x86-64 `stage1Im`, AArch64
`spr1_check_arm64`, semantic acceptance/rejection, 4-channel fault immunity, and x86-64
`sporos_vm` across `echo`, `fib`, empty `[]`, and corrupted `[0x53, 0x50, 0x52, 0x00]`. -/
theorem stage1_cathedral_ground_witnesses :
    -- 1. Valid `echo` ground witness across Root A x86-64, AArch64 `spr1_check_arm64`, and `sporos_vm`
    ((∃ oracle_x86,
        SporosX1.OracleAdmits (SporosX1.stage1Init Sporos.echo.enc [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ¬ SporosX1.HitsNomem (SporosX1.stage1Init Sporos.echo.enc [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ∃ k, k ≤ SporosX1.stage1Bound Sporos.echo.enc ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init Sporos.echo.enc [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 0) ∧
      (AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.echo.enc) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.echo.enc
          (SporosCheckArm64Bin.goodOracle Sporos.echo.enc)) (· = .exit 0)) ∧
      SporosX1.Accepted Sporos.echo.enc ∧
      (∀ oracle_vm,
        SporosVmSimBase.OracleAdmits (SporosVmReady.mkVmReadyState Sporos.echo 0 [104, 105] oracle_vm) →
        (∀ c ∈ oracle_vm, c ≠ .nomem) →
        let s₀ := SporosVmReady.mkVmReadyState Sporos.echo 0 [104, 105] oracle_vm
        ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
          X86Vm.step SporosVmCore.vmImage s_exit = .exit 0 ∧
          SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 0) ∧ s_exit.k.out = [104, 105])) ∧
    -- 2. Valid `fib` ground witness across Root A x86-64, AArch64 `spr1_check_arm64`, and `sporos_vm`
    ((∃ oracle_x86,
        SporosX1.OracleAdmits (SporosX1.stage1Init Sporos.Examples.fib.enc [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ¬ SporosX1.HitsNomem (SporosX1.stage1Init Sporos.Examples.fib.enc [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ∃ k, k ≤ SporosX1.stage1Bound Sporos.Examples.fib.enc ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init Sporos.Examples.fib.enc [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 0) ∧
      (AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState Sporos.Examples.fib.enc
          (SporosCheckArm64Bin.goodOracle Sporos.Examples.fib.enc)) (· = .exit 0)) ∧
      SporosX1.Accepted Sporos.Examples.fib.enc ∧
      (∀ oracle_vm,
        SporosVmSimBase.OracleAdmits (SporosVmReady.mkVmReadyState Sporos.Examples.fib 0 [] oracle_vm) →
        (∀ c ∈ oracle_vm, c ≠ .nomem) →
        let s₀ := SporosVmReady.mkVmReadyState Sporos.Examples.fib 0 [] oracle_vm
        ∃ s_exit, SporosVmCore.Reaches SporosVmCore.vmImage s₀ s_exit ∧
          X86Vm.step SporosVmCore.vmImage s_exit = .exit 0 ∧
          SporosVmCore.Ends SporosVmCore.vmImage s₀ (.exit 0) ∧
          s_exit.k.out = SporosVmWitness.fibBytes)) ∧
    -- 3. Invalid empty `[]` ground refusal + 4-channel fault immunity
    ((∃ oracle_x86,
        SporosX1.OracleAdmits (SporosX1.stage1Init [] [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ¬ SporosX1.HitsNomem (SporosX1.stage1Init [] [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ∃ k, k ≤ SporosX1.stage1Bound [] ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init [] [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 1) ∧
      (AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle []) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState []
          (SporosCheckArm64Bin.goodOracle [])) (· = .exit 5)) ∧
      ¬ SporosX1.Accepted [] ∧
      (∃ s0, SporosCheckArm64Bin.TripledState [] s0 ∧
        (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
          o' ≠ .exit 0) ∧
        (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
          y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
            SporosCheckArm64Bin.im (some fi) n (.ok s0) →
          0 < y.2 → y.1.1 ≠ .exit 0))) ∧
    -- 4. Invalid corrupted `[0x53, 0x50, 0x52, 0x00]` ground refusal + 4-channel fault immunity
    ((∃ oracle_x86,
        SporosX1.OracleAdmits (SporosX1.stage1Init [0x53, 0x50, 0x52, 0x00] [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ¬ SporosX1.HitsNomem (SporosX1.stage1Init [0x53, 0x50, 0x52, 0x00] [47, 112] [115] [] oracle_x86 0x100000000) ∧
        ∃ k, k ≤ SporosX1.stage1Bound [0x53, 0x50, 0x52, 0x00] ∧
          X86Int.run SporosX1.stage1Im k
            (.ok (SporosX1.stage1Init [0x53, 0x50, 0x52, 0x00] [47, 112] [115] [] oracle_x86 0x100000000)) = .exit 1) ∧
      (AArch64Env.OracleWf (SporosCheckArm64Bin.goodOracle [0x53, 0x50, 0x52, 0x00]) ∧
        SporosCheckArm64Bin.Ends (SporosCheckArm64Bin.initState [0x53, 0x50, 0x52, 0x00]
          (SporosCheckArm64Bin.goodOracle [0x53, 0x50, 0x52, 0x00])) (· = .exit 5)) ∧
      ¬ SporosX1.Accepted [0x53, 0x50, 0x52, 0x00] ∧
      (∃ s0, SporosCheckArm64Bin.TripledState [0x53, 0x50, 0x52, 0x00] s0 ∧
        (∀ n o', SporosCheckArm64Flat.OneFaultRun SporosCheckArm64Bin.alts SporosCheckArm64Bin.im n (.ok s0) o' →
          o' ≠ .exit 0) ∧
        (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
          y ∈ SporosCheckArm64Flat.prunWithFaultAt SporosCheckArm64Bin.alts r hr ch
            SporosCheckArm64Bin.im (some fi) n (.ok s0) →
          0 < y.2 → y.1.1 ≠ .exit 0))) := by
  have hec := stage1_concrete_accept_witness Sporos.echo.enc echo_bytes_lt_256
    (by decide +kernel) echo_arm64_accepts
  have hfb := stage1_concrete_accept_witness Sporos.Examples.fib.enc fib_bytes_lt_256
    (by decide +kernel) fib_arm64_accepts
  have hem := stage1_concrete_refuse_witness [] (fun _ h => by simp at h) (by decide) (by decide)
  have hbd := stage1_concrete_refuse_witness [0x53, 0x50, 0x52, 0x00] (by decide) (by decide)
    (by decide)
  refine ⟨⟨hec.1, echo_arm64_accepts, hec.2, fun o ho hn => SporosVmWitness.witness_vm_echo o ho hn⟩,
    ⟨hfb.1, fib_arm64_accepts, hfb.2, fun o ho hn => SporosVmWitness.witness_vm_fib o ho hn⟩,
    ⟨hem.1, empty_arm64_refuses, hem.2.1, hem.2.2⟩,
    ⟨hbd.1, bad_arm64_refuses, hbd.2.1, hbd.2.2⟩⟩

end BootstrapCathedral
