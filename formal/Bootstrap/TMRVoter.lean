/-
TMRVoter: formal verification of the canonical 38-byte x86-64 machine-code TMR
voter under architectural semantics (`X86Int.run`) and physical fault semantics
(`X86Int.prun` with `PDist`, `TMR`, and `TMRExposure`).

The voter takes three 64-bit word copies in `rax` (reg 0), `rbx` (reg 3), and
`rcx` (reg 1), compares all pairs, repairs any single dissenting copy on majority
and parks at `0x400018` (`voterDone`), or halts via Linux `exit(3)` when all
three copies disagree (`Out.exit 3`).
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.X86Env
import Bootstrap.X86Int
import Bootstrap.X86Dec

namespace TMRVoter

open X86Env X86Int X86Dec PDist TMR

/-! ### Decidable equality for `X86Int.St` and `X86Int.Out` -/

instance instDecidableEqSt : DecidableEq X86Int.St := fun a b => by
  cases a; cases b
  simp only [X86Int.St.mk.injEq]
  exact inferInstance

instance instDecidableEqOut : DecidableEq X86Int.Out := fun a b => by
  cases a <;> cases b
  · simp only [X86Int.Out.ok.injEq]; exact inferInstance
  · exact isFalse (by intro h; cases h)
  · exact isFalse (by intro h; cases h)
  · exact isFalse (by intro h; cases h)
  · simp only [X86Int.Out.exit.injEq]; exact inferInstance
  · exact isFalse (by intro h; cases h)
  · exact isFalse (by intro h; cases h)
  · exact isFalse (by intro h; cases h)
  · exact isTrue rfl

/-! ### Canonical 38-byte machine-code voter -/

def voterBase : Nat := 0x400000

def voterCode : List Nat :=
  [ 0x48, 0x39, 0xd8              -- 0x00: cmp rax, rbx
  , 0x74, 0x0d                    -- 0x03: je  0x400012
  , 0x48, 0x39, 0xc8              -- 0x05: cmp rax, rcx
  , 0x74, 0x08                    -- 0x08: je  0x400012
  , 0x48, 0x39, 0xcb              -- 0x0a: cmp rbx, rcx
  , 0x75, 0x0b                    -- 0x0d: jne 0x40001a
  , 0x48, 0x89, 0xd8              -- 0x0f: mov rax, rbx
  , 0x48, 0x89, 0xc3              -- 0x12: mov rbx, rax
  , 0x48, 0x89, 0xc1              -- 0x15: mov rcx, rax
  , 0xeb, 0xfe                    -- 0x18: jmp 0x400018
  , 0xbf, 0x03, 0x00, 0x00, 0x00  -- 0x1a: mov edi, 3
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x1f: mov eax, 60
  , 0x0f, 0x05                    -- 0x24: syscall
  ]

theorem voterCode_length : voterCode.length = 38 := by decide +kernel

def voterIm : Nat → Option (X86Int.Ins × Nat) :=
  X86Dec.imageOf voterCode voterBase

def voterAlts : X86Int.Alts := fun a =>
  if a < voterBase then []
  else match voterIm a with
    | some (_, len) => X86Dec.altsAt (fun i => voterCode[a - voterBase + i]?) a len
    | none => []

theorem voterAlts_len_00 : (voterAlts (voterBase + 0x00)).length = 24 := by decide +kernel
theorem voterAlts_len_24 : (voterAlts (voterBase + 0x24)).length = 16 := by decide +kernel

theorem voterIm_00 : voterIm voterBase = some (.cmp 0 3, 3) := by decide +kernel
theorem voterIm_03 : voterIm (voterBase + 0x03) = some (.jcc 4 (voterBase + 0x12), 2) := by decide +kernel
theorem voterIm_05 : voterIm (voterBase + 0x05) = some (.cmp 0 1, 3) := by decide +kernel
theorem voterIm_08 : voterIm (voterBase + 0x08) = some (.jcc 4 (voterBase + 0x12), 2) := by decide +kernel
theorem voterIm_0a : voterIm (voterBase + 0x0a) = some (.cmp 3 1, 3) := by decide +kernel
theorem voterIm_0d : voterIm (voterBase + 0x0d) = some (.jcc 5 (voterBase + 0x1a), 2) := by decide +kernel
theorem voterIm_0f : voterIm (voterBase + 0x0f) = some (.mov 0 3, 3) := by decide +kernel
theorem voterIm_12 : voterIm (voterBase + 0x12) = some (.mov 3 0, 3) := by decide +kernel
theorem voterIm_15 : voterIm (voterBase + 0x15) = some (.mov 1 0, 3) := by decide +kernel
theorem voterIm_18 : voterIm (voterBase + 0x18) = some (.jmp (voterBase + 0x18), 2) := by decide +kernel
theorem voterIm_1a : voterIm (voterBase + 0x1a) = some (.movI 7 3, 5) := by decide +kernel
theorem voterIm_1f : voterIm (voterBase + 0x1f) = some (.movI 0 60, 5) := by decide +kernel
theorem voterIm_24 : voterIm (voterBase + 0x24) = some (.syscall, 2) := by decide +kernel

/-- Every instruction offset of `voterCode` decodes to its exact `(Ins, len)`. -/
theorem voterIm_decode :
    voterIm (voterBase + 0x00) = some (.cmp 0 3, 3) ∧
    voterIm (voterBase + 0x03) = some (.jcc 4 (voterBase + 0x12), 2) ∧
    voterIm (voterBase + 0x05) = some (.cmp 0 1, 3) ∧
    voterIm (voterBase + 0x08) = some (.jcc 4 (voterBase + 0x12), 2) ∧
    voterIm (voterBase + 0x0a) = some (.cmp 3 1, 3) ∧
    voterIm (voterBase + 0x0d) = some (.jcc 5 (voterBase + 0x1a), 2) ∧
    voterIm (voterBase + 0x0f) = some (.mov 0 3, 3) ∧
    voterIm (voterBase + 0x12) = some (.mov 3 0, 3) ∧
    voterIm (voterBase + 0x15) = some (.mov 1 0, 3) ∧
    voterIm (voterBase + 0x18) = some (.jmp (voterBase + 0x18), 2) ∧
    voterIm (voterBase + 0x1a) = some (.movI 7 3, 5) ∧
    voterIm (voterBase + 0x1f) = some (.movI 0 60, 5) ∧
    voterIm (voterBase + 0x24) = some (.syscall, 2) := by
  decide +kernel

/-! ### Voter states and architectural verification -/

/-- Register file layout for the voter: length 16, copies in `rax`/`rcx`/`rbx`
(0/1/3), and syscall argument registers `2, 6, 7, 8, 9, 10` guaranteed defined. -/
def voterRegs (s : X86Int.St) (a b c : Nat) : List (Option Nat) :=
  [ some a
  , some c
  , some ((s.reg 2).getD 0)
  , some b
  , s.reg 4
  , s.reg 5
  , some ((s.reg 6).getD 0)
  , some ((s.reg 7).getD 0)
  , some ((s.reg 8).getD 0)
  , some ((s.reg 9).getD 0)
  , some ((s.reg 10).getD 0)
  , s.reg 11
  , s.reg 12
  , s.reg 13
  , s.reg 14
  , s.reg 15
  ]

/-- Flags left by `cmp v v`: `CF = false`, `ZF = true`, `SF = false`, `OF = false`. -/
def voterFlagsEq : X86Int.Flags := ⟨some false, some true, some false, some false⟩

def voterInit (s : X86Int.St) (a b c : Nat) : X86Int.St :=
  { s with
    regs := voterRegs s a b c
    rip := voterBase
    fl := X86Int.Flags.undef }

def voterDone (s : X86Int.St) (v : Nat) : X86Int.St :=
  { s with
    regs := voterRegs s v v v
    rip := voterBase + 0x18
    fl := voterFlagsEq }

theorem voterInit_regs_length (s : X86Int.St) (a b c : Nat) :
    (voterInit s a b c).regs.length = 16 := rfl

theorem voterDone_regs_length (s : X86Int.St) (v : Nat) :
    (voterDone s v).regs.length = 16 := rfl

theorem voterDone_idem (s : X86Int.St) (v w : Nat) :
    voterDone (voterDone s v) w = voterDone s w := by
  simp [voterDone, voterRegs, St.reg]

theorem voterDone_rip (s : X86Int.St) (v : Nat) :
    (voterDone s v).rip = 0x400018 := rfl

theorem voterDone_fl (s : X86Int.St) (v : Nat) :
    (voterDone s v).fl = ⟨some false, some true, some false, some false⟩ := rfl

theorem voterDone_reg0 (s : X86Int.St) (v : Nat) : (voterDone s v).reg 0 = some v := rfl
theorem voterDone_reg1 (s : X86Int.St) (v : Nat) : (voterDone s v).reg 1 = some v := rfl
theorem voterDone_reg3 (s : X86Int.St) (v : Nat) : (voterDone s v).reg 3 = some v := rfl

private theorem aluF_sub_eq (v : Nat) :
    (X86Int.aluF .sub v v).2 = voterFlagsEq := by
  simp only [X86Int.aluF, X86Int.msb, voterFlagsEq]
  have hsub : (v + X86Env.W - v) % X86Env.W = 0 := by
    rw [Nat.add_sub_cancel_left, Nat.mod_self]
  simp

private theorem aluF_sub_zf_ne {x y : Nat} (hx : x < X86Env.W) (hy : y < X86Env.W) (hne : x ≠ y) :
    (X86Int.aluF .sub x y).2.zf = some false := by
  simp only [X86Int.aluF]
  have hpos : 0 < x + X86Env.W - y := by omega
  have hlt : x + X86Env.W - y < 2 * X86Env.W := by omega
  have hneW : x + X86Env.W - y ≠ X86Env.W := by omega
  have hnz : (x + X86Env.W - y) % X86Env.W ≠ 0 := by
    intro hmod
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hmod
    rcases k with _ | _ | k
    · simp at hk; omega
    · simp at hk; exact hneW hk
    · have hk2 : 2 ≤ k + 2 := by omega
      have hge : 2 * X86Env.W ≤ X86Env.W * (k + 2) := by
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left X86Env.W hk2
      have hkeq : x + X86Env.W - y = X86Env.W * (k + 2) := hk
      omega
  simp [hnz]

private theorem run_exit (im : Nat → Option (X86Int.Ins × Nat)) (c : Nat) :
    ∀ n, X86Int.run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ n, X86Int.run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_add (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ m n o, X86Int.run im (m + n) o = X86Int.run im n (X86Int.run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [X86Int.run, run_add im m n (X86Int.step im s)]
  | _ + 1, n, .exit c => by simp [X86Int.run, run_exit]
  | _ + 1, n, .stuck => by simp [X86Int.run, run_stuck]

private theorem step_park (s : X86Int.St) (v : Nat) :
    X86Int.step voterIm (voterDone s v) = .ok (voterDone s v) := by
  simp [X86Int.step, voterIm_18, X86Int.exec, voterDone]

private theorem run_park (s : X86Int.St) (v : Nat) :
    ∀ k, X86Int.run voterIm k (.ok (voterDone s v)) = .ok (voterDone s v)
  | 0 => rfl
  | k + 1 => by simp only [X86Int.run, step_park, run_park]

private theorem step_win_a (s : X86Int.St) (v b c : Nat) :
    X86Int.run voterIm 2 (.ok { s with regs := voterRegs s v b c, rip := voterBase + 0x12, fl := voterFlagsEq }) =
      .ok (voterDone s v) := by
  simp [X86Int.run, X86Int.step, voterIm_12, voterIm_15, X86Int.exec, lift, bind, Option.bind, pure,
    St.reg, St.set, voterRegs, voterDone]

private theorem step_00_ne (s : X86Int.St) {a b c : Nat}
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hab : a ≠ b) :
    X86Int.run voterIm 2 (.ok (voterInit s a b c)) =
      .ok { s with regs := voterRegs s a b c, rip := voterBase + 0x05, fl := (X86Int.aluF .sub a b).2 } := by
  have hzf : (X86Int.aluF .sub a b).2.zf = some false := aluF_sub_zf_ne ha hb hab
  rw [X86Int.run, X86Int.step]
  dsimp only [voterInit]
  rw [voterIm_00]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_03]
  simp only [X86Int.exec, X86Int.cond, hzf, lift, bind, Option.bind, pure]
  rfl

private theorem step_05_ne (s : X86Int.St) {a b c : Nat} (fl : X86Int.Flags)
    (ha : a < X86Env.W) (hc : c < X86Env.W) (hac : a ≠ c) :
    X86Int.run voterIm 2 (.ok { s with regs := voterRegs s a b c, rip := voterBase + 0x05, fl := fl }) =
      .ok { s with regs := voterRegs s a b c, rip := voterBase + 0x0a, fl := (X86Int.aluF .sub a c).2 } := by
  have hzf : (X86Int.aluF .sub a c).2.zf = some false := aluF_sub_zf_ne ha hc hac
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_05]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_08]
  simp only [X86Int.exec, X86Int.cond, hzf, lift, bind, Option.bind, pure]
  rfl

/-- Architectural correctness of the 38-byte machine-code voter on majority:
for all 64-bit words `a b c < W` and any base state `s`, 9 steps of `voterIm`
from `voterInit s a b c` land on `voterDone s v`. -/
theorem voter_arch_majority (s : X86Int.St) {a b c v : Nat}
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some v) :
    X86Int.run voterIm 9 (.ok (voterInit s a b c)) = .ok (voterDone s v) := by
  by_cases hab : a = b
  · subst a
    have hveq : v = b := by
      unfold TMR.vote at hv
      simp at hv
      exact hv.symm
    subst v
    rw [show (9 : Nat) = 2 + (2 + 5) from rfl, run_add, run_add]
    have h1 : X86Int.run voterIm 2 (.ok (voterInit s b b c)) =
        .ok { s with regs := voterRegs s b b c, rip := voterBase + 0x12, fl := voterFlagsEq } := by
      rw [X86Int.run, X86Int.step]
      dsimp only [voterInit]
      rw [voterIm_00]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
      rw [X86Int.run, X86Int.step]
      dsimp only
      rw [voterIm_03]
      simp only [X86Int.exec, X86Int.cond, voterFlagsEq, lift, bind, Option.bind, pure]
      rfl
    rw [h1, step_win_a, run_park]
  · by_cases hac : a = c
    · subst a
      have hveq : v = c := by
        unfold TMR.vote at hv
        simp at hv
        exact hv.symm
      subst v
      rw [show (9 : Nat) = 2 + (2 + (2 + 3)) from rfl, run_add]
      rw [step_00_ne s ha hb hab, run_add]
      have h2 : X86Int.run voterIm 2
          (.ok { s with regs := voterRegs s c b c, rip := voterBase + 0x05, fl := (X86Int.aluF .sub c b).2 }) =
          .ok { s with regs := voterRegs s c b c, rip := voterBase + 0x12, fl := voterFlagsEq } := by
        rw [X86Int.run, X86Int.step]
        dsimp only
        rw [voterIm_05]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
        rw [X86Int.run, X86Int.step]
        dsimp only
        rw [voterIm_08]
        simp only [X86Int.exec, X86Int.cond, voterFlagsEq, lift, bind, Option.bind, pure]
        rfl
      rw [h2, run_add, step_win_a, run_park]
    · have hbc : b = c := by
        unfold TMR.vote at hv
        simp [hab, hac] at hv
        exact hv.1
      subst b
      have hveq : v = c := by
        unfold TMR.vote at hv
        simp [hab] at hv
        exact hv.symm
      subst v
      rw [show (9 : Nat) = 2 + (2 + (2 + (1 + 2))) from rfl, run_add]
      rw [step_00_ne s ha hb hab, run_add]
      rw [step_05_ne s (X86Int.aluF .sub a c).2 ha hc hac, run_add]
      have h3 : X86Int.run voterIm 2
          (.ok { s with regs := voterRegs s a c c, rip := voterBase + 0x0a, fl := (X86Int.aluF .sub a c).2 }) =
          .ok { s with regs := voterRegs s a c c, rip := voterBase + 0x0f, fl := voterFlagsEq } := by
        rw [X86Int.run, X86Int.step]
        dsimp only
        rw [voterIm_0a]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
        rw [X86Int.run, X86Int.step]
        dsimp only
        rw [voterIm_0d]
        simp only [X86Int.exec, X86Int.cond, voterFlagsEq, lift, bind, Option.bind, pure]
        rfl
      rw [h3, run_add]
      have h4 : X86Int.run voterIm 1
          (.ok { s with regs := voterRegs s a c c, rip := voterBase + 0x0f, fl := voterFlagsEq }) =
          .ok { s with regs := voterRegs s c c c, rip := voterBase + 0x12, fl := voterFlagsEq } := by
        rw [X86Int.run, X86Int.step]
        dsimp only
        rw [voterIm_0f]
        simp [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, voterRegs, X86Int.run]
      rw [h4, step_win_a]

/-- Architectural correctness of the 38-byte machine-code voter on refusal:
when all three 64-bit words differ (`TMR.vote a b c = none`), 9 steps of
`voterIm` from `voterInit s a b c` execute `mov edi, 3; mov eax, 60; syscall`
and exit with code 3. -/
theorem voter_arch_refuse (s : X86Int.St) {a b c : Nat}
    (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = none) :
    X86Int.run voterIm 9 (.ok (voterInit s a b c)) = .exit 3 := by
  have hab : a ≠ b := by
    intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hac : a ≠ c := by
    intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hbc : b ≠ c := by
    intro h; subst h; unfold TMR.vote at hv; simp [hab] at hv
  have hzf : (X86Int.aluF .sub b c).2.zf = some false := aluF_sub_zf_ne hb hc hbc
  rw [show (9 : Nat) = 2 + (2 + (2 + 3)) from rfl, run_add]
  rw [step_00_ne s ha hb hab, run_add]
  rw [step_05_ne s (X86Int.aluF .sub a b).2 ha hc hac, run_add]
  have h3 : X86Int.run voterIm 2
      (.ok { s with regs := voterRegs s a b c, rip := voterBase + 0x0a, fl := (X86Int.aluF .sub a c).2 }) =
      .ok { s with regs := voterRegs s a b c, rip := voterBase + 0x1a, fl := (X86Int.aluF .sub b c).2 } := by
    rw [X86Int.run, X86Int.step]
    dsimp only
    rw [voterIm_0a]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, voterRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]
    dsimp only
    rw [voterIm_0d]
    simp only [X86Int.exec, X86Int.cond, hzf, lift, bind, Option.bind, pure]
    rfl
  rw [h3]
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_1a]
  simp only [X86Int.exec, St.set, voterRegs, List.set]
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_1f]
  simp only [X86Int.exec, St.set, List.set]
  rw [X86Int.run, X86Int.step]
  dsimp only
  rw [voterIm_24]
  simp only [X86Int.exec, St.reg, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some, X86Env.syscall]
  rfl

/-! ### Concrete `TMR.Channel` instantiations for `X86Int` -/

/-- Lift `X86Int.step` to `X86Int.Out → X86Int.Out` (halted/stuck outcomes absorb). -/
def outStep (im : Nat → Option (X86Int.Ins × Nat)) : X86Int.Out → X86Int.Out
  | .ok s => X86Int.step im s
  | o => o

/-- Lift `X86Int.pstep` to `X86Int.Out → PDist.PDF X86Int.Out`. -/
def outPStep (alts : X86Int.Alts) (F : X86Int.Fault) (pf : Rat)
    (im : Nat → Option (X86Int.Ins × Nat)) : X86Int.Out → PDist.PDF X86Int.Out
  | .ok s => X86Int.pstep alts F pf im s
  | o => PDist.pureF o

theorem rates_state_q (r : X86Int.Rates) (hr : r.ok) :
    r.fetch + (r.state hr).q = r.total := rfl

theorem rates_total_nonneg (r : X86Int.Rates) (hr : r.ok) : 0 ≤ r.total :=
  X86Int.nonneg_add hr.1.1 (X86Int.nonneg_add hr.2.1.1 (X86Int.nonneg_add hr.2.2.1.1 hr.2.2.2.1))

/-- Concrete `TMR.Channel` for the full `X86Int.Out` machine semantics under
`X86Int.pstep` and `X86Int.Rates`. -/
def x86Channel (alts : X86Int.Alts) (im : Nat → Option (X86Int.Ins × Nat))
    (r : X86Int.Rates) (hr : r.ok) : TMR.Channel X86Int.Out where
  f := outStep im
  step := outPStep alts (r.state hr) r.fetch im
  q := r.total
  q0 := rates_total_nonneg r hr
  ok o := by
    cases o with
    | ok s => exact X86Int.pstep_ok alts (r.state hr) hr.1 im s
    | exit c => exact PDist.pureF_ok _
    | stuck => exact PDist.pureF_ok _
  clean o := by
    cases o with
    | ok s => exact X86Int.faultFree_pstep alts (r.state hr) r.fetch im s
    | exit c => rfl
    | stuck => rfl
  le o := by
    cases o with
    | ok s => exact X86Int.fm_pstep alts (r.state hr) hr.1 im s
    | exit c =>
      change PDist.fm (PDist.pureF (X86Int.Out.exit c)) ≤ r.total
      rw [PDist.fm_pureF]; exact rates_total_nonneg r hr
    | stuck =>
      change PDist.fm (PDist.pureF X86Int.Out.stuck) ≤ r.total
      rw [PDist.fm_pureF]; exact rates_total_nonneg r hr

/-- State-level projection fallback for `X86Int.St`. -/
def stOfOut (fallback : X86Int.St) : X86Int.Out → X86Int.St
  | .ok s => s
  | _ => fallback

/-- Concrete `TMR.Channel` for `X86Int.St` under `X86Int.pstep` and `X86Int.Rates`. -/
def x86StChannel (alts : X86Int.Alts) (im : Nat → Option (X86Int.Ins × Nat))
    (r : X86Int.Rates) (hr : r.ok) : TMR.Channel X86Int.St where
  f s := stOfOut s (X86Int.step im s)
  step s := X86Int.relabel (stOfOut s) (X86Int.pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok s := X86Int.relabel_ok _ (X86Int.pstep_ok alts (r.state hr) hr.1 im s)
  clean s := by rw [X86Int.faultFree_relabel, X86Int.faultFree_pstep]; rfl
  le s := by rw [X86Int.fm_relabel]; exact X86Int.fm_pstep alts (r.state hr) hr.1 im s

/-- Per-register projection channel of `X86Int.St` under `X86Int.pstep` and `X86Int.Rates`. -/
def x86RegProjChannel (alts : X86Int.Alts) (im : Nat → Option (X86Int.Ins × Nat))
    (r : X86Int.Rates) (hr : r.ok) (s0 : X86Int.St) (regIdx : Nat) : TMR.Channel Nat where
  f v :=
    let s := s0.set regIdx (some v)
    ((stOfOut s (X86Int.step im s)).reg regIdx).getD v
  step v :=
    let s := s0.set regIdx (some v)
    X86Int.relabel (fun o => ((stOfOut s o).reg regIdx).getD v)
      (X86Int.pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok v := X86Int.relabel_ok _ (X86Int.pstep_ok alts (r.state hr) hr.1 im (s0.set regIdx (some v)))
  clean v := by rw [X86Int.faultFree_relabel, X86Int.faultFree_pstep]; rfl
  le v := by rw [X86Int.fm_relabel]; exact X86Int.fm_pstep alts (r.state hr) hr.1 im (s0.set regIdx (some v))

/-- Per-memory-byte projection channel of `X86Int.St` under `X86Int.pstep` and `X86Int.Rates`
for a writable address `addr` in `s0.mem`. -/
def x86MemProjChannel (alts : X86Int.Alts) (im : Nat → Option (X86Int.Ins × Nat))
    (r : X86Int.Rates) (hr : r.ok) (s0 : X86Int.St) (addr : Nat) : TMR.Channel Nat where
  f v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    ((stOfOut s (X86Int.step im s)).mem.rd addr).getD v
  step v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    X86Int.relabel (fun o => ((stOfOut s o).mem.rd addr).getD v)
      (X86Int.pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    X86Int.relabel_ok _ (X86Int.pstep_ok alts (r.state hr) hr.1 im s)
  clean v := by rw [X86Int.faultFree_relabel, X86Int.faultFree_pstep]; rfl
  le v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    by rw [X86Int.fm_relabel]; exact X86Int.fm_pstep alts (r.state hr) hr.1 im s

/-- Per-register 64-bit word channel under `Rates.reg` single-bit upsets. -/
def regWordChannel (fn : Nat → Nat) (r : X86Int.Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v := X86Int.unif r.reg (fn v) ((List.range 64).map fun b => Nat.xor (fn v) (2 ^ b))
  q := r.reg
  q0 := hr.2.1.1
  ok _ := X86Int.unif_ok hr.2.1 _ _
  clean _ := X86Int.faultFree_unif _ _ _
  le _ := X86Int.fm_unif_le hr.2.1 _ _

/-- Per-byte memory channel under `Rates.mem` single-bit upsets. -/
def memByteChannel (fn : Nat → Nat) (r : X86Int.Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v := X86Int.unif r.mem (fn v) ((List.range 8).map fun b => Nat.xor (fn v) (2 ^ b))
  q := r.mem
  q0 := hr.2.2.1.1
  ok _ := X86Int.unif_ok hr.2.2.1 _ _
  clean _ := X86Int.faultFree_unif _ _ _
  le _ := X86Int.fm_unif_le hr.2.2.1 _ _

private theorem mem_unif {β : Type} {p : Rat} {c : β} {xs : List β} {y : (β × Bool) × Rat}
    (hy : y ∈ X86Int.unif p c xs) : y.1.1 = c ∨ y.1.1 ∈ xs := by
  cases xs with
  | nil =>
    simp only [X86Int.unif, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [X86Int.unif, List.mem_cons, List.mem_map] at hy
    rcases hy with h | ⟨z, hz, h⟩
    · rw [h]; exact Or.inl rfl
    · rw [← h]; exact Or.inr (by simpa using hz)

theorem regWordChannel_lt_W {fn : Nat → Nat} (hfn : ∀ v, v < X86Env.W → fn v < X86Env.W)
    (r : X86Int.Rates) (hr : r.ok) {v : Nat} (hv : v < X86Env.W) :
    ∀ y ∈ (regWordChannel fn r hr).step v, y.1.1 < X86Env.W := by
  intro y hy
  dsimp only [regWordChannel] at hy
  rcases mem_unif hy with heq | hmem
  · rw [heq]; exact hfn v hv
  · simp only [List.mem_map, List.mem_range] at hmem
    obtain ⟨b, hb, hxor⟩ := hmem
    rw [← hxor]
    exact Nat.xor_lt_two_pow (hfn v hv) (Nat.pow_lt_pow_right (by decide) hb)

/-! ### Faulty-silicon machine-code voter theorems -/

/-- 9 physical cycles of the 38-byte machine-code voter from `voterInit s a b c`. -/
def voterPhys (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (a b c : Nat) :
    PDist.PDF X86Int.Out :=
  X86Int.prun voterAlts (r.state hr) r.fetch voterIm 9 (.ok (voterInit s a b c))

theorem voter_phys_ok (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (a b c : Nat) :
    (voterPhys r hr s a b c).ok :=
  X86Int.prun_ok voterAlts (r.state hr) hr.1 voterIm 9 (.ok (voterInit s a b c))

theorem voter_phys_clean (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (a b c : Nat) :
    PDist.faultFree (voterPhys r hr s a b c) =
      [X86Int.run voterIm 9 (.ok (voterInit s a b c))] :=
  X86Int.faultFree_prun voterAlts (r.state hr) r.fetch voterIm 9 (.ok (voterInit s a b c))

theorem voter_phys_clean_majority (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St)
    {a b c v : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some v) :
    PDist.faultFree (voterPhys r hr s a b c) = [.ok (voterDone s v)] := by
  rw [voter_phys_clean, voter_arch_majority s ha hb hc hv]

theorem voter_phys_clean_refuse (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St)
    {a b c : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = none) :
    PDist.faultFree (voterPhys r hr s a b c) = [.exit 3] := by
  rw [voter_phys_clean, voter_arch_refuse s ha hb hc hv]

theorem voter_phys_bound (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (a b c : Nat) :
    PDist.fm (voterPhys r hr s a b c) ≤ 9 * r.total := by
  simpa [voterPhys, rates_state_q] using
    X86Int.fm_prun voterAlts (r.state hr) hr.1 voterIm 9 (.ok (voterInit s a b c))

theorem voter_phys_bound_iv (cf cr cm cl : PDist.Iv) (r : X86Int.Rates) (hr : r.ok)
    (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem) (hl : cl.mem r.flag)
    (s : X86Int.St) (a b c : Nat) :
    PDist.fm (voterPhys r hr s a b c) ≤ 9 * (cf.hi + (cr.hi + (cm.hi + cl.hi))) :=
  X86Int.baseline voterAlts voterIm cf cr cm cl r hr hf hg hm hl 9 (.ok (voterInit s a b c))

/-! ### Triplicated body step followed by the 9-step physical machine-code voter -/

/-- Three independent draws from a body distribution `D` (with target clean
outcome `g`), feeding their actual words `(a, b, c)` into `voterInit s a b c`
and running 9 physical cycles of `voterIm`. Any disagreement with `some g` on
the majority vote marks the step faulty (`decide (vote a b c ≠ some g)`). -/
def votedStepGen (D : PDist.PDF Nat) (g : Nat) (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) :
    PDist.PDF X86Int.Out :=
  TMR.bindW D fun a =>
    TMR.bindW D fun b =>
      TMR.bindW D fun c =>
        PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1
          (voterPhys r hr s a.1 b.1 c.1)

theorem votedStepGen_ok {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) :
    (votedStepGen D g r hr s).ok := by
  refine ⟨?_, ?_⟩
  · refine TMR.bindW_nonneg hD.1 fun a => TMR.bindW_nonneg hD.1 fun b =>
      TMR.bindW_nonneg hD.1 fun c => ?_
    intro y hy
    simp only [PDist.tagW, List.mem_map] at hy
    obtain ⟨z, hz, rfl⟩ := hy
    have hz0 : 0 ≤ z.2 := (voter_phys_ok r hr s a.1 b.1 c.1).1 z hz
    simpa [Rat.one_mul] using hz0
  · have hm : ∀ a b c, PDist.mass
        (PDist.tagW (decide (TMR.vote a b c ≠ some g)) 1 (voterPhys r hr s a b c)) = 1 := by
      intro a b c
      rw [PDist.mass_tagW, (voter_phys_ok r hr s a b c).2, Rat.one_mul]
    simp only [votedStepGen, TMR.mass_bindW, hm, TMR.sumW_const, hD.2, Rat.one_mul]

theorem votedStepGen_clean {D : PDist.PDF Nat} (hW : ∀ y ∈ D, y.1.1 < X86Env.W) (g : Nat)
    (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) :
    ∀ u ∈ PDist.faultFree (votedStepGen D g r hr s), u = .ok (voterDone s g) := by
  intro u hu
  obtain ⟨a, ha, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨b, hb, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨c, hc, hu⟩ := TMR.mem_faultFree_bindW hu
  rw [PDist.faultFree_tagW] at hu
  split at hu
  · simp at hu
  · next hne =>
    have hv : TMR.vote a.1.1 b.1.1 c.1.1 = some g := by
      simpa using hne
    rw [voter_phys_clean_majority r hr s (hW a ha) (hW b hb) (hW c hc) hv] at hu
    simpa using hu

private theorem sumW_add_const {α : Type} (D : PDist.PDF α) (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + PDist.mass D * k := by
  induction D with
  | nil => show (0 : Rat) = 0 + 0 * k; rw [Rat.zero_mul, Rat.add_zero]
  | cons x r ih =>
    show x.2 * (φ x.1 + k) + TMR.sumW r (fun y => φ y + k) =
      (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 + PDist.mass r) * k
    rw [ih, Rat.mul_add, Rat.add_mul]
    calc (x.2 * φ x.1 + x.2 * k) + (TMR.sumW r φ + PDist.mass r * k)
        = x.2 * φ x.1 + (x.2 * k + (TMR.sumW r φ + PDist.mass r * k)) := Rat.add_assoc _ _ _
      _ = x.2 * φ x.1 + ((x.2 * k + TMR.sumW r φ) + PDist.mass r * k) := by rw [Rat.add_assoc (x.2 * k)]
      _ = x.2 * φ x.1 + ((TMR.sumW r φ + x.2 * k) + PDist.mass r * k) := by rw [Rat.add_comm (x.2 * k)]
      _ = x.2 * φ x.1 + (TMR.sumW r φ + (x.2 * k + PDist.mass r * k)) := by rw [Rat.add_assoc (TMR.sumW r φ)]
      _ = (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 * k + PDist.mass r * k) := (Rat.add_assoc _ _ _).symm

private theorem sumW_add_const_ok {α : Type} {D : PDist.PDF α} (hD : D.ok)
    (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + k := by
  rw [sumW_add_const, hD.2, Rat.one_mul]

private theorem fm_tri_eq_sumW {σ : Type} [DecidableEq σ] (D : PDist.PDF σ) (g : σ) :
    PDist.fm (TMR.tri D g) =
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))))) := by
  have hm : ∀ v : Option σ, PDist.fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) := fun v => by
    show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
    rw [Rat.add_zero]; rfl
  simp only [TMR.tri, TMR.fm_bindW, hm]

private theorem exists_clean_of_faultFree_eq {α : Type} {D : PDist.PDF α} {g : α}
    (h : PDist.faultFree D = [g]) : ∃ a ∈ D, a.1.1 = g ∧ a.1.2 = false := by
  induction D with
  | nil => simp [PDist.faultFree] at h
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    cases t
    · simp only [PDist.faultFree, Bool.false_eq_true, ↓reduceIte, List.cons.injEq] at h
      exact ⟨((a, false), w), List.mem_cons_self .., h.1, rfl⟩
    · simp only [PDist.faultFree, ↓reduceIte] at h
      obtain ⟨y, hy, ha, ht⟩ := ih h
      exact ⟨y, List.mem_cons_of_mem _ hy, ha, ht⟩

private theorem mem_faultFree_bindW_intro {α β : Type} {D : PDist.PDF α}
    {K : α × Bool → PDist.PDF β} {a : (α × Bool) × Rat} {z : β}
    (ha : a ∈ D) (hz : z ∈ PDist.faultFree (K a.1)) :
    z ∈ PDist.faultFree (TMR.bindW D K) := by
  induction D with
  | nil => simp at ha
  | cons x r ih =>
    simp only [TMR.bindW, PDist.faultFree_append, TMR.faultFree_scaleL, List.mem_append]
    rcases List.mem_cons.1 ha with rfl | hr
    · exact Or.inl hz
    · exact Or.inr (ih hr)

private theorem vote_self (g : Nat) : TMR.vote g g g = some g := by
  unfold TMR.vote; simp

theorem votedStepGen_mem_clean {D : PDist.PDF Nat} {g : Nat} (hg : g < X86Env.W)
    (hc : PDist.faultFree D = [g]) (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) :
    X86Int.Out.ok (voterDone s g) ∈ PDist.faultFree (votedStepGen D g r hr s) := by
  obtain ⟨a, ha, ha1, _⟩ := exists_clean_of_faultFree_eq hc
  have hphys : X86Int.Out.ok (voterDone s g) ∈
      PDist.faultFree (PDist.tagW (decide (TMR.vote a.1.1 a.1.1 a.1.1 ≠ some g)) 1
        (voterPhys r hr s a.1.1 a.1.1 a.1.1)) := by
    rw [ha1]
    have htag : decide (TMR.vote g g g ≠ some g) = false := by simp [vote_self]
    rw [PDist.faultFree_tagW, if_neg (by simp [htag]),
      voter_phys_clean_majority r hr s hg hg hg (vote_self g)]
    exact List.mem_singleton.2 rfl
  exact mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha hphys))

theorem votedStepGen_bound_le {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) {qv : Rat} (hqv0 : 0 ≤ qv)
    (hqv : ∀ a b c, PDist.fm (voterPhys r hr s a b c) ≤ qv) :
    PDist.fm (votedStepGen D g r hr s) ≤ PDist.fm (TMR.tri D g) + qv := by
  rw [votedStepGen, TMR.fm_bindW]
  simp only [TMR.fm_bindW]
  have hcell : ∀ a b c : Nat,
      PDist.fm (PDist.tagW (decide (TMR.vote a b c ≠ some g)) 1 (voterPhys r hr s a b c)) ≤
        TMR.ι (decide (TMR.vote a b c ≠ some g)) + qv := by
    intro a b c
    rw [PDist.fm_tagW]
    by_cases ht : decide (TMR.vote a b c ≠ some g) = true
    · rw [if_pos ht, Rat.one_mul, (voter_phys_ok r hr s a b c).2]
      simp only [TMR.ι, ht, ↓reduceIte]
      have := add_le_add' (Rat.le_refl (a := (1 : Rat))) hqv0
      rwa [Rat.add_zero] at this
    · have hfalse : decide (TMR.vote a b c ≠ some g) = false := by
        cases h : decide (TMR.vote a b c ≠ some g) <;> simp_all
      rw [if_neg ht, Rat.one_mul]
      simp only [TMR.ι, hfalse, Bool.false_eq_true, ↓reduceIte, Rat.zero_add]
      exact hqv a b c
  have h1 :
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        PDist.fm (PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1
          (voterPhys r hr s a.1 b.1 c.1))))) ≤
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv))) := by
    refine TMR.sumW_le hD.1 fun a _ => TMR.sumW_le hD.1 fun b _ => TMR.sumW_le hD.1 fun c _ => ?_
    exact hcell a.1.1 b.1.1 c.1.1
  refine Rat.le_trans h1 ?_
  have hinner : ∀ a b : Nat × Bool,
      TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv) =
        TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))) + qv :=
    fun a b => sumW_add_const_ok hD _ _
  have hmid : ∀ a : Nat × Bool,
      TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv)) =
        TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)))) + qv := by
    intro a
    rw [TMR.sumW_congr D (fun b => hinner a b), sumW_add_const_ok hD]
  rw [TMR.sumW_congr D hmid, sumW_add_const_ok hD, fm_tri_eq_sumW]
  exact Rat.le_refl

theorem votedStepGen_bound {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) :
    PDist.fm (votedStepGen D g r hr s) ≤ PDist.fm (TMR.tri D g) + 9 * r.total :=
  votedStepGen_bound_le hD g r hr s
    (Rat.mul_nonneg (by decide) (rates_total_nonneg r hr))
    (fun a b c => voter_phys_bound r hr s a b c)

theorem votedStepGen_bound_iv {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    (cf cr cm cl : PDist.Iv) (hcf : cf.prob) (hcr : cr.prob) (hcm : cm.prob) (hcl : cl.prob)
    (r : X86Int.Rates) (hr : r.ok)
    (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem) (hl : cl.mem r.flag)
    (s : X86Int.St) :
    PDist.fm (votedStepGen D g r hr s) ≤
      PDist.fm (TMR.tri D g) + 9 * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  have hi0 : 0 ≤ cf.hi + (cr.hi + (cm.hi + cl.hi)) :=
    X86Int.nonneg_add (Rat.le_trans hcf.1 hcf.2.1)
      (X86Int.nonneg_add (Rat.le_trans hcr.1 hcr.2.1)
        (X86Int.nonneg_add (Rat.le_trans hcm.1 hcm.2.1) (Rat.le_trans hcl.1 hcl.2.1)))
  exact votedStepGen_bound_le hD g r hr s (Rat.mul_nonneg (by decide) hi0)
    (fun a b c => voter_phys_bound_iv cf cr cm cl r hr hf hg hm hl s a b c)

/-- One triplicated body step on channel `C` followed by the 9-step physical
machine-code voter. -/
def votedStep (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok) (x : Nat) (s : X86Int.St) :
    PDist.PDF X86Int.Out :=
  votedStepGen (C.step x) (C.f x) r hr s

theorem votedStep_err (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (x : Nat) (s : X86Int.St) :
    PDist.fm (votedStep C r hr x s) ≤ 3 * (C.q * C.q) + 9 * r.total :=
  Rat.le_trans (votedStepGen_bound (C.ok x) (C.f x) r hr s)
    (add_le_add' (TMR.tri_err C x) Rat.le_refl)

/-- A triplicated body step (rate `C.q`) followed by the 9-step physical
machine-code voter is a valid distribution, lands cleanly and non-vacuously on
`voterDone s (C.f x)` whenever copy outcomes stay 64-bit, and faults with
probability at most `3 * (C.q * C.q) + 9 * r.total`. -/
theorem voted_step_with_faulty_voter_bound (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (x : Nat) (s : X86Int.St)
    (hfW : C.f x < X86Env.W) (hW : ∀ y ∈ C.step x, y.1.1 < X86Env.W) :
    (votedStep C r hr x s).ok ∧
    X86Int.Out.ok (voterDone s (C.f x)) ∈ PDist.faultFree (votedStep C r hr x s) ∧
    (∀ u ∈ PDist.faultFree (votedStep C r hr x s), u = .ok (voterDone s (C.f x))) ∧
    PDist.fm (votedStep C r hr x s) ≤ 3 * (C.q * C.q) + 9 * r.total :=
  ⟨votedStepGen_ok (C.ok x) (C.f x) r hr s,
    votedStepGen_mem_clean hfW (C.clean x) r hr s,
    votedStepGen_clean hW (C.f x) r hr s,
    votedStep_err C r hr x s⟩

/-- Majority-outcome feeding bridge: running `TMR.tri (C.step x) (C.f x)` and
feeding any winning majority `some v` into `voterInit s v v v` (or refusal `none`
into `voterInit s 0 1 2`) followed by 9 physical cycles of `voterIm`. -/
def votedTriFeed (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok) (x : Nat) (s : X86Int.St) :
    PDist.PDF X86Int.Out :=
  PDist.bindF (TMR.tri (C.step x) (C.f x)) fun o =>
    match o with
    | some v => voterPhys r hr s v v v
    | none => voterPhys r hr s 0 1 2

theorem voted_tri_feed_bound (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (x : Nat) (s : X86Int.St) (hx : C.f x < X86Env.W) :
    (votedTriFeed C r hr x s).ok ∧
    (∀ u ∈ PDist.faultFree (votedTriFeed C r hr x s), u = .ok (voterDone s (C.f x))) ∧
    PDist.fm (votedTriFeed C r hr x s) ≤ 3 * (C.q * C.q) + 9 * r.total := by
  have hcont_ok : ∀ o : Option Nat,
      (match o with | some v => voterPhys r hr s v v v | none => voterPhys r hr s 0 1 2).ok := by
    intro o; cases o <;> exact voter_phys_ok r hr s _ _ _
  have hcont_le : ∀ o : Option Nat,
      PDist.fm (match o with | some v => voterPhys r hr s v v v | none => voterPhys r hr s 0 1 2) ≤
        9 * r.total := by
    intro o; cases o <;> exact voter_phys_bound r hr s _ _ _
  refine ⟨PDist.bindF_ok (TMR.tri_ok (C.ok x) (C.f x)) hcont_ok, ?_, ?_⟩
  · intro u hu
    simp only [votedTriFeed, PDist.faultFree_bindF, List.mem_flatMap] at hu
    obtain ⟨o, ho, hu⟩ := hu
    rw [TMR.tri_clean (C.f x) o ho] at hu
    rw [voter_phys_clean_majority r hr s hx hx hx (vote_self (C.f x))] at hu
    simpa using hu
  · have hrt : 0 ≤ 9 * r.total := Rat.mul_nonneg (by decide) (rates_total_nonneg r hr)
    exact Rat.le_trans
      (PDist.fm_bindF_le' (TMR.tri_ok (C.ok x) (C.f x)) hrt hcont_ok hcont_le)
      (add_le_add' (TMR.tri_err C x) Rat.le_refl)

/-- Exposure-weighted version via `TMRExposure.exposed`. -/
def votedStepExposed (C : TMR.Channel Nat) (k : Nat) (r : X86Int.Rates) (hr : r.ok)
    (x : Nat) (s : X86Int.St) : PDist.PDF X86Int.Out :=
  votedStepGen (TMR.exposed C k x) (TMR.iter C.f (k + 1) x) r hr s

private theorem mem_bindF {α β : Type} {d : PDist.PDF α} {f : α → PDist.PDF β}
    {y : (β × Bool) × Rat} (hy : y ∈ PDist.bindF d f) :
    ∃ x ∈ d, ∃ z ∈ f x.1.1, y.1.1 = z.1.1 := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x r ih =>
    simp only [PDist.bindF, List.mem_append] at hy
    rcases hy with hy | hy
    · simp only [PDist.tagW, List.mem_map] at hy
      obtain ⟨z, hz, rfl⟩ := hy
      exact ⟨x, List.mem_cons_self .., z, hz, rfl⟩
    · obtain ⟨w, hw, z, hz, hyeq⟩ := ih hy
      exact ⟨w, List.mem_cons_of_mem _ hw, z, hz, hyeq⟩

private theorem idleK_lt_W (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < X86Env.W → ∀ y ∈ C.step z, y.1.1 < X86Env.W) :
    ∀ k x, x < X86Env.W → ∀ y ∈ TMR.idleK C k x, y.1.1 < X86Env.W
  | 0, x, hx, y, hy => by
    simp only [TMR.idleK, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact hx
  | k + 1, x, hx, y, hy => by
    obtain ⟨w, hw, z, hz, hyeq⟩ := mem_bindF hy
    rw [hyeq]
    exact idleK_lt_W C hstepW k w.1.1 (hstepW x hx w hw) z hz

private theorem exposed_lt_W (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < X86Env.W → ∀ y ∈ C.step z, y.1.1 < X86Env.W)
    (k x : Nat) (hx : x < X86Env.W) :
    ∀ y ∈ TMR.exposed C k x, y.1.1 < X86Env.W := by
  intro y hy
  obtain ⟨w, hw, z, hz, hyeq⟩ := mem_bindF hy
  rw [hyeq]
  exact hstepW w.1.1 (idleK_lt_W C hstepW k x hx w hw) z hz

private theorem iter_lt_W {f : Nat → Nat} (hf : ∀ z, z < X86Env.W → f z < X86Env.W) :
    ∀ k x, x < X86Env.W → TMR.iter f k x < X86Env.W
  | 0, _, hx => hx
  | k + 1, x, hx => iter_lt_W hf k (f x) (hf x hx)

/-- Exposure-weighted triplicated body step (`TMRExposure.exposed`) followed by
the 9-step physical machine-code voter. -/
theorem voted_step_exposed_with_faulty_voter_bound (C : TMR.Channel Nat) (k : Nat)
    (r : X86Int.Rates) (hr : r.ok) (x : Nat) (s : X86Int.St)
    (hx : x < X86Env.W)
    (hW : ∀ z, z < X86Env.W → C.f z < X86Env.W ∧ ∀ y ∈ C.step z, y.1.1 < X86Env.W) :
    (votedStepExposed C k r hr x s).ok ∧
    X86Int.Out.ok (voterDone s (TMR.iter C.f (k + 1) x)) ∈
      PDist.faultFree (votedStepExposed C k r hr x s) ∧
    (∀ u ∈ PDist.faultFree (votedStepExposed C k r hr x s),
      u = .ok (voterDone s (TMR.iter C.f (k + 1) x))) ∧
    PDist.fm (votedStepExposed C k r hr x s) ≤
      3 * (((k + 1 : Nat) : Rat) * C.q) ^ 2 + 9 * r.total := by
  have hiter : TMR.iter C.f (k + 1) x < X86Env.W :=
    iter_lt_W (fun z hz => (hW z hz).1) (k + 1) x hx
  refine ⟨votedStepGen_ok (TMR.exposed_ok C k x) (TMR.iter C.f (k + 1) x) r hr s,
    votedStepGen_mem_clean hiter (TMR.faultFree_exposed C k x) r hr s,
    votedStepGen_clean (exposed_lt_W C (fun z hz => (hW z hz).2) k x hx)
      (TMR.iter C.f (k + 1) x) r hr s, ?_⟩
  have hpow : (((k + 1 : Nat) : Rat) * C.q) ^ 2 =
      (((k + 1 : Nat) : Rat) * C.q) * (((k + 1 : Nat) : Rat) * C.q) := by
    rw [Rat.pow_succ, Rat.pow_one]
  rw [hpow]
  exact Rat.le_trans (votedStepGen_bound (TMR.exposed_ok C k x) (TMR.iter C.f (k + 1) x) r hr s)
    (add_le_add' (TMR.exposure_err C k x) Rat.le_refl)

/-! ### Multi-step voted run with physical machine-code voter -/

/-- `n` voted steps, each running a triplicated body step on `C` followed by the
9-step physical machine-code voter `voterIm`. -/
def votedRun (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok) :
    Nat → X86Int.Out → PDist.PDF X86Int.Out
  | 0, o => PDist.pureF o
  | n + 1, .ok s =>
    match s.reg 0 with
    | some x => PDist.bindF (votedStep C r hr x s) (votedRun C r hr n)
    | none => PDist.pureF .stuck
  | _ + 1, o => PDist.pureF o

theorem votedRun_ok (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok) :
    ∀ n o, (votedRun C r hr n o).ok
  | 0, o => PDist.pureF_ok o
  | n + 1, .ok s => by
    simp only [votedRun]
    cases hreg : s.reg 0 with
    | none => exact PDist.pureF_ok _
    | some x => exact PDist.bindF_ok (votedStepGen_ok (C.ok x) (C.f x) r hr s) (votedRun_ok C r hr n)
  | _ + 1, .exit c => PDist.pureF_ok _
  | _ + 1, .stuck => PDist.pureF_ok _

theorem votedRun_clean_done (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (hW : ∀ z, z < X86Env.W → C.f z < X86Env.W ∧ ∀ y ∈ C.step z, y.1.1 < X86Env.W) :
    ∀ n (s : X86Int.St) {x : Nat}, x < X86Env.W →
      ∀ u ∈ PDist.faultFree (votedRun C r hr n (.ok (voterDone s x))),
        u = .ok (voterDone s (TMR.iter C.f n x))
  | 0, s, x, _, u, hu => by
    simpa [votedRun, PDist.faultFree_pureF, TMR.iter] using hu
  | n + 1, s, x, hx, u, hu => by
    have hreg : (voterDone s x).reg 0 = some x := rfl
    simp only [votedRun, hreg, PDist.faultFree_bindF, List.mem_flatMap] at hu
    obtain ⟨v, hv, hu⟩ := hu
    have hvEq : v = .ok (voterDone (voterDone s x) (C.f x)) :=
      votedStepGen_clean (hW x hx).2 (C.f x) r hr (voterDone s x) v hv
    rw [hvEq, voterDone_idem] at hu
    exact votedRun_clean_done C r hr hW n s (hW x hx).1 u hu

theorem votedRun_err (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok) :
    ∀ n o, PDist.fm (votedRun C r hr n o) ≤ (n : Rat) * (3 * (C.q * C.q) + 9 * r.total) := by
  have hsq : 0 ≤ 3 * (C.q * C.q) := by
    have := Rat.mul_nonneg C.q0 C.q0
    grind
  have hrt : 0 ≤ 9 * r.total := Rat.mul_nonneg (by decide) (rates_total_nonneg r hr)
  have hstepq : 0 ≤ 3 * (C.q * C.q) + 9 * r.total := X86Int.nonneg_add hsq hrt
  intro n
  induction n with
  | zero =>
    intro o
    rw [votedRun, PDist.fm_pureF]
    show (0 : Rat) ≤ ((0 : Nat) : Rat) * _
    rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
    exact Rat.le_refl
  | succ n ih =>
    intro o
    have hn : ((n + 1 : Nat) : Rat) * (3 * (C.q * C.q) + 9 * r.total) =
        (3 * (C.q * C.q) + 9 * r.total) + (n : Rat) * (3 * (C.q * C.q) + 9 * r.total) := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    have hnn : 0 ≤ (n : Rat) * (3 * (C.q * C.q) + 9 * r.total) :=
      Rat.mul_nonneg Rat.natCast_nonneg hstepq
    rw [hn]
    cases o with
    | ok s =>
      simp only [votedRun]
      cases hreg : s.reg 0 with
      | none =>
        rw [PDist.fm_pureF]
        exact X86Int.nonneg_add hstepq hnn
      | some x =>
        exact Rat.le_trans
          (PDist.fm_bindF_le' (votedStepGen_ok (C.ok x) (C.f x) r hr s) hnn
            (votedRun_ok C r hr n) ih)
          (add_le_add' (votedStep_err C r hr x s) Rat.le_refl)
    | exit c =>
      dsimp only [votedRun]
      rw [PDist.fm_pureF]
      exact X86Int.nonneg_add hstepq hnn
    | stuck =>
      dsimp only [votedRun]
      rw [PDist.fm_pureF]
      exact X86Int.nonneg_add hstepq hnn

theorem votedRun_mem_clean_done (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (hW : ∀ z, z < X86Env.W → C.f z < X86Env.W ∧ ∀ y ∈ C.step z, y.1.1 < X86Env.W) :
    ∀ n (s : X86Int.St) {x : Nat}, x < X86Env.W →
      X86Int.Out.ok (voterDone s (TMR.iter C.f n x)) ∈
        PDist.faultFree (votedRun C r hr n (.ok (voterDone s x)))
  | 0, s, x, _ => by simp [votedRun, PDist.faultFree_pureF, TMR.iter]
  | n + 1, s, x, hx => by
    have hreg : (voterDone s x).reg 0 = some x := rfl
    simp only [votedRun, hreg, PDist.faultFree_bindF, List.mem_flatMap]
    have hstep := votedStepGen_mem_clean (hW x hx).1 (C.clean x) r hr (voterDone s x)
    rw [voterDone_idem] at hstep
    exact ⟨.ok (voterDone s (C.f x)), hstep, votedRun_mem_clean_done C r hr hW n s (hW x hx).1⟩

/-- `n`-step voted run with the 9-step faulty machine-code voter at every step:
valid distribution, exact architectural `voterDone` state realized on `faultFree`,
and total fault probability at most `n * (3 * q_body² + 9 * r.total)`. -/
theorem voted_run_with_faulty_voter_bound (C : TMR.Channel Nat) (r : X86Int.Rates) (hr : r.ok)
    (hW : ∀ z, z < X86Env.W → C.f z < X86Env.W ∧ ∀ y ∈ C.step z, y.1.1 < X86Env.W)
    (n : Nat) (s : X86Int.St) {x : Nat} (hx : x < X86Env.W) :
    (votedRun C r hr n (.ok (voterDone s x))).ok ∧
    X86Int.Out.ok (voterDone s (TMR.iter C.f n x)) ∈
      PDist.faultFree (votedRun C r hr n (.ok (voterDone s x))) ∧
    (∀ u ∈ PDist.faultFree (votedRun C r hr n (.ok (voterDone s x))),
      u = .ok (voterDone s (TMR.iter C.f n x))) ∧
    PDist.fm (votedRun C r hr n (.ok (voterDone s x))) ≤
      (n : Rat) * (3 * (C.q * C.q) + 9 * r.total) :=
  ⟨votedRun_ok C r hr n _,
    votedRun_mem_clean_done C r hr hW n s hx,
    votedRun_clean_done C r hr hW n s hx,
    votedRun_err C r hr n _⟩

/-! ### Concrete `p7` (`[5·10⁻⁸, 2·10⁻⁷]`) rational witness -/

private theorem inc_mod_lt_W (v : Nat) (_ : v < X86Env.W) : (v + 1) % X86Env.W < X86Env.W :=
  Nat.mod_lt _ (by decide)

theorem p7_step_bound_rat :
    3 * ((1 / 5000000 : Rat) * (1 / 5000000)) + 9 * (4 / 5000000) = 180000003 / 25000000000000 := by
  decide +kernel

theorem p7_exposed9_bound_rat :
    3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 + 9 * (4 / 5000000) = 1800003 / 250000000000 := by
  decide +kernel

theorem p7_run1000_bound_rat :
    ((1000 : Nat) : Rat) * (180000003 / 25000000000000) = 180000003 / 25000000000 := by
  decide +kernel

/-- Concrete `p7` (`[5·10⁻⁸, 2·10⁻⁷]`) rational witness for a 64-bit register
increment channel under the 9-step faulty machine-code voter:
1. One voted step + physical voter is `PD.ok`, cleans to `voterDone s ((x + 1) % W)`
   (non-vacuously), and faults with probability `≤ 180000003 / 25000000000000`.
2. With 9-step idle exposure (`k = 9`), one voted step faults with probability
   `≤ 1800003 / 250000000000`.
3. A 1000-step voted run faults with probability `≤ 180000003 / 25000000000`. -/
theorem witness_p7 (r : X86Int.Rates) (hr : r.ok)
    (hf : PDist.p7.mem r.fetch) (hg : PDist.p7.mem r.reg)
    (hm : PDist.p7.mem r.mem) (hl : PDist.p7.mem r.flag)
    (s : X86Int.St) {x : Nat} (hx : x < X86Env.W) :
    let C := regWordChannel (fun v => (v + 1) % X86Env.W) r hr
    (votedStep C r hr x s).ok ∧
    X86Int.Out.ok (voterDone s ((x + 1) % X86Env.W)) ∈ PDist.faultFree (votedStep C r hr x s) ∧
    (∀ u ∈ PDist.faultFree (votedStep C r hr x s), u = .ok (voterDone s ((x + 1) % X86Env.W))) ∧
    PDist.fm (votedStep C r hr x s) ≤ 180000003 / 25000000000000 ∧
    PDist.fm (votedStepExposed C 9 r hr x s) ≤ 1800003 / 250000000000 ∧
    PDist.fm (votedRun C r hr 1000 (.ok (voterDone s x))) ≤ 180000003 / 25000000000 := by
  intro C
  have hC : ∀ z, z < X86Env.W → C.f z < X86Env.W ∧ ∀ y ∈ C.step z, y.1.1 < X86Env.W :=
    fun z hz => ⟨inc_mod_lt_W z hz, regWordChannel_lt_W inc_mod_lt_W r hr hz⟩
  have h1 := voted_step_with_faulty_voter_bound C r hr x s (hC x hx).1 (hC x hx).2
  have hexp := voted_step_exposed_with_faulty_voter_bound C 9 r hr x s hx hC
  have hrun := voted_run_with_faulty_voter_bound C r hr hC 1000 s hx
  have hreg_sq : r.reg * r.reg ≤ (1 / 5000000 : Rat) * (1 / 5000000) :=
    TMR.sq_le hr.2.1.1 hg.2
  have htot : r.total ≤ 4 / 5000000 := by
    have hsum : r.fetch + (r.reg + (r.mem + r.flag)) ≤
        (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) :=
      add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))
    have h4 : (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) = 4 / 5000000 := by
      decide +kernel
    exact Rat.le_trans hsum (h4 ▸ Rat.le_refl)
  have hstep_le : 3 * (C.q * C.q) + 9 * r.total ≤ 180000003 / 25000000000000 := by
    have h3 : 3 * (r.reg * r.reg) ≤ 3 * ((1 / 5000000 : Rat) * (1 / 5000000)) :=
      Rat.mul_le_mul_of_nonneg_left hreg_sq (by decide)
    have h9 : 9 * r.total ≤ 9 * (4 / 5000000 : Rat) :=
      Rat.mul_le_mul_of_nonneg_left htot (by decide)
    exact Rat.le_trans (add_le_add' h3 h9) (p7_step_bound_rat ▸ Rat.le_refl)
  refine ⟨h1.1, h1.2.1, h1.2.2.1, Rat.le_trans h1.2.2.2 hstep_le, ?_, ?_⟩
  · have h10 : ((9 + 1 : Nat) : Rat) * r.reg ≤ ((9 + 1 : Nat) : Rat) * (1 / 5000000) :=
      Rat.mul_le_mul_of_nonneg_left hg.2 Rat.natCast_nonneg
    have h10_0 : 0 ≤ ((9 + 1 : Nat) : Rat) * r.reg :=
      Rat.mul_nonneg Rat.natCast_nonneg hr.2.1.1
    have hsq10 : (((9 + 1 : Nat) : Rat) * r.reg) ^ 2 ≤ (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 := by
      rw [Rat.pow_succ, Rat.pow_one, Rat.pow_succ, Rat.pow_one]
      exact TMR.sq_le h10_0 h10
    have h3exp : 3 * (((9 + 1 : Nat) : Rat) * r.reg) ^ 2 ≤
        3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 :=
      Rat.mul_le_mul_of_nonneg_left hsq10 (by decide)
    have h9 : 9 * r.total ≤ 9 * (4 / 5000000 : Rat) :=
      Rat.mul_le_mul_of_nonneg_left htot (by decide)
    exact Rat.le_trans hexp.2.2.2
      (Rat.le_trans (add_le_add' h3exp h9) (p7_exposed9_bound_rat ▸ Rat.le_refl))
  · have h1000 : ((1000 : Nat) : Rat) * (3 * (C.q * C.q) + 9 * r.total) ≤
        ((1000 : Nat) : Rat) * (180000003 / 25000000000000) :=
      Rat.mul_le_mul_of_nonneg_left hstep_le Rat.natCast_nonneg
    exact Rat.le_trans hrun.2.2.2 (Rat.le_trans h1000 (p7_run1000_bound_rat ▸ Rat.le_refl))

end TMRVoter
