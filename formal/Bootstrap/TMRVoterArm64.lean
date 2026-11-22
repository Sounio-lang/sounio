/-
TMRVoterArm64: formal verification of the canonical 44-byte (11-instruction)
AArch64 machine-code TMR majority voter under architectural semantics
(`AArch64Phys.run`) and 4-channel physical fault semantics
(`AArch64Phys.prun` with `PDist`, `TMR`, and `TMRExposure`).

The voter operates over three disjoint callee-saved 64-bit copy registers
`x19` (copy A), `x20` (copy B), and `x21` (copy C), using zero scratch
registers on the majority path:
  - compares `(x19, x20)`, `(x19, x21)`, `(x20, x21)` via `cmp`/`ccmp`/`csel`;
  - on any 2-of-3 or 3-of-3 majority `v`, repairs all three copies to
    `x19 = x20 = x21 = v` and parks at `voterBase + 0x1c` (`voterPark`) with
    deterministic flags `NZCV = 0110`;
  - on 3-way disagreement (`TMR.vote a b c = none`), executes Linux AArch64
    `movz x0, #3; movz x8, #93; svc #0` and halts with `Out.exit 3`.

Proves:
  - `voterCode_length` (44 bytes = 11 fixed-width 32-bit instructions)
  - `voterIm_decode` (all 11 instructions decoded by `decide +kernel`)
  - `voter_arch_majority` & `voter_arch_refuse`
  - `voter_mask_single_copy`, `voter_single_bit_masking`, `voter_mask_nonreplica_reg`
  - `voter_phys_bound` (`≤ 11 * r.total`)
  - `voted_step_with_faulty_voter_bound` (`≤ 3 * q_body² + 11 * r.total`)
  - `voted_step_exposed_with_faulty_voter_bound` (`≤ 3 * ((k + 1) * q_body)² + 11 * r.total`)
  - `voted_run_with_faulty_voter_bound` (`≤ n * (3 * q_body² + 11 * r.total)`)
  - `witness_p7` (concrete exact rational `p7` witness)

Core Lean 4 only: zero Mathlib imports, zero axioms beyond core Lean.
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.AArch64Env
import Bootstrap.AArch64Phys

namespace TMRVoterArm64

open AArch64Env AArch64Phys PDist TMR

/-! ### Canonical 44-byte (11-instruction) AArch64 machine-code voter -/

def voterBase   : Nat := 0x400000
def voterPark   : Nat := voterBase + 0x1c
def voterRefuse : Nat := voterBase + 0x20

/-- Canonical 11-instruction (44-byte) AArch64 majority voter over `(x19, x20, x21)`. -/
def voterWords : List Nat :=
  [ 0xeb14027f  -- 0x00: cmp  x19, x20
  , 0xfa551266  -- 0x04: ccmp x19, x21, #6, ne
  , 0x9a940273  -- 0x08: csel x19, x19, x20, eq
  , 0xfa551286  -- 0x0c: ccmp x20, x21, #6, ne
  , 0x54000081  -- 0x10: b.ne +0x10 (-> 0x20)
  , 0xaa1303f4  -- 0x14: mov  x20, x19
  , 0xaa1303f5  -- 0x18: mov  x21, x19
  , 0x14000000  -- 0x1c: b    +0x00 (-> 0x1c)
  , 0xd2800060  -- 0x20: movz x0, #3
  , 0xd2800ba8  -- 0x24: movz x8, #93
  , 0xd4000001  -- 0x28: svc  #0
  ]

def voterCode : List Nat := wordsToBytes voterWords

theorem voterWords_length : voterWords.length = 11 := by decide +kernel
theorem voterCode_length  : voterCode.length = 44 := by decide +kernel

def voterIm : Img := imageOf voterCode voterBase
def voterAlts : Alts := imageAlts voterCode voterBase

theorem voterIm_fixedWidth : voterIm.FixedWidth :=
  imageOf_fixedWidth voterCode voterBase

theorem voterIm_00 : voterIm (voterBase + 0x00) = some (.addSubReg true true 64 31 19 20 .lsl 0, 4) := by decide +kernel
theorem voterIm_04 : voterIm (voterBase + 0x04) = some (.ccmpReg true 64 19 21 6 1, 4) := by decide +kernel
theorem voterIm_08 : voterIm (voterBase + 0x08) = some (.csel .csel 64 19 19 20 0, 4) := by decide +kernel
theorem voterIm_0c : voterIm (voterBase + 0x0c) = some (.ccmpReg true 64 20 21 6 1, 4) := by decide +kernel
theorem voterIm_10 : voterIm (voterBase + 0x10) = some (.bcond 1 16, 4) := by decide +kernel
theorem voterIm_14 : voterIm (voterBase + 0x14) = some (.logicReg .orr false 64 20 31 19 .lsl 0, 4) := by decide +kernel
theorem voterIm_18 : voterIm (voterBase + 0x18) = some (.logicReg .orr false 64 21 31 19 .lsl 0, 4) := by decide +kernel
theorem voterIm_1c : voterIm (voterBase + 0x1c) = some (.b false 0, 4) := by decide +kernel
theorem voterIm_20 : voterIm (voterBase + 0x20) = some (.movWide 2 64 0 3 0, 4) := by decide +kernel
theorem voterIm_24 : voterIm (voterBase + 0x24) = some (.movWide 2 64 8 93 0, 4) := by decide +kernel
theorem voterIm_28 : voterIm (voterBase + 0x28) = some (.svc 0, 4) := by decide +kernel

/-- Every 4-byte-aligned offset of `voterCode` decodes to its exact `(Ins, 4)`. -/
theorem voterIm_decode :
    voterIm (voterBase + 0x00) = some (.addSubReg true true 64 31 19 20 .lsl 0, 4) ∧
    voterIm (voterBase + 0x04) = some (.ccmpReg true 64 19 21 6 1, 4) ∧
    voterIm (voterBase + 0x08) = some (.csel .csel 64 19 19 20 0, 4) ∧
    voterIm (voterBase + 0x0c) = some (.ccmpReg true 64 20 21 6 1, 4) ∧
    voterIm (voterBase + 0x10) = some (.bcond 1 16, 4) ∧
    voterIm (voterBase + 0x14) = some (.logicReg .orr false 64 20 31 19 .lsl 0, 4) ∧
    voterIm (voterBase + 0x18) = some (.logicReg .orr false 64 21 31 19 .lsl 0, 4) ∧
    voterIm (voterBase + 0x1c) = some (.b false 0, 4) ∧
    voterIm (voterBase + 0x20) = some (.movWide 2 64 0 3 0, 4) ∧
    voterIm (voterBase + 0x24) = some (.movWide 2 64 8 93 0, 4) ∧
    voterIm (voterBase + 0x28) = some (.svc 0, 4) :=
  ⟨voterIm_00, voterIm_04, voterIm_08, voterIm_0c, voterIm_10, voterIm_14,
   voterIm_18, voterIm_1c, voterIm_20, voterIm_24, voterIm_28⟩

theorem voterAlts_len (k : Nat) (hk : k < 11) :
    (voterAlts (voterBase + 4 * k)).length = 32 := by
  unfold voterAlts imageAlts
  have hcond : (voterBase ≤ voterBase + 4 * k && voterBase + 4 * k + 4 ≤ voterBase + voterCode.length &&
      voterBase % 4 == 0 && (voterBase + 4 * k) % 4 == 0) = true := by
    rw [voterCode_length]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    have hvb : voterBase = 0x400000 := rfl
    rw [hvb]
    exact ⟨⟨⟨by omega, by omega⟩, by decide⟩, by omega⟩
  rw [if_pos hcond, altsAt_length]

theorem voterAlts_fixed_width (a : Nat) {o : Option (Ins × Nat)} {ins : Ins} {len : Nat}
    (hm : o ∈ voterAlts a) (ho : o = some (ins, len)) : len = 4 := by
  unfold voterAlts imageAlts at hm
  split at hm
  · exact altsAt_len_eq_4 hm ho
  · simp at hm

/-! ### Voter states and architectural verification -/

/-- Register file layout of length 31 holding the three 64-bit TMR copies in
`x19`, `x20`, `x21` while preserving all other registers from base state `s`. -/
def voterRegs (s : St) (a b c : Nat) : List (Option Nat) :=
  [ s.reg 0,  s.reg 1,  s.reg 2,  s.reg 3,  s.reg 4,  s.reg 5,  s.reg 6,  s.reg 7
  , s.reg 8,  s.reg 9,  s.reg 10, s.reg 11, s.reg 12, s.reg 13, s.reg 14, s.reg 15
  , s.reg 16, s.reg 17, s.reg 18, some a,   some b,   some c,   s.reg 22, s.reg 23
  , s.reg 24, s.reg 25, s.reg 26, s.reg 27, s.reg 28, s.reg 29, s.reg 30 ]

/-- Flags left on any majority path: `N=0, Z=1, C=1, V=0` (imm `#6` = `0b0110`). -/
def voterFlagsEq : Flags := ⟨some false, some true, some true, some false⟩

def voterInit (s : St) (a b c : Nat) : St :=
  { s with regs := voterRegs s a b c, pc := voterBase, fl := Flags.undef }

def voterDone (s : St) (v : Nat) : St :=
  { s with regs := voterRegs s v v v, pc := voterPark, fl := voterFlagsEq }

theorem voterInit_regs_length (s : St) (a b c : Nat) : (voterInit s a b c).regs.length = 31 := rfl
theorem voterDone_regs_length (s : St) (v : Nat) : (voterDone s v).regs.length = 31 := rfl

theorem voterDone_idem (s : St) (v w : Nat) : voterDone (voterDone s v) w = voterDone s w := by
  simp [voterDone, voterRegs, St.reg]

theorem voterDone_pc (s : St) (v : Nat) : (voterDone s v).pc = voterPark := rfl
theorem voterDone_fl (s : St) (v : Nat) : (voterDone s v).fl = voterFlagsEq := rfl
theorem voterDone_reg19 (s : St) (v : Nat) : (voterDone s v).reg 19 = some v := rfl
theorem voterDone_reg20 (s : St) (v : Nat) : (voterDone s v).reg 20 = some v := rfl
theorem voterDone_reg21 (s : St) (v : Nat) : (voterDone s v).reg 21 = some v := rfl

private theorem addSubOp_sub_eq (v : Nat) :
    (addSubOp true 64 v v).2 = voterFlagsEq := by
  unfold addSubOp voterFlagsEq
  have hpow : pow2 64 = W := rfl
  rw [hpow]
  simp [msb]

private theorem addSubOp_sub_z_ne {x y : Nat} (hx : x < W) (hy : y < W) (hne : x ≠ y) :
    (addSubOp true 64 x y).2.z = some false := by
  unfold addSubOp
  have hpow : pow2 64 = W := rfl
  rw [hpow]
  have hpos : 0 < x + W - y := by omega
  have hlt : x + W - y < 2 * W := by omega
  have hneW : x + W - y ≠ W := by omega
  have hnz : (x + W - y) % W ≠ 0 := by
    intro hmod
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hmod
    rcases k with _ | _ | k
    · simp at hk; omega
    · simp at hk; exact hneW hk
    · have hk2 : 2 ≤ k + 2 := by omega
      have hge : 2 * W ≤ W * (k + 2) := by
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left W hk2
      have hkeq : x + W - y = W * (k + 2) := hk
      omega
  simp [hnz]

private theorem run_exit (im : Img) (c : Nat) : ∀ n, run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck (im : Img) : ∀ n, run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_add (im : Img) :
    ∀ m n o, run im (m + n) o = run im n (run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [run, run_add im m n (step im s)]
  | _ + 1, n, .exit c => by simp [run, run_exit]
  | _ + 1, n, .stuck => by simp [run, run_stuck]

private theorem step_park (s : St) (v : Nat) :
    step voterIm (voterDone s v) = .ok (voterDone s v) := by
  unfold step
  dsimp only [voterDone, voterPark]
  rw [voterIm_1c]
  simp [exec, voterBase, W]

private theorem run_park (s : St) (v : Nat) :
    ∀ k, run voterIm k (.ok (voterDone s v)) = .ok (voterDone s v)
  | 0 => rfl
  | k + 1 => by simp only [run, step_park, run_park]

private theorem lor_zero_left (v : Nat) : Nat.lor 0 v = v := by
  change 0 ||| v = v
  exact Nat.zero_or v

private theorem voterRegs_get19 (s : St) (a b c : Nat) :
    ((voterRegs s a b c)[19]?).getD none = some a := rfl
private theorem voterRegs_get20 (s : St) (a b c : Nat) :
    ((voterRegs s a b c)[20]?).getD none = some b := rfl
private theorem voterRegs_get21 (s : St) (a b c : Nat) :
    ((voterRegs s a b c)[21]?).getD none = some c := rfl
private theorem voterRegs_set19 (s : St) (a b c v : Nat) :
    (voterRegs s a b c).set 19 (some v) = voterRegs s v b c := rfl
private theorem voterRegs_set20 (s : St) (a b c v : Nat) :
    (voterRegs s a b c).set 20 (some v) = voterRegs s a v c := rfl
private theorem voterRegs_set21 (s : St) (a b c v : Nat) :
    (voterRegs s a b c).set 21 (some v) = voterRegs s a b v := rfl
private theorem voterRegs_set0_set8_get8 (s : St) (a b c v0 v8 : Nat) :
    ((((voterRegs s a b c).set 0 (some v0)).set 8 (some v8))[8]?).getD none = some v8 := rfl
private theorem voterRegs_set0_set8_get0 (s : St) (a b c v0 v8 : Nat) :
    ((((voterRegs s a b c).set 0 (some v0)).set 8 (some v8))[0]?).getD none = some v0 := rfl

private theorem step_00 (s : St) {a b c : Nat} (ha : a < W) (hb : b < W) :
    step voterIm (voterInit s a b c) =
      .ok { s with regs := voterRegs s a b c, pc := voterBase + 0x04, fl := (addSubOp true 64 a b).2 } := by
  have haW : a % pow2 64 = a := Nat.mod_eq_of_lt ha
  have hbW : b % pow2 64 = b := Nat.mod_eq_of_lt hb
  have him0 : voterIm voterBase = some (.addSubReg true true 64 31 19 20 .lsl 0, 4) := voterIm_00
  unfold step; dsimp only [voterInit]
  rw [him0]
  simp only [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg,
    voterRegs_get19, voterRegs_get20,
    shiftVal, pow2, Nat.pow_zero, Nat.mul_one, Option.map_some,
    show (19 == 31) = false from rfl, show (20 == 31) = false from rfl,
    show (31 == 31) = true from rfl, Bool.false_eq_true, ↓reduceIte]
  simp only [show a % 2 ^ 64 = a from haW, show b % 2 ^ 64 = b from hbW,
    show (voterBase + 4) % W = voterBase + 0x04 by decide]

private theorem step_04_eq (s : St) {a b c : Nat} (ha : a < W) (hc : c < W) :
    step voterIm { s with regs := voterRegs s a b c, pc := voterBase + 0x04, fl := voterFlagsEq } =
      .ok { s with regs := voterRegs s a b c, pc := voterBase + 0x08, fl := voterFlagsEq } := by
  have haW : a % pow2 64 = a := Nat.mod_eq_of_lt ha
  have hcW : c % pow2 64 = c := Nat.mod_eq_of_lt hc
  have hc1 : AArch64Phys.cond voterFlagsEq 1 = some false := rfl
  unfold step; dsimp only; rw [voterIm_04]
  simp only [exec, hc1, lift, bind, Option.bind, pure, St.rdZR, St.reg,
    voterRegs_get19, voterRegs_get21, Option.map_some,
    show (19 == 31) = false from rfl, show (21 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, haW, hcW]
  rw [show (voterBase + 0x04 + 4) % W = voterBase + 0x08 by decide]
  rfl

private theorem step_04_ne (s : St) {a b c : Nat} {fl : Flags} (hz : fl.z = some false)
    (ha : a < W) (hc : c < W) :
    step voterIm { s with regs := voterRegs s a b c, pc := voterBase + 0x04, fl := fl } =
      .ok { s with regs := voterRegs s a b c, pc := voterBase + 0x08, fl := (addSubOp true 64 a c).2 } := by
  have haW : a % pow2 64 = a := Nat.mod_eq_of_lt ha
  have hcW : c % pow2 64 = c := Nat.mod_eq_of_lt hc
  have hc1 : AArch64Phys.cond fl 1 = some true := by simp [AArch64Phys.cond, hz]
  unfold step; dsimp only; rw [voterIm_04]
  simp only [exec, hc1, lift, bind, Option.bind, pure, St.rdZR, St.reg,
    voterRegs_get19, voterRegs_get21, Option.map_some,
    show (19 == 31) = false from rfl, show (21 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, haW, hcW]
  rw [show (voterBase + 0x04 + 4) % W = voterBase + 0x08 by decide]

private theorem step_08_eq (s : St) {a b c : Nat} (ha : a < W) (hb : b < W) :
    step voterIm { s with regs := voterRegs s a b c, pc := voterBase + 0x08, fl := voterFlagsEq } =
      .ok { s with regs := voterRegs s a b c, pc := voterBase + 0x0c, fl := voterFlagsEq } := by
  have haW : a % pow2 64 = a := Nat.mod_eq_of_lt ha
  have hbW : b % pow2 64 = b := Nat.mod_eq_of_lt hb
  have hc0 : AArch64Phys.cond voterFlagsEq 0 = some true := rfl
  unfold step; dsimp only; rw [voterIm_08]
  simp only [exec, hc0, lift, bind, Option.bind, pure, St.rdZR, St.wrZR,
    St.reg, St.setReg, voterRegs_get19, voterRegs_get20, voterRegs_set19, Option.map_some,
    show (19 == 31) = false from rfl, show (20 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, haW, hbW]
  rw [show (voterBase + 0x08 + 4) % W = voterBase + 0x0c by decide]

private theorem step_08_ne (s : St) {a b c : Nat} {fl : Flags} (hz : fl.z = some false)
    (ha : a < W) (hb : b < W) :
    step voterIm { s with regs := voterRegs s a b c, pc := voterBase + 0x08, fl := fl } =
      .ok { s with regs := voterRegs s b b c, pc := voterBase + 0x0c, fl := fl } := by
  have haW : a % pow2 64 = a := Nat.mod_eq_of_lt ha
  have hbW : b % pow2 64 = b := Nat.mod_eq_of_lt hb
  have hc0 : AArch64Phys.cond fl 0 = some false := by simp [AArch64Phys.cond, hz]
  unfold step; dsimp only; rw [voterIm_08]
  simp only [exec, hc0, lift, bind, Option.bind, pure, St.rdZR, St.wrZR,
    St.reg, St.setReg, voterRegs_get19, voterRegs_get20, voterRegs_set19, Option.map_some,
    show (19 == 31) = false from rfl, show (20 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, haW, hbW]
  rw [show (voterBase + 0x08 + 4) % W = voterBase + 0x0c by decide]

private theorem step_0c_eq (s : St) {v b c : Nat} (hb : b < W) (hc : c < W) :
    step voterIm { s with regs := voterRegs s v b c, pc := voterBase + 0x0c, fl := voterFlagsEq } =
      .ok { s with regs := voterRegs s v b c, pc := voterBase + 0x10, fl := voterFlagsEq } := by
  have hbW : b % pow2 64 = b := Nat.mod_eq_of_lt hb
  have hcW : c % pow2 64 = c := Nat.mod_eq_of_lt hc
  have hc1 : AArch64Phys.cond voterFlagsEq 1 = some false := rfl
  unfold step; dsimp only; rw [voterIm_0c]
  simp only [exec, hc1, lift, bind, Option.bind, pure, St.rdZR, St.reg,
    voterRegs_get20, voterRegs_get21, Option.map_some,
    show (20 == 31) = false from rfl, show (21 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, hbW, hcW]
  rw [show (voterBase + 0x0c + 4) % W = voterBase + 0x10 by decide]
  rfl

private theorem step_0c_ne (s : St) {v b c : Nat} {fl : Flags} (hz : fl.z = some false)
    (hb : b < W) (hc : c < W) :
    step voterIm { s with regs := voterRegs s v b c, pc := voterBase + 0x0c, fl := fl } =
      .ok { s with regs := voterRegs s v b c, pc := voterBase + 0x10, fl := (addSubOp true 64 b c).2 } := by
  have hbW : b % pow2 64 = b := Nat.mod_eq_of_lt hb
  have hcW : c % pow2 64 = c := Nat.mod_eq_of_lt hc
  have hc1 : AArch64Phys.cond fl 1 = some true := by simp [AArch64Phys.cond, hz]
  unfold step; dsimp only; rw [voterIm_0c]
  simp only [exec, hc1, lift, bind, Option.bind, pure, St.rdZR, St.reg,
    voterRegs_get20, voterRegs_get21, Option.map_some,
    show (20 == 31) = false from rfl, show (21 == 31) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, hbW, hcW]
  rw [show (voterBase + 0x0c + 4) % W = voterBase + 0x10 by decide]

private theorem step_10_nt (s : St) {v b c : Nat} :
    step voterIm { s with regs := voterRegs s v b c, pc := voterBase + 0x10, fl := voterFlagsEq } =
      .ok { s with regs := voterRegs s v b c, pc := voterBase + 0x14, fl := voterFlagsEq } := by
  have hc1 : AArch64Phys.cond voterFlagsEq 1 = some false := rfl
  unfold step; dsimp only; rw [voterIm_10]
  simp only [exec, hc1, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
  rw [show (voterBase + 0x10 + 4) % W = voterBase + 0x14 by decide]

private theorem step_14 (s : St) {v b c : Nat} (hv : v < W) :
    step voterIm { s with regs := voterRegs s v b c, pc := voterBase + 0x14, fl := voterFlagsEq } =
      .ok { s with regs := voterRegs s v v c, pc := voterBase + 0x18, fl := voterFlagsEq } := by
  have hvW : v % pow2 64 = v := Nat.mod_eq_of_lt hv
  unfold step; dsimp only; rw [voterIm_14]
  simp only [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, St.setReg,
    voterRegs_get19, voterRegs_set20,
    shiftVal, pow2, Nat.pow_zero, Nat.mul_one, Option.map_some,
    show (31 == 31) = true from rfl, show (19 == 31) = false from rfl,
    show (20 == 31) = false from rfl, show (LogicOp.orr == LogicOp.ands) = false from rfl,
    Bool.false_eq_true, ↓reduceIte]
  simp only [show v % 2 ^ 64 = v from hvW, lor_zero_left,
    show (voterBase + 0x14 + 4) % W = voterBase + 0x18 by decide]

private theorem step_18 (s : St) {v c : Nat} (hv : v < W) :
    step voterIm { s with regs := voterRegs s v v c, pc := voterBase + 0x18, fl := voterFlagsEq } =
      .ok (voterDone s v) := by
  have hvW : v % pow2 64 = v := Nat.mod_eq_of_lt hv
  unfold step; dsimp only; rw [voterIm_18]
  simp only [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, St.setReg,
    voterRegs_get19, voterRegs_set21,
    shiftVal, pow2, Nat.pow_zero, Nat.mul_one, Option.map_some,
    show (31 == 31) = true from rfl, show (19 == 31) = false from rfl,
    show (21 == 31) = false from rfl, show (LogicOp.orr == LogicOp.ands) = false from rfl,
    Bool.false_eq_true, ↓reduceIte]
  simp only [show v % 2 ^ 64 = v from hvW, lor_zero_left,
    show (voterBase + 0x18 + 4) % W = voterPark by decide]
  rfl

private theorem run3_ok {s0 s1 s2 s3 : St}
    (h0 : step voterIm s0 = .ok s1)
    (h1 : step voterIm s1 = .ok s2)
    (h2 : step voterIm s2 = .ok s3) :
    run voterIm 3 (.ok s0) = .ok s3 := by
  change run voterIm 2 (step voterIm s0) = .ok s3
  rw [h0]
  change run voterIm 1 (step voterIm s1) = .ok s3
  rw [h1]
  change run voterIm 0 (step voterIm s2) = .ok s3
  rw [h2]; rfl

private theorem run4_ok {s0 s1 s2 s3 s4 : St}
    (h0 : step voterIm s0 = .ok s1)
    (h1 : step voterIm s1 = .ok s2)
    (h2 : step voterIm s2 = .ok s3)
    (h3 : step voterIm s3 = .ok s4) :
    run voterIm 4 (.ok s0) = .ok s4 := by
  change run voterIm 3 (step voterIm s0) = .ok s4
  rw [h0]
  exact run3_ok h1 h2 h3

private theorem run4_exit {s0 s1 s2 s3 : St} {c : Nat}
    (h0 : step voterIm s0 = .ok s1)
    (h1 : step voterIm s1 = .ok s2)
    (h2 : step voterIm s2 = .ok s3)
    (h3 : step voterIm s3 = .exit c) :
    run voterIm 4 (.ok s0) = .exit c := by
  change run voterIm 3 (step voterIm s0) = .exit c
  rw [h0]
  change run voterIm 2 (step voterIm s1) = .exit c
  rw [h1]
  change run voterIm 1 (step voterIm s2) = .exit c
  rw [h2]
  change run voterIm 0 (step voterIm s3) = .exit c
  rw [h3]; rfl

private theorem step_repair (s : St) {v b c : Nat} (hv : v < W) :
    run voterIm 3 (.ok { s with regs := voterRegs s v b c, pc := voterBase + 0x10, fl := voterFlagsEq }) =
      .ok (voterDone s v) :=
  run3_ok (step_10_nt s) (step_14 s hv) (step_18 s hv)

/-- First 4 instructions on any majority input reach `0x10` with `x19 = v` and `fl = voterFlagsEq`. -/
private theorem steps_00_to_10_majority (s : St) {a b c v : Nat}
    (ha : a < W) (hb : b < W) (hc : c < W) (hv : TMR.vote a b c = some v) :
    run voterIm 4 (.ok (voterInit s a b c)) =
      .ok { s with regs := voterRegs s v b c, pc := voterBase + 0x10, fl := voterFlagsEq } := by
  by_cases hab : a = b
  · subst a
    have hveq : v = b := by unfold TMR.vote at hv; simp at hv; exact hv.symm
    subst v
    have h0 := step_00 (c := c) s hb hb
    rw [addSubOp_sub_eq] at h0
    exact run4_ok h0 (step_04_eq (b := b) s hb hc) (step_08_eq (c := c) s hb hb) (step_0c_eq (v := b) s hb hc)
  · have hzab : (addSubOp true 64 a b).2.z = some false := addSubOp_sub_z_ne ha hb hab
    by_cases hac : a = c
    · subst a
      have hveq : v = c := by unfold TMR.vote at hv; simp at hv; exact hv.symm
      subst v
      have h1 := step_04_ne (b := b) s hzab hc hc
      rw [addSubOp_sub_eq] at h1
      exact run4_ok (step_00 (c := c) s hc hb) h1 (step_08_eq (c := c) s hc hb) (step_0c_eq (v := c) s hb hc)
    · have hbc : b = c := by
        unfold TMR.vote at hv
        simp [hab, hac] at hv
        exact hv.1
      subst b
      have hveq : v = c := by unfold TMR.vote at hv; simp [hab] at hv; exact hv.symm
      subst v
      have hzac : (addSubOp true 64 a c).2.z = some false := addSubOp_sub_z_ne ha hc hac
      have h3 := step_0c_ne (v := c) s hzac hc hc
      rw [addSubOp_sub_eq] at h3
      exact run4_ok (step_00 (c := c) s ha hc) (step_04_ne (b := c) s hzab ha hc)
        (step_08_ne (c := c) s hzac ha hc) h3

/-- Majority reaches `voterDone s v` in 7 steps. -/
theorem voter_arch_majority_7 (s : St) {a b c v : Nat}
    (ha : a < W) (hb : b < W) (hc : c < W) (hv : TMR.vote a b c = some v) :
    run voterIm 7 (.ok (voterInit s a b c)) = .ok (voterDone s v) := by
  have hvW : v < W := by
    unfold TMR.vote at hv
    split at hv
    · cases hv; exact ha
    · split at hv
      · cases hv; exact hb
      · cases hv
  rw [show (7 : Nat) = 4 + 3 from rfl, run_add, steps_00_to_10_majority s ha hb hc hv, step_repair s hvW]

/-- Architectural correctness of the 44-byte (11-instruction) AArch64 voter on
majority: for all 64-bit words `a b c < W` and any base state `s`, 11 steps of
`voterIm` from `voterInit s a b c` land on `voterDone s v` (`x19 = x20 = x21 = v`). -/
theorem voter_arch_majority (s : St) {a b c v : Nat}
    (ha : a < W) (hb : b < W) (hc : c < W) (hv : TMR.vote a b c = some v) :
    run voterIm 11 (.ok (voterInit s a b c)) = .ok (voterDone s v) := by
  rw [show (11 : Nat) = 7 + 4 from rfl, run_add, voter_arch_majority_7 s ha hb hc hv, run_park]

private theorem step_10_t (s : St) {b c : Nat} {fl : Flags} (hz : fl.z = some false) :
    step voterIm { s with regs := voterRegs s b b c, pc := voterBase + 0x10, fl := fl } =
      .ok { s with regs := voterRegs s b b c, pc := voterBase + 0x20, fl := fl } := by
  have hc1 : AArch64Phys.cond fl 1 = some true := by simp [AArch64Phys.cond, hz]
  unfold step; dsimp only; rw [voterIm_10]
  simp only [exec, hc1, lift, bind, Option.bind, pure, ↓reduceIte]
  rw [show (voterBase + 0x10 + 16) % W = voterBase + 0x20 by decide]

private theorem step_20 (s : St) {b c : Nat} {fl : Flags} :
    step voterIm { s with regs := voterRegs s b b c, pc := voterBase + 0x20, fl := fl } =
      .ok { s with regs := (voterRegs s b b c).set 0 (some 3), pc := voterBase + 0x24, fl := fl } := by
  unfold step; dsimp only; rw [voterIm_20]
  simp only [exec, St.wrZR, St.setReg, pow2,
    show (0 == 31) = false from rfl, Bool.false_eq_true, ↓reduceIte,
    show (3 : Nat) % 2 ^ 64 = 3 by decide]
  rw [show (voterBase + 0x20 + 4) % W = voterBase + 0x24 by decide]

private theorem step_24 (s : St) {b c : Nat} {fl : Flags} :
    step voterIm { s with regs := (voterRegs s b b c).set 0 (some 3), pc := voterBase + 0x24, fl := fl } =
      .ok { s with regs := ((voterRegs s b b c).set 0 (some 3)).set 8 (some 93), pc := voterBase + 0x28, fl := fl } := by
  unfold step; dsimp only; rw [voterIm_24]
  simp only [exec, St.wrZR, St.setReg, pow2,
    show (8 == 31) = false from rfl, Bool.false_eq_true, ↓reduceIte,
    show (93 : Nat) % 2 ^ 64 = 93 by decide]
  rw [show (voterBase + 0x24 + 4) % W = voterBase + 0x28 by decide]

private theorem step_28 (s : St) {b c : Nat} {fl : Flags} :
    step voterIm { s with regs := ((voterRegs s b b c).set 0 (some 3)).set 8 (some 93), pc := voterBase + 0x28, fl := fl } =
      .exit 3 := by
  unfold step; dsimp only; rw [voterIm_28]
  simp only [exec, execSvc, applySys, St.reg,
    voterRegs_set0_set8_get8, voterRegs_set0_set8_get0, syscall]
  rfl

/-- Refusal reaches `Out.exit 3` in 8 steps. -/
theorem voter_arch_refuse_8 (s : St) {a b c : Nat}
    (ha : a < W) (hb : b < W) (hc : c < W) (hv : TMR.vote a b c = none) :
    run voterIm 8 (.ok (voterInit s a b c)) = .exit 3 := by
  have hab : a ≠ b := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hac : a ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hbc : b ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp [hab] at hv
  have hzab : (addSubOp true 64 a b).2.z = some false := addSubOp_sub_z_ne ha hb hab
  have hzac : (addSubOp true 64 a c).2.z = some false := addSubOp_sub_z_ne ha hc hac
  have hzbc : (addSubOp true 64 b c).2.z = some false := addSubOp_sub_z_ne hb hc hbc
  rw [show (8 : Nat) = 4 + 4 from rfl, run_add]
  have h4 : run voterIm 4 (.ok (voterInit s a b c)) =
      .ok { s with regs := voterRegs s b b c, pc := voterBase + 0x10, fl := (addSubOp true 64 b c).2 } :=
    run4_ok (step_00 (c := c) s ha hb) (step_04_ne (b := b) s hzab ha hc)
      (step_08_ne (c := c) s hzac ha hb) (step_0c_ne (v := b) s hzac hb hc)
  rw [h4]
  exact run4_exit (step_10_t s hzbc) (step_20 s) (step_24 s) (step_28 s)

/-- Architectural correctness of the 44-byte (11-instruction) AArch64 voter on
refusal: when all three 64-bit words differ (`TMR.vote a b c = none`), 11 steps
execute `movz x0, #3; movz x8, #93; svc #0` and exit with code 3. -/
theorem voter_arch_refuse (s : St) {a b c : Nat}
    (ha : a < W) (hb : b < W) (hc : c < W) (hv : TMR.vote a b c = none) :
    run voterIm 11 (.ok (voterInit s a b c)) = .exit 3 := by
  rw [show (11 : Nat) = 8 + 3 from rfl, run_add, voter_arch_refuse_8 s ha hb hc hv, run_exit]

/-! ### Single-perturbation masking theorems -/

/-- Any single-copy arbitrary 64-bit corruption (copy 0, 1, or 2) is completely
masked and repaired to `voterDone s g`. -/
theorem voter_mask_single_copy (s : St) {g bad : Nat} (hg : g < W) (hbad : bad < W) :
    run voterIm 11 (.ok (voterInit s bad g g)) = .ok (voterDone s g) ∧
    run voterIm 11 (.ok (voterInit s g bad g)) = .ok (voterDone s g) ∧
    run voterIm 11 (.ok (voterInit s g g bad)) = .ok (voterDone s g) := by
  refine ⟨?_, ?_, ?_⟩
  · exact voter_arch_majority s hbad hg hg (TMR.vote_of_two (Or.inr (Or.inr ⟨rfl, rfl⟩)))
  · exact voter_arch_majority s hg hbad hg (TMR.vote_of_two (Or.inr (Or.inl ⟨rfl, rfl⟩)))
  · exact voter_arch_majority s hg hg hbad (TMR.vote_of_two (Or.inl ⟨rfl, rfl⟩))

/-- Exhaustive single-bit perturbation masking across all 3 copies and all 64 bits:
flipping any single bit `bit < 64` of `x19`, `x20`, or `x21` always repairs all
three copies to `voterDone s g` (0 SDC, 0 refusals, 0 stuck). -/
theorem voter_single_bit_masking (s : St) {g bit : Nat} (hg : g < W) (hbit : bit < 64) :
    let flipped := Nat.xor g (2 ^ bit)
    run voterIm 11 (.ok (voterInit s flipped g g)) = .ok (voterDone s g) ∧
    run voterIm 11 (.ok (voterInit s g flipped g)) = .ok (voterDone s g) ∧
    run voterIm 11 (.ok (voterInit s g g flipped)) = .ok (voterDone s g) := by
  have hfW : Nat.xor g (2 ^ bit) < W :=
    Nat.xor_lt_two_pow hg (Nat.pow_lt_pow_right (by decide) hbit)
  exact voter_mask_single_copy s hg hfW

/-- Flipping any bit of any non-replica register (`i ∉ {19, 20, 21}`, including
`SP`) before entering the voter leaves the voted copies `(x19, x20, x21)` equal
to `g` and parks at `voterPark`. -/
theorem voter_mask_nonreplica_reg (s : St) {g i b : Nat} (hg : g < W)
    (h19 : i ≠ 19) (h20 : i ≠ 20) (h21 : i ≠ 21) :
    let s' := (voterInit s g g g).flipReg i b
    ∃ t, run voterIm 11 (.ok s') = .ok t ∧
      t.pc = voterPark ∧ t.reg 19 = some g ∧ t.reg 20 = some g ∧ t.reg 21 = some g := by
  intro s'
  have hpc : s'.pc = voterBase := by unfold s' St.flipReg; split <;> rfl
  have hfl : s'.fl = Flags.undef := by unfold s' St.flipReg; split <;> rfl
  have hlen : s'.regs.length = 31 := by
    unfold s' St.flipReg
    split
    · rfl
    · simp [St.setReg, voterInit, voterRegs]
  have hreg_eq : ∀ j, j ∈ [19, 20, 21] → s'.reg j = some g := by
    intro j hj
    have hne : i ≠ j := by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
      rcases hj with rfl | rfl | rfl <;> assumption
    unfold s' St.flipReg
    split
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
      rcases hj with rfl | rfl | rfl <;> rfl
    · simp only [St.reg, St.setReg, List.getElem?_set_ne hne]
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
      rcases hj with rfl | rfl | rfl <;> rfl
  have h19s : s'.reg 19 = some g := hreg_eq 19 (by decide)
  have h20s : s'.reg 20 = some g := hreg_eq 20 (by decide)
  have h21s : s'.reg 21 = some g := hreg_eq 21 (by decide)
  have hregs : s'.regs = voterRegs s' g g g := by
    apply List.ext_getElem (by simp [hlen, voterRegs])
    intro j hj1 _
    have hlt : j < 31 := by omega
    have hget : ∀ k (hk : k < 31), (voterRegs s' g g g)[k]'(by simp [voterRegs]; omega) = s'.regs[k]'(by omega) := by
      intro k hk
      match k with
      | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15
      | 16 | 17 | 18 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 =>
        simp only [voterRegs, St.reg, List.getElem_cons_zero, List.getElem_cons_succ]
        rw [List.getElem?_eq_getElem (by omega)]; rfl
      | 19 =>
        simp only [voterRegs, List.getElem_cons_zero, List.getElem_cons_succ]
        simp only [St.reg, List.getElem?_eq_getElem (by omega : 19 < s'.regs.length), Option.getD_some] at h19s
        exact h19s.symm
      | 20 =>
        simp only [voterRegs, List.getElem_cons_zero, List.getElem_cons_succ]
        simp only [St.reg, List.getElem?_eq_getElem (by omega : 20 < s'.regs.length), Option.getD_some] at h20s
        exact h20s.symm
      | 21 =>
        simp only [voterRegs, List.getElem_cons_zero, List.getElem_cons_succ]
        simp only [St.reg, List.getElem?_eq_getElem (by omega : 21 < s'.regs.length), Option.getD_some] at h21s
        exact h21s.symm
      | k + 31 => omega
    exact (hget j hlt).symm
  have hinit : s' = voterInit s' g g g := by
    have hmk : ∀ (t : St), t.regs = voterRegs t g g g → t.pc = voterBase → t.fl = Flags.undef →
        t = voterInit t g g g := by
      intro ⟨regs, sp, pc, fl, mem, k⟩ hr hp hf
      simp only at hr hp hf
      subst hp; subst hf
      unfold voterInit
      exact congrArg (fun r => St.mk r sp voterBase Flags.undef mem k) hr
    exact hmk s' hregs hpc hfl
  refine ⟨voterDone s' g, ?_, rfl, rfl, rfl, rfl⟩
  rw [hinit]
  exact voter_arch_majority s' hg hg hg (TMR.vote_of_two (Or.inl ⟨rfl, rfl⟩))

/-! ### Concrete `TMR.Channel` instantiations for `AArch64Phys` -/

def outStep (im : Img) : Out → Out
  | .ok s => step im s
  | o => o

def outPStep (alts : Alts) (F : Fault) (pf : Rat) (im : Img) : Out → PDF Out
  | .ok s => pstep alts F pf im s
  | o => pureF o

theorem rates_state_q (r : Rates) (hr : r.ok) :
    r.fetch + (r.state hr).q = r.total := rfl

theorem rates_total_nonneg (r : Rates) (hr : r.ok) : 0 ≤ r.total :=
  nonneg_add hr.1.1 (nonneg_add hr.2.1.1 (nonneg_add hr.2.2.1.1 hr.2.2.2.1))

def arm64Channel (alts : Alts) (im : Img) (r : Rates) (hr : r.ok) : TMR.Channel Out where
  f := outStep im
  step := outPStep alts (r.state hr) r.fetch im
  q := r.total
  q0 := rates_total_nonneg r hr
  ok o := by
    cases o with
    | ok s => exact pstep_ok alts (r.state hr) hr.1 im s
    | exit c => exact pureF_ok _
    | stuck => exact pureF_ok _
  clean o := by
    cases o with
    | ok s => exact faultFree_pstep alts (r.state hr) r.fetch im s
    | exit c => rfl
    | stuck => rfl
  le o := by
    cases o with
    | ok s => exact fm_pstep alts (r.state hr) hr.1 im s
    | exit c =>
      change fm (pureF (Out.exit c)) ≤ r.total
      rw [fm_pureF]; exact rates_total_nonneg r hr
    | stuck =>
      change fm (pureF Out.stuck) ≤ r.total
      rw [fm_pureF]; exact rates_total_nonneg r hr

def stOfOut (fallback : St) : Out → St
  | .ok s => s
  | _ => fallback

def arm64StChannel (alts : Alts) (im : Img) (r : Rates) (hr : r.ok) : TMR.Channel St where
  f s := stOfOut s (step im s)
  step s := relabel (stOfOut s) (pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok s := relabel_ok _ (pstep_ok alts (r.state hr) hr.1 im s)
  clean s := by rw [faultFree_relabel, faultFree_pstep]; rfl
  le s := by rw [fm_relabel]; exact fm_pstep alts (r.state hr) hr.1 im s

def arm64RegProjChannel (alts : Alts) (im : Img) (r : Rates) (hr : r.ok) (s0 : St) (regIdx : Nat) :
    TMR.Channel Nat where
  f v :=
    let s := s0.setReg regIdx (some v)
    ((stOfOut s (step im s)).reg regIdx).getD v
  step v :=
    let s := s0.setReg regIdx (some v)
    relabel (fun o => ((stOfOut s o).reg regIdx).getD v) (pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok v := relabel_ok _ (pstep_ok alts (r.state hr) hr.1 im (s0.setReg regIdx (some v)))
  clean v := by rw [faultFree_relabel, faultFree_pstep]; rfl
  le v := by rw [fm_relabel]; exact fm_pstep alts (r.state hr) hr.1 im (s0.setReg regIdx (some v))

def arm64MemProjChannel (alts : Alts) (im : Img) (r : Rates) (hr : r.ok) (s0 : St) (addr : Nat) :
    TMR.Channel Nat where
  f v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    ((stOfOut s (step im s)).mem.rd addr).getD v
  step v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    relabel (fun o => ((stOfOut s o).mem.rd addr).getD v) (pstep alts (r.state hr) r.fetch im s)
  q := r.total
  q0 := rates_total_nonneg r hr
  ok v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    relabel_ok _ (pstep_ok alts (r.state hr) hr.1 im s)
  clean v := by rw [faultFree_relabel, faultFree_pstep]; rfl
  le v :=
    let s := { s0 with mem := (s0.mem.wr addr v).getD s0.mem }
    by rw [fm_relabel]; exact fm_pstep alts (r.state hr) hr.1 im s

/-- Per-register 64-bit word channel under `Rates.reg` single-bit upsets. -/
def regWordChannel (fn : Nat → Nat) (r : Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v := unif r.reg (fn v) ((List.range 64).map fun b => Nat.xor (fn v) (2 ^ b))
  q := r.reg
  q0 := hr.2.1.1
  ok _ := unif_ok hr.2.1 _ _
  clean _ := faultFree_unif _ _ _
  le _ := fm_unif_le hr.2.1 _ _

/-- Per-byte memory channel under `Rates.mem` single-bit upsets. -/
def memByteChannel (fn : Nat → Nat) (r : Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v := unif r.mem (fn v) ((List.range 8).map fun b => Nat.xor (fn v) (2 ^ b))
  q := r.mem
  q0 := hr.2.2.1.1
  ok _ := unif_ok hr.2.2.1 _ _
  clean _ := faultFree_unif _ _ _
  le _ := fm_unif_le hr.2.2.1 _ _

private theorem mem_unif {β : Type} {p : Rat} {c : β} {xs : List β} {y : (β × Bool) × Rat}
    (hy : y ∈ unif p c xs) : y.1.1 = c ∨ y.1.1 ∈ xs := by
  cases xs with
  | nil =>
    simp only [unif, pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [unif, List.mem_cons, List.mem_map] at hy
    rcases hy with h | ⟨z, hz, h⟩
    · rw [h]; exact Or.inl rfl
    · rw [← h]; exact Or.inr (by simpa using hz)

theorem regWordChannel_lt_W {fn : Nat → Nat} (hfn : ∀ v, v < W → fn v < W)
    (r : Rates) (hr : r.ok) {v : Nat} (hv : v < W) :
    ∀ y ∈ (regWordChannel fn r hr).step v, y.1.1 < W := by
  intro y hy
  dsimp only [regWordChannel] at hy
  rcases mem_unif hy with heq | hmem
  · rw [heq]; exact hfn v hv
  · simp only [List.mem_map, List.mem_range] at hmem
    obtain ⟨b, hb, hxor⟩ := hmem
    rw [← hxor]
    exact Nat.xor_lt_two_pow (hfn v hv) (Nat.pow_lt_pow_right (by decide) hb)

/-! ### Faulty-silicon 11-step machine-code voter theorems -/

/-- 11 physical cycles of the 44-byte AArch64 voter from `voterInit s a b c`. -/
def voterPhys (r : Rates) (hr : r.ok) (s : St) (a b c : Nat) : PDF Out :=
  prun voterAlts (r.state hr) r.fetch voterIm 11 (.ok (voterInit s a b c))

theorem voter_phys_ok (r : Rates) (hr : r.ok) (s : St) (a b c : Nat) :
    (voterPhys r hr s a b c).ok :=
  prun_ok voterAlts (r.state hr) hr.1 voterIm 11 (.ok (voterInit s a b c))

theorem voter_phys_clean (r : Rates) (hr : r.ok) (s : St) (a b c : Nat) :
    faultFree (voterPhys r hr s a b c) = [run voterIm 11 (.ok (voterInit s a b c))] :=
  faultFree_prun voterAlts (r.state hr) r.fetch voterIm 11 (.ok (voterInit s a b c))

theorem voter_phys_clean_majority (r : Rates) (hr : r.ok) (s : St)
    {a b c v : Nat} (ha : a < W) (hb : b < W) (hc : c < W)
    (hv : TMR.vote a b c = some v) :
    faultFree (voterPhys r hr s a b c) = [.ok (voterDone s v)] := by
  rw [voter_phys_clean, voter_arch_majority s ha hb hc hv]

theorem voter_phys_clean_refuse (r : Rates) (hr : r.ok) (s : St)
    {a b c : Nat} (ha : a < W) (hb : b < W) (hc : c < W)
    (hv : TMR.vote a b c = none) :
    faultFree (voterPhys r hr s a b c) = [.exit 3] := by
  rw [voter_phys_clean, voter_arch_refuse s ha hb hc hv]

theorem voter_phys_bound (r : Rates) (hr : r.ok) (s : St) (a b c : Nat) :
    fm (voterPhys r hr s a b c) ≤ 11 * r.total := by
  simpa [voterPhys, rates_state_q] using
    fm_prun voterAlts (r.state hr) hr.1 voterIm 11 (.ok (voterInit s a b c))

theorem voter_phys_bound_iv (cf cr cm cl : Iv) (r : Rates) (hr : r.ok)
    (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem) (hl : cl.mem r.nzcv)
    (s : St) (a b c : Nat) :
    fm (voterPhys r hr s a b c) ≤ 11 * (cf.hi + (cr.hi + (cm.hi + cl.hi))) :=
  baseline voterAlts voterIm cf cr cm cl r hr hf hg hm hl 11 (.ok (voterInit s a b c))

/-! ### Triplicated body step followed by the 11-step physical machine-code voter -/

def votedStepGen (D : PDF Nat) (g : Nat) (r : Rates) (hr : r.ok) (s : St) : PDF Out :=
  TMR.bindW D fun a =>
    TMR.bindW D fun b =>
      TMR.bindW D fun c =>
        tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1 (voterPhys r hr s a.1 b.1 c.1)

theorem votedStepGen_ok {D : PDF Nat} (hD : D.ok) (g : Nat) (r : Rates) (hr : r.ok) (s : St) :
    (votedStepGen D g r hr s).ok := by
  refine ⟨?_, ?_⟩
  · refine TMR.bindW_nonneg hD.1 fun a => TMR.bindW_nonneg hD.1 fun b =>
      TMR.bindW_nonneg hD.1 fun c => ?_
    intro y hy
    simp only [tagW, List.mem_map] at hy
    obtain ⟨z, hz, rfl⟩ := hy
    have hz0 : 0 ≤ z.2 := (voter_phys_ok r hr s a.1 b.1 c.1).1 z hz
    simpa [Rat.one_mul] using hz0
  · have hm : ∀ a b c, mass (tagW (decide (TMR.vote a b c ≠ some g)) 1 (voterPhys r hr s a b c)) = 1 := by
      intro a b c
      rw [mass_tagW, (voter_phys_ok r hr s a b c).2, Rat.one_mul]
    simp only [votedStepGen, TMR.mass_bindW, hm, TMR.sumW_const, hD.2, Rat.one_mul]

theorem votedStepGen_clean {D : PDF Nat} (hW : ∀ y ∈ D, y.1.1 < W) (g : Nat)
    (r : Rates) (hr : r.ok) (s : St) :
    ∀ u ∈ faultFree (votedStepGen D g r hr s), u = .ok (voterDone s g) := by
  intro u hu
  obtain ⟨a, ha, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨b, hb, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨c, hc, hu⟩ := TMR.mem_faultFree_bindW hu
  rw [faultFree_tagW] at hu
  split at hu
  · simp at hu
  · next hne =>
    have hv : TMR.vote a.1.1 b.1.1 c.1.1 = some g := by simpa using hne
    rw [voter_phys_clean_majority r hr s (hW a ha) (hW b hb) (hW c hc) hv] at hu
    simpa using hu

private theorem sumW_add_const {α : Type} (D : PDF α) (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + mass D * k := by
  induction D with
  | nil => show (0 : Rat) = 0 + 0 * k; rw [Rat.zero_mul, Rat.add_zero]
  | cons x r ih =>
    show x.2 * (φ x.1 + k) + TMR.sumW r (fun y => φ y + k) =
      (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 + mass r) * k
    rw [ih, Rat.mul_add, Rat.add_mul]
    calc (x.2 * φ x.1 + x.2 * k) + (TMR.sumW r φ + mass r * k)
        = x.2 * φ x.1 + (x.2 * k + (TMR.sumW r φ + mass r * k)) := Rat.add_assoc _ _ _
      _ = x.2 * φ x.1 + ((x.2 * k + TMR.sumW r φ) + mass r * k) := by rw [Rat.add_assoc (x.2 * k)]
      _ = x.2 * φ x.1 + ((TMR.sumW r φ + x.2 * k) + mass r * k) := by rw [Rat.add_comm (x.2 * k)]
      _ = x.2 * φ x.1 + (TMR.sumW r φ + (x.2 * k + mass r * k)) := by rw [Rat.add_assoc (TMR.sumW r φ)]
      _ = (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 * k + mass r * k) := (Rat.add_assoc _ _ _).symm

private theorem sumW_add_const_ok {α : Type} {D : PDF α} (hD : D.ok)
    (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + k := by
  rw [sumW_add_const, hD.2, Rat.one_mul]

private theorem fm_tri_eq_sumW {σ : Type} [DecidableEq σ] (D : PDF σ) (g : σ) :
    fm (TMR.tri D g) =
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))))) := by
  have hm : ∀ v : Option σ, fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) := fun v => by
    show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
    rw [Rat.add_zero]; rfl
  simp only [TMR.tri, TMR.fm_bindW, hm]

private theorem exists_clean_of_faultFree_eq {α : Type} {D : PDF α} {g : α}
    (h : faultFree D = [g]) : ∃ a ∈ D, a.1.1 = g ∧ a.1.2 = false := by
  induction D with
  | nil => simp [faultFree] at h
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    cases t
    · simp only [faultFree, Bool.false_eq_true, ↓reduceIte, List.cons.injEq] at h
      exact ⟨((a, false), w), List.mem_cons_self .., h.1, rfl⟩
    · simp only [faultFree, ↓reduceIte] at h
      obtain ⟨y, hy, ha, ht⟩ := ih h
      exact ⟨y, List.mem_cons_of_mem _ hy, ha, ht⟩

private theorem mem_faultFree_bindW_intro {α β : Type} {D : PDF α}
    {K : α × Bool → PDF β} {a : (α × Bool) × Rat} {z : β}
    (ha : a ∈ D) (hz : z ∈ faultFree (K a.1)) :
    z ∈ faultFree (TMR.bindW D K) := by
  induction D with
  | nil => simp at ha
  | cons x r ih =>
    simp only [TMR.bindW, faultFree_append, TMR.faultFree_scaleL, List.mem_append]
    rcases List.mem_cons.1 ha with rfl | hr
    · exact Or.inl hz
    · exact Or.inr (ih hr)

private theorem vote_self (g : Nat) : TMR.vote g g g = some g := by
  unfold TMR.vote; simp

theorem votedStepGen_mem_clean {D : PDF Nat} {g : Nat} (hg : g < W)
    (hc : faultFree D = [g]) (r : Rates) (hr : r.ok) (s : St) :
    Out.ok (voterDone s g) ∈ faultFree (votedStepGen D g r hr s) := by
  obtain ⟨a, ha, ha1, _⟩ := exists_clean_of_faultFree_eq hc
  have hphys : Out.ok (voterDone s g) ∈
      faultFree (tagW (decide (TMR.vote a.1.1 a.1.1 a.1.1 ≠ some g)) 1
        (voterPhys r hr s a.1.1 a.1.1 a.1.1)) := by
    rw [ha1]
    have htag : decide (TMR.vote g g g ≠ some g) = false := by simp [vote_self]
    rw [faultFree_tagW, if_neg (by simp [htag]),
      voter_phys_clean_majority r hr s hg hg hg (vote_self g)]
    exact List.mem_singleton.2 rfl
  exact mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha hphys))

theorem votedStepGen_bound_le {D : PDF Nat} (hD : D.ok) (g : Nat)
    (r : Rates) (hr : r.ok) (s : St) {qv : Rat} (hqv0 : 0 ≤ qv)
    (hqv : ∀ a b c, fm (voterPhys r hr s a b c) ≤ qv) :
    fm (votedStepGen D g r hr s) ≤ fm (TMR.tri D g) + qv := by
  rw [votedStepGen, TMR.fm_bindW]
  simp only [TMR.fm_bindW]
  have hcell : ∀ a b c : Nat,
      fm (tagW (decide (TMR.vote a b c ≠ some g)) 1 (voterPhys r hr s a b c)) ≤
        TMR.ι (decide (TMR.vote a b c ≠ some g)) + qv := by
    intro a b c
    rw [fm_tagW]
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
        fm (tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1
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

theorem votedStepGen_bound {D : PDF Nat} (hD : D.ok) (g : Nat)
    (r : Rates) (hr : r.ok) (s : St) :
    fm (votedStepGen D g r hr s) ≤ fm (TMR.tri D g) + 11 * r.total :=
  votedStepGen_bound_le hD g r hr s
    (Rat.mul_nonneg (by decide) (rates_total_nonneg r hr))
    (fun a b c => voter_phys_bound r hr s a b c)

theorem votedStepGen_bound_iv {D : PDF Nat} (hD : D.ok) (g : Nat)
    (cf cr cm cl : Iv) (hcf : cf.prob) (hcr : cr.prob) (hcm : cm.prob) (hcl : cl.prob)
    (r : Rates) (hr : r.ok)
    (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem) (hl : cl.mem r.nzcv)
    (s : St) :
    fm (votedStepGen D g r hr s) ≤
      fm (TMR.tri D g) + 11 * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  have hi0 : 0 ≤ cf.hi + (cr.hi + (cm.hi + cl.hi)) :=
    nonneg_add (Rat.le_trans hcf.1 hcf.2.1)
      (nonneg_add (Rat.le_trans hcr.1 hcr.2.1)
        (nonneg_add (Rat.le_trans hcm.1 hcm.2.1) (Rat.le_trans hcl.1 hcl.2.1)))
  exact votedStepGen_bound_le hD g r hr s (Rat.mul_nonneg (by decide) hi0)
    (fun a b c => voter_phys_bound_iv cf cr cm cl r hr hf hg hm hl s a b c)

/-- One triplicated body step on channel `C` followed by the 11-step physical
44-byte AArch64 voter. -/
def votedStep (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) (x : Nat) (s : St) : PDF Out :=
  votedStepGen (C.step x) (C.f x) r hr s

theorem votedStep_err (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) (x : Nat) (s : St) :
    fm (votedStep C r hr x s) ≤ 3 * (C.q * C.q) + 11 * r.total :=
  Rat.le_trans (votedStepGen_bound (C.ok x) (C.f x) r hr s)
    (add_le_add' (TMR.tri_err C x) Rat.le_refl)

/-- Headline physical TMR bound for AArch64:
a triplicated body step (rate `q_body = C.q`) followed by the 11-step physical
44-byte AArch64 voter is a valid distribution, cleans non-vacuously to
`voterDone s (C.f x)`, and faults with probability at most
`3 * q_body² + 11 * r.total`. -/
theorem voted_step_with_faulty_voter_bound (C : TMR.Channel Nat) (r : Rates) (hr : r.ok)
    (x : Nat) (s : St)
    (hfW : C.f x < W) (hW : ∀ y ∈ C.step x, y.1.1 < W) :
    (votedStep C r hr x s).ok ∧
    Out.ok (voterDone s (C.f x)) ∈ faultFree (votedStep C r hr x s) ∧
    (∀ u ∈ faultFree (votedStep C r hr x s), u = .ok (voterDone s (C.f x))) ∧
    fm (votedStep C r hr x s) ≤ 3 * (C.q * C.q) + 11 * r.total :=
  ⟨votedStepGen_ok (C.ok x) (C.f x) r hr s,
    votedStepGen_mem_clean hfW (C.clean x) r hr s,
    votedStepGen_clean hW (C.f x) r hr s,
    votedStep_err C r hr x s⟩

def votedTriFeed (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) (x : Nat) (s : St) : PDF Out :=
  bindF (TMR.tri (C.step x) (C.f x)) fun o =>
    match o with
    | some v => voterPhys r hr s v v v
    | none => voterPhys r hr s 0 1 2

theorem voted_tri_feed_bound (C : TMR.Channel Nat) (r : Rates) (hr : r.ok)
    (x : Nat) (s : St) (hx : C.f x < W) :
    (votedTriFeed C r hr x s).ok ∧
    (∀ u ∈ faultFree (votedTriFeed C r hr x s), u = .ok (voterDone s (C.f x))) ∧
    fm (votedTriFeed C r hr x s) ≤ 3 * (C.q * C.q) + 11 * r.total := by
  have hcont_ok : ∀ o : Option Nat,
      (match o with | some v => voterPhys r hr s v v v | none => voterPhys r hr s 0 1 2).ok := by
    intro o; cases o <;> exact voter_phys_ok r hr s _ _ _
  have hcont_le : ∀ o : Option Nat,
      fm (match o with | some v => voterPhys r hr s v v v | none => voterPhys r hr s 0 1 2) ≤
        11 * r.total := by
    intro o; cases o <;> exact voter_phys_bound r hr s _ _ _
  refine ⟨bindF_ok (TMR.tri_ok (C.ok x) (C.f x)) hcont_ok, ?_, ?_⟩
  · intro u hu
    simp only [votedTriFeed, faultFree_bindF, List.mem_flatMap] at hu
    obtain ⟨o, ho, hu⟩ := hu
    rw [TMR.tri_clean (C.f x) o ho] at hu
    rw [voter_phys_clean_majority r hr s hx hx hx (vote_self (C.f x))] at hu
    simpa using hu
  · have hrt : 0 ≤ 11 * r.total := Rat.mul_nonneg (by decide) (rates_total_nonneg r hr)
    exact Rat.le_trans
      (fm_bindF_le' (TMR.tri_ok (C.ok x) (C.f x)) hrt hcont_ok hcont_le)
      (add_le_add' (TMR.tri_err C x) Rat.le_refl)

/-! ### Exposure-weighted and multi-step physical voter bounds -/

def votedStepExposed (C : TMR.Channel Nat) (k : Nat) (r : Rates) (hr : r.ok)
    (x : Nat) (s : St) : PDF Out :=
  votedStepGen (TMR.exposed C k x) (TMR.iter C.f (k + 1) x) r hr s

private theorem mem_bindF {α β : Type} {d : PDF α} {f : α → PDF β}
    {y : (β × Bool) × Rat} (hy : y ∈ bindF d f) :
    ∃ x ∈ d, ∃ z ∈ f x.1.1, y.1.1 = z.1.1 := by
  induction d with
  | nil => simp [bindF] at hy
  | cons x r ih =>
    simp only [bindF, List.mem_append] at hy
    rcases hy with hy | hy
    · simp only [tagW, List.mem_map] at hy
      obtain ⟨z, hz, rfl⟩ := hy
      exact ⟨x, List.mem_cons_self .., z, hz, rfl⟩
    · obtain ⟨w, hw, z, hz, hyeq⟩ := ih hy
      exact ⟨w, List.mem_cons_of_mem _ hw, z, hz, hyeq⟩

private theorem idleK_lt_W (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < W → ∀ y ∈ C.step z, y.1.1 < W) :
    ∀ k x, x < W → ∀ y ∈ TMR.idleK C k x, y.1.1 < W
  | 0, x, hx, y, hy => by
    simp only [TMR.idleK, pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact hx
  | k + 1, x, hx, y, hy => by
    obtain ⟨w, hw, z, hz, hyeq⟩ := mem_bindF hy
    rw [hyeq]
    exact idleK_lt_W C hstepW k w.1.1 (hstepW x hx w hw) z hz

private theorem exposed_lt_W (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < W → ∀ y ∈ C.step z, y.1.1 < W)
    (k x : Nat) (hx : x < W) :
    ∀ y ∈ TMR.exposed C k x, y.1.1 < W := by
  intro y hy
  obtain ⟨w, hw, z, hz, hyeq⟩ := mem_bindF hy
  rw [hyeq]
  exact hstepW w.1.1 (idleK_lt_W C hstepW k x hx w hw) z hz

private theorem iter_lt_W {f : Nat → Nat} (hf : ∀ z, z < W → f z < W) :
    ∀ k x, x < W → TMR.iter f k x < W
  | 0, _, hx => hx
  | k + 1, x, hx => iter_lt_W hf k (f x) (hf x hx)

theorem voted_step_exposed_with_faulty_voter_bound (C : TMR.Channel Nat) (k : Nat)
    (r : Rates) (hr : r.ok) (x : Nat) (s : St) (hx : x < W)
    (hW : ∀ z, z < W → C.f z < W ∧ ∀ y ∈ C.step z, y.1.1 < W) :
    (votedStepExposed C k r hr x s).ok ∧
    Out.ok (voterDone s (TMR.iter C.f (k + 1) x)) ∈ faultFree (votedStepExposed C k r hr x s) ∧
    (∀ u ∈ faultFree (votedStepExposed C k r hr x s),
      u = .ok (voterDone s (TMR.iter C.f (k + 1) x))) ∧
    fm (votedStepExposed C k r hr x s) ≤
      3 * (((k + 1 : Nat) : Rat) * C.q) ^ 2 + 11 * r.total := by
  have hiter : TMR.iter C.f (k + 1) x < W :=
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

def votedRun (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) : Nat → Out → PDF Out
  | 0, o => pureF o
  | n + 1, .ok s =>
    match s.reg 19 with
    | some x => bindF (votedStep C r hr x s) (votedRun C r hr n)
    | none => pureF .stuck
  | _ + 1, o => pureF o

theorem votedRun_ok (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) :
    ∀ n o, (votedRun C r hr n o).ok
  | 0, o => pureF_ok o
  | n + 1, .ok s => by
    simp only [votedRun]
    cases hreg : s.reg 19 with
    | none => exact pureF_ok _
    | some x => exact bindF_ok (votedStepGen_ok (C.ok x) (C.f x) r hr s) (votedRun_ok C r hr n)
  | _ + 1, .exit c => pureF_ok _
  | _ + 1, .stuck => pureF_ok _

theorem votedRun_clean_done (C : TMR.Channel Nat) (r : Rates) (hr : r.ok)
    (hW : ∀ z, z < W → C.f z < W ∧ ∀ y ∈ C.step z, y.1.1 < W) :
    ∀ n (s : St) {x : Nat}, x < W →
      ∀ u ∈ faultFree (votedRun C r hr n (.ok (voterDone s x))),
        u = .ok (voterDone s (TMR.iter C.f n x))
  | 0, s, x, _, u, hu => by
    simpa [votedRun, faultFree_pureF, TMR.iter] using hu
  | n + 1, s, x, hx, u, hu => by
    have hreg : (voterDone s x).reg 19 = some x := rfl
    simp only [votedRun, hreg, faultFree_bindF, List.mem_flatMap] at hu
    obtain ⟨v, hv, hu⟩ := hu
    have hvEq : v = .ok (voterDone (voterDone s x) (C.f x)) :=
      votedStepGen_clean (hW x hx).2 (C.f x) r hr (voterDone s x) v hv
    rw [hvEq, voterDone_idem] at hu
    exact votedRun_clean_done C r hr hW n s (hW x hx).1 u hu

theorem votedRun_mem_clean_done (C : TMR.Channel Nat) (r : Rates) (hr : r.ok)
    (hW : ∀ z, z < W → C.f z < W ∧ ∀ y ∈ C.step z, y.1.1 < W) :
    ∀ n (s : St) {x : Nat}, x < W →
      Out.ok (voterDone s (TMR.iter C.f n x)) ∈
        faultFree (votedRun C r hr n (.ok (voterDone s x)))
  | 0, s, x, _ => by simp [votedRun, faultFree_pureF, TMR.iter]
  | n + 1, s, x, hx => by
    have hreg : (voterDone s x).reg 19 = some x := rfl
    simp only [votedRun, hreg, faultFree_bindF, List.mem_flatMap]
    have hstep := votedStepGen_mem_clean (hW x hx).1 (C.clean x) r hr (voterDone s x)
    rw [voterDone_idem] at hstep
    exact ⟨.ok (voterDone s (C.f x)), hstep, votedRun_mem_clean_done C r hr hW n s (hW x hx).1⟩

theorem votedRun_err (C : TMR.Channel Nat) (r : Rates) (hr : r.ok) :
    ∀ n o, fm (votedRun C r hr n o) ≤ (n : Rat) * (3 * (C.q * C.q) + 11 * r.total) := by
  have hsq : 0 ≤ 3 * (C.q * C.q) := by
    have := Rat.mul_nonneg C.q0 C.q0
    grind
  have hrt : 0 ≤ 11 * r.total := Rat.mul_nonneg (by decide) (rates_total_nonneg r hr)
  have hstepq : 0 ≤ 3 * (C.q * C.q) + 11 * r.total := nonneg_add hsq hrt
  intro n
  induction n with
  | zero =>
    intro o
    rw [votedRun, fm_pureF]
    show (0 : Rat) ≤ ((0 : Nat) : Rat) * _
    rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
    exact Rat.le_refl
  | succ n ih =>
    intro o
    have hn : ((n + 1 : Nat) : Rat) * (3 * (C.q * C.q) + 11 * r.total) =
        (3 * (C.q * C.q) + 11 * r.total) + (n : Rat) * (3 * (C.q * C.q) + 11 * r.total) := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    have hnn : 0 ≤ (n : Rat) * (3 * (C.q * C.q) + 11 * r.total) :=
      Rat.mul_nonneg Rat.natCast_nonneg hstepq
    rw [hn]
    cases o with
    | ok s =>
      simp only [votedRun]
      cases hreg : s.reg 19 with
      | none =>
        rw [fm_pureF]
        exact nonneg_add hstepq hnn
      | some x =>
        exact Rat.le_trans
          (fm_bindF_le' (votedStepGen_ok (C.ok x) (C.f x) r hr s) hnn (votedRun_ok C r hr n) ih)
          (add_le_add' (votedStep_err C r hr x s) Rat.le_refl)
    | exit c =>
      dsimp only [votedRun]
      rw [fm_pureF]
      exact nonneg_add hstepq hnn
    | stuck =>
      dsimp only [votedRun]
      rw [fm_pureF]
      exact nonneg_add hstepq hnn

theorem voted_run_with_faulty_voter_bound (C : TMR.Channel Nat) (r : Rates) (hr : r.ok)
    (hW : ∀ z, z < W → C.f z < W ∧ ∀ y ∈ C.step z, y.1.1 < W)
    (n : Nat) (s : St) {x : Nat} (hx : x < W) :
    (votedRun C r hr n (.ok (voterDone s x))).ok ∧
    Out.ok (voterDone s (TMR.iter C.f n x)) ∈ faultFree (votedRun C r hr n (.ok (voterDone s x))) ∧
    (∀ u ∈ faultFree (votedRun C r hr n (.ok (voterDone s x))),
      u = .ok (voterDone s (TMR.iter C.f n x))) ∧
    fm (votedRun C r hr n (.ok (voterDone s x))) ≤
      (n : Rat) * (3 * (C.q * C.q) + 11 * r.total) :=
  ⟨votedRun_ok C r hr n _,
    votedRun_mem_clean_done C r hr hW n s hx,
    votedRun_clean_done C r hr hW n s hx,
    votedRun_err C r hr n _⟩

/-! ### Concrete `p7` (`[5·10⁻⁸, 2·10⁻⁷]`) exact rational witness -/

private theorem inc_mod_lt_W (v : Nat) (_ : v < W) : (v + 1) % W < W :=
  Nat.mod_lt _ (by decide)

theorem p7_step_bound_rat :
    3 * ((1 / 5000000 : Rat) * (1 / 5000000)) + 11 * (4 / 5000000) = 220000003 / 25000000000000 := by
  decide +kernel

theorem p7_exposed9_bound_rat :
    3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 + 11 * (4 / 5000000) = 2200003 / 250000000000 := by
  decide +kernel

theorem p7_run1000_bound_rat :
    ((1000 : Nat) : Rat) * (220000003 / 25000000000000) = 220000003 / 25000000000 := by
  decide +kernel

theorem witness_p7 (r : Rates) (hr : r.ok)
    (hf : p7.mem r.fetch) (hg : p7.mem r.reg) (hm : p7.mem r.mem) (hl : p7.mem r.nzcv)
    (s : St) {x : Nat} (hx : x < W) :
    let C := regWordChannel (fun v => (v + 1) % W) r hr
    (votedStep C r hr x s).ok ∧
    Out.ok (voterDone s ((x + 1) % W)) ∈ faultFree (votedStep C r hr x s) ∧
    (∀ u ∈ faultFree (votedStep C r hr x s), u = .ok (voterDone s ((x + 1) % W))) ∧
    fm (votedStep C r hr x s) ≤ 220000003 / 25000000000000 ∧
    fm (votedStepExposed C 9 r hr x s) ≤ 2200003 / 250000000000 ∧
    fm (votedRun C r hr 1000 (.ok (voterDone s x))) ≤ 220000003 / 25000000000 := by
  intro C
  have hC : ∀ z, z < W → C.f z < W ∧ ∀ y ∈ C.step z, y.1.1 < W :=
    fun z hz => ⟨inc_mod_lt_W z hz, regWordChannel_lt_W inc_mod_lt_W r hr hz⟩
  have h1 := voted_step_with_faulty_voter_bound C r hr x s (hC x hx).1 (hC x hx).2
  have hexp := voted_step_exposed_with_faulty_voter_bound C 9 r hr x s hx hC
  have hrun := voted_run_with_faulty_voter_bound C r hr hC 1000 s hx
  have hreg_sq : r.reg * r.reg ≤ (1 / 5000000 : Rat) * (1 / 5000000) :=
    TMR.sq_le hr.2.1.1 hg.2
  have htot : r.total ≤ 4 / 5000000 := by
    have hsum : r.fetch + (r.reg + (r.mem + r.nzcv)) ≤
        (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) :=
      add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))
    have h4 : (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) = 4 / 5000000 := by
      decide +kernel
    exact Rat.le_trans hsum (h4 ▸ Rat.le_refl)
  have hstep_le : 3 * (C.q * C.q) + 11 * r.total ≤ 220000003 / 25000000000000 := by
    have h3 : 3 * (r.reg * r.reg) ≤ 3 * ((1 / 5000000 : Rat) * (1 / 5000000)) :=
      Rat.mul_le_mul_of_nonneg_left hreg_sq (by decide)
    have h11 : 11 * r.total ≤ 11 * (4 / 5000000 : Rat) :=
      Rat.mul_le_mul_of_nonneg_left htot (by decide)
    exact Rat.le_trans (add_le_add' h3 h11) (p7_step_bound_rat ▸ Rat.le_refl)
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
    have h11 : 11 * r.total ≤ 11 * (4 / 5000000 : Rat) :=
      Rat.mul_le_mul_of_nonneg_left htot (by decide)
    exact Rat.le_trans hexp.2.2.2
      (Rat.le_trans (add_le_add' h3exp h11) (p7_exposed9_bound_rat ▸ Rat.le_refl))
  · have h1000 : ((1000 : Nat) : Rat) * (3 * (C.q * C.q) + 11 * r.total) ≤
        ((1000 : Nat) : Rat) * (220000003 / 25000000000000) :=
      Rat.mul_le_mul_of_nonneg_left hstep_le Rat.natCast_nonneg
    exact Rat.le_trans hrun.2.2.2 (Rat.le_trans h1000 (p7_run1000_bound_rat ▸ Rat.le_refl))

end TMRVoterArm64
