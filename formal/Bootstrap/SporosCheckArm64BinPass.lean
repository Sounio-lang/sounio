/-
SporosCheckArm64BinPass: verified binary execution of each `spr1_check_arm64` pass
(`p1`, `p2`, `p3`) over `AArch64Phys` / `AArch64Env`, matching
`SporosCheckArm64Flat.checkFlatRes` on any input in `[b, b + bs.length)` with `b + bs.length < HMAX`.
-/
import Bootstrap.SporosCheckArm64BinCore
import Bootstrap.SporosCheckArm64Flat

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

attribute [-simp] reg_mk_self
attribute [simp] reg_setReg_eq reg_setReg_ne

set_option maxHeartbeats 800000

/-! ### Non-eta-looping state update wrappers (`withPc`, `withFl`, `withMem`) -/

def _root_.AArch64Phys.St.withPc (s : St) (pc : Nat) : St := { s with pc := pc }
def _root_.AArch64Phys.St.withFl (s : St) (fl : Flags) : St := { s with fl := fl }
def _root_.AArch64Phys.St.withMem (s : St) (m : Mem) : St := { s with mem := m }

@[simp] theorem withPc_pc (s : St) (pc : Nat) : (s.withPc pc).pc = pc := by cases s; rfl
@[simp] theorem withPc_regs (s : St) (pc : Nat) : (s.withPc pc).regs = s.regs := by cases s; rfl
@[simp] theorem withPc_reg (s : St) (pc j : Nat) : (s.withPc pc).reg j = s.reg j := by cases s; rfl
@[simp] theorem withPc_sp (s : St) (pc : Nat) : (s.withPc pc).sp = s.sp := by cases s; rfl
@[simp] theorem withPc_fl (s : St) (pc : Nat) : (s.withPc pc).fl = s.fl := by cases s; rfl
@[simp] theorem withPc_mem (s : St) (pc : Nat) : (s.withPc pc).mem = s.mem := by cases s; rfl
@[simp] theorem withPc_k (s : St) (pc : Nat) : (s.withPc pc).k = s.k := by cases s; rfl
@[simp] theorem withPc_withPc (s : St) (pc1 pc2 : Nat) : (s.withPc pc1).withPc pc2 = s.withPc pc2 := by cases s; rfl

@[simp] theorem withFl_pc (s : St) (fl : Flags) : (s.withFl fl).pc = s.pc := by cases s; rfl
@[simp] theorem withFl_regs (s : St) (fl : Flags) : (s.withFl fl).regs = s.regs := by cases s; rfl
@[simp] theorem withFl_reg (s : St) (fl : Flags) (j : Nat) : (s.withFl fl).reg j = s.reg j := by cases s; rfl
@[simp] theorem withFl_sp (s : St) (fl : Flags) : (s.withFl fl).sp = s.sp := by cases s; rfl
@[simp] theorem withFl_fl (s : St) (fl : Flags) : (s.withFl fl).fl = fl := by cases s; rfl
@[simp] theorem withFl_mem (s : St) (fl : Flags) : (s.withFl fl).mem = s.mem := by cases s; rfl
@[simp] theorem withFl_k (s : St) (fl : Flags) : (s.withFl fl).k = s.k := by cases s; rfl
@[simp] theorem withFl_withFl (s : St) (f1 f2 : Flags) : (s.withFl f1).withFl f2 = s.withFl f2 := by cases s; rfl

@[simp] theorem withMem_pc (s : St) (m : Mem) : (s.withMem m).pc = s.pc := by cases s; rfl
@[simp] theorem withMem_regs (s : St) (m : Mem) : (s.withMem m).regs = s.regs := by cases s; rfl
@[simp] theorem withMem_reg (s : St) (m : Mem) (j : Nat) : (s.withMem m).reg j = s.reg j := by cases s; rfl
@[simp] theorem withMem_sp (s : St) (m : Mem) : (s.withMem m).sp = s.sp := by cases s; rfl
@[simp] theorem withMem_fl (s : St) (m : Mem) : (s.withMem m).fl = s.fl := by cases s; rfl
@[simp] theorem withMem_mem (s : St) (m : Mem) : (s.withMem m).mem = m := by cases s; rfl
@[simp] theorem withMem_k (s : St) (m : Mem) : (s.withMem m).k = s.k := by cases s; rfl

@[simp] theorem withPc_withFl (s : St) (fl : Flags) (pc : Nat) :
    (s.withFl fl).withPc pc = (s.withPc pc).withFl fl := by cases s; rfl
@[simp] theorem withPc_setReg (s : St) (i : Nat) (v : Option Nat) (pc : Nat) :
    (s.setReg i v).withPc pc = (s.withPc pc).setReg i v := by cases s; rfl
@[simp] theorem withFl_setReg (s : St) (i : Nat) (v : Option Nat) (fl : Flags) :
    (s.setReg i v).withFl fl = (s.withFl fl).setReg i v := by cases s; rfl
@[simp] theorem setReg_setReg_same (s : St) (i : Nat) (v1 v2 : Option Nat) :
    (s.setReg i v1).setReg i v2 = s.setReg i v2 := by
  cases s; simp [St.setReg]

/-! ### Refined step/frame helpers preserving `AArch64Env.W` and `2 ^ 64` shapes -/

theorem reach_at_pc {s : St} {a : Nat} {ins : Ins} {len : Nat} (hf : im a = some (ins, len))
    (ha : s.pc = a)
    {P : St → Prop}
    (h : match exec ins len (s.withPc a) with | .ok s' => Reaches s' P | _ => False) :
    Reaches s P := by
  have hs : s.withPc a = s := by cases s; cases ha; rfl
  rw [hs] at h
  exact reach_at hf ha h

theorem HMAX_flat_eq : SporosCheckArm64Flat.HMAX = SporosCheckArm64Bin.HMAX := rfl

theorem pow2_0 : pow2 0 = 1 := rfl
theorem pow2_hw0 : pow2 (0 * 16) = 1 := rfl
theorem pow2_hw1 : pow2 (1 * 16) = 2 ^ 16 := rfl
theorem pow2_hw2 : pow2 (2 * 16) = 2 ^ 32 := rfl
theorem pow2_hw3 : pow2 (3 * 16) = 2 ^ 48 := rfl
theorem pow2_hw0p : pow2 (0 * 16 + 16) = 2 ^ 16 := rfl
theorem pow2_hw1p : pow2 (1 * 16 + 16) = 2 ^ 32 := rfl
theorem pow2_hw2p : pow2 (2 * 16 + 16) = 2 ^ 48 := rfl
theorem pow2_hw3p : pow2 (3 * 16 + 16) = 2 ^ 64 := rfl

theorem shiftVal_lsl0_mod (w v : Nat) : shiftVal .lsl w (v % pow2 w) 0 = v % pow2 w := by
  simp [shiftVal, pow2]

theorem shiftVal_lsl0 (w v : Nat) : shiftVal .lsl w v 0 = v % pow2 w := by
  simp [shiftVal, pow2]

theorem shiftVal_lsl1 (v : Nat) : shiftVal .lsl 64 v 1 = (v % 2 ^ 64 * 2 ^ 1) % 2 ^ 64 := Eq.trans rfl rfl
theorem shiftVal_lsl2 (v : Nat) : shiftVal .lsl 64 v 2 = (v % 2 ^ 64 * 2 ^ 2) % 2 ^ 64 := Eq.trans rfl rfl
theorem shiftVal_lsl32 (v : Nat) : shiftVal .lsl 64 v 32 = (v % 2 ^ 64 * 2 ^ 32) % 2 ^ 64 := Eq.trans rfl rfl
theorem shiftVal_lsr32 (v : Nat) : shiftVal .lsr 64 v 32 = v % 2 ^ 64 / 2 ^ 32 := Eq.trans rfl rfl

theorem addSubOp_add_fst (w a b : Nat) : (addSubOp false w a b).1 = (a + b) % pow2 w := rfl
theorem addSubOp_sub_fst (w a b : Nat) : (addSubOp true w a b).1 = (a + pow2 w - b) % pow2 w := rfl
theorem lor_zero_left (v : Nat) : Nat.lor 0 v = v := Nat.zero_or v
theorem mod_1_64 : 1 % 2 ^ 64 = 1 := by decide
theorem mod_2_64 : 2 % 2 ^ 64 = 2 := by decide
theorem mod_12_64 : 12 % 2 ^ 64 = 12 := by decide

theorem wrZR_64_ne31 {s : St} {rt v : Nat} (hne : rt ≠ 31) :
    s.wrZR 64 rt v = s.setReg rt (some (v % 2 ^ 64)) := by
  rw [wrZR_ne31 hne, pow2_64]

theorem wrZR_32_lt {s : St} {rt v : Nat} (hne : rt ≠ 31) (hv : v < 2 ^ 32) :
    s.wrZR 32 rt v = s.setReg rt (some v) := by
  rw [wrZR_ne31 hne, pow2_32, Nat.mod_eq_of_lt hv]

theorem mod_mod_32 (v : Nat) : (v % 2 ^ 32) % 2 ^ 32 = v % 2 ^ 32 := Nat.mod_mod _ _
theorem mod_mod_64 (v : Nat) : (v % 2 ^ 64) % 2 ^ 64 = v % 2 ^ 64 := Nat.mod_mod _ _
theorem nat_bne_eq (a b : Nat) : (a != b) = !(a == b) := rfl

theorem exec_movz_64' {rd imm16 hw : Nat} {s : St} {len : Nat} :
    exec (.movWide 2 64 rd imm16 hw) len s =
      .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR 64 rd (imm16 * pow2 (hw * 16))) := by cases s; rfl

theorem exec_movn_64' {rd imm16 hw : Nat} {s : St} {len : Nat} :
    exec (.movWide 0 64 rd imm16 hw) len s =
      .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR 64 rd (pow2 64 - 1 - imm16 * pow2 (hw * 16))) := by cases s; rfl

theorem xmovk' {w rd imm16 hw : Nat} {s : St} {len : Nat} :
    exec (.movWide 3 w rd imm16 hw) len s =
      match s.rdZR w rd with
      | some old =>
        let sh := hw * 16
        .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR w rd
          (old % pow2 sh + imm16 * pow2 sh + old / pow2 (sh + 16) * pow2 (sh + 16)))
      | none => .stuck := by
  rw [xmovk]; cases s.rdZR w rd <;> rfl

private theorem wrSP_mod (s : St) (w rd v : Nat) :
    s.wrSP w rd (v % pow2 w) = s.wrSP w rd v := by
  unfold St.wrSP; split <;> simp

private theorem wrZR_mod (s : St) (w rd v : Nat) :
    s.wrZR w rd (v % pow2 w) = s.wrZR w rd v := by
  unfold St.wrZR; split
  · rfl
  · simp

theorem xaddSubImm_add {w rd rn imm : Nat} {s : St} {len : Nat} :
    exec (.addSubImm false false w rd rn imm) len s =
      match s.rdSP w rn with
      | some a => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrSP w rd (a + imm % pow2 w))
      | none => .stuck := by
  rw [xaddSubImm]; cases s.rdSP w rn with
  | none => rfl
  | some a => exact congrArg Out.ok (wrSP_mod _ _ _ _)

theorem xaddSubImm_sub {w rd rn imm : Nat} {s : St} {len : Nat} :
    exec (.addSubImm true false w rd rn imm) len s =
      match s.rdSP w rn with
      | some a => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrSP w rd (a + pow2 w - imm % pow2 w))
      | none => .stuck := by
  rw [xaddSubImm]; cases s.rdSP w rn with
  | none => rfl
  | some a => exact congrArg Out.ok (wrSP_mod _ _ _ _)

theorem xaddSubImm_cmp {w rn imm : Nat} {s : St} {len : Nat} :
    exec (.addSubImm true true w 31 rn imm) len s =
      match s.rdSP w rn with
      | some a => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).withFl (addSubOp true w a (imm % pow2 w)).2)
      | none => .stuck := by
  rw [xaddSubImm]; cases s.rdSP w rn <;> rfl

theorem xaddSubReg_add {w rd rn rm : Nat} {sh : ShiftKind} {amt : Nat} {s : St} {len : Nat} :
    exec (.addSubReg false false w rd rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR w rd (a + shiftVal sh w b amt))
      | _, _ => .stuck := by
  rw [xaddSubReg]; cases s.rdZR w rn with
  | none => rfl
  | some a => cases s.rdZR w rm with
    | none => rfl
    | some b => exact congrArg Out.ok (wrZR_mod _ _ _ _)

theorem xaddSubReg_sub {w rd rn rm : Nat} {sh : ShiftKind} {amt : Nat} {s : St} {len : Nat} :
    exec (.addSubReg true false w rd rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR w rd (a + pow2 w - shiftVal sh w b amt))
      | _, _ => .stuck := by
  rw [xaddSubReg]; cases s.rdZR w rn with
  | none => rfl
  | some a => cases s.rdZR w rm with
    | none => rfl
    | some b => exact congrArg Out.ok (wrZR_mod _ _ _ _)

theorem xaddSubReg_cmp {w rn rm : Nat} {sh : ShiftKind} {amt : Nat} {s : St} {len : Nat} :
    exec (.addSubReg true true w 31 rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).withFl (addSubOp true w a (shiftVal sh w b amt)).2)
      | _, _ => .stuck := by
  rw [xaddSubReg]; cases s.rdZR w rn <;> cases s.rdZR w rm <;> rfl

theorem xlogicReg' {op : LogicOp} {inv : Bool} {w rd rn rm : Nat} {sh : ShiftKind} {amt : Nat}
    {s : St} {len : Nat} :
    exec (.logicReg op inv w rd rn rm sh amt) len s =
      match s.rdZR w rn, s.rdZR w rm with
      | some a, some b =>
        let bSh := shiftVal sh w b amt
        let bOp := if inv then pow2 w - 1 - bSh else bSh
        let res := match op with
          | .and | .ands => Nat.land a bOp
          | .orr => Nat.lor a bOp
          | .eor => Nat.xor a bOp
        let s1 := s.withPc ((s.pc + len) % AArch64Env.W)
        if op == .ands then
          let f : Flags := ⟨some (msb w res), some (res == 0), some false, some false⟩
          .ok ((s1.withFl f).wrZR w rd res)
        else .ok (s1.wrZR w rd res)
      | _, _ => .stuck := by
  rw [xlogicReg]; cases s.rdZR w rn <;> cases s.rdZR w rm <;> rfl

theorem exec_adr' {isPage : Bool} {rd imm : Nat} {s : St} {len : Nat} :
    exec (.adr isPage rd imm) len s =
      let base := if isPage then (s.pc / 4096) * 4096 else s.pc
      .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR 64 rd ((base + imm) % AArch64Env.W)) := by cases s; rfl

theorem xldr' {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} :
    exec (.ldr sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match rdMem s.mem ((base + off) % AArch64Env.W) sz with
          | some v => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR sz.width rt v)
          | none => .stuck
      | none => .stuck := by
  rw [xldr]; cases s.rdSP 64 rn <;> rfl

theorem xldur' {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} :
    exec (.ldur sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match rdMem s.mem ((base + off) % AArch64Env.W) sz with
          | some v => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR sz.width rt v)
          | none => .stuck
      | none => .stuck := by
  rw [xldur]; cases s.rdSP 64 rn <;> rfl

theorem xstr' {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} :
    exec (.str sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        if rn == 31 && base % 16 != 0 then .stuck
        else match s.rdZR sz.width rt with
          | some v =>
            match wrMem s.mem ((base + off) % AArch64Env.W) v sz with
            | some m' => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).withMem m')
            | none => .stuck
          | none => .stuck
      | none => .stuck := by
  rw [xstr]; cases s.rdSP 64 rn <;> rfl

theorem xldr_ne31 {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} (hne : rn ≠ 31) :
    exec (.ldr sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        match rdMem s.mem ((base + off) % AArch64Env.W) sz with
        | some v => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR sz.width rt v)
        | none => .stuck
      | none => .stuck := by
  rw [xldr']; cases s.rdSP 64 rn with
  | none => rfl
  | some base => simp [hne]

theorem xldur_ne31 {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} (hne : rn ≠ 31) :
    exec (.ldur sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        match rdMem s.mem ((base + off) % AArch64Env.W) sz with
        | some v => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).wrZR sz.width rt v)
        | none => .stuck
      | none => .stuck := by
  rw [xldur']; cases s.rdSP 64 rn with
  | none => rfl
  | some base => simp [hne]

theorem xstr_ne31 {sz : MemSize} {rt rn off : Nat} {s : St} {len : Nat} (hne : rn ≠ 31) :
    exec (.str sz rt rn off) len s =
      match s.rdSP 64 rn with
      | some base =>
        match s.rdZR sz.width rt with
        | some v =>
          match wrMem s.mem ((base + off) % AArch64Env.W) v sz with
          | some m' => .ok ((s.withPc ((s.pc + len) % AArch64Env.W)).withMem m')
          | none => .stuck
        | none => .stuck
      | none => .stuck := by
  rw [xstr']; cases s.rdSP 64 rn with
  | none => rfl
  | some base => simp [hne]

theorem exec_b_false' {off : Nat} {s : St} {len : Nat} :
    exec (.b false off) len s = .ok (s.withPc ((s.pc + off) % AArch64Env.W)) := by cases s; rfl

theorem xbcond' {cc off : Nat} {s : St} {len : Nat} :
    exec (.bcond cc off) len s =
      match AArch64Phys.cond s.fl cc with
      | some true => .ok (s.withPc ((s.pc + off) % AArch64Env.W))
      | some false => .ok (s.withPc ((s.pc + len) % AArch64Env.W))
      | none => .stuck := by
  rw [xbcond]; cases AArch64Phys.cond s.fl cc with
  | none => rfl
  | some b => cases b <;> rfl

theorem xcbz' {nz : Bool} {w rt off : Nat} {s : St} {len : Nat} :
    exec (.cbz nz w rt off) len s =
      match s.rdZR w rt with
      | some v =>
        let taken := if nz then v != 0 else v == 0
        .ok (s.withPc (if taken then (s.pc + off) % AArch64Env.W else (s.pc + len) % AArch64Env.W))
      | none => .stuck := by
  rw [xcbz]; cases s.rdZR w rt <;> rfl

macro_rules
  | `(tactic| xs $f) =>
    `(tactic| (refine reach_at_pc $f (by first | (simp only [setReg_pc, wrZR_pc, withPc_pc, withFl_pc, withMem_pc, pc_mk]; try omega) | (simp only [setReg_pc, wrZR_pc, withPc_pc, withFl_pc, withMem_pc, pc_mk, W_eq, Nat.reducePow] at *; try omega) | rfl | omega) ?_;
               simp (config := { decide := true }) only [exec_movz_64', exec_movn_64', xmovk',
                 xaddSubImm_add, xaddSubImm_sub, xaddSubImm_cmp,
                 xaddSubReg_add, xaddSubReg_sub, xaddSubReg_cmp,
                 xlogicReg', exec_adr', xldr_ne31, xldur_ne31, xstr_ne31,
                 exec_b_false', xbcond', xcbz',
                 St.rdZR, St.rdSP, Option.map_some, beq_iff_eq, beq_eq_false_iff_ne, beq_false_of_ne,
                 beq_self_eq_true, nat_bne_eq, Bool.not_false, Bool.not_true,
                 rdZR_31, rdZR_of_reg, rdSP_of_reg, wrZR_31, MemSize.width, wrZR_32_lt, wrZR_64_ne31, wrSP_ne31,
                 lor_zero_left, mod_mod_32, mod_mod_64,
                 mod_1_64, mod_2_64, mod_12_64, Nat.zero_add,
                 List.length_set,
                 reg_setReg_eq, reg_setReg_ne, setReg_regs, setReg_regs_length,
                 setReg_sp, setReg_pc, setReg_fl, setReg_mem, setReg_k, setReg_setReg_same,
                 wrZR_regs_length, wrZR_sp, wrZR_pc, wrZR_fl, wrZR_mem, wrZR_k,
                 withPc_pc, withPc_regs, withPc_reg, withPc_sp, withPc_fl, withPc_mem, withPc_k, withPc_withPc,
                 withPc_withFl, withPc_setReg,
                 withFl_pc, withFl_regs, withFl_reg, withFl_sp, withFl_fl, withFl_mem, withFl_k, withFl_withFl,
                 withFl_setReg,
                 withMem_pc, withMem_regs, withMem_reg, withMem_sp, withMem_fl, withMem_mem, withMem_k,
                 shiftVal_lsl0_mod, shiftVal_lsl0, shiftVal_lsl1, shiftVal_lsl2, shiftVal_lsl32, shiftVal_lsr32,
                 pow2_hw0, pow2_hw1, pow2_hw2, pow2_hw3,
                 pow2_hw0p, pow2_hw1p, pow2_hw2p, pow2_hw3p,
                 pow2_64, pow2_32, pow2_0,
                 ↓reduceIte, reduceCtorEq, Bool.false_eq_true, Bool.true_eq_false,
                 Bool.false_and, Bool.and_false,
                 decide_true, decide_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true]))
  | `(tactic| xs $f using $hs,*) =>
    `(tactic| (refine reach_at_pc $f (by first | (simp only [setReg_pc, wrZR_pc, withPc_pc, withFl_pc, withMem_pc, pc_mk]; try omega) | (simp only [setReg_pc, wrZR_pc, withPc_pc, withFl_pc, withMem_pc, pc_mk, W_eq, Nat.reducePow] at *; try omega) | rfl | omega) ?_;
               simp (config := { decide := true }) only [exec_movz_64', exec_movn_64', xmovk',
                 xaddSubImm_add, xaddSubImm_sub, xaddSubImm_cmp,
                 xaddSubReg_add, xaddSubReg_sub, xaddSubReg_cmp,
                 xlogicReg', exec_adr', xldr_ne31, xldur_ne31, xstr_ne31,
                 exec_b_false', xbcond', xcbz',
                 St.rdZR, St.rdSP, Option.map_some, beq_iff_eq, beq_eq_false_iff_ne, beq_false_of_ne,
                 beq_self_eq_true, nat_bne_eq, Bool.not_false, Bool.not_true,
                 rdZR_31, rdZR_of_reg, rdSP_of_reg, wrZR_31, MemSize.width, wrZR_32_lt, wrZR_64_ne31, wrSP_ne31,
                 lor_zero_left, mod_mod_32, mod_mod_64,
                 mod_1_64, mod_2_64, mod_12_64, Nat.zero_add,
                 List.length_set,
                 reg_setReg_eq, reg_setReg_ne, setReg_regs, setReg_regs_length,
                 setReg_sp, setReg_pc, setReg_fl, setReg_mem, setReg_k, setReg_setReg_same,
                 wrZR_regs_length, wrZR_sp, wrZR_pc, wrZR_fl, wrZR_mem, wrZR_k,
                 withPc_pc, withPc_regs, withPc_reg, withPc_sp, withPc_fl, withPc_mem, withPc_k, withPc_withPc,
                 withPc_withFl, withPc_setReg,
                 withFl_pc, withFl_regs, withFl_reg, withFl_sp, withFl_fl, withFl_mem, withFl_k, withFl_withFl,
                 withFl_setReg,
                 withMem_pc, withMem_regs, withMem_reg, withMem_sp, withMem_fl, withMem_mem, withMem_k,
                 shiftVal_lsl0_mod, shiftVal_lsl0, shiftVal_lsl1, shiftVal_lsl2, shiftVal_lsl32, shiftVal_lsr32,
                 pow2_hw0, pow2_hw1, pow2_hw2, pow2_hw3,
                 pow2_hw0p, pow2_hw1p, pow2_hw2p, pow2_hw3p,
                 pow2_64, pow2_32, pow2_0,
                 ↓reduceIte, reduceCtorEq, Bool.false_eq_true, Bool.true_eq_false,
                 Bool.false_and, Bool.and_false,
                 decide_true, decide_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true, $hs,*]))

/-! ### Little-endian buffer bridges (`rdU32`, `rdMem`) and PC offset helpers -/

theorem relOff_add {B off tgt : Nat} (hB : B + 2728 < 2 ^ 64)
    (_hoff : off ≤ 2728) (htgt : tgt ≤ 2728) :
    (B + off + relOff (B + off) (B + tgt)) % AArch64Env.W = B + tgt := by
  unfold relOff; rw [W_eq]; omega

theorem modW_small {x : Nat} (h : x < 2 ^ 64) : x % AArch64Env.W = x := by
  rw [W_eq]; exact Nat.mod_eq_of_lt h

theorem ea_mod_small {x off : Nat} (hx : x < 2 ^ 64) (hxo : x + off < 2 ^ 64) :
    (x % 2 ^ 64 + off) % AArch64Env.W = x + off := by
  rw [W_eq, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hxo]

theorem rdU32_eq_some_u32 {bs : List Nat} (_hby : ∀ x ∈ bs, x < 256)
    {pos : Nat} (hp : pos + 4 ≤ bs.length) :
    rdU32 bs pos = some (u32 bs pos) := by
  have h0 : pos < bs.length := by omega
  have h1 : pos + 1 < bs.length := by omega
  have h2 : pos + 2 < bs.length := by omega
  have h3 : pos + 3 < bs.length := by omega
  have htake : (bs.drop pos).take 4 = [bs[pos], bs[pos + 1], bs[pos + 2], bs[pos + 3]] := by
    apply List.ext_getElem
    · simp; omega
    · intro i hi1 hi2
      rcases (show i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 by simp at hi2; omega) with
        rfl | rfl | rfl | rfl <;> simp [List.getElem_take, List.getElem_drop]
  simp only [rdU32, u32, htake, List.getElem?_eq_getElem h0, List.getElem?_eq_getElem h1,
    List.getElem?_eq_getElem h2, List.getElem?_eq_getElem h3, bind, Option.bind, pure, leVal]
  congr 1; omega

theorem rdU32_eq_none_of_lt {bs : List Nat} {pos : Nat} (hp : bs.length < pos + 4) :
    rdU32 bs pos = none := by
  cases h : rdU32 bs pos with
  | none => rfl
  | some v => exact absurd (rdU32_bounds h) (by omega)

theorem u32_lt_HMAX {bs : List Nat} (hby : ∀ x ∈ bs, x < 256) (pos : Nat) :
    u32 bs pos < SporosCheckArm64Flat.HMAX :=
  Nat.lt_trans (u32_lt hby pos) (by decide)

theorem rdMem_b1_eq {m : Mem} {b : Nat} {bs : List Nat} (hbuf : Buf m b bs)
    {pos : Nat} (hp : pos < bs.length) (x : Nat) (hx : bs[pos]? = some x) :
    rdMem m (b + pos) .b1 = some x := by
  have hge : bs[pos]? = some bs[pos] := List.getElem?_eq_getElem hp
  rw [hge] at hx; cases hx
  exact rdMem_b1_buf hbuf hp

/-! ### Balanced binary search tree for primitive stack effects (`EffTree`) -/

inductive EffTree where
  | leaf (eIn eOut : Nat)
  | branch (mid : Nat) (l r : EffTree)
deriving DecidableEq, Repr

def EffTree.bytes : EffTree → Nat
  | .leaf _ _ => 12
  | .branch _ l r => 8 + l.bytes + r.bytes

def EffTree.wf : EffTree → Bool
  | .leaf i o => i < 65536 && o < 65536
  | .branch m l r => m < 4096 && l.wf && r.wf && 8 + l.bytes + r.bytes < 4096

def EffTree.eval : EffTree → Nat → Nat × Nat
  | .leaf i o, _ => (i, o)
  | .branch m l r, op => if op < m then l.eval op else r.eval op

def EffTree.emit : EffTree → Nat → Nat → List (Nat × Ins × Nat)
  | .leaf i o, L, tgt =>
    [(L, .movWide 2 64 10 i 0, 4),
     (L + 4, .movWide 2 64 11 o 0, 4),
     (L + 8, .b false (relOff (L + 8) tgt), 4)]
  | .branch m l r, L, tgt =>
    [(L, .addSubImm true true 64 31 15 m, 4),
     (L + 4, .bcond 2 (l.bytes + 4), 4)] ++
    l.emit (L + 8) tgt ++
    r.emit (L + 8 + l.bytes) tgt

def effRuns : List (Nat × Nat × Nat) :=
  [(0, 1, 2), (1, 1, 0), (2, 2, 2), (3, 2, 3), (4, 3, 3), (5, 2, 1), (6, 2, 3),
   (7, 2, 1), (14, 1, 1), (15, 2, 1), (18, 1, 1), (19, 2, 1),
   (28, 1, 1), (29, 2, 0), (30, 1, 1), (31, 2, 0), (32, 1, 1), (33, 2, 0),
   (34, 1, 1), (35, 2, 0), (36, 3, 0), (38, 3, 1), (39, 1, 1), (40, 0, 1),
   (41, 1, 0), (42, 4, 1), (43, 2, 1), (44, 0, 1), (45, 3, 0), (46, 2, 2),
   (47, 0, 1), (48, 1, 0), (51, 0, 1), (53, 1, 1), (54, 1, 2), (56, 2, 1)]

def buildEffTree : List (Nat × Nat × Nat) → Nat → EffTree
  | _, 0 => .leaf 0 0
  | rs, fuel + 1 =>
    match rs with
    | [] => .leaf 0 0
    | [(_, i, o)] => .leaf i o
    | _ =>
      let mid := rs.length / 2
      let m := (rs[mid]?).map (·.1) |>.getD 0
      .branch m (buildEffTree (rs.take mid) fuel) (buildEffTree (rs.drop mid) fuel)

def primEffTree : EffTree := buildEffTree effRuns effRuns.length

theorem primEffTree_bytes : primEffTree.bytes = 712 := by decide +kernel
theorem primEffTree_wf : primEffTree.wf = true := by decide +kernel

theorem primEffTree_spec : ∀ op : Nat, op < 57 →
    primEff op = some (primEffTree.eval op) := by decide +kernel

theorem primEffTree_bound : ∀ op : Nat, op < 57 →
    (primEffTree.eval op).1 ≤ 4 ∧ (primEffTree.eval op).2 ≤ 4 := by decide +kernel

theorem primEffTree_emit_eq (B : Nat) :
    primEffTree.emit (B + 904) (B + 1616) = effT (B + 904) (B + 1616) := by
  have htree : primEffTree =
      buildEffTree effRuns 36 := rfl
  rw [htree]
  simp only [effRuns, buildEffTree, EffTree.emit, EffTree.bytes, effT,
    List.take, List.drop, List.length, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.map_some, Option.getD_some, Nat.add_assoc, List.cons_append, List.nil_append,
    Nat.reduceAdd, Nat.reduceDiv]

private theorem relOff_target {pc tgt : Nat} (hpc : pc < 2 ^ 64) (htgt : tgt < 2 ^ 64) :
    (pc + relOff pc tgt) % AArch64Env.W = tgt := by
  unfold relOff; rw [W_eq]
  rw [Nat.mod_eq_of_lt hpc]
  by_cases h : pc ≤ tgt
  · have : (tgt + 2 ^ 64 - pc) % 2 ^ 64 = tgt - pc := by
      rw [show tgt + 2 ^ 64 - pc = (tgt - pc) + 2 ^ 64 by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    rw [this, show pc + (tgt - pc) = tgt by omega]
    exact Nat.mod_eq_of_lt htgt
  · have : (tgt + 2 ^ 64 - pc) % 2 ^ 64 = tgt + 2 ^ 64 - pc := Nat.mod_eq_of_lt (by omega)
    rw [this, show pc + (tgt + 2 ^ 64 - pc) = tgt + 2 ^ 64 by omega, Nat.add_mod_right]
    exact Nat.mod_eq_of_lt htgt

private theorem relOff_any {L off tgt : Nat} (hL : L + off < 2 ^ 64) (htgt : tgt < 2 ^ 64) :
    (L + off + relOff (L + off) tgt) % AArch64Env.W = tgt :=
  relOff_target hL htgt

def PassScratch : List Nat := [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25]
def PassRegs (R : Nat) : List Nat := R :: PassScratch

theorem ne_of_mem_not_mem {S : List Nat} {i j : Nat} (hi : i ∈ S) (hj : j ∉ S) : i ≠ j :=
  fun h => hj (h ▸ hi)

macro_rules
  | `(tactic| frame_tac) =>
    `(tactic| (
      refine frame_regs (fun j hj => ?_) (by simp) (by simp) (by simp) (by simp)
      repeat (first
        | (rw [reg_wrZR_ne]; rotate_left; exact ne_of_mem_not_mem (by first | decide | simp [PassScratch, PassRegs]) hj)
        | (rw [reg_setReg_ne]; rotate_left; exact ne_of_mem_not_mem (by first | decide | simp [PassScratch, PassRegs]) hj)
        | (rw [getD_set_ne]; rotate_left; exact ne_of_mem_not_mem (by first | decide | simp [PassScratch, PassRegs]) hj)
        | simp only [withPc_reg]
        | simp only [withFl_reg]
        | simp only [withMem_reg])
      try rfl))

theorem eff_tree_spec : ∀ (t : EffTree) (L tgt op : Nat) (s : St),
    t.wf = true → CodeAt (t.emit L tgt) →
    L + t.bytes ≤ tgt → tgt < 2 ^ 64 →
    s.regs.length = 31 → s.pc = L → s.reg 15 = some op → op < 256 →
    Reaches s fun s' => Frame [10, 11] s s' ∧
      s'.pc = tgt ∧ s'.reg 10 = some (t.eval op).1 ∧ s'.reg 11 = some (t.eval op).2 ∧
      s'.regs.length = 31 ∧ s'.reg 15 = some op
  | .leaf i o, L, tgt, op, s, hwf, hc, hle, htgt, hl, hpc, h15, _ => by
    simp only [EffTree.wf, Bool.and_eq_true, decide_eq_true_eq] at hwf
    obtain ⟨hi, ho⟩ := hwf
    simp only [EffTree.bytes] at hle
    simp only [EffTree.emit, CodeAt_cons] at hc
    obtain ⟨f0, f1, f2, _⟩ := hc
    have hr0 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
    have hr1 : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
    have hr2 : (L + 8 + relOff (L + 8) tgt) % AArch64Env.W = tgt := relOff_any (by omega) htgt
    xs f0 using hl, hr0
    xs f1 using hl, hr1
    xs f2 using hl, hr2
    exact Reaches.refl ⟨by frame_tac, rfl,
      (by simp [EffTree.eval, hl, Nat.mod_eq_of_lt (by omega : i < 2 ^ 64)]),
      (by simp [EffTree.eval, hl, Nat.mod_eq_of_lt (by omega : o < 2 ^ 64)]),
      (by simp [hl]), (by simp [h15])⟩
  | .branch m l r, L, tgt, op, s, hwf, hc, hle, htgt, hl, hpc, h15, hop => by
    simp only [EffTree.wf, Bool.and_eq_true, decide_eq_true_eq] at hwf
    obtain ⟨⟨⟨hm, hwl⟩, hwr⟩, hsz⟩ := hwf
    simp only [EffTree.bytes] at hle
    simp only [EffTree.emit, CodeAt_append, CodeAt_cons] at hc
    obtain ⟨⟨⟨f0, f1, _⟩, hcl⟩, hcr⟩ := hc
    have hr0 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
    have hr1f : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
    have hr1t : (L + 4 + (l.bytes + 4)) % AArch64Env.W = L + 8 + l.bytes := by
      rw [modW_small (by omega)]; omega
    have hcs := cond_sub (show op < pow2 64 by rw [pow2_64]; omega)
      (show m < pow2 64 by rw [pow2_64]; omega)
    xs f0 using hl, h15, hr0
    by_cases hlt : op < m
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (m % 2 ^ 64)).2 2 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]
        simp [hlt]
      xs f1 using hfl2, hr1f
      refine (eff_tree_spec l (L + 8) tgt op _ hwl hcl (by omega) htgt (by simp [hl])
        rfl (by simp [h15]) hop).mono fun s' ⟨hf, hpc', h10, h11, hl', h15'⟩ => ?_
      exact ⟨Frame.trans_same (by frame_tac) hf, hpc',
        (by simp [EffTree.eval, hlt, h10]), (by simp [EffTree.eval, hlt, h11]), hl', h15'⟩
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (m % 2 ^ 64)).2 2 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]
        simp [show m ≤ op by omega]
      xs f1 using hfl2, hr1t
      refine (eff_tree_spec r (L + 8 + l.bytes) tgt op _ hwr hcr (by omega) htgt (by simp [hl])
        rfl (by simp [h15]) hop).mono fun s' ⟨hf, hpc', h10, h11, hl', h15'⟩ => ?_
      exact ⟨Frame.trans_same (by frame_tac) hf, hpc',
        (by simp [EffTree.eval, hlt, h10]), (by simp [EffTree.eval, hlt, h11]), hl', h15'⟩

/-! ### Pass instruction decomposition and `PassBlocks` -/

def dispatchCode (a : Nat) : List (Nat × Nat) → List (Nat × Ins × Nat)
  | [] => []
  | (k, tgt) :: rest =>
    (a, .addSubImm true true 64 31 15 k, 4) ::
    (a + 4, .bcond 0 (relOff (a + 4) tgt), 4) ::
    dispatchCode (a + 8) rest

def dispatchTgt (op defTgt : Nat) : List (Nat × Nat) → Nat
  | [] => defTgt
  | (k, tgt) :: rest => if op == k then tgt else dispatchTgt op defTgt rest

def verdictInsns (B R v : Nat) : List (Nat × Ins × Nat) :=
  let v0 := v % 65536
  let v1 := (v / 65536) % 65536
  let v2 := (v / 2 ^ 32) % 65536
  let v3 := (v / 2 ^ 48) % 65536
  [(B, .movWide 2 64 R v0 0, 4),
   (B + 4, .movWide 3 64 R v1 1, 4),
   (B + 8, .movWide 3 64 R v2 2, 4),
   (B + 12, .movWide 3 64 R v3 3, 4)]

def passCode (B adrImm R nxt : Nat) : List (Nat × Ins × Nat) :=
  let rej := B + 2708
  let ovf := B + 2688
  let acc := B + 2668
  [(B + 0, .adr true 15 adrImm, 4),
   (B + 4, .ldr .x8 0 15 0, 4),
   (B + 8, .ldr .x8 14 15 8, 4),
   (B + 12, .addSubReg false false 64 24 0 14 .lsl 0, 4),
   (B + 16, .addSubImm true true 64 31 14 16, 4),
   (B + 20, .bcond 3 (relOff (B + 20) rej), 4),
   (B + 24, .ldr .w4 14 0 0, 4),
   (B + 28, .movWide 2 64 15 20563 0, 4),
   (B + 32, .movWide 3 64 15 12626 1, 4),
   (B + 36, .addSubReg true true 64 31 14 15 .lsl 0, 4),
   (B + 40, .bcond 1 (relOff (B + 40) rej), 4),
   (B + 44, .ldr .w4 12 0 4, 4),
   (B + 48, .ldr .w4 1 0 8, 4),
   (B + 52, .ldr .w4 15 0 12, 4),
   (B + 56, .addSubImm false false 64 10 0 16, 4)] ++
  skipdataT (B + 60) rej ++
  [(B + 108, .logicReg .orr false 64 25 31 10 .lsl 0, 4),
   (B + 112, .logicReg .orr false 64 15 31 1 .lsl 0, 4)] ++
  skipwordsT (B + 116) rej ++
  [(B + 284, .addSubReg true true 64 31 10 24 .lsl 0, 4),
   (B + 288, .bcond 1 (relOff (B + 288) rej), 4),
   (B + 292, .addSubReg true true 64 31 12 1 .lsl 0, 4),
   (B + 296, .bcond 2 (relOff (B + 296) rej), 4),
   (B + 300, .logicReg .orr false 64 10 31 25 .lsl 0, 4),
   (B + 304, .logicReg .orr false 64 15 31 12 .lsl 0, 4)] ++
  skipwordsT (B + 308) rej ++
  [(B + 476, .ldr .w4 14 10 0, 4),
   (B + 480, .cbz true 64 14 (relOff (B + 480) rej), 4),
   (B + 484, .logicReg .orr false 64 12 31 25 .lsl 0, 4),
   (B + 488, .addSubReg true true 64 31 12 24 .lsl 0, 4),
   (B + 492, .bcond 0 (relOff (B + 492) acc), 4),
   (B + 496, .ldr .w4 11 12 12, 4),
   (B + 500, .cbz false 64 11 (relOff (B + 500) rej), 4),
   (B + 504, .addSubImm false false 64 9 12 20, 4)] ++
  skipcodeT (B + 508) 9 rej ++
  [(B + 612, .movWide 2 64 8 0 0, 4)] ++
  lookupT (B + 616) 8 ++
  [(B + 660, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 664, .bcond 0 (28), 4),
   (B + 668, .ldr .w4 14 10 4, 4),
   (B + 672, .ldr .w4 15 12 0, 4),
   (B + 676, .addSubReg true true 64 31 14 15 .lsl 0, 4),
   (B + 680, .bcond 1 (relOff (B + 680) rej), 4),
   (B + 684, .ldr .w4 14 10 8, 4),
   (B + 688, .cbz true 64 14 (relOff (B + 688) rej), 4),
   (B + 692, .ldr .w4 13 12 0, 4),
   (B + 696, .movWide 2 64 7 0 0, 4),
   (B + 700, .addSubImm false false 64 6 12 20, 4),
   (B + 704, .ldr .w4 15 12 12, 4),
   (B + 708, .addSubReg true true 64 31 8 15 .lsl 0, 4),
   (B + 712, .bcond 0 (1816), 4)] ++
  lookupT (B + 716) 8 ++
  [(B + 760, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 764, .bcond 0 (12), 4),
   (B + 768, .ldr .w4 13 10 4, 4),
   (B + 772, .ldr .w4 7 10 8, 4),
   (B + 776, .ldr .b1 15 6 0, 4),
   (B + 780, .addSubImm true true 64 31 15 57, 4),
   (B + 784, .bcond 3 (112), 4)] ++
  dispatchCode (B + 788)
    [(128, B + 1664), (129, B + 1688), (130, B + 1712), (131, B + 1748),
     (132, B + 1780), (133, B + 1812), (134, B + 1840), (135, B + 1880),
     (136, B + 1912), (137, B + 2000), (138, B + 2000), (139, B + 2096),
     (140, B + 2324)] ++
  [(B + 892, .b false (relOff (B + 892) rej), 4),
   (B + 896, .addSubImm true true 64 31 15 50, 4),
   (B + 900, .bcond 0 (752), 4)] ++
  effT (B + 904) (B + 1616) ++
  [(B + 1616, .addSubReg true true 64 31 13 10 .lsl 0, 4),
   (B + 1620, .bcond 3 (relOff (B + 1620) rej), 4),
   (B + 1624, .addSubReg true false 64 13 13 10 .lsl 0, 4),
   (B + 1628, .addSubReg false false 64 13 13 11 .lsl 0, 4)] ++
  chkT (B + 1632) 13 ovf ++
  [(B + 1644, .movWide 2 64 15 1 0, 4),
   (B + 1648, .b false (700), 4)] ++
  [(B + 1652, .cbz false 64 13 (relOff (B + 1652) rej), 4),
   (B + 1656, .movWide 2 64 15 1 0, 4),
   (B + 1660, .b false (788), 4)] ++
  [(B + 1664, .addSubImm false false 64 13 13 1, 4)] ++
  chkT (B + 1668) 13 ovf ++
  [(B + 1680, .movWide 2 64 15 9 0, 4),
   (B + 1684, .b false (664), 4)] ++
  [(B + 1688, .addSubImm false false 64 13 13 2, 4)] ++
  chkT (B + 1692) 13 ovf ++
  [(B + 1704, .movWide 2 64 15 5 0, 4),
   (B + 1708, .b false (640), 4)] ++
  [(B + 1712, .ldur .w4 10 6 1, 4),
   (B + 1716, .addSubReg true true 64 31 10 13 .lsl 0, 4),
   (B + 1720, .bcond 2 (relOff (B + 1720) rej), 4),
   (B + 1724, .addSubImm false false 64 13 13 1, 4)] ++
  chkT (B + 1728) 13 ovf ++
  [(B + 1740, .movWide 2 64 15 5 0, 4),
   (B + 1744, .b false (604), 4)] ++
  [(B + 1748, .cbz false 64 13 (relOff (B + 1748) rej), 4),
   (B + 1752, .addSubImm true false 64 13 13 1, 4),
   (B + 1756, .addSubImm false false 64 7 7 1, 4)] ++
  chkT (B + 1760) 7 ovf ++
  [(B + 1772, .movWide 2 64 15 1 0, 4),
   (B + 1776, .b false (572), 4)] ++
  [(B + 1780, .cbz false 64 7 (relOff (B + 1780) rej), 4),
   (B + 1784, .addSubImm true false 64 7 7 1, 4),
   (B + 1788, .addSubImm false false 64 13 13 1, 4)] ++
  chkT (B + 1792) 13 ovf ++
  [(B + 1804, .movWide 2 64 15 1 0, 4),
   (B + 1808, .b false (540), 4)] ++
  [(B + 1812, .cbz false 64 7 (relOff (B + 1812) rej), 4),
   (B + 1816, .addSubImm false false 64 13 13 1, 4)] ++
  chkT (B + 1820) 13 ovf ++
  [(B + 1832, .movWide 2 64 15 1 0, 4),
   (B + 1836, .b false (512), 4)] ++
  [(B + 1840, .ldur .w4 10 6 1, 4),
   (B + 1844, .ldr .w4 11 12 8, 4),
   (B + 1848, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 1852, .bcond 2 (relOff (B + 1852) rej), 4),
   (B + 1856, .addSubImm false false 64 13 13 1, 4)] ++
  chkT (B + 1860) 13 ovf ++
  [(B + 1872, .movWide 2 64 15 5 0, 4),
   (B + 1876, .b false (472), 4)] ++
  [(B + 1880, .cbz false 64 13 (relOff (B + 1880) rej), 4),
   (B + 1884, .ldur .w4 10 6 1, 4),
   (B + 1888, .ldr .w4 11 12 8, 4),
   (B + 1892, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 1896, .bcond 2 (relOff (B + 1896) rej), 4),
   (B + 1900, .addSubImm true false 64 13 13 1, 4),
   (B + 1904, .movWide 2 64 15 5 0, 4),
   (B + 1908, .b false (440), 4)] ++
  [(B + 1912, .ldur .w4 15 6 1, 4)] ++
  lookupT (B + 1916) 15 ++
  [(B + 1960, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 1964, .bcond 0 (relOff (B + 1964) rej), 4),
   (B + 1968, .ldr .w4 14 10 4, 4),
   (B + 1972, .addSubReg true true 64 31 14 13 .lsl 0, 4),
   (B + 1976, .bcond 1 (relOff (B + 1976) rej), 4),
   (B + 1980, .ldr .w4 14 10 8, 4),
   (B + 1984, .addSubReg true true 64 31 14 7 .lsl 0, 4),
   (B + 1988, .bcond 1 (relOff (B + 1988) rej), 4),
   (B + 1992, .movWide 2 64 15 5 0, 4),
   (B + 1996, .b false (452), 4)] ++
  [(B + 2000, .cbz false 64 13 (relOff (B + 2000) rej), 4),
   (B + 2004, .addSubImm true false 64 13 13 1, 4),
   (B + 2008, .ldur .w4 15 6 1, 4)] ++
  lookupT (B + 2012) 15 ++
  [(B + 2056, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 2060, .bcond 0 (relOff (B + 2060) rej), 4),
   (B + 2064, .ldr .w4 14 10 4, 4),
   (B + 2068, .addSubReg true true 64 31 14 13 .lsl 0, 4),
   (B + 2072, .bcond 1 (relOff (B + 2072) rej), 4),
   (B + 2076, .ldr .w4 14 10 8, 4),
   (B + 2080, .addSubReg true true 64 31 14 7 .lsl 0, 4),
   (B + 2084, .bcond 1 (relOff (B + 2084) rej), 4),
   (B + 2088, .movWide 2 64 15 5 0, 4),
   (B + 2092, .b false (256), 4)] ++
  [(B + 2096, .ldur .w4 15 6 1, 4),
   (B + 2100, .addSubReg true true 64 31 15 1 .lsl 0, 4),
   (B + 2104, .bcond 2 (relOff (B + 2104) rej), 4),
   (B + 2108, .logicReg .orr false 64 10 31 25 .lsl 0, 4)] ++
  skipwordsT (B + 2112) rej ++
  [(B + 2280, .ldr .w4 14 10 0, 4),
   (B + 2284, .addSubReg true true 64 31 13 14 .lsl 0, 4),
   (B + 2288, .bcond 3 (relOff (B + 2288) rej), 4),
   (B + 2292, .addSubReg true false 64 13 13 14 .lsl 0, 4),
   (B + 2296, .ldr .w4 14 10 4, 4),
   (B + 2300, .addSubReg false false 64 13 13 14 .lsl 0, 4)] ++
  chkT (B + 2304) 13 ovf ++
  [(B + 2316, .movWide 2 64 15 5 0, 4),
   (B + 2320, .b false (28), 4)] ++
  [(B + 2324, .ldr .w4 14 12 4, 4),
   (B + 2328, .addSubReg true true 64 31 13 14 .lsl 0, 4),
   (B + 2332, .bcond 1 (relOff (B + 2332) rej), 4),
   (B + 2336, .cbz true 64 7 (relOff (B + 2336) rej), 4),
   (B + 2340, .movWide 2 64 15 1 0, 4),
   (B + 2344, .b false (104), 4)] ++
  [(B + 2348, .addSubImm false false 64 8 8 1, 4),
   (B + 2352, .ldr .w4 14 12 12, 4),
   (B + 2356, .addSubReg true true 64 31 8 14 .lsl 0, 4),
   (B + 2360, .bcond 2 (relOff (B + 2360) rej), 4)] ++
  lookupT (B + 2364) 8 ++
  [(B + 2408, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 2412, .bcond 0 (28), 4),
   (B + 2416, .ldr .w4 14 10 4, 4),
   (B + 2420, .addSubReg true true 64 31 14 13 .lsl 0, 4),
   (B + 2424, .bcond 1 (relOff (B + 2424) rej), 4),
   (B + 2428, .ldr .w4 14 10 8, 4),
   (B + 2432, .addSubReg true true 64 31 14 7 .lsl 0, 4),
   (B + 2436, .bcond 1 (relOff (B + 2436) rej), 4),
   (B + 2440, .addSubReg false false 64 6 6 15 .lsl 0, 4),
   (B + 2444, .b false (18446744073709549876), 4)] ++
  [(B + 2448, .addSubImm false false 64 8 8 1, 4)] ++
  lookupT (B + 2452) 8 ++
  [(B + 2496, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 2500, .bcond 0 (16), 4),
   (B + 2504, .ldr .w4 13 10 4, 4),
   (B + 2508, .ldr .w4 7 10 8, 4),
   (B + 2512, .b false (18446744073709551544), 4),
   (B + 2516, .movWide 2 64 13 0 0, 4),
   (B + 2520, .movWide 2 64 7 0 0, 4),
   (B + 2524, .b false (18446744073709551532), 4)] ++
  [(B + 2528, .ldr .w4 14 12 16, 4),
   (B + 2532, .addSubReg false false 64 7 14 14 .lsl 1, 4),
   (B + 2536, .addSubReg false false 64 7 9 7 .lsl 2, 4),
   (B + 2540, .logicReg .orr false 64 6 31 9 .lsl 0, 4),
   (B + 2544, .addSubReg true true 64 31 6 7 .lsl 0, 4),
   (B + 2548, .bcond 0 (112), 4),
   (B + 2552, .ldr .w4 15 6 0, 4),
   (B + 2556, .ldr .w4 14 12 12, 4),
   (B + 2560, .addSubReg true true 64 31 15 14 .lsl 0, 4),
   (B + 2564, .bcond 2 (relOff (B + 2564) rej), 4)] ++
  lookupT (B + 2568) 15 ++
  [(B + 2612, .addSubReg true true 64 31 10 11 .lsl 0, 4),
   (B + 2616, .bcond 0 (relOff (B + 2616) rej), 4),
   (B + 2620, .ldr .w4 14 10 4, 4),
   (B + 2624, .ldr .w4 15 6 4, 4),
   (B + 2628, .addSubReg true true 64 31 14 15 .lsl 0, 4),
   (B + 2632, .bcond 1 (relOff (B + 2632) rej), 4),
   (B + 2636, .ldr .w4 14 10 8, 4),
   (B + 2640, .ldr .w4 15 6 8, 4),
   (B + 2644, .addSubReg true true 64 31 14 15 .lsl 0, 4),
   (B + 2648, .bcond 1 (relOff (B + 2648) rej), 4),
   (B + 2652, .addSubImm false false 64 6 6 12, 4),
   (B + 2656, .b false (18446744073709551504), 4),
   (B + 2660, .logicReg .orr false 64 12 31 7 .lsl 0, 4),
   (B + 2664, .b false (18446744073709549440), 4)] ++
  verdictInsns (B + 2668) R ACCEPT ++
  [(B + 2684, .b false (relOff (B + 2684) nxt), 4)] ++
  verdictInsns (B + 2688) R OVERFLOW ++
  [(B + 2704, .b false (relOff (B + 2704) nxt), 4)] ++
  verdictInsns (B + 2708) R REJECT ++
  [(B + 2724, .b false (relOff (B + 2724) nxt), 4)]

theorem p1_code_ok : subOK (passCode L_p1 2093056 19 L_p2) corePassTable = true := by decide +kernel
theorem p2_code_ok : subOK (passCode L_p2 2097152 20 L_p3) corePassTable = true := by decide +kernel
theorem p3_code_ok : subOK (passCode L_p3 2097152 21 L_vote) corePassTable = true := by decide +kernel

private theorem codeAt_of_corePass {c : List (Nat × Ins × Nat)} (h : subOK c corePassTable = true) :
    CodeAt c :=
  fun e he => table_ok e (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl (subOK_sound c _ h e he)))))

theorem p1_codeAt : CodeAt (passCode L_p1 2093056 19 L_p2) := codeAt_of_corePass p1_code_ok
theorem p2_codeAt : CodeAt (passCode L_p2 2097152 20 L_p3) := codeAt_of_corePass p2_code_ok
theorem p3_codeAt : CodeAt (passCode L_p3 2097152 21 L_vote) := codeAt_of_corePass p3_code_ok

structure PassBlocks (B adrImm R nxt : Nat) : Prop where
  cHdr : CodeAt
    [(B + 0, .adr true 15 adrImm, 4), (B + 4, .ldr .x8 0 15 0, 4),
     (B + 8, .ldr .x8 14 15 8, 4), (B + 12, .addSubReg false false 64 24 0 14 .lsl 0, 4),
     (B + 16, .addSubImm true true 64 31 14 16, 4), (B + 20, .bcond 3 (relOff (B + 20) (B + 2708)), 4),
     (B + 24, .ldr .w4 14 0 0, 4), (B + 28, .movWide 2 64 15 20563 0, 4),
     (B + 32, .movWide 3 64 15 12626 1, 4), (B + 36, .addSubReg true true 64 31 14 15 .lsl 0, 4),
     (B + 40, .bcond 1 (relOff (B + 40) (B + 2708)), 4), (B + 44, .ldr .w4 12 0 4, 4),
     (B + 48, .ldr .w4 1 0 8, 4), (B + 52, .ldr .w4 15 0 12, 4),
     (B + 56, .addSubImm false false 64 10 0 16, 4)]
  cSd : CodeAt (skipdataT (B + 60) (B + 2708))
  cMid1 : CodeAt
    [(B + 108, .logicReg .orr false 64 25 31 10 .lsl 0, 4),
     (B + 112, .logicReg .orr false 64 15 31 1 .lsl 0, 4)]
  cSw1 : CodeAt (skipwordsT (B + 116) (B + 2708))
  cMid2 : CodeAt
    [(B + 284, .addSubReg true true 64 31 10 24 .lsl 0, 4),
     (B + 288, .bcond 1 (relOff (B + 288) (B + 2708)), 4),
     (B + 292, .addSubReg true true 64 31 12 1 .lsl 0, 4),
     (B + 296, .bcond 2 (relOff (B + 296) (B + 2708)), 4),
     (B + 300, .logicReg .orr false 64 10 31 25 .lsl 0, 4),
     (B + 304, .logicReg .orr false 64 15 31 12 .lsl 0, 4)]
  cSw2 : CodeAt (skipwordsT (B + 308) (B + 2708))
  cMid3 : CodeAt
    [(B + 476, .ldr .w4 14 10 0, 4),
     (B + 480, .cbz true 64 14 (relOff (B + 480) (B + 2708)), 4),
     (B + 484, .logicReg .orr false 64 12 31 25 .lsl 0, 4),
     (B + 488, .addSubReg true true 64 31 12 24 .lsl 0, 4),
     (B + 492, .bcond 0 (relOff (B + 492) (B + 2668)), 4),
     (B + 496, .ldr .w4 11 12 12, 4),
     (B + 500, .cbz false 64 11 (relOff (B + 500) (B + 2708)), 4),
     (B + 504, .addSubImm false false 64 9 12 20, 4)]
  cSc : CodeAt (skipcodeT (B + 508) 9 (B + 2708))
  cMovIdx0 : CodeAt [(B + 612, .movWide 2 64 8 0 0, 4)]
  cLu0 : CodeAt (lookupT (B + 616) 8)
  cWalk0 : CodeAt
    [(B + 660, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 664, .bcond 0 (28), 4),
     (B + 668, .ldr .w4 14 10 4, 4),
     (B + 672, .ldr .w4 15 12 0, 4),
     (B + 676, .addSubReg true true 64 31 14 15 .lsl 0, 4),
     (B + 680, .bcond 1 (relOff (B + 680) (B + 2708)), 4),
     (B + 684, .ldr .w4 14 10 8, 4),
     (B + 688, .cbz true 64 14 (relOff (B + 688) (B + 2708)), 4),
     (B + 692, .ldr .w4 13 12 0, 4),
     (B + 696, .movWide 2 64 7 0 0, 4),
     (B + 700, .addSubImm false false 64 6 12 20, 4),
     (B + 704, .ldr .w4 15 12 12, 4),
     (B + 708, .addSubReg true true 64 31 8 15 .lsl 0, 4),
     (B + 712, .bcond 0 (1816), 4)]
  cLuI : CodeAt (lookupT (B + 716) 8)
  cDispPre : CodeAt
    [(B + 760, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 764, .bcond 0 (12), 4),
     (B + 768, .ldr .w4 13 10 4, 4),
     (B + 772, .ldr .w4 7 10 8, 4),
     (B + 776, .ldr .b1 15 6 0, 4),
     (B + 780, .addSubImm true true 64 31 15 57, 4),
     (B + 784, .bcond 3 (112), 4)]
  cDisp : CodeAt (dispatchCode (B + 788)
    [(128, B + 1664), (129, B + 1688), (130, B + 1712), (131, B + 1748),
     (132, B + 1780), (133, B + 1812), (134, B + 1840), (135, B + 1880),
     (136, B + 1912), (137, B + 2000), (138, B + 2000), (139, B + 2096),
     (140, B + 2324)])
  cDispEnd : CodeAt
    [(B + 892, .b false (relOff (B + 892) (B + 2708)), 4),
     (B + 896, .addSubImm true true 64 31 15 50, 4),
     (B + 900, .bcond 0 (752), 4)]
  cEff : CodeAt (effT (B + 904) (B + 1616))
  cHave : CodeAt
    [(B + 1616, .addSubReg true true 64 31 13 10 .lsl 0, 4),
     (B + 1620, .bcond 3 (relOff (B + 1620) (B + 2708)), 4),
     (B + 1624, .addSubReg true false 64 13 13 10 .lsl 0, 4),
     (B + 1628, .addSubReg false false 64 13 13 11 .lsl 0, 4)]
  cChkEff : CodeAt (chkT (B + 1632) 13 (B + 2688))
  cHaveEnd : CodeAt [(B + 1644, .movWide 2 64 15 1 0, 4), (B + 1648, .b false (700), 4)]
  cExit : CodeAt
    [(B + 1652, .cbz false 64 13 (relOff (B + 1652) (B + 2708)), 4),
     (B + 1656, .movWide 2 64 15 1 0, 4),
     (B + 1660, .b false (788), 4)]
  cLitPre : CodeAt [(B + 1664, .addSubImm false false 64 13 13 1, 4)]
  cChkLit : CodeAt (chkT (B + 1668) 13 (B + 2688))
  cLitEnd : CodeAt [(B + 1680, .movWide 2 64 15 9 0, 4), (B + 1684, .b false (664), 4)]
  cStrPre : CodeAt [(B + 1688, .addSubImm false false 64 13 13 2, 4)]
  cChkStr : CodeAt (chkT (B + 1692) 13 (B + 2688))
  cStrEnd : CodeAt [(B + 1704, .movWide 2 64 15 5 0, 4), (B + 1708, .b false (640), 4)]
  cPickPre : CodeAt
    [(B + 1712, .ldur .w4 10 6 1, 4),
     (B + 1716, .addSubReg true true 64 31 10 13 .lsl 0, 4),
     (B + 1720, .bcond 2 (relOff (B + 1720) (B + 2708)), 4),
     (B + 1724, .addSubImm false false 64 13 13 1, 4)]
  cChkPick : CodeAt (chkT (B + 1728) 13 (B + 2688))
  cPickEnd : CodeAt [(B + 1740, .movWide 2 64 15 5 0, 4), (B + 1744, .b false (604), 4)]
  cTorPre : CodeAt
    [(B + 1748, .cbz false 64 13 (relOff (B + 1748) (B + 2708)), 4),
     (B + 1752, .addSubImm true false 64 13 13 1, 4),
     (B + 1756, .addSubImm false false 64 7 7 1, 4)]
  cChkTor : CodeAt (chkT (B + 1760) 7 (B + 2688))
  cTorEnd : CodeAt [(B + 1772, .movWide 2 64 15 1 0, 4), (B + 1776, .b false (572), 4)]
  cFromrPre : CodeAt
    [(B + 1780, .cbz false 64 7 (relOff (B + 1780) (B + 2708)), 4),
     (B + 1784, .addSubImm true false 64 7 7 1, 4),
     (B + 1788, .addSubImm false false 64 13 13 1, 4)]
  cChkFromr : CodeAt (chkT (B + 1792) 13 (B + 2688))
  cFromrEnd : CodeAt [(B + 1804, .movWide 2 64 15 1 0, 4), (B + 1808, .b false (540), 4)]
  cRfetchPre : CodeAt
    [(B + 1812, .cbz false 64 7 (relOff (B + 1812) (B + 2708)), 4),
     (B + 1816, .addSubImm false false 64 13 13 1, 4)]
  cChkRfetch : CodeAt (chkT (B + 1820) 13 (B + 2688))
  cRfetchEnd : CodeAt [(B + 1832, .movWide 2 64 15 1 0, 4), (B + 1836, .b false (512), 4)]
  cLocgetPre : CodeAt
    [(B + 1840, .ldur .w4 10 6 1, 4),
     (B + 1844, .ldr .w4 11 12 8, 4),
     (B + 1848, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 1852, .bcond 2 (relOff (B + 1852) (B + 2708)), 4),
     (B + 1856, .addSubImm false false 64 13 13 1, 4)]
  cChkLocget : CodeAt (chkT (B + 1860) 13 (B + 2688))
  cLocgetEnd : CodeAt [(B + 1872, .movWide 2 64 15 5 0, 4), (B + 1876, .b false (472), 4)]
  cLocset : CodeAt
    [(B + 1880, .cbz false 64 13 (relOff (B + 1880) (B + 2708)), 4),
     (B + 1884, .ldur .w4 10 6 1, 4),
     (B + 1888, .ldr .w4 11 12 8, 4),
     (B + 1892, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 1896, .bcond 2 (relOff (B + 1896) (B + 2708)), 4),
     (B + 1900, .addSubImm true false 64 13 13 1, 4),
     (B + 1904, .movWide 2 64 15 5 0, 4),
     (B + 1908, .b false (440), 4)]
  cJmpPre : CodeAt [(B + 1912, .ldur .w4 15 6 1, 4)]
  cLuJmp : CodeAt (lookupT (B + 1916) 15)
  cJmpEnd : CodeAt
    [(B + 1960, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 1964, .bcond 0 (relOff (B + 1964) (B + 2708)), 4),
     (B + 1968, .ldr .w4 14 10 4, 4),
     (B + 1972, .addSubReg true true 64 31 14 13 .lsl 0, 4),
     (B + 1976, .bcond 1 (relOff (B + 1976) (B + 2708)), 4),
     (B + 1980, .ldr .w4 14 10 8, 4),
     (B + 1984, .addSubReg true true 64 31 14 7 .lsl 0, 4),
     (B + 1988, .bcond 1 (relOff (B + 1988) (B + 2708)), 4),
     (B + 1992, .movWide 2 64 15 5 0, 4),
     (B + 1996, .b false (452), 4)]
  cJzPre : CodeAt
    [(B + 2000, .cbz false 64 13 (relOff (B + 2000) (B + 2708)), 4),
     (B + 2004, .addSubImm true false 64 13 13 1, 4),
     (B + 2008, .ldur .w4 15 6 1, 4)]
  cLuJz : CodeAt (lookupT (B + 2012) 15)
  cJzEnd : CodeAt
    [(B + 2056, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 2060, .bcond 0 (relOff (B + 2060) (B + 2708)), 4),
     (B + 2064, .ldr .w4 14 10 4, 4),
     (B + 2068, .addSubReg true true 64 31 14 13 .lsl 0, 4),
     (B + 2072, .bcond 1 (relOff (B + 2072) (B + 2708)), 4),
     (B + 2076, .ldr .w4 14 10 8, 4),
     (B + 2080, .addSubReg true true 64 31 14 7 .lsl 0, 4),
     (B + 2084, .bcond 1 (relOff (B + 2084) (B + 2708)), 4),
     (B + 2088, .movWide 2 64 15 5 0, 4),
     (B + 2092, .b false (256), 4)]
  cCallPre : CodeAt
    [(B + 2096, .ldur .w4 15 6 1, 4),
     (B + 2100, .addSubReg true true 64 31 15 1 .lsl 0, 4),
     (B + 2104, .bcond 2 (relOff (B + 2104) (B + 2708)), 4),
     (B + 2108, .logicReg .orr false 64 10 31 25 .lsl 0, 4)]
  cSwCall : CodeAt (skipwordsT (B + 2112) (B + 2708))
  cCallMid : CodeAt
    [(B + 2280, .ldr .w4 14 10 0, 4),
     (B + 2284, .addSubReg true true 64 31 13 14 .lsl 0, 4),
     (B + 2288, .bcond 3 (relOff (B + 2288) (B + 2708)), 4),
     (B + 2292, .addSubReg true false 64 13 13 14 .lsl 0, 4),
     (B + 2296, .ldr .w4 14 10 4, 4),
     (B + 2300, .addSubReg false false 64 13 13 14 .lsl 0, 4)]
  cChkCall : CodeAt (chkT (B + 2304) 13 (B + 2688))
  cCallEnd : CodeAt [(B + 2316, .movWide 2 64 15 5 0, 4), (B + 2320, .b false (28), 4)]
  cRet : CodeAt
    [(B + 2324, .ldr .w4 14 12 4, 4),
     (B + 2328, .addSubReg true true 64 31 13 14 .lsl 0, 4),
     (B + 2332, .bcond 1 (relOff (B + 2332) (B + 2708)), 4),
     (B + 2336, .cbz true 64 7 (relOff (B + 2336) (B + 2708)), 4),
     (B + 2340, .movWide 2 64 15 1 0, 4),
     (B + 2344, .b false (104), 4)]
  cFallPre : CodeAt
    [(B + 2348, .addSubImm false false 64 8 8 1, 4),
     (B + 2352, .ldr .w4 14 12 12, 4),
     (B + 2356, .addSubReg true true 64 31 8 14 .lsl 0, 4),
     (B + 2360, .bcond 2 (relOff (B + 2360) (B + 2708)), 4)]
  cLuFall : CodeAt (lookupT (B + 2364) 8)
  cFallEnd : CodeAt
    [(B + 2408, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 2412, .bcond 0 (28), 4),
     (B + 2416, .ldr .w4 14 10 4, 4),
     (B + 2420, .addSubReg true true 64 31 14 13 .lsl 0, 4),
     (B + 2424, .bcond 1 (relOff (B + 2424) (B + 2708)), 4),
     (B + 2428, .ldr .w4 14 10 8, 4),
     (B + 2432, .addSubReg true true 64 31 14 7 .lsl 0, 4),
     (B + 2436, .bcond 1 (relOff (B + 2436) (B + 2708)), 4),
     (B + 2440, .addSubReg false false 64 6 6 15 .lsl 0, 4),
     (B + 2444, .b false (18446744073709549876), 4)]
  cNofallPre : CodeAt [(B + 2448, .addSubImm false false 64 8 8 1, 4)]
  cLuNofall : CodeAt (lookupT (B + 2452) 8)
  cNofallEnd : CodeAt
    [(B + 2496, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 2500, .bcond 0 (16), 4),
     (B + 2504, .ldr .w4 13 10 4, 4),
     (B + 2508, .ldr .w4 7 10 8, 4),
     (B + 2512, .b false (18446744073709551544), 4),
     (B + 2516, .movWide 2 64 13 0 0, 4),
     (B + 2520, .movWide 2 64 7 0 0, 4),
     (B + 2524, .b false (18446744073709551532), 4)]
  cWdonePre : CodeAt
    [(B + 2528, .ldr .w4 14 12 16, 4),
     (B + 2532, .addSubReg false false 64 7 14 14 .lsl 1, 4),
     (B + 2536, .addSubReg false false 64 7 9 7 .lsl 2, 4),
     (B + 2540, .logicReg .orr false 64 6 31 9 .lsl 0, 4),
     (B + 2544, .addSubReg true true 64 31 6 7 .lsl 0, 4),
     (B + 2548, .bcond 0 (112), 4),
     (B + 2552, .ldr .w4 15 6 0, 4),
     (B + 2556, .ldr .w4 14 12 12, 4),
     (B + 2560, .addSubReg true true 64 31 15 14 .lsl 0, 4),
     (B + 2564, .bcond 2 (relOff (B + 2564) (B + 2708)), 4)]
  cLuAnns : CodeAt (lookupT (B + 2568) 15)
  cAnnsEnd : CodeAt
    [(B + 2612, .addSubReg true true 64 31 10 11 .lsl 0, 4),
     (B + 2616, .bcond 0 (relOff (B + 2616) (B + 2708)), 4),
     (B + 2620, .ldr .w4 14 10 4, 4),
     (B + 2624, .ldr .w4 15 6 4, 4),
     (B + 2628, .addSubReg true true 64 31 14 15 .lsl 0, 4),
     (B + 2632, .bcond 1 (relOff (B + 2632) (B + 2708)), 4),
     (B + 2636, .ldr .w4 14 10 8, 4),
     (B + 2640, .ldr .w4 15 6 8, 4),
     (B + 2644, .addSubReg true true 64 31 14 15 .lsl 0, 4),
     (B + 2648, .bcond 1 (relOff (B + 2648) (B + 2708)), 4),
     (B + 2652, .addSubImm false false 64 6 6 12, 4),
     (B + 2656, .b false (18446744073709551504), 4),
     (B + 2660, .logicReg .orr false 64 12 31 7 .lsl 0, 4),
     (B + 2664, .b false (18446744073709549440), 4)]
  cAcc : CodeAt (verdictInsns (B + 2668) R ACCEPT)
  cAccB : CodeAt [(B + 2684, .b false (relOff (B + 2684) nxt), 4)]
  cOvf : CodeAt (verdictInsns (B + 2688) R OVERFLOW)
  cOvfB : CodeAt [(B + 2704, .b false (relOff (B + 2704) nxt), 4)]
  cRej : CodeAt (verdictInsns (B + 2708) R REJECT)
  cRejB : CodeAt [(B + 2724, .b false (relOff (B + 2724) nxt), 4)]

theorem passBlocks_of_codeAt {B adrImm R nxt : Nat} (hc : CodeAt (passCode B adrImm R nxt)) :
    PassBlocks B adrImm R nxt := by
  unfold passCode at hc
  dsimp only at hc
  obtain ⟨hc, x69⟩ := CodeAt_append.1 hc; obtain ⟨hc, x68⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x67⟩ := CodeAt_append.1 hc; obtain ⟨hc, x66⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x65⟩ := CodeAt_append.1 hc; obtain ⟨hc, x64⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x63⟩ := CodeAt_append.1 hc; obtain ⟨hc, x62⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x61⟩ := CodeAt_append.1 hc; obtain ⟨hc, x60⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x59⟩ := CodeAt_append.1 hc; obtain ⟨hc, x58⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x57⟩ := CodeAt_append.1 hc; obtain ⟨hc, x56⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x55⟩ := CodeAt_append.1 hc; obtain ⟨hc, x54⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x53⟩ := CodeAt_append.1 hc; obtain ⟨hc, x52⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x51⟩ := CodeAt_append.1 hc; obtain ⟨hc, x50⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x49⟩ := CodeAt_append.1 hc; obtain ⟨hc, x48⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x47⟩ := CodeAt_append.1 hc; obtain ⟨hc, x46⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x45⟩ := CodeAt_append.1 hc; obtain ⟨hc, x44⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x43⟩ := CodeAt_append.1 hc; obtain ⟨hc, x42⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x41⟩ := CodeAt_append.1 hc; obtain ⟨hc, x40⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x39⟩ := CodeAt_append.1 hc; obtain ⟨hc, x38⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x37⟩ := CodeAt_append.1 hc; obtain ⟨hc, x36⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x35⟩ := CodeAt_append.1 hc; obtain ⟨hc, x34⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x33⟩ := CodeAt_append.1 hc; obtain ⟨hc, x32⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x31⟩ := CodeAt_append.1 hc; obtain ⟨hc, x30⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x29⟩ := CodeAt_append.1 hc; obtain ⟨hc, x28⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x27⟩ := CodeAt_append.1 hc; obtain ⟨hc, x26⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x25⟩ := CodeAt_append.1 hc; obtain ⟨hc, x24⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x23⟩ := CodeAt_append.1 hc; obtain ⟨hc, x22⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x21⟩ := CodeAt_append.1 hc; obtain ⟨hc, x20⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x19⟩ := CodeAt_append.1 hc; obtain ⟨hc, x18⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x17⟩ := CodeAt_append.1 hc; obtain ⟨hc, x16⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x15⟩ := CodeAt_append.1 hc; obtain ⟨hc, x14⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x13⟩ := CodeAt_append.1 hc; obtain ⟨hc, x12⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x11⟩ := CodeAt_append.1 hc; obtain ⟨hc, x10⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x9⟩ := CodeAt_append.1 hc; obtain ⟨hc, x8⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x7⟩ := CodeAt_append.1 hc; obtain ⟨hc, x6⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x5⟩ := CodeAt_append.1 hc; obtain ⟨hc, x4⟩ := CodeAt_append.1 hc
  obtain ⟨hc, x3⟩ := CodeAt_append.1 hc; obtain ⟨x1, x2⟩ := CodeAt_append.1 hc
  exact ⟨x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19,
    x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38,
    x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57,
    x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69⟩

/-! ### Reusable sub-block specifications (`verdict`, `chk`, `skip*`, `lookup`, `dispatch`) -/

theorem verdict_spec {B nxt R v : Nat} {s : St}
    (hc : CodeAt (verdictInsns B R v))
    (hbr : im (B + 16) = some (.b false (relOff (B + 16) nxt), 4))
    (hB : B + 20 < 2 ^ 64) (hnxt : nxt < 2 ^ 64) (_hle : B + 16 ≤ nxt)
    (hv : v < 2 ^ 64) (hR : R < 31) (hl : s.regs.length = 31) (hpc : s.pc = B) :
    Reaches s fun s' => Frame [R] s s' ∧ s'.pc = nxt ∧ s'.reg R = some v ∧ s'.regs.length = 31 := by
  simp only [verdictInsns, CodeAt_cons] at hc
  obtain ⟨f0, f1, f2, f3, _⟩ := hc
  have hneR : R ≠ 31 := by omega
  have hr0 : (B + 4) % AArch64Env.W = B + 4 := modW_small (by omega)
  have hr1 : (B + 4 + 4) % AArch64Env.W = B + 8 := modW_small (by omega)
  have hr2 : (B + 8 + 4) % AArch64Env.W = B + 12 := modW_small (by omega)
  have hr3 : (B + 12 + 4) % AArch64Env.W = B + 16 := modW_small (by omega)
  have hr4 : (B + 16 + relOff (B + 16) nxt) % AArch64Env.W = nxt := relOff_any (by omega) hnxt
  have hw0 : v % 65536 * 1 % 2 ^ 64 = v % 2 ^ 16 := by omega
  have hw1 : (v % 2 ^ 16 % 2 ^ 64 % 2 ^ 16 + v / 65536 % 65536 * 2 ^ 16 +
      v % 2 ^ 16 % 2 ^ 64 / 2 ^ 32 * 2 ^ 32) % 2 ^ 64 = v % 2 ^ 32 := by omega
  have hw2 : (v % 2 ^ 32 % 2 ^ 64 % 2 ^ 32 + v / 2 ^ 32 % 65536 * 2 ^ 32 +
      v % 2 ^ 32 % 2 ^ 64 / 2 ^ 48 * 2 ^ 48) % 2 ^ 64 = v % 2 ^ 48 := by omega
  have hw3 : (v % 2 ^ 48 % 2 ^ 64 % 2 ^ 48 + v / 2 ^ 48 % 65536 * 2 ^ 48 +
      v % 2 ^ 48 % 2 ^ 64 / 2 ^ 64 * 2 ^ 64) % 2 ^ 64 = v := by omega
  xs f0 using hl, hneR, hw0, hr0
  xs f1 using hl, hR, hneR, hw1, hr1
  xs f2 using hl, hR, hneR, hw2, hr2
  xs f3 using hl, hR, hneR, hw3, hr3
  xs hbr using hr4
  exact Reaches.refl ⟨by frame_tac, rfl, (by simp [hl, hR]), by simp [hl]⟩

theorem chk_spec {L r ovf x : Nat} {s : St}
    (hc : CodeAt (chkT L r ovf)) (hL : L + 12 ≤ ovf) (hovf : ovf < 2 ^ 64)
    (hr : r < 31) (hrne10 : r ≠ 10) (hx : x < 2 ^ 64)
    (hl : s.regs.length = 31) (hpc : s.pc = L) (hsr : s.reg r = some x) :
    Reaches s fun s' => Frame [10] s s' ∧
      (if HMAX ≤ x then s'.pc = ovf else s'.pc = L + 12) ∧
      s'.regs.length = 31 ∧ s'.reg r = some x := by
  simp only [chkT, CodeAt_cons] at hc
  obtain ⟨f0, f1, f2, _⟩ := hc
  have h10ne : (10 : Nat) ≠ r := hrne10.symm
  have hne31 : r ≠ 31 := by omega
  have hr0 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
  have hr1 : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
  have hr2f : (L + 8 + 4) % AArch64Env.W = L + 12 := modW_small (by omega)
  have hr2t : (L + 8 + relOff (L + 8) ovf) % AArch64Env.W = ovf := relOff_any (by omega) hovf
  have hcs := cond_sub (show x < pow2 64 by rw [pow2_64]; exact hx)
    (show HMAX < pow2 64 by rw [pow2_64]; decide)
  xs f0 using hl, hr0
  xs f1 using hl, hne31, hsr, h10ne, hr1
  by_cases hov : HMAX ≤ x
  · have hfl : AArch64Phys.cond (addSubOp true 64 (x % 2 ^ 64) (1 * 2 ^ 48 % 2 ^ 64 % 2 ^ 64)).2 2 =
        some true := by
      rw [Nat.mod_eq_of_lt hx, show (1 * 2 ^ 48 : Nat) = HMAX by decide,
        Nat.mod_eq_of_lt (by decide : HMAX < 2 ^ 64),
        Nat.mod_eq_of_lt (by decide : HMAX < 2 ^ 64), hcs.2.2.1]
      simp [hov]
    xs f2 using hfl, hr2t
    exact Reaches.refl ⟨by frame_tac, by simp [hov], by simp [hl], by simp [h10ne, hsr]⟩
  · have hfl : AArch64Phys.cond (addSubOp true 64 (x % 2 ^ 64) (1 * 2 ^ 48 % 2 ^ 64 % 2 ^ 64)).2 2 =
        some false := by
      rw [Nat.mod_eq_of_lt hx, show (1 * 2 ^ 48 : Nat) = HMAX by decide,
        Nat.mod_eq_of_lt (by decide : HMAX < 2 ^ 64),
        Nat.mod_eq_of_lt (by decide : HMAX < 2 ^ 64), hcs.2.2.1]
      simp [hov]
    xs f2 using hfl, hr2f
    exact Reaches.refl ⟨by frame_tac, by simp [hov], by simp [hl], by simp [h10ne, hsr]⟩

theorem skipdata_spec {L rej b : Nat} {bs : List Nat}
    (hc : CodeAt (skipdataT L rej)) (hL : L + 48 ≤ rej) (hrej : rej < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX) :
    ∀ (cnt pos : Nat) (s : St),
      s.mem = m → s.regs.length = 31 → s.pc = L →
      s.reg 10 = some (b + pos) → s.reg 15 = some cnt → s.reg 24 = some (b + bs.length) →
      pos ≤ bs.length → cnt < 2 ^ 64 →
      Reaches s fun s' => Frame [10, 11, 14, 15] s s' ∧
        match skipData bs cnt pos with
        | none => s'.pc = rej
        | some pos' => s'.pc = L + 48 ∧ s'.reg 10 = some (b + pos') ∧ pos' ≤ bs.length
  | 0, pos, s, hm, hl, hpc, h10, h15, h24, hpos, _ => by
    simp only [skipdataT, CodeAt_cons] at hc
    obtain ⟨f0, _⟩ := hc
    have hr0 : (L + 48) % AArch64Env.W = L + 48 := modW_small (by omega)
    xs f0 using h15, hr0
    exact Reaches.refl ⟨by frame_tac, rfl, (by simp [h10]), hpos⟩
  | k + 1, pos, s, hm, hl, hpc, h10, h15, h24, hpos, hcnt => by
    have hc' := hc
    simp only [skipdataT, CodeAt_cons] at hc'
    obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, _⟩ := hc'
    have hk1 : (k + 1) % 2 ^ 64 ≠ 0 := by omega
    have h48 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
    have hr1 : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
    have hr2 : (L + 8 + 4) % AArch64Env.W = L + 12 := modW_small (by omega)
    have hr3t : (L + 12 + relOff (L + 12) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
    have hr3f : (L + 12 + 4) % AArch64Env.W = L + 16 := modW_small (by omega)
    have hbp4 : ((b + pos) % 2 ^ 64 + 4 % 2 ^ 64) % 2 ^ 64 = b + pos + 4 := by
      unfold HMAX at hsm; omega
    xs f0 using h15, hk1, h48
    xs f1 using hl, h10, hbp4, hr1
    have hcs1 := cond_sub (show b + pos + 4 < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    xs f2 using hl, h24, hr2
    by_cases hp4 : pos + 4 ≤ bs.length
    · have hfl3 : AArch64Phys.cond (addSubOp true 64 ((b + pos + 4) % 2 ^ 64)
          ((b + bs.length) % 2 ^ 64)).2 8 = some false := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
          Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.2.2.1]
        simp [show ¬ b + bs.length < b + pos + 4 by omega]
      xs f3 using hfl3, hr3f
      have hru := rdU32_eq_some_u32 hby hp4
      have hu := u32_lt hby pos
      have hrd4 : rdMem s.mem (((b + pos) % 2 ^ 64 + 0) % AArch64Env.W) .w4 = some (u32 bs pos) := by
        rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega), Nat.add_zero, hm]
        exact rdMem_w4_buf hbuf hp4
      have hr4 : (L + 16 + 4) % AArch64Env.W = L + 20 := modW_small (by omega)
      have hr5 : (L + 20 + 4) % AArch64Env.W = L + 24 := modW_small (by omega)
      have hr6 : (L + 24 + 4) % AArch64Env.W = L + 28 := modW_small (by omega)
      have hr7 : (L + 28 + 4) % AArch64Env.W = L + 32 := modW_small (by omega)
      have hbp4m : (b + pos + 4) % 2 ^ 64 = b + pos + 4 := Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega)
      have hrem : ((b + bs.length) % 2 ^ 64 + 2 ^ 64 - (b + pos + 4) % 2 ^ 64) % 2 ^ 64 =
          bs.length - (pos + 4) := by unfold HMAX at hsm; omega
      xs f4 using hl, h10, hrd4, hu, hr4
      xs f5 using hl, hbp4m, hr5
      xs f6 using hl, h24, hrem, hr6
      have hcs2 := cond_sub (show bs.length - (pos + 4) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        (show u32 bs pos < pow2 64 by rw [pow2_64]; omega)
      xs f7 using hl, hr7
      by_cases hlen : pos + 4 + u32 bs pos ≤ bs.length
      · have hfl8 : AArch64Phys.cond (addSubOp true 64
            ((bs.length - (pos + 4)) % 2 ^ 64) (u32 bs pos % 2 ^ 64)).2 3 =
            some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by omega : u32 bs pos < 2 ^ 64), hcs2.2.2.2.1]
          simp [show ¬ bs.length - (pos + 4) < u32 bs pos by omega]
        have hr8f : (L + 32 + 4) % AArch64Env.W = L + 36 := modW_small (by omega)
        have hr9 : (L + 36 + 4) % AArch64Env.W = L + 40 := modW_small (by omega)
        have hr10 : (L + 40 + 4) % AArch64Env.W = L + 44 := modW_small (by omega)
        have hr11 : (L + 44 + 18446744073709551572) % AArch64Env.W = L := by rw [W_eq]; omega
        have h10' : ((b + pos + 4) % 2 ^ 64 + u32 bs pos % 2 ^ 64) % 2 ^ 64 = b + (pos + 4 + u32 bs pos) := by
          unfold HMAX at hsm; omega
        have h15' : ((k + 1) % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = k := by omega
        xs f8 using hfl8, hr8f
        xs f9 using hl, h10', hr9
        xs f10 using hl, h15, h15', hr10
        xs f11 using hr11
        refine (skipdata_spec hc hL hrej hbuf hby hsm k (pos + 4 + u32 bs pos) _
          (by simp [hm]) (by simp [hl]) rfl
          (by simp [hl]) (by simp [hl]) (by simp [h24]) hlen (by omega)).mono fun s' ⟨hf, hs'⟩ => ?_
        refine ⟨Frame.trans_same (by frame_tac) hf, ?_⟩
        simpa [skipData, hru, hlen] using hs'
      · have hfl8 : AArch64Phys.cond (addSubOp true 64
            ((bs.length - (pos + 4)) % 2 ^ 64) (u32 bs pos % 2 ^ 64)).2 3 =
            some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by omega : u32 bs pos < 2 ^ 64), hcs2.2.2.2.1]
          simp [show bs.length - (pos + 4) < u32 bs pos by omega]
        have hr8t : (L + 32 + relOff (L + 32) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
        xs f8 using hfl8, hr8t
        simp only [skipData, hru, show ¬ pos + 4 + u32 bs pos ≤ bs.length from hlen, ↓reduceIte,
          bind, Option.bind]
        exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl3 : AArch64Phys.cond (addSubOp true 64 ((b + pos + 4) % 2 ^ 64)
          ((b + bs.length) % 2 ^ 64)).2 8 = some true := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
          Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.2.2.1]
        simp [show b + bs.length < b + pos + 4 by omega]
      xs f3 using hfl3, hr3t
      simp only [skipData, rdU32_eq_none_of_lt (bs := bs) (pos := pos) (by omega), bind, Option.bind]
      exact Reaches.refl ⟨by frame_tac, rfl⟩

private theorem modW_off {L off c res bound : Nat} (hL : L + bound < 2 ^ 64)
    (heq : off + c = res) (hle : res ≤ bound) :
    (L + off + c) % AArch64Env.W = L + res := by
  rw [Nat.add_assoc, heq]; exact modW_small (by omega)

private theorem modW_step0 {L bound : Nat} (hL : L + bound < 2 ^ 64)
    (c : Nat) (hle : c ≤ bound := by decide) :
    (L + c) % AArch64Env.W = L + c :=
  modW_small (by omega)

private theorem modW_step {L bound : Nat} (hL : L + bound < 2 ^ 64)
    (off c res : Nat) (heq : off + c = res := by decide) (hle : res ≤ bound := by decide) :
    (L + off + c) % AArch64Env.W = L + res := by
  rw [Nat.add_assoc, heq]; exact modW_small (by omega)

private theorem modW_back {L bound : Nat} (hL : L + bound < 2 ^ 64)
    (off d : Nat) (hd : off + d = 2 ^ 64 := by decide) :
    (L + off + d) % AArch64Env.W = L := by
  rw [W_eq]; omega

private theorem relOff_step {B bound : Nat} (hB : B + bound < 2 ^ 64)
    (off tgt : Nat) (h1 : off ≤ tgt := by decide) (h2 : tgt ≤ bound := by decide) :
    (B + off + relOff (B + off) (B + tgt)) % AArch64Env.W = B + tgt :=
  relOff_target (by omega) (by omega)

private theorem instrLen_none_mid {op : Nat} (h1 : ¬ op < 57) (h2 : op < 128) : instrLen op = none := by
  simp [instrLen, h1, show op ≠ 0x80 by omega, show op ≠ 0x81 by omega, show op ≠ 0x82 by omega,
    show op ≠ 0x83 by omega, show op ≠ 0x84 by omega, show op ≠ 0x85 by omega, show op ≠ 0x86 by omega,
    show op ≠ 0x87 by omega, show op ≠ 0x88 by omega, show op ≠ 0x89 by omega, show op ≠ 0x8a by omega,
    show op ≠ 0x8b by omega, show op ≠ 0x8c by omega]

private theorem instrLen_none_gt {op : Nat} (h : 140 < op) : instrLen op = none := by
  simp [instrLen, show ¬ op < 57 by omega, show op ≠ 0x80 by omega, show op ≠ 0x81 by omega,
    show op ≠ 0x82 by omega, show op ≠ 0x83 by omega, show op ≠ 0x84 by omega, show op ≠ 0x85 by omega,
    show op ≠ 0x86 by omega, show op ≠ 0x87 by omega, show op ≠ 0x88 by omega, show op ≠ 0x89 by omega,
    show op ≠ 0x8a by omega, show op ≠ 0x8b by omega, show op ≠ 0x8c by omega]

private theorem instrLen_129_130 {op : Nat} (h1 : ¬ op < 128) (h2 : op ≠ 128) (h3 : op < 131) :
    instrLen op = some 5 := by
  rcases (show op = 129 ∨ op = 130 by omega) with rfl | rfl <;> decide

private theorem instrLen_131_133 {op : Nat} (h1 : ¬ op < 131) (h2 : op < 134) :
    instrLen op = some 1 := by
  rcases (show op = 131 ∨ op = 132 ∨ op = 133 by omega) with rfl | rfl | rfl <;> decide

private theorem instrLen_134_139 {op : Nat} (h1 : ¬ op < 134) (h2 : ¬ 140 < op) (h3 : op ≠ 140) :
    instrLen op = some 5 := by
  rcases (show op = 134 ∨ op = 135 ∨ op = 136 ∨ op = 137 ∨ op = 138 ∨ op = 139 by omega) with
    rfl | rfl | rfl | rfl | rfl | rfl <;> decide

private theorem skipcode_step_len {L rej p b pos op : Nat} {bs : List Nat} {s : St}
    (f3 : im (L + 12) = some (.ldr .b1 14 p 0, 4))
    (f4 : im (L + 16) = some (.addSubImm true true 64 31 14 57, 4))
    (f5 : im (L + 20) = some (.bcond 3 60, 4))
    (f6 : im (L + 24) = some (.addSubImm true true 64 31 14 128, 4))
    (f7 : im (L + 28) = some (.bcond 3 (relOff (L + 28) rej), 4))
    (f8 : im (L + 32) = some (.bcond 0 40, 4))
    (f9 : im (L + 36) = some (.addSubImm true true 64 31 14 140, 4))
    (f10 : im (L + 40) = some (.bcond 8 (relOff (L + 40) rej), 4))
    (f11 : im (L + 44) = some (.bcond 0 36, 4))
    (f12 : im (L + 48) = some (.addSubImm true true 64 31 14 131, 4))
    (f13 : im (L + 52) = some (.bcond 3 12, 4))
    (f14 : im (L + 56) = some (.addSubImm true true 64 31 14 134, 4))
    (f15 : im (L + 60) = some (.bcond 3 20, 4))
    (f16 : im (L + 64) = some (.movWide 2 64 14 5 0, 4))
    (f17 : im (L + 68) = some (.b false 16, 4))
    (f18 : im (L + 72) = some (.movWide 2 64 14 9 0, 4))
    (f19 : im (L + 76) = some (.b false 8, 4))
    (f20 : im (L + 80) = some (.movWide 2 64 14 1 0, 4))
    (hL : L + 104 ≤ rej) (hrej : rej < 2 ^ 64)
    (hp : p = 9 ∨ p = 10) (hbuf : Buf s.mem b bs) (hpos : pos < bs.length)
    (hsm : b + bs.length < HMAX) (hop : bs[pos] = op) (hby : op < 256)
    (hl : s.regs.length = 31) (hpc : s.pc = L + 12) (hregp : s.reg p = some (b + pos)) :
    Reaches s fun s' => Frame [14] s s' ∧
      match instrLen op with
      | none => s'.pc = rej
      | some ilen => s'.pc = L + 84 ∧ s'.reg 14 = some ilen := by
  have hp31 : p ≠ 31 := by rcases hp with rfl | rfl <;> decide
  have hop32 : op < 2 ^ 32 := by omega
  have hL64 : L + 104 < 2 ^ 64 := by omega
  have hrd : rdMem s.mem (((b + pos) % 2 ^ 64 + 0) % AArch64Env.W) .b1 = some op := by
    rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega), Nat.add_zero, ← hop]
    exact rdMem_b1_buf hbuf hpos
  have hr3 : (L + 12 + 4) % AArch64Env.W = L + 16 := modW_off hL64 rfl (by decide)
  have hr4 : (L + 16 + 4) % AArch64Env.W = L + 20 := modW_off hL64 rfl (by decide)
  have hr5t : (L + 20 + 60) % AArch64Env.W = L + 80 := modW_off hL64 rfl (by decide)
  have hr5f : (L + 20 + 4) % AArch64Env.W = L + 24 := modW_off hL64 rfl (by decide)
  have hr6 : (L + 24 + 4) % AArch64Env.W = L + 28 := modW_off hL64 rfl (by decide)
  have hr7t : (L + 28 + relOff (L + 28) rej) % AArch64Env.W = rej := relOff_any (Nat.lt_trans (by omega) hL64) hrej
  have hr7f : (L + 28 + 4) % AArch64Env.W = L + 32 := modW_off hL64 rfl (by decide)
  have hr8t : (L + 32 + 40) % AArch64Env.W = L + 72 := modW_off hL64 rfl (by decide)
  have hr8f : (L + 32 + 4) % AArch64Env.W = L + 36 := modW_off hL64 rfl (by decide)
  have hr9 : (L + 36 + 4) % AArch64Env.W = L + 40 := modW_off hL64 rfl (by decide)
  have hr10t : (L + 40 + relOff (L + 40) rej) % AArch64Env.W = rej := relOff_any (Nat.lt_trans (by omega) hL64) hrej
  have hr10f : (L + 40 + 4) % AArch64Env.W = L + 44 := modW_off hL64 rfl (by decide)
  have hr11t : (L + 44 + 36) % AArch64Env.W = L + 80 := modW_off hL64 rfl (by decide)
  have hr11f : (L + 44 + 4) % AArch64Env.W = L + 48 := modW_off hL64 rfl (by decide)
  have hr12 : (L + 48 + 4) % AArch64Env.W = L + 52 := modW_off hL64 rfl (by decide)
  have hr13t : (L + 52 + 12) % AArch64Env.W = L + 64 := modW_off hL64 rfl (by decide)
  have hr13f : (L + 52 + 4) % AArch64Env.W = L + 56 := modW_off hL64 rfl (by decide)
  have hr14 : (L + 56 + 4) % AArch64Env.W = L + 60 := modW_off hL64 rfl (by decide)
  have hr15t : (L + 60 + 20) % AArch64Env.W = L + 80 := modW_off hL64 rfl (by decide)
  have hr15f : (L + 60 + 4) % AArch64Env.W = L + 64 := modW_off hL64 rfl (by decide)
  have hr16 : (L + 64 + 4) % AArch64Env.W = L + 68 := modW_off hL64 rfl (by decide)
  have hr17 : (L + 68 + 16) % AArch64Env.W = L + 84 := modW_off hL64 rfl (by decide)
  have hr18 : (L + 72 + 4) % AArch64Env.W = L + 76 := modW_off hL64 rfl (by decide)
  have hr19 : (L + 76 + 8) % AArch64Env.W = L + 84 := modW_off hL64 rfl (by decide)
  have hr20 : (L + 80 + 4) % AArch64Env.W = L + 84 := modW_off hL64 rfl (by decide)
  have hop64lt : op < pow2 64 := by rw [pow2_64]; exact Nat.lt_trans hby (by decide)
  have h57 := cond_sub hop64lt (show 57 < pow2 64 by rw [pow2_64]; decide)
  have h128 := cond_sub hop64lt (show 128 < pow2 64 by rw [pow2_64]; decide)
  have h140 := cond_sub hop64lt (show 140 < pow2 64 by rw [pow2_64]; decide)
  have h131 := cond_sub hop64lt (show 131 < pow2 64 by rw [pow2_64]; decide)
  have h134 := cond_sub hop64lt (show 134 < pow2 64 by rw [pow2_64]; decide)
  have hop64 : op % 2 ^ 64 = op := Nat.mod_eq_of_lt (Nat.lt_trans hby (by decide))
  xs f3 using hl, hp31, hregp, hrd, hop32, hr3
  xs f4 using hl, hr4
  by_cases hc57 : op < 57
  · have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (57 % 2 ^ 64)).2 3 = some true := by
      rw [hop64, Nat.mod_eq_of_lt (by decide), h57.2.2.2.1]; simp [hc57]
    xs f5 using hfl, hr5t
    xs f20 using hl, hr20
    simp only [instrLen, hc57, ↓reduceIte]
    exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩
  · have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (57 % 2 ^ 64)).2 3 = some false := by
      rw [hop64, Nat.mod_eq_of_lt (by decide), h57.2.2.2.1]; simp [hc57]
    xs f5 using hfl, hr5f
    xs f6 using hl, hr6
    by_cases hc128 : op < 128
    · have hfl7 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (128 % 2 ^ 64)).2 3 = some true := by
        rw [hop64, Nat.mod_eq_of_lt (by decide), h128.2.2.2.1]; simp [hc128]
      xs f7 using hfl7, hr7t
      rw [instrLen_none_mid hc57 hc128]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl7 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (128 % 2 ^ 64)).2 3 = some false := by
        rw [hop64, Nat.mod_eq_of_lt (by decide), h128.2.2.2.1]; simp [hc128]
      xs f7 using hfl7, hr7f
      by_cases he128 : op = 128
      · have hfl8 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (128 % 2 ^ 64)).2 0 = some true := by
          rw [hop64, Nat.mod_eq_of_lt (by decide), h128.1]; simp [he128]
        xs f8 using hfl8, hr8t
        xs f18 using hl, hr18
        xs f19 using hr19
        subst he128
        simp only [instrLen, show ¬ (128 < 57) by decide, ↓reduceIte]
        exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩
      · have hfl8 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (128 % 2 ^ 64)).2 0 = some false := by
          rw [hop64, Nat.mod_eq_of_lt (by decide), h128.1]; simp [he128]
        xs f8 using hfl8, hr8f
        xs f9 using hl, hr9
        by_cases hgt140 : 140 < op
        · have hfl10 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (140 % 2 ^ 64)).2 8 = some true := by
            rw [hop64, Nat.mod_eq_of_lt (by decide), h140.2.2.2.2.1]; simp [hgt140]
          xs f10 using hfl10, hr10t
          rw [instrLen_none_gt hgt140]
          exact Reaches.refl ⟨by frame_tac, rfl⟩
        · have hfl10 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (140 % 2 ^ 64)).2 8 = some false := by
            rw [hop64, Nat.mod_eq_of_lt (by decide), h140.2.2.2.2.1]; simp [hgt140]
          xs f10 using hfl10, hr10f
          by_cases he140 : op = 140
          · have hfl11 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (140 % 2 ^ 64)).2 0 = some true := by
              rw [hop64, Nat.mod_eq_of_lt (by decide), h140.1]; simp [he140]
            xs f11 using hfl11, hr11t
            xs f20 using hl, hr20
            subst he140
            simp only [instrLen, show ¬ (140 < 57) by decide, ↓reduceIte]
            exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩
          · have hfl11 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (140 % 2 ^ 64)).2 0 = some false := by
              rw [hop64, Nat.mod_eq_of_lt (by decide), h140.1]; simp [he140]
            xs f11 using hfl11, hr11f
            xs f12 using hl, hr12
            by_cases hc131 : op < 131
            · have hfl13 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (131 % 2 ^ 64)).2 3 = some true := by
                rw [hop64, Nat.mod_eq_of_lt (by decide), h131.2.2.2.1]; simp [hc131]
              xs f13 using hfl13, hr13t
              xs f16 using hl, hr16
              xs f17 using hr17
              rw [instrLen_129_130 hc128 he128 hc131]
              exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩
            · have hfl13 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (131 % 2 ^ 64)).2 3 = some false := by
                rw [hop64, Nat.mod_eq_of_lt (by decide), h131.2.2.2.1]; simp [hc131]
              xs f13 using hfl13, hr13f
              xs f14 using hl, hr14
              by_cases hc134 : op < 134
              · have hfl15 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (134 % 2 ^ 64)).2 3 = some true := by
                  rw [hop64, Nat.mod_eq_of_lt (by decide), h134.2.2.2.1]; simp [hc134]
                xs f15 using hfl15, hr15t
                xs f20 using hl, hr20
                rw [instrLen_131_133 hc131 hc134]
                exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩
              · have hfl15 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (134 % 2 ^ 64)).2 3 = some false := by
                  rw [hop64, Nat.mod_eq_of_lt (by decide), h134.2.2.2.1]; simp [hc134]
                xs f15 using hfl15, hr15f
                xs f16 using hl, hr16
                xs f17 using hr17
                rw [instrLen_134_139 hc134 hgt140 he140]
                exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl]⟩

theorem skipcode_spec {L p rej b : Nat} {bs : List Nat}
    (hc : CodeAt (skipcodeT L p rej)) (hp : p = 9 ∨ p = 10)
    (hL : L + 104 ≤ rej) (hrej : rej < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX) :
    ∀ (cnt pos : Nat) (s : St),
      s.mem = m → s.regs.length = 31 → s.pc = L →
      s.reg p = some (b + pos) → s.reg 11 = some cnt → s.reg 24 = some (b + bs.length) →
      pos ≤ bs.length → cnt < 2 ^ 64 →
      Reaches s fun s' => Frame [p, 11, 14] s s' ∧
        match skipCode bs cnt pos with
        | none => s'.pc = rej
        | some pos' => s'.pc = L + 104 ∧ s'.reg p = some (b + pos') ∧ s'.reg 11 = some 0 ∧ pos' ≤ bs.length
  | 0, pos, s, hm, hl, hpc, hregp, h11, h24, hpos, _ => by
    simp only [skipcodeT, CodeAt_cons] at hc
    obtain ⟨f0, _⟩ := hc
    have hr0 : (L + 104) % AArch64Env.W = L + 104 := modW_small (by omega)
    xs f0 using h11, hr0
    exact Reaches.refl ⟨by frame_tac, rfl, hregp, h11, hpos⟩
  | k + 1, pos, s, hm, hl, hpc, hregp, h11, h24, hpos, hcnt => by
    have hc' := hc
    simp only [skipcodeT, CodeAt_cons] at hc'
    obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17,
      f18, f19, f20, f21, f22, f23, f24, f25, _⟩ := hc'
    have hpl : p < 31 := by rcases hp with rfl | rfl <;> decide
    have hp11 : p ≠ 11 := by rcases hp with rfl | rfl <;> decide
    have h11p : (11 : Nat) ≠ p := hp11.symm
    have hp14 : p ≠ 14 := by rcases hp with rfl | rfl <;> decide
    have hp24 : p ≠ 24 := by rcases hp with rfl | rfl <;> decide
    have hp31 : p ≠ 31 := by rcases hp with rfl | rfl <;> decide
    have hk1 : (k + 1) % 2 ^ 64 ≠ 0 := by omega
    have hr0 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
    have hr1 : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
    xs f0 using h11, hk1, hr0
    have hcs1 := cond_sub (show b + pos < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    xs f1 using hl, hp31, hregp, h24, hr1
    by_cases hlt : pos < bs.length
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + pos) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 2 =
          some false := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
          Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.1]; simp [show ¬ bs.length ≤ pos by omega]
      have hr2 : (L + 8 + 4) % AArch64Env.W = L + 12 := modW_small (by omega)
      xs f2 using hfl2, hr2
      have hop := hby bs[pos] (List.getElem_mem hlt)
      refine (skipcode_step_len f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20
        hL hrej hp (by simpa [hm] using hbuf) hlt hsm rfl hop (by simp [hl]) rfl
        (by simp [hregp])).trans fun s1 ⟨hf1, hstep⟩ => ?_
      have hs1_len : s1.regs.length = 31 := by rw [hf1.len]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
      have hs1_p : s1.reg p = some (b + pos) := by rw [hf1.regs p (by simp [hp14])]; exact hregp
      have hs1_11 : s1.reg 11 = some (k + 1) := by rw [hf1.regs 11 (by decide)]; exact h11
      have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; exact h24
      cases hil : instrLen bs[pos] with
      | none =>
        rw [hil] at hstep
        simp only [skipCode, skipInstr, List.getElem?_eq_getElem hlt, hil, bind, Option.bind]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac)
          (Frame.mono hf1 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp)), hstep⟩
      | some ilen =>
        rw [hil] at hstep
        obtain ⟨hpc1, hs1_14⟩ := hstep
        have hilen_le : ilen ≤ 9 := by
          simp only [instrLen] at hil
          split at hil <;> try (cases hil; decide)
          · split at hil <;> (try cases hil; decide)
            split at hil <;> (try cases hil; decide)
            split at hil <;> cases hil <;> decide
        have hr21 : (L + 84 + 4) % AArch64Env.W = L + 88 := modW_small (by omega)
        have hr22 : (L + 88 + 4) % AArch64Env.W = L + 92 := modW_small (by omega)
        have hbpil : ((b + pos) % 2 ^ 64 + ilen % 2 ^ 64) % 2 ^ 64 = b + (pos + ilen) := by
          unfold HMAX at hsm; omega
        xs f21 using hs1_len, hp31, hs1_p, hs1_14, hbpil, hr21
        have hcs2 := cond_sub (show b + (pos + ilen) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f22 using hs1_len, hpl, hp31, hp24, hs1_24, hr22
        by_cases hfit : pos + ilen ≤ bs.length
        · have hfl23 : AArch64Phys.cond (addSubOp true 64 ((b + (pos + ilen)) % 2 ^ 64)
              ((b + bs.length) % 2 ^ 64)).2 8 = some false := by
            rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
              Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.2.2.2.2.1]
            simp [show ¬ bs.length < pos + ilen by omega]
          have hr23 : (L + 92 + 4) % AArch64Env.W = L + 96 := modW_small (by omega)
          have hr24 : (L + 96 + 4) % AArch64Env.W = L + 100 := modW_small (by omega)
          have hr25 : (L + 100 + 18446744073709551516) % AArch64Env.W = L := by rw [W_eq]; omega
          have hksub : ((k + 1) % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = k := by omega
          xs f23 using hfl23, hr23
          xs f24 using hs1_len, hp11, hs1_11, hksub, hr24
          xs f25 using hr25
          refine (skipcode_spec hc hp hL hrej hbuf hby hsm k (pos + ilen) _
            (by simp [hs1_mem]) (by simp [hs1_len]) rfl
            (by simp [h11p, hs1_len, hpl])
            (by simp [hs1_len])
            (by simp [hp24, hs1_24]) hfit (by omega)).mono fun s2 ⟨hf2, hs2⟩ => ?_
          refine ⟨Frame.trans_same (by frame_tac)
            (Frame.trans_same (Frame.mono hf1 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp))
              (Frame.trans_same (by frame_tac) hf2)), ?_⟩
          simpa [skipCode, skipInstr, List.getElem?_eq_getElem hlt, hil, hfit] using hs2
        · have hfl23 : AArch64Phys.cond (addSubOp true 64 ((b + (pos + ilen)) % 2 ^ 64)
              ((b + bs.length) % 2 ^ 64)).2 8 = some true := by
            rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
              Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.2.2.2.2.1]
            simp [show bs.length < pos + ilen by omega]
          have hr23 : (L + 92 + relOff (L + 92) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
          xs f23 using hfl23, hr23
          simp only [skipCode, skipInstr, List.getElem?_eq_getElem hlt, hil, show ¬ pos + ilen ≤ bs.length from hfit,
            ↓reduceIte, bind, Option.bind]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac)
            (Frame.trans_same (Frame.mono hf1 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp))
              (by frame_tac)), rfl⟩
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + pos) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 2 =
          some true := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
          Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.1]; simp [show bs.length ≤ pos by omega]
      have hr2 : (L + 8 + relOff (L + 8) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
      xs f2 using hfl2, hr2
      have hnone : bs[pos]? = none := List.getElem?_eq_none (by omega)
      simp only [skipCode, skipInstr, hnone, bind, Option.bind]
      exact Reaches.refl ⟨by frame_tac, rfl⟩

private theorem skipwords_pre {L rej b k pos : Nat} {bs : List Nat} {m : Mem} {s : St}
    (f0 : im L = some (.cbz false 64 15 168, 4))
    (f1 : im (L + 4) = some (.addSubImm false false 64 14 10 20, 4))
    (f2 : im (L + 8) = some (.addSubReg true true 64 31 14 24 .lsl 0, 4))
    (f3 : im (L + 12) = some (.bcond 8 (relOff (L + 12) rej), 4))
    (f4 : im (L + 16) = some (.ldr .w4 14 10 16, 4))
    (f5 : im (L + 20) = some (.addSubReg false false 64 15 15 14 .lsl 32, 4))
    (f6 : im (L + 24) = some (.ldr .w4 11 10 12, 4))
    (f7 : im (L + 28) = some (.addSubImm false false 64 10 10 20, 4))
    (hL : L + 168 ≤ rej) (hrej : rej < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hm : s.mem = m) (hl : s.regs.length = 31) (hpc : s.pc = L)
    (h10 : s.reg 10 = some (b + pos)) (h15 : s.reg 15 = some (k + 1)) (h24 : s.reg 24 = some (b + bs.length))
    (hpos : pos ≤ bs.length) (hcnt : k + 1 < 2 ^ 32) :
    Reaches s fun s0 => Frame [10, 11, 14, 15] s s0 ∧
      if pos + 20 ≤ bs.length then
        s0.pc = L + 32 ∧ s0.mem = m ∧ s0.regs.length = 31 ∧
        s0.reg 10 = some (b + (pos + 20)) ∧ s0.reg 11 = some (u32 bs (pos + 12)) ∧
        s0.reg 15 = some (k + 1 + u32 bs (pos + 16) * 2 ^ 32) ∧ s0.reg 24 = some (b + bs.length)
      else s0.pc = rej := by
  have hk1 : (k + 1) % 2 ^ 64 ≠ 0 := by omega
  have hL64 : L + 168 < 2 ^ 64 := by omega
  have hbp20 : ((b + pos) % 2 ^ 64 + 20 % 2 ^ 64) % 2 ^ 64 = b + (pos + 20) := by
    unfold HMAX at hsm; omega
  xs f0 using h15, hk1, modW_step0 hL64 4
  xs f1 using hl, h10, hbp20, modW_step hL64 4 4 8
  have hcs1 := cond_sub (show b + (pos + 20) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
  xs f2 using hl, h24, modW_step hL64 8 4 12
  by_cases hp20 : pos + 20 ≤ bs.length
  · have hfl3 : AArch64Phys.cond (addSubOp true 64 ((b + (pos + 20)) % 2 ^ 64)
        ((b + bs.length) % 2 ^ 64)).2 8 = some false := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.2.2.1]
      simp [show ¬ bs.length < pos + 20 by omega]
    have hna := u32_lt hby (pos + 16)
    have hnc := u32_lt hby (pos + 12)
    have hrd4 : rdMem s.mem (((b + pos) % 2 ^ 64 + 16) % AArch64Env.W) .w4 = some (u32 bs (pos + 16)) := by
      rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega),
        show b + pos + 16 = b + (pos + 16) by omega, hm]
      exact rdMem_w4_buf hbuf (by omega)
    have hrd6 : rdMem s.mem (((b + pos) % 2 ^ 64 + 12) % AArch64Env.W) .w4 = some (u32 bs (pos + 12)) := by
      rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega),
        show b + pos + 12 = b + (pos + 12) by omega, hm]
      exact rdMem_w4_buf hbuf (by omega)
    have hx15_eq : ((k + 1) % 2 ^ 64 + u32 bs (pos + 16) % 2 ^ 64 * 2 ^ 32 % 2 ^ 64) % 2 ^ 64 =
        k + 1 + u32 bs (pos + 16) * 2 ^ 32 := by
      rw [Nat.mod_eq_of_lt (by omega : k + 1 < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : u32 bs (pos + 16) < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : u32 bs (pos + 16) * 2 ^ 32 < 2 ^ 64),
        Nat.mod_eq_of_lt (by omega : k + 1 + u32 bs (pos + 16) * 2 ^ 32 < 2 ^ 64)]
    xs f3 using hfl3, modW_step hL64 12 4 16
    xs f4 using hl, h10, hrd4, hna, modW_step hL64 16 4 20
    xs f5 using hl, h15, hx15_eq, modW_step hL64 20 4 24
    xs f6 using hl, h10, hrd6, hnc, modW_step hL64 24 4 28
    xs f7 using hl, h10, hbp20, modW_step hL64 28 4 32
    exact Reaches.refl ⟨by frame_tac, by simp [hp20, hm, hl, h24]⟩
  · have hfl3 : AArch64Phys.cond (addSubOp true 64 ((b + (pos + 20)) % 2 ^ 64)
        ((b + bs.length) % 2 ^ 64)).2 8 = some true := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.2.2.2.2.1]
      simp [show bs.length < pos + 20 by omega]
    have hr3 : (L + 12 + relOff (L + 12) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
    xs f3 using hfl3, hr3
    exact Reaches.refl ⟨by frame_tac, by simp [hp20]⟩

private theorem skipwords_post {L rej b k na annOff : Nat} {bs : List Nat} {m : Mem} {s1 : St}
    (f136 : im (L + 136) = some (.addSubReg false false 64 14 31 15 .lsr 32, 4))
    (f140 : im (L + 140) = some (.logicReg .orr false 32 15 31 15 .lsl 0, 4))
    (f144 : im (L + 144) = some (.addSubReg false false 64 11 14 14 .lsl 1, 4))
    (f148 : im (L + 148) = some (.addSubReg false false 64 10 10 11 .lsl 2, 4))
    (f152 : im (L + 152) = some (.addSubReg true true 64 31 10 24 .lsl 0, 4))
    (f156 : im (L + 156) = some (.bcond 8 (relOff (L + 156) rej), 4))
    (f160 : im (L + 160) = some (.addSubImm true false 64 15 15 1, 4))
    (f164 : im (L + 164) = some (.b false 18446744073709551452, 4))
    (hL : L + 168 ≤ rej) (hrej : rej < 2 ^ 64)
    (hsm : b + bs.length < HMAX) (hna : na < 2 ^ 32) (hcnt : k + 1 < 2 ^ 32) (hann_le : annOff ≤ bs.length)
    (hm : s1.mem = m) (hl : s1.regs.length = 31) (hpc : s1.pc = L + 136)
    (h10 : s1.reg 10 = some (b + annOff)) (h15 : s1.reg 15 = some (k + 1 + na * 2 ^ 32))
    (h24 : s1.reg 24 = some (b + bs.length)) :
    Reaches s1 fun s2 => Frame [10, 11, 14, 15] s1 s2 ∧
      if annOff + na * 12 ≤ bs.length then
        s2.pc = L ∧ s2.mem = m ∧ s2.regs.length = 31 ∧
        s2.reg 10 = some (b + (annOff + na * 12)) ∧ s2.reg 15 = some k ∧ s2.reg 24 = some (b + bs.length)
      else s2.pc = rej := by
  have hL64 : L + 168 < 2 ^ 64 := by omega
  have hdiv : (k + 1 + na * 2 ^ 32) % 2 ^ 64 / 2 ^ 32 = na := by omega
  have hdiv64 : na % 2 ^ 64 = na := Nat.mod_eq_of_lt (by omega)
  have hmod : (k + 1 + na * 2 ^ 32) % 2 ^ 32 = k + 1 := by omega
  have h3na : (na % 2 ^ 64 + na % 2 ^ 64 * 2 ^ 1 % 2 ^ 64) % 2 ^ 64 = 3 * na := by omega
  have h10_val : ((b + annOff) % 2 ^ 64 + (3 * na) % 2 ^ 64 * 2 ^ 2 % 2 ^ 64) % 2 ^ 64 =
      b + (annOff + na * 12) := by
    rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega : b + annOff < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega : 3 * na < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega : 3 * na * 2 ^ 2 < 2 ^ 64)]
    unfold HMAX at hsm
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  xs f136 using hl, h15, hdiv, hdiv64, modW_step hL64 136 4 140
  xs f140 using hl, h15, hmod, Nat.mod_eq_of_lt hcnt, hcnt, modW_step hL64 140 4 144
  xs f144 using hl, h3na, modW_step hL64 144 4 148
  xs f148 using hl, h10, h10_val, modW_step hL64 148 4 152
  xs f152 using hl, h24, modW_step hL64 152 4 156
  have hcs2 := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
  by_cases hfit : annOff + na * 12 ≤ bs.length
  · have hfl156 : AArch64Phys.cond (addSubOp true 64
        ((b + (annOff + na * 12)) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 8 = some false := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.2.2.2.2.1]
      simp [show ¬ bs.length < annOff + na * 12 by omega]
    have hksub : ((k + 1) % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = k := by omega
    xs f156 using hfl156, modW_step hL64 156 4 160
    xs f160 using hl, hksub, modW_step hL64 160 4 164
    xs f164 using modW_back hL64 164 18446744073709551452
    exact Reaches.refl ⟨by frame_tac, by simp [hfit, hm, hl, h24]⟩
  · have hfl156 : AArch64Phys.cond (addSubOp true 64
        ((b + (annOff + na * 12)) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 8 = some true := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.2.2.2.2.1]
      simp [show bs.length < annOff + na * 12 by omega]
    have hr156 : (L + 156 + relOff (L + 156) rej) % AArch64Env.W = rej := relOff_any (by omega) hrej
    xs f156 using hfl156, hr156
    exact Reaches.refl ⟨by frame_tac, by simp [hfit]⟩

theorem skipwords_spec {L rej b : Nat} {bs : List Nat}
    (hc : CodeAt (skipwordsT L rej)) (hL : L + 168 ≤ rej) (hrej : rej < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX) :
    ∀ (cnt pos : Nat) (s : St),
      s.mem = m → s.regs.length = 31 → s.pc = L →
      s.reg 10 = some (b + pos) → s.reg 15 = some cnt → s.reg 24 = some (b + bs.length) →
      pos ≤ bs.length → cnt < 2 ^ 32 →
      Reaches s fun s' => Frame [10, 11, 14, 15] s s' ∧
        match skipWords bs cnt pos with
        | none => s'.pc = rej
        | some pos' => s'.pc = L + 168 ∧ s'.reg 10 = some (b + pos') ∧ s'.reg 15 = some 0 ∧ pos' ≤ bs.length
  | 0, pos, s, hm, hl, hpc, h10, h15, h24, hpos, _ => by
    simp only [skipwordsT, CodeAt_append, CodeAt_cons] at hc
    obtain ⟨⟨⟨f0, _⟩, _⟩, _⟩ := hc
    have hr0 : (L + 168) % AArch64Env.W = L + 168 := modW_small (by omega)
    xs f0 using h15, hr0
    exact Reaches.refl ⟨by frame_tac, rfl, h10, h15, hpos⟩
  | k + 1, pos, s, hm, hl, hpc, h10, h15, h24, hpos, hcnt => by
    have hc' := hc
    simp only [skipwordsT, CodeAt_append, CodeAt_cons] at hc'
    obtain ⟨⟨⟨f0, f1, f2, f3, f4, f5, f6, f7, _⟩, hsc⟩,
      f136, f140, f144, f148, f152, f156, f160, f164, _⟩ := hc'
    refine (skipwords_pre f0 f1 f2 f3 f4 f5 f6 f7 hL hrej hbuf hby hsm hm hl hpc h10 h15 h24 hpos hcnt).trans
      fun s0 ⟨hf0, hpre⟩ => ?_
    by_cases hp20 : pos + 20 ≤ bs.length
    · rw [if_pos hp20] at hpre
      obtain ⟨hpc0, hm0, hl0, h10_0, h11_0, h15_0, h24_0⟩ := hpre
      have hrnc := rdU32_eq_some_u32 hby (by omega : pos + 12 + 4 ≤ bs.length)
      have hrna := rdU32_eq_some_u32 hby (by omega : pos + 16 + 4 ≤ bs.length)
      refine (skipcode_spec hsc (Or.inr rfl) (by omega) hrej hbuf hby hsm (u32 bs (pos + 12)) (pos + 20) s0
        hm0 hl0 hpc0 h10_0 h11_0 h24_0 hp20 (Nat.lt_trans (u32_lt hby (pos + 12)) (by decide))).trans
        fun s1 ⟨hf1, hsc_res⟩ => ?_
      have hf01 : Frame [10, 11, 14, 15] s s1 := Frame.trans_same hf0 (Frame.mono hf1 (by decide))
      cases hsc_eq : skipCode bs (u32 bs (pos + 12)) (pos + 20) with
      | none =>
        rw [hsc_eq] at hsc_res
        simp only [skipWords, hp20, ↓reduceIte, hrnc, hrna, hsc_eq, bind, Option.bind]
        exact Reaches.refl ⟨hf01, hsc_res⟩
      | some annOff =>
        rw [hsc_eq] at hsc_res
        obtain ⟨hpc1, hs1_10, _, hann_le⟩ := hsc_res
        have hs1_len : s1.regs.length = 31 := by rw [hf1.len]; exact hl0
        have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm0
        have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; exact h24_0
        have hs1_15 : s1.reg 15 = some (k + 1 + u32 bs (pos + 16) * 2 ^ 32) := by
          rw [hf1.regs 15 (by decide)]; exact h15_0
        refine (skipwords_post f136 f140 f144 f148 f152 f156 f160 f164 hL hrej hsm
          (u32_lt hby (pos + 16)) hcnt hann_le hs1_mem hs1_len hpc1 hs1_10 hs1_15 hs1_24).trans
          fun s2 ⟨hf2, hpost⟩ => ?_
        have hf02 : Frame [10, 11, 14, 15] s s2 := Frame.trans_same hf01 hf2
        by_cases hfit : annOff + u32 bs (pos + 16) * 12 ≤ bs.length
        · rw [if_pos hfit] at hpost
          obtain ⟨hpc2, hm2, hl2, h10_2, h15_2, h24_2⟩ := hpost
          refine (skipwords_spec hc hL hrej hbuf hby hsm k (annOff + u32 bs (pos + 16) * 12) s2
            hm2 hl2 hpc2 h10_2 h15_2 h24_2 hfit (by omega)).mono fun s3 ⟨hf3, hs3⟩ => ?_
          refine ⟨Frame.trans_same hf02 hf3, ?_⟩
          simpa [skipWords, hp20, hrnc, hrna, hsc_eq, hfit] using hs3
        · rw [if_neg hfit] at hpost
          simp only [skipWords, hp20, ↓reduceIte, hrnc, hrna, hsc_eq, hfit, bind, Option.bind]
          exact Reaches.refl ⟨hf02, hpost⟩
    · rw [if_neg hp20] at hpre
      simp only [skipWords, hp20, ↓reduceIte]
      exact Reaches.refl ⟨hf0, hpre⟩

private theorem lookup_loop_spec {L t b annEnd target : Nat} {bs : List Nat}
    (f4 : im (L + 16) = some (.addSubReg true true 64 31 10 11 .lsl 0, 4))
    (f5 : im (L + 20) = some (.bcond 0 24, 4))
    (f6 : im (L + 24) = some (.ldr .w4 14 10 0, 4))
    (f7 : im (L + 28) = some (.addSubReg true true 64 31 14 t .lsl 0, 4))
    (f8 : im (L + 32) = some (.bcond 0 12, 4))
    (f9 : im (L + 36) = some (.addSubImm false false 64 10 10 12, 4))
    (f10 : im (L + 40) = some (.b false 18446744073709551592, 4))
    (hL : L + 44 < 2 ^ 64) (ht : t = 8 ∨ t = 15)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hend : annEnd ≤ bs.length) (htgt : target < 2 ^ 64) :
    ∀ (rem off : Nat) (s : St),
      off + rem * 12 = annEnd →
      s.mem = m → s.regs.length = 31 → s.pc = L + 16 →
      s.reg 10 = some (b + off) → s.reg 11 = some (b + annEnd) → s.reg t = some target →
      Reaches s fun s' => Frame [10, 14] s s' ∧
        s'.pc = L + 44 ∧ s'.reg 11 = some (b + annEnd) ∧ s'.regs.length = 31 ∧
        match lookupLoop bs target rem off with
        | none => s'.reg 10 = some (b + annEnd)
        | some (ad, aa) => ∃ q, off ≤ q ∧ q + 12 ≤ annEnd ∧
            s'.reg 10 = some (b + q) ∧ q ≠ annEnd ∧ u32 bs (q + 4) = ad ∧ u32 bs (q + 8) = aa
  | 0, off, s, heq, hm, hl, hpc, h10, h11, hrt => by
    have hoff : off = annEnd := by omega
    subst hoff
    have hcs := cond_sub (show b + off < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + off < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hr4 : (L + 16 + 4) % AArch64Env.W = L + 20 := modW_small (by omega)
    have hr5 : (L + 20 + 24) % AArch64Env.W = L + 44 := modW_small (by omega)
    xs f4 using hl, h10, h11, hr4
    have hfl : AArch64Phys.cond (addSubOp true 64 ((b + off) % 2 ^ 64) ((b + off) % 2 ^ 64)).2 0 =
        some true := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs.1]; simp
    xs f5 using hfl, hr5
    exact Reaches.refl ⟨by frame_tac, rfl, h11, hl, h10⟩
  | k + 1, off, s, heq, hm, hl, hpc, h10, h11, hrt => by
    have hp12 : off + 12 ≤ annEnd := by omega
    have hne : off ≠ annEnd := by omega
    have ht31 : t ≠ 31 := by rcases ht with rfl | rfl <;> decide
    have ht14 : (14 : Nat) ≠ t := by rcases ht with rfl | rfl <;> decide
    have ht10 : (10 : Nat) ≠ t := by rcases ht with rfl | rfl <;> decide
    have hcs1 := cond_sub (show b + off < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + annEnd < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hr4 : (L + 16 + 4) % AArch64Env.W = L + 20 := modW_small (by omega)
    have hr5 : (L + 20 + 4) % AArch64Env.W = L + 24 := modW_small (by omega)
    have hr6 : (L + 24 + 4) % AArch64Env.W = L + 28 := modW_small (by omega)
    have hr7 : (L + 28 + 4) % AArch64Env.W = L + 32 := modW_small (by omega)
    xs f4 using hl, h10, h11, hr4
    have hfl5 : AArch64Phys.cond (addSubOp true 64 ((b + off) % 2 ^ 64) ((b + annEnd) % 2 ^ 64)).2 0 =
        some false := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.1]; simp [hne]
    xs f5 using hfl5, hr5
    have hru0 := rdU32_eq_some_u32 hby (by omega : off + 4 ≤ bs.length)
    have hru4 := rdU32_eq_some_u32 hby (by omega : off + 4 + 4 ≤ bs.length)
    have hru8 := rdU32_eq_some_u32 hby (by omega : off + 8 + 4 ≤ bs.length)
    have hu0 := u32_lt hby off
    have hrd6 : rdMem s.mem (((b + off) % 2 ^ 64 + 0) % AArch64Env.W) .w4 = some (u32 bs off) := by
      rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega), Nat.add_zero, hm]
      exact rdMem_w4_buf hbuf (by omega)
    xs f6 using hl, h10, hrd6, hu0, hr6
    have hcs2 := cond_sub (show u32 bs off < pow2 64 by rw [pow2_64]; omega)
      (show target < pow2 64 by rw [pow2_64]; exact htgt)
    xs f7 using hl, ht31, ht14, hrt, hr7
    by_cases heqt : u32 bs off = target
    · have hfl8 : AArch64Phys.cond (addSubOp true 64 (u32 bs off % 2 ^ 64) (target % 2 ^ 64)).2 0 =
          some true := by
        rw [Nat.mod_eq_of_lt (by omega : u32 bs off < 2 ^ 64),
          Nat.mod_eq_of_lt htgt, hcs2.1]; simp [heqt]
      have hr8 : (L + 32 + 12) % AArch64Env.W = L + 44 := modW_small (by omega)
      xs f8 using hfl8, hr8
      simp only [lookupLoop, hru0, heqt, beq_self_eq_true, ↓reduceIte, hru4, hru8, bind, Option.bind, pure]
      exact Reaches.refl ⟨by frame_tac, rfl, (by simp [h11]), (by simp [hl]),
        ⟨off, Nat.le_refl _, hp12, (by simp [h10]), hne, rfl, rfl⟩⟩
    · have hfl8 : AArch64Phys.cond (addSubOp true 64 (u32 bs off % 2 ^ 64) (target % 2 ^ 64)).2 0 =
          some false := by
        rw [Nat.mod_eq_of_lt (by omega : u32 bs off < 2 ^ 64),
          Nat.mod_eq_of_lt htgt, hcs2.1]; simp [heqt]
      have hr8 : (L + 32 + 4) % AArch64Env.W = L + 36 := modW_small (by omega)
      have hr9 : (L + 36 + 4) % AArch64Env.W = L + 40 := modW_small (by omega)
      have hr10 : (L + 40 + 18446744073709551592) % AArch64Env.W = L + 16 := by rw [W_eq]; omega
      have hoff12 : ((b + off) % 2 ^ 64 + 12) % 2 ^ 64 = b + (off + 12) := by
        unfold HMAX at hsm; omega
      xs f8 using hfl8, hr8
      xs f9 using hl, h10, hoff12, hr9
      xs f10 using hr10
      refine (lookup_loop_spec f4 f5 f6 f7 f8 f9 f10 hL ht hbuf hby hsm hend htgt k (off + 12) _
        (by omega) (by simp [hm]) (by simp [hl]) rfl
        (by simp [hl]) (by simp [h11]) (by simp [ht10, ht14, hrt])).mono fun s' ⟨hf, hpc', h11', hl', hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) hf, hpc', h11', hl', ?_⟩
      simp only [lookupLoop, hru0, beq_false_of_ne heqt, bind, Option.bind]
      cases hll : lookupLoop bs target k (off + 12) with
      | none => rw [hll] at hres; exact hres
      | some p =>
        rw [hll] at hres
        obtain ⟨q, hq1, hq2, hq3, hq4, hq5, hq6⟩ := hres
        exact ⟨q, by omega, hq2, hq3, hq4, hq5, hq6⟩

theorem lookup_spec {L t b wordOff annOff na target : Nat} {bs : List Nat} {s : St}
    (hc : CodeAt (lookupT L t)) (hL : L + 44 < 2 ^ 64) (ht : t = 8 ∨ t = 15)
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hna : u32 bs (wordOff + 16) = na)
    (hann : annOff + na * 12 ≤ bs.length) (htgt : target < 2 ^ 64)
    (hl : s.regs.length = 31) (hpc : s.pc = L)
    (h12 : s.reg 12 = some (b + wordOff)) (h9 : s.reg 9 = some (b + annOff))
    (hrt : s.reg t = some target) :
    Reaches s fun s' => Frame [10, 11, 14] s s' ∧
      s'.pc = L + 44 ∧ s'.reg 11 = some (b + (annOff + na * 12)) ∧ s'.regs.length = 31 ∧
      match lookup bs annOff na target with
      | none => s'.reg 10 = some (b + (annOff + na * 12))
      | some (ad, aa) => ∃ q, annOff ≤ q ∧ q + 12 ≤ annOff + na * 12 ∧
          s'.reg 10 = some (b + q) ∧ q ≠ annOff + na * 12 ∧ u32 bs (q + 4) = ad ∧ u32 bs (q + 8) = aa := by
  simp only [lookupT, CodeAt_cons] at hc
  obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, _⟩ := hc
  have ht14 : (14 : Nat) ≠ t := by rcases ht with rfl | rfl <;> decide
  have ht11 : (11 : Nat) ≠ t := by rcases ht with rfl | rfl <;> decide
  have ht10 : (10 : Nat) ≠ t := by rcases ht with rfl | rfl <;> decide
  have hr0 : (L + 4) % AArch64Env.W = L + 4 := modW_small (by omega)
  have hr1 : (L + 4 + 4) % AArch64Env.W = L + 8 := modW_small (by omega)
  have hr2 : (L + 8 + 4) % AArch64Env.W = L + 12 := modW_small (by omega)
  have hr3 : (L + 12 + 4) % AArch64Env.W = L + 16 := modW_small (by omega)
  have hlt_na := u32_lt hby (wordOff + 16)
  rw [hna] at hlt_na
  have hrd0 : rdMem s.mem (((b + wordOff) % 2 ^ 64 + 16) % AArch64Env.W) .w4 = some na := by
    rw [ea_mod_small (by unfold HMAX at hsm; omega) (by unfold HMAX at hsm; omega),
      show b + wordOff + 16 = b + (wordOff + 16) by omega, ← hna]
    exact rdMem_w4_buf hbuf (by omega)
  have h3na : (na % 2 ^ 64 + na % 2 ^ 64 * 2 ^ 1 % 2 ^ 64) % 2 ^ 64 = 3 * na := by omega
  have hbam : (b + annOff) % 2 ^ 64 = b + annOff := Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega)
  have h11_val : ((b + annOff) % 2 ^ 64 + (3 * na) % 2 ^ 64 * 2 ^ 2 % 2 ^ 64) % 2 ^ 64 =
      b + (annOff + na * 12) := by
    rw [hbam, Nat.mod_eq_of_lt (by omega : 3 * na < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega : 3 * na * 2 ^ 2 < 2 ^ 64)]
    unfold HMAX at hsm
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  xs f0 using hl, h12, hrd0, hlt_na, hr0
  xs f1 using hl, h3na, hr1
  xs f2 using hl, h9, h11_val, hr2
  xs f3 using hl, h9, hbam, hr3
  refine (lookup_loop_spec f4 f5 f6 f7 f8 f9 f10 hL ht hbuf hby hsm hann htgt na annOff _
    rfl (by simp) (by simp [hl]) rfl
    (by simp [hl]) (by simp [hl])
    (by simp [ht14, ht11, ht10, hrt])).mono fun s' ⟨hf, hpc', h11', hl', hres⟩ => ?_
  exact ⟨Frame.trans_same (by frame_tac) (Frame.mono hf (by decide)), hpc', h11', hl', hres⟩

theorem dispatch_loop {B defTgt op : Nat} : ∀ (tbl : List (Nat × Nat)) (a : Nat) (s : St),
    CodeAt (dispatchCode a tbl) →
    im (a + 8 * tbl.length) = some (.b false (relOff (a + 8 * tbl.length) defTgt), 4) →
    (∀ p ∈ tbl, p.1 < 4096 ∧ B ≤ p.2 ∧ p.2 ≤ B + 2728) →
    B ≤ a → a + 8 * tbl.length ≤ B + 2728 → B ≤ defTgt → defTgt ≤ B + 2728 →
    B + 2728 < 2 ^ 64 → s.regs.length = 31 → s.pc = a → s.reg 15 = some op → op < 256 →
    Reaches s fun s' => Frame [] s s' ∧ s'.pc = dispatchTgt op defTgt tbl ∧
      s'.regs.length = 31 ∧ s'.reg 15 = some op
  | [], a, s, _, fj, _, _, ha, hd1, hd2, hB, hl, hpc, h15, _ => by
    simp only [List.length_nil, Nat.mul_zero, Nat.add_zero] at fj ha
    have hr : (a + relOff a defTgt) % AArch64Env.W = defTgt := relOff_target (by omega) (by omega)
    xs fj using hr
    exact Reaches.refl ⟨by frame_tac, rfl, hl, h15⟩
  | (k, tgt) :: rest, a, s, hc, fj, hps, hBa, ha, hd1, hd2, hB, hl, hpc, h15, hop => by
    simp only [dispatchCode, CodeAt_cons] at hc
    obtain ⟨f0, f1, hcr⟩ := hc
    obtain ⟨hk, hBt, htgt⟩ := hps (k, tgt) (by simp)
    have hcs := cond_sub (show op < pow2 64 by rw [pow2_64]; omega) (show k < pow2 64 by rw [pow2_64]; omega)
    have hr0 : (a + 4) % AArch64Env.W = a + 4 := modW_small (by simp at ha; omega)
    xs f0 using hl, h15, hr0
    by_cases heq : op = k
    · have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (k % 2 ^ 64)).2 0 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.1]; simp [heq]
      have hr1 : (a + 4 + relOff (a + 4) tgt) % AArch64Env.W = tgt :=
        relOff_target (by simp at ha; omega) (by omega)
      xs f1 using hfl, hr1
      exact Reaches.refl ⟨by frame_tac, by simp [dispatchTgt, heq], hl, h15⟩
    · have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (k % 2 ^ 64)).2 0 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.1]; simp [heq]
      have hr1 : (a + 4 + 4) % AArch64Env.W = a + 8 := modW_small (by simp at ha; omega)
      xs f1 using hfl, hr1
      have fj' : im (a + 8 + 8 * rest.length) = some (.b false (relOff (a + 8 + 8 * rest.length) defTgt), 4) := by
        rwa [show a + 8 + 8 * rest.length = a + 8 * ((k, tgt) :: rest).length by simp; omega]
      refine (dispatch_loop rest (a + 8) _ hcr fj' (fun p hp => hps p (by simp [hp]))
        (by omega) (by simp at ha; omega) hd1 hd2 hB (by simp [hl]) rfl
        (by simp [h15]) hop).mono fun s' ⟨hf, hpc', hl', h15'⟩ => ?_
      exact ⟨Frame.trans_same (by frame_tac) hf, by simp [dispatchTgt, heq, hpc'], hl', h15'⟩

/-! ### `skipWords` / `wordAt` / `parse` structural bridges -/

theorem skipInstr_bounds {bs : List Nat} {p q : Nat} (h : skipInstr bs p = some q) :
    p < q ∧ q ≤ bs.length ∧ ∃ op ilen, bs[p]? = some op ∧ instrLen op = some ilen ∧ q = p + ilen := by
  simp only [skipInstr, bind, Option.bind] at h
  cases h0 : bs[p]? with
  | none => simp only [h0] at h; cases h
  | some op =>
    simp only [h0] at h
    cases hil : instrLen op with
    | none => simp only [hil] at h; cases h
    | some ilen =>
      simp only [hil] at h
      have hpos_il : 0 < ilen := by
        simp only [instrLen] at hil
        repeat (split at hil <;> try (cases hil; decide))
        cases hil
      split at h
      · cases h; exact ⟨by omega, ‹_›, op, ilen, rfl, hil, rfl⟩
      · cases h

theorem skipCode_bounds {bs : List Nat} : ∀ n p q,
    skipCode bs n p = some q → p ≤ q ∧ (p ≤ bs.length → q ≤ bs.length)
  | 0, p, q, h => by simp [skipCode] at h; subst h; exact ⟨Nat.le_refl _, id⟩
  | n + 1, p, q, h => by
    simp only [skipCode, bind, Option.bind] at h
    cases hi : skipInstr bs p with
    | none => simp only [hi] at h; cases h
    | some m =>
      simp only [hi] at h
      have hm := skipInstr_bounds hi
      have ih := skipCode_bounds n m q h
      exact ⟨by omega, fun _ => ih.2 hm.2.1⟩

theorem rdU32_isSome {bs : List Nat} {pos : Nat} (hp : pos + 4 ≤ bs.length) :
    ∃ v, rdU32 bs pos = some v := by
  have h0 : bs[pos]? = some bs[pos] := List.getElem?_eq_getElem (by omega)
  have h1 : bs[pos + 1]? = some bs[pos + 1] := List.getElem?_eq_getElem (by omega)
  have h2 : bs[pos + 2]? = some bs[pos + 2] := List.getElem?_eq_getElem (by omega)
  have h3 : bs[pos + 3]? = some bs[pos + 3] := List.getElem?_eq_getElem (by omega)
  exact ⟨bs[pos] + 256 * bs[pos + 1] + 65536 * bs[pos + 2] + 16777216 * bs[pos + 3],
    by simp [rdU32, h0, h1, h2, h3, bind, Option.bind, pure]⟩

theorem readWordAt_iff_skipWords1 {bs : List Nat} {pos : Nat} :
    (∃ wm, readWordAt bs pos = some wm ∧ skipWords bs 1 pos = some wm.nextOff ∧
      pos + 20 ≤ wm.nextOff ∧ wm.nextOff ≤ bs.length) ↔
    ∃ nxt, skipWords bs 1 pos = some nxt := by
  constructor
  · rintro ⟨wm, _, hs, _⟩; exact ⟨wm.nextOff, hs⟩
  · rintro ⟨nxt, hs⟩
    unfold skipWords at hs
    split at hs
    · rename_i hp20
      simp only [bind, Option.bind] at hs
      cases hnc : rdU32 bs (pos + 12) with
      | none => simp only [hnc] at hs; cases hs
      | some nc =>
        simp only [hnc] at hs
        cases hna : rdU32 bs (pos + 16) with
        | none => simp only [hna] at hs; cases hs
        | some na =>
          simp only [hna] at hs
          cases hsc : skipCode bs nc (pos + 20) with
          | none => simp only [hsc] at hs; cases hs
          | some annOff =>
            simp only [hsc] at hs
            split at hs
            · rename_i hann
              simp only [skipWords] at hs
              cases hs
              obtain ⟨ins, hins⟩ := rdU32_isSome (by omega : pos + 4 ≤ bs.length)
              obtain ⟨outs, houts⟩ := rdU32_isSome (by omega : pos + 4 + 4 ≤ bs.length)
              obtain ⟨locs, hlocs⟩ := rdU32_isSome (by omega : pos + 8 + 4 ≤ bs.length)
              have hle := (skipCode_bounds nc (pos + 20) annOff hsc).1
              refine ⟨⟨ins, outs, locs, nc, na, pos + 20, annOff, annOff + na * 12⟩, ?_, ?_, by dsimp; omega, hann⟩
              · simp [readWordAt, hp20, hins, houts, hlocs, hnc, hna, hsc, hann, bind, Option.bind]
              · simp [skipWords, hp20, hnc, hna, hsc, hann, bind, Option.bind]
            · cases hs
    · cases hs

theorem skipWords_succ_iff {bs : List Nat} {n pos endPos : Nat} :
    skipWords bs (n + 1) pos = some endPos ↔
      ∃ wm, readWordAt bs pos = some wm ∧ skipWords bs 1 pos = some wm.nextOff ∧
        pos + 20 ≤ wm.nextOff ∧ wm.nextOff ≤ bs.length ∧ skipWords bs n wm.nextOff = some endPos := by
  constructor
  · intro h
    have h1 : ∃ mid, skipWords bs 1 pos = some mid ∧ skipWords bs n mid = some endPos := by
      unfold skipWords at h
      split at h
      · rename_i hp20
        simp only [bind, Option.bind] at h
        cases hnc : rdU32 bs (pos + 12) with
        | none => simp only [hnc] at h; cases h
        | some nc =>
          simp only [hnc] at h
          cases hna : rdU32 bs (pos + 16) with
          | none => simp only [hna] at h; cases h
          | some na =>
            simp only [hna] at h
            cases hsc : skipCode bs nc (pos + 20) with
            | none => simp only [hsc] at h; cases h
            | some annOff =>
              simp only [hsc] at h
              split at h
              · rename_i hann
                exact ⟨annOff + na * 12,
                  by simp [skipWords, hp20, hnc, hna, hsc, hann, bind, Option.bind], h⟩
              · cases h
      · cases h
    obtain ⟨mid, hm1, hmn⟩ := h1
    obtain ⟨wm, hwm, hs1, h20, hle⟩ := readWordAt_iff_skipWords1.2 ⟨mid, hm1⟩
    rw [hm1] at hs1; cases hs1
    exact ⟨wm, hwm, hm1, h20, hle, hmn⟩
  · rintro ⟨wm, _, hs1, _, _, hmn⟩
    unfold skipWords at hs1 ⊢
    split at hs1 <;> try cases hs1
    rename_i hp20
    rw [if_pos hp20]
    simp only [bind, Option.bind] at hs1 ⊢
    cases hnc : rdU32 bs (pos + 12) with
    | none => simp only [hnc] at hs1; cases hs1
    | some nc =>
      simp only [hnc] at hs1 ⊢
      cases hna : rdU32 bs (pos + 16) with
      | none => simp only [hna] at hs1; cases hs1
      | some na =>
        simp only [hna] at hs1 ⊢
        cases hsc : skipCode bs nc (pos + 20) with
        | none => simp only [hsc] at hs1; cases hs1
        | some annOff =>
          simp only [hsc] at hs1 ⊢
          split at hs1 <;> try cases hs1
          rename_i hann
          rw [if_pos hann]
          simp only [skipWords] at hs1
          rw [Option.some.inj hs1]; exact hmn

theorem skipWords_wordAt {bs : List Nat} : ∀ (nw w ws endPos : Nat),
    skipWords bs nw ws = some endPos → w < nw →
    ∃ off wm, skipWords bs w ws = some off ∧ readWordAt bs off = some wm ∧
      wordAt bs ws w = some wm ∧ off + 20 ≤ wm.nextOff ∧ wm.nextOff ≤ bs.length ∧
      skipWords bs (nw - w - 1) wm.nextOff = some endPos
  | 0, _, _, _, _, hw => by omega
  | _ + 1, 0, ws, endPos, h, _ => by
    obtain ⟨wm, hwm, _, h20, hle, hrest⟩ := skipWords_succ_iff.1 h
    exact ⟨ws, wm, rfl, hwm, by simp [wordAt, hwm], h20, hle, hrest⟩
  | k + 1, w + 1, ws, endPos, h, hw => by
    obtain ⟨wm, hwm, hs1, _, _, hrest⟩ := skipWords_succ_iff.1 h
    obtain ⟨off, wm', hsw, hrw, hwa, h20', hle', hr'⟩ :=
      skipWords_wordAt k w wm.nextOff endPos hrest (by omega)
    have hsw1 : skipWords bs (w + 1) ws = some off := by
      have : ∀ (a : Nat) (p : Nat), skipWords bs (a + 1) p =
          (skipWords bs 1 p).bind (skipWords bs a) := by
        intro a p
        unfold skipWords
        split
        · rename_i hp20
          simp only [bind, Option.bind]
          cases hnc : rdU32 bs (p + 12) <;> simp
          cases hna : rdU32 bs (p + 16) <;> simp
          cases hsc : skipCode bs _ (p + 20) <;> simp
          split <;> simp [skipWords]
        · simp [Option.bind]
      rw [this w ws, hs1, Option.bind_some]; exact hsw
    exact ⟨off, wm', hsw1, hrw, by simp [wordAt, hwm, hwa, bind, Option.bind], h20', hle',
      by rwa [show k + 1 - (w + 1) - 1 = k - w - 1 by omega]⟩

theorem skipData_bounds {bs : List Nat} : ∀ nd pos pos',
    skipData bs nd pos = some pos' → pos ≤ pos' ∧ (pos ≤ bs.length → pos' ≤ bs.length)
  | 0, pos, pos', h => by simp [skipData] at h; subst h; exact ⟨Nat.le_refl _, id⟩
  | k + 1, pos, pos', h => by
    simp only [skipData, bind, Option.bind] at h
    cases hl : rdU32 bs pos with
    | none => simp only [hl] at h; cases h
    | some len =>
      simp only [hl] at h
      split at h
      · have ih := skipData_bounds k (pos + 4 + len) pos' h
        exact ⟨by omega, fun _ => ih.2 ‹_›⟩
      · cases h

set_option maxHeartbeats 1600000 in
theorem parse_spec {B adrImm R nxt D b : Nat} {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt)
    (hB : B + 2728 < 2 ^ 64) (hD : D + 16 < 2 ^ 64)
    (hadr : ((B / 4096) * 4096 + adrImm) % AArch64Env.W = D)
    (hD0 : rdMem m D .x8 = some b) (hD8 : rdMem m (D + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hl : s.regs.length = 31) (hpc : s.pc = B) (hm : s.mem = m) :
    Reaches s fun s' => Frame PassScratch s s' ∧
      match parse bs with
      | none => s'.pc = B + 2708
      | some (main, nw, ws) =>
        if nw ≤ main then s'.pc = B + 2708
        else match wordAt bs ws main with
          | none => s'.pc = B + 2708
          | some mainWm =>
            if mainWm.ins ≠ 0 then s'.pc = B + 2708
            else s'.pc = B + 488 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
              s'.reg 0 = some b ∧ s'.reg 1 = some nw ∧ s'.reg 12 = some (b + ws) ∧
              s'.reg 24 = some (b + bs.length) ∧ s'.reg 25 = some (b + ws) ∧
              ws ≤ bs.length ∧ nw < 2 ^ 32 ∧ skipWords bs nw ws = some bs.length := by
  have hcHdr := pb.cHdr
  simp only [CodeAt_cons] at hcHdr
  obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, _⟩ := hcHdr
  have hcM1 := pb.cMid1; simp only [CodeAt_cons] at hcM1; obtain ⟨f108, f112, _⟩ := hcM1
  have hcM2 := pb.cMid2; simp only [CodeAt_cons] at hcM2; obtain ⟨f284, f288, f292, f296, f300, f304, _⟩ := hcM2
  have hcM3 := pb.cMid3; simp only [CodeAt_cons] at hcM3; obtain ⟨f476, f480, f484, _⟩ := hcM3
  have hd_lt : b + bs.length < 2 ^ 64 := by unfold HMAX at hsm; omega
  have hD_m : D % 2 ^ 64 = D := Nat.mod_eq_of_lt (by omega)
  have hb_m : b % 2 ^ 64 = b := Nat.mod_eq_of_lt (by omega)
  have hlen_m : bs.length % 2 ^ 64 = bs.length := Nat.mod_eq_of_lt (by omega)
  have hbend : (b % 2 ^ 64 + bs.length % 2 ^ 64) % 2 ^ 64 = b + bs.length := by omega
  have hrdD0 : rdMem m ((D % 2 ^ 64 + 0) % AArch64Env.W) .x8 = some b := by
    rw [ea_mod_small (by omega) (by omega), Nat.add_zero]; exact hD0
  have hrdD8 : rdMem m ((D % 2 ^ 64 + 8) % AArch64Env.W) .x8 = some bs.length := by
    rw [ea_mod_small (by omega) (by omega)]; exact hD8
  xs f0 using hl, Nat.add_zero, hadr, hD_m, modW_step0 hB 4
  xs f1 using hl, hm, hrdD0, hb_m, modW_step hB 4 4 8
  xs f2 using hl, hm, hrdD8, hlen_m, modW_step hB 8 4 12
  xs f3 using hl, hbend, modW_step hB 12 4 16
  have hcs0 := cond_sub (show bs.length < pow2 64 by rw [pow2_64]; omega) (show 16 < pow2 64 by rw [pow2_64]; decide)
  xs f4 using hl, modW_step hB 16 4 20
  by_cases h16 : bs.length < 16
  · have hfl5 : AArch64Phys.cond (addSubOp true 64 (bs.length % 2 ^ 64) (16 % 2 ^ 64)).2 3 = some true := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs0.2.2.2.1]; simp [h16]
    xs f5 using hfl5, relOff_step hB 20 2708
    simp only [parse, h16, ↓reduceIte]
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  · have hfl5 : AArch64Phys.cond (addSubOp true 64 (bs.length % 2 ^ 64) (16 % 2 ^ 64)).2 3 = some false := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs0.2.2.2.1]; simp [h16]
    have hru0 := rdU32_eq_some_u32 hby (by omega : 0 + 4 ≤ bs.length)
    have hu0 := u32_lt hby 0
    have hrd6 : rdMem m ((b % 2 ^ 64 + 0) % AArch64Env.W) .w4 = some (u32 bs 0) := by
      rw [ea_mod_small (by omega) (by omega), Nat.add_zero]
      simpa using rdMem_w4_buf hbuf (by omega : 0 + 4 ≤ bs.length)
    have hmag0 : 20563 * 1 % 2 ^ 64 = 20563 := by decide
    have hmag : (20563 % 2 ^ 16 + 12626 * 2 ^ 16 + 20563 / 2 ^ 32 * 2 ^ 32) % 2 ^ 64 = MAGIC := by decide
    xs f5 using hfl5, modW_step hB 20 4 24
    xs f6 using hl, hm, hrd6, hu0, modW_step hB 24 4 28
    xs f7 using hl, hmag0, modW_step hB 28 4 32
    xs f8 using hl, hmag, modW_step hB 32 4 36
    have hcs1 := cond_sub (show u32 bs 0 < pow2 64 by rw [pow2_64]; omega) (show MAGIC < pow2 64 by rw [pow2_64]; decide)
    xs f9 using hl, modW_step hB 36 4 40
    by_cases hmq : u32 bs 0 = MAGIC
    · have hfl10 : AArch64Phys.cond (addSubOp true 64 (u32 bs 0 % 2 ^ 64) (MAGIC % 2 ^ 64)).2 1 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs1.2.1]; simp [hmq]
      have hru4 := rdU32_eq_some_u32 hby (by omega : 4 + 4 ≤ bs.length)
      have hru8 := rdU32_eq_some_u32 hby (by omega : 8 + 4 ≤ bs.length)
      have hru12 := rdU32_eq_some_u32 hby (by omega : 12 + 4 ≤ bs.length)
      have hu4 := u32_lt hby 4
      have hu8 := u32_lt hby 8
      have hu12 := u32_lt hby 12
      have hrd11 : rdMem m ((b % 2 ^ 64 + 4) % AArch64Env.W) .w4 = some (u32 bs 4) := by
        rw [ea_mod_small (by omega) (by omega)]
        exact rdMem_w4_buf hbuf (by omega : 4 + 4 ≤ bs.length)
      have hrd12 : rdMem m ((b % 2 ^ 64 + 8) % AArch64Env.W) .w4 = some (u32 bs 8) := by
        rw [ea_mod_small (by omega) (by omega)]
        exact rdMem_w4_buf hbuf (by omega : 8 + 4 ≤ bs.length)
      have hrd13 : rdMem m ((b % 2 ^ 64 + 12) % AArch64Env.W) .w4 = some (u32 bs 12) := by
        rw [ea_mod_small (by omega) (by omega)]
        exact rdMem_w4_buf hbuf (by omega : 12 + 4 ≤ bs.length)
      have hb16 : (b % 2 ^ 64 + 16 % 2 ^ 64) % 2 ^ 64 = b + 16 := by omega
      xs f10 using hfl10, modW_step hB 40 4 44
      xs f11 using hl, hm, hrd11, hu4, modW_step hB 44 4 48
      xs f12 using hl, hm, hrd12, hu8, modW_step hB 48 4 52
      xs f13 using hl, hm, hrd13, hu12, modW_step hB 52 4 56
      xs f14 using hl, hb16, modW_step hB 56 4 60
      refine (skipdata_spec pb.cSd (by omega) (by omega) hbuf hby hsm (u32 bs 12) 16 _
        (by simp [hm]) (by simp [hl]) rfl (by simp [hl]) (by simp [hl]) (by simp [hl])
        (by omega) (by omega)).trans fun s1 ⟨hf1, hsd⟩ => ?_
      have hs1_len : s1.regs.length = 31 := by rw [hf1.len]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      have hs1_0 : s1.reg 0 = some b := by rw [hf1.regs 0 (by decide)]; simp [hl]
      have hs1_1 : s1.reg 1 = some (u32 bs 8) := by rw [hf1.regs 1 (by decide)]; simp [hl]
      have hs1_12 : s1.reg 12 = some (u32 bs 4) := by rw [hf1.regs 12 (by decide)]; simp [hl]
      have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; simp [hl]
      cases hsd_eq : skipData bs (u32 bs 12) 16 with
      | none =>
        rw [hsd_eq] at hsd
        simp only [parse, h16, ↓reduceIte, hru0, hmq, bne_self_eq_false, Bool.false_eq_true,
          hru4, hru8, hru12, hsd_eq, bind, Option.bind]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hsd⟩
      | some ws =>
        rw [hsd_eq] at hsd
        obtain ⟨hpc1, hs1_10, hws_le⟩ := hsd
        have hws_m : (b + ws) % 2 ^ 64 = b + ws := Nat.mod_eq_of_lt (by omega)
        have hu8_m : u32 bs 8 % 2 ^ 64 = u32 bs 8 := Nat.mod_eq_of_lt (by omega)
        xs f108 using hs1_len, hs1_10, hws_m, modW_step hB 108 4 112
        xs f112 using hs1_len, hs1_1, hu8_m, modW_step hB 112 4 116
        refine (skipwords_spec pb.cSw1 (by omega) (by omega) hbuf hby hsm (u32 bs 8) ws _
          (by simp [hs1_mem]) (by simp [hs1_len]) rfl (by simp [hs1_10])
          (by simp [hs1_len]) (by simp [hs1_24]) hws_le hu8).trans fun s2 ⟨hf2, hsw⟩ => ?_
        have hs2_len : s2.regs.length = 31 := by rw [hf2.len]; simp [hs1_len]
        have hs2_mem : s2.mem = m := by rw [hf2.mem]; simp [hs1_mem]
        have hs2_0 : s2.reg 0 = some b := by rw [hf2.regs 0 (by decide)]; simp [hs1_0]
        have hs2_1 : s2.reg 1 = some (u32 bs 8) := by rw [hf2.regs 1 (by decide)]; simp [hs1_1]
        have hs2_12 : s2.reg 12 = some (u32 bs 4) := by rw [hf2.regs 12 (by decide)]; simp [hs1_12]
        have hs2_24 : s2.reg 24 = some (b + bs.length) := by rw [hf2.regs 24 (by decide)]; simp [hs1_24]
        have hs2_25 : s2.reg 25 = some (b + ws) := by rw [hf2.regs 25 (by decide)]; simp [hs1_len]
        cases hsw_eq : skipWords bs (u32 bs 8) ws with
        | none =>
          rw [hsw_eq] at hsw
          simp only [parse, h16, ↓reduceIte, hru0, hmq, bne_self_eq_false, Bool.false_eq_true,
            hru4, hru8, hru12, hsd_eq, hsw_eq, bind, Option.bind]
          exact Reaches.refl ⟨Frame.trans_same (Frame.trans_same (by frame_tac)
            (Frame.mono hf1 (by decide))) (Frame.trans_same (by frame_tac) (Frame.mono hf2 (by decide))), hsw⟩
        | some endPos =>
          rw [hsw_eq] at hsw
          obtain ⟨hpc2, hs2_10, _, hend_le⟩ := hsw
          have hcs2 := cond_sub (show b + endPos < pow2 64 by rw [pow2_64]; omega)
            (show b + bs.length < pow2 64 by rw [pow2_64]; omega)
          xs f284 using hs2_len, hs2_10, hs2_24, modW_step hB 284 4 288
          by_cases heq_end : endPos = bs.length
          · subst heq_end
            have hfl288 : AArch64Phys.cond (addSubOp true 64 ((b + bs.length) % 2 ^ 64)
                ((b + bs.length) % 2 ^ 64)).2 1 = some false := by
              rw [Nat.mod_eq_of_lt (by omega), hcs2.2.1]; simp
            xs f288 using hfl288, modW_step hB 288 4 292
            have hcs3 := cond_sub (show u32 bs 4 < pow2 64 by rw [pow2_64]; omega)
              (show u32 bs 8 < pow2 64 by rw [pow2_64]; omega)
            xs f292 using hs2_len, hs2_12, hs2_1, modW_step hB 292 4 296
            have hp_eq : parse bs = some (u32 bs 4, u32 bs 8, ws) := by
              simp [parse, h16, hru0, hmq, hru4, hru8, hru12, hsd_eq, hsw_eq, bind, Option.bind]
            by_cases hmge : u32 bs 8 ≤ u32 bs 4
            · have hfl296 : AArch64Phys.cond (addSubOp true 64 (u32 bs 4 % 2 ^ 64)
                  (u32 bs 8 % 2 ^ 64)).2 2 = some true := by
                rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs3.2.2.1]; simp [hmge]
              xs f296 using hfl296, relOff_step hB 296 2708
              rw [hp_eq]
              simp only [hmge, ↓reduceIte]
              exact Reaches.refl ⟨Frame.trans_same (Frame.trans_same (by frame_tac)
                (Frame.mono hf1 (by decide))) (Frame.trans_same (by frame_tac)
                (Frame.trans_same (Frame.mono hf2 (by decide)) (by frame_tac))), rfl⟩
            · have hfl296 : AArch64Phys.cond (addSubOp true 64 (u32 bs 4 % 2 ^ 64)
                  (u32 bs 8 % 2 ^ 64)).2 2 = some false := by
                rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs3.2.2.1]; simp [hmge]
              have hu4_m : u32 bs 4 % 2 ^ 64 = u32 bs 4 := Nat.mod_eq_of_lt (by omega)
              xs f296 using hfl296, modW_step hB 296 4 300
              xs f300 using hs2_len, hs2_25, hws_m, modW_step hB 300 4 304
              xs f304 using hs2_len, hs2_12, hu4_m, modW_step hB 304 4 308
              obtain ⟨mainOff, mainWm, hsw_m, hrw_m, hwa_m, h20_m, hle_m, _⟩ :=
                skipWords_wordAt (u32 bs 8) (u32 bs 4) ws bs.length hsw_eq (by omega)
              have hm_ins : u32 bs mainOff = mainWm.ins := by
                unfold readWordAt at hrw_m
                split at hrw_m <;> try cases hrw_m
                simp only [bind, Option.bind] at hrw_m
                simp only [rdU32_eq_some_u32 hby (by omega : mainOff + 4 ≤ bs.length)] at hrw_m
                cases hrd4 : rdU32 bs (mainOff + 4) with
                | none => simp only [hrd4] at hrw_m; cases hrw_m
                | some _ =>
                  simp only [hrd4] at hrw_m
                  cases hrd8 : rdU32 bs (mainOff + 8) with
                  | none => simp only [hrd8] at hrw_m; cases hrw_m
                  | some _ =>
                    simp only [hrd8] at hrw_m
                    cases hrd12 : rdU32 bs (mainOff + 12) with
                    | none => simp only [hrd12] at hrw_m; cases hrw_m
                    | some nc =>
                      simp only [hrd12] at hrw_m
                      cases hrd16 : rdU32 bs (mainOff + 16) with
                      | none => simp only [hrd16] at hrw_m; cases hrw_m
                      | some _ =>
                        simp only [hrd16] at hrw_m
                        cases hsc : skipCode bs nc (mainOff + 20) with
                        | none => simp only [hsc] at hrw_m; cases hrw_m
                        | some _ =>
                          simp only [hsc] at hrw_m
                          split at hrw_m <;> cases hrw_m; rfl
              refine (skipwords_spec (b := b) pb.cSw2 (by omega) (by omega)
                (by simpa [hs2_mem] using hbuf) hby (by unfold HMAX at hsm ⊢; omega) (u32 bs 4) ws _
                rfl (by simp [hs2_len]) rfl
                (by simp [hs2_len]) (by simp [hs2_len])
                (by simp [hs2_24]) hws_le hu4).trans fun s3 ⟨hf3, hswm⟩ => ?_
              rw [hsw_m] at hswm
              obtain ⟨hpc3, hs3_10, _, hm_le⟩ := hswm
              have hs3_len : s3.regs.length = 31 := by rw [hf3.len]; simp [hs2_len]
              have hs3_mem : s3.mem = m := by rw [hf3.mem]; simp [hs2_mem]
              have hs3_0 : s3.reg 0 = some b := by rw [hf3.regs 0 (by decide)]; simp [hs2_0]
              have hs3_1 : s3.reg 1 = some (u32 bs 8) := by rw [hf3.regs 1 (by decide)]; simp [hs2_1]
              have hs3_24 : s3.reg 24 = some (b + bs.length) := by rw [hf3.regs 24 (by decide)]; simp [hs2_24]
              have hs3_25 : s3.reg 25 = some (b + ws) := by rw [hf3.regs 25 (by decide)]; simp [hs2_25]
              have hm_ins_lt32 : mainWm.ins < 2 ^ 32 := by rw [← hm_ins]; exact u32_lt hby mainOff
              have hm_ins_lt : mainWm.ins < 2 ^ 64 := by omega
              have hrd476 : rdMem s3.mem (((b + mainOff) % 2 ^ 64 + 0) % AArch64Env.W) .w4 = some mainWm.ins := by
                rw [ea_mod_small (by omega) (by omega), Nat.add_zero, hs3_mem, ← hm_ins]
                exact rdMem_w4_buf hbuf (by omega : mainOff + 4 ≤ bs.length)
              xs f476 using hs3_len, hs3_10, hrd476, hm_ins_lt32, modW_step hB 476 4 480
              by_cases hnz : mainWm.ins = 0
              · xs f480 using hs3_len, Nat.mod_eq_of_lt hm_ins_lt, hnz, modW_step hB 480 4 484
                xs f484 using hs3_len, hs3_25, hws_m, modW_step hB 484 4 488
                rw [hp_eq]
                simp only [hmge, ↓reduceIte, hwa_m, hnz, not_true_eq_false]
                refine Reaches.refl ⟨Frame.trans_same (Frame.trans_same (by frame_tac)
                  (Frame.mono hf1 (by decide))) (Frame.trans_same (by frame_tac)
                  (Frame.trans_same (Frame.mono hf2 (by decide))
                  (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf3 (by decide))
                  (by frame_tac))))), rfl, by simp [hs3_len], hs3_mem, ?_, ?_, ?_, ?_, ?_,
                  hws_le, hu8, hsw_eq⟩
                · simp [hs3_0]
                · simp [hs3_1]
                · simp [hs3_len]
                · simp [hs3_24]
                · simp [hs3_25]
              · xs f480 using hs3_len, Nat.mod_eq_of_lt hm_ins_lt, hnz, relOff_step hB 480 2708
                rw [hp_eq]
                simp only [hmge, ↓reduceIte, hwa_m, hnz, not_false_eq_true]
                exact Reaches.refl ⟨Frame.trans_same (Frame.trans_same (by frame_tac)
                  (Frame.mono hf1 (by decide))) (Frame.trans_same (by frame_tac)
                  (Frame.trans_same (Frame.mono hf2 (by decide))
                  (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf3 (by decide))
                  (by frame_tac))))), rfl⟩
          · have hfl288 : AArch64Phys.cond (addSubOp true 64 ((b + endPos) % 2 ^ 64)
                ((b + bs.length) % 2 ^ 64)).2 1 = some true := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs2.2.1]
              simp [heq_end]
            xs f288 using hfl288, relOff_step hB 288 2708
            have hp_none : parse bs = none := by
              simp [parse, h16, hru0, hmq, hru4, hru8, hru12, hsd_eq, hsw_eq, heq_end,
                bind, Option.bind]
            rw [hp_none]
            exact Reaches.refl ⟨Frame.trans_same (Frame.trans_same (by frame_tac)
              (Frame.mono hf1 (by decide))) (Frame.trans_same (by frame_tac)
              (Frame.trans_same (Frame.mono hf2 (by decide)) (by frame_tac))), rfl⟩
    · have hfl10 : AArch64Phys.cond (addSubOp true 64 (u32 bs 0 % 2 ^ 64) (MAGIC % 2 ^ 64)).2 1 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs1.2.1]; simp [hmq]
      have hr10 : (B + 40 + relOff (B + 40) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f10 using hfl10, hr10
      have hru0 := rdU32_eq_some_u32 hby (by omega : 0 + 4 ≤ bs.length)
      simp only [parse, h16, ↓reduceIte, hru0, show (u32 bs 0 != MAGIC) = true by simp [hmq],
        bind, Option.bind]
      exact Reaches.refl ⟨by frame_tac, rfl⟩

/-! ### `hstep` dispatch and opcode specifications (`[6?, 7, 10, 11, 13, 14, 15]` frame) -/

private theorem rdMem_word_field {m : Mem} {b wordOff fieldOff : Nat} {bs : List Nat}
    (hbuf : Buf m b bs) (hsm : b + bs.length < HMAX) (hw : wordOff + fieldOff + 4 ≤ bs.length) :
    rdMem m (((b + wordOff) % 2 ^ 64 + fieldOff) % AArch64Env.W) .w4 = some (u32 bs (wordOff + fieldOff)) := by
  have h1 : b + wordOff < 2 ^ 64 := by unfold HMAX at hsm; omega
  have h2 : b + wordOff + fieldOff < 2 ^ 64 := by unfold HMAX at hsm; omega
  rw [ea_mod_small h1 h2, show b + wordOff + fieldOff = b + (wordOff + fieldOff) by omega]
  exact rdMem_w4_buf hbuf hw

theorem hstep_prim_spec {B adrImm R nxt ws nw outs locals annOff na hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hhd : hd < HMAX) (hha : ha < HMAX) (hop : op < 57)
    (hm : s.mem = m) (hl : s.regs.length = 31) (hpc : s.pc = B + 896)
    (h13 : s.reg 13 = some hd) (h7 : s.reg 7 = some ha) (h15 : s.reg 15 = some op) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      match hstepF bs nw ws outs locals annOff na hd ha op 0 with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some 1
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some 1 ∧ nd < HMAX ∧ na' < HMAX := by
  have hcDE := pb.cDispEnd; simp only [CodeAt_cons] at hcDE; obtain ⟨_, f896, f900, _⟩ := hcDE
  have hcHv := pb.cHave; simp only [CodeAt_cons] at hcHv; obtain ⟨f1616, f1620, f1624, f1628, _⟩ := hcHv
  have hcHvE := pb.cHaveEnd; simp only [CodeAt_cons] at hcHvE; obtain ⟨f1644, f1648, _⟩ := hcHvE
  have hcEx := pb.cExit; simp only [CodeAt_cons] at hcEx; obtain ⟨f1652, f1656, f1660, _⟩ := hcEx
  have hpe := primEffTree_spec op hop
  have hpb := primEffTree_bound op hop
  have hcs50 := cond_sub (show op < pow2 64 by rw [pow2_64]; omega) (show 50 < pow2 64 by rw [pow2_64]; decide)
  have hr896 : (B + 896 + 4) % AArch64Env.W = B + 900 := modW_small (by omega)
  xs f896 using hl, h15, hr896
  by_cases h50 : op = 50
  · subst h50
    have hfl900 : AArch64Phys.cond (addSubOp true 64 (50 % 2 ^ 64) (50 % 2 ^ 64)).2 0 = some true := by
      rw [Nat.mod_eq_of_lt (by decide), hcs50.1]; simp
    have hr900 : (B + 900 + 752) % AArch64Env.W = B + 1652 := modW_small (by omega)
    xs f900 using hfl900, hr900
    have h13lt : hd < 2 ^ 64 := by unfold HMAX at hhd; omega
    by_cases hd0 : hd = 0
    · have hr1652 : (B + 1652 + relOff (B + 1652) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f1652 using h13, Nat.mod_eq_of_lt h13lt, hd0, hr1652
      simp only [hstepF, hpe, show (0 : Nat) < (primEffTree.eval 50).1 by
        have : primEffTree.eval 50 = (1, 0) := by decide
        rw [this]; omega, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hr1652 : (B + 1652 + 4) % AArch64Env.W = B + 1656 := modW_small (by omega)
      have hr1656 : (B + 1656 + 4) % AArch64Env.W = B + 1660 := modW_small (by omega)
      have hr1660 : (B + 1660 + 788) % AArch64Env.W = B + 2448 := modW_small (by omega)
      xs f1652 using h13, Nat.mod_eq_of_lt h13lt, hd0, hr1652
      xs f1656 using hl, hr1656
      xs f1660 using hr1660
      have hev : primEffTree.eval 50 = (1, 0) := by decide
      simp (config := { decide := true }) only [hstepF, hpe, hev, show ¬ hd < 1 by omega, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl], hm, by simp [hl]⟩
  · have hfl900 : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (50 % 2 ^ 64)).2 0 = some false := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs50.1]; simp [h50]
    have hr900 : (B + 900 + 4) % AArch64Env.W = B + 904 := modW_small (by omega)
    xs f900 using hfl900, hr900
    have hceff : CodeAt (primEffTree.emit (B + 904) (B + 1616)) := by
      rw [primEffTree_emit_eq]; exact pb.cEff
    refine (eff_tree_spec primEffTree (B + 904) (B + 1616) op _ primEffTree_wf hceff
      (by rw [primEffTree_bytes]; omega) (by omega) (by simp [hl]) rfl (by simp [h15]) (by omega)).trans
      fun s1 ⟨hf1, hpc1, hs1_10, hs1_11, hs1_len, _⟩ => ?_
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
    have hs1_13 : s1.reg 13 = some hd := by rw [hf1.regs 13 (by decide)]; simp [h13]
    have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
    let eIn := (primEffTree.eval op).1
    let eOut := (primEffTree.eval op).2
    change s1.reg 10 = some eIn at hs1_10
    change s1.reg 11 = some eOut at hs1_11
    have he1 : eIn ≤ 4 := hpb.1
    have he2 : eOut ≤ 4 := hpb.2
    have hcs_in := cond_sub (show hd < pow2 64 by rw [pow2_64]; unfold HMAX at hhd; omega)
      (show eIn < pow2 64 by rw [pow2_64]; omega)
    have hr1616 : (B + 1616 + 4) % AArch64Env.W = B + 1620 := modW_small (by omega)
    xs f1616 using hs1_len, hs1_13, hs1_10, hr1616
    by_cases hlt : hd < eIn
    · have hfl1620 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (eIn % 2 ^ 64)).2 3 = some true := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hhd; omega), Nat.mod_eq_of_lt (by omega), hcs_in.2.2.2.1]; simp [hlt]
      have hr1620 : (B + 1620 + relOff (B + 1620) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f1620 using hfl1620, hr1620
      have hlt' : hd < (primEffTree.eval op).1 := hlt
      simp only [hstepF, hpe, hlt', ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
    · have hfl1620 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (eIn % 2 ^ 64)).2 3 = some false := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hhd; omega), Nat.mod_eq_of_lt (by omega), hcs_in.2.2.2.1]; simp [hlt]
      have hr1620 : (B + 1620 + 4) % AArch64Env.W = B + 1624 := modW_small (by omega)
      have hr1624 : (B + 1624 + 4) % AArch64Env.W = B + 1628 := modW_small (by omega)
      have hr1628 : (B + 1628 + 4) % AArch64Env.W = B + 1632 := modW_small (by omega)
      have hs2_13 : (hd % 2 ^ 64 + 2 ^ 64 - eIn % 2 ^ 64) % 2 ^ 64 = hd - eIn := by
        unfold HMAX at hhd; omega
      have hd_add : ((hd - eIn) % 2 ^ 64 + eOut % 2 ^ 64) % 2 ^ 64 = hd - eIn + eOut := by
        unfold HMAX at hhd; omega
      xs f1620 using hfl1620, hr1620
      xs f1624 using hs1_len, hs1_13, hs1_10, hs2_13, hr1624
      xs f1628 using hs1_len, hs1_11, hd_add, hr1628
      refine (chk_spec pb.cChkEff (by omega) (by omega) (by decide) (by decide)
        (by unfold HMAX at hhd; omega : hd - eIn + eOut < 2 ^ 64) (by simp [hs1_len]) rfl
        (by simp [hs1_len])).trans
        fun s2 ⟨hf2, hpc2, hs2_len, hs2_13'⟩ => ?_
      have hs2_7 : s2.reg 7 = some ha := by rw [hf2.regs 7 (by decide)]; simp [hs1_7]
      have hs2_mem : s2.mem = m := by rw [hf2.mem]; simp [hs1_mem]
      have hlt' : ¬ hd < (primEffTree.eval op).1 := hlt
      by_cases hov : HMAX ≤ hd - eIn + eOut
      · rw [if_pos hov] at hpc2
        have hov' : HMAX ≤ hd - (primEffTree.eval op).1 + (primEffTree.eval op).2 := hov
        simp only [hstepF, hpe, hlt', beq_false_of_ne h50, Bool.false_eq_true, HMAX_flat_eq, hov', ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
          (Frame.trans_same (by frame_tac) (Frame.mono hf2 (by decide)))), hpc2⟩
      · rw [if_neg hov] at hpc2
        have hr1644 : (B + 1644 + 4) % AArch64Env.W = B + 1648 := modW_small (by omega)
        have hr1648 : (B + 1648 + 700) % AArch64Env.W = B + 2348 := modW_small (by omega)
        xs f1644 using hs2_len, hr1644
        xs f1648 using hr1648
        have hov' : ¬ HMAX ≤ hd - (primEffTree.eval op).1 + (primEffTree.eval op).2 := hov
        simp only [hstepF, hpe, hlt', beq_false_of_ne h50, Bool.false_eq_true, HMAX_flat_eq, hov', ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
          (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf2 (by decide)) (by frame_tac)))),
          rfl, by simp [hs2_len], hs2_mem, by simp [hs2_13', eIn, eOut], by simp [hs2_7], by simp [hs2_len],
          by omega, hha⟩

theorem hstep_push_rstack_spec {B adrImm R nxt b ws nw outs locals annOff na p hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX) (hp : p + 5 ≤ bs.length ∨ op = 0x80 ∨ op = 0x83 ∨ op = 0x84 ∨ op = 0x85)
    (hop : op = 0x80 ∨ op = 0x81 ∨ op = 0x82 ∨ op = 0x83 ∨ op = 0x84 ∨ op = 0x85)
    (hm : s.mem = m) (hl : s.regs.length = 31)
    (hpc : s.pc = if op = 0x80 then B + 1664 else if op = 0x81 then B + 1688 else if op = 0x82 then B + 1712
      else if op = 0x83 then B + 1748 else if op = 0x84 then B + 1780 else B + 1812)
    (h6 : s.reg 6 = some (b + p)) (h13 : s.reg 13 = some hd) (h7 : s.reg 7 = some ha) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if op == 0x82 then u32 bs (p + 1) else 0
      let ilen := if op == 0x80 then 9 else if op == 0x81 ∨ op == 0x82 then 5 else 1
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => False
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  have hd_lt64 : hd < 2 ^ 64 := by unfold HMAX at hhd; omega
  have ha_lt64 : ha < 2 ^ 64 := by unfold HMAX at hha; omega
  rcases hop with rfl | rfl | rfl | rfl | rfl | rfl
  · -- 0x80 lit
    simp only [↓reduceIte] at hpc
    have hc1 := pb.cLitPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, _⟩ := hc1
    have hc2 := pb.cLitEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f1, f2, _⟩ := hc2
    have hr0 : (B + 1664 + 4) % AArch64Env.W = B + 1668 := modW_small (by omega)
    have hd1 : (hd % 2 ^ 64 + 1) % 2 ^ 64 = hd + 1 := by unfold HMAX at hhd; omega
    xs f0 using hl, h13, hd1, hr0
    refine (chk_spec pb.cChkLit (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 1 < 2 ^ 64)
      (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
    have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
    by_cases hov : HMAX ≤ hd + 1
    · rw [if_pos hov] at hpc1
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x80 < 57),
        HMAX_flat_eq, hov, ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
    · rw [if_neg hov] at hpc1
      have hr1 : (B + 1680 + 4) % AArch64Env.W = B + 1684 := modW_small (by omega)
      have hr2 : (B + 1684 + 664) % AArch64Env.W = B + 2348 := modW_small (by omega)
      xs f1 using hs1_len, hr1
      xs f2 using hr2
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x80 < 57),
        HMAX_flat_eq, hov, ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
        rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, hha⟩
  · -- 0x81 str
    simp only [show ¬ (0x81 : Nat) = 0x80 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cStrPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, _⟩ := hc1
    have hc2 := pb.cStrEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f1, f2, _⟩ := hc2
    have hr0 : (B + 1688 + 4) % AArch64Env.W = B + 1692 := modW_small (by omega)
    have hd2 : (hd % 2 ^ 64 + 2) % 2 ^ 64 = hd + 2 := by unfold HMAX at hhd; omega
    xs f0 using hl, h13, hd2, hr0
    refine (chk_spec pb.cChkStr (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 2 < 2 ^ 64)
      (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
    have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
    by_cases hov : HMAX ≤ hd + 2
    · rw [if_pos hov] at hpc1
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x81 < 57),
        HMAX_flat_eq, hov, ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
    · rw [if_neg hov] at hpc1
      have hr1 : (B + 1704 + 4) % AArch64Env.W = B + 1708 := modW_small (by omega)
      have hr2 : (B + 1708 + 640) % AArch64Env.W = B + 2348 := modW_small (by omega)
      xs f1 using hs1_len, hr1
      xs f2 using hr2
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x81 < 57),
        HMAX_flat_eq, hov, ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
        rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, hha⟩
  · -- 0x82 pick
    rcases hp with hp5 | h | h | h | h <;> try contradiction
    simp only [show ¬ (0x82 : Nat) = 0x80 by decide, show ¬ (0x82 : Nat) = 0x81 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cPickPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, f3, _⟩ := hc1
    have hc2 := pb.cPickEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f4, f5, _⟩ := hc2
    have himm := u32_lt hby (p + 1)
    have hrd := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
    have hr0 : (B + 1712 + 4) % AArch64Env.W = B + 1716 := modW_small (by omega)
    have hr1 : (B + 1716 + 4) % AArch64Env.W = B + 1720 := modW_small (by omega)
    xs f0 using hl, hm, h6, hrd, himm, hr0
    have hcs := cond_sub (show u32 bs (p + 1) < pow2 64 by rw [pow2_64]; omega)
      (show hd < pow2 64 by rw [pow2_64]; exact hd_lt64)
    xs f1 using hl, h13, hr1
    by_cases hge : hd ≤ u32 bs (p + 1)
    · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (hd % 2 ^ 64)).2 2 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hd_lt64, hcs.2.2.1]; simp [hge]
      have hr2 : (B + 1720 + relOff (B + 1720) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f2 using hfl, hr2
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x82 < 57),
        hge, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (hd % 2 ^ 64)).2 2 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hd_lt64, hcs.2.2.1]; simp [hge]
      have hr2 : (B + 1720 + 4) % AArch64Env.W = B + 1724 := modW_small (by omega)
      have hr3 : (B + 1724 + 4) % AArch64Env.W = B + 1728 := modW_small (by omega)
      have hd1 : (hd % 2 ^ 64 + 1) % 2 ^ 64 = hd + 1 := by unfold HMAX at hhd; omega
      xs f2 using hfl, hr2
      xs f3 using hl, h13, hd1, hr3
      refine (chk_spec pb.cChkPick (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 1 < 2 ^ 64)
        (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
      have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      by_cases hov : HMAX ≤ hd + 1
      · rw [if_pos hov] at hpc1
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x82 < 57),
          HMAX_flat_eq, hge, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
      · rw [if_neg hov] at hpc1
        have hr4 : (B + 1740 + 4) % AArch64Env.W = B + 1744 := modW_small (by omega)
        have hr5 : (B + 1744 + 604) % AArch64Env.W = B + 2348 := modW_small (by omega)
        xs f4 using hs1_len, hr4
        xs f5 using hr5
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x82 < 57),
          HMAX_flat_eq, hge, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
          rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, hha⟩
  · -- 0x83 tor
    simp only [show ¬ (0x83 : Nat) = 0x80 by decide, show ¬ (0x83 : Nat) = 0x81 by decide,
      show ¬ (0x83 : Nat) = 0x82 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cTorPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, _⟩ := hc1
    have hc2 := pb.cTorEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f3, f4, _⟩ := hc2
    by_cases hd0 : hd = 0
    · have hr0 : (B + 1748 + relOff (B + 1748) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, hr0
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x83 < 57),
        ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hr0 : (B + 1748 + 4) % AArch64Env.W = B + 1752 := modW_small (by omega)
      have hr1 : (B + 1752 + 4) % AArch64Env.W = B + 1756 := modW_small (by omega)
      have hr2 : (B + 1756 + 4) % AArch64Env.W = B + 1760 := modW_small (by omega)
      have hd_sub : (hd % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = hd - 1 := by unfold HMAX at hhd; omega
      have ha1 : (ha % 2 ^ 64 + 1) % 2 ^ 64 = ha + 1 := by unfold HMAX at hha; omega
      xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, hr0
      xs f1 using hl, h13, hd_sub, hr1
      xs f2 using hl, h7, ha1, hr2
      refine (chk_spec pb.cChkTor (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hha; omega : ha + 1 < 2 ^ 64)
        (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_7⟩ => ?_
      have hs1_13 : s1.reg 13 = some (hd - 1) := by rw [hf1.regs 13 (by decide)]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      by_cases hov : HMAX ≤ ha + 1
      · rw [if_pos hov] at hpc1
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x83 < 57),
          HMAX_flat_eq, show ¬ hd < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
      · rw [if_neg hov] at hpc1
        have hr3 : (B + 1772 + 4) % AArch64Env.W = B + 1776 := modW_small (by omega)
        have hr4 : (B + 1776 + 572) % AArch64Env.W = B + 2348 := modW_small (by omega)
        xs f3 using hs1_len, hr3
        xs f4 using hr4
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x83 < 57),
          HMAX_flat_eq, show ¬ hd < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
          rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, by omega⟩
  · -- 0x84 fromr
    simp only [show ¬ (0x84 : Nat) = 0x80 by decide, show ¬ (0x84 : Nat) = 0x81 by decide,
      show ¬ (0x84 : Nat) = 0x82 by decide, show ¬ (0x84 : Nat) = 0x83 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cFromrPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, _⟩ := hc1
    have hc2 := pb.cFromrEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f3, f4, _⟩ := hc2
    by_cases ha0 : ha = 0
    · have hr0 : (B + 1780 + relOff (B + 1780) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f0 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, hr0
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x84 < 57),
        ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hr0 : (B + 1780 + 4) % AArch64Env.W = B + 1784 := modW_small (by omega)
      have hr1 : (B + 1784 + 4) % AArch64Env.W = B + 1788 := modW_small (by omega)
      have hr2 : (B + 1788 + 4) % AArch64Env.W = B + 1792 := modW_small (by omega)
      have ha_sub : (ha % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = ha - 1 := by unfold HMAX at hha; omega
      have hd1 : (hd % 2 ^ 64 + 1) % 2 ^ 64 = hd + 1 := by unfold HMAX at hhd; omega
      xs f0 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, hr0
      xs f1 using hl, h7, ha_sub, hr1
      xs f2 using hl, h13, hd1, hr2
      refine (chk_spec pb.cChkFromr (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 1 < 2 ^ 64)
        (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
      have hs1_7 : s1.reg 7 = some (ha - 1) := by rw [hf1.regs 7 (by decide)]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      by_cases hov : HMAX ≤ hd + 1
      · rw [if_pos hov] at hpc1
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x84 < 57),
          HMAX_flat_eq, show ¬ ha < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
      · rw [if_neg hov] at hpc1
        have hr3 : (B + 1804 + 4) % AArch64Env.W = B + 1808 := modW_small (by omega)
        have hr4 : (B + 1808 + 540) % AArch64Env.W = B + 2348 := modW_small (by omega)
        xs f3 using hs1_len, hr3
        xs f4 using hr4
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x84 < 57),
          HMAX_flat_eq, show ¬ ha < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
          rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, by omega⟩
  · -- 0x85 rfetch
    simp only [show ¬ (0x85 : Nat) = 0x80 by decide, show ¬ (0x85 : Nat) = 0x81 by decide,
      show ¬ (0x85 : Nat) = 0x82 by decide, show ¬ (0x85 : Nat) = 0x83 by decide,
      show ¬ (0x85 : Nat) = 0x84 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cRfetchPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, _⟩ := hc1
    have hc2 := pb.cRfetchEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f2, f3, _⟩ := hc2
    by_cases ha0 : ha = 0
    · have hr0 : (B + 1812 + relOff (B + 1812) (B + 2708)) % AArch64Env.W = B + 2708 :=
        relOff_target (by omega) (by omega)
      xs f0 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, hr0
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x85 < 57),
        ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hr0 : (B + 1812 + 4) % AArch64Env.W = B + 1816 := modW_small (by omega)
      have hr1 : (B + 1816 + 4) % AArch64Env.W = B + 1820 := modW_small (by omega)
      have hd1 : (hd % 2 ^ 64 + 1) % 2 ^ 64 = hd + 1 := by unfold HMAX at hhd; omega
      xs f0 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, hr0
      xs f1 using hl, h13, hd1, hr1
      refine (chk_spec pb.cChkRfetch (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 1 < 2 ^ 64)
        (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
      have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      by_cases hov : HMAX ≤ hd + 1
      · rw [if_pos hov] at hpc1
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x85 < 57),
          HMAX_flat_eq, show ¬ ha < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
      · rw [if_neg hov] at hpc1
        have hr2 : (B + 1832 + 4) % AArch64Env.W = B + 1836 := modW_small (by omega)
        have hr3 : (B + 1836 + 512) % AArch64Env.W = B + 2348 := modW_small (by omega)
        xs f2 using hs1_len, hr2
        xs f3 using hr3
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x85 < 57),
          HMAX_flat_eq, show ¬ ha < 1 by omega, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
          rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, hha⟩

private theorem hstep_loc_spec {B adrImm R nxt b ws nw wordOff outs locals annOff na p hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hlocs : u32 bs (wordOff + 8) = locals)
    (hp : p + 5 ≤ bs.length ∨ op = 0x8c) (hop : op = 0x86 ∨ op = 0x87)
    (hm : s.mem = m) (hl : s.regs.length = 31)
    (hpc : s.pc = if op = 0x86 then B + 1840 else if op = 0x87 then B + 1880 else if op = 0x88 then B + 1912
      else if op = 0x89 ∨ op = 0x8a then B + 2000 else if op = 0x8b then B + 2096 else B + 2324)
    (h6 : s.reg 6 = some (b + p)) (h7 : s.reg 7 = some ha)
    (h12 : s.reg 12 = some (b + wordOff)) (h13 : s.reg 13 = some hd) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if op == 0x8c then 0 else u32 bs (p + 1)
      let ilen := if op == 0x8c then 1 else 5
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some ilen
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  have hd_lt64 : hd < 2 ^ 64 := by unfold HMAX at hhd; omega
  have hloc_lt := u32_lt hby (wordOff + 8); rw [hlocs] at hloc_lt
  rcases hop with rfl | rfl
  · -- 0x86 locget
    rcases hp with hp5 | h | h <;> try contradiction
    simp only [↓reduceIte] at hpc
    have hc1 := pb.cLocgetPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, f3, f4, _⟩ := hc1
    have hc2 := pb.cLocgetEnd; simp only [CodeAt_cons] at hc2; obtain ⟨f5, f6, _⟩ := hc2
    have himm := u32_lt hby (p + 1)
    have hrd0 := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
    have hrd1 := rdMem_word_field hbuf hsm (by omega : wordOff + 8 + 4 ≤ bs.length)
    rw [hlocs] at hrd1
    xs f0 using hl, hm, h6, hrd0, himm, modW_step hB 1840 4 1844
    xs f1 using hl, hm, h12, hrd1, hloc_lt, modW_step hB 1844 4 1848
    have hcs := cond_sub (show u32 bs (p + 1) < pow2 64 by rw [pow2_64]; omega)
      (show locals < pow2 64 by rw [pow2_64]; omega)
    xs f2 using hl, modW_step hB 1848 4 1852
    by_cases hge : locals ≤ u32 bs (p + 1)
    · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (locals % 2 ^ 64)).2 2 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]; simp [hge]
      xs f3 using hfl, relOff_step hB 1852 2708
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x86 < 57),
        hge, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (locals % 2 ^ 64)).2 2 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]; simp [hge]
      have hd1 : (hd % 2 ^ 64 + 1) % 2 ^ 64 = hd + 1 := by unfold HMAX at hhd; omega
      xs f3 using hfl, modW_step hB 1852 4 1856
      xs f4 using hl, h13, hd1, modW_step hB 1856 4 1860
      refine (chk_spec pb.cChkLocget (by omega) (by omega) (by decide) (by decide) (by unfold HMAX at hhd; omega : hd + 1 < 2 ^ 64)
        (by simp [hl]) rfl (by simp [hl])).trans fun s1 ⟨hf1, hpc1, hs1_len, hs1_13⟩ => ?_
      have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      by_cases hov : HMAX ≤ hd + 1
      · rw [if_pos hov] at hpc1
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x86 < 57),
          HMAX_flat_eq, hge, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf1 (by decide)), hpc1⟩
      · rw [if_neg hov] at hpc1
        xs f5 using hs1_len, modW_step hB 1872 4 1876
        xs f6 using modW_step hB 1876 472 2348
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x86 < 57),
          HMAX_flat_eq, hge, hov, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
          rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len], by omega, hha⟩
  · -- 0x87 locset
    rcases hp with hp5 | h | h <;> try contradiction
    simp only [show ¬ (0x87 : Nat) = 0x86 by decide, ↓reduceIte] at hpc
    have hc := pb.cLocset; simp only [CodeAt_cons] at hc; obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, _⟩ := hc
    by_cases hd0 : hd = 0
    · xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, relOff_step hB 1880 2708
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x87 < 57),
        ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have himm := u32_lt hby (p + 1)
      have hrd1 := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
      have hrd2 := rdMem_word_field hbuf hsm (by omega : wordOff + 8 + 4 ≤ bs.length)
      rw [hlocs] at hrd2
      xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, modW_step hB 1880 4 1884
      xs f1 using hl, hm, h6, hrd1, himm, modW_step hB 1884 4 1888
      xs f2 using hl, hm, h12, hrd2, hloc_lt, modW_step hB 1888 4 1892
      have hcs := cond_sub (show u32 bs (p + 1) < pow2 64 by rw [pow2_64]; omega)
        (show locals < pow2 64 by rw [pow2_64]; omega)
      xs f3 using hl, modW_step hB 1892 4 1896
      by_cases hge : locals ≤ u32 bs (p + 1)
      · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (locals % 2 ^ 64)).2 2 = some true := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]; simp [hge]
        xs f4 using hfl, relOff_step hB 1896 2708
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x87 < 57),
          show ¬ hd < 1 by omega, hge, ↓reduceIte]
        exact Reaches.refl ⟨by frame_tac, rfl⟩
      · have hfl : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (locals % 2 ^ 64)).2 2 = some false := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.2.2.1]; simp [hge]
        have hd_sub : (hd % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = hd - 1 := by unfold HMAX at hhd; omega
        xs f4 using hfl, modW_step hB 1896 4 1900
        xs f5 using hl, h13, hd_sub, modW_step hB 1900 4 1904
        xs f6 using hl, modW_step hB 1904 4 1908
        xs f7 using modW_step hB 1908 440 2348
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x87 < 57),
          show ¬ hd < 1 by omega, hge, ↓reduceIte]
        exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl], hm, by simp [hl], by simp [h7], by simp [hl],
          by omega, hha⟩

private theorem hstep_jmp_spec {B adrImm R nxt b ws nw wordOff outs locals annOff na p hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length)
    (hp : p + 5 ≤ bs.length ∨ op = 0x8c) (hop : op = 0x88 ∨ op = 0x89 ∨ op = 0x8a)
    (hm : s.mem = m) (hl : s.regs.length = 31)
    (hpc : s.pc = if op = 0x86 then B + 1840 else if op = 0x87 then B + 1880 else if op = 0x88 then B + 1912
      else if op = 0x89 ∨ op = 0x8a then B + 2000 else if op = 0x8b then B + 2096 else B + 2324)
    (h6 : s.reg 6 = some (b + p)) (h7 : s.reg 7 = some ha)
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some hd) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if op == 0x8c then 0 else u32 bs (p + 1)
      let ilen := if op == 0x8c then 1 else 5
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some ilen
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  have hd_lt64 : hd < 2 ^ 64 := by unfold HMAX at hhd; omega
  have ha_lt64 : ha < 2 ^ 64 := by unfold HMAX at hha; omega
  rcases hop with rfl | h89a
  · -- 0x88 jmp
    rcases hp with hp5 | h <;> try contradiction
    simp only [show ¬ (0x88 : Nat) = 0x86 by decide, show ¬ (0x88 : Nat) = 0x87 by decide, ↓reduceIte] at hpc
    have hc1 := pb.cJmpPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, _⟩ := hc1
    have hc2 := pb.cJmpEnd; simp only [CodeAt_cons] at hc2
    obtain ⟨f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, _⟩ := hc2
    have himm := u32_lt hby (p + 1)
    have hrd0 := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
    xs f0 using hl, hm, h6, hrd0, himm, modW_step hB 1912 4 1916
    refine (lookup_spec pb.cLuJmp (by omega) (Or.inr rfl) (by simpa [hm] using hbuf) hby hsm hw hna hann
      (by omega : u32 bs (p + 1) < 2 ^ 64) (by simp [hl]) rfl (by simp [h12]) (by simp [h9])
      (by simp [hl])).trans
      fun s1 ⟨hf1, hpc1, hs1_11, hs1_len, hlu⟩ => ?_
    have hs1_13 : s1.reg 13 = some hd := by rw [hf1.regs 13 (by decide)]; simp [h13]
    have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
    cases hlu_eq : lookup bs annOff na (u32 bs (p + 1)) with
    | none =>
      rw [hlu_eq] at hlu
      have hcs := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      xs f1 using hs1_len, hlu, hs1_11, modW_step hB 1960 4 1964
      have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
          ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs.1]; simp
      xs f2 using hfl2, relOff_step hB 1964 2708
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x88 < 57),
        hlu_eq, ↓reduceIte]
      exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
    | some p' =>
      rcases p' with ⟨ad, aa⟩
      rw [hlu_eq] at hlu
      obtain ⟨q, _, hq12, hs1_10, hqne, rfl, rfl⟩ := hlu
      have had := u32_lt hby (q + 4)
      have haa := u32_lt hby (q + 8)
      have hcs0 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      xs f1 using hs1_len, hs1_10, hs1_11, modW_step hB 1960 4 1964
      have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
          ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
        rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
          Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp [hqne]
      have hrd3 := rdMem_word_field (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
      xs f2 using hfl2, modW_step hB 1964 4 1968
      xs f3 using hs1_len, hs1_10, hrd3, had, modW_step hB 1968 4 1972
      have hcs1 := cond_sub (show u32 bs (q + 4) < pow2 64 by rw [pow2_64]; omega)
        (show hd < pow2 64 by rw [pow2_64]; exact hd_lt64)
      xs f4 using hs1_len, hs1_13, modW_step hB 1972 4 1976
      by_cases heqd : u32 bs (q + 4) = hd
      · have hfl5 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (hd % 2 ^ 64)).2 1 = some false := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hd_lt64, hcs1.2.1]; simp [heqd]
        have hrd6 := rdMem_word_field (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
        xs f5 using hfl5, modW_step hB 1976 4 1980
        xs f6 using hs1_len, hs1_10, hrd6, haa, modW_step hB 1980 4 1984
        have hcs2 := cond_sub (show u32 bs (q + 8) < pow2 64 by rw [pow2_64]; omega)
          (show ha < pow2 64 by rw [pow2_64]; exact ha_lt64)
        xs f7 using hs1_len, hs1_7, modW_step hB 1984 4 1988
        by_cases heqa : u32 bs (q + 8) = ha
        · have hfl8 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (ha % 2 ^ 64)).2 1 = some false := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt ha_lt64, hcs2.2.1]; simp [heqa]
          xs f8 using hfl8, modW_step hB 1988 4 1992
          xs f9 using hs1_len, modW_step hB 1992 4 1996
          xs f10 using modW_step hB 1996 452 2448
          simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x88 < 57),
            hlu_eq, heqd, heqa, beq_self_eq_true, Bool.and_self, ↓reduceIte]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
            rfl, by simp [hs1_len], hs1_mem, by simp [hs1_len]⟩
        · have hfl8 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (ha % 2 ^ 64)).2 1 = some true := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt ha_lt64, hcs2.2.1]; simp [heqa]
          xs f8 using hfl8, relOff_step hB 1988 2708
          simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x88 < 57),
            hlu_eq, heqd, beq_self_eq_true, beq_false_of_ne heqa, Bool.and_false, ↓reduceIte]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
      · have hfl5 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (hd % 2 ^ 64)).2 1 = some true := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hd_lt64, hcs1.2.1]; simp [heqd]
        xs f5 using hfl5, relOff_step hB 1976 2708
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x88 < 57),
          hlu_eq, beq_false_of_ne heqd, Bool.false_and, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
  · -- 0x89 jz / 0x8a jnz (identical handler at B + 2000)
    have hp5 : p + 5 ≤ bs.length := by rcases hp with h | h; exact h; rcases h89a with rfl | rfl <;> contradiction
    have hpc2000 : s.pc = B + 2000 := by
      rcases h89a with rfl | rfl <;>
        simpa [show ¬ (0x89 : Nat) = 0x86 by decide, show ¬ (0x89 : Nat) = 0x87 by decide,
          show ¬ (0x89 : Nat) = 0x88 by decide, show ¬ (0x8a : Nat) = 0x86 by decide,
          show ¬ (0x8a : Nat) = 0x87 by decide, show ¬ (0x8a : Nat) = 0x88 by decide] using hpc
    clear hpc; have hpc : s.pc = B + 2000 := hpc2000; clear hpc2000
    have hstep_eq : op ≠ 0x8c ∧
        hstepF bs nw ws outs locals annOff na hd ha op (u32 bs (p + 1)) =
        if hd < 1 then .reject
        else match lookup bs annOff na (u32 bs (p + 1)) with
          | some (ad, aa) => if ad == hd - 1 && aa == ha then .next (hd - 1) ha else .reject
          | none => .reject := by
      rcases h89a with rfl | rfl
      · exact ⟨by decide, by simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x89 < 57), ↓reduceIte]; rfl⟩
      · exact ⟨by decide, by simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8a < 57), ↓reduceIte]; rfl⟩
    have hc1 := pb.cJzPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, _⟩ := hc1
    have hc2 := pb.cJzEnd; simp only [CodeAt_cons] at hc2
    obtain ⟨f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, _⟩ := hc2
    by_cases hd0 : hd = 0
    · have hstep0 : hstepF bs nw ws outs locals annOff na 0 ha op (u32 bs (p + 1)) = .reject := by
        rw [← hd0, hstep_eq.2, hd0]; rfl
      xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, relOff_step hB 2000 2708
      simp only [hstep_eq.1, ↓reduceIte, hstep0]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have himm := u32_lt hby (p + 1)
      have hrd2 := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
      have hd_sub : (hd % 2 ^ 64 + 2 ^ 64 - 1) % 2 ^ 64 = hd - 1 := by unfold HMAX at hhd; omega
      xs f0 using h13, Nat.mod_eq_of_lt hd_lt64, hd0, modW_step hB 2000 4 2004
      xs f1 using hl, h13, hd_sub, modW_step hB 2004 4 2008
      xs f2 using hl, hm, h6, hrd2, himm, modW_step hB 2008 4 2012
      refine (lookup_spec pb.cLuJz (by omega) (Or.inr rfl) (by simpa [hm] using hbuf) hby hsm hw hna hann
        (by omega : u32 bs (p + 1) < 2 ^ 64) (by simp [hl]) rfl (by simp [h12]) (by simp [h9])
        (by simp [hl])).trans
        fun s1 ⟨hf1, hpc1, hs1_11, hs1_len, hlu⟩ => ?_
      have hs1_13 : s1.reg 13 = some (hd - 1) := by rw [hf1.regs 13 (by decide)]; simp [hl]
      have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      cases hlu_eq : lookup bs annOff na (u32 bs (p + 1)) with
      | none =>
        rw [hlu_eq] at hlu
        have hcs := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f3 using hs1_len, hlu, hs1_11, modW_step hB 2056 4 2060
        have hfl4 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs.1]; simp
        xs f4 using hfl4, relOff_step hB 2060 2708
        simp only [hstep_eq.1, ↓reduceIte, hstep_eq.2, show ¬ hd < 1 by omega, hlu_eq]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
      | some p' =>
        rcases p' with ⟨ad, aa⟩
        rw [hlu_eq] at hlu
        obtain ⟨q, _, hq12, hs1_10, hqne, rfl, rfl⟩ := hlu
        have had := u32_lt hby (q + 4)
        have haa := u32_lt hby (q + 8)
        have hcs0 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f3 using hs1_len, hs1_10, hs1_11, modW_step hB 2056 4 2060
        have hfl4 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp [hqne]
        have hrd5 := rdMem_word_field (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
        xs f4 using hfl4, modW_step hB 2060 4 2064
        xs f5 using hs1_len, hs1_10, hrd5, had, modW_step hB 2064 4 2068
        have hcs1 := cond_sub (show u32 bs (q + 4) < pow2 64 by rw [pow2_64]; omega)
          (show hd - 1 < pow2 64 by rw [pow2_64]; omega)
        xs f6 using hs1_len, hs1_13, modW_step hB 2068 4 2072
        by_cases heqd : u32 bs (q + 4) = hd - 1
        · have hfl7 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) ((hd - 1) % 2 ^ 64)).2 1 = some false := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs1.2.1]; simp [heqd]
          have hrd8 := rdMem_word_field (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
          xs f7 using hfl7, modW_step hB 2072 4 2076
          xs f8 using hs1_len, hs1_10, hrd8, haa, modW_step hB 2076 4 2080
          have hcs2 := cond_sub (show u32 bs (q + 8) < pow2 64 by rw [pow2_64]; omega)
            (show ha < pow2 64 by rw [pow2_64]; exact ha_lt64)
          xs f9 using hs1_len, hs1_7, modW_step hB 2080 4 2084
          by_cases heqa : u32 bs (q + 8) = ha
          · have hfl10 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (ha % 2 ^ 64)).2 1 = some false := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt ha_lt64, hcs2.2.1]; simp [heqa]
            xs f10 using hfl10, modW_step hB 2084 4 2088
            xs f11 using hs1_len, modW_step hB 2088 4 2092
            xs f12 using modW_step hB 2092 256 2348
            simp only [hstep_eq.1, ↓reduceIte, hstep_eq.2, show ¬ hd < 1 by omega,
              hlu_eq, heqd, heqa, beq_self_eq_true, Bool.and_self]
            exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)),
              rfl, by simp [hs1_len], hs1_mem, by simp [hs1_13], by simp [hs1_7], by simp [hs1_len],
              by omega, hha⟩
          · have hfl10 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (ha % 2 ^ 64)).2 1 = some true := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt ha_lt64, hcs2.2.1]; simp [heqa]
            xs f10 using hfl10, relOff_step hB 2084 2708
            simp only [hstep_eq.1, ↓reduceIte, hstep_eq.2, show ¬ hd < 1 by omega,
              hlu_eq, heqd, beq_self_eq_true, beq_false_of_ne heqa, Bool.and_false]
            exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
        · have hfl7 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) ((hd - 1) % 2 ^ 64)).2 1 = some true := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs1.2.1]; simp [heqd]
          xs f7 using hfl7, relOff_step hB 2072 2708
          simp only [hstep_eq.1, ↓reduceIte, hstep_eq.2, show ¬ hd < 1 by omega,
            hlu_eq, beq_false_of_ne heqd, Bool.false_and]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩

private theorem hstep_call_ret_spec {B adrImm R nxt b ws nw wordOff outs locals annOff na p hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (houts : u32 bs (wordOff + 4) = outs)
    (hsw : skipWords bs nw ws = some bs.length) (hws_le : ws ≤ bs.length) (hnw : nw < 2 ^ 32)
    (hp : p + 5 ≤ bs.length ∨ op = 0x8c) (hop : op = 0x8b ∨ op = 0x8c)
    (hm : s.mem = m) (hl : s.regs.length = 31)
    (hpc : s.pc = if op = 0x86 then B + 1840 else if op = 0x87 then B + 1880 else if op = 0x88 then B + 1912
      else if op = 0x89 ∨ op = 0x8a then B + 2000 else if op = 0x8b then B + 2096 else B + 2324)
    (h1 : s.reg 1 = some nw) (h6 : s.reg 6 = some (b + p)) (h7 : s.reg 7 = some ha)
    (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some hd) (h24 : s.reg 24 = some (b + bs.length)) (h25 : s.reg 25 = some (b + ws)) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if op == 0x8c then 0 else u32 bs (p + 1)
      let ilen := if op == 0x8c then 1 else 5
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some ilen
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  have hd_lt64 : hd < 2 ^ 64 := by unfold HMAX at hhd; omega
  have ha_lt64 : ha < 2 ^ 64 := by unfold HMAX at hha; omega
  have hout_lt := u32_lt hby (wordOff + 4); rw [houts] at hout_lt
  rcases hop with rfl | rfl
  · -- 0x8b call
    rcases hp with hp5 | h <;> try contradiction
    simp only [show ¬ (0x8b : Nat) = 0x86 by decide, show ¬ (0x8b : Nat) = 0x87 by decide,
      show ¬ (0x8b : Nat) = 0x88 by decide, show ¬ ((0x8b : Nat) = 0x89 ∨ (0x8b : Nat) = 0x8a) by decide, ↓reduceIte] at hpc
    have hc1 := pb.cCallPre; simp only [CodeAt_cons] at hc1; obtain ⟨f0, f1, f2, f3, _⟩ := hc1
    have hc2 := pb.cCallMid; simp only [CodeAt_cons] at hc2; obtain ⟨f4, f5, f6, f7, f8, f9, _⟩ := hc2
    have hc3 := pb.cCallEnd; simp only [CodeAt_cons] at hc3; obtain ⟨f10, f11, _⟩ := hc3
    have himm := u32_lt hby (p + 1)
    have hrd0 := rdMem_word_field hbuf hsm (by omega : p + 1 + 4 ≤ bs.length)
    xs f0 using hl, hm, h6, hrd0, himm, modW_step hB 2096 4 2100
    have hcs0 := cond_sub (show u32 bs (p + 1) < pow2 64 by rw [pow2_64]; omega)
      (show nw < pow2 64 by rw [pow2_64]; omega)
    xs f1 using hl, h1, modW_step hB 2100 4 2104
    by_cases hge : nw ≤ u32 bs (p + 1)
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (nw % 2 ^ 64)).2 2 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs0.2.2.1]; simp [hge]
      xs f2 using hfl2, relOff_step hB 2104 2708
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8b < 57),
        hge, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (p + 1) % 2 ^ 64) (nw % 2 ^ 64)).2 2 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs0.2.2.1]; simp [hge]
      have hws_m : (b + ws) % 2 ^ 64 = b + ws := Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega)
      xs f2 using hfl2, modW_step hB 2104 4 2108
      xs f3 using hl, h25, hws_m, modW_step hB 2108 4 2112
      obtain ⟨wOff, cw, hsw_w, hrw_w, hwa_w, h20_w, hle_w, _⟩ :=
        skipWords_wordAt nw (u32 bs (p + 1)) ws bs.length hsw (by omega)
      have hcw_ins_outs : u32 bs wOff = cw.ins ∧ u32 bs (wOff + 4) = cw.outs := by
        unfold readWordAt at hrw_w
        split at hrw_w <;> try cases hrw_w
        simp only [bind, Option.bind] at hrw_w
        simp only [rdU32_eq_some_u32 hby (by omega : wOff + 4 ≤ bs.length),
          rdU32_eq_some_u32 hby (by omega : wOff + 4 + 4 ≤ bs.length)] at hrw_w
        cases hrd8 : rdU32 bs (wOff + 8) with
        | none => simp only [hrd8] at hrw_w; cases hrw_w
        | some _ =>
          simp only [hrd8] at hrw_w
          cases hrd12 : rdU32 bs (wOff + 12) with
          | none => simp only [hrd12] at hrw_w; cases hrw_w
          | some nc =>
            simp only [hrd12] at hrw_w
            cases hrd16 : rdU32 bs (wOff + 16) with
            | none => simp only [hrd16] at hrw_w; cases hrw_w
            | some _ =>
              simp only [hrd16] at hrw_w
              cases hsc : skipCode bs nc (wOff + 20) with
              | none => simp only [hsc] at hrw_w; cases hrw_w
              | some _ =>
                simp only [hsc] at hrw_w
                split at hrw_w <;> cases hrw_w; exact ⟨rfl, rfl⟩
      have hcins := u32_lt hby wOff; rw [hcw_ins_outs.1] at hcins
      have hcouts := u32_lt hby (wOff + 4); rw [hcw_ins_outs.2] at hcouts
      refine (skipwords_spec pb.cSwCall (by omega) (by omega) (by simpa [hm] using hbuf) hby hsm (u32 bs (p + 1)) ws _
        (by simp [hm]) (by simp [hl]) rfl (by simp [hl]) (by simp [hl]) (by simp [h24]) hws_le (by omega)).trans
        fun s1 ⟨hf1, hsw1⟩ => ?_
      rw [hsw_w] at hsw1
      obtain ⟨_, hs1_10, _, hw_le⟩ := hsw1
      have hs1_len : s1.regs.length = 31 := by rw [hf1.len]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; simp [hm]
      have hs1_13 : s1.reg 13 = some hd := by rw [hf1.regs 13 (by decide)]; simp [h13]
      have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hrd4 := rdMem_word_field (m := s1.mem) (wordOff := wOff) (fieldOff := 0) (by simpa [hs1_mem] using hbuf) hsm (by omega : wOff + 0 + 4 ≤ bs.length)
      rw [show u32 bs (wOff + 0) = cw.ins from by simpa using hcw_ins_outs.1] at hrd4
      xs f4 using hs1_len, hs1_10, hrd4, hcins, modW_step hB 2280 4 2284
      have hcs1 := cond_sub (show hd < pow2 64 by rw [pow2_64]; exact hd_lt64)
        (show cw.ins < pow2 64 by rw [pow2_64]; omega)
      xs f5 using hs1_len, hs1_13, modW_step hB 2284 4 2288
      by_cases hlt : hd < cw.ins
      · have hfl6 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (cw.ins % 2 ^ 64)).2 3 = some true := by
          rw [Nat.mod_eq_of_lt hd_lt64, Nat.mod_eq_of_lt (by omega), hcs1.2.2.2.1]; simp [hlt]
        xs f6 using hfl6, relOff_step hB 2288 2708
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8b < 57),
          hge, hwa_w, hlt, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
      · have hfl6 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (cw.ins % 2 ^ 64)).2 3 = some false := by
          rw [Nat.mod_eq_of_lt hd_lt64, Nat.mod_eq_of_lt (by omega), hcs1.2.2.2.1]; simp [hlt]
        have hrd8 := rdMem_word_field (m := s1.mem) (wordOff := wOff) (fieldOff := 4) (by simpa [hs1_mem] using hbuf) hsm (by omega : wOff + 4 + 4 ≤ bs.length)
        rw [hcw_ins_outs.2] at hrd8
        have hd_sub : (hd % 2 ^ 64 + 2 ^ 64 - cw.ins % 2 ^ 64) % 2 ^ 64 = hd - cw.ins := by
          unfold HMAX at hhd; omega
        have hd_new : ((hd - cw.ins) % 2 ^ 64 + cw.outs % 2 ^ 64) % 2 ^ 64 = hd - cw.ins + cw.outs := by
          unfold HMAX at hhd; omega
        xs f6 using hfl6, modW_step hB 2288 4 2292
        xs f7 using hs1_len, hs1_13, hd_sub, modW_step hB 2292 4 2296
        xs f8 using hs1_len, hs1_10, hrd8, hcouts, modW_step hB 2296 4 2300
        xs f9 using hs1_len, hd_new, modW_step hB 2300 4 2304
        refine (chk_spec pb.cChkCall (by omega) (by omega) (by decide) (by decide)
          (by unfold HMAX at hhd; omega : hd - cw.ins + cw.outs < 2 ^ 64) (by simp [hs1_len]) rfl
          (by simp [hs1_len])).trans fun s2 ⟨hf2, hpc2, hs2_len, hs2_13⟩ => ?_
        have hs2_7 : s2.reg 7 = some ha := by rw [hf2.regs 7 (by decide)]; simp [hs1_7]
        have hs2_mem : s2.mem = m := by rw [hf2.mem]; simp [hs1_mem]
        by_cases hov : HMAX ≤ hd - cw.ins + cw.outs
        · rw [if_pos hov] at hpc2
          simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8b < 57),
            HMAX_flat_eq, hge, hwa_w, hlt, hov, ↓reduceIte]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
            (Frame.trans_same (by frame_tac) (Frame.mono hf2 (by decide)))), hpc2⟩
        · rw [if_neg hov] at hpc2
          xs f10 using hs2_len, modW_step hB 2316 4 2320
          xs f11 using modW_step hB 2320 28 2348
          simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8b < 57),
            HMAX_flat_eq, hge, hwa_w, hlt, hov, ↓reduceIte]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
            (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf2 (by decide)) (by frame_tac)))),
            rfl, by simp [hs2_len], hs2_mem, by simp [hs2_13], by simp [hs2_7], by simp [hs2_len], by omega, hha⟩
  · -- 0x8c ret
    simp only [show ¬ (0x8c : Nat) = 0x86 by decide, show ¬ (0x8c : Nat) = 0x87 by decide,
      show ¬ (0x8c : Nat) = 0x88 by decide, show ¬ ((0x8c : Nat) = 0x89 ∨ (0x8c : Nat) = 0x8a) by decide,
      show ¬ (0x8c : Nat) = 0x8b by decide, ↓reduceIte] at hpc
    have hc := pb.cRet; simp only [CodeAt_cons] at hc; obtain ⟨f0, f1, f2, f3, f4, f5, _⟩ := hc
    have hrd0 := rdMem_word_field hbuf hsm (by omega : wordOff + 4 + 4 ≤ bs.length)
    rw [houts] at hrd0
    xs f0 using hl, hm, h12, hrd0, hout_lt, modW_step hB 2324 4 2328
    have hcs := cond_sub (show hd < pow2 64 by rw [pow2_64]; exact hd_lt64)
      (show outs < pow2 64 by rw [pow2_64]; omega)
    xs f1 using hl, h13, modW_step hB 2328 4 2332
    by_cases heqo : hd = outs
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (outs % 2 ^ 64)).2 1 = some false := by
        rw [Nat.mod_eq_of_lt hd_lt64, Nat.mod_eq_of_lt (by omega), hcs.2.1]; simp [heqo]
      xs f2 using hfl2, modW_step hB 2332 4 2336
      by_cases ha0 : ha = 0
      · xs f3 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, modW_step hB 2336 4 2340
        xs f4 using hl, modW_step hB 2340 4 2344
        xs f5 using modW_step hB 2344 104 2448
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8c < 57),
          heqo, beq_self_eq_true, Bool.and_self, ↓reduceIte]
        exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl], hm, by simp [hl]⟩
      · xs f3 using h7, Nat.mod_eq_of_lt ha_lt64, ha0, relOff_step hB 2336 2708
        simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8c < 57),
          heqo, beq_self_eq_true, beq_false_of_ne ha0, Bool.and_false, ↓reduceIte]
        exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl2 : AArch64Phys.cond (addSubOp true 64 (hd % 2 ^ 64) (outs % 2 ^ 64)).2 1 = some true := by
        rw [Nat.mod_eq_of_lt hd_lt64, Nat.mod_eq_of_lt (by omega), hcs.2.1]; simp [heqo]
      xs f2 using hfl2, relOff_step hB 2332 2708
      simp (config := { decide := true }) only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8c < 57),
        beq_false_of_ne heqo, Bool.false_and, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩

theorem hstep_loc_ctl_spec {B adrImm R nxt b ws nw wordOff outs locals annOff na p hd ha op : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length)
    (houts : u32 bs (wordOff + 4) = outs) (hlocs : u32 bs (wordOff + 8) = locals)
    (hsw : skipWords bs nw ws = some bs.length) (hws_le : ws ≤ bs.length)
    (hnw : nw < 2 ^ 32)
    (hp : p + 5 ≤ bs.length ∨ op = 0x8c)
    (hop : op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨ op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c)
    (hm : s.mem = m) (hl : s.regs.length = 31)
    (hpc : s.pc = if op = 0x86 then B + 1840 else if op = 0x87 then B + 1880 else if op = 0x88 then B + 1912
      else if op = 0x89 ∨ op = 0x8a then B + 2000 else if op = 0x8b then B + 2096 else B + 2324)
    (h1 : s.reg 1 = some nw) (h6 : s.reg 6 = some (b + p)) (h7 : s.reg 7 = some ha)
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some hd) (h24 : s.reg 24 = some (b + bs.length)) (h25 : s.reg 25 = some (b + ws)) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if op == 0x8c then 0 else u32 bs (p + 1)
      let ilen := if op == 0x8c then 1 else 5
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some ilen
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  rcases hop with h | h | h | h | h | h | h
  · exact hstep_loc_spec pb hB hbuf hby hsm hhd hha hw hlocs hp (Or.inl h) hm hl hpc h6 h7 h12 h13
  · exact hstep_loc_spec pb hB hbuf hby hsm hhd hha hw hlocs hp (Or.inr h) hm hl hpc h6 h7 h12 h13
  · exact hstep_jmp_spec pb hB hbuf hby hsm hhd hha hw hna hann hp (Or.inl h) hm hl hpc h6 h7 h9 h12 h13
  · exact hstep_jmp_spec pb hB hbuf hby hsm hhd hha hw hna hann hp (Or.inr (Or.inl h)) hm hl hpc h6 h7 h9 h12 h13
  · exact hstep_jmp_spec pb hB hbuf hby hsm hhd hha hw hna hann hp (Or.inr (Or.inr h)) hm hl hpc h6 h7 h9 h12 h13
  · exact hstep_call_ret_spec pb hB hbuf hby hsm hhd hha hw houts hsw hws_le hnw hp (Or.inl h) hm hl hpc h1 h6 h7 h12 h13 h24 h25
  · exact hstep_call_ret_spec pb hB hbuf hby hsm hhd hha hw houts hsw hws_le hnw hp (Or.inr h) hm hl hpc h1 h6 h7 h12 h13 h24 h25

end SporosCheckArm64Bin

