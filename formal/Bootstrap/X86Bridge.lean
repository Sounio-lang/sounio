/-
Forward simulation from the seed machine model `X86Min` to the stage-1
interpreter model `X86Int` + `X86Dec`, over their shared sub-ISA.

Contents:
1. `CommonIns`, `CommonDec`, and `toIntIns`: the overlapping x86-64 encodings
   accepted by both `X86Min.decode` and `X86Dec.decode`.
2. `decode_agree`: at any canonical address, `X86Min.decode` and `X86Dec.decode`
   agree on length and decoded instruction.
3. `Sim`: simulation invariant relating `X86Min.St` and `X86Int.St` (canonical
   `rip`, register refinement with defined non-`r11` registers in `X86Int`, flag
   refinement, 32-byte writable stack window, and exact stdin/stdout alignment).
4. `step_sim`, `run_sim`, and `prun_sim`: single-step and arbitrary-fuel
   forward simulation, lifted to the fault-free branch of `X86Int.prun` via
   `X86Int.faultFree_prun`.
5. `seed_to_stage1_bridge_witness`: kernel-checked (`decide +kernel`) end-to-end
   handoff where a common-ISA loader runs in `X86Min.exec`, emits a stage-1
   program using `X86Dec`-only instructions, both models agree on the loader
   execution, and the emitted bytes load and run via `X86Dec.imageOf` and
   `X86Env.initStack`.
-/
import Bootstrap.X86Min
import Bootstrap.X86Dec
import Bootstrap.X86Int
import Bootstrap.X86Env
import Bootstrap.Refine

namespace X86Bridge

open X86Env

/-! ### Arithmetic and bitwise helpers -/

theorem pow2_64_eq_W : X86Min.pow2 64 = W := rfl
theorem pow2_32_eq : X86Min.pow2 32 = 2 ^ 32 := rfl
theorem pow2_8_eq : X86Min.pow2 8 = 256 := rfl

theorem msb_64_eq (v : Nat) : X86Min.msb 64 v = X86Int.msb v := rfl

theorem sx8_eq (v : Nat) : X86Dec.sx8 v = X86Min.sx8 v := rfl
theorem sx32_eq (v : Nat) : X86Dec.sx32 v = X86Min.sx32 v := rfl

theorem leB_eq_le (b : Nat → Option Nat) (off n : Nat) :
    X86Dec.leB b off n = X86Min.le b off n := by
  induction n generalizing off with
  | zero => rfl
  | succ n ih => simp [X86Dec.leB, X86Min.le, ih]

theorem sx8_lt_W {v : Nat} (hv : v < 256) : X86Min.sx8 v < W := by
  unfold X86Min.sx8 W
  rw [pow2_64_eq_W]
  unfold W
  split <;> omega

theorem sx32_lt_W {v : Nat} (hv : v < 2 ^ 32) : X86Min.sx32 v < W := by
  unfold X86Min.sx32 W
  rw [pow2_64_eq_W, pow2_32_eq]
  unfold W
  split <;> omega

theorem land_lt_W {a b : Nat} (ha : a < W) : Nat.land a b < W :=
  Nat.lt_of_le_of_lt Nat.and_le_left ha

theorem lor_lt_W {a b : Nat} (ha : a < W) (hb : b < W) : Nat.lor a b < W :=
  Nat.or_lt_two_pow ha hb

theorem xor_lt_W {a b : Nat} (ha : a < W) (hb : b < W) : Nat.xor a b < W :=
  Nat.xor_lt_two_pow ha hb

theorem mod_add_mod_W (a len rel : Nat) : ((a + len) % W + rel) % W = (a + len + rel) % W := by
  rw [Nat.add_mod ((a + len) % W) rel W, Nat.mod_mod, ← Nat.add_mod]

/-! ### Deliverable 1: Shared Sub-ISA (`CommonIns`, `CommonDec`, `toIntIns`) -/

/-- Supported condition codes in both `X86Min.cond` and `X86Int.cond` (non-parity). -/
def supportedCC (cc : Nat) : Bool := cc < 16 && cc / 2 != 5

/-- Predicate on `X86Min.Ins` identifying instructions in the shared sub-ISA
between `X86Min` and `X86Dec`/`X86Int`. -/
def commonInsB : X86Min.Ins → Bool
  | .movImm d imm => d < 16 && imm < 2 ^ 32
  | .rr op w d src =>
    d < 16 && src < 16 &&
    match op, w with
    | .xor, 64 => true
    | .xor, 32 => d == src
    | .or, 64 => true
    | .test, 64 => true
    | .mov, 64 => true
    | .mov, 32 => true
    | _, _ => false
  | .ri op w d imm => op == .cmp && w == 64 && d < 16 && imm < W
  | .push imm => imm < W
  | .pop d => d < 16
  | .ldRsp8 d => d < 16
  | .stRsp8 src => src < 16
  | .jcc cc rel => supportedCC cc && rel < W
  | .jmp rel => rel < W
  | .syscall => true
  | .shlI _ _ _ => false
  | .inc _ _ => false

/-- Shared sub-ISA predicate on `X86Min.Ins`. -/
def CommonIns (ins : X86Min.Ins) : Prop := commonInsB ins = true

instance (ins : X86Min.Ins) : Decidable (CommonIns ins) :=
  inferInstanceAs (Decidable (commonInsB ins = true))

/-- Translate a common `X86Min.Ins` decoded at address `a` of length `len` to the
corresponding `X86Int.Ins` (with branch/jump targets resolved against `a + len`). -/
def toIntIns (a len : Nat) : X86Min.Ins → X86Int.Ins
  | .movImm d imm => .movI d imm
  | .rr .xor _ d src => .alu .xor d src
  | .rr .or _ d src => .alu .or d src
  | .rr .test _ d src => .test d src
  | .rr .mov 64 d src => .mov d src
  | .rr .mov _ d src => .mov32 d src
  | .rr _ _ d src => .mov d src
  | .ri _ _ d imm => .cmpI d imm
  | .push imm => .pushI imm
  | .pop d => .pop d
  | .ldRsp8 d => .ld8 d 4 0
  | .stRsp8 src => .st8 4 0 src
  | .jcc cc rel => .jcc cc ((a + len + rel) % W)
  | .jmp rel => .jmp ((a + len + rel) % W)
  | .syscall => .syscall
  | .shlI _ _ _ => .syscall
  | .inc _ _ => .syscall

/-- Helper checking that `n` bytes starting at `off` in byte source `b` are present and `< 256`. -/
def leBytesLt256 (b : Nat → Option Nat) (off : Nat) : Nat → Bool
  | 0 => true
  | n + 1 =>
    match b off with
    | some v => v < 256 && leBytesLt256 b (off + 1) n
    | none => false

def decHasRex (b0 : Nat) : Bool := 0x40 ≤ b0 && b0 ≤ 0x4f
def decRex (b0 : Nat) : Nat := if decHasRex b0 then b0 - 0x40 else 0
def decP (b0 : Nat) : Nat := if decHasRex b0 then 1 else 0
def decW (b0 : Nat) : Bool := decRex b0 / 8 % 2 == 1
def decR (b0 : Nat) : Nat := (decRex b0 / 4 % 2) * 8
def decX (b0 : Nat) : Bool := decRex b0 / 2 % 2 == 1
def decB (b0 : Nat) : Nat := (decRex b0 % 2) * 8

/-- Boolean recognizer for encodings accepted by both `X86Min.decode` and `X86Dec.decode`
with well-formed byte values (`< 256`). -/
def commonDecB (b : Nat → Option Nat) : Bool :=
  match b 0 with
  | none => false
  | some b0 =>
    let hasRex := decHasRex b0
    let p := decP b0
    let w := decW b0
    let r := decR b0
    let x := decX b0
    let bb := decB b0
    match b p with
    | none => false
    | some o =>
      if 0xb8 ≤ o && o ≤ 0xbf then
        !w && r == 0 && !x && leBytesLt256 b (p + 1) 4
      else if o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89 then
        match b (p + 1) with
        | none => false
        | some m =>
          m / 64 == 3 && !x &&
          let d := m % 8 + bb
          let s := m / 8 % 8 + r
          if o == 0x31 then w || d == s
          else if o == 0x89 then true
          else w
      else if o == 0x88 then
        match b (p + 1), b (p + 2) with
        | some m, some s =>
          m / 64 == 0 && m % 8 == 4 && s == 0x24 && !w && !x && bb == 0 &&
          (hasRex || m / 8 % 8 < 4)
        | _, _ => false
      else if o == 0x6a then
        !hasRex && match b 1 with | some imm => imm < 256 | none => false
      else if 0x58 ≤ o && o ≤ 0x5f then
        !w && r == 0 && !x
      else if o == 0x0f then
        match b (p + 1) with
        | some 0x05 => !hasRex
        | some 0xb6 =>
          match b (p + 2), b (p + 3) with
          | some m, some s =>
            m / 64 == 0 && m % 8 == 4 && s == 0x24 && !w && !x && bb == 0
          | _, _ => false
        | _ => false
      else if 0x70 ≤ o && o ≤ 0x7f then
        !hasRex && supportedCC (o - 0x70) &&
        match b 1 with | some d => d < 256 | none => false
      else if o == 0xeb then
        !hasRex && match b 1 with | some d => d < 256 | none => false
      else if o == 0xe9 then
        !hasRex && leBytesLt256 b 1 4
      else if o == 0x81 || o == 0x83 then
        match b (p + 1) with
        | none => false
        | some m =>
          m / 64 == 3 && m / 8 % 8 == 7 && w && r == 0 && !x &&
          if o == 0x81 then leBytesLt256 b (p + 2) 4
          else match b (p + 2) with | some imm => imm < 256 | none => false
      else false

/-- Predicate asserting a byte stream decodes identically under `X86Min.decode`
and `X86Dec.decode` in the shared sub-ISA. -/
def CommonDec (b : Nat → Option Nat) : Prop := commonDecB b = true

/-! ### Deliverable 2: Decoder Agreement (`decode_agree`) -/

theorem le_of_bytesLt256 {b : Nat → Option Nat} {off n : Nat}
    (hb : leBytesLt256 b off n = true) :
    ∃ v, X86Min.le b off n = some v ∧ v < 256 ^ n := by
  induction n generalizing off with
  | zero => exact ⟨0, rfl, by decide⟩
  | succ n ih =>
    simp only [leBytesLt256] at hb
    cases ho : b off with
    | none => rw [ho] at hb; cases hb
    | some x =>
      rw [ho] at hb
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
      obtain ⟨hx, hrest⟩ := hb
      obtain ⟨hi, hle, hlt⟩ := ih hrest
      refine ⟨x + 256 * hi, ?_, ?_⟩
      · simp [X86Min.le, ho, hle]
      · rw [Nat.pow_succ']
        omega

theorem decP_le_one (b0 : Nat) : decP b0 ≤ 1 := by
  unfold decP; split <;> omega

theorem decB_cases (b0 : Nat) : decB b0 = 0 ∨ decB b0 = 8 := by
  unfold decB decRex; omega

theorem decR_cases (b0 : Nat) : decR b0 = 0 ∨ decR b0 = 8 := by
  unfold decR decRex; omega

private theorem decode_agree_movImm {b : Nat → Option Nat} {a b0 o : Nat}
    (hb0 : b 0 = some b0) (hbb8 : decB b0 = 0 ∨ decB b0 = 8)
    (hbp : b (decP b0) = some o)
    (h_movI : (0xb8 ≤ o && o ≤ 0xbf) = true)
    (hc : (!decW b0 && decR b0 == 0 && !decX b0 && leBytesLt256 b (decP b0 + 1) 4) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at hc
  obtain ⟨⟨⟨hw, hr0⟩, hx⟩, hbytes⟩ := hc
  obtain ⟨imm, himm_eq, himm_lt⟩ := le_of_bytesLt256 hbytes
  have hlt232 : imm < 2 ^ 32 := by simpa using himm_lt
  have ho_bounds : 0xb8 ≤ o ∧ o ≤ 0xbf := by
    simpa [Bool.and_eq_true, decide_eq_true_eq] using h_movI
  have hd16 : o - 0xb8 + decB b0 < 16 := by rcases hbb8 with h | h <;> omega
  refine ⟨X86Min.Ins.movImm (o - 0xb8 + decB b0) imm, decP b0 + 5, ?_, ?_, by omega, ?_⟩
  · unfold X86Min.decode
    simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hw hr0 hx himm_eq ⊢
    simp [hb0, hbp, ho_bounds, hw, hr0, hx, himm_eq]
  · simp [CommonIns, commonInsB, hd16, hlt232]
  · unfold X86Dec.decode
    simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hw hr0 hx himm_eq ⊢
    simp [hb0, hbp, ho_bounds, hw, hr0, hx, leB_eq_le, himm_eq, toIntIns]

private theorem decode_agree_rr {b : Nat → Option Nat} {a b0 o : Nat}
    (hb0 : b 0 = some b0) (hbb8 : decB b0 = 0 ∨ decB b0 = 8) (hr8 : decR b0 = 0 ∨ decR b0 = 8)
    (hbp : b (decP b0) = some o)
    (h_rr : (o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89) = true)
    (hc : (match b (decP b0 + 1) with
      | none => false
      | some m =>
        m / 64 == 3 && !decX b0 &&
        let d := m % 8 + decB b0
        let s := m / 8 % 8 + decR b0
        if o == 0x31 then decW b0 || d == s
        else if o == 0x89 then true
        else decW b0) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  cases hbm : b (decP b0 + 1) with
  | none => rw [hbm] at hc; cases hc
  | some m =>
    rw [hbm] at hc
    dsimp only at hc
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
    obtain ⟨⟨hm64, hx⟩, hcond⟩ := hc
    have hd16 : m % 8 + decB b0 < 16 := by rcases hbb8 with h | h <;> omega
    have hs16 : m / 8 % 8 + decR b0 < 16 := by rcases hr8 with h | h <;> omega
    simp only [Bool.or_eq_true, beq_iff_eq] at h_rr
    rcases h_rr with ((rfl | rfl) | rfl) | rfl
    · simp only [if_true] at hcond
      cases hw : decW b0 with
      | true =>
        refine ⟨X86Min.Ins.rr .xor 64 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
        · unfold X86Min.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw]
        · simp [CommonIns, commonInsB, hd16, hs16]
        · unfold X86Dec.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw, X86Dec.aluOfRR, toIntIns]
      | false =>
        rw [hw] at hcond
        simp only [Bool.false_or, beq_iff_eq] at hcond
        refine ⟨X86Min.Ins.rr .xor 32 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
        · unfold X86Min.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw]
        · simp [CommonIns, commonInsB, hs16, hcond]
        · unfold X86Dec.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw hcond ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw, X86Dec.aluOfRR, hcond, toIntIns]
    · have hw : decW b0 = true := hcond
      refine ⟨X86Min.Ins.rr .or 64 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
      · unfold X86Min.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
        simp [hb0, hbp, hbm, hm64, hx, hw]
      · simp [CommonIns, commonInsB, hd16, hs16]
      · unfold X86Dec.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
        simp [hb0, hbp, hbm, hm64, hx, hw, X86Dec.aluOfRR, toIntIns]
    · have hw : decW b0 = true := hcond
      refine ⟨X86Min.Ins.rr .test 64 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
      · unfold X86Min.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
        simp [hb0, hbp, hbm, hm64, hx, hw]
      · simp [CommonIns, commonInsB, hd16, hs16]
      · unfold X86Dec.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
        simp [hb0, hbp, hbm, hm64, hx, hw, toIntIns]
    · cases hw : decW b0 with
      | true =>
        refine ⟨X86Min.Ins.rr .mov 64 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
        · unfold X86Min.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw]
        · simp [CommonIns, commonInsB, hd16, hs16]
        · unfold X86Dec.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw, toIntIns]
      | false =>
        refine ⟨X86Min.Ins.rr .mov 32 (m % 8 + decB b0) (m / 8 % 8 + decR b0), decP b0 + 2, ?_, ?_, by omega, ?_⟩
        · unfold X86Min.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw]
        · simp [CommonIns, commonInsB, hd16, hs16]
        · unfold X86Dec.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hx hw ⊢
          simp [hb0, hbp, hbm, hm64, hx, hw, toIntIns]

private theorem decode_agree_stRsp8 {b : Nat → Option Nat} {a b0 : Nat}
    (hb0 : b 0 = some b0) (hr8 : decR b0 = 0 ∨ decR b0 = 8)
    (hbp : b (decP b0) = some 0x88)
    (hc : (match b (decP b0 + 1), b (decP b0 + 2) with
      | some m, some s =>
        m / 64 == 0 && m % 8 == 4 && s == 0x24 && !decW b0 && !decX b0 && decB b0 == 0 &&
        (decHasRex b0 || m / 8 % 8 < 4)
      | _, _ => false) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  cases hbm : b (decP b0 + 1) with
  | none => rw [hbm] at hc; cases hc
  | some m =>
    cases hbs : b (decP b0 + 2) with
    | none => rw [hbm, hbs] at hc; cases hc
    | some s =>
      rw [hbm, hbs] at hc
      dsimp only at hc
      simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
      obtain ⟨⟨⟨⟨⟨⟨hm64, hm8⟩, hs⟩, hw⟩, hx⟩, hbb0⟩, hreg⟩ := hc
      have hs16 : m / 8 % 8 + decR b0 < 16 := by rcases hr8 with h | h <;> omega
      have hreg_imp : ¬(64 ≤ b0 ∧ b0 ≤ 79) → m / 8 % 8 < 4 := by
        simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, decHasRex] at hreg
        intro hnr
        rcases hreg with h | h
        · exact absurd h hnr
        · exact h
      refine ⟨X86Min.Ins.stRsp8 (m / 8 % 8 + decR b0), decP b0 + 3, ?_, ?_, by omega, ?_⟩
      · unfold X86Min.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hbs hw hx hbb0 ⊢
        simp [hb0, hbp, hbm, hbs, hm64, hm8, hs, hw, hx, hbb0]
        exact ⟨by omega, fun h => hreg_imp (by omega)⟩
      · simp [CommonIns, commonInsB, hs16]
      · unfold X86Dec.decode X86Dec.memOp
        simp [decHasRex, decRex, decP, decR, decX, decB] at hbp hbm hbs hx hbb0 ⊢
        simp [hb0, hbp, hbm, hbs, hm64, hm8, hs, hx, hbb0, toIntIns]
        exact fun h => hreg_imp (by omega)

private theorem decode_agree_pop {b : Nat → Option Nat} {a b0 o : Nat}
    (hb0 : b 0 = some b0) (hbb8 : decB b0 = 0 ∨ decB b0 = 8)
    (hbp : b (decP b0) = some o)
    (h_pop : (0x58 ≤ o && o ≤ 0x5f) = true)
    (hc : (!decW b0 && decR b0 == 0 && !decX b0) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at hc
  obtain ⟨⟨hw, hr0⟩, hx⟩ := hc
  have hpop_bounds : 0x58 ≤ o ∧ o ≤ 0x5f := by
    simpa [Bool.and_eq_true, decide_eq_true_eq] using h_pop
  have hd16 : o - 0x58 + decB b0 < 16 := by rcases hbb8 with h | h <;> omega
  refine ⟨X86Min.Ins.pop (o - 0x58 + decB b0), decP b0 + 1, ?_, ?_, by omega, ?_⟩
  · unfold X86Min.decode
    rw [hb0]
    simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hw hr0 hx ⊢
    rw [hbp]; simp only [Option.bind_some]
    repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
  · simp [CommonIns, commonInsB, hd16]
  · unfold X86Dec.decode
    rw [hb0]
    simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hw hr0 hx ⊢
    rw [hbp]; simp only [Option.bind_some]
    repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
    simp [toIntIns]

private theorem decode_agree_0f {b : Nat → Option Nat} {a b0 : Nat}
    (hb0 : b 0 = some b0) (hr8 : decR b0 = 0 ∨ decR b0 = 8)
    (hbp : b (decP b0) = some 0x0f)
    (hc : (match b (decP b0 + 1) with
      | some 0x05 => !decHasRex b0
      | some 0xb6 =>
        match b (decP b0 + 2), b (decP b0 + 3) with
        | some m, some s =>
          m / 64 == 0 && m % 8 == 4 && s == 0x24 && !decW b0 && !decX b0 && decB b0 == 0
        | _, _ => false
      | _ => false) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  cases hbo2 : b (decP b0 + 1) with
  | none => rw [hbo2] at hc; cases hc
  | some o2 =>
    rw [hbo2] at hc
    by_cases h05 : o2 = 0x05
    · subst h05
      simp only [Bool.not_eq_true'] at hc
      have hno : decHasRex b0 = false := hc
      have hp0 : decP b0 = 0 := by simp [decP, hno]
      rw [hp0] at hbp hbo2
      have hbo : b0 = 0x0f := Option.some.inj (hb0.symm.trans hbp)
      subst hbo
      refine ⟨X86Min.Ins.syscall, 2, ?_, ?_, by omega, ?_⟩
      · unfold X86Min.decode; simp [hb0, hbo2]
      · simp [CommonIns, commonInsB]
      · unfold X86Dec.decode; simp [hb0, hbo2, toIntIns]
    · by_cases hb6 : o2 = 0xb6
      · subst hb6
        cases hbm : b (decP b0 + 2) with
        | none => rw [hbm] at hc; cases hc
        | some m =>
          cases hbs : b (decP b0 + 3) with
          | none => rw [hbm, hbs] at hc; cases hc
          | some s =>
            rw [hbm, hbs] at hc
            dsimp only at hc
            simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
            obtain ⟨⟨⟨⟨⟨hm64, hm8⟩, hs⟩, hw⟩, hx⟩, hbb0⟩ := hc
            have hd16 : m / 8 % 8 + decR b0 < 16 := by rcases hr8 with h | h <;> omega
            have hm_lt64 : ¬(64 ≤ m) := by omega
            refine ⟨X86Min.Ins.ldRsp8 (m / 8 % 8 + decR b0), decP b0 + 4, ?_, ?_, by omega, ?_⟩
            · unfold X86Min.decode
              rw [hb0]
              simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbo2 hbm hbs hw hx hbb0 ⊢
              rw [hbp]; simp only [Option.bind_some]
              simp [hbo2, hbm, hbs, hm8, hs, hw, hx, hbb0, hm_lt64]
            · simp [CommonIns, commonInsB, hd16]
            · unfold X86Dec.decode X86Dec.memOp
              rw [hb0]
              simp [decHasRex, decRex, decP, decR, decX, decB] at hbp hbo2 hbm hbs hx hbb0 ⊢
              rw [hbp]; simp only [Option.bind_some]
              simp [hbo2, hbm, hbs, hm64, hm8, hs, hx, hbb0, toIntIns]
      · cases o2 with
        | zero => simp at hc
        | succ k =>
          have hne5 : k + 1 ≠ 5 := fun h => h05 h
          have hne182 : k + 1 ≠ 182 := fun h => hb6 h
          have : False := by revert hc; cases k <;> simp_all
          exact this.elim

private theorem decode_agree_cmpI {b : Nat → Option Nat} {a b0 o : Nat}
    (hb0 : b 0 = some b0) (hbb8 : decB b0 = 0 ∨ decB b0 = 8)
    (hbp : b (decP b0) = some o)
    (h_cmp : (o == 0x81 || o == 0x83) = true)
    (hc : (match b (decP b0 + 1) with
      | none => false
      | some m =>
        m / 64 == 3 && m / 8 % 8 == 7 && decW b0 && decR b0 == 0 && !decX b0 &&
        if o == 0x81 then leBytesLt256 b (decP b0 + 2) 4
        else match b (decP b0 + 2) with | some imm => imm < 256 | none => false) = true) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  have hp := decP_le_one b0
  cases hbm : b (decP b0 + 1) with
  | none => rw [hbm] at hc; cases hc
  | some m =>
    rw [hbm] at hc
    dsimp only at hc
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at hc
    obtain ⟨⟨⟨⟨⟨hm64, hm8⟩, hw⟩, hr0⟩, hx⟩, hrest⟩ := hc
    have hd16 : m % 8 + decB b0 < 16 := by rcases hbb8 with h | h <;> omega
    simp only [Bool.or_eq_true, beq_iff_eq] at h_cmp
    rcases h_cmp with rfl | rfl
    · simp only [if_true] at hrest
      obtain ⟨imm, himm_eq, himm_lt⟩ := le_of_bytesLt256 hrest
      have himm232 : imm < 2 ^ 32 := by simpa using himm_lt
      have hsx : X86Min.sx32 imm < W := sx32_lt_W himm232
      have hsx_mod : X86Min.sx32 imm % X86Min.pow2 64 = X86Min.sx32 imm := by
        rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx
      refine ⟨X86Min.Ins.ri .cmp 64 (m % 8 + decB b0) (X86Min.sx32 imm), decP b0 + 6, ?_, ?_, by omega, ?_⟩
      · unfold X86Min.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hw hr0 hx himm_eq ⊢
        simp [hb0, hbp, hbm, hm64, hm8, hw, hr0, hx, himm_eq, hsx_mod]
        repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
      · simp [CommonIns, commonInsB, hd16, hsx]
      · unfold X86Dec.decode
        simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hw hr0 hx himm_eq ⊢
        simp [hb0, hbp, hbm, hm64, hm8, hw, hr0, hx, leB_eq_le, himm_eq, sx32_eq, toIntIns]
        repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
    · simp at hrest
      cases hbi : b (decP b0 + 2) with
      | none => rw [hbi] at hrest; cases hrest
      | some imm =>
        rw [hbi] at hrest
        simp only [decide_eq_true_eq] at hrest
        have hsx : X86Min.sx8 imm < W := sx8_lt_W hrest
        have hsx_mod : X86Min.sx8 imm % X86Min.pow2 64 = X86Min.sx8 imm := by
          rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx
        refine ⟨X86Min.Ins.ri .cmp 64 (m % 8 + decB b0) (X86Min.sx8 imm), decP b0 + 3, ?_, ?_, by omega, ?_⟩
        · unfold X86Min.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hw hr0 hx hbi ⊢
          simp [hb0, hbp, hbm, hm64, hm8, hw, hr0, hx, hbi, hsx_mod]
          repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
        · simp [CommonIns, commonInsB, hd16, hsx]
        · unfold X86Dec.decode
          simp [decHasRex, decRex, decP, decW, decR, decX, decB] at hbp hbm hw hr0 hx hbi ⊢
          simp [hb0, hbp, hbm, hm64, hm8, hw, hr0, hx, hbi, sx8_eq, toIntIns]
          repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])

theorem decode_agree_raw {b : Nat → Option Nat} {a : Nat}
    (hc : CommonDec b) :
    ∃ ins len,
      X86Min.decode b = some (ins, len) ∧
      CommonIns ins ∧
      len ≤ 7 ∧
      X86Dec.decode b a = some (toIntIns a len ins, len) := by
  unfold CommonDec commonDecB at hc
  cases hb0 : b 0 with
  | none => rw [hb0] at hc; cases hc
  | some b0 =>
    rw [hb0] at hc
    dsimp only at hc
    have hp := decP_le_one b0
    have hbb8 := decB_cases b0
    have hr8 := decR_cases b0
    cases hbp : b (decP b0) with
    | none => rw [hbp] at hc; cases hc
    | some o =>
      rw [hbp] at hc
      dsimp only at hc
      by_cases h_movI : (0xb8 ≤ o && o ≤ 0xbf) = true
      · rw [if_pos h_movI] at hc; exact decode_agree_movImm hb0 hbb8 hbp h_movI hc
      · rw [if_neg h_movI] at hc
        by_cases h_rr : (o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89) = true
        · rw [if_pos h_rr] at hc; exact decode_agree_rr hb0 hbb8 hr8 hbp h_rr hc
        · rw [if_neg h_rr] at hc
          by_cases h_88 : (o == 0x88) = true
          · rw [if_pos h_88] at hc
            have ho : o = 0x88 := eq_of_beq h_88
            subst ho
            exact decode_agree_stRsp8 hb0 hr8 hbp hc
          · rw [if_neg h_88] at hc
            by_cases h_6a : (o == 0x6a) = true
            · rw [if_pos h_6a] at hc
              have ho : o = 0x6a := eq_of_beq h_6a
              subst ho
              have hno : decHasRex b0 = false := by
                cases hhr : decHasRex b0
                · rfl
                · simp [hhr] at hc
              have hp0 : decP b0 = 0 := by simp [decP, hno]
              rw [hp0] at hbp
              have hbo : b0 = 0x6a := Option.some.inj (hb0.symm.trans hbp)
              subst hbo
              cases hb1 : b 1 with
              | none => rw [hb1] at hc; simp [hno] at hc
              | some imm =>
                rw [hb1] at hc
                simp only [hno, Bool.not_false, Bool.true_and, decide_eq_true_eq] at hc
                have hsx : X86Min.sx8 imm < W := sx8_lt_W hc
                refine ⟨X86Min.Ins.push (X86Min.sx8 imm), 2, ?_, ?_, by omega, ?_⟩
                · unfold X86Min.decode; simp [hb0, hb1]
                · simp [CommonIns, commonInsB, hsx]
                · unfold X86Dec.decode; simp [hb0, hb1, sx8_eq, toIntIns]
            · rw [if_neg h_6a] at hc
              by_cases h_pop : (0x58 ≤ o && o ≤ 0x5f) = true
              · rw [if_pos h_pop] at hc; exact decode_agree_pop hb0 hbb8 hbp h_pop hc
              · rw [if_neg h_pop] at hc
                by_cases h_0f : (o == 0x0f) = true
                · rw [if_pos h_0f] at hc
                  have ho : o = 0x0f := eq_of_beq h_0f
                  subst ho
                  exact decode_agree_0f hb0 hr8 hbp hc
                · rw [if_neg h_0f] at hc
                  by_cases h_jcc : (0x70 ≤ o && o ≤ 0x7f) = true
                  · rw [if_pos h_jcc] at hc
                    have hno : decHasRex b0 = false := by
                      cases hhr : decHasRex b0
                      · rfl
                      · simp [hhr] at hc
                    have hp0 : decP b0 = 0 := by simp [decP, hno]
                    rw [hp0] at hbp
                    have hnr_bool : (decide (64 ≤ b0) && decide (b0 ≤ 79)) = false := by
                      simpa [decHasRex] using hno
                    cases hb1 : b 1 with
                    | none => rw [hb1] at hc; simp [hno] at hc
                    | some d =>
                      rw [hb1] at hc
                      simp only [hno, Bool.not_false, Bool.true_and, Bool.and_eq_true, decide_eq_true_eq] at hc
                      obtain ⟨hcc, hd_lt⟩ := hc
                      have hsx : X86Min.sx8 d < W := sx8_lt_W hd_lt
                      have hjcc_bounds : 0x70 ≤ o ∧ o ≤ 0x7f := by
                        simpa [Bool.and_eq_true, decide_eq_true_eq] using h_jcc
                      refine ⟨X86Min.Ins.jcc (o - 0x70) (X86Min.sx8 d), 2, ?_, ?_, by omega, ?_⟩
                      · unfold X86Min.decode
                        rw [hb0]
                        simp only [Bind.bind, Option.bind, hnr_bool, Bool.false_eq_true, if_false]
                        rw [hbp]
                        simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
                        repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
                        simp [hb1]
                      · simp [CommonIns, commonInsB, hcc, hsx]
                      · unfold X86Dec.decode
                        rw [hb0]
                        simp only [Bind.bind, Option.bind, hnr_bool, Bool.false_eq_true, if_false]
                        rw [hbp]
                        simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
                        repeat (first | rw [if_pos (by omega)] | rw [if_neg (by omega)])
                        simp [hb1, sx8_eq, toIntIns]
                  · rw [if_neg h_jcc] at hc
                    by_cases h_eb : (o == 0xeb) = true
                    · rw [if_pos h_eb] at hc
                      have ho : o = 0xeb := eq_of_beq h_eb
                      subst ho
                      have hno : decHasRex b0 = false := by
                        cases hhr : decHasRex b0
                        · rfl
                        · simp [hhr] at hc
                      have hp0 : decP b0 = 0 := by simp [decP, hno]
                      rw [hp0] at hbp
                      have hbo : b0 = 0xeb := Option.some.inj (hb0.symm.trans hbp)
                      subst hbo
                      cases hb1 : b 1 with
                      | none => rw [hb1] at hc; simp [hno] at hc
                      | some d =>
                        rw [hb1] at hc
                        simp only [hno, Bool.not_false, Bool.true_and, decide_eq_true_eq] at hc
                        have hsx : X86Min.sx8 d < W := sx8_lt_W hc
                        refine ⟨X86Min.Ins.jmp (X86Min.sx8 d), 2, ?_, ?_, by omega, ?_⟩
                        · unfold X86Min.decode; simp [hb0, hb1]
                        · simp [CommonIns, commonInsB, hsx]
                        · unfold X86Dec.decode; simp [hb0, hb1, sx8_eq, toIntIns]
                    · rw [if_neg h_eb] at hc
                      by_cases h_e9 : (o == 0xe9) = true
                      · rw [if_pos h_e9] at hc
                        have ho : o = 0xe9 := eq_of_beq h_e9
                        subst ho
                        simp only [Bool.and_eq_true, Bool.not_eq_true'] at hc
                        obtain ⟨hno, hbytes⟩ := hc
                        have hp0 : decP b0 = 0 := by simp [decP, hno]
                        rw [hp0] at hbp
                        have hbo : b0 = 0xe9 := Option.some.inj (hb0.symm.trans hbp)
                        subst hbo
                        obtain ⟨d, hd_eq, hd_lt⟩ := le_of_bytesLt256 hbytes
                        have hd232 : d < 2 ^ 32 := by simpa using hd_lt
                        have hsx : X86Min.sx32 d < W := sx32_lt_W hd232
                        refine ⟨X86Min.Ins.jmp (X86Min.sx32 d), 5, ?_, ?_, by omega, ?_⟩
                        · unfold X86Min.decode; simp [hb0, hd_eq]
                        · simp [CommonIns, commonInsB, hsx]
                        · unfold X86Dec.decode; simp [hb0, leB_eq_le, hd_eq, sx32_eq, toIntIns]
                      · rw [if_neg h_e9] at hc
                        by_cases h_cmp : (o == 0x81 || o == 0x83) = true
                        · rw [if_pos h_cmp] at hc; exact decode_agree_cmpI hb0 hbb8 hbp h_cmp hc
                        · rw [if_neg h_cmp] at hc; cases hc

/-- Decoder agreement (`decode_agree`): on any shared byte encoding at a canonical
address (`a + len < X86Env.W`), `X86Min.decode` and `X86Dec.decode` produce the
exact matching length and corresponding instruction. -/
theorem decode_agree {b : Nat → Option Nat} {a : Nat} {ins : X86Min.Ins} {len : Nat}
    (hc : CommonDec b) (hm : X86Min.decode b = some (ins, len))
    (_hcanon : a + len < W) :
    X86Dec.decode b a = some (toIntIns a len ins, len) ∧ CommonIns ins ∧ len ≤ 7 := by
  obtain ⟨ins', len', hm', hci, hlen, hd⟩ := decode_agree_raw (a := a) hc
  rw [hm] at hm'
  cases hm'
  exact ⟨hd, hci, hlen⟩

/-! ### Deliverable 3: State Simulation Relation (`Sim`) -/

/-- Flag refinement from `X86Min.Flags` to `X86Int.Flags`. -/
structure FlRel (mf : X86Min.Flags) (if_ : X86Int.Flags) : Prop where
  cf : X86Min.OLe mf.cf if_.cf
  zf : X86Min.OLe mf.zf if_.zf
  sf : X86Min.OLe mf.sf if_.sf
  of : X86Min.OLe mf.of if_.of

/-- Address `a` is mapped in a writable region of `X86Env.Mem`. -/
def MemWritable (mem : Mem) (a : Nat) : Prop :=
  ∃ idx r, mem.find a 0 = some (idx, r) ∧ r.rw = true

/-- State simulation invariant between `X86Min.St` and `X86Int.St`. -/
structure Sim (m : X86Min.St) (i : X86Int.St) : Prop where
  rip : m.rip = i.rip
  ripCanon : m.rip + 16 < W
  mLen : m.regs.length = 16
  iLen : i.regs.length = 16
  regsLe : ∀ r < 16, X86Min.OLe (m.reg r) (i.reg r)
  regsDef : ∀ r < 16, r ≠ 11 → (i.reg r).isSome
  regsBound : ∀ r v, i.reg r = some v → v < W
  fl : FlRel m.fl i.fl
  memLen : m.mem.length = X86Min.winSize
  loCanon : m.lo + X86Min.winSize ≤ W
  stackWritable : ∀ x, m.lo ≤ x → x < m.lo + X86Min.winSize → MemWritable i.mem x
  memLe : ∀ x, X86Min.OLe (m.rdByte x) (i.mem.rd x)
  memBound : ∀ x v, i.mem.rd x = some v → v < 256
  stdin : i.k.stdin = m.inp
  inpBound : ∀ c ∈ m.inp, c < 256
  out : i.k.out = m.out.reverse
  oracle : i.k.oracle = []

/-! ### Register, Flag, and Memory Lemmas for `Sim` -/

theorem reg_lt16_bound {m : X86Min.St} {i : X86Int.St} (h : Sim m i) {r v : Nat}
    (hr : r < 16) (hm : m.reg r = some v) : i.reg r = some v ∧ v < W := by
  have hi := h.regsLe r hr v hm
  exact ⟨hi, h.regsBound r v hi⟩

theorem m_reg_setReg {s : X86Min.St} (hlen : s.regs.length = 16) {d r : Nat}
    (hd : d < 16) (v : Option Nat) :
    (s.setReg d v).reg r = if r = d then v else s.reg r := by
  simp only [X86Min.St.reg, X86Min.St.setReg, X86Min.getD_set_reg]
  by_cases h : r = d
  · subst h; simp [hlen, hd]
  · have hne : ¬ (d = r ∧ d < s.regs.length) := fun hc => h hc.1.symm
    simp [h, hne]

theorem i_reg_set {s : X86Int.St} (hlen : s.regs.length = 16) {d r : Nat}
    (hd : d < 16) (v : Option Nat) :
    (s.set d v).reg r = if r = d then v else s.reg r := by
  simp only [X86Int.St.reg, X86Int.St.set, X86Min.getD_set_reg]
  by_cases h : r = d
  · subst h; simp [hlen, hd]
  · have hne : ¬ (d = r ∧ d < s.regs.length) := fun hc => h hc.1.symm
    simp [h, hne]

theorem cond_rel {mf : X86Min.Flags} {if_ : X86Int.Flags} (hf : FlRel mf if_)
    (cc : Nat) {b : Bool} (hc : X86Min.cond mf cc = some b) :
    X86Int.cond if_ cc = some b := by
  unfold X86Min.cond at hc
  unfold X86Int.cond
  cases hdiv : cc / 2 with
  | zero =>
    simp only [hdiv] at hc ⊢
    cases hof : mf.of with
    | none => rw [hof] at hc; cases hc
    | some v => rw [hof] at hc; rw [hf.of v hof]; exact hc
  | succ k1 =>
    cases k1 with
    | zero =>
      simp only [hdiv] at hc ⊢
      cases hcf : mf.cf with
      | none => rw [hcf] at hc; cases hc
      | some v => rw [hcf] at hc; rw [hf.cf v hcf]; exact hc
    | succ k2 =>
      cases k2 with
      | zero =>
        simp only [hdiv] at hc ⊢
        cases hzf : mf.zf with
        | none => rw [hzf] at hc; cases hc
        | some v => rw [hzf] at hc; rw [hf.zf v hzf]; exact hc
      | succ k3 =>
        cases k3 with
        | zero =>
          simp only [hdiv] at hc ⊢
          cases hcf : mf.cf with
          | none => rw [hcf] at hc; cases hc
          | some vc =>
            rw [hcf] at hc; rw [hf.cf vc hcf]
            cases hzf : mf.zf with
            | none => rw [hzf] at hc; cases hc
            | some vz => rw [hzf] at hc; rw [hf.zf vz hzf]; exact hc
        | succ k4 =>
          cases k4 with
          | zero =>
            simp only [hdiv] at hc ⊢
            cases hsf : mf.sf with
            | none => rw [hsf] at hc; cases hc
            | some v => rw [hsf] at hc; rw [hf.sf v hsf]; exact hc
          | succ k5 =>
            cases k5 with
            | zero => simp [hdiv] at hc
            | succ k6 =>
              cases k6 with
              | zero =>
                simp only [hdiv] at hc ⊢
                cases hsf : mf.sf with
                | none => rw [hsf] at hc; cases hc
                | some vs =>
                  rw [hsf] at hc; rw [hf.sf vs hsf]
                  cases hof : mf.of with
                  | none => rw [hof] at hc; cases hc
                  | some vo => rw [hof] at hc; rw [hf.of vo hof]; exact hc
              | succ k7 =>
                cases k7 with
                | zero =>
                  simp only [hdiv] at hc ⊢
                  cases hsf : mf.sf with
                  | none => rw [hsf] at hc; cases hc
                  | some vs =>
                    rw [hsf] at hc; rw [hf.sf vs hsf]
                    cases hof : mf.of with
                    | none => rw [hof] at hc; cases hc
                    | some vo =>
                      rw [hof] at hc; rw [hf.of vo hof]
                      cases hzf : mf.zf with
                      | none => rw [hzf] at hc; cases hc
                      | some vz => rw [hzf] at hc; rw [hf.zf vz hzf]; exact hc
                | succ k8 => simp [hdiv] at hc

theorem find_set_same_bounds (m : Mem) (idx : Nat) (r r' : Region)
    (hm : m[idx]? = some r) (hb : r'.base = r.base) (hh : r'.hi = r.hi) (x i : Nat) :
    Mem.find (m.set idx r') x i =
      (Mem.find m x i).map fun p => if p.1 = i + idx then (p.1, r') else p := by
  induction m generalizing idx i with
  | nil => simp at hm
  | cons s rest ih =>
    cases idx with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hm
      subst hm
      simp only [List.set_cons_zero, Mem.find]
      have hcond : (r'.base ≤ x ∧ x < r'.hi) ↔ (s.base ≤ x ∧ x < s.hi) := by rw [hb, hh]
      by_cases hc : s.base ≤ x ∧ x < s.hi
      · have hc' : r'.base ≤ x ∧ x < r'.hi := hcond.mpr hc
        simp [hc, hc']
      · have hc' : ¬ (r'.base ≤ x ∧ x < r'.hi) := fun h => hc (hcond.mp h)
        simp only [hc, hc', if_false]
        cases hf : Mem.find rest x (i + 1) with
        | none => rfl
        | some p =>
          have hge := (find_spec rest x (i + 1) hf).1
          have hne : p.1 ≠ i := by omega
          simp [hne]
    | succ idx =>
      simp only [List.getElem?_cons_succ] at hm
      simp only [List.set_cons_succ, Mem.find]
      by_cases hc : s.base ≤ x ∧ x < s.hi
      · simp [hc]
      · simp only [hc, if_false]
        rw [ih idx hm (i + 1)]
        cases Mem.find rest x (i + 1) with
        | none => rfl
        | some p =>
          simp only [Option.map_some]
          by_cases hp : p.1 = i + 1 + idx
          · have hp' : p.1 = i + (idx + 1) := by omega
            rw [if_pos hp, if_pos hp']
          · have hp' : p.1 ≠ i + (idx + 1) := by omega
            rw [if_neg hp, if_neg hp']

theorem find_bounds {m : Mem} {x i j : Nat} {r : Region} (h : Mem.find m x i = some (j, r)) :
    r.base ≤ x ∧ x < r.hi := by
  induction m generalizing i with
  | nil => simp [Mem.find] at h
  | cons s rest ih =>
    simp only [Mem.find] at h
    split at h
    · rename_i hc; simp at h; obtain ⟨_, rfl⟩ := h; exact hc
    · exact ih h

theorem mem_wr_of_writable {mem : Mem} {a v : Nat} (hw : MemWritable mem a) :
    ∃ mem', Mem.wr mem a v = some mem' ∧
      (∀ x, MemWritable mem' x ↔ MemWritable mem x) ∧
      (∀ x, Mem.rd mem' x = if x = a then some (v % 256) else Mem.rd mem x) := by
  obtain ⟨idx, r, hfind, hrw⟩ := hw
  have hspec := find_spec mem a 0 hfind
  simp only [Nat.sub_zero] at hspec
  have hba := find_bounds hfind
  have hadata : a - r.base < r.data.length := by
    unfold Region.hi at hba; omega
  let r' : Region := { r with data := r.data.set (a - r.base) (v % 256) }
  have hr'_base : r'.base = r.base := rfl
  have hr'_hi : r'.hi = r.hi := by simp [r', Region.hi]
  let mem' := mem.set idx r'
  have hwr : Mem.wr mem a v = some mem' := by
    simp [Mem.wr, hfind, hrw, mem', r']
  refine ⟨mem', hwr, ?_, ?_⟩
  · intro x
    have hf := find_set_same_bounds mem idx r r' hspec.2 hr'_base hr'_hi x 0
    simp only [Nat.zero_add] at hf
    constructor
    · rintro ⟨idx', s', hfs, hsrw⟩
      rw [hf] at hfs
      cases hfx : Mem.find mem x 0 with
      | none => rw [hfx] at hfs; cases hfs
      | some q =>
        rw [hfx] at hfs; simp only [Option.map_some] at hfs
        split at hfs
        · rename_i hq
          simp only [Option.some.injEq, Prod.mk.injEq] at hfs
          obtain ⟨rfl, rfl⟩ := hfs
          have hqr : q.2 = r := by
            have hs1 := (find_spec mem x 0 hfx).2
            simp only [Nat.sub_zero, hq] at hs1
            rw [hspec.2] at hs1; exact Option.some.inj hs1.symm
          exact ⟨q.1, q.2, hfx, by rw [hqr]; exact hrw⟩
        · cases hfs
          exact ⟨idx', s', hfx, hsrw⟩
    · rintro ⟨idx', s', hfx, hsrw⟩
      rw [hfx] at hf
      simp only [Option.map_some] at hf
      split at hf
      · exact ⟨idx', r', hf, by simpa [r'] using hrw⟩
      · exact ⟨idx', s', hf, hsrw⟩
  · intro x
    have hf := find_set_same_bounds mem idx r r' hspec.2 hr'_base hr'_hi x 0
    simp only [Nat.zero_add] at hf
    simp only [Mem.rd]
    rw [hf]
    cases hfx : Mem.find mem x 0 with
    | none =>
      by_cases hx : x = a
      · subst hx; rw [hfind] at hfx; cases hfx
      · simp [hx]
    | some q =>
      simp only [Option.map_some]
      by_cases hx : x = a
      · subst hx; rw [hfind] at hfx
        simp only [Option.some.injEq] at hfx; subst hfx
        simp [r', hadata]
      · simp only [hx, if_false]
        by_cases hq : q.1 = idx
        · rw [if_pos hq]
          have hxb := find_bounds hfx
          have hpeq : q.2 = r := by
            have hs1 := (find_spec mem x 0 hfx).2
            simp only [Nat.sub_zero, hq] at hs1
            rw [hspec.2] at hs1; exact Option.some.inj hs1.symm
          have hne : a - q.2.base ≠ x - q.2.base := by
            rw [hpeq] at hxb ⊢
            omega
          rw [← hpeq] at hadata
          simp [r', ← hpeq, hne]
        · rw [if_neg hq]

theorem mem_wrN_spec (mem : Mem) (a : Nat) (bs : List Nat)
    (hw : ∀ k < bs.length, MemWritable mem (a + k))
    (hb : ∀ b ∈ bs, b < 256) :
    ∃ mem', Mem.wrN mem a bs = some mem' ∧
      (∀ x, MemWritable mem' x ↔ MemWritable mem x) ∧
      (∀ x, Mem.rd mem' x =
        if h : a ≤ x ∧ x < a + bs.length then some bs[x - a] else Mem.rd mem x) := by
  induction bs generalizing mem a with
  | nil =>
    refine ⟨mem, rfl, fun _ => Iff.rfl, fun x => ?_⟩
    rw [dif_neg (by simp)]
  | cons b rest ih =>
    have hw0 : MemWritable mem a := by simpa using hw 0 (by simp)
    have hb0 : b < 256 := hb b (List.mem_cons_self ..)
    obtain ⟨mem1, hwr1, hw1_iff, hrd1⟩ := mem_wr_of_writable (v := b) hw0
    have hw_rest : ∀ k < rest.length, MemWritable mem1 (a + 1 + k) := by
      intro k hk
      rw [hw1_iff]
      have hk' : k + 1 < (b :: rest).length := by simp; omega
      have hwk := hw (k + 1) hk'
      rwa [show a + (k + 1) = a + 1 + k from by omega] at hwk
    have hb_rest : ∀ x ∈ rest, x < 256 := fun x hx => hb x (List.mem_cons_of_mem _ hx)
    obtain ⟨mem', hwrN, hw'_iff, hrd'⟩ := ih mem1 (a + 1) hw_rest hb_rest
    refine ⟨mem', ?_, ?_, ?_⟩
    · simp [Mem.wrN, hwr1, hwrN]
    · intro x; rw [hw'_iff, hw1_iff]
    · intro x
      rw [hrd']
      by_cases hx1 : a + 1 ≤ x ∧ x < a + 1 + rest.length
      · rw [dif_pos hx1]
        have hx : a ≤ x ∧ x < a + (b :: rest).length := by simp; omega
        rw [dif_pos hx]
        have hidx : x - a = (x - (a + 1)) + 1 := by omega
        simp [hidx]
      · rw [dif_neg hx1]
        rw [hrd1]
        by_cases hxa : x = a
        · subst x
          have hx : a ≤ a ∧ a < a + (b :: rest).length := by simp
          rw [dif_pos hx, if_pos rfl]
          simp [Nat.mod_eq_of_lt hb0]
        · have hx : ¬ (a ≤ x ∧ x < a + (b :: rest).length) := by simp at *; omega
          rw [dif_neg hx, if_neg hxa]

theorem min_wrByte_spec {s : X86Min.St} (hlen : s.mem.length = X86Min.winSize)
    {a v : Nat} (ha : s.lo ≤ a ∧ a < s.lo + X86Min.winSize) :
    ∃ s', s.wrByte a v = some s' ∧
      s' = { s with mem := s.mem.set (a - s.lo) (some v) } ∧
      s'.mem.length = X86Min.winSize ∧
      (∀ x, s'.rdByte x = if x = a then some v else s.rdByte x) := by
  have hcond : (s.lo ≤ a && a < s.lo + X86Min.winSize) = true := by simp [ha]
  refine ⟨_, by simp [X86Min.St.wrByte, hcond], rfl, by simp [hlen], ?_⟩
  intro x
  simp only [X86Min.St.rdByte]
  by_cases hxa : x = a
  · subst x
    have hidx : a - s.lo < s.mem.length := by rw [hlen]; omega
    simp [hcond, hidx]
  · by_cases hx : (s.lo ≤ x && x < s.lo + X86Min.winSize) = true
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at hx
      have hne : ¬ (a - s.lo = x - s.lo ∧ a - s.lo < s.mem.length) := by
        intro hc; apply hxa; omega
      simp [hx, X86Min.getD_set_reg, hne, hxa]
    · simp [hx, hxa]

theorem min_wrBytes_spec (n : Nat) {s s' : X86Min.St} (hlen : s.mem.length = X86Min.winSize)
    (a v : Nat) (h : s.wrBytes a v n = some s') :
    s'.mem.length = X86Min.winSize ∧
    s'.lo = s.lo ∧ s'.regs = s.regs ∧ s'.rip = s.rip ∧
    s'.fl = s.fl ∧ s'.inp = s.inp ∧ s'.out = s.out ∧
    (∀ k < n, s.lo ≤ a + k ∧ a + k < s.lo + X86Min.winSize) ∧
    (∀ x, s'.rdByte x =
      if a ≤ x ∧ x < a + n then some (v / 256 ^ (x - a) % 256) else s.rdByte x) := by
  induction n generalizing s a v with
  | zero =>
    simp [X86Min.St.wrBytes] at h
    subst h
    refine ⟨hlen, rfl, rfl, rfl, rfl, rfl, rfl, fun k hk => by omega, fun x => ?_⟩
    rw [if_neg (by omega)]
  | succ n ih =>
    simp only [X86Min.St.wrBytes, bind, Option.bind] at h
    cases h1 : s.wrByte a (v % 256) with
    | none => rw [h1] at h; cases h
    | some s1 =>
      rw [h1] at h
      simp only [X86Min.St.wrByte] at h1
      split at h1
      · rename_i hcond
        simp only [Option.some.injEq] at h1
        subst h1
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
        have hs1len : ({ s with mem := s.mem.set (a - s.lo) (some (v % 256)) } : X86Min.St).mem.length = X86Min.winSize := by
          simp [hlen]
        obtain ⟨hm, hlo, hr, hrip, hfl, hinp, hout, hbounds, hrd⟩ := ih hs1len (a + 1) (v / 256) h
        refine ⟨hm, hlo, hr, hrip, hfl, hinp, hout, ?_, ?_⟩
        · intro k hk
          cases k with
          | zero => exact hcond
          | succ k =>
            have hk' := hbounds k (by omega)
            rwa [show a + 1 + k = a + (k + 1) from by omega] at hk'
        · intro x
          rw [hrd]
          by_cases hx1 : a + 1 ≤ x ∧ x < a + 1 + n
          · rw [if_pos hx1, if_pos (by omega)]
            have hdiv : (v / 256) / 256 ^ (x - (a + 1)) = v / 256 ^ (x - a) := by
              rw [Nat.div_div_eq_div_mul]
              have hexp : 256 * 256 ^ (x - (a + 1)) = 256 ^ (x - a) := by
                have hd : x - a = (x - (a + 1)) + 1 := by omega
                rw [hd, Nat.pow_succ, Nat.mul_comm]
              rw [hexp]
            rw [hdiv]
          · rw [if_neg hx1]
            obtain ⟨_, _, heq, _, hrd1⟩ := min_wrByte_spec hlen hcond (v := v % 256)
            have hsame : ({ s with mem := s.mem.set (a - s.lo) (some (v % 256)) } : X86Min.St).rdByte x =
                if x = a then some (v % 256) else s.rdByte x := by
              simpa [heq] using hrd1 x
            rw [hsame]
            by_cases hxa : x = a
            · subst x
              rw [if_pos (by omega)]
              simp
            · have hnot : ¬ (a ≤ x ∧ x < a + (n + 1)) := by omega
              rw [if_neg hnot, if_neg hxa]
      · cases h1

theorem le_to_rdN_leVal {f : Nat → Option Nat} {mem : Mem}
    (hle : ∀ x, X86Min.OLe (f x) (mem.rd x))
    (hb : ∀ x v, f x = some v → v < 256) :
    ∀ n off v, X86Min.le f off n = some v →
      ∃ bs, mem.rdN off n = some bs ∧ X86Int.leVal bs = v ∧ v < 256 ^ n := by
  intro n
  induction n with
  | zero =>
    intro off v h
    simp [X86Min.le] at h
    subst h
    exact ⟨[], rfl, rfl, by decide⟩
  | succ n ih =>
    intro off v h
    simp only [X86Min.le, bind, Option.bind, pure] at h
    cases hf : f off with
    | none => rw [hf] at h; cases h
    | some lo =>
      rw [hf] at h
      cases hl : X86Min.le f (off + 1) n with
      | none => rw [hl] at h; cases h
      | some hi =>
        rw [hl] at h
        simp only [Option.some.injEq] at h
        subst h
        obtain ⟨bs, hrd, hval, hlt⟩ := ih (off + 1) hi hl
        have hmem_lo := hle off lo hf
        have hlo_lt := hb off lo hf
        refine ⟨lo :: bs, ?_, ?_, ?_⟩
        · simp [Mem.rdN, hmem_lo, hrd]
        · simp [X86Int.leVal, hval]
        · rw [Nat.pow_succ']; omega

/-! ### Deliverable 4: Single-Step and Multi-Step Forward Simulation -/

/-- Agreement and canonicality precondition on a pair of images `(imMin, imInt)`
restricted to a control-flow closed set of valid instruction addresses `P`. -/
structure ImgCompat (P : Nat → Prop) (imMin : X86Min.Img)
    (imInt : Nat → Option (X86Int.Ins × Nat)) : Prop where
  agree : ∀ a ins len, P a → imMin.decodeAt a = some (ins, len) →
    CommonIns ins ∧
    (∀ w, ins ≠ .rr .xor w 11 11) ∧
    a + len + 16 < W ∧
    imInt a = some (toIntIns a len ins, len)
  closed : ∀ a ins len, P a → imMin.decodeAt a = some (ins, len) →
    match ins with
    | .jmp rel => P ((a + len + rel) % W) ∧ (a + len + rel) % W + 16 < W
    | .jcc _ rel => P (a + len) ∧ P ((a + len + rel) % W) ∧ (a + len + rel) % W + 16 < W
    | _ => P (a + len)

/-- Decidable check for `ImgCompat` on a finite list of valid code addresses `addrs`. -/
def checkImgCompat (addrs : List Nat) (imMin : X86Min.Img)
    (imInt : Nat → Option (X86Int.Ins × Nat)) : Bool :=
  addrs.all fun a =>
    match imMin.decodeAt a with
    | none => true
    | some (ins, len) =>
      commonInsB ins &&
      !(match ins with | .rr .xor _ 11 11 => true | _ => false) &&
      a + len + 16 < W &&
      (imInt a == some (toIntIns a len ins, len)) &&
      match ins with
      | .jmp rel =>
        let t := (a + len + rel) % W
        (t ∈ addrs) && (t + 16 < W)
      | .jcc _ rel =>
        let t := (a + len + rel) % W
        ((a + len) ∈ addrs) && (t ∈ addrs) && (t + 16 < W)
      | _ => (a + len) ∈ addrs

theorem imgCompat_of_check {addrs : List Nat} {imMin : X86Min.Img}
    {imInt : Nat → Option (X86Int.Ins × Nat)}
    (h : checkImgCompat addrs imMin imInt = true) :
    ImgCompat (· ∈ addrs) imMin imInt := by
  have hall := List.all_eq_true.1 h
  constructor
  · intro a ins len ha hd
    have ha' := hall a ha
    rw [hd] at ha'
    simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, beq_iff_eq] at ha'
    obtain ⟨⟨⟨⟨hci, hno11⟩, hcan⟩, him⟩, _⟩ := ha'
    refine ⟨hci, ?_, hcan, him⟩
    intro w heq
    subst heq; simp at hno11
  · intro a ins len ha hd
    have ha' := hall a ha
    rw [hd] at ha'
    simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, beq_iff_eq] at ha'
    obtain ⟨_, hcl⟩ := ha'
    cases ins with
    | jmp rel =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hcl
      exact hcl
    | jcc cc rel =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hcl
      exact ⟨hcl.1.1, hcl.1.2, hcl.2⟩
    | movImm _ _ | rr _ _ _ _ | ri _ _ _ _ | shlI _ _ _ | inc _ _
    | push _ | pop _ | syscall | ldRsp8 _ | stRsp8 _ =>
      simpa using hcl

/-- Helper: setting one register `d < 16` to `some v` with `v < W` preserves `Sim`. -/
theorem sim_setReg {m : X86Min.St} {i : X86Int.St} (h : Sim m i)
    {d v : Nat} (hd : d < 16) (hv : v < W) (rip' : Nat) (hrip' : rip' + 16 < W)
    {mf' : X86Min.Flags} {if' : X86Int.Flags} (hfl : FlRel mf' if') :
    Sim { (m.setReg d (some v)) with rip := rip', fl := mf' }
        { (i.set d (some v)) with rip := rip', fl := if' } where
  rip := rfl
  ripCanon := hrip'
  mLen := by simp [X86Min.St.setReg, h.mLen]
  iLen := by simp [X86Int.St.set, h.iLen]
  regsLe r hr := by
    change X86Min.OLe ((m.setReg d (some v)).reg r) ((i.set d (some v)).reg r)
    rw [m_reg_setReg h.mLen hd, i_reg_set h.iLen hd]
    by_cases hrd : r = d
    · simp [hrd, X86Min.OLe.refl]
    · simp [hrd]; exact h.regsLe r hr
  regsDef r hr hr11 := by
    change ((i.set d (some v)).reg r).isSome = true
    rw [i_reg_set h.iLen hd]
    by_cases hrd : r = d
    · simp [hrd]
    · simp [hrd]; exact h.regsDef r hr hr11
  regsBound r u hu := by
    change (i.set d (some v)).reg r = some u at hu
    by_cases hr16 : r < 16
    · rw [i_reg_set h.iLen hd] at hu
      by_cases hrd : r = d
      · simp [hrd] at hu; subst hu; exact hv
      · simp [hrd] at hu; exact h.regsBound r u hu
    · have hlen : (i.set d (some v)).regs.length = 16 := by simp [X86Int.St.set, h.iLen]
      have hnone : (i.set d (some v)).regs[r]? = none :=
        List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
      simp [X86Int.St.reg, hnone] at hu
  fl := hfl
  memLen := h.memLen
  loCanon := h.loCanon
  stackWritable := h.stackWritable
  memLe := h.memLe
  memBound := h.memBound
  stdin := h.stdin
  inpBound := h.inpBound
  out := h.out
  oracle := h.oracle

theorem sim_with_rip_fl {m : X86Min.St} {i : X86Int.St} (h : Sim m i)
    (rip' : Nat) (hrip' : rip' + 16 < W)
    {mf' : X86Min.Flags} {if' : X86Int.Flags} (hfl : FlRel mf' if') :
    Sim { m with rip := rip', fl := mf' } { i with rip := rip', fl := if' } where
  rip := rfl
  ripCanon := hrip'
  mLen := h.mLen
  iLen := h.iLen
  regsLe := h.regsLe
  regsDef := h.regsDef
  regsBound := h.regsBound
  fl := hfl
  memLen := h.memLen
  loCanon := h.loCanon
  stackWritable := h.stackWritable
  memLe := h.memLe
  memBound := h.memBound
  stdin := h.stdin
  inpBound := h.inpBound
  out := h.out
  oracle := h.oracle

theorem sim_set_rip {m : X86Min.St} {i : X86Int.St} (h : Sim m i)
    (rip' : Nat) (hrip' : rip' + 16 < W) :
    Sim { m with rip := rip' } { i with rip := rip' } :=
  sim_with_rip_fl h rip' hrip' h.fl

/-- Single-step forward simulation (`step_sim`). -/
theorem step_sim {P : Nat → Prop} {imMin : X86Min.Img}
    {imInt : Nat → Option (X86Int.Ins × Nat)}
    (hic : ImgCompat P imMin imInt)
    {m : X86Min.St} {i : X86Int.St} (hp : P m.rip) (hsim : Sim m i) :
    (∀ m', X86Min.step imMin m = .inl m' →
      ∃ i', X86Int.step imInt i = .ok i' ∧ P m'.rip ∧ Sim m' i') ∧
    (∀ code out, X86Min.step imMin m = .inr (.exit code out) →
      X86Int.step imInt i = .exit code ∧ i.k.out = out) := by
  unfold X86Min.step X86Int.step
  rw [← hsim.rip]
  cases hdec : imMin.decodeAt m.rip with
  | none =>
    simp
  | some p =>
    obtain ⟨ins, len⟩ := p
    obtain ⟨hci, hno11, hcanon, him⟩ := hic.agree m.rip ins len hp hdec
    have hclosed := hic.closed m.rip ins len hp hdec
    rw [him]
    dsimp only
    have hnx_lt : m.rip + len < W := by omega
    have hnx_mod : (m.rip + len) % W = m.rip + len := Nat.mod_eq_of_lt hnx_lt
    have hnx_eq : (m.rip + len) % X86Min.pow2 64 = m.rip + len := by
      rw [pow2_64_eq_W]; exact hnx_mod
    have hrip_int : i.rip + len = m.rip + len := by rw [← hsim.rip]
    have hnx_canon : m.rip + len + 16 < W := hcanon
    cases ins with
    | shlI _ _ _ => simp [CommonIns, commonInsB] at hci
    | inc _ _ => simp [CommonIns, commonInsB] at hci
    | movImm d imm =>
      simp only [CommonIns, commonInsB, Bool.and_eq_true, decide_eq_true_eq] at hci
      obtain ⟨hd16, himm32⟩ := hci
      have himmW : imm < W := Nat.lt_trans himm32 (by decide)
      have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW
      simp only [X86Min.okOf, X86Min.St.wr, beq_iff_eq, show (32 : Nat) ≠ 8 by decide, if_false,
        X86Int.exec, toIntIns, himm_mod]
      constructor
      · intro m' hm'; cases hm'
        refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
        simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
          sim_setReg hsim hd16 himmW (m.rip + len) hnx_canon hsim.fl
      · intro code out h; cases h
    | rr op w d src =>
      cases op with
      | add | sub | cmp => simp [CommonIns, commonInsB] at hci
      | mov =>
        simp only [CommonIns, commonInsB, Bool.and_eq_true, decide_eq_true_eq] at hci
        obtain ⟨⟨hd16, hs16⟩, hw⟩ := hci
        by_cases hw64 : w = 64
        · subst hw64
          simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf]
          cases hms : m.reg src with
          | none =>
            simp [X86Min.St.rd, hms]
          | some vs =>
            obtain ⟨his, hvsW⟩ := reg_lt16_bound hsim hs16 hms
            have hvs_mod : vs % X86Min.pow2 64 = vs := by
              rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvsW
            simp only [X86Min.St.rd, hms, his, Option.map_some, hvs_mod, bind, Option.bind,
              X86Min.St.wr, show (64 == 8) = false by decide, Bool.false_eq_true, if_false, pure]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
                sim_setReg hsim hd16 hvsW (m.rip + len) hnx_canon hsim.fl
            · intro code out h; cases h
        · have hw32 : w = 32 := by
            revert hw hw64
            cases w <;> simp
            intro hw _
            split at hw <;> simp_all
          subst hw32
          simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf]
          cases hms : m.reg src with
          | none => simp [X86Min.St.rd, hms]
          | some vs =>
            obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
            have hv32_lt : vs % 2 ^ 32 < W :=
              Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
            simp only [X86Min.St.rd, pow2_32_eq, hms, his, Option.map_some, bind, Option.bind,
              X86Min.St.wr, show (32 == 8) = false by decide, Bool.false_eq_true, if_false, pure]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
                sim_setReg hsim hd16 hv32_lt (m.rip + len) hnx_canon hsim.fl
            · intro code out h; cases h
      | xor =>
        simp only [CommonIns, commonInsB, Bool.and_eq_true, decide_eq_true_eq] at hci
        obtain ⟨⟨hd16, hs16⟩, hw⟩ := hci
        have hw_cases : w = 64 ∨ (w = 32 ∧ d = src) := by
          revert hw
          cases w with
          | zero => simp
          | succ n =>
            cases n with
            | zero => simp
            | succ _ =>
              split <;> simp_all [beq_iff_eq]
        by_cases hds : d = src
        · subst hds
          have hd11 : d ≠ 11 := fun h => by subst h; exact hno11 w rfl
          have hid : ∃ a, i.reg d = some a := Option.isSome_iff_exists.1 (hsim.regsDef d hd16 hd11)
          obtain ⟨va, hia⟩ := hid
          have hw_not8 : (w == 8) = false := by
            rcases hw_cases with rfl | ⟨rfl, _⟩ <;> decide
          have hzero : Nat.xor va va = 0 := Nat.xor_self va
          have hfl_eq : FlRel
              (X86Min.alu .xor w 0 0).2
              (X86Int.aluF .xor va va).2 := by
            rcases hw_cases with rfl | ⟨rfl, _⟩
            · simp only [X86Min.alu, X86Int.aluF, hzero, X86Min.msb, X86Int.msb]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            · simp only [X86Min.alu, X86Int.aluF, hzero, X86Min.msb, X86Int.msb]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
          have hres0 : (X86Min.alu .xor w 0 0).1 = 0 := by
            rcases hw_cases with rfl | ⟨rfl, _⟩ <;> simp [X86Min.alu]
          simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, hia,
            show (X86Min.Op.xor == .xor && d == d) = true by simp, if_true,
            bind, Option.bind, pure, X86Min.St.wr, hw_not8, Bool.false_eq_true, if_false,
            show (X86Min.Op.xor == .test) = false by decide]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
            rw [hres0]
            simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Int.aluF, hzero] using
              sim_setReg hsim hd16 (by decide : 0 < W) (m.rip + len) hnx_canon hfl_eq
          · intro code out h; cases h
        · have hw64 : w = 64 := by
            rcases hw_cases with rfl | ⟨_, heq⟩
            · rfl
            · exact absurd heq hds
          subst hw64
          have hzero_f : (X86Min.Op.xor == .xor && d == src) = false := by simp [hds]
          simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
            Bool.false_eq_true, if_false]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            cases hms : m.reg src with
            | none => simp [X86Min.St.rd, hmd, hms]
            | some vb =>
              obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
              obtain ⟨his, hvbW⟩ := reg_lt16_bound hsim hs16 hms
              have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
              have hvb_mod : vb % X86Min.pow2 64 = vb := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvbW
              have hres_lt : Nat.xor va vb < W := xor_lt_W hvaW hvbW
              have hfl : FlRel (X86Min.alu .xor 64 va vb).2 (X86Int.aluF .xor va vb).2 := by
                simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
                bind, Option.bind, pure, X86Min.St.wr,
                show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.xor == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
                simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hnx_canon hfl
              · intro code out h; cases h
      | or =>
        simp only [CommonIns, commonInsB, Bool.and_eq_true, decide_eq_true_eq] at hci
        obtain ⟨⟨hd16, hs16⟩, hw⟩ := hci
        have hw64 : w = 64 := by revert hw; cases w <;> simp; split <;> simp_all
        subst hw64
        have hzero_f : (X86Min.Op.or == .xor && d == src) = false := by simp
        simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
          Bool.false_eq_true, if_false]
        cases hmd : m.reg d with
        | none => simp [X86Min.St.rd, hmd]
        | some va =>
          cases hms : m.reg src with
          | none => simp [X86Min.St.rd, hmd, hms]
          | some vb =>
            obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
            obtain ⟨his, hvbW⟩ := reg_lt16_bound hsim hs16 hms
            have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
            have hvb_mod : vb % X86Min.pow2 64 = vb := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvbW
            have hres_lt : Nat.lor va vb < W := lor_lt_W hvaW hvbW
            have hfl : FlRel (X86Min.alu .or 64 va vb).2 (X86Int.aluF .or va vb).2 := by
              simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
              bind, Option.bind, pure, X86Min.St.wr,
              show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
              show (X86Min.Op.or == .test) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                sim_setReg hsim hd16 hres_lt (m.rip + len) hnx_canon hfl
            · intro code out h; cases h
      | test =>
        simp only [CommonIns, commonInsB, Bool.and_eq_true, decide_eq_true_eq] at hci
        obtain ⟨⟨hd16, hs16⟩, hw⟩ := hci
        have hw64 : w = 64 := by revert hw; cases w <;> simp; split <;> simp_all
        subst hw64
        have hzero_f : (X86Min.Op.test == .xor && d == src) = false := by simp
        simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
          Bool.false_eq_true, if_false]
        cases hmd : m.reg d with
        | none => simp [X86Min.St.rd, hmd]
        | some va =>
          cases hms : m.reg src with
          | none => simp [X86Min.St.rd, hmd, hms]
          | some vb =>
            obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
            obtain ⟨his, hvbW⟩ := reg_lt16_bound hsim hs16 hms
            have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
            have hvb_mod : vb % X86Min.pow2 64 = vb := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvbW
            have hfl : FlRel (X86Min.alu .test 64 va vb).2 (X86Int.aluF .and va vb).2 := by
              simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
              bind, Option.bind, pure, show (X86Min.Op.test == .test) = true by decide, if_true]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, by simpa [hnx_eq] using hclosed, ?_⟩
              simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hnx_canon hfl
            · intro code out h; cases h
    | ri op w d imm =>
      simp only [CommonIns, commonInsB, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hci
      obtain ⟨⟨⟨hop, hw64⟩, hd16⟩, himmW⟩ := hci
      subst hop hw64
      have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW
      simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
      cases hmd : m.reg d with
      | none => simp [X86Min.St.rd, hmd]
      | some va =>
        obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
        have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
        have hfl : FlRel (X86Min.alu .cmp 64 va imm).2 (X86Int.aluF .sub va imm).2 := by
          simp only [X86Min.alu, X86Int.aluF, pow2_64_eq_W, msb_64_eq]
          exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
        simp only [X86Min.St.rd, hmd, hid, Option.map_some, hva_mod, bind, Option.bind, pure,
          show (X86Min.Op.cmp == .cmp) = true by decide, if_true]
        constructor
        · intro m' hm'; cases hm'
          refine ⟨_, rfl, by simpa [hnx_eq] using hclosed, ?_⟩
          simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hnx_canon hfl
        · intro code out h; cases h
    | push imm =>
      simp only [CommonIns, commonInsB, decide_eq_true_eq] at hci
      have himm_mod : imm % W = imm := Nat.mod_eq_of_lt hci
      simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        simp only [hi4, bind, Option.bind, pure]
        let sp' := (sp + W - 8) % W
        have hsp'_min : (sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 = sp' := rfl
        have hsp'W : sp' < W := Nat.mod_lt _ (by decide)
        rw [hsp'_min]
        cases hwr : X86Min.St.wrBytes { m with rip := (m.rip + len) % X86Min.pow2 64 } sp' imm 8 with
        | none => simp
        | some t =>
          dsimp only
          obtain ⟨htlen, htlo, htregs, htrip, htfl, htinp, htout, hbounds, htrd⟩ :=
            @min_wrBytes_spec 8 { m with rip := (m.rip + len) % X86Min.pow2 64 } t hsim.memLen sp' imm hwr
          have hbytes8_len : (X86Int.bytes8 imm).length = 8 := by simp [X86Int.bytes8]
          have hbytes8_get : ∀ k, (hk : k < 8) →
              (X86Int.bytes8 imm)[k]'(by simp [hbytes8_len, hk]) = imm / 256 ^ k % 256 := by
            intro k hk; simp [X86Int.bytes8]
          have hw_int : ∀ k < (X86Int.bytes8 imm).length, MemWritable i.mem (sp' + k) := by
            intro k hk
            rw [hbytes8_len] at hk
            have hb := hbounds k hk
            exact hsim.stackWritable (sp' + k) hb.1 hb.2
          have hb_int : ∀ b ∈ X86Int.bytes8 imm, b < 256 := by
            intro b hb
            simp only [X86Int.bytes8, List.mem_map] at hb
            obtain ⟨_, _, rfl⟩ := hb
            exact Nat.mod_lt _ (by decide)
          obtain ⟨mem', hwr64, hwritable', hrd'⟩ := mem_wrN_spec i.mem sp' (X86Int.bytes8 imm) hw_int hb_int
          have hwr64' : X86Int.wr64 i.mem sp' imm = some mem' := by
            simpa [X86Int.wr64] using hwr64
          rw [hwr64']
          dsimp only
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, by simpa [X86Min.St.setReg, htrip, hnx_eq] using hclosed, ?_⟩
            have hbase := sim_setReg hsim (by decide : 4 < 16) hsp'W (m.rip + len) hnx_canon hsim.fl
            exact {
              rip := by simp [X86Min.St.setReg, htrip, hnx_eq, hrip_int]
              ripCanon := by simp [X86Min.St.setReg, htrip, hnx_eq, hnx_canon]
              mLen := by simp [X86Min.St.setReg, htregs, hsim.mLen]
              iLen := hbase.iLen
              regsLe := fun r hr => by
                change X86Min.OLe ((t.setReg 4 (some sp')).reg r) ((i.set 4 (some sp')).reg r)
                have htr : t.regs.length = 16 := by rw [htregs]; exact hsim.mLen
                rw [m_reg_setReg htr (by decide : 4 < 16), i_reg_set hsim.iLen (by decide : 4 < 16)]
                by_cases hr4 : r = 4
                · simp [hr4, X86Min.OLe.refl]
                · simp only [hr4, if_false, X86Min.St.reg, htregs]
                  exact hsim.regsLe r hr
              regsDef := hbase.regsDef
              regsBound := hbase.regsBound
              fl := by simpa [X86Min.St.setReg, htfl] using hsim.fl
              memLen := by simpa [X86Min.St.setReg] using htlen
              loCanon := by simpa [X86Min.St.setReg, htlo] using hsim.loCanon
              stackWritable := fun x hx1 hx2 => by
                rw [hwritable']
                exact hsim.stackWritable x (by simpa [X86Min.St.setReg, htlo] using hx1) (by simpa [X86Min.St.setReg, htlo] using hx2)
              memLe := fun x v hv => by
                have hv' : t.rdByte x = some v := by
                  simpa [X86Min.St.setReg, X86Min.St.rdByte] using hv
                rw [htrd] at hv'
                rw [hrd']
                by_cases hx : sp' ≤ x ∧ x < sp' + 8
                · rw [if_pos hx] at hv'
                  have hx' : sp' ≤ x ∧ x < sp' + (X86Int.bytes8 imm).length := by
                    rw [hbytes8_len]; exact hx
                  rw [dif_pos hx']
                  rw [hbytes8_get (x - sp') (by omega)]
                  exact hv'
                · rw [if_neg hx] at hv'
                  have hx' : ¬ (sp' ≤ x ∧ x < sp' + (X86Int.bytes8 imm).length) := by
                    rw [hbytes8_len]; exact hx
                  rw [dif_neg hx']
                  have hv_m : m.rdByte x = some v := by
                    simpa [X86Min.St.rdByte, htlo] using hv'
                  exact hsim.memLe x v hv_m
              memBound := fun x v hv => by
                rw [hrd'] at hv
                split at hv
                · rename_i hsp_x
                  simp only [Option.some.injEq] at hv
                  subst hv
                  rw [hbytes8_get (x - sp') (by rw [hbytes8_len] at hsp_x; omega)]
                  exact Nat.mod_lt _ (by decide)
                · exact hsim.memBound x v hv
              stdin := by simpa [X86Min.St.setReg, htinp] using hsim.stdin
              inpBound := by simpa [X86Min.St.setReg, htinp] using hsim.inpBound
              out := by simpa [X86Min.St.setReg, htout] using hsim.out
              oracle := hsim.oracle }
          · intro code out h; cases h
    | pop d =>
      simp only [CommonIns, commonInsB, decide_eq_true_eq] at hci
      have hd16 : d < 16 := hci
      simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        simp only [hi4, bind, Option.bind, pure]
        cases hle : X86Min.le m.rdByte sp 8 with
        | none => simp
        | some v =>
          have hm_byte_lt : ∀ x u, m.rdByte x = some u → u < 256 := fun x u hu =>
            hsim.memBound x u (hsim.memLe x u hu)
          obtain ⟨bs, hrdN, hleVal, hv_lt⟩ := le_to_rdN_leVal hsim.memLe hm_byte_lt 8 sp v hle
          have hvW : v < W := by simpa [W] using hv_lt
          have hrd64 : X86Int.rd64 i.mem sp = some v := by
            simp [X86Int.rd64, hrdN, hleVal]
          rw [hrd64]
          dsimp only
          have hsp8W : (sp + 8) % W < W := Nat.mod_lt _ (by decide)
          have hsp8_eq : (sp + 8) % X86Min.pow2 64 = (sp + 8) % W := rfl
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
            have h1 := sim_setReg hsim (by decide : 4 < 16) hsp8W (m.rip + len) hnx_canon hsim.fl
            have h2 := sim_setReg h1 hd16 hvW (m.rip + len) hnx_canon hsim.fl
            simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, hsp8_eq] using h2
          · intro code out h; cases h
    | ldRsp8 d =>
      simp only [CommonIns, commonInsB, decide_eq_true_eq] at hci
      have hd16 : d < 16 := hci
      simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        have hea : X86Int.ea sp 0 = sp := by
          simp [X86Int.ea, Nat.mod_eq_of_lt hspW]
        simp only [hi4, hea, bind, Option.bind, pure]
        cases hrd : m.rdByte sp with
        | none => simp
        | some v =>
          have hird := hsim.memLe sp v hrd
          have hv256 := hsim.memBound sp v hird
          have hvW : v < W := Nat.lt_trans hv256 (by decide)
          simp only [hird, X86Min.St.wr, show (32 == 8) = false by decide,
            Bool.false_eq_true, if_false]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
            simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
              sim_setReg hsim hd16 hvW (m.rip + len) hnx_canon hsim.fl
          · intro code out h; cases h
    | stRsp8 src =>
      simp only [CommonIns, commonInsB, decide_eq_true_eq] at hci
      have hs16 : src < 16 := hci
      simp only [toIntIns, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        have hea : X86Int.ea sp 0 = sp := by
          simp [X86Int.ea, Nat.mod_eq_of_lt hspW]
        cases hms : m.reg src with
        | none => simp [X86Min.St.rd, hms]
        | some vs =>
          obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
          simp only [hi4, hea, X86Min.St.rd, pow2_8_eq, hms, his, Option.map_some,
            bind, Option.bind, pure]
          have hmod_idem : (vs % 256) % 256 = vs % 256 := Nat.mod_mod _ _
          by_cases hwin : m.lo ≤ sp ∧ sp < m.lo + X86Min.winSize
          · obtain ⟨t, hwr_m, heq_t, htlen, htrd⟩ :=
              min_wrByte_spec hsim.memLen hwin (v := vs % 256)
            have hwr_m1 : ({ m with rip := (m.rip + len) % X86Min.pow2 64 }).wrByte sp (vs % 256) =
                some { t with rip := (m.rip + len) % X86Min.pow2 64 } := by
              rw [heq_t]; simp [X86Min.St.wrByte, hwin]
            obtain ⟨mem', hwr_i, hwritable', hrd'⟩ :=
              mem_wr_of_writable (v := vs % 256) (hsim.stackWritable sp hwin.1 hwin.2)
            rw [hwr_m1, hwr_i]
            dsimp only
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, by simpa [hnx_eq] using hclosed, ?_⟩
              exact {
                rip := by simp [hnx_eq, hrip_int]
                ripCanon := by simp [hnx_eq, hnx_canon]
                mLen := by simp [heq_t, hsim.mLen]
                iLen := hsim.iLen
                regsLe := by simpa [heq_t] using hsim.regsLe
                regsDef := hsim.regsDef
                regsBound := hsim.regsBound
                fl := by simpa [heq_t] using hsim.fl
                memLen := by simpa [heq_t] using htlen
                loCanon := by simpa [heq_t] using hsim.loCanon
                stackWritable := fun x hx1 hx2 => by
                  rw [hwritable']
                  exact hsim.stackWritable x (by simpa [heq_t] using hx1) (by simpa [heq_t] using hx2)
                memLe := fun x u hu => by
                  have hu' : t.rdByte x = some u := by
                    simpa [heq_t, X86Min.St.rdByte] using hu
                  rw [htrd] at hu'
                  rw [hrd']
                  by_cases hx : x = sp
                  · subst hx; simp [hmod_idem] at hu' ⊢; exact hu'
                  · simp [hx] at hu' ⊢; exact hsim.memLe x u hu'
                memBound := fun x u hu => by
                  rw [hrd'] at hu
                  split at hu
                  · cases hu; exact Nat.mod_lt _ (by decide)
                  · exact hsim.memBound x u hu
                stdin := by simpa [heq_t] using hsim.stdin
                inpBound := by simpa [heq_t] using hsim.inpBound
                out := by simpa [heq_t] using hsim.out
                oracle := hsim.oracle }
            · intro code out h; cases h
          · have hnone : ({ m with rip := (m.rip + len) % X86Min.pow2 64 }).wrByte sp (vs % 256) = none := by
              simp [X86Min.St.wrByte, hwin]
            rw [hnone]
            simp
    | jmp rel =>
      simp only [toIntIns, X86Int.exec]
      rw [pow2_64_eq_W, mod_add_mod_W]
      obtain ⟨hp_t, ht_canon⟩ := hclosed
      constructor
      · intro m' hm'; cases hm'
        exact ⟨_, rfl, hp_t, sim_set_rip hsim ((m.rip + len + rel) % W) ht_canon⟩
      · intro code out h; cases h
    | jcc cc rel =>
      simp only [toIntIns, X86Int.exec, X86Int.lift, bind, Option.bind, pure]
      cases hc : X86Min.cond m.fl cc with
      | none => simp
      | some condb =>
        have hicond := cond_rel hsim.fl cc hc
        rw [hicond]
        obtain ⟨hp_nx, hp_t, ht_canon⟩ := hclosed
        cases condb with
        | false =>
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, by simpa [hnx_eq] using hp_nx, ?_⟩
            simpa [hnx_eq, hrip_int] using sim_set_rip hsim (m.rip + len) hnx_canon
          · intro code out h; cases h
        | true =>
          rw [pow2_64_eq_W, mod_add_mod_W, ← hsim.rip]
          constructor
          · intro m' hm'; cases hm'
            exact ⟨_, rfl, hp_t, sim_set_rip hsim ((m.rip + len + rel) % W) ht_canon⟩
          · intro code out h; cases h
    | syscall =>
      simp only [toIntIns, X86Int.exec]
      cases hm0 : m.reg 0 with
      | none => simp
      | some nr =>
        obtain ⟨hi0, _⟩ := reg_lt16_bound hsim (by decide : 0 < 16) hm0
        obtain ⟨va2, hi2⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 2 (by decide) (by decide))
        obtain ⟨va6, hi6⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 6 (by decide) (by decide))
        obtain ⟨va7, hi7⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 7 (by decide) (by decide))
        obtain ⟨va8, hi8⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 8 (by decide) (by decide))
        obtain ⟨va9, hi9⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 9 (by decide) (by decide))
        obtain ⟨va10, hi10⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 10 (by decide) (by decide))
        rw [hi0, hi7, hi6, hi2, hi10, hi8, hi9]
        dsimp only
        by_cases hnr0 : nr = 0
        · subst hnr0
          simp only [X86Min.okOf]
          cases hm7 : m.reg 7 with
          | none => simp
          | some r7 =>
            cases hm2 : m.reg 2 with
            | none => simp
            | some r2 =>
              by_cases hargs : r7 = 0 ∧ r2 = 1
              · obtain ⟨rfl, rfl⟩ := hargs
                have hi7_0 := hsim.regsLe 7 (by decide) 0 hm7
                have hi2_1 := hsim.regsLe 2 (by decide) 1 hm2
                rw [hi7_0] at hi7; simp only [Option.some.injEq] at hi7; subst hi7
                rw [hi2_1] at hi2; simp only [Option.some.injEq] at hi2; subst hi2
                cases hm6 : m.reg 6 with
                | none => simp
                | some ad =>
                  obtain ⟨hi6_ad, _⟩ := reg_lt16_bound hsim (by decide : 6 < 16) hm6
                  rw [hi6_ad] at hi6; simp only [Option.some.injEq] at hi6; subst va6
                  cases hinp : m.inp with
                  | nil =>
                    have hik : i.k.stdin = [] := by rw [hsim.stdin, hinp]
                    have hio : i.k.oracle = [] := hsim.oracle
                    have hsys : X86Env.syscall i.k i.mem 0 0 ad 1 va10 va8 va9 =
                        some (.ret 0 i.k i.mem) := by
                      have hkeq : ({ i.k with stdin := ([] : List Nat), oracle := ([] : List Choice) } : Kern) = i.k := by
                        cases hkk : i.k
                        simp_all
                      simp only [X86Env.syscall, if_true, hik, List.length_nil, Kern.count, Kern.next,
                        hsim.oracle, Nat.min_zero, List.take_nil, Mem.wrN, List.drop_nil, bind, Option.bind, pure]
                      exact congrArg (fun k => some (SysRes.ret 0 k i.mem)) hkeq
                    simp only [show (some 0 != some 0 || some 1 != some 1) = false by decide,
                      Bool.false_eq_true, if_false, bind, Option.bind, pure]
                    rw [hsys]
                    dsimp only
                    constructor
                    · intro m' hm'; cases hm'
                      refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
                      exact {
                        rip := by simp [X86Min.St.setReg, hnx_eq, hrip_int]
                        ripCanon := by simp [X86Min.St.setReg, hnx_eq, hnx_canon]
                        mLen := by simp [X86Min.St.setReg, hsim.mLen]
                        iLen := by simp [X86Int.St.set, hsim.iLen]
                        regsLe := fun r hr => by
                          change X86Min.OLe _ ((((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).reg r)
                          have hlen0 : ({ m with rip := (m.rip + len) % X86Min.pow2 64, inp := ([] : List Nat) } : X86Min.St).regs.length = 16 :=
                            hsim.mLen
                          have hilen1 : (i.set 0 (some 0)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                          have hilen2 : ((i.set 0 (some 0)).set 1 (some (i.rip + len))).regs.length = 16 := by
                            simp [X86Int.St.set, hsim.iLen]
                          rw [m_reg_setReg (by simp [X86Min.St.setReg, hsim.mLen]) (by decide : 0 < 16),
                              m_reg_setReg (by simp [X86Min.St.setReg, hsim.mLen]) (by decide : 11 < 16),
                              m_reg_setReg hlen0 (by decide : 1 < 16),
                              i_reg_set hilen2 (by decide : 11 < 16),
                              i_reg_set hilen1 (by decide : 1 < 16),
                              i_reg_set hsim.iLen (by decide : 0 < 16)]
                          by_cases hr11 : r = 11
                          · subst hr11; simp [X86Min.OLe.none]
                          · by_cases hr0 : r = 0
                            · subst hr0; simp [X86Min.OLe.refl]
                            · by_cases hr1 : r = 1
                              · subst hr1; simp [hnx_eq, hrip_int, X86Min.OLe.refl]
                              · simp [hr11, hr0, hr1]; exact hsim.regsLe r hr
                        regsDef := fun r hr hr11 => by
                          change ((((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).reg r).isSome = true
                          have hilen1 : (i.set 0 (some 0)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                          have hilen2 : ((i.set 0 (some 0)).set 1 (some (i.rip + len))).regs.length = 16 := by
                            simp [X86Int.St.set, hsim.iLen]
                          rw [i_reg_set hilen2 (by decide : 11 < 16),
                              i_reg_set hilen1 (by decide : 1 < 16),
                              i_reg_set hsim.iLen (by decide : 0 < 16)]
                          by_cases hr11' : r = 11
                          · exact absurd hr11' hr11
                          · by_cases hr1 : r = 1
                            · subst hr1; simp
                            · by_cases hr0 : r = 0
                              · subst hr0; simp
                              · simp [hr11', hr1, hr0]; exact hsim.regsDef r hr hr11
                        regsBound := fun r u hu => by
                          change (((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).reg r = some u at hu
                          by_cases hr16 : r < 16
                          · have hilen1 : (i.set 0 (some 0)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 0)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [i_reg_set hilen2 (by decide : 11 < 16),
                                i_reg_set hilen1 (by decide : 1 < 16),
                                i_reg_set hsim.iLen (by decide : 0 < 16)] at hu
                            by_cases hr11 : r = 11
                            · simp [hr11] at hu
                            · by_cases hr1 : r = 1
                              · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hnx_lt
                              · by_cases hr0 : r = 0
                                · subst hr0; simp at hu; subst hu; decide
                                · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                          · have hlen : (((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            have hnone : (((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                              List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                            simp [X86Int.St.reg, hnone] at hu
                        fl := by simpa [X86Min.St.setReg] using hsim.fl
                        memLen := by simpa [X86Min.St.setReg] using hsim.memLen
                        loCanon := by simpa [X86Min.St.setReg] using hsim.loCanon
                        stackWritable := fun x hx1 hx2 =>
                          hsim.stackWritable x (by simpa [X86Min.St.setReg] using hx1) (by simpa [X86Min.St.setReg] using hx2)
                        memLe := fun x => by simpa [X86Min.St.setReg, X86Min.St.rdByte] using hsim.memLe x
                        memBound := hsim.memBound
                        stdin := by simpa [X86Min.St.setReg, hinp] using hsim.stdin
                        inpBound := by simp [X86Min.St.setReg]
                        out := by simpa [X86Min.St.setReg] using hsim.out
                        oracle := hsim.oracle }
                    · intro code out h; cases h
                  | cons c rest =>
                    have hc_lt : c < 256 := hsim.inpBound c (by simp [hinp])
                    have hc_mod : c % 256 = c := Nat.mod_eq_of_lt hc_lt
                    have hik : i.k.stdin = c :: rest := by rw [hsim.stdin, hinp]
                    simp only [show (some 0 != some 0 || some 1 != some 1) = false by decide,
                      Bool.false_eq_true, if_false, bind, Option.bind, pure]
                    by_cases hwin : m.lo ≤ ad ∧ ad < m.lo + X86Min.winSize
                    · let s2 := ({ m with rip := (m.rip + len) % X86Min.pow2 64, inp := c :: rest }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none
                      have hs2_len : s2.mem.length = X86Min.winSize := by simp [s2, X86Min.St.setReg, hsim.memLen]
                      have hs2_lo : s2.lo = m.lo := rfl
                      obtain ⟨t, hwr_m, heq_t, htlen, htrd⟩ := min_wrByte_spec hs2_len (v := c) (by simpa [hs2_lo] using hwin)
                      obtain ⟨mem', hwr_i, hwritable', hrd'⟩ := mem_wr_of_writable (v := c) (hsim.stackWritable ad hwin.1 hwin.2)
                      have hcnt' : Kern.count i.k 1 (c :: rest).length = (1, i.k) := by
                        simp [Kern.count, Kern.next, hsim.oracle]
                      have hsys : X86Env.syscall i.k i.mem 0 0 ad 1 va10 va8 va9 =
                          some (.ret 1 { i.k with stdin := rest } mem') := by
                        simp only [X86Env.syscall, if_true, hik, hcnt', List.take, List.drop,
                          Mem.wrN, hwr_i, bind, Option.bind, pure]
                      have hwr_m' : (({ m with rip := (m.rip + len) % X86Min.pow2 64, inp := c :: rest }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).wrByte ad c = some t :=
                        hwr_m
                      rw [hwr_m', hsys]
                      dsimp only
                      constructor
                      · intro m' hm'; cases hm'
                        refine ⟨_, rfl, by simpa [X86Min.St.setReg, heq_t, s2, hnx_eq] using hclosed, ?_⟩
                        exact {
                          rip := by simp [X86Min.St.setReg, heq_t, s2, hnx_eq, hrip_int]
                          ripCanon := by simp [X86Min.St.setReg, heq_t, s2, hnx_eq, hnx_canon]
                          mLen := by simp [X86Min.St.setReg, heq_t, s2, hsim.mLen]
                          iLen := by simp [X86Int.St.set, hsim.iLen]
                          regsLe := fun r hr => by
                            change X86Min.OLe (({ t with inp := rest }.setReg 0 (some 1)).reg r) _
                            have ht_rlen : ({ t with inp := rest } : X86Min.St).regs.length = 16 := by
                              simp [heq_t, s2, X86Min.St.setReg, hsim.mLen]
                            have hlen1 : ({ m with rip := (m.rip + len) % X86Min.pow2 64, inp := c :: rest }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).regs.length = 16 := by
                              simp [X86Min.St.setReg, hsim.mLen]
                            have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [m_reg_setReg ht_rlen (by decide : 0 < 16)]
                            have htreg_eq : ({ t with inp := rest } : X86Min.St).reg r = s2.reg r := by
                              simp [X86Min.St.reg, heq_t]
                            rw [htreg_eq]
                            simp only [s2]
                            change X86Min.OLe _ ((((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r)
                            rw [m_reg_setReg hlen1 (by decide : 11 < 16),
                                m_reg_setReg (by simp [hsim.mLen]) (by decide : 1 < 16),
                                i_reg_set hilen2 (by decide : 11 < 16),
                                i_reg_set hilen1 (by decide : 1 < 16),
                                i_reg_set hsim.iLen (by decide : 0 < 16)]
                            by_cases hr0 : r = 0
                            · subst hr0; simp [X86Min.OLe.refl]
                            · by_cases hr11 : r = 11
                              · subst hr11; simp [X86Min.OLe.none]
                              · by_cases hr1 : r = 1
                                · subst hr1; simp [hnx_eq, hrip_int, X86Min.OLe.refl]
                                · simp [hr0, hr11, hr1]; exact hsim.regsLe r hr
                          regsDef := fun r hr hr11 => by
                            change ((((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r).isSome = true
                            have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [i_reg_set hilen2 (by decide : 11 < 16),
                                i_reg_set hilen1 (by decide : 1 < 16),
                                i_reg_set hsim.iLen (by decide : 0 < 16)]
                            by_cases hr11' : r = 11
                            · exact absurd hr11' hr11
                            · by_cases hr1 : r = 1
                              · subst hr1; simp
                              · by_cases hr0 : r = 0
                                · subst hr0; simp
                                · simp [hr11', hr1, hr0]; exact hsim.regsDef r hr hr11
                          regsBound := fun r u hu => by
                            change (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r = some u at hu
                            by_cases hr16 : r < 16
                            · have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                              have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              rw [i_reg_set hilen2 (by decide : 11 < 16),
                                  i_reg_set hilen1 (by decide : 1 < 16),
                                  i_reg_set hsim.iLen (by decide : 0 < 16)] at hu
                              by_cases hr11 : r = 11
                              · simp [hr11] at hu
                              · by_cases hr1 : r = 1
                                · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hnx_lt
                                · by_cases hr0 : r = 0
                                  · subst hr0; simp at hu; subst hu; decide
                                  · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                            · have hlen : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              have hnone : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                                List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                              simp [X86Int.St.reg, hnone] at hu
                          fl := by simpa [X86Min.St.setReg, heq_t, s2] using hsim.fl
                          memLen := by simpa [X86Min.St.setReg] using htlen
                          loCanon := by simpa [X86Min.St.setReg, heq_t, s2] using hsim.loCanon
                          stackWritable := fun x hx1 hx2 => by
                            rw [hwritable']
                            exact hsim.stackWritable x (by simpa [X86Min.St.setReg, heq_t, s2] using hx1) (by simpa [X86Min.St.setReg, heq_t, s2] using hx2)
                          memLe := fun x u hu => by
                            have hu' : t.rdByte x = some u := by
                              simpa [X86Min.St.setReg, X86Min.St.rdByte] using hu
                            rw [htrd] at hu'
                            rw [hrd']
                            by_cases hx : x = ad
                            · subst hx; simp [hc_mod] at hu' ⊢; exact hu'
                            · simp only [hx, if_false] at hu' ⊢
                              have hu_m : m.rdByte x = some u := by
                                simpa [s2, X86Min.St.setReg, X86Min.St.rdByte] using hu'
                              exact hsim.memLe x u hu_m
                          memBound := fun x u hu => by
                            rw [hrd'] at hu
                            split at hu
                            · cases hu; exact Nat.mod_lt _ (by decide)
                            · exact hsim.memBound x u hu
                          stdin := rfl
                          inpBound := fun x hx => by
                            simp only [X86Min.St.setReg] at hx
                            exact hsim.inpBound x (by simp [hinp, hx])
                          out := by simpa [X86Min.St.setReg, heq_t, s2] using hsim.out
                          oracle := hsim.oracle }
                      · intro code out h; cases h
                    · have hnone : (({ m with rip := (m.rip + len) % X86Min.pow2 64, inp := c :: rest }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).wrByte ad c = none := by
                        simp [X86Min.St.wrByte, X86Min.St.setReg, hwin]
                      rw [hnone]
                      simp
              · have hbad : (some r7 != some 0 || some r2 != some 1) = true := by
                  simp; revert hargs; omega
                simp [hbad]
        · by_cases hnr1 : nr = 1
          · subst hnr1
            simp only [X86Min.okOf]
            cases hm7 : m.reg 7 with
            | none => simp
            | some r7 =>
              cases hm2 : m.reg 2 with
              | none => simp
              | some r2 =>
                by_cases hargs : r7 = 1 ∧ r2 = 1
                · obtain ⟨rfl, rfl⟩ := hargs
                  have hi7_1 := hsim.regsLe 7 (by decide) 1 hm7
                  have hi2_1 := hsim.regsLe 2 (by decide) 1 hm2
                  rw [hi7_1] at hi7; simp only [Option.some.injEq] at hi7; subst hi7
                  rw [hi2_1] at hi2; simp only [Option.some.injEq] at hi2; subst hi2
                  cases hm6 : m.reg 6 with
                  | none => simp
                  | some ad =>
                    obtain ⟨hi6_ad, _⟩ := reg_lt16_bound hsim (by decide : 6 < 16) hm6
                    rw [hi6_ad] at hi6; simp only [Option.some.injEq] at hi6; subst va6
                    simp only [show (some 1 != some 1 || some 1 != some 1) = false by decide,
                      Bool.false_eq_true, if_false, bind, Option.bind, pure]
                    cases hrd : m.rdByte ad with
                    | none => simp
                    | some c =>
                      have hird := hsim.memLe ad c hrd
                      have hcnt : Kern.count i.k 1 1 = (1, i.k) := by
                        simp [Kern.count, Kern.next, hsim.oracle]
                      have hrdN : Mem.rdN i.mem ad 1 = some [c] := by
                        simp [Mem.rdN, hird]
                      have hsys : X86Env.syscall i.k i.mem 1 1 ad 1 va10 va8 va9 =
                          some (.ret 1 { i.k with out := i.k.out ++ [c] } i.mem) := by
                        simp [X86Env.syscall, hcnt, hrdN]
                      rw [hsys]
                      dsimp only
                      constructor
                      · intro m' hm'; cases hm'
                        refine ⟨_, rfl, by simpa [X86Min.St.setReg, hnx_eq] using hclosed, ?_⟩
                        exact {
                          rip := by simp [X86Min.St.setReg, hnx_eq, hrip_int]
                          ripCanon := by simp [X86Min.St.setReg, hnx_eq, hnx_canon]
                          mLen := by simp [X86Min.St.setReg, hsim.mLen]
                          iLen := by simp [X86Int.St.set, hsim.iLen]
                          regsLe := fun r hr => by
                            change X86Min.OLe _ ((((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r)
                            have hlen0 : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).regs.length = 16 := hsim.mLen
                            have hlen1 : ({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).regs.length = 16 := by
                              simp [X86Min.St.setReg, hsim.mLen]
                            have hlen2 : (({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).regs.length = 16 := by
                              simp [X86Min.St.setReg, hsim.mLen]
                            have hlen3 : ({ ({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none with out := c :: m.out } : X86Min.St).regs.length = 16 := by
                              simp [X86Min.St.setReg, hsim.mLen]
                            have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [m_reg_setReg hlen3 (by decide : 0 < 16)]
                            have htmp : ({ ({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none with out := c :: m.out } : X86Min.St).reg r =
                                (({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).reg r := rfl
                            rw [htmp, m_reg_setReg hlen1 (by decide : 11 < 16),
                                m_reg_setReg hlen0 (by decide : 1 < 16),
                                i_reg_set hilen2 (by decide : 11 < 16),
                                i_reg_set hilen1 (by decide : 1 < 16),
                                i_reg_set hsim.iLen (by decide : 0 < 16)]
                            by_cases hr0 : r = 0
                            · subst hr0; simp [X86Min.OLe.refl]
                            · by_cases hr11 : r = 11
                              · subst hr11; simp [X86Min.OLe.none]
                              · by_cases hr1 : r = 1
                                · subst hr1; simp [hnx_eq, hrip_int, X86Min.OLe.refl]
                                · simp [hr0, hr11, hr1]; exact hsim.regsLe r hr
                          regsDef := fun r hr hr11 => by
                            change ((((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r).isSome = true
                            have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [i_reg_set hilen2 (by decide : 11 < 16),
                                i_reg_set hilen1 (by decide : 1 < 16),
                                i_reg_set hsim.iLen (by decide : 0 < 16)]
                            by_cases hr11' : r = 11
                            · exact absurd hr11' hr11
                            · by_cases hr1 : r = 1
                              · subst hr1; simp
                              · by_cases hr0 : r = 0
                                · subst hr0; simp
                                · simp [hr11', hr1, hr0]; exact hsim.regsDef r hr hr11
                          regsBound := fun r u hu => by
                            change (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r = some u at hu
                            by_cases hr16 : r < 16
                            · have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                              have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              rw [i_reg_set hilen2 (by decide : 11 < 16),
                                  i_reg_set hilen1 (by decide : 1 < 16),
                                  i_reg_set hsim.iLen (by decide : 0 < 16)] at hu
                              by_cases hr11 : r = 11
                              · simp [hr11] at hu
                              · by_cases hr1 : r = 1
                                · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hnx_lt
                                · by_cases hr0 : r = 0
                                  · subst hr0; simp at hu; subst hu; decide
                                  · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                            · have hlen : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              have hnone : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                                List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                              simp [X86Int.St.reg, hnone] at hu
                          fl := by simpa [X86Min.St.setReg] using hsim.fl
                          memLen := by simpa [X86Min.St.setReg] using hsim.memLen
                          loCanon := by simpa [X86Min.St.setReg] using hsim.loCanon
                          stackWritable := fun x hx1 hx2 =>
                            hsim.stackWritable x (by simpa [X86Min.St.setReg] using hx1) (by simpa [X86Min.St.setReg] using hx2)
                          memLe := fun x => by simpa [X86Min.St.setReg, X86Min.St.rdByte] using hsim.memLe x
                          memBound := hsim.memBound
                          stdin := by simpa [X86Min.St.setReg] using hsim.stdin
                          inpBound := by simpa [X86Min.St.setReg] using hsim.inpBound
                          out := by simp [X86Min.St.setReg, hsim.out]
                          oracle := hsim.oracle }
                      · intro code out h; cases h
                · have hbad : (some r7 != some 1 || some r2 != some 1) = true := by
                    simp; revert hargs; omega
                  simp [hbad]
          · by_cases hnr60 : nr = 60
            · subst hnr60
              cases hm7 : m.reg 7 with
              | none => simp
              | some st =>
                have hi7_st := hsim.regsLe 7 (by decide) st hm7
                rw [hi7_st] at hi7; simp only [Option.some.injEq] at hi7; subst va7
                have hsys : X86Env.syscall i.k i.mem 60 st va6 va2 va10 va8 va9 =
                    some (.exit (st % 256)) := by
                  simp [X86Env.syscall]
                rw [hsys]
                simp only [hsim.out]
                constructor
                · intro m' h; cases h
                · intro code out h
                  simp only [Sum.inr.injEq, X86Min.Outcome.exit.injEq] at h
                  obtain ⟨rfl, rfl⟩ := h
                  exact ⟨rfl, rfl⟩
            · simp [hnr0, hnr1, hnr60]

theorem int_run_exit (im : Nat → Option (X86Int.Ins × Nat)) (n c : Nat) :
    X86Int.run im n (.exit c) = .exit c := by
  cases n <;> rfl

/-- Multi-step forward simulation over `runS` and `run` for arbitrary `fuel : Nat`. -/
theorem runS_sim {P : Nat → Prop} {imMin : X86Min.Img}
    {imInt : Nat → Option (X86Int.Ins × Nat)}
    (hic : ImgCompat P imMin imInt) :
    ∀ (fuel : Nat) {m : X86Min.St} {i : X86Int.St}, P m.rip → Sim m i →
      (∀ m', X86Min.runS imMin fuel m = .inl m' →
        ∃ i', X86Int.run imInt fuel (.ok i) = .ok i' ∧ P m'.rip ∧ Sim m' i') ∧
      (∀ code out, X86Min.run imMin fuel m = .exit code out →
        X86Int.run imInt fuel (.ok i) = .exit code ∧
        ∃ k < fuel, ∃ mf if_,
          X86Min.runS imMin k m = .inl mf ∧
          X86Int.run imInt k (.ok i) = .ok if_ ∧
          P mf.rip ∧ Sim mf if_ ∧
          X86Min.step imMin mf = .inr (.exit code out) ∧
          X86Int.step imInt if_ = .exit code ∧
          if_.k.out = out) := by
  intro fuel
  induction fuel with
  | zero =>
    intro m i hp hsim
    constructor
    · intro m' hm'
      change Sum.inl m = Sum.inl m' at hm'
      cases hm'
      exact ⟨i, rfl, hp, hsim⟩
    · intro code out h
      change X86Min.Outcome.outOfFuel = X86Min.Outcome.exit code out at h
      cases h
  | succ fuel ih =>
    intro m i hp hsim
    have hs := step_sim hic hp hsim
    cases hst : X86Min.step imMin m with
    | inl m1 =>
      obtain ⟨i1, hi1, hp1, hsim1⟩ := hs.1 m1 hst
      have ih1 := ih hp1 hsim1
      constructor
      · intro m' hm'
        rw [X86Min.runS_succ, hst] at hm'
        simpa [X86Int.run, hi1] using ih1.1 m' hm'
      · intro code out h
        rw [X86Min.run_succ, hst] at h
        obtain ⟨hrun_int, k, hk, mf, if_, hrunS_mf, hrun_if, hpmf, hsim_f, hstep_mf, hstep_if, hout_if⟩ :=
          ih1.2 code out h
        refine ⟨?_, k + 1, Nat.succ_lt_succ hk, mf, if_, ?_, ?_, hpmf, hsim_f, hstep_mf, hstep_if, hout_if⟩
        · simp [X86Int.run, hi1, hrun_int]
        · rw [X86Min.runS_succ, hst]; exact hrunS_mf
        · simp [X86Int.run, hi1, hrun_if]
    | inr o =>
      constructor
      · intro m' hm'
        rw [X86Min.runS_succ, hst] at hm'
        cases hm'
      · intro code out h
        rw [X86Min.run_succ, hst] at h
        cases h
        obtain ⟨hint, hout⟩ := hs.2 code out hst
        refine ⟨?_, 0, Nat.zero_lt_succ fuel, m, i, rfl, rfl, hp, hsim, hst, hint, hout⟩
        simp [X86Int.run, hint, int_run_exit]

/-- Multi-step forward simulation (`run_sim`): whenever `X86Min.run` exits with
`(code, out)` in `fuel` steps, `X86Int.run` exits with `code` and reaches an exit
step `k < fuel` on the trace from `(m, i)` whose kernel `out` matches `out`. -/
theorem run_sim {P : Nat → Prop} {imMin : X86Min.Img}
    {imInt : Nat → Option (X86Int.Ins × Nat)}
    (hic : ImgCompat P imMin imInt)
    {fuel : Nat} {m : X86Min.St} {i : X86Int.St}
    (hp : P m.rip) (hsim : Sim m i)
    {code : Nat} {out : List Nat}
    (hrun : X86Min.run imMin fuel m = .exit code out) :
    X86Int.run imInt fuel (.ok i) = .exit code ∧
    ∃ k < fuel, ∃ mf if_,
      X86Min.runS imMin k m = .inl mf ∧
      X86Int.run imInt k (.ok i) = .ok if_ ∧
      P mf.rip ∧ Sim mf if_ ∧
      X86Min.step imMin mf = .inr (.exit code out) ∧
      X86Int.step imInt if_ = .exit code ∧
      if_.k.out = out :=
  (runS_sim hic fuel hp hsim).2 code out hrun

/-- Lift `run_sim` to the physical semantics `X86Int.prun` via `X86Int.faultFree_prun`. -/
theorem prun_sim {P : Nat → Prop} {imMin : X86Min.Img}
    {imInt : Nat → Option (X86Int.Ins × Nat)}
    (hic : ImgCompat P imMin imInt)
    (alts : X86Int.Alts) (r : X86Int.Rates) (hr : r.ok)
    {fuel : Nat} {m : X86Min.St} {i : X86Int.St}
    (hp : P m.rip) (hsim : Sim m i)
    {code : Nat} {out : List Nat}
    (hrun : X86Min.run imMin fuel m = .exit code out) :
    (X86Int.prun alts (r.state hr) r.fetch imInt fuel (.ok i)).ok ∧
    PDist.fm (X86Int.prun alts (r.state hr) r.fetch imInt fuel (.ok i)) ≤ fuel * r.total ∧
    PDist.faultFree (X86Int.prun alts (r.state hr) r.fetch imInt fuel (.ok i)) = [X86Int.Out.exit code] ∧
    ∃ k < fuel, ∃ if_,
      X86Int.run imInt k (.ok i) = .ok if_ ∧
      PDist.faultFree (X86Int.prun alts (r.state hr) r.fetch imInt k (.ok i)) = [X86Int.Out.ok if_] ∧
      if_.k.out = out ∧
      X86Int.step imInt if_ = .exit code := by
  obtain ⟨hrun_int, k, hk, _, if_, _, hrun_if, _, _, _, hstep_if, hout_if⟩ := run_sim hic hp hsim hrun
  refine ⟨X86Int.prun_ok alts (r.state hr) hr.1 imInt fuel (.ok i), ?_, ?_, k, hk, if_, hrun_if, ?_, hout_if, hstep_if⟩
  · simpa [X86Int.Rates.total, X86Int.Rates.state, X86Int.Fault.seq, X86Int.Fault.uniform] using
      X86Int.fm_prun alts (r.state hr) hr.1 imInt fuel (.ok i)
  · rw [X86Int.faultFree_prun, hrun_int]
  · rw [X86Int.faultFree_prun, hrun_if]

/-! ### Deliverable 5: Kernel-Checked End-to-End ELF Handoff Witness -/

/-- A 38-byte common-ISA byte-stream loader program (`0x400078..0x40009e`):
reads bytes from stdin one by one into `[rsp]`, writes them to stdout, and exits 0 on EOF. -/
def loaderCode : List Nat :=
  [ 0x31, 0xc0                     -- 0x78: xor eax, eax
  , 0x31, 0xff                     -- 0x7a: xor edi, edi
  , 0x48, 0x89, 0xe6               -- 0x7c: mov rsi, rsp
  , 0x6a, 0x01                     -- 0x7f: push 1
  , 0x5a                           -- 0x81: pop rdx
  , 0x0f, 0x05                     -- 0x82: syscall (read(0, rsp, 1))
  , 0x48, 0x85, 0xc0               -- 0x84: test rax, rax
  , 0x74, 0x0e                     -- 0x87: je 0x400097
  , 0x6a, 0x01                     -- 0x89: push 1
  , 0x58                           -- 0x8b: pop rax
  , 0x89, 0xc7                     -- 0x8c: mov edi, eax
  , 0x48, 0x89, 0xe6               -- 0x8e: mov rsi, rsp
  , 0x89, 0xc2                     -- 0x91: mov edx, eax
  , 0x0f, 0x05                     -- 0x93: syscall (write(1, rsp, 1))
  , 0xeb, 0xe1                     -- 0x95: jmp 0x400078
  , 0x6a, 0x3c                     -- 0x97: push 60
  , 0x58                           -- 0x99: pop rax
  , 0x31, 0xff                     -- 0x9a: xor edi, edi
  , 0x0f, 0x05                     -- 0x9c: syscall (exit(0))
  ]

/-- Minimal 158-byte ELF64 executable wrapping `loaderCode` at `0x400000` (entry `0x400078`). -/
def loaderElf : List Nat :=
  -- ELF64 header (64 bytes)
  [ 0x7f, 0x45, 0x4c, 0x46         -- e_ident magic
  , 0x02, 0x01, 0x01, 0x00         -- ELF64, little-endian, version 1
  , 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  , 0x02, 0x00                     -- e_type = ET_EXEC
  , 0x3e, 0x00                     -- e_machine = EM_X86_64
  , 0x01, 0x00, 0x00, 0x00         -- e_version = 1
  , 0x78, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00 -- e_entry = 0x400078
  , 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- e_phoff = 0x40
  , 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- e_shoff = 0
  , 0x00, 0x00, 0x00, 0x00         -- e_flags = 0
  , 0x40, 0x00                     -- e_ehsize = 64
  , 0x38, 0x00                     -- e_phentsize = 56
  , 0x01, 0x00                     -- e_phnum = 1
  , 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
  -- Program header (56 bytes)
  , 0x01, 0x00, 0x00, 0x00         -- p_type = PT_LOAD
  , 0x05, 0x00, 0x00, 0x00         -- p_flags = PF_R | PF_X
  , 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_offset = 0
  , 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_vaddr = 0x400000
  , 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_paddr = 0x400000
  , 0x9e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_filesz = 158
  , 0x9e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_memsz = 158
  , 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 -- p_align = 0x1000
  ] ++ loaderCode

/-- Reachable instruction start addresses of `loaderElf` at `0x400000` (plus image end `0x40009e`). -/
def loaderAddrs : List Nat :=
  [ 0x400078, 0x40007a, 0x40007c, 0x40007f, 0x400081, 0x400082
  , 0x400084, 0x400087, 0x400089, 0x40008b, 0x40008c, 0x40008e
  , 0x400091, 0x400093, 0x400095, 0x400097, 0x400099, 0x40009a, 0x40009c
  , 0x40009e ]

/-- Stage-1 machine code using `X86Dec`-only instructions (`add rdi, 40` and `push rdi`):
pops `argc` from `[rsp]` into `rdi`, adds 40, pushes and pops `rdi`, and exits with `argc + 40` (= 42). -/
def stage1Code : List Nat :=
  [ 0x5f                           -- pop rdi (argc from initStack)
  , 0x48, 0x83, 0xc7, 0x28         -- add rdi, 40 (X86Dec-only aluI)
  , 0x57                           -- push rdi    (X86Dec-only push r64)
  , 0x5f                           -- pop rdi
  , 0xb8, 0x3c, 0x00, 0x00, 0x00   -- mov eax, 60
  , 0x0f, 0x05                     -- syscall
  ]

def loaderImMin : X86Min.Img := X86Min.Img.mk' loaderElf 0x400000 loaderElf.length
def loaderImInt : Nat → Option (X86Int.Ins × Nat) := X86Dec.imageOf loaderElf 0x400000

def loaderStMin : X86Min.St :=
  { regs := (List.replicate 16 none).set 4 (some 0x7ffffffde000)
  , rip := 0x400078
  , fl := X86Min.Flags.undef
  , mem := List.replicate X86Min.winSize none
  , lo := 0x7ffffffde000 - X86Min.winSize / 2
  , inp := stage1Code
  , out := [] }

def loaderStkPair : Region × Nat := initStack 0x7ffffffdf000 []

def loaderStInt : X86Int.St :=
  { regs := (List.replicate 16 (some 0)).set 4 (some loaderStkPair.2)
  , rip := 0x400078
  , fl := X86Int.Flags.undef
  , mem := [⟨0x400000, loaderElf, false⟩, loaderStkPair.1]
  , k := ⟨[], [], stage1Code, [], []⟩ }

def stage1StkPair : Region × Nat := initStack 0x7ffffffff000 [[115, 49], [47, 112]]

def stage1Im : Nat → Option (X86Int.Ins × Nat) := X86Dec.imageOf stage1Code 0x400000

def stage1StInt : X86Int.St :=
  { regs := (List.replicate 16 (some 0)).set 4 (some stage1StkPair.2)
  , rip := 0x400000
  , fl := X86Int.Flags.undef
  , mem := [⟨0x400000, stage1Code, false⟩, stage1StkPair.1]
  , k := ⟨[], [], [], [], []⟩ }

theorem loader_load_ok :
    X86Min.load loaderElf 0x7ffffffde000 stage1Code = some (loaderImMin, loaderStMin) := by
  decide +kernel

theorem loader_compat_ok :
    checkImgCompat loaderAddrs loaderImMin loaderImInt = true := by
  decide +kernel

theorem loader_exec_ok :
    X86Min.exec 300 loaderElf stage1Code = .exit 0 stage1Code := by
  decide +kernel

theorem loader_run_ok :
    X86Min.run loaderImMin 300 loaderStMin = .exit 0 stage1Code := by
  rw [← X86Min.exec_of_load loader_load_ok]
  exact loader_exec_ok

theorem loader_sim_entry : Sim loaderStMin loaderStInt where
  rip := rfl
  ripCanon := by decide +kernel
  mLen := by decide +kernel
  iLen := by decide +kernel
  regsLe r hr := by
    change X86Min.OLe (loaderStMin.reg r) (loaderStInt.reg r)
    have hchk : (List.range 16).all (fun k =>
        match loaderStMin.reg k with
        | none => true
        | some v => loaderStInt.reg k == some v) = true := by decide +kernel
    have hk := List.all_eq_true.1 hchk r (List.mem_range.2 hr)
    intro v hv
    rw [hv] at hk
    simpa [beq_iff_eq] using hk
  regsDef r hr hr11 := by
    have hchk : (List.range 16).all (fun k => (loaderStInt.reg k).isSome) = true := by decide +kernel
    exact List.all_eq_true.1 hchk r (List.mem_range.2 hr)
  regsBound r v hv := by
    by_cases hr16 : r < 16
    · have hchk : (List.range 16).all (fun k =>
          match loaderStInt.reg k with
          | none => true
          | some u => u < W) = true := by decide +kernel
      have hk := List.all_eq_true.1 hchk r (List.mem_range.2 hr16)
      rw [hv] at hk; simpa [decide_eq_true_eq] using hk
    · have hlen : loaderStInt.regs.length = 16 := by decide +kernel
      have hnone : loaderStInt.regs[r]? = none :=
        List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
      simp [X86Int.St.reg, hnone] at hv
  fl := ⟨X86Min.OLe.none _, X86Min.OLe.none _, X86Min.OLe.none _, X86Min.OLe.none _⟩
  memLen := by decide +kernel
  loCanon := by decide +kernel
  stackWritable x hx1 hx2 := by
    have hlo : loaderStMin.lo = 0x7ffffffddff0 := by decide +kernel
    have hidx_sp : loaderStkPair.1.base = 0x7ffffffdd000 ∧ loaderStkPair.1.hi = 0x7ffffffdf000 ∧ loaderStkPair.1.rw = true := by
      decide +kernel
    have hcode_hi : 0x400000 + loaderElf.length < 0x7ffffffddff0 := by decide +kernel
    have hx_ge : 0x7ffffffddff0 ≤ x := by rw [hlo] at hx1; exact hx1
    have hx_lt : x < 0x7ffffffde010 := by
      have hw : X86Min.winSize = 32 := rfl
      rw [hlo, hw] at hx2; exact hx2
    have hnot_code : ¬ (0x400000 ≤ x ∧ x < (⟨0x400000, loaderElf, false⟩ : Region).hi) := by
      simp only [Region.hi] at *; omega
    have hin_stk : loaderStkPair.1.base ≤ x ∧ x < loaderStkPair.1.hi := by omega
    refine ⟨1, loaderStkPair.1, ?_, hidx_sp.2.2⟩
    simp [loaderStInt, Mem.find, hnot_code, hin_stk]
  memLe x v hv := by
    have hnone : loaderStMin.rdByte x = none := by
      simp only [loaderStMin, X86Min.St.rdByte]
      split
      · simp [List.getElem?_replicate]
        split <;> rfl
      · rfl
    rw [hnone] at hv; cases hv
  memBound x v hv := by
    have hfind : ∀ a b, Mem.find loaderStInt.mem x 0 = some (a, b) → ∀ y ∈ b.data, y < 256 := by
      intro a b hfb y hy
      have h0 : ∀ y ∈ loaderElf, y < 256 := by decide +kernel
      have h1 : ∀ y ∈ loaderStkPair.1.data, y < 256 := by decide +kernel
      simp only [loaderStInt, Mem.find] at hfb
      split at hfb
      · simp at hfb; obtain ⟨_, rfl⟩ := hfb; exact h0 y hy
      · split at hfb
        · simp at hfb; obtain ⟨_, rfl⟩ := hfb; exact h1 y hy
        · simp at hfb
    simp only [Mem.rd, bind, Option.bind] at hv
    cases hf : Mem.find loaderStInt.mem x 0 with
    | none => rw [hf] at hv; cases hv
    | some p =>
      rw [hf] at hv
      exact hfind p.1 p.2 hf v (List.mem_of_getElem? hv)
  stdin := rfl
  inpBound c hc := by
    have h : ∀ x ∈ stage1Code, x < 256 := by decide +kernel
    exact h c hc
  out := rfl
  oracle := rfl

theorem eq_exit_of_code_eq {o : X86Int.Out} {c : Nat} (h : o.code = some c) :
    o = .exit c := by
  cases o with
  | ok _ => simp [X86Int.Out.code] at h
  | exit _ => simp [X86Int.Out.code] at h; subst h; rfl
  | stuck => simp [X86Int.Out.code] at h

theorem stage1_run_ok :
    X86Int.run stage1Im 10 (.ok stage1StInt) = .exit 42 := by
  apply eq_exit_of_code_eq
  decide +kernel

/-- End-to-end kernel-checked ELF handoff witness (`seed_to_stage1_bridge_witness`):
1. `X86Min.exec` runs `loaderElf` on `stage1Code` and exits `0` emitting `stage1Code`.
2. `X86Int.run` and the fault-free branch of `X86Int.prun` agree on the loader run
   (exiting `0` with identical stdout `stage1Code`).
3. Loading the emitted bytes `stage1Code` with `X86Dec.imageOf` and `X86Env.initStack`
   executes the stage-1-only instructions and exits with status `42`. -/
theorem seed_to_stage1_bridge_witness (alts : X86Int.Alts) (r : X86Int.Rates) (hr : r.ok) :
    X86Min.exec 300 loaderElf stage1Code = .exit 0 stage1Code ∧
    X86Int.run loaderImInt 300 (.ok loaderStInt) = .exit 0 ∧
    PDist.faultFree (X86Int.prun alts (r.state hr) r.fetch loaderImInt 300 (.ok loaderStInt)) = [.exit 0] ∧
    (∃ k < 300, ∃ if_,
      X86Int.run loaderImInt k (.ok loaderStInt) = .ok if_ ∧
      PDist.faultFree (X86Int.prun alts (r.state hr) r.fetch loaderImInt k (.ok loaderStInt)) = [.ok if_] ∧
      if_.k.out = stage1Code ∧
      X86Dec.imageOf if_.k.out 0x400000 = stage1Im ∧
      X86Int.step loaderImInt if_ = .exit 0) ∧
    X86Int.run stage1Im 10 (.ok stage1StInt) = .exit 42 := by
  have hic := imgCompat_of_check loader_compat_ok
  have hp0 : loaderStMin.rip ∈ loaderAddrs := by decide +kernel
  have hrun := run_sim hic hp0 loader_sim_entry loader_run_ok
  obtain ⟨_, _, hff, k, hk, if_, hrun_if, hprun_if, hout_if, hstep_if⟩ := prun_sim hic alts r hr hp0 loader_sim_entry loader_run_ok
  exact ⟨loader_exec_ok, hrun.1, hff,
    ⟨k, hk, if_, hrun_if, hprun_if, hout_if, by rw [hout_if]; rfl, hstep_if⟩,
    stage1_run_ok⟩

end X86Bridge
