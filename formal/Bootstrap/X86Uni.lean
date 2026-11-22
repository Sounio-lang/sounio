/-
X86Uni: a unified x86-64 machine model conservatively extending `X86Int` + `X86Dec`
with the five narrow instruction forms used by `seed-a` and `seed-b`.

`X86Uni.Ins` embeds every instruction of `X86Int.Ins` via `ofInt`, with proved
equality of architectural (`ofInt_exec`, `ofInt_step`) and physical (`ofInt_pstep`,
`ofInt_prun`) semantics, and adds:
  1. `AL` 8-bit immediate ops (`04` add, `2c` sub, `3c` cmp), preserving bits 8..63
     of `rax` and setting 8-bit CF/ZF/SF/OF.
  2. 32-bit register-register ops (`31` xor, `09` or, `85` test, `89` mov),
     zero-extending the destination to 64 bits and setting 32-bit flags.
  3. 32-bit `cmp r32, imm` (`81 /7`, `83 /7`).
  4. 32-bit `shl r32, imm8` (`c1 /4`), masking count mod 32, zero-extending even on
     count 0 while preserving flags, setting all 4 flags on count 1, and leaving
     `OF = none` on count > 1.
  5. `inc r32/r64` (`ff /0`), preserving `CF`, setting `ZF/SF/OF`, and
     zero-extending at width 32.

This closes the gap between `X86Min` and `X86Int` into a single physical machine
reusing `X86Int.{St,Flags,Out,Fault,Rates}`, `X86Env`, and `PDist`, with
unrestricted forward simulation from `X86Min` (`decode_min_agree`, `step_min_sim`,
`run_min_sim`, `exec_min_sim`, `prun_min_sim`) and kernel-checked runs of the real
committed `X86Min.Data.seedA` (271B) and `X86Min.Data.seedB` (283B).
-/
import Bootstrap.X86Min
import Bootstrap.X86Int
import Bootstrap.X86Dec
import Bootstrap.X86Env
import Bootstrap.PDist
import Bootstrap.Refine
import Bootstrap.SeedData

namespace X86Uni

open X86Env PDist

/-- `simpa [..] using t` closed by `exact` at default transparency. Lean 4.33 checks the
closing step of `simpa` at reducible transparency, which no longer sees through
`X86Min.St.setReg` or `X86Min.pow2 64`. -/
local macro "simpaX " "[" ls:Lean.Parser.Tactic.simpLemma,* "]" " using " t:term : tactic =>
  `(tactic| first
    | exact $t
    | (have hX := $t; simp [$ls,*] at hX ⊢; exact hX)
    | (have hX := $t; simp [$ls,*] at hX; exact hX)
    | (simp [$ls,*]; exact $t))

abbrev Reg := X86Int.Reg
abbrev Flags := X86Int.Flags
abbrev St := X86Int.St
abbrev Out := X86Int.Out
abbrev Fault := X86Int.Fault
abbrev Rates := X86Int.Rates

/-! ### Instruction set and `ofInt` embedding -/

inductive NarrowAlu8 where
  | add | sub | cmp
deriving DecidableEq, Repr

inductive NarrowRR32 where
  | xor | or | test | mov | add | sub
deriving DecidableEq, Repr

inductive Ins where
  /-- Any instruction of `X86Int.Ins` (all 45+ forms). -/
  | ofInt (ins : X86Int.Ins)
  /-- `op al, imm8` (`04` add, `2c` sub, `3c` cmp): preserves upper 56 bits of `rax`. -/
  | alImm8 (op : NarrowAlu8) (imm : Nat)
  /-- 32-bit `op r32, r32` (`31` xor, `09` or, `85` test, `89` mov, `01` add, `29` sub). -/
  | rr32 (op : NarrowRR32) (d s : Reg)
  /-- 32-bit `cmp r32, imm` (`81 /7`, `83 /7`). -/
  | cmpI32 (d imm : Nat)
  /-- 32-bit `add`/`sub r32, imm` (`81 /0`, `81 /5`, `83 /0`, `83 /5`): the destination is
  written zero-extended. `op` is never `cmp` (that is `cmpI32`). -/
  | aluI32 (op : NarrowAlu8) (d imm : Nat)
  /-- 32-bit `shl r32, imm8` (`c1 /4`). -/
  | shl32 (d c : Nat)
  /-- `inc r32/r64` (`ff /0`). -/
  | inc (w d : Nat)
deriving DecidableEq, Repr

abbrev ofInt : X86Int.Ins → Ins := Ins.ofInt

abbrev Alts := Nat → List (Option (Ins × Nat))

def ofIntImage (im : Nat → Option (X86Int.Ins × Nat)) : Nat → Option (Ins × Nat) :=
  fun a => (im a).map fun (ins, len) => (ofInt ins, len)

def ofIntAlts (alts : X86Int.Alts) : Alts :=
  fun a => (alts a).map fun o => o.map fun (ins, len) => (ofInt ins, len)

/-! ### Architectural semantics -/

/-- Most significant bit of a `w`-bit word. -/
def msbW (w v : Nat) : Bool := v / 2 ^ (w - 1) % 2 == 1

inductive AluOp where
  | add | sub | cmp | xor | or | test
deriving DecidableEq, Repr

def NarrowAlu8.toAluOp : NarrowAlu8 → AluOp
  | .add => .add
  | .sub => .sub
  | .cmp => .cmp

/-- Result and flags of an arithmetic or logical operation at width `w`. -/
def aluW (op : AluOp) (w a b : Nat) : Nat × Flags :=
  let m := 2 ^ w
  let res := match op with
    | .add => (a + b) % m
    | .sub | .cmp => (a + m - b) % m
    | .xor => Nat.xor a b
    | .or => Nat.lor a b
    | .test => Nat.land a b
  let cf := match op with
    | .add => m ≤ a + b
    | .sub | .cmp => a < b
    | _ => false
  let ofl := match op with
    | .add => msbW w a == msbW w b && msbW w res != msbW w a
    | .sub | .cmp => msbW w a != msbW w b && msbW w res != msbW w a
    | _ => false
  (res, ⟨some cf, some (res == 0), some (msbW w res), some ofl⟩)

/-- Execute one instruction `ins` of length `len` at `s.rip`. -/
def exec (ins : Ins) (len : Nat) (s : St) : Out :=
  let nx := s.rip + len
  let go (t : St) : St := { t with rip := nx }
  match ins with
  | .ofInt i => X86Int.exec i len s
  | .alImm8 op imm => X86Int.lift do
      let rax ← s.reg 0
      let a := rax % 256
      let (res, f) := aluW op.toAluOp 8 a (imm % 256)
      let t := go { s with fl := f }
      match op with
      | .cmp => pure t
      | .add | .sub => pure (t.set 0 (some (rax - a + res)))
  | .rr32 op d r => X86Int.lift do
      match op with
      | .mov =>
        let v ← s.reg r
        pure (go (s.set d (some (v % 2 ^ 32))))
      | .xor =>
        let zeroing := d == r
        let a ← if zeroing then some 0 else (s.reg d).map (· % 2 ^ 32)
        let b ← if zeroing then some 0 else (s.reg r).map (· % 2 ^ 32)
        let (res, f) := aluW .xor 32 a b
        pure (go { s.set d (some res) with fl := f })
      | .or =>
        let a ← s.reg d
        let b ← s.reg r
        let (res, f) := aluW .or 32 (a % 2 ^ 32) (b % 2 ^ 32)
        pure (go { s.set d (some res) with fl := f })
      | .test =>
        let a ← s.reg d
        let b ← s.reg r
        let (_, f) := aluW .test 32 (a % 2 ^ 32) (b % 2 ^ 32)
        pure (go { s with fl := f })
      | .add =>
        let a ← s.reg d
        let b ← s.reg r
        let (res, f) := aluW .add 32 (a % 2 ^ 32) (b % 2 ^ 32)
        pure (go { s.set d (some res) with fl := f })
      | .sub =>
        let a ← s.reg d
        let b ← s.reg r
        let (res, f) := aluW .sub 32 (a % 2 ^ 32) (b % 2 ^ 32)
        pure (go { s.set d (some res) with fl := f })
  | .cmpI32 d imm => X86Int.lift do
      let a ← s.reg d
      let (_, f) := aluW .cmp 32 (a % 2 ^ 32) (imm % 2 ^ 32)
      pure (go { s with fl := f })
  | .aluI32 op d imm => X86Int.lift do
      let a ← s.reg d
      let (res, f) := aluW op.toAluOp 32 (a % 2 ^ 32) (imm % 2 ^ 32)
      match op with
      | .cmp => pure (go { s with fl := f })
      | .add | .sub => pure (go { s.set d (some res) with fl := f })
  | .shl32 d c => X86Int.lift do
      let a ← (s.reg d).map (· % 2 ^ 32)
      let cnt := c % 32
      if cnt == 0 then
        pure (go (s.set d (some a)))
      else
        let res := a * 2 ^ cnt % 2 ^ 32
        let cf := a / 2 ^ (32 - cnt) % 2 == 1
        let ofl := if cnt == 1 then some (msbW 32 res != cf) else none
        pure (go { s.set d (some res) with fl := ⟨some cf, some (res == 0), some (msbW 32 res), ofl⟩ })
  | .inc w d => X86Int.lift do
      let a ← (s.reg d).map (· % 2 ^ w)
      let res := (a + 1) % 2 ^ w
      let f' : Flags := { s.fl with zf := some (res == 0), sf := some (msbW w res),
                                    of := some (res == 2 ^ (w - 1)) }
      pure (go { s.set d (some res) with fl := f' })

/-- The architectural step. -/
def step (im : Nat → Option (Ins × Nat)) (s : St) : Out :=
  match im s.rip with
  | none => .stuck
  | some (ins, len) => exec ins len s

/-- The architectural run: at most `n` steps. -/
def run (im : Nat → Option (Ins × Nat)) : Nat → Out → Out
  | 0, o => o
  | n + 1, .ok s => run im n (step im s)
  | _ + 1, o => o

/-! ### Decoder -/

/-- The 32-bit `op r32, imm` instruction: `cmp` is `cmpI32`, `add` and `sub` are `aluI32`. -/
def narrowRI (op : NarrowAlu8) (d imm : Nat) : Ins :=
  match op with
  | .cmp => .cmpI32 d imm
  | op => .aluI32 op d imm

/-- Decode the narrow `X86Min` forms omitted by `X86Dec.decode` (8-bit AL arithmetic, 32-bit
register and immediate ALU forms, 32-bit `shl`, `inc`), and `jmp r64` (`ff /4`, mod 11). -/
def decodeNarrow (b : Nat → Option Nat) : Option (Ins × Nat) := do
  let b0 ← b 0
  let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
  let rex := if hasRex then b0 - 0x40 else 0
  let p := if hasRex then 1 else 0
  let w := rex / 8 % 2 == 1
  let r := (rex / 4 % 2) * 8
  let x := rex / 2 % 2 == 1
  let bb := (rex % 2) * 8
  let wd := if w then 64 else 32
  let o ← b p
  if o == 0x04 || o == 0x3c || o == 0x2c then
    if hasRex then none else
    let i ← b 1
    let op := if o == 0x04 then NarrowAlu8.add else if o == 0x3c then NarrowAlu8.cmp else NarrowAlu8.sub
    pure (.alImm8 op i, 2)
  else if o == 0x01 || o == 0x29 || o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89 then
    if w || x then none else
    let m ← b (p + 1)
    if m / 64 != 3 then none else
    let d := m % 8 + bb
    let s := m / 8 % 8 + r
    let op := if o == 0x01 then NarrowRR32.add else if o == 0x29 then NarrowRR32.sub
      else if o == 0x31 then NarrowRR32.xor else if o == 0x09 then NarrowRR32.or
      else if o == 0x85 then NarrowRR32.test else NarrowRR32.mov
    pure (.rr32 op d s, p + 2)
  else if o == 0xc1 then
    if w || r != 0 || x then none else
    let m ← b (p + 1)
    let c ← b (p + 2)
    if m / 64 != 3 || m / 8 % 8 != 4 then none else
    pure (.shl32 (m % 8 + bb) c, p + 3)
  else if o == 0x81 || o == 0x83 then
    if w || r != 0 || x then none else
    let m ← b (p + 1)
    let f := m / 8 % 8
    if m / 64 != 3 || (f != 0 && f != 5 && f != 7) then none else
    let d := m % 8 + bb
    let op : NarrowAlu8 := if f == 0 then .add else if f == 5 then .sub else .cmp
    if o == 0x81 then
      let i ← X86Dec.leB b (p + 2) 4
      pure (narrowRI op d (X86Dec.sx32 i % 2 ^ 32), p + 6)
    else
      let i ← b (p + 2)
      pure (narrowRI op d (X86Dec.sx8 i % 2 ^ 32), p + 3)
  else if o == 0xff then
    if r != 0 || x then none else
    let m ← b (p + 1)
    if m / 64 != 3 then none else
    if m / 8 % 8 == 0 then pure (.inc wd (m % 8 + bb), p + 2) else
    if m / 8 % 8 == 4 then pure (.ofInt (.jmpR (m % 8 + bb)), p + 2) else
    none
  else none

/-- Unified byte decoder: tries `X86Dec.decode` first, then `decodeNarrow`. -/
def decode (b : Nat → Option Nat) (a : Nat) : Option (Ins × Nat) :=
  match X86Dec.decode b a with
  | some (ins, len) => some (.ofInt ins, len)
  | none => decodeNarrow b

def altsAt (b : Nat → Option Nat) (a len : Nat) : List (Option (Ins × Nat)) :=
  (List.range (8 * len)).map fun k => decode (X86Dec.flipBit b k) a

def imageOf (code : List Nat) (base : Nat) : Nat → Option (Ins × Nat) :=
  fun a => if a < base then none else decode (fun i => code[a - base + i]?) a

def imageAlts (code : List Nat) (base : Nat) : Alts :=
  fun a =>
    if a < base then []
    else match imageOf code base a with
      | some (_, len) => altsAt (fun i => code[a - base + i]?) a len
      | none => []

theorem decode_of_dec {b : Nat → Option Nat} {a : Nat} {ins : X86Int.Ins} {len : Nat}
    (h : X86Dec.decode b a = some (ins, len)) :
    decode b a = some (ofInt ins, len) := by
  simp [decode, h]

theorem imageOf_of_dec {code : List Nat} {base a : Nat} {ins : X86Int.Ins} {len : Nat}
    (h : X86Dec.imageOf code base a = some (ins, len)) :
    imageOf code base a = some (ofInt ins, len) := by
  unfold X86Dec.imageOf at h
  unfold imageOf
  split at h
  · cases h
  · rename_i hlt
    rw [if_neg hlt]
    exact decode_of_dec h

/-! ### Physical semantics and conservative `ofInt` embedding -/

def execO (s : St) : Option (Ins × Nat) → Out
  | some (j, l) => exec j l s
  | none => .stuck

def pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) (s : St) : PDF Out :=
  let fetched : PDF Out := match im s.rip with
    | none => pureF .stuck
    | some (ins, len) => X86Int.unif pf (exec ins len s) ((alts s.rip).map (execO s))
  bindF fetched fun o => match o with
    | .ok s' => X86Int.relabel X86Int.Out.ok (F.hit s')
    | o => pureF o

def prun (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) : Nat → Out → PDF Out
  | 0, o => pureF o
  | n + 1, .ok s => bindF (pstep alts F pf im s) (prun alts F pf im n)
  | _ + 1, o => pureF o

theorem ofInt_exec (ins : X86Int.Ins) (len : Nat) (s : St) :
    exec (ofInt ins) len s = X86Int.exec ins len s := rfl

theorem ofInt_step (im : Nat → Option (X86Int.Ins × Nat)) (s : St) :
    step (ofIntImage im) s = X86Int.step im s := by
  unfold step X86Int.step ofIntImage
  cases im s.rip with
  | none => rfl
  | some p => rfl

private theorem execO_ofIntAlts (alts : X86Int.Alts) (s : St) (a : Nat) :
    (ofIntAlts alts a).map (execO s) = (alts a).map (X86Int.execO s) := by
  unfold ofIntAlts
  rw [List.map_map]
  congr 1
  funext o
  cases o with
  | none => rfl
  | some p => rfl

theorem ofInt_pstep (alts : X86Int.Alts) (F : Fault) (pf : Rat)
    (im : Nat → Option (X86Int.Ins × Nat)) (s : St) :
    pstep (ofIntAlts alts) F pf (ofIntImage im) s = X86Int.pstep alts F pf im s := by
  unfold pstep X86Int.pstep ofIntImage
  cases im s.rip with
  | none => rfl
  | some p =>
    obtain ⟨ins, len⟩ := p
    dsimp only [Option.map]
    rw [execO_ofIntAlts]; rfl

theorem ofInt_prun (alts : X86Int.Alts) (F : Fault) (pf : Rat)
    (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ n o, prun (ofIntAlts alts) F pf (ofIntImage im) n o = X86Int.prun alts F pf im n o := by
  intro n
  induction n with
  | zero => intro o; rfl
  | succ n ih =>
    intro o
    cases o with
    | ok s =>
      simp only [prun, X86Int.prun, ofInt_pstep]
      congr 1
      funext x
      exact ih x
    | exit c => rfl
    | stuck => rfl

theorem faultFree_pstep (alts : Alts) (F : Fault) (pf : Rat)
    (im : Nat → Option (Ins × Nat)) (s : St) :
    faultFree (pstep alts F pf im s) = [step im s] := by
  unfold pstep step
  rw [faultFree_bindF]
  cases im s.rip with
  | none => simp [faultFree_pureF]
  | some p =>
    obtain ⟨ins, len⟩ := p
    simp only [X86Int.faultFree_unif, List.flatMap_cons, List.flatMap_nil, List.append_nil]
    exact X86Int.retire_clean F (exec ins len s)

private theorem fetched_ok {pf : Rat} (hpf : Rate pf) (alts : Alts)
    (im : Nat → Option (Ins × Nat)) (s : St) :
    (match im s.rip with
      | none => pureF X86Int.Out.stuck
      | some (ins, len) => X86Int.unif pf (exec ins len s) ((alts s.rip).map (execO s))).ok := by
  cases im s.rip with
  | none => exact pureF_ok _
  | some p => obtain ⟨ins, len⟩ := p; exact X86Int.unif_ok hpf _ _

private theorem fetched_le {pf : Rat} (hpf : Rate pf) (alts : Alts)
    (im : Nat → Option (Ins × Nat)) (s : St) :
    fm (match im s.rip with
      | none => pureF X86Int.Out.stuck
      | some (ins, len) => X86Int.unif pf (exec ins len s) ((alts s.rip).map (execO s))) ≤ pf := by
  cases im s.rip with
  | none => rw [fm_pureF]; exact hpf.1
  | some p => obtain ⟨ins, len⟩ := p; exact X86Int.fm_unif_le hpf _ _

theorem pstep_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (s : St) : (pstep alts F pf im s).ok :=
  bindF_ok (fetched_ok hpf alts im s) (X86Int.retire_ok F)

theorem fm_pstep (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (s : St) : fm (pstep alts F pf im s) ≤ pf + F.q :=
  Rat.le_trans (fm_bindF_le' (fetched_ok hpf alts im s) F.q0 (X86Int.retire_ok F) (X86Int.retire_le F))
    (add_le_add' (fetched_le hpf alts im s) Rat.le_refl)

theorem faultFree_prun (alts : Alts) (F : Fault) (pf : Rat)
    (im : Nat → Option (Ins × Nat)) :
    ∀ n o, faultFree (prun alts F pf im n o) = [run im n o] := by
  intro n
  induction n with
  | zero => intro o; rfl
  | succ n ih =>
    intro o
    cases o with
    | ok s =>
      simp only [prun, run]
      rw [faultFree_bindF, faultFree_pstep]
      simp [ih]
    | exit c => rfl
    | stuck => rfl

theorem prun_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) :
    ∀ n o, (prun alts F pf im n o).ok := by
  intro n
  induction n with
  | zero => intro o; exact pureF_ok _
  | succ n ih =>
    intro o
    cases o with
    | ok s => exact bindF_ok (pstep_ok alts F hpf im s) ih
    | exit c => exact pureF_ok _
    | stuck => exact pureF_ok _

theorem fm_prun (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) :
    ∀ n o, fm (prun alts F pf im n o) ≤ n * (pf + F.q) := by
  have h0 : 0 ≤ pf + F.q := X86Int.nonneg_add hpf.1 F.q0
  intro n
  induction n with
  | zero =>
    intro o; rw [prun, fm_pureF]
    show (0 : Rat) ≤ ((0 : Nat) : Rat) * _
    rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]; exact Rat.le_refl
  | succ n ih =>
    intro o
    have hn : ((n + 1 : Nat) : Rat) * (pf + F.q) = (pf + F.q) + n * (pf + F.q) := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    have hnn : 0 ≤ (n : Rat) * (pf + F.q) := Rat.mul_nonneg Rat.natCast_nonneg h0
    cases o with
    | ok s =>
      simp only [prun]
      rw [hn]
      exact Rat.le_trans (fm_bindF_le' (pstep_ok alts F hpf im s) hnn (prun_ok alts F hpf im n) ih)
        (add_le_add' (fm_pstep alts F hpf im s) Rat.le_refl)
    | exit c =>
      show fm (pureF (X86Int.Out.exit c)) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)
    | stuck =>
      show fm (pureF X86Int.Out.stuck) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)

theorem baseline (alts : Alts) (im : Nat → Option (Ins × Nat)) (cf cr cm cl : Iv)
    (r : Rates) (hr : r.ok) (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem)
    (hl : cl.mem r.flag) (n : Nat) (o : Out) :
    fm (prun alts (r.state hr) r.fetch im n o) ≤ n * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  refine Rat.le_trans (fm_prun alts (r.state hr) hr.1 im n o) ?_
  refine Rat.mul_le_mul_of_nonneg_left ?_ Rat.natCast_nonneg
  show r.fetch + (r.reg + (r.mem + r.flag)) ≤ _
  exact add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))

/-! ### Arithmetic and byte helpers for `X86Min -> X86Uni` simulation -/

theorem pow2_64_eq_W : X86Min.pow2 64 = W := rfl
theorem pow2_32_eq : X86Min.pow2 32 = 2 ^ 32 := rfl
theorem pow2_8_eq : X86Min.pow2 8 = 256 := rfl

theorem msb_64_eq (v : Nat) : X86Min.msb 64 v = X86Int.msb v := rfl
theorem msbW_eq (w v : Nat) : msbW w v = X86Min.msb w v := rfl

theorem sx8_eq (v : Nat) : X86Dec.sx8 v = X86Min.sx8 v := rfl
theorem sx32_eq (v : Nat) : X86Dec.sx32 v = X86Min.sx32 v := rfl

theorem leB_eq_le (b : Nat → Option Nat) (off n : Nat) :
    X86Dec.leB b off n = X86Min.le b off n := by
  induction n generalizing off with
  | zero => rfl
  | succ n ih => simp [X86Dec.leB, X86Min.le, ih]

theorem sx8_lt_W {v : Nat} (hv : v < 256) : X86Min.sx8 v < W := by
  unfold X86Min.sx8 W; rw [pow2_64_eq_W]; unfold W; split <;> omega

theorem sx32_lt_W {v : Nat} (hv : v < 2 ^ 32) : X86Min.sx32 v < W := by
  unfold X86Min.sx32 W; rw [pow2_64_eq_W, pow2_32_eq]; unfold W; split <;> omega

theorem le_lt_pow256 {b : Nat → Option Nat}
    (hb : ∀ i v, b i = some v → v < 256) :
    ∀ {off n v : Nat}, X86Min.le b off n = some v → v < 256 ^ n := by
  intro off n
  induction n generalizing off with
  | zero => intro v h; simp [X86Min.le] at h; subst h; decide
  | succ n ih =>
    intro v h
    simp only [X86Min.le, bind, Option.bind, pure] at h
    cases hbo : b off with
    | none => rw [hbo] at h; cases h
    | some lo =>
      rw [hbo] at h
      cases hhi : X86Min.le b (off + 1) n with
      | none => rw [hhi] at h; cases h
      | some hi =>
        rw [hhi] at h; simp only [Option.some.injEq] at h; subst h
        have hlo := hb off lo hbo
        have hlt := ih hhi
        rw [Nat.pow_succ']; omega

theorem mod_add_mod_W (a len rel : Nat) : ((a + len) % W + rel) % W = (a + len + rel) % W := by
  rw [Nat.add_mod ((a + len) % W) rel W, Nat.mod_mod, ← Nat.add_mod]

/-! ### Unrestricted `X86Min -> X86Uni` instruction translation & decoder agreement -/

/-- Translate any decoded `X86Min.Ins` at `(a, len)` into `X86Uni.Ins`. -/
def toUniIns (a len : Nat) : X86Min.Ins → Ins
  | .movImm d imm => .ofInt (.movI d imm)
  | .rr .xor 64 d src => .ofInt (.alu .xor d src)
  | .rr .xor 32 d src => if d == src then .ofInt (.alu .xor d d) else .rr32 .xor d src
  | .rr .xor _ d src => .rr32 .xor d src
  | .rr .or 64 d src => .ofInt (.alu .or d src)
  | .rr .or _ d src => .rr32 .or d src
  | .rr .test 64 d src => .ofInt (.test d src)
  | .rr .test _ d src => .rr32 .test d src
  | .rr .mov 64 d src => .ofInt (.mov d src)
  | .rr .mov _ d src => .ofInt (.mov32 d src)
  | .rr .add 64 d src => .ofInt (.alu .add d src)
  | .rr .add _ d src => .rr32 .add d src
  | .rr .sub 64 d src => .ofInt (.alu .sub d src)
  | .rr .sub _ d src => .rr32 .sub d src
  | .rr _ _ d src => .ofInt (.mov d src)
  | .ri .cmp 64 d imm => .ofInt (.cmpI d imm)
  | .ri .cmp 32 d imm => .cmpI32 d imm
  | .ri .add 64 d imm => .ofInt (.aluI .add d imm)
  | .ri .sub 64 d imm => .ofInt (.aluI .sub d imm)
  | .ri .add 32 d imm => .aluI32 .add d imm
  | .ri .sub 32 d imm => .aluI32 .sub d imm
  | .ri .add _ _ imm => .alImm8 .add imm
  | .ri .sub _ _ imm => .alImm8 .sub imm
  | .ri .cmp _ _ imm => .alImm8 .cmp imm
  | .ri _ _ d imm => .cmpI32 d imm
  | .shlI _ d c => .shl32 d c
  | .inc w d => .inc w d
  | .push imm => .ofInt (.pushI imm)
  | .pop d => .ofInt (.pop d)
  | .ldRsp8 d => .ofInt (.ld8 d 4 0)
  | .stRsp8 src => .ofInt (.st8 4 0 src)
  | .jcc cc rel => .ofInt (.jcc cc ((a + len + rel) % W))
  | .jmp rel => .ofInt (.jmp ((a + len + rel) % W))
  | .jmpReg r => .ofInt (.jmpR r)
  | .syscall => .ofInt .syscall

/-- Structural well-formedness satisfied by every instruction produced by `X86Min.decode`
on real bytes `< 256`. -/
def InsWf : X86Min.Ins → Prop
  | .movImm d imm => d < 16 ∧ imm < 2 ^ 32
  | .rr op w d src =>
    d < 16 ∧ src < 16 ∧ (w = 32 ∨ w = 64) ∧
    (op = .xor ∨ op = .or ∨ op = .test ∨ op = .mov ∨ op = .add ∨ op = .sub)
  | .ri op w d imm =>
    (op = .cmp ∧ (w = 32 ∨ w = 64) ∧ d < 16 ∧ imm < X86Min.pow2 w) ∨
    ((op = .add ∨ op = .sub ∨ op = .cmp) ∧ w = 8 ∧ d = 0 ∧ imm < 256) ∨
    ((op = .add ∨ op = .sub) ∧ (w = 32 ∨ w = 64) ∧ d < 16 ∧ imm < X86Min.pow2 w)
  | .shlI w d c => w = 32 ∧ d < 16 ∧ c < 256
  | .inc w d => (w = 32 ∨ w = 64) ∧ d < 16
  | .push imm => imm < W
  | .pop d => d < 16
  | .ldRsp8 d => d < 16
  | .stRsp8 src => src < 16
  | .jcc _ rel => rel < W
  | .jmp rel => rel < W
  | .jmpReg r => r < 16
  | .syscall => True

private theorem rex_parts (b0 : Nat) :
    let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
    let rex := if hasRex then b0 - 0x40 else 0
    let p := if hasRex then 1 else 0
    let r := (rex / 4 % 2) * 8
    let bb := (rex % 2) * 8
    (p = 0 ∨ p = 1) ∧ (r = 0 ∨ r = 8) ∧ (bb = 0 ∨ bb = 8) := by
  dsimp only
  refine ⟨by split <;> omega, by omega, by omega⟩

private theorem toUniIns_xor32 (a len d src : Nat) :
    toUniIns a len (X86Min.Ins.rr X86Min.Op.xor 32 d src) =
      if (d == src) = true then Ins.ofInt (X86Int.Ins.alu X86Int.Alu.xor d d) else Ins.rr32 NarrowRR32.xor d src := rfl

private theorem if_none_eq_some {α : Type _} {c : Bool} {t : Option α} {x : α}
    (h : (if c = true then none else t) = some x) : c ≠ true ∧ t = some x := by
  cases c <;> simp_all

set_option maxHeartbeats 1600000 in
private theorem decode_min_agree_aux {b : Nat → Option Nat} {a b0 o : Nat}
    (hb : ∀ i v, b i = some v → v < 256)
    (hb0 : b 0 = some b0)
    (hbp : b (if 0x40 ≤ b0 && b0 ≤ 0x4f then 1 else 0) = some o)
    {ins : X86Min.Ins} {len : Nat} (hm : X86Min.decode b = some (ins, len)) :
    decode b a = some (toUniIns a len ins, len) ∧ len ≤ 7 ∧ InsWf ins := by
  have hb0_lt := hb 0 b0 hb0
  have ho_lt := hb _ o hbp
  have ⟨hp01, hr08, hbb08⟩ := rex_parts b0
  let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
  let rex := if hasRex then b0 - 0x40 else 0
  let p := if hasRex then 1 else 0
  let w := rex / 8 % 2 == 1
  let r := (rex / 4 % 2) * 8
  let x := rex / 2 % 2 == 1
  let bb := (rex % 2) * 8
  let wd := if w then 64 else 32
  unfold X86Min.decode at hm
  rw [hb0] at hm
  simp only [Bind.bind, Option.bind] at hm
  rw [hbp] at hm
  dsimp only at hm
  -- Split along the 13 opcode branches of X86Min.decode without invoking mega-`split`
  by_cases hmovI : (0xb8 ≤ o && o ≤ 0xbf) = true
  · -- 0xb8..0xbf: movImm
    rw [if_pos hmovI] at hm
    obtain ⟨hbad, hm⟩ := if_none_eq_some hm
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
    obtain ⟨⟨hwf, hr0⟩, hxf⟩ := hbad
    cases hle : X86Min.le b (p + 1) 4 with
    | none => rw [hle] at hm; cases hm
    | some imm =>
      rw [hle] at hm
      simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have himm_lt := le_lt_pow256 hb hle
      have himm_lt32 : imm < 2 ^ 32 := by simpa using himm_lt
      have hdec : X86Dec.decode b a = some (.movI (o - 0xb8 + bb) imm, p + 5) := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
        have hrx : ¬((r != 0 || x) = true) := by
          intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
          rcases h with h | h; exact h hr0; exact hxf h
        rw [if_pos hmovI, if_neg hrx, if_neg hwf, leB_eq_le, hle]
        simp only [pure]; rfl
      refine ⟨decode_of_dec hdec, by omega, ?_⟩
      simp only [InsWf]
      have hmovb : 0xb8 ≤ o ∧ o ≤ 0xbf := by simpa [Bool.and_eq_true, decide_eq_true_eq] using hmovI
      omega
  rw [if_neg hmovI] at hm
  by_cases hrr6 : (o == 0x01 || o == 0x29 || o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89) = true
  · -- 0x01 || 0x29 || 0x31 || 0x09 || 0x85 || 0x89: rr
    rw [if_pos hrr6] at hm
    by_cases hnew : (o == 0x01 || o == 0x29) = true
    · -- 0x01 add, 0x29 sub: 64-bit through `X86Dec` (`alu`), 32-bit through `decodeNarrow` (`rr32`)
      have hnew' : o = 0x01 ∨ o = 0x29 := by simpa [Bool.or_eq_true, beq_iff_eq] using hnew
      cases hm_b : b (p + 1) with
      | none => rw [hm_b] at hm; cases hm
      | some m =>
        rw [hm_b] at hm
        dsimp only at hm
        have hm_lt := hb (p + 1) m hm_b
        obtain ⟨hbad, hm⟩ := if_none_eq_some hm
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
        obtain ⟨hm64, hxf⟩ := hbad
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        have hd16 : m % 8 + bb < 16 := by omega
        have hs16 : m / 8 % 8 + r < 16 := by omega
        have hwd : wd = 32 ∨ wd = 64 := by dsimp [wd]; cases w <;> simp
        have hmx : ¬((m / 64 != 3 || x) = true) := by
          intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
          rcases h with h | h; exact h hm64; exact hxf h
        have hmn : ¬((m / 64 != 3) = true) := by
          intro h; simp only [bne_iff_ne, ne_eq] at h; exact h hm64
        refine ⟨?_, by omega, ⟨hd16, hs16, hwd, by rcases hnew' with rfl | rfl <;> simp⟩⟩
        have hw' : (w = true) ↔ ((if (0x40 ≤ b0 && b0 ≤ 0x4f) then b0 - 0x40 else 0) / 8 % 2 = 1) := by
          simp [w, rex, hasRex]
        cases hw : w
        · -- 32-bit: `X86Dec` refuses, `decodeNarrow` gives `rr32`
          have hwd32 : wd = 32 := by simp [wd, hw]
          have hw0 : (if (0x40 ≤ b0 && b0 ≤ 0x4f) then b0 - 0x40 else 0) / 8 % 2 ≠ 1 := by
            intro h; have := hw'.mpr h; rw [hw] at this; cases this
          have hwx : ¬((w || x) = true) := by
            simp only [Bool.or_eq_true, hw, Bool.false_eq_true, false_or]; exact hxf
          have hdec_none : X86Dec.decode b a = none := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rcases hnew' with rfl | rfl <;>
            · repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
              rw [hm_b]; dsimp only
              rw [if_neg hmx]
              have hw0' := Nat.mod_two_ne_one.mp hw0
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hw0'
              simp [X86Dec.aluOfRR, hw0']
          have hnar : decodeNarrow b = some (.rr32 (if o = 0x01 then .add else .sub)
              (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold decodeNarrow
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rcases hnew' with rfl | rfl <;>
            · rw [if_neg (by decide), if_pos (by decide), if_neg hwx]
              rw [hm_b]; dsimp only
              rw [if_neg hmn]
              simp [bb, r, p, rex, hasRex]
          dsimp only [wd] at hwd32
          rw [hwd32]
          simp only [decode, hdec_none, hnar]
          rcases hnew' with rfl | rfl <;> simp [toUniIns, bb, r, p, rex, hasRex]
        · -- 64-bit: `X86Dec` gives `alu`
          have hwd64 : wd = 64 := by simp [wd, hw]
          have hw1 : (if (0x40 ≤ b0 && b0 ≤ 0x4f) then b0 - 0x40 else 0) / 8 % 2 = 1 := hw'.mp hw
          have hdec : X86Dec.decode b a = some (.alu (if o = 0x01 then .add else .sub)
              (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rcases hnew' with rfl | rfl <;>
            · repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
              rw [hm_b]; dsimp only
              rw [if_neg hmx]
              have hw1' := hw1
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hw1'
              simp [X86Dec.aluOfRR, hw1', bb, r, p, rex, hasRex]
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd64
          rw [hwd64]
          rcases hnew' with rfl | rfl <;> simpa [toUniIns, bb, r, p, rex, hasRex] using hres
    have h01 : ¬((o == 0x01) = true) := by intro h; apply hnew; simp [h]
    have h29 : ¬((o == 0x29) = true) := by intro h; apply hnew; simp [h]
    have hrr : (o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89) = true := by
      rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hrr6 :
          o = 0x01 ∨ o = 0x29 ∨ o = 0x31 ∨ o = 0x09 ∨ o = 0x85 ∨ o = 0x89) with h | h | h | h | h | h
      · exact absurd (by simp [h]) h01
      · exact absurd (by simp [h]) h29
      all_goals simp [h]
    rw [if_neg h01, if_neg h29] at hm
    cases hm_b : b (p + 1) with
    | none => rw [hm_b] at hm; cases hm
    | some m =>
      rw [hm_b] at hm
      dsimp only at hm
      have hm_lt := hb (p + 1) m hm_b
      obtain ⟨hbad, hm⟩ := if_none_eq_some hm
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
      obtain ⟨hm64, hxf⟩ := hbad
      simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have hd16 : m % 8 + bb < 16 := by omega
      have hs16 : m / 8 % 8 + r < 16 := by omega
      have hwd : wd = 32 ∨ wd = 64 := by dsimp [wd]; cases w <;> simp
      have hnot_movI : ¬((0xb8 ≤ o && o ≤ 0xbf) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hrr : o = 0x31 ∨ o = 0x09 ∨ o = 0x85 ∨ o = 0x89) with
          rfl | rfl | rfl | rfl <;> decide
      have hmx : ¬((m / 64 != 3 || x) = true) := by
        intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
        rcases h with h | h; exact h hm64; exact hxf h
      have hmn : ¬((m / 64 != 3) = true) := by
        intro h; simp only [bne_iff_ne, ne_eq] at h; exact h hm64
      have hm64_eq : (m / 64 == 3) = true := by simp only [hm64]; decide
      have hwf_ins : InsWf (X86Min.Ins.rr (if o == 0x31 then .xor else if o == 0x09 then .or
          else if o == 0x85 then .test else .mov) wd (m % 8 + bb) (m / 8 % 8 + r)) := by
        refine ⟨hd16, hs16, hwd, ?_⟩
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hrr : o = 0x31 ∨ o = 0x09 ∨ o = 0x85 ∨ o = 0x89) with
          rfl | rfl | rfl | rfl <;> simp
      refine ⟨?_, by omega, hwf_ins⟩
      cases hw : w
      · -- w = false (32-bit)
        have hwd32 : wd = 32 := by simp [wd, hw]
        have hwx : ¬((w || x) = true) := by
          simp only [Bool.or_eq_true, hw, Bool.false_eq_true, false_or]; exact hxf
        have hnw : (!w) = true := by simp only [hw]; decide
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hrr : o = 0x31 ∨ o = 0x09 ∨ o = 0x85 ∨ o = 0x89) with
          rfl | rfl | rfl | rfl
        · -- o = 0x31
          by_cases hds : m % 8 + bb = m / 8 % 8 + r
          · have hdec : X86Dec.decode b a = some (.alu .xor (m % 8 + bb) (m % 8 + bb), p + 2) := by
              unfold X86Dec.decode
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg hnot_movI, if_pos (by decide)]
              rw [hm_b]; dsimp only
              rw [if_neg hmx]
              dsimp only [X86Dec.aluOfRR]
              rw [if_neg (by decide), if_neg (by decide), if_neg (by decide), if_neg (by decide), if_pos (by decide)]
              simp only [pure]
              rw [if_neg (fun h => absurd (Eq.trans hw.symm h) Bool.false_ne_true)]
              rw [if_pos (by rw [beq_self_eq_true, Bool.true_and, beq_iff_eq]; exact hds)]
            have hres := decode_of_dec hdec
            dsimp only [wd] at hwd32
            rw [hwd32]
            rw [if_pos (by decide : (0x31 == 0x31) = true)]
            rw [toUniIns_xor32]
            rw [if_pos (beq_iff_eq.mpr hds)]
            exact hres
          · have hdec_none : X86Dec.decode b a = none := by
              unfold X86Dec.decode
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg hnot_movI, if_pos (by decide)]
              rw [hm_b]; dsimp only
              rw [if_neg hmx]
              dsimp only [X86Dec.aluOfRR]
              rw [if_neg (by decide), if_neg (by decide), if_neg (by decide), if_neg (by decide), if_pos (by decide)]
              simp only [pure]
              rw [if_neg (fun h => absurd (Eq.trans hw.symm h) Bool.false_ne_true)]
              rw [if_neg (fun h => hds (Bool.and_eq_true _ _ |>.mp h |>.2 |> eq_of_beq))]
            have hnar : decodeNarrow b = some (.rr32 .xor (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
              unfold decodeNarrow
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg (by decide), if_pos (by decide), if_neg hwx]
              rw [hm_b]; dsimp only
              rw [if_neg hmn]
              simp only [beq_self_eq_true, if_true, pure]; rfl
            dsimp only [wd] at hwd32
            rw [hwd32]
            simp only [decode, hdec_none, hnar]
            rw [if_pos (by decide : (0x31 == 0x31) = true)]
            rw [toUniIns_xor32]
            rw [if_neg (fun h => hds (eq_of_beq h))]
        · -- o = 0x09
          have hdec_none : X86Dec.decode b a = none := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rw [if_neg hnot_movI, if_pos (by decide)]
            rw [hm_b]; dsimp only
            rw [if_neg hmx]
            dsimp only [X86Dec.aluOfRR]
            rw [if_neg (by decide), if_neg (by decide), if_neg (by decide), if_pos (by decide)]
            simp only [pure]
            rw [if_neg (fun h => absurd (Eq.trans hw.symm h) Bool.false_ne_true)]
            rw [if_neg (fun h => absurd (Bool.and_eq_true _ _ |>.mp h).1 (by decide))]
          have hnar : decodeNarrow b = some (.rr32 .or (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold decodeNarrow
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rw [if_neg (by decide), if_pos (by decide), if_neg hwx]
            rw [hm_b]; dsimp only
            rw [if_neg hmn]
            simp only [beq_self_eq_true, if_true, pure]; rfl
          dsimp only [wd] at hwd32
          rw [hwd32]
          rw [if_neg (by decide : ¬((0x09 == 0x31) = true)), if_pos (by decide : (0x09 == 0x09) = true)]
          simp only [decode, hdec_none, hnar, toUniIns]; rfl
        · -- o = 0x85
          have hdec_none : X86Dec.decode b a = none := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            have hbad_test : ((!w || x) = true) := by
              simp only [Bool.or_eq_true, Bool.not_eq_true']; exact .inl hw
            rw [if_pos hbad_test]
          have hnar : decodeNarrow b = some (.rr32 .test (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold decodeNarrow
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            rw [if_neg (by decide), if_pos (by decide), if_neg hwx]
            rw [hm_b]; dsimp only
            rw [if_neg hmn]
            simp only [beq_self_eq_true, if_true, pure]; rfl
          dsimp only [wd] at hwd32
          rw [hwd32]
          rw [if_neg (by decide : ¬((0x85 == 0x31) = true)), if_neg (by decide : ¬((0x85 == 0x09) = true)),
            if_pos (by decide : (0x85 == 0x85) = true)]
          simp only [decode, hdec_none, hnar, toUniIns]; rfl
        · -- o = 0x89
          have hdec : X86Dec.decode b a = some (.mov32 (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            rw [if_neg hxf, hm_b]; dsimp only
            rw [if_pos hnw, if_pos hm64_eq]
            simp only [beq_self_eq_true, if_true, pure]; rfl
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd32
          rw [hwd32]
          simp only [toUniIns]
          exact hres
      · -- w = true (64-bit)
        have hwd64 : wd = 64 := by simp [wd, hw]
        have hw' : ((if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then b0 - 64 else 0) / 8 % 2 == 1) = true := hw
        have hnw : ¬((!w) = true) := by simp only [hw]; decide
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hrr : o = 0x31 ∨ o = 0x09 ∨ o = 0x85 ∨ o = 0x89) with
          rfl | rfl | rfl | rfl
        · have hdec : X86Dec.decode b a = some (.alu .xor (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            rw [hm_b]; dsimp only
            rw [if_neg hmx]
            dsimp only [X86Dec.aluOfRR]
            rw [if_neg (by decide), if_neg (by decide), if_neg (by decide), if_neg (by decide), if_pos (by decide)]
            dsimp only [Bind.bind, Option.bind, pure]
            rw [if_pos hw']
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd64
          rw [hwd64]
          simp only [toUniIns]
          exact hres
        · have hdec : X86Dec.decode b a = some (.alu .or (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            rw [hm_b]; dsimp only
            rw [if_neg hmx]
            dsimp only [X86Dec.aluOfRR]
            rw [if_neg (by decide), if_neg (by decide), if_neg (by decide), if_pos (by decide)]
            dsimp only [Bind.bind, Option.bind, pure]
            rw [if_pos hw']
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd64
          rw [hwd64]
          simp only [toUniIns]
          exact hres
        · have hdec : X86Dec.decode b a = some (.test (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            have hnot_wx : ¬((!w || x) = true) := by
              intro h; simp only [Bool.or_eq_true, Bool.not_eq_true', hw] at h
              rcases h with h | h; cases h; exact hxf h
            rw [if_neg hnot_wx, hm_b]; dsimp only
            rw [if_neg hmn]
            rfl
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd64
          rw [hwd64]
          simp only [toUniIns]
          exact hres
        · have hdec : X86Dec.decode b a = some (.mov (m % 8 + bb) (m / 8 % 8 + r), p + 2) := by
            unfold X86Dec.decode
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            rw [if_neg hxf, hm_b]; dsimp only
            rw [if_neg hnw, if_pos hm64_eq]
            simp only [beq_self_eq_true, if_true, pure]; rfl
          have hres := decode_of_dec hdec
          dsimp only [wd] at hwd64
          rw [hwd64]
          simp only [toUniIns]
          exact hres
  rw [if_neg hrr6] at hm
  by_cases ho88 : (o == 0x88) = true
  · -- 0x88: stRsp8
    rw [if_pos ho88] at hm
    have ho : o = 0x88 := eq_of_beq ho88; subst ho
    cases hm_b : b (p + 1) with
    | none => rw [hm_b] at hm; cases hm
    | some m =>
      rw [hm_b] at hm
      cases hs_b : b (p + 2) with
      | none => rw [hs_b] at hm; cases hm
      | some s =>
        rw [hs_b] at hm; dsimp only at hm
        have hm_lt := hb (p + 1) m hm_b
        obtain ⟨hbad, hm⟩ := if_none_eq_some hm
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
        obtain ⟨⟨⟨⟨⟨hm64, hm8⟩, hs24⟩, hwf⟩, hxf⟩, hbb0⟩ := hbad
        obtain ⟨hrex, hm⟩ := if_none_eq_some hm
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        have hmd0 : m / 64 = 0 := by omega
        have hdec : X86Dec.decode b a = some (.st8 4 0 (m / 8 % 8 + r), p + 3) := by
          unfold X86Dec.decode
          rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
          repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
          rw [if_neg hxf]
          rw [hm_b]; dsimp only
          have hx_f : x = false := Bool.eq_false_iff.mpr hxf
          have hbb0' : bb = 0 := hbb0
          have hm64_ne3 : (m / 64 == 3) = false := by simp only [hm64]; decide
          have hmem : X86Dec.memOp b (p + 2) m bb x = some (4, 0, p + 3) := by
            simp only [X86Dec.memOp, hx_f, hbb0', hm8, hs_b, hs24, hmd0,
              bne_self_eq_false, beq_self_eq_true, Bool.or_false,
              Bool.false_eq_true, if_false, if_true, Bind.bind, Option.bind, pure]
            change (if ((0 : Nat) == 3) = true then none
                    else if (true && (36 % 8 == 5)) = true then none
                    else some (36 % 8 + 0, 0, (p + 2) + 1)) = some (4, 0, p + 3)
            rw [if_neg (by decide), if_neg (by decide)]
          rw [hmem]; dsimp only
          rw [if_neg hrex]
          rfl
        refine ⟨decode_of_dec hdec, by omega, ?_⟩
        simp only [InsWf]; omega
  rw [if_neg ho88] at hm
  by_cases ho6a : (o == 0x6a) = true
  · -- 0x6a: push
    rw [if_pos ho6a] at hm
    have ho : o = 0x6a := eq_of_beq ho6a; subst ho
    obtain ⟨hno, hm⟩ := if_none_eq_some hm
    have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
    rw [hp0] at hbp
    have hbo : b0 = 0x6a := Option.some.inj (hb0.symm.trans hbp)
    subst hbo
    cases hb1 : b 1 with
    | none => rw [hb1] at hm; cases hm
    | some imm =>
      rw [hb1] at hm
      simp only [pure] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have himm_lt := hb 1 imm hb1
      have hdec : X86Dec.decode b a = some (.pushI (X86Dec.sx8 imm), 2) := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]
        simp [hb0, hb1]
      refine ⟨by simpa [sx8_eq, toUniIns] using decode_of_dec hdec, by omega, ?_⟩
      exact sx8_lt_W himm_lt
  rw [if_neg ho6a] at hm
  by_cases hpop : (0x58 ≤ o && o ≤ 0x5f) = true
  · -- 0x58..0x5f: pop
    rw [if_pos hpop] at hm
    obtain ⟨hbad, hm⟩ := if_none_eq_some hm
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
    obtain ⟨⟨hwf, hr0⟩, hxf⟩ := hbad
    simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
    obtain ⟨rfl, rfl⟩ := hm
    have hpopb : 0x58 ≤ o ∧ o ≤ 0x5f := by simpa [Bool.and_eq_true, decide_eq_true_eq] using hpop
    have hnot_movI : ¬((0xb8 ≤ o && o ≤ 0xbf) = true) := by simp; omega
    have hnot_rr : ¬((o == 0x01 || o == 0x29 || o == 0x21 || o == 0x09 || o == 0x31) = true) := by
      simp; omega
    have hnot_push : ¬((0x50 ≤ o && o ≤ 0x57) = true) := by simp; omega
    have hdec : X86Dec.decode b a = some (.pop (o - 0x58 + bb), p + 1) := by
      unfold X86Dec.decode
      rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
      rw [if_neg hnot_movI, if_neg hnot_rr]
      rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
      rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
      rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
      rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
      rw [if_neg (by simp; omega), if_neg (by simp; omega)]
      rw [if_neg hnot_push, if_pos hpop]
      have hbad_pop : ¬((w || r != 0 || x) = true) := by
        intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
        rcases h with (h | h) | h; exact hwf h; exact h hr0; exact hxf h
      rw [if_neg hbad_pop]
      rfl
    refine ⟨decode_of_dec hdec, by omega, ?_⟩
    simp only [InsWf]; omega
  rw [if_neg hpop] at hm
  by_cases ho0f : (o == 0x0f) = true
  · -- 0x0f: syscall or ldRsp8
    rw [if_pos ho0f] at hm
    have ho : o = 0x0f := eq_of_beq ho0f; subst ho
    cases ho2_b : b (p + 1) with
    | none => rw [ho2_b] at hm; cases hm
    | some o2 =>
      rw [ho2_b] at hm; dsimp only at hm
      by_cases ho2_05 : (o2 == 0x05) = true
      · -- syscall
        rw [if_pos ho2_05] at hm
        have ho2 : o2 = 0x05 := eq_of_beq ho2_05; subst ho2
        obtain ⟨hno, hm⟩ := if_none_eq_some hm
        have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
        have hp0' : p = 0 := hp0
        rw [hp0] at hbp
        rw [hp0'] at ho2_b
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        have hdec : X86Dec.decode b a = some (.syscall, 2) := by
          unfold X86Dec.decode
          have hb0_eq : b0 = 0x0f := Option.some.inj (hb0.symm.trans hbp)
          subst hb0_eq
          rw [hb0]; simp only [Bind.bind, Option.bind]
          simp [hb0, ho2_b]
        exact ⟨decode_of_dec hdec, by omega, trivial⟩
      · -- ldRsp8
        rw [if_neg ho2_05] at hm
        by_cases ho2_b6 : (o2 == 0xb6) = true
        · rw [if_pos ho2_b6] at hm
          have ho2 : o2 = 0xb6 := eq_of_beq ho2_b6; subst ho2
          cases hm_b : b (p + 2) with
          | none => rw [hm_b] at hm; cases hm
          | some m =>
            rw [hm_b] at hm
            cases hs_b : b (p + 3) with
            | none => rw [hs_b] at hm; cases hm
            | some s =>
              rw [hs_b] at hm; dsimp only at hm
              have hm_lt := hb (p + 2) m hm_b
              obtain ⟨hbad, hm⟩ := if_none_eq_some hm
              simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
              obtain ⟨⟨⟨⟨⟨hm64, hm8⟩, hs24⟩, hwf⟩, hxf⟩, hbb0⟩ := hbad
              simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
              obtain ⟨rfl, rfl⟩ := hm
              have hmd0 : m / 64 = 0 := by omega
              have hdec : X86Dec.decode b a = some (.ld8 (m / 8 % 8 + r) 4 0, p + 4) := by
                unfold X86Dec.decode
                rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
                repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
                rw [ho2_b]; dsimp only
                rw [if_neg (by decide), if_pos (by decide), if_neg hxf]
                rw [hm_b]; dsimp only
                have hx_f : x = false := Bool.eq_false_iff.mpr hxf
                have hbb0' : bb = 0 := hbb0
                have hm64_ne3 : (m / 64 == 3) = false := by simp only [hm64]; decide
                have hmem : X86Dec.memOp b (p + 3) m bb x = some (4, 0, p + 4) := by
                  simp only [X86Dec.memOp, hx_f, hbb0', hm8, hs_b, hs24, hmd0,
                    bne_self_eq_false, beq_self_eq_true, Bool.or_false,
                    Bool.false_eq_true, if_false, if_true, Bind.bind, Option.bind, pure]
                  change (if ((0 : Nat) == 3) = true then none
                          else if (true && (36 % 8 == 5)) = true then none
                          else some (36 % 8 + 0, 0, (p + 3) + 1)) = some (4, 0, p + 4)
                  rw [if_neg (by decide), if_neg (by decide)]
                rw [hmem]; rfl
              refine ⟨decode_of_dec hdec, by omega, ?_⟩
              simp only [InsWf]; omega
        · rw [if_neg ho2_b6] at hm; cases hm
  rw [if_neg ho0f] at hm
  by_cases hjcc : (0x70 ≤ o && o ≤ 0x7f) = true
  · -- 0x70..0x7f: jcc rel8
    rw [if_pos hjcc] at hm
    obtain ⟨hno, hm⟩ := if_none_eq_some hm
    have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
    rw [hp0] at hbp
    have hbo : b0 = o := Option.some.inj (hb0.symm.trans hbp)
    subst hbo
    cases hb1 : b 1 with
    | none => rw [hb1] at hm; cases hm
    | some d =>
      rw [hb1] at hm
      simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have hd_lt := hb 1 d hb1
      have hjccb : 0x70 ≤ b0 ∧ b0 ≤ 0x7f := by simpa [Bool.and_eq_true, decide_eq_true_eq] using hjcc
      have hnot_movI : ¬((0xb8 ≤ b0 && b0 ≤ 0xbf) = true) := by simp; omega
      have hnot_rr : ¬((b0 == 0x01 || b0 == 0x29 || b0 == 0x21 || b0 == 0x09 || b0 == 0x31) = true) := by
        simp; omega
      have hdec : X86Dec.decode b a = some (.jcc (b0 - 0x70) ((a + 2 + X86Dec.sx8 d) % W), 2) := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]
        rw [show (if hasRex then b0 - 0x40 else 0) = rex from rfl]
        rw [show (if hasRex then 1 else 0) = 0 from hp0, hb0]; dsimp only
        rw [if_neg hnot_movI, if_neg hnot_rr]
        rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
        rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
        rw [if_neg (by simp; omega), if_neg (by simp; omega), if_neg (by simp; omega)]
        rw [if_pos hjcc, if_neg hno, hb1]; rfl
      refine ⟨by simpa [sx8_eq, toUniIns] using decode_of_dec hdec, by omega, ?_⟩
      exact sx8_lt_W hd_lt
  rw [if_neg hjcc] at hm
  by_cases hoeb : (o == 0xeb) = true
  · -- 0xeb: jmp rel8
    rw [if_pos hoeb] at hm
    have ho : o = 0xeb := eq_of_beq hoeb; subst ho
    obtain ⟨hno, hm⟩ := if_none_eq_some hm
    have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
    rw [hp0] at hbp
    have hbo : b0 = 0xeb := Option.some.inj (hb0.symm.trans hbp)
    subst hbo
    cases hb1 : b 1 with
    | none => rw [hb1] at hm; cases hm
    | some d =>
      rw [hb1] at hm
      simp only [pure] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have hd_lt := hb 1 d hb1
      have hdec : X86Dec.decode b a = some (.jmp ((a + 2 + X86Dec.sx8 d) % W), 2) := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]
        simp [hb0, hb1]
      refine ⟨by simpa [sx8_eq, toUniIns] using decode_of_dec hdec, by omega, ?_⟩
      exact sx8_lt_W hd_lt
  rw [if_neg hoeb] at hm
  by_cases hoe9 : (o == 0xe9) = true
  · -- 0xe9: jmp rel32
    rw [if_pos hoe9] at hm
    have ho : o = 0xe9 := eq_of_beq hoe9; subst ho
    obtain ⟨hno, hm⟩ := if_none_eq_some hm
    have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
    rw [hp0] at hbp
    have hbo : b0 = 0xe9 := Option.some.inj (hb0.symm.trans hbp)
    subst hbo
    cases hle : X86Min.le b 1 4 with
    | none => rw [hle] at hm; cases hm
    | some d =>
      rw [hle] at hm
      simp only [pure] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have hd_lt : d < 2 ^ 32 := by simpa using le_lt_pow256 hb hle
      have hdec : X86Dec.decode b a = some (.jmp ((a + 5 + X86Dec.sx32 d) % W), 5) := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]
        simp [hb0, leB_eq_le, hle]
      refine ⟨by simpa [sx32_eq, toUniIns] using decode_of_dec hdec, by omega, ?_⟩
      exact sx32_lt_W hd_lt
  rw [if_neg hoe9] at hm
  by_cases hal : (o == 0x04 || o == 0x3c || o == 0x2c) = true
  · -- 0x04 || 0x3c || 0x2c: alImm8
    rw [if_pos hal] at hm
    obtain ⟨hno, hm⟩ := if_none_eq_some hm
    have hp0 : (if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then 1 else 0) = 0 := if_neg hno
    rw [hp0] at hbp
    have hbo : b0 = o := Option.some.inj (hb0.symm.trans hbp)
    subst hbo
    cases hb1 : b 1 with
    | none => rw [hb1] at hm; cases hm
    | some i =>
      rw [hb1] at hm
      simp only [pure] at hm
      obtain ⟨rfl, rfl⟩ := hm
      have hi_lt := hb 1 i hb1
      have hdec_none : X86Dec.decode b a = none := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hal : b0 = 0x04 ∨ b0 = 0x3c ∨ b0 = 0x2c) with
          rfl | rfl | rfl <;> unfold X86Dec.decode <;> simp [Bind.bind, Option.bind, hb0]
      have hnar : decodeNarrow b = some (.alImm8 (if b0 == 0x04 then .add else if b0 == 0x3c then .cmp else .sub) i, 2) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hal : b0 = 0x04 ∨ b0 = 0x3c ∨ b0 = 0x2c) with
          rfl | rfl | rfl <;> unfold decodeNarrow <;> simp [Bind.bind, Option.bind, hb0, hb1]
      refine ⟨?_, by omega, ?_⟩
      · rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hal : b0 = 0x04 ∨ b0 = 0x3c ∨ b0 = 0x2c) with
          rfl | rfl | rfl <;> simp [decode, hdec_none, hnar, toUniIns]
      · rcases (by simpa [Bool.or_eq_true, beq_iff_eq, or_assoc] using hal : b0 = 0x04 ∨ b0 = 0x3c ∨ b0 = 0x2c) with
          rfl | rfl | rfl <;> simp [InsWf, hi_lt]
  rw [if_neg hal] at hm
  by_cases hoc1 : (o == 0xc1) = true
  · -- 0xc1 /4: shl32
    rw [if_pos hoc1] at hm
    have ho : o = 0xc1 := eq_of_beq hoc1; subst ho
    cases hm_b : b (p + 1) with
    | none => rw [hm_b] at hm; cases hm
    | some m =>
      rw [hm_b] at hm
      cases hc_b : b (p + 2) with
      | none => rw [hc_b] at hm; cases hm
      | some c =>
        rw [hc_b] at hm; dsimp only at hm
        have hm_lt := hb (p + 1) m hm_b
        have hc_lt := hb (p + 2) c hc_b
        obtain ⟨hbad, hm⟩ := if_none_eq_some hm
        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
        obtain ⟨⟨⟨⟨hm64, hm8⟩, hw⟩, hr0⟩, hxf⟩ := hbad
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        have hdec_none : X86Dec.decode b a = none := by
          unfold X86Dec.decode
          rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
          repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
          have hbad_c1 : ((!w || r != 0 || x) = true) := by
            simp only [Bool.or_eq_true, Bool.not_eq_true']; exact .inl (.inl (Bool.eq_false_iff.mpr hw))
          rw [if_pos hbad_c1]
        have hnar : decodeNarrow b = some (.shl32 (m % 8 + bb) c, p + 3) := by
          unfold decodeNarrow
          rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
          repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
          have hwrx : ¬((w || r != 0 || x) = true) := by
            intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
            rcases h with (h | h) | h; exact hw h; exact h hr0; exact hxf h
          have hm648 : ¬((m / 64 != 3 || m / 8 % 8 != 4) = true) := by
            intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
            rcases h with h | h; exact h hm64; exact h hm8
          rw [if_neg hwrx, hm_b, hc_b]; dsimp only
          rw [if_neg hm648]
          rfl
        refine ⟨?_, by omega, ?_⟩
        · simp only [decode, hdec_none, hnar, toUniIns]; rfl
        · exact ⟨rfl, by omega, hc_lt⟩
  rw [if_neg hoc1] at hm
  by_cases hcmp : (o == 0x81 || o == 0x83) = true
  · -- 0x81 || 0x83: cmp r32/r64, imm
    rw [if_pos hcmp] at hm
    cases hm_b : b (p + 1) with
    | none => rw [hm_b] at hm; cases hm
    | some m =>
      rw [hm_b] at hm; dsimp only at hm
      have hm_lt := hb (p + 1) m hm_b
      obtain ⟨hbad, hm'⟩ := if_none_eq_some hm
      clear hm
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
      obtain ⟨⟨⟨hm64, hf⟩, hr0⟩, hxf⟩ := hbad
      have hf057 : m / 8 % 8 = 0 ∨ m / 8 % 8 = 5 ∨ m / 8 % 8 = 7 := by
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hf; omega
      have hm648 : ¬((m / 64 != 3 || (m / 8 % 8 != 0 && m / 8 % 8 != 5 && m / 8 % 8 != 7)) = true) := by
        simp only [Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq]; omega
      have hnot_movI : ¬((0xb8 ≤ o && o ≤ 0xbf) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hnot_rr : ¬((o == 0x01 || o == 0x29 || o == 0x21 || o == 0x09 || o == 0x31) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hnot_89 : ¬((o == 0x89 || o == 0x8b) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hnot_85 : ¬((o == 0x85) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hnot_6a : ¬((o == 0x6a) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hnot_39 : ¬((o == 0x39 || o == 0x3b) = true) := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hcmp_dec : (o == 0x83 || o == 0x81) = true := by
        rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl <;> decide
      have hm64_ne : ¬((m / 64 != 3) = true) := by
        intro h; simp only [bne_iff_ne, ne_eq] at h; exact h hm64
      by_cases hm8n : m / 8 % 8 ≠ 7
      · -- 0x81 || 0x83 with /0 add or /5 sub
        have hf05 : m / 8 % 8 = 0 ∨ m / 8 % 8 = 5 := by omega
        rcases hf05 with hfv | hfv <;>
        · cases hw : w
          · -- w = false (32-bit): `X86Dec` refuses, `decodeNarrow` gives `aluI32`
            have hw' : ((if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then b0 - 64 else 0) / 8 % 2 == 1) = false := hw
            have hwrx : ¬((w || r != 0 || x) = true) := by
              intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, hw, Bool.false_eq_true, false_or] at h
              rcases h with h | h; exact h hr0; exact hxf h
            have hdec_none : X86Dec.decode b a = none := by
              unfold X86Dec.decode
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
                if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
              have hbad_cmp : ((!w || r != 0 || x) = true) := by
                simp only [Bool.or_eq_true, Bool.not_eq_true']; exact .inl (.inl hw)
              rw [if_pos hbad_cmp]
            by_cases ho81 : (o == 0x81) = true
            · rw [if_pos ho81] at hm'
              have ho81' : o = 0x81 := eq_of_beq ho81; subst ho81'
              cases hle : X86Min.le b (p + 2) 4 with
              | none => rw [hle] at hm'; cases hm'
              | some i =>
                rw [hle] at hm'
                simp only [pure] at hm'
                obtain ⟨rfl, rfl⟩ := hm'
                have hnar : decodeNarrow b = some (narrowRI (if (m / 8 % 8 == 0) = true then .add
                    else if (m / 8 % 8 == 5) = true then .sub else .cmp) (m % 8 + bb)
                    (X86Dec.sx32 i % 2 ^ 32), p + 6) := by
                  unfold decodeNarrow
                  rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
                  repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
                  rw [if_neg hwrx, hm_b]; dsimp only
                  rw [if_neg hm648, if_pos (by decide), leB_eq_le, hle]
                  rfl
                refine ⟨?_, by omega, ?_⟩
                · simp only [decode, hdec_none, hnar]
                  dsimp only [toUniIns, wd]; rw [hw']
                  simp [hfv, narrowRI, sx32_eq, pow2_32_eq, bb, p, r, rex, hasRex]
                · dsimp only [InsWf, wd]; rw [hw']
                  simp only [Bool.false_eq_true, if_false, pow2_32_eq]
                  exact .inr (.inr ⟨by simp [hfv], .inl trivial, by omega, Nat.mod_lt _ (by decide)⟩)
            · rw [if_neg ho81] at hm'
              cases hbi : b (p + 2) with
              | none => rw [hbi] at hm'; cases hm'
              | some i =>
                rw [hbi] at hm'
                simp only [pure] at hm'
                obtain ⟨rfl, rfl⟩ := hm'
                have ho83 : o = 0x83 := by
                  rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl
                  · simp at ho81
                  · rfl
                subst ho83
                have hnar : decodeNarrow b = some (narrowRI (if (m / 8 % 8 == 0) = true then .add
                    else if (m / 8 % 8 == 5) = true then .sub else .cmp) (m % 8 + bb)
                    (X86Dec.sx8 i % 2 ^ 32), p + 3) := by
                  unfold decodeNarrow
                  rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
                  repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
                  rw [if_neg hwrx, hm_b]; dsimp only
                  rw [if_neg hm648, if_neg (by decide), hbi]
                  rfl
                refine ⟨?_, by omega, ?_⟩
                · simp only [decode, hdec_none, hnar]
                  dsimp only [toUniIns, wd]; rw [hw']
                  simp [hfv, narrowRI, sx8_eq, pow2_32_eq, bb, p, r, rex, hasRex]
                · dsimp only [InsWf, wd]; rw [hw']
                  simp only [Bool.false_eq_true, if_false, pow2_32_eq]
                  exact .inr (.inr ⟨by simp [hfv], .inl trivial, by omega, Nat.mod_lt _ (by decide)⟩)
          · -- w = true (64-bit): `X86Dec` gives `aluI`
            have hw' : ((if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then b0 - 64 else 0) / 8 % 2 == 1) = true := hw
            have hnwrx : ¬((!w || r != 0 || x) = true) := by
              intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', hw] at h
              rcases h with (h | h) | h; cases h; exact h hr0; exact hxf h
            by_cases ho81 : (o == 0x81) = true
            · rw [if_pos ho81] at hm'
              have ho81' : o = 0x81 := eq_of_beq ho81; subst ho81'
              cases hle : X86Min.le b (p + 2) 4 with
              | none => rw [hle] at hm'; cases hm'
              | some i =>
                rw [hle] at hm'
                simp only [pure] at hm'
                obtain ⟨rfl, rfl⟩ := hm'
                have hi_lt : i < 2 ^ 32 := by simpa using le_lt_pow256 hb hle
                have hsx_lt : X86Min.sx32 i < W := sx32_lt_W hi_lt
                have hsx_mod : X86Min.sx32 i % X86Min.pow2 64 = X86Min.sx32 i := by
                  rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx_lt
                have hdec : X86Dec.decode b a = some (.aluI (if m / 8 % 8 = 0 then .add else .sub)
                    (m % 8 + bb) (X86Dec.sx32 i), p + 6) := by
                  unfold X86Dec.decode
                  rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
                  rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
                    if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
                  rw [if_neg hnwrx, hm_b]; dsimp only
                  rw [if_neg hm64_ne]
                  rw [if_neg (by decide), leB_eq_le, hle]
                  simp [X86Dec.aluOfExt, hfv, bb, p, r, rex, hasRex]
                refine ⟨?_, by omega, ?_⟩
                · have hres := decode_of_dec hdec
                  dsimp only [toUniIns, wd]; rw [hw']
                  simpa [hfv, hsx_mod, sx32_eq, bb, p, r, rex, hasRex] using hres
                · dsimp only [InsWf, wd]; rw [hw']; simp only [if_true]
                  exact .inr (.inr ⟨by simp [hfv], .inr trivial, by omega,
                    by rw [hsx_mod, pow2_64_eq_W]; exact hsx_lt⟩)
            · rw [if_neg ho81] at hm'
              cases hbi : b (p + 2) with
              | none => rw [hbi] at hm'; cases hm'
              | some i =>
                rw [hbi] at hm'
                simp only [pure] at hm'
                obtain ⟨rfl, rfl⟩ := hm'
                have ho83 : o = 0x83 := by
                  rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl
                  · simp at ho81
                  · rfl
                subst ho83
                have hi_lt := hb (p + 2) i hbi
                have hsx_lt : X86Min.sx8 i < W := sx8_lt_W hi_lt
                have hsx_mod : X86Min.sx8 i % X86Min.pow2 64 = X86Min.sx8 i := by
                  rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx_lt
                have hdec : X86Dec.decode b a = some (.aluI (if m / 8 % 8 = 0 then .add else .sub)
                    (m % 8 + bb) (X86Dec.sx8 i), p + 3) := by
                  unfold X86Dec.decode
                  rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
                  rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
                    if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
                  rw [if_neg hnwrx, hm_b]; dsimp only
                  rw [if_neg hm64_ne]
                  rw [if_pos (by decide), hbi]
                  simp [X86Dec.aluOfExt, hfv, bb, p, r, rex, hasRex]
                refine ⟨?_, by omega, ?_⟩
                · have hres := decode_of_dec hdec
                  dsimp only [toUniIns, wd]; rw [hw']
                  simpa [hfv, hsx_mod, sx8_eq, bb, p, r, rex, hasRex] using hres
                · dsimp only [InsWf, wd]; rw [hw']; simp only [if_true]
                  exact .inr (.inr ⟨by simp [hfv], .inr trivial, by omega,
                    by rw [hsx_mod, pow2_64_eq_W]; exact hsx_lt⟩)
      -- /7 cmp: the operation chain of `X86Min.decode` selects `cmp`
      have hm8 : m / 8 % 8 = 7 := by omega
      have hop : (if (m / 8 % 8 == 0) = true then X86Min.Op.add
          else if (m / 8 % 8 == 5) = true then X86Min.Op.sub else X86Min.Op.cmp) = .cmp := by
        simp [hm8]
      rw [hop] at hm'
      cases hw : w
      · -- w = false (32-bit)
        have hw' : ((if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then b0 - 64 else 0) / 8 % 2 == 1) = false := hw
        have hwrx : ¬((w || r != 0 || x) = true) := by
          intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, hw, Bool.false_eq_true, false_or] at h
          rcases h with h | h; exact h hr0; exact hxf h
        have hdec_none : X86Dec.decode b a = none := by
          unfold X86Dec.decode
          rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
          rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
            if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
          have hbad_cmp : ((!w || r != 0 || x) = true) := by
            simp only [Bool.or_eq_true, Bool.not_eq_true']; exact .inl (.inl hw)
          rw [if_pos hbad_cmp]
        by_cases ho81 : (o == 0x81) = true
        · -- 0x81
          rw [if_pos ho81] at hm'
          have ho81' : o = 0x81 := eq_of_beq ho81; subst ho81'
          cases hle : X86Min.le b (p + 2) 4 with
          | none => rw [hle] at hm'; cases hm'
          | some i =>
            rw [hle] at hm'
            simp only [pure] at hm'
            obtain ⟨rfl, rfl⟩ := hm'
            have hnar : decodeNarrow b = some (.cmpI32 (m % 8 + bb) (X86Dec.sx32 i % 2 ^ 32), p + 6) := by
              unfold decodeNarrow
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
              rw [if_neg hwrx, hm_b]; dsimp only
              rw [if_neg hm648, if_pos (by decide), leB_eq_le, hle]
              simp [narrowRI, hm8, bb, p, r, rex, hasRex]
            refine ⟨?_, by omega, ?_⟩
            · simp only [decode, hdec_none, hnar]
              dsimp only [toUniIns, wd]; rw [hw']
              simp only [Bool.false_eq_true, if_false, sx32_eq, pow2_32_eq]; rfl
            · dsimp only [InsWf, wd]; rw [hw']
              simp only [Bool.false_eq_true, if_false, pow2_32_eq]
              exact .inl ⟨trivial, .inl trivial, by omega, Nat.mod_lt _ (by decide)⟩
        · -- 0x83
          rw [if_neg ho81] at hm'
          cases hbi : b (p + 2) with
          | none => rw [hbi] at hm'; cases hm'
          | some i =>
            rw [hbi] at hm'
            simp only [pure] at hm'
            obtain ⟨rfl, rfl⟩ := hm'
            have ho83 : o = 0x83 := by
              rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl
              · simp at ho81
              · rfl
            subst ho83
            have hnar : decodeNarrow b = some (.cmpI32 (m % 8 + bb) (X86Dec.sx8 i % 2 ^ 32), p + 3) := by
              unfold decodeNarrow
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
              rw [if_neg hwrx, hm_b]; dsimp only
              rw [if_neg hm648, if_neg (by decide), hbi]
              simp [narrowRI, hm8, bb, p, r, rex, hasRex]
            refine ⟨?_, by omega, ?_⟩
            · simp only [decode, hdec_none, hnar]
              dsimp only [toUniIns, wd]; rw [hw']
              simp only [Bool.false_eq_true, if_false, sx8_eq, pow2_32_eq]; rfl
            · dsimp only [InsWf, wd]; rw [hw']
              simp only [Bool.false_eq_true, if_false, pow2_32_eq]
              exact .inl ⟨trivial, .inl trivial, by omega, Nat.mod_lt _ (by decide)⟩
      · -- w = true (64-bit)
        have hw' : ((if (decide (64 ≤ b0) && decide (b0 ≤ 79)) = true then b0 - 64 else 0) / 8 % 2 == 1) = true := hw
        have hnwrx : ¬((!w || r != 0 || x) = true) := by
          intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, Bool.not_eq_true', hw] at h
          rcases h with (h | h) | h; cases h; exact h hr0; exact hxf h
        by_cases ho81 : (o == 0x81) = true
        · -- 0x81
          rw [if_pos ho81] at hm'
          have ho81' : o = 0x81 := eq_of_beq ho81; subst ho81'
          cases hle : X86Min.le b (p + 2) 4 with
          | none => rw [hle] at hm'; cases hm'
          | some i =>
            rw [hle] at hm'
            simp only [pure] at hm'
            obtain ⟨rfl, rfl⟩ := hm'
            have hi_lt : i < 2 ^ 32 := by simpa using le_lt_pow256 hb hle
            have hsx_lt : X86Min.sx32 i < W := sx32_lt_W hi_lt
            have hsx_mod : X86Min.sx32 i % X86Min.pow2 64 = X86Min.sx32 i := by
              rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx_lt
            have hdec : X86Dec.decode b a = some (.cmpI (m % 8 + bb) (X86Dec.sx32 i), p + 6) := by
              unfold X86Dec.decode
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
                if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
              rw [if_neg hnwrx, hm_b]; dsimp only
              rw [if_neg hm64_ne]
              rw [if_neg (by decide), leB_eq_le, hle]
              simp only [pure]
              rw [if_pos (by simp only [hm8]; decide)]
            refine ⟨?_, by omega, ?_⟩
            · have hres := decode_of_dec hdec
              dsimp only [toUniIns, wd]; rw [hw']
              simp only [if_true, hsx_mod]
              exact hres
            · dsimp only [InsWf, wd]; rw [hw']; simp only [if_true]
              exact .inl ⟨trivial, .inr trivial, by omega, by rw [hsx_mod, pow2_64_eq_W]; exact hsx_lt⟩
        · -- 0x83
          rw [if_neg ho81] at hm'
          cases hbi : b (p + 2) with
          | none => rw [hbi] at hm'; cases hm'
          | some i =>
            rw [hbi] at hm'
            simp only [pure] at hm'
            obtain ⟨rfl, rfl⟩ := hm'
            have ho83 : o = 0x83 := by
              rcases (by simpa [Bool.or_eq_true, beq_iff_eq] using hcmp : o = 0x81 ∨ o = 0x83) with rfl | rfl
              · simp at ho81
              · rfl
            subst ho83
            have hi_lt := hb (p + 2) i hbi
            have hsx_lt : X86Min.sx8 i < W := sx8_lt_W hi_lt
            have hsx_mod : X86Min.sx8 i % X86Min.pow2 64 = X86Min.sx8 i := by
              rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hsx_lt
            have hdec : X86Dec.decode b a = some (.cmpI (m % 8 + bb) (X86Dec.sx8 i), p + 3) := by
              unfold X86Dec.decode
              rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
              rw [if_neg hnot_movI, if_neg hnot_rr, if_neg hnot_89, if_neg hnot_85,
                if_neg hnot_6a, if_neg hnot_39, if_pos hcmp_dec]
              rw [if_neg hnwrx, hm_b]; dsimp only
              rw [if_neg hm64_ne]
              rw [if_pos (by decide), hbi]
              simp only [pure]
              rw [if_pos (by simp only [hm8]; decide)]
            refine ⟨?_, by omega, ?_⟩
            · have hres := decode_of_dec hdec
              dsimp only [toUniIns, wd]; rw [hw']
              simp only [if_true, hsx_mod]
              exact hres
            · dsimp only [InsWf, wd]; rw [hw']; simp only [if_true]
              exact .inl ⟨trivial, .inr trivial, by omega, by rw [hsx_mod, pow2_64_eq_W]; exact hsx_lt⟩
  rw [if_neg hcmp] at hm
  by_cases hoff : (o == 0xff) = true
  · -- 0xff /0: inc r32/r64
    rw [if_pos hoff] at hm
    have ho : o = 0xff := eq_of_beq hoff; subst ho
    cases hm_b : b (p + 1) with
    | none => rw [hm_b] at hm; cases hm
    | some m =>
      rw [hm_b] at hm; dsimp only at hm
      have hm_lt := hb (p + 1) m hm_b
      obtain ⟨hbad, hm⟩ := if_none_eq_some hm
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hbad
      obtain ⟨⟨hm64, hr0⟩, hxf⟩ := hbad
      have hrx : ¬((r != 0 || x) = true) := by
        intro h; simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at h
        rcases h with h | h; exact h hr0; exact hxf h
      have hm64n : ¬((m / 64 != 3) = true) := by
        intro h; simp only [bne_iff_ne, ne_eq] at h; exact h hm64
      -- `jmp qword [..]` in `X86Dec` needs a memory operand, so with mod 11 it refuses
      have hdec_none : X86Dec.decode b a = none := by
        unfold X86Dec.decode
        rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
        repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
        rw [hm_b]; dsimp only
        have hbad_jtab : (m / 8 % 8 != 4 || m / 64 == 3 || m % 8 != 4) = true := by
          simp only [Bool.or_eq_true, hm64]; exact .inl (.inr rfl)
        rw [if_pos hbad_jtab]
        simp only [ite_self]
      by_cases hm0 : (m / 8 % 8 == 0) = true
      · -- ff /0: inc
        rw [if_pos hm0] at hm
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
        obtain ⟨rfl, rfl⟩ := hm
        have hnar : decodeNarrow b = some (.inc wd (m % 8 + bb), p + 2) := by
          unfold decodeNarrow
          rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
          repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
          rw [if_neg hrx, hm_b]; dsimp only
          rw [if_neg hm64n, if_pos hm0]
          rfl
        refine ⟨?_, by omega, ?_⟩
        · simp only [decode, hdec_none, hnar, toUniIns]; rfl
        · have hwd : wd = 32 ∨ wd = 64 := by dsimp [wd]; cases w <;> simp
          exact ⟨hwd, by omega⟩
      · rw [if_neg hm0] at hm
        by_cases hm4 : (m / 8 % 8 == 4) = true
        · -- ff /4: jmp r64
          rw [if_pos hm4] at hm
          simp only [pure, Option.some.injEq, Prod.mk.injEq] at hm
          obtain ⟨rfl, rfl⟩ := hm
          have hnar : decodeNarrow b = some (.ofInt (.jmpR (m % 8 + bb)), p + 2) := by
            unfold decodeNarrow
            rw [hb0]; simp only [Bind.bind, Option.bind]; rw [hbp]; dsimp only
            repeat (first | rw [if_pos (by decide)] | rw [if_neg (by decide)])
            rw [if_neg hrx, hm_b]; dsimp only
            rw [if_neg hm64n, if_neg hm0, if_pos hm4]
            rfl
          refine ⟨?_, by omega, ?_⟩
          · simp only [decode, hdec_none, hnar, toUniIns]; rfl
          · show m % 8 + bb < 16
            omega
        · rw [if_neg hm4] at hm
          cases hm
  rw [if_neg hoff] at hm
  cases hm

/-- Unrestricted `X86Min -> X86Uni` decoder agreement: for EVERY byte stream `b`
with entries `< 256` and every address `a`, any instruction decoded by `X86Min.decode`
decodes under `X86Uni.decode` with the exact same length and translated instruction. -/
theorem decode_min_agree {b : Nat → Option Nat} {a : Nat} {ins : X86Min.Ins} {len : Nat}
    (hb : ∀ i v, b i = some v → v < 256)
    (hm : X86Min.decode b = some (ins, len)) :
    decode b a = some (toUniIns a len ins, len) ∧ len ≤ 7 ∧ InsWf ins := by
  cases hb0 : b 0 with
  | none => unfold X86Min.decode at hm; rw [hb0] at hm; cases hm
  | some b0 =>
    cases hbp : b (if 0x40 ≤ b0 && b0 ≤ 0x4f then 1 else 0) with
    | none =>
      unfold X86Min.decode at hm
      rw [hb0] at hm; dsimp only [Bind.bind, Option.bind] at hm; rw [hbp] at hm; cases hm
    | some o => exact decode_min_agree_aux hb hb0 hbp hm

/-! ### Simulation relation `Sim` and memory/register lemmas -/

/-- Flag refinement from `X86Min.Flags` to `X86Int.Flags`. -/
structure FlRel (mf : X86Min.Flags) (if_ : Flags) : Prop where
  cf : X86Min.OLe mf.cf if_.cf
  zf : X86Min.OLe mf.zf if_.zf
  sf : X86Min.OLe mf.sf if_.sf
  of : X86Min.OLe mf.of if_.of

/-- Address `a` lies inside a writable region of `X86Env.Mem`. -/
def MemWritable (mem : Mem) (a : Nat) : Prop :=
  ∃ idx r, mem.find a 0 = some (idx, r) ∧ r.rw = true

/-- Simulation invariant relating `X86Min.St` and `X86Int.St`. -/
structure Sim (m : X86Min.St) (i : St) : Prop where
  rip : m.rip = i.rip
  ripBound : m.rip < W
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

private theorem reg_lt16_bound {m : X86Min.St} {i : St} (h : Sim m i) {r v : Nat}
    (hr : r < 16) (hm : m.reg r = some v) : i.reg r = some v ∧ v < W :=
  let hi := h.regsLe r hr v hm
  ⟨hi, h.regsBound r v hi⟩

private theorem m_reg_setReg {s : X86Min.St} (hlen : s.regs.length = 16) {d r : Nat}
    (hd : d < 16) (v : Option Nat) :
    (s.setReg d v).reg r = if r = d then v else s.reg r := by
  simp only [X86Min.St.reg, X86Min.St.setReg, X86Min.getD_set_reg]
  by_cases h : r = d
  · subst h; simp [hlen, hd]
  · have hne : ¬ (d = r ∧ d < s.regs.length) := fun hc => h hc.1.symm
    simp [h, hne]

private theorem i_reg_set {s : St} (hlen : s.regs.length = 16) {d r : Nat}
    (hd : d < 16) (v : Option Nat) :
    (s.set d v).reg r = if r = d then v else s.reg r := by
  simp only [X86Int.St.reg, X86Int.St.set, X86Min.getD_set_reg]
  by_cases h : r = d
  · subst h; simp [hlen, hd]
  · have hne : ¬ (d = r ∧ d < s.regs.length) := fun hc => h hc.1.symm
    simp [h, hne]

private theorem cond_rel {mf : X86Min.Flags} {if_ : Flags} (hf : FlRel mf if_)
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

private theorem find_set_same_bounds (m : Mem) (idx : Nat) (r r' : Region)
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

private theorem find_bounds {m : Mem} {x i j : Nat} {r : Region} (h : Mem.find m x i = some (j, r)) :
    r.base ≤ x ∧ x < r.hi := by
  induction m generalizing i with
  | nil => simp [Mem.find] at h
  | cons s rest ih =>
    simp only [Mem.find] at h
    split at h
    · rename_i hc; simp at h; obtain ⟨_, rfl⟩ := h; exact hc
    · exact ih h

private theorem mem_wr_of_writable {mem : Mem} {a v : Nat} (hw : MemWritable mem a) :
    ∃ mem', Mem.wr mem a v = some mem' ∧
      (∀ x, MemWritable mem' x ↔ MemWritable mem x) ∧
      (∀ x, Mem.rd mem' x = if x = a then some (v % 256) else Mem.rd mem x) := by
  obtain ⟨idx, r, hfind, hrw⟩ := hw
  have hspec := find_spec mem a 0 hfind
  simp only [Nat.sub_zero] at hspec
  have hba := find_bounds hfind
  have hadata : a - r.base < r.data.length := by unfold Region.hi at hba; omega
  let r' : Region := { r with data := r.data.set (a - r.base) (v % 256) }
  have hr'_base : r'.base = r.base := rfl
  have hr'_hi : r'.hi = r.hi := by simp [r', Region.hi]
  let mem' := mem.set idx r'
  have hwr : Mem.wr mem a v = some mem' := by simp [Mem.wr, hfind, hrw, mem', r']
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
          have hne : a - q.2.base ≠ x - q.2.base := by rw [hpeq] at hxb ⊢; omega
          rw [← hpeq] at hadata
          simp [r', ← hpeq, hne]
        · rw [if_neg hq]

private theorem mem_wrN_spec (mem : Mem) (a : Nat) (bs : List Nat)
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

private theorem min_wrByte_spec {s : X86Min.St} (hlen : s.mem.length = X86Min.winSize)
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

private theorem min_wrBytes_spec (n : Nat) {s s' : X86Min.St} (hlen : s.mem.length = X86Min.winSize)
    (a v : Nat) (h : s.wrBytes a v n = some s') :
    s'.mem.length = X86Min.winSize ∧
    s'.lo = s.lo ∧ s'.regs = s.regs ∧ s'.rip = s.rip ∧
    s'.fl = s.fl ∧ s'.inp = s.inp ∧ s'.out = s.out ∧
    (∀ k < n, s.lo ≤ a + k ∧ a + k < s.lo + X86Min.winSize) ∧
    (∀ x, s'.rdByte x =
      if a ≤ x ∧ x < a + n then some (v / 256 ^ (x - a) % 256) else s.rdByte x) := by
  induction n generalizing s a v with
  | zero =>
    simp [X86Min.St.wrBytes] at h; subst h
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
        simp only [Option.some.injEq] at h1; subst h1
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
            · subst x; rw [if_pos (by omega)]; simp
            · have hnot : ¬ (a ≤ x ∧ x < a + (n + 1)) := by omega
              rw [if_neg hnot, if_neg hxa]
      · cases h1

private theorem le_to_rdN_leVal {f : Nat → Option Nat} {mem : Mem}
    (hle : ∀ x, X86Min.OLe (f x) (mem.rd x))
    (hb : ∀ x v, f x = some v → v < 256) :
    ∀ n off v, X86Min.le f off n = some v →
      ∃ bs, mem.rdN off n = some bs ∧ X86Int.leVal bs = v ∧ v < 256 ^ n := by
  intro n
  induction n with
  | zero =>
    intro off v h
    simp [X86Min.le] at h; subst h
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
        simp only [Option.some.injEq] at h; subst h
        obtain ⟨bs, hrd, hval, hlt⟩ := ih (off + 1) hi hl
        have hmem_lo := hle off lo hf
        have hlo_lt := hb off lo hf
        refine ⟨lo :: bs, ?_, ?_, ?_⟩
        · simp [Mem.rdN, hmem_lo, hrd]
        · simp [X86Int.leVal, hval]
        · rw [Nat.pow_succ']; omega

private theorem sim_setReg {m : X86Min.St} {i : St} (h : Sim m i)
    {d v : Nat} (hd : d < 16) (hv : v < W) (rip' : Nat) (hrip' : rip' < W)
    {mf' : X86Min.Flags} {if' : Flags} (hfl : FlRel mf' if') :
    Sim { (m.setReg d (some v)) with rip := rip', fl := mf' }
        { (i.set d (some v)) with rip := rip', fl := if' } where
  rip := rfl
  ripBound := hrip'
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

private theorem sim_with_rip_fl {m : X86Min.St} {i : St} (h : Sim m i)
    (rip' : Nat) (hrip' : rip' < W)
    {mf' : X86Min.Flags} {if' : Flags} (hfl : FlRel mf' if') :
    Sim { m with rip := rip', fl := mf' } { i with rip := rip', fl := if' } where
  rip := rfl
  ripBound := hrip'
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

private theorem sim_set_rip {m : X86Min.St} {i : St} (h : Sim m i)
    (rip' : Nat) (hrip' : rip' < W) :
    Sim { m with rip := rip' } { i with rip := rip' } :=
  sim_with_rip_fl h rip' hrip' h.fl

private theorem flRel_refl (f : X86Min.Flags) : FlRel f ⟨f.cf, f.zf, f.sf, f.of⟩ :=
  ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩

/-! ### Compatibility between `X86Min.Img` and `X86Uni` image -/

/-- Image compatibility between an `X86Min.Img` and a `X86Uni` decoded image:
every instruction decoded by `imMin` decodes identically in `imUni` (with no
`CommonIns` restriction), lives in canonical address space `a + len < W`, and
avoids `xor r11, r11` on an undefined `r11`. -/
structure MinCompat (imMin : X86Min.Img) (imUni : Nat → Option (Ins × Nat)) : Prop where
  agree : ∀ a ins len, imMin.decodeAt a = some (ins, len) →
    InsWf ins ∧ (∀ w, ins ≠ .rr .xor w 11 11) ∧ a + len < W ∧
    imUni a = some (toUniIns a len ins, len)
  /-- `jmp r64` jumps to a run-time register value, which need not keep `rip` canonical, so
  the simulation covers images without it (neither seed uses it). -/
  noJmpReg : ∀ a r len, imMin.decodeAt a = some (.jmpReg r, len) → False

/-- Conditions on an ELF code buffer `code` loaded at `vbase` ensuring `MinCompat`. -/
structure ImageOk (code : List Nat) (vbase : Nat) : Prop where
  bytesLt256 : ∀ b ∈ code, b < 256
  lenBound : code.length ≤ 2 ^ X86Min.tdepth
  vbaseCanon : vbase + code.length + 16 < W
  noXor11 : ∀ a ins len,
    (X86Min.Img.mk' code vbase code.length).decodeAt a = some (ins, len) →
    ∀ w, ins ≠ .rr .xor w 11 11
  noJmpReg : ∀ a r len,
    (X86Min.Img.mk' code vbase code.length).decodeAt a = some (.jmpReg r, len) → False

def checkImageOk (code : List Nat) (vbase : Nat) : Bool :=
  code.all (· < 256) &&
  decide (code.length ≤ 2 ^ X86Min.tdepth) &&
  decide (vbase + code.length + 16 < W) &&
  (List.range code.length).all fun off =>
    match (X86Min.Img.mk' code vbase code.length).decodeAt (vbase + off) with
    | some (.rr .xor _ 11 11, _) => false
    | some (.jmpReg _, _) => false
    | _ => true

/-- `decodeAt` on a built image, as a range fact and a table lookup. Stated once: the
`if` inside `decodeAt` carries decidability instances that still mention the image after
unfolding, which `split` and `rw` in Lean 4.33 do not see through. -/
theorem decodeAt_mk'_eq_some {code : List Nat} {vbase a : Nat} {x : X86Min.Ins × Nat}
    (h : (X86Min.Img.mk' code vbase code.length).decodeAt a = some x) :
    (vbase ≤ a ∧ a < vbase + code.length) ∧
      X86Min.Tree.get none X86Min.tdepth (X86Min.Img.mk' code vbase code.length).dec (a - vbase) = some x := by
  have h' := h
  simp [X86Min.Img.decodeAt, X86Min.Img.mk'] at h'
  exact ⟨⟨of_decide_eq_true h'.1.1, of_decide_eq_true h'.1.2⟩, h'.2⟩

theorem decodeAt_mk'_of_range {code : List Nat} {vbase a : Nat}
    (h1 : vbase ≤ a) (h2 : a < vbase + code.length) :
    (X86Min.Img.mk' code vbase code.length).decodeAt a =
      X86Min.Tree.get none X86Min.tdepth (X86Min.Img.mk' code vbase code.length).dec (a - vbase) := by
  simp [X86Min.Img.decodeAt, X86Min.Img.mk', h1, h2]

theorem imageOk_of_check {code : List Nat} {vbase : Nat}
    (h : checkImageOk code vbase = true) : ImageOk code vbase := by
  simp only [checkImageOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨hbytes, hlen⟩, hcanon⟩, hnox⟩ := h
  refine ⟨fun b hb => of_decide_eq_true (List.all_eq_true.1 hbytes b hb), hlen, hcanon, ?_, ?_⟩
  · intro a ins len hd w heq
    subst heq
    obtain ⟨hrange, hd⟩ := decodeAt_mk'_eq_some hd
    have hoff : a - vbase < code.length := by omega
    have hchk := List.all_eq_true.1 hnox (a - vbase) (List.mem_range.2 hoff)
    have ha_eq : vbase + (a - vbase) = a := by omega
    rw [ha_eq] at hchk
    rw [decodeAt_mk'_of_range hrange.1 hrange.2, hd] at hchk
    cases hchk
  · intro a r len hd
    obtain ⟨hrange, hd⟩ := decodeAt_mk'_eq_some hd
    have hoff : a - vbase < code.length := by omega
    have hchk := List.all_eq_true.1 hnox (a - vbase) (List.mem_range.2 hoff)
    have ha_eq : vbase + (a - vbase) = a := by omega
    rw [ha_eq] at hchk
    rw [decodeAt_mk'_of_range hrange.1 hrange.2, hd] at hchk
    cases hchk

theorem fetchIn_eq_getElem? (code : List Nat) (i : Nat) :
    X86Min.fetchIn code code.length i = code[i]? := by
  unfold X86Min.fetchIn
  by_cases hi : i < code.length
  · simp [hi]
  · simp [hi]

theorem minCompat_of_imageOk {code : List Nat} {vbase : Nat} (hok : ImageOk code vbase) :
    MinCompat (X86Min.Img.mk' code vbase code.length) (imageOf code vbase) := by
  refine ⟨?_, fun a r len hd => hok.noJmpReg a r len hd⟩
  intro a ins len hd
  have hnox := hok.noXor11 a ins len hd
  obtain ⟨hrange, hd⟩ := decodeAt_mk'_eq_some hd
  have hidx : a - vbase < 2 ^ X86Min.tdepth := Nat.lt_of_lt_of_le (by omega) hok.lenBound
  have hd' : X86Min.Tree.get none X86Min.tdepth (X86Min.Img.mk' code vbase code.length).dec (a - vbase) = some (ins, len) := hd
  rw [X86Min.Img.dec_spec code vbase code.length (a - vbase) hidx] at hd'
  have hb256 : ∀ i v, (fun j => X86Min.fetchIn code code.length (a - vbase + j)) i = some v → v < 256 := by
    intro i v hv
    dsimp only at hv
    rw [fetchIn_eq_getElem?] at hv
    exact hok.bytesLt256 v (List.mem_of_getElem? hv)
  obtain ⟨huni, hlen7, hwf⟩ := decode_min_agree (a := a) hb256 hd'
  have him : imageOf code vbase a = some (toUniIns a len ins, len) := by
    unfold imageOf
    rw [if_neg (by omega)]
    have hfn : (fun i => code[a - vbase + i]?) = (fun j => X86Min.fetchIn code code.length (a - vbase + j)) := by
      funext j; exact (fetchIn_eq_getElem? code (a - vbase + j)).symm
    rw [hfn]
    exact huni
  have hacanon : a + len < W := by have := hok.vbaseCanon; omega
  exact ⟨hwf, hnox, hacanon, him⟩

/-! ### Single-step and multi-step forward simulation (`X86Min -> X86Uni`) -/

private theorem bytes8_len (v : Nat) : (X86Int.bytes8 v).length = 8 := by
  simp [X86Int.bytes8]

private theorem bytes8_getElem (v i : Nat) (hi : i < 8)
    (hlt : i < (X86Int.bytes8 v).length) :
    (X86Int.bytes8 v)[i]'hlt = v / 256 ^ i % 256 := by
  rcases i with _ | _ | _ | _ | _ | _ | _ | _ | k
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact absurd hi (by omega)

theorem step_min_sim {imMin : X86Min.Img} {imUni : Nat → Option (Ins × Nat)}
    (hcompat : MinCompat imMin imUni)
    {m : X86Min.St} {i : St} (hsim : Sim m i) :
    (∀ m', X86Min.step imMin m = .inl m' →
      ∃ i', step imUni i = .ok i' ∧ Sim m' i') ∧
    (∀ code out, X86Min.step imMin m = .inr (.exit code out) →
      step imUni i = .exit code ∧ i.k.out = out) := by
  unfold X86Min.step step
  rw [← hsim.rip]
  cases hdec : imMin.decodeAt m.rip with
  | none => simp
  | some p =>
    obtain ⟨ins, len⟩ := p
    obtain ⟨hwf, hno11, hcanon, him⟩ := hcompat.agree m.rip ins len hdec
    rw [him]
    dsimp only
    have hnx_mod : (m.rip + len) % W = m.rip + len := Nat.mod_eq_of_lt hcanon
    have hnx_eq : (m.rip + len) % X86Min.pow2 64 = m.rip + len := by
      rw [pow2_64_eq_W]; exact hnx_mod
    have hrip_int : i.rip + len = m.rip + len := by rw [← hsim.rip]
    cases ins with
    | movImm d imm =>
      obtain ⟨hd16, himm32⟩ := hwf
      have himmW : imm < W := Nat.lt_trans himm32 (by decide)
      have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW
      simp only [X86Min.okOf, X86Min.St.wr, beq_iff_eq, show (32 : Nat) ≠ 8 by decide, if_false,
        exec, toUniIns, X86Int.exec, himm_mod]
      constructor
      · intro m' hm'; cases hm'
        refine ⟨_, rfl, ?_⟩
        simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
          sim_setReg hsim hd16 himmW (m.rip + len) hcanon hsim.fl
      · intro code out h; cases h
    | rr op w d src =>
      obtain ⟨hd16, hs16, hw3264, hwf_op⟩ := hwf
      cases op with
      | cmp =>
        rcases hwf_op with h | h | h | h | h | h <;> cases h
      | add =>
        have hzero_f : (X86Min.Op.add == .xor && d == src) = false := by simp
        rcases hw3264 with rfl | rfl
        · -- add32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, hzero_f, Bool.false_eq_true, if_false]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            cases hms : m.reg src with
            | none => simp [X86Min.St.rd, hmd, hms]
            | some vb =>
              obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
              obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
              have hres_lt : (va % 2 ^ 32 + vb % 2 ^ 32) % 2 ^ 32 < W :=
                Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
              have hfl : FlRel (X86Min.alu .add 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2
                  (aluW .add 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2 := by
                simp only [X86Min.alu, aluW, msbW_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, pow2_32_eq, hmd, hms, hid, his, Option.map_some,
                bind, Option.bind, pure, X86Min.St.wr,
                show (32 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.add == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW, pow2_32_eq] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
        · -- add64
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
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
              have hres_lt : (va + vb) % W < W := Nat.mod_lt _ (by decide)
              have hfl : FlRel (X86Min.alu .add 64 va vb).2 (X86Int.aluF .add va vb).2 := by
                simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
                bind, Option.bind, pure, X86Min.St.wr,
                show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.add == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpaX [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
      | sub =>
        have hzero_f : (X86Min.Op.sub == .xor && d == src) = false := by simp
        rcases hw3264 with rfl | rfl
        · -- sub32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, hzero_f, Bool.false_eq_true, if_false]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            cases hms : m.reg src with
            | none => simp [X86Min.St.rd, hmd, hms]
            | some vb =>
              obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
              obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
              have hres_lt : (va % 2 ^ 32 + 2 ^ 32 - vb % 2 ^ 32) % 2 ^ 32 < W :=
                Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
              have hfl : FlRel (X86Min.alu .sub 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2
                  (aluW .sub 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2 := by
                simp only [X86Min.alu, aluW, msbW_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, pow2_32_eq, hmd, hms, hid, his, Option.map_some,
                bind, Option.bind, pure, X86Min.St.wr,
                show (32 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.sub == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpaX [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
        · -- sub64
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
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
              have hres_lt : (va + W - vb) % W < W := Nat.mod_lt _ (by decide)
              have hfl : FlRel (X86Min.alu .sub 64 va vb).2 (X86Int.aluF .sub va vb).2 := by
                simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
                bind, Option.bind, pure, X86Min.St.wr,
                show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.sub == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpaX [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
      | mov =>
        rcases hw3264 with rfl | rfl
        · -- mov32
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf]
          cases hms : m.reg src with
          | none => simp [X86Min.St.rd, hms]
          | some vs =>
            obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
            have hv32_lt : vs % 2 ^ 32 < W := Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
            simp only [X86Min.St.rd, pow2_32_eq, hms, his, Option.map_some, bind, Option.bind,
              X86Min.St.wr, show (32 == 8) = false by decide, Bool.false_eq_true, if_false, pure]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
                sim_setReg hsim hd16 hv32_lt (m.rip + len) hcanon hsim.fl
            · intro code out h; cases h
        · -- mov64
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf]
          cases hms : m.reg src with
          | none => simp [X86Min.St.rd, hms]
          | some vs =>
            obtain ⟨his, hvsW⟩ := reg_lt16_bound hsim hs16 hms
            have hvs_mod : vs % X86Min.pow2 64 = vs := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvsW
            simp only [X86Min.St.rd, hms, his, Option.map_some, hvs_mod, bind, Option.bind,
              X86Min.St.wr, show (64 == 8) = false by decide, Bool.false_eq_true, if_false, pure]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
                sim_setReg hsim hd16 hvsW (m.rip + len) hcanon hsim.fl
            · intro code out h; cases h
      | xor =>
        by_cases hds : d = src
        · subst hds
          have hd11 : d ≠ 11 := fun h => by subst h; exact hno11 w rfl
          obtain ⟨va, hia⟩ := Option.isSome_iff_exists.1 (hsim.regsDef d hd16 hd11)
          have hzero : Nat.xor va va = 0 := Nat.xor_self va
          rcases hw3264 with rfl | rfl
          · -- xor32 d d
            have hfl : FlRel (X86Min.alu .xor 32 0 0).2 (X86Int.aluF .xor va va).2 := by
              simp only [X86Min.alu, X86Int.aluF, hzero, X86Min.msb, X86Int.msb]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [toUniIns, beq_self_eq_true, if_true, exec, X86Int.exec, X86Int.lift,
              X86Min.okOf, hia, if_true,
              bind, Option.bind, pure, X86Min.St.wr, show (32 == 8) = false by decide,
              Bool.false_eq_true, if_false, show (X86Min.Op.xor == .test) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              have hres0 : (X86Min.alu .xor 32 0 0).1 = 0 := by simp [X86Min.alu]
              rw [hres0]
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Int.aluF, hzero] using
                sim_setReg hsim hd16 (by decide : 0 < W) (m.rip + len) hcanon hfl
            · intro code out h; cases h
          · -- xor64 d d
            have hfl : FlRel (X86Min.alu .xor 64 0 0).2 (X86Int.aluF .xor va va).2 := by
              simp only [X86Min.alu, X86Int.aluF, hzero, X86Min.msb, X86Int.msb]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hia,
              show (X86Min.Op.xor == .xor && d == d) = true by simp, if_true,
              bind, Option.bind, pure, X86Min.St.wr, show (64 == 8) = false by decide,
              Bool.false_eq_true, if_false, show (X86Min.Op.xor == .test) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              have hres0 : (X86Min.alu .xor 64 0 0).1 = 0 := by simp [X86Min.alu]
              rw [hres0]
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Int.aluF, hzero] using
                sim_setReg hsim hd16 (by decide : 0 < W) (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · have hzero_f : (X86Min.Op.xor == .xor && d == src) = false := by simp [hds]
          rcases hw3264 with rfl | rfl
          · -- xor32 d src (d ≠ src)
            have hbeq : (d == src) = false := by simp [hds]
            simp only [toUniIns, hbeq, Bool.false_eq_true, if_false, exec, X86Int.lift, X86Min.okOf]
            cases hmd : m.reg d with
            | none => simp [X86Min.St.rd, hmd]
            | some va =>
              cases hms : m.reg src with
              | none => simp [X86Min.St.rd, hmd, hms]
              | some vb =>
                obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
                obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
                have hres_lt : Nat.xor (va % 2 ^ 32) (vb % 2 ^ 32) < W :=
                  Nat.lt_trans (Nat.xor_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))) (by decide)
                have hfl : FlRel (X86Min.alu .xor 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2
                    (aluW .xor 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2 := by
                  simp only [X86Min.alu, aluW, msbW_eq]
                  exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
                simp only [X86Min.St.rd, pow2_32_eq, hmd, hms, hid, his, Option.map_some,
                  Bool.false_eq_true, if_false, bind, Option.bind, pure,
                  X86Min.St.wr, show (32 == 8) = false by decide,
                  show (X86Min.Op.xor == .test) = false by decide]
                constructor
                · intro m' hm'; cases hm'
                  refine ⟨_, rfl, ?_⟩
                  simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW] using
                    sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
                · intro code out h; cases h
          · -- xor64 d src
            simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
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
                have hres_lt : Nat.xor va vb < W := Nat.xor_lt_two_pow hvaW hvbW
                have hfl : FlRel (X86Min.alu .xor 64 va vb).2 (X86Int.aluF .xor va vb).2 := by
                  simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
                  exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
                simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
                  bind, Option.bind, pure, X86Min.St.wr,
                  show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
                  show (X86Min.Op.xor == .test) = false by decide]
                constructor
                · intro m' hm'; cases hm'
                  refine ⟨_, rfl, ?_⟩
                  simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                    sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
                · intro code out h; cases h
      | or =>
        have hzero_f : (X86Min.Op.or == .xor && d == src) = false := by simp
        rcases hw3264 with rfl | rfl
        · -- or32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, hzero_f, Bool.false_eq_true, if_false]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            cases hms : m.reg src with
            | none => simp [X86Min.St.rd, hmd, hms]
            | some vb =>
              obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
              obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
              have hres_lt : Nat.lor (va % 2 ^ 32) (vb % 2 ^ 32) < W :=
                Nat.lt_trans (Nat.or_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))) (by decide)
              have hfl : FlRel (X86Min.alu .or 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2
                  (aluW .or 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2 := by
                simp only [X86Min.alu, aluW, msbW_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, pow2_32_eq, hmd, hms, hid, his, Option.map_some,
                bind, Option.bind, pure, X86Min.St.wr,
                show (32 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.or == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
        · -- or64
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
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
              have hres_lt : Nat.lor va vb < W := Nat.or_lt_two_pow hvaW hvbW
              have hfl : FlRel (X86Min.alu .or 64 va vb).2 (X86Int.aluF .or va vb).2 := by
                simp only [X86Min.alu, X86Int.aluF, msb_64_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, hmd, hms, hid, his, Option.map_some, hva_mod, hvb_mod,
                bind, Option.bind, pure, X86Min.St.wr,
                show (64 == 8) = false by decide, Bool.false_eq_true, if_false,
                show (X86Min.Op.or == .test) = false by decide]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, X86Int.aluF] using
                  sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
              · intro code out h; cases h
      | test =>
        have hzero_f : (X86Min.Op.test == .xor && d == src) = false := by simp
        rcases hw3264 with rfl | rfl
        · -- test32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, hzero_f, Bool.false_eq_true, if_false]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            cases hms : m.reg src with
            | none => simp [X86Min.St.rd, hmd, hms]
            | some vb =>
              obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
              obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
              have hfl : FlRel (X86Min.alu .test 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2
                  (aluW .test 32 (va % 2 ^ 32) (vb % 2 ^ 32)).2 := by
                simp only [X86Min.alu, aluW, msbW_eq]
                exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
              simp only [X86Min.St.rd, pow2_32_eq, hmd, hms, hid, his, Option.map_some,
                bind, Option.bind, pure, show (X86Min.Op.test == .test) = true by decide, if_true]
              constructor
              · intro m' hm'; cases hm'
                refine ⟨_, rfl, ?_⟩
                simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hcanon hfl
              · intro code out h; cases h
        · -- test64
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, hzero_f,
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
                refine ⟨_, rfl, ?_⟩
                simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hcanon hfl
              · intro code out h; cases h
    | ri op w d imm =>
      rcases hwf with ⟨rfl, hw3264, hd16, himmW⟩ | ⟨hop8, rfl, rfl, himm8⟩ | ⟨hopAS, hw3264, hd16, himmW⟩
      · -- cmp r32/r64, imm
        rcases hw3264 with rfl | rfl
        · -- cmpI32
          have himm_mod : imm % 2 ^ 32 = imm := Nat.mod_eq_of_lt (by simpa [pow2_32_eq] using himmW)
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
            have hfl : FlRel (X86Min.alu .cmp 32 (va % 2 ^ 32) imm).2
                (aluW .cmp 32 (va % 2 ^ 32) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_32_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, pow2_32_eq, hmd, hid, Option.map_some, bind, Option.bind,
              pure, show (X86Min.Op.cmp == .cmp) = true by decide, if_true]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- cmpI64
          have himm_mod : imm % W = imm := Nat.mod_eq_of_lt (by simpa [pow2_64_eq_W] using himmW)
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
            have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
            have hfl : FlRel (X86Min.alu .cmp 64 va imm).2 (X86Int.aluF .sub va imm).2 := by
              simp only [X86Min.alu, X86Int.aluF, msb_64_eq, pow2_64_eq_W]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, hmd, hid, Option.map_some, hva_mod, bind, Option.bind,
              pure, show (X86Min.Op.cmp == .cmp) = true by decide, if_true]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hcanon hfl
            · intro code out h; cases h
      · -- AL 8-bit imm
        have himm_mod : imm % 256 = imm := Nat.mod_eq_of_lt himm8
        rcases hop8 with rfl | rfl | rfl
        · -- 04 add
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod, NarrowAlu8.toAluOp]
          cases hm0 : m.reg 0 with
          | none => simp [X86Min.St.rd, hm0]
          | some old =>
            obtain ⟨hi0, holdW⟩ := reg_lt16_bound hsim (by decide : 0 < 16) hm0
            have hm_rd : m.rd 8 0 = some (old % 256) := by simp [X86Min.St.rd, pow2_8_eq, hm0]
            have hres_lt : (old % 256 + imm) % 256 < 256 := Nat.mod_lt _ (by decide)
            have hnew_lt : old - old % 256 + (old % 256 + imm) % 256 < W := by
              unfold W at holdW ⊢; omega
            have hfl : FlRel (X86Min.alu .add 8 (old % 256) imm).2
                (aluW .add 8 (old % 256) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_8_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [hm_rd, hi0, bind, Option.bind, pure,
              (show (X86Min.Op.add == .cmp) = false by decide), Bool.false_eq_true, if_false,
              X86Min.St.wr, (show (8 == 8) = true by decide), if_true, X86Min.St.reg]
            have hreg0 : ∀ (st : X86Min.St), st.regs = m.regs →
                (st.regs[0]?).getD none = some old := fun st h => by rw [h]; exact hm0
            rw [hreg0 _ rfl]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, X86Min.alu, aluW, pow2_8_eq, hnx_eq, hrip_int] using
                sim_setReg hsim (by decide : 0 < 16) hnew_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- 2c sub
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod, NarrowAlu8.toAluOp]
          cases hm0 : m.reg 0 with
          | none => simp [X86Min.St.rd, hm0]
          | some old =>
            obtain ⟨hi0, holdW⟩ := reg_lt16_bound hsim (by decide : 0 < 16) hm0
            have hm_rd : m.rd 8 0 = some (old % 256) := by simp [X86Min.St.rd, pow2_8_eq, hm0]
            have hres_lt : (old % 256 + 256 - imm) % 256 < 256 := Nat.mod_lt _ (by decide)
            have hnew_lt : old - old % 256 + (old % 256 + 256 - imm) % 256 < W := by
              unfold W at holdW ⊢; omega
            have hfl : FlRel (X86Min.alu .sub 8 (old % 256) imm).2
                (aluW .sub 8 (old % 256) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_8_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [hm_rd, hi0, bind, Option.bind, pure,
              (show (X86Min.Op.sub == .cmp) = false by decide), Bool.false_eq_true, if_false,
              X86Min.St.wr, (show (8 == 8) = true by decide), if_true, X86Min.St.reg]
            have hreg0 : ∀ (st : X86Min.St), st.regs = m.regs →
                (st.regs[0]?).getD none = some old := fun st h => by rw [h]; exact hm0
            rw [hreg0 _ rfl]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, X86Min.alu, aluW, pow2_8_eq, hnx_eq, hrip_int] using
                sim_setReg hsim (by decide : 0 < 16) hnew_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- 3c cmp
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod, NarrowAlu8.toAluOp]
          cases hm0 : m.reg 0 with
          | none => simp [X86Min.St.rd, hm0]
          | some old =>
            obtain ⟨hi0, _⟩ := reg_lt16_bound hsim (by decide : 0 < 16) hm0
            have hfl : FlRel (X86Min.alu .cmp 8 (old % 256) imm).2
                (aluW .cmp 8 (old % 256) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_8_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, pow2_8_eq, hm0, hi0, Option.map_some, bind, Option.bind,
              pure, show (X86Min.Op.cmp == .cmp) = true by decide, if_true]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [hnx_eq, hrip_int] using sim_with_rip_fl hsim (m.rip + len) hcanon hfl
            · intro code out h; cases h
      · -- add/sub r32/r64, imm
        rcases hopAS with rfl | rfl <;> rcases hw3264 with rfl | rfl
        · -- add r32, imm
          have himm32 : imm < 2 ^ 32 := by simpa [pow2_32_eq] using himmW
          have himm_mod : imm % 2 ^ 32 = imm := Nat.mod_eq_of_lt himm32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod, NarrowAlu8.toAluOp]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
            have hres_lt : (X86Min.alu .add 32 (va % 2 ^ 32) imm).1 < W := by
              simp only [X86Min.alu, pow2_32_eq]; exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
            have hfl : FlRel (X86Min.alu .add 32 (va % 2 ^ 32) imm).2 (aluW .add 32 (va % 2 ^ 32) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_32_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, pow2_32_eq, hmd, hid, Option.map_some, bind, Option.bind,
              pure, show (X86Min.Op.add == .cmp) = false by decide, Bool.false_eq_true, if_false,
              X86Min.St.wr, show (32 == 8) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW, pow2_32_eq] using
                sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- add r64, imm
          have himmW' : imm < W := by simpa [pow2_64_eq_W] using himmW
          have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW'
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
            have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
            have hres_lt : (X86Int.aluF .add va imm).1 < W := by
              simp only [X86Int.aluF]; exact Nat.mod_lt _ (by decide)
            have hfl : FlRel (X86Min.alu .add 64 va imm).2 (X86Int.aluF .add va imm).2 := by
              simp only [X86Min.alu, X86Int.aluF, msb_64_eq, pow2_64_eq_W]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, hmd, hid, Option.map_some, hva_mod, bind, Option.bind,
              pure, show (X86Min.Op.add == .cmp) = false by decide, Bool.false_eq_true, if_false,
              X86Min.St.wr, show (64 == 8) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hnx_mod, hrip_int, X86Min.alu, X86Int.aluF, pow2_64_eq_W] using
                sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- sub r32, imm
          have himm32 : imm < 2 ^ 32 := by simpa [pow2_32_eq] using himmW
          have himm_mod : imm % 2 ^ 32 = imm := Nat.mod_eq_of_lt himm32
          simp only [toUniIns, exec, X86Int.lift, X86Min.okOf, himm_mod, NarrowAlu8.toAluOp]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
            have hres_lt : (X86Min.alu .sub 32 (va % 2 ^ 32) imm).1 < W := by
              simp only [X86Min.alu, pow2_32_eq]; exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
            have hfl : FlRel (X86Min.alu .sub 32 (va % 2 ^ 32) imm).2 (aluW .sub 32 (va % 2 ^ 32) imm).2 := by
              simp only [X86Min.alu, aluW, msbW_eq, pow2_32_eq]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, pow2_32_eq, hmd, hid, Option.map_some, bind, Option.bind,
              pure, show (X86Min.Op.sub == .cmp) = false by decide, Bool.false_eq_true, if_false,
              X86Min.St.wr, show (32 == 8) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int, X86Min.alu, aluW, pow2_32_eq] using
                sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
        · -- sub r64, imm
          have himmW' : imm < W := by simpa [pow2_64_eq_W] using himmW
          have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW'
          simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
          cases hmd : m.reg d with
          | none => simp [X86Min.St.rd, hmd]
          | some va =>
            obtain ⟨hid, hvaW⟩ := reg_lt16_bound hsim hd16 hmd
            have hva_mod : va % X86Min.pow2 64 = va := by rw [pow2_64_eq_W]; exact Nat.mod_eq_of_lt hvaW
            have hres_lt : (X86Int.aluF .sub va imm).1 < W := by
              simp only [X86Int.aluF]; exact Nat.mod_lt _ (by decide)
            have hfl : FlRel (X86Min.alu .sub 64 va imm).2 (X86Int.aluF .sub va imm).2 := by
              simp only [X86Min.alu, X86Int.aluF, msb_64_eq, pow2_64_eq_W]
              exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
            simp only [X86Min.St.rd, hmd, hid, Option.map_some, hva_mod, bind, Option.bind,
              pure, show (X86Min.Op.sub == .cmp) = false by decide, Bool.false_eq_true, if_false,
              X86Min.St.wr, show (64 == 8) = false by decide]
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hnx_mod, hrip_int, X86Min.alu, X86Int.aluF, pow2_64_eq_W] using
                sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
            · intro code out h; cases h
    | shlI w d c =>
      obtain ⟨rfl, hd16, _⟩ := hwf
      simp only [toUniIns, exec, X86Int.lift, X86Min.okOf]
      cases hmd : m.reg d with
      | none => simp [X86Min.St.rd, hmd]
      | some va =>
        obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
        have ha_lt : va % 2 ^ 32 < W := Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
        simp only [X86Min.St.rd, pow2_32_eq, hmd, hid, Option.map_some, bind, Option.bind, pure]
        by_cases hcnt0 : (c % 32 == 0) = true
        · have hcnt0' : c % 32 = 0 := eq_of_beq hcnt0
          simp only [hcnt0, if_true, X86Min.St.wr, show (32 == 8) = false by decide,
            Bool.false_eq_true, if_false]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
              sim_setReg hsim hd16 ha_lt (m.rip + len) hcanon hsim.fl
          · intro code out h; cases h
        · simp only [hcnt0, Bool.false_eq_true, if_false, X86Min.St.wr,
            show (32 == 8) = false by decide]
          have hres_lt : (va % 2 ^ 32) * 2 ^ (c % 32) % 2 ^ 32 < W :=
            Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
          have hfl : FlRel
              ⟨some ((va % 2 ^ 32) / X86Min.pow2 (32 - c % 32) % 2 == 1),
               some (((va % 2 ^ 32) * X86Min.pow2 (c % 32) % 2 ^ 32) == 0),
               some (X86Min.msb 32 ((va % 2 ^ 32) * X86Min.pow2 (c % 32) % 2 ^ 32)),
               if c % 32 == 1 then
                 some (X86Min.msb 32 ((va % 2 ^ 32) * X86Min.pow2 (c % 32) % 2 ^ 32) !=
                   ((va % 2 ^ 32) / X86Min.pow2 (32 - c % 32) % 2 == 1))
               else none⟩
              ⟨some ((va % 2 ^ 32) / 2 ^ (32 - c % 32) % 2 == 1),
               some (((va % 2 ^ 32) * 2 ^ (c % 32) % 2 ^ 32) == 0),
               some (msbW 32 ((va % 2 ^ 32) * 2 ^ (c % 32) % 2 ^ 32)),
               if c % 32 == 1 then
                 some (msbW 32 ((va % 2 ^ 32) * 2 ^ (c % 32) % 2 ^ 32) !=
                   ((va % 2 ^ 32) / 2 ^ (32 - c % 32) % 2 == 1))
               else none⟩ := by
            simp only [msbW_eq]; exact ⟨X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            rw [hnx_eq]
            simpa [X86Min.St.setReg, X86Int.St.set, hrip_int, X86Min.pow2] using
              sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
          · intro code out h; cases h
    | inc w d =>
      obtain ⟨hw3264, hd16⟩ := hwf
      simp only [toUniIns, exec, X86Int.lift, X86Min.okOf]
      cases hmd : m.reg d with
      | none => simp [X86Min.St.rd, hmd]
      | some va =>
        obtain ⟨hid, _⟩ := reg_lt16_bound hsim hd16 hmd
        have hw_not8 : (w == 8) = false := by rcases hw3264 with rfl | rfl <;> decide
        have hres_lt : (va % 2 ^ w + 1) % 2 ^ w < W := by
          rcases hw3264 with rfl | rfl
          · exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
          · exact Nat.mod_lt _ (by decide)
        have hfl : FlRel
            { m.fl with zf := some (((va % X86Min.pow2 w + 1) % X86Min.pow2 w) == 0),
                        sf := some (X86Min.msb w ((va % X86Min.pow2 w + 1) % X86Min.pow2 w)),
                        of := some (((va % X86Min.pow2 w + 1) % X86Min.pow2 w) == X86Min.pow2 (w - 1)) }
            { i.fl with zf := some (((va % 2 ^ w + 1) % 2 ^ w) == 0),
                        sf := some (msbW w ((va % 2 ^ w + 1) % 2 ^ w)),
                        of := some (((va % 2 ^ w + 1) % 2 ^ w) == 2 ^ (w - 1)) } := by
          simp only [msbW_eq, X86Min.pow2]
          exact ⟨hsim.fl.cf, X86Min.OLe.refl _, X86Min.OLe.refl _, X86Min.OLe.refl _⟩
        simp only [X86Min.St.rd, hmd, hid, Option.map_some, bind, Option.bind, pure,
          X86Min.St.wr, hw_not8, Bool.false_eq_true, if_false]
        constructor
        · intro m' hm'; cases hm'
          refine ⟨_, rfl, ?_⟩
          rw [hnx_eq]
          simpa [X86Min.St.setReg, X86Int.St.set, hrip_int, X86Min.pow2] using
            sim_setReg hsim hd16 hres_lt (m.rip + len) hcanon hfl
        · intro code out h; cases h
    | push imm =>
      have himmW : imm < W := hwf
      have himm_mod : imm % W = imm := Nat.mod_eq_of_lt himmW
      simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf, himm_mod]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        simp only [hi4, bind, Option.bind, pure]
        cases hwr_m : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).wrBytes
            ((sp + X86Min.pow2 64 - 8) % X86Min.pow2 64) imm 8 with
        | none => simp
        | some t =>
          have ht_len : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).mem.length = X86Min.winSize :=
            hsim.memLen
          obtain ⟨hmlen', hlo', hregs', hrip', hfl', hinp', hout', hbounds, hrd'⟩ :=
            min_wrBytes_spec 8 ht_len ((sp + X86Min.pow2 64 - 8) % X86Min.pow2 64) imm hwr_m
          let sp' := (sp + W - 8) % W
          have hsp'_eq : (sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 = sp' := rfl
          have hsp'_lt : sp' < W := Nat.mod_lt _ (by decide)
          have hw_i : ∀ k < (X86Int.bytes8 imm).length, MemWritable i.mem (sp' + k) := by
            intro k hk
            simp only [X86Int.bytes8, List.length_map, List.length_range] at hk
            have hbk := hbounds k hk
            rw [hsp'_eq] at hbk
            exact hsim.stackWritable (sp' + k) hbk.1 hbk.2
          have hb_i : ∀ b ∈ X86Int.bytes8 imm, b < 256 := by
            intro b hb
            simp only [X86Int.bytes8, List.mem_map, List.mem_range] at hb
            obtain ⟨_, _, rfl⟩ := hb
            exact Nat.mod_lt _ (by decide)
          obtain ⟨mem', hwrN, hwritable', hrd_i⟩ := mem_wrN_spec i.mem sp' (X86Int.bytes8 imm) hw_i hb_i
          have hwr64 : X86Int.wr64 i.mem sp' imm = some mem' := by
            simp [X86Int.wr64, hwrN]
          rw [hwr64]
          dsimp only
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            exact {
              rip := by simp [X86Min.St.setReg, hrip', hnx_eq, hrip_int]
              ripBound := by simp [X86Min.St.setReg, hrip', hnx_eq, hcanon]
              mLen := by simp [X86Min.St.setReg, hregs', hsim.mLen]
              iLen := by simp [X86Int.St.set, hsim.iLen]
              regsLe := fun r hr => by
                change X86Min.OLe ((t.setReg 4 (some sp')).reg r) ((i.set 4 (some sp')).reg r)
                have hlen_t : t.regs.length = 16 := by rw [hregs']; exact hsim.mLen
                rw [m_reg_setReg hlen_t (by decide : 4 < 16), i_reg_set hsim.iLen (by decide : 4 < 16)]
                by_cases hr4 : r = 4
                · simp [hr4, X86Min.OLe.refl]
                · simp only [hr4, if_false]; rw [X86Min.St.reg, hregs']; exact hsim.regsLe r hr
              regsDef := fun r hr hr11 => by
                change ((i.set 4 (some sp')).reg r).isSome = true
                rw [i_reg_set hsim.iLen (by decide : 4 < 16)]
                by_cases hr4 : r = 4
                · simp [hr4]
                · simp [hr4]; exact hsim.regsDef r hr hr11
              regsBound := fun r u hu => by
                change (i.set 4 (some sp')).reg r = some u at hu
                by_cases hr16 : r < 16
                · rw [i_reg_set hsim.iLen (by decide : 4 < 16)] at hu
                  by_cases hr4 : r = 4
                  · simp [hr4] at hu; subst hu; exact hsp'_lt
                  · simp [hr4] at hu; exact hsim.regsBound r u hu
                · have hlen : (i.set 4 (some sp')).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                  have hnone : (i.set 4 (some sp')).regs[r]? = none :=
                    List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                  simp [X86Int.St.reg, hnone] at hu
              fl := by simpaX [X86Min.St.setReg, hfl'] using hsim.fl
              memLen := by simpa [X86Min.St.setReg] using hmlen'
              loCanon := by simpa [X86Min.St.setReg, hlo'] using hsim.loCanon
              stackWritable := fun x hx1 hx2 => by
                rw [hwritable']
                exact hsim.stackWritable x (by simpa [X86Min.St.setReg, hlo'] using hx1)
                  (by simpa [X86Min.St.setReg, hlo'] using hx2)
              memLe := fun x u hu => by
                have hu' : t.rdByte x = some u := by
                  simpaX [X86Min.St.setReg, X86Min.St.rdByte] using hu
                rw [hrd'] at hu'
                rw [hrd_i]
                by_cases hx : sp' ≤ x ∧ x < sp' + 8
                · have hxm : (sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 ≤ x ∧
                      x < (sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 + 8 := by simpa [hsp'_eq] using hx
                  rw [if_pos hxm] at hu'
                  have hxi : sp' ≤ x ∧ x < sp' + (X86Int.bytes8 imm).length := by
                    simpa [bytes8_len] using hx
                  rw [dif_pos hxi]
                  rw [bytes8_getElem imm (x - sp') (by omega), ← hu']; rfl
                · have hxm : ¬ ((sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 ≤ x ∧
                      x < (sp + X86Min.pow2 64 - 8) % X86Min.pow2 64 + 8) := by simpa [hsp'_eq] using hx
                  rw [if_neg hxm] at hu'
                  have hxi : ¬ (sp' ≤ x ∧ x < sp' + (X86Int.bytes8 imm).length) := by
                    simpa [bytes8_len] using hx
                  rw [dif_neg hxi]
                  have hu_m : m.rdByte x = some u := by simpa [X86Min.St.rdByte] using hu'
                  exact hsim.memLe x u hu_m
              memBound := fun x u hu => by
                rw [hrd_i] at hu
                split at hu
                · rename_i hx
                  have : u ∈ X86Int.bytes8 imm := by
                    have heq := Option.some.inj hu; subst heq; exact List.getElem_mem _
                  exact hb_i u this
                · exact hsim.memBound x u hu
              stdin := by simpaX [X86Min.St.setReg, hinp'] using hsim.stdin
              inpBound := by simpa [X86Min.St.setReg, hinp'] using hsim.inpBound
              out := by simpaX [X86Min.St.setReg, hout'] using hsim.out
              oracle := hsim.oracle }
          · intro code out h; cases h
    | pop d =>
      have hd16 : d < 16 := hwf
      simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, _⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        simp only [hi4, bind, Option.bind, pure]
        cases hle_m : X86Min.le ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).rdByte sp 8 with
        | none =>
          have hle_m' : X86Min.le m.rdByte sp 8 = none := by simpaX [X86Min.St.rdByte] using hle_m
          simp [hle_m']
        | some v =>
          have hle_m' : X86Min.le m.rdByte sp 8 = some v := by simpaX [X86Min.St.rdByte] using hle_m
          have hmb : ∀ x u, m.rdByte x = some u → u < 256 := fun x u hu => hsim.memBound x u (hsim.memLe x u hu)
          obtain ⟨bs, hrdN, hleVal, hv_lt⟩ := le_to_rdN_leVal hsim.memLe hmb 8 sp v hle_m'
          have hvW : v < W := by simpa [W] using hv_lt
          have hrd64 : X86Int.rd64 i.mem sp = some v := by
            simp [X86Int.rd64, hrdN, hleVal]
          simp only [hle_m', hrd64]
          let sp' := (sp + 8) % W
          have hsp'_lt : sp' < W := Nat.mod_lt _ (by decide)
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            have hsim1 : Sim
                { (m.setReg 4 (some sp')) with rip := (m.rip + len) % X86Min.pow2 64 }
                { (i.set 4 (some sp')) with rip := i.rip + len } := by
              rw [hnx_eq]
              simpa [X86Min.St.setReg, X86Int.St.set, hrip_int] using
                sim_setReg hsim (by decide : 4 < 16) hsp'_lt (m.rip + len) hcanon hsim.fl
            rw [hnx_eq]
            simpaX [sp', X86Min.St.setReg, X86Int.St.set, hrip_int] using
              sim_setReg hsim1 hd16 hvW (m.rip + len) hcanon hsim1.fl
          · intro code out h; cases h
    | ldRsp8 d =>
      have hd16 : d < 16 := hwf
      simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
        have hea : X86Int.ea sp 0 = sp := by unfold X86Int.ea; simpa using Nat.mod_eq_of_lt hspW
        simp only [hi4, hea, bind, Option.bind, pure]
        cases hrd : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).rdByte sp with
        | none =>
          have hrd_m : m.rdByte sp = none := by simpa [X86Min.St.rdByte] using hrd
          simp [hrd_m]
        | some v =>
          have hrd_m : m.rdByte sp = some v := by simpa [X86Min.St.rdByte] using hrd
          have hird := hsim.memLe sp v hrd_m
          have hv256 := hsim.memBound sp v hird
          have hvW : v < W := Nat.lt_trans hv256 (by decide)
          simp only [X86Min.St.rdByte] at hrd_m
          simp only [X86Min.St.rdByte, hrd_m, hird, bind, Option.bind, pure, X86Min.St.wr,
            show (32 == 8) = false by decide, Bool.false_eq_true, if_false]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            simpa [X86Min.St.setReg, X86Int.St.set, hnx_eq, hrip_int] using
              sim_setReg hsim hd16 hvW (m.rip + len) hcanon hsim.fl
          · intro code out h; cases h
    | stRsp8 src =>
      have hs16 : src < 16 := hwf
      simp only [toUniIns, exec, X86Int.exec, X86Int.lift, X86Min.okOf]
      cases hm4 : m.reg 4 with
      | none => simp
      | some sp =>
        cases hms : m.reg src with
        | none => simp [X86Min.St.rd, hms]
        | some vs =>
          obtain ⟨hi4, hspW⟩ := reg_lt16_bound hsim (by decide : 4 < 16) hm4
          obtain ⟨his, _⟩ := reg_lt16_bound hsim hs16 hms
          have hea : X86Int.ea sp 0 = sp := by unfold X86Int.ea; simpa using Nat.mod_eq_of_lt hspW
          simp only [hi4, hms, his, X86Min.St.rd, pow2_8_eq, Option.map_some, hea,
            bind, Option.bind, pure]
          by_cases hwin : (m.lo ≤ sp && sp < m.lo + X86Min.winSize) = true
          · have hwin' : m.lo ≤ sp ∧ sp < m.lo + X86Min.winSize := by
              simpa [Bool.and_eq_true, decide_eq_true_eq] using hwin
            obtain ⟨mem', hwr_i, hwritable', hrd_i⟩ :=
              mem_wr_of_writable (v := vs % 256) (hsim.stackWritable sp hwin'.1 hwin'.2)
            have hwr_i' : i.mem.wr sp (vs % 256) = some mem' := hwr_i
            have hwr_m : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).wrByte sp (vs % 256) =
                some { m with rip := (m.rip + len) % X86Min.pow2 64, mem := m.mem.set (sp - m.lo) (some (vs % 256)) } := by
              simp [X86Min.St.wrByte, hwin]
            rw [hwr_m, hwr_i']
            dsimp only
            constructor
            · intro m' hm'; cases hm'
              refine ⟨_, rfl, ?_⟩
              exact {
                rip := by simp [hnx_eq, hrip_int]
                ripBound := by simp [hnx_eq, hcanon]
                mLen := by simp [hsim.mLen]
                iLen := hsim.iLen
                regsLe := hsim.regsLe
                regsDef := hsim.regsDef
                regsBound := hsim.regsBound
                fl := hsim.fl
                memLen := by simp [hsim.memLen]
                loCanon := hsim.loCanon
                stackWritable := fun x hx1 hx2 => by
                  rw [hwritable']; exact hsim.stackWritable x hx1 hx2
                memLe := fun x u hu => by
                  have hrd_m1 : ({ m with mem := m.mem.set (sp - m.lo) (some (vs % 256)) } : X86Min.St).rdByte x =
                      if x = sp then some (vs % 256) else m.rdByte x := by
                    obtain ⟨_, _, heq, _, hrd1⟩ := min_wrByte_spec hsim.memLen hwin' (v := vs % 256)
                    simpa [heq] using hrd1 x
                  have hu' : ({ m with mem := m.mem.set (sp - m.lo) (some (vs % 256)) } : X86Min.St).rdByte x = some u := by
                    simpa [X86Min.St.rdByte] using hu
                  rw [hrd_m1] at hu'
                  rw [hrd_i]
                  by_cases hx : x = sp
                  · subst hx; simp at hu' ⊢; subst hu'; simp
                  · simp only [hx, if_false] at hu' ⊢; exact hsim.memLe x u hu'
                memBound := fun x u hu => by
                  rw [hrd_i] at hu
                  split at hu
                  · cases hu; exact Nat.mod_lt _ (by decide)
                  · exact hsim.memBound x u hu
                stdin := hsim.stdin
                inpBound := hsim.inpBound
                out := hsim.out
                oracle := hsim.oracle }
            · intro code out h; cases h
          · have hwin' : ¬(m.lo ≤ sp ∧ sp < m.lo + X86Min.winSize) := by
              simpa [Bool.and_eq_true, decide_eq_true_eq] using hwin
            simp [X86Min.St.wrByte, hwin']
    | jmpReg r => exact (hcompat.noJmpReg m.rip r len hdec).elim
    | jmp rel =>
      have hrelW : rel < W := hwf
      simp only [toUniIns, exec, X86Int.exec, mod_add_mod_W, pow2_64_eq_W]
      constructor
      · intro m' hm'; cases hm'
        refine ⟨_, rfl, ?_⟩
        exact sim_set_rip hsim ((m.rip + len + rel) % W) (Nat.mod_lt _ (by decide))
      · intro code out h; cases h
    | jcc cc rel =>
      have hrelW : rel < W := hwf
      simp only [toUniIns, exec, X86Int.exec, X86Int.lift]
      cases hcond : X86Min.cond m.fl cc with
      | none => simp
      | some b =>
        have hcond_i := cond_rel hsim.fl cc hcond
        rw [hcond_i]
        dsimp only [bind, Option.bind, pure]
        cases b
        · simp only [Bool.false_eq_true, if_false]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            simpa [hnx_eq, hrip_int] using sim_set_rip hsim (m.rip + len) hcanon
          · intro code out h; cases h
        · simp only [if_true, mod_add_mod_W, pow2_64_eq_W]
          constructor
          · intro m' hm'; cases hm'
            refine ⟨_, rfl, ?_⟩
            exact sim_set_rip hsim ((m.rip + len + rel) % W) (Nat.mod_lt _ (by decide))
          · intro code out h; cases h
    | syscall =>
      simp only [toUniIns, exec, X86Int.exec]
      obtain ⟨va1, hi1⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 1 (by decide) (by decide))
      obtain ⟨va2, hi2⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 2 (by decide) (by decide))
      obtain ⟨va6, hi6⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 6 (by decide) (by decide))
      obtain ⟨va7, hi7⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 7 (by decide) (by decide))
      obtain ⟨va8, hi8⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 8 (by decide) (by decide))
      obtain ⟨va9, hi9⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 9 (by decide) (by decide))
      obtain ⟨va10, hi10⟩ := Option.isSome_iff_exists.1 (hsim.regsDef 10 (by decide) (by decide))
      cases hm0 : m.reg 0 with
      | none => simp
      | some nr =>
        obtain ⟨hi0, _⟩ := reg_lt16_bound hsim (by decide : 0 < 16) hm0
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
                  simp only [show (some 0 != some 0 || some 1 != some 1) = false by decide,
                    Bool.false_eq_true, if_false, bind, Option.bind, pure]
                  cases hinp : m.inp with
                  | nil =>
                    have hstin : i.k.stdin = [] := by rw [hsim.stdin, hinp]
                    have hcnt : Kern.count i.k 1 0 = (0, i.k) := by
                      simp [Kern.count, Kern.next, hsim.oracle]
                    have hsys : X86Env.syscall i.k i.mem 0 0 ad 1 va10 va8 va9 =
                        some (.ret 0 { i.k with stdin := [] } i.mem) := by
                      simp [X86Env.syscall, hstin, hcnt, Mem.wrN]
                    rw [hsys]
                    dsimp only
                    constructor
                    · intro m' hm'; cases hm'
                      refine ⟨_, rfl, ?_⟩
                      exact {
                        rip := by simp [X86Min.St.setReg, hnx_eq, hrip_int]
                        ripBound := by simp [X86Min.St.setReg, hnx_eq, hcanon]
                        mLen := by simp [X86Min.St.setReg, hsim.mLen]
                        iLen := by simp [X86Int.St.set, hsim.iLen]
                        regsLe := fun r hr => by
                          change X86Min.OLe _ ((((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).reg r)
                          have hlen0 : ({ regs := m.regs, rip := (m.rip + len) % X86Min.pow2 64, fl := m.fl, mem := m.mem, lo := m.lo, inp := [], out := m.out } : X86Min.St).regs.length = 16 := hsim.mLen
                          have hlen1 : (({ regs := m.regs, rip := (m.rip + len) % X86Min.pow2 64, fl := m.fl, mem := m.mem, lo := m.lo, inp := [], out := m.out } : X86Min.St).setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).regs.length = 16 := by
                            simp [X86Min.St.setReg, hsim.mLen]
                          have hlen2 : ((({ regs := m.regs, rip := (m.rip + len) % X86Min.pow2 64, fl := m.fl, mem := m.mem, lo := m.lo, inp := [], out := m.out } : X86Min.St).setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).regs.length = 16 := by
                            simp [X86Min.St.setReg, hsim.mLen]
                          have hilen1 : (i.set 0 (some 0)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                          have hilen2 : ((i.set 0 (some 0)).set 1 (some (i.rip + len))).regs.length = 16 := by
                            simp [X86Int.St.set, hsim.iLen]
                          rw [m_reg_setReg hlen2 (by decide : 0 < 16),
                              m_reg_setReg hlen1 (by decide : 11 < 16),
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
                              · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hcanon
                              · by_cases hr0 : r = 0
                                · subst hr0; simp at hu; subst hu; decide
                                · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                          · have hlen : (((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            have hnone : (((i.set 0 (some 0)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                              List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                            simp [X86Int.St.reg, hnone] at hu
                        fl := by simpaX [X86Min.St.setReg] using hsim.fl
                        memLen := by simpa [X86Min.St.setReg] using hsim.memLen
                        loCanon := by simpa [X86Min.St.setReg] using hsim.loCanon
                        stackWritable := fun x hx1 hx2 =>
                          hsim.stackWritable x (by simpa [X86Min.St.setReg] using hx1) (by simpa [X86Min.St.setReg] using hx2)
                        memLe := fun x => by simpaX [X86Min.St.setReg, X86Min.St.rdByte] using hsim.memLe x
                        memBound := hsim.memBound
                        stdin := by simp [X86Min.St.setReg]
                        inpBound := fun c hc => by simp [X86Min.St.setReg] at hc
                        out := by simpa [X86Min.St.setReg] using hsim.out
                        oracle := hsim.oracle }
                    · intro code out h; cases h
                  | cons c rest =>
                    have hc256 : c < 256 := hsim.inpBound c (by simp [hinp])
                    have hc_mod : c % 256 = c := Nat.mod_eq_of_lt hc256
                    have hstin : i.k.stdin = c :: rest := by rw [hsim.stdin, hinp]
                    have hcnt : Kern.count i.k 1 (rest.length + 1) = (1, i.k) := by
                      simp [Kern.count, Kern.next, hsim.oracle]
                    by_cases hwin : (m.lo ≤ ad && ad < m.lo + X86Min.winSize) = true
                    · have hwin' : m.lo ≤ ad ∧ ad < m.lo + X86Min.winSize := by
                        simpa [Bool.and_eq_true, decide_eq_true_eq] using hwin
                      obtain ⟨mem', hwr_i, hwritable', hrd'⟩ :=
                        mem_wr_of_writable (v := c) (hsim.stackWritable ad hwin'.1 hwin'.2)
                      have hwrN : i.mem.wrN ad [c] = some mem' := by
                        simpa [Mem.wrN] using hwr_i
                      have hsys : X86Env.syscall i.k i.mem 0 0 ad 1 va10 va8 va9 =
                          some (.ret 1 { i.k with stdin := rest } mem') := by
                        simp [X86Env.syscall, hstin, hcnt, hwrN]
                      rw [hsys]
                      dsimp only
                      let s2 : X86Min.St :=
                        (({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1
                          (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none)
                      have hs2_len : s2.mem.length = X86Min.winSize := hsim.memLen
                      have hs2_win : s2.lo ≤ ad ∧ ad < s2.lo + X86Min.winSize := hwin'
                      obtain ⟨t, hwr_m, heq_t, htlen, htrd⟩ := min_wrByte_spec hs2_len hs2_win (v := c)
                      have hwr_m' : s2.wrByte ad c = some t := hwr_m
                      have hwr_m'' := hwr_m'
                      dsimp only [s2] at hwr_m''
                      rw [hinp] at hwr_m''
                      rw [hwr_m'']
                      dsimp only
                      constructor
                      · intro m' hm'; cases hm'
                        refine ⟨_, rfl, ?_⟩
                        exact {
                          rip := by simp [X86Min.St.setReg, heq_t, s2, hnx_eq, hrip_int]
                          ripBound := by simp [X86Min.St.setReg, heq_t, s2, hnx_eq, hcanon]
                          mLen := by simp [X86Min.St.setReg, heq_t, s2, hsim.mLen]
                          iLen := by simp [X86Int.St.set, hsim.iLen]
                          regsLe := fun r hr => by
                            change X86Min.OLe _ ((((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).reg r)
                            have hlen0 : ({ m with rip := (m.rip + len) % X86Min.pow2 64 } : X86Min.St).regs.length = 16 := hsim.mLen
                            have hlen1 : ({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1 (some ((m.rip + len) % X86Min.pow2 64))).regs.length = 16 := by
                              simp [X86Min.St.setReg, hsim.mLen]
                            have hlen2 : s2.regs.length = 16 := by
                              simp [s2, X86Min.St.setReg, hsim.mLen]
                            have hlen3 : ({ t with inp := rest } : X86Min.St).regs.length = 16 := by
                              simp [heq_t, hlen2]
                            have hilen1 : (i.set 0 (some 1)).regs.length = 16 := by simp [X86Int.St.set, hsim.iLen]
                            have hilen2 : ((i.set 0 (some 1)).set 1 (some (i.rip + len))).regs.length = 16 := by
                              simp [X86Int.St.set, hsim.iLen]
                            rw [m_reg_setReg hlen3 (by decide : 0 < 16)]
                            have htmp : ({ t with inp := rest } : X86Min.St).reg r = s2.reg r := by
                              simp [X86Min.St.reg, heq_t]
                            rw [htmp]
                            rw [m_reg_setReg hlen1 (by decide : 11 < 16),
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
                                · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hcanon
                                · by_cases hr0 : r = 0
                                  · subst hr0; simp at hu; subst hu; decide
                                  · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                            · have hlen : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              have hnone : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                                List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                              simp [X86Int.St.reg, hnone] at hu
                          fl := by simpaX [X86Min.St.setReg, heq_t, s2] using hsim.fl
                          memLen := by simpa [X86Min.St.setReg] using htlen
                          loCanon := by simpa [X86Min.St.setReg, heq_t, s2] using hsim.loCanon
                          stackWritable := fun x hx1 hx2 => by
                            rw [hwritable']
                            exact hsim.stackWritable x (by simpa [X86Min.St.setReg, heq_t, s2] using hx1)
                              (by simpa [X86Min.St.setReg, heq_t, s2] using hx2)
                          memLe := fun x u hu => by
                            have hu' : t.rdByte x = some u := by
                              simpaX [X86Min.St.setReg, X86Min.St.rdByte] using hu
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
                    · have hwin' : ¬(m.lo ≤ ad ∧ ad < m.lo + X86Min.winSize) := by
                        simpa [Bool.and_eq_true, decide_eq_true_eq] using hwin
                      have hnone : (({ m with rip := (m.rip + len) % X86Min.pow2 64 }.setReg 1
                          (some ((m.rip + len) % X86Min.pow2 64))).setReg 11 none).wrByte ad c = none := by
                        simp [X86Min.St.wrByte, X86Min.St.setReg]; intro h1; exact Decidable.byContradiction fun h2 => hwin' ⟨of_decide_eq_true h1, Decidable.byContradiction fun h3 => h2 (decide_eq_false h3)⟩
                      rw [hinp] at hnone
                      dsimp only
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
                      have hrdN : Mem.rdN i.mem ad 1 = some [c] := by simp [Mem.rdN, hird]
                      have hsys : X86Env.syscall i.k i.mem 1 1 ad 1 va10 va8 va9 =
                          some (.ret 1 { i.k with out := i.k.out ++ [c] } i.mem) := by
                        simp [X86Env.syscall, hcnt, hrdN]
                      rw [hsys]
                      dsimp only
                      constructor
                      · intro m' hm'; cases hm'
                        refine ⟨_, rfl, ?_⟩
                        exact {
                          rip := by simp [X86Min.St.setReg, hnx_eq, hrip_int]
                          ripBound := by simp [X86Min.St.setReg, hnx_eq, hcanon]
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
                                · subst hr1; simp at hu; subst hu; rw [hrip_int]; exact hcanon
                                · by_cases hr0 : r = 0
                                  · subst hr0; simp at hu; subst hu; decide
                                  · simp [hr11, hr1, hr0] at hu; exact hsim.regsBound r u hu
                            · have hlen : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs.length = 16 := by
                                simp [X86Int.St.set, hsim.iLen]
                              have hnone : (((i.set 0 (some 1)).set 1 (some (i.rip + len))).set 11 none).regs[r]? = none :=
                                List.getElem?_eq_none (by rw [hlen]; exact Nat.le_of_not_lt hr16)
                              simp [X86Int.St.reg, hnone] at hu
                          fl := by simpaX [X86Min.St.setReg] using hsim.fl
                          memLen := by simpa [X86Min.St.setReg] using hsim.memLen
                          loCanon := by simpa [X86Min.St.setReg] using hsim.loCanon
                          stackWritable := fun x hx1 hx2 =>
                            hsim.stackWritable x (by simpa [X86Min.St.setReg] using hx1) (by simpa [X86Min.St.setReg] using hx2)
                          memLe := fun x => by simpaX [X86Min.St.setReg, X86Min.St.rdByte] using hsim.memLe x
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

theorem run_exit (im : Nat → Option (Ins × Nat)) (n c : Nat) :
    run im n (.exit c) = .exit c := by
  cases n <;> rfl

theorem run_stuck (im : Nat → Option (Ins × Nat)) (n : Nat) :
    run im n .stuck = .stuck := by
  cases n <;> rfl

theorem run_succ_ok (im : Nat → Option (Ins × Nat)) (n : Nat) (s : St) :
    run im (n + 1) (.ok s) = run im n (step im s) := rfl

theorem run_fuel (im : Nat → Option (Ins × Nat)) :
    ∀ (F m : Nat) (o : Out) {c : Nat}, run im F o = .exit c → run im (F + m) o = .exit c := by
  intro F
  induction F with
  | zero => intro m o c h; cases o <;> simp_all [run, run_exit]
  | succ F ih =>
    intro m o c h
    cases o with
    | ok s =>
      rw [Nat.add_right_comm, run_succ_ok]
      rw [run_succ_ok] at h
      exact ih m (step im s) h
    | exit c' => simp [run] at h; subst h; simp [run_exit]
    | stuck => simp [run] at h

theorem run_det {im : Nat → Option (Ins × Nat)} {o : Out} {F G a b : Nat}
    (h1 : run im F o = .exit a) (h2 : run im G o = .exit b) : a = b := by
  have e1 := run_fuel im F G o h1
  have e2 := run_fuel im G F o h2
  rw [Nat.add_comm, e2] at e1
  cases e1; rfl

/-- Uniqueness of the exit step along a `run` trace. -/
theorem run_exit_step_unique {im : Nat → Option (Ins × Nat)} :
    ∀ {k1 k2 : Nat} {s s1 s2 : St} {c1 c2 : Nat},
      run im k1 (.ok s) = .ok s1 → step im s1 = .exit c1 →
      run im k2 (.ok s) = .ok s2 → step im s2 = .exit c2 →
      k1 = k2 ∧ s1 = s2 ∧ c1 = c2 := by
  intro k1
  induction k1 with
  | zero =>
    intro k2 s s1 s2 c1 c2 h1 hs1 h2 hs2
    simp only [run, X86Int.Out.ok.injEq] at h1; subst h1
    cases k2 with
    | zero =>
      simp only [run, X86Int.Out.ok.injEq] at h2; subst h2
      rw [hs1] at hs2; cases hs2; exact ⟨rfl, rfl, rfl⟩
    | succ k2 =>
      rw [run_succ_ok, hs1] at h2
      rw [run_exit] at h2; cases h2
  | succ k1 ih =>
    intro k2 s s1 s2 c1 c2 h1 hs1 h2 hs2
    cases k2 with
    | zero =>
      simp only [run, X86Int.Out.ok.injEq] at h2; subst h2
      rw [run_succ_ok, hs2, run_exit] at h1; cases h1
    | succ k2 =>
      rw [run_succ_ok] at h1 h2
      cases hst : step im s with
      | ok s' =>
        rw [hst] at h1 h2
        obtain ⟨rfl, rfl, rfl⟩ := ih h1 hs1 h2 hs2
        exact ⟨rfl, rfl, rfl⟩
      | exit c' => rw [hst, run_exit] at h1; cases h1
      | stuck => rw [hst, run_stuck] at h1; cases h1

theorem runS_min_sim {imMin : X86Min.Img} {imUni : Nat → Option (Ins × Nat)}
    (hcompat : MinCompat imMin imUni) :
    ∀ (fuel : Nat) {m : X86Min.St} {i : St}, Sim m i →
      (∀ m', X86Min.runS imMin fuel m = .inl m' →
        ∃ i', run imUni fuel (.ok i) = .ok i' ∧ Sim m' i') ∧
      (∀ code out, X86Min.run imMin fuel m = .exit code out →
        run imUni fuel (.ok i) = .exit code ∧
        ∃ k < fuel, ∃ mf if_,
          X86Min.runS imMin k m = .inl mf ∧
          run imUni k (.ok i) = .ok if_ ∧
          Sim mf if_ ∧
          X86Min.step imMin mf = .inr (.exit code out) ∧
          step imUni if_ = .exit code ∧
          if_.k.out = out) := by
  intro fuel
  induction fuel with
  | zero =>
    intro m i hsim
    constructor
    · intro m' hm'
      change Sum.inl m = Sum.inl m' at hm'; cases hm'
      exact ⟨i, rfl, hsim⟩
    · intro code out h
      change X86Min.Outcome.outOfFuel = X86Min.Outcome.exit code out at h
      cases h
  | succ fuel ih =>
    intro m i hsim
    have hs := step_min_sim hcompat hsim
    cases hst : X86Min.step imMin m with
    | inl m1 =>
      obtain ⟨i1, hi1, hsim1⟩ := hs.1 m1 hst
      have ih1 := ih hsim1
      constructor
      · intro m' hm'
        rw [X86Min.runS_succ, hst] at hm'
        simpa [run, hi1] using ih1.1 m' hm'
      · intro code out h
        rw [X86Min.run_succ, hst] at h
        obtain ⟨hrun_u, k, hk, mf, if_, hrunS_mf, hrun_if, hsim_f, hstep_mf, hstep_if, hout_if⟩ :=
          ih1.2 code out h
        refine ⟨?_, k + 1, Nat.succ_lt_succ hk, mf, if_, ?_, ?_, hsim_f, hstep_mf, hstep_if, hout_if⟩
        · simp [run, hi1, hrun_u]
        · rw [X86Min.runS_succ, hst]; exact hrunS_mf
        · simp [run, hi1, hrun_if]
    | inr o =>
      constructor
      · intro m' hm'
        rw [X86Min.runS_succ, hst] at hm'; cases hm'
      · intro code out h
        rw [X86Min.run_succ, hst] at h; cases h
        obtain ⟨huni, hout⟩ := hs.2 code out hst
        refine ⟨?_, 0, Nat.zero_lt_succ fuel, m, i, rfl, rfl, hsim, hst, huni, hout⟩
        simp [run, huni, run_exit]

theorem run_min_sim {imMin : X86Min.Img} {imUni : Nat → Option (Ins × Nat)}
    (hcompat : MinCompat imMin imUni)
    {fuel : Nat} {m : X86Min.St} {i : St} (hsim : Sim m i)
    {code : Nat} {out : List Nat}
    (hrun : X86Min.run imMin fuel m = .exit code out) :
    run imUni fuel (.ok i) = .exit code ∧
    ∃ k < fuel, ∃ mf if_,
      X86Min.runS imMin k m = .inl mf ∧
      run imUni k (.ok i) = .ok if_ ∧
      Sim mf if_ ∧
      X86Min.step imMin mf = .inr (.exit code out) ∧
      step imUni if_ = .exit code ∧
      if_.k.out = out :=
  (runS_min_sim hcompat fuel hsim).2 code out hrun

theorem prun_min_sim {imMin : X86Min.Img} {imUni : Nat → Option (Ins × Nat)}
    (hcompat : MinCompat imMin imUni)
    (alts : Alts) (r : Rates) (hr : r.ok)
    {fuel : Nat} {m : X86Min.St} {i : St} (hsim : Sim m i)
    {code : Nat} {out : List Nat}
    (hrun : X86Min.run imMin fuel m = .exit code out) :
    (prun alts (r.state hr) r.fetch imUni fuel (.ok i)).ok ∧
    fm (prun alts (r.state hr) r.fetch imUni fuel (.ok i)) ≤ fuel * r.total ∧
    faultFree (prun alts (r.state hr) r.fetch imUni fuel (.ok i)) = [X86Int.Out.exit code] ∧
    ∃ k < fuel, ∃ if_,
      run imUni k (.ok i) = .ok if_ ∧
      faultFree (prun alts (r.state hr) r.fetch imUni k (.ok i)) = [X86Int.Out.ok if_] ∧
      if_.k.out = out ∧
      step imUni if_ = .exit code := by
  obtain ⟨hrun_u, k, hk, _, if_, _, hrun_if, _, _, hstep_if, hout_if⟩ := run_min_sim hcompat hsim hrun
  refine ⟨prun_ok alts (r.state hr) hr.1 imUni fuel (.ok i), ?_, ?_, k, hk, if_, hrun_if, ?_, hout_if, hstep_if⟩
  · simpa [X86Int.Rates.total, X86Int.Rates.state, X86Int.Fault.seq, X86Int.Fault.uniform] using
      fm_prun alts (r.state hr) hr.1 imUni fuel (.ok i)
  · rw [faultFree_prun, hrun_u]
  · rw [faultFree_prun, hrun_if]

/-! ### `X86Min.load` and `X86Min.exec` lifting -/

def seedSp : Nat := 0x7ffffffde000
def stackBase : Nat := seedSp - X86Min.winSize / 2
def stackSize : Nat := X86Min.winSize

/-- Concrete initial `X86Int.St` matching `X86Min.load f seedSp inp`. -/
def initSt (rip : Nat) (inp : List Nat) : St :=
  { regs := (List.replicate 16 (some 0)).set 4 (some seedSp),
    rip := rip,
    fl := X86Int.Flags.undef,
    mem := [⟨stackBase, List.replicate stackSize 0, true⟩],
    k := ⟨[], [], inp, [], []⟩ }

private theorem memWritable_initStack (x : Nat)
    (hx1 : seedSp - X86Min.winSize / 2 ≤ x)
    (hx2 : x < seedSp - X86Min.winSize / 2 + X86Min.winSize) :
    MemWritable [⟨stackBase, List.replicate stackSize 0, true⟩] x := by
  have hbounds : stackBase ≤ x ∧ x < stackBase + stackSize := by
    simp only [stackBase, stackSize, seedSp, X86Min.winSize] at hx1 hx2 ⊢; omega
  exact ⟨0, ⟨stackBase, List.replicate stackSize 0, true⟩, by
    simp [Mem.find, Region.hi, hbounds], rfl⟩

theorem load_facts {f : List Nat} {inp : List Nat} {imMin : X86Min.Img} {m0 : X86Min.St}
    (hload : X86Min.load f seedSp inp = some (imMin, m0)) :
    imMin = X86Min.Img.mk' f imMin.vbase f.length ∧
    f.length ≤ 2 ^ X86Min.tdepth ∧
    m0 = { regs := (List.replicate 16 none).set 4 (some seedSp),
           rip := m0.rip, fl := .undef,
           mem := List.replicate X86Min.winSize none,
           lo := seedSp - X86Min.winSize / 2, inp := inp, out := [] } ∧
    imMin.vbase ≤ m0.rip ∧ m0.rip < imMin.vbase + f.length := by
  unfold X86Min.load at hload
  simp only [bind, Option.bind, pure] at hload
  cases h0 : X86Min.field f 0 4 with | none => rw [h0] at hload; cases hload | some magic =>
  rw [h0] at hload
  cases h1 : X86Min.field f 4 3 with | none => rw [h1] at hload; cases hload | some ident =>
  rw [h1] at hload
  cases h2 : X86Min.field f 16 2 with | none => rw [h2] at hload; cases hload | some ty =>
  rw [h2] at hload
  cases h3 : X86Min.field f 18 2 with | none => rw [h3] at hload; cases hload | some mach =>
  rw [h3] at hload
  cases h4 : X86Min.field f 24 8 with | none => rw [h4] at hload; cases hload | some entry =>
  rw [h4] at hload
  cases h5 : X86Min.field f 32 8 with | none => rw [h5] at hload; cases hload | some phoff =>
  rw [h5] at hload
  cases h6 : X86Min.field f 54 2 with | none => rw [h6] at hload; cases hload | some phentsize =>
  rw [h6] at hload
  cases h7 : X86Min.field f 56 2 with | none => rw [h7] at hload; cases hload | some phnum =>
  rw [h7] at hload
  dsimp only at hload
  cases h8 : X86Min.field f phoff 4 with | none => rw [h8] at hload; cases hload | some ptype =>
  rw [h8] at hload
  cases h9 : X86Min.field f (phoff + 4) 4 with | none => rw [h9] at hload; cases hload | some pflags =>
  rw [h9] at hload
  cases h10 : X86Min.field f (phoff + 8) 8 with | none => rw [h10] at hload; cases hload | some poff =>
  rw [h10] at hload
  cases h11 : X86Min.field f (phoff + 16) 8 with | none => rw [h11] at hload; cases hload | some vaddr =>
  rw [h11] at hload
  cases h12 : X86Min.field f (phoff + 32) 8 with | none => rw [h12] at hload; cases hload | some filesz =>
  rw [h12] at hload
  cases h13 : X86Min.field f (phoff + 40) 8 with | none => rw [h13] at hload; cases hload | some memsz =>
  rw [h13] at hload
  dsimp only at hload
  split at hload
  · cases hload
  · rename_i hbad
    simp only [Option.some.injEq, Prod.mk.injEq] at hload
    obtain ⟨rfl, rfl⟩ := hload
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq, not_or,
      Decidable.not_not, Nat.not_lt, Nat.not_le] at hbad
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, _⟩, _⟩, _⟩, _⟩, _⟩, _⟩, _⟩, _⟩, hfilesz⟩, _⟩, _⟩, hentry_lo⟩, hentry_hi⟩, hlen⟩ := hbad
    rw [hfilesz] at hentry_hi hlen ⊢
    exact ⟨rfl, hlen, rfl, hentry_lo, hentry_hi⟩

theorem sim_of_load {f inp : List Nat} {imMin : X86Min.Img} {m0 : X86Min.St}
    (hload : X86Min.load f seedSp inp = some (imMin, m0))
    (hcanon : imMin.vbase + imMin.len + 16 < W)
    (hinp : ∀ c ∈ inp, c < 256) :
    Sim m0 (initSt m0.rip inp) := by
  obtain ⟨him, hlen, hm0, _, hentry_hi⟩ := load_facts hload
  have hripW : m0.rip < W := by
    have hcanon' : imMin.vbase + imMin.len + 16 < W := hcanon
    rw [him] at hcanon'
    dsimp only [X86Min.Img.mk'] at hcanon'
    omega
  rw [hm0]
  refine {
    rip := rfl
    ripBound := hripW
    mLen := by simp
    iLen := by simp [initSt]
    regsLe := fun r hr v hv => by
      simp only [X86Min.St.reg, X86Min.getD_set_reg] at hv
      by_cases h4 : 4 = r
      · subst h4; simp at hv; subst hv; simp [initSt, X86Int.St.reg]
      · simp only [h4, false_and, ↓reduceIte, List.getElem?_replicate] at hv
        split at hv <;> cases hv
    regsDef := fun r hr _ => by
      simp only [initSt, X86Int.St.reg]
      by_cases h4 : 4 = r
      · subst h4; rw [List.getElem?_set_self (by simp)]; rfl
      · rw [List.getElem?_set_ne (by omega)]
        rw [List.getElem?_replicate, if_pos hr]
        rfl
    regsBound := fun r v hv => by
      simp only [initSt, X86Int.St.reg] at hv
      by_cases h4 : 4 = r
      · subst h4
        rw [List.getElem?_set_self (by simp)] at hv
        cases hv; decide
      · rw [List.getElem?_set_ne (by omega)] at hv
        simp only [List.getElem?_replicate] at hv
        split at hv <;> cases hv <;> decide
    fl := ⟨X86Min.OLe.none _, X86Min.OLe.none _, X86Min.OLe.none _, X86Min.OLe.none _⟩
    memLen := by simp
    loCanon := by dsimp only; decide
    stackWritable := fun x hx1 hx2 => memWritable_initStack x hx1 hx2
    memLe := fun x u hu => by
      simp only [X86Min.St.rdByte, List.getElem?_replicate] at hu
      split at hu
      · split at hu <;> cases hu
      · cases hu
    memBound := fun x v hv => by
      unfold initSt at hv
      simp only [Mem.rd, bind, Option.bind] at hv
      cases hf : Mem.find [⟨stackBase, List.replicate stackSize 0, true⟩] x 0 with
      | none => rw [hf] at hv; cases hv
      | some p =>
        rw [hf] at hv; dsimp only at hv
        simp only [Mem.find, Region.hi] at hf
        have hf' := hf; simp at hf'; obtain ⟨_, hp⟩ := hf'; subst hp
        · skip
          simp only [List.getElem?_replicate] at hv
          split at hv
          · cases hv; decide
          · cases hv
    stdin := rfl
    inpBound := hinp
    out := rfl
    oracle := rfl }

/-- Unrestricted forward simulation from `X86Min.exec` to `X86Uni.run`:
for any valid ELF64 file `f` in canonical address range without `xor r11, r11`
and any byte input `inp`, every exiting `X86Min.exec` run is matched step-for-step
by `X86Uni.run` with an explicit `k < fuel` final-state witness binding `if_.k.out = out`. -/
theorem exec_min_sim {fuel : Nat} {f inp : List Nat} {imMin : X86Min.Img} {m0 : X86Min.St}
    (hf : ∀ b ∈ f, b < 256)
    (hinp : ∀ c ∈ inp, c < 256)
    (hload : X86Min.load f seedSp inp = some (imMin, m0))
    (hcanon : imMin.vbase + imMin.len + 16 < W)
    (hno11 : ∀ a ins len, imMin.decodeAt a = some (ins, len) → ∀ w, ins ≠ .rr .xor w 11 11)
    (hnoJmp : ∀ a r len, imMin.decodeAt a = some (.jmpReg r, len) → False)
    {code : Nat} {out : List Nat}
    (hexec : X86Min.exec fuel f inp = .exit code out) :
    let imUni := imageOf f imMin.vbase
    let i0 := initSt m0.rip inp
    run imUni fuel (.ok i0) = .exit code ∧
    ∃ k < fuel, ∃ mf if_,
      X86Min.runS imMin k m0 = .inl mf ∧
      run imUni k (.ok i0) = .ok if_ ∧
      Sim mf if_ ∧
      X86Min.step imMin mf = .inr (.exit code out) ∧
      step imUni if_ = .exit code ∧
      if_.k.out = out := by
  obtain ⟨him, hlen, _, _, _⟩ := load_facts hload
  have hok : ImageOk f imMin.vbase := {
    bytesLt256 := hf
    lenBound := hlen
    vbaseCanon := by rw [him] at hcanon; exact hcanon
    noXor11 := by rw [← him]; exact hno11
    noJmpReg := by rw [← him]; exact hnoJmp }
  have hcompat : MinCompat imMin (imageOf f imMin.vbase) := by
    rw [him]; exact minCompat_of_imageOk hok
  have hsim := sim_of_load hload hcanon hinp
  rw [X86Min.exec_of_load hload] at hexec
  exact run_min_sim hcompat hsim hexec

theorem exec_prun_min_sim (alts : Alts) (r : Rates) (hr : r.ok)
    {fuel : Nat} {f inp : List Nat} {imMin : X86Min.Img} {m0 : X86Min.St}
    (hf : ∀ b ∈ f, b < 256)
    (hinp : ∀ c ∈ inp, c < 256)
    (hload : X86Min.load f seedSp inp = some (imMin, m0))
    (hcanon : imMin.vbase + imMin.len + 16 < W)
    (hno11 : ∀ a ins len, imMin.decodeAt a = some (ins, len) → ∀ w, ins ≠ .rr .xor w 11 11)
    (hnoJmp : ∀ a r len, imMin.decodeAt a = some (.jmpReg r, len) → False)
    {code : Nat} {out : List Nat}
    (hexec : X86Min.exec fuel f inp = .exit code out) :
    let imUni := imageOf f imMin.vbase
    let i0 := initSt m0.rip inp
    (prun alts (r.state hr) r.fetch imUni fuel (.ok i0)).ok ∧
    fm (prun alts (r.state hr) r.fetch imUni fuel (.ok i0)) ≤ fuel * r.total ∧
    faultFree (prun alts (r.state hr) r.fetch imUni fuel (.ok i0)) = [X86Int.Out.exit code] ∧
    ∃ k < fuel, ∃ if_,
      run imUni k (.ok i0) = .ok if_ ∧
      faultFree (prun alts (r.state hr) r.fetch imUni k (.ok i0)) = [X86Int.Out.ok if_] ∧
      if_.k.out = out ∧
      step imUni if_ = .exit code := by
  obtain ⟨him, hlen, _, _, _⟩ := load_facts hload
  have hok : ImageOk f imMin.vbase := {
    bytesLt256 := hf
    lenBound := hlen
    vbaseCanon := by rw [him] at hcanon; exact hcanon
    noXor11 := by rw [← him]; exact hno11
    noJmpReg := by rw [← him]; exact hnoJmp }
  have hcompat : MinCompat imMin (imageOf f imMin.vbase) := by
    rw [him]; exact minCompat_of_imageOk hok
  have hsim := sim_of_load hload hcanon hinp
  rw [X86Min.exec_of_load hload] at hexec
  exact prun_min_sim hcompat alts r hr hsim hexec

/-! ### Kernel-checked runs of the committed 271B `seedA` and 283B `seedB` -/

def seedAImage : Nat → Option (Ins × Nat) := imageOf X86Min.Data.seedA 0x400000
def seedBImage : Nat → Option (Ins × Nat) := imageOf X86Min.Data.seedB 0x400000

theorem seedA_len_eq : X86Min.Data.seedA.length = 271 := by decide +kernel
theorem seedB_len_eq : X86Min.Data.seedB.length = 283 := by decide +kernel

theorem seedA_imageOk : ImageOk X86Min.Data.seedA 0x400000 :=
  imageOk_of_check (by decide +kernel)

theorem seedB_imageOk : ImageOk X86Min.Data.seedB 0x400000 :=
  imageOk_of_check (by decide +kernel)

def outCode : Out → Option Nat
  | .exit c => some c
  | _ => none

theorem eq_exit_of_outCode_eq {o : Out} {c : Nat} (h : outCode o = some c) : o = .exit c := by
  cases o with
  | exit c' => simp [outCode] at h; subst h; rfl
  | ok _ | stuck => simp [outCode] at h

/-- Kernel-checked `X86Uni.run` and `X86Uni.prun` witnesses on the committed 271B `seedA`
and 283B `seedB` (on a valid listing and a refused listing). -/
def seedA_sample_inp : List Nat := [48, 49, 32, 97, 102, 10, 35, 103, 10, 102, 102]
def seedB_sample_inp : List Nat := [48, 48, 49, 9, 51, 55, 55, 59, 55, 10, 49, 52, 55]

def checkRunExit (im : Nat → Option (Ins × Nat)) (k : Nat) (s : St)
    (code : Nat) (out : List Nat) : Bool :=
  match run im k (.ok s) with
  | .ok if_ =>
    match step im if_ with
    | .exit c => c == code && if_.k.out == out
    | _ => false
  | _ => false

theorem checkRunExit_sound {im : Nat → Option (Ins × Nat)} {k : Nat} {s : St}
    {code : Nat} {out : List Nat} (h : checkRunExit im k s code out = true) :
    run im (k + 1) (.ok s) = .exit code ∧
    ∃ if_, run im k (.ok s) = .ok if_ ∧ step im if_ = .exit code ∧ if_.k.out = out := by
  unfold checkRunExit at h
  cases hr : run im k (.ok s) with
  | ok if_ =>
    rw [hr] at h
    dsimp only at h
    cases hs : step im if_ with
    | exit c =>
      rw [hs] at h
      dsimp only at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨rfl, rfl⟩ := h
      have hsucc : run im (k + 1) (.ok s) = .exit c := by
        have : ∀ n o, run im n o = .ok if_ → step im if_ = .exit c → run im (n + 1) o = .exit c := by
          intro n
          induction n with
          | zero => intro o ho hst; cases ho; exact hst
          | succ n ih =>
            intro o ho hst
            cases o with
            | ok st =>
              rw [run_succ_ok] at ho ⊢
              exact ih (step im st) ho hst
            | exit _ => simp [run] at ho
            | stuck => simp [run] at ho
        exact this k (.ok s) hr hs
      exact ⟨hsucc, if_, rfl, hs, rfl⟩
    | ok _ | stuck => rw [hs] at h; cases h
  | exit _ | stuck => rw [hr] at h; cases h

theorem seedA_uni_sample_run :
    outCode (run seedAImage 1000 (.ok (initSt 0x400078 seedA_sample_inp))) = some 0 := by
  decide +kernel

theorem seedA_uni_refuse_run :
    outCode (run seedAImage 1000 (.ok (initSt 0x400078 [48, 49, 103]))) = some 1 := by
  decide +kernel

theorem seedB_uni_sample_run :
    outCode (run seedBImage 1000 (.ok (initSt 0x400078 seedB_sample_inp))) = some 0 := by
  decide +kernel

theorem seedB_uni_refuse_run :
    outCode (run seedBImage 1000 (.ok (initSt 0x400078 [48, 48, 56]))) = some 1 := by
  decide +kernel

theorem seedA_uni_sample_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (hf : p7.mem r.fetch) (hg : p7.mem r.reg) (hm : p7.mem r.mem) (hl : p7.mem r.flag) :
    (prun alts (r.state hr) r.fetch seedAImage 1000 (.ok (initSt 0x400078 seedA_sample_inp))).ok ∧
    fm (prun alts (r.state hr) r.fetch seedAImage 1000 (.ok (initSt 0x400078 seedA_sample_inp))) ≤
      1000 * (4 / 5000000) ∧
    faultFree (prun alts (r.state hr) r.fetch seedAImage 1000 (.ok (initSt 0x400078 seedA_sample_inp))) =
      [.exit 0] := by
  refine ⟨prun_ok _ _ hr.1 _ _ _, ?_, ?_⟩
  · have := baseline alts seedAImage p7 p7 p7 p7 r hr hf hg hm hl 1000 (.ok (initSt 0x400078 seedA_sample_inp))
    exact Rat.le_trans this (by decide +kernel)
  · rw [faultFree_prun, eq_exit_of_outCode_eq seedA_uni_sample_run]

theorem seedB_uni_sample_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (hf : p7.mem r.fetch) (hg : p7.mem r.reg) (hm : p7.mem r.mem) (hl : p7.mem r.flag) :
    (prun alts (r.state hr) r.fetch seedBImage 1000 (.ok (initSt 0x400078 seedB_sample_inp))).ok ∧
    fm (prun alts (r.state hr) r.fetch seedBImage 1000 (.ok (initSt 0x400078 seedB_sample_inp))) ≤
      1000 * (4 / 5000000) ∧
    faultFree (prun alts (r.state hr) r.fetch seedBImage 1000 (.ok (initSt 0x400078 seedB_sample_inp))) =
      [.exit 0] := by
  refine ⟨prun_ok _ _ hr.1 _ _ _, ?_, ?_⟩
  · have := baseline alts seedBImage p7 p7 p7 p7 r hr hf hg hm hl 1000 (.ok (initSt 0x400078 seedB_sample_inp))
    exact Rat.le_trans this (by decide +kernel)
  · rw [faultFree_prun, eq_exit_of_outCode_eq seedB_uni_sample_run]

end X86Uni
