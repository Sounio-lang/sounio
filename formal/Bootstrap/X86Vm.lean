/-
X86Vm: architectural and 4-channel physical x86-64 machine conservatively
extending `X86Int` + `X86Dec` with the arithmetic, shift, memory-width,
SIB-indexed addressing, byte test/compare, conditional move/set, and indirect
jump forms needed by `sporos_vm` (raw x86-64 ELF64 Sporos bytecode VM).

`X86Vm.Ins` embeds every instruction of `X86Int.Ins` via `ofInt`, with proved
equality of architectural (`ofInt_exec`, `ofInt_step`, `ofInt_run`) and
physical (`ofInt_pstep`, `ofInt_prun`) semantics, and adds:

  1. 64-bit multiply/divide & unary ops (`48/49 f7 /2 /3 /4 /6 /7`, `48 99`,
     `48..4d 0f af /r`): `not`, `neg`, `mul`, `imul`, `div`, `idiv`, `cqo`.
     `div` and `idiv` stick cleanly (`.stuck`) on divisor 0 or quotient
     overflow (`#DE` on hardware).
  2. 64-bit variable shifts by `cl` (`48/49 d3 /4 /5 /7`) and immediate
     arithmetic shift right (`48/49 c1 /7 imm8`), masking count `mod 64` and
     leaving flags conservatively `Flags.undef`.
  3. 16-bit load/store and 32-bit store (`[REX] 0f b7 /r`, `66 [REX w=0] 89 /r`,
     `[REX w=0] 89 /r`): `ld16`, `st16`, `st32` (little-endian).
  4. SIB-indexed memory ops and `lea` (`[base + scale * idx + disp]`,
     `scale ∈ {1, 2, 4, 8}`, `idx ≠ 4`): `ldIdx`, `stIdx`, `leaIdx` across
     widths 8/16/32/64.
  5. Test/compare/conditional forms: `testI` (`48/49 f7 /0 imm32`),
     `cmpAlI` (`3c imm8`), `testAlI` (`a8 imm8`), `setcc` (`[REX] 0f 90..9f /0`),
     `cmovcc` (`[REX w=1] 0f 40..4f /r`), plus decoding `jmpR` (`[REX] ff /4`
     mod=3) into `ofInt (.jmpR r)`.

Reuses `X86Int.{Reg, Alu, Flags, St, Out, Fault, Rates}`, `X86Env`, and `PDist`.
Core Lean 4 only (no Mathlib, no `sorry`, no `axiom`, no `native_decide`).
-/
import Bootstrap.X86Int
import Bootstrap.X86Dec
import Bootstrap.X86Env
import Bootstrap.PDist

namespace X86Vm

open X86Env PDist

abbrev Reg := X86Int.Reg
abbrev Alu := X86Int.Alu
abbrev Flags := X86Int.Flags
abbrev St := X86Int.St
abbrev Out := X86Int.Out
abbrev Fault := X86Int.Fault
abbrev Rates := X86Int.Rates

/-! ### Instruction set and `ofInt` embedding -/

/-- Memory access width for SIB-indexed loads and stores. -/
inductive MemW where
  | w8 | w16 | w32 | w64
deriving DecidableEq, Repr

inductive Ins where
  /-- Any instruction of `X86Int.Ins` (all 45+ forms, plus `jmpR` decoded from `ff /4` mod=3). -/
  | ofInt (ins : X86Int.Ins)
  /-- `mul r64` (`48/49 f7 /4`): unsigned 64×64 → 128-bit `rdx:rax := rax * src`. -/
  | mul (src : Reg)
  /-- `imul r64, r64` (`[REX w=1, !x] 0f af /r`, mod=3): `d := (d * s) % W`. -/
  | imul (d s : Reg)
  /-- `div r64` (`48/49 f7 /6`): unsigned `rdx:rax / src`; sticks on `#DE` (`src=0` or `q ≥ W`). -/
  | div (src : Reg)
  /-- `idiv r64` (`48/49 f7 /7`): signed `rdx:rax / src` (trunc toward 0); sticks on `#DE`. -/
  | idiv (src : Reg)
  /-- `cqo` (`48 99`): sign-extend `rax` into `rdx:rax`. -/
  | cqo
  /-- `neg r64` (`48/49 f7 /3`): two's-complement negation, flags of `0 - d`. -/
  | neg (d : Reg)
  /-- `not r64` (`48/49 f7 /2`): bitwise complement, flags unchanged. -/
  | not (d : Reg)
  /-- `shl r64, cl` (`48/49 d3 /4`): shift count `rcx % 64`, flags `Flags.undef`. -/
  | shlR (d : Reg)
  /-- `shr r64, cl` (`48/49 d3 /5`): shift count `rcx % 64`, flags `Flags.undef`. -/
  | shrR (d : Reg)
  /-- `sar r64, cl` (`48/49 d3 /7`): arithmetic shift by `rcx % 64`, flags `Flags.undef`. -/
  | sarR (d : Reg)
  /-- `sar r64, imm8` (`48/49 c1 /7 imm8`): arithmetic shift by `c % 64`, flags `Flags.undef`. -/
  | sarI (d c : Nat)
  /-- `movzx r64, word [b + disp]` (`[REX !x] 0f b7 /r`, no `66`): 16-bit LE zero-extended load. -/
  | ld16 (d b disp : Nat)
  /-- `mov word [b + disp], r16` (`66 [REX !w, !x] 89 /r`): 16-bit LE store. -/
  | st16 (b disp s : Nat)
  /-- `mov dword [b + disp], r32` (`[REX !w, !x] 89 /r`): 32-bit LE store. -/
  | st32 (b disp s : Nat)
  /-- Indexed load from `[b + scale * i + disp]` (`scale ∈ {1, 2, 4, 8}`, `i ≠ 4`). -/
  | ldIdx (w : MemW) (d b i scale disp : Nat)
  /-- Indexed store to `[b + scale * i + disp]` (`scale ∈ {1, 2, 4, 8}`, `i ≠ 4`). -/
  | stIdx (w : MemW) (b i scale disp s : Nat)
  /-- `lea r64, [b + scale * i + disp]` (`[REX w=1] 8d /r` with SIB index `i ≠ 4`). -/
  | leaIdx (d b i scale disp : Nat)
  /-- `test r64, imm32` (`48/49 f7 /0 imm32`, sign-extended to 64 bits). -/
  | testI (a imm : Nat)
  /-- `cmp al, imm8` (`3c imm8`): 8-bit compare of `rax % 256` and `imm % 256`. -/
  | cmpAlI (imm : Nat)
  /-- `test al, imm8` (`a8 imm8`): 8-bit test of `rax % 256` and `imm % 256`. -/
  | testAlI (imm : Nat)
  /-- `setcc r8` (`[REX !w, r=0, !x] 0f (0x90+cc) /0`, mod=3): writes low byte to 0/1, preserves upper 56 bits & flags. -/
  | setcc (cc : Nat) (d : Reg)
  /-- `cmovcc r64, r64` (`[REX w=1, !x] 0f (0x40+cc) /r`, mod=3): conditional 64-bit move, preserves flags. -/
  | cmovcc (cc : Nat) (d s : Reg)
deriving DecidableEq, Repr

abbrev ofInt : X86Int.Ins → Ins := Ins.ofInt

abbrev Alts := Nat → List (Option (Ins × Nat))

def ofIntImage (im : Nat → Option (X86Int.Ins × Nat)) : Nat → Option (Ins × Nat) :=
  fun a => (im a).map fun (ins, len) => (ofInt ins, len)

def ofIntAlts (alts : X86Int.Alts) : Alts :=
  fun a => (alts a).map fun o => o.map fun (ins, len) => (ofInt ins, len)

/-! ### Arithmetic, signed conversion, and memory helpers -/

/-- Interpret a 64-bit word as a signed 64-bit integer in `[-2^63, 2^63 - 1]`. -/
def toSigned64 (x : Nat) : Int :=
  let x' := x % W
  if x' < 2 ^ 63 then (x' : Int) else (x' : Int) - (W : Int)

/-- Encode an integer modulo `2^64` into `[0, 2^64 - 1]`. -/
def ofSigned64 (i : Int) : Nat := (i % (W : Int)).toNat

/-- 64-bit arithmetic shift right by `c % 64`, matching x86-64 `sar r64, cl/imm8`. -/
def sar64 (a c : Nat) : Nat :=
  let a' := a % W
  let cnt := c % 64
  if X86Int.msb a' then a' / 2 ^ cnt + (W - W / 2 ^ cnt) else a' / 2 ^ cnt

/-- 8-bit MSB helper. -/
def msb8 (v : Nat) : Bool := v / 128 % 2 == 1

/-- 8-bit compare (`cmp al, imm8`) flags. -/
def cmp8F (a b : Nat) : Flags :=
  let a' := a % 256
  let b' := b % 256
  let res := (a' + 256 - b') % 256
  let cf := a' < b'
  let ofl := msb8 a' != msb8 b' && msb8 res != msb8 a'
  ⟨some cf, some (res == 0), some (msb8 res), some ofl⟩

/-- 8-bit test (`test al, imm8`) flags. -/
def test8F (a b : Nat) : Flags :=
  let res := Nat.land (a % 256) (b % 256)
  ⟨some false, some (res == 0), some (msb8 res), some false⟩

/-- Little-endian 2-byte and 4-byte encodings. -/
def bytes2 (v : Nat) : List Nat := [v % 256, v / 256 % 256]
def bytes4 (v : Nat) : List Nat := (List.range 4).map fun i => v / 256 ^ i % 256

def rd16 (m : Mem) (a : Nat) : Option Nat := (m.rdN a 2).map X86Int.leVal
def wr16 (m : Mem) (a v : Nat) : Option Mem := m.wrN a (bytes2 v)
def wr32 (m : Mem) (a v : Nat) : Option Mem := m.wrN a (bytes4 v)

def rdMem : MemW → Mem → Nat → Option Nat
  | .w8, m, a => m.rd a
  | .w16, m, a => rd16 m a
  | .w32, m, a => X86Int.rd32 m a
  | .w64, m, a => X86Int.rd64 m a

def wrMem : MemW → Mem → Nat → Nat → Option Mem
  | .w8, m, a, v => m.wr a (v % 256)
  | .w16, m, a, v => wr16 m a v
  | .w32, m, a, v => wr32 m a v
  | .w64, m, a, v => X86Int.wr64 m a v

/-- Indexed effective address `(b + scale * i + disp) % 2^64`. -/
def eaIdx (b i scale disp : Nat) : Nat := (b + scale * i + disp) % W

/-! ### Architectural semantics -/

/-- Execute one instruction `ins` of length `len` at `s.rip`. -/
def exec (ins : Ins) (len : Nat) (s : St) : Out :=
  let nx := s.rip + len
  let go (t : St) : St := { t with rip := nx }
  match ins with
  | .ofInt i => X86Int.exec i len s
  | .mul src => X86Int.lift do
      let a ← s.reg 0
      let b ← s.reg src
      let p := (a % W) * (b % W)
      let lo := p % W
      let hi := p / W
      let ov := hi != 0
      let f : Flags := ⟨some ov, none, none, some ov⟩
      pure (go { (s.set 0 (some lo)).set 2 (some hi) with fl := f })
  | .imul d src => X86Int.lift do
      let a ← s.reg d
      let b ← s.reg src
      let res := ((a % W) * (b % W)) % W
      let ov := toSigned64 a * toSigned64 b != toSigned64 res
      let f : Flags := ⟨some ov, none, none, some ov⟩
      pure (go { s.set d (some res) with fl := f })
  | .div src => X86Int.lift do
      let lo ← s.reg 0
      let hi ← s.reg 2
      let d ← s.reg src
      let den := d % W
      if den == 0 then none else
      let num := (hi % W) * W + (lo % W)
      let q := num / den
      let r := num % den
      if W ≤ q then none else
      pure (go { (s.set 0 (some q)).set 2 (some r) with fl := X86Int.Flags.undef })
  | .idiv src => X86Int.lift do
      let lo ← s.reg 0
      let hi ← s.reg 2
      let d ← s.reg src
      let den := toSigned64 d
      if den == 0 then none else
      let num : Int := toSigned64 hi * (W : Int) + ((lo % W : Nat) : Int)
      let q := Int.tdiv num den
      let r := Int.tmod num den
      let min64 : Int := -((2 ^ 63 : Nat) : Int)
      let lim64 : Int := ((2 ^ 63 : Nat) : Int)
      if decide (min64 ≤ q) && decide (q < lim64) then
        pure (go { (s.set 0 (some (ofSigned64 q))).set 2 (some (ofSigned64 r)) with
                   fl := X86Int.Flags.undef })
      else none
  | .cqo => X86Int.lift do
      let a ← s.reg 0
      let hi := if X86Int.msb (a % W) then W - 1 else 0
      pure (go (s.set 2 (some hi)))
  | .neg d => X86Int.lift do
      let a ← s.reg d
      let (v, f) := X86Int.aluF .sub 0 (a % W)
      pure (go { s.set d (some v) with fl := f })
  | .not d => X86Int.lift do
      let a ← s.reg d
      pure (go (s.set d (some (W - 1 - a % W))))
  | .shlR d => X86Int.lift do
      let a ← s.reg d
      let c ← s.reg 1
      pure (go { s.set d (some ((a % W) * 2 ^ (c % 64) % W)) with fl := X86Int.Flags.undef })
  | .shrR d => X86Int.lift do
      let a ← s.reg d
      let c ← s.reg 1
      pure (go { s.set d (some ((a % W) / 2 ^ (c % 64))) with fl := X86Int.Flags.undef })
  | .sarR d => X86Int.lift do
      let a ← s.reg d
      let c ← s.reg 1
      pure (go { s.set d (some (sar64 a c)) with fl := X86Int.Flags.undef })
  | .sarI d c => X86Int.lift do
      let a ← s.reg d
      pure (go { s.set d (some (sar64 a c)) with fl := X86Int.Flags.undef })
  | .ld16 d b disp => X86Int.lift do
      let a ← s.reg b
      let v ← rd16 s.mem (X86Int.ea a disp)
      pure (go (s.set d (some v)))
  | .st16 b disp r => X86Int.lift do
      let a ← s.reg b
      let v ← s.reg r
      let m ← wr16 s.mem (X86Int.ea a disp) v
      pure (go { s with mem := m })
  | .st32 b disp r => X86Int.lift do
      let a ← s.reg b
      let v ← s.reg r
      let m ← wr32 s.mem (X86Int.ea a disp) v
      pure (go { s with mem := m })
  | .ldIdx w d b i scale disp => X86Int.lift do
      let xb ← s.reg b
      let xi ← s.reg i
      let v ← rdMem w s.mem (eaIdx xb xi scale disp)
      pure (go (s.set d (some v)))
  | .stIdx w b i scale disp r => X86Int.lift do
      let xb ← s.reg b
      let xi ← s.reg i
      let v ← s.reg r
      let m ← wrMem w s.mem (eaIdx xb xi scale disp) v
      pure (go { s with mem := m })
  | .leaIdx d b i scale disp => X86Int.lift do
      let xb ← s.reg b
      let xi ← s.reg i
      pure (go (s.set d (some (eaIdx xb xi scale disp))))
  | .testI a imm => X86Int.lift do
      let x ← s.reg a
      pure (go { s with fl := (X86Int.aluF .and (x % W) (imm % W)).2 })
  | .cmpAlI imm => X86Int.lift do
      let rax ← s.reg 0
      pure (go { s with fl := cmp8F rax imm })
  | .testAlI imm => X86Int.lift do
      let rax ← s.reg 0
      pure (go { s with fl := test8F rax imm })
  | .setcc cc d => X86Int.lift do
      let c ← X86Int.cond s.fl cc
      let old ← s.reg d
      let old' := old % W
      pure (go (s.set d (some (old' - old' % 256 + if c then 1 else 0))))
  | .cmovcc cc d src => X86Int.lift do
      let c ← X86Int.cond s.fl cc
      let v ← s.reg src
      let old ← s.reg d
      pure (go (s.set d (some (if c then v else old))))

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

/-- Decode a SIB-indexed memory operand `[base + scale * idx + disp]` (`idx ≠ 4`,
`scale ∈ {1, 2, 4, 8}`, rejecting no-base `mod=0, SIB.base=5`).
Mutually exclusive with `X86Dec.memOp`. -/
def memOpIdx (b : Nat → Option Nat) (off m bb : Nat) (x : Bool) :
    Option (Nat × Nat × Nat × Nat × Nat) := do
  let md := m / 64
  let rm := m % 8
  if md == 3 || rm != 4 then none else
  let s ← b off
  let idx := s / 8 % 8 + (if x then 8 else 0)
  if idx == 4 then none else
  if md == 0 && s % 8 == 5 then none else
  let scale := 2 ^ (s / 64)
  let base := s % 8 + bb
  match md with
  | 0 => pure (base, idx, scale, 0, off + 1)
  | 1 => do let d ← b (off + 1); pure (base, idx, scale, X86Dec.sx8 d, off + 2)
  | _ => do let d ← X86Dec.leB b (off + 1) 4; pure (base, idx, scale, X86Dec.sx32 d, off + 5)

/-- Decode `X86Vm`-specific instruction forms omitted by `X86Dec.decode`. -/
def decodeVm (b : Nat → Option Nat) : Option (Ins × Nat) := do
  let b0 ← b 0
  if b0 == 0x66 then
    -- 16-bit store: `66 [REX !w] 89 /r` (non-indexed `st16` or SIB-indexed `stIdx .w16`)
    let b1 ← b 1
    let hasRex := 0x40 ≤ b1 && b1 ≤ 0x4f
    let rex := if hasRex then b1 - 0x40 else 0
    let p := if hasRex then 2 else 1
    let w := rex / 8 % 2 == 1
    let r := (rex / 4 % 2) * 8
    let x := rex / 2 % 2 == 1
    let bb := (rex % 2) * 8
    if w then none else
    let o ← b p
    if o != 0x89 then none else
    let m ← b (p + 1)
    if m / 64 == 3 then none else
    let rg := m / 8 % 8 + r
    if !x then
      match X86Dec.memOp b (p + 2) m bb x with
      | some (base, disp, off') => pure (.st16 base disp rg, off')
      | none =>
        let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
        pure (.stIdx .w16 base idx scale disp rg, off')
    else
      let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
      pure (.stIdx .w16 base idx scale disp rg, off')
  else
    let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
    let rex := if hasRex then b0 - 0x40 else 0
    let p := if hasRex then 1 else 0
    let w := rex / 8 % 2 == 1
    let r := (rex / 4 % 2) * 8
    let x := rex / 2 % 2 == 1
    let bb := (rex % 2) * 8
    let o ← b p
    if o == 0xf7 then
      if !w || r != 0 || x then none else
      let m ← b (p + 1)
      if m / 64 != 3 then none else
      let d := m % 8 + bb
      let ext := m / 8 % 8
      if ext == 0 then
        let imm ← X86Dec.leB b (p + 2) 4
        pure (.testI d (X86Dec.sx32 imm), p + 6)
      else if ext == 2 then pure (.not d, p + 2)
      else if ext == 3 then pure (.neg d, p + 2)
      else if ext == 4 then pure (.mul d, p + 2)
      else if ext == 6 then pure (.div d, p + 2)
      else if ext == 7 then pure (.idiv d, p + 2)
      else none
    else if o == 0x99 then
      if hasRex && w && r == 0 && !x && bb == 0 then pure (.cqo, 2) else none
    else if o == 0xd3 then
      if !w || r != 0 || x then none else
      let m ← b (p + 1)
      if m / 64 != 3 then none else
      let d := m % 8 + bb
      let ext := m / 8 % 8
      if ext == 4 then pure (.shlR d, p + 2)
      else if ext == 5 then pure (.shrR d, p + 2)
      else if ext == 7 then pure (.sarR d, p + 2)
      else none
    else if o == 0xc1 then
      if !w || r != 0 || x then none else
      let m ← b (p + 1)
      let c ← b (p + 2)
      if m / 64 != 3 || m / 8 % 8 != 7 then none else
      pure (.sarI (m % 8 + bb) c, p + 3)
    else if o == 0x3c then
      if hasRex then none else
      let imm ← b 1
      pure (.cmpAlI imm, 2)
    else if o == 0xa8 then
      if hasRex then none else
      let imm ← b 1
      pure (.testAlI imm, 2)
    else if o == 0x89 then
      let m ← b (p + 1)
      if m / 64 == 3 then none else
      let rg := m / 8 % 8 + r
      if !w then
        if !x then
          match X86Dec.memOp b (p + 2) m bb x with
          | some (base, disp, off') => pure (.st32 base disp rg, off')
          | none =>
            let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
            pure (.stIdx .w32 base idx scale disp rg, off')
        else
          let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
          pure (.stIdx .w32 base idx scale disp rg, off')
      else
        let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
        pure (.stIdx .w64 base idx scale disp rg, off')
    else if o == 0x8b then
      let m ← b (p + 1)
      if m / 64 == 3 then none else
      let rg := m / 8 % 8 + r
      let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
      let mw := if w then MemW.w64 else MemW.w32
      pure (.ldIdx mw rg base idx scale disp, off')
    else if o == 0x88 then
      let m ← b (p + 1)
      if m / 64 == 3 then none else
      let reg := m / 8 % 8
      if !hasRex && reg ≥ 4 then none else
      let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
      pure (.stIdx .w8 base idx scale disp (reg + r), off')
    else if o == 0x8d then
      if !w then none else
      let m ← b (p + 1)
      let (base, idx, scale, disp, off') ← memOpIdx b (p + 2) m bb x
      pure (.leaIdx (m / 8 % 8 + r) base idx scale disp, off')
    else if o == 0x0f then
      let o2 ← b (p + 1)
      if o2 == 0xb7 then
        let m ← b (p + 2)
        if m / 64 == 3 then none else
        let rg := m / 8 % 8 + r
        if !x then
          match X86Dec.memOp b (p + 3) m bb x with
          | some (base, disp, off') => pure (.ld16 rg base disp, off')
          | none =>
            let (base, idx, scale, disp, off') ← memOpIdx b (p + 3) m bb x
            pure (.ldIdx .w16 rg base idx scale disp, off')
        else
          let (base, idx, scale, disp, off') ← memOpIdx b (p + 3) m bb x
          pure (.ldIdx .w16 rg base idx scale disp, off')
      else if o2 == 0xb6 then
        let m ← b (p + 2)
        if m / 64 == 3 then none else
        let rg := m / 8 % 8 + r
        let (base, idx, scale, disp, off') ← memOpIdx b (p + 3) m bb x
        pure (.ldIdx .w8 rg base idx scale disp, off')
      else if o2 == 0xaf then
        if !w || x then none else
        let m ← b (p + 2)
        if m / 64 != 3 then none else
        pure (.imul (m / 8 % 8 + r) (m % 8 + bb), p + 3)
      else if 0x90 ≤ o2 && o2 ≤ 0x9f then
        if w || r != 0 || x then none else
        let m ← b (p + 2)
        if m / 64 != 3 || m / 8 % 8 != 0 then none else
        let rm := m % 8
        if !hasRex && rm ≥ 4 then none else
        pure (.setcc (o2 - 0x90) (rm + bb), p + 3)
      else if 0x40 ≤ o2 && o2 ≤ 0x4f then
        if !w || x then none else
        let m ← b (p + 2)
        if m / 64 != 3 then none else
        pure (.cmovcc (o2 - 0x40) (m / 8 % 8 + r) (m % 8 + bb), p + 3)
      else none
    else if o == 0xff then
      -- Indirect register jump `jmp r64`: `[REX !w, r=0, !x] ff /4` with mod=3
      if w || r != 0 || x then none else
      let m ← b (p + 1)
      if m / 64 != 3 || m / 8 % 8 != 4 then none else
      pure (.ofInt (.jmpR (m % 8 + bb)), p + 2)
    else none

/-- Unified VM byte decoder: tries `X86Dec.decode` first, then `decodeVm`. -/
def decode (b : Nat → Option Nat) (a : Nat) : Option (Ins × Nat) :=
  match X86Dec.decode b a with
  | some (ins, len) => some (.ofInt ins, len)
  | none => decodeVm b

/-- Hamming-distance-1 fetch neighbours for an instruction of length `len` at `a`. -/
def altsAt (b : Nat → Option Nat) (a len : Nat) : List (Option (Ins × Nat)) :=
  (List.range (8 * len)).map fun k => decode (X86Dec.flipBit b k) a

/-- The decoded instruction image of `code` loaded at virtual address `base`. -/
def imageOf (code : List Nat) (base : Nat) : Nat → Option (Ins × Nat) :=
  fun a => if a < base then none else decode (fun i => code[a - base + i]?) a

/-- Hamming-distance-1 fetch neighbour map for `code` loaded at `base`. -/
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

theorem altsAt_length (b : Nat → Option Nat) (a len : Nat) :
    (altsAt b a len).length = 8 * len := by
  simp [altsAt]

/-! ### Physical 4-channel semantics (`PDist`) -/

def execO (s : St) : Option (Ins × Nat) → Out
  | some (j, l) => exec j l s
  | none => .stuck

/-- One physical step: Hamming-1 fetch fault at rate `pf`, execution, then
retired-state faults `F` (register file, mapped memory, architectural flags). -/
def pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) (s : St) : PDF Out :=
  let fetched : PDF Out := match im s.rip with
    | none => pureF .stuck
    | some (ins, len) => X86Int.unif pf (exec ins len s) ((alts s.rip).map (execO s))
  bindF fetched fun o => match o with
    | .ok s' => X86Int.relabel X86Int.Out.ok (F.hit s')
    | o => pureF o

/-- Physical run of at most `n` steps. -/
def prun (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) : Nat → Out → PDF Out
  | 0, o => pureF o
  | n + 1, .ok s => bindF (pstep alts F pf im s) (prun alts F pf im n)
  | _ + 1, o => pureF o

/-! ### Conservative `ofInt` embedding theorems -/

theorem ofInt_exec (ins : X86Int.Ins) (len : Nat) (s : St) :
    exec (ofInt ins) len s = X86Int.exec ins len s := rfl

theorem ofInt_step (im : Nat → Option (X86Int.Ins × Nat)) (s : St) :
    step (ofIntImage im) s = X86Int.step im s := by
  unfold step X86Int.step ofIntImage
  cases im s.rip with
  | none => rfl
  | some p => rfl

theorem ofInt_run (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ n o, run (ofIntImage im) n o = X86Int.run im n o := by
  intro n
  induction n with
  | zero => intro o; rfl
  | succ n ih =>
    intro o
    cases o with
    | ok s => simp only [run, X86Int.run, ofInt_step, ih]
    | exit c => rfl
    | stuck => rfl

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

/-! ### Physical distribution, fault-free projection, and baseline bound -/

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

/-- The 4-channel baseline bound: for rates anywhere in their calibration
intervals `(cf, cr, cm, cl)`, a physical run of `n` steps faults with
probability at most `n * (cf.hi + cr.hi + cm.hi + cl.hi)`. -/
theorem baseline (alts : Alts) (im : Nat → Option (Ins × Nat)) (cf cr cm cl : Iv)
    (r : Rates) (hr : r.ok) (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem)
    (hl : cl.mem r.flag) (n : Nat) (o : Out) :
    fm (prun alts (r.state hr) r.fetch im n o) ≤ n * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  refine Rat.le_trans (fm_prun alts (r.state hr) hr.1 im n o) ?_
  refine Rat.mul_le_mul_of_nonneg_left ?_ Rat.natCast_nonneg
  show r.fetch + (r.reg + (r.mem + r.flag)) ≤ _
  exact add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))

/-! ### Determinism, composition, and semantic helper lemmas -/

theorem exec_deterministic {ins : Ins} {len : Nat} {s : St} {o₁ o₂ : Out}
    (h1 : exec ins len s = o₁) (h2 : exec ins len s = o₂) : o₁ = o₂ :=
  h1.symm.trans h2

theorem step_deterministic {im : Nat → Option (Ins × Nat)} {s : St} {o₁ o₂ : Out}
    (h1 : step im s = o₁) (h2 : step im s = o₂) : o₁ = o₂ :=
  h1.symm.trans h2

theorem run_deterministic {im : Nat → Option (Ins × Nat)} {n : Nat} {o o₁ o₂ : Out}
    (h1 : run im n o = o₁) (h2 : run im n o = o₂) : o₁ = o₂ :=
  h1.symm.trans h2

theorem faultFree_prun_unique (alts : Alts) (F : Fault) (pf : Rat)
    (im : Nat → Option (Ins × Nat)) (n : Nat) (o o' : Out)
    (h : o' ∈ faultFree (prun alts F pf im n o)) : o' = run im n o := by
  rw [faultFree_prun] at h
  simpa using h

theorem run_zero (im : Nat → Option (Ins × Nat)) (o : Out) : run im 0 o = o := rfl

theorem run_succ_ok (im : Nat → Option (Ins × Nat)) (n : Nat) (s : St) :
    run im (n + 1) (.ok s) = run im n (step im s) := rfl

theorem run_exit (im : Nat → Option (Ins × Nat)) (c : Nat) : ∀ n, run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

theorem run_stuck (im : Nat → Option (Ins × Nat)) : ∀ n, run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

theorem run_add (im : Nat → Option (Ins × Nat)) :
    ∀ (m n : Nat) (o : Out), run im (m + n) o = run im n (run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [run, run_add im m n (step im s)]
  | _ + 1, n, .exit c => by simp [run, run_exit]
  | _ + 1, n, .stuck => by simp [run, run_stuck]

theorem step_jtab {im : Nat → Option (Ins × Nat)} {s : St} {b i disp len x y t : Nat}
    (hi : im s.rip = some (.ofInt (.jtab b i disp), len)) (hx : s.reg b = some x)
    (hy : s.reg i = some y) (ht : X86Int.rd64 s.mem ((x + 8 * y + disp) % W) = some t) :
    step im s = .ok { s with rip := t } := by
  simp only [step, exec, X86Int.exec, hi, hx, hy, ht, X86Int.lift, bind, Option.bind, pure]

/-! ### Exact arithmetic bridges for `sporos_vm` proofs -/

theorem sar64_lt_W (a c : Nat) : sar64 a c < W := by
  unfold sar64
  have hmod : a % W < W := Nat.mod_lt _ (by decide)
  have hcnt : c % 64 < 64 := Nat.mod_lt _ (by decide)
  have hd_pos : 0 < 2 ^ (c % 64) := Nat.pow_pos (by decide)
  have hd_dvd : 2 ^ (c % 64) ∣ W := by
    unfold W
    exact Nat.pow_dvd_pow 2 (Nat.le_of_lt hcnt)
  have hdiv_lt : a % W / 2 ^ (c % 64) < W / 2 ^ (c % 64) := by
    rw [Nat.div_lt_iff_lt_mul hd_pos]
    rw [Nat.div_mul_cancel hd_dvd]
    exact hmod
  have hwd_le : W / 2 ^ (c % 64) ≤ W := Nat.div_le_self W _
  dsimp only
  split <;> omega

theorem sar64_eq_ofSigned64 (a c : Nat) :
    sar64 a c = ofSigned64 (toSigned64 a / ((2 ^ (c % 64) : Nat) : Int)) := by
  unfold sar64 toSigned64 ofSigned64
  dsimp only
  have hW : W = 2 ^ 64 := rfl
  have hmod : a % W < W := Nat.mod_lt _ (by decide)
  have hcnt : c % 64 < 64 := Nat.mod_lt _ (by decide)
  have hd_pos : 0 < 2 ^ (c % 64) := Nat.pow_pos (by decide)
  have hd_dvd : 2 ^ (c % 64) ∣ W := by
    rw [hW]; exact Nat.pow_dvd_pow 2 (Nat.le_of_lt hcnt)
  have hmul : (W / 2 ^ (c % 64)) * 2 ^ (c % 64) = W := Nat.div_mul_cancel hd_dvd
  have hdiv_lt : a % W / 2 ^ (c % 64) < W / 2 ^ (c % 64) := by
    rw [Nat.div_lt_iff_lt_mul hd_pos, hmul]; exact hmod
  have hwd_le : W / 2 ^ (c % 64) ≤ W := Nat.div_le_self W _
  by_cases hlt : a % W < 2 ^ 63
  · have hmsb : X86Int.msb (a % W) = false := by
      unfold X86Int.msb
      have : a % W / 2 ^ 63 = 0 := Nat.div_eq_of_lt hlt
      simp [this]
    simp only [hmsb, Bool.false_eq_true, ↓reduceIte, hlt]
    have hq_lt : a % W / 2 ^ (c % 64) < W := Nat.lt_of_lt_of_le hdiv_lt hwd_le
    have hdiv_nat : ((a % W : Nat) : Int) / ((2 ^ (c % 64) : Nat) : Int) = ((a % W / 2 ^ (c % 64) : Nat) : Int) := by
      exact_mod_cast (rfl : a % W / 2 ^ (c % 64) = a % W / 2 ^ (c % 64))
    rw [hdiv_nat, Int.emod_eq_of_lt (Int.natCast_nonneg _) (Int.ofNat_lt.mpr hq_lt)]
    rfl
  · have hge : 2 ^ 63 ≤ a % W := Nat.le_of_not_lt hlt
    have hmsb : X86Int.msb (a % W) = true := by
      unfold X86Int.msb
      have h1 : a % W / 2 ^ 63 = 1 := by
        have : W = 2 * 2 ^ 63 := by decide
        omega
      simp [h1]
    simp only [hmsb, ↓reduceIte, hlt]
    have hd_ne : ((2 ^ (c % 64) : Nat) : Int) ≠ 0 := by omega
    have hW_int : (W : Int) = ((W / 2 ^ (c % 64) : Nat) : Int) * ((2 ^ (c % 64) : Nat) : Int) := by
      exact_mod_cast hmul.symm
    have hdiv_nat : ((a % W : Nat) : Int) / ((2 ^ (c % 64) : Nat) : Int) = ((a % W / 2 ^ (c % 64) : Nat) : Int) := by
      exact_mod_cast (rfl : a % W / 2 ^ (c % 64) = a % W / 2 ^ (c % 64))
    have hdiv_int :
        (((a % W : Nat) : Int) - (W : Int)) / ((2 ^ (c % 64) : Nat) : Int) =
          (((a % W / 2 ^ (c % 64) : Nat) : Int) - ((W / 2 ^ (c % 64) : Nat) : Int)) := by
      rw [hW_int, Int.sub_eq_add_neg, ← Int.neg_mul, Int.add_mul_ediv_right _ _ hd_ne]
      rw [hdiv_nat, Int.sub_eq_add_neg]
    rw [hdiv_int]
    have hsum_eq :
        (((a % W / 2 ^ (c % 64) : Nat) : Int) - ((W / 2 ^ (c % 64) : Nat) : Int)) + (W : Int) =
          ((a % W / 2 ^ (c % 64) + (W - W / 2 ^ (c % 64)) : Nat) : Int) := by
      omega
    have hsum_lt : a % W / 2 ^ (c % 64) + (W - W / 2 ^ (c % 64)) < W := by omega
    have hmod_add :
        (((a % W / 2 ^ (c % 64) : Nat) : Int) - ((W / 2 ^ (c % 64) : Nat) : Int)) % (W : Int) =
          ((((a % W / 2 ^ (c % 64) : Nat) : Int) - ((W / 2 ^ (c % 64) : Nat) : Int)) + (W : Int)) % (W : Int) := by
      rw [Int.add_emod_right]
    rw [hmod_add, hsum_eq, Int.emod_eq_of_lt (Int.natCast_nonneg _) (Int.ofNat_lt.mpr hsum_lt)]
    rfl

/-- `cqo` sign-extends `rax` into `rdx:rax` so the 128-bit signed dividend equals `toSigned64 rax`. -/
theorem cqo_dividend_eq (a : Nat) :
    let hi := if X86Int.msb (a % W) then W - 1 else 0
    toSigned64 hi * (W : Int) + ((a % W : Nat) : Int) = toSigned64 a := by
  dsimp only
  unfold toSigned64
  have hmod : a % W < W := Nat.mod_lt _ (by decide)
  by_cases hlt : a % W < 2 ^ 63
  · have hmsb : X86Int.msb (a % W) = false := by
      unfold X86Int.msb; rw [Nat.div_eq_of_lt hlt]; decide
    simp [hmsb, hlt]
  · have hge : 2 ^ 63 ≤ a % W := Nat.le_of_not_lt hlt
    have hmsb : X86Int.msb (a % W) = true := by
      unfold X86Int.msb
      have : a % W / 2 ^ 63 = 1 := by
        have : W = 2 * 2 ^ 63 := by decide
        omega
      simp [this]
    have hwm1 : (W - 1) % W = W - 1 := Nat.mod_eq_of_lt (by decide)
    have hnot : ¬ (W - 1 < 2 ^ 63) := by decide
    have hneg1 : ((W - 1 : Nat) : Int) - (W : Int) = -1 := by decide
    simp only [hmsb, ↓reduceIte, hwm1, hnot, hlt, hneg1]
    omega

theorem div_zero_stuck {src len lo hi : Nat} {s : St}
    (hlo : s.reg 0 = some lo) (hhi : s.reg 2 = some hi) (hd : s.reg src = some 0) :
    exec (.div src) len s = .stuck := by
  simp [exec, hlo, hhi, hd, X86Int.lift]

theorem idiv_zero_stuck {src len lo hi : Nat} {s : St}
    (hlo : s.reg 0 = some lo) (hhi : s.reg 2 = some hi) (hd : s.reg src = some 0) :
    exec (.idiv src) len s = .stuck := by
  simp [exec, hlo, hhi, hd, toSigned64, X86Int.lift]

theorem idiv_int64min_neg1_stuck {src len : Nat} {s : St}
    (hlo : s.reg 0 = some (2 ^ 63)) (hhi : s.reg 2 = some (W - 1))
    (hd : s.reg src = some (W - 1)) :
    exec (.idiv src) len s = .stuck := by
  have hnz : (toSigned64 (W - 1) == 0) = false := by decide
  have hov :
      (decide (-((2 ^ 63 : Nat) : Int) ≤
          (toSigned64 (W - 1) * (W : Int) + ((2 ^ 63 % W : Nat) : Int)).tdiv (toSigned64 (W - 1))) &&
       decide ((toSigned64 (W - 1) * (W : Int) + ((2 ^ 63 % W : Nat) : Int)).tdiv (toSigned64 (W - 1)) <
          ((2 ^ 63 : Nat) : Int))) = false := by decide
  simp only [exec, hlo, hhi, hd, X86Int.lift, bind, Option.bind, pure, hnz, hov, Bool.false_eq_true, ↓reduceIte]

/-! ### Byte-encoding checks against Intel SDM vol. 2 -/

def at0 (bs : List Nat) (a : Nat := 0) : Option (Ins × Nat) := decode (fun i => bs[i]?) a

-- Group 3 (0xF7): testI, not, neg, mul, div, idiv
example : at0 [0x48, 0xf7, 0xc0, 0x0f, 0x00, 0x00, 0x00] = some (.testI 0 0x0f, 7) := by decide +kernel
example : at0 [0x49, 0xf7, 0xc0, 0xff, 0xff, 0xff, 0xff] = some (.testI 8 (W - 1), 7) := by decide +kernel
example : at0 [0x48, 0xf7, 0xd0] = some (.not 0, 3) := by decide +kernel
example : at0 [0x49, 0xf7, 0xd4] = some (.not 12, 3) := by decide +kernel
example : at0 [0x48, 0xf7, 0xd8] = some (.neg 0, 3) := by decide +kernel
example : at0 [0x49, 0xf7, 0xdf] = some (.neg 15, 3) := by decide +kernel
example : at0 [0x48, 0xf7, 0xe1] = some (.mul 1, 3) := by decide +kernel
example : at0 [0x49, 0xf7, 0xe2] = some (.mul 10, 3) := by decide +kernel
example : at0 [0x48, 0xf7, 0xf1] = some (.div 1, 3) := by decide +kernel
example : at0 [0x49, 0xf7, 0xf3] = some (.div 11, 3) := by decide +kernel
example : at0 [0x48, 0xf7, 0xf9] = some (.idiv 1, 3) := by decide +kernel
example : at0 [0x49, 0xf7, 0xfb] = some (.idiv 11, 3) := by decide +kernel

-- cqo & imul
example : at0 [0x48, 0x99] = some (.cqo, 2) := by decide +kernel
example : at0 [0x48, 0x0f, 0xaf, 0xc1] = some (.imul 0 1, 4) := by decide +kernel
example : at0 [0x4d, 0x0f, 0xaf, 0xd3] = some (.imul 10 11, 4) := by decide +kernel

-- variable shifts by CL & immediate sar
example : at0 [0x48, 0xd3, 0xe0] = some (.shlR 0, 3) := by decide +kernel
example : at0 [0x49, 0xd3, 0xe8] = some (.shrR 8, 3) := by decide +kernel
example : at0 [0x48, 0xd3, 0xf8] = some (.sarR 0, 3) := by decide +kernel
example : at0 [0x49, 0xd3, 0xff] = some (.sarR 15, 3) := by decide +kernel
example : at0 [0x48, 0xc1, 0xf8, 0x03] = some (.sarI 0 3, 4) := by decide +kernel

-- 16-bit load/store & 32-bit store
example : at0 [0x0f, 0xb7, 0x07] = some (.ld16 0 7 0, 3) := by decide +kernel
example : at0 [0x48, 0x0f, 0xb7, 0x45, 0xf8] = some (.ld16 0 5 (W - 8), 5) := by decide +kernel
example : at0 [0x66, 0x89, 0x07] = some (.st16 7 0 0, 3) := by decide +kernel
example : at0 [0x66, 0x45, 0x89, 0x44, 0x24, 0x10] = some (.st16 12 16 8, 6) := by decide +kernel
example : at0 [0x89, 0x07] = some (.st32 7 0 0, 2) := by decide +kernel
example : at0 [0x44, 0x89, 0x47, 0x04] = some (.st32 7 4 8, 4) := by decide +kernel
-- refused: 66 + REX.W on 89 (would be 64-bit, not 16-bit)
example : at0 [0x66, 0x48, 0x89, 0x07] = none := by decide +kernel

-- SIB-indexed loads, stores, and lea
-- 49 0f b6 04 3c: movzx rax, byte [r12 + rdi*1]
example : at0 [0x49, 0x0f, 0xb6, 0x04, 0x3c] = some (.ldIdx .w8 0 12 7 1 0, 5) := by decide +kernel
-- 49 0f b7 04 3c: movzx rax, word [r12 + rdi*1]
example : at0 [0x49, 0x0f, 0xb7, 0x04, 0x3c] = some (.ldIdx .w16 0 12 7 1 0, 5) := by decide +kernel
-- 41 8b 04 3c: mov eax, dword [r12 + rdi*1]
example : at0 [0x41, 0x8b, 0x04, 0x3c] = some (.ldIdx .w32 0 12 7 1 0, 4) := by decide +kernel
-- 49 8b 04 fd: mov rax, qword [r13 + rdi*8 + 0] (mod=1 disp8=0 because base=5!)
example : at0 [0x49, 0x8b, 0x44, 0xfd, 0x00] = some (.ldIdx .w64 0 13 7 8 0, 5) := by decide +kernel
-- 41 88 04 3c: mov byte [r12 + rdi*1], al
example : at0 [0x41, 0x88, 0x04, 0x3c] = some (.stIdx .w8 12 7 1 0 0, 4) := by decide +kernel
-- 66 41 89 04 3c: mov word [r12 + rdi*1], ax
example : at0 [0x66, 0x41, 0x89, 0x04, 0x3c] = some (.stIdx .w16 12 7 1 0 0, 5) := by decide +kernel
-- 41 89 04 3c: mov dword [r12 + rdi*1], eax
example : at0 [0x41, 0x89, 0x04, 0x3c] = some (.stIdx .w32 12 7 1 0 0, 4) := by decide +kernel
-- 49 89 54 fd 00: mov qword [r13 + rdi*8 + 0], rdx
example : at0 [0x49, 0x89, 0x54, 0xfd, 0x00] = some (.stIdx .w64 13 7 8 0 2, 5) := by decide +kernel
-- 4a 8d 44 e3 10: lea rax, [rbx + r12*8 + 16] (REX.WX, idx=12)
example : at0 [0x4a, 0x8d, 0x44, 0xe3, 0x10] = some (.leaIdx 0 3 12 8 16, 5) := by decide +kernel
-- refused: mod=0 with SIB.base=5 (no base register)
example : at0 [0x49, 0x8b, 0x04, 0xfd, 0x00, 0x00, 0x00, 0x00] = none := by decide +kernel

-- cmpAlI, testAlI, setcc, cmovcc, jmpR
example : at0 [0x3c, 0x80] = some (.cmpAlI 0x80, 2) := by decide +kernel
example : at0 [0xa8, 0x01] = some (.testAlI 0x01, 2) := by decide +kernel
example : at0 [0x0f, 0x9c, 0xc0] = some (.setcc 12 0, 3) := by decide +kernel
example : at0 [0x40, 0x0f, 0x94, 0xc6] = some (.setcc 4 6, 4) := by decide +kernel
example : at0 [0x0f, 0x94, 0xc6] = none := by decide +kernel
example : at0 [0x48, 0x0f, 0x4c, 0xc3] = some (.cmovcc 12 0 3, 4) := by decide +kernel
example : at0 [0xff, 0xe0] = some (.ofInt (.jmpR 0), 2) := by decide +kernel
example : at0 [0x41, 0xff, 0xe0] = some (.ofInt (.jmpR 8), 3) := by decide +kernel

/-! ### Kernel-checked execution and physical witness -/

/-- A concrete byte-assembled program exercising `neg`, `cqo`, `idiv`, `sarR`,
`st16`, `ld16`, `leaIdx`, `setcc`, `cmovcc`, and `syscall` (`exit`):
  1. `mov eax, 100`        -> rax = 100
  2. `neg rax`             -> rax = -100
  3. `cqo`                 -> rdx = -1
  4. `mov ecx, 7`          -> rcx = 7
  5. `idiv rcx`            -> rax = -14, rdx = -2
  6. `mov ecx, 1`          -> rcx = 1
  7. `sar rax, cl`         -> rax = -7 (arithmetic shift right!)
  8. `neg rax`             -> rax = 7
  9. `mov ebx, 0x1000`     -> rbx = 0x1000
 10. `mov word [rbx], ax`  -> mem[0x1000..0x1001] = 0x0007
 11. `movzx rdi, word [rbx]` -> rdi = 7
 12. `mov eax, 60; syscall` -> exit(7)
-/
def witnessCode : List Nat :=
  [ 0xb8, 0x64, 0x00, 0x00, 0x00       -- mov eax, 100
  , 0x48, 0xf7, 0xd8                   -- neg rax
  , 0x48, 0x99                         -- cqo
  , 0xb9, 0x07, 0x00, 0x00, 0x00       -- mov ecx, 7
  , 0x48, 0xf7, 0xf9                   -- idiv rcx
  , 0xb9, 0x01, 0x00, 0x00, 0x00       -- mov ecx, 1
  , 0x48, 0xd3, 0xf8                   -- sar rax, cl
  , 0x48, 0xf7, 0xd8                   -- neg rax
  , 0xbb, 0x00, 0x10, 0x00, 0x00       -- mov ebx, 0x1000
  , 0x66, 0x89, 0x03                   -- mov word [rbx], ax
  , 0x48, 0x0f, 0xb7, 0x3b             -- movzx rdi, word [rbx]
  , 0xb8, 0x3c, 0x00, 0x00, 0x00       -- mov eax, 60
  , 0x0f, 0x05                         -- syscall
  ]

def witnessMem : Mem := [⟨0x1000, List.replicate 16 0, true⟩]
def witnessSt : St := ⟨List.replicate 16 (some 0), 0x400000, X86Int.Flags.undef, witnessMem, ⟨[], [], [], [], []⟩⟩

theorem witness_vm_run :
    X86Int.Out.code (run (imageOf witnessCode 0x400000) 13 (.ok witnessSt)) = some 7 := by
  decide +kernel

theorem witness_vm_phys (r : Rates) (hr : r.ok)
    (hf : p7.mem r.fetch) (hg : p7.mem r.reg) (hm : p7.mem r.mem) (hl : p7.mem r.flag) :
    let im := imageOf witnessCode 0x400000
    let alts := imageAlts witnessCode 0x400000
    (prun alts (r.state hr) r.fetch im 13 (.ok witnessSt)).ok ∧
    fm (prun alts (r.state hr) r.fetch im 13 (.ok witnessSt)) ≤ 13 * (4 / 5000000) ∧
    faultFree (prun alts (r.state hr) r.fetch im 13 (.ok witnessSt)) =
      [run im 13 (.ok witnessSt)] := by
  refine ⟨prun_ok _ _ hr.1 _ _ _, ?_, faultFree_prun _ _ _ _ _ _⟩
  have := baseline (imageAlts witnessCode 0x400000) (imageOf witnessCode 0x400000)
    p7 p7 p7 p7 r hr hf hg hm hl 13 (.ok witnessSt)
  exact Rat.le_trans this (by decide +kernel)

end X86Vm
