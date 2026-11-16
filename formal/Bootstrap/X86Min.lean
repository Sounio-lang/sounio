/-
X86Min: a minimal semantics of x86-64 user code under Linux, written for the
two bootstrap seed loaders (bootstrap/roots/a/seed-a, bootstrap/roots/b/seed-b).

Scope, on purpose:
  - only the instruction forms the two seeds use are decoded; every other byte
    sequence decodes to nothing and the machine is stuck (fail closed);
  - registers, flags and stack bytes are `Option`: at entry only rsp is
    defined, and reading an undefined value makes the machine stuck. A run that
    ends in `exit` therefore also shows that nothing was read before it was
    written;
  - flags the architecture leaves undefined (OF after a shift by more than 1)
    become `none`;
  - the environment never fails: read returns the next input byte or end of
    input, write always writes one byte. The seeds' failure paths for a failed
    read or a short write are not exercised by this model;
  - PF and AF are not modelled; a branch on PF is stuck.

Values are natural numbers below 2^width; no Mathlib, no native code, so the
theorems in Bootstrap/Seeds.lean are checked by the kernel alone.

The semantics is tested against the CPU by the differential harness in
bootstrap/model/ (instruction level and whole-program level).
-/
namespace X86Min

def pow2 (w : Nat) : Nat := 2 ^ w

/-- Most significant bit of `v` at width `w`. -/
def msb (w v : Nat) : Bool := v / pow2 (w - 1) % 2 == 1

/-- Sign-extend an 8-bit value to 64 bits. -/
def sx8 (v : Nat) : Nat := if v < 128 then v else v + pow2 64 - 256

/-- Sign-extend a 32-bit value to 64 bits. -/
def sx32 (v : Nat) : Nat := if v < pow2 31 then v else v + pow2 64 - pow2 32

structure Flags where
  cf : Option Bool
  zf : Option Bool
  sf : Option Bool
  of : Option Bool
deriving DecidableEq, Repr

def Flags.undef : Flags := ⟨none, none, none, none⟩

/-- Register numbers: 0 rax 1 rcx 2 rdx 3 rbx 4 rsp 5 rbp 6 rsi 7 rdi 8..15 r8..r15. -/
abbrev Reg := Nat

inductive Op where
  | add | sub | cmp | xor | or | test | mov
deriving DecidableEq, Repr

inductive Ins where
  /-- `mov r32, imm32` (B8+r): writes the register zero-extended. -/
  | movImm (dst imm : Nat)
  /-- register-register `op r/m, r` at width 32 or 64 (31 xor, 09 or, 85 test, 89 mov). -/
  | rr (op : Op) (w dst src : Nat)
  /-- `op r/m, imm` at width 8 (AL forms 04 3c 2c), 32 or 64 (81 /7, 83 /7). `imm` is
  already reduced to the operand width. -/
  | ri (op : Op) (w dst imm : Nat)
  /-- `shl r/m32, imm8` (C1 /4). -/
  | shlI (w dst cnt : Nat)
  /-- `inc r/m` (FF /0). -/
  | inc (w dst : Nat)
  /-- `push imm8`, sign-extended to 64 bits (6A). -/
  | push (imm : Nat)
  /-- `pop r64` (58+r). -/
  | pop (dst : Nat)
  | syscall
  /-- `jcc rel` (70+cc rel8). `rel` is sign-extended to 64 bits. -/
  | jcc (cc rel : Nat)
  /-- `jmp rel8` (EB) or `jmp rel32` (E9), sign-extended to 64 bits. -/
  | jmp (rel : Nat)
  /-- `movzx r32, byte [rsp]` (0F B6 /r, SIB 24). -/
  | ldRsp8 (dst : Nat)
  /-- `mov byte [rsp], r8` (88 /r, SIB 24). -/
  | stRsp8 (src : Nat)
deriving DecidableEq, Repr

/-- Little-endian value of `n` bytes, each read through `b`. -/
def le (b : Nat → Option Nat) (off : Nat) : Nat → Option Nat
  | 0 => some 0
  | n + 1 => do
    let lo ← b off
    let hi ← le b (off + 1) n
    pure (lo + 256 * hi)

/-- Decode one instruction. `b i` is the byte at offset `i` from rip. Returns the
instruction and its length. -/
def decode (b : Nat → Option Nat) : Option (Ins × Nat) := do
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
  if 0xb8 ≤ o && o ≤ 0xbf then
    if w || r != 0 || x then none else
    let imm ← le b (p + 1) 4
    pure (.movImm (o - 0xb8 + bb) imm, p + 5)
  else if o == 0x31 || o == 0x09 || o == 0x85 || o == 0x89 then
    let m ← b (p + 1)
    if m / 64 != 3 || x then none else
    let op := if o == 0x31 then Op.xor else if o == 0x09 then Op.or
      else if o == 0x85 then Op.test else Op.mov
    pure (.rr op wd (m % 8 + bb) (m / 8 % 8 + r), p + 2)
  else if o == 0x88 then
    let m ← b (p + 1)
    let s ← b (p + 2)
    if m / 64 != 0 || m % 8 != 4 || s != 0x24 || w || x || bb != 0 then none else
    let reg := m / 8 % 8
    if !hasRex && reg ≥ 4 then none else
    pure (.stRsp8 (reg + r), p + 3)
  else if o == 0x6a then
    if hasRex then none else
    let i ← b 1
    pure (.push (sx8 i), 2)
  else if 0x58 ≤ o && o ≤ 0x5f then
    if w || r != 0 || x then none else
    pure (.pop (o - 0x58 + bb), p + 1)
  else if o == 0x0f then
    let o2 ← b (p + 1)
    if o2 == 0x05 then
      if hasRex then none else pure (.syscall, 2)
    else if o2 == 0xb6 then
      let m ← b (p + 2)
      let s ← b (p + 3)
      if m / 64 != 0 || m % 8 != 4 || s != 0x24 || w || x || bb != 0 then none else
      pure (.ldRsp8 (m / 8 % 8 + r), p + 4)
    else none
  else if 0x70 ≤ o && o ≤ 0x7f then
    if hasRex then none else
    let d ← b 1
    pure (.jcc (o - 0x70) (sx8 d), 2)
  else if o == 0xeb then
    if hasRex then none else
    let d ← b 1
    pure (.jmp (sx8 d), 2)
  else if o == 0xe9 then
    if hasRex then none else
    let d ← le b 1 4
    pure (.jmp (sx32 d), 5)
  else if o == 0x04 || o == 0x3c || o == 0x2c then
    if hasRex then none else
    let i ← b 1
    let op := if o == 0x04 then Op.add else if o == 0x3c then Op.cmp else Op.sub
    pure (.ri op 8 0 i, 2)
  else if o == 0xc1 then
    let m ← b (p + 1)
    let c ← b (p + 2)
    if m / 64 != 3 || m / 8 % 8 != 4 || w || r != 0 || x then none else
    pure (.shlI 32 (m % 8 + bb) c, p + 3)
  else if o == 0x81 || o == 0x83 then
    let m ← b (p + 1)
    if m / 64 != 3 || m / 8 % 8 != 7 || r != 0 || x then none else
    if o == 0x81 then
      let i ← le b (p + 2) 4
      pure (.ri .cmp wd (m % 8 + bb) (sx32 i % pow2 wd), p + 6)
    else
      let i ← b (p + 2)
      pure (.ri .cmp wd (m % 8 + bb) (sx8 i % pow2 wd), p + 3)
  else if o == 0xff then
    let m ← b (p + 1)
    if m / 64 != 3 || m / 8 % 8 != 0 || r != 0 || x then none else
    pure (.inc wd (m % 8 + bb), p + 2)
  else none

/-- Size of the stack window, in bytes; the window is centred on the entry rsp. -/
def winSize : Nat := 32

/-- A complete binary tree of depth `d` with a value at each of its `2^d` leaves. -/
inductive Tree (α : Type) where
  | leaf (v : α)
  | node (l r : Tree α)
deriving DecidableEq, Repr

/-- The tree whose leaf `i` holds `f (b + i)`. -/
def Tree.build {α : Type} (f : Nat → α) : Nat → Nat → Tree α
  | 0, b => .leaf (f b)
  | d + 1, b => .node (build f d b) (build f d (b + 2 ^ d))

def Tree.get {α : Type} (dflt : α) : Nat → Tree α → Nat → α
  | 0, .leaf v, _ => v
  | d + 1, .node l r, i => if i < 2 ^ d then get dflt d l i else get dflt d r (i - 2 ^ d)
  | _, _, _ => dflt

theorem Tree.get_build {α : Type} (dflt : α) (f : Nat → α) :
    ∀ (d b i : Nat), i < 2 ^ d → Tree.get dflt d (Tree.build f d b) i = f (b + i) := by
  intro d
  induction d with
  | zero => intro b i h; simp at h; subst h; rfl
  | succ d ih =>
    intro b i h
    simp only [Tree.build, Tree.get]
    split
    · exact ih b i ‹_›
    · rw [ih (b + 2 ^ d) (i - 2 ^ d) (by rw [Nat.pow_succ] at h; omega)]
      congr 1; omega

/-- Depth of the decode table: files of up to 512 bytes. -/
def tdepth : Nat := 9

/-- The loaded image: the file, the address it is mapped at, and the decode
table, whose entry `i` is by definition `decode` applied to the bytes at file
offset `i` (see `Img.dec_spec`). It is a parameter of `step` and `run`, not part of
the state, so that the kernel decodes each offset once for the whole run. -/
structure Img where
  code : List Nat
  vbase : Nat
  len : Nat
  dec : Tree (Option (Ins × Nat))
deriving DecidableEq

def fetchIn (code : List Nat) (len i : Nat) : Option Nat :=
  if i < len then code[i]? else none

def Img.mk' (code : List Nat) (vbase len : Nat) : Img :=
  ⟨code, vbase, len, Tree.build (fun i => decode (fun j => fetchIn code len (i + j))) tdepth 0⟩

/-- The instruction at address `a`, if `a` is in the image. -/
def Img.decodeAt (im : Img) (a : Nat) : Option (Ins × Nat) :=
  if im.vbase ≤ a && a < im.vbase + im.len then Tree.get none tdepth im.dec (a - im.vbase) else none

/-- The decode table holds what `decode` says at each offset of the file. -/
theorem Img.dec_spec (code : List Nat) (vbase len i : Nat) (h : i < 2 ^ tdepth) :
    Tree.get none tdepth (Img.mk' code vbase len).dec i
      = decode (fun j => fetchIn code len (i + j)) := by
  simp only [Img.mk']
  rw [Tree.get_build none _ tdepth 0 i h, Nat.zero_add]

structure St where
  regs : List (Option Nat)
  rip  : Nat
  fl   : Flags
  /-- stack bytes; `mem[i]` is the byte at address `lo + i` -/
  mem  : List (Option Nat)
  lo   : Nat
  inp  : List Nat
  /-- bytes written to stdout, most recent first -/
  out  : List Nat
deriving DecidableEq, Repr

def St.reg (s : St) (i : Nat) : Option Nat := (s.regs[i]?).getD none

def St.setReg (s : St) (i : Nat) (v : Option Nat) : St := { s with regs := s.regs.set i v }

def St.rdByte (s : St) (a : Nat) : Option Nat :=
  if s.lo ≤ a && a < s.lo + winSize then (s.mem[a - s.lo]?).getD none else none

def St.wrByte (s : St) (a v : Nat) : Option St :=
  if s.lo ≤ a && a < s.lo + winSize then some { s with mem := s.mem.set (a - s.lo) (some v) }
  else none

def St.wrBytes (s : St) (a v : Nat) : Nat → Option St
  | 0 => some s
  | n + 1 => do
    let s ← s.wrByte a (v % 256)
    s.wrBytes (a + 1) (v / 256) n

/-- Value of register `i` at width `w`: the low `w` bits. -/
def St.rd (s : St) (w i : Nat) : Option Nat := (s.reg i).map (· % pow2 w)

/-- Write `v` (below 2^w) to register `i` at width `w`: width 8 keeps bits 8..63,
width 32 zero-extends, width 64 replaces. -/
def St.wr (s : St) (w i v : Nat) : Option St :=
  if w == 8 then do
    let old ← s.reg i
    pure (s.setReg i (some (old - old % 256 + v)))
  else some (s.setReg i (some v))

/-- Result and flags of an arithmetic or logical operation at width `w`. -/
def alu (op : Op) (w a b : Nat) : Nat × Flags :=
  let m := pow2 w
  let res := match op with
    | .add => (a + b) % m
    | .sub | .cmp => (a + m - b) % m
    | .xor => Nat.xor a b
    | .or => Nat.lor a b
    | .test => Nat.land a b
    | .mov => b
  let cf := match op with
    | .add => a + b ≥ m
    | .sub | .cmp => a < b
    | _ => false
  let ofl := match op with
    | .add => msb w a == msb w b && msb w res != msb w a
    | .sub | .cmp => msb w a != msb w b && msb w res != msb w a
    | _ => false
  (res, ⟨some cf, some (res == 0), some (msb w res), some ofl⟩)

/-- Condition code `cc` of jcc, from the flags; `none` when a flag it reads is
undefined, or for PF (not modelled). -/
def cond (f : Flags) (cc : Nat) : Option Bool := do
  let base ← match cc / 2 with
    | 0 => f.of
    | 1 => f.cf
    | 2 => f.zf
    | 3 => do let c ← f.cf; let z ← f.zf; pure (c || z)
    | 4 => f.sf
    | 6 => do let s ← f.sf; let o ← f.of; pure (s != o)
    | 7 => do let s ← f.sf; let o ← f.of; let z ← f.zf; pure (z || s != o)
    | _ => none
  pure (if cc % 2 == 0 then base else !base)

inductive Outcome where
  | exit (status : Nat) (out : List Nat)
  /-- the machine is stuck at `rip`; `why` is 0 undecodable, 1 undefined read or
  unmodelled access, 2 unmodelled syscall -/
  | stuck (rip why : Nat)
  | outOfFuel
deriving DecidableEq, Repr

/-- The result of an instruction whose effect may fail: the next state, or stuck
at `rip` with reason 1 (an undefined read or an unmodelled access). -/
def okOf (rip : Nat) (o : Option St) : Sum St Outcome :=
  match o with
  | some t => .inl t
  | none => .inr (.stuck rip 1)

/-- One step: either the next state or the end of the run. -/
def step (im : Img) (s : St) : Sum St Outcome :=
  match im.decodeAt s.rip with
  | none => .inr (.stuck s.rip 0)
  | some (ins, len) =>
    let nx := (s.rip + len) % pow2 64
    let s1 := { s with rip := nx }
    let stuck := Sum.inr (Outcome.stuck s.rip 1)
    let ok := okOf s.rip
    match ins with
    | .movImm d i => ok (s1.wr 32 d i)
    | .rr .mov w d src => ok (do let v ← s.rd w src; s1.wr w d v)
    | .rr op w d src => ok (do
        -- `xor r, r` is 0 whatever r held, so it does not read r
        let zeroing := op == .xor && d == src
        let a ← if zeroing then some 0 else s.rd w d
        let b ← if zeroing then some 0 else s.rd w src
        let (res, f) := alu op w a b
        let t := { s1 with fl := f }
        if op == .test then pure t else t.wr w d res)
    | .ri op w d i => ok (do
        let a ← s.rd w d
        let (res, f) := alu op w a i
        let t := { s1 with fl := f }
        if op == .cmp then pure t else t.wr w d res)
    | .shlI w d c => ok (do
        let a ← s.rd w d
        let c := c % w
        -- a count of 0 leaves the flags alone, but a 32-bit destination is still
        -- written, so bits 32..63 are cleared (found by the CPU differential)
        if c == 0 then s1.wr w d a else
        let res := a * pow2 c % pow2 w
        let cf := a / pow2 (w - c) % 2 == 1
        let ofl := if c == 1 then some (msb w res != cf) else none
        let t := { s1 with fl := ⟨some cf, some (res == 0), some (msb w res), ofl⟩ }
        t.wr w d res)
    | .inc w d => ok (do
        let a ← s.rd w d
        let res := (a + 1) % pow2 w
        let t := { s1 with fl := { s.fl with zf := some (res == 0), sf := some (msb w res),
                                             of := some (res == pow2 (w - 1)) } }
        t.wr w d res)
    | .push i => ok (do
        let sp ← s.reg 4
        let sp' := (sp + pow2 64 - 8) % pow2 64
        let t ← s1.wrBytes sp' i 8
        pure (t.setReg 4 (some sp')))
    | .pop d => ok (do
        let sp ← s.reg 4
        let v ← le s.rdByte sp 8
        let t := s1.setReg 4 (some ((sp + 8) % pow2 64))
        pure (t.setReg d (some v)))
    | .ldRsp8 d => ok (do
        let sp ← s.reg 4
        let v ← s.rdByte sp
        s1.wr 32 d v)
    | .stRsp8 src => ok (do
        let sp ← s.reg 4
        let v ← s.rd 8 src
        s1.wrByte sp v)
    | .jmp rel => .inl { s1 with rip := (nx + rel) % pow2 64 }
    | .jcc cc rel => match cond s.fl cc with
      | none => stuck
      | some true => .inl { s1 with rip := (nx + rel) % pow2 64 }
      | some false => .inl s1
    | .syscall =>
      -- the kernel returns in rax and clobbers rcx (return address) and r11 (rflags)
      let s2 := (s1.setReg 1 (some nx)).setReg 11 none
      match s.reg 0 with
      | some 0 => ok (do
          if s.reg 7 != some 0 || s.reg 2 != some 1 then none else
          let a ← s.reg 6
          match s.inp with
          | [] => pure (s2.setReg 0 (some 0))
          | c :: rest => do
            let t ← s2.wrByte a c
            pure ({ t with inp := rest }.setReg 0 (some 1)))
      | some 1 => ok (do
          if s.reg 7 != some 1 || s.reg 2 != some 1 then none else
          let a ← s.reg 6
          let c ← s.rdByte a
          pure ({ s2 with out := c :: s.out }.setReg 0 (some 1)))
      | some 60 => match s.reg 7 with
        | some st => .inr (.exit (st % 256) s.out.reverse)
        | none => stuck
      | _ => .inr (.stuck s.rip 2)

/-- Run for at most `fuel` steps. Written with `Nat.rec` rather than by pattern
matching: structural recursion compiles to `brecOn`, which the kernel reduces by
building a table of all earlier calls, and that is what a proof by `decide` pays
for at every step. -/
def run (im : Img) (fuel : Nat) : St → Outcome :=
  Nat.rec (motive := fun _ => St → Outcome) (fun _ => .outOfFuel)
    (fun _ ih s => match step im s with
      | .inl t => ih t
      | .inr o => o) fuel

/-- Run for at most `n` steps and return the state reached, or the outcome if
the run ended earlier. -/
def runS (im : Img) (n : Nat) : St → Sum St Outcome :=
  Nat.rec (motive := fun _ => St → Sum St Outcome) (fun s => .inl s)
    (fun _ ih s => match step im s with
      | .inl t => ih t
      | .inr o => .inr o) n

theorem run_succ (im : Img) (n : Nat) (s : St) :
    run im (n + 1) s = match step im s with
      | .inl t => run im n t
      | .inr o => o := rfl

theorem runS_succ (im : Img) (n : Nat) (s : St) :
    runS im (n + 1) s = match step im s with
      | .inl t => runS im n t
      | .inr o => .inr o := rfl

/-- A run of `a + b` steps is a run of `a` steps followed by a run of `b`. -/
theorem run_add (im : Img) :
    ∀ (a b : Nat) (s : St), run im (a + b) s = match runS im a s with
      | .inl t => run im b t
      | .inr o => o := by
  intro a
  induction a with
  | zero => intro b s; simp [runS]
  | succ a ih =>
    intro b s
    rw [Nat.add_right_comm, run_succ, runS_succ]
    cases step im s with
    | inl t => exact ih b t
    | inr o => rfl

/-- The step used to chain checkpoint lemmas. -/
theorem run_chain {im : Img} {a b : Nat} {s t : St} (h : runS im a s = .inl t) :
    run im (a + b) s = run im b t := by
  rw [run_add, h]

/-- Little-endian field of a file. -/
def field (f : List Nat) (off n : Nat) : Option Nat := le (fun i => f[i]?) off n

/-- Load an ELF64 executable the way the seeds are built: one PT_LOAD segment,
read and execute only, mapping the whole file at a page-aligned address. Returns
the entry state with `sp` as the stack pointer and `inp` as standard input. -/
def load (f : List Nat) (sp : Nat) (inp : List Nat) : Option (Img × St) := do
  let magic ← field f 0 4
  let ident ← field f 4 3
  let ty ← field f 16 2
  let mach ← field f 18 2
  let entry ← field f 24 8
  let phoff ← field f 32 8
  let phentsize ← field f 54 2
  let phnum ← field f 56 2
  let ptype ← field f phoff 4
  let pflags ← field f (phoff + 4) 4
  let poff ← field f (phoff + 8) 8
  let vaddr ← field f (phoff + 16) 8
  let filesz ← field f (phoff + 32) 8
  let memsz ← field f (phoff + 40) 8
  if magic != 0x464c457f || ident != 0x010102 || ty != 2 || mach != 0x3e
     || phentsize != 56 || phnum != 1 || ptype != 1 || pflags != 5 || poff != 0
     || filesz != f.length || memsz != filesz || vaddr % 4096 != 0
     || entry < vaddr || entry ≥ vaddr + filesz || filesz > 2 ^ tdepth then none else
  pure (Img.mk' f vaddr filesz,
        { regs := (List.replicate 16 none).set 4 (some sp), rip := entry, fl := .undef,
          mem := List.replicate winSize none, lo := sp - winSize / 2, inp := inp, out := [] })

/-- Run the executable `f` on input `inp` for at most `fuel` instructions. -/
def exec (fuel : Nat) (f : List Nat) (inp : List Nat) : Outcome :=
  match load f 0x7ffffffde000 inp with
  | none => .stuck 0 3
  | some (im, s) => run im fuel s

/-- `exec` from a proved `load`. -/
theorem exec_of_load {fuel : Nat} {f inp : List Nat} {im : Img} {s : St}
    (h : load f 0x7ffffffde000 inp = some (im, s)) : exec fuel f inp = run im fuel s := by
  simp [exec, h]

end X86Min
