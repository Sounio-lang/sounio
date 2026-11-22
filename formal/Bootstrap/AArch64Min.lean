/-
AArch64Min: a minimal 32-bit fixed-width AArch64 semantics of Linux user code,
written for the third bootstrap seed loader (`bootstrap/roots/c/seed-c`).

Scope:
  - only the minimal AArch64 instruction forms needed by `seed-c` and its
    hardware differential test are decoded; every other 32-bit word decodes
    to `none` and the machine is stuck (fail closed);
  - general-purpose registers `x0..x30`, `sp`, `NZCV` flags, and stack bytes are
    `Option`: at entry only `sp` is defined, and reading an undefined value
    leaves the machine stuck;
  - register `31` means `SP` or `XZR`/`WZR` per ARMv8-A encoding rules, and
    stack-pointer memory accesses enforce 16-byte `SP` alignment;
  - Linux AArch64 syscalls (`x8 = 63 read / 64 write / 93 exit`, `svc #0`);
  - includes `StLe` state refinement, `step_mono`, `runU`, `runU_mono`, `leB`
    decidable refinement check, and fast kernel-checked sanity witnesses.

Zero Mathlib imports, zero axioms beyond core Lean.
-/
namespace AArch64Min

def pow2 (w : Nat) : Nat := 2 ^ w

/-- Most significant bit of `v` at width `w`. -/
def msb (w v : Nat) : Bool := v / pow2 (w - 1) % 2 == 1

/-- Sign-extend a `bits`-wide value to 64 bits. -/
def sx (bits v : Nat) : Nat :=
  if v < pow2 (bits - 1) then v else v + pow2 64 - pow2 bits

/-- Extract `len` bits starting at bit `lo` from `w`. -/
def bf (w lo len : Nat) : Nat := (w / pow2 lo) % pow2 len

/-- Arithmetic shift right of `w`-bit `v` by `sh < w`. -/
def asrVal (w v sh : Nat) : Nat :=
  let v := v % pow2 w
  if sh == 0 then v
  else if msb w v then v / pow2 sh + (pow2 w - pow2 (w - sh))
  else v / pow2 sh

/-- Rotate right of `w`-bit `v` by `sh < w`. -/
def rorVal (w v sh : Nat) : Nat :=
  let v := v % pow2 w
  if sh == 0 then v
  else v / pow2 sh + (v % pow2 sh) * pow2 (w - sh)

structure Flags where
  n : Option Bool
  z : Option Bool
  c : Option Bool
  v : Option Bool
deriving DecidableEq, Repr

def Flags.undef : Flags := ⟨none, none, none, none⟩

abbrev Reg := Nat

inductive ShiftKind where
  | lsl | lsr | asr | ror
deriving DecidableEq, Repr

inductive LogicOp where
  | and | orr | eor | ands
deriving DecidableEq, Repr

inductive MemSize where
  | b1 | w4 | x8
deriving DecidableEq, Repr

def MemSize.bytes : MemSize → Nat
  | .b1 => 1
  | .w4 => 4
  | .x8 => 8

def MemSize.width : MemSize → Nat
  | .b1 => 32
  | .w4 => 32
  | .x8 => 64

inductive Ins where
  /-- Move wide immediate (`kind`: 0 = MOVN, 2 = MOVZ, 3 = MOVK). -/
  | movWide (kind w rd imm16 hw : Nat)
  /-- Add/subtract immediate (`rn = 31` is SP; `rd = 31` is SP if `!setFlags`, ZR if `setFlags`). -/
  | addSubImm (isSub setFlags : Bool) (w rd rn imm : Nat)
  /-- Add/subtract shifted register (`31` is ZR). -/
  | addSubReg (isSub setFlags : Bool) (w rd rn rm : Nat) (sh : ShiftKind) (amt : Nat)
  /-- Logical shifted register (`31` is ZR). -/
  | logicReg (op : LogicOp) (inv : Bool) (w rd rn rm : Nat) (sh : ShiftKind) (amt : Nat)
  /-- Load unsigned immediate offset (`rn = 31` is SP, `rt = 31` is ZR). -/
  | ldr (sz : MemSize) (rt rn off : Nat)
  /-- Store unsigned immediate offset (`rn = 31` is SP, `rt = 31` is ZR). -/
  | str (sz : MemSize) (rt rn off : Nat)
  /-- Unconditional branch (`link = true` for BL). `off` is 64-bit sign-extended byte offset. -/
  | b (link : Bool) (off : Nat)
  /-- Conditional branch (`cc` 0..15). `off` is 64-bit sign-extended byte offset. -/
  | bcond (cc off : Nat)
  /-- Compare and branch on zero/nonzero (`31` is ZR). -/
  | cbz (nz : Bool) (w rt off : Nat)
  /-- Return to address in `rn` (`31` is ZR). -/
  | ret (rn : Nat)
  /-- Supervisor call (`imm = 0` only). -/
  | svc (imm : Nat)
deriving DecidableEq, Repr

/-- Little-endian value of `n` bytes, each read through `b`. -/
def le (b : Nat → Option Nat) (off : Nat) : Nat → Option Nat
  | 0 => some 0
  | n + 1 => do
    let lo ← b off
    let hi ← le b (off + 1) n
    pure (lo + 256 * hi)

/-- Decode a single 32-bit AArch64 instruction word. -/
def decodeWord (w : Nat) : Option Ins :=
  let sf := bf w 31 1
  let wd := if sf == 1 then 64 else 32
  if bf w 26 5 == 0x05 then
    let imm26 := bf w 0 26
    some (.b (sf == 1) (sx 28 (imm26 * 4)))
  else if bf w 24 8 == 0x54 && bf w 4 1 == 0 then
    let imm19 := bf w 5 19
    let cc := bf w 0 4
    some (.bcond cc (sx 21 (imm19 * 4)))
  else if bf w 25 6 == 0x1a then
    let nz := bf w 24 1 == 1
    let imm19 := bf w 5 19
    let rt := bf w 0 5
    some (.cbz nz wd rt (sx 21 (imm19 * 4)))
  else if w == 0xd4000001 then
    some (.svc 0)
  else if bf w 10 22 == 0x3597c0 && bf w 0 5 == 0 then
    some (.ret (bf w 5 5))
  else if bf w 23 6 == 0x22 then
    let isSub := bf w 30 1 == 1
    let setFlags := bf w 29 1 == 1
    let sh := bf w 22 1
    let imm12 := bf w 10 12
    let rn := bf w 5 5
    let rd := bf w 0 5
    let imm := if sh == 1 then imm12 * 4096 else imm12
    some (.addSubImm isSub setFlags wd rd rn imm)
  else if bf w 23 6 == 0x25 then
    let opc := bf w 29 2
    let hw := bf w 21 2
    let imm16 := bf w 5 16
    let rd := bf w 0 5
    if opc == 1 || (sf == 0 && hw ≥ 2) then none
    else some (.movWide opc wd rd imm16 hw)
  else if bf w 24 5 == 0x0b && bf w 21 1 == 0 then
    let isSub := bf w 30 1 == 1
    let setFlags := bf w 29 1 == 1
    let shiftBits := bf w 22 2
    let rm := bf w 16 5
    let imm6 := bf w 10 6
    let rn := bf w 5 5
    let rd := bf w 0 5
    if shiftBits == 3 || (sf == 0 && imm6 ≥ 32) then none
    else
      let sh := match shiftBits with | 0 => .lsl | 1 => .lsr | _ => .asr
      some (.addSubReg isSub setFlags wd rd rn rm sh imm6)
  else if bf w 24 5 == 0x0a then
    let opc := bf w 29 2
    let shiftBits := bf w 22 2
    let inv := bf w 21 1 == 1
    let rm := bf w 16 5
    let imm6 := bf w 10 6
    let rn := bf w 5 5
    let rd := bf w 0 5
    if sf == 0 && imm6 ≥ 32 then none
    else
      let op := match opc with | 0 => .and | 1 => .orr | 2 => .eor | _ => .ands
      let sh := match shiftBits with | 0 => .lsl | 1 => .lsr | 2 => .asr | _ => .ror
      some (.logicReg op inv wd rd rn rm sh imm6)
  else if bf w 24 6 == 0x39 then
    let szBits := bf w 30 2
    let opc := bf w 22 2
    let imm12 := bf w 10 12
    let rn := bf w 5 5
    let rt := bf w 0 5
    let szOpt : Option MemSize := match szBits with
      | 0 => some .b1 | 2 => some .w4 | 3 => some .x8 | _ => none
    match szOpt with
    | none => none
    | some sz =>
      let off := imm12 * sz.bytes
      if opc == 0 then some (.str sz rt rn off)
      else if opc == 1 then some (.ldr sz rt rn off)
      else none
  else none

/-- Decode one fixed-width 32-bit AArch64 instruction at offset 0. -/
def decode (b : Nat → Option Nat) : Option (Ins × Nat) := do
  let w ← le b 0 4
  let ins ← decodeWord w
  pure (ins, 4)

/-- Size of the stack window, in bytes, centered on entry `sp`. -/
def winSize : Nat := 64

/-- Complete binary tree of depth `d` with a value at each of its `2^d` leaves. -/
inductive Tree (α : Type) where
  | leaf (v : α)
  | node (l r : Tree α)
deriving DecidableEq, Repr

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

/-- Word depth of the decode table: 256 words = 1024 bytes. -/
def wdepth : Nat := 8

structure Img where
  code : List Nat
  vbase : Nat
  len : Nat
  dec : Tree (Option Ins)
deriving DecidableEq

def fetchIn (code : List Nat) (len i : Nat) : Option Nat :=
  if i < len then code[i]? else none

def Img.mk' (code : List Nat) (vbase len : Nat) : Img :=
  ⟨code, vbase, len,
    Tree.build (fun wi => (le (fun j => fetchIn code len (4 * wi + j)) 0 4).bind decodeWord) wdepth 0⟩

/-- The instruction at 4-byte aligned address `a`, if `a` is in the image. -/
def Img.decodeAt (im : Img) (a : Nat) : Option (Ins × Nat) :=
  if im.vbase ≤ a && a < im.vbase + im.len && (a - im.vbase) % 4 == 0 then
    (Tree.get none wdepth im.dec ((a - im.vbase) / 4)).map (·, 4)
  else none

theorem Img.dec_spec (code : List Nat) (vbase len wi : Nat) (h : wi < 2 ^ wdepth) :
    Tree.get none wdepth (Img.mk' code vbase len).dec wi
      = (le (fun j => fetchIn code len (4 * wi + j)) 0 4).bind decodeWord := by
  simp only [Img.mk']
  rw [Tree.get_build none _ wdepth 0 wi h, Nat.zero_add]

structure St where
  regs : List (Option Nat)
  sp   : Option Nat
  pc   : Nat
  fl   : Flags
  mem  : List (Option Nat)
  lo   : Nat
  inp  : List Nat
  out  : List Nat
deriving DecidableEq, Repr

def St.reg (s : St) (i : Nat) : Option Nat := (s.regs[i]?).getD none

def St.setReg (s : St) (i : Nat) (v : Option Nat) : St := { s with regs := s.regs.set i v }

def St.rdByte (s : St) (a : Nat) : Option Nat :=
  if s.lo ≤ a && a < s.lo + winSize then (s.mem[a - s.lo]?).getD none else none

def St.wrByte (s : St) (a v : Nat) : Option St :=
  if s.lo ≤ a && a < s.lo + winSize then some { s with mem := s.mem.set (a - s.lo) (some (v % 256)) }
  else none

def St.wrBytes (s : St) (a v : Nat) : Nat → Option St
  | 0 => some s
  | n + 1 => do
    let s ← s.wrByte a (v % 256)
    s.wrBytes (a + 1) (v / 256) n

/-- Read register `i` (`0..30`, or `31 = XZR/WZR`) at width `w`. -/
def St.rdZR (s : St) (w i : Nat) : Option Nat :=
  if i == 31 then some 0 else (s.reg i).map (· % pow2 w)

/-- Write `v` to register `i` (`0..30`, or `31 = XZR/WZR` discarded) at width `w` (32 zero-extends). -/
def St.wrZR (s : St) (w i v : Nat) : St :=
  if i == 31 then s else s.setReg i (some (v % pow2 w))

/-- Read register `i` (`0..30`, or `31 = SP`) at width `w`. -/
def St.rdSP (s : St) (w i : Nat) : Option Nat :=
  if i == 31 then s.sp.map (· % pow2 w) else (s.reg i).map (· % pow2 w)

/-- Write `v` to register `i` (`0..30`, or `31 = SP`) at width `w`. -/
def St.wrSP (s : St) (w i v : Nat) : St :=
  if i == 31 then { s with sp := some (v % pow2 w) } else s.setReg i (some (v % pow2 w))

def addSubOp (isSub : Bool) (w a b : Nat) : Nat × Flags :=
  let m := pow2 w
  let res := if isSub then (a + m - b) % m else (a + b) % m
  let cf := if isSub then a ≥ b else a + b ≥ m
  let vf := if isSub then msb w a != msb w b && msb w res != msb w a
            else msb w a == msb w b && msb w res != msb w a
  (res, ⟨some (msb w res), some (res == 0), some cf, some vf⟩)

def shiftVal (sh : ShiftKind) (w v amt : Nat) : Nat :=
  let v := v % pow2 w
  match sh with
  | .lsl => (v * pow2 amt) % pow2 w
  | .lsr => v / pow2 amt
  | .asr => asrVal w v amt
  | .ror => rorVal w v amt

/-- Evaluate ARM64 condition code `cc` (0..15). `14` (`AL`) and `15` (`NV`) are always true. -/
def cond (f : Flags) (cc : Nat) : Option Bool :=
  if cc ≥ 14 then some true
  else do
    let base ← match cc / 2 with
      | 0 => f.z
      | 1 => f.c
      | 2 => f.n
      | 3 => f.v
      | 4 => do let c ← f.c; let z ← f.z; pure (c && !z)
      | 5 => do let n ← f.n; let v ← f.v; pure (n == v)
      | 6 => do let z ← f.z; let n ← f.n; let v ← f.v; pure (!z && n == v)
      | _ => none
    pure (if cc % 2 == 0 then base else !base)

inductive Outcome where
  | exit (status : Nat) (out : List Nat)
  /-- Stuck at `pc`; `why`: 0 undecodable/unaligned, 1 undefined read/unmodelled access,
  2 unmodelled syscall, 3 ELF load failure or EOF guard. -/
  | stuck (pc why : Nat)
  | outOfFuel
deriving DecidableEq, Repr

def okOf (pc : Nat) (o : Option St) : Sum St Outcome :=
  match o with
  | some t => .inl t
  | none => .inr (.stuck pc 1)

def step (im : Img) (s : St) : Sum St Outcome :=
  match im.decodeAt s.pc with
  | none => .inr (.stuck s.pc 0)
  | some (ins, len) =>
    let nx := (s.pc + len) % pow2 64
    let s1 := { s with pc := nx }
    let stuck := Sum.inr (Outcome.stuck s.pc 1)
    let ok := okOf s.pc
    match ins with
    | .movWide kind w rd imm16 hw =>
      let sh := hw * 16
      let part := imm16 * pow2 sh
      match kind with
      | 2 => .inl (s1.wrZR w rd part)
      | 0 => .inl (s1.wrZR w rd (pow2 w - 1 - part))
      | 3 => ok (do
          let old ← s.rdZR w rd
          let lo := old % pow2 sh
          let hi := old / pow2 (sh + 16)
          pure (s1.wrZR w rd (lo + part + hi * pow2 (sh + 16))))
      | _ => stuck
    | .addSubImm isSub setFlags w rd rn imm => ok (do
        let a ← s.rdSP w rn
        let (res, f) := addSubOp isSub w a (imm % pow2 w)
        if setFlags then pure ({ s1 with fl := f }.wrZR w rd res)
        else pure (s1.wrSP w rd res))
    | .addSubReg isSub setFlags w rd rn rm sh amt => ok (do
        let a ← s.rdZR w rn
        let bRaw ← s.rdZR w rm
        let b := shiftVal sh w bRaw amt
        let (res, f) := addSubOp isSub w a b
        let t := if setFlags then { s1 with fl := f } else s1
        pure (t.wrZR w rd res))
    | .logicReg op inv w rd rn rm sh amt => ok (do
        let a ← s.rdZR w rn
        let bRaw ← s.rdZR w rm
        let bSh := shiftVal sh w bRaw amt
        let b := if inv then pow2 w - 1 - bSh else bSh
        let res := match op with
          | .and | .ands => Nat.land a b
          | .orr => Nat.lor a b
          | .eor => Nat.xor a b
        if op == .ands then
          let f : Flags := ⟨some (msb w res), some (res == 0), some false, some false⟩
          pure ({ s1 with fl := f }.wrZR w rd res)
        else
          pure (s1.wrZR w rd res))
    | .ldr sz rt rn off => ok (do
        let base ← s.rdSP 64 rn
        if rn == 31 && base % 16 != 0 then none else
        let addr := (base + off) % pow2 64
        let v ← le s.rdByte addr sz.bytes
        pure (s1.wrZR sz.width rt v))
    | .str sz rt rn off => ok (do
        let base ← s.rdSP 64 rn
        if rn == 31 && base % 16 != 0 then none else
        let addr := (base + off) % pow2 64
        let v ← s.rdZR sz.width rt
        s1.wrBytes addr v sz.bytes)
    | .b link off =>
      let s2 := if link then s1.setReg 30 (some nx) else s1
      .inl { s2 with pc := (s.pc + off) % pow2 64 }
    | .bcond cc off =>
      match cond s.fl cc with
      | none => stuck
      | some true => .inl { s1 with pc := (s.pc + off) % pow2 64 }
      | some false => .inl s1
    | .cbz nz w rt off => ok (do
        let v ← s.rdZR w rt
        let taken := if nz then v != 0 else v == 0
        if taken then pure { s1 with pc := (s.pc + off) % pow2 64 } else pure s1)
    | .ret rn => ok (do
        let tgt ← s.rdZR 64 rn
        pure { s1 with pc := tgt })
    | .svc imm =>
      if imm != 0 then .inr (.stuck s.pc 2)
      else match s.reg 8 with
      | some 63 => ok (do
          if s.reg 0 != some 0 || s.reg 2 != some 1 then none else
          let a ← s.reg 1
          match s.inp with
          | [] => pure (s1.setReg 0 (some 0))
          | c :: rest => do
            let t ← s1.wrByte a c
            pure ({ t with inp := rest }.setReg 0 (some 1)))
      | some 64 => ok (do
          if s.reg 0 != some 1 || s.reg 2 != some 1 then none else
          let a ← s.reg 1
          let c ← s.rdByte a
          pure ({ s1 with out := c :: s.out }.setReg 0 (some 1)))
      | some 93 => match s.reg 0 with
        | some st => .inr (.exit (st % 256) s.out.reverse)
        | none => stuck
      | _ => .inr (.stuck s.pc 2)

def run (im : Img) (fuel : Nat) : St → Outcome :=
  Nat.rec (motive := fun _ => St → Outcome) (fun _ => .outOfFuel)
    (fun _ ih s => match step im s with
      | .inl t => ih t
      | .inr o => o) fuel

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

theorem run_chain {im : Img} {a b : Nat} {s t : St} (h : runS im a s = .inl t) :
    run im (a + b) s = run im b t := by
  rw [run_add, h]

def field (f : List Nat) (off n : Nat) : Option Nat := le (fun i => f[i]?) off n

/-- Load a static `EM_AARCH64` (`e_machine = 183`) ELF64 executable with one `PT_LOAD`
segment mapped at a page-aligned address. -/
def load (f : List Nat) (spVal : Nat) (inp : List Nat) : Option (Img × St) := do
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
  if magic != 0x464c457f || ident != 0x010102 || ty != 2 || mach != 0xb7
     || phentsize != 56 || phnum != 1 || ptype != 1 || pflags != 5 || poff != 0
     || filesz != f.length || memsz != filesz || vaddr % 4096 != 0
     || entry < vaddr || entry ≥ vaddr + filesz || entry % 4 != 0
     || filesz > 4 * 2 ^ wdepth then none else
  pure (Img.mk' f vaddr filesz,
        { regs := List.replicate 31 none, sp := some spVal, pc := entry, fl := .undef,
          mem := List.replicate winSize none, lo := spVal - winSize / 2, inp := inp, out := [] })

def exec (fuel : Nat) (f : List Nat) (inp : List Nat) : Outcome :=
  match load f 0x7ffffffde000 inp with
  | none => .stuck 0 3
  | some (im, s) => run im fuel s

theorem exec_of_load {fuel : Nat} {f inp : List Nat} {im : Img} {s : St}
    (h : load f 0x7ffffffde000 inp = some (im, s)) : exec fuel f inp = run im fuel s := by
  simp [exec, h]

/-! ## State refinement and monotonicity -/

section Refine

variable {r o : List Nat}

def OLe {α : Type} (a b : Option α) : Prop := ∀ v, a = some v → b = some v

theorem OLe.refl {α : Type} (a : Option α) : OLe a a := fun _ h => h
theorem OLe.none {α : Type} (b : Option α) : OLe none b := fun _ h => by cases h
theorem OLe.trans {α : Type} {x y z : Option α} (h1 : OLe x y) (h2 : OLe y z) : OLe x z :=
  fun v hv => h2 v (h1 v hv)

structure FlLe (f g : Flags) : Prop where
  n : OLe f.n g.n
  z : OLe f.z g.z
  c : OLe f.c g.c
  v : OLe f.v g.v

theorem FlLe.refl (f : Flags) : FlLe f f := ⟨OLe.refl _, OLe.refl _, OLe.refl _, OLe.refl _⟩

structure StLe (r o : List Nat) (a b : St) : Prop where
  reg : ∀ i, OLe (a.reg i) (b.reg i)
  rlen : a.regs.length = b.regs.length
  sp : OLe a.sp b.sp
  pc : a.pc = b.pc
  fl : FlLe a.fl b.fl
  byte : ∀ x, OLe (a.rdByte x) (b.rdByte x)
  mlen : a.mem.length = b.mem.length
  lo : a.lo = b.lo
  inp : b.inp = a.inp ++ r
  out : b.out = a.out ++ o

theorem OLe_bind {α β : Type} {ox oy : Option α} {f g : α → Option β}
    (hxy : OLe ox oy) (hfg : ∀ v, OLe (f v) (g v)) : OLe (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v => rw [hx] at ht; rw [hxy v hx]; exact hfg v t ht

theorem rdZR_le {a b : St} (h : StLe r o a b) (w i : Nat) : OLe (a.rdZR w i) (b.rdZR w i) := by
  intro v hv
  simp only [St.rdZR] at hv ⊢
  by_cases hi : i == 31
  · simp only [hi, if_true] at hv ⊢; exact hv
  · simp only [hi, Bool.false_eq_true, if_false] at hv ⊢
    cases ha : a.reg i with
    | none => rw [ha] at hv; cases hv
    | some x => rw [ha] at hv; rw [h.reg i x ha]; exact hv

theorem rdSP_le {a b : St} (h : StLe r o a b) (w i : Nat) : OLe (a.rdSP w i) (b.rdSP w i) := by
  intro v hv
  simp only [St.rdSP] at hv ⊢
  by_cases hi : i == 31
  · simp only [hi, if_true] at hv ⊢
    cases ha : a.sp with
    | none => rw [ha] at hv; cases hv
    | some x => rw [ha] at hv; rw [h.sp x ha]; exact hv
  · simp only [hi, Bool.false_eq_true, if_false] at hv ⊢
    cases ha : a.reg i with
    | none => rw [ha] at hv; cases hv
    | some x => rw [ha] at hv; rw [h.reg i x ha]; exact hv

theorem le_mono {f g : Nat → Option Nat} (hfg : ∀ x, OLe (f x) (g x)) :
    ∀ (n off : Nat), OLe (le f off n) (le g off n) := by
  intro n; induction n with
  | zero => intro off v hv; simpa [le] using hv
  | succ n ih =>
    intro off v hv
    simp only [le] at hv ⊢
    cases hf : f off with
    | none => rw [hf] at hv; cases hv
    | some lo =>
      rw [hf] at hv; rw [hfg off lo hf]
      cases hl : le f (off + 1) n with
      | none => rw [hl] at hv; cases hv
      | some hi => rw [hl] at hv; rw [ih (off + 1) hi hl]; exact hv

theorem getD_set_reg (l : List (Option Nat)) (i j : Nat) (v : Option Nat) :
    ((l.set i v)[j]?).getD none = if i = j ∧ i < l.length then v else (l[j]?).getD none := by
  rw [List.getElem?_set]
  by_cases hij : i = j
  · subst hij
    by_cases hl : i < l.length
    · simp [hl]
    · simp [hl]
  · simp [hij]

theorem setReg_le {a b : St} (h : StLe r o a b) (i : Nat) (v : Option Nat) :
    StLe r o (a.setReg i v) (b.setReg i v) where
  reg j := by
    intro x hx
    simp only [St.reg, St.setReg, getD_set_reg] at hx ⊢
    rw [← h.rlen]
    by_cases hc : i = j ∧ i < a.regs.length
    · rw [if_pos hc] at hx ⊢; exact hx
    · rw [if_neg hc] at hx ⊢; exact h.reg j x hx
  rlen := by simp [St.setReg, h.rlen]
  sp := h.sp
  pc := h.pc
  fl := h.fl
  byte := h.byte
  mlen := h.mlen
  lo := h.lo
  inp := h.inp
  out := h.out

theorem wrZR_le {a b : St} (h : StLe r o a b) (w i v : Nat) :
    StLe r o (a.wrZR w i v) (b.wrZR w i v) := by
  simp only [St.wrZR]
  split
  · exact h
  · exact setReg_le h i (some (v % pow2 w))

theorem wrSP_le {a b : St} (h : StLe r o a b) (w i v : Nat) :
    StLe r o (a.wrSP w i v) (b.wrSP w i v) := by
  simp only [St.wrSP]
  split
  · exact ⟨h.reg, h.rlen, OLe.refl _, h.pc, h.fl, h.byte, h.mlen, h.lo, h.inp, h.out⟩
  · exact setReg_le h i (some (v % pow2 w))

def SLe (r o : List Nat) (oa ob : Option St) : Prop :=
  ∀ t, oa = some t → ∃ t', ob = some t' ∧ StLe r o t t'

theorem SLe.some {a b : St} (h : StLe r o a b) : SLe r o (some a) (some b) := by
  intro t ht; cases ht; exact ⟨b, rfl, h⟩

theorem SLe_bind {α : Type} {ox oy : Option α} {f g : α → Option St}
    (hxy : OLe ox oy) (hfg : ∀ v, SLe r o (f v) (g v)) : SLe r o (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v => rw [hx] at ht; rw [hxy v hx]; exact hfg v t ht

theorem SLe_none (ob : Option St) : SLe r o none ob := fun _ h => by cases h

theorem StLe.with_pc {a b : St} (h : StLe r o a b) (x : Nat) :
    StLe r o { a with pc := x } { b with pc := x } :=
  ⟨h.reg, h.rlen, h.sp, rfl, h.fl, h.byte, h.mlen, h.lo, h.inp, h.out⟩

theorem StLe.with_fl {a b : St} (h : StLe r o a b) (f : Flags) :
    StLe r o { a with fl := f } { b with fl := f } :=
  ⟨h.reg, h.rlen, h.sp, h.pc, FlLe.refl f, h.byte, h.mlen, h.lo, h.inp, h.out⟩

theorem StLe.with_inp {a b : St} (h : StLe r o a b) (l : List Nat) :
    StLe r o { a with inp := l } { b with inp := l ++ r } :=
  ⟨h.reg, h.rlen, h.sp, h.pc, h.fl, h.byte, h.mlen, h.lo, rfl, h.out⟩

theorem StLe.with_out {a b : St} (h : StLe r o a b) (l : List Nat) :
    StLe r o { a with out := l } { b with out := l ++ o } :=
  ⟨h.reg, h.rlen, h.sp, h.pc, h.fl, h.byte, h.mlen, h.lo, h.inp, rfl⟩

theorem wrByte_le {a b : St} (h : StLe r o a b) (x v : Nat) : SLe r o (a.wrByte x v) (b.wrByte x v) := by
  intro t ht
  simp only [St.wrByte, ← h.lo] at ht ⊢
  by_cases hw : (a.lo ≤ x && x < a.lo + winSize) = true
  · rw [if_pos hw] at ht ⊢; cases ht
    refine ⟨_, rfl, ⟨h.reg, h.rlen, h.sp, h.pc, h.fl, ?_, by simp [h.mlen], rfl, h.inp, h.out⟩⟩
    intro y u hu
    simp only [St.rdByte] at hu ⊢
    by_cases hy : (a.lo ≤ y && y < a.lo + winSize) = true
    · rw [if_pos hy] at hu ⊢
      rw [getD_set_reg] at hu ⊢
      rw [← h.mlen]
      by_cases hc : x - a.lo = y - a.lo ∧ x - a.lo < a.mem.length
      · rw [if_pos hc] at hu ⊢; exact hu
      · rw [if_neg hc] at hu ⊢
        have := h.byte y u
        simp only [St.rdByte, ← h.lo, if_pos hy] at this
        exact this hu
    · rw [if_neg hy] at hu; cases hu
  · rw [if_neg hw] at ht; cases ht

theorem wrBytes_le : ∀ (n : Nat) {a b : St} (_ : StLe r o a b) (x v : Nat),
    SLe r o (a.wrBytes x v n) (b.wrBytes x v n) := by
  intro n; induction n with
  | zero => intro a b h x v; exact SLe.some h
  | succ n ih =>
    intro a b h x v t ht
    simp only [St.wrBytes] at ht ⊢
    cases ha : a.wrByte x (v % 256) with
    | none => rw [ha] at ht; cases ht
    | some a1 =>
      rw [ha] at ht
      obtain ⟨b1, hb1, h1⟩ := wrByte_le h x (v % 256) a1 ha
      rw [hb1]; exact ih h1 (x + 1) (v / 256) t ht

def StepLe (r o : List Nat) (ra rb : Sum St Outcome) : Prop :=
  (∀ t, ra = .inl t → ∃ t', rb = .inl t' ∧ StLe r o t t') ∧
  (∀ st x, ra = .inr (.exit st x) → rb = .inr (.exit st (o.reverse ++ x)))

theorem StepLe.okOf {oa ob : Option St} (p : Nat) (h : SLe r o oa ob) :
    StepLe r o (okOf p oa) (okOf p ob) := by
  constructor
  · intro t ht
    cases ha : oa with
    | none => rw [ha] at ht; cases ht
    | some ta =>
      rw [ha] at ht
      obtain ⟨t', hb, ht'⟩ := h ta ha
      have e : ta = t := by injection ht
      subst e
      exact ⟨t', by rw [hb]; rfl, ht'⟩
  · intro st x hx
    cases ha : oa with
    | none => rw [ha] at hx; cases hx
    | some ta => rw [ha] at hx; cases hx

theorem StepLe.inl {a b : St} (h : StLe r o a b) : StepLe r o (.inl a) (.inl b) := by
  constructor
  · intro t ht; cases ht; exact ⟨b, rfl, h⟩
  · intro _ _ hx; cases hx

theorem StepLe.stuck (p w : Nat) (rb : Sum St Outcome) : StepLe r o (.inr (.stuck p w)) rb := by
  constructor
  · intro _ ht; cases ht
  · intro _ _ hx; cases hx

theorem StepLe.exit (st : Nat) (x : List Nat) :
    StepLe r o (.inr (.exit st x.reverse)) (.inr (.exit st (x ++ o).reverse)) := by
  constructor
  · intro _ ht; cases ht
  · intro st' x' hx; cases hx; rw [List.reverse_append]

theorem cond_le {f g : Flags} (h : FlLe f g) (cc : Nat) : OLe (cond f cc) (cond g cc) := by
  unfold cond
  split
  · exact OLe.refl _
  · generalize (fun base => (pure (if (cc % 2 == 0) = true then base else !base) : Option Bool)) = jp
    generalize cc / 2 = k
    match k with
    | 0 => exact OLe_bind h.z fun _ => OLe.refl _
    | 1 => exact OLe_bind h.c fun _ => OLe.refl _
    | 2 => exact OLe_bind h.n fun _ => OLe.refl _
    | 3 => exact OLe_bind h.v fun _ => OLe.refl _
    | 4 => exact OLe_bind h.c fun _ => OLe_bind h.z fun _ => OLe.refl _
    | 5 => exact OLe_bind h.n fun _ => OLe_bind h.v fun _ => OLe.refl _
    | 6 => exact OLe_bind h.z fun _ => OLe_bind h.n fun _ => OLe_bind h.v fun _ => OLe.refl _
    | _ + 7 => exact OLe.none _

theorem SLe_bindS {ox oy : Option St} {f g : St → Option St}
    (hxy : SLe r o ox oy) (hfg : ∀ t t', StLe r o t t' → SLe r o (f t) (g t')) :
    SLe r o (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v =>
    rw [hx] at ht
    obtain ⟨v', hv', hvv⟩ := hxy v hx
    rw [hv']; exact hfg v v' hvv t ht

theorem guard_le {p p' q q' : Option Nat} (hp : OLe p p') (hq : OLe q q') (x y : Nat)
    {k k' : Option St} (hk : SLe r o k k') :
    SLe r o (if (p != some x || q != some y) = true then none else k)
        (if (p' != some x || q' != some y) = true then none else k') := by
  intro t ht
  by_cases hpx : p = some x
  · by_cases hqy : q = some y
    · rw [hp x hpx, hq y hqy]; rw [hpx, hqy] at ht; simp at ht ⊢; exact hk t ht
    · simp [hqy] at ht
  · simp [hpx] at ht

def readsEOF (im : Img) (s : St) : Bool :=
  s.inp.isEmpty && s.reg 8 == some 63 &&
    match im.decodeAt s.pc with
    | some (.svc 0, _) => true
    | _ => false

theorem step_mono (im : Img) {a b : St} (h : StLe r o a b) (hg : r = [] ∨ readsEOF im a = false) :
    StepLe r o (step im a) (step im b) := by
  obtain ⟨ra, spa, pca, fa, ma, la, ia, oa⟩ := a
  obtain ⟨rb, spb, pcb, fb, mb, lb, ib, ob⟩ := b
  have e1 : pca = pcb := h.pc
  have e2 : la = lb := h.lo
  have e3 : ib = ia ++ r := h.inp
  have e4 : ob = oa ++ o := h.out
  subst e1 e2 e3 e4
  unfold step
  cases hd : im.decodeAt pca with
  | none => exact StepLe.stuck _ _ _
  | some p =>
    obtain ⟨ins, len⟩ := p
    have h1 := h.with_pc ((pca + len) % pow2 64)
    cases ins with
    | movWide kind w rd imm16 hw =>
      dsimp only
      match kind with
      | 2 => exact StepLe.inl (wrZR_le h1 w rd _)
      | 0 => exact StepLe.inl (wrZR_le h1 w rd _)
      | 3 => exact StepLe.okOf _ (SLe_bind (rdZR_le h w rd) fun old => SLe.some (wrZR_le h1 w rd _))
      | 1 | _ + 4 => exact StepLe.stuck _ _ _
    | addSubImm isSub setFlags w rd rn imm =>
      apply StepLe.okOf
      refine SLe_bind (rdSP_le h w rn) fun a => ?_
      dsimp only
      split
      · exact SLe.some (wrZR_le (h1.with_fl _) w rd _)
      · exact SLe.some (wrSP_le h1 w rd _)
    | addSubReg isSub setFlags w rd rn rm sh amt =>
      apply StepLe.okOf
      refine SLe_bind (rdZR_le h w rn) fun a => SLe_bind (rdZR_le h w rm) fun b => ?_
      dsimp only
      split
      · exact SLe.some (wrZR_le (h1.with_fl _) w rd _)
      · exact SLe.some (wrZR_le h1 w rd _)
    | logicReg op inv w rd rn rm sh amt =>
      apply StepLe.okOf
      refine SLe_bind (rdZR_le h w rn) fun a => SLe_bind (rdZR_le h w rm) fun b => ?_
      dsimp only
      split
      · exact SLe.some (wrZR_le (h1.with_fl _) w rd _)
      · exact SLe.some (wrZR_le h1 w rd _)
    | ldr sz rt rn off =>
      apply StepLe.okOf
      refine SLe_bind (rdSP_le h 64 rn) fun base => ?_
      dsimp only
      split
      · exact SLe_none _
      · exact SLe_bind (le_mono h.byte sz.bytes ((base + off) % pow2 64)) fun v =>
          SLe.some (wrZR_le h1 sz.width rt v)
    | str sz rt rn off =>
      apply StepLe.okOf
      refine SLe_bind (rdSP_le h 64 rn) fun base => ?_
      dsimp only
      split
      · exact SLe_none _
      · exact SLe_bind (rdZR_le h sz.width rt) fun v => wrBytes_le sz.bytes h1 _ v
    | b link off =>
      dsimp only
      split
      · exact StepLe.inl ((setReg_le h1 30 _).with_pc _)
      · exact StepLe.inl (h1.with_pc _)
    | bcond cc off =>
      dsimp only
      cases hc : cond fa cc with
      | none => exact StepLe.stuck _ _ _
      | some v =>
        rw [cond_le h.fl cc v hc]
        cases v
        · exact StepLe.inl h1
        · exact StepLe.inl (h.with_pc _)
    | cbz nz w rt off =>
      apply StepLe.okOf
      refine SLe_bind (rdZR_le h w rt) fun v => ?_
      dsimp only
      by_cases htk : (if nz then v != 0 else v == 0) = true
      · simp only [htk, if_true]; exact SLe.some (h.with_pc _)
      · simp [htk]; exact SLe.some h1
    | ret rn =>
      exact StepLe.okOf _ (SLe_bind (rdZR_le h 64 rn) fun tgt => SLe.some (h.with_pc tgt))
    | svc imm =>
      dsimp only
      split
      · exact StepLe.stuck _ _ _
      · rename_i himm
        have himm0 : imm = 0 := by simpa using himm
        subst himm0
        cases hr : St.reg ⟨ra, spa, pca, fa, ma, la, ia, oa⟩ 8 with
        | none => exact StepLe.stuck _ _ _
        | some v =>
          rw [h.reg 8 v hr]
          split
          · rename_i hv; cases hv
            apply StepLe.okOf
            refine guard_le (h.reg 0) (h.reg 2) 0 1 ?_
            refine SLe_bind (h.reg 1) fun ad => ?_
            cases ia with
            | nil =>
              have hr0 : r = [] := by
                rcases hg with hg | hg
                · exact hg
                · simp [readsEOF, hd, hr] at hg
              subst hr0
              exact SLe.some (setReg_le h1 _ _)
            | cons c rest =>
              exact SLe_bindS (wrByte_le h1 _ _) fun t t' ht => SLe.some (setReg_le (ht.with_inp rest) _ _)
          · apply StepLe.okOf
            refine guard_le (h.reg 0) (h.reg 2) 1 1 ?_
            exact SLe_bind (h.reg 1) fun ad => SLe_bind (h.byte ad) fun c =>
              SLe.some (setReg_le (h1.with_out _) _ _)
          · cases h0 : St.reg ⟨ra, spa, pca, fa, ma, la, ia, oa⟩ 0 with
            | none => exact StepLe.stuck _ _ _
            | some st =>
              rw [h.reg 0 st h0]
              exact StepLe.exit _ _
          · exact StepLe.stuck _ _ _

def stepG (g : Bool) (im : Img) (s : St) : Sum St Outcome :=
  if (g && readsEOF im s) = true then .inr (.stuck s.pc 3) else step im s

def runU (g : Bool) (im : Img) (head : Nat) (n : Nat) : St → Sum St Outcome :=
  Nat.rec (motive := fun _ => St → Sum St Outcome) (fun _ => .inr .outOfFuel)
    (fun _ ih s => match stepG g im s with
      | .inl t => if t.pc == head then .inl t else ih t
      | .inr o => .inr o) n

theorem runU_succ (g : Bool) (im : Img) (head n : Nat) (s : St) :
    runU g im head (n + 1) s = match stepG g im s with
      | .inl t => if t.pc == head then .inl t else runU g im head n t
      | .inr o => .inr o := rfl

theorem step_svc_none (im : Img) (s : St) (len : Nat)
    (hd : im.decodeAt s.pc = some (.svc 0, len)) (h8 : s.reg 8 = none) :
    ∃ w, step im s = .inr (.stuck s.pc w) := by
  unfold step; rw [hd]; dsimp only; rw [h8]; exact ⟨2, rfl⟩

theorem stepG_mono (g : Bool) (im : Img) {a b : St} (h : StLe r o a b) (hg : r = [] ∨ g = true) :
    StepLe r o (stepG g im a) (stepG g im b) := by
  unfold stepG
  by_cases ha : (g && readsEOF im a) = true
  · rw [if_pos ha]; exact StepLe.stuck _ _ _
  · rw [if_neg ha]
    have hga : r = [] ∨ readsEOF im a = false := by
      rcases hg with hg | hg
      · exact .inl hg
      · subst hg; simp at ha; exact .inr ha
    by_cases hb : (g && readsEOF im b) = true
    · rw [if_pos hb]
      simp only [Bool.and_eq_true, readsEOF] at hb
      obtain ⟨hgt, ⟨hbi, hb8⟩, hbd⟩ := hb
      have hai : a.inp = [] := by
        have := h.inp; simp only [List.isEmpty_iff] at hbi; rw [hbi] at this
        exact (List.append_eq_nil_iff.mp this.symm).1
      have hpc : b.pc = a.pc := h.pc.symm
      rw [hpc] at hbd
      split at hbd
      · rename_i len hdd
        have ha8 : a.reg 8 = none := by
          cases e : a.reg 8 with
          | none => rfl
          | some v =>
            have hv := h.reg 8 v e
            simp only [beq_iff_eq] at hb8; rw [hb8] at hv
            cases hv
            exfalso; apply ha; subst hgt
            simp [readsEOF, hai, e, hdd]
        obtain ⟨w, hw⟩ := step_svc_none im a len hdd ha8
        rw [hw]; exact StepLe.stuck _ _ _
      · cases hbd
    · rw [if_neg hb]; exact step_mono im h hga

theorem StepLe.fuel (rb : Sum St Outcome) : StepLe r o (.inr .outOfFuel) rb := by
  constructor
  · intro _ e; cases e
  · intro _ _ e; cases e

theorem StepLe.exit' (st : Nat) (y : List Nat) :
    StepLe r o (.inr (.exit st y)) (.inr (.exit st (o.reverse ++ y))) := by
  constructor
  · intro _ e; cases e
  · intro _ _ e; cases e; rfl

theorem runU_mono (g : Bool) (im : Img) (head : Nat) (hg : r = [] ∨ g = true) :
    ∀ (n : Nat) {a b : St}, StLe r o a b → StepLe r o (runU g im head n a) (runU g im head n b) := by
  intro n; induction n with
  | zero => intro a b _; exact StepLe.fuel _
  | succ n ih =>
    intro a b h
    rw [runU_succ, runU_succ]
    have hs := stepG_mono g im h hg
    cases ea : stepG g im a with
    | inl t =>
      obtain ⟨t', et', ht⟩ := hs.1 t ea
      rw [et']; dsimp only
      rw [← ht.pc]
      by_cases hh : (t.pc == head) = true
      · rw [if_pos hh, if_pos hh]; exact StepLe.inl ht
      · rw [if_neg hh, if_neg hh]; exact ih ht
    | inr x =>
      dsimp only
      cases x with
      | exit st y => rw [hs.2 st y ea]; exact StepLe.exit' _ _
      | stuck p w => exact StepLe.stuck _ _ _
      | outOfFuel => exact StepLe.fuel _

theorem stepG_inl {g : Bool} {im : Img} {s t : St} (h : stepG g im s = .inl t) : step im s = .inl t := by
  unfold stepG at h; split at h
  · cases h
  · exact h

theorem stepG_exit {g : Bool} {im : Img} {s : St} {st : Nat} {x : List Nat}
    (h : stepG g im s = .inr (.exit st x)) : step im s = .inr (.exit st x) := by
  unfold stepG at h; split at h
  · cases h
  · exact h

theorem runU_run {g : Bool} {im : Img} {head : Nat} :
    ∀ (n : Nat) {s t : St}, runU g im head n s = .inl t → ∃ k, ∀ F, run im (k + F) s = run im F t := by
  intro n; induction n with
  | zero => intro s t e; cases e
  | succ n ih =>
    intro s t e
    rw [runU_succ] at e
    cases es : stepG g im s with
    | inl u =>
      rw [es] at e; dsimp only at e
      have hs := stepG_inl es
      by_cases hh : (u.pc == head) = true
      · rw [if_pos hh] at e; cases e
        exact ⟨1, fun F => by rw [Nat.add_comm, run_succ, hs]⟩
      · rw [if_neg hh] at e
        obtain ⟨k, hk⟩ := ih e
        exact ⟨k + 1, fun F => by rw [Nat.add_right_comm, run_succ, hs]; exact hk F⟩
    | inr x => rw [es] at e; cases e

theorem runU_exit {g : Bool} {im : Img} {head : Nat} :
    ∀ (n : Nat) {s : St} {st : Nat} {x : List Nat}, runU g im head n s = .inr (.exit st x) →
      ∃ k, run im k s = .exit st x := by
  intro n; induction n with
  | zero => intro s st x e; cases e
  | succ n ih =>
    intro s st x e
    rw [runU_succ] at e
    cases es : stepG g im s with
    | inl u =>
      rw [es] at e; dsimp only at e
      have hs := stepG_inl es
      by_cases hh : (u.pc == head) = true
      · rw [if_pos hh] at e; cases e
      · rw [if_neg hh] at e
        obtain ⟨k, hk⟩ := ih e
        exact ⟨k + 1, by rw [run_succ, hs]; exact hk⟩
    | inr y =>
      rw [es] at e; dsimp only at e; cases e
      exact ⟨1, by rw [run_succ, stepG_exit es]⟩

theorem run_fuel {im : Img} : ∀ (F m : Nat) {s : St} {st : Nat} {x : List Nat},
    run im F s = .exit st x → run im (F + m) s = .exit st x := by
  intro F; induction F with
  | zero => intro m s st x e; cases e
  | succ F ih =>
    intro m s st x e
    rw [Nat.add_right_comm, run_succ]
    rw [run_succ] at e
    cases es : step im s with
    | inl t => rw [es] at e; exact ih m e
    | inr y => rw [es] at e; exact e

theorem StLe.trans {r1 o1 r2 o2 : List Nat} {a b c : St} (h1 : StLe r1 o1 a b) (h2 : StLe r2 o2 b c) :
    StLe (r1 ++ r2) (o1 ++ o2) a c :=
  ⟨fun i => (h1.reg i).trans (h2.reg i), h1.rlen.trans h2.rlen, h1.sp.trans h2.sp, h1.pc.trans h2.pc,
   ⟨h1.fl.n.trans h2.fl.n, h1.fl.z.trans h2.fl.z, h1.fl.c.trans h2.fl.c, h1.fl.v.trans h2.fl.v⟩,
   fun x => (h1.byte x).trans (h2.byte x), h1.mlen.trans h2.mlen, h1.lo.trans h2.lo,
   by rw [h2.inp, h1.inp, List.append_assoc], by rw [h2.out, h1.out, List.append_assoc]⟩

theorem StLe.shift_inp {x : List Nat} {a b : St} (h : StLe (x ++ r) o a b) :
    StLe r o { a with inp := a.inp ++ x } b :=
  ⟨h.reg, h.rlen, h.sp, h.pc, h.fl, h.byte, h.mlen, h.lo, by rw [h.inp, List.append_assoc], h.out⟩

theorem StLe.app_self (a : St) : StLe r o a { a with inp := a.inp ++ r, out := a.out ++ o } :=
  ⟨fun _ => OLe.refl _, rfl, OLe.refl _, rfl, FlLe.refl _, fun _ => OLe.refl _, rfl, rfl, rfl, rfl⟩

theorem run_det {im : Img} {s : St} {F G a b : Nat} {x y : List Nat}
    (h1 : run im F s = .exit a x) (h2 : run im G s = .exit b y) : a = b ∧ x = y := by
  have e1 := run_fuel F G h1
  have e2 := run_fuel G F h2
  rw [Nat.add_comm, e2] at e1
  cases e1; exact ⟨rfl, rfl⟩

theorem load_inp (f : List Nat) (spVal : Nat) (L : List Nat) :
    load f spVal L = (load f spVal []).map (fun q => (q.1, { q.2 with inp := L })) := by
  unfold load
  simp only [Option.bind_eq_bind, Option.map_bind, apply_ite (Option.map _), Option.map_none, pure,
    Option.map_some, Function.comp_def]

def oleB {α : Type} [BEq α] : Option α → Option α → Bool
  | none, _ => true
  | some x, y => y == some x

def lleB : List (Option Nat) → List (Option Nat) → Bool
  | [], [] => true
  | x :: l, y :: m => oleB x y && lleB l m
  | _, _ => false

def leB (r o : List Nat) (a b : St) : Bool :=
  lleB a.regs b.regs && oleB a.sp b.sp && a.pc == b.pc &&
  oleB a.fl.n b.fl.n && oleB a.fl.z b.fl.z && oleB a.fl.c b.fl.c && oleB a.fl.v b.fl.v &&
  lleB a.mem b.mem && a.lo == b.lo && b.inp == a.inp ++ r && b.out == a.out ++ o

theorem oleB_sound {α : Type} [BEq α] [LawfulBEq α] {x y : Option α} (h : oleB x y = true) : OLe x y := by
  intro v hv; subst hv; simp only [oleB, beq_iff_eq] at h; exact h

theorem lleB_sound : ∀ {l m : List (Option Nat)}, lleB l m = true →
    l.length = m.length ∧ ∀ i : Nat, OLe ((l[i]?).getD none) ((m[i]?).getD none)
  | [], [], _ => ⟨rfl, fun _ => OLe.refl _⟩
  | x :: l, y :: m, h => by
    simp only [lleB, Bool.and_eq_true] at h
    obtain ⟨hl, hi⟩ := lleB_sound h.2
    refine ⟨by simp [hl], fun i => ?_⟩
    cases i with
    | zero => exact oleB_sound h.1
    | succ i => exact hi i
  | [], _ :: _, h => by simp [lleB] at h
  | _ :: _, [], h => by simp [lleB] at h

theorem leB_sound {a b : St} (h : leB r o a b = true) : StLe r o a b := by
  simp only [leB, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hr, hsp⟩, hpc⟩, hn⟩, hz⟩, hc⟩, hv⟩, hm⟩, hlo⟩, hi⟩, ho⟩ := h
  obtain ⟨hrl, hri⟩ := lleB_sound hr
  obtain ⟨hml, hmi⟩ := lleB_sound hm
  refine ⟨fun i => hri i, hrl, oleB_sound hsp, hpc,
    ⟨oleB_sound hn, oleB_sound hz, oleB_sound hc, oleB_sound hv⟩,
    fun x => ?_, hml, hlo, hi, ho⟩
  simp only [St.rdByte, hlo]
  split
  · exact hmi _
  · exact OLe.refl _

end Refine

/-! ## Fast kernel-checked sanity witnesses -/

private def testSt (pc spVal : Nat) (regs : List (Option Nat)) : St :=
  { regs := regs, sp := some spVal, pc := pc, fl := .undef,
    mem := List.replicate winSize (some 0), lo := spVal - winSize / 2, inp := [], out := [] }

theorem decode_movz_x19 :
    decodeWord 0xd2800013 = some (.movWide 2 64 19 0 0) := by decide +kernel

theorem decode_add_sp :
    decodeWord 0x910003e1 = some (.addSubImm false false 64 1 31 0) := by decide +kernel

theorem decode_add_lsl :
    decodeWord 0x8b130413 = some (.addSubReg false false 64 19 0 19 .lsl 1) := by decide +kernel

theorem decode_svc0 :
    decodeWord 0xd4000001 = some (.svc 0) := by decide +kernel

theorem decode_svc1_refused :
    decodeWord 0xd4000021 = none := by decide +kernel

theorem sp_unaligned_stuck :
    let im := Img.mk' [0xe0, 0x03, 0x40, 0x39] 0x400000 4
    let s := testSt 0x400000 0x10001 (List.replicate 31 (some 0))
    step im s = .inr (.stuck 0x400000 1) := by decide +kernel

theorem movk_w0_clears_hi :
    let im := Img.mk' [0x80, 0x46, 0x82, 0x72] 0x400000 4
    let s := testSt 0x400000 0x10000 ((List.replicate 31 (some 0)).set 0 (some 0xdeadbeef00005678))
    (match step im s with
      | .inl t => t.reg 0 == some 0x1234
      | .inr _ => false) = true := by decide +kernel

theorem subs_borrow_flag :
    let im := Img.mk' [0x1f, 0x04, 0x00, 0xf1] 0x400000 4
    let s := testSt 0x400000 0x10000 ((List.replicate 31 (some 0)).set 0 (some 0))
    (match step im s with
      | .inl t => t.fl == ⟨some true, some false, some false, some false⟩
      | .inr _ => false) = true := by decide +kernel

end AArch64Min
