/-
AArch64Phys: full architectural and 4-channel physical fault-semantics model
of fixed-width 32-bit Linux AArch64 (`aarch64-linux-gnu`) machine code over
`AArch64Env` and `PDist`.

Two semantics, one machine:
  - `step` / `run`: deterministic architectural semantics over `AArch64Env`
    covering all instruction forms required by `seed-c`, `TMRVoterArm64`, and
    `spr1_check_arm64`:
      * `movWide` (`MOVZ`, `MOVN`, `MOVK`, 32/64-bit)
      * `addSubImm` (`ADD`, `ADDS`, `SUB`, `SUBS`, `CMP`, `CMN`, 32/64-bit)
      * `addSubReg` (`ADD`, `ADDS`, `SUB`, `SUBS`, `CMP`, `CMN` shifted reg)
      * `logicReg` (`AND`, `ORR`, `EOR`, `ANDS`, `BIC`, `ORN`, `EON`, `BICS`,
        `MOV`, `TST` shifted reg)
      * `bitfield` (`SBFM`, `BFM`, `UBFM` — `LSL`/`LSR`/`ASR`/`UBFX`/`SBFX`/`BFI`)
      * `adr` (`ADR`, `ADRP`)
      * `ccmpReg` (`CCMP`, `CCMN` register)
      * `csel` (`CSEL`, `CSINC`, `CSINV`, `CSNEG`)
      * `ldr`/`str` (unsigned scaled imm, 1/2/4/8B), `ldur`/`stur` (signed 9-bit
        unscaled imm, 1/2/4/8B), `ldrReg`/`strReg` (register offset with
        `UXTW`/`LSL`/`SXTW`/`SXTX` and optional scale, 1/2/4/8B)
      * `b`/`bl`, `bcond`, `cbz`/`cbnz`, `tbz`/`tbnz`, `br`/`blr`, `ret`
      * `svc #0` (Linux AArch64 ABI via `AArch64Env.syscall`)
  - `pstep` / `prun`: 4-channel physical perturbation semantics in `PDist`:
      1. `fetch`: 32 single-bit (Hamming-1) flips of the aligned 32-bit word (`altsAt`)
      2. `reg`: single-bit upsets across the 32 × 64-bit register file (`x0..x30` + `sp`)
      3. `mem`: single-bit upsets across every mapped byte in `s.mem`
      4. `nzcv`: single-bit upsets across defined `N, Z, C, V` flags

Key theorems:
  - `faultFree_prun`: zero-fault projection of `prun` equals deterministic `run`.
  - `prun_ok`, `fm_prun`, `baseline`: exact rational probability bounds.
  - `fetch_fault_fixed_boundary`: Fixed-Width 32-Bit Non-Desynchronization
    Theorem — every Hamming-1/k fetch perturbation of a 4-byte aligned word at
    `pc` depends strictly on `[pc, pc + 4)`, never reads `pc + 4`, never
    decodes to length `≠ 4`, and preserves downstream 4-byte boundaries `pc + 4j`.
  - `aligned_never_straddles_page`: an aligned 4-byte fetch inside a mapped
    page never crosses into the adjacent guard page.

Core Lean 4 only: zero Mathlib imports, zero axioms beyond core Lean.
-/
import Bootstrap.AArch64Env
import Bootstrap.PDist

namespace AArch64Phys

open AArch64Env PDist

/-! ### Bit-level arithmetic helpers -/

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

/-! ### Architectural types and instruction set -/

/-- General-purpose register index: `0..30` are `x0..x30`; `31` is `SP` or `XZR/WZR`. -/
abbrev Reg := Nat

structure Flags where
  n : Option Bool
  z : Option Bool
  c : Option Bool
  v : Option Bool
deriving DecidableEq, Repr

def Flags.undef : Flags := ⟨none, none, none, none⟩

inductive ShiftKind where
  | lsl | lsr | asr | ror
deriving DecidableEq, Repr

inductive LogicOp where
  | and | orr | eor | ands
deriving DecidableEq, Repr

inductive BitfieldKind where
  | sbfm | bfm | ubfm
deriving DecidableEq, Repr

inductive MemSize where
  | b1 | h2 | w4 | x8
deriving DecidableEq, Repr

def MemSize.bytes : MemSize → Nat
  | .b1 => 1
  | .h2 => 2
  | .w4 => 4
  | .x8 => 8

def MemSize.log2 : MemSize → Nat
  | .b1 => 0
  | .h2 => 1
  | .w4 => 2
  | .x8 => 3

def MemSize.width : MemSize → Nat
  | .b1 => 32
  | .h2 => 32
  | .w4 => 32
  | .x8 => 64

inductive ExtendKind where
  | uxtw | lsl | sxtw | sxtx
deriving DecidableEq, Repr

inductive CSelOp where
  | csel | csinc | csinv | csneg
deriving DecidableEq, Repr

inductive Ins where
  /-- Move wide immediate (`kind`: 0 = MOVN, 2 = MOVZ, 3 = MOVK). -/
  | movWide (kind w rd imm16 hw : Nat)
  /-- Add/subtract immediate (`rn = 31` is SP; `rd = 31` is SP if `!setFlags`, ZR if `setFlags`). -/
  | addSubImm (isSub setFlags : Bool) (w rd rn imm : Nat)
  /-- Add/subtract shifted register (`31` is ZR). -/
  | addSubReg (isSub setFlags : Bool) (w rd rn rm : Nat) (sh : ShiftKind) (amt : Nat)
  /-- Logical shifted register (`31` is ZR). -/
  | logicReg (op : LogicOp) (inv : Bool) (w rd rn rm : Nat) (sh : ShiftKind) (amt : Nat)
  /-- Bitfield move (`SBFM`, `BFM`, `UBFM`; `31` is ZR). -/
  | bitfield (kind : BitfieldKind) (w rd rn immr imms : Nat)
  /-- PC-relative address (`isPage = false` for `ADR`, `true` for `ADRP`; `31` is ZR). -/
  | adr (isPage : Bool) (rd imm : Nat)
  /-- Conditional compare register (`isSub = true` for `CCMP`, `false` for `CCMN`; `31` is ZR). -/
  | ccmpReg (isSub : Bool) (w rn rm nzcv cc : Nat)
  /-- Conditional select (`CSEL`, `CSINC`, `CSINV`, `CSNEG`; `31` is ZR). -/
  | csel (op : CSelOp) (w rd rn rm cc : Nat)
  /-- Load unsigned scaled immediate offset (`rn = 31` is SP, `rt = 31` is ZR). -/
  | ldr (sz : MemSize) (rt rn off : Nat)
  /-- Store unsigned scaled immediate offset (`rn = 31` is SP, `rt = 31` is ZR). -/
  | str (sz : MemSize) (rt rn off : Nat)
  /-- Load unscaled signed 9-bit immediate offset (`off` sign-extended to 64 bits). -/
  | ldur (sz : MemSize) (rt rn off : Nat)
  /-- Store unscaled signed 9-bit immediate offset (`off` sign-extended to 64 bits). -/
  | stur (sz : MemSize) (rt rn off : Nat)
  /-- Load register offset (`rn = 31` is SP, `rm`/`rt = 31` is ZR). -/
  | ldrReg (sz : MemSize) (rt rn rm : Nat) (ext : ExtendKind) (scaled : Bool)
  /-- Store register offset (`rn = 31` is SP, `rm`/`rt = 31` is ZR). -/
  | strReg (sz : MemSize) (rt rn rm : Nat) (ext : ExtendKind) (scaled : Bool)
  /-- Unconditional branch (`link = true` for `BL`). `off` is 64-bit sign-extended byte offset. -/
  | b (link : Bool) (off : Nat)
  /-- Conditional branch (`cc` 0..15). `off` is 64-bit sign-extended byte offset. -/
  | bcond (cc off : Nat)
  /-- Compare and branch on zero/nonzero (`31` is ZR). -/
  | cbz (nz : Bool) (w rt off : Nat)
  /-- Test bit `bit ∈ 0..63` and branch on zero/nonzero (`31` is ZR). -/
  | tbz (nz : Bool) (bit rt off : Nat)
  /-- Indirect branch (`link = false` for `BR`, `true` for `BLR`; `31` is ZR). -/
  | br (link : Bool) (rn : Nat)
  /-- Return to address in `rn` (`31` is ZR). -/
  | ret (rn : Nat)
  /-- Supervisor call (`imm = 0` only). -/
  | svc (imm : Nat)
deriving DecidableEq, Repr

/-- Whether `ins` has statically 4-byte-aligned control flow (everything except indirect `BR`/`BLR`/`RET`). -/
def Ins.isDirect : Ins → Bool
  | .br _ _ | .ret _ => false
  | _ => true

/-! ### Fixed-width 32-bit decoder and word-aligned images -/

/-- Little-endian value of `n` bytes read through `b`. -/
def le (b : Nat → Option Nat) (off : Nat) : Nat → Option Nat
  | 0 => some 0
  | n + 1 => do
    let lo ← b off
    let hi ← le b (off + 1) n
    pure (lo + 256 * hi)

def memSizeOf (szBits : Nat) : Option MemSize :=
  match szBits with
  | 0 => some .b1
  | 1 => some .h2
  | 2 => some .w4
  | 3 => some .x8
  | _ => none

def extendOf (optBits : Nat) : Option ExtendKind :=
  match optBits with
  | 2 => some .uxtw
  | 3 => some .lsl
  | 6 => some .sxtw
  | 7 => some .sxtx
  | _ => none

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
  else if bf w 25 6 == 0x1b then
    let nz := bf w 24 1 == 1
    let bit := sf * 32 + bf w 19 5
    let imm14 := bf w 5 14
    let rt := bf w 0 5
    some (.tbz nz bit rt (sx 16 (imm14 * 4)))
  else if w == 0xd4000001 then
    some (.svc 0)
  else if bf w 10 22 == 0x3587c0 && bf w 0 5 == 0 then
    some (.br false (bf w 5 5))
  else if bf w 10 22 == 0x358fc0 && bf w 0 5 == 0 then
    some (.br true (bf w 5 5))
  else if bf w 10 22 == 0x3597c0 && bf w 0 5 == 0 then
    some (.ret (bf w 5 5))
  else if bf w 24 5 == 0x10 then
    let isPage := sf == 1
    let immlo := bf w 29 2
    let immhi := bf w 5 19
    let rd := bf w 0 5
    let imm21 := immhi * 4 + immlo
    let imm := if isPage then sx 33 (imm21 * 4096) else sx 21 imm21
    some (.adr isPage rd imm)
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
  else if bf w 23 6 == 0x26 then
    let opc := bf w 29 2
    let nBit := bf w 22 1
    let immr := bf w 16 6
    let imms := bf w 10 6
    let rn := bf w 5 5
    let rd := bf w 0 5
    if opc == 3 || nBit != sf || (sf == 0 && (immr ≥ 32 || imms ≥ 32)) then none
    else
      let kind := match opc with | 0 => .sbfm | 1 => .bfm | _ => .ubfm
      some (.bitfield kind wd rd rn immr imms)
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
  else if bf w 21 8 == 0xd2 && bf w 29 1 == 1 && bf w 10 2 == 0 && bf w 4 1 == 0 then
    let isSub := bf w 30 1 == 1
    let rm := bf w 16 5
    let cc := bf w 12 4
    let rn := bf w 5 5
    let nzcv := bf w 0 4
    some (.ccmpReg isSub wd rn rm nzcv cc)
  else if bf w 21 8 == 0xd4 && bf w 29 1 == 0 && bf w 11 1 == 0 then
    let opBit := bf w 30 1 == 1
    let incBit := bf w 10 1 == 1
    let rm := bf w 16 5
    let cc := bf w 12 4
    let rn := bf w 5 5
    let rd := bf w 0 5
    let op := match opBit, incBit with
      | false, false => .csel
      | false, true  => .csinc
      | true,  false => .csinv
      | true,  true  => .csneg
    some (.csel op wd rd rn rm cc)
  else if bf w 24 6 == 0x39 then
    let szBits := bf w 30 2
    let opc := bf w 22 2
    let imm12 := bf w 10 12
    let rn := bf w 5 5
    let rt := bf w 0 5
    match memSizeOf szBits with
    | none => none
    | some sz =>
      let off := imm12 * sz.bytes
      if opc == 0 then some (.str sz rt rn off)
      else if opc == 1 then some (.ldr sz rt rn off)
      else none
  else if bf w 24 6 == 0x38 && bf w 21 1 == 0 && bf w 10 2 == 0 then
    let szBits := bf w 30 2
    let opc := bf w 22 2
    let imm9 := bf w 12 9
    let rn := bf w 5 5
    let rt := bf w 0 5
    match memSizeOf szBits with
    | none => none
    | some sz =>
      let off := sx 9 imm9
      if opc == 0 then some (.stur sz rt rn off)
      else if opc == 1 then some (.ldur sz rt rn off)
      else none
  else if bf w 24 6 == 0x38 && bf w 21 1 == 1 && bf w 10 2 == 2 then
    let szBits := bf w 30 2
    let opc := bf w 22 2
    let rm := bf w 16 5
    let optBits := bf w 13 3
    let sBit := bf w 12 1 == 1
    let rn := bf w 5 5
    let rt := bf w 0 5
    match memSizeOf szBits, extendOf optBits with
    | some sz, some ext =>
      if opc == 0 then some (.strReg sz rt rn rm ext sBit)
      else if opc == 1 then some (.ldrReg sz rt rn rm ext sBit)
      else none
    | _, _ => none
  else none

/-- Decode one fixed-width 32-bit AArch64 instruction at offset 0 of byte stream `b`. -/
def decode (b : Nat → Option Nat) : Option (Ins × Nat) := do
  let w ← le b 0 4
  let ins ← decodeWord w
  pure (ins, 4)

/-- An instruction image mapping 64-bit addresses to decoded `(Ins, len)`. -/
abbrev Img := Nat → Option (Ins × Nat)

/-- Hamming-1 fetch neighbours at address `a`. -/
abbrev Alts := Nat → List (Option (Ins × Nat))

/-- An image is fixed-width 32-bit aligned when every defined entry lies on a
4-byte boundary and has length 4. -/
def Img.FixedWidth (im : Img) : Prop :=
  ∀ a ins len, im a = some (ins, len) → a % 4 = 0 ∧ len = 4

/-- Byte stream `b` with bit `k` inverted (byte `k / 8`, bit `k % 8`). -/
def flipBit (b : Nat → Option Nat) (k : Nat) : Nat → Option Nat :=
  fun i => if i == k / 8 then (b i).map fun v => Nat.xor v (2 ^ (k % 8)) else b i

/-- Byte stream `b` after applying a sequence `ks` of bit flips (Hamming-k perturbation). -/
def flipBits (b : Nat → Option Nat) (ks : List Nat) : Nat → Option Nat :=
  ks.foldl flipBit b

/-- All 32 single-bit (Hamming-1) fetch neighbours of the 4-byte word at offset 0 of `b`. -/
def altsAt (b : Nat → Option Nat) : List (Option (Ins × Nat)) :=
  (List.range 32).map fun k => decode (flipBit b k)

/-- All 32 single-bit (Hamming-1) fetch neighbours directly from a 32-bit word `w`. -/
def altsWord (w : Nat) : List (Option (Ins × Nat)) :=
  (List.range 32).map fun k => (decodeWord (Nat.xor w (2 ^ k))).map (·, 4)

/-- 4-byte-aligned image of raw code bytes loaded at 4-byte-aligned `base`. -/
def imageOf (code : List Nat) (base : Nat) : Img :=
  fun a =>
    if base ≤ a && a + 4 ≤ base + code.length && base % 4 == 0 && a % 4 == 0 then
      decode (fun i => code[a - base + i]?)
    else none

/-- Hamming-1 fetch alternatives for every aligned 32-bit word in `code` at `base`. -/
def imageAlts (code : List Nat) (base : Nat) : Alts :=
  fun a =>
    if base ≤ a && a + 4 ≤ base + code.length && base % 4 == 0 && a % 4 == 0 then
      altsAt (fun i => code[a - base + i]?)
    else []

/-- Complete binary tree of depth `d` for fast O(log N) kernel-reducible word images. -/
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

/-- Word-aligned binary-tree instruction image of arbitrary depth `depth` (`2^depth` words). -/
structure WordImg where
  vbase : Nat
  depth : Nat
  len   : Nat
  words : Tree (Option Ins)
deriving DecidableEq

def WordImg.mk' (code : List Nat) (vbase depth : Nat) : WordImg :=
  let n := code.length
  ⟨vbase, depth, n,
    Tree.build (fun wi =>
      if 4 * wi + 4 ≤ n then (le (fun j => code[4 * wi + j]?) 0 4).bind decodeWord
      else none) depth 0⟩

def WordImg.toImg (im : WordImg) : Img :=
  fun a =>
    if im.vbase ≤ a && a + 4 ≤ im.vbase + im.len && im.vbase % 4 == 0 && a % 4 == 0 then
      (Tree.get none im.depth im.words ((a - im.vbase) / 4)).map (·, 4)
    else none

theorem WordImg.toImg_fixedWidth (im : WordImg) : im.toImg.FixedWidth := by
  intro a ins len h
  unfold WordImg.toImg at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    cases ht : Tree.get none im.depth im.words ((a - im.vbase) / 4) with
    | none => rw [ht] at h; cases h
    | some i =>
      rw [ht] at h
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨hc.2, h.2.symm⟩
  · cases h

theorem WordImg.toImg_eq_imageOf (code : List Nat) (vbase depth : Nat)
    (hlen : code.length ≤ 4 * 2 ^ depth) :
    (WordImg.mk' code vbase depth).toImg = imageOf code vbase := by
  funext a
  unfold WordImg.toImg WordImg.mk' imageOf
  dsimp only
  by_cases hc : (vbase ≤ a && a + 4 ≤ vbase + code.length && vbase % 4 == 0 && a % 4 == 0) = true
  · rw [if_pos hc, if_pos hc]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    have hwi : (a - vbase) / 4 < 2 ^ depth := by omega
    rw [Tree.get_build none _ depth 0 ((a - vbase) / 4) hwi, Nat.zero_add]
    have hmul : 4 * ((a - vbase) / 4) = a - vbase := by omega
    rw [hmul, if_pos (by omega)]
    unfold decode
    cases hle : le (fun i => code[a - vbase + i]?) 0 4 with
    | none => rfl
    | some w =>
      simp only [bind, Option.bind, pure]
      cases decodeWord w <;> rfl
  · rw [if_neg hc, if_neg hc]

/-! ### Machine state and deterministic execution -/

structure St where
  regs : List (Option Nat)
  sp   : Option Nat
  pc   : Nat
  fl   : Flags
  mem  : Mem
  k    : Kern
deriving DecidableEq, Repr

inductive Out where
  | ok (s : St)
  | exit (status : Nat)
  | stuck
deriving DecidableEq, Repr

def St.reg (s : St) (i : Reg) : Option Nat := (s.regs[i]?).getD none

def St.setReg (s : St) (i : Reg) (v : Option Nat) : St := { s with regs := s.regs.set i v }

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

/-- Little-endian byte list of length `n`. -/
def bytesN (v : Nat) : Nat → List Nat
  | 0 => []
  | n + 1 => (v % 256) :: bytesN (v / 256) n

def leVal : List Nat → Nat
  | [] => 0
  | b :: bs => b + 256 * leVal bs

def rdMem (m : Mem) (addr : Nat) (sz : MemSize) : Option Nat :=
  (m.rdN addr sz.bytes).map leVal

def wrMem (m : Mem) (addr v : Nat) (sz : MemSize) : Option Mem :=
  m.wrN addr (bytesN v sz.bytes)

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

/-- Evaluate ARMv8-A bitfield move (`SBFM`, `BFM`, `UBFM`) at width `w ∈ {32, 64}`. -/
def bitfieldOp (kind : BitfieldKind) (w src dst immr imms : Nat) : Nat :=
  let mask := pow2 w - 1
  let welem := pow2 (imms + 1) - 1
  let wmask := rorVal w welem immr
  let d := (imms + w - immr) % w
  let tmask := pow2 (d + 1) - 1
  let rsrc := rorVal w (src % pow2 w) immr
  let dstEff := match kind with | .bfm => dst % pow2 w | _ => 0
  let bot := Nat.lor (Nat.land dstEff (mask - wmask)) (Nat.land rsrc wmask)
  let signBit := (src / pow2 imms) % 2 == 1
  let top := match kind with
    | .sbfm => if signBit then mask else 0
    | .ubfm => 0
    | .bfm  => dstEff
  Nat.lor (Nat.land top (mask - tmask)) (Nat.land bot tmask)

def extendIdx (ext : ExtendKind) (v : Nat) : Nat :=
  match ext with
  | .uxtw => v % pow2 32
  | .lsl  => v % pow2 64
  | .sxtw => sx 32 (v % pow2 32)
  | .sxtx => v % pow2 64

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

def lift : Option St → Out
  | some s => .ok s
  | none => .stuck

def applySys (s1 : St) (res : Option SysRes) : Out :=
  match res with
  | some (.ret x0 k' m') => .ok { s1.setReg 0 (some (x0 % W)) with k := k', mem := m' }
  | some (.exit c) => .exit c
  | none => .stuck

/-- Execute one Linux AArch64 `svc #0` system call on state `s` advancing to `nx`. -/
def execSvc (s : St) (nx : Nat) : Out :=
  let s1 := { s with pc := nx }
  match s.reg 8 with
  | some 93 =>
    match s.reg 0 with
    | some a0 => applySys s1 (syscall s.k s.mem 93 a0 0 0 0 0 0)
    | none => .stuck
  | some 94 =>
    match s.reg 0 with
    | some a0 => applySys s1 (syscall s.k s.mem 94 a0 0 0 0 0 0)
    | none => .stuck
  | some 57 =>
    match s.reg 0 with
    | some a0 => applySys s1 (syscall s.k s.mem 57 a0 0 0 0 0 0)
    | none => .stuck
  | some 215 =>
    match s.reg 0, s.reg 1 with
    | some a0, some a1 => applySys s1 (syscall s.k s.mem 215 a0 a1 0 0 0 0)
    | _, _ => .stuck
  | some 56 =>
    match s.reg 0, s.reg 1, s.reg 2 with
    | some a0, some a1, some a2 =>
      applySys s1 (syscall s.k s.mem 56 a0 a1 a2 ((s.reg 3).getD 0) 0 0)
    | _, _, _ => .stuck
  | some 63 =>
    match s.reg 0, s.reg 1, s.reg 2 with
    | some a0, some a1, some a2 => applySys s1 (syscall s.k s.mem 63 a0 a1 a2 0 0 0)
    | _, _, _ => .stuck
  | some 64 =>
    match s.reg 0, s.reg 1, s.reg 2 with
    | some a0, some a1, some a2 => applySys s1 (syscall s.k s.mem 64 a0 a1 a2 0 0 0)
    | _, _, _ => .stuck
  | some 216 =>
    match s.reg 0, s.reg 1, s.reg 2, s.reg 3 with
    | some a0, some a1, some a2, some a3 => applySys s1 (syscall s.k s.mem 216 a0 a1 a2 a3 0 0)
    | _, _, _, _ => .stuck
  | some 222 =>
    match s.reg 0, s.reg 1, s.reg 2, s.reg 3, s.reg 4, s.reg 5 with
    | some a0, some a1, some a2, some a3, some a4, some a5 =>
      applySys s1 (syscall s.k s.mem 222 a0 a1 a2 a3 a4 a5)
    | _, _, _, _, _, _ => .stuck
  | _ => .stuck

/-- Execute decoded instruction `ins` of length `len` at `s.pc`. -/
def exec (ins : Ins) (len : Nat) (s : St) : Out :=
  let nx := (s.pc + len) % W
  let s1 := { s with pc := nx }
  match ins with
  | .movWide kind w rd imm16 hw =>
    let sh := hw * 16
    let part := imm16 * pow2 sh
    match kind with
    | 2 => .ok (s1.wrZR w rd part)
    | 0 => .ok (s1.wrZR w rd (pow2 w - 1 - part))
    | 3 => lift (do
        let old ← s.rdZR w rd
        let lo := old % pow2 sh
        let hi := old / pow2 (sh + 16)
        pure (s1.wrZR w rd (lo + part + hi * pow2 (sh + 16))))
    | _ => .stuck
  | .addSubImm isSub setFlags w rd rn imm => lift (do
      let a ← s.rdSP w rn
      let (res, f) := addSubOp isSub w a (imm % pow2 w)
      if setFlags then pure ({ s1 with fl := f }.wrZR w rd res)
      else pure (s1.wrSP w rd res))
  | .addSubReg isSub setFlags w rd rn rm sh amt => lift (do
      let a ← s.rdZR w rn
      let bRaw ← s.rdZR w rm
      let b := shiftVal sh w bRaw amt
      let (res, f) := addSubOp isSub w a b
      let t := if setFlags then { s1 with fl := f } else s1
      pure (t.wrZR w rd res))
  | .logicReg op inv w rd rn rm sh amt => lift (do
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
  | .bitfield kind w rd rn immr imms => lift (do
      let src ← s.rdZR w rn
      let dst ← match kind with
        | .bfm => s.rdZR w rd
        | _ => some 0
      pure (s1.wrZR w rd (bitfieldOp kind w src dst immr imms)))
  | .adr isPage rd imm =>
    let base := if isPage then (s.pc / 4096) * 4096 else s.pc
    .ok (s1.wrZR 64 rd ((base + imm) % W))
  | .ccmpReg isSub w rn rm nzcv cc => lift (do
      let a ← s.rdZR w rn
      let b ← s.rdZR w rm
      let c ← cond s.fl cc
      let f := if c then (addSubOp isSub w a b).2
               else ⟨some (bf nzcv 3 1 == 1), some (bf nzcv 2 1 == 1),
                     some (bf nzcv 1 1 == 1), some (bf nzcv 0 1 == 1)⟩
      pure { s1 with fl := f })
  | .csel op w rd rn rm cc => lift (do
      let a ← s.rdZR w rn
      let b ← s.rdZR w rm
      let c ← cond s.fl cc
      let res := if c then a
        else match op with
          | .csel  => b
          | .csinc => (b + 1) % pow2 w
          | .csinv => pow2 w - 1 - b
          | .csneg => (pow2 w - b) % pow2 w
      pure (s1.wrZR w rd res))
  | .ldr sz rt rn off | .ldur sz rt rn off => lift (do
      let base ← s.rdSP 64 rn
      if rn == 31 && base % 16 != 0 then none else
      let addr := (base + off) % W
      let v ← rdMem s.mem addr sz
      pure (s1.wrZR sz.width rt v))
  | .str sz rt rn off | .stur sz rt rn off => lift (do
      let base ← s.rdSP 64 rn
      if rn == 31 && base % 16 != 0 then none else
      let addr := (base + off) % W
      let v ← s.rdZR sz.width rt
      let m' ← wrMem s.mem addr v sz
      pure { s1 with mem := m' })
  | .ldrReg sz rt rn rm ext scaled => lift (do
      let base ← s.rdSP 64 rn
      if rn == 31 && base % 16 != 0 then none else
      let idxRaw ← s.rdZR 64 rm
      let idx := extendIdx ext idxRaw
      let sh := if scaled then sz.log2 else 0
      let addr := (base + (idx * pow2 sh) % W) % W
      let v ← rdMem s.mem addr sz
      pure (s1.wrZR sz.width rt v))
  | .strReg sz rt rn rm ext scaled => lift (do
      let base ← s.rdSP 64 rn
      if rn == 31 && base % 16 != 0 then none else
      let idxRaw ← s.rdZR 64 rm
      let idx := extendIdx ext idxRaw
      let sh := if scaled then sz.log2 else 0
      let addr := (base + (idx * pow2 sh) % W) % W
      let v ← s.rdZR sz.width rt
      let m' ← wrMem s.mem addr v sz
      pure { s1 with mem := m' })
  | .b link off =>
    let s2 := if link then s1.setReg 30 (some nx) else s1
    .ok { s2 with pc := (s.pc + off) % W }
  | .bcond cc off => lift (do
      let c ← cond s.fl cc
      pure (if c then { s1 with pc := (s.pc + off) % W } else s1))
  | .cbz nz w rt off => lift (do
      let v ← s.rdZR w rt
      let taken := if nz then v != 0 else v == 0
      pure (if taken then { s1 with pc := (s.pc + off) % W } else s1))
  | .tbz nz bit rt off => lift (do
      let v ← s.rdZR 64 rt
      let isSet := (v / pow2 bit) % 2 == 1
      let taken := if nz then isSet else !isSet
      pure (if taken then { s1 with pc := (s.pc + off) % W } else s1))
  | .br link rn => lift (do
      let tgt ← s.rdZR 64 rn
      let s2 := if link then s1.setReg 30 (some nx) else s1
      pure { s2 with pc := tgt })
  | .ret rn => lift (do
      let tgt ← s.rdZR 64 rn
      pure { s1 with pc := tgt })
  | .svc imm =>
    if imm != 0 then .stuck else execSvc s nx

/-- Deterministic architectural step. -/
def step (im : Img) (s : St) : Out :=
  match im s.pc with
  | none => .stuck
  | some (ins, len) => exec ins len s

/-- Deterministic architectural run of at most `n` steps. -/
def run (im : Img) : Nat → Out → Out
  | 0, o => o
  | n + 1, .ok s => run im n (step im s)
  | _ + 1, o => o

/-! ### Uniform distribution and relabelling helpers -/

def unif {β : Type} (p : Rat) (c : β) (xs : List β) : PDF β :=
  match xs with
  | [] => pureF c
  | _ :: _ => ((c, false), 1 - p) :: xs.map fun x => ((x, true), p / xs.length)

theorem faultFree_unif {β : Type} (p : Rat) (c : β) (xs : List β) :
    faultFree (unif p c xs) = [c] := by
  cases xs with
  | nil => rfl
  | cons x r =>
    simp only [unif, faultFree, Bool.false_eq_true, if_false, List.cons.injEq, true_and]
    generalize (p / (r.length + 1 : Nat)) = w
    generalize (x :: r) = l
    induction l with
    | nil => rfl
    | cons y l ih => simpa [faultFree] using ih

theorem mass_const {β : Type} (l : List β) (w : Rat) (t : Bool) :
    mass (l.map fun x => ((x, t), w)) = l.length * w := by
  induction l with
  | nil => show (0 : Rat) = ((0 : Nat) : Rat) * w; rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
  | cons x r ih =>
    simp only [List.map_cons, mass, ih, List.length_cons, Rat.natCast_add]
    rw [Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]

theorem fm_const {β : Type} (l : List β) (w : Rat) :
    fm (l.map fun x => ((x, true), w)) = l.length * w := by
  induction l with
  | nil => show (0 : Rat) = ((0 : Nat) : Rat) * w; rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
  | cons x r ih =>
    simp only [List.map_cons, fm, if_true, ih, List.length_cons, Rat.natCast_add]
    rw [Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]

theorem share_sum (n : Nat) (h : n ≠ 0) (p : Rat) : (n : Rat) * (p / n) = p := by
  rw [Rat.div_def, Rat.mul_comm p, ← Rat.mul_assoc, Rat.mul_inv_cancel _ (by exact_mod_cast h), Rat.one_mul]

theorem share_nonneg (n : Nat) (h : n ≠ 0) {p : Rat} (hp : 0 ≤ p) : 0 ≤ p / n := by
  rw [Rat.div_def]
  exact Rat.mul_nonneg hp (Rat.le_of_lt (Rat.inv_pos.2 (Rat.natCast_pos.2 (Nat.pos_of_ne_zero h))))

theorem unif_ok {β : Type} {p : Rat} (hp : Rate p) (c : β) (xs : List β) : (unif p c xs).ok := by
  cases xs with
  | nil => exact pureF_ok c
  | cons x r =>
    have hn : (x :: r).length ≠ 0 := by simp
    refine ⟨?_, ?_⟩
    · intro y hy
      simp only [unif, List.mem_cons, List.mem_map] at hy
      rcases hy with rfl | ⟨z, _, rfl⟩
      · exact (Rat.le_iff_sub_nonneg _ _).1 hp.2
      · exact share_nonneg _ hn hp.1
    · show (1 - p) + mass ((x :: r).map fun y => ((y, true), p / (x :: r).length)) = 1
      rw [mass_const, share_sum _ hn, Rat.sub_add_cancel]

theorem fm_unif_le {β : Type} {p : Rat} (hp : Rate p) (c : β) (xs : List β) :
    fm (unif p c xs) ≤ p := by
  cases xs with
  | nil => rw [unif, fm_pureF]; exact hp.1
  | cons x r =>
    have hn : (x :: r).length ≠ 0 := by simp
    show (if false = true then 1 - p else 0) + fm ((x :: r).map fun y => ((y, true), p / (x :: r).length)) ≤ p
    rw [fm_const, share_sum _ hn]
    simp only [Bool.false_eq_true, if_false]
    rw [Rat.zero_add]; exact Rat.le_refl

theorem nonneg_add {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  have := add_le_add' ha hb; rwa [Rat.add_zero] at this

def relabel {α β : Type} (g : α → β) (d : PDF α) : PDF β :=
  d.map fun x => ((g x.1.1, x.1.2), x.2)

theorem faultFree_relabel {α β : Type} (g : α → β) (d : PDF α) :
    faultFree (relabel g d) = (faultFree d).map g := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    cases t <;> simp_all [relabel, faultFree]

theorem mass_relabel {α β : Type} (g : α → β) (d : PDF α) : mass (relabel g d) = mass d := by
  induction d with
  | nil => rfl
  | cons x r ih => simp_all [relabel, mass]

theorem fm_relabel {α β : Type} (g : α → β) (d : PDF α) : fm (relabel g d) = fm d := by
  induction d with
  | nil => rfl
  | cons x r ih => obtain ⟨⟨a, t⟩, w⟩ := x; simp_all [relabel, fm]

theorem relabel_ok {α β : Type} (g : α → β) {d : PDF α} (h : d.ok) : (relabel g d).ok := by
  refine ⟨?_, by rw [mass_relabel]; exact h.2⟩
  intro y hy
  simp only [relabel, List.mem_map] at hy
  obtain ⟨x, hx, rfl⟩ := hy
  exact h.1 x hx

/-! ### 4-channel physical fault models (`fetch`, `reg`, `mem`, `nzcv`) -/

structure Fault where
  hit   : St → PDF St
  q     : Rat
  q0    : 0 ≤ q
  clean : ∀ s, faultFree (hit s) = [s]
  ok    : ∀ s, (hit s).ok
  le    : ∀ s, fm (hit s) ≤ q

def Fault.none : Fault :=
  ⟨pureF, 0, Rat.le_refl, fun _ => rfl, pureF_ok, fun _ => by rw [fm_pureF]; exact Rat.le_refl⟩

def Fault.seq (F G : Fault) : Fault where
  hit s := bindF (F.hit s) G.hit
  q := F.q + G.q
  q0 := nonneg_add F.q0 G.q0
  clean s := by rw [faultFree_bindF, F.clean]; simp [G.clean]
  ok s := bindF_ok (F.ok s) G.ok
  le s := Rat.le_trans (fm_bindF_le' (F.ok s) G.q0 G.ok G.le) (add_le_add' (F.le s) Rat.le_refl)

/-- Invert bit `b` of register `i` (`0..30` GPRs, `31 = SP`); undefined stays undefined. -/
def St.flipReg (s : St) (i b : Nat) : St :=
  if i == 31 then { s with sp := s.sp.map fun v => Nat.xor v (2 ^ b) }
  else s.setReg i ((s.reg i).map fun v => Nat.xor v (2 ^ b))

/-- Every single-bit upset of the 32 × 64-bit register file (`x0..x30` + `sp`). -/
def regFlips (s : St) : List St :=
  (List.range (32 * 64)).map fun j => s.flipReg (j / 64) (j % 64)

/-- Invert bit `b` of byte `j` of mapping `i`. -/
def St.flipMem (s : St) (i j b : Nat) : St :=
  { s with mem := s.mem.modify i fun r => { r with data := r.data.modify j fun v => Nat.xor v (2 ^ b) } }

/-- Every single-bit upset of the mapped bytes. -/
def memFlips (s : St) : List St :=
  (List.range s.mem.length).flatMap fun i =>
    (List.range ((s.mem[i]?.map fun r => r.data.length).getD 0)).flatMap fun j =>
      (List.range 8).map fun b => s.flipMem i j b

/-- Every single-bit upset of the defined `NZCV` flags. -/
def nzcvFlips (s : St) : List St :=
  (s.fl.n.toList.map fun b => { s with fl := { s.fl with n := some (!b) } }) ++
  (s.fl.z.toList.map fun b => { s with fl := { s.fl with z := some (!b) } }) ++
  (s.fl.c.toList.map fun b => { s with fl := { s.fl with c := some (!b) } }) ++
  (s.fl.v.toList.map fun b => { s with fl := { s.fl with v := some (!b) } })

def Fault.uniform (alt : St → List St) (p : Rat) (hp : Rate p) : Fault where
  hit s := unif p s (alt s)
  q := p
  q0 := hp.1
  clean s := faultFree_unif p s (alt s)
  ok s := unif_ok hp s (alt s)
  le s := fm_unif_le hp s (alt s)

def execO (s : St) : Option (Ins × Nat) → Out
  | some (j, l) => exec j l s
  | none => .stuck

/-- The rates of the four physical failure channels on AArch64. -/
structure Rates where
  fetch : Rat
  reg   : Rat
  mem   : Rat
  nzcv  : Rat

def Rates.ok (r : Rates) : Prop := Rate r.fetch ∧ Rate r.reg ∧ Rate r.mem ∧ Rate r.nzcv

def Rates.total (r : Rates) : Rat := r.fetch + (r.reg + (r.mem + r.nzcv))

def Rates.state (r : Rates) (h : r.ok) : Fault :=
  (Fault.uniform regFlips r.reg h.2.1).seq
    ((Fault.uniform memFlips r.mem h.2.2.1).seq
      (Fault.uniform nzcvFlips r.nzcv h.2.2.2))

/-- The physical step: Hamming-1 instruction fetch perturbation, execution, and
post-retirement state perturbation (`reg`, `mem`, `nzcv`). -/
def pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Img) (s : St) : PDF Out :=
  let fetched : PDF Out := match im s.pc with
    | none => pureF .stuck
    | some (ins, len) => unif pf (exec ins len s) ((alts s.pc).map (execO s))
  bindF fetched fun o => match o with
    | .ok s' => relabel Out.ok (F.hit s')
    | o => pureF o

theorem retire_ok (F : Fault) (o : Out) :
    (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o).ok := by
  cases o with
  | ok s' => exact relabel_ok _ (F.ok s')
  | exit c => exact pureF_ok _
  | stuck => exact pureF_ok _

theorem retire_le (F : Fault) (o : Out) :
    fm (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o) ≤ F.q := by
  cases o with
  | ok s' => rw [fm_relabel]; exact F.le s'
  | exit c => show fm (pureF (Out.exit c)) ≤ F.q; rw [fm_pureF]; exact F.q0
  | stuck => show fm (pureF Out.stuck) ≤ F.q; rw [fm_pureF]; exact F.q0

theorem retire_clean (F : Fault) (o : Out) :
    faultFree (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o) = [o] := by
  cases o with
  | ok s' => rw [faultFree_relabel, F.clean]; rfl
  | exit c => rfl
  | stuck => rfl

theorem faultFree_pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Img) (s : St) :
    faultFree (pstep alts F pf im s) = [step im s] := by
  unfold pstep step
  rw [faultFree_bindF]
  cases im s.pc with
  | none => simp [faultFree_pureF, retire_clean]
  | some p =>
    obtain ⟨ins, len⟩ := p
    simp [faultFree_unif, retire_clean]

theorem fetched_ok {pf : Rat} (hpf : Rate pf) (alts : Alts) (im : Img) (s : St) :
    (match im s.pc with
      | none => pureF Out.stuck
      | some (ins, len) => unif pf (exec ins len s) ((alts s.pc).map (execO s))).ok := by
  cases im s.pc with
  | none => exact pureF_ok _
  | some p => obtain ⟨ins, len⟩ := p; exact unif_ok hpf _ _

theorem fetched_le {pf : Rat} (hpf : Rate pf) (alts : Alts) (im : Img) (s : St) :
    fm (match im s.pc with
      | none => pureF Out.stuck
      | some (ins, len) => unif pf (exec ins len s) ((alts s.pc).map (execO s))) ≤ pf := by
  cases im s.pc with
  | none => rw [fm_pureF]; exact hpf.1
  | some p => obtain ⟨ins, len⟩ := p; exact fm_unif_le hpf _ _

theorem pstep_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Img) (s : St) :
    (pstep alts F pf im s).ok :=
  bindF_ok (fetched_ok hpf alts im s) (retire_ok F)

theorem fm_pstep (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Img) (s : St) :
    fm (pstep alts F pf im s) ≤ pf + F.q :=
  Rat.le_trans (fm_bindF_le' (fetched_ok hpf alts im s) F.q0 (retire_ok F) (retire_le F))
    (add_le_add' (fetched_le hpf alts im s) Rat.le_refl)

/-- The physical run: at most `n` steps of `pstep`. -/
def prun (alts : Alts) (F : Fault) (pf : Rat) (im : Img) : Nat → Out → PDF Out
  | 0, o => pureF o
  | n + 1, .ok s => bindF (pstep alts F pf im s) (prun alts F pf im n)
  | _ + 1, o => pureF o

/-- Zero-fault projection of the physical run equals the deterministic architectural run. -/
theorem faultFree_prun (alts : Alts) (F : Fault) (pf : Rat) (im : Img) :
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

theorem prun_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Img) :
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

theorem fm_prun (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Img) :
    ∀ n o, fm (prun alts F pf im n o) ≤ n * (pf + F.q) := by
  have h0 : 0 ≤ pf + F.q := nonneg_add hpf.1 F.q0
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
      show fm (pureF (Out.exit c)) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)
    | stuck =>
      show fm (pureF Out.stuck) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)

/-- Unhardened baseline bound across calibration intervals `(cf, cr, cm, cl)` for
`fetch`, `reg`, `mem`, and `nzcv`. -/
theorem baseline (alts : Alts) (im : Img) (cf cr cm cl : Iv)
    (r : Rates) (hr : r.ok) (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem)
    (hl : cl.mem r.nzcv) (n : Nat) (o : Out) :
    fm (prun alts (r.state hr) r.fetch im n o) ≤ n * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  refine Rat.le_trans (fm_prun alts (r.state hr) hr.1 im n o) ?_
  refine Rat.mul_le_mul_of_nonneg_left ?_ Rat.natCast_nonneg
  show r.fetch + (r.reg + (r.mem + r.nzcv)) ≤ _
  exact add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))

/-! ### Fixed-Width 32-Bit Non-Desynchronization Theorem -/

/-- `decode` inspects ONLY bytes `0, 1, 2, 3` of the byte stream (never byte `4` or beyond). -/
theorem decode_reads_only_4_bytes {b₁ b₂ : Nat → Option Nat}
    (h : ∀ i < 4, b₁ i = b₂ i) : decode b₁ = decode b₂ := by
  unfold decode
  have h0 : b₁ 0 = b₂ 0 := h 0 (by decide)
  have h1 : b₁ 1 = b₂ 1 := h 1 (by decide)
  have h2 : b₁ 2 = b₂ 2 := h 2 (by decide)
  have h3 : b₁ 3 = b₂ 3 := h 3 (by decide)
  simp only [le, h0, h1, h2, h3]

/-- Every instruction decoded by `decode` has exact byte length 4. -/
theorem decode_len_eq_4 {b : Nat → Option Nat} {ins : Ins} {len : Nat}
    (h : decode b = some (ins, len)) : len = 4 := by
  unfold decode at h
  cases hle : le b 0 4 with
  | none => rw [hle] at h; cases h
  | some w =>
    rw [hle] at h
    simp only [bind, Option.bind] at h
    cases hd : decodeWord w with
    | none => rw [hd] at h; cases h
    | some i =>
      rw [hd] at h
      simp only [pure, Option.some.injEq, Prod.mk.injEq] at h
      exact h.2.symm

/-- Every decoded instruction carries a `decodeWord` witness. -/
theorem decode_exists_word {b : Nat → Option Nat} {ins : Ins} {len : Nat}
    (h : decode b = some (ins, len)) : ∃ w, decodeWord w = some ins ∧ len = 4 := by
  unfold decode at h
  cases hle : le b 0 4 with
  | none => rw [hle] at h; cases h
  | some w =>
    rw [hle] at h
    simp only [bind, Option.bind] at h
    cases hd : decodeWord w with
    | none => rw [hd] at h; cases h
    | some i =>
      rw [hd] at h
      simp only [pure, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨w, h.1 ▸ hd, h.2.symm⟩

theorem imageOf_fixedWidth (code : List Nat) (base : Nat) : (imageOf code base).FixedWidth := by
  intro a ins len h
  unfold imageOf at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    exact ⟨hc.2, decode_len_eq_4 h⟩
  · cases h

theorem flipBit_local {b₁ b₂ : Nat → Option Nat} (h : ∀ i < 4, b₁ i = b₂ i) (k : Nat) :
    ∀ i < 4, flipBit b₁ k i = flipBit b₂ k i := by
  intro i hi
  unfold flipBit
  rw [h i hi]

theorem flipBits_local {b₁ b₂ : Nat → Option Nat} (h : ∀ i < 4, b₁ i = b₂ i) (ks : List Nat) :
    ∀ i < 4, flipBits b₁ ks i = flipBits b₂ ks i := by
  induction ks generalizing b₁ b₂ with
  | nil => exact h
  | cons k rest ih => exact ih (flipBit_local h k)

theorem altsAt_length (b : Nat → Option Nat) : (altsAt b).length = 32 := by
  simp [altsAt]

theorem altsWord_length (w : Nat) : (altsWord w).length = 32 := by
  simp [altsWord]

theorem altsAt_len_eq_4 {b : Nat → Option Nat} {o : Option (Ins × Nat)} {ins : Ins} {len : Nat}
    (hm : o ∈ altsAt b) (ho : o = some (ins, len)) : len = 4 := by
  simp only [altsAt, List.mem_map] at hm
  obtain ⟨k, _, hk⟩ := hm
  rw [ho] at hk
  exact decode_len_eq_4 hk

/-- Sign-extending a 4-byte-aligned offset (`imm * 4`) to 64 bits preserves 4-byte alignment. -/
private theorem sx_mul4_mod4 {bits imm : Nat} (hb : 2 ≤ bits) (h64 : bits ≤ 64) :
    sx bits (imm * 4) % 4 = 0 := by
  unfold sx pow2
  split
  · omega
  · have hdiv_b : 4 ∣ 2 ^ bits := by
      have h2 : bits = (bits - 2) + 2 := by omega
      rw [h2, Nat.pow_add]
      exact Nat.dvd_mul_left 4 _
    have hdiv_64 : 4 ∣ 2 ^ 64 := by decide
    have hle : 2 ^ bits ≤ 2 ^ 64 := Nat.pow_le_pow_right (by decide) h64
    obtain ⟨kb, hkb⟩ := hdiv_b
    obtain ⟨k64, hk64⟩ := hdiv_64
    omega

/-- Every direct branch decoded by `decodeWord` has a 4-byte-aligned signed offset. -/
def Ins.AlignedOff : Ins → Prop
  | .b _ off => off % 4 = 0
  | .bcond _ off => off % 4 = 0
  | .cbz _ _ _ off => off % 4 = 0
  | .tbz _ _ _ off => off % 4 = 0
  | _ => True

private theorem alignedOff_ite {c : Bool} {t e : Option Ins} {ins : Ins}
    (ht : ∀ i, t = some i → i.AlignedOff) (he : ∀ i, e = some i → i.AlignedOff)
    (h : (if c = true then t else e) = some ins) : ins.AlignedOff := by
  cases c
  · exact he ins h
  · exact ht ins h

theorem decodeWord_alignedOff {w : Nat} {ins : Ins} (h : decodeWord w = some ins) :
    ins.AlignedOff := by
  unfold decodeWord at h
  refine alignedOff_ite (fun i hi => by cases hi; exact sx_mul4_mod4 (by decide) (by decide)) ?_ h
  intro i h
  refine alignedOff_ite (fun i hi => by cases hi; exact sx_mul4_mod4 (by decide) (by decide)) ?_ h
  intro i h
  refine alignedOff_ite (fun i hi => by cases hi; exact sx_mul4_mod4 (by decide) (by decide)) ?_ h
  intro i h
  refine alignedOff_ite (fun i hi => by cases hi; exact sx_mul4_mod4 (by decide) (by decide)) ?_ h
  intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => alignedOff_ite (fun _ h0 => by cases h0) (fun _ h0 => by cases h0; trivial) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => alignedOff_ite (fun _ h0 => by cases h0) (fun _ h0 => by cases h0; trivial) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => alignedOff_ite (fun _ h0 => by cases h0) (fun _ h0 => by cases h0; trivial) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => alignedOff_ite (fun _ h0 => by cases h0) (fun _ h0 => by cases h0; trivial) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by cases hi; trivial) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by
    dsimp only at hi
    cases hm : memSizeOf (bf w 30 2) with
    | none => rw [hm] at hi; cases hi
    | some sz =>
      rw [hm] at hi
      exact alignedOff_ite (fun _ h0 => by cases h0; trivial)
        (fun _ h0 => alignedOff_ite (fun _ h1 => by cases h1; trivial) (fun _ h1 => by cases h1) h0) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by
    dsimp only at hi
    cases hm : memSizeOf (bf w 30 2) with
    | none => rw [hm] at hi; cases hi
    | some sz =>
      rw [hm] at hi
      exact alignedOff_ite (fun _ h0 => by cases h0; trivial)
        (fun _ h0 => alignedOff_ite (fun _ h1 => by cases h1; trivial) (fun _ h1 => by cases h1) h0) hi) ?_ h; intro i h
  refine alignedOff_ite (fun i hi => by
    dsimp only at hi
    cases hm : memSizeOf (bf w 30 2) with
    | none => rw [hm] at hi; cases hi
    | some sz =>
      cases he : extendOf (bf w 13 3) with
      | none => rw [hm, he] at hi; cases hi
      | some ext =>
        rw [hm, he] at hi
        exact alignedOff_ite (fun _ h0 => by cases h0; trivial)
          (fun _ h0 => alignedOff_ite (fun _ h1 => by cases h1; trivial) (fun _ h1 => by cases h1) h0) hi)
    (fun _ h0 => by cases h0) h

private theorem add_mod4_W {a b : Nat} (ha : a % 4 = 0) (hb : b % 4 = 0) :
    ((a + b) % W) % 4 = 0 := by
  have hW : W % 4 = 0 := by decide
  have hd : 4 ∣ W := Nat.dvd_of_mod_eq_zero hW
  rw [Nat.mod_mod_of_dvd (a + b) hd]
  omega

private theorem lift_ok_pc {o : Option St} {t : St} (h : lift o = .ok t) : o = some t := by
  cases o with
  | none => cases h
  | some s => cases h; rfl

private theorem wrZR_pc (s : St) (w i v : Nat) : (s.wrZR w i v).pc = s.pc := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_pc (s : St) (w i v : Nat) : (s.wrSP w i v).pc = s.pc := by
  unfold St.wrSP; split <;> rfl

private theorem applySys_pc {s1 t : St} {res : Option SysRes} (h : applySys s1 res = .ok t) :
    t.pc = s1.pc := by
  cases res with
  | none => cases h
  | some sr =>
    cases sr with
    | ret _ _ _ => cases h; rfl
    | exit _ => cases h

private theorem execSvc_pc {s t : St} {nx : Nat} (h : execSvc s nx = .ok t) : t.pc = nx := by
  unfold execSvc at h
  dsimp only at h
  split at h
  all_goals (try cases h)
  all_goals (try (split at h <;> first | exact applySys_pc h | cases h))

/-- Executing any direct instruction decoded from a 32-bit word at a 4-byte-aligned
PC always retires to a 4-byte-aligned PC. -/
theorem exec_direct_pc_aligned {w : Nat} {ins : Ins} {s t : St}
    (hd : decodeWord w = some ins) (hdir : ins.isDirect = true)
    (hpc : s.pc % 4 = 0) (hex : exec ins 4 s = .ok t) : t.pc % 4 = 0 := by
  have hoff := decodeWord_alignedOff hd
  have hnx : ((s.pc + 4) % W) % 4 = 0 := add_mod4_W hpc (by decide)
  cases ins with
  | movWide kind ww rd imm16 hw =>
    unfold exec at hex; dsimp only at hex
    cases kind with
    | zero => cases hex; rw [wrZR_pc]; exact hnx
    | succ k =>
      cases k with
      | zero => cases hex
      | succ k =>
        cases k with
        | zero => cases hex; rw [wrZR_pc]; exact hnx
        | succ k =>
          cases k with
          | zero =>
            have h1 := lift_ok_pc hex
            cases hr : s.rdZR ww rd with
            | none => rw [hr] at h1; cases h1
            | some v => rw [hr] at h1; cases h1; rw [wrZR_pc]; exact hnx
          | succ _ => cases hex
  | addSubImm isSub setFlags ww rd rn imm =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdSP ww rn with
    | none => rw [hr] at h1; cases h1
    | some a =>
      rw [hr] at h1; simp only [bind, Option.bind, pure] at h1
      cases setFlags <;> cases h1
      · rw [wrSP_pc]; exact hnx
      · rw [wrZR_pc]; exact hnx
  | addSubReg isSub setFlags ww rd rn rm sh amt =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr1 : s.rdZR ww rn <;> cases hr2 : s.rdZR ww rm <;> rw [hr1, hr2] at h1 <;>
      simp only [bind, Option.bind, pure] at h1 <;> try cases h1
    cases setFlags <;> rw [wrZR_pc] <;> exact hnx
  | logicReg op inv ww rd rn rm sh amt =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr1 : s.rdZR ww rn <;> cases hr2 : s.rdZR ww rm <;> rw [hr1, hr2] at h1 <;>
      simp only [bind, Option.bind, pure] at h1 <;> try cases h1
    cases op <;> cases h1 <;> rw [wrZR_pc] <;> exact hnx
  | bitfield kind ww rd rn immr imms =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr1 : s.rdZR ww rn with
    | none => rw [hr1] at h1; cases h1
    | some src =>
      rw [hr1] at h1; simp only [bind, Option.bind] at h1
      cases kind with
      | sbfm | ubfm => cases h1; rw [wrZR_pc]; exact hnx
      | bfm =>
        cases hr2 : s.rdZR ww rd with
        | none => rw [hr2] at h1; cases h1
        | some dst => rw [hr2] at h1; cases h1; rw [wrZR_pc]; exact hnx
  | adr isPage rd imm =>
    unfold exec at hex; dsimp only at hex
    cases hex; rw [wrZR_pc]; exact hnx
  | ccmpReg isSub ww rn rm nzcv cc =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr1 : s.rdZR ww rn <;> cases hr2 : s.rdZR ww rm <;> cases hc : cond s.fl cc <;>
      rw [hr1, hr2, hc] at h1 <;> simp only [bind, Option.bind, pure] at h1 <;> try cases h1
    exact hnx
  | csel op ww rd rn rm cc =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr1 : s.rdZR ww rn <;> cases hr2 : s.rdZR ww rm <;> cases hc : cond s.fl cc <;>
      rw [hr1, hr2, hc] at h1 <;> simp only [bind, Option.bind, pure] at h1 <;> try cases h1
    rw [wrZR_pc]; exact hnx
  | ldr sz rt rn off | ldur sz rt rn off =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdSP 64 rn with
    | none => rw [hr] at h1; cases h1
    | some base =>
      rw [hr] at h1; simp only [bind, Option.bind] at h1
      split at h1
      · cases h1
      · cases hm : rdMem s.mem ((base + off) % W) sz with
        | none => rw [hm] at h1; cases h1
        | some v => rw [hm] at h1; cases h1; rw [wrZR_pc]; exact hnx
  | str sz rt rn off | stur sz rt rn off =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdSP 64 rn with
    | none => rw [hr] at h1; cases h1
    | some base =>
      rw [hr] at h1; simp only [bind, Option.bind] at h1
      split at h1
      · cases h1
      · cases hv : s.rdZR sz.width rt with
        | none => rw [hv] at h1; cases h1
        | some v =>
          rw [hv] at h1; dsimp only at h1
          cases hm : wrMem s.mem ((base + off) % W) v sz with
          | none => rw [hm] at h1; cases h1
          | some m' => rw [hm] at h1; cases h1; exact hnx
  | ldrReg sz rt rn rm ext scaled =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdSP 64 rn with
    | none => rw [hr] at h1; cases h1
    | some base =>
      rw [hr] at h1; simp only [bind, Option.bind] at h1
      split at h1
      · cases h1
      · cases hi : s.rdZR 64 rm with
        | none => rw [hi] at h1; cases h1
        | some idxRaw =>
          rw [hi] at h1; dsimp only at h1
          cases hm : rdMem s.mem ((base + extendIdx ext idxRaw * pow2 (if scaled = true then sz.log2 else 0) % W) % W) sz with
          | none => rw [hm] at h1; cases h1
          | some v => rw [hm] at h1; cases h1; rw [wrZR_pc]; exact hnx
  | strReg sz rt rn rm ext scaled =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdSP 64 rn with
    | none => rw [hr] at h1; cases h1
    | some base =>
      rw [hr] at h1; simp only [bind, Option.bind] at h1
      split at h1
      · cases h1
      · cases hi : s.rdZR 64 rm with
        | none => rw [hi] at h1; cases h1
        | some idxRaw =>
          rw [hi] at h1; dsimp only at h1
          cases hv : s.rdZR sz.width rt with
          | none => rw [hv] at h1; cases h1
          | some v =>
            rw [hv] at h1; dsimp only at h1
            cases hm : wrMem s.mem ((base + extendIdx ext idxRaw * pow2 (if scaled = true then sz.log2 else 0) % W) % W) v sz with
            | none => rw [hm] at h1; cases h1
            | some m' => rw [hm] at h1; cases h1; exact hnx
  | b link off =>
    unfold exec at hex; dsimp only at hex
    cases hex; exact add_mod4_W hpc hoff
  | bcond cc off =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hc : cond s.fl cc with
    | none => rw [hc] at h1; cases h1
    | some c =>
      rw [hc] at h1; simp only [bind, Option.bind, pure] at h1
      cases c <;> cases h1
      · exact hnx
      · exact add_mod4_W hpc hoff
  | cbz nz ww rt off =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdZR ww rt with
    | none => rw [hr] at h1; cases h1
    | some v =>
      rw [hr] at h1; simp only [bind, Option.bind, pure] at h1
      cases ht : (if nz = true then v != 0 else v == 0) <;> rw [ht] at h1 <;> cases h1
      · exact hnx
      · exact add_mod4_W hpc hoff
  | tbz nz bit rt off =>
    unfold exec at hex; dsimp only at hex
    have h1 := lift_ok_pc hex
    cases hr : s.rdZR 64 rt with
    | none => rw [hr] at h1; cases h1
    | some v =>
      rw [hr] at h1; simp only [bind, Option.bind, pure] at h1
      cases ht : (if nz = true then (v / pow2 bit % 2 == 1) else !(v / pow2 bit % 2 == 1)) <;>
        rw [ht] at h1 <;> cases h1
      · exact hnx
      · exact add_mod4_W hpc hoff
  | br _ _ => simp [Ins.isDirect] at hdir
  | ret _ => simp [Ins.isDirect] at hdir
  | svc imm =>
    unfold exec at hex; dsimp only at hex
    split at hex
    · cases hex
    · rw [execSvc_pc hex]; exact hnx

/-- **Fixed-Width 32-Bit Non-Desynchronization Theorem.**
Under ANY Hamming-1 or Hamming-k bit perturbation `ks` of a 4-byte-aligned
instruction word at `pc`:
  1. Decoding depends strictly on `[pc, pc + 4)` and NEVER reads `pc + 4` or beyond:
     any byte stream `b₂` agreeing with `b₁` on `0..3` (including the stream
     truncated to `none` at `i ≥ 4`) decodes identically under `flipBits _ ks`.
  2. No perturbed instruction can ever decode with length `≠ 4`.
  3. Fall-through and every downstream aligned instruction boundary `pc + 4j` remain
     strictly 4-byte aligned (zero cascading instruction-boundary desynchronization,
     unlike variable-length x86-64).
  4. Every retiring direct instruction preserves `pc % 4 = 0`. -/
theorem fetch_fault_fixed_boundary {b₁ b₂ : Nat → Option Nat} {pc : Nat}
    (hagree : ∀ i < 4, b₁ i = b₂ i) (hpc : pc % 4 = 0) (ks : List Nat) :
    decode (flipBits b₁ ks) = decode (flipBits b₂ ks) ∧
    (∀ ins len, decode (flipBits b₁ ks) = some (ins, len) →
      len = 4 ∧
      (pc + len) % W = (pc + 4) % W ∧
      ((pc + len) % W) % 4 = 0 ∧
      (∀ j : Nat, pc + len + 4 * j = pc + 4 * (j + 1) ∧ (pc + 4 * j) % 4 = 0) ∧
      (∀ s t, ins.isDirect = true → s.pc = pc → exec ins len s = .ok t → t.pc % 4 = 0)) := by
  refine ⟨decode_reads_only_4_bytes (flipBits_local hagree ks), ?_⟩
  intro ins len hdec
  obtain ⟨w, hdw, rfl⟩ := decode_exists_word hdec
  refine ⟨rfl, rfl, add_mod4_W hpc (by decide), ?_, ?_⟩
  · intro j; constructor <;> omega
  · intro s t hdir hspc hex
    exact exec_direct_pc_aligned hdw hdir (hspc ▸ hpc) hex

/-- Corollary: truncating the fetch stream immediately after `[pc, pc + 4)` (e.g.,
a `PROT_NONE` guard page starting at `pc + 4`) never changes any Hamming-1 or
Hamming-k fetch outcome. -/
theorem fetch_fault_guard_page_isolated (b : Nat → Option Nat) (pc : Nat)
    (hpc : pc % 4 = 0) (ks : List Nat) :
    decode (flipBits b ks) = decode (flipBits (fun i => if i < 4 then b i else none) ks) :=
  (fetch_fault_fixed_boundary (fun i hi => by simp [hi]) hpc ks).1

/-- Corollary: a 4-byte-aligned address strictly before a 4 KiB page boundary
`pageEnd` NEVER straddles `pageEnd` (`pc + 4 ≤ pageEnd`), eliminating the
x86-64 variable-length page-straddle `#PF` window entirely. -/
theorem aligned_never_straddles_page {pc pageEnd : Nat}
    (hpc : pc % 4 = 0) (hpage : pageEnd % AArch64Env.page = 0) (hlt : pc < pageEnd) :
    pc + 4 ≤ pageEnd := by
  have hp4 : pageEnd % 4 = 0 := by
    unfold AArch64Env.page at hpage
    omega
  omega

private theorem testBit_add_256 (x y j : Nat) (hx : x < 256) :
    Nat.testBit (x + 256 * y) j = if j < 8 then Nat.testBit x j else Nat.testBit y (j - 8) := by
  simp only [Nat.testBit, Nat.shiftRight_eq_div_pow, Nat.one_and_eq_mod_two]
  by_cases hj : j < 8
  · rw [if_pos hj]
    have h2 : 256 = 2 ^ j * 2 ^ (8 - j) := by
      have he : j + (8 - j) = 8 := by omega
      rw [← Nat.pow_add, he]
    have h2e : 2 ^ (8 - j) = 2 * 2 ^ (7 - j) := by
      have he : 8 - j = (7 - j) + 1 := by omega
      rw [he, Nat.pow_succ, Nat.mul_comm]
    rw [h2, Nat.mul_assoc, Nat.add_mul_div_left _ _ (Nat.pow_pos (by decide)),
      h2e, Nat.mul_assoc, Nat.add_mul_mod_self_left]
  · rw [if_neg hj]
    have h2 : 2 ^ j = 256 * 2 ^ (j - 8) := by
      have h8 : (256 : Nat) = 2 ^ 8 := rfl
      rw [h8, ← Nat.pow_add, show 8 + (j - 8) = j by omega]
    rw [h2, ← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ (by decide),
      Nat.div_eq_of_lt hx, Nat.zero_add]

private theorem testBit_pack4 (a0 a1 a2 a3 : Nat)
    (ha0 : a0 < 256) (ha1 : a1 < 256) (ha2 : a2 < 256) (ha3 : a3 < 256) (i : Nat) :
    Nat.testBit (a0 + 256 * (a1 + 256 * (a2 + 256 * a3))) i =
      if i < 8 then Nat.testBit a0 i
      else if i < 16 then Nat.testBit a1 (i - 8)
      else if i < 24 then Nat.testBit a2 (i - 16)
      else if i < 32 then Nat.testBit a3 (i - 24)
      else false := by
  rw [testBit_add_256 a0 _ i ha0]
  by_cases h8 : i < 8
  · rw [if_pos h8, if_pos h8]
  · rw [if_neg h8, if_neg h8, testBit_add_256 a1 _ (i - 8) ha1]
    by_cases h16 : i < 16
    · rw [if_pos (by omega), if_pos h16]
    · rw [if_neg (by omega), if_neg h16, show i - 8 - 8 = i - 16 by omega,
        testBit_add_256 a2 _ (i - 16) ha2]
      by_cases h24 : i < 24
      · rw [if_pos (by omega), if_pos h24]
      · rw [if_neg (by omega), if_neg h24, show i - 16 - 8 = i - 24 by omega]
        by_cases h32 : i < 32
        · rw [if_pos h32]
        · rw [if_neg h32]
          have hpow : 2 ^ 8 ≤ 2 ^ (i - 24) := Nat.pow_le_pow_right (by decide) (by omega)
          exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le ha3 hpow)

private theorem testBit_nat_xor (a b i : Nat) :
    Nat.testBit (Nat.xor a b) i = (Nat.testBit a i ^^ Nat.testBit b i) :=
  Nat.testBit_xor a b i

/-- Byte-packing correspondence for Hamming-1 fetch neighbours: `altsAt` over four
bytes `< 256` matches `altsWord` on the packed 32-bit word. -/
private theorem xor_byte_word (b0 b1 b2 b3 k : Nat)
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) (hk : k < 32) :
    let b : Nat → Option Nat := fun i =>
      match i with | 0 => some b0 | 1 => some b1 | 2 => some b2 | 3 => some b3 | _ => none
    le (flipBit b k) 0 4 =
      some (Nat.xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) (2 ^ k)) := by
  dsimp only
  simp only [le, flipBit, bind, Option.bind, pure]
  have hbyte (x m : Nat) (hx : x < 256) (hm : m < 8) : Nat.xor x (2 ^ m) < 256 :=
    Nat.xor_lt_two_pow hx (Nat.pow_lt_pow_right (by decide) hm)
  have hcase : k / 8 = 0 ∨ k / 8 = 1 ∨ k / 8 = 2 ∨ k / 8 = 3 := by omega
  rcases hcase with hk0 | hk1 | hk2 | hk3
  · simp only [hk0, beq_self_eq_true, ↓reduceIte,
      show (1 == 0) = false from rfl, show (2 == 0) = false from rfl, show (3 == 0) = false from rfl,
      Bool.false_eq_true, Option.map_some, Nat.mul_zero, Nat.add_zero]
    congr 1; apply Nat.eq_of_testBit_eq; intro i
    rw [testBit_pack4 _ b1 b2 b3 (hbyte b0 (k % 8) h0 (by omega)) h1 h2 h3 i,
      testBit_nat_xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))),
      testBit_pack4 b0 b1 b2 b3 h0 h1 h2 h3 i, Nat.testBit_two_pow]
    by_cases hi : i < 8
    · simp [hi, Nat.testBit_two_pow, show k % 8 = k by omega]
    · simp [hi, show k ≠ i by omega]
  · simp only [hk1, beq_self_eq_true, ↓reduceIte,
      show (0 == 1) = false from rfl, show (2 == 1) = false from rfl, show (3 == 1) = false from rfl,
      Bool.false_eq_true, Option.map_some, Nat.mul_zero, Nat.add_zero]
    congr 1; apply Nat.eq_of_testBit_eq; intro i
    rw [testBit_pack4 b0 _ b2 b3 h0 (hbyte b1 (k % 8) h1 (by omega)) h2 h3 i,
      testBit_nat_xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))),
      testBit_pack4 b0 b1 b2 b3 h0 h1 h2 h3 i, Nat.testBit_two_pow]
    by_cases hi8 : i < 8
    · simp [hi8, show k ≠ i by omega]
    · by_cases hi16 : i < 16
      · simp [hi8, hi16, Nat.testBit_two_pow]; omega
      · simp [hi8, hi16, show k ≠ i by omega]
  · simp only [hk2, beq_self_eq_true, ↓reduceIte,
      show (0 == 2) = false from rfl, show (1 == 2) = false from rfl, show (3 == 2) = false from rfl,
      Bool.false_eq_true, Option.map_some, Nat.mul_zero, Nat.add_zero]
    congr 1; apply Nat.eq_of_testBit_eq; intro i
    rw [testBit_pack4 b0 b1 _ b3 h0 h1 (hbyte b2 (k % 8) h2 (by omega)) h3 i,
      testBit_nat_xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))),
      testBit_pack4 b0 b1 b2 b3 h0 h1 h2 h3 i, Nat.testBit_two_pow]
    by_cases hi8 : i < 8
    · simp [hi8, show k ≠ i by omega]
    · by_cases hi16 : i < 16
      · simp [hi8, hi16, show k ≠ i by omega]
      · by_cases hi24 : i < 24
        · simp [hi8, hi16, hi24, Nat.testBit_two_pow]; omega
        · simp [hi8, hi16, hi24, show k ≠ i by omega]
  · simp only [hk3, beq_self_eq_true, ↓reduceIte,
      show (0 == 3) = false from rfl, show (1 == 3) = false from rfl, show (2 == 3) = false from rfl,
      Bool.false_eq_true, Option.map_some, Nat.mul_zero, Nat.add_zero]
    congr 1; apply Nat.eq_of_testBit_eq; intro i
    rw [testBit_pack4 b0 b1 b2 _ h0 h1 h2 (hbyte b3 (k % 8) h3 (by omega)) i,
      testBit_nat_xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))),
      testBit_pack4 b0 b1 b2 b3 h0 h1 h2 h3 i, Nat.testBit_two_pow]
    by_cases hi8 : i < 8
    · simp [hi8, show k ≠ i by omega]
    · by_cases hi16 : i < 16
      · simp [hi8, hi16, show k ≠ i by omega]
      · by_cases hi24 : i < 24
        · simp [hi8, hi16, hi24, show k ≠ i by omega]
        · by_cases hi32 : i < 32
          · simp [hi8, hi16, hi24, hi32, Nat.testBit_two_pow]; omega
          · simp [hi8, hi16, hi24, hi32, show k ≠ i by omega]

theorem altsAt_eq_altsWord {b : Nat → Option Nat} {b0 b1 b2 b3 : Nat}
    (hb0 : b 0 = some b0) (hb1 : b 1 = some b1) (hb2 : b 2 = some b2) (hb3 : b 3 = some b3)
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    altsAt b = altsWord (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) := by
  unfold altsAt altsWord
  apply List.map_congr_left
  intro k hk
  simp only [List.mem_range] at hk
  let b' : Nat → Option Nat := fun i =>
    match i with | 0 => some b0 | 1 => some b1 | 2 => some b2 | 3 => some b3 | _ => none
  have hag : ∀ i < 4, b i = b' i := by
    intro i hi
    match i with
    | 0 => exact hb0
    | 1 => exact hb1
    | 2 => exact hb2
    | 3 => exact hb3
    | _ + 4 => omega
  rw [decode_reads_only_4_bytes (flipBit_local hag k)]
  unfold decode
  rw [xor_byte_word b0 b1 b2 b3 k h0 h1 h2 h3 hk]
  simp only [bind, Option.bind, pure]
  cases decodeWord (Nat.xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) (2 ^ k)) <;> rfl

/-! ### Kernel-checked witnesses -/

/-- Little-endian 4-byte list for a 32-bit instruction word. -/
def wordBytes (w : Nat) : List Nat :=
  [w % 256, (w / 256) % 256, (w / 65536) % 256, (w / 16777216) % 256]

def wordsToBytes (ws : List Nat) : List Nat := (ws.map wordBytes).flatten

/-- Witness program: computes `7` via `movz`/`add`/`lsl`/`lsr`/`csel`, exits with code 7. -/
def demoWords : List Nat :=
  [ 0xd2800060  -- 0x00: movz x0, #3
  , 0x91001000  -- 0x04: add  x0, x0, #4         (x0 = 7)
  , 0xd2800ba8  -- 0x08: movz x8, #93            (__NR_exit)
  , 0xd4000001  -- 0x0c: svc  #0
  ]

def demoCode : List Nat := wordsToBytes demoWords
def demoBase : Nat := 0x400000
def demoIm : Img := imageOf demoCode demoBase
def demoAlts : Alts := imageAlts demoCode demoBase

def k0 : Kern := ⟨[], [], [], [], []⟩
def s0 : St := ⟨List.replicate 31 (some 0), some 0x10000, demoBase, Flags.undef, [], k0⟩

def Out.code : Out → Option Nat
  | .exit c => some c
  | _ => none

theorem witness : (run demoIm 4 (.ok s0)).code = some 7 := by decide +kernel

theorem witness_phys (r : Rates) (hr : r.ok)
    (hf : p7.mem r.fetch) (hg : p7.mem r.reg) (hm : p7.mem r.mem) (hl : p7.mem r.nzcv) :
    (prun demoAlts (r.state hr) r.fetch demoIm 4 (.ok s0)).ok ∧
    fm (prun demoAlts (r.state hr) r.fetch demoIm 4 (.ok s0)) ≤ 4 * (4 / 5000000) ∧
    faultFree (prun demoAlts (r.state hr) r.fetch demoIm 4 (.ok s0)) = [run demoIm 4 (.ok s0)] := by
  refine ⟨prun_ok _ _ hr.1 _ _ _, ?_, faultFree_prun _ _ _ _ _ _⟩
  have := baseline demoAlts demoIm p7 p7 p7 p7 r hr hf hg hm hl 4 (.ok s0)
  refine Rat.le_trans this ?_
  decide +kernel

end AArch64Phys
