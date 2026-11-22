/-
X86DecFuzzModel: the Lean model side of the `X86Dec.altsAt` Hamming-1/2 and
adversarial byte-boundary differential fuzzer against the host x86-64 CPU.

Modes:
  1. `gen <valid_seeds> <adv_count> <seed> <cases_file>`
     Generates:
       (a) `<valid_seeds>` (default 5,000) random valid stage-1 instructions
           covering all 45 non-syscall forms plus `.syscall`,
       (b) All Hamming-1 bit-flip neighbours via `X86Dec.altsAt` (~180,000+
           candidates), verifying 100% agreement between `X86Dec.altsAt` and
           per-bit `X86Dec.decode (X86Dec.flipBit b k) insAddr` as well as
           explicit byte-flipped array decoding,
       (c) `10 * <valid_seeds>` (default 50,000) Hamming-2 bit-flip candidates,
       (d) `<adv_count>` (default 100,000) adversarial and random 1..15-byte
           sequences (covering REX `0x40..0x4f` before all opcodes, ModRM
           `rm=4/5`, SIB `idx=4/base=5`, `88/89/8b/8d/0f/ff/c1/81/83`, and
           invalid opcode/ModRM extensions).
     For every byte sequence accepted by `X86Dec.decode` (excluding `.syscall`
     `0f 05` and parity `jcc` `cc = 10, 11` where `X86Int.cond` is `none`),
     constructs a safe deterministic pre-state at `insAddr = page0End - len`
     (`page0End = 0x100000000`), runs `X86Int.step`, and emits the raw
     `len`-byte instruction + pre-state metadata + expected post-state to
     `<cases_file>` for `bootstrap/model/x86dec_fuzz_cpu`.

  2. `verify <cpu_report_file> <stats_file>`
     Reads the execution report from `x86dec_fuzz_cpu`, verifies that every
     accepted instruction executed on the CPU right against the `PROT_NONE`
     guard page at `0x100000000` with `0` guard-page over-reads, `0` length
     mismatches, and `0` state disagreements across all 16 GPRs, `CF/SF/ZF/OF`
     flags, `rip` delta, and the 64-byte memory window `[0x117e0, 0x11820)`.
-/
import Bootstrap.X86Dec

open X86Env X86Int X86Dec

def page0End : Nat := 0x100000000
def spInt : Nat := 0x11800
def winLo : Nat := 0x117e0
def winSize : Nat := 64
def poolSize : Nat := 32

def hexDigit (n : Nat) : Char :=
  match n with
  | 0 => '0' | 1 => '1' | 2 => '2' | 3 => '3'
  | 4 => '4' | 5 => '5' | 6 => '6' | 7 => '7'
  | 8 => '8' | 9 => '9' | 10 => 'a' | 11 => 'b'
  | 12 => 'c' | 13 => 'd' | 14 => 'e' | _ => 'f'

def hex2Tbl : Array String :=
  Array.ofFn fun (i : Fin 256) =>
    String.ofList [hexDigit (i.1 / 16), hexDigit (i.1 % 16)]

def hex2 (v : Nat) : String :=
  hex2Tbl[v % 256]!

def hex16 (v : Nat) : String :=
  let u := UInt64.ofNat v
  let b7 := hex2Tbl[((u >>> 56) &&& 0xff).toNat]!
  let b6 := hex2Tbl[((u >>> 48) &&& 0xff).toNat]!
  let b5 := hex2Tbl[((u >>> 40) &&& 0xff).toNat]!
  let b4 := hex2Tbl[((u >>> 32) &&& 0xff).toNat]!
  let b3 := hex2Tbl[((u >>> 24) &&& 0xff).toNat]!
  let b2 := hex2Tbl[((u >>> 16) &&& 0xff).toNat]!
  let b1 := hex2Tbl[((u >>> 8) &&& 0xff).toNat]!
  let b0 := hex2Tbl[(u &&& 0xff).toNat]!
  b7 ++ b6 ++ b5 ++ b4 ++ b3 ++ b2 ++ b1 ++ b0

def flagCh : Option Bool → Char
  | none => '-'
  | some true => '1'
  | some false => '0'

def xsStep (x : UInt64) : UInt64 :=
  let x1 := x ^^^ (x <<< 13)
  let x2 := x1 ^^^ (x1 >>> 7)
  x2 ^^^ (x2 <<< 17)

structure BaseState where
  regs : Array Nat
  cf : Bool
  sf : Bool
  zf : Bool
  of : Bool
  mem : List Nat
deriving Inhabited

/-- Construct deterministic base state `idx ∈ 0..31` from `seed` identically to C++. -/
def mkBaseState (seed : UInt64) (idx : Nat) : BaseState := Id.run do
  let mut st := xsStep (0x9e3779b97f4a7c15 ^^^ seed ^^^ UInt64.ofNat (idx * 0x10001 + 1))
  let mut regs : Array Nat := Array.mkEmpty 16
  for r in [:16] do
    st := xsStep st
    let v0 := st
    let m1 : UInt64 := 0xffffffffffff00ff
    let m2 : UInt64 := 0xffffffffffffff00
    let v : UInt64 :=
      if r == 4 then UInt64.ofNat spInt
      else if r < 4 then
        (v0 &&& m1) ||| (UInt64.ofNat (0xa1 + r * 0x11) <<< 8)
      else if r < 8 then
        (v0 &&& m2) ||| UInt64.ofNat (0x10 + r)
      else v0
    regs := regs.push v.toNat
  st := xsStep st
  let cf := (st &&& 1) != 0
  let sf := ((st >>> 1) &&& 1) != 0
  let zf := ((st >>> 2) &&& 1) != 0
  let of := ((st >>> 3) &&& 1) != 0
  let mut memRev : List Nat := []
  for _ in [:winSize] do
    st := xsStep st
    memRev := (st &&& 0xff).toNat :: memRev
  return { regs := regs, cf := cf, sf := sf, zf := zf, of := of, mem := memRev.reverse }

def setLe64 (m : List Nat) (off v : Nat) : List Nat :=
  let bs := bytes8 v
  (List.range 8).foldl (fun acc i => acc.set (off + i) (bs[i]?.getD 0)) m

def validCcs : Array Nat := #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]

/-- Emit a valid memory operand `[base + disp]` biased towards 32-bit displacements
so that valid seed instructions have >= 36 bits on average (~180k+ `altsAt` bits). -/
def emitValidMem (st0 : UInt64) (regField base : Nat) : UInt64 × List Nat := Id.run do
  let mut st := xsStep st0
  let rMd := (st % 10).toNat
  let mut md := if rMd < 6 then 2 else if rMd < 8 then 1 else 0
  if md == 0 && (base % 8) == 5 then md := 2
  st := xsStep st
  let useSib := (base % 8) == 4 || (st % 3 == 0)
  let mByte := md * 64 + (regField % 8) * 8 + (if useSib then 4 else base % 8)
  let mut out : List Nat := [mByte]
  if useSib then
    st := xsStep st
    let scale := (st % 4).toNat
    out := out ++ [scale * 64 + 4 * 8 + (base % 8)]
  if md == 1 then
    st := xsStep st
    out := out ++ [(st &&& 0xff).toNat]
  else if md == 2 then
    for _ in [:4] do
      st := xsStep st
      out := out ++ [(st &&& 0xff).toNat]
  return (st, out)

/-- Generate one valid stage-1 instruction of form `formIdx ∈ 0..45` (45 = syscall). -/
def genValidIns (st0 : UInt64) (formIdx : Nat) : UInt64 × List Nat := Id.run do
  let mut st := xsStep st0
  let r := (st % 16).toNat
  st := xsStep st
  let b := (st % 16).toNat
  let mkRex (w : Bool) (rr bb : Nat) (force : Bool) : UInt64 × List Nat := Id.run do
    let x := 0x40 + (if w then 8 else 0) + (if rr >= 8 then 4 else 0) + (if bb >= 8 then 1 else 0)
    let st' := xsStep st
    if x != 0x40 || force || (st' % 3 == 0) then return (st', [x]) else return (st', [])
  let mkImm (n : Nat) (sIn : UInt64) : UInt64 × List Nat := Id.run do
    let mut s := sIn
    let mut acc : List Nat := []
    for _ in [:n] do
      s := xsStep s
      acc := acc ++ [(s &&& 0xff).toNat]
    return (s, acc)
  match formIdx % 46 with
  | 0 =>
    let (s1, rx) := mkRex false 0 b false
    let (s2, im) := mkImm 4 s1
    return (s2, rx ++ [0xb8 + (b % 8)] ++ im)
  | 1 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 8 s1
    return (s2, rx ++ [0xb8 + (b % 8)] ++ im)
  | 2 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x01, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 3 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x29, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 4 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x21, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 5 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x09, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 6 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x31, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 7 =>
    let (s1, rx) := mkRex false r r false
    return (s1, rx ++ [0x31, 0xc0 + (r % 8) * 8 + (r % 8)])
  | 8 =>
    let (s1, rx) := mkRex false r b false
    return (s1, rx ++ [0x89, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 9 =>
    let (s1, rx) := mkRex false r b false
    return (s1, rx ++ [0x8b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 10 =>
    let (s1, rx) := mkRex false r b false
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8b] ++ mem)
  | 11 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x89, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 12 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x8b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 13 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x89] ++ mem)
  | 14 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8b] ++ mem)
  | 15 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x85, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 16 =>
    let (s1, im) := mkImm 1 st
    return (s1, [0x6a] ++ im)
  | 17 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x39, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 18 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x3b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 19 | 20 | 21 | 22 | 23 | 24 =>
    let extTbl : Array Nat := #[0, 1, 4, 5, 6, 7]
    let ext := extTbl[(formIdx % 46) - 19]!
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0x83, 0xc0 + ext * 8 + (b % 8)] ++ im)
  | 25 | 26 | 27 | 28 | 29 | 30 =>
    let extTbl : Array Nat := #[0, 1, 4, 5, 6, 7]
    let ext := extTbl[(formIdx % 46) - 25]!
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 4 s1
    return (s2, rx ++ [0x81, 0xc0 + ext * 8 + (b % 8)] ++ im)
  | 31 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0xc1, 0xe0 + (b % 8)] ++ im)
  | 32 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0xc1, 0xe8 + (b % 8)] ++ im)
  | 33 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8d] ++ mem)
  | 34 =>
    let (s1, rx) := mkRex (r % 2 == 0) r b (r >= 4)
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x88] ++ mem)
  | 35 =>
    let (s1, rx) := mkRex (r % 2 == 0) r b false
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x0f, 0xb6] ++ mem)
  | 36 =>
    let cc := validCcs[(r % 14)]!
    let (s1, im) := mkImm 4 st
    return (s1, [0x0f, 0x80 + cc] ++ im)
  | 37 =>
    let cc := validCcs[(r % 14)]!
    let (s1, im) := mkImm 1 st
    return (s1, [0x70 + cc] ++ im)
  | 38 =>
    let (s1, im) := mkImm 1 st
    return (s1, [0xeb] ++ im)
  | 39 =>
    let (s1, im) := mkImm 4 st
    return (s1, [0xe9] ++ im)
  | 40 =>
    let (s1, im) := mkImm 4 st
    return (s1, [0xe8] ++ im)
  | 41 =>
    return (st, [0xc3])
  | 42 =>
    let (s1, rx) := mkRex false 0 b false
    return (s1, rx ++ [0x50 + (b % 8)])
  | 43 =>
    let (s1, rx) := mkRex false 0 b false
    return (s1, rx ++ [0x58 + (b % 8)])
  | 44 =>
    let idx := if r == 4 then 5 else r
    let mut md := if (b % 3) == 0 then 0 else if (b % 3) == 1 then 1 else 2
    if md == 0 && (b % 8) == 5 then md := 2
    let rx := (if idx >= 8 then 2 else 0) + (if b >= 8 then 1 else 0)
    let pfx := if rx != 0 || (idx % 2 == 0) then [0x40 + rx] else []
    let mByte := md * 64 + 4 * 8 + 4
    let sByte := 3 * 64 + (idx % 8) * 8 + (b % 8)
    let (s1, dispBs) := if md == 0 then (st, []) else if md == 1 then mkImm 1 st else mkImm 4 st
    return (s1, pfx ++ [0xff, mByte, sByte] ++ dispBs)
  | _ =>
    return (st, [0x0f, 0x05])

/-- Pad a valid instruction to a 15-byte code window so `altsAt` bit-flips that
lengthen an instruction (or decoder length mutations) have trailing bytes. -/
def padTo15 (st0 : UInt64) (bs : List Nat) : UInt64 × Array Nat := Id.run do
  let mut st := st0
  let mut arr := bs.toArray
  while arr.size < 15 do
    st := xsStep st
    arr := arr.push (st &&& 0xff).toNat
  return (st, arr)

/-- Generate one adversarial / random 1..15 byte sequence covering decoder edge cases. -/
def genAdversarial (st0 : UInt64) (idx : Nat) : UInt64 × Array Nat := Id.run do
  let mut st := xsStep st0
  let fam := idx % 10
  let randBytes (n : Nat) (sIn : UInt64) : UInt64 × List Nat := Id.run do
    let mut s := sIn
    let mut acc : List Nat := []
    for _ in [:n] do
      s := xsStep s
      acc := acc ++ [(s &&& 0xff).toNat]
    return (s, acc)
  match fam with
  | 0 =>
    -- 0x88 boundary: with and without REX, reg in 0..7 (testing !hasRex && reg >= 4)
    let useRex := (idx / 10) % 2 == 1
    let rex := 0x40 + ((idx / 20) % 16)
    let reg := (idx / 10) % 8
    let md := (st % 4).toNat
    let rm := ((st >>> 4) % 8).toNat
    let m := md * 64 + reg * 8 + rm
    let (s1, tail) := randBytes 6 st
    let bs := (if useRex then [rex] else []) ++ [0x88, m] ++ tail
    return (s1, bs.toArray)
  | 1 =>
    -- 0x89 / 0x8b boundary: REX 0x40..0x4f or none, md=0..3, rm=4/5, SIB
    let useRex := (idx / 10) % 3 != 0
    let rex := 0x40 + ((idx / 30) % 16)
    let op := if (idx / 10) % 2 == 0 then 0x89 else 0x8b
    let (s1, tail) := randBytes 8 st
    let bs := (if useRex then [rex] else []) ++ [op] ++ tail
    return (s1, bs.toArray)
  | 2 =>
    -- 0x8d (lea) + ModRM rm=4/5 + SIB idx=4/base=5
    let rex := 0x40 + ((idx / 10) % 16)
    let md := (st % 4).toNat
    let reg := ((st >>> 2) % 8).toNat
    let rm := if (st >>> 5) % 2 == 0 then 4 else 5
    let sib := (((st >>> 8) % 4).toNat) * 64 + (((st >>> 10) % 8).toNat) * 8 + (((st >>> 13) % 8).toNat)
    let (s1, tail) := randBytes 5 st
    let bs := (if (idx / 10) % 4 == 0 then [] else [rex]) ++ [0x8d, md * 64 + reg * 8 + rm, sib] ++ tail
    return (s1, bs.toArray)
  | 3 =>
    -- 0xff jump-table boundary: /4 vs other ext, SIB scale=0..3, idx=4, md=0 base=5
    let useRex := (idx / 10) % 2 == 0
    let rex := 0x40 + ((idx / 20) % 16)
    let md := (st % 4).toNat
    let ext := if (st >>> 2) % 2 == 0 then 4 else ((st >>> 3) % 8).toNat
    let rm := if (st >>> 6) % 4 != 0 then 4 else ((st >>> 8) % 8).toNat
    let scale := if (st >>> 11) % 2 == 0 then 3 else ((st >>> 12) % 4).toNat
    let sibIdx := ((st >>> 14) % 8).toNat
    let sibBase := ((st >>> 17) % 8).toNat
    let (s1, tail) := randBytes 5 st
    let bs := (if useRex then [rex] else []) ++ [0xff, md * 64 + ext * 8 + rm, scale * 64 + sibIdx * 8 + sibBase] ++ tail
    return (s1, bs.toArray)
  | 4 =>
    -- 0x0f two-byte opcodes: 05, b6, 80..8f, and random o2 with/without REX
    let useRex := (idx / 10) % 3 == 0
    let rex := 0x40 + ((idx / 30) % 16)
    let o2Choice := (st % 4).toNat
    let o2 := if o2Choice == 0 then 0x05 else if o2Choice == 1 then 0xb6 else if o2Choice == 2 then 0x80 + ((st >>> 4) % 16).toNat else ((st >>> 4) &&& 0xff).toNat
    let (s1, tail) := randBytes 7 st
    let bs := (if useRex then [rex] else []) ++ [0x0f, o2] ++ tail
    return (s1, bs.toArray)
  | 5 =>
    -- 0x81, 0x83, 0xc1 with valid/invalid extensions e=0..7 and REX combinations
    let rex := 0x40 + ((idx / 10) % 16)
    let opTbl : Array Nat := #[0x81, 0x83, 0xc1]
    let op := opTbl[(idx / 10) % 3]!
    let md := if (st % 4) == 0 then (st % 3).toNat else 3
    let ext := ((st >>> 3) % 8).toNat
    let rm := ((st >>> 6) % 8).toNat
    let (s1, tail) := randBytes 5 st
    return (s1, ([rex, op, md * 64 + ext * 8 + rm] ++ tail).toArray)
  | 6 =>
    -- ALU / test / cmp: 01, 29, 21, 09, 31, 85, 39, 3b with w=0/1, md=0..3
    let useRex := (idx / 10) % 4 != 0
    let rex := 0x40 + ((idx / 40) % 16)
    let opTbl : Array Nat := #[0x01, 0x29, 0x21, 0x09, 0x31, 0x85, 0x39, 0x3b]
    let op := opTbl[(idx / 10) % 8]!
    let (s1, tail) := randBytes 4 st
    let bs := (if useRex then [rex] else []) ++ [op] ++ tail
    return (s1, bs.toArray)
  | 7 =>
    -- b8..bf, 50..5f, 6a, 70..7f, eb, e9, e8, c3 with/without REX
    let useRex := (idx / 10) % 2 == 0
    let rex := 0x40 + ((idx / 20) % 16)
    let kind := (st % 6).toNat
    let op :=
      if kind == 0 then 0xb8 + ((st >>> 4) % 8).toNat
      else if kind == 1 then 0x50 + ((st >>> 4) % 16).toNat
      else if kind == 2 then 0x70 + ((st >>> 4) % 16).toNat
      else if kind == 3 then 0x6a
      else if kind == 4 then (if (st >>> 4) % 2 == 0 then 0xeb else 0xe9)
      else (if (st >>> 4) % 2 == 0 then 0xe8 else 0xc3)
    let (s1, tail) := randBytes 9 st
    let bs := (if useRex then [rex] else []) ++ [op] ++ tail
    return (s1, bs.toArray)
  | 8 =>
    -- Truncated short sequences of length 1..5
    let len := 1 + (idx % 5)
    let (s1, bs) := randBytes len st
    return (s1, bs.toArray)
  | _ =>
    -- Pure random 1..15 byte sequence
    let len := 1 + (idx % 15)
    let (s1, bs) := randBytes len st
    return (s1, bs.toArray)

structure FuzzStats where
  totalCandidates : Nat := 0
  validSeeds : Nat := 0
  altsAtVerified : Nat := 0
  altsAtMismatches : Nat := 0
  hamming2Tested : Nat := 0
  advTested : Nat := 0
  acceptedTested : Nat := 0
  acceptedSyscall : Nat := 0
  acceptedParityJcc : Nat := 0
  rejected : Nat := 0
  modelStuck : Nat := 0

/-- Evaluate one candidate byte array `bs`: decode with `X86Dec.decode`, and if
accepted (`!= .syscall`), compute pre-state overrides + `X86Int.step` post-state
and emit a line to `casesOut`. -/
def evalCandidate (pool : Array BaseState) (casesOut : IO.FS.Stream)
    (bs : Array Nat) (st : FuzzStats) : IO FuzzStats := do
  let st := { st with totalCandidates := st.totalCandidates + 1 }
  match decode (fun i => bs[i]?) 0 with
  | none =>
    return { st with rejected := st.rejected + 1 }
  | some (_, len) =>
    if len == 0 || len > bs.size then
      return { st with modelStuck := st.modelStuck + 1 }
    let insAddr := page0End - len
    let cand := (bs.extract 0 len).toList
    match decode (fun i => cand[i]?) insAddr with
    | none =>
      return { st with modelStuck := st.modelStuck + 1 }
    | some (ins, len') =>
      if len' != len then
        return { st with modelStuck := st.modelStuck + 1 }
      match ins with
      | .syscall =>
        return { st with acceptedSyscall := st.acceptedSyscall + 1 }
      | .jcc cc _ =>
        if cc == 10 || cc == 11 then
          -- Parity condition codes (jp/jnp) are unmodelled in Flags -> cond = none
          return { st with acceptedParityJcc := st.acceptedParityJcc + 1 }
        else
          let caseNum := st.acceptedTested
          let poolIdx := caseNum % poolSize
          let base := pool[poolIdx]!
          let flags : Flags := ⟨some base.cf, some base.zf, some base.sf, some base.of⟩
          let s : St := {
            regs := base.regs.toList.map some,
            rip := insAddr,
            fl := flags,
            mem := [⟨winLo, base.mem, true⟩],
            k := ⟨[], [], [], [], []⟩
          }
          match step (imageOf cand insAddr) s with
          | .ok t =>
            let delta := (t.rip + W - insAddr) % W
            let candHex := String.join (cand.map hex2)
            let flStr := String.ofList [flagCh t.fl.cf, flagCh t.fl.sf, flagCh t.fl.zf, flagCh t.fl.of]
            casesOut.putStrLn s!"{candHex} {len} {poolIdx} 16 0000000000000000 64 0000000000000000 1 | {hex16 delta} 16 0000000000000000 16 0000000000000000 {flStr} 64 0 0000000000000000"
            return { st with acceptedTested := st.acceptedTested + 1 }
          | _ =>
            return { st with modelStuck := st.modelStuck + 1 }
      | _ =>
        let caseNum := st.acceptedTested
        let poolIdx := caseNum % poolSize
        let base := pool[poolIdx]!
        let (fixRegIdx, fixRegVal, fixMemOff, fixMemVal, isBranch) : Nat × Nat × Nat × Nat × Nat :=
          match ins with
          | .ld _ b disp | .st b disp _ =>
            let off := (caseNum * 7) % 57
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .ld32 _ b disp =>
            let off := (caseNum * 7) % 61
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .ld8 _ b disp | .st8 b disp _ =>
            let off := (caseNum * 7) % 64
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .jtab b i disp =>
            let off := (caseNum % 7) * 8
            let ea := winLo + off
            let regIVal := base.regs[i]!
            let rVal :=
              if b == i then
                ((UInt64.ofNat ea - UInt64.ofNat (disp % W)) * 0x8e38e38e38e38e39).toNat
              else
                (UInt64.ofNat ea - 8 * UInt64.ofNat regIVal - UInt64.ofNat (disp % W)).toNat
            let tgt := 0x80000000 + ((caseNum * 0x12345 + 0x1000) % 0x70000000)
            (b, rVal, off, tgt, 1)
          | .ret =>
            let tgt := 0x80000000 + ((caseNum * 0x12345 + 0x1000) % 0x70000000)
            (16, 0, 32, tgt, 1)
          | .jmp _ | .call _ =>
            (16, 0, 64, 0, 1)
          | _ =>
            (16, 0, 64, 0, 0)
        let regsIn : Array Nat :=
          if fixRegIdx < 16 then base.regs.set! fixRegIdx fixRegVal else base.regs
        let memIn : List Nat :=
          if fixMemOff <= 56 then setLe64 base.mem fixMemOff fixMemVal else base.mem
        let flags : Flags := ⟨some base.cf, some base.zf, some base.sf, some base.of⟩
        let s : St := {
          regs := regsIn.toList.map some,
          rip := insAddr,
          fl := flags,
          mem := [⟨winLo, memIn, true⟩],
          k := ⟨[], [], [], [], []⟩
        }
        match step (imageOf cand insAddr) s with
        | .ok t =>
          let delta := (t.rip + W - insAddr) % W
          let mut mod1Idx : Nat := 16
          let mut mod1Val : Nat := 0
          let mut mod2Idx : Nat := 16
          let mut mod2Val : Nat := 0
          let tRegs := t.regs.toArray
          for r in [:16] do
            let vAfter := (tRegs[r]?.getD none).getD 0
            let vBefore := regsIn[r]!
            if vAfter != vBefore then
              if mod1Idx == 16 then
                mod1Idx := r
                mod1Val := vAfter
              else
                mod2Idx := r
                mod2Val := vAfter
          let (modMemOff, modMemLen, modMemVal) : Nat × Nat × Nat :=
            match ins with
            | .st b disp r =>
              let ea := (regsIn[b]! + disp) % W
              (ea - winLo, 8, regsIn[r]!)
            | .st8 b disp r =>
              let ea := (regsIn[b]! + disp) % W
              (ea - winLo, 1, regsIn[r]! % 256)
            | .pushI imm =>
              (24, 8, imm % W)
            | .push r =>
              (24, 8, regsIn[r]!)
            | .call _ =>
              (24, 8, insAddr + len)
            | _ =>
              (64, 0, 0)
          let candHex := String.join (cand.map hex2)
          let flStr := String.ofList [flagCh t.fl.cf, flagCh t.fl.sf, flagCh t.fl.zf, flagCh t.fl.of]
          casesOut.putStrLn s!"{candHex} {len} {poolIdx} {fixRegIdx} {hex16 fixRegVal} {fixMemOff} {hex16 fixMemVal} {isBranch} | {hex16 delta} {mod1Idx} {hex16 mod1Val} {mod2Idx} {hex16 mod2Val} {flStr} {modMemOff} {modMemLen} {hex16 modMemVal}"
          return { st with acceptedTested := st.acceptedTested + 1 }
        | _ =>
          return { st with modelStuck := st.modelStuck + 1 }

def runGen (validSeeds advCount : Nat) (seed : UInt64) (casesPath statsPath : String) : IO UInt32 := do
  let pool : Array BaseState := Array.ofFn fun (i : Fin poolSize) => mkBaseState seed i.1
  let casesFile ← IO.FS.Handle.mk casesPath IO.FS.Mode.write
  let casesOut := IO.FS.Stream.ofHandle casesFile
  let mut rng := xsStep (0x9e3779b97f4a7c15 ^^^ seed)
  let mut st : FuzzStats := {}

  -- Part 1: `validSeeds` (default 5,000) valid stage-1 instructions + all
  -- Hamming-1 (`X86Dec.altsAt`) and 10x Hamming-2 bit-flip neighbours.
  for i in [:validSeeds] do
    let (r1, origBs) := genValidIns rng i
    let origLen := origBs.length
    let (r2, bs15) := padTo15 r1 origBs
    rng := r2
    st := { st with validSeeds := st.validSeeds + 1 }
    -- Evaluate the valid instruction itself
    st ← evalCandidate pool casesOut bs15 st
    -- Compute `X86Dec.altsAt` and verify 100% agreement with per-bit `decode (flipBit ...)`
    let origInsAddr := page0End - origLen
    let bFn : Nat → Option Nat := fun j => bs15[j]?
    let alts := altsAt bFn origInsAddr origLen
    let nbits := 8 * origLen
    if alts.length != nbits then
      st := { st with altsAtMismatches := st.altsAtMismatches + 1 }
    let altsArr := alts.toArray
    for k in [:nbits] do
      let decFlip := decode (flipBit bFn k) origInsAddr
      let byteIdx := k / 8
      let bitIdx := k % 8
      let bsFlip := bs15.set! byteIdx (Nat.xor bs15[byteIdx]! (1 <<< bitIdx))
      let decArr := decode (fun j => bsFlip[j]?) origInsAddr
      if altsArr[k]? != some decFlip || decFlip != decArr then
        st := { st with altsAtMismatches := st.altsAtMismatches + 1 }
      else
        st := { st with altsAtVerified := st.altsAtVerified + 1 }
      st ← evalCandidate pool casesOut bsFlip st
    -- Generate 10 Hamming-2 bit-flip neighbours per valid instruction (50,000 total)
    for _ in [:10] do
      rng := xsStep rng
      let k1 := (rng % UInt64.ofNat nbits).toNat
      rng := xsStep rng
      let k2Raw := (rng % UInt64.ofNat (if nbits > 1 then nbits - 1 else 1)).toNat
      let k2 := if k2Raw >= k1 && nbits > 1 then k2Raw + 1 else k2Raw
      let b1 := k1 / 8
      let bit1 := k1 % 8
      let b2 := k2 / 8
      let bit2 := k2 % 8
      let bsH1 := bs15.set! b1 (Nat.xor bs15[b1]! (1 <<< bit1))
      let bsH2 := bsH1.set! b2 (Nat.xor bsH1[b2]! (1 <<< bit2))
      st := { st with hamming2Tested := st.hamming2Tested + 1 }
      st ← evalCandidate pool casesOut bsH2 st

  -- Part 2: `advCount` (default 100,000) adversarial and random 1..15 byte sequences
  for i in [:advCount] do
    let (r1, advBs) := genAdversarial rng i
    rng := r1
    st := { st with advTested := st.advTested + 1 }
    st ← evalCandidate pool casesOut advBs st

  casesOut.flush
  let statsContent :=
    s!"total_candidates={st.totalCandidates}\n" ++
    s!"valid_seeds={st.validSeeds}\n" ++
    s!"hamming1_altsAt_verified={st.altsAtVerified}\n" ++
    s!"altsAt_mismatches={st.altsAtMismatches}\n" ++
    s!"hamming2_tested={st.hamming2Tested}\n" ++
    s!"adversarial_tested={st.advTested}\n" ++
    s!"accepted_tested={st.acceptedTested}\n" ++
    s!"accepted_syscall={st.acceptedSyscall}\n" ++
    s!"accepted_parity_jcc={st.acceptedParityJcc}\n" ++
    s!"rejected={st.rejected}\n" ++
    s!"model_stuck={st.modelStuck}\n"
  IO.FS.writeFile statsPath statsContent
  if st.altsAtMismatches != 0 || st.modelStuck != 0 then
    IO.eprintln s!"X86DEC_FUZZ_GEN_FAIL: altsAt_mismatches={st.altsAtMismatches} model_stuck={st.modelStuck}"
    return 1
  return 0

def parseNat (s : String) : Option Nat :=
  if s.isEmpty then none
  else s.foldl (fun acc c => do
    let a ← acc
    if '0' ≤ c && c ≤ '9' then some (a * 10 + (c.toNat - '0'.toNat)) else none) (some 0)

def lookupStat (lines : List String) (k : String) : Option Nat :=
  let pfx := k ++ "="
  lines.findSome? fun l =>
    if l.startsWith pfx then parseNat ((l.drop pfx.length).toString) else none

def runVerify (cpuReportPath statsPath : String) : IO UInt32 := do
  let statsRaw ← IO.FS.readFile statsPath
  let cpuRaw ← IO.FS.readFile cpuReportPath
  let sLines := (statsRaw.splitOn "\n").map (·.trimAsciiEnd.toString)
  let cLines := (cpuRaw.splitOn "\n").map (·.trimAsciiEnd.toString)
  match lookupStat sLines "total_candidates",
        lookupStat sLines "valid_seeds",
        lookupStat sLines "hamming1_altsAt_verified",
        lookupStat sLines "altsAt_mismatches",
        lookupStat sLines "hamming2_tested",
        lookupStat sLines "adversarial_tested",
        lookupStat sLines "accepted_tested",
        lookupStat sLines "accepted_syscall",
        lookupStat sLines "accepted_parity_jcc",
        lookupStat sLines "rejected",
        lookupStat sLines "model_stuck",
        lookupStat cLines "cpu_executed",
        lookupStat cLines "cpu_disagreements",
        lookupStat cLines "cpu_guard_faults",
        lookupStat cLines "cpu_len_mismatches" with
  | some tot, some vSeeds, some h1, some altsBad, some h2, some adv,
    some acc, some sysCnt, some parCnt, some rej, some mStuck,
    some cpuExec, some cpuDiffs, some guardFaults, some lenBad =>
    IO.println s!"X86Dec Fuzz Summary:"
    IO.println s!"  total_candidates:         {tot}"
    IO.println s!"  valid_seeds:              {vSeeds}"
    IO.println s!"  hamming1_altsAt_verified: {h1}"
    IO.println s!"  altsAt_mismatches:        {altsBad}"
    IO.println s!"  hamming2_tested:          {h2}"
    IO.println s!"  adversarial_tested:       {adv}"
    IO.println s!"  total_accepted_non_sys:   {acc}"
    IO.println s!"  accepted_syscall:         {sysCnt}"
    IO.println s!"  accepted_parity_jcc:      {parCnt}"
    IO.println s!"  total_rejected:           {rej}"
    IO.println s!"  cpu_executed:             {cpuExec}"
    IO.println s!"  cpu_guard_faults:         {guardFaults}"
    IO.println s!"  cpu_len_mismatches:       {lenBad}"
    IO.println s!"  disagreements:            {cpuDiffs}"
    for l in cLines do
      if l.startsWith "DIFF " || l.startsWith "  " then
        IO.eprintln l
    if altsBad == 0 && mStuck == 0 && cpuExec == acc && cpuDiffs == 0 && guardFaults == 0 && lenBad == 0 && acc > 0 then
      IO.println s!"X86DEC_FUZZ_OK: {tot} candidates ({h1} altsAt, {h2} Hamming-2, {adv} adversarial), {acc} CPU-verified, {rej} rejected, 0 disagreements"
      return 0
    else
      IO.eprintln s!"X86DEC_FUZZ_FAIL: altsBad={altsBad} modelStuck={mStuck} cpuExec={cpuExec}/{acc} diffs={cpuDiffs} guardFaults={guardFaults} lenBad={lenBad}"
      return 1
  | _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ =>
    IO.eprintln "X86DEC_FUZZ_FAIL: malformed stats or CPU report"
    return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | ["gen", vStr, aStr, sStr, casesPath, statsPath] =>
    let v := (parseNat vStr).getD 5000
    let a := (parseNat aStr).getD 100000
    let s := UInt64.ofNat ((parseNat sStr).getD 1)
    runGen v a s casesPath statsPath
  | ["verify", cpuReportPath, statsPath] =>
    runVerify cpuReportPath statsPath
  | _ =>
    IO.eprintln "usage: X86DecFuzzModel gen <valid_seeds> <adv_count> <seed> <cases_file> <stats_file>"
    IO.eprintln "   or: X86DecFuzzModel verify <cpu_report_file> <stats_file>"
    return 2
