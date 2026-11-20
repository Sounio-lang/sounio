/-
The binary bytecode of Sporos (decision 32 of the root-A stage-1 design): one
opcode byte per instruction, fixed-size little-endian immediates.

    file    := "SPR1" main:u32 nwords:u32 ndata:u32 data* word*
    data    := len:u32 byte*
    word    := ins:u32 outs:u32 locals:u32 ncode:u32 nann:u32 instr* ann*
    ann     := index:u32 d:u32 a:u32
    instr   := prim                    one byte, 0x00 .. 0x3f, `Prim` in order
             | 0x80 n:u64              lit
             | 0x81 i:u32              str
             | 0x82 n:u32              pick
             | 0x83 | 0x84 | 0x85      tor fromr rfetch
             | 0x86 i:u32 | 0x87 i:u32 locget locset
             | 0x88 t:u32 | 0x89 t:u32 | 0x8a t:u32   jmp jz jnz
             | 0x8b w:u32              call
             | 0x8c                    ret

`decode` is total: bytes that are not a well-formed file decode to `none`.
`decode_encode_echo` checks the round trip on the echo program.
-/
import Sporos.Core

namespace Sporos

/-- Every primitive, in opcode order. -/
def Prim.all : List Prim :=
  [.dup, .drop, .swap, .over, .rot, .nip, .tuck,
   .add, .sub, .mul, .divu, .modu, .divs, .mods, .neg,
   .band, .bor, .bxor, .bnot, .shl, .shr, .sar,
   .eq, .ne, .ltu, .lts, .leu, .les,
   .ld8, .st8, .ld16, .st16, .ld32, .st32, .ld64, .st64,
   .copy, .fill, .mcmp,
   .alloc, .mark, .release,
   .streq, .strhash,
   .tnew, .tput, .tget,
   .getb, .putb, .putn, .exit,
   .clock, .core, .sensor,
   .provget, .confget, .demand]

def Prim.code (p : Prim) : Nat := Prim.all.idxOf p

def leBytes (n k : Nat) : List Nat := (List.range k).map fun i => n / 256 ^ i % 256

def Instr.enc : Instr → List Nat
  | .prim p => [p.code]
  | .lit n => 0x80 :: leBytes n 8
  | .str i => 0x81 :: leBytes i 4
  | .pick n => 0x82 :: leBytes n 4
  | .tor => [0x83] | .fromr => [0x84] | .rfetch => [0x85]
  | .locget i => 0x86 :: leBytes i 4
  | .locset i => 0x87 :: leBytes i 4
  | .jmp t => 0x88 :: leBytes t 4
  | .jz t => 0x89 :: leBytes t 4
  | .jnz t => 0x8a :: leBytes t 4
  | .call w => 0x8b :: leBytes w 4
  | .ret => [0x8c]

def Word.enc (w : Word) : List Nat :=
  leBytes w.ins 4 ++ leBytes w.outs 4 ++ leBytes w.locals 4 ++ leBytes w.code.length 4 ++
    leBytes w.ann.length 4 ++ (w.code.flatMap Instr.enc) ++
    w.ann.flatMap fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++ leBytes x.2.a 4

def Prog.enc (p : Prog) : List Nat :=
  [0x53, 0x50, 0x52, 0x31] ++ leBytes p.main 4 ++ leBytes p.words.length 4 ++ leBytes p.data.length 4 ++
    (p.data.flatMap fun d => leBytes d.length 4 ++ d) ++ p.words.flatMap Word.enc

/-! ### Decoding -/

/-- A little-endian number of `k` bytes and the rest. -/
def rdLE (bs : List Nat) (k : Nat) : Option (Nat × List Nat) :=
  if bs.length < k then none
  else some (((bs.take k).zip (List.range k)).foldl (fun acc x => acc + x.1 * 256 ^ x.2) 0, bs.drop k)

def decInstr (bs : List Nat) : Option (Instr × List Nat) :=
  match bs with
  | [] => none
  | op :: rest =>
    if h : op < Prim.all.length then some (.prim (Prim.all[op]'h), rest)
    else
      let imm (k : Nat) (f : Nat → Instr) : Option (Instr × List Nat) :=
        (rdLE rest k).map fun x => (f x.1, x.2)
      match op with
      | 0x80 => imm 8 .lit
      | 0x81 => imm 4 .str
      | 0x82 => imm 4 .pick
      | 0x83 => some (.tor, rest)
      | 0x84 => some (.fromr, rest)
      | 0x85 => some (.rfetch, rest)
      | 0x86 => imm 4 .locget
      | 0x87 => imm 4 .locset
      | 0x88 => imm 4 .jmp
      | 0x89 => imm 4 .jz
      | 0x8a => imm 4 .jnz
      | 0x8b => imm 4 .call
      | 0x8c => some (.ret, rest)
      | _ => none

/-- `n` items, each read by `f`. -/
def decMany {α : Type} (f : List Nat → Option (α × List Nat)) : Nat → List Nat → Option (List α × List Nat)
  | 0, bs => some ([], bs)
  | n + 1, bs => do
    let (x, bs) ← f bs
    let (xs, bs) ← decMany f n bs
    pure (x :: xs, bs)

def decAnn (bs : List Nat) : Option ((Nat × Height) × List Nat) := do
  let (i, bs) ← rdLE bs 4
  let (d, bs) ← rdLE bs 4
  let (a, bs) ← rdLE bs 4
  pure ((i, ⟨d, a⟩), bs)

def decWord (bs : List Nat) : Option (Word × List Nat) := do
  let (ins, bs) ← rdLE bs 4
  let (outs, bs) ← rdLE bs 4
  let (locals, bs) ← rdLE bs 4
  let (nc, bs) ← rdLE bs 4
  let (na, bs) ← rdLE bs 4
  let (code, bs) ← decMany decInstr nc bs
  let (ann, bs) ← decMany decAnn na bs
  pure (⟨ins, outs, locals, code, ann⟩, bs)

def decData (bs : List Nat) : Option (List Nat × List Nat) := do
  let (n, bs) ← rdLE bs 4
  if bs.length < n then none else pure (bs.take n, bs.drop n)

/-- A whole file; trailing bytes are an error. -/
def decode (bs : List Nat) : Option Prog := do
  match bs with
  | 0x53 :: 0x50 :: 0x52 :: 0x31 :: bs =>
    let (main, bs) ← rdLE bs 4
    let (nw, bs) ← rdLE bs 4
    let (nd, bs) ← rdLE bs 4
    let (data, bs) ← decMany decData nd bs
    let (words, bs) ← decMany decWord nw bs
    if bs.isEmpty then pure ⟨words, data, main⟩ else none
  | _ => none

theorem Prim.mem_all (p : Prim) : p ∈ Prim.all := by cases p <;> decide

theorem Prim.all_nodup : Prim.all.Nodup := by decide

end Sporos
