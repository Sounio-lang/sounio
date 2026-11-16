/-
Σπόρος (Sporos), the language of stage 1 of bootstrap root A: reference
semantics. Design: claude/research/2026-11-15-root-a-stage1-design.md
(decisions 1-19).

* A stack machine over 64-bit words that wrap. A program is a list of words
  (subroutines); each declares its stack effect `(ins -- outs)` and its number
  of locals. Inside a word, control is free jumps to instruction indices.
* Every value carries its provenance: the set of origins (input byte,
  instruction, core) that determined it. The context label `pc` collects the
  origins of every branch condition taken and is never removed, so every value
  produced, and every byte written, carries it too.
* The physical world (clock, current core, sensors) and the confidence of a
  provenance are parameters of the semantics (`Phys`); the fault model is a
  separate layer.

This file: syntax, values, machine state, one step.
-/
namespace Sporos

def M : Nat := 2 ^ 64

/-- Where a value came from: an input byte (if any), the instruction that made it
(word, index), and the core that executed that instruction. -/
structure Origin where
  inp  : Option Nat
  word : Nat
  ins  : Nat
  core : Nat
deriving DecidableEq, Repr

/-- A provenance is a set of origins, kept as a list (order and repetition do
not matter to anything stated about it). -/
abbrev Prov := List Origin

structure Val where
  w : Nat
  p : Prov
deriving DecidableEq, Repr

/-- Operations that take their operands from the data stack. -/
inductive Prim where
  -- stack
  | dup | drop | swap | over | rot | nip | tuck
  -- arithmetic, 64-bit, wrapping
  | add | sub | mul | divu | modu | divs | mods | neg
  -- bits
  | band | bor | bxor | bnot | shl | shr | sar
  -- comparison (push 0 or 1)
  | eq | ne | ltu | lts | leu | les
  -- memory
  | ld8 | st8 | ld16 | st16 | ld32 | st32 | ld64 | st64
  | copy | fill | mcmp
  -- heap
  | alloc | mark | release
  -- strings (address, length)
  | streq | strhash
  -- tables keyed by 64-bit words
  | tnew | tput | tget
  -- input and output
  | getb | putb | putn | exit
  -- physics
  | clock | core | sensor
  -- epistemic
  | provget | confget | demand
deriving DecidableEq, Repr

/-- Data-stack operands taken and results pushed by a primitive. -/
def Prim.eff : Prim → Nat × Nat
  | .dup => (1, 2) | .drop => (1, 0) | .swap => (2, 2) | .over => (2, 3)
  | .rot => (3, 3) | .nip => (2, 1) | .tuck => (2, 3)
  | .add | .sub | .mul | .divu | .modu | .divs | .mods => (2, 1) | .neg => (1, 1)
  | .band | .bor | .bxor | .shl | .shr | .sar => (2, 1) | .bnot => (1, 1)
  | .eq | .ne | .ltu | .lts | .leu | .les => (2, 1)
  | .ld8 | .ld16 | .ld32 | .ld64 => (1, 1)
  | .st8 | .st16 | .st32 | .st64 => (2, 0)
  | .copy => (3, 0) | .fill => (3, 0) | .mcmp => (3, 1)
  | .alloc => (1, 1) | .mark => (0, 1) | .release => (1, 0)
  | .streq => (4, 1) | .strhash => (2, 1)
  | .tnew => (0, 1) | .tput => (3, 0) | .tget => (2, 2)
  | .getb => (0, 1) | .putb => (1, 0) | .putn => (1, 0) | .exit => (1, 0)
  | .clock => (0, 1) | .core => (0, 1) | .sensor => (1, 1)
  | .provget => (1, 2) | .confget => (1, 2) | .demand => (2, 1)

inductive Instr where
  | prim (p : Prim)
  | lit (n : Nat)
  /-- push the address and length of data item `i` -/
  | str (i : Nat)
  /-- copy the value `n` below the top (0 is `dup`) -/
  | pick (n : Nat)
  | tor | fromr | rfetch
  | locget (i : Nat) | locset (i : Nat)
  | jmp (t : Nat) | jz (t : Nat) | jnz (t : Nat)
  | call (w : Nat) | ret
deriving DecidableEq, Repr

/-- Stack heights at an instruction, relative to the word's entry: data stack
above the word's arguments' base, auxiliary stack above its entry. -/
structure Height where
  d : Nat
  a : Nat
deriving DecidableEq, Repr

structure Word where
  ins    : Nat
  outs   : Nat
  locals : Nat
  code   : List Instr
  /-- declared heights at instruction indices: required at every jump target
  and after every instruction that does not fall through -/
  ann    : List (Nat × Height)
deriving Repr

structure Prog where
  words : List Word
  /-- read-only data, loaded from address 0 in order -/
  data  : List (List Nat)
  main  : Nat
deriving Repr

/-- The physical world as the semantics sees it: what `clock`, `core` and
`sensor` return at step `k`, and the confidence (fixed point, 2^32 = 1) of a
provenance. The hierarchical fault model gives `conf` its meaning. -/
structure Phys where
  clock  : Nat → Nat
  core   : Nat → Nat
  sensor : Nat → Nat → Nat
  conf   : Prov → Nat

structure Frame where
  word : Nat
  pc   : Nat
  locs : List Val
  /-- data-stack height below the word's arguments -/
  base : Nat
  /-- auxiliary-stack height at entry -/
  abase : Nat
deriving Repr

structure St where
  ds    : List Val          -- data stack, top first
  aux   : List Val
  rs    : List Frame        -- current frame first
  mem   : Nat → Val
  msize : Nat
  hp    : Nat
  tabs  : List (List (Nat × Val))
  inp   : List Nat
  ipos  : Nat
  out   : List (Nat × Prov) -- bytes written, most recent first, with provenance
  pc    : Prov              -- the context label: never shrinks
  step  : Nat

inductive Err where
  | underflow | auxUnderflow | badLocal | badJump | badWord | noFrame
  | bounds | divZero | badTable | demand
deriving DecidableEq, Repr

inductive Res where
  | next (s : St)
  | halt (code : Nat) (out : List (Nat × Prov))
  | fail (e : Err)

/-! ### Words, signed values, memory -/

def toSigned (x : Nat) : Int := if x < 2 ^ 63 then x else (x : Int) - M
def ofInt (i : Int) : Nat := (i % (M : Int)).toNat

def wrap (x : Nat) : Nat := x % M

def Prov.join (a b : Prov) : Prov := a ++ b

/-- FNV-1a, 64 bits. -/
def fnv (bs : List Nat) : Nat :=
  bs.foldl (fun h b => (Nat.xor h b) * 1099511628211 % M) 14695981039346656037

def St.ld (s : St) (a n : Nat) : Option (Nat × Prov) :=
  if a + n ≤ s.msize then
    some ((List.range n).foldr (fun i acc => acc * 256 + (s.mem (a + i)).w % 256) 0,
          (List.range n).foldr (fun i acc => (s.mem (a + i)).p ++ acc) [])
  else none

def St.st (s : St) (a n v : Nat) (p : Prov) : Option St :=
  if a + n ≤ s.msize then
    some { s with mem := (fun x => if a ≤ x ∧ x < a + n then ⟨v / 256 ^ (x - a) % 256, p⟩ else s.mem x) }
  else none

end Sporos
