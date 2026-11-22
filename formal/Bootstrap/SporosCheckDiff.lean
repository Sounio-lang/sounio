/-
SporosCheckDiff: the Lean oracle side of the `spr1_check` differential gate
(bootstrap/model/spr1_check_gate.sh).

    lake --dir formal env lean --run formal/Bootstrap/SporosCheckDiff.lean FILE...
    lake --dir formal env lean --run formal/Bootstrap/SporosCheckDiff.lean --list LISTFILE
    lake --dir formal env lean --run formal/Bootstrap/SporosCheckDiff.lean gen DIR

(or, from formal/, `lake env lean --run Bootstrap/SporosCheckDiff.lean ...`, which
makes elan pick formal/lean-toolchain whatever the default toolchain is).

The first two modes print one line `<verdict> <file>` per input file, in order,
where the verdict is `accept` exactly when `Sporos.decode bs = some pr` and
`Sporos.checkProg pr = true` (`verdict` below), and `reject` otherwise. All the
files are judged in this one process, so a batch of ten thousand costs one Lean
start-up. `--list` reads the file names, one per line, from LISTFILE (empty
lines are skipped), which keeps long batches clear of the argument-size limit.
An unreadable file is reported as `error <file>` and makes the exit status 1.

`gen DIR` writes the four example programs `echo`, `hello`, `upper` and `fib`
to `DIR/<name>.spb` (the bytes of `Prog.enc`), and fails if any of them is not
accepted. `Sporos.Examples` declares a `main` of its own, so it cannot be
imported here; `hello`, `upper` and `fib` are restated below verbatim, and the
gate compares the files `gen` writes byte for byte with those written by the
`sporos-examples` executable built from `Sporos.Examples` itself. `echo` is
`Sporos.echo` (formal/Sporos/Run.lean).
-/
import Sporos.Bytecode
import Sporos.Run

namespace Sporos.CheckDiff

open Sporos

/-- The specification `spr1_check` implements: decode, then check. -/
def verdict (bs : List Nat) : Bool :=
  match decode bs with
  | some pr => checkProg pr
  | none => false

theorem verdict_iff (bs : List Nat) :
    verdict bs = true ↔ ∃ pr, decode bs = some pr ∧ checkProg pr = true := by
  unfold verdict
  cases h : decode bs with
  | none => simp
  | some pr => simp

/-! ### The example programs (verbatim from formal/Sporos/Examples.lean) -/

namespace Ex

def hello : Prog where
  words := [{
    ins := 0, outs := 0, locals := 2,
    code := [.str 0, .locset 1, .locset 0,
             .locget 1, .jz 17,
             .locget 0, .prim .ld8, .prim .putb,
             .locget 0, .lit 1, .prim .add, .locset 0,
             .locget 1, .lit 1, .prim .sub, .locset 1,
             .jmp 3,
             .lit 0, .prim .exit],
    ann := [(3, ⟨0, 0⟩), (17, ⟨0, 0⟩)] }]
  data := ["Hello from Sporos!\n".toList.map Char.toNat]
  main := 0

def upper : Prog where
  words := [{
    ins := 0, outs := 0, locals := 0,
    code := [.prim .getb, .prim .dup, .lit (M - 1), .prim .eq, .jnz 17,
             .prim .dup, .lit 97, .prim .ltu, .jnz 15,
             .prim .dup, .lit 123, .prim .ltu, .jz 15,
             .lit 32, .prim .sub,
             .prim .putb, .jmp 0,
             .prim .drop, .lit 0, .prim .exit],
    ann := [(0, ⟨0, 0⟩), (15, ⟨1, 0⟩), (17, ⟨1, 0⟩)] }]
  data := []
  main := 0

def printDec : Word where
  ins := 1
  outs := 0
  locals := 0
  code := [.prim .dup, .lit 10, .prim .ltu, .jnz 8,
           .prim .dup, .lit 10, .prim .divu, .call 1,
           .lit 10, .prim .modu, .lit 48, .prim .add, .prim .putb, .ret]
  ann := [(8, ⟨1, 0⟩)]

def fib : Prog where
  words := [{
    ins := 0, outs := 0, locals := 3,
    code := [.lit 0, .locset 0, .lit 1, .locset 1, .lit 20, .locset 2,
             .locget 2, .jz 23,
             .locget 0, .call 1, .lit 10, .prim .putb,
             .locget 0, .locget 1, .prim .add, .locget 1, .locset 0, .locset 1,
             .locget 2, .lit 1, .prim .sub, .locset 2,
             .jmp 6,
             .lit 0, .prim .exit],
    ann := [(6, ⟨0, 0⟩), (23, ⟨0, 0⟩)] }, printDec]
  data := []
  main := 0

def all : List (String × Prog) := [("echo", Sporos.echo), ("hello", hello), ("upper", upper), ("fib", fib)]

end Ex

/-! ### Sanity: the examples are accepted, damaged files are rejected -/

/-- A file header: magic, `main`, number of words, number of data items. -/
def hdr (main nw nd : Nat) : List Nat :=
  [0x53, 0x50, 0x52, 0x31] ++ leBytes main 4 ++ leBytes nw 4 ++ leBytes nd 4

/-- A word header: `ins outs locals ncode nann`. -/
def whdr (ins outs locals nc na : Nat) : List Nat :=
  leBytes ins 4 ++ leBytes outs 4 ++ leBytes locals 4 ++ leBytes nc 4 ++ leBytes na 4

example : verdict echo.enc = true := by decide +kernel
example : verdict Ex.hello.enc = true := by decide +kernel
example : verdict Ex.upper.enc = true := by decide +kernel
example : verdict Ex.fib.enc = true := by decide +kernel

/-- The smallest accepted file: one word `lit 0 exit`. -/
def tiny : List Nat := hdr 0 1 0 ++ whdr 0 0 0 2 0 ++ (0x80 :: leBytes 0 8) ++ [50]

example : verdict tiny = true := by decide +kernel
-- the empty file, a bare magic, a wrong magic
example : verdict [] = false := by decide +kernel
example : verdict [0x53, 0x50, 0x52, 0x31] = false := by decide +kernel
example : verdict ([0x53, 0x50, 0x52, 0x32] ++ tiny.drop 4) = false := by decide +kernel
-- decodes, but there is no word `main`
example : decode (hdr 0 0 0) ≠ none ∧ verdict (hdr 0 0 0) = false := by decide +kernel
-- main out of range
example : verdict (hdr 1 1 0 ++ whdr 0 0 0 2 0 ++ (0x80 :: leBytes 0 8) ++ [50]) = false := by
  decide +kernel
-- `main` takes an argument
example : verdict (hdr 0 1 0 ++ whdr 1 0 0 2 0 ++ (0x80 :: leBytes 0 8) ++ [50]) = false := by
  decide +kernel
-- a trailing byte, a truncated file
example : verdict (tiny ++ [0]) = false := by decide +kernel
example : verdict tiny.dropLast = false := by decide +kernel
-- the first opcode past the primitives, and the first past `ret`
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 2 0 ++ (0x80 :: leBytes 0 8) ++ [0x39]) = false := by
  decide +kernel
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 2 0 ++ (0x80 :: leBytes 0 8) ++ [0x8d]) = false := by
  decide +kernel
-- `exit` on an empty stack (ill-typed), an empty word, falling off the end
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 1 0 ++ [50]) = false := by decide +kernel
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 0 0) = false := by decide +kernel
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 1 0 ++ (0x80 :: leBytes 0 8)) = false := by
  decide +kernel
-- a huge instruction count in a short file
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 0xffffffff 0 ++ (0x80 :: leBytes 0 8) ++ [50]) = false := by
  decide +kernel
-- an undeclared local, a jump to an unannotated index
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 2 0 ++ [0x86, 0, 0, 0, 0] ++ [50]) = false := by
  decide +kernel
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 3 0 ++ [0x88, 1, 0, 0, 0] ++ (0x80 :: leBytes 0 8) ++ [50]) =
    false := by decide +kernel
-- the same jump, annotated: accepted; then a second, conflicting annotation: rejected
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 3 1 ++ [0x88, 1, 0, 0, 0] ++ (0x80 :: leBytes 0 8) ++ [50] ++
    leBytes 1 4 ++ leBytes 0 4 ++ leBytes 0 4) = true := by decide +kernel
example : verdict (hdr 0 1 0 ++ whdr 0 0 0 3 2 ++ [0x88, 1, 0, 0, 0] ++ (0x80 :: leBytes 0 8) ++ [50] ++
    leBytes 1 4 ++ leBytes 0 4 ++ leBytes 0 4 ++ leBytes 1 4 ++ leBytes 1 4 ++ leBytes 0 4) = false := by
  decide +kernel
/-- `echo` with the annotation of its loop head damaged. -/
def echoBadAnn : Prog where
  words := [{
    ins := 0, outs := 0, locals := 0,
    code := [.prim .getb, .prim .dup, .lit (M - 1), .prim .eq, .jnz 7, .prim .putb, .jmp 0,
             .prim .drop, .lit 0, .prim .exit],
    ann := [(0, ⟨1, 0⟩), (7, ⟨1, 0⟩)] }]
  data := []
  main := 0

example : verdict echoBadAnn.enc = false := by decide +kernel

end Sporos.CheckDiff

open Sporos Sporos.CheckDiff

def readBytes (path : String) : IO (List Nat) := do
  let b ← IO.FS.readBinFile path
  return b.toList.map UInt8.toNat

def judge (out : IO.FS.Stream) (files : List String) : IO UInt32 := do
  let mut rc : UInt32 := 0
  for f in files do
    match ← (readBytes f).toBaseIO with
    | .ok bs => out.putStrLn s!"{if verdict bs then "accept" else "reject"} {f}"
    | .error _ =>
      out.putStrLn s!"error {f}"
      rc := 1
  out.flush
  return rc

def main (args : List String) : IO UInt32 := do
  let out ← IO.getStdout
  match args with
  | ["gen", dir] =>
    IO.FS.createDirAll dir
    for (name, p) in Ex.all do
      if !verdict p.enc then
        (← IO.getStderr).putStrLn s!"SporosCheckDiff: example {name} is rejected"
        return 1
      IO.FS.writeBinFile s!"{dir}/{name}.spb" (ByteArray.mk (p.enc.map Nat.toUInt8).toArray)
    return 0
  | ["--list", lst] =>
    let names := (← IO.FS.lines lst).toList.filter (· ≠ "")
    judge out names
  | [] =>
    (← IO.getStderr).putStrLn
      "usage: SporosCheckDiff FILE... | --list LISTFILE | gen DIR"
    return 2
  | files => judge out files
