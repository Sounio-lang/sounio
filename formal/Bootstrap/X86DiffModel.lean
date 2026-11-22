/-
The model side of the instruction-level differential test.

  lean --run Bootstrap/X86DiffModel.lean < cpu.txt > model.txt

Reads the lines written by bootstrap/model/x86diff_cpu (case | CPU result),
runs one `step` of the model on each case, and writes the case followed by the
model's result in the same format. A flag the model leaves undefined is written
as `-`; bootstrap/model/x86diff_gate.sh compares the two files and does not
compare a `-` position.
-/
import Bootstrap.X86Min
open X86Min

def hexVal (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10) else none

def parseHex (s : String) : Option Nat :=
  s.foldl (fun acc c => do let a ← acc; let v ← hexVal c; pure (a * 16 + v)) (some 0)

def bytesOf (s : String) : Option (List Nat) :=
  let cs := s.toList
  let rec go : List Char → Option (List Nat)
    | [] => some []
    | a :: b :: rest => do
      let x ← hexVal a; let y ← hexVal b; let r ← go rest; pure ((x * 16 + y) :: r)
    | _ => none
  go cs

def hexN (digits v : Nat) : String :=
  String.ofList ((List.range digits).reverse.map fun i =>
    "0123456789abcdef".toList[(v / 16 ^ i) % 16]!)

def flagCh : Option Bool → Char
  | none => '-'
  | some true => '1'
  | some false => '0'

def sp : Nat := 0x10800
def base : Nat := 0x400000

def stateStr (regs : List (Option Nat)) (fl : Flags) (mem : List (Option Nat)) : String :=
  let r := " ".intercalate (regs.map fun o => match o with | some v => hexN 16 v | none => "----------------")
  let f := String.ofList [flagCh fl.cf, flagCh fl.sf, flagCh fl.zf, flagCh fl.of]
  let m := String.join (mem.map fun o => match o with | some v => hexN 2 v | none => "--")
  s!"{r} {f} {m}"

def caseResult (line : String) : Option String := do
  let fs := (line.splitOn " | ").head!.splitOn " "
  let ins ← bytesOf fs[0]!
  let regs ← (fs.drop 1 |>.take 16).mapM parseHex
  let fl := fs[17]!.toList
  let mem ← bytesOf fs[18]!
  let b (c : Char) : Option Bool := some (c == '1')
  let flags : Flags := ⟨b fl[0]!, b fl[2]!, b fl[1]!, b fl[3]!⟩
  let im := Img.mk' ins base ins.length
  let s : St := { regs := regs.map some, rip := base, fl := flags, mem := mem.map some,
                  lo := sp - winSize / 2, inp := [], out := [] }
  let caseStr := (line.splitOn " | ").head!
  match step im s with
  | .inl t =>
    pure s!"{caseStr} | {hexN 2 ((t.rip + 2 ^ 64 - base) % 2 ^ 64)} {stateStr t.regs t.fl t.mem}"
  | .inr _ => pure s!"{caseStr} | STUCK"

partial def loop (h : IO.FS.Stream) (o : IO.FS.Stream) (n bad : Nat) : IO (Nat × Nat) := do
  let line ← h.getLine
  if line.isEmpty then return (n, bad)
  let line := line.trimAsciiEnd.toString
  match caseResult line with
  | some r => o.putStrLn r; loop h o (n + 1) bad
  | none => o.putStrLn s!"{line} | UNPARSED"; loop h o (n + 1) (bad + 1)

def main : IO UInt32 := do
  let (n, bad) ← loop (← IO.getStdin) (← IO.getStdout) 0 0
  IO.eprintln s!"x86diff model: {n} cases, {bad} unparsed"
  return (if bad == 0 then 0 else 1)
