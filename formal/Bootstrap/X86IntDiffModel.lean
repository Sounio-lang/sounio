/-
The model side of the instruction-level differential test for `X86Int` and `X86Dec`.

  lake build x86int-diff && .lake/build/bin/x86int-diff < cpu.txt > model.txt
  (or: lean --run Bootstrap/X86IntDiffModel.lean < cpu.txt > model.txt)

Reads the lines written by `bootstrap/model/x86diff_cpu <cases> <seed> int`
(`case | CPU result`), decodes the instruction bytes with `X86Dec.decode`,
executes one `X86Int.step`, and writes the case followed by the model's result
in the same format. A flag the model leaves undefined is written as `-`;
`bootstrap/model/x86int_diff_gate.sh` compares the two files and does not
compare a `-` position. A case that `X86Dec.decode` refuses (or where `X86Int`
is stuck or exits) is reported as a failure of the harness, never skipped.
-/
import Bootstrap.X86Dec

open X86Env X86Int X86Dec

def insAddr : Nat := 0x20180
def sp : Nat := 0x11800
def winLo : Nat := 0x117e0
def winSize : Nat := 64

def hexVal (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10) else none

def parseHex (s : String) : Option Nat :=
  s.foldl (fun acc c => do let a ← acc; let v ← hexVal c; pure (a * 16 + v)) (some 0)

def bytesOf (s : String) : Option (List Nat) :=
  let rec go : List Char → Option (List Nat)
    | [] => some []
    | a :: b :: rest => do
      let x ← hexVal a; let y ← hexVal b; let r ← go rest; pure ((x * 16 + y) :: r)
    | _ => none
  go s.toList

def hexDigit (n : Nat) : Char :=
  match n with
  | 0 => '0' | 1 => '1' | 2 => '2' | 3 => '3'
  | 4 => '4' | 5 => '5' | 6 => '6' | 7 => '7'
  | 8 => '8' | 9 => '9' | 10 => 'a' | 11 => 'b'
  | 12 => 'c' | 13 => 'd' | 14 => 'e' | _ => 'f'

def hex2 (v : Nat) : String :=
  String.ofList [hexDigit ((v / 16) % 16), hexDigit (v % 16)]

def hex16 (v : Nat) : String :=
  let u := UInt64.ofNat v
  let rec go (i : Nat) (acc : List Char) : List Char :=
    match i with
    | 0 => acc
    | i' + 1 =>
      let nib := ((u >>> UInt64.ofNat (4 * (15 - i'))) &&& 15).toNat
      go i' (hexDigit nib :: acc)
  String.ofList (go 16 [])

def flagCh : Option Bool → Char
  | none => '-'
  | some true => '1'
  | some false => '0'

def stateStr (regs : List (Option Nat)) (fl : Flags) (mem : List Nat) : String :=
  let r := " ".intercalate (regs.map fun o => match o with | some v => hex16 v | none => "----------------")
  let f := String.ofList [flagCh fl.cf, flagCh fl.sf, flagCh fl.zf, flagCh fl.of]
  let m := String.join (mem.map hex2)
  s!"{r} {f} {m}"

def formNames : Array String := #[
  "movI32", "movI64",
  "alu_add", "alu_sub", "alu_and", "alu_or", "alu_xor", "xor32_self",
  "mov32_89", "mov32_8b", "ld32",
  "mov64_89", "mov64_8b", "st64", "ld64",
  "test64", "pushI",
  "cmp_39", "cmp_3b",
  "aluI8_add", "aluI8_or", "aluI8_and", "aluI8_sub", "aluI8_xor", "cmpI8",
  "aluI32_add", "aluI32_or", "aluI32_and", "aluI32_sub", "aluI32_xor", "cmpI32",
  "shlI", "shrI",
  "lea", "st8", "ld8",
  "jcc_rel32", "jcc_rel8", "jmp_rel8", "jmp_rel32", "call_rel32", "ret",
  "push_r64", "pop_r64", "jtab"
]

def classifyForm (ins : List Nat) (decoded : Ins) : Option Nat :=
  let b0 := ins.headD 0
  let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
  let p := if hasRex then 1 else 0
  let w := if hasRex then (b0 - 0x40) / 8 % 2 == 1 else false
  let o := (ins[p]?).getD 0
  match decoded with
  | .movI _ _ => some (if w then 1 else 0)
  | .alu .add _ _ => some 2
  | .alu .sub _ _ => some 3
  | .alu .and _ _ => some 4
  | .alu .or _ _ => some 5
  | .alu .xor _ _ => some (if w then 6 else 7)
  | .mov32 _ _ => some (if o == 0x89 then 8 else 9)
  | .ld32 _ _ _ => some 10
  | .mov _ _ => some (if o == 0x89 then 11 else 12)
  | .st _ _ _ => some 13
  | .ld _ _ _ => some 14
  | .test _ _ => some 15
  | .pushI _ => some 16
  | .cmp _ _ => some (if o == 0x39 then 17 else 18)
  | .aluI .add _ _ => some (if o == 0x83 then 19 else 25)
  | .aluI .or _ _ => some (if o == 0x83 then 20 else 26)
  | .aluI .and _ _ => some (if o == 0x83 then 21 else 27)
  | .aluI .sub _ _ => some (if o == 0x83 then 22 else 28)
  | .aluI .xor _ _ => some (if o == 0x83 then 23 else 29)
  | .cmpI _ _ => some (if o == 0x83 then 24 else 30)
  | .shlI _ _ => some 31
  | .shrI _ _ => some 32
  | .lea _ _ _ => some 33
  | .st8 _ _ _ => some 34
  | .ld8 _ _ _ => some 35
  | .jcc _ _ => some (if o == 0x0f then 36 else 37)
  | .jmp _ => some (if o == 0xeb then 38 else 39)
  | .call _ => some 40
  | .ret => some 41
  | .push _ => some 42
  | .pop _ => some 43
  | .jtab _ _ _ => some 44
  | .jmpR _ | .syscall => none

def evalCase (line : String) : String × Option Nat × Bool :=
  let caseStr := (line.splitOn " | ").head!
  let fs := caseStr.splitOn " "
  if fs.length != 19 then
    (s!"{line} | UNPARSED", none, true)
  else
    match bytesOf fs[0]!, (fs.drop 1 |>.take 16).mapM parseHex, bytesOf fs[18]! with
    | some ins, some regs, some memBytes =>
      let fl := fs[17]!.toList
      if fl.length != 4 || memBytes.length != winSize then
        (s!"{line} | UNPARSED", none, true)
      else
        match decode (fun i => ins[i]?) insAddr with
        | none => (s!"{caseStr} | DECODE_REFUSED", none, true)
        | some (decoded, len) =>
          if len != ins.length then
            (s!"{caseStr} | DECODE_LEN_MISMATCH", none, true)
          else
            match classifyForm ins decoded with
            | none => (s!"{caseStr} | UNEXPECTED_FORM", none, true)
            | some formIdx =>
              let b (c : Char) : Option Bool := some (c == '1')
              let flags : Flags := ⟨b fl[0]!, b fl[2]!, b fl[1]!, b fl[3]!⟩
              let s : St := {
                regs := regs.map some,
                rip := insAddr,
                fl := flags,
                mem := [⟨winLo, memBytes, true⟩],
                k := ⟨[], [], [], [], []⟩
              }
              let im := imageOf ins insAddr
              match step im s with
              | .ok t =>
                match t.mem.rdN winLo winSize with
                | some outMem =>
                  let delta := (t.rip + W - insAddr) % W
                  (s!"{caseStr} | {hex16 delta} {stateStr t.regs t.fl outMem}", some formIdx, false)
                | none =>
                  (s!"{caseStr} | MEM_READ_FAILED", some formIdx, true)
              | .exit c =>
                (s!"{caseStr} | EXIT_{c}", some formIdx, true)
              | .stuck =>
                (s!"{caseStr} | STUCK", some formIdx, true)
    | _, _, _ =>
      (s!"{line} | UNPARSED", none, true)

partial def loop (h : IO.FS.Stream) (o : IO.FS.Stream) (n bad : Nat) (counts : Array Nat) :
    IO (Nat × Nat × Array Nat) := do
  let line ← h.getLine
  if line.isEmpty then return (n, bad, counts)
  let line := line.trimAsciiEnd.toString
  let (outLine, formOpt, failed) := evalCase line
  o.putStrLn outLine
  let counts' := match formOpt with
    | some idx => counts.modify idx (· + 1)
    | none => counts
  loop h o (n + 1) (if failed then bad + 1 else bad) counts'

def main : IO UInt32 := do
  let initCounts := Array.replicate formNames.size 0
  let (n, bad, counts) ← loop (← IO.getStdin) (← IO.getStdout) 0 0 initCounts
  IO.eprintln s!"x86int_diff model: {n} cases, {bad} failed"
  let mut minCount := if counts.isEmpty then 0 else counts[0]!
  for i in [:formNames.size] do
    let c := counts[i]!
    if c < minCount then minCount := c
    IO.eprintln s!"  form {formNames[i]!}: {c}"
  IO.eprintln s!"min_form_count={minCount}"
  return (if bad == 0 then 0 else 1)
