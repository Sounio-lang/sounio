/-
The model side of the instruction-level differential test for `AArch64Min`.

  aarch64_diff_cpu <cases> <seed> | aarch64-diff > model.txt
  (or: lean --run Bootstrap/AArch64DiffModel.lean < cpu.txt > model.txt)

Reads lines written by `bootstrap/model/aarch64_diff_cpu` (`case | CPU result`),
decodes the 32-bit instruction with `AArch64Min.decode`, executes one
`AArch64Min.step`, and prints the case followed by the model's result in the
same format. Any case refused by `decode` or stuck/exiting in `step` is
reported as a failure, never skipped.
-/
import Bootstrap.AArch64Min

open AArch64Min

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

def optHex16 : Option Nat → String
  | some v => hex16 v
  | none => "----------------"

def optHex2 : Option Nat → String
  | some v => hex2 v
  | none => "--"

def stateStr (spOpt : Option Nat) (regs : List (Option Nat)) (fl : Flags) (mem : List (Option Nat)) : String :=
  let spS := optHex16 spOpt
  let r := " ".intercalate (regs.map optHex16)
  let f := String.ofList [flagCh fl.n, flagCh fl.z, flagCh fl.c, flagCh fl.v]
  let m := String.join (mem.map optHex2)
  s!"{spS} {r} {f} {m}"

def formNames : Array String := #[
  "movz32", "movz64",
  "movn32", "movn64",
  "movk32", "movk64",
  "addImm32", "addImm64", "addsImm",
  "subImm32", "subImm64", "subsImm",
  "addReg", "addsReg", "subReg", "subsReg",
  "andReg", "orrReg", "eorReg", "andsReg",
  "ldrb", "strb", "ldr32", "str32", "ldr64", "str64",
  "b", "bl", "bcond", "cbz", "cbnz", "ret"
]

def classifyForm : Ins → Option Nat
  | .movWide 2 w _ _ _ => some (if w == 32 then 0 else 1)
  | .movWide 0 w _ _ _ => some (if w == 32 then 2 else 3)
  | .movWide 3 w _ _ _ => some (if w == 32 then 4 else 5)
  | .addSubImm false false w _ _ _ => some (if w == 32 then 6 else 7)
  | .addSubImm false true _ _ _ _ => some 8
  | .addSubImm true false w _ _ _ => some (if w == 32 then 9 else 10)
  | .addSubImm true true _ _ _ _ => some 11
  | .addSubReg false false _ _ _ _ _ _ => some 12
  | .addSubReg false true _ _ _ _ _ _ => some 13
  | .addSubReg true false _ _ _ _ _ _ => some 14
  | .addSubReg true true _ _ _ _ _ _ => some 15
  | .logicReg .and _ _ _ _ _ _ _ => some 16
  | .logicReg .orr _ _ _ _ _ _ _ => some 17
  | .logicReg .eor _ _ _ _ _ _ _ => some 18
  | .logicReg .ands _ _ _ _ _ _ _ => some 19
  | .ldr .b1 _ _ _ => some 20
  | .str .b1 _ _ _ => some 21
  | .ldr .w4 _ _ _ => some 22
  | .str .w4 _ _ _ => some 23
  | .ldr .x8 _ _ _ => some 24
  | .str .x8 _ _ _ => some 25
  | .b false _ => some 26
  | .b true _ => some 27
  | .bcond _ _ => some 28
  | .cbz false _ _ _ => some 29
  | .cbz true _ _ _ => some 30
  | .ret _ => some 31
  | .movWide _ _ _ _ _ | .svc _ => none

private def singleTreeAux : Nat → Option Ins → Tree (Option Ins)
  | 0, v => .leaf v
  | d + 1, v => .node (singleTreeAux d v) (.leaf none)

private def singleImg (ins : List Nat) (pc : Nat) (decoded : Ins) : Img :=
  ⟨ins, pc, ins.length, singleTreeAux wdepth (some decoded)⟩

theorem singleImg_decodeAt (ins : List Nat) (pc : Nat) (d : Ins) (hlen : ins.length = 4) :
    (singleImg ins pc d).decodeAt pc = some (d, 4) := by
  simp [singleImg, Img.decodeAt, hlen, wdepth, singleTreeAux, Tree.get]

def evalCase (line : String) : String × Option Nat × Bool :=
  let caseStr := (line.splitOn " | ").head!
  let fs := caseStr.splitOn " "
  if fs.length != 37 then
    (s!"{line} | UNPARSED", none, true)
  else
    match bytesOf fs[0]!, parseHex fs[1]!, parseHex fs[2]!, parseHex fs[3]!,
          (fs.drop 4 |>.take 31).mapM parseHex, bytesOf fs[36]! with
    | some ins, some pc, some lo, some spVal, some regs, some memBytes =>
      let fl := fs[35]!.toList
      if ins.length != 4 || fl.length != 4 || memBytes.length != winSize then
        (s!"{line} | UNPARSED", none, true)
      else
        match decode (fun i => ins[i]?) with
        | none => (s!"{caseStr} | DECODE_REFUSED", none, true)
        | some (decoded, len) =>
          if len != 4 then
            (s!"{caseStr} | DECODE_LEN_MISMATCH", none, true)
          else
            match classifyForm decoded with
            | none => (s!"{caseStr} | UNEXPECTED_FORM", none, true)
            | some formIdx =>
              let b (c : Char) : Option Bool := some (c == '1')
              let flags : Flags := ⟨b fl[0]!, b fl[1]!, b fl[2]!, b fl[3]!⟩
              let s : St := {
                regs := regs.map some,
                sp := some spVal,
                pc := pc,
                fl := flags,
                mem := memBytes.map some,
                lo := lo,
                inp := [],
                out := []
              }
              let im := singleImg ins pc decoded
              match step im s with
              | .inl t =>
                let delta := (t.pc + pow2 64 - pc) % pow2 64
                (s!"{caseStr} | {hex16 delta} {stateStr t.sp t.regs t.fl t.mem}", some formIdx, false)
              | .inr (.exit c _) =>
                (s!"{caseStr} | EXIT_{c}", some formIdx, true)
              | .inr (.stuck _ why) =>
                (s!"{caseStr} | STUCK_{why}", some formIdx, true)
              | .inr .outOfFuel =>
                (s!"{caseStr} | OOF", some formIdx, true)
    | _, _, _, _, _, _ =>
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
  IO.eprintln s!"aarch64_diff model: {n} cases, {bad} failed"
  let mut minCount := if counts.isEmpty then 0 else counts[0]!
  for i in [:formNames.size] do
    let c := counts[i]!
    if c < minCount then minCount := c
    IO.eprintln s!"  form {formNames[i]!}: {c}"
  IO.eprintln s!"min_form_count={minCount}"
  return (if bad == 0 then 0 else 1)
