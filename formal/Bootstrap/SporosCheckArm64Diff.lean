/-
Differential oracle for `spr1_check_arm64`: runs `Sporos.decode` followed by
`Sporos.checkProg` directly on raw `.spb` files and agrees with
`SporosCheckArm64Flat.checkFlat`.

Usage (from `formal/`):
  lake env lean --run Bootstrap/SporosCheckArm64Diff.lean gen OUTDIR
  lake env lean --run Bootstrap/SporosCheckArm64Diff.lean --list LISTFILE
  lake env lean --run Bootstrap/SporosCheckArm64Diff.lean FILE.spb
-/
import Sporos.Bytecode
import Sporos.Check
import Sporos.Examples
import Bootstrap.SporosCheckArm64Flat

namespace SporosCheckArm64Diff

open Sporos
open SporosCheckArm64Flat

def verdict (bs : List Nat) : String :=
  match Sporos.decode bs with
  | some pr => if Sporos.checkProg pr then "accept" else "reject"
  | none => "reject"

theorem verdict_spec (bs : List Nat) (hfit : ∀ pr, Sporos.decode bs = some pr → Fits48 bs pr) :
    verdict bs = "accept" ↔ checkFlat bs = some true := by
  unfold verdict
  constructor
  · intro hv
    cases hd : Sporos.decode bs with
    | none => rw [hd] at hv; simp at hv
    | some pr =>
      rw [hd] at hv
      by_cases hc : Sporos.checkProg pr = true
      · exact checkFlat_complete hd hc (hfit pr hd)
      · have hf : Sporos.checkProg pr = false := Bool.eq_false_iff.2 hc
        simp [hf] at hv
  · intro hcf
    obtain ⟨pr, hd, hc⟩ := checkFlat_sound hcf
    simp [hd, hc]

theorem verdict_spec_small (bs : List Nat) (hby : ∀ b ∈ bs, b < 256) (hlen : bs.length < 65536) :
    verdict bs = "accept" ↔ checkFlat bs = some true := by
  rw [checkFlat_small_iff hby hlen]
  unfold verdict
  cases Sporos.decode bs with
  | none => simp
  | some pr => by_cases hc : Sporos.checkProg pr <;> simp [hc]

theorem oracle_examples_accept :
    verdict Sporos.echo.enc = "accept" ∧
    verdict Sporos.Examples.hello.enc = "accept" ∧
    verdict Sporos.Examples.upper.enc = "accept" ∧
    verdict Sporos.Examples.fib.enc = "accept" := by
  decide +kernel

theorem oracle_refusals :
    verdict [] = "reject" ∧
    verdict [0x53, 0x50, 0x52, 0x31] = "reject" ∧
    verdict (Sporos.echo.enc ++ [0]) = "reject" ∧
    verdict (Sporos.echo.enc.take 16) = "reject" := by
  decide +kernel

private def toByteArray (bs : List Nat) : ByteArray :=
  ⟨bs.map UInt8.ofNat |>.toArray⟩

private def writeExamples (dir : System.FilePath) : IO Unit := do
  IO.FS.createDirAll dir
  let items : List (String × Sporos.Prog) :=
    [("echo", Sporos.echo),
     ("hello", Sporos.Examples.hello),
     ("upper", Sporos.Examples.upper),
     ("fib", Sporos.Examples.fib)]
  for (name, pr) in items do
    IO.FS.writeBinFile (dir / s!"{name}.spb") (toByteArray pr.enc)

private def judgeFile (path : System.FilePath) : IO String := do
  let raw ← IO.FS.readBinFile path
  let bs := raw.toList.map UInt8.toNat
  return verdict bs

def main (args : List String) : IO UInt32 := do
  match args with
  | ["gen", dir] =>
    writeExamples dir
    return 0
  | ["--list", listPath] =>
    let txt ← IO.FS.readFile listPath
    for line in txt.splitOn "\n" do
      let p := line.trimAscii.toString
      if !p.isEmpty then
        let v ← judgeFile p
        IO.println s!"{v} {p}"
    return 0
  | [path] =>
    let v ← judgeFile path
    IO.println v
    return if v == "accept" then 0 else 1
  | _ =>
    IO.eprintln "usage: SporosCheckArm64Diff [gen DIR | --list FILE | FILE.spb]"
    return 2

end SporosCheckArm64Diff
