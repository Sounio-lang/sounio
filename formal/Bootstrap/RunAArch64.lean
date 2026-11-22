/-
RunAArch64: CLI runner executing a static EM_AARCH64 ELF64 binary over
`AArch64Min.exec`.

Usage:
  aarch64run <elf-file> [fuel]
    Reads stdin bytes, runs `AArch64Min.exec fuel elfBytes stdinBytes`, writes
    stdout bytes, and exits with the program's exit status (or 2 on load/stuck/OOF).

  aarch64run --batch <elf-file> <dir> <count> [fuel]
    Runs `AArch64Min.exec` over `<dir>/<i>.in` for `i` in `0..count-1` and
    verifies exact agreement on exit status and stdout bytes against `<dir>/<i>.rc`
    and `<dir>/<i>.out`.
-/
import Bootstrap.AArch64Min

open AArch64Min

/-- Tail-recursive mirror of `AArch64Min.run` for stack-safe native/VM execution. -/
private def runTail (im : Img) : Nat → St → Outcome
  | 0, _ => .outOfFuel
  | n + 1, s => match step im s with
    | .inl t => runTail im n t
    | .inr o => o

private theorem runTail_eq_run (im : Img) :
    ∀ (fuel : Nat) (s : St), runTail im fuel s = run im fuel s
  | 0, _ => rfl
  | n + 1, s => by
    rw [run_succ]; unfold runTail
    cases step im s with
    | inl t => exact runTail_eq_run im n t
    | inr o => rfl

/-- Tail-recursive mirror of `AArch64Min.exec`, proved identical to `exec`. -/
private def execTail (fuel : Nat) (f inp : List Nat) : Outcome :=
  match load f 0x7ffffffde000 inp with
  | none => .stuck 0 3
  | some (im, s) => runTail im fuel s

private theorem execTail_eq_exec (fuel : Nat) (f inp : List Nat) :
    execTail fuel f inp = exec fuel f inp := by
  unfold execTail exec
  cases load f 0x7ffffffde000 inp with
  | none => rfl
  | some q => exact runTail_eq_run q.1 fuel q.2

private def readAllStdin : IO ByteArray := do
  let stdin ← IO.getStdin
  let mut buf := ByteArray.empty
  repeat do
    let chunk ← stdin.read 4096
    if chunk.isEmpty then break
    buf := buf ++ chunk
  return buf

private def byteArrayToNats (ba : ByteArray) : List Nat :=
  (ba.data.toList).map UInt8.toNat

private def natsToByteArray (ns : List Nat) : ByteArray :=
  ByteArray.mk ((ns.map UInt8.ofNat).toArray)

private def runSingle (elfPath : System.FilePath) (fuel : Nat) : IO UInt32 := do
  let elfBa ← IO.FS.readBinFile elfPath
  let inpBa ← readAllStdin
  let elf := byteArrayToNats elfBa
  let inp := byteArrayToNats inpBa
  match execTail fuel elf inp with
  | .exit st out =>
    let stdout ← IO.getStdout
    stdout.write (natsToByteArray out)
    stdout.flush
    return UInt32.ofNat (st % 256)
  | .stuck pc why =>
    IO.eprintln s!"aarch64run: stuck at pc=0x{String.ofList (Nat.toDigits 16 pc)} why={why}"
    return 2
  | .outOfFuel =>
    IO.eprintln s!"aarch64run: out of fuel ({fuel})"
    return 2

private def runBatch (elfPath : System.FilePath) (dir : System.FilePath) (count fuel : Nat) : IO UInt32 := do
  let elfBa ← IO.FS.readBinFile elfPath
  let elf := byteArrayToNats elfBa
  let some (im, s0) := load elf 0x7ffffffde000 []
    | IO.eprintln "aarch64run: failed to load ELF"; return 2
  let mut bad := 0
  let mut acc := 0
  let mut rej := 0
  for i in [:count] do
    let base := dir / s!"{i}"
    let inpBa ← IO.FS.readBinFile (base.toString ++ ".in")
    let wantOutBa ← IO.FS.readBinFile (base.toString ++ ".out")
    let wantRcStr ← IO.FS.readFile (base.toString ++ ".rc")
    let some wantRc := wantRcStr.trimAscii.toString.toNat?
      | IO.eprintln s!"aarch64run: invalid rc in {base}.rc"
        return 2
    let inp := byteArrayToNats inpBa
    match runTail im fuel { s0 with inp := inp } with
    | .exit st out =>
      let wantOut := byteArrayToNats wantOutBa
      if st != wantRc || out != wantOut then
        if bad < 5 then
          IO.eprintln s!"FAIL fuzz case {i}: got rc={st} len={out.length}, want rc={wantRc} len={wantOut.length}"
        bad := bad + 1
      if wantRc == 0 then acc := acc + 1 else rej := rej + 1
    | .stuck pc why =>
      if bad < 5 then
        IO.eprintln s!"FAIL fuzz case {i}: stuck at pc=0x{String.ofList (Nat.toDigits 16 pc)} why={why}"
      bad := bad + 1
    | .outOfFuel =>
      if bad < 5 then
        IO.eprintln s!"FAIL fuzz case {i}: out of fuel ({fuel})"
      bad := bad + 1
  if bad == 0 then
    IO.println s!"ok   aarch64run batch fuzz: {count} cases ({acc} accepted, {rej} refused)"
    return 0
  else
    IO.eprintln s!"FAIL aarch64run batch fuzz: {bad} of {count} differ"
    return 1

def main (args : List String) : IO UInt32 := do
  let defaultFuel : Nat := 5000000
  match args with
  | ["--batch", elfPath, dir, countStr] =>
    match countStr.toNat? with
    | some count => runBatch elfPath dir count defaultFuel
    | none => IO.eprintln "usage: aarch64run --batch <elf> <dir> <count> [fuel]"; return 2
  | ["--batch", elfPath, dir, countStr, fuelStr] =>
    match countStr.toNat?, fuelStr.toNat? with
    | some count, some fuel => runBatch elfPath dir count fuel
    | _, _ => IO.eprintln "usage: aarch64run --batch <elf> <dir> <count> [fuel]"; return 2
  | [elfPath] => runSingle elfPath defaultFuel
  | [elfPath, fuelStr] =>
    match fuelStr.toNat? with
    | some fuel => runSingle elfPath fuel
    | none => IO.eprintln "usage: aarch64run <elf> [fuel]"; return 2
  | _ =>
    IO.eprintln "usage: aarch64run <elf> [fuel] | --batch <elf> <dir> <count> [fuel]"
    return 2
