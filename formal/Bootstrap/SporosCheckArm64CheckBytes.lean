/-
Checks that `SporosCheckArm64Bin.spr1Arm64Bytes` in
`formal/Bootstrap/SporosCheckArm64BinData.lean` matches the committed static
EM_AARCH64 binary `bootstrap/model/spr1_check_arm64` byte-for-byte via
`IO.FS.readBinFile`.

Usage (from `formal/`):
  lake env lean --run Bootstrap/SporosCheckArm64CheckBytes.lean [PATH]
-/
import Bootstrap.SporosCheckArm64BinData

def firstDiff (xs ys : List Nat) (idx : Nat := 0) : Option (Nat × Nat × Nat) :=
  match xs, ys with
  | [], [] => none
  | x :: xs', y :: ys' =>
    if x == y then firstDiff xs' ys' (idx + 1)
    else some (idx, x, y)
  | x :: _, [] => some (idx, x, 999)
  | [], y :: _ => some (idx, 999, y)

def main (args : List String) : IO UInt32 := do
  let path := args.headD "../bootstrap/model/spr1_check_arm64"
  let disk ← IO.FS.readBinFile path
  let diskBytes := disk.toList.map UInt8.toNat
  let modelBytes := SporosCheckArm64Bin.spr1Arm64Bytes
  if diskBytes == modelBytes then
    IO.println s!"spr1Arm64Bytes == {path} ({modelBytes.length} bytes, byte-for-byte match)"
    return 0
  else
    IO.eprintln s!"MISMATCH: {path} ({diskBytes.length} bytes) vs spr1Arm64Bytes ({modelBytes.length} bytes)"
    match firstDiff diskBytes modelBytes with
    | some (idx, d, m) => IO.eprintln s!"first diff at offset {idx}: disk={d} model={m}"
    | none => pure ()
    return 1
