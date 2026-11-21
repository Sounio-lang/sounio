/-
Byte-for-byte identity check between `SporosVmBin.vmBytes` and the committed
binary `bootstrap/model/sporos_vm` read via `IO.FS.readBinFile`.

Root A framing:
- Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
- `spr1_check` is an independent second check (not Root A or Root A's trust base).
- `sporos_vm` is the post-admission native x86-64 bytecode VM paired with
  `Sporos.decode` + `Sporos.checkProg`.
-/
import Bootstrap.SporosVmBinData

namespace SporosVmCheckBytes

def findBinPath : IO System.FilePath := do
  let candidates : List System.FilePath :=
    [ System.FilePath.mk "bootstrap/model/sporos_vm"
    , System.FilePath.mk "../bootstrap/model/sporos_vm"
    ]
  for p in candidates do
    if ← p.pathExists then
      return p
  throw <| IO.userError "SporosVmCheckBytes: cannot locate bootstrap/model/sporos_vm"

def checkVmBytes (path : System.FilePath) : IO Unit := do
  let raw ← IO.FS.readBinFile path
  let bytes := raw.toList.map (·.toNat)
  if bytes == SporosVmBin.vmBytes then
    IO.println s!"SporosVmCheckBytes: exact match ({bytes.length} bytes)"
  else
    throw <| IO.userError
      s!"SporosVmCheckBytes: byte mismatch (file={bytes.length}, vmBytes={SporosVmBin.vmBytes.length})"

#eval show IO Unit from do
  let p ← findBinPath
  checkVmBytes p

end SporosVmCheckBytes
