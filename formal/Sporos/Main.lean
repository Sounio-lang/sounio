/-
`sporos`: the reference interpreter, as a program.

    sporos PROGRAM.spb [MEMORY-BYTES]  < input  > output

It reads the bytecode file (decision 32), refuses it unless the checker accepts
it (the checker is inside the interpreter), runs it on standard
input and writes its output. The run is the DAG semantics with hash-consing and
the collector (`drunC` with `Alloc.hashcons`), which `drunC_init` proves equal to
the reference semantics `run` for every collection policy.

The collector is off (`gcPolicy` is constantly false). Measured on 2026-11-16:
`upper` on 400 bytes of input takes 2.0 s without it and is killed for lack of
memory after 80 s with a collection every 4096 steps. The cause is that
`DagGC.nodes` walks the DAG as a tree, which is exponential in the sharing.
Turning the collector off changes no output, because `drunC_init` holds for
every policy.

Exit status: the program's own status when it halts; 1 when the file does not
decode, the checker refuses it, or the run fails (the error is printed on
standard error).

This is the executable reference: correct by the theorems above, not fast.
Measured on 2026-11-16, `upper` on 50, 100, 200, 400 and 800 bytes of input
takes 0.04, 0.12, 0.46, 2.0 and 9.9 s, which is roughly quadratic. The arena is a
list, and `hbuild` searches it linearly. The hand-written x86 interpreter is
measured against this reference.
-/
import Sporos.Bytecode
import Sporos.DagGC
import Sporos.DagHash

namespace Sporos

/-- When to collect: never, until `DagGC.nodes` is linear (see the header). -/
def gcPolicy (_ : DSt) : Bool := false

instance : Inhabited DRes := ⟨.fail .noFrame⟩

/-- Run in chunks of `drunC` until the program halts or fails. -/
partial def runAll (pr : Prog) (s : DSt) : DRes :=
  match drunC Alloc.hashcons pr noPhys gcPolicy 65536 s with
  | .next t => runAll pr t
  | r => r

def errName : Err → String
  | .underflow => "underflow" | .auxUnderflow => "aux-underflow" | .badLocal => "bad-local"
  | .badJump => "bad-jump" | .badWord => "bad-word" | .noFrame => "no-frame"
  | .bounds => "bounds" | .divZero => "div-zero" | .badTable => "bad-table" | .demand => "demand"

end Sporos

open Sporos in
def main (args : List String) : IO UInt32 := do
  let stderr ← IO.getStderr
  match args with
  | path :: rest =>
    let msize := (rest.head?.bind String.toNat?).getD (1 <<< 24)
    let bytes ← IO.FS.readBinFile path
    match decode (bytes.toList.map UInt8.toNat) with
    | none => stderr.putStrLn s!"sporos: {path}: not a Sporos bytecode file"; return 1
    | some pr =>
      if !checkProg pr then
        stderr.putStrLn s!"sporos: {path}: refused by the checker"; return 1
      let stdin ← IO.getStdin
      let inp ← stdin.readBinToEnd
      match runAll pr (dinit pr msize (inp.toList.map UInt8.toNat)) with
      | DRes.halt c out _ =>
        let stdout ← IO.getStdout
        stdout.write (ByteArray.mk (out.reverse.map (fun x => x.1.toUInt8)).toArray)
        stdout.flush
        return c.toUInt32
      | DRes.fail e => stderr.putStrLn s!"sporos: run failed: {errName e}"; return 1
      | DRes.next _ => return 1
  | [] => stderr.putStrLn "usage: sporos PROGRAM.spb [MEMORY-BYTES] < input > output"; return 1
