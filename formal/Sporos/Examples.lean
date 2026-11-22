/-
Example Sporos programs, checked by the checker and round-tripped through the
bytecode, and `sporos-examples DIR`, which writes them as `.spb` files.
-/
import Sporos.Bytecode
import Sporos.Run

namespace Sporos.Examples

/-- Print a string from the read-only data. -/
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

/-- Copy the input to the output, upper-casing ASCII letters. -/
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

/-- Print `n` in decimal: `( n -- )`, recursive. -/
def printDec : Word where
  ins := 1
  outs := 0
  locals := 0
  code := [.prim .dup, .lit 10, .prim .ltu, .jnz 8,
           .prim .dup, .lit 10, .prim .divu, .call 1,
           .lit 10, .prim .modu, .lit 48, .prim .add, .prim .putb, .ret]
  ann := [(8, ⟨1, 0⟩)]

/-- The first twenty Fibonacci numbers, one per line. -/
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

def all : List (String × Prog) := [("echo", echo), ("hello", hello), ("upper", upper), ("fib", fib)]

/-- Each example passes the checker and survives the bytecode round trip;
`sporos-examples` refuses to write any that does not. -/
def valid (p : Prog) : Bool := checkProg p && ((decode p.enc).map Prog.enc) == some p.enc

end Sporos.Examples

open Sporos Sporos.Examples in
def main (args : List String) : IO UInt32 := do
  let dir := args.head?.getD "."
  IO.FS.createDirAll dir
  for (name, p) in all do
    if !valid p then
      (← IO.getStderr).putStrLn s!"sporos-examples: {name} fails the checker or the round trip"
      return 1
    IO.FS.writeBinFile s!"{dir}/{name}.spb" (ByteArray.mk (p.enc.map Nat.toUInt8).toArray)
  return 0
