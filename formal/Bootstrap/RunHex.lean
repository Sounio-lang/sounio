/-
RunHex: run a hex image in the `X86Int` model, as the Linux kernel would start
it, and report where it ends. The tool a stage of the root-A bootstrap is first
checked with, before any theorem is stated about it.

  lean --run Bootstrap/RunHex.lean IMAGE.hex FILESIZE [STOP [FROM TO]]

`IMAGE.hex` is in format A (`Bootstrap.X86Min.Formats.decA`), loaded at 0x400000 and
entered at the address in its ELF header. The process gets `argv = [stage, /p]`
and a file `/p` of `FILESIZE` bytes (a repeating pattern), and a kernel oracle
that places each new mapping 4 GiB past the previous one, starting at
0x7f0000000000. Linux returns mmap addresses above 4 GiB, so a pointer kept or
advanced in a 32-bit register loses its upper half here as it does on the
machine; an oracle with addresses below 4 GiB would hide that. The run ends at
an exit, at a step outside the model, or at the address `STOP` if one is given;
the last instructions executed are printed. `FROM TO` is a what-if: when the run reaches
`FROM` it continues at `TO`, which stands in for a jump the image does not have
yet, so what lies beyond can be checked before the fix is written.
-/
import Bootstrap.X86Dec
import Bootstrap.Formats

open X86Env X86Int X86Dec

def hx (n : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 n)

/-- A decimal number, or hexadecimal with a `0x` prefix. -/
def parseNat (s : String) : Option Nat :=
  match s.toList with
  | '0' :: 'x' :: ds => ds.foldlM (fun acc c =>
      let v := if c.isDigit then some (c.toNat - 48)
        else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
      v.map (acc * 16 + ·)) 0
  | _ => s.toNat?

def pattern (n : Nat) : List Nat := (List.range n).map (· % 251)

def oracle : List Choice := highAddrs.map .at

def startState (code file : List Nat) : St :=
  let (stk, sp) := initStack 0x7ffffffff000 [[115, 116, 97, 103, 101, 49], [47, 112]]
  let regs := (List.replicate 16 (some 0)).set 4 (some sp)
  ⟨regs, leVal ((code.drop 0x18).take 8), Flags.undef,
    [⟨0x400000, code, false⟩, stk], ⟨[([47, 112], file)], [], [], [], oracle⟩⟩

def line (im : Nat → Option (Ins × Nat)) (s : St) : String :=
  match im s.rip with
  | some (i, _) => s!"{hx s.rip}: {repr i}"
  | none => s!"{hx s.rip}: undecodable"

partial def loop (im : Nat → Option (Ins × Nat)) (stop : Option Nat) (hop : Option (Nat × Nat))
    (fuel : Nat) (s : St) (tr : List String) : IO UInt32 := do
  let report (msg : String) : IO UInt32 := do
    IO.println msg
    IO.println "last instructions, oldest first:"
    for l in (tr.take 8).reverse do IO.println s!"  {l}"
    pure 0
  if stop == some s.rip then
    report s!"reached {hx s.rip}: rbx = {(s.reg 3).map hx}, r12 = {(s.reg 12).map hx}, r13 = {s.reg 13}"
  else if fuel == 0 then report "out of fuel"
  else if hop.map (·.1) == some s.rip then
    loop im stop hop (fuel - 1) { s with rip := (hop.map (·.2)).getD s.rip } (s!"{hx s.rip}: (what-if jump)" :: tr)
  else
    let t := line im s
    match step im s with
    | .ok s' => loop im stop hop (fuel - 1) s' (t :: tr)
    | .exit c => report s!"exit {c}, after the instruction at {t}"
    | .stuck => report s!"stuck at {t}; regs rax..rdi = {(List.range 8).map s.reg}, r10 = {s.reg 10}"

def main (args : List String) : IO UInt32 := do
  match args with
  | path :: size :: rest =>
    let txt ← IO.FS.readFile path
    match X86Min.Formats.decA (txt.toList.map Char.toNat) false none [] with
    | none => IO.println "not format A"; pure 2
    | some code =>
      let stop := rest.head?.bind parseNat
      let hop := match rest with
        | _ :: a :: b :: _ => (parseNat a).bind fun x => (parseNat b).map fun y => (x, y)
        | _ => none
      let s := startState code (pattern size.toNat!)
      IO.println s!"{code.length} bytes, entry {hx s.rip}, file of {size} bytes"
      loop (imageOf code 0x400000) stop hop 2000000 s []
  | _ => IO.println "usage: RunHex IMAGE.hex FILESIZE [STOP [FROM TO]]"; pure 2
