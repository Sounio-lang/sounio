/-
# Sounio — file-loaded-style reflection for souc_sat LRAT certificates ("souc_check")

Lean core's verified LRAT checker (`Std.Tactic.BVDecide.LRAT.check`/`check_sound`)
proves a CNF unsatisfiable from an `Array IntAction` certificate. Embedding a large
certificate *as a term* (one `Action.addRup id #[…] #[…]` per LRAT line) hits a
term-size wall: ~1.5 k actions (K₇/6) already cost minutes; G₅₂₉'s 98 616 actions
are out of reach that way.

`bv_decide`/`bv_check` avoid this by never materialising the certificate as a term:
they load the LRAT and reflect over file/array data. We do the analogous thing with
**zero new trust**: embed the (renumbered) LRAT as a *single `String` literal* — an
atom the elaborator handles in milliseconds — and parse it to `Array IntAction`
*inside the reflective computation*, so parsing + checking both run as compiled
native code under `native_decide`.

The parser is **unverified and soundness-irrelevant**: `check_sound` guarantees that
*if the verified checker accepts the parsed actions* then the CNF is UNSAT, no matter
how those actions were produced. A buggy parser can only make the proof fail, never
make an unsound proof succeed.

LRAT id convention (must hold of the input): addition ids are contiguous starting at
`M+1` (M = #original clauses), because Lean's `DefaultFormula.insert` *pushes* added
clauses. `examples/erdos/gen_lean_sat_reflect.sh` renumbers drat-trim output to this.
-/
import Std.Tactic.BVDecide.LRAT.Checker

open Std.Sat
open Std.Tactic.BVDecide.LRAT

namespace SounioSatReflect

/-- Parse a (contiguous-id) LRAT text into verified-checker actions.
Lines: `id lit* 0 hint* 0` (RUP add; empty `lit*` ⇒ empty clause) or `id d id* 0`
(deletion). Unverified — see module docstring. -/
def parseLRAT (s : String) : Array IntAction := Id.run do
  let mut acc : Array IntAction := #[]
  for line in s.splitOn "\n" do
    let toks := (line.splitOn " ").filter (fun t => t ≠ "")
    if toks.length = 0 then continue
    else if toks.length ≥ 2 ∧ toks[1]! == "d" then
      let mut ids : Array Nat := #[]
      for t in toks.drop 2 do
        let n := t.toNat!
        if n == 0 then break
        ids := ids.push n
      acc := acc.push (Action.del ids)
    else
      let id := toks[0]!.toNat!
      let rest := toks.drop 1
      let mut lits : Array Int := #[]
      let mut i := 0
      while i < rest.length do
        let v := rest[i]!.toInt!
        if v == 0 then break
        lits := lits.push v
        i := i + 1
      i := i + 1
      let mut hints : Array Nat := #[]
      while i < rest.length do
        let v := rest[i]!.toInt!
        if v == 0 then break
        hints := hints.push v.toNat
        i := i + 1
      if lits.isEmpty then acc := acc.push (Action.addEmpty id hints)
      else acc := acc.push (Action.addRup id lits hints)
  return acc

end SounioSatReflect
