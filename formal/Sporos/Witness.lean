/-
Discriminating witnesses for two properties of single primitives, checked by
evaluation.

* `strhash` and `streq` carry the provenance of the bytes they read (PR #3594,
  review of 2026-11-15). Each byte of the string has an origin `u` that no
  operand carries; the result must carry `u`, and its word must not depend on
  the bytes' provenance.
* `copy` reads every destination byte from the memory before the copy, so
  overlapping blocks behave as memmove: [1,2,3,4], copy 0 → 1 of 3 bytes,
  gives [1,1,2,3], not [1,1,1,1].
-/
import Sporos.Run

namespace Sporos

/-- The origin of the string's bytes, and an instruction's origin. -/
def uo : Origin := ⟨some 99, 7, 7, 7⟩
def io : Origin := ⟨none, 0, 0, 0⟩

/-- Memory holding `bs` from address 0, every byte with provenance `p`. -/
def memOf (bs : List Nat) (p : Prov) : Nat → Val := fun a =>
  match bs[a]? with
  | some b => ⟨b, p⟩
  | none => ⟨0, []⟩

def stOf (bs : List Nat) (p : Prov) : St :=
  { ds := [], aux := [], rs := [], mem := memOf bs p, msize := 16, hp := 8, tabs := [],
    inp := [], ipos := 0, out := [], pc := [], step := 0 }

/-- The single result of a primitive: its word and whether its provenance has `uo`. -/
def resOf : PR → Option (Nat × Bool)
  | .ok [v] _ => some (v.w, v.p.contains uo)
  | _ => none

theorem strhash_carries_bytes :
    resOf (execPrim noPhys .strhash [⟨2, []⟩, ⟨0, []⟩] (stOf [97, 98] [uo]) io) =
      some ((resOf (execPrim noPhys .strhash [⟨2, []⟩, ⟨0, []⟩] (stOf [97, 98] []) io)).map Prod.fst
        |>.getD 0, true) := by decide

theorem strhash_without_bytes_has_no_u :
    (resOf (execPrim noPhys .strhash [⟨2, []⟩, ⟨0, []⟩] (stOf [97, 98] []) io)).map Prod.snd = some false := by
  decide

/-- Two equal strings at 0 and 2: equal, and the result carries the bytes' origin. -/
theorem streq_carries_bytes :
    resOf (execPrim noPhys .streq [⟨2, []⟩, ⟨2, []⟩, ⟨2, []⟩, ⟨0, []⟩] (stOf [97, 98, 97, 98] [uo]) io) =
      some (1, true) := by decide

/-- Different lengths: no byte is compared, so none of their origins enters. -/
theorem streq_lengths_differ :
    resOf (execPrim noPhys .streq [⟨1, []⟩, ⟨2, []⟩, ⟨2, []⟩, ⟨0, []⟩] (stOf [97, 98, 97, 98] [uo]) io) =
      some (0, false) := by decide

def bytesAfter : PR → Option (List Nat)
  | .ok _ s => some ((List.range 4).map fun i => (s.mem i).w)
  | _ => none

theorem copy_overlap :
    bytesAfter (execPrim noPhys .copy [⟨3, []⟩, ⟨1, []⟩, ⟨0, []⟩] (stOf [1, 2, 3, 4] []) io) =
      some [1, 1, 2, 3] := by decide

end Sporos
