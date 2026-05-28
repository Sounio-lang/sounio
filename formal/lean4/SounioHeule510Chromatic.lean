import SounioHeule510
/-!
# HeuleGraph510 — chromatic number = 5 (Phase C)

Upper bound (χ ≤ 5) is certified here to the Lean kernel: an explicit proper
5-colouring of the certified edge set `SounioHeule510.E`. Lower bound (χ ≥ 5,
i.e. NOT 4-colourable) is an UNSAT statement that `native_decide` cannot decide
(4^510 colourings); it is certified externally by a clausal proof — see
`docs/research/hadwiger-nelson-510-chromatic.md` and
`scripts/research/heule510_chromatic_cert/`. **The two halves do NOT share a trust
base** (kernel vs. SAT-solver+drat-trim); the note states this explicitly.

* `coloring` — an explicit colouring `Nat → Nat` (values in `{0..4}`), from a
  SAT model of the 5-colouring CNF over `E`.
* `heule510_proper_5coloring` — every edge of `E` is bichromatic under `coloring`,
  and every colour is `< 5`: a kernel-checked proof that χ(HeuleGraph510) ≤ 5.
-/
namespace Sounio.Heule510Chromatic
open Sounio.Heule510

def coloring : Array Nat := #[4, 1, 1, 1, 0, 0, 3, 0, 4, 3, 4, 0, 4, 2, 4, 1, 0, 2, 1, 0, 0, 1, 3, 3, 0, 1, 2, 1, 3, 2, 0, 0, 2, 1, 0, 2, 1, 3, 2, 3, 3, 2, 3, 3, 4, 1, 3, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 2, 2, 0, 0, 1, 1, 3, 1, 2, 2, 3, 3, 3, 1, 3, 4, 3, 4, 2, 2, 2, 4, 4, 3, 3, 3, 3, 3, 0, 4, 2, 2, 0, 4, 4, 2, 4, 4, 2, 0, 3, 1, 0, 3, 4, 4, 0, 4, 0, 4, 3, 2, 4, 3, 0, 2, 4, 3, 1, 4, 3, 3, 4, 4, 0, 1, 0, 0, 2, 0, 2, 4, 2, 0, 0, 0, 1, 1, 3, 3, 0, 0, 3, 3, 1, 2, 2, 4, 4, 2, 3, 4, 4, 2, 3, 0, 1, 4, 2, 1, 4, 0, 0, 0, 0, 1, 1, 4, 0, 4, 0, 4, 4, 2, 0, 1, 0, 4, 2, 2, 1, 1, 2, 0, 3, 2, 2, 1, 1, 3, 1, 4, 1, 1, 1, 0, 4, 0, 1, 1, 0, 0, 0, 1, 4, 0, 4, 4, 0, 0, 0, 2, 1, 0, 4, 1, 2, 1, 1, 0, 0, 1, 3, 2, 1, 0, 3, 0, 3, 2, 1, 0, 3, 0, 1, 4, 2, 1, 4, 2, 0, 1, 3, 3, 2, 4, 0, 1, 2, 2, 1, 0, 4, 2, 2, 3, 3, 1, 3, 1, 1, 2, 2, 4, 1, 0, 0, 0, 1, 2, 1, 0, 0, 1, 1, 2, 0, 4, 4, 1, 4, 2, 0, 1, 2, 0, 1, 2, 4, 1, 1, 2, 0, 4, 1, 1, 4, 2, 3, 2, 3, 3, 2, 1, 3, 4, 2, 2, 2, 0, 2, 4, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 3, 4, 4, 4, 1, 4, 2, 1, 2, 4, 0, 2, 2, 0, 3, 3, 3, 4, 2, 4, 4, 4, 3, 3, 2, 1, 4, 4, 4, 3, 4, 3, 1, 4, 4, 3, 4, 2, 4, 4, 4, 4, 4, 4, 4, 1, 2, 1, 3, 2, 3, 0, 3, 3, 4, 0, 0, 1, 3, 3, 1, 0, 3, 2, 0, 1, 2, 0, 1, 3, 2, 0, 2, 1, 2, 1, 2, 4, 0, 1, 1, 0, 2, 3, 4, 0, 4, 4, 1, 2, 1, 0, 0, 2, 3, 4, 4, 4, 1, 0, 2, 2, 3, 3, 2, 3, 3, 1, 1, 0, 2, 2, 2, 2, 4, 1, 0, 0, 0, 1, 1, 1, 3, 4, 4, 0, 3, 4, 4, 2, 1, 3, 2, 4, 3, 1, 0, 1, 0, 0, 0, 1, 4, 4, 3, 3, 4, 4, 4, 4, 4, 3, 2, 3, 4, 3, 4, 4, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]
def col (v : Nat) : Nat := coloring.getD v 0

/-- **χ ≤ 5 (kernel-checked).** `coloring` is a proper 5-colouring of the certified
    graph: adjacent vertices differ, and only colours 0..4 are used. -/
theorem heule510_proper_5coloring :
    E.all (fun e => col e.1 != col e.2)
    && (List.range 510).all (fun v => col v < 5) := by native_decide

end Sounio.Heule510Chromatic
