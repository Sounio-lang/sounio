# The Associator-Shadow Experiment — the non-associative lever, executed

**Status:** Level 2 (machine-checked separation). **Not** Level 3 (no new bound).

**Lean artifact:** `formal/lean4/SounioAssociatorShadow.lean` — builds green
(`lake build SounioAssociatorShadow`, ~2.7 s, no `sorry`, no Mathlib, all `native_decide`).
**Census mirror:** `scripts/research/associator_shadow_validate.py`.

**Verified substrate (unchanged):** `SounioZeroDivisorBridge` (`sedSigma`, `PrimSed`,
`validPrims = 84`, `orderedZDPairs = 336`, `isZeroPair`), Fano structure of
`SounioCayleyDickson`. No language/compiler/runtime changes.

---

## 0. What was deferred, and why we ran it

`SounioFanoArcsBlocking` proved the **linear** zero-divisor surgery (`UNLEARN`, right
multiplication) is *trapped*: its Fano point-shadows are always one of the **7 hyperovals**
(line-complements, all 4-arcs). Both that file and `SounioErdosUnitDistance` flagged the
same principled next lever: does **genuine non-associativity** do more? This note executes
that test.

For a zero-divisor pair `u · v = 0` the associator collapses to one product:

> `[p, u, v] = (p·u)·v − p·(u·v) = (p·u)·v`   (because `u·v = 0`).

So the **associator shadow** of a triple `(p, u, v)` with `u·v = 0` is the Fano
point-shadow (lo-class image in `{1..7}`) of the sedenion `(p·u)·v`. We range `p` over the
84 `validPrims` and `(u,v)` over the 336 `orderedZDPairs`.

---

## 1. Result: the lever fires (and exactly how)

Over all `84 × 336 = 28 224` triples, the associator shadows form **64 distinct sets** —
precisely **every subset of `{1..7}` of size ≤ 3**:

| size | count | what they are |
|------|-------|---------------|
| 0 | 1 | `∅` |
| 1 | 7 | every point |
| 2 | 21 | every pair |
| 3 | 35 | every triple — **28 triangles (arcs) + the 7 Fano lines (non-arcs)** |
| ≥4 | 0 | **none** |

Contrast with the linear surgery: **7 shadows, all 4-element hyperovals.**

Machine-checked statements (`SounioAssociatorShadow.lean`):

- `assoc_reaches_every_fano_line` + `witness_lines_are_the_seven` — for **each** of the 7
  Fano lines there is a genuine ZD pair `u·v = 0` and a probe `p` with
  `assocShadow p u v = L` (explicit per-line witnesses).
- `assoc_escapes_hyperovals` — those 7 shadows are **not** hyperovals and **not** arcs
  (each *is* a Fano line). The linear surgery can never produce a line.
- `assoc_shadow_max_size_le_3` — **the ceiling.** Every associator shadow over all 28 224
  triples has ≤ 3 points; the lever never produces a 4-set.
- `assoc_shadow_never_hyperoval`, `associator_shadow_summary`.

The full characterization (family = `{S ⊆ {1..7} : |S| ≤ 3}`) is validated in the Python
mirror (`distinct assoc shadows == {all S⊆{1..7}, |S|≤3}? True`).

---

## 2. Honest reading: a real separation, not a bound

**The lever has genuine geometric teeth.** It escapes the 7-hyperoval trap (64 ≠ 7), it
breaks the arc-invariance that the linear surgery could not (`UNLEARN` shadows are *always*
arcs), and it reaches the **7 Fano lines** — the collinear configurations the linear lever
structurally avoids. This is the first machine-checked evidence that genuine
non-associativity acts differently on incidence than any linear ZD map. That is exactly the
"first sign the lever exists" we set as the bar.

**But it escapes downward.** The associator never produces a shadow of size ≥ 4 — it sweeps
the *sub-maximal skeleton* of `PG(2,2)` (all points/pairs/triples), not any new maximal or
larger arc. Since `PG(2,2)` is fully understood and its max arc is 4 regardless, reaching
lines and triangles changes the incidence **type** without yielding any new or improved
configuration. So:

- **Linear surgery = the top:** the 7 maximal 4-arcs (hyperovals).
- **Associator surgery = the bottom:** the entire `|S| ≤ 3` skeleton, including the 7 lines.

They are **complementary co-flats** of the same tiny plane. This is a clean **Level-2
separation between the linear and non-associative regimes** — and an honest **negative for
Level 3**: the non-associative lever, on this finite `PG(2,2)` shadow, does not manufacture
a new extremal object. It enriches the family from 7 to 64 sets but stays inside the
already-known combinatorics.

---

## 3. What this does and doesn't change for the program

- It **closes** the deferred milestone with a definite answer: the associator lever is real
  and measurably richer than linear surgery, *and* it is bounded (≤ 3) on `PG(2,2)`.
- It **does not** move any named bound. Consistent with the standing obstruction: a fixed
  16-D (level-4) slice cannot reach the asymptotic/large-order regime where the open
  problems live (see `fano-arcs-blocking-sounio-note.md` §1, `sunflower-168-sounio-note.md`
  §4).
- The genuine next lever, if one wants to keep pushing, is **dimension, not parenthesization**:
  the same associator construction on a *larger* plane / a Cayley–Dickson level > 4, where
  "size ≤ 3 skeleton" could become a non-trivial arc/blocking statement. That requires new
  formalization (a tower), not new compiler work — and it is where the honest line between
  Level 2 and Level 3 currently sits.

## Reproduce

```bash
cd formal/lean4 && lake build SounioAssociatorShadow      # green, ~2.7 s
python3 scripts/research/associator_shadow_validate.py    # full census + characterization
```
