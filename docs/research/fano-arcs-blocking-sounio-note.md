<!-- docs:meta
topic_id: repo.docs.research.fano-arcs-blocking-sounio-note
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.fano-arcs-blocking-sounio-note
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Fano Arcs, Hyperovals, and Blocking Sets via the Verified 168 / ZD / Surgical Machine

**Status:** Level 1 (conceptual bridge) + Level 2 (machine-checked witnesses) complete.
Level 3 (improvement on a named open bound) explicitly *not* claimed — see §4, §6.

**Lean artifact:** `formal/lean4/SounioFanoArcsBlocking.lean` — builds green
(`lake build SounioFanoArcsBlocking`, 7 jobs, no `sorry`, no Mathlib, all
`native_decide`). Pre-validation mirror: `scripts/research/fano_arcs_blocking_validate.py` (Python check of
every theorem statement before formalization).

**Verified substrate used (unchanged):**
`SounioCayleyDickson` (`non_fano_count_168`, `fanoLines`), `SounioZeroDivisorBridge`
(`PrimSed`, `validPrims = 84`, `orderedZDPairs = 336`, `every_primitive_has_4_annihilators`,
`zd_pairs_intra_fiber`), `SounioSurgicalCalculus` (`applyOp`, `unlearn_card`,
`edit_card`, `gate_card`). No language/compiler/runtime changes.

---

## 1. The open problems we target (and the honest scope)

The prompt asks for *still-open* Erdős-style questions in the blocking-set / arc /
`(k,n)`-arc / hyperoval family, scoped tighter than Hadwiger–Nelson (#508/#704).

Two concrete, genuinely-open named candidates:

- **(Q1) Maximal `{k,n}`-arcs in non-Desarguesian planes.** Whether maximal arcs
  of every admissible degree exist in the Hall plane of order 16, and the open
  Mathon/Hamilton cases in non-Desarguesian planes of order 16/25, are unresolved.
  (Maximal arcs are *forbidden* in Desarguesian planes of odd order by
  Ball–Blokhuis–Mazzocca; the non-Desarguesian even/odd cases are the open frontier.)
- **(Q2) Hyperovals in non-Desarguesian planes of large even order** (e.g. order 64):
  classification is open even in the Desarguesian case and wide open otherwise.

**The lift obstruction (stated up front, not buried).** Both Q1 and Q2 live in
projective planes of order ≥ 9. A finite projective plane of order `n` is
coordinatized by a *planar ternary ring* (and the Moufang/translation cases by a
quasifield / semifield / alternative division ring). **Sedenions are not a division
ring or a quasifield** — they have zero divisors (`hasZeroDivisors 4 = true`,
`SounioCayleyDickson`). So the 16-dimensional sedenion algebra **cannot
coordinatize** a non-Desarguesian plane of order ≥ 9 in the classical sense. This
is a real wall, and we do not pretend otherwise.

What the verified machine *does* give, exactly and auditably, is:

1. a **finite shadow** of the arc/blocking theory inside the Fano plane `PG(2,2)`,
   the unique plane of order 2, which the octonion/sedenion structure *does*
   canonically coordinatize (the 7 Fano lines are the octonion associative cycles);
2. a **controlled twisting mechanism** — the 7 zero-divisor fibers × the 6 surgical
   operations — that acts on this shadow with provable incidence semantics.

That is a defensible Level 1–2 result. Level 3 (perturbing a *named* large-order
plane) is left as the explicit obligation in §5–§6.

---

## 2. The conceptual model: fibers and ops as incidence operators

The base object is `PG(2,2)`: points `{1..7}`, lines `fanoLines =
[(1,2,3),(1,4,5),(1,6,7),(2,4,6),(2,5,7),(3,4,7),(3,5,6)]` (the XOR-zero triples,
`fano_line_xor_zero`). Each line = 3 points; each point on 3 lines; self-dual.

The sedenion side has **7 zero-divisor fibers** labelled `{9..15}` (`zd_fiber_labels`),
each with **12 primitives** (`zd_fiber_sizes`). The proved correspondence
`zd_labels_mirror_fano_indices` strips the high bit: fiber `L ↦ L ⊕ 8 ∈ {1..7}`,
identifying the **7 fibers with the 7 points** (equivalently, by Fano self-duality,
the 7 lines).

A primitive `u = e_lo ± e_hi` carries a **point coordinate** `u.lo ∈ {1..7}` and a
**fiber coordinate** `xorLabel u = u.lo ⊕ u.hi ∈ {9..15}`. The three "shape" ops of
the surgical calculus act on a primitive and we read off the **point-shadow**
(the lo-index image of the output primitives):

| Op | Algebraic meaning (verified) | Geometric shadow in `PG(2,2)` (proved here) |
|----|------------------------------|---------------------------------------------|
| **UNLEARN** `u` | the 4 right-annihilators (`unlearn_card = 4`) | complement of a **LINE** → a **hyperoval** (4-arc) |
| **EDIT** `u` | the 12 fiber-mates (`edit_card = 12`) | complement of a **POINT** (6 points) |
| **GATE** `u` | the 79 preserved primitives (`gate_card = 79`) | the **whole plane** (all 7 points) |

So the three ops sweep the three natural "co-flats" of the plane: co-line
(UNLEARN), co-point (EDIT), full (GATE). UNLEARN/EDIT are point↔line *dual* in the
Fano sense. This is the geometric content of the surgical calculus.

---

## 3. The bridge theorem (machine-checked)

The sharp new statement, proved by `native_decide` over all 84 primitives:

> **UNLEARN's kernel is a Fano hyperoval.** For every primitive `u`, the lo-index
> set of its 4 right-annihilators is exactly the complement of the unique Fano line
> through `u.lo` and the fiber-mate point `xorLabel u ⊕ 8`.

Theorem names in `SounioFanoArcsBlocking.lean`:

- `unlearn_kernel_card4` — the 4 annihilator lo-indices are distinct.
- `unlearn_kernel_is_arc` — they form a 4-arc (no 3 collinear).
- `unlearn_kernel_is_complement_of_line` — it is the complement of a Fano line.
- `unlearn_kernel_line_is_through_lo_and_fiber` — **which** line, exactly.
- `all_seven_hyperovals_realized` — the 84 kernels realize all 7 hyperovals.

Interpretation: the **"4D forgettable subspace"** of `SounioZeroDivisorBridge`
(`zd_selective_forgetting_summary`) is not just any 4-set — it is the *maximal arc*
of the plane, the unique largest configuration containing no line. Forgetting, in
this model, deletes precisely a hyperoval; the 12-dimensional co-kernel projects
onto the line that the hyperoval avoids.

Supporting classical facts, also machine-checked over all `2^7 = 128` subsets:

- `fano_max_arc_is_4`, `fano_has_seven_hyperovals` — max arc = 4, exactly 7 of them.
- `line_complements_are_arcs` — each line-complement is a 4-arc.
- `fano_no_nontrivial_blocking_set` (`numBlockingSets = 0`) — `PG(2,2)` has **no**
  nontrivial blocking set: every line-transversal contains a line. (Known small
  fact; here it is the auditable null that frames what surgery could/can't break.)

---

## 4. How one would coordinatize / apply ops / measure (sketch)

This is the recipe the Lean file instantiates, written so a larger-plane attempt
could follow the same shape:

1. **Coordinatize.** Map plane points to a label set that the algebra distinguishes.
   Here: point `p ∈ {1..7}` ↔ octonion index; line ↔ XOR-zero triple ↔ fiber
   `p ⊕ 8`. (`onLine`, `isComplementOfLine` in the Lean file.)
2. **Lift a configuration.** Choose a primitive `u` (a point/fiber pair); its
   annihilator set is the algebraic carrier of a sub-configuration.
3. **Apply surgery.** Run `applyOp .unlearn / .edit / .gate u`. Take the
   point-shadow `(·).map (·.lo)`.
4. **Measure incidence.** Evaluate `isArc`, `containsLine`, `meetsAllLines`,
   `isBlockingSet`, arc size (`maxArcSize`) on the shadow, and compare to the
   pre-surgery configuration — all decidable, all `native_decide`.

The measurement layer (`§1–§3` of the Lean file) is plane-agnostic Bool predicates
on `List Nat` point-sets; only the line table and the coordinatization change if
one later swaps `PG(2,2)` for a different incidence structure.

---

## 5. What counts as a real result — three levels

- **Level 1 (conceptual note).** *Done.* The explicit bridge: surgical ops =
  incidence co-flat operators; UNLEARN kernel = hyperoval; lift obstruction stated.
- **Level 2 (Lean witnesses / cardinalities).** *Done.* `SounioFanoArcsBlocking.lean`
  — 12 `native_decide` theorems including the bridge and the point/line duality.
- **Level 3 (named-bound improvement).** *Open.* Would require either
  (a) an associator-based ("genuinely non-associative") surgery
  `((p · u) · v)` with `u · v = 0` that perturbs incidence beyond the 7 fixed
  hyperovals — the principled next lever, mirroring the `SounioErdosUnitDistance`
  associator milestone — or (b) a construction that legitimately *embeds* a
  larger-order plane's sub-configuration into the 16D ZD geometry, which the §1
  obstruction shows cannot be a classical coordinatization.

---

## 6. Cheapest next concrete step (no new compiler work)

**The single cheapest step is already runnable today:**

```bash
cd formal/lean4 && lake build SounioFanoArcsBlocking   # green, ~0.7s for the lib
python3 scripts/research/fano_arcs_blocking_validate.py                          # independent census mirror
```

The *next* research increment, still requiring **zero** compiler/runtime work:

> Add an `associatorShadow` to `SounioFanoArcsBlocking.lean` — the lo-index image
> of `((primA · u) · v)` for ZD pairs `u · v = 0`, reusing the *already-defined*
> `primProd` / `sedSigma` of `SounioZeroDivisorBridge` (no new algebra). Then prove
> by `native_decide` whether any associator shadow is **not** one of the 7 fixed
> hyperovals. The linear (right-multiplication) surgery is now machine-checked to
> stay inside the 7 hyperovals; the associator route is the only remaining lever
> for a non-trivial incidence perturbation, exactly as in the unit-distance file.

This keeps the entire effort inside the verified 168 / ZD / Surgical-Calculus
boundary and inside `native_decide`.
