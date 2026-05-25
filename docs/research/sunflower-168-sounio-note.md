<!-- docs:meta
topic_id: repo.docs.research.sunflower-168-sounio-note
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sunflower-168-sounio-note
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sunflowers on the Verified 168 / ZD / Surgical Set System (Erdős [20])

**Status:** Level 1 (conceptual dictionary) + Level 2 (machine-checked witnesses)
complete. Level 3 (new bound / counterexample for the classical `f(n)`) explicitly
*not* claimed — see §1, §4.

**Lean artifact:** `formal/lean4/SounioSunflower.lean` — builds green
(`lake build SounioSunflower`, 7 jobs, ~6.6 s, no `sorry`, no Mathlib, all
`native_decide`). Census mirror: `scripts/research/sunflower_168_validate.py`.

**Verified substrate used (unchanged):** `SounioZeroDivisorBridge`
(`validPrims = 84`, `orderedZDPairs = 336`, `every_primitive_has_4_annihilators`,
`zd_degree_4`, `zd_pairs_intra_fiber`, `zd_fiber_sizes`), `SounioSurgicalCalculus`
(`applyOp`, `unlearn_card`, `edit_card`, `gate_card`), `SounioCayleyDickson` (the
168 / Fano governance). No language / compiler / runtime changes.

---

## 0. Sunflower vocabulary

A **sunflower (Δ-system) with `p` petals** is a family `A₁ … A_p` whose pairwise
intersections all equal one common **core** `Y`: `A_i ∩ A_j = Y` for `i ≠ j`. The
**petals** `A_i \ Y` are then pairwise disjoint. Empty core ⇔ pairwise-disjoint sets.
Erdős problem **[20]** is the sunflower conjecture: `f(n, 3) ≤ C^n` for an absolute
constant `C` (the size-`n`, 3-petal threshold).

---

## 1. The dictionary (Task 1)

The 168/ZD machine offers **two** natural set systems, and the sunflower reading is
exact in both.

### (A) Edge system — the 168 ZD classes as 2-element sets

The annihilation graph is 4-regular (`zd_degree_4`, `every_primitive_has_4_annihilators`).
Reading each ZD class `{u, a}` (with `u · a = 0`) as a 2-element set:

> **Every primitive `u` is the core of a sunflower with exactly 4 petals.**
> Its 4 incident classes `{u, a₁}, …, {u, a₄}` pairwise intersect in exactly `{u}`;
> the petals are the 4 annihilators `aᵢ` (distinct, none equal to `u`).

This is a *one-line reinterpretation of an already-proved theorem*
(`star_is_4petal_sunflower` in the Lean file restates `every_primitive_has_4_annihilators`
in sunflower language). The **core = the point `u`**; the **petals = `UNLEARN u`**.

### (B) Neighborhood system — the 84 annihilator 4-sets `UNLEARN u`

Here `core`/`petals` map onto the algebra's "forget" geometry:

| Surgical op | Cardinality (verified) | Sunflower meaning |
|-------------|------------------------|-------------------|
| **UNLEARN** `u` | 4 (`unlearn_card`) | the **petals** — `u`'s forgettable 4-set |
| **EDIT** `u`    | 12 (`edit_card`)   | the **parallel class** (fiber / "bouquet") containing `u` |
| **GATE** `u`    | 79 (`gate_card`)   | the **non-petal universe** (everything outside petals + self) |
| **COMPOSE** `u` | 83 | the full non-self basis (`compose = unlearn ⊔ gate`) |
| **AUDIT** `u`   | 5  | `{u} ∪ petals` — the petals *with their core point* |
| **REVIVE** `u`  | 4  | the petals again (`unlearn = revive` on the kernel) |

So in family terms: **UNLEARN extracts petals, EDIT names the parallel class, GATE
returns the non-petal universe, AUDIT bundles a flower with its core point.** All of
these are the verified `applyOp` cardinalities re-read; `surgical_sunflower_dictionary`
in the Lean file checks the three load-bearing ones together.

---

## 2. A natural "168-sunflower" problem and its answer (Task 2)

**Problem.** On the neighborhood system `F = { UNLEARN u : u ∈ validPrims }` (84 sets of
size 4 over the 84 primitives), what is the largest sunflower, and how is it constrained
by the fiber structure?

**Answer (all `native_decide`):**

- `ann_lies_in_one_fiber` — every neighborhood sits inside a single xor-fiber.
- `fibers_pairwise_disjoint` — the 7 fibers partition the 84 primitives, so they form
  the canonical **empty-core 7-petal** sunflower (the parallel classes).
- `fiber_ann_3petal_sunflower_free` — **inside any one fiber, no 3 distinct neighborhoods
  form a sunflower** (the neighborhood system is fiber-wise 3-sunflower-free; max 2 petals).
- `fiber_no_three_disjoint_ann` — inside any one fiber, no 3 neighborhoods are pairwise
  disjoint (≤ 2 disjoint per fiber).
- `ann_cross_fiber_disjoint` — neighborhoods in different fibers are always disjoint.
- `fourteen_petal_disjoint_witness` + `witness14_valid` / `witness14_card` — an explicit
  14-primitive family (2 per fiber) whose neighborhoods are pairwise disjoint.

**Synthesis (the maximum is exactly 14).** A sunflower of neighborhoods splits by core:

- *Non-empty core.* If two neighborhoods share a point they lie in the same fiber
  (`ann_lies_in_one_fiber`); fiber-wise the system is 3-sunflower-free
  (`fiber_ann_3petal_sunflower_free`), so a non-empty-core sunflower has ≤ 2 petals.
- *Empty core (pairwise disjoint).* Each neighborhood is intra-fiber, fibers are disjoint
  (`ann_cross_fiber_disjoint`), and within a fiber at most 2 are pairwise disjoint
  (`fiber_no_three_disjoint_ann`). Hence ≤ `7 × 2 = 14`. The witness attains 14.

So the maximum sunflower of the neighborhood system is **exactly 14, empty-core**. The
two halves of this argument are each `native_decide`; the `7 × 2` arithmetic synthesis is
the one prose step (flagged honestly; closing it inside Lean is the §5 milestone).

This is the cleanest "real result" the system supports at Level 2: the entire sunflower
profile is *forced by the verified algebra* — 4-regularity gives uniform 4-petal stars
(system A), intra-fiber annihilation bounds cross-fiber sunflowers (system B).

---

## 3. The Lean model (Task 3)

`formal/lean4/SounioSunflower.lean`, structures over the *existing* finite objects:

- `annOf u := applyOp .unlearn u` (ties petals to the verified surgical op).
- `inter`, `seteq`, `disjointL`, `nubP` — set ops on `List PrimSed`.
- `isSunflower3 x y z` — decidable 3-set sunflower predicate (equal pairwise intersections).
- `fiberOf L`, `labels7` — the 7 parallel classes.
- Theorems: `star_is_4petal_sunflower`, `ann_lies_in_one_fiber`,
  `fibers_pairwise_disjoint`, `fiber_ann_3petal_sunflower_free`,
  `fiber_no_three_disjoint_ann`, `ann_cross_fiber_disjoint`,
  `fourteen_petal_disjoint_witness`, `surgical_sunflower_dictionary`,
  `sunflower_structure_summary`.

All quantifiers are `List.all` over `validPrims` / `labels7` / `witness14`, so every
statement is `native_decide`. No `PrimSed` / `validPrims` / `applyOp` redefinition.

---

## 4. Three levels of success (Task 4)

- **Level 1 (conceptual note).** *Done.* The dictionary (§1) — core = point, petals =
  `UNLEARN`, parallel classes = fibers, ops = core/petal manipulators — plus the lift
  obstruction.
- **Level 2 (Lean witnesses / obstructions).** *Done.* 11 `native_decide` theorems: the
  4-petal star reading, the 7-petal fiber sunflower, fiber-wise 3-sunflower-freeness, and
  the 14-petal witness.
- **Level 3 (classical `f(n)` bound / counterexample).** *Open, and not claimed.* The
  object is a **fixed finite set system**; it cannot move the asymptotic constant in
  Erdős [20]. A genuine Level-3 step would require an *infinite, algebra-generated*
  family — e.g. the associator-generated families `((p · u) · v)` with `u · v = 0`, or a
  Cayley–Dickson tower beyond level 4 — that produces size-`n` sets with controlled
  intersection patterns. The §5 obstruction explains why the *linear* ZD structure alone
  stays finite.

**Lift obstruction (stated up front).** Sunflower-lemma improvements live in the
size-`n` → ∞ regime; the verified ZD geometry is a single 16-dimensional (level-4) slice
with 84 primitives. Climbing the Cayley–Dickson tower multiplies zero-divisors but is not
yet formalized, and the 168/ZD/Surgical theorems are level-4-specific
(`SounioImpossibilityChain`: level 4 is the *unique first* level with exact selective 4D
forgetting). So the finite shadow is exact and auditable; the asymptotic lift is the gap.

---

## 5. Cheapest next concrete step (Task 5)

**Runnable today, zero compiler work:**

```bash
cd formal/lean4 && lake build SounioSunflower      # green, ~6.6 s
python3 scripts/research/sunflower_168_validate.py # independent census mirror
```

The next research increment, still **zero** compiler/runtime work:

> Close the neighborhood-system maximum at `= 14` *inside Lean*. The two decidable halves
> already exist (`fiber_no_three_disjoint_ann` for ≤ 2/fiber, `ann_cross_fiber_disjoint`
> for cross-fiber, `fourteen_petal_disjoint_witness` for ≥ 14). The remaining step is a
> short structural lemma "any pairwise-disjoint sub-family of intra-fiber sets has size
> `≤ Σ_fiber (max-disjoint-in-fiber)`", which needs only `List` reasoning over the 7
> fibers — no `native_decide` blow-up, no new algebra. That upgrades the §2 synthesis from
> a prose step to a theorem and gives a fully-formal headline `zd_ann_sunflower_max = 14`.

Everything stays inside the verified 168 / ZD / Surgical-Calculus boundary.
