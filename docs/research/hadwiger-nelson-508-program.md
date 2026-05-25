# Hadwiger–Nelson (#508): the discovery-engine program toward χ ≥ 6

**Goal (locked, honest framing).** Search for a finite **unit-distance graph in ℝ²
with chromatic number 6**, which would prove `χ(plane) ≥ 6` and narrow the standing gap
`5 ≤ χ ≤ 7`. This is the *highest-payoff, longest-odds* target. We pursue it as a
**finite construction with a verified certificate** — the only terms on which AI/compute
has ever cracked an open extremal problem — not as a conjecture-slaying claim.

This note states the architecture and records the **first stone**, already landed.

---

## 0. Why this is the right shape for the Sounio stack

The modern precedent for AI producing *new* results on extremal-combinatorics /
incidence problems is real and specific:

- **FunSearch** (DeepMind, *Nature* 2023) — LLM-in-a-loop + evaluator found a **new
  largest cap set** (an Erdős-family problem) and beat an online bin-packing bound.
- **AlphaTensor** (*Nature* 2022) — RL beat Strassen in specific matrix-mult cases.
- **Davies et al.** (*Nature* 2021) — ML guided a **proven** knot-theory theorem.

Every one is a **finite witness / construction / lower bound**, certified after the
fact — never the resolution of an infinitary upper-bound conjecture. #508 fits: a
6-chromatic graph is a finite object; "χ ≥ 6" is witnessed by it.

The split that organizes the whole program:

| Half | Role | Sounio tool |
|------|------|-------------|
| **Discovery** | propose finite graphs | GPU-parallel search + ML/FunSearch-style loop, seeded by algebraically-structured vertex families (rotations / number-ring lattices) |
| **Certifier** | turn a graph into proof | exact (float-free) unit-distance test + **verified non-k-colorability (UNSAT) certificate**, machine-code-reproducible |

The certifier is where full-stack control is a genuine edge: claimed unit-distance
graphs have been **invalidated by floating-point error**; a bit-exact, kernel-checked
certificate is not refutable on those grounds.

---

## 1. First stone (LANDED): the exact certifier on the χ = 4 landmark

`formal/lean4/SounioUnitDistanceExact.lean` — green (`lake build`, ~0.4 s, no `sorry`,
no Mathlib, all `native_decide`). Mirror: `scripts/research/unit_distance_exact_validate.py`.

We work in the real number field **ℚ(√3, √11)**, represented exactly as integer 4-tuples
`(a₀ + a₁√3 + a₂√11 + a₃√33)` (coordinates scaled ×12 to clear denominators; a unit edge
has exact squared length `144`). On the **Moser spindle** (7 vertices):

- `unit_pairs_are_the_11_edges` — the exact unit-distance graph is *exactly* the 11
  claimed edges (no spurious / missing unit pair; the irrational non-edges like
  `|BF|² = 7/6 − √33/6` are provably ≠ 1).
- `spindle_edge_count` = 11.
- `spindle_chromatic_eq_4` — χ = 4 by brute force over all `4⁷` colorings.
- `moser_spindle_certified` — all three.

**What this establishes:** the certifier (exact arithmetic + exact χ) is correct on the
canonical graph that *requires* irrational-exact arithmetic. This is the trustworthy base
every later claim must pass through. **It is not progress on #508 by itself** (χ ≥ 5
already needs ~500-vertex graphs).

---

## 2. Phase plan (honest about each gate)

**Phase A — exact certifier (DONE, §1).** ℚ(√3,√11), spindle χ=4.

**Phase B — scale the field + reproduce χ ≥ 5.** Extend the exact ring to the algebraic
numbers de Grey's vertices live in (rotations generating ℚ(√3, √11, …)); ingest a known
~500-vertex 5-chromatic graph (Heule's reduced graphs are public) and re-certify it
exactly. Gate: reproduce a *published* χ ≥ 5 with bit-exact arithmetic. No new math —
this validates scale.

**Phase C — verified UNSAT certificate.** χ ≥ 6 means "not 5-colorable" = an UNSAT
statement over a large CNF. `native_decide` does not scale here; the deliverable is a
**checked DRAT/LRAT-style certificate** of non-5-colorability, verified through the
Sounio stack. This is the certifier's real test and its differentiator. Gate: a
machine-checked non-5-colorability proof for a known 5-chromatic graph.

**Phase D — discovery engine.** GPU-parallel construction + ML/FunSearch-style search
over algebraically-structured vertex families (the hypercomplex/rotation structure as a
*search prior*, not a theorem), maximizing the SAT-solver's lower bound on χ. Gate: a
graph the solver cannot 5-color. **This is the long shot and may fail — χ could be 5.**

**Phase E — if Phase D finds a candidate:** certify it exactly (A) + certify
non-5-colorability (C). That, and only that, is the Level-3 result.

---

## 3. Honest odds and what counts

- Phases A–C are **engineering with defined gates** — achievable, and valuable
  infrastructure regardless (a reusable verified unit-distance certifier).
- Phase D is a **genuine research gamble**. Polymath16 pushed hard at the 5→6 frontier
  without reaching it; χ = 6 may be false. The honest success ladder:
  - *Infra win:* a verified, reproducible χ-certifier + UNSAT pipeline (A–C). Real,
    publishable as method.
  - *Level-3 win:* a machine-certified 6-chromatic unit-distance graph. Major, unlikely
    in one pass, and the only thing that would actually move #508.
- We will **not** label A–C as solving #508, and we will state at every step which gate
  is engineering and which is the gamble.

## Reproduce (Phase A)

```bash
cd formal/lean4 && lake build SounioUnitDistanceExact        # green, ~0.4 s
python3 scripts/research/unit_distance_exact_validate.py     # exact spindle, χ=4
```
