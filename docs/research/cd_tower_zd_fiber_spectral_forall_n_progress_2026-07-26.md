<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-spectral-forall-n-progress-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-spectral-forall-n-progress-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — ∀n progress on spectral completeness: strong evidence, structural theorems, and the exact wall

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ZD_FIBER_SPECTRAL_FORALL_N_STRONG_EVIDENCE_NOT_CLOSED`
**Parents:** `cd_tower_zd_fiber_spectral_classifier_2026-07-26.md`, `cd_tower_zd_fiber_signed_localization_2026-07-26.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_spectral_forall_n_progress_contract.py`

---

## 0. The result — an honest attack on the ∀n closure

Asked to attack the ∀n frontier, this rung records what a genuine two-flank assault found — and,
crucially, **where it stalls**. **No ∀n proof is claimed.**

> **Strong evidence, not a proof.** Spectral completeness (`#distinct A_σ spectra = 3·2^{n-5}`)
> holds for `n = 6, 7, 8, 9, 10` — **five** levels. Two clean structural theorems hold (doubling
> recursion; constant low rank). But the natural closure — a spectral doubling recursion — **does
> not close**: the low rank is *algebraically deep*, not a combinatorial blow-up, and the block
> cross-terms are irregular. The ∀n proof remains **OPEN**.

This is worth recording precisely: it strengthens the conjecture to `n≤10`, isolates two ∀n
structural facts, and **maps the exact wall** so the naive path is not re-chased.

---

## 1. Results

| Clause | Result | Status |
|---|---|---|
| `V1_EMPIRICAL` | `#distinct A_σ spectra = 6,12,24,48,96 = 3·2^{n-5}` for `n=6..10` | completeness holds, **5 levels** (extends `n≤8`); still not ∀n. |
| `V2_DOUBLING` | `A_σ(n)` restricted to lower-half lo-labels `[1,2^{n-2})` **= `A_σ(n-1)`** exactly (top-left block) | ∀n structural containment; proof reduces to the Lean-proven seam-flip law. |
| `V3_LOW_RANK` | `rank(A_σ) = 2^{n-2}-1` for **every** fiber (`n=6,7,8`) | constant nullity `2^{n-2}`. |
| `V4_BOUNDARY` | `A_σ` has **no twin vertices** (all signed rows distinct) and a **dense** null space | the low rank is **algebraically deep**, not a combinatorial blow-up ⇒ the naive block spectral recursion does **not** close. |

Verdict: `CD_TOWER_ZDFAN_VERDICT ZD_FIBER_SPECTRAL_FORALL_N_STRONG_EVIDENCE_NOT_CLOSED`.

---

## 2. The two flanks (both attacked, honestly)

- **Flank A — spectral doubling recursion (the closure).** `A_σ(n) = [[A_σ(n-1), Y],[Yᵀ, Z]]` with
  the top-left block exactly `A_σ(n-1)` (`V2`). But `Y, Z` are **not** sign-switches of `A_σ(n-1)`
  (`Z_sub ≠ ±X`, different support, `Y` irregular), and `A_σ` has **no twins** and a dense null
  space (`V4`). So the block spectrum is not simply determined, and the low rank has no simple
  combinatorial factorisation. **Stalled.** The remaining route is the explicit algebraic factorisation
  `A_σ = Cᵀ S C` (Walsh / character-sum type, in the spirit of the ∀n kernel-dimension proof) — not
  found in-session.
  > **Superseded 2026-07-31 — the factorisation was found.** It is not a Walsh character sum but a
  > rank-2 folding: `A_σ(l ⊕ L_lo, y) = −A_σ(l, y)`, giving `A_σ = Jᵀ M J` with `J Jᵀ = 2I`, hence
  > `rank(A_σ) ≤ 2^{n-2}−1` **derived ∀n** and an exact spectral halving. See
  > `cd_tower_zd_fiber_antisymmetry_lemma_spec_2026-07-31.md` (on the live ZD-fiber lane branch
  > `research/zd-fiber-antisymmetry-lemma-20260731`, byte-identical to PR #1580's copy; lands with
  > that lane — linked here without a hyperlink until it reaches main).
  > This closes the low-rank half only — `V1` (`#spectra = 3·2^{n-5}`) remains **OPEN**. The harness
  > and verdict token of *this* rung are deliberately left untouched: they are measured objects in the
  > R13/R14 kill-set corpora.
- **Flank B — empirical boundary.** Pushed completeness to `n=9` (48) and `n=10` (96), both matching
  `3·2^{n-5}` (vectorised via a precomputed sign table). Five levels now stand.

---

## 3. What this is / is NOT

- **Is:** honest ∀n *progress* — strong evidence (`n≤10`), two structural theorems, and a precise
  map of the wall.
- **Not** an ∀n proof of completeness or Fano injectivity — **OPEN**. The naive spectral recursion is
  shown *not* to close (`V4`); the closure needs the algebraic low-rank factorisation.
- **Not** symbolic beyond numerical eigenvalue/rank computation; **not** `D3`; **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/cd_tower_zd_fiber_spectral_forall_n_progress_contract.py
# expect: V1 (n=6..10) OK, V2/V3/V4 OK, VERDICT ...STRONG_EVIDENCE_NOT_CLOSED  (~60s)
```

---

## 5. AI disclosure

Probe and contract produced under human direction (2026-07-26), on an explicit instruction to attack
the ∀n frontier ("ataca caralho") from two flanks. **Honest outcome: strong evidence (`n≤10`) + two
structural theorems, but the ∀n proof is NOT closed** — the naive spectral doubling recursion is shown
not to close (no twins, dense null space, irregular block cross-terms), and the algebraic low-rank
factorisation that would close it was not found in-session. No fabricated proof (the session's
discipline: real progress + honest boundary, not a claimed solution). The substantive completeness
claim (`n≤8`) was Grok-reviewed in the parent classifier rung; this rung is verified-empirical progress
plus an explicit open boundary. Numerical certificate; ∀n OPEN. No semantic claim, no clinical content.
GAIDeT-ICMJE 2025.

**Post-session correction (2026-08-18, landed from the PR #1580 audit):** "not found in-session" was
true of the 2026-07-26 session and stayed in this disclosure untouched as a measured record — but the
low-rank factorisation WAS found on 2026-07-31 (rank-2 folding; see the supersession note in §2).
`V1` remains OPEN; ∀n completeness is still not closed.
