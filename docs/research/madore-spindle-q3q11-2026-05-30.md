# Madore's Moser spindle over ℚ(√3,√11) — parametric multiquadratic framework (2026-05-30)

Mathlib-free Lean 4. Two new files, both `lake build`-green, no `sorry`/`sorryAx`.

## What is proved

### `formal/lean4/SounioMoserSpindleQ311.lean` — the headline (Madore's χ ≥ 4 line)

The Moser spindle is realised with **exact** coordinates in ℚ(√3,√11) (integer 4-tuples
`a + b√3 + c√11 + d√33`, scaled ×12), and:

- `edges_unit` — each of the **11 unit edges** has squared distance exactly the rational `144`
  (all √3/√11/√33 components cancel), by `decide` on integer arithmetic. Axioms `{propext}`.
- `spindle_not_3_colourable` — **not 3-colourable**: the 11 edge-disequalities over `Fin 3` are
  refuted by `omega` (no `native_decide`). `spindle_4_colourable` gives the witness ⇒ **χ = 4**.
- `q311_plane_needs_4_colours` — via the generic `UnitDistanceChromatic` reduction:
  **χ(ℚ(√3,√11)²) ≥ 4**, the base case of Madore (arXiv:1509.07023). Axioms `{propext, Quot.sound}`.

χ(ℝ²) ≥ 4 is the intended corollary (each `Qf` point is a genuine real via the framework's
`sqrtR`); the full `Real × Real` instance needs the ring homomorphism `φ : Qf → Real`
(the `{3,11}` analogue of `E_sq4`) and is **not yet discharged** — noted explicitly, no overclaim.

### `formal/lean4/SounioMultiquadParam.lean` — parametric framework over `primes : List Nat`

- `sqrtR : Nat → Real` (generic Newton root) + `sqrtR_sq` (`(√p)² = p`, `p ≥ 1`).
- `radS`/`evalS` over `2^|S|` masks; `HasRadicals`/`IndepMultiquad`; base case `indep_base`.
- `sqrt_new_not_in_Q` — generic ℚ-irrationality core (any non-square `m` ⇒ √m ∉ ℚ).
- `Np_ne_zero` — generic degree-2 **domain core**: `a² − p·b² ≠ 0` for non-square `p`,
  `(a,b) ≠ 0` (generalises `N5_ne_zero` to arbitrary `p`; kernel of the conjugate inverse).
- `indep_3_11` (degree 4, the spindle basis `{1,√3,√11,√33}`) and `indep_3_5_11` (degree 8) —
  proved by bridging to the existing `indep8`. The latter re-grounds the **χ ≥ 5 over ℚ(√3,√5,√11)**
  line as the `S = [3,5,11]` instance of the framework.

## Open obligation (honest scope)

`MultiquadIndepTheorem : ∀ S, HasRadicals S → IndepMultiquad S` is stated as an explicit
obligation — *not* an axiom, *not* `sorry`. It is discharged for the **instantiated supports**
(`[]`, `[3,11]`, `[3,5,11]`). The unrestricted `∀S` proof (degree `2^|S|`) needs, Mathlib-free:
(i) a squarefree-by-prime-factorisation lemma feeding `no_rat_sqrt` for arbitrary radicands
`p·∏T`, and (ii) the recursive norm/conjugate inverse tower lifting `Np_ne_zero`/`Q35_inv` to all
levels. These are recorded as the remaining research work; `Np_ne_zero` is the proven generic
|S|=1 kernel.

## Review

`bin/llm-offload -t math-review -p xai` (Grok 4.1) on both files: **all proved statements [OK]**,
the open obligation correctly identified, and "no overclaim" on the ℝ² reading. No bug caught.
See `.claude/llm_offload_log.md` (2026-05-30 entry).
