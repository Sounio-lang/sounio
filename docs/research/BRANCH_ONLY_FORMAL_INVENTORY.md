<!-- docs:meta
topic_id: repo.docs.research.branch-only-formal-inventory
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.branch-only-formal-inventory
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Branch-only formal (Lean) developments: inventory

**Phase 3.3, inventory part only. Written 2026-10-06.** None of the files below was imported into `main`. Before any of them is imported it has to be built with the pinned toolchain (`formal/lean4/lean-toolchain`). That build was not done here. All counts come from reading the source text. Nothing was built.

Every branch named here is also preserved as tag `archive/2026-10-05/<branch>`, so the SHAs stay reachable if the branch is deleted.

## How the counts were taken

For each file, read at the branch tip with `git show <branch>:<path>`:

- **`sorry`** counts the token `sorry` in code, after stripping `--` line comments, nested `/- … -/` block and doc comments, and string literals.
- **`axiom`** counts declarations of the form `axiom …` in code (optionally `private`/`protected`/`noncomputable` or attribute-prefixed), after the same stripping.
- **`native_decide`** counts the token in code. It is listed because a `native_decide` proof trusts the compiler (`Lean.ofReduceBool`), not only the kernel.

A raw `grep` over the full text finds more `sorry`/`axiom` hits than the counts below. Every extra hit is inside a comment, for example "no `sorry`, no `native_decide`" or "The obligation is a hypothesis, not an axiom". These counts are **not** a substitute for `#print axioms`, which needs a build.

## 1. `wip/zd-two-mode-lean-20260921`: the 30 `formal/lean4/SounioZD*.lean` files

- Branch tip: `04853c6536d888139125f17440268a94a372175f` (2026-09-21, "[formal] WIP: ZD two-mode Lean files and research notes"). That is the only branch-only commit, and it is the last commit to touch every file below.
- On that branch, `formal/lean4/SounioZD*.lean` matches 34 files. Four of them (`SounioZDChi`, `SounioZDFiberAntisym`, `SounioZDTwoMode`, `SounioZDTwoModeBridge`) are already on `main`, byte-identical. The **30 below exist only on the branch.**
- Their only import outside the set is `SounioCDCocycle`, and it is on `main`. The 30 files form one import chain rooted at `SounioZDAlgebraBridge` → `SounioCDCocycle`.

| File | Lines | `sorry` | `axiom` decls | `native_decide` | Imports |
|---|---:|---:|---:|---:|---|
| `formal/lean4/SounioZDAlgebraBridge.lean` | 342 | 0 | 0 | 0 | SounioCDCocycle |
| `formal/lean4/SounioZDCounting.lean` | 976 | 0 | 0 | 0 | SounioZDSignedCongruence SounioZDUnsignedCode |
| `formal/lean4/SounioZDMatrixAssembly.lean` | 746 | 0 | 0 | 0 | SounioZDRecursion |
| `formal/lean4/SounioZDMinimalInvariant.lean` | 141 | 0 | 0 | 0 | SounioZDSignedState |
| `formal/lean4/SounioZDRecursion.lean` | 436 | 0 | 0 | 0 | SounioZDAlgebraBridge |
| `formal/lean4/SounioZDScalarBalanced.lean` | 405 | 0 | 0 | 0 | SounioZDScalarBoundaryOne |
| `formal/lean4/SounioZDScalarBit.lean` | 149 | 0 | 0 | 0 | SounioZDScalarCore |
| `formal/lean4/SounioZDScalarBoundaryOne.lean` | 543 | 0 | 0 | 0 | SounioZDScalarBoundaryTwo |
| `formal/lean4/SounioZDScalarBoundaryTwo.lean` | 491 | 0 | 0 | 0 | SounioZDScalarHalfSeparator |
| `formal/lean4/SounioZDScalarCore.lean` | 420 | 0 | 0 | 0 | SounioZDScalarInvariant |
| `formal/lean4/SounioZDScalarCrossOrigin.lean` | 352 | 0 | 0 | 0 | SounioZDScalarBit |
| `formal/lean4/SounioZDScalarDiscriminant.lean` | 383 | 0 | 0 | 0 | SounioZDScalarInnerBound |
| `formal/lean4/SounioZDScalarDivisibility.lean` | 235 | 0 | 0 | 0 | SounioZDScalarCrossOrigin |
| `formal/lean4/SounioZDScalarGap.lean` | 226 | 0 | 0 | 0 | SounioZDScalarBalanced |
| `formal/lean4/SounioZDScalarHalfSeparator.lean` | 476 | 0 | 0 | 0 | SounioZDScalarNarrowSeparator |
| `formal/lean4/SounioZDScalarInnerBound.lean` | 412 | 0 | 0 | 0 | SounioZDScalarTwoBlock |
| `formal/lean4/SounioZDScalarInvariant.lean` | 205 | 0 | 0 | 0 | SounioZDSignedState |
| `formal/lean4/SounioZDScalarLowCone.lean` | 305 | 0 | 0 | 0 | SounioZDScalarDivisibility |
| `formal/lean4/SounioZDScalarNarrowSeparator.lean` | 594 | 0 | 0 | 0 | SounioZDScalarWideSeparator |
| `formal/lean4/SounioZDScalarNativeBoundary.lean` | 267 | 0 | 0 | 0 | SounioZDScalarThirdBoundary |
| `formal/lean4/SounioZDScalarResidualSearch.lean` | 174 | 0 | 0 | 0 | SounioZDScalarNativeBoundary |
| `formal/lean4/SounioZDScalarThird.lean` | 709 | 0 | 0 | 0 | SounioZDScalarGap |
| `formal/lean4/SounioZDScalarThirdBoundary.lean` | 669 | 0 | 0 | 0 | SounioZDScalarThird |
| `formal/lean4/SounioZDScalarTwoBlock.lean` | 420 | 0 | 0 | 0 | SounioZDScalarLowCone |
| `formal/lean4/SounioZDScalarWideSeparator.lean` | 317 | 0 | 0 | 0 | SounioZDScalarDiscriminant |
| `formal/lean4/SounioZDSignedCongruence.lean` | 497 | 0 | 0 | 0 | SounioZDTwinCover |
| `formal/lean4/SounioZDSignedState.lean` | 757 | 0 | 0 | 0 | SounioZDSupportWord |
| `formal/lean4/SounioZDSupportWord.lean` | 470 | 0 | 0 | 0 | SounioZDCounting |
| `formal/lean4/SounioZDTwinCover.lean` | 360 | 0 | 0 | 0 | SounioZDMatrixAssembly |
| `formal/lean4/SounioZDUnsignedCode.lean` | 221 | 0 | 0 | 0 | (none) |
| **Total (30 files)** | **12,698** | **0** | **0** | **0** | |

Summary: 30 files, about 12.7k lines. No `sorry`, no `axiom` declaration and no `native_decide` appears in code. This is unverified until it is built with the pinned toolchain and checked with `#print axioms`.

## 2. `lane/fable-1/p0f-ffi-takeover`: §K Hurwitz and the Conjecture 6.8 record

Branch tip: `617f4bbee353e0657654bcb4c6be9ea436a143b3` (2026-09-01).

### 2.1 "§K Hurwitz in the kernel: norm multiplicativity for CD(n≤3) by double polarization"

- Commit: `038a95d09fda3f5126624dec38805188b6cc997f` (2026-09-01). It adds 233 lines to `formal/lean4/EpistemicEffectsNSA.lean`. The commit message describes 24 load-bearing theorems, "sorry-free, no native_decide, axioms ⊆ {propext, Quot.sound, Classical.choice}". It also extends the gate `scripts/ci/antigarbling_fusion_lean_gate.sh`.
- **This is not branch-only any more.** `main`'s `formal/lean4/EpistemicEffectsNSA.lean` (last touched by `c52b23e63108fbea4f17761be4b76ad94bd2fec4`, 2026-09-01, which adds §L) is a strict superset of the branch copy. Going from the branch to `main` the diff has 0 deleted lines and 343 added. §K sits at line 1404 on `main`. All §K theorem names are present on `main`: `lin_zero_of_basis`, `polar_zero_of_polarBasis`, `basis_bil_zero`, `bil_zero_of_polarBasis`, `norm_mult_of_polarBasis`, `octonion_norm_multiplicative`, `quaternion_norm_multiplicative`, `sedenion_norm_not_multiplicative`, `not_polarBasis4` and `shortcut_eq_sensitivity_of_polarBasis`. The other files that commit touched are `ANTIGARBLING_FUSION_THEOREM_2026-09-01.md` (on `main`) and the four `conj68_*` docs plus the Zhilina ref (branch-only, see 2.2). No action is needed for §K itself.

| File (branch copy) | SHA | Lines | `sorry` | `axiom` decls | `native_decide` | On `main`? |
|---|---|---:|---:|---:|---:|---|
| `formal/lean4/EpistemicEffectsNSA.lean` | `038a95d09fda3f5126624dec38805188b6cc997f` | 1647 | 0 | 0 | 0 | yes; `main` (2027 lines) is a superset |

### 2.2 Lean files on this branch that are still branch-only

| File | SHA (last commit touching it) | Date | Lines | `sorry` | `axiom` decls | `native_decide` |
|---|---|---|---:|---:|---:|---:|
| `formal/lean4/SounioConj68RankBound.lean` | `1e98fa97f78fb80d1933d4b70a5112a89fdcd1be` | 2026-08-31 | 243 | 0 | 0 | 8 |
| `formal/lean4/SounioConj68EulerLeg.lean` | `010993089ea09ca66fb70376418c3ffffcfb2e18` | 2026-08-31 | 170 | 0 | 0 | 4 |
| `formal/lean4/SounioZDCollapse.lean` | `3c002d8eb525b682a7f2a333c68ce349871fcb37` | 2026-08-02 | 501 | 0 | 0 | 0 |
| `formal/lean4/SounioZDE5Inductive.lean` | `f8c158ebb06ed99b035fedf909e701f776c14d52` | 2026-08-14 | 157 | 0 | 0 | 13 |
| `docs/research/lean/SounioBlackwellBridge.lean` | `c8bbc963ddbec0f4c3f3d54fcdbd156641e04c3a` | 2026-08-23 | 196 | 0 | 0 | 0 |
| `docs/research/lean/SounioOctonionFidelity.lean` | `c8bbc963ddbec0f4c3f3d54fcdbd156641e04c3a` | 2026-08-23 | 135 | 0 | 0 | 7 |
| `docs/research/lean/SounioTripleChannel.lean` | `4255ac408505a7614d0ce1afb3a785f7273a7a87` | 2026-08-23 | 166 | 0 | 0 | 2 |
| `docs/research/lean/SounioWarrantHolonomy.lean` | `bd3398802d6705754ad9b62bf507ea3657818d32` | 2026-08-22 | 99 | 0 | 0 | 0 |

`SounioZDCollapse` imports `SounioZDFiberAntisym`, which is on `main`. None of these files is listed in the branch's `formal/lean4/lakefile.lean`.

### 2.3 Conjecture 6.8: where it is mentioned, and the prior-art record next to each mention

**Prior-art record (verbatim).** It comes from commit `5d766010315aff5391b0c86f4d803e29e9fa5d25` (2026-09-01), "research(conj68): RESOLVED — Conjecture 6.8 is Zhilina Theorem 4.13 (arXiv 2608.26890)". The note was prepended to both conj68 research docs:

> RESOLVED 2026-09-01: Conjecture 6.8 is a THEOREM — Zhilina, arXiv 2608.26890, Theorem 4.13 (diam Γ_C^Z(𝕊)=3), a companion paper to 2608.26903 that we had cited but never fetched. Proof = dimension count: find (a',b')∈Im C(x')∩span{(a,-b),(b,a),(ab,0),(0,ab)}^⊥ (5∩codim4≥1), then Lemma 4.12 gives d(x,(a',b'))≤2; length-3 path. VERIFIED computationally 8/8. See refs/zhilina_diameter_commutativity_2608.26890_THEOREM.md. Our §§2-5 rank laws stand independently.

The full record is `docs/research/refs/zhilina_diameter_commutativity_2608.26890_THEOREM.md` on that branch (same commit). Its title is "Conjecture 6.8 IS A THEOREM — Zhilina, arXiv 2608.26890 (2026-08-27)". It records the chronology: arXiv 2608.26903 (Guterman–Zhilina, "Relation graphs of the sedenion algebra") states the conjecture, and the companion arXiv 2608.26890 (Zhilina, "Diameter of the commutativity graph of the real sedenions", same 27 Aug 2026 batch) proves it as Theorem 4.13.

**`main` mentions Conjecture 6.8 nowhere** (`git grep "Conjecture 6.8" origin/main` is empty on 2026-10-06). Every mention is on this branch. Each one is quoted below with the prior-art status beside it:

| Where it appears (branch `lane/fable-1/p0f-ffi-takeover`) | Quoted mention | Prior-art status |
|---|---|---|
| `formal/lean4/SounioConj68RankBound.lean`, lines 2–4 (header) | "Lean leg of the rank-bound half of the attack on Conjecture 6.8 (Guterman–Zhilina, arXiv:2608.26903): the commutativity graph of the sedenions restricted to zero-divisor imaginary parts has diameter 3." | **Already proved: Zhilina, arXiv 2608.26890, Theorem 4.13.** The file does not record this. Its rank laws stand on their own and are not a proof of the conjecture. |
| `formal/lean4/SounioConj68RankBound.lean`, line 235 | "…every configuration admits a witness — the last lemma of Conjecture 6.8, modulo the degeneracy-loci refinement." | Same. The diameter result is Zhilina Thm 4.13. This file does not record it. |
| `formal/lean4/SounioConj68EulerLeg.lean`, line 2 (header) | "route-alpha legs of the attack on Conjecture 6.8 (Guterman–Zhilina, arXiv:2608.26903)" | Same. The file's own header also carries a "RODADA 10 CORRECTION": the P3 relation is vacuous and the Euler obstruction argument "as stated below COLLAPSES". |
| `formal/lean4/SounioConj68EulerLeg.lean`, line 31 | "…which is the last lemma of Conjecture 6.8 modulo the degeneracy-loci refinement." | Same. |
| `docs/research/conj68_manuscript_draft.md`, line 31 | "conjecture (their Conjecture 6.8, supported by floating-point experiments): **Conjecture (GZ).** diam Γ_C^Z(𝕊) = 3." | The RESOLVED note above is at line 16 of the same file. |
| `docs/research/conj68_proof_strategy.md`, line 18 (title) | "# Conjecture 6.8 (diam Γ_C^Z(𝕊) = 3): proof strategy and obstruction map" | The RESOLVED note is at line 16, directly above the title. Lines 286 ("Honest status of Conjecture 6.8 … evidence, not proof") and 364 ("Conjecture 6.8 … is an **open target** in this repo") predate the resolution. |
| `docs/research/conj68_attack_log_2026-08-31.md` (6 mentions), `docs/research/conj68_commutator_kernel_machinery.md` (3 mentions) | attack-log and machinery notes, 2026-08-31 | These files have **no** RESOLVED note. Read them together with the record above. |

If any of these files is later imported into `main`, the prior-art line has to travel with it. That means the two `SounioConj68*.lean` headers, and the attack log and machinery notes too, since none of them carries it today.

## 3. What this inventory does not establish

- That any file builds. No `lake build` was run, and mathlib was not fetched.
- The axioms any theorem actually depends on. That needs `#print axioms` after a build.
- That the 30 ZD files, or the files in 2.2, are correct or worth importing. That decision belongs to Phase 3.3 proper.
