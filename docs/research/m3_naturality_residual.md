<!-- docs:meta
topic_id: repo.docs.research.m3-naturality-residual
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.m3-naturality-residual
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# M3 Naturality Residual — Distributional, Not Pointwise

**Status:** residual finding, 2026-05-01
**Plan reference:** `project_octonion_homology_functor.md` M3 (wk 12-14)
**Preprint reference:** `docs/research/ocssm_preprint_skeleton.md` §4 (R1-R7), §6.2 (F2)
**Predecessor commits:** `c3a1bf9e` (compositional decomposition), `3f9e7c56` (runtime table dump)

## Finding

The decide-vacuity of pointwise naturality on the 1024 Halton-Box-Muller rows is not a gap to close at the embed layer. It is a category error: pointwise naturality is the wrong formal object for the embed M3 specifies. The correct object is distributional naturality — discrepancy of the image measure under the discrete G₂ action — and that is what `embed_into` was constructed to support.

## Why path 1 (extending the action group) fails

The natural attempt to make naturality pointwise-decidable is to enlarge the action group beyond `Aut(Fano) ≅ PSL(2,7)` so that conjugation lives inside it. The smallest such extension is

    OctSym := Aut(Fano) ⋊ ⟨conj⟩  ≅  PSL(2,7) × Z/2,  |OctSym| = 336.

Replace `embed(h) := embed_Halton(h)` with the orbit construction

    embed(h) := actOctSym(g_{h mod 336}, v_seed)

for some fixed seed `v_seed` and an enumeration of the 336 group elements. Then for each `h` there is a hash relation `conj_hash(h)` such that `embed(conj_hash(h)) = conj(embed(h))` *exactly*, by composition in the group. The naturality identity decides over the 336² = 112 896 cases of the Cayley table by `native_decide`, no axiom, no `sorry`.

This is mechanically correct and constructible. It does not solve M3, because:

1. **The orbit embed forecloses content-awareness.** The Halton embed is a function `hash → S⁷` with no algebraic structure on the hash side, so the *content* of an utterance (mapped to its hash) is independent of the orbit position. A future content-aware refinement (learned codebook, semantic hash, ϕ-embedding) can place semantically-related utterances at orbit-related sphere points. The orbit embed cannot: every `embed(h)` is determined by `h mod 336` alone, so 1 024 utterances all sharing `h ≡ k (mod 336)` collapse to the same point. F2 on a real corpus would require the corpus's hash-mapping to coincide with the algebraic relation `conj_hash`, which it does not for any independent hash function on real text.

2. **F2 d_paired becomes structurally zero on synthetic corpora and structurally maximal on real ones.** For the orbit embed, the synthetic corpus (which the loader may construct to satisfy `h_b = conj_hash(h_a)`) gives `d_paired = 0` exactly. For a real corpus, where `h_b` is determined by utterance B's content independent of A's, `d_paired` saturates to the full random baseline and the F2 ratio is ≈ 1. The statistic loses discriminative power on the data we actually want to test.

3. **The trade is irreversible inside the embed.** Pointwise naturality and content-awareness are incompatible structures on the same function `hash → S⁷`. Choosing an algebraic embed today commits us to never testing F2 on real dialogue.

The Halton embed has the opposite tradeoff: pointwise naturality is decide-vacuous, but distributional naturality (R7 / G₂-uniformity by construction) holds, and a future content-aware refinement is still admissible because the embed has no algebraic obstruction.

## Why path 2 (weakening R4) is also unavailable

An alternative is to redefine F2 so the speaker-reversal correspondence is to *some* Fano involution σ ∈ Aut(Fano) of cycle type (2,2,1,1,1), rather than to octonion conjugation. The 336-element extension is then unnecessary; naturality decides over the original 168.

This severs R4. R4 ties (ii) speaker-direction to (iii) zero-divisors via the anti-homomorphism `\overline{ab} = \bar{b}\bar{a}` — the algebraic content of "swapping speakers swaps multiplication order." Any σ ∈ Aut(Fano) acting on basis indices is a *homomorphism* of 𝕆, not an anti-homomorphism. The (ii)↔(iii) bridge collapses; the three-correspondence thesis loses its load-bearing link. This is a paper-level edit, not a code edit, and is recorded here for completeness but not pursued.

## What M3 actually claims

The decomposition theorems landed in `c3a1bf9e` are the correct M3 deliverable. They establish, by `native_decide` over an `OctSh` shadow:

* `actG2 σ` preserves `normSq` for all σ ∈ Perm7.
* `actG2 σ` permutes the 8-element basis of `OctSh`.
* `actG2 σ` commutes with scalar multiplication.

These are *true* by construction on all of `OctSh`, not just on the Halton image. They are the structural lemmas the homology functor F needs to factor through `C_𝕆`. The pointwise-naturality goal that motivated path 1 was a stronger statement than M3 requires.

R7 / G₂-uniformity in §4 of the preprint is explicitly *distributional*: "isotropic-on-ℝ⁸ + radial projection = uniform on S⁷." Halton-on-cube → Box-Muller → unit-normalisation factors through an isotropic Gaussian, so the image measure is uniform on S⁷, and any O(8) action preserves uniform measure. The discrete G₂ subgroup is ⊂ O(8), so it preserves the uniform measure on S⁷ — distributionally, not pointwise. The Lean obligation for R7 is therefore a *bounded-discrepancy* claim, not a `decide` over the 1 024-row table.

The form of that claim:

    ∃ ε > 0, ∀ σ ∈ Perm7,
      |#{i : embed_row(i) ∈ σ·B} − #{i : embed_row(i) ∈ B}| ≤ ε · 1024
      for all measurable B ⊂ S⁷.

In Lean: a finite `decide` over the 14 generators × 1 024 rows × a fixed test-box family, bounding the supremum discrepancy. This is a different theorem than `embed(σ·h) = actG2(σ)(embed(h))` — and it is the theorem the preprint actually states.

## Recommended close-out for M3

1. **Keep `c3a1bf9e` and `3f9e7c56`** as the M3 deliverable. They establish what the homology functor needs.
2. **Document that pointwise-naturality is `decide`-vacuous by design** — not a residual gap. The correct naturality statement for the Halton embed is distributional.
3. **Reframe the M3 Lean obligation** as a bounded-discrepancy theorem on the 1024-row table, deferred to M3b alongside content-aware embed work.
4. **Land M3b as the load-bearing scientific work**: real SWDA ingestion + content-aware embed `e: utterance → S⁷` that admits a learnable refinement. F2 on this embed is the actual test of the homology claim.

## Artifacts retained

* `stdlib/algebra/fano_auts.sio` — 168×7 lex-sorted PSL(2,7) table, mirror of `formal/FanoLabellingOrbits.lean::fanoAuts`. Emitted during the path-1 design exploration; left in place as reusable algebra data with no current callers.
* `formal/lean4/SounioNaturalityG2Decomp.lean`, `SounioNaturalityG2Runtime.lean` — the decomposition + runtime-table artefacts. Stand as M3.

## What is *not* claimed by this note

* That the F2 hypothesis (`embed(B) ≈ conj(embed(A))` on real σ-pairs) is false. It is untested on real data. M3b is where it becomes testable.
* That the current synthetic F2 PASS in `tools/ocssm/swda_loader.py` is evidence for or against the homology claim. It is a unit test of the F2 measurement code against a constructed fixture, nothing more.
* That R4 should be weakened. R4 stands as written; the path-2 weakening was considered and rejected on §4 grounds.
