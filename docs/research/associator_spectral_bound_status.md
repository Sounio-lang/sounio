<!-- docs:meta
topic_id: repo.docs.research.associator-spectral-bound-status
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.associator-spectral-bound-status
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Associator Spectral Bound — Status Report

**Date**: 2026-04-13
**Thread**: δ. Scope: establish the current status of the associator spectral bound conjecture cited in `memory/project_oct_connectomics.md`.

## The conjecture

From `/workspace/.home/openvscode-server/.claude/projects/-workspace-sounio/memory/project_oct_connectomics.md` (research program commit 2026-03-20):

> Key conjecture: associator spectral bound (E[||[a,b,c]||] ≤ C*(1-λ₂/λ_max))

where λ₂ is the Fiedler value (second-smallest Laplacian eigenvalue) of the underlying graph, λ_max is the spectral radius of the Laplacian, and the expectation is (presumably) over random triples of edges/nodes with octonion labels.

Intuition: *well-connected* graphs have `λ₂/λ_max → 1`, making the RHS small — such graphs cannot support large expected associators. Poorly-connected graphs have `λ₂/λ_max → 0`, making the RHS large — they CAN support structural non-associativity. The bound would link global graph geometry to local algebraic non-associativity.

## What actually exists in the codebase

Catalogued 2026-04-13:

| Result | Status | Location |
|--------|--------|----------|
| Path product norm invariance (‖(ab)c‖ = ‖a(bc)‖ for octonions) | **PROVEN in Lean** | `formal/OctonionGraph.lean:105` `path_product_norm_invariant` |
| Norm multiplicativity (`‖ab‖² = ‖a‖²·‖b‖²`) | **PROVEN in Lean** | `formal/OctonionAlgebra.lean` (cited, `oct_norm_multiplicative`) |
| 168 theorem (count of non-associative basis triples in 𝕆 equals \|PSL(2,7)\|) | **PROVEN (paper + verified computationally)** | `docs/papers/main/168-theorem.typ` |
| Associator norm dichotomy ‖[eᵢ,eⱼ,eₖ]‖ ∈ {0, 2} for basis elements, all Cayley-Dickson algebras | **PROVEN (elementary)** | `docs/papers/main/168-binary-norm-proof.typ` |
| Tower scaling `T_k = 168·(P_k − 4·P_{k−1})` at `k = 3, 4, 5` | **VERIFIED computationally; proven for k ≤ 5, conjectural for k ≥ 6** | `docs/papers/main/168-tower-preprint.typ` |
| Spectral gap infrastructure (`λ_max`, `λ₂`, `gsp_spectral_gap`) | **IMPLEMENTED in Sounio** | `stdlib/graph/spectral.sio:462` |
| **Associator spectral bound `E[‖[a,b,c]‖] ≤ C·(1−λ₂/λ_max)`** | **NOT FOUND** | — |

I grepped for `spectral.*bound`, `λ₂/λ`, `lambda_2.*lambda_max`, `associator.*spectral`, `associator.*fiedler`, `E\[.*assoc`, and related variants across `docs/`, `stdlib/`, `formal/`. The conjecture is **not stated** in any research note, formal proof, Sounio module, or paper draft. It appears only in the memory index line.

## Assessment

The memory line is from the 2026-03-20 *research-program commit* that laid out the 19-sprint plan for the Non-Associative Epistemic Connectomics direction. That commit was aspirational — a sketch of what the connectomics work could prove, not a record of what had been proven.

Subsequent research effort (memory trace: `project_168_theorem.md`, `project_g2_bridge.md`, `project_non_assoc_connectomics.md`) moved to:
1. The 168 theorem and its tower extension to sedenions/trigintaduonions — **productive, proven, published-ready**.
2. The G₂ bridge — **falsified on CC200 Laplacian eigenmodes; closed door**.
3. Phase 1 non-assoc connectomics (ABIDE-I associator statistics) — **in progress, Phase 1 synthetic gate PASS, real pilot pending**.

The spectral bound is neither confirmed nor actively being pursued. It is best characterized as an **abandoned or deferred conjecture** from the original research plan. The `project_oct_connectomics.md` memory line is stale to that extent.

## Is the conjecture worth pursuing now?

**Technical analysis**: the bound as stated is under-specified. A rigorous formulation needs:
- The distribution over octonion labels (uniform on 𝕆 unit sphere? iid Gaussian components? induced from graph structure?).
- Whether the triple `(a, b, c)` is taken over *node* triples or *edge* triples.
- What `C` depends on (label-norm upper bound? degree sequence?).
- Whether the graph is directed/weighted, how the Laplacian is normalized.

Different specifications produce different bounds, some vacuous, some non-trivial. Without a specific operational interpretation tied to the ABIDE pipeline's actual computation, the conjecture has no Phase 2 role.

**Strategic analysis**: the Phase 2 test (ASD vs TD on `p95(A)` with null permutation whitening) does not require the spectral bound. The bound would provide a *null baseline* — "how large should the associator be under random-label assumption?" — but the Phase 2 protocol already uses an empirical within-subject null that sidesteps this theoretically.

If the bound were proven, it would:
- Explain why certain graph topologies *cannot* produce large associators regardless of labeling.
- Give an a-priori upper bound useful for power analysis.
- Connect the connectomics work to spectral graph theory literature.

If the bound is *false* (or trivial), abandoning it loses nothing operationally.

## Recommendation

**Do not pursue the spectral bound conjecture in the current phase**. Reasons:
1. Phase 2 protocol doesn't depend on it.
2. The 168 theorem + its tower extension is the far higher-yield mathematical direction for the draft.
3. The dissertation (octonion PBPK, γ-thread) is time-bounded and has no spectral-bound dependency.
4. The conjecture is under-specified — turning it into a precise theorem is itself a week of work before the proof question can even be addressed.

**Update the memory**: change `project_oct_connectomics.md`'s "Key conjecture: associator spectral bound" line to accurately reflect the current state — i.e., mark it as deferred/abandoned, not active. Otherwise future sessions continue to inherit the false impression that it's load-bearing.

**If revived later**: the natural operational specification would tie the LHS to the observed `p95(A)` statistic from the Phase 2 pipeline and the RHS to eigenvalues of the ROI adjacency matrix. Numerical evidence from Phase 2 (if the experiment runs) could then *empirically* support or refute the bound, motivating a formal statement. This is the scientifically efficient order: observe first, conjecture second, prove third.

## What `formal/OctonionGraph.lean` establishes (the real δ-level result)

The actual formal output of the connectomics thread is the **path product norm invariance** theorem in Lean. Worth stating explicitly because it *is* load-bearing for Part II of the draft:

> For any two parenthesizations of an octonion path product, the *norm-squared* is the same. The associator `[a,b,c] = (ab)c − a(bc)` is nonzero (as an octonion) but its magnitude constraint is subtle: `‖(ab)c‖² = ‖a(bc)‖²` (the two paths have the same magnitude), yet `(ab)c ≠ a(bc)` (they have different directions).

This distinguishes the octonion case from the associative case (where `(ab)c = a(bc)` as elements) and from a hypothetical "random" non-associative algebra (where norms might also differ). Norm-preservation + directional non-associativity is the specific structure the connectomics experiment probes: *variance in direction, not magnitude*.

This theorem is **proven**, **short** (three lines of Lean using `oct_norm_multiplicative`), and **exactly the theoretical underpinning Part II of the draft needs**. Cite it there; don't oversell by invoking an unstated spectral bound.

## δ-thread closure

δ was a status-check. The codebase does not contain the conjecture listed in memory; the scientifically defensible claim for Part II is the path product norm invariance theorem that *is* in Lean. Memory index will be corrected (separate commit). No compiler or research code changes emerge from this thread.
