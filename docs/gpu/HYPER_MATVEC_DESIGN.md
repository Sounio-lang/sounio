<!-- docs:meta
topic_id: repo.docs.gpu.hyper-matvec-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.hyper-matvec-design
-->

# Hypercomplex matrix–vector on tensor cores — the real-capacity primitive

## Why
The current O-SSM has A/B/C = single hypercomplex numbers (~8 params each). That is an identity test,
not a model. Real capacity needs a **hidden dimension** D_h (channels): the state is H ∈ 𝕆^{D_h}
(D_h octonion channels) and the weight A ∈ 𝕆^{D_h×D_h} is a **matrix of octonions**:

    H_out[i] = Σ_j A[i][j] ⊗ H[j]        i, j = 1..D_h channels        (octonion matrix–vector)

## Mapping to a dense tensor-core matmul ("bigger tiles per block")
Let N = D_h·8. Build the **block left-multiplication matrix** A_big ∈ ℝ^{N×N}:

    A_big[(i·8+k)][(j·8+l)] = σ(k⊕l, l) · A[i][j][k⊕l]        (each 8×8 block = L(A[i][j]))

Then the octonion matrix–vector is exactly a dense matmul over a 16-wide batch:

    H_out_big (N×16) = A_big (N×N) · H_big (N×16)

- σ(k⊕l,l) depends only on the intra-octonion indices (k,l) → reuse ossm_Lsgn[k*8+l].
- A_big built f16 in shared (N²·2 bytes); H_big staged f16 (N·16·2 bytes); wmma m16n16k16 tiles:
  M=N → N/16 m-tiles, N_out=16 → 1, K=N → N/16 k-chunks ⇒ (N/16)² tiles, each accumulating its k-chunks.
- D_h=8  → N=64,  A_big 8 KB,  16 tiles.   D_h=16 → N=128, A_big 32 KB, 64 tiles (fits 48 KB shared).

## Capacity
D_h² octonion weights (256 for D_h=16) vs 1 today — and the matmul now *fills* the tensor cores
(128×128), unlike the 8×8 single-octonion tile. Bigger tiles per block AND real capacity, together.

## Intrinsic
    oct_matvec_dh8 / oct_matvec_dh16 (pA, pH, pout):  H_out = A ⊛ H over D_h channels.
    pA = 𝕆^{D_h×D_h} (D_h²·8 f64), pH = 𝕆^{D_h}×16 batch (D_h·8·16 f64), pout = N×16.

## Status
Scaffolded; heavy build gated on the literature-novelty findings (framing the real-task/benchmark that
would make this a model result, not just a bigger identity test).
