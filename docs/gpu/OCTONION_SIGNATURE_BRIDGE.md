<!-- docs:meta
topic_id: repo.docs.gpu.octonion-signature-bridge
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.octonion-signature-bridge
-->

# The faithful bridge, resolved: the Cayley–Dickson associator is *structurally orthogonal* to the Massey product

`BORROMEAN_AINFINITY.md` left one thread open: the naive octonion associator of three *net* increments
missed the Borromean/Massey signal (48.9%), and we conjectured a *faithful* bridge — an **octonion-valued
path signature** whose depth-3 term is a genuine *temporal iterated associator* — might capture it. We
built it. It does not, and now we know exactly why.

## The construction (faithful)
Embed the path increments as imaginary octonions `g_t` on the **non-Fano** triple `e₁,e₂,e₄`
(`‖[e₁,e₂,e₄]‖ = 2 ≠ 0`; the Fano triple `e₁,e₂,e₃` gives `0`), and build the octonion signature by
ordered octonion products. Depth-3 has two bracketings; their difference is a genuine iterated associator
```
D = Σ_{r<s<t} ((g_r g_s) g_t) − (g_r (g_s g_t)) = Σ_{r<s<t} [g_r, g_s, g_t]   ∈ Im 𝕆 (8-dim)
```
Unlike the fully-antisymmetric signed-volume of the previous note (which was *identically zero*), **`D` is
genuinely nonzero** (std ≈ 4.0e2, max ≈ 6.9e3). So the temporal octonion signature has a real,
non-vanishing depth-3 non-associative term.

## Result (Borromean/Massey slice; chance = 50%)
| Feature | corr with μ | test acc |
|---|---|---|
| OCT-DEV `[S1,S2]` (octonion depth ≤ 2) | — | 50.0% |
| OCT-ASSOC `D` (iterated associator, depth-3) | **0.023** | 48.0% |
| OCT-FULL `[S1,S2,S3L,S3R,D]` | 0.145 | 47.6% |

The faithful iterated associator is **nonzero yet orthogonal** to the Massey invariant μ (correlation
0.02). This is not a modelling failure — it is a representation-theoretic fact.

## Why — the theorem behind the null (verified numerically to 1e-15)
The Cayley–Dickson associator on imaginary octonions is **alternating**:
```
[a,b,c] = −[b,a,c] = −[a,c,b]        (‖[a,b,c]+[b,a,c]‖ ≈ 5e-15),   [a,b,c] ∈ Im 𝕆
```
i.e. it is (a vector-valued instance of) the **G₂-invariant 3-form** — the fully-antisymmetric `Λ³`
component of a triple. The Borromean/Massey invariant `μ_k = ∫ A_ij dX^k` has **mixed symmetry** (the area
`A_ij` is antisymmetric in `i,j`, then paired with the third index `k`) — the hook `[2,1]` component. In
the decomposition of the depth-3 tensor into irreducibles, **`Λ³` (the associator) and the `[2,1]` hook
(the Massey product) are different summands**, hence orthogonal. No embedding of the Cayley–Dickson
associator — however faithful its temporal structure — can express the Massey product, because they are
*different pieces of the representation*. The empirical `corr = 0.02`, with `D` provably nonzero, confirms
it directly.

## What this settles
- The octonion associator is precisely an **alternating (G₂ 3-form) ternary operation** — this is what our
  compiler lowers to tensor cores and what the whole hypercomplex lane computes.
- It is therefore the exact tool for signals that are **static alternating ternary associators**
  (`NONASSOC_HEADTOHEAD.md`, `BRACKETING_TASK.md`: 95–100%), and **provably the wrong tool** for
  higher-homotopy obstructions of mixed symmetry (Massey/Borromean: orthogonal).
- The A∞ door is real (`BORROMEAN_AINFINITY.md`: the higher invariant is required), but it is a
  *different* non-associative structure than ours. A tool for it would need the mixed-symmetry (free-Lie
  `[2,1]`) bracket, not the Cayley–Dickson associator.

This is the honest, rigorous closure of the bridge question: a **theorem-backed negative** that sharply
delimits the artifact's reach rather than overselling it. Harness `octonion_signature.py`.
