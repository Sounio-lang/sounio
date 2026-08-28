<!-- docs:meta
topic_id: repo.docs.research.lyapunov-repositioning
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lyapunov-repositioning
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The instrument is Lyapunov-spectrum + covariant-Lyapunov-vector analysis — position it, and fix the numerical ceiling

*Four corrections (OPUS-4.8-EXTRA). The first removes a censorship that amputated the primary result; the
second removes a citation gap that would make the whole thing read as rediscovery; the last two are minutes.
The verdict from before stands and is recorded first: the product-spectrum gap is dead as a discriminant
(the rotating control killed it); the principal-angle alignment survived.*

## §1 (mandatory) — do not form the product; use discrete QR (`lyapunov_qr.py`)
Forming `P_T = J_T…J_1` explicitly destroys the tail: the condition number passes machine epsilon and the
small singular values stop being resolved, so `gap(T)` **censors at ~12–16 decades** — exactly where the
signature should be strongest. The fix is the standard Lyapunov-spectrum algorithm: `Q_0=I`,
`J_t Q_{t-1}=Q_t R_t`, `log σ_i(P_T) ≈ Σ_t log|R_t[i,i]|`, reorthonormalizing each step. Validated:

| T | direct min log₁₀σ | **QR min log₁₀σ** | QR gap(σ4→σ5) |
|---|---|---|---|
| 16 | −16.7 (censored) | **−19.6** | 0.1 |
| 64 | −16.8 (censored) | **−78.8** | 6.3 |
| 256 | −19.5 (censored) | **−312.7** | 34.6 |

The direct method saturates at machine epsilon; QR descends **linearly in T** (`log σ_i ≈ λ_i·T`), so the
gap grows without a ceiling. Same cost (one QR per step), and the `Q_t` frames are the leading Oseledets
directions the alignment needs — no SVD of the product.

## §2 (mandatory) — this IS Lyapunov analysis; cite it or be read as rediscovery
With §1 the object is explicit: `G(T) = α + βT` where **`β = λ_{m+1} − λ_m`, a Lyapunov-exponent
difference** (measured slope on the aligned test stack: `β ≈ 0.146` decades/step). RNN Lyapunov spectra are
established — **Engelken, Wolf & Abbott**; **Vogt et al.** (exponents ↔ trainability). And the
subspace-alignment measure that survived the rotating control is essentially **covariant Lyapunov vectors**
(**Ginelli et al.**) — the alignment of the Oseledets subspaces. So: the spectrum is not the novelty, and
neither is the alignment per se. The only claimable contribution is the **specific gap structure** (a
small-`k` shoulder with a healthy bulk, motivated by the algebra's `4/8/4`) and its interpretation — and
even that must be **positioned against CLVs, not claimed fresh**. Without these citations the reviewer
supplies them and the paper is a rediscovery.

## §3 (minutes) — decompose the LSTM state, don't choose it (`lstm_probe.py: block_report`)
The closed system is `(h,c)`; compute the full `P_T`, then report alignment **by block**. The `c→c` block
is **diagonal by architecture** (gates depend on `h_{t-1},x_t`, never on `c_{t-1}`) — a **free internal
positive control**: it reads `align≈1` trivially. The signature is claimable **only in the dense `h→h`
block**. If the full-state alignment matches `c→c` while `h→h` sits at the null, the alignment is entirely
architectural — discovered inside the same computation, no extra experiment.

## §4 (already done) — sweep all m
`align_curve.py` already sweeps `k = 1…d-1`; there was never an `m ≤ D/4` cut in this repo's version (that
was inherited from the `4/16` sedenion ratio, which does not translate to LSTM width `H`). The shoulder
position `m*/D` is whole data, not a chosen parameter.

## Status — ready to run, positioned honestly
The instrument now: (i) measures the spectrum by QR without a numerical ceiling, (ii) is positioned as
Lyapunov-spectrum + CLV analysis with the contribution narrowed to the structural gap hypothesis, (iii)
decomposes the LSTM state with a free positive control, (iv) sweeps all `m`. Primary target: a dense RNN
(LSTM/GRU) trained on a long-dependency task; S4/Mamba disqualified (diagonal `Ā`); S-SSM as declared
calibration. The next thing is not more method — it is a checkpoint and an afternoon of GPU. Five negatives
have been worth more than any positive on this line; the sixth, or the first positive, is one run away, and
only the data decides. Harnesses `lyapunov_qr.py`, `lstm_probe.py` (`lstm_step_jacobians`, `block_report`),
`align_curve.py`.
