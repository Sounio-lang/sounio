<!-- docs:meta
topic_id: repo.docs.research.train-and-probe-usage
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.train-and-probe-usage
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The end-to-end target: train an LSTM on a long-dependency task, then probe it (`train_and_probe_lstm.py`)

*The instrument, the target, and the training in one self-contained torch script. It answers the question
the whole line built toward: in an LSTM where vanishing gradient is known to operate, is that vanishing
MAGNITUDE (uniform slide) or SUBSPACE DEATH (a small-k shoulder in the dense h→h block, above the
conditional null)? No new method — the accumulated protocol, run.*

## What it does
1. **Trains** an LSTM (explicit `LSTMCell`) on the **adding problem** (Hochreiter–Schmidhuber) — the
   canonical long-dependency task where the gradient must survive across the sequence.
2. **Extracts** the per-step state Jacobians `∂[h;c]_{t+1}/∂[h;c]_t` (2H×2H) along realized sequences.
3. **Analyzes the h→h block** (dense — the only place the signature is claimable) with `align(k)` swept over
   all k, and the **c→c block** as the free architectural positive control (diagonal ⇒ align≈1).
4. **Null = orientation scramble** (`J_t → O_t J_t O_{t-1}^T`, O random orthogonal): **preserves the product
   spectrum exactly** (validated: max|Δσ|≈1e-15) and destroys only the geometric alignment — the conditional
   null of the mechanism. (Also run the untrained-init and shuffled-weights controls.)
5. **Discovery/confirmation split with m† frozen:** the shoulder position `m†` is chosen on a discovery half
   and *frozen*; the effect (trained alignment vs scramble null at `m†`) is measured on the held-out half —
   removing the selection bias of picking the k that maximizes the effect.

## Reading the result
The result is the **Cohen d** (trained − orientation-scramble null) at the frozen `m†`, **not a point
label**. A real signature requires: a shoulder at **small k** (`m†/H < ½`), trained alignment there **beating
the orientation-scramble null** (`d > ~0.8`), **and** the h→h shoulder beating the c→c architectural control.
If trained alignment sits at the scramble null, the vanishing gradient is magnitude — and that negative is
as valuable as any positive on this line.

## Notes
- Defaults `H=48, T=40, 4000 steps` run on CPU in minutes and GPU in seconds; scale `H,T` up for a sharper
  test. Per-step Jacobians cost 2H backward passes/step — the main expense; reduce `n_seq` first.
- For an S4/Mamba or transformer, do **not** reuse this directly: diagonal `Ā` (S4/Mamba) forces alignment by
  architecture, and residuals (transformer) require probing the branch `F'_l = J_l − I`. This target is a
  **dense** RNN by design.
- `gap(T)` and `β = λ_{m+1}−λ_m` via the QR method (`lyapunov_qr.py`) are the complementary magnitude-axis
  readout; the alignment (this script) is the discriminant that survived the rotating control.

Positioned honestly (see `lyapunov-repositioning.md`): this is Lyapunov-spectrum + covariant-Lyapunov-vector
analysis; the only claim is the structural gap hypothesis. The next thing is the run, not more method.
