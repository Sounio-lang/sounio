<!-- docs:meta
topic_id: repo.docs.research.probe-usage
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-usage
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The checkpoint probe — does a real trained model carry the sedenion signature?

*The cheap, no-training test (`probe_jacobian_spectrum.py`). Compute the singular spectrum of the composed
Jacobian on a few input paths and classify: structural annihilation (low-mult gap) vs magnitude (uniform
slide) vs rank collapse. The transferable core — the classifier — is pure numpy and is **calibrated**
below; the Jacobian itself is one `torch.autograd.functional.jacobian` call.*

## The classifier (validated, no dependencies)
`classify_spectrum(singular_values)` returns a verdict from the shape alone. Discriminant: the **dominant
gap** vs the **spread of the surviving bulk** (`gap_dominance = max_gap / bulk_span`). A clean cliff
(`gap_dominance > 1`, few modes dead behind it, bulk healthy) = **LOW-MULT-GAP**; a gap that is merely part
of a continuous decay (`gap_dominance < 1`) = **UNIFORM-SLIDE**; almost everything dead with 1–2 survivors
= **RANK-COLLAPSE**. Calibration on three known spectra:

| input spectrum | verdict | gap/bulk-spread |
|---|---|---|
| sedenion 4/8/4 tiers (from `spectral_signature.py`) | **LOW-MULT-GAP** | 6.94 |
| real Gaussian deep product | **UNIFORM-SLIDE** | 0.66 |
| rank-collapse toward 1 | **RANK-COLLAPSE** | (1 survivor) |

## Running it on your checkpoint (torch)
```python
from probe_jacobian_spectrum import probe_checkpoint, io_jacobian, classify_spectrum
# 1. build_model() -> your nn.Module (untrained skeleton)
# 2. ckpt_path -> your .pt (state_dict or {'state_dict': ...})
# 3. make_forward(model) -> a function whose Jacobian is the COMPOSITION you care about:
#      • feedforward: return model                       (input → output = J_L…J_1)
#      • recurrence (RNN/SSM/linear-attn): return h0 -> hT (its Jacobian is ∏_t ∂h_{t+1}/∂h_t)
# 4. sample_inputs(k) -> k input tensors (a few paths)
probe_checkpoint(build_model, ckpt_path, make_forward, sample_inputs, n_paths=8)
```

## What to probe, and reading the result
- **Best first target:** a *sequence* model where the state transition is a genuine product — RNN, SSM
  (your S-SSM especially), linear attention. The BPTT Jacobian `∂h_T/∂h_0 = ∏_t ∂h_{t+1}/∂h_t` is the
  natural, already-square composition; restrict `T` so the Jacobian is affordable.
- **Contrast target:** a transformer block — the literature (Dong et al.) predicts **RANK-COLLAPSE** there,
  a good negative control that the classifier should separate from LOW-MULT-GAP.
- **Tractability:** the full input→output Jacobian is `out_dim × in_dim`; for large models restrict to a
  submodule / an early-hidden→late-hidden block so the matrix is square-ish and the SVD is cheap.

**Verdict → meaning.** LOW-MULT-GAP on a real checkpoint = a failure mode dynamical isometry does not
measure (a small subspace died while the bulk lives; residuals preserve norm and do nothing for it) — the
first evidence the sedenion structure is *present in trained models*, not only constructed. UNIFORM-SLIDE =
it is magnitude, the structure is not there, and one afternoon of rented GPU said so. Either way the answer
is decisive and cheap. Harness `probe_jacobian_spectrum.py`.
