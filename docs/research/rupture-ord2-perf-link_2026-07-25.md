<!-- docs:meta
topic_id: repo.docs.research.rupture-ord2-perf-link-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-ord2-perf-link-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Ord 2″ protocol §5 — alignment vs long-sequence performance

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `ORD2_PERF_NO_LINK`  
**Harness:** `scripts/research/rupture_ord2_perf_link_probe.py`  
**Parents:** `probe-corrected-protocol.md` §5, `rupture-ord2-trained-lstm_2026-07-25.md`

---

## 1. Why this step exists

Multi-path alignment without a performance link is only a **spectrum shape**.
The corrected protocol’s last step asks whether alignment/gap predicts:

- loss plateaus,
- long-sequence degradation,
- or specific unlearned examples.

If not, it is not a diagnostic of composed annihilation in training.

---

## 2. Design

| Piece | Choice |
|---|---|
| Model | LSTMCell (same family as the NO_SIGNATURE probe) |
| Train | adding problem at \(T_{\mathrm{train}}\) |
| Path link | per-sequence h→h mean align(\(k\)) vs squared error at \(T_{\mathrm{train}}\) |
| Length gen | MSE at \(T \in \{T, 1.5T, 2T, 3T\}\) |
| Split | high-align vs low-align median split of path errors |

**Annihilation hypothesis (for a positive link):** higher dead-subspace alignment
→ worse path error / worse long-\(T\) behaviour.

---

## 3. Measured result (CI defaults)

Config: `H=24 T=24 STEPS=800 N_PATH=48 K=4`.

| Quantity | Value |
|---|---|
| test MSE @ \(T_{\mathrm{train}}\) | ≈ 0.019 |
| path Pearson(align, err) | **−0.15** (weak, wrong sign) |
| err high-align / low-align | 0.010 / 0.020 (high-align **better**) |
| MSE @ \(3T\) / MSE @ \(T\) | **≈ 8.2** (long-\(T\) degrades) |

**Verdict:** `ORD2_PERF_NO_LINK`

Long sequences **do** degrade (classical length generalisation / vanishing), but
that degradation is **not** predicted by dying-subspace alignment. High alignment
paths are not worse. Therefore the architectural alignment signature is **not a
performance diagnostic** on this target.

This completes the corrected protocol for the LSTM adding-problem line:

1. Instrument (ord 2″) — OK  
2. Trained multi-path + controls — **NO_SIGNATURE**  
3. Performance link — **NO_LINK**

---

## 4. Reproduce

```bash
./.venv/bin/python scripts/research/rupture_ord2_perf_link_probe.py

SOUNIO_SKIP_ORD2_PERF=1   # soft-skip
```

## 5. AI disclosure

Human-directed. No clinical claims. GAIDeT-ICMJE 2025.
