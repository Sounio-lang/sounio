<!-- docs:meta
topic_id: repo.docs.research.probe-result-multiseed
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-result-multiseed
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Multi-seed panels — LSTM init + ResMLP paired Δ (2026-07-21)

Closes the two multi-seed `[FILL]` markers left in `probe-preprint-draft.md` v0.2. Harnesses and JSON:

- `multiseed_lstm_init.py` → `artifacts/multiseed_lstm_init.json`
- `multiseed_resmlp.py` → `artifacts/multiseed_resmlp.json`

## 1. Untrained LSTM init panel (H=40, pure numpy)

Matches the architecture of the primary false-positive run (`PROBE-RESULT-lstm-adding.md`). Analytic per-step Jacobians (same family as `probe_h256_init.py`); no torch.

| parameter | value |
|---|---|
| H, T | 40, 30 |
| n_seeds | 16 |
| n_seq / seed | 16 |
| show k | 1, 2, 3, 4, 6, 8, 12 |

INIT h→h mean over seeds (sd in parentheses):

| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline √(k/H) | 0.16 | 0.22 | 0.27 | 0.32 | 0.39 | 0.45 | 0.55 |
| **INIT h→h** | 0.973 (0.032) | 0.979 (0.026) | 0.987 (0.015) | **0.992 (0.005)** | 0.994 (0.003) | 0.997 (0.002) | 0.996 (0.003) |

**@k=4 across seeds:** mean 0.992, sd 0.005, min 0.981, max 0.997. Fraction of seeds with INIT@4 > 0.95: **1.00**.

**Verdict.** High init alignment is **stable across seeds**, not a one-seed fluke. The primary-run trained value 0.92 at k=4 sits *below* every init seed. Control (iii) of the LSTM section holds under multi-seed.

## 2. ResMLP multi-seed paired Δ (clean target)

Same architecture as `deep_ffn_train.py` / `PROBE-RESULT-deep-ffn.md`. Torch CPU; early-stop when test acc ≥ 0.90 (min 1000 steps).

| parameter | value |
|---|---|
| W, L, d | 96, 8, 64 |
| n_seeds | 16 |
| n_input / seed | 16 |
| n_scrambles / input | 8 (median used as null) |
| pooled n for Δ | 16 × 16 = **256** |
| acc across seeds | mean **0.941 ± 0.006** (min 0.926) |
| steps | all seeds stopped at 1001 |

Mean over seeds of per-seed mean-over-inputs `align(k)`:

| k | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline √(k/W) | 0.10 | 0.14 | 0.20 | 0.29 | 0.41 | 0.58 | 0.71 |
| **trained** | 0.083 | 0.121 | 0.174 | 0.250 | 0.359 | 0.519 | 0.649 |
| init | 0.080 | 0.114 | 0.168 | 0.243 | 0.349 | 0.507 | 0.635 |
| scramble | 0.081 | 0.118 | 0.169 | 0.244 | 0.350 | 0.507 | 0.635 |

Paired $\Delta_i(k) = \mathrm{align}_i^{\mathrm{tr}}(k) - \operatorname{median}_b \mathrm{align}_{i,b}^{\mathrm{scr}}(k)$, pooled over seeds × inputs. One-sided sign-flip $p$ under $H_0$: $\Delta$ symmetric about 0 ($B{=}9999$):

| k | mean Δ | sd | med Δ | frac(Δ>0) | p_signflip | n |
|---|---:|---:|---:|---:|---:|---:|
| 1 | +0.0025 | 0.025 | −0.0001 | 0.50 | 0.053 | 256 |
| 2 | +0.0034 | 0.017 | +0.0036 | 0.56 | 0.0016 | 256 |
| 4 | +0.0043 | 0.013 | +0.0039 | 0.64 | 0.0001 | 256 |
| 8 | +0.0064 | 0.009 | +0.0068 | 0.77 | 0.0001 | 256 |
| 16 | +0.0090 | 0.006 | +0.0091 | 0.94 | 0.0001 | 256 |
| 32 | +0.0120 | 0.004 | +0.0121 | 1.00 | 0.0001 | 256 |
| 48 | +0.0134 | 0.003 | +0.0133 | 1.00 | 0.0001 | 256 |

**How to read this.** At large $n$ a mean Δ of order $0.01$ is *detectable* (p ≪ 0.05 for $k\ge 2$) but **not a subspace-annihilation signal**:

1. Pre-declared substantive threshold was meanΔ > 0.05 *and* p < 0.05 — **no k meets it**.
2. Absolute curves: trained ≈ init ≈ scramble at every k; all three sit *below* the analytic baseline (the §6.3 baseline caveat stands; scramble is the right comparator).
3. Shape: mean Δ *rises* with k (0.004 → 0.013), the opposite of a small-k annihilation shoulder.

**Verdict: NEGATIVE** for learned subspace annihilation on the clean target, now with multi-seed dispersion and an empirical p. The single-seed “trained exceeds untrained by 0.01–0.02 at large k” eyeball is real as a tiny offset, not as a signature.

## Reproduce

```bash
# LSTM init (pure numpy, ~1 min)
python3 docs/research/multiseed_lstm_init.py

# ResMLP (torch CPU; ~hours for 16 seeds — or set N_SEEDS=4 for a smoke)
export PYTHONPATH=...  # if needed for torch
N_SEEDS=16 N_INPUT=16 N_SCR=8 MAX_STEPS=5000 ACC_STOP=0.90 \
  python3 -u docs/research/multiseed_resmlp.py
```
