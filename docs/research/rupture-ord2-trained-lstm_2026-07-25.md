<!-- docs:meta
topic_id: repo.docs.research.rupture-ord2-trained-lstm-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-ord2-trained-lstm-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Ord 2″ trained target — LSTM on the adding problem (multi-path)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `ORD2_TRAINED_NO_SIGNATURE` (controls reject subspace death)  
**Harness:** `scripts/research/rupture_ord2_trained_lstm_probe.py`  
**Parents:** `probe-corrected-protocol.md`, `PROBE-RESULT-lstm-adding.md`,
`rupture-ord2-alignment_2026-07-25.md`, `train_and_probe_lstm.py`

---

## 1. What this is

The corrected protocol’s **primary scientific target**: a **non-sedenion** recurrent
model trained on a long-dependency task, probed with multi-path Jacobians and
full controls — not a point label from scramble-null alone.

| Piece | Choice |
|---|---|
| Model | `nn.LSTMCell` + linear head (explicit per-step Jacobians) |
| Task | adding problem (marks select two timesteps to sum) |
| Paths | \(N\) random sequences; discovery/confirmation split |
| Block | dense **h→h** only (claimable surface) |
| Nulls | orientation scramble; **untrained init**; `align(k)` curve shape |

Sedenion-aligned stacks remain the **declared positive control** in
`rupture_ord2_alignment_contract.py` (architecture, not learning).

---

## 2. Measured results

### CI defaults (`H=20 T=24 STEPS=600 N_SEQ=40`)

| Quantity | Trained | Init |
|---|---|---|
| confirm align @ m† | ≈ 0.98 | ≈ 1.00 |
| scramble null | ≈ 0.26 | ≈ 0.38 |
| Cohen d (vs scramble) | large (+) | large (+) |
| `align(k)` at large k | flat-high | flat-high |
| init ≥ trained? | **yes** | — |

### Heavier (`H=32 T=32 STEPS=1200 N_SEQ=64`, test MSE ≈ 0.009)

| Quantity | Trained | Init |
|---|---|---|
| confirm align | 0.937 | 0.995 |
| Cohen d vs scramble | +46 | — |
| init ≥ trained? | **yes** | — |
| shape | flat-high (not annihilation) | — |

Verdict unchanged under heavier training.

**Controls that reject subspace death:**

1. **Shape** — `align(k)` stays high through large \(k\) (low effective rank), does
   not peak at small \(k\) then drop to baseline (annihilation shape).
2. **Init** — untrained net has *higher* h→h alignment than trained → architectural.
3. Scramble alone would *false-positive* (d ≫ 1) — exactly the incomplete instrument
   trap documented in `PROBE-RESULT-lstm-adding.md`.

**Verdict:** `ORD2_TRAINED_NO_SIGNATURE`  
Vanishing gradient on this target is classical magnitude / rank / architecture —
not sedenion-style composing annihilation.

This matches the historical sixth negative, now **executable and gated**.

---

## 3. Decision rules (encoded)

`ORD2_TRAINED_SUBSPACE_DEATH` only if **all** hold:

- scramble Cohen d large,
- m† small,
- shape is *not* flat-high low-rank,
- trained alignment ≫ init.

Otherwise `ORD2_TRAINED_NO_SIGNATURE`.

---

## 4. Reproduce

```bash
# repo venv with torch+numpy
./.venv/bin/python scripts/research/rupture_ord2_trained_lstm_probe.py

# optional heavier run
ORD2_LSTM_H=40 ORD2_LSTM_STEPS=2000 ORD2_LSTM_NSEQ=80 \
  ./.venv/bin/python scripts/research/rupture_ord2_trained_lstm_probe.py

# skip in gate
SOUNIO_SKIP_ORD2_TRAINED=1 bash scripts/ci/rupture_abcd_contracts_gate.sh
```

---

## 5. Dependencies

Requires `numpy` and `torch` (CPU ok). The gate invokes `.venv/bin/python` when
present; without deps the probe emits `ORD2_TRAINED_SKIP` (soft OK).

## 6. AI disclosure

Executable landing of the existing corrected protocol under human direction
(2026-07-25). No clinical claims. GAIDeT-ICMJE 2025.
