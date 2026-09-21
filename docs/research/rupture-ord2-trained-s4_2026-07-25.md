<!-- docs:meta
topic_id: repo.docs.research.rupture-ord2-trained-s4-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-ord2-trained-s4-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Ord 2″ trained target — diagonal S4-style SSM (multi-path)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `ORD2_S4_NO_SIGNATURE`  
**Harness:** `scripts/research/rupture_ord2_trained_s4_probe.py`  
**Parents:** `rupture-ord2-trained-lstm_2026-07-25.md`, `probe-corrected-protocol.md`

---

## 1. Model (honest scope)

**Not** full complex HiPPO-S4 / Mamba. A minimal **real diagonal linear SSM**
in the structured-SSM family:

\[
h_{t+1} = \mathrm{diag}(\lambda)\, h_t + B x_t,\qquad
\lambda = -\mathrm{softplus}(a)\in(-\infty,0),\qquad
y = C h_T.
\]

Why this target:

- non-sedenion (no Cayley–Dickson product),
- still a **structured SSM** (protocol’s S4/Mamba slot),
- \(\partial h_{t+1}/\partial h_t\) is **diagonal-dominated by construction** —
  so architectural alignment is expected and the init control is decisive.

---

## 2. Protocol (same as LSTM probe)

Multi-path Jacobians, `align(k)` curve, orientation-scramble null, untrained
init, discovery/confirmation split. Same decision rules.

---

## 3. Measured results

### CI defaults (`H=20 T=24 STEPS=600 N_SEQ=40`)

| | Trained | Init |
|---|---|---|
| confirm align | **1.000** | **1.000** |
| scramble | ~0.18 | ~0.18 |
| `align(k)` | flat **1.0** all k | flat **1.0** |
| Cohen d vs scramble | huge | huge |

### Heavier (`H=32 T=32 STEPS=1200 N_SEQ=64`)

Same pattern: trained=init align **1.000**, flat-high curve, scramble d huge.
(MSE ~0.36 — task still hard for pure diagonal SSM; controls unchanged.)

**Verdict (both):** `ORD2_S4_NO_SIGNATURE`

Reasons encoded:

- init_control_fails_architectural (align ≡ 1 at init),
- align(k) flat-high / diagonal structure (not annihilation shoulder).

Scramble-alone would false-positive again — the incomplete instrument trap.

---

## 4. Interpretation

On this diagonal SSM, dying-subspace alignment is **pure architecture**
(\(\partial h'/\partial h \approx \mathrm{diag}(\lambda)\)). Training on the adding
problem does not create a sedenion-style composing-annihilation signature.
Together with the LSTM result, two non-sedenion families (gated RNN + structured
SSM) both reject subspace death under the full control suite.

---

## 5. Reproduce

```bash
./.venv/bin/python scripts/research/rupture_ord2_trained_s4_probe.py

# heavier
ORD2_S4_H=32 ORD2_S4_T=32 ORD2_S4_STEPS=1200 ORD2_S4_NSEQ=64 \
  ./.venv/bin/python scripts/research/rupture_ord2_trained_s4_probe.py

SOUNIO_SKIP_ORD2_S4=1   # soft-skip in gate
```

## 6. AI disclosure

Human-directed. No clinical claims. GAIDeT-ICMJE 2025.
