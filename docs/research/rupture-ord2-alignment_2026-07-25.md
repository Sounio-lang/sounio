<!-- docs:meta
topic_id: repo.docs.research.rupture-ord2-alignment-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-ord2-alignment-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Ord 2″ — subspace alignment as the mechanism of composed annihilation

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `ORD2_INSTRUMENT_OK` — instrument + controls green; **not** a trained-model discovery  
**Harness:** `scripts/research/rupture_ord2_alignment_contract.py`  
**Parents:** `probe-corrected-protocol.md`, `mechanism_analysis.py`,
`rupture-programme-synthesis_2026-07-25.md` §3

---

## 1. Why this order exists

R2 (partial / full) instruments the **zero-divisor locus** and its tube
(\(\det L_x\), fibers, \(d_{\mathrm{sing}}\)).

**Ord 2″** asks a different question: when Jacobians (or left-multiplications)
are **composed in depth**, when does structural annihilation *survive* composition?

The corrected probe protocol showed that the product-spectrum **gap alone is a
false positive**: a stack with 4/8/4 structure **per layer** but **rotating** dead
directions produces a large gap without composing structure. The mechanism is
**principal-angle alignment** of consecutive dying subspaces.

---

## 2. What the contract certifies

| Test | Expectation | Result (SEED=20260725) |
|---|---|---|
| Alignment, same ZD (positive control) | \(\approx 0.99\) | **0.989** |
| Alignment, rotating ZD per layer | near baseline \(\sqrt{k/d}\approx 0.5\) | **0.458** |
| Alignment separation | aligned − rotating \(> 0.2\) | **PASS** |
| Rotating gap_dominance at T=16 | often \(>1\) (FP under gap-only) | **12.1** |
| Gaussian / linear-RNN baselines | no forced high alignment from algebra | **0.42 / 0.59** |

**Declared positive control:** sedenion-aligned stack is *calibration* (architecture),
not evidence that a trained net learned annihilation.

**Non-sedenion baselines:** independent Gaussian layers and a shared-\(W\) linear-RNN
style stack. The **discovery** target remains a *trained* LSTM/S4 (not executed here).

---

## 3. Verdict vocabulary

| Verdict | Meaning |
|---|---|
| `ORD2_INSTRUMENT_OK` | alignment separates; gap alone invalid; baselines measured |
| `ORD2_PROBE_BROKEN` | separation failed |

**Not claimed:** trained-model annihilation; clinical content; D3; that linear-RNN
alignment is structural annihilation.

---

## 4. Reproduce

```bash
python3 scripts/research/rupture_ord2_alignment_contract.py
# expect: ORD2_VERDICT ORD2_INSTRUMENT_OK, ORD2_CONTRACT_OK

bash scripts/ci/rupture_abcd_contracts_gate.sh
```

Pure Python (Jacobi eigen / SVD on 16×16); no numpy dependency.

---

## 5. Trained non-sedenion target (done)

Executable multi-path probe on a **trained LSTM** (adding problem) with full
controls: `scripts/research/rupture_ord2_trained_lstm_probe.py`  
Doc: `rupture-ord2-trained-lstm_2026-07-25.md`  
Verdict on CI defaults: **`ORD2_TRAINED_NO_SIGNATURE`** (init + shape reject
annihilation; scramble alone would false-positive).

Remaining: heavier runs / S4; link alignment to long-sequence failure modes
(protocol §5).

## 6. AI disclosure

Contract under human direction (2026-07-25). GAIDeT-ICMJE 2025.
