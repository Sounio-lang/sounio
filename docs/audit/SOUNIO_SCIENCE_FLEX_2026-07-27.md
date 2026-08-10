<!-- docs:meta
topic_id: repo.docs.audit.sounio-science-flex-2026-07-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sounio-science-flex-2026-07-27
-->

# Sounio Science Flex — multi-domain computational receipt

**Date:** 2026-07-27  
**Status:** live  
**Example:** `examples/sounio_science_flex/main.sio`  
**Gate:** `scripts/ci/sounio_science_flex_gate.sh` → `SOUNIO_SCIENCE_FLEX_GATE_OK`

## Why this exists

While compiler residual lanes (tuple f64 float, self-host soft-keywords) are owned
elsewhere, Sounio still ships a **single-language science stack** that peers do
not: GUM uncertainty as a value type, Knightian p-boxes for clinical refuse/prescribe,
and non-associative algebra on the same spine.

## Measured receipt (lean_single, 2026-07-27)

| Pillar | Observable | Value | Pass |
|---|---|---:|---|
| P1 HEP-GUM | σ(e⁺e⁻→μμ) @ √s=10 GeV | 868.543939 ± 0.000977 pb | 3101 |
| P2 EW-GUM | Γ(Z→ee) | 0.083410 GeV (conf 846) | 3102 |
| P2 EW-amp | pole \|M\|² | 3.249468e-7 | 3102 |
| P3 α_s | α_s(M_Z) / α_s(1 TeV) | 0.117900 / 0.089815 | 3103 |
| P4 Clinical | Vanco Cmin pre-TDM | [9.05, 24.30] → **REFUSE** | 3104 |
| P4 Clinical | Vanco Cmin post-TDM | [12.82, 17.36] → **PRESCRIBE** | 3104 |
| P5 Algebra | octonion associator GUM var | 0.640000 (= 64σ²) | 3105 |

Marker: `SOUNIO_SCIENCE_FLEX_OK`

## What competitors do not ship as one typed binary

| Capability | ROOT / MadGraph | Julia SciML | Python + GUMPy | **Sounio** |
|---|---|---|---|---|
| PDG→QFT GUM chain | external | DIY | DIY | **native `Epistemic`** |
| Knightian clinical refuse | no | DIY | DIY | **native `PBox`** |
| Octonion GUM associator | no | DIY | DIY | **stdlib + GUM** |
| One ELF, zero FFI glue | no | no | no | **yes** |

## Reproduce

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/sounio_science_flex/main.sio
bash scripts/ci/sounio_science_flex_gate.sh
```

## Scope honesty

- Vancomycin module is a **compiler/stdlib Knightian trough screen**, not an
  AUC-guided clinical dosing engine (see `stdlib/clinical/vancomycin_pbpk.sio`).
- Chemistry CRN stack is a parallel lane (`stdlib/chemistry/**`, other claim).
- Madaros multimodule tuple residual is product-mitigated for NC couplings;
  this gate requires lean_single and treats Madaros as optional bonus.
