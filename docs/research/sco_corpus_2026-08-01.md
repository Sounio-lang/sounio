<!-- docs:meta
topic_id: repo.docs.research.sco-corpus-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sco-corpus-2026-08-01
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SCO Corpus (Step II)

**Status:** adversarial corpus specification + runnable drivers  
**Date:** 2026-08-01  
**Depends on:** [`scientific_compilation_obligation_2026-08-01.md`](./scientific_compilation_obligation_2026-08-01.md) (Step I)  
**Root:** `examples/sco_corpus/`  
**Gate:** `scripts/ci/sco_corpus_gate.sh`

---

## 1. Purpose

Step I defined SCO, \(\mathsf{Obs}\), \(\preceq\), \(E(\tau)\), and violation shapes V1–V3.  
Step II **builds programs that try to make the compiler lose** under the SCO adversary:

- multi-module native default path;
- exact integer sentinels (no approximate floats in the pass criterion);
- one designated sink per item;
- non-vacuity hooks reserved for Step III (ablation of \(E(\tau)\)).

This is not a regression dump of unrelated tests. Every file exists to force one obligation clause.

---

## 2. Design rules

1. **One violation class per item** (V1, V2, or V3).  
2. **Sentinel is an integer** printed on its own line (or a fixed multi-line block listed below).  
3. **Entitlement** is stated in this document; the gate only checks the printed sentinel.  
4. **Imports required** — monorepo single-file versions are not SCO items (adversary clause: split modules).  
5. **Default engine** — gate uses `./bin/souc` as users do; no alternate-engine escape hatch.  
6. **Pass today ≠ SCO discharged.** Green means the *current* default path happens to meet the sentinel. SCO itself still requires Step III theory respect + Step I clause 6 non-vacuity.

---

## 3. Items

### V1 — Silent coverage collapse  
**Path:** `examples/sco_corpus/v1_coverage/`  
**Files:** `v1_leaf.sio` (imported), `v1_main.sio` (driver)  
**Scientific entitlement \(O\):** finite-dof Type-A-dominant budget → Student-\(t\) coverage, not normal 1.96.  
**Sink:** print \(k_{95} \times 1000\) as `i64`.  
**Sentinel (exact):**

```text
2776
```

**SCO fail modes:**

| Print | Meaning |
|------:|---------|
| `2776` | coverage factor preserved under import+native (necessary, not sufficient for full SCO) |
| `1960` | classic silent collapse to normal \(k\) — **V1 adversary wins** |
| other / crash | ordinary defect or residual miscompile — gate red |

**Ablation hook (Step III):** force ordinary-float rewrite freedom / disable scientific lower checks → must not still be allowed to claim SCO if \(k\) changes or theory is off.

---

### V2 — Illegal algebraic reassociation  
**Path:** `examples/sco_corpus/v2_algebra/`  
**Files:** `v2_leaf.sio`, `v2_main.sio`  
**Scientific entitlement \(O\):** non-associative product variance class under domain algebra — Fano triple vs non-Fano triple must remain **distinct** after multi-module native.  
**Sink:** two integer lines, micro-unit scaling \(10^6\):

```text
250000
4250000
```

**Meaning:** Fano-augmented variance class `0.25` vs non-Fano `4.25` (κ=1, base σ²=0.25).  
**SCO fail modes:** equalised lines, swapped classes, or a single collapsed value after optimisations that pretend floats are associative — **V2 adversary wins**.

**Ablation hook (Step III):** with \(E(\tau)\) disabled, illegal reassoc may merge classes; SCO requires that default-path \(E\) prevents that *and* that ablation changes the witness (non-vacuity).

---

### V3 — Erased observation barrier  
**Path:** `examples/sco_corpus/v3_observe/`  
**Files:** `v3_leaf.sio`, `v3_main.sio`  
**Scientific entitlement \(O\):** payload originates behind an observation barrier on the import path; the designated sink may print the scientific payload class only after a discharge step whose effect row includes `Observe`.  
**Sink protocol (two lines):**

```text
1
7001
```

| Line | Entitlement |
|------|-------------|
| `1` | barrier tag from leaf (`sco_barrier_tag` = ω unobserved origin) |
| `7001` | `sco_discharge_observe` under `Observe` maps raw payload units 7 → positive class |

**SCO fail modes:** printing `7001` without barrier line `1`; discharging without `Observe` on the discharge function (must be effect-illegal); optimisations that drop the barrier report while keeping a plausible payload class — **V3 adversary wins**.

**Note:** External report protocol is fixed here. IR-level barrier non-merge (\(E(\tau)\) shrinking across Observe) is Step III; this item keeps the observable package shape live end-to-end.

---

## 4. Gate contract

```bash
bash scripts/ci/sco_corpus_gate.sh
```

| Result | Meaning |
|--------|---------|
| `SCO_CORPUS_GATE_OK` | all three items compile on default `./bin/souc`, run, match exact sentinels |
| non-zero exit | at least one SCO corpus adversary won or tool failed |

Environment:

- `SOUNIO_STDLIB_PATH` set to repo `stdlib/`  
- cwd = repo root (worktree root)

---

## 5. What green does *not* claim

- That \(E(\tau)\) is installed on the mid-end (Step III).  
- That non-vacuity (SCO clause 6) holds.  
- That separate-compilation theorems exist (Step V).  
- That every scientific program is safe — only these three adversarial shapes.

Green = **the corpus is live and the current default path does not already lose V1–V3 on these programs.**  
That is the floor for building SCO, not the ceiling.

---

## 6. Extension rules (later items)

New items must:

1. name a violation class (Vn) or a new class defined in an SCO note revision;  
2. state \(\mathsf{Obs}\) components under test;  
3. give exact sentinels;  
4. be multi-module;  
5. land under `examples/sco_corpus/` with a gate stanza.

Do not add “smoke” programs that do not attack \(\preceq\) or \(E(\tau)\).

---

## 7. Step II acceptance

Step II is complete when:

1. This document exists.  
2. V1–V3 drivers exist under `examples/sco_corpus/`.  
3. `scripts/ci/sco_corpus_gate.sh` encodes the sentinels above.  
4. A human can run the gate and interpret red/green against SCO — without re-reading Step I for sentinel values.

## 8. Forward link (Step III)

\(E(\tau)\) install at lower + combined gate:  
[`sco_etau_2026-08-02.md`](./sco_etau_2026-08-02.md) · `scripts/ci/sco_etau_install_gate.sh`

End of Step II specification.
