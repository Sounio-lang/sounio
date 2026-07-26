<!-- docs:meta
topic_id: repo.docs.research.rupture-r2-full-tubular-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-r2-full-tubular-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# R2-full — continuous \(d_{\mathrm{sing}}\) tubular law (measured)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `R2_FULL_MEASURED` — exact anchors + MC measurements; **not** a theorem  
**Harness:** `scripts/research/rupture_r2_full_tubular_probe.py`  
**Parents:** `rupture-abcd-claims_2026-07-24.md` §A, `rupture_r2_fiber_measure_contract.py` (partial)

---

## 1. What R2-partial already closed

- 84 projective primitives / 168 edges / 7 fibers, intra-fiber annihilation only  
- Frente A exact rational measure on the canonical slice (\(\mathrm{Var}=0\) on-locus)  
- Random mixed-half annihilation rate \(\ll 1\)

That is the **combinatorial** law. R2-full asks about the **continuous**
neighbourhood of the zero-divisor locus in \(\mathbb{S}^{15}\).

---

## 2. Distance to singularity

\[
d_{\mathrm{sing}}(x)=\frac{|\det L_x|^{1/16}}{\|x\|}
\]

(scale-free; vanishes exactly on \(\{\det L_x=0\}\)).

---

## 3. Exact anchors (fraction arithmetic / Bareiss)

| ID | Statement | Result |
|---|---|---|
| A0 | \(\det L_{e_1}=\pm 1\) (basis non-singular) | PASS |
| A1 | All 84 primitives: \(\det L_x=0\), rank \(L_x=12\) (corank 4) | PASS |
| A2 | Transversal slice on all 168 edges: vanishing order \(\approx 4\) at the locus, identical across 7 fibers | PASS |
| A2+ | Exact degree-16 interpolation per fiber: vanishing order **exactly 4**, leading coeff \(256\) | PASS |

Primitives are **non-generic** high-contact points of the discriminant hypersurface
(corank 4, not simple corank 1).

---

## 4. Measured statements (float MC, seed fixed)

| ID | Statement | Result (typical) |
|---|---|---|
| M1 | Uniform MC on \(\mathbb{S}^{15}\): essentially no mass with \(d_{\mathrm{sing}}<0.5\) | tube \(\ll \mu_G=1\) |
| M2 | Local slopes \(d_{\mathrm{sing}}(p+tu)/t^{1/4}\) agree across 7 fibers (rel spread \(<5\%\)) | PASS |
| M3 | Model tube mass from local \(t^{1/4}\) law (declared approx., \(t^*\le 0.05\)) | estimator only |

---

## 5. Verdict vocabulary

| Verdict | Meaning |
|---|---|
| `R2_FULL_MEASURED` | exact anchors pass and measured sections ran — **current** |
| `R2_FULL_PROBE_BROKEN` | exact anchors fail |

**Not claimed:** continuous law as theorem; D3; clinical content; alignment (ord 2″).

---

## 6. Reproduce

```bash
python3 scripts/research/rupture_r2_full_tubular_probe.py
# expect: R2_FULL_VERDICT R2_FULL_MEASURED, R2_FULL_PROBE_OK

bash scripts/ci/rupture_abcd_contracts_gate.sh
```

---

## 7. Provenance

Exact/measured design originated on branch `kimi/rupture-r2-full-tubular-20260725`
(`R2_FULL_MEASURED`), integrated into the A+B+C+D gate after R3/R4 greenlanded
on `main` via PR #1432.

## 8. AI disclosure

Probe under human direction. No clinical claims. GAIDeT-ICMJE 2025.
