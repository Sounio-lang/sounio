<!-- docs:meta
topic_id: repo.docs.research.mercyful-sounio-port-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-sounio-port-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning Sounio port — native runtime in the language

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parent:** `docs/research/mercyful_runtime_spec_2026-07-25.md` (M_GREEN, Python prototype)  
**Harness:** `tests/run-pass/mercyful_exposure_therapy.sio`  
**Gate:** `scripts/ci/mercyful_sounio_gate.sh`  
**Module:** `stdlib/clinical/mercyful.sio`

---

## 1. What this is

The Python prototype (`scripts/research/mercyful_runtime_contract.py`) proved the concept. This rung ports the scheduler to Sounio so the runtime is **native in the language**: a `stdlib/clinical/mercyful.sio` module that computes mercyful paths on finite graphs, and a `run-pass` test that executes the exposure-therapy benchmark through the compiler.

---

## 2. Design constraints

- **Fixed-size arrays** (Sounio stdlib convention, no dynamic vectors of paths).
- **Small graphs** (max 16 states, max 64 edges, max 256 enumerated paths).
- **Source-node quadrature** for `∫_γ s dℓ` (same as Python contract).
- **Anti-Goodhart constraint** enforced by target check.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **S1_MODULE_TYPECHECKS** | `stdlib/clinical/mercyful.sio` type-checks with `bin/souc check`. | `souc check` passes. |
| **S2_BENCHMARK_RUNS** | `tests/run-pass/mercyful_exposure_therapy.sio` executes and prints `MERCYFUL_SOUNIO_PASS`. | `souc run` succeeds. |
| **S3_ANTI_GOODHART** | The benchmark shows raw minimizer avoids recovery while mercyful scheduler reaches recovery. | Output contains both assertions. |
| **S4_PARETO_FRONTIER** | The benchmark computes at least two non-dominated frontier points. | Output contains frontier list. |
| **S5_NO_CLINICAL_CLAIM** | The module and test carry explicit "no clinical claim" warnings. | Text present in both files. |

---

## 4. What this is NOT

- **Not a clinical recommendation.** The exposure-therapy graph is synthetic.
- **Not a learned model.** The scheduler is combinatorial.
- **Not a substrate energy model.** Edge length is step count.

---

## 5. Reproduce

```bash
bin/souc check stdlib/clinical/mercyful.sio
bin/souc run tests/run-pass/mercyful_exposure_therapy.sio
bash scripts/ci/mercyful_sounio_gate.sh
# expect: MERCYFUL_SOUNIO_GATE_OK
```

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical or patient-level claim. GAIDeT-ICMJE 2025.
