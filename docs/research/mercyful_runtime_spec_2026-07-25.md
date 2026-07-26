<!-- docs:meta
topic_id: repo.docs.research.mercyful-runtime-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-runtime-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning Runtime — first substrate-aware suffering-budget scheduler

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parent:** `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §A  
**Harness:** `scripts/research/mercyful_runtime_contract.py`  
**Gate:** `scripts/ci/mercyful_runtime_gate.sh`

---

## 1. What this is

Mercyful Learning has a defensible formal core (§A of the registry) but no executable implementation. This document specifies the first runtime prototype: a graph-based scheduler that, given a **suffering field** over states and a **length budget**, computes the Pareto frontier between integrated suffering and peak suffering, subject to reaching a target state.

The runtime explicitly does **not** minimize raw suffering; that would be Goodhart-vulnerable. It minimizes **necessary** suffering subject to the target constraint.

---

## 2. Formal model

### Input

- Finite directed graph `G = (V, E)` with edge lengths `ℓ(e) > 0`.
- Suffering field `s : V → ℝ≥0`.
- Start state `start ∈ V` and target state `target ∈ V`.
- Peak-aversion parameter `μ ≥ 0`.
- Length budget `L0 > 0`.

### Path cost

For a path `γ` from `start` to `target` with `len(γ) = Σ ℓ(e) ≤ L0`, the integral is discretised by assigning each edge segment the suffering at its source node:

```
∫_γ s dℓ = Σ_{(u,v) ∈ γ} s(u) · ℓ((u,v))
```

The total cost is then:

```
cost(γ; μ) = ∫_γ s dℓ + μ · max_{v ∈ γ} s(v)
```

### Decision problem

```
γ* = argmin_{γ : start → target, len(γ) ≤ L0} cost(γ; μ)
```

The Pareto frontier is computed exactly for small graphs by exhaustive enumeration of simple paths; this is not a claim of a polynomial-time algorithm.

### Anti-Goodhart constraint

A path that does not reach `target` is **infeasible**, regardless of cost. A path that reaches `target` is feasible.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **M1_WELL_DEFINED** | For every finite input graph, the scheduler returns a feasible path or reports `INFEASIBLE`. | Returns path or `INFEASIBLE` on test graphs. |
| **M2_PARETO_FRONTIER** | The frontier `{(∫s, max s)}` is computed exactly for small graphs. | Enumerated frontier matches expected trade-offs. |
| **M3_ANTI_GOODHART** | Minimizing raw suffering without the target constraint produces a path that stays in low-suffering states and misses the target. | Exposure-therapy benchmark shows naive scheduler avoids recovery. |
| **M4_MERCYFUL_SELECTS_EXPOSURE** | With `μ` in a tested range, the mercyful scheduler selects a path that passes through distress and reaches recovery. | Exposure-therapy benchmark shows recovery path. |
| **M5_MU_CONTINUITY** | As `μ` increases, the selected path's peak suffering weakly decreases (peak-aversion is effective). | Frontier sweep shows monotone trend. |
| **M6_BUDGET_INFEASIBILITY** | If no path reaches `target` within `L0`, the scheduler reports `INFEASIBLE`, not a cheaper non-target path. | Tight budget test reports infeasibility. |

---

## 4. What this is NOT

- **Not a clinical recommendation.** The exposure-therapy graph is a synthetic toy model.
- **Not a learned model.** The scheduler is combinatorial; learning is future work.
- **Not a substrate energy model.** Edge length is interpreted as step count / compute steps; energy coupling is future work.
- **Not a proof of the Mercyful Learning principle.** It is an executable illustration on a controlled graph.

---

## 5. Reproduce

```bash
python3 scripts/research/mercyful_runtime_contract.py
# expect: M1..M6 PASS, MERCYFUL_RUNTIME_VERDICT M_GREEN

bash scripts/ci/mercyful_runtime_gate.sh
# expect: MERCYFUL_RUNTIME_GATE_OK
```

Pure Python.

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical or patient-level claim. GAIDeT-ICMJE 2025.
