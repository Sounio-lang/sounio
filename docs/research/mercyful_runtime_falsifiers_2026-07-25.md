<!-- docs:meta
topic_id: repo.docs.research.mercyful-runtime-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-runtime-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning Runtime — falsifiers and stop rules

**Companion to:** `docs/research/mercyful_runtime_spec_2026-07-25.md`  
**Harness:** `scripts/research/mercyful_runtime_contract.py`

---

## Clause-level falsifiers

### M1_WELL_DEFINED

**Falsifier:** The scheduler crashes, returns `None`, or returns a path that is not a valid sequence of edges on the input graph.

**Stop rule:** Basic correctness of the search is broken.

---

### M2_PARETO_FRONTIER

**Falsifier:** For a small graph where the exact frontier is known by hand, the computed frontier is missing a point or includes a dominated point.

**Stop rule:** The bi-criteria optimizer is wrong.

---

### M3_ANTI_GOODHART

**Falsifier:** In the exposure-therapy benchmark, a raw-suffering minimizer that is **not** constrained to reach `recovery` nevertheless reaches `recovery`.

**Stop rule:** The toy model is too easy; it cannot demonstrate the Goodhart hazard.

---

### M4_MERCYFUL_SELECTS_EXPOSURE

**Falsifier:** For `μ` in the tested range, the mercyful scheduler selects a path that avoids `moderate_distress` and never reaches `recovery`.

**Stop rule:** The scheduler fails to encode the target constraint or the peak-vs-integrated trade-off.

---

### M5_MU_CONTINUITY

**Falsifier:** Increasing `μ` produces a selected path with strictly higher peak suffering in at least one step.

**Stop rule:** The peak-aversion parameter is not wired into the decision rule correctly.

---

### M6_BUDGET_INFEASIBILITY

**Falsifier:** When no path reaches `target` within `L0`, the scheduler returns a non-target path instead of `INFEASIBLE`.

**Stop rule:** The budget constraint is not hard.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| M1 or M6 fails | `M_RED` | Core scheduler is unsafe to use. |
| M3 fails | `M_RED` | Benchmark does not illustrate the intended phenomenon. |
| M2, M4, or M5 fails | `M_AMBER` | Fix the specific clause before claiming. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). No clinical claims. GAIDeT-ICMJE 2025.
