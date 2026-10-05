<!-- docs:meta
topic_id: repo.docs.research.mercyful-scheduler-lean-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-scheduler-lean-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning scheduler — Lean 4 correctness formalization (Task 3)

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Lean file:** `formal/lean4/SounioMercyfulScheduler.lean` (`@[default_target]`
in `formal/lean4/lakefile.lean` — built by the CI `lean-proofs` job)
**Gate:** `scripts/ci/mercyful_lean_gate.sh` (**MERCYFUL_LEAN_GATE_OK**)
**Companions:** `stdlib/clinical/mercyful.sio` (runtime),
`scripts/research/mercyful_runtime_contract.py` (M1..M6),
`scripts/research/mercyful_mimic_iv_vancomycin_contract.py` (V1..V7),
`docs/research/mimic_iv_mercyful_validation_2026-07-26.md` (POSITIVE verdict)

**Verdict: GREEN** — the three requested claims are proved in Lean 4
(toolchain `leanprover/lean4:stable` = 4.32.1), Mathlib-free, no `sorry`:

1. the Mercyful scheduler always selects a target-reaching (therapeutic
   window) course when one exists and is feasible;
2. the naive toxicity minimizer always selects the lowest-toxicity arm
   regardless of the target, and on the MIMIC-IV synthetic graph that arm
   is the sub-therapeutic `FIXED_LOW` that cannot reach `TARGET`;
3. the anti-Goodhart (target-reaching) constraint is necessary and
   sufficient to prevent under-dosing.

> **Scope.** This formalizes a synthetic graph scheduler for research
> infrastructure. The graph, doses, p-boxes, and suffering values are
> synthetic constructions. This is not medical guidance, not a treatment
> recommendation, not a dosing suggestion, and not a clinical
> decision-support tool.

---

## 1. What is proved, precisely

All abstract theorems quantify over an arbitrary candidate-path list
`cands : List Path`, an arbitrary graph `g : MercyGraph`, and arbitrary
parameters `sp : SchedParams` — they are not tied to the vancomycin
instance. Costs are exact `Rat`.

| # | Theorem (namespace `Sounio.Mercyful`) | Claim |
|---|---|---|
| T1 | `mercyful_feasible_selection` | The scheduler's pick is feasible and cost-minimal among all feasible candidates |
| T2 | `mercyful_reaches_target` | **Sufficiency:** the pick always reaches the target (no under-dosing) |
| T2′ | `mercyful_selects_therapeutic_window` | If a feasible path exists, the scheduler returns one, reaching the target, at minimal feasible cost |
| T3 | `naive_minimizer_optimal` | The naive pick attains the minimum toxicity over all arms |
| T4 | `goodhart_trap` | **Necessity:** without the target constraint, if `[start]` is a zero-cost candidate and every target-reaching candidate costs > 0, the unconstrained pick never reaches the target |
| T4′ | `anti_goodhart_necessary_and_sufficient` | T2 ∧ T4 packed as a single iff-shaped statement |

Concrete MIMIC-IV instance theorems (namespace `Sounio.Mercyful.VancoMimicIV`,
all by `native_decide`, mirroring the runtime contract clauses):

| # | Theorem | Runtime clause it certifies |
|---|---|---|
| C1 | `vanco_feasible_unique` | the TDM-guided route is the UNIQUE feasible course (V5 uniqueness) |
| C2 | `vanco_mercyful_selects_tdm` | scheduler selects `START→VANCO_PRE→TDM_GUIDED→TARGET` (V5) |
| C3 | `vanco_tdm_route_cost` | exact costs ∫s = 735099/10⁶, peak = 675679/10⁶, total = 1410778/10⁶ at μ = 1 (V5 canonical numbers, clinical twin C1) |
| C4 | `vanco_naive_underdoses` | naive pick = `FIXED_LOW` (toxicity 0) with NO enumerated path to `TARGET` (V1) |
| C5 | `vanco_unconstrained_traps` | unconstrained minimizer selects `[START]`, cost 0 — never treats (V2) |
| C6 | `vanco_gate_is_causal` | with the gate opened, the same scheduler switches to the non-TDM arm `FIXED_STD` (V4 counterfactual) |

## 2. Modeling conventions (faithfulness to the runtime)

The Lean cost model mirrors `MercyGraph.path_cost` in
`mercyful_runtime_contract.py` exactly:

- **Unit edge lengths** (all lengths are 1.0 in the MIMIC-IV contract);
  the budget `L0` bounds the number of traversed edges (`p.length ≤ L0 + 1`).
- **Integral suffering** charges the *source* state of each traversed
  edge; the **peak** additionally ranges over the final state.
- **Total cost** `= integral + μ · peak`, argmin with strict-improvement
  update (`total < best_cost`), so ties keep the first-enumerated path —
  matching the Python loop.
- The **anti-Goodhart constraint** is the `reachesTarget` clause of
  `FeasibleB`; the unconstrained scheduler is the same argmin without it.
- The **naive toxicity minimizer** is argmin of a toxicity-only metric
  over arms. Target-blindness is *by construction*: `naiveToxPick`'s
  type has no target parameter, so no proof about it can secretly depend
  on the target — the strongest faithful reading of "regardless of
  target" in a total functional language.
- Suffering values are exact rationals: 3/5 (FIXED_LOW), 7/10
  (FIXED_STD), 675679/1000000 (VANCO_PRE), 59420/1000000 (TDM_GUIDED) —
  the contract's exact printed values, no float rounding.

**Necessity reading.** "Necessary to prevent under-dosing" is formalized
in the standard contrapositive-with-witness form: *removing* the
constraint admits under-dosing optima whenever the trap conditions hold
— relative to any candidate list containing the zero-cost start path
(`[start]` a candidate; every target-reaching candidate in that list
strictly positive) — proved abstractly (`goodhart_trap`) and witnessed
concretely (`vanco_unconstrained_traps`). C6 strengthens this: the
constraint is not merely decorative, it is exactly what makes the TDM
route optimal. The concrete suffering table is recorded as nonnegative
(`vanco_suffering_nonneg`), discharging the abstract theorems'
hypothesis for this instance; the strict-`<` tie-breaking policy shared
with the Python loop is recorded as `argminAux_tie_keeps`.

## 3. Proof architecture

- `argminAux`/`argmin` over `List α` with cost `α → Rat`; key lemmas
  `argmin_some_mem` (pick ∈ list) and `argmin_some_min` (pick attains the
  minimum), by induction with the strict-improvement case split.
- T1/T2/T2′ are corollaries via `List.mem_filter`; T4 uses
  `pathCost_singleton_zero` (the trivial course costs exactly 0),
  `pathCost_nonneg`, and `Rat.le_antisymm`/`Rat.lt_irrefl`.
- Concrete theorems evaluate the bounded simple-path enumerator
  `pathsFrom` (fuel 8 > 6 states) under `native_decide`.

**Axiom footprint** (`#print axioms`, verified 2026-07-26):

- Abstract theorems (T1–T4′): `[propext, Classical.choice, Quot.sound]`
  only — the same set the CI `lean-proofs` axiom check accepts.
- Concrete theorems (C1–C6): the above plus the per-theorem
  `native_decide` trust axiom (`..._native.native_decide.ax_1_1`),
  standard for computational certificates in this repository.

## 4. Scoped out (explicit)

1. **Enumerator completeness.** `pathsFrom g fuel start` with
   `fuel ≥ stateCount` enumerates every simple path; this is argued
   (simple paths visit each of 6 states at most once; fuel 8 > 6) but
   not mechanized. The concrete theorems are therefore statements about
   the enumerated candidate set — exactly the set the runtime scheduler
   itself optimizes over. Mechanizing completeness is future work.
2. **The Sounio runtime's BFS implementation** (`stdlib/clinical/
   mercyful.sio`) is modeled, not verified: the Lean scheduler is an
   abstract argmin, not a refinement proof of the BFS queue code.
3. **μ-crossover / Pareto frontier** theorems (runtime M2/M5) — not
   requested by Task 3.
4. **Real-valued (ℝ) costs** — `Rat` suffices for the exact contract
   values; a Float/ℝ lift would follow the Track-2/3 pattern of
   `SounioFrechetRat`/`SounioRealOrder`.
5. `topic-registry.v1.json` registration — left to the integrator to
   avoid conflicting with parallel governance lanes.

## 5. Falsifiers

| Clause | Falsifier |
|---|---|
| T1–T4′ | Any abstract theorem fails to compile, or gains an axiom outside `[propext, Classical.choice, Quot.sound]` |
| C1 | A second feasible course appears in the gated graph (uniqueness breaks) |
| C2/C3 | Selected path or any canonical number deviates from the runtime contract's printed values |
| C4 | The naive pick reaches `TARGET`, or is not `FIXED_LOW` |
| C5 | The unconstrained pick reaches `TARGET` (trap gone) |
| C6 | The open-gate optimum is not the non-TDM arm (gate decorative) |

Gate failure classification: build/bootstrap-path (lake/lean missing or
broken), harness-routing (gate script paths), ontology-kernel/checker
(n/a here), baseline noise (n/a). Any RED: `lake build
SounioMercyfulScheduler` fails, or a theorem statement is weakened, or
`sorry`/`admit`/`axiom` appears in the file.

## 6. Commands run

```bash
cd formal/lean4 && lake build SounioMercyfulScheduler   # green (Lean 4.32.1)
lake env lean /tmp/mercy_axioms.lean                    # axiom footprints as in §3
bash scripts/ci/mercyful_lean_gate.sh                   # MERCYFUL_LEAN_GATE_OK
```

## 7. LLM-offload review

Mandatory math-review offload (`bin/llm-offload -t math-review`, dual
xai/Grok 4.3 + zai/GLM-5.2 per M1 policy) run on this spec + the Lean
file; clinical/devil's-advocate review (`-t review`) run with xai/Grok
4.3 after DeepSeek returned `Insufficient Balance` (failure-mode rule).
Outcome: **PASS / ADDRESSED** —

- Grok math leg: `[OK]` on every abstract theorem and C1–C6; "no
  mathematical errors or leaps found." Z.AI leg independently re-derived
  every definition and proof line-by-line (output truncated at token cap
  mid re-check, zero `[WRONG]` flags).
- Grok review leg: 2 MAJOR + 2 MINOR + 1 NIT, all addressed in place:
  (1) abstract-model-vs-BFS boundary now stated in the Lean module doc
  and here (§4.2, pre-existing); (2) C1 uniqueness qualified to the
  enumerated candidate set with the fuel argument referenced; (3)
  tie-breaking policy recorded as the lemma `argminAux_tie_keeps`;
  (4) necessity reading qualified to "any candidate list containing the
  zero-cost start path" (§2); (5) concrete non-negativity instance check
  added (`vanco_suffering_nonneg`). Gate re-run green after the fixes.
- Full entries in `.claude/llm_offload_log.md` (2026-07-26 rows).
