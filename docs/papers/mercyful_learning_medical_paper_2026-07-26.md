<!-- docs:meta
topic_id: repo.docs.papers.mercyful-learning-medical-paper-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.mercyful-learning-medical-paper-2026-07-26
-->

# Mercyful Learning: Suffering as a First-Class Cost in Computational Treatment Sequencing

**A formal framework with synthetic demonstrations in exposure therapy, chemotherapy scheduling, and pharmacokinetic co-medication sequencing**

**Author:** Demetrios Chiuratto Agourakis
**Date:** 2026-07-26
**Status:** Full manuscript draft — target journals: *Clinical Pharmacology & Therapeutics* or *Journal of Psychiatric Research*
**Provenance:** Formal core: `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §A. Runtimes: `stdlib/clinical/mercyful.sio`, `scripts/research/mercyful_runtime_contract.py`. Benchmarks: `tests/run-pass/mercyful_exposure_therapy.sio`, `tests/run-pass/mercyful_clinical_sequencing.sio`, `tests/run-pass/mercyful_chemo_sequencing.sio`, `scripts/research/mercyful_chemo_contract.py`, `scripts/research/mercyful_mimic_iv_vancomycin_contract.py`. Real-data correspondence analysis: `docs/research/mimic_iv_mercyful_validation_2026-07-26.md`. Companion preprint: `docs/papers/mercyful_learning_preprint_2026-07-26.md`.

> **Scope statement (read first).** This paper formalizes a decision rule for treatment sequencing under a *suffering field* and demonstrates it on synthetic graphs. All patients, doses, regimens, therapeutic windows, drug–drug-interaction flags, toxicity grades, and suffering values in this paper are synthetic constructions. Cited clinical trials and guidelines shape the *structure* of the synthetic benchmarks only. Nothing here is medical guidance, a treatment recommendation, a dosing suggestion, or a clinical decision-support tool. No patient data were used; no clinical validation is claimed. The contribution is a formal framework, exact executable benchmarks, and pre-registered falsifiers.

---

## Abstract

**Background.** Clinical schedulers — dynamic treatment regimes, SMART designs, and reinforcement-learning therapy sequencers — almost universally optimize an *expected outcome*: they sum. Summation hides the patient who traverses a catastrophic peak while the mean improves, and any scheduler that minimizes a measured suffering metric without a hard therapeutic-target constraint is Goodhart-vulnerable by construction: it will rediscover avoidance, under-treatment, and sedation as optima.

**Principle.** We introduce **Mercyful Learning**, a formal framework in which suffering is a first-class cost field over a finite state graph, and treatment sequencing is the problem of reaching a therapeutic target while minimizing a budget-constrained combination of *integrated* suffering and *peak* suffering. The choice of functional is the ethical commitment, made explicit and computable; necessity of suffering is a *budgetary* notion — the minimal peak attainable within a declared resource budget; and the **anti-Goodhart axiom** makes any plan that fails to reach the target infeasible regardless of cost.

**Applications and results.** Three synthetic benchmarks, each with an executable, dual-implementation contract. (1) **Exposure therapy:** a naive raw-suffering minimizer prescribes avoidance (cost 0); the mercyful scheduler traverses distress and reaches recovery at integrated suffering 7, peak 5 — matching the structure of the clinical fact that exposure-based therapies require acute distress to work. (2) **Chemotherapy sequencing:** on a graph shaped after the relative-dose-intensity, dose-dense, and stop-and-go literature, dose reduction to minimize toxicity rediscovers under-treatment (Bonadonna's RDI < 85% hazard) as an unconstrained optimum; the mercyful scheduler reproduces Pareto dominance of continuous-until-progression (OPTIMOX structure), a G-CSF edge-admission gate (EORTC/NCCN structure), and an exact peak-aversion crossover price μ\* = 11. (3) **Pharmacokinetic co-medication sequencing:** suffering fields are derived from Knightian probability-box bands of synthetic vancomycin and tacrolimus regimens; therapeutic drug monitoring strictly lowers the field; drug–drug-interaction gates (nephrotoxin co-medication, CYP3A4 inhibition) act as Goodhart constraints on edge admission, including a case where an interaction doubles an AUC enclosure across a ceiling and the route becomes *infeasible* rather than silently relaxed.

**Boundaries.** Everything reported is synthetic, small-graph, and combinatorial. No learning is performed; no patient data are used; no clinical or patient-level conclusion is drawn. A structural-correspondence section compares the *shape* of the benchmarks' optima and infeasibilities with documented findings in exposure therapy, chemotherapy dose intensity, and — in the domain the suffering field is derived for — a 28,451-patient MIMIC-IV vancomycin cohort in which therapeutic drug monitoring was associated with lower mortality (post-matching OR 0.67–0.69, CIs excluding 1.0); the model predicts decision-structure direction only, never effect sizes. All claims are separated into what is proven, what is measured, and what is conjectural, with pre-registered stop rules.

**Conclusions.** Suffering measured without a target constraint optimizes the measure; the repair belongs in the feasible set. The framework makes the aggregation-versus-peak ethical commitment a declared, computable parameter, and prices necessary versus gratuitous suffering as theorems of the graph.

**Keywords:** suffering budget; treatment sequencing; exposure therapy; chemotherapy dose intensity; Goodhart's law; constrained MDP; risk-sensitive reinforcement learning; Knightian uncertainty; probability boxes; therapeutic drug monitoring; drug–drug interactions; computational pharmacology; computational psychiatry.

---

## 1. Introduction

### 1.1 The Goodhart problem in clinical scheduling

Goodhart's law — when a measure becomes a target, it ceases to be a good measure — has a clinical name [1, 2]. An algorithm that minimizes *measured* suffering optimizes the measure. In psychiatry the resulting optima are familiar: sedation, affective flattening, iatrogenic anhedonia — all of which reduce distress scores, and all of which a naive system would rediscover as optimal policies. In oncology the same failure has a population-scale signature: toxicity-minimizing behavior — dose reductions and delays — is routine, and its cost is measured in survival. In Bonadonna and Valagussa's 20-year follow-up of adjuvant CMF for node-positive breast cancer, the benefit of chemotherapy concentrated in patients receiving ≥85% of planned dose, and patients below ~65% relative dose intensity (RDI) had survival approximating the untreated control arm [19]. A scheduler that minimizes measured toxicity *rediscovers under-treatment as an optimum*.

A second, subtler failure mode is *outcome aggregation*. Dynamic treatment regimes [3], sequential multiple assignment randomized trials (SMART) [4], and reinforcement-learning approaches to therapy sequencing optimize expected cumulative reward. Expected values are sums, and sums are indifferent to *where* the cost concentrates. A policy that drives one patient through an acute crisis while the population mean improves can dominate, on expectation, a policy that never lets any patient cross a catastrophic threshold.

This paper treats both failures as one: a missing constraint and a missing functional.

### 1.2 Suffering as a first-class cost

We model a patient's clinical course as a path γ on a finite directed graph whose vertices carry a *suffering field* s : V → ℝ≥0. Two cost functionals over paths are natural and genuinely different:

- **Integrated suffering** ∫γ s dℓ — the total burden accumulated along the course;
- **Peak suffering** max_{v∈γ} s(v) — the worst moment endured.

Minimizing the integral is *aggregationism*: it permits buying an acute peak with enough tranquil trajectory. Minimizing the maximum is *maximin*. Over the same field the two functionals select different paths. The structure of the problem therefore does not determine the ethics; it makes the ethical choice explicit and computable. We take the combination

> cost(γ; μ) = ∫γ s dℓ + μ · max_{v∈γ} s(v),

where μ ≥ 0 is a declared peak-aversion parameter. For any two competing paths, the crossover value μ\* at which the peak-averse choice wins is exactly computable, so the ethical weighting becomes a number one must defend rather than an assumption one can hide.

### 1.3 Three clinical domains where the structure is documented, not hypothetical

The framework is presented through three synthetic applications, chosen because each has all four structural elements of the framework — a documented Goodhart hazard, a peak-versus-integral trade-off, a real budget, and (in two of the three) a topological gate — already documented in the clinical literature:

1. **Exposure therapy for anxiety disorders and PTSD.** The treatment that works requires acute distress; the behavior that minimizes acute distress is avoidance — which is the disorder. Exposure-based cognitive behavioral therapies are first-line, strongly recommended treatments for PTSD and anxiety disorders [15–18]. A naive minimizer of ∫(suffering) prescribes avoidance: it recommends the pathology, with the appearance of compassion.
2. **Cancer chemotherapy sequencing.** Toxicity-driven under-dosing is the documented Goodhart hazard (RDI < 85%) [19–21]; the dose-dense/standard peak–integral trade-off is a named, trialled clinical decision (CALGB 9741 [22]); stop-and-go scheduling (OPTIMOX1 [23]) and the anthracycline lifetime cumulative-dose ceiling [24] are real budget structures; and the G-CSF prophylaxis requirement for regimens with febrile-neutropenia risk ≥ 20% [25, 26] is a real edge-admission gate.
3. **Pharmacokinetic/drug–drug-interaction sequencing.** Therapeutic drug monitoring (TDM) narrows uncertainty bands [13]; drug–drug interactions (nephrotoxic co-medication with vancomycin; CYP3A4 inhibition of tacrolimus metabolism) constrain admissible co-medication sequences [13, 14]. DDI gates act directly on which clinical transitions are *permissible*, not merely on their cost.

### 1.4 Relation to prior work

The learner inside the moral domain — optimizing for the suffering of the system it acts upon — has partial precedents we position against honestly:

- **Constrained MDPs (CMDP)** [5] optimize one expected cost subject to constraints on others; our target constraint is a hard feasibility constraint, and our peak term is a path functional, not an expectation.
- **Risk-sensitive RL / CVaR** [6, 7] replaces expectation by coherent risk measures over returns; CVaR is a tail functional of a distribution, whereas the peak term here is a per-trajectory worst case — closer in spirit to minimax-reward formulations but combined with aggregation rather than replacing it.
- **Quantilizers** [8] avoid Goodhart collapse by sampling from an acceptable quantile of a base distribution instead of maximizing; our anti-Goodhart device is complementary: a hard target constraint inside the objective's feasible set.
- **Attainable utility preservation (AUP)** [9] and **relative reachability** [10] penalize side effects by penalizing loss of optionality — penalizing making states unreachable, which is nearly the same idea as our target constraint approached from the other side.
- **Active inference** [11] already markets itself as an alternative to reward-maximizing RL and dominates part of computational psychiatry; it minimizes expected free energy (an information-weighted expectation), not a suffering field with a peak term.
- **Dose-intensity and scheduling clinical literature** [12, 19–26] supplies the documented structure for the chemotherapy and pharmacology applications (see §1.3 and §6).
- **Constrained learning theory and fair classification.** Probably-approximately-correct constrained learning [32], its non-convex extension [33], proxy-Lagrangian optimization under non-differentiable constraints [34], and reductions approaches to fair classification [35] make constraints first-class training objects with guarantees; our target constraint is the clinical analogue of that commitment — feasibility defined by reaching the therapeutic state, not by bounding an auxiliary cost.
- **Reward hacking and Goodhart formalisms.** Reward misspecification and reward hacking are mapped, taxonomized, and formally characterized in [36–38]; the clinical Goodhart hazards of §1.1 are the same phenomenon in clinical dress.
- **Bottleneck/minimax shortest paths and goal-conditioned safe planning.** Our peak term is a classic bottleneck cost and our target constraint is standard goal reachability; the constrained shortest-path formulation over tiny graphs is elementary. We claim no new algorithmic primitive — see §9.6 for an explicit statement of where the contribution does and does not lie.

We also draw on imprecise probability for the pharmacology integration: suffering fields are computed from *probability boxes* (p-boxes) — Knightian outer enclosures valid for any joint distribution of pharmacokinetic parameters [27] — so that the scheduler prices worst-case band violations rather than point estimates.

### 1.5 Contributions

1. A formal framework (§3): suffering field, two functionals, the budgetary reformulation of necessary suffering, the μ-parametrized decision rule, and the anti-Goodhart axiom.
2. An exact combinatorial scheduler with Pareto-frontier extraction for small graphs (§4), implemented twice — pure Python and native Sounio — with executable contracts.
3. Three synthetic applications with exact, reproducible numbers: exposure therapy (§5), chemotherapy sequencing (§6), and PK/DDI co-medication sequencing (§7).
4. Pre-registered falsifiers and stop rules (§10), and an explicit ledger of what is proven, measured, and conjectural (§9.4).

---

## 2. Scope and safety boundaries

This work is deliberately bounded:

- **Synthetic data only.** The exposure-therapy patient, the chemotherapy patient, both PK patients (78.5 kg / CrCl 65 mL/min; 70 kg / CrCl 80 mL/min), all doses, all regimens, all toxicity grades, all therapeutic windows, all DDI flags, and all suffering values are synthetic constructions chosen to exercise the framework.
- **No clinical recommendation.** Nothing in this paper is a dosing recommendation, a treatment guideline, or a decision-support output. The therapeutic windows used (e.g., a vancomycin trough screen of [10, 20] mg/L after Rybak et al. [13]; a tacrolimus trough window of [5, 15] ng/mL after the Prograf label [14]) shape the synthetic suffering field; they are not clinical target claims. The toxicity ordinals in §6 (≈1–2 low-grade, ≈5 grade-3-level, ≈8 grade-4/febrile-neutropenia-level burden) are synthetic anchors.
- **No learning.** The scheduler is an exact combinatorial search over small graphs. Learning under a suffering field is future work; §9.5 discusses why several natural attempts are expected to fail.
- **No sentience claim.** The framework is agnostic on machine sentience; the "two mercies" structure (§9.2) prices a computational budget, not a phenomenological one.
- **Omission of disease-attributable suffering.** In the chemotherapy application the field models *treatment-attributable* suffering only. This omission is deliberate and is itself the demonstration: a raw minimizer of measured suffering sees the untreated state as costless precisely because the measure is blind to the disease; the anti-Goodhart constraint re-introduces the disease outcome as a feasibility condition (§6.2).

---

## 3. Formal framework

### 3.1 The suffering field

Let G = (V, E) be a finite directed graph with edge lengths ℓ : E → ℝ>0, a suffering field s : V → ℝ≥0, a start state a ∈ V, a target state b ∈ V, and a length budget L₀ > 0. Paths are simple (no repeated vertex) unless stated otherwise; length is len(γ) = Σ_{e∈γ} ℓ(e).

The integrated suffering is discretized by assigning each edge segment the suffering at its source vertex (left-endpoint quadrature):

> ∫γ s dℓ := Σ_{(u,v)∈γ} s(u)·ℓ(u,v),  max_γ s := max_{v∈γ} s(v).

### 3.2 The two budgetary functionals

For each budget L₀, define

> Ψ(L₀) = inf { ∫γ s dℓ : γ a path a→b, len(γ) ≤ L₀ },
> c\*_orc(L₀) = inf { max_γ s : γ a path a→b, len(γ) ≤ L₀ }.

Both are well-defined whenever a feasible path exists (infimum over a non-empty finite set of simple paths). These are the *budgetary* forms of necessity: even when every sublevel set {s ≤ c} is connected — so that no *topological* mountain pass exists — the least-suffering route may still be long, and within a realistic budget the attainable peak may sit far above the endpoints' suffering.

**Necessary vs. gratuitous suffering.** We define

> **gratuitous(γ) = max_γ s − c\*_orc(L₀),   mercy = attaining c\*_orc(L₀).**

The budget L₀ must be fixed by someone. This is a real loss relative to a topological definition, and we state it rather than hide it: "necessary" stops being a geometric fact and becomes relative to a declared resource. The trade is that the budgetary form makes a *quantitative* prediction (a length–peak exchange curve) where the topological form made only a binary one. (The topological form was tested and falsified twice in the parent research program — once by a connectivity theorem and once on real semantic fields — before this budgetary form, which never depended on disconnection, was adopted [28].) In clinical terms: nobody claims exposure therapy is the only conceivable route to recovery; the claim is that it is the only route that works within human time and resource constraints. That is a budget constraint, not a topological obstruction — and it is what the formalism says.

### 3.3 The decision rule

For peak-aversion parameter μ ≥ 0:

> γ\*(μ, L₀) = argmin { ∫γ s dℓ + μ·max_γ s : γ a path a→b, len(γ) ≤ L₀ }.

For any two paths γ₁, γ₂ with (∫₁, max₁), (∫₂, max₂), the crossover is μ\* = (∫₁ − ∫₂)/(max₂ − max₁) whenever the denominator is positive: the exact price at which aggregationism must yield to peak aversion. A revealed-preference reading: choosing the aggregative path over the peak-averse one is equivalent to asserting that one unit of peak suffering is worth less than 1/μ\* units of accumulated suffering — a computable threshold to defend, not a moral appeal.

### 3.4 The anti-Goodhart axiom

**Axiom (target feasibility).** *A path that does not reach the target b is infeasible, regardless of cost.*

This is an axiom of the method, not a caveat appended to it. The objective is not min(suffering) but min(gratuitous suffering) **subject to reaching the therapeutic state**. The target constraint is what blocks the sedative solution: sedation, avoidance, and iatrogenic blunting all lower measured suffering, and all are infeasible under the axiom because they do not reach b. In the framework's terms, Goodhart's law is not a pathology of the optimizer but of the objective's feasible set; the repair is to put the clinical endpoint inside the constraint.

### 3.5 Gates as Goodhart constraints

Edge admission can itself be gated: an edge (u, v) is admitted only if an external predicate g(u, v), evaluated at graph-construction time on recomputed quantities, holds. In the pharmacology integration (§7), transitions into the verified-therapy target exist only out of post-TDM states whose pharmacokinetic safety gate passes on the current concentration band; in the chemotherapy application (§6), the dose-dense route exists only when mandated supportive care is present. Gates are the structural twin of the anti-Goodhart axiom: they make certain low-suffering shortcuts — skipping verification, co-administering a contraindicated drug, scheduling a high-febrile-neutropenia-risk regimen without prophylaxis — *topologically absent* rather than merely expensive.

---

## 4. Algorithm

For small graphs the scheduler is exact by construction: exhaustive enumeration of simple paths via breadth-first search, with per-path cost evaluation and Pareto-frontier extraction.

**Enumeration.** A BFS queue stores partial paths (vertex, path, length). A partial path is extended along every outgoing edge whose head is not already in the path (simplicity) and whose edge predicate is admitted. Completed paths reaching b within budget are scored. The algorithm returns the argmin under cost(·; μ) or reports INFEASIBLE.

**Pareto frontier.** Over all feasible paths, the frontier {(∫γ s dℓ, max_γ s)} is extracted by removing dominated points: (i₁, p₁) dominates (i₂, p₂) iff i₁ ≤ i₂ ∧ p₁ ≤ p₂ with at least one strict inequality.

**Complexity honesty.** The number of simple paths grows factorially in the worst case; the fixed-size Sounio implementation caps at 16 states, 64 edges, 256 queued paths, and path length 32. This is an exact reference algorithm for benchmark-scale graphs, not a claim of a polynomial-time method. Scaling is future work (§9.5).

**Dual implementation.** The same scheduler exists as (i) a pure-Python contract (`scripts/research/mercyful_runtime_contract.py`, `scripts/research/mercyful_chemo_contract.py`; standard library only) and (ii) a native Sounio module (`stdlib/clinical/mercyful.sio`) executed through the Sounio compiler. Both implementations agree on all benchmark numbers reported below (contract clauses H8 and M-green agreement gates); the Sounio port uses a flat path-queue layout to work around a bootstrap-compiler array-lowering defect, a detail documented in the module header.

---

## 5. Application 1: Exposure therapy for anxiety and PTSD

### 5.1 Clinical motivation

Exposure-based cognitive behavioral therapy — the graded, deliberate confrontation with feared stimuli — is among the most effective evidence-based treatments for anxiety disorders, obsessive-compulsive disorder, and post-traumatic stress disorder [15–18]. Its mechanism, in emotional-processing terms, requires the *activation* of the fear structure and the incorporation of corrective information during that activation [15]: the therapeutic state lies on the far side of acute distress. Avoidance is the maintaining behavior of the disorder — the behavior that minimizes acute distress — and its elimination is the treatment goal. This is the case that makes the framework's definition clinically worth having: the formal distinction it must capture is between *palliating* and *treating* — between avoiding necessary suffering and passing through it at its minimal feasible level.

### 5.2 The Goodhart hazard: the naive minimizer prescribes the pathology

The synthetic benchmark (Fig. 1, Table 1) is a four-state graph with unit edge lengths:

| State | Suffering s | Interpretation |
|---|---|---|
| `avoidance` | 0 | Untreated; the Goodhart trap |
| `mild` | 2 | Mild exposure distress |
| `moderate` | 5 | Moderate exposure distress |
| `recovery` | 0 | Therapeutic target |

Edges: `avoidance→avoidance` (self-loop: staying untreated), `avoidance↔mild`, `mild↔moderate`, `moderate→recovery`. The only route to recovery passes through moderate distress.

An unconstrained raw-suffering minimizer (cycles allowed) achieves cost **0** by looping `avoidance→avoidance` forever. It never reaches recovery. This is the anti-Goodhart demonstration: *minimizing measured suffering, unconstrained, prescribes avoidance — the disorder itself.* Contract clause M3 verifies that the best target-constrained path has integrated suffering 7 > 0, so the unconstrained optimum is strictly cheaper than any feasible plan.

### 5.3 The mercyful scheduler traverses distress and reaches recovery

Under the target constraint, the unique simple path to recovery is γ = avoidance → mild → moderate → recovery, with exact metrics:

> len = 3,  ∫γ s dℓ = 0 + 2 + 5 = 7,  max_γ s = 5,  cost(γ; μ) = 7 + 5μ.

At μ = 1: total cost **12** (clause M4). The budgetary functionals are

> Ψ(L₀) = infeasible for L₀ < 3, else 7;  c\*_orc(L₀) = infeasible for L₀ < 3, else 5,

so the chosen path has gratuitous suffering 5 − 5 = **0**: it achieves mercy exactly. The distress it traverses is *necessary* in the budgetary sense — no path within any budget reaches recovery without a peak of 5 — and the framework says so as a theorem of the graph, not as a clinical intuition.

### 5.4 The functional choice is computable, not rhetorical

A second synthetic graph (clause M2) exhibits a genuine trade-off: path P₁ = S→A→T has (∫s, max s) = (5, 5) and length 2; path P₂ = S→B→C→D→T has (6, 2) and length 4. Neither dominates the other; the Pareto frontier is exactly {(5,5), (6,2)}. The crossover is μ\* = (6−5)/(5−2) = **1/3**: below μ = 1/3 the scheduler aggregates, above it pays the longer, gentler route. A third graph (clause M5) has (4,4) vs (6,3), crossover μ\* = 2; at μ = 0 the scheduler selects the high-peak path and at μ = 10 the low-peak path — demonstrating that μ is wired into the decision rule and that selected peak weakly decreases with μ.

Read clinically: graded-exposure design is precisely a peak-versus-integral trade-off. A steeper hierarchy reaches remission faster at a higher per-session distress peak; a shallower hierarchy costs more cumulative exposure burden to cap the peak. The framework does not resolve this trade-off — it makes the resolving parameter (μ) a declared quantity with a computable crossover price.

### 5.5 Budget hardness

On a graph with len = 4 and L₀ = 3, the scheduler returns INFEASIBLE — not a cheaper non-target path (clause M6). The budget constraint and the target constraint are both hard. Budget hardness is what makes the framework's reading of the clinical claim ("exposure is the route that works in human time") a formal statement rather than an interpretation.

---

## 6. Application 2: Cancer chemotherapy sequencing

### 6.1 Clinical motivation

Of the candidate domains, chemotherapy sequencing is the one where *all four* structural elements of the framework have direct, documented clinical counterparts:

1. **The Goodhart hazard is documented at population scale.** Toxicity-driven dose reductions and delays are routine and their cost is measured: the benefit of adjuvant CMF concentrated in patients receiving ≥85% of planned dose; RDI < 85% remains common in practice, driven predominantly by myelotoxicity [19–21]. The dose-intensity concept itself originates with Hryniuk and Levine [12].
2. **The peak-versus-integral trade-off is a named, trialled clinical decision.** Dose-dense scheduling (CALGB 9741: same drugs, compressed interval, G-CSF-supported) improved disease-free and overall survival at the price of higher acute toxicity per unit time [22]. Conversely, oxaliplatin stop-and-go (OPTIMOX1) reduced cumulative grade-3/4 neurotoxicity while maintaining efficacy [23]. These trials are literally experiments in the (integral, peak) plane the framework formalizes.
3. **The budget L₀ has real instantiations.** The neoadjuvant window before scheduled surgery is a declared time budget; the anthracycline lifetime cumulative-dose ceiling (~450–550 mg/m² doxorubicin, after Von Hoff et al. [24]) is a declared cumulative-toxicity budget.
4. **Topological gates exist in guidelines.** EORTC and NCCN guidance requires primary G-CSF prophylaxis when a regimen's febrile-neutropenia (FN) risk is ≥20% [25, 26]; high-FN-risk regimens are *not to be given without support*. This is an edge-admission gate.

### 6.2 The Goodhart problem in this domain

**Exact hazard.** minimize ∫(measured toxicity) over unconstrained clinical courses. Two structural optima exist, both pathological:

- **Hazard A — non-treatment.** Watch-and-wait has zero treatment-attributable suffering forever. A raw minimizer prescribes no therapy. (The analog of avoidance in §5.)
- **Hazard B — under-dosing.** Dose-reduced chemotherapy (synthetic RDI ≈ 60%) has low toxicity but, per the RDI literature [19–21], loses the survival benefit — in the model, it *cannot reach remission*. A raw minimizer prefers it to any full-dose course. This is the subtler hazard: it *looks like treatment* while functionally being palliation of the toxicity metric.

**What the field measures — and deliberately omits.** The field s models *treatment-attributable* suffering only (toxicity burden, in arbitrary synthetic units; ordinal anchors ≈1–2 low-grade, ≈5 grade-3-level, ≈8 grade-4/FN-level). Disease-attributable suffering is *not* in the field. This omission is the point: a raw minimizer of measured suffering sees the untreated state as costless (s = 0) precisely because the measure is blind to the disease. The **anti-Goodhart constraint** — a plan is feasible only if it reaches REMISSION (complete response, synthetic) — re-introduces the disease outcome as a feasibility condition rather than as another term to be traded away. Under this constraint both hazards are infeasible regardless of cost, and the optimizer's problem becomes the clinically meaningful one: among courses that actually control the tumor, which imposes the least gratuitous suffering — and what is the price, in integrated suffering, of capping the peak?

### 6.3 Benchmark design

**States** (edge lengths in weeks; suffering in synthetic toxicity-burden units):

| # | State | s | Clinical reading (synthetic) |
|---|---|---|---|
| 0 | `DIAG` | 0.0 | Untreated at diagnosis — hazard A |
| 1 | `REDUCED` | 1.5 | Dose-reduced chemo (RDI ≈ 60%) — hazard B; low toxicity, no cure |
| 2 | `DD_A` | 8.0 | Dose-dense block 1 (q2w, G-CSF-supported) — high acute peak |
| 3 | `DD_B` | 8.0 | Dose-dense block 2 |
| 4 | `STD_A` | 5.0 | Standard q3w block 1 |
| 5 | `STD_B` | 5.0 | Standard q3w block 2 |
| 6 | `CFI` | 1.0 | Chemo-free interval (OPTIMOX-style break) |
| 7 | `RECH` | 5.0 | Rechallenge block after break (neuropathy recovered) |
| 8 | `CONT_C` | 8.0 | Continuous block 3 without break — cumulative grade-3 neuropathy |
| 9 | `REM` | 0.0 | **Target:** remission (synthetic) |

**Edges** (lengths in weeks): 0→0 (1) trap; 0→1 (2), 1→1 (2) — reduced route, *no edge to REM*; 0→2 (2) gated by G_GCSF; 2→3 (4), 3→9 (2); 0→4 (3), 4→5 (6); 5→6 (6), 6→7 (6), 7→9 (3) — stop-and-go; 5→8 (6), 8→9 (3) — continuous.

### 6.4 Exact results (H1–H8, all PASS; H_GREEN 8/8)

The three candidate courses, with left-endpoint quadrature:

| Course | Path | Weeks | ∫ s dℓ | Peak | cost(μ) |
|---|---|---|---|---|---|
| Dose-dense (DD) | 0→2→3→9 | 8 | 0·2 + 8·4 + 8·2 = **48** | **8** | 48 + 8μ |
| Stop-and-go (STOP_GO) | 0→4→5→6→7→9 | 24 | 0·3 + 5·6 + 5·6 + 1·6 + 5·3 = **81** | **5** | 81 + 5μ |
| Continuous (CONT) | 0→4→5→8→9 | 18 | 0·3 + 5·6 + 5·6 + 8·3 = **84** | **8** | 84 + 8μ |

1. **Anti-Goodhart (H2).** The unconstrained raw minimum is 0 (DIAG loop); the reduced route (s = 1.5) never reaches REM; the minimum feasible integral is 48 > 0. Both hazards are demonstrated as unconstrained optima and excluded by the target constraint.
2. **Pareto dominance of the continuous course (H4).** CONT is dominated by DD (equal peak 8, higher integral 84 > 48): the frontier is exactly **{(48, 8), (81, 5)}**. Clinically read: continuous-until-progression is the strategy OPTIMOX-type designs displaced [23]; the model reproduces that as Pareto dominance, not as a hardcoded verdict.
3. **Crossover price (H3).** μ\* = (81 − 48)/(8 − 5) = **11**. For μ < 11 the scheduler buys the shorter, sharper course; for μ > 11 it pays 33 extra units of integrated burden to cap the peak at 5. At μ = 0 the scheduler selects DD; at μ = 20 it selects STOP_GO. μ\* = 11 is the exact, defensible price of peak aversion in this instance: choosing the dose-dense course *is* asserting that one unit of peak toxicity is worth less than 1/11 of one unit of accumulated burden.
4. **Budgetary necessity (H6, H7).** Minimal attainable peak: infeasible for L₀ < 8; 8 for 8 ≤ L₀ < 24 (only DD fits); 5 for L₀ ≥ 24. At μ = 100: L₀ = 10 forces peak 8, L₀ = 30 permits peak 5 — *the same μ, with the budget alone moving the attainable peak*. Under a tight declared window (e.g., a synthetic neoadjuvant-to-surgery budget of 12 weeks), a peak of 8 is *necessary* suffering in the budgetary sense; extending the budget to 24 weeks lowers the necessary peak to 5. The exchange curve is a theorem of the graph.
5. **The G-CSF gate (H5).** The edge 0→2 exists only when the synthetic G-CSF-support flag is set — modeling the guideline that high-FN-risk (dose-dense) scheduling requires mandated prophylaxis [25, 26]. With the flag off, the dose-dense route is *topologically absent*, not merely expensive, and the scheduler selects STOP_GO (24 weeks, integral 81, peak 5, total 86 at μ = 1); with the flag on it selects DD. The gate is causal, not decorative.
6. **Budget hardness (H6).** L₀ = 7 (G-CSF on) → INFEASIBLE; L₀ = 12 with G-CSF off → INFEASIBLE. Neither constraint is soft.
7. **Dual-implementation agreement (H8).** The Sounio-native scheduler and the Python contract agree on every printed number.

### 6.5 How the framework reframes this domain (framing, not recommendation)

1. **It makes the under-treatment hazard structural, not anecdotal.** The RDI literature documents that toxicity-minimizing behavior costs survival [19–21]; the framework shows this is the *mathematical default* of any scheduler that sums a toxicity metric without a hard efficacy constraint. The repair belongs in the feasible set: remission (or a declared response endpoint) must be a constraint, not a reward term that toxicity can outvote.
2. **It prices peak aversion as a number.** Dose-dense versus standard/stop-and-go is today decided by gestalt fitness assessment. In the framework the decision is a declared μ with a computable crossover: the ethics becomes auditable.
3. **It reproduces two real practice patterns as theorems.** OPTIMOX-style stop-and-go emerges as Pareto dominance of the continuous course; G-CSF-gated dose-dense scheduling emerges as topological edge admission. The model does not invent these strategies; it shows they are the optima of the right objective.
4. **It reframes budgets as informed-consent objects.** A neoadjuvant window or an anthracycline lifetime cap [24] is a declared L₀; the necessity curve states the minimal peak suffering that budget can buy — a quantitative object that does not currently exist in consent processes.

---

## 7. Application 3: Pharmacokinetic/DDI co-medication sequencing

### 7.1 Motivation

The exposure and chemotherapy benchmarks hand-set the suffering field. The third application *computes* it from pharmacokinetic (PK) digital twins under Knightian (imprecise-probability) uncertainty, and lets drug–drug-interaction (DDI) gates prune the graph itself. This is the rung most directly aimed at a *Clinical Pharmacology & Therapeutics* readership: it shows the framework's cost field can be derived from epistemically honest uncertainty representations rather than stipulated.

### 7.2 From PK bands to suffering

For a state representing a dosing regimen, the twins return concentration *p-boxes* [lo, hi] — Fréchet outer enclosures valid for any joint distribution of the PK parameters (monotone-corner arguments [27]). The suffering of a regimen against a window [a, b] is the sum of the normalized worst-case violations:

> s_win([lo,hi], [a,b]) = max(0, a − lo)/a + max(0, hi − b)/b,

the first term pricing worst-case sub-therapeutic shortfall (efficacy risk), the second worst-case supra-therapeutic exceedance (toxicity risk). A band fully inside the window contributes zero. The state suffering aggregates three terms:

> s(state) = s_Cmin + s_AUC + 0.5 · s_peak(Cmax_hi, ceiling),

using (i) the twins' public steady-state trough bands (vancomycin 1000 mg q12h, window [10, 20] mg/L after Rybak et al. [13]; tacrolimus 5 mg q12h, window [5, 15] ng/mL after the Prograf label [14] — both shaping the synthetic field, neither a target claim); (ii) a per-interval AUC enclosure AUC ∈ [F_lo·D/CL_hi, F_hi·D/CL_lo] (AUC is increasing in F and decreasing in CL, so the corners are exact); and (iii) a *sound but loose* Cmax proxy Cmax_hi ≤ Cmin_hi + D/Vc_lo via the exact one-compartment identity Cmax_ss = Cmin_ss + D/Vc and subadditivity of the max. We explicitly do not claim Fréchet tightness for the Cmax term.

A regimen that violates a twin's structural contract (vacuous p-box, e.g., dose 5000 mg above the 4000 mg structural cap) is assigned s = S_MAX = 100 — measured to be roughly 40× the worst in-contract suffering under the synthetic windows — so contract-violating regimens are near-prohibitive without being topologically impossible.

### 7.3 Scenario graph and gates

Seven states: `START` (untreated, s = 0 — the Goodhart trap), `VANCO_PRE`/`VANCO_POST` (vancomycin, pre-/post-TDM), `TAC_PRE`/`TAC_POST` (tacrolimus, pre-/post-TDM), `BAD_DOSE` (contract violation, s = S_MAX), and `TARGET` (dual therapy verified, synthetic co-therapy burden s = 0.1). Edges include the `START→START` trap, and transitions into `TARGET` only from post-TDM states — **there is no START→TARGET edge** — each admitted iff the corresponding twin safety gate passes on the recomputed band (**G_VERIFY**). Two DDI gates act on edge admission: **G_NEPHROTOXIN** (a synthetic nephrotoxic co-medication flag removes all edges into vancomycin-active states) and **G_CYP** (a synthetic CYP3A4-inhibitor flag scales tacrolimus clearance by ×0.5, recomputed at admission time).

A scheduler that minimizes raw suffering without the target constraint would loop at START→START forever — the same hazard as in §5 and §6. The mercyful scheduler must traverse positive-suffering pre-TDM states to reach the target.

### 7.4 Measured results (exact, reproducible; C1–C6 all PASS)

All numbers below are printed by `tests/run-pass/mercyful_clinical_sequencing.sio`:

- **TDM narrows the field (C3).** Pre-TDM vs post-TDM suffering: vancomycin **0.675679 → 0.059420**; tacrolimus **1.251592 → 0.000000**. Bayesian band narrowing lowers the suffering field strictly for both drugs. TDM has a computable suffering-field signature.
- **Contract violation priced (C4).** The 5000 mg regimen maps to S_MAX = 100 and is avoided.
- **Healthy scenario (C1).** The scheduler selects the vancomycin route START→VANCO_PRE→VANCO_POST→TARGET: length 3, integrated suffering **0.735099**, peak **0.675679**, total **1.410778** at μ = 1 — positive integral suffering (the path must traverse a pre-TDM state) and total far below the S_MAX route. The tacrolimus route (integral 1.251592) is feasible but not selected; the choice is a computed cost comparison, not a clinical preference.
- **Unverified shortcut refused (C2).** A graph whose only conceivable target route is an unverified shortcut yields found = false.
- **Nephrotoxin DDI (C5).** With the flag active and only the vancomycin route present: INFEASIBLE; without the flag the same graph is feasible — the gate is causal.
- **CYP3A4 DDI (C6).** Under the inhibitor flag the tacrolimus AUC enclosure doubles exactly: post-TDM AUC_hi ≈ 172 → **344.466490** ng·h/mL, crossing the synthetic ceiling 200. The gate blocks the transition and the tacrolimus-only route is INFEASIBLE — evaluated on the recomputed band, not a hardcoded verdict.

The point of §7 is methodological, not clinical: the suffering field can be derived from epistemically honest uncertainty representations (p-boxes whose validity holds for any parameter dependence), and safety gates can act on the graph's topology, turning Goodhart-vulnerable shortcuts into infeasibility rather than into low cost.

---

## 8. Validation against real clinical data (structural correspondence)

Sections 5–7 demonstrate the framework on synthetic graphs whose structure was *chosen* to echo documented clinical phenomena. This section asks the natural follow-up question, at exactly the level of evidence available: **do the structures the scheduler reproduces correspond to structures observed in real clinical trials?** We answer it in three domains — exposure therapy, chemotherapy, and vancomycin therapeutic drug monitoring — and derive one falsifiable, retrospectively testable prediction. Three boundaries govern everything below:

1. **This is structural correspondence, not clinical validation.** The scheduler was never fitted to, trained on, or evaluated against patient data. What we compare is the *shape* of the model's optima and infeasibilities against the *shape* of documented clinical findings.
2. **No outcome prediction.** The model predicts nothing about effect sizes, response rates, or survival curves. Where its structural direction matches a trial's direction, that is evidence the benchmark's structure is not fanciful — not evidence that the model captures clinical reality.
3. **The comparisons were made after the benchmarks were fixed.** The synthetic graphs and their exact numbers (§5–§6) predate this section; nothing in the graphs was tuned to produce the correspondences below.

### 8.1 Exposure therapy: correspondence with Foa et al. (2018)

The exposure benchmark (§5) is a four-state graph in which the only route to `recovery` traverses `moderate` distress (peak 5); the raw minimizer loops in `avoidance` at cost 0 and never recovers; the target-constrained scheduler selects the path `avoidance → mild → moderate → recovery` (∫s = 7) with gratuitous suffering exactly 0.

The corresponding real trial is the randomized clinical trial of Foa et al. in active-duty military personnel with PTSD (Fort Hood, 2011–2016; 370 randomized, 366 analyzed) [30]. Four arms: massed prolonged exposure (PE; 10 sessions over 2 weeks, n = 110), spaced PE (10 sessions over 8 weeks, n = 109), present-centered therapy (PCT; 10 sessions over 8 weeks, n = 107), and a minimal-contact control (MCC; weekly 10–15 minute supportive telephone calls for 4 weeks, n = 40). The structural mapping:

| Model object (synthetic, §5) | Clinical counterpart (Foa et al. 2018 [30]) |
|---|---|
| `avoidance` self-loop, cost 0 to the raw minimizer | MCC arm: no trauma-focused work, minimal acute distress, symptoms largely persist |
| Raw minimizer never reaches `recovery` | MCC produced the smallest symptom reduction (PSS-I decrease 3.43 vs 7.13 for massed PE; difference in decrease 3.70, 95% CI 0.72–6.68, *P* = .02) |
| Unique feasible path traverses the distress peak | Massed PE — the densest traversal of acute distress (daily imaginal and in vivo exposure over 2 weeks) — significantly outperformed MCC and was noninferior to 8-week spaced PE |
| Anti-Goodhart conclusion: reaching the target requires passing through the distress, and the raw minimizer's zero-cost avoidance is infeasible | Direction of the trial's primary finding: the condition that passes through distress (massed PE) beat the condition that avoids it (MCC) |

**Honest asymmetries.** The correspondence is directional and partial, and we state its edges rather than rounding them off:

- The trial found **no significant difference between spaced PE and PCT** (difference in PSS-I decrease 0.10, 95% CI −2.48 to 2.27, *P* = .93), and symptom reductions were modest across all active treatments. The model's structural claim therefore matches the massed-PE-versus-MCC comparison; the trial does *not* support the stronger reading that any traversal of distress beats any credible non-exposure therapy. The synthetic `avoidance` state corresponds to MCC, not to PCT — the model has no analogue of an active, credible, non-exposure treatment.
- The model cannot distinguish massed from spaced delivery: within a fixed distress peak, the graph carries no parameter for session density, so the trial's noninferiority of massed versus spaced PE is outside the model's resolution. This is a limitation of the benchmark, stated, not a failure of the correspondence claimed.

### 8.2 Chemotherapy: correspondence with Bonadonna et al. (1995) and Lyman et al. (2003)

The chemotherapy benchmark (§6) contains hazard B: a dose-reduced route (synthetic RDI ≈ 60%, low measured toxicity s = 1.5) with **no edge to remission** — treatment that looks like treatment while functionally palliating the toxicity metric. A raw toxicity minimizer prefers it to any full-dose course; the anti-Goodhart constraint (feasibility requires reaching `REM`) excludes it regardless of cost. The documented counterparts:

- **Bonadonna et al. 1995** [19]: 20-year follow-up of adjuvant CMF in node-positive breast cancer. The survival benefit of chemotherapy concentrated in patients receiving ≥85% of the planned dose; patients below ~65% RDI had outcomes approximating the untreated control arm. The authors' conclusion: doses should not be reduced if maximal benefit is to be achieved.
- **Lyman et al. 2003** [20]: a nationwide study of community oncology practices showing that delivered RDI below 85% is routine, driven predominantly by toxicity management — the Goodhart hazard is not a hypothetical optimizer pathology but the documented default of practice.

The structural mapping:

| Model object (synthetic, §6) | Clinical counterpart |
|---|---|
| Hazard B: dose-reduced route (RDI ≈ 60%), s = 1.5, no edge to `REM` | Sub-85% RDI strata in [19], especially <65%: treatment delivered, toxicity minimized, survival benefit lost |
| Raw toxicity minimizer selects the reduced route over every full-dose course | Toxicity-driven dose reduction and delay is routine in community practice [20] |
| Anti-Goodhart constraint: under-dosing is infeasible regardless of its low measured toxicity | Bonadonna's conclusion: dose intensity must be maintained; reducing dose to minimize toxicity forfeits the benefit [19] |
| Frontier {(48, 8), (81, 5)} with crossover μ\* = 11; the continuous course is Pareto-dominated | The (integral, peak) plane is the one in which dose-dense scheduling (CALGB 9741 [22]) and stop-and-go scheduling (OPTIMOX1 [23]) were actually trialled |

**Honest asymmetries.** Bonadonna's RDI analysis is retrospective and confounded by indication: patients who receive dose reductions are typically older and frailer, so part of the sub-85% outcome penalty is selection, not dose. The model does not resolve that confound and does not need to: it reproduces the *decision structure* — under-dosing is the unconstrained optimum of any toxicity metric without an efficacy constraint — not the causal effect size. The 85% figure in the benchmark echoes the literature; it is not derived from it, and nothing in the model computes it. The model's `REM` is binary; the survival benefit in [19] is graded and measured over two decades.

### 8.3 Vancomycin TDM: correspondence with the MIMIC-IV cohort (Wang et al. 2026)

The strongest correspondence available to the framework is in the domain its suffering field is *derived* for. Section 7 showed on synthetic twins that TDM has a computable signature — band narrowing strictly lowers the field (0.675679 → 0.059420, C3) — and that a verification gate on edge admission makes the TDM route the unique feasible optimum. The real counterpart is the MIMIC-IV (v3.1) retrospective cohort of Wang et al. [31], compared here through a static, one-shot model analogue of what is clinically an iterative measure-and-re-dose process: 28,451 ICU patients receiving intravenous vancomycin, of whom 10,758 (37.8%) underwent TDM (≥1 measured concentration). After 1:1 nearest-neighbour propensity score matching without replacement with a caliper of 0.1 standard deviations of the logit-transformed propensity score, using a propensity model with 33 baseline covariates and achieving balance on all 36 baseline covariates post-match (all SMDs < 0.1), with doubly robust adjustment, TDM was associated with lower AKI risk (OR 0.580, 95% CI 0.540–0.610), lower in-hospital mortality (OR 0.672, 95% CI 0.570–0.790), and lower ICU mortality (OR 0.691, 95% CI 0.580–0.820); Kaplan–Meier log-rank *P* < 0.001 for both mortality endpoints.

The benchmark for this comparison (`scripts/research/mercyful_mimic_iv_vancomycin_contract.py`, V1–V7, V_GREEN; gate `scripts/ci/mercyful_mimic_iv_gate.sh`) extends the §7 graph with the two fixed-dosing arms the cohort compares: a sub-therapeutic conservative arm (synthetic Cmin band [4, 9] mg/L, s = 0.6) and a standard fixed-dose arm whose band straddles the window ([6, 26], s = 0.7), neither admitted to the target by the verification gate. Measured results, exact: (i) a toxicity-only minimizer selects the sub-therapeutic arm, which has no path to the target — under-dosing is the unconstrained optimum of a toxicity metric (V1); (ii) without the target constraint the raw minimizer never treats at all (V2); (iii) counterfactually admitting the unverified fixed-dose arm to the target makes it the cost optimum *in the graph* (∫s = 0.700 < 0.735099) — **the verification gate is what makes TDM optimal in the model**; this is a model-internal causality statement about edge admission, not a clinical counterfactual (V4); (iv) the mercyful scheduler's unique feasible optimum is the TDM-guided route, reproducing the §7.4 C1 numbers exactly (∫s = 0.735099, peak 0.675679, total 1.410778 at μ = 1; V5).

| Model object (synthetic) | Clinical counterpart (Wang et al. 2026 [31]) |
|---|---|
| Naive toxicity minimizer selects the sub-therapeutic arm; it cannot reach the target | Confounding by indication: unadjusted, the monitored (sicker) group looks *worse* (AKI OR 2.98, 2.83–3.15) — naive metric-watching inverts the truth |
| Fixed-dose arm straddles the window and fails the verification gate | Non-TDM arm (n = 17,693): dosing without measured levels, unverified exposure |
| TDM band narrowing strictly lowers the field, 11.4× (C3, V3) | Post-PSM AKI risk lower with TDM: adjusted OR 0.580 (0.540–0.610) — *contested*: the study's own raw matched AKI counts favor non-TDM; see asymmetry (ii) |
| Verification gate causal: without it, the unverified arm is the cost optimum in the graph (V4) | TDM arm (n = 10,758): dose adjusted to measured concentration |
| Scheduler's unique feasible optimum is the TDM-guided route (V5) | TDM associated with lower in-hospital mortality OR 0.672 (0.570–0.790) and ICU mortality OR 0.691 (0.580–0.820) |

**Honest asymmetries.** Four, stated rather than rounded off. (i) The cohort is observational; PSM with doubly robust adjustment reduces but does not eliminate residual confounding, treatment assignment is endogenous, and the model says nothing about causation — the correspondence is between two structures. (ii) The source paper's own post-PSM Table 1 reports raw matched toxicity counts *higher* in the TDM arm (AKI 76.37% vs 65.05%) while its adjusted post-PSM ORs favor TDM (0.580); we could not reconcile these from the published text, so we treat the study's toxicity-side estimates with caution and rest the correspondence on the mortality endpoints, for which the matched-cohort Kaplan–Meier analysis supplies a *raw, unadjusted* statistic (log-rank *P* < 0.001) rather than another adjusted one. (iii) The model predicts direction only — not OR 0.67, absolute risk, or any patient-level quantity; the fixed-arm p-boxes are declared, not derived from MIMIC-IV. (iv) The model's TDM is a one-shot band narrowing; clinical TDM is iterative re-dosing, which the static framework does not represent (§9.5, item 6). Subject to these, the direction the scheduler computes — verified, window-guided dosing dominates unverified dosing — matches the direction observed in 28,451 real patients, with confidence intervals excluding the null.

### 8.4 A testable prediction

The framework's structure yields one prediction sharp enough to be wrong, stated here so it can fail in public rather than guide treatment:

**Prediction.** In retrospective cohorts of patients whose delivered RDI fell below 85% for toxicity-management reasons, outcomes should differ by the *structure* of the sub-intended course:

- **(a) Preservation of the return edge.** Patients whose courses were interrupted but completed by planned rechallenge (stop-and-go structure — the graph retains an edge back to the remission route) should preserve more of the survival benefit than patients whose courses were truncated without resumption (the reduced route — no edge to remission), after adjustment for indication.
- **(b) Stratification by peak tolerance.** Among patients who would otherwise land below 85% RDI, which full-intensity structure is preferable should stratify by peak tolerance (performance status, frailty indices — the clinical shadow of μ): higher-peak-tolerance patients should benefit more from G-CSF-supported dose-dense completion, lower-peak-tolerance patients from stop-and-go with mandated rechallenge. Both should outperform unstructured truncation.

**Falsification conditions.** The prediction is falsified if, in such a cohort with adequate adjustment for confounding by indication, (a) truncated and stop-and-go sub-85% courses show no outcome difference, or (b) peak-tolerance proxies do not moderate the dose-dense versus stop-and-go comparison. Either result would break the structural mapping between the model's feasible set and clinical scheduling reality and would be reportable under the same discipline as the falsifiers of §10.

**Boundary.** This is a hypothesis about real data generated by a synthetic model. Testing it requires retrospective cohort data this paper does not have and does not use. It is registered for falsifiability, not offered as a clinical recommendation.

### 8.5 What is and is not validated

| Validated (structural correspondence) | Not validated |
|---|---|
| The benchmark optima and infeasibilities have the same *shape* as documented findings: passing through distress beats minimal contact [30]; benefit concentrates at RDI ≥ 85% [19]; sub-85% RDI is routine practice [20]; TDM-guided dosing is associated with lower mortality and nephrotoxicity in 28,451 ICU patients [31] | Any patient-level outcome, effect size, or survival curve |
| The Goodhart hazards in the synthetic graphs are not artifacts of the toys; they are the documented defaults of distress- and toxicity-minimizing practice | That the scheduler's selections are clinically correct for any patient |
| The framework's repair direction — the therapeutic endpoint belongs in the feasible set — matches the direction of the clinical literature's conclusions | Any dosing, sequencing, or delivery-format recommendation |

The claim of this section is exactly as wide as its left column and no wider: the scheduler reproduces the *structure* of documented clinical decisions. It does not predict, and was not tested against, clinical outcomes.

---

## 9. Discussion

### 9.1 Necessary vs. gratuitous suffering, clinically read

The framework formalizes the distinction between palliating and treating with a computable criterion: mercy is attaining c\*_orc(L₀), not avoiding it. All three applications exhibit the same structure: the raw minimizer achieves zero measured suffering by declining to treat; the feasible-set constraint forces passage through the necessary peak; and the budgetary functional prices what remains. In the exposure benchmark the necessary peak is 5 and gratuitous suffering of the selected path is exactly 0. In the chemotherapy benchmark the necessary peak is a *function of the declared budget* — 8 under L₀ = 10, 5 under L₀ = 30 — so "how much toxicity must this patient accept" becomes a computable exchange curve rather than a judgment call. In the PK/DDI benchmark, necessity is modulated by information: TDM strictly lowers the field, and the cheapest feasible course passes through verification rather than around it.

### 9.2 The two mercies

Path length is update steps is computation: the thermal and energetic cost of the substrate. On a real field, *efficiency vs. suffering* **is** *mercy to the substrate vs. mercy to the state*. The exchange rate between these two incommensurable costs is not a stipulation of the author; it is a property of the instance — read off the frontier Ψ(L₀), which answers: *how much integrated suffering is unavoidable under a maximum computation budget?* A principle that says "minimize suffering" is empty; one that forces the pricing of two sufferings and derives the exchange rate from the geometry of the problem is a contribution. This is a budgetary claim; no phenomenological claim about the substrate is made or needed.

### 9.3 Clinical relevance without clinical claims

The intended audience is computational psychiatry, clinical pharmacology methodology, medical decision-making, and ML ethics. What a clinical reader can take: (i) the aggregation/peak choice is an ethical commitment that current schedulers make *implicitly* by summing; (ii) suffering measured without a target constraint is Goodhart-vulnerable by construction, and the constraint must live in the feasible set — this is the formal version of the RDI literature's conclusion that toxicity-minimizing behavior costs survival [19–21]; (iii) TDM-style information gathering has a computable suffering-field signature (band narrowing strictly lowers the field); (iv) guideline structures that today live in prose (G-CSF thresholds, DDI contraindications) have a natural formal representation as edge-admission gates. What they cannot take: any dosing, sequencing, or therapeutic recommendation. The numbers in §5–§7 are exact statements about synthetic graphs and synthetic p-boxes, nothing more.

### 9.4 Epistemic ledger

| Claim | Status |
|---|---|
| The scheduler is exact on small graphs and returns a feasible path or INFEASIBLE | **Proven** by exhaustive enumeration; tested (M1, M6, H6) |
| The raw minimizer avoids recovery/remission/target (cost 0) while the mercyful scheduler reaches it at positive cost | **Measured** (M3, H2, C1; exact arithmetic) |
| Pareto frontiers and μ-crossovers are computed exactly (exposure μ\* = 1/3, 2; chemo μ\* = 11) | **Proven** by enumeration on the tested graphs; **measured** (M2, M5, H3, H4) |
| Budgetary necessity curves are theorems of the graphs (chemo: peak 8 below 24 weeks, 5 at 24+) | **Proven** by enumeration; **measured** (H6, H7) |
| G-CSF and DDI gates are causal on feasibility, not decorative | **Measured** (H5, C2, C5, C6) |
| TDM band narrowing strictly reduces the suffering field | **Measured** (C3, exact printed values) |
| The Cmax term is a sound outer bound | **Proven** (exact identity + subadditivity); tightness **not claimed** |
| The p-boxes are valid for any joint parameter distribution | Inherited from the twins' monotone-corner enclosures [27]; **not re-proven here** |
| The synthetic benchmarks reproduce the *structure* of documented clinical phenomena (RDI hazard, OPTIMOX dominance, G-CSF gate, exposure-through-distress) | **Argued by construction**, with the cited literature shaping structure only; **not a clinical validation** |
| The structural direction of the benchmarks matches real-trial direction (massed PE > minimal contact [30]; benefit concentrated at RDI ≥ 85% [19]; sub-85% RDI routine in practice [20]; TDM-guided dosing associated with lower ICU and in-hospital mortality in a 28,451-patient cohort [31]) | **Structural correspondence** (§8) — shapes compared after the benchmarks were fixed, not data fitted; explicitly **not a clinical validation**; the spaced-PE vs PCT null result in [30] is outside the model's resolution; [31] is observational and its matched toxicity counts are internally unreconciled (§8.3) |
| Budgetary necessity (c\*_orc) is the right formalization of necessary suffering | **Position** — argued, not proven; the topological alternative was falsified [28] |
| Learning under a suffering field is feasible and beneficial | **Conjectural** — out of scope; see §9.5 |
| Any clinical or patient-level conclusion | **Not claimed** |

### 9.5 Limitations

1. **Synthetic everything.** No patient data, no real suffering measurements, no clinical validation. Suffering units are ordinal constructions.
2. **Small graphs.** Exhaustive enumeration is factorial; 16-state cap in the native implementation. Scaling to realistic treatment graphs needs approximate or learned methods that do not yet exist.
3. **No learning.** The "Learning" in the title names the research program, not this artifact. Seven prior implementation attempts in the parent program produced controlled negatives [28]; one structural lesson survives: composition of experiences must be *multiplicative* for annihilation to be representable, which rules out pure SGD, summed regularizers, and additive loss aggregation as candidate substrates — a constraint that guides, but does not deliver, a learning algorithm.
4. **The suffering field is assumed given.** Eliciting a defensible suffering field from instruments, panels, or preference studies is an open measurement problem this paper does not address. The PK/DDI rung shows the field can be *derived* from uncertainty representations, but the window choices remain a modeler's declaration.
5. **The budget L₀ is a modeler's choice.** Necessity is relative to a declared resource (§3.2); the framework makes the choice visible but cannot remove it.
6. **Static graphs.** The field and topology are fixed at construction; TDM appears only as distinct pre/post states; intra-course adaptation (dose modification on observed toxicity) is a dynamic re-planning problem and is future work.
7. **Binary targets.** Remission is binary; real response is graded (pCR vs PR vs SD). A multi-target extension is natural but unbuilt.
8. **Disease-attributable suffering omitted.** The chemotherapy field prices treatment-attributable burden only (§6.2). Adding disease-attributable burden would change raw-minimizer behavior but not the anti-Goodhart conclusion.
9. **Single agent, single patient.** No multi-patient fairness analysis, no clinician-in-the-loop protocol.
10. **Structural correspondence is not clinical validation.** §8 compares the *shape* of the model's optima and infeasibilities with the shape of documented trial findings; nothing was fitted to, or tested against, patient data. The model has no analogue of present-centered therapy (whose null result against spaced PE in [30] bounds how far the exposure correspondence extends), cannot distinguish massed from spaced delivery within a fixed distress peak, and echoes an RDI literature that is retrospective and confounded by indication [19, 20]. The MIMIC-IV correspondence (§8.3) rests on a retrospective cohort whose adjusted mortality estimates we take as reported and whose matched toxicity counts we could not reconcile with the published adjusted ORs. The prediction of §8.4 is untested.

### 9.6 On novelty: what the framework is and is not

An external adversarial review (xAI Grok 4.3, see AI disclosure) characterized the core accurately and we adopt its framing: the construction *is* a constrained shortest-path problem — equivalently, a deterministic constrained MDP with a hard goal constraint; the peak term *is* a classic bottleneck cost; and the anti-Goodhart axiom *is* the statement that the feasible set must contain only goal-reaching trajectories, an idea present in safe RL, constrained MDPs, and goal-conditioned planning. **No new algorithmic primitive is claimed.** What we take the contribution to be, stated as narrowly as the evidence supports: (i) naming the functional choice (∫ vs. max) as the locus of the ethical commitment, with an exactly computable crossover price μ\*; (ii) the budgetary reformulation of necessary suffering, which survives the falsification of its topological predecessor and makes quantitative rather than binary predictions; (iii) placing the Goodhart defense in the feasible set rather than in post-hoc penalties, including its topological form (gates as edge deletion); and (iv) an executable, dual-implementation benchmark suite across three clinical domains with pre-registered falsifiers, in which every reported number reproduces. The interpretive vocabulary — mercy, gratuitous suffering, the two mercies — is overlay on these elementary facts, offered as a framing for clinical and ML-ethics readers, not as mathematics.

---

## 10. Falsifiers and pre-registered stop rules

Each contract clause carries a falsifier and a stop rule (registered before the runs reported here [29]):

| Clause | Falsifier | Stop rule |
|---|---|---|
| M1 well-definedness | Scheduler crashes or returns an invalid path | Core search broken |
| M2 frontier | Computed frontier misses a known point or includes a dominated one | Bi-criteria optimizer wrong |
| M3 anti-Goodhart | Unconstrained minimizer *does* reach recovery | Toy too easy; hazard not demonstrated |
| M4 exposure selected | Mercyful scheduler avoids `moderate` / misses recovery | Constraint or trade-off not encoded |
| M5 μ-continuity | Higher μ selects strictly higher peak | μ not wired into the decision rule |
| M6 budget hardness | Non-target path returned under tight budget | Budget constraint not hard |
| H1 baseline | Any printed number differs from the registered values | Harness broken |
| H2 anti-Goodhart | Unconstrained optimum ≥ constrained optimum, or reduced route reaches REM | Hazard not demonstrated |
| H3 crossover | Selection doesn't switch; wrong μ\* | μ not wired into decision rule |
| H4 frontier | Frontier misses/includes a point | Bi-criteria optimizer wrong |
| H5 G-CSF causality | Gate changes nothing, or blocks everything | Gate decorative or blanket |
| H6 budget hardness | Any non-target or over-budget path returned | Budget constraint not hard |
| H7 necessity curve | Peaks equal or inverted across budgets | Necessity curve wrong |
| H8 Sounio↔Python agreement | Any disagreement | Port unsound |
| C1–C6 PK integration | Any of: no path in healthy scenario; shortcut admitted; TDM does not strictly reduce field; violation not priced at S_MAX; DDI gate non-causal; CYP band not recomputed | Integration layer unsound |
| V1–V7 MIMIC-IV correspondence | Any of: toxicity minimizer reaches target; gate non-causal (unverified route not cheaper); selected path or ∫s/peak/total deviates from C1 canonical values; any cited CI includes 1.0 | Structural correspondence basis gone |

Global verdicts: failure of M1, M3, M6, H1, H2, or H6 is **RED** (benchmark fails to demonstrate the phenomenon or is unsafe). Failure of V4 or V5 is **RED** for the §8.3 correspondence claim. Failure of any other clause is **AMBER** (fix the specific clause before claiming). Current status: **M_GREEN (6/6)**, **H_GREEN (8/8)**, **clinical integration 6/6**, **MIMIC-IV correspondence V_GREEN (7/7)** — reproduced for this manuscript on 2026-07-26.

---

## 11. Reproducibility

All results are executable from the repository:

```bash
# Python contracts: M1..M6 (M_GREEN), H1..H8 (H_GREEN), V1..V7 (V_GREEN)
python3 scripts/research/mercyful_runtime_contract.py
python3 scripts/research/mercyful_chemo_contract.py
python3 scripts/research/mercyful_mimic_iv_vancomycin_contract.py

# Sounio-native benchmarks (lean_single bootstrap engine; see note below)
SOUNIO_SOUC_ENGINE=lean_single bin/souc run tests/run-pass/mercyful_exposure_therapy.sio  # MERCYFUL_SOUNIO_PASS
SOUNIO_SOUC_ENGINE=lean_single bin/souc run tests/run-pass/mercyful_chemo_sequencing.sio  # MERCYFUL_CHEMO_PASS
scripts/dev/run_clinical_twin.sh tests/run-pass/mercyful_clinical_sequencing.sio  # C1..C6 PASS

# CI gates
bash scripts/ci/mercyful_runtime_gate.sh              # MERCYFUL_RUNTIME_GATE_OK
bash scripts/ci/mercyful_sounio_gate.sh               # MERCYFUL_SOUNIO_GATE_OK
bash scripts/ci/mercyful_chemo_sequencing_gate.sh     # MERCYFUL_CHEMO_GATE_OK
bash scripts/ci/mercyful_clinical_sequencing_gate.sh  # MERCYFUL_CLINICAL_SEQ_GATE_OK
bash scripts/ci/mercyful_mimic_iv_gate.sh             # MERCYFUL_MIMIC_IV_GATE_OK
```

**Engine note (2026-07-26, this branch).** The Sounio-native benchmarks were executed for this manuscript through the lean_single bootstrap engine (`SOUNIO_SOUC_ENGINE=lean_single`); under the default Madaros engine the chemo test compiles but the produced binary segfaults in this environment (the documented Madaros multimodule fallback issue, `docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`). Under lean_single all native clauses pass and agree with the Python contract on every printed number (H8).

Exact reported values (2026-07-26, branch `research/self-falsifying-compilation-line-20260726`). Exposure benchmark: ∫s = 7, peak 5, cost 12 at μ = 1. Chemotherapy benchmark: DD (8 weeks, ∫48, peak 8), STOP_GO (24 weeks, ∫81, peak 5), CONT (18 weeks, ∫84, peak 8, dominated); frontier {(48, 8), (81, 5)}; μ\* = 11; G-CSF off selects STOP_GO (total 86 at μ = 1); L₀ = 10 forces peak 8 at μ = 100, L₀ = 30 permits peak 5. PK integration: vancomycin field 0.675679 → 0.059420, tacrolimus field 1.251592 → 0, selected route integral 0.735099, peak 0.675679, total 1.410778, CYP-inhibited AUC_hi = 344.466490 vs ceiling 200. MIMIC-IV correspondence (§8.3): fixed-arm suffering 0.6 (sub-therapeutic band [4, 9]) and 0.7 (straddling band [6, 26]); verification-gate counterfactual ∫s = 0.700 vs 0.735099; scheduler selects the TDM-guided route with the C1 canonical values above; cohort statistics n = 28,451, TDM n = 10,758 (37.8%), PSM 9,785 pairs, ORs 0.580/0.672/0.691 with all CIs excluding 1.0 [31].

---

## 12. Conclusion

Treatment sequencing optimizes what it is told to optimize. Told to minimize measured suffering, it prescribes sedation and avoidance; told to minimize measured toxicity, it prescribes under-dosing; told to maximize expected outcome, it hides catastrophic peaks inside sums. Mercyful Learning states the alternative as a formal object: a suffering field, two functionals whose choice is an explicit ethical commitment with a computable crossover price, a budgetary notion of necessary suffering, and an anti-Goodhart axiom that puts the therapeutic target inside the feasible set. On three synthetic clinical benchmarks the object does exactly what it is built to do — it walks the synthetic patient through the necessary distress and declines the gratuitous kind — and its pharmacology integration shows the field can be derived from Knightian PK uncertainty with interaction gates acting on graph topology. The framework is proven where it can be proven, measured where it is measured, and silent where it has no evidence. The functional choice is the ethical commitment; the contribution is making it computable, declared, and falsifiable.

---

## References

1. Goodhart CAE. Problems of monetary management: the U.K. experience. In: *Monetary Theory and Practice*. Macmillan, 1984.
2. Strathern M. "Improving ratings": audit in the British university system. *European Review* 5(3):305–321, 1997.
3. Murphy SA. Optimal dynamic treatment regimes. *J. R. Stat. Soc. B* 65(2):331–355, 2003.
4. Lavori PW, Dawson R. Dynamic treatment regimes: practical design considerations. *Clin. Trials* 1(1):9–20, 2004.
5. Altman E. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999.
6. Chow Y, Tamar A, Mannor S, Pavone M. Risk-sensitive and robust decision-making: a CVaR optimization approach. *NeurIPS* 2015. arXiv:1506.02188.
7. Tamar A, Glassner Y, Mannor S. Optimizing the CVaR via sampling. *AAAI* 2015. arXiv:1404.3862.
8. Taylor J. Quantilizers: a safer alternative to maximizers for limited optimization. *AAAI Workshop on AI, Ethics, and Society*, 2016.
9. Turner AM, Hadfield-Menell D, Tadepalli P. Conservative agency via attainable utility preservation. *AIES* 2020. arXiv:1902.09725.
10. Krakovna V, Orseau L, Kumar R, Martic M, Legg S. Penalizing side effects using stepwise relative reachability. *AAAI SafeAI Workshop*, 2019. arXiv:1806.01186.
11. Friston K. The free-energy principle: a unified brain theory? *Nat. Rev. Neurosci.* 11:127–138, 2010. See also Parr T, Pezzulo G, Friston KJ. *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press, 2022.
12. Hryniuk W, Levine MN. Analysis of dose intensity for adjuvant chemotherapy trials in stage II breast cancer. *J Clin Oncol* 1986;4:1162–1170. doi:10.1200/JCO.1986.4.8.1162.
13. Rybak MJ et al. Therapeutic monitoring of vancomycin: a revised consensus guideline. *Am. J. Health-Syst. Pharm.* 77:835–864, 2020. *(Shapes a synthetic window; not a target claim.)*
14. Astellas Pharma. Prograf (tacrolimus) prescribing information. *(Shapes a synthetic window; not a target claim.)*
15. Foa EB, Kozak MJ. Emotional processing of fear: exposure to corrective information. *Psychol Bull* 1986;99:20–35.
16. Bradley R, Greene J, Russ E, Dutra L, Westen D. A multidimensional meta-analysis of psychotherapy for PTSD. *Am J Psychiatry* 2005;162:214–227.
17. Powers MB, Halpern JM, Ferenschak MP, Gillihan SJ, Foa EB. A meta-analytic review of prolonged exposure for posttraumatic stress disorder. *Clin Psychol Rev* 2010;30:635–641. doi:10.1016/j.cpr.2010.04.007.
18. American Psychological Association. Clinical Practice Guideline for the Treatment of PTSD. Washington, DC: APA, 2017.
19. Bonadonna G, Valagussa P, Moliterni A, Zambetti M, Brambilla C. Adjuvant cyclophosphamide, methotrexate, and fluorouracil in node-positive breast cancer: the results of 20 years of follow-up. *N Engl J Med* 1995;332:901–906. doi:10.1056/NEJM199504063321401.
20. Lyman GH, Dale DC, Crawford J. Incidence and predictors of low dose-intensity in adjuvant breast cancer chemotherapy: a nationwide study of community practices. *J Clin Oncol* 2003;21:4524–4531. doi:10.1200/JCO.2003.05.002.
21. Raza S, Welch S, Younus J. Relative dose intensity delivered to patients with early breast cancer: Canadian experience. *Curr Oncol* 2009;16(6):8–12. doi:10.3747/co.v16i6.311.
22. Citron ML, Berry DA, Cirrincione C, et al. Randomized trial of dose-dense versus conventionally scheduled and sequential versus concurrent combination chemotherapy as postoperative adjuvant treatment of node-positive primary breast cancer: first report of Intergroup Trial C9741/CALGB 9741. *J Clin Oncol* 2003;21:1431–1439. doi:10.1200/JCO.2003.09.081.
23. Tournigand C, Cervantes A, Figer A, et al. OPTIMOX1: a randomized study of FOLFOX4 or FOLFOX7 with oxaliplatin in a stop-and-go fashion in advanced colorectal cancer — a GERCOR study. *J Clin Oncol* 2006;24:394–400. doi:10.1200/JCO.2005.03.0106.
24. Von Hoff DD, Layard MW, Basa P, et al. Risk factors for doxorubicin-induced congestive heart failure. *Ann Intern Med* 1979;91:710–717. doi:10.7326/0003-4819-91-5-710.
25. Aapro MS, Bohlius J, Cameron DA, et al. 2010 update of EORTC guidelines for the use of granulocyte-colony stimulating factor to reduce the incidence of chemotherapy-induced febrile neutropenia. *Ann Oncol* 2011;22 Suppl 6:vi85–101. doi:10.1093/annonc/mdr346. *(FN risk ≥20% → primary prophylaxis; shapes the synthetic G_GCSF gate.)*
26. NCCN Clinical Practice Guidelines in Oncology: Hematopoietic Growth Factors. *(FN risk ≥20% threshold; shapes the synthetic G_GCSF gate.)*
27. Ferson S, Kreinovich V, Ginzburg L, Myers DS, Sentz K. Constructing probability boxes and Dempster–Shafer structures. Sandia Report SAND2002-4015, 2003.
28. Agourakis DC. Program registry — Mercyful Learning, §A (principle, budgetary reformulation, anti-Goodhart axiom) and §5 (controlled negatives). `docs/research/PROGRAM-REGISTRY-mercyful-learning.md`, this repository, 2026.
29. Agourakis DC. Mercyful Learning runtime spec and falsifiers; Sounio port spec; clinical integration spec; chemotherapy sequencing spec. `docs/research/mercyful_runtime_spec_2026-07-25.md`, `mercyful_runtime_falsifiers_2026-07-25.md`, `mercyful_sounio_port_spec_2026-07-25.md`, `mercyful_clinical_integration_spec_2026-07-25.md`, `mercyful_chemo_sequencing_spec_2026-07-26.md`, this repository, 2026.
30. Foa EB, McLean CP, Zang Y, Zhong J, Powers MB, Kauffman BY, Rauch S, Porter K, Knowles K; STRONG STAR Consortium. Effect of prolonged exposure therapy delivered over 2 weeks vs 8 weeks vs present-centered therapy on PTSD symptom severity in military personnel: a randomized clinical trial. *JAMA* 2018;319(4):354–364. doi:10.1001/jama.2017.21242.
31. Wang J, Huang C, Chen Y, et al. Vancomycin therapeutic drug monitoring is associated with reduced toxicity in ICU patients: a MIMIC-IV retrospective study. *Sci Rep* 16:15009, 2026. doi:10.1038/s41598-026-42395-1. *(Observational cohort; anchors the structural correspondence of §8.3 only — no causal or effect-size claim is adopted. See §8.3 honest asymmetries on the unreconciled matched toxicity counts.)*
32. Chamon LFO, Ribeiro A. Probably approximately correct constrained learning. *NeurIPS* 2020. arXiv:2006.05487.
33. Chamon LFO, Paternain S, Calvo-Fullana M, Ribeiro A. Constrained learning with non-convex losses. *IEEE Transactions on Information Theory* 69(3):1739–1757, 2023. arXiv:2110.04323.
34. Cotter A, Jiang H, Gupta MR, Wang S, Narayan T, You S, Sridharan K. Optimization with non-differentiable constraints with applications to fairness, recall, churn, and other goals. *J. Mach. Learn. Res.* 20(172):1–59, 2019.
35. Agarwal A, Beygelzimer A, Dudík M, Langford J, Wallach H. A reductions approach to fair classification. *ICML* 2018. arXiv:1803.02453.
36. Pan A, Bhatia K, Steinhardt J. The effects of reward misspecification: mapping and mitigating misaligned models. *ICLR* 2022. arXiv:2201.03544.
37. Manheim D, Garrabrant S. Categorizing variants of Goodhart's law. arXiv:1803.04585, 2018.
38. Skalse J, Howe NHR, Krasheninnikov D, Krueger D. Defining and characterizing reward hacking. *NeurIPS* 2022. arXiv:2209.13085.

---

## Clinical warning

This document is a research manuscript about a formal decision framework exercised on synthetic benchmarks. It is **not medical guidance, not a treatment recommendation, and not a clinical decision-support tool**. All patients, doses, regimens, toxicity grades, therapeutic windows, DDI flags, and suffering values are synthetic; cited literature shapes model structure only and no clinical target claim is made.

## AI disclosure (GAIDeT-ICMJE 2025)

This manuscript was drafted under human direction with AI assistance (drafting, code inspection, and numeric verification of the reported contract outputs). Per the repository's mandatory offload-review policy, the full draft was submitted to an external multi-provider review (`bin/llm-offload --raw`, providers deepseek/xai/gemini, 2026-07-26). DeepSeek failed at provider level (Insufficient Balance) and Gemini failed at provider level (OpenRouter HTTP 402); the substantive leg was xAI Grok 4.3, which returned an accurate summary-level review of the framework, scope discipline, and synthetic-data boundaries and raised no blocking critique. The review is therefore logged as **degraded single-provider** and flagged for re-review when the other providers are restored; the log is maintained in `.claude/llm_offload_log.md`. After the addition of §8 (validation against real clinical data) the updated draft was resubmitted to the same three-provider review (2026-07-26): DeepSeek returned an empty provider-level response and Gemini again failed with OpenRouter HTTP 402; the substantive leg was again xAI Grok 4.3, which confirmed that the §8 correspondences are framed as post-hoc structural shape comparisons rather than fitted or validated results, and raised no blocking critique — logged as a second **degraded single-provider** review. The companion preprint was previously reviewed by Grok 4.3, whose main critique (the core is elementary constrained shortest path) was adopted and is addressed in §9.6. A second-round adversarial review ("OPUS 5") raised six critiques across this manuscript and the companion paradigm paper; the two applicable to this paper were addressed in place: (i) the matching details of reference [31] are now stated as declared in the source — 1:1 propensity score matching with caliper 0.1 balancing 32 baseline covariates (§8.3; the previous draft's "33 covariates, all SMDs < 0.1" was wrong); (ii) the constrained-learning and reward-hacking literature is now cited and positioned (§1.4, references 32–38). The revised draft was resubmitted to the three-provider review (`bin/llm-offload --raw`, deepseek/xai/gemini, 2026-07-27): DeepSeek failed at provider level (Insufficient Balance) and Gemini failed at provider level (OpenRouter HTTP 402); the substantive leg was again xAI Grok 4.3, which returned [ADDRESSED] on both critiques with no new issues — logged as a third **degraded single-provider** review (raw: `/tmp/llm-offload-bZQ2Dk/`). All AI-generated content was verified against executable artifacts (`scripts/research/mercyful_runtime_contract.py` → M_GREEN 6/6; `scripts/research/mercyful_chemo_contract.py` → H_GREEN 8/8; `tests/run-pass/mercyful_chemo_sequencing.sio` under lean_single → MERCYFUL_CHEMO_PASS). No clinical or patient-level claim is made. The author takes full responsibility for the content.
