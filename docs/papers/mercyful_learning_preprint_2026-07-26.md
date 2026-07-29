<!-- docs:meta
topic_id: repo.docs.papers.mercyful-learning-preprint-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.mercyful-learning-preprint-2026-07-26
-->

# Mercyful Learning: A Formal Framework for Suffering-Budget-Aware Treatment Sequencing

**Author:** Demetrios Chiuratto Agourakis
**Date:** 2026-07-26
**Status:** Preprint draft (arXiv target: q-bio.QM / cs.CY; secondary stat.ME; medical-journal target: *Clinical Pharmacology & Therapeutics* or *Journal of Psychiatric Research*)
**Provenance:** Formal core in `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §A; runtimes in `stdlib/clinical/mercyful.sio` and `scripts/research/mercyful_runtime_contract.py`; benchmarks in `tests/run-pass/mercyful_exposure_therapy.sio` and `tests/run-pass/mercyful_clinical_sequencing.sio`.

> **Scope statement (read first).** This paper formalizes a decision rule for sequencing under a *suffering field* and demonstrates it on synthetic graphs. All patients, doses, windows, drug–drug-interaction flags, and suffering values in this paper are synthetic. Nothing here is medical guidance, a treatment recommendation, or a clinical decision-support tool. No patient data were used; no clinical validation is claimed. The contribution is a formal framework, an exact executable benchmark, and a set of pre-registered falsifiers.

---

## Abstract

Clinical schedulers — dynamic treatment regimes, SMART designs, reinforcement-learning therapy sequencers — almost universally optimize an *expected outcome*: they sum. Summation hides the patient who traverses a catastrophic peak while the mean improves. We introduce **Mercyful Learning**, a formal framework in which suffering is a first-class cost field over a finite state graph, and treatment sequencing is the problem of reaching a therapeutic target while minimizing a budget-constrained combination of *integrated* suffering $\int_\gamma s\,d\ell$ and *peak* suffering $\max_{v\in\gamma} s(v)$. The framework rests on three commitments: (i) the choice of functional *is* the ethical commitment, and must be made explicit and computable; (ii) necessity of suffering is a *budgetary* notion — the minimal peak attainable within a declared resource budget $L_0$ — not a topological one; and (iii) the **anti-Goodhart axiom**: any plan that fails to reach the target is infeasible regardless of cost, because an optimizer of measured suffering will otherwise rediscover the pathology (avoidance, sedation, iatrogenic blunting) as an optimum. We give an exact combinatorial scheduler for small graphs, an executable benchmark on a synthetic exposure-therapy graph in which a naive raw-suffering minimizer prescribes avoidance (cost 0) while the mercyful scheduler traverses distress and reaches recovery (integrated suffering 7, peak 5), and a second integration in which suffering fields are computed from Knightian pharmacokinetic bands of synthetic vancomycin and tacrolimus regimens and drug–drug-interaction gates act as Goodhart constraints on edge admission. All claims are separated into what is proven, what is measured, and what is conjectural; pre-registered stop rules are included.

**Keywords:** suffering budget; treatment sequencing; exposure therapy; Goodhart's law; constrained MDP; Knightian uncertainty; probability boxes; therapeutic drug monitoring; computational pharmacology.

---

## 1. Introduction

### 1.1 The Goodhart problem in clinical scheduling

Goodhart's law — when a measure becomes a target, it ceases to be a good measure — has a clinical name [1, 2]. An algorithm that minimizes *measured* suffering optimizes the measure. In psychiatry the resulting optima are familiar: sedation, affective flattening, iatrogenic anhedonia — all of which reduce distress scores, and all of which a naive system would rediscover as optimal policies. The same structural failure appears in a second, more subtle form: *outcome aggregation*. Dynamic treatment regimes [3], sequential multiple assignment randomized trials (SMART) [4], and reinforcement-learning approaches to therapy sequencing optimize expected cumulative reward. Expected values are sums, and sums are indifferent to *where* the cost concentrates. A policy that drives one patient through an acute crisis while the population mean improves can dominate, on expectation, a policy that never lets any patient cross a catastrophic threshold.

This paper treats both failures as one: a missing constraint and a missing functional.

### 1.2 Suffering as a first-class cost

We model a patient's clinical course as a path $\gamma$ on a finite directed graph whose vertices carry a *suffering field* $s : V \to \mathbb{R}_{\ge 0}$. Two cost functionals over paths are natural and genuinely different:

- **Integrated suffering** $\int_\gamma s\,d\ell$ — the total burden accumulated along the course;
- **Peak suffering** $\max_{v\in\gamma} s(v)$ — the worst moment endured.

Minimizing the integral is *aggregationism*: it permits buying an acute peak with enough tranquil trajectory. Minimizing the maximum is *maximin*. Over the same field the two functionals select different paths. The structure of the problem therefore does not determine the ethics; it makes the ethical choice explicit and computable. We take the combination

$$
\mathrm{cost}(\gamma;\mu) \;=\; \int_\gamma s\,d\ell \;+\; \mu \cdot \max_{v\in\gamma} s(v),
$$

where $\mu \ge 0$ is a declared peak-aversion parameter. For any two competing paths, the crossover value $\mu^*$ at which the peak-averse choice wins is exactly computable, so the ethical weighting becomes a number one must defend rather than an assumption one can hide.

### 1.3 The motivating case: exposure therapy

Exposure therapy is the case that makes the definition clinically worth having. The treatment that works requires acute distress; the behavior that minimizes acute distress is avoidance — which is the disorder. A naive minimizer of $\int$(suffering) prescribes avoidance: it recommends the pathology, with the appearance of compassion. The formal distinction the framework must capture is between *palliating* and *treating* — between avoiding necessary suffering and passing through it at its minimal feasible level.

### 1.4 Relation to prior work

The learner inside the moral domain — optimizing for the suffering of the system it acts upon — has partial precedents we position against honestly:

- **Constrained MDPs (CMDP)** [5] optimize one expected cost subject to constraints on others; our target constraint is a hard feasibility constraint, and our peak term is a path functional, not an expectation.
- **Risk-sensitive RL / CVaR** [6, 7] replaces expectation by coherent risk measures over returns; CVaR is a tail functional of a distribution, whereas the peak term here is a per-trajectory worst case — closer in spirit to minimax-reward formulations but combined with aggregation rather than replacing it.
- **Quantilizers** [8] avoid Goodhart collapse by sampling from an acceptable quantile of a base distribution instead of maximizing; our anti-Goodhart device is complementary: a hard target constraint inside the objective's feasible set.
- **Attainable utility preservation (AUP)** [9] and **relative reachability** [10] penalize side effects by penalizing loss of optionality — penalizing making states unreachable, which is nearly the same idea as our target constraint approached from the other side.
- **Active inference** [11] already markets itself as an alternative to reward-maximizing RL and dominates part of computational psychiatry; it minimizes expected free energy (an information-weighted expectation), not a suffering field with a peak term.
- **Bottleneck/minimax shortest paths and goal-conditioned safe planning.** Our peak term is a classic bottleneck cost and our target constraint is standard goal reachability; the constrained shortest-path formulation over tiny graphs is elementary. We claim no new algorithmic primitive — see §7.6 for an explicit statement of where the contribution does and does not lie.

We also draw on imprecise probability for the clinical integration: suffering fields are computed from *probability boxes* (p-boxes) — Knightian outer enclosures valid for any joint distribution of pharmacokinetic parameters [12] — so that the scheduler prices worst-case band violations rather than point estimates.

### 1.5 Contributions

1. A formal framework (§3): suffering field, two functionals, the budgetary reformulation of necessary suffering, the $\mu$-parametrized decision rule, and the anti-Goodhart axiom.
2. An exact combinatorial scheduler with Pareto-frontier extraction for small graphs (§4), implemented twice — once in pure Python and once natively in the Sounio language — with a six-clause executable contract.
3. A synthetic exposure-therapy benchmark (§5) demonstrating the anti-Goodhart phenomenon with exact numbers: the raw minimizer avoids recovery at cost 0; the mercyful scheduler reaches recovery through distress at cost $7 + 5\mu$.
4. A clinical-integration layer (§6) in which suffering fields are derived from Knightian PK bands of synthetic vancomycin and tacrolimus regimens, and DDI gates (nephrotoxin co-medication, CYP3A4 inhibition) act as Goodhart constraints on edge admission — including a demonstrated case where an interaction doubles an AUC enclosure across a ceiling and the route becomes *infeasible* rather than silently relaxed.
5. Pre-registered falsifiers and stop rules (§8), and an explicit ledger of what is proven, measured, and conjectural (§7.4).

---

## 2. Scope and safety boundaries

This work is deliberately bounded:

- **Synthetic data only.** The exposure-therapy patient, both PK patients (78.5 kg / CrCl 65 mL/min; 70 kg / CrCl 80 mL/min), all doses, all therapeutic windows, all DDI flags, and all suffering values are synthetic constructions chosen to exercise the framework.
- **No clinical recommendation.** Nothing in this paper is a dosing recommendation, a treatment guideline, or a decision-support output. The therapeutic windows used (e.g., a vancomycin trough screen of [10, 20] mg/L after Rybak et al. [13]; a tacrolimus trough window of [5, 15] ng/mL after the Prograf label [14]) shape the synthetic suffering field; they are not clinical target claims.
- **No learning.** The scheduler is an exact combinatorial search over small graphs. Learning under a suffering field is future work, and §7.5 discusses why several natural attempts are expected to fail.
- **No sentience claim.** The framework is agnostic on machine sentience; the "two mercies" structure (§7.2) prices a computational budget, not a phenomenological one.

---

## 3. Formal framework

### 3.1 The suffering field

Let $G = (V, E)$ be a finite directed graph with edge lengths $\ell : E \to \mathbb{R}_{>0}$, a suffering field $s : V \to \mathbb{R}_{\ge 0}$, a start state $a \in V$, a target state $b \in V$, and a length budget $L_0 > 0$. Paths are simple (no repeated vertex) unless stated otherwise; length is $\mathrm{len}(\gamma) = \sum_{e \in \gamma} \ell(e)$.

The integrated suffering is discretized by assigning each edge segment the suffering at its source vertex (left-endpoint quadrature):

$$
\int_\gamma s\,d\ell \;:=\; \sum_{(u,v)\in\gamma} s(u)\,\ell(u,v),
\qquad
\max_\gamma s \;:=\; \max_{v \in \gamma} s(v).
$$

### 3.2 The two budgetary functionals

For each budget $L_0$, define

$$
\Psi(L_0) \;=\; \inf_{\substack{\gamma : a \to b \\ \mathrm{len}(\gamma) \le L_0}} \int_\gamma s\,d\ell,
\qquad
c^*_{\mathrm{orc}}(L_0) \;=\; \inf_{\substack{\gamma : a \to b \\ \mathrm{len}(\gamma) \le L_0}} \max_\gamma s .
$$

Both are well-defined whenever a feasible path exists (infimum over a non-empty finite set of simple paths). These are the *budgetary* forms of necessity: even when every sublevel set $\{s \le c\}$ is connected — so that no *topological* mountain pass exists — the least-suffering route may still be long, and within a realistic budget the attainable peak may sit far above the endpoints' suffering.

**Necessary vs. gratuitous suffering.** We define

$$
\boxed{\;\text{gratuitous}(\gamma) \;=\; \max_\gamma s \;-\; c^*_{\mathrm{orc}}(L_0),\qquad \text{mercy} \;=\; \text{attaining } c^*_{\mathrm{orc}}(L_0).\;}
$$

The budget $L_0$ must be fixed by someone. This is a real loss relative to a topological definition, and we state it rather than hide it: "necessary" stops being a geometric fact and becomes relative to a declared resource. The trade is that the budgetary form makes a *quantitative* prediction (a length–peak exchange curve) where the topological form made only a binary one. (The topological form was tested and falsified twice in the parent research program — once by a connectivity theorem and once on real semantic fields — before this budgetary form, which never depended on disconnection, was adopted; see the program registry [15].)

### 3.3 The decision rule

For peak-aversion parameter $\mu \ge 0$:

$$
\gamma^*(\mu, L_0) \;=\; \arg\min_{\substack{\gamma : a \to b \\ \mathrm{len}(\gamma) \le L_0}} \Big( \int_\gamma s\,d\ell \;+\; \mu \max_\gamma s \Big).
$$

For any two paths $\gamma_1, \gamma_2$ with $(\int_i, \max_i)$, the crossover is $\mu^* = (\int_1 - \int_2)/(\max_2 - \max_1)$ whenever the denominator is positive: the exact price at which aggregationism must yield to peak aversion. A revealed-preference reading: choosing the aggregative path over the peak-averse one is equivalent to asserting that one unit of peak suffering is worth less than $1/\mu^*$ units of accumulated suffering — a computable threshold to defend, not a moral appeal.

### 3.4 The anti-Goodhart axiom

**Axiom (target feasibility).** *A path that does not reach the target $b$ is infeasible, regardless of cost.*

This is an axiom of the method, not a caveat appended to it. The objective is not $\min(\text{suffering})$ but $\min(\text{gratuitous suffering})$ **subject to reaching the therapeutic state**. The target constraint is what blocks the sedative solution: sedation, avoidance, and iatrogenic blunting all lower measured suffering, and all are infeasible under the axiom because they do not reach $b$. In the framework's terms, Goodhart's law is not a pathology of the optimizer but of the objective's feasible set; the repair is to put the clinical endpoint inside the constraint.

### 3.5 DDI gates as Goodhart constraints

Edge admission can itself be gated: an edge $(u, v)$ is admitted only if an external predicate $g(u, v)$ (evaluated at graph-construction time on recomputed quantities) holds. In the clinical integration (§6), transitions into the verified-therapy target exist only out of post-TDM states whose pharmacokinetic safety gate passes on the current concentration band. Gates are the structural twin of the anti-Goodhart axiom: they make certain low-suffering shortcuts — skipping verification, co-administering a contraindicated drug — *topologically absent* rather than merely expensive.

---

## 4. Algorithm

For small graphs the scheduler is exact by construction: exhaustive enumeration of simple paths via breadth-first search, with per-path cost evaluation and Pareto-frontier extraction.

**Enumeration.** A BFS queue stores partial paths (vertex, path, length). A partial path is extended along every outgoing edge whose head is not already in the path (simplicity). Completed paths reaching $b$ within budget are scored. The algorithm returns the argmin under $\mathrm{cost}(\cdot;\mu)$ or reports $\textsf{INFEASIBLE}$.

**Pareto frontier.** Over all feasible paths, the frontier $\{(\int_\gamma s\,d\ell,\ \max_\gamma s)\}$ is extracted by removing dominated points: $(i_1, p_1)$ dominates $(i_2, p_2)$ iff $i_1 \le i_2 \land p_1 \le p_2$ with at least one strict inequality.

**Complexity honesty.** The number of simple paths grows factorially in the worst case; the fixed-size Sounio implementation caps at 16 states, 64 edges, 256 queued paths, and path length 32. This is an exact reference algorithm for benchmark-scale graphs, not a claim of a polynomial-time method. Scaling is future work (§7.5).

**Dual implementation.** The same scheduler exists as (i) a pure-Python contract (`scripts/research/mercyful_runtime_contract.py`, standard library only) and (ii) a native Sounio module (`stdlib/clinical/mercyful.sio`) executed through the Sounio compiler. Both implementations agree on all benchmark numbers reported below; the Sounio port uses a flat path-queue layout to work around a bootstrap-compiler array-lowering defect, a detail documented in the module header.

---

## 5. Benchmark: the synthetic exposure-therapy patient

### 5.1 Setup

Four states with unit edge lengths:

| State | Suffering $s$ | Interpretation |
|---|---|---|
| `avoidance` | 0 | Untreated; the Goodhart trap |
| `mild` | 2 | Mild exposure distress |
| `moderate` | 5 | Moderate exposure distress |
| `recovery` | 0 | Therapeutic target |

Edges: `avoidance→avoidance` (self-loop: staying untreated), `avoidance↔mild`, `mild↔moderate`, `moderate→recovery`. The only route to recovery passes through moderate distress.

### 5.2 The naive minimizer prescribes the pathology

An unconstrained raw-suffering minimizer (cycles allowed) achieves cost **0** by looping `avoidance→avoidance` forever. It never reaches recovery. This is the anti-Goodhart demonstration: *minimizing measured suffering, unconstrained, prescribes avoidance — the disorder itself.* (Contract clause M3: the best target-constrained path has integrated suffering 7 > 0, so the unconstrained optimum is strictly cheaper than any feasible plan.)

### 5.3 The mercyful scheduler traverses distress and recovers

Under the target constraint, the unique simple path to recovery is $\gamma = \text{avoidance} \to \text{mild} \to \text{moderate} \to \text{recovery}$, with exact metrics:

$$
\mathrm{len} = 3,\qquad \int_\gamma s\,d\ell = 0 + 2 + 5 = 7,\qquad \max_\gamma s = 5,\qquad \mathrm{cost}(\gamma;\mu) = 7 + 5\mu .
$$

At $\mu = 1$: total cost **12**. The budgetary functionals are

$$
\Psi(L_0) = \begin{cases} \text{infeasible} & L_0 < 3 \\ 7 & L_0 \ge 3 \end{cases}
\qquad
c^*_{\mathrm{orc}}(L_0) = \begin{cases} \text{infeasible} & L_0 < 3 \\ 5 & L_0 \ge 3 \end{cases}
$$

so the chosen path has gratuitous suffering $5 - 5 = 0$: it achieves mercy exactly. The distress it traverses is *necessary* in the budgetary sense — no path within any budget reaches recovery without a peak of 5 — and the framework says so as a theorem of the graph, not as a clinical intuition.

### 5.4 The functional choice is computable, not rhetorical

A second synthetic graph (contract clause M2) exhibits a genuine trade-off: path $P_1 = S\!\to\!A\!\to\!T$ has $(\int s, \max s) = (5, 5)$ and length 2; path $P_2 = S\!\to\!B\!\to\!C\!\to\!D\!\to\!T$ has $(6, 2)$ and length 4. Neither dominates the other; the Pareto frontier is exactly $\{(5,5), (6,2)\}$. The crossover is $\mu^* = (6-5)/(5-2) = 1/3$: below $\mu = 1/3$ the scheduler aggregates, above it pays the longer, gentler route. A third graph (clause M5) has $(4,4)$ vs $(6,3)$, crossover $\mu^* = 2$; at $\mu = 0$ the scheduler selects the high-peak path and at $\mu = 10$ the low-peak path — demonstrating that $\mu$ is wired into the decision rule and that selected peak weakly decreases with $\mu$.

### 5.5 Budget hardness

On a graph with $\mathrm{len} = 4$ and $L_0 = 3$, the scheduler returns $\textsf{INFEASIBLE}$ — not a cheaper non-target path (clause M6). The budget constraint and the target constraint are both hard.

---

## 6. Clinical integration: suffering fields from Knightian PK bands

The exposure benchmark hand-sets the suffering field. The integration layer computes it from pharmacokinetic (PK) digital twins under Knightian (imprecise-probability) uncertainty, and lets drug–drug interaction (DDI) gates prune the graph itself.

### 6.1 From PK bands to suffering

For a state representing a dosing regimen, the twins return concentration *p-boxes* $[\mathrm{lo}, \mathrm{hi}]$ — Fréchet outer enclosures valid for any joint distribution of the PK parameters (monotone-corner arguments; [12]). The suffering of a regimen against a window $[a, b]$ is the sum of the normalized worst-case violations:

$$
s_{\mathrm{win}}([\mathrm{lo},\mathrm{hi}], [a,b]) \;=\; \frac{\max(0,\ a - \mathrm{lo})}{a} \;+\; \frac{\max(0,\ \mathrm{hi} - b)}{b},
$$

the first term pricing worst-case sub-therapeutic shortfall (efficacy risk), the second worst-case supra-therapeutic exceedance (toxicity risk). A band fully inside the window contributes zero. The state suffering aggregates three terms:

$$
s(\text{state}) \;=\; s_{\mathrm{Cmin}} \;+\; s_{\mathrm{AUC}} \;+\; \tfrac{1}{2}\, s_{\mathrm{peak}}(\mathrm{Cmax}_{\mathrm{hi}}, \text{ceiling}),
$$

using (i) the twins' public steady-state trough bands; (ii) a per-interval AUC enclosure $\mathrm{AUC} \in [F_{\mathrm{lo}}D/\mathrm{CL}_{\mathrm{hi}},\ F_{\mathrm{hi}}D/\mathrm{CL}_{\mathrm{lo}}]$ (AUC is increasing in $F$ and decreasing in $\mathrm{CL}$, so the corners are exact); and (iii) a *sound but loose* Cmax proxy $\mathrm{Cmax}_{\mathrm{hi}} \le \mathrm{Cmin}_{\mathrm{hi}} + D/V_{c,\mathrm{lo}}$ via the exact one-compartment identity $\mathrm{Cmax}_{ss} = \mathrm{Cmin}_{ss} + D/V_c$ and subadditivity of the max. We explicitly do not claim Fréchet tightness for the Cmax term.

A regimen that violates a twin's structural contract (vacuous p-box, e.g., dose 5000 mg above the 4000 mg structural cap) is assigned $s = S_{\max} = 100$ — measured to be roughly 40× the worst in-contract suffering under the synthetic windows — so contract-violating regimens are near-prohibitive without being topologically impossible.

### 6.2 Scenario graph and gates

Seven states: `START` (untreated, $s=0$ — the Goodhart trap), `VANCO_PRE`/`VANCO_POST` (1000 mg q12h vancomycin, pre-/post-TDM), `TAC_PRE`/`TAC_POST` (5 mg q12h tacrolimus, pre-/post-TDM), `BAD_DOSE` (contract violation, $s = S_{\max}$), and `TARGET` (dual therapy verified, synthetic co-therapy burden $s = 0.1$). Edges include the `START→START` trap, and transitions into `TARGET` only from post-TDM states — **there is no `START→TARGET` edge** — each admitted iff the corresponding twin safety gate passes on the recomputed band (`G_VERIFY`). Two DDI gates act on edge admission: `G_NEPHROTOXIN` (a synthetic nephrotoxic co-medication flag removes all edges into vancomycin-active states) and `G_CYP` (a synthetic CYP3A4-inhibitor flag scales tacrolimus clearance by ×0.5, recomputed at admission time).

### 6.3 Measured results (exact, reproducible)

All numbers below are printed by `tests/run-pass/mercyful_clinical_sequencing.sio` (six-clause contract, all PASS):

- **TDM narrows the field (C3).** Pre-TDM vs post-TDM suffering: vancomycin $0.675679 \to 0.059420$; tacrolimus $1.251592 \to 0.000000$. Bayesian band narrowing lowers the suffering field strictly for both drugs.
- **Contract violation priced (C4).** The 5000 mg regimen maps to $S_{\max} = 100$ and is avoided.
- **Healthy scenario (C1).** The scheduler selects the vancomycin route `START→VANCO_PRE→VANCO_POST→TARGET`: length 3, integrated suffering $0.735099$, peak $0.675679$, total $1.410778$ at $\mu = 1$ — positive integral suffering (the path must traverse a pre-TDM state) and total far below the $S_{\max}$ route. The tacrolimus route (integral $1.251592$) is feasible but not selected; the choice is a computed cost comparison, not a clinical preference.
- **Unverified shortcut refused (C2).** A graph whose only conceivable target route is an unverified shortcut yields `found = false`.
- **Nephrotoxin DDI (C5).** With the flag active and only the vancomycin route present: INFEASIBLE; without the flag the same graph is feasible — the gate is causal.
- **CYP3A4 DDI (C6).** Under the inhibitor flag the tacrolimus AUC enclosure doubles exactly: post-TDM $\mathrm{AUC}_{\mathrm{hi}} \approx 172 \to 344.466490$ ng·h/mL, crossing the synthetic ceiling 200. The gate blocks the transition and the tacrolimus-only route is INFEASIBLE — evaluated on the recomputed band, not a hardcoded verdict.

The point of §6 is methodological, not clinical: the suffering field can be *derived from epistemically honest uncertainty representations* (p-boxes whose validity holds for any parameter dependence), and safety gates can act on the graph's topology, turning Goodhart-vulnerable shortcuts into infeasibility rather than into low cost.

---

## 7. Discussion

### 7.1 Necessary vs. gratuitous suffering, clinically read

The framework formalizes the distinction between palliating and treating with a computable criterion: mercy is attaining $c^*_{\mathrm{orc}}(L_0)$, not avoiding it. Read against exposure therapy, the budgetary form is *more* clinically faithful than the topological one it replaced: no clinician claims exposure is the only conceivable route to recovery; the claim is that it is the only route that works within human time and resource constraints. That is a budget constraint, not a topological obstruction — and it is what the formalism now says.

### 7.2 The two mercies

Path length is update steps is computation: the thermal and energetic cost of the substrate. On a real field, *efficiency vs. suffering* **is** *mercy to the substrate vs. mercy to the state*. The exchange rate between these two incommensurable costs is not a stipulation of the author; it is a property of the instance — read off the frontier $\Psi(L_0)$, which answers: *how much integrated suffering is unavoidable under a maximum computation budget?* A principle that says "minimize suffering" is empty; one that forces the pricing of two sufferings and derives the exchange rate from the geometry of the problem is a contribution. We note again that this is a budgetary claim; no phenomenological claim about the substrate is made or needed.

### 7.3 Clinical relevance without clinical claims

The intended audience is computational psychiatry, clinical pharmacology methodology, and ML ethics. What a clinical reader can take: (i) the aggregation/peak choice is an ethical commitment that current schedulers make *implicitly* by summing; (ii) suffering measured without a target constraint is Goodhart-vulnerable by construction, and the constraint must live in the feasible set; (iii) TDM-style information gathering has a computable suffering-field signature (band narrowing strictly lowers the field). What they cannot take: any dosing, sequencing, or therapeutic recommendation. The numbers in §5 and §6 are exact statements about synthetic graphs and synthetic p-boxes, nothing more.

### 7.4 Epistemic ledger

| Claim | Status |
|---|---|
| The scheduler is exact on small graphs and returns a feasible path or INFEASIBLE | **Proven** by exhaustive enumeration; tested (M1, M6) |
| The raw minimizer avoids recovery (cost 0) while the mercyful scheduler reaches it (cost $7+5\mu$) | **Measured** (M3, M4; exact arithmetic on a 4-state graph) |
| Pareto frontiers and $\mu$-crossovers are computed exactly | **Proven** by enumeration on the tested graphs; **measured** (M2, M5) |
| TDM band narrowing strictly reduces the suffering field; DDI gates cause infeasibility | **Measured** (C1–C6, exact printed values) |
| The Cmax term is a sound outer bound | **Proven** (exact identity + subadditivity); tightness **not claimed** |
| The p-boxes are valid for any joint parameter distribution | Inherited from the twins' monotone-corner enclosures [12]; **not re-proven here** |
| Budgetary necessity ($c^*_{\mathrm{orc}}$) is the right formalization of necessary suffering | **Position** — argued, not proven; the topological alternative was falsified [15] |
| Learning under a suffering field is feasible and beneficial | **Conjectural** — out of scope; see §7.5 |
| Any clinical or patient-level conclusion | **Not claimed** |

### 7.5 Limitations

1. **Synthetic everything.** No patient data, no real suffering measurements, no clinical validation.
2. **Small graphs.** Exhaustive enumeration is factorial; 16-state cap in the native implementation. Scaling to realistic treatment graphs needs approximate or learned methods that do not yet exist.
3. **No learning.** The "Learning" in the title names the research program, not this artifact. The scheduler is combinatorial. Seven prior implementation attempts in the parent program produced controlled negatives [15]; one structural lesson survives: composition of experiences must be *multiplicative* for annihilation to be representable, which rules out pure SGD, summed regularizers, and additive loss aggregation as candidate substrates — a constraint that guides, but does not deliver, a learning algorithm.
4. **The suffering field is assumed given.** Eliciting a defensible suffering field from instruments, panels, or preference studies is an open measurement problem this paper does not address.
5. **The budget $L_0$ is a modeler's choice.** Necessity is relative to a declared resource (§3.2); the framework makes the choice visible but cannot remove it.
6. **Static graphs.** The field and topology are fixed at construction; TDM appears only as distinct pre/post states. Dynamic re-planning under streaming measurements is future work.
7. **Single agent, single patient.** No multi-patient fairness analysis, no clinician-in-the-loop protocol.

### 7.6 On novelty: what the framework is and is not

An external adversarial review (Grok 4.3, see AI disclosure) characterized the core accurately and we adopt its framing: the construction *is* a constrained shortest-path problem — equivalently, a deterministic constrained MDP with a hard goal constraint; the peak term *is* a classic bottleneck cost; and the anti-Goodhart axiom *is* the statement that the feasible set must contain only goal-reaching trajectories, an idea present in safe RL, constrained MDPs, and goal-conditioned planning. **No new algorithmic primitive is claimed.** What we take the contribution to be, stated as narrowly as the evidence supports: (i) naming the functional choice (∫ vs. max) as the locus of the ethical commitment, with an exactly computable crossover price $\mu^*$; (ii) the budgetary reformulation of necessary suffering, which survives the falsification of its topological predecessor and makes quantitative rather than binary predictions; (iii) placing the Goodhart defense in the feasible set rather than in post-hoc penalties, including its topological form (DDI gates as edge deletion); and (iv) an executable, dual-implementation benchmark with pre-registered falsifiers, in which every reported number reproduces. The interpretive vocabulary — mercy, gratuitous suffering, the two mercies — is overlay on these elementary facts, offered as a framing for clinical and ML-ethics readers, not as mathematics.

---

## 8. Falsifiers and pre-registered stop rules

Each contract clause carries a falsifier and a stop rule (from the runtime falsifiers document [16], registered before the runs reported here):

| Clause | Falsifier | Stop rule |
|---|---|---|
| M1 well-definedness | Scheduler crashes or returns an invalid path | Core search broken |
| M2 frontier | Computed frontier misses a known point or includes a dominated one | Bi-criteria optimizer wrong |
| M3 anti-Goodhart | Unconstrained minimizer *does* reach recovery | Toy too easy; hazard not demonstrated |
| M4 exposure selected | Mercyful scheduler avoids `moderate` / misses recovery | Constraint or trade-off not encoded |
| M5 $\mu$-continuity | Higher $\mu$ selects strictly higher peak | $\mu$ not wired into the decision rule |
| M6 budget hardness | Non-target path returned under tight budget | Budget constraint not hard |
| C1–C6 clinical integration | Any of: no path in healthy scenario; shortcut admitted; TDM does not strictly reduce field; violation not priced at $S_{\max}$; DDI gate non-causal; CYP band not recomputed | Integration layer unsound |

Global verdicts: failure of M1, M3, or M6 is **M_RED** (scheduler unsafe to use; benchmark fails to demonstrate the phenomenon). Failure of M2, M4, M5, or any C-clause is **M_AMBER** (fix the specific clause before claiming). Current status: **M_GREEN (6/6)** and **clinical integration 6/6**, reproduced for this preprint on 2026-07-26 (§9).

---

## 9. Reproducibility

All results are executable from the repository:

```bash
# Python contract: M1..M6, verdict M_GREEN
python3 scripts/research/mercyful_runtime_contract.py

# Sounio-native exposure-therapy benchmark: MERCYFUL_SOUNIO_PASS
bin/souc run tests/run-pass/mercyful_exposure_therapy.sio

# Clinical integration (Knightian PK bands, DDI gates): C1..C6 PASS
scripts/dev/run_clinical_twin.sh tests/run-pass/mercyful_clinical_sequencing.sio

# CI gates
bash scripts/ci/mercyful_runtime_gate.sh            # MERCYFUL_RUNTIME_GATE_OK
bash scripts/ci/mercyful_sounio_gate.sh             # MERCYFUL_SOUNIO_GATE_OK
bash scripts/ci/mercyful_clinical_sequencing_gate.sh # MERCYFUL_CLINICAL_SEQ_GATE_OK
```

Exact reported values (2026-07-26, this branch): exposure benchmark $\int s = 7$, peak $5$, cost $12$ at $\mu=1$; clinical integration — vancomycin field $0.675679 \to 0.059420$, tacrolimus field $1.251592 \to 0$, selected route integral $0.735099$, peak $0.675679$, total $1.410778$, CYP-inhibited $\mathrm{AUC}_{\mathrm{hi}} = 344.466490$ vs ceiling 200.

---

## 10. Conclusion

Treatment sequencing optimizes what it is told to optimize. Told to minimize measured suffering, it prescribes sedation and avoidance; told to maximize expected outcome, it hides catastrophic peaks inside sums. Mercyful Learning states the alternative as a formal object: a suffering field, two functionals whose choice is an explicit ethical commitment with a computable crossover price, a budgetary notion of necessary suffering, and an anti-Goodhart axiom that puts the therapeutic target inside the feasible set. On synthetic benchmarks the object does exactly what it is built to do — it walks the patient through the necessary distress and declines the gratuitous kind — and its clinical integration shows the field can be derived from Knightian PK uncertainty with DDI gates acting on graph topology. The framework is proven where it can be proven, measured where it is measured, and silent where it has no evidence. The learning algorithm that would justify the program's name has not been found; seven controlled negatives bound the search space, and the multiplicative-composition constraint narrows it further.

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
12. Ferson S, Kreinovich V, Ginzburg L, Myers DS, Sentz K. Constructing probability boxes and Dempster–Shafer structures. Sandia Report SAND2002-4015, 2003.
13. Rybak MJ et al. Therapeutic monitoring of vancomycin: a revised consensus guideline. *Am. J. Health-Syst. Pharm.* 77:835–864, 2020. *(Shapes a synthetic window; not a target claim.)*
14. Astellas Pharma. Prograf (tacrolimus) prescribing information. *(Shapes a synthetic window; not a target claim.)*
15. Agourakis DC. Program registry — Mercyful Learning, §A (principle, budgetary reformulation, anti-Goodhart axiom) and §5 (controlled negatives). `docs/research/PROGRAM-REGISTRY-mercyful-learning.md`, this repository, 2026.
16. Agourakis DC. Mercyful Learning runtime spec, falsifiers, Sounio port spec, and clinical integration spec. `docs/research/mercyful_runtime_spec_2026-07-25.md`, `mercyful_runtime_falsifiers_2026-07-25.md`, `mercyful_sounio_port_spec_2026-07-25.md`, `mercyful_clinical_integration_spec_2026-07-25.md`, this repository, 2026.

---

## AI disclosure (GAIDeT-ICMJE 2025)

This preprint was drafted under human direction with AI assistance (drafting, code inspection, and numeric verification of the reported contract outputs), and was reviewed by an external LLM provider (xAI Grok 4.3) via the repository's mandatory offload-review policy before circulation; two further providers (DeepSeek, Google Gemini) were attempted and failed at the provider level, so the review is logged as degraded single-provider with re-review flagged. The review's main critique — that the core is elementary constrained shortest path — was adopted and is addressed explicitly in §7.6. All AI-generated content was verified by the author against executable artifacts. No clinical or patient-level claim is made. The author takes full responsibility for the content.
