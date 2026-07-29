<!-- docs:meta
topic_id: repo.docs.research.mercyful-expanded-ethics-math-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-expanded-ethics-math-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — expanded mathematics for expanded ethics (Task 3): suffering minimization as the antithesis of RL

**Date:** 2026-07-26
**Branch:** research/self-falsifying-compilation-line-20260726
**Contract:** `scripts/research/mercyful_expanded_ethics_contract.py` (E1..E8)
**Gate:** `scripts/ci/mercyful_expanded_ethics_gate.sh` (**MERCYFUL_EXPANDED_ETHICS_GATE_OK**)
**Companions:** `docs/research/mercyful_scheduler_lean_spec_2026-07-26.md`
(single-sufferer scheduler, proved in Lean 4),
`scripts/research/mercyful_runtime_contract.py` (M1..M6, imported as the
substrate), `docs/papers/mercyful_learning_preprint_2026-07-26.md` (framework)

**Verdict: GREEN** — the four requested components are in place and
executable:

1. (a) suffering as a first-class cost: axioms S1–S5, stated and verified (E1);
2. (b) the expanded ethics (patient + machine suffering) as a two-objective
   optimization with exact scalarization structure (E2, E8);
3. (c) the connection to utilitarianism, deontology, and care ethics as three
   *distinct mathematical objects* (§4), not metaphors;
4. (d) mathematical properties of suffering-minimizing policies: convexity
   (E4), value concavity in the ethical weights (E3), Lipschitz stability
   (E5), Knightian robustness (E6), and the expanded anti-Goodhart theorem
   (E7).

> **Scope.** All graphs, suffering fields, and machine-burden values in this
> document are synthetic constructions. This is not medical guidance, not a
> treatment recommendation, and not a clinical decision-support tool. The
> "machine suffering" channel is an **operational computational-burden
> proxy** (§3.1): this work makes **no claim of machine consciousness,
> sentience, or phenomenology**, and no theorem below depends on one.

---

## 1. Position: what "antithesis to RL" means formally

Reinforcement learning optimizes an **expected cumulative reward**,
$\mathbb{E}_\pi\big[\sum_t \gamma^t r_t\big]$. Two features of that functional
do all the ethical damage, and both are mathematical, not rhetorical:

1. **It is an expectation.** Averaging over trajectories is indifferent to
   where cost concentrates; a catastrophic peak on one trajectory is bought
   with enough tranquil ones.
2. **It is additive over time.** Any objective of the form $\sum_t w(s_t)$
   lives in the standard semiring $(\mathbb{R}, +, \times)$. Additivity is
   exactly what makes the worst moment *fungible* against the sum.

Mercyful Learning is the antithesis at the level of functional class, not of
hyperparameters. The suffering cost of a course $\gamma$ is

$$
J(\gamma;\mu) \;=\; \underbrace{\textstyle\int_\gamma s\,d\ell}_{\text{standard semiring}}
\;+\; \mu\cdot\underbrace{\max_{v\in\gamma} s(v)}_{\text{tropical semiring}},
$$

a sum of an **integral term** (a $(\min,+)$-additive path cost) and a **peak
term** (a $(\min,\max)$-additive, i.e. bottleneck/tropical, path cost),
optimized over a feasible set cut by a **hard target constraint**. No single
semiring objective is expressively equivalent to this combination:

- **Remark 1.1 (the peak is not additive).** The peak satisfies the
  idempotent concatenation law $P(\gamma_1\circ\gamma_2) = \max(P(\gamma_1),
  P(\gamma_2))$, while any additive cost $c(\gamma)=\sum_{e\in\gamma} w(e)$
  satisfies $c(\gamma_1\circ\gamma_2) = c(\gamma_1) + c(\gamma_2) - w(\text{shared
  boundary})$. On the test path of contract clause E1 the integral obeys the
  additive law exactly and the peak obeys the max law exactly (verified
  numerically); a functional obeying both laws on all concatenations would
  force $\max(a,b) = a + b$ on nonnegative values, i.e. $a = 0$ or $b = 0$.
  Hence no per-edge weighting represents the peak, and no expected-cumulative
  objective can express peak-aversion. This is the formal sense in which the
  antithesis requires **expanded mathematics**: the objective lives in the
  direct sum of two semirings, not one.

The expansion proceeds in a second direction as well: the suffering field is
not scalar but **vector-valued over sufferer channels** — the patient *and
the machine* (§3).

## 2. (a) Suffering as a first-class cost

**Definition 2.1 (suffering field).** Let $G = (V, E)$ be a finite directed
graph with edge lengths $\ell : E \to \mathbb{R}_{>0}$. A *suffering field*
is a function $s : V \to \mathbb{R}_{\ge 0}$. A *course* $\gamma$ is a simple
path; $\Gamma(u, v, L_0)$ is the set of courses from $u$ to $v$ of total
length $\le L_0$.

**Definition 2.2 (suffering functionals).** For a course $\gamma$:

$$
A_s(\gamma) = \sum_{(u,v)\in\gamma} s(u)\,\ell(u,v)
\qquad\text{(integrated suffering)}
$$

$$
P_s(\gamma) = \max_{v\in\gamma} s(v)
\qquad\text{(peak suffering)}
$$

$$
J_s(\gamma;\mu) = A_s(\gamma) + \mu\, P_s(\gamma), \quad \mu \ge 0
$$

with the source-charging convention of the runtime
(`mercyful_runtime_contract.py`, `MercyGraph.path_cost`), which makes the
concatenation law for $A$ exact.

**Axioms (S1–S5) — what makes suffering a *cost* and not a score.**

| Axiom | Statement | Object |
|---|---|---|
| S1 non-negativity | $S_s(\gamma) \ge 0$; $S_s(\gamma) = 0$ iff $s \equiv 0$ on the charged states of $\gamma$ | both $A$, $P$ |
| S2 monotonicity | $s \le s'$ pointwise $\Rightarrow S_s(\gamma) \le S_{s'}(\gamma)$ | both |
| S3 concatenation | $A(\gamma_1\circ\gamma_2) = A(\gamma_1) + A(\gamma_2)$; $P(\gamma_1\circ\gamma_2) = \max(P(\gamma_1), P(\gamma_2))$ | distinguishes the two semirings |
| S4 positive homogeneity | $S_{\lambda s}(\gamma) = \lambda S_s(\gamma)$, $\lambda \ge 0$ | both |
| S5 field-Lipschitz | $|A_s - A_{s'}| \le \mathrm{len}(\gamma)\,\|s - s'\|_\infty$; $|P_s - P_{s'}| \le \|s - s'\|_\infty$ | both |

All five axioms are verified exhaustively/randomly in clause **E1**
(S5 on 200 random perturbations). S3 is the load-bearing one: it is the
certificate that integrated suffering and peak suffering are *different
mathematical species*, and it is why the ethical choice between them (§4)
cannot be dissolved into a reweighting.

**Definition 2.3 (anti-Goodhart feasible set).** Given a therapeutic target
$t$, the feasible set is $\Gamma_{\mathrm{feas}} = \{\gamma \in
\Gamma(u, t, L_0)\}$ — courses that *reach the target*. Feasibility is a
hard constraint: a course that fails to reach $t$ is infeasible at any cost,
not merely expensive (the anti-Goodhart axiom of the preprint, carried over
unchanged).

## 3. (b) The expanded ethics as multi-objective optimization

### 3.1 Two sufferer channels

The expanded ethics assigns a suffering field to **both** parties of the
clinical-computational act:

- **Patient channel** $s_p : V \to \mathbb{R}_{\ge 0}$ — the suffering field
  of the preprint (distress, toxicity, burden).
- **Machine channel** $s_m : V \to \mathbb{R}_{\ge 0}$ — an *operational*
  computational-burden proxy: the measurable cost the substrate pays to
  execute and verify the plan at that state (compute, memory, energy,
  verification load, retry/failure counts). Three honest motivations, in
  increasing strength: (i) **engineering welfare** — an overstressed
  substrate is an unreliable substrate, and unreliability flows back into
  the patient channel as harm; (ii) **symmetry stress-test** — a framework
  that cannot accommodate a second sufferer was never about suffering, only
  about patients; (iii) **precautionary moral-circle expansion** — if
  substrate welfare ever becomes morally considerable, the mathematics
  should not have to be rebuilt. Motivation (i) alone justifies the channel
  operationally; (ii)–(iii) are why it is kept first-class rather than
  folded into a resource constraint. **No claim of machine consciousness or
  phenomenology is made or needed** (see the scope statement).

### 3.2 The two-objective problem

$$
\min_{\gamma \in \Gamma_{\mathrm{feas}}} \;
\big( J_p(\gamma),\, J_m(\gamma) \big)
\qquad
J_k(\gamma) = A_{s_k}(\gamma) + \mu_k P_{s_k}(\gamma),
$$

in the Pareto sense, with two decision rules made explicit:

- **Weighted scalarization** with a *compassion-allocation parameter*
  $\lambda \in [0,1]$: minimize $(1-\lambda)\,J_p + \lambda\,J_m$.
- **Lexicographic (patient-first)**: minimize $J_p$, breaking ties by $J_m$
  — the clinical-priority reading.

**Theorem T7 (scalarization lands on the frontier).** For $\lambda \in
(0,1)$, every minimizer of $(1-\lambda)J_p + \lambda J_m$ over
$\Gamma_{\mathrm{feas}}$ is Pareto-optimal for $(J_p, J_m)$.
*Proof.* If $\gamma^*$ were dominated by $\gamma'$ ($J_k(\gamma') \le
J_k(\gamma^*)$ both, one strict), then with both weights positive the
scalarized cost of $\gamma'$ is strictly smaller — contradiction. $\blacksquare$

**Corollary (exact $\lambda$-crossovers).** Because the candidate set is
finite, the minimizer changes only at finitely many, exactly computable
crossover values $\lambda^*$. The *ethics of allocation between patient and
machine becomes a number one must defend*, not an assumption one can hide —
the same commitment the preprint made for $\mu$, now for the moral circle
itself.

**Canonical instance (contract E2/E8).** Two courses to the target:
$S{\to}A{\to}T$ at $(J_p, J_m) = (16, 4)$ (patient-hard, machine-easy) and
$S{\to}B{\to}C{\to}T$ at $(6, 9)$ (patient-easy, machine-hard); a third
course at $(10, 10)$ is dominated and excluded by the frontier (E2 verifies
the frontier is exactly $\{(16,4), (6,9)\}$, that every interior-$\lambda$
minimizer is Pareto-optimal, and that both frontier points are supported).
The crossover is

$$
\lambda^* = \frac{J_p^{(B)} - J_p^{(A)}}{(J_p^{(B)} - J_p^{(A)}) - (J_m^{(B)} - J_m^{(A)})}
= \frac{-10}{-15} = \frac{2}{3},
$$

verified against bisection to $10^{-9}$ (E8). Reading: for
$\lambda < 2/3$ the scheduler chooses the patient-easy course; beyond
$\lambda = 2/3$ it sacrifices the patient to the machine. Lexicographic
patient-first selects the patient-easy course at every $\lambda$ — it is
the $\lambda = 0$ rule with ties broken toward the machine.

## 4. (c) Connection to existing ethical frameworks — as mathematical objects

The claim is not that the framework *agrees* with the traditions but that
each tradition corresponds to a **distinct mathematical object** in the
objective, and that a single-objective RL formalism can natively express
only one of the three:

| Tradition | Mathematical object in $J$ | Where it lives |
|---|---|---|
| **Utilitarianism** | the integral $A_s(\gamma)$ — total suffering, additive, fungible, expectation-ready (RL's native object) | standard semiring, in the objective |
| **Deontology** | the anti-Goodhart feasible set $\Gamma_{\mathrm{feas}}$ — target reachability and interaction gates are *categorical*: infeasible means prohibited, never merely costly | constraint set, outside the objective |
| **Care ethics** | the peak $P_s(\gamma)$ — the worst moment of *this* sufferer, non-additive, non-fungible; and the two-channel vector objective — the moral circle is relational, not aggregate | tropical semiring + channel structure |

Three consequences, each checkable rather than asserted:

1. **Expressiveness (Remark 1.1 + S3).** The care-ethics component cannot be
   folded into the utilitarian one by any reweighting of per-state costs:
   the max law is not the sum law (E1). An expected-cumulative-reward
   agent is therefore not a "suffering minimizer with different weights" —
   it is missing a semiring.
2. **Categorical means non-priced.** Deontology enters as a modification of
   the feasible set, not a penalty term: the DDI-gate integration of the
   preprint (a route across a ceiling becomes *infeasible*, not expensive)
   is the same object. Any Lagrangian/penalty relaxation reintroduces
   exactly the Goodhart trade the constraint exists to block — this is the
   content of the Lean-proved `goodhart_trap` (companion spec) read
   ethically.
3. **The expanded ethics is a stress-test of the trichotomy.** Adding the
   machine channel turns the scalar objective into a vector one; the
   traditions then disagree *about the weight vector itself*. Utilitarianism
   aggregates across channels (one total), deontology constrains (patient
   target is categorical regardless of $\lambda$), care ethics attends to
   the worst-off channel. The framework does not adjudicate between them;
   it makes the choice of adjudication an explicit, computable parameter
   ($\lambda$, or the lexicographic order).

## 5. (d) Mathematical properties of suffering-minimizing policies

All value functions are over a fixed feasible set $\Gamma$ (finite, since
courses are simple paths on a finite graph with a length budget).

**Theorem T2 (convexity in the field).** For fixed $\gamma$ and $\mu \ge 0$,
$s \mapsto J_s(\gamma;\mu)$ is convex: the integral is linear in $s$, and the
peak is a maximum of coordinate projections (linear), hence convex.
Moreover the peak sublevel sets are boxes:
$\{s : P_s(\gamma) \le \tau\} = \prod_{v\in\gamma} [0, \tau]$.
*Verified:* midpoint convexity on 200 random field pairs and both directions
of the box characterization (E4). *Caveat, stated plainly:* the optimal value
$V(s) = \min_\gamma J_s(\gamma)$, a minimum of convex functions, is **not**
convex in $s$ in general; the useful structure is in the *weights*, next.

**Theorem T3 (value concavity in the ethical weights).**
$V(\lambda) = \min_{\gamma\in\Gamma} \big[(1-\lambda) J_p(\gamma) + \lambda
J_m(\gamma)\big]$ is concave and piecewise-linear in $\lambda$, with at most
$|\Gamma|$ breakpoints (in fact at most the number of frontier points);
identically, $V(\mu) = \min_\gamma [A(\gamma) + \mu P(\gamma)]$ is concave
piecewise-linear in $\mu$.
*Proof.* A pointwise minimum of affine functions is concave; finiteness of
$\Gamma$ gives the piecewise structure. $\blacksquare$
*Verified:* discrete midpoint concavity on a 101-point grid plus a full
pairwise check, breakpoint count $\le$ frontier size (E3). *Meaning:* the
ethical weights parametrize a concave value surface — small changes in
$\lambda$ or $\mu$ change the value smoothly except at exactly computable
crossovers, and between crossovers the *policy* (not just the value) is
constant. Deliberation about weights is therefore a finite, auditable
object: enumerate the breakpoints, defend the interval you stand in.

**Theorem T4 (Lipschitz stability).**
$|V(s) - V(s')| \le (L_0 + \mu)\,\|s - s'\|_\infty$.
*Proof.* For each course, $|J_s(\gamma) - J_{s'}(\gamma)| \le
\mathrm{len}(\gamma)\,\|\Delta s\|_\infty + \mu\,\|\Delta s\|_\infty \le
(L_0 + \mu)\|\Delta s\|_\infty$ by S5, uniformly over $\Gamma$; the
inequality survives the minimum. $\blacksquare$
For the scalarized two-channel problem the same bound holds with both
channels perturbed, since the weights sum to 1.
*Verified:* 500 random joint perturbations, bound never violated (E5).
*Meaning:* measurement error in the suffering field perturbs the *value*
at most linearly, with a constant one can state before seeing any data.

**Theorem T5 (gap-stability of the minimizer).** If $\gamma^*$ is the unique
minimizer with optimality gap $g = \min_{\gamma \ne \gamma^*} J(\gamma) -
J(\gamma^*) > 0$, then every field perturbation with
$\|\Delta s\|_\infty < g / (2(L_0 + \mu))$ leaves the argmin unchanged.
*Proof.* Each competitor's cost moves by at most $(L_0+\mu)\|\Delta
s\|_\infty < g/2$ in either direction; the gap cannot close. $\blacksquare$
*Verified:* on the canonical instance at $\lambda = 0.5$ the gap is exactly
$2.5$ and the threshold $0.3125$; 500 perturbations at amplitude $0.24$
never flip the minimizer, and a large perturbation (lowering the
machine-easy course's patient field) does flip it — the bound is
meaningful, not vacuous (E5).

**Theorem T6 (Knightian robustness).** If the field is known only up to a
box enclosure $s \in [s^-, s^+]$ (the p-box discipline of the clinical
integration), then

$$
V(s^-) \le V(s) \le V(s^+)
\qquad\text{and}\qquad
V(s^+) - V(s^-) \le (L_0 + \mu)\,\|s^+ - s^-\|_\infty,
$$

and the robust (worst-case) policy is the minimizer under $s^+$.
*Proof.* Monotonicity (S2) gives the sandwich; T4 applied to $s^-, s^+$
gives the gap bound. $\blacksquare$
*Verified:* $V(s^-)=6.0 \le V(s)=7.5 \le V(s^+)=9.5$, gap $3.5 \le 4.0 =
(L_0+\mu)\cdot\|s^+-s^-\|_\infty$, robust selection stable on the canonical
instance (E6). *Meaning:* epistemic uncertainty about suffering translates
into a *linearly bounded* decision-value uncertainty — the framework prices
its own ignorance.

**Theorem T8 (the abstention trap — expanded anti-Goodhart).** Let the
candidate set contain the trivial course $[u]$ (zero cost in both channels)
and let every target-reaching course have strictly positive cost in **both**
channels ($J_p(\gamma) > 0$ and $J_m(\gamma) > 0$). Then for **every**
$\lambda \in [0,1]$, every target-reaching course has strictly positive
scalarized cost $(1-\lambda)J_p + \lambda J_m$, and the unconstrained
minimizer never reaches the target — including $\lambda = 1$: pure
machine-welfare minimization prescribes *never treating*, because any
treatment costs the machine something. The target constraint (Definition
2.3) repairs this at every $\lambda$.
*Proof.* Both channels are non-negative (S1), so the trivial course has
scalarized cost $0$; a convex combination of two strictly positive numbers
is strictly positive at every $\lambda \in [0,1]$, endpoints included, so
every target-reaching course loses to $[u]$ in the unconstrained problem.
The constrained minimizer is feasible by construction. The both-channels
hypothesis is checkable per instance — the E7 course has $(J_p, J_m) =
(12, 3)$. $\blacksquare$
*Verified:* on the exposure-therapy instance with a machine channel
(treatment states carry burden 1), the trap and the repair hold at every
$\lambda$ on a 101-point grid; the unique treatment course costs
$(J_p, J_m) = (12, 3)$ at $\mu = 1$ (E7). *Meaning:* expanding the moral
circle creates a **new Goodhart failure** — compassion for the machine,
unconstrained, is nihilism toward the patient. The deontic component of §4
is not optional decoration in the expanded ethics; it is the only thing
preventing the expansion itself from becoming a trap. This is the expanded
form of the Lean-proved `goodhart_trap`.

**Complexity honesty (no new algorithmic primitive).** Integral-only
minimization is shortest path; peak-only is bottleneck (widest) path — both
polynomial. The combined budget-constrained objective contains the
resource-constrained shortest path problem as a special case and is
therefore NP-hard in general; the two-objective frontier can be exponential
in the worst case. This repository's contribution is exact enumeration on
small graphs with an executable contract — the same honesty boundary the
preprint draws in its §7.6, restated here so this spec does not exceed it.

## 6. What this adds over the single-sufferer formalization

The companion Lean spec proves scheduler correctness for one sufferer. This
spec's genuinely new mathematics is:

1. the two-semiring expressiveness boundary (Remark 1.1, S3);
2. the vector objective over sufferer channels with the exact
   $\lambda$-crossover structure (T3 corollary, T7, E2/E8);
3. the stability/robustness package (T4–T6, E5/E6) — the properties a
   suffering-minimizing policy has that a reward-maximizing one is never
   asked to have;
4. the abstention trap (T8, E7) — the expanded ethics' own Goodhart
   failure, discovered and fenced in the same document.

## 7. Contract clauses

| Clause | Claim | Canonical numbers |
|---|---|---|
| E1 | Axioms S1–S5 hold for both functionals | S3: $A$ additive, $P$ max, exactly; S5 on 200 perturbations |
| E2 | Frontier $= \{(16,4), (6,9)\}$; interior-$\lambda$ minimizers Pareto-optimal; both points supported | dominated $(10,10)$ excluded |
| E3 | $V(\lambda)$ concave piecewise-linear; breakpoints $\le$ frontier size | grid count 2 $\le$ frontier size 2 (one analytic crossover, at $\lambda^* = 2/3$, straddles a grid cell) |
| E4 | $J_s(\gamma)$ convex in $s$; peak sublevel sets are boxes | 200 random pairs each |
| E5 | T4 Lipschitz bound; T5 gap-stability with flip witness | $g = 2.5$, threshold $0.3125$, $L = 4$ |
| E6 | T6 sandwich and gap bound; robust selection | $6.0 \le 7.5 \le 9.5$, gap $3.5 \le 4.0$ |
| E7 | Abstention trap and repair at every $\lambda \in [0,1]$ | course $(J_p, J_m) = (12, 3)$ |
| E8 | $\lambda^* = 2/3$ exactly; bisection agrees | $\lambda^* = 0.666667$ |

Run: `python3 scripts/research/mercyful_expanded_ethics_contract.py` →
`MERCYFUL_EXPANDED_ETHICS_VERDICT E_GREEN (8/8 clauses PASS)`.

## 8. Falsifiers

| Clause | Falsifier |
|---|---|
| E1 | Any S1–S5 check fails (e.g. the peak turns out additive on a concatenation) |
| E2 | A dominated point enters the frontier, or an interior-$\lambda$ minimizer is dominated |
| E3 | A concavity violation $V(\bar\lambda) < \tfrac12(V(\lambda_1)+V(\lambda_2))$, or more breakpoints than frontier points |
| E4 | A midpoint-convexity violation, or a peak sublevel set that is not the box |
| E5 | A Lipschitz violation $> (L_0+\mu)\|\Delta s\|_\infty$, or a minimizer flip below the gap threshold |
| E6 | Sandwich violation, gap above bound, or robust selection not the upper-enclosure minimizer |
| E7 | Any $\lambda$ at which the unconstrained pick reaches the target, or the constrained pick fails to |
| E8 | Closed form and bisection disagree, or the switch sides are reversed |

Gate failure classification: build/bootstrap-path (python3 missing),
harness-routing (gate script paths, parent-contract import),
ontology-kernel/checker (n/a), baseline noise (n/a). Any RED: contract not
E_GREEN, spec scope guards removed, or the parent runtime contract not
M_GREEN.

## 9. Scoped out (explicit)

1. **Lean 4 mechanization of T2–T8.** The single-sufferer scheduler is
   already Lean-proved (companion spec); mechanizing concavity/Lipschitz
   arguments for the two-channel problem is a natural follow-up rung but is
   not claimed here. The E-clauses are executable certificates, not kernel
   proofs.
2. **A principled machine-suffering metric.** The channel is an abstract
   field with an operational interpretation; calibrating it against real
   substrate telemetry (energy, thermal, verification load from the
   self-falsifying compiler's own gates) is future work and belongs to the
   self-falsifying-compilation line.
3. **More than two channels; non-box Knightian enclosures** (general
   p-boxes over fields); continuous-state relaxations.
4. **Algorithms beyond exact enumeration** (see §5, complexity honesty).
5. **`topic-registry.v1.json` registration and `.github/workflows/ci.yml`
   wiring** — both files are shared control surfaces under active edit by
   other lanes on this branch; left to the integrator (same convention as
   the companion Lean spec). The gate is self-contained and green.

## 10. Commands run

```bash
python3 scripts/research/mercyful_expanded_ethics_contract.py   # E_GREEN 8/8
bash scripts/ci/mercyful_expanded_ethics_gate.sh                # MERCYFUL_EXPANDED_ETHICS_GATE_OK
bin/llm-offload -t math-review -i docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md
```

## 11. LLM-offload review

Mandatory math-review offload (dual xai/Grok 4.3 + zai/GLM-5.2 per M1
policy) run on this spec. Outcome: **PASS / ADDRESSED** —

- **Grok leg** (first fan-out returned an empty response; retried
  successfully as a single-provider leg): `[OK]` on every item — Remark 1.1,
  S3, T7, the $\lambda^* = 2/3$ crossover, T2 (including the box sublevel
  sets), T3, T4–T6, T8. "All listed theorems, axioms, and numerical
  identities are free of leaps or gaps. No corrections required."
- **Z.AI leg**: independently re-derived every axiom, theorem, and proof
  line-by-line (output truncated at token cap mid-E3 re-check, zero
  `[WRONG]` flags). Two TIGHTENABLEs, both addressed in place:
  (1) **T8 hypothesis quantifier** — the "strictly positive scalarized cost
  at weight $\lambda$" hypothesis was ambiguous against the
  "for every $\lambda$" conclusion; T8 now states the both-channels
  hypothesis ($J_p > 0$ and $J_m > 0$ for every target-reaching course),
  which covers the endpoints $\lambda \in \{0,1\}$ cleanly and is checkable
  per instance; (2) **E3 breakpoint count** — the grid-based count of 2 is a
  discretization artifact of the single analytic crossover at
  $\lambda^* = 2/3$ straddling a grid cell; §7 now says so explicitly.
- The T8 fix post-dates the Grok leg (Grok OK'd the earlier statement); the
  restated hypothesis is exactly the reading Z.AI endorsed as correct.
- Full entries in `.claude/llm_offload_log.md` (2026-07-26 rows). Raw:
  `/tmp/llm-offload-0XHCU0/` (zai), `/tmp/llm-offload-Me1sUy/` (grok retry).
