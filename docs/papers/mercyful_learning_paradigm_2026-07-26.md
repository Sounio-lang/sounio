<!-- docs:meta
topic_id: repo.docs.papers.mercyful-learning-paradigm-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.mercyful-learning-paradigm-2026-07-26
-->

# Mercyful Learning: A Suffering-Minimization Paradigm for Machine Learning

**Author:** Demetrios Chiuratto Agourakis
**Date:** 2026-07-26
**Status:** Paradigm paper (preprint draft; arXiv target: cs.LG / cs.CY; secondary cs.AI, stat.ML)
**Provenance:** Formal core in `docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md` (axioms S1–S5, theorems T2–T8); framework preprint `docs/papers/mercyful_learning_preprint_2026-07-26.md`; clinical companion `docs/papers/mercyful_learning_medical_paper_2026-07-26.md`; executable benchmark `scripts/research/mercyful_paradigm_benchmark.py` (contract clauses P1–P8).

> **Scope statement (read first).** This paper defines a *training paradigm* and demonstrates it on synthetic problems. Every patient, dose, need, suffering value, and machine-burden value in this paper is synthetic. Nothing here is medical guidance, a treatment recommendation, or a clinical decision-support tool; no patient data were used. The "machine suffering" channel is an **operational computational-burden proxy** (energy, FLOPs, parameter norm, verification load): this paper makes **no claim of machine consciousness, sentience, or phenomenology**, and no theorem below depends on one. All claims are separated into what is proven, what is measured (on synthetic data), and what is conjectural (§3.7, §8.3).

---

## Abstract

Machine learning trains by maximizing a score — accuracy, reward, likelihood. Reinforcement learning is the purest form of the commitment, and its failure modes are now named: reward hacking, Goodhart collapse, and the invisibility of suffering — an expectation is indifferent to where cost concentrates, and a reward has no channel for what the optimization costs the subjects it acts upon or the substrate it runs on. We introduce **Mercyful Learning**, a training paradigm that inverts the commitment: minimize suffering, subject to a hard performance target. The training objective is

$$
\mathcal{L}_{\mathrm{mercyful}}(\theta) \;=\; \mathcal{L}_{\mathrm{task}}(\theta) \;+\; \lambda\, S_{\mathrm{patient}}(\theta) \;+\; \mu\, S_{\mathrm{machine}}(\theta)
\qquad \text{subject to}\qquad \mathrm{Perf}(\theta) \ge \tau,
$$

where $S_{\mathrm{patient}}$ prices what the model's decisions cost the subject, $S_{\mathrm{machine}}$ prices what training and inference cost the substrate, and the **anti-Goodhart constraint** $\mathrm{Perf}(\theta) \ge \tau$ is a hard feasibility condition, not a penalty. We prove that this placement is forced *for cost penalties*: when suffering enters the objective as a penalty there is a computable weight beyond which the unconstrained objective prefers abstention (Theorem 2.1). This is consistent with — not contrary to — classical exact-penalty theory: penalizing the *violation* of a constraint is exact above a dual threshold and recovers the constrained solution [24]; Theorem 2.1 penalizes the *suffering*, a different object, and exhibits the crossover $\lambda^*$ at which that penalty buys mercy with the target. The theorem is about where the target must live, not about constraint-versus-penalty in general. We give suffering-aware gradient descent with a standard convergence guarantee (Theorem 3.1), anti-Goodhart early stopping with a soundness guarantee (Theorem 3.3), and stability theorems showing the mercyful value is Lipschitz in the suffering field (Theorems 3.4–3.5). Necessary suffering is defined as the minimum over the feasible set; everything above it is *gratuitous*, and the ethical weights $(\lambda, \mu)$ become numbers one must defend rather than assumptions one can hide. On a synthetic dose-response benchmark, standard ML reaches recovery 0.9995 at 3.2× the necessary patient suffering; a naive suffering minimizer prescribes the pathology — abstention, recovery 0.089; mercyful training reaches the target (0.9086 ≥ 0.90) at 0.9% above the *estimated* necessary suffering — a lower bound on the true gap, since the estimate upper-bounds the constrained minimum (§5.2) — using 87% fewer training epochs via anti-Goodhart early stopping. Mercyful Learning is the antithesis of reinforcement learning at the level of functional class: RL maximizes an expected cumulative score in one semiring for zero explicit sufferers; Mercyful Learning minimizes a two-channel cost — an additive integral plus a non-additive peak — under a categorical constraint. The antithesis is not minimize-versus-maximize — constrained and risk-sensitive RL already minimize costs under constraints — but *who inhabits the feasible set*: in constrained RL the feasible set bounds an auxiliary cost while a score is optimized; in Mercyful Learning the feasible set is inhabited by the competence target itself, and what is optimized is the suffering. We argue this is a paradigm, not a method: it redefines what is optimized (cost, not score), who counts (patient and machine), and what is inviolable (the target), while leaving the choice of architecture, optimizer, and estimator open.

**Keywords:** suffering minimization; training paradigm; Goodhart's law; anti-Goodhart constraint; constrained optimization; safe reinforcement learning; Green AI; machine ethics; patient-centered machine learning.

---

## 1. Introduction

### 1.1 The problem with maximizing reward

The training paradigms of machine learning are defined by what they maximize. Supervised learning maximizes accuracy (equivalently, minimizes a proxy loss); reinforcement learning maximizes expected cumulative reward; preference learning maximizes the probability of being preferred. Three structural failures of score maximization are by now well documented:

1. **Reward hacking and Goodhart collapse.** When a proxy becomes the target, optimization exploits the gap between proxy and intent [1, 2, 3]. The literature on concrete AI safety problems catalogs the resulting behaviors — reward tampering, side effects, unsafe exploration [4].
2. **Suffering invisibility.** An expectation is a sum, and a sum is indifferent to where cost concentrates. A policy that drives one trajectory through a catastrophic peak while the mean improves can dominate a policy that never lets any trajectory cross a catastrophic threshold. The objective has no term that *sees* the worst moment, and none that sees the subject's cost at all unless the designer folds it into the reward — where it becomes fungible against the score.
3. **Moral-circle narrowness.** Even where harms are priced, standard formulations price at most one sufferer, implicitly, inside the reward. The substrate that executes the optimization — its energy, its compute, its reliability — is treated as free. At datacenter scale this is no longer a metaphor: training compute is a measurable burden with measurable externalities [5, 6].

These are not bugs in particular algorithms. They are consequences of the *functional class* the field has chosen to optimize: an additive expectation of a scalar score, unconstrained by anything categorical.

### 1.2 The inversion

Mercyful Learning inverts the commitment at each of the three points:

- **Minimize a cost, not maximize a score.** Suffering is a first-class, non-negative, monotone cost field (axioms S1–S5, §2.1), not a negated reward. The task loss remains in the objective only as the price of competence; the paradigm's center of gravity is the suffering terms.
- **The target is a constraint, not a summand.** The model *must* reach target performance $\tau$; models below target are infeasible, not merely expensive. This is the anti-Goodhart constraint, and §2.3 proves that once suffering is priced inside the objective, the target is traded away at a computable weight — the failure the constraint exists to prevent.
- **Two sufferers, priced separately.** The patient channel prices what decisions cost the subject; the machine channel prices what optimization costs the substrate. Both are operational, measurable, and explicitly non-phenomenological (§2.5).

The result is a training paradigm whose guarantee shape is the mirror image of RL's: not "high expected reward, subject to nothing" but "target performance, guaranteed, at minimal suffering."

### 1.3 Why a naive version fails

The inversion cannot be performed naively. A trainer that minimizes suffering alone rediscovers the oldest clinical pathology: *avoidance*. In the exposure-therapy example that motivates the companion preprint [7], the behavior that minimizes acute distress is not treating — which is the disorder. In the training setting the same trap appears as abstention: a model that does nothing suffers nothing. Theorem 2.2 shows the trap is universal — at *every* choice of ethical weights, unconstrained suffering minimization prefers the abstaining model — and §5.2 shows it dynamically: the naive suffering minimizer converges to recovery 0.089. Mercy for the subject, unconstrained, is nihilism toward the task; mercy for the machine, unconstrained, is nihilism toward the subject (it prescribes never training). The anti-Goodhart constraint is not decoration on the paradigm; it is the only thing standing between suffering minimization and the prescription of pathology.

### 1.4 Position within the Mercyful Learning program

This repository contains an ongoing program in which suffering is treated as a formal, budgeted cost: a combinatorial treatment-sequencing framework with exact small-graph schedulers [7], a clinical companion applying it to synthetic vancomycin, tacrolimus, and chemotherapy sequencing [8], an expanded-ethics mathematical core proving the axioms and stability package for two-sufferer scheduling [9], a learned suffering field estimated from synthetic pharmacokinetic cohorts [10], and a Lean 4 mechanization of the single-sufferer scheduler [11]. All of these plan over *given* suffering fields by exact enumeration. This paper makes the complementary move: it lifts the framework from path planning to **gradient-based training**, where the suffering field is a functional of the parameters and the decision object is the trained model itself. The claim of this paper is that the resulting training discipline constitutes a *paradigm* — a reusable answer to "what do we optimize, for whom, under what is inviolable" — and not merely one more constrained-optimization method (§8.1).

### 1.5 Contributions

1. **The paradigm** (§2): the mercyful objective $\mathcal{L}_{\mathrm{task}} + \lambda S_{\mathrm{patient}} + \mu S_{\mathrm{machine}}$ under a hard target constraint; suffering axioms carried from paths to parameters; the necessary/gratuitous decomposition for trained models.
2. **The anti-Goodhart theorems** (§2.3–2.4): penalty formulations fail at a computable crossover weight $\lambda^*$ (Theorem 2.1; the closed form is matched by a grid consistency check — arithmetically forced, since both use the same measured quantities — and, substantively, realized dynamically by unconstrained gradient descent, §5.3); the abstention trap is universal across ethical weights (Theorem 2.2).
3. **The training mathematics** (§3): suffering-aware gradient descent with convergence rates (Theorem 3.1); mercyful regularization, subsuming weight decay as the degenerate machine-only fragment (§3.3); anti-Goodhart early stopping, sound by construction and finite-time terminating (Theorem 3.3); Lipschitz and gap stability of the mercyful value (Theorems 3.4–3.5).
4. **Implementable algorithms** (§4): the suffering-aware loss, the mercyful optimizer with feasibility restoration, and the stopping rule, each as short pseudocode with a complete reference implementation.
5. **A synthetic benchmark** (§5) with an eight-clause executable contract, all green: standard ML over-treats (3.2× necessary suffering), naive mercy abstains (the Goodhart pathology, made quantitative), mercyful training reaches target at 0.9% above the estimated necessary suffering — a lower bound on the true gap (§5.1) — at 13% of the training compute, the Theorem 2.1 crossover is realized dynamically, and the stability bound is never violated in 20 random field perturbations.
6. **An adoption argument and an honest ledger** (§8): why fragments of the paradigm are already in production use under other names; what is proven, measured, and conjectural; and pre-registered falsifiers.

---

## 2. The paradigm

### 2.1 Suffering as a first-class cost

Let $\theta \in \Theta$ be the parameters of a model, and let a *deployment population* be a distribution over subjects with features $x$. The model's decisions impose two measurable burdens:

**Definition 2.1 (suffering functionals, training form).** The *patient suffering* $S_{\mathrm{patient}}(\theta) = \mathbb{E}_x[\,b(x, \theta)\,]$ is the expected per-subject burden of the model's decisions — toxicity, distress, side-effect load, overdiagnosis — where $b \ge 0$ is a declared burden model. The *machine suffering* $S_{\mathrm{machine}}(\theta)$ is an operational computational-burden proxy of training and inference — energy, FLOPs, memory, parameter count, verification load — e.g. $S_{\mathrm{machine}}(\theta) = \rho\,\|\theta\|_2^2$.

Both functionals inherit the cost axioms of the path formulation [9], restated for parameters:

| Axiom | Statement |
|---|---|
| S1 non-negativity | $S(\theta) \ge 0$; $S(\theta) = 0$ iff the burden is identically zero on the population |
| S2 monotonicity | $b \le b'$ pointwise $\Rightarrow S_b \le S_{b'}$ |
| S3 structure | integrated burden is additive over subjects; peak burden (§2.6) concatenates by max, not sum — the two are different mathematical species, and no per-subject reweighting expresses the peak |
| S4 positive homogeneity | $S_{\lambda b} = \lambda S_b$ for $\lambda \ge 0$ |
| S5 field-Lipschitz | $|S_b(\theta) - S_{b'}(\theta)| \le \sup_x |b(x,\theta) - b'(x,\theta)|$ |

S1 is what makes suffering a *cost floor with a floor at zero* rather than a score: there is an identifiable state — imposing no burden — that scores cannot express, since any score can always be increased. S5 is what makes the paradigm *stable under measurement error* (Theorem 3.4).

### 2.2 The mercyful objective

**Definition 2.2 (Mercyful Learning).** Given a task loss $\mathcal{L}_{\mathrm{task}}$, a performance functional $\mathrm{Perf}$, a target $\tau$, and ethical weights $\lambda, \mu \ge 0$, mercyful training solves

$$
\boxed{\;\min_{\theta \in \Theta_\tau}\; \mathcal{L}_{\mathrm{mercyful}}(\theta)
\;=\; \mathcal{L}_{\mathrm{task}}(\theta) + \lambda\, S_{\mathrm{patient}}(\theta) + \mu\, S_{\mathrm{machine}}(\theta),
\qquad \Theta_\tau = \{\theta : \mathrm{Perf}(\theta) \ge \tau\}.\;}
$$

Each component has a distinct ethical reading, developed formally in [9]: the task loss and integral burden are the *utilitarian* objects (additive, expectation-ready); the constraint set $\Theta_\tau$ is the *deontic* object (categorical — infeasible means prohibited, never merely costly); the peak term and the two-channel structure are the *care-ethics* objects (the worst moment of *this* sufferer; the moral circle is relational). A single-objective RL formalism can natively express only the first of the three.

### 2.3 The anti-Goodhart constraint — and why it cannot be a penalty

The defining commitment of the paradigm is that the target lives in the feasible set. This is forced, not aesthetic.

**Theorem 2.1 (penalty failure — the Goodhart crossover).** Suppose the model class contains an abstaining model $\theta_0$ with $S_{\mathrm{patient}}(\theta_0) = 0$ and task loss $L_0 = \mathcal{L}_{\mathrm{task}}(\theta_0)$, and suppose every target-reaching model has $S_{\mathrm{patient}} \ge s_{\min} > 0$. Consider the unconstrained penalized objective $F_\lambda(\theta) = \mathcal{L}_{\mathrm{task}}(\theta) + \lambda\, S_{\mathrm{patient}}(\theta)$ and let $L_* = \min_{\theta \in \Theta_\tau} \mathcal{L}_{\mathrm{task}}(\theta)$. Then for every

$$
\lambda \;>\; \lambda^* \;:=\; \frac{L_0 - L_*}{s_{\min}},
$$

every feasible model has strictly larger penalized cost than the abstaining model; hence unconstrained minimization of $F_\lambda$ abstains and misses the target.

*Proof.* For $\theta \in \Theta_\tau$: $F_\lambda(\theta) \ge L_* + \lambda\, s_{\min} > L_* + (L_0 - L_*) = L_0 = F_\lambda(\theta_0)$, using $S_{\mathrm{patient}}(\theta) \ge s_{\min}$ and $\lambda > \lambda^*$. The abstaining model's own penalized cost is $L_0 + \lambda \cdot 0 = L_0$. $\blacksquare$

The theorem is constructive: $\lambda^*$ is *computable from declared quantities*, and §5.3 verifies the closed form against a grid consistency check on the synthetic benchmark (3.0814 vs 3.0750) and shows gradient descent dynamically realizing the switch. Its content is ethical as much as mathematical: **when suffering is a summand of an unconstrained objective, there is a computable weight at which it purchases abstention — exactly the trade the constraint exists to block.** An optimizer of the penalized objective treats the target as a price; mercyful training treats it as a boundary. This is the training-form analogue of the Lean-proved `goodhart_trap` of the companion formalization [11] and of the "categorical means non-priced" doctrine of the expanded-ethics core [9].

**Scope of Theorem 2.1: violation penalties versus cost penalties.** The theorem does not contradict classical exact-penalty theory — it is about a different penalty. Han–Mangasarian [24] prove that penalizing the *violation* of a constraint is *exact*: for a penalty parameter above a dual threshold, the unconstrained penalized problem recovers the constrained solution. Theorem 2.1 penalizes the *suffering* — the cost the paradigm exists to minimize — not the constraint violation. The two objects behave in opposite directions as their weight grows: a sufficiently large violation penalty enforces the constraint; a sufficiently large suffering penalty abolishes it, because the abstaining model's zero suffering eventually outvotes the target at any task-loss advantage. The theorem's content is therefore not "penalties fail" but "the target cannot be protected by the very cost it is meant to overrule": any formulation in which suffering enters the objective admits a computable weight at which that summand buys mercy with the target. Hence the target must live in the feasible set — a statement about placement, not about the general merits of constraint-versus-penalty formulations.

**Theorem 2.2 (the abstention trap is universal).** Let the model class contain $\theta_0$ with $S_{\mathrm{patient}}(\theta_0) = S_{\mathrm{machine}}(\theta_0) = 0$, and let every target-reaching model have strictly positive cost in *both* channels. Then for every $\lambda, \mu \ge 0$ (not both zero), the unconstrained minimizer of $\lambda S_{\mathrm{patient}} + \mu S_{\mathrm{machine}}$ is the abstaining model — including $\mu > 0, \lambda = 0$: pure machine-welfare minimization prescribes never treating, because any treatment costs the machine something. The target constraint repairs this at every weight.

*Proof.* Both channels are non-negative (S1), so $\theta_0$ has cost $0$; any weighted sum with a positive weight on a channel in which every feasible model is strictly positive is strictly positive for every feasible model. $\blacksquare$

Theorem 2.2 is the expanded-ethics abstention trap of [9, Thm. T8] restated for training, and it is the sense in which *expanding the moral circle creates a new Goodhart failure*: the paradigm's own ethics, unconstrained, turn against it. The deontic component is the fence. The theorem is a tautology — deliberately. Its content is not arithmetic but *placement*: that the abstention trap holds at every choice of weights, with a one-line proof, is what makes the constraint load-bearing rather than decorative. There is no tuning of $(\lambda, \mu)$ that escapes the trap; the only escape is the feasible set.

### 2.4 Necessary vs. gratuitous suffering

**Definition 2.3 (necessity is a constrained minimum).** The *necessary suffering* at target $\tau$ is

$$
S^*_{\mathrm{patient}}(\tau) \;=\; \min_{\theta \in \Theta_\tau} S_{\mathrm{patient}}(\theta),
\qquad
\text{gratuitous}(\theta) \;=\; S_{\mathrm{patient}}(\theta) - S^*_{\mathrm{patient}}(\tau)
\;\; (\theta \in \Theta_\tau),
$$

and *mercy* is attaining the minimum. Gratuitous suffering is undefined — not zero — for infeasible models: a model below target has not earned the comparison. Necessity is a *budgetary* notion (the least suffering compatible with reaching the target), not a claim that suffering is ever metaphysically required; raising $\tau$ raises the necessary suffering, which makes the price of ambition an explicit, computable curve $\tau \mapsto S^*(\tau)$ that a deployment decision must defend.

### 2.5 The expanded ethics: patient and machine

The two channels are deliberately asymmetric in status and identical in form. The patient channel prices the paradigm's primary concern. The machine channel has three honest motivations, in increasing strength [9]: (i) **engineering welfare** — an overstressed substrate is an unreliable substrate, and unreliability flows back into the patient channel as harm; (ii) **symmetry stress-test** — a framework that cannot accommodate a second sufferer was never about suffering, only about patients; (iii) **precautionary moral-circle expansion** — if substrate welfare ever becomes morally considerable, the mathematics should not have to be rebuilt. Motivation (i) alone justifies the channel operationally. **No claim of machine consciousness or phenomenology is made or needed.** The weights $(\lambda, \mu)$ are compassion-allocation parameters: because the feasible set is bounded, the value $V(\lambda, \mu)$ is concave piecewise-linear in the weights with finitely many, exactly computable crossovers [9, Thm. T3] — the ethics of allocation becomes a number one must defend, not an assumption one can hide. One boundary must be kept: the crossovers are exact where the alternatives are given, as in the path formulation's finite graph [9, Thm. T3], and are estimates wherever the alternatives must be trained, since each competitor is then represented by the output of a search (§5.6 measures the resulting sensitivity). The ethics remains a number one must defend; in the training setting it is a number with an error bar.

### 2.6 The antithesis to RL

| | Reinforcement learning | Mercyful Learning |
|---|---|---|
| Objective | maximize $\mathbb{E}[\sum_t \gamma^t r_t]$ | minimize $\mathcal{L}_{\mathrm{task}} + \lambda S_{\mathrm{patient}} + \mu S_{\mathrm{machine}}$ |
| Direction | maximize a score (no ceiling) | minimize a cost (floor at zero, S1) |
| Target | implicit in the reward; always tradeable | hard constraint $\mathrm{Perf} \ge \tau$; never tradeable |
| Sufferer | none explicit | patient + machine, priced separately |
| Cost structure | additive expectation (one semiring) | integral + peak (two semirings, S3) |
| Characteristic failure | reward hacking | abstention — fenced by the constraint (Thms. 2.1–2.2) |
| Guarantee shape | high mean return, no per-trajectory floor | target performance at near-minimal suffering |

**Scope of the antithesis.** The table opposes Mercyful Learning to *unconstrained* RL — the paradigm's pure antipode. The intermediate quadrant is already inhabited, and we do not claim it: constrained MDPs bound auxiliary expected costs in the feasible set [12]; risk-sensitive and CVaR objectives replace the expectation by a tail functional [25]; reward-constrained policy optimization keeps a reward objective under a cost constraint [26]; and constrained learning theory supplies generalization guarantees for learning under constraints [27, 28]. Against that literature the defensible antithesis is not *minimize versus maximize* — a CMDP already minimizes a cost under constraints — but **who inhabits the feasible set**. In CMDP, CVaR, and RCPO formulations the constraint (or the risk functional) governs an *auxiliary cost* while the optimized quantity remains a score; in Mercyful Learning the feasible set is inhabited by the *competence target itself*, and the optimized quantity is the suffering. The inversion is of which quantity is categorical and which is traded.

The antithesis is at the level of functional class, not hyperparameters. An expected-cumulative-reward agent is not "a suffering minimizer with different weights": it is missing a semiring (the peak is not additive, S3) and missing a feasible set (its target is a price, Theorem 2.1). Conversely, Mercyful Learning is not "RL with a negative reward": negating a cost keeps it inside the objective, where Theorem 2.1 shows it is traded away at a computable weight, and keeps the worst moment fungible against the sum. The paradigms differ in what may not be traded.

---

## 3. The mathematics

### 3.1 Setup

Let $\Theta \subseteq \mathbb{R}^d$ and $F(\theta) = \mathcal{L}_{\mathrm{mercyful}}(\theta)$. We use standard smoothness language: $g$ is $L$-smooth if $\|\nabla g(u) - \nabla g(v)\| \le L\|u - v\|$. If $\mathcal{L}_{\mathrm{task}}, S_{\mathrm{patient}}, S_{\mathrm{machine}}$ are $L_0, L_p, L_m$-smooth, then $F$ is $L_F$-smooth with $L_F = L_0 + \lambda L_p + \mu L_m$ (triangle inequality on the gradients). Suffering terms built from smooth burden models (quadratic toxicity, quadratic parameter norms, sigmoid outcomes — the §5 benchmark uses exactly these) satisfy this. The ethical weights therefore enter the *conditioning* of training, not just its objective: mercy has a price in smoothness, computable before training begins.

### 3.2 Suffering-aware gradient descent

Suffering-aware gradient descent is gradient descent on $F$, i.e. the task gradient *corrected* by the suffering gradients:

$$
\theta_{t+1} \;=\; \theta_t - \eta\,\big(\nabla \mathcal{L}_{\mathrm{task}}(\theta_t) + \lambda\,\nabla S_{\mathrm{patient}}(\theta_t) + \mu\,\nabla S_{\mathrm{machine}}(\theta_t)\big).
$$

**Theorem 3.1 (descent and convergence).** Let $F$ be $L_F$-smooth and bounded below by $F^* > -\infty$. Then gradient descent with step $\eta = 1/L_F$ satisfies $F(\theta_{t+1}) \le F(\theta_t) - \frac{1}{2L_F}\|\nabla F(\theta_t)\|^2$ and

$$
\min_{0 \le t < T} \|\nabla F(\theta_t)\|^2 \;\le\; \frac{2 L_F\, (F(\theta_0) - F^*)}{T}.
$$

If moreover $F$ is $\sigma$-strongly convex, then $F(\theta_T) - F^* \le \big(1 - \tfrac{\sigma}{L_F}\big)^T (F(\theta_0) - F^*)$.

*Proof.* The descent lemma gives $F(\theta - \eta\nabla F(\theta)) \le F(\theta) - \eta(1 - \eta L_F/2)\|\nabla F(\theta)\|^2$; at $\eta = 1/L_F$ the per-step decrease is $\|\nabla F\|^2/(2L_F)$. Telescoping over $T$ steps and bounding the minimum by the mean gives the rate; strong convexity gives the linear rate by the standard Polyak–Łojasiewicz argument. $\blacksquare$

The theorem is deliberately standard — that is the point. Suffering-aware training inherits the entire convergence theory of smooth optimization *unchanged*, because suffering enters as a smooth term. What changes is where the sequence converges (toward low-suffering models) and what may be returned (only feasible ones, next).

### 3.3 Mercyful regularization

Read in the other direction, the suffering terms are regularizers with an ethical semantics — and the field has, unknowingly, been using degenerate fragments of the paradigm for decades:

- **Weight decay** $\mu\,\rho\|\theta\|_2^2$ is exactly $S_{\mathrm{machine}}$ with a quadratic burden model: a machine-only mercy term with no patient channel and no target constraint.
- **Sparsity, pruning, quantization, distillation** are machine-suffering reductions under different burden models (memory, energy per inference).
- **Complexity penalties in medical models** (e.g., preferring lower-intensity regimens at equal fit) are patient-suffering terms in embryo.

Mercyful Learning is the completion of this trajectory: both channels, explicitly weighted, under the constraint that makes the weights safe. The claim is not that these classical devices were unethical; it is that the paradigm names what they were fragments of, and supplies the two missing pieces (the patient channel and the feasible set) that turn a bag of penalties into an ethics.

### 3.4 Constraint enforcement: feasibility restoration

We enforce $\Theta_\tau$ by *feasibility-restoration switching* — a filter-free instance of the classical restoration principle of nonlinear programming, which alternates progress on feasibility with progress on the objective [23]. While $\mathrm{Perf}(\theta_t) < \tau$, ascend on $\mathrm{Perf}$ alone; once feasible, descend $F$. Two properties matter for the paradigm:

**Theorem 3.2 (anti-Goodhart soundness).** Any model returned by a training procedure that (i) descends $F$ only on feasible iterates and (ii) returns only iterates certified feasible is feasible. The guarantee holds by construction — it does not depend on convergence, step size, or smoothness.

*Proof.* The return precondition is $\mathrm{Perf}(\theta) \ge \tau$; the returned object is in $\Theta_\tau$ by definition. $\blacksquare$

Theorem 3.2 is a tautology by construction — and that is the point. The certificate does not depend on convergence, step size, smoothness, or search quality; a guarantee that needs no hypotheses is the strongest statement a training procedure can issue about its output. The failure mode is equally explicit: if restoration cannot find feasibility (target too high, model class too weak), the procedure returns *no model* and raises an infeasibility alarm. This is the paradigm's safe failure: mercyful training never silently relaxes the target. Compare penalty methods, which by Theorem 2.1 relax it silently at exactly the moment the suffering weight grows.

### 3.5 Anti-Goodhart early stopping

Standard early stopping monitors a validation *score* and stops when it stops improving. Anti-Goodhart early stopping monitors the two mercyful objects instead:

> **Rule.** Stop at the first $t$ such that $\mathrm{Perf}(\theta_t) \ge \tau$ **and** the suffering total has improved by less than $\varepsilon$ over a window of $w$ steps. Return $\theta_t$.

**Theorem 3.3 (soundness and termination).** (i) *Soundness:* the rule never returns a model below target — by construction (Theorem 3.2 applies). (ii) *Termination:* the procedure always terminates, at worst at the declared horizon `max_epochs`. It fires *before* the horizon exactly when the runtime-checkable window condition holds while feasible; a sufficient condition for eventual firing is convergence of the iterates to an interior point of $\Theta_\tau$ at which $F$ has a local minimum. The sufficient condition is strong but verifiable in practice: the benchmark satisfies it, firing at epoch 39 of a 600-epoch horizon (§5.2).

*Proof.* (i) Immediate from the firing precondition. (ii) Termination by the horizon is trivial. For the sufficient condition: convergence to an interior feasible limit implies feasibility holds eventually always, and $\theta_t \to \theta^*$ with $\nabla F(\theta^*) = 0$ implies, by smoothness, that per-step changes in $F$ — hence in the suffering total once the task term has also settled — are eventually below $\varepsilon/w$ per step, so the window condition is met in finite time. $\blacksquare$

The rule's ethics: standard early stopping can return a model that is still improving in suffering (it stops when the *score* stalls); the mercyful rule keeps training only while mercy is still being bought, and stops the moment further epochs purchase no suffering reduction — which is also the moment further *machine* suffering (training compute) is gratuitous. On the benchmark the rule fires at epoch 39 of a 300-epoch horizon (§5.2): the machine channel prices its own training.

### 3.6 Stability theorems

A paradigm that prices suffering must be stable under the mis-measurement of suffering.

**Theorem 3.4 (Lipschitz stability of the mercyful value).** Let $V(b) = \min_{\theta \in \Theta_\tau} F(\theta; b)$ where the suffering field $b$ enters through $S_{\mathrm{patient}}$ satisfying S5 with constant 1. Then $|V(b) - V(b')| \le \lambda\, \sup_{x,\theta}|b(x,\theta) - b'(x,\theta)|$.

*Proof.* For each fixed $\theta$, $|F(\theta;b) - F(\theta;b')| = \lambda\,|S_b(\theta) - S_{b'}(\theta)| \le \lambda\,\|\Delta b\|_\infty$ by S5. Writing $F_b(\theta) \le F_{b'}(\theta) + \lambda\|\Delta b\|_\infty$ and minimizing both sides over $\Theta_\tau$ gives $V(b) \le V(b') + \lambda\|\Delta b\|_\infty$; symmetry gives the other direction. $\blacksquare$

**Theorem 3.5 (gap stability of the selected model).** If $F(\cdot; b)$ has a unique minimizer over $\Theta_\tau$ with optimality gap $g > 0$, then every field perturbation with $\lambda\|\Delta b\|_\infty < g/2$ leaves the selected model unchanged.

*Proof.* Each competitor's objective moves by less than $g/2$ in either direction (Theorem 3.4's estimate applied pointwise); the gap cannot close. $\blacksquare$

Both bounds have constants statable *before seeing data*. §5.4 verifies Theorem 3.4 numerically: 20 random 5% perturbations of the suffering field, zero bound violations, worst observed value movement at 0.2% of the bound. The path-form analogues (Lipschitz, gap, and Knightian sandwich bounds for the scheduler) are Theorems T4–T6 of [9].

### 3.7 Honesty ledger

| Claim | Status |
|---|---|
| Penalty failure at computable $\lambda^*$ (Thm. 2.1); abstention trap (Thm. 2.2) | **Proven** in this paper; numerically certified (P4, P5) |
| Convergence of suffering-aware GD (Thm. 3.1) | **Proven** (standard descent lemma, restated for $F$) |
| Soundness of constraint enforcement and early stopping (Thms. 3.2–3.3) | **Proven** (by-construction); termination under stated hypotheses |
| Value stability (Thms. 3.4–3.5) | **Proven**; certified on 20 perturbations (P8) |
| Mercyful training reaches target at near-minimal suffering on the benchmark | **Measured** (synthetic; P1–P3, P6, P7; the gap is a lower bound against the estimated constrained minimum, §5.1–5.2) |
| The machine channel selects structure, and the constraint bounds its appetite at a finite patient price | **Measured** (synthetic; K1–K8, §5.6). The crossover *values* are estimator-dependent (up to 50% shift); the switch *sequence* is not |
| Paradigm adoption dynamics (§8.2); clinical-channel calibration; machine-channel semantics beyond engineering welfare | **Conjectural / future work** (§8.3) |

---

## 4. Implementation

Three drop-in components. The reference implementation used for §5 is `scripts/research/mercyful_paradigm_benchmark.py` (NumPy, self-contained, deterministic, 8-clause contract).

**Algorithm 1 — Suffering-aware loss.**
```
def mercyful_loss(theta, batch, lam, mu):
    L  = task_loss(theta, batch)                # competence price
    Sp = mean(burden(x, theta) for x in batch)  # patient channel, b >= 0  (S1)
    Sm = rho * norm(theta)**2                   # machine channel (energy/FLOP proxy)
    return L + lam*Sp + mu*Sm
```

**Algorithm 2 — Mercyful optimizer (feasibility restoration).**
```
for t in range(max_epochs):
    if perf(theta, val) < tau:                  # INFEASIBLE: restore, do not optimize cost
        theta += lr_c * grad_perf(theta, train)     # ascend on performance
    else:                                       # FEASIBLE: buy mercy
        theta -= lr * grad_mercyful_loss(theta, train, lam, mu)
    if anti_goodhart_stop(theta):               # Algorithm 3
        return theta                            # certified feasible (Thm 3.2)
raise InfeasibilityAlarm(tau)                   # safe failure: no model returned
```

**Algorithm 3 — Anti-Goodhart early stopping.**
```
def anti_goodhart_stop(theta):
    if perf(theta, val) < tau:                  # soundness precondition (Thm 3.3i)
        return False                            # may never stop below target
    return suffering_total_recent_improvement() < eps   # mercy exhausted
```

Integration notes. The burden model $b$ is declared per domain (§6 gives three); the constraint functional $\mathrm{Perf}$ is evaluated on held-out data each epoch; the switching rule composes with any base optimizer (SGD, Adam) by replacing the gradient in the feasible branch; the stopping rule composes with any checkpointing discipline. Nothing in the paradigm requires a new framework — it is a loss, a branch, and a stopping rule.

**Overhead.** The suffering terms add one burden evaluation per example per epoch — $O(Nd)$ for linear or quadratic burden models, negligible against backpropagation. The constraint check is one forward pass over the validation set per epoch, the same order as the standard practice of computing validation loss; the branch itself is $O(1)$. Mercyful training's wall-clock overhead over standard training is therefore a small constant factor on the evaluation line, and is more than repaid whenever anti-Goodhart early stopping fires before the horizon (§5.2: 39 vs 300 epochs).

---

## 5. Experiments

### 5.1 Setup

A synthetic dose-response training problem, deliberately the simplest setting in which the paradigm's three failures and its repair can all be made quantitative. $N = 4000$ synthetic patients with features $x \sim \mathcal{N}(0, I_8)$; each has a required treatment intensity $\mathrm{need}(x) = \mathrm{clip}(\mathrm{softplus}(w^\top x + b), 0.2, 3.0)$. A one-layer model prescribes $\mathrm{dose}_\theta(x) = \mathrm{softplus}(\theta^\top x + \theta_0)$. Recovery probability is $\mathrm{sigmoid}(4\cdot(\mathrm{dose} - \mathrm{need}))$; performance is mean recovery, target $\tau = 0.90$. Patient suffering is the treatment burden $S_{\mathrm{patient}} = 0.1\cdot\mathbb{E}[\mathrm{dose}^2]$ — zero at dose zero, so that *not treating costs no suffering*: the avoidance pathology is priced in from the start. Machine suffering is $S_{\mathrm{machine}} = 10^{-3}\|\theta\|^2$. Three trainings, all full-batch gradient descent from the same init: **(A) standard ML** — task loss only, fixed 300 epochs; **(B) naive mercy** — suffering terms only, no task loss, no constraint; **(C) mercyful** — Definition 2.2 with $\lambda = \mu = 1$, feasibility restoration, anti-Goodhart early stopping. The necessary suffering $S^*_{\mathrm{patient}}(\tau)$ is estimated by boundary-pinned constrained search (best feasible iterate over 1500 epochs). Because this is a heuristic search returning the best feasible iterate found, its result upper-bounds the true constrained minimum: $S^*_{\mathrm{est}} \ge S^*_{\mathrm{true}}$. Every gratuitous-suffering figure below is therefore a **lower bound** on the true gap, $\mathrm{gratuitous}_{\mathrm{true}}(\theta) = S_{\mathrm{patient}}(\theta) - S^*_{\mathrm{true}} \ge S_{\mathrm{patient}}(\theta) - S^*_{\mathrm{est}}$ — the reported 0.9% is an optimistic floor, not a measured bound. Everything is deterministic (seed 7); every number below reproduces from the reference script.

### 5.2 Main result

| method | epochs | Perf | $S_{\mathrm{patient}}$ | $S_{\mathrm{machine}}$ | gratuitous (lower bound, §5.1) |
|---|---|---|---|---|---|
| standard ML | 300 | 0.9995 | 0.8881 | 0.0003 | **0.6137** |
| naive mercy | 300 | 0.0890 | 0.0020 | 0.0000 | *infeasible* |
| **mercyful** | **39** | **0.9086** | **0.2767** | 0.0003 | **0.0024** |

with $S^*_{\mathrm{est}}(0.90) = 0.2743$ (best feasible iterate found, at Perf 0.9016 — an upper-bound estimate of the constrained minimum, §5.1, so the gratuitous column is a lower bound on the true gap).

Reading the table. **Standard ML prescribes pathology in one direction**: it buys its last 9 points of recovery (0.9995 vs 0.9086) at 3.2× the necessary patient suffering — gratuitous suffering ≥ 0.6137, over 250× the mercyful model's. **Naive mercy prescribes pathology in the other direction**: it minimizes suffering to 0.002 by prescribing abstention — recovery 0.089, the exposure-therapy failure rendered as a training run (P2). **Mercyful training does what the paradigm promises**: reaches the target (0.9086 ≥ 0.90), at 0.9% above the *estimated* necessary suffering (gratuitous ≥ 0.0024; since the estimate upper-bounds the true minimum, 0.9% is a lower bound on the true relative gap — an optimistic floor, not a guarantee), and stops itself at epoch 39 — 13% of the standard training compute — because mercy was exhausted (P1, P3, P6, P7).

**The machine channel in this benchmark is carried by the stopping rule, not the gradient.** At the achieved parameter norms, $S_{\mathrm{machine}} = 0.0003$ — 0.03% of the mercyful objective — so the machine gradient is numerically inert in this instance: deleting $\mu S_{\mathrm{machine}}$ from the descent would not measurably move the trajectory. What actually exercises the channel is anti-Goodhart early stopping: halting at epoch 39 of a 300-epoch horizon avoids 87% of the training compute, which *is* the machine-suffering reduction the paradigm promises — here the channel prices training time through the stopping rule rather than through the loss. A benchmark in which the machine term moves the gradient (larger $\rho$, or an explicitly energy-priced optimizer) remains future work; §3.3's reading of weight decay as the machine-only fragment is where the channel already bites in practice.

### 5.3 The Goodhart demonstration

Theorem 2.1 predicts a computable crossover for the *unconstrained* penalized objective: below $\lambda^*$ training treats, above it training abstains. On the benchmark, the closed form from declared quantities gives $\lambda^* = 3.0814$; a grid measurement over the two regimes' objectives gives 3.0750 (agreement within grid resolution, P4). We flag that **P4 is a consistency check, not independent evidence**: the grid discretizes the two objective lines defined by the *same* measured quantities that enter the closed form, so agreement to grid resolution is arithmetically forced — the clause guards against implementation error and nothing more. The substantive test is P5. Dynamically: unconstrained gradient descent at $\lambda = 1.54 < \lambda^*$ converges to a treating model (Perf 0.8601), while at $\lambda = 4.62 > \lambda^*$ it converges to abstention (Perf 0.2743) — the switch realized by training, not just by arithmetic (P5). The penalized formulation is thus demonstrated, on the same problem, to prescribe the pathology at a weight a practitioner could easily choose; the constrained formulation cannot, at any weight.

### 5.4 Stability

Twenty random perturbations of the suffering field (per-patient burden coefficients jittered by ±5%) re-solve the constrained problem each time; the Theorem 3.4 bound is never violated, and the worst observed value movement is 0.2% of the bound (P8) — the bound is sound and, on this instance, loose by a comfortable margin. Gap stability (Theorem 3.5) holds trivially on this instance since the constrained minimizer is unique with a wide gap.

### 5.5 What is not shown

The benchmark is synthetic and one-dimensional in spirit: it demonstrates the *mechanics* (the three failures, the constraint, the crossover, the stopping rule) with exact reproducible numbers. It does not demonstrate clinical validity (no real patients), scale (no deep network), or burden-model calibration (the burden model is declared, not estimated — see [10] for a learned-field prototype). In particular the model class is single-layer: whether feasibility restoration remains reliable in richer nonconvex landscapes (e.g., small MLPs, where restoration steps and mercyful descent may interact less benignly) is untested and is the leading falsification target for the training-dynamics claims. Contract: `MERCYFUL_PARADIGM_BENCHMARK_VERDICT P_GREEN (8/8 clauses PASS)`.

### 5.6 The machine channel decides structure

§5.2 conceded that in the dose-response benchmark the machine term is numerically inert in the gradient: at the achieved parameter norms $\mu S_{\mathrm{machine}}$ is 0.03% of the objective, and $\rho\|\theta\|_2^2$ is weight decay under an ethical name. A second benchmark isolates the channel by making the machine burden **architectural and $\theta$-independent** — a parameter/FLOP count, $S_{\mathrm{machine}}(S_k) = \rho_m (k+1)$ with $\rho_m = 0.02$ — so that $\mu$ has *exactly zero* gradient within any structure and its entire effect is on which structure is selected.

The capacity ladder is $S_k$ = the $k$ input features of largest $|w_{\mathrm{true}}|$ plus an intercept, $k \in \{0,1,2,4,8\}$; $S_0$ is the constant-dose model, which can still reach the target but only by dosing everyone at the level the sickest patient needs. Each structure is represented by a boundary-pinned estimate of its own constrained minimum (minimum-$S_{\mathrm{patient}}$ feasible iterate), so the comparison is between structures rather than between training schedules; Perf lies in $[0.9000, 0.9010]$ for every structure.

| structure | params | Perf | $S_{\mathrm{patient}}$ | $S_{\mathrm{machine}}$ |
| --- | --- | --- | --- | --- |
| $S_0$ | 1 | 0.9010 | 0.3690 | 0.0200 |
| $S_1$ | 2 | 0.9001 | 0.3161 | 0.0400 |
| $S_2$ | 3 | 0.9000 | 0.2842 | 0.0600 |
| $S_4$ | 5 | 0.9001 | 0.2717 | 0.1000 |
| $S_8$ | 9 | 0.9000 | 0.2685 | 0.1800 |

Capacity buys patient mercy (K2) and costs substrate (K3); $\mu$ selects along the ladder with switches at $\mu^* = 0.0388$ ($S_8 \to S_4$), $0.3142$ ($S_4 \to S_2$), $1.5940$ ($S_2 \to S_1$), $2.6450$ ($S_1 \to S_0$) (K4).

**The fence, priced.** Without the target constraint, at $\mu = 10$ the machine channel selects $S_0$ at Perf 0.0683 — Theorem 2.2 realised in the *machine* channel rather than the patient one: unconstrained mercy to the substrate prescribes never treating (K5). With the constraint, selection saturates at the smallest feasible structure at Perf 0.9010, and the saturation has a finite, computable price: maximal machine mercy costs the patient $0.3690$ against $0.2685$, an exchange rate of **0.63 patient units per machine unit** on this instance (K6). This is the "two mercies" exchange rate of the path formulation [9, §9.2] exhibited rather than asserted.

**A measured limit on the paradigm's own rhetoric.** Repeating the construction with last-iterate rather than boundary-pinned representatives leaves the *sequence* of structural switches unchanged (K8) but moves their *prices* by up to 50% ($S_4 \to S_2$: $0.3142$ versus $0.6282$). The crossover $\mu^*$ is exact where the alternatives are given — as in the discrete scheduler, whose $\mu^* = 11$ is arithmetic on a fixed graph [8] — and is an *estimate* wherever the alternatives must be trained, inheriting the estimator's error. We report training-time crossovers as estimates accordingly.

Contract: `MERCYFUL_MACHINE_CHANNEL_STRUCTURAL_VERDICT K_GREEN (8/8 clauses PASS)`. The ladder is a feature-count ladder on a one-layer model; depth, sparsity, and quantisation ladders are the natural extensions and are untested.

---

## 6. Applications

### 6.1 Medical ML: the patient channel

The patient channel is the paradigm's home ground. Burden models $b(x, \theta)$ are domain-declared and auditable: toxicity-weighted dose burden in treatment recommendation; procedural burden and overdiagnosis cost in diagnostic models; distress trajectories in psychiatry-adjacent models. The anti-Goodhart constraint formalizes what clinical governance already insists on informally — a model that misses the sensitivity target is not "cheaper," it is *not a candidate*. The companion papers instantiate the full machinery on synthetic clinical sequencing problems: suffering fields derived from Knightian pharmacokinetic bands, drug–drug-interaction gates as edge-deletion constraints, and the exposure-therapy benchmark in which a naive minimizer prescribes avoidance at cost zero [7, 8]. A learned suffering field estimated from synthetic population-pharmacokinetic cohorts shows the channel can be estimated rather than declared [10]. Nothing in this line is a clinical recommendation; the contribution to medical ML is the discipline: suffering priced, target categorical, necessity computed.

### 6.2 Energy-efficient ML: the machine channel

The machine channel gives Green AI [5] a paradigmatic home. Energy and carbon accounting [6], efficient-model methods (pruning, quantization, distillation), and compute-capping policies are all machine-suffering reductions; Mercyful Learning adds the two things they lack as a *paradigm*: a principled objective position (the burden is a cost with a floor at zero, not a score penalty) and a hard competence floor that prevents efficiency pressure from silently degrading the model below target — the energy-version of the abstention trap (Theorem 2.2: pure machine-welfare minimization prescribes never training). Anti-Goodhart early stopping is itself an efficiency method with an ethical certificate: it stops when further compute purchases no mercy (§5.2: 87% training-compute reduction on the benchmark).

### 6.3 Ethical AI: both channels

For the broader ethics-of-AI discourse, the paradigm offers a concrete proposal for "do no harm" as a training objective: harm is priced per-sufferer, the duty of competence is categorical, and the trade-offs that remain are compressed into declared weights whose value function is concave piecewise-linear with finitely many, exactly computable crossovers [9] — i.e., an auditable deliberation object rather than a hidden hyperparameter. The paradigm is also a natural training-time complement to inference-time alignment techniques: RLHF shapes a score; mercyful training bounds a cost. We flag, and do not pursue here, the stronger conjecture that alignment itself is better framed as suffering minimization under a capability floor than as reward maximization under a preference model (§8.3).

---

## 7. Related work

**Reinforcement learning and its failures.** RL maximizes expected cumulative reward [1]; Goodhart's law [2, 3] and the concrete-problems agenda [4] document the consequences, and the reward-hacking literature supplies the taxonomy and the formalism — the Goodhart variants [32], reward misspecification mapped and mitigated [33], and reward hacking defined and characterized [31]. Mercyful Learning is the antithesis at the functional level (§2.6), not an RL variant.

**Safe and constrained RL.** Constrained MDPs optimize one expected cost subject to constraints on others [12]; CPO enforces near-constraint-satisfaction during policy learning [13]; safety-gym benchmarks measure violation rates [14]; risk-sensitive objectives replace the expectation by a tail functional [25]; reward-constrained policy optimization keeps a reward objective under a cost constraint [26]. The paradigm shares the hard-constraint instinct but differs in object and direction: constraints there bound *auxiliary expected costs*; here the constraint is the *target* (competence), and the *objective* is the suffering — the §2.6 antithesis of who inhabits the feasible set. The nonlinear-programming literature offers middle paths — exact penalties of the *violation*, which recover the constrained solution above a dual threshold [24], and filter/restoration methods [23]. Theorem 2.1 does not oppose that theory; it marks the boundary of what exactness can protect. Exactness holds for violation penalties, not for the suffering weight: in this setting the suffering weight is itself the dangerous penalty parameter, chosen for ethical reasons, and above a computable value it purchases mercy with the target (§2.3).

**Constrained learning theory.** Probably-approximately-correct constrained learning [27] and its non-convex extension [28] give generalization guarantees for learning under constraints; proxy-Lagrangian and two-player methods make non-differentiable constraints trainable [29]; reductions approaches cast constrained classification as a sequence of cost-sensitive problems [30]. This literature shares the paradigm's commitment that constraints are first-class objects of training. The differences remain those of §2.6: the constrained quantity there is an auxiliary cost or a fairness statistic rather than the competence target, and the optimized quantity remains a score rather than a suffering field.

**Alignment and side-effect avoidance.** Quantilizers avoid Goodhart collapse by sampling from an acceptable quantile of a base distribution [15]; attainable utility preservation and relative reachability penalize loss of optionality [16]; RLHF learns a reward from preferences [17]; cooperative inverse RL treats the objective as jointly owned [18]. These modify the *score*; Mercyful Learning changes the *direction* (minimize a cost) and adds the second sufferer. They are composable: a quantilized mercyful trainer is well-defined.

**Energy-efficient ML.** Green AI [5] and empirical energy measurement [6] supply the machine channel's semantics; §6.2 positions them as the machine-only fragment.

**Patient-centered ML.** Selective prediction and learning to defer give models an abstention action [19, 20]; fairness-constrained learning puts group constraints in the feasible set [21, 30]. The paradigm generalizes both: abstention is a model in the class, priced by the suffering terms and fenced by the target constraint — which is exactly the analysis §5.2 performs.

**Mercyful Learning program (this repository).** The path-form framework and anti-Goodhart axiom [7]; clinical instantiations [8]; the expanded-ethics core — axioms S1–S5, two-sufferer scalarization, theorems T2–T8 including the scheduling abstention trap [9]; the learned suffering field [10]; the Lean 4 scheduler mechanization with the proved `goodhart_trap` [11]. This paper is the program's training-paradigm statement; the scheduling line is its planning counterpart.

---

## 8. Discussion

### 8.1 Why this is a paradigm, not a method

A method answers "how"; a paradigm answers "what is optimized, for whom, under what is inviolable." Mercyful Learning fixes three paradigm-level commitments — (i) minimize a cost with a floor at zero rather than maximize a score without a ceiling; (ii) price two sufferers separately, with the allocation weights made explicit and their crossovers computable; (iii) hold the performance target categorical — and leaves everything else open: architecture, estimator, optimizer, burden model, even the constraint functional. Gradient descent, Adam, second-order methods, and exact combinatorial scheduling [7] are all mercyful *methods* under the same paradigm, exactly as policy gradients, Q-learning, and tree search are all RL under theirs. The paradigmatic test is generativity: the commitments spawn new questions (What is the burden model of a recommender system? Who sets $\tau$, and what does the $\tau \mapsto S^*(\tau)$ curve cost? What is the peak term for a population?) that are invisible inside score maximization. It also passes the falsifiability test: Theorems 2.1–2.2 name the paradigm's own characteristic failure and fence it, and §8.4 lists measurements that would refute the empirical claims.

### 8.2 Why people will adopt it

Adoption arguments, in decreasing strength:

1. **It is already partially adopted.** Weight decay, pruning, distillation, early stopping, and sensitivity floors in medical-ML regulation are fragments of the paradigm running without its name (§3.3). Adopting the paradigm is completion, not conversion.
2. **It is cheap.** One extra forward-computable term in the loss, one branch in the loop, one stopping rule. The benchmark's reference implementation is a single NumPy file; the paradigm requires no new framework.
3. **It saves compute while certifying behavior.** Anti-Goodhart early stopping reduces training cost *because* it is an ethical rule (stop when mercy is exhausted), and the returned model carries a feasibility certificate (Theorem 3.2) that accuracy-only training cannot issue.
4. **It is audit-shaped.** Regulators and institutional review boards ask exactly the paradigm's questions: what does the model cost those it acts upon, what performance is guaranteed, and who chose the trade-off. Mercyful Learning outputs those as numbers: $S_{\mathrm{patient}}$, $\tau$, $(\lambda, \mu)$, and the crossover prices at which the choices would flip.

### 8.3 Limitations

- **Burden-model calibration.** The paradigm prices suffering *as modeled*. A mis-specified $b$ is priced faithfully and wrongly; Theorem 3.4 bounds the damage linearly, and [10] prototypes estimation, but domain calibration is the open empirical problem.
- **No sentience claim — and its cost.** The machine channel is justified operationally (§2.5). Readers who believe substrate welfare is morally considerable will find the channel ready but uncalibrated; readers who do not lose nothing, since motivation (i) suffices.
- **Feasibility is a sample property.** $\mathrm{Perf} \ge \tau$ is certified on held-out data; distribution shift can void the certificate. The safe failure is loud (no model returned), but monitoring is required — as for any deployed constraint.
- **Restoration may not find feasibility.** If $\tau$ exceeds the model class's reach, training returns nothing. That is the designed behavior, but it makes target-setting a first-class, consequential act.
- **No new algorithmic primitive is claimed.** Feasibility restoration [23], exact penalties [24], constrained smooth optimization, and early stopping are classical; the scheduling counterpart inherits NP-hardness from resource-constrained shortest paths [9]. The contribution is the paradigm-level synthesis: the objective's direction, the two channels, the categorical target, the necessity decomposition, and the executable, falsifiable benchmark.
- **Single-architecture evidence.** The training-dynamics evidence is one synthetic problem with a one-layer model; extension to deep, nonconvex architectures (where restoration and mercyful descent interact nontrivially) is the leading open falsification direction.
- **Conjectures flagged, not claimed:** clinical-channel validity on real cohorts; the alignment-as-mercyful-training conjecture (§6.3); large-scale deep-learning instantiations.

### 8.4 Falsifiers

The empirical claims are refuted by: (i) any benchmark run in which the mercyful model is returned below target (breaks Theorem 3.2's implementation); (ii) any $\lambda$ at which the *constrained* trainer abstains (breaks the fence); (iii) a measured crossover disagreeing with the Theorem 2.1 closed form beyond grid resolution; (iv) a Theorem 3.4 bound violation. The benchmark script encodes these as contract clauses P1–P8 and currently reports `P_GREEN (8/8)`.

- **F4 (crossover stability).** §5.6 reports that the structural switch *sequence* survives a change of representative while the switch *prices* move by up to 50%. A representative set that reordered the switches — not merely repriced them — would refute the claim that the ladder's ordering is a structural property, and with it the auditability argument of §8.2(4): a $\mu$ whose selection is not stable in *order* is not an auditable deliberation object.

---

## 9. Conclusion

Machine learning's first paradigm maximized fit; its second maximized reward. Both treated the costs of optimization — to the subjects it acts upon, to the substrate it runs on — as someone else's term. Mercyful Learning moves those costs into the objective's center and moves the target out of the objective into the feasible set, where it can no longer be traded. The theorems of this paper are, deliberately, not deep: a computable crossover at which a suffering penalty purchases abstention, a descent lemma, a stopping rule that cannot lie, a Lipschitz bound. Their force is not mathematical surprise but ethical placement — each one marks a point where a choice that used to be hidden inside a score becomes explicit, computable, and defensible. The functional choice *is* the ethical commitment; Mercyful Learning is the paradigm that makes the commitment minimizing suffering, and says so.

---

## References

1. Sutton RS, Barto AG. *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
2. Goodhart CAE. Problems of monetary management: the U.K. experience. In *Monetary Theory and Practice*, 1984. ("Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes.")
3. Strathern M. 'Improving ratings': audit in the British University system. *European Review* 5(3):305–321, 1997.
4. Amodei D, Olah C, Steinhardt J, Christiano P, Schulman J, Mané D. Concrete problems in AI safety. arXiv:1606.06565, 2016.
5. Schwartz R, Dodge J, Smith NA, Etzioni O. Green AI. *Commun. ACM* 63(12):54–63, 2020. doi:10.1145/3381831.
6. Strubell E, Ganesh A, McCallum A. Energy and policy considerations for deep learning in NLP. *ACL 2019*, 3645–3650. doi:10.18653/v1/P19-1355.
7. Agourakis DC. Mercyful Learning: a formal framework for suffering-budget-aware treatment sequencing. `docs/papers/mercyful_learning_preprint_2026-07-26.md`, this repository, 2026.
8. Agourakis DC. Mercyful Learning: suffering-budget-aware treatment sequencing — clinical integrations. `docs/papers/mercyful_learning_medical_paper_2026-07-26.md`, this repository, 2026.
9. Agourakis DC. Mercyful Learning — expanded mathematics for expanded ethics: suffering minimization as the antithesis of RL. `docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`, this repository, 2026.
10. Agourakis DC. Mercyful Learning — learned suffering field s(v) (patient + machine). `docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md`, this repository, 2026.
11. Agourakis DC. Mercyful scheduler — Lean 4 mechanization spec. `docs/research/mercyful_scheduler_lean_spec_2026-07-26.md` and `formal/lean4/SounioMercyfulScheduler.lean`, this repository, 2026.
12. Altman E. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999.
13. Achiam J, Held D, Tamar A, Abbeel P. Constrained policy optimization. *ICML 2017*, 22–31.
14. Ray A, Achiam J, Amodei D. Benchmarking safe exploration in deep reinforcement learning. OpenAI, 2019.
15. Taylor J. Quantilizers: a safer alternative to maximizers for limited optimization. *AAAI Workshop on AI, Ethics and Society*, 2016.
16. Krakovna V, Orseau L, Kumar R, Martic M, Legg S. Penalizing side effects using stepwise relative reachability. *ICLR 2019 Workshop*; extended version arXiv:1806.01186.
17. Christiano P, Leike J, Brown TB, Martic M, Legg S, Amodei D. Deep reinforcement learning from human preferences. *NeurIPS 2017*, 4299–4307.
18. Hadfield-Menell D, Dragan A, Abbeel P, Russell S. Cooperative inverse reinforcement learning. *NeurIPS 2016*, 3909–3917.
19. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. *NeurIPS 2017*, 4878–4887.
20. Madras D, Pitassi T, Zemel R. Predict responsibly: improving fairness and accuracy by learning to defer. *NeurIPS 2018*, 6147–6157.
21. Zafar MB, Valera I, Gomez Rodriguez M, Gummadi KP. Fairness constraints: mechanisms for fair classification. *AISTATS 2017*, 962–970.
22. Foa EB, McLean CP, Zang Y, et al. Effect of prolonged exposure therapy delivered over 2 weeks vs 8 weeks vs present-centered therapy on PTSD symptom severity in military personnel. *JAMA* 319(4):354–364, 2018. doi:10.1001/jama.2017.21242. *(Motivating clinical structure — avoidance vs. necessary distress; no clinical claim adopted.)*
23. Fletcher R, Leyffer S. Nonlinear programming without a penalty function. *Mathematical Programming* 91(2):239–269, 2002. doi:10.1007/s101070100244.
24. Han SP, Mangasarian OL. Exact penalty functions in nonlinear programming. *Mathematical Programming* 17:251–269, 1979. doi:10.1007/BF01588250.
25. Chow Y, Tamar A, Mannor S, Pavone M. Risk-sensitive and robust decision-making: a CVaR optimization approach. *NeurIPS 2015*. arXiv:1506.02188.
26. Tessler C, Mankowitz DJ, Mannor S. Reward constrained policy optimization. *ICLR 2019*. arXiv:1805.11074.
27. Chamon LFO, Ribeiro A. Probably approximately correct constrained learning. *NeurIPS 2020*. arXiv:2006.05487.
28. Chamon LFO, Paternain S, Calvo-Fullana M, Ribeiro A. Constrained learning with non-convex losses. *IEEE Transactions on Information Theory* 69(3):1739–1757, 2023. arXiv:2110.04323.
29. Cotter A, Jiang H, Gupta MR, Wang S, Narayan T, You S, Sridharan K. Optimization with non-differentiable constraints with applications to fairness, recall, churn, and other goals. *J. Mach. Learn. Res.* 20(172):1–59, 2019.
30. Agarwal A, Beygelzimer A, Dudík M, Langford J, Wallach H. A reductions approach to fair classification. *ICML 2018*. arXiv:1803.02453.
31. Skalse J, Howe NHR, Krasheninnikov D, Krueger D. Defining and characterizing reward hacking. *NeurIPS 2022*. arXiv:2209.13085.
32. Manheim D, Garrabrant S. Categorizing variants of Goodhart's law. arXiv:1803.04585, 2018.
33. Pan A, Bhatia K, Steinhardt J. The effects of reward misspecification: mapping and mitigating misaligned models. *ICLR 2022*. arXiv:2201.03544.

---

## Reproducibility

```bash
.venv/bin/python scripts/research/mercyful_paradigm_benchmark.py
# -> MERCYFUL_PARADIGM_BENCHMARK_VERDICT P_GREEN (8/8 clauses PASS)
```

Single file, NumPy only, fixed seed 7; every number in §5 reproduces verbatim.

## Scope and safety boundaries

Synthetic data only; no patient records; no clinical recommendation; no machine-sentience claim. The burden models are declared, not validated. This manuscript is a paradigm statement with a synthetic proof-of-mechanism, not an empirical validation study.

## AI disclosure (GAIDeT-ICMJE 2025)

This manuscript was drafted under human direction with AI assistance (drafting, authorship of the companion benchmark code, and numeric verification of the reported contract outputs). Per the repository's mandatory offload-review policy, the full draft was submitted to an external multi-provider review (`bin/llm-offload --raw`, providers deepseek/xai/gemini, 2026-07-26). DeepSeek failed at provider level (Insufficient Balance) and Gemini failed at provider level (OpenRouter HTTP 402); the substantive leg was xAI Grok 4.3 — logged as **degraded single-provider** and flagged for re-review when the other providers are restored. Grok returned [OK] verdicts on Theorems 2.1, 2.2, 3.1, 3.2, 3.4, and 3.5, found no material overclaim against the stated scope, and confirmed the experimental claims match the reported numbers. Its three critiques were addressed in place: (1) Theorem 3.3's termination claim ([TIGHTENABLE]) was restated — the declared horizon is now the unconditional termination guarantee, with convergence to an interior feasible local minimum demoted to an explicit, runtime-checkable sufficient condition; (2) the restoration-principle and exact-penalty literatures are now cited and positioned (§3.4, §7, references 23–24); (3) the single-architecture limitation of the evidence is now stated explicitly as the leading falsification direction (§5.5, §8.3), and a per-epoch overhead analysis was added (§4). The review log is maintained in `.claude/llm_offload_log.md` (raw: `/tmp/llm-offload-qVA28e/`). A second-round adversarial review ("OPUS 5") subsequently raised six critiques across this manuscript and its clinical companion; the five applicable to this paper were addressed in place: (i) Theorem 2.1's scope is now stated explicitly — violation penalties are exact above a dual threshold (Han–Mangasarian [24]), cost (suffering) penalties fail at the crossover $\lambda^*$; the theorem is about where the target lives, not constraint-versus-penalty in general (abstract, §2.3, §7); (ii) the constrained-learning and reward-hacking literature is now cited and positioned (§2.6, §7, references 25–33); (iii) the 0.9% figure is declared a lower bound on the true gap, since the necessary-suffering estimator upper-bounds the constrained minimum (§5.1, §5.2); (iv) the P4 crossover check is disclosed as a tautological consistency check, with P5 identified as the substantive test (§1.5, §5.3), and the machine channel's inert gradient (0.03% of the objective) versus its real exercise through the 39-vs-300-epoch stopping rule is stated explicitly (§5.2); (v) Theorems 2.2 and 3.2 are acknowledged as deliberate tautologies, and the antithesis is restated as *who inhabits the feasible set* — the competence target, not an auxiliary cost — with the CMDP/CVaR/RCPO quadrant explicitly conceded (§2.6). The revised draft was resubmitted to the same three-provider review (`bin/llm-offload --raw`, deepseek/xai/gemini, 2026-07-27): DeepSeek failed at provider level (Insufficient Balance) and Gemini failed at provider level (OpenRouter HTTP 402); the substantive leg was again xAI Grok 4.3, which returned [ADDRESSED] on all five critiques applicable to this paper with no new issues — logged as a second **degraded single-provider** review and flagged for re-review when the other providers are restored (raw: `/tmp/llm-offload-Z3sIwz/`). All AI-generated content was verified against the executable artifact (`scripts/research/mercyful_paradigm_benchmark.py` → `MERCYFUL_PARADIGM_BENCHMARK_VERDICT P_GREEN (8/8 clauses PASS)`). No clinical or patient-level claim is made. The author takes full responsibility for the content.

A third adversarial round (Anthropic Claude Opus 5, 2026-07-27) audited the corrected manuscript and identified four residual internal inconsistencies created by the second round's own fixes — a dead falsifier at §8.4(iii), a stale ledger attribution to P4, a 600-versus-300 horizon collision at §3.5, and an unprotected gratuitous-suffering ratio at §5.2 whose value is an upper bound collapsing to 3.21 as the necessary-suffering estimate tightens — all four addressed in place. The same round supplied the structural machine-channel benchmark reported in §5.6, whose first contract run **failed** two clauses; the failing clause on selection-ordering invariance was reformulated to the statement the data support (switch sequence invariant, switch prices not) rather than relaxed, and the failing pinning clause was repaired by refining the boundary oscillation. Both the failure and the reformulation are recorded here because the paper's epistemic ledger would be worth less if only passing runs were reported.
