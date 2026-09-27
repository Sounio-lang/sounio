<!-- docs:meta
topic_id: repo.docs.research.future-potential-suffering-aware-agi-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.future-potential-suffering-aware-agi-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Future Potential Evaluation — Option 1: Suffering-Aware AGI (Mercyful Learning as an AGI principle)

**Date:** 2026-07-30
**Evaluator:** automated research-evaluation agent (no commit; report to parent)
**Object under evaluation:** Mercyful Learning — the suffering-minimization training
paradigm developed in this repository — evaluated specifically as a *principle for AGI*,
not merely as a narrow training method.
**Evidence base:** `docs/papers/mercyful_learning_paradigm_2026-07-26.md`,
`docs/papers/mercyful_learning_preprint_2026-07-26.md`,
`docs/research/mercyful-learning.md`,
`docs/research/PROGRAM-REGISTRY-mercyful-learning.md`,
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`,
`docs/research/suffering_aware_architecture_spec_2026-07-28.md` (SAN, 8/8 green),
`docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`,
`docs/research/mimic_iv_mercyful_validation_2026-07-26.md`,
`docs/research/mercyful_scheduler_lean_spec_2026-07-26.md`.

> **Honesty note.** This evaluation was produced inside the repository that hosts the
> program being evaluated. Where the program's own documents and the external state of
> the field diverge, the external state is taken as the arbiter. Scores are
> deliberately conservative.

---

## 1. What the direction actually is

Mercyful Learning inverts the standard training commitment: instead of maximizing a
score (reward, accuracy, likelihood), **minimize suffering subject to a hard
performance target**:

```
min  L_task(θ) + λ·S_patient(θ) + μ·S_machine(θ)   s.t.  Perf(θ) ≥ τ
```

Its load-bearing pieces, as they exist today:

- **Anti-Goodhart constraint.** The target lives in the feasible set, not in the
  objective. Theorem 2.1 (paradigm paper) shows any penalty placement admits a
  computable weight λ\* at which suffering buys abstention; Theorem 2.2 shows the
  abstention trap is universal across ethical weights. Both are elementary, correct,
  and genuinely useful as *placement* arguments.
- **Two-sufferer pricing.** Patient (subject) channel and machine (substrate) channel,
  the latter explicitly an operational compute-burden proxy with no sentience claim.
- **Necessary vs. gratuitous suffering.** Necessity = constrained minimum over the
  feasible set; mercy = attaining it. The original *topological* (mountain-pass) form
  of necessity was **falsified twice** (sedenion connectivity theorem; Pilot 1 on real
  semantic fields, DreamBank + PMI gpt2/Qwen). The surviving form is **budgetary**
  (least suffering under a declared budget L₀), which is weaker, more testable, and
  reintroduces a modeler choice the topological version claimed to eliminate.
- **Executable artifacts.** SAN architecture (8/8 contract green), runtime gates,
  learned suffering fields, a Lean 4 mechanization of the single-sufferer scheduler —
  all on **synthetic** data, with MIMIC-IV cross-validation work in progress.

## 2. Assumptions made in this evaluation

1. "AGI" means a system with general cross-domain competence at or above human level;
   no specific timeline for AGI itself is assumed, but the evaluation asks whether
   Mercyful Learning scales *with* capability rather than only at narrow scale.
2. The repo's own falsification record is accurate (no reason to doubt it; it is
   unusually candid).
3. The external baseline is the published state of the art as of mid-2026: CMDP/safe
   RL (Altman; Chow), risk-sensitive RL (CVaR, Tamar et al.), RLHF/constitutional
   methods, quantilizers, impact measures (AUP, relative reachability), active
   inference (Friston), Green AI, and the emerging AI-welfare discussion.
4. No external replication or adoption of Mercyful Learning outside this repository
   exists (none found).

## 3. Dimension-by-dimension evaluation

### 3.1 Technical feasibility — score 4/10 (narrow form: 7/10)

**What is feasible now.** Every component of the narrow paradigm is standard
constrained optimization: hard feasibility constraints, two-term cost objectives,
Lagrangian/exact-penalty machinery, early-exit architectures, FLOP metering. The
repo's own SAN benchmark demonstrates the training loop end-to-end on synthetic data.
Nothing here requires new mathematics or new hardware.

**What is not feasible, and is the actual AGI claim.** Scaling the paradigm to AGI
requires solving three problems the paradigm *presupposes* rather than solves:

1. **The burden model.** `S_patient(θ) = E_x[b(x,θ)]` requires a declared burden
   function `b` per domain. At narrow (clinical) scale this is defensible — toxicity,
   distress scores, harm matrices. At AGI scale, "suffering imposed by a decision" is
   the value-loading problem in a new costume. The paradigm relocates the hard
   problem of alignment from "specify reward" to "specify suffering + specify the
   categorical target"; it does not dissolve it.
2. **The target as a constraint.** `Perf(θ) ≥ τ` is well-defined for dose-response or
   classification. For a general agent, "performance ≥ τ" across an open task
   distribution is exactly what we cannot currently measure or enforce. The paradigm's
   central theorem (target must be categorical) is an argument *against* feasibility
   at AGI scale as much as for the paradigm.
3. **Estimating the necessity frontier.** Gratuitous suffering is defined against
   `S*(τ)`, a constrained minimum over all feasible models — computable by enumeration
   on small graphs (as in the repo's schedulers), intractable in general. The paradigm
   paper is honest about this (necessity is estimated, gaps are lower bounds), but at
   AGI scale the estimate is the whole game.

Verdict: buildable as a narrow training discipline today; as an AGI principle it is a
research agenda, not a buildable system.

### 3.2 Timeline — score 3/10

- **1 year:** more of what exists — synthetic benchmarks, position papers, Lean
  mechanization of finite cases. Realistic.
- **5 years:** credible clinical/narrow decision-support demonstrations on real data
  (the MIMIC-IV line is the right move); possible influence as a *framing* inside the
  safe-RL/clinical-ML community. Requires external adoption that has not yet begun.
- **10 years:** if the framing catches on, suffering-explicit objectives could be one
  dialect among several in alignment discourse (alongside CVaR-style risk sensitivity
  and constitutional methods). Marginal, not paradigmatic.
- **20+ years:** as "the" principle for AGI — contingent on AGI itself, on solving
  burden-model specification, and on differentiating from the mature safe-RL
  literature. Speculative.

### 3.3 Impact if realized — score 7/10

Genuine impact points:

- **Worst-case visibility.** The peak-vs-integral distinction (max concatenates, sum
  hides) is a real blind spot of expectation-based objectives, and making the worst
  moment a first-class object is valuable at any scale.
- **The placement theorem as an argument.** "A cost that appears as a summand can be
  traded away at a computable weight; what must be inviolable must be a constraint" is
  a clean, portable argument for the alignment community — probably the program's most
  exportable idea.
- **Ethics as a defended number.** The (λ, μ) weights with computable crossovers force
  ethical allocation into the open. This is a real rhetorical and methodological
  contribution even where the math is elementary.
- **Anti-sedation guard.** The abstention/avoidance trap (naive suffering minimization
  prescribes the pathology) is the correct diagnosis of the main failure mode, and
  making it a theorem rather than a caveat is good practice.

Capped at 7 because most of this impact accrues to *framing and placement arguments*,
not to capabilities the field lacks: CMDP theory already provides constrained
optimality, CVaR already prices tails, and "constraint not penalty" is known doctrine
in safe RL. The marginal impact over the existing literature is real but incremental.

### 3.4 Risks — score 7/10 (higher = riskier)

1. **Goodhart on the suffering measure itself.** The paradigm's own answer (the
   anti-Goodhart constraint) only relocates the problem: now the constraint and the
   burden model can be Goodharted. Sedation-as-optimum is blocked only if the target
   is correctly specified — which at AGI scale is the unsolved part.
2. **Novelty/differentiation risk.** The positioning section of the program registry
   itself lists the neighbors: CMDP, risk-sensitive RL, quantilizers, AUP, relative
   reachability, active inference. A skeptical reviewer can read the paradigm as
   "CMDP with two costs and a care-ethics vocabulary." The defensible novelty is the
   two-sufferer structure plus the necessity/gratuitous decomposition — thinner than
   "a new paradigm."
3. **Single-source, unreplicated.** One author, one repository, all-synthetic
   evidence, no external citations or reproductions. The program's falsification
   record is a strength epistemically but means several of its flagship constructs
   (topological necessity, sedenionic suffering measure, the 𝕊-bridge) are already
   dead; the surviving budgetary form is less distinctive.
4. **Moral-circle expansion creates new failure modes.** The program's own Theorem 2.2
   shows adding the machine channel unconstrained prescribes never treating. Each
   added sufferer adds a new abstention direction. As an AGI principle this is a live
   wire, not a solved problem.
5. **Vocabulary risk.** "Suffering-aware AGI" invites dismissal as anthropomorphism
   despite the program's careful disclaimers; the framing costs credibility in exactly
   the venues (ML venues) where adoption would matter.

### 3.5 Real research direction, or science fiction? — score 5/10

Neither pole fits. It is **not science fiction**: every mechanism exists, the theorems
are real, the honest ledger (proven / measured / conjectural) is exemplary. It is
**not an established research direction** either: there is no external community, no
replication, and the AGI-scale version presupposes solutions to the two hardest open
problems in alignment (value specification, verifiable performance at general
competence). The accurate classification:

- **As a narrow training discipline + position paper: real, now.** Publishable, and
  the clinical-sequencing application is the strongest concrete instantiation.
- **As a principle for AGI: speculative research program.** Comparable in kind to
  early impact-measure work (AUP, relative reachability circa 2018-2020) — a framing
  with one or two exportable theorems, whose fate depends on whether the community
  picks up the framing.

## 4. Score summary

| Dimension | Score (1–10) | Reading |
|---|---|---|
| Technical feasibility | 4 | Narrow form buildable now (7); AGI-scale presupposes unsolved value-loading and verifiable generality |
| Timeline | 3 | AGI-principle realization 15–20+ years; narrow fragments 1–5 years |
| Impact | 7 | High if adopted; mostly framing/placement contributions over existing safe-RL |
| Risk (lower is better) | 7 | Goodhart relocation, novelty risk vs CMDP/CVaR, unreplicated, vocabulary cost |
| **Overall future potential** | **5** | Real niche, speculative paradigm |

## 5. Recommendation: **explore**

Not "pursue" as a core AGI bet: the AGI-scale claim cannot be validated, the
differentiation from safe RL is thinner than the paradigm framing suggests, and the
program has already had to retreat from its most distinctive constructs. Not "defer":
the narrow form is executable, honestly ledgered, and contains at least two portable
contributions (the placement theorem; necessity-as-constrained-minimum with defended
weights) that are worth finishing and publishing.

Concrete next steps that would move the score:

1. **Real-data demonstration** (the MIMIC-IV line) showing the constraint-based
   objective beating expectation-based baselines on a worst-case subject metric —
   this is the single highest-value experiment available.
2. **Explicit differentiation paper** vs CMDP/CVaR: what does Mercyful Learning
   compute that they cannot, on a problem both can run?
3. **External engagement**: one workshop paper, one external reproduction. Community
   traction is the binding constraint on "real research direction" status.
4. Keep the falsification discipline — it is the program's strongest asset and the
   main reason this evaluation scores it above science fiction.

## 6. Cross-option note

Only Option 1 (Suffering-Aware AGI) was provided for evaluation in this task. Within
this evaluation set, **Option 1 is therefore the assessed direction**, with overall
future potential **5/10** and recommendation **explore**. If further options are
evaluated later, this document's scoring rubric (§4) is designed to be comparable
across options.
