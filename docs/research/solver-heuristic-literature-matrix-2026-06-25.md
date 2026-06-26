<!-- docs:meta
topic_id: repo.docs.research.solver-heuristic-literature-matrix-2026-06-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-heuristic-literature-matrix-2026-06-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver Heuristic Literature Matrix

Date: 2026-06-25

This note is a narrow claim-control matrix for the Sounio SAT/SMT solver
heuristic lane. It is not an exhaustive survey. Its purpose is to prevent
overclaiming around adaptive SAT heuristics, multi-armed-bandit framing,
Thompson sampling, polarity choice, and restart/reset interaction.

## Immediate Claim Boundary

Unsafe broad claims:

- "First Thompson Sampling SAT solver."
- "First bandit SAT/DPLL/CDCL branching heuristic."
- "First uncertainty-aware SAT branching heuristic."
- "State-of-the-art SAT/SMT/PB solver."
- "Hadwiger-Nelson chi(R^2) >= 6."

Current safe wording:

> Sounio has a candidate epistemic variable-score and polarity-sampling
> experiment in its bounded DPLL(T) SMT lane, with GUM-style uncertainty
> instrumentation and private canonical-wrapper evidence. It is not yet a
> public solver-performance or proof-certificate result.

## Literature Matrix

| Prior work | What it already covers | Consequence for Sounio wording |
| --- | --- | --- |
| Chaff / VSIDS, 2001 (`https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf`) | Established variable-activity decision strategy with decayed recent-conflict signal and low overhead. | Sounio cannot claim novelty for activity-based branching or recent-conflict decay. Mean/activity baselines must be treated as expected comparators. |
| CHB / ERWA, AAAI 2016 (`https://ojs.aaai.org/index.php/AAAI/article/view/10439`) | Conflict-history branching is explicitly inspired by exponential recency weighted average methods for bandit problems and learns branching variables from conflict-analysis feedback. | Sounio cannot claim first bandit-inspired conflict-history branching. |
| LRB, 2016 (`https://cs.uwaterloo.ca/~ppoupart/publications/sat/learning-rate-branching-heuristic-SAT.pdf`) | Frames SAT branching as an online optimization problem, models variable selection as a multi-armed bandit, and implements learning-rate branching in MiniSat/CryptoMiniSat. | Sounio's Wilson/LRB-like and epistemic-LRB surfaces must be positioned as local bounded-DPLL(T) experiments, not a new MAB branching paradigm. |
| Global Learning Rate, IJCAI 2018 (`https://www.ijcai.org/proceedings/2018/0745.pdf`) | Studies branching through a global learning-rate metric and machine-learning approximation of GLR maximization. | Sounio cannot claim that learning-rate-style objective framing is new. |
| Combining VSIDS and CHB using restarts, CP 2021 (`https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2021.20`) | Uses restarts to switch between VSIDS and CHB and reports competitive improvements. | Sounio cannot claim first restart-mediated switching among branching regimes. |
| RL reset policy with UCB and Thompson sampling, 2024 (`https://arxiv.org/html/2404.03753v2`) | Models reset/no-reset as a multi-armed bandit and uses UCB plus Thompson sampling in CDCL reset policy experiments. | Sounio cannot claim first Thompson Sampling use in a CDCL solver context. A narrower distinction may be variable-score and polarity sampling inside Sounio's bounded DPLL(T), but that remains a candidate pending deeper survey. |
| MAB-for-SAT survey, 2025 (`https://ojs.aaai.org/index.php/SOCS/article/download/35997/38152/40069`) | Surveys recent multi-armed-bandit algorithms across SAT solver heuristics, including CDCL, SLS, and parallel-solver strategies. | Any public novelty statement must be scoped against a broad existing MAB/SAT literature, not only against one or two classic papers. |

## What May Still Be Novel

The plausible narrow novelty is not "bandits in SAT" and not "Thompson in
SAT." The plausible narrow candidate is a combination of:

1. Source-level proof-profile gates for solver/certificate acceptance in
   Sounio.
2. Epistemic/GUM-style uncertainty instrumentation exposed at the language
   level.
3. Thompson-style variable-score sampling plus Beta-Bernoulli polarity sampling
   inside a bounded DPLL(T) SMT experiment.
4. Canonical-wrapper private evidence that the focused imported SMT harnesses
   run green.

This is still Level 2 private evidence. Level 3/public novelty requires:

- fixed-seed ablations with repetitions and timing/decision/conflict tables;
- external DIMACS, SMT-LIB, and OPB benchmark slices against real solvers;
- independent proof/certificate replay beyond microkernels;
- a fuller related-work review that includes CHB/LRB/GLR, restart switching,
  Thompson reset work, MAB survey coverage, and polarity-specific SAT
  heuristics;
- wording reviewed for no accidental theorem, SOTA, or Hadwiger-Nelson claim.

## Review Status

This matrix was seeded from a live web check on 2026-06-25 using primary or
near-primary sources. It is a guardrail, not a survey article. Before any paper
or public claim, expand it into a conventional related-work section with exact
bibliographic metadata, benchmark versions, solver versions, and archived URLs.
