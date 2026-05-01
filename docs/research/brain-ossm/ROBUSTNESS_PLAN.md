<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.robustness-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.robustness-plan
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM Robustness Plan

This note defines what counts as a serious seed-based robustness pass for the current Sounio benchmark suite.

## Core Principle

No single seed is intrinsically more trustworthy than another.

- `SEED=42` and `SEED=100` are both deterministic spot checks.
- Robustness comes from the distribution across many independent seeds.
- Headlines should be based on multi-seed summaries, not cherry-picked single runs.

## Seed Tiers

### Tier 0: Spot Check

- Size: `1 seed`
- Purpose: quick determinism check, smoke validation, debugging
- Interpretation: never use as the main scientific claim

### Tier 1: Preliminary Robustness

- Size: `10 seeds`
- Purpose: first serious screening pass
- Report:
  - mean
  - standard deviation
  - standard error
  - min / max seed outcome
- Interpretation: enough to reject fragile narratives, but still preliminary for publication

### Tier 2: Paper-Grade Robustness

- Size: `20 seeds`
- Purpose: default target for any benchmark that may support a paper claim
- Report:
  - mean
  - standard deviation
  - standard error
  - confidence interval
  - per-seed table in appendix or artifact bundle
- Interpretation: best cost/quality tradeoff for the current project

### Tier 3: Claim Consolidation

- Size: `50 seeds`
- Purpose: final confirmation for one or two flagship claims
- Use only when:
  - the result already looks promising at `20 seeds`
  - the benchmark is central to the paper
  - the cluster cost is justified
- Interpretation: not needed for every benchmark, only for the decisive ones

## Recommended Allocation

| Benchmark group | Recommended seeds | Reason |
| --- | --- | --- |
| Fractal-G2 v3 | `20` | strongest current synthetic signal |
| Brain classifier toy benchmark | `10` | useful negative control, but not the flagship claim |
| Native algebra | `10` | useful mechanistic support, lower priority than Fractal-G2 |
| Multi-head unit benchmark | `10` | ablation/mechanistic context |
| Direct associativity probe | `10` | sanity check, not currently a winning benchmark |
| ABIDE-scale temporal benchmark | `20` minimum | real-data claim path |
| ABIDE flagship result if promising | `50` | final consolidation |

## Stopping Rules

Stop increasing seed count when one of the following becomes true:

1. The mean and interval stabilize and extra seeds do not materially change the conclusion.
2. The gap collapses toward zero and no longer supports a strong claim.
3. The benchmark is clearly secondary and more cluster time should be spent on ABIDE-scale runs.

## Reporting Standard

For any result that may appear in a paper, slide deck, or summary:

- do not headline a single seed
- always report multi-seed mean and spread
- keep single-seed runs labeled as deterministic reproductions or spot checks
- separate:
  - synthetic evidence
  - real-data evidence
  - mechanistic ablations

## Practical Guidance For This Project

- Treat the existing `SEED=42` run as a deterministic spot check only.
- Treat the current multi-seed baseline as more serious than any single-seed replay.
- Move the real scientific weight toward:
  - `20-seed` Fractal-G2
  - `20-seed` ABIDE-scale temporal runs
- Reserve `50 seeds` for ABIDE only if the result is already strong at `20`.

## Short Version

- `1 seed`: debug
- `10 seeds`: serious preliminary evidence
- `20 seeds`: robust default
- `50 seeds`: flagship consolidation

For this project, `SEED=100` is not more serious than `SEED=42`.
`20 independent seeds` is what makes the benchmark more serious.
