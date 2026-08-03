# CS6 V7-B target-23 prospective epistemic replay

**Date:** 2026-08-03
**Status:** the predeclared 331-leaf, two-carrier replay completed on Slurm. All
662 fresh worker processes passed the frozen local epistemic rule. This is a
prospective replication on the bounded target-23 adaptive domain, not a global
H-PG certificate or a solution of an open problem.

## The question

The preceding adaptive epistemic cover was retrospective: the depth-4 and
depth-5 computations already existed when the six-enclosure acceptance rule was
defined. The next honest question was therefore:

> If the exact topology and rule are frozen first, do new executions reproduce
> the same strict-negative determinant conclusion on every selected leaf and
> both carriers?

The answer is yes for this implementation and this frozen target-23 domain.

## What was frozen before execution

Commit `0327d77d735e03c9aa1c29254441f5fb36d4f5e5` was pushed before the worker
was rebuilt or the Slurm job was submitted. It contains:

- the exact 331-leaf manifest: 231 depth-4 leaves and four depth-5 children for
  each of the other 25 depth-4 cells;
- both carriers for every leaf, giving `331 * 2 = 662` attempts;
- challenge and attempt-binding domains tied to the contract, manifest,
  pre-execution commit, leaf, input, and carrier;
- the exact acceptance rule and an independent raw-output verifier;
- 14 predeclared receipt mutations.

The attempt rule requires a successful probe, structural validity, valid
homogeneous computation, explicit false legacy-certificate flags, a strictly
negative Liouville interval, and a nonempty strictly negative intersection of
the C1, C2, affine, resident-reconstructed, homogeneous, and Liouville
determinant enclosures. Both carriers must pass at every leaf.

## Source-fresh execution

The CAPD worker was rebuilt after the frozen commit with the pinned CAPD 5.3.0
configuration and `g++ -std=c++17 -Og -g0 -DNDEBUG`. The worker binary hash is
`cc7269d60026c3f004db7099b18e2ff20f8976f295644c97779ce8ef7dcf0137`.

Slurm job `8531` ran on `gpuorangefs-multi-r740-proxmox` with 32 concurrent
CPU slots. Every attempt invoked a separate worker process, so solver, section,
map, and set state were not shared between attempts. The job completed in
`00:01:10` with exit code `0:0` and empty stderr.

The framed return stream declared 26,531,840 result bytes and SHA-256
`6d836c86ad6980264e998a379958ef8308977222805088e9dfae46840889a8f5`.
The locally received archive matched both values exactly.

## Result

| Check | Result |
|---|---:|
| predeclared leaves evaluated | 331 / 331 |
| fresh worker processes | 662 / 662 |
| attempt epistemic certificates | 662 / 662 |
| paired leaf certificates | 331 / 331 |
| new unique leaf challenges | 331 |
| overlap with 356 prior challenges | 0 |
| exact Liouville/joint endpoint and verdict replications | 662 / 662 |
| unique prospective stdout hashes | 662 |
| verifier mutations rejected | 14 / 14 |

Thus the retained strict-negative determinant conclusion reproduced exactly
under a post-freeze build, fresh processes, new audit challenges, and a new
Slurm execution. In all 662 attempts the exact six-way intersection remained
nonempty and strictly negative; its endpoints and certificate verdict matched
the retrospective audit exactly.

## What this establishes

This closes the specific prospective-replication objection to the retained
target-23 adaptive result. The acceptance rule and topology existed before the
new numerical outputs, all required attempts executed, and the raw evidence was
independently re-parsed with exact rational comparisons.

It also supplies a deterministic reproducibility observation: changing the
build identity, process instances, receipt bindings, and Slurm execution did
not change any retained Liouville or joint-intersection endpoint.

## What it does not establish

The new challenge is an audit binding; it is not a random perturbation of the
ODE, section, or input boxes. The replay uses the same CAPD implementation,
worker source, physical coordinates, and two carrier families. Therefore it is
not an independent numerical implementation, cross-toolchain replication,
sensitivity experiment, global H-PG proof, or statistical sample of a larger
domain.

Accordingly:

```text
PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=true
GLOBAL_HPG_CERTIFICATE=false
V7_B_ELIGIBILITY=false
PROMOTION_ELIGIBLE=false
OPEN_PROBLEM_SOLVED=false
NOVELTY_OR_PRIORITY_CLAIMED=false
FPGA_EXECUTION=false
```

The next scientifically stronger step is implementation diversity: recompute
the same 331 leaves with an independently authored determinant path or a second
validated interval engine, then require cross-implementation enclosure
intersection without changing this acceptance rule after seeing the result.

## Durable evidence

The retained receipt is
`scripts/research/receipts/cs6_v7b_target23_prospective_epistemic_replay_v1/`.
It includes the compressed full result, raw per-attempt evidence, Slurm and
fresh-build provenance, transport binding, local verification, mutation
results, replication comparison, and LLM-offload reviews.
