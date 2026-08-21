<!-- docs:meta
topic_id: repo.docs.audit.orphan-capability-census-2026-08-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.orphan-capability-census-2026-08-21
-->

# Capability that exists and has no consumer

**Date:** 2026-08-21. **Tool:** `scripts/dev/orphan_capability_census.py`.
**Full list:** `artifacts/audit/orphan_capabilities.json`.

A function is reported here when **nothing in the tree utters its name** — no
caller inside its own file, none outside, no test, no example, no benchmark.

## Why this was looked for

The shape was measured three times in one day, never once on purpose:

- the **ZD surgical family** — eight types, a Lean proof with no `sorry`, and a
  compiler that checks only that you wrote `with ZD`;
- the **provenance classifiers** — a full trust taxonomy reached by a live call
  chain, deciding nothing because `TypeEntry` drops the field before it arrives;
- the **unit and refinement resolvers** — present on one type-lowering spine and
  absent from the one annotations actually traverse, so `mg` did not exist under
  the default engine.

Three times is a pattern, not an accident. This census asks the tree how often
it happens.

## The numbers

| | |
|---|---:|
| functions defined in `stdlib/` and `self-hosted/` | 53549 |
| **named by nothing at all** | **5508** |
| of those, tests nobody runs | 1183 |
| **real capability with no consumer** | **4325** |
| — in `stdlib/` | 2349 |
| — in the compiler | 1976 |
| declared `pub` and never uttered | 817 |

A test that never runs is not a test, so the 1183 are a finding of their
own and are counted apart rather than folded in.

## The largest in stdlib

| lines | name | file | |
|---:|---|---|---|
| 400 | `fano_auts_into` | `stdlib/algebra/fano_auts.sio` | pub |
| 236 | `ep14_selftest_main` | `stdlib/darwin_pbpk/epistemic_pbpk14.sio` |  |
| 123 | `score_pose` | `stdlib/medical/docking.sio` |  |
| 111 | `gp_sample_posterior` | `stdlib/math/gp.sio` |  |
| 101 | `cox_fit` | `stdlib/medical/survival.sio` |  |
| 87 | `fairness_report` | `stdlib/fairness/metrics.sio` |  |
| 85 | `cd_associator_exact_i64` | `stdlib/algebra/cayley_dickson_exact_i64.sio` | pub |
| 84 | `parse_extended_model` | `stdlib/medlang/parser_ext.sio` |  |
| 80 | `brent_id` | `stdlib/roots/lib.sio` | pub |
| 79 | `eg_tridiag_iter_no_vecs` | `stdlib/linalg/eigen.sio` |  |
| 77 | `propagate_watched_dbg` | `stdlib/theorem/cdcl.sio` | pub |
| 72 | `plot_area_png` | `stdlib/graphics/area.sio` | pub |
| 72 | `tiled_plot_area_png` | `stdlib/graphics/area.sio` | pub |
| 71 | `hermite_interp` | `stdlib/math/interpolation.sio` |  |
| 69 | `compute_pk_sensitivity` | `stdlib/epistemic/pk_plugin.sio` |  |
| 69 | `numerical_rank` | `stdlib/math/functional.sio` |  |
| 68 | `eeg_main` | `stdlib/medical/sedenion_eeg.sio` |  |
| 67 | `tirzepatide_priors` | `stdlib/darwin_pbpk/drugs/tirzepatide.sio` | pub |
| 67 | `mc_div` | `stdlib/epistemic/montecarlo.sio` |  |
| 67 | `plot_area_raster` | `stdlib/graphics/area.sio` | pub |

This is not generic dead code. It is written science: Cox proportional-hazards
fitting, Gaussian-process posterior sampling, molecular docking scores,
fairness metrics, PK sensitivity analysis, a whole drug's priors.

## Where it concentrates, in stdlib

| count | module |
|---:|---|
| 66 | `stdlib/nn/autograd.sio` |
| 39 | `stdlib/systems/gpu.sio` |
| 31 | `stdlib/compiler/check/context.sio` |
| 30 | `stdlib/theorem/epistemic.sio` |
| 30 | `stdlib/theorem/tactics.sio` |
| 29 | `stdlib/linalg/vector.sio` |
| 25 | `stdlib/theorem/nat.sio` |
| 23 | `stdlib/theorem/real.sio` |
| 21 | `stdlib/compiler/types/type.sio` |
| 20 | `stdlib/ontology/model.sio` |
| 18 | `stdlib/theorem/logic.sio` |
| 18 | `stdlib/compiler/effects/compile.sio` |
| 18 | `stdlib/nn/tensor.sio` |
| 17 | `stdlib/epistemic/linalg.sio` |

## Where it concentrates, in the compiler

| count | module |
|---:|---|
| 121 | `self-hosted/check/epistemic.sio` |
| 57 | `self-hosted/lsp/hover.sio` |
| 56 | `self-hosted/ir/lower.sio` |
| 51 | `self-hosted/wasm/encode.sio` |
| 51 | `self-hosted/check/types.sio` |
| 50 | `self-hosted/gpu/ptx_advanced.sio` |
| 45 | `self-hosted/hlir/lower.sio` |
| 36 | `self-hosted/gpu/numerical.sio` |
| 34 | `self-hosted/gpu/quant/quantize.sio` |
| 33 | `self-hosted/check/borrows.sio` |
| 33 | `self-hosted/check/lint.sio` |
| 30 | `self-hosted/io/file_read.sio` |
| 30 | `self-hosted/test_gpu_profiler.sio` |
| 28 | `self-hosted/lsp/goto_def.sio` |

## What this measurement is not

- **The scan is by word.** A function reached only through dynamic dispatch, or
  named only in prose, counts as an orphan here. The number is a floor for
  "worth looking at", not a verdict of "delete".
- **A `pub` entry point with no in-tree caller may be deliberate public API.**
  This script cannot tell an interface for future users from abandonment.
- **The first version of this script flagged 57% of the tree**, because it
  counted only mentions *outside* the defining file — which flags every private
  helper its own neighbours call. That is encapsulation, not abandonment. It now
  counts mentions everywhere, the defining file included.

## What to do with it

Nothing automatic. The list is an inventory to be read module by module, and
the useful question for each entry is the one that separates the three cases
that produced it: is this **unfinished** (built, never wired), **superseded**
(wired once, replaced), or **deliberate** (public surface awaiting users)? Only
the first is loss.
