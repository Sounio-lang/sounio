<!-- docs:meta
topic_id: repo.docs.compiler.incremental-query-opt-p0
authority: historical
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.incremental-query-opt-p0
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Incremental Query Optimization (P0 Scaffold)

This document describes a concrete P0 artifact for query-based optimization in the compiler.

## What was added

- `crates/souc/src/mir/optimization/incremental_query.rs`
  - `IncrementalFunctionOptimizer`: wraps MIR optimization in `build::query::QueryDb`.
  - Keyed by:
    - function fingerprint (`Hash` of `MirFunction`)
    - optimization level (`O0..O3`)
    - pipeline epoch (`u32`) for manual invalidation when pass logic/order changes.
  - Returns `IncrementalOptimizationOutput` with:
    - optimized function
    - pass result
    - cache hit/miss signal.
- `crates/souc/examples/incremental_query_optimization_scaffold.rs`
  - Runnable example showing first-run miss and second-run hit.

## Why this is low-risk

- No default compiler behavior is changed.
- Existing pass manager and optimization logic remain the source of truth.
- The scaffold is opt-in and can be integrated where needed.

## Run now

```bash
cargo run -p souc --example incremental_query_optimization_scaffold
```

Expected output includes:

- first run `cache_hit: false`
- second run `cache_hit: true`
- non-zero query hit rate

## Integration point (next step)

Wrap existing MIR pass execution with `IncrementalFunctionOptimizer::optimize_function(...)`
in the codegen optimization path to avoid re-running unchanged function optimizations across
incremental compile loops.
