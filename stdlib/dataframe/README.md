# dataframe — Tabular Data Analysis

Tabular data structures with full epistemic uncertainty propagation for Sounio.

## Overview

The `dataframe` module provides DataFrame and Series types inspired by pandas/R,
with native support for `Knowledge<f64>` uncertainty tracking:

- **Pure Sounio** — No external dependencies, works everywhere
- **Epistemic-aware** — All numeric operations track uncertainty via GUM propagation
- **Arrow-compatible** — FFI layer for Apache Arrow when available

## Epistemic Differentiators

Unlike other DataFrame libraries, every numeric operation in Sounio's dataframe
returns `Knowledge<f64>` results with:

- **Confidence** — Probability that the true value is within stated bounds
- **Uncertainty** — GUM-compliant standard uncertainty
- **Provenance** — Chain of operations that produced the result
- **Automatic propagation** — Uncertainty compounds through operations

## Quickstart

```sio
use dataframe::pure::core
use dataframe::pure::epistemic

// Create from vectors
let names = vec!["x".to_string(), "y".to_string(), "z".to_string()]
let vecs = vec![
    ColumnData::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
    ColumnData::F64(vec![2.0, 4.0, 6.0, 8.0, 10.0]),
    ColumnData::F64(vec![1.5, 2.5, 3.5, 4.5, 5.5]),
]
let df = dataframe_from_vectors(names, vecs)? assert_eq!(df.n_rows, 5)

// Describe statistics
let stats = dataframe_describe(&df)

// Epistemic operations
let col = dataframe_column(&df, "x")?
let mean = epistemic_mean(col, 0.95)?
let (lo, hi) = epistemic_confidence_interval(col, 0.95)?
```

## Module Structure

| File | Description |
|------|-------------|
| `pure/types.sio` | Core types: DataFrame, Column, Series, DType |
| `pure/core.sio` | Operations: select, filter, groupby, join, sort, pivot |
| `pure/epistemic.sio` | Uncertainty-aware operations |
| `ffi/lib.sio` | FFI layer for Apache Arrow |

## Supported Operations

### Data Manipulation
- `dataframe_select` — Select columns
- `dataframe_filter` — Filter rows by mask
- `dataframe_sort` — Sort by columns
- `dataframe_head` / `dataframe_tail` — First/last rows

### Aggregation
- `dataframe_group_by` — Group by columns
- `groupby_agg` — Aggregate with Mean, Sum, Count, Min, Max, Std

### Joins
- `dataframe_merge` — Join on keys (inner, left, right, outer)

### Reshaping
- `dataframe_pivot` — Pivot to wide format
- `dataframe_melt` — Melt to long format
- `dataframe_concat` — Concatenate DataFrames

### Rolling
- `dataframe_rolling` — Rolling window operations

### Epistemic
- `epistemic_mean`, `epistemic_std` — With uncertainty
- `epistemic_correlation`, `epistemic_covariance` — With confidence
- `epistemic_confidence_interval` — GUM-compliant CI

## Benchmarks

See `../../benchmarks/README.md` for performance targets.

## Validation Status

- ✅ DataFrame creation from vectors
- ✅ Column selection and filtering
- ✅ Groupby aggregation
- ✅ Merge/join operations
- ✅ Epistemic statistics

## License

MIT / Apache-2.0 (same as Sounio)