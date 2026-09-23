# dataframe — Examples

## 1. Create DataFrame

```sio
use dataframe::pure::core
use dataframe::pure::types

let names = vec!["name".to_string(), "age".to_string(), "score".to_string()]
let vecs = vec![
    ColumnData::Text(vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()]),
    ColumnData::F64(vec![25.0, 30.0, 35.0]),
    ColumnData::F64(vec![85.5, 90.2, 78.8]),
]
let df = dataframe_from_vectors(names, vecs)?
assert_eq!(df.n_rows, 3)
assert_eq!(df.columns.len(), 3)
```

## 2. Filter and Select

```sio
use dataframe::pure::core

let df = dataframe_new()
let selected = dataframe_select(&df, vec!["name".to_string(), "score".to_string()])?

let mask = vec![true, false, true]
let filtered = dataframe_filter(&df, mask)?
```

## 3. Group By Aggregation

```sio
use dataframe::pure::core

let df = dataframe_new()
let gb = dataframe_group_by(&df, vec!["category".to_string()])?
let means = groupby_agg(&gb, AggFunc::Mean, "value")?
```

## 4. Merge / Join

```sio
use dataframe::pure::core
use dataframe::pure::types

let left_df = dataframe_new()
let right_df = dataframe_new()

let merged = dataframe_merge(
    &left_df,
    &right_df,
    "id".to_string(),
    "id".to_string(),
    MergeType::Inner,
)?
```

## 5. Epistemic Statistics

```sio
use dataframe::pure::epistemic::{ecolumn_new, ecolumn_push, ecolumn_mean, ecolumn_std}
use epistemic::knowledge::{ep_measured, ep_val}

// EColumn holds Epistemic values; ecolumn_std returns the std as an Epistemic.
var col = ecolumn_new("measurements")
ecolumn_push(&!col, ep_measured(10.1, 0.1))
ecolumn_push(&!col, ep_measured(10.2, 0.1))
ecolumn_push(&!col, ep_measured(9.9, 0.1))

let mean = ecolumn_mean(&col)
let std = ecolumn_std(&col)
print("Mean: ")
print_f64(ep_val(&mean))
print(" +/- ")
print_f64(ep_val(&std))
print("\n")
```

`epistemic_mean`, `epistemic_std(&col)`, and `epistemic_confidence_interval` are not in the module. The checked functions are `ecolumn_mean`, `ecolumn_std`, `ecolumn_sum`, and `ecolumn_correlation` in `stdlib/dataframe/pure/epistemic.sio`. There is no confidence-interval helper.

## 6. Rolling Window

```sio
use dataframe::pure::core

let df = dataframe_new()
let rolling = dataframe_rolling(&df, 3, "value", "mean")?
```