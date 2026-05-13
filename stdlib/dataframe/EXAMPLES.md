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
use dataframe::pure::epistemic
use dataframe::pure::types

let col = column_new("measurements".to_string(), ColumnData::F64(vec![
    10.1, 10.2, 9.9, 10.3, 10.0, 9.8, 10.1, 10.2
]))

let mean = epistemic_mean(&col, 0.95)?
let std = epistemic_std(&col)?
let (lo, hi) = epistemic_confidence_interval(&col, 0.95)?
print("Mean: {} +/- {}\n", mean, (hi - lo) / 2.0)
print("95% CI: [{}, {}]\n", lo, hi)
```

## 6. Rolling Window

```sio
use dataframe::pure::core

let df = dataframe_new()
let rolling = dataframe_rolling(&df, 3, "value", "mean")?
```