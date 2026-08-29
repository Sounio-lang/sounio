# stdlib/dataframe

Epistemic-aware dataframe with GUM variance propagation.

## Architecture

- `pure/types.sio` - DataFrame, Column types
- `pure/epistemic.sio` - Epistemic column operations with uncertainty
- `lib.sio` - Public API

## Storage Model

- Fixed capacity: 256 rows × 4 columns
- Columns stored as flat arrays with type markers
- Support for string, i64, f64, bool, datetime types

## Epistemic Features

- `EColumn` wraps values with `Epistemic` type
- GUM variance propagation for arithmetic operations
- `ecolumn_mean`, `ecolumn_std`, `ecolumn_correlation` with uncertainty

## Usage

```
use dataframe::lib

var df = dataframe_new()
dataframe_add_column(&! df, "temperature", ColType::F64)
dataframe_push(&! df, "temperature", 23.5)
let mean = ecolumn_mean(&df.columns[0])
```

## Tests

`tests/stdlib/dataframe/test_dataframe_core.sio` (check-only, Madaros gate)