# stdlib/dataframe

Epistemic-aware dataframe with GUM variance propagation.

## Architecture

- `pure/types.sio` - DataFrame, Column types
- `lib.sio` - Public API

## Storage Model

- Fixed capacity: 256 rows × 4 columns
- Columns stored as flat arrays with type markers
- Support for string, i64, f64, bool, datetime types

## Usage

```
use dataframe::lib

var df = dataframe_new()
dataframe_add_column(&! df, "temperature", ColType::F64)
dataframe_push(&! df, "temperature", 23.5)
```

## Tests

`tests/stdlib/dataframe/test_dataframe_core.sio` (check-only, Madaros gate)