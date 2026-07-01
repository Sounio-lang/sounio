# stdlib/simulation

Monte Carlo and time series simulation with epistemic uncertainty.

## Architecture

- `pure/types.sio` - TimeSeries, MonteCarloResult types
- `pure/epistemic.sio` - Epistemic time series operations
- `lib.sio` - Public API

## Capabilities

- TimeSeries with fixed 256-point capacity
- Monte Carlo sampling (normal, uniform distributions)
- Statistical aggregations: mean, std, variance
- Epistemic variants with GUM variance propagation

## Epistemic Features

- `ETimeSeries` wraps measurements with uncertainty
- `etimeseries_mean`, `etimeseries_std` propagate variance
- `emonte_carlo_normal` returns epistemic results

## Usage

```
use simulation::lib

var ts = etimeseries_new()
etimeseries_push(&! ts, 0.0, Epistemic::measured(1.0, 0.1))
etimeseries_push(&! ts, 1.0, Epistemic::measured(2.0, 0.1))
let mean = etimeseries_mean(&ts)
```

## Tests

`tests/stdlib/simulation/test_simulation_core.sio` (check-only, Madaros gate)