<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-signal-filter-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-signal-filter-vertical-design
-->

# Design — Harden the signal::filter vertical (coordinated, disjoint)
**Date:** 2026-07-14 · no compiler changes · `signal/` cold (no open PR touching signal/special/math). By-reference
API (`&!filter`, `&[f64;512]`) is cross-module-safe. Adds run-proof/example/gate; only `filter.sio` header optional (none needed). EN-UK.

## State
`stdlib/signal/filter.sio` — self-contained, green, native-compiles under **default Madaros**. Biosignal
digital filters: IIR1 (lowpass/highpass), IIR2 (notch/bandpass), FIR64 (moving average). Filters passed by
`&!`; batch via `&[f64;512]`/`&![f64;512]`. Verified externally: MA(4) DC gain 1, impulse 0.25×4, IIR1
lowpass const 2→2, highpass const 2→0.

## Run-proof (known DSP properties)
- **FIR MA(4)**: DC gain 1 (const 1 → 1); impulse response 0.25,0.25,0.25,0.25,0.
- **IIR1 lowpass** (fc=10, fs=1000): DC gain 1 (const 2 → 2 at steady state).
- **IIR1 highpass**: DC gain 0 (const 2 → 0; blocks DC).

## Layout / verification
`tests/stdlib/signal/test_filter_stdlib.sio`, `examples/signal/filter_report.sio`,
`scripts/signal_filter_gate.sh` (default Madaros) → `SIGNAL_FILTER_GATE_OK`. Math-review logged. No source/compiler edits.
