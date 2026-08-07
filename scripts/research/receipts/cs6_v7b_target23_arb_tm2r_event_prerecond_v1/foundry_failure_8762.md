# Job 8762 — accepted scientifically, verify papercut on combined time

## Worker result

- Classification: `PREDICTOR_CENTERED_PRERECOND_EVENT_ACCEPTED`
- Accepted power: `12`
- Raw projection preserves all six variables including `rho3`
- Post-QR residual rank forensic: `3` with reconditioned `rho3` false
- Exact section and strict Newton inclusion hold

## Verify failure

```text
prerecond verify error: combined event time is inconsistent with the centered chart
```

Cause: `combined_event_time` was formed as Arb addition
`rational_ball(center) + event_time_range`, whose serialized endpoints do not
equal the exact rational sum of the frozen center and the serialized residual
event-time endpoints. Fix: emit exact `Fraction` endpoints
`center + event_{lower,upper}`.

## Claim boundary

Even after a clean verify, acceptance is only an event-local residual chart
gate. It is not a covering relation, recurrent graph, chaos proof, or open
problem solution.
