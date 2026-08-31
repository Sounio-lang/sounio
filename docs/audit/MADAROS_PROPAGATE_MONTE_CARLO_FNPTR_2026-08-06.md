<!-- docs:meta
topic_id: repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-propagate-monte-carlo-fnptr-2026-08-06
-->

# Madaros residual — `propagate::monte_carlo` fn-ptr (2026-08-06)

**Status:** FAIL-CLOSED (wrong result, not SEGV)  
**Gate:** `scripts/ci/madaros_propagate_monte_carlo_fnptr_failclosed_gate.sh`  
**Probe:** `tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio`

## Finding

Under tip Madaros, `monte_carlo(x, square, N)` with a named `fn(f64)->f64`:

- `souc check` OK; native compile+run exits without SEGV;
- mean/variance are invalid (sentinel `MONTE_CARLO_FNPTR FAIL`);
- `lean_single` oracle prints `MONTE_CARLO_FNPTR PASS` (≈4.01 / 0.16).

Minimal same-file fn-pointer `apply(square, 2.0)` is green — residual is the
imported generic MC shape, not fn-pointers generally.

## Supported path

`monte_carlo_identity` / `monte_carlo_square` (value-style LCG) remain green via
`scripts/madaros_propagate_native_gate.sh`.

## Companion remeasure

Imported exclusive-ref Xoshiro first-draw is **GREEN** on tip Madaros:
`scripts/ci/madaros_xoshiro_imported_gate.sh` → `MADAROS_XOSHIRO_IMPORTED_GATE_OK`.
