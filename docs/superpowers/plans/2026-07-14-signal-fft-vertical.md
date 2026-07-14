<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-signal-fft-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-signal-fft-vertical
-->

# Signal::fft Hardening — Implementation Plan

> Compile-and-run is the gate (default Madaros — signal::fft is self-contained). Coordination: signal/ is cold/disjoint.

Spec: `docs/superpowers/specs/2026-07-14-signal-fft-vertical-design.md`.

## Task 1 — pub fields + header note
- [ ] Make `ComplexArray` fields `pub` in `stdlib/signal/fft.sio`; add a header usage note (construct `var arr = ComplexArray { re: [0.0;256], im: [0.0;256], n: 0 }`, pass by `&!`). `souc check` green. Prove import+run (DC signal → |X|[0]=4). Commit.

## Task 2 — run-proof driver
- [ ] `tests/stdlib/signal/test_fft_stdlib.sio` (inline in main). Assert: DC [1,1,1,1]→|X|[0]=4/rest 0; impulse→flat 1; round-trip forward+inverse recovers input; cosine cos(2πi/8)→peaks bins 1,7=4, exact-zero bins 0,2,4,6, leakage bins ≤2e-3. `FFT_STDLIB_OK`. Compile+run (default Madaros). No tolerance-retrofit. Commit.

## Task 3 — consumer example
- [ ] `examples/signal/spectrum_report.sio` — two-tone spectrum report (bin 1 = 8, bin 3 = 4). Compile+run. Commit.

## Task 4 — gate
- [ ] `scripts/signal_fft_gate.sh` — check fft.sio; compile+run driver (grep `FFT_STDLIB_OK`); compile+run example; `SIGNAL_FFT_GATE_OK`. Run it. Commit.

## Task 5 — math-review + PR
- [ ] `bin/llm-offload -t math-review -p xai` on the FFT properties; log it.
- [ ] `node scripts/docs/sync_governance_metadata.mjs`; commit governance + docs.
- [ ] Push; PR to `main`; ensure Contracts/CI Decision green; merge (rebase-on-conflict if needed).
