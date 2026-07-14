<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-signal-fft-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-signal-fft-vertical-design
-->

# Design — Harden the signal::fft vertical

**Status:** approved design, pre-implementation
**Date:** 2026-07-14
**Constraint:** No compiler changes. **Coordination: `signal/` is stable/cold (last touched by an unrelated
GPU commit; no open PR); disjoint from the active `stats-suite-*`, `codex/*` (IR), and `special` lanes.**
EN-UK.

## 1. Why
Seventh application of the playbook (GUM #860, units #873, linalg #892, prob #902, stats #909). `signal::fft`
is a high-value, self-contained real-world module (spectral analysis) that native-compiles under the
default Madaros engine (no lean_single needed) — the cleanest vertical yet.

## 2. Verified starting state
- `stdlib/signal/fft.sio` — radix-2 Cooley–Tukey FFT (N ≤ 256, power of 2), self-contained, green.
  `ComplexArray { re[256], im[256], n }` passed **by reference** (`&!`) — the module is explicitly designed
  to avoid large-struct SRET, so it sidesteps the cross-module by-value corruption (#913).
- API: `ca_load_real`, `ca_clear`, `fft_forward`, `fft_inverse`, `fft_magnitude`, `fft_power`, `fft_phase`,
  plus `fft_pi/sqrt/sin/cos`.
- **Gap:** `ComplexArray`'s fields were **private**, so an importing program could not construct one → the
  module was un-usable externally. Making them `pub` (additive, no behaviour change) closes this, matching
  the module's documented "callers declare `var arr = ComplexArray {...}`" pattern.
- **Runs externally under default Madaros** (verified): DC [1,1,1,1] → |X|[0]=4, others 0.

## 3. Goal
A program can `use signal::fft::*`, transform a real signal, and read its spectrum — proven by
compile-and-run against known FFT properties, gated, under the default engine.

## 4. Scope
### In
1. **Make `ComplexArray` fields `pub`** + header usage note (the only source change; additive).
2. **Run-proof driver** — DC signal, impulse (flat spectrum), forward→inverse round-trip, single-cosine
   frequency detection.
3. **Consumer example** — two-tone spectrum report.
4. **Gate** (default Madaros).
### Out
- No new transforms; no edit to other signal files (`spectral.sio`/`epoch.sio` fail check — untouched).
- No compiler edits.
- Math-review of the FFT properties is run and logged.

## 5. Design — run-proof assertions (known FFT properties)
- **DC** [1,1,1,1], N=4: |X|[0]=4 (=Σ), |X|[k>0]=0.
- **Impulse** [1,0,0,0], N=4: flat spectrum |X|[k]=1 ∀k.
- **Round-trip**: `fft_forward` then `fft_inverse` recovers the input (`arr.re[i]` ≈ original).
- **Single cosine** cos(2πi/8), N=8: peaks at bins 1 and 7 = N/2 = 4; exact-zero bins 0,2,4,6; residual at
  bins 3,5 ≤ 2e-3 is `fft_cos` input precision (~1e-3), NOT FFT error (documented, not retrofitted).
All inline in `main`; `ComplexArray` constructed in the caller frame, passed by `&!`.

## 6. Module layout
```
stdlib/signal/fft.sio                       (modify: pub fields + header note)
tests/stdlib/signal/test_fft_stdlib.sio     (new: run-proof driver)
examples/signal/spectrum_report.sio         (new: consumer example)
scripts/signal_fft_gate.sh                   (new: default-Madaros compile+run gate)
```

## 7. Verification
- `souc check stdlib/signal/fft.sio` green.
- `souc compile … && ./elf` (default Madaros) for driver + example.
- `scripts/signal_fft_gate.sh` → `SIGNAL_FFT_GATE_OK`.
- Math-review of the FFT properties logged.

## 8. Success criteria
1. A program `use`s `signal::fft`, runs under default Madaros, and reads a correct spectrum.
2. Run-proof asserts known FFT properties and passes.
3. Gate green.
4. Only `signal/fft.sio` touched (pub fields + note); disjoint from active lanes; no compiler files.

## 9. Risks
| Risk | Mitigation |
|---|---|
| fft_cos precision leaks into "zero" bins | Assert exact-zero bins strictly; bound leakage bins at the measured input-precision level (documented). |
| Making fields pub breaks the module's self-test | Additive; `souc check` stays green; run-proof verifies behaviour. |
| Another lane touches signal/ | signal/ is cold (no open PR); rebase-on-conflict if needed. |
