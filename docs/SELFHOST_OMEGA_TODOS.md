# Self-Hosted Omega TODOs

This file tracks open migration work while keeping Sprint 1 scope intact.
No existing TODOs were removed; this is an additive backlog.

## Rules

- Keep `L1/L2/L3 + QIR telemetry` as the hard gate path.
- Keep hardware-first implementation for K-AXI and accumulator.
- Avoid new Rust glue unless explicitly approved.

## Open TODOs

- [x] Remove optional `template-direct` fallback from `souc build --target hardware*` once we no longer need compatibility mode outside strict gate (strict gate already enforces `mode=selfhost-emitter`).
- [x] Port host-side hardware Epistemic Power polling into self-hosted `.sio` control paths.
- [x] Add self-hosted `.sio` adapters for K-AXI transaction flag packing and replay inspection.
- [x] Replace provenance XOR fold with Merkle-core-compatible lane interface in RTL.
- [x] Upgrade K-AXI variance propagation from fixed approximations to richer GUM op profiles.
- [x] Add bidirectional K-AXI return channel design doc (kept out of Sprint 1 hard gate).
- [x] Integrate quantum controller counter lane into accumulator input feed.
- [x] Add reproducibility artifact schema for hardware counters across reruns.
- [x] Add self-hosted regression suite focused on hardware telemetry contracts.
- [x] Wire K-AXI adapter self-check into an automated self-hosted test invocation.
- [x] Wire hardware_publish self-check into strict Sprint 1 gate (pure Sounio path).
- [x] Wire pure-Sounio PTX launch self-check (`tests/hardware/real_ptx_launch_test.sio`) into strict Sprint 1 gate.
- [x] Emit PTX launch telemetry artifact (`artifacts/ptx/omega/ptx_launch_report.json`) with schema validation in hardware telemetry regression gate.
- [x] Integrate PTX launch report into Epistemic Power score path (software + hybrid log-space Q32.32 helpers in pure Sounio).
- [x] Add hardware live-read telemetry artifact (`artifacts/fpga/hardware_epistemic_power_live.v1.json`) and gate wiring for `OMEGA_REQUIRE_HW_EPI_POWER`.
- [x] Default `OMEGA_REQUIRE_HW_EPI_POWER=1` in Sprint 1 gate and keep strict fallback-safe enforcement.
- [x] Add historical drift tracking for hardware live-read telemetry (`artifacts/fpga/hardware_epistemic_power_live_trend.v1.json`).
- [x] Bridge Omega artifacts into RL readiness evidence (`bootstrap/policies/rl_readiness.evidence.json`) with status smoke wiring.
- [x] Add explicit shadow-audit artifact (`artifacts/omega/shadow_audit.v1.json`) wired before RL readiness bridge.
- [x] Add RL readiness historical trend artifact (`artifacts/omega/rl_readiness_trend.v1.json`) with drift guards.
- [x] Enforce Sprint 1 policy-mode governance with guard artifact (`artifacts/omega/policy_mode_guard.v1.json`) keeping shadow default.
- [x] Add governance attestation artifact (`artifacts/omega/governance_attestation.v1.json`) with aggregate hash/signature hooks.
- [x] Add deterministic multi-run replay artifact for RL readiness (`artifacts/omega/rl_readiness_replay.v1.json`).
- [x] Require minimum stable trailing runs for `policy_mode=active` via policy-mode guard (`--min-stable-runs`, default `5`).
- [x] Add weekly governance drift report generator (`artifacts/omega/weekly_drift_report.v1.json` + `.md`) and scheduled workflow.
- [x] Add Sprint 1 release-readiness verdict artifact (`artifacts/omega/sprint1_release_readiness.v1.json`) with `hard+rollover` semantics.
- [x] Add Sprint 1 performance summary artifact (`artifacts/omega/performance_summary.v1.json`) with contract substitution governance for unavailable external baselines.
- [x] Enforce `OMEGA_REQUIRE_SPRINT1_SUCCESS_CRITERIA_NOW=1` by default in Sprint 1 gate.
- [x] Enforce signed governance attestation by default with local auto-key fallback when no key is provided.
- [x] Add external baseline collection artifact (`artifacts/omega/external_baseline_collection.v1.json`) and wire performance summary to prefer real external ingest when available.
- [x] Default `OMEGA_RUN_EXTERNAL_BASELINES=1` in Sprint 1 gate with adapter probes that degrade to governed substitution when dependencies are missing.
- [x] Add reusable external baseline stack installer (`scripts/omega/omega_install_external_baseline_stack.sh`) and venv-aware baseline probe wiring (`OMEGA_BASELINE_PYTHON`).
- [x] Add custom external baseline command hook templates (`scripts/omega/external_baseline_cmds.env.example`) and report emitter utility (`scripts/omega/omega_emit_baseline_report.py`).
- [x] Bridge legacy `RUN_EXTERNAL_BASELINES` to `OMEGA_RUN_EXTERNAL_BASELINES` inside Sprint 1 gate contract stage so external adapters run by default without manual env sync.
- [x] Default `OMEGA_REQUIRE_PURE_SOUNIO_KAXI=1` in Sprint 1 gate, while preserving override to `0` for explicit compatibility runs.
- [x] Add policy-smoke signature hardening (`scripts/omega/omega_prepare_policy_smoke.sh`) and wire signed policy smoke path into independence + Omega RL status checks.
- [x] Add baseline freeze artifact (`artifacts/omega/baseline_freeze.v1.json`) and enforce it in Sprint 1 gate + hardware telemetry regression strict path.
- [x] Extend weekly governance drift report to include baseline freeze summary and digest tracking.
- [x] Enforce canonical bootstrap key workflow (`scripts/omega_canonical_key_bootstrap.sh`) across policy status smoke, baseline freeze signing, and strict gate marker `CANONICAL_KEY_BOOTSTRAP_PASS`.
- [x] Replace policy status smoke path with canonical verifier wrapper (`scripts/omega/omega_policy_status.sh`) that hard-fails unless `signature=verified (canonical bootstrap key)`.
- [x] Wire governance attestation signing to canonical bootstrap key, removing ephemeral auto-generated PEM key fallback from Sprint 1 gate.
- [ ] Add first-class `souc opt policy sign --policy <path>` command to support in-place signing of existing policy documents without regeneration (currently unavailable in non-Rust scope).

## Nice-to-have (post Sprint 1)

- [x] Formalize accumulator approximation bounds against software log-space reference.
- [x] Add waveform snapshots for K-AXI + accumulator TBs in `artifacts/fpga/`.
- [x] Add hardware resource trend tracking (cells/wires over time) to gate report.
