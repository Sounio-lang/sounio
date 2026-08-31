<!-- docs:meta
topic_id: repo.docs.internal.implementation.paper-artifact-packaging-spec
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.paper-artifact-packaging-spec
-->

# Paper Artifact Packaging Spec (Track 4)

## Purpose

This document defines a paper-grade artifact structure for Sounio with explicit, machine-checkable evidence.
It converts existing CI and self-host gate outputs into a claim-backed evidence package and adds an enterprise workload lane design (ERP/e-commerce backend scenario) without exposing proprietary implementation details.

## Scope

- In scope: reproducibility protocol, gate markers, claim-to-evidence mapping, and enterprise-lane acceptance criteria.
- Out of scope: publishing proprietary code, data, or customer-specific benchmark fixtures.

## Hypotheses

- `H1`: Core compiler quality is reproducible from public sources using scripted gates.
- `H2`: Backend coverage claims are supported by explicit native/LLVM/GPU/ontology checks.
- `H3`: Self-host maturity claims are supported by strict no-fallback parse-all gates.
- `H4`: Enterprise backend behavior can be evaluated via anonymized compatibility cases with measurable pass/fail criteria.
- `H5`: Paper claims can be traced to concrete command outputs, file artifacts, and CI jobs.

## Lanes

### Lane A: Public Core Reproducibility

- Command surface:
  - `scripts/dev/fast_gate.sh`
  - `scripts/dev/e2e_gate.sh` (called by `fast_gate.sh`)
  - `scripts/ci/check_feature_matrix.sh`
  - `scripts/selfhost/selfhost_zero_fallback_gate.sh`
  - `scripts/ci/selfhost_driver_output_gate.sh`
  - `scripts/ci/poseidon_gate.sh`
- Primary CI mapping:
  - `.github/workflows/ci.yml` jobs: `fast-gate`, `selfhost-zero-fallback`, `joss-smoke`.

### Lane B: Enterprise Workload (ERP/e-commerce backend, anonymized)

Design goal: validate compiler/runtime parity and self-host behavior for a transactional backend workload profile without publishing proprietary business logic.

Workload classes (anonymized):

- `order-intake`: request validation and schema-level checks.
- `pricing-ledger`: discount/tax/total computations and accounting transitions.
- `inventory-commit`: stock reservation/decrement and consistency checks.
- `payment-reconcile`: idempotency/retry-safe reconciliation transitions.

Execution mechanism (already implemented):

- Use `scripts/ci/poseidon_gate.sh` with `MATRIX_FILE=<private_matrix>`.
- Matrix schema is defined in `scripts/poseidon_compat_matrix.txt`:
  - `case_id|mode|command|compare`
  - modes: `run_selfhost` and `direct`
  - compare tokens: `exit`, `stdout`, `stderr`

Required acceptance criteria:

- All enterprise matrix cases emit `PASS [<case_id>] baseline and candidate match (...)`.
- Gate emits `PASS [full-selfhost] ...`.
- Summary emits `Summary: PASS=<N> FAIL=0`.
- If `SOUNIO_SELFHOST_NO_RUST_FALLBACK=1`, no fallback/oracle markers are present in gate logs:
  - `SELFHOST=driver-first schema=v1 event=driver_orchestration ... status=fallback`
  - `SELFHOST=oracle backend=rust`

## Metrics and Gates

| Metric | Command | Expected marker(s) | Acceptance gate |
|---|---|---|---|
| Fast regression gate completion | `bash scripts/dev/fast_gate.sh` | `[fast-gate] ok` | Marker present |
| End-to-end backend checks | `bash scripts/dev/e2e_gate.sh` | `[e2e] ok` | Marker present |
| Feature matrix checks | `bash scripts/ci/check_feature_matrix.sh` | `[feature-matrix] ok` | Marker present |
| Self-host strict zero fallback summary | `bash scripts/selfhost/selfhost_zero_fallback_gate.sh` | `SELFHOST_ZERO_GATE_SUMMARY pass=<n> fail=<m>` | `fail=0` |
| Self-host driver output smoke | `bash scripts/ci/selfhost_driver_output_gate.sh` | `SELFHOST_DRIVER_OUTPUT_GATE_SUMMARY pass=<n> fail=<m>` | `fail=0` |
| Parse-all report completeness | same as above | `PASS [parse-all-report] ...` | Pass marker present |
| Parse-all shard completeness | same as above | `PASS [parse-all-shards] ...` | Pass marker present |
| Full self-host strict pass | same as above | `PASS [full-selfhost] strict no-fallback gate passed` | Pass marker present |
| Compatibility parity matrix | `bash scripts/ci/poseidon_gate.sh` | `PASS [<case_id>] baseline and candidate match (...)` | All matrix cases pass |
| Poseidon full-suite summary | same as above | `Summary: PASS=<n> FAIL=<m>` | `FAIL=0` |
| JOSS smoke examples | CI `joss-smoke` in `.github/workflows/ci.yml` | step success (`Run required JOSS examples`) | Job green |
| Compile-fail diagnostic behavior | CI `joss-smoke` in `.github/workflows/ci.yml` | step success (`Compile-fail smoke ...`) | Job green |

## Claim -> Evidence Matrix

| Claim ID | Paper-facing claim | Evidence source(s) | Marker(s) / artifact(s) |
|---|---|---|---|
| `C1` | Sounio build/test path is reproducible on public runners | `.github/workflows/ci.yml` `fast-gate`; `scripts/dev/fast_gate.sh` | `[fast-gate] ok` |
| `C2` | Multi-backend pipeline is exercised (native, LLVM when available, GPU compile smoke + runtime attestation gate) | `scripts/dev/e2e_gate.sh`, CI `fast-gate` | `[e2e] native build + run`, `[e2e] llvm ...` or skip marker, `[e2e] gpu backend compile smoke`, `[e2e] gpu runtime attestation gate`, `[e2e] ok` |
| `C3` | Self-host path passes strict no-fallback corpus gate | `scripts/selfhost/selfhost_zero_fallback_gate.sh`, CI `selfhost-zero-fallback` | `PASS [full-selfhost] ...`, `PASS [parse-all-report] ...`, `PASS [parse-all-shards] ...`, `SELFHOST_ZERO_GATE_SUMMARY ... fail=0` |
| `C3b` | Self-host driver can emit decodable bytecode artifacts (bootstrap subset) | `scripts/ci/selfhost_driver_output_gate.sh` | `PASS [ret_42] ...`, `PASS [print_boot] ...`, `SELFHOST_DRIVER_OUTPUT_GATE_SUMMARY ... fail=0` |
| `C4` | Ontology mismatch diagnostics are enforced in end-to-end checks | `scripts/dev/e2e_gate.sh` ontology cross-check | presence of `semantic distance` diagnostic in failure path check |
| `C5` | Self-host/non-self-host parity is regression-tested | `scripts/ci/poseidon_gate.sh`, `scripts/poseidon_compat_matrix.txt` | `PASS [<case_id>] baseline and candidate match (...)`, `Summary: PASS=<n> FAIL=0` |
| `C6` | Enterprise backend workload compatibility can be validated without disclosing proprietary implementation details | `scripts/ci/poseidon_gate.sh` with private `MATRIX_FILE` | anonymized `PASS [<enterprise_case>] ...` lines + `FAIL=0` summary + no fallback markers in strict mode |

## Reproducibility Protocol

### 1) Build an artifact bundle directory

```bash
ARTIFACT_ROOT="/tmp/sounio-paper-artifact-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARTIFACT_ROOT"
```

### 2) Run public lanes and collect logs

```bash
bash scripts/dev/fast_gate.sh | tee "$ARTIFACT_ROOT/fast_gate.log"
bash scripts/ci/check_feature_matrix.sh | tee "$ARTIFACT_ROOT/feature_matrix.log"
WORK_DIR="$ARTIFACT_ROOT/selfhost-zero" \
  bash scripts/selfhost/selfhost_zero_fallback_gate.sh | tee "$ARTIFACT_ROOT/selfhost_zero_gate.log"
WORK_DIR="$ARTIFACT_ROOT/selfhost-driver-output" \
  bash scripts/ci/selfhost_driver_output_gate.sh | tee "$ARTIFACT_ROOT/selfhost_driver_output_gate.log"
WORK_DIR="$ARTIFACT_ROOT/poseidon-public" \
  bash scripts/ci/poseidon_gate.sh | tee "$ARTIFACT_ROOT/poseidon_public.log"
```

Expected log markers:

- `"[fast-gate] ok"` in `fast_gate.log`
- `"[feature-matrix] ok"` in `feature_matrix.log`
- `"SELFHOST_ZERO_GATE_SUMMARY ... fail=0"` in `selfhost_zero_gate.log`
- `"SELFHOST_DRIVER_OUTPUT_GATE_SUMMARY ... fail=0"` in `selfhost_driver_output_gate.log`
- `"Summary: PASS="` plus `"FAIL=0"` in `poseidon_public.log`

### 3) Run enterprise lane with private matrix (no proprietary details published)

```bash
WORK_DIR="$ARTIFACT_ROOT/poseidon-enterprise" \
MATRIX_FILE="/secure/path/enterprise_compat_matrix.txt" \
SOUNIO_SELFHOST_NO_RUST_FALLBACK=1 \
bash scripts/ci/poseidon_gate.sh | tee "$ARTIFACT_ROOT/poseidon_enterprise.log"
```

Expected markers:

- `PASS [<enterprise_case_id>] baseline and candidate match (...)`
- `PASS [full-selfhost] ...`
- `Summary: PASS=<n> FAIL=0`
- No `SELFHOST=driver-first schema=v1 event=driver_orchestration ... status=fallback` in strict mode.

### 4) Verify marker presence quickly

```bash
rg -n "\\[fast-gate\\] ok" "$ARTIFACT_ROOT/fast_gate.log"
rg -n "\\[feature-matrix\\] ok" "$ARTIFACT_ROOT/feature_matrix.log"
rg -n "SELFHOST_ZERO_GATE_SUMMARY .*fail=0" "$ARTIFACT_ROOT/selfhost_zero_gate.log"
rg -n "SELFHOST_DRIVER_OUTPUT_GATE_SUMMARY .*fail=0" "$ARTIFACT_ROOT/selfhost_driver_output_gate.log"
rg -n "Summary: PASS=.* FAIL=0" "$ARTIFACT_ROOT/poseidon_public.log" "$ARTIFACT_ROOT/poseidon_enterprise.log"
```

### 5) CI mapping for paper appendix

- `fast-gate` job output (`.github/workflows/ci.yml`) -> `C1`, `C2`
- `selfhost-zero-fallback` job output and uploaded artifact `selfhost-zero-fallback-${{ runner.os }}` -> `C3` (runs `scripts/selfhost/selfhost_zero_fallback_gate.sh`)
- `joss-smoke` job output -> `C1`, compile-fail behavior support for `C4`

## Threats to Validity

- Hardware/runtime variance can change runtime magnitude; this spec treats gate pass/fail markers as primary evidence.
- GPU validation in `e2e_gate` includes compile smoke plus a remote-attested runtime gate; throughput claims still require separate hardware-specific benchmarking.
- Enterprise lane hides implementation details by design; reproducibility is at the interface/marker level, not full workload publication.
- Baseline commit selection in `poseidon_gate.sh` (`BASELINE_COMMIT`) may require periodic updates to avoid stale comparisons.

## Paper Packaging Checklist

- Include this spec: `docs/internal/implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md`.
- Include logs from the reproducibility protocol under a timestamped artifact directory.
- Include a short appendix table with `Claim ID -> log file -> marker`.
- For enterprise lane publication, share only:
  - matrix schema,
  - anonymized case IDs,
  - pass/fail counts and strict-mode fallback status.
