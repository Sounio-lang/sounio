# DISPATCH — Vancomycin correlation enclosure gate (reclassified)

**Opened:** 2026-06-29  
**Blocker-ID:** `BLK-20260629-stdlib-vancomycin-correlation-enclosure`  
**Status:** open — reclassified `compiler-semantics` (Madaros PBox marginal read)  
**Severity:** B1 (clinical E2E 2/3; dissertação Fréchet gate blocked on Madaros default)  
**Class:** `compiler-semantics` (multimodule struct return / field read) — **not** `stdlib-math`  
**Owner:** unassigned  
**Lane:** `stdlib/clinical` + Madaros native multimodule  
**Worktree:** `/workspace/sounio`  
**Branch:** `research/solver-ts3-parallel`  
**Evidence level:** E2 (bisection + lean_single contrast)

**Toolchain:**

| Engine | Identity |
|---|---|
| Madaros default | `artifacts/self-hosted/madaros` md5 `1a090ac0e4ac3df67ad2bb47c11279d0` |
| lean_single contrast | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc` |

**Related:** `BLK-20260629-stdlib-sret-pbox-clinical` (resolved) — same PBox consumption family; different entrypoint (`vp_vc_to_pbox` marginal lift vs `predict_cmin_knightian` vacuous path).

---

## §1 — Symptom (harness)

```text
FAIL  tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio (run exited 1)
```

Harness expects stdout `ENCLOSURE SMOKE PASS`. Native `println` from the ELF is not always visible in `souc run` compile noise; **exit code is authoritative**.

---

## §2 — Reproduction

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

# Production gate
./bin/souc run tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio
# Madaros: exit 1

# lean_single control (stdlib math + test logic OK)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio
# exit 0

# Primary isolator — marginal Vc PBox read
./bin/souc run docs/audit/VANCOMYCIN_CORRELATION_ENCLOSURE_2026-06-29/reference/vc_pbox_lo_probe.sio
# Madaros: exit 211 (pb_lo_mean(&vc_box) <= 0)
# lean_single: exit 0

# Band contract probe (all marginals)
./bin/souc run docs/audit/VANCOMYCIN_CORRELATION_ENCLOSURE_2026-06-29/reference/correlation_band_probe.sio
# Madaros: exit 211
# lean_single: exit 0
```

---

## §3 — Bisection matrix (E2)

| Witness | Madaros | lean_single | Notes |
|---|---:|---:|---|
| `vc_pbox_lo_probe.sio` | **211** | 0 | `vp_vc_to_pbox` → `pb_lo_mean` ≤ 0 (garbage read) |
| `correlation_band_probe.sio` | **211** | 0 | Fails `vc_lo > 0` gate before Phase A/B |
| `correlation_phase_a_only.sio` | **201** | — | Contract gate (encoded 201 = marginal ≤ 0) |
| `correlation_phase_b_only.sio` | **201** | — | Never reached — marginals invalid on Madaros |
| Full production test | **1** | **0** | Early `ENCLOSURE SMOKE FAIL` contract branch |

**Exit code map (witnesses):**

| Code | Meaning |
|---:|---|
| 0 | Pass |
| 1 | Full test: one or more enclosure failures (or contract fail with IO path) |
| 201 | Witness: any marginal endpoint ≤ 0 |
| 211–214 | `correlation_band_probe`: vc_lo / vc_hi / cl_lo / cl_hi ≤ 0 |
| 215 | Band inverted |

---

## §4 — Root-cause hypothesis (E2)

1. **Not** Fréchet monotonicity math: lean_single runs the **full** Phase A + Phase B test green.
2. **Not** Phase B correlation sampler in isolation: Phase A never runs on Madaros because marginals read as non-positive.
3. **Is:** Madaros multimodule native path returns or projects `PBox` from `vp_vc_to_pbox` / `vp_cl_to_pbox` such that `pb_lo_mean` observes `lo_mean ≤ 0` for physiologically valid inputs (`weight=70`, `crcl=80`, `tdm=3` → expected `vc_lo ≈ 29.75 L`).
4. `test_vancomycin_pbpk_v2.sio` passes on Madaros because it never calls `vp_vc_to_pbox` in `main` — only `predict_cmin_knightian` + `is_safe_dose` / `pb_gap` on the **Cmin** box.

---

## §5 — Acceptance gates

| Gate | Required |
|---|---|
| `vc_pbox_lo_probe.sio` | Madaros exit 0 |
| `correlation_band_probe.sio` | Madaros exit 0 |
| `correlation_phase_a_only.sio` | Madaros exit 0 |
| `correlation_phase_b_only.sio` | Madaros exit 0 |
| `test_vancomycin_correlation_sensitivity.sio` | exit 0 + stdout `ENCLOSURE SMOKE PASS` |
| lean_single regression | Must remain exit 0 (no stdlib math drift) |

**Do not** weaken `inside_band` ε or alter corner monotonicity comments to pass Madaros without compiler fix + `bin/llm-offload -t review -p deepseek`.

---

## §6 — Next action

1. Compiler lane: IR dump merged calls for `vp_vc_to_pbox` consumer path (`SOUNIO_DUMP_MERGED_CALLS=1`), compare with resolved `pb_new` / field-layout indices (same playbook as `BLK-20260629-stdlib-sret-pbox-clinical`).
2. Extend merge-finalize fix if additional stale `fn_id` or struct-return slots affect marginal `PBox` lifts.
3. Re-run witnesses §5 then close blocker.

---

## §7 — Reference files

| File | Role |
|---|---|
| `reference/vc_pbox_lo_probe.sio` | **Primary** isolator |
| `reference/correlation_band_probe.sio` | Marginal + Cmin band contract |
| `reference/correlation_phase_a_only.sio` | Phase A failure count (= exit) |
| `reference/correlation_phase_b_only.sio` | Phase B failure count (= exit) |
| `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio` | Production gate |