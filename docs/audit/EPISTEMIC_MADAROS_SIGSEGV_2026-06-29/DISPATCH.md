<!-- docs:meta
topic_id: repo.docs.audit.epistemic-madaros-sigsegv-2026-06-29.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-madaros-sigsegv-2026-06-29.dispatch
-->

# DISPATCH — Epistemic filter SIGSEGV cluster on Madaros default engine

**Opened:** 2026-06-29  
**Blocker-ID:** `BLK-20260629-stdlib-epistemic-madaros-sigsegv`  
**Status:** open  
**Severity:** B2 (epistemic E2E 18/60 on Madaros; 10× exit 139 in `epistemic` filter)  
**Class:** `compiler-runtime` (Madaros multimodule native) — **not** stdlib epistemic math  
**Owner:** Composer/compiler lane (serialized with correlation PBox fix)  
**Lane:** Madaros default + `tests/run-pass/epistemic_*`  
**Worktree:** `/workspace/sounio`  
**Branch:** `research/solver-ts3-parallel`  
**Evidence level:** E2 (paired Madaros vs lean_single on 10 witnesses)

**Baseline source:** `/tmp/epistemic_baseline_2026-06-29.log` — `bash scripts/run_sio_test_suite.sh epistemic --verbose` → Pass 18 / Fail 42 / Skip 2 / Total 62.

**Toolchain:**

| Engine | Role |
|---|---|
| Madaros | `./bin/souc` default → `artifacts/self-hosted/madaros` md5 `1a090ac0e4ac3df67ad2bb47c11279d0` |
| Control | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc` |

---

## §1 — Symptom

Ten tests in the `epistemic` harness filter fail with **exit 139** (SIGSEGV) on Madaros. All ten **`souc check` = 0**.

```text
FAIL  epistemic_arena_h2o.sio           (run exited 139)
FAIL  epistemic_bmi.sio                (run exited 139)
FAIL  epistemic_hessian_8inputs.sio    (run exited 139)
FAIL  epistemic_hessian_of.sio         (run exited 139)
FAIL  epistemic_mcts.sio               (run exited 139)
FAIL  epistemic_mcts_full.sio          (run exited 139)
FAIL  epistemic_molecule_h2o.sio       (run exited 139)
FAIL  epistemic_ode_14comp.sio         (run exited 139)
FAIL  epistemic_pbpk_multidrug.sio     (run exited 139)
FAIL  rapamycin_epistemic_adaptive.sio (run exited 139)
```

---

## §2 — lean_single contrast (E2 — primary evidence)

**All 10 pass on lean_single** (`run exit 0`). Stdlib + test logic is not the primary defect class for this cluster.

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
for t in epistemic_arena_h2o epistemic_bmi epistemic_hessian_8inputs \
         epistemic_hessian_of epistemic_mcts epistemic_mcts_full \
         epistemic_molecule_h2o epistemic_ode_14comp epistemic_pbpk_multidrug \
         rapamycin_epistemic_adaptive; do
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "tests/run-pass/${t}.sio" >/dev/null 2>&1
  echo "$t exit=$?"
done
# all exit=0 (measured 2026-06-29)
```

---

## §3 — Classification matrix

| Test path | Subsystem | Imports / shape | Madaros | lean_single | Lane |
|---|---|---|---:|---:|---|
| `tests/run-pass/epistemic_arena_h2o.sio` | chemistry arena | multimodule | 139 | 0 | compiler |
| `tests/run-pass/epistemic_bmi.sio` | epistemic GUM | small | 139 | 0 | compiler |
| `tests/run-pass/epistemic_hessian_8inputs.sio` | autodiff Hessian | heavy | 139 | 0 | compiler |
| `tests/run-pass/epistemic_hessian_of.sio` | autodiff Hessian | heavy | 139 | 0 | compiler |
| `tests/run-pass/epistemic_mcts.sio` | search / PUCT | self-contained structs | 139 | 0 | compiler |
| `tests/run-pass/epistemic_mcts_full.sio` | search / PUCT | extended | 139 | 0 | compiler |
| `tests/run-pass/epistemic_molecule_h2o.sio` | chemistry | multimodule | 139 | 0 | compiler |
| `tests/run-pass/epistemic_ode_14comp.sio` | PBPK ODE 14-comp | large single file | 139 | 0 | compiler |
| `tests/run-pass/epistemic_pbpk_multidrug.sio` | PBPK multidrug | multimodule | 139 | 0 | compiler |
| `tests/run-pass/rapamycin_epistemic_adaptive.sio` | dissertation PBPK | multimodule clinical | 139 | 0 | compiler |

**Pattern:** heterogeneous tests — no single stdlib file owns the bug. Failure is **engine-specific**.

---

## §4 — Separation from other epistemic failures (same baseline)

The remaining **32** failures in the 42-fail bucket are **not** this dispatch:

| Class | Count (approx.) | Exit | Owner lane |
|---|---:|---:|---|
| Logic / math / API | 28 | 1 | stdlib + offload |
| Harness annotation | 1 | compile-fail mismatch | harness |
| Timeout | 1 | 124 | graphics smoke |
| Other | 2 | 6, 41 | case-by-case |

**Do not** patch `stdlib/epistemic/` math to fix 139s without lean_single regression proof.

---

## §5 — Relationship to clinical PBox fix

`BLK-20260629-stdlib-sret-pbox-clinical` (resolved) fixed one merged-IR call-target family affecting `predict_cmin_knightian` vacuous consumption.

This cluster is **broader**: large/run-pass epistemic programs that never touch `clinical::vancomycin_pbpk` still SIGSEGV on Madaros. Likely shared native codegen / multimodule linking substrate; may require additional compiler work beyond the merge-finalize patch.

**Related open clinical blocker:** `BLK-20260629-stdlib-vancomycin-correlation-enclosure` — Madaros `vp_vc_to_pbox` marginal read (exit 211), not SIGSEGV but same PBox struct-return family.

---

## §6 — Acceptance gates

| Gate | Target |
|---|---|
| All §3 tests | Madaros `run exit 0` |
| lean_single regression | All §3 remain exit 0 |
| Epistemic filter | ≥ 28/60 pass (recover 10) without increasing 139 count elsewhere |
| No stdlib tolerance drift | Failures moving 139→1 require separate math dispatch |

---

## §7 — Suggested compiler bisection order

1. **Smallest:** `epistemic_bmi.sio` — fewest moving parts among 139s.
2. **Self-contained:** `epistemic_mcts.sio` — no stdlib import in header; struct-heavy.
3. **Heavy:** `epistemic_ode_14comp.sio` — stress (14-comp PBPK + GUM).
4. **Dissertation-linked:** `rapamycin_epistemic_adaptive.sio` — PBPK path.

Forensic playbook: gdb on Madaros ELF, `SOUNIO_DUMP_MERGED_CALLS=1`, compare fn_id/name resolution against lean_single codegen for the same source.

---

## §8 — Coordination

See `docs/audit/COORDINATION_2026-06-29_PBOX_LANES.md`. **No parallel edits** to `module_frontend.sio` on other worktrees until this cluster closes or handoff is recorded.

GPU lanes (`/tmp/sounio-gpu-*`) and `claude/codegen-largestruct-fix` are **disjoint** (no active `module_frontend` conflict at tip).

## §9 — Next action

Assign compiler lane owner to bisect `epistemic_bmi.sio` + `epistemic_mcts.sio` on Madaros vs lean_single codegen diff. Stdlib lane: **read-only** audit + witness pins — no math edits on this Blocker-ID.
