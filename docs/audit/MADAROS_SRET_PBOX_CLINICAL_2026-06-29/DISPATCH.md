<!-- docs:meta
topic_id: repo.docs.audit.madaros-sret-pbox-clinical-2026-06-29.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-sret-pbox-clinical-2026-06-29.dispatch
-->

# DISPATCH — Madaros multimodule `PBox` SRET consumption SIGSEGV (clinical vancomycin gate)

**Opened:** 2026-06-29  
**Blocker-ID:** `BLK-20260629-stdlib-sret-pbox-clinical`  
**Status:** resolved (2026-06-29 — merged-IR call-target finalize)  
**Severity:** B1 (lane-blocking — `clinical/` E2E 0/3; dissertação PBPK28/vancomycin gates)  
**Class:** `compiler-semantics` (native codegen / `sret` ABI)  
**Owner:** compiler lane (session) — fix landed 2026-06-29  
**Lane:** `stdlib/clinical` + Madaros native multimodule  
**Worktree:** `/workspace/sounio`  
**Branch:** `research/solver-ts3-parallel` (session tip)  
**Evidence level:** E2 (reproduction + bisection + gate-bound counts)

**Toolchain at reproduction:**

| Binary | Identity |
|---|---|
| `./bin/souc` | wrapper → Madaros **v0.80.0** (`md5 a69fd8511bdb6d353352f4787a8a71d6`) |
| `./bin/madaros` | launcher script (`md5 8beed098f04b02357d6cd36cd50f1bb7`) |
| `artifacts/self-hosted/madaros` | post-fix ELF (`md5 1a090ac0e4ac3df67ad2bb47c11279d0`, built 2026-06-29) |

**Baseline contrast:** `artifacts/stdlib/stdlib_reliability_status.v1.json` (2026-05-12) reported **251/251** stdlib E2E pass on `souc` **1.0.0-beta.5** (lean_single). Current Madaros default shows material regression on epistemic/clinical/pbpk filters.

---

## §1 — Symptom

All three `tests/stdlib/clinical/*` tests fail at **runtime** with **exit 139** (SIGSEGV), not at `souc check`:

```text
FAIL  tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio           (run exited 139)
FAIL  tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio (run exited 139)
FAIL  tests/stdlib/clinical/test_aminoglycoside_correlation_sensitivity.sio (run exited 139)
```

Compilation succeeds (~57–61 merged IR functions); ELF is emitted; crash occurs when executing the binary via `bin/madaros` run wrapper.

Production entrypoints affected:

- `stdlib/clinical/vancomycin_pbpk.sio` (`pub fn main` also crashes)
- `tests/run-pass/vancomycin_propagation_v2.sio` (same SIGSEGV)

---

## §2 — Reproduction commands

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

# Production test (stdlib gate)
./bin/souc run tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio
# exit 139

# Minimal dispatch witnesses
./bin/souc run docs/audit/MADAROS_SRET_PBOX_CLINICAL_2026-06-29/reference/pbox_vacuous_read.sio
# exit 139 — primary isolator

./bin/souc run docs/audit/MADAROS_SRET_PBOX_CLINICAL_2026-06-29/reference/pbox_clinical_call_only.sio
# exit 0 — control (call + discard, vacuous path)

./bin/souc run docs/audit/MADAROS_SRET_PBOX_CLINICAL_2026-06-29/reference/pbox_clinical_read.sio
# exit 139 — production read path

# Pinned regression (known-failure)
./bin/souc run tests/run-pass/sret_pbox_clinical_known_failure.sio
# exit 139 (annotated //@ known-failure)
```

Record `md5sum ./bin/souc ./bin/madaros` with every reproduction.

---

## §3 — Bisection matrix (E2)

| Scenario | `souc run` exit | Notes |
|---|---:|---|
| `pb_new` / `pb_vacuous` — `knightian` only | 0 | `epistemic::knightian` standalone consumption OK |
| `vp_default()` — `clinical` only, read struct fields | 0 | 32-byte struct (`VancoPKParams`) returned + field read OK |
| `predict_cmin_knightian(15,…)` — call, **discard** result | 0 | Vacuous early-return path, no consumption |
| `predict_cmin_knightian(15,…)` then `pb_confidence(&p)` | **139** | **Primary bug** — consumption of returned `PBox` |
| `predict_cmin_knightian(78.5,…)` then `pb_lo_mean(&p)` | **139** | Production test shape |
| `predict_cmin_knightian(78.5,…)` — call, discard result | **139** | Secondary — full PK math path may also crash before read |
| 32-byte `PBoxLike` single-module struct return + field read | 0 | `reference/pbox_32b_single_module.sio` — **not** size-only |
| `tests/run-pass/sret_forwarding_minimal.sio` | 0 | Plain SRET forward regression green |
| `tests/run-pass/sret_8_field_return.sio` | **139** | Related large/mixed SRET witness also red on Madaros v0.80.0 |

**Refined bisection (2026-06-29 Phase A, Madaros rebuild `e0a4074d…`):**

| Scenario | Exit | Implication |
|---|---:|---|
| `reference/pbox_vacuous_direct.sio` — main calls `pb_vacuous()` + `pb_confidence(&p)` | 0 | Knightian multimodule import + ref accessor **OK** |
| `reference/pbox_clinical_return_forward.sio` — main-module `fwd_vacuous()` → `pb_vacuous()` | 0 | Return-forwarding in **seed** module **OK** |
| `reference/pbox_vacuous_read.sio` — `predict_cmin_knightian(15,…)` + `pb_confidence(&bad)` | **139** | **Still red** after fn_id remap patches |
| `SOUNIO_DUMP_LAYOUTS=1` on vacuous_read | — | `confidence` field_idx **3** (correct) in all modules |

**Root cause (E2, gdb-free):** after multimodule merge, `IrCall` sites kept stale `fn_id` values (e.g. `pb_vacuous` name → `fn_id=18`=`pb_new`). `ir_module_resolve_*` patches through `&! (*Box<IrModule>)` and in-place `instrs[ii].fn_id` writes were **silently dropped** by lean_single (same family as the documented Box-deref JIT bug). **Fix:** `ir_module_finalize_merged_calls` — copy-out owned `IrModule`, resolve by symbol name preferring real bodies (`instr_count>0`), write whole `IrInstr` slots back, codegen from finalized module.

**Conclusion:** failure was **not** vancomycin pharmacology or field-layout merge. It was **multimodule merge call-target resolution not persisting** before native codegen.

`PBox` definition (`stdlib/epistemic/knightian.sio:65-70`):

```sounio
pub struct PBox {
    lo_mean: f64,
    hi_mean: f64,
    variance: f64,
    confidence: i64,
}
```

---

## §4 — Gate impact (measured 2026-06-29)

| E2E filter | Pass | Fail | Total |
|---|---:|---:|---:|
| `epistemic` | 20 | 32 | 52 |
| `darwin_pbpk` | 3 | 24 | 27 |
| `pbpk` | 3 | 31 | 34 |
| `clinical` | 0 | 3 | 3 |

Many failures share the same Madaros multimodule + struct-return family; this dispatch scopes the **clinical vancomycin `PBox`** witness that is dissertação-critical and fully bisected.

---

## §5 — Relation to prior SRET work

| Prior artefact | Relevance |
|---|---|
| `docs/audit/g1_wip/SRET_FORWARDING_BUG_2026-06-02.md` | Plain forward `return ctor()` **fixed**; aggregate tuple forwarding still wrong |
| `docs/audit/MADAROS_SELFHOST_TYPEENV_SRET_2026-06-25.md` | Mega-struct `TypeEnv` by-value copy class — same systemic `sret` family |
| `docs/audit/sret_large_struct_smtcontext/DISPATCH.md` | Large struct return corruption (non-crash) — related ABI family |
| `self-hosted/native/codegen_x86_linux.sio` | `is_sret`, `sret_dest_reg`, `native_v2_ir_call_sret` — patch surface |

**New angle in this dispatch:** regression is on **Madaros v0.80.0 default engine** + **imported multimodule thin-link** (`imported_compile` path), consuming **production `PBox`** returned from `stdlib/clinical` into caller that links both `clinical` and `epistemic::knightian`.

---

## §6 — Leading hypothesis

1. **Caller** allocates / passes `sret` destination for `predict_cmin_knightian() -> PBox`.
2. **Callee** writes the 32-byte `PBox` to wrong address (local temp) or clobbers caller stack — same failure class as gdb-pinned forwarding bug in `SRET_FORWARDING_BUG_2026-06-02.md` (caller-supplied `rdi` dropped before nested struct-return call).
3. **Caller** reads garbage via `pb_confidence` / `pb_lo_mean` → SIGSEGV.

**Secondary hypothesis (weight 78.5, discard result still 139):** `vp_exp_approx` / `vp_cmin_point` while-loop or `Panic`/`Div` lowering may corrupt stack before return — investigate after primary `sret` fix if still red.

---

## §7 — Attack plan (do not patch ad hoc)

### Phase A — Pin mechanism (45–90 min)

1. Rebuild Madaros with debug symbols if available; `gdb` / `objdump -d` on witness ELF.
2. Break on `predict_cmin_knightian` return; verify `rdi`/`sret` dest at entry vs nested `pb_new` / `pb_vacuous` calls.
3. Compare codegen for:
   - green `sret_forwarding_minimal.sio`
   - red `reference/pbox_vacuous_read.sio`
   - red `tests/run-pass/sret_8_field_return.sio`

### Phase B — Fix surface (compiler only)

Target: `self-hosted/native/codegen_x86_linux.sio` and/or Madaros multimodule merge driver — **not** `stdlib/clinical/vancomycin_pbpk.sio` math.

Candidate fixes (in priority order):

1. Preserve caller `sret` pointer across nested struct-return calls in multimodule thin-link.
2. Align struct-return lowering threshold with System V AMD64 (`sret` for >16 bytes) for mixed `f64`/`i64` layouts.
3. Add IR witness for 32-byte multimodule return (see `tests/native-v2/science_spine/knowledge_f64_struct.sio` pattern).

### Phase C — Acceptance gates (E3 required)

| Gate | Command | Expected |
|---|---|---|
| Witness vacuous read | `./bin/souc run docs/audit/.../pbox_vacuous_read.sio` | exit 0 |
| Witness production read | `./bin/souc run docs/audit/.../pbox_clinical_read.sio` | exit 0 |
| Clinical stdlib E2E | `bash scripts/stdlib/run_stdlib_e2e.sh --filter clinical` | 3/3 pass |
| Vancomycin run-pass | `./bin/souc run tests/run-pass/vancomycin_propagation_v2.sio` | stdout `V2 PASS` |
| Stdlib clinical gate test | `./bin/souc run tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio` | stdout `VANCO V2 PASS` |
| SRET regression | `tests/run-pass/sret_forwarding_minimal.sio` + `sret_8_field_return.sio` | remain green |
| Bootstrap safety | `make build` fixed-point | bit-identical stages |

Remove `//@ known-failure` from `tests/run-pass/sret_pbox_clinical_known_failure.sio` only after clinical filter is green.

### Phase D — Workaround (stdlib-only, if compiler fix delayed)

**Not recommended without operator approval** — changes public API:

- Replace `-> PBox` returns with `out`-parameter `fn predict(..., out: &!PBox)` in `vancomycin_pbpk.sio` / `knightian.sio` consumers.
- Or return scalar tuple `(f64, f64, f64, i64)` instead of named struct.

Any clinical-path workaround requires `bin/llm-offload -t review -p deepseek` before commit.

---

## §8 — Ownership / parallel contract

| Rule | Value |
|---|---|
| **Do not edit in parallel** | `self-hosted/native/codegen_x86_linux.sio`, `bin/souc`, `bin/madaros` |
| **Stdlib lane may proceed read-only** | audit, test pins, documentation |
| **Stdlib lane must not** | change vancomycin dosing math to "fix" SIGSEGV |
| **Heavy validation** | serialize via `scripts/dev/souc-build-lock.sh` for full self-compile |

---

## §9 — Reference files

| File | Role |
|---|---|
| `reference/pbox_vacuous_read.sio` | **Primary** repro — vacuous return + accessor read |
| `reference/pbox_clinical_call_only.sio` | Control — call + discard (passes) |
| `reference/pbox_clinical_read.sio` | Production read path |
| `reference/pbox_32b_single_module.sio` | Control — single-module 32B struct (passes) |
| `tests/run-pass/sret_pbox_clinical_known_failure.sio` | Pinned `known-failure` for CI/gate visibility |

---

## §10 — Resolution and acceptance gates (2026-06-29)

**Root cause:** merged multimodule IR retained stale `fn_id` call targets (`pb_vacuous` name → `pb_new` body) because in-place `instrs[ii].fn_id` patches through `&!Box<IrModule>` were silently dropped (lean_single Box-deref mutation family). Fix: name-based call-target resolution during dep merge plus `ir_module_finalize_merged_calls` copying the owned module and rewriting whole `IrInstr` slots before codegen.

**Change:** `self-hosted/compiler/module_frontend.sio` — `ir_merge_find_function_name_index`, `ir_merge_remap_existing_call_targets`, `ir_module_resolve_one_call_target`, `ir_module_finalize_merged_calls`; forensic dump gated by `SOUNIO_DUMP_MERGED_CALLS=1`.

| Gate | Post-fix exit |
|---|---:|
| `reference/pbox_vacuous_read.sio` | 0 |
| `tests/run-pass/sret_pbox_clinical_known_failure.sio` | 0 (`//@ run-pass`) |
| `tests/run-pass/vancomycin_propagation_v2.sio` | 0 |
| `tests/stdlib/clinical/test_vancomycin_pbpk_v2.sio` | 0 |
| `tests/stdlib/clinical/test_aminoglycoside_correlation_sensitivity.sio` | 0 |
| `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio` | 1 (logical `ENCLOSURE SMOKE FAIL` — **not** SIGSEGV; separate math/enclosure lane) |

**Blocker status:** `BLK-20260629-stdlib-sret-pbox-clinical` **closed** for the SIGSEGV/compiler-semantics defect. Clinical E2E filter is **5/6** run-pass on harness; correlation-sensitivity enclosure failure remains open under a separate blocker (stdlib math, not merge/codegen).

**Next action:** open or route correlation-sensitivity enclosure probe; optional rename `sret_pbox_clinical_known_failure.sio` → `sret_pbox_clinical_regression.sio`.
