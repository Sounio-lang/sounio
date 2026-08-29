<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave15c-issue901-scale-closeout-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave15c-issue901-scale-closeout-2026-07-22
-->

# Wave15 Agent C — #901 large multi-module native scale closeout (2026-07-22)

**Issue:** #901 — native compile fails on large multi-module import graphs  
**Lane:** Wave15 residual wave (post into-acc #1402, specialized-list DCE #1397)  
**Isolation:** dedicated worktree on `origin/main`  
**Compiler:** stock `./bin/souc` → Madaros v0.80.0 (`bin/madaros-linux-x86_64`)

## Mission

Attack **#901** / large multi-module native path after into-acc. Goal: either
**close a measured large multi-mod science graph** under default Madaros, or ship
a hard residual gate with exact fail class and a **smaller corpus that is now
green post-into-acc**.

Do **not** regress dual, order_spread, cd_exact_e2e. Avoid Wave15 A
(parser/items GLOBAL_VAR_INIT) and B (print_f64).

## Historical fail class (filed)

From `docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md` / issue body:

```
use prob::distributions::*   # → special::gamma + igamma + erf
# type-check OK (verdict=0)
Merged IR: 210 functions
Native compilation failed: imported_simple_ir_emit_failed
module_native_driver: compact IR ELF write failed; rc=1
error: multimodule native thin-link compilation failed   # rc=12
```

Workaround was `SOUNIO_SOUC_ENGINE=lean_single` + `chmod +x`.

## Measured result (reproducible, Wave15C)

### Filed acceptance — **CLOSED** on default Madaros

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc compile tests/run-pass/madaros_native_multimodule_scale_prob.sio -o /tmp/p.elf
# → into_acc … final_fn_count ~73; Merged IR: 73; Compilation successful!
/tmp/p.elf
# → m=5.000000
```

| Program | Result | Merged IR (approx) |
|---|---|---:|
| #901 acceptance (`uniform_mean`) | compile+run `m=5.000000` | ~73 |
| Textbook science (`PROB_TEXTBOOK_OK`) | pdf/cdf/exp/unif/pois all pass | ~71 |
| `tests/stdlib/prob/test_prob_stdlib.sio` | `PROB_STDLIB_OK` under **Madaros** | ~71 |
| `examples/prob/distribution_report.sio` | textbook report lines under Madaros | ~72 |
| special gamma+erf+igamma multi | `g=24…` `erf=0.8427…` | ~36 |
| #921 rational+cd_sigma handoff | `11\n` (still green) | ~48 |
| dual gum+knowledge | `DUAL_GUM_KNOWLEDGE_OK` | ~94 |
| order_spread native gate | scaled `2044225` | (gate green) |
| cd_exact_generic_i64 | `ZD PROVED` | (gate green) |

Log markers on the acceptance probe (default):

- `lower_array: into_acc_done …`
- `module_native_driver: using full IR path; compact experimental path disabled by default` (when present)
- `imported_compile: lower_done` / `final_fn_count ~73`
- **No** `multimodule native thin-link compilation failed`
- **No** hard `Failed to write native binary rc=12`

### Mechanism of close (already on `main`, measured not introduced here)

1. **PR #1236** — default multi-module route uses full IR; compact experimental
   is opt-in only (`SOUNIO_ENABLE_COMPACT_IMPORTED_IR=1`).
2. **into-acc dep lower (#1402)** — dependency bodies fold into the accumulator
   (`into_acc_done`); reduces live multi-mod pressure vs by-value module copies.
3. **Specialized-list reachability DCE (#1397)** — specialized ItemFn pressure
   reduced (measured 50→19 on cd_exact specialized list in that lane).
4. Subsequent full-IR multimodule fixes (kind-9 parity, arena rebox, A14 call
   rebind, specialize collapse #1392, etc.).

Wave15C **measures** the close and ships a **hard gate** so the filed fail class
cannot regress silently. No speculative compiler patch was required on this tip.

## Smaller green corpus (post-into-acc)

| Corpus | Fail class before | Status now |
|---|---|---|
| `prob::distributions` trivial mean | thin-link / scale | **GREEN** |
| textbook multi-call science graph | thin-link / scale | **GREEN** |
| special gamma+erf+igamma together | scale risk | **GREEN** |
| #921 rational + cd_sigma | thin-link rc=12 | **GREEN** (Wave14D) |

## Hard residual (exact fail class — not #901 scale)

| Program | compile under Madaros | Fail class |
|---|---|---|
| `tests/stdlib/stats/test_ols_diag_e2e.sio` | **RED** | **`E019` method calls** (`method calls are not supported for this type`) → visibility preflight — **not** thin-link / SEGV / OOM |
| `tests/stdlib/stats/test_scipy_e2e_vertical.sio` | **RED** | **AST closure / parse** incomplete (not thin-link) |
| Compact imported-simple-IR (opt-in) | fallback | `compact_emit_failed` → full IR succeeds (Wave14D) |

These residuals are **orthogonal** to the filed #901 scale/thin-link class. The
gate records the OLS residual class so a future scale-class regression on that
surface is detectable, without claiming OLS is green.

## Gate

```bash
bash scripts/madaros_native_multimodule_scale_901_gate.sh
# → MADAROS_NATIVE_MULTIMODULE_SCALE_901_GATE_OK
# receipt: artifacts/compiler/madaros_native_multimodule_scale_901_receipt.v1.json
```

## Non-regression (measured same tip)

```bash
bash scripts/madaros_dual_import_gate.sh              # MADAROS_DUAL_IMPORT_GATE_OK
bash scripts/madaros_order_spread_native_gate.sh      # MADAROS_ORDER_SPREAD_NATIVE_GATE_OK
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh # MADAROS_CD_EXACT_GENERIC_I64_GATE_OK
bash scripts/madaros_thinlink_921_residual_gate.sh    # MADAROS_THINLINK_921_RESIDUAL_GATE_OK
```

## What this does **not** claim

- All multi-module stdlib verticals (stats OLS, scipy e2e still red for other reasons).
- Compact imported-simple-IR is production-ready (opt-in residual remains).
- Exclusive-ref / memory-wall fragile chains beyond the measured corpus.
- Layout-catalog / nested-field #901 sublane (`codex/issue901-*`) — separate acceptance.

## Acceptance vs #901 body

| Acceptance item from issue | Status |
|---|---|
| `souc compile` (default Madaros) of `prob::distributions` probe produces runnable ELF | **PASS** |
| Program runs with correct textbook-scale values | **PASS** |
| Fail class no longer thin-link rc=12 / compact-only | **PASS** |
| lean_single still works as escape hatch | **unchanged** |

## Next action

1. Land gate + audit + stale-header cleanup (this PR).
2. Close or re-label GitHub #901 with evidence pointing at this audit + gate
   (filed multi-module **scale** acceptance). Leave layout-catalog #901 sublane
   open under its own PRs if still red.
3. Track OLS `E019` / scipy parse as separate D3 residuals, not #901 scale.
