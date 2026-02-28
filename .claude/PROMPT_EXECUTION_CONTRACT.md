# Prompt Execution Contract (Codex <-> Claude)

Status: active

This contract defines how prompt work is dispatched, merged, and validated without drift.

## Prompt Matrix (authoritative)

| Prompt | Owner model | Phase | Depends on | Target files | Required gates | Evidence artifacts |
|---|---|---|---|---|---|---|
| `codex_epistemic_expansion.md` | Codex | A.1 | none | `self-hosted/check/epistemic.sio`, `self-hosted/check/check.sio` | checker target tests | cutover logs + test output |
| `codex_borrow_integration.md` | Codex | A.2 | A.1 preferred | `self-hosted/check/borrow.sio`, `self-hosted/check/check.sio` | checker target tests | cutover logs + test output |
| `kimi_diagnostic_hardening.md` | Kimi 2.5 | A.3 | A.1, A.2 | `self-hosted/check/check.sio`, `tests/ui/type/*` | ignored-test triage loop | ignore triage report |
| `codex_closure_effects.md` | Codex | B.1 | Phase A stable | parser + checker closure files | parser/checker tests | closure-focused logs |
| `glm_package_manager.md` | GLM-5 | B.2 | none (separate lane) | `tools/pkg/*` | package-manager script checks | package-manager logs |
| `minimax_error_messages.md` | MiniMax | B.3 | B.1 stable | `self-hosted/check/check.sio` | diagnostics snapshot tests | message delta report |
| `glm_test_deignore.md` | GLM-5 | C.1 | A+B stable | `tests/*` ignores + expectations | de-ignore batches | ready/needs-fix/blocked report |
| `deepseek_epistemic_algebra.md` | DeepSeek | C.2 | A stable | math review docs | review signoff | algebra review report |
| `kimi_lsp_server.md` | Kimi 2.5 | LSP | independent, but no-rust strict | `tools/lsp/*`, `scripts/lsp_smoke_gate.sh` | `LSP_SMOKE_PASS` | `artifacts/omega/lsp_smoke.log` |

## Merge Policy (mandatory sequence per lane)

1. feature-complete
2. gate-complete
3. artifact refresh

Do not split these three stages across unrelated changes in the same lane.

## Conflict Protocol

1. `self-hosted/check/check.sio` call-path edits are serialized.
2. If two prompts need `check.sio`, integration windows are opened in order: A.1 -> A.2 -> A.3.
3. Each integration window must end with a targeted test run and concise evidence summary.

## Active Serialized `check.sio` Window

Machine-checkable window state:
- `.claude/check_sio_integration_window.v1.json`

Verification command:
- `bash scripts/check_check_sio_integration_window.sh`

## Done Criteria Per Prompt

1. Code changes land in declared target files only (or documented exceptions).
2. Required gates pass.
3. Evidence artifacts/log references are recorded.
4. No canonical-order drift against `PLAN_ORIGINAL.md` and `.claude/offload-specs`.
