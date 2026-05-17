#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_PATH="${LSP_SMOKE_LOG_PATH:-$ROOT_DIR/artifacts/omega/lsp_smoke.log}"
STATUS_PATH="${LSP_SMOKE_STATUS_PATH:-$ROOT_DIR/artifacts/omega/lsp_smoke_status.v1.json}"

mkdir -p "$(dirname "$LOG_PATH")"
: >"$LOG_PATH"
mkdir -p "$(dirname "$STATUS_PATH")"

resolve_souc() {
  if [ -n "${SOUC_BIN:-}" ] && [ -x "${SOUC_BIN:-}" ]; then
    printf '%s\n' "$SOUC_BIN"
    return 0
  fi
  if [ -x "$ROOT_DIR/.pinned-souc/souc-linux-x86_64" ]; then
    printf '%s\n' "$ROOT_DIR/.pinned-souc/souc-linux-x86_64"
    return 0
  fi
  if [ -x "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64" ]; then
    printf '%s\n' "$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64"
    return 0
  fi
  OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}" \
  OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
    bash "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" --print-path
}

SOUC_RESOLVED="$(resolve_souc)"
if [ -z "$SOUC_RESOLVED" ] || [ ! -x "$SOUC_RESOLVED" ]; then
  echo "error: unable to resolve executable SOUC_BIN for LSP smoke gate" >&2
  exit 1
fi

export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_LSP_STRICT_NO_RUST="${SOUNIO_LSP_STRICT_NO_RUST:-1}"
export SOUNIO_LSP_SOUC_BIN="$SOUC_RESOLVED"

echo "LSP_SMOKE_GATE_START souc=$SOUC_RESOLVED strict_no_rust=$SOUNIO_LSP_STRICT_NO_RUST" | tee -a "$LOG_PATH"
status="pass"

# Pure-Sounio LSP integration smoke (preferred). Exercises the 8 LSP
# methods end-to-end against `souc lsp`. Replaces the legacy bash+jq+
# python3 hybrid LSP smoke that lived under test_smoke.sh.
if [[ -x "$ROOT_DIR/tools/lsp/test_protocol.sh" ]]; then
  if ! bash "$ROOT_DIR/tools/lsp/test_protocol.sh" 2>&1 | tee -a "$LOG_PATH"; then
    status="fail"
  fi
elif [[ -x "$ROOT_DIR/tools/lsp/test_smoke.sh" ]]; then
  # Legacy hybrid LSP smoke (deprecated, kept only if the new pure-
  # Sounio harness is absent).
  if ! bash "$ROOT_DIR/tools/lsp/test_smoke.sh" 2>&1 | tee -a "$LOG_PATH"; then
    status="fail"
  fi
else
  echo "[lsp-smoke] error: no LSP smoke harness present" | tee -a "$LOG_PATH"
  status="fail"
fi
failure_hint=""
if [[ "$status" != "pass" ]]; then
  failure_hint="$(grep -E '\[lsp-smoke\]\[FAIL\]|Traceback|SystemExit|error:' "$LOG_PATH" | tail -n 1 || true)"
fi

if [[ "$status" == "pass" ]]; then
  echo "LSP_SMOKE_PASS" | tee -a "$LOG_PATH"
fi

jq -cn \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg status "$status" \
  --arg log_path "${LOG_PATH#$ROOT_DIR/}" \
  --arg souc "$SOUC_RESOLVED" \
  --arg failure_hint "$failure_hint" \
  --argjson strict_no_rust "$SOUNIO_LSP_STRICT_NO_RUST" \
  '{
    schema: "sounio.lsp.smoke.status.v1",
    generated_at_utc: $generated_at_utc,
    status: $status,
    log_path: $log_path,
    souc_bin: $souc,
    strict_no_rust: ($strict_no_rust == 1),
    pass_marker: ($status == "pass"),
    last_failure_hint: (if $status == "pass" then "" else $failure_hint end)
  }' >"$STATUS_PATH"

if [[ "$status" != "pass" ]]; then
  echo "error: LSP smoke gate failed (status file: $STATUS_PATH)" >&2
  exit 1
fi
