#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WINDOW_FILE=".claude/check_sio_integration_window.v1.json"

if [[ ! -f "$WINDOW_FILE" ]]; then
  echo "error: missing check.sio integration window file: $WINDOW_FILE" >&2
  exit 1
fi

jq -e '
  .schema == "sounio.check_sio.integration_window.v1"
  and (.status == "active" or .status == "idle" or .status == "closed")
  and (.window.target_file == "self-hosted/check/check.sio")
  and ((.window.depends_on // []) | length >= 3)
  and ((.queue // []) | type == "array")
' "$WINDOW_FILE" >/dev/null

if jq -e '.status == "active"' "$WINDOW_FILE" >/dev/null; then
  owner="$(jq -r '.window.owner_prompt // ""' "$WINDOW_FILE")"
  case "$owner" in
    codex_epistemic_expansion.md|codex_borrow_integration.md|kimi_diagnostic_hardening.md|minimax_error_messages.md)
      ;;
    *)
      echo "error: unsupported active owner_prompt for serialized check.sio window: $owner" >&2
      exit 1
      ;;
  esac
fi

if jq -e '.status == "closed"' "$WINDOW_FILE" >/dev/null; then
  if ! jq -e '.window.closed_at_utc and .window.closed_because' "$WINDOW_FILE" >/dev/null; then
    echo "error: closed check.sio window must have window.closed_at_utc and window.closed_because" >&2
    exit 1
  fi
fi

echo "CHECK_SIO_INTEGRATION_WINDOW_PASS"
