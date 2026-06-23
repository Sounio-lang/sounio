#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_ROOT="${SOUNIO_EDITOR_LOCAL_GATE_ARTIFACT_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-editor-local-gate.XXXXXX")}"
LOG_DIR="$ARTIFACT_ROOT/logs"
SUMMARY="$ARTIFACT_ROOT/summary.v1.tsv"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"
KNOWN_LIMITS="$ROOT_DIR/docs/compiler/KNOWN_LIMITATIONS.md"

mkdir -p "$LOG_DIR"

run_capture() {
  local name="$1"
  shift
  local stdout="$LOG_DIR/$name.stdout"
  local stderr="$LOG_DIR/$name.stderr"
  set +e
  "$@" >"$stdout" 2>"$stderr"
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "${stdout#$ARTIFACT_ROOT/}" "${stderr#$ARTIFACT_ROOT/}" >>"$SUMMARY"
  return "$rc"
}

require_contains() {
  local path="$1"
  local needle="$2"
  local label="$3"
  if ! grep -Fq "$needle" "$path"; then
    echo "editor local gate failed: missing $label: $needle" >&2
    echo "  in: $path" >&2
    exit 1
  fi
}

printf 'step\texit\tstdout_log\tstderr_log\n' >"$SUMMARY"

if [[ ! -x "$ROOT_DIR/bin/souc" ]]; then
  echo "editor local gate failed: bin/souc is missing or not executable" >&2
  exit 2
fi

if ! run_capture souc-help "$ROOT_DIR/bin/souc" --help; then
  echo "editor local gate failed: bin/souc --help failed" >&2
  exit 1
fi
require_contains "$LOG_DIR/souc-help.stdout" "souc format|fmt <file.sio>" "formatter help surface"
require_contains "$LOG_DIR/souc-help.stdout" "souc repl" "REPL help surface"

if ! run_capture formatter-idempotent bash "$ROOT_DIR/scripts/gates/g5a_formatter_idempotent.sh"; then
  echo "editor local gate failed: formatter idempotency gate failed" >&2
  exit 1
fi
if ! run_capture repl-eval bash "$ROOT_DIR/scripts/gates/g5b_repl_eval.sh"; then
  echo "editor local gate failed: REPL eval gate failed" >&2
  exit 1
fi

require_contains "$LOG_DIR/formatter-idempotent.stdout" "G5a idempotency PASS" "formatter idempotency pass"
require_contains "$LOG_DIR/repl-eval.stdout" "G5b REPL eval gate: PASS" "REPL eval pass"

require_contains "$KNOWN_LIMITS" "Formatter — Phase 1" "known-limits formatter scope"
require_contains "$KNOWN_LIMITS" "No AST round-trip guarantee" "known-limits formatter boundary"
require_contains "$KNOWN_LIMITS" "REPL" "known-limits REPL scope"
require_contains "$KNOWN_LIMITS" "fully-Sounio eval loop in \`self-hosted/repl/\` deferred" "known-limits REPL boundary"
require_contains "$KNOWN_LIMITS" "registry row \`tooling.editor\` says **prototype**" "known-limits LSP downgrade"

cat >"$RESULTS" <<EOF
# Sounio Editor Local Gate

| Field | Value |
|---|---|
| artifact_root | \`$ARTIFACT_ROOT\` |
| status | \`pass\` |

This gate validates the release-supported local editor/tooling scope:

- checked \`bin/souc\` exists and advertises \`format/fmt\` plus \`repl\`
- \`scripts/gates/g5a_formatter_idempotent.sh\` passes
- \`scripts/gates/g5b_repl_eval.sh\` passes
- \`docs/compiler/KNOWN_LIMITATIONS.md\` carries the formatter, REPL, and LSP scope boundaries
- \`scripts/ci/serious_language_claim_closure_gate.sh\` is responsible for checking
  the public claim-registry row that points at this gate

This gate does not validate IDE integration polish, a production LSP server, or
AST-aware formatting.
EOF

echo "Editor local gate passed. See $RESULTS"
