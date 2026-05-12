#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${SOUNIO_SERIOUS_CONFORMANCE_MANIFEST:-$ROOT_DIR/tests/conformance/manifest.v1.tsv}"
SOUC_BIN="${SOUNIO_SERIOUS_CONFORMANCE_SOUC_BIN:-$ROOT_DIR/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_SERIOUS_CONFORMANCE_STDLIB_PATH:-$ROOT_DIR/stdlib}"

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
ARTIFACT_ROOT="${SOUNIO_SERIOUS_CONFORMANCE_ARTIFACT_ROOT:-/tmp/sounio-serious-conformance-$timestamp}"
LOG_DIR="$ARTIFACT_ROOT/logs"
RESULTS="$ARTIFACT_ROOT/RESULTS.md"
SUMMARY_TSV="$ARTIFACT_ROOT/summary.v1.tsv"
SUMMARY_JSON="$ARTIFACT_ROOT/summary.v1.json"

mkdir -p "$LOG_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Conformance manifest not found: $MANIFEST" >&2
  exit 2
fi

if [[ ! -x "$SOUC_BIN" ]]; then
  echo "souc binary is not executable: $SOUC_BIN" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to write summary.v1.json" >&2
  exit 2
fi

expected_header=$'id\tmode\tsource\texpected_exit\tstdout_contains\tstderr_contains\tclaim_id'
actual_header="$(head -n 1 "$MANIFEST")"
actual_header="${actual_header%$'\r'}"
if [[ "$actual_header" != "$expected_header" ]]; then
  echo "Conformance manifest header mismatch." >&2
  echo "expected: $expected_header" >&2
  echo "actual:   $actual_header" >&2
  exit 2
fi

manifest_rel="${MANIFEST#$ROOT_DIR/}"
branch="$(git branch --show-current 2>/dev/null || true)"
commit="$(git rev-parse HEAD 2>/dev/null || true)"
souc_version="$("$SOUC_BIN" --version 2>/dev/null || true)"
souc_info="$("$SOUC_BIN" info 2>/dev/null || true)"

cat >"$RESULTS" <<EOF
# Sounio Serious-Language Conformance Gate

Generated at: $timestamp

| Field | Value |
|---|---|
| branch | \`$branch\` |
| commit | \`$commit\` |
| manifest | \`$manifest_rel\` |
| artifact_root | \`$ARTIFACT_ROOT\` |
| souc_bin | \`$SOUC_BIN\` |
| stdlib_path | \`$SOUNIO_STDLIB_PATH\` |
| souc_version | \`$souc_version\` |

This gate is a bounded conformance spine. It checks selected stable and validated-research surfaces that are safe to cite from the serious-language readiness ledger. It is not a complete language specification suite.

## Cases

| Case | Mode | Claim | Expected | Actual | Status |
|---|---|---|---|---|---|
EOF

printf 'id\tmode\tsource\texpected_exit\tactual_exit\tstatus\treason\tclaim_id\tstdout_log\tstderr_log\n' >"$SUMMARY_TSV"

total=0
passed=0
failed=0

append_summary_row() {
  local id="$1"
  local mode="$2"
  local source="$3"
  local expected_exit="$4"
  local actual_exit="$5"
  local status="$6"
  local reason="$7"
  local claim_id="$8"
  local stdout_log="$9"
  local stderr_log="${10}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" "$mode" "$source" "$expected_exit" "$actual_exit" "$status" "$reason" "$claim_id" "$stdout_log" "$stderr_log" >>"$SUMMARY_TSV"
  printf '| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
    "$id" "$mode" "$claim_id" "$expected_exit" "$actual_exit" "$status" >>"$RESULTS"
}

run_case() {
  local id="$1"
  local mode="$2"
  local source="$3"
  local expected_exit="$4"
  local stdout_contains="$5"
  local stderr_contains="$6"
  local claim_id="$7"

  local source_path="$ROOT_DIR/$source"
  local stdout_log="$LOG_DIR/$id.stdout"
  local stderr_log="$LOG_DIR/$id.stderr"
  local command_log="$LOG_DIR/$id.command"
  local stdout_rel="${stdout_log#$ARTIFACT_ROOT/}"
  local stderr_rel="${stderr_log#$ARTIFACT_ROOT/}"
  local actual_exit="not_run"
  local status="pass"
  local reason="ok"
  local -a command

  if [[ ! "$id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    status="fail"
    reason="invalid case id"
    append_summary_row "$id" "$mode" "$source" "$expected_exit" "$actual_exit" "$status" "$reason" "$claim_id" "$stdout_rel" "$stderr_rel"
    return
  fi

  if [[ "$source" = /* || "$source" == *".."* || ! -f "$source_path" ]]; then
    status="fail"
    reason="source not found or unsafe"
    append_summary_row "$id" "$mode" "$source" "$expected_exit" "$actual_exit" "$status" "$reason" "$claim_id" "$stdout_rel" "$stderr_rel"
    return
  fi

  case "$mode" in
    run)
      command=("$SOUC_BIN" run "$source")
      ;;
    check)
      command=("$SOUC_BIN" check "$source")
      ;;
    check-fail|compile-fail)
      command=("$SOUC_BIN" check "$source")
      ;;
    *)
      status="fail"
      reason="unknown mode"
      append_summary_row "$id" "$mode" "$source" "$expected_exit" "$actual_exit" "$status" "$reason" "$claim_id" "$stdout_rel" "$stderr_rel"
      return
      ;;
  esac

  printf '%q ' "${command[@]}" >"$command_log"
  printf '\n' >>"$command_log"

  set +e
  "${command[@]}" >"$stdout_log" 2>"$stderr_log"
  actual_exit=$?
  set -e

  local exit_ok=0
  case "$expected_exit" in
    0|zero)
      [[ "$actual_exit" -eq 0 ]] && exit_ok=1
      ;;
    nonzero)
      [[ "$actual_exit" -ne 0 ]] && exit_ok=1
      ;;
    *)
      if [[ "$expected_exit" =~ ^[0-9]+$ && "$actual_exit" -eq "$expected_exit" ]]; then
        exit_ok=1
      fi
      ;;
  esac

  if [[ "$exit_ok" -ne 1 ]]; then
    status="fail"
    reason="exit mismatch"
  fi

  if [[ "$stdout_contains" != "-" && "$stdout_contains" != "" ]]; then
    if ! grep -Fq -- "$stdout_contains" "$stdout_log"; then
      status="fail"
      if [[ "$reason" == "ok" ]]; then
        reason="stdout missing expected text"
      else
        reason="$reason; stdout missing expected text"
      fi
    fi
  fi

  if [[ "$stderr_contains" != "-" && "$stderr_contains" != "" ]]; then
    if ! grep -Fq -- "$stderr_contains" "$stderr_log"; then
      status="fail"
      if [[ "$reason" == "ok" ]]; then
        reason="stderr missing expected text"
      else
        reason="$reason; stderr missing expected text"
      fi
    fi
  fi

  append_summary_row "$id" "$mode" "$source" "$expected_exit" "$actual_exit" "$status" "$reason" "$claim_id" "$stdout_rel" "$stderr_rel"
}

while IFS=$'\t' read -r id mode source expected_exit stdout_contains stderr_contains claim_id || [[ -n "${id:-}" ]]; do
  [[ -z "${id:-}" || "$id" =~ ^# ]] && continue
  [[ "$id" == "id" ]] && continue

  total=$((total + 1))
  before_failures="$failed"
  run_case "$id" "$mode" "$source" "$expected_exit" "$stdout_contains" "$stderr_contains" "$claim_id"
  if tail -n 1 "$SUMMARY_TSV" | awk -F '\t' '{exit ($6 == "pass") ? 0 : 1}'; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
  if [[ "$failed" != "$before_failures" ]]; then
    echo "[conformance] FAIL $id"
  else
    echo "[conformance] PASS $id"
  fi
done <"$MANIFEST"

cat >>"$RESULTS" <<EOF

## Summary

| Total | Passed | Failed |
|---|---|---|
| \`$total\` | \`$passed\` | \`$failed\` |

Machine-readable summaries:

- \`summary.v1.tsv\`
- \`summary.v1.json\`

Raw per-case stdout and stderr logs are under \`logs/\`.

EOF

TOTAL="$total" PASSED="$passed" FAILED="$failed" \
  python3 - "$SUMMARY_TSV" "$SUMMARY_JSON" "$timestamp" "$branch" "$commit" "$SOUC_BIN" "$SOUNIO_STDLIB_PATH" "$manifest_rel" "$ARTIFACT_ROOT" <<'PY'
import csv
import json
import os
import sys

summary_tsv, summary_json, generated_at, branch, commit, souc_bin, stdlib_path, manifest, artifact_root = sys.argv[1:]

with open(summary_tsv, newline="", encoding="utf-8") as handle:
    cases = list(csv.DictReader(handle, delimiter="\t"))

payload = {
    "schema": "sounio.serious_language_conformance.v1",
    "generated_at": generated_at,
    "branch": branch,
    "commit": commit,
    "manifest": manifest,
    "artifact_root": artifact_root,
    "souc_bin": souc_bin,
    "stdlib_path": stdlib_path,
    "total": int(os.environ["TOTAL"]),
    "passed": int(os.environ["PASSED"]),
    "failed": int(os.environ["FAILED"]),
    "cases": cases,
}

with open(summary_json, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "Wrote serious-language conformance artifacts to $ARTIFACT_ROOT"
echo "Summary: total=$total passed=$passed failed=$failed"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
