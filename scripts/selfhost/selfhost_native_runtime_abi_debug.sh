#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_PATH="${MANIFEST_PATH:-tests/selfhost/native_runtime/abi_manifest.tsv}"
CHECKER_BIN="${CHECKER_BIN:-./bin/souc}"
BOOT4_BIN="${BOOT4_BIN:-artifacts/bootstrap/boot4.elf}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-selfhost-native-runtime-abi}"
TIMEOUT_SECS="${TIMEOUT_SECS:-20}"

mkdir -p "$WORK_DIR"

RESULTS_FILE="$WORK_DIR/results.tsv"
cat >"$RESULTS_FILE" <<'EOF'
case_id	check_rc	compile_rc	run_rc	expected_exit	program
EOF

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}" "$@"
    return $?
  fi

  "$@"
}

while IFS=$'\t' read -r case_id program expected_exit; do
  if [ -z "${case_id:-}" ]; then
    continue
  fi
  if [[ "$case_id" == \#* ]]; then
    continue
  fi

  elf_path="$WORK_DIR/${case_id}.elf"
  check_rc=0
  compile_rc=0
  run_rc=-

  set +e
  "$CHECKER_BIN" check "$program" >"$WORK_DIR/${case_id}.check.stdout" 2>"$WORK_DIR/${case_id}.check.stderr"
  check_rc=$?

  "$BOOT4_BIN" "$program" "$elf_path" >"$WORK_DIR/${case_id}.compile.stdout" 2>"$WORK_DIR/${case_id}.compile.stderr"
  compile_rc=$?

  if [ "$compile_rc" -eq 0 ]; then
    chmod +x "$elf_path" 2>/dev/null || true
    run_with_timeout "${TIMEOUT_SECS}s" "$elf_path" >"$WORK_DIR/${case_id}.run.stdout" 2>"$WORK_DIR/${case_id}.run.stderr"
    run_rc=$?
  fi
  set -e

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$check_rc" "$compile_rc" "$run_rc" "$expected_exit" "$program" >>"$RESULTS_FILE"
done <"$MANIFEST_PATH"

cat "$RESULTS_FILE"
