#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${SOUC_NATIVE_BIN:-}" && -x "${SOUC_NATIVE_BIN:-}" ]]; then
  export SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/scripts/ci/souc-native-wrapper.sh}"
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

DRIVER_SRC="${SOUNIO_NATIVE_V2_EPISTEMIC_DRIVER:-self-hosted/compiler/native_compile_driver.sio}"
MANIFEST_PATH="${SOUNIO_NATIVE_V2_EPISTEMIC_MANIFEST:-tests/native-v2/epistemic_runtime/manifest.tsv}"
WORK_DIR="${SOUNIO_NATIVE_V2_EPISTEMIC_DIR:-$(mktemp -d /tmp/sounio-native-v2-epistemic-runtime.XXXXXX)}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
TIMEOUT_SECS="${SOUNIO_NATIVE_V2_EPISTEMIC_TIMEOUT:-180}"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

echo "[native-v2-epistemic-runtime] souc=$SOUC_BIN"
if [[ -n "${SOUC_NATIVE_BIN:-}" ]]; then
  echo "[native-v2-epistemic-runtime] native_backend=$SOUC_NATIVE_BIN"
fi
echo "[native-v2-epistemic-runtime] driver=$DRIVER_SRC"
echo "[native-v2-epistemic-runtime] manifest=$MANIFEST_PATH"
echo "[native-v2-epistemic-runtime] out=$WORK_DIR"

if [[ ! -f "$DRIVER_SRC" ]]; then
  echo "[native-v2-epistemic-runtime] FAIL: missing driver $DRIVER_SRC" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[native-v2-epistemic-runtime] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi

run_with_timeout() {
  local seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
    return $?
  fi

  "$@" &
  local pid=$!
  local elapsed=0
  while kill -0 "$pid" >/dev/null 2>&1; do
    if [[ "$elapsed" -ge "$seconds" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "$pid"
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

assert_elf_x86_64() {
  local path="$1"
  local label="$2"
  local file_log="$LOG_DIR/${label}.file.txt"

  if command -v file >/dev/null 2>&1; then
    file "$path" >"$file_log"
    if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$file_log"; then
      echo "[native-v2-epistemic-runtime] FAIL: unexpected file kind for $path" >&2
      cat "$file_log" >&2 || true
      exit 1
    fi
  fi
}

cat >"$RESULTS_TSV" <<'EOF'
case_id	program	claim_class	expected_exit	run_exit	stdout	status	sha256	bytes
EOF

case_count=0
pass_count=0

while IFS=$'\t' read -r case_id program_path expected_exit expected_stdout claim_class; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  case_count=$((case_count + 1))
  if [[ ! -f "$program_path" ]]; then
    echo "[native-v2-epistemic-runtime] FAIL: missing program for $case_id: $program_path" >&2
    exit 1
  fi
  if [[ ! -f "$expected_stdout" ]]; then
    echo "[native-v2-epistemic-runtime] FAIL: missing stdout fixture for $case_id: $expected_stdout" >&2
    exit 1
  fi

  elf_path="$ARTIFACT_DIR/$case_id"
  compile_log="$LOG_DIR/$case_id.compile.log"
  run_stdout="$LOG_DIR/$case_id.stdout"
  run_stderr="$LOG_DIR/$case_id.stderr"

  if ! run_with_timeout "$TIMEOUT_SECS" "$SOUC_BIN" run "$DRIVER_SRC" -- "$program_path" -o "$elf_path" >"$compile_log" 2>&1; then
    echo "[native-v2-epistemic-runtime] FAIL: driver compile failed for $case_id" >&2
    tail -n 120 "$compile_log" >&2 || true
    printf '%s\t%s\t%s\t%s\t-\t-\tcompile_fail\t-\t-\n' \
      "$case_id" "$program_path" "$claim_class" "$expected_exit" >>"$RESULTS_TSV"
    exit 1
  fi
  chmod +x "$elf_path" 2>/dev/null || true
  assert_elf_x86_64 "$elf_path" "$case_id"

  set +e
  run_with_timeout "$TIMEOUT_SECS" "$elf_path" >"$run_stdout" 2>"$run_stderr"
  run_exit=$?
  set -e

  status="ok"
  stdout_status="ok"
  if [[ "$run_exit" != "$expected_exit" ]]; then
    status="exit_mismatch"
  elif ! cmp -s "$expected_stdout" "$run_stdout"; then
    status="stdout_mismatch"
    stdout_status="mismatch"
  fi

  sha="$(portable_sha256 "$elf_path")"
  bytes="$(portable_size "$elf_path")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program_path" "$claim_class" "$expected_exit" "$run_exit" \
    "$stdout_status" "$status" "$sha" "$bytes" >>"$RESULTS_TSV"

  if [[ "$status" != "ok" ]]; then
    echo "[native-v2-epistemic-runtime] FAIL: $case_id status=$status" >&2
    if [[ "$status" == "stdout_mismatch" ]]; then
      diff -u "$expected_stdout" "$run_stdout" >&2 || true
    fi
    if [[ -s "$run_stderr" ]]; then
      tail -n 80 "$run_stderr" >&2 || true
    fi
    exit 1
  fi

  pass_count=$((pass_count + 1))
  echo "[native-v2-epistemic-runtime] ok: $case_id ($claim_class)"
done <"$MANIFEST_PATH"

if [[ "$case_count" -eq 0 ]]; then
  echo "[native-v2-epistemic-runtime] FAIL: manifest had no cases" >&2
  exit 1
fi

echo "[native-v2-epistemic-runtime] PASS: pass=$pass_count total=$case_count results=$RESULTS_TSV"
