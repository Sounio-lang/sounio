#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-source-elf] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-source-elf] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

OUT_DIR="${SOUNIO_MADAROS_SOURCE_TO_ELF_DIR:-$(mktemp -d /tmp/sounio-madaros-source-elf.XXXXXX)}"
MANIFEST_PATH="${SOUNIO_MADAROS_SOURCE_TO_ELF_MANIFEST:-tests/madaros/source_to_elf/manifest.tsv}"
LOG_DIR="$OUT_DIR/logs"
BIN_DIR="$OUT_DIR/bin"
RESULTS_TSV="$OUT_DIR/results.tsv"

mkdir -p "$LOG_DIR" "$BIN_DIR"

echo "[madaros-source-elf] START: source-to-ELF semantic gate"
echo "[madaros-source-elf] madaros=$MADAROS_BIN"
echo "[madaros-source-elf] out=$OUT_DIR"
echo "[madaros-source-elf] manifest=$MANIFEST_PATH"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[madaros-source-elf] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

assert_elf_x86_64() {
  local path="$1"
  local label="$2"
  local file_log="$LOG_DIR/${label}.file.txt"

  if command -v file >/dev/null 2>&1; then
    file "$path" >"$file_log"
    if ! grep -q 'ELF 64-bit LSB executable, x86-64' "$file_log"; then
      echo "[madaros-source-elf] FAIL: unexpected file kind for $path" >&2
      cat "$file_log" >&2
      exit 1
    fi
  fi
}

printf 'case_id\tmode\tprogram\texpected_exit\tactual_exit\tclaim_class\tsha256\tstatus\n' >"$RESULTS_TSV"

while IFS=$'\t' read -r case_id program_path expected_exit claim_class; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  if [[ ! -f "$program_path" ]]; then
    echo "[madaros-source-elf] FAIL: missing program for $case_id: $program_path" >&2
    exit 1
  fi

  check_log="$LOG_DIR/$case_id.check.log"
  trace_log="$LOG_DIR/$case_id.trace.log"
  normal_compile_log="$LOG_DIR/$case_id.normal.compile.log"
  normal_stdout_log="$LOG_DIR/$case_id.normal.stdout"
  normal_stderr_log="$LOG_DIR/$case_id.normal.stderr"
  normal_bin="$BIN_DIR/$case_id.normal"
  nv2_compile_log="$LOG_DIR/$case_id.native_v2.compile.log"
  nv2_stdout_log="$LOG_DIR/$case_id.native_v2.stdout"
  nv2_stderr_log="$LOG_DIR/$case_id.native_v2.stderr"
  nv2_bin="$BIN_DIR/$case_id.native_v2"

  if ! "$MADAROS_BIN" --check "$program_path" >"$check_log" 2>&1; then
    echo "[madaros-source-elf] FAIL: check failed for $case_id" >&2
    tail -n 80 "$check_log" >&2 || true
    exit 1
  fi

  if ! "$MADAROS_BIN" --probe-load-ir-trace "$program_path" >"$trace_log" 2>&1; then
    echo "[madaros-source-elf] FAIL: IR trace failed for $case_id" >&2
    tail -n 80 "$trace_log" >&2 || true
    exit 1
  fi

  if ! "$MADAROS_BIN" "$program_path" -o "$normal_bin" >"$normal_compile_log" 2>&1; then
    echo "[madaros-source-elf] FAIL: normal compile failed for $case_id" >&2
    tail -n 80 "$normal_compile_log" >&2 || true
    exit 1
  fi

  chmod +x "$normal_bin" 2>/dev/null || true
  assert_elf_x86_64 "$normal_bin" "$case_id.normal"

  set +e
  "$normal_bin" >"$normal_stdout_log" 2>"$normal_stderr_log"
  normal_exit=$?
  set -e

  normal_status="ok"
  if [[ "$normal_exit" != "$expected_exit" ]]; then
    normal_status="exit_mismatch"
  fi

  normal_sha="$(portable_sha256 "$normal_bin")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "normal" "$program_path" "$expected_exit" "$normal_exit" "$claim_class" "$normal_sha" "$normal_status" \
    >>"$RESULTS_TSV"

  if [[ "$normal_status" != "ok" ]]; then
    echo "[madaros-source-elf] FAIL: $case_id normal expected_exit=$expected_exit actual_exit=$normal_exit" >&2
    tail -n 80 "$normal_compile_log" >&2 || true
    exit 1
  fi

  if ! "$MADAROS_BIN" --native-v2-compile "$program_path" -o "$nv2_bin" >"$nv2_compile_log" 2>&1; then
    echo "[madaros-source-elf] FAIL: native-v2 compile failed for $case_id" >&2
    tail -n 80 "$nv2_compile_log" >&2 || true
    exit 1
  fi

  chmod +x "$nv2_bin" 2>/dev/null || true
  assert_elf_x86_64 "$nv2_bin" "$case_id.native_v2"

  set +e
  "$nv2_bin" >"$nv2_stdout_log" 2>"$nv2_stderr_log"
  nv2_exit=$?
  set -e

  nv2_status="ok"
  if [[ "$nv2_exit" != "$expected_exit" ]]; then
    nv2_status="exit_mismatch"
  fi

  nv2_sha="$(portable_sha256 "$nv2_bin")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "native_v2" "$program_path" "$expected_exit" "$nv2_exit" "$claim_class" "$nv2_sha" "$nv2_status" \
    >>"$RESULTS_TSV"

  if [[ "$nv2_status" != "ok" ]]; then
    echo "[madaros-source-elf] FAIL: $case_id native_v2 expected_exit=$expected_exit actual_exit=$nv2_exit" >&2
    tail -n 80 "$nv2_compile_log" >&2 || true
    exit 1
  fi
done <"$MANIFEST_PATH"

echo "[madaros-source-elf] PASS: check, trace, normal/native-v2 compile, ELF execution, and exit-code semantics"
echo "[madaros-source-elf] results=$RESULTS_TSV"
