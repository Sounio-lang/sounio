#!/usr/bin/env bash
# madaros_multimodule_witness.sh — minimal multimodule native compile/run witnesses.
#
# Exercises the import-aware check/compile path (bin/madaros check|run|build),
# not --native-v2-compile (single-module bridge only).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-mm-witness] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-mm-witness] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

MANIFEST_PATH="${SOUNIO_MADAROS_MM_WITNESS_MANIFEST:-tests/madaros/multimodule_witness/manifest.tsv}"
OUT_DIR="${SOUNIO_MADAROS_MM_WITNESS_DIR:-$(mktemp -d /tmp/sounio-madaros-mm-witness.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
BIN_DIR="$OUT_DIR/bin"
RESULTS_TSV="$OUT_DIR/results.tsv"

mkdir -p "$LOG_DIR" "$BIN_DIR"

echo "[madaros-mm-witness] START"
echo "[madaros-mm-witness] madaros=$MADAROS_BIN"
echo "[madaros-mm-witness] manifest=$MANIFEST_PATH"
echo "[madaros-mm-witness] out=$OUT_DIR"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[madaros-mm-witness] FAIL: missing manifest $MANIFEST_PATH" >&2
  exit 1
fi

printf 'case_id\tprogram\tgate_mode\texpected_exit\tactual_exit\tstatus\n' >"$RESULTS_TSV"

pass_count=0
total_count=0

while IFS=$'\t' read -r case_id program_path expected_exit gate_mode || [[ -n "${case_id:-}" ]]; do
  if [[ -z "${case_id:-}" || "$case_id" == \#* ]]; then
    continue
  fi

  total_count=$((total_count + 1))

  if [[ ! -f "$program_path" ]]; then
    echo "[madaros-mm-witness] FAIL: missing program for $case_id: $program_path" >&2
    exit 1
  fi

  check_log="$LOG_DIR/$case_id.check.log"
  if ! "$MADAROS_BIN" check "$program_path" >"$check_log" 2>&1; then
    echo "[madaros-mm-witness] FAIL: check failed for $case_id" >&2
    tail -n 40 "$check_log" >&2 || true
    exit 1
  fi

  case "$gate_mode" in
    check)
      actual_exit="n/a"
      status="check_ok"
      ;;
    run)
      run_log="$LOG_DIR/$case_id.run.log"
      set +e
      "$MADAROS_BIN" run "$program_path" >"$run_log" 2>&1
      actual_exit=$?
      set -e
      status="ok"
      if [[ "$actual_exit" != "$expected_exit" ]]; then
        status="exit_mismatch"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$case_id" "$program_path" "$gate_mode" "$expected_exit" "$actual_exit" "$status" \
          >>"$RESULTS_TSV"
        echo "[madaros-mm-witness] FAIL: $case_id expected_exit=$expected_exit actual_exit=$actual_exit" >&2
        tail -n 40 "$run_log" >&2 || true
        exit 1
      fi
      ;;
    compile)
      compile_log="$LOG_DIR/$case_id.compile.log"
      out_bin="$BIN_DIR/$case_id.elf"
      rm -f "$out_bin"
      if ! "$MADAROS_BIN" build "$program_path" -o "$out_bin" >"$compile_log" 2>&1; then
        echo "[madaros-mm-witness] FAIL: compile failed for $case_id" >&2
        tail -n 40 "$compile_log" >&2 || true
        exit 1
      fi
      if [[ ! -s "$out_bin" ]]; then
        echo "[madaros-mm-witness] FAIL: no ELF produced for $case_id at $out_bin" >&2
        exit 1
      fi
      actual_exit="n/a"
      status="compile_ok"
      ;;
    *)
      echo "[madaros-mm-witness] FAIL: unknown gate_mode '$gate_mode' for $case_id" >&2
      exit 1
      ;;
  esac

  pass_count=$((pass_count + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$program_path" "$gate_mode" "$expected_exit" "$actual_exit" "$status" \
    >>"$RESULTS_TSV"
  echo "[madaros-mm-witness] PASS: $case_id ($gate_mode)"
done <"$MANIFEST_PATH"

echo "[madaros-mm-witness] PASS: $pass_count/$total_count witnesses"
echo "[madaros-mm-witness] results=$RESULTS_TSV"
