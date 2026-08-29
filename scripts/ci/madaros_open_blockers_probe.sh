#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_LOWERING_DIAGNOSTICS=0

usage() {
  cat <<'USAGE'
Usage: scripts/ci/madaros_open_blockers_probe.sh [options]

Reproduce known-open Madaros production blockers without promoting them into
required-pass source-to-ELF manifests.

Options:
  --diagnose-lowering  also run a read-only frontend/body-lowering trace for
                       the BSS/global witnesses and controls
  -h, --help           show this help
USAGE
}

while (($#)); do
  case "$1" in
    --diagnose-lowering)
      RUN_LOWERING_DIAGNOSTICS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

OUT_DIR="${SOUNIO_MADAROS_OPEN_BLOCKERS_DIR:-$(mktemp -d /tmp/sounio-madaros-open-blockers.XXXXXX)}"
SRC_DIR="$OUT_DIR/src"
BIN_DIR="$OUT_DIR/bin"
LOG_DIR="$OUT_DIR/logs"
REPORT="$OUT_DIR/report.tsv"
LOWERING_REPORT="$OUT_DIR/lowering_diagnostics.tsv"

mkdir -p "$SRC_DIR" "$BIN_DIR" "$LOG_DIR"

cat >"$SRC_DIR/call_noarg_exit42.sio" <<'SIO'
fn f() -> i64 with IO, Mut, Panic {
    42
}

fn main() -> i64 with IO, Mut, Panic {
    f()
}
SIO

cat >"$SRC_DIR/call_arg_id_exit42.sio" <<'SIO'
fn id(x: i64) -> i64 with IO, Mut, Panic {
    x
}

fn main() -> i64 with IO, Mut, Panic {
    id(42)
}
SIO

cat >"$SRC_DIR/global_read_exit4.sio" <<'SIO'
var COUNT: i64 = 4

fn main() -> i64 with IO, Mut, Panic {
    COUNT
}
SIO

cat >"$SRC_DIR/global_store_exit7.sio" <<'SIO'
var COUNT: i64 = 0

fn main() -> i64 with IO, Mut, Panic {
    COUNT = 7
    COUNT
}
SIO

echo "[madaros-open-blockers] START"
echo "[madaros-open-blockers] madaros=$MADAROS_BIN"
echo "[madaros-open-blockers] out=$OUT_DIR"

printf 'blocker_id\tcase_id\tmode\texpected_open_exit\tactual_exit\tstatus\n' >"$REPORT"

run_binary_exit() {
  local bin="$1"
  local stdout_log="$2"
  local stderr_log="$3"

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$bin" "$stdout_log" "$stderr_log" <<'PY'
import subprocess
import sys

with open(sys.argv[2], "wb") as out, open(sys.argv[3], "wb") as err:
    result = subprocess.run([sys.argv[1]], stdout=out, stderr=err)

code = result.returncode
if code < 0:
    code = 128 + (-code)
print(code)
PY
    return 0
  fi

  set +e
  "$bin" >"$stdout_log" 2>"$stderr_log"
  local rc=$?
  set -e
  echo "$rc"
}

run_case() {
  local blocker_id="$1"
  local case_id="$2"
  local src="$3"
  local mode="$4"
  local expected_open_exit="$5"

  local bin="$BIN_DIR/${case_id}.${mode}"
  local compile_log="$LOG_DIR/${case_id}.${mode}.compile.log"
  local stdout_log="$LOG_DIR/${case_id}.${mode}.stdout"
  local stderr_log="$LOG_DIR/${case_id}.${mode}.stderr"

  local cmd=()
  case "$mode" in
    normal)
      cmd=("$MADAROS_BIN" "$src" -o "$bin")
      ;;
    native_v2)
      cmd=("$MADAROS_BIN" --native-v2-compile "$src" -o "$bin")
      ;;
    build)
      cmd=("$MADAROS_BIN" build "$src" -o "$bin")
      ;;
    *)
      echo "error: unknown mode $mode" >&2
      exit 2
      ;;
  esac

  set +e
  "${cmd[@]}" >"$compile_log" 2>&1
  local compile_rc=$?
  local actual_exit="compile_rc_${compile_rc}"
  if [[ "$compile_rc" -eq 0 ]]; then
    chmod +x "$bin" 2>/dev/null || true
    actual_exit="$(run_binary_exit "$bin" "$stdout_log" "$stderr_log")"
  fi
  set -e

  local status="still_open"
  if [[ "$blocker_id" == "control" ]]; then
    status="ok"
  fi
  if [[ "$actual_exit" != "$expected_open_exit" ]]; then
    status="changed"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$blocker_id" "$case_id" "$mode" "$expected_open_exit" "$actual_exit" "$status" >>"$REPORT"
}

run_self_build_case() {
  local blocker_id="$1"
  local case_id="$2"
  local expected_open_exit="$3"

  local build_dir="$OUT_DIR/self_build"
  local out_bin="$build_dir/madaros"
  local build_log="$LOG_DIR/${case_id}.build.log"

  mkdir -p "$build_dir"

  set +e
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    bash scripts/ci/build_modular_madaros.sh "$out_bin" >"$build_log" 2>&1
  local build_rc=$?
  set -e

  local actual_exit="build_rc_${build_rc}"
  if [[ "$build_rc" -eq 0 && -x "$out_bin" ]]; then
    actual_exit="build_rc_0"
  fi

  local status="still_open"
  if [[ "$blocker_id" == "control" ]]; then
    status="ok"
  fi
  if [[ "$actual_exit" != "$expected_open_exit" ]]; then
    status="changed"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$blocker_id" "$case_id" "self_build" "$expected_open_exit" "$actual_exit" "$status" >>"$REPORT"
}

marker_state() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "present"
  else
    echo "absent"
  fi
}

run_lowering_diagnostic_case() {
  local case_id="$1"
  local src="$2"

  local out_bin="$BIN_DIR/${case_id}.diagnose_lowering"
  local log_path="$LOG_DIR/${case_id}.diagnose_lowering.log"
  local marker_m1="/tmp/nv2_m1_after_lower"
  local marker_m2="/tmp/nv2_m2_before_backend"
  local marker_m3="/tmp/nv2_m3_before_backend"

  rm -f "$marker_m1" "$marker_m2" "$marker_m3"

  set +e
  SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
    "$MADAROS_BIN" --native-v2-compile "$src" "$out_bin" >"$log_path" 2>&1
  local rc=$?
  set -e

  local last_lower_stage
  last_lower_stage="$(
    grep 'module_frontend_lower:' "$log_path" 2>/dev/null \
      | tail -n 1 \
      | sed 's/.*module_frontend_lower: //' \
      || true
  )"
  [[ -n "$last_lower_stage" ]] || last_lower_stage="none"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" \
    "$rc" \
    "$last_lower_stage" \
    "$(marker_state "$marker_m1")" \
    "$(marker_state "$marker_m2")" \
    "$(marker_state "$marker_m3")" \
    "$log_path" >>"$LOWERING_REPORT"
}

run_lowering_diagnostics() {
  printf 'case_id\trc\tlast_lower_stage\tmarker_m1_after_lower\tmarker_m2_before_backend\tmarker_m3_before_backend\tlog\n' >"$LOWERING_REPORT"

  run_lowering_diagnostic_case "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio"
  run_lowering_diagnostic_case "global_read_exit4" "$SRC_DIR/global_read_exit4.sio"
  run_lowering_diagnostic_case "global_store_exit7" "$SRC_DIR/global_store_exit7.sio"

  echo "[madaros-open-blockers] lowering_diagnostics=$LOWERING_REPORT"
  cat "$LOWERING_REPORT"
}

print_lowering_handoff_summary() {
  if [[ ! -s "$LOWERING_REPORT" ]]; then
    return 0
  fi

  local read_row store_row
  read_row="$(awk -F '\t' '$1 == "global_read_exit4" { print }' "$LOWERING_REPORT")"
  store_row="$(awk -F '\t' '$1 == "global_store_exit7" { print }' "$LOWERING_REPORT")"

  local read_rc read_stage read_m1 read_m2 read_m3 read_log
  local store_rc store_stage store_m1 store_m2 store_m3 store_log
  IFS=$'\t' read -r _ read_rc read_stage read_m1 read_m2 read_m3 read_log <<<"$read_row"
  IFS=$'\t' read -r _ store_rc store_stage store_m1 store_m2 store_m3 store_log <<<"$store_row"

  echo "[madaros-open-blockers] lowering_handoff:"
  echo "  blocker=BLK-20260621-codex-source-elf-normal-bss"
  echo "  status=resolved_expected"
  echo "  observed[global_read_exit4]=rc=$read_rc last_lower_stage=$read_stage markers=$read_m1/$read_m2/$read_m3 log=$read_log"
  echo "  observed[global_store_exit7]=rc=$store_rc last_lower_stage=$store_stage markers=$store_m1/$store_m2/$store_m3 log=$store_log"
  echo "  target=standard module_frontend_lower body lowering before backend handoff"
  echo "  acceptance=global_read_exit4 exits 4; global_store_exit7 exits 7; source-to-ELF gate green"
}

# Controls: user-function calls, with and without arguments, are no longer the
# open source-to-ELF blocker.
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "normal" "42"
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "native_v2" "42"
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "build" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "normal" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "native_v2" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "build" "42"

# Resolved source-to-ELF witnesses for BLK-20260621-codex-source-elf-normal-bss.
run_case "control" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "normal" "4"
run_case "control" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "native_v2" "4"
run_case "control" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "build" "4"
run_case "control" "global_store_exit7" "$SRC_DIR/global_store_exit7.sio" "build" "7"

# Resolved workspace self-build witness for
# BLK-20260621-codex-madaros-build-segfault.
run_self_build_case "control" "self_build_madaros" "build_rc_0"

if [[ "$RUN_LOWERING_DIAGNOSTICS" == "1" ]]; then
  run_lowering_diagnostics
  print_lowering_handoff_summary
fi

cat "$REPORT"

if awk -F '\t' 'NR > 1 && $6 == "changed" { found = 1 } END { exit found ? 0 : 1 }' "$REPORT"; then
  echo "[madaros-open-blockers] CHANGED: at least one known-open blocker witness changed; inspect $REPORT and update/close blocker records" >&2
  exit 1
fi

echo "[madaros-open-blockers] PASS: tracked blocker witnesses match expected status"
echo "[madaros-open-blockers] report=$REPORT"
