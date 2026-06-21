#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

OUT_DIR="${SOUNIO_MADAROS_OPEN_BLOCKERS_DIR:-$(mktemp -d /tmp/sounio-madaros-open-blockers.XXXXXX)}"
SRC_DIR="$OUT_DIR/src"
BIN_DIR="$OUT_DIR/bin"
LOG_DIR="$OUT_DIR/logs"
REPORT="$OUT_DIR/report.tsv"

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
  if [[ "$actual_exit" != "$expected_open_exit" ]]; then
    status="changed"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$blocker_id" "$case_id" "self_build" "$expected_open_exit" "$actual_exit" "$status" >>"$REPORT"
}

# Controls: user-function calls, with and without arguments, are no longer the
# open source-to-ELF blocker.
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "normal" "42"
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "native_v2" "42"
run_case "control" "call_noarg_exit42" "$SRC_DIR/call_noarg_exit42.sio" "build" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "normal" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "native_v2" "42"
run_case "control" "call_arg_id_exit42" "$SRC_DIR/call_arg_id_exit42.sio" "build" "42"

# BLK-20260621-codex-source-elf-normal-bss:
# Current main segfaults in the raw Madaros compiler while compiling minimal
# BSS-global read/store witnesses, before an ELF is produced.
run_case "BLK-20260621-codex-source-elf-normal-bss" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "normal" "compile_rc_139"
run_case "BLK-20260621-codex-source-elf-normal-bss" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "native_v2" "compile_rc_139"
run_case "BLK-20260621-codex-source-elf-normal-bss" "global_read_exit4" "$SRC_DIR/global_read_exit4.sio" "build" "compile_rc_139"
run_case "BLK-20260621-codex-source-elf-normal-bss" "global_store_exit7" "$SRC_DIR/global_store_exit7.sio" "build" "compile_rc_139"

# BLK-20260621-codex-madaros-build-segfault:
# Current main segfaults while the lean_single seed compiles
# self-hosted/compiler/main.sio into the modular Madaros artifact. This witness
# keeps that production-readiness blocker executable without editing compiler
# files owned by the active compiler lane.
run_self_build_case "BLK-20260621-codex-madaros-build-segfault" "self_build_madaros" "build_rc_139"

cat "$REPORT"

if awk -F '\t' 'NR > 1 && $6 == "changed" { found = 1 } END { exit found ? 0 : 1 }' "$REPORT"; then
  echo "[madaros-open-blockers] CHANGED: at least one known-open blocker witness changed; inspect $REPORT and update/close blocker records" >&2
  exit 1
fi

echo "[madaros-open-blockers] PASS: known-open blocker witnesses are still reproducible"
echo "[madaros-open-blockers] report=$REPORT"
