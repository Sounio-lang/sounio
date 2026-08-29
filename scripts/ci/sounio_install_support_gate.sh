#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${SOUNIO_INSTALL_SUPPORT_GATE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-install-support.XXXXXX")}"
KEEP_WORK_DIR="${SOUNIO_INSTALL_SUPPORT_GATE_KEEP_WORK_DIR:-0}"
PREFIX="$WORK_DIR/prefix"
LOG_DIR="$WORK_DIR/logs"

cleanup() {
  if [[ "$KEEP_WORK_DIR" != "1" ]]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$LOG_DIR"

run_step() {
  local label="$1"
  shift
  local log="$LOG_DIR/$label.log"

  echo "[install-support] >>> $label"
  if "$@" >"$log" 2>&1; then
    echo "[install-support] <<< $label PASS log=$log"
  else
    local rc=$?
    echo "[install-support] !!! $label FAIL rc=$rc log=$log" >&2
    tail -n 80 "$log" >&2 || true
    return "$rc"
  fi
}

assert_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "[install-support] FAIL: expected executable at $path" >&2
    exit 1
  fi
}

assert_nonempty_log() {
  local label="$1"
  local log="$LOG_DIR/$label.log"
  if [[ ! -s "$log" ]]; then
    echo "[install-support] FAIL: $label produced empty output ($log)" >&2
    exit 1
  fi
}

echo "SOUNIO_INSTALL_SUPPORT_GATE_START"
echo "repo=$ROOT_DIR"
echo "work_dir=$WORK_DIR"
echo "prefix=$PREFIX"

if [[ ! -x "$ROOT_DIR/scripts/install.sh" ]]; then
  echo "[install-support] FAIL: scripts/install.sh is not executable" >&2
  exit 1
fi

run_step install env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  bash "$ROOT_DIR/scripts/install.sh" --prefix "$PREFIX" --force

INSTALLED_SOUC="$PREFIX/bin/souc"
INSTALLED_MADAROS="$PREFIX/bin/madaros"
assert_executable "$INSTALLED_SOUC"

if [[ ! -x "$PREFIX/bin/madaros" || ! -x "$PREFIX/bin/madaros-linux-x86_64" ]]; then
  echo "[install-support] FAIL: installed prefix is missing the Madaros launcher/raw ELF needed by bin/souc --version/info" >&2
  echo "[install-support] expected: $PREFIX/bin/madaros and $PREFIX/bin/madaros-linux-x86_64" >&2
  echo "[install-support] install log: $LOG_DIR/install.log" >&2
  tail -n 40 "$LOG_DIR/install.log" >&2 || true
  exit 1
fi

INSTALLED_STDLIB="$PREFIX/lib/sounio/stdlib"
if [[ ! -d "$INSTALLED_STDLIB" ]]; then
  echo "[install-support] FAIL: expected installed stdlib at $INSTALLED_STDLIB" >&2
  exit 1
fi

run_step souc-version env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  SOUNIO_STDLIB_PATH="$INSTALLED_STDLIB" "$INSTALLED_SOUC" --version
assert_nonempty_log souc-version

run_step madaros-version env -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  "$INSTALLED_MADAROS" --version
assert_nonempty_log madaros-version

run_step souc-info env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  SOUNIO_STDLIB_PATH="$INSTALLED_STDLIB" "$INSTALLED_SOUC" info
assert_nonempty_log souc-info

run_step souc-check-hello env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  SOUNIO_STDLIB_PATH="$INSTALLED_STDLIB" "$INSTALLED_SOUC" check examples/hello.sio

if [[ "$KEEP_WORK_DIR" == "1" ]]; then
  echo "SOUNIO_INSTALL_SUPPORT_GATE_ARTIFACT_DIR=$WORK_DIR"
fi
echo "SOUNIO_INSTALL_SUPPORT_GATE_PASS prefix=$PREFIX"
