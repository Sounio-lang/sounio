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
# The installed launcher must name the same toolchain version as the tree it
# was installed from (ADR-018 §1). A non-empty log is not enough: before the
# Madaros helpers were shipped, --version died on a missing source file and an
# `unknown` version would also have been non-empty.
expected_version="$(sed -n 's/^## \[\(.*\)\].*/\1/p' "$ROOT_DIR/CHANGELOG.md" | head -n 1)"
[[ -n "$expected_version" ]] || expected_version="$(sed -n 's/^version: //p' "$ROOT_DIR/CITATION.cff" | head -n 1)"
reported_version="$(grep -m1 -oE '^[^ ]+ \(madaros [0-9a-f]+\)$' "$LOG_DIR/souc-version.log" | cut -d' ' -f1 || true)"
if [[ "$reported_version" != "$expected_version" ]]; then
  echo "[install-support] FAIL: installed souc --version reports '${reported_version:-<none>}', the tree is '$expected_version'" >&2
  cat "$LOG_DIR/souc-version.log" >&2
  exit 1
fi
echo "[install-support] souc-version names $reported_version"

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

# A release archive is this prefix, staged in a temporary directory and
# extracted elsewhere. Every link must therefore be relative, and the toolchain
# must still run once the prefix has moved: before this check, install.sh wrote
# absolute links into the staging directory and the extracted 3.0.0-rc.2
# bin/madaros died on a dangling scripts/lib/resolve_madaros.sh.
absolute_links="$(find "$PREFIX" -type l -lname '/*' -print)"
if [[ -n "$absolute_links" ]]; then
  echo "[install-support] FAIL: absolute symlinks in the prefix (they dangle once the archive is extracted):" >&2
  echo "$absolute_links" >&2
  exit 1
fi
MOVED="$WORK_DIR/moved-prefix"
mv "$PREFIX" "$MOVED"
run_step moved-madaros-version env -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  "$MOVED/bin/madaros" --version
assert_nonempty_log moved-madaros-version
# No SOUNIO_STDLIB_PATH: the moved prefix must find its stdlib through its own
# (relative) stdlib link.
run_step moved-souc-check-hello env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH \
  -u MADAROS_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  "$MOVED/bin/souc" check examples/hello.sio
mv "$MOVED" "$PREFIX"

if [[ "$KEEP_WORK_DIR" == "1" ]]; then
  echo "SOUNIO_INSTALL_SUPPORT_GATE_ARTIFACT_DIR=$WORK_DIR"
fi
echo "SOUNIO_INSTALL_SUPPORT_GATE_PASS prefix=$PREFIX"
