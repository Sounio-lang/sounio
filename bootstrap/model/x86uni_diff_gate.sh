#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FORMAL_DIR="$ROOT/formal"
CPU_SRC="$ROOT/bootstrap/model/x86uni_diff_cpu.cpp"
MODEL_LEAN="$FORMAL_DIR/Bootstrap/X86UniDiffModel.lean"
LAKE_BIN="${LAKE_BIN:-$(command -v lake || true)}"
if [ -z "$LAKE_BIN" ] && [ -x "$HOME/.elan/bin/lake" ]; then
  LAKE_BIN="$HOME/.elan/bin/lake"
fi

CASES="${DIFF_CASES:-60000}"
SEED="${DIFF_SEED:-0x554e4932303236}"

arch="$(uname -m)"
os="$(uname -s)"
if [ "$os" != "Linux" ] || [ "$arch" != "x86_64" ]; then
  echo "SKIP/FAIL: x86uni_diff_gate.sh requires Linux x86_64 hardware (got $os $arch)" >&2
  exit 1
fi

if [ -z "$LAKE_BIN" ]; then
  echo "FAIL: lake binary not found in PATH or \$HOME/.elan/bin" >&2
  exit 1
fi

work="$(mktemp -d /tmp/sounio-x86uni-diff.XXXXXX)"
cleanup() {
  rm -rf "$work"
}
trap cleanup EXIT

cpu_bin="$work/x86uni_diff_cpu"
cases_file="$work/cases.txt"
build_log="$work/lake_build.log"

CXX="${CXX:-g++}"
"$CXX" -O2 -std=c++20 -Wall -Wextra -Werror "$CPU_SRC" -o "$cpu_bin"
"$cpu_bin" "$cases_file" "$CASES" "$SEED"

UNI_DIFF_BIN="$FORMAL_DIR/.lake/build/bin/x86uni-diff"
if [ -x "$UNI_DIFF_BIN" ] && [ "${X86UNI_FORCE_LEAN_RUN:-0}" != "1" ] &&
   [ "$UNI_DIFF_BIN" -nt "$FORMAL_DIR/Bootstrap/X86Uni.lean" ] &&
   [ "$UNI_DIFF_BIN" -nt "$MODEL_LEAN" ]; then
  "$UNI_DIFF_BIN" "$cases_file" "$CASES"
else
  # Build upstream Bootstrap oleans quietly (surfacing build log only on failure),
  # ensure Bootstrap.X86Uni.olean is up to date, and run the model evaluator.
  if ! (cd "$FORMAL_DIR" && "$LAKE_BIN" build Bootstrap >"$build_log" 2>&1); then
    cat "$build_log" >&2
    exit 1
  fi
  uni_olean="$FORMAL_DIR/.lake/build/lib/lean/Bootstrap/X86Uni.olean"
  if [ ! -f "$uni_olean" ] || [ "$FORMAL_DIR/Bootstrap/X86Uni.lean" -nt "$uni_olean" ]; then
    mkdir -p "$(dirname "$uni_olean")"
    (cd "$FORMAL_DIR" && "$LAKE_BIN" env lean -o "$uni_olean" "Bootstrap/X86Uni.lean")
  fi
  (cd "$FORMAL_DIR" && "$LAKE_BIN" env lean --run "Bootstrap/X86UniDiffModel.lean" "$cases_file" "$CASES")
fi
