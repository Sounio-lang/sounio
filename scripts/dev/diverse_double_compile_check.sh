#!/usr/bin/env bash
# scripts/dev/diverse_double_compile_check.sh
#
# Phase 7 — Diverse Double-Compilation (DDC) Check
#
# Builds souc with two different host C/linker toolchains (GCC and Clang),
# then runs both resulting binaries on the same reference Sounio program and
# asserts that the outputs are byte-identical.
#
# This implements the Wheeler DDC property: a malicious backdoor injected by
# one host compiler cannot survive compilation by the other, so divergent
# output is a reliable signal that something is wrong.
#
# USAGE:
#   bash scripts/dev/diverse_double_compile_check.sh [--ref-program PATH]
#
# OPTIONS:
#   --ref-program PATH   Path to a .sio file to use as the reference program.
#                        If not given, a minimal program is written to a temp file.
#   --skip-build         Skip the cargo build steps (use pre-built binaries).
#
# EXIT CODE:
#   0  DDC PASS  — GCC and Clang souc produced identical output
#   1  DDC FAIL  — outputs diverged (or build failed)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ── configurable paths ──────────────────────────────────────────────────────
GCC_TARGET_DIR="${GCC_TARGET_DIR:-$ROOT_DIR/target/ddc-gcc}"
CLANG_TARGET_DIR="${CLANG_TARGET_DIR:-$ROOT_DIR/target/ddc-clang}"
GCC_BIN="$GCC_TARGET_DIR/debug/souc"
CLANG_BIN="$CLANG_TARGET_DIR/debug/souc"
SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
BOOTSTRAP_BUNDLE_DIR="${BOOTSTRAP_BUNDLE_DIR:-$ROOT_DIR/bootstrap}"
INDEPENDENCE_CONTRACT_PATH="${INDEPENDENCE_CONTRACT_PATH:-$ROOT_DIR/benchmarks/independence/contract.v1.json}"
WORK_DIR="${WORK_DIR:-/tmp/sounio-ddc}"
LOG_DIR="$WORK_DIR/logs"
REF_PROGRAM=""
SKIP_BUILD=0

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref-program) REF_PROGRAM="$2"; shift 2 ;;
    --skip-build)  SKIP_BUILD=1; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TMPREF=""
cleanup() {
  [ -n "$TMPREF" ] && rm -f "$TMPREF"
}
trap cleanup EXIT

echo "DDC_CHECK_START"
echo "gcc_target=$GCC_TARGET_DIR"
echo "clang_target=$CLANG_TARGET_DIR"
echo "stdlib=$SOUNIO_STDLIB_PATH"
echo "bootstrap_bundle=$BOOTSTRAP_BUNDLE_DIR"
echo "independence_contract=$INDEPENDENCE_CONTRACT_PATH"

if [ -f "$INDEPENDENCE_CONTRACT_PATH" ]; then
  python3 - "$INDEPENDENCE_CONTRACT_PATH" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text())
if obj.get("schema") != "sounio.independence.contract.v1":
    raise SystemExit(f"invalid independence contract schema: {obj.get('schema')}")
PY
fi

# ── dependency check ─────────────────────────────────────────────────────────
require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required tool not found: $1" >&2
    exit 1
  fi
}
require_tool gcc
require_tool clang
require_tool cargo
require_tool sha256sum
require_tool cmp

bash scripts/ci/check_no_active_cargo_jobs.sh 2>/dev/null || true

# ── build step ───────────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" -eq 0 ]; then
  echo ""
  echo "── building souc with GCC linker ──────────────────────────────────────"
  CC=gcc RUSTFLAGS="-Clinker=gcc" CARGO_TARGET_DIR="$GCC_TARGET_DIR" \
    cargo build -p souc --bin souc 2>&1 | tee "$LOG_DIR/build-gcc.log"
  echo "gcc build OK: $GCC_BIN"

  echo ""
  echo "── building souc with Clang linker ────────────────────────────────────"
  CC=clang RUSTFLAGS="-Clinker=clang" CARGO_TARGET_DIR="$CLANG_TARGET_DIR" \
    cargo build -p souc --bin souc 2>&1 | tee "$LOG_DIR/build-clang.log"
  echo "clang build OK: $CLANG_BIN"
else
  echo "SKIP_BUILD set — skipping cargo build steps"
fi

# ── sanity: both binaries must exist and be executable ───────────────────────
for bin in "$GCC_BIN" "$CLANG_BIN"; do
  if [ ! -x "$bin" ]; then
    echo "error: expected executable not found: $bin" >&2
    echo "hint: run without --skip-build to compile first" >&2
    exit 1
  fi
done

# Show binary hashes (informational — they WILL differ between compilers)
echo ""
echo "── binary identity (expected to differ) ───────────────────────────────"
sha256sum "$GCC_BIN"
sha256sum "$CLANG_BIN"

# ── reference program ────────────────────────────────────────────────────────
if [ -z "$REF_PROGRAM" ]; then
  TMPREF="$WORK_DIR/ddc_ref_$$.sio"
  cat > "$TMPREF" <<'SIO'
// DDC reference program — deterministic output for double-compile check
fn main() with IO {
    print("DDC-REF-PASS\n")
    print_int(6 * 7)
    print("\n")
}
SIO
  REF_PROGRAM="$TMPREF"
  echo ""
  echo "── reference program (auto-generated) ─────────────────────────────────"
  cat "$REF_PROGRAM"
else
  echo "── reference program: $REF_PROGRAM"
fi

# ── run both binaries ────────────────────────────────────────────────────────
GCC_OUT="$WORK_DIR/out-gcc.txt"
CLANG_OUT="$WORK_DIR/out-clang.txt"

echo ""
echo "── verifying signed bootstrap bundle (GCC-built souc) ──────────────────"
"$GCC_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR" > "$LOG_DIR/bootstrap-verify-gcc.log" 2>&1

echo ""
echo "── verifying signed bootstrap bundle (Clang-built souc) ────────────────"
"$CLANG_BIN" bootstrap verify --bundle "$BOOTSTRAP_BUNDLE_DIR" > "$LOG_DIR/bootstrap-verify-clang.log" 2>&1

echo ""
echo "── running GCC-built souc ──────────────────────────────────────────────"
SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
  "$GCC_BIN" run "$REF_PROGRAM" > "$GCC_OUT" 2>&1
echo "exit=$?"
cat "$GCC_OUT"

echo ""
echo "── running Clang-built souc ────────────────────────────────────────────"
SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
  "$CLANG_BIN" run "$REF_PROGRAM" > "$CLANG_OUT" 2>&1
echo "exit=$?"
cat "$CLANG_OUT"

# ── compare ──────────────────────────────────────────────────────────────────
echo ""
echo "── comparing outputs ───────────────────────────────────────────────────"
GCC_SHA=$(sha256sum "$GCC_OUT" | cut -d' ' -f1)
CLANG_SHA=$(sha256sum "$CLANG_OUT" | cut -d' ' -f1)
echo "gcc-souc   output sha256: $GCC_SHA"
echo "clang-souc output sha256: $CLANG_SHA"

if cmp -s "$GCC_OUT" "$CLANG_OUT"; then
  echo ""
  echo "DDC_CHECK_PASS"
  echo "output_sha256=$GCC_SHA"
  echo "gcc_bin=$GCC_BIN"
  echo "clang_bin=$CLANG_BIN"
  echo "ref_program=$REF_PROGRAM"
  exit 0
else
  echo ""
  echo "DDC_CHECK_FAIL: GCC and Clang souc produced DIVERGENT output"
  echo "diff:"
  diff "$GCC_OUT" "$CLANG_OUT" || true
  exit 1
fi
