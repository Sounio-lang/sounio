#!/usr/bin/env bash
# #2814: the two OSSM Hessian examples are the end-to-end pins for Madaros
# region reclamation (#2737, docs/decisions/adr-010-madaros-region-reclamation.md).
#
# Before #2737 the default engine died in both with "madaros: handles full"
# (rc=182): every >16 B struct built in the training loop kept a handle-table
# slot for the life of the process. The examples carried a //@ known-failure
# line for that; no runner covers examples/, so this gate is what runs them.
#
# For each example:
#   1. the default Madaros build must exit 0 and print the file's
#      `//@ expect-stdout:` line (ALL PASS);
#   2. the same build with SOUNIO_NO_REGION_RECLAIM=1 (the pre-#2737 lowering)
#      must still die with rc=182 "handles full" -- otherwise the example no
#      longer exercises reclamation and pins nothing.
#
# Instrument: MADAROS_RAW_BIN if set (a Madaros built from source), else the
# committed prebuilt that bin/madaros resolves. Each run is wrapped in a
# timeout; the slow examples take ~1 min each on a 2-core host.
#
# Usage: ossm_hessian_madaros_gate.sh [example.sio ...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="[ossm-hessian-madaros]"
fail() { echo "$TAG FAIL: $*" >&2; exit 1; }

case "$(uname -s 2>/dev/null)/$(uname -m 2>/dev/null)" in
  Linux/x86_64|Linux/amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

WORK="$(mktemp -d /tmp/sounio-ossm-hessian.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUNIO_NO_REGION_RECLAIM SOUNIO_RGN_BARRIER_SABOTAGE
RUN_TIMEOUT="${OSSM_GATE_RUN_TIMEOUT:-900}"

if [[ $# -gt 0 ]]; then
  EXAMPLES=("$@")
else
  EXAMPLES=(examples/ossm_hessian_newton.sio examples/ossm_hessian_lm_mandelbrot.sio)
fi

build() { timeout "$RUN_TIMEOUT" "$ROOT_DIR/bin/madaros" build "$1" -o "$2"; }

for rel in "${EXAMPLES[@]}"; do
  src="$ROOT_DIR/$rel"
  [[ -f "$src" ]] || fail "missing example $rel"
  if grep -q '^//@ known-failure' "$src"; then
    fail "$rel still carries //@ known-failure; this gate is its pin"
  fi
  want="$(sed -n 's|^//@ expect-stdout: *||p' "$src" | head -1)"
  [[ -n "$want" ]] || fail "$rel has no //@ expect-stdout line"
  base="$(basename "${rel%.sio}")"

  build "$src" "$WORK/$base.elf" >"$WORK/$base.build.log" 2>&1 \
    || { tail -n 30 "$WORK/$base.build.log" >&2; fail "$rel: Madaros build failed"; }
  rc=0; timeout "$RUN_TIMEOUT" "$WORK/$base.elf" >"$WORK/$base.out" 2>&1 || rc=$?
  if [[ "$rc" != 0 ]] || ! grep -qxF "$want" "$WORK/$base.out"; then
    tail -n 20 "$WORK/$base.out" >&2
    fail "$rel: exited $rc, expected 0 with '$want'"
  fi
  echo "$TAG PASS: $rel exits 0 and prints '$want'"

  SOUNIO_NO_REGION_RECLAIM=1 build "$src" "$WORK/$base.optout.elf" >"$WORK/$base.optout.build.log" 2>&1 \
    || { tail -n 30 "$WORK/$base.optout.build.log" >&2; fail "$rel: opt-out build failed"; }
  rc=0; timeout "$RUN_TIMEOUT" "$WORK/$base.optout.elf" >"$WORK/$base.optout.out" 2>&1 || rc=$?
  if [[ "$rc" != 182 ]] || ! grep -q "handles full" "$WORK/$base.optout.out"; then
    tail -n 20 "$WORK/$base.optout.out" >&2
    fail "$rel: without region reclamation it exited $rc (expected 182 'handles full'): the example no longer pins #2737"
  fi
  echo "$TAG PASS: $rel without region reclamation dies rc=182 'handles full' (pin has teeth)"
done
echo "$TAG PASS"
