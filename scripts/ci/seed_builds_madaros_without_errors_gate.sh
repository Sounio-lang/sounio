#!/usr/bin/env bash
# The committed seed must build Madaros without a single `error:` line.
#
# The lean_single seed bin/souc-lean-single-x86_64 reports a type error found
# in an imported module and still exits 0 and writes the ELF. Building
# self-hosted/compiler/main.sio printed 32 such errors on main at fd59aa3ecb,
# all in self-hosted/check/check.sio, and the Madaros it wrote was used anyway.
# The code at an error site is whatever the seed emitted for an expression it
# had refused: a `(i64, bool, TypeEntry)` tuple read `saved.2` from the wrong
# slot, and `read_i64`/`write_i64` in checker_current_fn_name_roundtrip were
# bound to ir/serialize.sio's private byte helpers of the same name.
#
# This gate builds Madaros with the committed seed, exactly as a bootstrap
# would, and fails on any line that starts with `error`, whatever the exit
# status. It does not reuse the "Build Current-Source Madaros" job: that job
# derives its own seed from lean_single.sio, runs in another job, and on a
# cache hit produces no compile log to inspect.
#
#   scripts/ci/seed_builds_madaros_without_errors_gate.sh
#   SOUNIO_SEED_BIN=<elf>  test another seed instead of the committed one
#
# About 45 s and under 6 GB of RAM on x86-64 Linux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SEED="${SOUNIO_SEED_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
SRC="$ROOT/self-hosted/compiler/main.sio"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

fail() { echo "[seed-builds-madaros-without-errors] FAIL: $*" >&2; exit 1; }
say() { echo "[seed-builds-madaros-without-errors] $*"; }

unset SOUC_BIN SOUNIO_SOUC_BIN
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s unlimited 2>/dev/null || true

[[ -x "$SEED" ]] || fail "missing seed $SEED"
say "seed: $SEED ($(sha256sum "$SEED" | cut -c1-12))"

rc=0
"$SEED" "$SRC" "$W/madaros" >"$W/build.log" 2>&1 || rc=$?
n_err="$(grep -c '^error' "$W/build.log" || true)"
say "seed($SRC): rc $rc, $n_err error line(s)"
if [[ "$n_err" -ne 0 ]]; then
  grep '^error' "$W/build.log" | head -40 >&2
  fail "the seed reported $n_err error(s) building Madaros"
fi
[[ $rc -eq 0 ]] || { tail -20 "$W/build.log" >&2; fail "the seed exited rc $rc building Madaros"; }
[[ -s "$W/madaros" ]] || fail "the seed wrote no Madaros ELF"
[[ "$(head -c4 "$W/madaros" | od -An -tx1 | tr -d ' ')" == "7f454c46" ]] \
  || fail "the seed's output is not an ELF"

echo "SEED_BUILDS_MADAROS_WITHOUT_ERRORS_OK"
