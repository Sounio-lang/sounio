#!/usr/bin/env bash
# ADR-023 §1 bounds-check probe. Building with SOUNIO_NV2_BOUNDS_CHECK=1 makes
# local, by-value parameter and struct-field array indexing compare the index
# (unsigned) with the logical length in the object header, and exit 101 when
# it is out of range. The probe is off by default; this gate keeps it honest
# until it is turned on.
#
# Checks, with the compiler under test:
#   - each out-of-range fixture exits 101 with the probe on and never prints
#     NOT_REACHED;
#   - the in-range control prints BOUNDS_IN_RANGE_OK with the probe on and off;
#   - the out-of-range fixtures do not exit 101 with the probe off (the probe,
#     not something else, is what stops them).
#
# Usage: madaros_bounds_check_probe_gate.sh   (MADAROS_BIN = compiler under
#                                              test, default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
DIR="$ROOT_DIR/tests/madaros-bounds"
WORK="$(mktemp -d /tmp/sounio-bounds-probe.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-bounds-probe] FAIL: $*" >&2; exit 1; }

build() { # build <src> <out> <flag>
  ( ulimit -s 1572864 2>/dev/null || true
    SOUNIO_NV2_BOUNDS_CHECK="$3" "$MADAROS" build "$1" "$2" ) >"$WORK/build.log" 2>&1 \
    || { tail -n 20 "$WORK/build.log" >&2; fail "build failed: $1 (probe=$3)"; }
  chmod +x "$2"
}
run() { # run <elf> -> sets RC and OUT
  set +e; OUT="$("$1" 2>&1)"; RC=$?; set -e
}

for name in local_oob param_negative struct_field_store; do
  build "$DIR/$name.sio" "$WORK/$name.on" 1
  run "$WORK/$name.on"
  [[ "$RC" -eq 101 ]] || fail "$name with the probe on: rc=$RC (want 101)"
  [[ "$OUT" != *NOT_REACHED* ]] || fail "$name ran past the out-of-range index"
  build "$DIR/$name.sio" "$WORK/$name.off" 0
  run "$WORK/$name.off"
  [[ "$RC" -ne 101 ]] || fail "$name exits 101 with the probe off; the probe is not what stops it"
done

for flag in 1 0; do
  build "$DIR/in_bounds.sio" "$WORK/in_bounds.$flag" "$flag"
  run "$WORK/in_bounds.$flag"
  [[ "$RC" -eq 0 && "$OUT" == "BOUNDS_IN_RANGE_OK" ]] || fail "in_bounds (probe=$flag): rc=$RC output=$(tr '\n' ' ' <<<"$OUT")"
done
echo "[madaros-bounds-probe] PASS: out-of-range local, parameter and struct-field indices exit 101; in-range code is unchanged"
