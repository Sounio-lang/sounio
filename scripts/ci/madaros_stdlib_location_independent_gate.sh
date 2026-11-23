#!/usr/bin/env bash
# A Madaros binary must not depend on where the stdlib is installed.
#
# The overflow trap of ADR-031 R1 writes "at path:line:col" into the binary.
# For a stdlib file that path was the absolute one, SOUNIO_STDLIB_PATH plus
# the module path, so one program compiled against the same stdlib in two
# directories gave two different ELFs. Measured 2026-11-23: with the stdlib at
# /tmp/claude-0/det/s1 and /tmp/claude-0/det/s2222 the ELFs differed at byte 25
# (e_entry), and two DARWIN runs of the same Madaros self-compile from the
# same commit gave H2 sha256 a617fc8f5be7 and 77cee681bdec because their work
# directories differed. A stdlib file is now embedded as "stdlib/<rest>".
#
# This gate builds Madaros from source with the committed seed (or takes
# SOUNIO_MADAROS_BIN), compiles one program that imports the stdlib against
# three copies of it (two directories of different length, and the first
# again with a trailing slash), and requires:
#   - the three ELFs to be byte-identical;
#   - the trap text to name stdlib/epistemic/claim_meet.sio;
#   - no copy's absolute directory to appear in any ELF.
#
#   scripts/ci/madaros_stdlib_location_independent_gate.sh
#   SOUNIO_MADAROS_BIN=<elf>  test that compiler instead of building one
#
# About 1 min and under 6 GB of RAM on x86-64 Linux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

fail() { echo "[madaros-stdlib-location-independent] FAIL: $*" >&2; exit 1; }
say() { echo "[madaros-stdlib-location-independent] $*"; }

unset SOUC_BIN SOUNIO_SOUC_BIN
ulimit -s unlimited 2>/dev/null || true

MAD="${SOUNIO_MADAROS_BIN:-}"
if [[ -z "$MAD" ]]; then
  SEED="$ROOT/bin/souc-lean-single-x86_64"
  [[ -x "$SEED" ]] || fail "missing seed $SEED"
  rc=0
  SOUNIO_STDLIB_PATH="$ROOT/stdlib" "$SEED" "$ROOT/self-hosted/compiler/main.sio" "$W/madaros" \
    >"$W/build.log" 2>&1 || rc=$?
  [[ $rc -eq 0 && -s "$W/madaros" ]] || { tail -20 "$W/build.log" >&2; fail "seed did not build Madaros (rc $rc)"; }
  chmod +x "$W/madaros"
  MAD="$W/madaros"
fi
[[ -x "$MAD" ]] || fail "Madaros $MAD is not executable"
say "madaros: $MAD ($(sha256sum "$MAD" | cut -c1-12))"

mkdir -p "$W/prog"
cat >"$W/prog/prog.sio" <<'EOF'
use epistemic::claim_meet::{held_centre, seat_code, seat_moments}

fn main() -> i64 with IO, Mut, Div, Epistemic {
    let s = seat_moments(1.0, 0.1, 1.2, 0.1)
    print("seat ")
    println(seat_code(s))
    0
}
EOF

A="$W/a"
B="$W/bbbbbbbbbbbbbbbbbbbbbbbb"
cp -r "$ROOT/stdlib" "$A"
cp -r "$ROOT/stdlib" "$B"

build() {  # build <stdlib root> <out>
  local rc=0
  (cd "$W/prog" && SOUNIO_STDLIB_PATH="$1" "$MAD" compile prog.sio -o "$2") >"$2.log" 2>&1 || rc=$?
  [[ $rc -eq 0 && -s "$2" ]] || { tail -20 "$2.log" >&2; fail "compile with SOUNIO_STDLIB_PATH=$1 failed (rc $rc)"; }
}
build "$A" "$W/out_a.elf"
build "$B" "$W/out_b.elf"
build "$A/" "$W/out_a_slash.elf"

for o in out_a out_b out_a_slash; do say "$o: $(sha256sum "$W/$o.elf" | cut -c1-12)"; done
cmp -s "$W/out_a.elf" "$W/out_b.elf" \
  || fail "the ELF depends on the stdlib directory ($(cmp "$W/out_a.elf" "$W/out_b.elf" | head -1))"
cmp -s "$W/out_a.elf" "$W/out_a_slash.elf" \
  || fail "the ELF depends on a trailing slash in SOUNIO_STDLIB_PATH"
grep -aq "at stdlib/epistemic/claim_meet.sio:" "$W/out_a.elf" \
  || fail "the trap text does not name stdlib/epistemic/claim_meet.sio"
for d in "$A" "$B"; do
  for o in out_a out_b out_a_slash; do
    if grep -aqF "$d/" "$W/$o.elf"; then fail "$o.elf contains the absolute directory $d"; fi
  done
done

echo "MADAROS_STDLIB_LOCATION_INDEPENDENT_OK"
