#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# Dynlink tables coherence.
#
# The dynlink symbol -> library mapping lives in TWO hand-written tables in
# self-hosted/native/codegen_x86_linux.sio:
#
#   native_v2_dynlink_lib_id_for_name    symbol name -> lib id
#   native_v2_dynlink_soname_for_lib_id  lib id      -> DT_NEEDED soname
#
# Nothing in the language ties them together, and they fail ASYMMETRICALLY.
# A lib id present in the first table but missing from the second does not
# error: the soname lookup falls off the end of its chain and returns its
# final literal, "libkl14b_probe.so". The ELF then carries a real DT_NEEDED
# naming the WRONG library, ld.so loads it, and the symbol resolves to
# whatever that library happens to export. It assembles, it links, it runs.
#
# Measured 2026-09-18 while removing the 4/8/256 dynlink caps: deleting the
# `if lib_id == 5 { return "libzstd.so.1" }` row alone left the compiler
# building rc=0 and a ZSTD_compress call would have been bound against
# libkl14b_probe.so. Two agreeing tables read exactly like consistency.
# This gate counts the two id sets instead of assuming them.
set -euo pipefail
cd "$(dirname "$0")/../.."
# The anti-vacuity primitives. This gate reads its facts out of source with
# regexes, and a regex that stops matching reports on ZERO rows and calls that
# agreement. So the count is floored, not trusted.
. "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_assert.sh"
gate_name "dynlink_tables_coherence"
CG=self-hosted/native/codegen_x86_linux.sio
CHECK=scripts/ci/lib/dynlink_tables_coherence.py
ART=artifacts/gates/dynlink_tables_coherence.v1.json
mkdir -p "$(dirname "$ART")"

run() { python3 "$CHECK" "$1"; }

# Positive control FIRST. A checker that has never failed has measured nothing:
# if the sabotaged copy passes, the gate is inspecting nothing and must not be
# allowed to report green on the real file. The sabotage removes one soname row
# -- the exact silent-wrong-library case above.
TRAPDIR=$(mktemp -d)
trap 'rm -rf "$TRAPDIR"' EXIT
mkdir -p "$TRAPDIR/self-hosted/native" "$TRAPDIR/scripts/ci/lib"
cp "$CHECK" "$TRAPDIR/scripts/ci/lib/dynlink_tables_coherence.py"
sed '/if lib_id == 5 { return "libzstd.so.1" }/d' \
  "$CG" > "$TRAPDIR/self-hosted/native/codegen_x86_linux.sio"
if python3 "$TRAPDIR/scripts/ci/lib/dynlink_tables_coherence.py" "$TRAPDIR" >/dev/null 2>&1; then
  echo "CONTROL_FAIL: the sabotaged table passed. This gate inspects nothing."
  printf '{"status":"fail","reason":"positive control did not fire","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: sabotaged soname table rejected, as required"

if out=$(run . 2>&1); then
  echo "$out"
  # A green over nothing is the failure this gate exists to prevent elsewhere,
  # so it must not be able to commit it itself. Six is the number of library
  # ids on the day the floor was set; the mapping only grows, so a drop means
  # the extraction broke, not that libraries left.
  n=$(printf '%s' "$out" | sed -n 's/.*lib_ids=\([0-9,]*\).*/\1/p' | tr ',' '\n' | grep -c '[0-9]' || true)
  require_min_count "$n" 6 "dynlink library ids compared across the tables"
  printf '{"status":"pass","metrics":{"total":%s,"passed":%s,"failed":0,"not_run":0}}\n' "$n" "$n" | gate_write_artifact "$ART"
  exit 0
fi
echo "$out"
printf '{"status":"fail","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
exit 1
