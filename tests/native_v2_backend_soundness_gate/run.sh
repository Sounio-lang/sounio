#!/usr/bin/env bash
# Backend soundness gate for --native-v2-compile.
#
# Guards against SILENT MISCOMPILES: well-typed source (--check OK) that the
# native-v2 backend lowers into an ELF that runs to the WRONG value. These are
# the release wall (bugs A/B/C, all in self-hosted/ir/lower.sio).
#
# Manifest-driven (MANIFEST.txt). For each witness:
#   <int>                  : --check OK AND ELF runs to exit <int>          [counts toward N/N]
#   REJECT                 : --native-v2-compile emits NO ELF + nonzero     [counts toward N/N]
#   KNOWN_MISCOMPILE:<int> : DEFERRED bug, still miscompiles to <int>       [does NOT count toward N/N]
#
# R5 (anti-xfail-lie): KNOWN_MISCOMPILE witnesses never inflate the green N/N.
# Instead they must STILL be live (still miscompile). run.sh prints a LOUD
# "UNRESOLVED SILENT MISCOMPILES: k" line listing them and exits NONZERO if k>0.
# A passing N/N therefore means ZERO live silent miscompiles, not "ones excluded".
# If a KNOWN_MISCOMPILE witness no longer miscompiles (got fixed), that is also
# flagged as nonzero exit so the manifest must be promoted to a real expectation.
#
# ANTI-OVERCLAIM: every compile rm's its -o target first, so a rejected/again
# compile can never read a stale prior ELF.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
WDIR="$DIR/witnesses"
MAN="$DIR/MANIFEST.txt"

pass=0
tot=0
unresolved=0
unresolved_list=""

# Compile a witness, return rc; ELF (if any) at $OUT. Always rm's $OUT first.
compile() {
  local src="$1"
  rm -f "$OUT"
  "$BIN" --native-v2-compile "$src" -o "$OUT" >/dev/null 2>&1
  return $?
}
checks_ok() { "$BIN" --check "$1" >/dev/null 2>&1; }

run_elf_exit() {
  # echoes the ELF exit code (only call when $OUT exists)
  chmod +x "$OUT"
  "$OUT" >/dev/null 2>&1
  echo $?
}

while read -r name expected bug; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  src="$WDIR/$name.sio"
  OUT="/tmp/bsgate_${name}.elf"
  if [ ! -f "$src" ]; then
    echo "FAIL  [$bug] $name (witness source missing: $src)"
    tot=$((tot+1)); continue
  fi

  case "$expected" in
    REJECT)
      tot=$((tot+1))
      compile "$src"; rc=$?
      if [ ! -f "$OUT" ] && [ "$rc" -ne 0 ]; then
        echo "PASS  [$bug] $name (REJECT: no ELF, nonzero)"
        pass=$((pass+1))
      else
        echo "FAIL  [$bug] $name (expected REJECT; elf_exists=$([ -f "$OUT" ] && echo YES || echo NO) rc=$rc)"
      fi
      ;;
    KNOWN_MISCOMPILE:*)
      bad="${expected#KNOWN_MISCOMPILE:}"
      # Does NOT count toward N/N. Assert the bug is STILL live: --check OK,
      # ELF emitted, runs to the documented WRONG value.
      if ! checks_ok "$src"; then
        echo "FLAG  [$bug] $name (KNOWN_MISCOMPILE but --check now REJECTS; promote manifest)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(check-changed)"
        continue
      fi
      compile "$src"
      if [ ! -f "$OUT" ]; then
        echo "FLAG  [$bug] $name (KNOWN_MISCOMPILE but now REJECTED/no-ELF; promote manifest)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(now-rejected)"
        continue
      fi
      got=$(run_elf_exit)
      if [ "$got" = "$bad" ]; then
        echo "KNOWN [$bug] $name (still miscompiles to $got; documented, NOT counted)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(->$got)"
      else
        echo "FLAG  [$bug] $name (KNOWN_MISCOMPILE:$bad but ELF now exits $got; promote manifest)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(->$got!=$bad)"
      fi
      ;;
    *)
      # Well-typed numeric expectation: --check OK AND ELF runs to exit <expected>.
      tot=$((tot+1))
      if ! checks_ok "$src"; then
        echo "FAIL  [$bug] $name (--check rejected a well-typed witness)"
        continue
      fi
      compile "$src"
      if [ ! -f "$OUT" ]; then
        echo "FAIL  [$bug] $name (no ELF emitted for well-typed witness)"
        continue
      fi
      got=$(run_elf_exit)
      if [ "$got" = "$expected" ]; then
        echo "PASS  [$bug] $name (exit $got)"
        pass=$((pass+1))
      else
        echo "FAIL  [$bug] $name (ELF exit $got want $expected) -- SILENT MISCOMPILE"
      fi
      ;;
  esac
done < "$MAN"

echo "---- $pass/$tot ----"
if [ "$unresolved" -ne 0 ]; then
  echo "UNRESOLVED SILENT MISCOMPILES: $unresolved ::$unresolved_list"
fi
# Green requires: all counted cases pass AND zero live unresolved miscompiles.
if [ "$pass" != "$tot" ] || [ "$unresolved" -ne 0 ]; then
  exit 1
fi
exit 0
