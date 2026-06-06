#!/usr/bin/env bash
# Multi-module link gate for --native-v2-compile (feature #3: lower+link imported
# module fn bodies into the native build).
#
# EVERY prior native-v2 gate was single-FILE. This one is multi-FILE: each witness
# is a DIRECTORY of .sio files with an entry prog.sio. The harness STAGES each
# witness into a fresh temp dir (preserving relative paths), cd's into the entry
# dir, and compiles prog.sio from there — modelling the verified manual repro
#   mkdir -p mm/helper; <write files>; cd mm; mc --native-v2-compile prog.sio
# A flat single-file invocation would NOT exercise import resolution (R8).
#
# Manifest-driven (MANIFEST.txt):
#   <int>                  : valid prog -> ELF runs to exit <int>            [counts N/N]
#   REJECT:<twin>          : ill-formed -> NO ELF + nonzero, AND its VALID <twin>
#                            must compile+run (proving the import resolved, so the
#                            reject is a real type/link reject not import-not-found)  [counts N/N]
#   KNOWN_MISCOMPILE:<int> : #2 import-typecheck-bypass slip, still slips to <int>;
#                            LOUD, NOT counted, forces nonzero exit               [uncounted]
#
# R5 (anti-xfail-lie): KNOWN_MISCOMPILE never inflates green N/N; each must STILL
# slip (still emit an ELF running to the documented wrong value). run.sh prints a
# LOUD "UNRESOLVED MULTI-MODULE SLIPS: k" line and exits NONZERO if k>0. A green
# N/N with k=0 would mean the #2 bypass was also closed.
# ANTI-OVERCLAIM: every compile rm's its -o target first (no stale-ELF read).
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
WDIR="$DIR/witnesses"
MAN="$DIR/MANIFEST.txt"

pass=0
tot=0
unresolved=0
unresolved_list=""

# Stage witness $1 into a fresh temp dir, compile entry prog.sio from the entry
# dir's cwd. Echoes one of: "ELF:<exit>"  or  "NOELF:<rc>".
stage_and_compile() {
  local wname="$1"
  local stage; stage="$(mktemp -d "/tmp/mmgate_${wname}.XXXXXX")"
  # copy the whole witness tree (preserves helper/ subdirs and relative paths)
  cp -a "$WDIR/$wname/." "$stage/"
  local out="$stage/out.elf"
  rm -f "$out"
  ( cd "$stage" && "$BIN" --native-v2-compile prog.sio -o "$out" >/dev/null 2>&1 )
  local crc=$?
  if [ -f "$out" ]; then
    chmod +x "$out"
    ( cd "$stage" && "./out.elf" >/dev/null 2>&1 ); local rc=$?
    rm -rf "$stage"
    echo "ELF:$rc"
  else
    rm -rf "$stage"
    echo "NOELF:$crc"
  fi
}

while read -r name expected rest; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  if [ ! -d "$WDIR/$name" ]; then
    echo "FAIL  $name (witness dir missing: $WDIR/$name)"
    tot=$((tot+1)); continue
  fi

  case "$expected" in
    REJECT:*)
      tot=$((tot+1))
      twin="${expected#REJECT:}"
      # 1) the valid twin must compile+run (proves the import path resolves)
      tres=$(stage_and_compile "$twin")
      if [[ "$tres" != ELF:* ]]; then
        echo "FAIL  $name (REJECT precheck: twin '$twin' did NOT compile -> can't prove import resolved; got $tres)"
        continue
      fi
      # 2) the defective witness must produce NO ELF + nonzero
      r=$(stage_and_compile "$name")
      if [[ "$r" == NOELF:* ]]; then
        rc="${r#NOELF:}"
        if [ "$rc" -ne 0 ]; then
          echo "PASS  $name (REJECT: no ELF, nonzero; twin '$twin' compiled -> genuine link/type reject)"
          pass=$((pass+1))
        else
          echo "FAIL  $name (no ELF but rc=0; expected nonzero reject)"
        fi
      else
        echo "FAIL  $name (expected REJECT; emitted ELF -> $r) -- UNSOUND, would link a collision"
      fi
      ;;
    KNOWN_MISCOMPILE:*)
      bad="${expected#KNOWN_MISCOMPILE:}"
      # Does NOT count toward N/N. Assert the slip is STILL live: ELF emitted,
      # runs to the documented WRONG value.
      r=$(stage_and_compile "$name")
      if [[ "$r" == NOELF:* ]]; then
        echo "FLAG  $name (KNOWN_MISCOMPILE but now REJECTED/no-ELF; promote manifest -> #2 may be landed)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(now-rejected)"
        continue
      fi
      got="${r#ELF:}"
      if [ "$got" = "$bad" ]; then
        echo "KNOWN $name (#2 import-typecheck bypass: ill-typed slips, ELF exit $got; documented, NOT counted)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(->$got)"
      else
        echo "FLAG  $name (KNOWN_MISCOMPILE:$bad but ELF now exits $got; promote/adjust manifest)"
        unresolved=$((unresolved+1)); unresolved_list="$unresolved_list $name(->$got!=$bad)"
      fi
      ;;
    *)
      # valid numeric expectation: ELF must run to exit <expected>.
      tot=$((tot+1))
      r=$(stage_and_compile "$name")
      if [[ "$r" == NOELF:* ]]; then
        echo "FAIL  $name (no ELF for VALID multi-module program; rc=${r#NOELF:}) -- FALSE-REJECT"
        continue
      fi
      got="${r#ELF:}"
      if [ "$got" = "$expected" ]; then
        echo "PASS  $name (ELF exit $got)"
        pass=$((pass+1))
      else
        echo "FAIL  $name (ELF exit $got want $expected) -- SILENT MISCOMPILE of valid multi-module"
      fi
      ;;
  esac
done < "$MAN"

echo "---- $pass/$tot ----"
if [ "$unresolved" -ne 0 ]; then
  echo "UNRESOLVED MULTI-MODULE SLIPS: $unresolved ::$unresolved_list"
  echo "  (these are the #2 import-typecheck-bypass class — DEFERRED; prereq: per-module"
  echo "   typecheck on the import path + fixing the single-module checker SRET/closure/"
  echo "   match-tuple crash class. See commit notes.)"
fi
# Green requires: all counted cases pass AND zero live unresolved slips.
if [ "$pass" != "$tot" ] || [ "$unresolved" -ne 0 ]; then
  exit 1
fi
exit 0
