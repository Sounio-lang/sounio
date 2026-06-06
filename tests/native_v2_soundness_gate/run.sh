#!/usr/bin/env bash
# Soundness gate for --native-v2-compile.
#
# The hole this gate guards: --native-v2-compile USED to skip the typechecker
# entirely (it only checked parse/lower errors), so ill-typed source was silently
# miscompiled into a runnable ELF. The fix (run_native_v2_compile_mode in
# self-hosted/compiler/main.sio) now runs preflight_multimodule_frontend — the
# same typecheck path as --check — before emitting any ELF.
#
# This gate asserts, for each witness:
#   STRICT ill-typed (clean type errors E001/E016/E010/E008):
#     --native-v2-compile  => emits NO ELF and exits non-zero
#     --check              => REJECTS (exits non-zero)
#   WEAK ill-typed (undefined-var-return):
#     --native-v2-compile  => emits NO ELF and exits non-zero
#     (NOTE: --check still prints "check: OK" here — single-module name resolution
#      is skipped ["Resolve skip: single module with no imports"], a SEPARATE
#      pre-existing checker gap unrelated to the typecheck-skip hole this gate
#      fixes. We therefore only assert no-ELF + nonzero for this one, not --check
#      rejection.)
#   WELL-TYPED control:
#     --check              => passes
#     --native-v2-compile  => emits a correct-running ELF (exit == EXPECTED)
#
# Prints "N/N" and exits non-zero on any failure.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"

pass=0
tot=0

check_rejects() {
  # returns 0 if --check rejects (nonzero exit)
  "$BIN" --check "$1" >/dev/null 2>&1
  [ $? -ne 0 ]
}

compile_emits_no_elf_nonzero() {
  # arg1=src arg2=out ; returns 0 if compile emitted NO elf AND exited nonzero
  rm -f "$2"
  "$BIN" --native-v2-compile "$1" -o "$2" >/dev/null 2>&1
  local rc=$?
  if [ -f "$2" ]; then return 1; fi
  [ "$rc" -ne 0 ]
}

# --- STRICT ill-typed witnesses: no ELF + nonzero compile, AND --check rejects ---
for src in "$DIR"/illtyped/*.sio; do
  n=$(basename "$src" .sio)
  out="/tmp/sgate_$n.elf"
  tot=$((tot+1))
  if compile_emits_no_elf_nonzero "$src" "$out" && check_rejects "$src"; then
    echo "PASS  illtyped/$n (rejected by compile + check)"
    pass=$((pass+1))
  else
    echo "FAIL  illtyped/$n (expected: compile emits no ELF & nonzero, --check rejects)"
  fi
done

# --- WEAK ill-typed witness: no ELF + nonzero compile only (separate checker gap) ---
WEAK="$DIR/undefined-var-return.sio"
if [ -f "$WEAK" ]; then
  n="undefined-var-return"
  out="/tmp/sgate_$n.elf"
  tot=$((tot+1))
  if compile_emits_no_elf_nonzero "$WEAK" "$out"; then
    echo "PASS  $n (compile emits no ELF & nonzero; --check gap noted)"
    pass=$((pass+1))
  else
    echo "FAIL  $n (expected: compile emits no ELF & nonzero)"
  fi
fi

# --- WELL-TYPED controls: --check passes AND compile emits correct-running ELF ---
declare -A EXP
while read -r name val; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  EXP[$name]=$val
done < "$DIR/EXPECTED.txt"

for src in "$DIR"/control/*.sio; do
  n=$(basename "$src" .sio)
  out="/tmp/sgate_$n.elf"
  tot=$((tot+1))
  want="${EXP[$n]:-}"
  rm -f "$out"
  "$BIN" --check "$src" >/dev/null 2>&1
  check_rc=$?
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ "$check_rc" -eq 0 ] && [ -f "$out" ]; then
    chmod +x "$out"; "$out" >/dev/null 2>&1; rc=$?
    if [ "$rc" = "$want" ]; then
      echo "PASS  control/$n (check OK, ELF exit $rc)"
      pass=$((pass+1))
    else
      echo "FAIL  control/$n (ELF exit $rc want $want)"
    fi
  else
    echo "FAIL  control/$n (check_rc=$check_rc elf_exists=$([ -f "$out" ] && echo YES || echo NO))"
  fi
done

echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
