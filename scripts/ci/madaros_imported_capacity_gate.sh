#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-imported-capacity] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-imported-capacity.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN:-$WORK/madaros}"
DEP="$WORK/cap_dep.sio"
BOUNDARY_MAIN="$WORK/boundary_main.sio"
BOUNDARY_OUT="$WORK/boundary.elf"
BOUNDARY_LOG="$WORK/boundary.log"
BOUNDARY_RUN_LOG="$WORK/boundary.run.log"
OVERFLOW_MAIN="$WORK/overflow_main.sio"
OUT="$WORK/overflow.elf"
LOG="$WORK/overflow.log"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "rebuilt Madaros is missing or not executable: $MADAROS_ELF"

# READ THE CAP, DO NOT PIN IT. This gate hardcoded 1023/1024 halves adding up to
# 2047/2048, and went red the moment IR_MAX_FUNCS moved (2048 -> 8192). A
# boundary gate whose boundary is a literal tests the literal, not the compiler.
# Same treatment as madaros_global_capacity_gate.sh.
#
# module_frontend_loaded_ir_slot_limit() reserves one transient slot when more
# than one module is loaded, so the limit these two-module witnesses actually hit
# is IR_MAX_FUNCS - 1. The witness splits it between an imported module and the
# main module, plus main itself.
IR_MAX_FUNCS="$(grep -E '^pub let IR_MAX_FUNCS: i64 = [0-9]+' \
    "$ROOT_DIR/self-hosted/ir/ir.sio" | grep -oE '[0-9]+$' | head -1)"
[[ -n "$IR_MAX_FUNCS" ]] || fail "IR_MAX_FUNCS is no longer declared where this gate looks"
SLOT_LIMIT="$((IR_MAX_FUNCS - 1))"
HALF="$(( (SLOT_LIMIT - 1) / 2 ))"        # deps, and locals at the boundary
LOCALS_OVER="$((SLOT_LIMIT - HALF))"      # one local past the limit

# THE PADDING MUST BE REACHABLE, or neither arm of this gate measures anything.
#
# It used to be padding nobody called -- `fn local<i>() { return i }`,
# `pub fn dep<i>() { return i }`, with main calling dep0 alone. That only ever
# worked because cross-module reachability DCE did not run on the ordinary
# multi-module path. This branch makes it run (that is how main.sio's 10705
# declared / 5997 live functions fit under the cap at all), and pruning happens
# BEFORE the slot-capacity check, by necessity -- checking first would refuse the
# very program the pruning exists to admit.
#
# So dead padding no longer occupies slots, and both arms silently stopped
# testing the ceiling:
#
#   overflow arm  lowered to 2 functions and COMPILED, so the gate went red with
#                 "aggregate capacity witness unexpectedly compiled"
#   boundary arm  lowered to 2 functions and PASSED -- the worse failure, because
#                 a green arm that never approaches the limit reads as proof the
#                 limit works
#
# The chain below is the construction tests/multimodule/ir_capacity already uses
# (`ir_cap_<i>() { ir_cap_<i+1>() + 1 }`), whose own comment warns that a capacity
# fixture whose witness stops reaching the ceiling "goes on passing for the wrong
# reason". Slot arithmetic is unchanged: every function is now LIVE, which is the
# only thing that changed.
#
# dep0 contributes the 7 the boundary arm checks and every other link adds
# nothing, so the chain evaluates to 7 at any length.
DEP_LAST="$((HALF - 1))"
for i in $(seq 0 "$DEP_LAST"); do
  if [[ "$i" -eq "$DEP_LAST" ]]; then
    # Terminator. Also the i==0 case when HALF==1, so dep0 must still yield 7.
    if [[ "$i" -eq 0 ]]; then
      printf 'pub fn dep%s() -> i64 { return 7 }\n' "$i" >>"$DEP"
    else
      printf 'pub fn dep%s() -> i64 { return 0 }\n' "$i" >>"$DEP"
    fi
  elif [[ "$i" -eq 0 ]]; then
    printf 'pub fn dep%s() -> i64 { return 7 + dep%s() }\n' "$i" "$((i + 1))" >>"$DEP"
  else
    printf 'pub fn dep%s() -> i64 { return dep%s() }\n' "$i" "$((i + 1))" >>"$DEP"
  fi
done

# `use` still names dep0 only. The rest of the chain is reached THROUGH it, so a
# green boundary arm now also proves the pass follows calls across the module
# boundary instead of stopping at the imported names it can see.
emit_local_chain() {
  local out="$1" last="$2" i
  for i in $(seq 0 "$last"); do
    if [[ "$i" -eq "$last" ]]; then
      printf 'fn local%s() -> i64 { return 0 }\n' "$i" >>"$out"
    else
      printf 'fn local%s() -> i64 { return local%s() }\n' "$i" "$((i + 1))" >>"$out"
    fi
  done
}

printf 'use cap_dep::{dep0}\n' >"$BOUNDARY_MAIN"
emit_local_chain "$BOUNDARY_MAIN" "$((HALF - 1))"
printf 'fn main() -> i64 { return dep0() + local0() }\n' >>"$BOUNDARY_MAIN"

# Compile via the raw ELF with a longer timeout than bin/madaros's default
# 300s. This witness is ~IR_MAX_FUNCS live functions across two modules; under
# CI load it has timed out at 300s (exit 124) after previously going green on
# the same tree. The wrapper timeout is for ordinary builds, not this probe.
compile_capacity_witness() {
  local src="$1" out="$2" log="$3"
  local stack_kb="${MADAROS_STACK_KB:-524288}"
  set +e
  (
    # Match bin/madaros: raw ELF needs ~512 MiB stack or this probe SEGVs (rc=139).
    if [[ "$stack_kb" == "0" ]]; then
      ulimit -s unlimited 2>/dev/null || true
    else
      ulimit -s "$stack_kb" 2>/dev/null || true
    fi
    ulimit -v "${MADAROS_VMEM_LIMIT_KB:-33554432}" 2>/dev/null || true
    exec timeout "${SOUNIO_MADAROS_IMPORTED_CAPACITY_TIMEOUT_SEC:-900}" \
      "$MADAROS_ELF" "$src" -o "$out"
  ) >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

set +e
boundary_compile_rc="$(compile_capacity_witness "$BOUNDARY_MAIN" "$BOUNDARY_OUT" "$BOUNDARY_LOG")"
set -e

if [[ "$boundary_compile_rc" -ne 0 ]]; then
  cat "$BOUNDARY_LOG" >&2
  fail "${SLOT_LIMIT}-slot imported boundary witness did not compile rc=$boundary_compile_rc"
fi
[[ -e "$BOUNDARY_OUT" ]] || fail "${SLOT_LIMIT}-slot imported boundary witness did not produce an output artifact"
chmod +x "$BOUNDARY_OUT"
set +e
"$BOUNDARY_OUT" >"$BOUNDARY_RUN_LOG" 2>&1
boundary_run_rc=$?
set -e
if [[ "$boundary_run_rc" -ne 7 ]]; then
  cat "$BOUNDARY_RUN_LOG" >&2
  fail "${SLOT_LIMIT}-slot imported boundary witness returned rc=$boundary_run_rc, expected 7"
fi

printf 'use cap_dep::{dep0}\n' >"$OVERFLOW_MAIN"
emit_local_chain "$OVERFLOW_MAIN" "$((LOCALS_OVER - 1))"
printf 'fn main() -> i64 { return dep0() + local0() }\n' >>"$OVERFLOW_MAIN"

set +e
compile_rc="$(compile_capacity_witness "$OVERFLOW_MAIN" "$OUT" "$LOG")"
set -e

if [[ "$compile_rc" -eq 0 ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity witness unexpectedly compiled"
fi
if [[ "$compile_rc" -ge 128 ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity witness terminated by signal rc=$compile_rc"
fi
if [[ -e "$OUT" ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity rejection left an output artifact: $OUT"
fi
grep -Fq "too many functions: shared IR module capacity exceeded (max ${SLOT_LIMIT} slots)" "$LOG" || {
  cat "$LOG" >&2
  fail "aggregate capacity diagnostic was missing or changed"
}
if grep -Fq 'ir_summary_failed' "$LOG"; then
  cat "$LOG" >&2
  fail "aggregate capacity rejection degraded to ir_summary_failed"
fi

echo "[madaros-imported-capacity] PASS: ${SLOT_LIMIT} imported slots execute and $((SLOT_LIMIT + 1)) are rejected before IR summary overflow"
