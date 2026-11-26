#!/usr/bin/env bash
# SOUNIO_STRICT_FIELD=1 refuses a guessed field index (E311); without it the
# compile is unchanged.
#
# What this gate pins:
#   positive   tests/run-pass/tuple_literal_struct_slot_field_typed.sio under
#              strict compiles and prints exactly "11 12 2": typed tuple slots
#              produce no E311
#   refusal    tests/strict_field/three_struct_slots.sio (a struct in a third
#              tuple slot, whose `x` sits at different indices in A and B) is
#              refused with E311 and a nonzero exit, standalone and through an
#              import (tests/strict_field/imported/)
#   opt-in     the same two refusal programs compile (rc 0) without the switch,
#              so the refusal is the switch and not a new default
# The refusal checks fail if the switch is ignored or the report is removed.
#
# In CI this gate does not build: it runs against the ELF kept by the
# current-source step (MADAROS_RAW_BIN). Standalone, with no ELF named, it
# builds Madaros from source.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="[madaros-strict-field]"
WORK="${SOUNIO_MADAROS_STRICT_FIELD_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-strict-field.XXXXXX)}"

fail() {
  echo "$TAG FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

mkdir -p "$WORK"
if [[ -z "${SOUNIO_MADAROS_STRICT_FIELD_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_STRICT_FIELD_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"

stack_kb="${SOUNIO_MADAROS_STRICT_FIELD_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable"
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || fail "could not raise soft stack limit to ${stack_kb} KiB"
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUNIO_STRICT_FIELD

# compile <label> <source> [ENV=VAL ...]  ->  rc in $WORK/<label>.rc, log in .log
compile() {
  local label="$1" src="$2"
  shift 2
  set +e
  env "$@" "$RAW" compile "$src" -o "$WORK/$label.elf" >"$WORK/$label.log" 2>&1
  echo $? >"$WORK/$label.rc"
  set -e
}

POS="$ROOT_DIR/tests/run-pass/tuple_literal_struct_slot_field_typed.sio"
REF="$ROOT_DIR/tests/strict_field/three_struct_slots.sio"
IMP="$ROOT_DIR/tests/strict_field/imported/main.sio"

# positive: strict, typed slots, exact output
compile positive "$POS" SOUNIO_STRICT_FIELD=1
[[ "$(cat "$WORK/positive.rc")" == 0 ]] || { cat "$WORK/positive.log" >&2; fail "positive witness refused under strict"; }
! grep -q 'error\[E311\]' "$WORK/positive.log" || fail "positive witness reported E311"
chmod +x "$WORK/positive.elf"
got="$("$WORK/positive.elf")" || fail "positive witness exited nonzero"
[[ "$got" == "11 12 2" ]] || fail "positive witness printed '$got', expected '11 12 2'"
echo "$TAG positive: strict compile clean, output '$got'"

# refusal: standalone and imported
for case in standalone:"$REF" imported:"$IMP"; do
  label="${case%%:*}"
  src="${case#*:}"
  compile "strict-$label" "$src" SOUNIO_STRICT_FIELD=1
  rc="$(cat "$WORK/strict-$label.rc")"
  [[ "$rc" != 0 ]] || fail "$label: strict compile exited 0"
  grep -q 'error\[E311\]: the index of field `x`' "$WORK/strict-$label.log" \
    || { cat "$WORK/strict-$label.log" >&2; fail "$label: no E311 for field x"; }
  echo "$TAG refusal $label: rc=$rc, E311 on x"

  compile "normal-$label" "$src"
  [[ "$(cat "$WORK/normal-$label.rc")" == 0 ]] || { cat "$WORK/normal-$label.log" >&2; fail "$label: compile without the switch failed"; }
  ! grep -q 'error\[E311\]' "$WORK/normal-$label.log" || fail "$label: E311 without the switch"
  echo "$TAG opt-in $label: rc=0 without SOUNIO_STRICT_FIELD"
done

echo "$TAG PASS"
