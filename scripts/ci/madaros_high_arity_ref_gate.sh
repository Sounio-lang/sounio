#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/tests/native-v2/madaros_high_arity_ref"

fail() {
  echo "[madaros-high-arity-ref] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_HIGH_ARITY_REF_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_HIGH_ARITY_REF_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing to remove pre-existing work directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-high-arity-ref.XXXXXX)"
fi

trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

RAW_MADAROS="${SOUNIO_MADAROS_HIGH_ARITY_REF_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW_MADAROS" ]]; then
  RAW_MADAROS="$WORK/madaros-current"
  compiler_source="current_source"
  bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW_MADAROS" \
    >"$WORK/build.log" 2>&1 || {
    cat "$WORK/build.log" >&2
    fail "current-source Madaros build failed"
  }
else
  compiler_source="override"
fi
[[ -x "$RAW_MADAROS" ]] || fail "Madaros ELF is missing or not executable: $RAW_MADAROS"

stack_kb="${SOUNIO_MADAROS_HIGH_ARITY_REF_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ && ${#stack_kb} -le 9 ]] || fail "invalid stack size: $stack_kb"
stack_before="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable"
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || fail "could not raise soft stack limit to ${stack_kb} KiB"
fi
stack_after="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable after update"
echo "[madaros-high-arity-ref] stack_kb before=$stack_before after=$stack_after requested=$stack_kb"
echo "[madaros-high-arity-ref] compiler_source=$compiler_source compiler_sha256=$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"

run_case() {
  local label="$1"
  local source="$2"
  local marker="$3"
  local elf="$WORK/${label}.elf"

  echo "[madaros-high-arity-ref] source=$label sha256=$(sha256sum "$source" | awk '{print $1}')"
  MADAROS_RAW_BIN="$RAW_MADAROS" "$ROOT_DIR/bin/madaros" check "$source" >"$WORK/${label}.check.log" 2>&1 || {
    cat "$WORK/${label}.check.log" >&2
    fail "$label witness did not check"
  }
  MADAROS_RAW_BIN="$RAW_MADAROS" "$ROOT_DIR/bin/madaros" compile "$source" -o "$elf" >"$WORK/${label}.compile.log" 2>&1 || {
    cat "$WORK/${label}.compile.log" >&2
    fail "$label witness did not compile"
  }
  [[ -s "$elf" ]] || fail "$label compiler did not emit an ELF"
  chmod +x "$elf"

  set +e
  timeout 30 "$elf" >"$WORK/${label}.stdout" 2>"$WORK/${label}.stderr"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]] || ! grep -Fxq "$marker" "$WORK/${label}.stdout"; then
    cat "$WORK/${label}.stdout" >&2 || true
    cat "$WORK/${label}.stderr" >&2 || true
    echo "[madaros-high-arity-ref] observed label=$label rc=$rc" >&2
    fail "$label witness did not preserve high-arity references"
  fi
  echo "[madaros-high-arity-ref] PASS label=$label rc=$rc marker=$marker"
}

expect_reborrow_rejected() {
  local label="$1"
  local source="$2"

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$ROOT_DIR/bin/madaros" check "$source" >"$WORK/${label}.check.log" 2>&1
  local rc=$?
  set -e
  tr -d '\r\n' <"$WORK/${label}.check.log" >"$WORK/${label}.check.normalized"
  if [[ "$rc" -eq 0 ]] || ! grep -Fq 'error[E009]' "$WORK/${label}.check.normalized" ||
     ! grep -Fq 'argument type does not match parameter' "$WORK/${label}.check.normalized" ||
     ! grep -Fq 'found &!&!' "$WORK/${label}.check.normalized"; then
    cat "$WORK/${label}.check.log" >&2 || true
    fail "$label did not preserve the explicit E009 reborrow boundary"
  fi
  echo "[madaros-high-arity-ref] PASS label=$label rc=$rc diagnostic=E009_ref_of_ref"
}

expect_ir_wall_rejected() {
  local label="$1"
  local source="$2"
  local elf="$WORK/${label}.elf"

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$ROOT_DIR/bin/madaros" compile "$source" -o "$elf" >"$WORK/${label}.compile.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]] || ! grep -Fq 'function `oversized_ir_body` needs' "$WORK/${label}.compile.log" ||
     ! grep -Fq 'IR instructions but IR_MAX_INSTRS is' "$WORK/${label}.compile.log" || [[ -e "$elf" ]]; then
    cat "$WORK/${label}.compile.log" >&2 || true
    echo "[madaros-high-arity-ref] observed label=$label rc=$rc elf=$([[ -e "$elf" ]] && echo present || echo absent)" >&2
    fail "$label IR_MAX_INSTRS overflow did not fail closed"
  fi
  echo "[madaros-high-arity-ref] PASS label=$label rc=$rc elf=absent diagnostic=IR_MAX_INSTRS"
}

run_case same_file "$FIXTURE_DIR/same_file.sio" MADAROS_HIGH_ARITY_REF_SAME_OK
run_case imported "$FIXTURE_DIR/imported_main.sio" MADAROS_HIGH_ARITY_REF_IMPORTED_OK
expect_reborrow_rejected reborrow_same "$FIXTURE_DIR/reborrow_negative_same.sio"
expect_reborrow_rejected reborrow_imported "$FIXTURE_DIR/reborrow_negative_imported.sio"
expect_ir_wall_rejected ir_max_instrs_same "$FIXTURE_DIR/ir_max_instrs_negative.sio"
expect_ir_wall_rejected ir_max_instrs_imported "$FIXTURE_DIR/ir_max_instrs_imported_main.sio"

echo "[madaros-high-arity-ref] PASS: argc=20..23 ordering, stack ref offsets=120/128/136/144, 20-input+3-output direct/bare calls, explicit reborrow rejection, and same/imported fail-closed IR wall"
