#!/usr/bin/env bash
# Native ABI gate for mixed scalar/aggregate call and return shapes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_RAW="${SOUC_RAW:-$ROOT_DIR/bin/souc-linux-x86_64}"
MADAROS="${MADAROS:-$ROOT_DIR/bin/madaros}"
TMP_DIR="$(mktemp -d /tmp/sounio-native-aggregate-abi.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[native-aggregate-abi] FAIL: $*" >&2
  exit 1
}

echo "[native-aggregate-abi] 1/3 mixed Name-like aggregate args"
"$SOUC_RAW" tests/run-pass/native_abi_name_many_args.sio "$TMP_DIR/native_abi_name_many_args.elf" >/tmp/native_aggregate_abi_name.build.log 2>&1
chmod +x "$TMP_DIR/native_abi_name_many_args.elf"
set +e
"$TMP_DIR/native_abi_name_many_args.elf"
ABI_RC=$?
set -e
if [[ "$ABI_RC" -ne 0 ]]; then
  cat /tmp/native_aggregate_abi_name.build.log >&2
  fail "native_abi_name_many_args expected exit 0, got $ABI_RC"
fi

echo "[native-aggregate-abi] 2/3 lexer aggregate static method"
"$SOUC_RAW" tests/diagnose/multi_module_aggregate_abi_repro.sio "$TMP_DIR/multi_module_aggregate_abi_repro.elf" >/tmp/native_aggregate_abi_diag.build.log 2>&1
chmod +x "$TMP_DIR/multi_module_aggregate_abi_repro.elf"
DIAG_OUT="$("$TMP_DIR/multi_module_aggregate_abi_repro.elf" 2>&1)"
if [[ "$DIAG_OUT" != *"a"* || "$DIAG_OUT" != *"b kind=7"* ]]; then
  cat /tmp/native_aggregate_abi_diag.build.log >&2
  printf '%s\n' "$DIAG_OUT" >&2
  fail "aggregate diagnostic did not report TokenKind::Fn as 7"
fi

echo "[native-aggregate-abi] 3/3 Madaros imported aggregate fallback"
"$MADAROS" compile tests/multimodule/abi_name_many_args_main.sio -o "$TMP_DIR/madaros_imported_aggregate.elf" >/tmp/native_aggregate_abi_madaros.build.log 2>&1
if [[ ! -s "$TMP_DIR/madaros_imported_aggregate.elf" ]]; then
  cat /tmp/native_aggregate_abi_madaros.build.log >&2
  fail "Madaros imported aggregate compile did not produce an ELF"
fi
chmod +x "$TMP_DIR/madaros_imported_aggregate.elf"
set +e
"$TMP_DIR/madaros_imported_aggregate.elf"
MADAROS_RC=$?
set -e
if [[ "$MADAROS_RC" -ne 0 ]]; then
  cat /tmp/native_aggregate_abi_madaros.build.log >&2
  fail "Madaros imported aggregate expected exit 0, got $MADAROS_RC"
fi
if ! grep -q 'falling back to full modular IR' /tmp/native_aggregate_abi_madaros.build.log; then
  cat /tmp/native_aggregate_abi_madaros.build.log >&2
  fail "Madaros imported aggregate did not exercise the full modular IR fallback"
fi
if ! grep -q 'falling back to stage0 full compiler' /tmp/native_aggregate_abi_madaros.build.log; then
  cat /tmp/native_aggregate_abi_madaros.build.log >&2
  fail "Madaros imported aggregate fallback classification did not reach the stage0 rescue path"
fi

echo "NATIVE_AGGREGATE_ABI_PASS"
