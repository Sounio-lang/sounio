#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${MADAROS_EXPECTED_SHA256:-}"
MADAROS="$ROOT/bin/madaros"
SINGLE="$ROOT/tests/selfhost/native_runtime/global_init_transport_single.sio"
IMPORTED="$ROOT/tests/selfhost/native_runtime/global_init_transport_imported.sio"
HOMONYM="$ROOT/tests/selfhost/native_runtime/global_init_homonym_imported.sio"

fail() {
  printf 'madaros-global-init-transport gate: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN is not executable: $RAW_MADAROS"
[[ "$EXPECTED_RAW_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || fail 'MADAROS_EXPECTED_SHA256 must pin the explicit Madaros ELF'
actual_raw_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$actual_raw_sha256" == "$EXPECTED_RAW_SHA256" ]] \
  || fail "Madaros SHA-256 mismatch: expected=$EXPECTED_RAW_SHA256 actual=$actual_raw_sha256"

grep -Fq 'global_init: Option<Box<Expr> >' "$ROOT/self-hosted/parser/ast.sio" \
  || fail 'global initializer is not owned by Item'
grep -Fq 'global_is_mutable: bool' "$ROOT/self-hosted/parser/ast.sio" \
  || fail 'global mutability is not preserved in Item'
grep -Fq 'global_init: Some(init_let_box)' "$ROOT/self-hosted/parser/items.sio" \
  || fail 'global let initializer is not attached to its Item'
grep -Fq 'global_init = Some(init_var_box)' "$ROOT/self-hosted/parser/items.sio" \
  || fail 'global var initializer is not attached to its Item'
if grep -Fq 'ast_lookup_global_var_init' "$ROOT/self-hosted/ir/lower.sio"; then
  fail 'IR lowering still consults the parser-global initializer side table'
fi
legacy_lookup_count="$(grep -R -F 'ast_lookup_global_var_init' "$ROOT/self-hosted" --include='*.sio' | wc -l)"
[[ "$legacy_lookup_count" == "1" ]] \
  || fail "legacy initializer side table still has consumers: occurrences=$legacy_lookup_count"
grep -Fq 'if init.0 {' "$ROOT/self-hosted/ir/lower.sio" \
  || fail 'explicit-zero initializer presence is not preserved'
grep -Fq 'instr.fn_id = global_fn_id' "$ROOT/self-hosted/ir/ir.sio" \
  || fail 'global load/store instructions do not carry slot identity'
grep -Fq 'fn lowerer_preseed_external_global_mut' "$ROOT/self-hosted/ir/lower.sio" \
  || fail 'imported globals are not preseeded with module identity'
grep -Fq 'func.param_count = func.param_count + bss_offset' "$ROOT/self-hosted/compiler/module_frontend.sio" \
  || fail 'active frontend merge does not rebase BSS slots'
grep -Fq '(*acc_box).bss_total_size = bss_offset + (*dep_box).bss_total_size' "$ROOT/self-hosted/compiler/module_frontend.sio" \
  || fail 'active frontend merge does not accumulate BSS size'

work="$(mktemp -d -t madaros-global-init-transport.XXXXXX)"
trap 'rm -rf "$work"' EXIT

run_witness() {
  local label="$1"
  local source="$2"
  local expected="$3"
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$work/$label.check.log" 2>&1 || {
    cat "$work/$label.check.log" >&2
    fail "$label witness did not check"
  }
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$source" >"$work/$label.run.log" 2>&1 || {
    cat "$work/$label.run.log" >&2
    fail "$label witness did not run"
  }
  cat "$work/$label.run.log"
  grep -Fxq "$expected" "$work/$label.run.log" \
    || fail "$label exact runtime receipt absent"
}

run_witness single "$SINGLE" \
  'GLOBAL_INIT_SINGLE neg=-3 pos=131072 version=5 zero=0 mutable=7'
run_witness imported "$IMPORTED" \
  'GLOBAL_INIT_IMPORTED neg=-3 direct_neg=-3 pos=131072 version=5 zero=0 mutable=7 direct_mutable=7 shared_after=41 nested=29'
run_witness homonym "$HOMONYM" \
  'GLOBAL_INIT_HOMONYM a_before=11 b_before=22 a_after=31 b_after=42'

printf 'madaros-global-init-transport gate: PASS\n'
printf 'madaros-global-init-transport receipt madaros_sha256=%s\n' "$actual_raw_sha256"
