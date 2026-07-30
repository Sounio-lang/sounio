#!/usr/bin/env bash
# #901: execute imported field access on both sides of the old 256-layout cliff.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_SHA="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_KEEP:-0}"
WORK="${SOUNIO_MADAROS_STRUCT_LAYOUT_CAPACITY_DIR:-}"
GENERATOR="$ROOT_DIR/scripts/research/generate_madaros_struct_layout_capacity_fixture.py"

fail() { echo "[madaros-struct-layout-capacity] FAIL: $*" >&2; exit 1; }
sha256() { sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'; }
elf_magic() { od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n'; }
elf_u16() { od -An -tu2 -j "$2" -N2 "$1" 2>/dev/null | tr -d ' \n'; }

assert_elf() {
  local path="$1" label="$2"
  [[ -s "$path" ]] || fail "$label is missing or empty"
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$label is not ELF"
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail "$label is not ET_EXEC"
  [[ "$(elf_u16 "$path" 18)" == 62 ]] || fail "$label is not x86-64"
}

assert_no_fallback() {
  local log="$1"
  if grep -Eiq 'native_prebundle:|falling back to full IR path|specialized lower failed|multi-mod fallback|compact modular IR table path|legacy compact IR differential enabled' "$log"; then
    cat "$log" >&2
    fail "fallback marker observed in $log"
  fi
}

[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit Madaros ELF'
assert_elf "$RAW_MADAROS" 'Madaros input'
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
RAW_SHA="$(sha256 "$RAW_MADAROS")"
[[ -z "$EXPECTED_SHA" || "$RAW_SHA" == "$EXPECTED_SHA" ]] || fail "compiler SHA mismatch expected=$EXPECTED_SHA actual=$RAW_SHA"
[[ -f "$GENERATOR" ]] || fail "generator missing: $GENERATOR"

if [[ -n "$WORK" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-struct-layout-capacity.XXXXXX)"
fi
[[ "$KEEP_WORK" == 1 ]] || trap 'rm -rf "$WORK"' EXIT

run_case() {
  local custom="$1" dir="$WORK/custom-$1"
  mkdir -p "$dir"
  python3 "$GENERATOR" --custom-layouts "$custom" --out-dir "$dir"
  local src="$dir/layout_capacity_main.sio" elf="$dir/witness.elf"
  local marker="$(<"$dir/expected_marker.txt")"

  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$RAW_MADAROS" --check "$src" >"$dir/check.log" 2>&1
  local check_rc=$?
  set -e
  [[ "$check_rc" -eq 0 ]] || { cat "$dir/check.log" >&2; fail "$custom-layout checker failed rc=$check_rc"; }

  rm -f "$elf"
  set +e
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" "$RAW_MADAROS" --native-v2-compile "$src" -o "$elf" >"$dir/compile.log" 2>&1
  local compile_rc=$?
  set -e
  [[ "$compile_rc" -eq 0 ]] || { cat "$dir/compile.log" >&2; fail "$custom-layout compile failed rc=$compile_rc"; }
  assert_no_fallback "$dir/compile.log"
  assert_elf "$elf" "$custom-layout witness"
  chmod +x "$elf"
  set +e
  "$elf" >"$dir/runtime.log" 2>&1
  local runtime_rc=$?
  set -e
  [[ "$runtime_rc" -eq 0 ]] || { cat "$dir/runtime.log" >&2; fail "$custom-layout ELF rc=$runtime_rc"; }
  grep -Fxq "$marker" "$dir/runtime.log" || { cat "$dir/runtime.log" >&2; fail "$custom-layout exact marker absent"; }
}

run_case 255
run_case 256
echo "[madaros-struct-layout-capacity] PASS raw_sha256=$RAW_SHA catalog_layouts=256,257 executable=1 fallback=0"
