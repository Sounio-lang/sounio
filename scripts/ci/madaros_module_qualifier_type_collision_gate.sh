#!/usr/bin/env bash
# Copilot review (PR #2515), comment 4113834074: a real `use`d module can
# collide with an unrelated, incidentally-same-named lowercase struct/enum.
# See tests/multimodule/module_qualifier_type_collision/README.md for why
# this gate checks (--check) rather than compiles-and-runs this fixture.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/module_qualifier_type_collision"
TAG="[madaros-module-qualifier-type-collision]"

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

if [[ -n "${SOUNIO_MADAROS_MODULE_QUALIFIER_TYPE_COLLISION_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_MODULE_QUALIFIER_TYPE_COLLISION_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-module-qualifier-type-collision.XXXXXX)"
fi
if [[ -z "${SOUNIO_MADAROS_MODULE_QUALIFIER_TYPE_COLLISION_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_MODULE_QUALIFIER_TYPE_COLLISION_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
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

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# --- basic: a real used module must beat an unrelated same-named struct ---
# --check only (never compiled/run): see this fixture's own comment and the
# README for why -- this identical ambiguous shape separately trips the
# STILL-OPEN lowerer-side gap (comment 4113834083) that this round's fix
# does not touch, so a native compile+run here would fail for a reason
# unrelated to what this gate exists to pin.
log="$WORK/basic.log"
if ! "$RAW" --check "$FIX/basic/main.sio" >"$log" 2>&1; then
  tail -n 40 "$log" >&2 || true
  fail "basic: --check rejected a real used module qualifier colliding with an unrelated same-named struct (E012 or similar -- the struct's method won instead of the module's free fn)"
fi
grep -Fq 'check: OK' "$log" || {
  tail -n 40 "$log" >&2 || true
  fail "basic: --check did not report OK"
}
echo "$TAG PASS(basic): a real used module wins over an unrelated same-named lowercase struct (--check)"

echo "$TAG PASS: module qualifier vs. type-name collision resolved correctly"
