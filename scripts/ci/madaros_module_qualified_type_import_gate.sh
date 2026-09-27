#!/usr/bin/env bash
# Copilot review (PR #2515), comment 4114124500: a path-form TYPE import
# (`use pkg::mod::Type;`) must still register its module-qualifier suffix.
# See tests/multimodule/module_qualified_type_import/README.md for why this
# gate checks (--check) rather than compiles-and-runs this fixture.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/module_qualified_type_import"
TAG="[madaros-module-qualified-type-import]"

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

if [[ -n "${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-module-qualified-type-import.XXXXXX)"
fi
if [[ -z "${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_MODULE_QUALIFIED_TYPE_IMPORT_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
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

# --- basic: a path-form type import must keep its module-qualifier suffix ---
# --check only (never compiled/run): see this fixture's own comment and the
# README for why -- this exact call shape (module::Type::method() through a
# PATH-FORM type import specifically) separately depends on lower.sio's
# callee_path_module_stripped_name, whose pre-existing uppercase-segment
# heuristic stops module-prefix stripping before a genuinely-registered
# "module::Type" suffix gets a chance to match, so a native compile+run
# here would fail for a reason unrelated to what this gate exists to pin.
log="$WORK/basic.log"
if ! "$RAW" --check "$FIX/basic/main.sio" >"$log" 2>&1; then
  tail -n 40 "$log" >&2 || true
  fail "basic: --check rejected a path-form type import's qualified associated call (its module-qualifier suffix was not registered)"
fi
grep -Fq 'check: OK' "$log" || {
  tail -n 40 "$log" >&2 || true
  fail "basic: --check did not report OK"
}
echo "$TAG PASS(basic): a path-form type import keeps its module-qualifier suffix (--check)"

echo "$TAG PASS: path-form type import resolved correctly"
