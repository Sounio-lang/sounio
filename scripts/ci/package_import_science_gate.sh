#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

# Package tests import stdlib modules and/or packages/*. lean_single resolves
# packages/ imports; Madaros default does not yet. Use lean_single for this gate.
LEAN_SOUC="$ROOT_DIR/bin/souc-lean-single-x86_64"
if [[ ! -x "$LEAN_SOUC" ]]; then
  LEAN_SOUC="$ROOT_DIR/bin/souc-linux-x86_64"
fi
if [[ ! -x "$LEAN_SOUC" ]]; then
  echo '[package-import-science] FAIL: lean_single compiler ELF not found' >&2
  exit 1
fi
SOUC_BIN="$LEAN_SOUC"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

PACKAGES=(
  packages/epistemic-core/sounio.toml
  packages/sounio-units/sounio.toml
  packages/sounio-formats/sounio.toml
  packages/sounio-io-primitives/sounio.toml
)

WITNESSES=(
  tests/packages/package_import_science_witness.sio
  tests/packages/package_import_units_witness.sio
)

NEGATIVE="tests/packages/package_import_missing_package.sio"
OUT_DIR="$(mktemp -d /tmp/sounio-package-import-science.XXXXXX)"

trap 'rm -rf "$OUT_DIR"' EXIT

run_package_test() {
  local label="$1"
  local path="$2"
  local log="$OUT_DIR/$(basename "$path").log"
  local elf="$OUT_DIR/$(basename "$path").elf"
  printf '[package-import-science] run %s (%s)\n' "$label" "$path"
  if ! "$SOUC_BIN" "$path" "$elf" >"$log" 2>&1; then
    cat "$log" >&2
    printf '[package-import-science] FAIL: compile failed: %s\n' "$label" >&2
    exit 1
  fi
  chmod +x "$elf"
  if ! "$elf" >"$log.run" 2>&1; then
    cat "$log" "$log.run" >&2
    printf '[package-import-science] FAIL: run failed: %s\n' "$label" >&2
    exit 1
  fi
  cat "$log" "$log.run"
}

printf '[package-import-science] souc=%s\n' "$SOUC_BIN"
printf '[package-import-science] stdlib=%s\n' "$SOUNIO_STDLIB_PATH"

for MANIFEST in "${PACKAGES[@]}"; do
  TEST_LIST="$OUT_DIR/$(basename "$(dirname "$MANIFEST")")_tests.tsv"
  printf '[package-import-science] manifest=%s\n' "$MANIFEST"

  python3 - "$ROOT_DIR" "$MANIFEST" >"$TEST_LIST" <<'PY'
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:
    print("python tomllib is required for sounio.toml validation", file=sys.stderr)
    raise SystemExit(2)

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
package_dir = root / manifest.parent
data = tomllib.loads((root / manifest).read_text(encoding="utf-8"))

errors = []
pkg = data.get("package", {})
lib = data.get("lib", {})
pkg_name = pkg.get("name", "")
lib_name = lib.get("name", "")
if not pkg_name:
    errors.append("missing [package].name")
if not lib_name:
    errors.append("missing [lib].name")
lib_path = lib.get("path")
if not lib_path:
    errors.append("missing [lib].path")
elif not (package_dir / lib_path).is_file():
    errors.append(f"missing lib path: {manifest.parent / lib_path}")

tests = data.get("test", [])
if not tests:
    errors.append("manifest must declare at least one [[test]]")

rows = []
for test in tests:
    if not test.get("run", True):
        continue
    name = test.get("name", "")
    rel = test.get("path", "")
    if not name:
        errors.append("manifest test missing name")
    if not rel:
        errors.append(f"manifest test {name or '<unnamed>'} missing path")
        continue
    full = package_dir / rel
    if not full.is_file():
        errors.append(f"manifest test path missing: {manifest.parent / rel}")
        continue
    rows.append((name, str(full)))

if errors:
    for error in errors:
        print(f"[package-import-science] manifest error: {error}", file=sys.stderr)
    raise SystemExit(1)

for name, path in rows:
    print(f"{name}\t{path}")
PY

  while IFS=$'\t' read -r name path; do
    [[ -n "$name" ]] || continue
    run_package_test "$name" "$path"
  done <"$TEST_LIST"
done

for WITNESS in "${WITNESSES[@]}"; do
  run_package_test "witness:$(basename "$WITNESS")" "$WITNESS"
done

for WITNESS in "${WITNESSES[@]}"; do
  case "$WITNESS" in
    *science_witness.sio)
      if ! grep -qF 'package import science witness:' "$OUT_DIR/$(basename "$WITNESS").log.run"; then
        echo '[package-import-science] FAIL: epistemic witness summary missing' >&2
        exit 1
      fi
      ;;
    *units_witness.sio)
      if ! grep -qF 'package import units witness: passed' "$OUT_DIR/$(basename "$WITNESS").log.run"; then
        echo '[package-import-science] FAIL: units witness summary missing' >&2
        exit 1
      fi
      ;;
  esac
done

printf '[package-import-science] compile negative missing-package fixture %s\n' "$NEGATIVE"
NEGATIVE_LOG="$OUT_DIR/missing_package.log"
NEGATIVE_ELF="$OUT_DIR/missing_package.elf"
set +e
"$SOUC_BIN" "$NEGATIVE" "$NEGATIVE_ELF" >"$NEGATIVE_LOG" 2>&1
negative_rc=$?
set -e
if [[ $negative_rc -eq 0 && -x "$NEGATIVE_ELF" ]]; then
  cat "$NEGATIVE_LOG" >&2
  echo '[package-import-science] FAIL: missing package import compiled successfully' >&2
  exit 1
fi
if ! grep -qE 'error(\[E[0-9]+\])?:' "$NEGATIVE_LOG"; then
  cat "$NEGATIVE_LOG" >&2
  echo '[package-import-science] FAIL: missing package import did not produce an error diagnostic' >&2
  exit 1
fi

echo '[package-import-science] PASS: multi-package import contract is gated'