#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SUBCOMMAND_SOUC_BIN="$SOUC_BIN"
if [[ "$(head -c 2 "$SOUC_BIN" 2>/dev/null)" != "#!" ]]; then
  export SOUC_NATIVE_BIN="$SOUC_BIN"
  SUBCOMMAND_SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
  chmod +x "$SUBCOMMAND_SOUC_BIN"
fi

MANIFEST="packages/epistemic-core/sounio.toml"
WITNESS="tests/packages/package_import_science_witness.sio"
NEGATIVE="tests/packages/package_import_missing_package.sio"
OUT_DIR="$(mktemp -d /tmp/sounio-package-import-science.XXXXXX)"
TEST_LIST="$OUT_DIR/epistemic_core_tests.tsv"

trap 'rm -rf "$OUT_DIR"' EXIT

printf '[package-import-science] souc=%s\n' "$SUBCOMMAND_SOUC_BIN"
printf '[package-import-science] stdlib=%s\n' "$SOUNIO_STDLIB_PATH"
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
if pkg.get("name") != "epistemic-core":
    errors.append("expected [package].name = epistemic-core")
if lib.get("name") != "epistemic_core":
    errors.append("expected [lib].name = epistemic_core")
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
    rows.append((name, str(manifest.parent / rel)))

if errors:
    for error in errors:
        print(f"[package-import-science] manifest error: {error}", file=sys.stderr)
    raise SystemExit(1)

for name, path in rows:
    print(f"{name}\t{path}")
PY

while IFS=$'\t' read -r name path; do
  [[ -n "$name" ]] || continue
  log="$OUT_DIR/${name}.log"
  printf '[package-import-science] run manifest test %s (%s)\n' "$name" "$path"
  if ! "$SUBCOMMAND_SOUC_BIN" run "$path" >"$log" 2>&1; then
    cat "$log" >&2
    printf '[package-import-science] FAIL: manifest test failed: %s\n' "$name" >&2
    exit 1
  fi
  cat "$log"
done <"$TEST_LIST"

printf '[package-import-science] run downstream witness %s\n' "$WITNESS"
WITNESS_LOG="$OUT_DIR/witness.log"
if ! "$SUBCOMMAND_SOUC_BIN" run "$WITNESS" >"$WITNESS_LOG" 2>&1; then
  cat "$WITNESS_LOG" >&2
  echo '[package-import-science] FAIL: downstream package witness failed' >&2
  exit 1
fi
cat "$WITNESS_LOG"
if ! grep -qF 'package import science witness:' "$WITNESS_LOG" || ! grep -qF 'passed' "$WITNESS_LOG"; then
  echo '[package-import-science] FAIL: downstream witness did not report a passing summary' >&2
  exit 1
fi

printf '[package-import-science] compile negative missing-package fixture %s\n' "$NEGATIVE"
NEGATIVE_LOG="$OUT_DIR/missing_package.log"
NEGATIVE_ELF="$OUT_DIR/missing_package.elf"
set +e
SOUNIO_SOUC_VERBOSE=1 "$SUBCOMMAND_SOUC_BIN" compile "$NEGATIVE" -o "$NEGATIVE_ELF" >"$NEGATIVE_LOG" 2>&1
negative_rc=$?
set -e
if [[ $negative_rc -eq 0 ]]; then
  cat "$NEGATIVE_LOG" >&2
  echo '[package-import-science] FAIL: missing package import compiled successfully' >&2
  exit 1
fi
if ! grep -qF 'error: unreadable import' "$NEGATIVE_LOG"; then
  cat "$NEGATIVE_LOG" >&2
  echo '[package-import-science] FAIL: missing package import did not produce unreadable import diagnostic' >&2
  exit 1
fi
if [[ -e "$NEGATIVE_ELF" ]]; then
  cat "$NEGATIVE_LOG" >&2
  echo '[package-import-science] FAIL: missing package import produced an executable' >&2
  exit 1
fi

echo '[package-import-science] PASS: package import contract is gated'
