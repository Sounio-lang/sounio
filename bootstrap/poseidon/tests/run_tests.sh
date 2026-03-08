#!/bin/bash
# Test runner for Poseidon VM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSEIDON="$SCRIPT_DIR/../poseidon"
POSEIDON_LIB_TESTS="$SCRIPT_DIR/../poseidon_lib_tests"
MANIFEST="$SCRIPT_DIR/fixtures_manifest.v1.json"
PASS=0
FAIL=0

hash_fixture_state() {
    python3 - "$MANIFEST" "$SCRIPT_DIR/fixtures" <<'PY'
import hashlib
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
fixture_dir = Path(sys.argv[2])
if not manifest_path.exists() or not fixture_dir.exists():
    print("")
    raise SystemExit(0)

digest = hashlib.sha256()
digest.update(manifest_path.read_bytes())
for fixture_path in sorted(fixture_dir.glob("*.soir")):
    digest.update(fixture_path.name.encode("utf-8"))
    digest.update(fixture_path.read_bytes())
print(digest.hexdigest())
PY
}

validate_manifest_and_print_cli_cases() {
    python3 - "$MANIFEST" "$SCRIPT_DIR/fixtures" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
fixture_dir = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
fixtures = payload.get("fixtures", [])

if payload.get("schema") != "sounio.poseidon.fixtures.v1":
    raise SystemExit("unexpected fixture manifest schema")

ids = [fixture["id"] for fixture in fixtures]
filenames = [fixture["filename"] for fixture in fixtures]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate fixture ids in manifest")
if len(filenames) != len(set(filenames)):
    raise SystemExit("duplicate fixture filenames in manifest")

manifest_files = sorted(Path(name).name for name in filenames)
actual_files = sorted(path.name for path in fixture_dir.glob("*.soir"))
if manifest_files != actual_files:
    raise SystemExit("manifest file list does not match fixtures directory")

for fixture in fixtures:
    if fixture["kind"] == "positive" and "c_cli" in fixture["tags"]:
        print(f"{fixture['id']}\t{fixture['filename']}\t{fixture['expected_exit_code']}")
PY
}

run_test() {
    local name="$1"
    local filename="$2"
    local expected="$3"
    local file="$SCRIPT_DIR/$filename"

    echo -n "Testing $name... "

    if [ ! -f "$file" ]; then
        echo "FAIL (file not found)"
        FAIL=$((FAIL + 1))
        return
    fi

    set +e
    "$POSEIDON" "$file" >/dev/null 2>&1
    local actual=$?
    set -e

    if [ "$actual" -eq "$expected" ]; then
        echo "PASS (exit code $actual)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

before_hash="$(hash_fixture_state)"

echo "Generating test SOIR files..."
python3 "$SCRIPT_DIR/gen_test_soir.py"

after_hash="$(hash_fixture_state)"
if [ -n "$before_hash" ] && [ "$before_hash" != "$after_hash" ]; then
    echo "FAIL (generator output changed checked-in fixtures or manifest)"
    exit 1
fi

echo ""
echo "Running tests..."
echo "================"

while IFS=$'\t' read -r fixture_id fixture_filename expected_exit; do
    [ -n "$fixture_id" ] || continue
    run_test "$fixture_id" "$fixture_filename" "$expected_exit"
done < <(validate_manifest_and_print_cli_cases)

echo ""
echo "================"
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ -x "$POSEIDON_LIB_TESTS" ]; then
    echo "Running library API tests..."
    if "$POSEIDON_LIB_TESTS"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    echo ""
    echo "================"
    echo "Combined results: $PASS passed, $FAIL failed"
    echo ""
fi

if [ "$FAIL" -eq 0 ]; then
    echo "All tests passed!"
    exit 0
fi

echo "Some tests failed."
exit 1
