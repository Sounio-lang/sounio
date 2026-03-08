#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$ROOT_DIR/tests"
MANIFEST="$TEST_DIR/fixtures_manifest.v1.json"

mkdir -p corpus/load_bytes corpus/run
find corpus/load_bytes -maxdepth 1 -type f -delete
find corpus/run -maxdepth 1 -type f -delete

(
  cd "$TEST_DIR"
  python3 gen_test_soir.py >/dev/null
)

python3 - "$MANIFEST" "$TEST_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
tests_dir = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

for fixture in payload["fixtures"]:
    if "fuzz_seed" not in fixture["tags"]:
        continue
    source = tests_dir / fixture["filename"]
    for target_dir in (Path("corpus/load_bytes"), Path("corpus/run")):
        shutil.copy2(source, target_dir / source.name)
PY

echo "Refreshed cargo-fuzz corpora from Poseidon SOIR fixtures."
