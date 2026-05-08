#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_HOF_CLOSURE_DIR:-$(mktemp -d /tmp/sounio-native-v2-hof-closure.XXXXXX)}"
MANIFEST_PATH="tests/native-v2/hof_closure/manifest.tsv"
SUMMARY_JSON="$OUT_DIR/artifacts/summary.json"

echo "[native-v2-hof] START: HOF closure promotion gate"
echo "[native-v2-hof] out=$OUT_DIR"
echo "[native-v2-hof] manifest=$MANIFEST_PATH"

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_PATH" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

python3 - "$SUMMARY_JSON" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
if payload.get("fallback_path") != "none":
    raise SystemExit(f"fallback_path={payload.get('fallback_path')!r}")
PY

echo "[native-v2-hof] PASS: HOF closure promotion gate; fallback_path=none"
echo "[native-v2-hof] summary=$SUMMARY_JSON"
