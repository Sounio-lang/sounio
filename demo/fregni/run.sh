#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOUC="${SOUC:-$ROOT/bin/souc}"
cd "$(dirname "$0")"

echo "=== Sounio x PPCR demo for Prof. Felipe Fregni ==="
echo ""

echo "--- 1. Reference arithmetic (Python) ---"
python3 reference.py
echo ""

echo "--- 2. Confidence-gated dosing pipeline (Sounio/Madares) ---"
"$SOUC" run fregni_demo.sio
echo ""

echo "--- 3. Compile-time provenance guard (expected to FAIL type check) ---"
set +e
"$SOUC" check bad_path.sio
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
    echo "(Expected failure: simulated value cannot be passed to measured-only extractor.)"
else
    echo "ERROR: bad_path.sio should have failed type check" >&2
    exit 1
fi

echo ""
echo "=== Demo complete ==="
