#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_SEMANTIC_HARDENING_DIR:-$(mktemp -d /tmp/sounio-native-v2-semantic-hardening.XXXXXX)}"

echo "[native-v2-semantic] out=$OUT_DIR"
SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="tests/native-v2/semantic_hardening/manifest.tsv" \
SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_SMOKE_TIMEOUT="${SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_SMOKE_TIMEOUT:-30}" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-semantic] PASS: generated stage1 semantic hardening corpus"
