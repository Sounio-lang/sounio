#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_F64_LADDER_DIR:-$(mktemp -d /tmp/sounio-native-v2-f64-ladder.XXXXXX)}"

mkdir -p "$OUT_DIR"

echo "[native-v2-f64] out=$OUT_DIR"
SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="tests/native-v2/f64_ladder/manifest.tsv" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-f64] PASS: generated stage1 f64 ladder including Knowledge<f64> witness"
