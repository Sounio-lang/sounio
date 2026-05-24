#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/tmp/sounio-dontology-ffi-importer}"

cc -std=c11 -Wall -Wextra -Werror -O2 \
  "$ROOT_DIR/scripts/ontology/dontology_ffi_importer.c" \
  -o "$OUT"

echo "$OUT"
