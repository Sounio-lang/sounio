#!/usr/bin/env bash
# The one entry point every Madaros build-once producer and consumer calls:
# reuse a verified build already sitting at OUT if there is one, else build
# fresh and write its manifest. Callers never need to know whether OUT came
# from actions/upload-artifact (same ci.yml run), actions/cache (a separate
# workflow file, best-effort), or nothing at all (falls back to building from
# source -- today's exact behavior, zero regression risk on a miss).
#
# Usage:
#   bash scripts/ci/ensure_madaros_build.sh [OUT=artifacts/self-hosted/madaros]
#
# The manifest lives alongside OUT at "${OUT}-manifest.json" (the dash form is
# already covered by the .gitignore rule `artifacts/self-hosted/madaros-*` for
# the default OUT; other OUT paths used by CI live under /tmp and are never
# tracked either way).
#
# Environment:
#   SOUNIO_EXPECTED_SOURCE_SHA — forwarded to verify_madaros_build_manifest.sh
#                                and write_madaros_build_manifest.sh (normally
#                                ${{ github.sha }} in CI).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-$ROOT_DIR/artifacts/self-hosted/madaros}"
MANIFEST="${OUT}-manifest.json"

if [[ -f "$OUT" && -f "$MANIFEST" ]] && bash "$ROOT_DIR/scripts/ci/verify_madaros_build_manifest.sh" "$OUT" "$MANIFEST"; then
    echo "ensure_madaros_build: reusing verified Madaros build: $OUT"
    exit 0
fi

echo "ensure_madaros_build: no verified build at $OUT (or verification failed) -- building from source"

# Same convention madaros-witness-gate/gate-wave-0 already apply before calling
# build_modular_madaros.sh directly.
ulimit -s 1048576 2>/dev/null || true

SEED_OUT="${OUT}-seed.elf"
rm -f "$SEED_OUT"
SOUNIO_MADAROS_SEED_ELF_OUT="$SEED_OUT" bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$OUT"

bash "$ROOT_DIR/scripts/ci/write_madaros_build_manifest.sh" "$OUT" "$MANIFEST" "$SEED_OUT"
rm -f "$SEED_OUT"

bash "$ROOT_DIR/scripts/ci/verify_madaros_build_manifest.sh" "$OUT" "$MANIFEST"
echo "ensure_madaros_build: built and verified: $OUT"
