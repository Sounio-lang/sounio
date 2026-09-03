#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECK_FRESHNESS=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/generated_ontology_gate.sh [--refresh-only|--check]

Regenerates .dontology bundles and generated ontology .sio stubs.

Options:
  --refresh-only  Regenerate artifacts without checking git dirtiness.
  --check         Regenerate artifacts and fail if generated outputs drift.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --refresh-only)
            CHECK_FRESHNESS=0
            shift
            ;;
        --check)
            CHECK_FRESHNESS=1
            shift
            ;;
        *)
            echo "error: unsupported argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

make generated-ontology

if [[ "$CHECK_FRESHNESS" -eq 0 ]]; then
    exit 0
fi

STATUS="$(
    git status --porcelain -- \
        scripts/ontology/build_bundle.py \
        scripts/ontology/validate_bundle.py \
        scripts/ontology/build_dontology_ffi_importer.sh \
        scripts/ontology/dontology_ffi_importer.c \
        scripts/ontology/generate_dontology_stubs.sh \
        stdlib/data/data/ontology/source \
        stdlib/data/data/ontology/bundles \
        stdlib/ontology/generated \
        stdlib/ontology/typed \
        tests/run-pass/ontology_generated_*.sio \
        tests/run-pass/ontology_typed_bridge_*.sio \
        tests/compile-fail/ontology_generated_*.sio
)"

if [[ -n "$STATUS" ]]; then
    echo "error: generated ontology artifacts are not fresh" >&2
    echo "$STATUS" >&2
    echo "" >&2
    echo "Run: make generated-ontology" >&2
    exit 1
fi
