#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 ]]; then
    FILTER_ARGS=("$@")
else
    FILTER_ARGS=("ontology")
fi

bash scripts/ci/run_ontology_validation.sh --mode diff "${FILTER_ARGS[@]}"
