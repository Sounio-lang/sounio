#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT_DIR/scripts/ci/science_boundary_gate.py" "$@"
