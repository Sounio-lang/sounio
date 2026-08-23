#!/usr/bin/env bash
# Deprecated fail-closed classifier — promoted 2026-08-23. Delegates to the green gate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$ROOT/scripts/ci/madaros_propagate_monte_carlo_fnptr_gate.sh" "$@"
