#!/usr/bin/env bash
# Proves the concrete shipped-Madaros exclusive-reference D3 witnesses remain green.
# This is deliberately not a claim about IrModule memory-wall or trait-for-i64 methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
version="$("$SOUC" --version 2>&1 | head -1)"
if ! printf '%s\n' "$version" | grep -q 'Madaros'; then
    echo "MADAROS_D3_EXCLREF_SHIPPED_GATE_FAIL reason=not_madaros engine=$version" >&2
    exit 1
fi

bash scripts/madaros_unsplit_oct_mul_gate.sh
bash scripts/madaros_associator_field_native_gate.sh

echo "MADAROS_D3_EXCLREF_SHIPPED_GATE_OK"
