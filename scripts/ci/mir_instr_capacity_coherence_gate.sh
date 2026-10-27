#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${MIR_INSTR_CAPACITY_SOURCE:-$ROOT/self-hosted/native/machine_ir.sio}"
EXPECTED_CAPACITY=4096

fail() {
  printf 'MIR_INSTR_CAPACITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail source_missing
command -v node >/dev/null 2>&1 || fail node_missing
node "$ROOT/scripts/ci/capacity_coherence.mjs" mir "$SOURCE" "$EXPECTED_CAPACITY" "$ROOT" || exit $?

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
printf 'MIR_INSTR_CAPACITY_PASS source=%s source_sha256=%s expected=%s head=%s\n' \
  "$SOURCE" "$source_sha256" "$EXPECTED_CAPACITY" "$head_sha"
