#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_unit_constraint_probe.sio"
POSITIVE="tests/frontend/knowledge_static_composite_positive.sio"
NEGATIVE_ONTOLOGY="tests/frontend/knowledge_static_composite_reject_ontology.sio"
NEGATIVE_UNIT="tests/frontend/knowledge_static_composite_reject_unit.sio"
NEGATIVE_REFINEMENT="tests/frontend/knowledge_static_composite_reject_refinement.sio"

run_expect() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local log="$TMP_DIR/$(basename "$path").log"
  if bin/souc run "$PROBE" -- "$path" >"$log" 2>&1 &&
     grep -q "knowledge_unit_ordinary_verdict=$expected" "$log" &&
     grep -q "knowledge_unit_semantic_verdict=$expected" "$log"; then
    printf 'PASS  %s %s\n' "$path" "$label"
  else
    printf 'FAIL  %s %s\n' "$path" "$label" >&2
    cat "$log" >&2
    exit 1
  fi
}

run_expect "$POSITIVE" 0 "accepted composite ontology+unit+numeric-refinement proof context"
run_expect "$NEGATIVE_ONTOLOGY" 1 "rejected composite proof context with failed ontology constraint"
run_expect "$NEGATIVE_UNIT" 1 "rejected composite proof context with failed unit constraint"
run_expect "$NEGATIVE_REFINEMENT" 1 "rejected composite proof context with weak numeric refinement"

cat <<'MSG'
Knowledge composite proof-context gate passed.
This proves a single `Knowledge<T where {...}>` context can combine ontology,
derived-unit dimensional, and numeric-refinement constraints as a conjunction in
the ordinary check::knowledge_context semantic bridge. The standalone
check::check import bridge remains covered by its own classifier. Runtime checks
for dynamic fields remain future work.
MSG
