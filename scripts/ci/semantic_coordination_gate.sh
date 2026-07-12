#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGISTRY="$ROOT_DIR/docs/internal/concepts/registry.tsv"
BINDINGS="$ROOT_DIR/docs/internal/concepts/bindings.tsv"
CONTRACT="$ROOT_DIR/docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md"
SCANNER="$ROOT_DIR/scripts/dev/sounio_semantic_status.sh"
HOTSPOT_TRIAGE="$ROOT_DIR/docs/internal/coordination/semantic-hotspot-triage-2026-07-12.md"
ENIR_CONTRACT="$ROOT_DIR/docs/internal/coordination/enir-semantic-interface-contract-2026-07-12.md"
DOCS_REGISTRY="$ROOT_DIR/docs/governance/topic-registry.v1.json"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/semantic-coordination.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[semantic-coordination] FAIL: $*" >&2
  exit 1
}

for file in "$REGISTRY" "$BINDINGS" "$CONTRACT" "$SCANNER" "$HOTSPOT_TRIAGE" "$ENIR_CONTRACT" "$DOCS_REGISTRY"; do
  [[ -s "$file" ]] || fail "missing or empty file: $file"
done

for state in INTEGRATED REVIEW_READY ACTIVE STALE_WITH_RESIDUE SCRATCH_COPY; do
  grep -Fq "$state" "$HOTSPOT_TRIAGE" || fail "hotspot triage omits state $state"
done

grep -Fq 'ce2f94407' "$ENIR_CONTRACT" || fail "ENIR contract is not pinned to the inspected E3D tip"
for concept in SOUNIO-EPISTEMIC-NUMERIC-VALUE SOUNIO-ZERO-PROVENANCE SOUNIO-EXPLICIT-DISCHARGE SOUNIO-PRECISION-PRESERVATION; do
  grep -Fq "$concept" "$ENIR_CONTRACT" || fail "ENIR contract omits concept $concept"
done

awk -F '\t' '
  /^#/ || NF == 0 { next }
  NF != 6 { print "bad registry field count at line " NR > "/dev/stderr"; bad=1 }
  $1 !~ /^SOUNIO-[A-Z0-9-]+$/ { print "bad concept id at line " NR > "/dev/stderr"; bad=1 }
  $2 !~ /^(garden|hypothesis|executable|integrated|claim-ready|superseded)$/ {
    print "bad concept status at line " NR > "/dev/stderr"; bad=1
  }
  seen[$1]++ { print "duplicate concept id " $1 > "/dev/stderr"; bad=1 }
  END { if (bad || length(seen) < 6) exit 1 }
' "$REGISTRY" || fail "registry schema validation failed"

while IFS=$'\t' read -r concept status authority contract canonical pending extra; do
  [[ -z "$concept" || "$concept" == \#* ]] && continue
  [[ -z "${extra:-}" ]] || fail "extra registry field for $concept"
  [[ -f "$ROOT_DIR/$contract" ]] || fail "$concept contract does not exist: $contract"
  grep -Fq "Concept-ID: \`$concept\`" "$ROOT_DIR/$contract" || \
    fail "$contract does not declare $concept"
  grep -Fq "\"repo_doc_path\": \"$contract\"" "$DOCS_REGISTRY" || \
    fail "$contract is absent from docs governance registry"
  grep -Fq "$concept" "$BINDINGS" || fail "$concept has no path bindings"
done < "$REGISTRY"

awk -F '\t' -v registry="$REGISTRY" '
  BEGIN {
    while ((getline line < registry) > 0) {
      if (line ~ /^#/ || line == "") continue
      split(line, f, "\t"); known[f[1]]=1
    }
  }
  /^#/ || NF == 0 { next }
  NF != 3 { print "bad binding field count at line " NR > "/dev/stderr"; bad=1 }
  !($1 in known) { print "unknown binding concept " $1 > "/dev/stderr"; bad=1 }
  $2 == "" || $3 == "" { print "empty binding at line " NR > "/dev/stderr"; bad=1 }
  END { if (bad) exit 1 }
' "$BINDINGS" || fail "binding schema validation failed"

required_lane_fields=(
  Semantic-Lane-ID Concept-IDs Intent-Preserved Transformation Types-Changed
  Effects-Changed IR-Changed Claims-Introduced Claims-Forbidden Assumptions
  Write-Set Positive-Witness Negative-Witness Acceptance-Gate
  Authoritative-Only-If
)
for field in "${required_lane_fields[@]}"; do
  grep -Fq "$field:" "$CONTRACT" || fail "lane contract missing field $field"
done

bash -n "$SCANNER" || fail "scanner syntax check failed"
bash "$SCANNER" --current-only --no-processes >"$TMP_DIR/status.log" || {
  cat "$TMP_DIR/status.log" >&2
  fail "current-only scanner execution failed"
}
grep -Fq '== Concept Registry ==' "$TMP_DIR/status.log" || fail "scanner omitted registry section"
grep -Fq '== Dirty Semantic Writers ==' "$TMP_DIR/status.log" || fail "scanner omitted writer section"
grep -Fq '== Dirty Worktree Activity ==' "$TMP_DIR/status.log" || fail "scanner omitted activity classification"
grep -Fq 'exact_path_collisions=0' "$TMP_DIR/status.log" || {
  cat "$TMP_DIR/status.log" >&2
  fail "current worktree produced a false path collision"
}

echo '[semantic-coordination] PASS: registry, bindings, lane contract, and read-only scanner are coherent'
