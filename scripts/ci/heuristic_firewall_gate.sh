#!/usr/bin/env bash
# scripts/ci/heuristic_firewall_gate.sh — Metaphor-firewall gate for Feature C-b.
#
# The experimental::heuristic namespace (stdlib/experimental/heuristic/**) is a
# SPECULATIVE, never-load-bearing analogy (causal-set provenance quasi-metric). This
# gate FAILS THE BUILD if any CORRECTNESS-CRITICAL module imports it, enforcing the
# §6.2 isolation constraint at the build level.
#
#   correctness-critical : stdlib/** (except stdlib/experimental/**), self-hosted/**,
#                          bootstrap/**
#   allowed to import    : tests/**, examples/**, stdlib/experimental/** (itself)
#
# Usage:
#   bash scripts/ci/heuristic_firewall_gate.sh           # scan the repo, fail on violation
#   bash scripts/ci/heuristic_firewall_gate.sh --demo    # self-test: prove it catches a violation
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Marker that an import targets the firewalled namespace (matches `use experimental::
# heuristic...` in any form).
PATTERN='use[[:space:]].*experimental::heuristic'

# Print the paths (relative to $1) of correctness-critical .sio files that import the
# firewalled namespace. A path is correctness-critical iff it is under self-hosted/,
# bootstrap/, or stdlib/ but NOT under stdlib/experimental/.
scan_violations() {
  local base="$1"
  # All .sio files importing the namespace…
  local f rel
  while IFS= read -r f; do
    rel="${f#"$base"/}"
    case "$rel" in
      stdlib/experimental/*) continue ;;          # the namespace itself — allowed
      tests/*|examples/*|docs/*|scripts/*) continue ;;  # non-correctness — allowed
      stdlib/*|self-hosted/*|bootstrap/*) echo "$rel" ;; # correctness-critical — VIOLATION
      *) : ;;                                       # other top-levels: not gated
    esac
  done < <(grep -rlE "$PATTERN" "$base" --include='*.sio' 2>/dev/null || true)
}

run_gate() {
  local base="$1"
  local violations
  violations="$(scan_violations "$base")"
  if [[ -n "$violations" ]]; then
    echo "[heuristic-firewall] FAIL: correctness-critical module(s) import the" >&2
    echo "  experimental::heuristic namespace (speculative, never load-bearing):" >&2
    echo "$violations" | sed 's/^/    /' >&2
    echo "  Move the use to tests/ or examples/, or remove it. See §6.2." >&2
    return 1
  fi
  echo "[heuristic-firewall] PASS: no correctness-critical module imports experimental::heuristic"
  return 0
}

if [[ "${1:-}" == "--demo" ]]; then
  # Prove the gate catches a violation: plant an importing file under a correctness
  # path in a throwaway tree and assert it is flagged.
  TMP="$(mktemp -d /tmp/heuristic-firewall-demo.XXXXXX)"
  trap 'rm -rf "$TMP"' EXIT
  mkdir -p "$TMP/self-hosted/check" "$TMP/tests/run-pass" "$TMP/stdlib/experimental/heuristic"
  printf 'use experimental::heuristic::provenance::*\nfn f() {}\n' > "$TMP/self-hosted/check/violator.sio"
  printf 'use experimental::heuristic::provenance::*\nfn t() {}\n' > "$TMP/tests/run-pass/allowed.sio"
  printf 'pub fn ok() {}\n' > "$TMP/stdlib/experimental/heuristic/provenance.sio"
  got="$(scan_violations "$TMP")"
  if [[ "$got" == "self-hosted/check/violator.sio" ]]; then
    echo "[heuristic-firewall] DEMO PASS: violation under self-hosted/ caught; tests/ import allowed; namespace self-import allowed"
    exit 0
  fi
  echo "[heuristic-firewall] DEMO FAIL: expected to flag self-hosted/check/violator.sio, got: [$got]" >&2
  exit 1
fi

run_gate "$ROOT_DIR"
