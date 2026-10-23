#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DECISION="$ROOT_DIR/scripts/ci/contracts_shard_decision.sh"
bash "$DECISION" success success >/dev/null
for result in failure cancelled skipped timed_out neutral unknown ""; do
  if bash "$DECISION" "$result" success >/dev/null 2>&1; then
    echo "accepted unsuccessful core: $result" >&2; exit 1
  fi
  if bash "$DECISION" success "$result" >/dev/null 2>&1; then
    echo "accepted unsuccessful ontology: $result" >&2; exit 1
  fi
done
for args in 0 1 3; do
  set --
  for ((i=0; i<args; i++)); do set -- "$@" success; done
  if bash "$DECISION" "$@" >/dev/null 2>&1; then
    echo "accepted wrong result count: $args" >&2; exit 1
  fi
done
echo CONTRACTS_SHARD_DECISION_SELFTEST_PASS
