#!/usr/bin/env bash
# Keep the required Contracts check successful only when both shards succeeded.
set -euo pipefail
if [[ $# != 2 || "$1" != success || "$2" != success ]]; then
  printf "CONTRACTS_SHARDS_FAIL core=%s ontology=%s\n" "${1:-missing}" "${2:-missing}" >&2
  exit 1
fi
printf "CONTRACTS_SHARDS_PASS core=success ontology=success\n"
