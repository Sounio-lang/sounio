#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT/bin/souc-linux-x86_64"

# Verify binary exists
if [[ ! -x "$SOUC" ]]; then
    echo "ERROR: binary not found: $SOUC" >&2
    exit 1
fi

# Gather 50 random CURIEs from the ontology kernel
mapfile -t curies < <("$SOUC" ontology list "" 2>/dev/null | sed 's/ | .*//' | shuf -n 50)

if [[ ${#curies[@]} -eq 0 ]]; then
    echo "ERROR: failed to collect CURIEs" >&2
    exit 1
fi

# Warm up: resolve once to ensure index is loaded
"$SOUC" ontology resolve "${curies[0]}" >/dev/null 2>&1

# Benchmark: resolve all 50 CURIEs via the hash index (fast path)
TIMEFORMAT='%R'
total_time_s=$({ time {
    for curie in "${curies[@]}"; do
        "$SOUC" ontology resolve "$curie" >/dev/null 2>&1
    done
}; } 2>&1)

# Report
awk -v total="$total_time_s" -v n="${#curies[@]}" 'BEGIN {
    printf "=== ontology hash benchmark ===\n"
    printf "Terms: 1008\n"
    printf "Resolutions: %d\n", n
    printf "Total time: %.3fs\n", total
    printf "Average per resolve: %.3fms\n", (total / n) * 1000
}'
