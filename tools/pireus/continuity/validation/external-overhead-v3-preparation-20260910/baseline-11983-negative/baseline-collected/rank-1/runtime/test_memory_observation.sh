#!/usr/bin/env bash
set -euo pipefail
ROOT="${PIREUS_MEMORY_TEST_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
source "$ROOT/tests/fixtures/spark_pair_arbiter/k8s_backend_fenced_unit.sh" "$ROOT/scripts/dev/spark_pair_arbiter_k8s_backend.sh" "$work"
UNIT_REPORT0+=" memory=1 watchdog=1 reported_monotonic=100 watchdog_monotonic=100"
UNIT_REPORT1+=" memory=1 watchdog=1 reported_monotonic=100 watchdog_monotonic=100"
mask="$(host_mask_from_facts "$(cat "$PERSISTED")" 'FreeMem=815 FreeMem=1' holder 7)"
(( (mask & 4096) != 0 ))
(( (mask & 32768) != 0 ))
good0="$UNIT_REPORT0";good1="$UNIT_REPORT1"
UNIT_REPORT0="${good0/memory=1/memory=0}"
mask="$(host_mask_from_facts "$(cat "$PERSISTED")" 'FreeMem=999999 FreeMem=999999' holder 7)"
(( (mask & 4096) == 0 ))
UNIT_REPORT0="$good0";UNIT_REPORT1="${good1/memory=1/memory=0}"
mask="$(host_mask_from_facts "$(cat "$PERSISTED")" '' holder 7)"
(( (mask & 4096) == 0 ))
UNIT_REPORT1="${good1/ memory=1/}"
mask="$(host_mask_from_facts "$(cat "$PERSISTED")" '' holder 7)"
(( (mask & 4096) == 0 ))
UNIT_REPORT1="${good1/watchdog_monotonic=100/watchdog_monotonic=1}"
mask="$(host_mask_from_facts "$(cat "$PERSISTED")" '' holder 7)"
(( (mask & 32768) == 0 ))
printf 'PIREUS_LIVE_MEMORY_OBSERVATION_PASS stale_slurm_ignored=1 both_host_floors_required=1 unknown_refused=1 stale_watchdog_refused=1\n'
