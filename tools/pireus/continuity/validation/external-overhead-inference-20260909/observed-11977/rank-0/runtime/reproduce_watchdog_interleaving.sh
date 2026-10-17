#!/usr/bin/env bash
# Diagnostic: simulate a grant commit after the watchdog sampled FENCED.
# This passes only when the stale-enforcement interleaving is reproduced.
set -euo pipefail
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/var/lib/pireus-spark-pair"
export NODE_NAME=spark-3c59 PIREUS_HOST_ROOT="$work" PIREUS_HOST_FENCE_LIBRARY_MODE=1
source "${1:?path to extracted frozen host-fence script required}"
printf 'mode=FENCED\n' > "$GRANT_FILE"
grant_valid() { return 1; }
device_barrier_attached() {
  # Concurrent commit completes after the watchdog chose the FENCED branch.
  printf 'mode=SLURM\n' > "$GRANT_FILE"
  return 0
}
enforce_fenced_compute_state() {
  [[ "$(grant_field mode)" == SLURM ]]
  printf 'STALE_FENCED_ENFORCEMENT_AFTER_SLURM_COMMIT\n'
  printf 'mode=FENCED\n' > "$GRANT_FILE"
}
legacy_gpu_inventory_exact() { return 0; }
known_gpu_services_quiesced() { return 0; }
managed_gpu_restarts_blocked() { return 0; }
active_docker_gpu_claims_zero() { return 0; }
gpu_consumer_set_exact() { return 0; }
managed_gpu_cgroups_empty() { return 0; }
live_memory_floor_met() { return 0; }
protected_resources_unchanged() { return 0; }
active_enforcement_cycle
[[ "$(grant_field mode)" == FENCED ]]
printf 'PIREUS_WATCHDOG_INTERLEAVING_REPRODUCED simulation=1 live_causality_proven=0\n'
