#!/usr/bin/env bash
# souc-build-lock.sh — global serializer for expensive self-hosted compiler builds.
#
# WHY THIS EXISTS
# ---------------
# The workspace pod was recycled twice on 2026-05-29 by the k8s liveness probe.
# Root cause was NOT OOM (peak 10/64 GB) and NOT disk (66%): it was CPU
# oversubscription. Multiple concurrent agent sessions on the shared
# /workspace/sounio checkout each launched a full `souc main.sio` bundle build
# (each pins a core at 100% for minutes). The 15-min load average hit ~153 on
# 64 cores; the pod's liveness probe timed out and k8s evicted/rescheduled it.
#
# WHAT THIS DOES
# --------------
# Takes a single exclusive flock on a repo-wide lockfile, then execs the build
# command. Concurrent heavy builds queue instead of stampeding the CPU, so the
# load never spikes high enough to trip the probe. Cheap `souc check` calls do
# NOT need this wrapper — only full self-compiles / bundle checks.
#
# USAGE
# -----
#   scripts/dev/souc-build-lock.sh ./bin/souc self-hosted/compiler/main.sio /tmp/out.elf
#   scripts/dev/souc-build-lock.sh make build
#
# Override the lock path with SOUNIO_BUILD_LOCK (must be shared across sessions,
# so keep it on a path all sessions see — default /tmp is per-pod, which is what
# we want: one build at a time inside one workspace pod).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <build command...>" >&2
  exit 64
fi

# OOM guard — two layers when not inside a SLURM job:
#
# 1. HARD BLOCK: main.sio full-self-host builds and --native-*-compile modes peak
#    at 26+ GB RSS and will OOM-kill or evict the workspace container. These must
#    run under SLURM (cpu-ops partition, --mem bound). Block them here so they
#    fail immediately with a clear message rather than silently growing to 26+ GB.
#
# 2. SOFT CAP: all other local builds are capped at 12 GiB virtual memory so any
#    unexpectedly large build dies with ENOMEM instead of OOMKilling the container.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  for _arg in "$@"; do
    case "${_arg}" in
      *main.sio|--native-compile|--native-v2-compile)
        echo "[souc-build-lock] BLOCKED: '${_arg}' peaks at 26+ GB RSS — must run via SLURM." >&2
        echo "[souc-build-lock] Use: WORKTREE=/workspace/sounio-ir bash slurm-jobs/ir-heap-indirect/submit-v2test.sh" >&2
        exit 1
        ;;
    esac
  done
  ulimit -v 12582912  # 12 GiB in KiB — safe cap for non-SLURM local builds
fi

LOCK="${SOUNIO_BUILD_LOCK:-/tmp/sounio-souc-build.lock}"
exec 9>"$LOCK"

if ! flock -n 9; then
  echo "[souc-build-lock] another heavy build holds the lock; waiting…" >&2
  flock 9
fi
echo "[souc-build-lock] acquired ($LOCK); running: $*" >&2
exec "$@"
