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
# Takes a single exclusive advisory lock on a repo-wide lockfile, then execs the
# build command. Linux uses flock(1); macOS falls back to Python's fcntl.flock.
# Concurrent heavy builds queue instead of stampeding the CPU, so the load never
# spikes high enough to trip the probe. Cheap `souc check` calls do NOT need this
# wrapper — only full self-compiles / bundle checks.
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

LOCK="${SOUNIO_BUILD_LOCK:-/tmp/sounio-souc-build.lock}"

# Counting semaphore. SOUNIO_BUILD_SLOTS=N allows N heavy builds at once; each
# build pins ONE core, so N=4 on the 64-core pod keeps load ~4 from builds, far
# below the ~153 that tripped the liveness probe. Default 1 = the old mutex.
# Slot i is the file $LOCK.i; a caller tries each slot non-blocking and, if all
# are busy, blocks on one chosen at random (spreads wakeups across slots).
SLOTS="${SOUNIO_BUILD_SLOTS:-1}"
if ! [[ "$SLOTS" =~ ^[1-9][0-9]*$ ]]; then SLOTS=1; fi

if ! command -v flock >/dev/null 2>&1; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: build locking requires flock(1) or python3 with fcntl support" >&2
    exit 69
  fi
  exec python3 "$(dirname "$0")/souc_build_lock.py" "$LOCK" "$@"
fi

if [[ "$SLOTS" -eq 1 ]]; then
  exec 9>"$LOCK"
  if ! flock -n 9; then
    echo "[souc-build-lock] another heavy build holds the lock; waiting..." >&2
    flock 9
  fi
  echo "[souc-build-lock] acquired ($LOCK); running: $*" >&2
  exec "$@"
fi

got=""
for ((i=1; i<=SLOTS; i++)); do
  exec 9>"$LOCK.$i"
  if flock -n 9; then got="$LOCK.$i"; break; fi
  exec 9>&-
done
if [[ -z "$got" ]]; then
  i=$(( (RANDOM % SLOTS) + 1 ))
  echo "[souc-build-lock] all $SLOTS build slots busy; waiting on slot $i..." >&2
  exec 9>"$LOCK.$i"
  flock 9
  got="$LOCK.$i"
fi
echo "[souc-build-lock] acquired $got (slots=$SLOTS); running: $*" >&2
exec "$@"
