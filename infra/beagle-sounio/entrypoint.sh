#!/usr/bin/env bash
# beagle-sounio entrypoint
# ========================
# Wires the OrangeFS mount into Sounio's expected filesystem layout, then
# execs the user's command. Runs unprivileged as UID 1000 (agent).
#
# Behavior:
#   - If /orangefs/training/sounio/abide-data exists (mounted by k8s), the
#     path /app/artifacts/research/abide is symlinked to it. This lets the
#     unmodified abide_preprocess.py / abide_fetch.py write directly to
#     OrangeFS without --output plumbing.
#   - If /orangefs is not a mount, the container runs in standalone mode and
#     the default /app/artifacts/research/abide is used (writes land in the
#     container's writable layer; suitable for local docker run tests).

set -euo pipefail

ABIDE_MOUNT="${BEAGLE_ABIDE_MOUNT:-/orangefs/training/sounio/abide-data}"
ABIDE_EXPECTED="/app/artifacts/research/abide"

# Only rewire when the OrangeFS parent actually exists — avoids dangling
# symlinks when someone runs the image without OrangeFS bound.
if [ -d "/orangefs" ] && [ -w "/orangefs" ]; then
    mkdir -p "$ABIDE_MOUNT" 2>/dev/null || true
    if [ -d "$ABIDE_MOUNT" ]; then
        mkdir -p "$(dirname "$ABIDE_EXPECTED")"
        # Remove any prior placeholder before symlinking. We only clobber
        # directories we own (owned by UID 1000 from image build).
        if [ -e "$ABIDE_EXPECTED" ] && [ ! -L "$ABIDE_EXPECTED" ]; then
            rmdir "$ABIDE_EXPECTED" 2>/dev/null || true
        fi
        if [ ! -L "$ABIDE_EXPECTED" ]; then
            ln -s "$ABIDE_MOUNT" "$ABIDE_EXPECTED"
        fi
    fi
fi

# If the command expects to be run from /app (abide_fetch.py computes ROOT
# from __file__ so it's path-independent, but preprocess.py uses cwd-relative
# paths), default WORKDIR to /app unless the user explicitly set one.
if [ -z "${BEAGLE_NO_CD:-}" ] && [ "$(pwd)" = "/workspace" ]; then
    cd /app
fi

exec "$@"
