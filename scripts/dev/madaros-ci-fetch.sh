#!/usr/bin/env bash
# scripts/dev/madaros-ci-fetch.sh — build Madaros for a ref on GitHub's runners
# and download the ELF, instead of building on the shared workspace pod.
#
# The madaros-prebuilt-refresh workflow already accepts any ref via
# workflow_dispatch and always uploads the built binary as artifact
# `madaros-built` (retention 5 days). For non-main refs it never commits back.
# So a verification build costs the pod nothing.
#
# usage: scripts/dev/madaros-ci-fetch.sh <ref> [out]     (needs gh, authenticated)
set -euo pipefail
REF="${1:?ref (branch or sha)}"
OUT="${2:-artifacts/self-hosted/madaros}"
WF="madaros-prebuilt-refresh.yml"

gh workflow run "$WF" -f "ref=$REF" >/dev/null
echo "→ dispatched $WF for $REF; waiting for the run to appear..."
sleep 8
RUN_ID=""
for _ in $(seq 1 20); do
  RUN_ID="$(gh run list --workflow "$WF" --event workflow_dispatch --limit 5 \
            --json databaseId,headBranch,status,createdAt \
            --jq "[.[] | select(.headBranch==\"$REF\")] | sort_by(.createdAt) | last | .databaseId // empty")"
  [[ -n "$RUN_ID" ]] && break
  sleep 5
done
[[ -n "$RUN_ID" ]] || { echo "error: run not found for $REF" >&2; exit 1; }
echo "→ run $RUN_ID: $(gh run view "$RUN_ID" --json url --jq .url)"
gh run watch "$RUN_ID" --exit-status || echo "warning: run reported failure; artifact may still exist (upload is if: always())" >&2
TMP="$(mktemp -d)"
gh run download "$RUN_ID" -n madaros-built -D "$TMP"
mkdir -p "$(dirname "$OUT")"
cp -f "$TMP/madaros" "$OUT"; chmod +x "$OUT"; rm -rf "$TMP"
echo "Madaros ready: $OUT ($(stat -c%s "$OUT") bytes) — built on GitHub from $REF"
