#!/usr/bin/env bash
# Lane A1 gate: bin/madaros must not report success when the raw compiler fails.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-a1.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

set +e
"$ROOT_DIR/bin/madaros" build "$WORK/does-not-exist.sio" -o "$WORK/out.elf" >"$WORK/log.txt" 2>&1
rc=$?
set -e
if [[ $rc -eq 0 ]]; then
  echo "FAIL: madaros build of missing source returned 0" >&2
  cat "$WORK/log.txt" >&2
  exit 1
fi
if [[ -s "$WORK/out.elf" ]]; then
  echo "FAIL: madaros left an artifact after a failed build" >&2
  exit 1
fi
if ! grep -q 'MADAROS_STACK_KB' "$ROOT_DIR/bin/madaros"; then
  echo "FAIL: MADAROS_STACK_KB not present in launcher" >&2
  exit 1
fi
if ! grep -Fq 'MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"' "$ROOT_DIR/bin/madaros"; then
  echo "FAIL: Madaros default stack reservation must remain 524288 KiB" >&2
  exit 1
fi
# Must not have unconditional unlimited without the named reservation branch.
if grep -n 'ulimit -s unlimited' "$ROOT_DIR/bin/madaros" | grep -v MADAROS_STACK_KB >/dev/null 2>&1; then
  # Allowed only inside MADAROS_STACK_KB==0 branch — check that unlimited is gated.
  if ! grep -A2 'MADAROS_STACK_KB' "$ROOT_DIR/bin/madaros" | grep -q 'unlimited'; then
    echo "FAIL: unlimited stack not gated on MADAROS_STACK_KB=0" >&2
    exit 1
  fi
fi
if ! grep -q 'compiler exited with status' "$ROOT_DIR/bin/madaros"; then
  echo "FAIL: A1 exit-status message missing from launcher" >&2
  exit 1
fi
echo "MADAROS_LAUNCHER_EXIT_STATUS_GATE_PASS"
