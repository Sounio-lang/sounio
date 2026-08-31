#!/usr/bin/env bash
# Lane A1 gate: bin/madaros must not report success when the raw compiler fails.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-a1.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

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
# The A1 exit-status message occurs twice in bin/madaros -- once in the GPU
# backend branch and once in the native branch -- with byte-identical text, so
# a bare `grep -q 'compiler exited with status'` was satisfied by either and
# could not notice an edit to the one that actually runs. bin/madaros IS the
# source (a checked-in shell launcher), so the two sites are distinguishable,
# but only by the discard message that follows each. Pin both by that context,
# and require the reachable one to be exercised for real below.
mapfile -t MSG_LINES < <(
  grep -nF 'error: madaros build: compiler exited with status' "$ROOT_DIR/bin/madaros" \
    | cut -d: -f1
)
if [[ "${#MSG_LINES[@]}" -ne 2 ]]; then
  echo "FAIL: expected 2 A1 exit-status messages in launcher, found ${#MSG_LINES[@]}" >&2
  exit 1
fi
gpu_branch=0
native_branch=0
for n in "${MSG_LINES[@]}"; do
  window="$(sed -n "$((n + 1)),$((n + 3))p" "$ROOT_DIR/bin/madaros")"
  if grep -qF 'discarding partial GPU artifact at' <<<"$window"; then
    gpu_branch=$((gpu_branch + 1))
  elif grep -qF 'discarding partial ELF at' <<<"$window"; then
    native_branch=$((native_branch + 1))
  else
    echo "FAIL: A1 exit-status message at bin/madaros:$n is in neither the GPU nor the native branch" >&2
    exit 1
  fi
done
if [[ "$gpu_branch" -ne 1 || "$native_branch" -ne 1 ]]; then
  echo "FAIL: A1 exit-status message must appear once per branch (gpu=$gpu_branch native=$native_branch)" >&2
  exit 1
fi

# Behavioural check on the occurrence that actually executes. The missing-source
# probe above never reaches the compiler -- bin/madaros rejects it with a usage
# error (rc=2) before RAW_MADAROS is invoked -- so it exercises neither site. A
# source that exists but does not compile drives the native branch through the
# real message.
BAD_SRC="$WORK/a1-bad.sio"
printf 'fn main() {\n  let x = \n}\n' >"$BAD_SRC"
set +e
"$ROOT_DIR/bin/madaros" build "$BAD_SRC" -o "$WORK/bad.elf" >"$WORK/bad-log.txt" 2>&1
bad_rc=$?
set -e
if [[ $bad_rc -eq 0 ]]; then
  echo "FAIL: madaros build of an ill-formed source returned 0" >&2
  cat "$WORK/bad-log.txt" >&2
  exit 1
fi
if ! grep -qF 'madaros build: compiler exited with status' "$WORK/bad-log.txt"; then
  echo "FAIL: native build failure did not emit the A1 exit-status message" >&2
  cat "$WORK/bad-log.txt" >&2
  exit 1
fi
if [[ -s "$WORK/bad.elf" ]]; then
  echo "FAIL: madaros left an artifact after a failed native build" >&2
  exit 1
fi
echo "MADAROS_LAUNCHER_EXIT_STATUS_GATE_PASS"
