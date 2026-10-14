#!/usr/bin/env bash
# Replay the runtime threshold witnesses with an explicitly source-built driver.
# Source receipt and driver digest must come from the same recorded build.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
if [[ $# != 3 ]]; then
  echo 'usage: knowledge_runtime_threshold_contract_gate.sh DRIVER SOURCE_RECEIPT DRIVER_SHA256' >&2
  exit 2
fi
driver=$(realpath "$1")
receipt=$(realpath "$2")
expected_sha="$3"
[[ -x "$driver" && -f "$receipt" ]] || exit 2
sha256sum -c "$receipt" >/dev/null
observed_sha=$(sha256sum "$driver" | cut -d ' ' -f 1)
[[ "$observed_sha" == "$expected_sha" ]] || { echo 'driver digest mismatch' >&2; exit 2; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
ulimit -s unlimited
ulimit -c 0
failures=0
while IFS=$'\t' read -r name expected; do
  [[ "$name" == name ]] && continue
  elf="$work/$name.elf"
  log="$work/$name.compile.log"
  rc=0
  timeout 60 "$driver" "tests/frontend/threshold-matrix/$name.sio" -o "$elf" >"$log" 2>&1 || rc=$?
  if [[ "$expected" == compile-fail ]]; then
    if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 139 && ! -e "$elf" ]] && grep -q 'Knowledge runtime threshold cannot be represented safely' "$log"; then
      printf 'PASS %s compile_rc=%s no_elf=1\n' "$name" "$rc"
    else printf 'FAIL %s compile_rc=%s\n' "$name" "$rc"; failures=$((failures+1)); fi
    continue
  fi
  run_rc=999
  if [[ $rc == 0 && -s "$elf" ]]; then chmod +x "$elf"; timeout 20 "$elf" >"$work/$name.run.log" 2>&1 && run_rc=0 || run_rc=$?; fi
  if [[ $rc == 0 && "$run_rc" == "$expected" ]] && grep -Eq 'row_stream_payload_guards=1 count=[1-9]' "$log"; then
    printf 'PASS %s runtime_rc=%s emitted_guard=1\n' "$name" "$run_rc"
  else printf 'FAIL %s compile_rc=%s runtime_rc=%s expected=%s\n' "$name" "$rc" "$run_rc" "$expected"; failures=$((failures+1)); fi
done < tests/frontend/threshold-matrix/matrix.tsv
[[ $failures == 0 ]]
