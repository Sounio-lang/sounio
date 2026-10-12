#!/usr/bin/env bash
# Shared source-built SCO compiler stack contract.
# 64 MiB matches the measured Foundry gate environment and preserves any
# higher/unlimited runner setting. A runner unable to provide it fails before
# compiler invocations, rather than misreporting stack exhaustion as semantics.
set -euo pipefail
stack_soft_before="$(ulimit -S -s)"
stack_hard_before="$(ulimit -H -s)"
if [[ "$stack_soft_before" =~ ^[0-9]+$ ]] && (( stack_soft_before < 65536 )); then
  ulimit -S -s 65536 || {
    echo "FAIL: SCO gates require at least a 64 MiB stack (soft=${stack_soft_before} KiB hard=${stack_hard_before} KiB)"
    exit 1
  }
fi
stack_soft_after="$(ulimit -S -s)"
if [[ "$stack_soft_after" =~ ^[0-9]+$ ]] && (( stack_soft_after < 65536 )); then
  echo "FAIL: SCO stack remains below 64 MiB (soft=${stack_soft_after} KiB hard=${stack_hard_before} KiB)"
  exit 1
fi
echo "SCO_STACK_LIMIT soft_before_kb=${stack_soft_before} hard_before_kb=${stack_hard_before} soft_after_kb=${stack_soft_after}"
