#!/usr/bin/env bash

set -euo pipefail
umask 077

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
CHANGE_ROOT="${1:-}"
RECEIPT="${2:-}"
LOOM="${SOUNIO_LOOM_BIN:-$SOURCE_ROOT/tools/loom/_build/default/src/loom.exe}"

if [[ -z "$CHANGE_ROOT" || -z "$RECEIPT" || $# -ne 2 ]]; then
  printf 'usage: %s CHANGE_ROOT COMMIT_RECEIPT\n' "$0" >&2
  exit 64
fi
if [[ ! -x "$LOOM" ]]; then
  printf 'loom runtime is absent or not executable: %s\n' "$LOOM" >&2
  exit 66
fi

CHANGE_ROOT="$(git -C "$CHANGE_ROOT" rev-parse --show-toplevel)"
RECEIPT="$(realpath "$RECEIPT")"

ci_output="$(
  "$LOOM" change-ci-admit --root "$CHANGE_ROOT" --receipt "$RECEIPT"
)"
[[ "$ci_output" == LOOM_CHANGE_CI_ADMITTED* ]] || {
  printf 'CI receipt consumption returned an invalid response: %s\n' "$ci_output" >&2
  exit 70
}
printf '%s\n' "$ci_output"

claim_output="$(
  "$LOOM" change-claim-ready --root "$CHANGE_ROOT" --receipt "$RECEIPT"
)"
[[ "$claim_output" == LOOM_CHANGE_CLAIM_READY*'claim_ready=true'* ]] || {
  printf 'Sounio claim admission returned an invalid response: %s\n' "$claim_output" >&2
  exit 71
}
printf '%s\n' "$claim_output"
printf '%s\n' \
  'SOUNIO_LOOM_CHANGE_ADMISSION PASS ci_policy=consume-not-reinterpret policy_executed_by_ci=false claim_policy_executed_by=Sounio write_attached=true commit_attached=true ci_attached=true claim_ready=true python_executed=false rust_executed=false'
