#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_principal_cell.sh"
SOURCE="$ROOT_DIR/tools/loom/src/loom_host_exec_quorum_principal_cell.cpp"
BARRIER="$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier.cpp"

fail() {
  printf 'sounio-loom-host-exec-quorum-principal-cell-selftest: FAIL reason=%s material_grant=false material_execution=false launch_open=false\n' "$*" >&2
  exit 1
}

for input in "$BUILDER" "$SOURCE" "$BARRIER"; do
  [[ -f "$input" && ! -L "$input" ]] || fail "required input is absent or linked: $input"
done
for tool in sha256sum cmp rg ldd mktemp chmod; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-exec-quorum-principal-cell-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
BINARY_ONE="$WORK/principal-cell-one"
BINARY_TWO="$WORK/principal-cell-two"

SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_OUTPUT="$BINARY_ONE" bash "$BUILDER" >/dev/null
SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_OUTPUT="$BINARY_TWO" bash "$BUILDER" >/dev/null
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two source-fresh builds differ'

selftest="$($BINARY_ONE --selftest)"
[[ "$selftest" == 'LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_SELFTEST PASS semantic_authority=Sounio material_language=C++20 material_role=MATERIAL_PARITY transitory=true frozen_barrier_reused=true arm_exact=true arm_authority=false read_ahead=false dynamic_user_required=true inherited_descriptor=true material_grant=false material_execution=false launch_open=false' ]] ||
  fail "source-fresh selftest diverged: $selftest"

set +e
public_output="$($BINARY_ONE --internal-host-exec-quorum 2>&1)"
public_status=$?
unknown_output="$($BINARY_ONE --unknown 2>&1)"
unknown_status=$?
set -e
[[ $public_status -eq 70 && "$public_output" == *'missing host PrincipalCell environment'* ]] ||
  fail 'internal mode did not refuse without systemd-bound environment'
[[ $unknown_status -eq 70 && "$unknown_output" == *'host PrincipalCell has no public material mode'* ]] ||
  fail 'unknown public mode was not refused'

if rg -n 'DENY49[1-9]|DENY500|DENY501|SOUNIO_KERNEL_INVOCATION_CELL_ALLOW|code=491|expected_result' "$SOURCE"; then
  fail 'host material cell encodes a Sounio result oracle'
fi
if rg -n 'python|python3|rustc|cargo' "$SOURCE" "$BUILDER"; then
  fail 'host material cell contains a prohibited language bridge'
fi
if ldd "$BINARY_ONE" | rg -n 'python|libpython|rust'; then
  fail 'host material cell links a prohibited language runtime'
fi

printf 'sounio-loom-host-exec-quorum-principal-cell-selftest: PASS semantic_authority=Sounio material_language=C++20 material_role=MATERIAL_PARITY transitory=true frozen_barrier_reused=true deterministic_rebuild=true arm_exact=true arm_authority=false read_ahead=false public_protocol=closed expected_results_encoded=false python_runtime=false rust_runtime=false material_grant=false material_execution=false launch_open=false source_sha256=%s barrier_source_sha256=%s binary_sha256=%s\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BARRIER" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"
