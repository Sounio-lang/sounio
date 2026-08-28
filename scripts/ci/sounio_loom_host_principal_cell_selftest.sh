#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-principal-cell.XXXXXX")"
BINARY_ONE="$TEST_ROOT/loom-host-principal-cell-one"
BINARY_TWO="$TEST_ROOT/loom-host-principal-cell-two"
SOURCE="$ROOT_DIR/tools/loom/src/loom_host_principal_cell.cpp"
CONTRACT="$ROOT_DIR/tools/loom/HOST_EXEC_GRANT_PRINCIPAL_CELL_V1.md"
ACTION_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
ACTION_MANIFEST_SHA256=8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-host-principal-cell-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$SOURCE" "$CONTRACT" "$ACTION_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
[[ "$(sha256sum "$ACTION_MANIFEST" | cut -d ' ' -f 1)" == "$ACTION_MANIFEST_SHA256" ]] ||
  fail 'frozen Sounio action 9030 manifest drifted'
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_exec_grant_cell_authority_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_HOST_PRINCIPAL_CELL_OUTPUT="$BINARY_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_host_principal_cell.sh" >/dev/null
SOUNIO_LOOM_HOST_PRINCIPAL_CELL_OUTPUT="$BINARY_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_host_principal_cell.sh" >/dev/null
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two C++20 PrincipalCell builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 ]] || fail 'PrincipalCell binary mode is not 0755'
[[ ! -u "$BINARY_ONE" && ! -g "$BINARY_ONE" ]] || fail 'PrincipalCell binary acquired set-id privilege'

dependencies="$(ldd "$BINARY_ONE")"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'PrincipalCell has a forbidden Python or Rust runtime dependency'
fi
if strings "$BINARY_ONE" | grep -Eq 'SOUNIO_KERNEL_EXEC_GRANT_CELL_(ALLOW|DENY)'; then
  fail 'C++20 PrincipalCell copied Sounio semantic result strings'
fi

selftest="$($BINARY_ONE --selftest)"
[[ "$selftest" == \
  'LOOM_HOST_PRINCIPAL_CELL_SELFTEST PASS language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9030 parser=bounded digest=verified proc_identity=read hostile_classifier=closed launch_open=false material_grant=false exec_attached=false' ]] ||
  fail "native selftest failed: $selftest"

grep -Fq 'DynamicUser=yes' "$CONTRACT" || fail 'contract omits DynamicUser boundary'
grep -Fq 'pidfd_getfd(pidfd, 1)' "$CONTRACT" || fail 'contract omits copied-pidfd extraction attack'
grep -Fq 'grant_extinction=false' "$CONTRACT" || fail 'contract launders process absence as grant extinction'
grep -Fq 'same_uid_peer_isolation=false' "$CONTRACT" || fail 'contract launders cross-UID isolation'

printf 'sounio-loom-host-principal-cell-selftest: PASS semantic_authority=Sounio action=9030 material_producer=C++20 material_role=MATERIAL_PARITY transitory=true action_manifest_sha256=%s source_sha256=%s binary_sha256=%s rebuilds=2 semantic_results_encoded=false runtime_dependencies=clean contract=verified launch_open=false kernel_distinct_principal_candidate=unmeasured material_grant=false grant_extinction=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n' \
  "$ACTION_MANIFEST_SHA256" "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"
