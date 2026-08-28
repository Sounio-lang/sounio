#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-principal-cell-barrier.XXXXXX")"
BINARY_ONE="$TEST_ROOT/barrier-one"
BINARY_TWO="$TEST_ROOT/barrier-two"
SOURCE="$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier.cpp"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PRINCIPAL_CELL_BARRIER_V1.md"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-principal-cell-barrier-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$SOURCE" "$GARDEN"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done

SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_OUTPUT="$BINARY_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_principal_cell_barrier.sh" >/dev/null
SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_OUTPUT="$BINARY_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_principal_cell_barrier.sh" >/dev/null
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two barrier builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 ]] || fail 'barrier binary mode is not 0755'
[[ ! -u "$BINARY_ONE" && ! -g "$BINARY_ONE" ]] || fail 'barrier binary acquired set-id privilege'

dependencies="$(ldd "$BINARY_ONE")"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'barrier has a prohibited Python or Rust runtime dependency'
fi
if strings "$BINARY_ONE" | grep -Eq 'SOUNIO_KERNEL_EXEC_GRANT_CELL_(ALLOW|DENY)'; then
  fail 'barrier copied Sounio semantic result strings'
fi

selftest="$($BINARY_ONE --selftest)"
[[ "$selftest" == \
  'LOOM_PRINCIPAL_CELL_BARRIER_SELFTEST PASS language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio treatment=CLOSED sabotage=OPEN causal_rule=descriptor-write eof=closed timeout=closed wrong_generation=closed truncated=closed oversized=closed duplicate=closed descriptor_absent=closed open_sentinels=1 command_surface=false material_grant=false material_execution=false launch_open=false exec_attached=false' ]] ||
  fail "native barrier selftest failed: $selftest"

grep -Fq 'Treatment: parent closes the release descriptor without writing.' "$GARDEN" ||
  fail 'Garden did not preregister treatment'
grep -Fq 'Isolated sabotage:' "$GARDEN" || fail 'Garden did not preregister sabotage'
grep -Fq 'No single object can cross the threshold.' "$GARDEN" ||
  fail 'Garden omitted the three-object quorum'

unknown_refusal="$($BINARY_ONE --release 2>&1 || true)"
[[ "$unknown_refusal" == 'loom-principal-cell-barrier: REFUSE reason=only --selftest is available' ]] ||
  fail 'binary exposed an unregistered public mode'

printf 'sounio-loom-principal-cell-barrier-selftest: PASS semantic_authority=Sounio material_producer=C++20 material_role=MATERIAL_PARITY transitory=true source_sha256=%s binary_sha256=%s rebuilds=2 treatment=closed causal_sabotage=open descriptor_only=true user_command_surface=false semantic_results_encoded=false negatives=6 runtime_dependencies=clean material_threshold_measured=true material_grant=false material_execution=false launch_open=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"

