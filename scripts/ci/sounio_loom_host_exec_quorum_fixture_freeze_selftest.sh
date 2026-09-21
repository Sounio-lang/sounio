#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/host_exec_quorum_fixture.freeze.v1
EVIDENCE=tools/loom/evidence/loom-host-exec-quorum-fixture-v1-20260828.txt

fail() {
  printf 'sounio-loom-host-exec-quorum-fixture-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || fail "frozen path is absent: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen path hash drifted: $path"
}

require_line() {
  local path="$1" value="$2"
  grep -Fxq "$value" "$path" || fail "required line is absent: $value"
}

expect_hash tools/loom/GARDEN_HOST_EXEC_QUORUM_V1.md 67aecd9785a1aa6e95f80cac41f7344bbf7a1fc0eb6c27e07ec378986fc5a7a0
expect_hash tools/loom/host_exec_quorum_fixture_main.sio b6b4ab0a4e623c742e44662e484982e483efd84e716d25cd60b8fa39af5a514a
expect_hash scripts/dev/build_sounio_loom_host_exec_quorum_fixture.sh 885d4f15538ce6a7f07d2383cb43cd5233f6a6236cded5bfdb9def1ed749db2e
expect_hash scripts/ci/sounio_loom_host_exec_quorum_fixture_selftest.sh a938234f2c6aa26cb38a31b436a6fa11c724cb97589a55b4af53d7557b2f62bb
expect_hash tools/loom/kernel_exec_grant_cell_authority.freeze.v1 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051
expect_hash "$EVIDENCE" 175e9f0e030ced98a34d6c430b17748aefcb615924601b980f225463467b3d28

require_line "$MANIFEST" 'schema=loom-host-exec-quorum-fixture-freeze-v1'
require_line "$MANIFEST" 'stage=SEMANTICS_FROZEN'
require_line "$MANIFEST" 'producing_language=Sounio'
require_line "$MANIFEST" 'language_role=SEMANTIC_FIXTURE_PRODUCER'
require_line "$MANIFEST" 'semantic_authority=Sounio'
require_line "$MANIFEST" 'action=9030'
require_line "$MANIFEST" 'executable_sha256=ff1f160a337f206ef4f2691486e499b5cd91fb9cb15809099b88f6cbe22e3649'
require_line "$MANIFEST" 'bundle_sha256=523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe'
require_line "$MANIFEST" 'expected_results_encoded_in_shell=false'
require_line "$MANIFEST" 'python_executable_invoked=false'
require_line "$MANIFEST" 'material_grant=false'
require_line "$MANIFEST" 'material_execution=false'
require_line "$MANIFEST" 'barrier_release=false'
require_line "$MANIFEST" 'exec_attached=false'
require_line "$MANIFEST" 'parity_open=false'
require_line "$MANIFEST" 'claim_ready=false'
require_line "$MANIFEST" 'evidence_sha256=175e9f0e030ced98a34d6c430b17748aefcb615924601b980f225463467b3d28'

require_line "$EVIDENCE" 'expected_results_source=tools/loom/kernel_exec_grant_cell_authority.freeze.v1'
require_line "$EVIDENCE" 'expected_results_encoded_in_shell=false'
require_line "$EVIDENCE" 'python_executable_invoked=false'
require_line "$EVIDENCE" 'deterministic_rebuild=true'
require_line "$EVIDENCE" 'material_grant=false'
require_line "$EVIDENCE" 'material_execution=false'
require_line "$EVIDENCE" 'barrier_release=false'
require_line "$EVIDENCE" 'exec_attached=false'
require_line "$EVIDENCE" 'parity_open=false'
require_line "$EVIDENCE" 'claim_ready=false'

result="$(bash scripts/ci/sounio_loom_host_exec_quorum_fixture_selftest.sh)"
[[ "$result" == sounio-loom-host-exec-quorum-fixture-selftest:\ PASS* ]] ||
  fail 'source-fresh fixture gate failed'
[[ "$result" == *'positive=issue+consume+close treatment=current python_control=refused python_executed=false'* ]] ||
  fail 'fixture decision classes drifted'
[[ "$result" == *'shell_expected_results=false runtime_dependencies=clean material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'fixture gate promoted beyond evidence'

printf 'sounio-loom-host-exec-quorum-fixture-freeze-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER action=9030 fixture_manifest_sha256=%s executable_sha256=ff1f160a337f206ef4f2691486e499b5cd91fb9cb15809099b88f6cbe22e3649 bundle_sha256=523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe positive=issue+consume+close treatment=current python_control=refused shell_expected_results=false material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
