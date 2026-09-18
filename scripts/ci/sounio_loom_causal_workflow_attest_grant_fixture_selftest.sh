#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-attest-grant.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"

fail() {
  printf 'sounio-loom-causal-workflow-attest-grant-fixture-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local line
  line="$(grep -m1 "^$1=" "$AUTHORITY_MANIFEST")"
  [[ -n "$line" ]] || fail "authority field $1 absent"
  printf '%s' "${line#*=}"
}

hex_u32() {
  local digest="$1" offset
  for ((offset = 0; offset < 64; offset += 8)); do
    printf ' %u' "$((16#${digest:offset:8}))"
  done
}

frame_for() {
  local label="$1" line parents key digest
  line="$(grep -m1 "^FRAME ${label} " "$WORK/bundle")"
  parents="$(grep -m1 '^PARENT_BINDINGS ' "$WORK/bundle")"
  printf '%s %s' "${line#"FRAME $label "}" "${parents#PARENT_BINDINGS }"
  for key in grant_identity command_environment peer_vector transition_journal source_semantics_toolchain result_receipt; do
    digest="$(sed -n "s/^BINDING ${key} //p" "$WORK/bundle")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "binding $key malformed"
    hex_u32 "$digest"
  done
}

for ordinal in one two; do
  SOUNIO_LOOM_CAUSAL_ATTEST_GRANT_OUTPUT="$WORK/fixture-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_attest_grant_fixture.sh" >/dev/null
done
cmp "$WORK/fixture-one" "$WORK/fixture-two" || fail 'fixture build differs'
"$WORK/fixture-one" > "$WORK/bundle"
[[ "$(wc -l < "$WORK/bundle")" == 16 ]] || fail 'bundle line count diverged'
SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$WORK/authority" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" >/dev/null

command="$(sed -n 's/^COMMAND //p' "$WORK/bundle")"
command_sha256="$(sed -n 's/^COMMAND_SHA256 //p' "$WORK/bundle")"
intent_sha256="$(sed -n 's/^INTENT_SHA256 //p' "$WORK/bundle")"
event_sha256="$(sed -n 's/^EVENT_SHA256 //p' "$WORK/bundle")"
[[ "$command" == 'loom-causal-cell-v1 ATTEST bind=source+semantics+artifact+result+toolchain+hardware' &&
   "$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)" == "$command_sha256" &&
   "$command_sha256" == 69557f86703244f866bd53f75e860d7a8f1608e01f76c88a37e5d0035e730ad2 &&
   "$intent_sha256" == 184c2d2d088e53f99991aac98886a4b6acfddfa1ce03137cdcdf3f420508432b &&
   "$event_sha256" == 75aed867cad6acf13cae40732f8dbdccc0af35d0a8d856472b59b104fa670780 ]] ||
  fail 'typed ATTEST command lineage diverged'

for tuple in 'issue issue_decision' 'consume consume_decision' 'close close_decision'; do
  read -r label expected <<< "$tuple"
  actual="$(printf '%s\n' "$(frame_for "$label")" | "$WORK/authority" || true)"
  [[ "$actual" == "$(manifest_value "$expected")" ]] ||
    fail "Sounio authority disagreed with $label"
done
mismatch="$(printf '%s\n' "$(frame_for command_mismatch)" | "$WORK/authority" || true)"
[[ "$mismatch" == 'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=492 '* ]] ||
  fail 'command mismatch control diverged'

ORACLE_EXECUTED="$WORK/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$WORK/$name"
  chmod 0755 "$WORK/$name"
done
PATH="$WORK:$PATH" "$WORK/fixture-one" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'

printf 'sounio-loom-causal-workflow-attest-grant-fixture-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER launch_action=9030 workflow_action=9037 fixtures=4 treatment=issue+consume+close command_mismatch=DENY492 causal_sabotage=PASS command_sha256=%s intent_sha256=%s event_sha256=%s source_sha256=%s executable_sha256=%s bundle_sha256=%s deterministic=true arbitrary_shell=false expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean material_grant=false material_execution=false host_launch_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$command_sha256" "$intent_sha256" "$event_sha256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/causal_workflow_attest_grant_fixture_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$WORK/fixture-one" | cut -d ' ' -f 1)" \
  "$(sha256sum "$WORK/bundle" | cut -d ' ' -f 1)"
