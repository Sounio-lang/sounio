#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md"
EXEC_SOURCE="$ROOT_DIR/tools/loom/src/loom_exec.ml"
HOOK_SOURCE="$ROOT_DIR/tools/loom/src/loom_hook.ml"
INGRESS="$ROOT_DIR/tools/loom/product_exec_ingress_dark.runtime.v1"
HOST_GRANT="$ROOT_DIR/tools/loom/host_exec_quorum_host.runtime.v1"
PROCESS_WITNESS="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"
PEER_MATERIAL="$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13.freeze.v1"

fail() {
  printf 'sounio-loom-product-dynamic-user-exec-counterexample-selftest: FAIL reason=%s\n' \
    "$*" >&2
  exit 1
}

field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$path field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

for path in "$GARDEN" "$EXEC_SOURCE" "$HOOK_SOURCE" "$INGRESS" \
  "$HOST_GRANT" "$PROCESS_WITNESS" "$PEER_MATERIAL"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required parent is absent: $path"
done

[[ "$(field "$INGRESS" stage)" == PRODUCT_DARK_ATTACHMENT_FROZEN && \
   "$(field "$INGRESS" semantic_authority)" == Sounio && \
   "$(field "$INGRESS" semantic_action)" == 9031 && \
   "$(field "$INGRESS" descriptor_dark_attached)" == true && \
   "$(field "$INGRESS" descriptor_is_bearer)" == false && \
   "$(field "$INGRESS" required_mode_default)" == false && \
   "$(field "$INGRESS" distinct_uid_product_broker)" == false && \
   "$(field "$INGRESS" material_execution)" == false ]] ||
  fail 'product ExecIngress no longer exposes the preregistered gap'

[[ "$(field "$HOST_GRANT" stage)" == MATERIAL_GRANT_FROZEN && \
   "$(field "$HOST_GRANT" semantic_authority)" == Sounio && \
   "$(field "$HOST_GRANT" action)" == 9030 && \
   "$(field "$HOST_GRANT" principal_distinct_uid)" == true && \
   "$(field "$HOST_GRANT" non_bearer_exec_quorum)" == true && \
   "$(field "$HOST_GRANT" material_grant)" == true && \
   "$(field "$HOST_GRANT" material_execution)" == false && \
   "$(field "$HOST_GRANT" exec_attached)" == false ]] ||
  fail 'host material grant parent drifted'

[[ "$(field "$PROCESS_WITNESS" stage)" == MATERIAL_EXECUTION_CORE_FROZEN && \
   "$(field "$PROCESS_WITNESS" semantic_authority)" == Sounio && \
   "$(field "$PROCESS_WITNESS" action)" == 9030 && \
   "$(field "$PROCESS_WITNESS" principal_distinct_uid)" == true && \
   "$(field "$PROCESS_WITNESS" process_witness_core)" == true && \
   "$(field "$PROCESS_WITNESS" affirmative_extinction)" == true && \
   "$(field "$PROCESS_WITNESS" material_grant)" == true && \
   "$(field "$PROCESS_WITNESS" material_execution)" == false && \
   "$(field "$PROCESS_WITNESS" exec_attached)" == false ]] ||
  fail 'ProcessWitness parent drifted'

[[ "$(field "$PEER_MATERIAL" semantic_authority)" == Sounio && \
   "$(field "$PEER_MATERIAL" same_uid_peer_isolation)" == true && \
   "$(field "$PEER_MATERIAL" material_coverage)" == true && \
   "$(field "$PEER_MATERIAL" complete_effects)" == true && \
   "$(field "$PEER_MATERIAL" material_execution)" == true && \
   "$(field "$PEER_MATERIAL" production_activation)" == false && \
   "$(field "$PEER_MATERIAL" exec_attached)" == false ]] ||
  fail 'material peer-isolation parent drifted'

# Bind named toplevel lets. Line-range sed is the defect that made this
# gate FAIL while broker_command_kernel and supervise_child still existed.
ocaml_toplevel_let() {
  local path="$1" name="$2" count prefix body
  count="$(grep -cE "^let ${name} " "$path" || true)"
  [[ "$count" == 1 ]] || fail "toplevel let ${name} occurs ${count} times in ${path#$ROOT_DIR/}"
  prefix="let ${name} "
  body="$(awk -v prefix="$prefix" '
    index($0, prefix) == 1 { p = 1 }
    p && /^[^[:space:](]/ && index($0, prefix) != 1 { exit }
    p { print }
  ' "$path")"
  [[ -n "$body" ]] || fail "toplevel let ${name} extracted empty"
  printf '%s' "$body"
}

grep -Fq 'let token_file = required_environment "SOUNIO_LOOM_TOKEN_FILE"' \
  "$EXEC_SOURCE" || fail 'product kernel control no longer exposes its bearer token file'

broker_slice="$(ocaml_toplevel_let "$EXEC_SOURCE" broker_command_kernel)"
[[ "$broker_slice" == $'let broker_command_kernel instance generation handle =\n'* ]] ||
  fail 'broker_command_kernel let shape drifted'
[[ "$broker_slice" == *'Unix.realpath Sys.executable_name'* && \
   "$broker_slice" == *'"exec-capability"'* ]] ||
  fail 'product command no longer returns to the same OCaml executable'

child_slice="$(ocaml_toplevel_let "$EXEC_SOURCE" supervise_child)"
[[ "$child_slice" == $'let supervise_child executable argv environment =\n'* ]] ||
  fail 'supervise_child let shape drifted'
[[ "$child_slice" == *'Unix.fork ()'* && \
   "$child_slice" == *'Unix.execve executable'* ]] ||
  fail 'same-principal fork/exec counterexample is absent'
if grep -Eq 'setuid|setgid|systemd|DynamicUser' <<< "$child_slice"; then
  fail 'material child unexpectedly changes principal in the frozen slice'
fi

SABOTAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-dynamic-user-counterexample.XXXXXX")"
cleanup() { rm -rf "$SABOTAGE_DIR"; }
trap cleanup EXIT
sed 's/"exec-capability"/"not-exec-capability"/g' "$EXEC_SOURCE" \
  > "$SABOTAGE_DIR/broker-sabotage.ml"
broker_sabotage="$(ocaml_toplevel_let "$SABOTAGE_DIR/broker-sabotage.ml" broker_command_kernel)"
[[ "$broker_sabotage" != *'"exec-capability"'* ]] ||
  fail 'broker reentry sabotage did not remove exec-capability'
sed 's/Unix.fork ()/Unix.fork_absent ()/g' "$EXEC_SOURCE" \
  > "$SABOTAGE_DIR/child-sabotage.ml"
child_sabotage="$(ocaml_toplevel_let "$SABOTAGE_DIR/child-sabotage.ml" supervise_child)"
[[ "$child_sabotage" != *'Unix.fork ()'* ]] ||
  fail 'fork/exec sabotage did not remove Unix.fork'

observe_line="$(grep -n -m1 'Loom_exec_ingress.observe' "$HOOK_SOURCE" | cut -d: -f1)"
issue_line="$(grep -n -m1 'Loom_exec.authorize_and_issue' "$HOOK_SOURCE" | cut -d: -f1)"
[[ "$observe_line" =~ ^[0-9]+$ && "$issue_line" =~ ^[0-9]+$ && \
   "$observe_line" -lt "$issue_line" ]] ||
  fail 'native hook no longer observes ExecIngress before issuing ExecGrant'

printf '%s\n' \
  'sounio-loom-product-dynamic-user-exec-counterexample-selftest: PASS semantic_authority=Sounio actions=9025+9030+9031 structural_producer=Bash structural_role=COUNTEREXAMPLE_ONLY probe=named-let-broker_command_kernel+supervise_child exec_ingress=descriptor-bound-dark required_mode_default=false host_material_grant=frozen+unattached process_witness=frozen+unattached peer_material_execution=true+product_unattached kernel_control=bearer-token-file broker_reentry=same-OCaml-executable child_execution=fork+execve child_principal=same-euid-by-inheritance lane_cell_attached=false distinct_uid_product_broker=false exec_cell_attached=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false causal_sabotage=PASS line_range_sed=false next=distinct-uid-LaneCell-canary'
