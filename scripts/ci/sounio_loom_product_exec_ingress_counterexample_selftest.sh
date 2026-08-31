#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PRODUCT_EXEC_INGRESS_V1.md"
CUSTODY_GATE="$ROOT_DIR/scripts/ci/sounio_loom_execution_custody_selftest.sh"
KERNEL_SOURCE="$ROOT_DIR/tools/loom/src/loom.ml"
EXEC_SOURCE="$ROOT_DIR/tools/loom/src/loom_exec.ml"
COUNTEREXAMPLE_COMMIT="eb853be79be289deb596bea0b3ab8a042509d8df"

fail() {
  printf 'sounio-loom-product-exec-ingress-counterexample-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] ||
    fail "hash drift: ${path#$ROOT_DIR/} expected=$expected actual=$actual"
}

expect_sha256 "$GARDEN" \
  067996ec5031fa77721664dc39c403bf20bea9cf979fcb7b841eed0f11f35c2b
expect_sha256 "$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1" \
  8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051
expect_sha256 "$ROOT_DIR/tools/loom/host_exec_quorum_host.runtime.v1" \
  8c0851bb5e0f2f1982ec220d3e335bfd8c41e6b0500a763c02a3f1901c834ac5
expect_sha256 "$ROOT_DIR/tools/loom/process_witness_host.runtime.v1" \
  eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00
expect_sha256 "$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13.freeze.v1" \
  f7adafcd1c79364b75ebe48b66999ec2d7b82a12d6b8e45d9c1cc4637a4ca9ca

grep -Fq 'tool_name:"exec_command"' "$CUSTODY_GATE" ||
  fail 'custody harness no longer fabricates an exec hook event'
grep -Fq '"$SOUNIO_LOOM_TEST_BINARY" agent-hook --agent codex' "$CUSTODY_GATE" ||
  fail 'custody harness no longer invokes the hook binary itself'
grep -Fq 'let token_file = required_environment "SOUNIO_LOOM_TOKEN_FILE"' \
  "$EXEC_SOURCE" || fail 'EXEC client no longer reads the shared bearer file'
grep -Fq 'token = kernel.token' "$KERNEL_SOURCE" ||
  fail 'kernel request no longer admits by the shared bearer value'

git -C "$ROOT_DIR" cat-file -e "$COUNTEREXAMPLE_COMMIT^{commit}" ||
  fail 'frozen counterexample commit is unavailable'
if git -C "$ROOT_DIR" grep -q -E \
    'SOUNIO_LOOM_EXEC_INGRESS_FD|product_exec_ingress_fd|ExecIngress' \
    "$COUNTEREXAMPLE_COMMIT" -- tools/loom/src; then
  fail 'frozen counterexample commit unexpectedly contained product ingress'
fi

for config in \
  "$ROOT_DIR/.codex/hooks.json" \
  "$ROOT_DIR/.claude/settings.json" \
  "$ROOT_DIR/.cursor/hooks.json" \
  "$ROOT_DIR/.grok/hooks/loom-native.json"; do
  [[ -f "$config" ]] || fail "native hook config missing: ${config#$ROOT_DIR/}"
  grep -Fq 'sounio-loom-runtime' "$config" ||
    fail "native OCaml hook missing: ${config#$ROOT_DIR/}"
  if grep -Eiq 'python|pypy|rustc|cargo|sounio_coord_agent_hook\.py' "$config"; then
    fail "non-native hook command present: ${config#$ROOT_DIR/}"
  fi
done

[[ ! -e "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" &&
   ! -e "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" ]] ||
  fail 'legacy Python compatibility boundary remains present'
[[ -f "$ROOT_DIR/tools/loom/native_hook_cutover.freeze.v1" &&
   -f "$ROOT_DIR/stdlib/coordination/loom_native_hook_cutover_authority.sio" ]] ||
  fail 'frozen Sounio native-hook cutover authority is missing'

custody_output="$(bash "$CUSTODY_GATE")" ||
  fail 'execution custody prerequisite failed'
[[ "$custody_output" == *'sounio-loom-execution-custody-selftest: PASS'* ]] ||
  fail 'execution custody prerequisite omitted its PASS receipt'
[[ "$custody_output" == *'same_uid_outside_ancestry=refused'* ]] ||
  fail 'execution custody prerequisite omitted its outside-ancestry control'

printf '%s\n' \
  'sounio-loom-product-exec-ingress-counterexample-selftest: PASS semantic_authority=Sounio action=9030 operational_kernel=OCaml frozen_counterexample_commit=eb853be79be289deb596bea0b3ab8a042509d8df current_hook_at_freeze=forged-JSON-from-harness frozen_counterexample=accepted counterexample_falsifies_product_attachment=true shared_bearer_file=true same_uid_same_executable=true same_harness_ancestry=true outside_ancestry_control=refused missing_fact=non-bearer-inherited-ingress native_hook_config=codex+claude+cursor+grok native_hook_cutover_action=9045 legacy_python_compatibility_bridge=absent python_executed=false rust_executed=false product_exec_ingress_observed_at_freeze=false same_ancestry_forgery_refused_at_freeze=false non_bearer_product_ingress_at_freeze=false production_activation=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next=descriptor-bound-dark-ingress'
