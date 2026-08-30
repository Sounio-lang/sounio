#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
INSTALLER="$ROOT_DIR/scripts/dev/install_loom_hostd.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-hostd-install.XXXXXX")"
INSTALLED_AGENT=loom-hostd-installed-test
INSTALLED_LANE=outside-checkout
SERVICE_USER="$(id -un)"
SERVICE_UID="$(id -u)"
SERVICE_GID="$(id -g)"

fail() {
  printf 'sounio-loom-hostd-install-selftest: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  if [[ -x "${installed_runtime:-}" ]]; then
    "$installed_runtime" stop --state-dir "$TEST_ROOT/installed-state" \
      --cwd "$TEST_ROOT/isolated-cwd" --agent "$INSTALLED_AGENT" \
      --lane "$INSTALLED_LANE" >/dev/null 2>&1 || true
  fi
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    find "$TEST_ROOT" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

runtime="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
authority="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-boot-reconciler"
resident="$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime-v5"
stage="$TEST_ROOT/stage"
mkdir -p "$stage"

exec_cell_capsule="${SOUNIO_LOOM_HOSTD_TEST_EXEC_CELL_CAPSULE:-}"
exec_cell_capsule_sha256="${SOUNIO_LOOM_HOSTD_TEST_EXEC_CELL_CAPSULE_SHA256:-}"
if [[ -z "$exec_cell_capsule" && -z "$exec_cell_capsule_sha256" ]]; then
  exec_cell_capsule="$TEST_ROOT/loom-host-exec-cell.tar"
  bash "$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_capsule.sh" \
    --output "$exec_cell_capsule" >/dev/null
  exec_cell_capsule_sha256="$(sha256sum "$exec_cell_capsule" | cut -d ' ' -f 1)"
fi
[[ "$exec_cell_capsule" == /* && -f "$exec_cell_capsule" &&
   ! -L "$exec_cell_capsule" &&
   "$exec_cell_capsule_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'ExecCell capsule fixture is incomplete'

bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null

first="$(bash "$INSTALLER" --install-root "$stage" \
  --runtime "$runtime" --authority "$authority" --resident "$resident" \
  --policy-root "$ROOT_DIR" --user "$SERVICE_USER" \
  --exec-cell-capsule "$exec_cell_capsule" \
  --exec-cell-capsule-sha256 "$exec_cell_capsule_sha256")"
[[ "$first" == *'exec_cell_bundle_present=true '* &&
   "$first" == *'exec_cell_canary_frozen=true '* &&
   "$first" == *'exec_cell_boot_gate_configured=true '* &&
   "$first" == *'exec_cell_boot_gate_test_only=true exec_result_transport_configured=true exec_intent_projection_configured=true '* &&
   "$first" == *'exact_fixture_result_attached=false exec_attached=false '* &&
   "$first" == *'activated=false automatic_lineage_resurrection=false'* ]] ||
  fail "staged installer widened activation: $first"

prefix="$stage/opt/sounio/loom-hostd"
unit="$stage/etc/systemd/system/sounio-loom-hostd.service"
manifest="$prefix/manifest.v1"
installed_runtime="$prefix/bin/sounio-loom-runtime"
installed_authority="$prefix/bin/sounio-loom-host-boot-reconciler"
installed_resident="$prefix/bin/sounio-loom-resident-membrane-runtime-v5"
policy_root="$prefix/policy/product-activation"
policy_manifest="$prefix/share/product-activation-policy.v1"
exec_cell_manifest="$prefix/share/exec-cell-bundle.v1"
socket_root="$stage/tmp/sounio-loom-$SERVICE_UID"
[[ -x "$installed_runtime" && -x "$installed_authority" && \
   -x "$installed_resident" && -f "$unit" && -f "$manifest" && \
   -f "$policy_manifest" && -f "$exec_cell_manifest" && \
   -d "$socket_root" ]] ||
  fail 'staged installation omitted an artifact'
[[ "$(stat -c %a "$socket_root")" == 700 ]] ||
  fail 'staged socket namespace is not mode 0700'
[[ "$(sha256sum "$installed_authority" | cut -d ' ' -f 1)" == \
   '99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b' ]] ||
  fail 'installed Sounio authority hash drifted'
grep -Fxq 'language=OCaml' < <("$installed_runtime" runtime-version) ||
  fail 'installed runtime is not OCaml'
[[ "$(printf '0\n' | "$installed_authority")" == \
   'SOUNIO_HOST_BOOT_RECONCILER_SELFTEST PASS cases=14' ]] ||
  fail 'installed Sounio authority does not selftest'
[[ "$(sha256sum "$installed_resident" | cut -d ' ' -f 1)" == \
   "$(sed -n 's/^runtime_sha256=//p' "$policy_root/tools/loom/resident_membrane.runtime.v5")" ]] ||
  fail 'installed resident runtime does not match its frozen manifest'
grep -Fxq 'semantic_authority=Sounio' "$policy_manifest" ||
  fail 'installed product activation policy lost Sounio authority'
grep -Fxq 'semantic_action=9031' "$policy_manifest" ||
  fail 'installed product activation policy lost action 9031'
grep -Eq '^policy_file_count=[1-9][0-9]*$' "$policy_manifest" ||
  fail 'installed product activation policy file count is absent'
grep -Fxq 'semantic_authority=Sounio' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle lost Sounio authority'
grep -Fxq 'semantic_actions=9030,9031,9033,9034' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle lost the frozen action order'
grep -Fxq 'bundle_present=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle is not present'
grep -Fxq "capsule_sha256=$exec_cell_capsule_sha256" "$exec_cell_manifest" ||
  fail 'installed ExecCell capsule hash drifted'
grep -Fxq 'controller_language=OCaml' "$exec_cell_manifest" ||
  fail 'installed ExecCell controller is not OCaml'
grep -Fxq 'controller_role=EFFECT_PARITY' "$exec_cell_manifest" ||
  fail 'installed ExecCell controller role drifted'
grep -Fxq 'material_language=C++20+Linux+systemd' "$exec_cell_manifest" ||
  fail 'installed ExecCell material bootstrap drifted'
grep -Fxq 'material_transitory=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell material bootstrap lost its transitory marker'
grep -Fxq 'python_executable_invoked=false' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle admitted a Python oracle'
grep -Fxq 'rust_executable_invoked=false' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle admitted a Rust oracle'
grep -Fxq 'exec_cell_canary_frozen=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell canary is not frozen'
grep -Fxq 'exec_cell_boot_gate_configured=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell boot gate is not configured'
grep -Fxq 'exec_cell_boot_gate_test_only=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell boot gate was misrepresented as general execution'
grep -Fxq 'exec_result_transport_configured=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle omitted the Sounio 9033 result transport'
grep -Fxq 'exec_intent_projection_configured=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle omitted the Sounio 9034 intent projection'
grep -Fxq 'provider_hook_fixture_configured=true' "$exec_cell_manifest" ||
  fail 'installed ExecCell bundle omitted the OCaml provider hook fixture'
grep -Fxq 'provider_hook_switched=false' "$exec_cell_manifest" ||
  fail 'installer preclaimed the provider hook switch'
grep -Fxq 'provider_lifecycle_attached=false' "$exec_cell_manifest" ||
  fail 'installer preclaimed provider lifecycle attachment'
grep -Fxq 'exact_fixture_result_attached=false' "$exec_cell_manifest" ||
  fail 'installer preclaimed execution of the exact result fixture'
grep -Fxq 'exec_attached=false' "$exec_cell_manifest" ||
  fail 'installer preclaimed product ExecCell attachment'
exec_cell_release_id="$(sed -n 's/^release_id=//p' "$exec_cell_manifest")"
exec_cell_release="$prefix/exec-cell/releases/$exec_cell_release_id"
[[ "$exec_cell_release_id" =~ ^9030-hostq-[0-9a-f]{32}$ &&
   -d "$exec_cell_release" && ! -L "$exec_cell_release" &&
   -x "$exec_cell_release/bin/loom-kernel-principal-broker" &&
   -x "$exec_cell_release/bin/loom-exec-grant-controller" &&
   -x "$exec_cell_release/bin/loom-process-witness-principal-cell" &&
   -x "$exec_cell_release/bin/sounio-loom-process-witness-handshake-v1" &&
   -x "$exec_cell_release/bin/sounio-loom-exec-result-handle" &&
   -x "$exec_cell_release/bin/sounio-loom-exec-intent-envelope" &&
   -x "$exec_cell_release/bin/sounio-loom-provider-hook-fixture" &&
   -f "$exec_cell_release/authority-root/tools/loom/exec_result_handle.freeze.v1" &&
   -f "$exec_cell_release/authority-root/tools/loom/exec_intent_envelope.freeze.v1" &&
   -f "$exec_cell_release/data/product-exec-cell-fixtures.v1" ]] ||
  fail 'installed ExecCell release omitted a bound runtime'
grep -Fxq 'semantic_action=9030' "$exec_cell_release/release.manifest.v1" ||
  fail 'installed ExecCell release lost action 9030'
grep -Fxq 'product_exec_ingress_action=9031' \
  "$exec_cell_release/release.manifest.v1" ||
  fail 'installed ExecCell release lost action 9031'
grep -Fxq 'product_exec_result_action=9033' \
  "$exec_cell_release/release.manifest.v1" ||
  fail 'installed ExecCell release lost action 9033'
grep -Fxq 'product_exec_intent_action=9034' \
  "$exec_cell_release/release.manifest.v1" ||
  fail 'installed ExecCell release lost action 9034'

grep -Fxq 'KillMode=process' "$unit" || fail 'unit would kill recovered lane processes'
grep -Fxq 'Restart=on-failure' "$unit" || fail 'unit does not restart after refusal or crash'
grep -Fxq 'NoNewPrivileges=true' "$unit" || fail 'unit lacks no-new-privileges boundary'
grep -Fxq 'ProtectSystem=strict' "$unit" || fail 'unit lacks strict filesystem protection'
grep -Fxq 'PrivateTmp=false' "$unit" ||
  fail 'unit cannot observe the lane socket namespace'
grep -Fxq 'Environment=XDG_RUNTIME_DIR=/tmp' "$unit" ||
  fail 'unit does not bind recovery to the managed socket namespace'
grep -Fxq "ReadWritePaths=/var/lib/sounio/loom /tmp/sounio-loom-$SERVICE_UID" "$unit" ||
  fail 'unit can neither mutate the managed namespace nor limits its write surface'
grep -Fxq 'ExecStart=/opt/sounio/loom-hostd/bin/sounio-loom-runtime host-supervise --state-dir /var/lib/sounio/loom --service-enabled --apply' "$unit" ||
  fail 'unit is not wired to the Sounio-authorized supervisor'
grep -Fq 'ExecStartPre=/usr/bin/systemd-run --quiet --wait --pipe --collect ' \
  "$unit" || fail 'unit is not wired to an isolated ExecCell boot gate'
grep -Fq ' --property=ReadWritePaths=/run\x20/var/lib/sounio/loom ' "$unit" ||
  fail 'isolated ExecCell boot gate lacks its bounded transient write policy'
grep -Fq ' --property=PrivateTmp=yes ' "$unit" ||
  fail 'isolated ExecCell boot gate lacks a private writable temporary root'
grep -Fq ' --setenv=SOUNIO_LOOM_RESIDENT_RECEIPT_LOG=/var/lib/sounio/loom/exec-cell-gate-authority.tsv ' \
  "$unit" || fail 'isolated ExecCell boot gate does not separate audit state from frozen code'
grep -Fq ' -- /opt/sounio/loom-hostd/exec-cell/releases/' "$unit" ||
  fail 'isolated boot gate is not wired to the frozen ExecCell release'
grep -Fq ' --selftest-product-exec-cell-host ' "$unit" ||
  fail 'unit boot gate does not execute the composed ExecCell canary'
grep -Fq ' --product-exec-cell-fixture-manifest ' "$unit" ||
  fail 'unit boot gate omitted the Sounio fixture manifest'
grep -Fq ' --product-exec-result-manifest ' "$unit" ||
  fail 'unit boot gate omitted the Sounio 9033 result manifest'
grep -Fq ' --product-provider-hook-fixture ' "$unit" ||
  fail 'unit boot gate omitted the OCaml provider hook fixture'
grep -Fxq 'TimeoutStartSec=240s' "$unit" ||
  fail 'unit boot gate lacks a bounded startup deadline'
grep -Fxq 'Environment=SOUNIO_LOOM_HOST_BOOT_AUTHORITY=/opt/sounio/loom-hostd/bin/sounio-loom-host-boot-reconciler' "$unit" ||
  fail 'unit omitted the frozen authority path'
grep -Fxq 'install_default=disabled' "$manifest" || fail 'install default is not disabled'
grep -Fxq 'service_enabled=false' "$manifest" || fail 'staged service was marked enabled'
grep -Fxq 'production_activation=false' "$manifest" || fail 'staged service was marked production-active'
grep -Fxq 'exec_cell_boot_gate_configured=true' "$manifest" ||
  fail 'hostd manifest omitted the ExecCell boot gate'
grep -Fxq 'exec_cell_boot_gate_test_only=true' "$manifest" ||
  fail 'hostd manifest widened the ExecCell boot gate'
grep -Fxq 'exec_result_transport_configured=true' "$manifest" ||
  fail 'hostd manifest omitted the Sounio 9033 result transport'
grep -Fxq 'exec_intent_projection_configured=true' "$manifest" ||
  fail 'hostd manifest omitted the Sounio 9034 intent projection'
grep -Fxq 'provider_hook_fixture_configured=true' "$manifest" ||
  fail 'hostd manifest omitted the OCaml provider hook fixture'
grep -Fxq 'provider_hook_switched=false' "$manifest" ||
  fail 'hostd manifest preclaimed the provider hook switch'
grep -Fxq 'provider_lifecycle_attached=false' "$manifest" ||
  fail 'hostd manifest preclaimed provider lifecycle attachment'
grep -Fxq 'exact_fixture_result_attached=false' "$manifest" ||
  fail 'hostd manifest preclaimed execution of the exact result fixture'
grep -Fxq 'socket_namespace=host-shared-tmp' "$manifest" ||
  fail 'manifest omitted the shared lane socket namespace'
grep -Fxq "service_user=$SERVICE_USER" "$manifest" ||
  fail 'manifest service user drifted'
grep -Fxq "service_uid=$SERVICE_UID" "$manifest" ||
  fail 'manifest service uid drifted'
grep -Fxq "service_gid=$SERVICE_GID" "$manifest" ||
  fail 'manifest service gid drifted'
grep -Fxq "socket_root=/tmp/sounio-loom-$SERVICE_UID" "$manifest" ||
  fail 'manifest omitted the managed socket root'
grep -Fxq 'private_tmp=false' "$manifest" ||
  fail 'manifest disagrees with the unit socket namespace'
[[ ! -e "$stage/etc/systemd/system/multi-user.target.wants/sounio-loom-hostd.service" ]] ||
  fail 'staged install created an enablement link'

unit_sha_before="$(sha256sum "$unit" | cut -d ' ' -f 1)"
manifest_sha_before="$(sha256sum "$manifest" | cut -d ' ' -f 1)"
policy_sha_before="$(sha256sum "$policy_manifest" | cut -d ' ' -f 1)"
exec_cell_sha_before="$(sha256sum "$exec_cell_manifest" | cut -d ' ' -f 1)"
bash "$INSTALLER" --install-root "$stage" --runtime "$runtime" \
  --authority "$authority" --resident "$resident" --policy-root "$ROOT_DIR" \
  --user "$SERVICE_USER" --exec-cell-capsule "$exec_cell_capsule" \
  --exec-cell-capsule-sha256 "$exec_cell_capsule_sha256" >/dev/null
[[ "$(sha256sum "$unit" | cut -d ' ' -f 1)" == "$unit_sha_before" && \
   "$(sha256sum "$manifest" | cut -d ' ' -f 1)" == "$manifest_sha_before" && \
   "$(sha256sum "$policy_manifest" | cut -d ' ' -f 1)" == "$policy_sha_before" && \
   "$(sha256sum "$exec_cell_manifest" | cut -d ' ' -f 1)" == \
     "$exec_cell_sha_before" ]] ||
  fail 'staged reinstall was not byte-idempotent'

mkdir -p "$TEST_ROOT/isolated-cwd"
SOUNIO_LOOM_DURABLE_LANE_CANARY=1 "$installed_runtime" start \
  --state-dir "$TEST_ROOT/installed-state" --agent "$INSTALLED_AGENT" \
  --lane "$INSTALLED_LANE" --session-id installed-policy-selftest \
  --cwd "$TEST_ROOT/isolated-cwd" -- "$installed_runtime" \
  _durable-lane-canary >"$TEST_ROOT/installed-start.out"
grep -q 'LOOM_STARTED' "$TEST_ROOT/installed-start.out" ||
  fail 'installed runtime did not start outside the source checkout'
"$installed_runtime" stop --state-dir "$TEST_ROOT/installed-state" \
  --cwd "$TEST_ROOT/isolated-cwd" --agent "$INSTALLED_AGENT" \
  --lane "$INSTALLED_LANE" >/dev/null

action_manifest="$policy_root/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
cp "$action_manifest" "$TEST_ROOT/action-manifest.clean"
chmod 0644 "$action_manifest"
printf 'X' | dd of="$action_manifest" bs=1 seek=128 conv=notrunc status=none
if SOUNIO_LOOM_DURABLE_LANE_CANARY=1 "$installed_runtime" start \
  --state-dir "$TEST_ROOT/mutated-policy-state" --agent policy-mutant \
  --lane denied --session-id mutated-policy --cwd "$TEST_ROOT/isolated-cwd" \
  -- "$installed_runtime" _durable-lane-canary \
  >"$TEST_ROOT/policy-mutant.out" 2>"$TEST_ROOT/policy-mutant.err"; then
  fail 'installed runtime admitted a mutated product activation policy'
fi
grep -q 'activation-dark-action-manifest-hash-mismatch' \
  "$TEST_ROOT/policy-mutant.err" ||
  fail 'mutated product activation policy was refused by the wrong boundary'
[[ ! -e "$TEST_ROOT/mutated-policy-state/sessions/policy-mutant--denied/session.state" ]] ||
  fail 'mutated product activation policy was detected after lane mutation'
cp "$TEST_ROOT/action-manifest.clean" "$action_manifest"
chmod 0444 "$action_manifest"

if bash "$INSTALLER" --install-root "$stage" --runtime "$runtime" \
  --authority "$authority" --resident "$resident" --policy-root "$ROOT_DIR" \
  --user "$SERVICE_USER" --activate \
  >"$TEST_ROOT/activate.out" 2>"$TEST_ROOT/activate.err"; then
  fail 'staged installer accepted activation'
fi
grep -q -- '--activate is forbidden for a staged install root' "$TEST_ROOT/activate.err" ||
  fail 'staged activation was refused for the wrong reason'

mkdir -p "$TEST_ROOT/capsule-negative-stage"
if bash "$INSTALLER" --install-root "$TEST_ROOT/capsule-negative-stage" \
  --runtime "$runtime" --authority "$authority" --resident "$resident" \
  --policy-root "$ROOT_DIR" --user "$SERVICE_USER" \
  --exec-cell-capsule "$exec_cell_capsule" \
  --exec-cell-capsule-sha256 \
    0000000000000000000000000000000000000000000000000000000000000000 \
  >"$TEST_ROOT/capsule-hash.out" 2>"$TEST_ROOT/capsule-hash.err"; then
  fail 'installer admitted an ExecCell capsule with the wrong expected hash'
fi
grep -q 'ExecCell capsule archive hash drifted' "$TEST_ROOT/capsule-hash.err" ||
  fail 'wrong ExecCell capsule hash was refused by the wrong boundary'

cp "$exec_cell_capsule" "$TEST_ROOT/exec-cell-capsule-mutant.tar"
printf 'X' | dd of="$TEST_ROOT/exec-cell-capsule-mutant.tar" bs=1 seek=128 \
  conv=notrunc status=none
mutant_capsule_sha256="$(sha256sum \
  "$TEST_ROOT/exec-cell-capsule-mutant.tar" | cut -d ' ' -f 1)"
if bash "$INSTALLER" --install-root "$TEST_ROOT/capsule-negative-stage" \
  --runtime "$runtime" --authority "$authority" --resident "$resident" \
  --policy-root "$ROOT_DIR" --user "$SERVICE_USER" \
  --exec-cell-capsule "$TEST_ROOT/exec-cell-capsule-mutant.tar" \
  --exec-cell-capsule-sha256 "$mutant_capsule_sha256" \
  >"$TEST_ROOT/capsule-mutant.out" 2>"$TEST_ROOT/capsule-mutant.err"; then
  fail 'installer admitted a tampered ExecCell capsule with a matching outer hash'
fi
grep -q 'ExecCell capsule verifier refused the archive' \
  "$TEST_ROOT/capsule-mutant.err" ||
  fail 'tampered ExecCell capsule was refused by the wrong boundary'

cp "$authority" "$TEST_ROOT/authority-mutant"
printf 'X' | dd of="$TEST_ROOT/authority-mutant" bs=1 seek=128 conv=notrunc status=none
chmod 0755 "$TEST_ROOT/authority-mutant"
mkdir -p "$TEST_ROOT/mutant-stage"
if bash "$INSTALLER" --install-root "$TEST_ROOT/mutant-stage" \
  --runtime "$runtime" --authority "$TEST_ROOT/authority-mutant" \
  --resident "$resident" --policy-root "$ROOT_DIR" --user "$SERVICE_USER" \
  >"$TEST_ROOT/mutant.out" 2>"$TEST_ROOT/mutant.err"; then
  fail 'installer admitted a mutated Sounio authority'
fi
grep -q 'Sounio authority hash drifted' "$TEST_ROOT/mutant.err" ||
  fail 'mutated authority was refused by the wrong boundary'

printf 'sounio-loom-hostd-install-selftest: PASS installer_transport=shell runtime_language=OCaml runtime_role=EFFECT_PARITY semantic_authority=Sounio actions=9030,9031,9033,9034,9041 installed_policy_root=PASS exec_cell_bundle=IMMUTABLE exec_cell_canary_frozen=true exec_result_transport_configured=true exec_intent_projection_configured=true exact_fixture_result_attached=false exec_attached=false outside_checkout_start=PASS policy_tamper=DENIED_PRE_MUTATION capsule_hash_mismatch=DENIED_PRE_INSTALL capsule_inner_tamper=DENIED_PRE_INSTALL systemd_unit=PASS socket_namespace=host-shared-tmp socket_root=/tmp/sounio-loom-%s socket_mode=0700 private_tmp=false kill_mode=process restart=on-failure install_default=disabled staged_activation=DENIED mutated_authority=DENIED byte_idempotent=PASS python_executed=false rust_executed=false production_activation=false\n' "$SERVICE_UID"
