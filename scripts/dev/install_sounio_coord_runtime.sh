#!/usr/bin/env bash

set -euo pipefail
umask 077

CLIENT_PROTOCOL=3

usage() {
  cat <<'USAGE'
Usage: scripts/dev/install_sounio_coord_runtime.sh [options]

Install and atomically activate the coordination runtime shared by every
worktree attached to this repository.

Options:
  --source-root PATH       source bundle root (default: current worktree)
  --runtime-dir PATH       shared runtime root override
  --activate RUNTIME_ID    activate an already installed version
  --list                   list installed versions
  -h, --help               show this help
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local manifest="$1" key="$2"
  sed -n "s/^${key}=//p" "$manifest" | head -1
}

verify_manifest_binary_sha256() {
  local manifest="$1" key="$2" binary="$3" expected actual
  expected="$(manifest_value "$manifest" "$key")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || \
    die "installed runtime has invalid $key: $manifest"
  [[ -f "$binary" ]] || die "installed runtime hash target is absent: $binary"
  actual="$(sha256sum "$binary" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || \
    die "installed runtime binary hash mismatch: $key expected=$expected actual=$actual"
}

ensure_obligation_activation() {
  local activation_dir activation_file lock_file manifest installed_utc epoch
  local earliest_epoch='' earliest_utc='' earliest_runtime='' candidate_runtime tmp_file
  activation_dir="$GIT_COMMON_DIR/sounio-coord-state"
  activation_file="$activation_dir/loom-obligation-activation.v1"
  lock_file="$GIT_COMMON_DIR/.sounio-coord-obligation-activation.lock"
  mkdir -p "$activation_dir"
  exec 8>"$lock_file"
  flock 8
  if [[ -f "$activation_file" ]]; then
    grep -q '^schema=loom-obligation-activation-v1$' "$activation_file" &&
      grep -Eq '^activated_epoch=[1-9][0-9]*$' "$activation_file" &&
      grep -Eq '^runtime_id=.+$' "$activation_file" ||
      die "invalid durable obligation activation watermark: $activation_file"
    flock -u 8
    return 0
  fi
  for manifest in "$RUNTIME_ROOT"/versions/*/manifest; do
    [[ -f "$manifest" ]] || continue
    grep -q '^capability=loom-durable-obligation-v1$' "$manifest" || continue
    installed_utc="$(manifest_value "$manifest" installed_utc)"
    candidate_runtime="$(manifest_value "$manifest" runtime_id)"
    [[ -n "$installed_utc" && -n "$candidate_runtime" ]] ||
      die "durable obligation runtime has incomplete activation metadata: $manifest"
    epoch="$(date -u -d "$installed_utc" +%s 2>/dev/null || true)"
    [[ "$epoch" =~ ^[1-9][0-9]*$ ]] ||
      die "durable obligation runtime has invalid installed_utc: $manifest"
    if [[ -z "$earliest_epoch" ]] || ((epoch < earliest_epoch)); then
      earliest_epoch="$epoch"
      earliest_utc="$installed_utc"
      earliest_runtime="$candidate_runtime"
    fi
  done
  [[ -n "$earliest_epoch" ]] ||
    die "cannot establish durable obligation activation watermark"
  tmp_file="$(mktemp "$activation_dir/.loom-obligation-activation.XXXXXX")"
  {
    printf 'schema=loom-obligation-activation-v1\n'
    printf 'activated_utc=%s\n' "$earliest_utc"
    printf 'activated_epoch=%s\n' "$earliest_epoch"
    printf 'runtime_id=%s\n' "$earliest_runtime"
    printf 'policy=post-activation-directed-request\n'
  } > "$tmp_file"
  mv "$tmp_file" "$activation_file"
  flock -u 8
}

activate_runtime() {
  local runtime_id="$1" version_dir manifest protocol link_tmp
  local previous_target='' previous_bundle='' previous_runtime=''
  local control_state control_status='' control_service_was_live=0
  local ensure_output='' ensure_rc=0
  version_dir="$RUNTIME_ROOT/versions/$runtime_id"
  manifest="$version_dir/manifest"
  [[ -f "$manifest" && -x "$version_dir/bin/sounio-coord-runtime" && \
    -f "$version_dir/hooks/sounio_coord_agent_hook_runtime.py" ]] || \
    die "installed runtime is incomplete: $runtime_id"
  protocol="$(manifest_value "$manifest" protocol_version)"
  [[ "$protocol" == "$CLIENT_PROTOCOL" ]] || \
    die "cannot activate protocol $protocol with installer protocol $CLIENT_PROTOCOL"
  if grep -q '^capability=agentd-transport-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-agentd-runtime" ]] || \
      die "installed runtime declares agentd transport but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=loom-kernel-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" ]] || \
      die "installed runtime declares Loom but omits its OCaml kernel: $runtime_id"
  fi
  if grep -q '^capability=loom-transactional-custody-transfer-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-custody-transfer-runtime" ]] || \
      die "installed runtime declares transactional custody transfer but omits Loom or frozen Sounio frame 9040: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_custody_transfer_semantics_sha256)" == \
      5f53d3edcb6731c5b0f4e58ff7b27d251e6c0b40eda8c68366e48b17e596f55c ]] || \
      die "installed custody transfer is not bound to frozen Sounio semantics: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" loom_custody_transfer_runtime_sha256 \
      "$version_dir/bin/sounio-loom-custody-transfer-runtime"
  fi
  if grep -q '^capability=loom-durable-execution-outcome-v1$' "$manifest"; then
    grep -q '^capability=loom-transactional-custody-transfer-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-execution-outcome-runtime" ]] || \
      die "installed runtime declares durable execution outcomes without transactional custody, Loom, or frozen Sounio frame 9022: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_execution_outcome_semantics_sha256)" == \
      c98c13d30d66ba2fb3d0fb34d75bd21b14b353bc88fd80acf7dbb385cb9fa914 ]] || \
      die "installed execution outcome is not bound to frozen Sounio semantics: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_execution_outcome_manifest_sha256)" == \
      f5e63a2fd6a946cea1a4cb57013ae0cfa1772c42c3cc52e42d300dfb7b45e16e ]] || \
      die "installed execution outcome has an unknown freeze manifest: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" loom_execution_outcome_runtime_sha256 \
      "$version_dir/bin/sounio-loom-execution-outcome-runtime"
  fi
  if grep -q '^capability=loom-native-agent-hook-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-language-authority-runtime" ]] || \
      die "installed runtime declares the native agent hook but omits its OCaml kernel or frozen Sounio authority: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_language_authority_semantics_sha256)" == \
      16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff ]] || \
      die "installed native hook is not bound to the frozen Sounio authority: $runtime_id"
  fi
  if grep -q '^capability=loom-runtime-authority-capsule-v1$' "$manifest"; then
    local authority_capsule="$version_dir/policy/language-authority"
    grep -q '^capability=loom-native-agent-hook-v1$' "$manifest" || \
      die "installed runtime declares an authority capsule without the native hook: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" \
      loom_language_authority_policy_manifest_sha256 \
      "$authority_capsule/tools/loom/language_authority.freeze.v1"
    verify_manifest_binary_sha256 "$manifest" \
      loom_language_authority_policy_source_sha256 \
      "$authority_capsule/stdlib/coordination/loom_language_authority.sio"
    verify_manifest_binary_sha256 "$manifest" \
      loom_language_authority_policy_entrypoint_sha256 \
      "$authority_capsule/tools/loom/language_authority_main.sio"
  fi
  if grep -q '^capability=loom-product-launch-dark-attachment-v1$' "$manifest"; then
    local activation_capsule="$version_dir/policy/product-activation"
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-resident-membrane-runtime-v5" ]] || \
      die "installed runtime declares product launch observation without Loom or resident Sounio v5: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_product_activation_action_manifest_sha256)" == \
      f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e && \
      "$(manifest_value "$manifest" loom_product_activation_operational_manifest_sha256)" == \
      d7521e8fb60501dc8192ebbeade4a09649164c5b509a2dda8af5c465bf3de793 && \
      "$(manifest_value "$manifest" loom_product_activation_resident_manifest_sha256)" == \
      b3cf8c1e0524be35fc67b2b5a779bad9a9291195d65dc82dbc87595396fb5353 && \
      "$(manifest_value "$manifest" loom_product_activation_projection_sha256)" == \
      8a72e9bcd510a751b856cf29960b7389486defcc4d13d7614546023d3d355014 ]] || \
      die "installed product launch observation is not bound to frozen Sounio action 9031: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_activation_action_manifest_sha256 \
      "$activation_capsule/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_activation_operational_manifest_sha256 \
      "$activation_capsule/tools/loom/kernel_peer_activation_capsule.runtime.v1"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_activation_resident_manifest_sha256 \
      "$activation_capsule/tools/loom/resident_membrane.runtime.v5"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_activation_projection_sha256 \
      "$activation_capsule/tools/loom/kernel_peer_activation_capsule.current.v1"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_activation_resident_runtime_sha256 \
      "$version_dir/bin/sounio-loom-resident-membrane-runtime-v5"
  fi
  if grep -q '^capability=loom-product-exec-ingress-dark-attachment-v1$' "$manifest"; then
    local exec_ingress_capsule="$version_dir/policy/product-exec-ingress"
    local exec_ingress_freeze="$exec_ingress_capsule/tools/loom/product_exec_ingress_dark.runtime.v1"
    grep -q '^capability=loom-native-agent-hook-v1$' "$manifest" &&
      grep -q '^capability=loom-native-hook-binary-attestation-v1$' "$manifest" &&
      grep -q '^capability=loom-product-launch-dark-attachment-v1$' "$manifest" ||
      die "installed runtime declares product ExecIngress without binary attestation, the native hook, and Sounio action 9031: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_exec_ingress_manifest_sha256 "$exec_ingress_freeze"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_exec_ingress_contract_sha256 \
      "$exec_ingress_capsule/tools/loom/PRODUCT_EXEC_INGRESS_DARK_ATTACHMENT_V1.md"
    verify_manifest_binary_sha256 "$manifest" \
      loom_product_exec_ingress_evidence_sha256 \
      "$exec_ingress_capsule/tools/loom/evidence/loom-product-exec-ingress-dark-v1-20260829.txt"
    local source_pair source_path_key source_hash_key source_rel source_expected source_actual
    for source_pair in \
      exec_ingress_source_path:exec_ingress_source_sha256 \
      hook_source_path:hook_source_sha256 \
      membrane_source_path:membrane_source_sha256 \
      cli_source_path:cli_source_sha256 \
      c_stub_path:c_stub_sha256 \
      dune_path:dune_sha256; do
      source_path_key="${source_pair%%:*}"
      source_hash_key="${source_pair#*:}"
      source_rel="$(manifest_value "$exec_ingress_freeze" "$source_path_key")"
      [[ "$source_rel" == tools/loom/src/* && "$source_rel" != *'..'* ]] ||
        die "installed product ExecIngress source path is unsafe: $source_rel"
      source_expected="$(manifest_value "$exec_ingress_freeze" "$source_hash_key")"
      source_actual="$(sha256sum "$exec_ingress_capsule/$source_rel" | awk '{print $1}')"
      [[ "$source_actual" == "$source_expected" ]] ||
        die "installed product ExecIngress source drifted: $source_rel"
    done
    [[ "$(manifest_value "$exec_ingress_freeze" semantic_authority)" == Sounio &&
      "$(manifest_value "$exec_ingress_freeze" semantic_action)" == 9031 &&
      "$(manifest_value "$exec_ingress_freeze" operational_language)" == OCaml &&
      "$(manifest_value "$exec_ingress_freeze" operational_role)" == OPERATIONAL_ATTACHMENT &&
      "$(manifest_value "$exec_ingress_freeze" descriptor_dark_attached)" == true &&
      "$(manifest_value "$exec_ingress_freeze" descriptor_is_bearer)" == false &&
      "$(manifest_value "$exec_ingress_freeze" same_uid_self_broker)" == refused &&
      "$(manifest_value "$exec_ingress_freeze" contract_sha256)" == \
        "$(manifest_value "$manifest" loom_product_exec_ingress_contract_sha256)" &&
      "$(manifest_value "$exec_ingress_freeze" evidence_sha256)" == \
        "$(manifest_value "$manifest" loom_product_exec_ingress_evidence_sha256)" &&
      "$(manifest_value "$exec_ingress_freeze" runtime_version)" == \
        "$(manifest_value "$manifest" runtime_version)" &&
      "$(manifest_value "$exec_ingress_freeze" runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_product_exec_ingress_reference_runtime_sha256)" &&
      "$(manifest_value "$exec_ingress_freeze" action_9031_manifest_sha256)" == \
        "$(manifest_value "$manifest" loom_product_activation_action_manifest_sha256)" &&
      "$(manifest_value "$exec_ingress_freeze" action_9031_runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_product_activation_operational_manifest_sha256)" ]] ||
      die "installed product ExecIngress is not bound to its frozen Sounio authority and OCaml runtime: $runtime_id"
  fi
  if grep -q '^capability=loom-sovereign-execution-kernel-product-v1$' "$manifest"; then
    local sovereign_capsule="$version_dir/policy/sovereign-execution"
    local sovereign_product="$sovereign_capsule/tools/loom/sovereign_execution_kernel_product.runtime.v1"
    grep -q '^capability=loom-native-agent-hook-v1$' "$manifest" &&
      grep -q '^capability=loom-native-hook-binary-attestation-v1$' "$manifest" &&
      [[ -x "$version_dir/bin/sounio-loom-runtime" &&
        -x "$version_dir/bin/sounio-loom-sovereign-execution-kernel" ]] ||
      die "installed sovereign execution product omits its native hook, OCaml kernel, or Sounio action 9042: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" \
      loom_sovereign_product_manifest_sha256 "$sovereign_product"
    verify_manifest_binary_sha256 "$manifest" \
      loom_sovereign_product_contract_sha256 \
      "$sovereign_capsule/tools/loom/SOVEREIGN_EXECUTION_KERNEL_PRODUCT_ATTACHMENT_V1.md"
    verify_manifest_binary_sha256 "$manifest" \
      loom_sovereign_product_evidence_sha256 \
      "$sovereign_capsule/tools/loom/evidence/loom-sovereign-execution-kernel-product-v1-20260831.txt"
    verify_manifest_binary_sha256 "$manifest" \
      loom_sovereign_runtime_sha256 \
      "$version_dir/bin/sounio-loom-sovereign-execution-kernel"
    [[ "$(manifest_value "$sovereign_product" semantic_authority)" == Sounio &&
      "$(manifest_value "$sovereign_product" semantic_action)" == 9042 &&
      "$(manifest_value "$sovereign_product" grant_residency)" == Loom_kernel_memory &&
      "$(manifest_value "$sovereign_product" grant_is_bearer)" == false &&
      "$(manifest_value "$sovereign_product" exported_token)" == false &&
      "$(manifest_value "$sovereign_product" exported_handle)" == false &&
      "$(manifest_value "$sovereign_product" interface_release_authority)" == zero &&
      "$(manifest_value "$sovereign_product" same_uid_peer_isolation)" == true &&
      "$(manifest_value "$sovereign_product" production_activation)" == true &&
      "$(manifest_value "$sovereign_product" exec_attached)" == true &&
      "$(manifest_value "$sovereign_product" semantic_manifest_sha256)" == \
        966f022c98bc7df89ce40a90ede9ec8a9a726499baec0fd21e72f327f286a176 &&
      "$(manifest_value "$sovereign_product" material_manifest_sha256)" == \
        1005da28d4375da8d67fecc4a301c0c6e768902d720952f93e3f82a74fd41f92 &&
      "$(manifest_value "$sovereign_product" sounio_runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_sovereign_runtime_sha256)" ]] ||
      die "installed sovereign execution product is not bound to frozen Sounio action 9042: $runtime_id"
    local sovereign_pair sovereign_path_key sovereign_hash_key
    local sovereign_rel sovereign_expected sovereign_actual
    for sovereign_pair in \
      contract_path:contract_sha256 \
      semantic_manifest_path:semantic_manifest_sha256 \
      material_manifest_path:material_manifest_sha256 \
      sounio_source_path:sounio_source_sha256 \
      sounio_entrypoint_path:sounio_entrypoint_sha256 \
      loom_source_path:loom_source_sha256 \
      exec_source_path:exec_source_sha256 \
      hook_source_path:hook_source_sha256 \
      sovereign_source_path:sovereign_source_sha256 \
      provider_fixture_path:provider_fixture_sha256 \
      c_stub_path:c_stub_sha256 \
      dune_path:dune_sha256 \
      loom_build_path:loom_build_sha256 \
      installer_path:installer_sha256 \
      coord_runtime_path:coord_runtime_sha256 \
      product_gate_path:product_gate_sha256 \
      freeze_gate_path:freeze_gate_sha256; do
      sovereign_path_key="${sovereign_pair%%:*}"
      sovereign_hash_key="${sovereign_pair#*:}"
      sovereign_rel="$(manifest_value "$sovereign_product" "$sovereign_path_key")"
      [[ -n "$sovereign_rel" && "$sovereign_rel" != /* &&
        "$sovereign_rel" != *'..'* ]] ||
        die "installed sovereign product source path is unsafe: $sovereign_rel"
      sovereign_expected="$(manifest_value "$sovereign_product" "$sovereign_hash_key")"
      sovereign_actual="$(sha256sum "$sovereign_capsule/$sovereign_rel" | awk '{print $1}')"
      [[ "$sovereign_actual" == "$sovereign_expected" ]] ||
        die "installed sovereign execution product source drifted: $sovereign_rel"
    done
  fi
  if grep -q '^capability=loom-sovereign-change-kernel-v2$' "$manifest"; then
    local change_capsule="$version_dir/policy/sovereign-change"
    local change_product="$change_capsule/tools/loom/sovereign_material_change_product.runtime.v2"
    [[ -x "$version_dir/bin/sounio-loom-runtime" &&
      -x "$version_dir/bin/sounio-loom-sovereign-change-kernel" &&
      -x "$version_dir/bin/sounio-loom-sovereign-material-change" &&
      -f "$change_product" ]] ||
      die "installed sovereign change product omits Loom or Sounio actions 9043/9044: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" loom_change_runtime_sha256 \
      "$version_dir/bin/sounio-loom-sovereign-change-kernel"
    verify_manifest_binary_sha256 "$manifest" loom_material_change_runtime_sha256 \
      "$version_dir/bin/sounio-loom-sovereign-material-change"
    verify_manifest_binary_sha256 "$manifest" loom_change_manifest_sha256 \
      "$change_capsule/tools/loom/sovereign_change_kernel.freeze.v1"
    verify_manifest_binary_sha256 "$manifest" loom_material_change_manifest_sha256 \
      "$change_capsule/tools/loom/sovereign_material_change.freeze.v2"
    verify_manifest_binary_sha256 "$manifest" loom_material_change_product_sha256 \
      "$change_product"
    [[ "$(manifest_value "$change_product" semantic_authority)" == Sounio &&
      "$(manifest_value "$change_product" producing_language)" == Sounio &&
      "$(manifest_value "$change_product" language_role)" == SEMANTIC_AUTHORITY &&
      "$(manifest_value "$change_product" action)" == 9044 &&
      "$(manifest_value "$change_product" stage)" == CLAIM_READY &&
      "$(manifest_value "$change_product" operational_language)" == OCaml &&
      "$(manifest_value "$change_product" operational_role)" == OPERATIONAL_ATTACHMENT &&
      "$(manifest_value "$change_product" operational_semantic_authority)" == false &&
      "$(manifest_value "$change_product" provider_root_readonly)" == true &&
      "$(manifest_value "$change_product" staging_outside_root)" == true &&
      "$(manifest_value "$change_product" grant_residency)" == Loom_kernel_memory &&
      "$(manifest_value "$change_product" grant_is_bearer)" == false &&
      "$(manifest_value "$change_product" grant_single_use)" == true &&
      "$(manifest_value "$change_product" consume_atomic)" == true &&
      "$(manifest_value "$change_product" exported_token)" == false &&
      "$(manifest_value "$change_product" exported_handle)" == false &&
      "$(manifest_value "$change_product" exact_call_id)" == true &&
      "$(manifest_value "$change_product" exact_patch_hash)" == true &&
      "$(manifest_value "$change_product" exact_worktree_state)" == true &&
      "$(manifest_value "$change_product" authenticated_peer)" == true &&
      "$(manifest_value "$change_product" exact_file_set)" == true &&
      "$(manifest_value "$change_product" write_attached)" == true &&
      "$(manifest_value "$change_product" commit_attached)" == true &&
      "$(manifest_value "$change_product" ci_attached)" == true &&
      "$(manifest_value "$change_product" ci_policy)" == consume-not-reinterpret &&
      "$(manifest_value "$change_product" policy_executed_by_ci)" == false &&
      "$(manifest_value "$change_product" parity_open)" == true &&
      "$(manifest_value "$change_product" parity_executed)" == false &&
      "$(manifest_value "$change_product" parity_receipts_semantic_authority)" == false &&
      "$(manifest_value "$change_product" claim_ready)" == true ]] ||
      die "installed sovereign change product is not bound to CLAIM_READY action 9044: $runtime_id"
    [[ "$(manifest_value "$change_product" parent_sounio_runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_change_runtime_sha256)" &&
      "$(manifest_value "$change_product" sounio_runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_material_change_runtime_sha256)" &&
      "$(manifest_value "$change_product" loom_runtime_sha256)" == \
        "$(manifest_value "$manifest" loom_runtime_sha256)" ]] ||
      die "installed sovereign change product runtime hashes diverged: $runtime_id"
    local change_pair change_path_key change_hash_key change_rel
    local change_expected change_actual
    for change_pair in \
      parent_manifest_path:parent_manifest_sha256 \
      material_manifest_path:material_manifest_sha256 \
      sounio_source_path:sounio_source_sha256 \
      loom_change_source_path:loom_change_source_sha256 \
      loom_source_path:loom_source_sha256 \
      hook_source_path:hook_source_sha256 \
      c_stub_path:c_stub_sha256 \
      provider_fixture_path:provider_fixture_sha256 \
      dune_path:dune_sha256 \
      loom_build_path:loom_build_sha256 \
      installer_path:installer_sha256 \
      operational_gate_path:operational_gate_sha256 \
      ci_entrypoint_path:ci_entrypoint_sha256; do
      change_path_key="${change_pair%%:*}"
      change_hash_key="${change_pair#*:}"
      change_rel="$(manifest_value "$change_product" "$change_path_key")"
      [[ -n "$change_rel" && "$change_rel" != /* && "$change_rel" != *'..'* ]] ||
        die "installed sovereign change source path is unsafe: $change_rel"
      change_expected="$(manifest_value "$change_product" "$change_hash_key")"
      change_actual="$(sha256sum "$change_capsule/$change_rel" | awk '{print $1}')"
      [[ "$change_actual" == "$change_expected" ]] ||
        die "installed sovereign change source drifted: $change_rel"
    done
  fi
  if grep -q '^capability=loom-truthful-lane-health-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-lane-health-runtime" && \
      -x "$version_dir/bin/sounio-loom-lane-health-parity-runtime" ]] || \
      die "installed runtime declares truthful lane health but omits its OCaml realization or frozen Sounio executables: $runtime_id"
    [[ "$(manifest_value "$manifest" loom_lane_health_semantics_sha256)" == \
      5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f ]] || \
      die "installed truthful lane health is not bound to the frozen Sounio semantics: $runtime_id"
  fi
  if grep -q '^capability=loom-native-hook-binary-attestation-v1$' "$manifest"; then
    grep -q '^capability=loom-native-agent-hook-v1$' "$manifest" || \
      die "installed runtime declares native hook attestation without the native hook: $runtime_id"
    verify_manifest_binary_sha256 "$manifest" coord_runtime_sha256 \
      "$version_dir/bin/sounio-coord-runtime"
    verify_manifest_binary_sha256 "$manifest" loom_runtime_sha256 \
      "$version_dir/bin/sounio-loom-runtime"
  fi
  if grep -q '^capability=loom-native-sounio-continuity-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-continuity-runtime" ]] || \
      die "installed runtime declares native Sounio continuity but omits its adapter: $runtime_id"
  fi
  if grep -q '^capability=loom-durable-obligation-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-obligation-runtime" ]] || \
      die "installed runtime declares durable obligations but omits Loom or native Sounio frame 9007: $runtime_id"
    ensure_obligation_activation
  fi
  if grep -q '^capability=loom-epistemic-machine-v0$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-epistemic-runtime" ]] || \
      die "installed runtime declares the epistemic machine but omits Loom or native Sounio frame 9008: $runtime_id"
  fi
  if grep -q '^capability=loom-attention-compiler-v0$' "$manifest"; then
    grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-attention-runtime" ]] || \
      die "installed runtime declares the attention compiler without the epistemic machine, Loom, or native Sounio frame 9009: $runtime_id"
  fi
  if grep -q '^capability=loom-pareto-portfolio-attention-v0$' "$manifest"; then
    grep -q '^capability=loom-attention-compiler-v0$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-portfolio-runtime" ]] || \
      die "installed runtime declares Pareto portfolio attention without the attention compiler, epistemic machine, Loom, or native Sounio frame 9010: $runtime_id"
  fi
  if grep -q '^capability=loom-robust-contingent-policy-v0$' "$manifest"; then
    grep -q '^capability=loom-pareto-portfolio-attention-v0$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-contingent-runtime" ]] || \
      die "installed runtime declares robust contingent policies without portfolio attention, the epistemic machine, Loom, or native Sounio frame 9011: $runtime_id"
  fi
  if grep -q '^capability=loom-atomic-outcome-resource-handoff-v0$' "$manifest"; then
    grep -q '^capability=loom-robust-contingent-policy-v0$' "$manifest" || \
      die "installed runtime declares atomic outcome handoff without robust contingent policies: $runtime_id"
  fi
  if grep -q '^capability=loom-signed-outcome-authority-v0$' "$manifest"; then
    grep -q '^capability=loom-robust-contingent-policy-v0$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-outcome-authority-runtime" ]] || \
      die "installed runtime declares signed outcome authority without contingent policies, the epistemic machine, Loom, or native Sounio frame 9012: $runtime_id"
  fi
  if grep -Eq '^capability=loom-(linear-outcome-evidence|journal-head-bound-consume)-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-signed-outcome-authority-v0$' "$manifest" || \
      die "installed runtime declares derived outcome-evidence capabilities without signed outcome authority: $runtime_id"
  fi
  if grep -q '^capability=loom-external-witness-mesh-v0$' "$manifest"; then
    grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-mesh-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares the external witness mesh without the epistemic machine, Loom, OpenSSL, or native Sounio frame 9013: $runtime_id"
  fi
  if grep -Eq '^capability=loom-(quorum-intersection-checkpoint|rollback-detection-through-checkpoint)-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-external-witness-mesh-v0$' "$manifest" || \
      die "installed runtime declares derived witness-mesh capabilities without the external witness mesh: $runtime_id"
  fi
  if grep -q '^capability=loom-external-witness-mesh-v1$' "$manifest"; then
    grep -q '^capability=loom-external-witness-mesh-v0$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-mesh-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-mesh-v1-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares witness mesh v1 without v0 compatibility, the epistemic machine, Loom, OpenSSL, or native Sounio frame 9014: $runtime_id"
  fi
  if grep -Eq '^capability=loom-(three-of-four-witness-quorum|one-dishonest-honest-intersection|one-fault-anchor-and-verify-availability)-v1$' \
      "$manifest"; then
    grep -q '^capability=loom-external-witness-mesh-v1$' "$manifest" || \
      die "installed runtime declares derived witness-mesh-v1 capabilities without the v1 mesh: $runtime_id"
  fi
  if grep -q '^capability=loom-proof-carrying-witness-epoch-handoff-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-external-witness-mesh-v1$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-mesh-v1-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-epoch-handoff-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares proof-carrying witness epoch handoff without witness mesh v1, the epistemic machine, Loom, OpenSSL, or native Sounio frame 9015: $runtime_id"
  fi
  if grep -Eq '^capability=loom-(joint-old-new-witness-quorum|atomic-witness-epoch-activation|witness-epoch-crash-recovery)-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-proof-carrying-witness-epoch-handoff-v0$' \
      "$manifest" || \
      die "installed runtime declares derived witness-epoch capabilities without the proof-carrying handoff: $runtime_id"
  fi
  if grep -q '^capability=loom-external-epoch-transparency-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-proof-carrying-witness-epoch-handoff-v0$' \
      "$manifest" && \
      grep -q '^capability=loom-external-witness-mesh-v1$' "$manifest" && \
      grep -q '^capability=loom-epistemic-machine-v0$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-mesh-v1-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-epoch-handoff-runtime" && \
        -x "$version_dir/bin/sounio-loom-witness-epoch-transparency-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares external epoch transparency without frame 9015, witness mesh v1, the epistemic machine, Loom, OpenSSL, or native Sounio frame 9016: $runtime_id"
  fi
  if grep -Eq '^capability=loom-(materialized-merkle-prefix-verification|witnessed-split-view-refusal|latest-quorum-witnessed-epoch-rollback-refusal|transparency-unreachable-fail-closed)-v0$' \
      "$manifest"; then
    grep -q '^capability=loom-external-epoch-transparency-v0$' "$manifest" || \
      die "installed runtime declares derived epoch-transparency capabilities without the root transparency capability: $runtime_id"
  fi
  if grep -q '^capability=loom-recoverable-control-service-v1$' "$manifest"; then
    grep -q '^capability=loom-durable-obligation-v1$' "$manifest" &&
      grep -q '^capability=loom-post-activation-request-bridge-v1$' "$manifest" &&
      [[ -x /usr/bin/setsid ]] ||
      die "installed runtime declares recoverable control service without durable bridge or setsid: $runtime_id"
  fi
  if grep -q '^capability=loom-signed-continuity-receipt-v2$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-loom-runtime" && \
      -x "$version_dir/bin/sounio-loom-continuity-runtime" && \
      -x /usr/bin/openssl ]] || \
      die "installed runtime declares signed continuity but omits Loom, its adapter, or OpenSSL: $runtime_id"
  fi
  if grep -q '^capability=loom-principal-independence-v1$' "$manifest"; then
    grep -q '^capability=loom-signed-continuity-receipt-v2$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-continuity-runtime" ]] || \
      die "installed runtime declares principal independence without signed Loom and native Sounio admission: $runtime_id"
  fi
  if grep -q '^capability=loom-independent-measurement-v1$' "$manifest"; then
    grep -q '^capability=loom-principal-independence-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-continuity-runtime" ]] || \
      die "installed runtime declares independent measurement without principal independence and native Sounio admission: $runtime_id"
  fi
  if grep -q '^capability=loom-observation-authority-v1$' "$manifest"; then
    grep -q '^capability=loom-independent-measurement-v1$' "$manifest" && \
      grep -q '^capability=loom-signed-continuity-receipt-v2$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-continuity-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares observation authority without signed Loom, independent measurement, native Sounio admission, and OpenSSL: $runtime_id"
  fi
  if grep -q '^capability=loom-journal-authority-quorum-v1$' "$manifest"; then
    grep -q '^capability=loom-observation-authority-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-loom-runtime" && \
        -x "$version_dir/bin/sounio-loom-continuity-runtime" && \
        -x /usr/bin/openssl ]] || \
      die "installed runtime declares journal quorum without observation authority, native Sounio admission, and OpenSSL: $runtime_id"
  fi
  if grep -q '^capability=loom-cross-node-replay-v1$' "$manifest"; then
    grep -q '^capability=loom-signed-continuity-receipt-v2$' "$manifest" && \
      grep -q '^capability=loom-separate-pod-inbox-replay-v1$' "$manifest" || \
      die "installed runtime declares cross-node replay without signed separate-Pod continuity: $runtime_id"
  fi
  if grep -q '^capability=fleet-launcher-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares the fleet launcher but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=fleet-proven-exit-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares proven-exit recovery but omits its launcher: $runtime_id"
  fi
  if grep -q '^capability=fleet-home-isolation-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares fleet HOME isolation but omits its launcher: $runtime_id"
  fi
  if grep -q '^capability=fleet-presentation-follow-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-agent-runtime" ]] || \
      die "installed runtime declares presentation following but omits its launcher: $runtime_id"
  fi
  if grep -q '^capability=fleet-event-log-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-runtime" ]] || \
      die "installed runtime declares fleet reconciliation but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=fleet-tla-model-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-tla-sabotage" && \
      -f "$version_dir/formal/SounioFleet.tla" && \
      -f "$version_dir/formal/SounioFleet.cfg" ]] || \
      die "installed runtime declares the TLA+ fleet model but omits its bundle: $runtime_id"
  fi
  if grep -q '^capability=fleet-trace-refinement-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-trace-verify" ]] || \
      die "installed runtime declares fleet trace refinement but omits its verifier: $runtime_id"
  fi
  if grep -q '^capability=fleet-temporal-authority-v1$' "$manifest"; then
    [[ -x "$version_dir/bin/sounio-fleet-runtime" && \
      -x "$version_dir/bin/sounio-fleet-trace-verify" ]] || \
      die "installed runtime declares temporal fleet authority but omits its implementation: $runtime_id"
  fi
  if grep -q '^capability=fleet-recovery-start-only-v1$' "$manifest"; then
    grep -q '^capability=fleet-temporal-authority-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-fleet-runtime" ]] || \
      die "installed runtime declares start-only fleet recovery without temporal authority and its reconciler: $runtime_id"
  fi
  if grep -q '^capability=fleet-recovery-directory-v1$' "$manifest"; then
    grep -q '^capability=fleet-recovery-start-only-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-fleet-runtime" ]] || \
      die "installed runtime declares recovery directories without bounded start-only recovery: $runtime_id"
  fi
  if grep -q '^capability=fleet-recovery-latch-trace-v1$' "$manifest"; then
    grep -q '^capability=fleet-recovery-directory-v1$' "$manifest" && \
      [[ -x "$version_dir/bin/sounio-fleet-trace-verify" ]] || \
      die "installed runtime declares recovery-latch refinement without its directory authority and verifier: $runtime_id"
  fi
  control_state="${SOUNIO_COORD_DIR:-$GIT_COMMON_DIR/sounio-coord-state}"
  if [[ -L "$RUNTIME_ROOT/current" ]]; then
    previous_target="$(readlink "$RUNTIME_ROOT/current" 2>/dev/null || true)"
    previous_bundle="$(readlink -f "$RUNTIME_ROOT/current" 2>/dev/null || true)"
    previous_runtime="$previous_bundle/bin/sounio-coord-runtime"
    if [[ -x "$previous_runtime" ]]; then
      control_status="$(
        SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$control_state" \
          "$previous_runtime" obligation-supervisor-status 2>/dev/null || true
      )"
      if grep -q '^LOOM_OBLIGATION_SUPERVISOR_STATUS state=live ' \
        <<< "$control_status"; then
        control_service_was_live=1
      fi
    fi
  fi
  [[ ! -e "$RUNTIME_ROOT/current" || -L "$RUNTIME_ROOT/current" ]] || \
    die "refusing to replace non-symlink runtime path: $RUNTIME_ROOT/current"
  link_tmp="$RUNTIME_ROOT/.current.$$.$RANDOM"
  ln -s "versions/$runtime_id" "$link_tmp"
  mv -Tf "$link_tmp" "$RUNTIME_ROOT/current"
  if ((control_service_was_live)); then
    set +e
    ensure_output="$(
      SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$control_state" \
        "$version_dir/bin/sounio-coord-runtime" obligation-supervisor-ensure \
        --interval-seconds 2 9>&- 2>&1
    )"
    ensure_rc=$?
    set -e
    if ((ensure_rc != 0)); then
      link_tmp="$RUNTIME_ROOT/.current-rollback.$$.$RANDOM"
      ln -s "$previous_target" "$link_tmp"
      mv -Tf "$link_tmp" "$RUNTIME_ROOT/current"
      if [[ -x "$previous_runtime" ]]; then
        SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$control_state" \
          "$previous_runtime" obligation-supervisor-ensure --interval-seconds 2 \
          9>&- >/dev/null 2>&1 || true
      fi
      die "runtime activation could not assume the control service and was rolled back: $ensure_output"
    fi
    printf '%s\n' "$ensure_output"
  fi
  printf 'ACTIVATED runtime_id=%s protocol=%s path=%s\n' \
    "$runtime_id" "$protocol" "$version_dir"
}

WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$WORKTREE" ]] || die "run this installer from a Git worktree"
WORKTREE="$(cd "$WORKTREE" && pwd -P)"
GIT_COMMON_DIR="$(git -C "$WORKTREE" rev-parse --git-common-dir 2>/dev/null || true)"
[[ -n "$GIT_COMMON_DIR" ]] || die "cannot resolve the shared Git directory"
case "$GIT_COMMON_DIR" in
  /*) ;;
  *) GIT_COMMON_DIR="$(cd "$WORKTREE/$GIT_COMMON_DIR" && pwd -P)" ;;
esac

SOURCE_ROOT="$WORKTREE"
RUNTIME_ROOT="${SOUNIO_COORD_RUNTIME_DIR:-$GIT_COMMON_DIR/sounio-coord-runtime}"
action=install
activate_id=''
while (($#)); do
  case "$1" in
    --source-root) (($# >= 2)) || die "$1 requires a value"; SOURCE_ROOT="$2"; shift 2 ;;
    --runtime-dir) (($# >= 2)) || die "$1 requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --activate) (($# >= 2)) || die "$1 requires a value"; action=activate; activate_id="$2"; shift 2 ;;
    --list) action=list; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown installer option: $1" ;;
  esac
done

SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
mkdir -p "$RUNTIME_ROOT/versions"
RUNTIME_ROOT="$(cd "$RUNTIME_ROOT" && pwd -P)"
exec 9>"$RUNTIME_ROOT/.install.lock"
flock 9

if [[ "$action" == list ]]; then
  current=''
  [[ ! -e "$RUNTIME_ROOT/current" ]] || current="$(basename "$(readlink -f "$RUNTIME_ROOT/current")")"
  version_paths=("$RUNTIME_ROOT"/versions/*)
  for version_dir in "${version_paths[@]}"; do
    [[ -d "$version_dir" && -f "$version_dir/manifest" ]] || continue
    runtime_id="$(basename "$version_dir")"
    marker=no
    [[ "$runtime_id" != "$current" ]] || marker=yes
    printf 'RUNTIME runtime_id=%s current=%s protocol=%s runtime_version=%s source_sha=%s\n' \
      "$runtime_id" "$marker" \
      "$(manifest_value "$version_dir/manifest" protocol_version)" \
      "$(manifest_value "$version_dir/manifest" runtime_version)" \
      "$(manifest_value "$version_dir/manifest" source_sha)"
  done
  exit 0
fi

if [[ "$action" == activate ]]; then
  activate_runtime "$activate_id"
  exit 0
fi

installer_source="$SOURCE_ROOT/scripts/dev/install_sounio_coord_runtime.sh"
runtime_source="$SOURCE_ROOT/scripts/dev/sounio_coord_runtime.sh"
hook_source="$SOURCE_ROOT/scripts/dev/sounio_coord_agent_hook_runtime.py"
causal_source="$SOURCE_ROOT/scripts/dev/sounio_coord_causal_runtime.py"
agentd_source="$SOURCE_ROOT/scripts/dev/sounio_coord_agentd.py"
fleet_source="$SOURCE_ROOT/scripts/dev/sounio_coord_fleet.py"
fleetd_source="$SOURCE_ROOT/scripts/dev/sounio_coord_fleetd.py"
fleet_model_source="$SOURCE_ROOT/formal/tla/SounioFleet.tla"
fleet_model_config="$SOURCE_ROOT/formal/tla/SounioFleet.cfg"
fleet_model_generator="$SOURCE_ROOT/scripts/dev/sounio_fleet_tla_sabotage.py"
fleet_trace_verifier="$SOURCE_ROOT/scripts/dev/sounio_fleet_trace_verify.py"
loom_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom.sh"
loom_language_authority_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_language_authority.sh"
loom_custody_transfer_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_custody_transfer.sh"
loom_execution_outcome_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_execution_outcome.sh"
loom_lane_health_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_lane_health.sh"
loom_lane_health_parity_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_lane_health_parity.sh"
loom_continuity_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_continuity_adapter.sh"
loom_obligation_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_obligation_adapter.sh"
loom_epistemic_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_epistemic_adapter.sh"
loom_attention_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_attention_adapter.sh"
loom_portfolio_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_portfolio_attention_adapter.sh"
loom_contingent_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_contingent_policy_adapter.sh"
loom_outcome_authority_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_outcome_authority_adapter.sh"
loom_witness_mesh_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_witness_mesh_adapter.sh"
loom_witness_mesh_v1_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_witness_mesh_v1_adapter.sh"
loom_witness_epoch_handoff_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_witness_epoch_handoff_adapter.sh"
loom_witness_epoch_transparency_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_witness_epoch_transparency_adapter.sh"
loom_project="$SOURCE_ROOT/tools/loom"
loom_language_authority_entrypoint="$SOURCE_ROOT/tools/loom/language_authority_main.sio"
loom_language_authority_module="$SOURCE_ROOT/stdlib/coordination/loom_language_authority.sio"
loom_language_authority_freeze="$SOURCE_ROOT/tools/loom/language_authority.freeze.v1"
loom_custody_transfer_entrypoint="$SOURCE_ROOT/tools/loom/custody_transfer_main.sio"
loom_custody_transfer_module="$SOURCE_ROOT/stdlib/coordination/loom_custody_transfer.sio"
loom_custody_transfer_freeze="$SOURCE_ROOT/tools/loom/custody_transfer.freeze.v1"
loom_execution_outcome_entrypoint="$SOURCE_ROOT/tools/loom/execution_outcome_main.sio"
loom_execution_outcome_module="$SOURCE_ROOT/stdlib/coordination/loom_execution_outcome_authority.sio"
loom_execution_outcome_freeze="$SOURCE_ROOT/tools/loom/execution_outcome.freeze.v1"
loom_lane_health_entrypoint="$SOURCE_ROOT/tools/loom/lane_health_main.sio"
loom_lane_health_parity_entrypoint="$SOURCE_ROOT/tools/loom/lane_health_parity_main.sio"
loom_lane_health_module="$SOURCE_ROOT/stdlib/coordination/loom_lane_health.sio"
loom_lane_health_freeze="$SOURCE_ROOT/tools/loom/lane_health.freeze.v1"
loom_lane_health_ocaml_receipt="$SOURCE_ROOT/tools/loom/lane_health.ocaml.v1"
loom_sha256_module="$SOURCE_ROOT/stdlib/crypto/sha256.sio"
loom_continuity_entrypoint="$SOURCE_ROOT/tools/loom/continuity_adapter_main.sio"
loom_continuity_module="$SOURCE_ROOT/stdlib/coordination/loom_continuity.sio"
loom_obligation_entrypoint="$SOURCE_ROOT/tools/loom/obligation_adapter_main.sio"
loom_obligation_module="$SOURCE_ROOT/stdlib/coordination/loom_obligation.sio"
loom_epistemic_entrypoint="$SOURCE_ROOT/tools/loom/epistemic_adapter_main.sio"
loom_epistemic_module="$SOURCE_ROOT/stdlib/coordination/loom_epistemic_machine.sio"
loom_attention_entrypoint="$SOURCE_ROOT/tools/loom/attention_adapter_main.sio"
loom_attention_module="$SOURCE_ROOT/stdlib/coordination/loom_attention_compiler.sio"
loom_portfolio_entrypoint="$SOURCE_ROOT/tools/loom/portfolio_attention_adapter_main.sio"
loom_portfolio_module="$SOURCE_ROOT/stdlib/coordination/loom_portfolio_attention.sio"
loom_contingent_entrypoint="$SOURCE_ROOT/tools/loom/contingent_policy_adapter_main.sio"
loom_contingent_module="$SOURCE_ROOT/stdlib/coordination/loom_contingent_policy.sio"
loom_outcome_authority_entrypoint="$SOURCE_ROOT/tools/loom/outcome_authority_adapter_main.sio"
loom_outcome_authority_module="$SOURCE_ROOT/stdlib/coordination/loom_outcome_authority.sio"
loom_witness_mesh_entrypoint="$SOURCE_ROOT/tools/loom/witness_mesh_adapter_main.sio"
loom_witness_mesh_module="$SOURCE_ROOT/stdlib/coordination/loom_witness_mesh.sio"
loom_witness_mesh_v1_entrypoint="$SOURCE_ROOT/tools/loom/witness_mesh_v1_adapter_main.sio"
loom_witness_mesh_v1_module="$SOURCE_ROOT/stdlib/coordination/loom_witness_mesh_v1.sio"
loom_witness_epoch_handoff_entrypoint="$SOURCE_ROOT/tools/loom/witness_epoch_handoff_adapter_main.sio"
loom_witness_epoch_handoff_module="$SOURCE_ROOT/stdlib/coordination/loom_witness_epoch_handoff.sio"
loom_witness_epoch_transparency_entrypoint="$SOURCE_ROOT/tools/loom/epoch_transparency_adapter_main.sio"
loom_witness_epoch_transparency_module="$SOURCE_ROOT/stdlib/coordination/loom_witness_epoch_transparency.sio"
loom_product_activation_garden="$SOURCE_ROOT/tools/loom/GARDEN_KERNEL_PEER_ACTIVATION_CAPSULE_V1.md"
loom_product_activation_source="$SOURCE_ROOT/stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio"
loom_product_activation_entrypoint="$SOURCE_ROOT/tools/loom/kernel_peer_activation_capsule_authority_main.sio"
loom_product_activation_action_freeze="$SOURCE_ROOT/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
loom_product_activation_operational_freeze="$SOURCE_ROOT/tools/loom/kernel_peer_activation_capsule.runtime.v1"
loom_product_activation_projection="$SOURCE_ROOT/tools/loom/kernel_peer_activation_capsule.current.v1"
loom_product_activation_resident_freeze="$SOURCE_ROOT/tools/loom/resident_membrane.runtime.v5"
loom_product_activation_parent_9023="$SOURCE_ROOT/tools/loom/subprocess_membrane.freeze.v1"
loom_product_activation_parent_9024="$SOURCE_ROOT/tools/loom/resident_authority.freeze.v1"
loom_product_activation_parent_9025="$SOURCE_ROOT/tools/loom/effect_closure_authority.freeze.v1"
loom_product_activation_parent_9029="$SOURCE_ROOT/tools/loom/kernel_invocation_cell_authority.freeze.v1"
loom_product_activation_parent_9030="$SOURCE_ROOT/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
loom_product_activation_parent_9025_v13="$SOURCE_ROOT/tools/loom/kernel_peer_material_judgment_v13.freeze.v1"
loom_product_activation_resident_v4="$SOURCE_ROOT/tools/loom/resident_membrane.runtime.v4"
loom_product_activation_dispatcher="$SOURCE_ROOT/tools/loom/resident_membrane_v5_main.sio"
loom_product_activation_build="$SOURCE_ROOT/scripts/dev/build_sounio_loom_resident_membrane_v5.sh"
loom_product_activation_gate="$SOURCE_ROOT/scripts/ci/sounio_loom_resident_transport_v5_selftest.sh"
loom_product_exec_ingress_freeze="$SOURCE_ROOT/tools/loom/product_exec_ingress_dark.runtime.v1"
loom_product_exec_ingress_contract="$SOURCE_ROOT/tools/loom/PRODUCT_EXEC_INGRESS_DARK_ATTACHMENT_V1.md"
loom_product_exec_ingress_evidence="$SOURCE_ROOT/tools/loom/evidence/loom-product-exec-ingress-dark-v1-20260829.txt"
loom_product_exec_ingress_sources=(
  "$SOURCE_ROOT/tools/loom/src/loom_exec_ingress.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_hook.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_membrane.ml"
  "$SOURCE_ROOT/tools/loom/src/loom.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_pty_stubs.c"
  "$SOURCE_ROOT/tools/loom/src/dune"
)
loom_sovereign_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh"
loom_sovereign_source="$SOURCE_ROOT/stdlib/coordination/loom_sovereign_execution_kernel_authority.sio"
loom_sovereign_entrypoint="$SOURCE_ROOT/tools/loom/sovereign_execution_kernel_authority_main.sio"
loom_sovereign_semantic_freeze="$SOURCE_ROOT/tools/loom/sovereign_execution_kernel.freeze.v1"
loom_sovereign_material_freeze="$SOURCE_ROOT/tools/loom/sovereign_execution_kernel_material.runtime.v1"
loom_sovereign_product_freeze="$SOURCE_ROOT/tools/loom/sovereign_execution_kernel_product.runtime.v1"
loom_sovereign_product_contract="$SOURCE_ROOT/tools/loom/SOVEREIGN_EXECUTION_KERNEL_PRODUCT_ATTACHMENT_V1.md"
loom_sovereign_product_evidence="$SOURCE_ROOT/tools/loom/evidence/loom-sovereign-execution-kernel-product-v1-20260831.txt"
loom_sovereign_product_gate="$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_execution_kernel_product_selftest.sh"
loom_sovereign_product_freeze_gate="$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_execution_kernel_product_freeze_selftest.sh"
loom_sovereign_sources=(
  "$SOURCE_ROOT/tools/loom/src/loom.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_exec.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_hook.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_sovereign_exec.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_sovereign_provider_fixture.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_pty_stubs.c"
  "$SOURCE_ROOT/tools/loom/src/dune"
  "$SOURCE_ROOT/scripts/dev/build_sounio_loom.sh"
  "$SOURCE_ROOT/scripts/dev/install_sounio_coord_runtime.sh"
  "$SOURCE_ROOT/scripts/dev/sounio_coord_runtime.sh"
  "$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_execution_kernel_product_selftest.sh"
  "$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_execution_kernel_product_freeze_selftest.sh"
)
loom_change_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_sovereign_change_kernel.sh"
loom_material_change_build_source="$SOURCE_ROOT/scripts/dev/build_sounio_loom_sovereign_material_change.sh"
loom_change_source="$SOURCE_ROOT/stdlib/coordination/loom_sovereign_change_kernel_authority.sio"
loom_change_entrypoint="$SOURCE_ROOT/tools/loom/sovereign_change_kernel_authority_main.sio"
loom_change_freeze="$SOURCE_ROOT/tools/loom/sovereign_change_kernel.freeze.v1"
loom_material_change_source="$SOURCE_ROOT/stdlib/coordination/loom_sovereign_material_change_authority.sio"
loom_material_change_entrypoint="$SOURCE_ROOT/tools/loom/sovereign_material_change_authority_main.sio"
loom_material_change_freeze="$SOURCE_ROOT/tools/loom/sovereign_material_change.freeze.v2"
loom_material_change_product="$SOURCE_ROOT/tools/loom/sovereign_material_change_product.runtime.v2"
loom_material_change_evidence="$SOURCE_ROOT/tools/loom/evidence/loom-sovereign-material-change-product-v2-20260831.txt"
loom_change_operational_gate="$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_change_kernel_operational_selftest.sh"
loom_change_ci_admit="$SOURCE_ROOT/scripts/ci/sounio_loom_sovereign_change_receipt_admit.sh"
loom_change_sources=(
  "$SOURCE_ROOT/tools/loom/src/loom_change.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_change_stubs.c"
  "$SOURCE_ROOT/tools/loom/src/loom_change_provider_fixture.ml"
  "$SOURCE_ROOT/tools/loom/src/loom.ml"
  "$SOURCE_ROOT/tools/loom/src/loom_hook.ml"
  "$SOURCE_ROOT/tools/loom/src/dune"
)
[[ -x "$installer_source" ]] || die "runtime installer source missing or not executable: $installer_source"
[[ -x "$runtime_source" ]] || die "runtime source missing or not executable: $runtime_source"
[[ -f "$hook_source" ]] || die "hook runtime source missing: $hook_source"
[[ -x "$causal_source" ]] || die "causal runtime source missing or not executable: $causal_source"
[[ -x "$agentd_source" ]] || die "agent supervisor source missing or not executable: $agentd_source"
[[ -x "$fleet_source" ]] || die "fleet launcher source missing or not executable: $fleet_source"
[[ -x "$fleetd_source" ]] || die "fleet reconciler source missing or not executable: $fleetd_source"
[[ -f "$fleet_model_source" ]] || die "fleet TLA+ model missing: $fleet_model_source"
[[ -f "$fleet_model_config" ]] || die "fleet TLC config missing: $fleet_model_config"
[[ -x "$fleet_model_generator" ]] || \
  die "fleet model sabotage generator missing or not executable: $fleet_model_generator"
[[ -x "$fleet_trace_verifier" ]] || \
  die "fleet trace verifier missing or not executable: $fleet_trace_verifier"
[[ -x "$loom_build_source" ]] || die "Loom build entrypoint missing or not executable: $loom_build_source"
[[ -x "$loom_language_authority_build_source" ]] || \
  die "Loom language-authority build entrypoint missing or not executable: $loom_language_authority_build_source"
[[ -x "$loom_custody_transfer_build_source" ]] || \
  die "Loom custody-transfer build entrypoint missing or not executable: $loom_custody_transfer_build_source"
[[ -x "$loom_execution_outcome_build_source" ]] || \
  die "Loom execution-outcome build entrypoint missing or not executable: $loom_execution_outcome_build_source"
[[ -x "$loom_lane_health_build_source" && \
  -x "$loom_lane_health_parity_build_source" ]] || \
  die "Loom lane-health build entrypoints are incomplete"
[[ -x "$loom_continuity_build_source" ]] || \
  die "Loom continuity build entrypoint missing or not executable: $loom_continuity_build_source"
[[ -x "$loom_obligation_build_source" ]] || \
  die "Loom obligation build entrypoint missing or not executable: $loom_obligation_build_source"
[[ -x "$loom_epistemic_build_source" ]] || \
  die "Loom epistemic build entrypoint missing or not executable: $loom_epistemic_build_source"
[[ -x "$loom_attention_build_source" ]] || \
  die "Loom attention build entrypoint missing or not executable: $loom_attention_build_source"
[[ -x "$loom_portfolio_build_source" ]] || \
  die "Loom portfolio build entrypoint missing or not executable: $loom_portfolio_build_source"
[[ -x "$loom_contingent_build_source" ]] || \
  die "Loom contingent-policy build entrypoint missing or not executable: $loom_contingent_build_source"
[[ -x "$loom_outcome_authority_build_source" ]] || \
  die "Loom outcome-authority build entrypoint missing or not executable: $loom_outcome_authority_build_source"
[[ -x "$loom_witness_mesh_build_source" ]] || \
  die "Loom witness-mesh build entrypoint missing or not executable: $loom_witness_mesh_build_source"
[[ -x "$loom_witness_mesh_v1_build_source" ]] || \
  die "Loom witness-mesh-v1 build entrypoint missing or not executable: $loom_witness_mesh_v1_build_source"
[[ -x "$loom_witness_epoch_handoff_build_source" ]] || \
  die "Loom witness-epoch-handoff build entrypoint missing or not executable: $loom_witness_epoch_handoff_build_source"
[[ -x "$loom_witness_epoch_transparency_build_source" ]] || \
  die "Loom witness-epoch-transparency build entrypoint missing or not executable: $loom_witness_epoch_transparency_build_source"
[[ -x "$loom_sovereign_build_source" && -f "$loom_sovereign_source" &&
  -f "$loom_sovereign_entrypoint" && -f "$loom_sovereign_semantic_freeze" &&
  -f "$loom_sovereign_material_freeze" && -f "$loom_sovereign_product_freeze" &&
  -f "$loom_sovereign_product_contract" && -f "$loom_sovereign_product_evidence" &&
  -x "$loom_sovereign_product_gate" && -x "$loom_sovereign_product_freeze_gate" ]] ||
  die "Loom sovereign execution product source bundle is incomplete"
[[ -f "$loom_continuity_entrypoint" && -f "$loom_continuity_module" ]] || \
  die "Loom native Sounio continuity source bundle is incomplete"
[[ -f "$loom_language_authority_entrypoint" && \
  -f "$loom_language_authority_module" && \
  -f "$loom_language_authority_freeze" ]] || \
  die "Loom frozen Sounio language-authority source bundle is incomplete"
[[ -f "$loom_custody_transfer_entrypoint" && \
  -f "$loom_custody_transfer_module" && \
  -f "$loom_custody_transfer_freeze" ]] || \
  die "Loom frozen Sounio custody-transfer source bundle is incomplete"
[[ -f "$loom_execution_outcome_entrypoint" && \
  -f "$loom_execution_outcome_module" && \
  -f "$loom_execution_outcome_freeze" ]] || \
  die "Loom frozen Sounio execution-outcome source bundle is incomplete"
[[ -f "$loom_lane_health_entrypoint" && \
  -f "$loom_lane_health_parity_entrypoint" && \
  -f "$loom_lane_health_module" && -f "$loom_lane_health_freeze" && \
  -f "$loom_lane_health_ocaml_receipt" && -f "$loom_sha256_module" ]] || \
  die "Loom frozen Sounio lane-health source bundle is incomplete"
[[ -f "$loom_obligation_entrypoint" && -f "$loom_obligation_module" ]] || \
  die "Loom native Sounio obligation source bundle is incomplete"
[[ -f "$loom_epistemic_entrypoint" && -f "$loom_epistemic_module" ]] || \
  die "Loom native Sounio epistemic source bundle is incomplete"
[[ -f "$loom_attention_entrypoint" && -f "$loom_attention_module" ]] || \
  die "Loom native Sounio attention source bundle is incomplete"
[[ -f "$loom_portfolio_entrypoint" && -f "$loom_portfolio_module" ]] || \
  die "Loom native Sounio portfolio source bundle is incomplete"
[[ -f "$loom_contingent_entrypoint" && -f "$loom_contingent_module" ]] || \
  die "Loom native Sounio contingent-policy source bundle is incomplete"
[[ -f "$loom_outcome_authority_entrypoint" && \
  -f "$loom_outcome_authority_module" ]] || \
  die "Loom native Sounio outcome-authority source bundle is incomplete"
[[ -f "$loom_witness_mesh_entrypoint" && \
  -f "$loom_witness_mesh_module" ]] || \
  die "Loom native Sounio witness-mesh source bundle is incomplete"
[[ -f "$loom_witness_mesh_v1_entrypoint" && \
  -f "$loom_witness_mesh_v1_module" ]] || \
  die "Loom native Sounio witness-mesh-v1 source bundle is incomplete"
[[ -f "$loom_witness_epoch_handoff_entrypoint" && \
  -f "$loom_witness_epoch_handoff_module" ]] || \
  die "Loom native Sounio witness-epoch-handoff source bundle is incomplete"
[[ -f "$loom_witness_epoch_transparency_entrypoint" && \
  -f "$loom_witness_epoch_transparency_module" ]] || \
  die "Loom native Sounio witness-epoch-transparency source bundle is incomplete"
for product_activation_source in \
  "$loom_product_activation_garden" \
  "$loom_product_activation_source" \
  "$loom_product_activation_entrypoint" \
  "$loom_product_activation_action_freeze" \
  "$loom_product_activation_operational_freeze" \
  "$loom_product_activation_projection" \
  "$loom_product_activation_resident_freeze" \
  "$loom_product_activation_parent_9023" \
  "$loom_product_activation_parent_9024" \
  "$loom_product_activation_parent_9025" \
  "$loom_product_activation_parent_9029" \
  "$loom_product_activation_parent_9030" \
  "$loom_product_activation_parent_9025_v13" \
  "$loom_product_activation_resident_v4" \
  "$loom_product_activation_dispatcher" \
  "$loom_product_activation_build" \
  "$loom_product_activation_gate"; do
  [[ -f "$product_activation_source" ]] || \
    die "Loom product activation capsule is incomplete: $product_activation_source"
done
for product_exec_ingress_source in \
  "$loom_product_exec_ingress_freeze" \
  "$loom_product_exec_ingress_contract" \
  "$loom_product_exec_ingress_evidence" \
  "${loom_product_exec_ingress_sources[@]}"; do
  [[ -f "$product_exec_ingress_source" ]] ||
    die "Loom product ExecIngress capsule is incomplete: $product_exec_ingress_source"
done
for sovereign_change_source in \
  "$loom_change_build_source" \
  "$loom_material_change_build_source" \
  "$loom_change_source" \
  "$loom_change_entrypoint" \
  "$loom_change_freeze" \
  "$loom_material_change_source" \
  "$loom_material_change_entrypoint" \
  "$loom_material_change_freeze" \
  "$loom_material_change_product" \
  "$loom_material_change_evidence" \
  "$loom_change_operational_gate" \
  "$loom_change_ci_admit" \
  "${loom_change_sources[@]}"; do
  [[ -f "$sovereign_change_source" ]] ||
    die "Loom sovereign change capsule is incomplete: $sovereign_change_source"
done
[[ -f "$loom_project/src/loom.ml" && -f "$loom_project/src/loom_arrow.ml" && \
  -f "$loom_project/src/loom_epistemic.ml" && \
  -f "$loom_project/src/loom_exec.ml" && \
  -f "$loom_project/src/loom_exec_ingress.ml" && \
  -f "$loom_project/src/loom_hook.ml" && \
  -f "$loom_project/src/loom_change.ml" && \
  -f "$loom_project/src/loom_change_stubs.c" && \
  -f "$loom_project/src/loom_sovereign_exec.ml" && \
  -f "$loom_project/src/loom_sovereign_provider_fixture.ml" && \
  -f "$loom_project/src/loom_lane_health.ml" && \
  -f "$loom_project/src/loom_witness.ml" && \
  -f "$loom_project/src/loom_witness_epoch.ml" && \
  -f "$loom_project/src/loom_witness_transparency.ml" && \
  -f "$loom_project/src/loom_ui.ml" && \
  -f "$loom_project/src/loom_pty_stubs.c" && \
  -f "$loom_project/src/dune" && -f "$loom_project/dune-project" ]] || \
  die "Loom OCaml source bundle is incomplete: $loom_project"

version_output="$(cd "$WORKTREE" && "$runtime_source" runtime-version)"
protocol="$(sed -n 's/^protocol_version=//p' <<< "$version_output" | head -1)"
runtime_version="$(sed -n 's/^runtime_version=//p' <<< "$version_output" | head -1)"
[[ "$protocol" == "$CLIENT_PROTOCOL" ]] || \
  die "source protocol $protocol is incompatible with installer protocol $CLIENT_PROTOCOL"
[[ -n "$runtime_version" ]] || die "source runtime did not report a version"

agentd_version_output="$($agentd_source runtime-version)"
agentd_protocol="$(sed -n 's/^protocol_version=//p' <<< "$agentd_version_output" | head -1)"
[[ "$agentd_protocol" == 1 ]] || die "agent supervisor protocol must be 1"

fleet_version_output="$($fleet_source runtime-version)"
fleet_protocol="$(sed -n 's/^protocol_version=//p' <<< "$fleet_version_output" | head -1)"
[[ "$fleet_protocol" == 1 ]] || die "fleet launcher protocol must be 1"

fleetd_version_output="$($fleetd_source runtime-version)"
fleetd_protocol="$(sed -n 's/^protocol_version=//p' <<< "$fleetd_version_output" | head -1)"
[[ "$fleetd_protocol" == 1 ]] || die "fleet reconciler protocol must be 1"

"$loom_build_source" >/dev/null
loom_binary="$loom_project/_build/default/src/loom.exe"
loom_language_authority_binary="$loom_project/.runtime/sounio-loom-language-authority-runtime"
loom_custody_transfer_binary="$loom_project/_build/default/src/sounio-loom-custody-transfer-runtime"
loom_execution_outcome_binary="$loom_project/.runtime/sounio-loom-execution-outcome-runtime"
loom_lane_health_binary="$loom_project/.runtime/sounio-loom-lane-health-runtime"
loom_lane_health_parity_binary="$loom_project/.runtime/sounio-loom-lane-health-parity-runtime"
loom_continuity_binary="$loom_project/_build/default/src/sounio-loom-continuity-runtime"
loom_obligation_binary="$loom_project/_build/default/src/sounio-loom-obligation-runtime"
loom_epistemic_binary="$loom_project/_build/default/src/sounio-loom-epistemic-runtime"
loom_attention_binary="$loom_project/_build/default/src/sounio-loom-attention-runtime"
loom_portfolio_binary="$loom_project/_build/default/src/sounio-loom-portfolio-runtime"
loom_contingent_binary="$loom_project/_build/default/src/sounio-loom-contingent-runtime"
loom_outcome_authority_binary="$loom_project/_build/default/src/sounio-loom-outcome-authority-runtime"
loom_witness_mesh_binary="$loom_project/_build/default/src/sounio-loom-witness-mesh-runtime"
loom_witness_mesh_v1_binary="$loom_project/_build/default/src/sounio-loom-witness-mesh-v1-runtime"
loom_witness_epoch_handoff_binary="$loom_project/_build/default/src/sounio-loom-witness-epoch-handoff-runtime"
loom_witness_epoch_transparency_binary="$loom_project/_build/default/src/sounio-loom-witness-epoch-transparency-runtime"
loom_product_activation_resident_binary="$loom_project/.runtime/sounio-loom-resident-membrane-runtime-v5"
loom_sovereign_binary="$loom_project/_build/default/src/sounio-loom-sovereign-execution-kernel"
loom_change_binary="$loom_project/_build/default/src/sounio-loom-sovereign-change-kernel"
loom_material_change_binary="$loom_project/_build/default/src/sounio-loom-sovereign-material-change"
[[ -x "$loom_binary" ]] || die "Loom build omitted its native executable"
[[ -x "$loom_sovereign_binary" ]] || \
  die "Loom build omitted frozen Sounio action 9042"
[[ -x "$loom_change_binary" && -x "$loom_material_change_binary" ]] ||
  die "Loom build omitted frozen Sounio actions 9043/9044"
loom_sovereign_expected_sha="$(manifest_value "$loom_sovereign_semantic_freeze" executable_sha256)"
loom_sovereign_actual_sha="$(sha256sum "$loom_sovereign_binary" | awk '{print $1}')"
[[ "$loom_sovereign_actual_sha" == "$loom_sovereign_expected_sha" ]] || \
  die "Loom Sounio action 9042 runtime failed frozen hash verification"
loom_sovereign_probe="$(printf '0\n' | "$loom_sovereign_binary")"
[[ "$loom_sovereign_probe" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL_SELFTEST PASS cases=14' ]] || \
  die "Loom Sounio action 9042 failed its install probe"
loom_change_expected_sha="$(manifest_value "$loom_change_freeze" executable_sha256)"
loom_material_change_expected_sha="$(
  manifest_value "$loom_material_change_freeze" executable_sha256
)"
[[ "$(sha256sum "$loom_change_binary" | awk '{print $1}')" == \
   "$loom_change_expected_sha" ]] ||
  die "Loom Sounio action 9043 runtime failed frozen hash verification"
[[ "$(sha256sum "$loom_material_change_binary" | awk '{print $1}')" == \
   "$loom_material_change_expected_sha" ]] ||
  die "Loom Sounio action 9044 runtime failed frozen hash verification"
[[ "$(printf '0\n' | "$loom_change_binary")" == \
   'SOUNIO_SOVEREIGN_CHANGE_KERNEL_SELFTEST PASS cases=21' ]] ||
  die "Loom Sounio action 9043 failed its install probe"
[[ "$(printf '0\n' | "$loom_material_change_binary")" == \
   'SOUNIO_SOVEREIGN_MATERIAL_CHANGE_SELFTEST PASS cases=8' ]] ||
  die "Loom Sounio action 9044 failed its install probe"
[[ -x "$loom_language_authority_binary" ]] || \
  die "Loom build omitted its frozen Sounio language-authority runtime"
[[ -x "$loom_custody_transfer_binary" ]] || \
  die "Loom build omitted its frozen Sounio custody-transfer runtime"
[[ -x "$loom_execution_outcome_binary" ]] || \
  die "Loom build omitted its frozen Sounio execution-outcome runtime"
[[ -x "$loom_lane_health_binary" && -x "$loom_lane_health_parity_binary" ]] || \
  die "Loom build omitted its frozen Sounio lane-health runtimes"
[[ -x "$loom_continuity_binary" ]] || \
  die "Loom build omitted its native Sounio continuity adapter"
[[ -x "$loom_obligation_binary" ]] || \
  die "Loom build omitted its native Sounio obligation adapter"
[[ -x "$loom_epistemic_binary" ]] || \
  die "Loom build omitted its native Sounio epistemic adapter"
[[ -x "$loom_attention_binary" ]] || \
  die "Loom build omitted its native Sounio attention adapter"
[[ -x "$loom_portfolio_binary" ]] || \
  die "Loom build omitted its native Sounio portfolio adapter"
[[ -x "$loom_contingent_binary" ]] || \
  die "Loom build omitted its native Sounio contingent-policy adapter"
[[ -x "$loom_outcome_authority_binary" ]] || \
  die "Loom build omitted its native Sounio outcome-authority adapter"
[[ -x "$loom_witness_mesh_binary" ]] || \
  die "Loom build omitted its native Sounio witness-mesh adapter"
[[ -x "$loom_witness_mesh_v1_binary" ]] || \
  die "Loom build omitted its native Sounio witness-mesh-v1 adapter"
[[ -x "$loom_witness_epoch_handoff_binary" ]] || \
  die "Loom build omitted its native Sounio witness-epoch-handoff adapter"
[[ -x "$loom_witness_epoch_transparency_binary" ]] || \
  die "Loom build omitted its native Sounio witness-epoch-transparency adapter"
[[ -x "$loom_product_activation_resident_binary" ]] || \
  die "Loom build omitted its frozen Sounio resident v5 runtime"
loom_version_output="$($loom_binary runtime-version)"
loom_protocol="$(sed -n 's/^protocol_version=//p' <<< "$loom_version_output" | head -1)"
loom_runtime_version="$(sed -n 's/^runtime_version=//p' <<< "$loom_version_output" | head -1)"
loom_language="$(sed -n 's/^language=//p' <<< "$loom_version_output" | head -1)"
[[ "$loom_protocol" == 1 && "$loom_language" == OCaml ]] || \
  die "Loom kernel must report protocol 1 and language OCaml"
[[ "$loom_runtime_version" == "$runtime_version" ]] || \
  die "Loom kernel version $loom_runtime_version does not match coordination runtime $runtime_version"
loom_language_authority_probe="$(printf '0\n' | "$loom_language_authority_binary")"
[[ "$loom_language_authority_probe" == \
  'SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33' ]] || \
  die "Loom frozen Sounio language-authority runtime failed its install probe"
loom_language_authority_expected_sha="$(
  manifest_value "$loom_language_authority_freeze" executable_sha256
)"
read -r loom_language_authority_actual_sha _ < <(
  sha256sum "$loom_language_authority_binary"
)
[[ "$loom_language_authority_actual_sha" == "$loom_language_authority_expected_sha" ]] || \
  die "Loom language-authority runtime does not match its freeze manifest"
loom_custody_transfer_probe="$(printf '0\n' | "$loom_custody_transfer_binary")"
[[ "$loom_custody_transfer_probe" == \
  'SOUNIO_CUSTODY_TRANSFER_SELFTEST PASS cases=30' ]] || \
  die "Loom frozen Sounio custody-transfer runtime failed its install probe"
loom_custody_transfer_expected_sha="$(
  manifest_value "$loom_custody_transfer_freeze" executable_sha256
)"
read -r loom_custody_transfer_actual_sha _ < <(
  sha256sum "$loom_custody_transfer_binary"
)
[[ "$loom_custody_transfer_actual_sha" == "$loom_custody_transfer_expected_sha" ]] || \
  die "Loom custody-transfer runtime does not match its freeze manifest"
loom_execution_outcome_probe="$(printf '0\n' | "$loom_execution_outcome_binary")"
[[ "$loom_execution_outcome_probe" == \
  'SOUNIO_EXECUTION_OUTCOME_SELFTEST PASS cases=28' ]] || \
  die "Loom frozen Sounio execution-outcome runtime failed its install probe"
loom_execution_outcome_expected_sha="$(
  manifest_value "$loom_execution_outcome_freeze" executable_sha256
)"
read -r loom_execution_outcome_actual_sha _ < <(
  sha256sum "$loom_execution_outcome_binary"
)
[[ "$loom_execution_outcome_actual_sha" == "$loom_execution_outcome_expected_sha" ]] || \
  die "Loom execution-outcome runtime does not match its freeze manifest"
loom_lane_health_probe="$(printf '0\n' | "$loom_lane_health_binary")"
[[ "$loom_lane_health_probe" == \
  'SOUNIO_LANE_HEALTH_SELFTEST PASS cases=28' ]] || \
  die "Loom frozen Sounio lane-health runtime failed its install probe"
loom_lane_health_expected_sha="$(
  manifest_value "$loom_lane_health_freeze" executable_sha256
)"
read -r loom_lane_health_actual_sha _ < <(
  sha256sum "$loom_lane_health_binary"
)
[[ "$loom_lane_health_actual_sha" == "$loom_lane_health_expected_sha" ]] || \
  die "Loom lane-health runtime does not match its freeze manifest"
loom_continuity_probe="$(
  printf '101 111 201 301 401 501 0 0 0 0 1 0 0\n' | "$loom_continuity_binary"
)"
[[ "$loom_continuity_probe" == 'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v1' ]] || \
  die "Loom native Sounio continuity adapter failed its install probe"
loom_obligation_probe="$(
  printf '9007 1 0 1 101 0 0 0 0 1 2 3 4 5 6 7 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n' | \
    "$loom_obligation_binary"
)"
[[ "$loom_obligation_probe" == \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=open state=1' ]] || \
  die "Loom native Sounio obligation adapter failed its install probe"
zeros='0 0 0 0 0 0 0 0'
loom_epistemic_probe="$(
  printf '9008 1 0 1 101 0 0 0 0 0 %s %s %s %s %s %s %s\n' \
    "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" "$zeros" | \
    "$loom_epistemic_binary"
)"
[[ "$loom_epistemic_probe" == \
  'SOUNIO_EPISTEMIC_ACCEPT schema=loom-native-epistemic-v0 transition=create state=active' ]] || \
  die "Loom native Sounio epistemic adapter failed its install probe"
ones='1 1 1 1 1 1 1 1'
twos='2 2 2 2 2 2 2 2'
threes='3 3 3 3 3 3 3 3'
loom_attention_probe="$(
  printf '9009 1 1 100 101 201 202 301 401 900 800 700 50 100 800 900 900 50 100 %s %s %s %s\n' \
    "$ones" "$twos" "$threes" "$zeros" | "$loom_attention_binary"
)"
[[ "$loom_attention_probe" == \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=information-first' ]] || \
  die "Loom native Sounio attention adapter failed its install probe"
fours='4 4 4 4 4 4 4 4'
fives='5 5 5 5 5 5 5 5'
sixes='6 6 6 6 6 6 6 6'
sevens='7 7 7 7 7 7 7 7'
loom_portfolio_probe="$(
  printf '9010 1 1 100 100 10 10 101 201 202 301 401 900 800 700 40 50 50 5 5 800 900 900 50 50 50 5 5 %s %s %s %s %s %s\n' \
    "$ones" "$twos" "$threes" "$fours" "$fives" "$zeros" | \
    "$loom_portfolio_binary"
)"
[[ "$loom_portfolio_probe" == \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=information-first' ]] || \
  die "Loom native Sounio portfolio adapter failed its install probe"
loom_contingent_probe="$(
  printf '9011 1 1 0 100 100 10 10 101 201 202 301 0 0 0 401 501 900 500 400 40 50 50 5 5 800 900 900 50 50 50 5 5 %s %s %s %s %s %s %s %s\n' \
    "$ones" "$twos" "$threes" "$fours" "$fives" "$sixes" "$sevens" \
    "$zeros" | "$loom_contingent_binary"
)"
[[ "$loom_contingent_probe" == \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=information-first' ]] || \
  die "Loom native Sounio contingent-policy adapter failed its install probe"
loom_outcome_authority_probe="$(
  printf '9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 %s %s %s %s %s %s %s %s\n' \
    "$ones" "$twos" "$twos" "$threes" "$threes" "$fours" "$fours" \
    "$fives" | "$loom_outcome_authority_binary"
)"
[[ "$loom_outcome_authority_probe" == \
  'SOUNIO_OUTCOME_AUTHORITY_ACCEPT schema=loom-native-outcome-authority-v0 transition=consume state=verified' ]] || \
  die "Loom native Sounio outcome-authority adapter failed its install probe"
loom_witness_mesh_probe="$(
  printf '9013 2 1 1 0 101 201 301 101 201 0 401 401 401 0 0 1 1 1 0 0 3 3 3 0 %s %s %s %s %s %s %s %s\n' \
    "$ones" "$ones" "$ones" "$zeros" "$twos" "$twos" "$twos" \
    "$zeros" | "$loom_witness_mesh_binary"
)"
[[ "$loom_witness_mesh_probe" == \
  'SOUNIO_WITNESS_MESH_ACCEPT schema=loom-native-witness-mesh-v0 transition=anchor state=quorum-verified' ]] || \
  die "Loom native Sounio witness-mesh adapter failed its install probe"
loom_witness_mesh_v1_probe="$(
  printf '9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 %s %s %s %s %s %s %s %s %s %s\n' \
    "$ones" "$ones" "$ones" "$ones" "$zeros" \
    "$twos" "$twos" "$twos" "$twos" "$zeros" | \
    "$loom_witness_mesh_v1_binary"
)"
[[ "$loom_witness_mesh_v1_probe" == \
  'SOUNIO_WITNESS_MESH_V1_ACCEPT schema=loom-native-witness-mesh-v1 transition=anchor state=quorum-verified' ]] || \
  die "Loom native Sounio witness-mesh-v1 adapter failed its install probe"
loom_witness_epoch_handoff_probe="$(
  printf '9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 %s %s %s %s %s %s %s %s %s\n' \
    "$ones" "$twos" "$threes" "$fours" "$fives" "$fives" \
    "$sixes" "$sevens" "$zeros" | \
    "$loom_witness_epoch_handoff_binary"
)"
[[ "$loom_witness_epoch_handoff_probe" == \
  'SOUNIO_WITNESS_EPOCH_HANDOFF_ACCEPT schema=loom-native-witness-epoch-handoff-v0 transition=joint-quorum state=prepared' ]] || \
  die "Loom native Sounio witness-epoch-handoff adapter failed its install probe"
loom_witness_epoch_transparency_probe="$(
  printf '9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 %s %s %s %s %s %s %s %s %s %s %s %s\n' \
    "$ones" "$ones" "$zeros" "$zeros" "$twos" "$twos" \
    "$threes" "$threes" "$fours" "$fives" "$sixes" "$sevens" | \
    "$loom_witness_epoch_transparency_binary"
)"
[[ "$loom_witness_epoch_transparency_probe" == \
  'SOUNIO_WITNESS_EPOCH_TRANSPARENCY_ACCEPT schema=loom-native-witness-epoch-transparency-v0 rollback_bound=latest-quorum-witnessed-epoch state=verified' ]] || \
  die "Loom native Sounio witness-epoch-transparency adapter failed its install probe"
loom_measurement_probe="$(
  printf '9004 1002 1101 1201 1301 2101 2201 2301 2401 2101 2201 2301 2401\n' | \
    "$loom_continuity_binary"
)"
[[ "$loom_measurement_probe" == \
  'SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v2 authority=disjoint-principals+measured-fact-agreement' ]] || \
  die "Loom native Sounio independent-measurement adapter failed its install probe"
loom_authority_probe="$({
  printf '9005 1002 1101 1201 1301 1401 1501 1 2101 2201 2301 2401 2101 2201 2301 2401'
  for start in 1 11 21 31 1 11 21 31; do
    for offset in 0 1 2 3 4 5 6 7; do
      printf ' %d' "$((start + offset))"
    done
  done
  printf '\n'
} | "$loom_continuity_binary")"
[[ "$loom_authority_probe" == \
  'SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v3 authority=three-principals+full-sha256-agreement' ]] || \
  die "Loom native Sounio observation-authority adapter failed its install probe"
loom_quorum_probe="$({
  printf '9006 1002 1101 1201 1301 1302 1303 2 2 1401 1501 1 2101 2201 2301 2401 2101 2201 2301 2401'
  for start in 1 11 21 31 1 11 21 31; do
    for offset in 0 1 2 3 4 5 6 7; do
      printf ' %d' "$((start + offset))"
    done
  done
  printf '\n'
} | "$loom_continuity_binary")"
[[ "$loom_quorum_probe" == \
  'SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v4 authority=five-principals+2-of-3-journal-quorum+full-sha256-agreement' ]] || \
  die "Loom native Sounio journal-quorum adapter failed its install probe"

bundle_sources=(
  "$installer_source" "$runtime_source" "$hook_source" "$causal_source"
  "$agentd_source"
  "$fleet_source" "$fleetd_source" "$fleet_model_source"
  "$fleet_model_config" "$fleet_model_generator" "$fleet_trace_verifier"
  "$loom_build_source" "$loom_language_authority_build_source"
  "$loom_language_authority_entrypoint" "$loom_language_authority_module"
  "$loom_language_authority_freeze"
  "$loom_custody_transfer_build_source" "$loom_custody_transfer_entrypoint"
  "$loom_custody_transfer_module" "$loom_custody_transfer_freeze"
  "$loom_execution_outcome_build_source" "$loom_execution_outcome_entrypoint"
  "$loom_execution_outcome_module" "$loom_execution_outcome_freeze"
  "$loom_lane_health_build_source" "$loom_lane_health_parity_build_source"
  "$loom_lane_health_entrypoint" "$loom_lane_health_parity_entrypoint"
  "$loom_lane_health_module" "$loom_lane_health_freeze"
  "$loom_lane_health_ocaml_receipt" "$loom_sha256_module"
  "$loom_project/dune-project" "$loom_project/src/dune"
  "$loom_project/src/loom.ml" "$loom_project/src/loom_arrow.ml"
  "$loom_project/src/loom_epistemic.ml" "$loom_project/src/loom_exec.ml"
  "$loom_project/src/loom_exec_ingress.ml"
  "$loom_project/src/loom_hook.ml"
  "$loom_project/src/loom_sovereign_exec.ml"
  "$loom_project/src/loom_sovereign_provider_fixture.ml"
  "$loom_project/src/loom_lane_health.ml"
  "$loom_project/src/loom_witness.ml"
  "$loom_project/src/loom_witness_epoch.ml"
  "$loom_project/src/loom_witness_transparency.ml" "$loom_project/src/loom_ui.ml"
  "$loom_project/src/loom_pty_stubs.c" "$loom_project/src/loom_arrow_stubs.c"
  "$loom_project/src/loom_nanoarrow.c" "$loom_project/src/loom_nanoarrow_ipc.c"
  "$loom_project/src/loom_flatcc.c"
  "$loom_continuity_build_source" "$loom_continuity_entrypoint"
  "$loom_continuity_module" "$loom_obligation_build_source"
  "$loom_obligation_entrypoint" "$loom_obligation_module"
  "$loom_epistemic_build_source" "$loom_epistemic_entrypoint"
  "$loom_epistemic_module" "$loom_attention_build_source"
  "$loom_attention_entrypoint" "$loom_attention_module"
  "$loom_portfolio_build_source" "$loom_portfolio_entrypoint"
  "$loom_portfolio_module" "$loom_contingent_build_source"
  "$loom_contingent_entrypoint" "$loom_contingent_module"
  "$loom_outcome_authority_build_source"
  "$loom_outcome_authority_entrypoint" "$loom_outcome_authority_module"
  "$loom_witness_mesh_build_source" "$loom_witness_mesh_entrypoint"
  "$loom_witness_mesh_module" "$loom_witness_mesh_v1_build_source"
  "$loom_witness_mesh_v1_entrypoint" "$loom_witness_mesh_v1_module"
  "$loom_witness_epoch_handoff_build_source"
  "$loom_witness_epoch_handoff_entrypoint"
  "$loom_witness_epoch_handoff_module"
  "$loom_witness_epoch_transparency_build_source"
  "$loom_witness_epoch_transparency_entrypoint"
  "$loom_witness_epoch_transparency_module"
  "$loom_product_activation_garden" "$loom_product_activation_source"
  "$loom_product_activation_entrypoint" "$loom_product_activation_action_freeze"
  "$loom_product_activation_operational_freeze"
  "$loom_product_activation_projection" "$loom_product_activation_resident_freeze"
  "$loom_product_activation_parent_9023" "$loom_product_activation_parent_9024"
  "$loom_product_activation_parent_9025" "$loom_product_activation_parent_9029"
  "$loom_product_activation_parent_9030" "$loom_product_activation_parent_9025_v13"
  "$loom_product_activation_resident_v4" "$loom_product_activation_dispatcher"
  "$loom_product_activation_build" "$loom_product_activation_gate"
  "$loom_product_exec_ingress_freeze" "$loom_product_exec_ingress_contract"
  "$loom_product_exec_ingress_evidence"
  "$loom_sovereign_build_source" "$loom_sovereign_source"
  "$loom_sovereign_entrypoint" "$loom_sovereign_semantic_freeze"
  "$loom_sovereign_material_freeze" "$loom_sovereign_product_freeze"
  "$loom_sovereign_product_contract" "$loom_sovereign_product_evidence"
  "$loom_sovereign_product_gate" "$loom_sovereign_product_freeze_gate"
  "$loom_change_build_source" "$loom_material_change_build_source"
  "$loom_change_source" "$loom_change_entrypoint" "$loom_change_freeze"
  "$loom_material_change_source" "$loom_material_change_entrypoint"
  "$loom_material_change_freeze" "$loom_material_change_product"
  "$loom_material_change_evidence" "$loom_change_operational_gate"
  "$loom_change_ci_admit" "${loom_change_sources[@]}"
  "$loom_project/src/loom_membrane.ml"
  "$loom_project/src/loom_peer_activation_capsule.ml"
  "$loom_project/src/loom_resident.ml"
)

source_sha=unknown
source_state=unversioned
if git -C "$SOURCE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  source_sha="$(git -C "$SOURCE_ROOT" rev-parse --short=12 HEAD)"
  source_paths=()
  for source in "${bundle_sources[@]}"; do
    source_paths+=("${source#"$SOURCE_ROOT"/}")
  done
  source_paths+=("tools/loom/src/vendor")
  source_dirty="$(
    git -C "$SOURCE_ROOT" status --porcelain=v1 --untracked-files=all -- \
      "${source_paths[@]}"
  )"
  if [[ -n "$source_dirty" ]]; then
    printf '%s\n' "$source_dirty" >&2
    die "runtime source bundle has uncommitted changes; commit the source before installation"
  fi
  source_state=clean
fi

bundle_sha="$({
  sha256sum "${bundle_sources[@]}"
  find "$loom_project/src/vendor" -type f -print0 | sort -z | xargs -0 sha256sum
} | \
    awk '{print $1}' | sha256sum | awk '{print $1}'
)"
safe_version="$(printf '%s' "$runtime_version" | tr -c 'A-Za-z0-9._-' '_')"
runtime_id="p${protocol}-${safe_version}-${bundle_sha:0:12}"
version_dir="$RUNTIME_ROOT/versions/$runtime_id"

if [[ -d "$version_dir" ]]; then
  installed_sha="$(manifest_value "$version_dir/manifest" bundle_sha256)"
  [[ "$installed_sha" == "$bundle_sha" ]] || \
    die "runtime id collision with different bundle: $runtime_id"
else
  stage="$(mktemp -d "$RUNTIME_ROOT/.install.XXXXXX")"
  cleanup_stage() {
    [[ -z "${stage:-}" ]] || rm -rf "$stage"
  }
  trap cleanup_stage EXIT
  mkdir -p "$stage/bin" "$stage/hooks" "$stage/formal" \
    "$stage/policy/language-authority/tools/loom" \
    "$stage/policy/language-authority/stdlib/coordination" \
    "$stage/policy/product-activation/tools/loom" \
    "$stage/policy/product-activation/stdlib/coordination" \
    "$stage/policy/product-activation/scripts/dev" \
    "$stage/policy/product-activation/scripts/ci" \
    "$stage/policy/product-exec-ingress/tools/loom/evidence" \
    "$stage/policy/product-exec-ingress/tools/loom/src" \
    "$stage/policy/sovereign-execution" \
    "$stage/policy/sovereign-change"
  install -m 0755 "$runtime_source" "$stage/bin/sounio-coord-runtime"
  install -m 0755 "$causal_source" "$stage/bin/sounio-coord-causal-runtime"
  install -m 0755 "$agentd_source" "$stage/bin/sounio-agentd-runtime"
  install -m 0755 "$fleet_source" "$stage/bin/sounio-fleet-agent-runtime"
  install -m 0755 "$fleetd_source" "$stage/bin/sounio-fleet-runtime"
  install -m 0755 "$fleet_model_generator" "$stage/bin/sounio-fleet-tla-sabotage"
  install -m 0755 "$fleet_trace_verifier" "$stage/bin/sounio-fleet-trace-verify"
  install -m 0755 "$loom_binary" "$stage/bin/sounio-loom-runtime"
  install -m 0555 "$loom_sovereign_binary" \
    "$stage/bin/sounio-loom-sovereign-execution-kernel"
  install -m 0555 "$loom_change_binary" \
    "$stage/bin/sounio-loom-sovereign-change-kernel"
  install -m 0555 "$loom_material_change_binary" \
    "$stage/bin/sounio-loom-sovereign-material-change"
  install -m 0755 "$loom_language_authority_binary" \
    "$stage/bin/sounio-loom-language-authority-runtime"
  install -m 0644 "$loom_language_authority_freeze" \
    "$stage/policy/language-authority/tools/loom/language_authority.freeze.v1"
  install -m 0644 "$loom_language_authority_entrypoint" \
    "$stage/policy/language-authority/tools/loom/language_authority_main.sio"
  install -m 0644 "$loom_language_authority_module" \
    "$stage/policy/language-authority/stdlib/coordination/loom_language_authority.sio"
  install -m 0755 "$loom_custody_transfer_binary" \
    "$stage/bin/sounio-loom-custody-transfer-runtime"
  install -m 0755 "$loom_execution_outcome_binary" \
    "$stage/bin/sounio-loom-execution-outcome-runtime"
  install -m 0755 "$loom_lane_health_binary" \
    "$stage/bin/sounio-loom-lane-health-runtime"
  install -m 0755 "$loom_lane_health_parity_binary" \
    "$stage/bin/sounio-loom-lane-health-parity-runtime"
  install -m 0755 "$loom_continuity_binary" \
    "$stage/bin/sounio-loom-continuity-runtime"
  install -m 0755 "$loom_obligation_binary" \
    "$stage/bin/sounio-loom-obligation-runtime"
  install -m 0755 "$loom_epistemic_binary" \
    "$stage/bin/sounio-loom-epistemic-runtime"
  install -m 0755 "$loom_attention_binary" \
    "$stage/bin/sounio-loom-attention-runtime"
  install -m 0755 "$loom_portfolio_binary" \
    "$stage/bin/sounio-loom-portfolio-runtime"
  install -m 0755 "$loom_contingent_binary" \
    "$stage/bin/sounio-loom-contingent-runtime"
  install -m 0755 "$loom_outcome_authority_binary" \
    "$stage/bin/sounio-loom-outcome-authority-runtime"
  install -m 0755 "$loom_witness_mesh_binary" \
    "$stage/bin/sounio-loom-witness-mesh-runtime"
  install -m 0755 "$loom_witness_mesh_v1_binary" \
    "$stage/bin/sounio-loom-witness-mesh-v1-runtime"
  install -m 0755 "$loom_witness_epoch_handoff_binary" \
    "$stage/bin/sounio-loom-witness-epoch-handoff-runtime"
  install -m 0755 "$loom_witness_epoch_transparency_binary" \
    "$stage/bin/sounio-loom-witness-epoch-transparency-runtime"
  install -m 0555 "$loom_product_activation_resident_binary" \
    "$stage/bin/sounio-loom-resident-membrane-runtime-v5"
  for product_activation_file in \
    "$loom_product_activation_garden" \
    "$loom_product_activation_entrypoint" \
    "$loom_product_activation_action_freeze" \
    "$loom_product_activation_operational_freeze" \
    "$loom_product_activation_projection" \
    "$loom_product_activation_resident_freeze" \
    "$loom_product_activation_parent_9023" \
    "$loom_product_activation_parent_9024" \
    "$loom_product_activation_parent_9025" \
    "$loom_product_activation_parent_9029" \
    "$loom_product_activation_parent_9030" \
    "$loom_product_activation_parent_9025_v13" \
    "$loom_product_activation_resident_v4" \
    "$loom_product_activation_dispatcher"; do
    install -m 0444 "$product_activation_file" \
      "$stage/policy/product-activation/tools/loom/$(basename "$product_activation_file")"
  done
  install -m 0444 "$loom_product_activation_source" \
    "$stage/policy/product-activation/stdlib/coordination/$(basename "$loom_product_activation_source")"
  install -m 0555 "$loom_product_activation_build" \
    "$stage/policy/product-activation/scripts/dev/$(basename "$loom_product_activation_build")"
  install -m 0555 "$loom_product_activation_gate" \
    "$stage/policy/product-activation/scripts/ci/$(basename "$loom_product_activation_gate")"
  install -m 0444 "$loom_product_exec_ingress_freeze" \
    "$stage/policy/product-exec-ingress/tools/loom/$(basename "$loom_product_exec_ingress_freeze")"
  install -m 0444 "$loom_product_exec_ingress_contract" \
    "$stage/policy/product-exec-ingress/tools/loom/$(basename "$loom_product_exec_ingress_contract")"
  install -m 0444 "$loom_product_exec_ingress_evidence" \
    "$stage/policy/product-exec-ingress/tools/loom/evidence/$(basename "$loom_product_exec_ingress_evidence")"
  for product_exec_ingress_source in "${loom_product_exec_ingress_sources[@]}"; do
    install -m 0444 "$product_exec_ingress_source" \
      "$stage/policy/product-exec-ingress/tools/loom/src/$(basename "$product_exec_ingress_source")"
  done
  sovereign_capsule_sources=(
    "$loom_sovereign_source" "$loom_sovereign_entrypoint"
    "$loom_sovereign_semantic_freeze" "$loom_sovereign_material_freeze"
    "$loom_sovereign_product_freeze" "$loom_sovereign_product_contract"
    "$loom_sovereign_product_evidence" "$loom_sovereign_build_source"
    "${loom_sovereign_sources[@]}"
  )
  for sovereign_source in "${sovereign_capsule_sources[@]}"; do
    sovereign_relative="${sovereign_source#"$SOURCE_ROOT/"}"
    sovereign_target="$stage/policy/sovereign-execution/$sovereign_relative"
    mkdir -p "$(dirname "$sovereign_target")"
    install -m 0444 "$sovereign_source" "$sovereign_target"
  done
  sovereign_frozen_head="$(
    manifest_value "$loom_sovereign_product_evidence" source_head
  )"
  [[ "$sovereign_frozen_head" =~ ^[0-9a-f]{40}$ ]] &&
    git -C "$SOURCE_ROOT" cat-file -e "$sovereign_frozen_head^{commit}" ||
    die "Loom sovereign execution product historical source commit is unavailable"
  for sovereign_pair in \
    contract_path:contract_sha256 \
    semantic_manifest_path:semantic_manifest_sha256 \
    material_manifest_path:material_manifest_sha256 \
    sounio_source_path:sounio_source_sha256 \
    sounio_entrypoint_path:sounio_entrypoint_sha256 \
    loom_source_path:loom_source_sha256 \
    exec_source_path:exec_source_sha256 \
    hook_source_path:hook_source_sha256 \
    sovereign_source_path:sovereign_source_sha256 \
    provider_fixture_path:provider_fixture_sha256 \
    c_stub_path:c_stub_sha256 \
    dune_path:dune_sha256 \
    loom_build_path:loom_build_sha256 \
    installer_path:installer_sha256 \
    coord_runtime_path:coord_runtime_sha256 \
    product_gate_path:product_gate_sha256 \
    freeze_gate_path:freeze_gate_sha256; do
    sovereign_path_key="${sovereign_pair%%:*}"
    sovereign_hash_key="${sovereign_pair#*:}"
    sovereign_relative="$(
      manifest_value "$loom_sovereign_product_freeze" "$sovereign_path_key"
    )"
    sovereign_expected="$(
      manifest_value "$loom_sovereign_product_freeze" "$sovereign_hash_key"
    )"
    [[ -n "$sovereign_relative" && "$sovereign_relative" != /* &&
      "$sovereign_relative" != *'..'* &&
      "$sovereign_expected" =~ ^[0-9a-f]{64}$ ]] ||
      die "Loom sovereign execution product has an unsafe frozen source entry"
    sovereign_target="$stage/policy/sovereign-execution/$sovereign_relative"
    sovereign_actual="$(sha256sum "$sovereign_target" | awk '{print $1}')"
    if [[ "$sovereign_actual" != "$sovereign_expected" ]]; then
      sovereign_temporary="$sovereign_target.frozen.$$.$RANDOM"
      git -C "$SOURCE_ROOT" cat-file blob \
        "$sovereign_frozen_head:$sovereign_relative" >"$sovereign_temporary" ||
        die "Loom sovereign execution frozen source is unavailable: $sovereign_relative"
      sovereign_actual="$(sha256sum "$sovereign_temporary" | awk '{print $1}')"
      [[ "$sovereign_actual" == "$sovereign_expected" ]] ||
        die "Loom sovereign execution historical source hash diverged: $sovereign_relative"
      chmod 0444 "$sovereign_temporary"
      mv "$sovereign_temporary" "$sovereign_target"
    fi
  done
  sovereign_change_capsule_sources=(
    "$installer_source"
    "$loom_change_build_source" "$loom_material_change_build_source"
    "$loom_change_source" "$loom_change_entrypoint" "$loom_change_freeze"
    "$loom_material_change_source" "$loom_material_change_entrypoint"
    "$loom_material_change_freeze" "$loom_material_change_product"
    "$loom_material_change_evidence" "$loom_change_operational_gate"
    "$loom_change_ci_admit" "${loom_change_sources[@]}"
  )
  for sovereign_change_source in "${sovereign_change_capsule_sources[@]}"; do
    sovereign_change_relative="${sovereign_change_source#"$SOURCE_ROOT/"}"
    sovereign_change_target="$stage/policy/sovereign-change/$sovereign_change_relative"
    mkdir -p "$(dirname "$sovereign_change_target")"
    case "$sovereign_change_relative" in
      scripts/dev/*|scripts/ci/*)
        install -m 0555 "$sovereign_change_source" "$sovereign_change_target"
        ;;
      *)
        install -m 0444 "$sovereign_change_source" "$sovereign_change_target"
        ;;
    esac
  done
  install -m 0644 "$fleet_model_source" "$stage/formal/SounioFleet.tla"
  install -m 0644 "$fleet_model_config" "$stage/formal/SounioFleet.cfg"
  install -m 0755 "$hook_source" "$stage/hooks/sounio_coord_agent_hook_runtime.py"
  coord_runtime_sha256="$(sha256sum "$stage/bin/sounio-coord-runtime" | awk '{print $1}')"
  loom_runtime_sha256="$(sha256sum "$stage/bin/sounio-loom-runtime" | awk '{print $1}')"
  loom_sovereign_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-sovereign-execution-kernel" | awk '{print $1}'
  )"
  loom_change_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-sovereign-change-kernel" | awk '{print $1}'
  )"
  loom_material_change_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-sovereign-material-change" | awk '{print $1}'
  )"
  loom_change_manifest_sha256="$(
    sha256sum "$stage/policy/sovereign-change/tools/loom/sovereign_change_kernel.freeze.v1" | awk '{print $1}'
  )"
  loom_material_change_manifest_sha256="$(
    sha256sum "$stage/policy/sovereign-change/tools/loom/sovereign_material_change.freeze.v2" | awk '{print $1}'
  )"
  loom_material_change_product_sha256="$(
    sha256sum "$stage/policy/sovereign-change/tools/loom/sovereign_material_change_product.runtime.v2" | awk '{print $1}'
  )"
  loom_sovereign_product_manifest_sha256="$(
    sha256sum "$stage/policy/sovereign-execution/tools/loom/sovereign_execution_kernel_product.runtime.v1" | awk '{print $1}'
  )"
  loom_sovereign_product_contract_sha256="$(
    sha256sum "$stage/policy/sovereign-execution/tools/loom/SOVEREIGN_EXECUTION_KERNEL_PRODUCT_ATTACHMENT_V1.md" | awk '{print $1}'
  )"
  loom_sovereign_product_evidence_sha256="$(
    sha256sum "$stage/policy/sovereign-execution/tools/loom/evidence/loom-sovereign-execution-kernel-product-v1-20260831.txt" | awk '{print $1}'
  )"
  loom_language_authority_policy_manifest_sha256="$(
    sha256sum "$stage/policy/language-authority/tools/loom/language_authority.freeze.v1" | awk '{print $1}'
  )"
  loom_language_authority_policy_source_sha256="$(
    sha256sum "$stage/policy/language-authority/stdlib/coordination/loom_language_authority.sio" | awk '{print $1}'
  )"
  loom_language_authority_policy_entrypoint_sha256="$(
    sha256sum "$stage/policy/language-authority/tools/loom/language_authority_main.sio" | awk '{print $1}'
  )"
  loom_custody_transfer_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-custody-transfer-runtime" | awk '{print $1}'
  )"
  loom_execution_outcome_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-execution-outcome-runtime" | awk '{print $1}'
  )"
  loom_product_activation_resident_runtime_sha256="$(
    sha256sum "$stage/bin/sounio-loom-resident-membrane-runtime-v5" | awk '{print $1}'
  )"
  loom_product_exec_ingress_manifest_sha256="$(
    sha256sum "$stage/policy/product-exec-ingress/tools/loom/product_exec_ingress_dark.runtime.v1" | awk '{print $1}'
  )"
  loom_product_exec_ingress_contract_sha256="$(
    sha256sum "$stage/policy/product-exec-ingress/tools/loom/PRODUCT_EXEC_INGRESS_DARK_ATTACHMENT_V1.md" | awk '{print $1}'
  )"
  loom_product_exec_ingress_evidence_sha256="$(
    sha256sum "$stage/policy/product-exec-ingress/tools/loom/evidence/loom-product-exec-ingress-dark-v1-20260829.txt" | awk '{print $1}'
  )"
  loom_product_exec_ingress_reference_runtime_sha256="$(
    manifest_value "$loom_product_exec_ingress_freeze" runtime_sha256
  )"
  loom_product_exec_ingress_reference_runtime_match=false
  if [[ "$loom_product_exec_ingress_reference_runtime_sha256" == \
    "$loom_runtime_sha256" ]]; then
    loom_product_exec_ingress_reference_runtime_match=true
  fi
  {
    printf 'runtime_id=%s\n' "$runtime_id"
    printf 'protocol_version=%s\n' "$protocol"
    printf 'agentd_protocol_version=%s\n' "$agentd_protocol"
    printf 'fleet_protocol_version=%s\n' "$fleet_protocol"
    printf 'fleetd_protocol_version=%s\n' "$fleetd_protocol"
    printf 'loom_protocol_version=%s\n' "$loom_protocol"
    printf 'loom_language_authority_language=Sounio\n'
    printf 'loom_language_authority_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_language_authority_stage=SEMANTICS_FROZEN\n'
    printf 'loom_language_authority_frame=9020\n'
    printf 'loom_language_authority_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff\n'
    printf 'loom_language_authority_manifest_sha256=5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e\n'
    printf 'loom_language_authority_policy_manifest_sha256=%s\n' \
      "$loom_language_authority_policy_manifest_sha256"
    printf 'loom_language_authority_policy_source_sha256=%s\n' \
      "$loom_language_authority_policy_source_sha256"
    printf 'loom_language_authority_policy_entrypoint_sha256=%s\n' \
      "$loom_language_authority_policy_entrypoint_sha256"
    printf 'loom_custody_transfer_language=Sounio\n'
    printf 'loom_custody_transfer_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_custody_transfer_stage=SEMANTICS_FROZEN\n'
    printf 'loom_custody_transfer_frame=9040\n'
    printf 'loom_custody_transfer_semantics_sha256=5f53d3edcb6731c5b0f4e58ff7b27d251e6c0b40eda8c68366e48b17e596f55c\n'
    printf 'loom_custody_transfer_manifest_sha256=ee4e5d128bf5b0fd7166e74c9815a17506a5b9844730c1be2155ac68c370be66\n'
    printf 'loom_execution_outcome_language=Sounio\n'
    printf 'loom_execution_outcome_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_execution_outcome_stage=SEMANTICS_FROZEN\n'
    printf 'loom_execution_outcome_realization=OCaml\n'
    printf 'loom_execution_outcome_frame=9022\n'
    printf 'loom_execution_outcome_semantics_sha256=c98c13d30d66ba2fb3d0fb34d75bd21b14b353bc88fd80acf7dbb385cb9fa914\n'
    printf 'loom_execution_outcome_manifest_sha256=f5e63a2fd6a946cea1a4cb57013ae0cfa1772c42c3cc52e42d300dfb7b45e16e\n'
    printf 'loom_lane_health_language=Sounio\n'
    printf 'loom_lane_health_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_lane_health_realization=OCaml\n'
    printf 'loom_lane_health_frame=9030\n'
    printf 'loom_lane_health_semantics_sha256=5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f\n'
    printf 'loom_lane_health_manifest_sha256=c0ef8162883bc1e44d29dadb2f28ed618779f8abf4257070258abcd24c2fab71\n'
    printf 'loom_product_activation_language=Sounio\n'
    printf 'loom_product_activation_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_product_activation_operational_attachment=OCaml\n'
    printf 'loom_product_activation_action=9031\n'
    printf 'loom_product_activation_action_manifest_sha256=f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e\n'
    printf 'loom_product_activation_operational_manifest_sha256=d7521e8fb60501dc8192ebbeade4a09649164c5b509a2dda8af5c465bf3de793\n'
    printf 'loom_product_activation_resident_manifest_sha256=b3cf8c1e0524be35fc67b2b5a779bad9a9291195d65dc82dbc87595396fb5353\n'
    printf 'loom_product_activation_projection_sha256=8a72e9bcd510a751b856cf29960b7389486defcc4d13d7614546023d3d355014\n'
    printf 'loom_product_activation_resident_runtime_sha256=%s\n' \
      "$loom_product_activation_resident_runtime_sha256"
    printf 'loom_product_exec_ingress_language=Sounio\n'
    printf 'loom_product_exec_ingress_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_product_exec_ingress_operational_attachment=OCaml\n'
    printf 'loom_product_exec_ingress_action=9031\n'
    printf 'loom_product_exec_ingress_manifest_sha256=%s\n' \
      "$loom_product_exec_ingress_manifest_sha256"
    printf 'loom_product_exec_ingress_contract_sha256=%s\n' \
      "$loom_product_exec_ingress_contract_sha256"
    printf 'loom_product_exec_ingress_evidence_sha256=%s\n' \
      "$loom_product_exec_ingress_evidence_sha256"
    printf 'loom_product_exec_ingress_reference_runtime_sha256=%s\n' \
      "$loom_product_exec_ingress_reference_runtime_sha256"
    printf 'loom_product_exec_ingress_reference_runtime_match=%s\n' \
      "$loom_product_exec_ingress_reference_runtime_match"
    printf 'loom_sovereign_language=Sounio\n'
    printf 'loom_sovereign_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_sovereign_operational_kernel=OCaml\n'
    printf 'loom_sovereign_action=9042\n'
    printf 'loom_sovereign_semantic_manifest_sha256=966f022c98bc7df89ce40a90ede9ec8a9a726499baec0fd21e72f327f286a176\n'
    printf 'loom_sovereign_material_manifest_sha256=1005da28d4375da8d67fecc4a301c0c6e768902d720952f93e3f82a74fd41f92\n'
    printf 'loom_sovereign_runtime_sha256=%s\n' \
      "$loom_sovereign_runtime_sha256"
    printf 'loom_sovereign_product_manifest_sha256=%s\n' \
      "$loom_sovereign_product_manifest_sha256"
    printf 'loom_sovereign_product_contract_sha256=%s\n' \
      "$loom_sovereign_product_contract_sha256"
    printf 'loom_sovereign_product_evidence_sha256=%s\n' \
      "$loom_sovereign_product_evidence_sha256"
    printf 'loom_change_language=Sounio\n'
    printf 'loom_change_role=SEMANTIC_AUTHORITY\n'
    printf 'loom_change_operational_kernel=OCaml\n'
    printf 'loom_change_actions=9043,9044\n'
    printf 'loom_change_manifest_sha256=%s\n' \
      "$loom_change_manifest_sha256"
    printf 'loom_material_change_manifest_sha256=%s\n' \
      "$loom_material_change_manifest_sha256"
    printf 'loom_change_runtime_sha256=%s\n' \
      "$loom_change_runtime_sha256"
    printf 'loom_material_change_runtime_sha256=%s\n' \
      "$loom_material_change_runtime_sha256"
    printf 'loom_material_change_product_sha256=%s\n' \
      "$loom_material_change_product_sha256"
    printf 'loom_change_ci_policy=consume-not-reinterpret\n'
    printf 'loom_change_claim_ready=true\n'
    printf 'loom_continuity_language=Sounio\n'
    printf 'loom_continuity_engine=lean_single\n'
    printf 'loom_obligation_language=Sounio\n'
    printf 'loom_obligation_frame=9007\n'
    printf 'loom_epistemic_language=Sounio\n'
    printf 'loom_epistemic_frame=9008\n'
    printf 'loom_attention_language=Sounio\n'
    printf 'loom_attention_frame=9009\n'
    printf 'loom_portfolio_language=Sounio\n'
    printf 'loom_portfolio_frame=9010\n'
    printf 'loom_contingent_language=Sounio\n'
    printf 'loom_contingent_frame=9011\n'
    printf 'loom_outcome_authority_language=Sounio\n'
    printf 'loom_outcome_authority_frame=9012\n'
    printf 'loom_witness_mesh_language=Sounio\n'
    printf 'loom_witness_mesh_frame=9013\n'
    printf 'loom_witness_mesh_v1_language=Sounio\n'
    printf 'loom_witness_mesh_v1_frame=9014\n'
    printf 'loom_witness_epoch_handoff_language=Sounio\n'
    printf 'loom_witness_epoch_handoff_frame=9015\n'
    printf 'loom_witness_epoch_transparency_language=Sounio\n'
    printf 'loom_witness_epoch_transparency_frame=9016\n'
    printf 'runtime_version=%s\n' "$runtime_version"
    printf 'bundle_sha256=%s\n' "$bundle_sha"
    printf 'coord_runtime_sha256=%s\n' "$coord_runtime_sha256"
    printf 'loom_runtime_sha256=%s\n' "$loom_runtime_sha256"
    printf 'loom_custody_transfer_runtime_sha256=%s\n' \
      "$loom_custody_transfer_runtime_sha256"
    printf 'loom_execution_outcome_runtime_sha256=%s\n' \
      "$loom_execution_outcome_runtime_sha256"
    printf 'source_sha=%s\n' "$source_sha"
    printf 'source_state=%s\n' "$source_state"
    printf 'capability=causal-experiment-receipts-v1\n'
    printf 'capability=crash-recovery-v1\n'
    printf 'capability=agentd-transport-v1\n'
    printf 'capability=agentd-argv-attestation-v1\n'
    printf 'capability=agentd-tui-submit-v1\n'
    printf 'capability=agentd-logical-command-v1\n'
    printf 'capability=agentd-runtime-registration-v1\n'
    printf 'capability=loom-kernel-v1\n'
    printf 'capability=loom-transactional-custody-transfer-v1\n'
    printf 'capability=loom-durable-execution-outcome-v1\n'
    printf 'capability=loom-native-agent-hook-v1\n'
    printf 'capability=loom-runtime-authority-capsule-v1\n'
    printf 'capability=loom-product-launch-dark-attachment-v1\n'
    printf 'capability=loom-sovereign-execution-kernel-product-v1\n'
    printf 'capability=loom-sovereign-change-kernel-v2\n'
    printf 'capability=loom-truthful-lane-health-v1\n'
    printf 'capability=loom-nondestructive-health-reconcile-v1\n'
    printf 'capability=loom-native-hook-binary-attestation-v1\n'
    printf 'capability=loom-native-sounio-continuity-v1\n'
    printf 'capability=loom-durable-obligation-v1\n'
    printf 'capability=loom-epistemic-machine-v0\n'
    printf 'capability=loom-epistemic-arrow-projection-v0\n'
    printf 'capability=loom-attention-compiler-v0\n'
    printf 'capability=loom-attention-linear-resource-v0\n'
    printf 'capability=loom-pareto-portfolio-attention-v0\n'
    printf 'capability=loom-atomic-multi-resource-attention-v0\n'
    printf 'capability=loom-robust-contingent-policy-v0\n'
    printf 'capability=loom-atomic-outcome-resource-handoff-v0\n'
    printf 'capability=loom-signed-outcome-authority-v0\n'
    printf 'capability=loom-linear-outcome-evidence-v0\n'
    printf 'capability=loom-journal-head-bound-consume-v0\n'
    printf 'capability=loom-external-witness-mesh-v0\n'
    printf 'capability=loom-quorum-intersection-checkpoint-v0\n'
    printf 'capability=loom-rollback-detection-through-checkpoint-v0\n'
    printf 'capability=loom-external-witness-mesh-v1\n'
    printf 'capability=loom-three-of-four-witness-quorum-v1\n'
    printf 'capability=loom-one-dishonest-honest-intersection-v1\n'
    printf 'capability=loom-one-fault-anchor-and-verify-availability-v1\n'
    printf 'capability=loom-proof-carrying-witness-epoch-handoff-v0\n'
    printf 'capability=loom-joint-old-new-witness-quorum-v0\n'
    printf 'capability=loom-atomic-witness-epoch-activation-v0\n'
    printf 'capability=loom-witness-epoch-crash-recovery-v0\n'
    printf 'capability=loom-external-epoch-transparency-v0\n'
    printf 'capability=loom-materialized-merkle-prefix-verification-v0\n'
    printf 'capability=loom-witnessed-split-view-refusal-v0\n'
    printf 'capability=loom-latest-quorum-witnessed-epoch-rollback-refusal-v0\n'
    printf 'capability=loom-transparency-unreachable-fail-closed-v0\n'
    printf 'capability=loom-post-activation-request-bridge-v1\n'
    printf 'capability=loom-recoverable-control-service-v1\n'
    printf 'capability=loom-beagle-coordination-endpoint-v1\n'
    printf 'capability=loom-separate-pod-inbox-replay-v1\n'
    printf 'capability=loom-signed-continuity-receipt-v2\n'
    printf 'capability=loom-principal-independence-v1\n'
    printf 'capability=loom-independent-measurement-v1\n'
    printf 'capability=loom-observation-authority-v1\n'
    printf 'capability=loom-journal-authority-quorum-v1\n'
    printf 'capability=loom-cross-node-replay-v1\n'
    printf 'capability=loom-cursor-replay-v1\n'
    printf 'capability=loom-exclusive-input-lease-v1\n'
    printf 'capability=loom-read-only-gui-v1\n'
    printf 'capability=loom-fusion-cockpit-v1\n'
    printf 'capability=loom-authority-overlay-v1\n'
    printf 'capability=loom-authority-overlay-v2\n'
    printf 'capability=coord-cockpit-snapshot-v1\n'
    printf 'capability=loom-persistent-provider-custody-v1\n'
    printf 'capability=coord-reply-command-v1\n'
    printf 'capability=loom-coord-transport-v1\n'
    printf 'capability=coord-generation-scoped-wake-v1\n'
    printf 'capability=loom-recoverable-guardian-v1\n'
    printf 'capability=loom-kernel-recovery-v1\n'
    printf 'capability=loom-dual-journal-v1\n'
    printf 'capability=loom-persistent-fleet-catalog-v1\n'
    printf 'capability=loom-fleet-custody-catalog-v2\n'
    printf 'capability=loom-fleet-custody-catalog-v3\n'
    printf 'capability=loom-conflict-free-active-adoption-v1\n'
    printf 'capability=loom-coordination-authority-binding-v1\n'
    printf 'capability=loom-post-pod-reconcile-v1\n'
    printf 'capability=coord-reply-correlation-v1\n'
    printf 'capability=fleet-launcher-v1\n'
    printf 'capability=fleet-proven-exit-v1\n'
    printf 'capability=fleet-home-isolation-v1\n'
    printf 'capability=fleet-presentation-follow-v1\n'
    printf 'capability=fleet-event-log-v1\n'
    printf 'capability=fleet-reconciler-v1\n'
    printf 'capability=fleet-linear-capability-v1\n'
    printf 'capability=fleet-ed25519-anchor-v1\n'
    printf 'capability=fleet-checkpoint-handoff-v1\n'
    printf 'capability=fleet-tla-model-v1\n'
    printf 'capability=fleet-trace-refinement-v1\n'
    printf 'capability=fleet-temporal-authority-v1\n'
    printf 'capability=fleet-recovery-start-only-v1\n'
    printf 'capability=fleet-recovery-directory-v1\n'
    printf 'capability=fleet-recovery-latch-trace-v1\n'
    printf 'installed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$stage/manifest"
  mv "$stage" "$version_dir"
  stage=''
  trap - EXIT
  printf 'INSTALLED runtime_id=%s protocol=%s path=%s\n' \
    "$runtime_id" "$protocol" "$version_dir"
fi

activate_runtime "$runtime_id"
