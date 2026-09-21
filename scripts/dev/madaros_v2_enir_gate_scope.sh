#!/usr/bin/env bash

madaros_v2_enir_gate_scope_or_skip() {
  local base_ref="$1"
  local gate_id="$2"
  local fail_message="$3"
  shift 3

  local protected_paths=("$@")
  local drift_paths
  local drift_list
  local enir_payload_paths=(
    self-hosted/enir
    tools/eisa
  )

  drift_paths="$(git diff --name-only "$base_ref" HEAD -- "${protected_paths[@]}")"
  if [[ -z "$drift_paths" ]]; then
    return 0
  fi
  drift_list="${drift_paths//$'\n'/,}"

  if git diff --quiet "$base_ref" HEAD -- "${enir_payload_paths[@]}"; then
    echo "${gate_id}_SKIP status=skip reason=protected_surface_drift_without_enir_payload base_ref=$base_ref drift_paths=$drift_list"
    exit 0
  fi

  fail "$fail_message: $drift_list"
}
