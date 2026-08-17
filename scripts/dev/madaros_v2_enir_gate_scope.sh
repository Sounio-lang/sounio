#!/usr/bin/env bash

madaros_v2_enir_gate_scope_or_skip() {
  local base_ref="$1"
  local gate_id="$2"
  local fail_message="$3"
  shift 3

  local protected_paths=("$@")
  local enir_payload_paths=(
    self-hosted/enir
    tools/eisa
  )

  if git diff --quiet "$base_ref" HEAD -- "${protected_paths[@]}"; then
    return 0
  fi

  if git diff --quiet "$base_ref" HEAD -- "${enir_payload_paths[@]}"; then
    echo "${gate_id}_SKIP not_applicable=protected_surface_changed_without_enir_payload"
    exit 0
  fi

  fail "$fail_message"
}
