#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RAW_HEAD="984cb982a8fdb88c3d835bae3e4f8ce993afcce2"
ARENA_HEAD="e226d70ce23f513a8e1fef527171624cf5653301"
PINNED_LEGACY_SHA="f841534799c53be79801c31d218b6f76bb1e7dfe3958b0c441475f516abfe3f7"
ARENA_SOURCE="self-hosted/ir/arena_v2_shadow.sio"
ARENA_WITNESS="tests/native-v2/ir_storage_identity_arena_v2_witness.sio"
ARENA_SOURCE_EXPECTED_SHA="8ac4b0c4e9b9441fc21072ff6258d44afd2d9d094659d2aecb9839f25ccf6e23"
ARENA_WITNESS_EXPECTED_SHA="e34f6178c744f81b46c90c6a0077275e88ee1454c09e625226dfceddd3a63b2f"
SOUC="$ROOT/bin/souc"

mode=""
legacy_elf=""
legacy_sha=""
arena_compiler_elf=""
arena_compiler_sha=""
receipt_json=""

usage() {
  cat <<'EOF'
usage: ir_storage_identity_differential_gate.sh \
  --mode characterize|strict \
  --legacy-elf /absolute/path/to/madaros \
  --legacy-sha SHA256 \
  --arena-compiler-elf /absolute/path/to/madaros \
  --arena-compiler-sha SHA256 \
  --receipt-json /absolute/path/to/receipt.json
EOF
}

fail() {
  printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_FAIL reason=%s\n' "$1" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || fail missing_mode_value
      mode="$2"
      shift 2
      ;;
    --legacy-elf)
      [[ $# -ge 2 ]] || fail missing_legacy_elf_value
      legacy_elf="$2"
      shift 2
      ;;
    --legacy-sha)
      [[ $# -ge 2 ]] || fail missing_legacy_sha_value
      legacy_sha="$2"
      shift 2
      ;;
    --arena-compiler-elf)
      [[ $# -ge 2 ]] || fail missing_arena_compiler_elf_value
      arena_compiler_elf="$2"
      shift 2
      ;;
    --arena-compiler-sha)
      [[ $# -ge 2 ]] || fail missing_arena_compiler_sha_value
      arena_compiler_sha="$2"
      shift 2
      ;;
    --receipt-json)
      [[ $# -ge 2 ]] || fail missing_receipt_json_value
      receipt_json="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "unknown_argument_$1"
      ;;
  esac
done

[[ "$mode" == "characterize" || "$mode" == "strict" ]] || fail invalid_mode
[[ "$legacy_elf" == /* ]] || fail legacy_elf_must_be_absolute
[[ "$arena_compiler_elf" == /* ]] || fail arena_compiler_elf_must_be_absolute
[[ "$receipt_json" == /* ]] || fail receipt_json_must_be_absolute
[[ -f "$legacy_elf" && -x "$legacy_elf" ]] || fail legacy_elf_not_executable
[[ -f "$arena_compiler_elf" && -x "$arena_compiler_elf" ]] || fail arena_compiler_elf_not_executable
[[ -x "$SOUC" ]] || fail souc_not_executable
[[ -f "$ARENA_SOURCE" ]] || fail arena_source_missing
[[ -f "$ARENA_WITNESS" ]] || fail arena_witness_missing
command -v jq >/dev/null 2>&1 || fail jq_missing

raw_head_ancestor=false
arena_head_ancestor=false
git merge-base --is-ancestor "$RAW_HEAD" HEAD && raw_head_ancestor=true
git merge-base --is-ancestor "$ARENA_HEAD" HEAD && arena_head_ancestor=true

magic="$(head -c4 "$legacy_elf" | od -An -tx1 | tr -d '[:space:]')"
[[ "$magic" == "7f454c46" ]] || fail legacy_elf_not_raw_elf
actual_legacy_sha="$(sha256sum "$legacy_elf" | awk '{print $1}')"
[[ "$actual_legacy_sha" == "$legacy_sha" ]] || fail legacy_sha_mismatch
arena_magic="$(head -c4 "$arena_compiler_elf" | od -An -tx1 | tr -d '[:space:]')"
[[ "$arena_magic" == "7f454c46" ]] || fail arena_compiler_not_raw_elf
actual_arena_compiler_sha="$(sha256sum "$arena_compiler_elf" | awk '{print $1}')"
[[ "$actual_arena_compiler_sha" == "$arena_compiler_sha" ]] || fail arena_compiler_sha_mismatch
if [[ "$mode" == "characterize" && "$legacy_sha" != "$PINNED_LEGACY_SHA" ]]; then
  fail characterize_requires_pinned_legacy_sha
fi

mkdir -p "$(dirname "$receipt_json")"
receipt_stem="${receipt_json%.json}"
legacy_log="${receipt_stem}.legacy.log"
arena_log="${receipt_stem}.arena.log"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-storage-identity-diff.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

arena_source_sha_before="$(sha256sum "$ARENA_SOURCE" | awk '{print $1}')"
[[ "$arena_source_sha_before" == "$ARENA_SOURCE_EXPECTED_SHA" ]] || fail arena_source_not_pinned_head
arena_witness_sha_before="$(sha256sum "$ARENA_WITNESS" | awk '{print $1}')"
[[ "$arena_witness_sha_before" == "$ARENA_WITNESS_EXPECTED_SHA" ]] || fail arena_witness_not_pinned_contract
differential_gate_sha_before="$(sha256sum scripts/ci/ir_storage_identity_differential_gate.sh | awk '{print $1}')"
souc_sha_before="$(sha256sum "$SOUC" | awk '{print $1}')"
arena_source_snapshot="$tmp/arena_v2_shadow.sio"
arena_witness_snapshot="$tmp/arena_v2_witness.sio"
cp "$ARENA_SOURCE" "$arena_source_snapshot"
cp "$ARENA_WITNESS" "$arena_witness_snapshot"
[[ "$(sha256sum "$arena_source_snapshot" | awk '{print $1}')" == "$arena_source_sha_before" ]] || fail arena_source_snapshot_mismatch
[[ "$(sha256sum "$arena_witness_snapshot" | awk '{print $1}')" == "$arena_witness_sha_before" ]] || fail arena_witness_snapshot_mismatch

set +e
timeout 300 "$legacy_elf" --ir-heap-bridge-self-test >"$legacy_log" 2>&1
legacy_rc=$?
set -e

legacy_state="unexpected"
legacy_observed=-1
if [[ "$legacy_rc" -ne 0 ]] &&
   grep -Fxq 'IR_MODULE_HEAP_BRIDGE_OBSERVED scalar=0' "$legacy_log" &&
   grep -Fxq 'IR_MODULE_HEAP_BRIDGE_STAGE_FAIL code=174' "$legacy_log" &&
   grep -Fxq 'IR_MODULE_HEAP_BRIDGE_FAIL reason=semantic_assertion' "$legacy_log" &&
   [[ "$(grep -Ec '^IR_MODULE_HEAP_BRIDGE_OBSERVED ' "$legacy_log")" -eq 1 ]] &&
   [[ "$(grep -Ec '^IR_MODULE_HEAP_BRIDGE_STAGE_FAIL ' "$legacy_log")" -eq 1 ]] &&
   [[ "$(grep -Ec '^IR_MODULE_HEAP_BRIDGE_FAIL ' "$legacy_log")" -eq 1 ]] &&
   ! grep -Fxq 'IR_MODULE_HEAP_BRIDGE_PASS' "$legacy_log"; then
  legacy_state="known_code174"
  legacy_observed=0
elif [[ "$legacy_rc" -eq 0 ]] &&
     grep -Fxq 'IR_MODULE_HEAP_BRIDGE_IDENTITY operation=scalar_mutation_fresh_reload mutation=1 observed=1 status=preserved' "$legacy_log" &&
     grep -Fxq 'IR_MODULE_HEAP_BRIDGE_PASS' "$legacy_log" &&
     ! grep -Fq 'IR_MODULE_HEAP_BRIDGE_FAIL' "$legacy_log"; then
  legacy_state="preserved"
  legacy_observed=1
fi

composite="$tmp/ir_storage_identity_arena_v2_composite.sio"
sed '/^module ir::arena_v2_shadow$/d' "$arena_source_snapshot" >"$composite"
sed '/^use ir::arena_v2_shadow::\*$/d' "$arena_witness_snapshot" >>"$composite"

arena_check_rc=0
env -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE -u SOUNIO_MADAROS_BIN \
  MADAROS_RAW_BIN="$arena_compiler_elf" timeout 300 "$SOUC" check "$composite" >"$tmp/arena-check.log" 2>&1 || arena_check_rc=$?
if [[ "$arena_check_rc" -eq 0 ]]; then
  set +e
  env -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE -u SOUNIO_MADAROS_BIN \
    MADAROS_RAW_BIN="$arena_compiler_elf" timeout 300 "$SOUC" run "$composite" >"$arena_log" 2>&1
  arena_rc=$?
  set -e
else
  arena_rc=-1
  cp "$tmp/arena-check.log" "$arena_log"
fi

arena_state="unexpected"
arena_observed=-1
if [[ "$arena_check_rc" -eq 0 && "$arena_rc" -eq 0 ]] &&
   grep -Fxq 'IR_STORAGE_IDENTITY_ARENA_V2 operation=scalar_mutation_fresh_lookup mutation=1 observed=1 status=preserved' "$arena_log" &&
   grep -Fxq 'IR_STORAGE_IDENTITY_ARENA_V2_PASS' "$arena_log" &&
   [[ "$(grep -Ec '^IR_STORAGE_IDENTITY_ARENA_V2 operation=' "$arena_log")" -eq 1 ]] &&
   [[ "$(grep -Ec '^IR_STORAGE_IDENTITY_ARENA_V2_PASS$' "$arena_log")" -eq 1 ]] &&
   ! grep -Fq 'IR_STORAGE_IDENTITY_ARENA_V2_FAIL' "$arena_log"; then
  arena_state="preserved"
  arena_observed=1
fi

protocol_comparable=false
if [[ "$arena_state" == "preserved" &&
      ( "$legacy_state" == "known_code174" || "$legacy_state" == "preserved" ) ]]; then
  protocol_comparable=true
fi
preserved_value_parity=false
if [[ "$legacy_observed" -eq 1 && "$arena_observed" -eq 1 ]]; then
  preserved_value_parity=true
fi

status="unexpected_evidence"
gate_rc=1
promotion_ready=false
if [[ "$arena_state" == "preserved" ]]; then
  if [[ "$mode" == "characterize" && "$legacy_state" == "known_code174" ]]; then
    status="known_divergence"
    gate_rc=0
  elif [[ "$mode" == "strict" && "$preserved_value_parity" == "true" ]]; then
    status="identity_protocol_parity"
    promotion_ready=true
    gate_rc=0
  elif [[ "$mode" == "strict" && "$legacy_state" == "known_code174" ]]; then
    status="blocked_identity_divergence"
    gate_rc=42
  fi
fi

legacy_log_sha="$(sha256sum "$legacy_log" | awk '{print $1}')"
arena_log_sha="$(sha256sum "$arena_log" | awk '{print $1}')"
arena_source_sha_after="$(sha256sum "$ARENA_SOURCE" | awk '{print $1}')"
arena_witness_sha_after="$(sha256sum "$ARENA_WITNESS" | awk '{print $1}')"
differential_gate_sha_after="$(sha256sum scripts/ci/ir_storage_identity_differential_gate.sh | awk '{print $1}')"
souc_sha_after="$(sha256sum "$SOUC" | awk '{print $1}')"
legacy_sha_after="$(sha256sum "$legacy_elf" | awk '{print $1}')"
arena_compiler_sha_after="$(sha256sum "$arena_compiler_elf" | awk '{print $1}')"
[[ "$arena_source_sha_after" == "$arena_source_sha_before" ]] || fail arena_source_changed_during_run
[[ "$arena_witness_sha_after" == "$arena_witness_sha_before" ]] || fail arena_witness_changed_during_run
[[ "$differential_gate_sha_after" == "$differential_gate_sha_before" ]] || fail differential_gate_changed_during_run
[[ "$souc_sha_after" == "$souc_sha_before" ]] || fail souc_wrapper_changed_during_run
[[ "$legacy_sha_after" == "$actual_legacy_sha" ]] || fail legacy_elf_changed_during_run
[[ "$arena_compiler_sha_after" == "$actual_arena_compiler_sha" ]] || fail arena_compiler_changed_during_run
integration_head="$(git rev-parse HEAD)"
worktree_clean=false
if [[ -z "$(git status --porcelain)" ]]; then
  worktree_clean=true
fi

jq -n \
  --arg schema 'sounio.ir-storage-identity-differential.v1' \
  --arg mode "$mode" \
  --arg status "$status" \
  --arg integration_head "$integration_head" \
  --arg raw_head "$RAW_HEAD" \
  --arg arena_head "$ARENA_HEAD" \
  --argjson raw_head_ancestor "$raw_head_ancestor" \
  --argjson arena_head_ancestor "$arena_head_ancestor" \
  --argjson worktree_clean "$worktree_clean" \
  --arg legacy_elf "$legacy_elf" \
  --arg legacy_sha "$actual_legacy_sha" \
  --arg legacy_state "$legacy_state" \
  --argjson legacy_rc "$legacy_rc" \
  --argjson legacy_observed "$legacy_observed" \
  --arg legacy_log "$legacy_log" \
  --arg legacy_log_sha "$legacy_log_sha" \
  --arg souc "$SOUC" \
  --arg souc_sha "$souc_sha_before" \
  --arg arena_compiler_elf "$arena_compiler_elf" \
  --arg arena_compiler_sha "$actual_arena_compiler_sha" \
  --arg arena_source_sha "$arena_source_sha_before" \
  --arg arena_witness_sha "$arena_witness_sha_before" \
  --arg differential_gate_sha "$differential_gate_sha_before" \
  --arg arena_state "$arena_state" \
  --argjson arena_check_rc "$arena_check_rc" \
  --argjson arena_rc "$arena_rc" \
  --argjson arena_observed "$arena_observed" \
  --arg arena_log "$arena_log" \
  --arg arena_log_sha "$arena_log_sha" \
  --argjson protocol_comparable "$protocol_comparable" \
  --argjson preserved_value_parity "$preserved_value_parity" \
  --argjson promotion_ready "$promotion_ready" \
  '{
    schema: $schema,
    mode: $mode,
    status: $status,
    scope: {
      operation: "identity_stable_scalar_mutation_then_fresh_authority_lookup",
      protocol_comparable: $protocol_comparable,
      observational_equivalence: $preserved_value_parity,
      payload_equivalence: false,
      same_build: false,
      full_ir_parity: false
    },
    provenance: {
      integration_head: $integration_head,
      raw_head: $raw_head,
      arena_head: $arena_head,
      raw_head_ancestor: $raw_head_ancestor,
      arena_head_ancestor: $arena_head_ancestor,
      worktree_clean: $worktree_clean,
      differential_gate_sha256: $differential_gate_sha
    },
    legacy: {
      backend: "IrModuleHeapBridge_raw_managed_handle",
      executable: $legacy_elf,
      sha256: $legacy_sha,
      process_rc: $legacy_rc,
      state: $legacy_state,
      stage_code: (if $legacy_state == "known_code174" then 174 else 0 end),
      mutation: 1,
      fresh_reload_observed: $legacy_observed,
      log: $legacy_log,
      log_sha256: $legacy_log_sha
    },
    arena_v2: {
      backend: "IrModuleArenaV2_shadow_typed_handle",
      compiler_wrapper: $souc,
      compiler_wrapper_sha256: $souc_sha,
      compiler_elf: $arena_compiler_elf,
      compiler_elf_sha256: $arena_compiler_sha,
      source_sha256: $arena_source_sha,
      witness_sha256: $arena_witness_sha,
      shadow_acceptance_gate: "not_run_in_differential",
      check_rc: $arena_check_rc,
      process_rc: $arena_rc,
      state: $arena_state,
      mutation: 1,
      fresh_lookup_observed: $arena_observed,
      log: $arena_log,
      log_sha256: $arena_log_sha
    },
    comparison: {
      observations_equal: ($legacy_observed == $arena_observed),
      expected_preserved_value: 1,
      preserved_value_parity: $preserved_value_parity
    },
    promotion: {
      ready_for_identity_operation_only: $promotion_ready,
      selected_backend: "not_evaluated",
      selection_substitution_allowed: false,
      legacy_kept_as_oracle: true
    }
  }' >"$receipt_json"

printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_RECEIPT mode=%s status=%s legacy_state=%s legacy_observed=%s arena_state=%s arena_observed=%s\n' \
  "$mode" "$status" "$legacy_state" "$legacy_observed" "$arena_state" "$arena_observed"
printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_SCOPE protocol_comparable=%s observational_equivalence=%s payload_equivalence=false same_build=false full_ir_parity=false legacy_kept=true\n' \
  "$protocol_comparable" "$preserved_value_parity"

if [[ "$gate_rc" -eq 0 ]]; then
  printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_PASS receipt=%s\n' "$receipt_json"
elif [[ "$gate_rc" -eq 42 ]]; then
  printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_BLOCKED reason=observations_differ receipt=%s\n' "$receipt_json" >&2
else
  printf 'IR_STORAGE_IDENTITY_DIFFERENTIAL_FAIL reason=unexpected_evidence receipt=%s\n' "$receipt_json" >&2
fi
exit "$gate_rc"
