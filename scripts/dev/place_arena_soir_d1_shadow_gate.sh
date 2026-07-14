#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

INTEGRATION_HEAD="4bc8c609d7d342d9dcf5fb8358d229fb99b70a24"
PLACE_HEAD="bfc6a7ca2436451946ff239164d7f6454824ac1f"
PLACE_PATH="self-hosted/ir/place_v0.sio"
PLACE_BLOB="7b00b7bd7b838856f6b297a47b0d0496b16512cb"
PLACE_SHA256="b4613b4eb40afef8ca03ff9b02ec08fab59e82a16bd72096f006e6daf48e7e91"
ARENA_HEAD="e226d70ce23f513a8e1fef527171624cf5653301"
ARENA_PATH="self-hosted/ir/arena_v2_shadow.sio"
ARENA_BLOB="17b92116da84e0c2ec4d2ef3860cc3b0378de4dc"
ARENA_SHA256="8ac4b0c4e9b9441fc21072ff6258d44afd2d9d094659d2aecb9839f25ccf6e23"
WRITER_HEAD="02f876b48d4656eb5f68695d92ea20eeb29d4ef6"
WRITER_PATH="self-hosted/ir/soir_writer.sio"
WRITER_BLOB="bb8634991af5e26d1c74e570bcb09fca292e8a2b"
WRITER_SHA256="1b9b683158f6ff50783617d66a03d35186ca8206ee39cb308d1cd29b53655bf2"

CI_RUN_ID="29338391194"
CI_RUN_URL="https://github.com/Sounio-lang/sounio/actions/runs/29338391194"
CI_ARTIFACT_ID="8312984651"
CI_ARTIFACT_NAME="native-compiler-linux-x86_64"
CI_ARTIFACT_FILE="souc-stage2"
COMPILER="/tmp/sounio-d1-compiler-899/souc-stage2"
COMPILER_SHA256="204dc3665af5bb1cc4dff298bcfffe15f5331d7d4604cbd3d49648724c2b9476"

FULL_DRIVER="/tmp/madaros-current-source-f64-lowering-899/madaros"
FULL_DRIVER_SHA256="f841534799c53be79801c31d218b6f76bb1e7dfe3958b0c441475f516abfe3f7"
WITNESS="tests/native-v2/place_arena_soir_d1_witness.sio"
GATE="scripts/dev/place_arena_soir_d1_shadow_gate.sh"
RECEIPT_JSON="/tmp/sounio-place-arena-soir-d1-receipt.json"
PATH_FINGERPRINT="154461756"
MAP_FINGERPRINT="700112127"

usage() {
  echo "usage: $0 [--compiler-elf /absolute/path] [--receipt-json /absolute/path]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiler-elf)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      COMPILER="$2"
      shift 2
      ;;
    --receipt-json)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RECEIPT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ "$RECEIPT_JSON" = /* ]] || { echo "receipt path must be absolute" >&2; exit 2; }
[[ "$COMPILER" = /* ]] || { echo "compiler ELF path must be absolute" >&2; exit 2; }
[[ -x "$COMPILER" ]] || { echo "missing Stage2 compiler: $COMPILER" >&2; exit 1; }
[[ -x "$FULL_DRIVER" ]] || { echo "missing pinned full Madaros driver: $FULL_DRIVER" >&2; exit 1; }

EVIDENCE_HEAD="$(git rev-parse HEAD)"
BASE_HEAD_EQUAL=false
[[ "$EVIDENCE_HEAD" == "$INTEGRATION_HEAD" ]] && BASE_HEAD_EQUAL=true
BASE_ANCESTOR=false
if git merge-base --is-ancestor "$INTEGRATION_HEAD" "$EVIDENCE_HEAD"; then BASE_ANCESTOR=true; fi
[[ "$BASE_ANCESTOR" == true ]] || {
  echo "integration head $INTEGRATION_HEAD is not an ancestor of $EVIDENCE_HEAD" >&2
  exit 1
}

# Both the committed and in-progress delta must remain within the authorized
# two-file surface. This permits a pre-commit gate after the first D1 commit.
mapfile -t committed_paths < <(git diff --name-only "$INTEGRATION_HEAD"..HEAD | sort)
[[ "${committed_paths[*]}" == "$GATE $WITNESS" ]] || {
  echo "committed delta from integration head is not exactly the two authorized files" >&2
  printf '%s\n' "${committed_paths[@]}" >&2
  exit 1
}
mapfile -t dirty_paths < <(git status --short | sed 's/^...//' | sort)
for dirty_path in "${dirty_paths[@]}"; do
  [[ "$dirty_path" == "$GATE" || "$dirty_path" == "$WITNESS" ]] || {
    echo "unauthorized dirty path: $dirty_path" >&2
    exit 1
  }
done
WORKTREE_CLEAN=false
[[ ${#dirty_paths[@]} -eq 0 ]] && WORKTREE_CLEAN=true

head_is_integration_ancestor() {
  local head="$1"
  if git merge-base --is-ancestor "$head" "$INTEGRATION_HEAD"; then
    echo true
  else
    echo false
  fi
}

PLACE_HEAD_ANCESTOR="$(head_is_integration_ancestor "$PLACE_HEAD")"
ARENA_HEAD_ANCESTOR="$(head_is_integration_ancestor "$ARENA_HEAD")"
WRITER_HEAD_ANCESTOR="$(head_is_integration_ancestor "$WRITER_HEAD")"

TMP="$(mktemp -d /tmp/sounio-place-arena-soir-d1.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
PLACE_SNAPSHOT="$TMP/place_v0.sio"
ARENA_SNAPSHOT="$TMP/arena_v2_shadow.sio"
WRITER_SNAPSHOT="$TMP/soir_writer.sio"
COMPOSITE="$TMP/place_arena_soir_d1_composite.sio"
ELF="$TMP/place_arena_soir_d1"
OUTPUT="$TMP/output.txt"
FULL_SELFTEST_OUTPUT="$TMP/full-madaros-selftest.log"
FULL_CHECK_OUTPUT="$TMP/full-madaros-check.log"
FULL_RUN_OUTPUT="$TMP/full-madaros-run.log"

verify_snapshot() {
  local head="$1" path="$2" blob="$3" content_sha="$4" output="$5"
  [[ "$(git rev-parse "$head:$path")" == "$blob" ]] || {
    echo "object mismatch for $head:$path" >&2
    exit 1
  }
  git show "$head:$path" > "$output"
  [[ "$(sha256sum "$output" | awk '{print $1}')" == "$content_sha" ]] || {
    echo "content mismatch for $head:$path" >&2
    exit 1
  }
}

verify_snapshot "$PLACE_HEAD" "$PLACE_PATH" "$PLACE_BLOB" "$PLACE_SHA256" "$PLACE_SNAPSHOT"
verify_snapshot "$ARENA_HEAD" "$ARENA_PATH" "$ARENA_BLOB" "$ARENA_SHA256" "$ARENA_SNAPSHOT"
verify_snapshot "$WRITER_HEAD" "$WRITER_PATH" "$WRITER_BLOB" "$WRITER_SHA256" "$WRITER_SNAPSHOT"
[[ "$(sha256sum "$COMPILER" | awk '{print $1}')" == "$COMPILER_SHA256" ]] || {
  echo "Stage2 compiler hash mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$FULL_DRIVER" | awk '{print $1}')" == "$FULL_DRIVER_SHA256" ]] || {
  echo "full Madaros driver hash mismatch" >&2
  exit 1
}

# D1 remains absent from default imports and compiler/bootstrap entrypoints.
if rg -n 'place_arena_soir_d1' self-hosted/ir/mod.sio self-hosted/compiler/main.sio scripts/bootstrap/bootstrap_concat.sh >/dev/null; then
  echo "D1 unexpectedly appears in a default pipeline surface" >&2
  exit 1
fi

sed '/^module ir::arena_v2_shadow$/d' "$ARENA_SNAPSHOT" > "$COMPOSITE"
printf '\n' >> "$COMPOSITE"
cat "$PLACE_SNAPSHOT" >> "$COMPOSITE"
printf '\n' >> "$COMPOSITE"
cat "$WITNESS" >> "$COMPOSITE"

COMPOSITE_SHA256="$(sha256sum "$COMPOSITE" | awk '{print $1}')"
WITNESS_SHA256="$(sha256sum "$WITNESS" | awk '{print $1}')"
GATE_SHA256="$(sha256sum "$GATE" | awk '{print $1}')"

# Exactly one Stage2 composite compile and one emitted-ELF execution establish
# the structured characterization. Full-driver evidence is reported separately.
timeout 300 "$COMPILER" "$COMPOSITE" "$ELF"
[[ -f "$ELF" ]] || { echo "compiler did not emit an ELF" >&2; exit 1; }
chmod +x "$ELF"
ELF_SHA256="$(sha256sum "$ELF" | awk '{print $1}')"
timeout 60 "$ELF" | tee "$OUTPUT"

require_once() {
  local line="$1"
  [[ "$(grep -Fxc "$line" "$OUTPUT")" == 1 ]] || {
    echo "expected exactly one runtime receipt: $line" >&2
    exit 1
  }
}

require_once 'D1_SYNTHETIC_VALUE_ONLY original_value=42 n1_value=42 collision=true status=partial classification=MORE_EVIDENCE_REQUIRED d1_comparable_legacy_oracle_executed=false'
require_once 'D1_KIND_BOUNDARY direct=3 by_value=3 by_ref=0 blocker=abi-transport/stage2_place_projection_ref_enum_field'
require_once "D1_STRUCTURED status=pass structured_exact_roundtrip=true payload_roundtrip=true path=Deref/Field/Index path_fingerprint=$PATH_FINGERPRINT logical_id=7001 source_arena=101 source_function_slot=0 source_function_generation=1 source_instr_slot=0 source_instr_generation=1 decoded_arena=202 decoded_function_slot=1 decoded_function_generation=2 decoded_instr_slot=1 decoded_instr_generation=2 map_fingerprint=$MAP_FINGERPRINT rekeyed=true linkage=test_local_external_rekey_map place_stored_in_arena=false"
require_once 'D1_NEGATIVE name=projection_type_layout_mismatch status=pass code=201 count=1 backend_ops=0 mutation=swap_complete_field_index_records'
require_once 'D1_NEGATIVE name=write_to_shared status=pass code=202 place_code=110 count=1 backend_ops=0 mutation=access_write'
require_once 'D1_NEGATIVE name=cross_module_identity status=pass code=203 count=1 backend_ops=0 mutation=header_module_only'
require_once 'D1_NEGATIVE name=stale_runtime_handle status=pass code=204 count=1 backend_ops=0 mutation=serialized_stale_candidate candidate_arena=202 candidate_slot=1 candidate_generation=1'
require_once 'D1_NEGATIVE name=invalid_root_id status=pass code=205 count=1 backend_ops=0 mutation=root_id_9999'
require_once 'D1_NEGATIVE name=unknown_access_tag status=pass code=206 count=1 backend_ops=0 mutation=access_tag_99'
require_once 'PLACE_ARENA_SOIR_D1_PARTIAL structured=pass synthetic_value_only=partial legacy_control=MORE_EVIDENCE_REQUIRED same_build=true default_pipeline=false legacy_kept=true d1_comparable_legacy_oracle_executed=false promotion_ready=false writer_contract_only=true backend_ops=0 actual_soir_version=none arena_public_payload_accessor=false composite_private_access=true blocker=abi-transport/stage2_tuple_index_2'
[[ "$(grep -c '^D1_NEGATIVE ' "$OUTPUT")" == 6 ]] || {
  echo "negative matrix must contain exactly six receipts" >&2
  exit 1
}
if grep -q 'D1_RAW_CONTROL\|status=information_loss' "$OUTPUT"; then
  echo "synthetic value-only characterization was mislabeled as a legacy/raw oracle" >&2
  exit 1
fi

# Keep the pinned full-driver characterization live and reproducible. This is
# separate from the single Stage2 compile/run above and is not a cross-artifact
# success claim.
set +e
timeout 300 "$FULL_DRIVER" --ir-heap-bridge-self-test >"$FULL_SELFTEST_OUTPUT" 2>&1
FULL_SELFTEST_RC=$?
(
  cd "$TMP" || exit 1
  timeout 300 "$FULL_DRIVER" check "$COMPOSITE"
) >"$FULL_CHECK_OUTPUT" 2>&1
FULL_CHECK_RC=$?
(
  cd "$TMP" || exit 1
  timeout 300 "$FULL_DRIVER" run "$COMPOSITE"
) >"$FULL_RUN_OUTPUT" 2>&1
FULL_RUN_RC=$?
set -e

[[ "$FULL_SELFTEST_RC" == 1 ]] || {
  cat "$FULL_SELFTEST_OUTPUT" >&2
  echo "expected pinned full Madaros heap-bridge self-test rc=1, got $FULL_SELFTEST_RC" >&2
  exit 1
}
grep -Fx 'IR_MODULE_HEAP_BRIDGE_OBSERVED scalar=0' "$FULL_SELFTEST_OUTPUT" >/dev/null
grep -Fx 'IR_MODULE_HEAP_BRIDGE_STAGE_FAIL code=174' "$FULL_SELFTEST_OUTPUT" >/dev/null
[[ "$FULL_CHECK_RC" == 0 ]] || {
  cat "$FULL_CHECK_OUTPUT" >&2
  echo "expected pinned full Madaros D1 check rc=0, got $FULL_CHECK_RC" >&2
  exit 1
}
[[ "$FULL_RUN_RC" == 128 ]] || {
  cat "$FULL_RUN_OUTPUT" >&2
  echo "expected pinned full Madaros D1 run rc=128, got $FULL_RUN_RC" >&2
  exit 1
}
grep -Fx 'D1_KIND_BOUNDARY direct=3 by_value=3 by_ref=3 blocker=none' "$FULL_RUN_OUTPUT" >/dev/null
grep -Fx 'D1_NEGATIVE name=cross_module_identity status=pass code=203 count=1 backend_ops=0 mutation=header_module_only' "$FULL_RUN_OUTPUT" >/dev/null
grep -q '^PLACE_ARENA_SOIR_D1_FAIL code=128' "$FULL_RUN_OUTPUT"

FULL_SELFTEST_SHA256="$(sha256sum "$FULL_SELFTEST_OUTPUT" | awk '{print $1}')"
FULL_CHECK_SHA256="$(sha256sum "$FULL_CHECK_OUTPUT" | awk '{print $1}')"
FULL_RUN_SHA256="$(sha256sum "$FULL_RUN_OUTPUT" | awk '{print $1}')"

mkdir -p "$(dirname "$RECEIPT_JSON")"
jq -n \
  --arg integration_head "$INTEGRATION_HEAD" \
  --arg evidence_head "$EVIDENCE_HEAD" \
  --argjson base_head_equal "$BASE_HEAD_EQUAL" \
  --argjson base_ancestor "$BASE_ANCESTOR" \
  --argjson worktree_clean "$WORKTREE_CLEAN" \
  --arg place_head "$PLACE_HEAD" \
  --arg place_blob "$PLACE_BLOB" \
  --arg place_sha256 "$PLACE_SHA256" \
  --argjson place_head_ancestor "$PLACE_HEAD_ANCESTOR" \
  --arg arena_head "$ARENA_HEAD" \
  --arg arena_blob "$ARENA_BLOB" \
  --arg arena_sha256 "$ARENA_SHA256" \
  --argjson arena_head_ancestor "$ARENA_HEAD_ANCESTOR" \
  --arg writer_head "$WRITER_HEAD" \
  --arg writer_blob "$WRITER_BLOB" \
  --arg writer_sha256 "$WRITER_SHA256" \
  --argjson writer_head_ancestor "$WRITER_HEAD_ANCESTOR" \
  --arg ci_run_id "$CI_RUN_ID" \
  --arg ci_run_url "$CI_RUN_URL" \
  --arg ci_artifact_id "$CI_ARTIFACT_ID" \
  --arg ci_artifact_name "$CI_ARTIFACT_NAME" \
  --arg ci_artifact_file "$CI_ARTIFACT_FILE" \
  --arg compiler_path "$COMPILER" \
  --arg compiler_sha256 "$COMPILER_SHA256" \
  --arg full_driver_path "$FULL_DRIVER" \
  --arg full_driver_sha256 "$FULL_DRIVER_SHA256" \
  --arg full_selftest_sha256 "$FULL_SELFTEST_SHA256" \
  --arg full_check_sha256 "$FULL_CHECK_SHA256" \
  --arg full_run_sha256 "$FULL_RUN_SHA256" \
  --arg witness_sha256 "$WITNESS_SHA256" \
  --arg gate_sha256 "$GATE_SHA256" \
  --arg composite_sha256 "$COMPOSITE_SHA256" \
  --arg elf_sha256 "$ELF_SHA256" \
  --argjson path_fingerprint "$PATH_FINGERPRINT" \
  --argjson map_fingerprint "$MAP_FINGERPRINT" \
  '{
    schema: "sounio.place-arena-soir-diff.v1",
    status: "partial",
    integration_head: $integration_head,
    evidence_head: $evidence_head,
    base_head_equal: $base_head_equal,
    base_ancestor: $base_ancestor,
    worktree_clean: $worktree_clean,
    snapshots: {
      place: {head: $place_head, object: $place_blob, content_sha256: $place_sha256, object_verified: true, content_verified: true, head_ancestor_of_integration: $place_head_ancestor},
      arena: {head: $arena_head, object: $arena_blob, content_sha256: $arena_sha256, object_verified: true, content_verified: true, head_ancestor_of_integration: $arena_head_ancestor},
      writer_contract: {head: $writer_head, object: $writer_blob, content_sha256: $writer_sha256, object_verified: true, content_verified: true, head_ancestor_of_integration: $writer_head_ancestor}
    },
    ci_artifact_provenance: {
      run_id: ($ci_run_id | tonumber), run_url: $ci_run_url,
      run_head: $integration_head, artifact_id: ($ci_artifact_id | tonumber),
      artifact_name: $ci_artifact_name, artifact_file: $ci_artifact_file,
      materialized_path: $compiler_path, file_sha256: $compiler_sha256
    },
    stage2_characterization: {
      status: "pass", compile_count: 1, run_count: 1,
      compiler_sha256: $compiler_sha256,
      composite_sha256: $composite_sha256,
      emitted_elf_sha256: $elf_sha256,
      same_build: true
    },
    structured: {
      status: "pass", structured_exact_roundtrip: true,
      payload_roundtrip: true, decoded_payload: {opcode: 9, value: 42},
      path: ["Deref", "Field", "Index"], path_fingerprint: $path_fingerprint,
      original: {arena_identity: 101, function: {slot: 0, generation: 1}, instr: {slot: 0, generation: 1}},
      decoded: {arena_identity: 202, function: {slot: 1, generation: 2}, instr: {slot: 1, generation: 2}},
      rekey_map: {logical_id: 7001, function_slot: 1, function_generation: 2, instr_slot: 1, instr_generation: 2, fingerprint: $map_fingerprint},
      raw_runtime_handle_fields: {serialized: false, arena_identity: null, module_identity: null, slot: null, generation: null},
      linkage: "test_local_external_rekey_map",
      place_stored_in_arena: false
    },
    synthetic_value_only: {
      status: "partial", classification: "MORE_EVIDENCE_REQUIRED",
      original_value: 42, n1_value: 42, collision: true,
      established_information_loss: false, d1_comparable_legacy_oracle_executed: false
    },
    negatives: [
      {name: "projection_type_layout_mismatch", code: 201, status: "pass", count: 1, backend_ops: 0, mutation: "swap_complete_field_index_records"},
      {name: "write_to_shared", code: 202, place_code: 110, status: "pass", count: 1, backend_ops: 0, mutation: "access_write"},
      {name: "cross_module_identity", code: 203, status: "pass", count: 1, backend_ops: 0, mutation: "header_module_only"},
      {name: "stale_runtime_handle", code: 204, status: "pass", count: 1, backend_ops: 0, mutation: "serialized_stale_candidate", candidate: {arena_identity: 202, slot: 1, generation: 1}},
      {name: "invalid_root_id", code: 205, status: "pass", count: 1, backend_ops: 0, mutation: "root_id_9999"},
      {name: "unknown_access_tag", code: 206, status: "pass", count: 1, backend_ops: 0, mutation: "access_tag_99"}
    ],
    transport: {
      arena_public_payload_accessor: false,
      composite_private_access: true,
      blocker: "abi-transport/stage2_tuple_index_2",
      value_semantics_blocker: "abi-transport/stage2_place_projection_ref_enum_field",
      projection_kind_boundary: {direct: 3, enum_by_value: 3, aggregate_by_ref: 0}
    },
    full_madaros_runtime: {
      status: "blocked",
      driver_path: $full_driver_path,
      driver_sha256: $full_driver_sha256,
      evidence_source: "same_gate_live_probe",
      legacy_heap_selftest: {executed: true, observed_value: 0, diagnostic_code: 174, rc: 1, log_sha256: $full_selftest_sha256},
      d1_composite_check: {rc: 0, log_sha256: $full_check_sha256},
      public_run: {
        rc: 128,
        failure_phase: "negative_matrix_n4_decode_after_cumulative_fixed_array_alias",
        log_sha256: $full_run_sha256,
        blocker_id: "BLK-20260714-madaros-fixed-array-ident-copy-alias"
      },
      cross_artifact_success_claimed: false
    },
    limitations: [
      "synthetic_value_only_is_not_a_legacy_or_raw_oracle",
      "d1_comparable_legacy_oracle_not_executed",
      "full_madaros_check_passes_but_public_run_is_blocked_at_n4_decode_rc_128_after_prior_wire_copies_alias",
      "full_madaros_legacy_heap_selftest_executed_observed_0_code_174_rc_1",
      "stage2_tuple_index_2_requires_test_only_composite_private_payload_access",
      "stage2_place_projection_ref_loses_enum_field_while_direct_and_by_value_are_correct",
      "place_metadata_is_not_stored_in_arena_and_current_linkage_is_a_test_local_external_rekey_map",
      "no_actual_soir_v5_or_v6",
      "no_compiler_selftest_claim"
    ],
    default_pipeline: false,
    legacy_kept: true,
    legacy_kept_meaning: "legacy_path_contained_d1_comparable_oracle_not_executed",
    promotion_ready: false,
    writer_contract_only: true,
    actual_soir_version: null,
    compiler_selftest_claimed: false
  }' > "$RECEIPT_JSON"

RECEIPT_SHA256="$(sha256sum "$RECEIPT_JSON" | awk '{print $1}')"
echo "PLACE_ARENA_SOIR_D1_GATE_PASS status=partial"
echo "integration_head=$INTEGRATION_HEAD evidence_head=$EVIDENCE_HEAD base_head_equal=$BASE_HEAD_EQUAL base_ancestor=$BASE_ANCESTOR"
echo "place_head=$PLACE_HEAD place_ancestor=$PLACE_HEAD_ANCESTOR place_blob=$PLACE_BLOB place_sha256=$PLACE_SHA256 verified=true"
echo "arena_head=$ARENA_HEAD arena_ancestor=$ARENA_HEAD_ANCESTOR arena_blob=$ARENA_BLOB arena_sha256=$ARENA_SHA256 verified=true"
echo "writer_head=$WRITER_HEAD writer_ancestor=$WRITER_HEAD_ANCESTOR writer_blob=$WRITER_BLOB writer_sha256=$WRITER_SHA256 verified=true"
echo "ci_run=$CI_RUN_ID ci_artifact_id=$CI_ARTIFACT_ID ci_artifact_name=$CI_ARTIFACT_NAME compiler_sha256=$COMPILER_SHA256"
echo "composite_sha256=$COMPOSITE_SHA256 elf_sha256=$ELF_SHA256 path_fingerprint=$PATH_FINGERPRINT map_fingerprint=$MAP_FINGERPRINT"
echo "full_madaros_runtime=blocked full_driver_sha256=$FULL_DRIVER_SHA256 selftest_rc=$FULL_SELFTEST_RC selftest_log_sha256=$FULL_SELFTEST_SHA256 check_rc=$FULL_CHECK_RC check_log_sha256=$FULL_CHECK_SHA256 public_run_rc=$FULL_RUN_RC public_run_log_sha256=$FULL_RUN_SHA256 blocker=BLK-20260714-madaros-fixed-array-ident-copy-alias"
echo "receipt=$RECEIPT_JSON receipt_sha256=$RECEIPT_SHA256"
