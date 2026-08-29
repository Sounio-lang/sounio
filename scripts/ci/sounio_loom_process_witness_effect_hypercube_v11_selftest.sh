#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-hypercube-v11.XXXXXX")"
BINARY_ONE="$TEST_ROOT/lab-one"
BINARY_TWO="$TEST_ROOT/lab-two"
SOUNIO_PLAN="$TEST_ROOT/sounio-plan"
SOUNIO_BUNDLE="$TEST_ROOT/sounio-bundle"
RECEIPTS="$TEST_ROOT/local-receipts"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11.freeze.v1"
V10_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v10.freeze.v1"
ROOT_TREE_SHA256=1111111111111111111111111111111111111111111111111111111111111111

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-hypercube-v11-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "receipt omitted field: $key"
}

for output in "$BINARY_ONE" "$BINARY_TWO"; do
  SOUNIO_LOOM_EFFECT_HYPERCUBE_V11_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_hypercube_v11.sh" \
      >/dev/null
done
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two source-fresh native V11 builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 && ! -u "$BINARY_ONE" &&
   ! -g "$BINARY_ONE" ]] || fail 'native V11 executable mode is unsafe'
if readelf -l "$BINARY_ONE" | grep -q 'INTERP'; then
  fail 'native V11 laboratory is dynamically linked'
fi

result="$($BINARY_ONE --selftest --policy-manifest "$POLICY_MANIFEST")"
[[ "$result" == LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11_SELFTEST\ PASS* ]] ||
  fail "native V11 laboratory selftest failed: $result"
[[ "$result" == *'semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true semantic_decision=false action=9025'* ]] ||
  fail 'native V11 language authority drifted'
[[ "$result" == *'families=12 probes=13 mechanism_dimensions=18 vertices=40 compiled_filters=12 '* &&
   "$result" == *'vertex_mode=true exec_transition_mode=true triple_hash_binding=true material_hypercube=false material_coverage=false'* ]] ||
  fail 'native V11 material apparatus shape drifted'

SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_OUTPUT="$SOUNIO_PLAN" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v11.sh" \
    >/dev/null
"$SOUNIO_PLAN" > "$SOUNIO_BUNDLE"
CELL_SHA256="$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"

declare -A invariant_by_probe
declare -A delta_by_vertex
material_vertices=0
: > "$RECEIPTS"

run_vertex() {
  local family="$1" probe="$2" bits="$3" expected_line expected witness
  local observed observation observed_witness invariant delta key
  expected_line="$(grep "^VERTEX family=${family} probe=${probe} bits=${bits} " \
    "$SOUNIO_BUNDLE" || true)"
  [[ -n "$expected_line" && "$(printf '%s\n' "$expected_line" | wc -l)" == 1 ]] ||
    fail "Sounio plan omitted or duplicated vertex ${family}/${probe}/${bits}"
  expected="$(field "$expected_line" expected)"
  witness="$(field "$expected_line" witness_kind)"
  observed="$($BINARY_ONE --vertex \
    --family "$family" --probe "$probe" --bits "$bits" \
    --policy-manifest "$POLICY_MANIFEST" \
    --cell-path "$BINARY_ONE" --cell-sha256 "$CELL_SHA256" \
    --root-tree-sha256 "$ROOT_TREE_SHA256" \
    --scratch-path "$TEST_ROOT/material-file" \
    --inet-address 127.0.0.1 --inet-port 9 \
    --unix-path "$TEST_ROOT/material.sock" \
    --principal-class LOCAL_SELFTEST)"
  [[ "$observed" == LOOM_EFFECT_VERTEX_V11\ OBSERVED* ]] ||
    fail "native vertex receipt diverged: ${family}/${probe}/${bits}"
  observation="$(field "$observed" observation)"
  observed_witness="$(field "$observed" witness_kind)"
  [[ "$observation" == "$expected" ]] ||
    fail "native observation disagreed with Sounio: ${family}/${probe}/${bits} expected=$expected observed=$observation"
  [[ "$(field "$observed" semantic_authority)" == Sounio &&
     "$(field "$observed" semantic_decision)" == false ]] ||
    fail "native vertex promoted itself to semantic authority: ${family}/${probe}/${bits}"
  invariant="$(field "$observed" invariant_sha256)"
  delta="$(field "$observed" delta_sha256)"
  [[ "$invariant" =~ ^[0-9a-f]{64}$ && "$delta" =~ ^[0-9a-f]{64}$ &&
     "$(field "$observed" witness_sha256)" =~ ^[0-9a-f]{64}$ ]] ||
    fail "native vertex omitted its causal hashes: ${family}/${probe}/${bits}"
  key="${family}/${probe}"
  if [[ -n "${invariant_by_probe[$key]:-}" &&
        "${invariant_by_probe[$key]}" != "$invariant" ]]; then
    fail "invariant drifted inside probe cube: $key"
  fi
  invariant_by_probe[$key]="$invariant"
  if [[ -n "${delta_by_vertex[$key/$bits]:-}" ]]; then
    fail "native vertex was executed twice: $key/$bits"
  fi
  delta_by_vertex[$key/$bits]="$delta"
  if [[ "$expected" == EFFECT_COMPLETED ]]; then
    [[ "$observed_witness" == "$witness" &&
       "$(field "$observed" witness_extinct)" == true ]] ||
      fail "completed effect lacks its Sounio witness or extinction: ${family}/${probe}/${bits}"
  else
    [[ "$observed_witness" == NONE ]] ||
      fail "refused effect fabricated a completion witness: ${family}/${probe}/${bits}"
  fi
  printf '%s\n' "$observed" >> "$RECEIPTS"
  material_vertices=$((material_vertices + 1))
}

for bits in 11 10 01 00; do
  run_vertex 1 repeat_exact_exec "$bits"
  run_vertex 1 first_wrong_flags_exec "$bits"
done
for family_probe in \
  '2 clone3_child' \
  '4 dup3_fd0_to_fd9' \
  '5 mmap_shared_write' \
  '6 io_uring_create' \
  '9 memfd_create' \
  '12 unlisted_getpid'; do
  read -r family probe <<< "$family_probe"
  run_vertex "$family" "$probe" 1
  run_vertex "$family" "$probe" 0
done
for family_probe in \
  '3 create_named_file' \
  '10 personality_change_restore' \
  '11 open_proc_self_mem_readonly'; do
  read -r family probe <<< "$family_probe"
  run_vertex "$family" "$probe" 01
  run_vertex "$family" "$probe" 00
done

[[ "$material_vertices" == 26 && "$(wc -l < "$RECEIPTS")" == 26 ]] ||
  fail 'local material vertex count diverged'
[[ "$(grep -c ' observation=REFUSED_BEFORE_EFFECT ' "$RECEIPTS" || true)" == 13 ]] ||
  fail 'local treatment-refusal count diverged'
[[ "$(grep -c ' observation=EFFECT_COMPLETED ' "$RECEIPTS" || true)" == 13 ]] ||
  fail 'local open-completion count diverged'
[[ "$(grep -c ' witness_extinct=true ' "$RECEIPTS" || true)" == 13 ]] ||
  fail 'local positive extinction count diverged'
if grep -Eq 'observation=(CROSSED_NAMED_RULE|EXPERIMENT_UNAVAILABLE)' "$RECEIPTS"; then
  fail 'local material subset crossed a rule or became unavailable'
fi

if "$BINARY_ONE" --selftest --policy-manifest "$V10_MANIFEST" >/dev/null 2>&1; then
  fail 'native V11 laboratory accepted the superseded V10 manifest'
fi
if "$BINARY_ONE" --vertex --family 4 --probe dup3_fd0_to_fd9 --bits 2 \
    --policy-manifest "$POLICY_MANIFEST" --cell-path "$BINARY_ONE" \
    --cell-sha256 "$CELL_SHA256" --root-tree-sha256 "$ROOT_TREE_SHA256" \
    --principal-class LOCAL_SELFTEST >/dev/null 2>&1; then
  fail 'native V11 laboratory accepted a noncanonical vertex bit'
fi
wrong_cell_sha=2222222222222222222222222222222222222222222222222222222222222222
if "$BINARY_ONE" --vertex --family 4 --probe dup3_fd0_to_fd9 --bits 0 \
    --policy-manifest "$POLICY_MANIFEST" --cell-path "$BINARY_ONE" \
    --cell-sha256 "$wrong_cell_sha" --root-tree-sha256 "$ROOT_TREE_SHA256" \
    --principal-class LOCAL_SELFTEST >/dev/null 2>&1; then
  fail 'native V11 laboratory accepted the wrong effect-cell identity'
fi

dependencies="$(ldd "$BINARY_ONE" 2>&1 || true)"
if ! printf '%s\n' "$dependencies" | grep -Eq 'not a dynamic executable|statically linked'; then
  fail 'native V11 executable did not prove static linkage'
fi
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'native V11 laboratory has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-hypercube-v11-selftest: PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true semantic_decision=false action=9025 policy_manifest_sha256=adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c families=12 probes=13 mechanism_dimensions=18 vertices=40 compiled_filters=12 local_material_families=10 local_material_probes=11 local_material_vertices=26 local_treatments=13 local_open_completions=13 local_extinctions=13 exec_vertices=8 Sounio_expected_results=true invariant_stable=true delta_distinct=true triple_hash_binding=true noncanonical_vertex=refused wrong_cell_identity=refused v10_manifest=refused deterministic=true runtime_dependencies=static source_sha256=%s executable_sha256=%s local_receipts_sha256=%s host_structural_families_pending=3+7+8+10+11 host_network_families_pending=7+8 material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_process_witness_effect_hypercube_v11.cpp" | cut -d ' ' -f 1)" \
  "$CELL_SHA256" \
  "$(sha256sum "$RECEIPTS" | cut -d ' ' -f 1)"
