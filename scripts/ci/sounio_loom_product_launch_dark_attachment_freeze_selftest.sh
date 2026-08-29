#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/product_launch_dark_attachment.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-product-launch-dark-attachment-v1-20260829.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-product-launch-freeze.XXXXXX")"
DUNE_BUILD="$TEST_ROOT/dune-build"
RUNTIME_ONE="$TEST_ROOT/loom-one"
RUNTIME_TWO="$TEST_ROOT/loom-two"
PROJECTION_ONE="$TEST_ROOT/projection-one"
PROJECTION_TWO="$TEST_ROOT/projection-two"
INSTALL_SOURCE="$TEST_ROOT/source"
INSTALL_RUNTIME="$TEST_ROOT/runtime"
INSTALL_STATE="$TEST_ROOT/installed-state"

cleanup() {
  if [[ -x "$INSTALL_RUNTIME/current/bin/sounio-loom-runtime" ]]; then
    "$INSTALL_RUNTIME/current/bin/sounio-loom-runtime" stop \
      --state-dir "$INSTALL_STATE" --agent freeze-installed --lane detached \
      --cwd "$TEST_ROOT/outside-repository" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-product-launch-dark-attachment-freeze-selftest: FAIL: %s\n' \
    "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "record field $key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

stream_hash() {
  local sum
  sum="$(sha256sum)"
  printf '%s' "${sum%% *}"
}

[[ -f "$MANIFEST" ]] || fail 'product launch manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'product launch evidence is missing'
[[ "$(field schema)" == loom-product-launch-dark-attachment-runtime-v1 ]] ||
  fail 'unknown manifest schema'
[[ "$(field stage)" == PRODUCT_LAUNCH_DARK_ATTACHMENT_FROZEN ]] ||
  fail 'wrong product stage'
[[ "$(field producing_language)" == OCaml ]] || fail 'producer is not OCaml'
[[ "$(field language_role)" == OPERATIONAL_ATTACHMENT ]] ||
  fail 'wrong language role'
[[ "$(field semantic_authority)" == Sounio ]] ||
  fail 'Sounio is not semantic authority'
[[ "$(field action)" == 9031 ]] || fail 'wrong semantic action'

for truth in pre_session_attached direct_start_attached provider_start_attached \
  provider_open_attached recover_attached unexpected_allow_refused \
  provider_allow_refused fail_closed receipt_hash_bound cwd_policy_ignored \
  current_projection_source_fresh policy_audit_roots_separated \
  installed_capsule_proven; do
  [[ "$(field "$truth")" == true ]] || fail "$truth was not frozen"
done
for boundary in authorizing production_activation live_material_frame \
  exec_attached commit_attached ci_attached parity_open claim_ready \
  same_uid_peer_isolation expected_results_encoded_in_ocaml \
  python_executable_invoked rust_executable_invoked shared_runtime_activated; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted"
done

implementation_commit="$(field implementation_commit)"
freeze_gate_commit="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${implementation_commit}^{commit}" ||
  fail 'implementation commit is absent'
git -C "$ROOT_DIR" cat-file -e "${freeze_gate_commit}^{commit}" ||
  fail 'freeze-gate commit is absent'

for pair in \
  garden_path:garden_sha256 \
  projection_path:projection_sha256 \
  membrane_source_path:membrane_source_sha256 \
  capsule_source_path:capsule_source_sha256 \
  resident_source_path:resident_source_sha256 \
  cli_source_path:cli_source_sha256 \
  dune_path:dune_sha256 \
  canonical_build_path:canonical_build_sha256 \
  projection_build_path:projection_build_sha256 \
  gate_script_path:gate_script_sha256 \
  capsule_regression_path:capsule_regression_sha256 \
  resident_regression_path:resident_regression_sha256 \
  installer_path:installer_sha256 \
  runtime_launcher_path:runtime_launcher_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$implementation_commit:$path" | stream_hash)" == \
    "$expected" ]] || fail "$path differs from the implementation commit"
done

freeze_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$freeze_path")" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze selftest drifted'
[[ "$(git -C "$ROOT_DIR" show "$freeze_gate_commit:$freeze_path" | stream_hash)" == \
  "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze selftest differs from its commit'

for source in \
  "$ROOT_DIR/$(field membrane_source_path)" \
  "$ROOT_DIR/$(field capsule_source_path)" \
  "$ROOT_DIR/$(field resident_source_path)" \
  "$ROOT_DIR/$(field cli_source_path)"; do
  if rg -n 'DENY50[2-9]|DENY510|ALLOW code=0 reason=allow' "$source" >/dev/null; then
    fail "$source encodes a Sounio semantic expected result"
  fi
done

action_manifest="$ROOT_DIR/$(field parent_9031_manifest_path)"
operational_manifest="$ROOT_DIR/$(field parent_operational_manifest_path)"
resident_manifest="$ROOT_DIR/$(field parent_resident_v5_manifest_path)"
[[ "$(file_hash "$action_manifest")" == "$(field parent_9031_manifest_sha256)" ]] ||
  fail 'action 9031 manifest drifted'
[[ "$(record_field "$action_manifest" stage)" == SEMANTICS_FROZEN && \
   "$(record_field "$action_manifest" producing_language)" == Sounio && \
   "$(record_field "$action_manifest" language_role)" == SEMANTIC_AUTHORITY && \
   "$(record_field "$action_manifest" fixture_bundle_sha256)" == \
     "$(field projection_sha256)" ]] ||
  fail 'action 9031 no longer owns the projected semantics'
[[ "$(file_hash "$operational_manifest")" == \
  "$(field parent_operational_manifest_sha256)" ]] ||
  fail 'affine OCaml realization manifest drifted'
[[ "$(record_field "$operational_manifest" semantic_authority)" == Sounio && \
   "$(record_field "$operational_manifest" producing_language)" == OCaml && \
   "$(record_field "$operational_manifest" production_activation)" == false ]] ||
  fail 'affine OCaml realization crossed its authority boundary'
[[ "$(file_hash "$resident_manifest")" == \
  "$(field parent_resident_v5_manifest_sha256)" ]] ||
  fail 'resident v5 manifest drifted'
[[ "$(record_field "$resident_manifest" process_model)" == single-resident-sounio-pid && \
   "$(record_field "$resident_manifest" route_9031)" == 6 && \
   "$(record_field "$resident_manifest" runtime_frozen)" == true ]] ||
  fail 'resident v5 lost its frozen action 9031 route'

ocamlc="$(field ocamlc_path)"
dune="$(field dune_executable_path)"
[[ "$(file_hash "$ocamlc")" == "$(field ocamlc_sha256)" && \
   "$($ocamlc -version)" == "$(field ocaml_version)" ]] ||
  fail 'OCaml toolchain drifted'
[[ "$(file_hash "$dune")" == "$(field dune_executable_sha256)" && \
   "$($dune --version)" == "$(field dune_version)" ]] ||
  fail 'Dune toolchain drifted'

bash "$ROOT_DIR/$(field canonical_build_path)" >/dev/null
resident_runtime="$ROOT_DIR/$(field resident_runtime_path)"
[[ -L "$resident_runtime" ]] || fail 'canonical resident v5 is not a symlink'
[[ "$(readlink "$resident_runtime")" == \
  "sha256-$(field resident_runtime_sha256)/sounio-loom-resident-membrane-runtime-v5" ]] ||
  fail 'canonical resident v5 symlink is not content addressed'
[[ "$(file_hash "$resident_runtime")" == "$(field resident_runtime_sha256)" && \
   ! -w "$(realpath "$resident_runtime")" ]] ||
  fail 'canonical resident v5 runtime is mutable or drifted'

SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT="$PROJECTION_ONE" \
  bash "$ROOT_DIR/$(field projection_build_path)" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT="$PROJECTION_TWO" \
  bash "$ROOT_DIR/$(field projection_build_path)" >/dev/null
cmp "$PROJECTION_ONE" "$PROJECTION_TWO" || fail 'two Sounio projections differ'
[[ "$(file_hash "$PROJECTION_ONE")" == "$(field projection_sha256)" ]] ||
  fail 'source-fresh Sounio projection differs from the freeze'

(
  flock -x 9
  "$dune" build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
  [[ "$(file_hash "$ROOT_DIR/tools/loom/_build/default/src/loom.exe")" == \
    "$(field runtime_sha256)" ]] || fail 'standard OCaml runtime hash differs'
  "$dune" build --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD" \
    src/loom.exe >/dev/null
  cp "$DUNE_BUILD/default/src/loom.exe" "$RUNTIME_ONE"
  "$dune" clean --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD"
  "$dune" build --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD" \
    src/loom.exe >/dev/null
  cp "$DUNE_BUILD/default/src/loom.exe" "$RUNTIME_TWO"
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two isolated OCaml rebuilds differ'

command="$(field command)"
[[ "$command" == \
  'bash scripts/ci/sounio_loom_product_launch_dark_attachment_selftest.sh' ]] ||
  fail 'unexpected adversarial gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] ||
  fail 'gate command hash differs'
result="$(bash "$ROOT_DIR/$(field gate_script_path)")"
[[ "$result" == "$(field result)" ]] || fail 'adversarial gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] ||
  fail 'adversarial gate result hash differs'

capsule_result="$(bash "$ROOT_DIR/$(field capsule_regression_path)")"
[[ "$capsule_result" == "$(field capsule_regression_result)" ]] ||
  fail 'affine capsule regression result differs'
resident_result="$(bash "$ROOT_DIR/$(field resident_regression_path)")"
[[ "$resident_result" == "$(field resident_regression_result)" ]] ||
  fail 'resident v5 regression result differs'

git clone --local --no-hardlinks --quiet "$ROOT_DIR" "$INSTALL_SOURCE"
git -C "$INSTALL_SOURCE" checkout --quiet --detach "$implementation_commit"
install_output="$(bash "$INSTALL_SOURCE/$(field installer_path)" \
  --source-root "$INSTALL_SOURCE" --runtime-dir "$INSTALL_RUNTIME")"
[[ "$install_output" == *"INSTALLED runtime_id=$(field installed_runtime_id)"* && \
   "$install_output" == *"ACTIVATED runtime_id=$(field installed_runtime_id)"* ]] ||
  fail "isolated installer did not activate the frozen runtime: $install_output"
installed_manifest="$INSTALL_RUNTIME/current/manifest"
[[ "$(record_field "$installed_manifest" runtime_id)" == \
   "$(field installed_runtime_id)" && \
   "$(record_field "$installed_manifest" bundle_sha256)" == \
   "$(field installed_bundle_sha256)" ]] ||
  fail 'installed runtime identity differs from the freeze'
grep -q '^capability=loom-product-launch-dark-attachment-v1$' "$installed_manifest" ||
  fail 'installed runtime omitted the product launch capability'
for pair in \
  loom_product_activation_action_manifest_sha256:parent_9031_manifest_sha256 \
  loom_product_activation_operational_manifest_sha256:parent_operational_manifest_sha256 \
  loom_product_activation_resident_manifest_sha256:parent_resident_v5_manifest_sha256 \
  loom_product_activation_projection_sha256:projection_sha256 \
  loom_product_activation_resident_runtime_sha256:resident_runtime_sha256; do
  installed_key="${pair%%:*}"
  frozen_key="${pair#*:}"
  [[ "$(record_field "$installed_manifest" "$installed_key")" == \
    "$(field "$frozen_key")" ]] || fail "installed $installed_key differs"
done

mkdir -p "$TEST_ROOT/outside-repository"
installed_loom="$INSTALL_RUNTIME/current/bin/sounio-loom-runtime"
installed_output="$($installed_loom start --state-dir "$INSTALL_STATE" \
  --agent freeze-installed --lane detached --session-id detached-session \
  --cwd "$TEST_ROOT/outside-repository" -- /usr/bin/tail -f /dev/null)"
[[ "$installed_output" == *'launch_source=start'* && \
   "$installed_output" == *"launch_dark_code=$(field current_material_code)"* && \
   "$installed_output" == *'authorizing=false'* && \
   "$installed_output" == *'production_activation=false'* ]] ||
  fail "installed Loom lost its Sounio observation: $installed_output"
installed_log="$INSTALL_STATE/product-launch-dark.tsv"
[[ -f "$installed_log" ]] || fail 'installed Loom omitted the durable receipt'
grep -Fq $'schema=loom-product-launch-dark-decision-v1\tlaunch_source=start\tdecision=DENY\tcode='"$(field current_material_code)"$'\t' \
  "$installed_log" || fail 'installed Loom receipt differs from Sounio authority'
"$installed_loom" stop --state-dir "$INSTALL_STATE" --agent freeze-installed \
  --lane detached --cwd "$TEST_ROOT/outside-repository" >/dev/null

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" ||
  fail 'evidence does not bind the product manifest'
grep -Fq "runtime_sha256=$(field runtime_sha256)" "$EVIDENCE" ||
  fail 'evidence does not bind the OCaml runtime'
grep -Fq "projection_sha256=$(field projection_sha256)" "$EVIDENCE" ||
  fail 'evidence does not bind the Sounio projection'
grep -Fq "installed_runtime_id=$(field installed_runtime_id)" "$EVIDENCE" ||
  fail 'evidence does not bind the installed runtime'

printf '%s\n' \
  "sounio-loom-product-launch-dark-attachment-freeze-selftest: PASS semantic_authority=Sounio operational_attachment=OCaml action=9031 real_paths=start+provider-start+provider-open+recover pre_session=true current_material=DENY$(field current_material_code)+CONTINUE causal_sabotage=ALLOW$(field seal_code)+NO_SESSION installed_runtime_id=$(field installed_runtime_id) installed_cwd=outside-repository manifest_sha256=$manifest_hash runtime_sha256=$(field runtime_sha256) projection_sha256=$(field projection_sha256) rebuilds=2 projection_rebuilds=2 receipts=hash-bound authorizing=false production_activation=false same_uid_peer_isolation=false shared_runtime_activated=false python_executed=false rust_executed=false"
