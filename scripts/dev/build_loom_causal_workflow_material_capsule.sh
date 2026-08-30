#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"

fail() {
  printf 'build-loom-causal-workflow-material-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --output ABSOLUTE_PATH\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

manifest_value() {
  local manifest="$1" key="$2" line count
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field count is invalid: $key=$count"
  line="$(grep -m1 "^${key}=" "$manifest")"
  printf '%s' "${line#*=}"
}

OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$OUTPUT" == /* && ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || usage
parent="$(dirname "$OUTPUT")"
mkdir -p "$parent"
[[ -d "$parent" && ! -L "$parent" ]] || fail 'capsule output parent is absent or linked'

for tool in git sha256sum stat install mktemp find sort chmod mv cp c++ ocamlfind dune; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done

[[ -z "$(git -C "$ROOT_DIR" status --porcelain=v1 --untracked-files=all)" ]] ||
  fail 'source tree is dirty; a promotable capsule must identify one exact commit'
for input in \
  scripts/dev/build_loom_kernel_principal_broker.sh \
  scripts/dev/build_loom_causal_workflow_material_cell.sh \
  scripts/dev/build_loom_exec_grant_controller.sh \
  scripts/dev/build_sounio_loom_resident_membrane_v4.sh \
  scripts/dev/build_sounio_loom_product_exec_cell_fixture.sh \
  scripts/dev/build_sounio_loom_exec_operation_grant_fixture.sh \
  scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh \
  scripts/dev/build_sounio_loom_exec_result_record_fixture.sh \
  scripts/dev/build_sounio_loom_causal_workflow_run_grant_fixture.sh \
  scripts/dev/build_sounio_loom_causal_workflow_attest_grant_fixture.sh \
  scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh \
  scripts/dev/build_loom_causal_workflow_journal_fixture.sh; do
  [[ -x "$ROOT_DIR/$input" && ! -L "$ROOT_DIR/$input" ]] ||
    fail "required source-fresh builder is absent, linked, or non-executable: $input"
done

SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'source commit is not canonical'
CONTROLLER_COMMIT="$(manifest_value "$ROOT_DIR/tools/loom/exec_grant_controller.runtime.v1" controller_commit)"
RESIDENT_COMMIT="$(manifest_value "$ROOT_DIR/tools/loom/resident_membrane.runtime.v4" sounio_resident_v4_commit)"
[[ "$CONTROLLER_COMMIT" =~ ^[0-9a-f]{40}$ && "$RESIDENT_COMMIT" =~ ^[0-9a-f]{40}$ ]] ||
  fail 'action-9030 dependency commits are not canonical'

CONTROLLER_RUNTIME_MANIFEST="$ROOT_DIR/tools/loom/exec_grant_controller.runtime.v1"
RESIDENT_RUNTIME_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v4"
[[ "$(manifest_value "$CONTROLLER_RUNTIME_MANIFEST" semantic_authority)" == Sounio &&
   "$(manifest_value "$CONTROLLER_RUNTIME_MANIFEST" action)" == 9030 &&
   "$(manifest_value "$CONTROLLER_RUNTIME_MANIFEST" controller_commit)" == "$CONTROLLER_COMMIT" &&
   "$(manifest_value "$RESIDENT_RUNTIME_MANIFEST" producing_language)" == Sounio &&
   "$(manifest_value "$RESIDENT_RUNTIME_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(manifest_value "$RESIDENT_RUNTIME_MANIFEST" sounio_resident_v4_commit)" == "$RESIDENT_COMMIT" ]] ||
  fail 'action-9030 dependency authority posture drifted'
[[ "$(manifest_value "$CONTROLLER_RUNTIME_MANIFEST" resident_runtime_manifest_sha256)" == "$(sha256_file "$RESIDENT_RUNTIME_MANIFEST")" ]] ||
  fail 'controller no longer binds the selected resident runtime manifest'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-workflow-material-capsule.XXXXXX")"
STAGE="$(mktemp -d "$parent/.loom-causal-workflow-material-capsule.stage.XXXXXX")"
cleanup() {
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
  if [[ -n "${STAGE:-}" ]]; then
    chmod -R u+rwX "$STAGE" 2>/dev/null || true
    rm -rf "$STAGE"
  fi
}
trap cleanup EXIT

RELEASE="$STAGE/release"
BIN="$RELEASE/bin"
DATA="$RELEASE/data"
AUTHORITY_ROOT="$RELEASE/authority-root"
META="$STAGE/meta"
mkdir -p "$BIN" "$DATA" "$AUTHORITY_ROOT/.git" "$META"
chmod 0700 "$STAGE" "$RELEASE" "$BIN" "$DATA" "$AUTHORITY_ROOT" "$META"
chmod 0555 "$AUTHORITY_ROOT/.git"

install_root_file() {
  local relative="$1" source="$ROOT_DIR/$1" destination="$AUTHORITY_ROOT/$1" mode=0444
  [[ "$relative" =~ ^[A-Za-z0-9._/-]+$ && "$relative" != /* &&
     "/$relative/" != *'/../'* ]] || fail "authority-root path is unsafe: $relative"
  [[ -f "$source" && ! -L "$source" ]] || fail "frozen authority input is absent or linked: $relative"
  [[ -x "$source" ]] && mode=0555
  install -d -m 0755 "$(dirname "$destination")"
  install -m "$mode" "$source" "$destination"
}

install_manifest_closure() {
  local manifest="$1"
  shift
  local key relative expected
  for key in "$@"; do
    relative="$(manifest_value "$manifest" "${key}_path")"
    expected="$(manifest_value "$manifest" "${key}_sha256")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] ||
      fail "manifest dependency hash is not canonical: $key"
    install_root_file "$relative"
    [[ "$(sha256_file "$AUTHORITY_ROOT/$relative")" == "$expected" ]] ||
      fail "manifest dependency drifted: $key"
  done
}

# These files are the Sounio semantic authorities and frozen manifests whose
# digests are consumed by the host broker. They are copied before the release
# is made read-only, so the host never reads the source checkout.
AUTHORITY_FILES=(
  bin/souc
  bin/souc-lean-single-x86_64
  tests/verify-ir/call_b.sio
  tools/loom/kernel_exec_grant_cell_authority.freeze.v1
  tools/loom/exec_intent_envelope.freeze.v1
  tools/loom/exec_operation_grant_fixture.freeze.v1
  tools/loom/exec_operation_catalog.freeze.v1
  tools/loom/exec_result_record.freeze.v1
  tools/loom/causal_workflow_kernel.freeze.v1
  tools/loom/causal_workflow_run_grant_fixture.freeze.v1
  tools/loom/causal_workflow_attest_grant_fixture.freeze.v1
  tools/loom/causal_workflow_journal.runtime.v1
	  tools/loom/causal_workflow_material.runtime.v1
	  tools/loom/exec_grant_controller.runtime.v1
	  tools/loom/resident_membrane.runtime.v4
  stdlib/coordination/loom_kernel_exec_grant_cell_authority.sio
  tools/loom/kernel_exec_grant_cell_authority_main.sio
  stdlib/coordination/loom_exec_operation_catalog_authority.sio
  tools/loom/exec_operation_catalog_authority_main.sio
  stdlib/coordination/loom_exec_result_record_authority.sio
  tools/loom/exec_result_record_authority_main.sio
  stdlib/coordination/loom_causal_workflow_kernel_authority.sio
  tools/loom/causal_workflow_kernel_authority_main.sio
  tools/loom/exec_operation_grant_fixture_main.sio
  tools/loom/causal_workflow_run_grant_fixture_main.sio
  tools/loom/causal_workflow_attest_grant_fixture_main.sio
  tools/loom/product_exec_cell_fixture_main.sio
  tools/loom/src/loom_causal_workflow.ml
  tools/loom/causal_workflow_journal_fixture.ml
)
for relative in "${AUTHORITY_FILES[@]}"; do
  install_root_file "$relative"
done

# The product ExecCell loads the 9030 grant fixture, then projects and executes
# through the 9035 catalog before issuing the 9036 result record. Copy every
# direct file/hash edge those three OCaml validators consume. This is an
# executable dependency closure, not a second source of semantic truth.
EXEC_GRANT_MANIFEST="$ROOT_DIR/tools/loom/exec_operation_grant_fixture.freeze.v1"
install_manifest_closure "$EXEC_GRANT_MANIFEST" \
  garden source authority_manifest catalog_manifest result_manifest \
  build_script selftest freeze_selftest evidence toolchain_wrapper toolchain_compiler

EXEC_CATALOG_MANIFEST="$ROOT_DIR/tools/loom/exec_operation_catalog.freeze.v1"
install_manifest_closure "$EXEC_CATALOG_MANIFEST" \
  garden contract source entrypoint build_script selftest evidence \
  parent_9030_manifest parent_9031_manifest parent_9033_manifest \
  parent_9034_manifest toolchain_wrapper toolchain_compiler

EXEC_RESULT_MANIFEST="$ROOT_DIR/tools/loom/exec_result_record.freeze.v1"
install_manifest_closure "$EXEC_RESULT_MANIFEST" \
  garden contract source entrypoint build_script selftest evidence \
  parent_9035_manifest toolchain_wrapper toolchain_compiler

CAUSAL_MANIFEST="$ROOT_DIR/tools/loom/causal_workflow_kernel.freeze.v1"
CAUSAL_DEPENDENCY_KEYS=(
  garden
  contract
  concept_registry
  source
  entrypoint
  build_script
  selftest
  first_manifest
  first_evidence
  freeze_evidence
  parent_9030_manifest
  parent_9031_manifest
  parent_9032_manifest
  parent_9033_manifest
  parent_9034_manifest
  parent_9035_manifest
  parent_9036_manifest
  toolchain_wrapper
  toolchain_compiler
  canonical_source
)
for key in "${CAUSAL_DEPENDENCY_KEYS[@]}"; do
  relative="$(manifest_value "$CAUSAL_MANIFEST" "${key}_path")"
  install_root_file "$relative"
  [[ "$(sha256_file "$AUTHORITY_ROOT/$relative")" == "$(manifest_value "$CAUSAL_MANIFEST" "${key}_sha256")" ]] ||
    fail "causal authority dependency drifted: $key"
done

# Action 9030's frozen controller and resident manifests pin source hashes.
# Build them from the manifest-pinned trees; the other three material layers
# below deliberately use the exact current working tree.
FROZEN_CONTROLLER_ROOT="$WORK/frozen-controller"
FROZEN_RESIDENT_ROOT="$WORK/frozen-resident"
mkdir -p "$FROZEN_CONTROLLER_ROOT" "$FROZEN_RESIDENT_ROOT"
git -C "$ROOT_DIR" archive "$CONTROLLER_COMMIT" | tar -x -C "$FROZEN_CONTROLLER_ROOT"
git -C "$ROOT_DIR" archive "$RESIDENT_COMMIT" | tar -x -C "$FROZEN_RESIDENT_ROOT"

verify_frozen_hash() {
  local frozen_root="$1" manifest="$2" path_key="$3" hash_key="$4" path expected
  path="$(manifest_value "$manifest" "$path_key")"
  expected="$(manifest_value "$manifest" "$hash_key")"
  [[ "$path" =~ ^[A-Za-z0-9._/-]+$ && "$path" != /* && "/$path/" != *'/../'* &&
     "$expected" =~ ^[0-9a-f]{64}$ && -f "$frozen_root/$path" && ! -L "$frozen_root/$path" &&
     "$(sha256_file "$frozen_root/$path")" == "$expected" ]] ||
    fail "frozen dependency input drifted: $path_key"
}

verify_frozen_hash "$FROZEN_CONTROLLER_ROOT" "$CONTROLLER_RUNTIME_MANIFEST" controller_source_path controller_source_sha256
verify_frozen_hash "$FROZEN_CONTROLLER_ROOT" "$CONTROLLER_RUNTIME_MANIFEST" resident_source_path resident_source_sha256
verify_frozen_hash "$FROZEN_CONTROLLER_ROOT" "$CONTROLLER_RUNTIME_MANIFEST" cell_source_path cell_source_sha256
verify_frozen_hash "$FROZEN_CONTROLLER_ROOT" "$CONTROLLER_RUNTIME_MANIFEST" build_script_path build_script_sha256
verify_frozen_hash "$FROZEN_RESIDENT_ROOT" "$RESIDENT_RUNTIME_MANIFEST" dispatcher_path dispatcher_sha256
verify_frozen_hash "$FROZEN_RESIDENT_ROOT" "$RESIDENT_RUNTIME_MANIFEST" build_script_path build_script_sha256
verify_frozen_hash "$FROZEN_RESIDENT_ROOT" "$RESIDENT_RUNTIME_MANIFEST" toolchain_wrapper_path toolchain_wrapper_sha256
verify_frozen_hash "$FROZEN_RESIDENT_ROOT" "$RESIDENT_RUNTIME_MANIFEST" toolchain_compiler_path toolchain_compiler_sha256

# Build every executable from the current source surface, never from a stale
# worktree artifact. Fixture executables emit their frozen bundle into data.
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BIN/loom-kernel-principal-broker" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
SOUNIO_LOOM_CAUSAL_MATERIAL_CELL_OUTPUT="$BIN/loom-causal-workflow-material-cell" \
  bash "$ROOT_DIR/scripts/dev/build_loom_causal_workflow_material_cell.sh" >/dev/null
SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT="$BIN/loom-exec-grant-controller" \
  bash "$FROZEN_CONTROLLER_ROOT/scripts/dev/build_loom_exec_grant_controller.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$BIN/sounio-loom-resident-membrane-runtime-v4" \
  bash "$FROZEN_RESIDENT_ROOT/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
[[ "$(sha256_file "$BIN/loom-exec-grant-controller")" == "$(manifest_value "$CONTROLLER_RUNTIME_MANIFEST" runtime_sha256)" ]] ||
  fail 'controller runtime is not reproducible from its frozen dependency commit'
[[ "$(sha256_file "$BIN/sounio-loom-resident-membrane-runtime-v4")" == "$(manifest_value "$RESIDENT_RUNTIME_MANIFEST" runtime_sha256)" ]] ||
  fail 'resident runtime is not reproducible from its frozen dependency commit'
SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_OUTPUT="$BIN/sounio-loom-product-exec-cell-fixture" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_product_exec_cell_fixture.sh" >/dev/null
SOUNIO_LOOM_EXEC_OPERATION_CATALOG_OUTPUT="$BIN/sounio-loom-exec-operation-catalog" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null
SOUNIO_LOOM_EXEC_RESULT_RECORD_OUTPUT="$BIN/sounio-loom-exec-result-record" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null
[[ "$(sha256_file "$BIN/sounio-loom-exec-operation-catalog")" == \
   "$(manifest_value "$EXEC_CATALOG_MANIFEST" executable_sha256)" ]] ||
  fail 'source-fresh operation catalog runtime drifted from its freeze'
[[ "$(sha256_file "$BIN/sounio-loom-exec-result-record")" == \
   "$(manifest_value "$EXEC_RESULT_MANIFEST" executable_sha256)" ]] ||
  fail 'source-fresh result record runtime drifted from its freeze'
dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
install -m 0555 "$ROOT_DIR/tools/loom/_build/default/src/loom.exe" \
  "$BIN/sounio-loom-runtime"

operation_fixture="$WORK/operation-fixture"
SOUNIO_LOOM_EXEC_OPERATION_GRANT_FIXTURE_OUTPUT="$operation_fixture" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_grant_fixture.sh" >/dev/null
"$operation_fixture" > "$DATA/operation-grant-fixtures.v1"
run_fixture="$WORK/run-fixture"
SOUNIO_LOOM_CAUSAL_RUN_GRANT_OUTPUT="$run_fixture" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_run_grant_fixture.sh" >/dev/null
"$run_fixture" > "$DATA/causal-run-grant-fixtures.v1"
attest_fixture="$WORK/attest-fixture"
SOUNIO_LOOM_CAUSAL_ATTEST_GRANT_OUTPUT="$attest_fixture" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_attest_grant_fixture.sh" >/dev/null
"$attest_fixture" > "$DATA/causal-attest-grant-fixtures.v1"
SOUNIO_LOOM_CAUSAL_WORKFLOW_JOURNAL_OUTPUT="$BIN/loom-causal-workflow-journal-fixture" \
  bash "$ROOT_DIR/scripts/dev/build_loom_causal_workflow_journal_fixture.sh" >/dev/null
CAUSAL_WORKFLOW_RUNTIME="$AUTHORITY_ROOT/tools/loom/_build/default/src/sounio-loom-causal-workflow-kernel"
SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$CAUSAL_WORKFLOW_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null
[[ "$(sha256_file "$CAUSAL_WORKFLOW_RUNTIME")" == "$(manifest_value "$CAUSAL_MANIFEST" executable_sha256)" ]] ||
  fail 'source-fresh Sounio causal workflow runtime drifted from its freeze'

[[ "$(sha256_file "$DATA/operation-grant-fixtures.v1")" == \
  52674ef4332a4b6d54e2d55ca9a58c55de21e6686aca82ae70b9d22b7d260af2 ]] ||
  fail 'source-fresh operation fixture bundle drifted'
[[ "$(sha256_file "$DATA/causal-run-grant-fixtures.v1")" == \
  d92d79520d2f6b4d6f2b64f38d758686fe341f833a8a4ce8300871d9ef02706a ]] ||
  fail 'source-fresh causal run fixture bundle drifted'
[[ "$(sha256_file "$DATA/causal-attest-grant-fixtures.v1")" == \
  ab4c7d85f3ddd5c4f1129cb660b77f46855f257032ae75223fda630312df2fb0 ]] ||
  fail 'source-fresh causal attest fixture bundle drifted'

chmod 0555 "$BIN"/*
chmod 0444 "$DATA"/*
chmod -R go-w "$STAGE"
while IFS= read -r -d '' directory; do
  chmod 0555 "$directory"
done < <(find "$RELEASE" -type d -print0)

ENTRIES="$META/payload.entries.v1"
: > "$ENTRIES"
while IFS= read -r -d '' path; do
  relative="${path#"$RELEASE"/}"
  [[ "$relative" != "$path" && "$relative" =~ ^[A-Za-z0-9._/-]+$ ]] ||
    fail "payload path is not representable: $relative"
  mode="$(stat -c '%a' "$path")"
  if [[ -d "$path" && ! -L "$path" ]]; then
    printf 'D|%s|-|%s\n' "$mode" "$relative" >> "$ENTRIES"
  elif [[ -f "$path" && ! -L "$path" ]]; then
    printf 'F|%s|%s|%s\n' "$mode" "$(sha256_file "$path")" "$relative" >> "$ENTRIES"
  else
    fail "payload has a non-regular, non-directory entry: $relative"
  fi
done < <(find "$RELEASE" -mindepth 1 -print0 | sort -z)
chmod 0444 "$ENTRIES"

MANIFEST="$STAGE/capsule.manifest.v1"
cat > "$MANIFEST" <<EOF
schema=loom-causal-workflow-material-host-capsule-v1
stage=MATERIAL_PARITY_HOST_PROBE
source_commit=$SOURCE_COMMIT
controller_dependency_commit=$CONTROLLER_COMMIT
resident_dependency_commit=$RESIDENT_COMMIT
semantic_authority=Sounio
workflow_action=9037
launch_action=9030
controller_language=OCaml
material_language=C++20
capsule_layout=unpacked-directory-v1
release_path=release
authority_root_path=release/authority-root
broker_path=release/bin/loom-kernel-principal-broker
controller_runtime_path=release/bin/loom-exec-grant-controller
resident_runtime_path=release/bin/sounio-loom-resident-membrane-runtime-v4
product_runtime_path=release/bin/sounio-loom-runtime
product_runtime_sha256=$(sha256_file "$BIN/sounio-loom-runtime")
product_runtime_language=OCaml
product_runtime_role=EFFECT_PARITY
product_fixture_runtime_path=release/bin/sounio-loom-product-exec-cell-fixture
product_fixture_runtime_language=Sounio
product_fixture_runtime_role=SEMANTIC_FIXTURE_PRODUCER
operation_catalog_runtime_path=release/bin/sounio-loom-exec-operation-catalog
operation_catalog_runtime_sha256=$(sha256_file "$BIN/sounio-loom-exec-operation-catalog")
operation_catalog_runtime_language=Sounio
operation_catalog_runtime_role=SEMANTIC_AUTHORITY
operation_result_runtime_path=release/bin/sounio-loom-exec-result-record
operation_result_runtime_sha256=$(sha256_file "$BIN/sounio-loom-exec-result-record")
operation_result_runtime_language=Sounio
operation_result_runtime_role=SEMANTIC_AUTHORITY
material_cell_path=release/bin/loom-causal-workflow-material-cell
journal_runtime_path=release/bin/loom-causal-workflow-journal-fixture
causal_workflow_runtime_path=release/authority-root/tools/loom/_build/default/src/sounio-loom-causal-workflow-kernel
controller_runtime_manifest_path=release/authority-root/tools/loom/exec_grant_controller.runtime.v1
resident_runtime_manifest_path=release/authority-root/tools/loom/resident_membrane.runtime.v4
operation_fixture_manifest_path=release/authority-root/tools/loom/exec_operation_grant_fixture.freeze.v1
operation_fixture_bundle_path=release/data/operation-grant-fixtures.v1
operation_catalog_manifest_path=release/authority-root/tools/loom/exec_operation_catalog.freeze.v1
operation_result_manifest_path=release/authority-root/tools/loom/exec_result_record.freeze.v1
causal_run_grant_manifest_path=release/authority-root/tools/loom/causal_workflow_run_grant_fixture.freeze.v1
causal_run_grant_bundle_path=release/data/causal-run-grant-fixtures.v1
causal_attest_grant_manifest_path=release/authority-root/tools/loom/causal_workflow_attest_grant_fixture.freeze.v1
causal_attest_grant_bundle_path=release/data/causal-attest-grant-fixtures.v1
causal_workflow_manifest_path=release/authority-root/tools/loom/causal_workflow_kernel.freeze.v1
payload_entries_path=meta/payload.entries.v1
payload_entries_sha256=$(sha256_file "$ENTRIES")
payload_entry_count=$(wc -l < "$ENTRIES" | tr -d ' ')
parity_open=false
claim_ready=false
production_activation=false
EOF
chmod 0444 "$MANIFEST"
chmod 0555 "$META" "$STAGE"

mv "$STAGE" "$OUTPUT"
STAGE=''
printf 'LOOM_CAUSAL_WORKFLOW_MATERIAL_CAPSULE_BUILD PASS capsule=%s manifest_sha256=%s source_commit=%s layout=unpacked-directory-v1 semantic_authority=Sounio workflow_action=9037 launch_action=9030 production_activation=false parity_open=false claim_ready=false\n' \
  "$OUTPUT" "$(sha256_file "$OUTPUT/capsule.manifest.v1")" "$SOURCE_COMMIT"
