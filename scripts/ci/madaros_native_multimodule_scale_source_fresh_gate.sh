#!/usr/bin/env bash
# Bind the filed #901 scale witness to a direct, current-source Madaros ELF.
#
# The historical scale gate intentionally exercises the public `bin/souc`
# route. This gate is stricter: it builds Madaros from this clean tree, reaches
# a Madaros-to-Madaros fixed point, and invokes that raw ELF directly.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS_ROOT_PATH="$ROOT_DIR/bin/madaros-linux-x86_64"
SOUC_WRAPPER_PATH="$ROOT_DIR/bin/souc"
MADAROS_WRAPPER_PATH="$ROOT_DIR/bin/madaros"
PROBE="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob.sio"
TEXTBOOK="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio"
DRIVER="$ROOT_DIR/tests/stdlib/prob/test_prob_stdlib.sio"
FACADE="$ROOT_DIR/tests/run-pass/madaros_native_multimodule_scale_prob_facade.sio"
CONTEXT_SCOPE="$ROOT_DIR/tests/run-pass/let_scope_binding_name.sio"
CONTEXT_POLICY="$ROOT_DIR/tests/run-pass/let_policy_binding_name.sio"
CONTEXT_IS="$ROOT_DIR/tests/run-pass/let_is_binding_name.sio"
CONTEXT_STUDY="$ROOT_DIR/tests/run-pass/let_study_binding_name.sio"
KEEP_WORK="${SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_KEEP:-0}"
ARCHIVE_PROVENANCE_FILE="$ROOT_DIR/.issue901-scale-source-provenance.tsv"
M0_COMPAT_PATCH="$ROOT_DIR/scripts/ci/fixtures/madaros_m0_source_compat.patch"
ARCHIVE_MANIFEST_FILE="$ROOT_DIR/.issue901-scale-source-manifest.tsv"
PROOF_INPUT_PATHS=(
  bin/souc
  bin/madaros
  bin/madaros-linux-x86_64
  self-hosted
  stdlib
  tools/science_boundary
  scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh
  scripts/ci/fixtures/madaros_m0_source_compat.patch
  tests/run-pass/madaros_native_multimodule_scale_prob.sio
  tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio
  tests/stdlib/prob/test_prob_stdlib.sio
  tests/run-pass/madaros_native_multimodule_scale_prob_facade.sio
  tests/run-pass/let_scope_binding_name.sio
  tests/run-pass/let_policy_binding_name.sio
  tests/run-pass/let_is_binding_name.sio
  tests/run-pass/let_study_binding_name.sio
)

fail() {
  echo "[madaros-issue901-scale-source-fresh] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

write_proof_input_manifest() {
  local output="$1"
  local paths_file
  paths_file="$(mktemp /tmp/issue901-proof-input-paths.XXXXXX)"
  : >"$paths_file"
  for input in "${PROOF_INPUT_PATHS[@]}"; do
    [[ -e "$ROOT_DIR/$input" ]] || {
      rm -f "$paths_file"
      fail "proof input is missing: $input"
    }
    if [[ -d "$ROOT_DIR/$input" ]]; then
      find "$ROOT_DIR/$input" -type f -print
    else
      printf '%s\n' "$ROOT_DIR/$input"
    fi
  done | LC_ALL=C sort -u >"$paths_file"

  : >"$output"
  while IFS= read -r absolute; do
    local relative="${absolute#"$ROOT_DIR"/}"
    printf '%s\t%s\t%s\n' "$(portable_mode "$absolute")" "$relative" "$(portable_sha256 "$absolute")" >>"$output"
  done <"$paths_file"
  rm -f "$paths_file"
}

require_raw_madaros() {
  local path="$1"
  local banner

  [[ -x "$path" && -s "$path" ]] || fail "Madaros ELF is missing, empty, or not executable: $path"
  [[ "$(head -c4 "$path" 2>/dev/null)" == $'\x7fELF' ]] || fail "Madaros output is not an ELF: $path"
  banner="$("$path" --version 2>&1 || true)"
  [[ "$banner" == *Madaros* ]] || fail "raw compiler does not identify as Madaros: $path"
}

require_clean_source() {
  [[ -z "$(git -C "$ROOT_DIR" status --porcelain)" ]] || fail 'source tree must be clean; commit the source under test before source-fresh evidence'
}

tsv_value() {
  local file="$1"
  local key="$2"
  local value

  value="$(awk -F '\t' -v key="$key" '
    $1 == key {
      count += 1
      value = $2
    }
    END {
      if (count != 1 || value == "") {
        exit 1
      }
      print value
    }
  ' "$file")" || fail "archive provenance must contain exactly one nonempty $key entry: $file"
  printf '%s\n' "$value"
}

assert_archive_sha256() {
  local key="$1"
  local path="$2"
  local expected="$3"
  local actual

  actual="$(portable_sha256 "$path")"
  [[ "$actual" == "$expected" ]] || fail "archive provenance hash mismatch for $key: expected=$expected actual=$actual"
}

verify_archive_provenance() {
  local actual_manifest
  actual_manifest="$(mktemp /tmp/issue901-proof-input-manifest.XXXXXX)"
  write_proof_input_manifest "$actual_manifest"
  cmp -s "$ARCHIVE_MANIFEST_FILE" "$actual_manifest" || {
    diff -u "$ARCHIVE_MANIFEST_FILE" "$actual_manifest" >&2 || true
    rm -f "$actual_manifest"
    fail 'archive proof-input manifest does not match the bytes used by the gate'
  }
  rm -f "$actual_manifest"
  assert_archive_sha256 proof_input_manifest_sha256 "$ARCHIVE_MANIFEST_FILE" "$PROOF_INPUT_MANIFEST_SHA256"
  assert_archive_sha256 main_sio_sha256 "$ROOT_DIR/self-hosted/compiler/main.sio" "$MAIN_SHA256"
  assert_archive_sha256 gate_script_sha256 "$ROOT_DIR/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh" "$GATE_SHA256"
  assert_archive_sha256 madaros_root_sha256 "$MADAROS_ROOT_PATH" "$MADAROS_ROOT_SHA256"
  assert_archive_sha256 souc_wrapper_sha256 "$SOUC_WRAPPER_PATH" "$SOUC_WRAPPER_SHA256"
  assert_archive_sha256 madaros_wrapper_sha256 "$MADAROS_WRAPPER_PATH" "$MADAROS_WRAPPER_SHA256"
  assert_archive_sha256 acceptance_probe_sha256 "$PROBE" "$PROBE_SHA256"
  assert_archive_sha256 textbook_probe_sha256 "$TEXTBOOK" "$TEXTBOOK_SHA256"
  assert_archive_sha256 stdlib_driver_sha256 "$DRIVER" "$DRIVER_SHA256"
  assert_archive_sha256 m0_compat_patch_sha256 "$M0_COMPAT_PATCH" "$M0_COMPAT_PATCH_SHA256"
}

load_source_provenance() {
  require_raw_madaros "$MADAROS_ROOT_PATH"
  [[ -f "$M0_COMPAT_PATCH" ]] || fail "missing tracked M0 compatibility overlay: $M0_COMPAT_PATCH"

  if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    local checkout_manifest
    SOURCE_PROVENANCE_MODE='git-checkout'
    require_clean_source
    SOURCE_HEAD="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    SOURCE_TREE="$(git -C "$ROOT_DIR" rev-parse 'HEAD^{tree}')"
    MAIN_SHA256="$(portable_sha256 "$ROOT_DIR/self-hosted/compiler/main.sio")"
    GATE_SHA256="$(portable_sha256 "$ROOT_DIR/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh")"
    MADAROS_ROOT_SHA256="$(portable_sha256 "$MADAROS_ROOT_PATH")"
    SOUC_WRAPPER_SHA256="$(portable_sha256 "$SOUC_WRAPPER_PATH")"
    MADAROS_WRAPPER_SHA256="$(portable_sha256 "$MADAROS_WRAPPER_PATH")"
    PROBE_SHA256="$(portable_sha256 "$PROBE")"
    TEXTBOOK_SHA256="$(portable_sha256 "$TEXTBOOK")"
    DRIVER_SHA256="$(portable_sha256 "$DRIVER")"
    M0_COMPAT_PATCH_SHA256="$(portable_sha256 "$M0_COMPAT_PATCH")"
    checkout_manifest="$(mktemp /tmp/issue901-proof-input-manifest.XXXXXX)"
    write_proof_input_manifest "$checkout_manifest"
    PROOF_INPUT_MANIFEST_SHA256="$(portable_sha256 "$checkout_manifest")"
    rm -f "$checkout_manifest"
    MADAROS_ROOT_BLOB="$(git -C "$ROOT_DIR" ls-files -s -- bin/madaros-linux-x86_64 | awk 'NR == 1 {print $2}')"
    [[ -n "$MADAROS_ROOT_BLOB" ]] || fail 'Madaros root is not tracked by this source tree'
    return
  fi

  SOURCE_PROVENANCE_MODE='git-archive-exact-commit'
  [[ -f "$ARCHIVE_PROVENANCE_FILE" ]] || fail "source has no Git metadata and no archive provenance: $ARCHIVE_PROVENANCE_FILE"
  [[ -f "$ARCHIVE_MANIFEST_FILE" ]] || fail "source archive has no proof-input manifest: $ARCHIVE_MANIFEST_FILE"
  [[ "$(tsv_value "$ARCHIVE_PROVENANCE_FILE" provenance_version)" == 'issue901-scale-source-fresh-archive-v3' ]] || fail 'archive provenance has an unsupported version'
  [[ "$(tsv_value "$ARCHIVE_PROVENANCE_FILE" source_origin)" == 'git-archive-exact-commit' ]] || fail 'archive provenance has an unsupported source origin'
  SOURCE_HEAD="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" source_head)"
  SOURCE_TREE="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" source_tree)"
  MAIN_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" main_sio_sha256)"
  GATE_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" gate_script_sha256)"
  MADAROS_ROOT_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" madaros_root_sha256)"
  SOUC_WRAPPER_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" souc_wrapper_sha256)"
  MADAROS_WRAPPER_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" madaros_wrapper_sha256)"
  PROBE_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" acceptance_probe_sha256)"
  TEXTBOOK_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" textbook_probe_sha256)"
  DRIVER_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" stdlib_driver_sha256)"
  M0_COMPAT_PATCH_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" m0_compat_patch_sha256)"
  PROOF_INPUT_MANIFEST_SHA256="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" proof_input_manifest_sha256)"
  MADAROS_ROOT_BLOB="$(tsv_value "$ARCHIVE_PROVENANCE_FILE" madaros_root_git_blob)"
  [[ "$SOURCE_HEAD" =~ ^[0-9a-f]{40}$ ]] || fail "archive provenance has an invalid source_head: $SOURCE_HEAD"
  [[ "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ ]] || fail "archive provenance has an invalid source_tree: $SOURCE_TREE"
  [[ "$MADAROS_ROOT_BLOB" =~ ^[0-9a-f]{40}$ ]] || fail "archive provenance has an invalid madaros_root_git_blob"
  verify_archive_provenance
}

assert_source_provenance_unchanged() {
  if [[ "$SOURCE_PROVENANCE_MODE" == 'git-checkout' ]]; then
    require_clean_source
    [[ "$(git -C "$ROOT_DIR" rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during source-fresh evidence'
    [[ "$(git -C "$ROOT_DIR" rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during source-fresh evidence'
  else
    verify_archive_provenance
  fi
}

build_from_madaros_seed() {
  local label="$1"
  local seed="$2"
  local output="$3"
  local log="$4"
  local source="$5"

  if ! (
    printf 'madaros_seed\t%s\n' "$seed"
    printf 'madaros_seed_mode\tnative-v2-compile\n'
    printf 'madaros_source\t%s\n' "$source"
    ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true
    exec env -i \
      PATH="$PATH" \
      LC_ALL=C \
      HOME="${HOME:-/tmp}" \
      TMPDIR="${TMPDIR:-/tmp}" \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$seed" --native-v2-compile "$source" "$output"
  ) >"$log" 2>&1; then
    cat "$log" >&2 || true
    fail "$label Madaros seed could not rebuild current source"
  fi
  grep -Fxq "madaros_seed"$'\t'"$seed" "$log" || {
    cat "$log" >&2 || true
    fail "$label did not record the expected direct Madaros seed: $seed"
  }
  grep -Fxq "madaros_source"$'\t'"$source" "$log" || {
    cat "$log" >&2 || true
    fail "$label did not record the expected source: $source"
  }
  assert_direct_raw_log "$log"
  for marker in \
    'imported_compile: begin' \
    'imported_compile: load_done' \
    'imported_compile: typecheck ok' \
    'Merged IR:' \
    "native_v2_compile: emitted path=$output"; do
    grep -Fq "$marker" "$log" || {
      cat "$log" >&2 || true
      fail "$label missed compiler-emitted marker: $marker"
    }
  done
}

assert_complete_closure() {
  local label="$1"
  local seed="$2"
  local source="$3"
  local allowed_source_root="$4"
  local log="$5"

  if ! env -i \
    PATH="$PATH" \
    LC_ALL=C \
    HOME="${HOME:-/tmp}" \
    TMPDIR="${TMPDIR:-/tmp}" \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    "$seed" --science-boundary-closure "$source" >"$log" 2>&1; then
    cat "$log" >&2 || true
    fail "$label could not enumerate its exact source closure"
  fi
  assert_direct_raw_log "$log"
  grep -Fxq 'status'$'\t''complete' "$log" || fail "$label source closure is not complete"
  grep -Fxq 'saturated'$'\t''false' "$log" || fail "$label source closure saturated its capacity"
  grep -Fxq 'parse_failed'$'\t''false' "$log" || fail "$label source closure contains a parse failure"
  while IFS=$'\t' read -r kind path; do
    [[ "$kind" == node ]] || continue
    case "$path" in
      "$allowed_source_root"/*|"$ROOT_DIR/stdlib"/*) ;;
      *) fail "$label source closure escaped allowed roots: $path" ;;
    esac
  done <"$log"
}

prepare_m0_compat_source() {
  local overlay_root="$1"
  local manifest="$2"
  local paths_file="$3"

  [[ -f "$M0_COMPAT_PATCH" ]] || fail "missing tracked M0 compatibility overlay: $M0_COMPAT_PATCH"
  grep -Fq '+++ /dev/null' "$M0_COMPAT_PATCH" && fail 'M0 compatibility overlay may not delete canonical source files'
  mkdir -p "$overlay_root"
  cp -a "$ROOT_DIR/self-hosted" "$overlay_root/self-hosted"
  awk '/^\+\+\+ b\// { print substr($0, 7) }' "$M0_COMPAT_PATCH" | LC_ALL=C sort -u >"$paths_file"
  [[ -s "$paths_file" ]] || fail 'M0 compatibility overlay declares no transformed paths'
  (
    cd "$overlay_root"
    git apply --check --unidiff-zero "$M0_COMPAT_PATCH"
    git apply --unidiff-zero "$M0_COMPAT_PATCH"
  ) || fail 'tracked M0 compatibility overlay did not apply exactly to canonical source'

  : >"$manifest"
  while IFS= read -r path; do
    local status='modify'
    local before_mode='absent'
    local before_sha256='absent'
    [[ "$path" == self-hosted/* && "$path" != *'..'* ]] || fail "M0 compatibility overlay escapes self-hosted source: $path"
    [[ "$path" != self-hosted/parser/* ]] || fail "M0 compatibility overlay must not alter canonical parser semantics: $path"
    [[ -f "$overlay_root/$path" ]] || fail "M0 compatibility overlay output is missing: $path"
    if [[ -f "$ROOT_DIR/$path" ]]; then
      before_mode="$(portable_mode "$ROOT_DIR/$path")"
      before_sha256="$(portable_sha256 "$ROOT_DIR/$path")"
    else
      status='add'
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$status" "$path" "$before_mode" "$before_sha256" \
      "$(portable_mode "$overlay_root/$path")" "$(portable_sha256 "$overlay_root/$path")" >>"$manifest"
  done <"$paths_file"
}

preflight_m0_compat_source() {
  local preflight
  preflight="$(mktemp -d /tmp/issue901-m0-compat-preflight.XXXXXX)"
  prepare_m0_compat_source "$preflight/source" "$preflight/delta-manifest.tsv" "$preflight/paths.txt"
  rm -rf "$preflight"
}

assert_direct_raw_log() {
  local log="$1"

  if grep -Eiq \
    'lean_single|falling back to the legacy|native_prebundle:|SELFHOST=fallback|compatibility fallback|multimodule native thin-link compilation failed|imported_simple_ir_emit_failed' \
    "$log"; then
    cat "$log" >&2
    fail "direct raw #901 compile observed a forbidden fallback or historical scale failure: $log"
  fi
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_native_multimodule_scale_source_fresh_gate.sh [--structural-only]'
  require_raw_madaros "$MADAROS_ROOT_PATH"
  [[ -x "$SOUC_WRAPPER_PATH" ]] || fail 'missing executable public souc wrapper'
  [[ -x "$MADAROS_WRAPPER_PATH" ]] || fail 'missing executable public Madaros wrapper'
  [[ -f "$PROBE" ]] || fail 'missing #901 acceptance probe'
  [[ -f "$TEXTBOOK" ]] || fail 'missing #901 textbook probe'
  [[ -f "$DRIVER" ]] || fail 'missing #901 stdlib driver'
  [[ -f "$FACADE" ]] || fail 'missing #901 imported facade witness'
  [[ -f "$CONTEXT_SCOPE" && -f "$CONTEXT_POLICY" && -f "$CONTEXT_IS" && -f "$CONTEXT_STUDY" ]] || fail 'missing contextual-keyword bootstrap witnesses'
  [[ -f "$M0_COMPAT_PATCH" ]] || fail 'missing tracked M0 compatibility overlay'
  git apply --numstat "$M0_COMPAT_PATCH" >/dev/null || fail 'tracked M0 compatibility overlay is not a valid patch'
  preflight_m0_compat_source
  echo '[madaros-issue901-scale-source-fresh] PASS: source-build, M0 overlay, and direct-raw scale wiring is present'
  exit 0
fi
if [[ "${1:-}" == '--provenance-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_native_multimodule_scale_source_fresh_gate.sh [--structural-only|--provenance-only]'
  cd "$ROOT_DIR"
  load_source_provenance
  assert_source_provenance_unchanged
  preflight_m0_compat_source
  printf '[madaros-issue901-scale-source-fresh] PASS: source provenance mode=%s head=%s tree=%s\n' "$SOURCE_PROVENANCE_MODE" "$SOURCE_HEAD" "$SOURCE_TREE"
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_native_multimodule_scale_source_fresh_gate.sh [--structural-only|--provenance-only]'

cd "$ROOT_DIR"
load_source_provenance

if [[ -n "${SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-issue901-scale-source-fresh.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

STAGE1="$WORK/madaros-stage1"
STAGE2="$WORK/madaros-stage2"
STAGE3="$WORK/madaros-stage3"
RECEIPT="$WORK/madaros_native_multimodule_scale_901_source_fresh_receipt.tsv"
M0_COMPAT_ROOT="$WORK/m0-compat-source"
M0_COMPAT_DELTA_MANIFEST="$WORK/m0-compat-delta-manifest.tsv"
M0_COMPAT_PATHS_FILE="$WORK/m0-compat-paths.txt"

prepare_m0_compat_source "$M0_COMPAT_ROOT" "$M0_COMPAT_DELTA_MANIFEST" "$M0_COMPAT_PATHS_FILE"
M0_COMPAT_DELTA_MANIFEST_SHA256="$(portable_sha256 "$M0_COMPAT_DELTA_MANIFEST")"
M0_COMPAT_TRANSFORMED_PATHS="$(paste -sd, "$M0_COMPAT_PATHS_FILE")"
M0_COMPAT_MAIN_SHA256="$(portable_sha256 "$M0_COMPAT_ROOT/self-hosted/compiler/main.sio")"

# The tracked Madaros ELF is the constrained root of trust. Every generated
# stage uses Madaros's explicit native-v2 ABI; no lean positional bridge is in
# the operational self-hosting chain. Only M0 -> M1 sees the declared overlay;
# M1 and all later stages compile the untouched canonical source.
assert_complete_closure stage1 "$MADAROS_ROOT_PATH" "$M0_COMPAT_ROOT/self-hosted/compiler/main.sio" "$M0_COMPAT_ROOT/self-hosted" "$WORK/stage1-closure.log"
build_from_madaros_seed stage1 "$MADAROS_ROOT_PATH" "$STAGE1" "$WORK/stage1-build.log" "$M0_COMPAT_ROOT/self-hosted/compiler/main.sio"
require_raw_madaros "$STAGE1"

assert_complete_closure stage2 "$STAGE1" "$ROOT_DIR/self-hosted/compiler/main.sio" "$ROOT_DIR/self-hosted" "$WORK/stage2-closure.log"
build_from_madaros_seed stage2 "$STAGE1" "$STAGE2" "$WORK/stage2-build.log" "$ROOT_DIR/self-hosted/compiler/main.sio"
require_raw_madaros "$STAGE2"

assert_complete_closure stage3 "$STAGE2" "$ROOT_DIR/self-hosted/compiler/main.sio" "$ROOT_DIR/self-hosted" "$WORK/stage3-closure.log"
build_from_madaros_seed stage3 "$STAGE2" "$STAGE3" "$WORK/stage3-build.log" "$ROOT_DIR/self-hosted/compiler/main.sio"
require_raw_madaros "$STAGE3"

assert_source_provenance_unchanged

STAGE1_SHA256="$(portable_sha256 "$STAGE1")"
STAGE2_SHA256="$(portable_sha256 "$STAGE2")"
STAGE3_SHA256="$(portable_sha256 "$STAGE3")"
[[ "$STAGE2_SHA256" == "$STAGE3_SHA256" ]] || fail "Madaros fixed point diverged: stage2=$STAGE2_SHA256 stage3=$STAGE3_SHA256"

for source in "$PROBE" "$TEXTBOOK" "$DRIVER"; do
  [[ -f "$source" ]] || fail "required #901 source is missing: $source"
done

CASE_MERGED=""
run_direct_case() {
  local label="$1"
  local source="$2"
  local expected="$3"
  local case_dir="$WORK/$label"
  local elf="$case_dir/$label.elf"
  local compile_log="$case_dir/compile.log"
  local run_log="$case_dir/run.log"

  mkdir -p "$case_dir"
  if ! (
    cd "$case_dir"
    exec env -i \
      PATH="$PATH" \
      LC_ALL=C \
      HOME="${HOME:-/tmp}" \
      TMPDIR="${TMPDIR:-/tmp}" \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$STAGE3" --native-v2-compile "$source" "$elf"
  ) >"$compile_log" 2>&1; then
    cat "$compile_log" >&2 || true
    fail "direct raw compile failed for $label"
  fi
  assert_direct_raw_log "$compile_log"
  [[ -s "$elf" ]] || fail "direct raw compile emitted no ELF for $label"
  [[ "$(head -c4 "$elf" 2>/dev/null)" == $'\x7fELF' ]] || fail "direct raw output is not an ELF for $label"
  chmod +x "$elf"
  if ! "$elf" >"$run_log" 2>&1; then
    cat "$run_log" >&2 || true
    fail "direct raw ELF exited nonzero for $label"
  fi
  grep -Eq "$expected" "$run_log" || {
    cat "$run_log" >&2 || true
    fail "direct raw ELF stdout missed expected witness for $label: /$expected/"
  }
  CASE_MERGED="$(awk '/Merged IR:/{n=$NF} END{print n}' "$compile_log" 2>/dev/null || true)"
  printf '[madaros-issue901-scale-source-fresh] PASS: %s merged_ir=%s\n' "$label" "${CASE_MERGED:-unknown}"
}

run_direct_case acceptance-probe "$PROBE" 'm=5(\.0+)?'
PROBE_MERGED="$CASE_MERGED"
run_direct_case textbook "$TEXTBOOK" 'PROB_TEXTBOOK_OK'
TEXTBOOK_MERGED="$CASE_MERGED"
run_direct_case stdlib-driver "$DRIVER" 'PROB_STDLIB_OK'
DRIVER_MERGED="$CASE_MERGED"
assert_complete_closure facade-elf-42 "$STAGE3" "$FACADE" "$ROOT_DIR/tests/run-pass" "$WORK/facade-elf-42-closure.log"
for required_facade_node in \
  "$ROOT_DIR/stdlib/prob/lib.sio" \
  "$ROOT_DIR/stdlib/prob/distributions.sio" \
  "$ROOT_DIR/stdlib/special/gamma.sio" \
  "$ROOT_DIR/stdlib/special/igamma.sio" \
  "$ROOT_DIR/stdlib/special/erf.sio"; do
  grep -Fxq 'node'$'\t'"$required_facade_node" "$WORK/facade-elf-42-closure.log" || fail "facade closure missed required physical module: $required_facade_node"
done
run_direct_case facade-elf-42 "$FACADE" '^42$'
FACADE_MERGED="$CASE_MERGED"
run_direct_case contextual-scope "$CONTEXT_SCOPE" '^LET_SCOPE_BINDING_OK$'
run_direct_case contextual-policy "$CONTEXT_POLICY" '^LET_POLICY_BINDING_OK$'
run_direct_case contextual-is "$CONTEXT_IS" '^LET_IS_BINDING_OK$'
run_direct_case contextual-study "$CONTEXT_STUDY" '^LET_STUDY_BINDING_OK$'

run_public_wrapper_case() {
  local source="$1"
  local expected="$2"
  local case_dir="$WORK/public-wrapper"
  local elf="$case_dir/public-wrapper.elf"
  local compile_log="$case_dir/compile.log"
  local run_log="$case_dir/run.log"

  mkdir -p "$case_dir"
  if ! (
    cd "$case_dir"
    exec env -i \
      PATH="$PATH" \
      LC_ALL=C \
      HOME="${HOME:-/tmp}" \
      TMPDIR="${TMPDIR:-/tmp}" \
      MADAROS_RAW_BIN="$STAGE3" \
      SOUNIO_SOUC_ENGINE=madaros \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$SOUC_WRAPPER_PATH" compile "$source" -o "$elf"
  ) >"$compile_log" 2>&1; then
    cat "$compile_log" >&2 || true
    fail 'public-wrapper Madaros compile failed for the #901 acceptance probe'
  fi
  assert_direct_raw_log "$compile_log"
  [[ -s "$elf" ]] || fail 'public-wrapper Madaros compile emitted no ELF for the #901 acceptance probe'
  [[ "$(head -c4 "$elf" 2>/dev/null)" == $'\x7fELF' ]] || fail 'public-wrapper Madaros output is not an ELF for the #901 acceptance probe'
  chmod +x "$elf"
  if ! "$elf" >"$run_log" 2>&1; then
    cat "$run_log" >&2 || true
    fail 'public-wrapper Madaros ELF exited nonzero for the #901 acceptance probe'
  fi
  grep -Eq "$expected" "$run_log" || {
    cat "$run_log" >&2 || true
    fail "public-wrapper Madaros stdout missed the #901 witness: /$expected/"
  }
  CASE_MERGED="$(awk '/Merged IR:/{n=$NF} END{print n}' "$compile_log" 2>/dev/null || true)"
  printf '[madaros-issue901-scale-source-fresh] PASS: public-wrapper merged_ir=%s\n' "${CASE_MERGED:-unknown}"
}

run_public_wrapper_case "$FACADE" '^42$'
PUBLIC_WRAPPER_MERGED="$CASE_MERGED"

assert_source_provenance_unchanged

printf 'receipt_version\tissue901-scale-source-fresh-v3\n' >"$RECEIPT"
printf 'source_provenance_mode\t%s\n' "$SOURCE_PROVENANCE_MODE" >>"$RECEIPT"
printf 'source_head\t%s\n' "$SOURCE_HEAD" >>"$RECEIPT"
printf 'source_tree\t%s\n' "$SOURCE_TREE" >>"$RECEIPT"
printf 'proof_input_manifest_sha256\t%s\n' "$PROOF_INPUT_MANIFEST_SHA256" >>"$RECEIPT"
printf 'main_sio_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'gate_script_sha256\t%s\n' "$GATE_SHA256" >>"$RECEIPT"
printf 'madaros_root_repo_path\tbin/madaros-linux-x86_64\n' >>"$RECEIPT"
printf 'madaros_root_git_blob\t%s\n' "$MADAROS_ROOT_BLOB" >>"$RECEIPT"
printf 'madaros_root_sha256\t%s\n' "$MADAROS_ROOT_SHA256" >>"$RECEIPT"
printf 'souc_wrapper_sha256\t%s\n' "$SOUC_WRAPPER_SHA256" >>"$RECEIPT"
printf 'madaros_wrapper_sha256\t%s\n' "$MADAROS_WRAPPER_SHA256" >>"$RECEIPT"
printf 'acceptance_probe_sha256\t%s\n' "$PROBE_SHA256" >>"$RECEIPT"
printf 'textbook_probe_sha256\t%s\n' "$TEXTBOOK_SHA256" >>"$RECEIPT"
printf 'stdlib_driver_sha256\t%s\n' "$DRIVER_SHA256" >>"$RECEIPT"
printf 'm0_compat_patch_sha256\t%s\n' "$M0_COMPAT_PATCH_SHA256" >>"$RECEIPT"
printf 'm0_compat_delta_manifest_sha256\t%s\n' "$M0_COMPAT_DELTA_MANIFEST_SHA256" >>"$RECEIPT"
printf 'm0_compat_transformed_paths\t%s\n' "$M0_COMPAT_TRANSFORMED_PATHS" >>"$RECEIPT"
printf 'm0_compat_main_sio_sha256\t%s\n' "$M0_COMPAT_MAIN_SHA256" >>"$RECEIPT"
printf 'proof_bootstrap_mode\ttracked-madaros-root-one-use-overlay-then-canonical-madaros-fixed-point\n' >>"$RECEIPT"
printf 'stage1_madaros_sha256\t%s\n' "$STAGE1_SHA256" >>"$RECEIPT"
printf 'stage2_madaros_sha256\t%s\n' "$STAGE2_SHA256" >>"$RECEIPT"
printf 'stage3_madaros_sha256\t%s\n' "$STAGE3_SHA256" >>"$RECEIPT"
printf 'stage1_seed\ttracked-madaros-root-direct\n' >>"$RECEIPT"
printf 'stage2_seed\tstage1-madaros-direct\n' >>"$RECEIPT"
printf 'stage3_seed\tstage2-madaros-direct\n' >>"$RECEIPT"
printf 'stage1_source_role\tm0-compat-overlay\n' >>"$RECEIPT"
printf 'stage2_source_role\tcanonical-source\n' >>"$RECEIPT"
printf 'stage3_source_role\tcanonical-source\n' >>"$RECEIPT"
printf 'stage1_source_main_sha256\t%s\n' "$M0_COMPAT_MAIN_SHA256" >>"$RECEIPT"
printf 'stage2_source_main_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'stage3_source_main_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'operational_fixed_point\tsha256-stage2-equals-stage3\n' >>"$RECEIPT"
printf 'direct_raw_acceptance_mode\tdirect-raw-elf-no-wrapper\n' >>"$RECEIPT"
printf 'engine_fallback\t0\n' >>"$RECEIPT"
printf 'compact_imported_ir\t0\n' >>"$RECEIPT"
printf 'default_merge_mode\tinto-acc\n' >>"$RECEIPT"
printf 'target_resolution\tauto-x86_64-linux\n' >>"$RECEIPT"
printf 'acceptance_probe\tm=5.000000\n' >>"$RECEIPT"
printf 'acceptance_probe_merged_ir\t%s\n' "${PROBE_MERGED:-unknown}" >>"$RECEIPT"
printf 'textbook_probe\tPROB_TEXTBOOK_OK\n' >>"$RECEIPT"
printf 'textbook_probe_merged_ir\t%s\n' "${TEXTBOOK_MERGED:-unknown}" >>"$RECEIPT"
printf 'stdlib_driver\tPROB_STDLIB_OK\n' >>"$RECEIPT"
printf 'stdlib_driver_merged_ir\t%s\n' "${DRIVER_MERGED:-unknown}" >>"$RECEIPT"
printf 'facade_vertical\tprob::lib::{uniform_mean}->ELF->42\n' >>"$RECEIPT"
printf 'facade_vertical_merged_ir\t%s\n' "${FACADE_MERGED:-unknown}" >>"$RECEIPT"
printf 'contextual_keyword_witnesses\tscope,policy,is,study\n' >>"$RECEIPT"
printf 'public_wrapper_route\tbin/souc-compile-pinned-to-stage3\n' >>"$RECEIPT"
printf 'public_wrapper_probe\tprob-facade-ELF-42\n' >>"$RECEIPT"
printf 'public_wrapper_probe_merged_ir\t%s\n' "${PUBLIC_WRAPPER_MERGED:-unknown}" >>"$RECEIPT"

cat "$RECEIPT"
echo "[madaros-issue901-scale-source-fresh] PASS: source_head=$SOURCE_HEAD stage3_sha256=$STAGE3_SHA256 receipt=$RECEIPT"
