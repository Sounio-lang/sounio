#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPILER_CLI="${SOUC_BIN:-$ROOT/bin/madaros}"
RAW_COMPILER="${SOIR_V5_EMPTY_READER_RAW_BIN:-${MADAROS_RAW_BIN:-$ROOT/bin/madaros-linux-x86_64}}"
EXPECTED_RAW_SHA256="${SOIR_V5_EMPTY_READER_EXPECTED_RAW_SHA256:-}"
COMPILER_SOURCE_SHA="${SOIR_V5_EMPTY_READER_COMPILER_SOURCE_SHA:-not_claimed}"
WRITER="self-hosted/ir/soir_writer.sio"
READER="self-hosted/ir/soir_reader.sio"
RUNTIME_TIMEOUT_SECONDS="${SOIR_V5_EMPTY_READER_RUNTIME_TIMEOUT_SECONDS:-15}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-soir-v5-empty-reader.XXXXXX")"

cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

fail() {
  printf 'SOIR_V5_EMPTY_READER_FAIL reason=%s\n' "$1" >&2
  exit 1
}

progress() {
  printf 'SOIR_V5_EMPTY_READER_PROGRESS stage=%s subject=%s status=%s\n' \
    "$1" "$2" "$3" >&2
}

run_compiler() {
  MADAROS_RAW_BIN="$RAW_COMPILER" "$COMPILER_CLI" "$@"
}

expected_paths() {
  printf '%s\n' \
    scripts/ci/soir_v5_empty_reader_gate.sh \
    self-hosted/ir/soir_reader.sio \
    self-hosted/ir/soir_writer.sio \
    tests/native-v2/soir_v5_empty_reader_reject_witness.sio \
    tests/native-v2/soir_v5_empty_reader_witness.sio
}

[[ -x "$COMPILER_CLI" ]] || fail compiler_cli_missing
[[ -x "$RAW_COMPILER" ]] || fail raw_compiler_missing
[[ "$(head -c 2 "$RAW_COMPILER" 2>/dev/null)" != '#!' ]] || fail raw_compiler_is_launcher
command -v timeout >/dev/null 2>&1 || fail timeout_command_missing
[[ "$RUNTIME_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail runtime_timeout_invalid
(( RUNTIME_TIMEOUT_SECONDS <= 60 )) || fail runtime_timeout_exceeds_cap_60

for path in "$WRITER" "$READER"; do
  [[ -f "$path" ]] || fail "missing_${path//\//_}"
done

grep -Fq 'pub let SOIR_READER_SCALAR_EMPTY_V5_SIZE: i64 = 320' "$READER" \
  || fail reader_size_contract_missing
grep -Fq 'pub fn soir_reader_decode_scalar_empty_module_v5(' "$READER" \
  || fail reader_decode_missing
grep -Fq 'pub let SOIR_READER_RECEIPT_WORDS: i64 = 8' "$READER" \
  || fail reader_receipt_contract_missing

dependency_pattern='use (parser|ir::ir|ir::serialize)|\b(IrModule|IrInstr|TyF128|TyF256|numeric_payload)\b'
set +e
if command -v rg >/dev/null 2>&1; then
  rg -n "$dependency_pattern" "$READER" >"$TMP/dependency-leak.log" 2>&1
else
  grep -En "$dependency_pattern" "$READER" >"$TMP/dependency-leak.log" 2>&1
fi
dependency_scan_rc=$?
set -e
if [[ "$dependency_scan_rc" -eq 0 ]]; then
  cat "$TMP/dependency-leak.log" >&2
  fail shadow_dependency_expanded
fi
[[ "$dependency_scan_rc" -eq 1 ]] || fail dependency_scan_error

for default_surface in \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/mod.sio \
  self-hosted/ir/lower.sio \
  self-hosted/check/check.sio \
  self-hosted/compiler/main.sio \
  self-hosted/compiler/module_frontend.sio; do
  [[ -f "$default_surface" ]] || continue
  if grep -Eq 'ir::soir_reader' "$default_surface"; then
    fail "shadow_imported_by_${default_surface//\//_}"
  fi
done

bash scripts/ci/madaros_f128_f256_format_identity_gate.sh --structural-only \
  >"$TMP/precision-identity.log" 2>&1 || {
    cat "$TMP/precision-identity.log" >&2
    fail precision_identity_regression
  }

head_sha="$(git rev-parse HEAD)"
tree_sha="$(git rev-parse 'HEAD^{tree}')"
mapfile -t evidence_paths < <(expected_paths)
for evidence_path in "${evidence_paths[@]}"; do
  git ls-files --error-unmatch -- "$evidence_path" >/dev/null 2>&1 \
    || fail evidence_path_untracked
done
dirty_tree="$(git status --porcelain --untracked-files=all)"
if [[ -n "$dirty_tree" ]]; then
  printf '%s\n' "$dirty_tree" >&2
  fail worktree_dirty
fi
if [[ "$COMPILER_SOURCE_SHA" != "not_claimed" && "$COMPILER_SOURCE_SHA" != "$head_sha" ]]; then
  fail compiler_source_sha_mismatch
fi

raw_compiler_sha256="$(sha256sum "$RAW_COMPILER" | awk '{print $1}')"
if [[ -n "$EXPECTED_RAW_SHA256" && "$raw_compiler_sha256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail raw_compiler_sha256_mismatch
fi
compiler_cli_sha256="$(sha256sum "$COMPILER_CLI" | awk '{print $1}')"

for source in "$WRITER" "$READER"; do
  name="$(basename "$source" .sio)"
  progress source_check "$name" begin
  run_compiler check "$source" >"$TMP/$name.check.log" 2>&1 || {
    cat "$TMP/$name.check.log" >&2
    fail "source_check_$name"
  }
  progress source_check "$name" pass
done

compose_and_run() {
  local witness="$1"
  local marker="$2"
  local name composite elf build_log run_log runtime_rc
  name="$(basename "$witness" .sio)"
  composite="$TMP/$name.sio"
  elf="$TMP/$name.elf"
  build_log="$TMP/$name.build.log"
  run_log="$TMP/$name.run.log"

  {
    printf 'module ir::%s_gate\n\n' "$name"
    sed '/^module ir::soir_writer$/d' "$WRITER"
    sed '/^module ir::soir_reader$/d' "$READER"
    sed -e '/^use ir::soir_writer::\*$/d' \
        -e '/^use ir::soir_reader::\*$/d' \
        "$witness"
  } >"$composite"

  progress composite_check "$name" begin
  run_compiler check "$composite" >"$build_log" 2>&1 || {
    cat "$build_log" >&2
    fail "composite_check_$name"
  }
  progress composite_build "$name" begin
  run_compiler --native-v2-compile "$composite" -o "$elf" >>"$build_log" 2>&1 || {
    cat "$build_log" >&2
    fail "composite_build_$name"
  }
  chmod +x "$elf"
  progress composite_runtime "$name" begin
  set +e
  timeout --signal=TERM --kill-after=2s "${RUNTIME_TIMEOUT_SECONDS}s" "$elf" >"$run_log" 2>&1
  runtime_rc=$?
  set -e
  if [[ "$runtime_rc" -eq 124 ]]; then
    cat "$run_log" >&2
    fail "composite_runtime_timeout_$name"
  fi
  if [[ "$runtime_rc" -ne 0 ]]; then
    cat "$run_log" >&2
    fail "composite_runtime_${name}_rc_${runtime_rc}"
  fi
  grep -Fxq "$marker" "$run_log" || {
    cat "$run_log" >&2
    fail "marker_missing_$name"
  }
  progress composite_runtime "$name" pass
}

compose_and_run \
  tests/native-v2/soir_v5_empty_reader_witness.sio \
  SOIR_V5_EMPTY_READER_CANONICAL_PASS
compose_and_run \
  tests/native-v2/soir_v5_empty_reader_reject_witness.sio \
  SOIR_V5_EMPTY_READER_REJECT_PASS
while IFS= read -r path; do
  sha256sum "$path"
done < <(expected_paths) >"$TMP/lane-content.sha256"
lane_content_sha256="$(sha256sum "$TMP/lane-content.sha256" | awk '{print $1}')"

printf '%s\n' 'SOIR_V5_EMPTY_READER_CHECK source=2/2 composite=2/2 precision_identity=preserved'
printf '%s\n' 'SOIR_V5_EMPTY_READER_WIRE exact_bytes=320 writer_length_bound=pass extension_count_bound=36 magic=checked version=5 reserved=zero-only truncated=reject trailing=reject unsupported_nonzero=reject request_cursor=local bounds_matrix=negative_start,negative_len,terminal_empty,insufficient_remaining,last_valid_frame repeat=pass offset_view=pass'
printf '%s\n' 'SOIR_V5_EMPTY_READER_RECEIPT storage=fixed_scalar_words semantic_words=8 physical_words=9 padding_words=1 decoded_canary_on_failure=preserved aggregate_return=none'
printf '%s\n' 'SOIR_V5_EMPTY_READER_BOUNDARY non_empty=not_claimed legacy_v1_v4=not_claimed opcode_decode=not_claimed arena_install=not_claimed issue_878=not_claimed default_pipeline=unchanged'
printf 'SOIR_V5_EMPTY_READER_PROVENANCE head=%s tree=%s worktree_clean=pass compiler_source_sha=%s\n' \
  "$head_sha" "$tree_sha" "$COMPILER_SOURCE_SHA"
printf 'SOIR_V5_EMPTY_READER_PASS compiler_cli=%s compiler_cli_sha256=%s raw_compiler=%s raw_compiler_sha256=%s head=%s tree=%s lane_content_sha256=%s\n' \
  "$COMPILER_CLI" "$compiler_cli_sha256" "$RAW_COMPILER" "$raw_compiler_sha256" "$head_sha" "$tree_sha" "$lane_content_sha256"
