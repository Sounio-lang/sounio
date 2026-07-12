#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVENT_NAME="${CI_EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
BASE_SHA="${CI_BASE_SHA:-}"
HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
SELECT_ONLY=0

if [[ "${1:-}" == "--select-only" ]]; then
  SELECT_ONLY=1
  shift
fi

paths=()
if (($#)); then
  paths=("$@")
elif [[ "$EVENT_NAME" == "pull_request" ]]; then
  [[ -n "$BASE_SHA" ]] || { echo 'error: CI_BASE_SHA is required for pull_request' >&2; exit 2; }
  mapfile -t paths < <(git -C "$ROOT_DIR" diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA")
else
  # Non-PR runs prove that the modular build and harness remain executable.
  paths=("tests/run-pass/generic_struct_return.sio")
fi

selected=()
for path in "${paths[@]}"; do
  case "$path" in tests/run-pass/*.sio) ;; *) continue ;; esac
  [[ -f "$ROOT_DIR/$path" ]] || continue
  grep -Fq '//@ requires: madaros' "$ROOT_DIR/$path" || continue
  selected+=("$path")
done

if [[ "$SELECT_ONLY" == "1" ]]; then
  printf '%s\n' "${selected[@]}"
  exit 0
fi

if ((${#selected[@]} == 0)); then
  echo 'MADAROS_CHANGED_TESTS_SKIP reason=no_changed_requires_madaros_tests'
  exit 0
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-changed.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
compiler="$work_dir/madaros"
test_list="$work_dir/tests.txt"
printf '%s\n' "${selected[@]}" > "$test_list"

echo "MADAROS_CHANGED_TESTS_START count=${#selected[@]}"
printf 'test=%s\n' "${selected[@]}"
bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$compiler"

SOUNIO_MADAROS_AVAILABLE=1 \
SOUNIO_TEST_SOUC_BIN="$compiler" \
  bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" --test-list "$test_list" --jobs "${SOUNIO_TEST_JOBS:-4}"

echo "MADAROS_CHANGED_TESTS_PASS count=${#selected[@]}"
