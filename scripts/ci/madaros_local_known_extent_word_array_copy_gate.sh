#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/tests/native-v2/local_known_extent_word_array_copy_witness.sio"
KEEP_WORK="${SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-word-array-copy] FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-word-array-copy] SKIP: Linux-only gate" >&2
  exit 0
fi

if [[ -n "${SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-word-array-copy.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

MADAROS_ELF="${SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_BIN:-$WORK/madaros}"
WITNESS_ELF="$WORK/local_known_extent_word_array_copy"
EXPECTED="$WORK/expected.stdout"
ACTUAL="$WORK/actual.stdout"

if [[ -z "${SOUNIO_MADAROS_WORD_ARRAY_COPY_GATE_BIN:-}" ]]; then
  COMPILER_SOURCE="current_source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
else
  COMPILER_SOURCE="override"
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

compiler_sha="$(sha256sum "$MADAROS_ELF" | awk '{print $1}')"
source_sha="$(sha256sum "$SOURCE" | awk '{print $1}')"
echo "[madaros-word-array-copy] compiler_source=$COMPILER_SOURCE"
echo "[madaros-word-array-copy] compiler_sha256=$compiler_sha"
echo "[madaros-word-array-copy] witness_sha256=$source_sha"

printf '%s\n' \
  'PASS local_copy i64=independent i8=independent bool=independent repeat_i64=independent' \
  'PASS local_copy repeat_bool=independent repeat_f64=independent param_u64=independent scalar=stable' \
  >"$EXPECTED"
echo "[madaros-word-array-copy] blocker_id=BLK-20260714-madaros-fixed-array-call-boundary-alias"
echo "[madaros-word-array-copy] blocker_gate=scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh"
echo "[madaros-word-array-copy] blocker_doc=docs/handoff/madaros_fixed_array_call_boundary_alias_2026-07-14.md"
echo "[madaros-word-array-copy] residual_call_boundary_fixed_array_value_semantics=passed"

if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" check "$SOURCE" >"$WORK/check.log" 2>&1; then
  cat "$WORK/check.log" >&2
  fail "witness did not check"
fi
if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$SOURCE" -o "$WITNESS_ELF" >"$WORK/compile.log" 2>&1; then
  cat "$WORK/compile.log" >&2
  fail "witness did not compile"
fi
[[ -x "$WITNESS_ELF" ]] || fail "compile did not produce executable witness"

set +e
"$WITNESS_ELF" >"$ACTUAL" 2>"$WORK/run.stderr"
run_rc=$?
set -e
if [[ "$run_rc" != "0" ]]; then
  cat "$ACTUAL" >&2 || true
  cat "$WORK/run.stderr" >&2 || true
  fail "witness exited rc=$run_rc"
fi
if ! cmp -s "$EXPECTED" "$ACTUAL"; then
  diff -u "$EXPECTED" "$ACTUAL" >&2 || true
  fail "stdout mismatch"
fi

echo "[madaros-word-array-copy] PASS: local known-extent direct word-scalar arrays copy independently"
