#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/tests/known_failures/madaros_fixed_array_call_boundary_alias_probe.sio"
KEEP_WORK="${SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-array-call-boundary] FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-array-call-boundary] SKIP: Linux-only gate" >&2
  exit 0
fi

if [[ -n "${SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-array-call-boundary.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

MADAROS_ELF="${SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_BIN:-$WORK/madaros}"
PROBE_ELF="$WORK/fixed_array_call_boundary_alias_probe"
PASS_EXPECTED="$WORK/pass.expected.stdout"
BLOCKED_EXPECTED="$WORK/blocked.expected.stdout"
ACTUAL="$WORK/actual.stdout"

if [[ -z "${SOUNIO_MADAROS_ARRAY_CALL_BOUNDARY_GATE_BIN:-}" ]]; then
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
echo "[madaros-array-call-boundary] compiler_source=$COMPILER_SOURCE"
echo "[madaros-array-call-boundary] compiler_sha256=$compiler_sha"
echo "[madaros-array-call-boundary] probe_sha256=$source_sha"
echo "[madaros-array-call-boundary] blocker_id=BLK-20260714-madaros-fixed-array-call-boundary-alias"

printf '%s\n' 'PASS fixed_array_call_boundary_value_semantics caller=unchanged' >"$PASS_EXPECTED"
printf '%s\n' 'BLOCKED fixed_array_call_boundary_alias caller_changed_after_by_value_param_mutation' >"$BLOCKED_EXPECTED"

if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" check "$SOURCE" >"$WORK/check.log" 2>&1; then
  cat "$WORK/check.log" >&2
  fail "probe did not check"
fi
if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$SOURCE" -o "$PROBE_ELF" >"$WORK/compile.log" 2>&1; then
  cat "$WORK/compile.log" >&2
  fail "probe did not compile"
fi
[[ -x "$PROBE_ELF" ]] || fail "compile did not produce executable probe"

set +e
"$PROBE_ELF" >"$ACTUAL" 2>"$WORK/run.stderr"
run_rc=$?
set -e

if [[ "$run_rc" == "0" ]] && cmp -s "$PASS_EXPECTED" "$ACTUAL"; then
  echo "[madaros-array-call-boundary] PASS: caller remains unchanged after by-value fixed-array parameter mutation"
  exit 0
fi

if [[ "$run_rc" == "61" ]] && cmp -s "$BLOCKED_EXPECTED" "$ACTUAL"; then
  cat "$ACTUAL" >&2
  echo "[madaros-array-call-boundary] BLOCKED: fixed-array call boundary aliases caller" >&2
  exit 61
fi

cat "$ACTUAL" >&2 || true
cat "$WORK/run.stderr" >&2 || true
echo "[madaros-array-call-boundary] observed_rc=$run_rc" >&2
fail "unexpected call-boundary result"
